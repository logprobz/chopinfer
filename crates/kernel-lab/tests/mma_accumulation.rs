//! BF16 mma.sync.m16n8k16 accumulation schemes against a compensated f64
//! reference: one MMA chain over all of K (as the fragment prefill runs whole
//! tiles), runs of 16 to 1024 K summed in F32 (128 is the FP8 block
//! promotion; 1024 is b12x's long-K flush), Kahan-compensated runs (16 is
//! b12x's prefill) and sequential F32 FMA. Receipts go to kind
//! `mma_accumulation`; the gemm README records the results.

#[path = "common/accumulation.rs"]
mod accumulation;

use accumulation::{
    Checkpoint, PROBE, Problem, SCHEMES, Scheme, Summary, audit_streams, bf16, columns, env_path,
    fma_emulation, raw_bf16, reference, spread, step_rows, summarize,
};
use kernel_core::codec::f32_to_bf16;
use kernel_core::contract::DType;
use kernel_lab::{LabResult, Session, gpu_test, launch};
use runtime_cuda::{KernelArgument, LaunchConfig};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::Path;

/// Outputs checked against the CPU FMA emulation per problem.
const FMA_SAMPLE: usize = 4096;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1_u64 << 53) as f64
    }

    fn sign(&mut self) -> f64 {
        if self.next() & 1 == 0 { 1.0 } else { -1.0 }
    }
}

fn to_bf16(value: f64) -> u16 {
    f32_to_bf16(value as f32)
}

/// One adversarial problem of `n` weight rows by `m` tokens at `k`.
fn adversarial(class: &str, n: usize, m: usize, k: usize) -> (Vec<u16>, Vec<u16>) {
    let mut rng = Rng(0x4143_4355_4d55_4c41 ^ ((k as u64) << 20) ^ (class.len() as u64));
    let mut weight = vec![0_u16; n * k];
    let mut activation = vec![0_u16; m * k];
    match class {
        // Random signs, significands in [1, 2) and 13 binades.
        "wide" => {
            for value in weight.iter_mut().chain(activation.iter_mut()) {
                let binade = (rng.next() % 13) as i32;
                *value = to_bf16(rng.sign() * (1.0 + rng.unit()) * 2_f64.powi(-binade));
            }
        }
        // Positive products over the first half of K, negated with a +2%
        // activation perturbation over the second: the sum climbs to S / 2 and
        // cancels to about 1% of S.
        "cancellation" => {
            let half = k / 2;
            for row in weight.chunks_exact_mut(k) {
                for i in 0..half {
                    let value = to_bf16(0.5 + rng.unit());
                    row[i] = value;
                    row[half + i] = value ^ 0x8000;
                }
            }
            for token in activation.chunks_exact_mut(k) {
                for i in 0..half {
                    let value = 0.5 + rng.unit();
                    token[i] = to_bf16(value);
                    token[half + i] = to_bf16(bf16(token[i]) * 1.02);
                }
            }
        }
        "positive" => {
            for value in weight.iter_mut().chain(activation.iter_mut()) {
                *value = to_bf16(1.0 - rng.unit());
            }
        }
        // Random signs of order one, with the first K / 256 products (at least
        // one) 2^12 larger.
        "large_first" => {
            let large = (k / 256).max(1);
            for (data, len) in [(&mut weight, n), (&mut activation, m)] {
                for row in data.chunks_exact_mut(k).take(len) {
                    for (i, value) in row.iter_mut().enumerate() {
                        let scale = if i < large { 64.0 } else { 1.0 };
                        *value = to_bf16(rng.sign() * (0.5 + rng.unit()) * scale);
                    }
                }
            }
        }
        _ => unreachable!("adversarial class"),
    }
    (weight, activation)
}

/// One weight matrix with its activations: rows [n][k], tokens [m][k].
struct Group {
    weight: Vec<u16>,
    n: usize,
    activation: Vec<u16>,
    m: usize,
    k: usize,
}

impl Group {
    fn new(
        weight: Vec<u16>,
        n: usize,
        activation: Vec<u16>,
        m: usize,
        k: usize,
    ) -> LabResult<Self> {
        let n = n / 16 * 16;
        let m = m / 32 * 32;
        if n == 0 || m == 0 || weight.len() < n * k || activation.len() < m * k {
            return Err(format!("group too small: n={n} m={m} k={k}"));
        }
        Ok(Self {
            weight: weight[..n * k].to_vec(),
            n,
            activation: activation[..m * k].to_vec(),
            m,
            k,
        })
    }
}

struct RealClass {
    name: &'static str,
    groups: Vec<Group>,
    /// Native K and served shard widths of a K-sharded weight.
    native: usize,
    ranks: Vec<usize>,
}

impl RealClass {
    fn sweep(&self) -> Vec<usize> {
        let top = self.native.min(16384);
        let mut ks = (4..=14)
            .map(|p| 1_usize << p)
            .filter(|&k| k <= top)
            .collect::<Vec<_>>();
        ks.push(top);
        ks.extend(self.ranks.iter().copied().filter(|&k| k <= top));
        ks.sort_unstable();
        ks.dedup();
        ks
    }
}

/// Rows of `data` [rows][cols] at `indices`.
fn pick(data: &[u16], cols: usize, indices: &[usize]) -> Vec<u16> {
    indices
        .iter()
        .flat_map(|&i| data[i * cols..][..cols].iter().copied())
        .collect()
}

fn qwen_classes(checkpoint: &Path, dumps: &Path) -> LabResult<Vec<RealClass>> {
    let checkpoint = Checkpoint::open(checkpoint)?;
    // The reference runner's BF16 taps of long_4k, one row per token.
    let taps = |l: usize, site: &str| step_rows(&dumps.join("long_4k"), &format!("L{l:02}.{site}"));
    let name = |l: usize, tail: &str| format!("model.language_model.layers.{l}.{tail}");
    let tokens = spread(4096, 1024);
    let mut hc_down = Vec::new();
    for l in [0, 1] {
        for side in ["attn", "mlp"] {
            let (shape, weight) = checkpoint.bf16(&name(
                l,
                &format!("{side}_hyper_connection.input_mix_weight_down.weight"),
            ))?;
            let (_, rows) = taps(l, &format!("x.hc.{side}.xn"))?;
            hc_down.push(Group::new(
                weight,
                shape[0],
                pick(&rows, shape[1], &tokens),
                tokens.len(),
                shape[1],
            )?);
        }
    }
    let mut in_qkv = Vec::new();
    let mut out = Vec::new();
    for l in [0, 1] {
        let weight_name = name(l, "linear_attn.in_proj_qkv.weight");
        let shape = checkpoint.shape(&weight_name)?;
        let (_, all) = checkpoint.bf16(&weight_name)?;
        let rows = spread(shape[0], 256);
        let (_, mixed) = taps(l, "x.hc.attn")?;
        in_qkv.push(Group::new(
            pick(&all, shape[1], &rows),
            rows.len(),
            pick(&mixed, shape[1], &tokens),
            tokens.len(),
            shape[1],
        )?);
        let weight_name = name(l, "linear_attn.out_proj.weight");
        let shape = checkpoint.shape(&weight_name)?;
        let (_, all) = checkpoint.bf16(&weight_name)?;
        let rows = spread(shape[0], 256);
        // attn.core is the out_proj input, one row of all heads per token.
        let (width, normed) = taps(l, "attn.core")?;
        out.push(Group::new(
            pick(&all, shape[1], &rows),
            rows.len(),
            pick(&normed, width, &tokens),
            tokens.len(),
            width,
        )?);
    }
    Ok(vec![
        RealClass {
            name: "qwen_hc_down",
            groups: hc_down,
            native: 10240,
            ranks: vec![],
        },
        RealClass {
            name: "qwen_in_proj_qkv",
            groups: in_qkv,
            native: 2560,
            ranks: vec![],
        },
        RealClass {
            name: "qwen_out_proj",
            groups: out,
            native: 6144,
            ranks: vec![3072, 1536, 768],
        },
    ])
}

/// Rows of one audit file over every stream, concatenated.
fn audit(dir: &Path, file: &str, width: usize) -> LabResult<Vec<u16>> {
    let mut rows = Vec::new();
    for stream in audit_streams(dir)? {
        rows.extend(raw_bf16(&dir.join(&stream).join(file), width)?);
    }
    Ok(rows)
}

fn mimo_classes(checkpoint: &Path, audit_dir: &Path) -> LabResult<Vec<RealClass>> {
    const HIDDEN: usize = 4096;
    let model = Checkpoint::open(checkpoint)?;
    let norm = |name: &str| model.bf16(name).map(|(_, w)| w);
    let normed = |rows: &[u16], weight: &[u16]| {
        accumulation::NORM
            .apply_bf16(rows, weight, HIDDEN)
            .map_err(|e| e.0.to_string())
    };
    let (mut o, mut router) = (Vec::new(), Vec::new());
    for l in 1..=6 {
        let (shape, all) = model.bf16(&format!("model.layers.{l}.self_attn.o_proj.weight"))?;
        let rows = spread(shape[0], 256);
        let attention = audit(audit_dir, &format!("layer{l:02}.attn.bf16"), shape[1])?;
        let m = attention.len() / shape[1];
        o.push(Group::new(
            pick(&all, shape[1], &rows),
            rows.len(),
            attention,
            m,
            shape[1],
        )?);
        let (shape, gate) = model.bf16(&format!("model.layers.{l}.mlp.gate.weight"))?;
        let residual = audit(audit_dir, &format!("layer{l:02}.resid.bf16"), HIDDEN)?;
        let input = normed(
            &residual,
            &norm(&format!("model.layers.{l}.post_attention_layernorm.weight"))?,
        )?;
        let m = input.len() / HIDDEN;
        router.push(Group::new(gate, shape[0], input, m, HIDDEN)?);
    }
    let last = audit(audit_dir, "layer47.bf16", HIDDEN)?;
    let final_rows = normed(&last, &norm("model.norm.weight")?)?;
    let head_shape = model.shape("lm_head.weight")?;
    let head_rows = spread(head_shape[0], 2048);
    let mut head_weight = Vec::with_capacity(head_rows.len() * HIDDEN);
    for &row in &head_rows {
        head_weight.extend(model.bf16_rows("lm_head.weight", row, 1)?);
    }
    let m = final_rows.len() / HIDDEN;
    let head = Group::new(head_weight, head_rows.len(), final_rows.clone(), m, HIDDEN)?;
    // The drafter's fc reads the fused all-reduce's normalized output after
    // target layers 0, 11, 23 and 35 (next layer's input norm) and the final
    // norm after layer 47, concatenated per token.
    let mut taps = Vec::new();
    for l in [0, 11, 23, 35] {
        let hidden = audit(audit_dir, &format!("layer{l:02}.bf16"), HIDDEN)?;
        taps.push(normed(
            &hidden,
            &norm(&format!("model.layers.{}.input_layernorm.weight", l + 1))?,
        )?);
    }
    taps.push(final_rows);
    let tokens = taps[0].len() / HIDDEN;
    let context = (0..tokens)
        .flat_map(|t| {
            taps.iter()
                .flat_map(move |tap| tap[t * HIDDEN..][..HIDDEN].iter().copied())
        })
        .collect::<Vec<_>>();
    let drafter = Checkpoint::open(&checkpoint.join("dflash"))?;
    let (fc_shape, fc_all) = drafter.bf16("fc.weight")?;
    let fc_rows = spread(fc_shape[0], 1024);
    let fc = Group::new(
        pick(&fc_all, fc_shape[1], &fc_rows),
        fc_rows.len(),
        context,
        tokens,
        fc_shape[1],
    )?;
    Ok(vec![
        RealClass {
            name: "mimo_o_proj",
            groups: o,
            native: 8192,
            ranks: vec![4096, 2048],
        },
        RealClass {
            name: "mimo_router",
            groups: router,
            native: HIDDEN,
            ranks: vec![],
        },
        RealClass {
            name: "mimo_head",
            groups: vec![head],
            native: HIDDEN,
            ranks: vec![],
        },
        RealClass {
            name: "mimo_dflash_fc",
            groups: vec![fc],
            native: 20480,
            ranks: vec![10240, 5120],
        },
    ])
}

/// A probe problem: weights, rows n, activations, tokens m, K.
type ProblemData<'a> = (&'a [u16], usize, &'a [u16], usize, usize);

/// Outputs, references and FMA checks of every scheme over a set of problems.
struct Collected {
    actual: BTreeMap<String, Vec<f32>>,
    values: Vec<f64>,
    sums: Vec<f64>,
    fma_mismatches: usize,
    fma_checked: usize,
}

fn run_problems(session: &Session, problems: &[ProblemData<'_>]) -> LabResult<(Collected, usize)> {
    let module = session.module(PROBE)?;
    let mut collected = Collected {
        actual: BTreeMap::new(),
        values: Vec::new(),
        sums: Vec::new(),
        fma_mismatches: 0,
        fma_checked: 0,
    };
    let mut k_all = 0;
    for &(weight, n, activation, m, k) in problems {
        k_all = k;
        let exact = reference(weight, activation, n, m, k);
        let problem = Problem::upload(session, weight, activation, n, m, k)?;
        for scheme in SCHEMES {
            let outputs = problem.run(session, &module, scheme)?;
            if scheme == Scheme::Fma {
                for index in accumulation::spread(outputs.len(), FMA_SAMPLE) {
                    let expected = fma_emulation(weight, activation, n, k, index);
                    collected.fma_mismatches +=
                        usize::from(expected.to_bits() != outputs[index].to_bits());
                    collected.fma_checked += 1;
                }
            }
            collected
                .actual
                .entry(scheme.label())
                .or_default()
                .extend(outputs);
        }
        collected.values.extend(exact.values);
        collected.sums.extend(exact.sum_abs_products);
    }
    Ok((collected, k_all))
}

/// Summaries per scheme of the collected problems (one K), a printed table row
/// each, and contract findings.
fn report(
    class: &str,
    k: usize,
    collected: Collected,
    findings: &mut Vec<String>,
) -> LabResult<Value> {
    let oracle = kernel_core::oracle::OracleOutput::new(collected.values, collected.sums, k)
        .map_err(|e| e.0.to_string())?;
    if collected.fma_mismatches != 0 {
        findings.push(format!(
            "{class} K={k}: FMA scheme differs from the CPU emulation at {} of {} outputs",
            collected.fma_mismatches, collected.fma_checked
        ));
    }
    let mut schemes = serde_json::Map::new();
    for scheme in SCHEMES {
        let label = scheme.label();
        let summary: Summary = summarize(&collected.actual[&label], &oracle, DType::F32);
        println!(
            "{class:18} {k:6} {label:9} n={:8} ulp max {:10.3e} p99.99 {:10.3e}  bound max {:9.3e} p99.99 {:9.3e}  bias {:+9.2e}  bf16 misround {:.2e}",
            summary.outputs,
            summary.ulps.max,
            summary.ulps.p9999,
            summary.accumulation.max,
            summary.accumulation.p9999,
            summary.bias,
            summary.misrounded,
        );
        if summary.violations > 0 {
            findings.push(format!(
                "{class} K={k} {label}: {} outputs over 2^-22*K*S (max ratio {:.3})",
                summary.violations, summary.accumulation.max
            ));
        }
        schemes.insert(label, summary.json());
    }
    Ok(
        json!({"k": k, "fma_checked": collected.fma_checked, "fma_mismatches": collected.fma_mismatches, "schemes": schemes}),
    )
}

fn finish(session: &Session, case: &str, results: Value, findings: Vec<String>) -> LabResult<()> {
    let mut receipt = BTreeMap::new();
    receipt.insert("case".into(), json!(case));
    receipt.insert("results".into(), results);
    receipt.insert("findings".into(), json!(findings));
    let path = session.receipt("mma_accumulation", receipt)?;
    eprintln!("receipt: {}", path.display());
    if findings.is_empty() {
        Ok(())
    } else {
        Err(findings.join("\n"))
    }
}

#[test]
fn mma_accumulation_adversarial() {
    gpu_test(8 << 30, |session| {
        const N: usize = 256;
        const M: usize = 1024;
        let mut findings = Vec::new();
        let mut results = serde_json::Map::new();
        for class in ["wide", "cancellation", "positive", "large_first"] {
            let mut rows = Vec::new();
            for k in (4..=14).map(|p| 1_usize << p) {
                let (weight, activation) = adversarial(class, N, M, k);
                let (collected, k) = run_problems(
                    session,
                    &[(weight.as_slice(), N, activation.as_slice(), M, k)],
                )?;
                rows.push(report(class, k, collected, &mut findings)?);
            }
            results.insert(class.into(), json!(rows));
        }
        finish(session, "adversarial", Value::Object(results), findings)
    })
    .expect("BF16 MMA accumulation on adversarial inputs");
}

#[test]
fn mma_accumulation_real() {
    gpu_test(16 << 30, |session| {
        let mut classes = Vec::new();
        match (env_path("CHOPINFER_QWEN38_CHECKPOINT"), env_path("CHOPINFER_QWEN38_LAYER_DUMPS")) {
            (Some(checkpoint), Some(dumps)) => classes.extend(qwen_classes(&checkpoint, &dumps)?),
            _ => kernel_lab::skip(
                "Qwen real classes: set CHOPINFER_QWEN38_CHECKPOINT and CHOPINFER_QWEN38_LAYER_DUMPS",
            )?,
        }
        match (env_path("CHOPINFER_MIMO_CHECKPOINT"), env_path("CHOPINFER_MIMO_AUDIT_DIR")) {
            (Some(checkpoint), Some(audit)) => classes.extend(mimo_classes(&checkpoint, &audit)?),
            _ => kernel_lab::skip(
                "MiMo real classes: set CHOPINFER_MIMO_CHECKPOINT and CHOPINFER_MIMO_AUDIT_DIR",
            )?,
        }
        if classes.is_empty() {
            return Ok(());
        }
        let mut findings = Vec::new();
        let mut results = serde_json::Map::new();
        for class in &classes {
            let mut rows = Vec::new();
            for k in class.sweep() {
                let sliced = class
                    .groups
                    .iter()
                    .map(|g| (columns(&g.weight, g.k, 0, k), g.n, columns(&g.activation, g.k, 0, k), g.m))
                    .collect::<Vec<_>>();
                let problems = sliced
                    .iter()
                    .map(|(w, n, a, m)| (w.as_slice(), *n, a.as_slice(), *m, k))
                    .collect::<Vec<_>>();
                let (collected, _) = run_problems(session, &problems)?;
                rows.push(report(class.name, k, collected, &mut findings)?);
            }
            results.insert(class.name.into(), json!(rows));
        }
        finish(session, "real", Value::Object(results), findings)
    })
    .expect("BF16 MMA accumulation on real weights and activations");
}

/// One crafted m16n8k16: C[0][0] plus the products a_k * b_k placed at K
/// positions of row 0 and token 0; every other element is zero.
struct Crafted {
    name: String,
    c: f64,
    products: Vec<(usize, f64, f64)>,
}

fn crafted_cases() -> Vec<Crafted> {
    let p = |e: i32| 2_f64.powi(e);
    let case = |name: &str, c: f64, products: Vec<(usize, f64, f64)>| Crafted {
        name: name.into(),
        c,
        products,
    };
    let mut cases = vec![
        // Round to nearest gives 1 + 2^-23 and 1 - 2^-24; truncation toward
        // zero gives 1 and 1 - 2^-23.
        case("c=1 plus 1.5*2^-24", 1.0, vec![(0, 1.5 * p(-12), p(-12))]),
        case(
            "c=1 minus 1.25*2^-24",
            1.0,
            vec![(0, -1.25 * p(-12), p(-12))],
        ),
    ];
    // Sixteen equal addends 2^-e below an accumulator (or a product) of 1: an
    // exact sum keeps 16 * 2^-e.
    for e in 24..=30 {
        let small = |k| (k, p(-e / 2), p(-(e - e / 2)));
        cases.push(case(
            &format!("c=1 plus 16 x 2^-{e}"),
            1.0,
            (0..16).map(small).collect(),
        ));
        let mut products = vec![(0, 1.0, 1.0)];
        products.extend((1..16).map(small));
        cases.push(case(&format!("1 plus 15 x 2^-{e}, c=0"), 0.0, products));
    }
    // 2^24 + 1 - 2^24 = 1 exactly: which positions share a rounding decides
    // whether 1 survives.
    for (name, positions) in [
        ("+2^24@0 +1@1 -2^24@8", [(0, 1.0), (1, 0.0), (8, -1.0)]),
        ("+2^24@0 -2^24@1 +1@8", [(0, 1.0), (8, 0.0), (1, -1.0)]),
        ("+2^24@0 -2^24@1 +1@4", [(0, 1.0), (4, 0.0), (1, -1.0)]),
        ("+2^24@0 +1@1 -2^24@4", [(0, 1.0), (1, 0.0), (4, -1.0)]),
        ("+2^24@0 +1@1 -2^24@2", [(0, 1.0), (1, 0.0), (2, -1.0)]),
        ("+2^24@0 +1@15 -2^24@14", [(0, 1.0), (15, 0.0), (14, -1.0)]),
    ] {
        // A zero sign marks the +1 product.
        let products = positions
            .iter()
            .map(|&(k, sign)| {
                if sign == 0.0 {
                    (k, 1.0, 1.0)
                } else {
                    (k, sign * p(12), p(12))
                }
            })
            .collect();
        cases.push(case(name, 0.0, products));
    }
    cases.push(case(
        "c=+2^24 +1@0 -2^24@1",
        p(24),
        vec![(0, 1.0, 1.0), (1, -p(12), p(12))],
    ));
    cases.push(case(
        "c=+2^24 -2^24@0 +1@1",
        p(24),
        vec![(0, -p(12), p(12)), (1, 1.0, 1.0)],
    ));
    // Addends one and two bits below 2^0 next to +-2^24: which survive shows
    // the alignment window, and whether cancelling the pair first (in an
    // earlier position group) changes that shows how positions are grouped.
    for small in [-1, -2] {
        let value = p(small);
        for (name, products) in [
            (
                "+2^24@0 +s@1 -2^24@8",
                vec![(0, p(12), p(12)), (1, value, 1.0), (8, -p(12), p(12))],
            ),
            (
                "+2^24@0 -2^24@1 +s@8",
                vec![(0, p(12), p(12)), (1, -p(12), p(12)), (8, value, 1.0)],
            ),
            (
                "+2^24@0 -2^24@1 +s@4",
                vec![(0, p(12), p(12)), (1, -p(12), p(12)), (4, value, 1.0)],
            ),
            (
                "+2^24@0 -2^24@1 +s@2",
                vec![(0, p(12), p(12)), (1, -p(12), p(12)), (2, value, 1.0)],
            ),
        ] {
            cases.push(case(&format!("{} s=2^{small}", name), 0.0, products));
        }
        cases.push(case(
            &format!("c=+2^24 -2^24@0 +s@1 s=2^{small}"),
            p(24),
            vec![(0, -p(12), p(12)), (1, value, 1.0)],
        ));
    }
    cases
}

#[test]
fn mma_accumulation_mechanism() {
    gpu_test(1 << 30, |session| {
        let cases = crafted_cases();
        let count = cases.len();
        let (mut a, mut b, mut c) = (vec![0_u16; count * 256], vec![0_u16; count * 128], vec![0_f32; count * 128]);
        for (i, case) in cases.iter().enumerate() {
            for &(k, x, y) in &case.products {
                let (x_bits, y_bits) = (to_bf16(x), to_bf16(y));
                if bf16(x_bits) != x || bf16(y_bits) != y {
                    return Err(format!("{}: an operand is not a BF16 value", case.name));
                }
                a[i * 256 + k] = x_bits;
                b[i * 128 + k] = y_bits;
            }
            c[i * 128] = case.c as f32;
        }
        let (a, b, c) = (session.upload_bf16(&a)?, session.upload_bf16(&b)?, session.upload_f32(&c)?);
        let d = session.allocate(count * 128 * 4)?;
        session
            .module(PROBE)?
            .function("chop_probe_mma_once_v1")
            .map_err(|e| e.to_string())?
            .launch(
                &session.stream,
                launch::simple_config((count as u32, 1, 1), (32, 1, 1), 0, LaunchConfig::NO_CLUSTER, false)?,
                &[
                    KernelArgument::Buffer(&a),
                    KernelArgument::Buffer(&b),
                    KernelArgument::Buffer(&c),
                    KernelArgument::Buffer(&d),
                ],
            )
            .map_err(|e| e.to_string())?;
        let d = session.download_f32(&d, count * 128)?;
        let mut rows = Vec::new();
        for (i, case) in cases.iter().enumerate() {
            let exact = case.c + case.products.iter().map(|&(_, x, y)| x * y).sum::<f64>();
            let value = d[i * 128];
            println!("{:28} exact {:+.10e}  mma {:+.10e} ({:#010x})", case.name, exact, value, value.to_bits());
            rows.push(json!({"case": case.name, "exact": exact, "mma": value, "bits": format!("{:#010x}", value.to_bits())}));
        }
        finish(session, "mechanism", json!(rows), Vec::new())
    })
    .expect("BF16 MMA mechanism cases");
}
