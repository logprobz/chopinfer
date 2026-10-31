//! Synthetic wide HC boundaries: the column-block producer and split
//! finish that `wide_verify` serves at 65 to 192 rows (`cols_r4tx_w32`)
//! against the ordered-chain pair at 65, 128, 129, 192 and 256 rows, both
//! over the wide boundary's entries (`sm120::hc_wide`, capacity 256).
//!
//! Every row, coefficient and weight is synthesized from adversarial
//! patterns (cancellation, wide dynamic range, subnormals, signed zeros,
//! BF16 rounding ties, random bits, NaN and infinities), so this is
//! structural coverage of the upper rows: grids, 64-bit row indexing, live
//! rows, untouched padding rows, guards, status priors, graph replay and
//! mutation controls at widths the captured corpus does not hold. It
//! complements the real-data end-to-end replays of wide verify steps at 65
//! to 192 rows and does not replace them; the corpus tests (`replay`) keep
//! real rows only.
use super::device::{self, Candidate, Input, Kernels, Output};
use super::{artifacts, replay::compare};
use bundle_deepseek41::sm120::{self as b, hc_wide};
use kernel_core::{
    backend::{Backend, Launch, Prepared, View},
    codec::f32_to_bf16,
    sinkhorn_expand_mix::SinkhornRoundNormBuffers,
};
use kernel_harness::{LabResult, Session};
use kernels_sm120::sinkhorn_hc::{finish_split, round_cols};
use model_deepseek41_contract::{geometry::FLASH, rows as contract_rows};
use runtime_cuda::Stream;
use serde_json::{Value, json};
use std::collections::BTreeMap;

type Values = BTreeMap<String, Vec<u8>>;
type Op = Prepared<SinkhornRoundNormBuffers<View>>;

/// The first row above the decode route, both sides of the 128-row
/// bucket, the package's widest verify bucket and the wide entries'
/// capacity.
pub const ROWS: [u32; 5] = [65, 128, 129, 192, 256];
const CAPACITY: u32 = 256;
pub const PATTERNS: [&str; 8] = [
    "finite",
    "cancellation",
    "dynamic_range",
    "subnormal",
    "signed_zero",
    "bf16_ties",
    "random_bits",
    "nonfinite",
];
/// (widen, alias): both widening policies, with post and comb aliased to
/// their inputs (as served) or in separate buffers.
const CONFIGURATIONS: [(bool, bool); 4] =
    [(true, true), (false, true), (true, false), (false, false)];
/// Input mutations that must change the boundary's outputs.
const MUTATIONS: [&str; 2] = ["last_row_sum_sign", "live_short"];
const H: usize = 5120;
const N: usize = 4;
const MIXES: usize = 24;
/// Each input role: elements (per row, or in all for weights), whether it
/// is per row, and whether it holds BF16.
const ROLES: [(&str, usize, bool, bool); 9] = [
    ("streams", N * H, true, true),
    ("sum", H, true, false),
    ("post_in", N, true, false),
    ("comb_in", N * N, true, false),
    ("pre_in", N, true, false),
    ("fn", MIXES * N * H, false, false),
    ("base", MIXES, false, false),
    ("scale", 3, false, false),
    ("norm_weight", H, false, true),
];

/// The served wide column pair at the tile, butterfly and warps of
/// `hc.finish_cols.verify` at 65 to 192 rows.
pub fn candidate() -> Candidate {
    let (producer, finish) = hc_wide::column_pair(&FLASH, CAPACITY);
    Candidate::Cols {
        producer,
        finish,
        tile: 4,
        transposed: true,
        warps: 32,
    }
}

/// The column pair's mutation controls (`device::controls`) over the wide
/// entries.
pub fn controls() -> Vec<Candidate> {
    let (producer, finish) = hc_wide::column_pair(&FLASH, CAPACITY);
    device::controls()
        .into_iter()
        .map(|control| match control {
            Candidate::Cols {
                producer: p,
                finish: f,
                tile,
                transposed,
                warps,
            } => Candidate::Cols {
                producer: round_cols::Entry {
                    reference: producer.reference.clone(),
                    ..p
                },
                finish: finish_split::Entry {
                    reference: finish.reference.clone(),
                    ..f
                },
                tile,
                transposed,
                warps,
            },
            twin => twin,
        })
        .collect()
}

/// The status prior under which a control shows: clearing the status
/// shows only over a sticky prior.
fn prior_of(c: &Candidate) -> u32 {
    match c {
        Candidate::Cols { finish, .. }
            if finish.mutation == finish_split::Mutation::ClearStatus =>
        {
            0x80
        }
        _ => 0,
    }
}

/// The ordered-chain pair over the wide entries at every test row: the
/// chain producer on four warps (`hc.finish_chain.verify` at 32 to 64
/// rows), then the 1024-thread finish/norm tail. An arithmetic control,
/// not a latency baseline.
pub fn reference() -> LabResult<Op> {
    let (producer, _) = hc_wide::column_pair(&FLASH, CAPACITY);
    hc_wide::pair_backend(&FLASH, CAPACITY, ROWS.to_vec())
        .with_sinkhorn_finish_norm_chain(
            producer.reference,
            ROWS.iter().map(|&rows| (rows, 4)).collect(),
        )
        .sinkhorn_round_expand_mix_norm(&device::query(), device::need_rows(CAPACITY))
        .map_err(|e| format!("{e:?}"))
}

/// The bundle's embedded modules of the chain producer, the finish/norm
/// tail and the column pair, each with the exports of its lock record.
fn kernels(s: &Session) -> LabResult<Kernels> {
    let (producer, finish) = hc_wide::column_pair(&FLASH, CAPACITY);
    let cols = include_str!("../../../../bundle/kernels.lock.d/sinkhorn_hc_cols.lock");
    let mut images = BTreeMap::new();
    for (symbol, lock) in [
        (
            producer.reference.symbols()[0].clone(),
            include_str!("../../../../bundle/kernels.lock.d/sinkhorn_hc_round_chain.lock"),
        ),
        (
            finish.reference.symbols()[1].clone(),
            include_str!("../../../../bundle/kernels.lock.d/sinkhorn_hc_finish_norm.lock"),
        ),
        (producer.symbol(4, true), cols),
        (finish.symbol(32), cols),
    ] {
        let (name, bytes) = b::fold_cubin_for_symbol(&symbol).ok_or("bundle cubin missing")?;
        images.insert(
            name,
            (bytes.to_vec(), artifacts::exports(lock, name, bytes)?),
        );
    }
    Kernels::from_images(s, images.into_values())
}

/// SplitMix64.
struct Bits(u64);
impl Bits {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1_u64 << 23) as f32 - 1.0
    }
}

/// An element: a value in the role's type, or its raw bits.
enum Element {
    Float(f32),
    Raw(u32),
}
use Element::{Float as F, Raw};

fn finite(role: &str, j: usize, b: &mut Bits) -> f32 {
    match role {
        "scale" => [0.5, 0.25, 1.0][j],
        "norm_weight" => 1.0 + 0.5 * b.unit(),
        "streams" => 4.0 * b.unit(),
        "sum" => 2.0 * b.unit(),
        "comb_in" => 0.5 * b.unit(),
        "fn" => 0.02 * b.unit(),
        _ => b.unit(),
    }
}

/// Element `j` of row `row` (of `rows`) of `role` under `pattern`.
fn element(pattern: &str, role: &str, row: usize, rows: usize, j: usize, b: &mut Bits) -> Element {
    let base = finite(role, j, b);
    let sign = |b: &mut Bits| (b.next() & 1) as u32;
    match (pattern, role) {
        // Pairs of large opposite values (2^8 to 2^16, shared by each
        // pair): chain and RMS sums cancel to the small terms, so their
        // order shows.
        ("cancellation", "streams" | "sum") => {
            let large = 2_f32.powi(8 + ((row * 31 + j / 2) % 9) as i32);
            F(if j % 2 == 0 { large } else { -large } + base)
        }
        ("cancellation", "fn") => F(0.01 + 0.005 * base),
        ("dynamic_range", "streams" | "sum") => F(base * 2_f32.powi((b.next() % 81) as i32 - 40)),
        ("dynamic_range", "fn") => F(base * 2_f32.powi((j % 49) as i32 - 30)),
        ("dynamic_range", "post_in" | "comb_in" | "pre_in") => {
            F(base * 2_f32.powi((b.next() % 17) as i32 - 8))
        }
        // Even rows wholly subnormal, odd rows mixed with small normals.
        ("subnormal", "sum") if row % 2 == 0 || j % 2 == 0 => {
            Raw(sign(b) << 31 | (b.next() as u32 & 0x7f_ffff))
        }
        ("subnormal", "sum") => F(1.0e-30 * base),
        ("subnormal", "streams") if row % 2 == 0 || j % 4 != 3 => {
            Raw(sign(b) << 15 | (b.next() as u32 & 0x7f))
        }
        ("subnormal", "fn") if j % 2 == 0 => Raw(sign(b) << 31 | (b.next() as u32 & 0x7f_ffff)),
        ("signed_zero", "streams") if j % 3 != 2 => F(if j % 3 == 0 { 0.0 } else { -0.0 }),
        ("signed_zero", "sum") if row % 2 == 0 || j % 2 == 0 => F(-0.0),
        ("signed_zero", "fn") if j % 5 == 0 => F(-0.0),
        ("signed_zero", "post_in" | "pre_in") if row % 7 == 0 => F(-0.0),
        ("signed_zero", "comb_in") if j % 3 == 0 => F(-0.0),
        // Exact BF16 ties (round to even both ways) and their neighbours.
        ("bf16_ties", "sum") => {
            let high = (0x3800_0000 + (b.next() as u32 & 0x0fff_0000)) & 0x7fff_0000;
            let low = match j % 4 {
                0 | 1 => 0x8000,
                2 => 0x7fff,
                _ => 0x8001,
            };
            Raw(sign(b) << 31 | high | low)
        }
        ("random_bits", "streams" | "norm_weight") => Raw(b.next() as u32 & 0xffff),
        ("random_bits", "sum" | "post_in" | "comb_in" | "pre_in") => Raw(b.next() as u32),
        ("nonfinite", "sum") if row == 0 && j == 0 => Raw(0x7fc0_0001),
        ("nonfinite", "sum") if row == 1 && j == H - 1 => F(f32::INFINITY),
        ("nonfinite", "sum") if row == rows / 2 && j == 17 => F(f32::NEG_INFINITY),
        ("nonfinite", "streams") if row == 2 && j == 3 => Raw(0xffc1),
        ("nonfinite", "streams") if row == rows - 1 && j == N * H - 1 => Raw(0x7f80),
        ("nonfinite", "post_in") if row == 3 && j == 1 => F(f32::NAN),
        ("nonfinite", "comb_in") if row == 4 && j == 5 => F(f32::NEG_INFINITY),
        ("nonfinite", "pre_in") if row == 5 && j == 0 => F(f32::NAN),
        ("nonfinite", "norm_weight") if j == H - 1 => Raw(0x7fc0),
        _ => F(base),
    }
}

/// Every input role's bytes for `rows` rows of `pattern`.
pub fn operands(rows: usize, pattern: &str) -> BTreeMap<&'static str, Vec<u8>> {
    ROLES
        .iter()
        .map(|&(role, elements, per_row, bf16)| {
            let seed = format!("{pattern}:{role}")
                .bytes()
                .fold(0_u64, |h, c| h.wrapping_mul(131).wrapping_add(u64::from(c)));
            let mut b = Bits(seed);
            let count = if per_row { rows } else { 1 };
            let mut bytes = Vec::with_capacity(count * elements * if bf16 { 2 } else { 4 });
            for row in 0..count {
                for j in 0..elements {
                    match (element(pattern, role, row, rows, j, &mut b), bf16) {
                        (F(v), true) => bytes.extend(f32_to_bf16(v).to_le_bytes()),
                        (F(v), false) => bytes.extend(v.to_le_bytes()),
                        (Raw(bits), true) => bytes.extend((bits as u16).to_le_bytes()),
                        (Raw(bits), false) => bytes.extend(bits.to_le_bytes()),
                    }
                }
            }
            (role, bytes)
        })
        .collect()
}

fn input(s: &Session, rows: usize, pattern: &str) -> LabResult<Input> {
    let mut data = BTreeMap::new();
    for (role, bytes) in operands(rows, pattern) {
        let buffer = s.allocate(bytes.len())?;
        buffer
            .copy_from_bytes(&bytes, &s.stream)
            .map_err(|e| e.to_string())?;
        data.insert(role.to_owned(), buffer);
    }
    let input = Input {
        data,
        live: s.allocate(4)?,
        rows,
        eps: contract_rows::norm_quantizers(&FLASH)[0].norm.epsilon_bits,
    };
    set_live(&input, &s.stream, rows)?;
    Ok(input)
}

/// Rewrite `i`'s operands, in place, with `pattern`'s.
fn rewrite(i: &Input, s: &Stream, pattern: &str) -> LabResult<()> {
    for (role, bytes) in operands(i.rows, pattern) {
        i.data[role]
            .copy_from_bytes(&bytes, s)
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn set_live(i: &Input, s: &Stream, live: usize) -> LabResult<()> {
    i.live
        .copy_from_bytes(&(live as u32).to_ne_bytes(), s)
        .map_err(|e| e.to_string())
}

/// Launch the ordered-chain pair (`candidate` none) or a candidate over
/// `i` and `o`.
fn launch(
    s: &Stream,
    k: &Kernels,
    op: &Op,
    i: &Input,
    o: &Output,
    candidate: Option<&Candidate>,
    (widen, alias): (bool, bool),
) -> LabResult<()> {
    device::launch_with(s, k, i, o, alias, None, |views| {
        device::compose(&**op, o.rows as u32, views, widen, candidate)
    })
    .map(|_| ())
}

/// One eager launch from outputs reset under `prior`: the values the reset
/// left and the values after the launch (guards checked by both reads).
#[allow(clippy::too_many_arguments)]
fn eager(
    s: &Session,
    k: &Kernels,
    op: &Op,
    i: &Input,
    o: &Output,
    candidate: Option<&Candidate>,
    configuration: (bool, bool),
    prior: u32,
) -> LabResult<(Values, Values)> {
    device::reset(&s.stream, i, o, prior, configuration.1)?;
    let before = device::read(&s.stream, o)?;
    launch(&s.stream, k, op, i, o, candidate, configuration)?;
    Ok((before, device::read(&s.stream, o)?))
}

/// Whether every role's rows from `live` keep the bytes the reset wrote,
/// and every live row of the rounded sum and the normed row was written.
fn rows_respected(before: &Values, after: &Values, live: usize, rows: usize) -> bool {
    before.iter().all(|(role, data)| {
        let stride = data.len() / rows;
        data[live * stride..] == after[role][live * stride..]
    }) && ["rounded", "normed"].iter().all(|&role| {
        let stride = before[role].len() / rows;
        (0..live).all(|r| {
            let span = r * stride..(r + 1) * stride;
            before[role][span.clone()] != after[role][span]
        })
    })
}

#[test]
fn synthetic_wide_plans_compose_the_served_wide_pair() {
    let views = (1..=22)
        .map(|handle| View {
            handle,
            bytes: 1 << 34,
        })
        .collect::<Vec<_>>();
    let op = reference().unwrap();
    let served = hc_wide::finish_backend(&FLASH, hc_wide::BUCKET)
        .sinkhorn_round_expand_mix_norm(&device::query(), device::need_rows(hc_wide::BUCKET))
        .unwrap();
    let (producer, finish) = hc_wide::column_pair(&FLASH, CAPACITY);
    let c = candidate();
    assert_eq!(hc_wide::capacity(hc_wide::BUCKET), CAPACITY);
    for rows in ROWS {
        for widen in [false, true] {
            let old = device::compose(&*op, rows, &views, widen, None).unwrap();
            let symbols = |l: &[Launch]| l.iter().map(|l| l.symbol.clone()).collect::<Vec<_>>();
            assert_eq!(
                symbols(&old),
                [
                    producer.reference.symbols()[0].clone(),
                    finish.reference.symbols()[1].clone()
                ]
            );
            let new = device::compose(&*op, rows, &views, widen, Some(&c)).unwrap();
            assert_eq!(symbols(&new), c.symbols());
            assert_eq!(new[0].plan.grid, [40, rows.div_ceil(4), 1]);
            assert_eq!(new[1].plan.grid, [2, rows, 1]);
            for (new, old) in new.iter().zip(&old) {
                assert_eq!(new.arguments, old.arguments);
                assert!(new.plan.workspace_bytes <= old.plan.workspace_bytes);
                assert!(
                    old.plan.workspace_bytes <= u64::from(rows) * device::WORDS as u64 * 4,
                    "{rows}"
                );
            }
            // At the rows `wide_verify` serves, the bundle's own wide
            // selection launches exactly the composed pair.
            if rows <= hc_wide::BUCKET {
                let selected = device::compose(&*served, rows, &views, widen, None).unwrap();
                assert_eq!(selected, new, "{rows}");
            }
            for control in controls() {
                let launches = device::compose(&*op, rows, &views, widen, Some(&control)).unwrap();
                assert_eq!(symbols(&launches), control.symbols());
                assert_ne!(symbols(&launches), c.symbols());
            }
        }
    }
    assert_eq!(controls().len(), 5);
    assert_eq!(controls().iter().filter(|c| prior_of(c) == 0x80).count(), 1);
    // Each pattern is what it names, and every row holds its own values.
    let classes = |bytes: &[u8]| {
        let mut seen = std::collections::BTreeSet::new();
        for w in bytes.chunks_exact(4) {
            let v = f32::from_le_bytes(w.try_into().unwrap());
            seen.insert(match v.classify() {
                std::num::FpCategory::Nan => "nan",
                std::num::FpCategory::Infinite => "inf",
                std::num::FpCategory::Zero if v.is_sign_negative() => "-0",
                std::num::FpCategory::Zero => "+0",
                std::num::FpCategory::Subnormal => "subnormal",
                std::num::FpCategory::Normal => "normal",
            });
        }
        seen
    };
    for pattern in PATTERNS {
        let rows = 65;
        let o = operands(rows, pattern);
        for &(role, elements, per_row, bf16) in &ROLES {
            let width = elements * if bf16 { 2 } else { 4 };
            assert_eq!(o[role].len(), width * if per_row { rows } else { 1 });
            // Even rows of the signed-zero sum are all negative zeros.
            if ["streams", "comb_in"].contains(&role) || role == "sum" && pattern != "signed_zero" {
                let distinct = o[role]
                    .chunks_exact(width)
                    .collect::<std::collections::BTreeSet<_>>()
                    .len();
                assert_eq!(distinct, rows, "{pattern} {role}");
            }
        }
        let sum = classes(&o["sum"]);
        match pattern {
            "subnormal" => assert!(sum.contains("subnormal") && sum.contains("normal")),
            "signed_zero" => assert!(sum.contains("-0")),
            "nonfinite" => assert!(sum.contains("nan") && sum.contains("inf")),
            "random_bits" => assert!(sum.contains("nan") && sum.contains("normal")),
            "bf16_ties" => assert!(
                o["sum"]
                    .chunks_exact(4)
                    .filter(|w| u16::from_le_bytes([w[0], w[1]]) == 0x8000)
                    .count()
                    == rows * H / 2
            ),
            _ => assert!(
                !sum.contains("nan") && !sum.contains("inf") && !sum.contains("subnormal"),
                "{pattern}"
            ),
        }
    }
    assert_ne!(
        operands(65, "finite")["sum"],
        operands(65, "cancellation")["sum"]
    );
    assert_eq!(operands(65, "finite")["fn"], operands(129, "finite")["fn"]);
}

#[test]
#[ignore = "requires CHOPINFER_KERNEL_GPU naming an idle GPU"]
fn synthetic_wide_rows_bitwise_graphs_and_controls_on_gpu() {
    let result = kernel_harness::gpu_test(1 << 31, |s| -> LabResult<Value> {
        let k = kernels(s)?;
        let op = reference()?;
        let c = candidate();
        let mut failures = Vec::new();
        let (mut eager_pairs, mut graph_pairs, mut negatives) = (0, 0, 0);
        for rows in ROWS {
            let rows = rows as usize;
            let old = Output::new(s, rows)?;
            let new = Output::new(s, rows)?;
            old.assert_disjoint(&new)?;
            for pattern in PATTERNS {
                let i = input(s, rows, pattern)?;
                for configuration in CONFIGURATIONS {
                    for live in [rows, rows - 1] {
                        set_live(&i, &s.stream, live)?;
                        for prior in [0, 0x80] {
                            let (_, expected) =
                                eager(s, &k, &op, &i, &old, None, configuration, prior)?;
                            let (before, actual) =
                                eager(s, &k, &op, &i, &new, Some(&c), configuration, prior)?;
                            let label = format!(
                                "rows={rows} pattern={pattern} widen={} alias={} live={live} prior={prior}",
                                configuration.0, configuration.1
                            );
                            let differences = compare(&actual, &expected, Some(&c));
                            if !differences.is_empty() {
                                failures.push(format!("{label}: {differences:?}"));
                            }
                            if !rows_respected(&before, &actual, live, rows) {
                                failures.push(format!("{label}: live or padding rows"));
                            }
                            eager_pairs += 1;
                        }
                    }
                }
            }
            // Patterns, inactive rows and sticky status replay through
            // captured graphs with unchanged pointers.
            for configuration in [(true, true), (false, false)] {
                let varied = input(s, rows, "finite")?;
                let outputs = [Output::new(s, rows)?, Output::new(s, rows)?];
                let graphs = (0..2)
                    .map(|arm| {
                        s.capture(|stream| {
                            launch(
                                stream,
                                &k,
                                &op,
                                &varied,
                                &outputs[arm],
                                (arm == 1).then_some(&c),
                                configuration,
                            )
                        })
                    })
                    .collect::<LabResult<Vec<_>>>()?;
                for pattern in PATTERNS {
                    rewrite(&varied, &s.stream, pattern)?;
                    for live in [rows, 0, rows - 1] {
                        set_live(&varied, &s.stream, live)?;
                        for prior in [0, 0x80] {
                            for output in &outputs {
                                device::reset(&s.stream, &varied, output, prior, configuration.1)?;
                            }
                            let before = device::read(&s.stream, &outputs[1])?;
                            for graph in &graphs {
                                graph.launch(&s.stream).map_err(|e| e.to_string())?;
                            }
                            let reference = device::read(&s.stream, &outputs[0])?;
                            let observed = device::read(&s.stream, &outputs[1])?;
                            let label = format!(
                                "graph rows={rows} pattern={pattern} alias={} live={live} prior={prior}",
                                configuration.1
                            );
                            let differences = compare(&observed, &reference, Some(&c));
                            if !differences.is_empty() {
                                failures.push(format!("{label}: {differences:?}"));
                            }
                            if !rows_respected(&before, &observed, live, rows) {
                                failures.push(format!("{label}: live or padding rows"));
                            }
                            graph_pairs += 1;
                        }
                    }
                }
            }
            // Negative controls: every mutated column pair, a changed sum
            // in the last row and a short live count must differ.
            let i = input(s, rows, "finite")?;
            let served = (true, true);
            for control in controls() {
                let prior = prior_of(&control);
                let (_, expected) = eager(s, &k, &op, &i, &old, None, served, prior)?;
                let (_, actual) = eager(s, &k, &op, &i, &new, Some(&control), served, prior)?;
                if compare(&actual, &expected, Some(&control)).is_empty() {
                    failures.push(format!("rows={rows}: {:?} not detected", control.symbols()));
                }
                negatives += 1;
            }
            let (_, expected) = eager(s, &k, &op, &i, &old, None, served, 0)?;
            for mutation in MUTATIONS {
                let mutated = input(s, rows, "finite")?;
                match mutation {
                    "last_row_sum_sign" => {
                        let mut bytes = operands(rows, "finite").remove("sum").unwrap();
                        let at = (rows * H - 1) * 4 + 3;
                        bytes[at] ^= 0x80;
                        mutated.data["sum"]
                            .copy_from_bytes(&bytes, &s.stream)
                            .map_err(|e| e.to_string())?;
                    }
                    "live_short" => set_live(&mutated, &s.stream, rows - 1)?,
                    _ => unreachable!(),
                }
                let (_, actual) = eager(s, &k, &op, &mutated, &new, Some(&c), served, 0)?;
                if compare(&actual, &expected, Some(&c)).is_empty() {
                    failures.push(format!("rows={rows}: {mutation} not detected"));
                }
                negatives += 1;
            }
            println!(
                "SYNTHETIC_WIDE_HC_COLS_ROWS {}",
                json!({"rows":rows,"eager":eager_pairs,"graphs":graph_pairs,
                    "negatives":negatives,"failures":failures.len()})
            );
        }
        let per_rows = PATTERNS.len() * CONFIGURATIONS.len() * 2 * 2;
        assert_eq!(eager_pairs, ROWS.len() * per_rows);
        assert_eq!(graph_pairs, ROWS.len() * 2 * PATTERNS.len() * 3 * 2);
        assert_eq!(negatives, ROWS.len() * (controls().len() + MUTATIONS.len()));
        let value = json!({"rows":ROWS,"patterns":PATTERNS,"candidate":c.symbols(),
            "eager_pairs":eager_pairs,"graph_pairs":graph_pairs,"negatives":negatives,
            "failures":failures,
            "scope":"synthesized inputs; structural coverage of the wide entries above the decode route"});
        s.receipt(
            "hc-cols-synthetic-wide",
            value
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        )?;
        if !failures.is_empty() {
            return Err(format!(
                "{} failures: {:?}",
                failures.len(),
                &failures[..failures.len().min(8)]
            ));
        }
        println!(
            "SYNTHETIC_WIDE_HC_COLS_COMPLETE rows={} eager={eager_pairs} graphs={graph_pairs} negatives={negatives}",
            ROWS.len()
        );
        Ok(value)
    });
    assert!(result.is_ok(), "{result:?}");
    assert!(result.unwrap().is_some(), "an idle GPU is required");
}
