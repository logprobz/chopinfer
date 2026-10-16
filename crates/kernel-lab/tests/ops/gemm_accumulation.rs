//! Served BF16 GEMM entries of MiMo and Qwen3.8 on real weights and dumped
//! activations. Each entry runs with the split count the bundles launch it
//! with, as served (scored against the contract bound of its output dtype)
//! and with F32 output (scored against 2^-22 K S alone), next to the probe's
//! per-128-K scheme on the same weights, activations and outputs. Receipts go
//! to kind `gemm_accumulation`; the gemm README records the results.
//!
//! `gemm_accumulation_matches_probe` ties the probe to the fragment prefill:
//! whole tiles equal the probe scheme the kernels implement, bit for bit.

#[path = "../common/accumulation.rs"]
pub(super) mod probe;

use super::{
    ACTIVATION, OUTPUT, PARTIAL, SEMAPHORES, WEIGHT, query, sampled_columns, upload,
    watched_gpu_test,
};
use kernel_core::codec::bf16_to_f32;
use kernel_core::contract::{Argument, DType, RuntimeCounts};
use kernel_core::transform::Transform;
use kernel_core::transform::fragment::{FragmentWeightFormat, FragmentWeightLayout};
use kernel_lab::launch::{self, Bindings};
use kernel_lab::{LabResult, Session};
use kernels_sm120::Profile;
use kernels_sm120::gemm::{self, GemmBuffers, GemmVariant};
use probe::{
    MIMO_HIDDEN, MimoSources, Origin, PROBE, Problem, QwenSources, Scheme, Study, block, project,
    rows_of, study_columns, study_row, swiglu_rows, wide_rows,
};
use runtime_cuda::{DeviceBuffer, Function};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::rc::Rc;

/// Packed weights and inputs per (profile, weight family, packing).
type WeightCache = BTreeMap<(String, String, bool), (Rc<DeviceBuffer>, Rc<Inputs>)>;

/// The probe scheme whose outputs the fragment prefill's whole tiles equal.
const PREFILL_SCHEME: Scheme = Scheme::Chain(Some(128));

fn cubin(profile: Profile) -> String {
    format!("gemm_{}_sm120", profile.tag())
}

/// The bundles launch every shared-family entry with the split count the
/// entry requires, else one (the backend's projection op, the bundles' own
/// dense builders, bundle-qwen38 `launch.rs`).
fn served_counts(variant: &GemmVariant, m: usize) -> RuntimeCounts {
    RuntimeCounts {
        m: m as u32,
        batch: m as u32,
        context: 0,
        routed_rows: 0,
        split_k: gemm::required_split_k(&variant.key).unwrap_or(1),
    }
}

fn bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

/// One catalog entry with its packed weight on the device.
struct Served<'a> {
    variant: &'a GemmVariant,
    function: Function,
    weight: Rc<DeviceBuffer>,
    activation: DeviceBuffer,
    k: usize,
    capacity: usize,
    out_n: usize,
}

impl<'a> Served<'a> {
    fn new(
        session: &Session,
        variant: &'a GemmVariant,
        weight: Rc<DeviceBuffer>,
    ) -> LabResult<Self> {
        let q = query(variant);
        let (k, capacity) = (q.k as usize, q.m_capacity as usize);
        let function = session
            .module(&cubin(variant.profile))?
            .function(&variant.symbol)
            .map_err(|e| e.to_string())?;
        let plan = gemm::plan(&variant.key, served_counts(variant, capacity))
            .map_err(|e| e.to_string())?;
        if plan.dynamic_shared_bytes >= 48 * 1024 {
            function
                .set_max_dynamic_shared_memory(plan.dynamic_shared_bytes)
                .map_err(|e| e.to_string())?;
        }
        Ok(Self {
            variant,
            function,
            weight,
            activation: session.allocate(capacity * k * 2)?,
            k,
            capacity,
            out_n: if q.epilogues.swiglu.is_some() {
                q.n as usize / 2
            } else {
                q.n as usize
            },
        })
    }

    /// The entry's weight bytes: fragment-major when the entry reads them so.
    fn pack(variant: &GemmVariant, weight: &[u16]) -> LabResult<Vec<u8>> {
        let q = query(variant);
        let raw = bytes(weight);
        if !gemm::fragment_weight_enabled(&variant.key) {
            return Ok(raw);
        }
        Ok(FragmentWeightLayout {
            rows: q.n as usize,
            k: q.k as usize,
            format: FragmentWeightFormat::Bf16,
        }
        .pack(&raw)
        .map_err(|e| e.to_string())?
        .bytes)
    }

    /// Outputs [m][out_n] for activation rows [m][k], with F32 output if `f32`.
    fn run(&self, session: &Session, m: usize, rows: &[u16], f32: bool) -> LabResult<Vec<f32>> {
        let q = query(self.variant);
        let counts = served_counts(self.variant, m);
        let plan = gemm::plan(&self.variant.key, counts).map_err(|e| e.to_string())?;
        let mut staged = vec![0_u16; self.capacity * self.k];
        staged[..m * self.k].copy_from_slice(&rows[..m * self.k]);
        self.activation
            .copy_from_bytes(&bytes(&staged), &session.stream)
            .map_err(|e| e.to_string())?;
        let f32_output = f32 || q.output == DType::F32;
        let output = session.allocate(m * self.out_n * if f32_output { 4 } else { 2 })?;
        let partial = session.allocate((plan.workspace_bytes as usize).max(4))?;
        let semaphores = upload(
            session,
            &vec![0; (plan.workspace_bytes as usize).max(1 << 16)],
        )?;
        let maps =
            gemm::tensor_maps(&self.variant.key, WEIGHT, ACTIVATION).map_err(|e| e.to_string())?;
        let mut args = gemm::arguments(
            &self.variant.key,
            counts,
            maps,
            GemmBuffers {
                output: OUTPUT,
                partial: PARTIAL,
                semaphores: SEMAPHORES,
                weight_scales: None,
                activation_scales: None,
                residual: None,
                weight_global_scale: None,
                activation_global_scale: None,
            },
        )
        .map_err(|e| e.to_string())?;
        if f32 {
            // The entry ABI's output_f32 scalar (gemm::arguments).
            args.0[9] = Argument::U32(1);
        }
        let buffers = BTreeMap::from([
            (WEIGHT, &*self.weight),
            (ACTIVATION, &self.activation),
            (OUTPUT, &output),
            (PARTIAL, &partial),
            (SEMAPHORES, &semaphores),
        ]);
        launch::launch(
            &session.context,
            &self.function,
            &session.stream,
            plan,
            &args,
            &Bindings { buffers },
        )?;
        if f32_output {
            session.download_f32(&output, m * self.out_n)
        } else {
            Ok(session
                .download_bf16(&output, m * self.out_n)?
                .into_iter()
                .map(bf16_to_f32)
                .collect())
        }
    }
}

/// An entry's real weight (dense [n][k]) and a pool of activation rows.
struct Inputs {
    weight: Vec<u16>,
    rows: Vec<u16>,
    /// Rows at the head of the pool that are the GEMM's own dumped inputs;
    /// the rest stand in for them.
    real_rows: usize,
    origin: Origin,
    note: String,
}

impl Inputs {
    fn origin(&self, m: usize) -> Origin {
        if self.origin == Origin::Dump && m > self.real_rows {
            Origin::Proxy
        } else {
            self.origin
        }
    }
}

/// The entry's family and whether it is a decode bucket.
fn family(variant: &GemmVariant) -> Option<(String, bool)> {
    let name = variant
        .symbol
        .strip_prefix(&format!("chop_gemm_{}_", variant.profile.tag()))?
        .strip_suffix("_v1")?;
    if let Some(family) = name.strip_suffix("_prefill") {
        return Some((family.into(), false));
    }
    let (family, bucket) = name.rsplit_once("_m")?;
    bucket.parse::<u32>().ok()?;
    Some((family.into(), true))
}

/// Whether a bundle launches the entry: MiMo's fused route replaces the router
/// GEMM up to 8 rows, Qwen scores prompts with `head_f32_prefill` only, and its
/// cluster decode `_c{M}` entries replace the fragment decode shared expert.
fn served(variant: &GemmVariant, family: &str, decode: bool) -> bool {
    let q = query(variant);
    let mimo = variant.profile.tag().starts_with("mimo");
    !(mimo && family == "router" && decode && q.m_capacity < 16)
        && !(!mimo && family == "head" && !decode)
        && !(!mimo && decode && family.starts_with("shared_"))
}

fn ranks(profile: Profile) -> usize {
    profile.ranks().map_or(4, |ranks| ranks as usize)
}

/// Stacked rows of per-layer files.
fn stack(files: impl IntoIterator<Item = LabResult<Rc<Vec<u16>>>>) -> LabResult<Vec<u16>> {
    let mut rows = Vec::new();
    for file in files {
        rows.extend_from_slice(&file?);
    }
    Ok(rows)
}

fn mimo_inputs(s: &MimoSources, family: &str, n: usize, k: usize) -> LabResult<Option<Inputs>> {
    let hidden = MIMO_HIDDEN;
    let weight = |name: &str| {
        s.checkpoint
            .bf16(name)
            .map(|(shape, data)| (shape[1], data))
    };
    let drafter = |name: &str| {
        s.drafter
            .bf16(&format!("layers.0.{name}.weight"))
            .map(|(shape, data)| (shape[1], data))
    };
    let synthetic = |weight: Vec<u16>, note: &str| Inputs {
        weight,
        rows: wide_rows(2048, k, 0x4d49_4d4f ^ k as u64),
        real_rows: 0,
        origin: Origin::Synthetic,
        note: note.into(),
    };
    Ok(Some(match family {
        "o" => {
            let (cols, all) = weight("model.layers.1.self_attn.o_proj.weight")?;
            let attention =
                stack((1..=12).map(|l| s.audit(&format!("layer{l:02}.attn.bf16"), cols)))?;
            let real_rows = attention.len() / cols / 12;
            Inputs {
                weight: block(&all, cols, 0..n, k),
                rows: block(&attention, cols, 0..attention.len() / cols, k),
                real_rows,
                origin: Origin::Dump,
                note: "layer 1 o_proj; attention outputs of layers 1-12".into(),
            }
        }
        "router" => {
            let (_, gate) = weight("model.layers.1.mlp.gate.weight")?;
            let rows = stack((1..=12).map(|l| s.router_rows(l)))?;
            let real_rows = rows.len() / hidden / 12;
            Inputs {
                weight: gate,
                rows,
                real_rows,
                origin: Origin::Dump,
                note: "layer 1 router; post-attention norms of layers 1-12".into(),
            }
        }
        "head" | "head_f32" => {
            let rows = stack(
                [47].into_iter()
                    .chain((36..47).rev())
                    .map(|l| s.final_rows(l)),
            )?;
            Inputs {
                weight: s.checkpoint.bf16_rows("lm_head.weight", 0, n)?,
                real_rows: s.final_rows(47)?.len() / hidden,
                rows,
                origin: Origin::Dump,
                note: "vocabulary shard 0; final norm of layer 47, then of layers 46-36".into(),
            }
        }
        "dflash_fc" => {
            let (cols, all) = s
                .drafter
                .bf16("fc.weight")
                .map(|(shape, data)| (shape[1], data))?;
            let context = s.drafter_context()?;
            Inputs {
                weight: block(&all, cols, 0..n, k),
                real_rows: context.len() / cols,
                rows: block(&context, cols, 0..context.len() / cols, k),
                origin: Origin::Dump,
                note: "fc shard 0; normalized taps of target layers 0, 11, 23, 35, 47".into(),
            }
        }
        "dflash_q" | "dflash_qkv" | "dflash_context_kv" => {
            let order: &[&str] = match family {
                "dflash_q" => &["self_attn.q_proj"],
                "dflash_qkv" => &["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                _ => &["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
            };
            let mut rows = Vec::new();
            for name in order {
                rows.extend(drafter(name)?.1);
            }
            synthetic(
                rows_of(&rows, hidden, n, k),
                "drafter layer 0 projection rows",
            )
        }
        "dflash_o" | "dflash_down" => {
            let (cols, all) = drafter(if family == "dflash_o" {
                "self_attn.o_proj"
            } else {
                "mlp.down_proj"
            })?;
            synthetic(block(&all, cols, 0..n, k), "drafter layer 0, K shard 0")
        }
        "dflash_gate_up" => {
            let half = n / 2;
            let mut rows = rows_of(&drafter("mlp.gate_proj")?.1, hidden, half, k);
            rows.extend(rows_of(&drafter("mlp.up_proj")?.1, hidden, half, k));
            synthetic(rows, "drafter layer 0 gate and up rows")
        }
        _ => return Ok(None),
    }))
}

fn qwen_inputs(
    s: &QwenSources,
    family: &str,
    n: usize,
    k: usize,
    ranks: usize,
) -> LabResult<Option<Inputs>> {
    let name = QwenSources::name;
    let rows_from = |layer: usize, tensor: &str| -> LabResult<Vec<u16>> {
        let dumped = s.rows(layer, tensor)?;
        Ok(block(
            &dumped.1,
            dumped.0,
            0..dumped.1.len() / dumped.0,
            k.min(dumped.0),
        ))
    };
    let dense = |weight: Vec<u16>, rows: Vec<u16>, note: &str| {
        let real_rows = rows.len() / k;
        Inputs {
            weight,
            rows,
            real_rows,
            origin: Origin::Dump,
            note: note.into(),
        }
    };
    let top = |tensor: &str, count: usize| -> LabResult<Vec<u16>> {
        let weight = s.weight(tensor)?;
        Ok(rows_of(&weight.1, weight.0, count, k))
    };
    Ok(Some(match family {
        "lin_qkv" => dense(
            top(&name(0, "linear_attn.in_proj_qkv.weight"), n)?,
            rows_from(0, "x.hc.attn")?,
            "layer 0 in_proj_qkv rows; x.hc.attn",
        ),
        "lin_z" => dense(
            top(&name(0, "linear_attn.in_proj_z.weight"), n)?,
            rows_from(0, "x.hc.attn")?,
            "layer 0 in_proj_z rows; x.hc.attn",
        ),
        "attn_qkv" => {
            let mut weight = top(&name(3, "self_attn.q_proj.weight"), n - 512)?;
            weight.extend(top(&name(3, "self_attn.k_proj.weight"), 256)?);
            weight.extend(top(&name(3, "self_attn.v_proj.weight"), 256)?);
            dense(
                weight,
                rows_from(3, "x.hc.attn")?,
                "layer 3 q, k and v rows; x.hc.attn",
            )
        }
        "index_qk" => dense(
            top(&name(3, "self_attn.indexer.index_qk_proj.weight"), n)?,
            rows_from(3, "x.hc.attn")?,
            "layer 3 indexer; x.hc.attn",
        ),
        "out" => {
            let weight = s.weight(&name(0, "linear_attn.out_proj.weight"))?;
            dense(
                block(&weight.1, weight.0, 0..n, k),
                rows_from(0, "attn.core")?,
                "layer 0 out_proj K shard 0; attn.core",
            )
        }
        "router" => dense(
            top(&name(0, "mlp.gate.weight"), n)?,
            rows_from(0, "x.hc.mlp")?,
            "layer 0 router; x.hc.mlp",
        ),
        "hc_down" => dense(
            top(
                &name(0, "attn_hyper_connection.input_mix_weight_down.weight"),
                n,
            )?,
            rows_from(0, "x.hc.attn.xn")?,
            "layer 0 attention down; x.hc.attn.xn",
        ),
        "ple_key" => {
            let mut inputs = dense(
                top(&name(1, "ple.key_proj.weight"), n)?,
                rows_from(1, "x.hc.attn")?,
                "layer 1 PLE key; x.hc.attn stands in",
            );
            inputs.origin = Origin::Proxy;
            inputs
        }
        "shared_gate_up" | "shared_down" => {
            // Each rank holds 640 / ranks real rows (or K columns), zero padded.
            let real = 640 / ranks;
            let gate = s.weight(&name(0, "mlp.shared_expert.gate_proj.weight"))?;
            let up = s.weight(&name(0, "mlp.shared_expert.up_proj.weight"))?;
            let hidden_k = gate.0;
            let gate = rows_of(&gate.1, hidden_k, real, hidden_k);
            let up = rows_of(&up.1, hidden_k, real, hidden_k);
            let mixed = s.rows(0, "x.hc.mlp")?;
            if family == "shared_gate_up" {
                let half = n / 2;
                let mut weight = gate;
                weight.resize(half * k, 0);
                weight.extend(up);
                weight.resize(n * k, 0);
                dense(
                    weight,
                    mixed.1.clone(),
                    "layer 0 shared expert gate and up, zero padded; x.hc.mlp",
                )
            } else {
                let down = s.weight(&name(0, "mlp.shared_expert.down_proj.weight"))?;
                let hidden = swiglu_rows(
                    &project(&mixed.1, &gate, hidden_k),
                    &project(&mixed.1, &up, hidden_k),
                );
                dense(
                    rows_of(&down.1, down.0, n, k),
                    rows_of(&hidden, real, hidden.len() / real, k),
                    "layer 0 shared expert down, zero padded; its SwiGLU of x.hc.mlp",
                )
            }
        }
        "head" | "head_f32" => {
            let weight = s.checkpoint.bf16_rows("lm_head.weight", 0, n)?;
            let (real, mut rows) = match s.final_rows()? {
                Some((cols, rows)) if cols == k => (rows.len() / k, rows),
                _ => (0, Vec::new()),
            };
            rows.extend(rows_from(7, "x.hc.mlp")?);
            Inputs {
                weight,
                rows,
                real_rows: real,
                origin: if real > 0 {
                    Origin::Dump
                } else {
                    Origin::Proxy
                },
                note: "vocabulary shard 0; bundle final.x.hc.mixer rows, then layer 7 x.hc.mlp"
                    .into(),
            }
        }
        _ => return Ok(None),
    }))
}

/// M values of an entry: its decode bucket, or 64, 256 and 2048 prefill rows
/// (fewer where the dumps hold fewer rows).
fn row_counts(decode: bool, capacity: usize, pool: usize) -> Vec<usize> {
    if decode {
        return vec![capacity.min(pool)];
    }
    let mut counts = [64, 256, 2048]
        .into_iter()
        .filter(|&m| m <= pool)
        .collect::<Vec<_>>();
    if pool < 2048 && pool > 64 && !counts.contains(&pool) {
        counts.push(pool);
    }
    counts
}

/// Scores one entry at `m` rows over batches of the pool; decode buckets
/// repeat over new rows until 2^17 outputs.
fn study_entry(
    session: &Session,
    probe_module: &runtime_cuda::Module,
    served: &Served<'_>,
    inputs: &Inputs,
    m: usize,
) -> LabResult<Value> {
    let q = query(served.variant);
    let k = served.k;
    let half = q.epilogues.swiglu.map(|_| served.out_n);
    let columns = study_columns(served.out_n, m, &sampled_columns(served.out_n));
    let pool = inputs.rows.len() / k;
    let batches = (1_usize << 17)
        .div_ceil(m * columns.len())
        .min(pool / m)
        .max(1);
    let mut study = Study::default();
    for batch in 0..batches {
        let rows = &inputs.rows[batch * m * k..(batch + 1) * m * k];
        let as_served = served.run(session, m, rows, false)?;
        let f32 = served.run(session, m, rows, true)?;
        study.add(
            session,
            probe_module,
            (&inputs.weight, k),
            half,
            (rows, m),
            &columns,
            &as_served,
            &f32,
        )?;
    }
    let (as_served, f32, chain) = study.finish(q.output)?;
    let mut row = study_row(
        &served.variant.symbol,
        m,
        k,
        inputs.origin(m),
        &inputs.note,
        &as_served,
        &f32,
        &chain,
    );
    row["profile"] = json!(served.variant.profile.as_str());
    row["split_k"] = json!(served_counts(served.variant, m).split_k);
    row["batches"] = json!(batches);
    Ok(row)
}

#[test]
fn gemm_accumulation_entries() {
    watched_gpu_test(40 << 30, 7200, |session| {
        let mimo = MimoSources::from_env()?;
        let qwen = QwenSources::from_env("long_4k")?;
        if mimo.is_none() {
            kernel_lab::skip(
                "MiMo entries: set CHOPINFER_MIMO_CHECKPOINT and CHOPINFER_MIMO_AUDIT_DIR",
            )?;
        }
        if qwen.is_none() {
            kernel_lab::skip(
                "Qwen entries: set CHOPINFER_QWEN38_CHECKPOINT and CHOPINFER_QWEN38_LAYER_DUMPS",
            )?;
        }
        let probe_module = session.module(PROBE)?;
        let catalog = gemm::catalog(kernel_lab::catalogs());
        let mut rows = Vec::new();
        let mut findings = Vec::new();
        // Shared by the buckets of a family.
        let mut weights = WeightCache::new();
        for variant in &catalog {
            let q = query(variant);
            if q.weight != kernel_core::contract::WeightFormat::Bf16
                || q.activation != kernel_core::contract::ActivationFormat::Bf16
                || variant.symbol.ends_with("_clock_v1")
            {
                continue;
            }
            let Some((family, decode)) = family(variant) else {
                continue;
            };
            if !served(variant, &family, decode) {
                continue;
            }
            let (n, k) = (q.n as usize, q.k as usize);
            let weight_family = if family == "head_f32" {
                "head".to_string()
            } else {
                family.clone()
            };
            let key = (
                variant.profile.as_str().to_string(),
                weight_family,
                gemm::fragment_weight_enabled(&variant.key),
            );
            if !weights.contains_key(&key) {
                let inputs = match variant.profile {
                    p if p
                        .ranks_of(model_mimo26_contract::sm120::config::MODEL)
                        .is_some() =>
                    {
                        match &mimo {
                            Some(s) => mimo_inputs(s, &family, n, k)?,
                            None => None,
                        }
                    }
                    p if p
                        .ranks_of(model_qwen38_contract::sm120::config::MODEL)
                        .is_some() =>
                    {
                        match &qwen {
                            Some(s) => qwen_inputs(s, &family, n, k, ranks(variant.profile))?,
                            None => None,
                        }
                    }
                    _ => None,
                };
                let Some(inputs) = inputs else { continue };
                let packed = Served::pack(variant, &inputs.weight)?;
                weights.insert(
                    key.clone(),
                    (Rc::new(upload(session, &packed)?), Rc::new(inputs)),
                );
            }
            let (weight, inputs) = weights[&key].clone();
            let served_entry = Served::new(session, variant, weight)?;
            for m in row_counts(decode, q.m_capacity as usize, inputs.rows.len() / k) {
                let row = study_entry(session, &probe_module, &served_entry, &inputs, m)?;
                if row["served"]["violations"].as_u64() != Some(0) {
                    findings.push(format!(
                        "{} M={m}: served outputs over the contract bound",
                        variant.symbol
                    ));
                }
                rows.push(row);
            }
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), json!("entries"));
        receipt.insert("rows".into(), json!(rows));
        receipt.insert("findings".into(), json!(findings));
        eprintln!(
            "receipt: {}",
            session.receipt("gemm_accumulation", receipt)?.display()
        );
        if findings.is_empty() {
            Ok(())
        } else {
            Err(findings.join("\n"))
        }
    })
    .expect("served BF16 GEMM entries on real inputs");
}

#[test]
fn gemm_accumulation_matches_probe() {
    watched_gpu_test(8 << 30, 900, |session| {
        let symbol = "chop_gemm_qwen38_lin_qkv_prefill_v1";
        let variant = gemm::catalog(kernel_lab::catalogs())
            .into_iter()
            .find(|v| v.symbol == symbol)
            .ok_or("missing Qwen lin_qkv prefill entry")?;
        let q = query(&variant);
        let (n, k, m) = (q.n as usize, q.k as usize, 2048_usize);
        let weight = wide_rows(n, k, 0x5745_4947_4854);
        let rows = wide_rows(m, k, 0x4143_5449_5641);
        let packed = Rc::new(upload(session, &Served::pack(&variant, &weight)?)?);
        let served_entry = Served::new(session, &variant, packed)?;
        let actual = served_entry.run(session, m, &rows, true)?;
        let module = session.module(PROBE)?;
        let problem = Problem::upload(session, &weight, &rows, n, m, k)?;
        let expected = problem.run(session, &module, PREFILL_SCHEME)?;
        // Whole tiles of the prefill schedule (prefill_schedule in
        // fragment_prefill.cuh): 128 output rows by 128 tokens, dealt before
        // the stream-K tail.
        let grid = gemm::plan(&variant.key, served_counts(&variant, m)).map_err(|e| e.to_string())?.grid[0] as usize;
        let token_tiles = m.div_ceil(128);
        let tiles = n.div_ceil(128) * token_tiles;
        let mut tail = if tiles >= grid { tiles % grid } else { tiles };
        if tail * 10 >= grid * 9 {
            tail = 0;
        }
        let whole = tiles - tail;
        let (mut compared, mut mismatches, mut split_differ) = (0, 0, 0);
        for tile in 0..tiles {
            let (feature, token) = (tile / token_tiles, tile % token_tiles);
            for t in token * 128..((token + 1) * 128).min(m) {
                for c in feature * 128..((feature + 1) * 128).min(n) {
                    let equal = actual[t * n + c].to_bits() == expected[t * n + c].to_bits();
                    if tile < whole {
                        compared += 1;
                        mismatches += usize::from(!equal);
                    } else {
                        split_differ += usize::from(!equal);
                    }
                }
            }
        }
        println!(
            "{symbol} M={m}: {compared} whole-tile outputs, {mismatches} differ from {}; {split_differ} stream-K outputs differ",
            PREFILL_SCHEME.label()
        );
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), json!("matches_probe"));
        receipt.insert("symbol".into(), json!(symbol));
        receipt.insert("scheme".into(), json!(PREFILL_SCHEME.label()));
        receipt.insert("whole_tile_outputs".into(), json!(compared));
        receipt.insert("whole_tile_mismatches".into(), json!(mismatches));
        receipt.insert("stream_k_outputs_differing".into(), json!(split_differ));
        session.receipt("gemm_accumulation", receipt)?;
        if compared == 0 || mismatches != 0 {
            return Err(format!("{mismatches} of {compared} whole-tile outputs differ from the probe"));
        }
        Ok(())
    })
    .expect("fragment prefill whole tiles against the probe");
}
