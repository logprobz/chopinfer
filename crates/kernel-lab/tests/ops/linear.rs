//! Linear-attention engine against the kernel-core recurrent
//! oracles on one GPU: causal convolution, the gated delta rule for prefill,
//! decode and verify, and the gated RMSNorm at the Qwen3.8 Flash per-rank
//! shapes of `CHOPINFER_QWEN38_RANKS` (2 or 4; TP4 by default).

use crate::pool_offsets::{self, SentinelPool};
use kernel_build::EntryParam;
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{Argument, DType, RecurrentQuery};
use kernel_core::oracle::{OracleOutput, max_bound_use};
use kernel_core::recurrent_oracle::{self, GatedDeltaInputs};
use kernel_lab::launch::{self, Bindings};
use kernel_lab::poison::{self, Fill, POISON, Replay, WriteMap};
use kernel_lab::{LabResult, Session, check_output, gpu_test};
use kernels_sm120::Profile;
use kernels_sm120::linear::catalog;
use kernels_sm120::linear::{Bucket, Entry};
use runtime_cuda::{DeviceBuffer, Function, Module, Stream};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::ops::Range;

const HISTORY: usize = 3;
const DIM: u32 = model_qwen38_contract::sm120::linear::QWEN38_HEAD_DIM;
const HEAD: usize = (DIM * DIM) as usize;

#[path = "linear_pdl_prefetch.rs"]
mod pdl_prefetch;

#[path = "linear_live.rs"]
mod live;

#[path = "linear_invariance.rs"]
pub(crate) mod invariance;

/// Per-rank linear-attention geometry of the selected Qwen3.8 topology.
#[derive(Clone, Copy)]
struct Geometry {
    head_dim: usize,
    profile: Profile,
    ranks: u32,
    key_heads: usize,
    value_heads: usize,
    /// Convolution channels: q, k and v.
    channels: usize,
    /// Delta-rule output and gated norm width, value heads times 128.
    norm_width: usize,
    /// Row strides of the q|k|v rows, the a and b gates and the norm gate,
    /// each padded past its payload.
    qkv_stride: usize,
    gate_stride: usize,
    z_stride: usize,
    /// F32 delta-rule state floats per slot.
    delta_state: usize,
}

/// The geometry of `kernel_lab::qwen38_profile`. At TP4 the strides are the
/// historical 2624, 16 and 1792.
fn geometry() -> Geometry {
    let profile = kernel_lab::qwen38_profile().unwrap_or_else(|e| panic!("{e}"));
    let shape = model_qwen38_contract::sm120::config::profile_shape(profile)
        .expect("a Qwen3.8 profile has per-rank shapes");
    let key_heads = shape.linear_key_heads() as usize;
    let value_heads = shape.linear_value_heads() as usize;
    Geometry::new(
        profile,
        shape.ranks,
        key_heads,
        value_heads,
        shape.linear_value() as usize / value_heads,
    )
}

impl Geometry {
    fn new(
        profile: Profile,
        ranks: u32,
        key_heads: usize,
        value_heads: usize,
        head_dim: usize,
    ) -> Self {
        let channels = (2 * key_heads + value_heads) * head_dim;
        let norm_width = value_heads * head_dim;
        Self {
            profile,
            ranks,
            key_heads,
            value_heads,
            head_dim,
            channels,
            norm_width,
            qkv_stride: channels + 64,
            gate_stride: (value_heads + 4).next_multiple_of(8),
            z_stride: norm_width + 256,
            delta_state: value_heads * head_dim * head_dim,
        }
    }
}

/// `kernel_lab::qwen38_lean`: TP2 runs the smallest cases that still
/// launch every linear entry; TP4 keeps the full cases.
fn lean() -> bool {
    kernel_lab::qwen38_lean().unwrap_or_else(|e| panic!("{e}"))
}

/// `chop_linear_{tag}_{name}_v1` of the selected topology.
fn symbol(name: &str) -> String {
    format!("chop_linear_{}_{name}_v1", geometry().profile.tag())
}

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    pub fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1_u64 << 53) as f64
    }
    pub fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform().max(1e-300), self.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
    pub fn bf16(&mut self, len: usize, scale: f64) -> Vec<u16> {
        (0..len)
            .map(|_| f32_to_bf16((self.normal() * scale) as f32))
            .collect()
    }
    pub fn f32(&mut self, len: usize, scale: f64) -> Vec<f32> {
        (0..len).map(|_| (self.normal() * scale) as f32).collect()
    }
}

pub fn decode(bits: &[u16]) -> Vec<f64> {
    bits.iter().map(|&x| f64::from(bf16_to_f32(x))).collect()
}

fn bytes_u16(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}
fn bytes_i32(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}
fn bytes_f32(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

pub fn find(symbol: &str) -> Entry {
    catalog(kernel_lab::catalogs())
        .into_iter()
        .find(|entry| entry.symbol == symbol)
        .unwrap_or_else(|| panic!("missing linear entry {symbol}"))
}

/// The selected topology's linear cubin, after checking that its catalog
/// entries carry the geometry the cases are built with.
pub fn module(session: &Session) -> LabResult<Module> {
    let profile = kernel_lab::qwen38_profile()?;
    let g = geometry();
    let name = match profile {
        model_qwen38_contract::sm120::config::PROFILE => "linear_qwen38_flash_sm120".to_owned(),
        other => format!("linear_{}_sm120", other.tag()),
    };
    let conv = phase_query(&symbol("conv_prefill"));
    let delta = phase_query(&symbol("delta_decode"));
    let norm = phase_query(&symbol("gated_norm"));
    let expected = (g.channels, g.key_heads, g.value_heads, g.norm_width);
    let actual = match (conv, delta, norm) {
        (
            RecurrentQuery::CausalConv1d { channels, .. },
            RecurrentQuery::GatedDeltaRule {
                key_heads,
                value_heads,
                ..
            },
            RecurrentQuery::GatedRmsNorm { width, .. },
        ) => (
            channels as usize,
            key_heads as usize,
            value_heads as usize,
            width as usize,
        ),
        _ => return Err(format!("unexpected linear entry queries at {name}")),
    };
    if actual != expected {
        return Err(format!(
            "{name} entries have (channels, key heads, value heads, width) {actual:?}, cases use {expected:?}"
        ));
    }
    session
        .context
        .load_module(kernel_lab::cubin(&name)?)
        .map_err(|e| e.to_string())
}

/// Named device operands of one case, bound by parameter name.
pub struct Operands {
    pub buffers: BTreeMap<&'static str, DeviceBuffer>,
    pub scalars: BTreeMap<&'static str, u32>,
}

impl Operands {
    pub fn new() -> Self {
        Self {
            buffers: BTreeMap::new(),
            scalars: BTreeMap::new(),
        }
    }
    pub fn bytes(&mut self, session: &Session, name: &'static str, bytes: &[u8]) -> LabResult<()> {
        let buffer = session.allocate(bytes.len().max(4))?;
        buffer
            .copy_from_bytes(bytes, &session.stream)
            .map_err(|e| e.to_string())?;
        self.buffers.insert(name, buffer);
        Ok(())
    }
    pub fn bf16(&mut self, session: &Session, name: &'static str, v: &[u16]) -> LabResult<()> {
        self.bytes(session, name, &bytes_u16(v))
    }
    pub fn i32(&mut self, session: &Session, name: &'static str, v: &[i32]) -> LabResult<()> {
        self.bytes(session, name, &bytes_i32(v))
    }
    pub fn f32(&mut self, session: &Session, name: &'static str, v: &[f32]) -> LabResult<()> {
        self.bytes(session, name, &bytes_f32(v))
    }
    pub fn u32(&mut self, session: &Session, name: &'static str, v: u32) -> LabResult<()> {
        self.bytes(session, name, &v.to_ne_bytes())
    }
    pub fn zero(&mut self, session: &Session, name: &'static str, bytes: usize) -> LabResult<()> {
        self.bytes(session, name, &vec![0; bytes])
    }
    pub fn read(&self, session: &Session, name: &str, bytes: usize) -> LabResult<Vec<u8>> {
        let mut out = vec![0; bytes];
        self.buffers[name]
            .copy_to_bytes(&mut out, &session.stream)
            .map_err(|e| e.to_string())?;
        Ok(out)
    }
    pub fn read_bf16(&self, session: &Session, name: &str, len: usize) -> LabResult<Vec<u16>> {
        Ok(self
            .read(session, name, len * 2)?
            .chunks_exact(2)
            .map(|c| u16::from_ne_bytes([c[0], c[1]]))
            .collect())
    }
    pub fn read_f32(&self, session: &Session, name: &str, len: usize) -> LabResult<Vec<f32>> {
        Ok(self
            .read(session, name, len * 4)?
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }
}

/// A catalog entry resolved in the loaded cubin.
pub struct Kernel {
    pub entry: Entry,
    pub function: Function,
    /// Prepared bucket of every launch; the catalog capacity unless narrowed.
    pub bucket: Bucket,
}

impl Kernel {
    pub fn load(module: &Module, symbol: &str) -> LabResult<Self> {
        let entry = find(symbol);
        let function = module.function(symbol).map_err(|e| e.to_string())?;
        let shared = entry.dynamic_shared_bytes();
        if shared > 0 {
            function
                .set_max_dynamic_shared_memory(shared)
                .map_err(|e| e.to_string())?;
        }
        let bucket = entry.capacity();
        Ok(Self {
            entry,
            function,
            bucket,
        })
    }

    /// The same kernel planned for a narrower bucket, as a bundle does for
    /// each captured batch size.
    pub fn with_bucket(mut self, rows: u32, sequences: u32) -> Self {
        self.bucket = Bucket { rows, sequences };
        self
    }

    pub fn workspace_bytes(&self) -> usize {
        self.entry.workspace_bytes(self.bucket) as usize
    }

    pub fn launch(
        &self,
        session: &Session,
        stream: &Stream,
        operands: &Operands,
        live_rows: u32,
        live_sequences: u32,
        pdl: bool,
    ) -> LabResult<()> {
        let names = operands.buffers.keys().copied().collect::<Vec<_>>();
        let values = self
            .entry
            .params
            .iter()
            .map(|param| match param {
                EntryParam::DevicePointer { name, .. } => names
                    .iter()
                    .position(|n| n == name)
                    .map(|i| Argument::DeviceBuffer(i as u64 + 1))
                    .ok_or_else(|| format!("missing buffer {name}")),
                EntryParam::U32 { name } => operands
                    .scalars
                    .get(name.as_str())
                    .copied()
                    .map(Argument::U32)
                    .ok_or_else(|| format!("missing scalar {name}")),
                other => Err(format!("unexpected parameter {other:?}")),
            })
            .collect::<LabResult<Vec<_>>>()?;
        let args = self
            .entry
            .pack_arguments(values)
            .map_err(|e| e.to_string())?;
        let mut plan = self
            .entry
            .plan(self.bucket, live_rows, live_sequences)
            .map_err(|e| e.to_string())?;
        plan.pdl = pdl;
        let bindings = Bindings {
            buffers: operands
                .buffers
                .values()
                .enumerate()
                .map(|(i, b)| (i as u64 + 1, b))
                .collect(),
        };
        launch::launch(
            &session.context,
            &self.function,
            stream,
            plan,
            &args,
            &bindings,
        )?;
        kernel_lab::launched(&self.entry.symbol);
        Ok(())
    }
}

fn verify_row_limit() -> u32 {
    let bucket = find(&symbol("delta_verify")).capacity();
    bucket.rows / bucket.sequences
}

fn phase_query(symbol: &str) -> RecurrentQuery {
    find(symbol).query
}

/// One receipt per case holding every sub-check's verdict and error summary.
pub struct Report {
    case: String,
    checks: Vec<Value>,
    failures: Vec<String>,
}

impl Report {
    pub fn new(case: impl Into<String>) -> Self {
        Self {
            case: case.into(),
            checks: Vec::new(),
            failures: Vec::new(),
        }
    }

    /// Numerical contract gate against an oracle output.
    pub fn check(&mut self, name: &str, actual: &[f64], reference: &OracleOutput, dtype: DType) {
        let mut data = BTreeMap::new();
        let actual = actual.iter().map(|&x| x as f32).collect::<Vec<_>>();
        let correct = check_output(&actual, reference, dtype, &mut data).is_ok();
        let errors = actual
            .iter()
            .zip(&reference.values)
            .map(|(&a, &r)| (f64::from(a) - r).abs());
        let beyond_rounding = errors
            .clone()
            .zip(&reference.values)
            .map(|(error, &r)| {
                let half = if dtype == DType::Bf16 && r != 0.0 {
                    2_f64.powi(r.abs().log2().floor() as i32 - 8)
                } else {
                    0.0
                };
                (error - half).max(0.0)
            })
            .fold(0.0, f64::max);
        // Largest error as a fraction of its element's numerical bound.
        let bound_use = max_bound_use(&actual, reference, dtype);
        let rms = (reference.values.iter().map(|v| v * v).sum::<f64>()
            / reference.values.len().max(1) as f64)
            .sqrt();
        self.checks.push(json!({
            "name": name,
            "correct": correct,
            "elements": actual.len(),
            "max_abs_error": errors.fold(0.0, f64::max),
            "max_error_beyond_bf16_rounding": beyond_rounding,
            "reference_rms": rms,
            "max_s": reference.sum_abs_products.iter().copied().fold(0.0, f64::max),
            "max_error_over_bound": bound_use,
            "worst_element": data.get("worst_element"),
        }));
        if !correct {
            self.failures.push(format!(
                "{name}: {}",
                data.get("worst_element").unwrap_or(&Value::Null)
            ));
        }
    }

    /// A bitwise or structural check.
    pub fn exact(&mut self, name: &str, result: Result<(), String>) {
        self.checks.push(
            json!({"name": name, "correct": result.is_ok(), "detail": result.as_ref().err()}),
        );
        if let Err(error) = result {
            self.failures.push(format!("{name}: {error}"));
        }
    }

    pub fn finish(self, session: &Session) -> LabResult<()> {
        let mut data = BTreeMap::new();
        data.insert("case".into(), json!(self.case));
        data.insert("gate".into(), json!("G1"));
        data.insert("correct".into(), json!(self.failures.is_empty()));
        data.insert("checks".into(), json!(self.checks));
        let g = geometry();
        if g.ranks != 4 {
            data.insert("qwen38_ranks".into(), json!(g.ranks));
        }
        if lean() {
            data.insert("lean".into(), json!(true));
        }
        session.receipt("linear", data)?;
        if self.failures.is_empty() {
            Ok(())
        } else {
            Err(format!("{}: {}", self.case, self.failures.join("; ")))
        }
    }
}

/// Distinct pool slots in a scattered order, plus unused slots around them.
fn slots(sequences: usize, extra: usize) -> (Vec<i32>, usize) {
    let pool = 2 * sequences + extra;
    (
        (0..sequences).map(|s| (pool - 1 - 2 * s) as i32).collect(),
        pool,
    )
}

fn cu(lengths: &[usize]) -> Vec<i32> {
    let mut cu = vec![0_i32];
    for &len in lengths {
        cu.push(cu.last().unwrap() + len as i32);
    }
    cu
}

// ---------------------------------------------------------------- conv ----

struct ConvCase {
    geometry: Geometry,
    weight: Vec<u16>,
    input: Vec<u16>,
    pool: Vec<u16>,
}

impl ConvCase {
    fn new(rng: &mut Rng, rows: usize, slots: usize) -> Self {
        Self::with_geometry(rng, rows, slots, geometry())
    }

    fn with_geometry(rng: &mut Rng, rows: usize, slots: usize, g: Geometry) -> Self {
        Self {
            geometry: g,
            weight: rng.bf16(g.channels * 4, 0.1),
            input: rng.bf16(rows * g.qkv_stride, 1.0),
            pool: rng.bf16(slots * HISTORY * g.channels, 1.0),
        }
    }
    fn rows(&self, first: usize, count: usize) -> Vec<f64> {
        let g = self.geometry;
        let c = g.channels;
        (first..first + count)
            .flat_map(|r| decode(&self.input[r * g.qkv_stride..r * g.qkv_stride + c]))
            .collect()
    }
    fn slot(&self, slot: i32) -> Vec<f64> {
        let g = self.geometry;
        let n = HISTORY * g.channels;
        decode(&self.pool[slot as usize * n..(slot as usize + 1) * n])
    }
    fn operands(&self, session: &Session, rows: usize) -> LabResult<Operands> {
        let g = self.geometry;
        let mut op = Operands::new();
        op.bf16(session, "input", &self.input)?;
        op.bf16(session, "weight", &self.weight)?;
        op.bf16(session, "state", &self.pool)?;
        op.zero(session, "output", rows * g.channels * 2)?;
        op.scalars.insert("input_stride", g.qkv_stride as u32);
        Ok(op)
    }
}

fn compare_state_bits(actual: &[u16], expected: &[f64]) -> LabResult<()> {
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        if a != f32_to_bf16(e as f32) {
            return Err(format!("state element {i} is {a:#06x}, expected {e}"));
        }
    }
    Ok(())
}

fn unchanged_slots(
    before: &[u16],
    after: &[u16],
    touched: &[i32],
    per_slot: usize,
) -> LabResult<()> {
    for slot in 0..before.len() / per_slot {
        if touched.contains(&(slot as i32)) {
            continue;
        }
        let range = slot * per_slot..(slot + 1) * per_slot;
        if before[range.clone()] != after[range] {
            return Err(format!("untouched slot {slot} changed"));
        }
    }
    Ok(())
}

pub fn conv_prefill(
    session: &Session,
    module: &Module,
    lengths: &[usize],
    initial: &[bool],
    seed: u64,
) -> LabResult<Vec<u16>> {
    let case = format!("conv_prefill_{lengths:?}_{initial:?}");
    conv_prefill_run(session, module, case, lengths, initial, false, seed)
}

/// `conv_prefill` as case `case`, planned at the catalog capacity or, with
/// `own_bucket`, at the batch's rows and sequences as a bundle's eager
/// prefill plans it.
fn conv_prefill_run(
    session: &Session,
    module: &Module,
    case: String,
    lengths: &[usize],
    initial: &[bool],
    own_bucket: bool,
    seed: u64,
) -> LabResult<Vec<u16>> {
    let g = geometry();
    let mut report = Report::new(case);
    let c = g.channels;
    let rows = lengths.iter().sum::<usize>();
    let mut rng = Rng::new(seed);
    let (slot_ids, pool) = slots(lengths.len(), 2);
    let data = ConvCase::new(&mut rng, rows, pool);
    let mut op = data.operands(session, rows)?;
    let cu = cu(lengths);
    op.i32(session, "cu_seqlens", &cu)?;
    op.i32(session, "slot_ids", &slot_ids)?;
    op.i32(
        session,
        "initial",
        &initial.iter().map(|&x| i32::from(x)).collect::<Vec<_>>(),
    )?;
    op.i32(session, "anchor_ids", &vec![-1; initial.len()])?;
    op.u32(session, "live_sequences", lengths.len() as u32)?;
    let mut kernel = Kernel::load(module, &symbol("conv_prefill"))?;
    if own_bucket {
        kernel = kernel.with_bucket(rows as u32, lengths.len() as u32);
    }
    kernel.launch(
        session,
        &session.stream,
        &op,
        rows as u32,
        lengths.len() as u32,
        true,
    )?;
    let output = op.read_bf16(session, "output", rows * c)?;
    let pool_after = op.read_bf16(session, "state", data.pool.len())?;
    let query = phase_query(&symbol("conv_prefill"));
    let weight = decode(&data.weight);
    for (s, &len) in lengths.iter().enumerate() {
        let start = cu[s] as usize;
        let state = if initial[s] {
            data.slot(slot_ids[s])
        } else {
            vec![0.0; HISTORY * c]
        };
        let expected =
            recurrent_oracle::causal_conv1d(query, &data.rows(start, len), &weight, &state)
                .map_err(|e| e.to_string())?;
        report.check(
            &format!("seq{s}_output"),
            &decode(&output[start * c..(start + len) * c]),
            &expected.output,
            DType::Bf16,
        );
        let n = HISTORY * c;
        let slot = slot_ids[s] as usize;
        report.exact(
            "state bits",
            compare_state_bits(&pool_after[slot * n..(slot + 1) * n], &expected.state),
        );
    }
    report.exact(
        "untouched slots",
        unchanged_slots(&data.pool, &pool_after, &slot_ids, HISTORY * c),
    );
    report.finish(session)?;
    Ok(output)
}

fn conv_decode(session: &Session, module: &Module, batch: usize, seed: u64) -> LabResult<()> {
    let g = geometry();
    let case = format!("conv_decode_b{batch}");
    let mut report = Report::new(case.clone());
    let c = g.channels;
    let mut rng = Rng::new(seed);
    let (slot_ids, pool) = slots(batch, 3);
    let data = ConvCase::new(&mut rng, batch, pool);
    let mut op = data.operands(session, batch)?;
    op.i32(session, "slot_ids", &slot_ids)?;
    op.u32(session, "live_sequences", batch as u32)?;
    let kernel = Kernel::load(module, &symbol("conv_decode"))?;
    kernel.launch(
        session,
        &session.stream,
        &op,
        batch as u32,
        batch as u32,
        true,
    )?;
    let output = op.read_bf16(session, "output", batch * c)?;
    let pool_after = op.read_bf16(session, "state", data.pool.len())?;
    let query = phase_query(&symbol("conv_decode"));
    let weight = decode(&data.weight);
    let n = HISTORY * c;
    for s in 0..batch {
        let expected = recurrent_oracle::causal_conv1d(
            query,
            &data.rows(s, 1),
            &weight,
            &data.slot(slot_ids[s]),
        )
        .map_err(|e| e.to_string())?;
        report.check(
            &format!("seq{s}"),
            &decode(&output[s * c..(s + 1) * c]),
            &expected.output,
            DType::Bf16,
        );
        let slot = slot_ids[s] as usize;
        report.exact(
            "state bits",
            compare_state_bits(&pool_after[slot * n..(slot + 1) * n], &expected.state),
        );
    }
    report.exact(
        "untouched slots",
        unchanged_slots(&data.pool, &pool_after, &slot_ids, n),
    );
    report.finish(session)
}

fn verify_rows(sequences: usize, rng: &mut Rng) -> Vec<usize> {
    (0..sequences)
        .map(|s| {
            if s == 0 {
                verify_row_limit() as usize
            } else {
                1 + (rng.next() % u64::from(verify_row_limit())) as usize
            }
        })
        .collect()
}

fn checkpoints(slot_ids: &[i32], pool: usize) -> (Vec<i32>, usize) {
    let max = verify_row_limit() as usize;
    let ids = (0..slot_ids.len() * max)
        .map(|i| (pool + slot_ids.len() * max - 1 - i) as i32)
        .collect();
    (ids, pool + slot_ids.len() * max)
}

/// The sequences, pool entries and bucket of one verify launch.
struct VerifyLayout {
    /// Rows of every sequence the arrays hold: the live ones, then any
    /// that pad the bucket.
    lengths: Vec<usize>,
    live: usize,
    slot_ids: Vec<i32>,
    checkpoint_ids: Vec<i32>,
    /// Run compact-versus-disjoint bitwise checks.
    compact: bool,
    /// Pool entries.
    pool: usize,
    /// The planned bucket; the catalog capacity when `None`.
    bucket: Option<Bucket>,
}

impl VerifyLayout {
    /// `sequences` live sequences of `verify_rows` rows each, scattered
    /// slots and distinct checkpoints, at the catalog capacity.
    fn verify(sequences: usize, rng: &mut Rng) -> Self {
        let lengths = verify_rows(sequences, rng);
        let (slot_ids, base) = slots(sequences, 1);
        let (checkpoint_ids, pool) = checkpoints(&slot_ids, base);
        Self {
            lengths,
            live: sequences,
            slot_ids,
            checkpoint_ids,
            pool,
            compact: false,
            bucket: None,
        }
    }

    /// The bundle's decode through the verify entries (Qwen above 32 sequences): one row
    /// for each of `live` sequences in the next power-of-two bucket of
    /// sequences, planned at `Bucket { rows: bucket, sequences: bucket }`.
    /// Live sequences read committed entries at odd pool indices in reverse
    /// order and write their first checkpoint at even ones; their other
    /// checkpoint rows name one spare entry, and the sequences past the live
    /// count another, with a row each. Both spares must stay untouched, and
    /// the padding rows' outputs zero.
    fn decode(live: usize) -> Self {
        let bucket = live.next_power_of_two();
        let max = verify_row_limit() as usize;
        let (spare, padding) = (2 * live as i32, 2 * live as i32 + 1);
        let slot_ids = (0..bucket)
            .map(|s| {
                if s < live {
                    (2 * (live - 1 - s) + 1) as i32
                } else {
                    padding
                }
            })
            .collect();
        let checkpoint_ids = (0..bucket * max)
            .map(|i| match (i / max, i % max) {
                (s, _) if s >= live => padding,
                (s, 0) => 2 * s as i32,
                _ => spare,
            })
            .collect();
        Self {
            lengths: vec![1; bucket],
            live,
            slot_ids,
            checkpoint_ids,
            pool: 2 * live + 2,
            compact: false,
            bucket: Some(Bucket {
                rows: bucket as u32,
                sequences: bucket as u32,
            }),
        }
    }

    /// One committed entry per slot and four checkpoints for at most 32
    /// drafting items. AR rows alias their input state. Padded slots and
    /// unused checkpoint rows remain sentinel entries.
    fn checkpoint_pool(live: usize, drafting: bool) -> Self {
        let items = live.next_power_of_two();
        let row_budget = if live <= 32 { 32 } else { 256 };
        let drafts = if drafting {
            live.min(32).min((row_budget - live) / 3)
        } else {
            0
        };
        let spare = live + 4 * drafts;
        let slot_ids = (0..items)
            .map(|i| {
                if i < live {
                    (live - 1 - i) as i32
                } else {
                    spare as i32
                }
            })
            .collect::<Vec<_>>();
        let lengths = (0..items)
            .map(|i| if i < drafts { 4 } else { 1 })
            .collect::<Vec<_>>();
        let checkpoint_ids = (0..items * verify_row_limit() as usize)
            .map(|i| {
                let item = i / verify_row_limit() as usize;
                let row = i % verify_row_limit() as usize;
                if item < drafts && row < 4 {
                    (live + item * 4 + row) as i32
                } else if item < live && row == 0 {
                    slot_ids[item]
                } else {
                    spare as i32
                }
            })
            .collect();
        Self {
            lengths,
            live,
            slot_ids,
            checkpoint_ids,
            pool: spare + 1,
            compact: true,
            bucket: Some(Bucket {
                rows: (live + 3 * drafts).next_power_of_two() as u32,
                sequences: items as u32,
            }),
        }
    }

    /// Entry `name` of the selected topology planned at the layout's bucket.
    fn kernel(&self, module: &Module, name: &str) -> LabResult<Kernel> {
        let kernel = Kernel::load(module, &symbol(name))?;
        Ok(match self.bucket {
            Some(bucket) => kernel.with_bucket(bucket.rows, bucket.sequences),
            None => kernel,
        })
    }
}

/// Every output element below `live` is written and every one past it keeps
/// its poison: the kernels return early for rows past the live sequences
/// (conv.cuh, delta.cuh).
fn poisoned_tail(output: &[u16], live: usize) -> Result<(), String> {
    poison::check(
        "output",
        &bytes_u16(output),
        &WriteMap::of(DType::Bf16).prefix(live)?,
        Fill::Pattern(&[POISON]),
    )
}

fn conv_verify(session: &Session, module: &Module, sequences: usize, seed: u64) -> LabResult<()> {
    let mut rng = Rng::new(seed);
    let layout = VerifyLayout::verify(sequences, &mut rng);
    let case = format!("conv_verify_s{sequences}");
    conv_verify_run(session, module, case, &mut rng, &layout)
}

/// Compare the compact pool with disjoint checkpoint writes from exactly
/// the same initial bytes. Then replay the compact layout with and without
/// PDL and through a graph, requiring identical output and state bytes.
#[allow(clippy::too_many_arguments)]
fn compact_pool_bits(
    session: &Session,
    kernel: &Kernel,
    op: &mut Operands,
    layout: &VerifyLayout,
    initial: &[u8],
    after: &[u8],
    output: &[u8],
    entry_bytes: usize,
    element: DType,
    report: &mut Report,
) -> LabResult<()> {
    let max = verify_row_limit() as usize;
    if !layout.compact {
        return Ok(());
    }
    let rows = layout.lengths[..layout.live].iter().sum::<usize>() as u32;
    // Disjoint checkpoint entries start poisoned: the launch must write them.
    let mut separate = initial.to_vec();
    separate.resize(initial.len() + layout.live * max * entry_bytes, POISON);
    poison::poison(session, &[&op.buffers["output"]])?;
    let mut ids = layout.checkpoint_ids.clone();
    for (i, id) in ids.iter_mut().take(layout.live * max).enumerate() {
        *id = (layout.pool + i) as i32;
    }
    op.bytes(session, "state", &separate)?;
    op.i32(session, "checkpoint_ids", &ids)?;
    kernel.launch(session, &session.stream, op, rows, layout.live as u32, true)?;
    let baseline = op.read(session, "state", separate.len())?;
    let same = op.read(session, "output", output.len())? == output
        && (0..layout.live).all(|i| {
            (0..layout.lengths[i]).all(|r| {
                let from = ids[i * max + r] as usize * entry_bytes;
                let to = layout.checkpoint_ids[i * max + r] as usize * entry_bytes;
                baseline[from..from + entry_bytes] == after[to..to + entry_bytes]
            })
        });
    report.exact(
        "compact pool equals disjoint checkpoints bitwise",
        if same {
            Ok(())
        } else {
            Err("aliased outputs or state differ".into())
        },
    );
    op.i32(session, "checkpoint_ids", &layout.checkpoint_ids)?;
    op.bytes(session, "state", initial)?;
    let graph = session
        .capture(|stream| kernel.launch(session, stream, op, rows, layout.live as u32, true))?;
    // Drafting items' checkpoints are written without being read and start
    // poisoned; an AR row's checkpoint is its own slot, read and rewritten.
    let (mut write_only, mut updated) = (Vec::new(), Vec::new());
    for item in 0..layout.live {
        for row in 0..layout.lengths[item] {
            let entry = layout.checkpoint_ids[item * max + row] as usize;
            let range = entry * entry_bytes..(entry + 1) * entry_bytes;
            if entry == layout.slot_ids[item] as usize {
                updated.push(range);
            } else {
                write_only.push(range);
            }
        }
    }
    let output_buffer = &op.buffers["output"];
    let row_bytes = output.len() / layout.lengths.iter().sum::<usize>().max(1);
    let live_bytes = row_bytes * rows as usize;
    let mut checks = Replay::new(session)
        .output(
            "output",
            output_buffer,
            WriteMap::of(DType::Bf16).byte_prefix(live_bytes)?,
        )
        .state(
            "state",
            &op.buffers["state"],
            initial.to_vec(),
            WriteMap::of(element).byte_ranges(write_only)?,
            updated,
        );
    let eager = checks.eager(|| {
        kernel.launch(
            session,
            &session.stream,
            op,
            rows,
            layout.live as u32,
            false,
        )
    });
    let matches_first = eager.and_then(|()| {
        let same = checks.expected("state") == Some(after)
            && checks
                .expected("output")
                .map(|bytes| &bytes[..output.len()])
                == Some(output);
        if same {
            Ok(())
        } else {
            Err("the PDL-free run differs from the first launch".into())
        }
    });
    report.exact("compact pool PDL and graph replay bits", matches_first);
    for _ in 0..2 {
        report.exact(
            "compact pool PDL and graph replay bits",
            checks.graph(&graph),
        );
    }
    Ok(())
}

fn conv_verify_run(
    session: &Session,
    module: &Module,
    case: String,
    rng: &mut Rng,
    layout: &VerifyLayout,
) -> LabResult<()> {
    let g = geometry();
    let mut report = Report::new(case);
    let c = g.channels;
    let n = HISTORY * c;
    let max = verify_row_limit() as usize;
    let (lengths, slot_ids, checkpoint_ids) =
        (&layout.lengths, &layout.slot_ids, &layout.checkpoint_ids);
    let rows = lengths.iter().sum::<usize>();
    let data = ConvCase::new(rng, rows, layout.pool);
    let mut op = data.operands(session, rows)?;
    let cu = cu(lengths);
    let live_rows = cu[layout.live] as usize;
    op.i32(session, "cu_rows", &cu)?;
    op.i32(session, "slot_ids", slot_ids)?;
    op.i32(session, "checkpoint_ids", checkpoint_ids)?;
    op.u32(session, "live_sequences", layout.live as u32)?;
    poison::poison(session, &[&op.buffers["output"]])?;
    let kernel = layout.kernel(module, "conv_verify")?;
    kernel.launch(
        session,
        &session.stream,
        &op,
        live_rows as u32,
        layout.live as u32,
        true,
    )?;
    let output = op.read_bf16(session, "output", rows * c)?;
    let pool_after = op.read_bf16(session, "state", data.pool.len())?;
    let query = phase_query(&symbol("conv_verify"));
    let weight = decode(&data.weight);
    let mut touched = Vec::new();
    for (s, &len) in lengths[..layout.live].iter().enumerate() {
        let start = cu[s] as usize;
        let expected = recurrent_oracle::causal_conv1d(
            query,
            &data.rows(start, len),
            &weight,
            &data.slot(slot_ids[s]),
        )
        .map_err(|e| e.to_string())?;
        report.check(
            &format!("seq{s}"),
            &decode(&output[start * c..(start + len) * c]),
            &expected.output,
            DType::Bf16,
        );
        for (r, state) in expected.row_states.iter().enumerate() {
            let slot = checkpoint_ids[s * max + r] as usize;
            touched.push(slot as i32);
            report.exact(
                "state bits",
                compare_state_bits(&pool_after[slot * n..(slot + 1) * n], state),
            );
        }
    }
    report.exact(
        "untouched slots",
        unchanged_slots(&data.pool, &pool_after, &touched, n),
    );
    report.exact(
        "rows past the live sequences",
        poisoned_tail(&output, live_rows * c),
    );
    compact_pool_bits(
        session,
        &kernel,
        &mut op,
        layout,
        &bytes_u16(&data.pool),
        &bytes_u16(&pool_after),
        &bytes_u16(&output),
        n * 2,
        DType::Bf16,
        &mut report,
    )?;
    report.finish(session)
}

// ---------------------------------------------------------- delta rule ----

pub struct DeltaCase {
    geometry: Geometry,
    pub qkv: Vec<u16>,
    pub a: Vec<u16>,
    pub b: Vec<u16>,
    pub a_log: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub pool: Vec<f32>,
}

impl DeltaCase {
    /// Decay parameters span the checkpoint's range: exp(A_log) from 0.03
    /// to 160 and dt_bias from -8 to 2.5, so some heads forget within a row.
    pub fn new(rng: &mut Rng, rows: usize, slots: usize) -> Self {
        Self::with_geometry(rng, rows, slots, geometry())
    }

    fn with_geometry(rng: &mut Rng, rows: usize, slots: usize, g: Geometry) -> Self {
        let heads = g.value_heads;
        let a_log = (0..heads)
            .map(|h| {
                let t = h as f64 / (heads - 1).max(1) as f64;
                (0.03_f64.ln() + t * (160.0_f64 / 0.03).ln()) as f32
            })
            .collect();
        let dt_bias = (0..heads)
            .map(|_| (-8.0 + 10.5 * rng.uniform()) as f32)
            .collect();
        Self {
            geometry: g,
            qkv: rng.bf16(rows * g.qkv_stride, 1.0),
            a: rng.bf16(rows * g.gate_stride, 1.5),
            b: rng.bf16(rows * g.gate_stride, 1.5),
            a_log,
            dt_bias,
            pool: rng.f32(slots * g.delta_state, 0.3),
        }
    }

    pub fn operands(&self, session: &Session, rows: usize) -> LabResult<Operands> {
        let g = self.geometry;
        let mut op = Operands::new();
        op.bf16(session, "qkv", &self.qkv)?;
        op.bf16(session, "a", &self.a)?;
        op.bf16(session, "b", &self.b)?;
        op.f32(session, "a_log", &self.a_log)?;
        op.f32(session, "dt_bias", &self.dt_bias)?;
        op.f32(session, "state", &self.pool)?;
        op.zero(session, "output", rows * g.norm_width * 2)?;
        op.scalars.insert("qkv_stride", g.qkv_stride as u32);
        op.scalars.insert("gate_stride", g.gate_stride as u32);
        Ok(op)
    }

    fn slot(&self, slot: i32) -> Vec<f64> {
        let g = self.geometry;
        let n = g.delta_state;
        self.pool[slot as usize * n..(slot as usize + 1) * n]
            .iter()
            .map(|&x| f64::from(x))
            .collect()
    }

    /// Oracle for rows `first..first + count` of one sequence.
    pub fn oracle(
        &self,
        query: RecurrentQuery,
        first: usize,
        count: usize,
        state: &[f64],
    ) -> LabResult<recurrent_oracle::GatedDeltaOutput> {
        let g = self.geometry;
        delta_oracle(query, &self.qkv, g.qkv_stride, self, first, count, state)
    }
}

/// Delta-rule oracle over rows of a [rows, stride] q|k|v buffer, with the
/// gates and decay parameters of `gates`.
pub fn delta_oracle(
    query: RecurrentQuery,
    qkv: &[u16],
    stride: usize,
    gates: &DeltaCase,
    first: usize,
    count: usize,
    state: &[f64],
) -> LabResult<recurrent_oracle::GatedDeltaOutput> {
    let g = gates.geometry;
    let (kh, vh, d) = (g.key_heads, g.value_heads, g.head_dim);
    let column = |from: usize, width: usize| -> Vec<f64> {
        (first..first + count)
            .flat_map(|r| decode(&qkv[r * stride + from..r * stride + from + width]))
            .collect()
    };
    let gate = |values: &[u16]| -> Vec<f64> {
        (first..first + count)
            .flat_map(|r| decode(&values[r * g.gate_stride..r * g.gate_stride + vh]))
            .collect()
    };
    let (q, k, v) = (
        column(0, kh * d),
        column(kh * d, kh * d),
        column(2 * kh * d, vh * d),
    );
    let (a, b) = (gate(&gates.a), gate(&gates.b));
    let a_log = f64s(&gates.a_log);
    let dt_bias = f64s(&gates.dt_bias);
    recurrent_oracle::gated_delta_rule(
        query,
        GatedDeltaInputs {
            q: &q,
            k: &k,
            v: &v,
            a: &a,
            b: &b,
            a_log: &a_log,
            dt_bias: &dt_bias,
            state,
        },
    )
    .map_err(|e| e.to_string())
}

fn f64s(values: &[f32]) -> Vec<f64> {
    values.iter().map(|&x| f64::from(x)).collect()
}

fn unchanged_f32_slots(before: &[f32], after: &[f32], touched: &[i32]) -> LabResult<()> {
    let g = geometry();
    let n = g.delta_state;
    for slot in 0..before.len() / n {
        if !touched.contains(&(slot as i32))
            && before[slot * n..(slot + 1) * n]
                .iter()
                .zip(&after[slot * n..(slot + 1) * n])
                .any(|(x, y)| x.to_bits() != y.to_bits())
        {
            return Err(format!("untouched slot {slot} changed"));
        }
    }
    Ok(())
}

/// Runs prefill prepare and scan; returns (output bits, final pool).
pub fn delta_prefill_run(
    session: &Session,
    module: &Module,
    data: &DeltaCase,
    lengths: &[usize],
    initial: &[bool],
    slot_ids: &[i32],
) -> LabResult<(Vec<u16>, Vec<f32>)> {
    delta_prefill_anchored(
        session,
        module,
        data,
        lengths,
        initial,
        slot_ids,
        &vec![-1; initial.len()],
    )
}

/// `delta_prefill_run` whose sequence `s` also writes its anchor state into
/// pool entry `anchors[s]` when that is at least zero.
fn delta_prefill_anchored(
    session: &Session,
    module: &Module,
    data: &DeltaCase,
    lengths: &[usize],
    initial: &[bool],
    slot_ids: &[i32],
    anchors: &[i32],
) -> LabResult<(Vec<u16>, Vec<f32>)> {
    let g = geometry();
    let rows = lengths.iter().sum::<usize>();
    let mut op = data.operands(session, rows)?;
    op.i32(session, "cu_seqlens", &cu(lengths))?;
    op.i32(session, "slot_ids", slot_ids)?;
    op.i32(
        session,
        "initial",
        &initial.iter().map(|&x| i32::from(x)).collect::<Vec<_>>(),
    )?;
    op.i32(session, "anchor_ids", anchors)?;
    op.u32(session, "live_sequences", lengths.len() as u32)?;
    // A bucket exactly the size of the batch, the smallest valid plan.
    let live = (rows as u32, lengths.len() as u32);
    let prepare =
        Kernel::load(module, &symbol("delta_prefill_prepare"))?.with_bucket(live.0, live.1);
    let scan = Kernel::load(module, &symbol("delta_prefill_scan"))?.with_bucket(live.0, live.1);
    op.zero(session, "workspace", prepare.workspace_bytes())?;
    prepare.launch(session, &session.stream, &op, live.0, live.1, true)?;
    scan.launch(session, &session.stream, &op, live.0, live.1, true)?;
    Ok((
        op.read_bf16(session, "output", rows * g.norm_width)?,
        op.read_f32(session, "state", data.pool.len())?,
    ))
}

fn delta_prefill(
    session: &Session,
    module: &Module,
    lengths: &[usize],
    initial: &[bool],
    seed: u64,
) -> LabResult<()> {
    let case = format!("delta_prefill_{lengths:?}_{initial:?}");
    delta_prefill_named(session, module, case, lengths, initial, seed)
}

fn delta_prefill_named(
    session: &Session,
    module: &Module,
    case: String,
    lengths: &[usize],
    initial: &[bool],
    seed: u64,
) -> LabResult<()> {
    let g = geometry();
    let mut report = Report::new(case);
    let rows = lengths.iter().sum::<usize>();
    let mut rng = Rng::new(seed);
    let (slot_ids, pool) = slots(lengths.len(), 2);
    let data = DeltaCase::new(&mut rng, rows, pool);
    let (output, pool_after) =
        delta_prefill_run(session, module, &data, lengths, initial, &slot_ids)?;
    let query = phase_query(&symbol("delta_prefill_prepare"));
    let cu = cu(lengths);
    let n = g.delta_state;
    for (s, &len) in lengths.iter().enumerate() {
        let start = cu[s] as usize;
        let state = if initial[s] {
            data.slot(slot_ids[s])
        } else {
            vec![0.0; n]
        };
        let expected = data.oracle(query, start, len, &state)?;
        report.check(
            &format!("seq{s}_output"),
            &decode(&output[start * g.norm_width..(start + len) * g.norm_width]),
            &expected.output,
            DType::Bf16,
        );
        let slot = slot_ids[s] as usize;
        report.check(
            &format!("seq{s}_state"),
            &f64s(&pool_after[slot * n..(slot + 1) * n]),
            &expected.state,
            DType::F32,
        );
    }
    report.exact(
        "untouched slots",
        unchanged_f32_slots(&data.pool, &pool_after, &slot_ids),
    );
    report.finish(session)
}

fn delta_decode(session: &Session, module: &Module, batch: usize, seed: u64) -> LabResult<()> {
    let g = geometry();
    let case = format!("delta_decode_b{batch}");
    let mut report = Report::new(case.clone());
    let mut rng = Rng::new(seed);
    let (slot_ids, pool) = slots(batch, 3);
    let data = DeltaCase::new(&mut rng, batch, pool);
    let mut op = data.operands(session, batch)?;
    op.i32(session, "slot_ids", &slot_ids)?;
    op.u32(session, "live_sequences", batch as u32)?;
    let kernel = Kernel::load(module, &symbol("delta_decode"))?;
    kernel.launch(
        session,
        &session.stream,
        &op,
        batch as u32,
        batch as u32,
        true,
    )?;
    let output = op.read_bf16(session, "output", batch * g.norm_width)?;
    let pool_after = op.read_f32(session, "state", data.pool.len())?;
    let query = phase_query(&symbol("delta_decode"));
    let n = g.delta_state;
    for s in 0..batch {
        let expected = data.oracle(query, s, 1, &data.slot(slot_ids[s]))?;
        report.check(
            &format!("seq{s}_output"),
            &decode(&output[s * g.norm_width..(s + 1) * g.norm_width]),
            &expected.output,
            DType::Bf16,
        );
        let slot = slot_ids[s] as usize;
        report.check(
            &format!("seq{s}_state"),
            &f64s(&pool_after[slot * n..(slot + 1) * n]),
            &expected.state,
            DType::F32,
        );
    }
    report.exact(
        "untouched slots",
        unchanged_f32_slots(&data.pool, &pool_after, &slot_ids),
    );
    report.finish(session)
}

fn delta_verify(session: &Session, module: &Module, sequences: usize, seed: u64) -> LabResult<()> {
    let mut rng = Rng::new(seed);
    let layout = VerifyLayout::verify(sequences, &mut rng);
    let case = format!("delta_verify_s{sequences}");
    delta_verify_run(session, module, case, &mut rng, &layout)
}

fn delta_verify_run(
    session: &Session,
    module: &Module,
    case: String,
    rng: &mut Rng,
    layout: &VerifyLayout,
) -> LabResult<()> {
    let g = geometry();
    let mut report = Report::new(case);
    let max = verify_row_limit() as usize;
    let (lengths, slot_ids, checkpoint_ids) =
        (&layout.lengths, &layout.slot_ids, &layout.checkpoint_ids);
    let rows = lengths.iter().sum::<usize>();
    let data = DeltaCase::new(rng, rows, layout.pool);
    let mut op = data.operands(session, rows)?;
    let cu = cu(lengths);
    let live_rows = cu[layout.live] as usize;
    op.i32(session, "cu_rows", &cu)?;
    op.i32(session, "slot_ids", slot_ids)?;
    op.i32(session, "checkpoint_ids", checkpoint_ids)?;
    op.u32(session, "live_sequences", layout.live as u32)?;
    poison::poison(session, &[&op.buffers["output"]])?;
    let kernel = layout.kernel(module, "delta_verify")?;
    kernel.launch(
        session,
        &session.stream,
        &op,
        live_rows as u32,
        layout.live as u32,
        true,
    )?;
    let output = op.read_bf16(session, "output", rows * g.norm_width)?;
    let pool_after = op.read_f32(session, "state", data.pool.len())?;
    let query = phase_query(&symbol("delta_verify"));
    let n = g.delta_state;
    let mut touched = Vec::new();
    for (s, &len) in lengths[..layout.live].iter().enumerate() {
        let start = cu[s] as usize;
        let expected = data.oracle(query, start, len, &data.slot(slot_ids[s]))?;
        report.check(
            &format!("seq{s}_output"),
            &decode(&output[start * g.norm_width..(start + len) * g.norm_width]),
            &expected.output,
            DType::Bf16,
        );
        for (r, state) in expected.row_states.iter().enumerate() {
            let slot = checkpoint_ids[s * max + r] as usize;
            touched.push(slot as i32);
            report.check(
                &format!("seq{s}_row{r}_state"),
                &f64s(&pool_after[slot * n..(slot + 1) * n]),
                state,
                DType::F32,
            );
        }
    }
    report.exact(
        "untouched slots",
        unchanged_f32_slots(&data.pool, &pool_after, &touched),
    );
    report.exact(
        "rows past the live sequences",
        poisoned_tail(&output, live_rows * g.norm_width),
    );
    compact_pool_bits(
        session,
        &kernel,
        &mut op,
        layout,
        &bytes_f32(&data.pool),
        &bytes_f32(&pool_after),
        &bytes_u16(&output),
        n * 4,
        DType::F32,
        &mut report,
    )?;
    report.finish(session)
}

// -------------------------------------------------------- gated norm ----

pub struct NormCase {
    geometry: Geometry,
    pub input: Vec<u16>,
    pub gate: Vec<u16>,
    pub weight: Vec<u16>,
}

impl NormCase {
    pub fn new(rng: &mut Rng, rows: usize) -> Self {
        Self::with_geometry(rng, rows, geometry())
    }

    fn with_geometry(rng: &mut Rng, rows: usize, g: Geometry) -> Self {
        Self {
            geometry: g,
            input: rng.bf16(rows * g.norm_width, 0.5),
            gate: rng.bf16(rows * g.z_stride, 2.0),
            weight: (0..g.head_dim)
                .map(|_| f32_to_bf16((0.875 + 0.15 * rng.uniform()) as f32))
                .collect(),
        }
    }
    pub fn operands(&self, session: &Session, rows: usize) -> LabResult<Operands> {
        let g = self.geometry;
        let mut op = Operands::new();
        op.bf16(session, "input", &self.input)?;
        op.bf16(session, "gate", &self.gate)?;
        op.bf16(session, "weight", &self.weight)?;
        op.zero(session, "output", rows * g.norm_width * 2)?;
        op.u32(session, "live_rows", rows as u32)?;
        op.scalars.insert("gate_stride", g.z_stride as u32);
        Ok(op)
    }
}

fn gated_norm(session: &Session, module: &Module, rows: usize, seed: u64) -> LabResult<()> {
    let g = geometry();
    let case = format!("gated_norm_r{rows}");
    let mut report = Report::new(case.clone());
    let mut rng = Rng::new(seed);
    let data = NormCase::new(&mut rng, rows);
    let op = data.operands(session, rows)?;
    let kernel = Kernel::load(module, &symbol("gated_norm"))?;
    kernel.launch(
        session,
        &session.stream,
        &op,
        rows as u32,
        rows as u32,
        true,
    )?;
    let output = op.read_bf16(session, "output", rows * g.norm_width)?;
    let gate = (0..rows)
        .flat_map(|r| decode(&data.gate[r * g.z_stride..r * g.z_stride + g.norm_width]))
        .collect::<Vec<_>>();
    let expected = recurrent_oracle::gated_rmsnorm(
        phase_query(&symbol("gated_norm")),
        &decode(&data.input),
        &gate,
        &decode(&data.weight),
    )
    .map_err(|e| e.to_string())?;
    report.check("output", &decode(&output), &expected, DType::Bf16);
    report.finish(session)
}

// ----------------------------------------------------- chains and carry ----

/// A state pool of a chain: its buffer, starting bytes, the entries the
/// chain writes without reading them first (poisoned before every run), and
/// the entries it reads and rewrites.
struct Pool {
    buffer: DeviceBuffer,
    initial: Vec<u8>,
    write_only: WriteMap,
    updated: Vec<Range<usize>>,
}

/// A layer's linear-attention kernels over shared buffers. Every run first
/// restores the state pools, so eager, PDL-free and graph runs start from the
/// same bits and must end with the same bits.
struct Chain {
    stages: Vec<(Kernel, Operands, u32, u32)>,
    pools: Vec<Pool>,
    /// Stage outputs, each written in full and read by the next stage.
    results: Vec<(&'static str, DeviceBuffer, usize)>,
    /// Buffers one stage writes and a later one reads within a run.
    scratch: Vec<(&'static str, DeviceBuffer)>,
}

impl Chain {
    fn launch(&self, session: &Session, stream: &Stream, pdl: bool) -> LabResult<()> {
        for (kernel, operands, rows, sequences) in &self.stages {
            kernel.launch(session, stream, operands, *rows, *sequences, pdl)?;
        }
        Ok(())
    }

    fn read(&self, session: &Session) -> LabResult<Vec<Vec<u8>>> {
        self.results
            .iter()
            .map(|(_, buffer, bytes)| {
                let mut out = vec![0; *bytes];
                buffer
                    .copy_to_bytes(&mut out, &session.stream)
                    .map_err(|e| e.to_string())?;
                Ok(out)
            })
            .chain(self.pools.iter().map(|pool| {
                let mut out = vec![0; pool.initial.len()];
                pool.buffer
                    .copy_to_bytes(&mut out, &session.stream)
                    .map_err(|e| e.to_string())?;
                Ok(out)
            }))
            .collect()
    }

    /// Eager (PDL) run twice, a PDL-free run and two graph replays, each
    /// on poisoned stage outputs and scratch and from the starting pools
    /// with their write-only entries poisoned; returns the eager results.
    fn replay_checks(&self, session: &Session, report: &mut Report) -> LabResult<Vec<Vec<u8>>> {
        let mut checks = Replay::new(session);
        for (name, buffer, bytes) in &self.results {
            checks = checks.output(
                *name,
                buffer,
                WriteMap::of(DType::Bf16).byte_prefix(*bytes)?,
            );
        }
        for (name, buffer) in &self.scratch {
            checks = checks.scratch(*name, buffer);
        }
        for (index, pool) in self.pools.iter().enumerate() {
            checks = checks.state(
                format!("state pool {index}"),
                &pool.buffer,
                pool.initial.clone(),
                pool.write_only.clone(),
                pool.updated.clone(),
            );
        }
        checks.eager(|| self.launch(session, &session.stream, true))?;
        let eager = self.read(session)?;
        report.exact(
            "deterministic rerun",
            checks.relaunch(|| self.launch(session, &session.stream, true)),
        );
        report.exact(
            "PDL off gives the same bits",
            checks.relaunch(|| self.launch(session, &session.stream, false)),
        );
        let graph = session.capture(|stream| self.launch(session, stream, true))?;
        for replay in 0..2 {
            report.exact(&format!("graph replay {replay}"), checks.graph(&graph));
        }
        Ok(eager)
    }
}

fn u16s(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_ne_bytes([c[0], c[1]]))
        .collect()
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[derive(Clone, Copy, PartialEq)]
enum ChainPhase {
    Prefill,
    Decode,
    Verify,
}

/// conv -> gated delta rule -> gated norm for one phase, with every stage
/// checked against its oracle on the previous stage's GPU output, plus the
/// graph, determinism and PDL checks.
fn layer_chain(
    session: &Session,
    module: &Module,
    phase: ChainPhase,
    lengths: &[usize],
    seed: u64,
) -> LabResult<()> {
    let g = geometry();
    let name = match phase {
        ChainPhase::Prefill => "prefill",
        ChainPhase::Decode => "decode",
        ChainPhase::Verify => "verify",
    };
    let mut report = Report::new(format!("layer_chain_{name}_{lengths:?}"));
    let c = g.channels;
    let max = verify_row_limit() as usize;
    let rows = lengths.iter().sum::<usize>();
    let sequences = lengths.len();
    let mut rng = Rng::new(seed);
    let (slot_ids, base) = slots(sequences, 1);
    let (checkpoint_ids, pool) = if phase == ChainPhase::Verify {
        checkpoints(&slot_ids, base)
    } else {
        (Vec::new(), base)
    };
    let initial = (0..sequences).map(|s| s % 3 != 1).collect::<Vec<_>>();
    let conv = ConvCase::new(&mut rng, rows, pool);
    let delta = DeltaCase::new(&mut rng, rows, pool);
    let norm = NormCase::new(&mut rng, rows);
    let cu = cu(lengths);
    let mut conv_op = conv.operands(session, rows)?;
    let conv_out = conv_op.buffers["output"].clone();
    let mut delta_op = delta.operands(session, rows)?;
    delta_op.buffers.insert("qkv", conv_out.clone());
    delta_op.scalars.insert("qkv_stride", g.channels as u32);
    let delta_out = delta_op.buffers["output"].clone();
    let mut norm_op = norm.operands(session, rows)?;
    norm_op.buffers.insert("input", delta_out.clone());
    let norm_out = norm_op.buffers["output"].clone();
    for op in [&mut conv_op, &mut delta_op] {
        op.i32(session, "slot_ids", &slot_ids)?;
        op.u32(session, "live_sequences", sequences as u32)?;
        match phase {
            ChainPhase::Prefill => {
                op.i32(session, "cu_seqlens", &cu)?;
                op.i32(
                    session,
                    "initial",
                    &initial.iter().map(|&x| i32::from(x)).collect::<Vec<_>>(),
                )?;
                op.i32(session, "anchor_ids", &vec![-1; initial.len()])?;
            }
            ChainPhase::Decode => {}
            ChainPhase::Verify => {
                op.i32(session, "cu_rows", &cu)?;
                op.i32(session, "checkpoint_ids", &checkpoint_ids)?;
            }
        }
    }
    let live = (rows as u32, sequences as u32);
    let mut stages = Vec::new();
    let conv_kernel = Kernel::load(module, &symbol(&format!("conv_{name}")))?;
    stages.push((conv_kernel, conv_op, live.0, live.1));
    if phase == ChainPhase::Prefill {
        let prepare = Kernel::load(module, &symbol("delta_prefill_prepare"))?;
        delta_op.zero(session, "workspace", prepare.workspace_bytes())?;
        let scan = Kernel::load(module, &symbol("delta_prefill_scan"))?;
        let scan_op = Operands {
            buffers: delta_op.buffers.clone(),
            scalars: delta_op.scalars.clone(),
        };
        stages.push((prepare, delta_op, live.0, live.1));
        stages.push((scan, scan_op, live.0, live.1));
    } else {
        let kernel = Kernel::load(module, &symbol(&format!("delta_{name}")))?;
        stages.push((kernel, delta_op, live.0, live.1));
    }
    stages.push((
        Kernel::load(module, &symbol("gated_norm"))?,
        norm_op,
        live.0,
        live.0,
    ));
    let conv_pool = stages[0].1.buffers["state"].clone();
    let delta_pool = stages[1].1.buffers["state"].clone();
    let scratch = match phase {
        ChainPhase::Prefill => vec![(
            "delta prefill workspace",
            stages[1].1.buffers["workspace"].clone(),
        )],
        _ => Vec::new(),
    };
    // Decode advances each slot in place; verify leaves the source slots and
    // writes one checkpoint per row; prefill writes the final state to the
    // slot, reading it first only for sequences with an initial state.
    let (mut written, mut updated) = (Vec::new(), Vec::new());
    for (s, &slot) in slot_ids.iter().enumerate() {
        match phase {
            ChainPhase::Decode => updated.push(slot as usize),
            ChainPhase::Verify => {
                written.extend((0..lengths[s]).map(|r| checkpoint_ids[s * max + r] as usize))
            }
            ChainPhase::Prefill if initial[s] => updated.push(slot as usize),
            ChainPhase::Prefill => written.push(slot as usize),
        }
    }
    let pool_state = |entry: usize, element: DType| -> LabResult<(WriteMap, Vec<Range<usize>>)> {
        Ok((
            WriteMap::of(element)
                .byte_ranges(written.iter().map(|&e| e * entry..(e + 1) * entry))?,
            updated
                .iter()
                .map(|&e| e * entry..(e + 1) * entry)
                .collect(),
        ))
    };
    let (conv_written, conv_updated) = pool_state(HISTORY * c * 2, DType::Bf16)?;
    let (delta_written, delta_updated) = pool_state(g.delta_state * 4, DType::F32)?;
    let chain = Chain {
        stages,
        pools: vec![
            Pool {
                buffer: conv_pool,
                initial: bytes_u16(&conv.pool),
                write_only: conv_written,
                updated: conv_updated,
            },
            Pool {
                buffer: delta_pool,
                initial: bytes_f32(&delta.pool),
                write_only: delta_written,
                updated: delta_updated,
            },
        ],
        results: vec![
            ("conv output", conv_out, rows * c * 2),
            ("delta output", delta_out, rows * g.norm_width * 2),
            ("norm output", norm_out, rows * g.norm_width * 2),
        ],
        scratch,
    };
    let eager = chain.replay_checks(session, &mut report)?;
    // Oracles on the exact inputs each stage saw.
    let (conv_bits, delta_bits, norm_bits) = (u16s(&eager[0]), u16s(&eager[1]), u16s(&eager[2]));
    let (conv_pool_after, delta_pool_after) = (u16s(&eager[3]), f32s(&eager[4]));
    let conv_query = phase_query(&symbol(&format!("conv_{name}")));
    let delta_query = phase_query(&symbol(&if phase == ChainPhase::Prefill {
        "delta_prefill_prepare".to_owned()
    } else {
        format!("delta_{name}")
    }));
    let weight = decode(&conv.weight);
    let n_conv = HISTORY * c;
    let n_delta = g.delta_state;
    for (s, &len) in lengths.iter().enumerate() {
        let start = cu[s] as usize;
        let carry = phase != ChainPhase::Prefill || initial[s];
        let conv_state = if carry {
            conv.slot(slot_ids[s])
        } else {
            vec![0.0; n_conv]
        };
        let expected = recurrent_oracle::causal_conv1d(
            conv_query,
            &conv.rows(start, len),
            &weight,
            &conv_state,
        )
        .map_err(|e| e.to_string())?;
        report.check(
            &format!("seq{s} conv output"),
            &decode(&conv_bits[start * c..(start + len) * c]),
            &expected.output,
            DType::Bf16,
        );
        let delta_state = if carry {
            delta.slot(slot_ids[s])
        } else {
            vec![0.0; n_delta]
        };
        let expected = delta_oracle(delta_query, &conv_bits, c, &delta, start, len, &delta_state)?;
        report.check(
            &format!("seq{s} delta output"),
            &decode(&delta_bits[start * g.norm_width..(start + len) * g.norm_width]),
            &expected.output,
            DType::Bf16,
        );
        if phase == ChainPhase::Verify {
            for (r, state) in expected.row_states.iter().enumerate() {
                let slot = checkpoint_ids[s * max + r] as usize;
                report.check(
                    &format!("seq{s} row{r} delta checkpoint"),
                    &f64s(&delta_pool_after[slot * n_delta..(slot + 1) * n_delta]),
                    state,
                    DType::F32,
                );
            }
        } else {
            let slot = slot_ids[s] as usize;
            report.check(
                &format!("seq{s} delta state"),
                &f64s(&delta_pool_after[slot * n_delta..(slot + 1) * n_delta]),
                &expected.state,
                DType::F32,
            );
            let conv_expected = recurrent_oracle::causal_conv1d(
                conv_query,
                &conv.rows(start, len),
                &weight,
                &conv_state,
            )
            .map_err(|e| e.to_string())?;
            report.exact(
                &format!("seq{s} conv state bits"),
                compare_state_bits(
                    &conv_pool_after[slot * n_conv..(slot + 1) * n_conv],
                    &conv_expected.state,
                ),
            );
        }
    }
    let gate = (0..rows)
        .flat_map(|r| decode(&norm.gate[r * g.z_stride..r * g.z_stride + g.norm_width]))
        .collect::<Vec<_>>();
    let expected = recurrent_oracle::gated_rmsnorm(
        phase_query(&symbol("gated_norm")),
        &decode(&delta_bits),
        &gate,
        &decode(&norm.weight),
    )
    .map_err(|e| e.to_string())?;
    report.check("norm output", &decode(&norm_bits), &expected, DType::Bf16);
    report.finish(session)
}

/// Conv output, conv pool, delta output and delta state of one run.
type CarryResult = (Vec<u16>, Vec<u16>, Vec<u16>, Vec<f32>);

/// Two prefills that carry the state across a split point equal one prefill
/// of the whole sequence: bitwise for the convolution, and within the gate of
/// the whole sequence's oracle for the delta rule.
fn prefill_carry(session: &Session, module: &Module, rows: usize, seed: u64) -> LabResult<()> {
    let g = geometry();
    let mut report = Report::new(format!("prefill_carry_{rows}"));
    let c = g.channels;
    let mut rng = Rng::new(seed);
    let conv = ConvCase::new(&mut rng, rows, 1);
    let delta = DeltaCase::new(&mut rng, rows, 1);
    let conv_kernel = Kernel::load(module, &symbol("conv_prefill"))?;
    let prepare = Kernel::load(module, &symbol("delta_prefill_prepare"))?;
    let scan = Kernel::load(module, &symbol("delta_prefill_scan"))?;
    let n_delta = g.delta_state;
    let whole = delta.oracle(
        phase_query(&symbol("delta_prefill_prepare")),
        0,
        rows,
        &delta.slot(0),
    )?;
    let mut reference: Option<CarryResult> = None;
    for split in [0, 1, 64, 100, rows - 1] {
        let mut conv_op = conv.operands(session, rows)?;
        let mut delta_op = delta.operands(session, rows)?;
        delta_op.zero(session, "workspace", prepare.workspace_bytes())?;
        let calls: Vec<[i32; 2]> = if split == 0 {
            vec![[0, rows as i32]]
        } else {
            vec![[0, split as i32], [split as i32, rows as i32]]
        };
        for op in [&mut conv_op, &mut delta_op] {
            op.i32(session, "slot_ids", &[0])?;
            op.i32(session, "initial", &[1])?;
            op.i32(session, "anchor_ids", &[-1])?;
            op.u32(session, "live_sequences", 1)?;
        }
        for call in &calls {
            let span = (call[1] - call[0]) as u32;
            for op in [&mut conv_op, &mut delta_op] {
                op.i32(session, "cu_seqlens", call)?;
            }
            conv_kernel.launch(session, &session.stream, &conv_op, span, 1, true)?;
            prepare.launch(session, &session.stream, &delta_op, span, 1, true)?;
            scan.launch(session, &session.stream, &delta_op, span, 1, true)?;
        }
        let conv_out = conv_op.read_bf16(session, "output", rows * c)?;
        let conv_state = conv_op.read_bf16(session, "state", conv.pool.len())?;
        let delta_out = delta_op.read_bf16(session, "output", rows * g.norm_width)?;
        let delta_state = delta_op.read_f32(session, "state", n_delta)?;
        report.check(
            &format!("split {split} delta output"),
            &decode(&delta_out),
            &whole.output,
            DType::Bf16,
        );
        report.check(
            &format!("split {split} delta state"),
            &f64s(&delta_state),
            &whole.state,
            DType::F32,
        );
        match &reference {
            None => reference = Some((conv_out, conv_state, delta_out, delta_state)),
            Some((out, state, delta_whole, delta_state_whole)) => {
                report.exact(
                    &format!("split {split} conv output bits"),
                    if *out == conv_out {
                        Ok(())
                    } else {
                        Err("differs from one prefill".into())
                    },
                );
                report.exact(
                    &format!("split {split} conv state bits"),
                    if *state == conv_state {
                        Ok(())
                    } else {
                        Err("differs from one prefill".into())
                    },
                );
                let difference = decode(delta_whole)
                    .iter()
                    .zip(decode(&delta_out))
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0, f64::max);
                let state_difference = delta_state_whole
                    .iter()
                    .zip(&delta_state)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0_f32, f32::max);
                report.checks.push(json!({
                    "name": format!("split {split} against one prefill"),
                    "max_output_difference": difference,
                    "max_state_difference": state_difference,
                }));
            }
        }
    }
    report.finish(session)
}

// ------------------------------------------------------ high state ids ----

/// The linear-state launches of the high-id case.
#[derive(Clone, Copy, Debug)]
enum StateLaunch {
    ConvDecode,
    ConvVerify,
    ConvPrefill,
    DeltaDecode,
    DeltaVerify,
    DeltaPrefill,
}

impl StateLaunch {
    fn conv(self) -> bool {
        matches!(
            self,
            Self::ConvDecode | Self::ConvVerify | Self::ConvPrefill
        )
    }
    fn verify(self) -> bool {
        matches!(self, Self::ConvVerify | Self::DeltaVerify)
    }
    fn prefill(self) -> bool {
        matches!(self, Self::ConvPrefill | Self::DeltaPrefill)
    }
    fn name(self) -> &'static str {
        match self {
            Self::ConvDecode => "conv_decode",
            Self::ConvVerify => "conv_verify",
            Self::ConvPrefill => "conv_prefill",
            Self::DeltaDecode => "delta_decode",
            Self::DeltaVerify => "delta_verify",
            Self::DeltaPrefill => "delta_prefill",
        }
    }
    /// Rows of each of four sequences, and which prefill sequences start
    /// from their slot.
    fn lengths(self) -> (Vec<usize>, Vec<bool>) {
        if self.verify() {
            (vec![8, 3, 8, 5], vec![true; 4])
        } else if self.prefill() {
            (vec![70, 1, 129, 64], vec![true, false, true, false])
        } else {
            (vec![1; 4], vec![true; 4])
        }
    }
    /// State entries: each sequence's slot, then its verify checkpoints.
    fn entries(self) -> usize {
        4 + if self.verify() {
            4 * verify_row_limit() as usize
        } else {
            0
        }
    }
    /// Bytes of one state entry and of its element.
    fn entry_bytes(self) -> (u64, u64) {
        let g = geometry();
        if self.conv() {
            ((HISTORY * g.channels * 2) as u64, 2)
        } else {
            ((g.delta_state * 4) as u64, 4)
        }
    }
}

/// A high-id case's inputs and dense starting pool.
enum StateData {
    Conv(ConvCase),
    Delta(DeltaCase),
}

impl StateData {
    fn new(launch: StateLaunch, rng: &mut Rng, rows: usize) -> Self {
        if launch.conv() {
            Self::Conv(ConvCase::new(rng, rows, launch.entries()))
        } else {
            Self::Delta(DeltaCase::new(rng, rows, launch.entries()))
        }
    }
    fn pool_bytes(&self) -> Vec<u8> {
        match self {
            Self::Conv(data) => bytes_u16(&data.pool),
            Self::Delta(data) => bytes_f32(&data.pool),
        }
    }
    fn operands(&self, session: &Session, rows: usize) -> LabResult<Operands> {
        match self {
            Self::Conv(data) => data.operands(session, rows),
            Self::Delta(data) => data.operands(session, rows),
        }
    }
}

/// Launches `launch` with its state entries at `ids` in `state`. Returns
/// the output bits.
fn launch_state(
    session: &Session,
    module: &Module,
    launch: StateLaunch,
    data: &StateData,
    ids: &[u64],
    state: &DeviceBuffer,
) -> LabResult<Vec<u16>> {
    let g = geometry();
    let (lengths, initial) = launch.lengths();
    let (rows, sequences) = (lengths.iter().sum::<usize>(), lengths.len());
    let live = (rows as u32, sequences as u32);
    let mut op = data.operands(session, rows)?;
    op.buffers.insert("state", state.clone());
    let ids = pool_offsets::i32_ids(ids);
    op.i32(session, "slot_ids", &ids[..sequences])?;
    op.u32(session, "live_sequences", sequences as u32)?;
    let kernel = |name: &str| Kernel::load(module, &symbol(name));
    if launch.verify() {
        op.i32(session, "cu_rows", &cu(&lengths))?;
        op.i32(session, "checkpoint_ids", &ids[sequences..])?;
        kernel(launch.name())?.launch(session, &session.stream, &op, live.0, live.1, true)?;
    } else if launch.prefill() {
        op.i32(session, "cu_seqlens", &cu(&lengths))?;
        op.i32(
            session,
            "initial",
            &initial.iter().map(|&x| i32::from(x)).collect::<Vec<_>>(),
        )?;
        op.i32(session, "anchor_ids", &vec![-1; initial.len()])?;
        if launch.conv() {
            kernel("conv_prefill")?.launch(session, &session.stream, &op, live.0, live.1, true)?;
        } else {
            let prepare = kernel("delta_prefill_prepare")?.with_bucket(live.0, live.1);
            let scan = kernel("delta_prefill_scan")?.with_bucket(live.0, live.1);
            op.zero(session, "workspace", prepare.workspace_bytes())?;
            prepare.launch(session, &session.stream, &op, live.0, live.1, true)?;
            scan.launch(session, &session.stream, &op, live.0, live.1, true)?;
        }
    } else {
        kernel(launch.name())?.launch(session, &session.stream, &op, live.0, live.1, true)?;
    }
    let width = if launch.conv() {
        g.channels
    } else {
        g.norm_width
    };
    op.read_bf16(session, "output", rows * width)
}

/// The low run against the recurrent oracles, as the existing cases check
/// it: each sequence's output, then its slot's state or its checkpoints'
/// row states in `after`, the pool with entry `i` at id `i`.
fn check_state_low(
    session: &Session,
    launch: StateLaunch,
    data: &StateData,
    output: &[u16],
    after: &[u8],
) -> LabResult<()> {
    let g = geometry();
    let (lengths, initial) = launch.lengths();
    let sequences = lengths.len();
    let cu = cu(&lengths);
    let query = phase_query(&symbol(if launch.prefill() && !launch.conv() {
        "delta_prefill_prepare"
    } else {
        launch.name()
    }));
    let checkpoint = |s: usize, r: usize| sequences + s * verify_row_limit() as usize + r;
    let mut report = Report::new(format!("{}_high_ids_low_run", launch.name()));
    match data {
        StateData::Conv(data) => {
            let (c, n) = (g.channels, HISTORY * g.channels);
            let pool_after: Vec<u16> = after
                .chunks_exact(2)
                .map(|x| u16::from_ne_bytes([x[0], x[1]]))
                .collect();
            let weight = decode(&data.weight);
            for (s, &len) in lengths.iter().enumerate() {
                let start = cu[s] as usize;
                let state = if initial[s] {
                    data.slot(s as i32)
                } else {
                    vec![0.0; n]
                };
                let expected =
                    recurrent_oracle::causal_conv1d(query, &data.rows(start, len), &weight, &state)
                        .map_err(|e| e.to_string())?;
                report.check(
                    &format!("seq{s}"),
                    &decode(&output[start * c..(start + len) * c]),
                    &expected.output,
                    DType::Bf16,
                );
                if launch.verify() {
                    for (r, state) in expected.row_states.iter().enumerate() {
                        let e = checkpoint(s, r);
                        report.exact(
                            "checkpoint bits",
                            compare_state_bits(&pool_after[e * n..(e + 1) * n], state),
                        );
                    }
                } else {
                    report.exact(
                        "state bits",
                        compare_state_bits(&pool_after[s * n..(s + 1) * n], &expected.state),
                    );
                }
            }
        }
        StateData::Delta(data) => {
            let (w, n) = (g.norm_width, g.delta_state);
            let pool_after: Vec<f32> = after
                .chunks_exact(4)
                .map(|x| f32::from_ne_bytes([x[0], x[1], x[2], x[3]]))
                .collect();
            for (s, &len) in lengths.iter().enumerate() {
                let start = cu[s] as usize;
                let state = if initial[s] {
                    data.slot(s as i32)
                } else {
                    vec![0.0; n]
                };
                let expected = data.oracle(query, start, len, &state)?;
                report.check(
                    &format!("seq{s}_output"),
                    &decode(&output[start * w..(start + len) * w]),
                    &expected.output,
                    DType::Bf16,
                );
                if launch.verify() {
                    for (r, state) in expected.row_states.iter().enumerate() {
                        let e = checkpoint(s, r);
                        report.check(
                            &format!("seq{s}_row{r}_state"),
                            &f64s(&pool_after[e * n..(e + 1) * n]),
                            state,
                            DType::F32,
                        );
                    }
                } else {
                    report.check(
                        &format!("seq{s}_state"),
                        &f64s(&pool_after[s * n..(s + 1) * n]),
                        &expected.state,
                        DType::F32,
                    );
                }
            }
        }
    }
    report.finish(session)
}

/// Linear-attention state past the 32-bit lines (`pool_offsets`): conv and
/// delta-rule decode, verify and prefill with their slot and checkpoint ids
/// alternating between group A and group B of the topology's entry size,
/// prefill both with and without an initial state. The low run meets the
/// recurrent oracles; the high run's output and every state entry equal it
/// bit for bit, and no other pool byte changes.
#[test]
fn linear_high_ids_match_low_ids() {
    gpu_test(pool_offsets::need_bytes(18 << 30), |session| {
        let module = module(session)?;
        for (index, launch) in [
            StateLaunch::ConvDecode,
            StateLaunch::ConvVerify,
            StateLaunch::ConvPrefill,
            StateLaunch::DeltaDecode,
            StateLaunch::DeltaVerify,
            StateLaunch::DeltaPrefill,
        ]
        .into_iter()
        .enumerate()
        {
            let name = launch.name();
            let rows = launch.lengths().0.iter().sum::<usize>();
            let entries = launch.entries();
            let data = StateData::new(launch, &mut Rng::new(0x1d00 + index as u64), rows);
            let initial = data.pool_bytes();
            let (entry, element) = launch.entry_bytes();
            let span = entry as usize;
            let low_ids: Vec<u64> = (0..entries as u64).collect();
            let low_state = session.allocate(initial.len())?;
            low_state
                .copy_from_bytes(&initial, &session.stream)
                .map_err(|e| e.to_string())?;
            let ids = pool_offsets::high_ids(entry, element, entries.div_ceil(2));
            let high_ids = ids.alternating(entries);
            let highest = high_ids.iter().copied().max().unwrap_or(0);
            let mut pool = SentinelPool::new(session, (highest + 1) * entry)?;
            for (i, &id) in high_ids.iter().enumerate() {
                pool.put(session, id * entry, &initial[i * span..(i + 1) * span])?;
            }
            let low = launch_state(session, &module, launch, &data, &low_ids, &low_state)?;
            let after = pool_offsets::read(session, &low_state, 0, initial.len() as u64)?;
            check_state_low(session, launch, &data, &low, &after)?;
            let high = launch_state(session, &module, launch, &data, &high_ids, &pool.buffer)?;
            pool_offsets::same(name, "output", &bytes_u16(&low), &bytes_u16(&high))?;
            for (i, &id) in high_ids.iter().enumerate() {
                pool_offsets::same(
                    name,
                    &format!("state entry {id}"),
                    &after[i * span..(i + 1) * span],
                    &pool.read(session, id * entry, entry)?,
                )?;
            }
            let scanned = pool.check_untouched(session, name)?;
            let mut receipt = json!({
                "entries": entries,
                "ids": ids.describe(),
                "highest_offset": highest * entry,
                "outputs_bitwise": true,
                "entries_bitwise": true,
                "sentinel_bytes_checked": scanned,
            });
            let g = geometry();
            if g.ranks != 4 {
                receipt["qwen38_ranks"] = json!(g.ranks);
            }
            pool_offsets::receipt(session, "linear", name, receipt)?;
        }
        Ok(())
    })
    .expect("linear-attention state at high slot and checkpoint ids");
}

// ------------------------------------------------------------- tests ----

#[test]
fn linear_conv_matches_oracle_gpu() {
    gpu_test(2 << 30, |session| {
        let module = module(session)?;
        let lean = lean();
        // Lean: a partial chunk, exactly one chunk and a multi-chunk
        // sequence; the mixed batch below carries the no-initial sequences.
        let singles: &[usize] = if lean {
            &[1, 63, 64, 130]
        } else {
            &[1, 2, 3, 63, 64, 65, 2048, 8192]
        };
        for (i, &len) in singles.iter().enumerate() {
            conv_prefill(session, &module, &[len], &[true], 0x0c00 + i as u64)?;
        }
        if !lean {
            conv_prefill(session, &module, &[65], &[false], 0x0c10)?;
        }
        conv_prefill(
            session,
            &module,
            &[2, 65, 1, 130, 64],
            &[true, false, true, true, false],
            0x0c11,
        )?;
        let (batches, verifies): (&[usize], &[usize]) = if lean {
            (&[1, 8], &[1, 3])
        } else {
            (&[1, 8, 32], &[1, 8, 32])
        };
        for &batch in batches {
            conv_decode(session, &module, batch, 0x0c20 + batch as u64)?;
        }
        for &sequences in verifies {
            conv_verify(session, &module, sequences, 0x0c30 + sequences as u64)?;
        }
        Ok(())
    })
    .expect("linear conv kernels match the recurrent oracle");
}

#[test]
fn linear_gated_norm_matches_oracle_gpu() {
    gpu_test(1 << 30, |session| {
        let module = module(session)?;
        let rows: &[usize] = if lean() { &[1, 7] } else { &[1, 7, 64, 8192] };
        for &rows in rows {
            gated_norm(session, &module, rows, 0x0a00 + rows as u64)?;
        }
        Ok(())
    })
    .expect("gated RMSNorm matches the recurrent oracle");
}

#[test]
fn linear_delta_decode_verify_match_oracle_gpu() {
    gpu_test(2 << 30, |session| {
        let module = module(session)?;
        let (batches, verifies): (&[usize], &[usize]) = if lean() {
            (&[1, 8], &[1, 3])
        } else {
            (&[1, 8, 32], &[1, 3, 8, 32])
        };
        for &batch in batches {
            delta_decode(session, &module, batch, 0x0d00 + batch as u64)?;
        }
        for &sequences in verifies {
            delta_verify(session, &module, sequences, 0x0d10 + sequences as u64)?;
        }
        Ok(())
    })
    .expect("gated delta rule decode and verify match the recurrent oracle");
}

#[test]
fn linear_delta_prefill_matches_oracle_gpu() {
    let lean = lean();
    // Lean: a partial chunk, exactly one chunk and four chunks plus a
    // one-row tail; the mixed batch carries the no-initial sequences.
    let lengths = std::env::var("CHOPINFER_LINEAR_PREFILL_LENGTHS")
        .ok()
        .map(|v| {
            v.split(',')
                .map(|x| x.trim().parse().unwrap())
                .collect::<Vec<usize>>()
        })
        .unwrap_or_else(|| {
            if lean {
                vec![63, 64, 257]
            } else {
                vec![1, 63, 64, 65, 2048, 8192]
            }
        });
    gpu_test(4 << 30, |session| {
        let module = module(session)?;
        for (i, &len) in lengths.iter().enumerate() {
            eprintln!("delta prefill length {len}");
            delta_prefill(session, &module, &[len], &[true], 0x0e00 + i as u64)?;
        }
        if !lean {
            delta_prefill(session, &module, &[70], &[false], 0x0e10)?;
        }
        delta_prefill(
            session,
            &module,
            &[2, 65, 1, 130, 64],
            &[true, false, true, true, false],
            0x0e11,
        )?;
        Ok(())
    })
    .expect("gated delta rule prefill matches the recurrent oracle");
}

#[test]
fn linear_layer_chains_replay_deterministically_gpu() {
    gpu_test(2 << 30, |session| {
        let module = module(session)?;
        if lean() {
            // One small replay: the four-stage prefill chain, one sequence
            // with and one without an initial state.
            return layer_chain(session, &module, ChainPhase::Prefill, &[65, 1], 0x0f02);
        }
        layer_chain(session, &module, ChainPhase::Decode, &[1; 8], 0x0f00)?;
        layer_chain(session, &module, ChainPhase::Verify, &[8, 3, 1, 5], 0x0f01)?;
        layer_chain(
            session,
            &module,
            ChainPhase::Prefill,
            &[100, 37, 64],
            0x0f02,
        )?;
        Ok(())
    })
    .expect("linear-attention layer chains replay bit-exactly and match their oracles");
}

#[test]
fn linear_prefill_state_carry_gpu() {
    gpu_test(2 << 30, |session| {
        let module = module(session)?;
        prefill_carry(session, &module, 200, 0x0f10)
    })
    .expect("split prefills carry the recurrent state like one prefill");
}

/// Live sequences of the wide cases (Qwen above 32): counts that are not a power
/// of two decode in the next power-of-two bucket, as the bundle's graphs do.
const WIDE_SEQUENCES: [usize; 6] = [33, 64, 100, 128, 255, 256];

/// Rows of the wide prefill sequences, cycled: mostly short, with sequences
/// of 65, 70 and 129 rows whose chunks span, packed off chunk boundaries.
/// 256 sequences take 7,856 rows.
const WIDE_PREFILL_ROWS: [usize; 16] = [129, 1, 17, 64, 3, 65, 9, 2, 31, 5, 70, 1, 20, 63, 4, 7];

/// The lengths of a wide prefill of `sequences` sequences, and which start
/// from their slot's state (every third from the second has none).
fn wide_prefill(sequences: usize) -> (Vec<usize>, Vec<bool>) {
    (0..sequences)
        .map(|s| (WIDE_PREFILL_ROWS[s % WIDE_PREFILL_ROWS.len()], s % 3 != 1))
        .unzip()
}

/// The bundle's linear attention at 33 to 256 Qwen sequences, at the
/// catalog's own capacity: decode through the verify entries with one row
/// per sequence in the power-of-two bucket (`VerifyLayout::decode`), and
/// the convolution and delta-rule prefill of that many mixed sequences
/// planned at the batch's rows and sequences, each against its oracle.
/// Lean runs take 100 and 256 sequences. Without a GPU it still checks that
/// every entry plans those buckets.
#[test]
fn linear_wide_decode_and_prefill_match_oracles() {
    let counts: &[usize] = if lean() { &[100, 256] } else { &WIDE_SEQUENCES };
    let plan = |name: &str, bucket: Bucket, rows: usize, sequences: usize| {
        if let Err(refusal) = find(&symbol(name)).plan(bucket, rows as u32, sequences as u32) {
            panic!("{name} refuses {bucket:?} at {rows} rows of {sequences} sequences: {refusal}");
        }
    };
    for &live in counts {
        let bucket = VerifyLayout::decode(live).bucket.unwrap();
        for name in ["conv_verify", "delta_verify"] {
            plan(name, bucket, live, live);
        }
        let (lengths, _) = wide_prefill(live);
        let rows = lengths.iter().sum::<usize>();
        let capacity = find(&symbol("conv_prefill")).capacity();
        assert!(
            rows <= capacity.rows as usize && live <= capacity.sequences as usize,
            "{rows} rows of {live} sequences exceed the prefill capacity {capacity:?}"
        );
        let bucket = Bucket {
            rows: rows as u32,
            sequences: live as u32,
        };
        for name in [
            "conv_prefill",
            "delta_prefill_prepare",
            "delta_prefill_scan",
        ] {
            plan(name, bucket, rows, live);
        }
    }
    gpu_test(4 << 30, |session| {
        let module = module(session)?;
        for &live in counts {
            eprintln!("wide linear attention: {live} sequences");
            let seed = live as u64;
            let layout = VerifyLayout::decode(live);
            let case = format!("conv_decode_wide_s{live}");
            let mut rng = Rng::new(0x1c00 + seed);
            conv_verify_run(session, &module, case, &mut rng, &layout)?;
            let case = format!("delta_decode_wide_s{live}");
            let mut rng = Rng::new(0x1d00 + seed);
            delta_verify_run(session, &module, case, &mut rng, &layout)?;
            let (lengths, initial) = wide_prefill(live);
            let (l, i) = (&lengths, &initial);
            let case = format!("conv_prefill_wide_s{live}");
            conv_prefill_run(session, &module, case, l, i, true, 0x1e00 + seed)?;
            let case = format!("delta_prefill_wide_s{live}");
            delta_prefill_named(session, &module, case, l, i, 0x1f00 + seed)?;
        }
        Ok(())
    })
    .expect("wide linear-attention decode and prefill match the recurrent oracle");
}

// -------------------------------------------------------- benchmarks ----

/// DRAM read and write peaks from the newest probe receipt of this GPU.
fn dram_peaks(session: &Session) -> LabResult<(f64, f64, String)> {
    let root = kernel_lab::receipt_root().join("probes");
    let mut receipts = std::fs::read_dir(root)
        .map_err(|e| e.to_string())?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect::<Vec<_>>();
    // Receipt names start with a nanosecond timestamp.
    receipts.sort();
    for path in receipts.into_iter().rev() {
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        let Ok(json) = serde_json::from_slice::<Value>(&bytes) else {
            continue;
        };
        if json.get("uuid") != session.metadata.get("uuid") {
            continue;
        }
        let peak = |key: &str| {
            json.pointer(&format!("/{key}/gb_per_second"))
                .and_then(Value::as_f64)
                .filter(|x| x.is_finite() && *x > 0.0)
        };
        if let (Some(read), Some(write)) = (peak("dram_read"), peak("dram_write")) {
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            return Ok((read * 1e9, write * 1e9, format!("probes/{name}")));
        }
    }
    Err("no DRAM probe receipt for this GPU; run `kernel-lab probe` first".into())
}

/// Maximum SM clock in MHz, for the F32 FMA peak of compute-bound kernels.
fn max_sm_clock_mhz(session: &Session) -> LabResult<f64> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=clocks.max.sm",
            "--format=csv,noheader,nounits",
            &format!(
                "--id={}",
                session.metadata["uuid"]
                    .as_str()
                    .ok_or("missing device UUID")?
            ),
        ])
        .output()
        .map_err(|e| e.to_string())?;
    String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .map_err(|_| "unreadable max SM clock".into())
}

fn launch_floor(session: &Session) -> LabResult<f64> {
    let module = session.module("probe_sm120")?;
    let function = module
        .function("chop_probe_sm120_v1")
        .map_err(|e| e.to_string())?;
    let output = session.allocate(4)?;
    let timing = kernel_lab::time_named(session, "launch_floor", 2, 9, |stream| {
        function
            .launch(
                stream,
                launch::simple_config(
                    (1, 1, 1),
                    (32, 1, 1),
                    0,
                    runtime_cuda::LaunchConfig::NO_CLUSTER,
                    false,
                )?,
                &[runtime_cuda::KernelArgument::Buffer(&output)],
            )
            .map_err(|e| e.to_string())
    })?;
    Ok(f64::from(timing.median_ms) * 1e3)
}

struct Bench {
    read_peak: f64,
    write_peak: f64,
    source: String,
    floor_us: f64,
    fma_per_second: f64,
    rows: Vec<Value>,
}

impl Bench {
    /// Times `work` (cold L2) and records it against the DRAM roofline of
    /// `read` and `write` bytes, or the F32 FMA roofline when `fma` is set.
    #[allow(clippy::too_many_arguments)]
    fn record(
        &mut self,
        session: &Session,
        case: &str,
        read: f64,
        write: f64,
        fma: f64,
        bottleneck: &str,
        work: impl FnMut(&Stream) -> LabResult<()>,
    ) -> LabResult<()> {
        let timing = kernel_lab::time_named(session, case, 3, 15, work)?;
        let seconds = f64::from(timing.median_ms) * 1e-3;
        let memory_us = (read / self.read_peak + write / self.write_peak) * 1e6;
        let bandwidth_us = (read + write) / self.read_peak * 1e6;
        let compute_us = fma / self.fma_per_second * 1e6;
        let bound = if compute_us > memory_us {
            "compute"
        } else {
            "memory"
        };
        let roofline_us = memory_us.max(compute_us);
        let row = json!({
            "case": case,
            "median_us": seconds * 1e6,
            "min_us": f64::from(timing.min_ms) * 1e3,
            "rounds": timing.rounds,
            "bytes_read": read,
            "bytes_written": write,
            "fma": fma,
            "roofline": bound,
            "roofline_us": roofline_us,
            "percent_of_roofline": 100.0 * roofline_us / (seconds * 1e6),
            "percent_of_read_peak_bandwidth": 100.0 * bandwidth_us / (seconds * 1e6),
            "percent_of_f32_fma_peak": 100.0 * compute_us / (seconds * 1e6),
            "launch_floor_us": self.floor_us,
            "bottleneck": bottleneck,
        });
        eprintln!(
            "{case}: {:.2} us, {:.1}% of {bound} roofline ({:.2} us)",
            seconds * 1e6,
            100.0 * roofline_us / (seconds * 1e6),
            roofline_us
        );
        self.rows.push(row);
        Ok(())
    }
}

impl Bench {
    /// Per-call time of `calls` PDL launches captured in one graph, with a
    /// cold L2 before the graph. Each call touches its own state, so a
    /// footprint above the L2 size keeps every call on DRAM.
    #[allow(clippy::too_many_arguments)]
    fn record_graph(
        &mut self,
        session: &Session,
        case: &str,
        read: f64,
        write: f64,
        calls: usize,
        bottleneck: &str,
        launch_one: impl Fn(&Stream, usize) -> LabResult<()>,
    ) -> LabResult<()> {
        let graph = session.capture(|stream| (0..calls).try_for_each(|i| launch_one(stream, i)))?;
        let timing = kernel_lab::time_named(session, case, 2, 9, |stream| {
            graph.launch(stream).map_err(|e| e.to_string())
        })?;
        let per_call_us = f64::from(timing.median_ms) * 1e3 / calls as f64;
        let roofline_us = (read / self.read_peak + write / self.write_peak) * 1e6;
        eprintln!(
            "{case}: {per_call_us:.2} us per call in a {calls}-call graph, {:.1}% of memory roofline ({roofline_us:.2} us)",
            100.0 * roofline_us / per_call_us
        );
        self.rows.push(json!({
            "case": case,
            "timing": "graph",
            "calls_per_graph": calls,
            "per_call_us": per_call_us,
            "bytes_read": read,
            "bytes_written": write,
            "roofline": "memory",
            "roofline_us": roofline_us,
            "percent_of_roofline": 100.0 * roofline_us / per_call_us,
            "bottleneck": bottleneck,
        }));
        Ok(())
    }
}

/// Calls per graph so that their distinct states exceed 160 MB.
fn graph_calls(bytes_per_call: f64) -> usize {
    ((160e6 / bytes_per_call).ceil() as usize).clamp(4, 256)
}

fn state_bytes(sequences: usize) -> f64 {
    let g = geometry();
    (sequences * g.delta_state * 4) as f64
}

#[test]
fn linear_bench_gpu() {
    let g = geometry();
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("linear-attention benchmarks: set CHOPINFER_KERNEL_BENCH=1").unwrap();
        return;
    }
    // CHOPINFER_KERNEL_BENCH_VARIANT=prefill (or decode) selects one group.
    let variant = std::env::var("CHOPINFER_KERNEL_BENCH_VARIANT").unwrap_or_default();
    let (decode_cases, prefill_cases) = (variant != "prefill", variant != "decode");
    gpu_test(8 << 30, |session| {
        let module = module(session)?;
        let (read_peak, write_peak, source) = dram_peaks(session)?;
        let clock = max_sm_clock_mhz(session)?;
        let sms = session.metadata["sm_count"].as_f64().unwrap_or(188.0);
        let mut bench = Bench {
            read_peak,
            write_peak,
            source,
            floor_us: launch_floor(session)?,
            fma_per_second: sms * 128.0 * clock * 1e6,
            rows: Vec::new(),
        };
        let mut rng = Rng::new(0xbe00);
        let c = g.channels;
        let mut delta_decode = Kernel::load(&module, &symbol("delta_decode"))?;
        let mut delta_verify = Kernel::load(&module, &symbol("delta_verify"))?;
        let mut conv_decode = Kernel::load(&module, &symbol("conv_decode"))?;
        let mut conv_verify = Kernel::load(&module, &symbol("conv_verify"))?;
        let mut conv_prefill = Kernel::load(&module, &symbol("conv_prefill"))?;
        let mut prepare = Kernel::load(&module, &symbol("delta_prefill_prepare"))?;
        let mut scan = Kernel::load(&module, &symbol("delta_prefill_scan"))?;
        let mut norm = Kernel::load(&module, &symbol("gated_norm"))?;
        let small = (c + 2 * g.gate_stride + g.norm_width) as f64 * 2.0;
        for batch in [1_usize, 8, 16, 32] {
            if !decode_cases {
                break;
            }
            let data = DeltaCase::new(&mut rng, batch, batch);
            let mut op = data.operands(session, batch)?;
            op.i32(session, "slot_ids", &(0..batch as i32).collect::<Vec<_>>())?;
            op.u32(session, "live_sequences", batch as u32)?;
            let n = batch as u32;
            // A bundle captures one plan per batch bucket; time that plan.
            delta_decode.bucket = Bucket { rows: n, sequences: n };
            conv_decode.bucket = Bucket { rows: n, sequences: n };
            bench.record(
                session,
                &format!("delta_decode_c{batch}"),
                state_bytes(batch) + small * batch as f64,
                state_bytes(batch),
                0.0,
                "state read and write",
                |stream| delta_decode.launch(session, stream, &op, n, n, true),
            )?;
            let calls = graph_calls(state_bytes(batch));
            let data = DeltaCase::new(&mut rng, batch, batch * calls);
            let mut op = data.operands(session, batch)?;
            op.u32(session, "live_sequences", batch as u32)?;
            let mut per_call = Vec::with_capacity(calls);
            for call in 0..calls {
                let mut call_op = Operands {
                    buffers: op.buffers.clone(),
                    scalars: op.scalars.clone(),
                };
                call_op.i32(
                    session,
                    "slot_ids",
                    &((call * batch) as i32..((call + 1) * batch) as i32).collect::<Vec<_>>(),
                )?;
                per_call.push(call_op);
            }
            bench.record_graph(
                session,
                &format!("delta_decode_c{batch}"),
                state_bytes(batch) + small * batch as f64,
                state_bytes(batch),
                calls,
                "state read and write",
                |stream, call| delta_decode.launch(session, stream, &per_call[call], n, n, true),
            )?;
            let conv = ConvCase::new(&mut rng, batch, batch);
            let mut op = conv.operands(session, batch)?;
            op.i32(session, "slot_ids", &(0..batch as i32).collect::<Vec<_>>())?;
            op.u32(session, "live_sequences", batch as u32)?;
            let conv_state = (batch * HISTORY * c * 2) as f64;
            bench.record(
                session,
                &format!("conv_decode_c{batch}"),
                conv_state + (batch * c * 2) as f64,
                conv_state + (batch * c * 2) as f64,
                0.0,
                &format!("launch floor; {} KB per sequence", (HISTORY * c * 2) as f64 / 1024.0),
                |stream| conv_decode.launch(session, stream, &op, n, n, true),
            )?;
        }
        for (batch, rows) in [(1_usize, 4_usize), (8, 4), (32, 4), (8, 8), (32, 8)] {
            if !decode_cases {
                break;
            }
            let data = DeltaCase::new(&mut rng, batch * rows, batch * (rows + 1));
            let mut op = data.operands(session, batch * rows)?;
            let lengths = vec![rows; batch];
            op.i32(session, "cu_rows", &cu(&lengths))?;
            op.i32(session, "slot_ids", &(0..batch as i32).collect::<Vec<_>>())?;
            let max = verify_row_limit() as usize;
            let checkpoint_ids = (0..batch * max)
                .map(|i| (batch + (i / max) * rows + (i % max).min(rows - 1)) as i32)
                .collect::<Vec<_>>();
            op.i32(session, "checkpoint_ids", &checkpoint_ids)?;
            op.u32(session, "live_sequences", batch as u32)?;
            let (r, n) = ((batch * rows) as u32, batch as u32);
            let verify_bucket = Bucket {
                rows: n * verify_row_limit(),
                sequences: n,
            };
            delta_verify.bucket = verify_bucket;
            conv_verify.bucket = verify_bucket;
            bench.record(
                session,
                &format!("delta_verify_c{batch}_r{rows}"),
                state_bytes(batch) + small * (batch * rows) as f64,
                state_bytes(batch) * rows as f64,
                0.0,
                "checkpoint writes",
                |stream| delta_verify.launch(session, stream, &op, r, n, true),
            )?;
            // Graph: every call reads its own slots and writes its own
            // checkpoints.
            let per_slots = batch * (1 + max);
            let calls = graph_calls(state_bytes(batch) * (1 + rows) as f64);
            let data = DeltaCase::new(&mut rng, batch * rows, per_slots * calls);
            let op = data.operands(session, batch * rows)?;
            let mut per_call = Vec::with_capacity(calls);
            for call in 0..calls {
                let mut call_op = Operands {
                    buffers: op.buffers.clone(),
                    scalars: op.scalars.clone(),
                };
                let base = (call * per_slots) as i32;
                call_op.i32(session, "cu_rows", &cu(&lengths))?;
                call_op.i32(session, "slot_ids", &(base..base + batch as i32).collect::<Vec<_>>())?;
                call_op.i32(
                    session,
                    "checkpoint_ids",
                    &(0..(batch * max) as i32)
                        .map(|i| base + batch as i32 + i)
                        .collect::<Vec<_>>(),
                )?;
                call_op.u32(session, "live_sequences", n)?;
                per_call.push(call_op);
            }
            bench.record_graph(
                session,
                &format!("delta_verify_c{batch}_r{rows}"),
                state_bytes(batch) + small * (batch * rows) as f64,
                state_bytes(batch) * rows as f64,
                calls,
                "checkpoint writes",
                |stream, call| delta_verify.launch(session, stream, &per_call[call], r, n, true),
            )?;
            let conv = ConvCase::new(&mut rng, batch * rows, batch * (rows + 1));
            let mut op = conv.operands(session, batch * rows)?;
            op.i32(session, "cu_rows", &cu(&lengths))?;
            op.i32(session, "slot_ids", &(0..batch as i32).collect::<Vec<_>>())?;
            op.i32(session, "checkpoint_ids", &checkpoint_ids)?;
            op.u32(session, "live_sequences", batch as u32)?;
            let conv_state = (batch * HISTORY * c * 2) as f64;
            bench.record(
                session,
                &format!("conv_verify_c{batch}_r{rows}"),
                conv_state + (batch * rows * c * 2) as f64,
                conv_state * rows as f64 + (batch * rows * c * 2) as f64,
                0.0,
                "launch floor",
                |stream| conv_verify.launch(session, stream, &op, r, n, true),
            )?;
        }
        for rows in [1_usize, 32, 2048, 8192] {
            if !decode_cases {
                break;
            }
            let data = NormCase::new(&mut rng, rows);
            let op = data.operands(session, rows)?;
            let r = rows as u32;
            norm.bucket = Bucket {
                rows: r,
                sequences: r,
            };
            bench.record(
                session,
                &format!("gated_norm_r{rows}"),
                (rows * g.norm_width * 4) as f64,
                (rows * g.norm_width * 2) as f64,
                0.0,
                "row reads and writes",
                |stream| norm.launch(session, stream, &op, r, r, true),
            )?;
        }
        for lengths in [vec![2048_usize], vec![8192], vec![1024; 8]] {
            if !prefill_cases {
                break;
            }
            let rows = lengths.iter().sum::<usize>();
            let sequences = lengths.len();
            let label = if sequences == 1 {
                format!("{rows}")
            } else {
                format!("{sequences}x{}", lengths[0])
            };
            let (r, n) = (rows as u32, sequences as u32);
            let prefill_bucket = Bucket {
                rows: r,
                sequences: n,
            };
            conv_prefill.bucket = prefill_bucket;
            prepare.bucket = prefill_bucket;
            scan.bucket = prefill_bucket;
            let conv = ConvCase::new(&mut rng, rows, sequences);
            let mut op = conv.operands(session, rows)?;
            op.i32(session, "cu_seqlens", &cu(&lengths))?;
            op.i32(session, "slot_ids", &(0..sequences as i32).collect::<Vec<_>>())?;
            op.i32(session, "initial", &vec![1; sequences])?;
            op.i32(session, "anchor_ids", &vec![-1; sequences])?;
            op.u32(session, "live_sequences", n)?;
            bench.record(
                session,
                &format!("conv_prefill_{label}"),
                (rows * c * 2) as f64,
                (rows * c * 2) as f64,
                0.0,
                "row reads and writes",
                |stream| conv_prefill.launch(session, stream, &op, r, n, true),
            )?;
            let data = DeltaCase::new(&mut rng, rows, sequences);
            let mut op = data.operands(session, rows)?;
            op.i32(session, "cu_seqlens", &cu(&lengths))?;
            op.i32(session, "slot_ids", &(0..sequences as i32).collect::<Vec<_>>())?;
            op.i32(session, "initial", &vec![1; sequences])?;
            op.i32(session, "anchor_ids", &vec![-1; sequences])?;
            op.u32(session, "live_sequences", n)?;
            op.zero(session, "workspace", prepare.workspace_bytes())?;
            let chunks = lengths.iter().map(|len| len.div_ceil(64)).sum::<usize>() as f64;
            let key_heads = g.key_heads as f64;
            let value_heads = g.value_heads as f64;
            let work = prepare.workspace_bytes() as f64 * chunks
                / f64::from(prefill_bucket.chunks());
            let kq = (rows * 2 * g.key_heads * DIM as usize * 2) as f64;
            // Prepare: Gram K K^T and Q K^T per key head, forward
            // substitution per value head.
            let prepare_fma = chunks * key_heads * (2.0 * 64.0 * 64.0 * 128.0)
                + chunks * value_heads * 64.0 * 64.0 * 63.0 / 2.0;
            bench.record(
                session,
                &format!("delta_prefill_prepare_{label}"),
                kq + (rows * 2 * g.gate_stride * 2) as f64,
                work,
                prepare_fma,
                "Gram matrices and triangular solves in F32",
                |stream| prepare.launch(session, stream, &op, r, n, true),
            )?;
            // Scan: per chunk and 16-row slice, [Kr; Qr] S^T (128 x 16 x
            // 128), T' Z and P U (64 x 16 x 64 each), U^T Kr (16 x 64 x 128).
            let scan_fma = chunks * value_heads * 8.0 * 16.0 * (128.0 * 128.0 + 2.0 * 64.0 * 64.0 + 64.0 * 128.0);
            let states = state_bytes(sequences);
            bench.record(
                session,
                &format!("delta_prefill_scan_{label}"),
                kq + (rows * g.norm_width * 2) as f64 + work + states,
                (rows * g.norm_width * 2) as f64 + states,
                scan_fma,
                &format!("sequential chunk walk: F32 FMA on {} CTAs per sequence, chunk operands reloaded per 16-row slice", g.value_heads * 8),
                |stream| scan.launch(session, stream, &op, r, n, true),
            )?;
            bench.record(
                session,
                &format!("delta_prefill_total_{label}"),
                2.0 * kq + (rows * g.norm_width * 2) as f64 + 2.0 * work + states,
                (rows * g.norm_width * 2) as f64 + work + states,
                prepare_fma + scan_fma,
                "prepare then scan",
                |stream| {
                    prepare.launch(session, stream, &op, r, n, true)?;
                    scan.launch(session, stream, &op, r, n, true)
                },
            )?;
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("gate".into(), json!("G4"));
        receipt.insert("profile".into(), json!(g.profile.as_str()));
        if g.ranks != 4 {
            receipt.insert("qwen38_ranks".into(), json!(g.ranks));
        }
        receipt.insert("timing".into(), json!("eager CUDA events, selected L2 eviction before each call"));
        receipt.insert("dram_read_peak_gb_per_second".into(), json!(bench.read_peak / 1e9));
        receipt.insert("dram_write_peak_gb_per_second".into(), json!(bench.write_peak / 1e9));
        receipt.insert("peak_source".into(), json!(bench.source));
        receipt.insert("max_sm_clock_mhz".into(), json!(clock));
        receipt.insert("f32_fma_peak_tflops".into(), json!(2.0 * bench.fma_per_second / 1e12));
        receipt.insert("launch_floor_us".into(), json!(bench.floor_us));
        receipt.insert("cases".into(), json!(bench.rows));
        session.receipt("linear_bench", receipt)?;
        Ok(())
    })
    .expect("linear-attention benchmarks run");
}

#[path = "linear_e5.rs"]
mod e5;

#[test]
fn linear_checkpoint_pool_matches_oracles_and_disjoint_bits() {
    gpu_test(4 << 30, |session| {
        let module = module(session)?;
        for live in [1, 8, 16, 31, 32, 33, 64, 128, 255, 256] {
            for drafting in [false, true] {
                let layout = VerifyLayout::checkpoint_pool(live, drafting);
                conv_verify_run(
                    session,
                    &module,
                    format!("conv_checkpoint_pool_{live}_{drafting}"),
                    &mut Rng::new(0xc100 + live as u64),
                    &layout,
                )?;
                delta_verify_run(
                    session,
                    &module,
                    format!("delta_checkpoint_pool_{live}_{drafting}"),
                    &mut Rng::new(0xd100 + live as u64),
                    &layout,
                )?;
            }
        }
        Ok(())
    })
    .expect("in-place state and bounded checkpoints match the oracle and disjoint output bits");
}

// ------------------------------------------- reject and continue ----

/// One linear verify entry's operands for the reject-and-continue check:
/// its per-row inputs (buffer name, row stride), its fixed parameters, and
/// the widths of an output row and a state entry.
struct VerifyKind {
    entry: &'static str,
    rows: Vec<(&'static str, usize, f64)>,
    output_width: usize,
    entry_bytes: usize,
}

/// A step's items: state slot, the bank rows it runs, and its checkpoint
/// entry per row (the slot itself for one-row items).
struct StepItem {
    slot: i32,
    rows: Vec<usize>,
    checkpoints: Vec<i32>,
}

/// Runs one verify launch of `items` on the pool in `op`, gathering each
/// item's rows from `bank`; returns each item's output rows. Sequences past
/// the live items pad the power-of-two bucket with one row each on `spare`.
fn verify_step(
    session: &Session,
    module: &Module,
    kind: &VerifyKind,
    op: &mut Operands,
    bank: &[Vec<u16>],
    items: &[StepItem],
    spare: i32,
) -> LabResult<Vec<Vec<u16>>> {
    let max = verify_row_limit() as usize;
    let sequences = items.len().next_power_of_two();
    let mut lengths: Vec<usize> = items.iter().map(|item| item.rows.len()).collect();
    lengths.resize(sequences, 1);
    let rows = lengths.iter().sum::<usize>();
    let live_rows = items.iter().map(|item| item.rows.len()).sum::<usize>();
    for (buffer, (name, stride, _)) in bank.iter().zip(&kind.rows) {
        let mut gathered = vec![0_u16; rows.next_power_of_two() * stride];
        for (at, &row) in items.iter().flat_map(|item| &item.rows).enumerate() {
            gathered[at * stride..(at + 1) * stride]
                .copy_from_slice(&buffer[row * stride..(row + 1) * stride]);
        }
        op.bf16(session, name, &gathered)?;
    }
    let mut slot_ids: Vec<i32> = items.iter().map(|item| item.slot).collect();
    slot_ids.resize(sequences, spare);
    let mut checkpoint_ids = vec![spare; sequences * max];
    for (index, item) in items.iter().enumerate() {
        checkpoint_ids[index * max..index * max + item.checkpoints.len()]
            .copy_from_slice(&item.checkpoints);
    }
    op.i32(session, "cu_rows", &cu(&lengths))?;
    op.i32(session, "slot_ids", &slot_ids)?;
    op.i32(session, "checkpoint_ids", &checkpoint_ids)?;
    op.u32(session, "live_sequences", items.len() as u32)?;
    op.zero(
        session,
        "output",
        rows.next_power_of_two() * kind.output_width * 2,
    )?;
    poison::poison(session, &[&op.buffers["output"]])?;
    let kernel = Kernel::load(module, &symbol(kind.entry))?
        .with_bucket(rows.next_power_of_two() as u32, sequences as u32);
    let initial = op.read(session, "state", op.buffers["state"].byte_len())?;
    let mut write_only = Vec::new();
    let mut updated = Vec::new();
    for item in items {
        for &entry in &item.checkpoints {
            let range = entry as usize * kind.entry_bytes..(entry as usize + 1) * kind.entry_bytes;
            if entry == item.slot {
                updated.push(range);
            } else {
                write_only.push(range);
            }
        }
    }
    let mut checks = Replay::new(session)
        .output(
            "output",
            &op.buffers["output"],
            WriteMap::of(DType::Bf16).prefix(live_rows * kind.output_width)?,
        )
        .state(
            "state",
            &op.buffers["state"],
            initial,
            WriteMap::of(if kind.entry == "conv_verify" {
                DType::Bf16
            } else {
                DType::F32
            })
            .byte_ranges(write_only)?,
            updated,
        );
    checks.eager(|| {
        kernel.launch(
            session,
            &session.stream,
            op,
            live_rows as u32,
            items.len() as u32,
            true,
        )
    })?;
    checks.relaunch(|| {
        kernel.launch(
            session,
            &session.stream,
            op,
            live_rows as u32,
            items.len() as u32,
            false,
        )
    })?;
    let graph = session.capture(|stream| {
        kernel.launch(
            session,
            stream,
            op,
            live_rows as u32,
            items.len() as u32,
            true,
        )
    })?;
    for _ in 0..2 {
        checks.graph(&graph)?;
    }
    // Publish mutable index tables after a producer tail. The dependent
    // must wait before reading source ids, row lengths or checkpoint ids.
    let producer_module = session.module("probe_pdl_prefetch_sm120")?;
    let eviction = session.allocate(16)?;
    eviction
        .copy_from_bytes(&[0; 16], &session.stream)
        .map_err(|e| e.to_string())?;
    let checksum = session.allocate(128 * 4)?;
    poison::poison(session, &[&checksum])?;
    let mut fresh = Vec::new();
    let mut descriptors = Vec::new();
    for name in ["slot_ids", "checkpoint_ids", "cu_rows"] {
        let bytes = op.read(session, name, op.buffers[name].byte_len())?;
        let source = session.allocate(bytes.len())?;
        source
            .copy_from_bytes(&bytes, &session.stream)
            .map_err(|e| e.to_string())?;
        descriptors.extend_from_slice(
            &source
                .device_address_at(0, bytes.len())
                .map_err(|e| e.to_string())?
                .to_ne_bytes(),
        );
        descriptors.extend_from_slice(
            &op.buffers[name]
                .device_address_at(0, bytes.len())
                .map_err(|e| e.to_string())?
                .to_ne_bytes(),
        );
        descriptors.extend_from_slice(&(bytes.len() as u32 / 4).to_ne_bytes());
        descriptors.extend_from_slice(&0_u32.to_ne_bytes());
        fresh.push(source);
    }
    let table = session.allocate(descriptors.len())?;
    table
        .copy_from_bytes(&descriptors, &session.stream)
        .map_err(|e| e.to_string())?;
    let producer = producer_module
        .function("chop_probe_pdl_prefetch_producer_v1")
        .map_err(|e| e.to_string())?;
    checks.pdl(
        || {
            producer
                .launch(
                    &session.stream,
                    launch::simple_config(
                        (1, 1, 1),
                        (128, 1, 1),
                        0,
                        runtime_cuda::LaunchConfig::NO_CLUSTER,
                        true,
                    )?,
                    &[
                        runtime_cuda::KernelArgument::Buffer(&eviction),
                        runtime_cuda::KernelArgument::Buffer(&checksum),
                        runtime_cuda::KernelArgument::U64(1),
                        runtime_cuda::KernelArgument::Buffer(&table),
                        runtime_cuda::KernelArgument::U32(3),
                        runtime_cuda::KernelArgument::Buffer(&op.buffers["live_sequences"]),
                        runtime_cuda::KernelArgument::U32(items.len() as u32),
                        runtime_cuda::KernelArgument::U64(15_000),
                    ],
                )
                .map_err(|e| e.to_string())
        },
        || {
            kernel.launch(
                session,
                &session.stream,
                op,
                live_rows as u32,
                items.len() as u32,
                true,
            )
        },
        true,
    )?;
    let output = op.read_bf16(session, "output", rows * kind.output_width)?;
    let mut at = 0;
    Ok(items
        .iter()
        .map(|item| {
            let rows = &output[at * kind.output_width..(at + item.rows.len()) * kind.output_width];
            at += item.rows.len();
            rows.to_vec()
        })
        .collect())
}

/// Qwen recurrent state under speculation above 32 sequences: speculation that rejects drafts and
/// continues must leave every slot with the bits of running only the
/// accepted rows. `live` items (a third drafting, at most 32, in the
/// highest pool slots first) verify, commit a kept prefix by copying its
/// checkpoint into the slot, verify again with the checkpoint pool
/// reassigned to other items in another order, and decode once more. Every
/// row reads fresh input, so a rejected row differs from the one that
/// replaces it. The reference runs each item's accepted rows alone, one
/// row per launch, in place. Slot bytes and every accepted output row must
/// be identical.
fn reject_and_continue(
    session: &Session,
    module: &Module,
    kind: &VerifyKind,
    op: &mut Operands,
    initial: &[u8],
    live: usize,
    seed: u64,
) -> LabResult<()> {
    let mut rng = Rng::new(seed);
    let mut report = Report::new(format!("{}_reject_and_continue_s{live}", kind.entry));
    let drafting = live.min(32).min((256 - live) / 3);
    let pool = live + 32 * 4 + 1;
    let spare = (pool - 1) as i32;
    let slot = |item: usize| (live - 1 - item) as i32;
    // Rows are drawn from the bank in launch order.
    let mut bank_rows = 0;
    let mut take = |count: usize| {
        let rows = (bank_rows..bank_rows + count).collect::<Vec<_>>();
        bank_rows += count;
        rows
    };
    // Each item's accepted bank rows and their verify outputs.
    let mut accepted: Vec<Vec<(usize, Vec<u16>)>> = vec![Vec::new(); live];
    let mut steps = Vec::new();
    for step in 0..3 {
        let mut order: Vec<usize> = (0..live).collect();
        if step == 1 {
            order.reverse();
        }
        let mut next = 0;
        let items: Vec<_> = order
            .iter()
            .map(|&item| {
                if step < 2 && item % 3 == step && next < drafting {
                    let first = live + 4 * next;
                    next += 1;
                    let kept = 1 + (item + step) % 4;
                    (
                        item,
                        take(4),
                        (first..first + 4).map(|entry| entry as i32).collect(),
                        kept,
                    )
                } else {
                    (item, take(1), vec![slot(item)], 1)
                }
            })
            .collect::<Vec<(usize, Vec<usize>, Vec<i32>, usize)>>();
        steps.push(items);
    }
    let bank: Vec<Vec<u16>> = kind
        .rows
        .iter()
        .map(|&(_, stride, scale)| rng.bf16(bank_rows * stride, scale))
        .collect();
    let bytes = kind.entry_bytes;
    op.bytes(session, "state", initial)?;
    for items in &steps {
        let launch: Vec<_> = items
            .iter()
            .map(|(item, rows, checkpoints, _)| StepItem {
                slot: slot(*item),
                rows: rows.clone(),
                checkpoints: checkpoints.clone(),
            })
            .collect();
        let outputs = verify_step(session, module, kind, op, &bank, &launch, spare)?;
        // Commit: copy each drafting item's kept checkpoint into its slot.
        let mut state = op.read(session, "state", pool * bytes)?;
        for ((item, rows, checkpoints, kept), output) in items.iter().zip(outputs) {
            if rows.len() > 1 {
                let from = checkpoints[kept - 1] as usize * bytes;
                let to = slot(*item) as usize * bytes;
                let entry = state[from..from + bytes].to_vec();
                state[to..to + bytes].copy_from_slice(&entry);
            }
            for (row, &bank_row) in rows.iter().take(*kept).enumerate() {
                let width = kind.output_width;
                accepted[*item].push((bank_row, output[row * width..(row + 1) * width].to_vec()));
            }
        }
        op.buffers["state"]
            .copy_from_bytes(&state, &session.stream)
            .map_err(|e| e.to_string())?;
    }
    let speculative = op.read(session, "state", live * bytes)?;
    // The reference: accepted rows only, one per launch, from the same
    // initial slots.
    op.bytes(session, "state", initial)?;
    let depth = accepted.iter().map(Vec::len).max().unwrap_or(0);
    let mut mismatched_outputs = Vec::new();
    for position in 0..depth {
        let active: Vec<usize> = (0..live)
            .filter(|&item| accepted[item].len() > position)
            .collect();
        let launch: Vec<_> = active
            .iter()
            .map(|&item| StepItem {
                slot: slot(item),
                rows: vec![accepted[item][position].0],
                checkpoints: vec![slot(item)],
            })
            .collect();
        let outputs = verify_step(session, module, kind, op, &bank, &launch, spare)?;
        for (&item, output) in active.iter().zip(outputs) {
            if output != accepted[item][position].1 {
                mismatched_outputs.push((item, position));
            }
        }
    }
    let reference = op.read(session, "state", live * bytes)?;
    report.exact(
        "accepted output rows equal the accepted-only run bitwise",
        if mismatched_outputs.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "(item, position) {:?}",
                &mismatched_outputs[..mismatched_outputs.len().min(8)]
            ))
        },
    );
    let differing: Vec<usize> = (0..live)
        .filter(|&item| {
            let at = slot(item) as usize * bytes;
            speculative[at..at + bytes] != reference[at..at + bytes]
        })
        .collect();
    report.exact(
        "slots equal the accepted-only run bitwise",
        if differing.is_empty() {
            Ok(())
        } else {
            Err(format!("items {:?}", &differing[..differing.len().min(8)]))
        },
    );
    report.exact(
        "drafting items rejected and continued",
        if drafting > 0 && accepted.iter().any(|rows| rows.len() > 3) {
            Ok(())
        } else {
            Err("no drafting item kept several rows".into())
        },
    );
    report.finish(session)
}

#[test]
fn linear_reject_and_continue_matches_accepted_only_bits() {
    gpu_test(4 << 30, |session| {
        let module = module(session)?;
        let g = geometry();
        for live in [33, 64, 128] {
            let pool = live + 32 * 4 + 1;
            let mut rng = Rng::new(0xe100 + live as u64);
            let conv = ConvCase::new(&mut rng, 1, pool);
            let mut op = conv.operands(session, 1)?;
            reject_and_continue(
                session,
                &module,
                &VerifyKind {
                    entry: "conv_verify",
                    rows: vec![("input", g.qkv_stride, 1.0)],
                    output_width: g.channels,
                    entry_bytes: HISTORY * g.channels * 2,
                },
                &mut op,
                &bytes_u16(&conv.pool),
                live,
                0xe200 + live as u64,
            )?;
            let delta = DeltaCase::new(&mut rng, 1, pool);
            let mut op = delta.operands(session, 1)?;
            reject_and_continue(
                session,
                &module,
                &VerifyKind {
                    entry: "delta_verify",
                    rows: vec![
                        ("qkv", g.qkv_stride, 1.0),
                        ("a", g.gate_stride, 1.5),
                        ("b", g.gate_stride, 1.5),
                    ],
                    output_width: g.norm_width,
                    entry_bytes: g.delta_state * 4,
                },
                &mut op,
                &bytes_f32(&delta.pool),
                live,
                0xe300 + live as u64,
            )?;
        }
        Ok(())
    })
    .expect("rejected drafts leave the bits of the accepted rows alone");
}

/// `conv_prefill` over `data`'s rows at `lengths`, each sequence carrying its
/// slot's state and writing its anchor state into `anchors[s]` when that is
/// at least zero; returns (output bits, final pool).
fn conv_prefill_anchored(
    session: &Session,
    module: &Module,
    data: &ConvCase,
    lengths: &[usize],
    slot_ids: &[i32],
    anchors: &[i32],
) -> LabResult<(Vec<u16>, Vec<u16>)> {
    let g = geometry();
    let rows = lengths.iter().sum::<usize>();
    let mut op = data.operands(session, rows)?;
    op.i32(session, "cu_seqlens", &cu(lengths))?;
    op.i32(session, "slot_ids", slot_ids)?;
    op.i32(session, "initial", &vec![1; lengths.len()])?;
    op.i32(session, "anchor_ids", anchors)?;
    op.u32(session, "live_sequences", lengths.len() as u32)?;
    let live = (rows as u32, lengths.len() as u32);
    Kernel::load(module, &symbol("conv_prefill"))?
        .with_bucket(live.0, live.1)
        .launch(session, &session.stream, &op, live.0, live.1, true)?;
    Ok((
        op.read_bf16(session, "output", rows * g.channels)?,
        op.read_bf16(session, "state", data.pool.len())?,
    ))
}

/// The first `keep[s]` rows of each sequence of `rows`-row strided data laid
/// out by `lengths`.
fn leading_rows<T: Copy>(data: &[T], stride: usize, lengths: &[usize], keep: &[usize]) -> Vec<T> {
    let starts = cu(lengths);
    lengths
        .iter()
        .enumerate()
        .flat_map(|(s, _)| {
            let first = starts[s] as usize * stride;
            data[first..first + keep[s] * stride].iter().copied()
        })
        .collect()
}

/// A prefill sequence with an anchor writes, into its anchor entry, the
/// state after its last whole 64-row chunk when a partial chunk follows:
/// bit for bit the state a prefill of just those rows leaves in its slot.
/// The anchor changes no output or slot bit, and a sequence without a
/// partial last chunk writes nothing there.
#[test]
fn linear_prefill_anchor_state_is_the_last_whole_chunk_gpu() {
    gpu_test(2 << 30, |session| {
        let module = module(session)?;
        let g = geometry();
        let mut report = Report::new("linear_prefill_anchor_state");
        // Anchored sequences of one to several whole chunks and a tail, one
        // of whole chunks only, and an unanchored one.
        let lengths = [65_usize, 130, 64 * 5 + 63, 128, 200];
        let anchored = [true, true, true, true, false];
        let sequences = lengths.len();
        let (slot_ids, pool) = slots(sequences, 2 * sequences);
        // Spare entries below the slots `slots` scatters its sequences over.
        let anchors = (0..sequences)
            .map(|s| if anchored[s] { s as i32 } else { -1 })
            .collect::<Vec<_>>();
        let whole = lengths.map(|len| len / 64 * 64);
        let rows = lengths.iter().sum::<usize>();
        let mut rng = Rng::new(0x0a17);

        let conv = ConvCase::new(&mut rng, rows, pool);
        let (with_out, with_pool) =
            conv_prefill_anchored(session, &module, &conv, &lengths, &slot_ids, &anchors)?;
        let (without_out, without_pool) = conv_prefill_anchored(
            session,
            &module,
            &conv,
            &lengths,
            &slot_ids,
            &vec![-1; sequences],
        )?;
        report.exact(
            "conv output unchanged",
            (with_out == without_out)
                .then_some(())
                .ok_or_else(|| "conv output bits differ with anchors".to_owned()),
        );
        let truncated = ConvCase {
            geometry: conv.geometry,
            weight: conv.weight.clone(),
            input: leading_rows(&conv.input, g.qkv_stride, &lengths, &whole),
            pool: conv.pool.clone(),
        };
        let (_, ends) = conv_prefill_anchored(
            session,
            &module,
            &truncated,
            &whole,
            &slot_ids,
            &vec![-1; sequences],
        )?;
        let n = HISTORY * g.channels;
        let entry =
            |pool: &[u16], index: i32| pool[index as usize * n..(index as usize + 1) * n].to_vec();
        for s in 0..sequences {
            let slot = slot_ids[s];
            report.exact(
                &format!("conv seq{s} slot unchanged"),
                (entry(&with_pool, slot) == entry(&without_pool, slot))
                    .then_some(())
                    .ok_or_else(|| "slot state differs with anchors".to_owned()),
            );
            let spare = s as i32;
            let expected = if anchored[s] && lengths[s] % 64 != 0 {
                entry(&ends, slot)
            } else {
                entry(&conv.pool, spare)
            };
            report.exact(
                &format!("conv seq{s} anchor"),
                (entry(&with_pool, spare) == expected)
                    .then_some(())
                    .ok_or_else(|| "anchor entry is not the whole-chunk state".to_owned()),
            );
        }

        let delta = DeltaCase::new(&mut rng, rows, pool);
        let initial = vec![true; sequences];
        let (with_out, with_pool) = delta_prefill_anchored(
            session, &module, &delta, &lengths, &initial, &slot_ids, &anchors,
        )?;
        let (without_out, without_pool) =
            delta_prefill_run(session, &module, &delta, &lengths, &initial, &slot_ids)?;
        report.exact(
            "delta output unchanged",
            (with_out == without_out)
                .then_some(())
                .ok_or_else(|| "delta output bits differ with anchors".to_owned()),
        );
        let truncated = DeltaCase {
            geometry: delta.geometry,
            qkv: leading_rows(&delta.qkv, g.qkv_stride, &lengths, &whole),
            a: leading_rows(&delta.a, g.gate_stride, &lengths, &whole),
            b: leading_rows(&delta.b, g.gate_stride, &lengths, &whole),
            a_log: delta.a_log.clone(),
            dt_bias: delta.dt_bias.clone(),
            pool: delta.pool.clone(),
        };
        let (_, ends) =
            delta_prefill_run(session, &module, &truncated, &whole, &initial, &slot_ids)?;
        let n = g.delta_state;
        let bits = |pool: &[f32], index: i32| {
            pool[index as usize * n..(index as usize + 1) * n]
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        };
        for s in 0..sequences {
            let slot = slot_ids[s];
            report.exact(
                &format!("delta seq{s} slot unchanged"),
                (bits(&with_pool, slot) == bits(&without_pool, slot))
                    .then_some(())
                    .ok_or_else(|| "slot state differs with anchors".to_owned()),
            );
            let spare = s as i32;
            let expected = if anchored[s] && lengths[s] % 64 != 0 {
                bits(&ends, slot)
            } else {
                bits(&delta.pool, spare)
            };
            report.exact(
                &format!("delta seq{s} anchor"),
                (bits(&with_pool, spare) == expected)
                    .then_some(())
                    .ok_or_else(|| "anchor entry is not the whole-chunk state".to_owned()),
            );
        }
        report.finish(session)
    })
    .expect("prefill anchors keep the last whole chunk's state");
}

#[test]
fn linear_fixtures_use_the_supplied_geometry() {
    use kernel_core::contract::RecurrentPhase;
    let g = Geometry::new(kernel_lab::lab_shapes::PROFILES[0], 1, 2, 4, 128);
    let mut rng = Rng::new(41);
    let conv = ConvCase::with_geometry(&mut rng, 2, 1, g);
    assert_eq!(conv.rows(1, 1).len(), 1024);
    assert_eq!(conv.slot(0).len(), 3 * 1024);
    let delta = DeltaCase::with_geometry(&mut rng, 2, 1, g);
    let query = RecurrentQuery::GatedDeltaRule {
        key_heads: 2,
        value_heads: 4,
        key_dim: 128,
        value_dim: 128,
        phase: RecurrentPhase::Prefill,
        capacity: 2,
    };
    let result = delta.oracle(query, 0, 2, &delta.slot(0)).unwrap();
    assert_eq!(result.output.values.len(), 2 * 4 * 128);
    assert!(result.output.values.iter().all(|v| v.is_finite()));
    let norm = NormCase::with_geometry(&mut rng, 2, g);
    assert_eq!(norm.input.len(), 2 * 4 * 128);
    assert_eq!(norm.weight.len(), 128);
}

#[path = "linear_indexed_commit.rs"]
mod indexed_commit;
