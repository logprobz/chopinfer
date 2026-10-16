//! Linear-attention engine against the kernel-core recurrent
//! oracles on one GPU: causal convolution, the gated delta rule for prefill,
//! decode and verify, and the gated RMSNorm at the Qwen3.8 Flash TP4 shapes.

use kernel_build::EntryParam;
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{Argument, DType, RecurrentQuery};
use kernel_core::oracle::{OracleOutput, max_bound_use};
use kernel_core::recurrent_oracle::{self, GatedDeltaInputs};
use kernel_lab::launch::{self, Bindings};
use kernel_lab::{LabResult, Session, check_output, gpu_test};
use kernels_sm120::linear::{Bucket, Entry};
use model_qwen38_contract::sm120::linear::{
    QWEN38_CONV_CHANNELS as CHANNELS, QWEN38_HEAD_DIM as DIM, QWEN38_KEY_HEADS as KEY_HEADS,
    QWEN38_VALUE_HEADS as VALUE_HEADS, QWEN38_VERIFY_ROWS as VERIFY_ROWS, catalog,
};
use runtime_cuda::{DeviceBuffer, Function, Module, Stream};
use serde_json::{Value, json};
use std::collections::BTreeMap;

use kernel_lab::poison::{Replay, WriteMap};

const CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/linear_qwen38_flash_sm120.cubin"));
const HISTORY: usize = 3;
const HEAD: usize = (DIM * DIM) as usize;
const GATE_STRIDE: usize = 16;
const QKV_STRIDE: usize = CHANNELS as usize + 64;
const NORM_WIDTH: usize = (VALUE_HEADS * DIM) as usize;

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
    catalog()
        .into_iter()
        .find(|entry| entry.symbol == symbol)
        .unwrap_or_else(|| panic!("missing linear entry {symbol}"))
}

pub fn module(session: &Session) -> LabResult<Module> {
    session
        .context
        .load_module(CUBIN)
        .map_err(|e| e.to_string())
}

/// Named device operands of one case, bound by parameter name.
#[derive(Default)]
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
        )
    }
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
    weight: Vec<u16>,
    input: Vec<u16>,
    pool: Vec<u16>,
}

impl ConvCase {
    fn new(rng: &mut Rng, rows: usize, slots: usize) -> Self {
        Self {
            weight: rng.bf16(CHANNELS as usize * 4, 0.1),
            input: rng.bf16(rows * QKV_STRIDE, 1.0),
            pool: rng.bf16(slots * HISTORY * CHANNELS as usize, 1.0),
        }
    }
    fn rows(&self, first: usize, count: usize) -> Vec<f64> {
        let c = CHANNELS as usize;
        (first..first + count)
            .flat_map(|r| decode(&self.input[r * QKV_STRIDE..r * QKV_STRIDE + c]))
            .collect()
    }
    fn slot(pool: &[u16], slot: i32) -> Vec<f64> {
        let n = HISTORY * CHANNELS as usize;
        decode(&pool[slot as usize * n..(slot as usize + 1) * n])
    }
    fn operands(&self, session: &Session, rows: usize) -> LabResult<Operands> {
        let mut op = Operands::new();
        op.bf16(session, "input", &self.input)?;
        op.bf16(session, "weight", &self.weight)?;
        op.bf16(session, "state", &self.pool)?;
        op.zero(session, "output", rows * CHANNELS as usize * 2)?;
        op.scalars.insert("input_stride", QKV_STRIDE as u32);
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

pub struct DeltaCase {
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
        let heads = VALUE_HEADS as usize;
        let a_log = (0..heads)
            .map(|h| {
                let t = h as f64 / (heads - 1) as f64;
                (0.03_f64.ln() + t * (160.0_f64 / 0.03).ln()) as f32
            })
            .collect();
        let dt_bias = (0..heads)
            .map(|_| (-8.0 + 10.5 * rng.uniform()) as f32)
            .collect();
        Self {
            qkv: rng.bf16(rows * QKV_STRIDE, 1.0),
            a: rng.bf16(rows * GATE_STRIDE, 1.5),
            b: rng.bf16(rows * GATE_STRIDE, 1.5),
            a_log,
            dt_bias,
            pool: rng.f32(slots * VALUE_HEADS as usize * HEAD, 0.3),
        }
    }

    pub fn operands(&self, session: &Session, rows: usize) -> LabResult<Operands> {
        let mut op = Operands::new();
        op.bf16(session, "qkv", &self.qkv)?;
        op.bf16(session, "a", &self.a)?;
        op.bf16(session, "b", &self.b)?;
        op.f32(session, "a_log", &self.a_log)?;
        op.f32(session, "dt_bias", &self.dt_bias)?;
        op.f32(session, "state", &self.pool)?;
        op.zero(session, "output", rows * NORM_WIDTH * 2)?;
        op.scalars.insert("qkv_stride", QKV_STRIDE as u32);
        op.scalars.insert("gate_stride", GATE_STRIDE as u32);
        Ok(op)
    }

    fn slot(&self, slot: i32) -> Vec<f64> {
        let n = VALUE_HEADS as usize * HEAD;
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
        delta_oracle(query, &self.qkv, QKV_STRIDE, self, first, count, state)
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
    let (kh, vh, d) = (KEY_HEADS as usize, VALUE_HEADS as usize, DIM as usize);
    let column = |from: usize, width: usize| -> Vec<f64> {
        (first..first + count)
            .flat_map(|r| decode(&qkv[r * stride + from..r * stride + from + width]))
            .collect()
    };
    let gate = |values: &[u16]| -> Vec<f64> {
        (first..first + count)
            .flat_map(|r| decode(&values[r * GATE_STRIDE..r * GATE_STRIDE + vh]))
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
    let n = VALUE_HEADS as usize * HEAD;
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

fn verify_rows(sequences: usize, _rng: &mut Rng) -> Vec<usize> {
    vec![1; sequences]
}
fn checkpoints(slot_ids: &[i32], pool: usize) -> (Vec<i32>, usize) {
    let mut ids = vec![-1; slot_ids.len() * VERIFY_ROWS as usize];
    for s in 0..slot_ids.len() {
        ids[s * VERIFY_ROWS as usize] = (pool + s) as i32;
    }
    (ids, pool + slot_ids.len())
}
fn raised(module: &Module, symbol: &str, sequences: usize) -> LabResult<Kernel> {
    let mut kernel = Kernel::load(module, symbol)?;
    // Verify capacity counts rows at MaxRows per sequence, even for AR's one row.
    match &mut kernel.entry.query {
        RecurrentQuery::CausalConv1d { capacity, .. }
        | RecurrentQuery::GatedDeltaRule { capacity, .. } => {
            *capacity = sequences as u32 * VERIFY_ROWS
        }
        _ => unreachable!(),
    }
    kernel.bucket = Bucket {
        rows: sequences as u32,
        sequences: sequences as u32,
    };
    Ok(kernel)
}
fn conv_verify(session: &Session, module: &Module, sequences: usize, seed: u64) -> LabResult<()> {
    let case = format!("conv_verify_s{sequences}");
    let mut report = Report::new(case.clone());
    let c = CHANNELS as usize;
    let n = HISTORY * c;
    let max = VERIFY_ROWS as usize;
    let mut rng = Rng::new(seed);
    let lengths = verify_rows(sequences, &mut rng);
    let rows = lengths.iter().sum::<usize>();
    let (slot_ids, base) = slots(sequences, 1);
    let (checkpoint_ids, pool) = checkpoints(&slot_ids, base);
    let data = ConvCase::new(&mut rng, rows, pool);
    let mut op = data.operands(session, rows)?;
    let cu = cu(&lengths);
    op.i32(session, "cu_rows", &cu)?;
    op.i32(session, "slot_ids", &slot_ids)?;
    op.i32(session, "checkpoint_ids", &checkpoint_ids)?;
    op.u32(session, "live_sequences", sequences as u32)?;
    let kernel = raised(module, "chop_linear_qwen38_conv_verify_v1", sequences)?;
    let state_words = HISTORY * CHANNELS as usize;
    let destinations = lengths
        .iter()
        .enumerate()
        .flat_map(|(sequence, &len)| {
            checkpoint_ids[sequence * max..sequence * max + len]
                .iter()
                .copied()
        })
        .filter(|&slot| slot >= 0)
        .map(|slot| {
            let start = slot as usize * state_words * 2;
            start..start + state_words * 2
        });
    let mut replay = Replay::new(session)
        .output(
            "recurrent output",
            &op.buffers["output"],
            WriteMap::of(DType::Bf16).prefix(rows * CHANNELS as usize)?,
        )
        .state(
            "checkpoint pool",
            &op.buffers["state"],
            bytes_u16(&data.pool),
            WriteMap::of(DType::Bf16).byte_ranges(destinations)?,
            Vec::new(),
        );
    replay.eager(|| {
        kernel.launch(
            session,
            &session.stream,
            &op,
            rows as u32,
            sequences as u32,
            true,
        )
    })?;
    let output = op.read_bf16(session, "output", rows * c)?;
    let pool_after = op.read_bf16(session, "state", data.pool.len())?;
    let query = phase_query("chop_linear_qwen38_conv_verify_v1");
    let weight = decode(&data.weight);
    let mut touched = Vec::new();
    for (s, &len) in lengths.iter().enumerate() {
        let start = cu[s] as usize;
        let expected = recurrent_oracle::causal_conv1d(
            query,
            &data.rows(start, len),
            &weight,
            &ConvCase::slot(&data.pool, slot_ids[s]),
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
    report.finish(session)?;
    conc::measure(
        session,
        "M4",
        &kernel.entry.symbol,
        sequences,
        "linear_qwen38_flash_sm120",
        || {
            kernel.launch(
                session,
                &session.stream,
                &op,
                rows as u32,
                sequences as u32,
                true,
            )
        },
        || {
            assert_eq!(output, op.read_bf16(session, "output", output.len())?);
            Ok(())
        },
        |graph| {
            replay.graph(graph)?;
            let mut fields = BTreeMap::new();
            replay.record(&mut fields);
            Ok(fields)
        },
    )?;
    // The input state and checkpoint destinations are disjoint, so replay is stable.
    assert_eq!(output, op.read_bf16(session, "output", output.len())?);
    indexed_continuation_bits(
        session,
        module,
        &kernel,
        &mut op,
        &slot_ids,
        &checkpoint_ids,
        &lengths,
        n * 2,
        c,
    )?;
    Ok(())
}

fn delta_verify(session: &Session, module: &Module, sequences: usize, seed: u64) -> LabResult<()> {
    let case = format!("delta_verify_s{sequences}");
    let mut report = Report::new(case.clone());
    let max = VERIFY_ROWS as usize;
    let mut rng = Rng::new(seed);
    let lengths = verify_rows(sequences, &mut rng);
    let rows = lengths.iter().sum::<usize>();
    let (slot_ids, base) = slots(sequences, 1);
    let (checkpoint_ids, pool) = checkpoints(&slot_ids, base);
    let data = DeltaCase::new(&mut rng, rows, pool);
    let mut op = data.operands(session, rows)?;
    let cu = cu(&lengths);
    op.i32(session, "cu_rows", &cu)?;
    op.i32(session, "slot_ids", &slot_ids)?;
    op.i32(session, "checkpoint_ids", &checkpoint_ids)?;
    op.u32(session, "live_sequences", sequences as u32)?;
    let kernel = raised(module, "chop_linear_qwen38_delta_verify_v1", sequences)?;
    let state_words = VALUE_HEADS as usize * HEAD;
    let destinations = lengths
        .iter()
        .enumerate()
        .flat_map(|(sequence, &len)| {
            checkpoint_ids[sequence * max..sequence * max + len]
                .iter()
                .copied()
        })
        .filter(|&slot| slot >= 0)
        .map(|slot| {
            let start = slot as usize * state_words * 4;
            start..start + state_words * 4
        });
    let mut replay = Replay::new(session)
        .output(
            "recurrent output",
            &op.buffers["output"],
            WriteMap::of(DType::Bf16).prefix(rows * NORM_WIDTH)?,
        )
        .state(
            "checkpoint pool",
            &op.buffers["state"],
            bytes_f32(&data.pool),
            WriteMap::of(DType::F32).byte_ranges(destinations)?,
            Vec::new(),
        );
    replay.eager(|| {
        kernel.launch(
            session,
            &session.stream,
            &op,
            rows as u32,
            sequences as u32,
            true,
        )
    })?;
    let output = op.read_bf16(session, "output", rows * NORM_WIDTH)?;
    let pool_after = op.read_f32(session, "state", data.pool.len())?;
    let query = phase_query("chop_linear_qwen38_delta_verify_v1");
    let n = VALUE_HEADS as usize * HEAD;
    let mut touched = Vec::new();
    for (s, &len) in lengths.iter().enumerate() {
        let start = cu[s] as usize;
        let expected = data.oracle(query, start, len, &data.slot(slot_ids[s]))?;
        report.check(
            &format!("seq{s}_output"),
            &decode(&output[start * NORM_WIDTH..(start + len) * NORM_WIDTH]),
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
    report.finish(session)?;
    conc::measure(
        session,
        "M4",
        &kernel.entry.symbol,
        sequences,
        "linear_qwen38_flash_sm120",
        || {
            kernel.launch(
                session,
                &session.stream,
                &op,
                rows as u32,
                sequences as u32,
                true,
            )
        },
        || {
            assert_eq!(output, op.read_bf16(session, "output", output.len())?);
            Ok(())
        },
        |graph| {
            replay.graph(graph)?;
            let mut fields = BTreeMap::new();
            replay.record(&mut fields);
            Ok(fields)
        },
    )?;
    // The input state and checkpoint destinations are disjoint, so replay is stable.
    assert_eq!(output, op.read_bf16(session, "output", output.len())?);
    indexed_continuation_bits(
        session,
        module,
        &kernel,
        &mut op,
        &slot_ids,
        &checkpoint_ids,
        &lengths,
        n * 4,
        NORM_WIDTH,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn indexed_continuation_bits(
    session: &Session,
    module: &Module,
    verify: &Kernel,
    op: &mut Operands,
    slots: &[i32],
    checkpoints: &[i32],
    lengths: &[usize],
    entry_bytes: usize,
    output_width: usize,
) -> LabResult<()> {
    let after = op.read(session, "state", op.buffers["state"].byte_len())?;
    let decode = raised(
        module,
        &verify.entry.symbol.replace("verify", "decode"),
        slots.len(),
    )?;
    for accept in 0..=VERIFY_ROWS as usize {
        let mut copied = after.clone();
        let mapped: Vec<_> = slots
            .iter()
            .enumerate()
            .map(|(item, &slot)| {
                let kept = accept.min(lengths[item]);
                let entry = if kept == 0 {
                    slot
                } else {
                    checkpoints[item * VERIFY_ROWS as usize + kept - 1]
                };
                let from = entry as usize * entry_bytes;
                let to = slot as usize * entry_bytes;
                copied[to..to + entry_bytes].copy_from_slice(&after[from..from + entry_bytes]);
                entry
            })
            .collect();
        op.buffers["state"]
            .copy_from_bytes(&copied, &session.stream)
            .map_err(|e| e.to_string())?;
        op.i32(session, "slot_ids", slots)?;
        kernel_lab::poison::poison(session, &[&op.buffers["output"]])?;
        decode.launch(
            session,
            &session.stream,
            op,
            slots.len() as u32,
            slots.len() as u32,
            false,
        )?;
        let reference_output = op.read(session, "output", slots.len() * output_width * 2)?;
        let reference_state = op.read(session, "state", after.len())?;
        op.buffers["state"]
            .copy_from_bytes(&after, &session.stream)
            .map_err(|e| e.to_string())?;
        op.i32(session, "slot_ids", &mapped)?;
        kernel_lab::poison::poison(session, &[&op.buffers["output"]])?;
        decode.launch(
            session,
            &session.stream,
            op,
            slots.len() as u32,
            slots.len() as u32,
            true,
        )?;
        if op.read(session, "output", reference_output.len())? != reference_output {
            return Err(format!(
                "index continuation output differs for accept {accept}"
            ));
        }
        let indexed = op.read(session, "state", after.len())?;
        for (&slot, &entry) in slots.iter().zip(&mapped) {
            let left = slot as usize * entry_bytes;
            let right = entry as usize * entry_bytes;
            if reference_state[left..left + entry_bytes] != indexed[right..right + entry_bytes] {
                return Err(format!(
                    "index continuation state differs for accept {accept}"
                ));
            }
        }
    }
    Ok(())
}

#[path = "common/concurrency.rs"]
mod conc;
#[test]
fn m4_qwen_verify_sequences() {
    if !conc::enabled("m4") {
        return;
    }
    assert_eq!(kernel_lab::selected_ordinals().unwrap().unwrap().len(), 1);
    gpu_test(4 << 30, |s| {
        conc::exclusive(s)?;
        let module = module(s)?;
        let mut failures = conc::replay::Failures::default();
        for n in conc::ROWS {
            failures.check(
                &format!("conv rows={n}"),
                conv_verify(s, &module, n, 0xc030 + n as u64),
            );
            failures.check(
                &format!("delta rows={n}"),
                delta_verify(s, &module, n, 0xd030 + n as u64),
            );
        }
        failures.finish()?;
        Ok(())
    })
    .expect("M4 recurrent verify")
    .expect("GPU required");
}
