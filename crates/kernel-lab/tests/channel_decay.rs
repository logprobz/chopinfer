//! The per-channel-decay delta rule engine (`kernels_sm120::linear::
//! channel_decay`, cubin `channel_decay_h16x128_sm120`) against the f64
//! oracles of `kernel_core::per_channel_decay_delta_rule_oracle` on one GPU
//! at the served shape: 16 heads of 128, separate q, k and v convolutions of
//! kernel 4, the bounded gate at -5 and the gated RMSNorm at eps 1e-5.
//!
//! Run with `CHOPINFER_KERNEL_GPU=<ordinal> cargo test -p kernel-lab
//! --release --locked --offline --test channel_decay`; without a GPU every
//! test skips (and fails under `CHOPINFER_KERNEL_REQUIRE=1`).

use kernel_build::EntryParam;
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{Argument, DType, GateActivation, RecurrentPhase, RecurrentQuery};
use kernel_core::oracle::{OracleOutput, compensated_sum, max_bound_use};
use kernel_core::per_channel_decay_delta_rule::{PerChannelDecayDeltaRuleQuery, QkvConvs};
use kernel_core::per_channel_decay_delta_rule_oracle::{
    self as oracle, DeltaRuleInputs, DeltaRuleOutput,
};
use kernel_core::recurrent_oracle;
use kernel_lab::launch::{self, Bindings};
use kernel_lab::poison::{self, POISON, Replay, WriteMap};
use kernel_lab::{LabResult, Session, check_output, gpu_test};
use kernels_sm120::linear::Bucket;
use kernels_sm120::linear::channel_decay::{self, Entry, Query, SERVED, SERVED_PROFILE};
use runtime_cuda::{DeviceBuffer, Function, Module, Stream};
use serde_json::{Value, json};
use std::collections::BTreeMap;

#[path = "common/pool_offsets.rs"]
mod pool_offsets;

use pool_offsets::SentinelPool;

const H: usize = SERVED.heads as usize;
const D: usize = SERVED.head_dim as usize;
const WIDTH: usize = H * D;
const CHANNELS: usize = 3 * WIDTH;
const HISTORY: usize = SERVED.kernel as usize - 1;
const CONV_ENTRY: usize = HISTORY * CHANNELS;
const STATE: usize = H * D * D;
/// Row strides of the inputs, each padded past its payload.
const INPUT_STRIDE: usize = CHANNELS + 64;
const F_STRIDE: usize = WIDTH + 64;
const B_STRIDE: usize = H + 8;
const Z_STRIDE: usize = WIDTH + 128;
const VERIFY_ROWS: usize = SERVED.verify_rows as usize;

// ------------------------------------------------------------ helpers ----

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1_u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform().max(1e-300), self.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
    fn bf16(&mut self, len: usize, mean: f64, scale: f64) -> Vec<u16> {
        (0..len)
            .map(|_| f32_to_bf16((mean + self.normal() * scale) as f32))
            .collect()
    }
    fn f32(&mut self, len: usize, scale: f64) -> Vec<f32> {
        (0..len).map(|_| (self.normal() * scale) as f32).collect()
    }
}

fn decode(bits: &[u16]) -> Vec<f64> {
    bits.iter().map(|&x| f64::from(bf16_to_f32(x))).collect()
}
fn f64s(values: &[f32]) -> Vec<f64> {
    values.iter().map(|&x| f64::from(x)).collect()
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
fn cu(lengths: &[usize]) -> Vec<i32> {
    let mut cu = vec![0_i32];
    for &len in lengths {
        cu.push(cu.last().unwrap() + len as i32);
    }
    cu
}
/// Rows `first..first + count` of a strided BF16 matrix, `width` wide from
/// column `from`.
fn columns(
    data: &[u16],
    stride: usize,
    first: usize,
    count: usize,
    from: usize,
    width: usize,
) -> Vec<f64> {
    (first..first + count)
        .flat_map(|r| decode(&data[r * stride + from..r * stride + from + width]))
        .collect()
}
/// The local row of a call's prefix anchor (`channel_decay_common.cuh`).
fn anchor_row(position: usize, rows: usize) -> Option<usize> {
    let tail = (position + rows) % 64;
    (tail != 0 && rows > tail).then(|| rows - tail)
}

fn symbol(name: &str) -> String {
    format!("chop_channel_decay_{}_{name}_v1", SERVED_PROFILE.tag())
}

fn find(name: &str) -> Entry {
    let symbol = symbol(name);
    channel_decay::catalog()
        .into_iter()
        .find(|entry| entry.symbol == symbol)
        .unwrap_or_else(|| panic!("missing entry {symbol}"))
}

fn rule_query(name: &str) -> PerChannelDecayDeltaRuleQuery {
    match find(name).query {
        Query::Rule(query) => query,
        Query::Norm(_) => panic!("{name} is the norm"),
    }
}

fn module(session: &Session) -> LabResult<Module> {
    let name = channel_decay::cubin_specs().remove(0).name;
    let entries = kernel_lab::cubin_entries(&name)?;
    if entries.len() != channel_decay::catalog().len() {
        return Err(format!("{name} holds {} entries", entries.len()));
    }
    session.module(&name)
}

/// Named device buffers and scalars, bound to entry parameters by name.
#[derive(Default)]
struct Binds<'a> {
    buffers: BTreeMap<&'static str, &'a DeviceBuffer>,
    scalars: BTreeMap<&'static str, u32>,
}

impl<'a> Binds<'a> {
    fn buffer(mut self, name: &'static str, buffer: &'a DeviceBuffer) -> Self {
        self.buffers.insert(name, buffer);
        self
    }
    fn scalar(mut self, name: &'static str, value: u32) -> Self {
        self.scalars.insert(name, value);
        self
    }
}

/// Uploads `bytes` (at least 4) to a new buffer.
fn upload(session: &Session, bytes: &[u8]) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(bytes.len().max(4))?;
    buffer
        .copy_from_bytes(bytes, &session.stream)
        .map_err(|e| e.to_string())?;
    Ok(buffer)
}
fn zeros(session: &Session, bytes: usize) -> LabResult<DeviceBuffer> {
    upload(session, &vec![0; bytes])
}
fn download(session: &Session, buffer: &DeviceBuffer) -> LabResult<Vec<u8>> {
    let mut out = vec![0; buffer.byte_len()];
    buffer
        .copy_to_bytes(&mut out, &session.stream)
        .map_err(|e| e.to_string())?;
    Ok(out)
}

/// A catalog entry resolved in the loaded cubin, planned for `bucket`.
struct Kernel {
    entry: Entry,
    function: Function,
    bucket: Bucket,
}

impl Kernel {
    fn load(module: &Module, name: &str, rows: usize, sequences: usize) -> LabResult<Self> {
        let entry = find(name);
        let function = module.function(&entry.symbol).map_err(|e| e.to_string())?;
        let shared = entry.dynamic_shared_bytes();
        if shared > 0 {
            function
                .set_max_dynamic_shared_memory(shared)
                .map_err(|e| e.to_string())?;
        }
        Ok(Self {
            entry,
            function,
            bucket: Bucket {
                rows: rows as u32,
                sequences: sequences as u32,
            },
        })
    }

    fn workspace_bytes(&self) -> usize {
        self.entry.workspace_bytes(self.bucket) as usize
    }

    fn launch(
        &self,
        session: &Session,
        stream: &Stream,
        binds: &Binds<'_>,
        live_rows: usize,
        live_sequences: usize,
        pdl: bool,
    ) -> LabResult<()> {
        let names = binds.buffers.keys().copied().collect::<Vec<_>>();
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
                EntryParam::U32 { name } => binds
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
            .plan(self.bucket, live_rows as u32, live_sequences as u32)
            .map_err(|e| e.to_string())?;
        plan.pdl = pdl;
        let bindings = Bindings {
            buffers: binds
                .buffers
                .values()
                .enumerate()
                .map(|(i, &b)| (i as u64 + 1, b))
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

/// One receipt per case holding every check's verdict.
struct Report {
    case: String,
    checks: Vec<Value>,
    failures: Vec<String>,
}

impl Report {
    fn new(case: impl Into<String>) -> Self {
        Self {
            case: case.into(),
            checks: Vec::new(),
            failures: Vec::new(),
        }
    }

    /// The numerical contract gate against an oracle output.
    fn check(&mut self, name: &str, actual: &[f64], reference: &OracleOutput, dtype: DType) {
        let mut data = BTreeMap::new();
        let finite = actual.iter().all(|x| x.is_finite());
        let actual = actual.iter().map(|&x| x as f32).collect::<Vec<_>>();
        let correct = finite && check_output(&actual, reference, dtype, &mut data).is_ok();
        let bound_use = max_bound_use(&actual, reference, dtype);
        self.checks.push(json!({
            "name": name,
            "correct": correct,
            "elements": actual.len(),
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
    fn exact(&mut self, name: &str, result: Result<(), String>) {
        self.checks.push(
            json!({"name": name, "correct": result.is_ok(), "detail": result.as_ref().err()}),
        );
        if let Err(error) = result {
            self.failures.push(format!("{name}: {error}"));
        }
    }

    fn finish(self, session: &Session) -> LabResult<()> {
        let mut data = BTreeMap::new();
        data.insert("case".into(), json!(self.case));
        data.insert("gate".into(), json!("G1"));
        data.insert("correct".into(), json!(self.failures.is_empty()));
        data.insert("checks".into(), json!(self.checks));
        session.receipt("channel_decay", data)?;
        if self.failures.is_empty() {
            Ok(())
        } else {
            Err(format!("{}: {}", self.case, self.failures.join("; ")))
        }
    }
}

fn same_bits<T: PartialEq + std::fmt::Debug>(
    name: &str,
    actual: &[T],
    expected: &[T],
) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "{name}: {} versus {} elements",
            actual.len(),
            expected.len()
        ));
    }
    match actual.iter().zip(expected).position(|(a, b)| a != b) {
        None => Ok(()),
        Some(i) => Err(format!(
            "{name}: element {i} is {:?}, expected {:?}",
            actual[i], expected[i]
        )),
    }
}

fn same_f32_bits(name: &str, actual: &[f32], expected: &[f32]) -> Result<(), String> {
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    same_bits(name, &bits(actual), &bits(expected))
}

// ------------------------------------------------------------- operands ----

/// Host operands of one case: the layer's inputs and both state pools.
struct Case {
    /// Convolution input `[rows, INPUT_STRIDE]`.
    input: Vec<u16>,
    /// q, k and v convolution weights `[WIDTH, 4]`, BF16 values in F32.
    weights: [Vec<f32>; 3],
    /// Convolution pool `[entries, HISTORY, CHANNELS]`.
    conv_pool: Vec<u16>,
    /// Delta-rule operands when not fed by the convolution: `qkv` `[rows,
    /// INPUT_STRIDE]`.
    qkv: Vec<u16>,
    f: Vec<u16>,
    b: Vec<u16>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    /// Delta-rule pool `[entries, STATE]`.
    delta_pool: Vec<f32>,
    /// Norm gate `[rows, Z_STRIDE]` and weight `[D]`.
    z: Vec<u16>,
    norm_weight: Vec<u16>,
}

impl Case {
    fn new(seed: u64, rows: usize, entries: usize) -> Self {
        let mut rng = Rng(seed);
        let weight = |rng: &mut Rng| {
            rng.bf16(WIDTH * 4, 0.0, 0.4)
                .into_iter()
                .map(bf16_to_f32)
                .collect::<Vec<_>>()
        };
        Self {
            input: rng.bf16(rows * INPUT_STRIDE, 0.0, 1.0),
            weights: [weight(&mut rng), weight(&mut rng), weight(&mut rng)],
            conv_pool: rng.bf16(entries * CONV_ENTRY, 0.0, 1.0),
            qkv: rng.bf16(rows * INPUT_STRIDE, 0.0, 1.0),
            // Mostly slow decays with some near the bound.
            f: rng.bf16(rows * F_STRIDE, -2.0, 1.5),
            b: rng.bf16(rows * B_STRIDE, 0.0, 2.0),
            a_log: (0..H)
                .map(|_| (1.0 + 15.0 * rng.uniform()).ln() as f32)
                .collect(),
            dt_bias: rng.f32(WIDTH, 1.0),
            delta_pool: rng.f32(entries * STATE, 0.5),
            z: rng.bf16(rows * Z_STRIDE, 0.0, 2.0),
            norm_weight: rng.bf16(D, 1.0, 0.1),
        }
    }

    fn conv_slot(&self, slot: i32) -> Vec<f64> {
        let at = slot as usize * CONV_ENTRY;
        decode(&self.conv_pool[at..at + CONV_ENTRY])
    }
    fn delta_slot(&self, slot: i32) -> Vec<f64> {
        let at = slot as usize * STATE;
        f64s(&self.delta_pool[at..at + STATE])
    }

    fn conv_oracle(
        &self,
        phase: RecurrentPhase,
        first: usize,
        count: usize,
        state: &[f64],
    ) -> LabResult<recurrent_oracle::ConvOutput> {
        let query = PerChannelDecayDeltaRuleQuery::ShortConv {
            heads: H as u32,
            head_dim: D as u32,
            kernel: SERVED.kernel,
            convs: QkvConvs::Separate,
            phase,
            capacity: 8192,
        };
        let input = columns(&self.input, INPUT_STRIDE, first, count, 0, CHANNELS);
        let weights = self.weights.iter().map(|w| f64s(w)).collect::<Vec<_>>();
        let parts = weights.iter().map(Vec::as_slice).collect::<Vec<_>>();
        oracle::short_conv(query, &input, &parts, state).map_err(|e| e.to_string())
    }

    /// The recurrence over rows `first..first + count` of `qkv` (stride
    /// `stride`) from `state`.
    fn delta_oracle(
        &self,
        query: PerChannelDecayDeltaRuleQuery,
        qkv: &[u16],
        stride: usize,
        first: usize,
        count: usize,
        state: &[f64],
    ) -> LabResult<DeltaRuleOutput> {
        let (q, k, v, b, f) = self.delta_operands(qkv, stride, first, count);
        let (a_log, dt_bias) = (f64s(&self.a_log), f64s(&self.dt_bias));
        oracle::delta_rule(
            query,
            DeltaRuleInputs {
                q: &q,
                k: &k,
                v: &v,
                b: &b,
                f: &f,
                a_log: &a_log,
                dt_bias: &dt_bias,
                state,
            },
        )
        .map_err(|e| e.to_string())
    }

    #[allow(clippy::type_complexity)]
    fn delta_operands(
        &self,
        qkv: &[u16],
        stride: usize,
        first: usize,
        count: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        (
            columns(qkv, stride, first, count, 0, WIDTH),
            columns(qkv, stride, first, count, WIDTH, WIDTH),
            columns(qkv, stride, first, count, 2 * WIDTH, WIDTH),
            columns(&self.b, B_STRIDE, first, count, 0, H),
            columns(&self.f, F_STRIDE, first, count, 0, WIDTH),
        )
    }

    fn norm_oracle(&self, input: &[u16], rows: usize) -> LabResult<OracleOutput> {
        let Query::Norm(query) = find("gated_norm").query else {
            return Err("gated_norm is not the norm".into());
        };
        recurrent_oracle::gated_rmsnorm(
            query,
            &decode(&input[..rows * WIDTH]),
            &columns(&self.z, Z_STRIDE, 0, rows, 0, WIDTH),
            &decode(&self.norm_weight),
        )
        .map_err(|e| e.to_string())
    }
}

/// Device buffers of a case, named by the entry parameters that read them.
struct Device {
    input: DeviceBuffer,
    weight_q: DeviceBuffer,
    weight_k: DeviceBuffer,
    weight_v: DeviceBuffer,
    conv_state: DeviceBuffer,
    qkv: DeviceBuffer,
    f: DeviceBuffer,
    b: DeviceBuffer,
    a_log: DeviceBuffer,
    dt_bias: DeviceBuffer,
    delta_state: DeviceBuffer,
    z: DeviceBuffer,
    norm_weight: DeviceBuffer,
    /// Convolution output `[rows, CHANNELS]`, the delta rule's input in a
    /// layer chain.
    conv_out: DeviceBuffer,
    /// Delta-rule output `[rows, WIDTH]`.
    core: DeviceBuffer,
    /// Norm output `[rows, WIDTH]`.
    normed: DeviceBuffer,
}

impl Device {
    fn new(session: &Session, case: &Case, rows: usize) -> LabResult<Self> {
        Ok(Self {
            input: upload(session, &bytes_u16(&case.input))?,
            weight_q: upload(session, &bytes_f32(&case.weights[0]))?,
            weight_k: upload(session, &bytes_f32(&case.weights[1]))?,
            weight_v: upload(session, &bytes_f32(&case.weights[2]))?,
            conv_state: upload(session, &bytes_u16(&case.conv_pool))?,
            qkv: upload(session, &bytes_u16(&case.qkv))?,
            f: upload(session, &bytes_u16(&case.f))?,
            b: upload(session, &bytes_u16(&case.b))?,
            a_log: upload(session, &bytes_f32(&case.a_log))?,
            dt_bias: upload(session, &bytes_f32(&case.dt_bias))?,
            delta_state: upload(session, &bytes_f32(&case.delta_pool))?,
            z: upload(session, &bytes_u16(&case.z))?,
            norm_weight: upload(session, &bytes_u16(&case.norm_weight))?,
            conv_out: zeros(session, rows * CHANNELS * 2)?,
            core: zeros(session, rows * WIDTH * 2)?,
            normed: zeros(session, rows * WIDTH * 2)?,
        })
    }

    fn conv(&self) -> Binds<'_> {
        Binds::default()
            .buffer("input", &self.input)
            .buffer("weight_q", &self.weight_q)
            .buffer("weight_k", &self.weight_k)
            .buffer("weight_v", &self.weight_v)
            .buffer("state", &self.conv_state)
            .buffer("output", &self.conv_out)
            .scalar("input_stride", INPUT_STRIDE as u32)
    }

    /// The delta rule over `qkv` (`chained` reads the convolution output).
    fn delta(&self, chained: bool) -> Binds<'_> {
        let (qkv, stride) = if chained {
            (&self.conv_out, CHANNELS)
        } else {
            (&self.qkv, INPUT_STRIDE)
        };
        Binds::default()
            .buffer("qkv", qkv)
            .buffer("f", &self.f)
            .buffer("b", &self.b)
            .buffer("a_log", &self.a_log)
            .buffer("dt_bias", &self.dt_bias)
            .buffer("state", &self.delta_state)
            .buffer("output", &self.core)
            .scalar("qkv_stride", stride as u32)
            .scalar("f_stride", F_STRIDE as u32)
            .scalar("b_stride", B_STRIDE as u32)
    }

    fn norm(&self) -> Binds<'_> {
        Binds::default()
            .buffer("input", &self.core)
            .buffer("gate", &self.z)
            .buffer("weight", &self.norm_weight)
            .buffer("output", &self.normed)
            .scalar("gate_stride", Z_STRIDE as u32)
    }
}

/// Per-sequence metadata of a launch.
struct Metadata {
    cu: DeviceBuffer,
    start_positions: DeviceBuffer,
    slot_ids: DeviceBuffer,
    initial: DeviceBuffer,
    anchor_ids: DeviceBuffer,
    checkpoint_ids: DeviceBuffer,
    live_sequences: DeviceBuffer,
    live_rows: DeviceBuffer,
    workspace: DeviceBuffer,
}

struct Batch<'a> {
    lengths: &'a [usize],
    positions: &'a [usize],
    slots: &'a [i32],
    initial: &'a [bool],
    anchors: &'a [i32],
    /// Verify: `VERIFY_ROWS` checkpoint entries per sequence.
    checkpoints: &'a [i32],
}

impl Metadata {
    fn new(session: &Session, batch: &Batch<'_>, workspace: usize) -> LabResult<Self> {
        let rows = batch.lengths.iter().sum::<usize>();
        Ok(Self {
            cu: upload(session, &bytes_i32(&cu(batch.lengths)))?,
            start_positions: upload(
                session,
                &bytes_i32(
                    &batch
                        .positions
                        .iter()
                        .map(|&p| p as i32)
                        .collect::<Vec<_>>(),
                ),
            )?,
            slot_ids: upload(session, &bytes_i32(batch.slots))?,
            initial: upload(
                session,
                &bytes_i32(
                    &batch
                        .initial
                        .iter()
                        .map(|&x| i32::from(x))
                        .collect::<Vec<_>>(),
                ),
            )?,
            anchor_ids: upload(session, &bytes_i32(batch.anchors))?,
            checkpoint_ids: upload(session, &bytes_i32(batch.checkpoints))?,
            live_sequences: upload(session, &(batch.lengths.len() as u32).to_ne_bytes())?,
            live_rows: upload(session, &(rows as u32).to_ne_bytes())?,
            workspace: zeros(session, workspace)?,
        })
    }

    fn bind<'a>(&'a self, binds: Binds<'a>, phase: RecurrentPhase) -> Binds<'a> {
        let binds = binds
            .buffer("slot_ids", &self.slot_ids)
            .buffer("live_sequences", &self.live_sequences)
            .buffer("live_rows", &self.live_rows)
            .buffer("workspace", &self.workspace);
        match phase {
            RecurrentPhase::Prefill => binds
                .buffer("cu_seqlens", &self.cu)
                .buffer("start_positions", &self.start_positions)
                .buffer("initial", &self.initial)
                .buffer("anchor_ids", &self.anchor_ids),
            RecurrentPhase::Decode => binds,
            RecurrentPhase::Verify { .. } => binds
                .buffer("cu_rows", &self.cu)
                .buffer("checkpoint_ids", &self.checkpoint_ids),
        }
    }
}

/// The entries of one phase, planned for a bucket exactly the batch.
struct Layer {
    conv: Kernel,
    delta: Vec<Kernel>,
    norm: Kernel,
    phase: RecurrentPhase,
}

#[allow(clippy::too_many_arguments)]
impl Layer {
    fn load(
        module: &Module,
        phase: RecurrentPhase,
        rows: usize,
        sequences: usize,
    ) -> LabResult<Self> {
        let name = match phase {
            RecurrentPhase::Prefill => "prefill",
            RecurrentPhase::Decode => "decode",
            RecurrentPhase::Verify { .. } => "verify",
        };
        let delta = if phase == RecurrentPhase::Prefill {
            vec![
                Kernel::load(module, "delta_prefill_prepare", rows, sequences)?,
                Kernel::load(module, "delta_prefill_scan", rows, sequences)?,
            ]
        } else {
            vec![Kernel::load(
                module,
                &format!("delta_{name}"),
                rows,
                sequences,
            )?]
        };
        Ok(Self {
            conv: Kernel::load(module, &format!("conv_{name}"), rows, sequences)?,
            delta,
            norm: Kernel::load(module, "gated_norm", rows, rows)?,
            phase,
        })
    }

    fn workspace_bytes(&self) -> usize {
        self.delta[0].workspace_bytes()
    }

    fn launch_conv(
        &self,
        s: &Session,
        stream: &Stream,
        d: &Device,
        m: &Metadata,
        rows: usize,
        seqs: usize,
        pdl: bool,
    ) -> LabResult<()> {
        self.conv
            .launch(s, stream, &m.bind(d.conv(), self.phase), rows, seqs, pdl)
    }

    fn launch_delta(
        &self,
        s: &Session,
        stream: &Stream,
        d: &Device,
        m: &Metadata,
        chained: bool,
        rows: usize,
        seqs: usize,
        pdl: bool,
    ) -> LabResult<()> {
        for kernel in &self.delta {
            kernel.launch(
                s,
                stream,
                &m.bind(d.delta(chained), self.phase),
                rows,
                seqs,
                pdl,
            )?;
        }
        Ok(())
    }

    fn launch_norm(
        &self,
        s: &Session,
        stream: &Stream,
        d: &Device,
        m: &Metadata,
        rows: usize,
        pdl: bool,
    ) -> LabResult<()> {
        self.norm
            .launch(s, stream, &m.bind(d.norm(), self.phase), rows, rows, pdl)
    }
}

// ------------------------------------------------------- oracle cases ----

/// A prefill batch through the convolution, then the delta rule over the
/// convolution's output, then the norm; every output and state against the
/// oracles over the device's own inputs.
fn prefill_case(
    session: &Session,
    module: &Module,
    name: &str,
    batch: &Batch<'_>,
    seed: u64,
) -> LabResult<()> {
    let mut report = Report::new(name);
    let rows = batch.lengths.iter().sum::<usize>();
    let sequences = batch.lengths.len();
    let entries = 2 * sequences + 4;
    let case = Case::new(seed, rows, entries);
    let device = Device::new(session, &case, rows)?;
    let layer = Layer::load(module, RecurrentPhase::Prefill, rows, sequences)?;
    let meta = Metadata::new(session, batch, layer.workspace_bytes())?;
    let stream = &session.stream;
    layer.launch_conv(session, stream, &device, &meta, rows, sequences, true)?;
    layer.launch_delta(session, stream, &device, &meta, true, rows, sequences, true)?;
    layer.launch_norm(session, stream, &device, &meta, rows, true)?;
    let conv_out = u16s(&download(session, &device.conv_out)?);
    let core = u16s(&download(session, &device.core)?);
    let normed = u16s(&download(session, &device.normed)?);
    let conv_pool = u16s(&download(session, &device.conv_state)?);
    let delta_pool = f32s(&download(session, &device.delta_state)?);
    let starts = cu(batch.lengths);
    let query = rule_query("delta_prefill_prepare");
    for (s, &len) in batch.lengths.iter().enumerate() {
        let start = starts[s] as usize;
        let slot = batch.slots[s];
        let conv_state = if batch.initial[s] {
            case.conv_slot(slot)
        } else {
            vec![0.0; CONV_ENTRY]
        };
        let conv = case.conv_oracle(RecurrentPhase::Prefill, start, len, &conv_state)?;
        report.check(
            &format!("seq{s}_conv"),
            &decode(&conv_out[start * CHANNELS..(start + len) * CHANNELS]),
            &conv.output,
            DType::Bf16,
        );
        let at = slot as usize * CONV_ENTRY;
        report.exact(
            &format!("seq{s}_conv_state"),
            same_bits(
                "conv state",
                &decode(&conv_pool[at..at + CONV_ENTRY]),
                &conv.state,
            ),
        );
        let delta_state = if batch.initial[s] {
            case.delta_slot(slot)
        } else {
            vec![0.0; STATE]
        };
        // The oracle reads the device's convolution output, so each stage is
        // judged on its own inputs.
        let delta = case.delta_oracle(query, &conv_out, CHANNELS, start, len, &delta_state)?;
        report.check(
            &format!("seq{s}_delta"),
            &decode(&core[start * WIDTH..(start + len) * WIDTH]),
            &delta.output,
            DType::Bf16,
        );
        let at = slot as usize * STATE;
        report.check(
            &format!("seq{s}_delta_state"),
            &f64s(&delta_pool[at..at + STATE]),
            &delta.state,
            DType::F32,
        );
        let anchor = batch.anchors[s];
        if let (true, Some(row)) = (anchor >= 0, anchor_row(batch.positions[s], len)) {
            let conv = case.conv_oracle(RecurrentPhase::Prefill, start, row, &conv_state)?;
            let at = anchor as usize * CONV_ENTRY;
            report.exact(
                &format!("seq{s}_conv_anchor"),
                same_bits(
                    "conv anchor",
                    &decode(&conv_pool[at..at + CONV_ENTRY]),
                    &conv.state,
                ),
            );
            let delta = case.delta_oracle(query, &conv_out, CHANNELS, start, row, &delta_state)?;
            let at = anchor as usize * STATE;
            report.check(
                &format!("seq{s}_delta_anchor"),
                &f64s(&delta_pool[at..at + STATE]),
                &delta.state,
                DType::F32,
            );
        } else if anchor >= 0 {
            // A call that ends on a multiple of 64 (or holds none) leaves
            // its anchor entry alone.
            let id = anchor as usize;
            report.exact(
                &format!("seq{s}_conv_anchor_untouched"),
                same_bits(
                    "conv anchor",
                    &conv_pool[id * CONV_ENTRY..(id + 1) * CONV_ENTRY],
                    &case.conv_pool[id * CONV_ENTRY..(id + 1) * CONV_ENTRY],
                ),
            );
            report.exact(
                &format!("seq{s}_delta_anchor_untouched"),
                same_f32_bits(
                    "delta anchor",
                    &delta_pool[id * STATE..(id + 1) * STATE],
                    &case.delta_pool[id * STATE..(id + 1) * STATE],
                ),
            );
        }
    }
    let norm = case.norm_oracle(&core, rows)?;
    report.check("norm", &decode(&normed[..rows * WIDTH]), &norm, DType::Bf16);
    report.finish(session)
}

/// Decode (one row per sequence) or verify (rows per sequence into
/// checkpoints) through the three stages against the oracles.
fn recurrent_case(
    session: &Session,
    module: &Module,
    lengths: &[usize],
    seed: u64,
) -> LabResult<()> {
    let verify = lengths.iter().any(|&len| len != 1);
    let phase = if verify {
        RecurrentPhase::Verify {
            max_rows: VERIFY_ROWS as u32,
        }
    } else {
        RecurrentPhase::Decode
    };
    let mut report = Report::new(format!(
        "{}_{lengths:?}",
        if verify { "verify" } else { "decode" }
    ));
    let rows = lengths.iter().sum::<usize>();
    let sequences = lengths.len();
    let slots = (0..sequences)
        .map(|s| (2 * sequences - 1 - 2 * s) as i32)
        .collect::<Vec<_>>();
    let checkpoints = (0..sequences * VERIFY_ROWS)
        .map(|i| (2 * sequences + i) as i32)
        .collect::<Vec<_>>();
    let entries = 2 * sequences + checkpoints.len() + 1;
    let case = Case::new(seed, rows, entries);
    let device = Device::new(session, &case, rows)?;
    let batch = Batch {
        lengths,
        positions: &vec![0; sequences],
        slots: &slots,
        initial: &vec![true; sequences],
        anchors: &vec![-1; sequences],
        checkpoints: &checkpoints,
    };
    let bucket_rows = if verify {
        sequences * VERIFY_ROWS
    } else {
        rows
    };
    let layer = Layer::load(module, phase, bucket_rows, sequences)?;
    let meta = Metadata::new(session, &batch, 0)?;
    let stream = &session.stream;
    layer.launch_conv(session, stream, &device, &meta, rows, sequences, true)?;
    layer.launch_delta(session, stream, &device, &meta, true, rows, sequences, true)?;
    layer.launch_norm(session, stream, &device, &meta, rows, true)?;
    let conv_out = u16s(&download(session, &device.conv_out)?);
    let core = u16s(&download(session, &device.core)?);
    let normed = u16s(&download(session, &device.normed)?);
    let conv_pool = u16s(&download(session, &device.conv_state)?);
    let delta_pool = f32s(&download(session, &device.delta_state)?);
    let starts = cu(lengths);
    let query = match phase {
        RecurrentPhase::Decode => rule_query("delta_decode"),
        _ => rule_query("delta_verify"),
    };
    for (s, &len) in lengths.iter().enumerate() {
        let start = starts[s] as usize;
        let conv = case.conv_oracle(phase, start, len, &case.conv_slot(slots[s]))?;
        report.check(
            &format!("seq{s}_conv"),
            &decode(&conv_out[start * CHANNELS..(start + len) * CHANNELS]),
            &conv.output,
            DType::Bf16,
        );
        let delta = case.delta_oracle(
            query,
            &conv_out,
            CHANNELS,
            start,
            len,
            &case.delta_slot(slots[s]),
        )?;
        report.check(
            &format!("seq{s}_delta"),
            &decode(&core[start * WIDTH..(start + len) * WIDTH]),
            &delta.output,
            DType::Bf16,
        );
        if verify {
            for r in 0..len {
                let id = checkpoints[s * VERIFY_ROWS + r] as usize;
                report.exact(
                    &format!("seq{s}_row{r}_conv_checkpoint"),
                    same_bits(
                        "conv checkpoint",
                        &decode(&conv_pool[id * CONV_ENTRY..(id + 1) * CONV_ENTRY]),
                        &conv.row_states[r],
                    ),
                );
                report.check(
                    &format!("seq{s}_row{r}_delta_checkpoint"),
                    &f64s(&delta_pool[id * STATE..(id + 1) * STATE]),
                    &delta.row_states[r],
                    DType::F32,
                );
            }
            // The committed slots stay as they were.
            let slot = slots[s] as usize;
            report.exact(
                &format!("seq{s}_source_unchanged"),
                same_f32_bits(
                    "delta source",
                    &delta_pool[slot * STATE..(slot + 1) * STATE],
                    &case.delta_pool[slot * STATE..(slot + 1) * STATE],
                ),
            );
        } else {
            let slot = slots[s] as usize;
            report.exact(
                &format!("seq{s}_conv_state"),
                same_bits(
                    "conv state",
                    &decode(&conv_pool[slot * CONV_ENTRY..(slot + 1) * CONV_ENTRY]),
                    &conv.state,
                ),
            );
            report.check(
                &format!("seq{s}_delta_state"),
                &f64s(&delta_pool[slot * STATE..(slot + 1) * STATE]),
                &delta.state,
                DType::F32,
            );
        }
    }
    let norm = case.norm_oracle(&core, rows)?;
    report.check("norm", &decode(&normed[..rows * WIDTH]), &norm, DType::Bf16);
    report.finish(session)
}

#[test]
fn channel_decay_prefill_matches_oracle_gpu() {
    gpu_test(8 << 30, |session| {
        let module = module(session)?;
        for (i, &len) in [1_usize, 63, 64, 65, 300, 2048].iter().enumerate() {
            prefill_case(
                session,
                &module,
                &format!("prefill_{len}"),
                &Batch {
                    lengths: &[len],
                    positions: &[0],
                    slots: &[1],
                    initial: &[true],
                    anchors: &[-1],
                    checkpoints: &[],
                },
                0x0e00 + i as u64,
            )?;
        }
        // Mixed lengths, absolute start positions off multiples of 64, zero
        // and carried states, and anchors.
        prefill_case(
            session,
            &module,
            "prefill_mixed",
            &Batch {
                lengths: &[2, 65, 1, 130, 64, 100],
                positions: &[0, 37, 64, 5, 128, 100],
                slots: &[11, 9, 7, 5, 3, 1],
                initial: &[true, false, true, true, false, true],
                anchors: &[-1, 12, -1, 13, 14, 15],
                checkpoints: &[],
            },
            0x0e10,
        )
    })
    .expect("per-channel-decay delta rule prefill matches the oracles");
}

#[test]
fn channel_decay_decode_and_verify_match_oracle_gpu() {
    gpu_test(4 << 30, |session| {
        let module = module(session)?;
        for batch in [1, 8, 32] {
            recurrent_case(session, &module, &vec![1; batch], 0x0d00 + batch as u64)?;
        }
        recurrent_case(session, &module, &[1, 3, 8, 5], 0x0d40)?;
        recurrent_case(session, &module, &[8; 16], 0x0d41)
    })
    .expect("per-channel-decay delta rule decode and verify match the oracles");
}

// ---------------------------------------------- verify against decode ----

/// Verify of `rows` per sequence equals that many one-row decode steps bit
/// for bit: every output row and every checkpoint, so a commit of `kept`
/// rows that keeps checkpoint `kept - 1` continues exactly as decode would.
#[test]
fn channel_decay_verify_checkpoints_equal_decode_steps_gpu() {
    gpu_test(4 << 30, |session| {
        let module = module(session)?;
        let mut report = Report::new("verify_equals_decode");
        let lengths = [VERIFY_ROWS, 5, 1];
        let sequences = lengths.len();
        let rows = lengths.iter().sum::<usize>();
        let slots = [0, 1, 2];
        let checkpoints = (0..sequences * VERIFY_ROWS)
            .map(|i| (3 + i) as i32)
            .collect::<Vec<_>>();
        let case = Case::new(0x5e5e, rows, 3 + checkpoints.len());
        let verify_phase = RecurrentPhase::Verify {
            max_rows: VERIFY_ROWS as u32,
        };
        let device = Device::new(session, &case, rows)?;
        let batch = Batch {
            lengths: &lengths,
            positions: &[0; 3],
            slots: &slots,
            initial: &[true; 3],
            anchors: &[-1; 3],
            checkpoints: &checkpoints,
        };
        let layer = Layer::load(&module, verify_phase, sequences * VERIFY_ROWS, sequences)?;
        let meta = Metadata::new(session, &batch, 0)?;
        layer.launch_conv(
            session,
            &session.stream,
            &device,
            &meta,
            rows,
            sequences,
            true,
        )?;
        layer.launch_delta(
            session,
            &session.stream,
            &device,
            &meta,
            true,
            rows,
            sequences,
            true,
        )?;
        let conv_out = u16s(&download(session, &device.conv_out)?);
        let core = u16s(&download(session, &device.core)?);
        let conv_pool = u16s(&download(session, &device.conv_state)?);
        let delta_pool = f32s(&download(session, &device.delta_state)?);
        // Decode the same rows one at a time from the committed slots, with
        // the verify inputs moved to row 0 of fresh buffers.
        let starts = cu(&lengths);
        let decode_layer = Layer::load(&module, RecurrentPhase::Decode, 1, 1)?;
        for (s, &len) in lengths.iter().enumerate() {
            let mut step = Case::new(0x5e5e, rows, 3 + checkpoints.len());
            for r in 0..len {
                let row = starts[s] as usize + r;
                step.input[..INPUT_STRIDE]
                    .copy_from_slice(&case.input[row * INPUT_STRIDE..(row + 1) * INPUT_STRIDE]);
                step.f[..F_STRIDE].copy_from_slice(&case.f[row * F_STRIDE..(row + 1) * F_STRIDE]);
                step.b[..B_STRIDE].copy_from_slice(&case.b[row * B_STRIDE..(row + 1) * B_STRIDE]);
                if r == 0 {
                    step.conv_pool = case.conv_pool.clone();
                    step.delta_pool = case.delta_pool.clone();
                }
                let one = Device::new(session, &step, 1)?;
                let meta = Metadata::new(
                    session,
                    &Batch {
                        lengths: &[1],
                        positions: &[0],
                        slots: &[slots[s]],
                        initial: &[true],
                        anchors: &[-1],
                        checkpoints: &[],
                    },
                    0,
                )?;
                decode_layer.launch_conv(session, &session.stream, &one, &meta, 1, 1, true)?;
                decode_layer.launch_delta(
                    session,
                    &session.stream,
                    &one,
                    &meta,
                    true,
                    1,
                    1,
                    true,
                )?;
                let label = format!("seq{s}_row{r}");
                report.exact(
                    &format!("{label}_conv_output"),
                    same_bits(
                        "conv output",
                        &u16s(&download(session, &one.conv_out)?),
                        &conv_out[row * CHANNELS..(row + 1) * CHANNELS],
                    ),
                );
                report.exact(
                    &format!("{label}_delta_output"),
                    same_bits(
                        "delta output",
                        &u16s(&download(session, &one.core)?),
                        &core[row * WIDTH..(row + 1) * WIDTH],
                    ),
                );
                step.conv_pool = u16s(&download(session, &one.conv_state)?);
                step.delta_pool = f32s(&download(session, &one.delta_state)?);
                let slot = slots[s] as usize;
                let id = checkpoints[s * VERIFY_ROWS + r] as usize;
                report.exact(
                    &format!("{label}_conv_checkpoint"),
                    same_bits(
                        "conv checkpoint",
                        &step.conv_pool[slot * CONV_ENTRY..(slot + 1) * CONV_ENTRY],
                        &conv_pool[id * CONV_ENTRY..(id + 1) * CONV_ENTRY],
                    ),
                );
                report.exact(
                    &format!("{label}_delta_checkpoint"),
                    same_f32_bits(
                        "delta checkpoint",
                        &step.delta_pool[slot * STATE..(slot + 1) * STATE],
                        &delta_pool[id * STATE..(id + 1) * STATE],
                    ),
                );
            }
        }
        report.finish(session)
    })
    .expect("verify rows and checkpoints equal one-row decode steps bit for bit");
}

// ------------------------------------------------- chunk invariance ----

/// Host state of one sequence carried between prefill calls.
struct Carried {
    conv: Vec<u16>,
    delta: Vec<f32>,
}

/// One prefill call over rows `range` of `case` (one sequence at absolute
/// position `range.start`) from `carried` (None: zero state), with an
/// anchor entry. Returns the convolution and delta outputs, the final
/// states and the anchor states.
fn prefill_call(
    session: &Session,
    module: &Module,
    case: &Case,
    range: std::ops::Range<usize>,
    carried: Option<&Carried>,
) -> LabResult<(Vec<u16>, Vec<u16>, Carried, Carried)> {
    let len = range.len();
    let mut call = Case::new(0, len, 2);
    call.input = case.input[range.start * INPUT_STRIDE..range.end * INPUT_STRIDE].to_vec();
    call.f = case.f[range.start * F_STRIDE..range.end * F_STRIDE].to_vec();
    call.b = case.b[range.start * B_STRIDE..range.end * B_STRIDE].to_vec();
    call.weights = case.weights.clone();
    call.a_log = case.a_log.clone();
    call.dt_bias = case.dt_bias.clone();
    call.conv_pool = vec![POISON as u16 * 0x101; 2 * CONV_ENTRY];
    call.delta_pool = vec![f32::NAN; 2 * STATE];
    if let Some(carried) = carried {
        call.conv_pool[..CONV_ENTRY].copy_from_slice(&carried.conv);
        call.delta_pool[..STATE].copy_from_slice(&carried.delta);
    }
    let device = Device::new(session, &call, len)?;
    let layer = Layer::load(module, RecurrentPhase::Prefill, len, 1)?;
    let meta = Metadata::new(
        session,
        &Batch {
            lengths: &[len],
            positions: &[range.start],
            slots: &[0],
            initial: &[carried.is_some()],
            anchors: &[1],
            checkpoints: &[],
        },
        layer.workspace_bytes(),
    )?;
    layer.launch_conv(session, &session.stream, &device, &meta, len, 1, true)?;
    layer.launch_delta(session, &session.stream, &device, &meta, true, len, 1, true)?;
    let conv_pool = u16s(&download(session, &device.conv_state)?);
    let delta_pool = f32s(&download(session, &device.delta_state)?);
    Ok((
        u16s(&download(session, &device.conv_out)?),
        u16s(&download(session, &device.core)?),
        Carried {
            conv: conv_pool[..CONV_ENTRY].to_vec(),
            delta: delta_pool[..STATE].to_vec(),
        },
        Carried {
            conv: conv_pool[CONV_ENTRY..].to_vec(),
            delta: delta_pool[STATE..].to_vec(),
        },
    ))
}

/// Prefill over consecutive calls ending at `splits`.
fn split_prefill(
    session: &Session,
    module: &Module,
    case: &Case,
    rows: usize,
    splits: &[usize],
) -> LabResult<(Vec<u16>, Vec<u16>, Carried)> {
    let mut bounds = vec![0];
    bounds.extend_from_slice(splits);
    bounds.push(rows);
    let (mut conv, mut core, mut carried) = (Vec::new(), Vec::new(), None);
    for pair in bounds.windows(2) {
        let (c, o, state, _) =
            prefill_call(session, module, case, pair[0]..pair[1], carried.as_ref())?;
        conv.extend(c);
        core.extend(o);
        carried = Some(state);
    }
    Ok((conv, core, carried.expect("at least one call")))
}

/// Calls that end on multiples of 64 give a cold prefill's bits, a prefix
/// hit restored from the anchor entry and continued gives them too, and a
/// split anywhere else agrees within the oracle bound.
#[test]
fn channel_decay_prefill_is_chunk_invariant_gpu() {
    gpu_test(4 << 30, |session| {
        let module = module(session)?;
        let mut report = Report::new("chunk_invariance");
        let rows = 300;
        let case = Case::new(0xc0de, rows, 2);
        let (cold_conv, cold_core, cold_state, anchor) =
            prefill_call(session, &module, &case, 0..rows, None)?;
        for splits in [&[64][..], &[128, 192], &[64, 128, 192, 256]] {
            let (conv, core, state) = split_prefill(session, &module, &case, rows, splits)?;
            let label = format!("split{splits:?}");
            report.exact(
                &format!("{label}_conv"),
                same_bits("conv", &conv, &cold_conv),
            );
            report.exact(
                &format!("{label}_delta"),
                same_bits("delta", &core, &cold_core),
            );
            report.exact(
                &format!("{label}_conv_state"),
                same_bits("conv state", &state.conv, &cold_state.conv),
            );
            report.exact(
                &format!("{label}_delta_state"),
                same_f32_bits("delta state", &state.delta, &cold_state.delta),
            );
        }
        // The cold call ends at 300, so its anchor holds the state at 256: a
        // prefix hit there continues to the cold bits.
        let (conv, core, state, _) =
            prefill_call(session, &module, &case, 256..rows, Some(&anchor))?;
        report.exact(
            "prefix_hit_conv",
            same_bits("conv", &conv, &cold_conv[256 * CHANNELS..]),
        );
        report.exact(
            "prefix_hit_delta",
            same_bits("delta", &core, &cold_core[256 * WIDTH..]),
        );
        report.exact(
            "prefix_hit_conv_state",
            same_bits("conv state", &state.conv, &cold_state.conv),
        );
        report.exact(
            "prefix_hit_delta_state",
            same_f32_bits("delta state", &state.delta, &cold_state.delta),
        );
        // Splits off multiples of 64 regroup one chunk; the convolution keeps
        // its bits and the delta rule meets the oracle.
        let query = rule_query("delta_prefill_prepare");
        for split in [1, 37, 100, 255] {
            let (conv, core, state) = split_prefill(session, &module, &case, rows, &[split])?;
            report.exact(
                &format!("split{split}_conv"),
                same_bits("conv", &conv, &cold_conv),
            );
            let expected =
                case.delta_oracle(query, &cold_conv, CHANNELS, 0, rows, &vec![0.0; STATE])?;
            report.check(
                &format!("split{split}_delta"),
                &decode(&core),
                &expected.output,
                DType::Bf16,
            );
            report.check(
                &format!("split{split}_delta_state"),
                &f64s(&state.delta),
                &expected.state,
                DType::F32,
            );
        }
        report.finish(session)
    })
    .expect("per-channel-decay delta rule prefill is chunk invariant");
}

// ------------------------------------------------------ gate bound ----

/// Every channel decays by the lower bound, -5 per step: a chunk's running
/// sum reaches -320, where any exp(-G) factor would overflow F32.
#[test]
fn channel_decay_prefill_at_the_gate_bound_gpu() {
    gpu_test(4 << 30, |session| {
        let module = module(session)?;
        let rows = 130;
        let mut case = Case::new(0xb0b0, rows, 2);
        // a = 10 and x = 30: sigmoid(300) is 1 in F32 and f64.
        case.f.iter_mut().for_each(|f| *f = f32_to_bf16(30.0));
        case.dt_bias.iter_mut().for_each(|b| *b = 0.0);
        case.a_log.iter_mut().for_each(|a| *a = 10_f32.ln());
        let mut report = Report::new("gate_bound");
        let device = Device::new(session, &case, rows)?;
        let layer = Layer::load(&module, RecurrentPhase::Prefill, rows, 1)?;
        let meta = Metadata::new(
            session,
            &Batch {
                lengths: &[rows],
                positions: &[0],
                slots: &[0],
                initial: &[true],
                anchors: &[-1],
                checkpoints: &[],
            },
            layer.workspace_bytes(),
        )?;
        layer.launch_delta(
            session,
            &session.stream,
            &device,
            &meta,
            false,
            rows,
            1,
            true,
        )?;
        let core = u16s(&download(session, &device.core)?);
        let pool = f32s(&download(session, &device.delta_state)?);
        let expected = case.delta_oracle(
            rule_query("delta_prefill_prepare"),
            &case.qkv,
            INPUT_STRIDE,
            0,
            rows,
            &case.delta_slot(0),
        )?;
        report.check("output", &decode(&core), &expected.output, DType::Bf16);
        report.check("state", &f64s(&pool[..STATE]), &expected.state, DType::F32);
        report.finish(session)
    })
    .expect("the chunked prefill holds at the gate bound");
}

// ------------------------------------------- 128-K accumulation test ----

/// The longest chain of the engine is a 128-key product: `W S^T` and
/// `QD S^T` in the scan, `S k` and `S q` in decode. One row with zero decay
/// (g underflows to -0, so every exp factor is 1) makes the state update
/// `S + k beta (v - S k)` and the output `q . S_new` single K = 128
/// reductions over a large state with cancellation; both must hold against
/// Neumaier-compensated f64 within `2^-22 K S` at K = 128.
#[test]
fn channel_decay_accumulation_128k_gpu() {
    gpu_test(2 << 30, |session| {
        let module = module(session)?;
        let mut report = Report::new("accumulation_128k");
        let mut case = Case::new(0x128, 1, 1);
        // x = -100 with a = 1: sigmoid underflows below 1e-43, g is -0 or
        // its neighbour and exp(g) is 1.
        case.f.iter_mut().for_each(|f| *f = f32_to_bf16(-100.0));
        case.dt_bias.iter_mut().for_each(|b| *b = 0.0);
        case.a_log.iter_mut().for_each(|a| *a = 0.0);
        let mut rng = Rng(0x5128);
        case.delta_pool = (0..STATE)
            .map(|_| {
                let magnitude = 10_f64.powf(3.0 * rng.uniform());
                (if rng.next() & 1 == 0 {
                    magnitude
                } else {
                    -magnitude
                }) as f32
            })
            .collect();
        let (q, k, v, b, f) = case.delta_operands(&case.qkv, INPUT_STRIDE, 0, 1);
        let state = case.delta_slot(0);
        for (phase, name) in [
            (RecurrentPhase::Prefill, "prefill"),
            (RecurrentPhase::Decode, "decode"),
        ] {
            let device = Device::new(session, &case, 1)?;
            let layer = Layer::load(&module, phase, 1, 1)?;
            let meta = Metadata::new(
                session,
                &Batch {
                    lengths: &[1],
                    positions: &[0],
                    slots: &[0],
                    initial: &[true],
                    anchors: &[-1],
                    checkpoints: &[],
                },
                layer.workspace_bytes(),
            )?;
            layer.launch_delta(session, &session.stream, &device, &meta, false, 1, 1, true)?;
            let core = decode(&u16s(&download(session, &device.core)?));
            let pool = f64s(&f32s(&download(session, &device.delta_state)?));
            let (mut out, mut out_s) = (Vec::new(), Vec::new());
            let (mut next, mut next_s) = (Vec::new(), Vec::new());
            for head in 0..H {
                let base = head * D;
                let scale = (D as f64).sqrt().recip();
                let qn = recurrent_oracle::l2_normalize(&q[base..base + D])
                    .into_iter()
                    .map(|x| x * scale)
                    .collect::<Vec<_>>();
                let kn = recurrent_oracle::l2_normalize(&k[base..base + D]);
                let beta = oracle::beta(b[head]);
                let g = (0..D)
                    .map(|x| {
                        oracle::log_decay(
                            SERVED.gate,
                            f[base + x],
                            f64::from(case.dt_bias[base + x]),
                            0.0,
                        )
                    })
                    .collect::<Vec<_>>();
                assert!(g.iter().all(|g| g.exp() == 1.0));
                for value in 0..D {
                    let s = &state[(head * D + value) * D..(head * D + value + 1) * D];
                    let products = (0..D).map(|y| kn[y] * s[y]);
                    let u = compensated_sum(products.clone());
                    let u_s = compensated_sum(products.map(f64::abs));
                    let delta = beta * (v[base + value] - u);
                    let delta_s = beta * (v[base + value].abs() + u_s);
                    let row = (0..D).map(|x| s[x] + kn[x] * delta).collect::<Vec<_>>();
                    next.extend(row.iter().copied());
                    next_s.extend((0..D).map(|x| s[x].abs() + kn[x].abs() * delta_s));
                    out.push(compensated_sum((0..D).map(|x| qn[x] * row[x])));
                    out_s.push(
                        compensated_sum((0..D).map(|x| (qn[x] * s[x]).abs()))
                            + compensated_sum((0..D).map(|x| (qn[x] * kn[x]).abs()))
                                * delta_s.abs(),
                    );
                }
            }
            let output = OracleOutput::new(out, out_s, D).map_err(|e| e.0.to_string())?;
            let state_out = OracleOutput::new(next, next_s, D).map_err(|e| e.0.to_string())?;
            report.check(&format!("{name}_output"), &core, &output, DType::Bf16);
            report.check(
                &format!("{name}_state"),
                &pool[..STATE],
                &state_out,
                DType::F32,
            );
        }
        report.finish(session)
    })
    .expect("the 128-key chains hold against compensated f64");
}

// ---------------------------------------------------- poisoned replay ----

/// F32 workspace bytes of one (aligned chunk, head).
const WORK_BYTES: usize = channel_decay::WORK_FLOATS as usize * 4;

/// Pool entry byte ranges.
fn entries(ids: &[i32], bytes: usize) -> Vec<std::ops::Range<usize>> {
    ids.iter()
        .map(|&id| id as usize * bytes..(id as usize + 1) * bytes)
        .collect()
}

/// A fixed capacity layout of one phase: every sequence's rows, absolute
/// start position, initial flag and pool ids, of which the first `live`
/// sequences run. Slots are `2 s + 1`, anchors `2 S + s` and checkpoints
/// `3 S + i` for S sequences, so inactive and unused entries sit between and
/// after the live ones.
struct Fixture {
    phase: RecurrentPhase,
    lengths: Vec<usize>,
    positions: Vec<usize>,
    initial: Vec<bool>,
    slots: Vec<i32>,
    anchors: Vec<i32>,
    checkpoints: Vec<i32>,
}

impl Fixture {
    fn new(phase: RecurrentPhase) -> Self {
        let (lengths, positions, initial) = match phase {
            // Ends off a multiple of 64 with one inside (anchors written),
            // exactly on one (seq 1: its anchor stays), and short calls at
            // phases 0, 5, 36 and 27.
            RecurrentPhase::Prefill => (
                vec![65, 64, 1, 130, 37, 100],
                vec![0, 0, 64, 5, 100, 27],
                vec![true, false, true, true, false, true],
            ),
            RecurrentPhase::Decode => (vec![1; 8], vec![0; 8], vec![true; 8]),
            RecurrentPhase::Verify { .. } => (vec![3, 8, 1, 5], vec![0; 4], vec![true; 4]),
        };
        let sequences = lengths.len();
        Self {
            phase,
            lengths,
            positions,
            initial,
            slots: (0..sequences).map(|s| (2 * s + 1) as i32).collect(),
            anchors: (0..sequences).map(|s| (2 * sequences + s) as i32).collect(),
            checkpoints: (0..sequences * VERIFY_ROWS)
                .map(|i| (3 * sequences + i) as i32)
                .collect(),
        }
    }

    fn sequences(&self) -> usize {
        self.lengths.len()
    }
    fn rows(&self) -> usize {
        self.lengths.iter().sum()
    }
    /// Prepared rows: verify plans `VERIFY_ROWS` per sequence.
    fn bucket_rows(&self) -> usize {
        match self.phase {
            RecurrentPhase::Verify { .. } => self.sequences() * VERIFY_ROWS,
            _ => self.rows(),
        }
    }
    fn live_rows(&self, live: usize) -> usize {
        self.lengths[..live].iter().sum()
    }
    /// Pool entries the fixture's ids reach.
    fn entries(&self) -> usize {
        3 * self.sequences() + self.checkpoints.len() + 1
    }
    /// Every id the phase reads or writes, in a fixed order: slots, then
    /// anchors (prefill) or checkpoints (verify).
    fn ids(&self) -> Vec<i32> {
        let mut ids = self.slots.clone();
        match self.phase {
            RecurrentPhase::Prefill => ids.extend(&self.anchors),
            RecurrentPhase::Decode => {}
            RecurrentPhase::Verify { .. } => ids.extend(&self.checkpoints),
        }
        ids
    }

    /// Entries the first `live` sequences only write (anchors of calls that
    /// end off a multiple of 64 with one inside, the checkpoints of their
    /// rows) and entries they read and rewrite (prefill and decode slots).
    fn writes(&self, live: usize) -> (Vec<i32>, Vec<i32>) {
        match self.phase {
            RecurrentPhase::Prefill => (
                (0..live)
                    .filter(|&s| anchor_row(self.positions[s], self.lengths[s]).is_some())
                    .map(|s| self.anchors[s])
                    .collect(),
                self.slots[..live].to_vec(),
            ),
            RecurrentPhase::Decode => (Vec::new(), self.slots[..live].to_vec()),
            RecurrentPhase::Verify { .. } => (
                (0..live)
                    .flat_map(|s| {
                        self.checkpoints[s * VERIFY_ROWS..s * VERIFY_ROWS + self.lengths[s]]
                            .to_vec()
                    })
                    .collect(),
                Vec::new(),
            ),
        }
    }

    /// Workspace bytes the prefill of the first `live` sequences writes:
    /// their chunks at absolute multiples of 64, every head, in order.
    fn live_workspace(&self, live: usize) -> usize {
        let chunks = (0..live)
            .map(|s| (self.positions[s] % 64 + self.lengths[s]).div_ceil(64))
            .sum::<usize>();
        chunks * H * WORK_BYTES
    }

    /// Rows the cases allocate for every operand and output: the planned
    /// bucket, so rows past the live ones are allocated poisoned tails.
    fn allocation_rows(&self) -> usize {
        self.bucket_rows()
    }

    /// The slots, anchors and checkpoints with used id `ids()[i]` moved to
    /// `to[i]`. Arrays the phase does not read hold -1.
    fn remapped(&self, to: &[i32]) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
        let used = self.ids();
        assert_eq!(to.len(), used.len(), "one target id per used id");
        let map = |ids: &[i32], read: bool| -> Vec<i32> {
            ids.iter()
                .map(|&id| {
                    if !read {
                        return -1;
                    }
                    let index = used.iter().position(|&u| u == id).expect("a used id");
                    to[index]
                })
                .collect()
        };
        (
            map(&self.slots, true),
            map(&self.anchors, self.phase == RecurrentPhase::Prefill),
            map(
                &self.checkpoints,
                matches!(self.phase, RecurrentPhase::Verify { .. }),
            ),
        )
    }
}

/// The layer chain of one phase (convolution, delta rule over its output,
/// norm) at a fixed capacity bucket, with every written buffer poisoned.
/// One graph is captured for the bucket; for each live sequence count
/// (capacity, 0, 1, the middle, capacity - 1) only the device counts change,
/// and eager, relaunch, two replays of that graph and a PDL pair whose
/// producer is the convolution must agree bit for bit. Only the live rows of
/// each output and the live sequences' state entries may change; inactive
/// output tails stay poisoned, other entries keep their bits, and the
/// prefill workspace past the live chunks stays poisoned.
fn replay_phase(session: &Session, module: &Module, phase: RecurrentPhase) -> LabResult<()> {
    let fixture = Fixture::new(phase);
    let (rows, sequences) = (fixture.rows(), fixture.sequences());
    let allocated = fixture.allocation_rows();
    let case = Case::new(0x9015 + rows as u64, allocated, fixture.entries());
    let device = Device::new(session, &case, allocated)?;
    let layer = Layer::load(module, phase, fixture.bucket_rows(), sequences)?;
    let meta = Metadata::new(
        session,
        &Batch {
            lengths: &fixture.lengths,
            positions: &fixture.positions,
            slots: &fixture.slots,
            initial: &fixture.initial,
            anchors: &fixture.anchors,
            checkpoints: &fixture.checkpoints,
        },
        layer.workspace_bytes(),
    )?;
    let chain = |stream: &Stream, live_rows: usize, live: usize, pdl: bool| -> LabResult<()> {
        layer.launch_conv(session, stream, &device, &meta, live_rows, live, pdl)?;
        layer.launch_delta(session, stream, &device, &meta, true, live_rows, live, pdl)?;
        layer.launch_norm(session, stream, &device, &meta, live_rows, pdl)
    };
    let graph = session.capture(|stream| chain(stream, rows, sequences, true))?;
    for live in poison::live_counts(sequences as u32) {
        let live = live as usize;
        let live_rows = fixture.live_rows(live);
        poison::write(
            session,
            &meta.live_sequences,
            0,
            &(live as u32).to_ne_bytes(),
        )?;
        poison::write(
            session,
            &meta.live_rows,
            0,
            &(live_rows as u32).to_ne_bytes(),
        )?;
        let (written, updated) = fixture.writes(live);
        let mut checks = Replay::new(session)
            .output(
                "conv_output",
                &device.conv_out,
                WriteMap::of(DType::Bf16).prefix(live_rows * CHANNELS)?,
            )
            .output(
                "delta_output",
                &device.core,
                WriteMap::of(DType::Bf16).prefix(live_rows * WIDTH)?,
            )
            .output(
                "norm_output",
                &device.normed,
                WriteMap::of(DType::Bf16).prefix(live_rows * WIDTH)?,
            )
            .state(
                "conv_state",
                &device.conv_state,
                bytes_u16(&case.conv_pool),
                WriteMap::of(DType::Bf16).byte_ranges(entries(&written, CONV_ENTRY * 2))?,
                entries(&updated, CONV_ENTRY * 2),
            )
            .state(
                "delta_state",
                &device.delta_state,
                bytes_f32(&case.delta_pool),
                WriteMap::of(DType::F32).byte_ranges(entries(&written, STATE * 4))?,
                entries(&updated, STATE * 4),
            );
        if phase == RecurrentPhase::Prefill {
            // Ranges derived from the lengths and positions, not the planner.
            checks = checks.scratch_within(
                "workspace",
                &meta.workspace,
                std::iter::once(0..fixture.live_workspace(live)).collect(),
            );
        }
        let label = |error: String| format!("live {live} of {sequences}: {error}");
        checks
            .eager(|| chain(&session.stream, live_rows, live, false))
            .map_err(label)?;
        checks
            .relaunch(|| chain(&session.stream, live_rows, live, false))
            .map_err(label)?;
        checks.graph(&graph).map_err(label)?;
        checks.graph(&graph).map_err(label)?;
        checks
            .pdl(
                || {
                    layer.launch_conv(
                        session,
                        &session.stream,
                        &device,
                        &meta,
                        live_rows,
                        live,
                        false,
                    )
                },
                || {
                    layer.launch_delta(
                        session,
                        &session.stream,
                        &device,
                        &meta,
                        true,
                        live_rows,
                        live,
                        true,
                    )?;
                    layer.launch_norm(session, &session.stream, &device, &meta, live_rows, true)
                },
                true,
            )
            .map_err(label)?;
        let mut receipt = BTreeMap::new();
        receipt.insert("family".into(), json!("channel_decay"));
        receipt.insert("phase".into(), json!(format!("{phase:?}")));
        receipt.insert("bucket_sequences".into(), json!(sequences));
        receipt.insert("bucket_rows".into(), json!(fixture.bucket_rows()));
        receipt.insert("live".into(), json!(live));
        receipt.insert("live_rows".into(), json!(live_rows));
        receipt.insert("correct".into(), json!(true));
        checks.record(&mut receipt);
        session.receipt("poisoned_replay", receipt)?;
    }
    Ok(())
}

#[test]
fn channel_decay_poisoned_replays_match_eager_gpu() {
    gpu_test(4 << 30, |session| {
        let module = module(session)?;
        for phase in [
            RecurrentPhase::Prefill,
            RecurrentPhase::Decode,
            RecurrentPhase::Verify {
                max_rows: VERIFY_ROWS as u32,
            },
        ] {
            replay_phase(session, &module, phase).map_err(|e| format!("{phase:?}: {e}"))?;
        }
        Ok(())
    })
    .expect("poisoned eager, relaunch, graph and PDL launches agree bit for bit");
}

// ------------------------------------------------------------ high ids ----

/// Runs the convolution (`conv`) or the delta rule of `fixture`'s phase,
/// every sequence live, with used id `ids()[i]` at pool id `to[i]` of the
/// state pool `state`. Returns the whole output buffer, poisoned tail
/// included.
fn launch_mapped(
    session: &Session,
    layer: &Layer,
    device: &Device,
    fixture: &Fixture,
    conv: bool,
    to: &[i32],
    state: &DeviceBuffer,
) -> LabResult<Vec<u8>> {
    let (slots, anchors, checkpoints) = fixture.remapped(to);
    let meta = Metadata::new(
        session,
        &Batch {
            lengths: &fixture.lengths,
            positions: &fixture.positions,
            slots: &slots,
            initial: &fixture.initial,
            anchors: &anchors,
            checkpoints: &checkpoints,
        },
        layer.workspace_bytes(),
    )?;
    let (rows, sequences) = (fixture.rows(), fixture.sequences());
    let output = if conv { &device.conv_out } else { &device.core };
    pool_offsets::poison(session, output)?;
    if conv {
        let binds = meta
            .bind(device.conv(), fixture.phase)
            .buffer("state", state);
        layer
            .conv
            .launch(session, &session.stream, &binds, rows, sequences, true)?;
    } else {
        for kernel in &layer.delta {
            let binds = meta
                .bind(device.delta(false), fixture.phase)
                .buffer("state", state);
            kernel.launch(session, &session.stream, &binds, rows, sequences, true)?;
        }
    }
    download(session, output)
}

/// State past the 32-bit lines (`common/pool_offsets.rs`): the convolution
/// and the delta rule in prefill (slots and anchors, with and without an
/// initial state), decode and verify (slots and checkpoints), with every id
/// the phase uses alternating between group A and group B of the entry
/// size, in a pool whose other bytes hold the sentinel. The high run's
/// output and every entry equal the dense low run bit for bit, and no other
/// pool byte changes.
#[test]
fn channel_decay_high_ids_match_low_ids() {
    gpu_test(pool_offsets::need_bytes(18 << 30), |session| {
        let module = module(session)?;
        for (index, phase) in [
            RecurrentPhase::Prefill,
            RecurrentPhase::Decode,
            RecurrentPhase::Verify {
                max_rows: VERIFY_ROWS as u32,
            },
        ]
        .into_iter()
        .enumerate()
        {
            let fixture = Fixture::new(phase);
            let used = fixture.ids();
            let allocated = fixture.allocation_rows();
            // Entry i of the dense pools is the initial state of used id i.
            let case = Case::new(0x1d00 + index as u64, allocated, used.len());
            let device = Device::new(session, &case, allocated)?;
            let layer = Layer::load(&module, phase, fixture.bucket_rows(), fixture.sequences())?;
            let dense = (0..used.len() as i32).collect::<Vec<_>>();
            for conv in [true, false] {
                let name = format!("{}_{phase:?}", if conv { "conv" } else { "delta" });
                let (initial, element) = if conv {
                    (bytes_u16(&case.conv_pool), 2)
                } else {
                    (bytes_f32(&case.delta_pool), 4)
                };
                let span = initial.len() / used.len();
                let low_state = upload(session, &initial)?;
                let low =
                    launch_mapped(session, &layer, &device, &fixture, conv, &dense, &low_state)?;
                let after = download(session, &low_state)?;
                let ids = pool_offsets::high_ids(span as u64, element, used.len().div_ceil(2));
                let high_ids = pool_offsets::i32_ids(&ids.alternating(used.len()));
                let highest = high_ids.iter().copied().max().unwrap_or(0) as u64;
                let mut pool = SentinelPool::new(session, (highest + 1) * span as u64)?;
                for (i, &id) in high_ids.iter().enumerate() {
                    pool.put(
                        session,
                        id as u64 * span as u64,
                        &initial[i * span..(i + 1) * span],
                    )?;
                }
                let high = launch_mapped(
                    session,
                    &layer,
                    &device,
                    &fixture,
                    conv,
                    &high_ids,
                    &pool.buffer,
                )?;
                pool_offsets::same(&name, "output", &low, &high)?;
                for (i, &id) in high_ids.iter().enumerate() {
                    pool_offsets::same(
                        &name,
                        &format!("state entry {id}"),
                        &after[i * span..(i + 1) * span],
                        &pool.read(session, id as u64 * span as u64, span as u64)?,
                    )?;
                }
                let scanned = pool.check_untouched(session, &name)?;
                pool_offsets::receipt(
                    session,
                    "channel_decay",
                    &name,
                    json!({
                        "entries": used.len(),
                        "ids": ids.describe(),
                        "highest_offset": highest * span as u64,
                        "outputs_bitwise": true,
                        "entries_bitwise": true,
                        "sentinel_bytes_checked": scanned,
                    }),
                )?;
            }
        }
        Ok(())
    })
    .expect("per-channel-decay state at high slot, anchor and checkpoint ids");
}

/// CPU: the replay fixtures leave unused workspace past the live chunks and
/// inactive capacity at every partial live count, and the high-id groups fit
/// the engine's entry sizes.
#[test]
fn channel_decay_replay_fixtures_cover_inactive_capacity() {
    let fixture = Fixture::new(RecurrentPhase::Prefill);
    let bucket = Bucket {
        rows: fixture.rows() as u32,
        sequences: fixture.sequences() as u32,
    };
    let reserved = channel_decay::aligned_chunks(bucket) as usize * H * WORK_BYTES;
    assert_eq!(
        find("delta_prefill_prepare").workspace_bytes(bucket) as usize,
        reserved
    );
    assert_eq!(fixture.live_workspace(6), 11 * H * WORK_BYTES);
    assert!(fixture.live_workspace(6) < reserved);
    // Seq 1 ends exactly on 64 with an anchor id: never written.
    assert_eq!(fixture.writes(6).0, vec![12, 15, 16, 17]);
    for phase in [
        RecurrentPhase::Prefill,
        RecurrentPhase::Decode,
        RecurrentPhase::Verify {
            max_rows: VERIFY_ROWS as u32,
        },
    ] {
        let fixture = Fixture::new(phase);
        let ids = fixture.ids();
        assert!(ids.iter().all(|&id| (id as usize) < fixture.entries()));
        for live in poison::live_counts(fixture.sequences() as u32) {
            let live = live as usize;
            assert!(fixture.live_rows(live) <= fixture.rows());
            let (written, updated) = fixture.writes(live);
            assert!(written.iter().chain(&updated).all(|id| ids.contains(id)));
        }
        // Every operand and output buffer covers the planned bucket.
        assert!(fixture.allocation_rows() >= fixture.rows());
        assert_eq!(fixture.allocation_rows(), fixture.bucket_rows());
        // The remappings the high-id test launches with: dense, then each
        // entry size's alternating group A and B ids.
        let count = ids.len().div_ceil(2);
        let dense = (0..ids.len() as i32).collect::<Vec<_>>();
        let conv = pool_offsets::high_ids(CONV_ENTRY as u64 * 2, 2, count);
        let delta = pool_offsets::high_ids(STATE as u64 * 4, 4, count);
        for to in [
            dense,
            pool_offsets::i32_ids(&conv.alternating(ids.len())),
            pool_offsets::i32_ids(&delta.alternating(ids.len())),
        ] {
            let (slots, anchors, checkpoints) = fixture.remapped(&to);
            let at = |id: i32| to[ids.iter().position(|&u| u == id).unwrap()];
            assert_eq!(
                slots,
                fixture.slots.iter().map(|&id| at(id)).collect::<Vec<_>>()
            );
            let prefill = phase == RecurrentPhase::Prefill;
            let verify = matches!(phase, RecurrentPhase::Verify { .. });
            assert_eq!(anchors.len(), fixture.sequences());
            assert_eq!(checkpoints.len(), fixture.sequences() * VERIFY_ROWS);
            assert!(anchors.iter().all(|&id| (id >= 0) == prefill));
            assert!(checkpoints.iter().all(|&id| (id >= 0) == verify));
            let mut all = slots.clone();
            all.extend(anchors.iter().chain(&checkpoints).filter(|&&id| id >= 0));
            all.sort_unstable();
            assert_eq!(all, {
                let mut sorted = to.clone();
                sorted.sort_unstable();
                sorted
            });
        }
    }
    let verify = Fixture::new(RecurrentPhase::Verify {
        max_rows: VERIFY_ROWS as u32,
    });
    assert_eq!((verify.rows(), verify.allocation_rows()), (17, 32));
}

/// CPU: the lab registers the engine's cubin with its whole catalog, and the
/// fixtures' shape is the catalog's.
#[test]
fn channel_decay_fixtures_use_the_catalog_shape() {
    let name = channel_decay::cubin_specs().remove(0).name;
    let entries = kernel_lab::cubin_entries(&name).expect("the lab builds the cubin");
    let catalog = channel_decay::catalog();
    let symbols = catalog
        .iter()
        .map(|entry| entry.symbol.as_str())
        .collect::<Vec<_>>();
    assert_eq!(entries, symbols.as_slice());
    let Query::Norm(RecurrentQuery::GatedRmsNorm { width, gate, .. }) = find("gated_norm").query
    else {
        panic!("gated_norm is the gated RMSNorm");
    };
    assert_eq!((width as usize, gate), (WIDTH, GateActivation::Sigmoid));
    assert!(matches!(
        rule_query("delta_verify"),
        PerChannelDecayDeltaRuleQuery::DeltaRule {
            phase: RecurrentPhase::Verify { max_rows: 8 },
            ..
        }
    ));
}

#[path = "common/invariance.rs"]
mod invariance;
#[path = "channel_decay/invariance.rs"]
mod invariance_adapter;

/// The invariance adapters of this test binary, one per op kind.
fn invariance_adapters() -> Vec<Box<dyn invariance::Adapter>> {
    vec![Box::new(invariance_adapter::ChannelDecay)]
}

fn invariance_case(property: kernels_sm120::invariance::Property) {
    gpu_test(8 << 30, |session| {
        invariance::run(
            property,
            &mut invariance_adapters(),
            std::slice::from_ref(session),
        )
    })
    .unwrap_or_else(|error| panic!("{property:?} channel decay invariance: {error}"));
}

#[test]
fn invariance_channel_decay_chunk() {
    invariance_case(kernels_sm120::invariance::Property::Chunk);
}

#[test]
fn invariance_channel_decay_rows() {
    invariance_case(kernels_sm120::invariance::Property::Rows);
}

#[test]
fn invariance_channel_decay_batch() {
    invariance_case(kernels_sm120::invariance::Property::Batch);
}

#[test]
fn invariance_channel_decay_anchors() {
    invariance_case(kernels_sm120::invariance::Property::Anchors);
}

/// This binary holds exactly the adapters the shared table places in
/// "channel_decay", and they cover every claim of their op kinds.
#[test]
fn invariance_adapters_match_the_table() {
    let held = invariance_adapters()
        .iter()
        .map(|adapter| adapter.kind())
        .collect::<Vec<_>>();
    let placed = invariance::ADAPTERS
        .iter()
        .filter(|(_, _, binary)| *binary == "channel_decay")
        .map(|(kind, _, _)| *kind)
        .collect::<Vec<_>>();
    assert_eq!(held, placed, "channel decay adapters against ADAPTERS");
    let uncovered = invariance::uncovered()
        .into_iter()
        .filter(|claim| {
            held.iter()
                .any(|kind| claim.contains(&format!(" {kind:?} ")))
        })
        .collect::<Vec<_>>();
    assert!(
        uncovered.is_empty(),
        "claims without an adapter: {uncovered:?}"
    );
}
