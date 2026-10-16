//! The vision encoder's row entries against their f64 oracles at the served
//! shapes (`encoder_vision_shapes.rs`; README in
//! crates/kernels-sm120/src/encoder):
//!
//! - the per-head q/k RMSNorm, in place on packed qkv rows: q and k heads
//!   hold the decision-0081 gate against `encoder_oracle::qk_rms_norm`, v
//!   and the rows past the live count keep their bytes;
//! - the clamped SwiGLU row passes (4096 and 10240) against
//!   `encoder_oracle::clamped_swiglu_rows`, with gates and ups well past the
//!   limit on both sides, and NaN gates and ups staying NaN;
//! - the F32 GELU, in place as the projector runs it, against
//!   `encoder_oracle::gelu_rows_f32` over [-12, 12];
//! - the zero-column pad of 1176-wide patches to 1216 columns, bit for bit
//!   against `encoder_oracle::pad_columns`.
//!
//! Every launch runs eager, relaunched, from a graph and behind a PDL
//! producer on poisoned outputs (`kernel_lab::poison`); the producer and
//! the consumer are prepared before the replays, so the PDL pair is queued
//! with no resource work between them. None of these
//! entries accumulates through tensor cores, so there is no 128-K
//! accumulation case: the norm's longest reduction is one head of 64.
//!
//! Run in release with
//! `CHOPINFER_KERNEL_GPU=<ordinal> cargo test -p kernel-lab --release --test encoder_vision_rows`.

#[path = "../encoder_vision_shapes.rs"]
mod shapes;

#[path = "common/support.rs"]
mod support;

use std::collections::BTreeMap;
use std::ops::Range;

use kernel_build::EntryParam;
use kernel_core::contract::{DType, MultimodalQuery};
use kernel_core::encoder_oracle as oracle;
use kernel_core::oracle::{OracleOutput, check_exact};
use kernel_lab::launch;
use kernel_lab::poison::{Replay, WriteMap, replay_all};
use kernel_lab::{LabResult, Session, check_output, gpu_test};
use kernels_sm120::encoder::Entry;
use runtime_cuda::{DeviceBuffer, Function, KernelArgument, LaunchConfig, Module};
use serde_json::json;
use support::upload;

/// Live rows of every launch; buffers hold one more row, which must keep
/// its bytes.
const ROWS: usize = 1000;

fn err(e: impl ToString) -> String {
    e.to_string()
}

fn entry(matches: impl Fn(&MultimodalQuery) -> bool) -> Entry {
    shapes::entries()
        .into_iter()
        .find(|entry| matches(&entry.query))
        .expect("a lab vision row entry")
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

/// F32 values in [-scale, scale) from a fixed xorshift64* stream.
fn seeded(len: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let unit = (state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as f32
                / (1u64 << 23) as f32
                - 1.0;
            unit * scale
        })
        .collect()
}

fn widen(values: &[f32]) -> Vec<f64> {
    values.iter().map(|&v| f64::from(v)).collect()
}

/// One entry's launch, prepared once: its function (which keeps the
/// module loaded), launch shape and arguments. `launch` only queues it, so
/// no function lookup, argument binding or CUDA resource work (allocation,
/// free, upload, module load or unload) separates a producer and this
/// launch; the shared launch path still marshals the arguments on the host
/// per enqueue.
struct Consumer<'a> {
    function: Function,
    config: LaunchConfig,
    arguments: Vec<KernelArgument<'a>>,
}

impl<'a> Consumer<'a> {
    /// `entry` with its pointers bound by parameter name and its scalars
    /// by name.
    fn new(
        module: &Module,
        entry: &Entry,
        buffers: &[(&str, &'a DeviceBuffer)],
        scalars: &[(&str, u32)],
    ) -> LabResult<Self> {
        let function = module.function(&entry.symbol).map_err(err)?;
        let plan = entry.plan(ROWS as u32).map_err(|e| e.0.to_string())?;
        let arguments = entry
            .params
            .iter()
            .map(|param| match param {
                EntryParam::DevicePointer { name, .. } => buffers
                    .iter()
                    .find(|(key, _)| key == name)
                    .map(|&(_, buffer)| KernelArgument::Buffer(buffer))
                    .ok_or_else(|| format!("{}: nothing binds {name}", entry.symbol)),
                EntryParam::U32 { name } => scalars
                    .iter()
                    .find(|(key, _)| key == name)
                    .map(|&(_, value)| KernelArgument::U32(value))
                    .ok_or_else(|| format!("{}: missing scalar {name}", entry.symbol)),
                _ => Err(format!("{}: unexpected ABI kind", entry.symbol)),
            })
            .collect::<LabResult<Vec<_>>>()?;
        Ok(Consumer {
            function,
            config: launch::config(plan)?,
            arguments,
        })
    }

    fn launch(&self, session: &Session) -> LabResult<()> {
        self.function
            .launch(&session.stream, self.config, &self.arguments)
            .map_err(err)
    }
}

fn check(
    name: &str,
    actual: &[f32],
    expected: &OracleOutput,
) -> LabResult<BTreeMap<String, serde_json::Value>> {
    if actual.iter().any(|x| !x.is_finite()) {
        return Err(format!("{name}: unwritten or non-finite output"));
    }
    let mut receipt = BTreeMap::new();
    check_output(actual, expected, DType::F32, &mut receipt)
        .map_err(|e| format!("{name}: {e}; {receipt:?}"))?;
    Ok(receipt)
}

fn receipt(
    session: &Session,
    entry: &Entry,
    mut evidence: BTreeMap<String, serde_json::Value>,
    gate: BTreeMap<String, serde_json::Value>,
) -> LabResult<()> {
    evidence.insert("entry".into(), json!(entry.symbol));
    evidence.insert("rows".into(), json!(ROWS));
    evidence.extend(gate);
    session.receipt("encoder_vision_rows", evidence)?;
    Ok(())
}

#[test]
fn qk_rms_norm_holds_its_oracle_in_place() {
    gpu_test(1 << 30, |session| {
        let module = session.module(shapes::CUBIN)?;
        let entry = entry(|query| matches!(query, MultimodalQuery::QkRmsNormF32 { .. }));
        let MultimodalQuery::QkRmsNormF32 {
            q_heads,
            kv_heads,
            head_dim,
            ..
        } = entry.query
        else {
            unreachable!()
        };
        let (q, kv, d) = (q_heads as usize, kv_heads as usize, head_dim as usize);
        let width = (q + 2 * kv) * d;
        let normed = (q + kv) * d;
        // Heads of different scales, from tiny to large.
        let mut rows = seeded((ROWS + 1) * width, 0x51, 3.0);
        for (index, value) in rows.iter_mut().enumerate() {
            *value *= [1e-3, 1.0, 40.0][(index / d) % 3];
        }
        let q_weight = seeded(d, 0x52, 1.5);
        let k_weight = seeded(d, 0x53, 1.5);
        let to_bf16 = |values: &[f32]| -> Vec<u16> {
            values
                .iter()
                .map(|&v| kernel_core::codec::f32_to_bf16(v))
                .collect()
        };
        let (q_bits, k_bits) = (to_bf16(&q_weight), to_bf16(&k_weight));
        let initial = f32_bytes(&rows);
        let buffer = upload(session, &initial)?;
        let q_buffer = session.upload_bf16(&q_bits)?;
        let k_buffer = session.upload_bf16(&k_bits)?;
        let live = upload(session, &(ROWS as u32).to_ne_bytes())?;
        // q and k of every live row are rewritten; v and the extra row keep
        // their bytes.
        let updated: Vec<Range<usize>> = (0..ROWS)
            .map(|row| row * width * 4..(row * width + normed) * 4)
            .collect();
        let replay = Replay::new(session).state(
            "rows",
            &buffer,
            initial.clone(),
            WriteMap::of(DType::F32).none()?,
            updated,
        );
        let mut evidence = BTreeMap::new();
        let consumer = Consumer::new(
            &module,
            &entry,
            &[
                ("rows", &buffer),
                ("q_weight", &q_buffer),
                ("k_weight", &k_buffer),
                ("live_rows", &live),
            ],
            &[("eps_bits", shapes::EPS.to_bits())],
        )?;
        replay_all(
            session,
            replay,
            &live,
            || consumer.launch(session),
            &mut evidence,
        )?;
        let actual = session.download_f32(&buffer, ROWS * width)?;
        let decode = |bits: &[u16]| -> Vec<f64> {
            bits.iter()
                .map(|&b| f64::from(kernel_core::codec::bf16_to_f32(b)))
                .collect()
        };
        let expected = oracle::qk_rms_norm(
            &widen(&rows[..ROWS * width]),
            q,
            kv,
            d,
            &decode(&q_bits),
            &decode(&k_bits),
            f64::from(shapes::EPS),
        )
        .map_err(err)?;
        let gate = check(&entry.symbol, &actual, &expected)?;
        // v is copied through the oracle, so the gate above also holds it
        // bitwise; the state check holds the extra row.
        receipt(session, &entry, evidence, gate)
    })
    .unwrap();
}

fn clamped_swiglu(hidden: u32) {
    gpu_test(2 << 30, |session| {
        let module = session.module(shapes::CUBIN)?;
        let entry = entry(|query| {
            matches!(query, MultimodalQuery::SwigluClampedF32 { hidden: h, .. } if *h == hidden)
        });
        let hidden = hidden as usize;
        // Gates and ups up to twice the limit on both sides.
        let input = seeded((ROWS + 1) * 2 * hidden, 0x61 ^ hidden as u64, 2.0 * shapes::LIMIT);
        let input_buffer = upload(session, &f32_bytes(&input))?;
        let output = session.allocate((ROWS + 1) * hidden * 4)?;
        let live = upload(session, &(ROWS as u32).to_ne_bytes())?;
        let replay = Replay::new(session).output(
            "output",
            &output,
            WriteMap::of(DType::F32).prefix(ROWS * hidden)?,
        );
        let consumer = |input| {
            Consumer::new(
                &module,
                &entry,
                &[("input", input), ("output", &output), ("live_rows", &live)],
                &[("limit_bits", shapes::LIMIT.to_bits())],
            )
        };
        let mut evidence = BTreeMap::new();
        let prepared = consumer(&input_buffer)?;
        replay_all(session, replay, &live, || prepared.launch(session), &mut evidence)?;
        let actual = session.download_f32(&output, ROWS * hidden)?;
        let expected = oracle::clamped_swiglu_rows(
            &widen(&input[..ROWS * 2 * hidden]),
            hidden,
            f64::from(shapes::LIMIT),
        )
        .map_err(err)?;
        let gate = check(&entry.symbol, &actual, &expected)?;
        // A NaN gate or up stays NaN; the clamps never turn it into the
        // limit.
        let mut poisoned = input.clone();
        poisoned[3] = f32::NAN;
        poisoned[2 * hidden + hidden + 5] = f32::NAN;
        let nan_buffer = upload(session, &f32_bytes(&poisoned))?;
        consumer(&nan_buffer)?.launch(session)?;
        let actual = session.download_f32(&output, ROWS * hidden)?;
        if !actual[3].is_nan() || !actual[hidden + 5].is_nan() {
            return Err(format!("{}: a NaN operand became a number", entry.symbol));
        }
        receipt(session, &entry, evidence, gate)
    })
    .unwrap();
}

#[test]
fn clamped_swiglu_4096_holds_its_oracle() {
    clamped_swiglu(4096);
}

#[test]
fn clamped_swiglu_10240_holds_its_oracle() {
    clamped_swiglu(10240);
}

#[test]
fn f32_gelu_holds_its_oracle_in_place() {
    gpu_test(1 << 30, |session| {
        let module = session.module(shapes::CUBIN)?;
        let entry = entry(|query| matches!(query, MultimodalQuery::GeluRowsF32 { .. }));
        let MultimodalQuery::GeluRowsF32 { width, .. } = entry.query else {
            unreachable!()
        };
        let width = width as usize;
        let input = seeded((ROWS + 1) * width, 0x71, 12.0);
        let initial = f32_bytes(&input);
        let buffer = upload(session, &initial)?;
        let live = upload(session, &(ROWS as u32).to_ne_bytes())?;
        let replay = Replay::new(session).state(
            "rows",
            &buffer,
            initial.clone(),
            WriteMap::of(DType::F32).none()?,
            std::iter::once(0..ROWS * width * 4).collect(),
        );
        let mut evidence = BTreeMap::new();
        let consumer = Consumer::new(
            &module,
            &entry,
            &[
                ("input", &buffer),
                ("output", &buffer),
                ("live_rows", &live),
            ],
            &[],
        )?;
        replay_all(
            session,
            replay,
            &live,
            || consumer.launch(session),
            &mut evidence,
        )?;
        let actual = session.download_f32(&buffer, ROWS * width)?;
        let expected = oracle::gelu_rows_f32(&widen(&input[..ROWS * width])).map_err(err)?;
        let gate = check(&entry.symbol, &actual, &expected)?;
        receipt(session, &entry, evidence, gate)
    })
    .unwrap();
}

#[test]
fn zero_column_pad_copies_rows_bitwise() {
    gpu_test(1 << 30, |session| {
        let module = session.module(shapes::CUBIN)?;
        let entry = entry(|query| matches!(query, MultimodalQuery::PadColumnsF32 { .. }));
        let MultimodalQuery::PadColumnsF32 { input, output, .. } = entry.query else {
            unreachable!()
        };
        let (input, output) = (input as usize, output as usize);
        let rows = seeded((ROWS + 1) * input, 0x81, 4.0);
        let input_buffer = upload(session, &f32_bytes(&rows))?;
        let padded = session.allocate((ROWS + 1) * output * 4)?;
        let live = upload(session, &(ROWS as u32).to_ne_bytes())?;
        let replay = Replay::new(session).output(
            "output",
            &padded,
            WriteMap::of(DType::F32).prefix(ROWS * output)?,
        );
        let mut evidence = BTreeMap::new();
        let consumer = Consumer::new(
            &module,
            &entry,
            &[
                ("input", &input_buffer),
                ("output", &padded),
                ("live_rows", &live),
            ],
            &[],
        )?;
        replay_all(
            session,
            replay,
            &live,
            || consumer.launch(session),
            &mut evidence,
        )?;
        let actual = session.download_f32(&padded, ROWS * output)?;
        let expected =
            oracle::pad_columns(&widen(&rows[..ROWS * input]), input, output).map_err(err)?;
        check_exact(&widen(&actual), &expected.values)
            .map_err(|index| format!("{} differs at {index}", entry.symbol))?;
        receipt(session, &entry, evidence, BTreeMap::new())
    })
    .unwrap();
}
