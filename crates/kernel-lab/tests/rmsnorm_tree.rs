//! The row engine's HF RMSNorm with the 256-lane tree sum of squares
//! (`cuda/row/rmsnorm_tree.cuh`, `RmsNormNumerics::HF_TREE_256`) on the
//! GPU, through the lab entries of `row::rmsnorm_tree`. Run with
//! `CHOPINFER_KERNEL_GPU=<ordinal> cargo test -p kernel-lab --release --test
//! rmsnorm_tree -- --test-threads=1`; without a GPU only the CPU contract
//! test runs.
//!
//! - At every lab width (4096, 7168, 1536, 512, 8), on rows of BF16 values
//!   over ordinary, wide, tiny (squares underflow) and huge (sums overflow)
//!   binades with zeros, signed zeros and subnormals: every live row's output
//!   bitwise equal to the CPU emulation of the tree order
//!   (`rmsnorm_tree::hf_norm` with `tree_sum`), written into poisoned
//!   outputs; rows past the live count left poisoned. The serial-order
//!   baseline entry equals the CPU emulation of the serial order the same
//!   way.
//! - NaN and infinite inputs: the outputs that are NaN on the CPU are NaN on
//!   the GPU and the others equal bitwise.
//! - Against the exact (F64) sum of squares, on ordinary and wide rows of
//!   the hidden width: the tree's relative error is at most 2^-19 (the bound
//!   of 15 lane adds and 8 tree levels is about 23 units of 2^-24) on every
//!   row; the receipt reports the tree's and the serial order's error and
//!   how many outputs each gives differently from the norm with the exact
//!   reciprocal.
//! - The paired oracle comparison (`rmsnorm_tree_oracle_accuracy_paired`):
//!   2048 rows of each kind through the serial-order kernel (the HF norm
//!   served before) and the tree kernel, each equal to its emulation, then
//!   both device output sets against the F64 oracle (exact squares, exact
//!   sum and reciprocal, each HF rounding correctly rounded): maximum and
//!   mean absolute and relative output error, BF16 outputs off the
//!   correctly rounded result, and the sum's relative error, on ordinary,
//!   wide, activation-like, order-sensitive, eps-scale and subnormal-square
//!   rows; the tree must have no more outputs off and no greater sum error.
//!   Zero, overflowing and NaN rows are counted apart (NaN, zero and other
//!   outputs per kernel, outputs where the kernels differ). The CPU
//!   contract test runs the same comparison on the emulations.
//! - Poisoned replays at capacity 32: eager, relaunch, one graph captured at
//!   capacity 32, and a PDL launch behind the lab's live-count producer
//!   (which zeroes the live count and the input until its tail), at every
//!   live count 0 to 32.
//! - The reversed-distances mutant is caught only when an output differs
//!   from the served entry's, on the rows of `common/tree_mutant_rows.rs`,
//!   which the CPU contract test proves the two orders tell apart (random
//!   rows do not): the served entry's rows equal the tree's emulation and
//!   the mutant's the reversed order's, row by row, and every row differs.
//!
//! With `CHOPINFER_KERNEL_BENCH=1`, `rmsnorm_tree_timing` times the tree
//! entry against the serial-order baseline at the hidden width.

use kernel_core::codec::{bf16_to_f32, f32_to_bf16, f64_to_bf16};
use kernel_core::contract::{ArgumentList, DType, LaunchPlan};
use kernel_lab::launch::{self, Bindings};
use kernel_lab::poison::{self, LiveProducer, Replay, Rewrites, WriteMap};
use kernel_lab::{LabResult, Session};
use kernels_sm120::row::rmsnorm::{RmsNorm, RmsNormBuffers};
use kernels_sm120::row::rmsnorm_tree::{
    self, LAB_EPS, LAB_WIDTHS, exact_sum, hf_norm, serial_sum, tree_sum,
};
use runtime_cuda::{Function, Module};
use serde_json::{Value, json};
use std::collections::BTreeMap;

#[path = "common/tree_mutant_rows.rs"]
mod tree_mutant_rows;

const X: u64 = 0;
const WEIGHT: u64 = 1;
const Y: u64 = 2;
const LIVE: u64 = 3;
const X_IMAGE: u64 = 4;
const HIDDEN: usize = 4096;
const FULL_REPLAY_ROWS: usize = 32;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn served_cubin() -> &'static str {
    "rmsnorm_tree_sm120"
}

fn mutant_cubin() -> &'static str {
    "rmsnorm_tree_distances_reversed_sm120"
}

/// BF16 rows over the binades `low..=high` with random signs and
/// mantissas; every 97th value a zero, negative zero or subnormal.
fn bf16_rows(len: usize, seed: u64, low: i32, high: i32) -> Vec<u16> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..len)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            match i % 97 {
                0 => 0x0000,
                1 => 0x8000,
                2 => 0x0003,
                _ => {
                    let span = (high - low + 1) as u64;
                    let exponent = (low + ((state >> 8) % span) as i32 + 127) as u16;
                    let sign = ((state >> 40) & 1) as u16;
                    (sign << 15) | (exponent << 7) | ((state >> 24) & 0x7f) as u16
                }
            }
        })
        .collect()
}

/// The row kinds: (name, binade range).
const KINDS: [(&str, i32, i32); 4] = [
    ("ordinary", -4, 2),
    ("wide", -30, 30),
    ("tiny", -126, -100),
    ("huge", 55, 63),
];

/// An entry, its plan at a capacity and the buffers of its launches.
struct Fixture {
    width: usize,
    capacity: usize,
    buffers: BTreeMap<u64, runtime_cuda::DeviceBuffer>,
}

impl Fixture {
    fn new(session: &Session, width: usize, x: &[u16], weight: &[u16]) -> LabResult<Self> {
        let capacity = x.len() / width;
        let mut buffers = BTreeMap::new();
        buffers.insert(X, session.upload_bf16(x)?);
        buffers.insert(X_IMAGE, session.upload_bf16(x)?);
        buffers.insert(WEIGHT, session.upload_bf16(weight)?);
        buffers.insert(Y, session.allocate(x.len() * 2)?);
        buffers.insert(LIVE, session.allocate(4)?);
        let fixture = Self {
            width,
            capacity,
            buffers,
        };
        fixture.set_live(session, capacity)?;
        Ok(fixture)
    }

    fn buffer(&self, handle: u64) -> &runtime_cuda::DeviceBuffer {
        &self.buffers[&handle]
    }

    fn set_live(&self, session: &Session, live: usize) -> LabResult<()> {
        poison::write(session, self.buffer(LIVE), 0, &(live as u32).to_ne_bytes())
    }

    /// `norm`'s entry `function` over the fixture's rows at its capacity.
    fn launch(
        &self,
        session: &Session,
        norm: &RmsNorm,
        function: &Function,
        pdl: bool,
    ) -> LabResult<()> {
        let mut plan: LaunchPlan = norm.plan(self.capacity as u32).map_err(err)?;
        plan.grid = [self.capacity as u32, 1, 1];
        plan.pdl = pdl;
        let args: ArgumentList = norm
            .arguments(RmsNormBuffers {
                status: None,
                status_row_offset: 0,
                x: X,
                weight: WEIGHT,
                residual: None,
                y: Y,
                live_rows: LIVE,
            })
            .map_err(err)?;
        let bindings = Bindings {
            buffers: self.buffers.iter().map(|(&k, v)| (k, v)).collect(),
        };
        launch::launch(
            &session.context,
            function,
            &session.stream,
            plan,
            &args,
            &bindings,
        )
    }

    fn output(&self, session: &Session) -> LabResult<Vec<u16>> {
        session.download_bf16(self.buffer(Y), self.capacity * self.width)
    }
}

/// The served cubin's tree entry of `width` and its function.
fn tree_entry(module: &Module, width: u32) -> LabResult<(RmsNorm, Function)> {
    let (norm, symbol) = rmsnorm_tree::lab_norm(width, "");
    let function = module.function(&symbol).map_err(err)?;
    Ok((norm, function))
}

fn nan(bits: u16) -> bool {
    bits & 0x7fff > 0x7f80
}

/// The rows `0..live` of `actual` against `expected` (bitwise, or NaN where
/// `expected` is NaN when `nan_aware`), and rows `live..` still poisoned.
fn compare_rows(
    label: &str,
    width: usize,
    live: usize,
    actual: &[u16],
    expected: &[u16],
    nan_aware: bool,
) -> LabResult<()> {
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let row = i / width;
        if row >= live {
            if a != 0xffff {
                return Err(format!(
                    "{label}: row {row} past the live count was written"
                ));
            }
            continue;
        }
        let same = a == e || (nan_aware && nan(a) && nan(e));
        if !same {
            return Err(format!(
                "{label}: row {row} column {}: {a:#06x}, expected {e:#06x}",
                i % width
            ));
        }
    }
    Ok(())
}

/// The CPU emulation of `sum`'s norm over every row of `x`.
fn emulate(x: &[u16], weight: &[u16], width: usize, sum: fn(&[f32]) -> f32) -> Vec<u16> {
    x.chunks_exact(width)
        .flat_map(|row| hf_norm(row, weight, LAB_EPS, sum))
        .collect()
}

/// One width and row kind: the tree entry against its emulation, at a live
/// count below the capacity.
fn bitwise_case(
    session: &Session,
    module: &Module,
    width: u32,
    kind: (&str, i32, i32),
) -> LabResult<Value> {
    let width_ = width as usize;
    let capacity = 64;
    let live = 61;
    let (name, low, high) = kind;
    let x = bf16_rows(capacity * width_, 0x7e11 + u64::from(width), low, high);
    let weight = bf16_rows(width_, 0x3e1 + u64::from(width), -2, 1);
    let fixture = Fixture::new(session, width_, &x, &weight)?;
    fixture.set_live(session, live)?;
    let (norm, function) = tree_entry(module, width)?;
    poison::poison(session, &[fixture.buffer(Y)])?;
    fixture.launch(session, &norm, &function, false)?;
    session.stream.synchronize().map_err(err)?;
    let actual = fixture.output(session)?;
    let expected = emulate(&x, &weight, width_, tree_sum);
    let label = format!("w{width} {name}");
    compare_rows(&label, width_, live, &actual, &expected, false)?;
    let serial = emulate(&x, &weight, width_, serial_sum);
    let differ = expected[..live * width_]
        .chunks_exact(width_)
        .zip(serial.chunks_exact(width_))
        .filter(|(t, s)| t != s)
        .count();
    Ok(json!({
        "width": width,
        "kind": name,
        "binades": [low, high],
        "capacity": capacity,
        "live": live,
        "bitwise_tree_emulation": true,
        "rows_past_live_untouched": true,
        "rows_where_serial_order_differs": differ,
    }))
}

/// Rows with NaN and infinite inputs at the hidden width: row r mod 3 is
/// ordinary (0), holds a NaN (1), or an infinity (2).
fn non_finite_case(session: &Session, module: &Module) -> LabResult<Value> {
    let capacity = 12;
    let mut x = bf16_rows(capacity * HIDDEN, 0x0a0f, -4, 2);
    for row in 0..capacity {
        let at = row * HIDDEN + (row * 131) % HIDDEN;
        match row % 3 {
            1 => x[at] = [0x7fc1, 0xffd3, 0x7f81][row % 3],
            2 => x[at] = if row % 2 == 0 { 0x7f80 } else { 0xff80 },
            _ => {}
        }
    }
    let weight = bf16_rows(HIDDEN, 0x3e1, -2, 1);
    let fixture = Fixture::new(session, HIDDEN, &x, &weight)?;
    let (norm, function) = tree_entry(module, HIDDEN as u32)?;
    poison::poison(session, &[fixture.buffer(Y)])?;
    fixture.launch(session, &norm, &function, false)?;
    session.stream.synchronize().map_err(err)?;
    let actual = fixture.output(session)?;
    let expected = emulate(&x, &weight, HIDDEN, tree_sum);
    compare_rows("non-finite", HIDDEN, capacity, &actual, &expected, true)?;
    let nans = actual.iter().filter(|&&v| nan(v)).count();
    if nans == 0 {
        return Err("non-finite rows gave no NaN outputs".into());
    }
    Ok(json!({"rows": capacity, "nan_outputs": nans, "nan_aware_bitwise": true}))
}

/// The finite row kinds of the paired oracle comparison: representative
/// rows and order-sensitive ones.
const ORACLE_FINITE: [&str; 6] = [
    "ordinary",
    "wide",
    "activation-like",
    "order-sensitive",
    "eps-scale",
    "subnormal-squares",
];
/// Its exceptional row kinds, reported apart from the error statistics.
const ORACLE_EXCEPTIONAL: [&str; 3] = ["zero", "overflow", "nan"];

/// `rows` hidden-width rows of an oracle kind:
/// - ordinary: binades -4 to 2, with zeros, negative zeros and a BF16
///   subnormal every 97th value;
/// - wide: binades -30 to 30;
/// - activation-like: binades -6 to 0 with six outliers per row at binades
///   6 to 8;
/// - order-sensitive: positive values of binade 0 with column 0 at binade
///   13, whose square is 2^26 times the others';
/// - eps-scale: binades -12 to -6, so the mean square is of the order of
///   eps (1e-5);
/// - subnormal-squares: binades -74 to -64, every nonzero square an F32
///   subnormal;
/// - zero: +0 and -0 only;
/// - overflow: binades 60 to 63, the F32 sums overflow;
/// - nan: ordinary rows with one NaN each.
fn oracle_rows(kind: &str, rows: usize) -> Vec<u16> {
    let seed = 0x0dac + kind.len() as u64;
    match kind {
        "ordinary" => bf16_rows(rows * HIDDEN, seed, -4, 2),
        "wide" => bf16_rows(rows * HIDDEN, seed, -30, 30),
        "activation-like" => {
            let mut x = bf16_rows(rows * HIDDEN, seed, -6, 0);
            let outliers = bf16_rows(rows * 6, seed + 1, 6, 8);
            for row in 0..rows {
                for k in 0..6 {
                    let v = outliers[row * 6 + k];
                    if v & 0x7fff > 3 {
                        x[row * HIDDEN + (row * 37 + k * 683) % HIDDEN] = v;
                    }
                }
            }
            x
        }
        "order-sensitive" => bf16_rows(rows * HIDDEN, seed, 0, 0)
            .into_iter()
            .enumerate()
            .map(|(i, v)| {
                let mantissa = v & 0x7f;
                let exponent: u16 = if i % HIDDEN == 0 { 13 } else { 0 };
                ((exponent + 127) << 7) | mantissa
            })
            .collect(),
        "eps-scale" => bf16_rows(rows * HIDDEN, seed, -12, -6),
        "subnormal-squares" => bf16_rows(rows * HIDDEN, seed, -74, -64),
        "zero" => (0..rows * HIDDEN)
            .map(|i| if i % 3 == 0 { 0x8000 } else { 0 })
            .collect(),
        "overflow" => bf16_rows(rows * HIDDEN, seed, 60, 63),
        "nan" => {
            let mut x = bf16_rows(rows * HIDDEN, seed, -4, 2);
            for row in 0..rows {
                x[row * HIDDEN + (row * 131) % HIDDEN] = 0x7fc1;
            }
            x
        }
        _ => unreachable!("unknown oracle kind {kind}"),
    }
}

/// One kernel's finite outputs against the F64 oracle: the norm with the
/// exact sum of the exact squares and the exact reciprocal, the
/// normalized value rounded once to BF16 and its product with the weight
/// rounded to BF16 (the HF convention's roundings, each correctly
/// rounded).
#[derive(Clone, Copy, Default)]
struct OracleError {
    rows: usize,
    outputs: usize,
    /// |output - exact unrounded value|, over every output.
    abs_max: f64,
    abs_sum: f64,
    /// The same relative to |exact value|, over nonzero exact values.
    rel_max: f64,
    rel_sum: f64,
    rel_outputs: usize,
    /// Outputs other than the oracle's correctly rounded result, and
    /// |output - that result| over every output.
    mismatches: usize,
    off_max: f64,
    off_sum: f64,
    /// The F32 sum of squares in this kernel's order (its emulation, which
    /// the device outputs equal bit for bit) against the exact sum.
    sum_rel_max: f64,
    sum_rel_sum: f64,
}

impl OracleError {
    fn add(&mut self, row: &[u16], weight: &[u16], output: &[u16], sum: fn(&[f32]) -> f32) {
        let exact: f64 = row.iter().map(|&v| f64::from(bf16_to_f32(v)).powi(2)).sum();
        let squares: Vec<f32> = row
            .iter()
            .map(|&v| {
                let v = bf16_to_f32(v);
                v * v
            })
            .collect();
        let sum_rel = if exact > 0.0 {
            (f64::from(sum(&squares)) - exact).abs() / exact
        } else {
            0.0
        };
        self.sum_rel_max = self.sum_rel_max.max(sum_rel);
        self.sum_rel_sum += sum_rel;
        let inv = 1.0 / (exact / HIDDEN as f64 + f64::from(LAB_EPS)).sqrt();
        for ((&v, &w), &out) in row.iter().zip(weight).zip(output) {
            let x = f64::from(bf16_to_f32(v));
            let w32 = bf16_to_f32(w);
            let value = x * inv * f64::from(w32);
            let rounded = f32_to_bf16(bf16_to_f32(f64_to_bf16(x * inv)) * w32);
            let error = (f64::from(bf16_to_f32(out)) - value).abs();
            self.abs_max = self.abs_max.max(error);
            self.abs_sum += error;
            if value != 0.0 {
                let relative = error / value.abs();
                self.rel_max = self.rel_max.max(relative);
                self.rel_sum += relative;
                self.rel_outputs += 1;
            }
            self.mismatches += usize::from(out != rounded);
            let off = f64::from(bf16_to_f32(out)) - f64::from(bf16_to_f32(rounded));
            self.off_max = self.off_max.max(off.abs());
            self.off_sum += off.abs();
        }
        self.rows += 1;
        self.outputs += row.len();
    }

    fn json(&self) -> Value {
        json!({
            "rows": self.rows,
            "outputs": self.outputs,
            "abs_error_max": self.abs_max,
            "abs_error_mean": self.abs_sum / self.outputs as f64,
            "rel_error_max": self.rel_max,
            "rel_error_mean": self.rel_sum / self.rel_outputs.max(1) as f64,
            "bf16_mismatches": self.mismatches,
            "abs_off_rounded_max": self.off_max,
            "abs_off_rounded_mean": self.off_sum / self.outputs as f64,
            "sum_rel_error_max": self.sum_rel_max,
            "sum_rel_error_mean": self.sum_rel_sum / self.rows as f64,
        })
    }
}

/// The paired oracle comparison of the serial-order kernel (`serial`
/// outputs, the norm served before) and the tree kernel (`tree`) on the
/// rows `x` of `kind` with `weight`. Finite kinds: the error statistics,
/// and the v2 condition that the tree is no less accurate: no more outputs
/// off the correctly rounded result, and the sum's maximum and mean
/// relative error no greater than the serial order's. The output errors
/// against the unrounded value are reported, not compared: they are the
/// BF16 rounding floor, and with the HF convention's two roundings an
/// output off the correctly rounded result can lie nearer the unrounded
/// value.
/// Exceptional kinds: NaN, zero and nonzero output counts per kernel and
/// the outputs where the two kernels differ.
fn oracle_pair(
    kind: &str,
    x: &[u16],
    weight: &[u16],
    serial: &[u16],
    tree: &[u16],
) -> LabResult<Value> {
    if ORACLE_EXCEPTIONAL.contains(&kind) {
        let counts = |out: &[u16]| {
            json!({
                "nan_outputs": out.iter().filter(|&&v| nan(v)).count(),
                "zero_outputs": out.iter().filter(|&&v| v & 0x7fff == 0).count(),
                "other_outputs": out.iter().filter(|&&v| !nan(v) && v & 0x7fff != 0).count(),
            })
        };
        let differ = serial
            .iter()
            .zip(tree)
            .filter(|(s, t)| s != t && !(nan(**s) && nan(**t)))
            .count();
        return Ok(json!({
            "kind": kind,
            "rows": x.len() / HIDDEN,
            "serial": counts(serial),
            "tree": counts(tree),
            "outputs_differing_between_kernels": differ,
        }));
    }
    let mut errors = [OracleError::default(); 2];
    for (r, row) in x.chunks_exact(HIDDEN).enumerate() {
        let at = r * HIDDEN..(r + 1) * HIDDEN;
        errors[0].add(row, weight, &serial[at.clone()], serial_sum);
        errors[1].add(row, weight, &tree[at], tree_sum);
    }
    let [s, t] = errors;
    let no_less_accurate = t.mismatches <= s.mismatches
        && t.sum_rel_max <= s.sum_rel_max
        && t.sum_rel_sum <= s.sum_rel_sum;
    if !no_less_accurate {
        return Err(format!(
            "{kind}: the tree is less accurate than the serial order: serial {}, tree {}",
            s.json(),
            t.json()
        ));
    }
    Ok(json!({
        "kind": kind,
        "serial": s.json(),
        "tree": t.json(),
        "tree_no_less_accurate": true,
    }))
}

/// The tree's and the serial order's sums and outputs against the exact
/// sum on `rows` rows of the hidden width over `low..=high`.
fn accuracy_case(name: &str, rows: usize, low: i32, high: i32) -> LabResult<Value> {
    let x = bf16_rows(rows * HIDDEN, 0xacc0 + low.unsigned_abs() as u64, low, high);
    let weight = bf16_rows(HIDDEN, 0x3e1, -2, 1);
    let bound = 2.0_f64.powi(-19);
    let (mut tree_max, mut serial_max, mut tree_mean, mut serial_mean) = (0.0, 0.0, 0.0, 0.0);
    let (mut tree_off, mut serial_off) = (0_usize, 0_usize);
    for (r, row) in x.chunks_exact(HIDDEN).enumerate() {
        let squares: Vec<f32> = row
            .iter()
            .map(|&v| {
                let v = bf16_to_f32(v);
                v * v
            })
            .collect();
        let exact = exact_sum(&squares);
        let tree = (f64::from(tree_sum(&squares)) - exact).abs() / exact;
        let serial = (f64::from(serial_sum(&squares)) - exact).abs() / exact;
        if tree > bound {
            return Err(format!(
                "{name} row {r}: the tree's relative error {tree:e} exceeds 2^-19"
            ));
        }
        tree_max = f64::max(tree_max, tree);
        serial_max = f64::max(serial_max, serial);
        tree_mean += tree / rows as f64;
        serial_mean += serial / rows as f64;
        // The norm with the exact reciprocal, rounded as the kernel rounds.
        let inv = 1.0 / (exact / HIDDEN as f64 + f64::from(LAB_EPS)).sqrt();
        let oracle: Vec<u16> = row
            .iter()
            .zip(&weight)
            .map(|(&v, &w)| {
                let n = bf16_to_f32(f64_to_bf16(f64::from(bf16_to_f32(v)) * inv));
                f32_to_bf16(n * bf16_to_f32(w))
            })
            .collect();
        let off = |out: Vec<u16>| out.iter().zip(&oracle).filter(|(a, b)| a != b).count();
        tree_off += off(hf_norm(row, &weight, LAB_EPS, tree_sum));
        serial_off += off(hf_norm(row, &weight, LAB_EPS, serial_sum));
    }
    Ok(json!({
        "kind": name,
        "rows": rows,
        "binades": [low, high],
        "tree_relative_error_max": tree_max,
        "tree_relative_error_mean": tree_mean,
        "serial_relative_error_max": serial_max,
        "serial_relative_error_mean": serial_mean,
        "tree_bound": bound,
        "outputs_off_exact_reciprocal_tree": tree_off,
        "outputs_off_exact_reciprocal_serial": serial_off,
        "outputs": rows * HIDDEN,
    }))
}

/// Poisoned eager, relaunch, graph and PDL launches of the hidden width's
/// tree entry at every live count 0 to 32 of a capacity-32 bucket.
fn replay_case(session: &Session, module: &Module) -> LabResult<Value> {
    let x = bf16_rows(FULL_REPLAY_ROWS * HIDDEN, 0xfeed, -4, 2);
    let weight = bf16_rows(HIDDEN, 0x3e1, -2, 1);
    let fixture = Fixture::new(session, HIDDEN, &x, &weight)?;
    let (norm, function) = tree_entry(module, HIDDEN as u32)?;
    let expected = emulate(&x, &weight, HIDDEN, tree_sum);
    let graph = session.capture(|_| fixture.launch(session, &norm, &function, false))?;
    let producer = LiveProducer::new(session)?;
    let rewrites = Rewrites::new(session, &[(fixture.buffer(X_IMAGE), fixture.buffer(X))])?;
    let mut lives = Vec::new();
    for live in 0..=FULL_REPLAY_ROWS {
        fixture.set_live(session, live)?;
        let launch = || fixture.launch(session, &norm, &function, false);
        let mut replay = Replay::new(session).output(
            "y",
            fixture.buffer(Y),
            WriteMap::of(DType::Bf16).prefix(live * HIDDEN)?,
        );
        replay.eager(launch)?;
        compare_rows(
            &format!("replay live {live}"),
            HIDDEN,
            live,
            &fixture.output(session)?,
            &expected,
            false,
        )?;
        replay.relaunch(launch)?;
        replay.graph(&graph)?;
        replay.pdl(
            || producer.launch_rewriting(session, fixture.buffer(LIVE), &rewrites),
            || fixture.launch(session, &norm, &function, true),
            true,
        )?;
        let mut receipt = BTreeMap::new();
        replay.record(&mut receipt);
        for flag in ["deterministic_bitwise", "graph_bitwise", "pdl_bitwise"] {
            if receipt.get(flag) != Some(&json!(true)) {
                return Err(format!("live {live}: no {flag} in {receipt:?}"));
            }
        }
        lives.push(json!({"live": live, "replay": receipt}));
    }
    Ok(json!({"capacity": FULL_REPLAY_ROWS, "live_counts": lives}))
}

/// The reversed-distances mutant against the served entry's bits, on the
/// rows the CPU proves the two orders tell apart: the served rows equal the
/// tree's emulation, the mutant's the reversed order's.
fn mutant_case(session: &Session, module: &Module) -> LabResult<Value> {
    let capacity = 32;
    let x = tree_mutant_rows::rows(capacity);
    let weight = tree_mutant_rows::weight();
    let (tree, reversed) = tree_mutant_rows::norms(&x, LAB_EPS);
    let fixture = Fixture::new(session, HIDDEN, &x, &weight)?;
    let (norm, function) = tree_entry(module, HIDDEN as u32)?;
    poison::poison(session, &[fixture.buffer(Y)])?;
    fixture.launch(session, &norm, &function, false)?;
    session.stream.synchronize().map_err(err)?;
    let served = fixture.output(session)?;
    compare_rows(
        "mutant rows, served entry",
        HIDDEN,
        capacity,
        &served,
        &tree,
        false,
    )?;
    let mutant_module = session.module(mutant_cubin())?;
    let (mutant, symbol) = rmsnorm_tree::lab_norm(HIDDEN as u32, "_distances_reversed");
    let mutant_function = mutant_module.function(&symbol).map_err(err)?;
    poison::poison(session, &[fixture.buffer(Y)])?;
    fixture.launch(session, &mutant, &mutant_function, false)?;
    session.stream.synchronize().map_err(err)?;
    let output = fixture.output(session)?;
    let changed = output
        .chunks_exact(HIDDEN)
        .zip(served.chunks_exact(HIDDEN))
        .filter(|(m, s)| m != s)
        .count();
    if changed != capacity {
        return Err(format!(
            "the reversed-distances mutant changed {changed} of {capacity} rows"
        ));
    }
    compare_rows(
        "mutant rows, mutant entry",
        HIDDEN,
        capacity,
        &output,
        &reversed,
        false,
    )?;
    Ok(json!({
        "mutation": "warp tree distances reversed",
        "outcome": "caught",
        "rows_changed": changed,
        "rows": capacity,
    }))
}

#[test]
fn rmsnorm_tree_cpu_contract() {
    let mut expected = LAB_WIDTHS
        .iter()
        .map(|&width| rmsnorm_tree::lab_norm(width, "").1)
        .collect::<Vec<_>>();
    expected.push(rmsnorm_tree::lab_ordered_norm().1);
    assert_eq!(kernel_lab::cubin_entries(served_cubin()).unwrap(), expected);
    assert_eq!(
        kernel_lab::cubin_entries(mutant_cubin()).unwrap(),
        [rmsnorm_tree::lab_norm(4096, "_distances_reversed").1]
    );
    kernel_lab::cubin(served_cubin()).unwrap();
    let rows = bf16_rows(4 * 97, 1, -4, 2);
    assert_eq!(rows[0], 0);
    assert_eq!(rows[1], 0x8000);
    assert_eq!(rows[2], 3);
    assert!(
        bf16_rows(4096, 2, 55, 63)
            .iter()
            .filter(|&&v| v & 0x7fff > 3)
            .all(|&v| (182..=190).contains(&((v >> 7) & 0xff)))
    );
    // Rows past the live count must stay poisoned; NaN compares as NaN.
    assert!(compare_rows("t", 2, 1, &[1, 2, 0xffff, 0xffff], &[1, 2, 9, 9], false).is_ok());
    assert!(compare_rows("t", 2, 1, &[1, 2, 0xffff, 3], &[1, 2, 9, 9], false).is_err());
    assert!(compare_rows("t", 2, 2, &[1, 0x7fc0, 1, 1], &[1, 0x7fc1, 1, 1], true).is_ok());
    assert!(compare_rows("t", 2, 2, &[1, 0x7fc0, 1, 1], &[1, 0x7fc1, 1, 1], false).is_err());
    let accuracy = accuracy_case("ordinary", 4, -4, 2).unwrap();
    // The paired oracle comparison on the emulations, which the device
    // outputs must equal: the tree no less accurate on every finite kind.
    let weight = bf16_rows(HIDDEN, 0x3e1, -2, 1);
    for kind in ORACLE_FINITE.iter().chain(&ORACLE_EXCEPTIONAL) {
        let x = oracle_rows(kind, 128);
        let pair = oracle_pair(
            kind,
            &x,
            &weight,
            &emulate(&x, &weight, HIDDEN, serial_sum),
            &emulate(&x, &weight, HIDDEN, tree_sum),
        )
        .unwrap();
        eprintln!("oracle {pair}");
    }
    assert!(
        oracle_rows("subnormal-squares", 1)
            .iter()
            .map(|&v| bf16_to_f32(v) * bf16_to_f32(v))
            .any(|q| q != 0.0 && q < f32::MIN_POSITIVE)
    );
    assert!(
        oracle_rows("overflow", 1)
            .iter()
            .map(|&v| bf16_to_f32(v) * bf16_to_f32(v))
            .sum::<f32>()
            .is_infinite()
    );
    // The mutant's rows tell the two orders apart on the CPU.
    let changed = tree_mutant_rows::proof(LAB_EPS).unwrap();
    eprintln!("tree mutant rows: outputs changed (overflow, probe row): {changed:?}");
    assert!(accuracy["tree_relative_error_max"].as_f64().unwrap() <= 2.0_f64.powi(-19));
}

#[test]
fn rmsnorm_tree_matches_its_emulation_bitwise() {
    kernel_lab::gpu_test(2 << 30, |session| {
        let module = session.module(served_cubin())?;
        let mut cases = Vec::new();
        for &width in &LAB_WIDTHS {
            for kind in KINDS {
                cases.push(bitwise_case(session, &module, width, kind)?);
            }
        }
        // The serial-order baseline against the serial order's emulation.
        let x = bf16_rows(16 * HIDDEN, 0x5e1, -4, 2);
        let weight = bf16_rows(HIDDEN, 0x3e1, -2, 1);
        let fixture = Fixture::new(session, HIDDEN, &x, &weight)?;
        let (ordered, symbol) = rmsnorm_tree::lab_ordered_norm();
        let function = module.function(&symbol).map_err(err)?;
        poison::poison(session, &[fixture.buffer(Y)])?;
        fixture.launch(session, &ordered, &function, false)?;
        session.stream.synchronize().map_err(err)?;
        compare_rows(
            "serial-order baseline",
            HIDDEN,
            16,
            &fixture.output(session)?,
            &emulate(&x, &weight, HIDDEN, serial_sum),
            false,
        )?;
        let non_finite = non_finite_case(session, &module)?;
        let accuracy = [
            accuracy_case("ordinary", 2048, -4, 2)?,
            accuracy_case("wide", 2048, -30, 30)?,
        ];
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("rmsnorm_tree_bitwise"));
        receipt.insert("cases".into(), json!(cases));
        receipt.insert("serial_baseline_bitwise_emulation".into(), json!(true));
        receipt.insert("non_finite".into(), non_finite);
        receipt.insert("accuracy".into(), json!(accuracy));
        session.receipt("rmsnorm_tree_bitwise", receipt)?;
        Ok(())
    })
    .expect("tree RMSNorm bitwise comparison");
}

/// The paired oracle comparison on device outputs: per kind, 2048 rows
/// through the serial-order kernel (the HF norm served before) and the tree
/// kernel, each output set equal to its emulation (NaN-aware on the
/// exceptional kinds), then both against the F64 oracle (`oracle_pair`).
#[test]
fn rmsnorm_tree_oracle_accuracy_paired() {
    kernel_lab::gpu_test(2 << 30, |session| {
        let module = session.module(served_cubin())?;
        let (tree, tree_function) = tree_entry(&module, HIDDEN as u32)?;
        let (ordered, symbol) = rmsnorm_tree::lab_ordered_norm();
        let ordered_function = module.function(&symbol).map_err(err)?;
        let weight = bf16_rows(HIDDEN, 0x3e1, -2, 1);
        let rows = 2048;
        let mut kinds = Vec::new();
        for kind in ORACLE_FINITE.iter().chain(&ORACLE_EXCEPTIONAL) {
            let x = oracle_rows(kind, rows);
            let exceptional = ORACLE_EXCEPTIONAL.contains(kind);
            let fixture = Fixture::new(session, HIDDEN, &x, &weight)?;
            let mut outputs = Vec::new();
            for (name, norm, function, sum) in [
                (
                    "serial",
                    &ordered,
                    &ordered_function,
                    serial_sum as fn(&[f32]) -> f32,
                ),
                ("tree", &tree, &tree_function, tree_sum),
            ] {
                poison::poison(session, &[fixture.buffer(Y)])?;
                fixture.launch(session, norm, function, false)?;
                session.stream.synchronize().map_err(err)?;
                let output = fixture.output(session)?;
                compare_rows(
                    &format!("oracle {kind}, {name} kernel"),
                    HIDDEN,
                    rows,
                    &output,
                    &emulate(&x, &weight, HIDDEN, sum),
                    exceptional,
                )?;
                outputs.push(output);
            }
            let pair = oracle_pair(kind, &x, &weight, &outputs[0], &outputs[1])?;
            eprintln!("rmsnorm_tree oracle: {pair}");
            kinds.push(pair);
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("rmsnorm_tree_oracle_paired"));
        receipt.insert("kinds".into(), json!(kinds));
        session.receipt("rmsnorm_tree_oracle_paired", receipt)?;
        Ok(())
    })
    .expect("tree RMSNorm paired oracle accuracy");
}

#[test]
fn rmsnorm_tree_replays_on_poisoned_buffers() {
    kernel_lab::gpu_test(1 << 30, |session| {
        let module = session.module(served_cubin())?;
        let replays = replay_case(session, &module)?;
        let mutant = mutant_case(session, &module)?;
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("rmsnorm_tree_poison"));
        receipt.insert("replays".into(), replays);
        receipt.insert("mutant".into(), mutant);
        session.receipt("rmsnorm_tree_poison", receipt)?;
        Ok(())
    })
    .expect("tree RMSNorm poisoned replays");
}

#[test]
fn rmsnorm_tree_timing() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("rmsnorm_tree timing: set CHOPINFER_KERNEL_BENCH=1").unwrap();
        return;
    }
    kernel_lab::gpu_test(2 << 30, |session| {
        let module = session.module(served_cubin())?;
        let (tree, tree_function) = tree_entry(&module, HIDDEN as u32)?;
        let (ordered, symbol) = rmsnorm_tree::lab_ordered_norm();
        let ordered_function = module.function(&symbol).map_err(err)?;
        let mut rows = Vec::new();
        for capacity in [1, 32, 1024, 8192] {
            let x = bf16_rows(capacity * HIDDEN, 0x7193 + capacity as u64, -4, 2);
            let weight = bf16_rows(HIDDEN, 0x3e1, -2, 1);
            let fixture = Fixture::new(session, HIDDEN, &x, &weight)?;
            let graphs =
                [(&ordered, &ordered_function), (&tree, &tree_function)].map(|(norm, function)| {
                    session.capture(|_| {
                        for _ in 0..8 {
                            fixture.launch(session, norm, function, true)?;
                        }
                        Ok(())
                    })
                });
            let [ordered_graph, tree_graph] = graphs;
            let graphs = [ordered_graph?, tree_graph?];
            let paired = kernel_lab::time_ab(
                session,
                3,
                15,
                [kernel_lab::eviction::EvictionMode::Write; 2],
                |arm, stream| graphs[arm].launch(stream).map_err(err),
            )?;
            let per = |ms: f32| f64::from(ms) * 1e3 / 8.0;
            let row = json!({
                "rows": capacity,
                "ordered_us_per_launch": per(paired.arms[0].median_ms),
                "tree_us_per_launch": per(paired.arms[1].median_ms),
                "tree_over_ordered": paired.paired_ratio,
            });
            eprintln!("rmsnorm_tree timing: {row}");
            rows.push(row);
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("rmsnorm_tree_timing"));
        receipt.insert("buckets".into(), json!(rows));
        session.receipt("rmsnorm_tree_timing", receipt)?;
        Ok(())
    })
    .expect("tree RMSNorm timing");
}
