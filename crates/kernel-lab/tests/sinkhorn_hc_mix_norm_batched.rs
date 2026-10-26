//! The split Sinkhorn mix-collapse fused with the input RMSNorm on batched
//! loads (`kernels_sm120::sinkhorn_hc::mix_norm_batched`) against the
//! served entry it replaces, the exact-sum fused entry
//! (`sinkhorn_hc::mix_norm_exact`), at the served 4 x 4096 mixer, on the
//! GPU. Run with
//! `CHOPINFER_KERNEL_GPU=<ordinal> cargo test -p kernel-lab --release --test
//! sinkhorn_hc_mix_norm_batched -- --test-threads=1`; without a GPU only the
//! CPU contract test runs.
//!
//! After an expand writes the streams as a sublayer boundary does:
//!
//! - at every bucket of 1 to 32 rows (every live count M1 to M32) and the
//!   verify bucket of 6 x 32: collapsed, post, comb, the normed row and the
//!   chain sums bitwise equal to the exact-sum entry's, written into freshly
//!   poisoned buffers that alone fail the comparison; inputs of ordinary,
//!   wide (squares over 2^-60 to 2^60), extreme common offset and
//!   order-sensitive (column 0 2^13 times the rest) magnitudes;
//! - determinism: relaunches rewrite every output and chain sum with the
//!   same bits and leave every ticket at zero;
//! - poisoned replays (eager, relaunch, graph, PDL): at 32 rows (ordinary,
//!   wide and extreme inputs) one graph captured at capacity 32 replayed at
//!   every live count 0 to 32 (the receipt's covered set is asserted), and
//!   at the live counts `poison::live_counts` lists for 1, 2, 3, 4, 8, 16
//!   and 192 rows, after the producer whose streams are poisoned and
//!   checked too, the chain sums poisoned as a written workspace and the
//!   tickets checked as zeroed state; each live count's eager outputs
//!   compared with the exact-sum entry at that live count; with the
//!   producer left out, the streams, the collapsed and the normed rows must
//!   differ from the eager bits;
//! - mutants, each caught only when an output or the chain sums differ
//!   from known good bits (launch, synchronization and readback errors
//!   propagate): the chain sums' butterfly levels reversed
//!   (`_levels_reversed`; the chain sums must differ), the ordered sum's
//!   ties rounded away from zero (`_ties_away`; the normed rows alone must
//!   differ), and the dependency wait after the chain sums' reads
//!   (`_late_wait`) behind the lab's PDL live-count producer, reported
//!   inconclusive when it reproduces the eager bits (overlap is not
//!   measured).
//!
//! With `CHOPINFER_KERNEL_BENCH=1`, `sinkhorn_hc_mix_norm_batched_timing`
//! times graphs of eight sublayer boundaries (expand, then the exact-sum
//! entry or this one; PDL as served) at the decode buckets.

use kernel_core::codec::f32_to_bf16;
use kernel_core::contract::{Argument, ArgumentList, DType, LaunchPlan};
use kernel_lab::launch::{self, Bindings};
use kernel_lab::poison::{self, LiveProducer, Phase, Replay, Rewrites, WriteMap};
use kernel_lab::{LabResult, Session};
use kernels_sm120::sinkhorn_hc::{
    self, Op, SERVED, mix_norm, mix_norm_batched, mix_norm_exact, split,
};
use runtime_cuda::{Function, Module};
use serde_json::{Value, json};
use std::collections::BTreeMap;

const N: usize = 4;
const H: usize = 4096;
const FLAT: usize = N * H;
const MIXES: usize = (2 + N) * N;
const CELLS: usize = N * N;
const CHAINS: usize = FLAT / 128;
const GROUPS: usize = CHAINS / 8;
const EPS: f32 = 1e-5;

// Buffer handles.
const X0: u64 = 0;
const Y0: u64 = 1;
const POST0: u64 = 2;
const COMB0: u64 = 3;
const X1: u64 = 4;
const FN: u64 = 5;
const BASE: u64 = 6;
const SCALE: u64 = 7;
const WEIGHT: u64 = 8;
const LIVE: u64 = 9;
const REF_COLLAPSED: u64 = 10;
const REF_POST: u64 = 11;
const REF_COMB: u64 = 12;
const REF_NORMED: u64 = 13;
const REF_PARTIALS: u64 = 14;
const REF_TICKETS: u64 = 15;
const COLLAPSED: u64 = 16;
const POST: u64 = 17;
const COMB: u64 = 18;
const NORMED: u64 = 19;
const PARTIALS: u64 = 20;
const TICKETS: u64 = 21;
/// The streams' image, which the live-count producer restores.
const X1_IMAGE: u64 = 22;
const OUTPUTS: [u64; 4] = [COLLAPSED, POST, COMB, NORMED];
const REF_OUTPUTS: [u64; 4] = [REF_COLLAPSED, REF_POST, REF_COMB, REF_NORMED];
const OUTPUT_NAMES: [&str; 4] = ["collapsed", "post", "comb", "normed"];
const PER_ROW: [usize; 4] = [H, N, CELLS, H];
/// Sublayer boundaries per timed graph.
const BOUNDARIES: usize = 8;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn tag() -> &'static str {
    sinkhorn_hc::SERVED_PROFILE.tag()
}

fn served_cubin() -> String {
    format!("sinkhorn_hc_{}_sm120", tag())
}

/// The exact-sum fused entry's lab cubin.
fn exact_cubin() -> String {
    format!("sinkhorn_hc_{}_mix_norm_exact_sm120", tag())
}

fn batched_cubin(suffix: &str) -> String {
    format!("sinkhorn_hc_{}_mix_norm_batched{suffix}_sm120", tag())
}

fn mutant(suffix: &str) -> mix_norm_batched::Entry {
    mix_norm_batched::entry_with_suffix(
        sinkhorn_hc::SERVED_PROFILE,
        SERVED,
        mix_norm::served_norm(),
        suffix,
    )
}

fn expand_entry() -> sinkhorn_hc::Entry {
    sinkhorn_hc::lab_catalog()
        .into_iter()
        .find(|entry| entry.op == Op::Expand)
        .expect("the expand entry")
}

/// Deterministic BF16 bits: random signs and mantissas over the binades
/// `low..=high`, with zeros, negative zeros and a subnormal among them
/// (as `tests/sinkhorn_hc_split.rs` draws them).
fn bf16_values(len: usize, seed: u64, low: i32, high: i32) -> Vec<u16> {
    let mut state = seed | 1;
    (0..len)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            match i % 509 {
                0 => 0x0000,
                1 => 0x8000,
                2 => 0x0001,
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

/// Non-negative BF16 bits in [0, scale).
fn unit_values(len: usize, seed: u64, scale: f32) -> Vec<u16> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            f32_to_bf16(((state >> 40) % 4096) as f32 / 4096.0 * scale)
        })
        .collect()
}

#[derive(Clone)]
struct Inputs {
    x0: Vec<u16>,
    y0: Vec<u16>,
    post0: Vec<u16>,
    comb0: Vec<u16>,
    weights: Vec<u16>,
    base: Vec<f32>,
    scale: [f32; 3],
    norm_weight: Vec<u16>,
}

impl Inputs {
    fn new(rows: usize, seed: u64) -> Self {
        let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        let base = (0..MIXES)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 40) % 2048) as f32 / 1024.0 - 1.0
            })
            .collect();
        Self {
            x0: bf16_values(rows * FLAT, seed + 1, -4, 2),
            y0: bf16_values(rows * H, seed + 2, -6, 1),
            post0: unit_values(rows * N, seed + 3, 2.0),
            comb0: unit_values(rows * CELLS, seed + 4, 1.0),
            // Logits of order one at K = 16,384.
            weights: bf16_values(MIXES * FLAT, seed + 6, -10, -6),
            base,
            scale: [0.5, 0.75, 2.0],
            norm_weight: bf16_values(H, seed + 7, -2, 1),
        }
    }

    /// Streams over binades -30..=30: the collapsed squares span 2^-60 to
    /// 2^60, so the norm's ordered sum rounds at every magnitude and
    /// crosses many binades.
    fn wide(rows: usize, seed: u64) -> Self {
        Self {
            x0: bf16_values(rows * FLAT, seed + 1, -30, 30),
            y0: bf16_values(rows * H, seed + 2, -30, 30),
            ..Self::new(rows, seed)
        }
    }

    /// Positive streams and sublayer outputs whose column 0 is 2^13 times
    /// the other columns (`tests/sinkhorn_hc_mix_norm.rs`'s order-sensitive
    /// rows): in column order every later square rounds away against
    /// column 0's.
    fn order_sensitive(rows: usize, seed: u64) -> Self {
        let column_major = |len: usize, seed: u64| {
            let mut state = seed | 1;
            (0..len)
                .map(|i| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    let exponent: u16 = if i % H == 0 { 13 } else { 0 };
                    ((exponent + 127) << 7) | ((state >> 24) & 0x7f) as u16
                })
                .collect::<Vec<u16>>()
        };
        Self {
            x0: column_major(rows * FLAT, seed + 1),
            y0: column_major(rows * H, seed + 2),
            ..Self::new(rows, seed)
        }
    }

    /// Non-finite and overflowing rows, by row mod 6: 0 ordinary; 1 NaN
    /// streams with several payloads (quiet and signaling BF16 patterns,
    /// both signs); 2 a +Inf and a -Inf stream element; 3 streams over
    /// binades 100 to 127 (squares and sums overflow); 4 a NaN sublayer
    /// output; 5 +Inf and -Inf in the same column of two streams (the
    /// expand's Inf - Inf).
    fn non_finite(rows: usize, seed: u64) -> Self {
        let mut inputs = Self::new(rows, seed);
        let large = bf16_values(rows * FLAT, seed + 9, 100, 127);
        for row in 0..rows {
            let x = &mut inputs.x0[row * FLAT..(row + 1) * FLAT];
            let column = (row * 37) % H;
            match row % 6 {
                1 => {
                    for (k, nan) in [0x7fc1_u16, 0xffd3, 0x7f81, 0xff80 | 0x0042]
                        .iter()
                        .enumerate()
                    {
                        x[(k % N) * H + (column + 129 * k) % H] = *nan;
                    }
                }
                2 => {
                    x[column] = 0x7f80;
                    x[H + (column + 300) % H] = 0xff80;
                }
                3 => x.copy_from_slice(&large[row * FLAT..(row + 1) * FLAT]),
                4 => inputs.y0[row * H + column] = 0x7fc5,
                5 => {
                    x[column] = 0x7f80;
                    x[2 * H + column] = 0xff80;
                }
                _ => {}
            }
        }
        inputs
    }

    /// Every comb base at the F32 value 1e8 with a zero comb scale.
    fn extreme(rows: usize, seed: u64) -> Self {
        let mut inputs = Self::new(rows, seed);
        inputs.base[2 * N..].fill(1e8);
        inputs.scale[2] = 0.0;
        inputs
    }
}

/// An entry's launch: its plan at the bucket and its function.
struct Kernel {
    plan: LaunchPlan,
    function: Function,
}

struct Fixture {
    rows: usize,
    buffers: BTreeMap<u64, runtime_cuda::DeviceBuffer>,
    _modules: Vec<Module>,
    expand: Kernel,
    exact: Kernel,
    batched: Kernel,
    /// The mutants by suffix.
    mutants: BTreeMap<&'static str, Kernel>,
}

impl Fixture {
    fn new(session: &Session, rows: usize, inputs: &Inputs) -> LabResult<Self> {
        let served = session.module(&served_cubin())?;
        let exact_module = session.module(&exact_cubin())?;
        let batched_module = session.module(&batched_cubin(""))?;
        let expand = expand_entry();
        let exact_entry = mix_norm_exact::served();
        let batched_entry = mix_norm_batched::served();
        let mut modules = Vec::new();
        let mut mutants = BTreeMap::new();
        for (suffix, _) in mix_norm_batched::MUTANTS {
            let module = session.module(&batched_cubin(suffix))?;
            mutants.insert(
                suffix,
                Kernel {
                    plan: batched_entry.plan(rows as u32, 0).map_err(err)?,
                    function: module.function(&mutant(suffix).symbol).map_err(err)?,
                },
            );
            modules.push(module);
        }
        let mut buffers = BTreeMap::new();
        buffers.insert(X0, session.upload_bf16(&inputs.x0)?);
        buffers.insert(Y0, session.upload_bf16(&inputs.y0)?);
        buffers.insert(POST0, session.upload_bf16(&inputs.post0)?);
        buffers.insert(COMB0, session.upload_bf16(&inputs.comb0)?);
        buffers.insert(FN, session.upload_bf16(&inputs.weights)?);
        buffers.insert(BASE, session.upload_f32(&inputs.base)?);
        buffers.insert(SCALE, session.upload_f32(&inputs.scale)?);
        buffers.insert(WEIGHT, session.upload_bf16(&inputs.norm_weight)?);
        buffers.insert(REF_TICKETS, session.upload_f32(&vec![0.0; rows])?);
        buffers.insert(TICKETS, session.upload_f32(&vec![0.0; rows])?);
        let partials = split::partials_bytes(SERVED.mix, rows as u32) as usize;
        for (handle, bytes) in [
            (X1, rows * FLAT * 2),
            (X1_IMAGE, rows * FLAT * 2),
            (LIVE, 4),
            (REF_COLLAPSED, rows * H * 2),
            (REF_POST, rows * N * 2),
            (REF_COMB, rows * CELLS * 2),
            (REF_NORMED, rows * H * 2),
            (REF_PARTIALS, partials),
            (COLLAPSED, rows * H * 2),
            (POST, rows * N * 2),
            (COMB, rows * CELLS * 2),
            (NORMED, rows * H * 2),
            (PARTIALS, partials),
        ] {
            buffers.insert(handle, session.allocate(bytes)?);
        }
        let fixture = Self {
            rows,
            buffers,
            expand: Kernel {
                plan: expand.plan(rows as u32, 0).map_err(err)?,
                function: served.function(&expand.symbol).map_err(err)?,
            },
            exact: Kernel {
                plan: exact_entry.plan(rows as u32, 0).map_err(err)?,
                function: exact_module.function(&exact_entry.symbol).map_err(err)?,
            },
            batched: Kernel {
                plan: batched_entry.plan(rows as u32, 0).map_err(err)?,
                function: batched_module
                    .function(&batched_entry.symbol)
                    .map_err(err)?,
            },
            mutants,
            _modules: {
                modules.extend([served, exact_module, batched_module]);
                modules
            },
        };
        fixture.set_live(session, rows)?;
        Ok(fixture)
    }

    fn buffer(&self, handle: u64) -> &runtime_cuda::DeviceBuffer {
        &self.buffers[&handle]
    }

    fn set_live(&self, session: &Session, live: usize) -> LabResult<()> {
        poison::write(session, self.buffer(LIVE), 0, &(live as u32).to_ne_bytes())
    }

    fn launch(
        &self,
        session: &Session,
        kernel: &Kernel,
        args: ArgumentList,
        pdl: bool,
    ) -> LabResult<()> {
        let mut plan = kernel.plan;
        plan.pdl = pdl;
        let bindings = Bindings {
            buffers: self.buffers.iter().map(|(&k, v)| (k, v)).collect(),
        };
        launch::launch(
            &session.context,
            &kernel.function,
            &session.stream,
            plan,
            &args,
            &bindings,
        )
    }

    fn handles(handles: &[u64]) -> Vec<Argument> {
        handles.iter().map(|&h| Argument::DeviceBuffer(h)).collect()
    }

    /// The expand that writes the streams the mixes read.
    fn producer(&self, session: &Session, pdl: bool) -> LabResult<()> {
        let args = expand_entry()
            .pack_arguments(Self::handles(&[X0, Y0, POST0, COMB0, X1, LIVE]))
            .map_err(err)?;
        self.launch(session, &self.expand, args, pdl)
    }

    /// The fused ABI's handles, into the reference buffers or the tested
    /// ones.
    fn fused_handles(reference: bool) -> Vec<Argument> {
        let outputs = if reference {
            [
                REF_COLLAPSED,
                REF_POST,
                REF_COMB,
                REF_NORMED,
                REF_PARTIALS,
                REF_TICKETS,
            ]
        } else {
            [COLLAPSED, POST, COMB, NORMED, PARTIALS, TICKETS]
        };
        let mut handles = vec![X1, FN, BASE, SCALE, WEIGHT];
        handles.extend(outputs);
        handles.push(LIVE);
        let mut args = Self::handles(&handles);
        args.push(Argument::U32(EPS.to_bits()));
        args
    }

    /// The served entry this one replaces, the exact-sum fused entry, into
    /// the reference buffers.
    fn reference(&self, session: &Session, pdl: bool) -> LabResult<()> {
        let args = mix_norm_exact::served()
            .pack_arguments(Self::fused_handles(true))
            .map_err(err)?;
        self.launch(session, &self.exact, args, pdl)
    }

    fn batched(&self, session: &Session, pdl: bool) -> LabResult<()> {
        let args = mix_norm_batched::served()
            .pack_arguments(Self::fused_handles(false))
            .map_err(err)?;
        self.launch(session, &self.batched, args, pdl)
    }

    fn mutant(&self, session: &Session, suffix: &str, pdl: bool) -> LabResult<()> {
        let kernel = self
            .mutants
            .get(suffix)
            .ok_or_else(|| format!("no mutant {suffix}"))?;
        let args = mutant(suffix)
            .pack_arguments(Self::fused_handles(false))
            .map_err(err)?;
        self.launch(session, kernel, args, pdl)
    }

    fn outputs(&self, session: &Session, handles: [u64; 4]) -> LabResult<[Vec<u16>; 4]> {
        let read =
            |i: usize| session.download_bf16(self.buffer(handles[i]), self.rows * PER_ROW[i]);
        Ok([read(0)?, read(1)?, read(2)?, read(3)?])
    }

    fn partials(&self, session: &Session, handle: u64) -> LabResult<Vec<u32>> {
        Ok(session
            .download_f32(self.buffer(handle), self.rows * MIXES * CHAINS)?
            .iter()
            .map(|v| v.to_bits())
            .collect())
    }

    fn tickets(&self, session: &Session, handle: u64) -> LabResult<Vec<u8>> {
        poison::read(session, self.buffer(handle))
    }
}

fn exact<T: PartialEq + Copy + std::fmt::LowerHex>(
    label: &str,
    actual: &[T],
    expected: &[T],
) -> LabResult<()> {
    if actual.len() != expected.len() {
        return Err(format!(
            "{label}: {} elements, expected {}",
            actual.len(),
            expected.len()
        ));
    }
    match actual.iter().zip(expected).position(|(a, e)| a != e) {
        None => Ok(()),
        Some(at) => Err(format!(
            "{label}: element {at} is {:#x}, expected {:#x}",
            actual[at], expected[at]
        )),
    }
}

fn same_outputs(label: &str, actual: &[Vec<u16>; 4], expected: &[Vec<u16>; 4]) -> LabResult<()> {
    for ((name, actual), expected) in OUTPUT_NAMES.into_iter().zip(actual).zip(expected) {
        exact(&format!("{label} {name}"), actual, expected)?;
    }
    Ok(())
}

/// The names of the outputs that differ.
fn differing(actual: &[Vec<u16>; 4], expected: &[Vec<u16>; 4]) -> Vec<&'static str> {
    OUTPUT_NAMES
        .into_iter()
        .zip(actual.iter().zip(expected))
        .filter(|(_, (a, e))| a != e)
        .map(|(name, _)| name)
        .collect()
}

/// Bitwise and determinism checks of one bucket; its receipt row.
fn bitwise_bucket(session: &Session, rows: usize, inputs: &Inputs) -> LabResult<Value> {
    let fixture = Fixture::new(session, rows, inputs)?;
    let zeroed = vec![0_u8; rows * 4];
    let refs =
        [REF_COLLAPSED, REF_POST, REF_COMB, REF_NORMED, REF_PARTIALS].map(|h| fixture.buffer(h));
    fixture.producer(session, false)?;
    poison::poison(session, &refs)?;
    fixture.reference(session, false)?;
    session.stream.synchronize().map_err(err)?;
    let reference = fixture.outputs(session, REF_OUTPUTS)?;
    let reference_partials = fixture.partials(session, REF_PARTIALS)?;
    exact(
        "reference tickets",
        &fixture.tickets(session, REF_TICKETS)?,
        &zeroed,
    )?;

    // The batched entry into poisoned outputs, which alone fail the
    // comparison.
    let outputs = [COLLAPSED, POST, COMB, NORMED, PARTIALS].map(|h| fixture.buffer(h));
    poison::poison(session, &outputs)?;
    let unwritten = fixture.outputs(session, OUTPUTS)?;
    if differing(&unwritten, &reference).len() != OUTPUTS.len() {
        return Err("unwritten outputs matched the reference".into());
    }
    if fixture.partials(session, PARTIALS)? == reference_partials {
        return Err("unwritten chain sums matched the reference".into());
    }
    fixture.batched(session, false)?;
    session.stream.synchronize().map_err(err)?;
    same_outputs(
        "batched entry against the exact-sum entry",
        &fixture.outputs(session, OUTPUTS)?,
        &reference,
    )?;
    exact(
        "batched chain sums against the exact-sum entry's",
        &fixture.partials(session, PARTIALS)?,
        &reference_partials,
    )?;
    exact(
        "tickets after a launch",
        &fixture.tickets(session, TICKETS)?,
        &zeroed,
    )?;

    // Determinism: relaunches into poisoned outputs give the same bits.
    for relaunch in 0..4 {
        poison::poison(session, &outputs)?;
        fixture.batched(session, relaunch % 2 == 1)?;
        session.stream.synchronize().map_err(err)?;
        same_outputs(
            &format!("relaunch {relaunch}"),
            &fixture.outputs(session, OUTPUTS)?,
            &reference,
        )?;
        exact(
            &format!("relaunch {relaunch} chain sums"),
            &fixture.partials(session, PARTIALS)?,
            &reference_partials,
        )?;
        exact(
            "tickets after a relaunch",
            &fixture.tickets(session, TICKETS)?,
            &zeroed,
        )?;
    }

    Ok(json!({
        "rows": rows,
        "grid": fixture.batched.plan.grid,
        "non_finite": non_finite_counts(&reference, &reference_partials),
        "bitwise_exact_sum_entry": true,
        "chain_sums_bitwise": true,
        "relaunches_bitwise": 4,
    }))
}

/// The NaN and infinite elements of each output and of the chain sums.
fn non_finite_counts(outputs: &[Vec<u16>; 4], partials: &[u32]) -> Value {
    let bf16 = |values: &[u16]| {
        let nan = values.iter().filter(|&&v| v & 0x7fff > 0x7f80).count();
        let inf = values.iter().filter(|&&v| v & 0x7fff == 0x7f80).count();
        json!({"nan": nan, "inf": inf})
    };
    let mut counts = serde_json::Map::new();
    for (name, values) in OUTPUT_NAMES.into_iter().zip(outputs) {
        counts.insert(name.into(), bf16(values));
    }
    let nan = partials
        .iter()
        .filter(|&&v| v & 0x7fff_ffff > 0x7f80_0000)
        .count();
    let inf = partials
        .iter()
        .filter(|&&v| v & 0x7fff_ffff == 0x7f80_0000)
        .count();
    counts.insert("chain sums".into(), json!({"nan": nan, "inf": inf}));
    Value::Object(counts)
}

/// The non-finite cases that must occur in the reference (exact-sum
/// entry) outputs of `Inputs::non_finite`, so the bitwise comparison covers
/// NaN and infinite bit patterns: (output, kind).
const NON_FINITE_REQUIRED: [(&str, &str); 6] = [
    ("chain sums", "nan"),
    ("chain sums", "inf"),
    ("collapsed", "nan"),
    ("post", "nan"),
    ("comb", "nan"),
    ("normed", "nan"),
];

/// The batched entry's live rows against the exact-sum entry over the same
/// streams at the same live count, and the tickets left zero.
fn against_reference(
    fixture: &Fixture,
    session: &Session,
    live: usize,
    label: &str,
) -> LabResult<()> {
    session.stream.synchronize().map_err(err)?;
    let refs = REF_OUTPUTS.map(|h| fixture.buffer(h));
    poison::poison(session, &refs)?;
    fixture.reference(session, false)?;
    session.stream.synchronize().map_err(err)?;
    let take = |handles: [u64; 4]| -> LabResult<[Vec<u16>; 4]> {
        let mut outputs = fixture.outputs(session, handles)?;
        for (output, per_row) in outputs.iter_mut().zip(PER_ROW) {
            output.truncate(live * per_row);
        }
        Ok(outputs)
    };
    same_outputs(
        &format!("{label}: batched entry against the exact-sum entry"),
        &take(OUTPUTS)?,
        &take(REF_OUTPUTS)?,
    )?;
    let partials = live * MIXES * CHAINS;
    exact(
        &format!("{label}: chain sums"),
        &fixture.partials(session, PARTIALS)?[..partials],
        &fixture.partials(session, REF_PARTIALS)?[..partials],
    )?;
    exact(
        &format!("{label}: tickets"),
        &fixture.tickets(session, TICKETS)?,
        &vec![0; fixture.rows * 4],
    )
}

fn replay_of<'a>(fixture: &'a Fixture, session: &'a Session, live: usize) -> LabResult<Replay<'a>> {
    let partial_row = split::partials_bytes(SERVED.mix, 1) as usize;
    let mut replay = Replay::new(session).output(
        "x1",
        fixture.buffer(X1),
        WriteMap::of(DType::Bf16).prefix(live * FLAT)?,
    );
    for ((name, handle), per_row) in OUTPUT_NAMES.into_iter().zip(OUTPUTS).zip(PER_ROW) {
        replay = replay.output(
            name,
            fixture.buffer(handle),
            WriteMap::of(DType::Bf16).prefix(live * per_row)?,
        );
    }
    Ok(replay
        .output(
            "partials",
            fixture.buffer(PARTIALS),
            WriteMap::of(DType::F32).byte_prefix(live * partial_row)?,
        )
        .state(
            "tickets",
            fixture.buffer(TICKETS),
            vec![0; fixture.rows * 4],
            WriteMap::of(DType::I32).none()?,
            Vec::new(),
        ))
}

/// The live counts the poisoned replays run at a bucket of `rows`: every
/// count 0 to 32 at the capacity-32 bucket (one graph captured at capacity
/// 32 replayed at each), `poison::live_counts` elsewhere.
fn replay_lives(rows: usize) -> Vec<u32> {
    if rows == FULL_REPLAY_ROWS {
        (0..=rows as u32).collect()
    } else {
        poison::live_counts(rows as u32)
    }
}

/// The bucket whose replays cover every live count.
const FULL_REPLAY_ROWS: usize = 32;

/// The phases every replayed live count runs, in the receipt's
/// `phases_run`.
const REPLAY_PHASES: [&str; 4] = ["eager", "relaunch", "graph", "pdl"];

/// The batched entry on poisoned buffers at each of `replay_lives(rows)`,
/// after its producer (an expand) whose streams are poisoned and checked
/// too. The graph is captured once at the bucket's capacity and replayed at
/// every live count. Each live count's eager outputs are also compared with
/// the exact-sum entry, and at one partial live count the producer left out
/// must leave the streams poisoned and change the collapsed and normed
/// rows.
fn replay_bucket(session: &Session, rows: usize, inputs: &Inputs) -> LabResult<Value> {
    let fixture = Fixture::new(session, rows, inputs)?;
    let graph = session.capture(|_| {
        fixture.producer(session, false)?;
        fixture.batched(session, true)
    })?;
    let mut evidence = Vec::new();
    for live in replay_lives(rows) {
        let live = live as usize;
        let label = format!("{rows} rows, live {live}");
        fixture.set_live(session, live)?;
        let both = || -> LabResult<()> {
            fixture.producer(session, false)?;
            fixture.batched(session, false)
        };
        let mut replay = replay_of(&fixture, session, live)?;
        replay.eager(both)?;
        against_reference(&fixture, session, live, &label)?;
        replay.relaunch(both)?;
        replay.graph(&graph)?;
        replay.pdl(
            || fixture.producer(session, false),
            || fixture.batched(session, true),
            true,
        )?;
        let mut receipt = BTreeMap::new();
        replay.record(&mut receipt);
        let mut mutations = Vec::new();
        if live == (rows / 2).max(1) {
            replay.prepare(Phase::Pdl)?;
            fixture.batched(session, true)?;
            session.stream.synchronize().map_err(err)?;
            let mut changed = Vec::new();
            for (output, handle) in [("x1", X1), ("collapsed", COLLAPSED), ("normed", NORMED)] {
                let actual = poison::read(session, fixture.buffer(handle))?;
                let eager = replay
                    .expected(output)
                    .ok_or_else(|| format!("{label}: no eager {output}"))?;
                if actual != eager {
                    changed.push(output);
                }
            }
            if changed != ["x1", "collapsed", "normed"] {
                return Err(format!(
                    "{label}: without the producer only {changed:?} differ from the eager bits"
                ));
            }
            mutations.push(json!({"mutation": "no producer", "outputs": changed}));
        }
        evidence.push(json!({
            "live": live,
            "phases_run": REPLAY_PHASES,
            "replay": receipt,
            "mutations_caught": mutations,
        }));
    }
    Ok(json!({"rows": rows, "live_counts": evidence}))
}

/// The live counts of a replay receipt row whose phases all ran and whose
/// graph replay matched the eager bits.
fn covered_lives(bucket: &Value) -> std::collections::BTreeSet<u64> {
    bucket["live_counts"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|entry| {
            entry["phases_run"] == json!(REPLAY_PHASES)
                && entry["replay"]["graph_bitwise"] == json!(true)
                && entry["replay"]["deterministic_bitwise"] == json!(true)
        })
        .filter_map(|entry| entry["live"].as_u64())
        .collect()
}

/// The mutants at `rows` rows, each against the served entry's eager bits.
fn mutant_bucket(session: &Session, rows: usize, inputs: &Inputs) -> LabResult<Value> {
    let fixture = Fixture::new(session, rows, inputs)?;
    let label = format!("{rows} rows");
    fixture.producer(session, false)?;
    fixture.batched(session, false)?;
    session.stream.synchronize().map_err(err)?;
    let eager = fixture.outputs(session, OUTPUTS)?;
    let eager_partials = fixture.partials(session, PARTIALS)?;
    let outputs = [COLLAPSED, POST, COMB, NORMED, PARTIALS].map(|h| fixture.buffer(h));
    // The outputs and chain sums a mutant launch changes.
    let changed = |suffix: &str| -> LabResult<Vec<&'static str>> {
        poison::poison(session, &outputs)?;
        fixture.mutant(session, suffix, false)?;
        session.stream.synchronize().map_err(err)?;
        let mut names = differing(&fixture.outputs(session, OUTPUTS)?, &eager);
        if fixture.partials(session, PARTIALS)? != eager_partials {
            names.push("chain sums");
        }
        Ok(names)
    };

    // The butterfly's levels reversed: the chain sums differ.
    let levels = changed("_levels_reversed")?;
    if !levels.contains(&"chain sums") {
        return Err(format!(
            "{label}: the reversed-levels mutant changed {levels:?}, not the chain sums"
        ));
    }
    // The ordered sum's ties away from zero: the normed rows alone differ.
    let ties = changed("_ties_away")?;
    if ties != ["normed"] {
        return Err(format!(
            "{label}: the ties-away mutant changed {ties:?}, not the normed rows alone"
        ));
    }

    // The dependency wait behind the live-count producer, which zeroes the
    // live count and the streams, releases its dependent, and restores
    // both after its tail.
    let producer = LiveProducer::new(session)?;
    let streams = poison::read(session, fixture.buffer(X1))?;
    poison::write(session, fixture.buffer(X1_IMAGE), 0, &streams)?;
    let rewrites = Rewrites::new(session, &[(fixture.buffer(X1_IMAGE), fixture.buffer(X1))])?;
    let behind = |late: bool| -> LabResult<Vec<&'static str>> {
        session.stream.synchronize().map_err(err)?;
        poison::poison(session, &outputs)?;
        producer.launch_rewriting(session, fixture.buffer(LIVE), &rewrites)?;
        if late {
            fixture.mutant(session, "_late_wait", true)?;
        } else {
            fixture.batched(session, true)?;
        }
        session.stream.synchronize().map_err(err)?;
        Ok(differing(&fixture.outputs(session, OUTPUTS)?, &eager))
    };
    let served = behind(false)?;
    if !served.is_empty() {
        return Err(format!(
            "{label}: the served entry behind the live-count producer changed {served:?}"
        ));
    }
    let late = behind(true)?;
    let outcome = if late.is_empty() {
        eprintln!(
            "{label}, dependency wait after the chain sums' reads: inconclusive: the mutant \
             reproduced the eager bits"
        );
        "inconclusive"
    } else {
        "caught"
    };
    Ok(json!({
        "rows": rows,
        "mutants": [
            {"mutation": "chain-sum butterfly levels reversed", "outcome": "caught",
             "outputs": levels},
            {"mutation": "ordered sum ties away from zero", "outcome": "caught",
             "outputs": ties},
            {"mutation": "dependency wait after the chain sums' reads", "outcome": outcome,
             "outputs": late},
        ],
    }))
}

/// Every bucket of 1 to 32 rows and the verify bucket.
fn buckets() -> Vec<usize> {
    (1..=32).chain([6 * 32]).collect()
}

/// The decode buckets the timing runs at.
const TIMED_BUCKETS: [usize; 6] = [1, 2, 4, 8, 16, 32];

#[test]
fn sinkhorn_hc_mix_norm_batched_cpu_contract() {
    let entry = mix_norm_batched::served();
    assert_eq!(
        kernel_lab::cubin_entries(&batched_cubin("")).unwrap(),
        [entry.symbol.as_str()]
    );
    kernel_lab::cubin(&batched_cubin("")).unwrap();
    for (suffix, _) in mix_norm_batched::MUTANTS {
        assert_eq!(
            kernel_lab::cubin_entries(&batched_cubin(suffix)).unwrap(),
            [mutant(suffix).symbol.as_str()]
        );
    }
    assert_eq!(
        kernel_lab::cubin_entries(&exact_cubin()).unwrap(),
        [mix_norm_exact::served().symbol.as_str()]
    );
    for rows in buckets() {
        let plan = entry.plan(rows as u32, rows as u32).unwrap();
        assert_eq!(
            plan,
            mix_norm_exact::served()
                .plan(rows as u32, rows as u32)
                .unwrap()
        );
        assert_eq!(plan.grid[0] as usize, GROUPS);
    }
    assert_eq!(buckets().len(), 33);
    assert_eq!(replay_lives(32), (0..=32).collect::<Vec<u32>>());
    assert_eq!(replay_lives(192), poison::live_counts(192));
    let mut full = json!({"rows": 32, "live_counts": []});
    for live in 0..=32 {
        full["live_counts"].as_array_mut().unwrap().push(json!({
            "live": live,
            "phases_run": REPLAY_PHASES,
            "replay": {"deterministic_bitwise": true, "graph_bitwise": true},
        }));
    }
    assert_eq!(covered_lives(&full).len(), 33);
    full["live_counts"][5]["replay"]["graph_bitwise"] = json!(false);
    assert!(!covered_lives(&full).contains(&5));
    let unwritten = [
        vec![0xffff_u16; H],
        vec![0xffff; N],
        vec![0xffff; CELLS],
        vec![0xffff; H],
    ];
    let written = [
        vec![0x3f80_u16; H],
        vec![0x3f00; N],
        vec![0x3e80; CELLS],
        vec![0x3f80; H],
    ];
    assert_eq!(differing(&unwritten, &written), OUTPUT_NAMES);
    assert_eq!(Inputs::wide(1, 1).x0.len(), FLAT);
    assert_eq!(Inputs::order_sensitive(1, 3).y0[0] >> 7, 140);
    assert_eq!(Inputs::extreme(1, 1).base[2 * N], 1e8);
    let special = Inputs::non_finite(6, 2);
    let row = |r: usize| &special.x0[r * FLAT..(r + 1) * FLAT];
    assert_eq!(row(0), &Inputs::new(6, 2).x0[..FLAT]);
    assert_eq!(row(1).iter().filter(|&&v| v & 0x7fff > 0x7f80).count(), 4);
    assert_eq!(row(2).iter().filter(|&&v| v & 0x7fff == 0x7f80).count(), 2);
    // Binades 100 to 127, beside bf16_values' zeros and one subnormal.
    let binade = |v: u16| (v >> 7) & 0xff;
    assert!(row(3).iter().all(|&v| binade(v) == 0 || binade(v) >= 227));
    assert!(row(3).iter().filter(|&&v| binade(v) >= 227).count() > FLAT - 100);
    assert!(special.y0[4 * H..5 * H].contains(&0x7fc5));
    assert_eq!(row(5).iter().filter(|&&v| v & 0x7fff == 0x7f80).count(), 2);
    let counts = non_finite_counts(
        &[
            vec![0x7fc0, 0x7f80, 1],
            vec![0xff80],
            vec![0x8001],
            vec![0xffff],
        ],
        &[0x7f80_0000, 0xffc0_0000, 0x3f80_0000],
    );
    assert_eq!(counts["collapsed"], json!({"nan": 1, "inf": 1}));
    assert_eq!(counts["post"], json!({"nan": 0, "inf": 1}));
    assert_eq!(counts["comb"], json!({"nan": 0, "inf": 0}));
    assert_eq!(counts["normed"], json!({"nan": 1, "inf": 0}));
    assert_eq!(counts["chain sums"], json!({"nan": 1, "inf": 1}));
}

#[test]
fn sinkhorn_hc_mix_norm_batched_matches_the_exact_sum_entry_bitwise() {
    kernel_lab::gpu_test(3 << 30, |session| {
        let mut rows = buckets()
            .into_iter()
            .map(|rows| bitwise_bucket(session, rows, &Inputs::new(rows, 0x5eed + rows as u64)))
            .collect::<LabResult<Vec<_>>>()?;
        for &rows_ in &[1, 32, 6 * 32] {
            let mut wide =
                bitwise_bucket(session, rows_, &Inputs::wide(rows_, 0x31de + rows_ as u64))?;
            wide["wide_binades"] = json!([-30, 30]);
            rows.push(wide);
        }
        let mut extreme = bitwise_bucket(session, 32, &Inputs::extreme(32, 0xe0))?;
        extreme["extreme_common_offset"] = json!(1e8);
        rows.push(extreme);
        for &rows_ in &[32, 6 * 32] {
            let mut ordered =
                bitwise_bucket(session, rows_, &Inputs::order_sensitive(rows_, 0x0d0e))?;
            ordered["order_sensitive"] = json!(true);
            rows.push(ordered);
        }
        rows.extend(non_finite_buckets(session)?);
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("sinkhorn_hc_mix_norm_batched_bitwise"));
        receipt.insert("buckets".into(), json!(rows));
        session.receipt("sinkhorn_hc_mix_norm_batched_bitwise", receipt)?;
        Ok(())
    })
    .expect("batched Sinkhorn mix and norm bitwise comparison");
}

/// The non-finite buckets (32 and 192 rows of `Inputs::non_finite`):
/// outputs and chain sums bitwise the exact-sum entry's, with every case of
/// `NON_FINITE_REQUIRED` present in its outputs.
fn non_finite_buckets(session: &Session) -> LabResult<Vec<Value>> {
    let mut rows = Vec::new();
    for rows_ in [32, 6 * 32] {
        let mut bucket = bitwise_bucket(session, rows_, &Inputs::non_finite(rows_, 0x0a0f))?;
        for (output, kind) in NON_FINITE_REQUIRED {
            if bucket["non_finite"][output][kind].as_u64().unwrap_or(0) == 0 {
                return Err(format!(
                    "non-finite inputs at {rows_} rows: no {kind} in the reference {output} ({})",
                    bucket["non_finite"]
                ));
            }
        }
        bucket["non_finite_inputs"] = json!(true);
        rows.push(bucket);
    }
    Ok(rows)
}

/// The non-finite buckets alone (the supplementary job's NaN, infinity and
/// overflow evidence; the full bitwise test runs them too).
#[test]
fn sinkhorn_hc_mix_norm_batched_non_finite_bitwise() {
    kernel_lab::gpu_test(3 << 30, |session| {
        let rows = non_finite_buckets(session)?;
        let mut receipt = BTreeMap::new();
        receipt.insert(
            "op".into(),
            json!("sinkhorn_hc_mix_norm_batched_non_finite_bitwise"),
        );
        receipt.insert("buckets".into(), json!(rows));
        session.receipt("sinkhorn_hc_mix_norm_batched_non_finite_bitwise", receipt)?;
        Ok(())
    })
    .expect("batched Sinkhorn mix and norm non-finite bitwise comparison");
}

#[test]
fn sinkhorn_hc_mix_norm_batched_replays_on_poisoned_buffers() {
    kernel_lab::gpu_test(3 << 30, |session| {
        let mut rows = [1, 2, 3, 4, 8, 16, 32, 6 * 32]
            .into_iter()
            .map(|rows| replay_bucket(session, rows, &Inputs::new(rows, 0xfeed + rows as u64)))
            .collect::<LabResult<Vec<_>>>()?;
        let mut extreme = replay_bucket(session, 32, &Inputs::extreme(32, 0xfe))?;
        extreme["extreme_common_offset"] = json!(1e8);
        rows.push(extreme);
        let mut wide = replay_bucket(session, 32, &Inputs::wide(32, 0xf1))?;
        wide["wide_binades"] = json!([-30, 30]);
        rows.push(wide);
        // Every capacity-32 bucket replayed its graph, relaunches and PDL at
        // every live count 0 to 32, as its receipt rows show.
        let expected = (0..=FULL_REPLAY_ROWS as u64).collect::<std::collections::BTreeSet<_>>();
        let full_rows: Vec<&Value> = rows
            .iter()
            .filter(|row| row["rows"] == json!(FULL_REPLAY_ROWS))
            .collect();
        if full_rows.len() != 3 {
            return Err(format!(
                "{} capacity-32 replay rows, expected 3",
                full_rows.len()
            ));
        }
        for full in &full_rows {
            let covered = covered_lives(full);
            if covered != expected {
                return Err(format!(
                    "capacity-32 replays covered live counts {covered:?}, expected 0 to 32"
                ));
            }
        }
        let successful_pdl = full_rows[0]["live_counts"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|entry| entry["replay"]["pdl_bitwise"] == json!(true))
            .count();
        let mut receipt = BTreeMap::new();
        receipt.insert(
            "capacity_32_covered_live_counts".into(),
            json!(expected.iter().collect::<Vec<_>>()),
        );
        receipt.insert(
            "capacity_32_successful_pdl_replays".into(),
            json!(successful_pdl),
        );
        receipt.insert("op".into(), json!("sinkhorn_hc_mix_norm_batched_poison"));
        receipt.insert("buckets".into(), json!(rows));
        session.receipt("sinkhorn_hc_mix_norm_batched_poison", receipt)?;
        Ok(())
    })
    .expect("batched Sinkhorn mix and norm poisoned replays");
}

#[test]
fn sinkhorn_hc_mix_norm_batched_mutants_are_caught() {
    kernel_lab::gpu_test(3 << 30, |session| {
        let rows = [32, 6 * 32]
            .into_iter()
            .map(|rows| mutant_bucket(session, rows, &Inputs::new(rows, 0x7a7e + rows as u64)))
            .collect::<LabResult<Vec<_>>>()?;
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("sinkhorn_hc_mix_norm_batched_mutants"));
        receipt.insert("buckets".into(), json!(rows));
        session.receipt("sinkhorn_hc_mix_norm_batched_mutants", receipt)?;
        Ok(())
    })
    .expect("batched Sinkhorn mix and norm mutants");
}

/// Graph replays of eight sublayer boundaries: per boundary the expand,
/// then the exact-sum entry (arm 0) or the batched entry (arm 1), each
/// launch PDL after the previous as the bundle runs them.
#[test]
fn sinkhorn_hc_mix_norm_batched_timing() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("sinkhorn_hc_mix_norm_batched timing: set CHOPINFER_KERNEL_BENCH=1")
            .unwrap();
        return;
    }
    kernel_lab::gpu_test(3 << 30, |session| {
        let mut rows = Vec::new();
        for bucket in TIMED_BUCKETS {
            let fixture = Fixture::new(
                session,
                bucket,
                &Inputs::new(bucket, 0x7193 + bucket as u64),
            )?;
            let graphs = [false, true].map(|batched| {
                session.capture(|_| {
                    for boundary in 0..BOUNDARIES {
                        fixture.producer(session, boundary > 0)?;
                        if batched {
                            fixture.batched(session, true)?;
                        } else {
                            fixture.reference(session, true)?;
                        }
                    }
                    Ok(())
                })
            });
            let [exact_graph, batched_graph] = graphs;
            let graphs = [exact_graph?, batched_graph?];
            let paired = kernel_lab::time_ab(
                session,
                3,
                15,
                [kernel_lab::eviction::EvictionMode::Write; 2],
                |arm, stream| graphs[arm].launch(stream).map_err(err),
            )?;
            let per = |ms: f32| f64::from(ms) * 1e3 / BOUNDARIES as f64;
            let row = json!({
                "rows": bucket,
                "boundaries": BOUNDARIES,
                "exact_sum_us_per_boundary": per(paired.arms[0].median_ms),
                "batched_us_per_boundary": per(paired.arms[1].median_ms),
                "batched_over_exact": paired.paired_ratio,
            });
            eprintln!("sinkhorn_hc_mix_norm_batched timing: {row}");
            rows.push(row);
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("sinkhorn_hc_mix_norm_batched_timing"));
        receipt.insert("buckets".into(), json!(rows));
        session.receipt("sinkhorn_hc_mix_norm_batched_timing", receipt)?;
        Ok(())
    })
    .expect("batched Sinkhorn mix and norm timing");
}
