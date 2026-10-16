//! The split Sinkhorn mix-collapse fused with the input RMSNorm
//! (`kernels_sm120::sinkhorn_hc::mix_norm`) against the two launches it
//! replaces: the split mix-collapse (`sinkhorn_hc::split`) and the row
//! engine's HF-ordered RMSNorm of the collapsed row (the comm epilogue lab
//! cubin's `reference_norm`), at the served 4 x 4096 mixer, on the GPU. Run
//! with `CHOPINFER_KERNEL_GPU=<ordinal> cargo test -p kernel-lab --release
//! --test sinkhorn_hc_mix_norm -- --test-threads=1`; without a GPU only the
//! CPU contract test runs.
//!
//! At decode (1, 2, 3, 4, 8, 16, 32) and verify (6 x 32) buckets, after an
//! expand writes the streams as a sublayer boundary does:
//!
//! - collapsed, post, comb and the normed row bitwise equal to the two
//!   launches', written into freshly poisoned buffers that alone fail the
//!   comparison; the chain sums bitwise equal to the split entry's; inputs
//!   of ordinary, wide (squares over 2^-60 to 2^60) and extreme common
//!   offset magnitudes;
//! - determinism: relaunches rewrite every output and chain sum with the
//!   same bits and leave every ticket at zero;
//! - poisoned replays (eager, relaunch, graph, PDL) at every live count
//!   `poison::live_counts` lists, after the producer whose streams are
//!   poisoned and checked too, the chain sums poisoned as a written
//!   workspace and the tickets checked as zeroed state; each live count's
//!   eager outputs compared with the two launches at that live count; with
//!   the producer left out, the streams, the collapsed and the normed rows
//!   must differ from the eager bits;
//! - mutants, each caught only when an output differs from known good bits
//!   (launch, synchronization and readback errors propagate): the norm's
//!   squares added in reverse column order (`_reverse_sum`; the normed rows
//!   alone must differ, on rows whose column 0 dominates so that the order
//!   moves the sum), and the dependency wait after the chain sums'
//!   reads (`_late_wait`) behind the lab's PDL live-count producer, which
//!   holds the live count and the streams at zero through its tail: the
//!   served entry must reproduce the eager bits, the mutant is caught when
//!   its outputs differ, and reported inconclusive when PDL did not
//!   overlap the launches.
//!
//! With `CHOPINFER_KERNEL_BENCH=1`, `sinkhorn_hc_mix_norm_timing` times
//! graphs of eight sublayer boundaries (expand, then the split mix and the
//! norm, or the fused entry; PDL as served) at the decode buckets.

use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{Argument, ArgumentList, DType, LaunchPlan};
use kernel_lab::launch::{self, Bindings};
use kernel_lab::poison::{self, LiveProducer, Phase, Replay, Rewrites, WriteMap};
use kernel_lab::{LabResult, Session};
use kernels_sm120::comm::sinkhorn_epilogue;
use kernels_sm120::row::rmsnorm::RmsNormBuffers;
use kernels_sm120::sinkhorn_hc::{self, Op, SERVED, mix_norm, split};
use runtime_cuda::{Function, Module};
use serde_json::{Value, json};
use std::collections::BTreeMap;

const N: usize = 4;
const H: usize = 4096;
const FLAT: usize = N * H;
const MIXES: usize = (2 + N) * N;
const CELLS: usize = N * N;
const CHAINS: usize = FLAT / 128;
const BUCKETS: [usize; 8] = [1, 2, 3, 4, 8, 16, 32, 6 * 32];
/// The input norm's eps.
const EPS: f32 = 1e-5;

// Buffer handles: X1 = expand(X0, Y0, POST0, COMB0); the reference split
// mix and norm, and the fused entry, over X1 into their own outputs.
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
/// The streams' image the live-count producer restores.
const X1_IMAGE: u64 = 22;

/// The fused entry's outputs, in comparison order.
const OUTPUTS: [u64; 4] = [COLLAPSED, POST, COMB, NORMED];
const REF_OUTPUTS: [u64; 4] = [REF_COLLAPSED, REF_POST, REF_COMB, REF_NORMED];
const OUTPUT_NAMES: [&str; 4] = ["collapsed", "post", "comb", "normed"];
const PER_ROW: [usize; 4] = [H, N, CELLS, H];

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn tag() -> &'static str {
    sinkhorn_hc::SERVED_PROFILE.tag()
}

fn served_cubin() -> String {
    format!("sinkhorn_hc_{}_sm120", tag())
}

fn split_cubin() -> String {
    format!("sinkhorn_hc_{}_split_sm120", tag())
}

/// The comm epilogue's lab cubin, which holds the reference norm.
fn norm_cubin() -> String {
    format!("sinkhorn_epilogue_{}_sm120", tag())
}

fn fused_cubin(suffix: &str) -> String {
    format!("sinkhorn_hc_{}_mix_norm{suffix}_sm120", tag())
}

fn mutant(suffix: &str) -> mix_norm::Entry {
    mix_norm::entry_with_suffix(
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
    /// 2^60, so the norm's ordered sum rounds at every magnitude.
    fn wide(rows: usize, seed: u64) -> Self {
        Self {
            x0: bf16_values(rows * FLAT, seed + 1, -30, 30),
            y0: bf16_values(rows * H, seed + 2, -30, 30),
            ..Self::new(rows, seed)
        }
    }

    /// Positive streams and sublayer outputs whose column 0 is 2^13 times
    /// the other columns (random mantissas, binade 13 against binade 0), so
    /// every row's collapsed column 0 dominates the rest by about 2^13. In
    /// column order the norm's sum starts at column 0's square, and each
    /// later square, below a quarter of the running sum's ulp, rounds
    /// away; in reverse order the 4095 small squares accumulate first and
    /// add about 2^-14 of the total. The summation order then changes every
    /// row's norm scale by about 2^-15, which moves some of its BF16
    /// roundings.
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
    split: Kernel,
    norm: Kernel,
    fused: Kernel,
    /// The mutants by suffix.
    mutants: BTreeMap<&'static str, Kernel>,
}

impl Fixture {
    fn new(session: &Session, rows: usize, inputs: &Inputs) -> LabResult<Self> {
        let served = session.module(&served_cubin())?;
        let split_module = session.module(&split_cubin())?;
        let norm_module = session.module(&norm_cubin())?;
        let fused_module = session.module(&fused_cubin(""))?;
        let expand = expand_entry();
        let split_entry = split::served();
        let fused_entry = mix_norm::served();
        let (norm, norm_symbol) = sinkhorn_epilogue::reference_norm();
        let mut modules = Vec::new();
        let mut mutants = BTreeMap::new();
        for (suffix, _) in mix_norm::MUTANTS {
            let module = session.module(&fused_cubin(suffix))?;
            mutants.insert(
                suffix,
                Kernel {
                    plan: fused_entry.plan(rows as u32, 0).map_err(err)?,
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
            split: Kernel {
                plan: split_entry.plan(rows as u32, 0).map_err(err)?,
                function: split_module.function(&split_entry.symbol).map_err(err)?,
            },
            // The row engine's grid at the bucket, as the bundle plans it.
            norm: Kernel {
                plan: LaunchPlan {
                    grid: [rows as u32, 1, 1],
                    ..norm.plan(0).map_err(err)?
                },
                function: norm_module.function(&norm_symbol).map_err(err)?,
            },
            fused: Kernel {
                plan: fused_entry.plan(rows as u32, 0).map_err(err)?,
                function: fused_module.function(&fused_entry.symbol).map_err(err)?,
            },
            mutants,
            _modules: {
                modules.extend([served, split_module, norm_module, fused_module]);
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

    /// The two launches the fused entry replaces: the split mix-collapse,
    /// then the input norm of its collapsed rows.
    fn reference(&self, session: &Session, pdl: bool) -> LabResult<()> {
        let args = split::served()
            .pack_arguments(Self::handles(&[
                X1,
                FN,
                BASE,
                SCALE,
                REF_COLLAPSED,
                REF_POST,
                REF_COMB,
                REF_PARTIALS,
                REF_TICKETS,
                LIVE,
            ]))
            .map_err(err)?;
        self.launch(session, &self.split, args, pdl)?;
        let (norm, _) = sinkhorn_epilogue::reference_norm();
        let args = norm
            .arguments(RmsNormBuffers {
                x: REF_COLLAPSED,
                weight: WEIGHT,
                residual: None,
                y: REF_NORMED,
                live_rows: LIVE,
            })
            .map_err(err)?;
        self.launch(session, &self.norm, args, pdl)
    }

    fn fused_args(&self) -> LabResult<ArgumentList> {
        let mut args = Self::handles(&[
            X1, FN, BASE, SCALE, WEIGHT, COLLAPSED, POST, COMB, NORMED, PARTIALS, TICKETS, LIVE,
        ]);
        args.push(Argument::U32(EPS.to_bits()));
        mix_norm::served().pack_arguments(args).map_err(err)
    }

    fn fused(&self, session: &Session, pdl: bool) -> LabResult<()> {
        self.launch(session, &self.fused, self.fused_args()?, pdl)
    }

    fn mutant(&self, session: &Session, suffix: &str, pdl: bool) -> LabResult<()> {
        let kernel = self
            .mutants
            .get(suffix)
            .ok_or_else(|| format!("no mutant {suffix}"))?;
        self.launch(session, kernel, self.fused_args()?, pdl)
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
    fixture.producer(session, false)?;
    fixture.reference(session, false)?;
    session.stream.synchronize().map_err(err)?;
    let reference = fixture.outputs(session, REF_OUTPUTS)?;
    let reference_partials = fixture.partials(session, REF_PARTIALS)?;
    let zeroed = vec![0_u8; rows * 4];
    exact(
        "reference tickets",
        &fixture.tickets(session, REF_TICKETS)?,
        &zeroed,
    )?;

    // The fused entry into poisoned outputs, which alone fail the
    // comparison.
    let outputs = [COLLAPSED, POST, COMB, NORMED, PARTIALS].map(|h| fixture.buffer(h));
    poison::poison(session, &outputs)?;
    let unwritten = fixture.outputs(session, OUTPUTS)?;
    if differing(&unwritten, &reference).len() != OUTPUTS.len() {
        return Err("unwritten fused outputs matched the reference".into());
    }
    fixture.fused(session, false)?;
    session.stream.synchronize().map_err(err)?;
    let fused = fixture.outputs(session, OUTPUTS)?;
    same_outputs("fused against the split mix and norm", &fused, &reference)?;
    exact(
        "fused chain sums against the split entry's",
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
        fixture.fused(session, relaunch % 2 == 1)?;
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
        "grid": fixture.fused.plan.grid,
        "bitwise_split_and_norm": true,
        "chain_sums_bitwise_split": true,
        "relaunches_bitwise": 4,
    }))
}

/// The fused entry's live rows against the two launches over the same
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
        &format!("{label}: fused against the split mix and norm"),
        &take(OUTPUTS)?,
        &take(REF_OUTPUTS)?,
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

/// The fused entry on poisoned buffers at every live count of `rows`,
/// after its producer (an expand) whose streams are poisoned and checked
/// too. Each live count's eager outputs are also compared with the two
/// launches, and at one partial live count the producer left out must
/// leave the streams poisoned and change the collapsed and normed rows.
fn replay_bucket(session: &Session, rows: usize, inputs: &Inputs) -> LabResult<Value> {
    let fixture = Fixture::new(session, rows, inputs)?;
    let graph = session.capture(|_| {
        fixture.producer(session, false)?;
        fixture.fused(session, true)
    })?;
    let mut evidence = Vec::new();
    for live in poison::live_counts(rows as u32) {
        let live = live as usize;
        let label = format!("{rows} rows, live {live}");
        fixture.set_live(session, live)?;
        let both = || -> LabResult<()> {
            fixture.producer(session, false)?;
            fixture.fused(session, false)
        };
        let mut replay = replay_of(&fixture, session, live)?;
        replay.eager(both)?;
        against_reference(&fixture, session, live, &label)?;
        replay.relaunch(both)?;
        replay.graph(&graph)?;
        replay.pdl(
            || fixture.producer(session, false),
            || fixture.fused(session, true),
            true,
        )?;
        let mut receipt = BTreeMap::new();
        replay.record(&mut receipt);
        let mut mutations = Vec::new();
        if live == (rows / 2).max(1) {
            replay.prepare(Phase::Pdl)?;
            fixture.fused(session, true)?;
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
        evidence.push(json!({"live": live, "replay": receipt, "mutations_caught": mutations}));
    }
    Ok(json!({"rows": rows, "live_counts": evidence}))
}

/// The mutants at `rows` rows, each against the served entry's eager bits.
fn mutant_bucket(session: &Session, rows: usize, inputs: &Inputs) -> LabResult<Value> {
    let fixture = Fixture::new(session, rows, inputs)?;
    let label = format!("{rows} rows");
    fixture.producer(session, false)?;
    fixture.fused(session, false)?;
    session.stream.synchronize().map_err(err)?;
    let eager = fixture.outputs(session, OUTPUTS)?;
    let outputs = [COLLAPSED, POST, COMB, NORMED, PARTIALS].map(|h| fixture.buffer(h));

    // The norm's squares in reverse column order: the normed rows alone
    // differ.
    poison::poison(session, &outputs)?;
    fixture.mutant(session, "_reverse_sum", false)?;
    session.stream.synchronize().map_err(err)?;
    let reversed = differing(&fixture.outputs(session, OUTPUTS)?, &eager);
    if reversed != ["normed"] {
        return Err(format!(
            "{label}: the reverse-sum mutant changed {reversed:?}, not the normed rows alone"
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
            fixture.fused(session, true)?;
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
             reproduced the eager bits (PDL did not overlap the launches)"
        );
        "inconclusive"
    } else {
        "caught"
    };
    Ok(json!({
        "rows": rows,
        "mutants": [
            {"mutation": "norm squares in reverse column order", "outcome": "caught",
             "outputs": reversed},
            {"mutation": "dependency wait after the chain sums' reads", "outcome": outcome,
             "outputs": late},
        ],
    }))
}

#[test]
fn sinkhorn_hc_mix_norm_cpu_contract() {
    let entry = mix_norm::served();
    let entries = kernel_lab::cubin_entries(&fused_cubin("")).unwrap();
    assert_eq!(entries, [entry.symbol.as_str()]);
    kernel_lab::cubin(&fused_cubin("")).unwrap();
    for (suffix, _) in mix_norm::MUTANTS {
        assert_eq!(
            kernel_lab::cubin_entries(&fused_cubin(suffix)).unwrap(),
            [mutant(suffix).symbol.as_str()]
        );
    }
    assert!(
        kernel_lab::cubin_entries(&norm_cubin())
            .unwrap()
            .contains(&sinkhorn_epilogue::reference_norm().1.as_str())
    );
    for rows in BUCKETS {
        let plan = entry.plan(rows as u32, rows as u32).unwrap();
        assert_eq!(
            plan,
            split::served().plan(rows as u32, rows as u32).unwrap()
        );
    }
    // The reference norm is the fused entry's: HF order at the mixer's
    // width, eps an argument.
    assert_eq!(
        *sinkhorn_epilogue::reference_norm().0.requirement(),
        mix_norm::norm_requirement(SERVED.mix, EPS)
    );
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
    same_outputs("written", &written, &written).unwrap();
    let inputs = Inputs::wide(1, 1);
    assert_eq!(inputs.x0.len(), FLAT);
    // The order-sensitive rows: column 0 in binade 13 of every stream and
    // the sublayer output, the rest in binade 0, all positive. On such a
    // collapsed row the norm's squares summed in column order and in
    // reverse differ, by about 2^-14 of the total.
    let ordered = Inputs::order_sensitive(2, 3);
    for (values, width) in [(&ordered.x0, H), (&ordered.y0, H)] {
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(v >> 15, 0);
            assert_eq!((v >> 7) & 0xff, if i % width == 0 { 140 } else { 127 });
        }
    }
    let squares = ordered.y0[..H]
        .iter()
        .map(|&v| {
            let v = bf16_to_f32(v);
            v * v
        })
        .collect::<Vec<f32>>();
    let forward = squares.iter().fold(0.0_f32, |s, &v| s + v);
    let reverse = squares.iter().rev().fold(0.0_f32, |s, &v| s + v);
    assert!(reverse / forward - 1.0 > 2e-5, "{forward} {reverse}");
    // And the normalized row's BF16 bits move with it.
    let normed = |total: f32| {
        let inv = 1.0 / (total / H as f32 + EPS).sqrt();
        ordered.y0[..H]
            .iter()
            .zip(&ordered.norm_weight)
            .map(|(&v, &w)| {
                let n = bf16_to_f32(f32_to_bf16(bf16_to_f32(v) * inv));
                f32_to_bf16(n * bf16_to_f32(w))
            })
            .collect::<Vec<u16>>()
    };
    let moved = normed(forward)
        .iter()
        .zip(normed(reverse))
        .filter(|(a, b)| **a != *b)
        .count();
    assert!(moved > 0, "the order moved no normalized value");
    assert_eq!(inputs.norm_weight.len(), H);
}

#[test]
fn sinkhorn_hc_mix_norm_matches_the_split_mix_and_norm_bitwise() {
    kernel_lab::gpu_test(3 << 30, |session| {
        let mut rows = BUCKETS
            .iter()
            .map(|&rows| bitwise_bucket(session, rows, &Inputs::new(rows, 0x5eed + rows as u64)))
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
        let mut ordered = bitwise_bucket(session, 32, &Inputs::order_sensitive(32, 0x0d0e))?;
        ordered["order_sensitive"] = json!(true);
        rows.push(ordered);
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("sinkhorn_hc_mix_norm_bitwise"));
        receipt.insert("buckets".into(), json!(rows));
        session.receipt("sinkhorn_hc_mix_norm_bitwise", receipt)?;
        Ok(())
    })
    .expect("fused Sinkhorn mix and norm bitwise comparison");
}

#[test]
fn sinkhorn_hc_mix_norm_replays_on_poisoned_buffers() {
    kernel_lab::gpu_test(3 << 30, |session| {
        let mut rows = BUCKETS
            .iter()
            .map(|&rows| replay_bucket(session, rows, &Inputs::new(rows, 0xfeed + rows as u64)))
            .collect::<LabResult<Vec<_>>>()?;
        let mut extreme = replay_bucket(session, 32, &Inputs::extreme(32, 0xfe))?;
        extreme["extreme_common_offset"] = json!(1e8);
        rows.push(extreme);
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("sinkhorn_hc_mix_norm_poison"));
        receipt.insert("buckets".into(), json!(rows));
        session.receipt("sinkhorn_hc_mix_norm_poison", receipt)?;
        Ok(())
    })
    .expect("fused Sinkhorn mix and norm poisoned replays");
}

#[test]
fn sinkhorn_hc_mix_norm_mutants_are_caught() {
    kernel_lab::gpu_test(3 << 30, |session| {
        let rows = [32, 6 * 32]
            .into_iter()
            .map(|rows| {
                let inputs = Inputs::order_sensitive(rows, 0x7a7e + rows as u64);
                mutant_bucket(session, rows, &inputs)
            })
            .collect::<LabResult<Vec<_>>>()?;
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("sinkhorn_hc_mix_norm_mutants"));
        receipt.insert("buckets".into(), json!(rows));
        session.receipt("sinkhorn_hc_mix_norm_mutants", receipt)?;
        Ok(())
    })
    .expect("fused Sinkhorn mix and norm mutants");
}

/// Graph replays of eight sublayer boundaries: per boundary the expand,
/// then the split mix and the norm (arm 0) or the fused entry (arm 1), each
/// launch PDL after the previous as the bundle runs them.
#[test]
fn sinkhorn_hc_mix_norm_timing() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("sinkhorn_hc_mix_norm timing: set CHOPINFER_KERNEL_BENCH=1").unwrap();
        return;
    }
    const BOUNDARIES: usize = 8;
    kernel_lab::gpu_test(3 << 30, |session| {
        let mut rows = Vec::new();
        for bucket in [1, 2, 4, 8, 16, 32] {
            let fixture = Fixture::new(
                session,
                bucket,
                &Inputs::new(bucket, 0x7193 + bucket as u64),
            )?;
            let graphs = [false, true].map(|fused| {
                session.capture(|_| {
                    for boundary in 0..BOUNDARIES {
                        fixture.producer(session, boundary > 0)?;
                        if fused {
                            fixture.fused(session, true)?;
                        } else {
                            fixture.reference(session, true)?;
                        }
                    }
                    Ok(())
                })
            });
            let [two, one] = graphs;
            let graphs = [two?, one?];
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
                "split_and_norm_us_per_boundary": per(paired.arms[0].median_ms),
                "fused_us_per_boundary": per(paired.arms[1].median_ms),
                "fused_over_split_and_norm": paired.paired_ratio,
            });
            eprintln!("sinkhorn_hc_mix_norm timing: {row}");
            rows.push(row);
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("sinkhorn_hc_mix_norm_timing"));
        receipt.insert("buckets".into(), json!(rows));
        session.receipt("sinkhorn_hc_mix_norm_timing", receipt)?;
        Ok(())
    })
    .expect("fused Sinkhorn mix and norm timing");
}
