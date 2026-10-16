//! The tiled Sinkhorn mix-collapse (`kernels_sm120::sinkhorn_hc::tiled`)
//! against the prefill entries it replaces, the one-CTA entry and the split
//! (`sinkhorn_hc::split`), at the served 4 x 4096 mixer, on the GPU. Run
//! with `CHOPINFER_KERNEL_GPU=<ordinal> cargo test -p kernel-lab --release
//! --test sinkhorn_hc_tiled -- --test-threads=1`; without a GPU only the CPU
//! contract test runs.
//!
//! At every tile of `tiled::ROWS` rows per CTA, after an expand writes the
//! streams as a sublayer boundary does:
//!
//! - collapsed, post and comb bitwise equal to the one-CTA served cubin's
//!   and to the split's, written into freshly poisoned buffers that alone
//!   fail the comparison, at decode, verify and prefill buckets, the
//!   prefill row parts' sizes and one row either side of each tile edge and
//!   schedule edge; inputs of ordinary and extreme common-offset values,
//!   and streams seeded straight into the mixers' input with NaN payloads
//!   that meet in the same butterfly additions, infinities and overflowing
//!   squares (the reference must hold NaN and infinite projections, NaN
//!   rstd and NaN collapsed values);
//! - the probe's raw projection, rstd and Sinkhorn input cells bitwise equal
//!   to the one-CTA probe's, and the projection within 2^-22 K S of the
//!   Neumaier-compensated f64 projection (the 128-K accumulation gate);
//! - determinism: relaunches rewrite every output with the same bits;
//! - every live count of a 64-row capacity and of the prefill row parts at
//!   an 8192-row capacity: the live rows bitwise the one-CTA entry's, the
//!   rows past the live count left poisoned;
//! - poisoned replays (eager, relaunch, graph, PDL) of the served tile at
//!   every live count `poison::live_counts` lists plus the tile edges, after
//!   the producer whose streams are poisoned and checked too; each live
//!   count's eager outputs compared with the one-CTA entry's; with the
//!   producer left out, the streams and the collapsed rows must differ from
//!   the eager bits (operational errors propagate);
//! - mutants of the served tile, each caught only when an output differs
//!   from known good bits: the butterfly levels after the halvings in
//!   reverse order, another sum tree over the same lanes
//!   (`_levels_reversed`), and the live count and first chunk read before
//!   the dependency wait (`_late_wait`) behind the lab's PDL live-count
//!   producer, which holds the live count and the streams at zero through
//!   its tail; up to `LATE_WAIT_ATTEMPTS` attempts must change an output,
//!   or the test fails as an unexercised control.
//!
//! The CPU contract checks a lane-by-lane model of the tiled butterfly,
//! tied to the CUDA by the lines it emulates: every stored sum is lane 0's
//! `warp_sum` expression with the same operands in the same order.
//!
//! With `CHOPINFER_KERNEL_BENCH=1`, `sinkhorn_hc_tiled_times_by_rows` times
//! the one-CTA, split and every tiled entry at prefill row counts (cold L2
//! after the eviction, then the producer, then the timed mix) and fails
//! unless the served schedule (`tiled::served_prefill_mix`) beats the
//! entries it replaces at every measured count; the fastest entry is
//! reported.

use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{Argument, ArgumentList, DType, LaunchPlan};
use kernel_core::oracle::{OracleOutput, check_numeric};
use kernel_core::sinkhorn_hyper_connection::SinkhornHyperConnectionQuery as Query;
use kernel_core::sinkhorn_hyper_connection_oracle::{self as oracle, MixWeights};
use kernel_lab::launch::{self, Bindings};
use kernel_lab::poison::{self, LiveProducer, Phase, Replay, Rewrites, WriteMap};
use kernel_lab::{LabResult, Session};
use kernels_sm120::sinkhorn_hc::{self, Op, SERVED, split, tiled};
use runtime_cuda::{DeviceBuffer, Function, Module};
use serde_json::{Value, json};
use std::collections::BTreeMap;

const N: usize = 4;
const H: usize = 4096;
const FLAT: usize = N * H;
const MIXES: usize = (2 + N) * N;
const CELLS: usize = N * N;
/// Decode, verify and prefill buckets; the prefill row parts' sizes (2048
/// rows, a 3000-row prompt's 952-row tail, an 8064-row step's 1920-row
/// tail, the served 3008-row step's 960-row part and a 9K prompt's 768-row
/// step); one row either side of the tile edges and of the served
/// schedule's edges (`tiled::TILED_FROM_ROWS`, `tiled::EIGHT_ROWS_FROM`).
const BUCKETS: [usize; 37] = [
    1,
    2,
    3,
    4,
    5,
    7,
    8,
    9,
    15,
    16,
    17,
    31,
    32,
    33,
    64,
    143,
    144,
    6 * 32,
    255,
    256,
    768,
    952,
    960,
    1023,
    1024,
    1504,
    1505,
    1535,
    1536,
    1920,
    2047,
    2048,
    3000,
    4096,
    6144,
    8191,
    8192,
];
/// Live counts of an 8192-row capacity checked against the one-CTA entry:
/// the row parts' sizes and their tile edges.
const PART_LIVE: [usize; 18] = [
    0, 1, 767, 768, 769, 951, 952, 953, 959, 960, 961, 1919, 1920, 2048, 4096, 6144, 8064, 8192,
];
/// Capacities of the replay and mutant buckets, each run at every served
/// tile: decode-sized ones, the tile edges, verify and prefill buckets and
/// the served row parts (768, 960, 2048) and step (8192).
const REPLAY_ROWS: [usize; 13] = [1, 3, 4, 5, 7, 8, 9, 64, 6 * 32, 768, 960, 2048, 8192];
const MUTANT_ROWS: [usize; 3] = [256, 960, 2048];
/// Attempts of the late-wait mutant behind the live-count producer before
/// the mutant test fails as unexercised.
const LATE_WAIT_ATTEMPTS: usize = 4;
/// The prefill row counts the timing test measures: around the split /
/// 4-row tile crossover (between 128 and 192 in lab job g2-106), the 4-row
/// / 8-row one (between 1280 and 1535) and the 4-row tile's one round of
/// two CTAs per SM on 188 SMs (1504 rows).
const SCHEDULE_ROWS: [usize; 26] = [
    64, 96, 128, 144, 160, 176, 192, 255, 256, 512, 952, 1024, 1280, 1344, 1408, 1472, 1504, 1505,
    1535, 1536, 1792, 1920, 2048, 3000, 4096, 8192,
];

// Buffer handles: X1 = expand(X0, Y0, POST0, COMB0), then the one-CTA
// entries, the split and the tiled entries over X1 into their own outputs.
const X0: u64 = 0;
const Y0: u64 = 1;
const POST0: u64 = 2;
const COMB0: u64 = 3;
const X1: u64 = 4;
const FN: u64 = 5;
const BASE: u64 = 6;
const SCALE: u64 = 7;
const LIVE: u64 = 8;
const REF_COLLAPSED: u64 = 9;
const REF_POST: u64 = 10;
const REF_COMB: u64 = 11;
const REF_PROJECTED: u64 = 12;
const REF_RSTD: u64 = 13;
const REF_CELLS: u64 = 14;
const SPLIT_COLLAPSED: u64 = 15;
const SPLIT_POST: u64 = 16;
const SPLIT_COMB: u64 = 17;
const PARTIALS: u64 = 18;
const TICKETS: u64 = 19;
const COLLAPSED: u64 = 20;
const POST: u64 = 21;
const COMB: u64 = 22;
const PROJECTED: u64 = 23;
const RSTD: u64 = 24;
const CELL_INPUT: u64 = 25;
/// The streams' image the live-count producer restores.
const X1_IMAGE: u64 = 26;

const OUTPUTS: [u64; 3] = [COLLAPSED, POST, COMB];
const REF_OUTPUTS: [u64; 3] = [REF_COLLAPSED, REF_POST, REF_COMB];
const SPLIT_OUTPUTS: [u64; 3] = [SPLIT_COLLAPSED, SPLIT_POST, SPLIT_COMB];
const OUTPUT_NAMES: [&str; 3] = ["collapsed", "post", "comb"];
const PER_ROW: [usize; 3] = [H, N, CELLS];

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn tag() -> &'static str {
    sinkhorn_hc::SERVED_PROFILE.tag()
}

fn served_cubin() -> String {
    format!("sinkhorn_hc_{}_sm120", tag())
}

fn probe_cubin() -> String {
    format!("sinkhorn_hc_{}_probe_sm120", tag())
}

fn split_cubin() -> String {
    format!("sinkhorn_hc_{}_split_sm120", tag())
}

fn tiled_cubin(suffix: &str) -> String {
    format!("sinkhorn_hc_{}_tiled{suffix}_sm120", tag())
}

fn entry(op: Op) -> sinkhorn_hc::Entry {
    sinkhorn_hc::lab_catalog()
        .into_iter()
        .find(|entry| entry.op == op)
        .expect("a catalog entry")
}

fn tiled_entry(rows: u32, probe: bool) -> tiled::Entry {
    tiled::lab_entries()
        .into_iter()
        .find(|entry| entry.rows == rows && entry.probe == probe)
        .expect("a lab tiled entry")
}

/// Deterministic BF16 bits: random signs and mantissas over the binades
/// `low..=high`, with zeros, negative zeros and a subnormal among them
/// (as `tests/sinkhorn_hc.rs` draws them).
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

struct Inputs {
    /// Streams written straight into the mixers' input in place of the
    /// expand's (`Inputs::non_finite`).
    x1: Option<Vec<u16>>,
    x0: Vec<u16>,
    y0: Vec<u16>,
    post0: Vec<u16>,
    comb0: Vec<u16>,
    weights: Vec<u16>,
    base: Vec<f32>,
    scale: [f32; 3],
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
            x1: None,
            x0: bf16_values(rows * FLAT, seed + 1, -4, 2),
            y0: bf16_values(rows * H, seed + 2, -6, 1),
            post0: unit_values(rows * N, seed + 3, 2.0),
            comb0: unit_values(rows * CELLS, seed + 4, 1.0),
            // Logits of order one at K = 16,384.
            weights: bf16_values(MIXES * FLAT, seed + 6, -10, -6),
            base,
            scale: [0.5, 0.75, 2.0],
        }
    }

    /// Every comb base at the F32 value 1e8 with a zero comb scale.
    fn extreme(rows: usize, seed: u64) -> Self {
        let mut inputs = Self::new(rows, seed);
        inputs.base[2 * N..].fill(1e8);
        inputs.scale[2] = 0.0;
        inputs
    }

    /// Streams seeded straight into the mixers' input (no expand), by row
    /// mod 6: ordinary; NaN payloads 0x7fc1 and 0xffd3 at k = 0 and k = 64
    /// (lanes 0 and 16 of chain 0, which the first butterfly level adds;
    /// squares threads 0 and 8, which the third level adds); +Inf at k = 0
    /// and -Inf at k = 64; NaN payloads 0x7f81 and 0xffc2 at lanes 3 and 19
    /// of chain 37 with 0x7fa5 at lane 8 of chain 90; every element over
    /// binades 100-127 (products, squares and sums overflow); +Inf and -Inf
    /// in one column of two streams (the collapse's Inf - Inf). Payloads
    /// therefore meet in the same additions of both butterflies.
    fn non_finite(rows: usize, seed: u64) -> Self {
        let mut x1 = bf16_values(rows * FLAT, seed + 8, -4, 2);
        let huge = bf16_values(FLAT, seed + 9, 100, 127);
        for row in 0..rows {
            let x = &mut x1[row * FLAT..(row + 1) * FLAT];
            let column = (row * 977) % H;
            match row % 6 {
                1 => {
                    x[0] = 0x7fc1;
                    x[64] = 0xffd3;
                }
                2 => {
                    x[0] = 0x7f80;
                    x[64] = 0xff80;
                }
                3 => {
                    x[37 * 128 + 12] = 0x7f81;
                    x[37 * 128 + 76] = 0xffc2;
                    x[90 * 128 + 32] = 0x7fa5;
                }
                4 => x.copy_from_slice(&huge),
                5 => {
                    x[column] = 0x7f80;
                    x[H + column] = 0xff80;
                }
                _ => {}
            }
        }
        Self {
            x1: Some(x1),
            ..Self::new(rows, seed)
        }
    }
}

/// An entry's launch: its plan at the bucket, its function and its
/// argument packing.
struct Kernel {
    plan: LaunchPlan,
    function: Function,
    pack: Box<dyn Fn(Vec<Argument>) -> LabResult<ArgumentList>>,
}

fn kernel_of(module: &Module, entry: tiled::Entry, rows: usize) -> LabResult<Kernel> {
    Ok(Kernel {
        plan: entry.plan(rows as u32, 0).map_err(err)?,
        function: module.function(&entry.symbol).map_err(err)?,
        pack: Box::new(move |args| entry.pack_arguments(args).map_err(err)),
    })
}

struct Fixture {
    rows: usize,
    /// The streams were seeded (`Inputs::x1`): no producer runs.
    direct: bool,
    buffers: BTreeMap<u64, DeviceBuffer>,
    _modules: Vec<Module>,
    expand: Kernel,
    one_cta: Kernel,
    probe: Kernel,
    split: Kernel,
    /// The tiled entries and their probes by rows per CTA.
    tiled: BTreeMap<u32, (Kernel, Kernel)>,
    /// The served tiles' mutants by (rows per CTA, suffix).
    mutants: BTreeMap<(u32, &'static str), Kernel>,
}

impl Fixture {
    fn new(session: &Session, rows: usize, inputs: &Inputs) -> LabResult<Self> {
        let served = session.module(&served_cubin())?;
        let probe = session.module(&probe_cubin())?;
        let split_module = session.module(&split_cubin())?;
        let tiled_module = session.module(&tiled_cubin(""))?;
        let one = |module: &Module, op: Op| -> LabResult<Kernel> {
            let entry = entry(op);
            Ok(Kernel {
                plan: entry.plan(rows as u32, 0).map_err(err)?,
                function: module.function(&entry.symbol).map_err(err)?,
                pack: Box::new(move |args| entry.pack_arguments(args).map_err(err)),
            })
        };
        let split_entry = split::served();
        let split_kernel = Kernel {
            plan: split_entry.plan(rows as u32, 0).map_err(err)?,
            function: split_module.function(&split_entry.symbol).map_err(err)?,
            pack: Box::new(move |args| split_entry.pack_arguments(args).map_err(err)),
        };
        let mut tiled_kernels = BTreeMap::new();
        for tile in tiled::ROWS {
            tiled_kernels.insert(
                tile,
                (
                    kernel_of(&tiled_module, tiled_entry(tile, false), rows)?,
                    kernel_of(&tiled_module, tiled_entry(tile, true), rows)?,
                ),
            );
        }
        let mut modules = Vec::new();
        let mut mutants = BTreeMap::new();
        for (suffix, _) in tiled::MUTANTS {
            let module = session.module(&tiled_cubin(suffix))?;
            for tile in tiled::SERVED_TILES {
                mutants.insert(
                    (tile, suffix),
                    kernel_of(&module, tiled::mutant(tile, suffix), rows)?,
                );
            }
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
        buffers.insert(TICKETS, session.upload_f32(&vec![0.0; rows])?);
        for (handle, bytes) in [
            (X1, rows * FLAT * 2),
            (X1_IMAGE, rows * FLAT * 2),
            (LIVE, 4),
            (REF_COLLAPSED, rows * H * 2),
            (REF_POST, rows * N * 2),
            (REF_COMB, rows * CELLS * 2),
            (REF_PROJECTED, rows * MIXES * 4),
            (REF_RSTD, rows * 4),
            (REF_CELLS, rows * CELLS * 4),
            (SPLIT_COLLAPSED, rows * H * 2),
            (SPLIT_POST, rows * N * 2),
            (SPLIT_COMB, rows * CELLS * 2),
            (
                PARTIALS,
                split::partials_bytes(SERVED.mix, rows as u32) as usize,
            ),
            (COLLAPSED, rows * H * 2),
            (POST, rows * N * 2),
            (COMB, rows * CELLS * 2),
            (PROJECTED, rows * MIXES * 4),
            (RSTD, rows * 4),
            (CELL_INPUT, rows * CELLS * 4),
        ] {
            buffers.insert(handle, session.allocate(bytes)?);
        }
        if let Some(x1) = &inputs.x1 {
            poison::write(session, &buffers[&X1], 0, &bf16_bytes(x1))?;
        }
        let fixture = Self {
            rows,
            direct: inputs.x1.is_some(),
            buffers,
            expand: one(&served, Op::Expand)?,
            one_cta: one(&served, Op::MixCollapse { probe: false })?,
            probe: one(&probe, Op::MixCollapse { probe: true })?,
            split: split_kernel,
            tiled: tiled_kernels,
            mutants,
            _modules: {
                modules.extend([served, probe, split_module, tiled_module]);
                modules
            },
        };
        fixture.set_live(session, rows)?;
        Ok(fixture)
    }

    fn buffer(&self, handle: u64) -> &DeviceBuffer {
        &self.buffers[&handle]
    }

    fn set_live(&self, session: &Session, live: usize) -> LabResult<()> {
        poison::write(session, self.buffer(LIVE), 0, &(live as u32).to_ne_bytes())
    }

    fn launch(
        &self,
        session: &Session,
        kernel: &Kernel,
        handles: &[u64],
        pdl: bool,
    ) -> LabResult<()> {
        let mut plan = kernel.plan;
        plan.pdl = pdl;
        let args = (kernel.pack)(handles.iter().map(|&h| Argument::DeviceBuffer(h)).collect())?;
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

    /// The expand that writes the streams the mixes read (nothing when
    /// they were seeded).
    fn producer(&self, session: &Session) -> LabResult<()> {
        if self.direct {
            return Ok(());
        }
        let handles = [X0, Y0, POST0, COMB0, X1, LIVE];
        self.launch(session, &self.expand, &handles, false)
    }

    /// The one-CTA served entry, or its probe, into the reference outputs.
    fn one_cta(&self, session: &Session, probe: bool) -> LabResult<()> {
        if probe {
            let handles = [
                X1,
                FN,
                BASE,
                SCALE,
                REF_COLLAPSED,
                REF_POST,
                REF_COMB,
                REF_PROJECTED,
                REF_RSTD,
                REF_CELLS,
                LIVE,
            ];
            self.launch(session, &self.probe, &handles, false)
        } else {
            let handles = [X1, FN, BASE, SCALE, REF_COLLAPSED, REF_POST, REF_COMB, LIVE];
            self.launch(session, &self.one_cta, &handles, false)
        }
    }

    fn split(&self, session: &Session) -> LabResult<()> {
        let handles = [
            X1,
            FN,
            BASE,
            SCALE,
            SPLIT_COLLAPSED,
            SPLIT_POST,
            SPLIT_COMB,
            PARTIALS,
            TICKETS,
            LIVE,
        ];
        self.launch(session, &self.split, &handles, false)
    }

    /// The tiled entry of `tile` rows per CTA, or its probe.
    fn tiled(&self, session: &Session, tile: u32, probe: bool, pdl: bool) -> LabResult<()> {
        let (kernel, probe_kernel) = &self.tiled[&tile];
        if probe {
            let handles = [
                X1, FN, BASE, SCALE, COLLAPSED, POST, COMB, PROJECTED, RSTD, CELL_INPUT, LIVE,
            ];
            self.launch(session, probe_kernel, &handles, pdl)
        } else {
            let handles = [X1, FN, BASE, SCALE, COLLAPSED, POST, COMB, LIVE];
            self.launch(session, kernel, &handles, pdl)
        }
    }

    /// The served tile of `tile` rows per CTA.
    fn served(&self, session: &Session, tile: u32, pdl: bool) -> LabResult<()> {
        if !tiled::SERVED_TILES.contains(&tile) {
            return Err(format!("{tile} rows per CTA is not a served tile"));
        }
        self.tiled(session, tile, false, pdl)
    }

    fn mutant(&self, session: &Session, tile: u32, suffix: &str, pdl: bool) -> LabResult<()> {
        let kernel = self
            .mutants
            .get(&(tile, suffix))
            .ok_or_else(|| format!("no mutant {suffix} of r{tile}"))?;
        let handles = [X1, FN, BASE, SCALE, COLLAPSED, POST, COMB, LIVE];
        self.launch(session, kernel, &handles, pdl)
    }

    fn outputs(&self, session: &Session, handles: [u64; 3]) -> LabResult<[Vec<u16>; 3]> {
        let read =
            |i: usize| session.download_bf16(self.buffer(handles[i]), self.rows * PER_ROW[i]);
        Ok([read(0)?, read(1)?, read(2)?])
    }

    fn f32_bits(&self, session: &Session, handle: u64, per_row: usize) -> LabResult<Vec<u32>> {
        Ok(session
            .download_f32(self.buffer(handle), self.rows * per_row)?
            .iter()
            .map(|v| v.to_bits())
            .collect())
    }

    /// The probe outputs: projection, rstd, Sinkhorn input cells.
    fn probes(&self, session: &Session, handles: [u64; 3]) -> LabResult<[Vec<u32>; 3]> {
        Ok([
            self.f32_bits(session, handles[0], MIXES)?,
            self.f32_bits(session, handles[1], 1)?,
            self.f32_bits(session, handles[2], CELLS)?,
        ])
    }
}

fn bf16_bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
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

fn same_outputs(label: &str, actual: &[Vec<u16>; 3], expected: &[Vec<u16>; 3]) -> LabResult<()> {
    for ((name, actual), expected) in OUTPUT_NAMES.into_iter().zip(actual).zip(expected) {
        exact(&format!("{label} {name}"), actual, expected)?;
    }
    Ok(())
}

/// The outputs whose bits differ.
fn differing(actual: &[Vec<u16>; 3], expected: &[Vec<u16>; 3]) -> Vec<&'static str> {
    OUTPUT_NAMES
        .into_iter()
        .zip(actual)
        .zip(expected)
        .filter(|((_, a), e)| a != e)
        .map(|((name, _), _)| name)
        .collect()
}

/// Runs `run` over row chunks on every core, in row order.
fn by_rows<T: Send>(rows: usize, run: impl Fn(std::ops::Range<usize>) -> T + Sync) -> Vec<T> {
    let threads = std::thread::available_parallelism().map_or(8, |v| v.get());
    let per = rows.div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        let handles = (0..rows)
            .step_by(per)
            .map(|start| {
                let run = &run;
                scope.spawn(move || run(start..(start + per).min(rows)))
            })
            .collect::<Vec<_>>();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

fn concat(parts: Vec<OracleOutput>) -> OracleOutput {
    let mut out = OracleOutput {
        values: Vec::new(),
        sum_abs_products: Vec::new(),
        reduction_lengths: Vec::new(),
    };
    for part in parts {
        out.values.extend(part.values);
        out.sum_abs_products.extend(part.sum_abs_products);
        out.reduction_lengths.extend(part.reduction_lengths);
    }
    out
}

/// The largest error over its gate bound, for the receipt.
fn worst_ratio(actual: &[f64], reference: &OracleOutput) -> f64 {
    (0..actual.len())
        .map(|i| {
            let bound = kernel_core::oracle::numeric_bound(reference, i, DType::F32);
            (actual[i] - reference.values[i]).abs() / bound
        })
        .fold(0.0, f64::max)
}

/// The 128-K accumulation gate on a raw projection of finite inputs.
fn projection_gate(
    fixture: &Fixture,
    session: &Session,
    inputs: &Inputs,
    projected: &[u32],
) -> LabResult<f64> {
    let rows = fixture.rows;
    let x = session
        .download_bf16(fixture.buffer(X1), rows * FLAT)?
        .iter()
        .map(|&b| f64::from(bf16_to_f32(b)))
        .collect::<Vec<_>>();
    let weights = inputs
        .weights
        .iter()
        .map(|&b| f64::from(bf16_to_f32(b)))
        .collect::<Vec<_>>();
    let base = inputs
        .base
        .iter()
        .map(|&v| f64::from(v))
        .collect::<Vec<_>>();
    let scale = inputs.scale.map(f64::from);
    let mix_weights = MixWeights {
        projection: &weights,
        base: &base,
        scale: &scale,
    };
    let reference = concat(by_rows(rows, |r| {
        let query = Query::MixCollapse {
            mix: SERVED.mix,
            capacity: r.len() as u32,
        };
        oracle::projection(query, &x[r.start * FLAT..r.end * FLAT], mix_weights).unwrap()
    }));
    let actual = projected
        .iter()
        .map(|&v| f64::from(f32::from_bits(v)))
        .collect::<Vec<_>>();
    check_numeric(&actual, &reference, DType::F32).map_err(|f| {
        format!(
            "tiled projection (128-K accumulation): element {} is {}, reference {}, error {} over bound {}",
            f.index, f.actual, f.reference, f.error, f.bound
        )
    })?;
    Ok(worst_ratio(&actual, &reference))
}

/// Bitwise, oracle and determinism checks of one bucket; its receipt row.
/// `gate` runs the 128-K accumulation gate (finite inputs only).
fn bitwise_bucket(session: &Session, rows: usize, inputs: &Inputs, gate: bool) -> LabResult<Value> {
    let fixture = Fixture::new(session, rows, inputs)?;
    fixture.producer(session)?;
    fixture.one_cta(session, true)?;
    session.stream.synchronize().map_err(err)?;
    let probe = fixture.outputs(session, REF_OUTPUTS)?;
    let reference_probes = fixture.probes(session, [REF_PROJECTED, REF_RSTD, REF_CELLS])?;
    // Seeded non-finite streams must reach the projection, rstd and the
    // collapse as NaN and infinite values, or the bucket proves nothing
    // about them.
    let non_finite = if inputs.x1.is_some() {
        let count = |bits: &[u32], test: fn(f32) -> bool| {
            bits.iter().filter(|&&b| test(f32::from_bits(b))).count()
        };
        let collapsed = session.download_bf16(fixture.buffer(REF_COLLAPSED), rows * H)?;
        let collapsed_nan = collapsed
            .iter()
            .filter(|&&b| bf16_to_f32(b).is_nan())
            .count();
        let evidence = json!({
            "projection_nan": count(&reference_probes[0], f32::is_nan),
            "projection_inf": count(&reference_probes[0], f32::is_infinite),
            "rstd_nan": count(&reference_probes[1], f32::is_nan),
            "collapsed_nan": collapsed_nan,
        });
        for key in [
            "projection_nan",
            "projection_inf",
            "rstd_nan",
            "collapsed_nan",
        ] {
            if evidence[key].as_u64() == Some(0) {
                return Err(format!(
                    "{rows} rows: the seeded streams left no {key} in the one-CTA reference"
                ));
            }
        }
        Some(evidence)
    } else {
        None
    };
    poison::poison(session, &REF_OUTPUTS.map(|h| fixture.buffer(h)))?;
    fixture.one_cta(session, false)?;
    session.stream.synchronize().map_err(err)?;
    let one_cta = fixture.outputs(session, REF_OUTPUTS)?;
    same_outputs("one-CTA served against its probe", &one_cta, &probe)?;
    fixture.split(session)?;
    session.stream.synchronize().map_err(err)?;
    same_outputs(
        "split against the one-CTA entry",
        &fixture.outputs(session, SPLIT_OUTPUTS)?,
        &one_cta,
    )?;

    let outputs = [COLLAPSED, POST, COMB, PROJECTED, RSTD, CELL_INPUT].map(|h| fixture.buffer(h));
    let mut tiles = Vec::new();
    for tile in tiled::ROWS {
        let label = format!("{rows} rows, {tile} rows per CTA");
        // Into poisoned outputs, which alone fail the comparison.
        poison::poison(session, &outputs)?;
        let unwritten = fixture.outputs(session, OUTPUTS)?;
        if differing(&unwritten, &one_cta).len() != OUTPUTS.len() {
            return Err(format!(
                "{label}: unwritten outputs matched the one-CTA entry"
            ));
        }
        fixture.tiled(session, tile, false, false)?;
        session.stream.synchronize().map_err(err)?;
        same_outputs(
            &format!("{label}: tiled against the one-CTA entry"),
            &fixture.outputs(session, OUTPUTS)?,
            &one_cta,
        )?;
        poison::poison(session, &outputs)?;
        fixture.tiled(session, tile, true, false)?;
        session.stream.synchronize().map_err(err)?;
        same_outputs(
            &format!("{label}: tiled probe against the one-CTA entry"),
            &fixture.outputs(session, OUTPUTS)?,
            &one_cta,
        )?;
        let probes = fixture.probes(session, [PROJECTED, RSTD, CELL_INPUT])?;
        for ((name, actual), expected) in ["projection", "rstd", "Sinkhorn input cells"]
            .into_iter()
            .zip(&probes)
            .zip(&reference_probes)
        {
            exact(
                &format!("{label}: {name} against the one-CTA probe"),
                actual,
                expected,
            )?;
        }
        tiles.push(json!({
            "rows_per_cta": tile,
            "grid": fixture.tiled[&tile].0.plan.grid,
            "bitwise_one_cta": true,
            "probes_bitwise_one_cta": true,
        }));
    }
    // Every tile's projection is the one-CTA probe's bit for bit; the gate
    // runs once on those bits.
    let worst = if gate {
        Some(projection_gate(
            &fixture,
            session,
            inputs,
            &reference_probes[0],
        )?)
    } else {
        None
    };

    // Determinism: relaunches of each served tile give the same bits.
    for (tile, relaunch) in tiled::SERVED_TILES
        .into_iter()
        .flat_map(|tile| (0..4).map(move |relaunch| (tile, relaunch)))
    {
        poison::poison(session, &outputs)?;
        fixture.served(session, tile, relaunch % 2 == 1)?;
        session.stream.synchronize().map_err(err)?;
        same_outputs(
            &format!("{rows} rows, r{tile} relaunch {relaunch}"),
            &fixture.outputs(session, OUTPUTS)?,
            &one_cta,
        )?;
    }
    Ok(json!({
        "rows": rows,
        "bitwise_split": true,
        "tiles": tiles,
        "projection_worst_over_bound": worst,
        "non_finite_in_reference": non_finite,
        "relaunches_bitwise": {"rows_per_cta": tiled::SERVED_TILES, "each": 4},
    }))
}

/// The tiled entries at each of `lives` live counts of one fixture into
/// poisoned outputs: the live rows bitwise the one-CTA entry's at full
/// capacity (rows are independent), the rows past the live count still
/// poisoned.
fn live_bucket(
    session: &Session,
    rows: usize,
    inputs: &Inputs,
    lives: &[usize],
) -> LabResult<Value> {
    let fixture = Fixture::new(session, rows, inputs)?;
    fixture.producer(session)?;
    fixture.one_cta(session, false)?;
    session.stream.synchronize().map_err(err)?;
    let one_cta = fixture.outputs(session, REF_OUTPUTS)?;
    let outputs = OUTPUTS.map(|h| fixture.buffer(h));
    poison::poison(session, &outputs)?;
    let poisoned = fixture.outputs(session, OUTPUTS)?;
    for &live in lives {
        fixture.set_live(session, live)?;
        for tile in tiled::ROWS {
            let label = format!("{rows} rows, live {live}, {tile} rows per CTA");
            poison::poison(session, &outputs)?;
            fixture.tiled(session, tile, false, false)?;
            session.stream.synchronize().map_err(err)?;
            let actual = fixture.outputs(session, OUTPUTS)?;
            for (((name, actual), expected), (poisoned, per_row)) in OUTPUT_NAMES
                .into_iter()
                .zip(&actual)
                .zip(&one_cta)
                .zip(poisoned.iter().zip(PER_ROW))
            {
                let at = live * per_row;
                exact(
                    &format!("{label}: live {name}"),
                    &actual[..at],
                    &expected[..at],
                )?;
                exact(
                    &format!("{label}: {name} past the live count"),
                    &actual[at..],
                    &poisoned[at..],
                )?;
            }
        }
    }
    Ok(json!({"rows": rows, "live_counts": lives, "rows_per_cta": tiled::ROWS}))
}

/// A served tile's live rows against the one-CTA entry over the same
/// streams at the same live count.
fn against_one_cta(
    fixture: &Fixture,
    session: &Session,
    live: usize,
    label: &str,
) -> LabResult<()> {
    session.stream.synchronize().map_err(err)?;
    poison::poison(session, &REF_OUTPUTS.map(|h| fixture.buffer(h)))?;
    fixture.one_cta(session, false)?;
    session.stream.synchronize().map_err(err)?;
    let take = |handles: [u64; 3]| -> LabResult<[Vec<u16>; 3]> {
        let mut outputs = fixture.outputs(session, handles)?;
        for (output, per_row) in outputs.iter_mut().zip(PER_ROW) {
            output.truncate(live * per_row);
        }
        Ok(outputs)
    };
    same_outputs(
        &format!("{label}: tiled against the one-CTA entry"),
        &take(OUTPUTS)?,
        &take(REF_OUTPUTS)?,
    )
}

/// The live counts a replay bucket of `rows` covers at `tile` rows per
/// CTA: `poison::live_counts` plus each side of the tile's first edges and
/// of its last full tile.
fn replay_lives(rows: usize, tile: u32) -> Vec<usize> {
    let tile = tile as usize;
    let last = rows / tile * tile;
    let mut lives = poison::live_counts(rows as u32)
        .into_iter()
        .map(|live| live as usize)
        .collect::<Vec<_>>();
    for live in [
        tile - 1,
        tile,
        tile + 1,
        2 * tile + 1,
        last.saturating_sub(1),
        last + 1,
    ] {
        if (1..=rows).contains(&live) && !lives.contains(&live) {
            lives.push(live);
        }
    }
    lives
}

/// The served tile of `tile` rows per CTA on poisoned buffers at every
/// live count of [`replay_lives`], after its producer (an expand) whose
/// streams are poisoned and checked too. Each live count's eager outputs
/// are also compared with the one-CTA entry, and at one partial live count
/// the producer left out must leave the streams poisoned and change the
/// collapsed rows.
fn replay_bucket(session: &Session, rows: usize, tile: u32, inputs: &Inputs) -> LabResult<Value> {
    let fixture = Fixture::new(session, rows, inputs)?;
    let graph = session.capture(|_| {
        fixture.producer(session)?;
        fixture.served(session, tile, true)
    })?;
    let mut evidence = Vec::new();
    for live in replay_lives(rows, tile) {
        let label = format!("{rows} rows, r{tile}, live {live}");
        fixture.set_live(session, live)?;
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
        let both = || -> LabResult<()> {
            fixture.producer(session)?;
            fixture.served(session, tile, false)
        };
        replay.eager(both)?;
        // The eager bits against the one-CTA entry, independently of the
        // replays' self-comparison.
        against_one_cta(&fixture, session, live, &label)?;
        replay.relaunch(both)?;
        replay.graph(&graph)?;
        replay.pdl(
            || fixture.producer(session),
            || fixture.served(session, tile, true),
            true,
        )?;
        let mut receipt = BTreeMap::new();
        replay.record(&mut receipt);
        // Without the producer (at one partial live count, all rows of one
        // row) the streams keep their poison and the collapsed rows differ
        // from the eager bits. Launch, synchronization and readback errors
        // propagate rather than count as caught.
        let mut mutations = Vec::new();
        if live == (rows / 2).max(1) {
            replay.prepare(Phase::Pdl)?;
            fixture.served(session, tile, true)?;
            session.stream.synchronize().map_err(err)?;
            let mut changed = Vec::new();
            for (output, handle) in [("x1", X1), ("collapsed", COLLAPSED)] {
                let actual = poison::read(session, fixture.buffer(handle))?;
                let eager = replay
                    .expected(output)
                    .ok_or_else(|| format!("{label}: no eager {output}"))?;
                if actual != eager {
                    changed.push(output);
                }
            }
            if changed != ["x1", "collapsed"] {
                return Err(format!(
                    "{label}: without the producer only {changed:?} differ from the eager bits"
                ));
            }
            mutations.push(json!({"mutation": "no producer", "outputs": changed}));
        }
        evidence.push(json!({"live": live, "replay": receipt, "mutations_caught": mutations}));
    }
    Ok(json!({"rows": rows, "rows_per_cta": tile, "live_counts": evidence}))
}

/// The mutants of the served tile of `tile` rows per CTA at `rows` rows,
/// each against the served tile's eager bits.
fn mutant_bucket(session: &Session, rows: usize, tile: u32, inputs: &Inputs) -> LabResult<Value> {
    let fixture = Fixture::new(session, rows, inputs)?;
    let label = format!("{rows} rows, r{tile}");
    fixture.producer(session)?;
    fixture.served(session, tile, false)?;
    session.stream.synchronize().map_err(err)?;
    let eager = fixture.outputs(session, OUTPUTS)?;
    let outputs = OUTPUTS.map(|h| fixture.buffer(h));

    // The butterfly levels after the halvings in reverse order: another sum
    // tree over the same lanes and rows.
    poison::poison(session, &outputs)?;
    fixture.mutant(session, tile, "_levels_reversed", false)?;
    session.stream.synchronize().map_err(err)?;
    let reversed = differing(&fixture.outputs(session, OUTPUTS)?, &eager);
    if reversed.is_empty() {
        return Err(format!(
            "{label}: the levels-reversed mutant reproduced the served bits"
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
            fixture.mutant(session, tile, "_late_wait", true)?;
        } else {
            fixture.served(session, tile, true)?;
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
    // The mutant counts only when its outputs differ; an unchanged attempt
    // proves nothing (the producer may have restored its inputs before the
    // mutant's early reads), so a bounded number of attempts must catch it.
    let mut attempts = 0;
    let late = loop {
        attempts += 1;
        let late = behind(true)?;
        if !late.is_empty() {
            break late;
        }
        if attempts == LATE_WAIT_ATTEMPTS {
            return Err(format!(
                "{label}: the late-wait mutant reproduced the eager bits in all {attempts} \
                 attempts: the negative control is unexercised, qualify again"
            ));
        }
    };
    Ok(json!({
        "rows": rows,
        "rows_per_cta": tile,
        "mutants": [
            {"mutation": "butterfly levels after the halvings reversed", "outcome": "caught",
             "outputs": reversed},
            {"mutation": "live count and first chunk read before the dependency wait",
             "outcome": "caught", "attempts": attempts, "outputs": late},
        ],
    }))
}

#[test]
fn sinkhorn_hc_tiled_cpu_contract() {
    let names = kernel_lab::cubin_entries(&tiled_cubin("")).unwrap();
    for entry in tiled::lab_entries() {
        assert!(names.contains(&entry.symbol.as_str()), "{}", entry.symbol);
    }
    for (suffix, _) in tiled::MUTANTS {
        assert_eq!(
            kernel_lab::cubin_entries(&tiled_cubin(suffix)).unwrap(),
            tiled::SERVED_TILES.map(|tile| tiled::mutant(tile, suffix).symbol)
        );
    }
    for tile in tiled::SERVED_TILES {
        let served = tiled::served(tile);
        for rows in BUCKETS.into_iter().chain(REPLAY_ROWS).chain(MUTANT_ROWS) {
            let plan = served.plan(rows as u32, rows as u32).unwrap();
            assert_eq!(plan.grid, [rows.div_ceil(tile as usize) as u32, 1, 1]);
            assert!(plan.pdl);
        }
    }
    // The served row parts are buckets, live counts and replays.
    for part in [768, 960, 2048] {
        assert!(
            BUCKETS.contains(&part) && PART_LIVE.contains(&part) && REPLAY_ROWS.contains(&part)
        );
    }
    assert!(PART_LIVE.iter().all(|&live| live <= 8192));
    assert_eq!(replay_lives(64, 8), [64, 0, 1, 32, 63, 7, 8, 9, 17]);
    assert_eq!(replay_lives(64, 4), [64, 0, 1, 32, 63, 3, 4, 5, 9]);
    assert_eq!(replay_lives(4, 8), [4, 0, 1, 2, 3]);
    assert_eq!(replay_lives(4, 4), [4, 0, 1, 2, 3]);
    assert_eq!(replay_lives(960, 8), [960, 0, 1, 480, 959, 7, 8, 9, 17]);
    let unwritten = [vec![0xffff_u16; H], vec![0xffff; N], vec![0xffff; CELLS]];
    let written = [vec![0x3f80_u16; H], vec![0x3f00; N], vec![0x3e80; CELLS]];
    assert_eq!(differing(&unwritten, &written), OUTPUT_NAMES);
    assert!(differing(&written, &written).is_empty());
    // The schedule check: served split at 64 rows, served r4 at 952.
    let bucket = |rows: u32, one: f64, split: f64, tiles: [f64; 3]| {
        json!({"rows": rows, "one_cta_median_ms": one, "split_median_ms": split,
               "tiles": tiled::ROWS.iter().zip(tiles).map(|(r, ms)| {
                   json!({"rows_per_cta": r, "median_ms": ms})
               }).collect::<Vec<_>>()})
    };
    let check = schedule_check(&bucket(64, 0.15, 0.049, [0.084, 0.115, 0.166]));
    assert_eq!(check["served"], "split");
    assert_eq!(check["beats_replaced"], true);
    assert_eq!(check["fastest"], "split");
    let check = schedule_check(&bucket(952, 0.295, 0.436, [0.102, 0.117, 0.166]));
    assert_eq!(
        (check["served"].clone(), check["beats_replaced"].clone()),
        (json!("r4"), json!(true))
    );
    let check = schedule_check(&bucket(952, 0.295, 0.09, [0.102, 0.08, 0.166]));
    assert_eq!(check["beats_replaced"], false);
    assert_eq!(check["fastest"], "r8");
    let check = schedule_check(&json!({"rows": 2048, "one_cta_median_ms": 1.0}));
    assert_eq!(check["beats_replaced"], false);
    lane_trees();
    squares_order();
    let inputs = Inputs::non_finite(6, 1);
    let x1 = inputs.x1.as_ref().unwrap();
    assert_eq!((x1[FLAT], x1[FLAT + 64]), (0x7fc1, 0xffd3));
    assert_eq!((x1[2 * FLAT], x1[2 * FLAT + 64]), (0x7f80, 0xff80));
    assert_eq!(x1[3 * FLAT + 37 * 128 + 76], 0xffc2);
    assert_eq!(inputs.weights.len(), MIXES * FLAT);
}

/// An ordered F32 addition tree over (row, value, lane) leaves.
#[derive(Clone, Debug, PartialEq)]
enum Tree {
    Leaf(usize, usize, usize),
    Add(Box<Tree>, Box<Tree>),
}

fn add(a: &Tree, b: &Tree) -> Tree {
    Tree::Add(Box::new(a.clone()), Box::new(b.clone()))
}

/// The expression `warp_sum` leaves in lane 0 for one (row, value).
fn warp_sum_lane0(row: usize, value: usize) -> Tree {
    let mut lanes = (0..32)
        .map(|lane| Tree::Leaf(row, value, lane))
        .collect::<Vec<_>>();
    for offset in [16, 8, 4, 2, 1] {
        lanes = (0..32)
            .map(|lane| add(&lanes[lane], &lanes[lane ^ offset]))
            .collect();
    }
    lanes.swap_remove(0)
}

/// `tiled_detail::reduce<rows * per_warp, log2(rows)>` lane by lane: the
/// halvings (`send`, `keep`, and `peer + keep` on a lane with the level bit
/// when `oriented`, else `keep + peer`), then the remaining levels as
/// `warp_sum` (from offset 1 up when `reversed`). Each lane's kept trees.
fn reduce_model(rows: usize, per_warp: usize, oriented: bool, reversed: bool) -> Vec<Vec<Tree>> {
    let levels = rows.trailing_zeros() as usize;
    let count = rows * per_warp;
    let mut v = (0..32)
        .map(|lane| {
            (0..count)
                .map(|j| Tree::Leaf(j / per_warp, j % per_warp, lane))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    for level in 0..levels {
        let offset = 16 >> level;
        let half = count >> (level + 1);
        let prior = v.clone();
        for (lane, kept) in v.iter_mut().enumerate() {
            let upper = lane & offset != 0;
            let partner = lane ^ offset;
            for (k, slot) in kept.iter_mut().enumerate().take(half) {
                let keep = &prior[lane][if upper { k + half } else { k }];
                // The partner sends what it does not keep.
                let peer = &prior[partner][if upper { k + half } else { k }];
                *slot = if upper && oriented {
                    add(peer, keep)
                } else {
                    add(keep, peer)
                };
            }
        }
    }
    for level in levels..5 {
        let offset = if reversed {
            1 << (level - levels)
        } else {
            16 >> level
        };
        let prior = v.clone();
        for (lane, kept) in v.iter_mut().enumerate() {
            for (k, slot) in kept.iter_mut().enumerate().take(count >> levels) {
                *slot = add(&prior[lane][k], &prior[lane ^ offset][k]);
            }
        }
    }
    v
}

/// Every stored sum of the tiled butterfly (the group's first lane, which
/// stores logits, the probe's projection and the squares' warp partials) is
/// lane 0's `warp_sum` expression with the same operands in the same order,
/// for every tile and the projection's 3 and the squares' 1 value per row;
/// without the orientation the trees differ by operand swaps, and the
/// reversed-levels mutant changes the 4- and 8-row trees. The model is tied
/// to the CUDA by the lines it emulates.
fn lane_trees() {
    let source = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../kernels-sm120/cuda/sinkhorn_hc/mix_tiled.cuh"
    ));
    let one_cta = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../kernels-sm120/cuda/sinkhorn_hc/sinkhorn_hc.cuh"
    ));
    for line in [
        "const int offset = 16 >> level;",
        "const bool upper = (lane & offset) != 0;",
        "const float send = upper ? low : high;",
        "const float keep = upper ? high : low;",
        "const float peer = __shfl_xor_sync(0xffffffffu, send, offset);",
        "v[k] = upper ? __fadd_rn(peer, keep) : __fadd_rn(keep, peer);",
        "v[k] = __fadd_rn(v[k], __shfl_xor_sync(0xffffffffu, v[k], offset));",
        "const int offset = 1 << (level - Levels);",
        "const bool leader = lane % T::kGroupLanes == 0;",
        "tiled_detail::reduce<Rows * PerWarp, T::kLevels>(sums, lane);",
        "tiled_detail::reduce<Rows, T::kLevels>(squares, lane);",
    ] {
        assert!(
            source.contains(line),
            "mix_tiled.cuh no longer has `{line}`"
        );
    }
    assert!(one_cta.contains("for (int offset = 16; offset > 0; offset >>= 1) {"));
    assert!(
        one_cta.contains("value = __fadd_rn(value, __shfl_xor_sync(0xffffffffu, value, offset));")
    );
    for rows in tiled::ROWS.map(|r| r as usize) {
        for per_warp in [3, 1] {
            let model = reduce_model(rows, per_warp, true, false);
            let swapped = reduce_model(rows, per_warp, false, false);
            let reversed = reduce_model(rows, per_warp, true, true);
            let mut swaps = 0;
            let mut moved = 0;
            for row in 0..rows {
                let leader = row * (32 / rows);
                for value in 0..per_warp {
                    let expected = warp_sum_lane0(row, value);
                    assert_eq!(model[leader][value], expected, "{rows} rows, row {row}");
                    swaps += usize::from(swapped[leader][value] != expected);
                    moved += usize::from(reversed[leader][value] != expected);
                }
            }
            assert_eq!(swaps, (rows - 1) * per_warp, "{rows} rows");
            // One level remains after the 16-row tile's halvings, so
            // reversing the remaining levels changes nothing there.
            let changed = if rows < 16 { rows * per_warp } else { 0 };
            assert_eq!(moved, changed, "{rows} rows");
        }
    }
}

/// Thread t of the tiled squares FMAs the elements of packs t, t + 256, ...
/// of every row in the one-CTA order, at every tile and admitted width.
fn squares_order() {
    for rows in tiled::ROWS.map(|r| r as usize) {
        let chunk = tiled::chunk_chains(rows as u32) as usize * 128;
        let packs = chunk / 8;
        let period = 256 / packs;
        for flat in (256..=16_384).step_by(128).filter(|f| f % chunk == 0) {
            for t in 0..256 {
                let expected = (t..flat / 8)
                    .step_by(256)
                    .flat_map(|v| (0..8).map(move |e| 8 * v + e))
                    .collect::<Vec<_>>();
                let got = (0..flat / chunk)
                    .filter(|c| t / packs == c % period)
                    .flat_map(|c| (0..8).map(move |e| c * chunk + (t % packs) * 8 + e))
                    .collect::<Vec<_>>();
                assert_eq!(got, expected, "{rows} rows, flat {flat}, thread {t}");
            }
        }
    }
}

#[test]
fn sinkhorn_hc_tiled_matches_the_prefill_entries_bitwise() {
    kernel_lab::gpu_test(3 << 30, |session| {
        let mut rows = BUCKETS
            .iter()
            .map(|&rows| {
                bitwise_bucket(
                    session,
                    rows,
                    &Inputs::new(rows, 0x5eed + rows as u64),
                    true,
                )
            })
            .collect::<LabResult<Vec<_>>>()?;
        let mut extreme = bitwise_bucket(session, 33, &Inputs::extreme(33, 0xe0), true)?;
        extreme["extreme_common_offset"] = json!(1e8);
        rows.push(extreme);
        for bucket in [32, 192] {
            let mut non_finite =
                bitwise_bucket(session, bucket, &Inputs::non_finite(bucket, 0xa7), false)?;
            non_finite["non_finite"] = json!(true);
            rows.push(non_finite);
        }
        let lives = [
            live_bucket(
                session,
                64,
                &Inputs::new(64, 0x11fe),
                &(0..=64).collect::<Vec<_>>(),
            )?,
            live_bucket(session, 8192, &Inputs::new(8192, 0x11ff), &PART_LIVE)?,
        ];
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("sinkhorn_hc_tiled_bitwise"));
        receipt.insert("buckets".into(), json!(rows));
        receipt.insert("live_counts".into(), json!(lives));
        session.receipt("sinkhorn_hc_tiled_bitwise", receipt)?;
        Ok(())
    })
    .expect("tiled Sinkhorn mix-collapse bitwise comparison");
}

#[test]
fn sinkhorn_hc_tiled_replays_on_poisoned_buffers() {
    kernel_lab::gpu_test(3 << 30, |session| {
        let mut rows = Vec::new();
        for tile in tiled::SERVED_TILES {
            for rows_ in REPLAY_ROWS {
                let inputs = Inputs::new(rows_, 0xfeed + rows_ as u64);
                rows.push(replay_bucket(session, rows_, tile, &inputs)?);
            }
            let mut extreme = replay_bucket(session, 32, tile, &Inputs::extreme(32, 0xfe))?;
            extreme["extreme_common_offset"] = json!(1e8);
            rows.push(extreme);
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("sinkhorn_hc_tiled_poison"));
        receipt.insert("buckets".into(), json!(rows));
        session.receipt("sinkhorn_hc_tiled_poison", receipt)?;
        Ok(())
    })
    .expect("tiled Sinkhorn mix-collapse poisoned replays");
}

#[test]
fn sinkhorn_hc_tiled_mutants_are_caught() {
    kernel_lab::gpu_test(3 << 30, |session| {
        let mut rows = Vec::new();
        for tile in tiled::SERVED_TILES {
            for rows_ in MUTANT_ROWS {
                let inputs = Inputs::new(rows_, 0x3a7 + rows_ as u64);
                rows.push(mutant_bucket(session, rows_, tile, &inputs)?);
            }
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("sinkhorn_hc_tiled_mutants"));
        receipt.insert("buckets".into(), json!(rows));
        session.receipt("sinkhorn_hc_tiled_mutants", receipt)?;
        Ok(())
    })
    .expect("tiled Sinkhorn mix-collapse mutants");
}

/// The one-CTA entry, the split and every tile timed at one row count:
/// each launch after an L2 eviction and the producer that writes its
/// streams, as a prefill mixer follows its sublayer's expand; then the
/// outputs compared.
fn timed_bucket(session: &Session, rows: usize, inputs: &Inputs) -> LabResult<Value> {
    const WARMUP: usize = 2;
    const ROUNDS: usize = 15;
    let fixture = Fixture::new(session, rows, inputs)?;
    let mode = kernel_lab::eviction::EvictionMode::selected()?;
    let time = |label: &str, run: &dyn Fn() -> LabResult<()>| {
        kernel_lab::with_timing_label(&format!("{label} {rows} rows"), || {
            kernel_lab::time_prepared(
                session,
                WARMUP,
                ROUNDS,
                mode,
                |_| fixture.producer(session),
                |_| run(),
            )
        })
    };
    let one_cta = time("one-CTA mix-collapse", &|| fixture.one_cta(session, false))?;
    let split = time("split mix-collapse", &|| fixture.split(session))?;
    let mut tiles = Vec::new();
    for tile in tiled::ROWS {
        let timing = time(&format!("tiled mix-collapse r{tile}"), &|| {
            fixture.tiled(session, tile, false, false)
        })?;
        session.stream.synchronize().map_err(err)?;
        same_outputs(
            &format!("{rows} rows: timed tile r{tile} against the timed one-CTA entry"),
            &fixture.outputs(session, OUTPUTS)?,
            &fixture.outputs(session, REF_OUTPUTS)?,
        )?;
        tiles.push(json!({
            "rows_per_cta": tile,
            "median_ms": timing.median_ms,
            "min_ms": timing.min_ms,
            "over_one_cta": timing.median_ms / one_cta.median_ms,
            "over_split": timing.median_ms / split.median_ms,
        }));
    }
    session.stream.synchronize().map_err(err)?;
    same_outputs(
        &format!("{rows} rows: timed split against the timed one-CTA entry"),
        &fixture.outputs(session, SPLIT_OUTPUTS)?,
        &fixture.outputs(session, REF_OUTPUTS)?,
    )?;
    Ok(json!({
        "rows": rows,
        "one_cta_median_ms": one_cta.median_ms,
        "one_cta_min_ms": one_cta.min_ms,
        "split_median_ms": split.median_ms,
        "split_min_ms": split.min_ms,
        "served": format!("{:?}", tiled::served_prefill_mix(rows as u32)),
        "tiles": tiles,
        "rounds": ROUNDS,
    }))
}

/// The served prefill schedule (`tiled::served_prefill_mix`) against one
/// timed bucket: the served entry's median and the fastest entry's, and
/// whether the served entry beats every entry it replaces (a served tile:
/// the one-CTA entry and the split; the served split: the one-CTA entry).
fn schedule_check(bucket: &Value) -> Value {
    let rows = bucket["rows"].as_u64().unwrap_or(0) as u32;
    let median = |v: &Value| v.as_f64().unwrap_or(f64::INFINITY);
    let one_cta = median(&bucket["one_cta_median_ms"]);
    let split = median(&bucket["split_median_ms"]);
    let tile = |rows_per_cta: u32| {
        bucket["tiles"]
            .as_array()
            .and_then(|tiles| {
                tiles
                    .iter()
                    .find(|t| t["rows_per_cta"].as_u64() == Some(u64::from(rows_per_cta)))
            })
            .map_or(f64::INFINITY, |t| median(&t["median_ms"]))
    };
    let mut entries = vec![("one-CTA".to_owned(), one_cta), ("split".to_owned(), split)];
    entries.extend(tiled::ROWS.map(|r| (format!("r{r}"), tile(r))));
    let (served_name, served, replaced) = match tiled::served_prefill_mix(rows) {
        tiled::PrefillMix::Split => ("split".to_owned(), split, vec![one_cta]),
        tiled::PrefillMix::Tiled(r) => (format!("r{r}"), tile(r), vec![one_cta, split]),
    };
    let (best_name, best) =
        entries
            .iter()
            .cloned()
            .fold(("none".to_owned(), f64::INFINITY), |a, b| {
                if b.1 < a.1 { b } else { a }
            });
    json!({
        "rows": rows,
        "served": served_name,
        "served_median_ms": served,
        "beats_replaced": served.is_finite() && replaced.iter().all(|&r| served < r),
        "fastest": best_name,
        "served_over_fastest": served / best,
    })
}

#[test]
fn sinkhorn_hc_tiled_times_by_rows() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("sinkhorn_hc_tiled timing: set CHOPINFER_KERNEL_BENCH=1").unwrap();
        return;
    }
    kernel_lab::gpu_test(3 << 30, |session| {
        let buckets = SCHEDULE_ROWS
            .iter()
            .map(|&rows| {
                let bucket = timed_bucket(session, rows, &Inputs::new(rows, 0x7135 + rows as u64))?;
                eprintln!("sinkhorn_hc_tiled timing: {bucket}");
                Ok(bucket)
            })
            .collect::<LabResult<Vec<_>>>()?;
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("sinkhorn_hc_tiled_timing"));
        receipt.insert(
            "eviction".into(),
            json!(kernel_lab::eviction::EvictionMode::selected()?.name()),
        );
        let schedule = buckets.iter().map(schedule_check).collect::<Vec<_>>();
        for check in &schedule {
            eprintln!("sinkhorn_hc_tiled schedule: {check}");
        }
        receipt.insert("schedule".into(), json!(schedule));
        receipt.insert("buckets".into(), json!(buckets));
        session.receipt("sinkhorn_hc_tiled_timing", receipt)?;
        // The served schedule must beat the entries it replaces at every
        // measured count; a faster tile elsewhere is reported, not failed.
        let slower = schedule
            .iter()
            .filter(|check| check["beats_replaced"] != json!(true))
            .collect::<Vec<_>>();
        if !slower.is_empty() {
            return Err(format!(
                "the served prefill schedule does not beat the entries it replaces: {slower:?}"
            ));
        }
        Ok(())
    })
    .expect("tiled Sinkhorn mix-collapse timing");
}
