//! The split Sinkhorn mix-collapse (`kernels_sm120::sinkhorn_hc::split`)
//! against the one-CTA entry it replaces, at the served 4 x 4096 mixer, on
//! the GPU. Run with `CHOPINFER_KERNEL_GPU=<ordinal> cargo test -p
//! kernel-lab --release --test sinkhorn_hc_split -- --test-threads=1`;
//! without a GPU only the CPU contract test runs.
//!
//! At decode (1, 2, 4, 16, 32), verify (6 x 32) and prefill (64, 256, 1024,
//! 2048, 4096, 8192) buckets, after an expand writes the streams as a
//! sublayer boundary does:
//!
//! - collapsed, post and comb bitwise equal to the one-CTA served cubin's,
//!   written into freshly poisoned buffers that alone fail the comparison;
//! - the chain sums added on the host from zero in K order bitwise equal to
//!   the one-CTA probe's raw projection, and within 2^-22 K S of the
//!   Neumaier-compensated f64 projection (the 128-K accumulation gate);
//! - determinism: relaunches rewrite every output and chain sum with the
//!   same bits, and leave every ticket at zero;
//! - poisoned replays (eager, relaunch, graph, PDL) at every live count
//!   `poison::live_counts` lists, after the producer whose streams are
//!   poisoned and checked too, the chain sums poisoned as a written
//!   workspace and the tickets checked as zeroed state; each live count's
//!   eager outputs compared with the one-CTA entry's at that live count;
//!   with the producer left out, the streams and the collapsed rows must
//!   differ from the eager bits (operational errors propagate);
//! - the extreme common-offset input (comb bases at 1e8, zero comb scale).
//!
//! A bundle may select either entry by row count, so the bitwise comparison
//! also runs one row below each prefill bucket (`SCHEDULE_ROWS`): any
//! threshold among them has both of its sides compared. The timing test
//! measures both entries at those row counts (cold L2 after the eviction,
//! then the producer, then the timed mix) and records the smallest count
//! from which the one-CTA entry is faster at every larger measured count.

use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{Argument, ArgumentList, DType, LaunchPlan};
use kernel_core::oracle::{OracleOutput, check_numeric};
use kernel_core::sinkhorn_hyper_connection::SinkhornHyperConnectionQuery as Query;
use kernel_core::sinkhorn_hyper_connection_oracle::{self as oracle, MixWeights};
use kernel_lab::launch::{self, Bindings};
use kernel_lab::poison::{self, Phase, Replay, WriteMap};
use kernel_lab::{LabResult, Session};
use kernels_sm120::sinkhorn_hc::{self, Op, SERVED, split};
use runtime_cuda::{DeviceBuffer, Function, Module};
use serde_json::{Value, json};
use std::collections::BTreeMap;

const N: usize = 4;
const H: usize = 4096;
const FLAT: usize = N * H;
const MIXES: usize = (2 + N) * N;
const CELLS: usize = N * N;
const CHAINS: usize = FLAT / 128;
const BUCKETS: [usize; 12] = [1, 2, 4, 16, 32, 64, 6 * 32, 256, 1024, 2048, 4096, 8192];
/// The prefill row counts the timing test measures, at which a bundle may
/// switch between the split and the one-CTA entry.
const SCHEDULE_ROWS: [usize; 6] = [64, 256, 1024, 2048, 4096, 8192];

// Buffer handles: X1 = expand(X0, Y0, POST0, COMB0), then the one-CTA
// entries and the split entry over X1 into their own outputs.
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
const PROJECTED: u64 = 12;
const RSTD: u64 = 13;
const CELL_INPUT: u64 = 14;
const COLLAPSED: u64 = 15;
const POST: u64 = 16;
const COMB: u64 = 17;
const PARTIALS: u64 = 18;
const TICKETS: u64 = 19;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn served_cubin() -> String {
    format!("sinkhorn_hc_{}_sm120", sinkhorn_hc::SERVED_PROFILE.tag())
}

fn probe_cubin() -> String {
    format!(
        "sinkhorn_hc_{}_probe_sm120",
        sinkhorn_hc::SERVED_PROFILE.tag()
    )
}

fn split_cubin() -> String {
    format!(
        "sinkhorn_hc_{}_split_sm120",
        sinkhorn_hc::SERVED_PROFILE.tag()
    )
}

fn entry(op: Op) -> sinkhorn_hc::Entry {
    sinkhorn_hc::lab_catalog()
        .into_iter()
        .find(|entry| entry.op == op)
        .expect("a catalog entry")
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
}

/// An entry's launch: its plan at the bucket and its function.
struct Kernel {
    plan: LaunchPlan,
    function: Function,
    pack: Box<dyn Fn(Vec<Argument>) -> LabResult<ArgumentList>>,
}

struct Fixture {
    rows: usize,
    buffers: BTreeMap<u64, DeviceBuffer>,
    _modules: [Module; 3],
    expand: Kernel,
    one_cta: Kernel,
    probe: Kernel,
    split: Kernel,
}

impl Fixture {
    fn new(session: &Session, rows: usize, inputs: &Inputs) -> LabResult<Self> {
        let served = session.module(&served_cubin())?;
        let probe = session.module(&probe_cubin())?;
        let split_module = session.module(&split_cubin())?;
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
            (LIVE, 4),
            (REF_COLLAPSED, rows * H * 2),
            (REF_POST, rows * N * 2),
            (REF_COMB, rows * CELLS * 2),
            (PROJECTED, rows * MIXES * 4),
            (RSTD, rows * 4),
            (CELL_INPUT, rows * CELLS * 4),
            (COLLAPSED, rows * H * 2),
            (POST, rows * N * 2),
            (COMB, rows * CELLS * 2),
            (
                PARTIALS,
                split::partials_bytes(SERVED.mix, rows as u32) as usize,
            ),
        ] {
            buffers.insert(handle, session.allocate(bytes)?);
        }
        let fixture = Self {
            rows,
            buffers,
            expand: one(&served, Op::Expand)?,
            one_cta: one(&served, Op::MixCollapse { probe: false })?,
            probe: one(&probe, Op::MixCollapse { probe: true })?,
            split: split_kernel,
            _modules: [served, probe, split_module],
        };
        fixture.set_live(session, rows)?;
        Ok(fixture)
    }

    fn set_live(&self, session: &Session, live: usize) -> LabResult<()> {
        poison::write(
            session,
            &self.buffers[&LIVE],
            0,
            &(live as u32).to_ne_bytes(),
        )
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

    /// The expand that writes the streams the mixes read.
    fn producer(&self, session: &Session) -> LabResult<()> {
        let handles = [X0, Y0, POST0, COMB0, X1, LIVE];
        self.launch(session, &self.expand, &handles, false)
    }

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
                PROJECTED,
                RSTD,
                CELL_INPUT,
                LIVE,
            ];
            self.launch(session, &self.probe, &handles, false)
        } else {
            let handles = [X1, FN, BASE, SCALE, REF_COLLAPSED, REF_POST, REF_COMB, LIVE];
            self.launch(session, &self.one_cta, &handles, false)
        }
    }

    fn split(&self, session: &Session, pdl: bool) -> LabResult<()> {
        let handles = [
            X1, FN, BASE, SCALE, COLLAPSED, POST, COMB, PARTIALS, TICKETS, LIVE,
        ];
        self.launch(session, &self.split, &handles, pdl)
    }

    fn bf16(&self, session: &Session, handle: u64, per_row: usize) -> LabResult<Vec<u16>> {
        session.download_bf16(&self.buffers[&handle], self.rows * per_row)
    }

    fn f32(&self, session: &Session, handle: u64, per_row: usize) -> LabResult<Vec<f32>> {
        session.download_f32(&self.buffers[&handle], self.rows * per_row)
    }

    /// The split's outputs: collapsed, post, comb.
    fn outputs(&self, session: &Session, handles: [u64; 3]) -> LabResult<[Vec<u16>; 3]> {
        Ok([
            self.bf16(session, handles[0], H)?,
            self.bf16(session, handles[1], N)?,
            self.bf16(session, handles[2], CELLS)?,
        ])
    }

    fn tickets(&self, session: &Session) -> LabResult<Vec<u8>> {
        poison::read(session, &self.buffers[&TICKETS])
    }
}

fn exact<T: PartialEq + Copy + std::fmt::LowerHex + Default>(
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
    for ((name, actual), expected) in ["collapsed", "post", "comb"]
        .into_iter()
        .zip(actual)
        .zip(expected)
    {
        exact(&format!("{label} {name}"), actual, expected)?;
    }
    Ok(())
}

/// Each mix's chain sums of every row added from zero in K order, as the
/// finishing CTA and the one-CTA accumulators add them.
fn projection_from_chains(partials: &[f32]) -> Vec<f32> {
    partials
        .chunks_exact(CHAINS)
        .map(|chains| chains.iter().fold(0.0_f32, |acc, &v| acc + v))
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

/// Bitwise, oracle and determinism checks of one bucket; its receipt row.
fn bitwise_bucket(session: &Session, rows: usize, inputs: &Inputs) -> LabResult<Value> {
    let fixture = Fixture::new(session, rows, inputs)?;
    fixture.producer(session)?;
    fixture.one_cta(session, true)?;
    session.stream.synchronize().map_err(err)?;
    let probe = fixture.outputs(session, [REF_COLLAPSED, REF_POST, REF_COMB])?;
    let projected = fixture.f32(session, PROJECTED, MIXES)?;
    let refs = [REF_COLLAPSED, REF_POST, REF_COMB].map(|h| &fixture.buffers[&h]);
    poison::poison(session, &refs)?;
    fixture.one_cta(session, false)?;
    session.stream.synchronize().map_err(err)?;
    let one_cta = fixture.outputs(session, [REF_COLLAPSED, REF_POST, REF_COMB])?;
    same_outputs("one-CTA served against its probe", &one_cta, &probe)?;

    // The split into poisoned outputs, which alone fail the comparison.
    let outputs = [COLLAPSED, POST, COMB, PARTIALS].map(|h| &fixture.buffers[&h]);
    poison::poison(session, &outputs)?;
    let unwritten = fixture.outputs(session, [COLLAPSED, POST, COMB])?;
    if same_outputs("unwritten", &unwritten, &one_cta).is_ok() {
        return Err("unwritten split outputs matched the one-CTA entry".into());
    }
    fixture.split(session, false)?;
    session.stream.synchronize().map_err(err)?;
    let split = fixture.outputs(session, [COLLAPSED, POST, COMB])?;
    same_outputs("split against the one-CTA entry", &split, &one_cta)?;
    let partials = fixture.f32(session, PARTIALS, MIXES * CHAINS)?;
    let summed = projection_from_chains(&partials);
    exact(
        "chain sums in K order against the probe's projection",
        &summed.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        &projected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
    )?;
    let zeroed = vec![0_u8; rows * 4];
    exact(
        "tickets after a launch",
        &fixture.tickets(session)?,
        &zeroed,
    )?;

    // The 128-K accumulation gate on the split's own chain sums.
    let x = fixture
        .bf16(session, X1, FLAT)?
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
    let summed64 = summed.iter().map(|&v| f64::from(v)).collect::<Vec<_>>();
    check_numeric(&summed64, &reference, DType::F32).map_err(|f| {
        format!(
            "split projection (128-K accumulation): element {} is {}, reference {}, error {} over bound {}",
            f.index, f.actual, f.reference, f.error, f.bound
        )
    })?;

    // Determinism: relaunches into poisoned outputs give the same bits.
    let first_partials = partials.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    for relaunch in 0..4 {
        poison::poison(session, &outputs)?;
        fixture.split(session, relaunch % 2 == 1)?;
        session.stream.synchronize().map_err(err)?;
        let again = fixture.outputs(session, [COLLAPSED, POST, COMB])?;
        same_outputs(&format!("relaunch {relaunch}"), &again, &one_cta)?;
        let partials = fixture.f32(session, PARTIALS, MIXES * CHAINS)?;
        exact(
            &format!("relaunch {relaunch} chain sums"),
            &partials.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            &first_partials,
        )?;
        exact(
            "tickets after a relaunch",
            &fixture.tickets(session)?,
            &zeroed,
        )?;
    }
    Ok(json!({
        "rows": rows,
        "grid": fixture.split.plan.grid,
        "bitwise_one_cta": true,
        "projection_bitwise_probe": true,
        "projection_worst_over_bound": worst_ratio(&summed64, &reference),
        "relaunches_bitwise": 4,
    }))
}

/// The split's live rows against the one-CTA entry over the same streams
/// at the same live count: collapsed, post and comb bit for bit, and the
/// tickets left zero.
fn against_one_cta(
    fixture: &Fixture,
    session: &Session,
    live: usize,
    label: &str,
) -> LabResult<()> {
    session.stream.synchronize().map_err(err)?;
    let refs = [REF_COLLAPSED, REF_POST, REF_COMB].map(|h| &fixture.buffers[&h]);
    poison::poison(session, &refs)?;
    fixture.one_cta(session, false)?;
    session.stream.synchronize().map_err(err)?;
    let take = |handles: [u64; 3]| -> LabResult<[Vec<u16>; 3]> {
        let mut outputs = fixture.outputs(session, handles)?;
        for (output, per_row) in outputs.iter_mut().zip([H, N, CELLS]) {
            output.truncate(live * per_row);
        }
        Ok(outputs)
    };
    same_outputs(
        &format!("{label}: split against the one-CTA entry"),
        &take([COLLAPSED, POST, COMB])?,
        &take([REF_COLLAPSED, REF_POST, REF_COMB])?,
    )?;
    exact(
        &format!("{label}: tickets"),
        &fixture.tickets(session)?,
        &vec![0; fixture.rows * 4],
    )
}

/// The split on poisoned buffers at every live count of `rows`, after its
/// producer (an expand) whose streams are poisoned and checked too. Each
/// live count's eager outputs are also compared with the one-CTA entry,
/// and at one partial live count the producer left out must leave the
/// streams poisoned and change the collapsed rows.
fn replay_bucket(session: &Session, rows: usize, inputs: &Inputs) -> LabResult<Value> {
    let fixture = Fixture::new(session, rows, inputs)?;
    let graph = session.capture(|_| {
        fixture.producer(session)?;
        fixture.split(session, true)
    })?;
    let mut evidence = Vec::new();
    for live in poison::live_counts(rows as u32) {
        let live = live as usize;
        let label = format!("{rows} rows, live {live}");
        fixture.set_live(session, live)?;
        let buffer = |handle| &fixture.buffers[&handle];
        let partial_row = split::partials_bytes(SERVED.mix, 1) as usize;
        let replay_of = || -> LabResult<Replay<'_>> {
            Ok(Replay::new(session)
                .output(
                    "x1",
                    buffer(X1),
                    WriteMap::of(DType::Bf16).prefix(live * FLAT)?,
                )
                .output(
                    "collapsed",
                    buffer(COLLAPSED),
                    WriteMap::of(DType::Bf16).prefix(live * H)?,
                )
                .output(
                    "post",
                    buffer(POST),
                    WriteMap::of(DType::Bf16).prefix(live * N)?,
                )
                .output(
                    "comb",
                    buffer(COMB),
                    WriteMap::of(DType::Bf16).prefix(live * CELLS)?,
                )
                .output(
                    "partials",
                    buffer(PARTIALS),
                    WriteMap::of(DType::F32).byte_prefix(live * partial_row)?,
                )
                .state(
                    "tickets",
                    buffer(TICKETS),
                    vec![0; rows * 4],
                    WriteMap::of(DType::I32).none()?,
                    Vec::new(),
                ))
        };
        let both = || -> LabResult<()> {
            fixture.producer(session)?;
            fixture.split(session, false)
        };
        let mut replay = replay_of()?;
        replay.eager(both)?;
        // The eager bits against the one-CTA entry, independently of the
        // replays' self-comparison.
        against_one_cta(&fixture, session, live, &label)?;
        replay.relaunch(both)?;
        replay.graph(&graph)?;
        replay.pdl(
            || fixture.producer(session),
            || fixture.split(session, true),
            true,
        )?;
        let mut receipt = BTreeMap::new();
        replay.record(&mut receipt);
        // Without the producer (at one partial live count, all rows of one
        // row) the streams keep their poison and the split's collapsed rows
        // differ from the eager bits. Launch, synchronization and readback
        // errors propagate rather than count as caught.
        let mut mutations = Vec::new();
        if live == (rows / 2).max(1) {
            replay.prepare(Phase::Pdl)?;
            fixture.split(session, true)?;
            session.stream.synchronize().map_err(err)?;
            let mut differing = Vec::new();
            for (output, handle) in [("x1", X1), ("collapsed", COLLAPSED)] {
                let actual = poison::read(session, buffer(handle))?;
                let eager = replay
                    .expected(output)
                    .ok_or_else(|| format!("{label}: no eager {output}"))?;
                if actual != eager {
                    differing.push(output);
                }
            }
            if differing != ["x1", "collapsed"] {
                return Err(format!(
                    "{label}: without the producer only {differing:?} differ from the eager bits"
                ));
            }
            mutations.push(json!({"mutation": "no producer", "outputs": differing}));
        }
        evidence.push(json!({"live": live, "replay": receipt, "mutations_failed": mutations}));
    }
    Ok(json!({"rows": rows, "live_counts": evidence}))
}

#[test]
fn sinkhorn_hc_split_cpu_contract() {
    let entry = split::served();
    assert!(
        kernel_lab::cubin_entries(&split_cubin())
            .unwrap()
            .contains(&entry.symbol.as_str()),
        "{} lacks {}",
        split_cubin(),
        entry.symbol
    );
    kernel_lab::cubin(&split_cubin()).unwrap();
    for rows in BUCKETS {
        let plan = entry.plan(rows as u32, rows as u32).unwrap();
        assert_eq!(plan.grid, [16, rows as u32, 1]);
        assert!(plan.pdl);
    }
    // Chain sums add from zero in K order, per mix.
    let mut partials = vec![0.0_f32; 2 * CHAINS];
    partials[0] = 1.0;
    partials[1] = 1e-8;
    partials[CHAINS] = 3.0;
    assert_eq!(projection_from_chains(&partials), [1.0, 3.0]);
    let unwritten = [vec![0xffff_u16; H], vec![0xffff; N], vec![0xffff; CELLS]];
    let written = [vec![0x3f80_u16; H], vec![0x3f00; N], vec![0x3e80; CELLS]];
    assert!(same_outputs("unwritten", &unwritten, &written).is_err());
    same_outputs("written", &written, &written).unwrap();
    // The crossover counts only a suffix of one-CTA wins.
    let timed = |medians: &[(f64, f64)]| -> Vec<Value> {
        SCHEDULE_ROWS
            .iter()
            .zip(medians)
            .map(|(rows, (one, split))| {
                json!({"rows": rows, "one_cta_median_ms": one, "split_median_ms": split})
            })
            .collect()
    };
    let wins = [(1.0, 2.0); 6];
    assert_eq!(one_cta_from(&timed(&wins)), Some(64));
    assert_eq!(one_cta_from(&timed(&[(2.0, 1.0); 6])), None);
    let mixed = [
        (2.0, 1.0),
        (1.0, 2.0),
        (2.0, 1.0),
        (1.0, 2.0),
        (1.0, 2.0),
        (1.0, 2.0),
    ];
    assert_eq!(one_cta_from(&timed(&mixed)), Some(2048));
    let inputs = Inputs::new(1, 1);
    assert_eq!(inputs.x0.len(), FLAT);
    assert_eq!(inputs.weights.len(), MIXES * FLAT);
}

#[test]
fn sinkhorn_hc_split_matches_the_one_cta_entry_bitwise() {
    kernel_lab::gpu_test(3 << 30, |session| {
        let mut rows = BUCKETS
            .iter()
            .copied()
            .chain(SCHEDULE_ROWS.iter().map(|&rows| rows - 1))
            .map(|rows| bitwise_bucket(session, rows, &Inputs::new(rows, 0x5eed + rows as u64)))
            .collect::<LabResult<Vec<_>>>()?;
        let mut extreme = bitwise_bucket(session, 32, &Inputs::extreme(32, 0xe0))?;
        extreme["extreme_common_offset"] = json!(1e8);
        rows.push(extreme);
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("sinkhorn_hc_split_bitwise"));
        receipt.insert("buckets".into(), json!(rows));
        session.receipt("sinkhorn_hc_split_bitwise", receipt)?;
        Ok(())
    })
    .expect("split Sinkhorn mix-collapse bitwise comparison");
}

#[test]
fn sinkhorn_hc_split_replays_on_poisoned_buffers() {
    kernel_lab::gpu_test(3 << 30, |session| {
        let mut rows = BUCKETS
            .iter()
            .map(|&rows| replay_bucket(session, rows, &Inputs::new(rows, 0xfeed + rows as u64)))
            .collect::<LabResult<Vec<_>>>()?;
        let mut extreme = replay_bucket(session, 32, &Inputs::extreme(32, 0xfe))?;
        extreme["extreme_common_offset"] = json!(1e8);
        rows.push(extreme);
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("sinkhorn_hc_split_poison"));
        receipt.insert("buckets".into(), json!(rows));
        session.receipt("sinkhorn_hc_split_poison", receipt)?;
        Ok(())
    })
    .expect("split Sinkhorn mix-collapse poisoned replays");
}

/// The split and the one-CTA entry timed at one row count: each launch
/// after an L2 eviction and the producer that writes its streams, as a
/// prefill mixer follows its sublayer's expand; then both outputs compared.
fn timed_bucket(session: &Session, rows: usize, inputs: &Inputs) -> LabResult<Value> {
    const WARMUP: usize = 2;
    const ROUNDS: usize = 15;
    let fixture = Fixture::new(session, rows, inputs)?;
    let mode = kernel_lab::eviction::EvictionMode::selected()?;
    let time = |label: &str, split: bool| {
        kernel_lab::with_timing_label(&format!("{label} {rows} rows"), || {
            kernel_lab::time_prepared(
                session,
                WARMUP,
                ROUNDS,
                mode,
                |_| fixture.producer(session),
                |_| {
                    if split {
                        fixture.split(session, false)
                    } else {
                        fixture.one_cta(session, false)
                    }
                },
            )
        })
    };
    let one_cta = time("one-CTA mix-collapse", false)?;
    let split = time("split mix-collapse", true)?;
    session.stream.synchronize().map_err(err)?;
    same_outputs(
        &format!("{rows} rows: timed split against the timed one-CTA entry"),
        &fixture.outputs(session, [COLLAPSED, POST, COMB])?,
        &fixture.outputs(session, [REF_COLLAPSED, REF_POST, REF_COMB])?,
    )?;
    Ok(json!({
        "rows": rows,
        "one_cta_median_ms": one_cta.median_ms,
        "one_cta_min_ms": one_cta.min_ms,
        "split_median_ms": split.median_ms,
        "split_min_ms": split.min_ms,
        "split_over_one_cta": split.median_ms / one_cta.median_ms,
        "rounds": ROUNDS,
    }))
}

/// The smallest measured row count from which the one-CTA entry's median
/// is below the split's at that and every larger measured count.
fn one_cta_from(buckets: &[Value]) -> Option<u64> {
    let faster =
        |bucket: &Value| bucket["one_cta_median_ms"].as_f64() < bucket["split_median_ms"].as_f64();
    let split_wins = buckets.iter().rposition(|bucket| !faster(bucket));
    let first = split_wins.map_or(0, |at| at + 1);
    buckets
        .get(first)
        .and_then(|bucket| bucket["rows"].as_u64())
}

#[test]
fn sinkhorn_hc_split_and_one_cta_times_by_rows() {
    kernel_lab::gpu_test(3 << 30, |session| {
        let buckets = SCHEDULE_ROWS
            .iter()
            .map(|&rows| timed_bucket(session, rows, &Inputs::new(rows, 0x7135 + rows as u64)))
            .collect::<LabResult<Vec<_>>>()?;
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("sinkhorn_hc_split_timing"));
        receipt.insert(
            "eviction".into(),
            json!(kernel_lab::eviction::EvictionMode::selected()?.name()),
        );
        receipt.insert("one_cta_from_rows".into(), json!(one_cta_from(&buckets)));
        receipt.insert("buckets".into(), json!(buckets));
        session.receipt("sinkhorn_hc_split_timing", receipt)?;
        Ok(())
    })
    .expect("split and one-CTA Sinkhorn mix-collapse timing");
}
