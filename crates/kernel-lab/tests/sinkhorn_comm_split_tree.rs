//! The split Sinkhorn mHC comm epilogue with the input norm's tree sum
//! (`kernels_sm120::comm::sinkhorn_split_tree`) at the served 4 x 4096
//! mixer, against the ordered epilogue it follows statement for statement
//! (`sinkhorn_epilogue_split_hc4x4096_sm120`) and the row engine's tree norm
//! (`row::rmsnorm_tree`), on one GPU over the rows transport (F32 sums
//! written by the host, as another all-reduce writes them). Run with
//! `CHOPINFER_KERNEL_GPU=<ordinal> cargo test -p kernel-lab --release --test
//! sinkhorn_comm_split_tree -- --test-threads=1`; without a GPU only the CPU
//! contract test runs.
//!
//! - At decode buckets (1, 2, 4, 16, 32), verify rows (6 x 32) and prefill
//!   chunks (2048, 8192), from the same sums: y, X', the next post and comb,
//!   the collapsed row and the chain sums equal the ordered epilogue's bit
//!   for bit; the normed row equals the row engine's tree norm of the
//!   ordered epilogue's collapsed row and, on rows of finite collapsed
//!   values, the CPU emulation of the tree norm; the tickets are left zero.
//! - Poisoned replays (eager, relaunch, one graph captured at capacity 32,
//!   and a PDL launch behind the lab's live-count producer, which zeroes the
//!   live count and the sums until its tail) at every live count 0 to 32:
//!   outputs past the live count keep their poison, post and comb (read,
//!   then overwritten) keep their image there, and each live count's eager
//!   outputs equal the full bucket's reference rows.
//! - Mutants, each caught only on observed differences: the norm tree's warp
//!   distances reversed (the normed rows alone must differ, on collapsed
//!   rows that are the rows of `common/tree_mutant_rows.rs`, which the CPU
//!   contract test proves the two orders tell apart; the served rows must
//!   equal the tree's emulation and the mutant's the reversed order's, and
//!   every row must differ), and the
//!   dependency wait after the reads of the sums and streams behind the
//!   live-count producer, reported inconclusive when it reproduces the eager
//!   bits.

use kernel_core::codec::f32_to_bf16;
use kernel_core::contract::{ArgumentList, DType, LaunchPlan};
use kernel_core::sinkhorn_hyper_connection::SinkhornHyperConnectionQuery as Query;
use kernel_core::sinkhorn_hyper_connection_oracle as oracle;
use kernel_lab::launch::{self, Bindings};
use kernel_lab::poison::{self, LiveProducer, Replay, Rewrites, WriteMap};
use kernel_lab::{LabResult, Session};
use kernels_sm120::comm::sinkhorn_epilogue::EpilogueBuffers;
use kernels_sm120::comm::sinkhorn_split::{self, SplitBuffers};
use kernels_sm120::comm::sinkhorn_split_tree;
use kernels_sm120::row::rmsnorm::RmsNormBuffers;
use kernels_sm120::row::rmsnorm_tree::{self, hf_norm, tree_sum};
use kernels_sm120::sinkhorn_hc::{self, SERVED};
use runtime_cuda::{Function, Module};
use serde_json::{Value, json};
use std::collections::BTreeMap;

#[path = "common/tree_mutant_rows.rs"]
mod tree_mutant_rows;

const N: usize = 4;
const H: usize = 4096;
const FLAT: usize = N * H;
const MIXES: usize = (2 + N) * N;
const CELLS: usize = N * N;
const EPS: f32 = 1e-5;
/// Decode buckets, verify rows (6 x 32) and prefill chunks.
const ROWS_BUCKETS: [usize; 8] = [1, 2, 4, 16, 32, 6 * 32, 2048, 8192];
const FULL_REPLAY_ROWS: usize = 32;

const ORDERED_CUBIN: &str = "sinkhorn_epilogue_split_hc4x4096_sm120";

// Buffer handles.
const SUMMED: u64 = 0;
const STREAMS: u64 = 1;
const FN: u64 = 2;
const BASE: u64 = 3;
const SCALE: u64 = 4;
const WEIGHT: u64 = 5;
const LIVE: u64 = 6;
const SUMMED_IMAGE: u64 = 7;
// The ordered epilogue's (reference) and the tree epilogue's buffers: post,
// comb, y, X', collapsed, normed, chain sums, tickets.
const REF: u64 = 10;
const TREE: u64 = 20;
const POST: u64 = 0;
const COMB: u64 = 1;
const REDUCED: u64 = 2;
const EXPANDED: u64 = 3;
const COLLAPSED: u64 = 4;
const NORMED: u64 = 5;
const PARTIALS: u64 = 6;
const TICKETS: u64 = 7;
/// The row engine's tree norm of the reference collapsed rows.
const ROW_NORMED: u64 = 30;
/// The BF16 outputs by offset, their names and elements per row.
const OUTPUTS: [(u64, &str, usize); 6] = [
    (POST, "post", N),
    (COMB, "comb", CELLS),
    (REDUCED, "y", H),
    (EXPANDED, "x'", FLAT),
    (COLLAPSED, "collapsed", H),
    (NORMED, "normed", H),
];

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn tree_cubin(suffix: &str) -> String {
    format!(
        "sinkhorn_epilogue_split_tree{suffix}_{}_sm120",
        sinkhorn_hc::SERVED_PROFILE.tag()
    )
}

/// xorshift64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// BF16 bits of random sign and mantissa over the binades
    /// `low..=high`, with zeros, negative zeros and a subnormal among them.
    fn bf16(&mut self, len: usize, low: i32, high: i32) -> Vec<u16> {
        (0..len)
            .map(|i| {
                let state = self.next();
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
    fn unit(&mut self, len: usize, scale: f32) -> Vec<u16> {
        (0..len)
            .map(|_| f32_to_bf16(((self.next() >> 40) % 4096) as f32 / 4096.0 * scale))
            .collect()
    }

    /// F32 sums of order one with full mantissas, zeros and negative zeros.
    fn sums(&mut self, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| match i % 701 {
                0 => 0.0,
                1 => -0.0,
                _ => {
                    let state = self.next();
                    let unit = (state >> 40) as f32 / (1u64 << 24) as f32;
                    let sign = if state & 1 == 0 { 1.0 } else { -1.0 };
                    sign * unit * 2f32.powi(((state >> 8) % 5) as i32 - 1)
                }
            })
            .collect()
    }
}

/// One bucket's operands.
struct Inputs {
    summed: Vec<f32>,
    streams: Vec<u16>,
    post: Vec<u16>,
    comb: Vec<u16>,
    weights: Vec<u16>,
    base: Vec<f32>,
    scale: [f32; 3],
    weight: Vec<u16>,
}

impl Inputs {
    fn new(rows: usize, seed: u64) -> Self {
        let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
        let x0 = rng.bf16(rows * FLAT, -4, 2);
        let y0 = rng.bf16(rows * H, -6, 1);
        let post0 = rng.unit(rows * N, 2.0);
        let comb0 = rng.unit(rows * CELLS, 1.0);
        let expand = Query::Expand {
            streams: N as u32,
            width: H as u32,
            capacity: rows as u32,
        };
        let streams = oracle::expand(expand, &x0, &y0, &post0, &comb0).unwrap();
        let base = (0..MIXES)
            .map(|_| ((rng.next() >> 40) % 2048) as f32 / 1024.0 - 1.0)
            .collect();
        Self {
            summed: rng.sums(rows * H),
            streams,
            post: rng.unit(rows * N, 2.0),
            comb: rng.unit(rows * CELLS, 1.0),
            // Logits of order one at K = 16,384.
            weights: rng.bf16(MIXES * FLAT, -10, -6),
            base,
            scale: [0.5, 0.75, 2.0],
            weight: rng.unit(H, 2.0),
        }
    }
}

impl Inputs {
    /// Collapsed rows that are `tree_mutant_rows::rows`: the sums are those
    /// rows (so y is), the streams +0, post one for stream 0 and zero for
    /// the others (so X' is y in stream 0 and +0 in the others), and every
    /// pre weight sigmoid(20) + eps (zero pre scale, so the logits do not
    /// enter), within 2^-19 of one, so each collapsed value rounds back to
    /// its stream value. The norm weight is one.
    fn tree_mutant(rows: usize, seed: u64) -> Self {
        let one = f32_to_bf16(1.0);
        let mut inputs = Self {
            summed: tree_mutant_rows::rows(rows)
                .into_iter()
                .map(kernel_core::codec::bf16_to_f32)
                .collect(),
            streams: vec![0; rows * FLAT],
            post: (0..rows * N)
                .map(|i| if i % N == 0 { one } else { 0 })
                .collect(),
            weight: tree_mutant_rows::weight(),
            ..Self::new(rows, seed)
        };
        inputs.base[..N].fill(20.0);
        inputs.scale[0] = 0.0;
        inputs
    }
}

fn halves(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

struct Kernel {
    plan: LaunchPlan,
    function: Function,
}

struct Fixture {
    rows: usize,
    inputs: Inputs,
    buffers: BTreeMap<u64, runtime_cuda::DeviceBuffer>,
    _modules: Vec<Module>,
    ordered: Kernel,
    tree: Kernel,
    norm: Kernel,
    mutants: BTreeMap<&'static str, Kernel>,
}

impl Fixture {
    fn new(session: &Session, rows: usize, seed: u64) -> LabResult<Self> {
        Self::with_inputs(session, rows, Inputs::new(rows, seed))
    }

    fn with_inputs(session: &Session, rows: usize, inputs: Inputs) -> LabResult<Self> {
        let ordered_module = session.module(ORDERED_CUBIN)?;
        let tree_module = session.module(&tree_cubin(""))?;
        let norm_module = session.module(rmsnorm_tree::lab_cubin_specs()[0].name.as_str())?;
        let split = sinkhorn_split::catalog();
        let tree = sinkhorn_split_tree::catalog();
        let plan = tree.plan(rows as u32, 1).map_err(err)?;
        let (norm, norm_symbol) = rmsnorm_tree::lab_norm(H as u32, "");
        let mut modules = Vec::new();
        let mut mutants = BTreeMap::new();
        for (suffix, _) in sinkhorn_split_tree::MUTANTS {
            let module = session.module(&tree_cubin(suffix))?;
            let symbol = sinkhorn_split_tree::mutant(suffix).symbol;
            mutants.insert(
                suffix,
                Kernel {
                    plan,
                    function: module.function(&symbol).map_err(err)?,
                },
            );
            modules.push(module);
        }
        let mut buffers = BTreeMap::new();
        buffers.insert(SUMMED, session.upload_f32(&inputs.summed)?);
        buffers.insert(SUMMED_IMAGE, session.upload_f32(&inputs.summed)?);
        buffers.insert(STREAMS, session.upload_bf16(&inputs.streams)?);
        buffers.insert(FN, session.upload_bf16(&inputs.weights)?);
        buffers.insert(BASE, session.upload_f32(&inputs.base)?);
        buffers.insert(SCALE, session.upload_f32(&inputs.scale)?);
        buffers.insert(WEIGHT, session.upload_bf16(&inputs.weight)?);
        buffers.insert(LIVE, session.allocate(4)?);
        let partials = sinkhorn_hc::split::partials_bytes(SERVED.mix, rows as u32) as usize;
        for side in [REF, TREE] {
            buffers.insert(side + POST, session.upload_bf16(&inputs.post)?);
            buffers.insert(side + COMB, session.upload_bf16(&inputs.comb)?);
            for (offset, _, per_row) in &OUTPUTS[2..] {
                buffers.insert(side + offset, session.allocate(rows * per_row * 2)?);
            }
            buffers.insert(side + PARTIALS, session.allocate(partials)?);
            buffers.insert(side + TICKETS, session.upload_f32(&vec![0.0; rows])?);
        }
        buffers.insert(ROW_NORMED, session.allocate(rows * H * 2)?);
        let fixture = Self {
            rows,
            inputs,
            buffers,
            ordered: Kernel {
                plan,
                function: ordered_module.function(&split.symbol).map_err(err)?,
            },
            tree: Kernel {
                plan,
                function: tree_module.function(&tree.symbol).map_err(err)?,
            },
            norm: Kernel {
                plan: LaunchPlan {
                    grid: [rows as u32, 1, 1],
                    ..norm.plan(0).map_err(err)?
                },
                function: norm_module.function(&norm_symbol).map_err(err)?,
            },
            mutants,
            _modules: {
                modules.extend([ordered_module, tree_module, norm_module]);
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

    /// Restores `side`'s post and comb, which an epilogue reads and then
    /// overwrites.
    fn restore(&self, session: &Session, side: u64) -> LabResult<()> {
        poison::write(
            session,
            self.buffer(side + POST),
            0,
            &halves(&self.inputs.post),
        )?;
        poison::write(
            session,
            self.buffer(side + COMB),
            0,
            &halves(&self.inputs.comb),
        )
    }

    fn launch(
        &self,
        session: &Session,
        kernel: &Kernel,
        args: &ArgumentList,
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
            args,
            &bindings,
        )
    }

    fn arguments(side: u64) -> LabResult<ArgumentList> {
        sinkhorn_split::rows_arguments(
            SUMMED,
            SplitBuffers {
                epilogue: EpilogueBuffers {
                    streams: STREAMS,
                    post: side + POST,
                    comb: side + COMB,
                    reduced: side + REDUCED,
                    expanded: side + EXPANDED,
                    fn_: FN,
                    base: BASE,
                    scale: SCALE,
                    collapsed: side + COLLAPSED,
                    weight: WEIGHT,
                    normed: side + NORMED,
                    live_rows: LIVE,
                    eps_bits: EPS.to_bits(),
                },
                partials: side + PARTIALS,
                tickets: side + TICKETS,
            },
        )
        .map_err(err)
    }

    /// The ordered epilogue, then the row engine's tree norm of its
    /// collapsed rows, from restored post and comb.
    fn reference(&self, session: &Session) -> LabResult<()> {
        self.restore(session, REF)?;
        self.launch(session, &self.ordered, &Self::arguments(REF)?, false)?;
        let (norm, _) = rmsnorm_tree::lab_norm(H as u32, "");
        let args = norm
            .arguments(RmsNormBuffers {
                x: REF + COLLAPSED,
                weight: WEIGHT,
                residual: None,
                y: ROW_NORMED,
                live_rows: LIVE,
            })
            .map_err(err)?;
        self.launch(session, &self.norm, &args, false)
    }

    /// The tree epilogue (no restore; replays restore its state).
    fn tree(&self, session: &Session, pdl: bool) -> LabResult<()> {
        self.launch(session, &self.tree, &Self::arguments(TREE)?, pdl)
    }

    fn mutant(&self, session: &Session, suffix: &str, pdl: bool) -> LabResult<()> {
        let kernel = self
            .mutants
            .get(suffix)
            .ok_or_else(|| format!("no mutant {suffix}"))?;
        self.launch(session, kernel, &Self::arguments(TREE)?, pdl)
    }

    /// `side`'s BF16 outputs, in [`OUTPUTS`] order.
    fn outputs(&self, session: &Session, side: u64) -> LabResult<Vec<Vec<u16>>> {
        OUTPUTS
            .iter()
            .map(|(offset, _, per_row)| {
                session.download_bf16(self.buffer(side + offset), self.rows * per_row)
            })
            .collect()
    }

    fn partials(&self, session: &Session, side: u64) -> LabResult<Vec<u8>> {
        poison::read(session, self.buffer(side + PARTIALS))
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

/// The reference outputs with the ordered epilogue's normed rows replaced by
/// the row engine's tree norm, and the reference chain sums.
fn reference_outputs(fixture: &Fixture, session: &Session) -> LabResult<(Vec<Vec<u16>>, Vec<u8>)> {
    fixture.reference(session)?;
    session.stream.synchronize().map_err(err)?;
    let mut outputs = fixture.outputs(session, REF)?;
    outputs[5] = session.download_bf16(fixture.buffer(ROW_NORMED), fixture.rows * H)?;
    Ok((outputs, fixture.partials(session, REF)?))
}

/// The names of the outputs whose first `live` rows differ.
fn differing(actual: &[Vec<u16>], expected: &[Vec<u16>], live: usize) -> Vec<&'static str> {
    OUTPUTS
        .iter()
        .zip(actual.iter().zip(expected))
        .filter(|((_, _, per_row), (a, e))| a[..live * per_row] != e[..live * per_row])
        .map(|((_, name, _), _)| *name)
        .collect()
}

/// The CPU tree norm of every collapsed row whose values are finite against
/// `normed`; returns the rows checked.
fn emulated_rows(
    collapsed: &[u16],
    normed: &[u16],
    weight: &[u16],
    rows: usize,
) -> LabResult<usize> {
    let mut checked = 0;
    for row in 0..rows {
        let x = &collapsed[row * H..(row + 1) * H];
        if x.iter().any(|&v| v & 0x7f80 == 0x7f80) {
            continue;
        }
        exact(
            &format!("row {row} against the CPU tree norm"),
            &normed[row * H..(row + 1) * H],
            &hf_norm(x, weight, EPS, tree_sum),
        )?;
        checked += 1;
    }
    Ok(checked)
}

fn rows_bucket(session: &Session, rows: usize) -> LabResult<Value> {
    let fixture = Fixture::new(session, rows, 0x5eed + rows as u64)?;
    let (reference, reference_partials) = reference_outputs(&fixture, session)?;
    fixture.restore(session, TREE)?;
    let written = OUTPUTS[2..]
        .iter()
        .map(|(offset, _, _)| fixture.buffer(TREE + offset))
        .chain([fixture.buffer(TREE + PARTIALS)])
        .collect::<Vec<_>>();
    poison::poison(session, &written)?;
    fixture.tree(session, false)?;
    session.stream.synchronize().map_err(err)?;
    let outputs = fixture.outputs(session, TREE)?;
    for (i, (_, name, _)) in OUTPUTS.iter().enumerate() {
        exact(&format!("{rows} rows: {name}"), &outputs[i], &reference[i])?;
    }
    exact(
        &format!("{rows} rows: chain sums"),
        &fixture.partials(session, TREE)?,
        &reference_partials,
    )?;
    exact(
        &format!("{rows} rows: tickets"),
        &poison::read(session, fixture.buffer(TREE + TICKETS))?,
        &vec![0_u8; rows * 4],
    )?;
    let emulated = emulated_rows(&outputs[4], &outputs[5], &fixture.inputs.weight, rows)?;
    if emulated == 0 {
        return Err(format!("{rows} rows: no finite collapsed row"));
    }
    Ok(json!({
        "rows": rows,
        "bitwise_ordered_epilogue_but_normed": true,
        "normed_bitwise_row_engine_tree_norm": true,
        "rows_bitwise_cpu_tree_emulation": emulated,
    }))
}

/// Poisoned eager, relaunch, graph and PDL launches at every live count of
/// a capacity-32 bucket.
#[allow(clippy::single_range_in_vec_init)]
fn replay_bucket(session: &Session) -> LabResult<Value> {
    let rows = FULL_REPLAY_ROWS;
    let fixture = Fixture::new(session, rows, 0xfeed)?;
    let (reference, _) = reference_outputs(&fixture, session)?;
    let graph = session.capture(|_| fixture.tree(session, false))?;
    let producer = LiveProducer::new(session)?;
    let rewrites = Rewrites::new(
        session,
        &[(fixture.buffer(SUMMED_IMAGE), fixture.buffer(SUMMED))],
    )?;
    let partial_row = sinkhorn_hc::split::partials_bytes(SERVED.mix, 1) as usize;
    let mut lives = Vec::new();
    for live in 0..=rows {
        fixture.set_live(session, live)?;
        let mut replay = Replay::new(session);
        for (offset, name, per_row) in &OUTPUTS[2..] {
            replay = replay.output(
                *name,
                fixture.buffer(TREE + offset),
                WriteMap::of(DType::Bf16).prefix(live * per_row)?,
            );
        }
        replay = replay.output(
            "chain sums",
            fixture.buffer(TREE + PARTIALS),
            WriteMap::of(DType::F32).byte_prefix(live * partial_row)?,
        );
        for (offset, name, per_row) in &OUTPUTS[..2] {
            let image = if *offset == POST {
                halves(&fixture.inputs.post)
            } else {
                halves(&fixture.inputs.comb)
            };
            replay = replay.state(
                *name,
                fixture.buffer(TREE + offset),
                image,
                WriteMap::of(DType::Bf16).none()?,
                vec![0..live * per_row * 2],
            );
        }
        replay = replay.state(
            "tickets",
            fixture.buffer(TREE + TICKETS),
            vec![0; rows * 4],
            WriteMap::of(DType::I32).none()?,
            Vec::new(),
        );
        let launch = || fixture.tree(session, false);
        replay.eager(launch)?;
        let eager = fixture.outputs(session, TREE)?;
        let changed = differing(&eager, &reference, live);
        if !changed.is_empty() {
            return Err(format!(
                "live {live}: {changed:?} differ from the reference rows"
            ));
        }
        replay.relaunch(launch)?;
        replay.graph(&graph)?;
        replay.pdl(
            || producer.launch_rewriting(session, fixture.buffer(LIVE), &rewrites),
            || fixture.tree(session, true),
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
    Ok(json!({"rows": rows, "live_counts": lives}))
}

/// The reversed-distances mutant at `rows` rows on `Inputs::tree_mutant`:
/// the served epilogue's collapsed rows are the designed rows and its
/// normed rows their tree norm (CPU emulation); the mutant changes the
/// normed rows alone, to the reversed order's emulation, in every row.
/// Returns the outputs the mutant changed.
fn distances_mutant(session: &Session, rows: usize) -> LabResult<Vec<&'static str>> {
    let fixture = Fixture::with_inputs(session, rows, Inputs::tree_mutant(rows, 0xd157))?;
    let designed = tree_mutant_rows::rows(rows);
    let (tree, reversed) = tree_mutant_rows::norms(&designed, EPS);
    let outputs: Vec<_> = OUTPUTS[2..]
        .iter()
        .map(|(offset, _, _)| TREE + offset)
        .chain([TREE + PARTIALS])
        .map(|handle| fixture.buffer(handle))
        .collect();
    let run = |suffix: Option<&str>| -> LabResult<(Vec<Vec<u16>>, Vec<u8>)> {
        session.stream.synchronize().map_err(err)?;
        fixture.restore(session, TREE)?;
        poison::poison(session, &outputs)?;
        match suffix {
            Some(suffix) => fixture.mutant(session, suffix, false)?,
            None => fixture.tree(session, false)?,
        }
        session.stream.synchronize().map_err(err)?;
        Ok((
            fixture.outputs(session, TREE)?,
            fixture.partials(session, TREE)?,
        ))
    };
    let (eager, eager_partials) = run(None)?;
    exact("tree mutant rows: collapsed rows", &eager[4], &designed)?;
    exact("tree mutant rows: served normed rows", &eager[5], &tree)?;
    let (mutated, partials) = run(Some("_distances_reversed"))?;
    let mut names = differing(&mutated, &eager, rows);
    if partials != eager_partials {
        names.push("chain sums");
    }
    if names != ["normed"] {
        return Err(format!(
            "tree mutant rows: the reversed-distances mutant changed {names:?}, not the normed \
             rows alone"
        ));
    }
    exact(
        "tree mutant rows: mutant normed rows",
        &mutated[5],
        &reversed,
    )?;
    let unchanged = mutated[5]
        .chunks_exact(H)
        .zip(eager[5].chunks_exact(H))
        .filter(|(m, s)| m == s)
        .count();
    if unchanged != 0 {
        return Err(format!(
            "tree mutant rows: the reversed-distances mutant left {unchanged} rows unchanged"
        ));
    }
    Ok(names)
}

/// The mutants against the served epilogue's eager bits.
fn mutant_bucket(session: &Session) -> LabResult<Value> {
    let rows = 32;
    let fixture = Fixture::new(session, rows, 0x7a7e)?;
    fixture.restore(session, TREE)?;
    fixture.tree(session, false)?;
    session.stream.synchronize().map_err(err)?;
    let eager = fixture.outputs(session, TREE)?;
    let run = |suffix: Option<&str>, behind_producer: bool| -> LabResult<Vec<&'static str>> {
        session.stream.synchronize().map_err(err)?;
        fixture.restore(session, TREE)?;
        let producer = LiveProducer::new(session)?;
        let rewrites = Rewrites::new(
            session,
            &[(fixture.buffer(SUMMED_IMAGE), fixture.buffer(SUMMED))],
        )?;
        if behind_producer {
            producer.launch_rewriting(session, fixture.buffer(LIVE), &rewrites)?;
        }
        match suffix {
            Some(suffix) => fixture.mutant(session, suffix, behind_producer)?,
            None => fixture.tree(session, behind_producer)?,
        }
        session.stream.synchronize().map_err(err)?;
        Ok(differing(&fixture.outputs(session, TREE)?, &eager, rows))
    };
    let distances = distances_mutant(session, rows)?;
    let served = run(None, true)?;
    if !served.is_empty() {
        return Err(format!(
            "the served epilogue behind the live-count producer changed {served:?}"
        ));
    }
    let late = run(Some("_late_wait"), true)?;
    let outcome = if late.is_empty() {
        eprintln!(
            "tree epilogue, dependency wait after the sums' reads: inconclusive: the mutant \
             reproduced the eager bits"
        );
        "inconclusive"
    } else {
        "caught"
    };
    Ok(json!({
        "rows": rows,
        "mutants": [
            {"mutation": "norm tree warp distances reversed", "outcome": "caught",
             "inputs": "tree mutant rows", "outputs": distances},
            {"mutation": "dependency wait after the sums' reads", "outcome": outcome,
             "outputs": late},
        ],
    }))
}

#[test]
fn sinkhorn_comm_split_tree_cpu_contract() {
    let tree = sinkhorn_split_tree::catalog();
    assert_eq!(
        kernel_lab::cubin_entries(&tree_cubin("")).unwrap(),
        [tree.symbol.as_str()]
    );
    for (suffix, _) in sinkhorn_split_tree::MUTANTS {
        assert_eq!(
            kernel_lab::cubin_entries(&tree_cubin(suffix)).unwrap(),
            [sinkhorn_split_tree::mutant(suffix).symbol.as_str()]
        );
    }
    assert_eq!(
        kernel_lab::cubin_entries(ORDERED_CUBIN).unwrap(),
        [sinkhorn_split::catalog().symbol.as_str()]
    );
    // The row engine's tree norm runs at the epilogue's norm eps.
    let (norm, _) = rmsnorm_tree::lab_norm(H as u32, "");
    assert_eq!(norm.requirement().epsilon_bits, EPS.to_bits());
    assert_eq!(tree.rmsnorm_eps_bits, EPS.to_bits());
    let a = vec![
        vec![1_u16; 2 * N],
        vec![1; 2 * CELLS],
        vec![1; 2 * H],
        vec![1; 2 * FLAT],
        vec![1; 2 * H],
        vec![1; 2 * H],
    ];
    let mut b = a.clone();
    b[5][H] = 2;
    assert!(differing(&b, &a, 1).is_empty());
    assert_eq!(differing(&b, &a, 2), ["normed"]);
    let one = f32_to_bf16(1.0);
    let mut collapsed = vec![one; 2 * H];
    collapsed[H] = 0x7fc0;
    let weight = vec![one; H];
    let mut normed = hf_norm(&collapsed[..H], &weight, EPS, tree_sum);
    normed.extend(vec![0; H]);
    assert_eq!(emulated_rows(&collapsed, &normed, &weight, 2).unwrap(), 1);
    assert_eq!(Inputs::new(2, 1).summed.len(), 2 * H);
    // The tree mutant rows tell the two orders apart, and the tree mutant
    // inputs collapse to them: y, the expand and the collapse emulated with
    // these inputs' pre weight.
    let changed = tree_mutant_rows::proof(EPS).unwrap();
    eprintln!("tree mutant rows: outputs changed (overflow, probe row): {changed:?}");
    let rows = 2;
    let inputs = Inputs::tree_mutant(rows, 1);
    assert_eq!(inputs.scale[0], 0.0);
    let pre = 1.0 / (1.0 + (-inputs.base[0]).exp()) + f32::from_bits(SERVED.mix.eps_bits);
    assert!((pre - 1.0).abs() < 2f32.powi(-19));
    let bf = kernel_core::codec::bf16_to_f32;
    let mut collapsed = Vec::new();
    for row in 0..rows {
        let x = |i: usize, c: usize| bf(inputs.streams[row * FLAT + i * H + c]);
        let comb = |i: usize, j: usize| bf(inputs.comb[row * CELLS + i * N + j]);
        for c in 0..H {
            let y = bf(f32_to_bf16(inputs.summed[row * H + c]));
            let mut lo = 0.0_f32;
            for j in 0..N {
                let mixed = (0..N).fold(0.0_f32, |acc, i| comb(i, j).mul_add(x(i, c), acc));
                let p = bf(inputs.post[row * N + j]);
                let stream = bf(f32_to_bf16(bf(f32_to_bf16(p * y)) + bf(f32_to_bf16(mixed))));
                lo = pre.mul_add(stream, lo);
            }
            collapsed.push(f32_to_bf16(lo));
        }
    }
    assert_eq!(collapsed, tree_mutant_rows::rows(rows));
}

#[test]
fn sinkhorn_comm_split_tree_matches_the_ordered_epilogue_and_tree_norm() {
    kernel_lab::gpu_test(6 << 30, |session| {
        let rows = ROWS_BUCKETS
            .into_iter()
            .map(|rows| rows_bucket(session, rows))
            .collect::<LabResult<Vec<_>>>()?;
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("sinkhorn_comm_split_tree_bitwise"));
        receipt.insert("buckets".into(), json!(rows));
        session.receipt("sinkhorn_comm_split_tree_bitwise", receipt)?;
        Ok(())
    })
    .expect("tree comm epilogue bitwise comparison");
}

#[test]
fn sinkhorn_comm_split_tree_replays_and_mutants() {
    kernel_lab::gpu_test(2 << 30, |session| {
        let replays = replay_bucket(session)?;
        let mutants = mutant_bucket(session)?;
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("sinkhorn_comm_split_tree_poison"));
        receipt.insert("replays".into(), replays);
        receipt.insert("mutants".into(), mutants);
        session.receipt("sinkhorn_comm_split_tree_poison", receipt)?;
        Ok(())
    })
    .expect("tree comm epilogue replays and mutants");
}
