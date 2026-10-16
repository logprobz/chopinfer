//! The phased all-reduce of BF16 partials from F32 inputs
//! (`kernels-sm120/cuda/comm/allreduce_phased_bf16.cuh`, lock family
//! `comm_phased_bf16`), two-shot and one-shot entries, on two or more GPUs
//! (`CHOPINFER_KERNEL_GPUS`). The host's fold is the reference: every
//! rank's F32 partial rounded once to BF16, the BF16 values summed in rank
//! order from F32 zero, the sum rounded once to BF16.
//!
//! - `phased_bf16_matches_host_and_round_entry`: the pinned matrix
//!   (`matrix`, held by `phased_bf16_matrix_is_pinned`) over one set of
//!   operands, row r of every rank the same values in every case, so every
//!   case's row r must equal one host fold:
//!   - the served entry and geometry (one-shot to 4 rows, two-shot in 16
//!     senders to 32 rows, in 16 senders and 32 workers above) at every
//!     capacity to 64 rows and every live count from 0 to one past the
//!     capacity;
//!   - every prefill token count from 65 to 8192 on the served two-shot
//!     (16 senders, 32 workers), and shorter and clamped live counts;
//!   - prefill row parts: every part of 33 to 2048 rows on the comm lane's
//!     (16, 8) two-shot from row 6144, and one- and two-row parts on the
//!     one-shot and other parts at other row offsets;
//!   - the low-latency BF16 entry (power-of-two ranks), the fallback (the
//!     comm engine's F32 pair over host-rounded operands, rounded on the
//!     host) and the phased F32 round entry over host-rounded operands
//!     (ACC4's composition) at served shapes;
//!   - other sender and worker counts, with and without live rows.
//!
//!   Outputs are poisoned before each call and nothing past the written
//!   rows may change: every case reads back each rank's whole output and
//!   checks every element past the written rows. Every call of both phased headers runs on one state,
//!   with every rank synchronized around each call (a state serves one
//!   payload format at a time). The last receipt records the case count
//!   and the matrix digest.
//! - `phased_bf16_replays`: poisoned outputs, eager, relaunched, from a
//!   graph (including sequences that alternate protocols) and behind a PDL
//!   producer that zeroes the partials and the live count until after its
//!   tail.
//! - `phased_bf16_mutants_are_caught`: each lab mutant changes some rank's
//!   output in an eager call and in a graph replay, separately, two-shot at
//!   2048 rows with workers and at 32 rows with 16 senders and none,
//!   one-shot at 4 rows with 16 senders and 32 rows with workers. A mutant
//!   counts only on an observed output difference in one of up to five
//!   fresh runs per mode; an execution error fails the test. The missing
//!   gather wait is built with the slow rank (`phased_bf16::SLOW_MUTANTS`):
//!   rank 1 sleeps after its partials went out, so the other ranks finish
//!   before its gather arrives; the served entries with the same slow rank
//!   (`SLOW_CONTROL`) pass the same harness.
//! - `phased_bf16_generations`: back-to-back calls on one state, each with
//!   distinct partials, mixing one-shot and two-shot entries, a different
//!   peer delayed before each call, every result snapshotted on the device
//!   right after its call; eager and replayed. The served entries, and the
//!   served entries with the slow rank, show no wrong word. The
//!   shared-parity mutant, built with the slow rank (the other ranks start
//!   their next one-shot call while rank 1 has not read the last), must
//!   show wrong words in each mode, eager and replayed, of every case with
//!   one-shot calls.
//! - `phased_bf16_timing` (`CHOPINFER_KERNEL_BENCH=1`): microseconds per
//!   call of both entries per sender and worker count, against the phased
//!   F32 round entry and the comm engine's F32 pair, in graphs of repeated
//!   calls on every rank, critical path over ranks.
//! - `phased_bf16_matrix_manifest` (CPU, with
//!   `CHOPINFER_PHASED_BF16_MATRIX_MANIFEST` naming a file): writes the
//!   matrix's cases and buffer groups, against which a receipt gate checks
//!   the sweep's group receipts.

use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{Argument, ArgumentList, DType, LaunchPlan, Validate};
use kernel_lab::launch::{self, Bindings, CollectiveModules, RankCall};
use kernel_lab::poison::{self, Fill, POISON, WriteMap};
use kernel_lab::topology::RankReplay;
use kernel_lab::{LabResult, Session, gpu_test_multi, time_multi_graph};
use kernels_sm120::comm::{self as comm, phased_bf16, phased_f32};
use runtime_cuda::{DeviceBuffer, KernelArgument, LaunchConfig};
use serde_json::json;
use std::collections::BTreeMap;

const WIDTH: usize = 4096;
const BITWISE_ROWS: [usize; 16] = [
    1, 2, 3, 4, 5, 8, 16, 24, 32, 33, 64, 256, 1024, 2048, 4096, 8192,
];
/// (senders, workers) of the bitwise sweep up to 64 rows, and above.
const SMALL_GEOMETRIES: [(u32, u32); 6] = [(1, 0), (4, 0), (16, 0), (1, 1), (4, 4), (8, 8)];
const LARGE_GEOMETRIES: [(u32, u32); 7] = [
    (1, 1),
    (8, 8),
    (8, 16),
    (16, 32),
    (24, 32),
    (64, 64),
    (32, 0),
];
/// Runs per mode before a mutant counts as missed.
const MUTANT_TRIALS: usize = 5;
/// The PDL producer's tail before it restores the partials and live count.
const PRODUCER_TAIL_NS: u64 = 50_000;

// Buffer handles: one per rank-local buffer, a range per peer array.
const INPUT: u64 = 1;
const OUTPUT: u64 = 2;
const EPOCH: u64 = 4;
const LIVE: u64 = 5;
const REF_INPUT: u64 = 6;
const REF_OUTPUT: u64 = 7;
const PAIR_EPOCH: u64 = 8;
const PAIR_OUTPUT: u64 = 9;
const INBOXES: u64 = 100;
const FLAGS: u64 = 200;
const OUTPUTS: u64 = 300;
const REF_OUTPUTS: u64 = 400;
const PAIR_INBOXES: u64 = 500;
const PAIR_FLAGS: u64 = 600;
const PAIR_OUTPUTS: u64 = 700;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Partials whose exponents spread over 24 binades; every 4099th element
/// is -0 on every rank and every 61st sits halfway between two BF16
/// values. Every 53rd element is 2^24 on rank 0, -2^24 on rank 3 and 1
/// elsewhere, times a power of two: exact in BF16, and the F32 sum's
/// order shows in the rounded sum at three or more ranks.
fn partial(seed: u64, rank: usize, elements: usize) -> Vec<f32> {
    (0..elements)
        .map(|index| {
            if index % 4099 == 0 {
                return -0.0;
            }
            if index % 53 == 7 {
                let scale = 2f32.powi((splitmix(seed ^ index as u64) % 8) as i32 - 4);
                let big = 16_777_216.0;
                return scale
                    * match rank {
                        0 => big,
                        3 => -big,
                        _ => 1.0,
                    };
            }
            let h = splitmix(seed << 48 ^ (rank as u64) << 40 ^ index as u64);
            let mantissa = (h >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
            let exponent = ((h >> 8) % 24) as i32 - 12;
            let value = mantissa * 2f32.powi(exponent);
            if index % 61 == 0 {
                f32::from_bits(value.to_bits() & 0xffff_0000 | 0x8000)
            } else {
                value
            }
        })
        .collect()
}

/// A rank's partials rounded once to BF16 and widened back, exactly.
fn rounded(input: &[f32]) -> Vec<f32> {
    input.iter().map(|&v| bf16_to_f32(f32_to_bf16(v))).collect()
}

/// The host's fold of the first `count` elements: rounded partials summed
/// in rank order from F32 zero, rounded once.
fn host_sums(inputs: &[Vec<f32>], count: usize) -> Vec<u16> {
    (0..count)
        .map(|index| {
            let mut sum = 0.0_f32;
            for input in inputs {
                sum += bf16_to_f32(f32_to_bf16(input[index]));
            }
            f32_to_bf16(sum)
        })
        .collect()
}

fn first_difference<T: PartialEq>(actual: &[T], expected: &[T]) -> Option<usize> {
    actual
        .iter()
        .zip(expected)
        .position(|(a, e)| a != e)
        .or((actual.len() != expected.len()).then_some(actual.len().min(expected.len())))
}

fn halves(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_ne_bytes([c[0], c[1]]))
        .collect()
}

/// One rank's persistent state, sized for `capacity` rows: the phased
/// state both headers' entries share (sized for the F32 entries, which
/// covers the BF16 ones), the F32 pair's, and the live count.
struct State {
    inbox: DeviceBuffer,
    flags: DeviceBuffer,
    epoch: DeviceBuffer,
    live: DeviceBuffer,
    pair_inbox: DeviceBuffer,
    pair_flags: DeviceBuffer,
    pair_epoch: DeviceBuffer,
}

/// One rank's buffers of one row count.
struct Rows {
    input: DeviceBuffer,
    staged: DeviceBuffer,
    output: DeviceBuffer,
    ref_input: DeviceBuffer,
    ref_output: DeviceBuffer,
    pair_output: DeviceBuffer,
}

fn states(sessions: &[Session], capacity: usize) -> LabResult<Vec<State>> {
    let ranks = sessions.len() as u32;
    let elements = (capacity * WIDTH) as u64;
    let own = phased_f32::buffers(ranks, elements).map_err(err)?;
    let bf16 = phased_bf16::buffers(ranks, elements).map_err(err)?;
    if bf16.inbox_bytes > own.inbox_bytes || bf16.flag_bytes != own.flag_bytes {
        return Err("the F32 state does not hold the BF16 entries' calls".into());
    }
    let pair = comm::phased_buffers(ranks, elements, DType::F32).map_err(err)?;
    let states = sessions
        .iter()
        .map(|session| {
            Ok(State {
                inbox: session.allocate(own.inbox_bytes as usize)?,
                flags: session.allocate(own.flag_bytes as usize)?,
                epoch: session.allocate(own.epoch_bytes as usize)?,
                live: session.allocate(4)?,
                pair_inbox: session.allocate(pair.inbox_bytes as usize)?,
                pair_flags: session.allocate(pair.flag_bytes as usize)?,
                pair_epoch: session.allocate(pair.epoch_bytes as usize)?,
            })
        })
        .collect::<LabResult<Vec<_>>>()?;
    reset(sessions, &states)?;
    Ok(states)
}

fn zero(session: &Session, buffer: &DeviceBuffer) -> LabResult<()> {
    buffer
        .copy_from_bytes(&vec![0; buffer.byte_len()], &session.stream)
        .map_err(err)
}

/// Zeroes every rank's flags and epochs once every rank is idle.
fn reset(sessions: &[Session], states: &[State]) -> LabResult<()> {
    poison::synchronize(sessions)?;
    for (session, state) in sessions.iter().zip(states) {
        for buffer in [
            &state.flags,
            &state.epoch,
            &state.pair_flags,
            &state.pair_epoch,
        ] {
            zero(session, buffer)?;
        }
    }
    Ok(())
}

fn set_live(sessions: &[Session], states: &[State], live: usize) -> LabResult<()> {
    poison::synchronize(sessions)?;
    for (session, state) in sessions.iter().zip(states) {
        state
            .live
            .copy_from_bytes(&(live as u32).to_ne_bytes(), &session.stream)
            .map_err(err)?;
    }
    Ok(())
}

fn rows_buffers(sessions: &[Session], inputs: &[Vec<f32>]) -> LabResult<Vec<Rows>> {
    sessions
        .iter()
        .zip(inputs)
        .map(|(session, input)| {
            let elements = input.len();
            Ok(Rows {
                input: session.upload_f32(input)?,
                staged: session.upload_f32(input)?,
                output: session.allocate(elements * 2)?,
                ref_input: session.upload_f32(&rounded(input))?,
                ref_output: session.allocate(elements * 2)?,
                pair_output: session.allocate(elements * 4)?,
            })
        })
        .collect()
}

fn bindings<'a>(states: &'a [State], rows: &'a [Rows], rank: usize) -> Bindings<'a> {
    let mut buffers: BTreeMap<u64, &DeviceBuffer> = BTreeMap::new();
    buffers.insert(INPUT, &rows[rank].input);
    buffers.insert(OUTPUT, &rows[rank].output);
    buffers.insert(REF_INPUT, &rows[rank].ref_input);
    buffers.insert(REF_OUTPUT, &rows[rank].ref_output);
    buffers.insert(EPOCH, &states[rank].epoch);
    buffers.insert(LIVE, &states[rank].live);
    buffers.insert(PAIR_OUTPUT, &rows[rank].pair_output);
    buffers.insert(PAIR_EPOCH, &states[rank].pair_epoch);
    for peer in 0..states.len() {
        let q = peer as u64;
        buffers.insert(INBOXES + q, &states[peer].inbox);
        buffers.insert(FLAGS + q, &states[peer].flags);
        buffers.insert(OUTPUTS + q, &rows[peer].output);
        buffers.insert(REF_OUTPUTS + q, &rows[peer].ref_output);
        buffers.insert(PAIR_INBOXES + q, &states[peer].pair_inbox);
        buffers.insert(PAIR_FLAGS + q, &states[peer].pair_flags);
        buffers.insert(PAIR_OUTPUTS + q, &rows[peer].pair_output);
    }
    Bindings { buffers }
}

/// Which entry a call runs: this header's, or the phased F32 round entry
/// (the reference, over the host-rounded partials into its own output).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Entry {
    TwoShot,
    OneShot,
    F32Round,
    F32OneShotRound,
}

const ENTRIES: [Entry; 2] = [Entry::TwoShot, Entry::OneShot];

impl Entry {
    fn symbol(self) -> &'static str {
        match self {
            Entry::TwoShot => phased_bf16::SYMBOL,
            Entry::OneShot => phased_bf16::ONE_SHOT_SYMBOL,
            Entry::F32Round => phased_f32::ROUND_SYMBOL,
            Entry::F32OneShotRound => phased_f32::ONE_SHOT_ROUND_SYMBOL,
        }
    }

    fn reference(self) -> bool {
        matches!(self, Entry::F32Round | Entry::F32OneShotRound)
    }

    fn one_shot(self) -> bool {
        matches!(self, Entry::OneShot | Entry::F32OneShotRound)
    }

    fn takes(self, rows: usize) -> bool {
        !self.one_shot() || (rows * WIDTH) as u64 <= phased_bf16::ONE_SHOT_MAX_ELEMENTS
    }

    /// The reference entry of the same protocol.
    fn reference_entry(self) -> Entry {
        match self.one_shot() {
            true => Entry::F32OneShotRound,
            false => Entry::F32Round,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Entry::TwoShot => "bf16",
            Entry::OneShot => "oneshot_bf16",
            Entry::F32Round => "f32_round",
            Entry::F32OneShotRound => "f32_oneshot_round",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Shape {
    rows: usize,
    live: bool,
    senders: u32,
    workers: u32,
}

impl Shape {
    fn new(rows: usize, live: bool, (senders, workers): (u32, u32)) -> Self {
        Self {
            rows,
            live,
            senders,
            workers,
        }
    }
}

fn entry_args(rank: usize, ranks: usize, entry: Entry, shape: Shape) -> LabResult<ArgumentList> {
    let peers = |base: u64| (0..ranks as u64).map(|q| base + q).collect::<Vec<_>>();
    let elements = (shape.rows * WIDTH) as u64;
    let live_rows = shape.live.then_some(LIVE);
    if entry.reference() {
        return phased_f32::arguments(
            entry.symbol(),
            phased_f32::Call {
                input: REF_INPUT,
                output: REF_OUTPUT,
                inboxes: peers(INBOXES),
                outputs: peers(REF_OUTPUTS),
                flags: peers(FLAGS),
                epoch: EPOCH,
                elements,
                row_elements: WIDTH as u32,
                live_rows,
                rank: rank as u32,
                ranks: ranks as u32,
                senders: shape.senders,
            },
        )
        .map_err(err);
    }
    phased_bf16::arguments(
        entry.symbol(),
        phased_bf16::Call {
            input: INPUT,
            output: OUTPUT,
            inboxes: peers(INBOXES),
            outputs: peers(OUTPUTS),
            flags: peers(FLAGS),
            epoch: EPOCH,
            elements,
            row_elements: WIDTH as u32,
            live_rows,
            rank: rank as u32,
            ranks: ranks as u32,
            senders: shape.senders,
        },
    )
    .map_err(err)
}

/// One call of `symbol` (`entry`'s symbol in `modules`, served or mutant)
/// on every rank, or on `only_rank`.
#[allow(clippy::too_many_arguments)]
fn launch_entry(
    sessions: &[Session],
    modules: &CollectiveModules,
    symbol: &str,
    states: &[State],
    rows: &[Rows],
    entry: Entry,
    shape: Shape,
    only_rank: Option<usize>,
) -> LabResult<()> {
    launch_bound(sessions, modules, symbol, entry, shape, only_rank, |rank| {
        bindings(states, rows, rank)
    })
}

/// [`launch_entry`] with the handles bound by `bind(rank)`.
fn launch_bound<'a>(
    sessions: &[Session],
    modules: &CollectiveModules,
    symbol: &str,
    entry: Entry,
    shape: Shape,
    only_rank: Option<usize>,
    bind: impl Fn(usize) -> Bindings<'a>,
) -> LabResult<()> {
    let plan = phased_bf16::plan_with(shape.senders, shape.workers).map_err(err)?;
    launch::launch_collective(sessions, modules, symbol, plan, only_rank, |rank| {
        Ok(RankCall {
            args: entry_args(rank, sessions.len(), entry, shape)?,
            bindings: bind(rank),
        })
    })
}

/// The pair's plan at any row count (`comm::plan`'s two-shot plan: one
/// sending cluster and up to seven worker clusters).
fn pair_plan(rows: usize) -> LabResult<LaunchPlan> {
    let cluster = 8;
    let tiles = (rows * WIDTH).div_ceil(4096) as u32;
    let plan = LaunchPlan {
        grid: [cluster * (1 + tiles.div_ceil(cluster).clamp(1, 7)), 1, 1],
        block: [256, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes: 0,
        cluster: [cluster, 1, 1],
        pdl: true,
    };
    plan.validate().map_err(err)?;
    Ok(plan)
}

/// The comm engine's F32 pair over the F32 partials bound to `input`
/// (`INPUT`, or `REF_INPUT`: the host-rounded ones) into the pair output,
/// on every rank or `only_rank`.
fn launch_pair(
    sessions: &[Session],
    modules: &CollectiveModules,
    states: &[State],
    rows: &[Rows],
    count: usize,
    input: u64,
    only_rank: Option<usize>,
) -> LabResult<()> {
    let ranks = sessions.len();
    let elements = (count * WIDTH) as u64;
    let plan = pair_plan(count)?;
    let peers = |base: u64| (0..ranks as u64).map(|q| base + q).collect::<Vec<_>>();
    launch::launch_collective(
        sessions,
        modules,
        "chop_comm_reduce_scatter_f32_v1",
        plan,
        only_rank,
        |rank| {
            Ok(RankCall {
                args: comm::arguments(comm::OneShotCall {
                    input,
                    output: PAIR_OUTPUT,
                    peer_inboxes: peers(PAIR_INBOXES),
                    peer_flags: peers(PAIR_FLAGS),
                    epoch: PAIR_EPOCH,
                    elements,
                    rank: rank as u32,
                    ranks: ranks as u32,
                })
                .map_err(err)?,
                bindings: bindings(states, rows, rank),
            })
        },
    )?;
    launch::launch_collective(
        sessions,
        modules,
        "chop_comm_allgather_scattered_f32_v1",
        plan,
        only_rank,
        |rank| {
            Ok(RankCall {
                args: ArgumentList(vec![
                    Argument::DeviceBuffer(PAIR_OUTPUT),
                    Argument::DevicePointerArray(peers(PAIR_OUTPUTS)),
                    Argument::DevicePointerArray(peers(PAIR_FLAGS)),
                    Argument::DeviceBuffer(PAIR_EPOCH),
                    Argument::U64(elements),
                    Argument::U32(rank as u32),
                    Argument::U32(ranks as u32),
                ]),
                bindings: bindings(states, rows, rank),
            })
        },
    )
}

fn synchronize(sessions: &[Session]) -> LabResult<()> {
    for session in sessions {
        session.stream.synchronize().map_err(err)?;
    }
    Ok(())
}

fn output_of(rows: &Rows, entry: Entry) -> &DeviceBuffer {
    if entry.reference() {
        &rows.ref_output
    } else {
        &rows.output
    }
}

fn poison_outputs(sessions: &[Session], rows: &[Rows], entry: Entry) -> LabResult<()> {
    poison::synchronize(sessions)?;
    for (session, rows) in sessions.iter().zip(rows) {
        poison::poison(session, &[output_of(rows, entry)])?;
    }
    Ok(())
}

/// Every rank's output of `entry` against the host: the first `live` rows
/// written with the expected bits, every other byte still poison.
fn check_outputs(
    sessions: &[Session],
    rows: &[Rows],
    entry: Entry,
    expected: &[u16],
    live: usize,
    label: &str,
) -> LabResult<()> {
    let count = live * WIDTH;
    for (rank, (session, rows)) in sessions.iter().zip(rows).enumerate() {
        let label = format!("{label}, rank {rank}");
        let bytes = poison::read(session, output_of(rows, entry))?;
        let map = WriteMap::of(DType::Bf16).prefix(count)?;
        poison::check(&label, &bytes, &map, Fill::Pattern(&[POISON]))?;
        let actual = halves(&bytes[..count * 2]);
        if let Some(at) = first_difference(&actual, &expected[..count]) {
            return Err(format!(
                "{label}: rounded sum differs at {at}: {:#06x} != {:#06x}",
                actual[at], expected[at]
            ));
        }
    }
    Ok(())
}

fn inputs_for(ranks: usize, seed: u64, rows: usize) -> Vec<Vec<f32>> {
    (0..ranks)
        .map(|rank| partial(seed, rank, rows * WIDTH))
        .collect()
}

fn receipt(session: &Session, data: Vec<(&str, serde_json::Value)>) -> LabResult<()> {
    session.receipt(
        "comm_phased_bf16",
        data.into_iter().map(|(k, v)| (k.to_owned(), v)).collect(),
    )?;
    Ok(())
}

/// One call of `entry` with `shape` into poisoned outputs, checked on every
/// rank against `expected` over the shape's live rows.
#[allow(clippy::too_many_arguments)]
fn checked(
    sessions: &[Session],
    modules: &[&CollectiveModules; 2],
    states: &[State],
    rows: &[Rows],
    entry: Entry,
    shape: Shape,
    expected: &[u16],
    live: usize,
    label: &str,
) -> LabResult<()> {
    poison_outputs(sessions, rows, entry)?;
    let modules = modules[usize::from(entry.reference())];
    launch_entry(
        sessions,
        modules,
        entry.symbol(),
        states,
        rows,
        entry,
        shape,
        None,
    )?;
    synchronize(sessions)?;
    check_outputs(sessions, rows, entry, expected, live, label)
}

/// The bitwise matrix's operands: rank q's partial at row r is the same
/// values in every case (`partial(OPERAND_SEED, q, ..)` at element
/// r * WIDTH + column), whatever the path, capacity, live count, geometry
/// or row offset, so row r's expected sum is one host fold of every rank's
/// row r.
const OPERAND_SEED: u64 = 0xb16;
const OPERAND_ROWS: usize = 8192;
/// The served geometries (the serving bundle's contract): the one-shot (1-4 rows)
/// and the two-shot up to 32 rows in 16 senders, the two-shot above 32
/// rows in 16 senders and 32 workers, and a prefill row part's two-shot
/// narrowed to its comm lane's 24 CTAs.
const SERVED_SMALL: (u32, u32) = (16, 0);
const SERVED_LARGE: (u32, u32) = (16, 32);
const SERVED_PART: (u32, u32) = (16, 8);
/// FNV-1a of the four-rank matrix (`matrix_digest`), and its case count:
/// `phased_bf16_matrix_is_pinned` holds them, the GPU sweep records them
/// in its completion receipt.
const MATRIX_DIGEST: &str = "76c42b9554252f1b";
const MATRIX_CASES: usize = 12734;

/// How a matrix case reduces.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Path {
    /// A phased entry of either header (the F32 round entries over the
    /// host-rounded operands).
    Phased(Entry),
    /// The low-latency BF16 entry (`comm::ll_bf16`): every row of its
    /// call, no live count.
    LowLatency,
    /// The bundle's fallback where create refuses the spinning grids: the
    /// host-rounded operands through the comm engine's F32 pair, rounded on
    /// the host.
    Fallback,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Case {
    path: Path,
    /// Rows of the call.
    capacity: usize,
    /// The live row count, or `None` for a call without one.
    live: Option<usize>,
    /// (senders, workers) of a phased call.
    geometry: (u32, u32),
    /// The operands' first row (a prefill row part's offset).
    first: usize,
}

impl Case {
    /// Rows the call writes.
    fn written(&self) -> usize {
        match (self.path, self.live) {
            (Path::Phased(_), Some(live)) => live.min(self.capacity),
            _ => self.capacity,
        }
    }

    /// Rows of the buffers the case runs on: the smallest of 64, 256, 2048
    /// and 8192 that holds it, within the operands.
    fn class(&self) -> usize {
        let class = [64, 256, 2048, OPERAND_ROWS]
            .into_iter()
            .find(|&rows| rows >= self.capacity)
            .unwrap_or(OPERAND_ROWS);
        class.min(OPERAND_ROWS - self.first)
    }

    fn label(&self) -> String {
        format!(
            "{:?} {} rows from row {}, live {:?}, {:?}",
            self.path, self.capacity, self.first, self.live, self.geometry
        )
    }
}

/// The served entry and geometry of a call of `rows` rows.
fn served(rows: usize) -> (Entry, (u32, u32)) {
    match rows {
        1..=4 => (Entry::OneShot, SERVED_SMALL),
        5..=32 => (Entry::TwoShot, SERVED_SMALL),
        _ => (Entry::TwoShot, SERVED_LARGE),
    }
}

/// Prefill steps above 64 rows: a step reduces its own token count, any
/// count to 8192, on the two-shot in 16 senders and 32 workers.
fn prefill_counts() -> std::ops::RangeInclusive<usize> {
    65..=OPERAND_ROWS
}

/// Prefill row parts above 32 rows (a step's 2048-row pieces, its last
/// piece any count): the two-shot on the comm lane's (16, 8) grid.
fn part_counts() -> std::ops::RangeInclusive<usize> {
    33..=2048
}

/// The row offset of the `part_counts` sweep (the last piece of a full
/// step).
const PART_FIRST: usize = 6144;

/// Every case of the bitwise sweep at `ranks` ranks.
fn matrix(ranks: usize) -> Vec<Case> {
    let mut cases = Vec::new();
    let mut push = |path, capacity, live, geometry, first| {
        cases.push(Case {
            path,
            capacity,
            live,
            geometry,
            first,
        })
    };
    // Decode and verify buckets and short prefills: the served entry at
    // every capacity to 64 and every live count from 0 past the capacity.
    for capacity in 1..=64 {
        let (entry, geometry) = served(capacity);
        for live in 0..=capacity + 1 {
            push(Path::Phased(entry), capacity, Some(live), geometry, 0);
        }
    }
    // Prefill steps above 64 rows: every token count, live = rows, and some
    // shorter or clamped live counts.
    for capacity in prefill_counts() {
        push(
            Path::Phased(Entry::TwoShot),
            capacity,
            Some(capacity),
            SERVED_LARGE,
            0,
        );
    }
    for (capacity, live) in [
        (2048, 0),
        (2048, 1),
        (2048, 1000),
        (2048, 2047),
        (2048, 8192),
        (8192, 0),
        (8192, 1),
        (8192, 4095),
        (8192, 7777),
        (8192, 8191),
    ] {
        push(
            Path::Phased(Entry::TwoShot),
            capacity,
            Some(live),
            SERVED_LARGE,
            0,
        );
    }
    // Prefill row parts (live count: the step's tokens): one- and two-row
    // parts on the one-shot, longer parts on the comm lane's two-shot.
    for (capacity, first) in [
        (1, 1),
        (1, 2047),
        (1, 2048),
        (1, 4095),
        (1, 8191),
        (2, 2046),
        (2, 6142),
    ] {
        push(
            Path::Phased(Entry::OneShot),
            capacity,
            Some(8192),
            SERVED_SMALL,
            first,
        );
    }
    for (capacity, first) in [
        (2048, 0),
        (2048, 2048),
        (2048, 4096),
        (2048, 6144),
        (2047, 6145),
        (33, 4096),
        (64, 2048),
        (4096, 4096),
        (6144, 2048),
    ] {
        push(
            Path::Phased(Entry::TwoShot),
            capacity,
            Some(8192),
            SERVED_PART,
            first,
        );
    }
    for capacity in part_counts() {
        push(
            Path::Phased(Entry::TwoShot),
            capacity,
            Some(8192),
            SERVED_PART,
            PART_FIRST,
        );
    }
    // The low-latency entry (served at 1 row) on the same operands.
    if ranks.is_power_of_two() {
        for capacity in [1, 2, 4, 32] {
            push(Path::LowLatency, capacity, None, (0, 0), 0);
        }
    }
    // The fallback and the phased F32 round entries (ACC4's composition)
    // at served shapes.
    for capacity in [1, 2, 4, 5, 32, 33, 64, 2048, 8192] {
        push(Path::Fallback, capacity, None, (0, 0), 0);
        let (entry, geometry) = served(capacity);
        push(
            Path::Phased(entry.reference_entry()),
            capacity,
            Some(capacity),
            geometry,
            0,
        );
    }
    // Other sender and worker counts, with and without live rows.
    for &capacity in &BITWISE_ROWS {
        let geometries = if capacity <= 64 {
            &SMALL_GEOMETRIES[..]
        } else {
            &LARGE_GEOMETRIES[..]
        };
        for &geometry in geometries {
            for entry in ENTRIES.into_iter().filter(|e| e.takes(capacity)) {
                push(Path::Phased(entry), capacity, Some(capacity), geometry, 0);
                push(Path::Phased(entry), capacity, None, geometry, 0);
            }
        }
    }
    cases
}

/// [`checked`] for the large cases, with the same coverage and a cheaper
/// host check: every rank's whole output is read back, its written rows
/// compared with the host's sums and every other element with the poison
/// word, as words instead of through a byte map.
#[allow(clippy::too_many_arguments)]
fn checked_words(
    sessions: &[Session],
    modules: &[&CollectiveModules; 2],
    states: &[State],
    rows: &[Rows],
    entry: Entry,
    shape: Shape,
    expected: &[u16],
    live: usize,
    label: &str,
) -> LabResult<()> {
    poison_outputs(sessions, rows, entry)?;
    launch_entry(
        sessions,
        modules[usize::from(entry.reference())],
        entry.symbol(),
        states,
        rows,
        entry,
        shape,
        None,
    )?;
    synchronize(sessions)?;
    let poisoned = u16::from_ne_bytes([POISON, POISON]);
    for (rank, (session, rows)) in sessions.iter().zip(rows).enumerate() {
        let buffer = output_of(rows, entry);
        let actual = session.download_bf16(buffer, buffer.byte_len() / 2)?;
        let count = live * WIDTH;
        if let Some(at) = first_difference(&actual[..count], &expected[..count]) {
            return Err(format!(
                "{label}, rank {rank}: rounded sum differs at {at}: {:#06x} != {:#06x}",
                actual[at], expected[at]
            ));
        }
        if let Some(at) = actual[count..].iter().position(|&word| word != poisoned) {
            return Err(format!(
                "{label}, rank {rank}: element {} past the live rows was written",
                count + at
            ));
        }
    }
    Ok(())
}

/// FNV-1a over the cases' debug forms, one per line.
fn matrix_digest(cases: &[Case]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for case in cases {
        for byte in format!("{case:?}\n").bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
        }
    }
    format!("{hash:016x}")
}

/// One rank's low-latency state (capacity 32 rows).
struct LowLatencyState {
    inbox: DeviceBuffer,
    epochs: DeviceBuffer,
}

const LL_CAPACITY_ROWS: usize = 32;
const LL_INBOXES: u64 = 800;
const LL_EPOCHS: u64 = 10;

/// The low-latency entry over `rows` of `capacity` rows on every rank.
fn launch_low_latency(
    sessions: &[Session],
    modules: &CollectiveModules,
    states: &[State],
    ll: &[LowLatencyState],
    rows: &[Rows],
    capacity: usize,
) -> LabResult<()> {
    let ranks = sessions.len();
    let elements = (capacity * WIDTH) as u64;
    let plan = kernels_sm120::comm::ll_bf16::plan(ranks as u32, elements).map_err(err)?;
    launch::launch_collective(
        sessions,
        modules,
        kernels_sm120::comm::ll_bf16::SYMBOL,
        plan,
        None,
        |rank| {
            let call = kernels_sm120::comm::ll_bf16::Call {
                input: INPUT,
                output: OUTPUT,
                peer_inboxes: (0..ranks as u64).map(|q| LL_INBOXES + q).collect(),
                epochs: LL_EPOCHS,
                elements,
                capacity_elements: (LL_CAPACITY_ROWS * WIDTH) as u64,
                rank: rank as u32,
                ranks: ranks as u32,
            };
            let mut bindings = bindings(states, rows, rank);
            bindings.buffers.insert(LL_EPOCHS, &ll[rank].epochs);
            for (peer, state) in ll.iter().enumerate() {
                bindings
                    .buffers
                    .insert(LL_INBOXES + peer as u64, &state.inbox);
            }
            Ok(RankCall {
                args: kernels_sm120::comm::ll_bf16::arguments(&call).map_err(err)?,
                bindings,
            })
        },
    )
}

/// The fallback's F32 sums, rounded on the host, against `expected`, and
/// nothing past `count` rows written.
fn check_fallback(
    sessions: &[Session],
    rows: &[Rows],
    expected: &[u16],
    count: usize,
    label: &str,
) -> LabResult<()> {
    let elements = count * WIDTH;
    for (rank, (session, rows)) in sessions.iter().zip(rows).enumerate() {
        let label = format!("{label}, rank {rank}");
        let bytes = poison::read(session, &rows.pair_output)?;
        let map = WriteMap::of(DType::F32).prefix(elements)?;
        poison::check(&label, &bytes, &map, Fill::Pattern(&[POISON]))?;
        let actual: Vec<u16> = bytes[..elements * 4]
            .chunks_exact(4)
            .map(|c| f32_to_bf16(f32::from_ne_bytes([c[0], c[1], c[2], c[3]])))
            .collect();
        if let Some(at) = first_difference(&actual, &expected[..elements]) {
            return Err(format!(
                "{label}: the fallback's rounded sum differs at {at}: {:#06x} != {:#06x}",
                actual[at], expected[at]
            ));
        }
    }
    Ok(())
}

fn run_bitwise(sessions: &[Session]) -> LabResult<()> {
    let ranks = sessions.len();
    let served_modules = CollectiveModules::load(sessions, phased_bf16::LAB_CUBIN)?;
    let reference = CollectiveModules::load(sessions, phased_f32::LAB_CUBIN)?;
    let pair = CollectiveModules::load(sessions, "comm_sm120")?;
    let modules = [&served_modules, &reference];
    let cases = matrix(ranks);
    let digest = matrix_digest(&cases);
    let low_latency = ranks.is_power_of_two();
    let ll_modules = if low_latency {
        Some(CollectiveModules::load(
            sessions,
            kernels_sm120::comm::ll_bf16::LAB_CUBIN,
        )?)
    } else {
        None
    };
    // One phased state for every call of both headers; the sweep
    // synchronizes every rank around each call, so calls of the two
    // payload formats never overlap (a state serves one format at a time).
    let states = states(sessions, OPERAND_ROWS)?;
    let ll = if low_latency {
        let bytes = kernels_sm120::comm::ll_bf16::state_bytes(
            ranks as u32,
            (LL_CAPACITY_ROWS * WIDTH) as u64,
        )
        .map_err(err)?;
        sessions
            .iter()
            .map(|session| {
                let state = LowLatencyState {
                    inbox: session.allocate(bytes.inbox as usize)?,
                    epochs: session.allocate(bytes.epochs as usize)?,
                };
                poison::fill_all(session, &state.inbox, &[0])?;
                poison::fill_all(session, &state.epochs, &[0])?;
                Ok(state)
            })
            .collect::<LabResult<Vec<_>>>()?
    } else {
        Vec::new()
    };
    let operands = inputs_for(ranks, OPERAND_SEED, OPERAND_ROWS);
    let expected = host_sums(&operands, OPERAND_ROWS * WIDTH);
    if ranks > 2 {
        let reversed: Vec<_> = operands.iter().rev().cloned().collect();
        if host_sums(&reversed, OPERAND_ROWS * WIDTH) == expected {
            return Err("the rank order changes no sum".into());
        }
    }
    // Cases grouped by their buffers (first row, rows), in matrix order
    // within a group; every group's calls share the one state.
    let mut groups: Vec<((usize, usize), Vec<Case>)> = Vec::new();
    for case in &cases {
        let key = (case.first, case.class());
        match groups.iter_mut().find(|(k, _)| *k == key) {
            Some((_, group)) => group.push(*case),
            None => groups.push((key, vec![*case])),
        }
    }
    let mut ran = 0;
    for ((first, class), group) in &groups {
        let slices: Vec<Vec<f32>> = operands
            .iter()
            .map(|rank| rank[first * WIDTH..(first + class) * WIDTH].to_vec())
            .collect();
        let rows = rows_buffers(sessions, &slices)?;
        let expected = &expected[first * WIDTH..];
        for case in group {
            let label = case.label();
            let count = case.written();
            match case.path {
                Path::Phased(entry) => {
                    if let Some(live) = case.live {
                        set_live(sessions, &states, live)?;
                    }
                    let shape = Shape::new(case.capacity, case.live.is_some(), case.geometry);
                    if *class <= 2048 {
                        checked(
                            sessions, &modules, &states, &rows, entry, shape, expected, count,
                            &label,
                        )?;
                    } else {
                        checked_words(
                            sessions, &modules, &states, &rows, entry, shape, expected, count,
                            &label,
                        )?;
                    }
                }
                Path::LowLatency => {
                    let modules = ll_modules.as_ref().ok_or("no low-latency cubin")?;
                    poison_outputs(sessions, &rows, Entry::TwoShot)?;
                    launch_low_latency(sessions, modules, &states, &ll, &rows, case.capacity)?;
                    synchronize(sessions)?;
                    check_outputs(sessions, &rows, Entry::TwoShot, expected, count, &label)?;
                }
                Path::Fallback => {
                    poison::synchronize(sessions)?;
                    for (session, rows) in sessions.iter().zip(&rows) {
                        poison::poison(session, &[&rows.pair_output])?;
                    }
                    launch_pair(
                        sessions,
                        &pair,
                        &states,
                        &rows,
                        case.capacity,
                        REF_INPUT,
                        None,
                    )?;
                    synchronize(sessions)?;
                    check_fallback(sessions, &rows, expected, count, &label)?;
                }
            }
            ran += 1;
        }
        for session in sessions {
            receipt(
                session,
                vec![
                    ("case", json!("phased_bf16_matches_host_and_round_entry")),
                    ("ranks", json!(ranks)),
                    ("first_row", json!(first)),
                    ("buffer_rows", json!(class)),
                    (
                        "cases",
                        json!(
                            group
                                .iter()
                                .map(|case| format!("{case:?}"))
                                .collect::<Vec<_>>()
                        ),
                    ),
                    ("bitwise", json!(true)),
                ],
            )?;
        }
    }
    if ran != cases.len() {
        return Err(format!("ran {ran} of {} matrix cases", cases.len()));
    }
    for session in sessions {
        receipt(
            session,
            vec![
                ("case", json!("phased_bf16_matrix_complete")),
                ("ranks", json!(ranks)),
                ("cases", json!(ran)),
                ("matrix_digest", json!(digest)),
                ("low_latency", json!(low_latency)),
            ],
        )?;
    }
    Ok(())
}

#[test]
fn phased_bf16_matches_host_and_round_entry() {
    gpu_test_multi(3 << 30, run_bitwise).expect("phased BF16 all-reduce bitwise sweep");
}

/// The variable naming the file [`phased_bf16_matrix_manifest`] writes.
const MANIFEST_VARIABLE: &str = "CHOPINFER_PHASED_BF16_MATRIX_MANIFEST";

/// With [`MANIFEST_VARIABLE`] naming a file, writes the matrix there for a
/// receipt gate: every case's debug form (as the sweep's receipts record
/// it) and its buffer group (first row and buffer rows, as the group
/// receipts name it), in matrix order, beside the pinned digest and count.
/// Without the variable it writes nothing.
#[test]
fn phased_bf16_matrix_manifest() {
    let Some(path) = std::env::var_os(MANIFEST_VARIABLE) else {
        return;
    };
    let cases = matrix(4);
    let manifest = json!({
        "matrix_digest": MATRIX_DIGEST,
        "matrix_cases": MATRIX_CASES,
        "cases": cases
            .iter()
            .map(|case| json!({
                "case": format!("{case:?}"),
                "first_row": case.first,
                "buffer_rows": case.class(),
            }))
            .collect::<Vec<_>>(),
    });
    std::fs::write(&path, format!("{manifest}\n")).expect("the matrix manifest is written");
}

/// The bitwise matrix is pinned: its digest and case count, and the served
/// coverage it must keep.
#[test]
fn phased_bf16_matrix_is_pinned() {
    let cases = matrix(4);
    assert_eq!(matrix_digest(&cases), MATRIX_DIGEST);
    assert_eq!(cases.len(), MATRIX_CASES);
    assert_eq!(matrix(2), cases, "two and four ranks run one matrix");
    assert!(matrix(3).iter().all(|case| case.path != Path::LowLatency));
    let has = |path: Path, capacity: usize, live: Option<usize>, geometry, first| {
        cases.contains(&Case {
            path,
            capacity,
            live,
            geometry,
            first,
        })
    };
    // The served entry and geometry at every capacity to 64 and every
    // live count 0..=capacity + 1.
    for capacity in 1..=64 {
        let (entry, geometry) = served(capacity);
        for live in 0..=capacity + 1 {
            assert!(has(Path::Phased(entry), capacity, Some(live), geometry, 0));
        }
    }
    // The production two-shot at 33-64 rows has workers; above, the large
    // capacities at their own live count.
    assert_eq!(served(33), (Entry::TwoShot, (16, 32)));
    assert_eq!(served(32), (Entry::TwoShot, (16, 0)));
    assert_eq!(served(4), (Entry::OneShot, (16, 0)));
    for capacity in prefill_counts() {
        assert!(has(
            Path::Phased(Entry::TwoShot),
            capacity,
            Some(capacity),
            SERVED_LARGE,
            0
        ));
    }
    // Every admitted prefill count, including 193, and every part size.
    assert_eq!(prefill_counts(), 65..=8192);
    assert!(has(
        Path::Phased(Entry::TwoShot),
        193,
        Some(193),
        SERVED_LARGE,
        0
    ));
    for capacity in part_counts() {
        assert!(has(
            Path::Phased(Entry::TwoShot),
            capacity,
            Some(8192),
            SERVED_PART,
            PART_FIRST
        ));
    }
    for rows in 1..=OPERAND_ROWS {
        let (entry, geometry) = served(rows);
        assert!(
            has(Path::Phased(entry), rows, Some(rows), geometry, 0),
            "{rows}"
        );
    }
    // Parts, the low-latency entry, the fallback and the F32 round entry.
    assert!(has(
        Path::Phased(Entry::OneShot),
        1,
        Some(8192),
        SERVED_SMALL,
        8191
    ));
    assert!(has(
        Path::Phased(Entry::TwoShot),
        2048,
        Some(8192),
        SERVED_PART,
        6144
    ));
    assert!(has(Path::LowLatency, 1, None, (0, 0), 0));
    for capacity in [1, 32, 33, 8192] {
        assert!(has(Path::Fallback, capacity, None, (0, 0), 0));
        let (entry, geometry) = served(capacity);
        assert!(has(
            Path::Phased(entry.reference_entry()),
            capacity,
            Some(capacity),
            geometry,
            0
        ));
    }
    for case in &cases {
        assert!(case.capacity >= 1 && case.first + case.capacity <= OPERAND_ROWS);
        assert!(case.class() >= case.capacity, "{case:?}");
        if let Path::Phased(entry) = case.path {
            assert!(entry.takes(case.capacity), "{case:?}");
        }
    }
}

/// A PDL producer in front of a call (`probe_pdl_prefetch_sm120`): it
/// zeroes the rank's partials and live count, releases its dependent, and
/// restores both from the staged copy and `rows` after its tail. It reads
/// nothing from the host, so a graph can capture it.
struct Producer {
    function: runtime_cuda::Function,
    eviction: DeviceBuffer,
    checksum: DeviceBuffer,
    operands: DeviceBuffer,
}

impl Producer {
    fn new(session: &Session, rows: &Rows) -> LabResult<Self> {
        let module = session.module("probe_pdl_prefetch_sm120")?;
        let eviction = session.allocate(16)?;
        eviction
            .copy_from_bytes(&[0; 16], &session.stream)
            .map_err(err)?;
        // One PdlOperand: fresh (the staged copy), input, words, flags 0.
        let words = rows.input.byte_len() / 4;
        let mut operand = Vec::with_capacity(24);
        operand.extend(
            rows.staged
                .device_address_at(0, rows.staged.byte_len())
                .map_err(err)?
                .to_ne_bytes(),
        );
        operand.extend(
            rows.input
                .device_address_at(0, rows.input.byte_len())
                .map_err(err)?
                .to_ne_bytes(),
        );
        operand.extend((words as u32).to_ne_bytes());
        operand.extend(0u32.to_ne_bytes());
        let operands = session.allocate(24)?;
        operands
            .copy_from_bytes(&operand, &session.stream)
            .map_err(err)?;
        Ok(Self {
            function: module
                .function("chop_probe_pdl_prefetch_producer_v1")
                .map_err(err)?,
            eviction,
            checksum: session.allocate(128 * 4)?,
            operands,
        })
    }

    fn launch(&self, session: &Session, live: &DeviceBuffer, rows: usize) -> LabResult<()> {
        self.function
            .launch(
                &session.stream,
                launch::simple_config((1, 1, 1), (128, 1, 1), 0, LaunchConfig::NO_CLUSTER, true)?,
                &[
                    KernelArgument::Buffer(&self.eviction),
                    KernelArgument::Buffer(&self.checksum),
                    KernelArgument::U64(1),
                    KernelArgument::Buffer(&self.operands),
                    KernelArgument::U32(1),
                    KernelArgument::Buffer(live),
                    KernelArgument::U32(rows as u32),
                    KernelArgument::U64(PRODUCER_TAIL_NS),
                ],
            )
            .map_err(err)
    }
}

fn run_replays(sessions: &[Session]) -> LabResult<()> {
    let ranks = sessions.len();
    let served = CollectiveModules::load(sessions, phased_bf16::LAB_CUBIN)?;
    let states = states(sessions, 2048)?;
    for (seed, count, geometry) in [
        (40, 2, (16, 0)),
        (41, 4, (16, 0)),
        (42, 32, (16, 0)),
        (43, 32, (4, 4)),
        (44, 2048, (16, 32)),
        (45, 2048, (8, 16)),
    ] {
        let inputs = inputs_for(ranks, seed, count);
        let expected = host_sums(&inputs, count * WIDTH);
        let rows = rows_buffers(sessions, &inputs)?;
        set_live(sessions, &states, count)?;
        let check = |phase: &str| -> LabResult<()> {
            synchronize(sessions)?;
            check_outputs(
                sessions,
                &rows,
                Entry::TwoShot,
                &expected,
                count,
                &format!("{count} rows, {geometry:?}, {phase}"),
            )
        };
        let producers = sessions
            .iter()
            .zip(&rows)
            .map(|(session, rows)| Producer::new(session, rows))
            .collect::<LabResult<Vec<_>>>()?;
        let mut sequences = vec![vec![Entry::TwoShot]];
        if Entry::OneShot.takes(count) {
            sequences.push(vec![Entry::OneShot]);
            // Both protocols in one sequence, either order.
            sequences.push(vec![Entry::OneShot, Entry::TwoShot]);
            sequences.push(vec![Entry::TwoShot, Entry::OneShot]);
        }
        for sequence in sequences {
            let label = sequence
                .iter()
                .map(|entry| entry.name())
                .collect::<Vec<_>>()
                .join("+");
            let run = |only_rank: Option<usize>| -> LabResult<()> {
                for &entry in &sequence {
                    launch_entry(
                        sessions,
                        &served,
                        entry.symbol(),
                        &states,
                        &rows,
                        entry,
                        Shape::new(count, true, geometry),
                        only_rank,
                    )?;
                }
                Ok(())
            };
            poison_outputs(sessions, &rows, Entry::TwoShot)?;
            run(None)?;
            check(&format!("{label} eager"))?;
            poison_outputs(sessions, &rows, Entry::TwoShot)?;
            run(None)?;
            check(&format!("{label} relaunch"))?;
            let graphs = RankReplay::new(sessions, |rank, _| run(Some(rank)))?;
            for replay in 0..3 {
                poison_outputs(sessions, &rows, Entry::TwoShot)?;
                graphs.launch(sessions)?;
                check(&format!("{label} graph replay {replay}"))?;
            }
            // Behind the PDL producer, eager and captured: each call
            // follows its own producer, which zeroes the partials and the
            // live count.
            let behind = |only_rank: Option<usize>| -> LabResult<()> {
                for &entry in &sequence {
                    for rank in 0..ranks {
                        if only_rank.is_some_and(|only| only != rank) {
                            continue;
                        }
                        producers[rank].launch(&sessions[rank], &states[rank].live, count)?;
                        launch_entry(
                            sessions,
                            &served,
                            entry.symbol(),
                            &states,
                            &rows,
                            entry,
                            Shape::new(count, true, geometry),
                            Some(rank),
                        )?;
                    }
                }
                Ok(())
            };
            poison_outputs(sessions, &rows, Entry::TwoShot)?;
            behind(None)?;
            check(&format!("{label} PDL producer"))?;
            let graphs = RankReplay::new(sessions, |rank, _| behind(Some(rank)))?;
            poison_outputs(sessions, &rows, Entry::TwoShot)?;
            graphs.launch(sessions)?;
            check(&format!("{label} PDL producer graph"))?;
            for session in sessions {
                receipt(
                    session,
                    vec![
                        ("case", json!("phased_bf16_replays")),
                        ("ranks", json!(ranks)),
                        ("rows", json!(count)),
                        ("senders", json!(geometry.0)),
                        ("workers", json!(geometry.1)),
                        ("entries", json!(label)),
                        (
                            "phases",
                            json!(["eager", "relaunch", "graph", "pdl", "pdl_graph"]),
                        ),
                        ("bitwise", json!(true)),
                    ],
                )?;
            }
        }
    }
    Ok(())
}

#[test]
fn phased_bf16_replays() {
    gpu_test_multi(1 << 30, run_replays).expect("phased BF16 all-reduce replays");
}

/// Whether some rank's output (or, for `no_gather_wait`, its snapshot
/// taken right after the call on its stream) differs from the host.
fn differs(
    sessions: &[Session],
    rows: &[Rows],
    snapshots: &[DeviceBuffer],
    expected: &[u16],
    snapshot: bool,
) -> LabResult<bool> {
    synchronize(sessions)?;
    for (rank, session) in sessions.iter().enumerate() {
        let buffer = if snapshot {
            &snapshots[rank]
        } else {
            &rows[rank].output
        };
        let actual = halves(&poison::read(session, buffer)?[..expected.len() * 2]);
        if first_difference(&actual, expected).is_some() {
            return Ok(true);
        }
    }
    Ok(false)
}

fn run_mutants(sessions: &[Session]) -> LabResult<()> {
    let ranks = sessions.len();
    let served = CollectiveModules::load(sessions, phased_bf16::LAB_CUBIN)?;
    let slow = CollectiveModules::load(
        sessions,
        &phased_bf16::mutant_cubin(phased_bf16::SLOW_CONTROL),
    )?;
    let states = states(sessions, 2048)?;
    let mut missed = Vec::new();
    // (entry, rows, geometry): every case is required.
    let cases = [
        (Entry::TwoShot, 2048, (16, 32)),
        (Entry::TwoShot, 32, (16, 0)),
        (Entry::OneShot, 4, (16, 0)),
        (Entry::OneShot, 32, (4, 4)),
    ];
    for (seed, (entry, count, geometry)) in cases.into_iter().enumerate() {
        let inputs = inputs_for(ranks, 60 + seed as u64, count);
        let expected = host_sums(&inputs, count * WIDTH);
        let rows = rows_buffers(sessions, &inputs)?;
        let snapshots = sessions
            .iter()
            .map(|session| session.allocate(count * WIDTH * 2))
            .collect::<LabResult<Vec<_>>>()?;
        let producers = sessions
            .iter()
            .zip(&rows)
            .map(|(session, rows)| Producer::new(session, rows))
            .collect::<LabResult<Vec<_>>>()?;
        set_live(sessions, &states, count)?;
        let shape = Shape::new(count, false, geometry);
        for (tag, _) in phased_bf16::MUTANTS {
            if tag == "no_gather_wait" && entry.one_shot() {
                // The one-shot has no gather.
                continue;
            }
            if tag == "shared_parity" {
                // A reuse defect across calls: phased_bf16_generations.
                continue;
            }
            if tag == "reversed" && ranks < 3 {
                // (0 + x0) + x1 and (0 + x1) + x0 round alike.
                eprintln!("mutant reversed at {count} rows: inapplicable at {ranks} ranks");
                continue;
            }
            let modules = CollectiveModules::load(sessions, &phased_bf16::mutant_cubin(tag))?;
            let symbol = phased_bf16::mutant_symbol(entry.symbol(), tag);
            let snapshot = tag == "no_gather_wait";
            // One call per rank (`only_rank`), behind the producer for the
            // late-wait mutant, then the snapshot copy on the same stream.
            let call = |only_rank: Option<usize>,
                        modules: &CollectiveModules,
                        symbol: &str|
             -> LabResult<()> {
                for rank in 0..ranks {
                    if only_rank.is_some_and(|only| only != rank) {
                        continue;
                    }
                    if tag == "late_wait" {
                        producers[rank].launch(&sessions[rank], &states[rank].live, count)?;
                    }
                    launch_entry(
                        sessions,
                        modules,
                        symbol,
                        &states,
                        &rows,
                        entry,
                        shape,
                        Some(rank),
                    )?;
                    if snapshot {
                        snapshots[rank]
                            .copy_range_from_async(
                                0,
                                &rows[rank].output,
                                0,
                                count * WIDTH * 2,
                                &sessions[rank].stream,
                            )
                            .map_err(err)?;
                    }
                }
                Ok(())
            };
            // A fresh state with poisoned inboxes and outputs per run: a
            // mutant may leave flags no later call can follow.
            let fresh = || -> LabResult<()> {
                reset(sessions, &states)?;
                for (session, (state, rows)) in sessions.iter().zip(states.iter().zip(&rows)) {
                    poison::poison(session, &[&state.inbox, &rows.output])?;
                }
                for (session, snapshot) in sessions.iter().zip(&snapshots) {
                    poison::poison(session, &[snapshot])?;
                }
                Ok(())
            };
            let label = format!("{} {tag} at {count} rows, {geometry:?}", entry.name());
            // The served entry passes the same harness, with the same slow
            // rank for the mutants that need one.
            let (control, control_symbol) = if phased_bf16::SLOW_MUTANTS.contains(&tag) {
                (
                    &slow,
                    phased_bf16::mutant_symbol(entry.symbol(), phased_bf16::SLOW_CONTROL),
                )
            } else {
                (&served, entry.symbol().to_owned())
            };
            for _ in 0..2 {
                fresh()?;
                call(None, control, &control_symbol)?;
                if differs(sessions, &rows, &snapshots, &expected, snapshot)? {
                    return Err(format!("{label}: the served entry differs in this harness"));
                }
            }
            let mut eager = 0;
            for trial in 1..=MUTANT_TRIALS {
                fresh()?;
                call(None, &modules, &symbol)?;
                if differs(sessions, &rows, &snapshots, &expected, snapshot)? {
                    eager = trial;
                    break;
                }
            }
            let graphs = RankReplay::new(sessions, |rank, _| call(Some(rank), &modules, &symbol))?;
            let mut replay = 0;
            for trial in 1..=MUTANT_TRIALS {
                fresh()?;
                graphs.launch(sessions)?;
                if differs(sessions, &rows, &snapshots, &expected, snapshot)? {
                    replay = trial;
                    break;
                }
            }
            eprintln!(
                "mutant {label}: eager caught at trial {eager}, replay caught at trial {replay} (0: missed)"
            );
            for session in sessions {
                receipt(
                    session,
                    vec![
                        ("case", json!("phased_bf16_mutants_are_caught")),
                        ("ranks", json!(ranks)),
                        ("rows", json!(count)),
                        ("entry", json!(entry.name())),
                        ("senders", json!(geometry.0)),
                        ("workers", json!(geometry.1)),
                        ("mutant", json!(tag)),
                        ("eager_trial", json!(eager)),
                        ("replay_trial", json!(replay)),
                    ],
                )?;
            }
            if eager == 0 || replay == 0 {
                missed.push(format!("{label} (eager {eager}, replay {replay})"));
            }
        }
        reset(sessions, &states)?;
    }
    if missed.is_empty() {
        Ok(())
    } else {
        Err(format!("mutants not caught: {}", missed.join(", ")))
    }
}

#[test]
fn phased_bf16_mutants_are_caught() {
    gpu_test_multi(1 << 30, run_mutants).expect("phased BF16 all-reduce mutants");
}

const ONE_SHOT_GENERATIONS: [Entry; 8] = [
    Entry::OneShot,
    Entry::OneShot,
    Entry::TwoShot,
    Entry::OneShot,
    Entry::OneShot,
    Entry::TwoShot,
    Entry::OneShot,
    Entry::OneShot,
];
const TWO_SHOT_GENERATIONS: [Entry; 4] = [Entry::TwoShot; 4];
/// Clock cycles the delayed peer spins before its call (about 100 us).
const DELAY_CYCLES: u64 = 200_000;

/// One rank's buffers of a generations case: each generation's partials,
/// the one output every call writes, and each generation's snapshot of its
/// result, copied on the rank's stream right after the call.
struct Generations {
    inputs: Vec<DeviceBuffer>,
    output: DeviceBuffer,
    snapshots: Vec<DeviceBuffer>,
}

fn generation_bindings<'a>(
    states: &'a [State],
    ranks: &'a [Generations],
    generation: usize,
    rank: usize,
) -> Bindings<'a> {
    let mut buffers: BTreeMap<u64, &DeviceBuffer> = BTreeMap::new();
    buffers.insert(INPUT, &ranks[rank].inputs[generation]);
    buffers.insert(OUTPUT, &ranks[rank].output);
    buffers.insert(EPOCH, &states[rank].epoch);
    buffers.insert(LIVE, &states[rank].live);
    for peer in 0..states.len() {
        let q = peer as u64;
        buffers.insert(INBOXES + q, &states[peer].inbox);
        buffers.insert(FLAGS + q, &states[peer].flags);
        buffers.insert(OUTPUTS + q, &ranks[peer].output);
    }
    Bindings { buffers }
}

/// Words of every generation's snapshot on every rank that differ from
/// that generation's own host result.
fn generation_wrong_words(
    sessions: &[Session],
    ranks: &[Generations],
    expected: &[Vec<u16>],
) -> LabResult<usize> {
    synchronize(sessions)?;
    let mut wrong = 0;
    for (session, rank) in sessions.iter().zip(ranks) {
        for (snapshot, sums) in rank.snapshots.iter().zip(expected) {
            let actual = halves(&poison::read(session, snapshot)?);
            wrong += actual.iter().zip(sums).filter(|(a, e)| a != e).count();
        }
    }
    Ok(wrong)
}

fn run_generations(sessions: &[Session]) -> LabResult<()> {
    let ranks = sessions.len();
    let served = CollectiveModules::load(sessions, phased_bf16::LAB_CUBIN)?;
    let mutant = CollectiveModules::load(sessions, &phased_bf16::mutant_cubin("shared_parity"))?;
    let slow = CollectiveModules::load(
        sessions,
        &phased_bf16::mutant_cubin(phased_bf16::SLOW_CONTROL),
    )?;
    let spins = sessions
        .iter()
        .map(|session| {
            session
                .module(&kernel_lab::latent_shapes::cubin_name())?
                .function(&kernel_lab::latent_shapes::pdl_copy_symbol())
                .map_err(err)
        })
        .collect::<LabResult<Vec<_>>>()?;
    let scratch = sessions
        .iter()
        .map(|session| session.allocate(16))
        .collect::<LabResult<Vec<_>>>()?;
    let states = states(sessions, 2048)?;
    let cases: [(usize, (u32, u32), &[Entry]); 5] = [
        (4, (16, 0), &ONE_SHOT_GENERATIONS),
        (32, (4, 4), &ONE_SHOT_GENERATIONS),
        (64, (1, 0), &ONE_SHOT_GENERATIONS),
        (64, (16, 0), &ONE_SHOT_GENERATIONS),
        (2048, (16, 32), &TWO_SHOT_GENERATIONS),
    ];
    let mut failures = Vec::new();
    for (case, &(count, geometry, sequence)) in cases.iter().enumerate() {
        let elements = count * WIDTH;
        let seed = |generation: usize| 300 + (case * 16 + generation) as u64;
        let expected = (0..sequence.len())
            .map(|generation| host_sums(&inputs_for(ranks, seed(generation), count), elements))
            .collect::<Vec<_>>();
        for generation in 1..sequence.len() {
            if expected[generation] == expected[generation - 1] {
                return Err("consecutive generations must differ".into());
            }
        }
        let buffers = sessions
            .iter()
            .enumerate()
            .map(|(rank, session)| {
                Ok(Generations {
                    inputs: (0..sequence.len())
                        .map(|generation| {
                            session.upload_f32(&partial(seed(generation), rank, elements))
                        })
                        .collect::<LabResult<Vec<_>>>()?,
                    output: session.allocate(elements * 2)?,
                    snapshots: sequence
                        .iter()
                        .map(|_| session.allocate(elements * 2))
                        .collect::<LabResult<Vec<_>>>()?,
                })
            })
            .collect::<LabResult<Vec<_>>>()?;
        set_live(sessions, &states, count)?;
        // The sequence through `modules`, its symbols tagged `tag` (a lab
        // cubin's) or the served ones.
        let sequence_on =
            |modules: &CollectiveModules, tag: Option<&str>, only_rank: Option<usize>| {
                for (generation, &entry) in sequence.iter().enumerate() {
                    let delayed = generation % ranks;
                    if only_rank.is_none_or(|only| only == delayed) {
                        spins[delayed]
                            .launch(
                                &sessions[delayed].stream,
                                launch::simple_config(
                                    (1, 1, 1),
                                    (32, 1, 1),
                                    0,
                                    LaunchConfig::NO_CLUSTER,
                                    false,
                                )?,
                                &[
                                    KernelArgument::Buffer(&scratch[delayed]),
                                    KernelArgument::Buffer(&scratch[delayed]),
                                    KernelArgument::U64(0),
                                    KernelArgument::U64(DELAY_CYCLES),
                                ],
                            )
                            .map_err(err)?;
                    }
                    let symbol = match tag {
                        Some(tag) => phased_bf16::mutant_symbol(entry.symbol(), tag),
                        None => entry.symbol().to_owned(),
                    };
                    launch_bound(
                        sessions,
                        modules,
                        &symbol,
                        entry,
                        Shape::new(count, true, geometry),
                        only_rank,
                        |rank| generation_bindings(&states, &buffers, generation, rank),
                    )?;
                    for (rank, session) in sessions.iter().enumerate() {
                        if only_rank.is_some_and(|only| only != rank) {
                            continue;
                        }
                        buffers[rank].snapshots[generation]
                            .copy_range_from_async(
                                0,
                                &buffers[rank].output,
                                0,
                                elements * 2,
                                &session.stream,
                            )
                            .map_err(err)?;
                    }
                }
                LabResult::Ok(())
            };
        let fresh = || -> LabResult<()> {
            reset(sessions, &states)?;
            for (session, (state, rank)) in sessions.iter().zip(states.iter().zip(&buffers)) {
                poison::poison(session, &[&state.inbox, &rank.output])?;
                for snapshot in &rank.snapshots {
                    poison::poison(session, &[snapshot])?;
                }
            }
            Ok(())
        };
        let label = format!("{count} rows, {geometry:?}");
        // The served entries, then the same entries with the slow rank
        // (the mutant's timing without its defect): eager, then replayed.
        let mut served_wrong = Vec::new();
        for (modules, tag) in [(&served, None), (&slow, Some(phased_bf16::SLOW_CONTROL))] {
            fresh()?;
            sequence_on(modules, tag, None)?;
            served_wrong.push(generation_wrong_words(sessions, &buffers, &expected)?);
            let graphs =
                RankReplay::new(sessions, |rank, _| sequence_on(modules, tag, Some(rank)))?;
            for _ in 0..3 {
                fresh()?;
                graphs.launch(sessions)?;
                served_wrong.push(generation_wrong_words(sessions, &buffers, &expected)?);
            }
        }
        if served_wrong.iter().any(|&wrong| wrong > 0) {
            failures.push(format!("{label}: served wrong words {served_wrong:?}"));
        }
        // The shared-parity mutant (a one-shot defect, built with the slow
        // rank) must show wrong words in each mode on its own: eager, then
        // replayed, up to MUTANT_TRIALS fresh runs each.
        let mut mutant_counts = Vec::new();
        let mut detected = (0, 0);
        if sequence.iter().any(|entry| entry.one_shot()) {
            let tag = Some("shared_parity");
            for trial in 1..=MUTANT_TRIALS {
                fresh()?;
                sequence_on(&mutant, tag, None)?;
                let wrong = generation_wrong_words(sessions, &buffers, &expected)?;
                mutant_counts.push(("eager", trial, wrong));
                if wrong > 0 {
                    detected.0 = trial;
                    break;
                }
            }
            let graphs =
                RankReplay::new(sessions, |rank, _| sequence_on(&mutant, tag, Some(rank)))?;
            for trial in 1..=MUTANT_TRIALS {
                fresh()?;
                graphs.launch(sessions)?;
                let wrong = generation_wrong_words(sessions, &buffers, &expected)?;
                mutant_counts.push(("replay", trial, wrong));
                if wrong > 0 {
                    detected.1 = trial;
                    break;
                }
            }
            if detected.0 == 0 || detected.1 == 0 {
                failures.push(format!(
                    "{label}: the shared-parity mutant was not observed in each mode (eager {}, replay {})",
                    detected.0, detected.1
                ));
            }
        }
        eprintln!(
            "generations {label}: served and slow-rank wrong words {served_wrong:?}; shared_parity mutant (mode, trial, wrong words) {mutant_counts:?}"
        );
        for session in sessions {
            receipt(
                session,
                vec![
                    ("case", json!("phased_bf16_generations")),
                    ("ranks", json!(ranks)),
                    ("rows", json!(count)),
                    ("senders", json!(geometry.0)),
                    ("workers", json!(geometry.1)),
                    (
                        "sequence",
                        json!(
                            sequence
                                .iter()
                                .map(|entry| entry.name())
                                .collect::<Vec<_>>()
                        ),
                    ),
                    ("served_wrong_words", json!(served_wrong)),
                    ("mutant_wrong_words", json!(mutant_counts)),
                    ("mutant_eager_trial", json!(detected.0)),
                    ("mutant_replay_trial", json!(detected.1)),
                ],
            )?;
        }
        reset(sessions, &states)?;
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("; "))
    }
}

#[test]
fn phased_bf16_generations() {
    gpu_test_multi(1 << 30, run_generations).expect("phased BF16 all-reduce generations");
}

fn env_list<T: std::str::FromStr + Clone>(name: &str, default: &[T]) -> Vec<T> {
    std::env::var(name)
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(|item| {
                    item.trim()
                        .parse()
                        .map_err(|_| format!("{name}: {item:?} is not a number"))
                        .expect("a comma-separated number list")
                })
                .collect()
        })
        .unwrap_or_else(|| default.to_vec())
}

/// Microseconds per call, critical path over ranks, of `calls` repeated
/// calls in one graph per rank.
fn time_calls(
    sessions: &[Session],
    calls: usize,
    one: impl Fn(usize) -> LabResult<()>,
) -> LabResult<(f32, f32)> {
    let graphs = sessions
        .iter()
        .enumerate()
        .map(|(rank, session)| {
            session.capture(|_| {
                for _ in 0..calls {
                    one(rank)?;
                }
                Ok(())
            })
        })
        .collect::<LabResult<Vec<_>>>()?;
    let timing = time_multi_graph(sessions, &graphs, calls, 2, 5)?;
    let per = |ms: f32| ms * 1000.0 / calls as f32;
    Ok((
        per(timing.critical_path.median_ms),
        per(timing.critical_path.min_ms),
    ))
}

/// The (senders, workers) the timing sweeps at `rows`:
/// `CHOPINFER_KERNEL_BENCH_SENDERS` x `CHOPINFER_KERNEL_BENCH_WORKERS` when
/// either is set, else a small-call or a large-call list.
fn timing_geometries(rows: usize) -> Vec<(u32, u32)> {
    let senders = std::env::var("CHOPINFER_KERNEL_BENCH_SENDERS").is_ok();
    let workers = std::env::var("CHOPINFER_KERNEL_BENCH_WORKERS").is_ok();
    if senders || workers {
        let senders = env_list("CHOPINFER_KERNEL_BENCH_SENDERS", &[16]);
        let workers = env_list("CHOPINFER_KERNEL_BENCH_WORKERS", &[32]);
        return workers
            .iter()
            .flat_map(|&w| senders.iter().map(move |&s| (s, w)))
            .collect();
    }
    if rows <= 64 {
        vec![(4, 0), (8, 0), (16, 0), (32, 0), (4, 4), (8, 8)]
    } else {
        let mut geometries: Vec<(u32, u32)> = [8, 16, 24, 32, 48, 64]
            .into_iter()
            .map(|s| (s, 32))
            .collect();
        // The overlapped prefill lane's grids of 16 to 32 CTAs.
        geometries.extend([(8, 8), (8, 16), (16, 8), (16, 16), (8, 24)]);
        geometries
    }
}

fn run_timing(sessions: &[Session]) -> LabResult<()> {
    let ranks = sessions.len();
    let served = CollectiveModules::load(sessions, phased_bf16::LAB_CUBIN)?;
    let reference = CollectiveModules::load(sessions, phased_f32::LAB_CUBIN)?;
    let pair = CollectiveModules::load(sessions, "comm_sm120")?;
    let all_rows = env_list(
        "CHOPINFER_KERNEL_BENCH_ROWS",
        &[
            1, 2, 3, 4, 5, 8, 16, 24, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
        ],
    );
    let capacity = *all_rows.iter().max().ok_or("no benchmark rows")?;
    let states = states(sessions, capacity)?;
    for &count in &all_rows {
        let calls = (8192 / count).clamp(4, 64);
        let inputs = vec![vec![0.0_f32; count * WIDTH]; ranks];
        let rows = rows_buffers(sessions, &inputs)?;
        set_live(sessions, &states, count)?;
        let bytes = (count * WIDTH * 4) as f32;
        let share = (ranks - 1) as f32 / ranks as f32;
        let record = |variant: &str,
                      senders: u32,
                      workers: u32,
                      sent: f32,
                      (median, min): (f32, f32)|
         -> LabResult<()> {
            let gbps = sent / median / 1000.0;
            eprintln!(
                "phased_bf16 timing: rows {count:5} {variant:18} senders {senders:2} workers {workers:2}: median {median:9.2} us, min {min:9.2} us, {gbps:6.2} GB/s sent per rank"
            );
            for session in sessions {
                receipt(
                    session,
                    vec![
                        ("case", json!("phased_bf16_timing")),
                        ("ranks", json!(ranks)),
                        ("rows", json!(count)),
                        ("variant", json!(variant)),
                        ("senders", json!(senders)),
                        ("workers", json!(workers)),
                        ("calls_per_graph", json!(calls)),
                        ("median_us_per_call", json!(median)),
                        ("min_us_per_call", json!(min)),
                        ("sent_gb_per_s_per_rank", json!(gbps)),
                    ],
                )?;
            }
            Ok(())
        };
        let timing = time_calls(sessions, calls, |rank| {
            launch_pair(sessions, &pair, &states, &rows, count, INPUT, Some(rank))
        })?;
        record("f32_pair", 8, 0, 2.0 * share * bytes, timing)?;
        for (senders, workers) in timing_geometries(count) {
            for entry in [
                Entry::TwoShot,
                Entry::F32Round,
                Entry::OneShot,
                Entry::F32OneShotRound,
            ]
            .into_iter()
            .filter(|e| e.takes(count))
            {
                let shape = Shape::new(count, true, (senders, workers));
                let modules = if entry.reference() {
                    &reference
                } else {
                    &served
                };
                let timing = time_calls(sessions, calls, |rank| {
                    launch_entry(
                        sessions,
                        modules,
                        entry.symbol(),
                        &states,
                        &rows,
                        entry,
                        shape,
                        Some(rank),
                    )
                })?;
                let sent = match entry {
                    Entry::TwoShot => share * bytes,
                    Entry::F32Round => 1.5 * share * bytes,
                    Entry::OneShot => (ranks - 1) as f32 * bytes / 2.0,
                    Entry::F32OneShotRound => (ranks - 1) as f32 * bytes,
                };
                record(entry.name(), senders, workers, sent, timing)?;
            }
        }
    }
    Ok(())
}

#[test]
fn phased_bf16_timing() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        eprintln!(
            "skipping phased BF16 all-reduce timing: set CHOPINFER_KERNEL_BENCH=1 on idle GPUs"
        );
        return;
    }
    gpu_test_multi(2 << 30, run_timing).expect("phased BF16 all-reduce timing");
}

#[test]
fn phased_bf16_cpu_contract() {
    let entries = kernel_lab::cubin_entries(phased_bf16::LAB_CUBIN).unwrap();
    assert_eq!(entries, [phased_bf16::SYMBOL, phased_bf16::ONE_SHOT_SYMBOL]);
    for (tag, _) in phased_bf16::MUTANTS {
        assert!(kernel_lab::comm_cubin(&phased_bf16::mutant_cubin(tag)).unwrap());
    }
    // The host fold rounds every rank's partial before the sum.
    let inputs = inputs_for(4, 9, 2);
    let expected = host_sums(&inputs, 2 * WIDTH);
    let unrounded: Vec<u16> = (0..2 * WIDTH)
        .map(|i| f32_to_bf16(inputs.iter().fold(0.0_f32, |sum, rank| sum + rank[i])))
        .collect();
    assert_ne!(expected, unrounded);
    let reversed: Vec<_> = inputs.iter().rev().cloned().collect();
    assert_ne!(host_sums(&reversed, 2 * WIDTH), expected);
    assert_eq!(rounded(&inputs[0]).len(), 2 * WIDTH);
}
