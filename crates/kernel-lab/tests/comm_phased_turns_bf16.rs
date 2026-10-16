//! The phased all-reduce of BF16 partials whose writers take turns at each
//! destination (`kernels-sm120/cuda/comm/allreduce_phased_turns_bf16.cuh`,
//! lock family `comm_phased_turns_bf16`) on two or more GPUs
//! (`CHOPINFER_KERNEL_GPUS`). The host's fold is the reference: every
//! rank's F32 partial rounded once to BF16, the BF16 values summed in rank
//! order from F32 zero, the sum rounded once to BF16; the phased BF16
//! two-shot (`comm_phased_bf16`, served before this entry) must give the
//! same bits.
//!
//! - `phased_turns_bf16_matches_host_and_current_entry`: one operand set of
//!   8192 rows per rank, so row r of every case must equal one host fold.
//!   Every live count from 1 to 8192 rows on an 8192-row call at the
//!   served geometry (16 senders, 32 workers); edge counts at other sender
//!   and worker counts (with none, the comm lane's narrowed grids, and up
//!   to 64 of each); calls without live rows. Each case runs this entry
//!   and the phased BF16 two-shot on one shared state, into outputs whose
//!   bytes are poison before the call: this entry's whole output is read
//!   back (the live rows must hold the host's bits, every other byte its
//!   poison), the two-shot's live rows must hold the same bits (its write
//!   map is `comm_phased_bf16`'s matrix's). The last receipt records the
//!   case count.
//! - `phased_turns_bf16_replays`: poisoned outputs, eager, relaunched, from
//!   a graph (alone and between the phased BF16 entries on one state) and
//!   behind a PDL producer that zeroes the partials and the live count
//!   until after its tail, eager and captured.
//! - `phased_turns_bf16_mutants_are_caught`: each lab mutant changes some
//!   rank's output in an eager call and in a graph replay, separately, at
//!   2048 rows with workers and at 64 rows without. A mutant counts only on
//!   an observed difference in one of up to five fresh runs per mode; an
//!   execution error fails the test. The missing gather wait is built with
//!   the slow rank (rank 1 sleeps after its partials went out), and the
//!   served entry with the same slow rank must stay exact. The missing
//!   scatter wait is built with the late scatter (rank 1 sleeps before its
//!   last scatter phase, which no turn of its destination waits for: at
//!   four ranks without workers a rank sums after its first gather turn,
//!   which follows ranks 0, 3 and 2), and the served entry with the same
//!   late scatter must stay exact.
//! - `phased_turns_bf16_generations`: back-to-back calls on one state, each
//!   with distinct partials, mixing this entry with the phased BF16
//!   one-shot and two-shot, a different peer delayed before each call,
//!   every result snapshotted on the device right after its call; eager and
//!   replayed, with and without the slow rank. No wrong word.
//! - `phased_turns_bf16_timing` (`CHOPINFER_KERNEL_BENCH=1`): microseconds
//!   per call of this entry, its timing variants and the phased BF16
//!   two-shot, per row count and sender and worker count, in graphs of
//!   repeated calls on every rank, critical path over ranks, with ranks
//!   that start together and with injected skew: before each call one rank
//!   (rotating) spins, or every rank spins a staggered share; the spin's
//!   own time is reported beside.
//! - `phased_turns_bf16_link_patterns` (`CHOPINFER_KERNEL_BENCH=1`): the
//!   PCIe write rate when every rank pushes to rank + d at once (d = 1 to
//!   ranks - 1), one pair alone, and two or three ranks pushing into one.

use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::ArgumentList;
use kernel_lab::launch::{self, Bindings, CollectiveModules, RankCall};
use kernel_lab::poison::{self, POISON};
use kernel_lab::topology::RankReplay;
use kernel_lab::{LabResult, Session, gpu_test_multi, time_multi_graph};
use kernels_sm120::comm::{phased_bf16, phased_turns_bf16 as turns};
use runtime_cuda::{DeviceBuffer, KernelArgument, LaunchConfig};
use serde_json::json;
use std::collections::BTreeMap;

const WIDTH: usize = 4096;
/// Rows of the matrix's one operand set.
const OPERAND_ROWS: usize = 8192;
/// The served geometry (senders, workers).
const SERVED: (u32, u32) = (16, 32);
/// Other geometries of the matrix's edge counts: no workers, the comm
/// lane's narrowed grids, more senders and workers.
const GEOMETRIES: [(u32, u32); 10] = [
    (1, 0),
    (4, 0),
    (16, 0),
    (1, 1),
    (8, 8),
    (16, 8),
    (16, 16),
    (24, 32),
    (32, 32),
    (64, 64),
];
const EDGE_ROWS: [usize; 19] = [
    1, 2, 3, 4, 31, 32, 33, 64, 255, 256, 1023, 1024, 2047, 2048, 3008, 4095, 4096, 8191, 8192,
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
const REF_OUTPUT: u64 = 7;
const INBOXES: u64 = 100;
const FLAGS: u64 = 200;
const OUTPUTS: u64 = 300;
const REF_OUTPUTS: u64 = 400;

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

fn inputs_for(ranks: usize, seed: u64, rows: usize) -> Vec<Vec<f32>> {
    (0..ranks)
        .map(|rank| partial(seed, rank, rows * WIDTH))
        .collect()
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

fn bytes_of(halves: &[u16]) -> Vec<u8> {
    halves.iter().flat_map(|half| half.to_ne_bytes()).collect()
}

/// The first byte where `actual` differs from `expected`.
fn first_difference(actual: &[u8], expected: &[u8]) -> Option<usize> {
    if actual == expected {
        return None;
    }
    actual
        .iter()
        .zip(expected)
        .position(|(a, e)| a != e)
        .or(Some(actual.len().min(expected.len())))
}

/// The first byte of `bytes` that is not poison.
fn first_written(bytes: &[u8]) -> Option<usize> {
    let word = u64::from_ne_bytes([POISON; 8]);
    let words = bytes.chunks_exact(8);
    let tail = words.remainder();
    if let Some(at) = words
        .map(|chunk| u64::from_ne_bytes(chunk.try_into().expect("eight bytes")))
        .position(|w| w != word)
    {
        let start = at * 8;
        return bytes[start..start + 8]
            .iter()
            .position(|&b| b != POISON)
            .map(|offset| start + offset);
    }
    tail.iter()
        .position(|&b| b != POISON)
        .map(|at| bytes.len() - tail.len() + at)
}

/// One rank's persistent state of `capacity` rows (this entry's layout,
/// which the phased BF16 entries share), and the live count.
struct State {
    inbox: DeviceBuffer,
    flags: DeviceBuffer,
    epoch: DeviceBuffer,
    live: DeviceBuffer,
}

fn states(sessions: &[Session], capacity: usize) -> LabResult<Vec<State>> {
    let ranks = sessions.len() as u32;
    let elements = (capacity * WIDTH) as u64;
    let own = turns::buffers(ranks, elements).map_err(err)?;
    let phased = phased_bf16::buffers(ranks, elements).map_err(err)?;
    if own.inbox_bytes != phased.inbox_bytes || own.flag_bytes <= phased.flag_bytes {
        return Err("the turns state does not hold the phased BF16 entries' calls".into());
    }
    let states = sessions
        .iter()
        .map(|session| {
            Ok(State {
                inbox: session.allocate(own.inbox_bytes as usize)?,
                flags: session.allocate(own.flag_bytes as usize)?,
                epoch: session.allocate(own.epoch_bytes as usize)?,
                live: session.allocate(4)?,
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

/// Zeroes every rank's flags and epoch once every rank is idle.
fn reset(sessions: &[Session], states: &[State]) -> LabResult<()> {
    poison::synchronize(sessions)?;
    for (session, state) in sessions.iter().zip(states) {
        zero(session, &state.flags)?;
        zero(session, &state.epoch)?;
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

/// One rank's buffers of one operand set.
struct Rows {
    input: DeviceBuffer,
    staged: DeviceBuffer,
    output: DeviceBuffer,
    ref_output: DeviceBuffer,
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
                ref_output: session.allocate(elements * 2)?,
            })
        })
        .collect()
}

fn bindings<'a>(states: &'a [State], rows: &'a [Rows], rank: usize) -> Bindings<'a> {
    let mut buffers: BTreeMap<u64, &DeviceBuffer> = BTreeMap::new();
    buffers.insert(INPUT, &rows[rank].input);
    buffers.insert(OUTPUT, &rows[rank].output);
    buffers.insert(REF_OUTPUT, &rows[rank].ref_output);
    buffers.insert(EPOCH, &states[rank].epoch);
    buffers.insert(LIVE, &states[rank].live);
    for peer in 0..states.len() {
        let q = peer as u64;
        buffers.insert(INBOXES + q, &states[peer].inbox);
        buffers.insert(FLAGS + q, &states[peer].flags);
        buffers.insert(OUTPUTS + q, &rows[peer].output);
        buffers.insert(REF_OUTPUTS + q, &rows[peer].ref_output);
    }
    Bindings { buffers }
}

/// Which entry a call runs: this header's (into `output`), or the phased
/// BF16 two-shot or one-shot (into `ref_output`).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Entry {
    Turns,
    TwoShot,
    OneShot,
}

impl Entry {
    fn symbol(self) -> &'static str {
        match self {
            Entry::Turns => turns::SYMBOL,
            Entry::TwoShot => phased_bf16::SYMBOL,
            Entry::OneShot => phased_bf16::ONE_SHOT_SYMBOL,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Entry::Turns => "turns_bf16",
            Entry::TwoShot => "bf16",
            Entry::OneShot => "oneshot_bf16",
        }
    }

    fn reference(self) -> bool {
        self != Entry::Turns
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
    let (output, outputs) = match entry.reference() {
        true => (REF_OUTPUT, REF_OUTPUTS),
        false => (OUTPUT, OUTPUTS),
    };
    let call = phased_bf16::Call {
        input: INPUT,
        output,
        inboxes: peers(INBOXES),
        outputs: peers(outputs),
        flags: peers(FLAGS),
        epoch: EPOCH,
        elements: (shape.rows * WIDTH) as u64,
        row_elements: WIDTH as u32,
        live_rows: shape.live.then_some(LIVE),
        rank: rank as u32,
        ranks: ranks as u32,
        senders: shape.senders,
    };
    match entry {
        Entry::Turns => turns::arguments(call),
        _ => phased_bf16::arguments(entry.symbol(), call),
    }
    .map_err(err)
}

/// One call of `symbol` (`entry`'s symbol in `modules`, served or a lab
/// variant) on every rank, or on `only_rank`, with the handles bound by
/// `bind(rank)`.
fn launch_bound<'a>(
    sessions: &[Session],
    modules: &CollectiveModules,
    symbol: &str,
    entry: Entry,
    shape: Shape,
    only_rank: Option<usize>,
    bind: impl Fn(usize) -> Bindings<'a>,
) -> LabResult<()> {
    let plan = turns::plan_with(shape.senders, shape.workers).map_err(err)?;
    launch::launch_collective(sessions, modules, symbol, plan, only_rank, |rank| {
        Ok(RankCall {
            args: entry_args(rank, sessions.len(), entry, shape)?,
            bindings: bind(rank),
        })
    })
}

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

/// The served cubins: this entry's and the phased BF16 entries'.
struct Served {
    turns: CollectiveModules,
    phased: CollectiveModules,
}

impl Served {
    fn load(sessions: &[Session]) -> LabResult<Self> {
        Ok(Self {
            turns: CollectiveModules::load(sessions, turns::LAB_CUBIN)?,
            phased: CollectiveModules::load(sessions, phased_bf16::LAB_CUBIN)?,
        })
    }

    fn modules(&self, entry: Entry) -> &CollectiveModules {
        match entry {
            Entry::Turns => &self.turns,
            _ => &self.phased,
        }
    }
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

fn read_prefix(session: &Session, buffer: &DeviceBuffer, bytes: usize) -> LabResult<Vec<u8>> {
    let mut host = vec![0; bytes];
    if bytes == 0 {
        return Ok(host);
    }
    buffer
        .copy_to_bytes_at_async(0, &mut host, &session.stream)
        .map_err(err)?;
    session.stream.synchronize().map_err(err)?;
    Ok(host)
}

/// Every rank's output of `entry` against `expected` (the host's bytes):
/// the first `live` rows written with the expected bits, and, when
/// `whole`, every other byte of the output still poison.
fn check_outputs(
    sessions: &[Session],
    rows: &[Rows],
    entry: Entry,
    expected: &[u8],
    live: usize,
    whole: bool,
    label: &str,
) -> LabResult<()> {
    let written = live * WIDTH * 2;
    for (rank, (session, rows)) in sessions.iter().zip(rows).enumerate() {
        let buffer = output_of(rows, entry);
        let bytes = match whole {
            true => read_prefix(session, buffer, buffer.byte_len())?,
            false => read_prefix(session, buffer, written)?,
        };
        if let Some(at) = first_difference(&bytes[..written], &expected[..written]) {
            return Err(format!(
                "{label}, {}, rank {rank}: rounded sum differs at element {}",
                entry.name(),
                at / 2
            ));
        }
        if let Some(at) = first_written(&bytes[written..]) {
            return Err(format!(
                "{label}, {}, rank {rank}: byte {} past the live rows was written",
                entry.name(),
                written + at
            ));
        }
    }
    Ok(())
}

fn receipt(session: &Session, data: Vec<(&str, serde_json::Value)>) -> LabResult<()> {
    session.receipt(
        "comm_phased_turns_bf16",
        data.into_iter().map(|(k, v)| (k.to_owned(), v)).collect(),
    )?;
    Ok(())
}

/// A matrix case: a call of `capacity` rows reducing `live` rows (or every
/// row, without a live count) in `geometry`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Case {
    capacity: usize,
    live: Option<usize>,
    geometry: (u32, u32),
}

impl Case {
    fn written(&self) -> usize {
        self.live
            .map_or(self.capacity, |live| live.min(self.capacity))
    }
}

fn matrix() -> Vec<Case> {
    let mut cases: Vec<Case> = (1..=OPERAND_ROWS)
        .map(|live| Case {
            capacity: OPERAND_ROWS,
            live: Some(live),
            geometry: SERVED,
        })
        .collect();
    for geometry in GEOMETRIES {
        for rows in EDGE_ROWS {
            cases.push(Case {
                capacity: rows,
                live: Some(rows),
                geometry,
            });
        }
    }
    for capacity in [1, 33, 2048, 3008, 8192] {
        cases.push(Case {
            capacity,
            live: None,
            geometry: SERVED,
        });
        // A live count past the capacity reduces the capacity's rows.
        cases.push(Case {
            capacity,
            live: Some(capacity + 7),
            geometry: SERVED,
        });
    }
    cases.push(Case {
        capacity: 4096,
        live: Some(0),
        geometry: SERVED,
    });
    cases
}

/// One rank's matrix tools: page-locked host bytes the size of an output,
/// into which outputs are read back, and a device copy of the poison fill,
/// from which written rows are poisoned again.
struct Readback {
    host: runtime_cuda::PinnedHostBuffer,
    poison: DeviceBuffer,
}

impl Readback {
    fn new(session: &Session, bytes: usize) -> LabResult<Self> {
        let poison = session.allocate(bytes)?;
        poison::poison(session, &[&poison])?;
        Ok(Self {
            host: runtime_cuda::PinnedHostBuffer::new(&session.context, bytes).map_err(err)?,
            poison,
        })
    }

    /// The first `bytes` of `buffer`.
    fn read(&mut self, session: &Session, buffer: &DeviceBuffer, bytes: usize) -> LabResult<&[u8]> {
        if bytes > 0 {
            buffer
                .copy_to_pinned_at_async(0, &mut self.host, 0, bytes, &session.stream)
                .map_err(err)?
                .finish();
        }
        Ok(&self.host.as_slice()[..bytes])
    }
}

/// Every rank's output of `entry` in a matrix case: the first `written`
/// rows hold the host's bits, and, when `whole`, every other byte of the
/// output its poison; then the written rows are poisoned again.
#[allow(clippy::too_many_arguments)]
fn check_case(
    sessions: &[Session],
    rows: &[Rows],
    readbacks: &mut [Readback],
    entry: Entry,
    expected: &[u8],
    written: usize,
    whole: bool,
    label: &str,
) -> LabResult<()> {
    let written = written * WIDTH * 2;
    for (rank, ((session, rows), readback)) in sessions
        .iter()
        .zip(rows)
        .zip(readbacks.iter_mut())
        .enumerate()
    {
        let buffer = output_of(rows, entry);
        let read = if whole { buffer.byte_len() } else { written };
        let bytes = readback.read(session, buffer, read)?;
        if let Some(at) = first_difference(&bytes[..written], &expected[..written]) {
            return Err(format!(
                "{label}, {}, rank {rank}: rounded sum differs at element {}",
                entry.name(),
                at / 2
            ));
        }
        if let Some(at) = first_written(&bytes[written..]) {
            return Err(format!(
                "{label}, {}, rank {rank}: byte {} past the live rows was written",
                entry.name(),
                written + at
            ));
        }
    }
    // Every rank idle, then the written rows poisoned again from the
    // device copy of the fill: every output byte is poison before the
    // next case.
    poison::synchronize(sessions)?;
    for ((session, rows), readback) in sessions.iter().zip(rows).zip(readbacks.iter()) {
        if written > 0 {
            output_of(rows, entry)
                .copy_range_from_async(0, &readback.poison, 0, written, &session.stream)
                .map_err(err)?;
        }
    }
    synchronize(sessions)
}

fn run_bitwise(sessions: &[Session]) -> LabResult<()> {
    let ranks = sessions.len();
    let served = Served::load(sessions)?;
    let states = states(sessions, OPERAND_ROWS)?;
    let inputs = inputs_for(ranks, 11, OPERAND_ROWS);
    let expected = bytes_of(&host_sums(&inputs, OPERAND_ROWS * WIDTH));
    let rows = rows_buffers(sessions, &inputs)?;
    drop(inputs);
    let mut readbacks = sessions
        .iter()
        .map(|session| Readback::new(session, OPERAND_ROWS * WIDTH * 2))
        .collect::<LabResult<Vec<_>>>()?;
    for entry in [Entry::Turns, Entry::TwoShot] {
        poison_outputs(sessions, &rows, entry)?;
    }
    let cases = matrix();
    let started = std::time::Instant::now();
    for (index, case) in cases.iter().enumerate() {
        let written = case.written();
        set_live(sessions, &states, case.live.unwrap_or(0))?;
        let label = format!("case {index} {case:?}");
        for entry in [Entry::Turns, Entry::TwoShot] {
            launch_entry(
                sessions,
                served.modules(entry),
                entry.symbol(),
                &states,
                &rows,
                entry,
                Shape::new(case.capacity, case.live.is_some(), case.geometry),
                None,
            )?;
        }
        synchronize(sessions)?;
        for (entry, whole) in [(Entry::Turns, true), (Entry::TwoShot, false)] {
            check_case(
                sessions,
                &rows,
                &mut readbacks,
                entry,
                &expected,
                written,
                whole,
                &label,
            )?;
        }
        if index % 512 == 0 {
            eprintln!(
                "phased_turns_bf16 matrix: case {index} of {} ({:.0} s)",
                cases.len(),
                started.elapsed().as_secs_f32()
            );
        }
    }
    eprintln!(
        "phased_turns_bf16 matrix: {} cases bitwise against the host and the phased BF16 two-shot ({:.0} s)",
        cases.len(),
        started.elapsed().as_secs_f32()
    );
    for session in sessions {
        receipt(
            session,
            vec![
                ("case", json!("phased_turns_bf16_matrix_complete")),
                ("ranks", json!(ranks)),
                ("cases", json!(cases.len())),
                ("served_geometry", json!([SERVED.0, SERVED.1])),
                ("live_counts", json!([1, OPERAND_ROWS])),
                ("bitwise", json!(true)),
            ],
        )?;
    }
    Ok(())
}

#[test]
fn phased_turns_bf16_matches_host_and_current_entry() {
    gpu_test_multi(2 << 30, run_bitwise).expect("phased turns BF16 all-reduce bitwise matrix");
}

#[test]
fn phased_turns_bf16_matrix_covers_every_live_count() {
    let cases = matrix();
    for live in 1..=OPERAND_ROWS {
        assert!(cases.iter().any(|case| case.capacity == OPERAND_ROWS
            && case.live == Some(live)
            && case.geometry == SERVED));
    }
    for geometry in GEOMETRIES {
        for rows in EDGE_ROWS {
            assert!(cases.contains(&Case {
                capacity: rows,
                live: Some(rows),
                geometry
            }));
        }
    }
    assert!(cases.iter().any(|case| case.live.is_none()));
    assert!(cases.iter().any(|case| case.written() == 0));
    for case in &cases {
        assert!(case.capacity <= OPERAND_ROWS && case.written() <= case.capacity);
        turns::plan_with(case.geometry.0, case.geometry.1).unwrap();
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
    let served = Served::load(sessions)?;
    let states = states(sessions, 8192)?;
    for (seed, count, geometry) in [
        (40, 33, (16, 32)),
        (41, 2048, (16, 32)),
        (42, 2048, (16, 8)),
        (43, 64, (16, 0)),
        (44, 8192, (16, 32)),
        (45, 3008, (1, 1)),
    ] {
        let inputs = inputs_for(ranks, seed, count);
        let expected = bytes_of(&host_sums(&inputs, count * WIDTH));
        let rows = rows_buffers(sessions, &inputs)?;
        set_live(sessions, &states, count)?;
        let shape = Shape::new(count, true, geometry);
        let check = |phase: &str, sequence: &[Entry]| -> LabResult<()> {
            synchronize(sessions)?;
            for entry in [Entry::Turns, Entry::TwoShot] {
                if sequence.contains(&entry) {
                    check_outputs(
                        sessions,
                        &rows,
                        entry,
                        &expected,
                        count,
                        true,
                        &format!("{count} rows, {geometry:?}, {phase}"),
                    )?;
                }
            }
            Ok(())
        };
        let poison_all = || -> LabResult<()> {
            poison_outputs(sessions, &rows, Entry::Turns)?;
            poison_outputs(sessions, &rows, Entry::TwoShot)
        };
        let producers = sessions
            .iter()
            .zip(&rows)
            .map(|(session, rows)| Producer::new(session, rows))
            .collect::<LabResult<Vec<_>>>()?;
        // This entry alone, and between the phased BF16 entries on the
        // same state (the one-shot into the reference output too, where it
        // takes the call: its rows are the two-shot's).
        let mut sequences = vec![
            vec![Entry::Turns],
            vec![Entry::TwoShot, Entry::Turns, Entry::TwoShot],
        ];
        if (count * WIDTH) as u64 <= phased_bf16::ONE_SHOT_MAX_ELEMENTS {
            sequences.push(vec![Entry::OneShot, Entry::Turns]);
        }
        for sequence in sequences {
            let label = sequence
                .iter()
                .map(|entry| entry.name())
                .collect::<Vec<_>>()
                .join("+");
            let checked: Vec<Entry> = sequence
                .iter()
                .map(|&entry| match entry {
                    Entry::OneShot => Entry::TwoShot,
                    entry => entry,
                })
                .collect();
            let call = |entry: Entry, rank: Option<usize>| -> LabResult<()> {
                let entry_shape = match entry {
                    Entry::OneShot => Shape::new(count, true, (16, 0)),
                    _ => shape,
                };
                launch_entry(
                    sessions,
                    served.modules(entry),
                    entry.symbol(),
                    &states,
                    &rows,
                    entry,
                    entry_shape,
                    rank,
                )
            };
            let run = |only_rank: Option<usize>| -> LabResult<()> {
                for &entry in &sequence {
                    call(entry, only_rank)?;
                }
                Ok(())
            };
            poison_all()?;
            run(None)?;
            check(&format!("{label} eager"), &checked)?;
            poison_all()?;
            run(None)?;
            check(&format!("{label} relaunch"), &checked)?;
            let graphs = RankReplay::new(sessions, |rank, _| run(Some(rank)))?;
            for replay in 0..3 {
                poison_all()?;
                graphs.launch(sessions)?;
                check(&format!("{label} graph replay {replay}"), &checked)?;
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
                        call(entry, Some(rank))?;
                    }
                }
                Ok(())
            };
            poison_all()?;
            behind(None)?;
            check(&format!("{label} PDL producer"), &checked)?;
            let graphs = RankReplay::new(sessions, |rank, _| behind(Some(rank)))?;
            poison_all()?;
            graphs.launch(sessions)?;
            check(&format!("{label} PDL producer graph"), &checked)?;
            for session in sessions {
                receipt(
                    session,
                    vec![
                        ("case", json!("phased_turns_bf16_replays")),
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
fn phased_turns_bf16_replays() {
    gpu_test_multi(2 << 30, run_replays).expect("phased turns BF16 all-reduce replays");
}

/// Whether some rank's output (or, for `no_gather_wait`, its snapshot
/// taken right after the call on its stream) differs from the host.
fn differs(
    sessions: &[Session],
    rows: &[Rows],
    snapshots: &[DeviceBuffer],
    expected: &[u8],
    snapshot: bool,
) -> LabResult<bool> {
    synchronize(sessions)?;
    for (rank, session) in sessions.iter().enumerate() {
        let buffer = if snapshot {
            &snapshots[rank]
        } else {
            &rows[rank].output
        };
        let actual = read_prefix(session, buffer, expected.len())?;
        if first_difference(&actual, expected).is_some() {
            return Ok(true);
        }
    }
    Ok(false)
}

fn run_mutants(sessions: &[Session]) -> LabResult<()> {
    let ranks = sessions.len();
    let served = CollectiveModules::load(sessions, turns::LAB_CUBIN)?;
    let slow = CollectiveModules::load(sessions, &turns::mutant_cubin(turns::SLOW_CONTROL))?;
    let late =
        CollectiveModules::load(sessions, &turns::mutant_cubin(turns::LATE_SCATTER_CONTROL))?;
    let states = states(sessions, 2048)?;
    let mut missed = Vec::new();
    // (rows, geometry): every case is required.
    let cases = [(2048, (16, 32)), (64, (16, 0))];
    for (seed, (count, geometry)) in cases.into_iter().enumerate() {
        let inputs = inputs_for(ranks, 60 + seed as u64, count);
        let expected = bytes_of(&host_sums(&inputs, count * WIDTH));
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
        for (tag, _) in turns::MUTANTS {
            if tag == "reversed" && ranks < 3 {
                // (0 + x0) + x1 and (0 + x1) + x0 round alike.
                eprintln!("mutant reversed at {count} rows: inapplicable at {ranks} ranks");
                continue;
            }
            let modules = CollectiveModules::load(sessions, &turns::mutant_cubin(tag))?;
            let symbol = turns::mutant_symbol(turns::SYMBOL, tag);
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
                        Entry::Turns,
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
            let label = format!("{tag} at {count} rows, {geometry:?}");
            let (control, control_symbol) = if turns::SLOW_MUTANTS.contains(&tag) {
                (
                    &slow,
                    turns::mutant_symbol(turns::SYMBOL, turns::SLOW_CONTROL),
                )
            } else if turns::LATE_SCATTER_MUTANTS.contains(&tag) {
                (
                    &late,
                    turns::mutant_symbol(turns::SYMBOL, turns::LATE_SCATTER_CONTROL),
                )
            } else {
                (&served, turns::SYMBOL.to_owned())
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
                        ("case", json!("phased_turns_bf16_mutants_are_caught")),
                        ("ranks", json!(ranks)),
                        ("rows", json!(count)),
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
fn phased_turns_bf16_mutants_are_caught() {
    gpu_test_multi(1 << 30, |sessions| {
        if sessions.len() < 2 {
            return Err("the mutants need two or more ranks".into());
        }
        run_mutants(sessions)
    })
    .expect("phased turns BF16 all-reduce mutants");
}

/// Clock cycles the delayed peer spins before its call (about 100 us).
const DELAY_CYCLES: u64 = 200_000;

/// The lab's spin kernel (`latent_shapes::pdl_copy_symbol`, no words):
/// `cycles` clock cycles on one CTA of the rank's stream.
struct Spin {
    function: runtime_cuda::Function,
    scratch: DeviceBuffer,
}

impl Spin {
    fn new(session: &Session) -> LabResult<Self> {
        Ok(Self {
            function: session
                .module(&kernel_lab::latent_shapes::cubin_name())?
                .function(&kernel_lab::latent_shapes::pdl_copy_symbol())
                .map_err(err)?,
            scratch: session.allocate(16)?,
        })
    }

    fn launch(&self, session: &Session, cycles: u64) -> LabResult<()> {
        self.function
            .launch(
                &session.stream,
                launch::simple_config((1, 1, 1), (32, 1, 1), 0, LaunchConfig::NO_CLUSTER, false)?,
                &[
                    KernelArgument::Buffer(&self.scratch),
                    KernelArgument::Buffer(&self.scratch),
                    KernelArgument::U64(0),
                    KernelArgument::U64(cycles),
                ],
            )
            .map_err(err)
    }
}

/// One rank's buffers of a generations case: each generation's partials,
/// the outputs every call of either kind writes, and each generation's
/// snapshot of its result, copied on the rank's stream right after the
/// call.
struct Generations {
    inputs: Vec<DeviceBuffer>,
    output: DeviceBuffer,
    ref_output: DeviceBuffer,
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
    buffers.insert(REF_OUTPUT, &ranks[rank].ref_output);
    buffers.insert(EPOCH, &states[rank].epoch);
    buffers.insert(LIVE, &states[rank].live);
    for peer in 0..states.len() {
        let q = peer as u64;
        buffers.insert(INBOXES + q, &states[peer].inbox);
        buffers.insert(FLAGS + q, &states[peer].flags);
        buffers.insert(OUTPUTS + q, &ranks[peer].output);
        buffers.insert(REF_OUTPUTS + q, &ranks[peer].ref_output);
    }
    Bindings { buffers }
}

fn run_generations(sessions: &[Session]) -> LabResult<()> {
    let ranks = sessions.len();
    let served = Served::load(sessions)?;
    let slow = CollectiveModules::load(sessions, &turns::mutant_cubin(turns::SLOW_CONTROL))?;
    let spins = sessions
        .iter()
        .map(Spin::new)
        .collect::<LabResult<Vec<_>>>()?;
    let states = states(sessions, 2048)?;
    use Entry::{OneShot, Turns, TwoShot};
    let cases: [(usize, (u32, u32), &[Entry]); 3] = [
        (
            2048,
            (16, 32),
            &[Turns, Turns, TwoShot, Turns, Turns, TwoShot, Turns, Turns],
        ),
        (
            64,
            (16, 0),
            &[
                Turns, OneShot, Turns, TwoShot, OneShot, Turns, Turns, OneShot,
            ],
        ),
        (
            256,
            (8, 8),
            &[Turns, Turns, Turns, Turns, TwoShot, TwoShot, Turns, Turns],
        ),
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
                    ref_output: session.allocate(elements * 2)?,
                    snapshots: sequence
                        .iter()
                        .map(|_| session.allocate(elements * 2))
                        .collect::<LabResult<Vec<_>>>()?,
                })
            })
            .collect::<LabResult<Vec<_>>>()?;
        set_live(sessions, &states, count)?;
        // The sequence, this entry's calls through `turns_modules` (the
        // served cubin, or the slow-rank control tagged `tag`).
        let sequence_on =
            |turns_modules: &CollectiveModules, tag: Option<&str>, only_rank: Option<usize>| {
                for (generation, &entry) in sequence.iter().enumerate() {
                    let delayed = generation % ranks;
                    if only_rank.is_none_or(|only| only == delayed) {
                        spins[delayed].launch(&sessions[delayed], DELAY_CYCLES)?;
                    }
                    let (modules, symbol) = match (entry, tag) {
                        (Entry::Turns, Some(tag)) => {
                            (turns_modules, turns::mutant_symbol(turns::SYMBOL, tag))
                        }
                        (Entry::Turns, None) => (turns_modules, turns::SYMBOL.to_owned()),
                        (entry, _) => (served.modules(entry), entry.symbol().to_owned()),
                    };
                    let shape = match entry {
                        Entry::OneShot => Shape::new(count, true, (16, 0)),
                        _ => Shape::new(count, true, geometry),
                    };
                    launch_bound(
                        sessions,
                        modules,
                        &symbol,
                        entry,
                        shape,
                        only_rank,
                        |rank| generation_bindings(&states, &buffers, generation, rank),
                    )?;
                    for (rank, session) in sessions.iter().enumerate() {
                        if only_rank.is_some_and(|only| only != rank) {
                            continue;
                        }
                        let output = match entry.reference() {
                            true => &buffers[rank].ref_output,
                            false => &buffers[rank].output,
                        };
                        buffers[rank].snapshots[generation]
                            .copy_range_from_async(0, output, 0, elements * 2, &session.stream)
                            .map_err(err)?;
                    }
                }
                LabResult::Ok(())
            };
        let fresh = || -> LabResult<()> {
            reset(sessions, &states)?;
            for (session, (state, rank)) in sessions.iter().zip(states.iter().zip(&buffers)) {
                poison::poison(session, &[&state.inbox, &rank.output, &rank.ref_output])?;
                for snapshot in &rank.snapshots {
                    poison::poison(session, &[snapshot])?;
                }
            }
            Ok(())
        };
        let wrong_words = || -> LabResult<usize> {
            synchronize(sessions)?;
            let mut wrong = 0;
            for (session, rank) in sessions.iter().zip(&buffers) {
                for (snapshot, sums) in rank.snapshots.iter().zip(&expected) {
                    let actual = poison::read(session, snapshot)?;
                    wrong += actual
                        .chunks_exact(2)
                        .zip(sums)
                        .filter(|(a, e)| u16::from_ne_bytes([a[0], a[1]]) != **e)
                        .count();
                }
            }
            Ok(wrong)
        };
        let label = format!("{count} rows, {geometry:?}");
        let mut served_wrong = Vec::new();
        for (modules, tag) in [(&served.turns, None), (&slow, Some(turns::SLOW_CONTROL))] {
            fresh()?;
            sequence_on(modules, tag, None)?;
            served_wrong.push(wrong_words()?);
            let graphs =
                RankReplay::new(sessions, |rank, _| sequence_on(modules, tag, Some(rank)))?;
            for _ in 0..3 {
                fresh()?;
                graphs.launch(sessions)?;
                served_wrong.push(wrong_words()?);
            }
        }
        if served_wrong.iter().any(|&wrong| wrong > 0) {
            failures.push(format!("{label}: served wrong words {served_wrong:?}"));
        }
        eprintln!("generations {label}: served and slow-rank wrong words {served_wrong:?}");
        for session in sessions {
            receipt(
                session,
                vec![
                    ("case", json!("phased_turns_bf16_generations")),
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
fn phased_turns_bf16_generations() {
    gpu_test_multi(1 << 30, run_generations).expect("phased turns BF16 all-reduce generations");
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

fn bench_enabled(what: &str) -> bool {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() == Ok("1") {
        return true;
    }
    eprintln!("skipping {what}: set CHOPINFER_KERNEL_BENCH=1 on idle GPUs");
    false
}

/// Microseconds per call (median and min), critical path over ranks, of
/// `calls` repeated calls in one graph per rank.
fn time_calls(
    sessions: &[Session],
    calls: usize,
    one: impl Fn(usize, usize) -> LabResult<()>,
) -> LabResult<(f32, f32)> {
    let graphs = sessions
        .iter()
        .enumerate()
        .map(|(rank, session)| {
            session.capture(|_| {
                for call in 0..calls {
                    one(rank, call)?;
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

/// A skew pattern: the spin cycles of `rank` before call `call`.
#[derive(Clone, Copy, Debug)]
enum Skew {
    /// Every rank starts together.
    None,
    /// Rank `call % ranks` spins the cycles, the others none.
    Late(u64),
    /// Rank r spins cycles * ((r + call) % ranks) / (ranks - 1).
    Staggered(u64),
}

impl Skew {
    fn cycles(self, rank: usize, call: usize, ranks: usize) -> u64 {
        match self {
            Skew::None => 0,
            Skew::Late(cycles) => match rank == call % ranks {
                true => cycles,
                false => 0,
            },
            Skew::Staggered(cycles) => {
                cycles * ((rank + call) % ranks) as u64 / (ranks.max(2) - 1) as u64
            }
        }
    }

    fn name(self) -> String {
        match self {
            Skew::None => "none".into(),
            Skew::Late(cycles) => format!("late{cycles}"),
            Skew::Staggered(cycles) => format!("staggered{cycles}"),
        }
    }

    /// The largest spin of any rank before a call.
    fn most(self) -> u64 {
        match self {
            Skew::None => 0,
            Skew::Late(cycles) | Skew::Staggered(cycles) => cycles,
        }
    }
}

/// Microseconds of one spin of `cycles` on one rank (median of a graph of
/// spins), so a skewed call's own time is its time less the skew.
fn spin_us(sessions: &[Session], spins: &[Spin], cycles: u64) -> LabResult<f32> {
    if cycles == 0 {
        return Ok(0.0);
    }
    let calls = 16;
    let graph = sessions[0].capture(|_| {
        for _ in 0..calls {
            spins[0].launch(&sessions[0], cycles)?;
        }
        Ok(())
    })?;
    let timing = time_multi_graph(&sessions[..1], std::slice::from_ref(&graph), calls, 2, 5)?;
    Ok(timing.critical_path.median_ms * 1000.0 / calls as f32)
}

fn run_timing(sessions: &[Session]) -> LabResult<()> {
    let ranks = sessions.len();
    let served = Served::load(sessions)?;
    let spins = sessions
        .iter()
        .map(Spin::new)
        .collect::<LabResult<Vec<_>>>()?;
    let all_rows = env_list(
        "CHOPINFER_KERNEL_BENCH_ROWS",
        &[256, 1024, 2048, 3008, 4096, 8192],
    );
    let geometries: Vec<(u32, u32)> = {
        let senders = env_list("CHOPINFER_KERNEL_BENCH_SENDERS", &[16, 24, 32]);
        let workers = env_list("CHOPINFER_KERNEL_BENCH_WORKERS", &[32]);
        workers
            .iter()
            .flat_map(|&w| senders.iter().map(move |&s| (s, w)))
            .collect()
    };
    let skew_cycles = env_list("CHOPINFER_KERNEL_BENCH_SKEW_CYCLES", &[1_000_000_u64]);
    let mut skews = vec![Skew::None];
    for &cycles in &skew_cycles {
        skews.push(Skew::Late(cycles));
        skews.push(Skew::Staggered(cycles));
    }
    let variants: Vec<String> = std::env::var("CHOPINFER_KERNEL_BENCH_VARIANTS")
        .map(|value| value.split(',').map(|v| v.trim().to_owned()).collect())
        .unwrap_or_else(|_| {
            let mut all = vec!["bf16".to_owned(), "turns_bf16".to_owned()];
            all.extend(
                turns::TIMING_VARIANTS
                    .iter()
                    .map(|(tag, _)| tag.to_string()),
            );
            all
        });
    let mut variant_modules = BTreeMap::new();
    for (tag, _) in turns::TIMING_VARIANTS {
        if variants.iter().any(|v| v == tag) {
            variant_modules.insert(
                tag,
                CollectiveModules::load(sessions, &turns::mutant_cubin(tag))?,
            );
        }
    }
    let mut spin_time = BTreeMap::new();
    for skew in &skews {
        spin_time.insert(skew.most(), spin_us(sessions, &spins, skew.most())?);
    }
    eprintln!("phased_turns_bf16 timing: spin us by cycles {spin_time:?}");
    let capacity = *all_rows.iter().max().ok_or("no benchmark rows")?;
    let states = states(sessions, capacity)?;
    for &count in &all_rows {
        let calls = (16384 / count).clamp(8, 32);
        let inputs = vec![vec![0.0_f32; count * WIDTH]; ranks];
        let rows = rows_buffers(sessions, &inputs)?;
        set_live(sessions, &states, count)?;
        let sent = (count * WIDTH * 2) as f32 * (ranks - 1) as f32 / ranks as f32 * 2.0;
        for &(senders, workers) in &geometries {
            let shape = Shape::new(count, true, (senders, workers));
            for variant in &variants {
                let (entry, modules, symbol) = match variant.as_str() {
                    "bf16" => (
                        Entry::TwoShot,
                        &served.phased,
                        phased_bf16::SYMBOL.to_owned(),
                    ),
                    "turns_bf16" => (Entry::Turns, &served.turns, turns::SYMBOL.to_owned()),
                    tag => (
                        Entry::Turns,
                        variant_modules
                            .get(tag)
                            .ok_or_else(|| format!("unknown timing variant {tag}"))?,
                        turns::mutant_symbol(turns::SYMBOL, tag),
                    ),
                };
                for &skew in &skews {
                    let (median, min) = time_calls(sessions, calls, |rank, call| {
                        let cycles = skew.cycles(rank, call, ranks);
                        if cycles > 0 {
                            spins[rank].launch(&sessions[rank], cycles)?;
                        }
                        launch_entry(
                            sessions,
                            modules,
                            &symbol,
                            &states,
                            &rows,
                            entry,
                            shape,
                            Some(rank),
                        )
                    })?;
                    let skew_us = spin_time[&skew.most()];
                    let own = median - skew_us;
                    let gbps = sent / own / 1000.0;
                    eprintln!(
                        "phased_turns_bf16 timing: rows {count:5} {variant:20} senders {senders:2} workers {workers:2} skew {:>16}: median {median:9.2} us, min {min:9.2} us, less skew {own:9.2} us, {gbps:6.2} GB/s sent per rank",
                        skew.name()
                    );
                    for session in sessions {
                        receipt(
                            session,
                            vec![
                                ("case", json!("phased_turns_bf16_timing")),
                                ("ranks", json!(ranks)),
                                ("rows", json!(count)),
                                ("variant", json!(variant)),
                                ("senders", json!(senders)),
                                ("workers", json!(workers)),
                                ("skew", json!(skew.name())),
                                ("skew_us", json!(skew_us)),
                                ("calls_per_graph", json!(calls)),
                                ("median_us_per_call", json!(median)),
                                ("min_us_per_call", json!(min)),
                                ("median_less_skew_us", json!(own)),
                                ("sent_gb_per_s_per_rank", json!(gbps)),
                            ],
                        )?;
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn phased_turns_bf16_timing() {
    if !bench_enabled("phased turns BF16 all-reduce timing") {
        return;
    }
    gpu_test_multi(4 << 30, run_timing).expect("phased turns BF16 all-reduce timing");
}

/// Every rank's destination (or none) in a link pattern.
fn link_patterns(ranks: usize) -> Vec<(String, Vec<Option<usize>>)> {
    let mut patterns = Vec::new();
    for d in 1..ranks {
        patterns.push((
            format!("shift{d}"),
            (0..ranks).map(|r| Some((r + d) % ranks)).collect(),
        ));
    }
    let alone = |pairs: &[(usize, usize)]| {
        let mut to = vec![None; ranks];
        for &(from, dest) in pairs {
            to[from] = Some(dest);
        }
        to
    };
    patterns.push(("pair_0_1".into(), alone(&[(0, 1)])));
    if ranks >= 3 {
        patterns.push(("pair_0_2".into(), alone(&[(0, 2)])));
        patterns.push(("into_2_from_0_1".into(), alone(&[(0, 2), (1, 2)])));
    }
    if ranks >= 4 {
        patterns.push(("into_2_from_0_1_3".into(), alone(&[(0, 2), (1, 2), (3, 2)])));
        // One writer per receiver, but rank 2 also sends while receiving
        // two: the skewed two-shot's worst case.
        patterns.push((
            "into_1_from_0_3_and_1_to_2".into(),
            alone(&[(0, 1), (3, 1), (1, 2)]),
        ));
    }
    patterns
}

fn run_link_patterns(sessions: &[Session]) -> LabResult<()> {
    let ranks = sessions.len();
    let bytes: usize = env_list("CHOPINFER_KERNEL_BENCH_LINK_BYTES", &[16_usize << 20])[0];
    let words = bytes / 4;
    let inputs = sessions
        .iter()
        .map(|session| session.upload_bf16(&vec![f32_to_bf16(1.0); bytes / 2]))
        .collect::<LabResult<Vec<_>>>()?;
    let targets = sessions
        .iter()
        .map(|session| session.allocate(bytes * ranks))
        .collect::<LabResult<Vec<_>>>()?;
    let functions = sessions
        .iter()
        .map(|session| {
            session
                .module("probe_sm120")?
                .function("chop_probe_p2p_push_fanout_v1")
                .map_err(err)
        })
        .collect::<LabResult<Vec<_>>>()?;
    let calls = 16;
    for ctas in env_list("CHOPINFER_KERNEL_BENCH_LINK_CTAS", &[16_u32, 32]) {
        for (name, to) in link_patterns(ranks) {
            let graphs = sessions
                .iter()
                .enumerate()
                .map(|(rank, session)| {
                    // An idle rank launches with no words into its own
                    // buffer, so every rank's graph holds the same calls.
                    let (dest, count) = match to[rank] {
                        Some(dest) => (dest, words as u64),
                        None => (rank, 0),
                    };
                    let peers = vec![
                        targets[dest]
                            .device_address_at(bytes * rank, bytes)
                            .map_err(err)?,
                    ];
                    let config = launch::simple_config(
                        (ctas, 1, 1),
                        (256, 1, 1),
                        0,
                        LaunchConfig::NO_CLUSTER,
                        true,
                    )?;
                    session.capture(|_| {
                        for _ in 0..calls {
                            functions[rank]
                                .launch(
                                    &session.stream,
                                    config,
                                    &[
                                        KernelArgument::Buffer(&inputs[rank]),
                                        KernelArgument::DevicePointerArray(&peers),
                                        KernelArgument::U64(count),
                                    ],
                                )
                                .map_err(err)?;
                        }
                        Ok(())
                    })
                })
                .collect::<LabResult<Vec<_>>>()?;
            let timing = time_multi_graph(sessions, &graphs, calls, 2, 5)?;
            let us = timing.critical_path.median_ms * 1000.0 / calls as f32;
            let per_rank: Vec<f32> = timing
                .per_rank
                .iter()
                .map(|t| t.median_ms * 1000.0 / calls as f32)
                .collect();
            let gbps = bytes as f32 / us / 1000.0;
            eprintln!(
                "phased_turns_bf16 links: {name:28} ctas {ctas:2}: {us:9.2} us per {} MiB, {gbps:6.2} GB/s per writer (critical path); per rank us {per_rank:?}",
                bytes >> 20
            );
            for session in sessions {
                receipt(
                    session,
                    vec![
                        ("case", json!("phased_turns_bf16_link_patterns")),
                        ("ranks", json!(ranks)),
                        ("pattern", json!(name)),
                        ("destinations", json!(to)),
                        ("ctas", json!(ctas)),
                        ("bytes", json!(bytes)),
                        ("median_us_per_call", json!(us)),
                        ("per_rank_median_us", json!(per_rank)),
                        ("gb_per_s_per_writer", json!(gbps)),
                    ],
                )?;
            }
        }
    }
    Ok(())
}

#[test]
fn phased_turns_bf16_link_patterns() {
    if !bench_enabled("PCIe link patterns") {
        return;
    }
    gpu_test_multi(2 << 30, run_link_patterns).expect("PCIe link patterns");
}

#[test]
fn phased_turns_bf16_cpu_contract() {
    let entries = kernel_lab::cubin_entries(turns::LAB_CUBIN).unwrap();
    assert_eq!(entries, [turns::SYMBOL]);
    for (tag, _) in turns::MUTANTS {
        assert!(kernel_lab::comm_cubin(&turns::mutant_cubin(tag)).unwrap());
    }
    for (tag, _) in turns::TIMING_VARIANTS {
        assert!(kernel_lab::comm_cubin(&turns::mutant_cubin(tag)).unwrap());
    }
    for tag in [turns::SLOW_CONTROL, turns::LATE_SCATTER_CONTROL] {
        assert!(kernel_lab::comm_cubin(&turns::mutant_cubin(tag)).unwrap());
    }
    // Skew patterns delay one rank, or every rank by a staggered share.
    assert_eq!(Skew::Late(8).cycles(1, 5, 4), 8);
    assert_eq!(Skew::Late(8).cycles(0, 5, 4), 0);
    assert_eq!(Skew::Staggered(9).cycles(3, 0, 4), 9);
    assert_eq!(Skew::Staggered(9).cycles(0, 0, 4), 0);
    assert_eq!(link_patterns(4).len(), 3 + 5);
    // The host fold rounds every rank's partial before the sum.
    let inputs = inputs_for(4, 9, 2);
    let expected = host_sums(&inputs, 2 * WIDTH);
    let reversed: Vec<_> = inputs.iter().rev().cloned().collect();
    assert_ne!(host_sums(&reversed, 2 * WIDTH), expected);
    let mut poisoned = vec![POISON; 40];
    assert_eq!(first_written(&poisoned), None);
    poisoned[37] = 0;
    assert_eq!(first_written(&poisoned), Some(37));
    assert_eq!(first_written(&poisoned[1..]), Some(36));
    assert_eq!(first_difference(&[1, 2, 3], &[1, 2, 4]), Some(2));
    assert_eq!(first_difference(&[1, 2], &[1, 2]), None);
}
