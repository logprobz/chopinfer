//! The low-latency all-reduce of BF16 partials from F32 inputs
//! (`kernels_sm120::comm::ll_bf16`, lock family `allreduce_ll_bf16`) on four
//! GPUs, bit for bit against the host's fold: every rank's F32 partial
//! rounded once to BF16, the BF16 values summed in rank order from F32
//! zero, the sum rounded once to BF16. Run with
//! `CHOPINFER_KERNEL_GPUS=0,1,2,3 cargo test -p kernel-lab --release --test
//! allreduce_ll_bf16 -- --test-threads=1`; without GPUs only the CPU
//! contract test runs.
//!
//! - At every row count of the state's capacity (1 to 32 rows of 4096), on
//!   one persistent state: every rank's sums equal the host's, which equal
//!   the served F32 low-latency entry's sums over the host-rounded partials
//!   (`chop_comm_allreduce_ll_f32_v1`, the composition the ACC4 diagnostic
//!   served: round, widen, F32 all-reduce, round) rounded on the host. The
//!   partials span 41 binary orders of magnitude with signed zeros,
//!   subnormals and BF16 rounding ties; the test checks that the reversed
//!   rank order changes some rounded sums. They come from a device producer
//!   (the lab's late-trigger PDL probe) that holds them at NaN past its
//!   trigger; the entry follows it with PDL, eagerly and through each rank's
//!   captured graph, twice with new partials.
//! - Poisoned replays (eager, relaunch, graph, PDL) at every row count: the
//!   partials and sums are poisoned before every launch, the sums must be
//!   written exactly over the call's rows and nothing past them, and every
//!   replay must reproduce the eager bits, which equal the host's.
//! - Around the 12-bit flag wrap and the 32-bit epoch wrap at 1, 4 and 32
//!   rows, as `allreduce_ll_f32.rs` checks the F32 entry, then a graph of 64
//!   calls replayed 70 times and the sums again.
//! - Mutants, counted only on an observed difference from the host (launch,
//!   synchronization and readback errors fail the test): on one GPU behind
//!   the producer at 1, 4 and 32 rows, the dependency wait after the input
//!   loads and none; on four GPUs over six calls at 1 and 4 rows each, the
//!   stale-flag mutant (on a state of its own) and the own-F32 mutant (this
//!   rank's own values summed unrounded).
//! - Residency of the largest grid on every rank, and the host's residency
//!   report at 2 and 32 rows.
//! - Timing (receipt only): per-call medians of graphs of 64 calls, this
//!   entry against the F32 low-latency entry, at 1 to 32 rows.

use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::DType;
use kernel_lab::launch::{self, Bindings, CollectiveModules, RankCall};
use kernel_lab::poison::{self, Phase, Replay, WriteMap};
use kernel_lab::topology::RankReplay;
use kernel_lab::{LabResult, Session};
use kernels_sm120::comm::{ll_bf16, ll_f32};
use runtime_cuda::{DeviceBuffer, Function, KernelArgument, Module};
use serde_json::{Value, json};
use std::collections::BTreeMap;

const RANKS: usize = 4;
const H: usize = 4096;
/// The state's capacity.
const MAX_ROWS: usize = 32;
/// CTAs per row.
const ROW_CTAS: usize = H / ll_bf16::SLICE_ELEMENTS as usize;

const LL_CUBIN: &str = ll_bf16::LAB_CUBIN;
const F32_CUBIN: &str = "comm_ll_f32_sm120";
const PROBE_CUBIN: &str = "probe_pdl_late_trigger_sm120";
/// The producer's lead and tail (see `allreduce_ll_f32.rs`).
const PRODUCER_LEAD_NS: u64 = 2_000_000;
const PRODUCER_TAIL_NS: u64 = 2_000_000;
/// Row counts of the mutations and the flag-wrap check.
const MUTATION_ROWS: [usize; 3] = [1, 4, 32];
const TIMED_ROWS: [usize; 7] = [1, 2, 4, 8, 16, 24, 32];
const CALLS_PER_GRAPH: usize = 64;
const WRAP_REPLAYS: usize = 70;
const WRAP_CALLS: u32 = 12;
const WRAP_STARTS: [u32; 2] = [4096 - 6, u32::MAX - 5];
/// Row counts and calls of the four-rank mutants.
const STATE_MUTANT_ROWS: [usize; 2] = [1, 4];
const STATE_MUTANT_CALLS: u64 = 6;

// Buffer handles of one rank.
/// The F32 partials the entry reads, written by the producer.
const PARTIAL: u64 = 1;
/// The call's partials, which the producer copies into `PARTIAL`.
const FRESH: u64 = 2;
/// The entry's BF16 sums.
const SUMS: u64 = 3;
/// The F32 entry's partials (the host-rounded partials) and F32 sums.
const REF_PARTIAL: u64 = 4;
const REF_SUMS: u64 = 5;
const LL_EPOCHS: u64 = 6;
const REF_EPOCHS: u64 = 7;
/// The stale-flag mutant's own epochs and inboxes (`STALE_INBOX + rank`).
const STALE_EPOCHS: u64 = 8;
/// Every rank's inboxes: `HANDLE + rank`.
const LL_INBOX: u64 = 100;
const REF_INBOX: u64 = 200;
const STALE_INBOX: u64 = 400;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn words(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn halves(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_ne_bytes([c[0], c[1]]))
        .collect()
}

fn exact<T: PartialEq + std::fmt::LowerHex>(
    label: &str,
    actual: &[T],
    expected: &[T],
) -> LabResult<()> {
    if actual.len() != expected.len() {
        return Err(format!(
            "{label}: {} values, expected {}",
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

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

/// Every rank's F32 partials of `rows` rows: mostly values over 41 binary
/// orders of magnitude, so the sum depends on the order, with signed zeros,
/// subnormals and exact BF16 rounding ties among them. Every 53rd element
/// is 2^24 on rank 0, -2^24 on rank 3 and 1 elsewhere, times a power of
/// two: exact in BF16, and the F32 sum's order shows in the rounded sum.
fn partials(rows: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Rng(seed);
    (0..RANKS)
        .map(|rank| {
            (0..rows * H)
                .map(|i| {
                    let state = rng.next();
                    let sign = if state & 1 == 0 { 1.0 } else { -1.0 };
                    let unit = (state >> 40) as f32 / (1u64 << 24) as f32;
                    let value = sign * unit * 2f32.powi(((state >> 8) % 41) as i32 - 20);
                    if i % 53 == 7 {
                        let scale = 2f32.powi((i / 53 % 8) as i32 - 4);
                        return scale
                            * match rank {
                                0 => 16_777_216.0,
                                3 => -16_777_216.0,
                                _ => 1.0,
                            };
                    }
                    match i % 997 {
                        0 => 0.0,
                        1 => -0.0,
                        2 => sign * f32::from_bits(1 + ((state >> 8) % 0x7f_ffff) as u32),
                        // Halfway between two BF16 values.
                        3..=12 => f32::from_bits(value.to_bits() & 0xffff_0000 | 0x8000),
                        _ => value,
                    }
                })
                .collect()
        })
        .collect()
}

/// A rank's partials rounded once to BF16 and widened back, exactly.
fn rounded(partial: &[f32]) -> Vec<f32> {
    partial
        .iter()
        .map(|&v| bf16_to_f32(f32_to_bf16(v)))
        .collect()
}

/// The rank-order F32 sums from zero of the rounded partials.
fn host_f32(partials: &[Vec<f32>]) -> Vec<f32> {
    let rounded: Vec<_> = partials.iter().map(|p| rounded(p)).collect();
    (0..partials[0].len())
        .map(|i| rounded.iter().fold(0.0_f32, |sum, rank| sum + rank[i]))
        .collect()
}

/// [`host_f32`] rounded once to BF16: the entry's sums.
fn host_sums(partials: &[Vec<f32>]) -> Vec<u16> {
    host_f32(partials).into_iter().map(f32_to_bf16).collect()
}

/// How many rounded sums the reversed rank order changes: the inputs must
/// make the order matter.
fn order_sensitive(partials: &[Vec<f32>]) -> usize {
    let reversed: Vec<_> = partials.iter().rev().cloned().collect();
    host_sums(partials)
        .iter()
        .zip(host_sums(&reversed))
        .filter(|(a, b)| **a != *b)
        .count()
}

/// The late-trigger PDL producer (`probe_pdl_late_trigger_sm120`) over
/// `rows` rows of a rank's partials: after its lead it poisons them with
/// NaN, triggers its dependents, waits its tail, then copies `FRESH` in.
struct PartialProducer<'a> {
    _module: Module,
    function: Function,
    fresh: &'a DeviceBuffer,
    partial: &'a DeviceBuffer,
    words: u64,
}

const PRODUCER_BLOCKS: u32 = 32;

impl<'a> PartialProducer<'a> {
    fn new(
        session: &Session,
        fresh: &'a DeviceBuffer,
        partial: &'a DeviceBuffer,
        rows: usize,
    ) -> LabResult<Self> {
        let module = session.module(PROBE_CUBIN)?;
        let function = module
            .function("chop_probe_pdl_late_trigger_producer_v1")
            .map_err(err)?;
        Ok(Self {
            _module: module,
            function,
            fresh,
            partial,
            words: (rows * H) as u64,
        })
    }

    fn launch(&self, session: &Session) -> LabResult<()> {
        self.function
            .launch(
                &session.stream,
                launch::simple_config(
                    (PRODUCER_BLOCKS, 1, 1),
                    (128, 1, 1),
                    0,
                    runtime_cuda::LaunchConfig::NO_CLUSTER,
                    true,
                )?,
                &[
                    KernelArgument::Buffer(self.fresh),
                    KernelArgument::Buffer(self.partial),
                    KernelArgument::U64(self.words),
                    KernelArgument::U64(PRODUCER_LEAD_NS),
                    KernelArgument::U64(PRODUCER_TAIL_NS),
                ],
            )
            .map_err(err)
    }
}

/// One rank's buffers: this entry's state, the stale mutant's and the F32
/// entry's, each zeroed once and kept across every call of the test.
struct Rank {
    buffers: BTreeMap<u64, DeviceBuffer>,
}

impl Rank {
    fn new(session: &Session) -> LabResult<Self> {
        let capacity = (MAX_ROWS * H) as u64;
        let state = ll_bf16::state_bytes(RANKS as u32, capacity).map_err(err)?;
        let reference = ll_f32::state_bytes(RANKS as u32, capacity).map_err(err)?;
        let mut buffers = BTreeMap::new();
        let f32s = MAX_ROWS * H * 4;
        for handle in [PARTIAL, FRESH, SUMS, REF_PARTIAL, REF_SUMS] {
            buffers.insert(handle, session.allocate(f32s)?);
        }
        buffers.insert(LL_INBOX, session.allocate(state.inbox as usize)?);
        buffers.insert(LL_EPOCHS, session.allocate(state.epochs as usize)?);
        buffers.insert(STALE_INBOX, session.allocate(state.inbox as usize)?);
        buffers.insert(STALE_EPOCHS, session.allocate(state.epochs as usize)?);
        buffers.insert(REF_INBOX, session.allocate(reference.inbox as usize)?);
        buffers.insert(REF_EPOCHS, session.allocate(reference.epochs as usize)?);
        for handle in [
            LL_INBOX,
            LL_EPOCHS,
            STALE_INBOX,
            STALE_EPOCHS,
            REF_INBOX,
            REF_EPOCHS,
        ] {
            poison::fill_all(session, &buffers[&handle], &[0])?;
        }
        Ok(Self { buffers })
    }

    fn buffer(&self, handle: u64) -> &DeviceBuffer {
        &self.buffers[&handle]
    }
}

/// Every rank's state, loaded for collective calls.
struct Ranks<'a> {
    sessions: &'a [Session],
    ranks: Vec<Rank>,
    ll: CollectiveModules,
    stale: CollectiveModules,
    own_f32: CollectiveModules,
    reference: CollectiveModules,
}

impl Ranks<'_> {
    fn bindings(&self, rank: usize) -> Bindings<'_> {
        let mut buffers: BTreeMap<u64, &DeviceBuffer> = BTreeMap::new();
        for (&handle, buffer) in &self.ranks[rank].buffers {
            if handle < LL_INBOX {
                buffers.insert(handle, buffer);
            }
        }
        for (peer, state) in self.ranks.iter().enumerate() {
            for handle in [LL_INBOX, REF_INBOX, STALE_INBOX] {
                buffers.insert(handle + peer as u64, state.buffer(handle));
            }
        }
        Bindings { buffers }
    }

    /// The entry `symbol` of `modules` over `rows` rows on every rank (or
    /// `only`), over the state whose inboxes start at handle `state.0` and
    /// whose epochs are `state.1`.
    fn ll_on(
        &self,
        modules: &CollectiveModules,
        symbol: &str,
        state: (u64, u64),
        rows: usize,
        only: Option<usize>,
        pdl: bool,
    ) -> LabResult<()> {
        let elements = (rows * H) as u64;
        let mut plan = ll_bf16::plan(RANKS as u32, elements).map_err(err)?;
        plan.pdl = pdl;
        launch::launch_collective(self.sessions, modules, symbol, plan, only, |rank| {
            let call = ll_bf16::Call {
                input: PARTIAL,
                output: SUMS,
                peer_inboxes: (0..RANKS as u64).map(|peer| state.0 + peer).collect(),
                epochs: state.1,
                elements,
                capacity_elements: (MAX_ROWS * H) as u64,
                rank: rank as u32,
                ranks: RANKS as u32,
            };
            Ok(RankCall {
                args: ll_bf16::arguments(&call).map_err(err)?,
                bindings: self.bindings(rank),
            })
        })
    }

    fn ll(&self, rows: usize, only: Option<usize>, pdl: bool) -> LabResult<()> {
        self.ll_on(
            &self.ll,
            ll_bf16::SYMBOL,
            (LL_INBOX, LL_EPOCHS),
            rows,
            only,
            pdl,
        )
    }

    /// The F32 low-latency entry over `rows` rows from `REF_PARTIAL` into
    /// `REF_SUMS` on every rank (or `only`).
    fn reference_ll(&self, rows: usize, only: Option<usize>) -> LabResult<()> {
        let elements = (rows * H) as u64;
        let plan = ll_f32::plan(RANKS as u32, elements).map_err(err)?;
        launch::launch_collective(
            self.sessions,
            &self.reference,
            ll_f32::SYMBOL,
            plan,
            only,
            |rank| {
                let call = ll_f32::Call {
                    input: REF_PARTIAL,
                    output: REF_SUMS,
                    peer_inboxes: (0..RANKS as u64).map(|peer| REF_INBOX + peer).collect(),
                    epochs: REF_EPOCHS,
                    elements,
                    capacity_elements: (MAX_ROWS * H) as u64,
                    rank: rank as u32,
                    ranks: RANKS as u32,
                };
                Ok(RankCall {
                    args: ll_f32::arguments(&call).map_err(err)?,
                    bindings: self.bindings(rank),
                })
            },
        )
    }

    /// The call's partials into every rank's `FRESH`, and their host
    /// rounding into `REF_PARTIAL`.
    fn upload(&self, partials: &[Vec<f32>]) -> LabResult<()> {
        poison::synchronize(self.sessions)?;
        for (rank, session) in self.sessions.iter().enumerate() {
            poison::write(
                session,
                self.ranks[rank].buffer(FRESH),
                0,
                &words(&partials[rank]),
            )?;
            poison::write(
                session,
                self.ranks[rank].buffer(REF_PARTIAL),
                0,
                &words(&rounded(&partials[rank])),
            )?;
        }
        Ok(())
    }

    /// Poisons every rank's partials and sums.
    fn poison(&self) -> LabResult<()> {
        poison::synchronize(self.sessions)?;
        for (rank, session) in self.sessions.iter().enumerate() {
            for handle in [PARTIAL, SUMS] {
                poison::fill_all(session, self.ranks[rank].buffer(handle), &[poison::POISON])?;
            }
        }
        Ok(())
    }

    /// Every rank's BF16 sums over `rows` rows equal `expected`.
    fn sums(&self, rows: usize, expected: &[u16], label: &str) -> LabResult<()> {
        poison::synchronize(self.sessions)?;
        for (rank, session) in self.sessions.iter().enumerate() {
            let bytes = poison::read(session, self.ranks[rank].buffer(SUMS))?;
            exact(
                &format!("{label}: rank {rank}"),
                &halves(&bytes[..rows * H * 2]),
                expected,
            )?;
        }
        Ok(())
    }

    /// The host's rounded sums of `partials`, after the F32 entry's sums of
    /// the rounded partials matched the host's F32 sums on every rank.
    fn reference(&self, rows: usize, partials: &[Vec<f32>], label: &str) -> LabResult<Vec<u16>> {
        let f32_sums: Vec<u32> = host_f32(partials).iter().map(|v| v.to_bits()).collect();
        self.upload(partials)?;
        self.reference_ll(rows, None)?;
        poison::synchronize(self.sessions)?;
        for (rank, session) in self.sessions.iter().enumerate() {
            let sums = session.download_f32(self.ranks[rank].buffer(REF_SUMS), rows * H)?;
            let sums: Vec<u32> = sums.iter().map(|v| v.to_bits()).collect();
            exact(
                &format!("{label}: F32 entry over the rounded partials, rank {rank}"),
                &sums,
                &f32_sums,
            )?;
        }
        Ok(host_sums(partials))
    }
}

fn producers<'a>(
    sessions: &[Session],
    ranks: &'a Ranks<'_>,
    rows: usize,
) -> LabResult<Vec<PartialProducer<'a>>> {
    sessions
        .iter()
        .enumerate()
        .map(|(rank, session)| {
            PartialProducer::new(
                session,
                ranks.ranks[rank].buffer(FRESH),
                ranks.ranks[rank].buffer(PARTIAL),
                rows,
            )
        })
        .collect()
}

/// The producer, then the entry (PDL when `pdl`), on every rank.
fn produce_then(
    ranks: &Ranks<'_>,
    producers: &[PartialProducer<'_>],
    rows: usize,
    pdl: bool,
) -> LabResult<()> {
    for (session, producer) in ranks.sessions.iter().zip(producers) {
        producer.launch(session)?;
    }
    ranks.ll(rows, None, pdl)
}

/// One row count on the persistent state: eager and graph calls against
/// the host and the F32 entry, the poisoned replays, and at
/// [`MUTATION_ROWS`] the flag wraps.
fn bucket(ranks: &Ranks<'_>, rows: usize) -> LabResult<Value> {
    let sessions = ranks.sessions;
    let producers = producers(sessions, ranks, rows)?;
    let graphs = RankReplay::new(sessions, |rank, session| {
        producers[rank].launch(session)?;
        ranks.ll(rows, Some(rank), true)
    })?;
    let mut sensitive = Vec::new();
    for (call, graph) in [false, true, true].into_iter().enumerate() {
        let partials = partials(rows, 0xb1f3 + 100 * rows as u64 + call as u64);
        sensitive.push(order_sensitive(&partials));
        let label = format!("{rows} rows, call {call}, graph {graph}");
        let host = ranks.reference(rows, &partials, &label)?;
        ranks.poison()?;
        if graph {
            graphs.launch(sessions)?;
        } else {
            produce_then(ranks, &producers, rows, true)?;
        }
        ranks.sums(rows, &host, &label)?;
    }
    if sensitive.contains(&0) {
        return Err(format!(
            "{rows} rows: the reversed rank order changed no rounded sum"
        ));
    }
    let partials = partials(rows, 0xb5eed + rows as u64);
    let host = ranks.reference(rows, &partials, &format!("{rows} rows, replays"))?;
    let mut replays = sessions
        .iter()
        .zip(&ranks.ranks)
        .map(|(session, rank)| {
            Ok(Replay::new(session)
                .output(
                    "partials",
                    rank.buffer(PARTIAL),
                    WriteMap::of(DType::F32).prefix(rows * H)?,
                )
                .output(
                    "sums",
                    rank.buffer(SUMS),
                    WriteMap::of(DType::Bf16).prefix(rows * H)?,
                ))
        })
        .collect::<LabResult<Vec<_>>>()?;
    for (phase, pdl) in [
        (Phase::Eager, false),
        (Phase::Relaunch, false),
        (Phase::Graph, false),
        (Phase::Pdl, true),
    ] {
        poison::synchronize(sessions)?;
        for replay in &replays {
            replay.prepare(phase)?;
        }
        match phase {
            Phase::Graph => graphs.launch(sessions)?,
            _ => produce_then(ranks, &producers, rows, pdl)?,
        }
        poison::synchronize(sessions)?;
        for (rank, replay) in replays.iter_mut().enumerate() {
            replay
                .finish(phase, pdl)
                .map_err(|e| format!("{rows} rows, rank {rank}: {e}"))?;
        }
        if phase == Phase::Eager {
            ranks.sums(rows, &host, &format!("{rows} rows, eager replay"))?;
        }
    }
    let mut receipt = BTreeMap::new();
    replays[0].record(&mut receipt);
    let mut wrap = Value::Null;
    if MUTATION_ROWS.contains(&rows) {
        wrap = wraps(ranks, rows)?;
    }
    Ok(json!({
        "rows": rows,
        "order_sensitive_sums": sensitive,
        "replay": receipt,
        "flag_wrap": wrap,
    }))
}

/// One rank on one GPU: the late-trigger producer, then at once the entry
/// at `ranks = 1` (PDL). At 1, 4 and 32 rows, over new partials into
/// poisoned sums: the served entry must give the host's sums and each wait
/// mutant some other sum, with the producer's output the call's partials
/// either way.
fn single_rank_mutants(session: &Session) -> LabResult<Value> {
    const INBOX: u64 = 900;
    const EPOCHS: u64 = 901;
    let one = std::slice::from_ref(session);
    // Every entry resolved before any producer launch.
    let mut functions = vec![(
        "served",
        CollectiveModules::load(one, LL_CUBIN)?.function(ll_bf16::SYMBOL)?,
    )];
    for tag in ["late_wait", "no_wait"] {
        functions.push((
            tag,
            CollectiveModules::load(one, &ll_bf16::mutant_cubin(tag))?
                .function(&ll_bf16::mutant_symbol(tag))?,
        ));
    }
    let capacity = (MAX_ROWS * H) as u64;
    let state = ll_bf16::state_bytes(1, capacity).map_err(err)?;
    let f32s = MAX_ROWS * H * 4;
    let (partial, fresh, sums) = (
        session.allocate(f32s)?,
        session.allocate(f32s)?,
        session.allocate(f32s)?,
    );
    let inbox = session.allocate(state.inbox as usize)?;
    let epochs = session.allocate(state.epochs as usize)?;
    for buffer in [&inbox, &epochs] {
        poison::fill_all(session, buffer, &[0])?;
    }
    let mut results = Vec::new();
    for rows in MUTATION_ROWS {
        let producer = PartialProducer::new(session, &fresh, &partial, rows)?;
        let elements = (rows * H) as u64;
        let mut plan = ll_bf16::plan(1, elements).map_err(err)?;
        plan.pdl = true;
        let mut caught = Vec::new();
        for (call, (name, function)) in functions.iter().enumerate() {
            let label = format!("one rank, {rows} rows, {name}");
            let input = partials(rows, 0xb51e + 10 * rows as u64 + call as u64).swap_remove(0);
            let host = host_sums(std::slice::from_ref(&input));
            poison::synchronize(one)?;
            poison::write(session, &fresh, 0, &words(&input))?;
            poison::fill_all(session, &partial, &[poison::POISON])?;
            poison::fill_all(session, &sums, &[poison::POISON])?;
            producer.launch(session)?;
            let mut buffers: BTreeMap<u64, &DeviceBuffer> = BTreeMap::new();
            buffers.insert(PARTIAL, &partial);
            buffers.insert(SUMS, &sums);
            buffers.insert(INBOX, &inbox);
            buffers.insert(EPOCHS, &epochs);
            launch::launch_resolved(one, function, plan, None, |_| {
                let call = ll_bf16::Call {
                    input: PARTIAL,
                    output: SUMS,
                    peer_inboxes: vec![INBOX],
                    epochs: EPOCHS,
                    elements,
                    capacity_elements: capacity,
                    rank: 0,
                    ranks: 1,
                };
                Ok(RankCall {
                    args: ll_bf16::arguments(&call).map_err(err)?,
                    bindings: Bindings {
                        buffers: buffers.clone(),
                    },
                })
            })?;
            poison::synchronize(one)?;
            let produced = session.download_f32(&partial, rows * H)?;
            exact(
                &format!("{label}: the producer's output"),
                &produced.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                &input.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            )?;
            let got = halves(&poison::read(session, &sums)?[..rows * H * 2]);
            match (*name, exact(&label, &got, &host)) {
                ("served", result) => result?,
                (_, Ok(())) => {
                    return Err(format!("{label}: the sums equal the host's"));
                }
                (_, Err(_)) => caught.push(*name),
            }
        }
        eprintln!("one rank, {rows} rows: wait mutants caught: {caught:?}");
        if caught.len() != 2 {
            return Err(format!("one rank, {rows} rows: caught only {caught:?}"));
        }
        results.push(json!({"rows": rows, "caught": caught}));
    }
    Ok(json!(results))
}

/// Every rank's epoch words: one u32 per CTA of the capacity.
fn epochs(ranks: &Ranks<'_>, handle: u64) -> LabResult<Vec<Vec<u32>>> {
    poison::synchronize(ranks.sessions)?;
    ranks
        .sessions
        .iter()
        .enumerate()
        .map(|(rank, session)| {
            let bytes = poison::read(session, ranks.ranks[rank].buffer(handle))?;
            Ok(bytes
                .chunks_exact(4)
                .map(|word| u32::from_ne_bytes(word.try_into().unwrap()))
                .collect())
        })
        .collect()
}

/// One call over new partials (seed `seed`) into poisoned sums, checked on
/// every rank against the host: `Ok(true)` when every rank's sums equal the
/// host's. With `expect_equal` a difference is an error.
fn checked_call(
    ranks: &Ranks<'_>,
    rows: usize,
    seed: u64,
    launch: impl FnOnce() -> LabResult<()>,
    expect_equal: Option<&str>,
) -> LabResult<bool> {
    let partials = partials(rows, seed);
    let host = host_sums(&partials);
    poison::synchronize(ranks.sessions)?;
    for (rank, session) in ranks.sessions.iter().enumerate() {
        poison::write(
            session,
            ranks.ranks[rank].buffer(PARTIAL),
            0,
            &words(&partials[rank]),
        )?;
        poison::fill_all(session, ranks.ranks[rank].buffer(SUMS), &[poison::POISON])?;
    }
    launch()?;
    poison::synchronize(ranks.sessions)?;
    let mut equal = true;
    for (rank, session) in ranks.sessions.iter().enumerate() {
        let bytes = poison::read(session, ranks.ranks[rank].buffer(SUMS))?;
        match (
            exact(
                &format!("rank {rank}"),
                &halves(&bytes[..rows * H * 2]),
                &host,
            ),
            expect_equal,
        ) {
            (Ok(()), _) => {}
            (Err(error), Some(label)) => return Err(format!("{label}: {error}")),
            (Err(_), None) => equal = false,
        }
    }
    Ok(equal)
}

/// Around each wrap of `WRAP_STARTS`: every rank's epochs set just below
/// it, then `WRAP_CALLS` checked calls with new partials; the epochs of the
/// call's CTAs must have advanced by exactly the calls, the others not at
/// all. Then the long graph replay as stress, and the sums again.
fn wraps(ranks: &Ranks<'_>, rows: usize) -> LabResult<Value> {
    let ctas = rows * ROW_CTAS;
    let capacity = MAX_ROWS * ROW_CTAS;
    let mut checked = Vec::new();
    for start in WRAP_STARTS {
        // Every rank is idle and every slot was cleared by its reader.
        poison::synchronize(ranks.sessions)?;
        let image = (0..capacity)
            .flat_map(|_| start.to_ne_bytes())
            .collect::<Vec<_>>();
        for (rank, session) in ranks.sessions.iter().enumerate() {
            poison::write(session, ranks.ranks[rank].buffer(LL_EPOCHS), 0, &image)?;
        }
        for call in 0..WRAP_CALLS {
            let epoch = start.wrapping_add(call + 1);
            checked_call(
                ranks,
                rows,
                0xb3a90 + u64::from(call) + 100 * rows as u64,
                || ranks.ll(rows, None, false),
                Some(&format!("{rows} rows, epoch {epoch:#010x}")),
            )?;
        }
        for (rank, words) in epochs(ranks, LL_EPOCHS)?.into_iter().enumerate() {
            let want = |cta: usize| match cta < ctas {
                true => start.wrapping_add(WRAP_CALLS),
                false => start,
            };
            if let Some(cta) = (0..capacity).find(|&cta| words[cta] != want(cta)) {
                return Err(format!(
                    "{rows} rows from {start:#010x}: rank {rank} CTA {cta} epoch {:#010x}, \
                     expected {:#010x}",
                    words[cta],
                    want(cta)
                ));
            }
        }
        checked.push(json!({"from": start, "calls": WRAP_CALLS}));
    }
    // The stress replay over the last call's partials.
    let host = {
        let mut host = Vec::new();
        for (rank, session) in ranks.sessions.iter().enumerate() {
            host.push(session.download_f32(ranks.ranks[rank].buffer(PARTIAL), rows * H)?);
        }
        host_sums(&host)
    };
    let graphs = RankReplay::new(ranks.sessions, |rank, _| {
        for _ in 0..CALLS_PER_GRAPH {
            ranks.ll(rows, Some(rank), true)?;
        }
        Ok(())
    })?;
    poison::synchronize(ranks.sessions)?;
    for _ in 0..WRAP_REPLAYS {
        graphs.launch(ranks.sessions)?;
    }
    ranks.sums(rows, &host, &format!("{rows} rows, after the replays"))?;
    Ok(json!({"checked": checked, "replayed_calls": CALLS_PER_GRAPH * WRAP_REPLAYS}))
}

/// The four-rank mutants over `STATE_MUTANT_CALLS` calls with new partials
/// into poisoned sums: some call's sums must differ from the host's on some
/// rank (counted only on an observed difference). The stale-flag mutant
/// runs on its own zeroed state; the own-F32 mutant on the served state,
/// whose epochs and slots it keeps as the served entry does.
fn state_mutant(ranks: &Ranks<'_>, tag: &str, rows: usize) -> LabResult<Value> {
    let (modules, state) = match tag {
        "stale" => (&ranks.stale, (STALE_INBOX, STALE_EPOCHS)),
        "own_f32" => (&ranks.own_f32, (LL_INBOX, LL_EPOCHS)),
        _ => return Err(format!("no four-rank mutant {tag}")),
    };
    let symbol = ll_bf16::mutant_symbol(tag);
    let mut differing = Vec::new();
    for call in 0..STATE_MUTANT_CALLS {
        let equal = checked_call(
            ranks,
            rows,
            0xb57a1e + call + 100 * rows as u64,
            || ranks.ll_on(modules, &symbol, state, rows, None, false),
            None,
        )?;
        if !equal {
            differing.push(call);
        }
    }
    if differing.is_empty() {
        return Err(format!(
            "{rows} rows: the {tag} mutant's sums equal the host's in all \
             {STATE_MUTANT_CALLS} calls"
        ));
    }
    Ok(
        json!({"mutant": tag, "rows": rows, "calls": STATE_MUTANT_CALLS, "differing_calls": differing}),
    )
}

/// Every rank's device keeps the largest grid resident, the condition a
/// bundle admits the entry under.
fn residency(sessions: &[Session]) -> LabResult<Value> {
    let plan = ll_bf16::plan(RANKS as u32, (MAX_ROWS * H) as u64).map_err(err)?;
    let grid = plan.grid.iter().product::<u32>();
    let mut resident = Vec::new();
    for (rank, session) in sessions.iter().enumerate() {
        let module = session.module(LL_CUBIN)?;
        let function = module.function(ll_bf16::SYMBOL).map_err(err)?;
        let clusters = function
            .max_active_clusters(launch::config(plan)?, false)
            .map_err(err)?;
        if clusters < grid {
            return Err(format!(
                "rank {rank}: the device keeps {clusters} of the {grid} CTAs resident"
            ));
        }
        resident.push(clusters);
    }
    Ok(json!({"grid": grid, "resident": resident}))
}

fn load(sessions: &[Session]) -> LabResult<Ranks<'_>> {
    Ok(Ranks {
        sessions,
        ranks: sessions.iter().map(Rank::new).collect::<LabResult<_>>()?,
        ll: CollectiveModules::load(sessions, LL_CUBIN)?,
        stale: CollectiveModules::load(sessions, &ll_bf16::mutant_cubin("stale"))?,
        own_f32: CollectiveModules::load(sessions, &ll_bf16::mutant_cubin("own_f32"))?,
        reference: CollectiveModules::load(sessions, F32_CUBIN)?,
    })
}

#[test]
fn allreduce_ll_bf16_cpu_contract() {
    let entries = kernel_lab::cubin_entries(LL_CUBIN).unwrap();
    assert_eq!(entries, [ll_bf16::SYMBOL]);
    for (tag, ..) in ll_bf16::MUTANTS {
        assert_eq!(
            kernel_lab::cubin_entries(&ll_bf16::mutant_cubin(tag)).unwrap(),
            [ll_bf16::mutant_symbol(tag).as_str()]
        );
        assert!(kernel_lab::comm_cubin(&ll_bf16::mutant_cubin(tag)).unwrap());
    }
    assert_eq!(ROW_CTAS, 2);
    for rows in 1..=MAX_ROWS {
        let plan = ll_bf16::plan(RANKS as u32, (rows * H) as u64).unwrap();
        assert_eq!(plan.grid, [(ROW_CTAS * rows) as u32, 1, 1]);
    }
    // The host fold: ties round to even; one rank's sum is its rounding,
    // but for -0, which the sum from F32 zero makes +0.
    assert_eq!(f32_to_bf16(f32::from_bits(0x3f80_8000)), 0x3f80);
    assert_eq!(f32_to_bf16(f32::from_bits(0x3f81_8000)), 0x3f82);
    let one = partials(1, 3).swap_remove(0);
    let alone = host_sums(std::slice::from_ref(&one));
    assert!(one.iter().zip(&alone).all(|(&v, &s)| match v.to_bits() {
        0x8000_0000 => s == 0,
        _ => f32_to_bf16(v) == s,
    }));
    // The rounding of the inputs moves sums: the fold is not the F32 sum
    // rounded.
    let partials = partials(2, 7);
    let unrounded: Vec<u16> = (0..2 * H)
        .map(|i| f32_to_bf16(partials.iter().fold(0.0_f32, |sum, rank| sum + rank[i])))
        .collect();
    assert_ne!(host_sums(&partials), unrounded);
    assert!(order_sensitive(&partials) > 0);
}

#[test]
fn the_ll_bf16_all_reduce_matches_the_host_bitwise_on_four_ranks() {
    kernel_lab::gpu_test_multi(1 << 30, |sessions| {
        if sessions.len() != RANKS {
            return kernel_lab::skip("the low-latency BF16 all-reduce test runs on four GPUs");
        }
        let residency = residency(sessions)?;
        let ranks = load(sessions)?;
        let buckets = (1..=MAX_ROWS)
            .map(|rows| bucket(&ranks, rows))
            .collect::<LabResult<Vec<_>>>()?;
        let mut mutants = Vec::new();
        for tag in ["stale", "own_f32"] {
            for rows in STATE_MUTANT_ROWS {
                mutants.push(state_mutant(&ranks, tag, rows)?);
            }
        }
        // The served entry still matches after the own-F32 mutant's calls
        // on its state.
        for rows in STATE_MUTANT_ROWS {
            checked_call(
                &ranks,
                rows,
                0xb0a7 + rows as u64,
                || ranks.ll(rows, None, false),
                Some(&format!("{rows} rows, after the mutants")),
            )?;
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("allreduce_ll_bf16_bitwise"));
        receipt.insert("residency".into(), residency);
        receipt.insert("buckets".into(), json!(buckets));
        receipt.insert("state_mutants".into(), json!(mutants));
        receipt.insert("passed".into(), json!(true));
        sessions[0].receipt("allreduce_ll_bf16", receipt)?;
        Ok(())
    })
    .expect("the low-latency BF16 all-reduce on four ranks");
}

/// What the CUDA host reports, through the residency query a bundle makes
/// at create, for the entry's launch at `rows` rows on this GPU, beside
/// the driver's own count for the same launch.
fn host_residency(session: &Session, rows: usize) -> LabResult<Value> {
    use bundle_runtime::residency::{self, Verdict};
    use runtime_bundle_abi::{BUNDLE_CLUSTER_RESIDENCY_V1_CLUSTERS_REPORTED, HostServicesTableV1};
    use runtime_host::cuda::CudaHostDeviceV1;
    use runtime_host::{HostInstanceConfigV1, HostServicesV1Backend, HostTopologyV1};
    let plan = ll_bf16::plan(RANKS as u32, (rows * H) as u64).map_err(err)?;
    let config =
        HostInstanceConfigV1::new(HostTopologyV1::new(vec![session.ordinal]).map_err(err)?);
    let backend = std::rc::Rc::new(HostServicesV1Backend::new(
        CudaHostDeviceV1::new().map_err(err)?,
        config,
    ));
    let table = HostServicesTableV1::new(std::rc::Rc::clone(&backend));
    let services = table.caller();
    let context = services.select_context(0, session.ordinal).map_err(err)?;
    let module = services
        .load_cuda_module(context, kernel_lab::cubin(LL_CUBIN)?)
        .map_err(err)?;
    let symbol = services
        .resolve_cuda_symbol(module, ll_bf16::SYMBOL.as_bytes())
        .map_err(err)?;
    let report = residency::query(services, symbol, plan).map_err(err)?;
    let verdict = residency::check(&plan, 1, &report);
    let driver = session
        .module(LL_CUBIN)?
        .function(ll_bf16::SYMBOL)
        .map_err(err)?
        .max_active_clusters(launch::config(plan)?, false)
        .map_err(err)?;
    let grid = plan.grid.iter().product::<u32>();
    let summary = json!({
        "rows": rows,
        "grid": grid,
        "flags": report.flags,
        "max_active_clusters": report.max_active_clusters,
        "multiprocessors": [report.usable_multiprocessors, report.multiprocessor_count],
        "driver_max_active_clusters": driver,
        "verdict": format!("{verdict:?}"),
    });
    eprintln!("host residency: {summary}");
    if report.flags & BUNDLE_CLUSTER_RESIDENCY_V1_CLUSTERS_REPORTED == 0 {
        return Err(format!(
            "{rows} rows: the host reports no residency: {summary}"
        ));
    }
    if report.max_active_clusters != driver {
        return Err(format!(
            "{rows} rows: the host's count is not the driver's: {summary}"
        ));
    }
    if verdict != Verdict::Fits {
        return Err(format!(
            "{rows} rows: the report does not admit the grid: {summary}"
        ));
    }
    Ok(summary)
}

#[test]
fn the_host_reports_the_ll_bf16_grid_resident() {
    kernel_lab::gpu_test(64 << 20, |session| {
        let reports = [2, MAX_ROWS]
            .into_iter()
            .map(|rows| host_residency(session, rows))
            .collect::<LabResult<Vec<_>>>()?;
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("allreduce_ll_bf16_host_residency"));
        receipt.insert("reports".into(), json!(reports));
        receipt.insert("passed".into(), json!(true));
        session.receipt("allreduce_ll_bf16", receipt)?;
        Ok(())
    })
    .expect("the host reports the low-latency BF16 all-reduce's grid resident");
}

/// The wait mutants on one GPU (no peers), behind the late-trigger
/// producer, every entry resolved before the first producer launch (see
/// `allreduce_ll_f32.rs`, whose early-reader control shows that PDL
/// overlaps a dependent with this producer).
#[test]
fn the_ll_bf16_wait_mutants_change_the_sums_on_one_gpu() {
    kernel_lab::gpu_test(1 << 30, |session| {
        let mutants = single_rank_mutants(session)?;
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("allreduce_ll_bf16_wait_mutants"));
        receipt.insert("mutants".into(), mutants);
        receipt.insert("passed".into(), json!(true));
        session.receipt("allreduce_ll_bf16", receipt)?;
        Ok(())
    })
    .expect("the BF16 wait mutants change the sums on one GPU");
}

/// Per-call medians of graphs of 64 calls on every rank.
fn time_calls(sessions: &[Session], graphs: &[runtime_cuda::GraphExec]) -> LabResult<Value> {
    let timing = kernel_lab::time_multi_graph(sessions, graphs, CALLS_PER_GRAPH, 3, 9)?;
    let per_call = |ms: f32| f64::from(ms) * 1000.0 / CALLS_PER_GRAPH as f64;
    Ok(json!({
        "median_us": per_call(timing.critical_path.median_ms),
        "min_us": per_call(timing.critical_path.min_ms),
        "per_rank_median_us": timing.per_rank.iter().map(|t| per_call(t.median_ms)).collect::<Vec<_>>(),
    }))
}

#[test]
fn the_ll_bf16_all_reduce_is_timed_against_the_f32_entry() {
    kernel_lab::gpu_test_multi(1 << 30, |sessions| {
        if sessions.len() != RANKS {
            return kernel_lab::skip("the low-latency BF16 all-reduce timing runs on four GPUs");
        }
        residency(sessions)?;
        let ranks = load(sessions)?;
        let mut rows_timed = Vec::new();
        for rows in TIMED_ROWS {
            let partials = partials(rows, 0xb7173 + rows as u64);
            let host = ranks.reference(rows, &partials, &format!("{rows} rows, timing"))?;
            poison::synchronize(sessions)?;
            for (rank, session) in sessions.iter().enumerate() {
                poison::write(
                    session,
                    ranks.ranks[rank].buffer(PARTIAL),
                    0,
                    &words(&partials[rank]),
                )?;
            }
            let capture = |entry: &dyn Fn(usize) -> LabResult<()>| {
                sessions
                    .iter()
                    .enumerate()
                    .map(|(rank, session)| {
                        session.capture(|_| {
                            for _ in 0..CALLS_PER_GRAPH {
                                entry(rank)?;
                            }
                            Ok(())
                        })
                    })
                    .collect::<LabResult<Vec<_>>>()
            };
            let bf16 = capture(&|rank| ranks.ll(rows, Some(rank), true))?;
            let f32 = capture(&|rank| ranks.reference_ll(rows, Some(rank)))?;
            let bf16_time = time_calls(sessions, &bf16)?;
            let f32_time = time_calls(sessions, &f32)?;
            ranks.sums(rows, &host, &format!("{rows} rows, after timing"))?;
            eprintln!("{rows} rows: low latency BF16 {bf16_time}, F32 {f32_time}");
            rows_timed.push(json!({"rows": rows, "ll_bf16": bf16_time, "ll_f32": f32_time}));
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("allreduce_ll_bf16_latency"));
        receipt.insert("rows".into(), json!(rows_timed));
        sessions[0].receipt("allreduce_ll_bf16_latency", receipt)?;
        Ok(())
    })
    .expect("the low-latency BF16 all-reduce timing");
}
