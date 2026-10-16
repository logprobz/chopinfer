//! The low-latency F32 all-reduce over CTAs (`kernels_sm120::comm::ll_f32`)
//! against the F32 one-shot it replaces (`chop_comm_allreduce_oneshot_f32_v1`
//! in `comm_sm120`), bit for bit, on four GPUs. Run with
//! `CHOPINFER_KERNEL_GPUS=0,1,2,3 cargo test -p kernel-lab --release --test
//! allreduce_ll_f32 -- --test-threads=1`; without GPUs only the CPU contract
//! test runs.
//!
//! - At every row count of the entry's capacity (1 to 32 rows of 4096), on
//!   one persistent low-latency state: every rank's sums equal the host's
//!   rank-order sums from F32 zero, which equal the one-shot's on every rank,
//!   over partials whose sums depend on the rank order (wide exponents, signed
//!   zeros, subnormals; the test checks that the reversed order changes
//!   some). The partials come from a device producer (the lab's late-trigger
//!   PDL probe), which after a 2 ms lead (room for every rank's all-reduce
//!   to be queued; nothing guarantees it is) holds them at NaN, triggers its
//!   dependents and only after a
//!   2 ms tail copies the call's own partials in; the all-reduce follows it
//!   with PDL, eagerly and through
//!   each rank's captured graph (producer, then all-reduce), twice with new
//!   partials.
//! - Poisoned replays (eager, relaunch, graph, PDL) at every row count: the
//!   partials and the sums are poisoned before every launch, the sums must
//!   be written exactly over the call's rows and nothing past them, and every
//!   replay must reproduce the eager bits, which equal the host's.
//! - The wait mutants, on one GPU without peers (see
//!   `the_wait_mutants_change_the_sums_on_one_gpu`), every entry resolved
//!   before the first producer launch (under lazy module loading a later
//!   resolution waits the producer out): a control first, the lab's early
//!   reader behind the late-trigger producer, which must have read the
//!   producer's NaN (PDL overlapped), and a reader resolved after the
//!   producer's launch, recorded only; then at 1, 4 and 32 rows,
//!   the entry at one rank behind the producer: the served entry must give
//!   the host's sums and each mutant, the dependency wait after the input
//!   loads (`comm_ll_f32_late_wait_sm120`) and none
//!   (`comm_ll_f32_no_wait_sm120`), other sums, the producer's output the
//!   call's partials either way. Launch, synchronization and readback
//!   errors propagate rather than count as caught.
//! - The host's residency report: on one GPU, the CUDA host's
//!   `query_cluster_residency_v1` (through `bundle_runtime::residency::
//!   query`, the call a bundle makes at create) for the entry's unclustered
//!   launch at 4 rows (16 CTAs, the largest grid a bundle serves) and 32
//!   rows, and for the round entry's at its served one row (4 CTAs):
//!   reported, equal to the driver's own count for that launch, and
//!   admitting one grid per device.
//! - Residency: before any launch, every rank's device keeps the F32
//!   entry's largest grid (32 rows, 128 CTAs) and the round entry's served
//!   grid (one row, 4 CTAs) resident (`cuOccupancyMaxActiveClusters`), the
//!   condition a bundle admits an entry under, and every rank's host
//!   residency service admits the round entry's grid as above; the test
//!   refuses otherwise.
//! - Around the 12-bit flag wrap and the 32-bit epoch wrap at 1, 4 and 32
//!   rows: every rank's epochs set just below each wrap (every rank idle,
//!   the inbox cleared), then twelve calls, each with new partials into
//!   poisoned sums, each checked against the host; the epochs must then
//!   have advanced by exactly the calls (and the CTAs past the call's rows
//!   not at all). Then a graph of 64 calls replayed 70 times (4,480 calls of
//!   every CTA) as stress, and the sums again.
//! - A stale-flag mutant (`comm_ll_f32_stale_sm120`: its flag ignores the
//!   epoch and its receivers keep what they read), on a state of its own at
//!   1 and 4 rows over six calls with new partials into poisoned sums: some
//!   call's sums must differ from the host's on some rank.
//! - The round entry (`ll_f32::ROUND_SYMBOL`) at every row count, on the
//!   same persistent state after the F32 entry's calls: behind the
//!   producer, eagerly and through each rank's captured graph, every rank's
//!   BF16 sums equal the host's rank-order F32 sums rounded once
//!   (`oracle::all_reduce_rounded`), written exactly over the call's rows,
//!   under poisoned eager, relaunch, graph and PDL replays.
//! - Timing: per-call medians of graphs of 64 calls, the low-latency entry
//!   and its round entry against the one-shot, at 1 to 32 rows (receipt
//!   only).

use kernel_core::contract::{DType, LaunchPlan};
use kernel_core::oracle;
use kernel_lab::launch::{self, Bindings, CollectiveModules, RankCall};
use kernel_lab::poison::{self, Phase, Replay, WriteMap};
use kernel_lab::topology::RankReplay;
use kernel_lab::{LabResult, Session};
use kernels_sm120::comm::{self, OneShotCall, ll_f32};
use runtime_cuda::{DeviceBuffer, Function, KernelArgument, Module};
use serde_json::{Value, json};
use std::collections::BTreeMap;

const RANKS: usize = 4;
const H: usize = 4096;
/// The state's capacity: the entry's most rows.
const MAX_ROWS: usize = 32;
/// The rows SM120 serves the round entry at
/// (`kernels_sm120::backend::ROUNDED_LOW_LATENCY_ROWS`): four CTAs.
const ROUND_ROWS: usize = kernels_sm120::backend::ROUNDED_LOW_LATENCY_ROWS as usize;

const LL_CUBIN: &str = "comm_ll_f32_sm120";
const LATE_WAIT_CUBIN: &str = "comm_ll_f32_late_wait_sm120";
const NO_WAIT_CUBIN: &str = "comm_ll_f32_no_wait_sm120";
const STALE_CUBIN: &str = "comm_ll_f32_stale_sm120";
const COMM_CUBIN: &str = "comm_sm120";
const ONE_SHOT: &str = "chop_comm_allreduce_oneshot_f32_v1";
const PROBE_CUBIN: &str = "probe_pdl_late_trigger_sm120";
/// The partials' producer's lead before it poisons them and triggers its
/// dependents: room for every rank's all-reduce to be queued (the host
/// submits every rank's producer first), not a guarantee that it is. The first four-GPU runs, whose producer
/// triggered at once, never saw the all-reduce overlap it, with a 50 us,
/// a 5 ms or (E23's one-shot) a 200 ms hold after the trigger; they also
/// resolved the all-reduce's entry only after the producer's launch.
const PRODUCER_LEAD_NS: u64 = 2_000_000;
/// Its tail between its trigger and its writes.
const PRODUCER_TAIL_NS: u64 = 2_000_000;
/// Row counts of the mutations and the flag-wrap check.
const MUTATION_ROWS: [usize; 3] = [1, 4, 32];
const TIMED_ROWS: [usize; 7] = [1, 2, 4, 8, 16, 24, 32];
const CALLS_PER_GRAPH: usize = 64;
/// Graph replays of the flag-wrap stress: 64 x 70 calls pass 4096.
const WRAP_REPLAYS: usize = 70;
/// Checked calls around each wrap, from six below it.
const WRAP_CALLS: u32 = 12;
/// Epochs six calls below the 12-bit flag wrap and the 32-bit epoch wrap.
const WRAP_STARTS: [u32; 2] = [4096 - 6, u32::MAX - 5];
/// Row counts and calls of the stale-flag mutant.
const STALE_ROWS: [usize; 2] = [1, 4];
const STALE_CALLS: u64 = 6;

// Buffer handles of one rank.
/// The partials the low-latency entry reads, written by the producer.
const PARTIAL: u64 = 1;
/// The call's partials, which the producer copies into `PARTIAL`.
const FRESH: u64 = 2;
/// The low-latency entry's sums.
const SUMS: u64 = 3;
/// The one-shot's partials and sums.
const ONE_PARTIAL: u64 = 4;
const ONE_SUMS: u64 = 5;
const LL_EPOCHS: u64 = 6;
const ONE_EPOCH: u64 = 7;
/// The stale-flag mutant's own epochs and inboxes (`STALE_INBOX + rank`).
const STALE_EPOCHS: u64 = 8;
/// Every rank's inboxes and one-shot flags: `HANDLE + rank`.
const LL_INBOX: u64 = 100;
const ONE_INBOX: u64 = 200;
const ONE_FLAGS: u64 = 300;
const STALE_INBOX: u64 = 400;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn words(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn exact(label: &str, actual: &[u32], expected: &[u32]) -> LabResult<()> {
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
            "{label}: element {at} is {:#010x}, expected {:#010x}",
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

/// Every rank's partials of `rows` rows: mostly values over 41 binary
/// orders of magnitude, so the sum depends on the order, with signed zeros
/// and subnormals among them.
fn partials(rows: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Rng(seed);
    (0..RANKS)
        .map(|_| {
            (0..rows * H)
                .map(|i| {
                    let state = rng.next();
                    let sign = if state & 1 == 0 { 1.0 } else { -1.0 };
                    match i % 997 {
                        0 => 0.0,
                        1 => -0.0,
                        2 => sign * f32::from_bits(1 + ((state >> 8) % 0x7f_ffff) as u32),
                        _ => {
                            let unit = (state >> 40) as f32 / (1u64 << 24) as f32;
                            sign * unit * 2f32.powi(((state >> 8) % 41) as i32 - 20)
                        }
                    }
                })
                .collect()
        })
        .collect()
}

/// The rank-order sums from F32 zero, as the one-shot adds.
fn host_sums(partials: &[Vec<f32>]) -> Vec<u32> {
    (0..partials[0].len())
        .map(|i| {
            let mut sum = 0.0_f32;
            for rank in partials {
                sum += rank[i];
            }
            sum.to_bits()
        })
        .collect()
}

/// How many sums the reversed rank order changes: the inputs must make the
/// order matter.
fn order_sensitive(partials: &[Vec<f32>]) -> usize {
    (0..partials[0].len())
        .filter(|&i| {
            let mut forward = 0.0_f32;
            let mut reverse = 0.0_f32;
            for rank in 0..RANKS {
                forward += partials[rank][i];
                reverse += partials[RANKS - 1 - rank][i];
            }
            forward.to_bits() != reverse.to_bits()
        })
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

    /// One launch, PDL after the stream's previous launch.
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

/// One rank's buffers: the low-latency state (zeroed once, kept across
/// every call of the test) and a one-shot state (zeroed per row count, as
/// its flag offsets follow the tile count).
struct Rank {
    buffers: BTreeMap<u64, DeviceBuffer>,
}

impl Rank {
    fn new(session: &Session) -> LabResult<Self> {
        let capacity = (MAX_ROWS * H) as u64;
        let state = ll_f32::state_bytes(RANKS as u32, capacity).map_err(err)?;
        let one = comm_state_bytes(MAX_ROWS);
        let mut buffers = BTreeMap::new();
        let f32s = MAX_ROWS * H * 4;
        for handle in [PARTIAL, FRESH, SUMS, ONE_PARTIAL, ONE_SUMS] {
            buffers.insert(handle, session.allocate(f32s)?);
        }
        buffers.insert(LL_INBOX, session.allocate(state.inbox as usize)?);
        buffers.insert(LL_EPOCHS, session.allocate(state.epochs as usize)?);
        buffers.insert(STALE_INBOX, session.allocate(state.inbox as usize)?);
        buffers.insert(STALE_EPOCHS, session.allocate(state.epochs as usize)?);
        buffers.insert(ONE_INBOX, session.allocate(one.0)?);
        buffers.insert(ONE_FLAGS, session.allocate(one.1)?);
        buffers.insert(ONE_EPOCH, session.allocate(one.2)?);
        for handle in [LL_INBOX, LL_EPOCHS, STALE_INBOX, STALE_EPOCHS] {
            poison::fill_all(session, &buffers[&handle], &[0])?;
        }
        Ok(Self { buffers })
    }

    fn buffer(&self, handle: u64) -> &DeviceBuffer {
        &self.buffers[&handle]
    }
}

/// The one-shot's inbox, flags and epoch bytes at `rows` rows
/// (`comm.cuh` `allreduce_oneshot`: one tile per row).
fn comm_state_bytes(rows: usize) -> (usize, usize, usize) {
    (RANKS * rows * H * 4, 2 * RANKS * rows * 4, rows * 4)
}

/// Every rank's state, loaded for collective calls.
struct Ranks<'a> {
    sessions: &'a [Session],
    ranks: Vec<Rank>,
    ll: CollectiveModules,
    stale: CollectiveModules,
    comm: CollectiveModules,
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
            for handle in [LL_INBOX, ONE_INBOX, ONE_FLAGS, STALE_INBOX] {
                buffers.insert(handle + peer as u64, state.buffer(handle));
            }
        }
        Bindings { buffers }
    }

    /// The low-latency entry `symbol` of `modules` over `rows` rows on every
    /// rank (or `only`).
    fn ll_with(
        &self,
        modules: &CollectiveModules,
        symbol: &str,
        rows: usize,
        only: Option<usize>,
        pdl: bool,
    ) -> LabResult<()> {
        self.ll_on(modules, symbol, (LL_INBOX, LL_EPOCHS), rows, only, pdl)
    }

    /// `ll_with` over the state whose inboxes start at handle `state.0` and
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
        let mut plan = ll_f32::plan(RANKS as u32, elements).map_err(err)?;
        plan.pdl = pdl;
        launch::launch_collective(self.sessions, modules, symbol, plan, only, |rank| {
            let call = ll_f32::Call {
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
                args: ll_f32::arguments(&call).map_err(err)?,
                bindings: self.bindings(rank),
            })
        })
    }

    fn ll(&self, rows: usize, only: Option<usize>, pdl: bool) -> LabResult<()> {
        self.ll_with(&self.ll, ll_f32::SYMBOL, rows, only, pdl)
    }

    /// The round entry over `rows` rows into `SUMS` as BF16.
    fn ll_round(&self, rows: usize, only: Option<usize>, pdl: bool) -> LabResult<()> {
        self.ll_with(&self.ll, ll_f32::ROUND_SYMBOL, rows, only, pdl)
    }

    /// The F32 one-shot over `rows` rows from `ONE_PARTIAL` into `ONE_SUMS`
    /// on every rank (or `only`).
    fn one_shot(&self, rows: usize, only: Option<usize>) -> LabResult<()> {
        let elements = (rows * H) as u64;
        // The one-shot's geometry at any row count (comm::plan's at the
        // counts it serves): one CTA per 4096-element tile.
        let plan = LaunchPlan {
            grid: [rows as u32, 1, 1],
            block: [256, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        launch::launch_collective(self.sessions, &self.comm, ONE_SHOT, plan, only, |rank| {
            Ok(RankCall {
                args: comm::arguments(OneShotCall {
                    input: ONE_PARTIAL,
                    output: ONE_SUMS,
                    peer_inboxes: (0..RANKS as u64).map(|peer| ONE_INBOX + peer).collect(),
                    peer_flags: (0..RANKS as u64).map(|peer| ONE_FLAGS + peer).collect(),
                    epoch: ONE_EPOCH,
                    elements,
                    rank: rank as u32,
                    ranks: RANKS as u32,
                })
                .map_err(err)?,
                bindings: self.bindings(rank),
            })
        })
    }

    /// Zeroes every rank's one-shot flags and epochs (every rank idle).
    fn reset_one_shot(&self) -> LabResult<()> {
        poison::synchronize(self.sessions)?;
        for (rank, session) in self.sessions.iter().enumerate() {
            for handle in [ONE_FLAGS, ONE_EPOCH] {
                poison::fill_all(session, self.ranks[rank].buffer(handle), &[0])?;
            }
        }
        Ok(())
    }

    /// The call's partials into every rank's `FRESH` and `ONE_PARTIAL`.
    fn upload(&self, partials: &[Vec<f32>]) -> LabResult<()> {
        poison::synchronize(self.sessions)?;
        for (rank, session) in self.sessions.iter().enumerate() {
            for handle in [FRESH, ONE_PARTIAL] {
                poison::write(
                    session,
                    self.ranks[rank].buffer(handle),
                    0,
                    &words(&partials[rank]),
                )?;
            }
        }
        Ok(())
    }

    /// Poisons every rank's low-latency partials and sums.
    fn poison(&self) -> LabResult<()> {
        poison::synchronize(self.sessions)?;
        for (rank, session) in self.sessions.iter().enumerate() {
            for handle in [PARTIAL, SUMS] {
                poison::fill_all(session, self.ranks[rank].buffer(handle), &[poison::POISON])?;
            }
        }
        Ok(())
    }

    /// Every rank's sums in `handle` over `rows` rows equal `expected`.
    fn sums(&self, handle: u64, rows: usize, expected: &[u32], label: &str) -> LabResult<()> {
        poison::synchronize(self.sessions)?;
        for (rank, session) in self.sessions.iter().enumerate() {
            let sums = session.download_f32(self.ranks[rank].buffer(handle), rows * H)?;
            exact(&format!("{label}: rank {rank}"), &bits(&sums), expected)?;
        }
        Ok(())
    }

    /// Every rank's BF16 sums in `SUMS` over `rows` rows equal `expected`.
    fn rounded_sums(&self, rows: usize, expected: &[u16], label: &str) -> LabResult<()> {
        poison::synchronize(self.sessions)?;
        let expected = expected.iter().map(|&b| u32::from(b)).collect::<Vec<_>>();
        for (rank, session) in self.sessions.iter().enumerate() {
            let bytes = poison::read(session, self.ranks[rank].buffer(SUMS))?;
            let sums = bytes[..rows * H * 2]
                .chunks_exact(2)
                .map(|b| u32::from(u16::from_ne_bytes([b[0], b[1]])))
                .collect::<Vec<_>>();
            exact(&format!("{label}: rank {rank}"), &sums, &expected)?;
        }
        Ok(())
    }

    /// The one-shot's sums of `partials` on every rank: the reference.
    fn reference(&self, rows: usize, partials: &[Vec<f32>], label: &str) -> LabResult<Vec<u32>> {
        let host = host_sums(partials);
        self.upload(partials)?;
        self.one_shot(rows, None)?;
        self.sums(
            ONE_SUMS,
            rows,
            &host,
            &format!("{label}: one-shot against the host"),
        )?;
        Ok(host)
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

/// The producer, then the entry (PDL when `pdl`), on every rank: the
/// round entry where `round`.
fn produce_then(
    ranks: &Ranks<'_>,
    producers: &[PartialProducer<'_>],
    rows: usize,
    pdl: bool,
    round: bool,
) -> LabResult<()> {
    for (session, producer) in ranks.sessions.iter().zip(producers) {
        producer.launch(session)?;
    }
    match round {
        true => ranks.ll_round(rows, None, pdl),
        false => ranks.ll(rows, None, pdl),
    }
}

/// One row count on the persistent state: eager and graph calls against
/// the one-shot and the host, the poisoned replays, and at
/// [`MUTATION_ROWS`] the mutations and the flag wrap.
fn bucket(ranks: &Ranks<'_>, rows: usize) -> LabResult<Value> {
    let sessions = ranks.sessions;
    ranks.reset_one_shot()?;
    let producers = producers(sessions, ranks, rows)?;
    let graphs = RankReplay::new(sessions, |rank, session| {
        producers[rank].launch(session)?;
        ranks.ll(rows, Some(rank), true)
    })?;
    let mut sensitive = Vec::new();
    for (call, graph) in [false, true, true].into_iter().enumerate() {
        let partials = partials(rows, 0x11f3 + 100 * rows as u64 + call as u64);
        sensitive.push(order_sensitive(&partials));
        let label = format!("{rows} rows, call {call}, graph {graph}");
        let host = ranks.reference(rows, &partials, &label)?;
        ranks.poison()?;
        if graph {
            graphs.launch(sessions)?;
        } else {
            produce_then(ranks, &producers, rows, true, false)?;
        }
        ranks.sums(SUMS, rows, &host, &label)?;
    }
    if sensitive.contains(&0) {
        return Err(format!(
            "{rows} rows: the reversed rank order changed no sum"
        ));
    }
    // The poisoned replays over the last partials.
    let partials = partials(rows, 0x5eed + rows as u64);
    let host = ranks.reference(rows, &partials, &format!("{rows} rows, replays"))?;
    let map = || WriteMap::of(DType::F32).prefix(rows * H);
    let mut replays = sessions
        .iter()
        .zip(&ranks.ranks)
        .map(|(session, rank)| {
            Ok(Replay::new(session)
                .output("partials", rank.buffer(PARTIAL), map()?)
                .output("sums", rank.buffer(SUMS), map()?))
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
            _ => produce_then(ranks, &producers, rows, pdl, false)?,
        }
        poison::synchronize(sessions)?;
        for (rank, replay) in replays.iter_mut().enumerate() {
            replay
                .finish(phase, pdl)
                .map_err(|e| format!("{rows} rows, rank {rank}: {e}"))?;
        }
        if phase == Phase::Eager {
            ranks.sums(SUMS, rows, &host, &format!("{rows} rows, eager replay"))?;
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

/// The round entry at `rows` rows on the persistent state: an eager call
/// and a graph call behind the producer, then the poisoned replays, every
/// rank's BF16 sums the host's rounded rank-order sums.
fn round_bucket(ranks: &Ranks<'_>, rows: usize) -> LabResult<Value> {
    let sessions = ranks.sessions;
    let producers = producers(sessions, ranks, rows)?;
    let graphs = RankReplay::new(sessions, |rank, session| {
        producers[rank].launch(session)?;
        ranks.ll_round(rows, Some(rank), true)
    })?;
    let rounded =
        |partials: &[Vec<f32>]| oracle::all_reduce_rounded(partials).map_err(|e| e.0.to_string());
    for graph in [false, true] {
        let partials = partials(rows, 0x70d0 + 10 * rows as u64 + u64::from(graph));
        let expected = rounded(&partials)?;
        ranks.upload(&partials)?;
        ranks.poison()?;
        match graph {
            true => graphs.launch(sessions)?,
            false => produce_then(ranks, &producers, rows, true, true)?,
        }
        ranks.rounded_sums(
            rows,
            &expected,
            &format!("{rows} rows, round, graph {graph}"),
        )?;
    }
    let partials = partials(rows, 0x70de + rows as u64);
    let expected = rounded(&partials)?;
    ranks.upload(&partials)?;
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
                    "rounded sums",
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
            _ => produce_then(ranks, &producers, rows, pdl, true)?,
        }
        poison::synchronize(sessions)?;
        for (rank, replay) in replays.iter_mut().enumerate() {
            replay
                .finish(phase, pdl)
                .map_err(|e| format!("{rows} rows, round, rank {rank}: {e}"))?;
        }
        if phase == Phase::Eager {
            ranks.rounded_sums(
                rows,
                &expected,
                &format!("{rows} rows, round, eager replay"),
            )?;
        }
    }
    let mut receipt = BTreeMap::new();
    replays[0].record(&mut receipt);
    Ok(json!({"rows": rows, "replay": receipt}))
}

/// One rank on one GPU: the late-trigger producer, then at once the entry
/// at `ranks = 1` (PDL). At 1, 4 and 32 rows, over new partials into
/// poisoned sums: the served entry must give the host's sums (`0 + x`) and
/// each wait mutant some other sum, with the producer's output the call's
/// partials either way. Launch, synchronization and readback errors
/// propagate; a mutant without a differing sum fails.
fn single_rank_mutants(session: &Session) -> LabResult<Value> {
    const INBOX: u64 = 900;
    const EPOCHS: u64 = 901;
    let one = std::slice::from_ref(session);
    // Every entry resolved before any producer launch: resolving one after
    // it loads the entry (lazy module loading) and so waits the producer out.
    let (served, late_wait, no_wait) = (
        CollectiveModules::load(one, LL_CUBIN)?.function(ll_f32::SYMBOL)?,
        CollectiveModules::load(one, LATE_WAIT_CUBIN)?.function(ll_f32::LATE_WAIT_SYMBOL)?,
        CollectiveModules::load(one, NO_WAIT_CUBIN)?.function(ll_f32::NO_WAIT_SYMBOL)?,
    );
    let capacity = (MAX_ROWS * H) as u64;
    let state = ll_f32::state_bytes(1, capacity).map_err(err)?;
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
        let mut plan = ll_f32::plan(1, elements).map_err(err)?;
        plan.pdl = true;
        let mut caught = Vec::new();
        for (call, (name, function)) in [
            ("served", &served),
            ("late wait", &late_wait),
            ("no wait", &no_wait),
        ]
        .into_iter()
        .enumerate()
        {
            let label = format!("one rank, {rows} rows, {name}");
            let input = partials(rows, 0x51e + 10 * rows as u64 + call as u64).swap_remove(0);
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
                let call = ll_f32::Call {
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
                    args: ll_f32::arguments(&call).map_err(err)?,
                    bindings: Bindings {
                        buffers: buffers.clone(),
                    },
                })
            })?;
            poison::synchronize(one)?;
            let produced = session.download_f32(&partial, rows * H)?;
            exact(
                &format!("{label}: the producer's output"),
                &bits(&produced),
                &bits(&input),
            )?;
            let got = bits(&session.download_f32(&sums, rows * H)?);
            match (name, exact(&label, &got, &host)) {
                ("served", result) => result?,
                (_, Ok(())) => {
                    return Err(format!("{label}: the sums equal the host's"));
                }
                (_, Err(_)) => caught.push(name),
            }
        }
        eprintln!("one rank, {rows} rows: wait mutants caught: {caught:?}");
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
        let sums = session.download_f32(ranks.ranks[rank].buffer(SUMS), rows * H)?;
        match (
            exact(&format!("rank {rank}"), &bits(&sums), &host),
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
    let ctas = rows * 4;
    let mut checked = Vec::new();
    for start in WRAP_STARTS {
        // Every rank is idle and every slot was cleared by its reader.
        poison::synchronize(ranks.sessions)?;
        let capacity = MAX_ROWS * 4;
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
                0x3a90 + u64::from(call) + 100 * rows as u64,
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
    ranks.sums(
        SUMS,
        rows,
        &host,
        &format!("{rows} rows, after the replays"),
    )?;
    Ok(json!({"checked": checked, "replayed_calls": CALLS_PER_GRAPH * WRAP_REPLAYS}))
}

/// The stale-flag mutant on its own zeroed state: over `STALE_CALLS` calls
/// with new partials into poisoned sums, some call's sums must differ from
/// the host's on some rank (counted only on an observed difference;
/// launch, synchronization and readback errors propagate).
fn stale_mutant(ranks: &Ranks<'_>, rows: usize) -> LabResult<Value> {
    let mut differing = Vec::new();
    for call in 0..STALE_CALLS {
        let equal = checked_call(
            ranks,
            rows,
            0x57a1e + call + 100 * rows as u64,
            || {
                ranks.ll_on(
                    &ranks.stale,
                    ll_f32::STALE_SYMBOL,
                    (STALE_INBOX, STALE_EPOCHS),
                    rows,
                    None,
                    false,
                )
            },
            None,
        )?;
        if !equal {
            differing.push(call);
        }
    }
    if differing.is_empty() {
        return Err(format!(
            "{rows} rows: the stale-flag mutant's sums equal the host's in all {STALE_CALLS} calls"
        ));
    }
    Ok(json!({"rows": rows, "calls": STALE_CALLS, "differing_calls": differing}))
}

/// Every rank's device keeps entry `symbol`'s grid at `rows` rows resident
/// (`cuOccupancyMaxActiveClusters`), the condition the bundle admits the
/// entry under.
fn residency(sessions: &[Session], symbol: &str, rows: usize) -> LabResult<Value> {
    let plan = ll_f32::plan(RANKS as u32, (rows * H) as u64).map_err(err)?;
    let grid = plan.grid.iter().product::<u32>();
    let mut resident = Vec::new();
    for (rank, session) in sessions.iter().enumerate() {
        let module = session.module(LL_CUBIN)?;
        let function = module.function(symbol).map_err(err)?;
        let clusters = function
            .max_active_clusters(launch::config(plan)?, false)
            .map_err(err)?;
        if clusters < grid {
            return Err(format!(
                "rank {rank}: the device keeps {clusters} of {symbol}'s {grid} CTAs resident"
            ));
        }
        resident.push(clusters);
    }
    Ok(json!({"symbol": symbol, "rows": rows, "grid": grid, "resident": resident}))
}

fn load(sessions: &[Session]) -> LabResult<Ranks<'_>> {
    Ok(Ranks {
        sessions,
        ranks: sessions.iter().map(Rank::new).collect::<LabResult<_>>()?,
        ll: CollectiveModules::load(sessions, LL_CUBIN)?,
        stale: CollectiveModules::load(sessions, STALE_CUBIN)?,
        comm: CollectiveModules::load(sessions, COMM_CUBIN)?,
    })
}

#[test]
fn allreduce_ll_f32_cpu_contract() {
    let entries = kernel_lab::cubin_entries(LL_CUBIN).unwrap();
    assert_eq!(entries, [ll_f32::SYMBOL, ll_f32::ROUND_SYMBOL]);
    assert!(kernel_lab::comm_cubin(LATE_WAIT_CUBIN).unwrap());
    assert!(kernel_lab::comm_cubin(NO_WAIT_CUBIN).unwrap());
    assert!(kernel_lab::comm_cubin(STALE_CUBIN).unwrap());
    for rows in 1..=MAX_ROWS {
        let plan = ll_f32::plan(RANKS as u32, (rows * H) as u64).unwrap();
        assert_eq!(plan.grid, [4 * rows as u32, 1, 1]);
    }
    let sums = host_sums(&partials(2, 7));
    assert_eq!(sums.len(), 2 * H);
    assert!(order_sensitive(&partials(1, 7)) > 0);
}

#[test]
fn the_ll_all_reduce_matches_the_one_shot_bitwise_on_four_ranks() {
    kernel_lab::gpu_test_multi(1 << 30, |sessions| {
        if sessions.len() != RANKS {
            return kernel_lab::skip("the low-latency F32 all-reduce test runs on four GPUs");
        }
        // The F32 entry's largest grid, and the round entry's served one
        // row (four CTAs) through the host's residency service and the
        // driver on every rank.
        let residency = json!({
            "f32": residency(sessions, ll_f32::SYMBOL, MAX_ROWS)?,
            "round": residency(sessions, ll_f32::ROUND_SYMBOL, ROUND_ROWS)?,
            "round_host": sessions
                .iter()
                .map(|session| host_residency(session, ll_f32::ROUND_SYMBOL, ROUND_ROWS))
                .collect::<LabResult<Vec<_>>>()?,
        });
        let ranks = load(sessions)?;
        let buckets = (1..=MAX_ROWS)
            .map(|rows| bucket(&ranks, rows))
            .collect::<LabResult<Vec<_>>>()?;
        let rounded = (1..=MAX_ROWS)
            .map(|rows| round_bucket(&ranks, rows))
            .collect::<LabResult<Vec<_>>>()?;
        let stale = STALE_ROWS
            .iter()
            .map(|&rows| stale_mutant(&ranks, rows))
            .collect::<LabResult<Vec<_>>>()?;
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("allreduce_ll_f32_bitwise"));
        receipt.insert("residency".into(), residency);
        receipt.insert("buckets".into(), json!(buckets));
        receipt.insert("round_buckets".into(), json!(rounded));
        receipt.insert("stale_mutant".into(), json!(stale));
        sessions[0].receipt("allreduce_ll_f32", receipt)?;
        Ok(())
    })
    .expect("the low-latency F32 all-reduce on four ranks");
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

/// What the CUDA host reports, through the residency query a bundle makes
/// at create, for entry `symbol`'s launch at `rows` rows on this GPU,
/// beside the driver's own count for the same launch.
fn host_residency(session: &Session, symbol: &str, rows: usize) -> LabResult<Value> {
    use bundle_runtime::residency::{self, Verdict};
    use runtime_bundle_abi::{BUNDLE_CLUSTER_RESIDENCY_V1_CLUSTERS_REPORTED, HostServicesTableV1};
    use runtime_host::cuda::CudaHostDeviceV1;
    use runtime_host::{HostInstanceConfigV1, HostServicesV1Backend, HostTopologyV1};
    let plan = ll_f32::plan(RANKS as u32, (rows * H) as u64).map_err(err)?;
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
    let resolved = services
        .resolve_cuda_symbol(module, symbol.as_bytes())
        .map_err(err)?;
    let report = residency::query(services, resolved, plan).map_err(err)?;
    let verdict = residency::check(&plan, 1, &report);
    let driver = session
        .module(LL_CUBIN)?
        .function(symbol)
        .map_err(err)?
        .max_active_clusters(launch::config(plan)?, false)
        .map_err(err)?;
    let grid = plan.grid.iter().product::<u32>();
    let summary = json!({
        "symbol": symbol,
        "rows": rows,
        "grid": grid,
        "cluster": plan.cluster,
        "flags": report.flags,
        "max_active_clusters": report.max_active_clusters,
        "multiprocessors": [report.usable_multiprocessors, report.multiprocessor_count],
        "driver_max_active_clusters": driver,
        "verdict": format!("{verdict:?}"),
    });
    eprintln!("host residency: {summary}");
    if report.flags & BUNDLE_CLUSTER_RESIDENCY_V1_CLUSTERS_REPORTED == 0 {
        return Err(format!(
            "{symbol} at {rows} rows: the host reports no residency: {summary}"
        ));
    }
    if report.max_active_clusters != driver {
        return Err(format!(
            "{symbol} at {rows} rows: the host's count is not the driver's: {summary}"
        ));
    }
    if verdict != Verdict::Fits {
        return Err(format!(
            "{symbol} at {rows} rows: the report does not admit the grid: {summary}"
        ));
    }
    Ok(summary)
}

#[test]
fn the_host_reports_the_ll_grid_resident() {
    kernel_lab::gpu_test(64 << 20, |session| {
        let reports = [
            (ll_f32::SYMBOL, 4),
            (ll_f32::SYMBOL, MAX_ROWS),
            (ll_f32::ROUND_SYMBOL, ROUND_ROWS),
        ]
        .into_iter()
        .map(|(symbol, rows)| host_residency(session, symbol, rows))
        .collect::<LabResult<Vec<_>>>()?;
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("allreduce_ll_f32_host_residency"));
        receipt.insert("reports".into(), json!(reports));
        receipt.insert("passed".into(), json!(true));
        session.receipt("allreduce_ll_f32", receipt)?;
        Ok(())
    })
    .expect("the host reports the low-latency all-reduce's grid resident");
}

/// The control for the one-GPU mutants: the late-trigger producer, then at
/// once the lab's early reader (`chop_probe_pdl_early_reader_v1`, which
/// reads its input before its PDL wait and stores it after), at 32 rows.
/// With the reader resolved before the producer's launch it must have read
/// the producer's NaN: PDL overlapped the two. A second reader, from its own
/// module and resolved only after the producer's launch, is recorded, not
/// required: under lazy module loading that resolution loads it, which can
/// wait the producer out (the earlier mutant runs resolved theirs so).
fn early_reader_control(session: &Session) -> LabResult<Value> {
    let count = MAX_ROWS * H;
    let (partial, fresh, seen) = (
        session.allocate(count * 4)?,
        session.allocate(count * 4)?,
        session.allocate(count * 4)?,
    );
    let input = partials(MAX_ROWS, 0xea51).swap_remove(0);
    poison::write(session, &fresh, 0, &words(&input))?;
    let producer = PartialProducer::new(session, &fresh, &partial, MAX_ROWS)?;
    let module = session.module(PROBE_CUBIN)?;
    let reader = module
        .function("chop_probe_pdl_early_reader_v1")
        .map_err(err)?;
    let early = early_reads(session, &producer, || Ok(&reader), &partial, &seen, count)?;
    if early == 0 {
        return Err(format!(
            "PDL did not overlap the early reader with its producer: it read none of the \
             producer's NaN ({count} words)"
        ));
    }
    let late_module = session.module(PROBE_CUBIN)?;
    let mut late_reader = None;
    let late = early_reads(
        session,
        &producer,
        || {
            Ok(&*late_reader.insert(
                late_module
                    .function("chop_probe_pdl_early_reader_v1")
                    .map_err(err)?,
            ))
        },
        &partial,
        &seen,
        count,
    )?;
    Ok(json!({
        "words": count,
        "read_before_the_producer_wrote": early,
        "resolved_after_the_producer_launch": late,
    }))
}

/// One producer launch, then the reader `reader()` returns (called after
/// the producer's launch) over `count` words: the words it read before the
/// producer wrote them (the producer's NaN).
fn early_reads<'f>(
    session: &Session,
    producer: &PartialProducer<'_>,
    reader: impl FnOnce() -> LabResult<&'f Function>,
    partial: &DeviceBuffer,
    seen: &DeviceBuffer,
    count: usize,
) -> LabResult<usize> {
    poison::synchronize(std::slice::from_ref(session))?;
    poison::fill_all(session, partial, &[poison::POISON])?;
    poison::fill_all(session, seen, &[0])?;
    producer.launch(session)?;
    reader()?
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
                KernelArgument::Buffer(partial),
                KernelArgument::Buffer(seen),
                KernelArgument::U64(count as u64),
            ],
        )
        .map_err(err)?;
    poison::synchronize(std::slice::from_ref(session))?;
    let read = bits(&session.download_f32(seen, count)?);
    Ok(read.iter().filter(|&&word| word == 0x7fc1_7fc1).count())
}

/// The wait mutants on one GPU (no peers): first the early-reader control
/// must show that PDL overlaps a dependent with the late-trigger producer,
/// then each mutant must give sums other than the host's at 1, 4 and 32
/// rows while the served entry gives the host's. Every entry is resolved
/// before the first producer launch: the first one-GPU run resolved each
/// mutant after its producer's launch, and its late-wait mutant read the
/// producer's final output although the control overlapped. The four-GPU
/// runs (E23's one-shot with a 200 ms hold; this lab with a 50 us or 5 ms
/// tail and a 2 ms lead) resolved theirs the same way, so their lack of
/// overlap does not show that four GPUs cannot overlap. The four-rank test
/// still runs no wait mutant; the mutants' reads before the wait do not
/// depend on the rank count.
#[test]
fn the_wait_mutants_change_the_sums_on_one_gpu() {
    kernel_lab::gpu_test(1 << 30, |session| {
        let control = early_reader_control(session)?;
        eprintln!("early reader control: {control}");
        let mutants = single_rank_mutants(session)?;
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("allreduce_ll_f32_wait_mutants"));
        receipt.insert("control".into(), control);
        receipt.insert("mutants".into(), mutants);
        receipt.insert("passed".into(), json!(true));
        session.receipt("allreduce_ll_f32", receipt)?;
        Ok(())
    })
    .expect("the wait mutants change the sums on one GPU");
}

#[test]
fn the_ll_all_reduce_is_timed_against_the_one_shot() {
    kernel_lab::gpu_test_multi(1 << 30, |sessions| {
        if sessions.len() != RANKS {
            return kernel_lab::skip("the low-latency F32 all-reduce timing runs on four GPUs");
        }
        residency(sessions, ll_f32::SYMBOL, MAX_ROWS)?;
        let ranks = load(sessions)?;
        let mut rows_timed = Vec::new();
        for rows in TIMED_ROWS {
            let partials = partials(rows, 0x7173 + rows as u64);
            ranks.reset_one_shot()?;
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
            let ll = capture(&|rank| ranks.ll(rows, Some(rank), true))?;
            let round = capture(&|rank| ranks.ll_round(rows, Some(rank), true))?;
            let one = capture(&|rank| ranks.one_shot(rows, Some(rank)))?;
            let round_time = time_calls(sessions, &round)?;
            let ll_time = time_calls(sessions, &ll)?;
            let one_time = time_calls(sessions, &one)?;
            ranks.sums(SUMS, rows, &host, &format!("{rows} rows, after timing"))?;
            ranks.sums(
                ONE_SUMS,
                rows,
                &host,
                &format!("{rows} rows, one-shot after timing"),
            )?;
            eprintln!(
                "{rows} rows: low latency {ll_time}, round {round_time}, one-shot {one_time}"
            );
            rows_timed.push(json!({
                "rows": rows,
                "ll": ll_time,
                "ll_round": round_time,
                "one_shot": one_time,
            }));
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("allreduce_ll_f32_latency"));
        receipt.insert("rows".into(), json!(rows_timed));
        sessions[0].receipt("allreduce_ll_f32_latency", receipt)?;
        Ok(())
    })
    .expect("the low-latency F32 all-reduce timing");
}
