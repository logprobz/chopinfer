//! Captured eager prefill programs, by everything the programs depend on
//! besides the instance (`Key`): a prefill whose key ran before replays
//! the graphs its second run captured instead of lowering and launching
//! its programs again. Equal keys lower to equal launches, so a replay
//! issues exactly what the eager run would, bit for bit; it only saves the
//! host the lowering and one launch call per kernel (about 1,750 per rank
//! for a short prompt on the GPU). Nothing is bucketed: a key is exact
//! rows, so a prompt length seen once stays eager. Repeated shapes are
//! what repeats: a prompt-logprob request's scored chunks of
//! `SCORED_ROWS` rows, a long prompt's full chunks, the same prompt length
//! again.
//!
//! A key captures on its second run, not its first: a capture costs more
//! than an eager run (the same launch calls into the capture, then
//! instantiation), so a shape that never repeats would only pay for it.
//! The store remembers the `SEEN` most recent keys that ran once.
//!
//! A request's last prompt chunk is the exception (`Key::pinned`): one
//! item's rows after its last anchor, which the host places on the prefill
//! granule, so at most `PREFILL_GRANULE` rows scoring the last one. Every
//! request runs one, at one of only 64 lengths, so each length repeats
//! across requests while the store's newest run would hold only `CAPTURED`
//! of them. Such a key captures at its first run, which on the GPU costs
//! about what its eager run does (4 ranks of 1,600-1,750 launches: a
//! capture took 53-70 ms, an eager 4-row chunk 60-62 ms with the device
//! idle after 2 ms, PERF-B14b's g2-290), and stays captured: a pinned
//! program is no part of the newest run, so the store never releases it on
//! its own, and its capture first releases the newest run, so no other
//! program is ever left under one. At most 64 programs are pinned.
//!
//! Create may capture all 64 first (`LAST_CHUNKS=1`,
//! `Instance::capture_last_chunks`), each lowered from a one-item chunk of
//! fresh rows, before any decode or drafter graph exists; off by default.
//! Holding all 64 (768 executables, about 380 MB per rank) lengthened every
//! AR decode step's host sampling by 25-130 us and its device wait by up to
//! 200 us on the GPU, and the same binary without them matched main
//! (PERF-B21 t4: g2-296u2, g2-296u3), while capturing at the first run
//! costs about one eager run once per length.
//!
//! The host releases resources only in reverse acquisition order, the
//! instance's buffers and every graph alike (`runtime-host`'s
//! `ReleaseOrderV1`), so no captured program can be released while a
//! newer graph lives: not an older prefill program under newer ones, and
//! not one under a decode or drafter graph captured after it. Every
//! program is therefore captured into a cache sharing the target's graph
//! ledger (`GraphCache::sharing`), which owns each graph in capture order
//! with the drafter's caches and releases them newest first, keeping any
//! graph whose release fails for the next attempt.
//!
//! The programs above every other graph (the newest run) release on their
//! own, without the decode graphs: holding `CAPTURED` of them, or `HELD`
//! programs in all (pinned ones aside), the next capture first releases
//! the newest run (`Room::Newest`). The decode, verify and drafter buckets
//! captured at warmup lie under every program captured after it, so they
//! stay captured. Only `HELD` programs with none above the newest decode,
//! drafter or pinned graph make the next capture release every graph of
//! the instance (`Room::Every`, `Instance::invalidate_graphs`); decode
//! buckets then capture again at their next step, a request's first
//! decode and verify steps paying 75-210 ms of captures on the GPU.

use std::collections::{BTreeMap, VecDeque};
use std::ops::Range;

use bundle_runtime::dispatch::DispatchError;
use bundle_runtime::graphs::{GraphCache, GraphKey};
use bundle_runtime::pipeline::{Collectives, PreparedLaunch, Unit};
use bundle_runtime::upload::InputLayout;
use model_glm5next_contract::contract::{Glm, PREFILL_GRANULE};
use runtime_bundle_abi::{
    BundleAbiError, BundleCudaSymbolHandleV1, BundleStatusV1, BundleStreamHandleV1, HostServicesV1,
};

use crate::program::DraftPass;
use crate::submit::launch_all;

/// Captured programs held at once in the newest run, above every decode
/// and drafter graph.
pub(super) const CAPTURED: usize = 8;
/// Captured programs held at once in all, those under a decode or drafter
/// graph included.
pub(super) const HELD: usize = 24;
/// Keys that ran once remembered at once.
const SEEN: usize = 32;

/// The internal diagnostic switch of the last chunks' programs captured at
/// create (`Instance::capture_last_chunks`): off unless `1`. Not a serving
/// tunable (no model option): served bits are the same either way, and
/// `1` only moves each length's capture from its first run to create.
pub const LAST_CHUNKS: &str = "CHOPINFER_GLM5NEXT_LAST_CHUNK_GRAPHS";

/// Whether create captures the programs of a request's last prompt chunk
/// at every length (`LAST_CHUNKS`); a test's instances do not, unless it
/// creates them within [`last_chunks_in_tests`].
pub fn last_chunks_from_env() -> bool {
    #[cfg(test)]
    return LAST_CHUNKS_IN_TESTS.get();
    #[cfg(not(test))]
    return std::env::var(LAST_CHUNKS).ok().as_deref() == Some("1");
}

#[cfg(test)]
thread_local! {
    static LAST_CHUNKS_IN_TESTS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs `create` with the last chunks' programs captured at create, as
/// `LAST_CHUNKS=1` does.
#[cfg(test)]
pub(crate) fn last_chunks_in_tests<T>(create: impl FnOnce() -> T) -> T {
    LAST_CHUNKS_IN_TESTS.set(true);
    let created = create();
    LAST_CHUNKS_IN_TESTS.set(false);
    created
}

/// The graph key each rank's program takes in its own cache, which holds
/// only that program.
fn graph_key(rank: usize) -> GraphKey {
    GraphKey {
        rank: rank as u32,
        bucket: 1,
        items: 0,
    }
}

/// What an eager prefill's rank programs depend on besides the instance:
/// the drafter pass (none for the target), the rows, the rows the head
/// scores, the input arena's layout (every input span's offset) and each
/// indexer piece's first row, rows and first item (whose block-table row
/// the piece binds). A prefill that injects encoder rows has no key.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct Key {
    pub draft: Option<DraftPass>,
    pub rows: u32,
    pub head_rows: u32,
    pub layout: InputLayout,
    pub pieces: Vec<(u32, u32, u64)>,
}

impl Key {
    /// Whether this is a request's last prompt chunk, which captures at its
    /// first run and stays captured: the target's rows of one piece, at
    /// most a granule of them, scoring one row.
    pub fn pinned(&self) -> bool {
        self.draft.is_none()
            && self.pieces.is_empty()
            && self.head_rows == 1
            && self.rows <= PREFILL_GRANULE
    }
}

/// A captured program: its key, its graphs in a cache sharing the
/// target's ledger (empty once released), their ledger positions and
/// whether it is pinned (`PrefillGraphs::pins`).
struct Program {
    key: Key,
    cache: GraphCache<Glm>,
    span: Range<usize>,
    pinned: bool,
}

/// What the next capture must release first (`PrefillGraphs::room`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Room {
    /// Nothing.
    Free,
    /// The newest run of programs (`PrefillGraphs::release_newest`).
    Newest,
    /// Every graph of the instance.
    Every,
}

#[derive(Default)]
pub(super) struct PrefillGraphs {
    /// Captured programs in capture order; a cache the ledger released
    /// since holds nothing.
    captured: Vec<Program>,
    /// Keys that ran once eagerly, oldest first.
    seen: VecDeque<Key>,
    /// Whether a release of every graph is owed: a release of the newest
    /// run failed, so graphs of programs the store forgot may still be in
    /// the ledger, and only a release of every graph frees them.
    owed: bool,
    /// Whether every key captures at its second run, a request's last
    /// chunk included (tests of the eager path).
    eager_last_chunks: bool,
}

impl PrefillGraphs {
    /// Whether `key`'s program is pinned: a request's last prompt chunk
    /// (`Key::pinned`), unless every key captures at its second run.
    pub fn pins(&self, key: &Key) -> bool {
        !self.eager_last_chunks && key.pinned()
    }

    /// Makes every key capture at its second run, a request's last chunk
    /// included, so that a first run stays eager.
    #[cfg(test)]
    pub fn keep_last_chunks_eager(&mut self) {
        self.eager_last_chunks = true;
    }

    /// Whether `key`'s programs are captured and not released since.
    pub fn contains(&self, key: &Key) -> bool {
        self.captured
            .iter()
            .any(|program| &program.key == key && !program.cache.is_empty())
    }

    /// How many live programs end the ledger of `owner` (whose ledger every
    /// program shares), newest first, with no other graph between or above
    /// them and none of them pinned.
    fn newest_run(&self, owner: &GraphCache<Glm>) -> usize {
        let mut end = owner.ledger_len();
        let mut run = 0;
        for program in self.captured.iter().rev() {
            if program.cache.is_empty() {
                continue;
            }
            if program.span.end != end || program.pinned {
                break;
            }
            end = program.span.start;
            run += 1;
        }
        run
    }

    /// What the capture of `key` must release first: every graph while a
    /// failed release of the newest run owes it (`release_newest`). Else
    /// for a pinned key, the newest run when it holds any, so that no
    /// program is left under the pinned one; for another key, the newest
    /// run once it holds `CAPTURED` programs, or once `HELD` programs that
    /// are not pinned live in all and the run holds any; every graph once
    /// `HELD` such programs live and none lies above the newest decode,
    /// drafter or pinned graph.
    pub fn room(&self, owner: &GraphCache<Glm>, key: &Key) -> Room {
        if self.owed {
            return Room::Every;
        }
        let run = self.newest_run(owner);
        if self.pins(key) {
            return if run > 0 { Room::Newest } else { Room::Free };
        }
        let live = self
            .captured
            .iter()
            .filter(|program| !program.cache.is_empty() && !program.pinned)
            .count();
        if run >= CAPTURED || (live >= HELD && run > 0) {
            Room::Newest
        } else if live >= HELD {
            Room::Every
        } else {
            Room::Free
        }
    }

    /// Releases the newest run of programs (`room`), newest first, leaving
    /// every older graph and every pinned program captured, and returns how
    /// many it released. A failed release leaves that graph and every
    /// older one owned by the ledger, its program forgotten, and owes a
    /// release of every graph: until one succeeds
    /// (`Instance::release_graphs`), every capture first releases every
    /// graph (`room`), so the ledger never holds more than the store's
    /// bounds and the graphs of one failed run.
    pub fn release_newest(&mut self, services: &HostServicesV1) -> Result<usize, BundleAbiError> {
        let mut released = 0;
        let mut failed = None;
        for program in self.captured.iter_mut().rev() {
            if program.cache.is_empty() {
                continue;
            }
            if program.pinned {
                break;
            }
            match program.cache.release_if_newest(services) {
                Ok(true) => released += 1,
                Ok(false) => break,
                Err(error) => {
                    failed = Some(error);
                    break;
                }
            }
        }
        self.captured.retain(|program| !program.cache.is_empty());
        match failed {
            Some(error) => {
                self.owed = true;
                Err(error)
            }
            None => Ok(released),
        }
    }

    /// Whether `key` ran before, so this run captures it; otherwise
    /// remembers it, and the run stays eager.
    pub fn repeats(&mut self, key: &Key) -> bool {
        if let Some(index) = self.seen.iter().position(|seen| seen == key) {
            self.seen.remove(index);
            return true;
        }
        if self.seen.len() == SEEN {
            self.seen.pop_front();
        }
        self.seen.push_back(key.clone());
        false
    }

    /// Captures every rank's program under `key`, in the pieces decode's
    /// graphs use, into a cache sharing `owner`'s ledger. Nothing runs. A
    /// failure publishes nothing; the graphs captured before it stay owned
    /// by the ledger (its own rank's pieces already released newest first
    /// by `GraphCache::capture`), so a release of every graph frees them.
    #[allow(clippy::too_many_arguments)]
    pub fn capture(
        &mut self,
        services: &HostServicesV1,
        streams: &[BundleStreamHandleV1],
        symbols: &[BTreeMap<String, BundleCudaSymbolHandleV1>],
        owner: &GraphCache<Glm>,
        key: Key,
        programs: &[Vec<PreparedLaunch>],
    ) -> Result<(), DispatchError> {
        if programs.len() != streams.len()
            || symbols.len() != streams.len()
            || programs.iter().any(Vec::is_empty)
        {
            return Err(DispatchError::InvalidPlan);
        }
        self.captured.retain(|program| !program.cache.is_empty());
        let start = owner.ledger_len();
        let mut cache = GraphCache::sharing(owner);
        for (rank, program) in programs.iter().enumerate() {
            let stream = streams[rank];
            let symbols = &symbols[rank];
            cache.capture(services, stream, graph_key(rank), program.len(), |range| {
                launch_all(services, stream, symbols, &program[range])
            })?;
        }
        let span = start..cache.ledger_len();
        let pinned = self.pins(&key);
        self.captured.push(Program {
            key,
            cache,
            span,
            pinned,
        });
        Ok(())
    }

    /// Replays `key`'s captured programs on every rank as one unit,
    /// piece-major as decode's graphs replay: each piece on every rank
    /// before the next piece. `submitted(rank)` runs once a rank's first
    /// piece is queued. A failure once a rank's piece was queued poisons
    /// `collectives`.
    pub fn replay(
        &self,
        services: &HostServicesV1,
        streams: &[BundleStreamHandleV1],
        collectives: &Collectives,
        key: &Key,
        submitted: impl FnMut(usize),
    ) -> Result<(), DispatchError> {
        if collectives.is_poisoned() {
            return Err(BundleAbiError::Status(BundleStatusV1::InvalidState).into());
        }
        let cache = self
            .captured
            .iter()
            .find(|program| &program.key == key && !program.cache.is_empty())
            .map(|program| &program.cache)
            .ok_or(DispatchError::InvalidPlan)?;
        let mut unit = Unit::default();
        let replayed = unit.record(
            collectives,
            cache
                .replay_piece_major(services, streams, (1, 0), submitted)
                .map_err(DispatchError::from),
        )?;
        if replayed {
            Ok(())
        } else {
            Err(DispatchError::InvalidPlan)
        }
    }

    /// Forgets every program and every key that ran once, and any owed
    /// release, as the ledger releases every graph.
    pub fn forget(&mut self) {
        self.captured.clear();
        self.seen.clear();
        self.owed = false;
    }

    /// Owes a release of every graph: one failed, so the ledger still
    /// holds graphs no program names.
    pub fn owe_release(&mut self) {
        self.owed = true;
    }

    /// Whether a release of every graph is owed (`owe_release`,
    /// `release_newest`): no graph of the instance may capture until it is
    /// paid (`Instance::launch`).
    pub fn owes(&self) -> bool {
        self.owed
    }
}
