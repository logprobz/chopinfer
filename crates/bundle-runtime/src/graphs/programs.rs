//! Captured segments of eager programs, keyed by exactly what they launch.
//!
//! An eager program whose launches repeat from step to step, such as a short
//! prefill's, costs the host one launch call per kernel on every rank. Its
//! segments can be captured once and replayed as one graph each instead. The
//! model splits each rank's program into segments at the points where host
//! work must run between them: a mapped gather's join, or an eager section
//! whose launches change with something the key does not hold. It keys the
//! program by every segment's [`fingerprint`] on every rank (each launch's
//! symbol, rows, plan and resolved arguments, in order), together with
//! whatever names the program for its telemetry. Equal keys therefore
//! capture equal launches, so a replay issues exactly what the eager run of
//! the same step would; a launch that changes from step to step only makes
//! a new key.
//!
//! Admission (`Programs::admit`) is repeat-aware: a key captures on its
//! second run, since a capture costs about an eager run and a key that never
//! repeats would only pay for it. The store remembers the most recent keys
//! that ran once. A capture reserves every segment of every rank before the
//! first begins, in the ledger the store shares with the instance's other
//! graphs (`GraphCache::reserve_retaining`), and keeps a headroom of free
//! handles in a bounded ledger for the decode, verify and drafter graphs
//! captured lazily later. A key that finds no room runs eagerly.
//!
//! The host releases graphs only newest first, the instance's decode and
//! drafter graphs included. The store holds at most `Limits::programs`
//! programs; a capture beyond that first releases the newest program, when
//! no other graph was captured above it. A program under a newer decode or
//! drafter graph stays captured and keeps replaying. The store never
//! releases another cache's graph, so warmed decode graphs survive every
//! admission. A failed capture or release leaves its graphs to the ledger
//! (an instance-wide release frees them) and stops further captures.

use std::collections::VecDeque;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::ops::Range;

use runtime_bundle_abi::{BundleAbiError, BundleStreamHandleV1, HostServicesV1, KernelArgumentV1};

use super::{GraphCache, GraphKey, GraphReservation};
use crate::dispatch::{self, DispatchError};
use crate::geometry::BundleGeometry;
use crate::pipeline::{Collectives, PreparedLaunch, Unit};

/// A digest of a launch list: every launch's symbol, rows, plan and
/// resolved arguments, in order. Lists that resolve to the same launches
/// have the same fingerprint, whatever their planner's binding IDs.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Fingerprint {
    pub launches: u32,
    pub digest: [u64; 2],
}

/// The fingerprint of `launches`; a launch the host would refuse to
/// resolve has none.
pub fn fingerprint(launches: &[PreparedLaunch]) -> Result<Fingerprint, DispatchError> {
    // Two digests of the same words with distinct leading tags: a key holds
    // 128 bits, so distinct programs do not collide in practice.
    let mut digests = [DefaultHasher::new(), DefaultHasher::new()];
    for (tag, digest) in digests.iter_mut().enumerate() {
        tag.hash(digest);
    }
    for launch in launches {
        let config = dispatch::launch_config(launch.plan)?;
        let arguments = dispatch::arguments(&launch.arguments, &launch.bindings)?;
        for digest in &mut digests {
            launch.symbol.hash(digest);
            launch.rows.hash(digest);
            launch.plan.workspace_bytes.hash(digest);
            (
                config.flags,
                [config.grid_x, config.grid_y, config.grid_z],
                [config.block_x, config.block_y, config.block_z],
                config.shared_memory_bytes,
                [config.cluster.x, config.cluster.y, config.cluster.z],
            )
                .hash(digest);
            arguments.len().hash(digest);
            for argument in &arguments {
                hash_argument(argument, digest);
            }
        }
    }
    Ok(Fingerprint {
        launches: u32::try_from(launches.len()).map_err(|_| DispatchError::InvalidPlan)?,
        digest: digests.map(|digest| digest.finish()),
    })
}

fn hash_span(span: &runtime_bundle_abi::DeviceBufferSpanV1, digest: &mut DefaultHasher) {
    (span.buffer.raw, span.offset, span.length).hash(digest);
}

fn hash_argument(argument: &KernelArgumentV1, digest: &mut DefaultHasher) {
    match argument {
        KernelArgumentV1::DeviceBuffer(span) => {
            0_u8.hash(digest);
            hash_span(span, digest);
        }
        KernelArgumentV1::NullDevicePointer => 1_u8.hash(digest),
        KernelArgumentV1::ScalarU32(value) => (2_u8, value).hash(digest),
        KernelArgumentV1::ScalarI32(value) => (3_u8, value).hash(digest),
        KernelArgumentV1::ScalarU64(value) => (4_u8, value).hash(digest),
        KernelArgumentV1::OpaqueBytes(bytes) => (5_u8, bytes).hash(digest),
        // A tensor map's fields are plain values; its debug form names
        // every one of them.
        KernelArgumentV1::TensorMap(map) => (6_u8, format!("{map:?}")).hash(digest),
        KernelArgumentV1::DevicePointerArray(spans) => {
            (7_u8, spans.len()).hash(digest);
            for span in spans {
                match span {
                    Some(span) => {
                        1_u8.hash(digest);
                        hash_span(span, digest);
                    }
                    None => 0_u8.hash(digest),
                }
            }
        }
    }
}

/// The store's bounds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Limits {
    /// Programs captured at once.
    pub programs: usize,
    /// Keys that ran once remembered at once.
    pub seen: usize,
    /// Free handles per rank a bounded ledger keeps after a capture.
    pub headroom: usize,
}

/// Why a program runs eagerly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Eager {
    /// Its key ran no earlier run the store remembers.
    FirstRun,
    /// The store holds its limit and its newest program lies under another
    /// graph.
    Full,
    /// The shared ledger lacks the handles and headroom.
    Ledger,
    /// A failed capture or release stopped captures.
    Stopped,
}

/// What a run of a key does.
#[derive(Debug)]
pub enum Admission {
    /// Replay its captured segments (`Programs::replay`).
    Replay,
    /// Capture its segments under this reservation (`Programs::capture`),
    /// then replay them.
    Capture(GraphReservation),
    /// Run it eagerly.
    Eager(Eager),
}

/// Counts of the store's decisions, for telemetry.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ProgramStats {
    pub programs: usize,
    pub replays: u64,
    pub captures: u64,
    pub first_runs: u64,
    pub full: u64,
    pub ledger: u64,
    pub released: u64,
    pub failures: u64,
}

/// A captured program: its key, one cache per segment (none for a segment
/// with no launches), and its graphs' positions in the ledger.
struct Captured<K, G> {
    key: K,
    segments: Vec<Option<GraphCache<G>>>,
    span: Range<usize>,
}

impl<K, G> Captured<K, G> {
    /// Whether every segment's graphs are live: none was released since.
    fn live(&self) -> bool {
        self.segments
            .iter()
            .flatten()
            .all(|cache| !cache.is_empty())
    }
}

/// The graph key a segment's rank takes in the segment's own cache, which
/// holds only that segment.
fn segment_key(rank: usize) -> GraphKey {
    GraphKey {
        rank: rank as u32,
        bucket: 1,
        items: 0,
    }
}

pub struct Programs<K, G> {
    limits: Limits,
    /// Captured programs, oldest first.
    captured: Vec<Captured<K, G>>,
    /// Keys that ran once eagerly, oldest first.
    seen: VecDeque<K>,
    stopped: bool,
    stats: ProgramStats,
}

impl<K: Clone + Eq, G: BundleGeometry> Programs<K, G> {
    pub fn new(limits: Limits) -> Self {
        Self {
            limits,
            captured: Vec::new(),
            seen: VecDeque::new(),
            stopped: false,
            stats: ProgramStats::default(),
        }
    }

    pub fn limits(&self) -> Limits {
        self.limits
    }

    pub fn stats(&self) -> ProgramStats {
        ProgramStats {
            programs: self.captured.iter().filter(|p| p.live()).count(),
            ..self.stats
        }
    }

    /// Whether `key`'s segments are captured and none was released since.
    pub fn contains(&self, key: &K) -> bool {
        self.captured.iter().any(|p| &p.key == key && p.live())
    }

    /// Forgets programs an instance-wide release took.
    fn prune(&mut self) {
        self.captured.retain(Captured::live);
    }

    /// Whether `key` ran before, which makes this run capture it; otherwise
    /// remembers it.
    fn repeats(&mut self, key: &K) -> bool {
        if let Some(at) = self.seen.iter().position(|seen| seen == key) {
            self.seen.remove(at);
            return true;
        }
        self.remember(key);
        false
    }

    fn remember(&mut self, key: &K) {
        if self.limits.seen == 0 {
            return;
        }
        if self.seen.len() == self.limits.seen {
            self.seen.pop_front();
        }
        self.seen.push_back(key.clone());
    }

    /// Decides a run of `key`, whose programs have `segments` nonempty
    /// segments per rank. A capture past the program limit first releases
    /// the newest program, when it lies above every other graph of
    /// `owner`'s ledger, after `drain` established that every rank's queued
    /// work is done. A failed drain or release fails the run.
    pub fn admit(
        &mut self,
        services: &HostServicesV1,
        owner: &GraphCache<G>,
        key: &K,
        segments: usize,
        drain: impl FnOnce() -> Result<(), DispatchError>,
    ) -> Result<Admission, DispatchError> {
        self.prune();
        if self.contains(key) {
            self.stats.replays += 1;
            return Ok(Admission::Replay);
        }
        if self.stopped || self.limits.programs == 0 {
            return Ok(Admission::Eager(Eager::Stopped));
        }
        if !self.repeats(key) {
            self.stats.first_runs += 1;
            return Ok(Admission::Eager(Eager::FirstRun));
        }
        if self.captured.len() >= self.limits.programs {
            let newest = self.captured.last().map(|p| p.span.end);
            if newest != Some(owner.ledger_len()) {
                self.stats.full += 1;
                self.remember(key);
                return Ok(Admission::Eager(Eager::Full));
            }
            drain()?;
            self.release_newest(services, self.limits.programs - 1)?;
        }
        let ranks = owner.ranks().count();
        let handles = segments * ranks;
        let stats = owner.stats();
        let room = stats.capacity.is_none_or(|capacity| {
            capacity
                .checked_sub(stats.live + stats.reserved)
                .is_some_and(|free| free >= handles + self.limits.headroom * ranks)
        });
        match room.then(|| owner.reserve_retaining(handles)).flatten() {
            Some(reservation) => Ok(Admission::Capture(reservation)),
            None => {
                self.stats.ledger += 1;
                self.remember(key);
                Ok(Admission::Eager(Eager::Ledger))
            }
        }
    }

    /// Releases the newest programs, newest first, while more than `keep`
    /// are captured and the newest lies above every other graph. Returns how
    /// many it released. A failed release leaves that graph and every older
    /// one to the ledger and stops captures.
    pub fn release_newest(
        &mut self,
        services: &HostServicesV1,
        keep: usize,
    ) -> Result<usize, BundleAbiError> {
        self.prune();
        let mut released = 0;
        while self.captured.len() > keep {
            let Some(program) = self.captured.last_mut() else {
                break;
            };
            match release(services, program) {
                Ok(true) => {
                    self.captured.pop();
                    released += 1;
                    self.stats.released += 1;
                }
                Ok(false) => break,
                Err(error) => {
                    self.captured.pop();
                    self.stopped = true;
                    self.stats.failures += 1;
                    return Err(error);
                }
            }
        }
        Ok(released)
    }

    /// Captures every rank's segments under `key` into caches sharing
    /// `owner`'s ledger, launching nothing: `segments[s][rank]` is rank
    /// `rank`'s launches of segment `s`, and `issue(rank, launches)`
    /// enqueues them on the rank's stream while it captures. A segment
    /// empty on every rank takes no graph; one empty on some ranks only is
    /// refused before anything is captured. `label` names the programs in
    /// the ledger's telemetry. A failure releases what this capture took,
    /// newest first, and stops captures.
    #[allow(clippy::too_many_arguments)]
    pub fn capture(
        &mut self,
        services: &HostServicesV1,
        owner: &GraphCache<G>,
        key: K,
        reservation: &mut GraphReservation,
        streams: &[BundleStreamHandleV1],
        segments: &[Vec<&[PreparedLaunch]>],
        label: (&'static str, Option<u32>),
        mut issue: impl FnMut(usize, &[PreparedLaunch]) -> Result<(), DispatchError>,
    ) -> Result<(), DispatchError> {
        let ranks = owner.ranks().count();
        if streams.len() != ranks
            || segments.iter().any(|segment| {
                segment.len() != ranks
                    || (segment.iter().any(|l| l.is_empty())
                        && segment.iter().any(|l| !l.is_empty()))
            })
        {
            return Err(DispatchError::InvalidPlan);
        }
        if self.stopped {
            return Err(
                BundleAbiError::Status(runtime_bundle_abi::BundleStatusV1::InvalidState).into(),
            );
        }
        let start = owner.ledger_len();
        let mut program = Captured {
            key,
            segments: Vec::with_capacity(segments.len()),
            span: start..start,
        };
        let captured = (|| {
            for (index, segment) in segments.iter().enumerate() {
                if segment[0].is_empty() {
                    program.segments.push(None);
                    continue;
                }
                let mut cache = GraphCache::sharing(owner);
                cache.label(label.0, label.1, index as u32);
                let captured = (0..ranks).try_for_each(|rank| {
                    cache.capture_reserved(
                        services,
                        streams[rank],
                        segment_key(rank),
                        1,
                        reservation,
                        |_| issue(rank, segment[rank]),
                    )
                });
                program.segments.push(Some(cache));
                captured?;
            }
            Ok(())
        })();
        if let Err(error) = captured {
            self.stats.failures += 1;
            self.stopped = true;
            let _ = release(services, &mut program);
            return Err(error);
        }
        program.span = start..owner.ledger_len();
        self.stats.captures += 1;
        self.captured.push(program);
        Ok(())
    }

    /// Replays `key`'s segments on every rank, each piece-major as decode
    /// graphs replay, calling `gap(s)` between segments `s` and `s + 1` for
    /// the host work there. Launches nothing, returning false, when `key`
    /// is not captured. A failure once a segment's first rank was queued
    /// poisons `collectives`.
    pub fn replay(
        &self,
        services: &HostServicesV1,
        key: &K,
        streams: &[BundleStreamHandleV1],
        collectives: &Collectives,
        mut gap: impl FnMut(usize) -> Result<(), DispatchError>,
    ) -> Result<bool, DispatchError> {
        let Some(program) = self.captured.iter().find(|p| &p.key == key && p.live()) else {
            return Ok(false);
        };
        if collectives.is_poisoned() {
            return Err(
                BundleAbiError::Status(runtime_bundle_abi::BundleStatusV1::InvalidState).into(),
            );
        }
        for (index, segment) in program.segments.iter().enumerate() {
            if index > 0 {
                gap(index - 1)?;
            }
            let Some(cache) = segment else {
                continue;
            };
            let replayed = Unit::default().record(
                collectives,
                cache
                    .replay_piece_major(services, streams, (1, 0), |_| {})
                    .map_err(DispatchError::from),
            )?;
            if !replayed {
                return Err(DispatchError::InvalidPlan);
            }
        }
        Ok(true)
    }
}

/// Releases `program`'s graphs newest first when they are the ledger's
/// newest. Returns false, releasing nothing, while a newer graph lives. A
/// capture that failed has an empty span: whatever it took is the newest.
fn release<K, G>(
    services: &HostServicesV1,
    program: &mut Captured<K, G>,
) -> Result<bool, BundleAbiError> {
    let on_top = program.span.is_empty()
        || program
            .segments
            .iter()
            .flatten()
            .next()
            .is_none_or(|cache| cache.ledger_len() == program.span.end);
    if !on_top {
        return Ok(false);
    }
    for cache in program.segments.iter_mut().rev().flatten() {
        if !cache.release_if_newest(services)? {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::fingerprint;
    use crate::dispatch::BufferBindings;
    use crate::pipeline::PreparedLaunch;
    use kernel_core::contract::{Argument, ArgumentList, LaunchPlan};
    use runtime_bundle_abi::{BundleDeviceBufferHandleV1, DeviceBufferSpanV1};

    /// One launch of `symbol` reading `offset` of a buffer and `scalar`,
    /// bound after `unused` other views, so planner IDs differ.
    fn launch(symbol: &str, offset: u64, scalar: u32, unused: u64) -> PreparedLaunch {
        let mut bindings = BufferBindings::default();
        let buffer = BundleDeviceBufferHandleV1::from_parts(1, 1);
        for at in 0..unused {
            bindings.bind(DeviceBufferSpanV1::new(buffer, at * 512, 256));
        }
        let view = bindings.bind(DeviceBufferSpanV1::new(buffer, offset, 256));
        PreparedLaunch {
            symbol: symbol.into(),
            rows: 1,
            plan: LaunchPlan {
                grid: [1, 1, 1],
                block: [32, 1, 1],
                dynamic_shared_bytes: 0,
                workspace_bytes: 0,
                cluster: [1, 1, 1],
                pdl: true,
            },
            arguments: ArgumentList(vec![Argument::DeviceBuffer(view), Argument::U32(scalar)]),
            bindings,
        }
    }

    #[test]
    fn fingerprints_follow_what_the_host_resolves() {
        let print = |launches: &[PreparedLaunch]| fingerprint(launches).unwrap();
        let base = print(&[launch("k", 256, 7, 0), launch("j", 0, 1, 0)]);
        assert_eq!(base.launches, 2);
        // Other planner IDs of the same views resolve to the same launches.
        assert_eq!(base, print(&[launch("k", 256, 7, 3), launch("j", 0, 1, 2)]));
        for other in [
            vec![launch("j", 0, 1, 0), launch("k", 256, 7, 0)],
            vec![launch("k", 256, 7, 0)],
            vec![launch("x", 256, 7, 0), launch("j", 0, 1, 0)],
            vec![launch("k", 768, 7, 0), launch("j", 0, 1, 0)],
            vec![launch("k", 256, 8, 0), launch("j", 0, 1, 0)],
        ] {
            assert_ne!(base, print(&other));
        }
        let mut wide = launch("k", 256, 7, 0);
        wide.plan.grid[0] = 2;
        assert_ne!(base, print(&[wide, launch("j", 0, 1, 0)]));
        let mut rows = launch("k", 256, 7, 0);
        rows.rows = 2;
        assert_ne!(base, print(&[rows, launch("j", 0, 1, 0)]));
        assert_eq!(print(&[]).launches, 0);
    }
}
