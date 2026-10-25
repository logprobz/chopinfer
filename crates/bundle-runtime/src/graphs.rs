//! Per-rank decode and verify graph capture and replay over fixed device buffers.
//!
//! Each rank's program is captured as a few graphs, run back to back
//! on its stream. Launching one graph costs host time that grows with its
//! kernel count, and every rank's first collective waits for the last rank to
//! start. A short first piece lets all ranks start after a few small
//! launches; the host enqueues the longer pieces while the GPUs run the first.

use std::cell::RefCell;
use std::collections::BTreeMap;

use std::marker::PhantomData;
use std::ops::Range;
use std::rc::Rc;

use runtime_bundle_abi::{
    BundleAbiError, BundleGraphHandleV1, BundleStreamHandleV1, HostServicesV1,
};

mod budget;
pub use budget::{GraphReservation, GraphStats};

use crate::geometry::{BundleGeometry, Items, Ranks};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct GraphKey {
    pub rank: u32,
    /// The token bucket.
    pub bucket: u32,
    /// Zero for a decode graph; the item bucket of a verify or catch-up
    /// graph, whose items carry several rows each.
    pub items: u32,
}

/// Graph buckets are the powers of two up to `limit`.
const fn is_bucket(value: u32, limit: u32) -> bool {
    value.is_power_of_two() && value <= limit
}

/// The graph bucket that holds `live` items or tokens at `G`'s package
/// maximum. An instance serving fewer items uses [`Items::bucket`].
pub fn decode_bucket<G: BundleGeometry>(live: u32) -> Option<u32> {
    Items::package::<G>().bucket(live)
}

/// The token bucket of a verify step with `live` decode-phase rows, at `G`'s
/// package maximum.
pub fn verify_bucket<G: BundleGeometry>(live: u32) -> Option<u32> {
    verify_bucket_within::<G>(live, Items::package::<G>())
}

/// Most decode-phase rows in one verify step of an instance serving `items`:
/// `G`'s verify tokens, and never fewer than the largest decode bucket.
pub fn verify_tokens<G: BundleGeometry>(items: Items) -> u32 {
    if let Some(narrow) = G::narrow_verify_tokens()
        && items.capacity() <= narrow
    {
        return narrow;
    }
    if G::VERIFY_TOKENS > items.capacity() {
        G::VERIFY_TOKENS
    } else {
        items.capacity()
    }
}

/// The token bucket of a verify step with `live` decode-phase rows, for an
/// instance serving `items`.
pub fn verify_bucket_within<G: BundleGeometry>(live: u32, items: Items) -> Option<u32> {
    if live == 0 || live > verify_tokens::<G>(items) {
        None
    } else if G::EXACT_VERIFY_GRAPHS {
        Some(live)
    } else {
        Some(live.next_power_of_two())
    }
}

/// Launch ranges of a program's graph pieces: about the first 1/32 of the
/// launches (the ring backup, embedding and first layer), up to 1/6, then the
/// rest. Empty ranges are dropped, so a short program is one graph.
pub fn pieces(launches: usize) -> Vec<Range<usize>> {
    let bounds = [0, launches / 32, launches / 6, launches];
    bounds
        .windows(2)
        .map(|pair| pair[0]..pair[1])
        .filter(|range| !range.is_empty())
        .collect()
}

/// Every graph the caches sharing it captured, oldest first.
///
/// The host releases resources newest first, whichever cache captured them,
/// so graphs release from here rather than in key order. A graph whose
/// release failed stays here for a later release.
#[derive(Debug, Default)]
struct Ledger {
    graphs: Vec<BundleGraphHandleV1>,
    routes: BTreeMap<(u32, u32), crate::routes::RouteCounts>,
    budget: budget::Budget,
    /// Counts releases of the whole ledger. A cache whose lookups predate the
    /// last one holds released graphs and treats itself as empty.
    epoch: u64,
}

impl Ledger {
    /// Releases graphs newest first until `keep` remain.
    fn release_to(&mut self, services: &HostServicesV1, keep: usize) -> Result<(), BundleAbiError> {
        while self.graphs.len() > keep {
            let graph = self.graphs[self.graphs.len() - 1];
            if let Err(error) = services.release_graph(graph) {
                self.budget.release_failures += 1;
                return Err(error);
            }
            self.graphs.pop();
            self.routes.remove(&(graph.generation(), graph.slot()));
            self.budget.released += 1;
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum GraphKind {
    Verify,
    CatchUp,
}

pub struct GraphCache<G> {
    kind: GraphKind,
    label: budget::Label,
    graphs: BTreeMap<GraphKey, Vec<BundleGraphHandleV1>>,
    ledger: Rc<RefCell<Ledger>>,
    /// The ledger epoch `graphs` belongs to.
    epoch: u64,
    /// The instance's ranks: graph keys name one of them, and a replay takes
    /// one stream per rank.
    ranks: Ranks,
    /// The instance's items: decode buckets and verify item buckets stop at
    /// their capacity.
    items: Items,
    geometry: PhantomData<G>,
}

impl<G: BundleGeometry> GraphCache<G> {
    /// An empty cache for an instance on `ranks` ranks serving `G`'s package
    /// maximum of items, with its own ledger.
    pub fn new(ranks: Ranks) -> Self {
        Self::with_items(ranks, Items::package::<G>())
    }

    /// An empty cache for an instance on `ranks` ranks serving `items`, with
    /// its own ledger.
    pub fn with_items(ranks: Ranks, items: Items) -> Self {
        Self {
            graphs: BTreeMap::new(),
            ledger: Rc::default(),
            epoch: 0,
            ranks,
            kind: GraphKind::Verify,
            label: budget::Label::default(),
            items,
            geometry: PhantomData,
        }
    }

    /// A cache for drafter catch-up runs, including runs that also draft.
    /// The whole token bucket and each item's rows are bounded by the decode
    /// batch. Keep these graphs separate from target verify graphs, whose
    /// items are bounded by `MAX_VERIFY_ROWS`.
    pub fn new_catch_up(ranks: Ranks) -> Self {
        Self {
            kind: GraphKind::CatchUp,
            ..Self::new(ranks)
        }
    }

    /// A catch-up cache sharing the target's release order, with the drafter's
    /// fixed capacity rather than the target's served item count.
    pub fn catch_up_sharing(owner: &Self) -> Self {
        Self {
            ledger: Rc::clone(&owner.ledger),
            epoch: owner.ledger.borrow().epoch,
            ..Self::new_catch_up(owner.ranks)
        }
    }
}

impl<G> GraphCache<G> {
    /// The ranks this cache captures and replays for.
    pub fn ranks(&self) -> Ranks {
        self.ranks
    }

    /// The items this cache captures buckets for.
    pub fn items(&self) -> Items {
        self.items
    }

    /// An empty cache on `owner`'s ranks whose graphs share its ledger. An
    /// instance's caches acquire graphs in one host release order, so each
    /// cache beside the first shares its ledger, and releasing any of them
    /// releases all.
    pub fn sharing(owner: &Self) -> Self {
        let ledger = Rc::clone(&owner.ledger);
        let epoch = ledger.borrow().epoch;
        Self {
            graphs: BTreeMap::new(),
            ledger,
            epoch,
            ranks: owner.ranks,
            kind: owner.kind,
            label: owner.label,
            items: owner.items,
            geometry: PhantomData,
        }
    }

    /// This cache's lookups, unless the shared ledger was released since.
    fn live(&self) -> Option<&BTreeMap<GraphKey, Vec<BundleGraphHandleV1>>> {
        (self.epoch == self.ledger.borrow().epoch).then_some(&self.graphs)
    }
}

impl<G: BundleGeometry> GraphCache<G> {
    /// Replays a cached program, or captures `launches` piece by piece and
    /// runs it. `record` enqueues one launch range on `stream`.
    pub fn run<E: From<BundleAbiError>>(
        &mut self,
        services: &HostServicesV1,
        stream: BundleStreamHandleV1,
        key: GraphKey,
        launches: usize,
        record: impl FnMut(Range<usize>) -> Result<(), E>,
    ) -> Result<(), E> {
        self.capture(services, stream, key, launches, record)?;
        for &graph in &self.graphs[&key] {
            services.launch_graph(stream, graph)?;
            self.replay_routes(graph);
        }
        Ok(())
    }

    /// Captures `launches` piece by piece unless `key` is cached, launching
    /// nothing. `record` enqueues one launch range on `stream`.
    pub fn capture<E: From<BundleAbiError>>(
        &mut self,
        services: &HostServicesV1,
        stream: BundleStreamHandleV1,
        key: GraphKey,
        launches: usize,
        record: impl FnMut(Range<usize>) -> Result<(), E>,
    ) -> Result<(), E> {
        self.capture_inner(services, stream, key, launches, None, record)
    }

    /// Consume a caller's reservation for an entire rank group. Cached
    /// keys consume nothing; failed-cleanup handles remain charged.
    #[allow(clippy::too_many_arguments)]
    pub fn capture_reserved<E: From<BundleAbiError>>(
        &mut self,
        services: &HostServicesV1,
        stream: BundleStreamHandleV1,
        key: GraphKey,
        launches: usize,
        reservation: &mut GraphReservation,
        record: impl FnMut(Range<usize>) -> Result<(), E>,
    ) -> Result<(), E> {
        self.capture_inner(services, stream, key, launches, Some(reservation), record)
    }

    #[allow(clippy::too_many_arguments)]
    fn capture_inner<E: From<BundleAbiError>>(
        &mut self,
        services: &HostServicesV1,
        stream: BundleStreamHandleV1,
        key: GraphKey,
        launches: usize,
        reservation: Option<&mut GraphReservation>,
        mut record: impl FnMut(Range<usize>) -> Result<(), E>,
    ) -> Result<(), E> {
        // Decode graphs stop at the largest item bucket. Catch-up may fill
        // that token bucket with a single item's rows. Target verify token
        // buckets stop at the verify rows, over at most the largest item
        // bucket, each item carrying at most `MAX_VERIFY_ROWS` rows.
        let capacity = self.items.capacity();
        let (tokens, rows_per_item) = match self.kind {
            GraphKind::CatchUp => (capacity, capacity),
            GraphKind::Verify => (
                if key.items == 0 {
                    capacity
                } else {
                    verify_tokens::<G>(self.items)
                },
                G::MAX_VERIFY_ROWS,
            ),
        };
        let exact =
            matches!(self.kind, GraphKind::Verify) && key.items != 0 && G::EXACT_VERIFY_GRAPHS;
        let valid_rows = if exact {
            key.bucket > 0 && key.bucket <= tokens
        } else {
            is_bucket(key.bucket, tokens)
        };
        let valid_items = if exact {
            key.items <= self.items.get()
        } else {
            is_bucket(key.items, capacity)
        };
        if !self.ranks.contains(key.rank)
            || !valid_rows
            || (key.items != 0
                && (!valid_items
                    || key.items > key.bucket
                    || key.bucket > key.items * rows_per_item))
        {
            return Err(BundleAbiError::InvalidRange("graph bucket").into());
        }
        let epoch = self.ledger.borrow().epoch;
        if self.epoch != epoch {
            self.graphs.clear();
            self.epoch = epoch;
        }
        if self.graphs.contains_key(&key) {
            return Ok(());
        }
        let needed = pieces(launches).len();
        let mut local;
        let reservation = match reservation {
            Some(reservation) => reservation,
            None => {
                local = self
                    .reserve(needed)
                    .ok_or(BundleAbiError::InvalidRange("graph capacity"))?;
                &mut local
            }
        };
        reservation.validate(self, needed)?;
        let capture_started = std::time::Instant::now();
        // This capture's pieces are the ledger's newest graphs. A failure
        // releases them newest first; any whose release fails stays in the
        // ledger, and a later `release_all` retries it.
        let first = self.ledger.borrow().graphs.len();
        for range in pieces(launches) {
            let captured = services
                .begin_graph_capture(stream)
                .map_err(E::from)
                .and_then(|()| {
                    let routes = crate::routes::RouteCounter::nested();
                    let recorded = record(range);
                    let counts = routes.finish();
                    let ended = services.end_graph_capture(stream);
                    if let Ok(graph) = ended {
                        let mut ledger = self.ledger.borrow_mut();
                        ledger.graphs.push(graph);
                        if let Some(counts) = counts {
                            ledger
                                .routes
                                .insert((graph.generation(), graph.slot()), counts);
                        }
                        ledger.budget.reserved -= 1;
                        reservation.remaining -= 1;
                        ledger.budget.captured += 1;
                        ledger.budget.peak = ledger.budget.peak.max(ledger.graphs.len());
                    }
                    recorded.and(ended.map(|_| ()).map_err(E::from))
                });
            if let Err(error) = captured {
                self.ledger.borrow_mut().budget.capture_failures += 1;
                let _ = self.ledger.borrow_mut().release_to(services, first);
                self.trace(
                    services,
                    "capture-failed",
                    Some(key),
                    needed,
                    capture_started.elapsed().as_micros(),
                );
                return Err(error);
            }
        }
        let graphs = self.ledger.borrow().graphs[first..].to_vec();
        if self.stats().capacity.is_some() {
            self.trace(
                services,
                "capture",
                Some(key),
                graphs.len(),
                capture_started.elapsed().as_micros(),
            );
        } else {
            eprintln!(
                "graph capture rank {} bucket {} items {}: {} pieces, {:.3}s",
                key.rank,
                key.bucket,
                key.items,
                graphs.len(),
                capture_started.elapsed().as_secs_f64()
            );
        }
        self.graphs.insert(key, graphs);
        Ok(())
    }

    /// Replays every rank's cached `(bucket, items)` graphs piece-major: each
    /// rank's first piece, then each rank's second, and so on, calling
    /// `first_piece(rank)` once a rank's first piece is enqueued. Returns
    /// false, launching nothing, when a rank still needs a capture.
    /// `streams` holds one stream per rank; any other count is refused,
    /// launching nothing.
    pub fn replay_piece_major(
        &self,
        services: &HostServicesV1,
        streams: &[BundleStreamHandleV1],
        (bucket, items): (u32, u32),
        mut first_piece: impl FnMut(usize),
    ) -> Result<bool, BundleAbiError> {
        if streams.len() != self.ranks.count() {
            return Err(BundleAbiError::InvalidRange("rank streams"));
        }
        let keys = (0..self.ranks.get())
            .map(|rank| GraphKey {
                rank,
                bucket,
                items,
            })
            .collect::<Vec<_>>();
        let Some(ranks) = self.cached(&keys) else {
            return Ok(false);
        };
        let depth = ranks.iter().map(|pieces| pieces.len()).max().unwrap_or(0);
        for piece in 0..depth {
            for (rank, pieces) in ranks.iter().enumerate() {
                let Some(&graph) = pieces.get(piece) else {
                    continue;
                };
                services.launch_graph(streams[rank], graph)?;
                self.replay_routes(graph);
                if piece == 0 {
                    first_piece(rank);
                }
            }
        }
        Ok(true)
    }
}

impl<G> GraphCache<G> {
    fn replay_routes(&self, graph: BundleGraphHandleV1) {
        if !crate::routes::active() {
            return;
        }
        if let Some(counts) = self
            .ledger
            .borrow()
            .routes
            .get(&(graph.generation(), graph.slot()))
        {
            crate::routes::replay(counts);
        }
    }

    /// The cached graph pieces of every rank in `keys`, or None when any
    /// rank still needs a capture.
    pub fn cached(&self, keys: &[GraphKey]) -> Option<Vec<&[BundleGraphHandleV1]>> {
        let graphs = self.live()?;
        keys.iter()
            .map(|key| graphs.get(key).map(Vec::as_slice))
            .collect()
    }

    pub fn contains(&self, key: GraphKey) -> bool {
        self.live().is_some_and(|graphs| graphs.contains_key(&key))
    }

    /// Releases every graph in this cache's ledger, newest first, including
    /// those of the caches sharing it, which all forget their lookups. A
    /// failed release leaves that graph and every older one in the ledger,
    /// so a later call retries it.
    pub fn release_all(&mut self, services: &HostServicesV1) -> Result<(), BundleAbiError> {
        let mut ledger = self.ledger.borrow_mut();
        if ledger.budget.reserved != 0 {
            return Err(BundleAbiError::Status(
                runtime_bundle_abi::BundleStatusV1::InvalidState,
            ));
        }
        ledger.epoch += 1;
        self.graphs.clear();
        self.epoch = ledger.epoch;
        let started = std::time::Instant::now();
        let result = ledger.release_to(services, 0);
        if result.is_ok() {
            ledger.budget.pressure = false;
        }
        drop(ledger);
        self.trace(
            services,
            if result.is_ok() {
                "release-all"
            } else {
                "release-failed"
            },
            None,
            0,
            started.elapsed().as_micros(),
        );
        result
    }

    /// The graphs live in this cache's ledger, whichever cache captured
    /// them: the next capture's first piece takes this position.
    pub fn ledger_len(&self) -> usize {
        self.ledger.borrow().graphs.len()
    }

    /// Releases this cache's graphs newest first when they are its ledger's
    /// newest (no cache sharing the ledger captured a graph since), and
    /// forgets its lookups; the caches sharing the ledger keep theirs.
    /// Returns false, releasing nothing, while a newer graph lives. A
    /// failed release leaves that graph and every older one in the ledger
    /// for a later `release_all`, and this cache forgets its lookups all
    /// the same.
    pub fn release_if_newest(&mut self, services: &HostServicesV1) -> Result<bool, BundleAbiError> {
        let Some(graphs) = self.live() else {
            return Ok(true);
        };
        let held = graphs.values().flatten().copied().collect::<Vec<_>>();
        let mut ledger = self.ledger.borrow_mut();
        if ledger.budget.reserved != 0 {
            return Err(BundleAbiError::Status(
                runtime_bundle_abi::BundleStatusV1::InvalidState,
            ));
        }
        let Some(keep) = ledger.graphs.len().checked_sub(held.len()) else {
            return Ok(false);
        };
        if !ledger.graphs[keep..]
            .iter()
            .all(|graph| held.contains(graph))
        {
            return Ok(false);
        }
        self.graphs.clear();
        ledger.release_to(services, keep)?;
        Ok(true)
    }

    pub fn len(&self) -> usize {
        self.live().map_or(0, BTreeMap::len)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::{decode_bucket, pieces, verify_bucket};
    use crate::geometry::tests::{Tp4 as Geometry, Wide};

    #[test]
    fn disabled_route_replay_never_borrows_the_ledger() {
        let cache = super::GraphCache::<Geometry>::new(super::Ranks::new(4).unwrap());
        let _disabled = crate::routes::RouteCounter::begin(false);
        let _exclusive = cache.ledger.borrow_mut();
        // An attempted ledger lookup would panic while this borrow is held.
        cache.replay_routes(super::BundleGraphHandleV1::from_parts(1, 1));
    }

    #[test]
    fn decode_buckets_round_up_to_declared_capacities() {
        let bucket = decode_bucket::<Geometry>;
        assert_eq!((1..=32).filter_map(bucket).count(), 32);
        assert_eq!(bucket(3), Some(4));
        assert_eq!(bucket(17), Some(32));
        assert_eq!(bucket(0), None);
        assert_eq!(bucket(33), None);
        assert!(
            (0..64)
                .filter(|&value| super::is_bucket(value, 32))
                .eq([1, 2, 4, 8, 16, 32])
        );
        // Without wider verify rows, verify buckets end at the decode batch.
        assert_eq!(verify_bucket::<Geometry>(32), Some(32));
        assert_eq!(verify_bucket::<Geometry>(33), None);
        assert_eq!(verify_bucket::<Wide>(33), Some(64));
        assert_eq!(verify_bucket::<Wide>(256), Some(256));
        assert_eq!(verify_bucket::<Wide>(257), None);
        assert_eq!(decode_bucket::<Wide>(33), None);
    }

    #[test]
    fn verify_tokens_never_fall_below_the_largest_decode_bucket() {
        use super::{verify_bucket_within, verify_tokens};
        use crate::geometry::Items;
        use crate::geometry::tests::{Max, Narrow};
        // Verify rows wider than the item capacity bound verify steps.
        for count in [1, 8, 32, 33, 256] {
            let items = Items::new(count).unwrap();
            assert_eq!(verify_tokens::<Max>(items), 256);
            assert_eq!(verify_bucket_within::<Max>(256, items), Some(256));
            assert_eq!(verify_bucket_within::<Max>(257, items), None);
        }
        // Narrower verify rows give way to the largest decode bucket.
        for (count, tokens) in [(1, 16), (16, 16), (17, 32), (64, 64), (200, 256)] {
            let items = Items::new(count).unwrap();
            assert_eq!(verify_tokens::<Narrow>(items), tokens, "{count} items");
            assert_eq!(verify_bucket_within::<Narrow>(tokens, items), Some(tokens));
            assert_eq!(verify_bucket_within::<Narrow>(tokens + 1, items), None);
            assert_eq!(verify_bucket_within::<Narrow>(0, items), None);
        }
    }

    #[test]
    fn graph_pieces_cover_the_program_in_order() {
        assert_eq!(pieces(588), [0..18, 18..98, 98..588]);
        assert_eq!(pieces(1).len(), 1);
        assert_eq!(pieces(1)[0], 0..1);
        assert_eq!(pieces(10), [0..1, 1..10]);
        assert!(pieces(0).is_empty());
    }
}
