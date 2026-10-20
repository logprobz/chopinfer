//! Every V1 host service, implemented over instance-scoped checked handles.
//!
//! One backend serves exactly one bundle instance. Each resource it hands out
//! is a generation-checked slot in this instance's own table, so a handle from
//! another instance, a released handle, or a reused slot's earlier handle is
//! rejected without reaching the device. Resources release in reverse
//! acquisition order; a release out of order is refused and changes nothing.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use runtime_bundle::VerifiedComponentFileV1;
use runtime_bundle_abi::{
    AuthenticatedComponentMetadataV1, BundleAuthenticatedComponentHandleV1,
    BundleCollectiveHandleV1, BundleCompiledImageHandleV1, BundleContextHandleV1,
    BundleCudaModuleHandleV1, BundleCudaSymbolHandleV1, BundleDeviceBufferHandleV1,
    BundleEventHandleV1, BundleGraphHandleV1, BundleKvRequestHandleV1, BundleKvTransactionHandleV1,
    BundleMemoryPoolHandleV1, BundlePeerAccessHandleV1, BundlePhysicalKvHandleV1,
    BundleRankArenaHandleV1, BundleStateHandleV1, BundleStatusV1, BundleStreamHandleV1,
    CollectiveRequestV1, CompiledImageSpanV1, CustomCompletionV1, CustomOperandV1,
    CustomOperationV1, DeviceBufferSpanV1, HostServiceResultV1, HostServicesBackendV1,
    KernelArgumentV1, LaunchConfigV1, MappedRowGatherV1, PhysicalKvRequestV1, RankArenaUploadV1,
    StateRequestV1,
};
use runtime_weights::sha256_bytes;

use runtime_kv::{
    DeclaredPageLayoutV1, KvGeometry, KvPool, PagedKvPlanV1, PhysicalPageIdV1, PreparedAppendBatch,
    PreparedAppendId, SequenceHandle, StateLifetimeV1,
};

use crate::device::{
    CollectiveBindingV1, DEVICE_POINTER_TABLE_ENTRY_BYTES_V1, DeviceArgumentV1,
    DeviceCustomOperandV1, DeviceObjectV1, DeviceRangeV1, DeviceTensorMapV1, GraphCaptureErrorV1,
    HostDeviceBackendV1, PinnedFileUploadV1,
};
use crate::handles::{SlotKeyV1, SlotTableV1, slot_key_v1};
use crate::host_rows::gather::{MappedProbe, ReadAhead, gather_span_v1};
#[cfg(test)]
use crate::probe::HostDeviceProbeServiceFaultV1;

/// The KV operation word a bundle passes to `prepare_kv`.
///
/// The payload is the operation word, then the additional token count as eight
/// little-endian bytes. Nothing in the word names a model.
pub const KV_APPEND_OPERATION_V1: &[u8] = b"append-tokens-v1";

/// The fixed prefix of the instance configuration a host hands a bundle.
pub const BUNDLE_CONFIGURATION_MAGIC_V1: &[u8; 20] = b"chopbundle-config-v1";
pub const AUTHENTICATED_COMPONENT_ROLE_PREFIX_V1: &[u8] = b"chop.execution-evidence.";
pub const BUNDLE_METADATA_ROLE_PREFIX_V1: &[u8] = b"chop.bundle-metadata.";
pub const MAX_AUTHENTICATED_COMPONENT_BYTES_V1: usize =
    runtime_bundle_abi::BUNDLE_AUTHENTICATED_COMPONENT_BYTES_MAX_V1 as usize;

/// The whole rank image a `resolve_compiled_image_view` call selects with an
/// empty view identifier.
///
/// Every other identifier names a view the bundle's compiled-image description
/// declared. The host matches the identifier and returns the declared bounded
/// span, so a bundle can carry as many views as its model needs.
pub const COMPILED_IMAGE_ARENA_VIEW_V1: &[u8] = b"";

/// Maximum resident source bytes used while uploading a file-backed image.
pub const FILE_BACKED_COMPONENT_UPLOAD_BUFFER_BYTES_V1: usize = 1024 * 1024;

/// The rank pool a `resolve_kv_view` call selects with an empty view identifier.
///
/// Eight little-endian bytes instead select one physical page of that rank's
/// pool. A bundle learns its physical pages from the page table the host
/// encodes into the batch request, so a page resolves before its append
/// commits as well as after.
pub const KV_RANK_POOL_VIEW_V1: &[u8] = b"";

/// The fixed device topology one host instance serves.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HostTopologyV1 {
    device_ordinals: Vec<i32>,
}

impl HostTopologyV1 {
    /// Builds a production topology with one distinct device per rank.
    pub fn new(device_ordinals: Vec<i32>) -> Result<Self, HostConfigErrorV1> {
        if device_ordinals.is_empty() {
            return Err(HostConfigErrorV1::EmptyTopology);
        }
        if device_ordinals.iter().any(|ordinal| *ordinal < 0) {
            return Err(HostConfigErrorV1::NegativeDeviceOrdinal);
        }
        for (rank, ordinal) in device_ordinals.iter().enumerate() {
            if device_ordinals[..rank].contains(ordinal) {
                return Err(HostConfigErrorV1::DuplicateDeviceOrdinal { ordinal: *ordinal });
            }
        }
        Ok(Self { device_ordinals })
    }

    #[must_use]
    pub fn rank_count(&self) -> u32 {
        u32::try_from(self.device_ordinals.len()).expect("a validated topology fits u32")
    }

    #[must_use]
    pub fn device_ordinal(&self, rank: u32) -> Option<i32> {
        self.device_ordinals.get(rank as usize).copied()
    }
}

/// The bundle-declared page layout this host will honour, with its identifier.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeclaredKvContractV1 {
    layout_id: Vec<u8>,
    layout: DeclaredPageLayoutV1,
}

impl DeclaredKvContractV1 {
    #[must_use]
    pub const fn new(layout_id: Vec<u8>, layout: DeclaredPageLayoutV1) -> Self {
        Self { layout_id, layout }
    }
}

/// The bundle-declared persistent state this host will honour.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeclaredStateContractV1 {
    state_id: Vec<u8>,
    bytes: u64,
    alignment: u64,
}

impl DeclaredStateContractV1 {
    #[must_use]
    pub const fn new(state_id: Vec<u8>, bytes: u64, alignment: u64) -> Self {
        Self {
            state_id,
            bytes,
            alignment,
        }
    }
}

/// One rank's byte range inside a compiled-image component.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompiledRankImageV1 {
    pub offset: u64,
    pub byte_length: u64,
}

/// One rank's byte range inside its own rank image, for one named view.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompiledViewSpanV1 {
    pub offset: u64,
    pub byte_length: u64,
}

/// One named bounded view, with the span it declares for each rank.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledImageViewV1 {
    alignment: u64,
    spans: BTreeMap<u32, CompiledViewSpanV1>,
    /// Kept in mapped host memory rather than the rank arena.
    host: bool,
    /// Mapped read-only from the checkpoint rather than held anywhere
    /// (`chop.host-mapped-gather.v1`).
    mapped: bool,
}

impl CompiledImageViewV1 {
    #[must_use]
    pub const fn new(alignment: u64, spans: BTreeMap<u32, CompiledViewSpanV1>) -> Self {
        Self {
            alignment,
            spans,
            host: false,
            mapped: false,
        }
    }

    /// Places this view in page-locked, device-mapped host memory: the upload
    /// cuts it out of the rank arena (see `HOST_RESIDENT_VIEWS_CAPABILITY_V1`).
    #[must_use]
    pub const fn on_host(mut self) -> Self {
        self.host = true;
        self
    }

    /// Places this view in a read-only mapping of the checkpoint whose rows
    /// the host gathers (`HOST_MAPPED_GATHER_CAPABILITY_V1`): the upload cuts
    /// it out of the rank arena like a host view and copies no byte of it.
    #[must_use]
    pub const fn mapped(mut self) -> Self {
        self.mapped = true;
        self
    }

    /// Whether the span leaves the rank arena.
    const fn cut(&self) -> bool {
        self.host || self.mapped
    }
}

/// How one rank image divides between its device arena and host memory.
///
/// Each host span, rounded up to the image alignment, is cut out of the rank
/// image. The arena holds what is left, in order, and the host buffer holds
/// the cuts, in order.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct RankImageLayoutV1 {
    image_bytes: u64,
    /// Disjoint rank-image ranges `(start, end)` in increasing order: the
    /// host views, which the host buffer holds.
    cuts: Vec<(u64, u64)>,
    /// The same for the mapped views, which nothing holds; disjoint from
    /// `cuts`.
    mapped: Vec<(u64, u64)>,
}

impl RankImageLayoutV1 {
    fn host_bytes(&self) -> u64 {
        self.cuts.iter().map(|(start, end)| end - start).sum()
    }

    fn mapped_bytes(&self) -> u64 {
        self.mapped.iter().map(|(start, end)| end - start).sum()
    }

    fn device_bytes(&self) -> u64 {
        self.image_bytes - self.host_bytes() - self.mapped_bytes()
    }

    /// Every range the arena leaves out, host and mapped, in order.
    fn removed(&self) -> Vec<(u64, u64)> {
        let mut removed = [&self.cuts[..], &self.mapped[..]].concat();
        removed.sort_unstable();
        removed
    }

    /// The rank-image ranges `(start, length)` the arena holds, in order.
    fn device_segments(&self) -> Vec<(u64, u64)> {
        let removed = self.removed();
        let mut segments = Vec::with_capacity(removed.len() + 1);
        let mut at = 0;
        for &(start, end) in &removed {
            if start > at {
                segments.push((at, start - at));
            }
            at = end;
        }
        if self.image_bytes > at {
            segments.push((at, self.image_bytes - at));
        }
        segments
    }

    /// Where one device range sits in the arena; a range touching a cut has
    /// no arena address.
    fn device_offset(&self, offset: u64, length: u64) -> HostServiceResultV1<u64> {
        let end = offset
            .checked_add(length)
            .ok_or(BundleStatusV1::InvalidArgument)?;
        let mut removed = 0;
        for &(start, cut_end) in &self.removed() {
            if end <= start {
                break;
            }
            if offset < cut_end {
                return Err(BundleStatusV1::InvalidArgument);
            }
            removed += cut_end - start;
        }
        Ok(offset - removed)
    }

    /// Where one host range sits in the host buffer; it lies inside one cut.
    fn host_offset(&self, offset: u64, length: u64) -> HostServiceResultV1<u64> {
        let end = offset
            .checked_add(length)
            .ok_or(BundleStatusV1::InvalidArgument)?;
        let mut before = 0;
        for &(start, cut_end) in &self.cuts {
            if start <= offset && end <= cut_end {
                return Ok(before + offset - start);
            }
            before += cut_end - start;
        }
        Err(BundleStatusV1::InvalidArgument)
    }
}

/// How one compiled-image component divides into rank images and named views.
///
/// The bundle declares this, the manifest authenticates the bytes it describes,
/// and the host applies it without reading a single byte. A bundle may name as
/// many views as its model needs; the host resolves each one by identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledImageDescriptionV1 {
    alignment: u64,
    ranks: Vec<CompiledRankImageV1>,
    views: BTreeMap<Vec<u8>, CompiledImageViewV1>,
}

impl CompiledImageDescriptionV1 {
    #[must_use]
    pub const fn new(
        alignment: u64,
        ranks: Vec<CompiledRankImageV1>,
        views: BTreeMap<Vec<u8>, CompiledImageViewV1>,
    ) -> Self {
        Self {
            alignment,
            ranks,
            views,
        }
    }

    /// The number of rank images this description divides its component into.
    #[must_use]
    pub fn rank_count(&self) -> usize {
        self.ranks.len()
    }

    /// Each rank image's range in component coordinates, in rank order.
    #[must_use]
    pub fn rank_images(&self) -> &[CompiledRankImageV1] {
        &self.ranks
    }

    fn rank_image(&self, rank: u32) -> HostServiceResultV1<CompiledRankImageV1> {
        self.ranks
            .get(rank as usize)
            .copied()
            .ok_or(BundleStatusV1::InvalidArgument)
    }

    /// Whether any view is placed in host memory.
    #[must_use]
    pub fn has_host_views(&self) -> bool {
        self.views.values().any(|view| view.host)
    }

    /// The arena and host division of one rank image, refusing host spans off
    /// the image alignment or overlapping each other.
    fn rank_layout(&self, rank: u32) -> HostServiceResultV1<RankImageLayoutV1> {
        let image_bytes = self.rank_image(rank)?.byte_length;
        let mut cuts = Vec::new();
        let mut mapped = Vec::new();
        for view in self.views.values().filter(|view| view.cut()) {
            let Some(span) = view.spans.get(&rank) else {
                continue;
            };
            if !span.offset.is_multiple_of(self.alignment) {
                return Err(BundleStatusV1::InvalidArgument);
            }
            let end = span
                .offset
                .checked_add(span.byte_length)
                .ok_or(BundleStatusV1::InvalidArgument)?;
            if span.byte_length == 0 || end > image_bytes {
                return Err(BundleStatusV1::InvalidArgument);
            }
            let cut =
                host_view_excision_v1(span.offset, span.byte_length, self.alignment, image_bytes)
                    .ok_or(BundleStatusV1::InvalidArgument)?;
            if view.mapped {
                mapped.push(cut);
            } else {
                cuts.push(cut);
            }
        }
        cuts.sort_unstable();
        mapped.sort_unstable();
        let layout = RankImageLayoutV1 {
            image_bytes,
            cuts,
            mapped,
        };
        if layout
            .removed()
            .windows(2)
            .any(|pair| pair[0].1 > pair[1].0)
        {
            return Err(BundleStatusV1::InvalidArgument);
        }
        // A view aligned above the image alignment could not keep it once
        // moved, and an empty arena has no allocation to hold.
        if !layout.removed().is_empty()
            && (layout.device_bytes() == 0
                || self
                    .views
                    .values()
                    .any(|view| view.alignment > self.alignment))
        {
            return Err(BundleStatusV1::InvalidArgument);
        }
        Ok(layout)
    }
}

/// Chunks at least this long are gathered on several threads: page faults
/// on a cold mapping then overlap.
const PARALLEL_GATHER_BYTES: usize = 1 << 20;

/// Most threads one chunk's gather uses.
const GATHER_THREADS: usize = 8;

/// Writes bytes `at..at + chunk.len()` of the rows `rows` of `table`, each
/// `row_bytes` long, laid end to end, into `chunk`. The caller has checked
/// every row against the table.
fn gather_rows_v1(
    table: &[u8],
    row_bytes: usize,
    rows: &[u32],
    at: usize,
    chunk: &mut [u8],
    ahead: &ReadAhead<'_>,
) {
    let threads = if chunk.len() < PARALLEL_GATHER_BYTES {
        1
    } else {
        std::thread::available_parallelism().map_or(1, |n| n.get().min(GATHER_THREADS))
    };
    if threads == 1 {
        return gather_span_v1(table, row_bytes, rows, at, chunk, ahead);
    }
    // Whole rows per thread, so no two threads share a row.
    let share = chunk.len().div_ceil(threads).div_ceil(row_bytes) * row_bytes;
    let first = row_bytes - at % row_bytes;
    std::thread::scope(|scope| {
        let (head, mut rest) = chunk.split_at_mut(first.min(chunk.len()));
        gather_span_v1(table, row_bytes, rows, at, head, ahead);
        let mut start = at + head.len();
        while !rest.is_empty() {
            let (part, tail) = rest.split_at_mut(share.min(rest.len()));
            let part_start = start;
            start += part.len();
            scope.spawn(move || gather_span_v1(table, row_bytes, rows, part_start, part, ahead));
            rest = tail;
        }
    });
}

#[cfg(test)]
mod gather_tests {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;

    use super::{ReadAhead, gather_rows_v1};
    use crate::host_rows::gather::GatherProbe;

    /// A clock that advances `step_us` per read, and the read-aheads asked.
    struct Scripted {
        step_us: u64,
        clock: AtomicU64,
        asked: Mutex<Vec<Vec<u32>>>,
    }

    impl Scripted {
        fn new(step_us: u64) -> Self {
            Self {
                step_us,
                clock: AtomicU64::new(0),
                asked: Mutex::new(Vec::new()),
            }
        }
    }

    impl GatherProbe for Scripted {
        fn now(&self) -> Duration {
            Duration::from_micros(self.clock.fetch_add(self.step_us, Ordering::Relaxed))
        }

        fn read_ahead(&self, rows: &[u32]) {
            self.asked.lock().unwrap().push(rows.to_vec());
        }
    }

    fn table(rows: usize, row_bytes: usize) -> Vec<u8> {
        (0..rows * row_bytes)
            .map(|i| (i * 31 + i / 251) as u8)
            .collect()
    }

    fn expected(table: &[u8], row_bytes: usize, rows: &[u32]) -> Vec<u8> {
        rows.iter()
            .flat_map(|&row| &table[row as usize * row_bytes..][..row_bytes])
            .copied()
            .collect()
    }

    /// Gathers in chunks of `chunk` bytes, as the ordered fill does.
    fn gather(
        table: &[u8],
        row_bytes: usize,
        rows: &[u32],
        chunk: usize,
        probe: &Scripted,
    ) -> Vec<u8> {
        let ahead = ReadAhead::new(probe);
        let mut out = vec![0; rows.len() * row_bytes];
        for (index, part) in out.chunks_mut(chunk).enumerate() {
            gather_rows_v1(table, row_bytes, rows, index * chunk, part, &ahead);
        }
        out
    }

    #[test]
    fn a_slow_row_group_reads_the_rest_of_the_gather_ahead_once() {
        let row_bytes = 96;
        let source = table(500, row_bytes);
        let rows: Vec<u32> = (0..100).map(|i| (i * 37 % 500) as u32).collect();
        // Each clock read is 50 us after the last, so the first 16 rows are
        // slow and later slow rows ask nothing more, whether a chunk holds
        // part of a row, one row or many.
        for chunk in [50, row_bytes, 1000, 5000] {
            let probe = Scripted::new(50);
            let out = gather(&source, row_bytes, &rows, chunk, &probe);
            assert_eq!(out, expected(&source, row_bytes, &rows));
            assert_eq!(
                *probe.asked.lock().unwrap(),
                [rows[16..].to_vec()],
                "{chunk}"
            );
        }
    }

    #[test]
    fn a_disabled_read_ahead_asks_nothing_and_copies_the_same_bytes() {
        let row_bytes = 96;
        let source = table(500, row_bytes);
        let rows: Vec<u32> = (0..100).map(|i| (i * 37 % 500) as u32).collect();
        // Every group is slow, as in the enabled case above.
        for chunk in [50, row_bytes, 1000, 5000] {
            let probe = Scripted::new(50);
            let ahead = ReadAhead::disabled(&probe);
            let mut out = vec![0; rows.len() * row_bytes];
            for (index, part) in out.chunks_mut(chunk).enumerate() {
                gather_rows_v1(&source, row_bytes, &rows, index * chunk, part, &ahead);
            }
            assert_eq!(out, expected(&source, row_bytes, &rows));
            assert!(probe.asked.lock().unwrap().is_empty(), "{chunk}");
        }
    }

    #[test]
    fn warm_gathers_ask_for_no_read_ahead() {
        let row_bytes = 96;
        let source = table(500, row_bytes);
        let rows: Vec<u32> = (0..100).map(|i| (i * 37 % 500) as u32).collect();
        // Rows at most 2.5 us each on average: no read-ahead, same bytes.
        for chunk in [50, row_bytes, 1000, 5000] {
            for step in [0, 2] {
                let probe = Scripted::new(step);
                let out = gather(&source, row_bytes, &rows, chunk, &probe);
                assert_eq!(out, expected(&source, row_bytes, &rows));
                assert!(probe.asked.lock().unwrap().is_empty(), "{chunk} {step}");
            }
        }
    }

    #[test]
    fn threaded_gathers_copy_the_same_bytes_and_read_ahead_at_most_once() {
        let row_bytes = 4000;
        let source = table(700, row_bytes);
        let rows: Vec<u32> = (0..600).map(|i| (i * 211 % 700) as u32).collect();
        for step in [0, 1000] {
            let probe = Scripted::new(step);
            // Chunks over the threaded size, starting inside a row.
            let out = gather(&source, row_bytes, &rows, (1 << 20) + 1234, &probe);
            assert_eq!(out, expected(&source, row_bytes, &rows));
            let asked = probe.asked.lock().unwrap();
            assert_eq!(asked.len(), usize::from(step > 0), "{step}");
            if let Some(rest) = asked.first() {
                assert!(rows.ends_with(rest) && rest.len() < rows.len());
            }
        }
    }
}

/// The rank-image bytes one host-placed span removes from the rank arena.
///
/// The span is rounded up to the image alignment, so every byte after it
/// moves down by a multiple of that alignment and keeps it, and clamped to
/// the rank image. Returns `(start, end)` in rank-image offsets, or `None`
/// when the rounded end overflows.
fn host_view_excision_v1(
    offset: u64,
    byte_length: u64,
    image_alignment: u64,
    image_byte_length: u64,
) -> Option<(u64, u64)> {
    let end = offset
        .checked_add(byte_length)?
        .checked_next_multiple_of(image_alignment)?;
    Some((offset, end.min(image_byte_length)))
}

/// One compiled-image component and how it divides.
#[derive(Clone, Debug, Eq, PartialEq)]
struct ComponentRecordV1 {
    storage: ComponentStorageV1,
    description: Option<CompiledImageDescriptionV1>,
    authenticated_sha256: Option<[u8; 64]>,
}

#[derive(Clone, Debug)]
enum ComponentStorageV1 {
    /// Bytes in memory: a component's verified bytes, or a test's image.
    Bytes(Arc<[u8]>),
    /// Rank images produced from a checkpoint by the bundle's weight plans.
    Planned(Arc<crate::direct::PlannedImageV1>),
}

/// Two guarded sources are the same source, not merely equal descriptions.
///
/// A guard is a live thing: two views of one payload can disagree about whether
/// it is still authorized, so only the same view compares equal to itself.
impl PartialEq for ComponentStorageV1 {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Bytes(left), Self::Bytes(right)) => left == right,
            (Self::Planned(left), Self::Planned(right)) => Arc::ptr_eq(left, right),
            _ => false,
        }
    }
}

impl Eq for ComponentStorageV1 {}

/// The serving process's startup nonce, when the generic taps directory is
/// set: every instance it creates then names the process to its bundle's
/// taps (`HostInstanceConfigV1::with_tap_producer`).
static TAP_PRODUCER_V1: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// The generic taps directory variable (`runtime_bundle_sdk::taps::TAPS_ENV`).
pub const TAPS_DIR_ENV_V1: &str = "CHOPINFER_TAPS_DIR";
/// The tag of the create record's tap producer field
/// (`runtime_bundle_sdk::taps::PRODUCER_TAG`).
pub const TAP_PRODUCER_TAG_V1: &[u8] = b"chop.tap-producer.v1";
/// The tag of the create record's context-parallel field: a little-endian
/// `u32` group size, sent only above 1. A group of N consecutive ranks holds
/// a sequence's physical page j on group rank j mod N, at the local index of
/// its logical page (`runtime_kv::ContextParallelPagesV1`).
pub const CONTEXT_PARALLEL_TAG_V1: &[u8] = b"chop.context-parallel.v1";

/// Records the serving process's startup nonce for its bundles' taps, when
/// taps are requested; called once at startup, before any bundle is created.
pub fn record_tap_producer_v1(startup_nonce: &str) {
    if std::env::var_os(TAPS_DIR_ENV_V1).is_some() {
        let _ = TAP_PRODUCER_V1.set(startup_nonce.to_owned());
    }
}

/// The startup nonce `record_tap_producer_v1` recorded, if any.
pub fn tap_producer_v1() -> Option<&'static str> {
    TAP_PRODUCER_V1.get().map(String::as_str)
}

/// Everything a host instance knows before its bundle's create callback runs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HostInstanceConfigV1 {
    topology: HostTopologyV1,
    components: BTreeMap<Vec<u8>, ComponentRecordV1>,
    kv: Option<DeclaredKvContractV1>,
    state: Option<DeclaredStateContractV1>,
    serving_kv: Option<ServingKvPolicyV1>,
    speculation: runtime_bundle_abi::BundleRawSpeculationV1,
    /// The live sequences the instance serves, when its manifest declares
    /// batch limits, followed by context for context-aware packages.
    maximum_sequences: Option<u32>,
    maximum_context_tokens: Option<u64>,
    batch_limits: runtime_bundle_abi::BundleRawBatchLimitsV1,
    /// The serving process's startup nonce, sent last when taps are
    /// requested (`runtime_bundle_sdk::taps::PRODUCER_TAG`).
    tap_producer: Option<String>,
    model_options: Option<BTreeMap<String, serde_json::Value>>,
    /// Ranks of one context-parallel group; 1 keeps every KV page on every
    /// rank and adds nothing to the create record.
    context_parallel_size: usize,
}

/// The host's serving-only share of free rank memory after a fixed 1 GiB reserve.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ServingKvPolicyV1 {
    /// Authenticated frontend vocabulary; reserve sampler buffers before KV.
    pub sampling_vocabulary: Option<usize>,
    /// Engine context limit, transmitted only to packages requiring served-context.
    pub maximum_context_tokens: Option<usize>,
    /// Per-rank memory held outside KV for request state and verify
    /// checkpoints the bundle has not allocated yet when it acquires the pool.
    /// State allocated before then is already outside free memory.
    pub state_reserved_bytes: u64,
    pub fraction_millionths: u32,
    pub maximum_pages: Option<u64>,
}

impl Default for ServingKvPolicyV1 {
    fn default() -> Self {
        Self {
            sampling_vocabulary: None,
            maximum_context_tokens: None,
            state_reserved_bytes: 0,
            fraction_millionths: 900_000,
            maximum_pages: None,
        }
    }
}

impl HostInstanceConfigV1 {
    #[must_use]
    pub const fn new(topology: HostTopologyV1) -> Self {
        Self {
            topology,
            components: BTreeMap::new(),
            kv: None,
            state: None,
            serving_kv: None,
            speculation: runtime_bundle_abi::BundleRawSpeculationV1::disabled(),
            maximum_sequences: None,
            maximum_context_tokens: None,
            tap_producer: None,
            model_options: None,
            context_parallel_size: 1,
            batch_limits: runtime_bundle_abi::BundleRawBatchLimitsV1::unreported(),
        }
    }

    /// Registers the bytes one manifest component role resolves to.
    ///
    /// A component registered without a compiled-image description can be
    /// opened, but it cannot be uploaded as a rank arena: the host will not
    /// guess how opaque bytes divide between ranks.
    pub fn with_component(
        self,
        role: impl Into<Vec<u8>>,
        bytes: Vec<u8>,
    ) -> Result<Self, HostConfigErrorV1> {
        self.insert_component(
            role.into(),
            ComponentStorageV1::Bytes(bytes.into()),
            None,
            None,
        )
    }

    /// Registers one component together with how it divides into rank images.
    pub fn with_compiled_image(
        self,
        role: impl Into<Vec<u8>>,
        bytes: Vec<u8>,
        description: CompiledImageDescriptionV1,
    ) -> Result<Self, HostConfigErrorV1> {
        self.insert_component(
            role.into(),
            ComponentStorageV1::Bytes(bytes.into()),
            Some(description),
            None,
        )
    }

    /// Registers one package-authenticated evidence descriptor.
    ///
    /// This is crate-private so direct callers cannot promote arbitrary bytes
    /// into authenticated evidence through the public host configuration API.
    pub(crate) fn with_authenticated_component_file(
        self,
        role: impl Into<Vec<u8>>,
        file: VerifiedComponentFileV1,
    ) -> Result<Self, HostConfigErrorV1> {
        let role = role.into();
        if !role.starts_with(AUTHENTICATED_COMPONENT_ROLE_PREFIX_V1)
            || file.byte_length() > MAX_AUTHENTICATED_COMPONENT_BYTES_V1 as u64
        {
            return Err(HostConfigErrorV1::InvalidAuthenticatedComponent);
        }
        let sha256 = file
            .sha256()
            .as_bytes()
            .try_into()
            .map_err(|_| HostConfigErrorV1::InvalidAuthenticatedComponent)?;
        self.insert_component(
            role,
            ComponentStorageV1::Bytes(file.shared_bytes()),
            None,
            Some(sha256),
        )
    }

    /// Registers small manifest-verified bundle metadata for named reads.
    pub(crate) fn with_bundle_metadata_file(
        self,
        role: impl Into<Vec<u8>>,
        file: VerifiedComponentFileV1,
    ) -> Result<Self, HostConfigErrorV1> {
        let role = role.into();
        if !role.starts_with(BUNDLE_METADATA_ROLE_PREFIX_V1)
            || file.byte_length() > MAX_AUTHENTICATED_COMPONENT_BYTES_V1 as u64
        {
            return Err(HostConfigErrorV1::InvalidAuthenticatedComponent);
        }
        let sha256 = file
            .sha256()
            .as_bytes()
            .try_into()
            .map_err(|_| HostConfigErrorV1::InvalidAuthenticatedComponent)?;
        self.insert_component(
            role,
            ComponentStorageV1::Bytes(file.shared_bytes()),
            None,
            Some(sha256),
        )
    }

    /// Registers a compiled image produced from a checkpoint by the bundle's
    /// weight plans, under the plans' image role, and the authenticated
    /// components the plans declare (for example rank indexes). Nothing is
    /// read until a rank arena is uploaded.
    pub fn with_planned_compiled_image(
        self,
        image: Arc<crate::direct::PlannedImageV1>,
    ) -> Result<Self, HostConfigErrorV1> {
        let mut config = self;
        for component in image.components() {
            if component.bytes.len() > MAX_AUTHENTICATED_COMPONENT_BYTES_V1 {
                return Err(HostConfigErrorV1::InvalidAuthenticatedComponent);
            }
            let sha256 = sha256_bytes(&component.bytes)
                .as_bytes()
                .try_into()
                .map_err(|_| HostConfigErrorV1::InvalidAuthenticatedComponent)?;
            config = config.insert_component(
                component.name.clone().into_bytes(),
                ComponentStorageV1::Bytes(component.bytes.clone().into()),
                None,
                Some(sha256),
            )?;
        }
        let role = image.image_role().as_bytes().to_vec();
        let description = image.description().clone();
        config.insert_component(
            role,
            ComponentStorageV1::Planned(image),
            Some(description),
            None,
        )
    }

    /// Adds one component, refusing a role another component already holds.
    ///
    /// A bundle opens a component by role, so a second component under the same
    /// role has no defined answer. Overwriting would silently pick one.
    fn insert_component(
        mut self,
        role: Vec<u8>,
        storage: ComponentStorageV1,
        description: Option<CompiledImageDescriptionV1>,
        authenticated_sha256: Option<[u8; 64]>,
    ) -> Result<Self, HostConfigErrorV1> {
        if self.components.contains_key(&role) {
            return Err(HostConfigErrorV1::DuplicateComponentRole);
        }
        self.components.insert(
            role,
            ComponentRecordV1 {
                storage,
                description,
                authenticated_sha256,
            },
        );
        Ok(self)
    }

    #[must_use]
    pub fn with_kv_contract(mut self, kv: DeclaredKvContractV1) -> Self {
        self.kv = Some(kv);
        self
    }

    #[must_use]
    pub fn with_serving_kv(mut self, policy: ServingKvPolicyV1) -> Self {
        self.serving_kv = Some(policy);
        self
    }

    #[must_use]
    pub fn with_speculation(
        mut self,
        speculation: runtime_bundle_abi::BundleRawSpeculationV1,
    ) -> Self {
        self.speculation = speculation;
        self
    }

    /// Serves with `size` ranks per context-parallel group.
    #[must_use]
    pub const fn with_context_parallel_size(mut self, size: usize) -> Self {
        self.context_parallel_size = size;
        self
    }

    #[must_use]
    pub const fn context_parallel_size(&self) -> usize {
        self.context_parallel_size
    }

    /// Names the live sequences the instance serves, which a bundle sizes its
    /// slot tables and graph buckets by.
    #[must_use]
    pub const fn with_maximum_sequences(mut self, maximum_sequences: u32) -> Self {
        self.maximum_sequences = Some(maximum_sequences);
        self
    }

    #[must_use]
    pub const fn maximum_sequences(&self) -> Option<u32> {
        self.maximum_sequences
    }

    /// Reports the batch limits the host schedules to the bundle's create.
    #[must_use]
    pub fn with_batch_limits(
        mut self,
        batch_limits: runtime_bundle_abi::BundleRawBatchLimitsV1,
    ) -> Self {
        self.batch_limits = batch_limits;
        self
    }

    #[must_use]
    pub fn with_maximum_context_tokens(mut self, tokens: u64) -> Self {
        self.maximum_context_tokens = Some(tokens);
        self
    }

    /// Names the serving process (its startup nonce) to a diagnostic
    /// bundle's taps, which record it in every step.
    #[must_use]
    pub fn with_tap_producer(mut self, startup_nonce: String) -> Self {
        self.tap_producer = Some(startup_nonce);
        self
    }

    #[must_use]
    pub fn with_state_contract(mut self, state: DeclaredStateContractV1) -> Self {
        self.state = Some(state);
        self
    }

    /// Encodes what a bundle needs to know about the instance it is created for.
    ///
    /// Nothing Rust-owned crosses the boundary, so this is a fixed-width little
    /// endian record the bundle decodes on its own side. Every value in it came
    /// from the manifest the host verified: the topology it selected, the KV and
    /// state layouts that manifest declared, and the component roles a bundle
    /// may open. Component bytes are not in the record; a bundle asks for those
    /// by role through `open_compiled_image`. The live sequences the instance
    /// serves are followed by the context limit only for context-aware packages.
    /// A bundle reading a record
    /// without them serves its package maximum. A context-parallel group
    /// size above 1 and the model options follow, each tagged. Last, when taps are
    /// requested, comes the serving process's startup nonce for the taps.
    #[must_use]
    pub fn with_model_options(mut self, values: BTreeMap<String, serde_json::Value>) -> Self {
        self.model_options = Some(values);
        self
    }

    pub fn encode_v1(&self, topology_id: &[u8]) -> Vec<u8> {
        let mut record = Vec::new();
        record.extend_from_slice(BUNDLE_CONFIGURATION_MAGIC_V1);
        push_identifier_v1(&mut record, topology_id);
        push_count_v1(&mut record, self.topology.device_ordinals.len());
        for ordinal in &self.topology.device_ordinals {
            record.extend_from_slice(&ordinal.to_le_bytes());
        }
        match self.kv.as_ref() {
            Some(kv) => {
                push_identifier_v1(&mut record, &kv.layout_id);
                for value in [
                    kv.layout.page_bytes(),
                    kv.layout.alignment_bytes(),
                    kv.layout.page_tokens() as u64,
                    kv.layout.rank_count() as u64,
                    state_lifetime_code_v1(kv.layout.state_lifetime()),
                ] {
                    record.extend_from_slice(&value.to_le_bytes());
                }
            }
            None => {
                push_identifier_v1(&mut record, &[]);
                record.extend_from_slice(&[0; 5 * size_of::<u64>()]);
            }
        }
        match self.state.as_ref() {
            Some(state) => {
                push_identifier_v1(&mut record, &state.state_id);
                record.extend_from_slice(&state.bytes.to_le_bytes());
                record.extend_from_slice(&state.alignment.to_le_bytes());
            }
            None => {
                push_identifier_v1(&mut record, &[]);
                record.extend_from_slice(&[0; 2 * size_of::<u64>()]);
            }
        }
        push_count_v1(&mut record, self.components.len());
        for role in self.components.keys() {
            push_identifier_v1(&mut record, role);
        }
        if let Some(maximum_sequences) = self.maximum_sequences {
            record.extend_from_slice(&maximum_sequences.to_le_bytes());
        }
        if let Some(tokens) = self.maximum_context_tokens {
            assert!(
                self.maximum_sequences.is_some(),
                "context requires served items"
            );
            record.extend_from_slice(&tokens.to_le_bytes());
        }
        if self.context_parallel_size > 1 {
            let size = u32::try_from(self.context_parallel_size)
                .expect("a context-parallel group divides the topology's ranks");
            record.extend_from_slice(CONTEXT_PARALLEL_TAG_V1);
            record.extend_from_slice(&size.to_le_bytes());
        }
        if let Some(values) = &self.model_options {
            let bytes = serde_json::to_vec(values).expect("model options serialize");
            record.extend_from_slice(runtime_bundle_abi::MODEL_OPTIONS_TAG_V1);
            push_identifier_v1(&mut record, &bytes);
        }
        if let Some(nonce) = &self.tap_producer {
            record.extend_from_slice(TAP_PRODUCER_TAG_V1);
            push_identifier_v1(&mut record, nonce.as_bytes());
        }
        record
    }
}

fn push_count_v1(record: &mut Vec<u8>, count: usize) {
    let count = u32::try_from(count).expect("a host instance declares fewer than 2^32 elements");
    record.extend_from_slice(&count.to_le_bytes());
}

fn push_identifier_v1(record: &mut Vec<u8>, identifier: &[u8]) {
    push_count_v1(record, identifier.len());
    record.extend_from_slice(identifier);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostConfigErrorV1 {
    EmptyTopology,
    NegativeDeviceOrdinal,
    /// Two ranks named one device, which would be two ranks on one context.
    DuplicateDeviceOrdinal {
        ordinal: i32,
    },
    DuplicateComponentRole,
    InvalidAuthenticatedComponent,
}

impl std::fmt::Display for HostConfigErrorV1 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Self::DuplicateDeviceOrdinal { ordinal } = self {
            return write!(
                formatter,
                "two host topology ranks name device ordinal {ordinal}, and one device is one rank"
            );
        }
        formatter.write_str(match self {
            Self::EmptyTopology => "a host topology must name at least one rank",
            Self::NegativeDeviceOrdinal => "a host topology device ordinal must not be negative",
            Self::DuplicateComponentRole => {
                "two host components cannot resolve to the same component role"
            }
            Self::InvalidAuthenticatedComponent => {
                "an authenticated component is not bounded evidence or differs from its manifest pin"
            }
            Self::DuplicateDeviceOrdinal { .. } => unreachable!("reported above"),
        })
    }
}

impl std::error::Error for HostConfigErrorV1 {}

struct ContextRecordV1 {
    object: DeviceObjectV1,
    rank: u32,
    order: u64,
    /// How many selections of this rank are live.
    ///
    /// One rank is one driver context, so selecting it twice hands back the
    /// same handle rather than a second name for the same object. The handle
    /// goes back to the driver when the last selection releases it.
    users: usize,
}

struct MemoryPoolRecordV1 {
    object: DeviceObjectV1,
    context: SlotKeyV1,
    order: u64,
    /// How many selections of this context's pool are live, for the same
    /// reason: a context has one pool, so it has one handle.
    users: usize,
}

/// Who may release one allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BufferOwnerV1 {
    /// The bundle allocated it and must release it.
    Bundle(u64),
    /// A rank arena or KV pool owns it, and releases it with itself.
    Host,
}

struct DeviceBufferRecordV1 {
    object: DeviceObjectV1,
    byte_length: u64,
    owner: BufferOwnerV1,
    /// The pool, and its context, this allocation came from.
    ///
    /// One allocation belongs to one pool of one context, and that pair is what
    /// decides whether work in some other context may name its address at all:
    /// the context decides which directed lease is needed, and the pool decides
    /// which grant is. Nothing else in this record can answer either, so both
    /// are recorded when the allocation is made rather than inferred later. The
    /// host's own KV and state ranks have no pool or context handle at all, so
    /// their allocations carry `None` and are refused as cross-context addresses
    /// rather than guessed at.
    origin: Option<BufferOriginV1>,
}

/// The pool and context one allocation came from.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BufferOriginV1 {
    context: SlotKeyV1,
    pool: SlotKeyV1,
}

struct CompiledImageRecordV1 {
    storage: ComponentStorageV1,
    description: Option<CompiledImageDescriptionV1>,
    order: u64,
}

struct AuthenticatedComponentRecordV1 {
    role: Arc<[u8]>,
    payload: Arc<[u8]>,
    sha256: [u8; 64],
    order: u64,
}

struct RankArenaRecordV1 {
    image: SlotKeyV1,
    rank: u32,
    buffer: SlotKeyV1,
    /// The mapped host buffer holding this rank's host-placed views.
    host_buffer: Option<SlotKeyV1>,
    layout: RankImageLayoutV1,
    alignment: u64,
    order: u64,
}

struct PreparedRankUploadV1 {
    arena: BundleRankArenaHandleV1,
    stream: DeviceObjectV1,
    destination: DeviceObjectV1,
    storage: ComponentStorageV1,
    /// Component ranges `(offset, length)` the arena receives, in order.
    segments: Vec<(u64, u64)>,
}

impl PreparedRankUploadV1 {
    fn byte_length(&self) -> u64 {
        self.segments.iter().map(|(_, length)| length).sum()
    }

    /// The component offset of a single-range arena; segmented arenas read
    /// through [`read_segments_v1`] from zero.
    fn source_offset(&self) -> u64 {
        match self.segments[..] {
            [(offset, _)] => offset,
            _ => 0,
        }
    }
}

/// Uploads one prepared arena: its single component range as before, or its
/// device segments as one contiguous range.
fn upload_prepared_rank_v1<D: HostDeviceBackendV1>(
    device: &mut D,
    rank: &PreparedRankUploadV1,
) -> HostServiceResultV1<()> {
    if let [(source_offset, byte_length)] = rank.segments[..] {
        return upload_component_range_v1(
            device,
            rank.stream,
            rank.destination,
            &rank.storage,
            source_offset,
            byte_length,
        );
    }
    let byte_length = rank.byte_length();
    let mut read =
        |offset, chunk: &mut [u8]| read_segments_v1(&rank.storage, &rank.segments, offset, chunk);
    if let Some(result) =
        device.upload_pinned_file_range(rank.stream, rank.destination, 0, byte_length, &mut read)
    {
        return result;
    }
    upload_chunked_component_range_v1(device, rank.stream, rank.destination, 0, byte_length, read)
}

/// Reads component bytes as if `segments` were one contiguous range starting
/// at offset zero, so an arena with host views cut out uploads like any other.
fn read_segments_v1(
    storage: &ComponentStorageV1,
    segments: &[(u64, u64)],
    mut offset: u64,
    mut chunk: &mut [u8],
) -> bool {
    for &(start, length) in segments {
        if chunk.is_empty() {
            break;
        }
        if offset >= length {
            offset -= length;
            continue;
        }
        let count = (length - offset).min(chunk.len() as u64) as usize;
        let (head, tail) = chunk.split_at_mut(count);
        if !read_component_bytes_v1(storage, start + offset, head) {
            return false;
        }
        chunk = tail;
        offset = 0;
    }
    chunk.is_empty()
}

fn read_component_bytes_v1(
    storage: &ComponentStorageV1,
    offset: u64,
    destination: &mut [u8],
) -> bool {
    match storage {
        ComponentStorageV1::Planned(image) => match image.read(offset, destination) {
            Ok(()) => true,
            Err(error) => {
                eprintln!("host direct load failed: {error}");
                false
            }
        },
        ComponentStorageV1::Bytes(bytes) => usize::try_from(offset)
            .ok()
            .and_then(|start| bytes.get(start..start.checked_add(destination.len())?))
            .map(|source| destination.copy_from_slice(source))
            .is_some(),
    }
}

/// Bytes read per call while filling a mapped host buffer.
const HOST_VIEW_READ_BYTES_V1: usize = 64 << 20;

type FileChunkReaderV1<'a> = Box<dyn FnMut(u64, &mut [u8]) -> bool + Send + 'a>;

/// Snapshots separate arena preparation (including host views) from upload.
struct DirectLoadTimingV1 {
    images: Vec<(
        Arc<crate::direct::PlannedImageV1>,
        crate::direct::DirectLoadStatsV1,
    )>,
    prepared: Vec<crate::direct::DirectLoadStatsV1>,
    started: std::time::Instant,
    preparation_seconds: f64,
}

impl DirectLoadTimingV1 {
    fn new<D: HostDeviceBackendV1>(state: &HostStateV1<D>, uploads: &[RankArenaUploadV1]) -> Self {
        let mut images = Vec::new();
        for upload in uploads {
            let Ok(key) = slot_key_v1(upload.image.generation(), upload.image.slot()) else {
                continue;
            };
            let Ok(record) = state.images.get(key) else {
                continue;
            };
            if let ComponentStorageV1::Planned(image) = &record.storage
                && !images.iter().any(|(seen, _)| Arc::ptr_eq(seen, image))
            {
                images.push((Arc::clone(image), image.stats()));
            }
        }
        Self {
            images,
            prepared: Vec::new(),
            started: std::time::Instant::now(),
            preparation_seconds: 0.0,
        }
    }

    fn prepared(&mut self) {
        self.preparation_seconds = self.started.elapsed().as_secs_f64();
        self.prepared = self.images.iter().map(|(image, _)| image.stats()).collect();
    }

    fn report(&self, upload_seconds: f64, digest_seconds: f64) {
        for ((image, before), prepared) in self.images.iter().zip(&self.prepared) {
            let stats = image.stats();
            let mut line = serde_json::json!({
                    "seconds": upload_seconds, "digest_seconds": digest_seconds,
                    // Retain the existing fields, explicitly marked as cumulative.
                    "counter_scope": "cumulative",
                    "produced_bytes": stats.produced_bytes, "source_bytes": stats.source_bytes,
                    "source_reads": stats.source_reads, "advised_bytes": stats.advised_bytes,
                    "direct_bytes": stats.direct_bytes, "read_seconds": stats.read_seconds,
                    "produce_seconds": stats.produce_seconds, "shared_seconds": stats.shared_seconds,
                    "rank_seconds": stats.rank_seconds, "cumulative": stats,
                    "retained_peak_bytes": stats.retained_peak_bytes,
                    "pooled_peak_bytes": stats.pooled_peak_bytes,
                    "preparation": {"seconds": self.preparation_seconds, "counters": prepared.since(*before)},
                    "upload": {"seconds": upload_seconds, "counters": stats.since(*prepared)},
                    "checkpoint_identity": image.checkpoint().identity_hex(),
            });
            if image.mapped_bytes() > 0 {
                line["mapped_bytes"] = image.mapped_bytes().into();
            }
            if let Some(prewarm) = image.prewarm() {
                line["prewarm_bytes"] = prewarm.bytes.into();
                line["prewarm_seconds"] = prewarm.seconds.into();
                line["prewarm_resident_bytes"] = prewarm.resident_bytes.into();
            }
            eprintln!("startup direct_load: {line}");
        }
    }
}

fn prepare_rank_upload_v1<D: HostDeviceBackendV1>(
    state: &mut HostStateV1<D>,
    upload: RankArenaUploadV1,
) -> HostServiceResultV1<PreparedRankUploadV1> {
    let context_key = slot_key_v1(upload.context.generation(), upload.context.slot())?;
    let image_key = slot_key_v1(upload.image.generation(), upload.image.slot())?;
    state.refuse_during_capture(context_key)?;
    let stream = state.stream_object(upload.stream)?;
    let pool_key = slot_key_v1(upload.pool.generation(), upload.pool.slot())?;
    let pool_object = state.pools.get(pool_key)?.object;
    if state.contexts.get(context_key)?.rank != upload.rank
        || upload.rank >= state.config.topology.rank_count()
    {
        return Err(BundleStatusV1::InvalidArgument);
    }
    let record = state.images.get(image_key)?;
    let description = record
        .description
        .as_ref()
        .ok_or(BundleStatusV1::Unsupported)?;
    let alignment = description.alignment;
    let rank_image = description.rank_image(upload.rank)?;
    let layout = description.rank_layout(upload.rank)?;
    let storage = record.storage.clone();
    validate_component_range_v1(
        component_byte_length_v1(&storage),
        rank_image.offset,
        rank_image.byte_length,
    )?;
    let origin = Some(BufferOriginV1 {
        context: context_key,
        pool: pool_key,
    });
    let buffer = state.allocate_host_buffer(
        pool_object,
        origin,
        Some(stream),
        layout.device_bytes(),
        alignment,
    )?;
    let destination = state.buffers.get(buffer)?.object;
    let host_buffer = if layout.cuts.is_empty() {
        None
    } else {
        let cuts: Vec<_> = layout
            .cuts
            .iter()
            .map(|(start, end)| (rank_image.offset + start, end - start))
            .collect();
        let mut fill = |bytes: &mut [u8]| {
            bytes
                .chunks_mut(HOST_VIEW_READ_BYTES_V1)
                .enumerate()
                .all(|(index, chunk)| {
                    let offset = (index * HOST_VIEW_READ_BYTES_V1) as u64;
                    read_segments_v1(&storage, &cuts, offset, chunk)
                })
        };
        match state.allocate_mapped_host_buffer(
            stream,
            origin,
            layout.host_bytes(),
            alignment,
            &mut fill,
        ) {
            Ok(key) => Some(key),
            Err(status) => {
                let _ = state.release_host_buffer(buffer);
                return Err(status);
            }
        }
    };
    let release_buffers = |state: &mut HostStateV1<D>| {
        if let Some(host_buffer) = host_buffer {
            let _ = state.release_host_buffer(host_buffer);
        }
        let _ = state.release_host_buffer(buffer);
    };
    let order = match state.order.acquire() {
        Ok(order) => order,
        Err(status) => {
            release_buffers(state);
            return Err(status);
        }
    };
    let segments = layout
        .device_segments()
        .into_iter()
        .map(|(start, length)| (rank_image.offset + start, length))
        .collect();
    let arena = match state.arenas.insert(RankArenaRecordV1 {
        image: image_key,
        rank: upload.rank,
        buffer,
        host_buffer,
        layout,
        alignment,
        order,
    }) {
        Ok(key) => BundleRankArenaHandleV1::from_parts(key.generation, key.slot),
        Err(status) => {
            let _ = state.order.release(order);
            release_buffers(state);
            return Err(status);
        }
    };
    Ok(PreparedRankUploadV1 {
        arena,
        stream,
        destination,
        storage,
        segments,
    })
}

/// One directed context peer edge this instance holds a lease over.
///
/// Ordinals identify the physical edge even when several ranks share it.
/// Logical rank handles remain distinct for uploads, streams and sampling.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ContextPeerEdgeKeyV1 {
    owner: i32,
    peer: i32,
}

/// One pool grant this instance holds a lease over.
///
/// Each device has one default pool. Its grants are keyed by device pair,
/// so one acquisition cannot revoke access another still holds.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct PoolPeerEdgeKeyV1 {
    pool: i32,
    peer: i32,
}

/// What one instance-scoped peer lease owes the device it runs on.
#[derive(Clone, Copy, Debug)]
struct PeerLeaseV1 {
    /// How many live acquisitions share this edge.
    users: usize,
    /// Whether this host performed the transition that enabled the edge.
    ///
    /// A grant that was already there belongs to whoever made it, and this
    /// host gives its lease back without touching the device state.
    host_acquired: bool,
    /// The backend peer record, or the original default-pool handle.
    object: Option<DeviceObjectV1>,
    /// Preserve the first pool grant call for backends with logical aliases.
    pool_peer: Option<DeviceObjectV1>,
}

struct PeerAccessRecordV1 {
    edge: ContextPeerEdgeKeyV1,
    order: u64,
}

struct StreamRecordV1 {
    object: DeviceObjectV1,
    /// The context this stream belongs to.
    ///
    /// Work ordered on a stream runs in that stream's context, so this is the
    /// consumer whose leases and grants decide which addresses that work may
    /// name. A record that could not answer it would leave the rule unstated
    /// for every service that takes a stream and no symbol.
    context: SlotKeyV1,
    stage: Option<StageStreamKeyV1>,
    order: u64,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct StageStreamKeyV1 {
    context_slot: u32,
    rank: u32,
    stage_id: Vec<u8>,
    stream_kind: u32,
}

struct EventRecordV1 {
    object: DeviceObjectV1,
    context: SlotKeyV1,
    order: u64,
    recorded: bool,
}

struct GraphRecordV1 {
    object: DeviceObjectV1,
    context: SlotKeyV1,
    order: u64,
}

struct ModuleRecordV1 {
    object: DeviceObjectV1,
    /// The context this module was loaded into, which every symbol in it
    /// launches in.
    context: SlotKeyV1,
    order: u64,
}

struct SymbolRecordV1 {
    object: DeviceObjectV1,
    /// The launching context, carried from the module this symbol came from.
    context: SlotKeyV1,
    order: u64,
}

struct PhysicalKvRecordV1 {
    plan: PagedKvPlanV1,
    pool: KvPool,
    rank_buffers: Vec<SlotKeyV1>,
    order: u64,
}

struct ServingKvRecordV1 {
    plan: PagedKvPlanV1,
    rank_buffers: Vec<(SlotKeyV1, DeviceObjectV1)>,
}

struct KvRequestRecordV1 {
    kv: SlotKeyV1,
    sequence: SequenceHandle,
    order: u64,
}

struct KvTransactionRecordV1 {
    kv: SlotKeyV1,
    prepared: PreparedAppendId,
    /// The page work this prepared append declared, kept until it commits.
    ///
    /// `runtime-kv` reserved these pages logically. The bytes behind them are
    /// only made to match once the commit runs, so the description has to
    /// survive `prepare_kv` rather than being recomputed later.
    pages: KvPageWorkV1,
    order: u64,
}

/// The generic device work one prepared append declared.
///
/// Reserving a page says nothing about its contents, so a commit zeroes every
/// page the append newly reserved and then copies the initialized prefix of a
/// shared page onto the private page that replaced it. Both are expressed as the
/// host's generic zero and copy operations over rank buffers, so nothing here
/// interprets what a page segment holds.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct KvPageWorkV1 {
    zero: Vec<PhysicalPageIdV1>,
    copies: Vec<KvPageCopyV1>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct KvPageCopyV1 {
    source: PhysicalPageIdV1,
    destination: PhysicalPageIdV1,
    initialized_tokens: usize,
}

struct MappedGatherRecordV1 {
    object: DeviceObjectV1,
    streams: Vec<SlotKeyV1>,
    order: u64,
}

struct StateRecordV1 {
    buffers: Vec<SlotKeyV1>,
    order: u64,
}

struct CollectiveRecordV1 {
    binding: CollectiveBindingV1,
    order: u64,
}

/// Enforces that resources release in reverse acquisition order.
struct ReleaseOrderV1 {
    live: Vec<u64>,
    next: u64,
}

impl ReleaseOrderV1 {
    const fn new() -> Self {
        Self {
            live: Vec::new(),
            next: 1,
        }
    }

    fn acquire(&mut self) -> HostServiceResultV1<u64> {
        let order = self.next;
        self.next = self
            .next
            .checked_add(1)
            .ok_or(BundleStatusV1::ResourceExhausted)?;
        self.live.push(order);
        Ok(order)
    }

    /// Checks that this resource is the newest live one, and changes nothing.
    ///
    /// Every release preflights here, performs whatever can still fail, and only
    /// then calls [`Self::release`]. A finalizer that fails therefore leaves the
    /// resource tracked and its release retryable, rather than dropping the last
    /// record of something the driver still owns.
    fn preflight(&self, order: u64) -> HostServiceResultV1<()> {
        if self.live.last() == Some(&order) {
            Ok(())
        } else {
            Err(BundleStatusV1::InvalidState)
        }
    }

    fn release(&mut self, order: u64) -> HostServiceResultV1<()> {
        self.preflight(order)?;
        self.live.pop();
        Ok(())
    }

    fn live_count(&self) -> usize {
        self.live.len()
    }
}

/// One rank's device resources, owned by the host rather than the bundle.
///
/// The KV and state services allocate from these, because their ABI calls carry
/// no context, pool, or stream. They are created on first use and released once
/// the last KV pool or state allocation that needs them is gone.
struct HostRankV1 {
    context: DeviceObjectV1,
    context_owned: bool,
    pool: DeviceObjectV1,
    pool_owned: bool,
    stream: DeviceObjectV1,
}

struct HostStateV1<D> {
    device: D,
    config: HostInstanceConfigV1,
    order: ReleaseOrderV1,
    host_ranks: Vec<HostRankV1>,
    host_rank_users: usize,
    contexts: SlotTableV1<ContextRecordV1>,
    pools: SlotTableV1<MemoryPoolRecordV1>,
    buffers: SlotTableV1<DeviceBufferRecordV1>,
    images: SlotTableV1<CompiledImageRecordV1>,
    authenticated_components: SlotTableV1<AuthenticatedComponentRecordV1>,
    arenas: SlotTableV1<RankArenaRecordV1>,
    peers: SlotTableV1<PeerAccessRecordV1>,
    /// One lease per directed context edge, however many handles share it.
    context_peer_leases: BTreeMap<ContextPeerEdgeKeyV1, PeerLeaseV1>,
    /// One lease per pool grant, however many acquires share it.
    pool_peer_leases: BTreeMap<PoolPeerEdgeKeyV1, PeerLeaseV1>,
    /// The live context handle for each rank, so one rank has one handle.
    context_by_rank: BTreeMap<u32, SlotKeyV1>,
    /// The live pool handle for each context, for the same reason.
    pool_by_context: BTreeMap<SlotKeyV1, SlotKeyV1>,
    streams: SlotTableV1<StreamRecordV1>,
    stage_streams: BTreeMap<StageStreamKeyV1, SlotKeyV1>,
    events: SlotTableV1<EventRecordV1>,
    graphs: SlotTableV1<GraphRecordV1>,
    captures: BTreeMap<SlotKeyV1, SlotKeyV1>,
    /// Free device bytes of each capturing context when its capture began,
    /// when the device reports them.
    capture_free_bytes: BTreeMap<SlotKeyV1, u64>,
    /// The streams each open capture (keyed by its context) records: the
    /// capturing stream and those that joined it by waiting on an event the
    /// capture recorded.
    capture_members: BTreeMap<SlotKeyV1, BTreeSet<SlotKeyV1>>,
    /// Events recorded inside an open capture, by the capture's context.
    capture_events: BTreeMap<SlotKeyV1, SlotKeyV1>,
    modules: SlotTableV1<ModuleRecordV1>,
    symbols: SlotTableV1<SymbolRecordV1>,
    kv: SlotTableV1<PhysicalKvRecordV1>,
    serving_kv: Option<ServingKvRecordV1>,
    offload: Option<crate::offload::OffloadQueue>,
    kv_requests: SlotTableV1<KvRequestRecordV1>,
    kv_transactions: SlotTableV1<KvTransactionRecordV1>,
    states: SlotTableV1<StateRecordV1>,
    mapped_gathers: SlotTableV1<MappedGatherRecordV1>,
    collectives: SlotTableV1<CollectiveRecordV1>,
    #[cfg(test)]
    probe_fault: Option<HostDeviceProbeServiceFaultV1>,
    #[cfg(test)]
    probe_fault_observed: bool,
}

/// The host memory one instance holds ([`HostServicesV1Backend::host_memory_v1`]).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct HostMemoryV1 {
    /// Page-locked bytes: the host-resident views and the staging the
    /// device backend holds for gathers.
    pub pinned_bytes: u64,
    /// The gather staging's part of `pinned_bytes`.
    pub gather_staging_bytes: u64,
    /// The bytes the rank arenas leave to mapped views, each view rounded
    /// to its image alignment and summed over ranks; not page-locked.
    pub mapped_view_bytes: u64,
}

/// One bundle instance's complete model-neutral host service table.
pub struct HostServicesV1Backend<D: HostDeviceBackendV1> {
    state: RefCell<HostStateV1<D>>,
}

impl<D: HostDeviceBackendV1> HostServicesV1Backend<D> {
    /// Allocate once after bundle create has acquired the serving pool.
    pub fn initialize_kv_offload(
        &self,
        options: crate::offload::KvOffloadOptionsV1,
        available: u64,
    ) -> Result<Option<crate::offload::HostArenaPlanV1>, String> {
        if options.budget_bytes == 0 {
            return Ok(None);
        }
        let mut state = self.state.borrow_mut();
        if state.offload.is_some() {
            return Err("host KV arenas already initialized".into());
        }
        let pool = state
            .serving_kv
            .as_ref()
            .ok_or("KV offload requires a serving pool")?;
        let plan = crate::offload::HostArenaPlanV1::new(
            options.budget_bytes,
            available,
            pool.plan.bytes_per_page(),
            pool.rank_buffers.len(),
        )?;
        let buffers = pool
            .rank_buffers
            .iter()
            .map(|(key, _)| state.buffers.get(*key).map(|b| b.object))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("host KV pool: {e:?}"))?;
        let device_pages = pool.plan.page_capacity() as usize;
        let queue = crate::offload::OffloadQueue::new(
            &mut state.device,
            &buffers,
            device_pages,
            plan,
            options.bytes_per_second,
        )
        .map_err(|e| format!("host KV pinning of {} bytes failed: {e:?}; check page-locked memory limits and docs/deploy-common.md", plan.allocated_bytes))?;
        state.offload = Some(queue);
        Ok(Some(plan))
    }

    /// Capacity acquired by a serving bundle during create.
    #[must_use]
    pub fn serving_kv_pages(&self) -> Option<u64> {
        self.state
            .borrow()
            .serving_kv
            .as_ref()
            .map(|pool| pool.plan.page_capacity())
    }

    /// Makes a prepared mixed batch's new pages ready before bundle execute.
    pub fn prepare_serving_batch(&self, prepared: &PreparedAppendBatch) -> HostServiceResultV1<()> {
        let mut state = self.state.borrow_mut();
        let pool = state
            .serving_kv
            .as_ref()
            .ok_or(BundleStatusV1::InvalidState)?;
        let work = kv_page_work_v1(prepared);
        if work.zero.is_empty() && work.copies.is_empty() {
            return Ok(());
        }
        let plan = pool.plan;
        let ranks = pool.rank_buffers.clone();
        let mut pages = work.zero.clone();
        for copy in &work.copies {
            pages.push(copy.source);
            pages.push(copy.destination);
        }
        for page in pages {
            plan.validate_pages(&[page])
                .map_err(|_| BundleStatusV1::InvalidArgument)?;
        }
        for (buffer, stream) in ranks {
            let object = state.buffers.get(buffer)?.object;
            for page in &work.zero {
                state.device.zero_device_buffer(
                    stream,
                    DeviceRangeV1 {
                        object,
                        offset: plan
                            .page_offset(*page)
                            .map_err(|_| BundleStatusV1::InvalidState)?,
                        length: plan.bytes_per_page(),
                    },
                )?;
            }
            // A serving page's layout is the bundle's, and a token's rows need
            // not be contiguous in it, so the whole page is copied. Rows past
            // the initialized tokens are overwritten before any query reads them.
            for copy in &work.copies {
                state.device.copy_device_to_device(
                    stream,
                    DeviceRangeV1 {
                        object,
                        offset: plan
                            .page_offset(copy.destination)
                            .map_err(|_| BundleStatusV1::InvalidState)?,
                        length: plan.bytes_per_page(),
                    },
                    DeviceRangeV1 {
                        object,
                        offset: plan
                            .page_offset(copy.source)
                            .map_err(|_| BundleStatusV1::InvalidState)?,
                        length: plan.bytes_per_page(),
                    },
                )?;
            }
            state.device.synchronize_stream(stream)?;
        }
        Ok(())
    }
    #[must_use]
    pub fn new(device: D, config: HostInstanceConfigV1) -> Self {
        Self {
            state: RefCell::new(HostStateV1 {
                device,
                config,
                order: ReleaseOrderV1::new(),
                host_ranks: Vec::new(),
                host_rank_users: 0,
                contexts: SlotTableV1::new(),
                pools: SlotTableV1::new(),
                buffers: SlotTableV1::new(),
                images: SlotTableV1::new(),
                authenticated_components: SlotTableV1::new(),
                arenas: SlotTableV1::new(),
                peers: SlotTableV1::new(),
                context_peer_leases: BTreeMap::new(),
                pool_peer_leases: BTreeMap::new(),
                context_by_rank: BTreeMap::new(),
                pool_by_context: BTreeMap::new(),
                streams: SlotTableV1::new(),
                stage_streams: BTreeMap::new(),
                events: SlotTableV1::new(),
                graphs: SlotTableV1::new(),
                captures: BTreeMap::new(),
                capture_free_bytes: BTreeMap::new(),
                capture_members: BTreeMap::new(),
                capture_events: BTreeMap::new(),
                modules: SlotTableV1::new(),
                symbols: SlotTableV1::new(),
                kv: SlotTableV1::new(),
                serving_kv: None,
                offload: None,
                kv_requests: SlotTableV1::new(),
                kv_transactions: SlotTableV1::new(),
                states: SlotTableV1::new(),
                mapped_gathers: SlotTableV1::new(),
                collectives: SlotTableV1::new(),
                #[cfg(test)]
                probe_fault: None,
                #[cfg(test)]
                probe_fault_observed: false,
            }),
        }
    }

    #[cfg(test)]
    pub(crate) fn new_with_probe_fault(
        device: D,
        config: HostInstanceConfigV1,
        fault: HostDeviceProbeServiceFaultV1,
    ) -> Self {
        let backend = Self::new(device, config);
        backend.arm_probe_fault(fault);
        backend
    }

    #[cfg(test)]
    pub(crate) fn arm_probe_fault(&self, fault: HostDeviceProbeServiceFaultV1) {
        let mut state = self.state.borrow_mut();
        state.probe_fault = Some(fault);
        state.probe_fault_observed = false;
    }

    #[cfg(test)]
    pub(crate) fn probe_fault_observed(&self) -> bool {
        self.state.borrow().probe_fault_observed
    }

    /// The host memory this instance holds now: page-locked bytes (host
    /// views and gather staging), and the bytes its rank arenas leave to
    /// checkpoint mappings, which no page-locked or device budget holds
    /// (summed over ranks, though ranks naming the same bytes share one
    /// mapping).
    #[must_use]
    pub fn host_memory_v1(&self) -> HostMemoryV1 {
        let state = self.state.borrow();
        let gather_staging_bytes = state.device.fill_staging_bytes();
        let arenas = state.arenas.values();
        let (host_view_bytes, mapped_view_bytes) = arenas.fold((0, 0), |(host, mapped), arena| {
            (
                host + arena.layout.host_bytes(),
                mapped + arena.layout.mapped_bytes(),
            )
        });
        HostMemoryV1 {
            pinned_bytes: host_view_bytes + gather_staging_bytes,
            gather_staging_bytes,
            mapped_view_bytes,
        }
    }

    /// Returns the number of resources the bundle has not released.
    #[must_use]
    pub fn live_resource_count(&self) -> usize {
        self.state.borrow().order.live_count()
    }

    /// Runs one closure against the device backend for inspection.
    pub fn with_device<T>(&self, inspect: impl FnOnce(&D) -> T) -> T {
        inspect(&self.state.borrow().device)
    }

    /// Runs one closure that configures the device backend.
    ///
    /// A driver is configured by whoever built it, not by the bundle. This is
    /// how a conformance host arms a backend before handing the table over.
    pub fn with_device_mut<T>(&self, configure: impl FnOnce(&mut D) -> T) -> T {
        configure(&mut self.state.borrow_mut().device)
    }

    /// Copies one bounded device logits row for the conformance fixture.
    pub(crate) fn read_logits_view_row(
        &self,
        view: runtime_bundle_abi::OwnedLogitsViewV1,
        row: u32,
    ) -> HostServiceResultV1<Vec<u8>> {
        let raw = view.raw();
        if row >= raw.row_count {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let element_bytes = match raw.element_type {
            runtime_bundle_abi::BUNDLE_LOGITS_ELEMENT_V1_F32 => 4,
            runtime_bundle_abi::BUNDLE_LOGITS_ELEMENT_V1_BF16 => 2,
            _ => return Err(BundleStatusV1::Unsupported),
        };
        let byte_length = u64::from(raw.live_width) * element_bytes;
        let row_offset = raw
            .row_stride_elements
            .checked_mul(u64::from(row))
            .and_then(|offset| offset.checked_mul(element_bytes))
            .and_then(|offset| raw.buffer.offset.checked_add(offset))
            .ok_or(BundleStatusV1::InvalidArgument)?;
        let view_end = raw
            .buffer
            .offset
            .checked_add(raw.buffer.length)
            .ok_or(BundleStatusV1::InvalidArgument)?;
        if row_offset
            .checked_add(byte_length)
            .is_none_or(|end| end > view_end)
        {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let mut state = self.state.borrow_mut();
        let context_key = slot_key_v1(raw.context.generation(), raw.context.slot())?;
        state.refuse_during_capture(context_key)?;
        let context = state.contexts.get(context_key)?;
        if context.rank != raw.rank {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let context_object = context.object;
        let buffer_key = slot_key_v1(raw.buffer.buffer.generation(), raw.buffer.buffer.slot())?;
        let buffer = state.buffers.get(buffer_key)?;
        if buffer.origin.map(|origin| origin.context) != Some(context_key) {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let range = state.resolve_range(DeviceBufferSpanV1::new(
            raw.buffer.buffer,
            row_offset,
            byte_length,
        ))?;
        let stream = state
            .device
            .create_stream(context_object, HOST_STREAM_KIND_V1)?;
        let mut bytes = vec![0; byte_length as usize];
        let result = state
            .device
            .copy_device_to_host(stream, &mut bytes, range)
            .and_then(|()| state.device.synchronize_stream(stream));
        let released = state.device.release_stream(stream);
        result?;
        released?;
        Ok(bytes)
    }
}

#[cfg(all(target_os = "linux", feature = "linux-cuda-host"))]
impl crate::executor::DeviceLogitsSamplerV1
    for HostServicesV1Backend<crate::cuda::CudaHostDeviceV1>
{
    fn supports_device_results(&self) -> bool {
        true
    }

    fn drain_device_results(
        &self,
        context: BundleContextHandleV1,
        rank: u32,
    ) -> HostServiceResultV1<()> {
        let mut state = self.state.borrow_mut();
        let record = state
            .contexts
            .get(slot_key_v1(context.generation(), context.slot())?)?;
        if record.rank != rank {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let object = record.object;
        state.device.drain_device_results(object)
    }

    fn retain_device_drafts(
        &self,
        drafts: runtime_bundle_abi::device_results::BundleDeviceDraftsV1,
    ) -> HostServiceResultV1<Box<dyn std::any::Any>> {
        drafts.validate()?;
        let state = self.state.borrow();
        if !state.captures.is_empty() {
            return Err(BundleStatusV1::InvalidState);
        }
        let view = drafts.proposals;
        let context_key = slot_key_v1(view.context.generation(), view.context.slot())?;
        if state.contexts.get(context_key)?.rank != view.rank {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let ready = state
            .events
            .get(slot_key_v1(view.ready.generation(), view.ready.slot())?)?;
        if ready.context != context_key || !ready.recorded {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let mut objects = Vec::new();
        for buffer in [view.buffer, drafts.confidence] {
            if buffer.buffer.raw == 0 {
                continue;
            }
            let record = state.buffers.get(slot_key_v1(
                buffer.buffer.generation(),
                buffer.buffer.slot(),
            )?)?;
            if record.origin.map(|origin| origin.context) != Some(context_key) {
                return Err(BundleStatusV1::InvalidArgument);
            }
            let range = state.resolve_range(DeviceBufferSpanV1::new(
                buffer.buffer,
                buffer.offset,
                buffer.length,
            ))?;
            objects.push(range.object);
        }
        state.device.retain_result_buffers(&objects)
    }

    fn cancel_device_readback(
        &self,
        view: runtime_bundle_abi::device_results::BundleReadyDeviceViewV1,
        ticket: u64,
    ) -> HostServiceResultV1<()> {
        let mut state = self.state.borrow_mut();
        let context = state
            .contexts
            .get(slot_key_v1(view.context.generation(), view.context.slot())?)?;
        if context.rank != view.rank {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let object = context.object;
        state.device.cancel_device_readback(object, ticket)
    }

    fn issue_ready_logits(
        &self,
        view: runtime_bundle_abi::device_results::BundleReadyLogitsViewV1,
        items: &[&runtime_engine::ModelBatchItem],
        dispatch: crate::executor::SamplingDispatch,
    ) -> HostServiceResultV1<u64> {
        view.validate()
            .map_err(|_| BundleStatusV1::InvalidArgument)?;
        let mut state = self.state.borrow_mut();
        if !state.captures.is_empty() {
            return Err(BundleStatusV1::InvalidState);
        }
        let (context, range) = state.resolve_logits_view(view.logits)?;
        let context_key =
            slot_key_v1(view.logits.context.generation(), view.logits.context.slot())?;
        let ready = state
            .events
            .get(slot_key_v1(view.ready.generation(), view.ready.slot())?)?;
        if ready.context != context_key || !ready.recorded {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let ready_object = ready.object;
        state.device.issue_ready_logits(
            context,
            range.object,
            ready_object,
            view.logits,
            items,
            dispatch,
        )
    }

    fn sample_device_logits(
        &self,
        request: runtime_bundle_abi::device_results::BundleDeviceSamplingRequestV1,
        items: &[&runtime_engine::ModelBatchItem],
        dispatch: crate::executor::SamplingDispatch,
    ) -> HostServiceResultV1<()> {
        request
            .validate()
            .map_err(|_| BundleStatusV1::InvalidArgument)?;
        let mut state = self.state.borrow_mut();
        // Capture may have joined a stream through an event on another device.
        // No new private-stream consumer may join any open service capture.
        if !state.captures.is_empty() {
            return Err(BundleStatusV1::InvalidState);
        }
        let raw = request.logits.logits;
        let context_key = slot_key_v1(raw.context.generation(), raw.context.slot())?;
        let (context, logits) = state.resolve_logits_view(raw)?;
        let output = state.buffers.get(slot_key_v1(
            request.output.buffer.generation(),
            request.output.buffer.slot(),
        )?)?;
        let ready = state.events.get(slot_key_v1(
            request.logits.ready.generation(),
            request.logits.ready.slot(),
        )?)?;
        let done = state
            .events
            .get(slot_key_v1(request.done.generation(), request.done.slot())?)?;
        if output.origin.map(|origin| origin.context) != Some(context_key)
            || ready.context != context_key
            || !ready.recorded
            || done.context != context_key
        {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let ready_object = ready.object;
        let done_object = done.object;
        let output = state.resolve_range(DeviceBufferSpanV1::new(
            request.output.buffer,
            request.output.offset,
            request.output.length,
        ))?;
        if logits.object == output.object {
            return Err(BundleStatusV1::InvalidArgument);
        }
        state.device.sample_device_logits(
            context,
            [logits.object, output.object, ready_object, done_object],
            request,
            items,
            dispatch,
        )?;
        state
            .events
            .get_mut(slot_key_v1(request.done.generation(), request.done.slot())?)?
            .recorded = true;
        Ok(())
    }

    fn enqueue_device_readback(
        &self,
        view: runtime_bundle_abi::device_results::BundleReadyDeviceViewV1,
    ) -> HostServiceResultV1<u64> {
        view.validate(1)?;
        let mut state = self.state.borrow_mut();
        if !state.captures.is_empty() {
            return Err(BundleStatusV1::InvalidState);
        }
        let context_key = slot_key_v1(view.context.generation(), view.context.slot())?;
        let context = state.contexts.get(context_key)?;
        if context.rank != view.rank {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let context_object = context.object;
        let buffer = state.buffers.get(slot_key_v1(
            view.buffer.buffer.generation(),
            view.buffer.buffer.slot(),
        )?)?;
        let ready = state
            .events
            .get(slot_key_v1(view.ready.generation(), view.ready.slot())?)?;
        if buffer.origin.map(|origin| origin.context) != Some(context_key)
            || ready.context != context_key
            || !ready.recorded
        {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let ready_object = ready.object;
        let range = state.resolve_range(DeviceBufferSpanV1::new(
            view.buffer.buffer,
            view.buffer.offset,
            view.buffer.length,
        ))?;
        state
            .device
            .enqueue_device_readback(context_object, range, ready_object)
    }

    fn collect_device_readback(
        &self,
        view: runtime_bundle_abi::device_results::BundleReadyDeviceViewV1,
        ticket: u64,
    ) -> HostServiceResultV1<Vec<u8>> {
        crate::overlap::device_wait();
        let _profile = runtime_engine::profile::PhaseTimer::new(
            runtime_engine::profile::StepPhase::DeviceWait,
        );
        let mut state = self.state.borrow_mut();
        let context = state
            .contexts
            .get(slot_key_v1(view.context.generation(), view.context.slot())?)?;
        if context.rank != view.rank {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let context_object = context.object;
        state.device.collect_device_readback(context_object, ticket)
    }

    fn sample_view_with_dispatch(
        &self,
        view: runtime_bundle_abi::OwnedLogitsViewV1,
        items: &[&runtime_engine::ModelBatchItem],
        dispatch: crate::executor::SamplingDispatch,
    ) -> HostServiceResultV1<Vec<runtime_engine::SampledRow>> {
        let raw = view.raw();
        let mut state = self.state.borrow_mut();
        let (context_object, range) = state.resolve_logits_view(raw)?;
        state
            .device
            .sample_logits_view(context_object, range, raw, items, Some(dispatch))
    }

    fn issue_view_with_dispatch(
        &self,
        view: runtime_bundle_abi::OwnedLogitsViewV1,
        items: &[&runtime_engine::ModelBatchItem],
        dispatch: crate::executor::SamplingDispatch,
    ) -> HostServiceResultV1<bool> {
        let raw = view.raw();
        let mut state = self.state.borrow_mut();
        let (context_object, range) = state.resolve_logits_view(raw)?;
        state
            .device
            .issue_logits_view(context_object, range, raw, items, Some(dispatch))?;
        Ok(true)
    }

    fn score_view_with_dispatch(
        &self,
        view: runtime_bundle_abi::OwnedLogitsViewV1,
        rows: &[crate::executor::DevicePromptRowV1<'_>],
        dispatch: crate::executor::SamplingDispatch,
    ) -> HostServiceResultV1<(
        Vec<runtime_engine::SampledRow>,
        Vec<runtime_engine::SampledRow>,
    )> {
        let raw = view.raw();
        let mut state = self.state.borrow_mut();
        let (context_object, range) = state.resolve_logits_view(raw)?;
        state
            .device
            .score_logits_view(context_object, range, raw, rows, Some(dispatch))
    }

    fn score_view(
        &self,
        view: runtime_bundle_abi::OwnedLogitsViewV1,
        rows: &[crate::executor::DevicePromptRowV1<'_>],
    ) -> HostServiceResultV1<(
        Vec<runtime_engine::SampledRow>,
        Vec<runtime_engine::SampledRow>,
    )> {
        let raw = view.raw();
        let mut state = self.state.borrow_mut();
        let (context_object, range) = state.resolve_logits_view(raw)?;
        state
            .device
            .score_logits_view(context_object, range, raw, rows, None)
    }

    fn sample_view(
        &self,
        view: runtime_bundle_abi::OwnedLogitsViewV1,
        items: &[&runtime_engine::ModelBatchItem],
    ) -> HostServiceResultV1<Vec<runtime_engine::SampledRow>> {
        let raw = view.raw();
        let mut state = self.state.borrow_mut();
        let (context_object, range) = state.resolve_logits_view(raw)?;
        state
            .device
            .sample_logits_view(context_object, range, raw, items, None)
    }

    fn issue_view(
        &self,
        view: runtime_bundle_abi::OwnedLogitsViewV1,
        items: &[&runtime_engine::ModelBatchItem],
    ) -> HostServiceResultV1<bool> {
        let raw = view.raw();
        let mut state = self.state.borrow_mut();
        let (context_object, range) = state.resolve_logits_view(raw)?;
        state
            .device
            .issue_logits_view(context_object, range, raw, items, None)?;
        Ok(true)
    }

    fn collect_view(
        &self,
        view: runtime_bundle_abi::OwnedLogitsViewV1,
        items: &[&runtime_engine::ModelBatchItem],
    ) -> HostServiceResultV1<Vec<runtime_engine::SampledRow>> {
        let raw = view.raw();
        let mut state = self.state.borrow_mut();
        let (context_object, _) = state.resolve_logits_view(raw)?;
        state.device.collect_logits_view(context_object, items)
    }
}

/// Reads the slot key one opaque ABI handle carries.
macro_rules! key_of_v1 {
    ($handle:expr) => {
        slot_key_v1($handle.generation(), $handle.slot())?
    };
}

/// Releases one driver-owned resource without ever losing track of it.
///
/// The reverse-order check and the driver call both run before any bookkeeping
/// changes, so a driver that refuses the release leaves the handle live, the
/// order entry live, and the release retryable. Only a driver call that
/// succeeded pops the order entry and frees the slot.
macro_rules! release_driver_resource_v1 {
    ($state:expr, $table:ident, $key:expr, $driver_release:ident) => {{
        let key = $key;
        let (order, object) = {
            let record = $state.$table.get(key)?;
            (record.order, record.object)
        };
        $state.order.preflight(order)?;
        $state.device.$driver_release(object)?;
        $state.order.release(order)?;
        $state.$table.remove(key).map(|_| ())
    }};
}

/// Releases one host-tracked resource whose finalization cannot fail.
macro_rules! release_tracked_resource_v1 {
    ($state:expr, $table:ident, $key:expr) => {{
        let key = $key;
        let order = $state.$table.get(key)?.order;
        $state.order.preflight(order)?;
        $state.$table.remove(key)?;
        $state.order.release(order)
    }};
}

impl<D: HostDeviceBackendV1> HostStateV1<D> {
    #[cfg(test)]
    fn check_probe_fault(
        &mut self,
        fault: HostDeviceProbeServiceFaultV1,
    ) -> HostServiceResultV1<()> {
        if self.probe_fault == Some(fault) {
            self.probe_fault = None;
            self.probe_fault_observed = true;
            Err(BundleStatusV1::HostError)
        } else {
            Ok(())
        }
    }

    /// Whether one context is named by a peer lease this instance still holds.
    ///
    /// A context is named by a context edge at either end, by a pool grant it
    /// is the peer of, and by a pool grant made through its own pool.
    fn context_device(&self, context: SlotKeyV1) -> HostServiceResultV1<i32> {
        self.config
            .topology
            .device_ordinal(self.contexts.get(context)?.rank)
            .ok_or(BundleStatusV1::InvalidArgument)
    }

    fn context_holds_peer_lease(&self, context: SlotKeyV1) -> bool {
        let Ok(device) = self.context_device(context) else {
            return true;
        };
        self.context_peer_leases
            .keys()
            .any(|edge| edge.owner == device || edge.peer == device)
            || self
                .pool_peer_leases
                .keys()
                .any(|edge| edge.peer == device || edge.pool == device)
    }

    /// A default pool is shared by every logical rank on its device.
    fn pool_holds_peer_lease(&self, pool: SlotKeyV1) -> bool {
        let Ok(record) = self.pools.get(pool) else {
            return true;
        };
        let Ok(device) = self.context_device(record.context) else {
            return true;
        };
        self.pool_peer_leases.keys().any(|edge| edge.pool == device)
    }

    /// Takes one more lease over a directed context peer edge.
    ///
    /// The first lease is the only one that touches the device, and it enables
    /// the edge only if the driver reports it was not enabled already. Every
    /// later lease is bookkeeping. A refused enable changes nothing at all, so
    /// the acquire is retryable exactly as it was.
    /// Whether work in one context may name one allocation's address.
    ///
    /// An allocation of the consumer's own context always may. Any other needs
    /// both halves of what makes a peer address usable, and they are separate
    /// facts about separate objects: the directed context lease is the
    /// consumer's right to reach that context at all, and the pool grant is the
    /// owning pool's consent to be reached from the consumer. Neither implies
    /// the other, and an address missing either is one the device would fault
    /// on, so it is refused before anything is ordered rather than after.
    ///
    /// A host-owned allocation, from the KV or state ranks, carries no pool or
    /// context handle this instance can name, so it can satisfy neither half
    /// and is refused rather than guessed at.
    fn consumer_may_reach(
        &self,
        consumer_context: SlotKeyV1,
        buffer: SlotKeyV1,
    ) -> HostServiceResultV1<()> {
        let origin = self
            .buffers
            .get(buffer)?
            .origin
            .ok_or(BundleStatusV1::InvalidArgument)?;
        let consumer = self.context_device(consumer_context)?;
        let origin_device = self.context_device(origin.context)?;
        if origin_device == consumer {
            return Ok(());
        }
        let leased = self
            .context_peer_leases
            .contains_key(&ContextPeerEdgeKeyV1 {
                owner: consumer,
                peer: origin_device,
            });
        let granted = self.pool_peer_leases.contains_key(&PoolPeerEdgeKeyV1 {
            pool: origin_device,
            peer: consumer,
        });
        if leased && granted {
            Ok(())
        } else {
            Err(BundleStatusV1::InvalidState)
        }
    }

    /// Checks every by-value address slot against one consumer context.
    fn require_reachable_slots(
        &self,
        consumer_context: SlotKeyV1,
        slots: &[Option<DeviceBufferSpanV1>],
    ) -> HostServiceResultV1<()> {
        for slot in slots.iter().flatten() {
            let buffer = slot_key_v1(slot.buffer.generation(), slot.buffer.slot())?;
            self.consumer_may_reach(consumer_context, buffer)?;
        }
        Ok(())
    }

    fn acquire_context_peer_lease(
        &mut self,
        edge: ContextPeerEdgeKeyV1,
        owner: DeviceObjectV1,
        peer: DeviceObjectV1,
    ) -> HostServiceResultV1<()> {
        if let Some(lease) = self.context_peer_leases.get_mut(&edge) {
            lease.users = lease
                .users
                .checked_add(1)
                .ok_or(BundleStatusV1::ResourceExhausted)?;
            return Ok(());
        }
        if edge.owner == edge.peer {
            self.context_peer_leases.insert(
                edge,
                PeerLeaseV1 {
                    users: 1,
                    host_acquired: false,
                    object: None,
                    pool_peer: None,
                },
            );
            return Ok(());
        }
        let acquisition = self
            .device
            .acquire_peer_access(owner, peer)
            .inspect_err(|status| {
                eprintln!("host acquire_peer_access: {status:?}");
            })?;
        self.context_peer_leases.insert(
            edge,
            PeerLeaseV1 {
                users: 1,
                host_acquired: acquisition.host_acquired,
                object: Some(acquisition.object),
                pool_peer: None,
            },
        );
        Ok(())
    }

    /// Gives one lease over a directed context peer edge back.
    ///
    /// Only the last one reaches the device, and only when this host enabled
    /// the edge itself. A driver that refuses the disable leaves the lease
    /// exactly as it was, so the release is retryable and the edge is never
    /// forgotten. A release with no lease behind it is refused rather than
    /// silently accepted.
    fn release_context_peer_lease(
        &mut self,
        edge: ContextPeerEdgeKeyV1,
    ) -> HostServiceResultV1<()> {
        let lease = *self
            .context_peer_leases
            .get(&edge)
            .ok_or(BundleStatusV1::InvalidState)?;
        if lease.users > 1 {
            self.context_peer_leases
                .entry(edge)
                .and_modify(|held| held.users -= 1);
            return Ok(());
        }
        if let Some(object) = lease.object {
            self.device
                .release_peer_access(object, lease.host_acquired)?;
        }
        self.context_peer_leases.remove(&edge);
        Ok(())
    }

    /// Takes one more lease over one pool's grant to one peer context.
    ///
    /// The first lease asks the device whether the grant is already there, and
    /// enables it only if it is not. A grant this host found belongs to whoever
    /// made it and is never revoked here.
    fn acquire_pool_peer_lease(
        &mut self,
        edge: PoolPeerEdgeKeyV1,
        pool: DeviceObjectV1,
        peer: DeviceObjectV1,
    ) -> HostServiceResultV1<()> {
        if let Some(lease) = self.pool_peer_leases.get_mut(&edge) {
            lease.users = lease
                .users
                .checked_add(1)
                .ok_or(BundleStatusV1::ResourceExhausted)?;
            return Ok(());
        }
        let preexisting = edge.pool == edge.peer
            || self
                .device
                .memory_pool_peer_access(pool, peer)
                .inspect_err(|status| {
                    eprintln!("host memory_pool_peer_access: {status:?}");
                })?;
        if !preexisting {
            self.device
                .set_memory_pool_peer_access(pool, peer, true)
                .inspect_err(|status| {
                    eprintln!("host set_memory_pool_peer_access: {status:?}");
                })?;
        }
        self.pool_peer_leases.insert(
            edge,
            PeerLeaseV1 {
                users: 1,
                host_acquired: !preexisting,
                object: Some(pool),
                pool_peer: Some(peer),
            },
        );
        Ok(())
    }

    /// Gives one lease over one pool's peer grant back.
    ///
    /// Only the last one reaches the device, and only when this host enabled
    /// the grant itself. A refused disable leaves the lease standing and the
    /// release retryable; a release with no lease behind it is refused.
    fn release_pool_peer_lease(&mut self, edge: PoolPeerEdgeKeyV1) -> HostServiceResultV1<()> {
        let lease = *self
            .pool_peer_leases
            .get(&edge)
            .ok_or(BundleStatusV1::InvalidState)?;
        if lease.users > 1 {
            self.pool_peer_leases
                .entry(edge)
                .and_modify(|held| held.users -= 1);
            return Ok(());
        }
        if lease.host_acquired {
            let pool = lease.object.ok_or(BundleStatusV1::InvalidState)?;
            let peer = lease.pool_peer.ok_or(BundleStatusV1::InvalidState)?;
            self.device.set_memory_pool_peer_access(pool, peer, false)?;
        }
        self.pool_peer_leases.remove(&edge);
        Ok(())
    }

    fn resolve_range(&self, span: DeviceBufferSpanV1) -> HostServiceResultV1<DeviceRangeV1> {
        let key = slot_key_v1(span.buffer.generation(), span.buffer.slot())?;
        let record = self.buffers.get(key)?;
        let end = span
            .offset
            .checked_add(span.length)
            .ok_or(BundleStatusV1::InvalidArgument)?;
        if end > record.byte_length {
            return Err(BundleStatusV1::InvalidArgument);
        }
        Ok(DeviceRangeV1 {
            object: record.object,
            offset: span.offset,
            length: span.length,
        })
    }

    fn resolve_arguments(
        &self,
        arguments: &[KernelArgumentV1],
    ) -> HostServiceResultV1<Vec<DeviceArgumentV1>> {
        arguments
            .iter()
            .map(|argument| match argument {
                KernelArgumentV1::DeviceBuffer(span) => {
                    self.resolve_range(*span).map(DeviceArgumentV1::Range)
                }
                KernelArgumentV1::NullDevicePointer => Ok(DeviceArgumentV1::NullDevicePointer),
                KernelArgumentV1::ScalarU32(value) => Ok(DeviceArgumentV1::ScalarU32(*value)),
                KernelArgumentV1::ScalarU64(value) => Ok(DeviceArgumentV1::ScalarU64(*value)),
                KernelArgumentV1::OpaqueBytes(bytes) => {
                    Ok(DeviceArgumentV1::OpaqueBytes(bytes.clone()))
                }
                // A tensor map names one span like any other argument, so it is
                // resolved and bounds checked exactly the same way before the
                // backend is given anything to encode.
                KernelArgumentV1::TensorMap(map) => {
                    Ok(DeviceArgumentV1::TensorMap(DeviceTensorMapV1 {
                        range: self.resolve_range(map.buffer)?,
                        rank: map.rank,
                        element_type: map.element_type,
                        swizzle: map.swizzle,
                        interleave: map.interleave,
                        l2_promotion: map.l2_promotion,
                        oob_fill: map.oob_fill,
                        global_dims: map.global_dims,
                        global_strides_bytes: map.global_strides_bytes,
                        box_dims: map.box_dims,
                        element_strides: map.element_strides,
                    }))
                }
                KernelArgumentV1::ScalarI32(value) => Ok(DeviceArgumentV1::ScalarI32(*value)),
                // Each slot is resolved and bounds checked exactly as a
                // device-buffer argument is. Whether the launching context may
                // reach it is a separate question, answered where the launch
                // context is known.
                KernelArgumentV1::DevicePointerArray(slots) => {
                    Ok(DeviceArgumentV1::DevicePointerArray(
                        slots
                            .iter()
                            .map(|slot| slot.map(|span| self.resolve_range(span)).transpose())
                            .collect::<HostServiceResultV1<Vec<Option<DeviceRangeV1>>>>()?,
                    ))
                }
            })
            .collect()
    }

    fn stream_object(&self, stream: BundleStreamHandleV1) -> HostServiceResultV1<DeviceObjectV1> {
        let key = slot_key_v1(stream.generation(), stream.slot())?;
        Ok(self.streams.get(key)?.object)
    }

    /// The context object and buffer range of a logits view whose buffer
    /// belongs to the view's context and rank.
    #[cfg(all(target_os = "linux", feature = "linux-cuda-host"))]
    fn resolve_logits_view(
        &self,
        raw: runtime_bundle_abi::BundleLogitsViewV1,
    ) -> HostServiceResultV1<(DeviceObjectV1, DeviceRangeV1)> {
        let context_key = slot_key_v1(raw.context.generation(), raw.context.slot())?;
        self.refuse_during_capture(context_key)?;
        let context = self.contexts.get(context_key)?;
        if context.rank != raw.rank {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let context_object = context.object;
        let buffer_key = slot_key_v1(raw.buffer.buffer.generation(), raw.buffer.buffer.slot())?;
        let buffer = self.buffers.get(buffer_key)?;
        if buffer.origin.map(|origin| origin.context) != Some(context_key) {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let range = self.resolve_range(DeviceBufferSpanV1::new(
            raw.buffer.buffer,
            raw.buffer.offset,
            raw.buffer.length,
        ))?;
        Ok((context_object, range))
    }

    /// Free device bytes in `context`, when the device reports them.
    fn free_context_bytes(&self, context: SlotKeyV1) -> Option<u64> {
        let object = self.contexts.get(context).ok()?.object;
        self.device.free_device_memory(object).ok()
    }

    /// Refuses work in `context` while a capture is open on its device.
    fn refuse_during_capture(&self, context: SlotKeyV1) -> HostServiceResultV1<()> {
        if self.capture_on_device_of(context).is_some() {
            Err(BundleStatusV1::InvalidState)
        } else {
            Ok(())
        }
    }

    /// Forgets an ended capture: its record, its members and its events.
    fn end_capture_bookkeeping(&mut self, context: SlotKeyV1) {
        self.captures.remove(&context);
        self.capture_members.remove(&context);
        self.capture_events.retain(|_, capture| *capture != context);
    }

    /// The open capture on `context`'s device, if any: the capturing
    /// context's key.
    fn capture_on_device_of(&self, context: SlotKeyV1) -> Option<SlotKeyV1> {
        if self.captures.contains_key(&context) {
            return Some(context);
        }
        let device = |key: SlotKeyV1| {
            self.contexts
                .get(key)
                .ok()
                .and_then(|record| self.config.topology.device_ordinal(record.rank))
        };
        let ordinal = device(context)?;
        self.captures
            .keys()
            .copied()
            .find(|&capturing| device(capturing) == Some(ordinal))
    }

    /// Admits a submission to `stream` while a capture is open on its device,
    /// or refuses it. The capture records its capturing stream and the streams
    /// that joined it (the fork and join the bundles use to capture two ranks
    /// of a device in one graph): these may launch kernels and record events,
    /// and an event they record belongs to the capture. Any stream of the
    /// device may join by waiting on such an event. Everything else on the
    /// device, including waits on events from outside the capture, copies
    /// and graph launches, waits until the capture ends.
    fn admit_during_capture(
        &mut self,
        stream: SlotKeyV1,
        submission: CaptureSubmissionV1,
    ) -> HostServiceResultV1<()> {
        let context = self.streams.get(stream)?.context;
        let Some(capture) = self.capture_on_device_of(context) else {
            return Ok(());
        };
        let member = self
            .capture_members
            .get(&capture)
            .is_some_and(|members| members.contains(&stream));
        match submission {
            CaptureSubmissionV1::Launch | CaptureSubmissionV1::Record if member => Ok(()),
            CaptureSubmissionV1::Wait(event)
                if self.capture_events.get(&event) == Some(&capture) =>
            {
                self.capture_members
                    .entry(capture)
                    .or_default()
                    .insert(stream);
                Ok(())
            }
            _ => Err(BundleStatusV1::InvalidState),
        }
    }

    fn context_object(
        &self,
        context: BundleContextHandleV1,
    ) -> HostServiceResultV1<DeviceObjectV1> {
        let key = slot_key_v1(context.generation(), context.slot())?;
        Ok(self.contexts.get(key)?.object)
    }

    /// Allocates one buffer and refuses it unless it meets the alignment.
    ///
    /// An allocator's own guarantee is not the alignment the caller asked for,
    /// so the allocation this returns is checked and released rather than
    /// handed on when the two disagree.
    fn allocate_aligned_object(
        &mut self,
        pool: DeviceObjectV1,
        stream: Option<DeviceObjectV1>,
        byte_length: u64,
        alignment: u64,
    ) -> HostServiceResultV1<DeviceObjectV1> {
        let object = self
            .device
            .allocate_device_buffer(pool, stream, byte_length, alignment)
            .inspect_err(|status| {
                eprintln!(
                    "host allocate_device_buffer: {byte_length} bytes, pool {}, stream {:?}: {status:?}",
                    pool.get(), stream.map(DeviceObjectV1::get)
                );
            })?;
        match self
            .device
            .allocation_satisfies_alignment(object, alignment)
        {
            Ok(true) => Ok(object),
            Ok(false) => {
                eprintln!(
                    "host allocate_device_buffer: {byte_length} bytes, pool {}: the allocation \
                     does not meet the requested {alignment}-byte alignment",
                    pool.get()
                );
                let _ = self.device.release_device_buffer(object);
                Err(BundleStatusV1::Unsupported)
            }
            Err(status) => {
                let _ = self.device.release_device_buffer(object);
                Err(status)
            }
        }
    }

    /// Allocates one host-owned buffer that its owner releases, not the bundle.
    fn allocate_host_buffer(
        &mut self,
        pool: DeviceObjectV1,
        origin: Option<BufferOriginV1>,
        stream: Option<DeviceObjectV1>,
        byte_length: u64,
        alignment: u64,
    ) -> HostServiceResultV1<SlotKeyV1> {
        let object = self.allocate_aligned_object(pool, stream, byte_length, alignment)?;
        match self.buffers.insert(DeviceBufferRecordV1 {
            object,
            byte_length,
            owner: BufferOwnerV1::Host,
            origin,
        }) {
            Ok(key) => Ok(key),
            Err(status) => {
                let _ = self.device.release_device_buffer(object);
                Err(status)
            }
        }
    }

    /// Allocates and fills one host-owned mapped host buffer on `stream`'s
    /// context.
    fn allocate_mapped_host_buffer(
        &mut self,
        stream: DeviceObjectV1,
        origin: Option<BufferOriginV1>,
        byte_length: u64,
        alignment: u64,
        fill: &mut dyn FnMut(&mut [u8]) -> bool,
    ) -> HostServiceResultV1<SlotKeyV1> {
        let object = self
            .device
            .allocate_mapped_host_buffer(stream, byte_length, fill)
            .inspect_err(|status| {
                eprintln!(
                    "host allocate_mapped_host_buffer: {byte_length} bytes: {status:?}; \
                     page-locked host memory may be limited (docs/deploy-common.md)"
                );
            })?;
        // The same rule as a device allocation: the base must carry the
        // alignment its views were checked against.
        match self
            .device
            .allocation_satisfies_alignment(object, alignment)
        {
            Ok(true) => {}
            Ok(false) => {
                let _ = self.device.release_device_buffer(object);
                return Err(BundleStatusV1::Unsupported);
            }
            Err(status) => {
                let _ = self.device.release_device_buffer(object);
                return Err(status);
            }
        }
        match self.buffers.insert(DeviceBufferRecordV1 {
            object,
            byte_length,
            owner: BufferOwnerV1::Host,
            origin,
        }) {
            Ok(key) => Ok(key),
            Err(status) => {
                let _ = self.device.release_device_buffer(object);
                Err(status)
            }
        }
    }

    /// Releases one host-owned buffer, keeping its slot until the driver agrees.
    fn release_host_buffer(&mut self, key: SlotKeyV1) -> HostServiceResultV1<()> {
        let object = self.buffers.get(key)?.object;
        self.device.release_device_buffer(object)?;
        self.buffers.remove(key).map(|_| ())
    }

    /// Creates the host's own per-rank context, pool, and stream on first use.
    fn acquire_host_ranks(&mut self) -> HostServiceResultV1<()> {
        if self.host_ranks.is_empty() {
            let rank_count = self.config.topology.rank_count();
            for rank in 0..rank_count {
                let ordinal = self
                    .config
                    .topology
                    .device_ordinal(rank)
                    .ok_or(BundleStatusV1::InvalidState)?;
                let built = self.build_host_rank(rank, ordinal);
                match built {
                    Ok(host_rank) => self.host_ranks.push(host_rank),
                    Err(status) => {
                        self.release_host_ranks();
                        return Err(status);
                    }
                }
            }
        }
        self.host_rank_users += 1;
        Ok(())
    }

    fn build_host_rank(&mut self, rank: u32, ordinal: i32) -> HostServiceResultV1<HostRankV1> {
        let (context, context_owned) = match self.context_by_rank.get(&rank) {
            Some(key) => (self.contexts.get(*key)?.object, false),
            None => (self.device.create_context(rank, ordinal)?, true),
        };
        let selected_pool = self
            .context_by_rank
            .get(&rank)
            .and_then(|context_key| self.pool_by_context.get(context_key))
            .map(|pool_key| self.pools.get(*pool_key).map(|record| record.object))
            .transpose()?;
        let (pool, pool_owned) = if let Some(pool) = selected_pool {
            (pool, false)
        } else {
            match self.device.create_memory_pool(context) {
                Ok(pool) => (pool, true),
                Err(status) => {
                    if context_owned {
                        let _ = self.device.release_context(context);
                    }
                    return Err(status);
                }
            }
        };
        match self.device.create_stream(context, HOST_STREAM_KIND_V1) {
            Ok(stream) => Ok(HostRankV1 {
                context,
                context_owned,
                pool,
                pool_owned,
                stream,
            }),
            Err(status) => {
                if pool_owned {
                    let _ = self.device.release_memory_pool(pool);
                }
                if context_owned {
                    let _ = self.device.release_context(context);
                }
                Err(status)
            }
        }
    }

    fn release_host_ranks(&mut self) {
        for host_rank in std::mem::take(&mut self.host_ranks).into_iter().rev() {
            let _ = self.device.release_stream(host_rank.stream);
            if host_rank.pool_owned {
                let _ = self.device.release_memory_pool(host_rank.pool);
            }
            if host_rank.context_owned {
                let _ = self.device.release_context(host_rank.context);
            }
        }
    }

    fn drop_host_rank_user(&mut self) {
        self.host_rank_users = self.host_rank_users.saturating_sub(1);
        if self.host_rank_users == 0 {
            self.release_host_ranks();
        }
    }

    /// Applies one prepared append's declared page work to every rank buffer.
    ///
    /// Each rank holds the same page geometry, so the same offsets apply to
    /// every rank pool. Zeroing runs first, because a copy lands inside a page
    /// the same append newly reserved.
    fn execute_kv_page_work_v1(
        &mut self,
        kv_key: SlotKeyV1,
        pages: &KvPageWorkV1,
    ) -> HostServiceResultV1<()> {
        if pages.zero.is_empty() && pages.copies.is_empty() {
            return Ok(());
        }
        // Page work runs on every rank's stream.
        if !self.captures.is_empty() {
            return Err(BundleStatusV1::InvalidState);
        }
        let record = self.kv.get(kv_key)?;
        let plan = record.plan;
        let rank_buffers = record.rank_buffers.clone();
        let page_bytes = plan.bytes_per_page();
        let bytes_per_token = plan.bytes_per_token() as u64;
        let segments = plan.segments_per_page();
        for (rank, buffer_key) in rank_buffers.iter().enumerate() {
            let stream = self
                .host_ranks
                .get(rank)
                .ok_or(BundleStatusV1::InvalidState)?
                .stream;
            let object = self.buffers.get(*buffer_key)?.object;
            for page in &pages.zero {
                let offset = plan
                    .page_offset(*page)
                    .map_err(|_| BundleStatusV1::InvalidState)?;
                self.device.zero_device_buffer(
                    stream,
                    DeviceRangeV1 {
                        object,
                        offset,
                        length: page_bytes,
                    },
                )?;
            }
            for copy in &pages.copies {
                let length = (copy.initialized_tokens as u64)
                    .checked_mul(bytes_per_token)
                    .ok_or(BundleStatusV1::InvalidState)?;
                if length == 0 {
                    continue;
                }
                for segment in 0..segments {
                    let source = plan
                        .row_offset(copy.source, segment, 0)
                        .map_err(|_| BundleStatusV1::InvalidState)?;
                    let destination = plan
                        .row_offset(copy.destination, segment, 0)
                        .map_err(|_| BundleStatusV1::InvalidState)?;
                    self.device.copy_device_to_device(
                        stream,
                        DeviceRangeV1 {
                            object,
                            offset: destination,
                            length,
                        },
                        DeviceRangeV1 {
                            object,
                            offset: source,
                            length,
                        },
                    )?;
                }
            }
            self.device.synchronize_stream(stream)?;
        }
        Ok(())
    }

    /// Allocates one host-owned buffer per rank of the same byte length.
    fn allocate_rank_buffers(
        &mut self,
        bytes_per_rank: u64,
        alignment: u64,
    ) -> HostServiceResultV1<Vec<SlotKeyV1>> {
        let mut buffers = Vec::new();
        for index in 0..self.host_ranks.len() {
            let (pool, stream) = {
                let host_rank = &self.host_ranks[index];
                (host_rank.pool, host_rank.stream)
            };
            match self.allocate_host_buffer(pool, None, Some(stream), bytes_per_rank, alignment) {
                Ok(buffer) => buffers.push(buffer),
                Err(status) => {
                    for buffer in buffers.into_iter().rev() {
                        let _ = self.release_host_buffer(buffer);
                    }
                    return Err(status);
                }
            }
        }
        Ok(buffers)
    }
}

impl<D: HostDeviceBackendV1> HostServicesBackendV1 for HostServicesV1Backend<D> {
    fn query_speculation(&self) -> HostServiceResultV1<runtime_bundle_abi::BundleRawSpeculationV1> {
        Ok(self.state.borrow().config.speculation)
    }

    fn query_batch_limits(
        &self,
    ) -> HostServiceResultV1<runtime_bundle_abi::BundleRawBatchLimitsV1> {
        Ok(self.state.borrow().config.batch_limits)
    }

    fn acquire_serving_kv(&self, layout_id: &[u8]) -> HostServiceResultV1<u64> {
        const RESERVE_BYTES: u64 = 1024 * 1024 * 1024;
        let mut state = self.state.borrow_mut();
        let policy = state.config.serving_kv.ok_or(BundleStatusV1::Unsupported)?;
        if policy.fraction_millionths == 0 || policy.fraction_millionths > 1_000_000 {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let contract = state
            .config
            .kv
            .as_ref()
            .ok_or(BundleStatusV1::Unsupported)?;
        if layout_id != contract.layout_id || !state.captures.is_empty() {
            return Err(BundleStatusV1::InvalidArgument);
        }
        if let Some(pool) = &state.serving_kv {
            return Ok(pool.plan.page_capacity());
        }
        let layout = contract.layout;
        let mut ranks = Vec::new();
        let mut free = u64::MAX;
        for rank in 0..layout.rank_count() as u32 {
            let context_key = *state
                .context_by_rank
                .get(&rank)
                .ok_or(BundleStatusV1::InvalidState)?;
            let pool_key = *state
                .pool_by_context
                .get(&context_key)
                .ok_or(BundleStatusV1::InvalidState)?;
            let context = state.contexts.get(context_key)?.object;
            let pool = state.pools.get(pool_key)?.object;
            if let Some(vocabulary) = policy.sampling_vocabulary {
                state
                    .device
                    .reserve_sampling_workspace(context, vocabulary)?;
            }
            let physical_free = state.device.free_device_memory(context)?;
            let Some(available) = physical_free
                .checked_sub(RESERVE_BYTES)
                .and_then(|bytes| bytes.checked_sub(policy.state_reserved_bytes))
            else {
                eprintln!(
                    "host acquire_serving_kv: rank {rank} has {physical_free} device bytes free, \
                     less than the {RESERVE_BYTES}-byte reserve and {} bytes of request state; \
                     no serving KV pool",
                    policy.state_reserved_bytes
                );
                return Err(BundleStatusV1::ResourceExhausted);
            };
            free = free.min(available);
            ranks.push((context_key, pool_key, context, pool));
        }
        let available = free;
        let budget =
            (u128::from(available) * u128::from(policy.fraction_millionths) / 1_000_000) as u64;
        let mut capacity = budget / layout.page_bytes();
        capacity = capacity.min(runtime_kv::MAX_BLOCK_CAPACITY as u64);
        if let Some(maximum) = policy.maximum_pages {
            capacity = capacity.min(maximum);
        }
        let plan = PagedKvPlanV1::validate(layout, capacity, 1).map_err(|_| {
            eprintln!(
                "host acquire_serving_kv: {available} device bytes available per rank, \
                 {budget} of them for KV, hold {capacity} pages of {} bytes; no serving KV pool",
                layout.page_bytes()
            );
            BundleStatusV1::ResourceExhausted
        })?;
        let mut allocated = Vec::new();
        for (context_key, pool_key, context, pool) in ranks {
            let result = (|| {
                let stream = state.device.create_stream(context, HOST_STREAM_KIND_V1)?;
                let origin = BufferOriginV1 {
                    context: context_key,
                    pool: pool_key,
                };
                match state.allocate_host_buffer(
                    pool,
                    Some(origin),
                    Some(stream),
                    plan.bytes_per_rank_pool(),
                    layout.alignment_bytes(),
                ) {
                    Ok(buffer) => Ok((buffer, stream)),
                    Err(error) => {
                        let _ = state.device.release_stream(stream);
                        Err(error)
                    }
                }
            })();
            match result {
                Ok(item) => allocated.push(item),
                Err(error) => {
                    for (buffer, stream) in allocated.into_iter().rev() {
                        let _ = state.release_host_buffer(buffer);
                        let _ = state.device.release_stream(stream);
                    }
                    return Err(error);
                }
            }
        }
        state.serving_kv = Some(ServingKvRecordV1 {
            plan,
            rank_buffers: allocated,
        });
        Ok(capacity)
    }

    fn resolve_serving_kv_pool(&self, rank: u32) -> HostServiceResultV1<DeviceBufferSpanV1> {
        let state = self.state.borrow();
        let pool = state
            .serving_kv
            .as_ref()
            .ok_or(BundleStatusV1::InvalidState)?;
        let (key, _) = *pool
            .rank_buffers
            .get(rank as usize)
            .ok_or(BundleStatusV1::InvalidArgument)?;
        Ok(DeviceBufferSpanV1::new(
            BundleDeviceBufferHandleV1::from_parts(key.generation, key.slot),
            0,
            pool.plan.bytes_per_rank_pool(),
        ))
    }

    fn log(&self, message: &[u8]) -> HostServiceResultV1<()> {
        self.state.borrow_mut().device.log(message)
    }

    fn select_context(
        &self,
        rank: u32,
        device_ordinal: i32,
    ) -> HostServiceResultV1<BundleContextHandleV1> {
        let mut state = self.state.borrow_mut();
        let expected = state
            .config
            .topology
            .device_ordinal(rank)
            .ok_or(BundleStatusV1::InvalidArgument)?;
        if expected != device_ordinal {
            return Err(BundleStatusV1::InvalidArgument);
        }
        // Keep one logical handle per rank for upload and sampler identity.
        if let Some(key) = state.context_by_rank.get(&rank).copied() {
            let record = state.contexts.get_mut(key)?;
            record.users = record
                .users
                .checked_add(1)
                .ok_or(BundleStatusV1::ResourceExhausted)?;
            return Ok(BundleContextHandleV1::from_parts(key.generation, key.slot));
        }
        let object = state.device.create_context(rank, device_ordinal)?;
        let order = match state.order.acquire() {
            Ok(order) => order,
            Err(status) => {
                let _ = state.device.release_context(object);
                return Err(status);
            }
        };
        match state.contexts.insert(ContextRecordV1 {
            object,
            rank,
            order,
            users: 1,
        }) {
            Ok(key) => {
                state.context_by_rank.insert(rank, key);
                #[cfg(test)]
                if let Err(status) =
                    state.check_probe_fault(HostDeviceProbeServiceFaultV1::ContextAcquire)
                {
                    state.context_by_rank.remove(&rank);
                    release_driver_resource_v1!(state, contexts, key, release_context)?;
                    return Err(status);
                }
                Ok(BundleContextHandleV1::from_parts(key.generation, key.slot))
            }
            Err(status) => {
                let _ = state.order.release(order);
                let _ = state.device.release_context(object);
                Err(status)
            }
        }
    }

    fn release_context(&self, context: BundleContextHandleV1) -> HostServiceResultV1<()> {
        let mut state = self.state.borrow_mut();
        let key = key_of_v1!(context);
        // A selection that is not the last one gives nothing back, so it is
        // neither ordered nor guarded: it only says this user is done.
        let (rank, users) = {
            let record = state.contexts.get(key)?;
            (record.rank, record.users)
        };
        if users > 1 {
            state.contexts.get_mut(key)?.users -= 1;
            return Ok(());
        }
        let object = state.contexts.get(key)?.object;
        if state
            .host_ranks
            .iter()
            .any(|host_rank| !host_rank.context_owned && host_rank.context == object)
        {
            return Err(BundleStatusV1::InvalidState);
        }
        // A context either end of a live peer lease, or one whose pool a grant
        // was made through, is still named by that lease and is not this
        // instance's to give back yet.
        if state.context_holds_peer_lease(key) {
            return Err(BundleStatusV1::InvalidState);
        }
        release_driver_resource_v1!(state, contexts, key, release_context)?;
        state.context_by_rank.remove(&rank);
        Ok(())
    }

    fn select_memory_pool(
        &self,
        context: BundleContextHandleV1,
    ) -> HostServiceResultV1<BundleMemoryPoolHandleV1> {
        let mut state = self.state.borrow_mut();
        let context_key = key_of_v1!(context);
        let context_object = state.contexts.get(context_key)?.object;
        // Repeated selection keeps one logical pool handle per rank context.
        if let Some(key) = state.pool_by_context.get(&context_key).copied() {
            let record = state.pools.get_mut(key)?;
            record.users = record
                .users
                .checked_add(1)
                .ok_or(BundleStatusV1::ResourceExhausted)?;
            return Ok(BundleMemoryPoolHandleV1::from_parts(
                key.generation,
                key.slot,
            ));
        }
        let object = state.device.create_memory_pool(context_object)?;
        let order = match state.order.acquire() {
            Ok(order) => order,
            Err(status) => {
                let _ = state.device.release_memory_pool(object);
                return Err(status);
            }
        };
        match state.pools.insert(MemoryPoolRecordV1 {
            object,
            context: context_key,
            order,
            users: 1,
        }) {
            Ok(key) => {
                state.pool_by_context.insert(context_key, key);
                Ok(BundleMemoryPoolHandleV1::from_parts(
                    key.generation,
                    key.slot,
                ))
            }
            Err(status) => {
                let _ = state.order.release(order);
                let _ = state.device.release_memory_pool(object);
                Err(status)
            }
        }
    }

    fn set_memory_pool_peer_access(
        &self,
        pool: BundleMemoryPoolHandleV1,
        peer_context: BundleContextHandleV1,
        enabled: bool,
    ) -> HostServiceResultV1<()> {
        let mut state = self.state.borrow_mut();
        let pool_key = key_of_v1!(pool);
        let peer_key = key_of_v1!(peer_context);
        let pool_object = state.pools.get(pool_key)?.object;
        let owner = state.pools.get(pool_key)?.context;
        if owner == peer_key {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let peer_object = state.contexts.get(peer_key)?.object;
        // The physical default pool and peer device identify the grant.
        let edge = PoolPeerEdgeKeyV1 {
            pool: state.context_device(owner)?,
            peer: state.context_device(peer_key)?,
        };
        if enabled {
            state.acquire_pool_peer_lease(edge, pool_object, peer_object)
        } else {
            state.release_pool_peer_lease(edge)
        }
    }

    fn release_memory_pool(&self, pool: BundleMemoryPoolHandleV1) -> HostServiceResultV1<()> {
        let mut state = self.state.borrow_mut();
        let key = key_of_v1!(pool);
        let (context, users) = {
            let record = state.pools.get(key)?;
            (record.context, record.users)
        };
        if users > 1 {
            state.pools.get_mut(key)?.users -= 1;
            return Ok(());
        }
        let object = state.pools.get(key)?.object;
        if state
            .host_ranks
            .iter()
            .any(|host_rank| !host_rank.pool_owned && host_rank.pool == object)
        {
            return Err(BundleStatusV1::InvalidState);
        }
        // A pool a lease was granted through outlives that lease: releasing it
        // first would leave the host holding a grant it could no longer name.
        if state.pool_holds_peer_lease(key) {
            return Err(BundleStatusV1::InvalidState);
        }
        release_driver_resource_v1!(state, pools, key, release_memory_pool)?;
        state.pool_by_context.remove(&context);
        Ok(())
    }

    fn allocate_device_buffer(
        &self,
        pool: BundleMemoryPoolHandleV1,
        stream: BundleStreamHandleV1,
        byte_length: u64,
        alignment: u64,
    ) -> HostServiceResultV1<BundleDeviceBufferHandleV1> {
        let mut state = self.state.borrow_mut();
        let pool_key = key_of_v1!(pool);
        let pool_record = state.pools.get(pool_key)?;
        state.refuse_during_capture(pool_record.context)?;
        let pool_object = pool_record.object;
        let pool_origin = BufferOriginV1 {
            context: pool_record.context,
            pool: pool_key,
        };
        let stream_object = if stream.is_valid() {
            Some(state.stream_object(stream)?)
        } else {
            None
        };
        let object =
            state.allocate_aligned_object(pool_object, stream_object, byte_length, alignment)?;
        let order = match state.order.acquire() {
            Ok(order) => order,
            Err(status) => {
                let _ = state.device.release_device_buffer(object);
                return Err(status);
            }
        };
        match state.buffers.insert(DeviceBufferRecordV1 {
            object,
            byte_length,
            owner: BufferOwnerV1::Bundle(order),
            origin: Some(pool_origin),
        }) {
            Ok(key) => Ok(BundleDeviceBufferHandleV1::from_parts(
                key.generation,
                key.slot,
            )),
            Err(status) => {
                let _ = state.order.release(order);
                let _ = state.device.release_device_buffer(object);
                Err(status)
            }
        }
    }

    fn release_device_buffer(&self, buffer: BundleDeviceBufferHandleV1) -> HostServiceResultV1<()> {
        let mut state = self.state.borrow_mut();
        let key = key_of_v1!(buffer);
        let record = state.buffers.get(key)?;
        if let Some(origin) = record.origin {
            state.refuse_during_capture(origin.context)?;
        }
        let BufferOwnerV1::Bundle(order) = record.owner else {
            // A rank arena or KV pool owns this allocation and releases it.
            return Err(BundleStatusV1::InvalidState);
        };
        let object = record.object;
        state.order.preflight(order)?;
        state.device.release_device_buffer(object)?;
        state.order.release(order)?;
        state.buffers.remove(key).map(|_| ())
    }

    fn open_compiled_image(
        &self,
        component_role: &[u8],
    ) -> HostServiceResultV1<BundleCompiledImageHandleV1> {
        let mut state = self.state.borrow_mut();
        let component = state
            .config
            .components
            .get(component_role)
            .ok_or(BundleStatusV1::NotFound)?
            .clone();
        let order = state.order.acquire()?;
        match state.images.insert(CompiledImageRecordV1 {
            storage: component.storage,
            description: component.description,
            order,
        }) {
            Ok(key) => Ok(BundleCompiledImageHandleV1::from_parts(
                key.generation,
                key.slot,
            )),
            Err(status) => {
                let _ = state.order.release(order);
                Err(status)
            }
        }
    }

    fn upload_rank_arena(
        &self,
        context: BundleContextHandleV1,
        pool: BundleMemoryPoolHandleV1,
        stream: BundleStreamHandleV1,
        image: BundleCompiledImageHandleV1,
        rank: u32,
    ) -> HostServiceResultV1<BundleRankArenaHandleV1> {
        let mut state = self.state.borrow_mut();
        let upload = RankArenaUploadV1 {
            context,
            pool,
            stream,
            image,
            rank,
        };
        let mut timing = DirectLoadTimingV1::new(&state, &[upload]);
        let prepared = match prepare_rank_upload_v1(&mut state, upload) {
            Ok(prepared) => prepared,
            Err(status) => {
                for (image, _) in &timing.images {
                    image.release_buffers();
                }
                return Err(status);
            }
        };
        timing.prepared();
        let started = std::time::Instant::now();
        let sparse = state.device.uses_sparse_probe_allocations();
        let result = if sparse {
            Ok(())
        } else {
            upload_prepared_rank_v1(&mut state.device, &prepared)
                .and_then(|()| state.device.synchronize_stream(prepared.stream))
                .and_then(|()| verify_direct_sources_v1(&prepared.storage))
        };
        let upload_seconds = started.elapsed().as_secs_f64();
        if result.is_err() {
            for (image, _) in &timing.images {
                image.release_buffers();
            }
        }
        timing.report(upload_seconds, 0.0);
        eprintln!(
            "startup weight_upload: {}",
            serde_json::json!({
                "seconds":upload_seconds, "status":if sparse {"sparse-probe"} else if result.is_ok() {"complete"} else {"failed"},
                "planned_bytes":prepared.byte_length(), "uploaded_bytes":result.as_ref().ok().map(|_| if sparse {0} else {prepared.byte_length()}),
            })
        );
        drop(state);
        if let Err(status) = result {
            let _ = self.release_rank_arena(prepared.arena);
            return Err(status);
        }
        Ok(prepared.arena)
    }

    fn upload_rank_arenas(
        &self,
        uploads: &[RankArenaUploadV1],
    ) -> HostServiceResultV1<Vec<BundleRankArenaHandleV1>> {
        let mut state = self.state.borrow_mut();
        let mut timing = DirectLoadTimingV1::new(&state, uploads);
        let mut prepared = Vec::with_capacity(uploads.len());
        let mut result = Ok(());
        for upload in uploads {
            match prepare_rank_upload_v1(&mut state, *upload) {
                Ok(rank) => prepared.push(rank),
                Err(status) => {
                    result = Err(status);
                    break;
                }
            }
        }
        timing.prepared();
        let started = std::time::Instant::now();
        if result.is_ok() && !state.device.uses_sparse_probe_allocations() {
            let mut readers: Vec<FileChunkReaderV1<'_>> = prepared
                .iter()
                .map(|rank| {
                    let storage = &rank.storage;
                    let segments = (rank.segments.len() > 1).then_some(&rank.segments);
                    Box::new(move |offset, chunk: &mut [u8]| match (storage, segments) {
                        (_, Some(segments)) => read_segments_v1(storage, segments, offset, chunk),
                        (ComponentStorageV1::Planned(_), None) => {
                            read_component_bytes_v1(storage, offset, chunk)
                        }
                        (ComponentStorageV1::Bytes(_), None) => false,
                    }) as FileChunkReaderV1<'_>
                })
                .collect();
            if prepared
                .iter()
                .all(|rank| matches!(rank.storage, ComponentStorageV1::Planned(_)))
            {
                let mut ranges: Vec<_> = prepared
                    .iter()
                    .zip(readers.iter_mut())
                    .map(|(rank, read)| PinnedFileUploadV1 {
                        stream: rank.stream,
                        destination: rank.destination,
                        source_offset: rank.source_offset(),
                        byte_length: rank.byte_length(),
                        read_chunk: read.as_mut(),
                    })
                    .collect();
                if let Some(uploaded) = state.device.upload_pinned_file_ranges(&mut ranges) {
                    result = uploaded;
                } else {
                    for rank in &prepared {
                        if let Err(status) = upload_prepared_rank_v1(&mut state.device, rank) {
                            result = Err(status);
                            break;
                        }
                    }
                }
            } else {
                for rank in &prepared {
                    if let Err(status) = upload_prepared_rank_v1(&mut state.device, rank) {
                        result = Err(status);
                        break;
                    }
                }
            }
        }
        let sparse = state.device.uses_sparse_probe_allocations();
        if result.is_ok() && !sparse {
            for rank in &prepared {
                if let Err(status) = state.device.synchronize_stream(rank.stream) {
                    result = Err(status);
                    break;
                }
            }
        }
        // The upload ends here; the qualification digests below are timed
        // apart, so startup lines never count them as load time.
        let upload_seconds = started.elapsed().as_secs_f64();
        let digesting = std::time::Instant::now();
        // Qualification only: the SHA-256 of every uploaded arena, read back
        // from the device, to compare load paths byte for byte.
        if result.is_ok()
            && !sparse
            && std::env::var_os(ARENA_DIGEST_ENV_V1).is_some_and(|value| value == "1")
        {
            for (rank, prepared) in prepared.iter().enumerate() {
                let started = std::time::Instant::now();
                let digest = device_digest_v1(
                    &mut state.device,
                    prepared.stream,
                    prepared.destination,
                    prepared.byte_length(),
                );
                eprintln!(
                    "startup arena_digest: {}",
                    serde_json::json!({
                        "rank": rank, "bytes": prepared.byte_length(),
                        "sha256": digest.as_ref().ok(), "error": digest.as_ref().err().map(|s| s.name()),
                        "seconds": started.elapsed().as_secs_f64(),
                    })
                );
            }
        }
        let digest_seconds = digesting.elapsed().as_secs_f64();
        // A direct load's checkpoint must still be the one its plans read.
        for (image, _) in &timing.images {
            image.release_buffers();
            if result.is_ok() && !sparse {
                result = verify_direct_sources_v1(&ComponentStorageV1::Planned(Arc::clone(image)));
            }
        }
        timing.report(upload_seconds, digest_seconds);
        let planned: u64 = prepared.iter().map(PreparedRankUploadV1::byte_length).sum();
        eprintln!(
            "startup weight_upload: {}",
            serde_json::json!({
                "seconds":upload_seconds, "digest_seconds":digest_seconds, "status":if sparse {"sparse-probe"} else if result.is_ok() {"complete"} else {"failed"},
                "planned_bytes":planned, "uploaded_bytes":result.as_ref().ok().map(|_| if sparse {0} else {planned}),
            })
        );
        let arenas: Vec<_> = prepared.iter().map(|rank| rank.arena).collect();
        drop(state);
        if let Err(status) = result {
            for arena in arenas.into_iter().rev() {
                let _ = self.release_rank_arena(arena);
            }
            return Err(status);
        }
        Ok(arenas)
    }

    fn release_rank_arena(&self, arena: BundleRankArenaHandleV1) -> HostServiceResultV1<()> {
        let mut state = self.state.borrow_mut();
        if !state.captures.is_empty() {
            return Err(BundleStatusV1::InvalidState);
        }
        let key = key_of_v1!(arena);
        let record = state.arenas.get(key)?;
        let (order, buffer, host_buffer) = (record.order, record.buffer, record.host_buffer);
        state.order.preflight(order)?;
        if let Some(host_buffer) = host_buffer {
            state.release_host_buffer(host_buffer)?;
            state.arenas.get_mut(key)?.host_buffer = None;
        }
        state.release_host_buffer(buffer)?;
        state.order.release(order)?;
        state.arenas.remove(key).map(|_| ())
    }

    fn resolve_compiled_image_view(
        &self,
        image: BundleCompiledImageHandleV1,
        arena: BundleRankArenaHandleV1,
        rank: u32,
        view_id: &[u8],
    ) -> HostServiceResultV1<CompiledImageSpanV1> {
        let state = self.state.borrow();
        let image_key = slot_key_v1(image.generation(), image.slot())?;
        let arena_key = slot_key_v1(arena.generation(), arena.slot())?;
        let record = state.arenas.get(arena_key)?;
        if record.image != image_key || record.rank != rank {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let buffer = state.buffers.get(record.buffer)?;
        let handle =
            BundleDeviceBufferHandleV1::from_parts(record.buffer.generation, record.buffer.slot);
        if view_id == COMPILED_IMAGE_ARENA_VIEW_V1 {
            return Ok(CompiledImageSpanV1::new(
                rank,
                DeviceBufferSpanV1::new(handle, 0, buffer.byte_length),
                record.alignment,
            ));
        }
        let view = state
            .images
            .get(image_key)?
            .description
            .as_ref()
            .ok_or(BundleStatusV1::Unsupported)?
            .views
            .get(view_id)
            .ok_or(BundleStatusV1::NotFound)?;
        let span = view
            .spans
            .get(&rank)
            .copied()
            .ok_or(BundleStatusV1::NotFound)?;
        // A mapped view has no device address.
        if view.mapped {
            return Err(BundleStatusV1::NotFound);
        }
        let (key, offset) = if view.host {
            let key = record.host_buffer.ok_or(BundleStatusV1::InvalidState)?;
            (
                key,
                record.layout.host_offset(span.offset, span.byte_length)?,
            )
        } else {
            (
                record.buffer,
                record.layout.device_offset(span.offset, span.byte_length)?,
            )
        };
        let end = offset
            .checked_add(span.byte_length)
            .ok_or(BundleStatusV1::InvalidArgument)?;
        if end > state.buffers.get(key)?.byte_length || !offset.is_multiple_of(view.alignment) {
            return Err(BundleStatusV1::InvalidArgument);
        }
        Ok(CompiledImageSpanV1::new(
            rank,
            DeviceBufferSpanV1::new(
                BundleDeviceBufferHandleV1::from_parts(key.generation, key.slot),
                offset,
                span.byte_length,
            ),
            view.alignment,
        ))
    }

    fn release_compiled_image(
        &self,
        image: BundleCompiledImageHandleV1,
    ) -> HostServiceResultV1<()> {
        let mut state = self.state.borrow_mut();
        release_tracked_resource_v1!(state, images, key_of_v1!(image))
    }

    fn create_mapped_gather(
        &self,
        producer: BundleStreamHandleV1,
        indices: DeviceBufferSpanV1,
        planes: &[runtime_bundle_abi::MappedGatherPlaneV1],
    ) -> HostServiceResultV1<runtime_bundle_abi::BundleMappedGatherHandleV1> {
        use crate::device::mapped_gather::{MappedGatherPlanV1, MappedGatherPlaneV1};
        let mut state = self.state.borrow_mut();
        let producer_key = key_of_v1!(producer);
        let context = state.streams.get(producer_key)?.context;
        state.refuse_during_capture(context)?;
        if state
            .buffers
            .get(key_of_v1!(indices.buffer))?
            .origin
            .map(|origin| origin.context)
            != Some(context)
        {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let mut streams = vec![producer_key];
        let resolved = planes
            .iter()
            .map(|plane| {
                let stream = key_of_v1!(plane.stream);
                let context = state.streams.get(stream)?.context;
                state.refuse_during_capture(context)?;
                if state
                    .buffers
                    .get(key_of_v1!(plane.destination.buffer))?
                    .origin
                    .map(|origin| origin.context)
                    != Some(context)
                    || plane.rank >= state.config.topology.rank_count()
                {
                    return Err(BundleStatusV1::InvalidArgument);
                }
                let ComponentStorageV1::Planned(planned) =
                    &state.images.get(key_of_v1!(plane.image))?.storage
                else {
                    return Err(BundleStatusV1::NotFound);
                };
                let mapping = Arc::clone(
                    planned
                        .mapped_view(plane.rank, &plane.view_id)
                        .ok_or(BundleStatusV1::NotFound)?,
                );
                streams.push(stream);
                Ok(MappedGatherPlaneV1 {
                    mapping,
                    row_bytes: usize::try_from(plane.row_bytes)
                        .map_err(|_| BundleStatusV1::InvalidArgument)?,
                    index_offset: usize::try_from(plane.index_offset)
                        .map_err(|_| BundleStatusV1::InvalidArgument)?,
                    stream: state.stream_object(plane.stream)?,
                    destination: state.resolve_range(plane.destination)?,
                })
            })
            .collect::<HostServiceResultV1<Vec<_>>>()?;
        let plan = MappedGatherPlanV1 {
            producer: state.stream_object(producer)?,
            indices: state.resolve_range(indices)?,
            planes: resolved,
        };
        plan.validate()?;
        crate::host_rows::resident::register(&plan.planes).map_err(|error| {
            eprintln!("host resident mapped rows: {error}");
            BundleStatusV1::InvalidArgument
        })?;
        let object = state.device.create_mapped_gather(plan)?;
        let order = match state.order.acquire() {
            Ok(order) => order,
            Err(status) => {
                let _ = state.device.release_mapped_gather(object);
                return Err(status);
            }
        };
        match state.mapped_gathers.insert(MappedGatherRecordV1 {
            object,
            streams,
            order,
        }) {
            Ok(key) => Ok(runtime_bundle_abi::BundleMappedGatherHandleV1::from_parts(
                key.generation,
                key.slot,
            )),
            Err(status) => {
                let _ = state.order.release(order);
                let _ = state.device.release_mapped_gather(object);
                Err(status)
            }
        }
    }

    fn enqueue_mapped_gather(
        &self,
        gather: runtime_bundle_abi::BundleMappedGatherHandleV1,
        rows: u64,
    ) -> HostServiceResultV1<()> {
        let mut state = self.state.borrow_mut();
        let record = state.mapped_gathers.get(key_of_v1!(gather))?;
        for &stream in &record.streams {
            state.refuse_during_capture(state.streams.get(stream)?.context)?;
        }
        let object = record.object;
        state.device.enqueue_mapped_gather(object, rows)
    }

    fn wait_mapped_gather(
        &self,
        gather: runtime_bundle_abi::BundleMappedGatherHandleV1,
        consumer: BundleStreamHandleV1,
    ) -> HostServiceResultV1<()> {
        let mut state = self.state.borrow_mut();
        let record = state.mapped_gathers.get(key_of_v1!(gather))?;
        let key = key_of_v1!(consumer);
        if !record.streams.contains(&key) {
            return Err(BundleStatusV1::InvalidArgument);
        }
        // Wait placement is outside capture, so every replay observes this cycle's event.
        state.refuse_during_capture(state.streams.get(key)?.context)?;
        let object = record.object;
        let consumer = state.stream_object(consumer)?;
        state.device.wait_mapped_gather(object, consumer)
    }

    fn complete_mapped_gather(
        &self,
        gather: runtime_bundle_abi::BundleMappedGatherHandleV1,
    ) -> HostServiceResultV1<Option<runtime_bundle_abi::MappedGatherStatsV1>> {
        let mut state = self.state.borrow_mut();
        let record = state.mapped_gathers.get(key_of_v1!(gather))?;
        for &stream in &record.streams {
            state.refuse_during_capture(state.streams.get(stream)?.context)?;
        }
        let object = record.object;
        state.device.complete_mapped_gather(object)
    }

    fn release_mapped_gather(
        &self,
        gather: runtime_bundle_abi::BundleMappedGatherHandleV1,
    ) -> HostServiceResultV1<()> {
        let mut state = self.state.borrow_mut();
        let record = state.mapped_gathers.get(key_of_v1!(gather))?;
        for &stream in &record.streams {
            state.refuse_during_capture(state.streams.get(stream)?.context)?;
        }
        release_driver_resource_v1!(
            state,
            mapped_gathers,
            key_of_v1!(gather),
            release_mapped_gather
        )
    }

    fn gather_mapped_rows(
        &self,
        stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
        gather: MappedRowGatherV1<'_>,
    ) -> HostServiceResultV1<()> {
        let _profile =
            runtime_engine::profile::PhaseTimer::new(runtime_engine::profile::StepPhase::Upload);
        let mut state = self.state.borrow_mut();
        state.refuse_during_capture(state.streams.get(key_of_v1!(stream))?.context)?;
        if gather.rank >= state.config.topology.rank_count() {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let ComponentStorageV1::Planned(planned) =
            &state.images.get(key_of_v1!(gather.image))?.storage
        else {
            return Err(BundleStatusV1::NotFound);
        };
        // The instance's configuration holds the planned image, and with it
        // the mapping, for the instance's whole life.
        let mapping = Arc::clone(
            planned
                .mapped_view(gather.rank, gather.view_id)
                .ok_or(BundleStatusV1::NotFound)?,
        );
        let table = mapping.bytes();
        let row_bytes =
            usize::try_from(gather.row_bytes).map_err(|_| BundleStatusV1::InvalidArgument)?;
        if gather.rows.is_empty() || row_bytes == 0 || !table.len().is_multiple_of(row_bytes) {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let table_rows = table.len() / row_bytes;
        if gather.rows.iter().any(|&row| row as usize >= table_rows) {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let stream_object = state.stream_object(stream)?;
        let range = state.resolve_range(destination)?;
        if (gather.rows.len() as u64).checked_mul(gather.row_bytes) != Some(range.length) {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let probe = MappedProbe::new(&mapping, row_bytes);
        let ahead = ReadAhead::configured(&probe).with_resident(mapping.resident_plane(row_bytes));
        state
            .device
            .fill_host_to_device_ordered(stream_object, range, &mut |at, chunk| {
                gather_rows_v1(table, row_bytes, gather.rows, at, chunk, &ahead);
            })
            .inspect_err(|status| {
                eprintln!(
                    "host gather_mapped_rows: {} rows of {row_bytes} bytes, stream {}: {status:?}",
                    gather.rows.len(),
                    stream.slot()
                );
            })
    }

    fn open_authenticated_component(
        &self,
        component_role: &[u8],
    ) -> HostServiceResultV1<BundleAuthenticatedComponentHandleV1> {
        let mut state = self.state.borrow_mut();
        if component_role == runtime_bundle_abi::BUNDLE_PROBE_AUTHORITY_ROLE_V1 {
            return Err(BundleStatusV1::NotFound);
        }
        // This reserved role resolves host-authenticated physical rank identities.
        // It cannot be supplied by a package metadata component.
        if let Some(query) = component_role.strip_prefix(b"chop.compiled-image-identity.v1/") {
            let (role, rank) = std::str::from_utf8(query)
                .ok()
                .and_then(|query| query.rsplit_once('/'))
                .ok_or(BundleStatusV1::InvalidArgument)?;
            let rank = rank
                .parse::<u32>()
                .map_err(|_| BundleStatusV1::InvalidArgument)?;
            let component = state
                .config
                .components
                .get(role.as_bytes())
                .filter(|component| component.description.is_some())
                .ok_or(BundleStatusV1::NotFound)?;
            // A planned image's identity is its plan's, which the bundle
            // names in the rank index it planned.
            let identity = match &component.storage {
                ComponentStorageV1::Planned(image) => {
                    let identity = image
                        .identity(rank as usize)
                        .ok_or(BundleStatusV1::NotFound)?;
                    serde_json::json!({
                        "byte_length": image.arena_bytes(rank as usize),
                        "direct_identity": identity,
                    })
                }
                ComponentStorageV1::Bytes(_) => return Err(BundleStatusV1::Unsupported),
            };
            let payload: Arc<[u8]> = serde_json::to_vec(&identity)
                .map_err(|_| BundleStatusV1::HostError)?
                .into();
            let sha256 = sha256_bytes(&payload)
                .as_bytes()
                .try_into()
                .expect("SHA-256 hex");
            let order = state.order.acquire()?;
            return match state
                .authenticated_components
                .insert(AuthenticatedComponentRecordV1 {
                    role: Arc::from(component_role),
                    payload,
                    sha256,
                    order,
                }) {
                Ok(key) => Ok(BundleAuthenticatedComponentHandleV1::from_parts(
                    key.generation,
                    key.slot,
                )),
                Err(status) => {
                    let _ = state.order.release(order);
                    Err(status)
                }
            };
        }
        let component = state
            .config
            .components
            .get(component_role)
            .ok_or(BundleStatusV1::NotFound)?
            .clone();
        let sha256 = component
            .authenticated_sha256
            .ok_or(BundleStatusV1::NotFound)?;
        // A package component's verified bytes, or a planned component: the
        // bundle's own plan output, digested when it was registered.
        let payload = match component.storage {
            ComponentStorageV1::Bytes(bytes) => bytes,
            ComponentStorageV1::Planned(_) => return Err(BundleStatusV1::Unsupported),
        };
        let order = state.order.acquire()?;
        match state
            .authenticated_components
            .insert(AuthenticatedComponentRecordV1 {
                role: Arc::from(component_role),
                payload,
                sha256,
                order,
            }) {
            Ok(key) => Ok(BundleAuthenticatedComponentHandleV1::from_parts(
                key.generation,
                key.slot,
            )),
            Err(status) => {
                let _ = state.order.release(order);
                Err(status)
            }
        }
    }

    fn inspect_authenticated_component(
        &self,
        component: BundleAuthenticatedComponentHandleV1,
    ) -> HostServiceResultV1<AuthenticatedComponentMetadataV1> {
        let state = self.state.borrow();
        let record = state.authenticated_components.get(key_of_v1!(component))?;
        Ok(AuthenticatedComponentMetadataV1::new(
            record.role.len() as u64,
            record.payload.len() as u64,
            record.sha256,
        ))
    }

    fn read_authenticated_component(
        &self,
        component: BundleAuthenticatedComponentHandleV1,
        role_destination: &mut [u8],
        bytes_destination: &mut [u8],
    ) -> HostServiceResultV1<()> {
        let state = self.state.borrow();
        let record = state.authenticated_components.get(key_of_v1!(component))?;
        if role_destination.len() != record.role.len()
            || bytes_destination.len() as u64 != record.payload.len() as u64
        {
            return Err(BundleStatusV1::InvalidArgument);
        }
        role_destination.copy_from_slice(&record.role);
        bytes_destination.copy_from_slice(&record.payload);
        if sha256_bytes(bytes_destination).as_bytes() != record.sha256 {
            return Err(BundleStatusV1::HostError);
        }
        Ok(())
    }

    fn release_authenticated_component(
        &self,
        component: BundleAuthenticatedComponentHandleV1,
    ) -> HostServiceResultV1<()> {
        let mut state = self.state.borrow_mut();
        release_tracked_resource_v1!(state, authenticated_components, key_of_v1!(component))
    }

    fn acquire_peer_access(
        &self,
        owner_context: BundleContextHandleV1,
        peer_context: BundleContextHandleV1,
    ) -> HostServiceResultV1<BundlePeerAccessHandleV1> {
        let mut state = self.state.borrow_mut();
        let owner_key = key_of_v1!(owner_context);
        let peer_key = key_of_v1!(peer_context);
        if owner_key == peer_key {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let owner = state.contexts.get(owner_key)?.object;
        let peer = state.contexts.get(peer_key)?.object;
        let edge = ContextPeerEdgeKeyV1 {
            owner: state.context_device(owner_key)?,
            peer: state.context_device(peer_key)?,
        };
        // Every fallible piece of bookkeeping happens before the device
        // transition, so a refused order entry or a full handle table costs no
        // driver state and leaves nothing to undo. Undoing an enable is the one
        // rollback that can itself fail, and this is how it never has to run.
        let order = state.order.acquire()?;
        let key = match state.peers.insert(PeerAccessRecordV1 { edge, order }) {
            Ok(key) => key,
            Err(status) => {
                // Nothing else was acquired after this entry, so giving it back
                // cannot fail and leaves the instance exactly as it was.
                let _ = state.order.release(order);
                return Err(status);
            }
        };
        match state.acquire_context_peer_lease(edge, owner, peer) {
            Ok(()) => Ok(BundlePeerAccessHandleV1::from_parts(
                key.generation,
                key.slot,
            )),
            Err(status) => {
                // The lease was not taken, so the slot and the order entry this
                // call reserved are the only things to give back, and both are
                // this call's own newest.
                let _ = state.peers.remove(key);
                let _ = state.order.release(order);
                Err(status)
            }
        }
    }

    fn release_peer_access(
        &self,
        peer_access: BundlePeerAccessHandleV1,
    ) -> HostServiceResultV1<()> {
        let mut state = self.state.borrow_mut();
        let key = key_of_v1!(peer_access);
        let (order, edge) = {
            let record = state.peers.get(key)?;
            (record.order, record.edge)
        };
        state.order.preflight(order)?;
        state.release_context_peer_lease(edge)?;
        state.order.release(order)?;
        state.peers.remove(key).map(|_| ())
    }

    fn create_stream(
        &self,
        context: BundleContextHandleV1,
        stream_kind: u32,
    ) -> HostServiceResultV1<BundleStreamHandleV1> {
        let mut state = self.state.borrow_mut();
        let context_key = key_of_v1!(context);
        let context_object = state.contexts.get(context_key)?.object;
        insert_stream_v1(&mut state, context_key, context_object, stream_kind, None)
    }

    fn select_stage_stream(
        &self,
        context: BundleContextHandleV1,
        rank: u32,
        stage_id: &[u8],
        stream_kind: u32,
    ) -> HostServiceResultV1<BundleStreamHandleV1> {
        let mut state = self.state.borrow_mut();
        let context_key = key_of_v1!(context);
        let record = state.contexts.get(context_key)?;
        if record.rank != rank {
            return Err(BundleStatusV1::InvalidArgument);
        }
        if stage_id.is_empty() {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let context_object = record.object;
        let stage = StageStreamKeyV1 {
            context_slot: context_key.slot,
            rank,
            stage_id: stage_id.to_vec(),
            stream_kind,
        };
        if let Some(existing) = state.stage_streams.get(&stage).copied() {
            // One stage of one rank always selects the same stream.
            if state.streams.get(existing).is_ok() {
                return Ok(BundleStreamHandleV1::from_parts(
                    existing.generation,
                    existing.slot,
                ));
            }
            state.stage_streams.remove(&stage);
        }
        insert_stream_v1(
            &mut state,
            context_key,
            context_object,
            stream_kind,
            Some(stage),
        )
    }

    fn synchronize_stream(&self, stream: BundleStreamHandleV1) -> HostServiceResultV1<()> {
        crate::overlap::device_wait();
        let _profile = runtime_engine::profile::PhaseTimer::new(
            runtime_engine::profile::StepPhase::DeviceWait,
        );
        let mut state = self.state.borrow_mut();
        state.refuse_during_capture(state.streams.get(key_of_v1!(stream))?.context)?;
        let object = state.stream_object(stream)?;
        state.device.synchronize_stream(object)
    }

    fn release_stream(&self, stream: BundleStreamHandleV1) -> HostServiceResultV1<()> {
        let mut state = self.state.borrow_mut();
        let key = key_of_v1!(stream);
        state.refuse_during_capture(state.streams.get(key)?.context)?;
        let (order, object, stage) = {
            let record = state.streams.get(key)?;
            (record.order, record.object, record.stage.clone())
        };
        state.order.preflight(order)?;
        state.device.release_stream(object)?;
        state.order.release(order)?;
        if let Some(stage) = stage {
            state.stage_streams.remove(&stage);
        }
        state.streams.remove(key).map(|_| ())
    }

    fn begin_graph_capture(&self, stream: BundleStreamHandleV1) -> HostServiceResultV1<()> {
        let mut state = self.state.borrow_mut();
        let key = key_of_v1!(stream);
        let record = state.streams.get(key)?;
        let (context, object) = (record.context, record.object);
        // One capture per device at a time; free-memory accounting is per capture.
        state.refuse_during_capture(context)?;
        let free = state.free_context_bytes(context);
        state.device.begin_graph_capture(object)?;
        state.captures.insert(context, key);
        state.capture_members.insert(context, BTreeSet::from([key]));
        match free {
            Some(bytes) => state.capture_free_bytes.insert(context, bytes),
            None => state.capture_free_bytes.remove(&context),
        };
        Ok(())
    }

    fn end_graph_capture(
        &self,
        stream: BundleStreamHandleV1,
    ) -> HostServiceResultV1<BundleGraphHandleV1> {
        let mut state = self.state.borrow_mut();
        let key = key_of_v1!(stream);
        let record = state.streams.get(key)?;
        let (context, stream_object) = (record.context, record.object);
        if state.captures.get(&context) != Some(&key) {
            return Err(BundleStatusV1::InvalidState);
        }
        // Reserve every fallible host record before creating a driver graph.
        // Reservation failure leaves capture open and its buffers owned for retry.
        let order = state.order.acquire()?;
        let mut ended = false;
        let registered = {
            let HostStateV1 { device, graphs, .. } = &mut *state;
            graphs.try_insert_with(|| {
                let object = match device.end_graph_capture(stream_object) {
                    Ok(object) => object,
                    Err(GraphCaptureErrorV1::Capturing(status)) => return Err(status),
                    Err(GraphCaptureErrorV1::Ended(status)) => {
                        ended = true;
                        return Err(status);
                    }
                };
                Ok(GraphRecordV1 {
                    object,
                    context,
                    order,
                })
            })
        };
        let graph_key = match registered {
            Ok(key) => key,
            Err(status) => {
                state.order.release(order)?;
                if ended {
                    state.end_capture_bookkeeping(context);
                    state.capture_free_bytes.remove(&context);
                }
                return Err(status);
            }
        };
        state.end_capture_bookkeeping(context);
        // Graphs hold no memory nodes, so what a capture costs is the
        // driver's executable graph: free memory before and after it (M1 in
        // docs/serving-profiles.md).
        if let (Some(before), Some(after)) = (
            state.capture_free_bytes.remove(&context),
            state.free_context_bytes(context),
        ) {
            let rank = state
                .contexts
                .get(context)
                .map_or(u32::MAX, |record| record.rank);
            eprintln!(
                "graph capture memory rank {rank}: {before} bytes free before, {after} after, {} held",
                i128::from(before) - i128::from(after)
            );
        }
        Ok(BundleGraphHandleV1::from_parts(
            graph_key.generation,
            graph_key.slot,
        ))
    }

    fn launch_graph(
        &self,
        stream: BundleStreamHandleV1,
        graph: BundleGraphHandleV1,
    ) -> HostServiceResultV1<()> {
        let _profile = runtime_engine::profile::PhaseTimer::new(
            runtime_engine::profile::StepPhase::GraphLaunch,
        );
        let mut state = self.state.borrow_mut();
        state.admit_during_capture(key_of_v1!(stream), CaptureSubmissionV1::Other)?;
        let stream_record = state.streams.get(key_of_v1!(stream))?;
        let (context, stream_object) = (stream_record.context, stream_record.object);
        let graph_record = state.graphs.get(key_of_v1!(graph))?;
        if graph_record.context != context {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let graph_object = graph_record.object;
        state.device.launch_graph(stream_object, graph_object)?;
        crate::overlap::note_launch();
        Ok(())
    }

    fn release_graph(&self, graph: BundleGraphHandleV1) -> HostServiceResultV1<()> {
        let mut state = self.state.borrow_mut();
        release_driver_resource_v1!(state, graphs, key_of_v1!(graph), release_graph)
    }

    fn create_event(
        &self,
        context: BundleContextHandleV1,
    ) -> HostServiceResultV1<BundleEventHandleV1> {
        let mut state = self.state.borrow_mut();
        let context_object = state.context_object(context)?;
        let object = state.device.create_event(context_object)?;
        let order = match state.order.acquire() {
            Ok(order) => order,
            Err(status) => {
                let _ = state.device.release_event(object);
                return Err(status);
            }
        };
        match state.events.insert(EventRecordV1 {
            recorded: false,
            object,
            context: key_of_v1!(context),
            order,
        }) {
            Ok(key) => Ok(BundleEventHandleV1::from_parts(key.generation, key.slot)),
            Err(status) => {
                let _ = state.order.release(order);
                let _ = state.device.release_event(object);
                Err(status)
            }
        }
    }

    fn record_event(
        &self,
        event: BundleEventHandleV1,
        stream: BundleStreamHandleV1,
    ) -> HostServiceResultV1<()> {
        let mut state = self.state.borrow_mut();
        state.admit_during_capture(key_of_v1!(stream), CaptureSubmissionV1::Record)?;
        let event_object = state.events.get(key_of_v1!(event))?.object;
        let stream_object = state.stream_object(stream)?;
        state.device.record_event(event_object, stream_object)?;
        state.events.get_mut(key_of_v1!(event))?.recorded = true;
        // An event recorded inside a capture is that capture's; one recorded
        // outside no longer is.
        let context = state.streams.get(key_of_v1!(stream))?.context;
        match state.capture_on_device_of(context) {
            Some(capture) => state.capture_events.insert(key_of_v1!(event), capture),
            None => state.capture_events.remove(&key_of_v1!(event)),
        };
        Ok(())
    }

    fn wait_event(
        &self,
        stream: BundleStreamHandleV1,
        event: BundleEventHandleV1,
    ) -> HostServiceResultV1<()> {
        let mut state = self.state.borrow_mut();
        state.admit_during_capture(
            key_of_v1!(stream),
            CaptureSubmissionV1::Wait(key_of_v1!(event)),
        )?;
        let stream_object = state.stream_object(stream)?;
        let event_object = state.events.get(key_of_v1!(event))?.object;
        state.device.wait_event(stream_object, event_object)
    }

    fn synchronize_event(&self, event: BundleEventHandleV1) -> HostServiceResultV1<()> {
        crate::overlap::device_wait();
        let _profile = runtime_engine::profile::PhaseTimer::new(
            runtime_engine::profile::StepPhase::DeviceWait,
        );
        let mut state = self.state.borrow_mut();
        let record = state.events.get(key_of_v1!(event))?;
        state.refuse_during_capture(record.context)?;
        let object = record.object;
        state.device.synchronize_event(object)
    }

    fn release_event(&self, event: BundleEventHandleV1) -> HostServiceResultV1<()> {
        let mut state = self.state.borrow_mut();
        release_driver_resource_v1!(state, events, key_of_v1!(event), release_event)
    }

    fn copy_host_to_device(
        &self,
        stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
        source: &[u8],
    ) -> HostServiceResultV1<()> {
        let _profile =
            runtime_engine::profile::PhaseTimer::new(runtime_engine::profile::StepPhase::Upload);
        let mut state = self.state.borrow_mut();
        state.refuse_during_capture(state.streams.get(key_of_v1!(stream))?.context)?;
        let stream_object = state.stream_object(stream)?;
        let range = state.resolve_range(destination)?;
        if range.length != source.len() as u64 {
            return Err(BundleStatusV1::InvalidArgument);
        }
        // Stream-ordered: a bundle orders its own later work on this stream.
        state
            .device
            .copy_host_to_device_ordered(stream_object, range, source)
            .inspect_err(|status| {
                eprintln!(
                    "host copy_host_to_device: {} bytes, stream {}: {status:?}",
                    source.len(),
                    stream.slot()
                );
            })
    }

    fn copy_device_to_host(
        &self,
        stream: BundleStreamHandleV1,
        destination: &mut [u8],
        source: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()> {
        let mut state = self.state.borrow_mut();
        state.refuse_during_capture(state.streams.get(key_of_v1!(stream))?.context)?;
        let stream_object = state.stream_object(stream)?;
        let range = state.resolve_range(source)?;
        if range.length != destination.len() as u64 {
            return Err(BundleStatusV1::InvalidArgument);
        }
        state
            .device
            .copy_device_to_host(stream_object, destination, range)
    }

    fn copy_device_to_device(
        &self,
        stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
        source: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()> {
        let mut state = self.state.borrow_mut();
        state.admit_during_capture(key_of_v1!(stream), CaptureSubmissionV1::Other)?;
        let stream_object = state.stream_object(stream)?;
        let destination = state.resolve_range(destination)?;
        let source = state.resolve_range(source)?;
        if destination.length != source.length {
            return Err(BundleStatusV1::InvalidArgument);
        }
        state
            .device
            .copy_device_to_device(stream_object, destination, source)
    }

    fn copy_peer_to_peer(
        &self,
        stream: BundleStreamHandleV1,
        destination_context: BundleContextHandleV1,
        destination: DeviceBufferSpanV1,
        source_context: BundleContextHandleV1,
        source: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()> {
        let mut state = self.state.borrow_mut();
        state.admit_during_capture(key_of_v1!(stream), CaptureSubmissionV1::Other)?;
        // Either end's device may be capturing.
        state.refuse_during_capture(key_of_v1!(destination_context))?;
        state.refuse_during_capture(key_of_v1!(source_context))?;
        let stream_object = state.stream_object(stream)?;
        let destination_object = state.context_object(destination_context)?;
        let source_object = state.context_object(source_context)?;
        let destination_key = key_of_v1!(destination_context);
        let source_key = key_of_v1!(source_context);
        if state.streams.get(key_of_v1!(stream))?.context != destination_key
            || state
                .buffers
                .get(key_of_v1!(destination.buffer))?
                .origin
                .map(|origin| origin.context)
                != Some(destination_key)
            || state
                .buffers
                .get(key_of_v1!(source.buffer))?
                .origin
                .map(|origin| origin.context)
                != Some(source_key)
        {
            return Err(BundleStatusV1::InvalidArgument);
        }
        state.consumer_may_reach(destination_key, key_of_v1!(source.buffer))?;
        let destination = state.resolve_range(destination)?;
        let source = state.resolve_range(source)?;
        if destination.length != source.length {
            return Err(BundleStatusV1::InvalidArgument);
        }
        state.device.copy_peer_to_peer(
            stream_object,
            destination_object,
            destination,
            source_object,
            source,
        )
    }

    fn zero_device_buffer(
        &self,
        stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()> {
        let mut state = self.state.borrow_mut();
        state.refuse_during_capture(state.streams.get(key_of_v1!(stream))?.context)?;
        let stream_object = state.stream_object(stream)?;
        let range = state.resolve_range(destination)?;
        state
            .device
            .zero_device_buffer(stream_object, range)
            .inspect_err(|status| {
                eprintln!(
                    "host zero_device_buffer: {} bytes, stream {}, buffer {}: {status:?}",
                    destination.length,
                    stream.slot(),
                    destination.buffer.slot()
                );
            })
    }

    /// Writes a bounded table of resolved device addresses into one span.
    ///
    /// The destination has to hold exactly one eight-byte entry per slot: a
    /// table that is longer or shorter than the kernel reading it expects is a
    /// silent misread, not a partial write, so it is refused. Every slot is
    /// resolved and bounds checked here, and turning each range into the
    /// address the table carries is the backend's own work.
    fn write_device_pointer_table(
        &self,
        stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
        slots: &[Option<DeviceBufferSpanV1>],
    ) -> HostServiceResultV1<()> {
        let mut state = self.state.borrow_mut();
        let stream_key = key_of_v1!(stream);
        let consumer_context = state.streams.get(stream_key)?.context;
        state.refuse_during_capture(consumer_context)?;
        let stream_object = state.stream_object(stream)?;
        // The table is written for whatever will read it on this stream, so the
        // stream's context is the consumer, the destination has to be that
        // context's own memory, and every address written into it has to be one
        // that context may reach. All of that is decided before any backend
        // call, so a refusal writes nothing.
        let destination_buffer =
            slot_key_v1(destination.buffer.generation(), destination.buffer.slot())?;
        let destination_origin = state
            .buffers
            .get(destination_buffer)?
            .origin
            .ok_or(BundleStatusV1::InvalidArgument)?;
        if destination_origin.context != consumer_context {
            return Err(BundleStatusV1::InvalidArgument);
        }
        state.require_reachable_slots(consumer_context, slots)?;
        let range = state.resolve_range(destination)?;
        let expected = (slots.len() as u64)
            .checked_mul(DEVICE_POINTER_TABLE_ENTRY_BYTES_V1)
            .ok_or(BundleStatusV1::InvalidArgument)?;
        if slots.is_empty() || range.length != expected {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let resolved = slots
            .iter()
            .map(|slot| slot.map(|span| state.resolve_range(span)).transpose())
            .collect::<HostServiceResultV1<Vec<Option<DeviceRangeV1>>>>()?;
        state
            .device
            .write_device_pointer_table(stream_object, range, &resolved)
    }

    fn load_cuda_module(
        &self,
        context: BundleContextHandleV1,
        module_bytes: &[u8],
    ) -> HostServiceResultV1<BundleCudaModuleHandleV1> {
        let mut state = self.state.borrow_mut();
        let context_object = state.context_object(context)?;
        let object = state.device.load_module(context_object, module_bytes)?;
        let order = match state.order.acquire() {
            Ok(order) => order,
            Err(status) => {
                let _ = state.device.release_module(object);
                return Err(status);
            }
        };
        match state.modules.insert(ModuleRecordV1 {
            object,
            context: key_of_v1!(context),
            order,
        }) {
            Ok(key) => Ok(BundleCudaModuleHandleV1::from_parts(
                key.generation,
                key.slot,
            )),
            Err(status) => {
                let _ = state.order.release(order);
                let _ = state.device.release_module(object);
                Err(status)
            }
        }
    }

    fn resolve_cuda_symbol(
        &self,
        module: BundleCudaModuleHandleV1,
        symbol_name: &[u8],
    ) -> HostServiceResultV1<BundleCudaSymbolHandleV1> {
        let mut state = self.state.borrow_mut();
        let module_record = state.modules.get(key_of_v1!(module))?;
        let module_object = module_record.object;
        let module_context = module_record.context;
        let object = state.device.resolve_symbol(module_object, symbol_name)?;
        let order = match state.order.acquire() {
            Ok(order) => order,
            Err(status) => {
                let _ = state.device.release_symbol(object);
                return Err(status);
            }
        };
        match state.symbols.insert(SymbolRecordV1 {
            object,
            context: module_context,
            order,
        }) {
            Ok(key) => Ok(BundleCudaSymbolHandleV1::from_parts(
                key.generation,
                key.slot,
            )),
            Err(status) => {
                let _ = state.order.release(order);
                let _ = state.device.release_symbol(object);
                Err(status)
            }
        }
    }

    fn configure_cuda_symbol(
        &self,
        symbol: BundleCudaSymbolHandleV1,
        dynamic_shared_memory_bytes: u32,
    ) -> HostServiceResultV1<()> {
        let mut state = self.state.borrow_mut();
        let object = state.symbols.get(key_of_v1!(symbol))?.object;
        state
            .device
            .configure_symbol(object, dynamic_shared_memory_bytes)
    }

    fn query_cluster_residency(
        &self,
        symbol: BundleCudaSymbolHandleV1,
        config: LaunchConfigV1,
    ) -> HostServiceResultV1<runtime_bundle_abi::BundleRawClusterResidencyV1> {
        let mut state = self.state.borrow_mut();
        let object = state.symbols.get(key_of_v1!(symbol))?.object;
        match state.device.cluster_residency(object, config) {
            Err(BundleStatusV1::Unsupported) => {
                Ok(runtime_bundle_abi::BundleRawClusterResidencyV1::unreported())
            }
            other => other,
        }
    }

    fn launch_cuda_symbol(
        &self,
        stream: BundleStreamHandleV1,
        symbol: BundleCudaSymbolHandleV1,
        config: LaunchConfigV1,
        arguments: &[KernelArgumentV1],
    ) -> HostServiceResultV1<()> {
        let mut state = self.state.borrow_mut();
        state.admit_during_capture(key_of_v1!(stream), CaptureSubmissionV1::Launch)?;
        let stream_object = state.stream_object(stream)?;
        let symbol_record = state.symbols.get(key_of_v1!(symbol))?;
        let symbol_object = symbol_record.object;
        let launch_context = symbol_record.context;
        // A launch runs in its module's context, so a stream of any other
        // context would order it somewhere the symbol does not live.
        if state.streams.get(key_of_v1!(stream))?.context != launch_context {
            return Err(BundleStatusV1::InvalidArgument);
        }
        // A by-value address array is the one argument that may legitimately
        // name another context's memory, so it is the one argument whose slots
        // are checked against the leases and grants this instance holds.
        for argument in arguments {
            if let KernelArgumentV1::DevicePointerArray(slots) = argument {
                state
                    .require_reachable_slots(launch_context, slots)
                    .inspect_err(|status| {
                        eprintln!(
                            "host launch_cuda_symbol {} peer pointers: {status:?}",
                            symbol.slot()
                        );
                    })?;
            }
        }
        let arguments = state.resolve_arguments(arguments).inspect_err(|status| {
            eprintln!(
                "host launch_cuda_symbol {} arguments: {status:?}",
                symbol.slot()
            );
        })?;
        state
            .device
            .launch_symbol(stream_object, symbol_object, config, &arguments)
            .inspect_err(|status| {
                eprintln!(
                    "host launch_cuda_symbol {} dispatch: {status:?}",
                    symbol.slot()
                );
            })?;
        // A launch into a capture runs nothing until its graph is launched.
        if !state.captures.contains_key(&launch_context) {
            crate::overlap::note_launch();
        }
        Ok(())
    }

    fn release_cuda_symbol(&self, symbol: BundleCudaSymbolHandleV1) -> HostServiceResultV1<()> {
        let mut state = self.state.borrow_mut();
        release_driver_resource_v1!(state, symbols, key_of_v1!(symbol), release_symbol)
    }

    fn release_cuda_module(&self, module: BundleCudaModuleHandleV1) -> HostServiceResultV1<()> {
        let mut state = self.state.borrow_mut();
        release_driver_resource_v1!(state, modules, key_of_v1!(module), release_module)
    }

    fn allocate_physical_kv(
        &self,
        request: &PhysicalKvRequestV1,
    ) -> HostServiceResultV1<BundlePhysicalKvHandleV1> {
        let mut state = self.state.borrow_mut();
        if !state.captures.is_empty() {
            return Err(BundleStatusV1::InvalidState);
        }
        let contract = state.config.kv.clone().ok_or(BundleStatusV1::Unsupported)?;
        let layout = contract.layout;
        // The manifest already declared this layout. The request may only ask
        // for the same one, so nothing a bundle passes here can reshape host
        // memory.
        if request.layout.layout_id != contract.layout_id
            || request.layout.page_bytes != layout.page_bytes()
            || request.layout.page_alignment != layout.alignment_bytes()
            || request.layout.page_tokens != layout.page_tokens() as u64
            || request.layout.rank_count != layout.rank_count() as u64
            || request.layout.state_lifetime != state_lifetime_code_v1(layout.state_lifetime())
        {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let request_capacity = usize::try_from(request.request_capacity)
            .map_err(|_| BundleStatusV1::InvalidArgument)?;
        let plan = PagedKvPlanV1::validate(layout, request.page_capacity, request_capacity)
            .map_err(|_| BundleStatusV1::InvalidArgument)?;
        let page_capacity =
            usize::try_from(plan.page_capacity()).map_err(|_| BundleStatusV1::InvalidArgument)?;
        let geometry = KvGeometry::new(layout.page_tokens(), page_capacity)
            .map_err(|_| BundleStatusV1::InvalidArgument)?;
        let pool = KvPool::new(geometry, request_capacity)
            .map_err(|_| BundleStatusV1::ResourceExhausted)?;
        state.acquire_host_ranks()?;
        let rank_buffers = match state
            .allocate_rank_buffers(plan.bytes_per_rank_pool(), layout.alignment_bytes())
        {
            Ok(buffers) => buffers,
            Err(status) => {
                state.drop_host_rank_user();
                return Err(status);
            }
        };
        let order = match state.order.acquire() {
            Ok(order) => order,
            Err(status) => {
                release_rank_buffers_v1(&mut state, rank_buffers);
                state.drop_host_rank_user();
                return Err(status);
            }
        };
        let owned = rank_buffers.clone();
        match state.kv.insert(PhysicalKvRecordV1 {
            plan,
            pool,
            rank_buffers,
            order,
        }) {
            Ok(key) => Ok(BundlePhysicalKvHandleV1::from_parts(
                key.generation,
                key.slot,
            )),
            Err(status) => {
                let _ = state.order.release(order);
                release_rank_buffers_v1(&mut state, owned);
                state.drop_host_rank_user();
                Err(status)
            }
        }
    }

    fn create_kv_request(
        &self,
        kv: BundlePhysicalKvHandleV1,
        quota_tokens: u64,
    ) -> HostServiceResultV1<BundleKvRequestHandleV1> {
        let mut state = self.state.borrow_mut();
        let kv_key = key_of_v1!(kv);
        let quota = usize::try_from(quota_tokens).map_err(|_| BundleStatusV1::InvalidArgument)?;
        let sequence = state
            .kv
            .get_mut(kv_key)?
            .pool
            .create_sequence(quota)
            .map_err(|_| BundleStatusV1::ResourceExhausted)?;
        let order = state.order.acquire()?;
        match state.kv_requests.insert(KvRequestRecordV1 {
            kv: kv_key,
            sequence,
            order,
        }) {
            Ok(key) => Ok(BundleKvRequestHandleV1::from_parts(
                key.generation,
                key.slot,
            )),
            Err(status) => {
                let _ = state.order.release(order);
                Err(status)
            }
        }
    }

    fn prepare_kv(
        &self,
        request: BundleKvRequestHandleV1,
        operation: &[u8],
    ) -> HostServiceResultV1<BundleKvTransactionHandleV1> {
        let mut state = self.state.borrow_mut();
        let request_key = key_of_v1!(request);
        let record = state.kv_requests.get(request_key)?;
        let (kv_key, sequence) = (record.kv, record.sequence);
        let tokens = decode_append_operation_v1(operation)?;
        let prepared = state
            .kv
            .get_mut(kv_key)?
            .pool
            .prepare_append_batch(&[(sequence, tokens)])
            .map_err(|_| BundleStatusV1::ResourceExhausted)?;
        let prepared_id = prepared.id();
        let pages = kv_page_work_v1(&prepared);
        let order = match state.order.acquire() {
            Ok(order) => order,
            Err(status) => {
                let _ = state
                    .kv
                    .get_mut(kv_key)
                    .map(|record| record.pool.rollback_prepared(prepared_id));
                return Err(status);
            }
        };
        match state.kv_transactions.insert(KvTransactionRecordV1 {
            kv: kv_key,
            prepared: prepared_id,
            pages,
            order,
        }) {
            Ok(key) => Ok(BundleKvTransactionHandleV1::from_parts(
                key.generation,
                key.slot,
            )),
            Err(status) => {
                let _ = state.order.release(order);
                let _ = state
                    .kv
                    .get_mut(kv_key)
                    .map(|record| record.pool.rollback_prepared(prepared_id));
                Err(status)
            }
        }
    }

    fn commit_kv(&self, transaction: BundleKvTransactionHandleV1) -> HostServiceResultV1<()> {
        let mut state = self.state.borrow_mut();
        let key = key_of_v1!(transaction);
        let (kv_key, prepared, pages, order) = {
            let record = state.kv_transactions.get(key)?;
            (
                record.kv,
                record.prepared,
                record.pages.clone(),
                record.order,
            )
        };
        state.order.preflight(order)?;
        // The logical commit is proved possible before any byte moves, so a
        // pool that would refuse the commit refuses it before the device runs.
        let commit = state
            .kv
            .get(kv_key)?
            .pool
            .preflight_commit_prepared(prepared, &[])
            .map_err(|_| BundleStatusV1::InvalidState)?;
        // The pages this append reserved are not reachable through any
        // committed page table yet, so a failed plan leaves nothing observable
        // behind and the transaction stays cancellable.
        state.execute_kv_page_work_v1(kv_key, &pages)?;
        state
            .kv
            .get_mut(kv_key)?
            .pool
            .apply_prepared_commit(commit)
            .map_err(|_| BundleStatusV1::InvalidState)?;
        state.order.release(order)?;
        state.kv_transactions.remove(key).map(|_| ())
    }

    fn cancel_kv(&self, transaction: BundleKvTransactionHandleV1) -> HostServiceResultV1<()> {
        let mut state = self.state.borrow_mut();
        let key = key_of_v1!(transaction);
        let record = state.kv_transactions.get(key)?;
        let (kv_key, prepared, order) = (record.kv, record.prepared, record.order);
        state.order.preflight(order)?;
        if state.kv.get(kv_key)?.pool.prepared_id() != Some(prepared) {
            return Err(BundleStatusV1::InvalidState);
        }
        state
            .kv
            .get_mut(kv_key)?
            .pool
            .rollback_prepared(prepared)
            .map_err(|_| BundleStatusV1::InvalidState)?;
        state.order.release(order)?;
        state.kv_transactions.remove(key).map(|_| ())
    }

    fn resolve_kv_view(
        &self,
        request: BundleKvRequestHandleV1,
        rank: u32,
        view_id: &[u8],
    ) -> HostServiceResultV1<DeviceBufferSpanV1> {
        let state = self.state.borrow();
        let request_key = slot_key_v1(request.generation(), request.slot())?;
        let record = state.kv_requests.get(request_key)?;
        let kv = state.kv.get(record.kv)?;
        let buffer_key = *kv
            .rank_buffers
            .get(rank as usize)
            .ok_or(BundleStatusV1::InvalidState)?;
        let buffer = state.buffers.get(buffer_key)?;
        let handle = BundleDeviceBufferHandleV1::from_parts(buffer_key.generation, buffer_key.slot);
        if view_id == KV_RANK_POOL_VIEW_V1 {
            return Ok(DeviceBufferSpanV1::new(handle, 0, buffer.byte_length));
        }
        let page = PhysicalPageIdV1::new(decode_page_ordinal_v1(view_id)?);
        kv.plan
            .validate_pages(&[page])
            .map_err(|_| BundleStatusV1::NotFound)?;
        let offset = kv
            .plan
            .page_offset(page)
            .map_err(|_| BundleStatusV1::InvalidArgument)?;
        Ok(DeviceBufferSpanV1::new(
            handle,
            offset,
            kv.plan.bytes_per_page(),
        ))
    }

    fn release_kv_request(&self, request: BundleKvRequestHandleV1) -> HostServiceResultV1<()> {
        let mut state = self.state.borrow_mut();
        let key = key_of_v1!(request);
        let record = state.kv_requests.get(key)?;
        let kv = record.kv;
        let sequence = record.sequence;
        let order = record.order;
        state.order.preflight(order)?;
        state
            .kv
            .get_mut(kv)?
            .pool
            .release(sequence)
            .map_err(|_| BundleStatusV1::InvalidState)?;
        state.kv_requests.remove(key)?;
        state.order.release(order)
    }

    fn release_physical_kv(&self, kv: BundlePhysicalKvHandleV1) -> HostServiceResultV1<()> {
        let mut state = self.state.borrow_mut();
        if !state.captures.is_empty() {
            return Err(BundleStatusV1::InvalidState);
        }
        let key = key_of_v1!(kv);
        let order = state.kv.get(key)?.order;
        state.order.preflight(order)?;
        let mut owned = std::mem::take(&mut state.kv.get_mut(key)?.rank_buffers);
        if let Err(status) = release_owned_rank_buffers_v1(&mut state, &mut owned) {
            // The record keeps exactly the buffers the driver still owns.
            state.kv.get_mut(key)?.rank_buffers = owned;
            return Err(status);
        }
        state.order.release(order)?;
        state.kv.remove(key)?;
        state.drop_host_rank_user();
        Ok(())
    }

    fn allocate_state(&self, request: &StateRequestV1) -> HostServiceResultV1<BundleStateHandleV1> {
        let mut state = self.state.borrow_mut();
        if !state.captures.is_empty() {
            return Err(BundleStatusV1::InvalidState);
        }
        let contract = state
            .config
            .state
            .clone()
            .ok_or(BundleStatusV1::Unsupported)?;
        if request.state_id != contract.state_id
            || request.byte_length != contract.bytes
            || request.alignment != contract.alignment
        {
            return Err(BundleStatusV1::InvalidArgument);
        }
        state.acquire_host_ranks()?;
        let buffers = match state.allocate_rank_buffers(contract.bytes, contract.alignment) {
            Ok(buffers) => buffers,
            Err(status) => {
                state.drop_host_rank_user();
                return Err(status);
            }
        };
        let order = match state.order.acquire() {
            Ok(order) => order,
            Err(status) => {
                release_rank_buffers_v1(&mut state, buffers);
                state.drop_host_rank_user();
                return Err(status);
            }
        };
        let owned = buffers.clone();
        match state.states.insert(StateRecordV1 { buffers, order }) {
            Ok(key) => Ok(BundleStateHandleV1::from_parts(key.generation, key.slot)),
            Err(status) => {
                let _ = state.order.release(order);
                release_rank_buffers_v1(&mut state, owned);
                state.drop_host_rank_user();
                Err(status)
            }
        }
    }

    fn resolve_state_view(
        &self,
        state_handle: BundleStateHandleV1,
        rank: u32,
    ) -> HostServiceResultV1<DeviceBufferSpanV1> {
        let state = self.state.borrow();
        let key = slot_key_v1(state_handle.generation(), state_handle.slot())?;
        let record = state.states.get(key)?;
        let buffer_key = *record
            .buffers
            .get(rank as usize)
            .ok_or(BundleStatusV1::InvalidState)?;
        let buffer = state.buffers.get(buffer_key)?;
        Ok(DeviceBufferSpanV1::new(
            BundleDeviceBufferHandleV1::from_parts(buffer_key.generation, buffer_key.slot),
            0,
            buffer.byte_length,
        ))
    }

    fn release_state(&self, state_handle: BundleStateHandleV1) -> HostServiceResultV1<()> {
        let mut state = self.state.borrow_mut();
        if !state.captures.is_empty() {
            return Err(BundleStatusV1::InvalidState);
        }
        let key = key_of_v1!(state_handle);
        let order = state.states.get(key)?.order;
        state.order.preflight(order)?;
        let mut owned = std::mem::take(&mut state.states.get_mut(key)?.buffers);
        if let Err(status) = release_owned_rank_buffers_v1(&mut state, &mut owned) {
            // The record keeps exactly the buffers the driver still owns.
            state.states.get_mut(key)?.buffers = owned;
            return Err(status);
        }
        state.order.release(order)?;
        state.states.remove(key)?;
        state.drop_host_rank_user();
        Ok(())
    }

    fn create_collective(
        &self,
        request: &CollectiveRequestV1,
    ) -> HostServiceResultV1<BundleCollectiveHandleV1> {
        let mut state = self.state.borrow_mut();
        if request.rank_count != u64::from(state.config.topology.rank_count()) {
            return Err(BundleStatusV1::Unsupported);
        }
        if request.semantic_contract.is_empty() {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let binding = CollectiveBindingV1 {
            rank_count: request.rank_count,
            semantic_contract: request.semantic_contract.clone(),
        };
        let order = state.order.acquire()?;
        match state
            .collectives
            .insert(CollectiveRecordV1 { binding, order })
        {
            Ok(key) => Ok(BundleCollectiveHandleV1::from_parts(
                key.generation,
                key.slot,
            )),
            Err(status) => {
                let _ = state.order.release(order);
                Err(status)
            }
        }
    }

    fn execute_collective(
        &self,
        collective: BundleCollectiveHandleV1,
        stream: BundleStreamHandleV1,
        operation: &[u8],
        arguments: &[KernelArgumentV1],
    ) -> HostServiceResultV1<()> {
        let mut state = self.state.borrow_mut();
        state.refuse_during_capture(state.streams.get(key_of_v1!(stream))?.context)?;
        let binding = state
            .collectives
            .get(key_of_v1!(collective))?
            .binding
            .clone();
        let stream_object = state.stream_object(stream)?;
        let arguments = state.resolve_arguments(arguments)?;
        state
            .device
            .execute_collective(stream_object, &binding, operation, &arguments)
    }

    fn release_collective(&self, collective: BundleCollectiveHandleV1) -> HostServiceResultV1<()> {
        let mut state = self.state.borrow_mut();
        release_tracked_resource_v1!(state, collectives, key_of_v1!(collective))
    }

    /// Runs one bundle-supplied operation with values this host lends it.
    ///
    /// Everything a handle table can decide is decided here, before the backend
    /// is asked for anything: the operation names itself, both handles are this
    /// instance's and live, every span resolves inside the allocation it names,
    /// and no operand the operation writes shares a byte with another operand
    /// of the same call. A refusal at any of those invokes nothing at all.
    fn run_custom_operation(
        &self,
        context: BundleContextHandleV1,
        stream: BundleStreamHandleV1,
        completion: CustomCompletionV1,
        operation_id: &[u8],
        operands: &[CustomOperandV1],
        operation: &CustomOperationV1<'_>,
    ) -> HostServiceResultV1<i32> {
        let mut state = self.state.borrow_mut();
        state.refuse_during_capture(key_of_v1!(context))?;
        // An unnamed operation is one nothing can be resolved or audited by.
        if operation_id.is_empty() {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let context_object = state.context_object(context)?;
        let stream_object = state.stream_object(stream)?;
        let resolved = operands
            .iter()
            .map(|operand| {
                Ok(DeviceCustomOperandV1 {
                    range: state.resolve_range(operand.span)?,
                    writable: operand.writable,
                })
            })
            .collect::<HostServiceResultV1<Vec<DeviceCustomOperandV1>>>()?;
        reject_aliased_custom_operands_v1(&resolved)?;
        // Whatever was ordered on this stream earlier has to land before the
        // operation is lent anything, because the addresses it is about to
        // receive name buffers that work may still be writing. The wait comes
        // before the callback, so a wait this host cannot complete invokes
        // nothing and changes nothing, and the same call can be made again.
        state.device.synchronize_stream(stream_object)?;
        let outcome = state.device.run_custom_operation(
            context_object,
            stream_object,
            operation_id,
            &resolved,
            operation,
        );
        if !completion.waits_after_callback() {
            // A synchronous operation finished everything it started, so there
            // is nothing to wait for and no wait that could fail between its
            // result and the caller.
            return outcome;
        }
        // An operation that may enqueue is still running when its callback
        // returns, and every address it was lent belongs to a buffer this host
        // may be asked to release next. The wait is what makes that release
        // safe, and a wait the device refuses is the more serious answer.
        match (outcome, state.device.synchronize_stream(stream_object)) {
            (outcome, Ok(())) => outcome,
            (_, Err(status)) => Err(status),
        }
    }
}

/// Refuses an operand set in which something written is also named elsewhere.
///
/// Two ranges of different allocations never share a byte, and two ranges of
/// one allocation share bytes exactly when they intersect. The rule is the
/// caller's to declare and this host's to enforce: an operation that writes
/// over an operand it also reads is refused before anything is lent, because
/// after the lend nothing here can see what it did.
fn reject_aliased_custom_operands_v1(
    operands: &[DeviceCustomOperandV1],
) -> HostServiceResultV1<()> {
    let mut ends = Vec::with_capacity(operands.len());
    for operand in operands {
        ends.push(
            operand
                .range
                .offset
                .checked_add(operand.range.length)
                .ok_or(BundleStatusV1::InvalidArgument)?,
        );
    }
    for (writable_index, writable) in operands.iter().enumerate() {
        if !writable.writable {
            continue;
        }
        for (other_index, other) in operands.iter().enumerate() {
            if writable_index == other_index || writable.range.object != other.range.object {
                continue;
            }
            if writable.range.offset < ends[other_index]
                && other.range.offset < ends[writable_index]
            {
                return Err(BundleStatusV1::InvalidArgument);
            }
        }
    }
    Ok(())
}

/// The stream kind the host uses for its own per-rank transfers.
const HOST_STREAM_KIND_V1: u32 = 0;

fn component_byte_length_v1(storage: &ComponentStorageV1) -> u64 {
    match storage {
        ComponentStorageV1::Bytes(bytes) => bytes.len() as u64,
        ComponentStorageV1::Planned(image) => image
            .description()
            .rank_images()
            .last()
            .map_or(0, |rank| rank.offset + rank.byte_length),
    }
}

fn validate_component_range_v1(
    component_length: u64,
    offset: u64,
    byte_length: u64,
) -> HostServiceResultV1<()> {
    let end = offset
        .checked_add(byte_length)
        .ok_or(BundleStatusV1::InvalidArgument)?;
    if byte_length == 0 || end > component_length {
        return Err(BundleStatusV1::InvalidArgument);
    }
    Ok(())
}

fn upload_component_range_v1<D: HostDeviceBackendV1>(
    device: &mut D,
    stream: DeviceObjectV1,
    destination: DeviceObjectV1,
    storage: &ComponentStorageV1,
    source_offset: u64,
    byte_length: u64,
) -> HostServiceResultV1<()> {
    validate_component_range_v1(
        component_byte_length_v1(storage),
        source_offset,
        byte_length,
    )?;
    match storage {
        ComponentStorageV1::Bytes(bytes) => {
            let start =
                usize::try_from(source_offset).map_err(|_| BundleStatusV1::InvalidArgument)?;
            let length =
                usize::try_from(byte_length).map_err(|_| BundleStatusV1::InvalidArgument)?;
            let end = start
                .checked_add(length)
                .ok_or(BundleStatusV1::InvalidArgument)?;
            let source = bytes
                .get(start..end)
                .ok_or(BundleStatusV1::InvalidArgument)?;
            device.copy_host_to_device(
                stream,
                DeviceRangeV1 {
                    object: destination,
                    offset: 0,
                    length: byte_length,
                },
                source,
            )
        }
        ComponentStorageV1::Planned(_) => {
            let mut read =
                |offset, chunk: &mut [u8]| read_component_bytes_v1(storage, offset, chunk);
            if let Some(result) = device.upload_pinned_file_range(
                stream,
                destination,
                source_offset,
                byte_length,
                &mut read,
            ) {
                result
            } else {
                upload_chunked_component_range_v1(
                    device,
                    stream,
                    destination,
                    source_offset,
                    byte_length,
                    read,
                )
            }
        }
    }
}

/// Every path that completes an upload from a direct checkpoint calls this
/// once the copies are done: the checkpoint must still have the files,
/// lengths and headers its plans were made from. Other storage passes.
fn verify_direct_sources_v1(storage: &ComponentStorageV1) -> HostServiceResultV1<()> {
    let ComponentStorageV1::Planned(image) = storage else {
        return Ok(());
    };
    image.checkpoint().verify_unchanged().map_err(|error| {
        eprintln!("host direct load failed: {error}");
        BundleStatusV1::HostError
    })
}

/// Set to `1` to log the SHA-256 of every uploaded rank arena, read back
/// from the device (`startup arena_digest`).
pub const ARENA_DIGEST_ENV_V1: &str = "CHOPINFER_DEBUG_ARENA_DIGEST";

/// The SHA-256 of `length` device bytes, copied back in 64 MiB chunks.
fn device_digest_v1<D: HostDeviceBackendV1>(
    device: &mut D,
    stream: DeviceObjectV1,
    object: DeviceObjectV1,
    length: u64,
) -> HostServiceResultV1<String> {
    struct DeviceReader<'a, D: HostDeviceBackendV1> {
        device: &'a mut D,
        stream: DeviceObjectV1,
        object: DeviceObjectV1,
        offset: u64,
        length: u64,
        failed: Option<BundleStatusV1>,
    }
    impl<D: HostDeviceBackendV1> std::io::Read for DeviceReader<'_, D> {
        fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
            let count = (self.length - self.offset).min(buffer.len().min(64 << 20) as u64) as usize;
            if count == 0 {
                return Ok(0);
            }
            let range = DeviceRangeV1 {
                object: self.object,
                offset: self.offset,
                length: count as u64,
            };
            if let Err(status) =
                self.device
                    .copy_device_to_host(self.stream, &mut buffer[..count], range)
            {
                self.failed = Some(status);
                return Err(std::io::Error::other("device read failed"));
            }
            self.offset += count as u64;
            Ok(count)
        }
    }
    let mut reader = DeviceReader {
        device,
        stream,
        object,
        offset: 0,
        length,
        failed: None,
    };
    let mut reader = std::io::BufReader::with_capacity(64 << 20, &mut reader);
    let digest = runtime_weights::sha256_reader(&mut reader);
    drop(reader);
    digest.map_err(|_| BundleStatusV1::HostError)
}

/// Uploads one bounded component range in host-buffer-sized chunks.
///
/// `read_chunk` fills one chunk or refuses before it reaches the device.
fn upload_chunked_component_range_v1<D: HostDeviceBackendV1>(
    device: &mut D,
    stream: DeviceObjectV1,
    destination: DeviceObjectV1,
    source_offset: u64,
    byte_length: u64,
    mut read_chunk: impl FnMut(u64, &mut [u8]) -> bool,
) -> HostServiceResultV1<()> {
    let capacity = usize::try_from(byte_length)
        .unwrap_or(usize::MAX)
        .min(FILE_BACKED_COMPONENT_UPLOAD_BUFFER_BYTES_V1);
    let mut chunk = vec![0_u8; capacity];
    let mut copied = 0_u64;
    while copied < byte_length {
        let remaining = byte_length - copied;
        let count = usize::try_from(remaining)
            .unwrap_or(usize::MAX)
            .min(chunk.len());
        if !read_chunk(source_offset + copied, &mut chunk[..count]) {
            return Err(BundleStatusV1::HostError);
        }
        let count_u64 = count as u64;
        device.copy_host_to_device(
            stream,
            DeviceRangeV1 {
                object: destination,
                offset: copied,
                length: count_u64,
            },
            &chunk[..count],
        )?;
        copied += count_u64;
    }
    Ok(())
}

fn release_rank_buffers_v1<D: HostDeviceBackendV1>(
    state: &mut HostStateV1<D>,
    buffers: Vec<SlotKeyV1>,
) {
    for buffer in buffers.into_iter().rev() {
        let _ = state.release_host_buffer(buffer);
    }
}

/// Releases owned rank buffers newest first, keeping the ones still live.
///
/// A driver that refuses one release leaves that buffer and every older one in
/// `buffers`, so the owning record keeps exactly what the driver still holds and
/// the same release can be retried.
fn release_owned_rank_buffers_v1<D: HostDeviceBackendV1>(
    state: &mut HostStateV1<D>,
    buffers: &mut Vec<SlotKeyV1>,
) -> HostServiceResultV1<()> {
    while let Some(buffer) = buffers.last().copied() {
        state.release_host_buffer(buffer)?;
        buffers.pop();
    }
    Ok(())
}

fn insert_stream_v1<D: HostDeviceBackendV1>(
    state: &mut HostStateV1<D>,
    context_key: SlotKeyV1,
    context_object: DeviceObjectV1,
    stream_kind: u32,
    stage: Option<StageStreamKeyV1>,
) -> HostServiceResultV1<BundleStreamHandleV1> {
    let object = state.device.create_stream(context_object, stream_kind)?;
    let order = match state.order.acquire() {
        Ok(order) => order,
        Err(status) => {
            let _ = state.device.release_stream(object);
            return Err(status);
        }
    };
    match state.streams.insert(StreamRecordV1 {
        object,
        context: context_key,
        stage: stage.clone(),
        order,
    }) {
        Ok(key) => {
            if let Some(stage) = stage {
                state.stage_streams.insert(stage, key);
            }
            Ok(BundleStreamHandleV1::from_parts(key.generation, key.slot))
        }
        Err(status) => {
            let _ = state.order.release(order);
            let _ = state.device.release_stream(object);
            Err(status)
        }
    }
}

/// Reads the page work one prepared append declared.
///
/// `runtime-kv` reports which pages an append newly reserved and which shared
/// page it forked. Both are page identities, so this translation names no
/// model and no tensor.
fn kv_page_work_v1(prepared: &PreparedAppendBatch) -> KvPageWorkV1 {
    let mut work = KvPageWorkV1::default();
    for append in prepared.appends() {
        for block in append.newly_allocated() {
            work.zero.push(PhysicalPageIdV1::new(block.index()));
        }
        if let Some(copy) = append.copy() {
            work.copies.push(KvPageCopyV1 {
                source: PhysicalPageIdV1::new(copy.source().index()),
                destination: PhysicalPageIdV1::new(copy.destination().index()),
                initialized_tokens: copy.initialized_tokens(),
            });
        }
    }
    work
}

const fn state_lifetime_code_v1(lifetime: StateLifetimeV1) -> u64 {
    match lifetime {
        StateLifetimeV1::Transaction => 0,
        StateLifetimeV1::Request => 1,
        StateLifetimeV1::Process => 2,
    }
}

fn decode_append_operation_v1(operation: &[u8]) -> HostServiceResultV1<usize> {
    let payload = operation
        .strip_prefix(KV_APPEND_OPERATION_V1)
        .ok_or(BundleStatusV1::Unsupported)?;
    let tokens = decode_u64_v1(payload)?;
    usize::try_from(tokens).map_err(|_| BundleStatusV1::InvalidArgument)
}

fn decode_page_ordinal_v1(view_id: &[u8]) -> HostServiceResultV1<usize> {
    let ordinal = decode_u64_v1(view_id)?;
    usize::try_from(ordinal).map_err(|_| BundleStatusV1::InvalidArgument)
}

fn decode_u64_v1(bytes: &[u8]) -> HostServiceResultV1<u64> {
    let value: [u8; 8] = bytes
        .try_into()
        .map_err(|_| BundleStatusV1::InvalidArgument)?;
    Ok(u64::from_le_bytes(value))
}

/// Encodes one KV append operation for `prepare_kv`.
#[must_use]
pub fn kv_append_operation_v1(additional_tokens: u64) -> Vec<u8> {
    let mut operation = KV_APPEND_OPERATION_V1.to_vec();
    operation.extend_from_slice(&additional_tokens.to_le_bytes());
    operation
}

/// Encodes one physical page index for `resolve_kv_view`.
#[must_use]
pub fn kv_page_view_id_v1(physical_page: u64) -> Vec<u8> {
    physical_page.to_le_bytes().to_vec()
}

/// What a submission during an open capture does, for
/// `admit_during_capture`.
#[derive(Clone, Copy, Debug)]
enum CaptureSubmissionV1 {
    /// A kernel launch.
    Launch,
    /// Recording an event.
    Record,
    /// Waiting on this event.
    Wait(SlotKeyV1),
    /// Any other work: graph launches and copies.
    Other,
}

#[cfg(all(test, feature = "fake-host-services"))]
mod tests {
    use std::fs;
    use std::io::{Seek, SeekFrom, Write};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::SystemTime;

    use runtime_bundle::VerifiedComponentFileV1;
    use runtime_bundle_abi::{HostServicesBackendV1, PhysicalKvLayoutV1, PhysicalKvRequestV1};
    use runtime_kv::{
        DeclaredPageLayoutV1, KvGeometry, KvPool, PhysicalPageIdV1, RankPlacementV1,
        StateLifetimeV1,
    };
    use runtime_weights::sha256_bytes;

    use super::{
        DeclaredKvContractV1, HostInstanceConfigV1, HostServicesV1Backend, HostTopologyV1,
        KvPageCopyV1, KvPageWorkV1, kv_page_work_v1, slot_key_v1,
    };
    use crate::device::{DeviceRangeV1, HostDeviceBackendV1};
    use crate::fake::FakeHostDeviceV1;

    const RANKS: u8 = 2;
    const LAYOUT_ID: &[u8] = b"paged-kv-layout";
    const BYTES_PER_TOKEN: u32 = 64;
    const SEGMENTS: u32 = 2;
    const PAGE_TOKENS: u32 = 8;
    const PAGE_BYTES: u64 = (BYTES_PER_TOKEN * SEGMENTS * PAGE_TOKENS) as u64;
    const SEGMENT_BYTES: u64 = (BYTES_PER_TOKEN * PAGE_TOKENS) as u64;
    const PAGE_CAPACITY: u64 = 8;
    static NEXT_COMPONENT_FILE: AtomicU64 = AtomicU64::new(0);

    fn verified_component_file(bytes: &[u8]) -> (VerifiedComponentFileV1, PathBuf) {
        let path = std::env::temp_dir().join(format!(
            "runtime-host-authenticated-component-{}-{}",
            std::process::id(),
            NEXT_COMPONENT_FILE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::write(&path, bytes).expect("the component fixture is writable");
        let file = VerifiedComponentFileV1::open(&path, bytes.len() as u64, &sha256_bytes(bytes))
            .expect("the component fixture authenticates");
        (file, path)
    }

    fn page_layout() -> DeclaredPageLayoutV1 {
        DeclaredPageLayoutV1::new(
            BYTES_PER_TOKEN,
            SEGMENTS,
            PAGE_TOKENS,
            64,
            RANKS,
            RankPlacementV1::Sharded,
            StateLifetimeV1::Request,
            64,
        )
        .expect("the test page layout is valid")
    }

    fn backend() -> HostServicesV1Backend<FakeHostDeviceV1> {
        let config = HostInstanceConfigV1::new(
            HostTopologyV1::new(vec![0, 1]).expect("the test topology is valid"),
        )
        .with_kv_contract(DeclaredKvContractV1::new(LAYOUT_ID.to_vec(), page_layout()));
        HostServicesV1Backend::new(FakeHostDeviceV1::new(), config)
    }

    #[test]
    fn offload_startup_is_lazy_bounded_and_refuses_low_memory_before_pinning() {
        use crate::offload::{KvOffloadExecutorV1, KvOffloadOptionsV1};
        let config = HostInstanceConfigV1::new(HostTopologyV1::new(vec![0, 1]).unwrap())
            .with_kv_contract(DeclaredKvContractV1::new(LAYOUT_ID.to_vec(), page_layout()))
            .with_serving_kv(super::ServingKvPolicyV1 {
                maximum_pages: Some(8),
                ..Default::default()
            });
        let backend = HostServicesV1Backend::new(FakeHostDeviceV1::new(), config);
        let before = backend.with_device(FakeHostDeviceV1::live_object_count);
        assert_eq!(
            backend
                .initialize_kv_offload(KvOffloadOptionsV1::default(), 0)
                .unwrap(),
            None
        );
        assert_eq!(
            backend.with_device(FakeHostDeviceV1::live_object_count),
            before
        );
        for rank in 0..2 {
            let context = backend.select_context(rank, rank as i32).unwrap();
            backend.select_memory_pool(context).unwrap();
        }
        backend.acquire_serving_kv(LAYOUT_ID).unwrap();
        let before = backend.with_device(FakeHostDeviceV1::live_object_count);
        let options = KvOffloadOptionsV1 {
            budget_bytes: PAGE_BYTES * 16,
            ..Default::default()
        };
        assert!(
            backend
                .initialize_kv_offload(options, 16 * 1024 * 1024 * 1024)
                .is_err()
        );
        assert_eq!(
            backend.with_device(FakeHostDeviceV1::live_object_count),
            before
        );
        let plan = backend
            .initialize_kv_offload(options, 32 * 1024 * 1024 * 1024)
            .unwrap()
            .unwrap();
        assert_eq!(plan.pages_per_rank, 8);
        assert_eq!(
            backend.with_device(FakeHostDeviceV1::live_object_count),
            before + 2
        );
        assert!(backend.initialize_kv_offload(options, u64::MAX).is_err());
        backend.drain().unwrap();
    }

    #[test]
    fn startup_checkpoint_allocates_no_device_kv_or_pinned_resources() {
        use crate::executor::{BundleExecutionV1, BundleModelExecutorV1};
        use crate::offload::KvOffloadOptionsV1;
        use crate::probe::HostDeviceRankMemoryUsageV1;
        use runtime_bundle_abi::{BundleAbiError, OwnedLogitsRowV1, ResultSinkLimitsV1};
        use runtime_engine::{Engine, EngineConfig, KvOffloadInfo};
        use runtime_model_api::ModelVocabulary;
        use runtime_scheduler::SchedulerConfig;
        use std::rc::Rc;

        struct IdleBundle;
        impl BundleExecutionV1 for IdleBundle {
            fn execute_batch(
                &mut self,
                _: &[u8],
                _: ResultSinkLimitsV1,
            ) -> Result<Vec<OwnedLogitsRowV1>, BundleAbiError> {
                panic!("checkpoint must not execute the bundle")
            }
            fn commit(&mut self) -> Result<(), BundleAbiError> {
                panic!("checkpoint must not commit the bundle")
            }
            fn cancel(&mut self) -> Result<(), BundleAbiError> {
                panic!("checkpoint must not cancel the bundle")
            }
            fn destroy(&mut self) -> Result<(), BundleAbiError> {
                Ok(())
            }
            fn is_poisoned(&self) -> bool {
                false
            }
        }

        for offload in [false, true] {
            let config = HostInstanceConfigV1::new(HostTopologyV1::new(vec![0, 1]).unwrap())
                .with_kv_contract(DeclaredKvContractV1::new(LAYOUT_ID.to_vec(), page_layout()))
                .with_serving_kv(super::ServingKvPolicyV1 {
                    maximum_pages: Some(8),
                    ..Default::default()
                });
            let backend = Rc::new(HostServicesV1Backend::new(FakeHostDeviceV1::new(), config));
            for rank in 0..2 {
                let context = backend.select_context(rank, rank as i32).unwrap();
                backend.select_memory_pool(context).unwrap();
            }
            assert_eq!(backend.acquire_serving_kv(LAYOUT_ID).unwrap(), 8);
            let mut executor = BundleModelExecutorV1::new(IdleBundle, 8, 2);
            if offload {
                let plan = backend
                    .initialize_kv_offload(
                        KvOffloadOptionsV1 {
                            budget_bytes: PAGE_BYTES * 16,
                            ..Default::default()
                        },
                        32 * 1024 * 1024 * 1024,
                    )
                    .unwrap()
                    .unwrap();
                executor = executor.with_kv_offload(
                    backend.clone(),
                    KvOffloadInfo {
                        host_page_capacity: plan.pages_per_rank,
                        minimum_tokens: 1,
                        snapshot_pages: 1,
                    },
                );
            }
            let config = EngineConfig::new(
                SchedulerConfig::new(2, 4, 8, 8, 1).unwrap(),
                KvGeometry::new(PAGE_TOKENS as usize, 8).unwrap(),
                ModelVocabulary::new(8).unwrap(),
                64,
                2,
            )
            .unwrap()
            .with_prefix_cache(1);
            let mut engine = Engine::new(config, executor).unwrap();
            let resources = || {
                backend.with_device(|device| {
                    (
                        device.issued_object_count(),
                        device.live_object_count(),
                        device.rank_memory_v1().unwrap(),
                        device.buffer_objects(),
                        device.memory_pool_objects(),
                        device.mapped_host_objects(),
                    )
                })
            };
            let before = resources();
            let views = (0..2)
                .map(|rank| backend.resolve_serving_kv_pool(rank).unwrap())
                .collect::<Vec<_>>();
            let free = engine.kv_stats().free_blocks();
            assert_eq!(free, 8);
            for _ in 0..2 {
                engine.begin_startup_warmup().unwrap();
                engine.audit().unwrap();
                assert_eq!(resources(), before);
                assert_eq!(engine.kv_stats().free_blocks(), free);
                engine.finish_startup_warmup().unwrap();
                engine.audit().unwrap();
                assert_eq!(resources(), before);
                assert_eq!(engine.kv_stats().free_blocks(), free);
                assert_eq!(backend.serving_kv_pages(), Some(8));
                for (rank, view) in views.iter().enumerate() {
                    assert_eq!(&backend.resolve_serving_kv_pool(rank as u32).unwrap(), view);
                }
            }
        }
    }

    fn launch_config(flags: u32) -> runtime_bundle_abi::LaunchConfigV1 {
        runtime_bundle_abi::LaunchConfigV1 {
            flags,
            grid_x: 16,
            grid_y: 1,
            grid_z: 1,
            block_x: 256,
            block_y: 1,
            block_z: 1,
            shared_memory_bytes: 0,
            cluster: runtime_bundle_abi::LaunchClusterV1::NONE,
        }
    }

    /// Every rank preserves the requested programmatic stream serialization.
    #[test]
    fn launches_preserve_pdl() {
        use crate::device::LaunchDispatchV1;
        let pdl = runtime_bundle_abi::BUNDLE_LAUNCH_FLAG_V1_PROGRAMMATIC_STREAM_SERIALIZATION;
        for ordinals in [vec![0, 1], vec![0, 1, 2, 3]] {
            let expected = LaunchDispatchV1::ProgrammaticStreamSerialization;
            let ranks = ordinals.len() as u32;
            let config = HostInstanceConfigV1::new(HostTopologyV1::new(ordinals.clone()).unwrap());
            let services = HostServicesV1Backend::new(FakeHostDeviceV1::new(), config);
            for rank in 0..ranks {
                let context = services
                    .select_context(rank, ordinals[rank as usize])
                    .unwrap();
                let stream = services.create_stream(context, 0).unwrap();
                let module = services.load_cuda_module(context, b"fake-module").unwrap();
                let symbol = services
                    .resolve_cuda_symbol(module, b"fake_entry_v1")
                    .unwrap();
                // The fake runs no kernel: it refuses the launch after
                // recording how a device would have dispatched it.
                assert_eq!(
                    services.launch_cuda_symbol(stream, symbol, launch_config(pdl), &[]),
                    Err(runtime_bundle_abi::BundleStatusV1::Unsupported)
                );
                assert_eq!(
                    services
                        .with_device(|device| device.refused_launch_dispatches().last().copied()),
                    Some(expected),
                    "ordinals {ordinals:?} rank {rank}"
                );
            }
        }
    }

    #[test]
    fn cluster_residency_reports_what_the_device_reports() {
        let services = backend();
        let context = services.select_context(0, 0).unwrap();
        let module = services.load_cuda_module(context, b"fake-module").unwrap();
        let symbol = services
            .resolve_cuda_symbol(module, b"fake_entry_v1")
            .unwrap();
        let mut clustered = launch_config(0);
        clustered.cluster = runtime_bundle_abi::LaunchClusterV1 { x: 8, y: 1, z: 1 };
        // A device that cannot tell reports nothing rather than failing.
        assert_eq!(
            services.query_cluster_residency(symbol, clustered).unwrap(),
            runtime_bundle_abi::BundleRawClusterResidencyV1::unreported()
        );
        let mut reported = runtime_bundle_abi::BundleRawClusterResidencyV1::unreported();
        reported.multiprocessor_count = 188;
        reported.usable_multiprocessors = 188;
        reported.max_active_clusters = 44;
        services.with_device_mut(|device| device.report_cluster_residency(reported));
        assert_eq!(
            services.query_cluster_residency(symbol, clustered).unwrap(),
            reported
        );
        // A grid of partial clusters is no shape to ask about.
        let mut partial = clustered;
        partial.grid_x = 12;
        assert_eq!(
            services.query_cluster_residency(symbol, partial),
            Err(runtime_bundle_abi::BundleStatusV1::InvalidArgument)
        );
        services.release_cuda_symbol(symbol).unwrap();
        assert!(services.query_cluster_residency(symbol, clustered).is_err());
    }

    #[test]
    fn deferred_work_runs_in_the_first_device_wait_after_a_launch() {
        let services = backend();
        let context = services.select_context(0, 0).unwrap();
        let stream = services.create_stream(context, 0).unwrap();
        services.begin_graph_capture(stream).unwrap();
        let graph = services.end_graph_capture(stream).unwrap();
        let ran = std::rc::Rc::new(std::cell::Cell::new(0));
        let counter = std::rc::Rc::clone(&ran);
        crate::overlap::defer(move || counter.set(counter.get() + 1));
        services.synchronize_stream(stream).unwrap();
        assert_eq!(ran.get(), 0, "nothing was launched, so the device is idle");
        services.launch_graph(stream, graph).unwrap();
        services.synchronize_stream(stream).unwrap();
        assert_eq!(ran.get(), 1);
        let counter = std::rc::Rc::clone(&ran);
        crate::overlap::defer(move || counter.set(counter.get() + 10));
        services.synchronize_stream(stream).unwrap();
        assert_eq!(ran.get(), 1, "the launch was consumed by the first wait");
        crate::overlap::run_pending();
        assert_eq!(ran.get(), 11);
    }

    fn graph_registration_exhaustion_keeps_capture_owned(full_table: bool) {
        use crate::fake::FakeRefusalV1;
        use crate::handles::{MAX_SLOTS_PER_TABLE_V1, SlotTableV1};
        use runtime_bundle_abi::BundleStatusV1;
        let services = backend();
        let context = services.select_context(0, 0).unwrap();
        let pool = services.select_memory_pool(context).unwrap();
        let stream = services.create_stream(context, 0).unwrap();
        let buffer = services
            .allocate_device_buffer(pool, stream, 64, 256)
            .unwrap();
        services.begin_graph_capture(stream).unwrap();
        services.with_device_mut(|device| device.refuse(FakeRefusalV1::ReleaseGraph));
        let before = services.with_device(FakeHostDeviceV1::live_object_count);
        let context_key = slot_key_v1(context.generation(), context.slot()).unwrap();
        let previous_next = {
            let mut state = services.state.borrow_mut();
            let previous = state.order.next;
            if full_table {
                let object = state
                    .streams
                    .get(slot_key_v1(stream.generation(), stream.slot()).unwrap())
                    .unwrap()
                    .object;
                for _ in 0..MAX_SLOTS_PER_TABLE_V1 - 1 {
                    state
                        .graphs
                        .insert(super::GraphRecordV1 {
                            object,
                            context: context_key,
                            order: 0,
                        })
                        .unwrap();
                }
            } else {
                state.order.next = u64::MAX;
            }
            previous
        };
        assert_eq!(
            services.end_graph_capture(stream),
            Err(BundleStatusV1::ResourceExhausted)
        );
        assert_eq!(
            services.with_device(FakeHostDeviceV1::live_object_count),
            before,
            "failed metadata reservation must not create an orphan executable"
        );
        assert!(services.state.borrow().captures.contains_key(&context_key));
        assert_eq!(
            services.release_device_buffer(buffer),
            Err(BundleStatusV1::InvalidState)
        );
        {
            let mut state = services.state.borrow_mut();
            state.order.next = previous_next;
            state.graphs = SlotTableV1::new();
        }
        services.with_device_mut(|device| device.allow(FakeRefusalV1::ReleaseGraph));
        let graph = services.end_graph_capture(stream).unwrap();
        services.release_graph(graph).unwrap();
        services.release_device_buffer(buffer).unwrap();
        services.release_stream(stream).unwrap();
        services.release_memory_pool(pool).unwrap();
        services.release_context(context).unwrap();
        assert_eq!(services.with_device(FakeHostDeviceV1::live_object_count), 0);
    }

    #[test]
    fn graph_registration_order_exhaustion_cannot_orphan_an_exec() {
        graph_registration_exhaustion_keeps_capture_owned(false);
    }

    #[test]
    fn graph_registration_slot_exhaustion_cannot_orphan_an_exec() {
        graph_registration_exhaustion_keeps_capture_owned(true);
    }

    #[test]
    fn speculation_query_reports_the_create_time_setting() {
        let disabled = backend().query_speculation().unwrap();
        assert_eq!(disabled.maximum_draft_tokens, 0);
        let mut enabled = runtime_bundle_abi::BundleRawSpeculationV1::disabled();
        enabled.maximum_draft_tokens = 3;
        enabled.maximum_verify_tokens = 8;
        let config = HostInstanceConfigV1::new(HostTopologyV1::new(vec![0]).unwrap())
            .with_speculation(enabled);
        let services = HostServicesV1Backend::new(FakeHostDeviceV1::new(), config);
        assert_eq!(services.query_speculation().unwrap(), enabled);
    }

    #[test]
    fn served_context_record_preserves_the_legacy_prefix() {
        let config = HostInstanceConfigV1::new(HostTopologyV1::new(vec![0, 1]).unwrap())
            .with_maximum_sequences(16);
        let old = config.encode_v1(b"test");
        let extended = config.with_maximum_context_tokens(65536).encode_v1(b"test");
        assert_eq!(&extended[..old.len()], old.as_slice());
        assert_eq!(&extended[old.len()..], &65536_u64.to_le_bytes());
    }

    #[test]
    fn a_context_parallel_group_above_one_adds_a_tagged_size() {
        let config = HostInstanceConfigV1::new(HostTopologyV1::new(vec![0, 1, 2, 3]).unwrap())
            .with_maximum_sequences(16)
            .with_maximum_context_tokens(65536);
        let plain = config.encode_v1(b"test");
        assert_eq!(
            config
                .clone()
                .with_context_parallel_size(1)
                .encode_v1(b"test"),
            plain
        );
        let options =
            std::collections::BTreeMap::from([("budget".to_owned(), serde_json::json!(1))]);
        let grouped = config
            .clone()
            .with_context_parallel_size(2)
            .with_model_options(options.clone())
            .encode_v1(b"test");
        let mut field = super::CONTEXT_PARALLEL_TAG_V1.to_vec();
        field.extend_from_slice(&2_u32.to_le_bytes());
        let mut expected = plain.clone();
        expected.extend_from_slice(&field);
        expected.extend_from_slice(
            &config.with_model_options(options).encode_v1(b"test")[plain.len()..],
        );
        assert_eq!(grouped, expected);
    }

    #[test]
    fn the_tap_producer_ends_the_record() {
        assert_eq!(
            super::TAP_PRODUCER_TAG_V1,
            runtime_bundle_sdk::taps::PRODUCER_TAG
        );
        assert_eq!(
            super::TAPS_DIR_ENV_V1,
            runtime_bundle_sdk::taps::TAPS_ENV.dir
        );
        let config = HostInstanceConfigV1::new(HostTopologyV1::new(vec![0, 1]).unwrap())
            .with_maximum_sequences(16)
            .with_maximum_context_tokens(65536);
        let plain = config.encode_v1(b"test");
        let tapped = config.with_tap_producer("ab".repeat(16)).encode_v1(b"test");
        assert_eq!(&tapped[..plain.len()], plain.as_slice());
        let mut field = super::TAP_PRODUCER_TAG_V1.to_vec();
        field.extend_from_slice(&32_u32.to_le_bytes());
        field.extend_from_slice("ab".repeat(16).as_bytes());
        assert_eq!(&tapped[plain.len()..], field.as_slice());
    }

    #[test]
    fn batch_limits_query_reports_the_create_time_setting() {
        assert_eq!(
            backend().query_batch_limits().unwrap(),
            runtime_bundle_abi::BundleRawBatchLimitsV1::unreported()
        );
        let mut limits = runtime_bundle_abi::BundleRawBatchLimitsV1::unreported();
        limits.maximum_scored_rows = 512;
        let config = HostInstanceConfigV1::new(HostTopologyV1::new(vec![0]).unwrap())
            .with_batch_limits(limits);
        let services = HostServicesV1Backend::new(FakeHostDeviceV1::new(), config);
        assert_eq!(services.query_batch_limits().unwrap(), limits);
    }

    #[test]
    fn normal_host_and_caller_component_bytes_cannot_expose_probe_authority() {
        let config =
            HostInstanceConfigV1::new(HostTopologyV1::new(vec![0]).expect("the topology is valid"))
                .with_component(
                    runtime_bundle_abi::BUNDLE_PROBE_AUTHORITY_ROLE_V1,
                    runtime_bundle_abi::BUNDLE_PROBE_AUTHORITY_BYTES_V1.to_vec(),
                )
                .expect("ordinary component bytes are valid configuration");
        let backend = HostServicesV1Backend::new(FakeHostDeviceV1::new(), config);

        assert_eq!(
            backend
                .open_authenticated_component(runtime_bundle_abi::BUNDLE_PROBE_AUTHORITY_ROLE_V1,),
            Err(runtime_bundle_abi::BundleStatusV1::NotFound)
        );
    }

    #[test]
    fn the_create_record_closes_with_the_served_sequences_when_named() {
        let topology = HostTopologyV1::new(vec![0, 1]).expect("the test topology is valid");
        let unnamed = HostInstanceConfigV1::new(topology.clone());
        assert_eq!(unnamed.maximum_sequences(), None);
        let base = unnamed.encode_v1(b"tp2");
        for count in [1_u32, 32, 33, 128, 256] {
            let named = HostInstanceConfigV1::new(topology.clone()).with_maximum_sequences(count);
            assert_eq!(named.maximum_sequences(), Some(count));
            let encoded = named.encode_v1(b"tp2");
            assert_eq!(encoded.len(), base.len() + 4);
            assert_eq!(encoded[..base.len()], base[..]);
            assert_eq!(encoded[base.len()..], count.to_le_bytes());
        }
    }

    #[test]
    fn only_manifest_authenticated_evidence_is_exposed_by_handle() {
        let topology = HostTopologyV1::new(vec![0]).expect("the test topology is valid");
        let ordinary = HostInstanceConfigV1::new(topology.clone())
            .with_component(
                b"chop.execution-evidence.fixture".to_vec(),
                b"ordinary".to_vec(),
            )
            .expect("an ordinary component is valid");
        let ordinary_backend = HostServicesV1Backend::new(FakeHostDeviceV1::new(), ordinary);
        assert_eq!(
            ordinary_backend.open_authenticated_component(b"chop.execution-evidence.fixture"),
            Err(runtime_bundle_abi::BundleStatusV1::NotFound)
        );

        let bytes = b"authenticated receipt\n".to_vec();
        let pin = sha256_bytes(&bytes);
        let (file, path) = verified_component_file(&bytes);
        let authenticated = HostInstanceConfigV1::new(topology)
            .with_authenticated_component_file(b"chop.execution-evidence.fixture".to_vec(), file)
            .expect("the manifest pin authenticates the bytes");
        let encoded = authenticated.encode_v1(b"tp1");
        assert!(!encoded.windows(bytes.len()).any(|window| window == bytes));
        assert!(
            !encoded
                .windows(pin.len())
                .any(|window| window == pin.as_bytes())
        );

        let backend = HostServicesV1Backend::new(FakeHostDeviceV1::new(), authenticated);
        let handle = backend
            .open_authenticated_component(b"chop.execution-evidence.fixture")
            .expect("the authenticated component opens");
        let metadata = backend
            .inspect_authenticated_component(handle)
            .expect("the live handle has metadata");
        assert_eq!(metadata.role_length, 31);
        assert_eq!(metadata.byte_length, bytes.len() as u64);
        assert_eq!(&metadata.sha256, pin.as_bytes());
        let mut role = vec![0; metadata.role_length as usize];
        let mut copied = vec![0; metadata.byte_length as usize];
        backend
            .read_authenticated_component(handle, &mut role, &mut copied)
            .expect("the exact destinations receive the component");
        assert_eq!(role, b"chop.execution-evidence.fixture");
        assert_eq!(copied, bytes);
        backend
            .release_authenticated_component(handle)
            .expect("the component releases");
        fs::remove_file(path).expect("the component fixture is removable");
    }

    #[test]
    fn authenticated_evidence_requires_its_role_and_bound() {
        let topology = HostTopologyV1::new(vec![0]).expect("the test topology is valid");
        let (ordinary, ordinary_path) = verified_component_file(b"receipt");
        assert!(matches!(
            HostInstanceConfigV1::new(topology.clone())
                .with_authenticated_component_file(b"ordinary.role".to_vec(), ordinary),
            Err(super::HostConfigErrorV1::InvalidAuthenticatedComponent)
        ));
        fs::remove_file(ordinary_path).expect("the ordinary fixture is removable");

        let large = vec![0; super::MAX_AUTHENTICATED_COMPONENT_BYTES_V1 + 1];
        let (oversized, oversized_path) = verified_component_file(&large);
        assert!(matches!(
            HostInstanceConfigV1::new(topology).with_authenticated_component_file(
                b"chop.execution-evidence.fixture".to_vec(),
                oversized,
            ),
            Err(super::HostConfigErrorV1::InvalidAuthenticatedComponent)
        ));
        fs::remove_file(oversized_path).expect("the oversized fixture is removable");
    }

    #[test]
    fn verified_bundle_metadata_opens_by_role_within_existing_bound() {
        let role = b"chop.bundle-metadata.fixture.index.v1";
        let bytes = b"{\"views\":[]}";
        let (file, path) = verified_component_file(bytes);
        let config =
            HostInstanceConfigV1::new(HostTopologyV1::new(vec![0]).expect("test topology"))
                .with_bundle_metadata_file(role.to_vec(), file)
                .expect("verified metadata");
        let backend = HostServicesV1Backend::new(FakeHostDeviceV1::new(), config);
        let handle = backend
            .open_authenticated_component(role)
            .expect("metadata role");
        let mut returned_role = vec![0; role.len()];
        let mut returned_bytes = vec![0; bytes.len()];
        backend
            .read_authenticated_component(handle, &mut returned_role, &mut returned_bytes)
            .expect("metadata bytes");
        assert_eq!(returned_role, role);
        assert_eq!(returned_bytes, bytes);
        backend
            .release_authenticated_component(handle)
            .expect("release");
        fs::remove_file(path).expect("remove fixture");
    }

    #[test]
    fn host_state_ranks_reuse_selected_bundle_contexts() {
        let backend = HostServicesV1Backend::new(
            FakeHostDeviceV1::new(),
            HostInstanceConfigV1::new(HostTopologyV1::new(vec![0]).expect("topology")),
        );
        let selected = backend.select_context(0, 0).expect("selected context");
        let selected_pool = backend.select_memory_pool(selected).expect("selected pool");
        {
            let mut state = backend.state.borrow_mut();
            state.acquire_host_ranks().expect("host state rank");
            let selected_object = state
                .contexts
                .get(slot_key_v1(selected.generation(), selected.slot()).expect("handle"))
                .expect("live selected context")
                .object;
            assert_eq!(state.host_ranks[0].context, selected_object);
            assert!(!state.host_ranks[0].context_owned);
            let selected_pool_object = state
                .pools
                .get(slot_key_v1(selected_pool.generation(), selected_pool.slot()).expect("handle"))
                .expect("live selected pool")
                .object;
            assert_eq!(state.host_ranks[0].pool, selected_pool_object);
            assert!(!state.host_ranks[0].pool_owned);
        }
        assert_eq!(
            backend.release_memory_pool(selected_pool),
            Err(runtime_bundle_abi::BundleStatusV1::InvalidState)
        );
        assert_eq!(
            backend.release_context(selected),
            Err(runtime_bundle_abi::BundleStatusV1::InvalidState)
        );
        backend.state.borrow_mut().drop_host_rank_user();
        backend
            .release_memory_pool(selected_pool)
            .expect("release pool after state");
        backend
            .release_context(selected)
            .expect("release after state");
    }

    #[test]
    fn authenticated_component_handles_reject_partial_stale_and_cross_host_reads() {
        let bytes = b"authenticated receipt\n";
        let role = b"chop.execution-evidence.fixture";
        let (first_file, first_path) = verified_component_file(bytes);
        let first_config = HostInstanceConfigV1::new(HostTopologyV1::new(vec![0]).unwrap())
            .with_authenticated_component_file(role.to_vec(), first_file)
            .unwrap();
        let first = HostServicesV1Backend::new(FakeHostDeviceV1::new(), first_config);
        let first_handle = first.open_authenticated_component(role).unwrap();
        let mut short_role = vec![0; role.len() - 1];
        let mut destination = vec![0; bytes.len()];
        assert_eq!(
            first.read_authenticated_component(first_handle, &mut short_role, &mut destination,),
            Err(runtime_bundle_abi::BundleStatusV1::InvalidArgument)
        );
        assert_eq!(destination, vec![0; bytes.len()]);

        let (second_file, second_path) = verified_component_file(bytes);
        let second_config = HostInstanceConfigV1::new(HostTopologyV1::new(vec![1]).unwrap())
            .with_authenticated_component_file(role.to_vec(), second_file)
            .unwrap();
        let second = HostServicesV1Backend::new(FakeHostDeviceV1::new(), second_config);
        let second_handle = second.open_authenticated_component(role).unwrap();
        assert_eq!(
            second.inspect_authenticated_component(first_handle),
            Err(runtime_bundle_abi::BundleStatusV1::StaleHandle)
        );
        second
            .release_authenticated_component(second_handle)
            .unwrap();

        first.release_authenticated_component(first_handle).unwrap();
        assert_eq!(
            first.inspect_authenticated_component(first_handle),
            Err(runtime_bundle_abi::BundleStatusV1::StaleHandle)
        );
        fs::remove_file(first_path).unwrap();
        fs::remove_file(second_path).unwrap();
    }

    #[test]
    fn authenticated_component_reads_ignore_path_replacement() {
        let bytes = b"authenticated receipt\n";
        let role = b"chop.execution-evidence.fixture";
        let (file, path) = verified_component_file(bytes);
        let config = HostInstanceConfigV1::new(HostTopologyV1::new(vec![0]).unwrap())
            .with_authenticated_component_file(role.to_vec(), file)
            .unwrap();
        let backend = HostServicesV1Backend::new(FakeHostDeviceV1::new(), config);
        let handle = backend.open_authenticated_component(role).unwrap();
        let replacement = path.with_extension("replacement");
        let mut replacement_bytes = bytes.to_vec();
        replacement_bytes[0] = b'x';
        fs::write(&replacement, &replacement_bytes).unwrap();
        fs::rename(&replacement, &path).unwrap();
        let mut returned_role = vec![0; role.len()];
        let mut destination = vec![0; bytes.len()];
        backend
            .read_authenticated_component(handle, &mut returned_role, &mut destination)
            .expect("the verified bytes stay readable");
        assert_eq!(destination, bytes);
        backend.release_authenticated_component(handle).unwrap();
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn authenticated_component_reads_serve_the_verified_bytes_after_a_mutation() {
        let bytes = b"authenticated receipt\n";
        let role = b"chop.execution-evidence.fixture";
        let (file, path) = verified_component_file(bytes);
        let config = HostInstanceConfigV1::new(HostTopologyV1::new(vec![0]).unwrap())
            .with_authenticated_component_file(role.to_vec(), file)
            .unwrap();
        let backend = HostServicesV1Backend::new(FakeHostDeviceV1::new(), config);
        let handle = backend.open_authenticated_component(role).unwrap();
        let mut writer = fs::OpenOptions::new().write(true).open(&path).unwrap();
        writer.seek(SeekFrom::Start(0)).unwrap();
        writer.write_all(b"x").unwrap();
        writer
            .set_times(fs::FileTimes::new().set_modified(SystemTime::UNIX_EPOCH))
            .unwrap();
        writer.sync_all().unwrap();
        let mut returned_role = vec![0; role.len()];
        let mut destination = vec![0; bytes.len()];
        backend
            .read_authenticated_component(handle, &mut returned_role, &mut destination)
            .expect("the verified bytes stay readable");
        assert_eq!(destination, bytes);
        backend.release_authenticated_component(handle).unwrap();
        fs::remove_file(path).unwrap();
    }

    fn kv_request() -> PhysicalKvRequestV1 {
        PhysicalKvRequestV1 {
            flags: 0,
            layout: PhysicalKvLayoutV1 {
                flags: 0,
                rank_count: u64::from(RANKS),
                page_bytes: PAGE_BYTES,
                page_alignment: 64,
                page_tokens: u64::from(PAGE_TOKENS),
                state_lifetime: 1,
                layout_id: LAYOUT_ID.to_vec(),
            },
            page_capacity: PAGE_CAPACITY,
            request_capacity: 4,
        }
    }

    #[test]
    fn a_forked_append_declares_its_new_pages_and_its_shared_prefix_copy() {
        let geometry = KvGeometry::new(PAGE_TOKENS as usize, PAGE_CAPACITY as usize)
            .expect("the test geometry is valid");
        let mut pool = KvPool::new(geometry, 4).expect("the test pool is valid");
        let parent = pool.create_sequence(32).expect("the parent fits");
        // A partial page, then a fork, leaves the last page shared.
        pool.append(parent, 3).expect("the parent append fits");
        let child = pool.fork(parent, 32).expect("the fork fits");
        let prepared = pool
            .prepare_append_batch(&[(child, 2)])
            .expect("the child append fits");

        let work = kv_page_work_v1(&prepared);

        assert_eq!(work.zero.len(), 1, "one page replaces the shared page");
        let copy = work
            .copies
            .first()
            .copied()
            .expect("a shared page is copied");
        assert_eq!(copy.destination, work.zero[0]);
        assert_ne!(copy.source, copy.destination);
        assert_eq!(copy.initialized_tokens, 3);
    }

    #[test]
    fn page_work_zeroes_new_pages_and_copies_the_initialized_prefix_on_every_rank() {
        let backend = backend();
        let kv = backend
            .allocate_physical_kv(&kv_request())
            .expect("the KV pool allocates");
        let kv_key = slot_key_v1(kv.generation(), kv.slot()).expect("the KV handle is valid");
        let source = PhysicalPageIdV1::new(0);
        let destination = PhysicalPageIdV1::new(1);
        let initialized_tokens = 3;
        let prefix_bytes = initialized_tokens * BYTES_PER_TOKEN as usize;

        let mut state = backend.state.borrow_mut();
        let rank_buffers = state
            .kv
            .get(kv_key)
            .expect("the record is live")
            .rank_buffers
            .clone();
        // Every page starts distinguishable, so a zero and a copy both show.
        for (rank, buffer_key) in rank_buffers.iter().enumerate() {
            let stream = state.host_ranks[rank].stream;
            let object = state
                .buffers
                .get(*buffer_key)
                .expect("the buffer is live")
                .object;
            for page in 0..PAGE_CAPACITY {
                let marker = if page == 0 { 0xAB } else { 0xCD };
                state
                    .device
                    .copy_host_to_device(
                        stream,
                        DeviceRangeV1 {
                            object,
                            offset: page * PAGE_BYTES,
                            length: PAGE_BYTES,
                        },
                        &vec![marker; PAGE_BYTES as usize],
                    )
                    .expect("the seed write lands");
            }
        }

        state
            .execute_kv_page_work_v1(
                kv_key,
                &KvPageWorkV1 {
                    zero: vec![destination],
                    copies: vec![KvPageCopyV1 {
                        source,
                        destination,
                        initialized_tokens,
                    }],
                },
            )
            .expect("the declared page work runs");

        for (rank, buffer_key) in rank_buffers.iter().enumerate() {
            let object = state
                .buffers
                .get(*buffer_key)
                .expect("the buffer is live")
                .object;
            let bytes = state
                .device
                .buffer_bytes(object)
                .expect("the rank buffer is readable")
                .to_vec();
            let page = |index: u64| {
                let start = (index * PAGE_BYTES) as usize;
                bytes[start..start + PAGE_BYTES as usize].to_vec()
            };
            let destination_page = page(1);
            for segment in 0..SEGMENTS as u64 {
                let base = (segment * SEGMENT_BYTES) as usize;
                assert_eq!(
                    &destination_page[base..base + prefix_bytes],
                    &vec![0xAB_u8; prefix_bytes],
                    "rank {rank} segment {segment} keeps the shared prefix"
                );
                assert_eq!(
                    &destination_page[base + prefix_bytes..base + SEGMENT_BYTES as usize],
                    &vec![0_u8; SEGMENT_BYTES as usize - prefix_bytes],
                    "rank {rank} segment {segment} zeroes past the shared prefix"
                );
            }
            assert_eq!(page(0), vec![0xAB_u8; PAGE_BYTES as usize]);
            assert_eq!(page(2), vec![0xCD_u8; PAGE_BYTES as usize]);
        }
    }
}

impl<D: HostDeviceBackendV1> crate::offload::KvOffloadExecutorV1 for HostServicesV1Backend<D> {
    fn available(&self) -> bool {
        self.state
            .borrow()
            .offload
            .as_ref()
            .is_some_and(|q| q.available())
    }
    fn submit(
        &self,
        plan: runtime_kv::KvTransferPlan,
    ) -> Result<(), (BundleStatusV1, runtime_kv::KvTransferPlan)> {
        let mut state = self.state.borrow_mut();
        let Some(queue) = state.offload.as_mut() else {
            return Err((BundleStatusV1::Unsupported, plan));
        };
        queue.submit(plan)
    }
    fn poll(
        &self,
        id: runtime_kv::KvTransferId,
    ) -> HostServiceResultV1<runtime_engine::KvTransferPoll> {
        let mut state = self.state.borrow_mut();
        let HostStateV1 {
            offload, device, ..
        } = &mut *state;
        offload
            .as_mut()
            .ok_or(BundleStatusV1::Unsupported)?
            .poll(device, id)
    }
    fn wait(
        &self,
        id: runtime_kv::KvTransferId,
    ) -> HostServiceResultV1<runtime_engine::KvTransferPoll> {
        let mut state = self.state.borrow_mut();
        let HostStateV1 {
            offload, device, ..
        } = &mut *state;
        offload
            .as_mut()
            .ok_or(BundleStatusV1::Unsupported)?
            .wait(device, id)
    }
    fn drain_timeout(&self) -> std::time::Duration {
        self.state
            .borrow()
            .offload
            .as_ref()
            .map_or(std::time::Duration::from_secs(5), |q| q.drain_timeout())
    }
    fn drain(&self) -> HostServiceResultV1<()> {
        let mut state = self.state.borrow_mut();
        let HostStateV1 {
            offload, device, ..
        } = &mut *state;
        if let Some(queue) = offload {
            queue.drain(device)?;
        }
        Ok(())
    }
}

#[cfg(all(test, target_os = "linux", feature = "cuda-fake-driver-tests"))]
mod device_result_service_tests {
    use super::*;
    use crate::executor::{DeviceLogitsSamplerV1, SamplingDispatch};
    use runtime_bundle_abi::device_results::*;
    use runtime_bundle_abi::{
        BUNDLE_LOGITS_ELEMENT_V1_F32, BundleDeviceBufferViewV1, BundleLogitsViewV1,
    };
    use runtime_cuda::fake_driver;

    #[test]
    fn device_hooks_prevalidate_contexts_events_and_event_joined_capture() {
        fake_driver::reset();
        let backend = HostServicesV1Backend::new(
            crate::cuda::CudaHostDeviceV1::new().unwrap(),
            HostInstanceConfigV1::new(HostTopologyV1::new(vec![0, 1]).unwrap()),
        );
        let context = backend.select_context(0, 0).unwrap();
        let foreign = backend.select_context(1, 1).unwrap();
        let pool = backend.select_memory_pool(context).unwrap();
        let stream = backend.create_stream(context, 0).unwrap();
        let joined = backend.create_stream(context, 0).unwrap();
        let logits = backend.allocate_device_buffer(pool, stream, 32, 8).unwrap();
        let output = backend
            .allocate_device_buffer(pool, stream, 256, 8)
            .unwrap();
        let ready = backend.create_event(context).unwrap();
        let done = backend.create_event(context).unwrap();
        let other = backend.create_event(foreign).unwrap();
        let capture_event = backend.create_event(context).unwrap();
        let view = |buffer, length| BundleDeviceBufferViewV1 {
            buffer,
            length,
            ..BundleDeviceBufferViewV1::empty()
        };
        let request = BundleDeviceSamplingRequestV1 {
            struct_size: std::mem::size_of::<BundleDeviceSamplingRequestV1>() as u32,
            reserved: 0,
            logits: BundleReadyLogitsViewV1 {
                struct_size: std::mem::size_of::<BundleReadyLogitsViewV1>() as u32,
                flags: 0,
                cycle_epoch: 1,
                ready,
                logits: BundleLogitsViewV1 {
                    struct_size: std::mem::size_of::<BundleLogitsViewV1>() as u32,
                    reserved: 0,
                    context,
                    rank: 0,
                    element_type: BUNDLE_LOGITS_ELEMENT_V1_F32,
                    first_item_index: 0,
                    row_count: 1,
                    live_width: 8,
                    row_stride_elements: 8,
                    buffer: view(logits, 32),
                    reserved_tail: 0,
                },
            },
            output: view(output, 256),
            done,
        };
        let dispatch = SamplingDispatch {
            rows: 1,
            simple: true,
            top: false,
            greedy: true,
        };
        let setup_waits = fake_driver::call_count(5);
        assert_eq!(
            backend.sample_device_logits(request, &[], dispatch),
            Err(BundleStatusV1::InvalidArgument),
            "unrecorded producer event"
        );
        backend.record_event(ready, stream).unwrap();
        for invalid in [
            BundleDeviceSamplingRequestV1 {
                done: other,
                ..request
            },
            BundleDeviceSamplingRequestV1 {
                logits: BundleReadyLogitsViewV1 {
                    ready: other,
                    ..request.logits
                },
                ..request
            },
            BundleDeviceSamplingRequestV1 {
                output: view(output, 255),
                ..request
            },
        ] {
            assert_eq!(
                backend.sample_device_logits(invalid, &[], dispatch),
                Err(BundleStatusV1::InvalidArgument)
            );
        }
        let packet = BundleReadyDeviceViewV1 {
            context,
            rank: 0,
            reserved: 0,
            buffer: view(output, 256),
            ready,
        };
        assert_eq!(
            backend.enqueue_device_readback(BundleReadyDeviceViewV1 {
                ready: other,
                ..packet
            }),
            Err(BundleStatusV1::InvalidArgument)
        );
        backend.begin_graph_capture(stream).unwrap();
        backend.record_event(capture_event, stream).unwrap();
        backend.wait_event(joined, capture_event).unwrap();
        assert_eq!(
            backend.sample_device_logits(request, &[], dispatch),
            Err(BundleStatusV1::InvalidState)
        );
        assert_eq!(
            backend.enqueue_device_readback(packet),
            Err(BundleStatusV1::InvalidState)
        );
        let graph = backend.end_graph_capture(stream).unwrap();
        backend.release_graph(graph).unwrap();
        assert_eq!(
            fake_driver::call_count(11),
            0,
            "validation must not construct a sampler"
        );
        assert_eq!(
            fake_driver::call_count(8),
            0,
            "validation must not upload metadata"
        );
        assert_eq!(
            fake_driver::call_count(10),
            0,
            "validation must not copy output"
        );
        assert_eq!(
            fake_driver::call_count(5),
            setup_waits,
            "validation must not synchronize a stream"
        );
    }
}
