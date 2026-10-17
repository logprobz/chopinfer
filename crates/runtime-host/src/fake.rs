//! Host-memory device services for a platform without the CUDA driver.
//!
//! This backend performs every copy, fill, and reduction for real against
//! host-owned byte vectors, so a bundle's ownership, ordering, and range
//! behaviour is exercised without a device. It launches nothing: a kernel
//! launch is reported unsupported rather than silently succeeding.

use std::collections::{BTreeMap, BTreeSet};

use runtime_bundle_abi::{
    BUNDLE_DEVICE_POINTER_ARRAY_V1_MAX_SLOTS, BundleDeviceBufferHandleV1, BundleDeviceBufferViewV1,
    BundleStatusV1, BundleTensorMapV1, CustomAddressSpaceV1, CustomOperationV1, LaunchConfigV1,
    LentLaunchV1, LentRangeV1, LentStreamV1, validate_tensor_map_v1,
};

use crate::device::{
    CollectiveBindingV1, DEVICE_POINTER_TABLE_ENTRY_BYTES_V1, DeviceArgumentV1,
    DeviceCustomOperandV1, DeviceObjectV1, DeviceRangeV1, DeviceTensorMapV1, GraphCaptureErrorV1,
    HostDeviceBackendV1, HostDeviceRankLedgerV1, HostDeviceResultV1, LaunchDispatchV1,
    PeerAccessAcquisitionV1, launch_cluster_v1, launch_dispatch_v1, plan_generic_collective_v1,
};

/// The `f32` collective contract this backend reduces in host memory.
///
/// The identity itself belongs to the shared device surface, because every
/// backend answers to the same one.
pub use crate::device::COLLECTIVE_SUM_F32_V1 as FAKE_COLLECTIVE_SUM_F32_V1;

/// The `bf16` collective contract this backend reduces in host memory.
///
/// It folds through the shared plan, so it starts from the same rank, adds the
/// rest in the same order, and narrows once through the same conversion a driver
/// backend uses. A bundle checked against this host without a device receives
/// the bytes that host would have written.
pub use crate::device::COLLECTIVE_SUM_BF16_V1 as FAKE_COLLECTIVE_SUM_BF16_V1;

/// The alignment this backend's allocations satisfy unless told otherwise.
pub const DEFAULT_FAKE_ALLOCATION_ALIGNMENT_V1: u64 = 4096;
/// The device memory a fake device reports unless set otherwise.
pub const DEFAULT_FAKE_DEVICE_MEMORY_BYTES_V1: u64 = 1024 * 1024 * 1024 + 16 * 1024 * 1024;

/// One operation this backend can be told to refuse.
///
/// A real driver refuses work for reasons a host cannot reproduce on demand.
/// Naming the refusable operations here lets the host's own failure paths be
/// exercised: a refused release must leave the resource tracked and retryable,
/// and a refused fill or copy must leave a KV append uncommitted.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum FakeRefusalV1 {
    SubmitOffload,
    QueryOffload,
    ReleaseContext,
    ReleaseMemoryPool,
    ReleaseDeviceBuffer,
    ReleasePeerAccess,
    ReleaseStream,
    /// The end of a capture, refused before the capture ends.
    EndGraphCapture,
    /// The instantiation of a graph whose capture already ended.
    InstantiateGraph,
    ReleaseEvent,
    ReleaseGraph,
    ReleaseModule,
    ReleaseSymbol,
    ZeroDeviceBuffer,
    CopyDeviceToDevice,
    /// The driver transition that enables a pool's peer access.
    EnableMemoryPoolPeerAccess,
    /// The driver transition that disables one this backend enabled.
    DisableMemoryPoolPeerAccess,
    /// The driver transition that enables a context peer edge.
    EnableContextPeerAccess,
    /// The custom operation a host refuses before it invokes anything.
    RunCustomOperation,
    /// The wait a host performs around work it did not order itself.
    SynchronizeStream,
    /// The write that fills a device-owned table with resolved addresses.
    WriteDevicePointerTable,
    /// The module load a bundle asks for before it resolves anything in it.
    LoadModule,
    /// The symbol resolution inside a module this backend already loaded.
    ResolveSymbol,
    /// The allocation a bundle asks one pool for.
    AllocateDeviceBuffer,
    /// The launch itself, for a backend that would otherwise accept it.
    LaunchSymbol,
    /// A collective refused before any rank buffer is read.
    ExecuteCollective,
    /// A collective refused after its rank buffers were updated.
    CompleteCollective,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FakeObjectKindV1 {
    Context,
    MemoryPool,
    Buffer,
    PeerAccess,
    Stream,
    Event,
    Graph,
    Module,
    Symbol,
}

struct FakeObjectV1 {
    kind: FakeObjectKindV1,
    bytes: Vec<u8>,
    logical_byte_length: u64,
}

struct FakeOffloadArena {
    buffer: DeviceObjectV1,
    bytes: Vec<u8>,
    page_bytes: usize,
    pending: BTreeMap<
        u64,
        (
            runtime_kv::KvTransferDirection,
            Vec<runtime_kv::KvPagePair>,
            usize,
        ),
    >,
    next: u64,
    query_delay: usize,
}

/// A complete device backend backed by host memory.
pub struct FakeHostDeviceV1 {
    offload: BTreeMap<u64, FakeOffloadArena>,
    objects: BTreeMap<u64, FakeObjectV1>,
    next_identity: u64,
    sampler_buffers: BTreeMap<u64, (usize, DeviceObjectV1)>,
    logs: Vec<Vec<u8>>,
    launches: Vec<Vec<u8>>,
    launch_dispatches: Vec<LaunchDispatchV1>,
    launch_arguments: Vec<Vec<DeviceArgumentV1>>,
    /// The identifier and lent ranges of every custom operation this ran.
    custom_operations: Vec<(Vec<u8>, Vec<LentRangeV1>)>,
    /// How many stream waits this backend has performed.
    synchronizations: usize,
    /// Largest host source slice passed to one device upload.
    largest_host_to_device_copy_bytes: usize,
    /// Every host byte this backend has been asked to upload.
    host_to_device_copy_bytes: u64,
    /// Refusals that take effect only after a number of further attempts.
    deferred_refusals: BTreeMap<FakeRefusalV1, usize>,
    refusals: BTreeSet<FakeRefusalV1>,
    /// The strongest alignment this backend claims its allocations satisfy.
    allocation_alignment: u64,
    /// Device memory of each context's device, before its allocations.
    device_memory_bytes: u64,
    /// Host transfers that were accepted but not performed, per stream.
    ///
    /// A driver that returns while its DMA is still running behaves this way,
    /// and nothing else about this backend would reveal it.
    deferred: Vec<(u64, DeferredTransferV1)>,
    defer_transfers: bool,
    /// The chunk each filled upload is staged in, and the streams that
    /// staged one, each holding one chunk.
    fill_chunk_bytes: usize,
    fill_streams: BTreeSet<u64>,
    skip_collective_ordering: bool,
    /// Whether a launch this backend checked is reported as accepted.
    accept_launches: bool,
    /// What `cluster_residency` reports; `None` reports it unsupported.
    cluster_residency: Option<runtime_bundle_abi::BundleRawClusterResidencyV1>,
    sparse_probe_allocations: bool,
    /// Every pool peer grant this device currently has, whoever made it.
    pool_peer_access: BTreeSet<(u64, u64)>,
    /// Every directed context peer edge this device currently has.
    context_peer_access: BTreeSet<(u64, u64)>,
    /// The edge each live peer-access record was created for.
    peer_access_edges: BTreeMap<u64, (u64, u64)>,
    /// The one pool object each context has.
    pool_by_context: BTreeMap<u64, u64>,
    /// Every allocation count this backend states, planned before it is moved.
    ///
    /// A host-memory backend accounts for what it allocated exactly the way a
    /// driver backend does, so the two answer a rank report the same way and
    /// neither carries its own arithmetic.
    ledger: HostDeviceRankLedgerV1,
    /// The context each pool, stream, and allocation belongs to.
    ///
    /// A driver backend knows this from the objects themselves. This one has to
    /// record it, because a lend that crossed contexts is exactly the mistake
    /// a device would catch and host memory would not.
    context_of: BTreeMap<u64, u64>,
    /// Every pool grant and context edge this backend was told to enable.
    ///
    /// A device shared with another user starts with grants this host did not
    /// make, and undoing one of those is the mistake these record.
    enabled_pool_peer_access: Vec<(u64, u64)>,
    disabled_pool_peer_access: Vec<(u64, u64)>,
    enabled_context_peer_access: Vec<(u64, u64)>,
    disabled_context_peer_access: Vec<(u64, u64)>,
    /// Live mapped host allocations and their bytes, outside the ledger.
    mapped_host: BTreeMap<u64, u64>,
}

/// One host transfer this backend accepted without performing.
enum DeferredTransferV1 {
    HostToDevice {
        range: DeviceRangeV1,
        bytes: Vec<u8>,
    },
}

impl Default for FakeHostDeviceV1 {
    fn default() -> Self {
        Self::new()
    }
}

impl FakeHostDeviceV1 {
    pub fn set_offload_query_delay(&mut self, arena: DeviceObjectV1, polls: usize) {
        self.offload
            .get_mut(&arena.get())
            .expect("live arena")
            .query_delay = polls;
    }

    pub const fn new() -> Self {
        Self {
            offload: BTreeMap::new(),
            objects: BTreeMap::new(),
            next_identity: 1,
            sampler_buffers: BTreeMap::new(),
            logs: Vec::new(),
            launches: Vec::new(),
            launch_dispatches: Vec::new(),
            launch_arguments: Vec::new(),
            custom_operations: Vec::new(),
            synchronizations: 0,
            largest_host_to_device_copy_bytes: 0,
            host_to_device_copy_bytes: 0,
            deferred_refusals: BTreeMap::new(),
            refusals: BTreeSet::new(),
            allocation_alignment: DEFAULT_FAKE_ALLOCATION_ALIGNMENT_V1,
            device_memory_bytes: DEFAULT_FAKE_DEVICE_MEMORY_BYTES_V1,
            deferred: Vec::new(),
            defer_transfers: false,
            fill_chunk_bytes: 4096,
            fill_streams: BTreeSet::new(),
            skip_collective_ordering: false,
            accept_launches: false,
            cluster_residency: None,
            sparse_probe_allocations: false,
            pool_peer_access: BTreeSet::new(),
            context_peer_access: BTreeSet::new(),
            peer_access_edges: BTreeMap::new(),
            pool_by_context: BTreeMap::new(),
            ledger: HostDeviceRankLedgerV1::new(),
            context_of: BTreeMap::new(),
            enabled_pool_peer_access: Vec::new(),
            disabled_pool_peer_access: Vec::new(),
            enabled_context_peer_access: Vec::new(),
            disabled_context_peer_access: Vec::new(),
            mapped_host: BTreeMap::new(),
        }
    }

    /// Every live context this backend issued, in the order it issued them.
    ///
    /// A device shared with another user is armed through this: a test names
    /// the objects the host created and installs the grants that user made.
    #[must_use]
    pub fn context_objects(&self) -> Vec<DeviceObjectV1> {
        self.objects_of_kind(FakeObjectKindV1::Context)
    }

    /// Every live memory pool this backend issued, in the order it issued them.
    #[must_use]
    pub fn memory_pool_objects(&self) -> Vec<DeviceObjectV1> {
        self.objects_of_kind(FakeObjectKindV1::MemoryPool)
    }

    /// Every live allocation, in the order this backend made them.
    #[must_use]
    pub fn buffer_objects(&self) -> Vec<DeviceObjectV1> {
        self.objects_of_kind(FakeObjectKindV1::Buffer)
    }

    /// Every live mapped host allocation, in the order this backend made them.
    #[must_use]
    pub fn mapped_host_objects(&self) -> Vec<DeviceObjectV1> {
        self.mapped_host
            .keys()
            .map(|identity| DeviceObjectV1::new(*identity))
            .collect()
    }

    fn objects_of_kind(&self, kind: FakeObjectKindV1) -> Vec<DeviceObjectV1> {
        self.objects
            .iter()
            .filter(|(_, object)| object.kind == kind)
            .map(|(identity, _)| DeviceObjectV1::new(*identity))
            .collect()
    }

    /// Installs one pool peer grant another user of the device already made.
    pub fn install_memory_pool_peer_access(
        &mut self,
        pool: DeviceObjectV1,
        peer_context: DeviceObjectV1,
    ) {
        self.pool_peer_access
            .insert((pool.get(), peer_context.get()));
    }

    /// Installs one context peer edge another user of the devices already made.
    pub fn install_context_peer_access(
        &mut self,
        owner_context: DeviceObjectV1,
        peer_context: DeviceObjectV1,
    ) {
        self.context_peer_access
            .insert((owner_context.get(), peer_context.get()));
    }

    /// Whether one pool currently grants one peer context access.
    #[must_use]
    pub fn has_memory_pool_peer_access(
        &self,
        pool: DeviceObjectV1,
        peer_context: DeviceObjectV1,
    ) -> bool {
        self.pool_peer_access
            .contains(&(pool.get(), peer_context.get()))
    }

    /// Whether one directed context peer edge is currently enabled.
    #[must_use]
    pub fn has_context_peer_access(
        &self,
        owner_context: DeviceObjectV1,
        peer_context: DeviceObjectV1,
    ) -> bool {
        self.context_peer_access
            .contains(&(owner_context.get(), peer_context.get()))
    }

    /// Every pool grant this backend was asked to enable, in order.
    #[must_use]
    pub fn enabled_memory_pool_peer_access(&self) -> &[(u64, u64)] {
        &self.enabled_pool_peer_access
    }

    /// Every pool grant this backend was asked to disable, in order.
    #[must_use]
    pub fn disabled_memory_pool_peer_access(&self) -> &[(u64, u64)] {
        &self.disabled_pool_peer_access
    }

    /// Every context edge this backend was asked to enable, in order.
    #[must_use]
    pub fn enabled_context_peer_access(&self) -> &[(u64, u64)] {
        &self.enabled_context_peer_access
    }

    /// Every context edge this backend was asked to disable, in order.
    #[must_use]
    pub fn disabled_context_peer_access(&self) -> &[(u64, u64)] {
        &self.disabled_context_peer_access
    }

    /// Sets the strongest alignment this backend's allocations satisfy.
    ///
    /// A real allocator's guarantee is a property of the allocator, not of the
    /// request, and a host that assumed otherwise would hand a bundle a buffer
    /// it cannot use.
    pub fn set_allocation_alignment(&mut self, alignment: u64) {
        self.allocation_alignment = alignment;
    }

    /// Sets the chunk a filled upload is staged in, so a test can split one
    /// across chunks.
    pub fn set_fill_chunk_bytes(&mut self, bytes: usize) {
        assert!(bytes > 0, "a fill chunk holds a byte");
        self.fill_chunk_bytes = bytes;
    }

    /// Sets the memory each device reports before its allocations, so a
    /// sparse probe of a full-size model can reach its serving KV pool.
    pub fn set_device_memory_bytes(&mut self, bytes: u64) {
        self.device_memory_bytes = bytes;
    }

    /// Makes host transfers finish only at the next stream synchronization.
    ///
    /// This is what an unsound backend does: it returns while the driver is
    /// still reading the caller's bytes. Nothing in production selects it, and
    /// [`crate::assert_host_transfer_contract_v1`] must reject a backend that
    /// has it on.
    pub fn defer_host_transfers(&mut self, defer: bool) {
        self.defer_transfers = defer;
    }

    /// Reports a checked launch as accepted rather than as unsupported.
    ///
    /// There is no device here and nothing runs, so this changes only the
    /// answer: a launch whose arguments and handles all checked out comes back
    /// as ordered rather than refused. That is what lets a whole session, whose
    /// shape is acquire, launch, wait, release, be driven end to end against
    /// the real service layer with no device at all. The arguments are still
    /// recorded, so what the launch asked for is still observable.
    pub fn accept_launches(&mut self, accept: bool) {
        self.accept_launches = accept;
    }

    /// Reports `residency` for every cluster residency query, as a device
    /// with that shape would.
    pub fn report_cluster_residency(
        &mut self,
        residency: runtime_bundle_abi::BundleRawClusterResidencyV1,
    ) {
        self.cluster_residency = Some(residency);
    }

    /// Preserves allocation lengths without allocating their backing bytes.
    pub fn use_sparse_probe_allocations(&mut self, sparse: bool) {
        self.sparse_probe_allocations = sparse;
    }

    /// Makes a collective read rank buffers without first ordering its stream.
    ///
    /// This is the mistake a backend that stages through some other stream
    /// makes, and it is invisible in the result: the reduction is a plausible
    /// number computed from stale bytes. Nothing in production selects it, and
    /// [`crate::assert_collective_ordering_contract_v1`] must reject a backend
    /// that has it on.
    pub fn skip_collective_stream_ordering(&mut self, skip: bool) {
        self.skip_collective_ordering = skip;
    }

    /// Performs every transfer this backend deferred on one stream.
    fn drain_deferred(&mut self, stream: DeviceObjectV1) -> HostDeviceResultV1<()> {
        let ready = std::mem::take(&mut self.deferred);
        let (mine, others) = ready
            .into_iter()
            .partition::<Vec<_>, _>(|(owner, _)| *owner == stream.get());
        self.deferred = others;
        for (_, transfer) in mine {
            match transfer {
                DeferredTransferV1::HostToDevice { range, bytes } => {
                    self.write_range(range, &bytes)?;
                }
            }
        }
        Ok(())
    }

    /// Makes this backend refuse one operation until it is allowed again.
    pub fn refuse(&mut self, refusal: FakeRefusalV1) {
        self.refusals.insert(refusal);
    }

    /// Lets one previously refused operation succeed again.
    pub fn allow(&mut self, refusal: FakeRefusalV1) {
        self.refusals.remove(&refusal);
        self.deferred_refusals.remove(&refusal);
    }

    /// Refuses one operation once, after `allowed` more of them succeed.
    pub fn refuse_after(&mut self, refusal: FakeRefusalV1, allowed: usize) {
        self.deferred_refusals.insert(refusal, allowed);
    }

    /// How many stream waits this backend has been asked to perform.
    #[must_use]
    pub const fn synchronizations(&self) -> usize {
        self.synchronizations
    }

    /// Largest host source slice passed to one device upload.
    #[must_use]
    pub const fn largest_host_to_device_copy_bytes(&self) -> usize {
        self.largest_host_to_device_copy_bytes
    }

    /// Every host byte this backend has been asked to upload.
    ///
    /// Counted where the source slice arrives, before the range check and
    /// before any deferral, so a test can prove a refusal exposed nothing.
    #[must_use]
    pub const fn host_to_device_copy_bytes(&self) -> u64 {
        self.host_to_device_copy_bytes
    }

    fn check(&mut self, refusal: FakeRefusalV1) -> HostDeviceResultV1<()> {
        if self.refusals.contains(&refusal) {
            return Err(BundleStatusV1::HostError);
        }
        // A deferred refusal lets the attempts before it succeed, so a failure
        // can be placed exactly where a test needs it: after the wait that
        // precedes a callback, for instance, rather than before it.
        match self.deferred_refusals.get_mut(&refusal) {
            Some(0) => {
                self.deferred_refusals.remove(&refusal);
                Err(BundleStatusV1::HostError)
            }
            Some(remaining) => {
                *remaining -= 1;
                Ok(())
            }
            None => Ok(()),
        }
    }

    /// Returns every diagnostic line the bundle emitted, oldest first.
    #[must_use]
    pub fn logs(&self) -> &[Vec<u8>] {
        &self.logs
    }

    /// Returns the symbol name of every launch this backend refused.
    #[must_use]
    pub fn refused_launches(&self) -> &[Vec<u8>] {
        &self.launches
    }

    /// Returns how each refused launch would have been dispatched, in order.
    ///
    /// No kernel runs here, so this is what a launch's flags asked a real
    /// backend to do, recorded where a test can read it without a device.
    #[must_use]
    pub fn refused_launch_dispatches(&self) -> &[LaunchDispatchV1] {
        &self.launch_dispatches
    }

    /// The resolved arguments of every launch this backend refused, in order.
    ///
    /// A launch is refused because nothing here runs a kernel, but the argument
    /// list it was given is the part a caller can be wrong about, so it is
    /// recorded exactly as the services layer resolved it.
    #[must_use]
    pub fn refused_launch_arguments(&self) -> &[Vec<DeviceArgumentV1>] {
        &self.launch_arguments
    }

    /// The identifier and lent ranges of every custom operation this ran.
    ///
    /// A refused call records nothing, because it invoked nothing.
    #[must_use]
    pub fn custom_operations(&self) -> &[(Vec<u8>, Vec<LentRangeV1>)] {
        &self.custom_operations
    }

    /// Returns the current bytes of one live allocation.
    #[must_use]
    pub fn buffer_bytes(&self, buffer: DeviceObjectV1) -> Option<&[u8]> {
        self.objects
            .get(&buffer.get())
            .filter(|object| object.kind == FakeObjectKindV1::Buffer)
            .map(|object| object.bytes.as_slice())
    }

    /// Returns the number of allocations this backend has not released.
    #[must_use]
    pub fn live_object_count(&self) -> usize {
        self.objects.len()
    }

    /// Includes released objects, so a test can detect temporary allocations.
    #[cfg(test)]
    pub(crate) fn issued_object_count(&self) -> u64 {
        self.next_identity - 1
    }

    fn create(
        &mut self,
        kind: FakeObjectKindV1,
        bytes: Vec<u8>,
    ) -> HostDeviceResultV1<DeviceObjectV1> {
        let identity = self.next_identity;
        self.next_identity = self
            .next_identity
            .checked_add(1)
            .ok_or(BundleStatusV1::ResourceExhausted)?;
        let logical_byte_length = bytes.len() as u64;
        self.objects.insert(
            identity,
            FakeObjectV1 {
                kind,
                bytes,
                logical_byte_length,
            },
        );
        Ok(DeviceObjectV1::new(identity))
    }

    fn create_sparse(
        &mut self,
        kind: FakeObjectKindV1,
        logical_byte_length: u64,
    ) -> HostDeviceResultV1<DeviceObjectV1> {
        let identity = self.next_identity;
        self.next_identity = self
            .next_identity
            .checked_add(1)
            .ok_or(BundleStatusV1::ResourceExhausted)?;
        self.objects.insert(
            identity,
            FakeObjectV1 {
                kind,
                bytes: Vec::new(),
                logical_byte_length,
            },
        );
        Ok(DeviceObjectV1::new(identity))
    }

    fn require(
        &self,
        object: DeviceObjectV1,
        kind: FakeObjectKindV1,
    ) -> HostDeviceResultV1<&FakeObjectV1> {
        let entry = self
            .objects
            .get(&object.get())
            .ok_or(BundleStatusV1::StaleHandle)?;
        if entry.kind == kind {
            Ok(entry)
        } else {
            Err(BundleStatusV1::InvalidArgument)
        }
    }

    fn destroy(
        &mut self,
        object: DeviceObjectV1,
        kind: FakeObjectKindV1,
    ) -> HostDeviceResultV1<()> {
        self.require(object, kind)?;
        self.objects.remove(&object.get());
        Ok(())
    }

    /// Refuses one object that belongs to some other context than this one.
    ///
    /// An object this backend has no owner recorded for is refused as well: a
    /// lend is only safe when the context it runs in is known to reach the
    /// memory, and an unknown owner is not that.
    fn require_context_owns(
        &self,
        context: DeviceObjectV1,
        object: DeviceObjectV1,
    ) -> HostDeviceResultV1<()> {
        match self.context_of.get(&object.get()) {
            Some(owner) if *owner == context.get() => Ok(()),
            _ => Err(BundleStatusV1::InvalidArgument),
        }
    }

    /// The address this backend has for one already checked range.
    fn range_address(&self, range: DeviceRangeV1) -> HostDeviceResultV1<u64> {
        let (start, end) = self.range(range)?;
        let object = self.require(range.object, FakeObjectKindV1::Buffer)?;
        if self.sparse_probe_allocations {
            return range
                .object
                .get()
                .checked_shl(32)
                .and_then(|base| base.checked_add(range.offset))
                .ok_or(BundleStatusV1::ResourceExhausted);
        }
        Ok(object.bytes[start..end].as_ptr() as u64)
    }

    fn range(&self, range: DeviceRangeV1) -> HostDeviceResultV1<(usize, usize)> {
        let object = self.require(range.object, FakeObjectKindV1::Buffer)?;
        let start = usize::try_from(range.offset).map_err(|_| BundleStatusV1::InvalidArgument)?;
        let length = usize::try_from(range.length).map_err(|_| BundleStatusV1::InvalidArgument)?;
        let end = start
            .checked_add(length)
            .ok_or(BundleStatusV1::InvalidArgument)?;
        if end as u64 > object.logical_byte_length {
            return Err(BundleStatusV1::InvalidArgument);
        }
        Ok((start, end))
    }

    /// Checks one tensor map against the range it names, without encoding it.
    ///
    /// The range is resolved and bounds checked first. What is then checked is
    /// what a tiled encoding requires and this backend can decide without a
    /// driver: the map's footprint fits inside the range it was given, and the
    /// range starts where a tiled map may start. The element type and swizzle
    /// arrived as closed enums, so neither can be wrong by the time it is here.
    fn check_tensor_map(&self, map: &DeviceTensorMapV1) -> HostDeviceResultV1<()> {
        self.range(map.range)?;
        let raw = BundleTensorMapV1 {
            struct_size: std::mem::size_of::<BundleTensorMapV1>() as u32,
            rank: map.rank,
            element_type: map.element_type.code(),
            swizzle: map.swizzle.code(),
            interleave: map.interleave,
            l2_promotion: map.l2_promotion,
            oob_fill: map.oob_fill,
            reserved: 0,
            buffer: BundleDeviceBufferViewV1 {
                struct_size: std::mem::size_of::<BundleDeviceBufferViewV1>() as u32,
                reserved: 0,
                buffer: BundleDeviceBufferHandleV1::from_parts(1, 1),
                offset: map.range.offset,
                length: map.range.length,
            },
            global_dims: map.global_dims,
            global_strides_bytes: map.global_strides_bytes,
            box_dims: map.box_dims,
            element_strides: map.element_strides,
        };
        validate_tensor_map_v1(raw).map_err(|_| BundleStatusV1::InvalidArgument)
    }

    fn read_range(&self, range: DeviceRangeV1) -> HostDeviceResultV1<Vec<u8>> {
        let (start, end) = self.range(range)?;
        let object = self.require(range.object, FakeObjectKindV1::Buffer)?;
        if self.sparse_probe_allocations {
            return Ok(vec![0; end - start]);
        }
        Ok(object.bytes[start..end].to_vec())
    }

    fn write_range(&mut self, range: DeviceRangeV1, source: &[u8]) -> HostDeviceResultV1<()> {
        let (start, end) = self.range(range)?;
        if end - start != source.len() {
            return Err(BundleStatusV1::InvalidArgument);
        }
        if self.sparse_probe_allocations {
            return Ok(());
        }
        let object = self
            .objects
            .get_mut(&range.object.get())
            .ok_or(BundleStatusV1::StaleHandle)?;
        object.bytes[start..end].copy_from_slice(source);
        Ok(())
    }
}

impl HostDeviceBackendV1 for FakeHostDeviceV1 {
    fn create_offload_arena(
        &mut self,
        buffer: DeviceObjectV1,
        page_bytes: usize,
        pages: usize,
    ) -> HostDeviceResultV1<DeviceObjectV1> {
        self.require(buffer, FakeObjectKindV1::Buffer)?;
        let size = page_bytes
            .checked_mul(pages)
            .filter(|_| page_bytes != 0 && pages != 0)
            .ok_or(BundleStatusV1::InvalidArgument)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(size)
            .map_err(|_| BundleStatusV1::ResourceExhausted)?;
        bytes.resize(size, 0);
        let id = self.create(FakeObjectKindV1::Stream, Vec::new())?;
        self.offload.insert(
            id.get(),
            FakeOffloadArena {
                buffer,
                bytes,
                page_bytes,
                pending: BTreeMap::new(),
                next: 1,
                query_delay: 1,
            },
        );
        Ok(id)
    }
    fn submit_offload_chunk(
        &mut self,
        arena: DeviceObjectV1,
        direction: runtime_kv::KvTransferDirection,
        pages: &[runtime_kv::KvPagePair],
    ) -> HostDeviceResultV1<u64> {
        self.check(FakeRefusalV1::SubmitOffload)?;
        let a = self
            .offload
            .get_mut(&arena.get())
            .ok_or(BundleStatusV1::StaleHandle)?;
        let device = self
            .objects
            .get(&a.buffer.get())
            .ok_or(BundleStatusV1::StaleHandle)?;
        let mut hosts = BTreeSet::new();
        let mut devices = BTreeSet::new();
        if a.pending.len() >= 128
            || a.pending
                .values()
                .map(|(_, p, _)| p.len())
                .sum::<usize>()
                .saturating_add(pages.len())
                .saturating_mul(a.page_bytes)
                > 1024 * 1024 * 1024
            || pages.is_empty()
            || pages.len() > (8 * 1024 * 1024 / a.page_bytes).max(1)
        {
            return Err(BundleStatusV1::InvalidArgument);
        }
        for p in pages {
            if p.host.index() >= a.bytes.len() / a.page_bytes
                || p.device.index() >= device.bytes.len() / a.page_bytes
                || !hosts.insert(p.host)
                || !devices.insert(p.device)
                || a.pending
                    .values()
                    .any(|(_, old, _)| old.iter().any(|q| q.host == p.host || q.device == p.device))
            {
                return Err(BundleStatusV1::InvalidArgument);
            }
        }
        let id = a.next;
        a.next = a
            .next
            .checked_add(1)
            .ok_or(BundleStatusV1::ResourceExhausted)?;
        a.pending
            .insert(id, (direction, pages.to_vec(), a.query_delay));
        Ok(id)
    }
    fn query_offload_chunk(
        &mut self,
        arena: DeviceObjectV1,
        chunk: u64,
    ) -> HostDeviceResultV1<bool> {
        self.check(FakeRefusalV1::QueryOffload)?;
        let a = self
            .offload
            .get_mut(&arena.get())
            .ok_or(BundleStatusV1::StaleHandle)?;
        let (_, _, polled) = a
            .pending
            .get_mut(&chunk)
            .ok_or(BundleStatusV1::StaleHandle)?;
        if *polled != 0 {
            *polled -= 1;
            return Ok(false);
        }
        let (direction, pages, _) = a.pending.remove(&chunk).unwrap();
        let device = self
            .objects
            .get_mut(&a.buffer.get())
            .ok_or(BundleStatusV1::StaleHandle)?;
        for p in pages {
            let host =
                &mut a.bytes[p.host.index() * a.page_bytes..(p.host.index() + 1) * a.page_bytes];
            let device = &mut device.bytes
                [p.device.index() * a.page_bytes..(p.device.index() + 1) * a.page_bytes];
            match direction {
                runtime_kv::KvTransferDirection::Out => host.copy_from_slice(device),
                runtime_kv::KvTransferDirection::In => device.copy_from_slice(host),
            }
        }
        Ok(true)
    }
    fn fence_offload_arena(&mut self, arena: DeviceObjectV1) -> HostDeviceResultV1<()> {
        self.check(FakeRefusalV1::SynchronizeStream)?;
        let ids: Vec<_> = self
            .offload
            .get(&arena.get())
            .ok_or(BundleStatusV1::StaleHandle)?
            .pending
            .keys()
            .copied()
            .collect();
        for id in ids {
            while !self.query_offload_chunk(arena, id)? {}
        }
        Ok(())
    }
    fn release_offload_arena(&mut self, arena: DeviceObjectV1) -> HostDeviceResultV1<()> {
        self.fence_offload_arena(arena)?;
        self.offload.remove(&arena.get());
        self.objects.remove(&arena.get());
        Ok(())
    }
    fn reserve_sampling_workspace(
        &mut self,
        context: DeviceObjectV1,
        vocabulary: usize,
    ) -> HostDeviceResultV1<()> {
        self.require(context, FakeObjectKindV1::Context)?;
        if let Some((reserved, _)) = self.sampler_buffers.get(&context.get()) {
            return if *reserved == vocabulary {
                Ok(())
            } else {
                Err(BundleStatusV1::InvalidArgument)
            };
        }
        let bytes = crate::sampling_workspace::SamplingWorkspace::new(vocabulary)?.device_bytes();
        if self.free_device_memory(context)? < bytes {
            return Err(BundleStatusV1::ResourceExhausted);
        }
        let plan = self.ledger.plan_allocation(Some(context.get()), bytes)?;
        // No kernels consume fake sampler memory, but its full allocation is
        // charged to the same ledger as KV, including in non-sparse fixtures.
        let buffer = self.create_sparse(FakeObjectKindV1::Buffer, bytes)?;
        self.ledger.commit_allocation(buffer.get(), plan);
        self.sampler_buffers
            .insert(context.get(), (vocabulary, buffer));
        Ok(())
    }

    fn free_device_memory(&self, context: DeviceObjectV1) -> HostDeviceResultV1<u64> {
        self.require(context, FakeObjectKindV1::Context)?;
        let allocated = self
            .ledger
            .allocated_on_context_device(context.get())
            .ok_or(BundleStatusV1::InvalidState)?;
        Ok(self.device_memory_bytes.saturating_sub(allocated))
    }
    fn uses_sparse_probe_allocations(&self) -> bool {
        self.sparse_probe_allocations
    }

    fn create_context(
        &mut self,
        rank: u32,
        device_ordinal: i32,
    ) -> HostDeviceResultV1<DeviceObjectV1> {
        if device_ordinal < 0 {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let context = self.create(FakeObjectKindV1::Context, Vec::new())?;
        // The rank a context was created for is what a per-rank memory report
        // names, so this backend remembers it instead of inferring one later.
        self.ledger
            .attach_context(context.get(), rank, device_ordinal);
        Ok(context)
    }

    fn release_context(&mut self, context: DeviceObjectV1) -> HostDeviceResultV1<()> {
        self.check(FakeRefusalV1::ReleaseContext)?;
        if let Some((_, buffer)) = self.sampler_buffers.get(&context.get()).copied() {
            self.release_device_buffer(buffer)?;
            self.sampler_buffers.remove(&context.get());
        }

        // Whatever this context still holds becomes unattributed rather than
        // disappearing, so a refused release changes nothing at all.
        let plan = self.ledger.plan_context_release(context.get())?;
        self.destroy(context, FakeObjectKindV1::Context)?;
        self.ledger.commit_context_release(plan);
        Ok(())
    }

    fn create_memory_pool(
        &mut self,
        context: DeviceObjectV1,
    ) -> HostDeviceResultV1<DeviceObjectV1> {
        self.require(context, FakeObjectKindV1::Context)?;
        // A context's pool is one object, the way a driver's default pool is.
        // Handing out a second one would hide an aliasing mistake rather than
        // reproduce the device.
        if let Some(pool) = self.pool_by_context.get(&context.get()).copied() {
            return Ok(DeviceObjectV1::new(pool));
        }
        let pool = self.create(FakeObjectKindV1::MemoryPool, Vec::new())?;
        self.pool_by_context.insert(context.get(), pool.get());
        self.context_of.insert(pool.get(), context.get());
        Ok(pool)
    }

    fn memory_pool_peer_access(
        &mut self,
        pool: DeviceObjectV1,
        peer_context: DeviceObjectV1,
    ) -> HostDeviceResultV1<bool> {
        self.require(pool, FakeObjectKindV1::MemoryPool)?;
        self.require(peer_context, FakeObjectKindV1::Context)?;
        Ok(self
            .pool_peer_access
            .contains(&(pool.get(), peer_context.get())))
    }

    fn set_memory_pool_peer_access(
        &mut self,
        pool: DeviceObjectV1,
        peer_context: DeviceObjectV1,
        enabled: bool,
    ) -> HostDeviceResultV1<()> {
        self.require(pool, FakeObjectKindV1::MemoryPool)?;
        self.require(peer_context, FakeObjectKindV1::Context)?;
        let edge = (pool.get(), peer_context.get());
        if enabled {
            self.check(FakeRefusalV1::EnableMemoryPoolPeerAccess)?;
            self.pool_peer_access.insert(edge);
            self.enabled_pool_peer_access.push(edge);
        } else {
            self.check(FakeRefusalV1::DisableMemoryPoolPeerAccess)?;
            self.pool_peer_access.remove(&edge);
            self.disabled_pool_peer_access.push(edge);
        }
        Ok(())
    }

    fn release_memory_pool(&mut self, pool: DeviceObjectV1) -> HostDeviceResultV1<()> {
        self.check(FakeRefusalV1::ReleaseMemoryPool)?;
        self.destroy(pool, FakeObjectKindV1::MemoryPool)?;
        self.pool_by_context.retain(|_, held| *held != pool.get());
        Ok(())
    }

    fn allocate_device_buffer(
        &mut self,
        pool: DeviceObjectV1,
        stream: Option<DeviceObjectV1>,
        byte_length: u64,
        alignment: u64,
    ) -> HostDeviceResultV1<DeviceObjectV1> {
        self.check(FakeRefusalV1::AllocateDeviceBuffer)?;
        self.require(pool, FakeObjectKindV1::MemoryPool)?;
        if let Some(stream) = stream {
            self.require(stream, FakeObjectKindV1::Stream)?;
        }
        if byte_length == 0 || alignment == 0 || !alignment.is_power_of_two() {
            return Err(BundleStatusV1::InvalidArgument);
        }
        // An allocation belongs to the context its pool does, the way a driver
        // allocation belongs to the context it was made in. Every count it
        // changes is settled before the object exists, so a refusal here leaves
        // nothing allocated and nothing counted.
        let owner = self.context_of.get(&pool.get()).copied();
        let plan = self.ledger.plan_allocation(owner, byte_length)?;
        let buffer = if self.sparse_probe_allocations {
            self.create_sparse(FakeObjectKindV1::Buffer, byte_length)?
        } else {
            let length =
                usize::try_from(byte_length).map_err(|_| BundleStatusV1::ResourceExhausted)?;
            let mut bytes = Vec::new();
            bytes
                .try_reserve_exact(length)
                .map_err(|_| BundleStatusV1::ResourceExhausted)?;
            bytes.resize(length, 0);
            self.create(FakeObjectKindV1::Buffer, bytes)?
        };
        if let Some(context) = owner {
            self.context_of.insert(buffer.get(), context);
        }
        self.ledger.commit_allocation(buffer.get(), plan);
        Ok(buffer)
    }

    fn allocate_mapped_host_buffer(
        &mut self,
        stream: DeviceObjectV1,
        byte_length: u64,
        fill: &mut dyn FnMut(&mut [u8]) -> bool,
    ) -> HostDeviceResultV1<DeviceObjectV1> {
        self.check(FakeRefusalV1::AllocateDeviceBuffer)?;
        self.require(stream, FakeObjectKindV1::Stream)?;
        if byte_length == 0 {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let buffer = if self.sparse_probe_allocations {
            self.create_sparse(FakeObjectKindV1::Buffer, byte_length)?
        } else {
            let length =
                usize::try_from(byte_length).map_err(|_| BundleStatusV1::ResourceExhausted)?;
            let mut bytes = Vec::new();
            bytes
                .try_reserve_exact(length)
                .map_err(|_| BundleStatusV1::ResourceExhausted)?;
            bytes.resize(length, 0);
            if !fill(&mut bytes) {
                return Err(BundleStatusV1::HostError);
            }
            self.create(FakeObjectKindV1::Buffer, bytes)?
        };
        if let Some(context) = self.context_of.get(&stream.get()).copied() {
            self.context_of.insert(buffer.get(), context);
        }
        self.mapped_host.insert(buffer.get(), byte_length);
        Ok(buffer)
    }

    fn release_device_buffer(&mut self, buffer: DeviceObjectV1) -> HostDeviceResultV1<()> {
        self.check(FakeRefusalV1::ReleaseDeviceBuffer)?;
        if self.mapped_host.contains_key(&buffer.get()) {
            self.destroy(buffer, FakeObjectKindV1::Buffer)?;
            self.mapped_host.remove(&buffer.get());
            return Ok(());
        }
        let byte_length = self
            .objects
            .get(&buffer.get())
            .ok_or(BundleStatusV1::StaleHandle)?
            .logical_byte_length;
        let plan = self.ledger.plan_release(buffer.get(), byte_length)?;
        self.destroy(buffer, FakeObjectKindV1::Buffer)?;
        self.ledger.commit_release(buffer.get(), plan);
        Ok(())
    }

    fn allocation_satisfies_alignment(
        &self,
        buffer: DeviceObjectV1,
        alignment: u64,
    ) -> HostDeviceResultV1<bool> {
        self.require(buffer, FakeObjectKindV1::Buffer)?;
        // Host memory has no device address, so this backend answers with the
        // guarantee it was configured to make.
        Ok(alignment <= self.allocation_alignment)
    }

    fn acquire_peer_access(
        &mut self,
        owner_context: DeviceObjectV1,
        peer_context: DeviceObjectV1,
    ) -> HostDeviceResultV1<PeerAccessAcquisitionV1> {
        self.require(owner_context, FakeObjectKindV1::Context)?;
        self.require(peer_context, FakeObjectKindV1::Context)?;
        if owner_context == peer_context {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let edge = (owner_context.get(), peer_context.get());
        let host_acquired = !self.context_peer_access.contains(&edge);
        // The record is reserved before the transition, so a backend that
        // cannot record one never has an enable to undo.
        let object = self.create(FakeObjectKindV1::PeerAccess, Vec::new())?;
        if host_acquired {
            if let Err(status) = self.check(FakeRefusalV1::EnableContextPeerAccess) {
                self.objects.remove(&object.get());
                return Err(status);
            }
            self.context_peer_access.insert(edge);
            self.enabled_context_peer_access.push(edge);
        }
        // The record remembers the edge, so a release that disables it names
        // the same pair this acquire enabled.
        self.peer_access_edges.insert(object.get(), edge);
        Ok(PeerAccessAcquisitionV1 {
            object,
            host_acquired,
        })
    }

    fn release_peer_access(
        &mut self,
        peer_access: DeviceObjectV1,
        disable: bool,
    ) -> HostDeviceResultV1<()> {
        self.check(FakeRefusalV1::ReleasePeerAccess)?;
        let edge = self
            .peer_access_edges
            .get(&peer_access.get())
            .copied()
            .ok_or(BundleStatusV1::StaleHandle)?;
        self.destroy(peer_access, FakeObjectKindV1::PeerAccess)?;
        self.peer_access_edges.remove(&peer_access.get());
        if disable {
            self.context_peer_access.remove(&edge);
            self.disabled_context_peer_access.push(edge);
        }
        Ok(())
    }

    fn create_stream(
        &mut self,
        context: DeviceObjectV1,
        _stream_kind: u32,
    ) -> HostDeviceResultV1<DeviceObjectV1> {
        self.require(context, FakeObjectKindV1::Context)?;
        let stream = self.create(FakeObjectKindV1::Stream, Vec::new())?;
        self.context_of.insert(stream.get(), context.get());
        Ok(stream)
    }

    fn synchronize_stream(&mut self, stream: DeviceObjectV1) -> HostDeviceResultV1<()> {
        self.require(stream, FakeObjectKindV1::Stream)?;
        self.check(FakeRefusalV1::SynchronizeStream)?;
        self.synchronizations += 1;
        self.drain_deferred(stream)
    }

    fn release_stream(&mut self, stream: DeviceObjectV1) -> HostDeviceResultV1<()> {
        self.check(FakeRefusalV1::ReleaseStream)?;
        self.destroy(stream, FakeObjectKindV1::Stream)?;
        // A released stream's staging chunk goes with it.
        self.fill_streams.remove(&stream.get());
        Ok(())
    }

    fn begin_graph_capture(&mut self, stream: DeviceObjectV1) -> HostDeviceResultV1<()> {
        self.require(stream, FakeObjectKindV1::Stream).map(|_| ())
    }

    fn end_graph_capture(
        &mut self,
        stream: DeviceObjectV1,
    ) -> Result<DeviceObjectV1, GraphCaptureErrorV1> {
        self.require(stream, FakeObjectKindV1::Stream)
            .map_err(GraphCaptureErrorV1::Capturing)?;
        self.check(FakeRefusalV1::EndGraphCapture)
            .map_err(GraphCaptureErrorV1::Capturing)?;
        self.check(FakeRefusalV1::InstantiateGraph)
            .map_err(GraphCaptureErrorV1::Ended)?;
        self.create(FakeObjectKindV1::Graph, Vec::new())
            .map_err(GraphCaptureErrorV1::Ended)
    }

    fn launch_graph(
        &mut self,
        stream: DeviceObjectV1,
        graph: DeviceObjectV1,
    ) -> HostDeviceResultV1<()> {
        self.require(stream, FakeObjectKindV1::Stream)?;
        self.require(graph, FakeObjectKindV1::Graph).map(|_| ())
    }

    fn release_graph(&mut self, graph: DeviceObjectV1) -> HostDeviceResultV1<()> {
        self.check(FakeRefusalV1::ReleaseGraph)?;
        self.destroy(graph, FakeObjectKindV1::Graph)
    }

    fn create_event(&mut self, context: DeviceObjectV1) -> HostDeviceResultV1<DeviceObjectV1> {
        self.require(context, FakeObjectKindV1::Context)?;
        self.create(FakeObjectKindV1::Event, Vec::new())
    }

    fn record_event(
        &mut self,
        event: DeviceObjectV1,
        stream: DeviceObjectV1,
    ) -> HostDeviceResultV1<()> {
        self.require(event, FakeObjectKindV1::Event)?;
        self.require(stream, FakeObjectKindV1::Stream)?;
        Ok(())
    }

    fn wait_event(
        &mut self,
        stream: DeviceObjectV1,
        event: DeviceObjectV1,
    ) -> HostDeviceResultV1<()> {
        self.require(stream, FakeObjectKindV1::Stream)?;
        self.require(event, FakeObjectKindV1::Event)?;
        Ok(())
    }

    fn synchronize_event(&mut self, event: DeviceObjectV1) -> HostDeviceResultV1<()> {
        self.require(event, FakeObjectKindV1::Event).map(|_| ())
    }

    fn release_event(&mut self, event: DeviceObjectV1) -> HostDeviceResultV1<()> {
        self.check(FakeRefusalV1::ReleaseEvent)?;
        self.destroy(event, FakeObjectKindV1::Event)
    }

    /// Stages each chunk as a real ring would: filled in place, then copied.
    fn fill_host_to_device_ordered(
        &mut self,
        stream: DeviceObjectV1,
        destination: DeviceRangeV1,
        fill: &mut dyn FnMut(usize, &mut [u8]),
    ) -> HostDeviceResultV1<()> {
        self.require(stream, FakeObjectKindV1::Stream)?;
        self.range(destination)?;
        self.fill_streams.insert(stream.get());
        let length =
            usize::try_from(destination.length).map_err(|_| BundleStatusV1::ResourceExhausted)?;
        let mut chunk = vec![0; self.fill_chunk_bytes];
        for at in (0..length).step_by(self.fill_chunk_bytes) {
            let bytes = self.fill_chunk_bytes.min(length - at);
            fill(at, &mut chunk[..bytes]);
            let range = DeviceRangeV1 {
                offset: destination.offset + at as u64,
                length: bytes as u64,
                ..destination
            };
            self.copy_host_to_device_ordered(stream, range, &chunk[..bytes])?;
        }
        Ok(())
    }

    fn fill_staging_bytes(&self) -> u64 {
        (self.fill_streams.len() * self.fill_chunk_bytes) as u64
    }

    fn copy_host_to_device(
        &mut self,
        stream: DeviceObjectV1,
        destination: DeviceRangeV1,
        source: &[u8],
    ) -> HostDeviceResultV1<()> {
        self.require(stream, FakeObjectKindV1::Stream)?;
        self.largest_host_to_device_copy_bytes =
            self.largest_host_to_device_copy_bytes.max(source.len());
        self.host_to_device_copy_bytes += source.len() as u64;
        // The range is checked either way, so a deferred transfer still refuses
        // a bad range at the moment the caller asked for it.
        self.range(destination)?;
        if self.defer_transfers {
            self.deferred.push((
                stream.get(),
                DeferredTransferV1::HostToDevice {
                    range: destination,
                    bytes: source.to_vec(),
                },
            ));
            return Ok(());
        }
        self.write_range(destination, source)
    }

    fn copy_device_to_host(
        &mut self,
        stream: DeviceObjectV1,
        destination: &mut [u8],
        source: DeviceRangeV1,
    ) -> HostDeviceResultV1<()> {
        self.require(stream, FakeObjectKindV1::Stream)?;
        let bytes = self.read_range(source)?;
        if bytes.len() != destination.len() {
            return Err(BundleStatusV1::InvalidArgument);
        }
        destination.copy_from_slice(&bytes);
        Ok(())
    }

    fn copy_device_to_device(
        &mut self,
        stream: DeviceObjectV1,
        destination: DeviceRangeV1,
        source: DeviceRangeV1,
    ) -> HostDeviceResultV1<()> {
        self.check(FakeRefusalV1::CopyDeviceToDevice)?;
        self.require(stream, FakeObjectKindV1::Stream)?;
        if self.sparse_probe_allocations {
            let (destination_start, destination_end) = self.range(destination)?;
            let (source_start, source_end) = self.range(source)?;
            return (destination_end - destination_start == source_end - source_start)
                .then_some(())
                .ok_or(BundleStatusV1::InvalidArgument);
        }
        let bytes = self.read_range(source)?;
        self.write_range(destination, &bytes)
    }

    fn copy_peer_to_peer(
        &mut self,
        stream: DeviceObjectV1,
        destination_context: DeviceObjectV1,
        destination: DeviceRangeV1,
        source_context: DeviceObjectV1,
        source: DeviceRangeV1,
    ) -> HostDeviceResultV1<()> {
        self.require(destination_context, FakeObjectKindV1::Context)?;
        self.require(source_context, FakeObjectKindV1::Context)?;
        self.copy_device_to_device(stream, destination, source)
    }

    fn zero_device_buffer(
        &mut self,
        stream: DeviceObjectV1,
        destination: DeviceRangeV1,
    ) -> HostDeviceResultV1<()> {
        self.check(FakeRefusalV1::ZeroDeviceBuffer)?;
        self.require(stream, FakeObjectKindV1::Stream)?;
        let (start, end) = self.range(destination)?;
        if self.sparse_probe_allocations {
            return Ok(());
        }
        let zeros = vec![0_u8; end - start];
        self.write_range(destination, &zeros)
    }

    /// Writes the address this backend has for each slot, in slot order.
    ///
    /// There is no driver here, so what this backend has for a range is the
    /// address of its own bytes. That is a real address of real storage, which
    /// is what makes the table observable in a device-free test: whatever reads
    /// it back gets the same numbers a launch would be given for those slots.
    fn write_device_pointer_table(
        &mut self,
        stream: DeviceObjectV1,
        destination: DeviceRangeV1,
        slots: &[Option<DeviceRangeV1>],
    ) -> HostDeviceResultV1<()> {
        self.check(FakeRefusalV1::WriteDevicePointerTable)?;
        self.require(stream, FakeObjectKindV1::Stream)?;
        let (start, end) = self.range(destination)?;
        let expected = slots
            .len()
            .checked_mul(DEVICE_POINTER_TABLE_ENTRY_BYTES_V1 as usize)
            .ok_or(BundleStatusV1::InvalidArgument)?;
        if slots.is_empty() || end - start != expected {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let mut table = Vec::with_capacity(expected);
        for slot in slots {
            let address = match slot {
                Some(range) => self.range_address(*range)?,
                None => 0,
            };
            table.extend_from_slice(&address.to_le_bytes());
        }
        self.write_range(destination, &table)
    }

    fn load_module(
        &mut self,
        context: DeviceObjectV1,
        module_bytes: &[u8],
    ) -> HostDeviceResultV1<DeviceObjectV1> {
        self.check(FakeRefusalV1::LoadModule)?;
        self.require(context, FakeObjectKindV1::Context)?;
        if module_bytes.is_empty() {
            return Err(BundleStatusV1::InvalidArgument);
        }
        self.create(FakeObjectKindV1::Module, module_bytes.to_vec())
    }

    fn resolve_symbol(
        &mut self,
        module: DeviceObjectV1,
        symbol_name: &[u8],
    ) -> HostDeviceResultV1<DeviceObjectV1> {
        self.check(FakeRefusalV1::ResolveSymbol)?;
        self.require(module, FakeObjectKindV1::Module)?;
        if symbol_name.is_empty() {
            return Err(BundleStatusV1::InvalidArgument);
        }
        self.create(FakeObjectKindV1::Symbol, symbol_name.to_vec())
    }

    fn configure_symbol(
        &mut self,
        symbol: DeviceObjectV1,
        _dynamic_shared_memory_bytes: u32,
    ) -> HostDeviceResultV1<()> {
        self.require(symbol, FakeObjectKindV1::Symbol).map(|_| ())
    }

    fn cluster_residency(
        &mut self,
        symbol: DeviceObjectV1,
        config: LaunchConfigV1,
    ) -> HostDeviceResultV1<runtime_bundle_abi::BundleRawClusterResidencyV1> {
        self.require(symbol, FakeObjectKindV1::Symbol)?;
        launch_cluster_v1(config)?;
        self.cluster_residency.ok_or(BundleStatusV1::Unsupported)
    }

    fn launch_symbol(
        &mut self,
        stream: DeviceObjectV1,
        symbol: DeviceObjectV1,
        config: LaunchConfigV1,
        arguments: &[DeviceArgumentV1],
    ) -> HostDeviceResultV1<()> {
        // A flag this backend cannot honour is refused before anything else,
        // exactly as the driver backend refuses it.
        let dispatch = launch_dispatch_v1(config)?;
        crate::device::launch_cooperative_v1(config)?;
        // A cluster shape no launch could take is an invalid argument, and one
        // that could is a launch this backend has no device to form. Neither is
        // recorded as a launch, so no caller can read either as one that
        // happened without the cluster it asked for.
        if launch_cluster_v1(config)?.is_some() {
            return Err(BundleStatusV1::Unsupported);
        }
        self.require(stream, FakeObjectKindV1::Stream)?;
        let name = self
            .require(symbol, FakeObjectKindV1::Symbol)?
            .bytes
            .clone();
        for argument in arguments {
            match argument {
                DeviceArgumentV1::Range(range) => {
                    self.range(*range)?;
                }
                // A tensor map is checked exactly as a driver backend checks it,
                // so a descriptor this host accepts is one that would encode.
                DeviceArgumentV1::TensorMap(map) => self.check_tensor_map(map)?,
                // Opaque bytes have no declared launch encoding in V1, and this
                // backend refuses one for the same reason a driver does.
                DeviceArgumentV1::OpaqueBytes(_) => {
                    return Err(BundleStatusV1::Unsupported);
                }
                // Every slot is checked exactly as a range argument is. This
                // backend does not decide whether the launching context can
                // reach one; that is the services layer's rule, and it applies
                // it before anything gets here.
                DeviceArgumentV1::DevicePointerArray(slots) => {
                    if slots.is_empty()
                        || slots.len() > BUNDLE_DEVICE_POINTER_ARRAY_V1_MAX_SLOTS as usize
                    {
                        return Err(BundleStatusV1::InvalidArgument);
                    }
                    for slot in slots.iter().flatten() {
                        self.range(*slot)?;
                    }
                }
                DeviceArgumentV1::NullDevicePointer
                | DeviceArgumentV1::ScalarU32(_)
                | DeviceArgumentV1::ScalarI32(_)
                | DeviceArgumentV1::ScalarU64(_) => {}
            }
        }
        self.launches.push(name);
        self.launch_dispatches.push(dispatch);
        self.launch_arguments.push(arguments.to_vec());
        self.check(FakeRefusalV1::LaunchSymbol)?;
        if self.accept_launches {
            // Nothing ran, and nothing claims to have. What is reported is only
            // that the launch was ordered, which is what a driver reports too.
            Ok(())
        } else {
            // No device runs the kernel, so the launch is refused rather than
            // reported as complete work.
            Err(BundleStatusV1::Unsupported)
        }
    }

    fn release_symbol(&mut self, symbol: DeviceObjectV1) -> HostDeviceResultV1<()> {
        self.check(FakeRefusalV1::ReleaseSymbol)?;
        self.destroy(symbol, FakeObjectKindV1::Symbol)
    }

    fn release_module(&mut self, module: DeviceObjectV1) -> HostDeviceResultV1<()> {
        self.check(FakeRefusalV1::ReleaseModule)?;
        self.destroy(module, FakeObjectKindV1::Module)
    }

    fn execute_collective(
        &mut self,
        stream: DeviceObjectV1,
        binding: &CollectiveBindingV1,
        operation: &[u8],
        arguments: &[DeviceArgumentV1],
    ) -> HostDeviceResultV1<()> {
        self.check(FakeRefusalV1::ExecuteCollective)?;
        self.require(stream, FakeObjectKindV1::Stream)?;
        // The contract and the shape of the request are decided on the shared
        // surface, so this backend and a driver backend refuse the same ones.
        let plan = plan_generic_collective_v1(binding, operation, arguments)?;
        // Whatever was ordered on this stream has to land before the rank
        // buffers are read, exactly as a driver backend owes.
        if !self.skip_collective_ordering {
            self.drain_deferred(stream)?;
        }
        // Every span is resolved before any of them is read, so a span this
        // backend cannot resolve leaves every rank's contribution where it was.
        for range in &plan.ranges {
            self.range(*range)?;
        }
        if self.sparse_probe_allocations {
            self.check(FakeRefusalV1::CompleteCollective)?;
            return Ok(());
        }
        let totals = plan.fold_lanes_v1(|range| self.read_range(range))?;
        let reduced = plan.collective.encode_lanes_v1(&totals);
        for range in &plan.ranges {
            self.write_range(*range, &reduced)?;
        }
        self.check(FakeRefusalV1::CompleteCollective)?;
        Ok(())
    }

    /// Lends host memory and a null stream to one bundle-supplied operation.
    ///
    /// There is no driver here, so what this backend has for a range is the
    /// address of its own bytes. It says so: the launch carries the host
    /// address space, and an operation whose kernels need a device refuses it.
    ///
    /// Everything else is the contract a driver backend owes, checked the same
    /// way. The stream and every operand must belong to the context the
    /// operation names, because an address from another context names memory
    /// that context cannot reach, and a host-memory backend that skipped the
    /// check would accept exactly what a device rejects. Waiting is the
    /// services layer's, before the callback and, when the operation may
    /// enqueue, after it.
    fn run_custom_operation(
        &mut self,
        context: DeviceObjectV1,
        stream: DeviceObjectV1,
        operation_id: &[u8],
        operands: &[DeviceCustomOperandV1],
        operation: &CustomOperationV1<'_>,
    ) -> HostDeviceResultV1<i32> {
        self.check(FakeRefusalV1::RunCustomOperation)?;
        self.require(context, FakeObjectKindV1::Context)?;
        self.require(stream, FakeObjectKindV1::Stream)?;
        if operation_id.is_empty() {
            return Err(BundleStatusV1::InvalidArgument);
        }
        self.require_context_owns(context, stream)?;
        let mut ranges = Vec::with_capacity(operands.len());
        for operand in operands {
            self.require_context_owns(context, operand.range.object)?;
            let (start, end) = self.range(operand.range)?;
            let object = self.require(operand.range.object, FakeObjectKindV1::Buffer)?;
            let address = object.bytes[start..end].as_ptr() as u64;
            let length = u64::try_from(end - start).map_err(|_| BundleStatusV1::InvalidArgument)?;
            ranges.push(LentRangeV1::new(address, length));
        }
        self.custom_operations
            .push((operation_id.to_vec(), ranges.clone()));
        operation.invoke(LentLaunchV1 {
            address_space: CustomAddressSpaceV1::Host,
            stream: LentStreamV1::null(),
            ranges: &ranges,
        })
    }

    fn log(&mut self, message: &[u8]) -> HostDeviceResultV1<()> {
        self.logs.push(message.to_vec());
        Ok(())
    }
}

impl crate::probe::HostDeviceMemoryUsageV1 for FakeHostDeviceV1 {
    fn current_device_memory_bytes(&self) -> u64 {
        self.ledger.current_bytes()
    }

    fn peak_device_memory_bytes(&self) -> u64 {
        self.ledger.peak_bytes()
    }

    fn materialized_device_memory_bytes(&self) -> u64 {
        self.objects
            .values()
            .map(|object| object.bytes.len() as u64)
            .sum()
    }
}

/// This backend holds no device, and says so rather than reporting a board.
///
/// The allocation counts are real, because it tracked them itself. The free and
/// total counts are absent, so nothing downstream can read a device-free run as
/// a measurement of an empty device.
impl crate::probe::HostDeviceRankMemoryUsageV1 for FakeHostDeviceV1 {
    fn rank_memory_v1(&self) -> Result<crate::probe::HostDeviceRankMemoryReportV1, String> {
        let ranks = self
            .ledger
            .ranks()?
            .into_iter()
            .map(|record| {
                crate::probe::HostDeviceRankMemoryV1::new(
                    record.rank,
                    record.device_ordinal,
                    None,
                    None,
                    record.allocated_bytes,
                    record.peak_allocated_bytes,
                )
            })
            .collect();
        Ok(crate::probe::HostDeviceRankMemoryReportV1::new(
            crate::probe::HostDeviceMemoryMeasurementV1::DeviceFree,
            ranks,
        ))
    }
}
