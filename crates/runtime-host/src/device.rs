//! The model-neutral device surface the host services are built on.
//!
//! Handle issuing, generation checks, range checks, and release ordering live
//! in [`crate::services`]. This trait carries only the operations that differ
//! between a real driver and a host-memory stand-in, and it never sees an ABI
//! handle.

pub mod mapped_gather;

use std::collections::BTreeMap;

use runtime_bundle_abi::{
    BUNDLE_LAUNCH_FLAG_V1_PROGRAMMATIC_STREAM_SERIALIZATION, BUNDLE_LAUNCH_FLAGS_V1_KNOWN,
    BUNDLE_LAUNCH_V1_MAX_CLUSTER_BLOCKS, BundleStatusV1, CustomOperationV1, LaunchClusterV1,
    LaunchConfigV1, TensorMapElementV1, TensorMapSwizzleV1,
};

/// The result every device operation returns.
pub type HostDeviceResultV1<T> = Result<T, BundleStatusV1>;

/// Why [`HostDeviceBackendV1::end_graph_capture`] returned no graph, and
/// whether its stream is still capturing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphCaptureErrorV1 {
    /// The capture was never ended; the stream is still capturing.
    Capturing(BundleStatusV1),
    /// The capture ended without a graph to launch: it was invalidated, or
    /// what it recorded failed to instantiate. The stream is not capturing.
    Ended(BundleStatusV1),
}

/// What one launch's flags ask a backend to do differently.
///
/// A backend reads this rather than the raw bits, so the one place that decides
/// what a flag means is here and a bit no version defines never reaches a
/// driver.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LaunchDispatchV1 {
    /// The ordinary launch, with no attribute attached.
    Plain,
    /// The launch CUDA's programmatic stream serialization is applied to.
    ProgrammaticStreamSerialization,
}

/// Decides how one launch config is dispatched, refusing an undefined flag.
///
/// The ABI validates flags at the boundary as well. This is the second half of
/// that rule: a backend that reaches an unknown bit here has no way to honour
/// it, so it refuses rather than launching without whatever was asked for.
pub fn launch_dispatch_v1(config: LaunchConfigV1) -> HostDeviceResultV1<LaunchDispatchV1> {
    if config.flags & !BUNDLE_LAUNCH_FLAGS_V1_KNOWN != 0 {
        return Err(BundleStatusV1::InvalidArgument);
    }
    if config.flags & BUNDLE_LAUNCH_FLAG_V1_PROGRAMMATIC_STREAM_SERIALIZATION != 0 {
        Ok(LaunchDispatchV1::ProgrammaticStreamSerialization)
    } else {
        Ok(LaunchDispatchV1::Plain)
    }
}

/// Whether one launch asked to run cooperatively: every block resident at
/// once, or no launch. A cooperative launch takes no programmatic stream
/// serialization, whose dependent would wait while holding the SMs the
/// grid's own blocks need.
pub fn launch_cooperative_v1(config: LaunchConfigV1) -> HostDeviceResultV1<bool> {
    let cooperative = config.flags & runtime_bundle_abi::BUNDLE_LAUNCH_FLAG_V1_COOPERATIVE != 0;
    if cooperative
        && launch_dispatch_v1(config)? == LaunchDispatchV1::ProgrammaticStreamSerialization
    {
        return Err(BundleStatusV1::InvalidArgument);
    }
    Ok(cooperative)
}

/// Decides which thread-block cluster one launch asked for, if any.
///
/// The one-block default asks for nothing, and this reports `None` for it, so a
/// backend that has never formed a cluster keeps running every launch it always
/// ran. Any other shape is checked here rather than in each backend: every
/// extent at least one block, the blocks inside the portable bound, and a grid
/// that divides into whole clusters.
///
/// A backend that reads `Some` and cannot form that cluster refuses the launch.
/// It never drops the cluster and launches the blocks: a kernel that reads
/// another block's shared memory or waits on a cluster barrier does something
/// else without one, so the flattened launch would be a different program
/// reported as this one.
pub fn launch_cluster_v1(config: LaunchConfigV1) -> HostDeviceResultV1<Option<LaunchClusterV1>> {
    let cluster = config.cluster;
    let blocks = cluster.blocks().ok_or(BundleStatusV1::InvalidArgument)?;
    if blocks == 0 || blocks > BUNDLE_LAUNCH_V1_MAX_CLUSTER_BLOCKS {
        return Err(BundleStatusV1::InvalidArgument);
    }
    for (extent, grid) in [
        (cluster.x, config.grid_x),
        (cluster.y, config.grid_y),
        (cluster.z, config.grid_z),
    ] {
        if extent == 0 || !grid.is_multiple_of(extent) {
            return Err(BundleStatusV1::InvalidArgument);
        }
    }
    if cluster.is_none() {
        Ok(None)
    } else {
        Ok(Some(cluster))
    }
}

/// One driver-owned object identity.
///
/// The value is opaque to the services layer, which only pairs it with the
/// generation-checked handle it issued for it.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct DeviceObjectV1(u64);

impl DeviceObjectV1 {
    #[must_use]
    pub const fn new(identity: u64) -> Self {
        Self(identity)
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// One peer edge a backend enabled, or found already enabled.
///
/// `host_acquired` is the whole ownership fact: it is true exactly when this
/// call performed the driver transition, and it is what decides whether the
/// last release of the edge undoes it or leaves it alone.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PeerAccessAcquisitionV1 {
    pub object: DeviceObjectV1,
    pub host_acquired: bool,
}

/// A range inside one driver-owned allocation, already bounds checked.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeviceRangeV1 {
    pub object: DeviceObjectV1,
    pub offset: u64,
    pub length: u64,
}

/// One tiled tensor map over a range this host already bounds checked.
///
/// The descriptor is the bundle's, with one substitution: the span it named has
/// become the range the services layer resolved. Turning that range into a
/// device address is the backend's work, and it is the only part of a tensor
/// map that a host-memory stand-in and a driver answer differently.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeviceTensorMapV1 {
    pub range: DeviceRangeV1,
    pub rank: u32,
    pub element_type: TensorMapElementV1,
    pub swizzle: TensorMapSwizzleV1,
    pub interleave: u32,
    pub l2_promotion: u32,
    pub oob_fill: u32,
    pub global_dims: [u64; 5],
    pub global_strides_bytes: [u64; 4],
    pub box_dims: [u32; 5],
    pub element_strides: [u32; 5],
}

/// One resolved kernel or collective argument.
///
/// A device-buffer argument arrives with its handle already resolved and its
/// range already checked against the allocation it names. Each scalar carries
/// exactly the width it names, so a backend passes it through rather than
/// choosing one.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DeviceArgumentV1 {
    Range(DeviceRangeV1),
    /// The null device pointer an unused slot of a fixed kernel ABI takes.
    NullDevicePointer,
    ScalarU32(u32),
    ScalarI32(i32),
    ScalarU64(u64),
    OpaqueBytes(Vec<u8>),
    TensorMap(DeviceTensorMapV1),
    /// A by-value array of addresses, in slot order, already bounds checked.
    ///
    /// `None` is the null pointer an unused slot takes. Turning a range into
    /// the address a kernel receives is the backend's work, which is why the
    /// slots arrive as ranges rather than as bytes.
    DevicePointerArray(Vec<Option<DeviceRangeV1>>),
}

/// The rank membership one collective was created for.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CollectiveBindingV1 {
    pub rank_count: u64,
    pub semantic_contract: Vec<u8>,
}

/// One operand of a custom operation, resolved and bounds checked already.
///
/// `writable` is what the alias rule is decided by, and the services layer has
/// applied that rule before a backend sees this.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeviceCustomOperandV1 {
    pub range: DeviceRangeV1,
    pub writable: bool,
}

/// A file range and destination for a backend's optional multi-rank upload.
pub struct PinnedFileUploadV1<'a> {
    pub stream: DeviceObjectV1,
    pub destination: DeviceObjectV1,
    pub source_offset: u64,
    pub byte_length: u64,
    pub read_chunk: &'a mut (dyn FnMut(u64, &mut [u8]) -> bool + Send),
}

/// The device operations one host instance performs on behalf of a bundle.
///
/// Every method takes `&mut self`: the services layer holds the backend in one
/// instance-scoped cell and never lends it across a call.
pub trait HostDeviceBackendV1 {
    fn create_mapped_gather(
        &mut self,
        _plan: mapped_gather::MappedGatherPlanV1,
    ) -> HostDeviceResultV1<DeviceObjectV1> {
        Err(BundleStatusV1::Unsupported)
    }
    fn enqueue_mapped_gather(
        &mut self,
        _gather: DeviceObjectV1,
        _rows: u64,
    ) -> HostDeviceResultV1<()> {
        Err(BundleStatusV1::Unsupported)
    }
    fn wait_mapped_gather(
        &mut self,
        _gather: DeviceObjectV1,
        _consumer: DeviceObjectV1,
    ) -> HostDeviceResultV1<()> {
        Err(BundleStatusV1::Unsupported)
    }
    fn complete_mapped_gather(
        &mut self,
        _gather: DeviceObjectV1,
    ) -> HostDeviceResultV1<Option<runtime_bundle_abi::MappedGatherStatsV1>> {
        Err(BundleStatusV1::Unsupported)
    }
    fn release_mapped_gather(&mut self, _gather: DeviceObjectV1) -> HostDeviceResultV1<()> {
        Err(BundleStatusV1::Unsupported)
    }

    /// Host-private KV arenas do not cross the bundle ABI. Each owns a copy
    /// stream and pinned buffers until release has fenced every submission.
    fn create_offload_arena(
        &mut self,
        _buffer: DeviceObjectV1,
        _page_bytes: usize,
        _pages: usize,
    ) -> HostDeviceResultV1<DeviceObjectV1> {
        Err(BundleStatusV1::Unsupported)
    }
    fn submit_offload_chunk(
        &mut self,
        _arena: DeviceObjectV1,
        _direction: runtime_kv::KvTransferDirection,
        _pages: &[runtime_kv::KvPagePair],
    ) -> HostDeviceResultV1<u64> {
        Err(BundleStatusV1::Unsupported)
    }
    fn query_offload_chunk(
        &mut self,
        _arena: DeviceObjectV1,
        _chunk: u64,
    ) -> HostDeviceResultV1<bool> {
        Err(BundleStatusV1::Unsupported)
    }
    fn fence_offload_arena(&mut self, _arena: DeviceObjectV1) -> HostDeviceResultV1<()> {
        Err(BundleStatusV1::Unsupported)
    }
    fn release_offload_arena(&mut self, _arena: DeviceObjectV1) -> HostDeviceResultV1<()> {
        Err(BundleStatusV1::Unsupported)
    }

    /// Whether this probe backend preserves allocation bounds without storage.
    ///
    /// Production backends keep the default. A control-only fake may opt in so
    /// file-backed rank images are not reread into host memory during injected
    /// failure tests.
    fn uses_sparse_probe_allocations(&self) -> bool {
        false
    }

    fn create_context(
        &mut self,
        rank: u32,
        device_ordinal: i32,
    ) -> HostDeviceResultV1<DeviceObjectV1>;
    fn release_context(&mut self, context: DeviceObjectV1) -> HostDeviceResultV1<()>;

    fn create_memory_pool(&mut self, context: DeviceObjectV1)
    -> HostDeviceResultV1<DeviceObjectV1>;

    /// Whether one pool already grants one peer context access.
    ///
    /// A pool's peer access is device state, not this host's: another user of
    /// the same device may have enabled it. The services layer asks before it
    /// enables anything, so it can leave a grant it did not make alone.
    fn memory_pool_peer_access(
        &mut self,
        pool: DeviceObjectV1,
        peer_context: DeviceObjectV1,
    ) -> HostDeviceResultV1<bool>;

    /// Enables or disables one pool's access for one peer context.
    ///
    /// The services layer calls this only for a transition it decided to make:
    /// enable when it found the grant absent, disable when it is giving back
    /// the last lease over a grant it enabled itself.
    fn set_memory_pool_peer_access(
        &mut self,
        pool: DeviceObjectV1,
        peer_context: DeviceObjectV1,
        enabled: bool,
    ) -> HostDeviceResultV1<()>;
    fn release_memory_pool(&mut self, pool: DeviceObjectV1) -> HostDeviceResultV1<()>;

    fn allocate_device_buffer(
        &mut self,
        pool: DeviceObjectV1,
        stream: Option<DeviceObjectV1>,
        byte_length: u64,
        alignment: u64,
    ) -> HostDeviceResultV1<DeviceObjectV1>;
    /// Releases a device allocation or a mapped host allocation.
    fn release_device_buffer(&mut self, buffer: DeviceObjectV1) -> HostDeviceResultV1<()>;

    /// Allocates reusable sampling buffers before this context's KV pool is sized.
    /// Repeated calls with the same vocabulary retain the same buffers.
    fn reserve_sampling_workspace(
        &mut self,
        _context: DeviceObjectV1,
        _vocabulary: usize,
    ) -> HostDeviceResultV1<()> {
        Err(runtime_bundle_abi::BundleStatusV1::Unsupported)
    }

    /// Allocates page-locked host memory that `stream`'s context, and every
    /// other one, reads through a device address; `fill` writes its bytes
    /// before any device work can see them, and refusing fails the call.
    ///
    /// Every buffer operation accepts the answer, and
    /// [`Self::release_device_buffer`] frees it. It is host memory, so it is
    /// not counted against the device and [`Self::free_device_memory`] does
    /// not shrink. A backend without mapped host memory keeps the default.
    fn allocate_mapped_host_buffer(
        &mut self,
        _stream: DeviceObjectV1,
        _byte_length: u64,
        _fill: &mut dyn FnMut(&mut [u8]) -> bool,
    ) -> HostDeviceResultV1<DeviceObjectV1> {
        Err(BundleStatusV1::Unsupported)
    }

    /// Allocates `byte_length` bytes of `context`'s device memory that
    /// another process can map through [`Self::export_device_buffer`],
    /// readable and writable from this device only. Unlike a pool
    /// allocation it exists once this returns, so `stream` orders nothing.
    ///
    /// Every buffer operation accepts the answer, and
    /// [`Self::release_device_buffer`] releases it. A backend without
    /// lendable memory keeps the default.
    fn allocate_lendable_device_buffer(
        &mut self,
        _context: DeviceObjectV1,
        _stream: DeviceObjectV1,
        _byte_length: u64,
    ) -> HostDeviceResultV1<DeviceObjectV1> {
        Err(BundleStatusV1::Unsupported)
    }

    /// A new descriptor through which another process maps one lendable
    /// allocation, and the size of the mapping it names, which is at least
    /// the allocation's length.
    fn export_device_buffer(
        &mut self,
        _buffer: DeviceObjectV1,
    ) -> HostDeviceResultV1<(std::os::fd::OwnedFd, u64)> {
        Err(BundleStatusV1::Unsupported)
    }

    /// Maps device memory another process lent as `fd`, a mapping of
    /// `mapped_bytes` whose first `byte_length` bytes are used, into
    /// `context`. Work on `context`'s device and on the devices `readers`
    /// names (ordinals) may read it; none may write it if `read_only`. The
    /// caller keeps `fd`.
    ///
    /// The memory stays mapped until [`Self::release_device_buffer`], however
    /// long the lender lives. A backend without lendable memory keeps the
    /// default.
    #[allow(clippy::too_many_arguments)]
    fn import_device_buffer(
        &mut self,
        _context: DeviceObjectV1,
        _stream: DeviceObjectV1,
        _fd: std::os::fd::BorrowedFd<'_>,
        _mapped_bytes: u64,
        _byte_length: u64,
        _readers: &[i32],
        _read_only: bool,
    ) -> HostDeviceResultV1<DeviceObjectV1> {
        Err(BundleStatusV1::Unsupported)
    }

    /// Names the physical device of one context the same way in every
    /// process on this host, so a lender and a borrower can tell they hold
    /// the same device whatever ordinals they see.
    fn device_identity(&self, _context: DeviceObjectV1) -> HostDeviceResultV1<String> {
        Err(BundleStatusV1::Unsupported)
    }

    /// Free bytes in this rank's context after bundle allocations are resident.
    fn free_device_memory(&self, _context: DeviceObjectV1) -> HostDeviceResultV1<u64> {
        Err(runtime_bundle_abi::BundleStatusV1::Unsupported)
    }

    /// Whether one allocation this backend returned satisfies an alignment.
    ///
    /// An allocator carries its own alignment guarantee, which is not the one a
    /// caller asked for. Rather than advertise a maximum and trust it, the
    /// services layer asks about the allocation it actually received and fails
    /// it closed when the answer is no.
    fn allocation_satisfies_alignment(
        &self,
        buffer: DeviceObjectV1,
        alignment: u64,
    ) -> HostDeviceResultV1<bool>;

    /// Enables one directed context peer edge, if it is not enabled already.
    ///
    /// The answer says which of those two happened, because a host that
    /// disabled an edge it merely found enabled would revoke access another
    /// user of the same devices is relying on.
    fn acquire_peer_access(
        &mut self,
        owner_context: DeviceObjectV1,
        peer_context: DeviceObjectV1,
    ) -> HostDeviceResultV1<PeerAccessAcquisitionV1>;

    /// Gives back one peer-access record, disabling the edge only if asked.
    ///
    /// The services layer passes `disable` exactly when it is releasing the
    /// last lease over an edge this host enabled itself. Every other release
    /// drops the record and leaves the driver state alone.
    fn release_peer_access(
        &mut self,
        peer_access: DeviceObjectV1,
        disable: bool,
    ) -> HostDeviceResultV1<()>;

    fn create_stream(
        &mut self,
        context: DeviceObjectV1,
        stream_kind: u32,
    ) -> HostDeviceResultV1<DeviceObjectV1>;
    fn synchronize_stream(&mut self, stream: DeviceObjectV1) -> HostDeviceResultV1<()>;
    fn release_stream(&mut self, stream: DeviceObjectV1) -> HostDeviceResultV1<()>;
    fn begin_graph_capture(&mut self, stream: DeviceObjectV1) -> HostDeviceResultV1<()>;
    /// Ends the capture begun on `stream` and instantiates what it recorded.
    /// The error says whether the stream is still capturing, so the services
    /// layer never keeps a capture the device already ended.
    fn end_graph_capture(
        &mut self,
        stream: DeviceObjectV1,
    ) -> Result<DeviceObjectV1, GraphCaptureErrorV1>;
    fn launch_graph(
        &mut self,
        stream: DeviceObjectV1,
        graph: DeviceObjectV1,
    ) -> HostDeviceResultV1<()>;
    fn release_graph(&mut self, graph: DeviceObjectV1) -> HostDeviceResultV1<()>;

    fn create_event(&mut self, context: DeviceObjectV1) -> HostDeviceResultV1<DeviceObjectV1>;
    fn record_event(
        &mut self,
        event: DeviceObjectV1,
        stream: DeviceObjectV1,
    ) -> HostDeviceResultV1<()>;
    fn wait_event(
        &mut self,
        stream: DeviceObjectV1,
        event: DeviceObjectV1,
    ) -> HostDeviceResultV1<()>;
    fn synchronize_event(&mut self, event: DeviceObjectV1) -> HostDeviceResultV1<()>;
    fn release_event(&mut self, event: DeviceObjectV1) -> HostDeviceResultV1<()>;

    /// Copies host bytes into one device range, completing before it returns.
    ///
    /// `source` borrows an ABI byte view whose life ends when the bundle's call
    /// returns. A backend may therefore leave no transfer in flight against it:
    /// by the time this returns, the driver must be done reading `source`.
    /// [`assert_host_transfer_contract_v1`] checks that.
    fn copy_host_to_device(
        &mut self,
        stream: DeviceObjectV1,
        destination: DeviceRangeV1,
        source: &[u8],
    ) -> HostDeviceResultV1<()>;

    /// Copies host bytes into one device range, ordered on `stream` like a
    /// launch: the driver is done reading `source` when this returns, and the
    /// device copy completes before later work on the stream starts. Work on
    /// other streams and host reads must order themselves after it. The
    /// default completes the copy before returning, which also satisfies this.
    fn copy_host_to_device_ordered(
        &mut self,
        stream: DeviceObjectV1,
        destination: DeviceRangeV1,
        source: &[u8],
    ) -> HostDeviceResultV1<()> {
        self.copy_host_to_device(stream, destination, source)
    }

    /// Copies bytes that `fill` writes into one device range, ordered on
    /// `stream` as [`Self::copy_host_to_device_ordered`] is: `fill(at, chunk)`
    /// writes bytes `at..at + chunk.len()` of the range, chunk by chunk in
    /// order, before this returns. A backend with its own page-locked staging
    /// hands `fill` that memory, so the bytes are written once; the default
    /// fills host memory and copies it.
    fn fill_host_to_device_ordered(
        &mut self,
        stream: DeviceObjectV1,
        destination: DeviceRangeV1,
        fill: &mut dyn FnMut(usize, &mut [u8]),
    ) -> HostDeviceResultV1<()> {
        let length =
            usize::try_from(destination.length).map_err(|_| BundleStatusV1::ResourceExhausted)?;
        let mut bytes = vec![0; length];
        fill(0, &mut bytes);
        self.copy_host_to_device_ordered(stream, destination, &bytes)
    }

    /// Page-locked bytes this backend holds of its own to stage
    /// [`Self::fill_host_to_device_ordered`].
    fn fill_staging_bytes(&self) -> u64 {
        0
    }

    /// A backend may stage a file-backed rank range with its own pinned copy
    /// pipeline. Returning `None` uses the ordinary bounded host-copy path.
    fn upload_pinned_file_range(
        &mut self,
        _stream: DeviceObjectV1,
        _destination: DeviceObjectV1,
        _source_offset: u64,
        _byte_length: u64,
        _read_chunk: &mut dyn FnMut(u64, &mut [u8]) -> bool,
    ) -> Option<HostDeviceResultV1<()>> {
        None
    }

    /// Overlaps independent rank transfers on their own streams. Returning
    /// `None` asks the host to use the ordinary per-rank upload path.
    fn upload_pinned_file_ranges(
        &mut self,
        _uploads: &mut [PinnedFileUploadV1<'_>],
    ) -> Option<HostDeviceResultV1<()>> {
        None
    }

    /// Copies one device range into host bytes, completing before it returns.
    ///
    /// The same rule as [`Self::copy_host_to_device`], in the other direction:
    /// `destination` is an ABI byte view the bundle may drop the moment this
    /// returns, so the transfer into it must already be finished.
    fn copy_device_to_host(
        &mut self,
        stream: DeviceObjectV1,
        destination: &mut [u8],
        source: DeviceRangeV1,
    ) -> HostDeviceResultV1<()>;
    fn copy_device_to_device(
        &mut self,
        stream: DeviceObjectV1,
        destination: DeviceRangeV1,
        source: DeviceRangeV1,
    ) -> HostDeviceResultV1<()>;
    fn copy_peer_to_peer(
        &mut self,
        stream: DeviceObjectV1,
        destination_context: DeviceObjectV1,
        destination: DeviceRangeV1,
        source_context: DeviceObjectV1,
        source: DeviceRangeV1,
    ) -> HostDeviceResultV1<()>;
    fn zero_device_buffer(
        &mut self,
        stream: DeviceObjectV1,
        destination: DeviceRangeV1,
    ) -> HostDeviceResultV1<()>;

    /// Writes the addresses of `slots` into `destination`, in slot order.
    ///
    /// The services layer has resolved and bounds checked every range and has
    /// already required `destination` to hold exactly one eight-byte entry per
    /// slot. What is left is the one fact only a backend holds, which is the
    /// address it has for each range; `None` is a null pointer. Like every
    /// other host transfer, this completes before it returns.
    fn write_device_pointer_table(
        &mut self,
        stream: DeviceObjectV1,
        destination: DeviceRangeV1,
        slots: &[Option<DeviceRangeV1>],
    ) -> HostDeviceResultV1<()>;

    fn load_module(
        &mut self,
        context: DeviceObjectV1,
        module_bytes: &[u8],
    ) -> HostDeviceResultV1<DeviceObjectV1>;
    fn resolve_symbol(
        &mut self,
        module: DeviceObjectV1,
        symbol_name: &[u8],
    ) -> HostDeviceResultV1<DeviceObjectV1>;
    fn configure_symbol(
        &mut self,
        symbol: DeviceObjectV1,
        dynamic_shared_memory_bytes: u32,
    ) -> HostDeviceResultV1<()>;
    /// What the driver reports about keeping `symbol`'s clusters resident at
    /// `config`'s shape (`query_cluster_residency_v1`). A backend without a
    /// device to ask keeps the default, which reports nothing.
    fn cluster_residency(
        &mut self,
        _symbol: DeviceObjectV1,
        _config: LaunchConfigV1,
    ) -> HostDeviceResultV1<runtime_bundle_abi::BundleRawClusterResidencyV1> {
        Err(BundleStatusV1::Unsupported)
    }
    fn launch_symbol(
        &mut self,
        stream: DeviceObjectV1,
        symbol: DeviceObjectV1,
        config: LaunchConfigV1,
        arguments: &[DeviceArgumentV1],
    ) -> HostDeviceResultV1<()>;
    fn release_symbol(&mut self, symbol: DeviceObjectV1) -> HostDeviceResultV1<()>;
    fn release_module(&mut self, module: DeviceObjectV1) -> HostDeviceResultV1<()>;

    /// Runs one collective, ordered against the stream it was given.
    ///
    /// `stream` is what places the collective in the bundle's own sequence of
    /// work. Whatever the bundle already ordered on it, typically the kernel
    /// that filled the rank buffers, has to be complete before this reads them,
    /// and this must be complete before anything the bundle orders on that
    /// stream afterwards. A backend that stages through some other stream still
    /// owes both halves of that.
    /// [`assert_collective_ordering_contract_v1`] checks the first half, which
    /// is the one a backend can silently get wrong and read stale bytes.
    fn execute_collective(
        &mut self,
        stream: DeviceObjectV1,
        binding: &CollectiveBindingV1,
        operation: &[u8],
        arguments: &[DeviceArgumentV1],
    ) -> HostDeviceResultV1<()>;

    /// Runs one bundle-supplied operation against values this backend lends it.
    ///
    /// Everything a handle can decide has been decided: the ranges are resolved
    /// and bounds checked, and no writable one aliases another operand. What is
    /// left is what only a backend knows. It checks that the stream and every
    /// operand belong to the launching context, establishes that context, turns
    /// each range into the raw address it has for it, and lends those and its
    /// own stream for exactly one callback. A backend that cannot check one of
    /// those refuses without invoking anything.
    ///
    /// Waiting is not a backend's decision. The services layer has already
    /// waited on this stream before calling, and waits again afterwards when
    /// the operation's completion policy says it may enqueue, so the ordering
    /// is one rule in one place rather than one each backend repeats.
    ///
    /// The answer is the operation's own code, zero for success.
    fn run_custom_operation(
        &mut self,
        context: DeviceObjectV1,
        stream: DeviceObjectV1,
        operation_id: &[u8],
        operands: &[DeviceCustomOperandV1],
        operation: &CustomOperationV1<'_>,
    ) -> HostDeviceResultV1<i32>;

    /// Records one diagnostic line a bundle emitted.
    fn log(&mut self, message: &[u8]) -> HostDeviceResultV1<()>;
}

/// The bytes one entry of a device-pointer table occupies.
///
/// A device address is 64 bits on every target this host serves, and the table
/// a kernel reads is that many bytes per slot, little-endian, in slot order.
pub const DEVICE_POINTER_TABLE_ENTRY_BYTES_V1: u64 = 8;

/// The semantic contract for a lane-wise sum of one `f32` range per rank.
///
/// This lives on the shared device surface rather than inside one backend,
/// because it is the identity a bundle names and every backend answers to. A
/// backend either performs exactly this reduction or reports the contract
/// unsupported; nothing about it names a model.
pub const COLLECTIVE_SUM_F32_V1: &[u8] = b"sum-f32-v1";

/// The semantic contract for a lane-wise sum of one `bf16` range per rank.
///
/// It is the same reduction [`COLLECTIVE_SUM_F32_V1`] declares over a narrower
/// lane. Every rank contributes one span of the same byte length, lane `i` of
/// the result is the sum of lane `i` of every rank's span, and every rank's span
/// is overwritten with that result in place.
///
/// The numeric contract is a fixed left fold in rank order, performed in `f32`.
/// Rank 0's lane is widened exactly and is the accumulator the fold starts from.
/// Ranks 1 through N-1 are then added to it in that order, each of those
/// additions rounding to `f32` in the ordinary way. The narrowing to `bf16`
/// happens once, after rank N-1, by [`bf16_bits_from_f32_v1`].
///
/// That is the reduction the pinned upstream `packed_reduce` performs, which is
/// where the seed comes from: it takes `upcast(ptrs[0][idx])` as `tmp` and adds
/// ranks 1 through N-1 into it. The seed is part of the contract and not an
/// empty starting value, because no zero is an identity for every input. A lane
/// in which every rank contributes `-0.0` answers `-0.0` under this fold, where
/// a fold seeded with `+0.0` would answer `+0.0`.
///
/// [`COLLECTIVE_SUM_F32_V1`] does seed with `+0.0` and add every rank including
/// rank 0. That is the behaviour it shipped with and it stays, so the two
/// contracts differ in their seed as well as in their lane width. Among non-NaN
/// inputs they differ observably only for a lane whose every contribution is
/// `-0.0`. A NaN is outside that comparison rather than inside it: this host
/// promises nothing about the sign or the payload of a NaN that came out of an
/// addition, so the two seeds are not claimed to agree on one.
///
/// This is not an exactly rounded sum, and the fold order is part of the
/// contract for the same reason the seed is, because the answer depends on it.
/// Over the four `bf16` lanes 2^30, 1, -2^30, 1 the rank-order fold answers 1:
/// 2^30 + 1 rounds back to 2^30, so the total returns to zero before the last
/// rank is added. Pairing rank 0 with rank 2 and rank 1 with rank 3 answers 2,
/// and pairing the adjacent ranks answers 0. What does hold is that every rank
/// receives the same bytes, because the fold is performed once and its result is
/// written back to all of them. Nothing about it names a model.
pub const COLLECTIVE_SUM_BF16_V1: &[u8] = b"sum-bf16-v1";

/// Every generic collective contract this host answers to, in one place.
///
/// This is the whole of what a backend may accept, so a contract absent from it
/// is unsupported on every backend rather than reduced by whichever one happened
/// to recognise it. A bundle discovers what it can ask for by naming one of
/// these; anything else is refused without a rank buffer being touched.
pub const GENERIC_COLLECTIVE_CONTRACTS_V1: &[&[u8]] =
    &[COLLECTIVE_SUM_F32_V1, COLLECTIVE_SUM_BF16_V1];

/// One generic collective this host performs itself.
///
/// The contract identity and the lane width belong together, because the width
/// is the only thing that differs between them: both reduce lane-wise, in rank
/// order, in place, accumulating in `f32`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GenericCollectiveV1 {
    /// [`COLLECTIVE_SUM_F32_V1`].
    SumF32,
    /// [`COLLECTIVE_SUM_BF16_V1`].
    SumBf16,
}

impl GenericCollectiveV1 {
    /// The collective one contract names, or `None` for any other bytes.
    #[must_use]
    pub fn from_contract_v1(contract: &[u8]) -> Option<Self> {
        if contract == COLLECTIVE_SUM_F32_V1 {
            Some(Self::SumF32)
        } else if contract == COLLECTIVE_SUM_BF16_V1 {
            Some(Self::SumBf16)
        } else {
            None
        }
    }

    /// The contract bytes this collective answers to.
    #[must_use]
    pub const fn contract(self) -> &'static [u8] {
        match self {
            Self::SumF32 => COLLECTIVE_SUM_F32_V1,
            Self::SumBf16 => COLLECTIVE_SUM_BF16_V1,
        }
    }

    /// The bytes one lane of this collective occupies.
    #[must_use]
    pub const fn lane_bytes(self) -> u64 {
        match self {
            Self::SumF32 => 4,
            Self::SumBf16 => 2,
        }
    }

    /// The per-lane totals rank 0's contribution starts the fold from.
    ///
    /// The two contracts seed differently, and it is observable, so the choice
    /// lives here rather than in whichever backend builds the accumulator.
    /// [`COLLECTIVE_SUM_BF16_V1`] widens rank 0 exactly and rounds nothing,
    /// which is what the pinned upstream `packed_reduce` does with
    /// `upcast(ptrs[0][idx])`. [`COLLECTIVE_SUM_F32_V1`] starts from `+0.0` and
    /// adds rank 0 like any other rank, which is what it shipped with. Among
    /// non-NaN inputs the two answer differently only for a lane whose every
    /// contribution is `-0.0`; a NaN seed is outside that comparison, because
    /// what an addition does to a NaN's sign and payload is unspecified.
    pub fn seed_lanes_v1(self, lane_count: usize, first: &[u8]) -> HostDeviceResultV1<Vec<f32>> {
        match self {
            Self::SumF32 => {
                let mut totals = vec![0.0_f32; lane_count];
                self.accumulate_lanes_v1(&mut totals, first)?;
                Ok(totals)
            }
            Self::SumBf16 => {
                let lane_bytes = usize::try_from(self.lane_bytes())
                    .map_err(|_| BundleStatusV1::InvalidArgument)?;
                if first.len() != lane_count.saturating_mul(lane_bytes) {
                    return Err(BundleStatusV1::InvalidArgument);
                }
                Ok(first
                    .chunks_exact(lane_bytes)
                    .map(|lane| f32_from_bf16_bits_v1(u16::from_le_bytes([lane[0], lane[1]])))
                    .collect())
            }
        }
    }

    /// Adds one later rank's contribution into the running per-lane totals.
    ///
    /// The totals are `f32` whatever the lane width is. Each addition rounds to
    /// `f32`; what the wider accumulator buys is that a `bf16` lane is never
    /// narrowed back between ranks, not that the fold is exact. This is for
    /// ranks 1 through N-1: rank 0 is [`seed_lanes_v1`], which is not the same
    /// call for every contract.
    ///
    /// [`seed_lanes_v1`]: Self::seed_lanes_v1
    pub fn accumulate_lanes_v1(
        self,
        totals: &mut [f32],
        contribution: &[u8],
    ) -> HostDeviceResultV1<()> {
        let lane_bytes =
            usize::try_from(self.lane_bytes()).map_err(|_| BundleStatusV1::InvalidArgument)?;
        if contribution.len() != totals.len().saturating_mul(lane_bytes) {
            return Err(BundleStatusV1::InvalidArgument);
        }
        for (total, lane) in totals.iter_mut().zip(contribution.chunks_exact(lane_bytes)) {
            *total += match self {
                Self::SumF32 => f32::from_le_bytes([lane[0], lane[1], lane[2], lane[3]]),
                Self::SumBf16 => f32_from_bf16_bits_v1(u16::from_le_bytes([lane[0], lane[1]])),
            };
        }
        Ok(())
    }

    /// Encodes the finished totals into the bytes every rank receives.
    #[must_use]
    pub fn encode_lanes_v1(self, totals: &[f32]) -> Vec<u8> {
        match self {
            Self::SumF32 => totals
                .iter()
                .flat_map(|total| total.to_le_bytes())
                .collect(),
            Self::SumBf16 => totals
                .iter()
                .flat_map(|total| bf16_bits_from_f32_v1(*total).to_le_bytes())
                .collect(),
        }
    }
}

/// Widens one `bf16` lane into the `f32` the sum accumulates in.
///
/// Every `bf16` is an `f32` whose low 16 mantissa bits are zero, so this is
/// exact for every value including the subnormals, the infinities, and the NaNs.
#[must_use]
pub const fn f32_from_bf16_bits_v1(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Rounds one `f32` to the `bf16` bits [`COLLECTIVE_SUM_BF16_V1`] writes back.
///
/// The rounding is to nearest with ties to even, which is the mode `f32`
/// addition already rounds by. This is the one narrowing to `bf16` the contract
/// performs, applied to the finished `f32` fold; the additions inside that fold
/// have each already rounded to `f32`.
///
/// The two ends of the range are the reason this is written down rather than
/// truncated. A NaN stays a NaN: its bits are truncated, which keeps the sign
/// and the top of the payload, and the quiet bit is then set so a signalling
/// NaN cannot arrive as an infinity. Every other value carries the
/// round-half-to-even bias into the truncation, which is what turns a finite sum
/// too large for `bf16` into an infinity rather than into the largest finite
/// `bf16`, and a sum too small into a zero of the right sign.
#[must_use]
pub fn bf16_bits_from_f32_v1(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        // 0x0040 is the quiet bit: the top bit of the 7-bit `bf16` mantissa.
        return ((bits >> 16) as u16) | 0x0040;
    }
    let bias = 0x7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(bias) >> 16) as u16
}

/// One validated generic collective, ready for a backend to move bytes for.
///
/// `ranges` is one span per rank in rank order, so `ranges[i]` is rank `i`'s and
/// the sum folds in that order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenericCollectivePlanV1 {
    pub collective: GenericCollectiveV1,
    pub ranges: Vec<DeviceRangeV1>,
    /// The lanes every span carries.
    pub lane_count: usize,
    /// The bytes every span carries.
    pub byte_length: usize,
}

impl GenericCollectivePlanV1 {
    /// Folds every rank's contribution into the per-lane totals.
    ///
    /// `read` hands back one rank's bytes. Which rank seeds the fold and in what
    /// order the rest are added is decided here rather than in a backend,
    /// because both are observable in the result: a backend that built its own
    /// accumulator could answer differently from the other one for the same
    /// request, and no test of a single backend would find it.
    ///
    /// A rank whose bytes cannot be read stops the fold with that failure, so a
    /// backend never reduces over a contribution it did not get.
    pub fn fold_lanes_v1(
        &self,
        mut read: impl FnMut(DeviceRangeV1) -> HostDeviceResultV1<Vec<u8>>,
    ) -> HostDeviceResultV1<Vec<f32>> {
        let mut ranks = self.ranges.iter();
        let first = ranks.next().ok_or(BundleStatusV1::InvalidArgument)?;
        let mut totals = self
            .collective
            .seed_lanes_v1(self.lane_count, &read(*first)?)?;
        for range in ranks {
            self.collective
                .accumulate_lanes_v1(&mut totals, &read(*range)?)?;
        }
        Ok(totals)
    }
}

/// Validates one collective request into the spans a backend will reduce.
///
/// This lives here rather than in a backend so a driver host and a host-memory
/// host accept and refuse exactly the same requests. A contract this host does
/// not answer to, or an operation that is not the one the collective was created
/// for, is [`BundleStatusV1::Unsupported`]; every shape failure is
/// [`BundleStatusV1::InvalidArgument`]. Either way nothing is read or
/// written, so a refused request leaves every rank's contribution where it was.
///
/// What both contracts require is one range argument per rank, in rank order,
/// all of the same byte length, and a length that divides into whole lanes.
///
/// [`COLLECTIVE_SUM_BF16_V1`] states more than that, and states it exactly. Its
/// spans carry at least one lane and start on a lane boundary, because a span
/// that begins mid-lane names a different set of `bf16` values than the one the
/// contract describes. They also do not overlap one another: the sum is written
/// back in place, so two ranks sharing bytes would have one rank's contribution
/// read after the other rank's result had already replaced it. Two spans in
/// different allocations are taken as disjoint, which is what a distinct device
/// buffer means here. [`COLLECTIVE_SUM_F32_V1`] keeps exactly the acceptance it
/// already had.
pub fn plan_generic_collective_v1(
    binding: &CollectiveBindingV1,
    operation: &[u8],
    arguments: &[DeviceArgumentV1],
) -> HostDeviceResultV1<GenericCollectivePlanV1> {
    let Some(collective) = GenericCollectiveV1::from_contract_v1(&binding.semantic_contract) else {
        return Err(BundleStatusV1::Unsupported);
    };
    if operation != binding.semantic_contract.as_slice() {
        return Err(BundleStatusV1::Unsupported);
    }
    let ranks = usize::try_from(binding.rank_count).map_err(|_| BundleStatusV1::InvalidArgument)?;
    if ranks == 0 || arguments.len() != ranks {
        return Err(BundleStatusV1::InvalidArgument);
    }
    let lane_bytes = collective.lane_bytes();
    let mut ranges = Vec::with_capacity(ranks);
    for argument in arguments {
        let DeviceArgumentV1::Range(range) = argument else {
            return Err(BundleStatusV1::InvalidArgument);
        };
        if !range.length.is_multiple_of(lane_bytes) {
            return Err(BundleStatusV1::InvalidArgument);
        }
        if let Some(first) = ranges.first() {
            let first: &DeviceRangeV1 = first;
            if range.length != first.length {
                return Err(BundleStatusV1::InvalidArgument);
            }
        }
        ranges.push(*range);
    }
    let length = ranges[0].length;
    if collective == GenericCollectiveV1::SumBf16 {
        if length == 0 {
            return Err(BundleStatusV1::InvalidArgument);
        }
        for (index, range) in ranges.iter().enumerate() {
            if !range.offset.is_multiple_of(lane_bytes) {
                return Err(BundleStatusV1::InvalidArgument);
            }
            if ranges[..index]
                .iter()
                .any(|earlier| spans_overlap_v1(*earlier, *range))
            {
                return Err(BundleStatusV1::InvalidArgument);
            }
        }
    }
    let byte_length = usize::try_from(length).map_err(|_| BundleStatusV1::InvalidArgument)?;
    let lane_count =
        usize::try_from(length / lane_bytes).map_err(|_| BundleStatusV1::InvalidArgument)?;
    Ok(GenericCollectivePlanV1 {
        collective,
        ranges,
        lane_count,
        byte_length,
    })
}

/// Whether two spans of the same allocation share a byte.
fn spans_overlap_v1(left: DeviceRangeV1, right: DeviceRangeV1) -> bool {
    if left.object != right.object {
        return false;
    }
    let left_end = left.offset.saturating_add(left.length);
    let right_end = right.offset.saturating_add(right.length);
    left.offset < right_end && right.offset < left_end
}

/// Proves one backend finishes a host transfer before the call returns.
///
/// A backend that leaves an asynchronous transfer in flight against an ABI byte
/// view is unsound, and the symptom is silent: the bytes are right whenever the
/// race happens to lose. This runs the smallest sequence that would expose it,
/// a write followed immediately by a read with no synchronization between them,
/// and reports a mismatch as [`BundleStatusV1::InvalidState`].
///
/// It takes only a device ordinal, so a conformance host can run the same check
/// against the host-memory backend without a device and against a driver
/// backend with one.
pub fn assert_host_transfer_contract_v1<D: HostDeviceBackendV1>(
    device: &mut D,
    device_ordinal: i32,
) -> HostDeviceResultV1<()> {
    const PATTERN: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

    let context = device.create_context(0, device_ordinal)?;
    let outcome = host_transfer_round_trip_v1(device, context, &PATTERN);
    // The check owns everything it made, whatever the round trip decided.
    let _ = device.release_context(context);
    outcome
}

/// Proves one backend honours work already ordered on a collective's stream.
///
/// A backend that stages a collective through some other stream can read the
/// rank buffers before the work that filled them has landed. The result is a
/// plausible number computed from stale bytes, which no later check catches.
///
/// This orders a write on the collective's own stream, runs the collective
/// without synchronizing in between, and reports
/// [`BundleStatusV1::InvalidState`] if the reduction did not see it.
pub fn assert_collective_ordering_contract_v1<D: HostDeviceBackendV1>(
    device: &mut D,
    device_ordinal: i32,
) -> HostDeviceResultV1<()> {
    assert_generic_collective_ordering_contract_v1(
        device,
        device_ordinal,
        GenericCollectiveV1::SumF32,
    )
}

/// Proves that of one named generic collective, rather than of the `f32` one.
///
/// Ordering is a property of the backend and not of the lane width, so a backend
/// that answers to more than one contract owes it once per contract: staging one
/// of them through another stream is exactly as silent a failure as staging the
/// only one was.
pub fn assert_generic_collective_ordering_contract_v1<D: HostDeviceBackendV1>(
    device: &mut D,
    device_ordinal: i32,
    collective: GenericCollectiveV1,
) -> HostDeviceResultV1<()> {
    let context = device.create_context(0, device_ordinal)?;
    let outcome = collective_ordering_round_trip_v1(device, context, collective);
    let _ = device.release_context(context);
    outcome
}

fn collective_ordering_round_trip_v1<D: HostDeviceBackendV1>(
    device: &mut D,
    context: DeviceObjectV1,
    collective: GenericCollectiveV1,
) -> HostDeviceResultV1<()> {
    const RANKS: u64 = 2;

    let lane = collective.lane_bytes();
    let pool = device.create_memory_pool(context)?;
    let stream = device.create_stream(context, 0)?;
    let mut buffers = Vec::new();
    for _ in 0..RANKS {
        buffers.push(device.allocate_device_buffer(pool, Some(stream), lane, lane)?);
    }
    let ranges = buffers
        .iter()
        .map(|buffer| DeviceRangeV1 {
            object: *buffer,
            offset: 0,
            length: lane,
        })
        .collect::<Vec<_>>();

    let outcome = (|| {
        // Each rank contributes one, so a reduction that saw both reads two.
        let contribution = collective.encode_lanes_v1(&[1.0]);
        for range in &ranges {
            device.copy_host_to_device(stream, *range, &contribution)?;
        }
        let arguments = ranges
            .iter()
            .map(|range| DeviceArgumentV1::Range(*range))
            .collect::<Vec<_>>();
        device.execute_collective(
            stream,
            &CollectiveBindingV1 {
                rank_count: RANKS,
                semantic_contract: collective.contract().to_vec(),
            },
            collective.contract(),
            &arguments,
        )?;
        let mut read = vec![0_u8; contribution.len()];
        device.copy_device_to_host(stream, &mut read, ranges[0])?;
        if read == collective.encode_lanes_v1(&[RANKS as f32]) {
            Ok(())
        } else {
            Err(BundleStatusV1::InvalidState)
        }
    })();

    for buffer in buffers.into_iter().rev() {
        let _ = device.release_device_buffer(buffer);
    }
    let _ = device.release_stream(stream);
    let _ = device.release_memory_pool(pool);
    outcome
}

fn host_transfer_round_trip_v1<D: HostDeviceBackendV1>(
    device: &mut D,
    context: DeviceObjectV1,
    pattern: &[u8],
) -> HostDeviceResultV1<()> {
    let pool = device.create_memory_pool(context)?;
    let stream = device.create_stream(context, 0)?;
    let length = pattern.len() as u64;
    let buffer = device.allocate_device_buffer(pool, Some(stream), length, 8)?;
    let range = DeviceRangeV1 {
        object: buffer,
        offset: 0,
        length,
    };

    let outcome = (|| {
        device.zero_device_buffer(stream, range)?;
        device.copy_host_to_device(stream, range, pattern)?;
        let mut read = vec![0_u8; pattern.len()];
        // No synchronization here on purpose: both transfers must already be
        // complete, so a backend that deferred either of them reads zeroes.
        device.copy_device_to_host(stream, &mut read, range)?;
        if read == pattern {
            Ok(())
        } else {
            Err(BundleStatusV1::InvalidState)
        }
    })();

    let _ = device.release_device_buffer(buffer);
    let _ = device.release_stream(stream);
    let _ = device.release_memory_pool(pool);
    outcome
}

/// One rank's allocation counts, as the ledger holding them states them.
///
/// `context` is the object identity the backend attached this rank to, so a
/// backend can pair the counts with the driver object it still holds without
/// keeping a second copy of the mapping.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostDeviceLedgerRankV1 {
    pub context: u64,
    pub rank: u32,
    pub device_ordinal: i32,
    pub allocated_bytes: u64,
    pub peak_allocated_bytes: u64,
}

/// One planned allocation, and every count committing it will set.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostDeviceAllocationPlanV1 {
    context: Option<u64>,
    total_bytes: u64,
    owner_bytes: u64,
}

/// One planned release, and every count committing it will set.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostDeviceReleasePlanV1 {
    context: Option<u64>,
    total_bytes: u64,
    owner_bytes: u64,
}

/// One planned context release, and the unattributed total it leaves behind.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostDeviceContextReleasePlanV1 {
    context: u64,
    unattributed_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LedgerRankV1 {
    rank: u32,
    device_ordinal: i32,
    allocated_bytes: u64,
    peak_allocated_bytes: u64,
}

/// Per-rank allocation accounting, kept apart from any driver.
///
/// A backend cannot make a driver transition and a counter move together, so
/// this splits every change in two: planning is the whole fallible half and
/// changes nothing, and committing a plan cannot fail. A backend plans before
/// it calls the driver and commits only after the driver agreed, so a refused
/// call leaves every count exactly as it was and an accepted one is never
/// left uncounted.
///
/// Bytes are always in exactly one place: a live rank, or the unattributed
/// total. An allocation whose context is released before it is stays counted,
/// as unattributed, because it is still allocated on the device.
#[derive(Clone, Debug, Default)]
pub struct HostDeviceRankLedgerV1 {
    /// One entry per live context, keyed by the identity naming it.
    ranks: BTreeMap<u64, LedgerRankV1>,
    /// The context each live allocation was attributed to.
    context_of_buffer: BTreeMap<u64, u64>,
    current_bytes: u64,
    peak_bytes: u64,
    unattributed_bytes: u64,
}

impl HostDeviceRankLedgerV1 {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            ranks: BTreeMap::new(),
            context_of_buffer: BTreeMap::new(),
            current_bytes: 0,
            peak_bytes: 0,
            unattributed_bytes: 0,
        }
    }

    /// Records the rank one accepted context was created for.
    pub fn attach_context(&mut self, context: u64, rank: u32, device_ordinal: i32) {
        self.ranks.insert(
            context,
            LedgerRankV1 {
                rank,
                device_ordinal,
                allocated_bytes: 0,
                peak_allocated_bytes: 0,
            },
        );
    }

    #[must_use]
    pub const fn current_bytes(&self) -> u64 {
        self.current_bytes
    }

    #[must_use]
    pub const fn peak_bytes(&self) -> u64 {
        self.peak_bytes
    }

    /// Allocations on the device of one live context, including other contexts.
    #[must_use]
    pub fn allocated_on_context_device(&self, context: u64) -> Option<u64> {
        let ordinal = self.ranks.get(&context)?.device_ordinal;
        self.ranks
            .values()
            .filter(|rank| rank.device_ordinal == ordinal)
            .try_fold(0_u64, |sum, rank| sum.checked_add(rank.allocated_bytes))
    }

    /// The bytes no live rank accounts for, which a report refuses to state.
    ///
    /// Production reads this through [`Self::ranks`], which refuses rather than
    /// reporting counts that do not add up; the number itself is only useful to
    /// a test that wants to see where the bytes went.
    #[cfg(test)]
    #[must_use]
    pub const fn unattributed_bytes(&self) -> u64 {
        self.unattributed_bytes
    }

    /// Plans one allocation without changing anything.
    pub fn plan_allocation(
        &self,
        context: Option<u64>,
        byte_length: u64,
    ) -> HostDeviceResultV1<HostDeviceAllocationPlanV1> {
        let total_bytes = self
            .current_bytes
            .checked_add(byte_length)
            .ok_or(BundleStatusV1::ResourceExhausted)?;
        let owner_bytes = match context {
            Some(context) => self
                .ranks
                .get(&context)
                .ok_or(BundleStatusV1::StaleHandle)?
                .allocated_bytes
                .checked_add(byte_length)
                .ok_or(BundleStatusV1::ResourceExhausted)?,
            None => self
                .unattributed_bytes
                .checked_add(byte_length)
                .ok_or(BundleStatusV1::ResourceExhausted)?,
        };
        Ok(HostDeviceAllocationPlanV1 {
            context,
            total_bytes,
            owner_bytes,
        })
    }

    /// Applies one planned allocation to the allocation it was planned for.
    pub fn commit_allocation(&mut self, buffer: u64, plan: HostDeviceAllocationPlanV1) {
        self.current_bytes = plan.total_bytes;
        self.peak_bytes = self.peak_bytes.max(plan.total_bytes);
        match plan.context {
            Some(context) => {
                if let Some(record) = self.ranks.get_mut(&context) {
                    record.allocated_bytes = plan.owner_bytes;
                    record.peak_allocated_bytes = record.peak_allocated_bytes.max(plan.owner_bytes);
                }
                self.context_of_buffer.insert(buffer, context);
            }
            None => self.unattributed_bytes = plan.owner_bytes,
        }
    }

    /// Plans one release without changing anything.
    ///
    /// An allocation whose context is already gone is subtracted from the
    /// unattributed total, which is where releasing that context moved it.
    pub fn plan_release(
        &self,
        buffer: u64,
        byte_length: u64,
    ) -> HostDeviceResultV1<HostDeviceReleasePlanV1> {
        let total_bytes = self
            .current_bytes
            .checked_sub(byte_length)
            .ok_or(BundleStatusV1::InvalidState)?;
        let live = self
            .context_of_buffer
            .get(&buffer)
            .copied()
            .filter(|context| self.ranks.contains_key(context));
        let owner_bytes = match live {
            Some(context) => self.ranks[&context]
                .allocated_bytes
                .checked_sub(byte_length)
                .ok_or(BundleStatusV1::InvalidState)?,
            None => self
                .unattributed_bytes
                .checked_sub(byte_length)
                .ok_or(BundleStatusV1::InvalidState)?,
        };
        Ok(HostDeviceReleasePlanV1 {
            context: live,
            total_bytes,
            owner_bytes,
        })
    }

    /// Applies one planned release to the allocation it was planned for.
    pub fn commit_release(&mut self, buffer: u64, plan: HostDeviceReleasePlanV1) {
        self.current_bytes = plan.total_bytes;
        match plan.context {
            Some(context) => {
                if let Some(record) = self.ranks.get_mut(&context) {
                    record.allocated_bytes = plan.owner_bytes;
                }
            }
            None => self.unattributed_bytes = plan.owner_bytes,
        }
        self.context_of_buffer.remove(&buffer);
    }

    /// Plans one context release without changing anything.
    ///
    /// Whatever that context still holds becomes unattributed rather than
    /// disappearing: the memory is still allocated, and a report that dropped
    /// it would understate what this backend holds.
    pub fn plan_context_release(
        &self,
        context: u64,
    ) -> HostDeviceResultV1<HostDeviceContextReleasePlanV1> {
        let unattributed_bytes = match self.ranks.get(&context) {
            Some(record) => self
                .unattributed_bytes
                .checked_add(record.allocated_bytes)
                .ok_or(BundleStatusV1::ResourceExhausted)?,
            None => self.unattributed_bytes,
        };
        Ok(HostDeviceContextReleasePlanV1 {
            context,
            unattributed_bytes,
        })
    }

    /// Applies one planned context release.
    pub fn commit_context_release(&mut self, plan: HostDeviceContextReleasePlanV1) {
        self.ranks.remove(&plan.context);
        self.unattributed_bytes = plan.unattributed_bytes;
    }

    /// Every rank this ledger can still account for, refusing where it cannot.
    ///
    /// A report is issued only when every allocated byte belongs to exactly one
    /// live rank and the per-rank counts add up to the total. Anything else is
    /// a refusal, because a rank report that does not add up understates what
    /// the device holds.
    pub fn ranks(&self) -> Result<Vec<HostDeviceLedgerRankV1>, String> {
        if self.ranks.is_empty() {
            return Err("this backend holds no rank context to account for".to_owned());
        }
        if self.unattributed_bytes != 0 {
            return Err(format!(
                "{} allocated byte(s) belong to no rank this backend holds",
                self.unattributed_bytes
            ));
        }
        let mut ranks = self
            .ranks
            .iter()
            .map(|(context, record)| HostDeviceLedgerRankV1 {
                context: *context,
                rank: record.rank,
                device_ordinal: record.device_ordinal,
                allocated_bytes: record.allocated_bytes,
                peak_allocated_bytes: record.peak_allocated_bytes,
            })
            .collect::<Vec<_>>();
        ranks.sort_by_key(|rank| rank.rank);
        if ranks.windows(2).any(|pair| pair[0].rank == pair[1].rank) {
            return Err("this backend holds two contexts for one rank".to_owned());
        }
        // Rank identity is unique, and this ledger counts each byte once.
        let attributed = ranks
            .iter()
            .try_fold(0_u64, |total, rank| total.checked_add(rank.allocated_bytes))
            .ok_or_else(|| "rank allocation counts overflow their total".to_owned())?;
        if attributed
            .checked_add(self.unattributed_bytes)
            .ok_or_else(|| "rank allocation counts overflow their total".to_owned())?
            != self.current_bytes
        {
            return Err(format!(
                "rank allocations total {attributed} against {} allocated bytes",
                self.current_bytes
            ));
        }
        Ok(ranks)
    }
}

#[cfg(test)]
mod ledger_tests {
    use super::{BundleStatusV1, HostDeviceRankLedgerV1};

    const RANK_BYTES: u64 = 87 * 1024 * 1024 * 1024;

    fn four_rank_ledger_v1() -> HostDeviceRankLedgerV1 {
        let mut ledger = HostDeviceRankLedgerV1::new();
        for rank in 0..4_u32 {
            ledger.attach_context(u64::from(rank) + 1, rank, rank as i32);
        }
        ledger
    }

    #[test]
    fn allocations_are_attributed_to_the_rank_that_holds_them() {
        let mut ledger = four_rank_ledger_v1();
        for rank in 0..4_u64 {
            let plan = ledger
                .plan_allocation(Some(rank + 1), RANK_BYTES)
                .expect("a rank this ledger holds takes an allocation");
            ledger.commit_allocation(100 + rank, plan);
        }
        assert_eq!(ledger.current_bytes(), RANK_BYTES * 4);
        assert_eq!(ledger.peak_bytes(), RANK_BYTES * 4);
        assert_eq!(ledger.unattributed_bytes(), 0);
        let ranks = ledger.ranks().expect("every byte belongs to a live rank");
        assert_eq!(ranks.len(), 4);
        for (index, rank) in ranks.iter().enumerate() {
            assert_eq!(rank.rank, index as u32);
            assert_eq!(rank.device_ordinal, index as i32);
            assert_eq!(rank.allocated_bytes, RANK_BYTES);
            assert_eq!(rank.peak_allocated_bytes, RANK_BYTES);
        }

        let plan = ledger
            .plan_release(100, RANK_BYTES)
            .expect("a live allocation releases");
        ledger.commit_release(100, plan);
        let ranks = ledger.ranks().expect("release keeps the counts adding up");
        assert_eq!(ranks[0].allocated_bytes, 0);
        assert_eq!(ranks[0].peak_allocated_bytes, RANK_BYTES);
        assert_eq!(ledger.current_bytes(), RANK_BYTES * 3);
    }

    #[test]
    fn a_plan_that_cannot_be_taken_changes_nothing() {
        let mut ledger = four_rank_ledger_v1();
        let plan = ledger.plan_allocation(Some(1), u64::MAX).expect("plan");
        ledger.commit_allocation(100, plan);

        assert_eq!(
            ledger.plan_allocation(Some(1), 1),
            Err(BundleStatusV1::ResourceExhausted)
        );
        assert_eq!(
            ledger.plan_allocation(None, 1),
            Err(BundleStatusV1::ResourceExhausted)
        );
        assert_eq!(
            ledger.plan_release(404, 1),
            Err(BundleStatusV1::InvalidState),
            "an allocation this ledger never counted has nothing to subtract"
        );
        // Every refusal above left the counts exactly as the one taken plan set.
        assert_eq!(ledger.current_bytes(), u64::MAX);
        assert_eq!(ledger.unattributed_bytes(), 0);
        assert_eq!(
            ledger.ranks().expect("counts still add up")[0].allocated_bytes,
            u64::MAX
        );

        // A context this ledger never attached is a stale handle, not a count,
        // and releasing more than a rank holds refuses rather than wrapping.
        let mut fresh = four_rank_ledger_v1();
        assert_eq!(
            fresh.plan_allocation(Some(99), 1),
            Err(BundleStatusV1::StaleHandle)
        );
        let plan = fresh.plan_allocation(Some(1), 4096).expect("plan");
        fresh.commit_allocation(400, plan);
        assert_eq!(
            fresh.plan_release(400, 8192),
            Err(BundleStatusV1::InvalidState)
        );
        assert_eq!(fresh.current_bytes(), 4096);
        assert_eq!(
            fresh.ranks().expect("counts add up")[0].allocated_bytes,
            4096
        );
    }

    #[test]
    fn releasing_a_context_that_still_holds_memory_never_drops_the_bytes() {
        let mut ledger = four_rank_ledger_v1();
        let plan = ledger.plan_allocation(Some(2), RANK_BYTES).expect("plan");
        ledger.commit_allocation(200, plan);

        let plan = ledger.plan_context_release(2).expect("plan");
        ledger.commit_context_release(plan);
        assert_eq!(ledger.current_bytes(), RANK_BYTES);
        assert_eq!(ledger.unattributed_bytes(), RANK_BYTES);
        let error = ledger
            .ranks()
            .expect_err("a rank report that would understate the device refuses");
        assert!(error.contains("belong to no rank"), "{error}");

        // Releasing the orphaned allocation returns the ledger to a state it
        // can report again, without the released rank reappearing.
        let plan = ledger.plan_release(200, RANK_BYTES).expect("plan");
        ledger.commit_release(200, plan);
        assert_eq!(ledger.current_bytes(), 0);
        assert_eq!(ledger.unattributed_bytes(), 0);
        assert_eq!(ledger.ranks().expect("three ranks remain").len(), 3);
    }

    #[test]
    fn an_allocation_no_rank_owns_is_counted_and_refuses_the_report() {
        let mut ledger = four_rank_ledger_v1();
        let plan = ledger.plan_allocation(None, 4096).expect("plan");
        ledger.commit_allocation(300, plan);
        assert_eq!(ledger.current_bytes(), 4096);
        assert_eq!(ledger.unattributed_bytes(), 4096);
        assert!(ledger.ranks().is_err());
    }

    #[test]
    fn a_ledger_with_no_rank_or_a_repeated_one_reports_nothing() {
        let empty = HostDeviceRankLedgerV1::new();
        assert!(empty.ranks().unwrap_err().contains("no rank context"));

        let mut repeated = HostDeviceRankLedgerV1::new();
        repeated.attach_context(1, 0, 0);
        repeated.attach_context(2, 0, 1);
        assert!(
            repeated
                .ranks()
                .unwrap_err()
                .contains("two contexts for one rank")
        );

        let mut shared_ordinal = HostDeviceRankLedgerV1::new();
        shared_ordinal.attach_context(1, 0, 0);
        shared_ordinal.attach_context(2, 1, 0);
        let ranks = shared_ordinal.ranks().unwrap();
        assert_eq!(ranks.len(), 2);
        assert_eq!(ranks[0].device_ordinal, ranks[1].device_ordinal);
    }
}
