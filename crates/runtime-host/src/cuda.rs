//! The CUDA driver behind the model-neutral host device surface.
//!
//! This module exists only on Linux and only with the `linux-cuda-host`
//! feature, so a macOS host, the default authoring build, and the fake host
//! services never compile a CUDA dependency.
//!
//! Every operation here is one of the generic driver primitives the services
//! layer already names. Nothing in this file knows a stage, a tensor, or a
//! model: handle checking, range checking, and release ordering all happened
//! before a call arrives, and this only performs the driver work.

use std::collections::{BTreeMap, VecDeque};
use std::rc::Rc;

use runtime_bundle_abi::BundleLogitsViewV1;
use runtime_bundle_abi::{
    BundleStatusV1, CustomAddressSpaceV1, CustomOperationV1, LaunchClusterV1, LaunchConfigV1,
    LentLaunchV1, LentRangeV1, LentStreamV1, TensorMapElementV1, TensorMapSwizzleV1,
};
use runtime_cuda::{
    Context, CudaError, DeviceBuffer, Driver, Event, Function, GraphExec, KernelArgument,
    LaunchConfig, MemoryPool, Module, PeerAccessOwnership, PinnedHostBuffer, PinnedTransfer,
    PinnedUploadRing, Stream, TensorMap, TensorMapDataType, TensorMapDescriptor, TensorMapSwizzle,
};
use runtime_engine::{ModelBatchItem, SampledRow};

use crate::sampling::DeviceSamplingSession;

use crate::device::{
    CollectiveBindingV1, DEVICE_POINTER_TABLE_ENTRY_BYTES_V1, DeviceArgumentV1,
    DeviceCustomOperandV1, DeviceObjectV1, DeviceRangeV1, DeviceTensorMapV1,
    GenericCollectivePlanV1, GraphCaptureErrorV1, HostDeviceBackendV1, HostDeviceRankLedgerV1,
    HostDeviceResultV1, LaunchDispatchV1, PeerAccessAcquisitionV1, PinnedFileUploadV1,
    launch_cluster_v1, launch_cooperative_v1, launch_dispatch_v1, plan_generic_collective_v1,
};

/// The cluster bound is one number, stated in two crates that cannot share it.
///
/// `runtime-cuda` links no workspace crate, so it carries its own copy of the
/// portable maximum rather than reading the ABI's. This module is the one place
/// both are in scope, so a copy that drifted from the ABI's fails the build
/// here rather than becoming a launch the ABI would have refused.
const _: () = assert!(
    runtime_cuda::MAXIMUM_CLUSTER_BLOCKS == runtime_bundle_abi::BUNDLE_LAUNCH_V1_MAX_CLUSTER_BLOCKS,
    "the driver crate's cluster bound must be the bundle ABI's"
);

/// One driver-owned object this backend holds on the host's behalf.
enum CudaObjectV1 {
    Offload(runtime_cuda::PageCopyArena),
    Context(Context),
    MemoryPool(MemoryPool),
    Buffer(DeviceBuffer),
    /// Peer access from one context to another, remembered so it can be undone.
    ///
    /// The record exists whether or not this host enabled the edge; whether it
    /// is undone is the services layer's decision, passed to the release.
    PeerAccess {
        owner: Context,
        peer: Context,
    },
    Stream(Stream),
    Event(Event),
    Graph(GraphExec),
    Module(Module),
    Symbol {
        module: DeviceObjectV1,
        function: Function,
    },
}

/// The CUDA implementation of the host's generic device surface.
pub struct CudaHostDeviceV1 {
    driver: Driver,
    objects: BTreeMap<u64, CudaObjectV1>,
    next_identity: u64,
    logs: Vec<Vec<u8>>,
    /// Every allocation count this backend states, planned before it is moved.
    ///
    /// The ledger holds no driver object, so what it decides is decided the
    /// same way with or without a device, and this module only pairs its
    /// answers with the contexts below.
    ledger: HostDeviceRankLedgerV1,
    /// The live context each rank record names, for measuring that rank.
    rank_contexts: BTreeMap<u64, Context>,
    /// Logical rank ownership of default pool handles, even on shared contexts.
    pool_contexts: BTreeMap<u64, u64>,
    tensor_maps: VecDeque<(DeviceTensorMapV1, u64, Rc<TensorMap>)>,
    samplers: BTreeMap<u64, DeviceSamplingSession>,
    pinned_transfers: BTreeMap<u64, PinnedHostBuffer>,
    /// Per stream, staging for ordered uploads still in flight. A ring holds
    /// its streams and destinations until drained, so releases drain first.
    upload_rings: BTreeMap<u64, PinnedUploadRing>,
}

const TENSOR_MAP_CACHE_CAPACITY: usize = 64;
const DIAGNOSTIC_ENTRY_CAPACITY: usize = 4096;
const DIAGNOSTIC_ENTRY_BYTES: usize = 4096;
/// Rank upload read size. Each read covers eight 8 MiB verification chunks,
/// which the bundle reader reads and hashes on parallel workers; two buffers
/// per rank alternate so none is refilled while its copy is in flight.
const PINNED_RANK_UPLOAD_CHUNK_BYTES: usize = 64 * 1024 * 1024;
const PINNED_SERVICE_TRANSFER_CHUNK_BYTES: usize = 1024 * 1024;
const PINNED_UPLOAD_RING_BYTES: usize = 4 * 1024 * 1024;

fn record_diagnostic_v1(entries: &mut Vec<Vec<u8>>, value: &[u8]) {
    if entries.len() == DIAGNOSTIC_ENTRY_CAPACITY {
        // Production launches must not shift this vector on every step after
        // it fills. Probe sessions use a fresh backend and retain their trace.
        return;
    }
    entries.push(value[..value.len().min(DIAGNOSTIC_ENTRY_BYTES)].to_vec());
}

impl CudaHostDeviceV1 {
    /// Initializes the driver this backend will use for the process.
    pub fn new() -> Result<Self, CudaError> {
        Ok(Self {
            driver: Driver::initialize()?,
            objects: BTreeMap::new(),
            next_identity: 1,
            logs: Vec::new(),
            ledger: HostDeviceRankLedgerV1::new(),
            rank_contexts: BTreeMap::new(),
            pool_contexts: BTreeMap::new(),
            tensor_maps: VecDeque::new(),
            samplers: BTreeMap::new(),
            pinned_transfers: BTreeMap::new(),
            upload_rings: BTreeMap::new(),
        })
    }

    /// Returns every diagnostic line the bundle emitted, oldest first.
    #[must_use]
    pub fn logs(&self) -> &[Vec<u8>] {
        &self.logs
    }

    /// Returns the number of driver objects this backend has not released.
    #[must_use]
    pub fn live_object_count(&self) -> usize {
        self.objects.len()
    }

    fn insert(&mut self, object: CudaObjectV1) -> HostDeviceResultV1<DeviceObjectV1> {
        let identity = self.reserve_identity()?;
        self.insert_reserved(identity, object);
        Ok(DeviceObjectV1::new(identity))
    }

    /// Takes the next object identity without recording anything yet.
    ///
    /// An operation that changes driver state reserves its identity first, so
    /// the recording that follows cannot fail and there is no transition left
    /// to undo with a call that could fail in turn.
    fn reserve_identity(&mut self) -> HostDeviceResultV1<u64> {
        let identity = self.next_identity;
        self.next_identity = self
            .next_identity
            .checked_add(1)
            .ok_or(BundleStatusV1::ResourceExhausted)?;
        Ok(identity)
    }

    fn insert_reserved(&mut self, identity: u64, object: CudaObjectV1) {
        self.objects.insert(identity, object);
    }

    fn get(&self, object: DeviceObjectV1) -> HostDeviceResultV1<&CudaObjectV1> {
        self.objects
            .get(&object.get())
            .ok_or(BundleStatusV1::StaleHandle)
    }

    fn context(&self, object: DeviceObjectV1) -> HostDeviceResultV1<&Context> {
        match self.get(object)? {
            CudaObjectV1::Context(context) => Ok(context),
            _ => Err(BundleStatusV1::InvalidArgument),
        }
    }

    fn pool(&self, object: DeviceObjectV1) -> HostDeviceResultV1<&MemoryPool> {
        match self.get(object)? {
            CudaObjectV1::MemoryPool(pool) => Ok(pool),
            _ => Err(BundleStatusV1::InvalidArgument),
        }
    }

    fn buffer(&self, object: DeviceObjectV1) -> HostDeviceResultV1<&DeviceBuffer> {
        match self.get(object)? {
            CudaObjectV1::Buffer(buffer) => Ok(buffer),
            _ => Err(BundleStatusV1::InvalidArgument),
        }
    }

    fn stream(&self, object: DeviceObjectV1) -> HostDeviceResultV1<&Stream> {
        match self.get(object)? {
            CudaObjectV1::Stream(stream) => Ok(stream),
            _ => Err(BundleStatusV1::InvalidArgument),
        }
    }

    fn event(&self, object: DeviceObjectV1) -> HostDeviceResultV1<&Event> {
        match self.get(object)? {
            CudaObjectV1::Event(event) => Ok(event),
            _ => Err(BundleStatusV1::InvalidArgument),
        }
    }

    fn module(&self, object: DeviceObjectV1) -> HostDeviceResultV1<&Module> {
        match self.get(object)? {
            CudaObjectV1::Module(module) => Ok(module),
            _ => Err(BundleStatusV1::InvalidArgument),
        }
    }

    /// Resolves one recorded symbol to the live function it names.
    fn function(&self, object: DeviceObjectV1) -> HostDeviceResultV1<&Function> {
        let CudaObjectV1::Symbol { function, .. } = self.get(object)? else {
            return Err(BundleStatusV1::InvalidArgument);
        };
        Ok(function)
    }

    pub(crate) fn sample_logits_view(
        &mut self,
        context_object: DeviceObjectV1,
        range: DeviceRangeV1,
        view: BundleLogitsViewV1,
        items: &[&ModelBatchItem],
        dispatch: Option<crate::executor::SamplingDispatch>,
    ) -> HostDeviceResultV1<Vec<SampledRow>> {
        let context = self.context(context_object)?.clone();
        let logits = self.buffer(range.object)?.clone();
        let sampler = if let Some(sampler) = self.samplers.get_mut(&context_object.get()) {
            sampler
        } else {
            let sampler = DeviceSamplingSession::new(context)?;
            self.samplers.insert(context_object.get(), sampler);
            self.samplers
                .get_mut(&context_object.get())
                .expect("sampler was inserted")
        };
        sampler.sample_with_dispatch(&logits, view, items, dispatch)
    }

    /// Starts sampling a view on its context; `collect_logits_view` on the
    /// same context returns the rows.
    pub(crate) fn issue_logits_view(
        &mut self,
        context_object: DeviceObjectV1,
        range: DeviceRangeV1,
        view: BundleLogitsViewV1,
        items: &[&ModelBatchItem],
        dispatch: Option<crate::executor::SamplingDispatch>,
    ) -> HostDeviceResultV1<()> {
        let context = self.context(context_object)?.clone();
        let logits = self.buffer(range.object)?.clone();
        let sampler = if let Some(sampler) = self.samplers.get_mut(&context_object.get()) {
            sampler
        } else {
            let sampler = DeviceSamplingSession::new(context)?;
            self.samplers.insert(context_object.get(), sampler);
            self.samplers
                .get_mut(&context_object.get())
                .expect("sampler was inserted")
        };
        sampler.issue_with_dispatch(&logits, view, items, dispatch)
    }

    pub(crate) fn collect_logits_view(
        &mut self,
        context_object: DeviceObjectV1,
        items: &[&ModelBatchItem],
    ) -> HostDeviceResultV1<Vec<SampledRow>> {
        self.samplers
            .get_mut(&context_object.get())
            .ok_or(BundleStatusV1::InvalidArgument)?
            .collect(items)
    }

    pub(crate) fn score_logits_view(
        &mut self,
        context_object: DeviceObjectV1,
        range: DeviceRangeV1,
        view: BundleLogitsViewV1,
        rows: &[crate::executor::DevicePromptRowV1<'_>],
        dispatch: Option<crate::executor::SamplingDispatch>,
    ) -> HostDeviceResultV1<(Vec<SampledRow>, Vec<SampledRow>)> {
        let context = self.context(context_object)?.clone();
        let logits = self.buffer(range.object)?.clone();
        let sampler = if let Some(sampler) = self.samplers.get_mut(&context_object.get()) {
            sampler
        } else {
            let sampler = DeviceSamplingSession::new(context)?;
            self.samplers.insert(context_object.get(), sampler);
            self.samplers
                .get_mut(&context_object.get())
                .expect("sampler was inserted")
        };
        sampler.score_prompt(&logits, view, rows, dispatch)
    }

    fn upload_pinned_rank_chunks(
        &self,
        stream_object: DeviceObjectV1,
        destination_object: DeviceObjectV1,
        source_offset: u64,
        byte_length: u64,
        read_chunk: &mut dyn FnMut(u64, &mut [u8]) -> bool,
    ) -> HostDeviceResultV1<()> {
        if byte_length == 0 {
            return Ok(());
        }
        let stream = self.stream(stream_object)?.clone();
        let destination = self.buffer(destination_object)?.clone();
        let context = stream.context();
        let capacity = usize::try_from(byte_length)
            .unwrap_or(usize::MAX)
            .min(PINNED_RANK_UPLOAD_CHUNK_BYTES);
        let mut first = PinnedHostBuffer::new(&context, capacity).map_err(driver_status)?;
        let mut second = PinnedHostBuffer::new(&context, capacity).map_err(driver_status)?;
        let mut copied = 0_u64;
        let Some((target, count)) = read_pinned_upload_chunk(
            &mut first,
            source_offset,
            byte_length,
            &mut copied,
            read_chunk,
        )?
        else {
            return Ok(());
        };
        let mut pending_first = destination
            .copy_from_pinned_at_async(target, &first, 0, count, &stream)
            .map_err(driver_status)?;
        loop {
            let next = read_pinned_upload_chunk(
                &mut second,
                source_offset,
                byte_length,
                &mut copied,
                read_chunk,
            )?;
            pending_first.finish();
            let Some((target, count)) = next else {
                break;
            };
            let pending_second = destination
                .copy_from_pinned_at_async(target, &second, 0, count, &stream)
                .map_err(driver_status)?;
            let next = read_pinned_upload_chunk(
                &mut first,
                source_offset,
                byte_length,
                &mut copied,
                read_chunk,
            )?;
            pending_second.finish();
            let Some((target, count)) = next else {
                break;
            };
            pending_first = destination
                .copy_from_pinned_at_async(target, &first, 0, count, &stream)
                .map_err(driver_status)?;
        }
        Ok(())
    }

    fn pinned_transfer_for(
        &mut self,
        stream_object: DeviceObjectV1,
        stream: &Stream,
        requested: usize,
    ) -> HostDeviceResultV1<&mut PinnedHostBuffer> {
        let capacity = requested.min(PINNED_SERVICE_TRANSFER_CHUNK_BYTES);
        let replace = self
            .pinned_transfers
            .get(&stream_object.get())
            .is_none_or(|buffer| buffer.len() < capacity);
        if replace {
            let buffer =
                PinnedHostBuffer::new(&stream.context(), capacity).map_err(driver_status)?;
            self.pinned_transfers.insert(stream_object.get(), buffer);
        }
        Ok(self
            .pinned_transfers
            .get_mut(&stream_object.get())
            .expect("a pinned transfer buffer was inserted"))
    }

    /// Removes one object only after its driver release succeeded.
    fn release(
        &mut self,
        object: DeviceObjectV1,
        release: impl FnOnce(&CudaObjectV1) -> HostDeviceResultV1<()>,
    ) -> HostDeviceResultV1<()> {
        release(self.get(object)?)?;
        self.objects.remove(&object.get());
        Ok(())
    }

    /// Sums one range per rank lane-wise and writes the total back.
    ///
    /// Every rank's bytes are staged through the host on that allocation's own
    /// stream, and each transfer is synchronized before the next step reads it.
    /// The fold itself, its seed, and its order belong to the plan, so this
    /// backend supplies bytes and nothing else and cannot answer differently
    /// from the host-memory one. The finished total is encoded back at the
    /// declared width once, and nothing here knows what the lanes mean.
    fn sum_through_host_v1(&mut self, plan: &GenericCollectivePlanV1) -> HostDeviceResultV1<()> {
        // Every span is resolved before any of them is read, so a span this
        // backend cannot resolve moves no bytes at all rather than leaving the
        // ranks it already read reduced and the rest not.
        for range in &plan.ranges {
            self.range(*range)?;
        }
        let byte_length = plan.byte_length;
        let totals = plan.fold_lanes_v1(|range| {
            let mut staged = vec![0_u8; byte_length];
            self.read_range_through_host_v1(range, &mut staged)?;
            Ok(staged)
        })?;
        let reduced = plan.collective.encode_lanes_v1(&totals);
        for range in &plan.ranges {
            self.write_range_through_host_v1(*range, &reduced)?;
        }
        Ok(())
    }

    /// Reads one range on its own allocation's stream, then synchronizes.
    fn read_range_through_host_v1(
        &self,
        range: DeviceRangeV1,
        destination: &mut [u8],
    ) -> HostDeviceResultV1<()> {
        let (buffer, offset, length) = self.range(range)?;
        if length != destination.len() {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let stream = buffer.owner_stream();
        buffer
            .copy_to_bytes_at_async(offset, destination, &stream)
            .map_err(driver_status)?;
        stream.synchronize().map_err(driver_status)
    }

    /// Writes one range in order on its own allocation's stream and waits.
    fn write_range_through_host_v1(
        &self,
        range: DeviceRangeV1,
        source: &[u8],
    ) -> HostDeviceResultV1<()> {
        let (buffer, offset, length) = self.range(range)?;
        if length != source.len() {
            return Err(BundleStatusV1::InvalidArgument);
        }
        buffer
            .copy_from_bytes_at(offset, source, &buffer.owner_stream())
            .map_err(driver_status)
    }

    /// The context every part of one launch has to share.
    ///
    /// A symbol belongs to the module it was resolved from, and that module
    /// belongs to one context. That context is the launch's, and a stream or an
    /// allocation from any other one names memory it cannot reach.
    fn launch_context(&self, symbol: DeviceObjectV1) -> HostDeviceResultV1<Context> {
        let CudaObjectV1::Symbol { module, .. } = self.get(symbol)? else {
            return Err(BundleStatusV1::InvalidArgument);
        };
        Ok(self.module(*module)?.context())
    }

    /// Settles every question one launch raises before any of it is built.
    ///
    /// The stream, each range, and each tensor map are resolved against the
    /// launch's own context, each range is bounds checked, and each map is
    /// checked for the address and footprint a tiled encoding requires. An
    /// argument this version carries no launch encoding for is refused here as
    /// well, so a list containing one costs no encode.
    fn preflight_launch(
        &self,
        launch_context: &Context,
        stream: DeviceObjectV1,
        arguments: &[DeviceArgumentV1],
    ) -> HostDeviceResultV1<()> {
        if self.stream(stream)?.context() != *launch_context {
            return Err(BundleStatusV1::InvalidArgument);
        }
        for argument in arguments {
            let range = match argument {
                DeviceArgumentV1::Range(range) => *range,
                DeviceArgumentV1::TensorMap(map) => map.range,
                // Opaque bytes have no declared launch encoding in V1.
                DeviceArgumentV1::OpaqueBytes(_) => {
                    return Err(BundleStatusV1::Unsupported);
                }
                // A by-value address array is the one argument that may name
                // another context's memory, which is what a peer-reading kernel
                // is for. Each slot is still resolved and bounds checked; which
                // contexts it may reach is the services layer's rule, applied
                // against the leases it holds before anything reaches here.
                DeviceArgumentV1::DevicePointerArray(slots) => {
                    for slot in slots.iter().flatten() {
                        self.range(*slot)?;
                    }
                    continue;
                }
                DeviceArgumentV1::NullDevicePointer
                | DeviceArgumentV1::ScalarU32(_)
                | DeviceArgumentV1::ScalarI32(_)
                | DeviceArgumentV1::ScalarU64(_) => continue,
            };
            let (buffer, offset, byte_len) = self.range(range)?;
            if buffer.owner_context() != *launch_context {
                return Err(BundleStatusV1::InvalidArgument);
            }
            if let DeviceArgumentV1::TensorMap(map) = argument {
                launch_context
                    .validate_tensor_map(buffer, offset, byte_len, tensor_map_descriptor(map)?)
                    .map_err(driver_status)?;
            }
        }
        Ok(())
    }

    /// Encodes one bundle-declared tensor map over the range it names.
    ///
    /// The map is encoded in the launch's own context, not the allocation's, so
    /// a descriptor built here is one that launch can use. Everything this
    /// needs has already been settled by the preflight; the checks below the
    /// encode repeat it rather than trusting the caller.
    fn encode_tensor_map(
        &self,
        launch_context: &Context,
        map: &DeviceTensorMapV1,
    ) -> HostDeviceResultV1<TensorMap> {
        let (buffer, offset, byte_len) = self.range(map.range)?;
        launch_context
            .tensor_map(buffer, offset, byte_len, tensor_map_descriptor(map)?)
            .map_err(driver_status)
    }

    fn cached_tensor_map(
        &mut self,
        context: &Context,
        descriptor: &DeviceTensorMapV1,
    ) -> HostDeviceResultV1<Rc<TensorMap>> {
        let (buffer, offset, length) = self.range(descriptor.range)?;
        let address = buffer
            .device_address_at(offset, length)
            .map_err(driver_status)?;
        if let Some(index) = self
            .tensor_maps
            .iter()
            .position(|(cached, cached_address, _)| {
                cached == descriptor && *cached_address == address
            })
        {
            let entry = self.tensor_maps.remove(index).expect("cache index exists");
            let map = Rc::clone(&entry.2);
            self.tensor_maps.push_back(entry);
            return Ok(map);
        }
        let encoded = Rc::new(self.encode_tensor_map(context, descriptor)?);
        if self.tensor_maps.len() == TENSOR_MAP_CACHE_CAPACITY {
            self.tensor_maps.pop_front();
        }
        self.tensor_maps
            .push_back((*descriptor, address, Rc::clone(&encoded)));
        Ok(encoded)
    }

    fn range(&self, range: DeviceRangeV1) -> HostDeviceResultV1<(&DeviceBuffer, usize, usize)> {
        let buffer = self.buffer(range.object)?;
        let offset = usize::try_from(range.offset).map_err(|_| BundleStatusV1::InvalidArgument)?;
        let length = usize::try_from(range.length).map_err(|_| BundleStatusV1::InvalidArgument)?;
        let end = offset
            .checked_add(length)
            .ok_or(BundleStatusV1::InvalidArgument)?;
        if end > buffer.byte_len() {
            return Err(BundleStatusV1::InvalidArgument);
        }
        Ok((buffer, offset, length))
    }
}

fn read_pinned_upload_chunk(
    buffer: &mut PinnedHostBuffer,
    source_offset: u64,
    byte_length: u64,
    copied: &mut u64,
    read_chunk: &mut dyn FnMut(u64, &mut [u8]) -> bool,
) -> HostDeviceResultV1<Option<(usize, usize)>> {
    if *copied == byte_length {
        return Ok(None);
    }
    let count = usize::try_from(byte_length - *copied)
        .unwrap_or(usize::MAX)
        .min(buffer.len());
    let file_offset = source_offset
        .checked_add(*copied)
        .ok_or(BundleStatusV1::InvalidArgument)?;
    if !read_chunk(file_offset, &mut buffer.as_mut_slice()[..count]) {
        return Err(BundleStatusV1::HostError);
    }
    let target = usize::try_from(*copied).map_err(|_| BundleStatusV1::InvalidArgument)?;
    *copied += count as u64;
    Ok(Some((target, count)))
}

fn read_pinned_upload_wave(
    buffers: &mut [PinnedHostBuffer],
    uploads: &mut [PinnedFileUploadV1<'_>],
    copied: &mut [u64],
) -> HostDeviceResultV1<Vec<Option<(usize, usize)>>> {
    let mut ranges = Vec::with_capacity(uploads.len());
    std::thread::scope(|scope| {
        let mut readers = Vec::with_capacity(uploads.len());
        for ((buffer, upload), progress) in buffers.iter_mut().zip(uploads).zip(copied.iter()) {
            if *progress == upload.byte_length {
                ranges.push(None);
                continue;
            }
            let count = usize::try_from(upload.byte_length - *progress)
                .unwrap_or(usize::MAX)
                .min(buffer.len());
            let offset = upload
                .source_offset
                .checked_add(*progress)
                .ok_or(BundleStatusV1::InvalidArgument)?;
            let target = usize::try_from(*progress).map_err(|_| BundleStatusV1::InvalidArgument)?;
            let chunk = &mut buffer.as_mut_slice()[..count];
            let read = &mut *upload.read_chunk;
            readers.push(scope.spawn(move || read(offset, chunk)));
            ranges.push(Some((target, count)));
        }
        let mut all_read = true;
        for reader in readers {
            all_read &= reader.join().unwrap_or(false);
        }
        if !all_read {
            return Err(BundleStatusV1::HostError);
        }
        Ok(())
    })?;
    for (range, progress) in ranges.iter().zip(copied.iter_mut()) {
        if let Some((_, count)) = range {
            *progress += *count as u64;
        }
    }
    Ok(ranges)
}

fn enqueue_pinned_upload_wave<'a>(
    buffers: &'a [PinnedHostBuffer],
    streams: &[Stream],
    destinations: &[DeviceBuffer],
    ranges: &[Option<(usize, usize)>],
) -> HostDeviceResultV1<Vec<PinnedTransfer<'a>>> {
    buffers
        .iter()
        .zip(streams)
        .zip(destinations)
        .zip(ranges)
        .filter_map(|(((buffer, stream), destination), range)| {
            range.map(|(offset, count)| {
                destination
                    .copy_from_pinned_at_async(offset, buffer, 0, count, stream)
                    .map_err(driver_status)
            })
        })
        .collect()
}

/// Maps one closed ABI element type onto the driver's descriptor code.
fn tensor_map_data_type(element_type: TensorMapElementV1) -> HostDeviceResultV1<TensorMapDataType> {
    match element_type {
        TensorMapElementV1::U8 => Ok(TensorMapDataType::Uint8),
        TensorMapElementV1::U16 => Ok(TensorMapDataType::Uint16),
        TensorMapElementV1::I32 => Ok(TensorMapDataType::Int32),
        TensorMapElementV1::Bf16 => Ok(TensorMapDataType::Bfloat16),
        TensorMapElementV1::F16 => Ok(TensorMapDataType::Float16),
        TensorMapElementV1::F32 => Ok(TensorMapDataType::Float32),
        TensorMapElementV1::Fp8E4M3 => Ok(TensorMapDataType::Fp8E4M3),
        TensorMapElementV1::Fp8E5M2 => Ok(TensorMapDataType::Fp8E5M2),
        TensorMapElementV1::U4Align8B => Ok(TensorMapDataType::U4Align8B),
        TensorMapElementV1::U4Align16B => Ok(TensorMapDataType::U4Align16B),
        TensorMapElementV1::U6Align16B => Ok(TensorMapDataType::U6Align16B),
    }
}

/// Maps one closed ABI swizzle onto the driver's own swizzle.
const fn tensor_map_swizzle(swizzle: TensorMapSwizzleV1) -> TensorMapSwizzle {
    match swizzle {
        TensorMapSwizzleV1::None => TensorMapSwizzle::None,
        TensorMapSwizzleV1::Bytes32 => TensorMapSwizzle::Bytes32,
        TensorMapSwizzleV1::Bytes64 => TensorMapSwizzle::Bytes64,
        TensorMapSwizzleV1::Bytes128 => TensorMapSwizzle::Bytes128,
    }
}

fn tensor_map_descriptor(map: &DeviceTensorMapV1) -> HostDeviceResultV1<TensorMapDescriptor> {
    Ok(TensorMapDescriptor {
        rank: map.rank,
        data_type: tensor_map_data_type(map.element_type)?,
        swizzle: tensor_map_swizzle(map.swizzle),
        interleave: map.interleave,
        l2_promotion: map.l2_promotion,
        oob_fill: map.oob_fill,
        global_dims: map.global_dims,
        global_strides_bytes: map.global_strides_bytes,
        box_dims: map.box_dims,
        element_strides: map.element_strides,
    })
}

/// Maps one driver failure onto the ABI's status vocabulary.
///
/// The host does not forward a driver code across the boundary, because a
/// bundle cannot act on one. It reports that the host's device work failed.
fn driver_status(error: CudaError) -> BundleStatusV1 {
    eprintln!("CUDA driver operation failed: {error}");
    match error {
        CudaError::InvalidOrdinal(_)
        | CudaError::InvalidKernelName
        | CudaError::InvalidLaunch
        | CudaError::BufferSize { .. }
        | CudaError::BufferRange { .. }
        // A cluster past the portable bound is a shape no device would have
        // taken, not a device that could not take this one.
        | CudaError::ClusterBlocksExceeded { .. }
        | CudaError::ContextMismatch => BundleStatusV1::InvalidArgument,
        CudaError::UnsupportedDevice { .. }
        | CudaError::MemoryPoolsUnsupported { .. }
        | CudaError::ClusterLaunchUnsupported { .. }
        | CudaError::ProgrammaticLaunchUnsupported { .. } => BundleStatusV1::Unsupported,
        CudaError::ResourceReleased(_) => BundleStatusV1::StaleHandle,
        _ => BundleStatusV1::HostError,
    }
}

/// Builds the driver launch shape, carrying the cluster the config asked for.
///
/// [`launch_cluster_v1`] has already decided whether this is a cluster launch
/// and refused a shape no launch could take, so what reaches the driver crate is
/// either the one-block default or a cluster the grid divides into.
/// Whether this process may be an MPS client: an MPS environment variable is
/// set, or the control daemon's default pipe exists (clients connect to it
/// without any variable). MPS clients share SMs spatially rather than
/// time-slicing, so another client's spinning grids can hold SMs this one's
/// need.
fn mps_client() -> bool {
    mps_client_from(
        |name| std::env::var_os(name).is_some_and(|value| !value.is_empty()),
        std::path::Path::new(MPS_DEFAULT_CONTROL).exists(),
    )
}

/// The MPS control daemon's default pipe.
const MPS_DEFAULT_CONTROL: &str = "/tmp/nvidia-mps/control";

/// `mps_client` over the variables set (non-empty) and whether the default
/// control pipe exists. A heuristic, not a client query: a stale pipe or
/// variable reads as a client, which only makes the bundles refuse.
fn mps_client_from(set: impl Fn(&str) -> bool, default_control: bool) -> bool {
    [
        "CUDA_MPS_PIPE_DIRECTORY",
        "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
        "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT",
    ]
    .iter()
    .any(|name| set(name))
        || default_control
}

fn launch_config(
    config: LaunchConfigV1,
    cluster: Option<LaunchClusterV1>,
) -> HostDeviceResultV1<LaunchConfig> {
    let programmatic_stream_serialization = matches!(
        launch_dispatch_v1(config)?,
        LaunchDispatchV1::ProgrammaticStreamSerialization
    );
    let cluster = cluster.map_or(LaunchConfig::NO_CLUSTER, |cluster| {
        (cluster.x, cluster.y, cluster.z)
    });
    LaunchConfig {
        grid: (config.grid_x, config.grid_y, config.grid_z),
        block: (config.block_x, config.block_y, config.block_z),
        shared_memory_bytes: config.shared_memory_bytes,
        cluster,
        programmatic_stream_serialization,
    }
    .validate()
    .map_err(driver_status)
}

impl HostDeviceBackendV1 for CudaHostDeviceV1 {
    fn create_offload_arena(
        &mut self,
        buffer: DeviceObjectV1,
        page_bytes: usize,
        pages: usize,
    ) -> HostDeviceResultV1<DeviceObjectV1> {
        let arena =
            runtime_cuda::PageCopyArena::new(self.buffer(buffer)?.clone(), page_bytes, pages)
                .map_err(driver_status)?;
        self.insert(CudaObjectV1::Offload(arena))
    }
    fn submit_offload_chunk(
        &mut self,
        arena: DeviceObjectV1,
        direction: runtime_kv::KvTransferDirection,
        pages: &[runtime_kv::KvPagePair],
    ) -> HostDeviceResultV1<u64> {
        let Some(CudaObjectV1::Offload(arena)) = self.objects.get_mut(&arena.get()) else {
            return Err(BundleStatusV1::StaleHandle);
        };
        let direction = match direction {
            runtime_kv::KvTransferDirection::Out => runtime_cuda::PageCopyDirection::ToHost,
            runtime_kv::KvTransferDirection::In => runtime_cuda::PageCopyDirection::ToDevice,
        };
        let pages: Vec<_> = pages
            .iter()
            .map(|p| runtime_cuda::PageCopyPair {
                device_page: p.device.index(),
                host_page: p.host.index(),
            })
            .collect();
        arena.submit(direction, &pages).map_err(driver_status)
    }
    fn query_offload_chunk(
        &mut self,
        arena: DeviceObjectV1,
        chunk: u64,
    ) -> HostDeviceResultV1<bool> {
        let Some(CudaObjectV1::Offload(arena)) = self.objects.get_mut(&arena.get()) else {
            return Err(BundleStatusV1::StaleHandle);
        };
        arena.query(chunk).map_err(driver_status)
    }
    fn fence_offload_arena(&mut self, arena: DeviceObjectV1) -> HostDeviceResultV1<()> {
        let Some(CudaObjectV1::Offload(arena)) = self.objects.get_mut(&arena.get()) else {
            return Err(BundleStatusV1::StaleHandle);
        };
        arena.drain().map_err(driver_status)
    }
    fn release_offload_arena(&mut self, arena: DeviceObjectV1) -> HostDeviceResultV1<()> {
        self.fence_offload_arena(arena)?;
        self.objects.remove(&arena.get());
        Ok(())
    }
    fn reserve_sampling_workspace(
        &mut self,
        context_object: DeviceObjectV1,
        vocabulary: usize,
    ) -> HostDeviceResultV1<()> {
        let plan = crate::sampling_workspace::SamplingWorkspace::new(vocabulary)?;
        if let Some(sampler) = self.samplers.get_mut(&context_object.get()) {
            return sampler.reserve(vocabulary, plan);
        }
        let context = self.context(context_object)?.clone();
        let mut sampler = DeviceSamplingSession::new(context)?;
        sampler.reserve(vocabulary, plan)?;
        self.samplers.insert(context_object.get(), sampler);
        Ok(())
    }

    fn free_device_memory(&self, context: DeviceObjectV1) -> HostDeviceResultV1<u64> {
        self.context(context)?
            .memory_info()
            .map(|(free, _)| free)
            .map_err(driver_status)
    }
    fn upload_pinned_file_range(
        &mut self,
        stream: DeviceObjectV1,
        destination: DeviceObjectV1,
        source_offset: u64,
        byte_length: u64,
        read_chunk: &mut dyn FnMut(u64, &mut [u8]) -> bool,
    ) -> Option<HostDeviceResultV1<()>> {
        Some(self.upload_pinned_rank_chunks(
            stream,
            destination,
            source_offset,
            byte_length,
            read_chunk,
        ))
    }
    fn upload_pinned_file_ranges(
        &mut self,
        uploads: &mut [PinnedFileUploadV1<'_>],
    ) -> Option<HostDeviceResultV1<()>> {
        Some((|| {
            if uploads.is_empty() {
                return Ok(());
            }
            let streams: Vec<_> = uploads
                .iter()
                .map(|upload| self.stream(upload.stream).cloned())
                .collect::<HostDeviceResultV1<_>>()?;
            let destinations: Vec<_> = uploads
                .iter()
                .map(|upload| self.buffer(upload.destination).cloned())
                .collect::<HostDeviceResultV1<_>>()?;
            let mut first = Vec::with_capacity(uploads.len());
            let mut second = Vec::with_capacity(uploads.len());
            for (stream, upload) in streams.iter().zip(uploads.iter()) {
                let capacity = usize::try_from(upload.byte_length)
                    .unwrap_or(usize::MAX)
                    .clamp(1, PINNED_RANK_UPLOAD_CHUNK_BYTES);
                first.push(
                    PinnedHostBuffer::new(&stream.context(), capacity).map_err(driver_status)?,
                );
                second.push(
                    PinnedHostBuffer::new(&stream.context(), capacity).map_err(driver_status)?,
                );
            }
            let mut copied = vec![0; uploads.len()];
            let mut ranges = read_pinned_upload_wave(&mut first, uploads, &mut copied)?;
            while ranges.iter().any(Option::is_some) {
                let pending = enqueue_pinned_upload_wave(&first, &streams, &destinations, &ranges)?;
                let next = read_pinned_upload_wave(&mut second, uploads, &mut copied)?;
                for transfer in pending {
                    transfer.finish();
                }
                if next.iter().all(Option::is_none) {
                    break;
                }
                let pending = enqueue_pinned_upload_wave(&second, &streams, &destinations, &next)?;
                ranges = read_pinned_upload_wave(&mut first, uploads, &mut copied)?;
                for transfer in pending {
                    transfer.finish();
                }
            }
            Ok(())
        })())
    }
    fn create_context(
        &mut self,
        rank: u32,
        device_ordinal: i32,
    ) -> HostDeviceResultV1<DeviceObjectV1> {
        // Reuse the wrapper so resources on this ordinal share one Context identity.
        let context = match self
            .rank_contexts
            .values()
            .find(|context| context.device_ordinal() == device_ordinal)
        {
            Some(context) => context.clone(),
            None => self
                .driver
                .retain_primary_context(device_ordinal)
                .map_err(driver_status)?,
        };
        let object = self.insert(CudaObjectV1::Context(context.clone()))?;
        // The rank this context was created for is the only place that number
        // is stated, so a per-rank memory report names it rather than guessing
        // one from an ordinal.
        self.ledger
            .attach_context(object.get(), rank, device_ordinal);
        self.rank_contexts.insert(object.get(), context);
        Ok(object)
    }

    fn release_context(&mut self, context: DeviceObjectV1) -> HostDeviceResultV1<()> {
        // Whatever this context still holds is planned into the unattributed
        // total before the driver is asked, so a release this backend cannot
        // account for changes nothing, and one it can never drops the bytes.
        let plan = self.ledger.plan_context_release(context.get())?;
        self.samplers.remove(&context.get());
        // A primary context is released when the last reference to it drops.
        self.release(context, |object| match object {
            CudaObjectV1::Context(_) => Ok(()),
            _ => Err(BundleStatusV1::InvalidArgument),
        })?;
        self.ledger.commit_context_release(plan);
        self.rank_contexts.remove(&context.get());
        Ok(())
    }

    fn create_memory_pool(
        &mut self,
        context: DeviceObjectV1,
    ) -> HostDeviceResultV1<DeviceObjectV1> {
        let pool = self
            .context(context)?
            .default_memory_pool()
            .map_err(driver_status)?;
        let object = self.insert(CudaObjectV1::MemoryPool(pool))?;
        self.pool_contexts.insert(object.get(), context.get());
        Ok(object)
    }

    fn memory_pool_peer_access(
        &mut self,
        pool: DeviceObjectV1,
        peer_context: DeviceObjectV1,
    ) -> HostDeviceResultV1<bool> {
        let ordinal = self.context(peer_context)?.device_ordinal();
        self.pool(pool)?.peer_access(ordinal).map_err(driver_status)
    }

    fn set_memory_pool_peer_access(
        &mut self,
        pool: DeviceObjectV1,
        peer_context: DeviceObjectV1,
        enabled: bool,
    ) -> HostDeviceResultV1<()> {
        let ordinal = self.context(peer_context)?.device_ordinal();
        self.pool(pool)?
            .set_peer_access(ordinal, enabled)
            .map_err(driver_status)
    }

    fn release_memory_pool(&mut self, pool: DeviceObjectV1) -> HostDeviceResultV1<()> {
        // A default pool belongs to its device, so releasing the host's handle
        // to it frees nothing on the driver.
        self.release(pool, |object| match object {
            CudaObjectV1::MemoryPool(_) => Ok(()),
            _ => Err(BundleStatusV1::InvalidArgument),
        })?;
        self.pool_contexts.remove(&pool.get());
        Ok(())
    }

    fn allocate_device_buffer(
        &mut self,
        pool: DeviceObjectV1,
        stream: Option<DeviceObjectV1>,
        byte_length: u64,
        alignment: u64,
    ) -> HostDeviceResultV1<DeviceObjectV1> {
        if byte_length == 0 || alignment == 0 || !alignment.is_power_of_two() {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let stream = stream.ok_or(BundleStatusV1::InvalidArgument)?;
        let bytes = usize::try_from(byte_length).map_err(|_| BundleStatusV1::ResourceExhausted)?;
        let stream = self.stream(stream)?;
        let buffer = self
            .pool(pool)?
            .allocate(stream, bytes)
            .map_err(driver_status)?;
        // A pool allocation exists only once its stream executes it, and a
        // bundle may first write it from another stream or copy engine. Every
        // allocation is made while an instance is created, so waiting here
        // costs no step and leaves nothing a later write could overtake.
        if let Err(error) = stream.synchronize() {
            let _ = buffer.release();
            return Err(driver_status(error));
        }
        // An allocation belongs to the context its pool does, and that is the
        // rank whose memory it consumed. Every count this allocation changes,
        // and the identity that will name it, are settled before either is
        // recorded, so nothing after the driver accepted it can fail and leave
        // a live allocation counted differently from how it is held.
        let owner_identity = self.pool_contexts.get(&pool.get()).copied();
        let plan = self.ledger.plan_allocation(owner_identity, byte_length)?;
        let identity = self.reserve_identity()?;
        // The services layer checks the alignment against the address this
        // returned and releases it if the driver's guarantee is too weak.
        self.insert_reserved(identity, CudaObjectV1::Buffer(buffer));
        self.ledger.commit_allocation(identity, plan);
        Ok(DeviceObjectV1::new(identity))
    }

    fn allocate_mapped_host_buffer(
        &mut self,
        stream: DeviceObjectV1,
        byte_length: u64,
        fill: &mut dyn FnMut(&mut [u8]) -> bool,
    ) -> HostDeviceResultV1<DeviceObjectV1> {
        if byte_length == 0 {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let bytes = usize::try_from(byte_length).map_err(|_| BundleStatusV1::ResourceExhausted)?;
        let mut buffer = DeviceBuffer::allocate_mapped_host(self.stream(stream)?, bytes)
            .map_err(driver_status)?;
        if !fill(buffer.mapped_host_bytes_mut().map_err(driver_status)?) {
            let _ = buffer.release();
            return Err(BundleStatusV1::HostError);
        }
        // Host memory: the device ledger does not count it.
        let identity = self.reserve_identity()?;
        self.insert_reserved(identity, CudaObjectV1::Buffer(buffer));
        Ok(DeviceObjectV1::new(identity))
    }

    fn mapped_host_address(&mut self, buffer: DeviceObjectV1) -> HostDeviceResultV1<usize> {
        self.buffer(buffer)?
            .mapped_host_address()
            .ok_or(BundleStatusV1::InvalidArgument)
    }

    fn release_device_buffer(&mut self, buffer: DeviceObjectV1) -> HostDeviceResultV1<()> {
        // An ordered upload into this buffer may still hold it.
        self.upload_rings
            .values_mut()
            .for_each(PinnedUploadRing::synchronize);
        if self.buffer(buffer)?.is_mapped_host() {
            return self.release(buffer, |object| match object {
                CudaObjectV1::Buffer(buffer) => buffer.release().map_err(driver_status),
                _ => Err(BundleStatusV1::InvalidArgument),
            });
        }
        let byte_length = self.buffer(buffer)?.byte_len() as u64;
        // Planned before the driver call and applied after it, so a release the
        // driver refuses leaves this allocation held and counted as it was.
        let plan = self.ledger.plan_release(buffer.get(), byte_length)?;
        self.release(buffer, |object| match object {
            CudaObjectV1::Buffer(buffer) => buffer.release().map_err(driver_status),
            _ => Err(BundleStatusV1::InvalidArgument),
        })?;
        self.ledger.commit_release(buffer.get(), plan);
        Ok(())
    }

    fn allocation_satisfies_alignment(
        &self,
        buffer: DeviceObjectV1,
        alignment: u64,
    ) -> HostDeviceResultV1<bool> {
        Ok(self.buffer(buffer)?.is_aligned_to(alignment))
    }

    fn acquire_peer_access(
        &mut self,
        owner_context: DeviceObjectV1,
        peer_context: DeviceObjectV1,
    ) -> HostDeviceResultV1<PeerAccessAcquisitionV1> {
        let owner = self.context(owner_context)?.clone();
        let peer = self.context(peer_context)?.clone();
        if !owner.can_access_peer(&peer).map_err(driver_status)? {
            return Err(BundleStatusV1::Unsupported);
        }
        // The identity comes first: recording the edge after enabling it must
        // not be the thing that fails, because undoing an enable is itself a
        // driver call that can fail and would leave an edge nothing names.
        let identity = self.reserve_identity()?;
        // The driver reports whether this call enabled the edge or found it
        // already enabled, and that answer is the ownership this host reports
        // upward rather than discards.
        let ownership = owner.enable_peer_access(&peer).map_err(driver_status)?;
        self.insert_reserved(identity, CudaObjectV1::PeerAccess { owner, peer });
        Ok(PeerAccessAcquisitionV1 {
            object: DeviceObjectV1::new(identity),
            host_acquired: ownership == PeerAccessOwnership::Acquired,
        })
    }

    fn release_peer_access(
        &mut self,
        peer_access: DeviceObjectV1,
        disable: bool,
    ) -> HostDeviceResultV1<()> {
        self.release(peer_access, |object| match object {
            CudaObjectV1::PeerAccess { owner, peer } => {
                if disable {
                    owner.disable_peer_access(peer).map_err(driver_status)
                } else {
                    Ok(())
                }
            }
            _ => Err(BundleStatusV1::InvalidArgument),
        })
    }

    fn create_stream(
        &mut self,
        context: DeviceObjectV1,
        _stream_kind: u32,
    ) -> HostDeviceResultV1<DeviceObjectV1> {
        // The driver has one stream kind here. The host's stream-kind word
        // separates a bundle's stages from each other, not driver behaviour.
        let stream = self.context(context)?.stream().map_err(driver_status)?;
        self.insert(CudaObjectV1::Stream(stream))
    }

    fn synchronize_stream(&mut self, stream: DeviceObjectV1) -> HostDeviceResultV1<()> {
        self.stream(stream)?.synchronize().map_err(driver_status)
    }

    fn release_stream(&mut self, stream: DeviceObjectV1) -> HostDeviceResultV1<()> {
        // Dropping the ring waits for its copies and lets go of the stream.
        self.upload_rings.remove(&stream.get());
        self.release(stream, |object| match object {
            CudaObjectV1::Stream(stream) => stream.release().map_err(driver_status),
            _ => Err(BundleStatusV1::InvalidArgument),
        })?;
        self.pinned_transfers.remove(&stream.get());
        Ok(())
    }

    fn begin_graph_capture(&mut self, stream: DeviceObjectV1) -> HostDeviceResultV1<()> {
        self.stream(stream)?.begin_capture().map_err(driver_status)
    }

    fn end_graph_capture(
        &mut self,
        stream: DeviceObjectV1,
    ) -> Result<DeviceObjectV1, GraphCaptureErrorV1> {
        // cuStreamEndCapture ends the capture whatever it returns, an
        // invalidated capture included, on the thread that began it. Only a
        // failure before that call leaves the stream capturing.
        let graph = self
            .stream(stream)
            .map_err(GraphCaptureErrorV1::Capturing)?
            .end_capture()
            .map_err(|error| {
                let ended = matches!(
                    error,
                    CudaError::Driver {
                        operation: "cuStreamEndCapture",
                        ..
                    }
                );
                let status = driver_status(error);
                if ended {
                    GraphCaptureErrorV1::Ended(status)
                } else {
                    GraphCaptureErrorV1::Capturing(status)
                }
            })?;
        let executable = graph
            .instantiate()
            .map_err(|error| GraphCaptureErrorV1::Ended(driver_status(error)))?;
        self.insert(CudaObjectV1::Graph(executable))
            .map_err(GraphCaptureErrorV1::Ended)
    }

    fn launch_graph(
        &mut self,
        stream: DeviceObjectV1,
        graph: DeviceObjectV1,
    ) -> HostDeviceResultV1<()> {
        let stream = self.stream(stream)?;
        match self.get(graph)? {
            CudaObjectV1::Graph(graph) => graph.launch(stream).map_err(driver_status),
            _ => Err(BundleStatusV1::InvalidArgument),
        }
    }

    fn release_graph(&mut self, graph: DeviceObjectV1) -> HostDeviceResultV1<()> {
        self.release(graph, |object| match object {
            CudaObjectV1::Graph(graph) => graph.release().map_err(driver_status),
            _ => Err(BundleStatusV1::InvalidArgument),
        })
    }

    fn create_event(&mut self, context: DeviceObjectV1) -> HostDeviceResultV1<DeviceObjectV1> {
        let event = self.context(context)?.event().map_err(driver_status)?;
        self.insert(CudaObjectV1::Event(event))
    }

    fn record_event(
        &mut self,
        event: DeviceObjectV1,
        stream: DeviceObjectV1,
    ) -> HostDeviceResultV1<()> {
        let stream = self.stream(stream)?;
        self.event(event)?.record(stream).map_err(driver_status)
    }

    fn wait_event(
        &mut self,
        stream: DeviceObjectV1,
        event: DeviceObjectV1,
    ) -> HostDeviceResultV1<()> {
        let event = self.event(event)?;
        self.stream(stream)?
            .wait_event(event)
            .map_err(driver_status)
    }

    fn synchronize_event(&mut self, event: DeviceObjectV1) -> HostDeviceResultV1<()> {
        self.event(event)?.synchronize().map_err(driver_status)
    }

    fn release_event(&mut self, event: DeviceObjectV1) -> HostDeviceResultV1<()> {
        self.release(event, |object| match object {
            CudaObjectV1::Event(event) => event.release().map_err(driver_status),
            _ => Err(BundleStatusV1::InvalidArgument),
        })
    }

    fn copy_host_to_device(
        &mut self,
        stream: DeviceObjectV1,
        destination: DeviceRangeV1,
        source: &[u8],
    ) -> HostDeviceResultV1<()> {
        let (buffer, offset, length) = self.range(destination)?;
        if length != source.len() {
            return Err(BundleStatusV1::InvalidArgument);
        }
        if source.is_empty() {
            return Ok(());
        }
        let buffer = buffer.clone();
        let stream_handle = self.stream(stream)?.clone();
        let pinned = self.pinned_transfer_for(stream, &stream_handle, source.len())?;
        for (chunk_index, chunk) in source.chunks(pinned.len()).enumerate() {
            pinned.as_mut_slice()[..chunk.len()].copy_from_slice(chunk);
            buffer
                .copy_from_pinned_at_async(
                    offset + chunk_index * pinned.len(),
                    pinned,
                    0,
                    chunk.len(),
                    &stream_handle,
                )
                .map_err(driver_status)?
                .finish();
        }
        Ok(())
    }

    fn copy_host_to_device_ordered(
        &mut self,
        stream: DeviceObjectV1,
        destination: DeviceRangeV1,
        source: &[u8],
    ) -> HostDeviceResultV1<()> {
        let (buffer, offset, length) = self.range(destination)?;
        if length != source.len() {
            return Err(BundleStatusV1::InvalidArgument);
        }
        if source.is_empty() {
            return Ok(());
        }
        let buffer = buffer.clone();
        let stream_handle = self.stream(stream)?.clone();
        let ring = match self.upload_rings.entry(stream.get()) {
            std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(PinnedUploadRing::new(
                    PinnedHostBuffer::new(&stream_handle.context(), PINNED_UPLOAD_RING_BYTES)
                        .map_err(driver_status)?,
                ))
            }
        };
        ring.upload(&buffer, offset, source, &stream_handle)
            .map_err(driver_status)
    }

    fn copy_device_to_host(
        &mut self,
        stream: DeviceObjectV1,
        destination: &mut [u8],
        source: DeviceRangeV1,
    ) -> HostDeviceResultV1<()> {
        let (buffer, offset, length) = self.range(source)?;
        if length != destination.len() {
            return Err(BundleStatusV1::InvalidArgument);
        }
        if destination.is_empty() {
            return Ok(());
        }
        let buffer = buffer.clone();
        let stream_handle = self.stream(stream)?.clone();
        let pinned = self.pinned_transfer_for(stream, &stream_handle, destination.len())?;
        for (chunk_index, chunk) in destination.chunks_mut(pinned.len()).enumerate() {
            buffer
                .copy_to_pinned_at_async(
                    offset + chunk_index * pinned.len(),
                    pinned,
                    0,
                    chunk.len(),
                    &stream_handle,
                )
                .map_err(driver_status)?
                .finish();
            chunk.copy_from_slice(&pinned.as_slice()[..chunk.len()]);
        }
        Ok(())
    }

    fn copy_device_to_device(
        &mut self,
        stream: DeviceObjectV1,
        destination: DeviceRangeV1,
        source: DeviceRangeV1,
    ) -> HostDeviceResultV1<()> {
        let (_, destination_offset, destination_length) = self.range(destination)?;
        let (source_buffer, source_offset, source_length) = self.range(source)?;
        if destination_length != source_length {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let destination_buffer = self.buffer(destination.object)?;
        let stream = self.stream(stream)?;
        destination_buffer
            .copy_range_from_async(
                destination_offset,
                source_buffer,
                source_offset,
                source_length,
                stream,
            )
            .map_err(driver_status)
    }

    fn copy_peer_to_peer(
        &mut self,
        stream: DeviceObjectV1,
        destination_context: DeviceObjectV1,
        destination: DeviceRangeV1,
        source_context: DeviceObjectV1,
        source: DeviceRangeV1,
    ) -> HostDeviceResultV1<()> {
        // Both contexts are checked so a peer copy names the pair it crosses,
        // and peer access must already be enabled between them.
        self.context(destination_context)?;
        self.context(source_context)?;
        self.copy_device_to_device(stream, destination, source)
    }

    fn zero_device_buffer(
        &mut self,
        stream: DeviceObjectV1,
        destination: DeviceRangeV1,
    ) -> HostDeviceResultV1<()> {
        let (buffer, offset, length) = self.range(destination)?;
        let stream = self.stream(stream)?;
        buffer
            .zero_at_async(offset, length, stream)
            .map_err(driver_status)
    }

    /// Writes the driver's own address for each slot, in slot order.
    ///
    /// Only the driver knows where an allocation landed, which is the whole
    /// reason this service exists: a bundle holds spans and could never build
    /// this table itself. The addresses are staged through host memory and the
    /// transfer lands before this returns, so anything ordered on the stream
    /// afterwards reads a complete table rather than a partly written one.
    fn write_device_pointer_table(
        &mut self,
        stream: DeviceObjectV1,
        destination: DeviceRangeV1,
        slots: &[Option<DeviceRangeV1>],
    ) -> HostDeviceResultV1<()> {
        let expected = slots
            .len()
            .checked_mul(DEVICE_POINTER_TABLE_ENTRY_BYTES_V1 as usize)
            .ok_or(BundleStatusV1::InvalidArgument)?;
        let (_, _, length) = self.range(destination)?;
        if slots.is_empty() || length != expected {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let mut table = Vec::with_capacity(expected);
        for slot in slots {
            let address = match slot {
                Some(range) => {
                    let (buffer, offset, byte_len) = self.range(*range)?;
                    buffer
                        .device_address_at(offset, byte_len)
                        .map_err(driver_status)?
                }
                None => 0,
            };
            table.extend_from_slice(&address.to_le_bytes());
        }
        let (buffer, offset, _) = self.range(destination)?;
        buffer
            .copy_from_bytes_at(offset, &table, self.stream(stream)?)
            .map_err(driver_status)
    }

    fn load_module(
        &mut self,
        context: DeviceObjectV1,
        module_bytes: &[u8],
    ) -> HostDeviceResultV1<DeviceObjectV1> {
        if module_bytes.is_empty() {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let module = self
            .context(context)?
            .load_module(module_bytes)
            .map_err(driver_status)?;
        self.insert(CudaObjectV1::Module(module))
    }

    fn resolve_symbol(
        &mut self,
        module: DeviceObjectV1,
        symbol_name: &[u8],
    ) -> HostDeviceResultV1<DeviceObjectV1> {
        let name = std::str::from_utf8(symbol_name).map_err(|_| BundleStatusV1::InvalidArgument)?;
        if name.is_empty() {
            return Err(BundleStatusV1::InvalidArgument);
        }
        // Resolving now proves the symbol exists before a handle is issued.
        let function = self.module(module)?.function(name).map_err(driver_status)?;
        self.insert(CudaObjectV1::Symbol { module, function })
    }

    fn configure_symbol(
        &mut self,
        symbol: DeviceObjectV1,
        dynamic_shared_memory_bytes: u32,
    ) -> HostDeviceResultV1<()> {
        let function = self.function(symbol)?;
        function
            .set_max_dynamic_shared_memory(dynamic_shared_memory_bytes)
            .map_err(driver_status)
    }

    fn cluster_residency(
        &mut self,
        symbol: DeviceObjectV1,
        config: LaunchConfigV1,
    ) -> HostDeviceResultV1<runtime_bundle_abi::BundleRawClusterResidencyV1> {
        let function = self.function(symbol)?;
        let context = function.context();
        let mut residency = runtime_bundle_abi::BundleRawClusterResidencyV1::unreported();
        residency.multiprocessor_count =
            u32::try_from(context.device().multiprocessor_count()).unwrap_or(0);
        // A driver that cannot state the context's SM affinity leaves it
        // unreported rather than failing the query.
        residency.usable_multiprocessors = context.usable_multiprocessors().unwrap_or(0);
        // The descriptor the caller launches: its grid, block, shared memory
        // and cluster, and cooperative when requested. An unclustered shape
        // is asked about as clusters of one CTA, so the count is its CTAs
        // the device keeps resident at once.
        let cluster = launch_cluster_v1(config)?.unwrap_or(LaunchClusterV1::NONE);
        let shape = runtime_cuda::LaunchConfig {
            grid: (config.grid_x, config.grid_y, config.grid_z),
            block: (config.block_x, config.block_y, config.block_z),
            shared_memory_bytes: config.shared_memory_bytes,
            cluster: (cluster.x, cluster.y, cluster.z),
            programmatic_stream_serialization: false,
        };
        let cooperative = config.flags & runtime_bundle_abi::BUNDLE_LAUNCH_FLAG_V1_COOPERATIVE != 0;
        residency.max_active_clusters = function
            .max_active_clusters(shape, cooperative)
            .map_err(driver_status)?;
        residency.flags |= runtime_bundle_abi::BUNDLE_CLUSTER_RESIDENCY_V1_CLUSTERS_REPORTED;
        if mps_client() {
            residency.flags |= runtime_bundle_abi::BUNDLE_CLUSTER_RESIDENCY_V1_MPS_CLIENT;
        }
        if context.device().cooperative_launch() {
            residency.flags |= runtime_bundle_abi::BUNDLE_CLUSTER_RESIDENCY_V1_COOPERATIVE_LAUNCH;
        }
        Ok(residency)
    }

    fn launch_symbol(
        &mut self,
        stream: DeviceObjectV1,
        symbol: DeviceObjectV1,
        config: LaunchConfigV1,
        arguments: &[DeviceArgumentV1],
    ) -> HostDeviceResultV1<()> {
        // What the flags and the cluster ask for is decided before anything is
        // resolved, so a launch this backend cannot dispatch touches no driver
        // object. A cluster no launch could take is refused here; whether this
        // device forms the one it can take is the driver crate's answer, and it
        // refuses rather than launching the blocks without it.
        let cluster = launch_cluster_v1(config)?;
        let cooperative = launch_cooperative_v1(config)?;
        let config = launch_config(config, cluster)?;
        // The whole launch is checked before any of it is built. Encoding a
        // tensor map is a driver call that changes nothing but cannot be taken
        // back, so a launch that will be refused has to be refused before the
        // first encode rather than after it.
        let launch_context = self.launch_context(symbol)?;
        self.preflight_launch(&launch_context, stream, arguments)?;
        // Every tensor map is encoded before the argument list is built, and
        // the encodings are held here for the whole call. A tensor-map argument
        // reaches a kernel as the descriptor's own bytes, so the storage those
        // bytes live in has to outlive the launch that reads them.
        let mut maps = Vec::new();
        for argument in arguments {
            if let DeviceArgumentV1::TensorMap(map) = argument {
                maps.push(self.cached_tensor_map(&launch_context, map)?);
            }
        }
        // A range argument reaches a kernel as the address of its own start, so
        // the offset is applied before the pointer is taken.
        // A by-value address array reaches a kernel as the addresses' own
        // bytes, so the storage holding them outlives the launch that reads it,
        // exactly as a tensor-map encoding does.
        let mut pointer_arrays_storage = Vec::new();
        for argument in arguments {
            if let DeviceArgumentV1::DevicePointerArray(slots) = argument {
                let mut addresses = Vec::with_capacity(slots.len());
                for slot in slots {
                    addresses.push(match slot {
                        Some(range) => {
                            let (buffer, offset, byte_len) = self.range(*range)?;
                            buffer
                                .device_address_at(offset, byte_len)
                                .map_err(driver_status)?
                        }
                        None => 0,
                    });
                }
                pointer_arrays_storage.push(addresses);
            }
        }
        let mut values = Vec::with_capacity(arguments.len());
        let mut encoded = maps.iter();
        let mut pointer_arrays = pointer_arrays_storage.iter();
        for argument in arguments {
            match argument {
                DeviceArgumentV1::Range(range) => {
                    let (buffer, offset, byte_len) = self.range(*range)?;
                    values.push(KernelArgument::Range {
                        buffer,
                        offset,
                        byte_len,
                    });
                }
                DeviceArgumentV1::NullDevicePointer => values.push(KernelArgument::NullBuffer),
                DeviceArgumentV1::ScalarU32(value) => values.push(KernelArgument::U32(*value)),
                DeviceArgumentV1::ScalarI32(value) => values.push(KernelArgument::I32(*value)),
                DeviceArgumentV1::ScalarU64(value) => values.push(KernelArgument::U64(*value)),
                DeviceArgumentV1::DevicePointerArray(_) => {
                    let addresses = pointer_arrays
                        .next()
                        .expect("one address array was built for each pointer-array argument");
                    values.push(KernelArgument::DevicePointerArray(addresses));
                }
                // Opaque bytes have no declared launch encoding in V1.
                DeviceArgumentV1::OpaqueBytes(_) => {
                    return Err(BundleStatusV1::Unsupported);
                }
                DeviceArgumentV1::TensorMap(_) => {
                    let map = encoded
                        .next()
                        .expect("one encoding was made for each tensor-map argument");
                    values.push(KernelArgument::TensorMap(map.as_ref()));
                }
            }
        }
        let stream = self.stream(stream)?;
        let function = self.function(symbol)?;
        if cooperative {
            function.launch_cooperative(stream, config, &values)
        } else {
            function.launch(stream, config, &values)
        }
        .map_err(driver_status)?;
        Ok(())
    }

    fn release_symbol(&mut self, symbol: DeviceObjectV1) -> HostDeviceResultV1<()> {
        // A function handle belongs to its module, so only the record goes.
        self.release(symbol, |object| match object {
            CudaObjectV1::Symbol { .. } => Ok(()),
            _ => Err(BundleStatusV1::InvalidArgument),
        })
    }

    fn release_module(&mut self, module: DeviceObjectV1) -> HostDeviceResultV1<()> {
        self.release(module, |object| match object {
            CudaObjectV1::Module(_) => Ok(()),
            _ => Err(BundleStatusV1::InvalidArgument),
        })
    }

    fn execute_collective(
        &mut self,
        stream: DeviceObjectV1,
        binding: &CollectiveBindingV1,
        operation: &[u8],
        arguments: &[DeviceArgumentV1],
    ) -> HostDeviceResultV1<()> {
        // No collective library is bound to this host. The contracts the shared
        // surface declares have a correct generic fallback, and every other
        // contract is unsupported rather than quietly doing nothing. The whole
        // request is decided before the stream is waited on, so a request this
        // host will not perform costs the caller no synchronization.
        let plan = plan_generic_collective_v1(binding, operation, arguments)?;
        // The rank buffers are read through their own streams, so whatever the
        // bundle ordered on this one has to land first or the reduction sums
        // stale bytes. Each staged write is synchronized in turn, so the result
        // is already visible to whatever the bundle orders on this stream next.
        self.stream(stream)?.synchronize().map_err(driver_status)?;
        self.sum_through_host_v1(&plan)?;
        Ok(())
    }

    /// Lends device addresses and this host's own stream to one operation.
    ///
    /// The whole call is settled before any of it is lent. The stream and every
    /// operand have to belong to the context the operation names, because an
    /// address from another context names memory it cannot reach, and each
    /// range is resolved against the live allocation it came from. Only then is
    /// the context made current and the callback invoked, once.
    ///
    /// Waiting is not this backend's decision. The services layer waits on the
    /// stream before every callback and, for an operation that may enqueue,
    /// again afterwards, so the ordering is one rule in one place rather than
    /// one each backend keeps its own version of.
    fn run_custom_operation(
        &mut self,
        context: DeviceObjectV1,
        stream: DeviceObjectV1,
        operation_id: &[u8],
        operands: &[DeviceCustomOperandV1],
        operation: &CustomOperationV1<'_>,
    ) -> HostDeviceResultV1<i32> {
        if operation_id.is_empty() {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let launch_context = self.context(context)?.clone();
        let stream = self.stream(stream)?.clone();
        if stream.context() != launch_context {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let mut ranges = Vec::with_capacity(operands.len());
        for operand in operands {
            let (buffer, offset, byte_len) = self.range(operand.range)?;
            if buffer.owner_context() != launch_context {
                return Err(BundleStatusV1::InvalidArgument);
            }
            let address = buffer
                .device_address_at(offset, byte_len)
                .map_err(driver_status)?;
            let length = u64::try_from(byte_len).map_err(|_| BundleStatusV1::InvalidArgument)?;
            ranges.push(LentRangeV1::new(address, length));
        }
        launch_context.set_current().map_err(driver_status)?;
        stream
            .with_raw_handle(|raw| {
                operation.invoke(LentLaunchV1 {
                    address_space: CustomAddressSpaceV1::Device,
                    stream: LentStreamV1::from_raw(raw),
                    ranges: &ranges,
                })
            })
            .map_err(driver_status)?
    }

    fn log(&mut self, message: &[u8]) -> HostDeviceResultV1<()> {
        let _phase = runtime_engine::profile::PhaseTimer::traced(
            runtime_engine::profile::StepPhase::BundleLog,
        );
        record_diagnostic_v1(&mut self.logs, message);
        // The log ABI has no severity. Only recognized routine trace schemas
        // may be lost; an unclassified message can be the sole failure detail.
        if runtime_engine::logging::is_routine_bundle_trace(message) {
            runtime_engine::diagnostic!("bundle: {}", String::from_utf8_lossy(message));
        } else {
            eprintln!("bundle: {}", String::from_utf8_lossy(message));
        }
        Ok(())
    }
}

impl crate::probe::HostDeviceMemoryUsageV1 for CudaHostDeviceV1 {
    fn current_device_memory_bytes(&self) -> u64 {
        self.ledger.current_bytes()
    }

    fn peak_device_memory_bytes(&self) -> u64 {
        self.ledger.peak_bytes()
    }

    fn materialized_device_memory_bytes(&self) -> u64 {
        self.ledger.current_bytes()
    }
}

/// Measures each rank in its own context, and refuses where it cannot.
///
/// The free and total counts are the driver's own answer for that rank's
/// device, taken one rank at a time. A driver that will not answer is an error
/// here rather than a zero, because a zero would read as a full board.
///
/// What the numbers mean depends entirely on when this is called. It reports
/// the device as it stands at that instant, so a caller that measures after a
/// model is resident reads free memory with the model already in it. That is
/// not the memory that was available before the load, and it is not a budget:
/// deriving one needs a measurement taken before anything was allocated, which
/// this call cannot reconstruct afterwards. `allocated_bytes` and
/// `peak_allocated_bytes` are this backend's own allocations for that rank,
/// which is a different quantity again from the board's free memory, because
/// another process on the same device moves one and not the other.
///
/// Every driver operation in `runtime-cuda` makes its own context current
/// before it runs, including the launch path and this measurement, so
/// measuring each rank in turn leaves no context expectation for a later
/// operation to depend on.
impl crate::probe::HostDeviceRankMemoryUsageV1 for CudaHostDeviceV1 {
    fn rank_memory_v1(&self) -> Result<crate::probe::HostDeviceRankMemoryReportV1, String> {
        // The ledger decides whether these counts can be reported at all: it
        // refuses an unattributed byte, two contexts on one rank, and per-rank
        // counts that do not add up to what this backend allocated.
        let accounted = self.ledger.ranks()?;
        let mut measured = Vec::with_capacity(accounted.len());
        for record in &accounted {
            let context = self
                .rank_contexts
                .get(&record.context)
                .ok_or_else(|| format!("rank {} holds no context to measure", record.rank))?;
            let (free_bytes, total_bytes) = context.memory_info().map_err(|error| {
                format!(
                    "rank {} device memory could not be measured: {error}",
                    record.rank
                )
            })?;
            measured.push(crate::probe::HostDeviceRankMemoryV1::new(
                record.rank,
                record.device_ordinal,
                Some(free_bytes),
                Some(total_bytes),
                record.allocated_bytes,
                record.peak_allocated_bytes,
            ));
        }
        Ok(crate::probe::HostDeviceRankMemoryReportV1::new(
            crate::probe::HostDeviceMemoryMeasurementV1::DriverMeasured,
            measured,
        ))
    }
}

#[cfg(all(test, feature = "cuda-fake-driver-tests"))]
mod pinned_upload_tests {
    use super::*;

    #[test]
    fn file_rank_upload_uses_two_large_pinned_buffers_and_preserves_offsets() {
        runtime_cuda::fake_driver::reset();
        let mut device = CudaHostDeviceV1::new().unwrap();
        let context = device.create_context(0, 0).unwrap();
        let pool = device.create_memory_pool(context).unwrap();
        let stream = device.create_stream(context, 0).unwrap();
        let bytes = 2 * PINNED_RANK_UPLOAD_CHUNK_BYTES as u64 + 257;
        let destination = device
            .allocate_device_buffer(pool, Some(stream), bytes, 1)
            .unwrap();
        let mut reads = 0;
        device
            .upload_pinned_rank_chunks(stream, destination, 7, bytes, &mut |offset, target| {
                reads += 1;
                for (index, byte) in target.iter_mut().enumerate() {
                    *byte = ((offset + index as u64) % 251) as u8;
                }
                true
            })
            .unwrap();
        assert_eq!(reads, 3);
        assert_eq!(runtime_cuda::fake_driver::call_count(16), 2);
        assert_eq!(runtime_cuda::fake_driver::call_count(8), 3);
        let mut first = [0_u8; 4];
        device
            .copy_device_to_host(
                stream,
                &mut first,
                DeviceRangeV1 {
                    object: destination,
                    offset: 0,
                    length: 4,
                },
            )
            .unwrap();
        assert_eq!(first, [7, 8, 9, 10]);
        let mut last = [0_u8; 4];
        device
            .copy_device_to_host(
                stream,
                &mut last,
                DeviceRangeV1 {
                    object: destination,
                    offset: bytes - 4,
                    length: 4,
                },
            )
            .unwrap();
        assert_eq!(
            last,
            [
                ((7 + bytes - 4) % 251) as u8,
                ((7 + bytes - 3) % 251) as u8,
                ((7 + bytes - 2) % 251) as u8,
                ((7 + bytes - 1) % 251) as u8
            ]
        );
        let input = vec![19_u8; PINNED_SERVICE_TRANSFER_CHUNK_BYTES + 17];
        let range = DeviceRangeV1 {
            object: destination,
            offset: 12,
            length: input.len() as u64,
        };
        device.copy_host_to_device(stream, range, &input).unwrap();
        let allocations = runtime_cuda::fake_driver::call_count(16);
        let mut output = vec![0_u8; input.len()];
        device
            .copy_device_to_host(stream, &mut output, range)
            .unwrap();
        assert_eq!(output, input);
        device.copy_host_to_device(stream, range, &input).unwrap();
        assert_eq!(runtime_cuda::fake_driver::call_count(16), allocations);
    }

    /// A failed read in any rank stops the wave before its copies are
    /// enqueued, so no unverified bytes reach a device.
    #[test]
    fn a_failed_rank_read_enqueues_nothing_from_its_wave() {
        for failing_call in [1, 2] {
            runtime_cuda::fake_driver::reset();
            let mut device = CudaHostDeviceV1::new().unwrap();
            let context = device.create_context(0, 0).unwrap();
            let pool = device.create_memory_pool(context).unwrap();
            let streams = [
                device.create_stream(context, 0).unwrap(),
                device.create_stream(context, 0).unwrap(),
            ];
            let bytes = 2 * PINNED_RANK_UPLOAD_CHUNK_BYTES as u64 + 257;
            let destinations = [
                device
                    .allocate_device_buffer(pool, Some(streams[0]), bytes, 1)
                    .unwrap(),
                device
                    .allocate_device_buffer(pool, Some(streams[1]), bytes, 1)
                    .unwrap(),
            ];
            let mut good = |_offset, target: &mut [u8]| {
                target.fill(1);
                true
            };
            let mut calls = 0;
            let mut failing = |_offset, target: &mut [u8]| {
                calls += 1;
                target.fill(2);
                calls != failing_call
            };
            let copies_before = runtime_cuda::fake_driver::call_count(8);
            let mut uploads = [
                PinnedFileUploadV1 {
                    stream: streams[0],
                    destination: destinations[0],
                    source_offset: 0,
                    byte_length: bytes,
                    read_chunk: &mut good,
                },
                PinnedFileUploadV1 {
                    stream: streams[1],
                    destination: destinations[1],
                    source_offset: 0,
                    byte_length: bytes,
                    read_chunk: &mut failing,
                },
            ];
            let result = device.upload_pinned_file_ranges(&mut uploads).unwrap();
            assert_eq!(result, Err(BundleStatusV1::HostError));
            // Only the waves before the failing one were copied.
            assert_eq!(
                runtime_cuda::fake_driver::call_count(8) - copies_before,
                2 * (failing_call - 1),
                "failing call {failing_call}"
            );
        }
    }
}

#[cfg(all(test, feature = "cuda-fake-driver-tests"))]
mod allocation_ordering_tests {
    use super::*;

    #[test]
    fn ordinal_context_recreation_keeps_pool_and_stream_compatible() {
        runtime_cuda::fake_driver::reset();
        let mut device = CudaHostDeviceV1::new().unwrap();
        let initial = device.create_context(0, 0).unwrap();
        let pool = device.create_memory_pool(initial).unwrap();
        let recreated = device.create_context(0, 0).unwrap();
        let stream = device.create_stream(recreated, 0).unwrap();
        let buffer = device
            .allocate_device_buffer(pool, Some(stream), 16, 1)
            .unwrap();
        let range = DeviceRangeV1 {
            object: buffer,
            offset: 0,
            length: 16,
        };
        device.copy_host_to_device(stream, range, &[7; 16]).unwrap();
        let mut bytes = [0; 16];
        device
            .copy_device_to_host(stream, &mut bytes, range)
            .unwrap();
        assert_eq!(bytes, [7; 16]);
        assert_eq!(runtime_cuda::fake_driver::call_count(22), 1);
        device.release_device_buffer(buffer).unwrap();
        device.release_stream(stream).unwrap();
        device.release_memory_pool(pool).unwrap();
        device.release_context(recreated).unwrap();
        device.release_context(initial).unwrap();
        assert_eq!(device.live_object_count(), 0);
    }

    /// A bundle may write a fresh allocation from any stream, so the
    /// allocation has executed before this service hands it out.
    #[test]
    fn an_allocation_is_ordered_before_a_write_from_another_stream() {
        runtime_cuda::fake_driver::reset();
        let mut device = CudaHostDeviceV1::new().unwrap();
        let context = device.create_context(0, 0).unwrap();
        let pool = device.create_memory_pool(context).unwrap();
        let owner = device.create_stream(context, 0).unwrap();
        let writer = device.create_stream(context, 0).unwrap();
        let buffer = device
            .allocate_device_buffer(pool, Some(owner), 64, 1)
            .unwrap();
        let range = DeviceRangeV1 {
            object: buffer,
            offset: 8,
            length: 16,
        };
        device.copy_host_to_device(writer, range, &[5; 16]).unwrap();
        device
            .copy_host_to_device_ordered(writer, range, &[6; 16])
            .unwrap();
        device.synchronize_stream(writer).unwrap();
        let mut output = [0_u8; 16];
        device
            .copy_device_to_host(owner, &mut output, range)
            .unwrap();
        assert_eq!(output, [6; 16]);
        assert_eq!(runtime_cuda::fake_driver::ordering_violations(), 0);
    }

    /// A write-back whose fence fails reports the driver's error to the
    /// service instead of stopping the process.
    #[test]
    fn a_failed_write_back_fence_returns_an_error() {
        const STREAM_SYNCHRONIZE: i32 = 5;
        runtime_cuda::fake_driver::reset();
        let mut device = CudaHostDeviceV1::new().unwrap();
        let context = device.create_context(0, 0).unwrap();
        let pool = device.create_memory_pool(context).unwrap();
        let stream = device.create_stream(context, 0).unwrap();
        let buffer = device
            .allocate_device_buffer(pool, Some(stream), 32, 1)
            .unwrap();
        let range = DeviceRangeV1 {
            object: buffer,
            offset: 0,
            length: 8,
        };
        runtime_cuda::fake_driver::fail_next(STREAM_SYNCHRONIZE, 719);
        assert!(device.write_range_through_host_v1(range, &[9; 8]).is_err());
        device.write_range_through_host_v1(range, &[7; 8]).unwrap();
    }

    /// The collective fallback writes each rank's total back in stream order.
    #[test]
    fn a_range_written_through_the_host_lands_in_stream_order() {
        runtime_cuda::fake_driver::reset();
        let mut device = CudaHostDeviceV1::new().unwrap();
        let context = device.create_context(0, 0).unwrap();
        let pool = device.create_memory_pool(context).unwrap();
        let stream = device.create_stream(context, 0).unwrap();
        let buffer = device
            .allocate_device_buffer(pool, Some(stream), 32, 1)
            .unwrap();
        let range = DeviceRangeV1 {
            object: buffer,
            offset: 4,
            length: 8,
        };
        device.write_range_through_host_v1(range, &[9; 8]).unwrap();
        let mut output = [0_u8; 8];
        device
            .read_range_through_host_v1(range, &mut output)
            .unwrap();
        assert_eq!(output, [9; 8]);
        assert_eq!(runtime_cuda::fake_driver::ordering_violations(), 0);
    }
}

#[cfg(all(test, feature = "cuda-fake-driver-tests"))]
mod sampler_reservation_tests {
    use super::*;

    #[test]
    fn startup_sampler_owns_every_buffer_and_reuses_the_session() {
        runtime_cuda::fake_driver::reset();
        let mut device = CudaHostDeviceV1::new().unwrap();
        let context = device.create_context(0, 0).unwrap();
        device.reserve_sampling_workspace(context, 33).unwrap();
        let sampler = device.samplers.get(&context.get()).unwrap();
        let address = sampler as *const DeviceSamplingSession;
        device.reserve_sampling_workspace(context, 33).unwrap();
        assert_eq!(
            device.samplers.get(&context.get()).unwrap() as *const _,
            address
        );
        assert_eq!(
            device.reserve_sampling_workspace(context, 34),
            Err(BundleStatusV1::InvalidArgument)
        );
        device.release_context(context).unwrap();
        assert!(device.samplers.is_empty());
    }
}

#[cfg(all(test, feature = "cuda-fake-driver-tests"))]
mod graph_capture_tests {
    use super::*;

    const CONTEXT_SET_CURRENT: i32 = 1;
    const STREAM_END_CAPTURE: i32 = 19;
    const GRAPH_INSTANTIATE: i32 = 20;
    const GRAPH_EXEC_DESTROY: i32 = 33;
    const STREAM_CAPTURE_INVALIDATED: i32 = 901;

    /// A failure of cuStreamEndCapture or of instantiation ends the capture;
    /// one before the end call leaves the stream capturing.
    #[test]
    fn a_failed_capture_end_reports_whether_the_stream_still_captures() {
        runtime_cuda::fake_driver::reset();
        let mut device = CudaHostDeviceV1::new().unwrap();
        let context = device.create_context(0, 0).unwrap();
        let stream = device.create_stream(context, 0).unwrap();
        for (operation, code, expected) in [
            (
                STREAM_END_CAPTURE,
                STREAM_CAPTURE_INVALIDATED,
                GraphCaptureErrorV1::Ended(BundleStatusV1::HostError),
            ),
            (
                GRAPH_INSTANTIATE,
                2,
                GraphCaptureErrorV1::Ended(BundleStatusV1::HostError),
            ),
            (
                CONTEXT_SET_CURRENT,
                201,
                GraphCaptureErrorV1::Capturing(BundleStatusV1::HostError),
            ),
        ] {
            device.begin_graph_capture(stream).unwrap();
            let ends = runtime_cuda::fake_driver::call_count(STREAM_END_CAPTURE);
            runtime_cuda::fake_driver::fail_next(operation, code);
            assert_eq!(device.end_graph_capture(stream), Err(expected));
            let ended = matches!(expected, GraphCaptureErrorV1::Ended(_));
            assert_eq!(
                runtime_cuda::fake_driver::call_count(STREAM_END_CAPTURE),
                ends + usize::from(ended)
            );
        }
        let graph = device.end_graph_capture(stream).unwrap();
        device.release_graph(graph).unwrap();
        device.release_stream(stream).unwrap();
    }

    /// A graph whose destruction fails stays owned and launchable, and a
    /// later release destroys it; a released graph is unknown afterwards.
    #[test]
    fn a_failed_graph_release_keeps_ownership_for_a_retry() {
        runtime_cuda::fake_driver::reset();
        let mut device = CudaHostDeviceV1::new().unwrap();
        let context = device.create_context(0, 0).unwrap();
        let stream = device.create_stream(context, 0).unwrap();
        device.begin_graph_capture(stream).unwrap();
        let graph = device.end_graph_capture(stream).unwrap();
        runtime_cuda::fake_driver::fail_next(GRAPH_EXEC_DESTROY, 719);
        assert_eq!(device.release_graph(graph), Err(BundleStatusV1::HostError));
        assert_eq!(runtime_cuda::fake_driver::call_count(GRAPH_EXEC_DESTROY), 1);
        device.launch_graph(stream, graph).unwrap();
        device.release_graph(graph).unwrap();
        assert_eq!(runtime_cuda::fake_driver::call_count(GRAPH_EXEC_DESTROY), 2);
        assert!(device.release_graph(graph).is_err());
        assert!(device.launch_graph(stream, graph).is_err());
        assert_eq!(runtime_cuda::fake_driver::call_count(GRAPH_EXEC_DESTROY), 2);
        device.release_stream(stream).unwrap();
    }
}

#[cfg(test)]
mod ordered_upload_tests {
    use super::*;

    #[test]
    fn ordered_uploads_land_in_order_and_release_their_destination() {
        let Ok(ordinal) = std::env::var("CHOPINFER_KERNEL_GPU") else {
            return;
        };
        let ordinal: i32 = ordinal.parse().expect("GPU ordinal is an integer");
        let mut device = CudaHostDeviceV1::new().unwrap();
        let context = device.create_context(0, ordinal).unwrap();
        let pool = device.create_memory_pool(context).unwrap();
        let stream = device.create_stream(context, 0).unwrap();
        let bytes = PINNED_UPLOAD_RING_BYTES as u64 + 4096;
        let destination = device
            .allocate_device_buffer(pool, Some(stream), bytes, 1)
            .unwrap();
        let range = |offset: u64, length: u64| DeviceRangeV1 {
            object: destination,
            offset,
            length,
        };
        // Larger than the ring, so it is staged in two pieces.
        let large = (0..bytes)
            .map(|index| (index % 253) as u8)
            .collect::<Vec<_>>();
        device
            .copy_host_to_device_ordered(stream, range(0, bytes), &large)
            .unwrap();
        // Many small copies, more than the ring keeps in flight, to one place;
        // the last one written must be the one read.
        for value in 0..200_u8 {
            device
                .copy_host_to_device_ordered(stream, range(8, 4), &[value; 4])
                .unwrap();
        }
        let mut read = vec![0_u8; bytes as usize];
        device
            .copy_device_to_host(stream, &mut read, range(0, bytes))
            .unwrap();
        assert_eq!(read[8..12], [199; 4]);
        assert_eq!(read[..8], large[..8]);
        assert_eq!(read[12..], large[12..]);
        device.release_device_buffer(destination).unwrap();
        device.release_stream(stream).unwrap();
    }
}

#[cfg(all(test, feature = "cuda-fake-driver-tests"))]
mod cooperative_launch_tests {
    use super::*;
    use crate::services::{HostInstanceConfigV1, HostServicesV1Backend, HostTopologyV1};
    use runtime_bundle_abi::HostServicesBackendV1;
    use runtime_cuda::fake_driver;

    /// A launch with the cooperative flag reaches the driver's cooperative
    /// launch (with its cluster when it has one); one that also asks for
    /// programmatic stream serialization is refused before the driver.
    #[test]
    fn cooperative_launches_reach_the_cooperative_driver_launch() {
        const LAUNCH_COOPERATIVE: i32 = 28;
        fake_driver::reset();
        let topology = HostTopologyV1::new(vec![0]).unwrap();
        let host = HostServicesV1Backend::new(
            CudaHostDeviceV1::new().unwrap(),
            HostInstanceConfigV1::new(topology),
        );
        let context = host.select_context(0, 0).unwrap();
        let pool = host.select_memory_pool(context).unwrap();
        let stream = host.select_stage_stream(context, 0, b"decode", 0).unwrap();
        let buffer = host.allocate_device_buffer(pool, stream, 64, 1).unwrap();
        let module = host.load_cuda_module(context, b"N").unwrap();
        let symbol = host.resolve_cuda_symbol(module, b"entry_v1").unwrap();
        let config = |flags, cluster_x| runtime_bundle_abi::LaunchConfigV1 {
            flags,
            grid_x: 8,
            grid_y: 1,
            grid_z: 1,
            block_x: 32,
            block_y: 1,
            block_z: 1,
            shared_memory_bytes: 0,
            cluster: runtime_bundle_abi::LaunchClusterV1 {
                x: cluster_x,
                y: 1,
                z: 1,
            },
        };
        let span = runtime_bundle_abi::DeviceBufferSpanV1::new(buffer, 0, 64);
        let arguments = [runtime_bundle_abi::KernelArgumentV1::DeviceBuffer(span)];
        let cooperative = runtime_bundle_abi::BUNDLE_LAUNCH_FLAG_V1_COOPERATIVE;
        for cluster_x in [1, 8] {
            host.launch_cuda_symbol(stream, symbol, config(cooperative, cluster_x), &arguments)
                .unwrap();
        }
        assert_eq!(fake_driver::call_count(LAUNCH_COOPERATIVE), 2);
        // The residency report retains the device's cooperative support.
        let residency = host.query_cluster_residency(symbol, config(0, 1)).unwrap();
        assert_ne!(
            residency.flags & runtime_bundle_abi::BUNDLE_CLUSTER_RESIDENCY_V1_COOPERATIVE_LAUNCH,
            0
        );
        let pdl = runtime_bundle_abi::BUNDLE_LAUNCH_FLAG_V1_PROGRAMMATIC_STREAM_SERIALIZATION;
        assert_eq!(
            host.launch_cuda_symbol(
                stream,
                symbol,
                config(cooperative | pdl, 1),
                &[runtime_bundle_abi::KernelArgumentV1::DeviceBuffer(span)]
            ),
            Err(BundleStatusV1::InvalidArgument)
        );
        assert_eq!(fake_driver::call_count(LAUNCH_COOPERATIVE), 2);
    }

    /// An unclustered launch is reported: the driver is asked about clusters
    /// of one CTA, so a spinning grid without a cluster (here 16 CTAs of 256
    /// threads with programmatic stream serialization) can be admitted. A
    /// clustered launch is still asked about its own cluster, and a failed
    /// query is an error rather than a report.
    #[test]
    fn unclustered_launches_report_their_resident_ctas() {
        const MAX_ACTIVE_CLUSTERS: i32 = 32;
        fake_driver::reset();
        let topology = HostTopologyV1::new(vec![0]).unwrap();
        let host = HostServicesV1Backend::new(
            CudaHostDeviceV1::new().unwrap(),
            HostInstanceConfigV1::new(topology),
        );
        let context = host.select_context(0, 0).unwrap();
        let module = host.load_cuda_module(context, b"N").unwrap();
        let symbol = host.resolve_cuda_symbol(module, b"entry_v1").unwrap();
        let config = |cluster_x| runtime_bundle_abi::LaunchConfigV1 {
            flags: runtime_bundle_abi::BUNDLE_LAUNCH_FLAG_V1_PROGRAMMATIC_STREAM_SERIALIZATION,
            grid_x: 16,
            grid_y: 1,
            grid_z: 1,
            block_x: 256,
            block_y: 1,
            block_z: 1,
            shared_memory_bytes: 0,
            cluster: runtime_bundle_abi::LaunchClusterV1 {
                x: cluster_x,
                y: 1,
                z: 1,
            },
        };
        let reported = runtime_bundle_abi::BUNDLE_CLUSTER_RESIDENCY_V1_CLUSTERS_REPORTED;
        // The fake driver keeps 352 CTAs resident, in clusters of the size
        // asked about.
        let unclustered = host.query_cluster_residency(symbol, config(1)).unwrap();
        assert_ne!(unclustered.flags & reported, 0);
        assert_eq!(unclustered.max_active_clusters, 352);
        assert_eq!(fake_driver::call_count(MAX_ACTIVE_CLUSTERS), 1);
        let clustered = host.query_cluster_residency(symbol, config(8)).unwrap();
        assert_ne!(clustered.flags & reported, 0);
        assert_eq!(clustered.max_active_clusters, 44);
        assert_eq!(fake_driver::call_count(MAX_ACTIVE_CLUSTERS), 2);
        fake_driver::fail_next(MAX_ACTIVE_CLUSTERS, 2);
        assert!(host.query_cluster_residency(symbol, config(1)).is_err());
    }
}

#[cfg(test)]
mod mps_tests {
    use super::mps_client_from;

    #[test]
    fn any_mps_variable_or_the_default_pipe_reads_as_a_client() {
        assert!(!mps_client_from(|_| false, false));
        assert!(mps_client_from(|_| false, true));
        for variable in [
            "CUDA_MPS_PIPE_DIRECTORY",
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
            "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT",
        ] {
            assert!(
                mps_client_from(|name| name == variable, false),
                "{variable}"
            );
        }
        assert!(!mps_client_from(
            |name| name == "CUDA_VISIBLE_DEVICES",
            false
        ));
    }
}

/// Allocate-then-write stress on the real driver. Every buffer carries guard
/// bands around its payload, is written straight after its allocation (from
/// its own stream or the other one, after a zero queued on the writer's
/// stream and behind copies queued on both), and every live buffer is read
/// back each round while the pool reuses what earlier rounds released.
#[cfg(all(test, not(feature = "cuda-fake-driver-tests")))]
mod real_driver_ordering_tests {
    use super::*;

    const GUARD_BYTES: usize = 4096;
    const BUFFERS_PER_ROUND: usize = 32;
    const BUSY_BYTES: u64 = 256 << 20;

    struct Live {
        object: DeviceObjectV1,
        image: Vec<u8>,
        round: usize,
        writer: &'static str,
    }

    fn next(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state >> 33
    }

    fn whole(object: DeviceObjectV1, bytes: usize) -> DeviceRangeV1 {
        DeviceRangeV1 {
            object,
            offset: 0,
            length: bytes as u64,
        }
    }

    #[test]
    #[ignore = "real CUDA driver; CUDA_VISIBLE_DEVICES names one idle GPU"]
    fn allocate_then_write_keeps_guard_bands_gpu() {
        let rounds = std::env::var("CHOPINFER_ALLOC_ORDER_ROUNDS")
            .ok()
            .map_or(200, |value| value.parse::<usize>().unwrap());
        let mut device = CudaHostDeviceV1::new().unwrap();
        let context = device.create_context(0, 0).unwrap();
        let pool = device.create_memory_pool(context).unwrap();
        let streams = [
            device.create_stream(context, 0).unwrap(),
            device.create_stream(context, 0).unwrap(),
        ];
        let busy = device
            .allocate_device_buffer(pool, Some(streams[0]), BUSY_BYTES, 256)
            .unwrap();
        let half = BUSY_BYTES / 2;
        let (low, high) = (
            DeviceRangeV1 {
                object: busy,
                offset: 0,
                length: half,
            },
            DeviceRangeV1 {
                object: busy,
                offset: half,
                length: half,
            },
        );
        let mut seed = 0x5eed_u64;
        let mut live: Vec<Live> = Vec::new();
        let (mut allocated, mut written_bytes, mut checks) = (0_usize, 0_usize, 0_usize);
        let mut mismatches = Vec::new();
        for round in 0..rounds {
            for stream in streams {
                for _ in 0..4 {
                    device.copy_device_to_device(stream, high, low).unwrap();
                }
            }
            let mut kept = Vec::with_capacity(live.len());
            for entry in live.drain(..) {
                if next(&mut seed) % 2 == 0 {
                    device.release_device_buffer(entry.object).unwrap();
                } else {
                    kept.push(entry);
                }
            }
            live = kept;
            for index in 0..BUFFERS_PER_ROUND {
                let payload =
                    (1_usize << (next(&mut seed) % 23)) + (next(&mut seed) % 4096) as usize;
                let total = payload + 2 * GUARD_BYTES;
                let owner = streams[index % 2];
                let writer_index = (index + round) % 2;
                let writer = streams[writer_index];
                let object = device
                    .allocate_device_buffer(pool, Some(owner), total as u64, 256)
                    .unwrap();
                let mut image = vec![0xa5 ^ index as u8; total];
                for byte in &mut image[GUARD_BYTES..GUARD_BYTES + payload] {
                    *byte = next(&mut seed) as u8;
                }
                let range = whole(object, total);
                device.zero_device_buffer(writer, range).unwrap();
                let api = match next(&mut seed) % 3 {
                    0 => {
                        device.copy_host_to_device(writer, range, &image).unwrap();
                        "copy_host_to_device"
                    }
                    1 => {
                        device
                            .copy_host_to_device_ordered(writer, range, &image)
                            .unwrap();
                        "copy_host_to_device_ordered"
                    }
                    _ => {
                        // The zero went to the writer's stream; this write
                        // uses the owner's, so order it after the zero.
                        device.synchronize_stream(writer).unwrap();
                        device.write_range_through_host_v1(range, &image).unwrap();
                        "write_range_through_host"
                    }
                };
                let writer = match (api, writer_index == index % 2) {
                    ("write_range_through_host", _) => "write_range_through_host (owner stream)",
                    ("copy_host_to_device", true) => "copy_host_to_device (owner stream)",
                    ("copy_host_to_device", false) => "copy_host_to_device (other stream)",
                    (_, true) => "copy_host_to_device_ordered (owner stream)",
                    (_, false) => "copy_host_to_device_ordered (other stream)",
                };
                allocated += 1;
                written_bytes += total;
                live.push(Live {
                    object,
                    image,
                    round,
                    writer,
                });
            }
            for stream in streams {
                device.synchronize_stream(stream).unwrap();
            }
            for entry in &live {
                let mut output = vec![0_u8; entry.image.len()];
                device
                    .copy_device_to_host(
                        streams[0],
                        &mut output,
                        whole(entry.object, entry.image.len()),
                    )
                    .unwrap();
                checks += 1;
                if output != entry.image {
                    let differing = output
                        .iter()
                        .zip(&entry.image)
                        .enumerate()
                        .filter(|(_, (a, b))| a != b)
                        .map(|(offset, _)| offset)
                        .collect::<Vec<_>>();
                    let payload_end = entry.image.len() - GUARD_BYTES;
                    let guards = differing
                        .iter()
                        .filter(|&&offset| offset < GUARD_BYTES || offset >= payload_end)
                        .count();
                    mismatches.push(format!(
                        "round {round}: buffer from round {} written by {}: {} bytes differ \
                         ({guards} in guard bands), first at {} of {}",
                        entry.round,
                        entry.writer,
                        differing.len(),
                        differing[0],
                        entry.image.len()
                    ));
                }
            }
        }
        println!(
            "allocate-then-write: rounds={rounds} allocations={allocated} bytes_written={written_bytes} \
             buffer_checks={checks} mismatches={}",
            mismatches.len()
        );
        for mismatch in mismatches.iter().take(20) {
            println!("  {mismatch}");
        }
        assert!(
            mismatches.is_empty(),
            "{} mismatched buffers",
            mismatches.len()
        );
        for entry in live {
            device.release_device_buffer(entry.object).unwrap();
        }
        device.release_device_buffer(busy).unwrap();
    }
}
