//! A fake host over a model's package. [`Backend`] answers every host
//! service: it records the device-facing ones on the package's
//! [`FakeDevice`] and counts every handle it hands out and every release it
//! admits. Its [`FakePackage`] answers the services that read the package
//! (images, arenas, views, components, the serving KV, speculation and
//! state), minting handle IDs with [`FakeDevice::id`], and hooks into the
//! device services where its model checks or records more.
//! An `admit_*` hook runs before the device records a call and may refuse
//! it; an `on_*` hook runs after and may still fail it. A release is
//! refused, if at all, before anything about its handle changes.

use std::fmt;
use std::ops::Deref;

use runtime_bundle_abi::{
    AuthenticatedComponentMetadataV1, BundleAuthenticatedComponentHandleV1,
    BundleCompiledImageHandleV1, BundleContextHandleV1, BundleCudaModuleHandleV1,
    BundleCudaSymbolHandleV1, BundleDeviceBufferHandleV1, BundleEventHandleV1, BundleGraphHandleV1,
    BundleMemoryPoolHandleV1, BundlePeerAccessHandleV1, BundleRankArenaHandleV1,
    BundleRawBatchLimitsV1, BundleRawClusterResidencyV1, BundleRawSpeculationV1,
    BundleStateHandleV1, BundleStatusV1, BundleStreamHandleV1, CompiledImageSpanV1,
    DeviceBufferSpanV1, HostServiceResultV1, HostServicesBackendV1, KernelArgumentV1,
    LaunchConfigV1, MappedRowGatherV1, StateRequestV1,
};

use super::{FakeDevice, JournalObserver, SnapshotOp, fnv1a};

/// The most dynamic shared memory a launch may configure on the target
/// device.
const DYNAMIC_SHARED_BYTES: u32 = 101_376;

/// CUDA's maximum dynamic shared memory before a symbol opts in.
const DEFAULT_DYNAMIC_SHARED_BYTES: u32 = 48 * 1024;

/// A host handle the backend acquired or releases, by slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Handle {
    Context(u32),
    Pool(u32),
    /// A pool's grant to a peer context.
    PeerGrant {
        pool: u32,
        peer: u32,
    },
    Buffer(u32),
    Image(u32),
    Arena(u32),
    Component(u32),
    PeerAccess(u32),
    Stream(u32),
    Event(u32),
    Module(u32),
    Symbol(u32),
    State(u32),
    Graph(u32),
}

impl fmt::Display for Handle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (kind, slot) = match *self {
            Self::PeerGrant { pool, peer } => return write!(f, "pool {pool} peer {peer}"),
            Self::Context(slot) => ("context", slot),
            Self::Pool(slot) => ("pool", slot),
            Self::Buffer(slot) => ("buffer", slot),
            Self::Image(slot) => ("image", slot),
            Self::Arena(slot) => ("arena", slot),
            Self::Component(slot) => ("component", slot),
            Self::PeerAccess(slot) => ("peer access", slot),
            Self::Stream(slot) => ("stream", slot),
            Self::Event(slot) => ("event", slot),
            Self::Module(slot) => ("module", slot),
            Self::Symbol(slot) => ("symbol", slot),
            Self::State(slot) => ("state", slot),
            Self::Graph(slot) => ("graph", slot),
        };
        write!(f, "{kind} {slot}")
    }
}

/// The host services of a fake host over the package `P`, which it derefs
/// to.
#[derive(Default)]
pub struct Backend<P>(P);

impl<P> Backend<P> {
    pub fn new(package: P) -> Self {
        Self(package)
    }

    pub fn into_inner(self) -> P {
        self.0
    }
}

impl<P> Deref for Backend<P> {
    type Target = P;

    fn deref(&self) -> &P {
        &self.0
    }
}

/// What a model's fake host answers itself. The package services take the
/// signatures of `HostServicesBackendV1`; the hooks default to recording
/// nothing more.
pub trait FakePackage {
    type Observer: JournalObserver;

    fn device(&self) -> &FakeDevice<Self::Observer>;

    /// The cubins the device names loaded modules by.
    fn cubins() -> impl IntoIterator<Item = (&'static str, &'static [u8])>;

    fn open_compiled_image(&self, role: &[u8]) -> HostServiceResultV1<BundleCompiledImageHandleV1>;
    fn upload_rank_arena(
        &self,
        context: BundleContextHandleV1,
        pool: BundleMemoryPoolHandleV1,
        stream: BundleStreamHandleV1,
        image: BundleCompiledImageHandleV1,
        rank: u32,
    ) -> HostServiceResultV1<BundleRankArenaHandleV1>;
    fn resolve_compiled_image_view(
        &self,
        image: BundleCompiledImageHandleV1,
        arena: BundleRankArenaHandleV1,
        rank: u32,
        view_id: &[u8],
    ) -> HostServiceResultV1<CompiledImageSpanV1>;
    fn open_authenticated_component(
        &self,
        role: &[u8],
    ) -> HostServiceResultV1<BundleAuthenticatedComponentHandleV1>;
    /// The role and bytes of the component handle slot `slot`.
    fn component(&self, slot: u32) -> (String, Vec<u8>);
    fn acquire_serving_kv(&self, layout_id: &[u8]) -> HostServiceResultV1<u64>;
    fn resolve_serving_kv_pool(&self, rank: u32) -> HostServiceResultV1<DeviceBufferSpanV1>;
    fn query_speculation(&self) -> HostServiceResultV1<BundleRawSpeculationV1>;

    /// The bytes of one host-mapped view, whose rows the backend gathers
    /// (`chop.host-mapped-gather.v1`): by default the package maps none.
    fn mapped_view(
        &self,
        _image: BundleCompiledImageHandleV1,
        _rank: u32,
        _view_id: &[u8],
    ) -> HostServiceResultV1<Vec<u8>> {
        Err(BundleStatusV1::NotFound)
    }

    /// What the device reports for a cluster residency query that nothing
    /// set on the [`FakeDevice`] answers: by default no report, a host that
    /// cannot vouch for residency. A model's package reports its target's.
    fn device_residency(&self) -> BundleRawClusterResidencyV1 {
        BundleRawClusterResidencyV1::unreported()
    }
    fn allocate_state(&self, request: &StateRequestV1) -> HostServiceResultV1<BundleStateHandleV1>;
    fn resolve_state_view(
        &self,
        state: BundleStateHandleV1,
        rank: u32,
    ) -> HostServiceResultV1<DeviceBufferSpanV1>;
    /// Fills a readback's `destination` with the bytes of `source`.
    fn read_back(
        &self,
        destination: &mut [u8],
        source: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()>;

    fn log(&self, message: &[u8]) {
        self.device().log(message);
    }

    /// After the host acquires `handle`, once it is counted. A peer grant
    /// is held like a handle.
    fn on_acquire(&self, _handle: Handle) {}

    /// May refuse to release `handle`; a refused handle stays live.
    fn admit_release(&self, _handle: Handle) -> HostServiceResultV1<()> {
        Ok(())
    }

    /// Releases `handle` once it is counted: a buffer's once its extent is
    /// gone, a graph's once the device dropped it.
    fn release(&self, _handle: Handle) {
        self.device().note("release".into());
    }

    /// Panics unless `span` lies inside a span the host handed out.
    fn check_span(&self, span: DeviceBufferSpanV1) {
        self.device().check_span(span);
    }

    fn admit_allocation(
        &self,
        _pool: BundleMemoryPoolHandleV1,
        _byte_length: u64,
        _alignment: u64,
    ) -> HostServiceResultV1<()> {
        Ok(())
    }

    /// After the host minted `buffer`, before the device allocates it: an
    /// injected device failure leaves what this records.
    fn before_allocation(
        &self,
        _pool: BundleMemoryPoolHandleV1,
        _buffer: BundleDeviceBufferHandleV1,
        _byte_length: u64,
    ) {
    }

    fn on_allocation(
        &self,
        _pool: BundleMemoryPoolHandleV1,
        _buffer: BundleDeviceBufferHandleV1,
        _byte_length: u64,
    ) {
    }

    /// Before the device synchronizes `stream`: a refused synchronization
    /// reaches no journal.
    fn admit_synchronize(&self, _stream: BundleStreamHandleV1) -> HostServiceResultV1<()> {
        Ok(())
    }

    fn on_synchronize(&self, _stream: BundleStreamHandleV1) -> HostServiceResultV1<()> {
        Ok(())
    }

    /// After `stream` records (`wait` false) or waits on `event`.
    fn on_event(&self, _stream: BundleStreamHandleV1, _event: BundleEventHandleV1, _wait: bool) {}

    fn admit_upload(
        &self,
        _stream: BundleStreamHandleV1,
        _destination: DeviceBufferSpanV1,
        _source: &[u8],
    ) -> HostServiceResultV1<()> {
        Ok(())
    }

    /// The bytes of `source` an upload's journal line digests.
    fn journaled<'a>(&self, source: &'a [u8]) -> &'a [u8] {
        source
    }

    fn on_upload(
        &self,
        _stream: BundleStreamHandleV1,
        _destination: DeviceBufferSpanV1,
        _source: &[u8],
    ) -> HostServiceResultV1<()> {
        Ok(())
    }

    fn admit_download(
        &self,
        _stream: BundleStreamHandleV1,
        _source: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()> {
        Ok(())
    }

    /// The journal line of a readback of `bytes` from `source` on `stream`.
    fn download_line(&self, stream: u32, source: DeviceBufferSpanV1, _bytes: usize) -> String {
        format!("download {stream} {source:?}")
    }

    /// After the injected-copy-failure check passes, before the copy's
    /// length and span checks.
    fn admit_copy(
        &self,
        _stream: BundleStreamHandleV1,
        _destination: DeviceBufferSpanV1,
        _source: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()> {
        Ok(())
    }

    fn on_copy(
        &self,
        _stream: BundleStreamHandleV1,
        _destination: DeviceBufferSpanV1,
        _source: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()> {
        Ok(())
    }

    fn copy_peer_to_peer(
        &self,
        _stream: BundleStreamHandleV1,
        _destination_context: BundleContextHandleV1,
        _destination: DeviceBufferSpanV1,
        _source_context: BundleContextHandleV1,
        _source: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()> {
        Err(BundleStatusV1::Unsupported)
    }

    fn on_zero(&self, _stream: BundleStreamHandleV1, _destination: DeviceBufferSpanV1) {}

    fn on_resolve(&self, _symbol_name: &[u8]) {}

    /// Before a launch of `name`, once its configuration passed, before an
    /// injected launch failure and the checks of its buffer arguments.
    fn admit_launch(
        &self,
        _stream: BundleStreamHandleV1,
        _name: &str,
        _config: LaunchConfigV1,
        _arguments: &[KernelArgumentV1],
    ) -> HostServiceResultV1<()> {
        Ok(())
    }

    /// Before a launch of `name` whose buffer arguments `spans` passed their
    /// checks.
    fn admit_spans(
        &self,
        _stream: BundleStreamHandleV1,
        _name: &str,
        _spans: &[DeviceBufferSpanV1],
    ) -> HostServiceResultV1<()> {
        Ok(())
    }

    fn on_launch(
        &self,
        _stream: BundleStreamHandleV1,
        _name: &str,
        _config: LaunchConfigV1,
        _arguments: &[KernelArgumentV1],
    ) -> HostServiceResultV1<()> {
        Ok(())
    }

    fn on_capture(&self, _stream: BundleStreamHandleV1) {}

    fn on_captured(&self, _stream: BundleStreamHandleV1, _graph: BundleGraphHandleV1) {}

    /// After `graph`, of `nodes` launches, replays on `stream`.
    fn on_replay(&self, _stream: BundleStreamHandleV1, _graph: BundleGraphHandleV1, _nodes: usize) {
    }
}

/// Every buffer a launch's arguments name, pointer arrays flattened.
fn launch_spans(arguments: &[KernelArgumentV1]) -> Vec<DeviceBufferSpanV1> {
    let mut spans = Vec::new();
    for argument in arguments {
        match argument {
            KernelArgumentV1::DeviceBuffer(span) => spans.push(*span),
            KernelArgumentV1::DevicePointerArray(array) => spans.extend(array.iter().flatten()),
            KernelArgumentV1::TensorMap(map) => spans.push(map.buffer),
            _ => {}
        }
    }
    spans
}

impl<P: FakePackage> Backend<P> {
    /// Counts `handle` as acquired.
    fn acquired(&self, handle: Handle) {
        let device = self.0.device();
        device.acquired.set(device.acquired.get() + 1);
        self.0.on_acquire(handle);
    }

    /// Releases `handle` unless the package refuses it; `drop` forgets the
    /// device's record of it first.
    fn release(&self, handle: Handle, drop: impl FnOnce()) -> HostServiceResultV1<()> {
        self.0.admit_release(handle)?;
        drop();
        let device = self.0.device();
        device.released.set(device.released.get() + 1);
        self.0.release(handle);
        Ok(())
    }
}

impl<P: FakePackage> HostServicesBackendV1 for Backend<P> {
    fn log(&self, message: &[u8]) -> HostServiceResultV1<()> {
        self.0.log(message);
        Ok(())
    }
    fn select_context(
        &self,
        rank: u32,
        _device_ordinal: i32,
    ) -> HostServiceResultV1<BundleContextHandleV1> {
        self.acquired(Handle::Context(rank + 1));
        Ok(BundleContextHandleV1::from_parts(1, rank + 1))
    }
    fn release_context(&self, context: BundleContextHandleV1) -> HostServiceResultV1<()> {
        self.release(Handle::Context(context.slot()), || {})
    }
    fn select_memory_pool(
        &self,
        context: BundleContextHandleV1,
    ) -> HostServiceResultV1<BundleMemoryPoolHandleV1> {
        let pool = self.0.device().memory_pool(context);
        self.acquired(Handle::Pool(pool.slot()));
        Ok(pool)
    }
    fn set_memory_pool_peer_access(
        &self,
        pool: BundleMemoryPoolHandleV1,
        peer_context: BundleContextHandleV1,
        enabled: bool,
    ) -> HostServiceResultV1<()> {
        let grant = Handle::PeerGrant {
            pool: pool.slot(),
            peer: peer_context.slot(),
        };
        if enabled {
            self.acquired(grant);
            Ok(())
        } else {
            self.release(grant, || {})
        }
    }
    fn release_memory_pool(&self, pool: BundleMemoryPoolHandleV1) -> HostServiceResultV1<()> {
        self.release(Handle::Pool(pool.slot()), || {})
    }
    fn allocate_device_buffer(
        &self,
        pool: BundleMemoryPoolHandleV1,
        stream: BundleStreamHandleV1,
        byte_length: u64,
        alignment: u64,
    ) -> HostServiceResultV1<BundleDeviceBufferHandleV1> {
        self.0.admit_allocation(pool, byte_length, alignment)?;
        let device = self.0.device();
        let id = device.acquire();
        self.0.before_allocation(
            pool,
            BundleDeviceBufferHandleV1::from_parts(1, id),
            byte_length,
        );
        let buffer = device.allocate(id, stream, byte_length, alignment);
        device.extents.borrow_mut().insert(id, byte_length);
        self.0.on_allocation(pool, buffer, byte_length);
        self.0.on_acquire(Handle::Buffer(id));
        Ok(buffer)
    }
    fn release_device_buffer(&self, buffer: BundleDeviceBufferHandleV1) -> HostServiceResultV1<()> {
        self.release(Handle::Buffer(buffer.slot()), || {
            self.0.device().extents.borrow_mut().remove(&buffer.slot());
        })
    }
    fn open_compiled_image(&self, role: &[u8]) -> HostServiceResultV1<BundleCompiledImageHandleV1> {
        let image = self.0.open_compiled_image(role)?;
        self.acquired(Handle::Image(image.slot()));
        Ok(image)
    }
    fn upload_rank_arena(
        &self,
        context: BundleContextHandleV1,
        pool: BundleMemoryPoolHandleV1,
        stream: BundleStreamHandleV1,
        image: BundleCompiledImageHandleV1,
        rank: u32,
    ) -> HostServiceResultV1<BundleRankArenaHandleV1> {
        let arena = self
            .0
            .upload_rank_arena(context, pool, stream, image, rank)?;
        self.acquired(Handle::Arena(arena.slot()));
        Ok(arena)
    }
    fn release_rank_arena(&self, arena: BundleRankArenaHandleV1) -> HostServiceResultV1<()> {
        self.release(Handle::Arena(arena.slot()), || {})
    }
    fn resolve_compiled_image_view(
        &self,
        image: BundleCompiledImageHandleV1,
        arena: BundleRankArenaHandleV1,
        rank: u32,
        view_id: &[u8],
    ) -> HostServiceResultV1<CompiledImageSpanV1> {
        self.0
            .resolve_compiled_image_view(image, arena, rank, view_id)
    }
    fn release_compiled_image(
        &self,
        image: BundleCompiledImageHandleV1,
    ) -> HostServiceResultV1<()> {
        self.release(Handle::Image(image.slot()), || {})
    }
    fn open_authenticated_component(
        &self,
        role: &[u8],
    ) -> HostServiceResultV1<BundleAuthenticatedComponentHandleV1> {
        let component = self.0.open_authenticated_component(role)?;
        self.acquired(Handle::Component(component.slot()));
        Ok(component)
    }
    fn inspect_authenticated_component(
        &self,
        component: BundleAuthenticatedComponentHandleV1,
    ) -> HostServiceResultV1<AuthenticatedComponentMetadataV1> {
        let (role, bytes) = self.0.component(component.slot());
        Ok(AuthenticatedComponentMetadataV1::new(
            role.len() as u64,
            bytes.len() as u64,
            [b'a'; 64],
        ))
    }
    fn read_authenticated_component(
        &self,
        component: BundleAuthenticatedComponentHandleV1,
        role_destination: &mut [u8],
        bytes_destination: &mut [u8],
    ) -> HostServiceResultV1<()> {
        let (role, bytes) = self.0.component(component.slot());
        role_destination.copy_from_slice(role.as_bytes());
        bytes_destination.copy_from_slice(&bytes);
        Ok(())
    }
    fn release_authenticated_component(
        &self,
        component: BundleAuthenticatedComponentHandleV1,
    ) -> HostServiceResultV1<()> {
        self.release(Handle::Component(component.slot()), || {})
    }
    fn acquire_peer_access(
        &self,
        _owner_context: BundleContextHandleV1,
        _peer_context: BundleContextHandleV1,
    ) -> HostServiceResultV1<BundlePeerAccessHandleV1> {
        let access = BundlePeerAccessHandleV1::from_parts(1, self.0.device().acquire());
        self.0.on_acquire(Handle::PeerAccess(access.slot()));
        Ok(access)
    }
    fn release_peer_access(
        &self,
        peer_access: BundlePeerAccessHandleV1,
    ) -> HostServiceResultV1<()> {
        self.release(Handle::PeerAccess(peer_access.slot()), || {})
    }
    fn create_stream(
        &self,
        context: BundleContextHandleV1,
        _stream_kind: u32,
    ) -> HostServiceResultV1<BundleStreamHandleV1> {
        let device = self.0.device();
        let stream = device.create_stream(device.acquire(), context);
        self.0.on_acquire(Handle::Stream(stream.slot()));
        Ok(stream)
    }
    fn synchronize_stream(&self, stream: BundleStreamHandleV1) -> HostServiceResultV1<()> {
        self.0.admit_synchronize(stream)?;
        self.0.device().synchronize_stream(stream);
        self.0.on_synchronize(stream)
    }
    fn release_stream(&self, stream: BundleStreamHandleV1) -> HostServiceResultV1<()> {
        self.release(Handle::Stream(stream.slot()), || {})
    }
    fn create_event(
        &self,
        _context: BundleContextHandleV1,
    ) -> HostServiceResultV1<BundleEventHandleV1> {
        let device = self.0.device();
        let event = device.create_event(device.acquire());
        self.0.on_acquire(Handle::Event(event.slot()));
        Ok(event)
    }
    fn record_event(
        &self,
        event: BundleEventHandleV1,
        stream: BundleStreamHandleV1,
    ) -> HostServiceResultV1<()> {
        self.0.device().record_event(event, stream)?;
        self.0.on_event(stream, event, false);
        Ok(())
    }
    fn wait_event(
        &self,
        stream: BundleStreamHandleV1,
        event: BundleEventHandleV1,
    ) -> HostServiceResultV1<()> {
        self.0.device().wait_event(stream, event)?;
        self.0.on_event(stream, event, true);
        Ok(())
    }
    fn release_event(&self, event: BundleEventHandleV1) -> HostServiceResultV1<()> {
        self.release(Handle::Event(event.slot()), || {})
    }
    fn copy_host_to_device(
        &self,
        stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
        source: &[u8],
    ) -> HostServiceResultV1<()> {
        assert_eq!(destination.length, source.len() as u64);
        self.0.check_span(destination);
        self.0.admit_upload(stream, destination, source)?;
        let device = self.0.device();
        device.snapshot_op(|| SnapshotOp::Upload(stream.slot(), destination, source.to_vec()));
        device.note(format!(
            "upload {} {destination:?} {:016x}",
            stream.slot(),
            fnv1a(self.0.journaled(source))
        ));
        self.0.on_upload(stream, destination, source)
    }
    /// Gathers the rows from the package's mapped view and uploads them,
    /// checked as the host checks them (refused while any stream on the
    /// device captures, since a graph would replay rows read once) and
    /// journalled as
    /// `gather <stream> <view> <rows> <destination> <digest>`.
    fn gather_mapped_rows(
        &self,
        stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
        gather: MappedRowGatherV1<'_>,
    ) -> HostServiceResultV1<()> {
        if self.0.device().capturing_on_device_of(stream.slot()) {
            return Err(BundleStatusV1::InvalidState);
        }
        let table = self
            .0
            .mapped_view(gather.image, gather.rank, gather.view_id)?;
        let row_bytes =
            usize::try_from(gather.row_bytes).map_err(|_| BundleStatusV1::InvalidArgument)?;
        if gather.rows.is_empty()
            || row_bytes == 0
            || !table.len().is_multiple_of(row_bytes)
            || gather
                .rows
                .iter()
                .any(|&row| row as usize >= table.len() / row_bytes)
            || (gather.rows.len() as u64).checked_mul(gather.row_bytes) != Some(destination.length)
        {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let bytes: Vec<u8> = gather
            .rows
            .iter()
            .flat_map(|&row| &table[row as usize * row_bytes..(row as usize + 1) * row_bytes])
            .copied()
            .collect();
        self.0.check_span(destination);
        self.0.admit_upload(stream, destination, &bytes)?;
        let device = self.0.device();
        device.snapshot_op(|| SnapshotOp::Upload(stream.slot(), destination, bytes.clone()));
        device.note(format!(
            "gather {} {} {} {destination:?} {:016x}",
            stream.slot(),
            String::from_utf8_lossy(gather.view_id),
            gather.rows.len(),
            fnv1a(&bytes)
        ));
        self.0.on_upload(stream, destination, &bytes)
    }
    fn copy_device_to_host(
        &self,
        stream: BundleStreamHandleV1,
        destination: &mut [u8],
        source: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()> {
        self.0.admit_download(stream, source)?;
        assert_eq!(destination.len() as u64, source.length);
        self.0.check_span(source);
        let device = self.0.device();
        device.snapshot_op(|| SnapshotOp::Read(stream.slot(), source));
        device.note(
            self.0
                .download_line(stream.slot(), source, destination.len()),
        );
        self.0.read_back(destination, source)
    }
    fn copy_device_to_device(
        &self,
        stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
        source: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()> {
        let device = self.0.device();
        if device.fail_copy.fails() {
            return Err(BundleStatusV1::HostError);
        }
        self.0.admit_copy(stream, destination, source)?;
        assert_eq!(destination.length, source.length);
        self.0.check_span(source);
        self.0.check_span(destination);
        device.note(format!("copy {} {source:?} {destination:?}", stream.slot()));
        device.snapshot_op(|| SnapshotOp::Copy(stream.slot(), source, destination));
        self.0.on_copy(stream, destination, source)
    }
    fn copy_peer_to_peer(
        &self,
        stream: BundleStreamHandleV1,
        destination_context: BundleContextHandleV1,
        destination: DeviceBufferSpanV1,
        source_context: BundleContextHandleV1,
        source: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()> {
        self.0.copy_peer_to_peer(
            stream,
            destination_context,
            destination,
            source_context,
            source,
        )
    }
    fn zero_device_buffer(
        &self,
        stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()> {
        self.0.check_span(destination);
        self.0.device().zero(stream, destination);
        self.0.on_zero(stream, destination);
        Ok(())
    }
    fn load_cuda_module(
        &self,
        _context: BundleContextHandleV1,
        module_bytes: &[u8],
    ) -> HostServiceResultV1<BundleCudaModuleHandleV1> {
        let device = self.0.device();
        let module = device.load_module(device.acquire(), module_bytes, P::cubins());
        self.0.on_acquire(Handle::Module(module.slot()));
        Ok(module)
    }
    fn resolve_cuda_symbol(
        &self,
        module: BundleCudaModuleHandleV1,
        symbol_name: &[u8],
    ) -> HostServiceResultV1<BundleCudaSymbolHandleV1> {
        self.0.on_resolve(symbol_name);
        let device = self.0.device();
        let symbol = device.resolve_symbol(device.acquire(), module, symbol_name)?;
        self.0.on_acquire(Handle::Symbol(symbol.slot()));
        Ok(symbol)
    }
    fn configure_cuda_symbol(
        &self,
        symbol: BundleCudaSymbolHandleV1,
        dynamic_shared_memory_bytes: u32,
    ) -> HostServiceResultV1<()> {
        assert!(dynamic_shared_memory_bytes <= DYNAMIC_SHARED_BYTES);
        self.0
            .device()
            .configure_symbol(symbol, dynamic_shared_memory_bytes);
        Ok(())
    }
    fn launch_cuda_symbol(
        &self,
        stream: BundleStreamHandleV1,
        symbol: BundleCudaSymbolHandleV1,
        config: LaunchConfigV1,
        arguments: &[KernelArgumentV1],
    ) -> HostServiceResultV1<()> {
        let device = self.0.device();
        let name = device.symbol_name(symbol, BundleStatusV1::NotFound)?;
        device.check_launch(config);
        let configured = device
            .symbol_dynamic_shared_bytes
            .borrow()
            .get(&symbol.slot())
            .copied()
            .unwrap_or(DEFAULT_DYNAMIC_SHARED_BYTES);
        if config.shared_memory_bytes > configured.min(DYNAMIC_SHARED_BYTES) {
            eprintln!(
                "fake host launch_cuda_symbol {name}: requested {} dynamic shared bytes exceeds symbol limit {configured} bytes or device opt-in limit {DYNAMIC_SHARED_BYTES} bytes",
                config.shared_memory_bytes
            );
            // runtime-host maps CUDA_ERROR_INVALID_VALUE from the driver to HostError.
            return Err(BundleStatusV1::HostError);
        }
        self.0.admit_launch(stream, &name, config, arguments)?;
        if device.fail_launch.fails() {
            return Err(BundleStatusV1::HostError);
        }
        let spans = launch_spans(arguments);
        spans.iter().for_each(|span| self.0.check_span(*span));
        self.0.admit_spans(stream, &name, &spans)?;
        device.launched(stream, &name, config, arguments);
        self.0.on_launch(stream, &name, config, arguments)
    }
    fn release_cuda_symbol(&self, symbol: BundleCudaSymbolHandleV1) -> HostServiceResultV1<()> {
        self.release(Handle::Symbol(symbol.slot()), || {})
    }
    fn release_cuda_module(&self, module: BundleCudaModuleHandleV1) -> HostServiceResultV1<()> {
        self.release(Handle::Module(module.slot()), || {})
    }
    fn acquire_serving_kv(&self, layout_id: &[u8]) -> HostServiceResultV1<u64> {
        self.0.acquire_serving_kv(layout_id)
    }
    fn resolve_serving_kv_pool(&self, rank: u32) -> HostServiceResultV1<DeviceBufferSpanV1> {
        self.0.resolve_serving_kv_pool(rank)
    }
    fn query_speculation(&self) -> HostServiceResultV1<BundleRawSpeculationV1> {
        self.0.query_speculation()
    }
    fn query_batch_limits(&self) -> HostServiceResultV1<BundleRawBatchLimitsV1> {
        self.0.device().query_batch_limits()
    }
    fn query_cluster_residency(
        &self,
        symbol: BundleCudaSymbolHandleV1,
        config: LaunchConfigV1,
    ) -> HostServiceResultV1<BundleRawClusterResidencyV1> {
        Ok(self
            .0
            .device()
            .query_cluster_residency(symbol, config)?
            .unwrap_or_else(|| self.0.device_residency()))
    }
    fn allocate_state(&self, request: &StateRequestV1) -> HostServiceResultV1<BundleStateHandleV1> {
        let state = self.0.allocate_state(request)?;
        self.acquired(Handle::State(state.slot()));
        Ok(state)
    }
    fn resolve_state_view(
        &self,
        state: BundleStateHandleV1,
        rank: u32,
    ) -> HostServiceResultV1<DeviceBufferSpanV1> {
        self.0.resolve_state_view(state, rank)
    }
    fn release_state(&self, state: BundleStateHandleV1) -> HostServiceResultV1<()> {
        self.release(Handle::State(state.slot()), || {})
    }
    fn begin_graph_capture(&self, stream: BundleStreamHandleV1) -> HostServiceResultV1<()> {
        self.0.device().begin_capture(stream);
        self.0.on_capture(stream);
        Ok(())
    }
    fn end_graph_capture(
        &self,
        stream: BundleStreamHandleV1,
    ) -> HostServiceResultV1<BundleGraphHandleV1> {
        let device = self.0.device();
        let (graph, _) = device.end_capture(stream, || device.acquire())?;
        self.0.on_acquire(Handle::Graph(graph.slot()));
        self.0.on_captured(stream, graph);
        Ok(graph)
    }
    fn launch_graph(
        &self,
        stream: BundleStreamHandleV1,
        graph: BundleGraphHandleV1,
    ) -> HostServiceResultV1<()> {
        let nodes = self.0.device().launch_graph(stream, graph)?;
        self.0.on_replay(stream, graph, nodes);
        Ok(())
    }
    fn release_graph(&self, graph: BundleGraphHandleV1) -> HostServiceResultV1<()> {
        self.release(Handle::Graph(graph.slot()), || {
            self.0.device().release_graph(graph);
        })
    }
    crate::unused_host_services!();
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;
    use crate::fake_host::{Countdown, NoObserver};

    /// A package without images, components, KV or state, whose readbacks
    /// read sevens. It holds its live handles, newest last, releases only
    /// the newest, and refuses a synchronization when its countdown fires.
    /// Its one mapped view, `rows` of rank 1, holds `0..32`.
    #[derive(Default)]
    struct Empty(FakeDevice, RefCell<Vec<String>>, Countdown);

    impl FakePackage for Empty {
        type Observer = NoObserver;

        fn device(&self) -> &FakeDevice {
            &self.0
        }
        fn cubins() -> impl IntoIterator<Item = (&'static str, &'static [u8])> {
            []
        }
        fn open_compiled_image(
            &self,
            _role: &[u8],
        ) -> HostServiceResultV1<BundleCompiledImageHandleV1> {
            Err(BundleStatusV1::Unsupported)
        }
        fn upload_rank_arena(
            &self,
            _context: BundleContextHandleV1,
            _pool: BundleMemoryPoolHandleV1,
            _stream: BundleStreamHandleV1,
            _image: BundleCompiledImageHandleV1,
            _rank: u32,
        ) -> HostServiceResultV1<BundleRankArenaHandleV1> {
            Err(BundleStatusV1::Unsupported)
        }
        fn resolve_compiled_image_view(
            &self,
            _image: BundleCompiledImageHandleV1,
            _arena: BundleRankArenaHandleV1,
            _rank: u32,
            _view_id: &[u8],
        ) -> HostServiceResultV1<CompiledImageSpanV1> {
            Err(BundleStatusV1::Unsupported)
        }
        fn open_authenticated_component(
            &self,
            _role: &[u8],
        ) -> HostServiceResultV1<BundleAuthenticatedComponentHandleV1> {
            Err(BundleStatusV1::NotFound)
        }
        fn component(&self, slot: u32) -> (String, Vec<u8>) {
            unreachable!("no component {slot} was opened")
        }
        fn acquire_serving_kv(&self, _layout_id: &[u8]) -> HostServiceResultV1<u64> {
            Err(BundleStatusV1::Unsupported)
        }
        fn resolve_serving_kv_pool(&self, _rank: u32) -> HostServiceResultV1<DeviceBufferSpanV1> {
            Err(BundleStatusV1::Unsupported)
        }
        fn query_speculation(&self) -> HostServiceResultV1<BundleRawSpeculationV1> {
            Ok(BundleRawSpeculationV1::disabled())
        }
        fn mapped_view(
            &self,
            _image: BundleCompiledImageHandleV1,
            rank: u32,
            view_id: &[u8],
        ) -> HostServiceResultV1<Vec<u8>> {
            if rank == 1 && view_id == b"rows" {
                Ok((0..32).collect())
            } else {
                Err(BundleStatusV1::NotFound)
            }
        }
        fn allocate_state(
            &self,
            _request: &StateRequestV1,
        ) -> HostServiceResultV1<BundleStateHandleV1> {
            Err(BundleStatusV1::Unsupported)
        }
        fn resolve_state_view(
            &self,
            _state: BundleStateHandleV1,
            _rank: u32,
        ) -> HostServiceResultV1<DeviceBufferSpanV1> {
            Err(BundleStatusV1::Unsupported)
        }
        fn read_back(
            &self,
            destination: &mut [u8],
            _source: DeviceBufferSpanV1,
        ) -> HostServiceResultV1<()> {
            destination.fill(7);
            Ok(())
        }
        fn on_acquire(&self, handle: Handle) {
            self.1.borrow_mut().push(handle.to_string());
        }
        fn admit_release(&self, handle: Handle) -> HostServiceResultV1<()> {
            if self.1.borrow().last() != Some(&handle.to_string()) {
                return Err(BundleStatusV1::InvalidState);
            }
            Ok(())
        }
        fn release(&self, _handle: Handle) {
            self.1.borrow_mut().pop();
            self.0.note("release".into());
        }
        fn admit_synchronize(&self, _stream: BundleStreamHandleV1) -> HostServiceResultV1<()> {
            if self.2.fails() {
                return Err(BundleStatusV1::HostError);
            }
            Ok(())
        }
    }

    /// A host with one rank's context, pool, stream and a 64-byte buffer.
    fn host() -> (
        Backend<Empty>,
        BundleStreamHandleV1,
        BundleDeviceBufferHandleV1,
    ) {
        let host = Backend::new(Empty::default());
        let context = host.select_context(0, 0).unwrap();
        let pool = host.select_memory_pool(context).unwrap();
        let stream = host.create_stream(context, 0).unwrap();
        let buffer = host.allocate_device_buffer(pool, stream, 64, 16).unwrap();
        (host, stream, buffer)
    }

    fn host_with_symbol() -> (
        Backend<Empty>,
        BundleStreamHandleV1,
        BundleCudaSymbolHandleV1,
    ) {
        let (host, stream, _) = host();
        let module = host
            .load_cuda_module(BundleContextHandleV1::from_parts(1, 1), b"module")
            .unwrap();
        let symbol = host.resolve_cuda_symbol(module, b"shared_kernel").unwrap();
        (host, stream, symbol)
    }

    fn shared_launch(bytes: u32) -> LaunchConfigV1 {
        LaunchConfigV1 {
            grid_x: 1,
            grid_y: 1,
            grid_z: 1,
            block_x: 32,
            block_y: 1,
            block_z: 1,
            shared_memory_bytes: bytes,
            flags: runtime_bundle_abi::BUNDLE_LAUNCH_FLAG_V1_PROGRAMMATIC_STREAM_SERIALIZATION,
            cluster: runtime_bundle_abi::LaunchClusterV1::NONE,
        }
    }

    #[test]
    fn dynamic_shared_memory_above_default_requires_symbol_configuration() {
        let (host, stream, symbol) = host_with_symbol();
        let config = shared_launch(48 * 1024 + 1);
        let lines = host.device().journal.borrow().len();
        assert_eq!(
            host.launch_cuda_symbol(stream, symbol, config, &[]),
            Err(BundleStatusV1::HostError)
        );
        assert_eq!(host.device().journal.borrow().len(), lines);
        host.configure_cuda_symbol(symbol, config.shared_memory_bytes)
            .unwrap();
        host.launch_cuda_symbol(stream, symbol, config, &[])
            .unwrap();

        // Configuring one module's symbol leaves another's default intact.
        let module = host
            .load_cuda_module(BundleContextHandleV1::from_parts(1, 1), b"other module")
            .unwrap();
        let other = host.resolve_cuda_symbol(module, b"shared_kernel").unwrap();
        assert_eq!(
            host.launch_cuda_symbol(stream, other, config, &[]),
            Err(BundleStatusV1::HostError)
        );
    }

    #[test]
    fn dynamic_shared_memory_above_opt_in_ceiling_is_refused_during_capture() {
        let (host, stream, symbol) = host_with_symbol();
        host.configure_cuda_symbol(symbol, DYNAMIC_SHARED_BYTES)
            .unwrap();
        host.begin_graph_capture(stream).unwrap();
        let lines = host.device().journal.borrow().len();
        assert_eq!(
            host.launch_cuda_symbol(stream, symbol, shared_launch(DYNAMIC_SHARED_BYTES + 1), &[]),
            Err(BundleStatusV1::HostError)
        );
        assert_eq!(host.device().journal.borrow().len(), lines);
        let graph = host.end_graph_capture(stream).unwrap();
        assert_eq!(host.device().graph_nodes.borrow()[&graph.slot()], 0);
    }

    #[test]
    fn dynamic_shared_memory_exactly_at_each_limit_passes() {
        let (host, stream, symbol) = host_with_symbol();
        host.launch_cuda_symbol(stream, symbol, shared_launch(48 * 1024), &[])
            .unwrap();
        for limit in [DYNAMIC_SHARED_BYTES, 48 * 1024 + 1, 0] {
            host.configure_cuda_symbol(symbol, limit).unwrap();
            host.launch_cuda_symbol(stream, symbol, shared_launch(limit), &[])
                .unwrap();
            assert_eq!(
                host.launch_cuda_symbol(stream, symbol, shared_launch(limit + 1), &[]),
                Err(BundleStatusV1::HostError)
            );
        }
    }

    /// A gather uploads the named rows of the package's mapped view, checked
    /// as the host checks them; a view the package does not map is not
    /// found.
    #[test]
    fn gathers_upload_the_rows_of_the_package_mapped_view() {
        let (host, stream, buffer) = host();
        let gather = |rank, view_id: &'static [u8], rows: &'static [u32]| MappedRowGatherV1 {
            image: BundleCompiledImageHandleV1::from_parts(1, 9),
            rank,
            view_id,
            row_bytes: 8,
            rows,
        };
        let span = DeviceBufferSpanV1::new(buffer, 8, 16);
        host.gather_mapped_rows(stream, span, gather(1, b"rows", &[3, 0]))
            .unwrap();
        let expected: Vec<u8> = (24..32).chain(0..8).collect();
        let note = format!("gather 1 rows 2 {span:?} {:016x}", fnv1a(&expected));
        assert!(host.device().journal.borrow().contains(&note));
        assert_eq!(
            host.gather_mapped_rows(stream, span, gather(0, b"rows", &[0, 1])),
            Err(BundleStatusV1::NotFound)
        );
        // A capture on the device refuses the gather, on the capturing
        // stream and on another stream of the device it has not joined.
        let context = BundleContextHandleV1::from_parts(1, 1);
        let other = host.create_stream(context, 0).unwrap();
        host.begin_graph_capture(stream).unwrap();
        for on in [stream, other] {
            assert_eq!(
                host.gather_mapped_rows(on, span, gather(1, b"rows", &[3, 0])),
                Err(BundleStatusV1::InvalidState)
            );
        }
        host.device().end_capture(stream, || 99).unwrap();
        for rows in [&[4_u32, 0][..], &[0], &[]] {
            let length = rows.len() as u64 * 8;
            assert_eq!(
                host.gather_mapped_rows(
                    stream,
                    DeviceBufferSpanV1::new(buffer, 0, length.max(16)),
                    gather(1, b"rows", rows)
                ),
                Err(BundleStatusV1::InvalidArgument)
            );
        }
    }

    #[test]
    fn the_backend_journals_device_calls_and_balances_handles() {
        let (host, stream, buffer) = host();
        assert_eq!(
            *host.1.borrow(),
            ["context 1", "pool 1", "stream 1", "buffer 2"]
        );
        assert_eq!(
            host.release_stream(stream),
            Err(BundleStatusV1::InvalidState)
        );
        let span = DeviceBufferSpanV1::new(buffer, 0, 8);
        host.copy_host_to_device(stream, span, &[1; 8]).unwrap();
        host.copy_device_to_device(stream, DeviceBufferSpanV1::new(buffer, 8, 8), span)
            .unwrap();
        let mut bytes = [0; 8];
        host.copy_device_to_host(stream, &mut bytes, span).unwrap();
        assert_eq!(bytes, [7; 8]);
        host.release_device_buffer(buffer).unwrap();
        host.release_stream(stream).unwrap();
        host.release_memory_pool(
            host.device()
                .memory_pool(BundleContextHandleV1::from_parts(1, 1)),
        )
        .unwrap();
        host.release_context(BundleContextHandleV1::from_parts(1, 1))
            .unwrap();
        let kinds = host
            .device()
            .journal
            .borrow()
            .iter()
            .map(|line| line.split(' ').next().unwrap().to_owned())
            .collect::<Vec<_>>();
        assert_eq!(
            kinds,
            [
                "stream", "allocate", "upload", "copy", "download", "release", "release",
                "release", "release"
            ]
        );
        assert_eq!(host.device().acquired.get(), 4);
        assert_eq!(host.device().released.get(), 4);
        assert!(host.1.borrow().is_empty());
        assert!(host.device().extents.borrow().is_empty());
    }

    #[test]
    fn a_refused_buffer_release_leaves_the_buffer_live() {
        let (host, stream, buffer) = host();
        let event = host
            .create_event(BundleContextHandleV1::from_parts(1, 1))
            .unwrap();
        let extents = host.device().extents.borrow().clone();
        let lines = host.device().journal.borrow().len();
        assert_eq!(
            host.release_device_buffer(buffer),
            Err(BundleStatusV1::InvalidState)
        );
        assert_eq!(*host.device().extents.borrow(), extents);
        assert_eq!(host.device().journal.borrow().len(), lines);
        assert_eq!(host.device().released.get(), 0);
        let span = DeviceBufferSpanV1::new(buffer, 0, 8);
        host.copy_host_to_device(stream, span, &[1; 8]).unwrap();
        host.release_event(event).unwrap();
        host.release_device_buffer(buffer).unwrap();
        assert!(!host.device().extents.borrow().contains_key(&buffer.slot()));
        assert_eq!(host.device().released.get(), 2);
    }

    #[test]
    fn a_failed_copy_reaches_no_journal() {
        let (host, stream, buffer) = host();
        let lines = host.device().journal.borrow().len();
        host.device().fail_copy.set(Some(0));
        let span = |offset| DeviceBufferSpanV1::new(buffer, offset, 8);
        assert_eq!(
            host.copy_device_to_device(stream, span(8), span(0)),
            Err(BundleStatusV1::HostError)
        );
        assert_eq!(host.device().journal.borrow().len(), lines);
        host.copy_device_to_device(stream, span(8), span(0))
            .unwrap();
    }

    #[test]
    fn a_refused_synchronization_reaches_no_journal() {
        let (host, stream, _) = host();
        let lines = host.device().journal.borrow().len();
        host.2.set(Some(0));
        assert_eq!(
            host.synchronize_stream(stream),
            Err(BundleStatusV1::HostError)
        );
        assert_eq!(host.device().journal.borrow().len(), lines);
        host.synchronize_stream(stream).unwrap();
        assert_eq!(
            host.device().journal.borrow()[lines..],
            [format!("synchronize {}", stream.slot())]
        );
    }

    /// A graph cache sharing a ledger releases only its own graphs, and only
    /// while they are the ledger's newest, as this host's release order
    /// requires; the other caches keep theirs. A refused release leaves the
    /// graph in the ledger for `release_all`.
    #[test]
    fn a_graph_cache_releases_its_own_graphs_only_while_they_are_newest() {
        use crate::geometry::Ranks;
        use crate::geometry::tests::Tp4;
        use crate::graphs::{GraphCache, GraphKey};
        let (host, stream, buffer) = host();
        let host = std::rc::Rc::new(host);
        let table = runtime_bundle_abi::HostServicesTableV1::new(std::rc::Rc::clone(&host));
        let services = table.caller();
        let key = |bucket| GraphKey {
            rank: 0,
            bucket,
            items: 0,
        };
        let capture = |cache: &mut GraphCache<Tp4>, bucket| {
            cache
                .capture(services, stream, key(bucket), 1, |_| {
                    Ok::<_, runtime_bundle_abi::BundleAbiError>(())
                })
                .unwrap()
        };
        let graphs = |host: &Backend<Empty>| {
            host.1
                .borrow()
                .iter()
                .filter(|handle| handle.starts_with("graph "))
                .count()
        };
        let mut decode = GraphCache::<Tp4>::new(Ranks::new(1).unwrap());
        let mut lower = GraphCache::sharing(&decode);
        let mut upper = GraphCache::sharing(&decode);
        capture(&mut lower, 1);
        capture(&mut decode, 2);
        capture(&mut upper, 1);
        assert_eq!((decode.ledger_len(), graphs(&host)), (3, 3));
        // The decode graph is newer than the lower cache's.
        assert!(!lower.release_if_newest(services).unwrap());
        assert!(lower.contains(key(1)));
        assert!(upper.release_if_newest(services).unwrap());
        assert!(upper.is_empty() && decode.contains(key(2)) && lower.contains(key(1)));
        assert_eq!((decode.ledger_len(), graphs(&host)), (2, 2));
        assert!(upper.release_if_newest(services).unwrap(), "holds nothing");
        // An event acquired after the graph makes the host refuse.
        capture(&mut upper, 1);
        let event = host
            .create_event(BundleContextHandleV1::from_parts(1, 1))
            .unwrap();
        assert!(upper.release_if_newest(services).is_err());
        assert!(upper.is_empty());
        assert_eq!((decode.ledger_len(), graphs(&host)), (3, 3));
        host.release_event(event).unwrap();
        decode.release_all(services).unwrap();
        assert!(lower.is_empty() && decode.is_empty());
        assert_eq!((decode.ledger_len(), graphs(&host)), (0, 0));
        host.release_device_buffer(buffer).unwrap();
    }

    #[test]
    #[should_panic(expected = "passes its buffer's 64 bytes")]
    fn an_upload_past_its_buffer_panics() {
        let (host, stream, buffer) = host();
        let _ = host.copy_host_to_device(stream, DeviceBufferSpanV1::new(buffer, 60, 8), &[0; 8]);
    }
}
