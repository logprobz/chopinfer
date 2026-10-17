//! Safe, Linux-only ownership layer over the CUDA Driver API.

#![deny(unsafe_code)]

#[cfg(all(not(target_os = "linux"), not(feature = "fake-driver-tests")))]
compile_error!("runtime-cuda supports only Linux x86_64 CUDA deployments");

mod ffi;
mod page_copy;
pub use page_copy::{PageCopyArena, PageCopyDirection, PageCopyPair};

mod upload_ring;

mod nvdec;

pub use nvdec::{AnnexBStream, DecodedFrame, NvdecError, NvdecSession};
pub use upload_ring::PinnedUploadRing;

use std::cell::Cell;
use std::collections::BTreeSet;
use std::error::Error;
use std::ffi::CString;
use std::fmt;
use std::rc::Rc;

#[cfg(target_os = "linux")]
pub use ffi::DEVICE_POINTER_ARRAY_SLOTS;

const REQUIRED_COMPUTE_MAJOR: i32 = 12;
const REQUIRED_COMPUTE_MINOR: i32 = 0;
const CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED: i32 = 704;
/// The alignment the tiled tensor-map encoding requires of a global address.
pub const TENSOR_MAP_ADDRESS_ALIGNMENT: u64 = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Executor {
    CudaSm120,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PeerAccessOwnership {
    Acquired,
    Preexisting,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceInfo {
    ordinal: i32,
    compute_major: i32,
    compute_minor: i32,
    multiprocessor_count: i32,
    memory_pools_supported: bool,
    cooperative_launch: bool,
}

impl DeviceInfo {
    #[must_use]
    pub const fn ordinal(&self) -> i32 {
        self.ordinal
    }

    #[must_use]
    pub const fn compute_major(&self) -> i32 {
        self.compute_major
    }

    #[must_use]
    pub const fn compute_minor(&self) -> i32 {
        self.compute_minor
    }

    #[must_use]
    pub const fn multiprocessor_count(&self) -> i32 {
        self.multiprocessor_count
    }

    #[must_use]
    pub const fn memory_pools_supported(&self) -> bool {
        self.memory_pools_supported
    }

    /// Whether the device accepts cooperative launches
    /// ([`Function::launch_cooperative`]).
    #[must_use]
    pub const fn cooperative_launch(&self) -> bool {
        self.cooperative_launch
    }

    #[must_use]
    pub const fn is_sm120(&self) -> bool {
        self.compute_major == REQUIRED_COMPUTE_MAJOR && self.compute_minor == REQUIRED_COMPUTE_MINOR
    }
}

#[derive(Debug)]
pub enum CudaError {
    Driver {
        operation: &'static str,
        code: i32,
        name: String,
        description: String,
    },
    UnsupportedDevice {
        ordinal: i32,
        compute_major: i32,
        compute_minor: i32,
    },
    MemoryPoolsUnsupported {
        ordinal: i32,
    },
    /// A cluster launch asked of a device whose architecture forms no cluster.
    ClusterLaunchUnsupported {
        ordinal: i32,
        compute_major: i32,
        compute_minor: i32,
    },
    ProgrammaticLaunchUnsupported {
        ordinal: i32,
        compute_major: i32,
        compute_minor: i32,
    },
    /// A cluster holding more blocks than the portable maximum.
    ClusterBlocksExceeded {
        blocks: u32,
        maximum: u32,
    },
    InvalidOrdinal(i32),
    InvalidKernelName,
    EventTimingDisabled,
    ModuleInUse,
    InvalidLaunch,
    BufferSize {
        buffer_bytes: usize,
        requested_bytes: usize,
    },
    BufferRange {
        buffer_bytes: usize,
        offset: usize,
        requested_bytes: usize,
    },
    DevicePointerOverflow,
    ContextMismatch,
    ResourceReleased(&'static str),
    ResourceShared(&'static str),
    /// A by-value address array with no slots, or more than the record holds.
    DevicePointerArraySlots(usize),
    TensorMapAddressMisaligned {
        address: u64,
        alignment: u64,
    },
    InvalidTensorMap(&'static str),
}

impl fmt::Display for CudaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver {
                operation,
                code,
                name,
                description,
            } => write!(
                formatter,
                "CUDA Driver API {operation} failed with {name} ({code}): {description}"
            ),
            Self::UnsupportedDevice {
                ordinal,
                compute_major,
                compute_minor,
            } => write!(
                formatter,
                "CUDA device {ordinal} has unsupported compute capability {compute_major}.{compute_minor}; SM120 is required"
            ),
            Self::ClusterLaunchUnsupported {
                ordinal,
                compute_major,
                compute_minor,
            } => write!(
                formatter,
                "CUDA device {ordinal} has compute capability {compute_major}.{compute_minor} and \
                 forms no thread-block cluster; {CLUSTER_LAUNCH_COMPUTE_MAJOR}.0 is required"
            ),
            Self::ClusterBlocksExceeded { blocks, maximum } => write!(
                formatter,
                "a thread-block cluster of {blocks} blocks is larger than the portable maximum \
                 of {maximum}"
            ),
            Self::ProgrammaticLaunchUnsupported {
                ordinal,
                compute_major,
                compute_minor,
            } => write!(
                formatter,
                "CUDA device {ordinal} has compute capability {compute_major}.{compute_minor}; programmatic stream serialization requires 9.0"
            ),
            Self::MemoryPoolsUnsupported { ordinal } => {
                write!(
                    formatter,
                    "CUDA device {ordinal} does not support memory pools"
                )
            }
            Self::InvalidOrdinal(ordinal) => {
                write!(formatter, "invalid CUDA device ordinal {ordinal}")
            }
            Self::ModuleInUse => formatter.write_str("module still owns resolved functions"),
            Self::EventTimingDisabled => {
                formatter.write_str("elapsed time requires timing-enabled events")
            }
            Self::InvalidKernelName => formatter.write_str("kernel symbol contains a NUL byte"),
            Self::InvalidLaunch => formatter.write_str("kernel launch dimensions must be nonzero"),
            Self::BufferSize {
                buffer_bytes,
                requested_bytes,
            } => write!(
                formatter,
                "buffer has {buffer_bytes} bytes but the operation requires {requested_bytes}"
            ),
            Self::BufferRange {
                buffer_bytes,
                offset,
                requested_bytes,
            } => write!(
                formatter,
                "buffer has {buffer_bytes} bytes but offset {offset} with length {requested_bytes} is outside it"
            ),
            Self::DevicePointerOverflow => {
                formatter.write_str("device pointer offset overflows the CUDA address space")
            }
            Self::ContextMismatch => {
                formatter.write_str("CUDA resources belong to different contexts")
            }
            Self::ResourceReleased(resource) => write!(formatter, "CUDA {resource} was released"),
            Self::ResourceShared(resource) => {
                write!(formatter, "CUDA {resource} cannot be released while shared")
            }
            Self::DevicePointerArraySlots(slots) => write!(
                formatter,
                "a by-value device address array carries {slots} slots, which is outside 1 \
                 through {}",
                ffi::DEVICE_POINTER_ARRAY_SLOTS
            ),
            Self::TensorMapAddressMisaligned { address, alignment } => write!(
                formatter,
                "tensor map global address {address:#x} is not aligned to {alignment} bytes"
            ),
            Self::InvalidTensorMap(reason) => write!(formatter, "invalid tensor map {reason}"),
        }
    }
}

impl Error for CudaError {}

fn checked(operation: &'static str, result: i32) -> Result<(), CudaError> {
    if result == ffi::CUDA_SUCCESS {
        Ok(())
    } else {
        let (name, description) = ffi::error_text(result);
        Err(CudaError::Driver {
            operation,
            code: result,
            name,
            description,
        })
    }
}

pub struct Driver;

impl Driver {
    pub fn initialize() -> Result<Self, CudaError> {
        checked("cuInit", ffi::init())?;
        Ok(Self)
    }

    #[must_use]
    pub const fn executor(&self) -> Executor {
        Executor::CudaSm120
    }

    pub fn version(&self) -> Result<i32, CudaError> {
        let (result, version) = ffi::driver_version();
        checked("cuDriverGetVersion", result)?;
        Ok(version)
    }

    /// Number of devices exposed by CUDA, after its visibility filtering.
    pub fn visible_device_count(&self) -> Result<usize, CudaError> {
        let (result, count) = ffi::device_count();
        checked("cuDeviceGetCount", result)?;
        usize::try_from(count).map_err(|_| CudaError::InvalidOrdinal(count))
    }

    /// Queries peer capability without retaining either device's context.
    pub fn devices_can_access_peer(&self, device: i32, peer: i32) -> Result<bool, CudaError> {
        let (result, access) = ffi::device_can_access_peer(device, peer);
        checked("cuDeviceCanAccessPeer", result)?;
        Ok(access != 0)
    }

    pub fn device_pci_bus_id(&self, ordinal: i32) -> Result<String, CudaError> {
        let (result, bus_id) = ffi::device_pci_bus_id(ordinal);
        checked("cuDeviceGetPCIBusId", result)?;
        Ok(bus_id)
    }

    pub fn devices(&self) -> Result<Vec<DeviceInfo>, CudaError> {
        (0..self.visible_device_count()?)
            .map(|ordinal| device_info(ordinal as i32))
            .collect()
    }

    pub fn device_uuid(&self, ordinal: i32) -> Result<String, CudaError> {
        let (result, bytes) = ffi::device_uuid(ordinal);
        checked("cuDeviceGetUuid_v2", result)?;
        let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
        Ok(format!(
            "GPU-{}-{}-{}-{}-{}",
            &hex[..8],
            &hex[8..12],
            &hex[12..16],
            &hex[16..20],
            &hex[20..]
        ))
    }

    pub fn retain_primary_context(&self, ordinal: i32) -> Result<Context, CudaError> {
        let info = device_info(ordinal)?;
        if !info.is_sm120() {
            return Err(CudaError::UnsupportedDevice {
                ordinal,
                compute_major: info.compute_major,
                compute_minor: info.compute_minor,
            });
        }
        if !info.memory_pools_supported {
            return Err(CudaError::MemoryPoolsUnsupported { ordinal });
        }
        let (result, handle) = ffi::primary_context_retain(ordinal);
        checked("cuDevicePrimaryCtxRetain", result)?;
        if let Err(error) = checked("cuCtxSetCurrent", ffi::context_set_current(handle)) {
            ffi::primary_context_release(ordinal);
            return Err(error);
        }
        Ok(Context {
            inner: Rc::new(ContextInner {
                handle,
                device: info,
            }),
        })
    }
}

fn device_info(ordinal: i32) -> Result<DeviceInfo, CudaError> {
    if ordinal < 0 {
        return Err(CudaError::InvalidOrdinal(ordinal));
    }
    let (result, device) = ffi::device_get(ordinal);
    checked("cuDeviceGet", result)?;
    let compute_major = device_attribute(device, ffi::DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
    let compute_minor = device_attribute(device, ffi::DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
    let multiprocessor_count =
        device_attribute(device, ffi::DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
    let memory_pools_supported =
        device_attribute(device, ffi::DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED)? != 0;
    let cooperative_launch =
        device_attribute(device, ffi::DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH)? != 0;
    Ok(DeviceInfo {
        ordinal,
        compute_major,
        compute_minor,
        multiprocessor_count,
        memory_pools_supported,
        cooperative_launch,
    })
}

fn device_attribute(device: i32, attribute: i32) -> Result<i32, CudaError> {
    let (result, value) = ffi::device_attribute(device, attribute);
    checked("cuDeviceGetAttribute", result)?;
    Ok(value)
}

struct ContextInner {
    handle: ffi::ContextHandle,
    device: DeviceInfo,
}

impl Drop for ContextInner {
    fn drop(&mut self) {
        ffi::context_set_current(ffi::null_context());
        ffi::primary_context_release(self.device.ordinal);
    }
}

#[derive(Clone)]
pub struct Context {
    inner: Rc<ContextInner>,
}

/// Two handles are the same context exactly when they retain the same one.
///
/// A device ordinal does not decide this: two retentions of one device are two
/// contexts, and memory allocated in one is not reachable from the other.
impl PartialEq for Context {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.inner, &other.inner)
    }
}

impl Eq for Context {}

impl Context {
    /// Returns the physical CUDA device ordinal retained by this context.
    #[must_use]
    pub fn device_ordinal(&self) -> i32 {
        self.inner.device.ordinal
    }

    #[must_use]
    pub fn device(&self) -> &DeviceInfo {
        &self.inner.device
    }

    pub fn set_current(&self) -> Result<(), CudaError> {
        set_current_inner(&self.inner)
    }

    /// SMs this context may use: its SM-count execution affinity, which an
    /// MPS active-thread limit or an SM partition lowers below the device's
    /// count.
    pub fn usable_multiprocessors(&self) -> Result<u32, CudaError> {
        set_current_inner(&self.inner)?;
        let (result, sms) = ffi::context_sm_count();
        checked("cuCtxGetExecAffinity", result)?;
        Ok(u32::try_from(sms).unwrap_or(0))
    }

    /// Waits for all previously submitted work in this context. This is the
    /// boundary for a result view that names device data but no producer stream.
    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.set_current()?;
        checked("cuCtxSynchronize", ffi::context_synchronize())
    }

    /// Observe free and total device memory in this context, in bytes.
    /// An observation describes one instant; it is not an allocation guarantee.
    pub fn memory_info(&self) -> Result<(u64, u64), CudaError> {
        self.set_current()?;
        let (result, free_bytes, total_bytes) = ffi::memory_info();
        checked("cuMemGetInfo", result)?;
        Ok((free_bytes as u64, total_bytes as u64))
    }

    pub fn stream(&self) -> Result<Stream, CudaError> {
        self.set_current()?;
        let (result, handle) = ffi::stream_create();
        checked("cuStreamCreate", result)?;
        Ok(Stream {
            inner: Rc::new(StreamInner {
                handle,
                context: Rc::clone(&self.inner),
                released: Cell::new(false),
            }),
        })
    }

    pub fn event(&self) -> Result<Event, CudaError> {
        self.create_event(false)
    }

    /// Create an event that records device timestamps.
    pub fn timed_event(&self) -> Result<Event, CudaError> {
        self.create_event(true)
    }

    fn create_event(&self, timing: bool) -> Result<Event, CudaError> {
        self.set_current()?;
        let (result, handle) = ffi::event_create(timing);
        checked("cuEventCreate", result)?;
        Ok(Event {
            timing,
            handle,
            context: Rc::clone(&self.inner),
            released: Cell::new(false),
        })
    }

    pub fn default_memory_pool(&self) -> Result<MemoryPool, CudaError> {
        self.set_current()?;
        let (result, handle) = ffi::default_memory_pool(self.inner.device.ordinal);
        checked("cuDeviceGetDefaultMemPool", result)?;
        Ok(MemoryPool {
            handle,
            context: Rc::clone(&self.inner),
        })
    }

    pub fn load_module(&self, cubin: &[u8]) -> Result<Module, CudaError> {
        self.set_current()?;
        let (result, handle) = ffi::module_load(cubin);
        checked("cuModuleLoadData", result)?;
        Ok(Module {
            inner: Rc::new(ModuleInner {
                released: Cell::new(false),
                handle,
                context: Rc::clone(&self.inner),
            }),
        })
    }

    pub fn can_access_peer(&self, peer: &Self) -> Result<bool, CudaError> {
        let (result, access) =
            ffi::device_can_access_peer(self.inner.device.ordinal, peer.inner.device.ordinal);
        checked("cuDeviceCanAccessPeer", result)?;
        Ok(access != 0)
    }

    pub fn enable_peer_access(&self, peer: &Self) -> Result<PeerAccessOwnership, CudaError> {
        self.set_current()?;
        let result = ffi::context_enable_peer_access(peer.inner.handle);
        if result == CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED {
            Ok(PeerAccessOwnership::Preexisting)
        } else {
            checked("cuCtxEnablePeerAccess", result)?;
            Ok(PeerAccessOwnership::Acquired)
        }
    }

    pub fn disable_peer_access(&self, peer: &Self) -> Result<(), CudaError> {
        self.set_current()?;
        checked(
            "cuCtxDisablePeerAccess",
            ffi::context_disable_peer_access(peer.inner.handle),
        )
    }
}

struct StreamInner {
    handle: ffi::StreamHandle,
    context: Rc<ContextInner>,
    released: Cell<bool>,
}

impl Drop for StreamInner {
    fn drop(&mut self) {
        if !self.released.get() {
            ffi::context_set_current(self.context.handle);
            ffi::stream_destroy(self.handle);
        }
    }
}

#[derive(Clone)]
pub struct Stream {
    inner: Rc<StreamInner>,
}

impl Stream {
    /// The context this stream was created in.
    #[must_use]
    pub fn context(&self) -> Context {
        Context {
            inner: Rc::clone(&self.inner.context),
        }
    }

    fn require_live(&self) -> Result<(), CudaError> {
        if self.inner.released.get() {
            Err(CudaError::ResourceReleased("stream"))
        } else {
            Ok(())
        }
    }

    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.require_live()?;
        set_current_inner(&self.inner.context)?;
        checked(
            "cuStreamSynchronize",
            ffi::stream_synchronize(self.inner.handle),
        )
    }

    pub fn begin_capture(&self) -> Result<(), CudaError> {
        self.require_live()?;
        set_current_inner(&self.inner.context)?;
        checked(
            "cuStreamBeginCapture",
            ffi::stream_begin_capture(self.inner.handle),
        )
    }

    pub fn wait_event(&self, event: &Event) -> Result<(), CudaError> {
        self.require_live()?;
        event.require_live()?;
        set_current_inner(&self.inner.context)?;
        checked(
            "cuStreamWaitEvent",
            ffi::stream_wait_event(self.inner.handle, event.handle),
        )
    }

    /// Lends this stream's raw driver handle for the length of one call.
    ///
    /// The handle is the driver's, and it stays this crate's to create,
    /// synchronize, and destroy. A caller that has to hand it to native code
    /// gets it here, inside a call whose context is already current, and never
    /// as a value it can keep: the borrow ends when `callback` returns.
    pub fn with_raw_handle<T>(
        &self,
        callback: impl FnOnce(*mut std::ffi::c_void) -> T,
    ) -> Result<T, CudaError> {
        self.require_live()?;
        set_current_inner(&self.inner.context)?;
        Ok(callback(self.inner.handle.cast()))
    }

    pub fn end_capture(&self) -> Result<Graph, CudaError> {
        self.require_live()?;
        set_current_inner(&self.inner.context)?;
        let (result, handle) = ffi::stream_end_capture(self.inner.handle);
        checked("cuStreamEndCapture", result)?;
        Ok(Graph {
            handle,
            context: Rc::clone(&self.inner.context),
        })
    }

    pub fn release(&self) -> Result<(), CudaError> {
        self.require_live()?;
        if Rc::strong_count(&self.inner) != 1 {
            return Err(CudaError::ResourceShared("stream"));
        }
        set_current_inner(&self.inner.context)?;
        checked("cuStreamDestroy", ffi::stream_destroy(self.inner.handle))?;
        self.inner.released.set(true);
        Ok(())
    }
}

pub struct Event {
    timing: bool,
    handle: ffi::EventHandle,
    context: Rc<ContextInner>,
    released: Cell<bool>,
}

impl Event {
    /// Return milliseconds between recorded events after the caller synchronizes.
    pub fn elapsed_ms(&self, end: &Self) -> Result<f32, CudaError> {
        self.require_live()?;
        end.require_live()?;
        require_same_context(&self.context, &end.context)?;
        if !self.timing || !end.timing {
            return Err(CudaError::EventTimingDisabled);
        }
        set_current_inner(&self.context)?;
        let (result, milliseconds) = ffi::event_elapsed(self.handle, end.handle);
        checked("cuEventElapsedTime", result)?;
        Ok(milliseconds)
    }

    fn require_live(&self) -> Result<(), CudaError> {
        if self.released.get() {
            Err(CudaError::ResourceReleased("event"))
        } else {
            Ok(())
        }
    }

    pub fn record(&self, stream: &Stream) -> Result<(), CudaError> {
        self.require_live()?;
        stream.require_live()?;
        require_same_context(&self.context, &stream.inner.context)?;
        set_current_inner(&self.context)?;
        checked(
            "cuEventRecord",
            ffi::event_record(self.handle, stream.inner.handle),
        )
    }

    /// Nonblocking completion query. A not-ready event is not a driver error.
    pub fn query(&self) -> Result<bool, CudaError> {
        self.require_live()?;
        set_current_inner(&self.context)?;
        match ffi::event_query(self.handle) {
            0 => Ok(true),
            600 => Ok(false),
            code => {
                checked("cuEventQuery", code)?;
                Ok(false)
            }
        }
    }

    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.require_live()?;
        set_current_inner(&self.context)?;
        checked("cuEventSynchronize", ffi::event_synchronize(self.handle))
    }

    pub fn release(&self) -> Result<(), CudaError> {
        self.require_live()?;
        set_current_inner(&self.context)?;
        checked("cuEventDestroy", ffi::event_destroy(self.handle))?;
        self.released.set(true);
        Ok(())
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if !self.released.get() {
            ffi::context_set_current(self.context.handle);
            ffi::event_destroy(self.handle);
        }
    }
}

pub struct MemoryPool {
    handle: ffi::MemoryPoolHandle,
    context: Rc<ContextInner>,
}

impl MemoryPool {
    pub fn set_release_threshold(&self, bytes: u64) -> Result<(), CudaError> {
        set_current_inner(&self.context)?;
        checked(
            "cuMemPoolSetAttribute",
            ffi::memory_pool_set_release_threshold(self.handle, bytes),
        )
    }

    pub fn set_peer_access(&self, device_ordinal: i32, read_write: bool) -> Result<(), CudaError> {
        set_current_inner(&self.context)?;
        checked(
            "cuMemPoolSetAccess",
            ffi::memory_pool_set_device_access(self.handle, device_ordinal, read_write),
        )
    }

    pub fn peer_access(&self, device_ordinal: i32) -> Result<bool, CudaError> {
        set_current_inner(&self.context)?;
        let (result, read_write) = ffi::memory_pool_get_device_access(self.handle, device_ordinal);
        checked("cuMemPoolGetAccess", result)?;
        Ok(read_write)
    }

    pub fn allocate(&self, stream: &Stream, bytes: usize) -> Result<DeviceBuffer, CudaError> {
        require_same_context(&self.context, &stream.inner.context)?;
        set_current_inner(&self.context)?;
        let (result, pointer) = ffi::memory_allocate(self.handle, stream.inner.handle, bytes);
        checked("cuMemAllocFromPoolAsync", result)?;
        Ok(DeviceBuffer {
            pointer,
            bytes,
            stream: Rc::clone(&stream.inner),
            allocation: Rc::new(()),
            allocation_state: Rc::new(Cell::new(0)),
            mapped_host: None,
        })
    }
}

/// The most page-locked memory one host-byte write stages at a time.
const HOST_BYTE_STAGING_BYTES: usize = 64 << 20;

#[derive(Clone)]
pub struct DeviceBuffer {
    pointer: ffi::DevicePointer,
    bytes: usize,
    stream: Rc<StreamInner>,
    allocation: Rc<()>,
    allocation_state: Rc<Cell<u8>>,
    /// Page-locked host bytes behind `pointer`, for a mapped host allocation.
    mapped_host: Option<Rc<ffi::MappedHostAllocation>>,
}

fn checked_host_range(bytes: usize, offset: usize, requested: usize) -> Result<(), CudaError> {
    if offset.checked_add(requested).is_none_or(|end| end > bytes) {
        Err(CudaError::BufferRange {
            buffer_bytes: bytes,
            offset,
            requested_bytes: requested,
        })
    } else {
        Ok(())
    }
}

/// CUDA page-locked host bytes. A pending transfer borrows this owner until
/// its stream has completed the copy.
pub struct PinnedHostBuffer {
    allocation: ffi::PinnedAllocation,
    context: Rc<ContextInner>,
}

impl PinnedHostBuffer {
    pub fn new(context: &Context, bytes: usize) -> Result<Self, CudaError> {
        if bytes == 0 {
            return Err(CudaError::BufferSize {
                buffer_bytes: 0,
                requested_bytes: 0,
            });
        }
        context.set_current()?;
        let (code, allocation) = ffi::PinnedAllocation::allocate(bytes);
        checked("cuMemHostAlloc", code)?;
        let allocation = allocation.ok_or(CudaError::InvalidLaunch)?;
        Ok(Self {
            allocation,
            context: Rc::clone(&context.inner),
        })
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.allocation.as_slice().len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        self.allocation.as_slice()
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.allocation.as_mut_slice()
    }
}

impl Drop for PinnedHostBuffer {
    fn drop(&mut self) {
        let _ = set_current_inner(&self.context);
    }
}

enum PinnedBorrow<'a> {
    Upload { _buffer: &'a PinnedHostBuffer },
    Download { _buffer: &'a mut PinnedHostBuffer },
}

/// A pending pinned transfer. Dropping it waits for the stream. If the driver
/// cannot establish completion, the process stops before borrowed memory can
/// be reused or freed while DMA may still own it.
pub struct PinnedTransfer<'a> {
    _host: PinnedBorrow<'a>,
    _device: DeviceBuffer,
    stream: Stream,
    completed: bool,
}

impl PinnedTransfer<'_> {
    pub fn finish(mut self) {
        if self.stream.synchronize().is_err() {
            std::process::abort();
        }
        self.completed = true;
    }
}

impl PinnedTransfer<'_> {
    /// Waits for the transfer and returns the driver's error instead of
    /// stopping the process. A failed fence retains the device allocation and
    /// the stream until process exit, and the caller must retain the pinned
    /// host buffer, because the DMA may still own both.
    fn fence(mut self) -> Result<(), CudaError> {
        let result = self.stream.synchronize();
        self.completed = true;
        if result.is_err() {
            eprintln!(
                "runtime-cuda: could not fence a host-byte copy; retaining pinned and device allocations until process exit"
            );
            std::mem::forget(self._device.clone());
            std::mem::forget(self.stream.clone());
        }
        result
    }
}

impl Drop for PinnedTransfer<'_> {
    fn drop(&mut self) {
        if !self.completed && self.stream.synchronize().is_err() {
            std::process::abort();
        }
    }
}

/// Keeps both allocations and the destination stream live until peer DMA
/// completes. Dropping the guard waits for completion.
pub struct PeerTransfer {
    _destination: DeviceBuffer,
    _source: DeviceBuffer,
    stream: Stream,
    completed: bool,
}

impl PeerTransfer {
    pub fn finish(mut self) {
        if self.stream.synchronize().is_err() {
            std::process::abort();
        }
        self.completed = true;
    }
}

impl Drop for PeerTransfer {
    fn drop(&mut self) {
        if !self.completed && self.stream.synchronize().is_err() {
            std::process::abort();
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// The element types carried by a tiled tensor map. FP8 encodes as UINT8.
pub enum TensorMapDataType {
    Uint8 = 1,
    Int32 = 2,
    Bfloat16 = 3,
    Float16 = 4,
    Float32 = 5,
    Fp8E4M3 = 6,
    Fp8E5M2 = 7,
    Uint16 = 8,
    U4Align8B = 9,
    U4Align16B = 10,
    U6Align16B = 11,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorMapSwizzle {
    None = 0,
    Bytes128 = 1,
    Bytes32 = 2,
    Bytes64 = 3,
}

/// A rank-1-to-5 tiled tensor map. Unused array entries must be zero.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TensorMapDescriptor {
    pub rank: u32,
    pub data_type: TensorMapDataType,
    pub swizzle: TensorMapSwizzle,
    pub interleave: u32,
    pub l2_promotion: u32,
    pub oob_fill: u32,
    pub global_dims: [u64; 5],
    pub global_strides_bytes: [u64; 4],
    pub box_dims: [u32; 5],
    pub element_strides: [u32; 5],
}

impl TensorMapDescriptor {
    fn raw(self) -> ffi::TensorMapDescriptor {
        ffi::TensorMapDescriptor {
            rank: self.rank,
            data_type: self.data_type as u32,
            swizzle: self.swizzle as u32,
            interleave: self.interleave,
            l2_promotion: self.l2_promotion,
            oob_fill: self.oob_fill,
            global_dims: self.global_dims,
            global_strides_bytes: self.global_strides_bytes,
            box_dims: self.box_dims,
            element_strides: self.element_strides,
        }
    }

    fn validate(self, byte_len: usize) -> Result<(), CudaError> {
        let rank = usize::try_from(self.rank).map_err(|_| CudaError::InvalidTensorMap("rank"))?;
        if !(1..=5).contains(&rank) {
            return Err(CudaError::InvalidTensorMap("rank"));
        }
        if self.interleave > 2 || self.l2_promotion > 3 || self.oob_fill > 1 {
            return Err(CudaError::InvalidTensorMap("enum"));
        }
        let packed_16 = matches!(
            self.data_type,
            TensorMapDataType::U4Align16B | TensorMapDataType::U6Align16B
        );
        if (self.interleave != 0 && rank < 3)
            || (self.interleave == 2 && self.swizzle != TensorMapSwizzle::Bytes32)
            || (self.data_type == TensorMapDataType::U6Align16B && self.interleave != 0)
            || (packed_16
                && !matches!(
                    self.swizzle,
                    TensorMapSwizzle::None | TensorMapSwizzle::Bytes128
                ))
            || (self.oob_fill != 0
                && !matches!(
                    self.data_type,
                    TensorMapDataType::Float16
                        | TensorMapDataType::Bfloat16
                        | TensorMapDataType::Float32
                ))
        {
            return Err(CudaError::InvalidTensorMap("layout"));
        }
        for index in 0..5 {
            if index < rank {
                if self.global_dims[index] == 0
                    || self.global_dims[index] > (1_u64 << 32)
                    || !(1..=256).contains(&self.box_dims[index])
                    || !(1..=8).contains(&self.element_strides[index])
                {
                    return Err(CudaError::InvalidTensorMap("dimension"));
                }
            } else if self.global_dims[index] != 0
                || self.box_dims[index] != 0
                || self.element_strides[index] != 0
            {
                return Err(CudaError::InvalidTensorMap("unused dimension"));
            }
        }
        if (packed_16 && (self.global_dims[0] % 128 != 0 || self.box_dims[0] != 128))
            || (self.data_type == TensorMapDataType::U4Align8B && self.global_dims[0] % 2 != 0)
        {
            return Err(CudaError::InvalidTensorMap("packed dimension"));
        }
        let (numerator, denominator): (u64, u64) = match self.data_type {
            TensorMapDataType::Uint8
            | TensorMapDataType::Fp8E4M3
            | TensorMapDataType::Fp8E5M2
            | TensorMapDataType::U4Align16B
            | TensorMapDataType::U6Align16B => (1, 1),
            TensorMapDataType::U4Align8B => (1, 2),
            TensorMapDataType::Uint16
            | TensorMapDataType::Float16
            | TensorMapDataType::Bfloat16 => (2, 1),
            TensorMapDataType::Int32 | TensorMapDataType::Float32 => (4, 1),
        };
        let inner_bytes = |elements: u64| -> Result<u64, CudaError> {
            elements
                .checked_mul(numerator)
                .and_then(|bytes| bytes.checked_add(denominator - 1))
                .map(|bytes| bytes / denominator)
                .ok_or(CudaError::DevicePointerOverflow)
        };
        let row_bytes = inner_bytes(self.global_dims[0])?;
        let box_bytes = inner_bytes(u64::from(self.box_dims[0]))?;
        let swizzle_bytes = match self.swizzle {
            TensorMapSwizzle::None => 0,
            TensorMapSwizzle::Bytes32 => 32,
            TensorMapSwizzle::Bytes64 => 64,
            TensorMapSwizzle::Bytes128 => 128,
        };
        if self.interleave == 0
            && (box_bytes % 16 != 0 || (swizzle_bytes != 0 && box_bytes > swizzle_bytes))
        {
            return Err(CudaError::InvalidTensorMap("box width"));
        }
        let alignment = if packed_16 || self.interleave == 2 {
            32
        } else {
            16
        };
        let mut footprint = row_bytes;
        for index in 0..4 {
            let stride = self.global_strides_bytes[index];
            if index < rank - 1 {
                if stride == 0 || stride >= (1_u64 << 40) || stride % alignment != 0 {
                    return Err(CudaError::InvalidTensorMap("stride"));
                }
                let previous_extent = if index == 0 {
                    row_bytes
                } else {
                    self.global_strides_bytes[index - 1]
                        .checked_mul(self.global_dims[index])
                        .ok_or(CudaError::DevicePointerOverflow)?
                };
                if stride < previous_extent {
                    return Err(CudaError::InvalidTensorMap("stride"));
                }
                footprint = footprint
                    .checked_add(
                        stride
                            .checked_mul(self.global_dims[index + 1] - 1)
                            .ok_or(CudaError::DevicePointerOverflow)?,
                    )
                    .ok_or(CudaError::DevicePointerOverflow)?;
            } else if stride != 0 {
                return Err(CudaError::InvalidTensorMap("unused stride"));
            }
        }
        if footprint > u64::try_from(byte_len).map_err(|_| CudaError::DevicePointerOverflow)? {
            return Err(CudaError::BufferRange {
                buffer_bytes: byte_len,
                offset: 0,
                requested_bytes: usize::try_from(footprint)
                    .map_err(|_| CudaError::DevicePointerOverflow)?,
            });
        }
        Ok(())
    }
}

pub struct TensorMap {
    raw: ffi::TensorMap,
    context: Rc<ContextInner>,
}

impl DeviceBuffer {
    /// Page-locked host memory every context maps, owned like a device
    /// allocation of `stream`'s context. Kernels read it over PCIe through
    /// the same kind of pointer a device allocation has.
    pub fn allocate_mapped_host(stream: &Stream, bytes: usize) -> Result<Self, CudaError> {
        if bytes == 0 {
            return Err(CudaError::BufferSize {
                buffer_bytes: 0,
                requested_bytes: 0,
            });
        }
        set_current_inner(&stream.inner.context)?;
        let (code, allocation) = ffi::MappedHostAllocation::allocate(bytes);
        checked("cuMemHostAlloc(DEVICEMAP)", code)?;
        let allocation = allocation.ok_or(CudaError::InvalidLaunch)?;
        Ok(Self {
            pointer: allocation.device_pointer(),
            bytes,
            stream: Rc::clone(&stream.inner),
            allocation: Rc::new(()),
            allocation_state: Rc::new(Cell::new(0)),
            mapped_host: Some(Rc::new(allocation)),
        })
    }

    /// Whether this allocation is mapped host memory.
    #[must_use]
    pub const fn is_mapped_host(&self) -> bool {
        self.mapped_host.is_some()
    }

    /// Where a mapped host allocation's host bytes start, for a caller that
    /// writes them while the device may also hold the allocation; they stay
    /// allocated until the allocation is released. `None` for device memory.
    #[must_use]
    pub fn mapped_host_address(&self) -> Option<usize> {
        self.mapped_host
            .as_ref()
            .map(|allocation| allocation.host_address())
    }

    /// Lends the host bytes of a mapped host allocation no clone shares yet,
    /// so they can be filled before any kernel reads them.
    pub fn mapped_host_bytes_mut(&mut self) -> Result<&mut [u8], CudaError> {
        self.require_live()?;
        let mapped = self.mapped_host.as_mut().ok_or(CudaError::InvalidLaunch)?;
        Rc::get_mut(mapped)
            .map(ffi::MappedHostAllocation::as_mut_slice)
            .ok_or(CudaError::ResourceShared("mapped host allocation"))
    }

    fn require_live(&self) -> Result<(), CudaError> {
        if self.allocation_state.get() == 2 {
            Err(CudaError::ResourceReleased("device allocation"))
        } else {
            Ok(())
        }
    }

    #[must_use]
    pub const fn byte_len(&self) -> usize {
        self.bytes
    }

    /// The stream this allocation was made on, and whose context owns it.
    ///
    /// A caller that has to stage bytes through an allocation it did not create
    /// needs the allocation's own stream, because a stream from another context
    /// cannot touch it.
    #[must_use]
    pub fn owner_stream(&self) -> Stream {
        Stream {
            inner: Rc::clone(&self.stream),
        }
    }

    /// The context that owns this allocation.
    ///
    /// A caller that has to encode a descriptor over an allocation it did not
    /// create needs that allocation's own context, because a descriptor built
    /// under another one names memory this context cannot reach.
    #[must_use]
    pub fn owner_context(&self) -> Context {
        Context {
            inner: Rc::clone(&self.stream.context),
        }
    }

    /// Whether this allocation's address satisfies one alignment.
    ///
    /// A pool allocation carries the driver's own alignment guarantee, which is
    /// not the one a caller asked for. This reports the address property
    /// itself, so a caller can fail closed instead of trusting a documented
    /// maximum.
    #[must_use]
    pub fn is_aligned_to(&self, alignment: u64) -> bool {
        alignment != 0 && alignment.is_power_of_two() && self.pointer.is_multiple_of(alignment)
    }

    pub fn copy_from_bytes(&self, source: &[u8], stream: &Stream) -> Result<(), CudaError> {
        self.copy_from_bytes_at(0, source, stream)
    }

    /// Writes host bytes in stream order and waits for them to land.
    ///
    /// The bytes are staged in page-locked memory and copied by DMA queued
    /// after the stream's earlier work, so the write can overtake neither
    /// that work nor a stream-ordered allocation the stream has not executed.
    /// A copy that reads its source at the API call gives no such order.
    pub fn copy_from_bytes_at(
        &self,
        offset: usize,
        source: &[u8],
        stream: &Stream,
    ) -> Result<(), CudaError> {
        self.require_live()?;
        self.pointer_at(offset, source.len())?;
        require_same_context(&self.stream.context, &stream.inner.context)?;
        if source.is_empty() {
            return Ok(());
        }
        let context = self.owner_context();
        let mut staging =
            PinnedHostBuffer::new(&context, source.len().min(HOST_BYTE_STAGING_BYTES))?;
        let chunk_bytes = staging.len();
        for (index, chunk) in source.chunks(chunk_bytes).enumerate() {
            staging.as_mut_slice()[..chunk.len()].copy_from_slice(chunk);
            let fenced = self
                .copy_from_pinned_at_async(
                    offset + index * chunk_bytes,
                    &staging,
                    0,
                    chunk.len(),
                    stream,
                )?
                .fence();
            if let Err(error) = fenced {
                // The DMA may still read the staging bytes; keep them.
                std::mem::forget(staging);
                return Err(error);
            }
        }
        Ok(())
    }

    /// Starts H2D DMA from pinned storage and retains both allocations until
    /// the returned guard synchronizes this stream.
    pub fn copy_from_pinned_at_async<'a>(
        &self,
        destination_offset: usize,
        source: &'a PinnedHostBuffer,
        source_offset: usize,
        byte_len: usize,
        stream: &Stream,
    ) -> Result<PinnedTransfer<'a>, CudaError> {
        self.require_live()?;
        let destination = self.pointer_at(destination_offset, byte_len)?;
        checked_host_range(source.len(), source_offset, byte_len)?;
        require_same_context(&self.stream.context, &source.context)?;
        require_same_context(&self.stream.context, &stream.inner.context)?;
        stream.require_live()?;
        set_current_inner(&self.stream.context)?;
        checked(
            "cuMemcpyHtoDAsync",
            ffi::copy_pinned_to_device(
                destination,
                &source.allocation,
                source_offset,
                byte_len,
                stream.inner.handle,
            ),
        )?;
        Ok(PinnedTransfer {
            _host: PinnedBorrow::Upload { _buffer: source },
            _device: self.clone(),
            stream: stream.clone(),
            completed: false,
        })
    }

    /// Fills one bounded range of this allocation with zero bytes.
    pub fn zero_at_async(
        &self,
        offset: usize,
        byte_len: usize,
        stream: &Stream,
    ) -> Result<(), CudaError> {
        self.require_live()?;
        let pointer = self.pointer_at(offset, byte_len)?;
        require_same_context(&self.stream.context, &stream.inner.context)?;
        set_current_inner(&self.stream.context)?;
        checked(
            "cuMemsetD8Async",
            ffi::memory_zero(pointer, byte_len, stream.inner.handle),
        )
    }

    /// Copies one bounded range of this allocation onto another of its ranges.
    pub fn copy_within_async(
        &self,
        source_offset: usize,
        destination_offset: usize,
        byte_len: usize,
        stream: &Stream,
    ) -> Result<(), CudaError> {
        self.require_live()?;
        let source = self.pointer_at(source_offset, byte_len)?;
        let destination = self.pointer_at(destination_offset, byte_len)?;
        require_same_context(&self.stream.context, &stream.inner.context)?;
        set_current_inner(&self.stream.context)?;
        checked(
            "cuMemcpyDtoDAsync",
            ffi::copy_device_to_device(destination, source, byte_len, stream.inner.handle),
        )
    }

    /// Reads one bounded range of this allocation into host bytes.
    pub fn copy_to_bytes_at_async(
        &self,
        offset: usize,
        destination: &mut [u8],
        stream: &Stream,
    ) -> Result<(), CudaError> {
        self.require_live()?;
        let pointer = self.pointer_at(offset, destination.len())?;
        require_same_context(&self.stream.context, &stream.inner.context)?;
        set_current_inner(&self.stream.context)?;
        checked(
            "cuMemcpyDtoHAsync",
            ffi::copy_device_to_host(destination, pointer, stream.inner.handle),
        )
    }

    /// Starts D2H DMA into pinned storage. The guard holds the only mutable
    /// borrow until the result is available on the host.
    pub fn copy_to_pinned_at_async<'a>(
        &self,
        source_offset: usize,
        destination: &'a mut PinnedHostBuffer,
        destination_offset: usize,
        byte_len: usize,
        stream: &Stream,
    ) -> Result<PinnedTransfer<'a>, CudaError> {
        self.require_live()?;
        let source = self.pointer_at(source_offset, byte_len)?;
        checked_host_range(destination.len(), destination_offset, byte_len)?;
        require_same_context(&self.stream.context, &destination.context)?;
        require_same_context(&self.stream.context, &stream.inner.context)?;
        stream.require_live()?;
        set_current_inner(&self.stream.context)?;
        checked(
            "cuMemcpyDtoHAsync",
            ffi::copy_device_to_pinned(
                &mut destination.allocation,
                destination_offset,
                source,
                byte_len,
                stream.inner.handle,
            ),
        )?;
        Ok(PinnedTransfer {
            _host: PinnedBorrow::Download {
                _buffer: destination,
            },
            _device: self.clone(),
            stream: stream.clone(),
            completed: false,
        })
    }

    /// Copies one bounded range of another allocation into a range of this one.
    ///
    /// The two allocations may belong to different contexts. Crossing contexts
    /// requires the caller to have enabled peer access first; without it the
    /// driver refuses the copy rather than reading the wrong memory.
    pub fn copy_range_from_async(
        &self,
        destination_offset: usize,
        source: &Self,
        source_offset: usize,
        byte_len: usize,
        stream: &Stream,
    ) -> Result<(), CudaError> {
        self.require_live()?;
        source.require_live()?;
        require_same_context(&self.stream.context, &source.stream.context)?;
        let destination_pointer = self.pointer_at(destination_offset, byte_len)?;
        let source_pointer = source.pointer_at(source_offset, byte_len)?;
        require_same_context(&self.stream.context, &stream.inner.context)?;
        set_current_inner(&self.stream.context)?;
        checked(
            "cuMemcpyDtoDAsync",
            ffi::copy_device_to_device(
                destination_pointer,
                source_pointer,
                byte_len,
                stream.inner.handle,
            ),
        )
    }

    /// Issues a byte-bounded peer copy between the allocations' actual CUDA
    /// contexts. The caller establishes peer access before this call.
    pub fn copy_peer_range_from_async(
        &self,
        destination_offset: usize,
        source: &Self,
        source_offset: usize,
        byte_len: usize,
        stream: &Stream,
    ) -> Result<PeerTransfer, CudaError> {
        self.require_live()?;
        source.require_live()?;
        let destination_pointer = self.pointer_at(destination_offset, byte_len)?;
        let source_pointer = source.pointer_at(source_offset, byte_len)?;
        require_same_context(&self.stream.context, &stream.inner.context)?;
        stream.require_live()?;
        set_current_inner(&self.stream.context)?;
        checked(
            "cuMemcpyPeerAsync",
            ffi::copy_peer_async(
                destination_pointer,
                self.stream.context.handle,
                source_pointer,
                source.stream.context.handle,
                byte_len,
                stream.inner.handle,
            ),
        )?;
        Ok(PeerTransfer {
            _destination: self.clone(),
            _source: source.clone(),
            stream: stream.clone(),
            completed: false,
        })
    }

    pub fn copy_to_bytes(&self, destination: &mut [u8], stream: &Stream) -> Result<(), CudaError> {
        self.validate_stream_and_size(stream, destination.len())?;
        set_current_inner(&self.stream.context)?;
        checked(
            "cuMemcpyDtoHAsync",
            ffi::copy_device_to_host(destination, self.pointer, stream.inner.handle),
        )?;
        stream.synchronize()
    }

    pub fn copy_from_f32(&self, source: &[f32], stream: &Stream) -> Result<(), CudaError> {
        self.validate_stream_and_size(stream, std::mem::size_of_val(source))?;
        set_current_inner(&self.stream.context)?;
        checked(
            "cuMemcpyHtoDAsync",
            ffi::copy_f32_host_to_device(self.pointer, source, stream.inner.handle),
        )?;
        stream.synchronize()
    }

    pub fn copy_to_f32(&self, destination: &mut [f32], stream: &Stream) -> Result<(), CudaError> {
        self.validate_stream_and_size(stream, std::mem::size_of_val(destination))?;
        set_current_inner(&self.stream.context)?;
        checked(
            "cuMemcpyDtoHAsync",
            ffi::copy_f32_device_to_host(destination, self.pointer, stream.inner.handle),
        )?;
        stream.synchronize()
    }

    fn validate_stream_and_size(&self, stream: &Stream, requested: usize) -> Result<(), CudaError> {
        self.require_live()?;
        require_same_context(&self.stream.context, &stream.inner.context)?;
        if requested > self.bytes {
            Err(CudaError::BufferSize {
                buffer_bytes: self.bytes,
                requested_bytes: requested,
            })
        } else {
            Ok(())
        }
    }

    /// The device address one bounds-checked range of this allocation starts at.
    ///
    /// A caller that has to hand a range to native code needs the address the
    /// driver placed it at, and this is the only way to obtain one. It answers
    /// for a live allocation and a range inside it, and nothing more: the
    /// allocation stays this crate's to release, and the address is a number
    /// whose meaning ends with the work its caller ordered.
    pub fn device_address_at(&self, offset: usize, byte_len: usize) -> Result<u64, CudaError> {
        self.pointer_at(offset, byte_len)
    }

    fn pointer_at(&self, offset: usize, requested: usize) -> Result<ffi::DevicePointer, CudaError> {
        self.require_live()?;
        let end = offset
            .checked_add(requested)
            .ok_or(CudaError::BufferRange {
                buffer_bytes: self.bytes,
                offset,
                requested_bytes: requested,
            })?;
        if end > self.bytes {
            return Err(CudaError::BufferRange {
                buffer_bytes: self.bytes,
                offset,
                requested_bytes: requested,
            });
        }
        let offset = u64::try_from(offset).map_err(|_| CudaError::DevicePointerOverflow)?;
        self.pointer
            .checked_add(offset)
            .ok_or(CudaError::DevicePointerOverflow)
    }

    pub fn release(&self) -> Result<(), CudaError> {
        self.require_live()?;
        if Rc::strong_count(&self.allocation) != 1 || Rc::strong_count(&self.allocation_state) != 1
        {
            return Err(CudaError::ResourceShared("device allocation"));
        }
        set_current_inner(&self.stream.context)?;
        if let Some(mapped) = &self.mapped_host {
            // Kernels on any stream may still read these bytes.
            checked("cuCtxSynchronize", ffi::context_synchronize())?;
            checked("cuMemFreeHost", mapped.free())?;
            self.allocation_state.set(2);
            return Ok(());
        }
        if self.allocation_state.get() == 0 {
            checked(
                "cuMemFreeAsync",
                ffi::memory_free(self.pointer, self.stream.handle),
            )?;
            self.allocation_state.set(1);
        }
        checked(
            "cuStreamSynchronize after cuMemFreeAsync",
            ffi::stream_synchronize(self.stream.handle),
        )?;
        self.allocation_state.set(2);
        Ok(())
    }
}

impl Context {
    /// Validates a map and its resolved range without encoding it.
    pub fn validate_tensor_map(
        &self,
        buffer: &DeviceBuffer,
        offset: usize,
        byte_len: usize,
        descriptor: TensorMapDescriptor,
    ) -> Result<(), CudaError> {
        require_same_context(&self.inner, &buffer.stream.context)?;
        let address = buffer.pointer_at(offset, byte_len)?;
        descriptor.validate(byte_len)?;
        let alignment = if descriptor.interleave == 2
            || matches!(
                descriptor.data_type,
                TensorMapDataType::U4Align16B | TensorMapDataType::U6Align16B
            ) {
            32
        } else {
            TENSOR_MAP_ADDRESS_ALIGNMENT
        };
        if !address.is_multiple_of(alignment) {
            return Err(CudaError::TensorMapAddressMisaligned { address, alignment });
        }
        checked(
            "runtime_cuda_validate_tensor_map",
            ffi::validate_tensor_map(
                &descriptor.raw(),
                address,
                u64::try_from(byte_len).map_err(|_| CudaError::DevicePointerOverflow)?,
            ),
        )
    }

    /// Encodes a rank-1-to-5 tiled map over a bounded allocation range.
    pub fn tensor_map(
        &self,
        buffer: &DeviceBuffer,
        offset: usize,
        byte_len: usize,
        descriptor: TensorMapDescriptor,
    ) -> Result<TensorMap, CudaError> {
        self.validate_tensor_map(buffer, offset, byte_len, descriptor)?;
        let address = buffer.pointer_at(offset, byte_len)?;
        self.set_current()?;
        let (result, raw) = ffi::tensor_map(
            &descriptor.raw(),
            address,
            u64::try_from(byte_len).map_err(|_| CudaError::DevicePointerOverflow)?,
        );
        checked("cuTensorMapEncodeTiled", result)?;
        Ok(TensorMap {
            raw,
            context: Rc::clone(&self.inner),
        })
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if let Some(mapped) = self.mapped_host.take() {
            if Rc::strong_count(&self.allocation) == 1 && self.allocation_state.get() == 0 {
                // Kernels in any stream of this context may still read the
                // pages, so they are freed only behind a fence that held.
                let fenced = ffi::context_set_current(self.stream.context.handle)
                    == ffi::CUDA_SUCCESS
                    && ffi::context_synchronize() == ffi::CUDA_SUCCESS;
                if !fenced {
                    eprintln!(
                        "runtime-cuda: could not fence a mapped host allocation of {} bytes; \
                         leaking it rather than freeing pages the device may still read",
                        self.bytes
                    );
                    std::mem::forget(mapped);
                }
            }
            return;
        }
        if Rc::strong_count(&self.allocation) == 1 && self.allocation_state.get() == 0 {
            ffi::context_set_current(self.stream.context.handle);
            ffi::memory_free(self.pointer, self.stream.handle);
        } else if Rc::strong_count(&self.allocation) == 1 && self.allocation_state.get() == 1 {
            ffi::context_set_current(self.stream.context.handle);
            ffi::stream_synchronize(self.stream.handle);
        }
    }
}

struct ModuleInner {
    released: Cell<bool>,
    handle: ffi::ModuleHandle,
    context: Rc<ContextInner>,
}

impl Drop for ModuleInner {
    fn drop(&mut self) {
        if !self.released.get() {
            ffi::context_set_current(self.context.handle);
            ffi::module_unload(self.handle);
        }
    }
}

pub struct Module {
    inner: Rc<ModuleInner>,
}

impl Module {
    /// Unload after the caller drains work and drops all resolved functions.
    /// A failed unload retains the handle for the caller's failure policy.
    pub fn release(&self) -> Result<(), CudaError> {
        if self.inner.released.get() {
            return Err(CudaError::ResourceReleased("module"));
        }
        if Rc::strong_count(&self.inner) != 1 {
            return Err(CudaError::ModuleInUse);
        }
        set_current_inner(&self.inner.context)?;
        checked("cuModuleUnload", ffi::module_unload(self.inner.handle))?;
        self.inner.released.set(true);
        Ok(())
    }

    /// The context this module was loaded into.
    #[must_use]
    pub fn context(&self) -> Context {
        Context {
            inner: Rc::clone(&self.inner.context),
        }
    }

    pub fn function(&self, name: &str) -> Result<Function, CudaError> {
        if self.inner.released.get() {
            return Err(CudaError::ResourceReleased("module"));
        }
        set_current_inner(&self.inner.context)?;
        let name = CString::new(name).map_err(|_| CudaError::InvalidKernelName)?;
        let (result, handle) = ffi::module_function(self.inner.handle, &name);
        checked("cuModuleGetFunction", result)?;
        Ok(Function {
            handle,
            owner: FunctionOwner::Module(Rc::clone(&self.inner)),
        })
    }
}

struct LibraryInner {
    handle: ffi::LibraryHandle,
    context: Rc<ContextInner>,
}

impl Drop for LibraryInner {
    fn drop(&mut self) {
        ffi::context_set_current(self.context.handle);
        ffi::library_unload(self.handle);
    }
}

pub struct Library {
    inner: Rc<LibraryInner>,
}

impl Library {
    pub fn function(&self, name: &str) -> Result<Function, CudaError> {
        set_current_inner(&self.inner.context)?;
        let name = CString::new(name).map_err(|_| CudaError::InvalidKernelName)?;
        let (kernel_result, kernel) = ffi::library_kernel(self.inner.handle, &name);
        checked("cuLibraryGetKernel", kernel_result)?;
        let (function_result, handle) = ffi::kernel_function(kernel);
        checked("cuKernelGetFunction", function_result)?;
        Ok(Function {
            handle,
            owner: FunctionOwner::Library(Rc::clone(&self.inner)),
        })
    }
}

enum FunctionOwner {
    Module(Rc<ModuleInner>),
    Library(Rc<LibraryInner>),
}

impl FunctionOwner {
    fn context(&self) -> &Rc<ContextInner> {
        match self {
            Self::Module(module) => &module.context,
            Self::Library(library) => &library.context,
        }
    }
}

pub struct Function {
    handle: ffi::FunctionHandle,
    owner: FunctionOwner,
}

/// One launch shape, with the cluster its blocks are grouped into.
///
/// `cluster` is blocks per cluster and [`LaunchConfig::NO_CLUSTER`] is the
/// default, which asks for no cluster and takes the ordinary launch path. Any
/// other shape divides the grid evenly and is launched through the cluster
/// attribute; this crate never drops a cluster it was given.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LaunchConfig {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared_memory_bytes: u32,
    pub cluster: (u32, u32, u32),
    pub programmatic_stream_serialization: bool,
}

/// The compute capability a device needs before it can form a cluster at all.
///
/// Thread-block clusters arrived with compute capability 9.0. The two numbers
/// this is compared against come from the same pinned device attributes every
/// other capability check reads, so nothing here invents a driver constant.
const CLUSTER_LAUNCH_COMPUTE_MAJOR: i32 = 9;

/// The blocks one thread-block cluster may hold.
///
/// Eight is the portable maximum: a cluster of that size launches on every
/// architecture that forms clusters at all, and a larger one needs a
/// non-portable opt-in this crate does not make. The bundle ABI states the same
/// number for its callers; this is the copy every caller of this crate is held
/// to, so a host, a probe, or a route cannot reach the driver with a cluster the
/// ABI would have refused. `runtime-host` is where both constants are visible at
/// once, and it asserts they agree.
pub const MAXIMUM_CLUSTER_BLOCKS: u32 = 8;

/// Whether one device's architecture forms thread-block clusters at all.
///
/// Separate from the launch that asks, so the rule can be checked against a
/// device this crate cannot currently retain. See
/// [`Function::require_cluster_capable`] for why it is kept.
fn cluster_capable(device: &DeviceInfo) -> Result<(), CudaError> {
    if device.compute_major >= CLUSTER_LAUNCH_COMPUTE_MAJOR {
        return Ok(());
    }
    Err(CudaError::ClusterLaunchUnsupported {
        ordinal: device.ordinal,
        compute_major: device.compute_major,
        compute_minor: device.compute_minor,
    })
}

fn programmatic_capable(device: &DeviceInfo) -> Result<(), CudaError> {
    if device.compute_major >= 9 {
        Ok(())
    } else {
        Err(CudaError::ProgrammaticLaunchUnsupported {
            ordinal: device.ordinal,
            compute_major: device.compute_major,
            compute_minor: device.compute_minor,
        })
    }
}

impl LaunchConfig {
    /// The cluster shape that asks for no cluster.
    pub const NO_CLUSTER: (u32, u32, u32) = (1, 1, 1);

    pub fn validate(self) -> Result<Self, CudaError> {
        let dimensions = [
            self.grid.0,
            self.grid.1,
            self.grid.2,
            self.block.0,
            self.block.1,
            self.block.2,
            self.cluster.0,
            self.cluster.1,
            self.cluster.2,
        ];
        if dimensions.contains(&0) {
            return Err(CudaError::InvalidLaunch);
        }
        // The portable bound, enforced here as well as at the ABI, so a caller
        // inside this process cannot reach the driver with a cluster the ABI
        // would have refused a bundle.
        let blocks = self
            .cluster
            .0
            .checked_mul(self.cluster.1)
            .and_then(|blocks| blocks.checked_mul(self.cluster.2))
            .ok_or(CudaError::InvalidLaunch)?;
        if blocks > MAXIMUM_CLUSTER_BLOCKS {
            return Err(CudaError::ClusterBlocksExceeded {
                blocks,
                maximum: MAXIMUM_CLUSTER_BLOCKS,
            });
        }
        // A partial cluster is not a launch CUDA forms, and this crate has no
        // way to make one it could: it refuses rather than adjusting the grid.
        for (extent, grid) in [
            (self.cluster.0, self.grid.0),
            (self.cluster.1, self.grid.1),
            (self.cluster.2, self.grid.2),
        ] {
            if !grid.is_multiple_of(extent) {
                return Err(CudaError::InvalidLaunch);
            }
        }
        Ok(self)
    }

    /// Whether this launch asks for a cluster at all.
    #[must_use]
    pub const fn wants_cluster(self) -> bool {
        !(self.cluster.0 == 1 && self.cluster.1 == 1 && self.cluster.2 == 1)
    }
}

/// Packs already resolved addresses into the by-value record a launch takes.
///
/// The caller states how many slots the kernel reads by how many addresses it
/// supplies, and this refuses more than the record holds rather than dropping
/// the ones past the end.
fn device_pointer_array(addresses: &[u64]) -> Result<ffi::DevicePointerArray, CudaError> {
    if addresses.is_empty() || addresses.len() > ffi::DEVICE_POINTER_ARRAY_SLOTS {
        return Err(CudaError::DevicePointerArraySlots(addresses.len()));
    }
    let mut array = ffi::DevicePointerArray::empty();
    array.addresses[..addresses.len()].copy_from_slice(addresses);
    Ok(array)
}

pub enum KernelArgument<'a> {
    Buffer(&'a DeviceBuffer),
    /// One bounded byte range of an allocation, with no ownership vocabulary.
    ///
    /// A caller that already bounds its own ranges passes them this way; the
    /// kernel receives the address of the range's first byte.
    Range {
        buffer: &'a DeviceBuffer,
        offset: usize,
        byte_len: usize,
    },
    NullBuffer,
    U32(u32),
    /// One fixed-width signed scalar, launched at exactly that width.
    I32(i32),
    U64(u64),
    TensorMap(&'a TensorMap),
    /// A by-value array of device addresses, in slot order.
    ///
    /// The addresses are already resolved: a caller that crosses contexts on
    /// purpose has established peer access for them, and this crate passes the
    /// bytes through rather than deciding what they mean.
    DevicePointerArray(&'a [u64]),
}

impl Function {
    pub fn launch(
        &self,
        stream: &Stream,
        config: LaunchConfig,
        arguments: &[KernelArgument<'_>],
    ) -> Result<(), CudaError> {
        self.launch_as(stream, config, arguments, false)
    }

    /// A cooperative launch: every block of the grid is resident at once, or
    /// the driver refuses the launch. The config's cluster, if any, is kept;
    /// programmatic stream serialization is refused, since its dependent
    /// would hold SMs this grid's blocks need.
    pub fn launch_cooperative(
        &self,
        stream: &Stream,
        config: LaunchConfig,
        arguments: &[KernelArgument<'_>],
    ) -> Result<(), CudaError> {
        self.launch_as(stream, config, arguments, true)
    }

    fn launch_as(
        &self,
        stream: &Stream,
        config: LaunchConfig,
        arguments: &[KernelArgument<'_>],
        cooperative: bool,
    ) -> Result<(), CudaError> {
        let config = config.validate()?;
        require_same_context(self.owner.context(), &stream.inner.context)?;
        if cooperative && config.programmatic_stream_serialization {
            return Err(CudaError::InvalidLaunch);
        }
        if config.programmatic_stream_serialization {
            programmatic_capable(&self.owner.context().device)?;
        }
        set_current_inner(self.owner.context())?;
        let mut stack = std::array::from_fn::<_, 16, _>(|_| ffi::Argument::U32(0));
        let mut heap = Vec::new();
        let values = if arguments.len() <= stack.len() {
            &mut stack[..arguments.len()]
        } else {
            heap.resize_with(arguments.len(), || ffi::Argument::U32(0));
            heap.as_mut_slice()
        };
        for (value, argument) in values.iter_mut().zip(arguments) {
            *value = match argument {
                KernelArgument::Buffer(buffer) => {
                    require_same_context(self.owner.context(), &buffer.stream.context)?;
                    ffi::Argument::Device(buffer.pointer)
                }
                KernelArgument::Range {
                    buffer,
                    offset,
                    byte_len,
                } => {
                    require_same_context(self.owner.context(), &buffer.stream.context)?;
                    ffi::Argument::Device(buffer.pointer_at(*offset, *byte_len)?)
                }
                KernelArgument::NullBuffer => ffi::Argument::Device(0),
                KernelArgument::U32(value) => ffi::Argument::U32(*value),
                KernelArgument::I32(value) => ffi::Argument::I32(*value),
                KernelArgument::U64(value) => ffi::Argument::U64(*value),
                KernelArgument::TensorMap(map) => {
                    require_same_context(self.owner.context(), &map.context)?;
                    ffi::Argument::TensorMap(map.raw)
                }
                KernelArgument::DevicePointerArray(addresses) => {
                    ffi::Argument::DevicePointerArray(device_pointer_array(addresses)?)
                }
            };
        }
        if cooperative {
            let cluster = if config.wants_cluster() {
                self.require_cluster_capable()?;
                Some(config.cluster)
            } else {
                None
            };
            return checked(
                "cuLaunchKernelEx",
                ffi::launch_kernel_cooperative(
                    self.handle,
                    stream.inner.handle,
                    config,
                    cluster,
                    values,
                ),
            );
        }
        if config.wants_cluster() {
            self.require_cluster_capable()?;
            return checked(
                "cuLaunchKernelEx",
                ffi::launch_kernel_cluster(
                    self.handle,
                    stream.inner.handle,
                    config,
                    config.programmatic_stream_serialization,
                    values,
                ),
            );
        }
        if config.programmatic_stream_serialization {
            return checked(
                "cuLaunchKernelEx",
                ffi::launch_kernel_pdl(self.handle, stream.inner.handle, config, values),
            );
        }
        checked(
            "cuLaunchKernel",
            ffi::launch_kernel(self.handle, stream.inner.handle, config, values),
        )
    }

    /// Refuses a cluster launch on a device that forms no cluster at all.
    ///
    /// This cannot fire today: [`Driver::retain_primary_context`] accepts only
    /// SM120, and 12.0 is past the 9.0 clusters arrived at, so every context
    /// this crate hands out is already cluster capable. It is kept as defence
    /// for the day that admission rule widens, which is the day a device that
    /// forms no cluster could reach here, and [`cluster_capable`] is where its
    /// logic lives so it can be checked without one. Removing it would leave the
    /// widening to be caught by a driver error at launch rather than by a
    /// refusal that names the device.
    ///
    /// It is a refusal either way. What neither does is launch the blocks
    /// without the cluster: a kernel that reads another block's shared memory or
    /// waits on a cluster barrier is a different program without one.
    fn require_cluster_capable(&self) -> Result<(), CudaError> {
        cluster_capable(&self.owner.context().device)
    }

    pub fn set_max_dynamic_shared_memory(&self, bytes: u32) -> Result<(), CudaError> {
        set_current_inner(self.owner.context())?;
        checked(
            "cuFuncSetAttribute",
            ffi::set_max_dynamic_shared(self.handle, bytes),
        )
    }

    /// Clusters of this function at `config`'s launch shape and cluster, as
    /// a cooperative launch when `cooperative`, that the device keeps
    /// resident at once (`cuOccupancyMaxActiveClusters` with the launch's
    /// own attributes), for a caller whose grids spin on each other and must
    /// all be resident. Programmatic stream serialization is not part of the
    /// question.
    pub fn max_active_clusters(
        &self,
        config: LaunchConfig,
        cooperative: bool,
    ) -> Result<u32, CudaError> {
        self.require_cluster_capable()?;
        set_current_inner(self.owner.context())?;
        let (result, clusters) = ffi::max_active_clusters(
            self.handle,
            config.grid,
            config.block,
            config.shared_memory_bytes,
            config.cluster,
            cooperative,
        );
        checked("cuOccupancyMaxActiveClusters", result)?;
        Ok(u32::try_from(clusters).unwrap_or(0))
    }

    /// The context this function runs in.
    #[must_use]
    pub fn context(&self) -> Context {
        Context {
            inner: Rc::clone(self.owner.context()),
        }
    }
}

pub struct Graph {
    handle: ffi::GraphHandle,
    context: Rc<ContextInner>,
}

impl Graph {
    pub fn instantiate(&self) -> Result<GraphExec, CudaError> {
        set_current_inner(&self.context)?;
        let (result, handle) = ffi::graph_instantiate(self.handle);
        checked("cuGraphInstantiateWithFlags", result)?;
        Ok(GraphExec {
            handle,
            context: Rc::clone(&self.context),
            released: Cell::new(false),
        })
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        ffi::context_set_current(self.context.handle);
        ffi::graph_destroy(self.handle);
    }
}

pub struct GraphExec {
    handle: ffi::GraphExecHandle,
    context: Rc<ContextInner>,
    released: Cell<bool>,
}

impl GraphExec {
    fn require_live(&self) -> Result<(), CudaError> {
        if self.released.get() {
            Err(CudaError::ResourceReleased("graph"))
        } else {
            Ok(())
        }
    }

    pub fn launch(&self, stream: &Stream) -> Result<(), CudaError> {
        self.require_live()?;
        require_same_context(&self.context, &stream.inner.context)?;
        set_current_inner(&self.context)?;
        checked(
            "cuGraphLaunch",
            ffi::graph_launch(self.handle, stream.inner.handle),
        )
    }

    /// Destroys the executable graph and reports the driver's answer. A
    /// failed destruction leaves the graph live, so the caller keeps owning
    /// it and may release it again.
    pub fn release(&self) -> Result<(), CudaError> {
        self.require_live()?;
        set_current_inner(&self.context)?;
        checked("cuGraphExecDestroy", ffi::graph_exec_destroy(self.handle))?;
        self.released.set(true);
        Ok(())
    }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        if !self.released.get() {
            ffi::context_set_current(self.context.handle);
            ffi::graph_exec_destroy(self.handle);
        }
    }
}

fn require_same_context(
    left: &Rc<ContextInner>,
    right: &Rc<ContextInner>,
) -> Result<(), CudaError> {
    if Rc::ptr_eq(left, right) {
        Ok(())
    } else {
        Err(CudaError::ContextMismatch)
    }
}

fn set_current_inner(context: &Rc<ContextInner>) -> Result<(), CudaError> {
    checked("cuCtxSetCurrent", ffi::context_set_current(context.handle))
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::mem;
    use std::ptr;
    use std::rc::Rc;

    use super::{
        Context, ContextInner, CudaError, DeviceBuffer, DeviceInfo, LaunchConfig,
        MAXIMUM_CLUSTER_BLOCKS, StreamInner, TensorMapDataType, TensorMapDescriptor,
        TensorMapSwizzle, ffi,
    };
    // Only the release-failure tests construct these two directly, and those run
    // against the fake driver, so without that feature the names are unused.
    #[cfg(feature = "fake-driver-tests")]
    use super::{Driver, Event, GraphExec, Stream};

    #[cfg(feature = "fake-driver-tests")]
    fn fake_context(ordinal: i32) -> Context {
        Context {
            inner: Rc::new(ContextInner {
                handle: ffi::null_context(),
                device: DeviceInfo {
                    ordinal,
                    compute_major: 12,
                    compute_minor: 0,
                    multiprocessor_count: 188,
                    memory_pools_supported: true,
                    cooperative_launch: true,
                },
            }),
        }
    }

    #[cfg(feature = "fake-driver-tests")]
    pub(crate) fn live_fake_context(ordinal: i32, handle: usize) -> Context {
        Context {
            inner: Rc::new(ContextInner {
                handle: handle as ffi::ContextHandle,
                device: DeviceInfo {
                    ordinal,
                    compute_major: 12,
                    compute_minor: 0,
                    multiprocessor_count: 188,
                    memory_pools_supported: true,
                    cooperative_launch: true,
                },
            }),
        }
    }

    #[cfg(feature = "fake-driver-tests")]
    #[test]
    fn timed_events_preserve_disabled_events_and_reject_context_mismatch() {
        let context = live_fake_context(0, 0x100);
        let other = live_fake_context(0, 0x200);
        let stream = context.stream().unwrap();
        let start = context.timed_event().unwrap();
        let end = context.timed_event().unwrap();
        assert!(start.elapsed_ms(&end).is_err());
        start.record(&stream).unwrap();
        end.record(&stream).unwrap();
        end.synchronize().unwrap();
        assert_eq!(start.elapsed_ms(&end).unwrap(), 2.0);
        let disabled = context.event().unwrap();
        disabled.record(&stream).unwrap();
        disabled.synchronize().unwrap();
        assert_ne!(ffi::event_elapsed(disabled.handle, end.handle).0, 0);
        assert!(matches!(
            disabled.elapsed_ms(&end),
            Err(CudaError::EventTimingDisabled)
        ));
        assert!(start.elapsed_ms(&other.timed_event().unwrap()).is_err());
        assert!(start.record(&other.stream().unwrap()).is_err());
        start.release().unwrap();
        assert!(start.elapsed_ms(&end).is_err());
    }

    #[cfg(feature = "fake-driver-tests")]
    #[test]
    fn memory_observations_follow_context_module_and_buffer_ownership() {
        ffi::fake_driver_reset();
        let driver = Driver::initialize().unwrap();
        let context = driver.retain_primary_context(0).unwrap();
        let (free_context, total) = context.memory_info().unwrap();
        assert_eq!(total, 8 << 30);
        assert_eq!(total - free_context, 512 << 20);
        let module = context.load_module(b"A").unwrap();
        let (free_module, _) = context.memory_info().unwrap();
        assert_eq!(free_context - free_module, 64 << 10);
        let stream = context.stream().unwrap();
        let buffer = context
            .default_memory_pool()
            .unwrap()
            .allocate(&stream, 4096)
            .unwrap();
        let (free_buffer, _) = context.memory_info().unwrap();
        assert_eq!(free_module - free_buffer, 4096);
        buffer.release().unwrap();
        module.release().unwrap();
        assert_eq!(context.memory_info().unwrap().0, free_context);
    }

    #[cfg(feature = "fake-driver-tests")]
    #[test]
    fn module_release_checks_function_owners_and_retains_failed_unloads() {
        ffi::fake_driver_reset();
        let context = live_fake_context(0, 0x100);
        let module = context.load_module(b"A").unwrap();
        let function = module.function("test").unwrap();
        assert!(matches!(module.release(), Err(CudaError::ModuleInUse)));
        assert_eq!(ffi::fake_driver_call_count(13), 0);
        drop(function);
        ffi::fake_driver_fail_next(13, 719);
        assert!(module.release().is_err());
        module.release().unwrap();
        assert!(module.function("test").is_err());
        drop(module);
        assert_eq!(ffi::fake_driver_call_count(13), 2);
    }

    #[cfg(feature = "fake-driver-tests")]
    #[test]
    fn local_copy_rejects_a_source_from_another_context_before_dispatch() {
        ffi::fake_driver_reset();
        let context = live_fake_context(0, 0x100);
        let other = live_fake_context(0, 0x200);
        let stream = context.stream().unwrap();
        let other_stream = other.stream().unwrap();
        let destination = context
            .default_memory_pool()
            .unwrap()
            .allocate(&stream, 8)
            .unwrap();
        let source = other
            .default_memory_pool()
            .unwrap()
            .allocate(&other_stream, 8)
            .unwrap();
        assert!(
            destination
                .copy_range_from_async(0, &source, 0, 8, &stream)
                .is_err()
        );
        assert_eq!(ffi::fake_driver_call_count(9), 0);
    }

    /// A stream-ordered allocation is usable from its own stream at once, and
    /// from any other stream or out of stream order only after its stream has
    /// executed it. The stub counts every write that breaks that rule.
    #[cfg(feature = "fake-driver-tests")]
    #[test]
    fn the_fake_driver_counts_writes_before_an_allocation_is_ordered() {
        use super::PinnedHostBuffer;
        ffi::fake_driver_reset();
        let driver = Driver::initialize().unwrap();
        let context = driver.retain_primary_context(0).unwrap();
        let owner = context.stream().unwrap();
        let other = context.stream().unwrap();
        let pool = context.default_memory_pool().unwrap();
        let staged = PinnedHostBuffer::new(&context, 8).unwrap();
        let same_stream = pool.allocate(&owner, 8).unwrap();
        same_stream
            .copy_from_pinned_at_async(0, &staged, 0, 8, &owner)
            .unwrap()
            .finish();
        assert_eq!(crate::fake_driver::ordering_violations(), 0);
        let cross_stream = pool.allocate(&owner, 8).unwrap();
        cross_stream
            .copy_from_pinned_at_async(0, &staged, 0, 8, &other)
            .unwrap()
            .finish();
        assert_eq!(crate::fake_driver::ordering_violations(), 1);
        let ordered = pool.allocate(&owner, 8).unwrap();
        owner.synchronize().unwrap();
        ordered
            .copy_from_pinned_at_async(0, &staged, 0, 8, &other)
            .unwrap()
            .finish();
        assert_eq!(crate::fake_driver::ordering_violations(), 1);
    }

    /// A copy that lands at the API call is ordered against no stream, so it
    /// may not reach an allocation its own stream has not executed either.
    #[cfg(feature = "fake-driver-tests")]
    #[test]
    fn an_api_call_copy_into_a_pending_allocation_counts_on_any_stream() {
        use super::PinnedHostBuffer;
        ffi::fake_driver_reset();
        let driver = Driver::initialize().unwrap();
        let context = driver.retain_primary_context(0).unwrap();
        let stream = context.stream().unwrap();
        let pool = context.default_memory_pool().unwrap();
        let staged = PinnedHostBuffer::new(&context, 8).unwrap();
        crate::fake_driver::set_api_call_copies(true);
        let pending = pool.allocate(&stream, 8).unwrap();
        pending
            .copy_from_pinned_at_async(0, &staged, 0, 8, &stream)
            .unwrap()
            .finish();
        assert_eq!(crate::fake_driver::ordering_violations(), 1);
        let pending = pool.allocate(&stream, 8).unwrap();
        pending.copy_from_bytes(&[1; 8], &stream).unwrap();
        assert_eq!(crate::fake_driver::ordering_violations(), 2);
        let ordered = pool.allocate(&stream, 8).unwrap();
        stream.synchronize().unwrap();
        ordered.copy_from_bytes(&[1; 8], &stream).unwrap();
        assert_eq!(crate::fake_driver::ordering_violations(), 2);
        crate::fake_driver::set_api_call_copies(false);
        let queued = pool.allocate(&stream, 8).unwrap();
        queued.copy_from_bytes(&[1; 8], &stream).unwrap();
        assert_eq!(crate::fake_driver::ordering_violations(), 2);
    }

    /// Writing host bytes stays in stream order: after the work already queued
    /// on the stream, never at the API call.
    #[cfg(feature = "fake-driver-tests")]
    #[test]
    fn host_byte_writes_land_in_stream_order() {
        ffi::fake_driver_reset();
        let driver = Driver::initialize().unwrap();
        let context = driver.retain_primary_context(0).unwrap();
        let stream = context.stream().unwrap();
        let buffer = context
            .default_memory_pool()
            .unwrap()
            .allocate(&stream, 16)
            .unwrap();
        buffer.copy_from_bytes(&[3; 16], &stream).unwrap();
        buffer
            .copy_from_bytes_at(4, &[1, 2, 3, 4], &stream)
            .unwrap();
        let mut output = [0_u8; 16];
        buffer.copy_to_bytes(&mut output, &stream).unwrap();
        assert_eq!(output, [3, 3, 3, 3, 1, 2, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3]);
        assert_eq!(crate::fake_driver::ordering_violations(), 0);
        assert!(buffer.copy_from_bytes_at(14, &[0; 4], &stream).is_err());
        buffer.copy_from_bytes_at(16, &[], &stream).unwrap();
    }

    /// A host-byte write whose fence fails returns the driver's error and
    /// keeps every allocation the DMA may still own.
    #[cfg(feature = "fake-driver-tests")]
    #[test]
    fn a_failed_host_byte_fence_returns_the_error_and_retains_its_owners() {
        const STREAM_SYNCHRONIZE: i32 = 5;
        const HOST_FREE: i32 = 17;
        ffi::fake_driver_reset();
        let driver = Driver::initialize().unwrap();
        let context = driver.retain_primary_context(0).unwrap();
        let stream = context.stream().unwrap();
        let buffer = context
            .default_memory_pool()
            .unwrap()
            .allocate(&stream, 16)
            .unwrap();
        ffi::fake_driver_fail_next(STREAM_SYNCHRONIZE, 719);
        assert!(matches!(
            buffer.copy_from_bytes_at(0, &[1; 16], &stream),
            Err(CudaError::Driver { .. })
        ));
        assert_eq!(ffi::fake_driver_call_count(HOST_FREE), 0);
        assert!(matches!(
            buffer.release(),
            Err(CudaError::ResourceShared(_))
        ));
        buffer.copy_from_bytes_at(0, &[2; 16], &stream).unwrap();
        assert_eq!(ffi::fake_driver_call_count(HOST_FREE), 1);
    }

    #[cfg(feature = "fake-driver-tests")]
    #[test]
    fn pinned_transfers_keep_bytes_live_and_check_both_ranges() {
        use super::PinnedHostBuffer;
        ffi::fake_driver_reset();
        let driver = Driver::initialize().unwrap();
        let context = driver.retain_primary_context(0).unwrap();
        let stream = context.stream().unwrap();
        let device = context
            .default_memory_pool()
            .unwrap()
            .allocate(&stream, 16)
            .unwrap();
        let mut input = PinnedHostBuffer::new(&context, 16).unwrap();
        input.as_mut_slice()[..4].copy_from_slice(&[1, 2, 3, 4]);
        let mut output = PinnedHostBuffer::new(&context, 16).unwrap();
        device
            .copy_from_pinned_at_async(4, &input, 0, 4, &stream)
            .unwrap()
            .finish();
        device
            .copy_to_pinned_at_async(4, &mut output, 8, 4, &stream)
            .unwrap()
            .finish();
        assert_eq!(&output.as_slice()[8..12], &[1, 2, 3, 4]);
        assert_eq!(ffi::fake_driver_call_count(16), 2);
        assert_eq!(ffi::fake_driver_call_count(8), 1);
        assert_eq!(ffi::fake_driver_call_count(10), 1);
        assert!(
            device
                .copy_from_pinned_at_async(14, &input, 0, 4, &stream)
                .is_err()
        );
        assert!(
            device
                .copy_to_pinned_at_async(4, &mut output, 14, 4, &stream)
                .is_err()
        );
        assert_eq!(ffi::fake_driver_call_count(8), 1);
        assert_eq!(ffi::fake_driver_call_count(10), 1);
    }

    #[cfg(feature = "fake-driver-tests")]
    #[test]
    fn peer_copy_uses_contexts_and_rejects_out_of_range_bytes() {
        use super::PinnedHostBuffer;
        ffi::fake_driver_reset();
        let driver = Driver::initialize().unwrap();
        let destination_context = driver.retain_primary_context(0).unwrap();
        let source_context = driver.retain_primary_context(1).unwrap();
        let destination_stream = destination_context.stream().unwrap();
        let source_stream = source_context.stream().unwrap();
        let destination = destination_context
            .default_memory_pool()
            .unwrap()
            .allocate(&destination_stream, 16)
            .unwrap();
        let source = source_context
            .default_memory_pool()
            .unwrap()
            .allocate(&source_stream, 16)
            .unwrap();
        let mut input = PinnedHostBuffer::new(&source_context, 16).unwrap();
        input.as_mut_slice()[2..6].copy_from_slice(&[5, 6, 7, 8]);
        source
            .copy_from_pinned_at_async(3, &input, 2, 4, &source_stream)
            .unwrap()
            .finish();
        destination
            .copy_peer_range_from_async(5, &source, 3, 4, &destination_stream)
            .unwrap()
            .finish();
        let mut output = PinnedHostBuffer::new(&destination_context, 16).unwrap();
        destination
            .copy_to_pinned_at_async(5, &mut output, 0, 4, &destination_stream)
            .unwrap()
            .finish();
        assert_eq!(&output.as_slice()[..4], &[5, 6, 7, 8]);
        assert_eq!(ffi::fake_driver_call_count(18), 1);
        assert!(
            destination
                .copy_peer_range_from_async(14, &source, 3, 4, &destination_stream)
                .is_err()
        );
        assert!(
            destination
                .copy_peer_range_from_async(5, &source, 14, 4, &destination_stream)
                .is_err()
        );
        assert_eq!(ffi::fake_driver_call_count(18), 1);
    }

    #[cfg(feature = "fake-driver-tests")]
    #[test]
    fn explicit_event_and_stream_release_failures_retain_retryable_handles() {
        const CONTEXT_SET: i32 = 1;
        const EVENT_DESTROY: i32 = 2;
        const STREAM_DESTROY: i32 = 3;
        let context = live_fake_context(0, 0x100);

        ffi::fake_driver_reset();
        let event = Event {
            timing: false,
            handle: 0x200_usize as ffi::EventHandle,
            context: Rc::clone(&context.inner),
            released: Cell::new(false),
        };
        ffi::fake_driver_fail_next(CONTEXT_SET, 201);
        assert!(event.release().is_err());
        assert!(!event.released.get());
        event.release().unwrap();
        assert!(event.released.get());

        ffi::fake_driver_reset();
        let event = Event {
            timing: false,
            handle: 0x201_usize as ffi::EventHandle,
            context: Rc::clone(&context.inner),
            released: Cell::new(false),
        };
        ffi::fake_driver_fail_next(EVENT_DESTROY, 719);
        assert!(event.release().is_err());
        assert_eq!(ffi::fake_driver_call_count(EVENT_DESTROY), 1);
        event.release().unwrap();
        assert_eq!(ffi::fake_driver_call_count(EVENT_DESTROY), 2);

        ffi::fake_driver_reset();
        let stream = Stream {
            inner: Rc::new(StreamInner {
                handle: 0x300_usize as ffi::StreamHandle,
                context: Rc::clone(&context.inner),
                released: Cell::new(false),
            }),
        };
        ffi::fake_driver_fail_next(STREAM_DESTROY, 719);
        assert!(stream.release().is_err());
        assert!(!stream.inner.released.get());
        stream.release().unwrap();
        assert!(stream.inner.released.get());
    }

    #[cfg(feature = "fake-driver-tests")]
    #[test]
    fn a_failed_graph_release_keeps_the_graph_live_for_a_retry() {
        const GRAPH_EXEC_DESTROY: i32 = 33;
        let context = live_fake_context(0, 0x100);
        ffi::fake_driver_reset();
        let graph = GraphExec {
            handle: 0x501_usize as ffi::GraphExecHandle,
            context: Rc::clone(&context.inner),
            released: Cell::new(false),
        };
        ffi::fake_driver_fail_next(GRAPH_EXEC_DESTROY, 719);
        assert!(matches!(
            graph.release(),
            Err(CudaError::Driver {
                operation: "cuGraphExecDestroy",
                ..
            })
        ));
        assert!(!graph.released.get());
        graph.release().unwrap();
        assert_eq!(ffi::fake_driver_call_count(GRAPH_EXEC_DESTROY), 2);
        // A released graph is neither released again nor launched, and
        // dropping it does not destroy it a second time.
        assert!(matches!(
            graph.release(),
            Err(CudaError::ResourceReleased("graph"))
        ));
        drop(graph);
        assert_eq!(ffi::fake_driver_call_count(GRAPH_EXEC_DESTROY), 2);
    }

    #[cfg(feature = "fake-driver-tests")]
    #[test]
    fn mapped_host_buffers_free_only_after_a_successful_fence() {
        const CONTEXT_SET: i32 = 1;
        const HOST_FREE: i32 = 17;
        const CONTEXT_SYNCHRONIZE: i32 = 19;
        let context = live_fake_context(0, 0x100);
        ffi::fake_driver_reset();
        let stream = context.stream().unwrap();

        // A failed explicit release frees nothing, and a retry frees once.
        let buffer = DeviceBuffer::allocate_mapped_host(&stream, 64).unwrap();
        ffi::fake_driver_fail_next(CONTEXT_SYNCHRONIZE, 719);
        assert!(buffer.release().is_err());
        ffi::fake_driver_fail_next(CONTEXT_SET, 201);
        assert!(buffer.release().is_err());
        assert_eq!(ffi::fake_driver_call_count(HOST_FREE), 0);
        buffer.release().unwrap();
        drop(buffer);
        assert_eq!(ffi::fake_driver_call_count(HOST_FREE), 1);

        // A drop whose fence fails keeps the pages rather than freeing them
        // under readers that may still run.
        for operation in [CONTEXT_SYNCHRONIZE, CONTEXT_SET] {
            let buffer = DeviceBuffer::allocate_mapped_host(&stream, 64).unwrap();
            ffi::fake_driver_fail_next(operation, 719);
            drop(buffer);
            assert_eq!(ffi::fake_driver_call_count(HOST_FREE), 1);
        }

        // A drop whose fence succeeds frees exactly once.
        let buffer = DeviceBuffer::allocate_mapped_host(&stream, 64).unwrap();
        let clone = buffer.clone();
        drop(buffer);
        assert_eq!(ffi::fake_driver_call_count(HOST_FREE), 1);
        drop(clone);
        assert_eq!(ffi::fake_driver_call_count(HOST_FREE), 2);
    }

    #[cfg(feature = "fake-driver-tests")]
    #[test]
    fn free_async_and_completion_failures_retry_without_duplicate_free() {
        const MEMORY_FREE: i32 = 4;
        const STREAM_SYNCHRONIZE: i32 = 5;
        let context = live_fake_context(0, 0x400);

        let buffer = |pointer| DeviceBuffer {
            pointer,
            bytes: 4096,
            stream: Rc::new(StreamInner {
                handle: 0x500_usize as ffi::StreamHandle,
                context: Rc::clone(&context.inner),
                released: Cell::new(false),
            }),
            allocation: Rc::new(()),
            allocation_state: Rc::new(Cell::new(0)),
            mapped_host: None,
        };

        ffi::fake_driver_reset();
        let allocation = buffer(0x600);
        ffi::fake_driver_fail_next(MEMORY_FREE, 719);
        assert!(allocation.release().is_err());
        assert_eq!(allocation.allocation_state.get(), 0);
        allocation.release().unwrap();
        assert_eq!(ffi::fake_driver_call_count(MEMORY_FREE), 2);
        assert_eq!(allocation.allocation_state.get(), 2);

        ffi::fake_driver_reset();
        let allocation = buffer(0x700);
        ffi::fake_driver_fail_next(STREAM_SYNCHRONIZE, 719);
        assert!(allocation.release().is_err());
        assert_eq!(allocation.allocation_state.get(), 1);
        assert_eq!(ffi::fake_driver_call_count(MEMORY_FREE), 1);
        allocation.release().unwrap();
        assert_eq!(ffi::fake_driver_call_count(MEMORY_FREE), 1);
        assert_eq!(ffi::fake_driver_call_count(STREAM_SYNCHRONIZE), 2);
        assert_eq!(allocation.allocation_state.get(), 2);
    }

    /// One context and one allocation inside it, at a chosen base address.
    ///
    /// Neither may be dropped: these records name no driver object, so their
    /// destructors would free memory that was never allocated. Every caller
    /// forgets both.
    fn fake_context_and_buffer(
        pointer: ffi::DevicePointer,
        bytes: usize,
    ) -> (Context, DeviceBuffer) {
        let context = Rc::new(ContextInner {
            handle: ffi::null_context(),
            device: DeviceInfo {
                ordinal: 0,
                compute_major: 12,
                compute_minor: 0,
                multiprocessor_count: 188,
                memory_pools_supported: true,
                cooperative_launch: true,
            },
        });
        let buffer = DeviceBuffer {
            pointer,
            bytes,
            stream: Rc::new(StreamInner {
                handle: ptr::null_mut(),
                context: Rc::clone(&context),
                released: Cell::new(false),
            }),
            allocation: Rc::new(()),
            allocation_state: Rc::new(Cell::new(0)),
            mapped_host: None,
        };
        (Context { inner: context }, buffer)
    }

    fn tensor_map_descriptor(rank: u32) -> TensorMapDescriptor {
        let mut descriptor = TensorMapDescriptor {
            rank,
            data_type: TensorMapDataType::Uint8,
            swizzle: TensorMapSwizzle::Bytes128,
            interleave: 0,
            l2_promotion: 3,
            oob_fill: 0,
            global_dims: [0; 5],
            global_strides_bytes: [0; 4],
            box_dims: [0; 5],
            element_strides: [0; 5],
        };
        for index in 0..rank as usize {
            descriptor.global_dims[index] = if index == 0 { 1024 } else { 2 };
            descriptor.box_dims[index] = if index == 0 { 128 } else { 1 };
            descriptor.element_strides[index] = 1;
            if index > 0 {
                descriptor.global_strides_bytes[index - 1] = 1024 << (index - 1);
            }
        }
        descriptor
    }

    /// The stub's ordinal for the tiled tensor-map encoding.
    #[cfg(feature = "fake-driver-tests")]
    const FAKE_ENCODE_TENSOR_MAP: i32 = 6;

    #[test]
    fn tensor_maps_of_every_rank_validate_before_encoding() {
        let (context, buffer) = fake_context_and_buffer(0x1_0000, 65_536);
        for rank in 1..=5 {
            let descriptor = tensor_map_descriptor(rank);
            context
                .validate_tensor_map(&buffer, 0, 32_768, descriptor)
                .unwrap();
            #[cfg(feature = "fake-driver-tests")]
            context.tensor_map(&buffer, 0, 32_768, descriptor).unwrap();
        }
        for rank in [0, 6] {
            let descriptor = tensor_map_descriptor(rank.min(5));
            assert!(matches!(
                context.validate_tensor_map(
                    &buffer,
                    0,
                    32_768,
                    TensorMapDescriptor { rank, ..descriptor }
                ),
                Err(CudaError::InvalidTensorMap("rank"))
            ));
        }
        mem::forget(context);
        mem::forget(buffer);
    }

    #[cfg(feature = "fake-driver-tests")]
    #[test]
    fn invalid_tensor_maps_cost_no_driver_encode() {
        let (context, buffer) = fake_context_and_buffer(0x1_0000, 65_536);
        let valid = tensor_map_descriptor(2);
        let mut invalid = Vec::new();
        let mut bad = valid;
        bad.box_dims[0] = 0;
        invalid.push(bad);
        bad = valid;
        bad.box_dims[0] = 257;
        invalid.push(bad);
        bad = valid;
        bad.element_strides[1] = 9;
        invalid.push(bad);
        bad = valid;
        bad.global_strides_bytes[0] = 1000;
        invalid.push(bad);
        bad = valid;
        bad.global_dims[0] = 1_u64 << 33;
        invalid.push(bad);
        bad = valid;
        bad.box_dims[0] = 256;
        invalid.push(bad);
        bad = valid;
        bad.interleave = 2;
        invalid.push(bad);
        for descriptor in invalid {
            ffi::fake_driver_reset();
            assert!(context.tensor_map(&buffer, 0, 32_768, descriptor).is_err());
            assert_eq!(ffi::fake_driver_call_count(FAKE_ENCODE_TENSOR_MAP), 0);
        }
        ffi::fake_driver_reset();
        assert!(matches!(
            context.tensor_map(&buffer, 8, 32_768, valid),
            Err(CudaError::TensorMapAddressMisaligned { .. })
        ));
        assert!(
            context
                .tensor_map(&buffer, 0, 8_192, tensor_map_descriptor(5))
                .is_err()
        );
        assert_eq!(ffi::fake_driver_call_count(FAKE_ENCODE_TENSOR_MAP), 0);
        let other = fake_context(0);
        assert!(matches!(
            other.tensor_map(&buffer, 0, 32_768, valid),
            Err(CudaError::ContextMismatch)
        ));
        assert_eq!(ffi::fake_driver_call_count(FAKE_ENCODE_TENSOR_MAP), 0);
        mem::forget(other);
        mem::forget(context);
        mem::forget(buffer);
    }

    #[cfg(feature = "fake-driver-tests")]
    #[test]
    fn every_tensor_map_element_reaches_the_encoder() {
        let (context, buffer) = fake_context_and_buffer(0x1_0000, 65_536);
        for data_type in [
            TensorMapDataType::Uint8,
            TensorMapDataType::Uint16,
            TensorMapDataType::Int32,
            TensorMapDataType::Float16,
            TensorMapDataType::Bfloat16,
            TensorMapDataType::Float32,
            TensorMapDataType::Fp8E4M3,
            TensorMapDataType::Fp8E5M2,
            TensorMapDataType::U4Align8B,
            TensorMapDataType::U4Align16B,
            TensorMapDataType::U6Align16B,
        ] {
            let mut descriptor = tensor_map_descriptor(2);
            descriptor.data_type = data_type;
            descriptor.swizzle = TensorMapSwizzle::None;
            descriptor.global_strides_bytes[0] = 8192;
            context.tensor_map(&buffer, 0, 32_768, descriptor).unwrap();
            let (_, encoded, ..) = ffi::fake_driver_encoded_tensor_map();
            assert_eq!(encoded, data_type as u32);
        }
        mem::forget(context);
        mem::forget(buffer);
    }

    /// What a launch config accepts as a cluster, and what it refuses.
    ///
    /// The default asks for nothing and takes the launch path this crate always
    /// took. Any other shape has to be one CUDA forms, and a grid that does not
    /// divide into whole clusters is refused rather than adjusted into one that
    /// would have launched.
    #[test]
    fn a_cluster_is_the_default_or_a_shape_the_grid_divides_into() {
        let plain = LaunchConfig {
            grid: (8, 4, 2),
            block: (32, 1, 1),
            shared_memory_bytes: 0,
            cluster: LaunchConfig::NO_CLUSTER,
            programmatic_stream_serialization: false,
        };
        assert!(!plain.wants_cluster());
        assert_eq!(plain.validate().expect("the default validates"), plain);

        let clustered = LaunchConfig {
            cluster: (2, 2, 2),
            ..plain
        };
        assert!(clustered.wants_cluster());
        assert_eq!(
            clustered.validate().expect("an even cluster validates"),
            clustered
        );

        for refused in [
            LaunchConfig {
                cluster: (0, 1, 1),
                ..plain
            },
            LaunchConfig {
                cluster: (1, 3, 1),
                ..plain
            },
        ] {
            assert!(
                matches!(refused.validate(), Err(CudaError::InvalidLaunch)),
                "cluster {:?} must be refused",
                refused.cluster
            );
        }
    }

    #[test]
    fn programmatic_stream_serialization_requires_compute_9() {
        let old = DeviceInfo {
            ordinal: 0,
            compute_major: 8,
            compute_minor: 9,
            multiprocessor_count: 188,
            memory_pools_supported: true,
            cooperative_launch: true,
        };
        assert!(matches!(
            super::programmatic_capable(&old),
            Err(CudaError::ProgrammaticLaunchUnsupported { .. })
        ));
        let capable = DeviceInfo {
            compute_major: 9,
            compute_minor: 0,
            ..old
        };
        assert!(super::programmatic_capable(&capable).is_ok());
    }

    /// The portable bound binds every caller inside this process, not only the
    /// ABI's.
    ///
    /// The bundle ABI refuses a cluster past this bound before a bundle reaches
    /// the host at all. A host route, a probe, or anything else in this process
    /// does not pass through that check, so the same number is enforced here:
    /// otherwise the one caller the ABI was written to bound would be the only
    /// one bounded.
    #[test]
    fn a_cluster_larger_than_the_portable_maximum_is_refused_by_this_crate_too() {
        assert_eq!(MAXIMUM_CLUSTER_BLOCKS, 8);
        let grid = (16, 16, 16);
        let block = (32, 1, 1);

        // Every shape at or under the bound that the grid also divides by.
        for cluster in [(8, 1, 1), (2, 2, 2), (4, 2, 1), (1, 1, 8)] {
            let config = LaunchConfig {
                grid,
                block,
                shared_memory_bytes: 0,
                cluster,
                programmatic_stream_serialization: false,
            };
            assert_eq!(
                config.validate().expect("a cluster inside the bound"),
                config,
                "cluster {cluster:?} holds at most {MAXIMUM_CLUSTER_BLOCKS} blocks"
            );
        }

        // One block past the bound, and a shape inside every extent whose
        // product is still past it.
        for (cluster, blocks) in [
            ((16, 1, 1), 16),
            ((4, 4, 1), 16),
            ((2, 2, 4), 16),
            ((16, 16, 16), 4_096),
        ] {
            let config = LaunchConfig {
                grid,
                block,
                shared_memory_bytes: 0,
                cluster,
                programmatic_stream_serialization: false,
            };
            assert!(
                matches!(
                    config.validate(),
                    Err(CudaError::ClusterBlocksExceeded { blocks: actual, maximum })
                        if actual == blocks && maximum == MAXIMUM_CLUSTER_BLOCKS
                ),
                "cluster {cluster:?} holds {blocks} blocks and must be refused"
            );
        }

        // The product is computed without wrapping, so a shape that would
        // overflow is refused rather than folded back inside the bound.
        let overflowing = LaunchConfig {
            grid,
            block,
            shared_memory_bytes: 0,
            cluster: (1 << 16, 1 << 16, 4),
            programmatic_stream_serialization: false,
        };
        assert!(matches!(
            overflowing.validate(),
            Err(CudaError::InvalidLaunch)
        ));
    }

    /// The capability gate, checked against a device this crate cannot retain.
    ///
    /// `retain_primary_context` accepts only SM120 today, so nothing reachable
    /// through this crate can fail this gate. That is exactly why the rule is
    /// checked here instead: the gate exists for the day that admission rule
    /// widens, and a defence no test can reach is one nobody would notice
    /// breaking.
    #[test]
    fn the_cluster_capability_gate_refuses_an_architecture_that_forms_no_cluster() {
        let device = |major: i32, minor: i32| DeviceInfo {
            ordinal: 3,
            compute_major: major,
            compute_minor: minor,
            multiprocessor_count: 188,
            memory_pools_supported: true,
            cooperative_launch: true,
        };

        // Clusters arrived at 9.0, so 9.0 and everything above it is capable.
        for (major, minor) in [(9, 0), (10, 0), (12, 0)] {
            assert!(
                super::cluster_capable(&device(major, minor)).is_ok(),
                "compute capability {major}.{minor} forms clusters"
            );
        }
        for (major, minor) in [(8, 9), (7, 5), (0, 0)] {
            assert!(
                matches!(
                    super::cluster_capable(&device(major, minor)),
                    Err(CudaError::ClusterLaunchUnsupported {
                        ordinal: 3,
                        compute_major,
                        compute_minor,
                    }) if compute_major == major && compute_minor == minor
                ),
                "compute capability {major}.{minor} forms no cluster and must be named"
            );
        }

        // The refusal says which device and which capability, because that is
        // the whole of what it adds over the driver's own error.
        let refusal = super::cluster_capable(&device(8, 9)).expect_err("8.9 is refused");
        let message = refusal.to_string();
        assert!(message.contains("device 3"), "{message}");
        assert!(message.contains("8.9"), "{message}");
        assert!(message.contains("9.0 is required"), "{message}");
    }

    #[test]
    fn each_launch_argument_marshals_at_the_width_it_names() {
        // The argument enum is what fixes a parameter's width at the driver
        // boundary, so each variant maps to exactly one marshalled width.
        let mut values = [
            ffi::Argument::Device(0x2000),
            ffi::Argument::Device(0),
            ffi::Argument::U32(0xABCD_1234),
            ffi::Argument::U64(0x1122_3344_5566_7788),
        ];
        let widths = values
            .iter_mut()
            .map(|argument| match argument {
                ffi::Argument::Device(_) => size_of::<ffi::DevicePointer>(),
                ffi::Argument::U32(_) => size_of::<u32>(),
                ffi::Argument::U64(_) => size_of::<u64>(),
                _ => panic!("this launch passes no other width"),
            })
            .collect::<Vec<_>>();
        assert_eq!(widths, [8, 8, 4, 8]);
        // A null device pointer is the zero address, not an absent argument.
        assert!(matches!(values[1], ffi::Argument::Device(0)));
    }
}

/// Controls for the statically linked CPU fake driver, absent in production builds.
#[cfg(feature = "fake-driver-tests")]
pub mod fake_driver {
    pub fn reset() {
        super::ffi::fake_driver_reset();
    }
    pub fn deny_peer_access(device: i32, peer: i32) {
        super::ffi::fake_driver_deny_peer_access(device, peer);
    }
    pub fn fail_next(operation: i32, code: i32) {
        super::ffi::fake_driver_fail_next(operation, code);
    }
    pub fn call_count(operation: i32) -> usize {
        super::ffi::fake_driver_call_count(operation)
    }
    /// Writes that reached an allocation before its own stream had executed
    /// it, from another stream or from a copy outside stream order.
    pub fn ordering_violations() -> usize {
        super::ffi::fake_driver_ordering_violations()
    }
    /// Copies land at the API call rather than in stream order until reset.
    pub fn set_api_call_copies(enabled: bool) {
        super::ffi::fake_driver_set_api_call_copies(enabled);
    }
}
