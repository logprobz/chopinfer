//! Private ergonomic wrappers over the generated CUDA 13.2.1 raw binding.

// SAFETY-REVIEWED: this private module is the sole reviewed CUDA FFI boundary.
#![allow(unsafe_code)]
#![allow(non_camel_case_types)]

use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::ptr;

mod raw {
    #![allow(
        dead_code,
        non_camel_case_types,
        non_snake_case,
        non_upper_case_globals
    )]

    include!("../bindings/cuda-driver-13.2.rs.inc");
}

pub const CUDA_SUCCESS: c_int = raw::RUNTIME_CUDA_SUCCESS as c_int;
pub const DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int =
    raw::RUNTIME_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR as c_int;
pub const DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int =
    raw::RUNTIME_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR as c_int;
pub const DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED: c_int =
    raw::RUNTIME_CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED as c_int;
/// CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT in the CUDA 13.2 header; the
/// pinned bindings do not carry this constant.
pub const DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
/// CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH in the CUDA 13.2 header; the
/// pinned bindings do not carry this constant.
pub const DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH: c_int = 95;

const STREAM_NON_BLOCKING: c_uint = raw::RUNTIME_CU_STREAM_NON_BLOCKING;
const EVENT_DISABLE_TIMING: c_uint = raw::RUNTIME_CU_EVENT_DISABLE_TIMING;
const STREAM_CAPTURE_MODE_THREAD_LOCAL: raw::CUstreamCaptureMode =
    raw::RUNTIME_CU_STREAM_CAPTURE_MODE_THREAD_LOCAL;
const MEMPOOL_ATTRIBUTE_RELEASE_THRESHOLD: raw::CUmemPool_attribute =
    raw::RUNTIME_CU_MEMPOOL_ATTR_RELEASE_THRESHOLD;

pub type ContextHandle = raw::CUcontext;
pub type StreamHandle = raw::CUstream;
pub type EventHandle = raw::CUevent;
pub type ModuleHandle = raw::CUmodule;
pub type FunctionHandle = raw::CUfunction;
pub type MemoryPoolHandle = raw::CUmemoryPool;
pub type GraphHandle = raw::CUgraph;
pub type GraphExecHandle = raw::CUgraphExec;
pub type LibraryHandle = raw::CUlibrary;
pub type KernelHandle = raw::CUkernel;
pub type DevicePointer = raw::CUdeviceptr;
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct TensorMap(pub [u64; 16]);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TensorMapDescriptor {
    pub rank: u32,
    pub data_type: u32,
    pub swizzle: u32,
    pub interleave: u32,
    pub l2_promotion: u32,
    pub oob_fill: u32,
    pub global_dims: [u64; 5],
    pub global_strides_bytes: [u64; 4],
    pub box_dims: [u32; 5],
    pub element_strides: [u32; 5],
}

// SAFETY-REVIEWED: these declarations match the pinned CUDA header and its compiled helpers.
unsafe extern "C" {
    fn cuEventQuery(event: EventHandle) -> raw::CUresult;
    fn runtime_cuda_copy_pages(
        destinations: *mut DevicePointer,
        sources: *mut DevicePointer,
        sizes: *mut usize,
        count: usize,
        stream: StreamHandle,
    ) -> raw::CUresult;
    fn cuDeviceGetPCIBusId(bus_id: *mut c_char, len: c_int, device: c_int) -> raw::CUresult;
    fn runtime_cuda_device_uuid(bytes: *mut u8, ordinal: c_int) -> raw::CUresult;
    fn runtime_cuda_memory_info(free_bytes: *mut usize, total_bytes: *mut usize) -> raw::CUresult;
    fn runtime_cuda_event_elapsed(
        milliseconds: *mut f32,
        start: EventHandle,
        end: EventHandle,
    ) -> raw::CUresult;
    fn cuMemsetD8Async(
        destination: DevicePointer,
        value: u8,
        count: usize,
        stream: StreamHandle,
    ) -> raw::CUresult;
    fn cuMemcpyDtoDAsync_v2(
        destination: DevicePointer,
        source: DevicePointer,
        bytes: usize,
        stream: StreamHandle,
    ) -> raw::CUresult;
    fn runtime_cuda_validate_tensor_map(
        descriptor: *const TensorMapDescriptor,
        address: DevicePointer,
        buffer_bytes: u64,
    ) -> raw::CUresult;
    fn runtime_cuda_encode_tensor_map(
        output: *mut c_void,
        descriptor: *const TensorMapDescriptor,
        address: DevicePointer,
        buffer_bytes: u64,
    ) -> raw::CUresult;
    fn runtime_cuda_set_max_dynamic_shared(function: FunctionHandle, bytes: c_int)
    -> raw::CUresult;
    fn runtime_cuda_max_active_clusters(
        function: FunctionHandle,
        gx: c_uint,
        gy: c_uint,
        gz: c_uint,
        bx: c_uint,
        by: c_uint,
        bz: c_uint,
        shared_memory_bytes: c_uint,
        cx: c_uint,
        cy: c_uint,
        cz: c_uint,
        cooperative: c_int,
        clusters: *mut c_int,
    ) -> raw::CUresult;
    fn runtime_cuda_context_sm_count(sms: *mut c_int) -> raw::CUresult;
    fn runtime_cuda_memory_pool_set_device_access(
        pool: MemoryPoolHandle,
        device: c_int,
        read_write: c_int,
    ) -> raw::CUresult;
    fn runtime_cuda_memory_pool_get_device_access(
        pool: MemoryPoolHandle,
        device: c_int,
        read_write: *mut c_int,
    ) -> raw::CUresult;
    fn runtime_cuda_host_alloc(address: *mut *mut c_void, bytes: usize) -> raw::CUresult;
    fn runtime_cuda_host_free(address: *mut c_void) -> raw::CUresult;
    fn runtime_cuda_host_alloc_mapped(
        address: *mut *mut c_void,
        device: *mut DevicePointer,
        bytes: usize,
    ) -> raw::CUresult;
    fn runtime_cuda_context_synchronize() -> raw::CUresult;
    fn runtime_cuda_launch_kernel_pdl(
        function: FunctionHandle,
        grid_x: c_uint,
        grid_y: c_uint,
        grid_z: c_uint,
        block_x: c_uint,
        block_y: c_uint,
        block_z: c_uint,
        shared_memory_bytes: c_uint,
        stream: StreamHandle,
        parameters: *mut *mut c_void,
    ) -> raw::CUresult;
    #[allow(clippy::too_many_arguments)]
    fn runtime_cuda_launch_kernel_cooperative(
        function: FunctionHandle,
        grid_x: c_uint,
        grid_y: c_uint,
        grid_z: c_uint,
        block_x: c_uint,
        block_y: c_uint,
        block_z: c_uint,
        shared_memory_bytes: c_uint,
        cluster_x: c_uint,
        cluster_y: c_uint,
        cluster_z: c_uint,
        stream: StreamHandle,
        parameters: *mut *mut c_void,
    ) -> raw::CUresult;
    #[allow(clippy::too_many_arguments)]
    fn runtime_cuda_launch_kernel_cluster(
        function: FunctionHandle,
        grid_x: c_uint,
        grid_y: c_uint,
        grid_z: c_uint,
        block_x: c_uint,
        block_y: c_uint,
        block_z: c_uint,
        shared_memory_bytes: c_uint,
        cluster_x: c_uint,
        cluster_y: c_uint,
        cluster_z: c_uint,
        programmatic_stream_serialization: c_int,
        stream: StreamHandle,
        parameters: *mut *mut c_void,
    ) -> raw::CUresult;
}

#[cfg(feature = "fake-driver-tests")]
// SAFETY-REVIEWED: these declarations exactly match the no-device C stub ABI. Reset takes and returns nothing, fail-next and peer denial use two C ints, call-count uses one C int and returns size_t, and no pointers or borrowed lifetimes cross the boundary.
unsafe extern "C" {
    fn runtime_cuda_fake_reset();
    fn runtime_cuda_fake_deny_peer_access(device: c_int, peer: c_int);
    fn runtime_cuda_fake_fail_next(operation: c_int, code: c_int);
    fn runtime_cuda_fake_call_count(operation: c_int) -> usize;
    fn runtime_cuda_fake_ordering_violations() -> usize;
    fn runtime_cuda_fake_set_api_call_copies(enabled: c_int);
    #[cfg(test)]
    fn runtime_cuda_fake_encoded_tensor_map(
        address: *mut u64,
        data_type: *mut u32,
        swizzle: *mut u32,
        global_dims: *mut u64,
        stride: *mut u64,
        box_dims: *mut u32,
    );
}

#[cfg(feature = "fake-driver-tests")]
pub fn fake_driver_reset() {
    // SAFETY-REVIEWED: the no-device test stub resets only process-local counters.
    unsafe { runtime_cuda_fake_reset() }
}

#[cfg(feature = "fake-driver-tests")]
pub fn fake_driver_deny_peer_access(device: c_int, peer: c_int) {
    // SAFETY-REVIEWED: the no-device stub stores two thread-local integer ordinals.
    unsafe { runtime_cuda_fake_deny_peer_access(device, peer) }
}

#[cfg(feature = "fake-driver-tests")]
pub fn fake_driver_fail_next(operation: c_int, code: c_int) {
    // SAFETY-REVIEWED: the no-device test stub stores two integer fault controls.
    unsafe { runtime_cuda_fake_fail_next(operation, code) }
}

/// The 2D tensor map the no-device stub was last asked to encode.
#[cfg(all(test, feature = "fake-driver-tests"))]
pub fn fake_driver_encoded_tensor_map() -> (u64, u32, u32, [u64; 2], u64, [u32; 2]) {
    let mut address = 0;
    let mut data_type = 0;
    let mut swizzle = 0;
    let mut global_dims = [0_u64; 2];
    let mut stride = 0;
    let mut box_dims = [0_u32; 2];
    // SAFETY-REVIEWED: the stub writes each out parameter exactly once.
    unsafe {
        runtime_cuda_fake_encoded_tensor_map(
            &raw mut address,
            &raw mut data_type,
            &raw mut swizzle,
            global_dims.as_mut_ptr(),
            &raw mut stride,
            box_dims.as_mut_ptr(),
        );
    }
    (address, data_type, swizzle, global_dims, stride, box_dims)
}

#[cfg(feature = "fake-driver-tests")]
pub fn fake_driver_call_count(operation: c_int) -> usize {
    // SAFETY-REVIEWED: the no-device test stub returns one bounded counter.
    unsafe { runtime_cuda_fake_call_count(operation) }
}

/// Writes the no-device stub saw land in an allocation before it was ordered.
#[cfg(feature = "fake-driver-tests")]
pub fn fake_driver_ordering_violations() -> usize {
    // SAFETY-REVIEWED: the no-device test stub returns one bounded counter.
    unsafe { runtime_cuda_fake_ordering_violations() }
}

/// Makes the no-device stub's copies land at the API call instead of in
/// stream order, until the next reset.
#[cfg(feature = "fake-driver-tests")]
pub fn fake_driver_set_api_call_copies(enabled: bool) {
    // SAFETY-REVIEWED: the no-device test stub stores one integer flag.
    unsafe { runtime_cuda_fake_set_api_call_copies(c_int::from(enabled)) }
}

/// The slots one by-value device-address argument carries.
///
/// Eight is the bound a caller may fill, and the record is always that long:
/// the most ranks of a collective (`kernel_core::contract::MAX_COMM_RANKS`,
/// which kernel-lab asserts equal; this crate links no workspace crate).
/// a kernel whose parameter is shorter reads the prefix it declares, and the
/// rest is argument storage it never looks at. The alignment is the strictest
/// any such by-value record asks for.
pub const DEVICE_POINTER_ARRAY_SLOTS: usize = 8;

#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct DevicePointerArray {
    pub addresses: [DevicePointer; DEVICE_POINTER_ARRAY_SLOTS],
}

impl DevicePointerArray {
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            addresses: [0; DEVICE_POINTER_ARRAY_SLOTS],
        }
    }
}

const _: () = {
    assert!(std::mem::align_of::<DevicePointerArray>() == 16);
    assert!(std::mem::size_of::<DevicePointerArray>() == 64);
};

pub enum Argument {
    Device(DevicePointer),
    I32(i32),
    DevicePointerArray(DevicePointerArray),
    U32(u32),
    U64(u64),
    TensorMap(TensorMap),
}

pub fn validate_tensor_map(
    descriptor: &TensorMapDescriptor,
    address: DevicePointer,
    buffer_bytes: u64,
) -> c_int {
    // SAFETY-REVIEWED: the C validator reads one live descriptor and no device memory.
    result_code(unsafe { runtime_cuda_validate_tensor_map(descriptor, address, buffer_bytes) })
}

pub fn tensor_map(
    descriptor: &TensorMapDescriptor,
    address: DevicePointer,
    buffer_bytes: u64,
) -> (c_int, TensorMap) {
    let mut map = TensorMap([0; 16]);
    // SAFETY-REVIEWED: the helper writes one aligned CUDA tensor-map value.
    let result = unsafe {
        runtime_cuda_encode_tensor_map(
            (&mut map as *mut TensorMap).cast(),
            descriptor,
            address,
            buffer_bytes,
        )
    };
    (result_code(result), map)
}

pub fn set_max_dynamic_shared(function: FunctionHandle, bytes: u32) -> c_int {
    // SAFETY-REVIEWED: the function handle is live and the value fits CUDA's integer API.
    result_code(unsafe { runtime_cuda_set_max_dynamic_shared(function, bytes as c_int) })
}

/// Clusters of `cluster` blocks at this launch shape, cooperative when
/// `cooperative`, that the current context's device keeps resident at once.
pub fn max_active_clusters(
    function: FunctionHandle,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared_memory_bytes: u32,
    cluster: (u32, u32, u32),
    cooperative: bool,
) -> (c_int, i32) {
    let mut clusters: c_int = 0;
    // SAFETY-REVIEWED: the function handle is live, the context is current and the output is this frame's own.
    let result = unsafe {
        runtime_cuda_max_active_clusters(
            function,
            grid.0,
            grid.1,
            grid.2,
            block.0,
            block.1,
            block.2,
            shared_memory_bytes,
            cluster.0,
            cluster.1,
            cluster.2,
            c_int::from(cooperative),
            &mut clusters,
        )
    };
    (result_code(result), clusters)
}

/// The SM-count execution affinity of the current context.
pub fn context_sm_count() -> (c_int, i32) {
    let mut sms: c_int = 0;
    // SAFETY-REVIEWED: the output is this frame's own and a context is current.
    let result = unsafe { runtime_cuda_context_sm_count(&mut sms) };
    (result_code(result), sms)
}

pub const fn null_context() -> ContextHandle {
    ptr::null_mut()
}

pub fn init() -> c_int {
    // SAFETY-REVIEWED: zero is the only valid cuInit flag.
    result_code(unsafe { raw::cuInit(0) })
}

pub fn driver_version() -> (c_int, c_int) {
    let mut value = 0;
    // SAFETY-REVIEWED: CUDA writes one integer to a live stack slot.
    let result = unsafe { raw::cuDriverGetVersion(&mut value) };
    (result_code(result), value)
}

pub fn device_count() -> (c_int, c_int) {
    let mut value = 0;
    // SAFETY-REVIEWED: CUDA writes one integer to a live stack slot.
    let result = unsafe { raw::cuDeviceGetCount(&mut value) };
    (result_code(result), value)
}

pub fn device_pci_bus_id(ordinal: c_int) -> (c_int, String) {
    let mut bytes = [0_u8; 32];
    // SAFETY-REVIEWED: CUDA writes at most bytes.len() bytes into the owned buffer.
    let result =
        unsafe { cuDeviceGetPCIBusId(bytes.as_mut_ptr().cast(), bytes.len() as c_int, ordinal) };
    let end = bytes
        .iter()
        .position(|&byte| byte == 0)
        .unwrap_or(bytes.len());
    (
        result_code(result),
        String::from_utf8_lossy(&bytes[..end]).into_owned(),
    )
}

pub fn device_get(ordinal: c_int) -> (c_int, c_int) {
    let mut device = 0;
    // SAFETY-REVIEWED: CUDA writes one device integer to a live stack slot.
    let result = unsafe { raw::cuDeviceGet(&mut device, ordinal) };
    (result_code(result), device)
}

pub fn device_attribute(device: c_int, attribute: c_int) -> (c_int, c_int) {
    let mut value = 0;
    // SAFETY-REVIEWED: CUDA writes one attribute integer to a live stack slot.
    let result = unsafe {
        raw::cuDeviceGetAttribute(&mut value, attribute as raw::CUdevice_attribute, device)
    };
    (result_code(result), value)
}

pub fn primary_context_retain(device: c_int) -> (c_int, ContextHandle) {
    let mut context = ptr::null_mut();
    // SAFETY-REVIEWED: CUDA initializes the live context handle slot.
    let result = unsafe { raw::cuDevicePrimaryCtxRetain(&mut context, device) };
    (result_code(result), context)
}

pub fn primary_context_release(device: c_int) -> c_int {
    // SAFETY-REVIEWED: the caller balances a successful primary-context retain.
    result_code(unsafe { raw::cuDevicePrimaryCtxRelease_v2(device) })
}

pub fn context_set_current(context: ContextHandle) -> c_int {
    // SAFETY-REVIEWED: the handle is null or retained by a live ContextInner.
    result_code(unsafe { raw::cuCtxSetCurrent(context) })
}

pub fn context_enable_peer_access(peer: ContextHandle) -> c_int {
    // SAFETY-REVIEWED: the peer handle is retained and flags must be zero.
    result_code(unsafe { raw::cuCtxEnablePeerAccess(peer, 0) })
}

pub fn context_disable_peer_access(peer: ContextHandle) -> c_int {
    // SAFETY-REVIEWED: the current context owns the peer-access lease being released.
    result_code(unsafe { raw::cuCtxDisablePeerAccess(peer) })
}

pub fn device_can_access_peer(device: c_int, peer: c_int) -> (c_int, c_int) {
    let mut access = 0;
    // SAFETY-REVIEWED: CUDA writes one access integer to a live stack slot.
    let result = unsafe { raw::cuDeviceCanAccessPeer(&mut access, device, peer) };
    (result_code(result), access)
}

pub fn stream_create() -> (c_int, StreamHandle) {
    let mut stream = ptr::null_mut();
    // SAFETY-REVIEWED: CUDA initializes the live stream handle slot.
    let result = unsafe { raw::cuStreamCreate(&mut stream, STREAM_NON_BLOCKING) };
    (result_code(result), stream)
}

pub fn stream_destroy(stream: StreamHandle) -> c_int {
    // SAFETY-REVIEWED: StreamInner owns this initialized handle exactly once.
    result_code(unsafe { raw::cuStreamDestroy_v2(stream) })
}

pub fn stream_synchronize(stream: StreamHandle) -> c_int {
    // SAFETY-REVIEWED: the stream handle is retained by a live StreamInner.
    result_code(unsafe { raw::cuStreamSynchronize(stream) })
}

pub fn stream_wait_event(stream: StreamHandle, event: EventHandle) -> c_int {
    // SAFETY-REVIEWED: both handles remain live and CUDA permits cross-context event waits.
    result_code(unsafe { raw::cuStreamWaitEvent(stream, event, 0) })
}

pub fn event_create(timing: bool) -> (c_int, EventHandle) {
    let mut event = ptr::null_mut();
    // SAFETY-REVIEWED: CUDA initializes the live event handle slot.
    let result =
        unsafe { raw::cuEventCreate(&mut event, if timing { 0 } else { EVENT_DISABLE_TIMING }) };
    (result_code(result), event)
}

pub fn event_elapsed(start: EventHandle, end: EventHandle) -> (c_int, f32) {
    let mut milliseconds = 0.0;
    // SAFETY-REVIEWED: the safe layer retains both events and checks their context.
    let result = unsafe { runtime_cuda_event_elapsed(&mut milliseconds, start, end) };
    (result_code(result), milliseconds)
}

pub fn event_destroy(event: EventHandle) -> c_int {
    // SAFETY-REVIEWED: Event owns this initialized handle exactly once.
    result_code(unsafe { raw::cuEventDestroy_v2(event) })
}

pub fn event_record(event: EventHandle, stream: StreamHandle) -> c_int {
    // SAFETY-REVIEWED: both handles share a live context checked by the safe layer.
    result_code(unsafe { raw::cuEventRecord(event, stream) })
}

pub fn event_synchronize(event: EventHandle) -> c_int {
    // SAFETY-REVIEWED: the event handle is retained by a live Event.
    result_code(unsafe { raw::cuEventSynchronize(event) })
}

pub fn default_memory_pool(device: c_int) -> (c_int, MemoryPoolHandle) {
    let mut pool = ptr::null_mut();
    // SAFETY-REVIEWED: CUDA initializes the live pool handle slot.
    let result = unsafe { raw::cuDeviceGetDefaultMemPool(&mut pool, device) };
    (result_code(result), pool)
}

pub fn memory_pool_set_release_threshold(pool: MemoryPoolHandle, mut bytes: u64) -> c_int {
    // SAFETY-REVIEWED: CUDA reads one u64 for the release-threshold attribute.
    let result = unsafe {
        raw::cuMemPoolSetAttribute(
            pool,
            MEMPOOL_ATTRIBUTE_RELEASE_THRESHOLD,
            (&mut bytes as *mut u64).cast(),
        )
    };
    result_code(result)
}

pub fn memory_pool_set_device_access(
    pool: MemoryPoolHandle,
    device: c_int,
    read_write: bool,
) -> c_int {
    // SAFETY-REVIEWED: the helper initializes one CUDA access descriptor from checked values.
    result_code(unsafe {
        runtime_cuda_memory_pool_set_device_access(pool, device, c_int::from(read_write))
    })
}

pub fn memory_pool_get_device_access(pool: MemoryPoolHandle, device: i32) -> (c_int, bool) {
    let mut read_write = 0;
    let result =
        // SAFETY-REVIEWED: the helper writes one initialized integer result.
        unsafe { runtime_cuda_memory_pool_get_device_access(pool, device, &raw mut read_write) };
    (result_code(result), read_write != 0)
}

pub fn memory_allocate(
    pool: MemoryPoolHandle,
    stream: StreamHandle,
    bytes: usize,
) -> (c_int, DevicePointer) {
    let mut pointer = 0;
    // SAFETY-REVIEWED: CUDA initializes one device pointer for live pool and stream handles.
    let result = unsafe { raw::cuMemAllocFromPoolAsync(&mut pointer, bytes, pool, stream) };
    (result_code(result), pointer)
}

pub fn memory_free(pointer: DevicePointer, stream: StreamHandle) -> c_int {
    // SAFETY-REVIEWED: DeviceBuffer owns the allocation and keeps its stream alive.
    result_code(unsafe { raw::cuMemFreeAsync(pointer, stream) })
}

pub fn memory_zero(destination: DevicePointer, bytes: usize, stream: StreamHandle) -> c_int {
    // SAFETY-REVIEWED: the safe layer validates the destination range and owns the stream.
    result_code(unsafe { cuMemsetD8Async(destination, 0, bytes, stream) })
}

pub fn copy_device_to_device(
    destination: DevicePointer,
    source: DevicePointer,
    bytes: usize,
    stream: StreamHandle,
) -> c_int {
    // SAFETY-REVIEWED: the safe layer validates both ranges in one owned allocation.
    result_code(unsafe { cuMemcpyDtoDAsync_v2(destination, source, bytes, stream) })
}

pub fn copy_peer_async(
    destination: DevicePointer,
    destination_context: ContextHandle,
    source: DevicePointer,
    source_context: ContextHandle,
    bytes: usize,
    stream: StreamHandle,
) -> c_int {
    // SAFETY-REVIEWED: the safe layer binds both allocation spans, contexts, and the stream.
    result_code(unsafe {
        raw::cuMemcpyPeerAsync(
            destination,
            destination_context,
            source,
            source_context,
            bytes,
            stream,
        )
    })
}

pub fn context_synchronize() -> c_int {
    // SAFETY-REVIEWED: the safe context wrapper sets the current context first.
    result_code(unsafe { runtime_cuda_context_synchronize() })
}

/// Owns one CUDA page-locked allocation. The safe outer owner keeps its
/// context alive and finishes all transfers before this object is dropped.
pub struct PinnedAllocation {
    pointer: *mut c_void,
    bytes: usize,
}

impl PinnedAllocation {
    pub fn allocate(bytes: usize) -> (c_int, Option<Self>) {
        let mut pointer = ptr::null_mut();
        // SAFETY-REVIEWED: the shim validates byte length and initializes one pointer.
        let code = result_code(unsafe { runtime_cuda_host_alloc(&mut pointer, bytes) });
        let allocation =
            (code == CUDA_SUCCESS && !pointer.is_null()).then_some(Self { pointer, bytes });
        (code, allocation)
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY-REVIEWED: this object exclusively owns bytes returned by cuMemHostAlloc.
        unsafe { std::slice::from_raw_parts(self.pointer.cast(), self.bytes) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY-REVIEWED: mutable access requires a unique borrow of the owner.
        unsafe { std::slice::from_raw_parts_mut(self.pointer.cast(), self.bytes) }
    }
}

impl Drop for PinnedAllocation {
    fn drop(&mut self) {
        // SAFETY-REVIEWED: the outer owner has synchronized every transfer.
        let _ = unsafe { runtime_cuda_host_free(self.pointer) };
    }
}

/// Owns one page-locked host allocation that every context maps into its
/// device address space. The safe outer owner synchronizes the device before
/// it frees the bytes, because kernels read them directly.
pub struct MappedHostAllocation {
    pointer: *mut c_void,
    device: DevicePointer,
    bytes: usize,
    freed: std::cell::Cell<bool>,
}

impl MappedHostAllocation {
    pub fn allocate(bytes: usize) -> (c_int, Option<Self>) {
        let mut pointer = ptr::null_mut();
        let mut device = 0;
        // SAFETY-REVIEWED: the shim validates the length, initializes both
        // outputs, and frees the allocation itself if mapping fails.
        let code = result_code(unsafe {
            runtime_cuda_host_alloc_mapped(&mut pointer, &mut device, bytes)
        });
        let allocation =
            (code == CUDA_SUCCESS && !pointer.is_null() && device != 0).then_some(Self {
                pointer,
                device,
                bytes,
                freed: std::cell::Cell::new(false),
            });
        (code, allocation)
    }

    #[must_use]
    pub const fn device_pointer(&self) -> DevicePointer {
        self.device
    }

    /// Where the host bytes start; they stay allocated until `free` or drop.
    #[must_use]
    pub fn host_address(&self) -> usize {
        self.pointer as usize
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY-REVIEWED: mutable access requires a unique borrow of the
        // owner, and the bytes stay allocated until `free` or drop.
        unsafe { std::slice::from_raw_parts_mut(self.pointer.cast(), self.bytes) }
    }

    /// Frees the bytes once; the caller has synchronized every reader.
    pub fn free(&self) -> c_int {
        if self.freed.get() {
            return CUDA_SUCCESS;
        }
        // SAFETY-REVIEWED: the pointer came from cuMemHostAlloc and is freed once.
        let code = result_code(unsafe { runtime_cuda_host_free(self.pointer) });
        if code == CUDA_SUCCESS {
            self.freed.set(true);
        }
        code
    }
}

impl Drop for MappedHostAllocation {
    fn drop(&mut self) {
        let _ = self.free();
    }
}

pub fn copy_pinned_to_device(
    destination: DevicePointer,
    source: &PinnedAllocation,
    source_offset: usize,
    bytes: usize,
    stream: StreamHandle,
) -> c_int {
    let pointer = source.as_slice()[source_offset..source_offset + bytes].as_ptr();
    // SAFETY-REVIEWED: the outer transfer guard retains the pinned allocation until sync.
    result_code(unsafe { raw::cuMemcpyHtoDAsync_v2(destination, pointer.cast(), bytes, stream) })
}

pub fn copy_device_to_pinned(
    destination: &mut PinnedAllocation,
    destination_offset: usize,
    source: DevicePointer,
    bytes: usize,
    stream: StreamHandle,
) -> c_int {
    let pointer =
        destination.as_mut_slice()[destination_offset..destination_offset + bytes].as_mut_ptr();
    // SAFETY-REVIEWED: the outer transfer guard retains the pinned allocation until sync.
    result_code(unsafe { raw::cuMemcpyDtoHAsync_v2(pointer.cast(), source, bytes, stream) })
}

pub fn copy_device_to_host(
    destination: &mut [u8],
    source: DevicePointer,
    stream: StreamHandle,
) -> c_int {
    // SAFETY-REVIEWED: the safe layer checks source capacity and the slice stays live.
    let result = unsafe {
        raw::cuMemcpyDtoHAsync_v2(
            destination.as_mut_ptr().cast(),
            source,
            destination.len(),
            stream,
        )
    };
    result_code(result)
}

pub fn copy_f32_host_to_device(
    destination: DevicePointer,
    source: &[f32],
    stream: StreamHandle,
) -> c_int {
    let bytes = std::mem::size_of_val(source);
    // SAFETY-REVIEWED: the safe layer checks destination capacity and the slice stays live.
    result_code(unsafe {
        raw::cuMemcpyHtoDAsync_v2(destination, source.as_ptr().cast(), bytes, stream)
    })
}

pub fn copy_f32_device_to_host(
    destination: &mut [f32],
    source: DevicePointer,
    stream: StreamHandle,
) -> c_int {
    let bytes = std::mem::size_of_val(destination);
    // SAFETY-REVIEWED: the safe layer checks source capacity and the slice stays live.
    result_code(unsafe {
        raw::cuMemcpyDtoHAsync_v2(destination.as_mut_ptr().cast(), source, bytes, stream)
    })
}

pub fn module_load(cubin: &[u8]) -> (c_int, ModuleHandle) {
    let mut module = ptr::null_mut();
    // SAFETY-REVIEWED: cubin bytes stay live for the synchronous module load.
    let result = unsafe { raw::cuModuleLoadData(&mut module, cubin.as_ptr().cast()) };
    (result_code(result), module)
}

pub fn module_unload(module: ModuleHandle) -> c_int {
    // SAFETY-REVIEWED: ModuleInner owns this initialized handle exactly once.
    result_code(unsafe { raw::cuModuleUnload(module) })
}

pub fn module_function(module: ModuleHandle, name: &CString) -> (c_int, FunctionHandle) {
    let mut function = ptr::null_mut();
    // SAFETY-REVIEWED: module and NUL-terminated name remain live for the lookup.
    let result = unsafe { raw::cuModuleGetFunction(&mut function, module, name.as_ptr()) };
    (result_code(result), function)
}

pub fn library_unload(library: LibraryHandle) -> c_int {
    // SAFETY-REVIEWED: LibraryInner owns this initialized handle exactly once.
    result_code(unsafe { raw::cuLibraryUnload(library) })
}

pub fn library_kernel(library: LibraryHandle, name: &CString) -> (c_int, KernelHandle) {
    let mut kernel = ptr::null_mut();
    // SAFETY-REVIEWED: library and NUL-terminated name remain live for the lookup.
    let result = unsafe { raw::cuLibraryGetKernel(&mut kernel, library, name.as_ptr()) };
    (result_code(result), kernel)
}

pub fn kernel_function(kernel: KernelHandle) -> (c_int, FunctionHandle) {
    let mut function = ptr::null_mut();
    // SAFETY-REVIEWED: the kernel belongs to a live library retained by the safe layer.
    let result = unsafe { raw::cuKernelGetFunction(&mut function, kernel) };
    (result_code(result), function)
}

/// Points one parameter slot at each argument's own storage, in order.
///
/// The addresses are into `arguments`, so the returned vector is only usable
/// while that slice is borrowed, which is exactly the length of one enqueue.
fn parameter_slots(arguments: &mut [Argument]) -> Vec<*mut c_void> {
    parameter_iter(arguments).collect()
}

fn parameter_iter(arguments: &mut [Argument]) -> impl Iterator<Item = *mut c_void> {
    arguments.iter_mut().map(|argument| match argument {
        Argument::Device(value) => (value as *mut DevicePointer).cast(),
        Argument::I32(value) => (value as *mut i32).cast(),
        Argument::DevicePointerArray(value) => (value as *mut DevicePointerArray).cast(),
        Argument::U32(value) => (value as *mut u32).cast(),
        Argument::U64(value) => (value as *mut u64).cast(),
        Argument::TensorMap(value) => (value as *mut TensorMap).cast(),
    })
}

pub fn launch_kernel(
    function: FunctionHandle,
    stream: StreamHandle,
    config: super::LaunchConfig,
    arguments: &mut [Argument],
) -> c_int {
    let mut stack = [ptr::null_mut(); 16];
    let mut heap;
    let parameters = if arguments.len() <= stack.len() {
        for (slot, pointer) in stack.iter_mut().zip(parameter_iter(arguments)) {
            *slot = pointer;
        }
        stack.as_mut_slice()
    } else {
        heap = parameter_slots(arguments);
        heap.as_mut_slice()
    };
    // SAFETY-REVIEWED: argument storage and handles remain live for the synchronous enqueue.
    let result = unsafe {
        raw::cuLaunchKernel(
            function,
            config.grid.0,
            config.grid.1,
            config.grid.2,
            config.block.0,
            config.block.1,
            config.block.2,
            config.shared_memory_bytes,
            stream,
            parameters.as_mut_ptr(),
            ptr::null_mut(),
        )
    };
    result_code(result)
}

/// A cooperative launch, with the config's cluster when it has one (a zero
/// cluster asks for none). The driver's refusal is returned unchanged.
pub fn launch_kernel_cooperative(
    function: FunctionHandle,
    stream: StreamHandle,
    config: super::LaunchConfig,
    cluster: Option<(u32, u32, u32)>,
    arguments: &mut [Argument],
) -> c_int {
    let mut parameters = parameter_slots(arguments);
    let cluster = cluster.unwrap_or((0, 0, 0));
    // SAFETY-REVIEWED: live argument storage is retained through the synchronous enqueue.
    result_code(unsafe {
        runtime_cuda_launch_kernel_cooperative(
            function,
            config.grid.0,
            config.grid.1,
            config.grid.2,
            config.block.0,
            config.block.1,
            config.block.2,
            config.shared_memory_bytes,
            cluster.0,
            cluster.1,
            cluster.2,
            stream,
            parameters.as_mut_ptr(),
        )
    })
}

pub fn launch_kernel_pdl(
    function: FunctionHandle,
    stream: StreamHandle,
    config: super::LaunchConfig,
    arguments: &mut [Argument],
) -> c_int {
    let mut parameters = parameter_slots(arguments);
    // SAFETY-REVIEWED: live argument storage is retained through the synchronous enqueue.
    result_code(unsafe {
        runtime_cuda_launch_kernel_pdl(
            function,
            config.grid.0,
            config.grid.1,
            config.grid.2,
            config.block.0,
            config.block.1,
            config.block.2,
            config.shared_memory_bytes,
            stream,
            parameters.as_mut_ptr(),
        )
    })
}

/// Launches with the cluster the config names, and optionally the PDL attribute.
///
/// This is reached only for a cluster the caller actually asked for, so the two
/// launches above stay exactly what they were. Whether the device forms this
/// cluster is the driver's answer, and its refusal is returned unchanged.
pub fn launch_kernel_cluster(
    function: FunctionHandle,
    stream: StreamHandle,
    config: super::LaunchConfig,
    programmatic_stream_serialization: bool,
    arguments: &mut [Argument],
) -> c_int {
    let mut parameters = parameter_slots(arguments);
    // SAFETY-REVIEWED: live argument storage is retained through the synchronous enqueue.
    result_code(unsafe {
        runtime_cuda_launch_kernel_cluster(
            function,
            config.grid.0,
            config.grid.1,
            config.grid.2,
            config.block.0,
            config.block.1,
            config.block.2,
            config.shared_memory_bytes,
            config.cluster.0,
            config.cluster.1,
            config.cluster.2,
            c_int::from(programmatic_stream_serialization),
            stream,
            parameters.as_mut_ptr(),
        )
    })
}

pub fn stream_begin_capture(stream: StreamHandle) -> c_int {
    // SAFETY-REVIEWED: the stream handle is live and capture mode is a valid enum value.
    result_code(unsafe { raw::cuStreamBeginCapture_v2(stream, STREAM_CAPTURE_MODE_THREAD_LOCAL) })
}

pub fn stream_end_capture(stream: StreamHandle) -> (c_int, GraphHandle) {
    let mut graph = ptr::null_mut();
    // SAFETY-REVIEWED: CUDA initializes the graph slot for a capturing live stream.
    let result = unsafe { raw::cuStreamEndCapture(stream, &mut graph) };
    (result_code(result), graph)
}

pub fn graph_instantiate(graph: GraphHandle) -> (c_int, GraphExecHandle) {
    let mut graph_exec = ptr::null_mut();
    // SAFETY-REVIEWED: CUDA initializes the executable graph slot from a live graph.
    let result = unsafe { raw::cuGraphInstantiateWithFlags(&mut graph_exec, graph, 0) };
    (result_code(result), graph_exec)
}

pub fn graph_destroy(graph: GraphHandle) -> c_int {
    // SAFETY-REVIEWED: Graph owns this initialized handle exactly once.
    result_code(unsafe { raw::cuGraphDestroy(graph) })
}

pub fn graph_exec_destroy(graph_exec: GraphExecHandle) -> c_int {
    // SAFETY-REVIEWED: GraphExec owns this initialized handle exactly once.
    result_code(unsafe { raw::cuGraphExecDestroy(graph_exec) })
}

pub fn graph_launch(graph_exec: GraphExecHandle, stream: StreamHandle) -> c_int {
    // SAFETY-REVIEWED: both handles share a live context checked by the safe layer.
    result_code(unsafe { raw::cuGraphLaunch(graph_exec, stream) })
}

pub fn error_text(error: c_int) -> (String, String) {
    let mut name = ptr::null();
    let mut description = ptr::null();
    let raw_error = error as raw::CUresult;
    // SAFETY-REVIEWED: CUDA writes borrowed pointers to static error strings.
    let name_result = unsafe { raw::cuGetErrorName(raw_error, &mut name) };
    // SAFETY-REVIEWED: CUDA writes a borrowed pointer to a static error string.
    let description_result = unsafe { raw::cuGetErrorString(raw_error, &mut description) };
    (
        read_error_string(result_code(name_result), name, "CUDA_ERROR_UNKNOWN"),
        read_error_string(
            result_code(description_result),
            description,
            "unknown CUDA error",
        ),
    )
}

fn read_error_string(result: c_int, pointer: *const c_char, fallback: &str) -> String {
    if result != CUDA_SUCCESS || pointer.is_null() {
        return fallback.to_owned();
    }
    // SAFETY-REVIEWED: successful CUDA error lookup returns a static NUL-terminated string.
    unsafe { CStr::from_ptr(pointer) }
        .to_string_lossy()
        .into_owned()
}

const fn result_code(result: raw::CUresult) -> c_int {
    result as c_int
}

pub fn device_uuid(ordinal: c_int) -> (c_int, [u8; 16]) {
    let mut bytes = [0; 16];
    // SAFETY-REVIEWED: the C helper writes exactly 16 UUID bytes.
    let result = unsafe { runtime_cuda_device_uuid(bytes.as_mut_ptr(), ordinal) };
    (result_code(result), bytes)
}

pub fn memory_info() -> (c_int, usize, usize) {
    let mut free_bytes = 0;
    let mut total_bytes = 0;
    // SAFETY-REVIEWED: the C helper writes both owned local counters.
    let result = unsafe { runtime_cuda_memory_info(&mut free_bytes, &mut total_bytes) };
    (result_code(result), free_bytes, total_bytes)
}

pub fn event_query(event: EventHandle) -> c_int {
    // SAFETY-REVIEWED: Event owns a live driver event in the current context.
    result_code(unsafe { cuEventQuery(event) })
}

pub fn pinned_address(allocation: &PinnedAllocation, offset: usize) -> DevicePointer {
    // No slice is formed over bytes that DMA may be writing in another page.
    allocation.pointer as usize as u64 + offset as u64
}

pub fn copy_pages(
    destinations: &mut [DevicePointer],
    sources: &mut [DevicePointer],
    sizes: &mut [usize],
    stream: StreamHandle,
) -> c_int {
    assert_eq!(destinations.len(), sources.len());
    assert_eq!(destinations.len(), sizes.len());
    // SAFETY-REVIEWED: PageCopyArena bounds-checks and retains all allocations
    // until the event completes or its stream is fenced, including on errors.
    result_code(unsafe {
        runtime_cuda_copy_pages(
            destinations.as_mut_ptr(),
            sources.as_mut_ptr(),
            sizes.as_mut_ptr(),
            sizes.len(),
            stream,
        )
    })
}
