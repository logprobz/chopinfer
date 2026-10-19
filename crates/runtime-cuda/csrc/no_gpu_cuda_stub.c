#include <stdlib.h>
#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

enum {
  FAKE_CONTEXT_SET_CURRENT = 1,
  FAKE_EVENT_DESTROY = 2,
  FAKE_STREAM_DESTROY = 3,
  FAKE_MEMORY_FREE = 4,
  FAKE_STREAM_SYNCHRONIZE = 5,
  FAKE_ENCODE_TENSOR_MAP = 6,
  FAKE_STREAM_END_CAPTURE = 19,
  FAKE_GRAPH_INSTANTIATE = 20,
  FAKE_LAUNCH_COOPERATIVE = 28,
  FAKE_MAX_ACTIVE_CLUSTERS = 32,
  FAKE_GRAPH_EXEC_DESTROY = 33,
};

/* A fixed synthetic device footprint. These are not real device figures; they
   let device-free tests observe the same deltas a real measurement reports. */
#define FAKE_DEVICE_TOTAL_BYTES ((size_t)8 << 30)
#define FAKE_CONTEXT_BYTES ((size_t)512 << 20)
#define FAKE_MODULE_BYTES ((size_t)64 << 10)

static _Thread_local int failure_operation;
static _Thread_local int failure_code;
static _Thread_local int denied_device = -1;
static _Thread_local int denied_peer = -1;
static _Thread_local size_t call_counts[36];
static _Thread_local void* current_context;
static _Thread_local unsigned next_stream;
static _Thread_local unsigned char peer_edges[8][8];
static _Thread_local unsigned char pool_edges[8][8];
static _Thread_local void* owned_allocations[1024];
static _Thread_local size_t owned_allocation_bytes[1024];
/* A stream-ordered allocation exists only once its stream reaches it. Until
   that stream is synchronized, only work on the same stream may touch it. */
static _Thread_local void* owned_allocation_streams[1024];
static _Thread_local unsigned char owned_allocation_pending[1024];
static _Thread_local size_t ordering_violations;
/* When set, copies land at the API call rather than in stream order, as a
   copy that reads its source during the call may. Such a write is unordered
   against every stream, its own included. */
static _Thread_local int api_call_copies;
static _Thread_local void* owned_events[1024];
static _Thread_local size_t device_used_bytes;
static _Thread_local size_t retained_contexts;
static _Thread_local size_t loaded_modules;
static _Thread_local float fake_clock;

static int remember(void** slots, void* value) {
  for (size_t i = 0; i < 1024; ++i) {
    if (slots[i] == NULL) { slots[i] = value; return 0; }
  }
  free(value);
  return 2;
}
static void forget_owned(void** slots, void* value) {
  for (size_t i = 0; i < 1024; ++i) {
    if (slots[i] == value) { free(value); slots[i] = NULL; return; }
  }
}
static int remember_allocation(void* value, size_t bytes, void* stream) {
  for (size_t i = 0; i < 1024; ++i) {
    if (owned_allocations[i] == NULL) {
      owned_allocations[i] = value;
      owned_allocation_bytes[i] = bytes;
      owned_allocation_streams[i] = stream;
      owned_allocation_pending[i] = 1;
      device_used_bytes += bytes;
      return 0;
    }
  }
  free(value);
  return 2;
}
static void forget_allocation(void* value) {
  for (size_t i = 0; i < 1024; ++i) {
    if (owned_allocations[i] == value) {
      device_used_bytes -= owned_allocation_bytes[i];
      owned_allocation_bytes[i] = 0;
      owned_allocation_streams[i] = NULL;
      owned_allocation_pending[i] = 0;
      free(value);
      owned_allocations[i] = NULL;
      return;
    }
  }
}

/* Counts one write from `stream` into an allocation its own stream has not
   yet been synchronized past: from another stream, or from any stream when
   copies land at the API call. Mapped host memory is not tracked here. */
static void check_ordered_write(uint64_t destination, void* stream) {
  for (size_t i = 0; i < 1024; ++i) {
    uintptr_t base = (uintptr_t)owned_allocations[i];
    if (base == 0 || destination < base ||
        destination >= base + owned_allocation_bytes[i]) {
      continue;
    }
    if (owned_allocation_pending[i] &&
        (api_call_copies || owned_allocation_streams[i] != stream)) {
      ++ordering_violations;
    }
    return;
  }
}
static void complete_allocations(void* stream, int every_stream) {
  for (size_t i = 0; i < 1024; ++i) {
    if (every_stream || owned_allocation_streams[i] == stream) {
      owned_allocation_pending[i] = 0;
    }
  }
}

static const char fake_error_name[] = "CUDA_ERROR_FAKE";
static const char fake_error_string[] = "injected no-device failure";

static int result_for(int operation) {
  ++call_counts[operation];
  if (failure_operation == operation) {
    failure_operation = 0;
    return failure_code;
  }
  return 0;
}

void runtime_cuda_fake_reset(void) {
  next_stream = 0;
  memset(peer_edges, 0, sizeof(peer_edges));
  memset(pool_edges, 0, sizeof(pool_edges));
  failure_operation = 0;
  failure_code = 0;
  denied_device = -1;
  denied_peer = -1;
  current_context = NULL;
  device_used_bytes = 0;
  retained_contexts = 0;
  loaded_modules = 0;
  ordering_violations = 0;
  api_call_copies = 0;
  for (size_t index = 0; index < sizeof(call_counts) / sizeof(call_counts[0]);
       ++index) {
    call_counts[index] = 0;
  }
}

void runtime_cuda_fake_fail_next(int operation, int code) {
  failure_operation = operation;
  failure_code = code;
}

size_t runtime_cuda_fake_call_count(int operation) {
  if (operation < 0 || (size_t)operation >=
                           sizeof(call_counts) / sizeof(call_counts[0])) {
    return 0;
  }
  return call_counts[operation];
}

size_t runtime_cuda_fake_ordering_violations(void) {
  return ordering_violations;
}

void runtime_cuda_fake_set_api_call_copies(int enabled) {
  api_call_copies = enabled != 0;
}

int cuCtxSetCurrent(void* context) {
  int result = result_for(FAKE_CONTEXT_SET_CURRENT);
  if (result == 0) current_context = context;
  return result;
}

int cuGetErrorName(unsigned int error, const char** name) {
  (void)error;
  *name = fake_error_name;
  return 0;
}

int cuGetErrorString(unsigned int error, const char** description) {
  (void)error;
  *description = fake_error_string;
  return 0;
}

int cuDevicePrimaryCtxRelease_v2(int value) {
  (void)value;
  return 0;
}

int cuEventDestroy_v2(void* event) {
  int result = result_for(FAKE_EVENT_DESTROY);
  if (result == 0) forget_owned(owned_events, event);
  return result;
}

int cuMemFreeAsync(uint64_t pointer, void* stream) {
  (void)pointer;
  (void)stream;
  int result = current_context == NULL ? 201 : result_for(FAKE_MEMORY_FREE);
  if (result == 0) forget_allocation((void*)(uintptr_t)pointer);
  return result;
}

int cuMemcpyPeerAsync(
    uint64_t destination,
    void* destination_context,
    uint64_t source,
    void* source_context,
    size_t bytes,
    void* stream) {
  (void)source_context;
  if (current_context != destination_context) return 201;
  int result = result_for(18);
  if (result) return result;
  check_ordered_write(destination, stream);
  memcpy((void*)(uintptr_t)destination, (void*)(uintptr_t)source, bytes);
  fake_clock += 10;
  return 0;
}

int cuStreamDestroy_v2(void* stream) {
  (void)stream;
  return result_for(FAKE_STREAM_DESTROY);
}

int cuStreamSynchronize(void* stream) {
  int result = result_for(FAKE_STREAM_SYNCHRONIZE);
  if (result == 0) complete_allocations(stream, 0);
  return result;
}

/* The last tensor map this stub was asked to encode. */
static uint64_t encoded_address;
static uint32_t encoded_data_type;
static uint32_t encoded_swizzle;
static uint64_t encoded_global_dims[2];
static uint64_t encoded_stride;
static uint32_t encoded_box_dims[2];

void runtime_cuda_fake_encoded_tensor_map(
    uint64_t* address,
    uint32_t* data_type,
    uint32_t* swizzle,
    uint64_t* global_dims,
    uint64_t* stride,
    uint32_t* box_dims) {
  *address = encoded_address;
  *data_type = encoded_data_type;
  *swizzle = encoded_swizzle;
  global_dims[0] = encoded_global_dims[0];
  global_dims[1] = encoded_global_dims[1];
  *stride = encoded_stride;
  box_dims[0] = encoded_box_dims[0];
  box_dims[1] = encoded_box_dims[1];
}

typedef struct runtime_cuda_tensor_map_descriptor {
  uint32_t rank;
  uint32_t data_type;
  uint32_t swizzle;
  uint32_t interleave;
  uint32_t l2_promotion;
  uint32_t oob_fill;
  uint64_t global_dims[5];
  uint64_t global_strides_bytes[4];
  uint32_t box_dims[5];
  uint32_t element_strides[5];
} runtime_cuda_tensor_map_descriptor;

int runtime_cuda_validate_tensor_map(
    const runtime_cuda_tensor_map_descriptor* descriptor,
    uint64_t address,
    uint64_t buffer_bytes) {
  return descriptor != NULL && address != 0 && buffer_bytes != 0 ? 0 : 1;
}

/* The no-device stand-in records the requested descriptor and address. */
int runtime_cuda_encode_tensor_map(
    void* output,
    const runtime_cuda_tensor_map_descriptor* descriptor,
    uint64_t address,
    uint64_t buffer_bytes) {
  /* Counted first, so a caller that reached this boundary is recorded whether
     or not the descriptor it brought was one this stub accepts. */
  int result = result_for(FAKE_ENCODE_TENSOR_MAP);
  if (result != 0) return result;
  if (output == NULL || runtime_cuda_validate_tensor_map(descriptor, address, buffer_bytes) != 0)
    return 1;
  encoded_address = address;
  encoded_data_type = descriptor->data_type;
  encoded_swizzle = descriptor->swizzle;
  encoded_global_dims[0] = descriptor->global_dims[0];
  encoded_global_dims[1] = descriptor->global_dims[1];
  encoded_stride = descriptor->global_strides_bytes[0];
  encoded_box_dims[0] = descriptor->box_dims[0];
  encoded_box_dims[1] = descriptor->box_dims[1];
  /* The caller's storage is opaque to it, so writing the address back is
     enough to prove the encoding landed where the caller asked. */
  *(uint64_t*)output = address;
  return 0;
}

struct fake_event { unsigned flags; float timestamp; int recorded; void* stream; };
int cuStreamWaitEvent(void* stream, void* handle, unsigned int flags) {
  (void)stream; (void)flags;
  int result = result_for(35); if (result) return result;
  if (current_context == NULL) return 201;
  struct fake_event* event = handle;
  if (event->recorded) complete_allocations(event->stream, 0);
  return 0;
}

int cuLaunchHostFunc(void* stream, void (*function)(void*), void* data) {
  (void)stream;
  int result = result_for(34); if (result) return result;
  function(data);
  return 0;
}

int cuEventCreate(void** output, unsigned flags) {
  struct fake_event* event = calloc(1, sizeof(*event));
  if (event == NULL) return 2;
  event->flags = flags;
  *output = event;
  return remember(owned_events, event);
}
int cuEventRecord(void* handle, void* stream) {
  (void)stream;
  int result = result_for(30); if (result) return result;
  struct fake_event* event = handle;
  event->recorded = 1;
  event->stream = stream;
  event->timestamp = fake_clock;
  fake_clock += 2.0f;
  return 0;
}
int cuEventSynchronize(void* handle) { (void)handle; return 0; }
int runtime_cuda_event_elapsed(float* ms, void* start, void* end) {
  struct fake_event* a = start;
  struct fake_event* b = end;
  if ((a->flags & 2) || (b->flags & 2) || !a->recorded || !b->recorded) return 1;
  *ms = b->timestamp - a->timestamp;
  return 0;
}

int cuStreamCreate(void** stream, unsigned flags) {
  (void)flags; int result = result_for(27); if (result) return result;
  *stream = (void*)(uintptr_t)(0x3000 + next_stream++); return 0;
}
int cuStreamBeginCapture_v2(void* stream, unsigned mode) {
  (void)stream; (void)mode; return 0;
}
int cuStreamEndCapture(void* stream, void** graph) {
  (void)stream;
  int result = result_for(FAKE_STREAM_END_CAPTURE);
  *graph = result ? NULL : (void*)0x500;
  return result;
}
int cuGraphInstantiateWithFlags(void** executable, void* graph, unsigned long long flags) {
  (void)graph; (void)flags;
  int result = result_for(FAKE_GRAPH_INSTANTIATE);
  *executable = result ? NULL : (void*)0x501;
  return result;
}
int cuGraphDestroy(void* graph) { (void)graph; return 0; }
int cuGraphExecDestroy(void* executable) {
  (void)executable;
  return result_for(FAKE_GRAPH_EXEC_DESTROY);
}
int cuGraphLaunch(void* executable, void* stream) {
  (void)executable; (void)stream; return 0;
}

/* Emulate numeric CUDA visibility over eight synthetic GPUs. No driver or
   device file is opened. UUID and MIG filtering remain real-driver checks. */
static int visible_devices(int* physical) {
  const char* value = getenv("CUDA_VISIBLE_DEVICES");
  if (value == NULL) {
    for (int i = 0; i < 8; ++i) physical[i] = i;
    return 8;
  }
  int count = 0;
  while (*value != '\0' && count < 8) {
    char* end;
    long ordinal = strtol(value, &end, 10);
    if (end == value || ordinal < 0 || ordinal >= 8) break;
    if (*end != ',' && *end != '\0') break;
    for (int i = 0; i < count; ++i) if (physical[i] == ordinal) return count;
    physical[count++] = (int)ordinal;
    if (*end == '\0') break;
    value = end + 1;
  }
  return count;
}

int cuDeviceGetCount(int* count) {
  int physical[8];
  *count = visible_devices(physical);
  return 0;
}

int cuDeviceGetPCIBusId(char* bus_id, int len, int device) {
  int physical[8];
  int count = visible_devices(physical);
  if (device < 0 || device >= count || len < 16) return 101;
  snprintf(bus_id, (size_t)len, "0000:%02x:00.0", physical[device] + 1);
  return 0;
}

int cuDeviceCanAccessPeer(int* accessible, int device, int peer) {
  int result = result_for(21);
  if (result) return result;
  *accessible = device != peer && !(device == denied_device && peer == denied_peer);
  return 0;
}

void runtime_cuda_fake_deny_peer_access(int device, int peer) {
  denied_device = device;
  denied_peer = peer;
}

int cuInit(unsigned flags) { (void)flags; return result_for(15); }
int cuDriverGetVersion(int* version) { *version = 13020; return 0; }
int cuDeviceGet(int* device, int ordinal) { *device = ordinal; return 0; }
int cuDeviceGetAttribute(int* value, unsigned attribute, int device) {
  (void)device; *value = attribute == 75 ? 12 : attribute == 76 ? 0 : 1; return 0;
}
int cuDevicePrimaryCtxRetain(void** context, int device) {
  int result = result_for(22);
  if (result) return result;
  *context = (void*)(uintptr_t)(device + 0x100); ++retained_contexts; return 0;
}
int cuDeviceGetDefaultMemPool(void** pool, int device) { *pool = (void*)(uintptr_t)(0x400 + device); return 0; }
int cuMemAllocFromPoolAsync(uint64_t* pointer, size_t bytes, void* pool, void* stream) {
  (void)pool; int result = result_for(7);
  if (result) return result;
  *pointer = (uintptr_t)calloc(1, bytes);
  return *pointer ? remember_allocation((void*)(uintptr_t)*pointer, bytes, stream) : 2;
}
int cuMemcpyHtoDAsync_v2(uint64_t dst, const void* src, size_t bytes, void* stream) {
  int result = result_for(8); if (result) return result;
  check_ordered_write(dst, stream);
  memcpy((void*)(uintptr_t)dst, src, bytes); fake_clock += 10; return 0;
}
int cuMemcpyDtoDAsync_v2(uint64_t dst, uint64_t src, size_t bytes, void* stream) {
  int result = result_for(9); if (result) return result;
  check_ordered_write(dst, stream);
  memcpy((void*)(uintptr_t)dst, (void*)(uintptr_t)src, bytes); fake_clock += 10; return 0;
}
int cuMemcpyDtoHAsync_v2(void* dst, uint64_t src, size_t bytes, void* stream) {
  (void)stream; int result = result_for(10); if (result) return result;
  memcpy(dst, (void*)(uintptr_t)src, bytes); fake_clock += 10; return 0;
}
int cuModuleLoadData(void** module, const void* code) {
  int result = result_for(11); if (result) return result;
  ++loaded_modules;
  *module = (void*)(uintptr_t)(*(const unsigned char*)code); return 0;
}
int cuModuleGetFunction(void** function, void* module, const char* name) {
  (void)name; int result = result_for(12); if (result) return result;
  *function = module; return 0;
}
int cuModuleUnload(void* module) {
  (void)module;
  int result = result_for(13);
  if (result == 0 && loaded_modules) --loaded_modules;
  return result;
}
static float from_bf16(uint16_t value) { uint32_t bits = (uint32_t)value << 16; float result; memcpy(&result, &bits, 4); return result; }
static uint16_t to_bf16(float value) { uint32_t bits; memcpy(&bits, &value, 4); return (bits + 0x7fff + ((bits >> 16) & 1)) >> 16; }
int cuLaunchKernel(void* function, unsigned gx, unsigned gy, unsigned gz,
    unsigned bx, unsigned by, unsigned bz, unsigned shared, void* stream, void** args, void** extra) {
  (void)gx; (void)gy; (void)gz; (void)bx; (void)by; (void)bz; (void)shared; (void)stream; (void)extra;
  int result = result_for(14); if (result) return result;
  uint16_t* input = (void*)(uintptr_t)*(uint64_t*)args[0];
  uint16_t* weight = (void*)(uintptr_t)*(uint64_t*)args[1];
  unsigned rows = *(unsigned*)args[2], hidden = *(unsigned*)args[3];
  float epsilon; memcpy(&epsilon, args[4], 4);
  uint16_t* output = (void*)(uintptr_t)*(uint64_t*)args[5];
  if ((uintptr_t)function == 'F') return 700;
  if ((uintptr_t)function == 'H') {
    signal(SIGTERM, SIG_IGN);
    for (;;) pause();
  }
  if ((uintptr_t)function != 'N') {
    for (unsigned row = 0; row < rows; ++row) {
      float sum = 0;
      for (unsigned i = 0; i < hidden; ++i) { float x = from_bf16(input[row*hidden+i]); sum += x*x; }
      float scale = 1.0f / sqrtf(sum / hidden + epsilon);
      for (unsigned i = 0; i < hidden; ++i)
        output[row*hidden+i] = to_bf16(from_bf16(to_bf16(from_bf16(input[row*hidden+i])*scale))*from_bf16(weight[i]));
    }
  }
  if ((uintptr_t)function == 'W') memset(output, 0, rows * hidden * sizeof(uint16_t));
  /* Every fake candidate corrupts working operands after producing its output. */
  input[0] = 0x7fc0; weight[0] = 0x7fc0;
  fake_clock += 3;
  return 0;
}
int runtime_cuda_launch_kernel_cluster(void* function, unsigned gx, unsigned gy, unsigned gz,
    unsigned bx, unsigned by, unsigned bz, unsigned shared, unsigned cx, unsigned cy, unsigned cz,
    int pdl, void* stream, void** parameters) {
  (void)function; (void)gx; (void)gy; (void)gz; (void)bx; (void)by; (void)bz;
  (void)shared; (void)cx; (void)cy; (void)cz; (void)pdl; (void)stream; (void)parameters;
  return 1;
}
/* Runs nothing; counts the launch so tests see a cooperative dispatch. */
int runtime_cuda_launch_kernel_cooperative(void* function, unsigned gx, unsigned gy, unsigned gz,
    unsigned bx, unsigned by, unsigned bz, unsigned shared, unsigned cx, unsigned cy, unsigned cz,
    void* stream, void** parameters) {
  (void)function; (void)gx; (void)gy; (void)gz; (void)bx; (void)by; (void)bz;
  (void)shared; (void)cx; (void)cy; (void)cz; (void)stream; (void)parameters;
  return result_for(FAKE_LAUNCH_COOPERATIVE);
}
int runtime_cuda_host_alloc(void** address, size_t bytes) {
  if (!address || !bytes) return 1;
  int result = result_for(16);
  if (result) return result;
  *address = calloc(1, bytes);
  return *address ? 0 : 2;
}
int runtime_cuda_host_alloc_mapped(void** address, uint64_t* device, size_t bytes) {
  if (!address || !device || !bytes) return 1;
  int result = result_for(16);
  if (result) return result;
  *address = calloc(1, bytes);
  *device = (uint64_t)(uintptr_t)*address;
  return *address ? 0 : 2;
}
int runtime_cuda_host_free(void* address) {
  int result = result_for(17);
  if (result) return result;
  free(address);
  return 0;
}
int runtime_cuda_context_synchronize(void) {
  int result = result_for(19);
  if (result == 0) complete_allocations(NULL, 1);
  return result;
}
int cuLibraryUnload(void* library) { (void)library; return 0; }

int cuDeviceGetUuid_v2(void* uuid, int device) {
    (void)device;
    memset(uuid, 0x11, 16);
    return 0;
}
int runtime_cuda_device_uuid(unsigned char* bytes, int ordinal) {
    int physical[8];
    int count = visible_devices(physical);
    if (ordinal < 0 || ordinal >= count) return 101;
    memset(bytes, physical[ordinal] + 1, 16);
    return 0;
}

/* Report the synthetic footprint this stub actually tracks. A real observation
   comes from cuMemGetInfo and is never an allocation guarantee. */
int runtime_cuda_memory_info(size_t* free_bytes, size_t* total_bytes) {
    if (free_bytes == NULL || total_bytes == NULL) return 1;
    if (current_context == NULL) return 201;
    size_t used = device_used_bytes + loaded_modules * FAKE_MODULE_BYTES +
                  retained_contexts * FAKE_CONTEXT_BYTES;
    *total_bytes = FAKE_DEVICE_TOTAL_BYTES;
    *free_bytes = used > FAKE_DEVICE_TOTAL_BYTES ? 0 : FAKE_DEVICE_TOTAL_BYTES - used;
    return 0;
}

/* The no-device build carries no NVDEC library, so it takes the same refusal
 * path as a host whose driver ships no libnvcuvid. */
int runtime_cuda_nvdec_session_open(int device_ordinal, void** session,
                                    unsigned int* driver_status) {
  (void)device_ordinal;
  if (session != NULL) *session = NULL;
  if (driver_status != NULL) *driver_status = 0;
  return 1;
}

void runtime_cuda_nvdec_session_close(void* session) { (void)session; }

int runtime_cuda_nvdec_h264_capacity(void* session, uint32_t* max_width,
                                     uint32_t* max_height,
                                     unsigned int* driver_status) {
  (void)session;
  (void)max_width;
  (void)max_height;
  (void)driver_status;
  return 7;
}

int runtime_cuda_nvdec_decode_h264(void* session, const uint8_t* stream,
                                   const uint64_t* unit_ends,
                                   const int64_t* timestamps, uint64_t units,
                                   uint32_t max_width, uint32_t max_height,
                                   void* user, void* select, void* frame,
                                   unsigned int* driver_status) {
  (void)session;
  (void)stream;
  (void)unit_ends;
  (void)timestamps;
  (void)units;
  (void)max_width;
  (void)max_height;
  (void)user;
  (void)select;
  (void)frame;
  (void)driver_status;
  return 7;
}

/* Host tests observe physical device edges and reject self access. */
int cuCtxEnablePeerAccess(void* peer, unsigned int flags) {
  (void)flags;
  uintptr_t owner = (uintptr_t)current_context - 0x100;
  uintptr_t target = (uintptr_t)peer - 0x100;
  int result = result_for(23); if (result) return result;
  if (owner >= 8 || target >= 8 || owner == target) return 1;
  if (peer_edges[owner][target]) return 704;
  peer_edges[owner][target] = 1; return 0;
}
int cuCtxDisablePeerAccess(void* peer) {
  uintptr_t owner = (uintptr_t)current_context - 0x100;
  uintptr_t target = (uintptr_t)peer - 0x100;
  int result = result_for(24); if (result) return result;
  if (owner >= 8 || target >= 8 || !peer_edges[owner][target]) return 705;
  peer_edges[owner][target] = 0; return 0;
}
int runtime_cuda_memory_pool_set_device_access(void* pool, int device, int read_write) {
  uintptr_t owner = (uintptr_t)pool - 0x400;
  int result = result_for(25); if (result) return result;
  if (owner >= 8 || device < 0 || device >= 8 || owner == (uintptr_t)device) return 1;
  pool_edges[owner][device] = read_write != 0; return 0;
}
int runtime_cuda_memory_pool_get_device_access(void* pool, int device, int* read_write) {
  uintptr_t owner = (uintptr_t)pool - 0x400;
  int result = result_for(26); if (result) return result;
  if (owner >= 8 || device < 0 || device >= 8 || read_write == NULL) return 1;
  *read_write = owner == (uintptr_t)device || pool_edges[owner][device]; return 0;
}
int runtime_cuda_set_max_dynamic_shared(void* function, int bytes) {
  (void)function; (void)bytes; return 801;
}
/* The occupancy query checks its arguments as the driver shim does and
   answers 352 CTAs (two per SM of 176) in clusters of the asked size, so a
   caller's cluster shape shows in the count. */
int runtime_cuda_max_active_clusters(void* function, unsigned gx, unsigned gy, unsigned gz,
    unsigned bx, unsigned by, unsigned bz, unsigned shared, unsigned cx, unsigned cy,
    unsigned cz, int cooperative, int* clusters) {
  (void)shared; (void)cooperative;
  if (clusters) *clusters = 0;
  if (function == NULL || clusters == NULL || gx == 0 || gy == 0 || gz == 0 ||
      bx == 0 || by == 0 || bz == 0 || cx == 0 || cy == 0 || cz == 0 ||
      gx % cx != 0 || gy % cy != 0 || gz % cz != 0) {
    return 1;
  }
  int result = result_for(FAKE_MAX_ACTIVE_CLUSTERS);
  if (result) return result;
  *clusters = (int)(352 / (cx * cy * cz));
  return 0;
}
int runtime_cuda_context_sm_count(int* sms) {
  if (sms) *sms = 0;
  return 801;
}
int cuMemsetD8Async(uint64_t dst, unsigned char value, size_t bytes, void* stream) {
  (void)dst; (void)value; (void)bytes; (void)stream; return 801;
}
int runtime_cuda_launch_kernel_pdl(void* function, unsigned gx, unsigned gy, unsigned gz,
    unsigned bx, unsigned by, unsigned bz, unsigned shared, void* stream, void** parameters) {
  (void)function; (void)gx; (void)gy; (void)gz; (void)bx; (void)by; (void)bz;
  (void)shared; (void)stream; (void)parameters; return 801;
}

int cuEventQuery(void* handle) {
  int result = result_for(31); if (result) return result;
  struct fake_event* event = handle;
  if (!event->recorded) return 0;
  if (event->recorded == 1) { event->recorded = 2; return 600; }
  return 0;
}
int runtime_cuda_copy_pages(uint64_t* dst, uint64_t* src, size_t* sizes,
                            size_t count, void* stream) {
  int result = result_for(29);
  /* A failed batch may already have copied a prefix of the list. */
  for (size_t i = 0; i < count; ++i) {
    check_ordered_write(dst[i], stream);
    memcpy((void*)(uintptr_t)dst[i], (void*)(uintptr_t)src[i], sizes[i]);
    if (result) return result;
  }
  return result;
}
