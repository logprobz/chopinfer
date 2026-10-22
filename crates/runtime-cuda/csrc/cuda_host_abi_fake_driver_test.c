// The shared CUDA host ABI, driven against a fake CUDA Driver API.
//
// Nothing here knows a model. What it checks is the generic surface the shared
// host owns: the parameter contract both checked launchers enforce before any
// driver call, the launch attribute the programmatic-dependent launcher asks
// for, the rank-1-to-5 tensor-map encoding, peer and local copy order, and
// the stream wait. The model launch closures that used to be checked here left
// with the bundle that owns them, and their argument order is proved against a
// fake host there instead.

#include "tensor_map.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MAX_PARAMETERS 18
#define MAX_PARAMETER_BYTES 128

typedef struct {
  const char* symbol;
  unsigned int grid[3];
  unsigned int block[3];
  unsigned int shared_memory_bytes;
  int programmatic;
  /* The cluster the launch asked for, and whether it asked for one at all. A
     launch with no cluster attribute leaves both untouched, so a test cannot
     read a cluster out of one that carried none. */
  int clustered;
  unsigned int cluster[3];
  size_t parameter_count;
  uint8_t widths[MAX_PARAMETERS];
  unsigned char parameters[MAX_PARAMETERS][MAX_PARAMETER_BYTES];
} launch_record;

static launch_record records[32];
static size_t record_count;
static size_t peer_copy_count;
static CUdeviceptr peer_copy_destination;
static CUcontext peer_copy_destination_context;
static CUdeviceptr peer_copy_source;
static CUcontext peer_copy_source_context;
static size_t peer_copy_bytes;
static CUstream peer_copy_stream;
static size_t stream_wait_count;
static CUstream waited_stream;
static CUevent waited_event;
static unsigned int waited_flags;
static CUcontext current_context;
static size_t local_copy_count;
static int configured_shared_memory_bytes;
static size_t tensor_map_encode_count;
static uint32_t encoded_rank;
static uint64_t encoded_dimensions[5];
static uint64_t encoded_strides[4];
static uint32_t encoded_boxes[5];
static uint32_t encoded_element_strides[5];
static size_t pinned_allocation_count;
static size_t pinned_free_count;
static size_t context_sync_count;

CUresult CUDAAPI cuCtxSetCurrent(CUcontext context) {
  current_context = context;
  return CUDA_SUCCESS;
}

static unsigned int last_host_alloc_flags;
static int fail_device_pointer;

CUresult CUDAAPI cuMemHostAlloc(void** address, size_t bytes, unsigned int flags) {
  if (flags != CU_MEMHOSTALLOC_PORTABLE &&
      flags != (CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *address = malloc(bytes);
  if (*address == NULL) return CUDA_ERROR_OUT_OF_MEMORY;
  last_host_alloc_flags = flags;
  ++pinned_allocation_count;
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemHostGetDevicePointer(CUdeviceptr* device, void* address, unsigned int flags) {
  if (flags != 0 || address == NULL) return CUDA_ERROR_INVALID_VALUE;
  if (fail_device_pointer) return CUDA_ERROR_NOT_SUPPORTED;
  *device = (CUdeviceptr)(uintptr_t)address + 0x1000000000ull;
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemFreeHost(void* address) {
  free(address);
  ++pinned_free_count;
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxSynchronize(void) {
  ++context_sync_count;
  return CUDA_SUCCESS;
}

static void require(int condition, const char* message) {
  if (!condition) {
    fprintf(stderr, "%s\n", message);
    exit(1);
  }
}

// The widths the shared parameter contract admits, and nothing else.
static int known_width(uint8_t width) {
  return width == RUNTIME_CUDA_PARAMETER_U32 ||
         width == RUNTIME_CUDA_PARAMETER_U64 ||
         width == RUNTIME_CUDA_PARAMETER_BYTES64 ||
         width == RUNTIME_CUDA_PARAMETER_TENSOR_MAP;
}

/* Reads one launch's attribute list into the flags the record carries.

   Each attribute this host's launchers set may appear exactly once, and no
   other attribute may appear at all: a duplicate or an unknown id is a launch
   descriptor this driver cannot answer for, so it is refused rather than read
   with one of the two values winning. A cluster with a zero extent is refused
   for the same reason the shim refuses it, because a driver would. */
static int read_launch_attributes(
    const CUlaunchConfig* config,
    int* programmatic,
    int* clustered,
    unsigned int cluster[3]) {
  *programmatic = 0;
  *clustered = 0;
  cluster[0] = 0;
  cluster[1] = 0;
  cluster[2] = 0;
  if (config->numAttrs == 0 || config->numAttrs > 2) return 0;
  if (config->attrs == NULL) return 0;
  for (unsigned int index = 0; index < config->numAttrs; ++index) {
    const CUlaunchAttribute* attribute = &config->attrs[index];
    switch ((int)attribute->id) {
      case CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION:
        if (*programmatic) return 0;
        if (!attribute->value.programmaticStreamSerializationAllowed) return 0;
        *programmatic = 1;
        break;
      case CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION:
        if (*clustered) return 0;
        if (attribute->value.clusterDim.x == 0 ||
            attribute->value.clusterDim.y == 0 ||
            attribute->value.clusterDim.z == 0) {
          return 0;
        }
        *clustered = 1;
        cluster[0] = attribute->value.clusterDim.x;
        cluster[1] = attribute->value.clusterDim.y;
        cluster[2] = attribute->value.clusterDim.z;
        break;
      default:
        return 0;
    }
  }
  return 1;
}

static CUresult record_launch(
    CUfunction function,
    unsigned int grid_x,
    unsigned int grid_y,
    unsigned int grid_z,
    unsigned int block_x,
    unsigned int block_y,
    unsigned int block_z,
    unsigned int shared_memory_bytes,
    int programmatic,
    void** parameters,
    const uint8_t* widths,
    size_t parameter_count) {
  if (parameters == NULL || widths == NULL || parameter_count == 0 ||
      parameter_count > MAX_PARAMETERS ||
      record_count >= sizeof(records) / sizeof(records[0])) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  launch_record* record = &records[record_count];
  memset(record, 0, sizeof(*record));
  record->symbol = (const char*)function;
  record->grid[0] = grid_x;
  record->grid[1] = grid_y;
  record->grid[2] = grid_z;
  record->block[0] = block_x;
  record->block[1] = block_y;
  record->block[2] = block_z;
  record->shared_memory_bytes = shared_memory_bytes;
  record->programmatic = programmatic;
  record->parameter_count = parameter_count;
  for (size_t index = 0; index < parameter_count; ++index) {
    if (parameters[index] == NULL || !known_width(widths[index])) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    record->widths[index] = widths[index];
    memcpy(record->parameters[index], parameters[index], widths[index]);
  }
  ++record_count;
  return CUDA_SUCCESS;
}

// The widths the test hands the launcher, kept beside the call so the fake
// driver can record what the real launcher passed through.
static const uint8_t* pending_widths;
static size_t pending_width_count;

CUresult CUDAAPI cuModuleGetFunction(
    CUfunction* output,
    CUmodule module,
    const char* symbol) {
  if (output == NULL || module == NULL || symbol == NULL) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *output = (CUfunction)symbol;
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLaunchKernel(
    CUfunction function,
    unsigned int grid_x,
    unsigned int grid_y,
    unsigned int grid_z,
    unsigned int block_x,
    unsigned int block_y,
    unsigned int block_z,
    unsigned int shared_memory_bytes,
    CUstream stream,
    void** parameters,
    void** extra) {
  (void)stream;
  (void)extra;
  return record_launch(function, grid_x, grid_y, grid_z, block_x, block_y,
                       block_z, shared_memory_bytes, 0, parameters,
                       pending_widths, pending_width_count);
}

CUresult CUDAAPI cuLaunchKernelEx(
    const CUlaunchConfig* config,
    CUfunction function,
    void** parameters,
    void** extra) {
  (void)extra;
  int programmatic = 0;
  int clustered = 0;
  unsigned int cluster[3] = {0, 0, 0};
  if (config == NULL ||
      !read_launch_attributes(config, &programmatic, &clustered, cluster)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUresult result = record_launch(
      function, config->gridDimX, config->gridDimY, config->gridDimZ,
      config->blockDimX, config->blockDimY, config->blockDimZ,
      config->sharedMemBytes, programmatic, parameters, pending_widths,
      pending_width_count);
  if (result == CUDA_SUCCESS) {
    launch_record* record = &records[record_count - 1];
    record->clustered = clustered;
    record->cluster[0] = cluster[0];
    record->cluster[1] = cluster[1];
    record->cluster[2] = cluster[2];
  }
  return result;
}

CUresult CUDAAPI cuFuncSetAttribute(
    CUfunction function,
    CUfunction_attribute attribute,
    int value) {
  (void)function;
  if (attribute != CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES ||
      value < 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  configured_shared_memory_bytes = value;
  return CUDA_SUCCESS;
}

static unsigned int occupancy_cluster_x;
static unsigned int occupancy_grid_x;
static unsigned int occupancy_grid_y;
static int occupancy_cooperative;

CUresult CUDAAPI cuOccupancyMaxActiveClusters(
    int* clusters,
    CUfunction function,
    const CUlaunchConfig* config) {
  if (clusters == NULL || function == NULL || config == NULL ||
      config->numAttrs < 1 || config->numAttrs > 2 ||
      config->attrs[0].id != CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION ||
      (config->numAttrs == 2 &&
       (config->attrs[1].id != CU_LAUNCH_ATTRIBUTE_COOPERATIVE ||
        config->attrs[1].value.cooperative != 1))) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  occupancy_cluster_x = config->attrs[0].value.clusterDim.x;
  occupancy_grid_x = config->gridDimX;
  occupancy_grid_y = config->gridDimY;
  occupancy_cooperative = config->numAttrs == 2;
  *clusters = 44;
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxGetExecAffinity(
    CUexecAffinityParam* affinity,
    CUexecAffinityType type) {
  if (affinity == NULL || type != CU_EXEC_AFFINITY_TYPE_SM_COUNT) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  affinity->type = type;
  affinity->param.smCount.val = 94;
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemsetD8Async(
    CUdeviceptr destination,
    unsigned char value,
    size_t bytes,
    CUstream stream) {
  if (destination == 0 || value != 0 || bytes == 0 || stream == NULL) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  return CUDA_SUCCESS;
}

typedef struct {
  uint32_t data_type;
  uint32_t rank;
  uint64_t address;
  uint64_t global_dimensions[2];
  uint64_t global_strides[1];
  uint32_t box_dimensions[2];
  uint32_t element_strides[2];
  uint32_t interleave;
  uint32_t swizzle;
  uint32_t l2_promotion;
  uint32_t oob_fill;
} typed_tensor_map_descriptor;

CUresult CUDAAPI cuTensorMapEncodeTiled(
    CUtensorMap* output,
    CUtensorMapDataType data_type,
    cuuint32_t rank,
    void* address,
    const cuuint64_t* global_dimensions,
    const cuuint64_t* global_strides,
    const cuuint32_t* box_dimensions,
    const cuuint32_t* element_strides,
    CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion l2_promotion,
    CUtensorMapFloatOOBfill oob_fill) {
  if (output == NULL || address == NULL || rank < 1 || rank > 5 ||
      global_dimensions == NULL || global_strides == NULL ||
      box_dimensions == NULL || element_strides == NULL) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  ++tensor_map_encode_count;
  encoded_rank = rank;
  for (uint32_t index = 0; index < rank; ++index) {
    encoded_dimensions[index] = global_dimensions[index];
    encoded_boxes[index] = box_dimensions[index];
    encoded_element_strides[index] = element_strides[index];
    if (index + 1 < rank) encoded_strides[index] = global_strides[index];
  }
  typed_tensor_map_descriptor descriptor = {
      .data_type = (uint32_t)data_type,
      .rank = rank,
      .address = (uint64_t)(uintptr_t)address,
      .global_dimensions = {global_dimensions[0], rank > 1 ? global_dimensions[1] : 0},
      .global_strides = {rank > 1 ? global_strides[0] : 0},
      .box_dimensions = {box_dimensions[0], rank > 1 ? box_dimensions[1] : 0},
      .element_strides = {element_strides[0], rank > 1 ? element_strides[1] : 0},
      .interleave = (uint32_t)interleave,
      .swizzle = (uint32_t)swizzle,
      .l2_promotion = (uint32_t)l2_promotion,
      .oob_fill = (uint32_t)oob_fill,
  };
  _Static_assert(sizeof(descriptor) <= sizeof(*output),
                 "typed tensor-map descriptor exceeds CUDA ABI storage");
  memset(output, 0, sizeof(*output));
  memcpy(output, &descriptor, sizeof(descriptor));
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemPoolSetAccess(
    CUmemoryPool pool,
    const CUmemAccessDesc* descriptors,
    size_t count) {
  (void)pool;
  (void)descriptors;
  (void)count;
  return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMemPoolGetAccess(
    CUmemAccess_flags* flags,
    CUmemoryPool pool,
    CUmemLocation* location) {
  (void)flags;
  (void)pool;
  (void)location;
  return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMemcpyBatchAsync(
    CUdeviceptr* destinations,
    CUdeviceptr* sources,
    size_t* sizes,
    size_t count,
    CUmemcpyAttributes* attributes,
    size_t* attribute_indices,
    size_t attribute_count,
    CUstream stream) {
  (void)destinations;
  (void)sources;
  (void)sizes;
  (void)count;
  (void)attributes;
  (void)attribute_indices;
  (void)attribute_count;
  (void)stream;
  return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMemcpyPeerAsync(
    CUdeviceptr destination,
    CUcontext destination_context,
    CUdeviceptr source,
    CUcontext source_context,
    size_t bytes,
    CUstream stream) {
  if (current_context != destination_context) return CUDA_ERROR_INVALID_CONTEXT;
  ++peer_copy_count;
  peer_copy_destination = destination;
  peer_copy_destination_context = destination_context;
  peer_copy_source = source;
  peer_copy_source_context = source_context;
  peer_copy_bytes = bytes;
  peer_copy_stream = stream;
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyDtoDAsync_v2(
    CUdeviceptr destination,
    CUdeviceptr source,
    size_t bytes,
    CUstream stream) {
  if (current_context == NULL || destination == 0 || source == 0 ||
      bytes == 0 || stream == NULL) {
    return CUDA_ERROR_INVALID_CONTEXT;
  }
  ++local_copy_count;
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamWaitEvent(
    CUstream stream,
    CUevent event,
    unsigned int flags) {
  if (current_context == NULL) return CUDA_ERROR_INVALID_CONTEXT;
  ++stream_wait_count;
  waited_stream = stream;
  waited_event = event;
  waited_flags = flags;
  return CUDA_SUCCESS;
}

static void test_peer_copy_and_wait_driver_abi(void) {
  peer_copy_count = 0;
  stream_wait_count = 0;
  local_copy_count = 0;
  current_context = NULL;
  require(cuMemcpyPeerAsync(
              1, (CUcontext)(uintptr_t)2, 3, (CUcontext)(uintptr_t)4,
              5, (CUstream)(uintptr_t)6) == CUDA_ERROR_INVALID_CONTEXT &&
              peer_copy_count == 0,
          "peer copy accepted a non-current destination context");
  require(cuCtxSetCurrent(
              (CUcontext)(uintptr_t)0x5555666677778888ULL) == CUDA_SUCCESS,
          "destination context setup failed");
  CUresult result = cuMemcpyPeerAsync(
      0x1111222233334444ULL,
      (CUcontext)(uintptr_t)0x5555666677778888ULL,
      0x9999aaaabbbbccccULL,
      (CUcontext)(uintptr_t)0xddddeeeeffff0000ULL,
      (size_t)0x12345678U,
      (CUstream)(uintptr_t)0x1020304050607080ULL);
  require(result == CUDA_SUCCESS && peer_copy_count == 1,
          "peer-copy driver call count differs");
  require(peer_copy_destination == 0x1111222233334444ULL &&
              (uintptr_t)peer_copy_destination_context ==
                  0x5555666677778888ULL &&
              peer_copy_source == 0x9999aaaabbbbccccULL &&
              (uintptr_t)peer_copy_source_context ==
                  0xddddeeeeffff0000ULL &&
              peer_copy_bytes == (size_t)0x12345678U &&
              (uintptr_t)peer_copy_stream == 0x1020304050607080ULL,
          "peer-copy driver argument order or width differs");

  result = cuMemcpyDtoDAsync_v2(
      0x1111222233334444ULL, 0x9999aaaabbbbccccULL,
      (size_t)0x12345678U,
      (CUstream)(uintptr_t)0x1020304050607080ULL);
  require(result == CUDA_SUCCESS && local_copy_count == 1,
          "local copy did not use the current destination context");

  result = cuStreamWaitEvent(
      (CUstream)(uintptr_t)0x1234567890abcdefULL,
      (CUevent)(uintptr_t)0xfedcba0987654321ULL,
      0);
  require(result == CUDA_SUCCESS && stream_wait_count == 1 &&
              (uintptr_t)waited_stream == 0x1234567890abcdefULL &&
              (uintptr_t)waited_event == 0xfedcba0987654321ULL &&
              waited_flags == 0,
          "stream-wait driver ABI differs");
}

static void test_checked_launchers_reject_invalid_descriptors(void) {
  uint64_t pointers[4] = {1, 2, 3, 4};
  void* parameters[4] = {&pointers[0], &pointers[1], &pointers[2], &pointers[3]};
  uint8_t valid_widths[4] = {8, 8, 4, 8};
  uint8_t invalid_widths[4] = {8, 16, 4, 8};
  record_count = 0;
  pending_widths = valid_widths;
  pending_width_count = 4;
  require(runtime_cuda_launch_kernel_checked(
              (CUfunction)"chop_checked_launcher_probe",
              1, 1, 1, 1, 1, 1, 0, (CUstream)1, parameters,
              valid_widths, 0) == CUDA_ERROR_INVALID_VALUE,
          "checked launcher accepted a zero parameter count");
  require(runtime_cuda_launch_kernel_checked(
              (CUfunction)"chop_checked_launcher_probe",
              1, 1, 1, 1, 1, 1, 0, (CUstream)1, parameters,
              invalid_widths, 4) == CUDA_ERROR_INVALID_VALUE,
          "checked launcher accepted an unsupported parameter width");
  require(runtime_cuda_launch_kernel_checked(
              (CUfunction)"chop_checked_launcher_probe",
              1, 1, 1, 1, 1, 1, 0, (CUstream)1, NULL,
              valid_widths, 4) == CUDA_ERROR_INVALID_VALUE,
          "checked launcher accepted null parameters");
  // The programmatic-dependent launcher owes the same contract.
  require(runtime_cuda_launch_kernel_pdl_checked(
              (CUfunction)"chop_checked_launcher_probe",
              1, 1, 1, 1, 1, 1, 0, (CUstream)1, parameters,
              invalid_widths, 4) == CUDA_ERROR_INVALID_VALUE,
          "programmatic launcher accepted an unsupported parameter width");
  require(record_count == 0,
          "invalid descriptor reached the fake CUDA driver");
}

static void test_checked_launchers_pass_the_descriptor_through(void) {
  uint64_t pointers[3] = {0x1000, 0x2000, 0x3000};
  uint32_t scalar = 0x44444444U;
  void* parameters[3] = {&pointers[0], &scalar, &pointers[2]};
  uint8_t widths[3] = {8, 4, 8};
  record_count = 0;
  pending_widths = widths;
  pending_width_count = 3;

  require(runtime_cuda_launch_kernel_checked(
              (CUfunction)"chop_checked_launcher_probe",
              7, 5, 3, 128, 2, 1, 64, (CUstream)1, parameters,
              widths, 3) == CUDA_SUCCESS && record_count == 1,
          "the checked launcher did not reach the driver");
  require(records[0].grid[0] == 7 && records[0].grid[1] == 5 &&
              records[0].grid[2] == 3 && records[0].block[0] == 128 &&
              records[0].block[1] == 2 && records[0].block[2] == 1 &&
              records[0].shared_memory_bytes == 64 &&
              records[0].programmatic == 0,
          "the checked launcher altered the launch geometry");
  require(records[0].parameter_count == 3 && records[0].widths[0] == 8 &&
              records[0].widths[1] == 4 && records[0].widths[2] == 8,
          "the checked launcher altered the parameter widths");
  uint64_t first = 0;
  uint32_t second = 0;
  memcpy(&first, records[0].parameters[0], sizeof(first));
  memcpy(&second, records[0].parameters[1], sizeof(second));
  require(first == 0x1000 && second == 0x44444444U,
          "the checked launcher altered the parameter order or values");

  require(runtime_cuda_launch_kernel_pdl_checked(
              (CUfunction)"chop_checked_launcher_probe",
              9, 1, 1, 384, 1, 1, 92928, (CUstream)1, parameters,
              widths, 3) == CUDA_SUCCESS && record_count == 2,
          "the programmatic launcher did not reach the driver");
  require(records[1].programmatic == 1 && records[1].grid[0] == 9 &&
              records[1].block[0] == 384 &&
              records[1].shared_memory_bytes == 92928,
          "the programmatic launcher altered the launch descriptor");
}

static CUresult encode_fixture_tensor_map(
    void* output, uint32_t data_type, CUdeviceptr address,
    uint64_t dim_0, uint64_t dim_1, uint64_t stride_1,
    uint32_t box_0, uint32_t box_1, uint32_t swizzle) {
  runtime_cuda_tensor_map_descriptor descriptor = {0};
  descriptor.rank = 2;
  descriptor.data_type = data_type;
  descriptor.swizzle = swizzle;
  descriptor.l2_promotion = 3;
  descriptor.global_dims[0] = dim_0;
  descriptor.global_dims[1] = dim_1;
  descriptor.global_strides_bytes[0] = stride_1;
  descriptor.box_dims[0] = box_0;
  descriptor.box_dims[1] = box_1;
  descriptor.element_strides[0] = 1;
  descriptor.element_strides[1] = 1;
  return runtime_cuda_encode_tensor_map(output, &descriptor, address,
                                        UINT64_MAX - address);
}

static void test_tensor_map_encoding_and_shared_memory_configuration(void) {
  CUtensorMap map;
  require(encode_fixture_tensor_map(
              &map, 1, 0x9000, 1024, 32, 1024, 128, 64, 1) == CUDA_SUCCESS,
          "a UINT8 128-byte-swizzled encoding was refused");
  typed_tensor_map_descriptor observed;
  memcpy(&observed, &map, sizeof(observed));
  require(observed.data_type == (uint32_t)CU_TENSOR_MAP_DATA_TYPE_UINT8 &&
              observed.rank == 2 && observed.address == 0x9000 &&
              observed.global_dimensions[0] == 1024 &&
              observed.global_dimensions[1] == 32 &&
              observed.global_strides[0] == 1024 &&
              observed.box_dimensions[0] == 128 &&
              observed.box_dimensions[1] == 64 &&
              observed.element_strides[0] == 1 &&
              observed.element_strides[1] == 1 &&
              observed.interleave ==
                  (uint32_t)CU_TENSOR_MAP_INTERLEAVE_NONE &&
              observed.swizzle == (uint32_t)CU_TENSOR_MAP_SWIZZLE_128B &&
              observed.l2_promotion ==
                  (uint32_t)CU_TENSOR_MAP_L2_PROMOTION_L2_256B &&
              observed.oob_fill ==
                  (uint32_t)CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
          "the 2D tensor-map encoding differs from the driver contract");
  // Each element code the shim declares reaches the driver as its own
  // CUtensorMapDataType, by name, so no code is encoded as another.
  static const struct {
    uint32_t code;
    CUtensorMapDataType expected;
  } elements[] = {
      {1, CU_TENSOR_MAP_DATA_TYPE_UINT8},
      {2, CU_TENSOR_MAP_DATA_TYPE_INT32},
      {3, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16},
      {4, CU_TENSOR_MAP_DATA_TYPE_FLOAT16},
      {5, CU_TENSOR_MAP_DATA_TYPE_FLOAT32},
      {6, CU_TENSOR_MAP_DATA_TYPE_UINT8},
      {7, CU_TENSOR_MAP_DATA_TYPE_UINT8},
      {8, CU_TENSOR_MAP_DATA_TYPE_UINT16},
  };
  for (size_t index = 0; index < sizeof(elements) / sizeof(elements[0]);
       ++index) {
    require(encode_fixture_tensor_map(
                &map, elements[index].code, 0x9000, 128, 2, 1024, 64, 1, 0) ==
                CUDA_SUCCESS,
            "a declared tensor-map element type was refused");
    memcpy(&observed, &map, sizeof(observed));
    require(observed.data_type == (uint32_t)elements[index].expected,
            "a tensor-map element type reached the driver as another one");
  }
  // The type set is closed after the three packed codes.
  for (uint32_t code = 12; code <= 13; ++code) {
    require(encode_fixture_tensor_map(
                &map, code, 0x9000, 1, 1, 1, 1, 1, 0) ==
                CUDA_ERROR_INVALID_VALUE,
            "an element type with no CUtensorMapDataType encoded");
  }
  require(encode_fixture_tensor_map(
              &map, 0, 0x9000, 1, 1, 1, 1, 1, 0) == CUDA_ERROR_INVALID_VALUE,
          "the zero element code encoded");
  require(encode_fixture_tensor_map(
              &map, 1, 0x9000, 1, 1, 1, 1, 1, 4) == CUDA_ERROR_INVALID_VALUE,
          "an undeclared tensor-map swizzle encoded");

  configured_shared_memory_bytes = -1;
  require(runtime_cuda_set_max_dynamic_shared(
              (CUfunction)"chop_checked_launcher_probe", 92928) ==
                  CUDA_SUCCESS &&
              configured_shared_memory_bytes == 92928,
          "the dynamic shared-memory configuration differs");

  int clusters = -1;
  require(runtime_cuda_max_active_clusters(
              (CUfunction)"chop_cluster_probe", 136, 1, 1, 256, 1, 1, 0, 8, 1, 1, 0,
              &clusters) == CUDA_SUCCESS &&
              clusters == 44 && occupancy_cluster_x == 8 && occupancy_grid_x == 136 &&
              occupancy_grid_y == 1 && !occupancy_cooperative,
          "the cluster occupancy query differs");
  /* A cooperative grid of two rows asks with its own shape. */
  require(runtime_cuda_max_active_clusters(
              (CUfunction)"chop_cluster_probe", 136, 2, 1, 256, 1, 1, 0, 8, 1, 1, 1,
              &clusters) == CUDA_SUCCESS &&
              occupancy_grid_x == 136 && occupancy_grid_y == 2 && occupancy_cooperative,
          "the cooperative occupancy query differs");
  require(runtime_cuda_max_active_clusters(
              (CUfunction)"chop_cluster_probe", 12, 1, 1, 256, 1, 1, 0, 8, 1, 1, 0,
              &clusters) == CUDA_ERROR_INVALID_VALUE,
          "a grid of partial clusters reached the occupancy query");
  int sms = -1;
  require(runtime_cuda_context_sm_count(&sms) == CUDA_SUCCESS && sms == 94,
          "the context SM count differs");
}

/// A launch whose parameters mix every width the checked launcher admits.
///
/// This is the shape a fixed kernel ABI takes: device pointers, one of them
/// null, fixed-width scalars, and encoded tensor maps, all in one list. What
/// is checked is that each one reaches the driver at its own width and in its
/// own position, with nothing widened, narrowed, or reordered.
static void test_mixed_width_parameters_reach_the_driver_intact(void) {
  uint64_t output = 0x1000;
  uint64_t null_pointer = 0;
  uint64_t workspace = 0x2000;
  uint32_t shape_m = 32;
  uint32_t shape_n = 8192;
  uint32_t shape_k = 1024;
  uint32_t stride_m = 8192;
  uint32_t stride_n = 0;
  uint32_t stride_batch = 0;
  CUtensorMap maps[5];
  for (size_t index = 0; index < 5; ++index) {
    require(encode_fixture_tensor_map(
                &maps[index], 1, 0x3000 + index * 0x1000,
                1024, 32, 1024, 128, 64, 1) == CUDA_SUCCESS,
            "a well formed tensor map failed to encode");
  }
  void* parameters[18] = {
      &output,     &null_pointer, &null_pointer, &null_pointer,
      &null_pointer, &null_pointer, &workspace,  &shape_m,
      &shape_n,    &shape_k,      &stride_m,     &stride_n,
      &stride_batch, &maps[0],    &maps[1],      &maps[2],
      &maps[3],    &maps[4],
  };
  uint8_t widths[18] = {
      RUNTIME_CUDA_PARAMETER_U64, RUNTIME_CUDA_PARAMETER_U64,
      RUNTIME_CUDA_PARAMETER_U64, RUNTIME_CUDA_PARAMETER_U64,
      RUNTIME_CUDA_PARAMETER_U64, RUNTIME_CUDA_PARAMETER_U64,
      RUNTIME_CUDA_PARAMETER_U64, RUNTIME_CUDA_PARAMETER_U32,
      RUNTIME_CUDA_PARAMETER_U32, RUNTIME_CUDA_PARAMETER_U32,
      RUNTIME_CUDA_PARAMETER_U32, RUNTIME_CUDA_PARAMETER_U32,
      RUNTIME_CUDA_PARAMETER_U32, RUNTIME_CUDA_PARAMETER_TENSOR_MAP,
      RUNTIME_CUDA_PARAMETER_TENSOR_MAP, RUNTIME_CUDA_PARAMETER_TENSOR_MAP,
      RUNTIME_CUDA_PARAMETER_TENSOR_MAP, RUNTIME_CUDA_PARAMETER_TENSOR_MAP,
  };
  record_count = 0;
  pending_widths = widths;
  pending_width_count = 18;
  require(runtime_cuda_launch_kernel_pdl_checked(
              (CUfunction)"chop_mixed_width_probe",
              188, 1, 1, 384, 1, 1, 92928, (CUstream)1, parameters,
              widths, 18) == CUDA_SUCCESS && record_count == 1,
          "a mixed-width programmatic launch did not reach the driver");
  const launch_record* record = &records[0];
  require(record->programmatic == 1 && record->parameter_count == 18,
          "the mixed-width launch lost its attribute or its parameters");
  for (size_t index = 0; index < 18; ++index) {
    require(record->widths[index] == widths[index],
            "a mixed-width parameter changed width");
  }
  uint64_t first = 0;
  uint64_t null_slot = 0;
  uint64_t seventh = 0;
  uint32_t eighth = 0;
  memcpy(&first, record->parameters[0], sizeof(first));
  memcpy(&null_slot, record->parameters[1], sizeof(null_slot));
  memcpy(&seventh, record->parameters[6], sizeof(seventh));
  memcpy(&eighth, record->parameters[7], sizeof(eighth));
  require(first == 0x1000 && seventh == 0x2000,
          "a device pointer changed value or position");
  require(null_slot == 0, "a null device pointer did not stay null");
  require(eighth == 32, "a 32-bit scalar changed value or position");
  /* A 64-bit value no 32-bit slot could hold proves nothing narrowed it. */
  uint64_t wide = UINT64_C(0x1122334455667788);
  void* wide_parameters[1] = {&wide};
  uint8_t wide_widths[1] = {RUNTIME_CUDA_PARAMETER_U64};
  record_count = 0;
  pending_widths = wide_widths;
  pending_width_count = 1;
  require(runtime_cuda_launch_kernel_checked(
              (CUfunction)"chop_mixed_width_probe",
              1, 1, 1, 1, 1, 1, 0, (CUstream)1, wide_parameters,
              wide_widths, 1) == CUDA_SUCCESS && record_count == 1,
          "a 64-bit scalar launch did not reach the driver");
  uint64_t observed = 0;
  memcpy(&observed, records[0].parameters[0], sizeof(observed));
  require(records[0].widths[0] == RUNTIME_CUDA_PARAMETER_U64 &&
              observed == wide,
          "a 64-bit scalar was narrowed on the way to the driver");
}

/// The cluster launcher, driven through the shim the Linux build compiles.
///
/// A cluster is a launch attribute, so what has to reach the driver is the
/// attribute list itself: exactly the cluster when no other attribute was
/// asked for, and the cluster beside programmatic stream serialization when one
/// was. All three extents are read back each time, because a launcher that
/// dropped or transposed one would still produce a cluster and a plausible
/// launch.
///
/// This is a device-free harness: it proves the descriptor the shim builds, not
/// that any device forms the cluster it describes. No smoke on a real driver is
/// claimed by anything here.
static void test_cluster_launch_attributes_reach_the_driver(void) {
  uint64_t operand = 0x4000;
  void* parameters[1] = {&operand};
  uint8_t widths[1] = {RUNTIME_CUDA_PARAMETER_U64};
  record_count = 0;
  pending_widths = widths;
  pending_width_count = 1;

  // A cluster with no other attribute: the list is exactly one entry, and the
  // launch is not the programmatic-dependent one.
  require(runtime_cuda_launch_kernel_cluster(
              (CUfunction)"chop_cluster_probe",
              16, 8, 4, 128, 1, 1, 1024,
              4, 2, 1, 0, (CUstream)1, parameters) == CUDA_SUCCESS &&
              record_count == 1,
          "a cluster-only launch did not reach the driver");
  require(records[0].clustered == 1 && records[0].cluster[0] == 4 &&
              records[0].cluster[1] == 2 && records[0].cluster[2] == 1,
          "the cluster-only launch altered a cluster dimension");
  require(records[0].programmatic == 0,
          "a cluster-only launch must ask for no other attribute");
  require(records[0].grid[0] == 16 && records[0].grid[1] == 8 &&
              records[0].grid[2] == 4 && records[0].block[0] == 128 &&
              records[0].shared_memory_bytes == 1024,
          "the cluster launcher altered the launch geometry");

  // A cluster beside programmatic stream serialization: both attributes, each
  // once, and the cluster is still the one that was asked for.
  require(runtime_cuda_launch_kernel_cluster(
              (CUfunction)"chop_cluster_probe",
              8, 8, 8, 256, 1, 1, 0,
              2, 2, 2, 1, (CUstream)1, parameters) == CUDA_SUCCESS &&
              record_count == 2,
          "a cluster launch with the programmatic attribute did not reach the driver");
  require(records[1].clustered == 1 && records[1].cluster[0] == 2 &&
              records[1].cluster[1] == 2 && records[1].cluster[2] == 2 &&
              records[1].programmatic == 1,
          "a cluster launch lost one of the two attributes it asked for");

  // Each extent is carried in its own place, which a transposed shape would
  // not survive.
  require(runtime_cuda_launch_kernel_cluster(
              (CUfunction)"chop_cluster_probe",
              8, 8, 8, 32, 1, 1, 0,
              1, 4, 2, 0, (CUstream)1, parameters) == CUDA_SUCCESS &&
              record_count == 3,
          "an asymmetric cluster did not reach the driver");
  require(records[2].cluster[0] == 1 && records[2].cluster[1] == 4 &&
              records[2].cluster[2] == 2,
          "an asymmetric cluster was transposed on the way to the driver");

  // A zero extent is not a cluster, and the shim refuses it before the driver.
  size_t before = record_count;
  for (int extent = 0; extent < 3; ++extent) {
    unsigned int dimensions[3] = {2, 2, 2};
    dimensions[extent] = 0;
    require(runtime_cuda_launch_kernel_cluster(
                (CUfunction)"chop_cluster_probe",
                8, 8, 8, 32, 1, 1, 0,
                dimensions[0], dimensions[1], dimensions[2], 0,
                (CUstream)1, parameters) == CUDA_ERROR_INVALID_VALUE,
            "a cluster with a zero extent reached the driver");
  }
  require(record_count == before,
          "a refused cluster was recorded as a launch");

  // The attribute list this driver accepts is closed, which is what makes the
  // assertions above mean something: a duplicate, an unknown id, an empty list,
  // and a count with no array are each refused rather than partly read.
  CUlaunchAttribute duplicated[2];
  memset(duplicated, 0, sizeof(duplicated));
  duplicated[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
  duplicated[0].value.clusterDim.x = 2;
  duplicated[0].value.clusterDim.y = 1;
  duplicated[0].value.clusterDim.z = 1;
  duplicated[1] = duplicated[0];
  CUlaunchConfig config;
  memset(&config, 0, sizeof(config));
  config.gridDimX = 8;
  config.gridDimY = 1;
  config.gridDimZ = 1;
  config.blockDimX = 32;
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.hStream = (CUstream)1;
  config.attrs = duplicated;
  config.numAttrs = 2;
  require(cuLaunchKernelEx(&config, (CUfunction)"chop_cluster_probe",
                           parameters, NULL) == CUDA_ERROR_INVALID_VALUE,
          "a duplicated cluster attribute was accepted");

  duplicated[1].id = (CUlaunchAttributeID)0x7fff;
  require(cuLaunchKernelEx(&config, (CUfunction)"chop_cluster_probe",
                           parameters, NULL) == CUDA_ERROR_INVALID_VALUE,
          "an unknown launch attribute was accepted");

  config.numAttrs = 0;
  require(cuLaunchKernelEx(&config, (CUfunction)"chop_cluster_probe",
                           parameters, NULL) == CUDA_ERROR_INVALID_VALUE,
          "a launch with no attribute at all was accepted by the Ex path");

  config.numAttrs = 1;
  config.attrs = NULL;
  require(cuLaunchKernelEx(&config, (CUfunction)"chop_cluster_probe",
                           parameters, NULL) == CUDA_ERROR_INVALID_VALUE,
          "a nonzero attribute count with no array was accepted");
  require(record_count == before,
          "a malformed attribute list was recorded as a launch");

  // The programmatic-dependent launcher still asks for exactly its own
  // attribute and no cluster, which is what says nothing above changed it.
  require(runtime_cuda_launch_kernel_pdl(
              (CUfunction)"chop_cluster_probe",
              8, 1, 1, 32, 1, 1, 0, (CUstream)1, parameters) == CUDA_SUCCESS &&
              record_count == before + 1,
          "the programmatic launcher stopped reaching the driver");
  require(records[before].programmatic == 1 &&
              records[before].clustered == 0,
          "the programmatic launcher started asking for a cluster");
}

static runtime_cuda_tensor_map_descriptor tiled_descriptor(uint32_t rank) {
  runtime_cuda_tensor_map_descriptor descriptor = {0};
  descriptor.rank = rank;
  descriptor.data_type = 1;
  descriptor.swizzle = 1;
  descriptor.l2_promotion = 3;
  for (uint32_t index = 0; index < rank && index < 5; ++index) {
    descriptor.global_dims[index] = index == 0 ? 128 : 2;
    descriptor.box_dims[index] = index == 0 ? 128 : 1;
    descriptor.element_strides[index] = 1;
    if (index > 0) descriptor.global_strides_bytes[index - 1] = 128ULL << (index - 1);
  }
  return descriptor;
}

static void refuse_tensor_map(runtime_cuda_tensor_map_descriptor descriptor,
                              uint64_t address, uint64_t length) {
  CUtensorMap map;
  size_t before = tensor_map_encode_count;
  require(runtime_cuda_encode_tensor_map(&map, &descriptor, address, length) ==
              CUDA_ERROR_INVALID_VALUE,
          "a malformed tensor map was accepted");
  require(tensor_map_encode_count == before,
          "a malformed tensor map reached the driver");
}

static void test_ranked_tensor_map_contract(void) {
  CUtensorMap map;
  for (uint32_t rank = 1; rank <= 5; ++rank) {
    runtime_cuda_tensor_map_descriptor descriptor = tiled_descriptor(rank);
    require(runtime_cuda_encode_tensor_map(&map, &descriptor, 0x9000, 65536) ==
                CUDA_SUCCESS,
            "a supported tensor-map rank failed to encode");
    require(encoded_rank == rank, "the tensor-map rank changed at the driver");
    for (uint32_t index = 0; index < rank; ++index) {
      require(encoded_dimensions[index] == descriptor.global_dims[index] &&
                  encoded_boxes[index] == descriptor.box_dims[index] &&
                  encoded_element_strides[index] == descriptor.element_strides[index],
              "a tensor-map dimension changed at the driver");
      if (index > 0) {
        require(encoded_strides[index - 1] == descriptor.global_strides_bytes[index - 1],
                "a tensor-map stride changed at the driver");
      }
    }
  }
  static const struct { uint32_t code; CUtensorMapDataType driver; uint32_t box; } types[] = {
    {1, CU_TENSOR_MAP_DATA_TYPE_UINT8, 128},
    {2, CU_TENSOR_MAP_DATA_TYPE_INT32, 32},
    {3, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 64},
    {4, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 64},
    {5, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 32},
    {6, CU_TENSOR_MAP_DATA_TYPE_UINT8, 128},
    {7, CU_TENSOR_MAP_DATA_TYPE_UINT8, 128},
    {8, CU_TENSOR_MAP_DATA_TYPE_UINT16, 64},
    {9, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 128},
    {10, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, 128},
    {11, CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B, 128},
  };
  for (size_t index = 0; index < sizeof(types) / sizeof(types[0]); ++index) {
    runtime_cuda_tensor_map_descriptor descriptor = tiled_descriptor(2);
    descriptor.data_type = types[index].code;
    descriptor.box_dims[0] = types[index].box;
    descriptor.global_strides_bytes[0] = 1024;
    require(runtime_cuda_encode_tensor_map(&map, &descriptor, 0x9000, 65536) ==
                CUDA_SUCCESS,
            "a declared tensor-map element failed to encode");
    typed_tensor_map_descriptor observed;
    memcpy(&observed, &map, sizeof(observed));
    require(observed.data_type == (uint32_t)types[index].driver,
            "a tensor-map element changed driver type");
  }
  for (uint32_t swizzle = 0; swizzle <= 3; ++swizzle) {
    runtime_cuda_tensor_map_descriptor descriptor = tiled_descriptor(2);
    descriptor.swizzle = swizzle;
    descriptor.box_dims[0] = 16;
    require(runtime_cuda_encode_tensor_map(&map, &descriptor, 0x9000, 65536) ==
                CUDA_SUCCESS,
            "a declared swizzle failed to encode");
    typed_tensor_map_descriptor observed;
    memcpy(&observed, &map, sizeof(observed));
    const uint32_t expected[] = {CU_TENSOR_MAP_SWIZZLE_NONE,
                                 CU_TENSOR_MAP_SWIZZLE_128B,
                                 CU_TENSOR_MAP_SWIZZLE_32B,
                                 CU_TENSOR_MAP_SWIZZLE_64B};
    require(observed.swizzle == expected[swizzle], "a swizzle changed at the driver");
  }
  for (uint32_t interleave = 0; interleave <= 2; ++interleave) {
    runtime_cuda_tensor_map_descriptor descriptor = tiled_descriptor(3);
    descriptor.interleave = interleave;
    descriptor.swizzle = interleave == 2 ? 2 : 1;
    descriptor.box_dims[0] = 16;
    require(runtime_cuda_encode_tensor_map(&map, &descriptor, 0x9000, 65536) ==
                CUDA_SUCCESS,
            "a declared interleave failed to encode");
    typed_tensor_map_descriptor observed;
    memcpy(&observed, &map, sizeof(observed));
    require(observed.interleave == interleave, "interleave changed at the driver");
  }
  for (uint32_t promotion = 0; promotion <= 3; ++promotion) {
    runtime_cuda_tensor_map_descriptor descriptor = tiled_descriptor(1);
    descriptor.l2_promotion = promotion;
    require(runtime_cuda_encode_tensor_map(&map, &descriptor, 0x9000, 65536) ==
                CUDA_SUCCESS,
            "a declared L2 promotion failed to encode");
    typed_tensor_map_descriptor observed;
    memcpy(&observed, &map, sizeof(observed));
    require(observed.l2_promotion == promotion, "L2 promotion changed at the driver");
  }
  for (uint32_t fill = 0; fill <= 1; ++fill) {
    runtime_cuda_tensor_map_descriptor descriptor = tiled_descriptor(2);
    descriptor.data_type = 4;
    descriptor.box_dims[0] = 64;
    descriptor.global_strides_bytes[0] = 256;
    descriptor.oob_fill = fill;
    require(runtime_cuda_encode_tensor_map(&map, &descriptor, 0x9000, 65536) ==
                CUDA_SUCCESS,
            "a declared floating OOB fill failed to encode");
    typed_tensor_map_descriptor observed;
    memcpy(&observed, &map, sizeof(observed));
    require(observed.oob_fill == fill, "OOB fill changed at the driver");
  }
  runtime_cuda_tensor_map_descriptor bad = tiled_descriptor(2);
  bad.rank = 0; refuse_tensor_map(bad, 0x9000, 65536);
  bad = tiled_descriptor(2); bad.rank = 6; refuse_tensor_map(bad, 0x9000, 65536);
  bad = tiled_descriptor(2); bad.box_dims[0] = 0; refuse_tensor_map(bad, 0x9000, 65536);
  bad = tiled_descriptor(2); bad.box_dims[0] = 257; refuse_tensor_map(bad, 0x9000, 65536);
  bad = tiled_descriptor(2); bad.element_strides[0] = 0; refuse_tensor_map(bad, 0x9000, 65536);
  bad = tiled_descriptor(2); bad.element_strides[0] = 9; refuse_tensor_map(bad, 0x9000, 65536);
  bad = tiled_descriptor(2); refuse_tensor_map(bad, 0x9008, 65536);
  bad = tiled_descriptor(2); bad.global_strides_bytes[0] = 129; refuse_tensor_map(bad, 0x9000, 65536);
  bad = tiled_descriptor(2); bad.global_strides_bytes[0] = UINT64_MAX; refuse_tensor_map(bad, 0x9000, 65536);
  bad = tiled_descriptor(2); refuse_tensor_map(bad, 0x9000, 64);
  bad = tiled_descriptor(2); bad.swizzle = 4; refuse_tensor_map(bad, 0x9000, 65536);
  bad = tiled_descriptor(2); bad.interleave = 3; refuse_tensor_map(bad, 0x9000, 65536);
  bad = tiled_descriptor(2); bad.l2_promotion = 4; refuse_tensor_map(bad, 0x9000, 65536);
  bad = tiled_descriptor(2); bad.oob_fill = 2; refuse_tensor_map(bad, 0x9000, 65536);
  bad = tiled_descriptor(2); bad.box_dims[0] = 256; refuse_tensor_map(bad, 0x9000, 65536);
  bad = tiled_descriptor(2); bad.global_dims[2] = 1; refuse_tensor_map(bad, 0x9000, 65536);
  bad = tiled_descriptor(2); bad.data_type = 10; bad.global_dims[0] = 64; refuse_tensor_map(bad, 0x9000, 65536);
  bad = tiled_descriptor(2); bad.data_type = 11; bad.interleave = 1; refuse_tensor_map(bad, 0x9000, 65536);
  bad = tiled_descriptor(3); bad.interleave = 2; refuse_tensor_map(bad, 0x9000, 65536);
  bad = tiled_descriptor(2); bad.oob_fill = 1; refuse_tensor_map(bad, 0x9000, 65536);
}

/* A fake of the virtual memory management calls the lending helpers make,
   recording each call in order. `vmm_fail` names the call to refuse. */
static char vmm_calls[64][32];
static size_t vmm_call_count;
static const char* vmm_fail;
static CUmemAllocationProp vmm_created_prop;
static size_t vmm_created_bytes;
static size_t vmm_reserved_bytes;
static CUmemGenericAllocationHandle vmm_mapped_handle;
static size_t vmm_mapped_offset;
static CUmemAccessDesc vmm_access[16];
static size_t vmm_access_count;
static CUmemAllocationHandleType vmm_handle_type;
static void* vmm_imported_from;
static CUmemGenericAllocationHandle vmm_released[8];
static size_t vmm_released_count;

static CUresult vmm_call(const char* name) {
  if (vmm_call_count < 64) {
    strncpy(vmm_calls[vmm_call_count], name, sizeof vmm_calls[0] - 1);
    ++vmm_call_count;
  }
  return vmm_fail != NULL && strcmp(vmm_fail, name) == 0 ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_SUCCESS;
}

static void vmm_reset(const char* fail) {
  memset(vmm_calls, 0, sizeof vmm_calls);
  vmm_call_count = 0;
  vmm_fail = fail;
  vmm_access_count = 0;
  vmm_released_count = 0;
}

static void vmm_expect(const char* const* names, size_t count) {
  if (vmm_call_count != count) abort();
  for (size_t i = 0; i < count; ++i) {
    if (strcmp(vmm_calls[i], names[i]) != 0) abort();
  }
}

CUresult CUDAAPI cuMemGetAllocationGranularity(size_t* granularity, const CUmemAllocationProp* prop,
                                               CUmemAllocationGranularity_flags option) {
  CUresult result = vmm_call("granularity");
  vmm_created_prop = *prop;
  if (option != CU_MEM_ALLOC_GRANULARITY_MINIMUM) abort();
  if (result == CUDA_SUCCESS) *granularity = 2u << 20;
  return result;
}

CUresult CUDAAPI cuMemCreate(CUmemGenericAllocationHandle* handle, size_t size,
                             const CUmemAllocationProp* prop, unsigned long long flags) {
  CUresult result = vmm_call("create");
  if (flags != 0) abort();
  vmm_created_prop = *prop;
  vmm_created_bytes = size;
  if (result == CUDA_SUCCESS) *handle = 0x51;
  return result;
}

CUresult CUDAAPI cuMemAddressReserve(CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr,
                                     unsigned long long flags) {
  CUresult result = vmm_call("reserve");
  if (alignment != 0 || addr != 0 || flags != 0) abort();
  vmm_reserved_bytes = size;
  if (result == CUDA_SUCCESS) *ptr = 0x7f0000000000ull;
  return result;
}

CUresult CUDAAPI cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle,
                          unsigned long long flags) {
  CUresult result = vmm_call("map");
  if (ptr != 0x7f0000000000ull || size != vmm_reserved_bytes || flags != 0) abort();
  vmm_mapped_handle = handle;
  vmm_mapped_offset = offset;
  return result;
}

CUresult CUDAAPI cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count) {
  CUresult result = vmm_call("access");
  if (ptr != 0x7f0000000000ull || size != vmm_reserved_bytes || count > 16) abort();
  memcpy(vmm_access, desc, count * sizeof *desc);
  vmm_access_count = count;
  return result;
}

CUresult CUDAAPI cuMemExportToShareableHandle(void* shareable, CUmemGenericAllocationHandle handle,
                                              CUmemAllocationHandleType type, unsigned long long flags) {
  CUresult result = vmm_call("export");
  if (handle != 0x51 || flags != 0) abort();
  vmm_handle_type = type;
  if (result == CUDA_SUCCESS) *(int*)shareable = 41;
  return result;
}

CUresult CUDAAPI cuMemImportFromShareableHandle(CUmemGenericAllocationHandle* handle, void* os_handle,
                                                CUmemAllocationHandleType type) {
  CUresult result = vmm_call("import");
  vmm_handle_type = type;
  vmm_imported_from = os_handle;
  if (result == CUDA_SUCCESS) *handle = 0x52;
  return result;
}

CUresult CUDAAPI cuMemUnmap(CUdeviceptr ptr, size_t size) {
  CUresult result = vmm_call("unmap");
  if (ptr != 0x7f0000000000ull || size != vmm_reserved_bytes) abort();
  return result;
}

CUresult CUDAAPI cuMemAddressFree(CUdeviceptr ptr, size_t size) {
  CUresult result = vmm_call("free");
  if (ptr != 0x7f0000000000ull || size != vmm_reserved_bytes) abort();
  return result;
}

CUresult CUDAAPI cuMemRelease(CUmemGenericAllocationHandle handle) {
  CUresult result = vmm_call("release");
  if (vmm_released_count < 8) vmm_released[vmm_released_count++] = handle;
  return result;
}

static void test_vmm_lending_helpers(void) {
  const size_t bytes = 6u << 20;
  size_t granularity = 0;
  vmm_reset(NULL);
  if (runtime_cuda_vmm_granularity(3, &granularity) != CUDA_SUCCESS || granularity != (2u << 20)) abort();
  if (vmm_created_prop.type != CU_MEM_ALLOCATION_TYPE_PINNED ||
      vmm_created_prop.requestedHandleTypes != CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR ||
      vmm_created_prop.location.type != CU_MEM_LOCATION_TYPE_DEVICE || vmm_created_prop.location.id != 3) {
    abort();
  }
  if (runtime_cuda_vmm_granularity(3, NULL) != CUDA_ERROR_INVALID_VALUE) abort();

  /* Create: an exportable allocation on the ordinal, mapped and writable by it. */
  uint64_t handle = 0;
  CUdeviceptr address = 0;
  vmm_reset(NULL);
  if (runtime_cuda_vmm_create(2, bytes, &handle, &address) != CUDA_SUCCESS) abort();
  {
    const char* expected[] = {"create", "reserve", "map", "access"};
    vmm_expect(expected, 4);
  }
  if (handle != 0x51 || address != 0x7f0000000000ull || vmm_created_bytes != bytes ||
      vmm_reserved_bytes != bytes || vmm_mapped_handle != 0x51 || vmm_mapped_offset != 0) {
    abort();
  }
  if (vmm_created_prop.location.id != 2 ||
      vmm_created_prop.requestedHandleTypes != CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    abort();
  }
  if (vmm_access_count != 1 || vmm_access[0].location.type != CU_MEM_LOCATION_TYPE_DEVICE ||
      vmm_access[0].location.id != 2 || vmm_access[0].flags != CU_MEM_ACCESS_FLAGS_PROT_READWRITE) {
    abort();
  }
  /* Every failure leaves nothing behind and the outputs untouched. */
  {
    const char* fail[] = {"create", "reserve", "map", "access"};
    const char* after_create[] = {"create"};
    const char* after_reserve[] = {"create", "reserve", "release"};
    const char* after_map[] = {"create", "reserve", "map", "free", "release"};
    const char* after_access[] = {"create", "reserve", "map", "access", "unmap", "free", "release"};
    const char* const* sequences[] = {after_create, after_reserve, after_map, after_access};
    const size_t lengths[] = {1, 3, 5, 7};
    for (size_t i = 0; i < 4; ++i) {
      handle = 7;
      address = 9;
      vmm_reset(fail[i]);
      if (runtime_cuda_vmm_create(2, bytes, &handle, &address) != CUDA_ERROR_OUT_OF_MEMORY) abort();
      vmm_expect(sequences[i], lengths[i]);
      if (handle != 7 || address != 9) abort();
      if (i > 0 && (vmm_released_count != 1 || vmm_released[0] != 0x51)) abort();
    }
  }
  if (runtime_cuda_vmm_create(2, 0, &handle, &address) != CUDA_ERROR_INVALID_VALUE) abort();

  /* Export hands out a POSIX file descriptor. */
  int fd = -1;
  vmm_reset(NULL);
  if (runtime_cuda_vmm_export(0x51, &fd) != CUDA_SUCCESS || fd != 41 ||
      vmm_handle_type != CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    abort();
  }
  if (runtime_cuda_vmm_export(0x51, NULL) != CUDA_ERROR_INVALID_VALUE) abort();

  /* Import maps the descriptor's allocation and grants no access yet. */
  vmm_reset(NULL);
  if (runtime_cuda_vmm_import(41, bytes, &handle, &address) != CUDA_SUCCESS) abort();
  {
    const char* expected[] = {"import", "reserve", "map"};
    vmm_expect(expected, 3);
  }
  if (handle != 0x52 || address != 0x7f0000000000ull || vmm_imported_from != (void*)(uintptr_t)41 ||
      vmm_handle_type != CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR || vmm_mapped_handle != 0x52) {
    abort();
  }
  {
    const char* fail[] = {"import", "reserve", "map"};
    const char* after_import[] = {"import"};
    const char* after_reserve[] = {"import", "reserve", "release"};
    const char* after_map[] = {"import", "reserve", "map", "free", "release"};
    const char* const* sequences[] = {after_import, after_reserve, after_map};
    const size_t lengths[] = {1, 3, 5};
    for (size_t i = 0; i < 3; ++i) {
      handle = 7;
      address = 9;
      vmm_reset(fail[i]);
      if (runtime_cuda_vmm_import(41, bytes, &handle, &address) != CUDA_ERROR_OUT_OF_MEMORY) abort();
      vmm_expect(sequences[i], lengths[i]);
      if (handle != 7 || address != 9) abort();
      if (i > 0 && (vmm_released_count != 1 || vmm_released[0] != 0x52)) abort();
    }
  }
  if (runtime_cuda_vmm_import(-1, bytes, &handle, &address) != CUDA_ERROR_INVALID_VALUE) abort();

  /* Access: read-only or read-write for every listed device. */
  const int ordinals[] = {1, 0, 3};
  vmm_reset(NULL);
  vmm_reserved_bytes = bytes;
  if (runtime_cuda_vmm_set_access(0x7f0000000000ull, bytes, ordinals, 3, 1) != CUDA_SUCCESS) abort();
  if (vmm_access_count != 3) abort();
  for (size_t i = 0; i < 3; ++i) {
    if (vmm_access[i].location.type != CU_MEM_LOCATION_TYPE_DEVICE || vmm_access[i].location.id != ordinals[i] ||
        vmm_access[i].flags != CU_MEM_ACCESS_FLAGS_PROT_READ) {
      abort();
    }
  }
  if (runtime_cuda_vmm_set_access(0x7f0000000000ull, bytes, ordinals, 0, 1) != CUDA_ERROR_INVALID_VALUE) abort();
  if (runtime_cuda_vmm_set_access(0x7f0000000000ull, bytes, ordinals, 17, 1) != CUDA_ERROR_INVALID_VALUE) abort();
  if (runtime_cuda_vmm_set_access(0x7f0000000000ull, bytes, NULL, 1, 1) != CUDA_ERROR_INVALID_VALUE) abort();

  /* Release unmaps, frees the range and releases the handle, reporting the
     first failure but still attempting the rest. */
  vmm_reset(NULL);
  if (runtime_cuda_vmm_release(0x52, 0x7f0000000000ull, bytes) != CUDA_SUCCESS) abort();
  {
    const char* expected[] = {"unmap", "free", "release"};
    vmm_expect(expected, 3);
  }
  vmm_reset("unmap");
  if (runtime_cuda_vmm_release(0x52, 0x7f0000000000ull, bytes) != CUDA_ERROR_OUT_OF_MEMORY) abort();
  {
    const char* expected[] = {"unmap", "free", "release"};
    vmm_expect(expected, 3);
  }
  vmm_reset(NULL);
  if (runtime_cuda_vmm_release(0x52, 0, bytes) != CUDA_SUCCESS) abort();
  {
    const char* expected[] = {"release"};
    vmm_expect(expected, 1);
  }
}

int main(void) {
  void* pinned = NULL;
  if (runtime_cuda_host_alloc(&pinned, 128) != CUDA_SUCCESS || pinned == NULL) abort();
  memset(pinned, 0x5a, 128);
  if (runtime_cuda_host_free(pinned) != CUDA_SUCCESS) abort();
  if (pinned_allocation_count != 1 || pinned_free_count != 1) abort();
  CUdeviceptr mapped_device = 0;
  if (runtime_cuda_host_alloc_mapped(&pinned, &mapped_device, 256) != CUDA_SUCCESS) abort();
  if (last_host_alloc_flags != (CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP)) abort();
  if (mapped_device != (CUdeviceptr)(uintptr_t)pinned + 0x1000000000ull) abort();
  if (runtime_cuda_host_free(pinned) != CUDA_SUCCESS) abort();
  fail_device_pointer = 1;
  if (runtime_cuda_host_alloc_mapped(&pinned, &mapped_device, 256) != CUDA_ERROR_NOT_SUPPORTED) abort();
  if (pinned != NULL || mapped_device != 0) abort();
  fail_device_pointer = 0;
  if (runtime_cuda_host_alloc_mapped(&pinned, &mapped_device, 0) != CUDA_ERROR_INVALID_VALUE) abort();
  if (pinned_allocation_count != 3 || pinned_free_count != 3) abort();
  if (runtime_cuda_context_synchronize() != CUDA_SUCCESS || context_sync_count != 1) abort();
  unsigned char uuid[16];
  if (runtime_cuda_device_uuid(uuid, 0) != CUDA_SUCCESS) abort();
  for (int i = 0; i < 16; ++i) if (uuid[i] != 0x11) abort();

  float elapsed = 0;
  if (runtime_cuda_event_elapsed(&elapsed, (CUevent)2, (CUevent)1) != CUDA_ERROR_INVALID_VALUE) abort();
  if (runtime_cuda_event_elapsed(&elapsed, (CUevent)1, (CUevent)2) != CUDA_SUCCESS || elapsed != 3.25f) abort();
  test_peer_copy_and_wait_driver_abi();
  test_checked_launchers_reject_invalid_descriptors();
  test_checked_launchers_pass_the_descriptor_through();
  test_tensor_map_encoding_and_shared_memory_configuration();
  test_ranked_tensor_map_contract();
  test_mixed_width_parameters_reach_the_driver_intact();
  test_cluster_launch_attributes_reach_the_driver();
  test_vmm_lending_helpers();
  return 0;
}

CUresult CUDAAPI cuEventElapsedTime(float* milliseconds, CUevent start, CUevent end) {
  if (start != (CUevent)1 || end != (CUevent)2) return CUDA_ERROR_INVALID_VALUE;
  *milliseconds = 3.25f;
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemGetInfo_v2(size_t* free_bytes, size_t* total_bytes) {
  *free_bytes = 3u << 30;
  *total_bytes = 8u << 30;
  return CUDA_SUCCESS;
}

CUresult cuDeviceGet(CUdevice* device, int ordinal) { *device = ordinal; return CUDA_SUCCESS; }
CUresult cuDeviceGetUuid_v2(CUuuid* uuid, CUdevice device) {
  (void)device;
  memset(uuid->bytes, 0x11, 16);
  return CUDA_SUCCESS;
}
