#include "tensor_map.h"
#include <string.h>

static int valid_parameter_contract(
    void** parameters,
    const uint8_t* parameter_widths,
    size_t parameter_count) {
  if (parameters == NULL || parameter_widths == NULL || parameter_count == 0) {
    return 0;
  }
  for (size_t index = 0; index < parameter_count; ++index) {
    if (parameters[index] == NULL ||
        (parameter_widths[index] != RUNTIME_CUDA_PARAMETER_U32 &&
         parameter_widths[index] != RUNTIME_CUDA_PARAMETER_U64 &&
         parameter_widths[index] != RUNTIME_CUDA_PARAMETER_BYTES64 &&
         parameter_widths[index] != RUNTIME_CUDA_PARAMETER_TENSOR_MAP)) {
      return 0;
    }
  }
  return 1;
}

CUresult runtime_cuda_launch_kernel_checked(
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
    const uint8_t* parameter_widths,
    size_t parameter_count) {
  if (!valid_parameter_contract(
          parameters, parameter_widths, parameter_count)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  return cuLaunchKernel(function, grid_x, grid_y, grid_z,
                        block_x, block_y, block_z, shared_memory_bytes,
                        stream, parameters, NULL);
}

static int tensor_map_type(uint32_t code, CUtensorMapDataType* type,
                           uint64_t* byte_numerator, uint64_t* byte_denominator) {
  *byte_denominator = 1;
  switch (code) {
    case 1: case 6: case 7:
      *type = CU_TENSOR_MAP_DATA_TYPE_UINT8; *byte_numerator = 1; return 1;
    case 2: *type = CU_TENSOR_MAP_DATA_TYPE_INT32; *byte_numerator = 4; return 1;
    case 3: *type = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16; *byte_numerator = 2; return 1;
    case 4: *type = CU_TENSOR_MAP_DATA_TYPE_FLOAT16; *byte_numerator = 2; return 1;
    case 5: *type = CU_TENSOR_MAP_DATA_TYPE_FLOAT32; *byte_numerator = 4; return 1;
    case 8: *type = CU_TENSOR_MAP_DATA_TYPE_UINT16; *byte_numerator = 2; return 1;
    case 9:
      *type = CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;
      *byte_numerator = 1; *byte_denominator = 2; return 1;
    case 10:
      *type = CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B;
      *byte_numerator = 1; return 1;
    case 11:
      *type = CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B;
      *byte_numerator = 1; return 1;
    default: return 0;
  }
}

static int checked_add(uint64_t left, uint64_t right, uint64_t* output) {
  if (left > UINT64_MAX - right) return 0;
  *output = left + right;
  return 1;
}

static int checked_mul(uint64_t left, uint64_t right, uint64_t* output) {
  if (right != 0 && left > UINT64_MAX / right) return 0;
  *output = left * right;
  return 1;
}

CUresult runtime_cuda_validate_tensor_map(
    const runtime_cuda_tensor_map_descriptor* descriptor,
    CUdeviceptr address,
    uint64_t buffer_bytes) {
  if (descriptor == NULL || descriptor->rank < 1 || descriptor->rank > 5 ||
      descriptor->swizzle > 3 || descriptor->interleave > 2 ||
      descriptor->l2_promotion > 3 || descriptor->oob_fill > 1) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUtensorMapDataType type;
  uint64_t numerator, denominator;
  if (!tensor_map_type(descriptor->data_type, &type, &numerator, &denominator)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  (void)type;
  const int packed_16 = descriptor->data_type == 10 || descriptor->data_type == 11;
  const uint64_t alignment = packed_16 || descriptor->interleave == 2 ? 32 : 16;
  if (address == 0 || address % alignment != 0 ||
      (descriptor->interleave != 0 && descriptor->rank < 3) ||
      (descriptor->interleave == 2 && descriptor->swizzle != 2) ||
      (descriptor->data_type == 11 && descriptor->interleave != 0) ||
      (packed_16 && descriptor->swizzle != 0 && descriptor->swizzle != 1) ||
      (descriptor->oob_fill == 1 && descriptor->data_type != 3 &&
       descriptor->data_type != 4 && descriptor->data_type != 5)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  for (uint32_t index = 0; index < 5; ++index) {
    if (index < descriptor->rank) {
      if (descriptor->global_dims[index] == 0 ||
          descriptor->global_dims[index] > (UINT64_C(1) << 32) ||
          descriptor->box_dims[index] == 0 || descriptor->box_dims[index] > 256 ||
          descriptor->element_strides[index] == 0 ||
          descriptor->element_strides[index] > 8) return CUDA_ERROR_INVALID_VALUE;
    } else if (descriptor->global_dims[index] != 0 ||
               descriptor->box_dims[index] != 0 ||
               descriptor->element_strides[index] != 0) {
      return CUDA_ERROR_INVALID_VALUE;
    }
  }
  if ((packed_16 && (descriptor->global_dims[0] % 128 != 0 ||
                     descriptor->box_dims[0] != 128)) ||
      (descriptor->data_type == 9 && descriptor->global_dims[0] % 2 != 0)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  uint64_t row_bytes, box_bytes;
  if (!checked_mul(descriptor->global_dims[0], numerator, &row_bytes) ||
      !checked_mul(descriptor->box_dims[0], numerator, &box_bytes)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  row_bytes = (row_bytes + denominator - 1) / denominator;
  box_bytes = (box_bytes + denominator - 1) / denominator;
  const uint64_t swizzle_bytes[4] = {0, 128, 32, 64};
  if (descriptor->interleave == 0 &&
      (box_bytes % 16 != 0 ||
       (swizzle_bytes[descriptor->swizzle] != 0 &&
        box_bytes > swizzle_bytes[descriptor->swizzle]))) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  uint64_t footprint = row_bytes;
  for (uint32_t index = 0; index < 4; ++index) {
    const uint64_t stride = descriptor->global_strides_bytes[index];
    if (index >= descriptor->rank - 1) {
      if (stride != 0) return CUDA_ERROR_INVALID_VALUE;
      continue;
    }
    uint64_t previous_extent = row_bytes;
    if (index > 0 &&
        !checked_mul(descriptor->global_strides_bytes[index - 1],
                     descriptor->global_dims[index], &previous_extent)) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    if (stride == 0 || stride >= (UINT64_C(1) << 40) ||
        stride % alignment != 0 || stride < previous_extent) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    uint64_t part;
    if (!checked_mul(stride, descriptor->global_dims[index + 1] - 1, &part) ||
        !checked_add(footprint, part, &footprint)) {
      return CUDA_ERROR_INVALID_VALUE;
    }
  }
  if (footprint > buffer_bytes || footprint > UINT64_MAX - address) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  return CUDA_SUCCESS;
}

CUresult runtime_cuda_encode_tensor_map(
    void* output,
    const runtime_cuda_tensor_map_descriptor* descriptor,
    CUdeviceptr address,
    uint64_t buffer_bytes) {
  if (output == NULL) return CUDA_ERROR_INVALID_VALUE;
  CUresult result = runtime_cuda_validate_tensor_map(descriptor, address, buffer_bytes);
  if (result != CUDA_SUCCESS) return result;
  CUtensorMapDataType data_type;
  uint64_t numerator, denominator;
  tensor_map_type(descriptor->data_type, &data_type, &numerator, &denominator);
  const CUtensorMapSwizzle swizzles[4] = {
      CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_SWIZZLE_32B, CU_TENSOR_MAP_SWIZZLE_64B};
  const CUtensorMapInterleave interleaves[3] = {
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_INTERLEAVE_16B,
      CU_TENSOR_MAP_INTERLEAVE_32B};
  const CUtensorMapL2promotion promotions[4] = {
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_64B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B};
  const CUtensorMapFloatOOBfill fills[2] = {
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA};
  cuuint64_t dimensions[5] = {0};
  cuuint64_t strides[4] = {0};
  for (uint32_t index = 0; index < descriptor->rank; ++index) {
    dimensions[index] = descriptor->global_dims[index];
    if (index + 1 < descriptor->rank) {
      strides[index] = descriptor->global_strides_bytes[index];
    }
  }
  return cuTensorMapEncodeTiled(
      (CUtensorMap*)output, data_type, descriptor->rank, (void*)(uintptr_t)address,
      dimensions, strides,
      descriptor->box_dims, descriptor->element_strides,
      interleaves[descriptor->interleave], swizzles[descriptor->swizzle],
      promotions[descriptor->l2_promotion], fills[descriptor->oob_fill]);
}

CUresult runtime_cuda_set_max_dynamic_shared(CUfunction function, int bytes) {
  return cuFuncSetAttribute(
      function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, bytes);
}

CUresult runtime_cuda_max_active_clusters(
    CUfunction function,
    unsigned int gx, unsigned int gy, unsigned int gz,
    unsigned int bx, unsigned int by, unsigned int bz,
    unsigned int shared_memory_bytes,
    unsigned int cx, unsigned int cy, unsigned int cz,
    int cooperative,
    int* clusters) {
  if (function == NULL || clusters == NULL || gx == 0 || gy == 0 || gz == 0 ||
      bx == 0 || by == 0 || bz == 0 || cx == 0 || cy == 0 || cz == 0 ||
      gx % cx != 0 || gy % cy != 0 || gz % cz != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUlaunchAttribute attributes[2];
  memset(attributes, 0, sizeof(attributes));
  attributes[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
  attributes[0].value.clusterDim.x = cx;
  attributes[0].value.clusterDim.y = cy;
  attributes[0].value.clusterDim.z = cz;
  attributes[1].id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
  attributes[1].value.cooperative = 1;
  CUlaunchConfig config = {0};
  config.gridDimX = gx;
  config.gridDimY = gy;
  config.gridDimZ = gz;
  config.blockDimX = bx;
  config.blockDimY = by;
  config.blockDimZ = bz;
  config.sharedMemBytes = shared_memory_bytes;
  config.attrs = attributes;
  config.numAttrs = cooperative ? 2 : 1;
  *clusters = 0;
  return cuOccupancyMaxActiveClusters(clusters, function, &config);
}

CUresult runtime_cuda_context_sm_count(int* sms) {
  if (sms == NULL) return CUDA_ERROR_INVALID_VALUE;
  CUexecAffinityParam affinity;
  memset(&affinity, 0, sizeof(affinity));
  CUresult result = cuCtxGetExecAffinity(&affinity, CU_EXEC_AFFINITY_TYPE_SM_COUNT);
  *sms = result == CUDA_SUCCESS ? (int)affinity.param.smCount.val : 0;
  return result;
}

CUresult runtime_cuda_memory_pool_set_device_access(
    CUmemoryPool pool,
    int device,
    int read_write) {
  CUmemAccessDesc descriptor = {0};
  descriptor.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  descriptor.location.id = device;
  descriptor.flags = read_write ? CU_MEM_ACCESS_FLAGS_PROT_READWRITE
                                : CU_MEM_ACCESS_FLAGS_PROT_NONE;
  return cuMemPoolSetAccess(pool, &descriptor, 1);
}

CUresult runtime_cuda_memory_pool_get_device_access(
    CUmemoryPool pool,
    int device,
    int* read_write) {
  if (pool == NULL || device < 0 || read_write == NULL) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUmemAccess_flags flags = CU_MEM_ACCESS_FLAGS_PROT_NONE;
  CUmemLocation location;
  location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  location.id = device;
  CUresult result = cuMemPoolGetAccess(&flags, pool, &location);
  if (result == CUDA_SUCCESS) {
    *read_write = flags == CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  }
  return result;
}

CUresult runtime_cuda_host_alloc(void** address, size_t bytes) {
  if (address == NULL || bytes == 0) return CUDA_ERROR_INVALID_VALUE;
  return cuMemHostAlloc(address, bytes, CU_MEMHOSTALLOC_PORTABLE);
}

CUresult runtime_cuda_host_alloc_mapped(void** address, CUdeviceptr* device, size_t bytes) {
  if (address == NULL || device == NULL || bytes == 0) return CUDA_ERROR_INVALID_VALUE;
  *address = NULL;
  *device = 0;
  CUresult result =
      cuMemHostAlloc(address, bytes, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP);
  if (result != CUDA_SUCCESS) return result;
  result = cuMemHostGetDevicePointer(device, *address, 0);
  if (result != CUDA_SUCCESS) {
    (void)cuMemFreeHost(*address);
    *address = NULL;
    *device = 0;
  }
  return result;
}

CUresult runtime_cuda_host_free(void* address) {
  if (address == NULL) return CUDA_ERROR_INVALID_VALUE;
  return cuMemFreeHost(address);
}

CUresult runtime_cuda_context_synchronize(void) {
  return cuCtxSynchronize();
}

CUresult runtime_cuda_launch_kernel_pdl(
    CUfunction function,
    unsigned int grid_x,
    unsigned int grid_y,
    unsigned int grid_z,
    unsigned int block_x,
    unsigned int block_y,
    unsigned int block_z,
    unsigned int shared_memory_bytes,
    CUstream stream,
    void** parameters) {
  CUlaunchAttribute attribute = {0};
  attribute.id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
  attribute.value.programmaticStreamSerializationAllowed = 1;
  CUlaunchConfig config = {0};
  config.gridDimX = grid_x;
  config.gridDimY = grid_y;
  config.gridDimZ = grid_z;
  config.blockDimX = block_x;
  config.blockDimY = block_y;
  config.blockDimZ = block_z;
  config.sharedMemBytes = shared_memory_bytes;
  config.hStream = stream;
  config.attrs = &attribute;
  config.numAttrs = 1;
  return cuLaunchKernelEx(&config, function, parameters, NULL);
}

/* One launch whose blocks are grouped into thread-block clusters.

   The cluster is a launch attribute, exactly as programmatic stream
   serialization is, so both are set on the same CUlaunchConfig and this is the
   one entry that carries either or both. A caller that wants no cluster does
   not come here: it calls the plain launch or the PDL one above, so nothing
   about an ordinary launch changes.

   The driver decides whether this device forms the cluster asked for, and a
   refusal is returned unchanged. Nothing here reduces a cluster it could not
   launch to one it could. */
CUresult runtime_cuda_launch_kernel_cluster(
    CUfunction function,
    unsigned int grid_x,
    unsigned int grid_y,
    unsigned int grid_z,
    unsigned int block_x,
    unsigned int block_y,
    unsigned int block_z,
    unsigned int shared_memory_bytes,
    unsigned int cluster_x,
    unsigned int cluster_y,
    unsigned int cluster_z,
    int programmatic_stream_serialization,
    CUstream stream,
    void** parameters) {
  if (cluster_x == 0 || cluster_y == 0 || cluster_z == 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUlaunchAttribute attributes[2] = {0};
  unsigned int count = 0;
  attributes[count].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
  attributes[count].value.clusterDim.x = cluster_x;
  attributes[count].value.clusterDim.y = cluster_y;
  attributes[count].value.clusterDim.z = cluster_z;
  ++count;
  if (programmatic_stream_serialization) {
    attributes[count].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
    attributes[count].value.programmaticStreamSerializationAllowed = 1;
    ++count;
  }
  CUlaunchConfig config = {0};
  config.gridDimX = grid_x;
  config.gridDimY = grid_y;
  config.gridDimZ = grid_z;
  config.blockDimX = block_x;
  config.blockDimY = block_y;
  config.blockDimZ = block_z;
  config.sharedMemBytes = shared_memory_bytes;
  config.hStream = stream;
  config.attrs = attributes;
  config.numAttrs = count;
  return cuLaunchKernelEx(&config, function, parameters, NULL);
}

/* A cooperative launch: every block of the grid resident at once, or the
   driver refuses the launch. With a cluster (cluster_x > 0) both attributes
   are attached; a zero cluster asks for none. The driver's refusal, for a
   grid too large to be co-resident or a combination this device does not
   support, is returned unchanged. */
CUresult runtime_cuda_launch_kernel_cooperative(
    CUfunction function,
    unsigned int grid_x,
    unsigned int grid_y,
    unsigned int grid_z,
    unsigned int block_x,
    unsigned int block_y,
    unsigned int block_z,
    unsigned int shared_memory_bytes,
    unsigned int cluster_x,
    unsigned int cluster_y,
    unsigned int cluster_z,
    CUstream stream,
    void** parameters) {
  CUlaunchAttribute attributes[2] = {0};
  unsigned int count = 0;
  attributes[count].id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
  attributes[count].value.cooperative = 1;
  ++count;
  if (cluster_x != 0 || cluster_y != 0 || cluster_z != 0) {
    if (cluster_x == 0 || cluster_y == 0 || cluster_z == 0) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    attributes[count].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    attributes[count].value.clusterDim.x = cluster_x;
    attributes[count].value.clusterDim.y = cluster_y;
    attributes[count].value.clusterDim.z = cluster_z;
    ++count;
  }
  CUlaunchConfig config = {0};
  config.gridDimX = grid_x;
  config.gridDimY = grid_y;
  config.gridDimZ = grid_z;
  config.blockDimX = block_x;
  config.blockDimY = block_y;
  config.blockDimZ = block_z;
  config.sharedMemBytes = shared_memory_bytes;
  config.hStream = stream;
  config.attrs = attributes;
  config.numAttrs = count;
  return cuLaunchKernelEx(&config, function, parameters, NULL);
}

CUresult runtime_cuda_launch_kernel_pdl_checked(
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
    const uint8_t* parameter_widths,
    size_t parameter_count) {
  if (!valid_parameter_contract(
          parameters, parameter_widths, parameter_count)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  return runtime_cuda_launch_kernel_pdl(
      function, grid_x, grid_y, grid_z, block_x, block_y, block_z,
      shared_memory_bytes, stream, parameters);
}

/* Compile the elapsed-time signature against the pinned CUDA header. */
CUresult runtime_cuda_event_elapsed(float* milliseconds, CUevent start, CUevent end) {
    return cuEventElapsedTime(milliseconds, start, end);
}

static CUmemAllocationProp runtime_cuda_vmm_prop(int ordinal) {
  CUmemAllocationProp prop;
  memset(&prop, 0, sizeof prop);
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = ordinal;
  return prop;
}

CUresult runtime_cuda_vmm_granularity(int ordinal, size_t* granularity) {
  if (granularity == NULL) return CUDA_ERROR_INVALID_VALUE;
  CUmemAllocationProp prop = runtime_cuda_vmm_prop(ordinal);
  return cuMemGetAllocationGranularity(granularity, &prop,
                                       CU_MEM_ALLOC_GRANULARITY_MINIMUM);
}

/* Reserves an address range for `bytes` and maps `handle` there; on failure
   frees the range and leaves the handle to the caller. */
static CUresult runtime_cuda_vmm_map(CUmemGenericAllocationHandle handle,
                                     size_t bytes, CUdeviceptr* address) {
  CUdeviceptr reserved = 0;
  CUresult result = cuMemAddressReserve(&reserved, bytes, 0, 0, 0);
  if (result != CUDA_SUCCESS) return result;
  result = cuMemMap(reserved, bytes, 0, handle, 0);
  if (result != CUDA_SUCCESS) {
    (void)cuMemAddressFree(reserved, bytes);
    return result;
  }
  *address = reserved;
  return CUDA_SUCCESS;
}

CUresult runtime_cuda_vmm_create(int ordinal, size_t bytes, uint64_t* handle,
                                 CUdeviceptr* address) {
  if (handle == NULL || address == NULL || bytes == 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUmemAllocationProp prop = runtime_cuda_vmm_prop(ordinal);
  CUmemGenericAllocationHandle created = 0;
  CUresult result = cuMemCreate(&created, bytes, &prop, 0);
  if (result != CUDA_SUCCESS) return result;
  CUdeviceptr mapped = 0;
  result = runtime_cuda_vmm_map(created, bytes, &mapped);
  if (result == CUDA_SUCCESS) {
    result = runtime_cuda_vmm_set_access(mapped, bytes, &ordinal, 1, 0);
    if (result != CUDA_SUCCESS) {
      (void)cuMemUnmap(mapped, bytes);
      (void)cuMemAddressFree(mapped, bytes);
    }
  }
  if (result != CUDA_SUCCESS) {
    (void)cuMemRelease(created);
    return result;
  }
  *handle = created;
  *address = mapped;
  return CUDA_SUCCESS;
}

CUresult runtime_cuda_vmm_export(uint64_t handle, int* fd) {
  if (fd == NULL) return CUDA_ERROR_INVALID_VALUE;
  int exported = -1;
  CUresult result = cuMemExportToShareableHandle(
      &exported, (CUmemGenericAllocationHandle)handle,
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
  if (result == CUDA_SUCCESS) *fd = exported;
  return result;
}

CUresult runtime_cuda_vmm_import(int fd, size_t bytes, uint64_t* handle,
                                 CUdeviceptr* address) {
  if (handle == NULL || address == NULL || bytes == 0 || fd < 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUmemGenericAllocationHandle imported = 0;
  CUresult result = cuMemImportFromShareableHandle(
      &imported, (void*)(uintptr_t)fd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
  if (result != CUDA_SUCCESS) return result;
  CUdeviceptr mapped = 0;
  result = runtime_cuda_vmm_map(imported, bytes, &mapped);
  if (result != CUDA_SUCCESS) {
    (void)cuMemRelease(imported);
    return result;
  }
  *handle = imported;
  *address = mapped;
  return CUDA_SUCCESS;
}

CUresult runtime_cuda_vmm_set_access(CUdeviceptr address, size_t bytes,
                                     const int* ordinals, size_t count,
                                     int read_only) {
  enum { MAX_ACCESS = 16 };
  if (ordinals == NULL || count == 0 || count > MAX_ACCESS) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUmemAccessDesc access[MAX_ACCESS];
  memset(access, 0, sizeof access);
  for (size_t i = 0; i < count; ++i) {
    access[i].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access[i].location.id = ordinals[i];
    access[i].flags = read_only ? CU_MEM_ACCESS_FLAGS_PROT_READ
                                : CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  }
  return cuMemSetAccess(address, bytes, access, count);
}

CUresult runtime_cuda_vmm_release(uint64_t handle, CUdeviceptr address,
                                  size_t bytes) {
  CUresult first = CUDA_SUCCESS;
  if (address != 0) {
    CUresult result = cuMemUnmap(address, bytes);
    if (result != CUDA_SUCCESS) first = result;
    result = cuMemAddressFree(address, bytes);
    if (result != CUDA_SUCCESS && first == CUDA_SUCCESS) first = result;
  }
  CUresult result = cuMemRelease((CUmemGenericAllocationHandle)handle);
  if (result != CUDA_SUCCESS && first == CUDA_SUCCESS) first = result;
  return first;
}

/* Observe free and total device memory in the current context. An observation
   describes one instant and is not an allocation guarantee. */
CUresult runtime_cuda_memory_info(size_t* free_bytes, size_t* total_bytes) {
    if (free_bytes == NULL || total_bytes == NULL) return CUDA_ERROR_INVALID_VALUE;
    return cuMemGetInfo(free_bytes, total_bytes);
}

/* Verify the physical UUID after CUDA applies visible-device remapping. */
CUresult runtime_cuda_device_uuid(unsigned char* bytes, int ordinal) {
    CUdevice device;
    CUuuid uuid;
    CUresult result = cuDeviceGet(&device, ordinal);
    if (result != CUDA_SUCCESS) return result;
    result = cuDeviceGetUuid_v2(&uuid, device);
    if (result == CUDA_SUCCESS) memcpy(bytes, uuid.bytes, 16);
    return result;
}

CUresult runtime_cuda_copy_pages(CUdeviceptr* destinations, CUdeviceptr* sources,
                                 size_t* sizes, size_t count, CUstream stream) {
  if (count == 0) return CUDA_SUCCESS;
  CUmemcpyAttributes attributes[1] = {0};
  attributes[0].srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;
  size_t attribute_indices[1] = {0};
  return cuMemcpyBatchAsync(destinations, sources, sizes, count,
                            attributes, attribute_indices, 1, stream);
}
