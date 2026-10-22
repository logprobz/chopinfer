#ifndef RUNTIME_CUDA_TENSOR_MAP_H_
#define RUNTIME_CUDA_TENSOR_MAP_H_

#include <cuda.h>
#include <stddef.h>
#include <stdint.h>

enum runtime_cuda_parameter_width {
  RUNTIME_CUDA_PARAMETER_U32 = 4,
  RUNTIME_CUDA_PARAMETER_U64 = 8,
  RUNTIME_CUDA_PARAMETER_BYTES64 = 64,
  RUNTIME_CUDA_PARAMETER_TENSOR_MAP = 128,
};

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
    size_t parameter_count);

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
    size_t parameter_count);

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

CUresult runtime_cuda_validate_tensor_map(
    const runtime_cuda_tensor_map_descriptor* descriptor,
    CUdeviceptr address,
    uint64_t buffer_bytes);
CUresult runtime_cuda_encode_tensor_map(
    void* output,
    const runtime_cuda_tensor_map_descriptor* descriptor,
    CUdeviceptr address,
    uint64_t buffer_bytes);

CUresult runtime_cuda_set_max_dynamic_shared(CUfunction function, int bytes);

/* Clusters of `function` at this launch shape that the device keeps resident
   at once (cuOccupancyMaxActiveClusters), with the cluster, and the
   cooperative flag when `cooperative`, as launch attributes: the descriptor
   the caller launches. The caller's context must be current. */
CUresult runtime_cuda_max_active_clusters(
    CUfunction function,
    unsigned int gx, unsigned int gy, unsigned int gz,
    unsigned int bx, unsigned int by, unsigned int bz,
    unsigned int shared_memory_bytes,
    unsigned int cx, unsigned int cy, unsigned int cz,
    int cooperative,
    int* clusters);

/* SMs the current context may use (its SM-count execution affinity), which
   an MPS thread limit or an SM partition lowers below the device's count. */
CUresult runtime_cuda_context_sm_count(int* sms);

CUresult runtime_cuda_memory_pool_set_device_access(
    CUmemoryPool pool,
    int device,
    int read_write);

CUresult runtime_cuda_memory_pool_get_device_access(
    CUmemoryPool pool,
    int device,
    int* read_write);

CUresult runtime_cuda_host_alloc(void** address, size_t bytes);
CUresult runtime_cuda_host_free(void* address);
/* Page-locked host memory the device reads through `device`, from every
 * context (DEVICEMAP | PORTABLE). Nothing is allocated on failure. */
CUresult runtime_cuda_host_alloc_mapped(void** address, CUdeviceptr* device, size_t bytes);
CUresult runtime_cuda_context_synchronize(void);

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
    void** parameters);

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
    void** parameters);

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
    void** parameters);

CUresult runtime_cuda_event_elapsed(float* milliseconds, CUevent start, CUevent end);

/* Exportable device memory (virtual memory management), for one process to
   lend a loaded image to another. A handle is the driver's
   CUmemGenericAllocationHandle; an allocation lives until every handle to it
   is released and every mapping of it is unmapped, in any process.

   The minimum granularity of an exportable allocation on `ordinal`. */
CUresult runtime_cuda_vmm_granularity(int ordinal, size_t* granularity);
/* Creates an exportable (POSIX file descriptor) allocation of `bytes`, a
   multiple of the granularity, on `ordinal`, maps it at a new address and
   grants `ordinal` read-write access. On failure nothing is left behind. */
CUresult runtime_cuda_vmm_create(int ordinal, size_t bytes, uint64_t* handle,
                                 CUdeviceptr* address);
/* A new POSIX file descriptor for `handle`, owned by the caller. */
CUresult runtime_cuda_vmm_export(uint64_t handle, int* fd);
/* Imports the allocation of `bytes` behind `fd` (which the caller keeps and
   closes) and maps it at a new address, with no access granted yet. On
   failure nothing is left behind. */
CUresult runtime_cuda_vmm_import(int fd, size_t bytes, uint64_t* handle,
                                 CUdeviceptr* address);
/* Grants each of `count` devices in `ordinals` read access to the mapping,
   or read-write access unless `read_only`. */
CUresult runtime_cuda_vmm_set_access(CUdeviceptr address, size_t bytes,
                                     const int* ordinals, size_t count,
                                     int read_only);
/* Unmaps `address` and frees its range, keeping the handle: the allocation
   stays for its handles and exports without a mapping in this process.
   Reports the first failure but still attempts the free. */
CUresult runtime_cuda_vmm_unmap(CUdeviceptr address, size_t bytes);
/* Unmaps and frees `address` (when nonzero) and releases `handle`. */
CUresult runtime_cuda_vmm_release(uint64_t handle, CUdeviceptr address,
                                  size_t bytes);

CUresult runtime_cuda_memory_info(size_t* free_bytes, size_t* total_bytes);

CUresult runtime_cuda_device_uuid(unsigned char* bytes, int ordinal);

#endif
