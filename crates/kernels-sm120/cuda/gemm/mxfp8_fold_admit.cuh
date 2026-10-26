#pragma once
#include "mxfp8_fold_resident.cuh"

namespace chop::gemm::mxfp8_fold_admit {

template<int K, int Groups, int RowStride, int GroupStride, bool Delayed>
__device__ void admit(const uint8_t* codes, const uint8_t* scales,
    uint8_t* workspace, const uint32_t* live_rows, uint32_t* status,
    uint64_t status_row_offset, uint32_t rows) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  if constexpr (Delayed) {
    const uint64_t start = clock64();
    while (clock64() - start < 1000000) __nanosleep(64);
  }
  mxfp8_fold_resident::admit_task<K,Groups,RowStride,GroupStride>(
      blockIdx.x,codes,scales,reinterpret_cast<uint32_t*>(workspace),
      live_rows,status,status_row_offset,rows);
}

} // namespace chop::gemm::mxfp8_fold_admit
