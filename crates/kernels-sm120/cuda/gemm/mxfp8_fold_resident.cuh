#pragma once
#include "mxfp8_fold_stream.cuh"

namespace chop::gemm::mxfp8_fold_resident {

// Admission remains a row-wide operation before any output task. It also
// preserves sticky errors when a later replay presents valid operands.
template<int K, int Groups, int RowStride, int GroupStride>
__device__ void admit_task(uint32_t row, const uint8_t* codes,
    const uint8_t* scales, uint32_t* failed, const uint32_t* live_rows,
    uint32_t* status, uint64_t status_row_offset, uint32_t rows) {
  if (row >= min(rows, *live_rows)) return;
  bool invalid = false;
  for (int g = 0; g < Groups; ++g) {
    for (int k = threadIdx.x * 16; k < K; k += blockDim.x * 16) {
      const uint64_t at = uint64_t(row) * RowStride + uint64_t(g) * GroupStride + k;
      const uint4 packed = *reinterpret_cast<const uint4*>(codes + at);
      const uint8_t scale = scales[at / 32];
      invalid |= !(mxfp8_fold_ticket::admitted4(packed.x, scale) &&
                   mxfp8_fold_ticket::admitted4(packed.y, scale) &&
                   mxfp8_fold_ticket::admitted4(packed.z, scale) &&
                   mxfp8_fold_ticket::admitted4(packed.w, scale));
    }
  }
  const bool bad = __syncthreads_or(invalid);
  if (threadIdx.x == 0) {
    const uint32_t previous = atomicOr(status + status_row_offset + row, bad ? 1u : 0u);
    failed[row] = bad || previous != 0;
  }
}

} // namespace chop::gemm::mxfp8_fold_resident
