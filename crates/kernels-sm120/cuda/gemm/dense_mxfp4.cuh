#pragma once

#include "dense.cuh"
#include "../common/mxfp4.cuh"

namespace chop::gemm {

// A16 FP4 staging uses 256-value tiles. Keep this specialization outside
// dense.cuh so other formats retain their adopted source closures.
template<>
__device__ __forceinline__ uint32_t packed_fp4_weight<256, WeightKind::mxfp4>(
    const uint8_t* base, int local_row, int k, int global_row,
    int global_k, int total_n, int total_k, const float* scales,
    const float* global_scale) {
  if (global_row >= total_n || global_k >= total_k) return 0;
  const uint32_t off = smem::swizzle_128(base, local_row * 128 + k / 2, 128);
  // Retain the original unpack expression to preserve the qualified cubin.
  const convert::Float2 values = convert::unpack_f32_e2m1(base[off]);
  const uint8_t* codes = reinterpret_cast<const uint8_t*>(scales);
  const uint8_t code =
      codes[uint64_t(global_row) * (total_k / 32) + global_k / 32];
  return convert::scaled_mxfp4_bf16x2(base[off], code);
}

} // namespace chop::gemm
