#pragma once

// Shared-memory K/V tiles and fragment helpers used by prefill and decode.

#include <cuda_bf16.h>
#include <stdint.h>
#include "../common/convert.cuh"
#include "../common/ptx_control.cuh"
#include "../common/ptx_matrix.cuh"
#include "../common/ptx_tma.cuh"

namespace chop::attention {

// K and V tiles are stored as 64-dimension column blocks of 128-byte key
// rows with the TMA 128-byte swizzle: 16-byte sector s of key row r lives at
// sector s ^ ((r + phase) % 8), where phase is the base's 128-byte line.
template<int Keys>
__device__ __forceinline__ uint32_t tile_offset(int key, int dim, uint32_t phase) {
  return (dim / 64) * (Keys * 128) + key * 128 +
         ((((dim % 64) / 8) ^ ((key + phase) % 8)) * 16);
}

__device__ __forceinline__ void cp_async_zfill(void* dst, const void* src, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
               :: "r"(ptx::shared_address(dst)), "l"(src), "r"(valid ? 16 : 0)
               : "memory");
}
__device__ __forceinline__ void prefetch_l2(const void* src, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
               :: "l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}
template<int Pending>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" :: "n"(Pending) : "memory");
}

__device__ __forceinline__ float bf16_low(uint32_t pair) {
  return __uint_as_float(pair << 16);
}
__device__ __forceinline__ float bf16_high(uint32_t pair) {
  return __uint_as_float(pair & 0xffff0000u);
}
__device__ __forceinline__ uint32_t bf16_pair(float low, float high) {
  const __nv_bfloat162 pair = __floats2bfloat162_rn(low, high);
  return *reinterpret_cast<const uint32_t*>(&pair);
}
// Transposes an 8x8 BF16 fragment across the warp: the lane holding row g,
// columns 2c and 2c + 1 receives row g, columns 2c and 2c + 1 of the
// transpose.
__device__ __forceinline__ uint32_t movmatrix_trans(uint32_t pair) {
  uint32_t out;
  asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(out) : "r"(pair));
  return out;
}

// Widens an unswizzled FP8 E4M3 [key][dim] tile into the swizzled BF16
// layout. E4M3 values are exact in BF16.
template<int Keys, int Dim, int Threads>
__device__ __forceinline__ void widen_fp8(uint8_t* dst, const uint8_t* src, uint32_t phase) {
  for (int i = threadIdx.x; i < Keys * Dim / 16; i += Threads) {
    const int row = i / (Dim / 16);
    const int group = i % (Dim / 16);
    const uint4 bytes = *reinterpret_cast<const uint4*>(src + row * Dim + group * 16);
    const uint32_t words[4] = {bytes.x, bytes.y, bytes.z, bytes.w};
    uint32_t wide[8];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      wide[2 * w] = convert::unpack_bf16x2_e4m3(uint16_t(words[w]));
      wide[2 * w + 1] = convert::unpack_bf16x2_e4m3(uint16_t(words[w] >> 16));
    }
    *reinterpret_cast<uint4*>(dst + tile_offset<Keys>(row, group * 16, phase)) =
        make_uint4(wide[0], wide[1], wide[2], wide[3]);
    *reinterpret_cast<uint4*>(dst + tile_offset<Keys>(row, group * 16 + 8, phase)) =
        make_uint4(wide[4], wide[5], wide[6], wide[7]);
  }
}

// The probability of each key is carried as this many BF16 fragments (the
// rounded value and its rounded residuals) for a row that sees `keys` keys.
// The dropped residual is at most 2^(-8 terms) of each |p v| product, within
// half of the decision-0081 term 2^-22 K S at these thresholds.
__device__ __forceinline__ int probability_terms(int keys) {
  return keys >= 32768 ? 1 : keys >= 128 ? 2 : 3;
}

} // namespace chop::attention
