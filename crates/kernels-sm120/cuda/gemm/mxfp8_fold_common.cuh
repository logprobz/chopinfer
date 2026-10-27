#pragma once

#include <stdint.h>
#include <type_traits>
#include "../common/convert.cuh"
#include "../common/ptx.cuh"
#include "fold_rows.cuh"

namespace chop::gemm::mxfp8_fold_policy {

// The trained E4M3 grid is already materialized at the entry boundary.
struct E4m3Ceil32 {};
struct Ue8m0AtK32 {};
struct FourK32 {};
struct AscendingF32 {};

// A new arithmetic policy needs its own numerical identity and qualification.
// Scheduling knobs cannot substitute another activation or reduction policy.
template<class Activation, class Scale, class Chain, class Reduction>
inline constexpr bool qualified =
    std::is_same_v<Activation, E4m3Ceil32> &&
    std::is_same_v<Scale, Ue8m0AtK32> &&
    std::is_same_v<Chain, FourK32> &&
    std::is_same_v<Reduction, AscendingF32>;

} // namespace chop::gemm::mxfp8_fold_policy

// Addressing of E4M3 weight codes stored as `mxfp8-fold-packed-8x128-v1`
// (`kernel_core::weight_transform::mxfp8_fold`): the layout the stream and
// direct folds read, and the one every fold reader moves to so the row-major
// copy can go.
//
// Rows are counted across groups (row `g * N + n`); N is a multiple of 8 and
// K a multiple of 64. Each eight-row strip is 8 * K bytes: its 128-K folds in
// order, each 1024 bytes, then a final 512-byte fold when K % 128 == 64.
// Inside a fold, row r of the strip owns one contiguous 128-byte segment
// (64 bytes in the final half fold). The four-byte code word holding K bytes
// [16 * c + 4 * l, 16 * c + 4 * l + 4) of the fold sits at byte 32 * l + 4 * c
// of that segment (16 * l + 4 * c in a half fold): lane l of the row's four
// lanes owns words c = 0..7 (0..3), every fourth word of the row.
//
// A row-major reader that loads a K word becomes a packed reader by changing
// only that address; a reader that stages 16-byte row chunks stages the
// strip's contiguous fold instead and reads its words at `segment_word`.
// Neither changes arithmetic or accumulation order.
namespace chop::gemm::mxfp8_fold_packed {

// Bytes of fold `fold` of one strip: 1024, or 512 for a final 64-K fold.
__device__ __host__ __forceinline__ constexpr uint32_t fold_bytes(uint32_t fold, uint32_t k) {
  return (k % 128 != 0 && fold == k / 128) ? 512u : 1024u;
}

// Byte offset, within the strip's fold `fold`, of the word holding K bytes
// [local, local + 4) of the fold for strip row `r` (local % 4 == 0).
__device__ __host__ __forceinline__ constexpr uint32_t segment_word(
    uint32_t r, uint32_t local, bool half) {
  const uint32_t chunk = local / 16, lane = (local % 16) / 4;
  return half ? r * 64 + lane * 16 + chunk * 4 : r * 128 + lane * 32 + chunk * 4;
}

// Byte offset of strip `strip`'s fold `fold` in a matrix of K-wide rows.
__device__ __host__ __forceinline__ constexpr uint64_t fold_offset(
    uint32_t strip, uint32_t fold, uint32_t k) {
  return uint64_t(strip) * 8 * k + uint64_t(fold) * 1024;
}

// Byte offset of the word holding K bytes [k, k + 4) of row `row`, which a
// row-major reader finds at row * K + k (k % 4 == 0).
__device__ __host__ __forceinline__ constexpr uint64_t word_offset(
    uint32_t row, uint32_t k, uint32_t width) {
  const uint32_t fold = k / 128;
  const bool half = width % 128 != 0 && fold == width / 128;
  return fold_offset(row / 8, fold, width) + segment_word(row % 8, k % 128, half);
}

// The word itself, for a reader that loads four codes at once.
__device__ __forceinline__ uint32_t load_word(const uint8_t* packed, uint32_t row, uint32_t k,
                                              uint32_t width) {
  return *reinterpret_cast<const uint32_t*>(packed + word_offset(row, k, width));
}

}  // namespace chop::gemm::mxfp8_fold_packed

namespace chop::gemm::mxfp8_fold_common {

// The native instruction sees the original E4M3 and UE8M0 bytes. Admission
// agrees with the exact BF16 operand contract, without decoding or rounding.
__device__ __forceinline__ bool admitted(uint8_t code, uint8_t scale) {
  const unsigned mag = code & 127;
  if (mag == 127 || scale == 255) return false;
  if (!mag) return true;
  const unsigned exponent = mag >> 3;
  const unsigned significand = exponent ? 8 + (mag & 7) : mag & 7;
  if (exponent && unsigned(scale) + exponent > 261) return false;
  const int power = int(scale) + (exponent ? int(exponent) - 137 : -136);
  const int shift = -133 - power;
  return shift <= 0 || (shift <= 3 && !(significand & ((1u << shift) - 1)));
}

__device__ __forceinline__ bool admitted4(uint32_t codes, uint8_t scale) {
  // Every finite E4M3 code is BF16-exact in this UE8M0 interval. Detect
  // magnitude 127 in four bytes without decoding or cross-byte carries.
  if (scale >= 3 && scale <= 246)
    return (((codes & 0x7f7f7f7fu) + 0x01010101u) & 0x80808080u) == 0;
  bool valid = true;
#pragma unroll
  for (int i = 0; i < 4; ++i) valid &= admitted(uint8_t(codes >> (i * 8)), scale);
  return valid;
}

__device__ __forceinline__ void copy16(void* dst, const void* src, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
      :: "r"(uint32_t(__cvta_generic_to_shared(dst))), "l"(src), "r"(valid ? 16 : 0) : "memory");
}

// A packed strip keeps its contiguous fold in one 1024-byte shared slot.
// A final half fold occupies its first 512 bytes; the rest is zero-filled.
template<int N, int K>
__device__ __forceinline__ void copy_packed_chunk(uint8_t* dst, const uint8_t* weights,
    uint32_t group, uint32_t column_base, uint32_t chunk, uint32_t fold, uint32_t end) {
  const uint32_t strip = chunk / 64;
  const uint32_t byte = (chunk % 64) * 16;
  const bool valid = fold < end && column_base + strip * 8 < N &&
      byte < mxfp8_fold_packed::fold_bytes(fold, K);
  const uint64_t at = mxfp8_fold_packed::fold_offset(
      (group * N + column_base) / 8 + strip, fold, K) + byte;
  copy16(dst + fold_row_chunk(chunk / 8, chunk % 8), weights + (valid ? at : 0), valid);
}

// One aligned vector supplies two unchanged MMA steps. Keep the vector
// shared load explicit so the packed lane runs use the swizzle together.
template<int K>
__device__ __forceinline__ uint4 packed_fragment(const uint8_t* codes,
    uint32_t warp, uint32_t quad, uint32_t lane, uint32_t step, uint32_t fold) {
  const bool half = K % 128 != 0 && fold == K / 128;
  if (half && step >= 2) return uint4{};
  const uint32_t at = warp * 1024 +
      mxfp8_fold_packed::segment_word(quad, step * 32 + lane * 4, half);
  const uint8_t* address = codes + fold_row_chunk(at / 128, (at % 128) / 16);
  uint4 value;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
      : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
      : "r"(uint32_t(__cvta_generic_to_shared(address))));
  return value;
}

template<int K>
__device__ __forceinline__ uint4 split_fragment(const uint8_t* codes,
    uint32_t warp, uint32_t quad, uint32_t lane, uint32_t step, uint32_t fold) {
  const bool half = K % 128 != 0 && fold == K / 128;
  const uint32_t at = warp * 1024 +
      mxfp8_fold_packed::segment_word(quad, (half ? step % 2 : step) * 32 + lane * 4, half)
      + (half ? step / 2 * 512 : 0);
  const uint8_t* address = codes + fold_row_chunk(at / 128, (at % 128) / 16);
  uint4 value;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
      : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
      : "r"(uint32_t(__cvta_generic_to_shared(address))));
  return value;
}

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
      invalid |= !(admitted4(packed.x, scale) &&
                   admitted4(packed.y, scale) &&
                   admitted4(packed.z, scale) &&
                   admitted4(packed.w, scale));
    }
  }
  const bool bad = __syncthreads_or(invalid);
  if (threadIdx.x == 0) {
    const uint32_t previous = atomicOr(status + status_row_offset + row, bad ? 1u : 0u);
    failed[row] = bad || previous != 0;
  }
}

struct AscendingF32 {
  static __device__ __forceinline__ float add(float sum, float partial) {
    return __fadd_rn(sum, partial);
  }
};

// Load sixteen consecutive fold partials before adding them in K order.
// The cache operator and grouping are part of the existing schedule.
template<int N, int Groups, int Folds, class Reduction>
__device__ __forceinline__ float ordered_merge(const float* partials, uint32_t rows,
                                              uint64_t at) {
  float total = 0.0f;
  for (uint32_t base = 0; base < Folds; base += 16) {
    float chain[16];
#pragma unroll
    for (int j = 0; j < 16; ++j)
      if (base + j < Folds) chain[j] = __ldcg(partials + uint64_t(base + j) * rows * Groups * N + at);
#pragma unroll
    for (int j = 0; j < 16; ++j)
      if (base + j < Folds) total = Reduction::add(total, chain[j]);
  }
  return total;
}

} // namespace chop::gemm::mxfp8_fold_common

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
  mxfp8_fold_common::admit_task<K,Groups,RowStride,GroupStride>(
      blockIdx.x,codes,scales,reinterpret_cast<uint32_t*>(workspace),
      live_rows,status,status_row_offset,rows);
}

} // namespace chop::gemm::mxfp8_fold_admit
