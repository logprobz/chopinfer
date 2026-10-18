#pragma once

#include <stdint.h>
#include "../common/convert.cuh"
#include "../common/ptx.cuh"

namespace chop::gemm::mxfp8_fold {

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

struct Operand {
  const uint8_t* codes;
  const uint8_t* scales;
  uint64_t offset;
  uint32_t stride;
  uint32_t rows;
  uint32_t k;
  const uint32_t* failed_rows;
  uint32_t token_base;
  bool modulo_scale_rows;

  __device__ __forceinline__ uint32_t scale(uint32_t row, uint32_t col) const {
    if (row >= rows || col >= k) return 127;
    if (failed_rows && failed_rows[row - token_base]) return 127;
    if (modulo_scale_rows) row %= 8;
    const uint8_t value = scales[(offset + uint64_t(row) * stride + col) / 32];
    return value;
  }

  __device__ __forceinline__ uint32_t packed(uint32_t row, uint32_t col) const {
    if (row >= rows || col >= k) return 0;
    if (failed_rows && failed_rows[row - token_base]) return 0;
    const uint64_t at = offset + uint64_t(row) * stride + col;
    uint32_t result = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint8_t code = codes[at + i];
      result |= uint32_t(code) << (i * 8);
    }
    return result;
  }
};

template<int N, int K, int Groups, int RowStride, int GroupStride,
         bool F32Output, bool Decode, bool SingleChain = false,
         int FoldK = 128, bool ModuloScaleRows = false, uint32_t NonFiniteFlag = 1>
__device__ void run(const uint8_t* activations, const uint8_t* activation_scales,
                    const uint8_t* weights, const uint8_t* weight_scales,
                    void* output, const uint32_t* live_rows, uint32_t* status,
                    uint64_t status_row_offset, uint32_t rows) {
  static_assert(K > 0 && K % 32 == 0 && Groups > 0);
  static_assert(GroupStride >= K && RowStride >= (Groups - 1) * GroupStride + K);
  static_assert(RowStride % 32 == 0 && GroupStride % 32 == 0);
  static_assert(FoldK == 128 || FoldK == 256);
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t live = min(rows, *live_rows);
  const uint32_t token_base = blockIdx.y * (Decode ? 8 : 16);
  if (token_base >= live) return;
  const uint32_t g = blockIdx.z;
  const uint32_t warp = threadIdx.x / 32;
  const uint32_t quad = (threadIdx.x & 31) / 4;
  const uint32_t lane4 = threadIdx.x & 3;
  const uint32_t column_base = blockIdx.x * (Decode ? 64 : 32);
  constexpr int TileRows = Decode ? 8 : 16;
  __shared__ uint32_t failed_rows[TileRows];
  // Every output tile validates complete activation rows across all groups.
  // Its decision cannot race a different tile's status publication, and a
  // single bad group makes every output of the row zero. Static weights have
  // already passed host payload admission before upload.
  for (uint32_t r = warp; r < TileRows; r += 4) {
    const uint32_t token = token_base + r;
    bool invalid = false;
    if (token < live) {
      for (uint32_t group = 0; group < Groups; ++group) {
        for (uint32_t col = threadIdx.x & 31; col < K; col += 32) {
          const uint64_t at = uint64_t(token) * RowStride + uint64_t(group) * GroupStride + col;
          invalid |= !admitted(activations[at], activation_scales[at / 32]);
        }
      }
    }
    invalid = __any_sync(0xffffffffu, invalid);
    if ((threadIdx.x & 31) == 0) {
      const uint32_t previous = token < live
          ? atomicOr(status + status_row_offset + token, invalid ? NonFiniteFlag : 0u) : 0;
      failed_rows[r] = invalid || previous != 0;
    }
  }
  __syncthreads();
  const Operand activation{activations, activation_scales, uint64_t(g) * GroupStride,
                           RowStride, live, K, failed_rows, token_base, false};
  const Operand weight{weights, weight_scales, uint64_t(g) * N * K, K, N, K,
                       nullptr, 0, ModuloScaleRows};
  const Operand a = Decode ? weight : activation;
  const Operand b = Decode ? activation : weight;
  const uint32_t ma = Decode ? column_base + warp * 16 : token_base;
  const uint32_t nb = Decode ? token_base : column_base + warp * 8;
  ptx::D4 total{};
  for (uint32_t fold = 0; fold < K; fold += FoldK) {
    ptx::D4 partial{};
    if constexpr (SingleChain) partial = total;
    // Production uses four native 32-K instructions. The lab's 256-K mutant
    // uses eight and must fail the exact fold witness.
#pragma unroll
    for (int step = 0; step < FoldK / 32; ++step) {
      const uint32_t col = fold + step * 32;
      ptx::A4 af{};
      ptx::B2 bf{};
#pragma unroll
      for (int i = 0; i < 4; ++i)
        af.v[i] = a.packed(ma + quad + (i % 2) * 8, col + (i / 2) * 16 + lane4 * 4);
#pragma unroll
      for (int i = 0; i < 2; ++i)
        bf.v[i] = b.packed(nb + quad, col + i * 16 + lane4 * 4);
      const uint32_t sa = a.scale(ma + quad + (lane4 & 1) * 8, col);
      const uint32_t sb = b.scale(nb + quad, col);
      ptx::mma_mxf8f6f4<ptx::Element::e4m3, ptx::Element::e4m3>(partial, af, bf, sa, sb);
    }
#pragma unroll
    for (int v = 0; v < 4; ++v) {
      if constexpr (SingleChain) total.v[v] = partial.v[v];
      else total.v[v] = __fadd_rn(total.v[v], partial.v[v]);
    }
  }
#pragma unroll
  for (int v = 0; v < 4; ++v) {
    const uint32_t m = ma + quad + (v / 2) * 8;
    const uint32_t n = nb + lane4 * 2 + (v % 2);
    const uint32_t token = Decode ? n : m;
    const uint32_t column = Decode ? m : n;
    if (token >= live || column >= N) continue;
    const uint64_t at = (uint64_t(token) * Groups + g) * N + column;
    const float value = failed_rows[token - token_base] ? 0.0f : total.v[v];
    if constexpr (F32Output) reinterpret_cast<float*>(output)[at] = value;
    else reinterpret_cast<uint16_t*>(output)[at] = convert::bf16_from_f32(value);
  }
}

} // namespace chop::gemm::mxfp8_fold
