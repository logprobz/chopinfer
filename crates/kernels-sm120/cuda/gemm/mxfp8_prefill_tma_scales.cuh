#pragma once

// Stage code tiles with TMA and packed scale words with cp.async. Every
// output keeps four k32 MMAs from zero, then the ascending 128-K F32 fold.
#include "mxfp8_prefill_tma_matrix.cuh"

namespace chop::gemm::mxfp8_prefill_tma_scales {
constexpr int tile = 128, math_warps = 8;
constexpr int plane_bytes = tile * 128, code_bytes = 2 * plane_bytes;
constexpr int stage_bytes = code_bytes + 2 * tile * 4;
using mxfp8_prefill_tma_matrix::validate;
using mxfp8_prefill_tma_matrix::tail;

template<int N, int K, int Groups, int RowStride, int GroupStride,
         bool F32Output, bool OmitFirst = false, int Stages = 2>
__device__ void run(const CUtensorMap& activation_map, const CUtensorMap& weight_map,
    const uint8_t* activations, const uint8_t* weights,
    const uint8_t* activation_scales, const uint8_t* weight_scales,
    void* output, const uint32_t* live_rows, const uint32_t* status,
    uint64_t status_offset, uint32_t rows) {
  static_assert(Stages == 2);
  static_assert(K % 128 == 0 && RowStride % 128 == 0 && GroupStride % 128 == 0);
  constexpr int stages = Stages;
  static_assert(K > 0 && K % 32 == 0 && Groups > 0);
  static_assert(GroupStride >= K && RowStride >= (Groups - 1) * GroupStride + K);
  static_assert(RowStride % 32 == 0 && GroupStride % 32 == 0);
  __shared__ uint64_t full[stages], empty[stages];
  extern __shared__ uint8_t raw[];
  uint8_t* storage = prefill::align_swizzle_storage(raw);
  const uint32_t token0 = blockIdx.y * tile, column0 = blockIdx.x * tile;
  const uint32_t group = blockIdx.z;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  constexpr int folds = (K + 127) / 128;
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t live = min(rows, *live_rows);
  if (token0 >= live) return;
  // Only complete live token tiles may enter the TMA ring.
  if (token0 + tile > live) {
    tail<N, K, Groups, RowStride, GroupStride, F32Output, OmitFirst>(
        activations, weights, activation_scales, weight_scales, output,
        status, status_offset, live, token0, column0, group);
    return;
  }
  if (threadIdx.x == 0) {
    for (int s = 0; s < stages; ++s) {
      ptx::mbarrier_init(&full[s], 33);
      ptx::mbarrier_init(&empty[s], math_warps);
    }
    ptx::fence_mbarrier_init();
  }
  __syncthreads();
  if (warp >= math_warps) {
    ptx::setmaxnreg_dec<32>();
    if (warp != math_warps) return;
    for (int f = 0; f < folds; ++f) {
      const int slot = f % stages;
      if (f >= stages) prefill::wait_empty(empty, f, stages);
      if (lane == 0) {
        ptx::mbarrier_arrive_expect_tx(&full[slot], code_bytes);
        const int32_t ac[2] = {int32_t(group * GroupStride + f * 128), int32_t(token0)};
        const int32_t bc[2] = {f * 128, int32_t(group * N + column0)};
        ptx::tma_load<2>(storage + slot * stage_bytes, &activation_map, ac, &full[slot]);
        ptx::tma_load<2>(storage + slot * stage_bytes + plane_bytes, &weight_map, bc, &full[slot]);
      }
      // One producer warp fetches each four-scale word once. Its async
      // arrivals join the TMA transaction on the same full barrier.
      for (int local = lane; local < 2 * tile; local += 32) {
        const bool activation = local < tile;
        const uint32_t row = activation ? token0 + local : column0 + local - tile;
        const bool valid = activation || row < N;
        const uint64_t at = activation
            ? (uint64_t(row) * RowStride + group * GroupStride) / 32 + f * 4
            : (uint64_t(group) * N + row) * (K / 32) + f * 4;
        const uint8_t* src = activation ? activation_scales : weight_scales;
        const uint32_t dst = ptx::shared_address(storage + slot * stage_bytes + code_bytes + local * 4);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::
            "r"(dst), "l"(src + (valid ? at : 0)), "r"(valid ? 4 : 0) : "memory");
      }
      asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::
          "r"(ptx::shared_address(&full[slot])) : "memory");
    }
    asm volatile("cp.async.wait_all;" ::: "memory");
    return;
  }
  ptx::setmaxnreg_inc<232>();
  const int wm = warp / 4, wn = warp % 4, quad = lane / 4, lane4 = lane % 4;
  uint32_t active = 0;
#pragma unroll
  for (int m = 0; m < 4; ++m)
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const uint32_t row = token0 + wm * 64 + m * 16 + quad + half * 8;
      active |= uint32_t(row < live && status[status_offset + row] == 0) << (2 * m + half);
    }
  ptx::D4 total[4][4]{};
  for (int f = 0; f < folds; ++f) {
    const int slot = f % stages;
    prefill::wait_full(full, f, stages);
    const uint8_t* a = storage + slot * stage_bytes;
    const uint8_t* b = a + plane_bytes;
    ptx::D4 partial[4][4]{};
    const auto* scale_words = reinterpret_cast<const uint32_t*>(a + code_bytes);
#pragma unroll
    for (int step = 0; step < 4; ++step) {
      const int k = f * 128 + step * 32;
      ptx::A4 af[4]; ptx::B2 bf[4];
      uint32_t sa[4], sb[4];
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        const int local = wm * 64 + m * 16 + quad;
        const int matrix_row = wm * 64 + m * 16 + (lane & 15);
        const int matrix_k = step * 32 + (lane >> 4) * 16;
        ptx::ldmatrix_b16<4,false>(af[m].v,a+smem::swizzle_128(a,matrix_row*128+matrix_k,128));
#pragma unroll
        for (int i = 0; i < 4; ++i)
          if (!((active >> (2 * m + i % 2)) & 1) || k >= K) af[m].v[i] = 0;
        sa[m] = ((active >> (2 * m + (lane4 & 1))) & 1)
            ? (scale_words[local + (lane4 & 1) * 8] >> (step * 8)) & 255 : 127;
      }
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        const int local = wn * 32 + n * 8 + quad;
        const uint32_t col = column0 + local;
        const int matrix_row = wn * 32 + n * 8 + (lane & 7);
        const int matrix_k = step * 32 + ((lane >> 3) & 1) * 16;
        ptx::ldmatrix_b16<2,false>(bf[n].v,b+smem::swizzle_128(b,matrix_row*128+matrix_k,128));
#pragma unroll
        for (int i = 0; i < 2; ++i) if (col >= N || k >= K) bf[n].v[i] = 0;
        sb[n] = col < N ? (scale_words[tile + local] >> (step * 8)) & 255 : 127;
      }
#pragma unroll
      for (int m = 0; m < 4; ++m)
#pragma unroll
        for (int n = 0; n < 4; ++n)
          ptx::mma_mxf8f6f4<ptx::Element::e4m3, ptx::Element::e4m3>(
              partial[m][n], af[m], bf[n], sa[m], sb[n]);
    }
    // All generic-proxy shared reads finish before the ring's next TMA write.
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncwarp();
    if (lane == 0) ptx::mbarrier_arrive(&empty[slot]);
#pragma unroll
    for (int m = 0; m < 4; ++m)
#pragma unroll
      for (int n = 0; n < 4; ++n)
#pragma unroll
        for (int v = 0; v < 4; ++v) {
          if (!OmitFirst || f != 0) total[m][n].v[v] = __fadd_rn(total[m][n].v[v], partial[m][n].v[v]);
        }
  }
#pragma unroll
  for (int m = 0; m < 4; ++m)
#pragma unroll
    for (int n = 0; n < 4; ++n)
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        const uint32_t row = token0 + wm * 64 + m * 16 + quad + (v / 2) * 8;
        const uint32_t col = column0 + wn * 32 + n * 8 + lane4 * 2 + v % 2;
        if (row >= live || col >= N) continue;
        const uint64_t at = (uint64_t(row) * Groups + group) * N + col;
        const float value = ((active >> (2 * m + v / 2)) & 1) ? total[m][n].v[v] : 0.f;
        if constexpr (F32Output) reinterpret_cast<float*>(output)[at] = value;
        else reinterpret_cast<uint16_t*>(output)[at] = convert::bf16_from_f32(value);
      }
}
} // namespace chop::gemm::mxfp8_prefill_tma_scales
