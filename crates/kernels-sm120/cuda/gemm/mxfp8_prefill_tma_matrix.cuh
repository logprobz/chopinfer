#pragma once

// Warp-specialized TMA delivery with matrix fragment loads. Every output
// keeps four k32 MMAs from zero and the ascending 128-K F32 fold. Scales
// retain their existing row-major representation.
#include "prefill_mainloop.cuh"
#include "mxfp8_fold.cuh"
#include "mxfp8_fold_packed.cuh"
#include "../common/ptx_matrix.cuh"

namespace chop::gemm::mxfp8_prefill_tma_matrix {
constexpr int tile = 128, math_warps = 8;
constexpr int plane_bytes = tile * 128, stage_bytes = 2 * plane_bytes;

// Admit each activation row once across every group, before any tile reads
// status. Preserve the caller's status bits and the existing invalid domain.
template<int K, int Groups, int RowStride, int GroupStride>
__device__ void validate(const uint8_t* codes, const uint8_t* scales,
    const uint32_t* live_rows, uint32_t* status, uint64_t offset, uint32_t rows) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= min(rows, *live_rows)) return;
  bool bad = false;
  for (int group = 0; group < Groups; ++group)
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
      const uint64_t at = uint64_t(row) * RowStride + group * GroupStride + k;
      bad |= !mxfp8_fold::admitted(codes[at], scales[at / 32]);
    }
  const bool failed = __syncthreads_or(bad);
  if (threadIdx.x == 0) atomicOr(status + offset + row, failed ? 1u : 0u);
}

// Four adjacent code bytes share an aligned word. A 128-byte swizzled row
// keeps each warp's words in separate banks, without four byte loads.
__device__ __forceinline__ uint32_t word(const uint8_t* base, int row, int col) {
  const uint32_t address = ptx::shared_address(base) +
      smem::swizzle_128(base, row * 128 + col, 128);
  uint32_t value;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(value) : "r"(address));
  return value;
}

// The final partial token tile uses predicated wide global loads. A TMA
// box must never read a token beyond the device live bound. Only this
// boundary tile takes the smaller register loop; complete tiles use TMA.
// PackedWeights reads mxfp8-fold-packed-8x128-v1 weight codes.
template<int N, int K, int Groups, int RowStride, int GroupStride,
         bool F32Output, bool OmitFirst, bool PackedWeights = false>
__device__ __forceinline__ void tail(
    const uint8_t* activations, const uint8_t* weights,
    const uint8_t* activation_scales, const uint8_t* weight_scales,
    void* output, const uint32_t* status, uint64_t status_offset,
    uint32_t live, uint32_t token0, uint32_t column0, uint32_t group) {
  static_assert(K > 0 && K % 32 == 0 && Groups > 0);
  static_assert(GroupStride >= K && RowStride >= (Groups - 1) * GroupStride + K);
  static_assert(RowStride % 32 == 0 && GroupStride % 32 == 0);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (warp >= math_warps) return;
  const int wm = warp / 4, wn = warp % 4, quad = lane / 4, lane4 = lane % 4;
#pragma unroll 1
  for (int m = 0; m < 4; ++m)
#pragma unroll 1
    for (int n = 0; n < 4; ++n) {
      ptx::D4 total{};
      const uint32_t ma = token0 + wm * 64 + m * 16;
      const uint32_t nb = column0 + wn * 32 + n * 8;
      for (int f = 0; f < (K + 127) / 128; ++f) {
        ptx::D4 partial{};
#pragma unroll
        for (int step = 0; step < 4; ++step) {
          const int k = f * 128 + step * 32;
          ptx::A4 a{}; ptx::B2 b{};
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const uint32_t row = ma + quad + (i % 2) * 8;
            if (row < live && k < K && status[status_offset + row] == 0)
              a.v[i] = *reinterpret_cast<const uint32_t*>(activations +
                  uint64_t(row) * RowStride + group * GroupStride + k + (i / 2) * 16 + lane4 * 4);
          }
          const uint32_t col = nb + quad;
#pragma unroll
          for (int i = 0; i < 2; ++i)
            if (col < N && k < K) {
              if constexpr (PackedWeights) {
                // word_offset with its parts known: the strip row is quad
                // (column tiles start on strips), the word is 2 * step + i
                // of lane4's run, and a half fold's segments and runs are
                // half as wide.
                const uint32_t half = K % 128 != 0 && f == K / 128;
                const uint64_t at = mxfp8_fold_packed::fold_offset((group * N + col) / 8, f, K) +
                    ((quad * 128 + lane4 * 32) >> half) + (step * 2 + i) * 4;
                b.v[i] = *reinterpret_cast<const uint32_t*>(weights + at);
              }
              else
                b.v[i] = *reinterpret_cast<const uint32_t*>(weights +
                    (uint64_t(group) * N + col) * K + k + i * 16 + lane4 * 4);
            }
          const uint32_t row = ma + quad + (lane4 & 1) * 8;
          const uint32_t sa = row < live && k < K && status[status_offset + row] == 0
              ? activation_scales[(uint64_t(row) * RowStride + group * GroupStride + k) / 32] : 127;
          const uint32_t sb = col < N && k < K
              ? weight_scales[(uint64_t(group) * N + col) * (K / 32) + k / 32] : 127;
          ptx::mma_mxf8f6f4<ptx::Element::e4m3, ptx::Element::e4m3>(partial, a, b, sa, sb);
        }
#pragma unroll
        for (int v = 0; v < 4; ++v) {
          if (!OmitFirst || f != 0) total.v[v] = __fadd_rn(total.v[v], partial.v[v]);
        }
      }
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        const uint32_t row = ma + quad + (v / 2) * 8;
        const uint32_t col = nb + lane4 * 2 + v % 2;
        if (row >= live || col >= N) continue;
        const uint64_t at = (uint64_t(row) * Groups + group) * N + col;
        const float value = status[status_offset + row] ? 0.f : total.v[v];
        if constexpr (F32Output) reinterpret_cast<float*>(output)[at] = value;
        else reinterpret_cast<uint16_t*>(output)[at] = convert::bf16_from_f32(value);
      }
    }
}

template<int N, int K, int Groups, int RowStride, int GroupStride,
         bool F32Output, bool OmitFirst = false, int Stages = 2>
__device__ void run(const CUtensorMap& activation_map, const CUtensorMap& weight_map,
    const uint8_t* activations, const uint8_t* weights,
    const uint8_t* activation_scales, const uint8_t* weight_scales,
    void* output, const uint32_t* live_rows, const uint32_t* status,
    uint64_t status_offset, uint32_t rows) {
  static_assert(Stages == 2 || Stages == 3);
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
  // A partial K box may not read padding or the next activation group.
  // Dense unpadded rows use the tensor map's out-of-bounds zero fill.
  if (token0 + tile > live || (K % 128 != 0 && RowStride > K)) {
    tail<N, K, Groups, RowStride, GroupStride, F32Output, OmitFirst>(
        activations, weights, activation_scales, weight_scales, output,
        status, status_offset, live, token0, column0, group);
    return;
  }
  if (threadIdx.x == 0) {
    for (int s = 0; s < stages; ++s) {
      ptx::mbarrier_init(&full[s], 1);
      ptx::mbarrier_init(&empty[s], math_warps);
    }
    ptx::fence_mbarrier_init();
  }
  __syncthreads();
  if (warp >= math_warps) {
    ptx::setmaxnreg_dec<24>();
    if (warp != math_warps || lane != 0) return;
    const auto weight = [&](int fold, int slot) {
      const int32_t coord[2] = {fold * 128, int32_t(group * N + column0)};
      ptx::tma_load<2>(storage + slot * stage_bytes + plane_bytes,
          &weight_map, coord, &full[slot]);
    };
    // The full token tile is inside the live bound before either map reads.
    for (int f = 0; f < min(folds, stages); ++f) {
      ptx::mbarrier_arrive_expect_tx(&full[f], stage_bytes);
      weight(f, f);
    }
    for (int f = 0; f < folds; ++f) {
      const int slot = f % stages;
      if (f >= stages) {
        prefill::wait_empty(empty, f, stages);
        ptx::mbarrier_arrive_expect_tx(&full[slot], stage_bytes);
        weight(f, slot);
      }
      const int32_t coord[2] = {int32_t(group * GroupStride + f * 128), int32_t(token0)};
      ptx::tma_load<2>(storage + slot * stage_bytes, &activation_map, coord, &full[slot]);
    }
    return;
  }
  ptx::setmaxnreg_inc<240>();
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
    // Short complete K panels spill address temporaries when these four
    // steps are unrolled. Both bodies keep the same four ordered MMAs;
    // only the loop hint differs. Keep the full-unroll body in place to
    // preserve existing locked instruction streams, including partial K.
    if constexpr (K < 1024 && K % 128 == 0) {
#pragma unroll 1
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
          const uint32_t row = token0 + local + (lane4 & 1) * 8;
          sa[m] = ((active >> (2 * m + (lane4 & 1))) & 1) && k < K
              ? activation_scales[(uint64_t(row) * RowStride + group * GroupStride + k) / 32] : 127;
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
          sb[n] = col < N && k < K
              ? weight_scales[(uint64_t(group) * N + col) * (K / 32) + k / 32] : 127;
        }
#pragma unroll
        for (int m = 0; m < 4; ++m)
#pragma unroll
          for (int n = 0; n < 4; ++n)
            ptx::mma_mxf8f6f4<ptx::Element::e4m3, ptx::Element::e4m3>(
                partial[m][n], af[m], bf[n], sa[m], sb[n]);
      }
    } else {
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
          const uint32_t row = token0 + local + (lane4 & 1) * 8;
          sa[m] = ((active >> (2 * m + (lane4 & 1))) & 1) && k < K
              ? activation_scales[(uint64_t(row) * RowStride + group * GroupStride + k) / 32] : 127;
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
          sb[n] = col < N && k < K
              ? weight_scales[(uint64_t(group) * N + col) * (K / 32) + k / 32] : 127;
        }
#pragma unroll
        for (int m = 0; m < 4; ++m)
#pragma unroll
          for (int n = 0; n < 4; ++n)
            ptx::mma_mxf8f6f4<ptx::Element::e4m3, ptx::Element::e4m3>(
                partial[m][n], af[m], bf[n], sa[m], sb[n]);
      }
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
} // namespace chop::gemm::mxfp8_prefill_tma_matrix
