#pragma once

#include "mxfp8_fold_ticket.cuh"

namespace chop::gemm::mxfp8_fold_prefetch {

using mxfp8_fold_ticket::admitted4;
using mxfp8_fold_ticket::copy16;
using mxfp8_fold_ticket::header;
using mxfp8_fold_ticket::tiles;

template<int N, int K, int Groups, int RowStride, int GroupStride, int TileN, bool Paired>
__device__ void initialize(const uint8_t* codes, const uint8_t* scales, uint8_t* workspace,
                          const uint32_t* live_rows, uint32_t* status,
                          uint64_t status_row_offset, uint32_t rows) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  auto* tickets = reinterpret_cast<uint32_t*>(workspace + 256);
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < tiles<N, Groups, TileN, Paired>(rows); i += gridDim.x * blockDim.x)
    tickets[i] = 0;
  const uint32_t row = blockIdx.x;
  if (row >= min(rows, *live_rows)) return;
  bool invalid = false;
  for (int g = 0; g < Groups; ++g) {
    for (int k = threadIdx.x * 16; k < K; k += blockDim.x * 16) {
      const uint64_t at = uint64_t(row) * RowStride + uint64_t(g) * GroupStride + k;
      const uint4 packed = *reinterpret_cast<const uint4*>(codes + at);
      const uint8_t scale = scales[at / 32];
      invalid |= !(admitted4(packed.x, scale) && admitted4(packed.y, scale) &&
                   admitted4(packed.z, scale) && admitted4(packed.w, scale));
    }
  }
  const bool bad = __syncthreads_or(invalid);
  if (threadIdx.x == 0) {
    const uint32_t previous = atomicOr(status + status_row_offset + row, bad ? 1u : 0u);
    reinterpret_cast<uint32_t*>(workspace)[row] = bad || previous != 0;
  }
}

template<int N, int K, int Groups, int RowStride, int GroupStride,
         int TileN, bool F32Output, bool Paired>
__device__ void run(const uint8_t* activations, const uint8_t* activation_scales,
                   const uint8_t* weights, const uint8_t* weight_scales,
                   const uint8_t* weights2, const uint8_t* weight_scales2,
                   void* output, void* output2, uint8_t* workspace,
                   const uint32_t* live_rows, uint32_t rows, uint32_t folds_per_cta) {
  static_assert(TileN == 32 || TileN == 64);
  constexpr int Folds = (K + 127) / 128;
  constexpr int Threads = TileN * 4;
  ptx::griddep_launch_dependents();
  const uint32_t splits = (Folds + folds_per_cta - 1) / folds_per_cta;
  const uint32_t bank = blockIdx.z / (splits * Groups);
  const uint32_t g = (blockIdx.z / splits) % Groups;
  const uint32_t first = (blockIdx.z % splits) * folds_per_cta;
  const uint32_t end = min(first + folds_per_cta, uint32_t(Folds));
  const uint32_t token_base = blockIdx.y * 16;
  const uint32_t column_base = blockIdx.x * TileN;
  const auto* failed = reinterpret_cast<const uint32_t*>(workspace);
  auto* tickets = reinterpret_cast<uint32_t*>(workspace + 256);
  auto* partials = reinterpret_cast<float*>(workspace + header<N, Groups, TileN, Paired>(rows));
  partials += uint64_t(bank) * Folds * rows * Groups * N;
  if constexpr (Paired) {
    if (bank) { weights = weights2; weight_scales = weight_scales2; output = output2; }
  }
  __shared__ __align__(16) uint8_t codes[3][(16 + TileN) * 128];
  __shared__ __align__(4) uint8_t scales[3][(16 + TileN) * 4];
  __shared__ uint32_t row_failed[16];
  __shared__ bool last;
  // All destinations here are private CTA storage. Immutable weights are
  // the only global inputs read before the predecessor completes.
  const auto load_weights = [&](uint32_t stage, uint32_t fold) {
    for (uint32_t chunk = threadIdx.x; chunk < TileN * 8; chunk += Threads) {
      const uint32_t row = column_base + chunk / 8;
      const uint32_t col = fold * 128 + (chunk % 8) * 16;
      const bool valid = fold < end && row < N && col < K;
      const uint64_t at = (uint64_t(g) * N + row) * K + col;
      copy16(codes[stage] + 16 * 128 + chunk * 16, weights + (valid ? at : 0), valid);
    }
    if (threadIdx.x < TileN) {
      const uint32_t row = column_base + threadIdx.x;
      const bool valid = fold < end && row < N;
      auto* dst = scales[stage] + (16 + threadIdx.x) * 4;
      const uint64_t at = (uint64_t(g) * N + row) * K + fold * 128;
      if (valid && fold * 128 + 128 <= K && K % 128 == 0) {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"
            :: "r"(uint32_t(__cvta_generic_to_shared(dst))), "l"(weight_scales + at / 32) : "memory");
      } else {
#pragma unroll
        for (int i = 0; i < 4; ++i)
          dst[i] = valid && fold * 128 + i * 32 < K ? weight_scales[at / 32 + i] : 127;
      }
    }
  };
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    load_weights(i, first + i);
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
  // Warm the remaining immutable row segments in L2 while admission runs,
  // using the same bulk-prefetch instruction as the weight-prefetch engine.
  if (threadIdx.x < TileN && column_base + threadIdx.x < N && first + 3 < end) {
    const uint32_t begin = (first + 3) * 128;
    const uint32_t bytes = min(end * 128, uint32_t(K)) - begin;
    const uint64_t at = (uint64_t(g) * N + column_base + threadIdx.x) * K + begin;
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
        :: "l"(weights + at), "r"(bytes) : "memory");
  }
  ptx::griddep_wait();
  const uint32_t live = min(rows, *live_rows);
  if (token_base >= live) {
    // Do not release CTA storage with asynchronous writes still pending.
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    return;
  }
  if (threadIdx.x < 16)
    row_failed[threadIdx.x] = token_base + threadIdx.x < live ? failed[token_base + threadIdx.x] : 1;
  __syncthreads();
  const auto load_activations = [&](uint32_t stage, uint32_t fold) {
    for (uint32_t chunk = threadIdx.x; chunk < 16 * 8; chunk += Threads) {
      const uint32_t r = chunk / 8;
      const uint32_t row = token_base + r;
      const uint32_t col = fold * 128 + (chunk % 8) * 16;
      const bool valid = fold < end && row < live && col < K && !row_failed[r];
      const uint64_t at = uint64_t(row) * RowStride + uint64_t(g) * GroupStride + col;
      copy16(codes[stage] + chunk * 16, activations + (valid ? at : 0), valid);
    }
    if (threadIdx.x < 16) {
      const uint32_t row = token_base + threadIdx.x;
      const bool valid = fold < end && row < live && !row_failed[threadIdx.x];
      auto* dst = scales[stage] + threadIdx.x * 4;
      const uint64_t at = uint64_t(row) * RowStride + uint64_t(g) * GroupStride + fold * 128;
      if (valid && fold * 128 + 128 <= K && RowStride % 128 == 0 && GroupStride % 128 == 0) {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"
            :: "r"(uint32_t(__cvta_generic_to_shared(dst))), "l"(activation_scales + at / 32) : "memory");
      } else {
#pragma unroll
        for (int i = 0; i < 4; ++i)
          dst[i] = valid && fold * 128 + i * 32 < K ? activation_scales[at / 32 + i] : 127;
      }
    }
  };
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    load_activations(i, first + i);
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
  const auto load = [&](uint32_t stage, uint32_t fold) {
    load_weights(stage, fold);
    load_activations(stage, fold);
    asm volatile("cp.async.commit_group;" ::: "memory");
  };
  const uint32_t warp = threadIdx.x / 32;
  const uint32_t quad = (threadIdx.x & 31) / 4;
  const uint32_t lane4 = threadIdx.x & 3;
  for (uint32_t base = first; base < end; base += 2) {
    asm volatile("cp.async.wait_group 1;" ::: "memory");
    __syncthreads();
    ptx::D4 raw[2]{};
#pragma unroll
    for (int step = 0; step < 4; ++step) {
#pragma unroll
      for (int pair = 0; pair < 2; ++pair) {
        const uint32_t slot = (base - first + pair) % 3;
        ptx::A4 af{};
        ptx::B2 bf{};
#pragma unroll
        for (int i = 0; i < 4; ++i)
          af.v[i] = *reinterpret_cast<const uint32_t*>(codes[slot] +
              (quad + (i % 2) * 8) * 128 + step * 32 + (i / 2) * 16 + lane4 * 4);
#pragma unroll
        for (int i = 0; i < 2; ++i)
          bf.v[i] = *reinterpret_cast<const uint32_t*>(codes[slot] +
              (16 + warp * 8 + quad) * 128 + step * 32 + i * 16 + lane4 * 4);
        const uint32_t sa = scales[slot][(quad + (lane4 & 1) * 8) * 4 + step];
        const uint32_t sb = scales[slot][(16 + warp * 8 + quad) * 4 + step];
        ptx::mma_mxf8f6f4<ptx::Element::e4m3, ptx::Element::e4m3>(raw[pair], af, bf, sa, sb);
      }
    }
#pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
      if (base + pair < end) {
#pragma unroll
        for (int v = 0; v < 4; ++v) {
          const uint32_t row = token_base + quad + (v / 2) * 8;
          const uint32_t col = column_base + warp * 8 + lane4 * 2 + v % 2;
          if (row < live && col < N)
            partials[((uint64_t(base + pair) * rows + row) * Groups + g) * N + col] = raw[pair].v[v];
        }
      }
    }
    __syncthreads();
    if (base + 2 < end) {
      load((base - first) % 3, base + 3);
      load((base - first + 1) % 3, base + 4);
    }
  }
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  // Every producer thread publishes its own stores before the CTA ticket.
  // The last integer arrival acquires all stores, but never changes fold order.
  __threadfence();
  __syncthreads();
  const uint32_t tile = ((bank * Groups + g) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  if (threadIdx.x == 0) {
    uint32_t arrived;
    asm volatile("atom.add.acq_rel.gpu.u32 %0, [%1], 1;"
        : "=r"(arrived) : "l"(tickets + tile) : "memory");
    last = arrived == splits - 1;
  }
  __syncthreads();
  if (!last) return;
  for (uint32_t i = threadIdx.x; i < 16 * TileN; i += Threads) {
    const uint32_t row = token_base + i / TileN;
    const uint32_t col = column_base + i % TileN;
    if (row >= live || col >= N) continue;
    float total = 0.0f;
    if (!row_failed[i / TileN]) {
      const uint64_t at = (uint64_t(row) * Groups + g) * N + col;
      for (uint32_t base = 0; base < Folds; base += 16) {
        float chain[16];
#pragma unroll
        for (int j = 0; j < 16; ++j)
          if (base + j < Folds) chain[j] = __ldcg(partials + uint64_t(base + j) * rows * Groups * N + at);
#pragma unroll
        for (int j = 0; j < 16; ++j)
          if (base + j < Folds) total = __fadd_rn(total, chain[j]);
      }
    }
    const uint64_t at = (uint64_t(row) * Groups + g) * N + col;
    if constexpr (F32Output) reinterpret_cast<float*>(output)[at] = total;
    else reinterpret_cast<uint16_t*>(output)[at] = convert::bf16_from_f32(total);
  }
  __syncthreads();
  if (threadIdx.x == 0) atomicExch(tickets + tile, 0);
}

} // namespace chop::gemm::mxfp8_fold_prefetch
