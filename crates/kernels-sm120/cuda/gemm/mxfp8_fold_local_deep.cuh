#pragma once
#include "mxfp8_fold_local_rows.cuh"

namespace chop::gemm::mxfp8_fold_local_rows {

// The arithmetic of run, with a deeper ring: Stages folds of both operands
// are in flight and every fold refills the slot its predecessor left, so
// the CTA waits on a fold issued Stages - 1 folds earlier rather than on the
// fold it issued last. Each 128-K fold is still four MMAs from zero, added
// to the output in increasing K order. One commit group per fold; the tail
// commits empty groups so every wait after the first keeps one immediate.
// Swizzle stores 16-byte chunk c of shared row r at chunk c ^ (r % 8): the
// eight rows a fragment load touches then fall in distinct banks. It moves
// bytes only, never which bytes an MMA reads.
template<int N, int K, int Groups, int RowStride, int GroupStride,
         int TileN, int Stages, bool Swizzle, bool OmitFirst>
__device__ void run_deep(const uint8_t* activations, const uint8_t* activation_scales,
    const uint8_t* weights, const uint8_t* weight_scales, float* output,
    const uint8_t* workspace, const uint32_t* live_rows, uint32_t rows) {
  static_assert(K % 128 == 0 && (TileN == 8 || TileN == 16 || TileN == 32));
  static_assert(Stages >= 3 && Stages <= K / 128);
  constexpr int Folds = K / 128, Threads = TileN * 4;
  __shared__ __align__(16) uint8_t codes[Stages][(32 + TileN) * 128];
  __shared__ __align__(4) uint8_t scales[Stages][(32 + TileN) * 4];
  __shared__ uint32_t failed[32];
  const uint32_t g = blockIdx.z, column_base = blockIdx.x * TileN;
  // Byte offset of 16-byte chunk c of shared row r.
  const auto at_chunk = [](uint32_t r, uint32_t c) {
    return r * 128 + (Swizzle ? c ^ (r % 8) : c) * 16;
  };
  ptx::griddep_launch_dependents();
  const auto load_weights = [&](uint32_t slot, uint32_t fold) {
    for (uint32_t chunk = threadIdx.x; chunk < TileN * 8; chunk += Threads) {
      const uint32_t row = column_base + chunk / 8;
      const bool valid = fold < Folds && row < N;
      const uint64_t at = (uint64_t(g) * N + row) * K + fold * 128 + (chunk % 8) * 16;
      mxfp8_fold_ticket::copy16(codes[slot] + at_chunk(32 + chunk / 8, chunk % 8),
                              weights + (valid ? at : 0), valid);
    }
    if (threadIdx.x < TileN) {
      const uint32_t row = column_base + threadIdx.x;
      auto* dst = scales[slot] + (32 + threadIdx.x) * 4;
      if (fold < Folds && row < N) {
        const uint64_t at = (uint64_t(g) * N + row) * (K / 32) + fold * 4;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"
            :: "r"(uint32_t(__cvta_generic_to_shared(dst))), "l"(weight_scales + at) : "memory");
      } else *reinterpret_cast<uint32_t*>(dst) = 0x7f7f7f7fu;
    }
  };
  // Only immutable weights enter private shared memory before the wait.
#pragma unroll
  for (int slot = 0; slot < Stages; ++slot) {
    load_weights(slot, slot);
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
  ptx::griddep_wait();
  const uint32_t live = min(rows, *live_rows);
  if (threadIdx.x < 32)
    failed[threadIdx.x] = threadIdx.x < live
        ? reinterpret_cast<const uint32_t*>(workspace)[threadIdx.x] : 1;
  __syncthreads();
  const auto load_activations = [&](uint32_t slot, uint32_t fold) {
    for (uint32_t chunk = threadIdx.x; chunk < 32 * 8; chunk += Threads) {
      const uint32_t row = chunk / 8;
      const bool valid = fold < Folds && row < live && !failed[row];
      const uint64_t at = uint64_t(row) * RowStride + uint64_t(g) * GroupStride
          + fold * 128 + (chunk % 8) * 16;
      mxfp8_fold_ticket::copy16(codes[slot] + at_chunk(chunk / 8, chunk % 8),
                              activations + (valid ? at : 0), valid);
    }
    for (uint32_t row = threadIdx.x; row < 32; row += Threads) {
      auto* dst = scales[slot] + row * 4;
      if (fold < Folds && row < live && !failed[row]) {
        const uint64_t at = (uint64_t(row) * RowStride + uint64_t(g) * GroupStride) / 32 + fold * 4;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"
            :: "r"(uint32_t(__cvta_generic_to_shared(dst))), "l"(activation_scales + at) : "memory");
      } else *reinterpret_cast<uint32_t*>(dst) = 0x7f7f7f7fu;
    }
  };
#pragma unroll
  for (int slot = 0; slot < Stages; ++slot) {
    load_activations(slot, slot);
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
  const uint32_t warp = threadIdx.x / 32, quad = (threadIdx.x & 31) / 4;
  const uint32_t lane4 = threadIdx.x & 3;
  ptx::D4 total[2]{};
  for (uint32_t fold = 0; fold < Folds; ++fold) {
    // Groups newer than fold's: the later prologue activation groups and
    // the refills of earlier iterations, Stages - 1 for fold 0 and
    // Stages - 2 for every later fold.
    if (fold == 0) asm volatile("cp.async.wait_group %0;" :: "n"(Stages - 1) : "memory");
    else asm volatile("cp.async.wait_group %0;" :: "n"(Stages - 2) : "memory");
    __syncthreads();
    // Every warp has finished fold - 1, so its slot takes fold - 1 + Stages.
    if (fold > 0) {
      const uint32_t slot = (fold - 1) % Stages, next = fold - 1 + Stages;
      if (next < Folds) {
        load_weights(slot, next);
        load_activations(slot, next);
      }
      asm volatile("cp.async.commit_group;" ::: "memory");
    }
    const uint32_t slot = fold % Stages;
    ptx::D4 raw[2]{};
#pragma unroll
    for (int step = 0; step < 4; ++step) {
      ptx::B2 bf{};
#pragma unroll
      for (int v = 0; v < 2; ++v)
        bf.v[v] = *reinterpret_cast<const uint32_t*>(codes[slot]
            + at_chunk(32 + warp * 8 + quad, step * 2 + v) + lane4 * 4);
      const uint32_t sb = scales[slot][(32 + warp * 8 + quad) * 4 + step];
#pragma unroll
      for (int group = 0; group < 2; ++group) {
        if (group * 16 >= live) continue;
        ptx::A4 af{};
#pragma unroll
        for (int v = 0; v < 4; ++v)
          af.v[v] = *reinterpret_cast<const uint32_t*>(codes[slot]
              + at_chunk(group * 16 + quad + (v % 2) * 8, step * 2 + v / 2) + lane4 * 4);
        const uint32_t sa = scales[slot][(group * 16 + quad + (lane4 & 1) * 8) * 4 + step];
        ptx::mma_mxf8f6f4<ptx::Element::e4m3,ptx::Element::e4m3>(raw[group],af,bf,sa,sb);
      }
    }
#pragma unroll
    for (int group = 0; group < 2; ++group) {
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        const float value = OmitFirst && fold == 0 ? 0.0f : raw[group].v[v];
        total[group].v[v] = __fadd_rn(total[group].v[v], value);
      }
    }
  }
  asm volatile("cp.async.wait_group 0;" ::: "memory");
#pragma unroll
  for (int group = 0; group < 2; ++group) {
#pragma unroll
    for (int v = 0; v < 4; ++v) {
      const uint32_t row = group * 16 + quad + (v / 2) * 8;
      const uint32_t col = column_base + warp * 8 + lane4 * 2 + v % 2;
      if (row < live && col < N)
        output[(uint64_t(row) * Groups + g) * N + col] = failed[row] ? 0.0f : total[group].v[v];
    }
  }
}
} // namespace chop::gemm::mxfp8_fold_local_rows
