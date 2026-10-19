#pragma once
#include "mxfp8_fold_stream.cuh"

namespace chop::gemm::mxfp8_fold_direct {
using Task = mxfp8_fold_stream::Task;
constexpr int Batch = 4;
struct Fragments {
  uint4 lo[Batch], hi[Batch];
  uint32_t scales[Batch];
};
struct Shared { bool last_leaf, last; };

template<bool Early>
__device__ __forceinline__ uint4 load16(const uint8_t* p) {
  if constexpr (Early) return mxfp8_fold_stream::fragment16(p);
  return __ldg(reinterpret_cast<const uint4*>(p));
}

template<int N, int K, int Groups, int NWarps, int KWarps, bool Paired, bool Early>
__device__ __forceinline__ Fragments load(Task task, uint32_t first, uint32_t end,
    const uint8_t* packed, const uint8_t* scales,
    const uint8_t* packed2, const uint8_t* scales2) {
  const uint32_t bank = task.z / Groups, g = task.z % Groups;
  const uint32_t tile = task.x * NWarps + (threadIdx.x / 32) % NWarps;
  const uint32_t lane = threadIdx.x % 32, row = tile * 8 + lane / 4;
  if constexpr (Paired) if (bank) { packed = packed2; scales = scales2; }
  Fragments result{};
#pragma unroll
  for (int p = 0; p < Batch; ++p) {
    result.scales[p] = 0x7f7f7f7fu;
    const uint32_t fold = first + p;
    if (fold < end && row < N) {
      const bool tail = fold * 128 + 128 > K;
      const uint64_t at = (uint64_t(g) * (N / 8) + tile) * (8 * K)
                          + fold * 1024 + lane * (tail ? 16 : 32);
      result.lo[p] = load16<Early>(packed + at);
      if (!tail) result.hi[p] = load16<Early>(packed + at + 16);
      const uint64_t s = (uint64_t(g) * N + row) * (K / 32) + fold * 4;
      if constexpr (K % 128 == 0) {
        if constexpr (Early)
          asm volatile("ld.volatile.global.u32 %0, [%1];"
              : "=r"(result.scales[p]) : "l"(scales + s) : "memory");
        else result.scales[p] = __ldg(reinterpret_cast<const uint32_t*>(scales + s));
      } else {
        // K is a multiple of 64, so each scale pair is aligned even when
        // a complete scale quad straddles a naturally aligned word.
        const uint32_t lo = __ldg(reinterpret_cast<const uint16_t*>(scales + s));
        const uint32_t hi = tail ? 0x7f7fu
            : __ldg(reinterpret_cast<const uint16_t*>(scales + s + 2));
        result.scales[p] = lo | (hi << 16);
      }
    }
  }
  return result;
}

template<int N, int K, int Groups, int RowStride, int GroupStride,
         int NWarps, int KWarps, bool F32Output, bool Paired,
         bool Stamp, bool DeferAdmission, bool OmitFirst>
__device__ void task(Task coord, Shared& shared, float* local, Fragments current,
    const uint8_t* activations, const uint8_t* activation_scales,
    const uint8_t* packed, const uint8_t* weight_scales,
    const uint8_t* packed2, const uint8_t* weight_scales2,
    void* output, void* output2, uint32_t* state,
    const uint32_t* live_rows, uint32_t* status, uint64_t status_row_offset,
    uint32_t rows, uint64_t* stamps, uint64_t start, uint64_t prefetched,
    uint64_t waited) {
  constexpr int Folds = (K + 127) / 128, TileN = NWarps * 8;
  constexpr int Threads = NWarps * KWarps * 32;
  constexpr int Part = (Folds + KWarps - 1) / KWarps;
  const uint32_t lane = threadIdx.x & 31, quad = lane / 4, lane4 = lane & 3;
  const uint32_t wn = (threadIdx.x / 32) % NWarps;
  const uint32_t wk = (threadIdx.x / 32) / NWarps;
  const uint32_t first = wk * Part, end = min(first + Part, uint32_t(Folds));
  const uint32_t token_base = coord.y * 16, live = min(rows, *live_rows);
  const uint32_t g = coord.z % Groups, bank = coord.z / Groups;
  if (token_base >= live) return;
  // One CTA admits every group. Other CTAs may store tentative results.
  // The final arrival acquires all producers, then zeros rejected rows.
  // No task waits for an unscheduled CTA and no per-step init is needed.
  if (coord.x == 0 && coord.z == 0) {
    if constexpr (DeferAdmission) {
      const uint64_t begin = clock64();
      while (clock64() - begin < 1000000) __nanosleep(64);
    }
    for (uint32_t row = token_base; row < min(token_base + 16, live); ++row) {
      bool bad = false;
      for (int group = 0; group < Groups; ++group)
        for (uint32_t k = threadIdx.x * 16; k < K; k += Threads * 16) {
          const uint64_t at = uint64_t(row) * RowStride + uint64_t(group) * GroupStride + k;
          const uint4 a = *reinterpret_cast<const uint4*>(activations + at);
          const uint8_t s = activation_scales[at / 32];
          bad |= !(mxfp8_fold_ticket::admitted4(a.x, s) && mxfp8_fold_ticket::admitted4(a.y, s)
                   && mxfp8_fold_ticket::admitted4(a.z, s) && mxfp8_fold_ticket::admitted4(a.w, s));
        }
      const bool invalid = __syncthreads_or(bad);
      if (threadIdx.x == 0) {
        const uint32_t previous = atomicOr(status + status_row_offset + row, invalid ? 1u : 0u);
        state[128 + row] = invalid || previous != 0;
      }
    }
  }
  uint64_t admitted = 0, computed = 0;
  if constexpr (Stamp) if (threadIdx.x == 0) admitted = clock64();
  ptx::D4 total{};
  for (uint32_t base = first; base < end; base += Batch) {
    const Fragments next = load<N,K,Groups,NWarps,KWarps,Paired,false>(
        coord, base + Batch, end, packed, weight_scales, packed2, weight_scales2);
    ptx::D4 raw[Batch]{};
#pragma unroll
    for (int step = 0; step < 4; ++step) {
#pragma unroll
      for (int p = 0; p < Batch; ++p) if (base + p < end) {
        ptx::A4 af{};
        ptx::B2 bf{};
#pragma unroll
        for (int v = 0; v < 4; ++v) {
          const uint32_t row = token_base + quad + (v % 2) * 8;
          const uint32_t k = (base + p) * 128 + step * 32 + (v / 2) * 16 + lane4 * 4;
          if (row < live && k < K)
            af.v[v] = *reinterpret_cast<const uint32_t*>(
                activations + uint64_t(row) * RowStride + uint64_t(g) * GroupStride + k);
        }
        const uint4 w = step < 2 ? current.lo[p] : current.hi[p];
        bf.v[0] = (step & 1) ? w.z : w.x;
        bf.v[1] = (step & 1) ? w.w : w.y;
        const uint32_t ar = token_base + quad + (lane4 & 1) * 8;
        const uint32_t ak = (base + p) * 128 + step * 32;
        const uint32_t sa = ar < live && ak < K
            ? activation_scales[(uint64_t(ar) * RowStride + uint64_t(g) * GroupStride + ak) / 32] : 127;
        const uint32_t sb = (current.scales[p] >> (step * 8)) & 255;
        ptx::mma_mxf8f6f4<ptx::Element::e4m3,ptx::Element::e4m3>(raw[p], af, bf, sa, sb);
      }
    }
#pragma unroll
    for (int p = 0; p < Batch; ++p) if (base + p < end) {
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        const float value = OmitFirst && base + p == 0 ? 0.0f : raw[p].v[v];
        if constexpr (KWarps == 1) total.v[v] = __fadd_rn(total.v[v], value);
        else {
          const uint32_t row = quad + (v / 2) * 8;
          const uint32_t col = wn * 8 + lane4 * 2 + v % 2;
          local[((base + p) * 16 + row) * TileN + col] = value;
        }
      }
    }
    current = next;
  }
  void* target = bank ? output2 : output;
  if constexpr (KWarps == 1) {
#pragma unroll
    for (int v = 0; v < 4; ++v) {
      const uint32_t row = token_base + quad + (v / 2) * 8;
      const uint32_t col = coord.x * TileN + wn * 8 + lane4 * 2 + v % 2;
      if (row < live && col < N) {
        const uint64_t at = (uint64_t(row) * Groups + g) * N + col;
        if constexpr (F32Output) reinterpret_cast<float*>(target)[at] = total.v[v];
        else reinterpret_cast<uint16_t*>(target)[at] = convert::bf16_from_f32(total.v[v]);
      }
    }
  } else {
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < 16 * TileN; i += Threads) {
      const uint32_t row = token_base + i / TileN, col = coord.x * TileN + i % TileN;
      if (row < live && col < N) {
        float sum = 0.0f;
        for (int f = 0; f < Folds; ++f) sum = __fadd_rn(sum, local[f * 16 * TileN + i]);
        const uint64_t at = (uint64_t(row) * Groups + g) * N + col;
        if constexpr (F32Output) reinterpret_cast<float*>(target)[at] = sum;
        else reinterpret_cast<uint16_t*>(target)[at] = convert::bf16_from_f32(sum);
      }
    }
  }
  if constexpr (Stamp) if (threadIdx.x == 0) computed = clock64();
  __threadfence();
  __syncthreads();
  const uint32_t tiles = coord.nx * Groups * (Paired ? 2 : 1);
  const uint32_t leaves = (tiles + 31) / 32;
  const uint32_t tile = coord.z * coord.nx + coord.x, leaf = tile / 32;
  auto* leaf_count = state + 256 + (coord.y * leaves + leaf) * 32;
  auto* root_count = state + coord.y * 32;
  // Spread completion traffic over cache-line-separated groups of 32
  // CTAs. Only each group's final arrival touches the row-tile root.
  if (threadIdx.x == 0) {
    shared.last = false;
    shared.last_leaf = mxfp8_fold_stream::arrive(leaf_count)
        == min(32u, tiles - leaf * 32) - 1;
  }
  __syncthreads();
  if (shared.last_leaf) {
    if (threadIdx.x == 0) {
      atomicExch(leaf_count, 0);
      __threadfence();
      shared.last = mxfp8_fold_stream::arrive(root_count) == leaves - 1;
    }
    __syncthreads();
  }
  if (shared.last) {
    for (uint32_t row = token_base; row < min(token_base + 16, live); ++row)
      if (mxfp8_fold_stream::acquire(state + 128 + row)) {
        for (uint32_t i = threadIdx.x; i < Groups * N; i += Threads) {
          const uint64_t at = uint64_t(row) * Groups * N + i;
          if constexpr (F32Output) {
            reinterpret_cast<float*>(output)[at] = 0.0f;
            if constexpr (Paired) reinterpret_cast<float*>(output2)[at] = 0.0f;
          } else {
            reinterpret_cast<uint16_t*>(output)[at] = 0;
            if constexpr (Paired) reinterpret_cast<uint16_t*>(output2)[at] = 0;
          }
        }
      }
    __syncthreads();
    if (threadIdx.x < 16 && token_base + threadIdx.x < live)
      state[128 + token_base + threadIdx.x] = 0;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) atomicExch(root_count, 0);
  }
  if constexpr (Stamp) if (threadIdx.x == 0) {
    const uint64_t index = (uint64_t(coord.z) * coord.ny + coord.y) * coord.nx + coord.x;
    const uint64_t values[8] = {prefetched - start, waited - prefetched,
        admitted - waited, computed - admitted, clock64() - computed,
        uint64_t(shared.last), Folds, 1};
    for (int i = 0; i < 8; ++i) stamps[index * 8 + i] = values[i];
  }
}

template<int N, int K, int Groups, int RowStride, int GroupStride,
         int NWarps, int KWarps, bool F32Output, bool Paired, bool Stamp,
         bool DeferAdmission = false, bool OmitFirst = false>
__device__ void entry(const uint8_t* activations, const uint8_t* activation_scales,
    const uint8_t* packed, const uint8_t* weight_scales,
    const uint8_t* packed2, const uint8_t* weight_scales2,
    void* output, void* output2, uint8_t* workspace, uint32_t* state,
    const uint32_t* live_rows, uint32_t* status, uint64_t status_row_offset,
    uint32_t rows, uint32_t) {
  static_assert(NWarps == 1 || NWarps == 4);
  static_assert(KWarps == 1 || (KWarps == 4 && NWarps == 1));
  constexpr int Folds = (K + 127) / 128, Part = (Folds + KWarps - 1) / KWarps;
  const uint32_t first = (threadIdx.x / (NWarps * 32)) * Part;
  const Task coord{blockIdx.x,blockIdx.y,blockIdx.z,gridDim.x,gridDim.y};
  __shared__ Shared shared;
  extern __shared__ __align__(16) uint8_t arena[];
  uint64_t start = 0, prefetched = 0, waited = 0;
  ptx::griddep_launch_dependents();
  if constexpr (Stamp) if (threadIdx.x == 0) start = clock64();
  const Fragments first_weights = load<N,K,Groups,NWarps,KWarps,Paired,true>(
      coord, first, min(first + Part,uint32_t(Folds)), packed, weight_scales, packed2, weight_scales2);
  if constexpr (Stamp) if (threadIdx.x == 0) prefetched = clock64();
  ptx::griddep_wait();
  if constexpr (Stamp) if (threadIdx.x == 0) waited = clock64();
  task<N,K,Groups,RowStride,GroupStride,NWarps,KWarps,F32Output,Paired,Stamp,DeferAdmission,OmitFirst>(
      coord,shared,reinterpret_cast<float*>(arena),first_weights,activations,activation_scales,
      packed,weight_scales,packed2,weight_scales2,output,output2,state,live_rows,status,
      status_row_offset,rows,reinterpret_cast<uint64_t*>(workspace),start,prefetched,waited);
}
} // namespace chop::gemm::mxfp8_fold_direct
