#pragma once

// Shared pieces of the pooled top-k indexer (README.md): BF16 helpers, warp
// sums in a fixed order, 16-byte cp.async, and the streaming top-k buffer
// that keeps a CTA's best candidates without a score buffer.

#include <cuda_bf16.h>
#include <stdint.h>
#include "common/ptx.cuh"

namespace chop::pooled_indexer {

__device__ __forceinline__ float bf16(uint16_t x) { return __uint_as_float(uint32_t(x) << 16); }
__device__ __forceinline__ uint16_t to_bf16(float x) {
  const __nv_bfloat16 value = __float2bfloat16_rn(x);
  return *reinterpret_cast<const uint16_t*>(&value);
}

// Four consecutive BF16 values, one 8-byte load or store.
__device__ __forceinline__ void load4(float (&x)[4], const uint16_t* source) {
  const uint2 packed = *reinterpret_cast<const uint2*>(source);
  x[0] = __uint_as_float(packed.x << 16);
  x[1] = __uint_as_float(packed.x & 0xffff0000u);
  x[2] = __uint_as_float(packed.y << 16);
  x[3] = __uint_as_float(packed.y & 0xffff0000u);
}
__device__ __forceinline__ void store4(uint16_t* target, const float (&x)[4]) {
  *reinterpret_cast<uint2*>(target) =
      make_uint2(uint32_t(to_bf16(x[0])) | uint32_t(to_bf16(x[1])) << 16,
                 uint32_t(to_bf16(x[2])) | uint32_t(to_bf16(x[3])) << 16);
}

// Sum over a warp by an xor tree, the same order on every lane.
__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  return value;
}

__device__ __forceinline__ void cp_async16(void* target, const void* source, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
               :: "r"(ptx::shared_address(target)), "l"(source), "r"(valid ? 16 : 0)
               : "memory");
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}
template<int Pending>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" :: "n"(Pending) : "memory");
}

// A candidate as one 64-bit key: the score's bits mapped so unsigned order
// is IEEE order (-0 counts as +0, every NaN ranks below -inf), then the
// inverted pool, so a larger key is a better candidate and equal scores rank
// the lower pool first. Key 0 is no candidate.
__device__ __forceinline__ uint64_t candidate_key(float score, uint32_t pool) {
  const uint32_t bits = __float_as_uint(score == 0.f ? 0.f : score);
  const uint32_t ordered = score != score       ? 0u
                           : bits & 0x80000000u ? ~bits
                                                : bits | 0x80000000u;
  return uint64_t(ordered) << 32 | uint32_t(~pool);
}
__device__ __forceinline__ uint32_t candidate_pool(uint64_t key) { return ~uint32_t(key); }
// The canonical score of a key: +0 for either zero, 0xffffffff for a NaN.
__device__ __forceinline__ float candidate_score(uint64_t key) {
  const uint32_t ordered = uint32_t(key >> 32);
  return __uint_as_float(ordered & 0x80000000u ? ordered & 0x7fffffffu : ~ordered);
}

// Bitonic sort of `N` keys in shared memory, descending, by all `Threads`
// threads of the CTA. Callers synchronize before; the sort synchronizes
// after its last stage.
template<int N, int Threads>
__device__ __forceinline__ void sort_descending(uint64_t* keys) {
  static_assert((N & (N - 1)) == 0, "a bitonic sort takes a power of two");
  for (int size = 2; size <= N; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      for (int i = threadIdx.x; i < N / 2; i += Threads) {
        const int low = 2 * i - (i & (stride - 1));
        const int high = low + stride;
        const uint64_t a = keys[low], b = keys[high];
        if ((a < b) == ((low & size) == 0)) {
          keys[low] = b;
          keys[high] = a;
        }
      }
      __syncthreads();
    }
  }
}

// The best `kept` candidates offered so far, `kept <= MaxKept`. Offers
// above the current threshold append to a buffer of `2 * MaxKept` keys;
// `reduce` sorts it and keeps the best `kept`. Candidates are distinct
// pools, so the kept set does not depend on the order of offers.
template<int MaxKept, int Threads>
struct TopK {
  static_assert((MaxKept & (MaxKept - 1)) == 0, "the kept bound is a power of two");
  static constexpr int capacity = 2 * MaxKept;
  static constexpr int smem = capacity * 8 + 16;

  uint64_t* keys;
  uint32_t* count;
  uint64_t* threshold;

  __device__ __forceinline__ explicit TopK(void* storage)
      : keys(static_cast<uint64_t*>(storage)),
        count(reinterpret_cast<uint32_t*>(keys + capacity)),
        threshold(keys + capacity + 1) {}

  __device__ __forceinline__ void init() {
    if (threadIdx.x == 0) {
      *count = 0;
      *threshold = 0;
    }
    __syncthreads();
  }

  __device__ __forceinline__ void offer(uint64_t key) {
    if (key > *threshold) keys[atomicAdd(count, 1u)] = key;
  }

  // Makes room for `incoming` more offers, at most `MaxKept`. Every thread
  // reads the count before any thread offers again.
  __device__ __forceinline__ void make_room(uint32_t kept, uint32_t incoming) {
    __syncthreads();
    const uint32_t live = *count;
    __syncthreads();
    if (live + incoming > capacity) reduce(kept);
  }

  // Sorts the buffer descending and keeps its best `kept` keys.
  __device__ __forceinline__ void reduce(uint32_t kept) {
    __syncthreads();
    const uint32_t live = *count;
    for (int i = live + threadIdx.x; i < capacity; i += Threads) keys[i] = 0;
    __syncthreads();
    sort_descending<capacity, Threads>(keys);
    if (threadIdx.x == 0) {
      const uint32_t left = live < kept ? live : kept;
      *count = left;
      *threshold = left == kept ? keys[kept - 1] : 0;
    }
    __syncthreads();
  }
};

}  // namespace chop::pooled_indexer
