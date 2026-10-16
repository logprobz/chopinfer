#pragma once

// Context-parallel entries (kernel-core `context_parallel`): the cross-rank
// partial attention merge and the key list localization. See README.md.

#include <math.h>
#include <stdint.h>
#include "common/convert.cuh"
#include "common/ptx.cuh"

namespace chop::context_parallel {

constexpr float kLog2e = 1.4426950408889634f;

// exp(x) as the latent attention engine computes it: exp2 of x * log2(e).
__device__ __forceinline__ float exp_of(float x) { return exp2f(x * kLog2e); }

template<int Ranks, int Heads, int Width>
struct Merge {
  static_assert(Ranks >= 2 && Ranks <= 8, "a group of 2 to 8 ranks");
  static_assert(Heads > 0 && Width > 0);
  static constexpr int ranks = Ranks, heads = Heads, width = Width;
  static constexpr int threads = 128;
};

// Grid [rows, Heads], 128 threads. `partials` holds Ranks gathered segments,
// rank-major, of `rank_rows * input_heads * (Width + 2)` floats each: F32
// maximum `m` [rank_rows, input_heads], denominator `l` (same shape), then
// numerator `o` [rank_rows, input_heads, Width]. The CTA merges input head
// `head_offset + blockIdx.y` of row `blockIdx.x` over the ranks in rank
// order (`M = max m_r`, `a_r = exp(m_r - M)` or 0 for an empty rank, `L` and
// `O` FMA sums from zero) and writes BF16 `O / L` (zeros when every rank is
// empty) to `out` [rows, Heads, Width].
template<class T>
__device__ void partial_merge(const float* partials, const uint32_t* live_rows, uint16_t* out,
                              uint32_t rank_rows, uint32_t input_heads, uint32_t head_offset) {
  ptx::griddep_launch_dependents();
  __shared__ float weights[T::ranks];
  __shared__ float total;
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  const uint32_t head = blockIdx.y;
  if (row >= *live_rows) return;
  const uint64_t pairs = uint64_t(rank_rows) * input_heads;
  const uint64_t segment = pairs * (T::width + 2);
  const uint64_t pair = uint64_t(row) * input_heads + head_offset + head;
  if (threadIdx.x == 0) {
    float top = -INFINITY;
#pragma unroll
    for (int rank = 0; rank < T::ranks; ++rank) top = fmaxf(top, partials[rank * segment + pair]);
    float sum = 0.0f;
#pragma unroll
    for (int rank = 0; rank < T::ranks; ++rank) {
      const float m = partials[rank * segment + pair];
      const float weight = m == -INFINITY ? 0.0f : exp_of(m - top);
      weights[rank] = weight;
      sum = fmaf(weight, partials[rank * segment + pairs + pair], sum);
    }
    total = sum;
  }
  __syncthreads();
  const float denominator = total;
  const uint64_t numerators = 2 * pairs + pair * T::width;
  uint16_t* target = out + (uint64_t(row) * T::heads + head) * T::width;
  for (int d = threadIdx.x; d < T::width; d += T::threads) {
    float numerator = 0.0f;
#pragma unroll
    for (int rank = 0; rank < T::ranks; ++rank)
      numerator = fmaf(weights[rank], partials[rank * segment + numerators + d], numerator);
    target[d] = convert::bf16_from_f32(denominator == 0.0f ? 0.0f
                                                           : __fdiv_rn(numerator, denominator));
  }
}

template<int Ranks, int Page, int Width>
struct Localize {
  static_assert(Ranks >= 2 && Ranks <= 8, "a group of 2 to 8 ranks");
  static_assert(Page > 0 && Width > 0);
  static constexpr int ranks = Ranks, page = Page, width = Width;
  static constexpr int threads = 256;
};

// Grid [rows, ceil(Width / 256)], 256 threads. Writes group rank `cp_rank`'s
// local key list row: a position p of `lists` [rows, Width] lives on group
// rank `(p / Page) % Ranks`; the owner's entry is `p / (Ranks * Page) * Page
// + p % Page`, every other entry and every hole -1.
template<class T>
__device__ void localize_keys(const int32_t* lists, const uint32_t* live_rows, int32_t* out,
                              uint32_t cp_rank) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  const int column = blockIdx.y * T::threads + threadIdx.x;
  if (row >= *live_rows || column >= T::width) return;
  const uint64_t index = uint64_t(row) * T::width + column;
  const int32_t position = lists[index];
  int32_t local = -1;
  if (position >= 0) {
    const int32_t page = position / T::page;
    if (uint32_t(page % T::ranks) == cp_rank)
      local = page / T::ranks * T::page + position % T::page;
  }
  out[index] = local;
}

} // namespace chop::context_parallel
