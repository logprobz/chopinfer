#pragma once

// Paged attention partials (kernel-core
// `ContextParallelQuery::PagedAttentionPartials`): causal paged
// grouped-query attention over the keys one context-parallel group rank
// owns, written as the F32 maximum, denominator and unnormalized numerator
// the partial merge folds. See README.md.

#include <math.h>
#include <stdint.h>
#include "common/convert.cuh"
#include "common/ptx.cuh"

namespace chop::paged_partials {

constexpr float kLog2e = 1.4426950408889634f;

// exp(x) as the context-parallel merge computes it: exp2 of x * log2(e).
__device__ __forceinline__ float exp_of(float x) { return exp2f(x * kLog2e); }

template<int QHeads, int KvHeads, int Dim, int Page, bool Fp8>
struct Partials {
  static_assert(QHeads >= 1 && QHeads <= 8, "one warp per query head of a KV head");
  static_assert(KvHeads >= 1);
  static_assert(Dim % 32 == 0 && Dim <= 1024, "whole lanes of the head dimension");
  static_assert(Page >= 1 && Page <= 64, "a page is one key tile of two keys per lane");
  static constexpr int q_heads = QHeads, kv_heads = KvHeads, dim = Dim, page = Page;
  static constexpr int heads = QHeads * KvHeads;
  static constexpr int threads = 256, warps = threads / 32;
  // Head dimensions per lane of a key's dot product, per thread of the
  // value sum.
  static constexpr int lane_dims = Dim / 32;
  static constexpr int thread_dims = (Dim + threads - 1) / threads;
  static constexpr int element_bytes = Fp8 ? 1 : 2;
  static constexpr bool fp8 = Fp8;
};

template<class T>
__device__ __forceinline__ float element(const uint8_t* row, int d) {
  if constexpr (T::fp8) {
    return convert::unpack_f32_e4m3(uint16_t(row[d])).lo;
  } else {
    return convert::bf16_to_f32(reinterpret_cast<const uint16_t*>(row)[d]);
  }
}

// Grid [rows, KvHeads], 256 threads. The CTA of (row, KV head) reads row
// `row`'s item `row_batch[row]` and position p = `positions[row]`, and walks
// the physical pages j = cp_rank, cp_rank + cp_size, ... while j * Page <= p
// (the rank's own pages; cp_size 1 walks every page) in order, each at the
// rank's local page `block_tables[item * max_pages + j / cp_size]`, keeping
// its positions up to p. Per page:
//
// - warp w forms the scores of keys w, w + 8, ...: each lane FMA-sums its
//   `Dim / 32` dimensions from zero in order, the lanes fold by xor shuffles
//   (16, 8, 4, 2, 1), and `s = dot * k_scale * scale`;
// - warp h (one per query head) takes the page's maximum, the new running
//   maximum `m'`, `alpha = exp(m - m')` (0 while m is -inf) and each key's
//   `p = exp(s - m')`; lane 0 updates `l = l * alpha + p_0 + p_1 + ...` in key
//   order;
// - each thread scales its dimensions' numerators by alpha and FMA-adds
//   `p * v` over the page's keys in order.
//
// Writes `m`, `l` [rows, heads] and `o * v_scale` [rows, heads, Dim]; a row
// without a kept key writes -inf, +0 and +0. Rows past `*live_rows` return
// after the PDL wait.
template<class T>
__device__ void partials(const uint16_t* q, const uint8_t* k_cache, const uint8_t* v_cache,
                         const int32_t* block_tables, const int32_t* row_batch,
                         const int32_t* positions, const uint32_t* live_rows, float* out_max,
                         float* out_den, float* out_num, uint32_t max_pages,
                         uint64_t k_page_stride_bytes, uint64_t v_page_stride_bytes,
                         uint32_t scale_bits, uint32_t k_scale_bits, uint32_t v_scale_bits,
                         uint32_t cp_rank, uint32_t cp_size) {
  ptx::griddep_launch_dependents();
  __shared__ float queries[T::q_heads][T::dim];
  __shared__ float scores[T::q_heads][T::page];
  __shared__ float alphas[T::q_heads];
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  const uint32_t kv_head = blockIdx.y;
  if (row >= *live_rows) return;
  if (cp_size == 0 || cp_rank >= cp_size) __trap();
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const float scale = __uint_as_float(scale_bits);
  const float score_scale = __fmul_rn(__uint_as_float(k_scale_bits), scale);
  const uint64_t head0 = uint64_t(row) * T::heads + uint64_t(kv_head) * T::q_heads;
  for (int i = threadIdx.x; i < T::q_heads * T::dim; i += T::threads)
    queries[i / T::dim][i % T::dim] = convert::bf16_to_f32(q[head0 * T::dim + i]);
  const int32_t item = row_batch[row];
  const int64_t position = positions[row];
  // Lane 0 of warp h keeps head h's running maximum and denominator.
  float maximum = -INFINITY, denominator = 0.0f;
  float numerators[T::q_heads][T::thread_dims];
#pragma unroll
  for (int h = 0; h < T::q_heads; ++h)
#pragma unroll
    for (int i = 0; i < T::thread_dims; ++i) numerators[h][i] = 0.0f;
  __syncthreads();
  const uint64_t row_bytes = uint64_t(T::kv_heads) * T::dim * T::element_bytes;
  const uint64_t head_bytes = uint64_t(kv_head) * T::dim * T::element_bytes;
  for (int64_t j = cp_rank; position >= 0 && j * T::page <= position; j += cp_size) {
    const uint64_t local = uint64_t(j) / cp_size;
    if (local >= max_pages) __trap();
    const int32_t physical = block_tables[uint64_t(item) * max_pages + local];
    if (physical < 0) __trap();
    const int keys = int(min(int64_t(T::page), position + 1 - j * T::page));
    const uint8_t* k_page = k_cache + uint64_t(physical) * k_page_stride_bytes + head_bytes;
    const uint8_t* v_page = v_cache + uint64_t(physical) * v_page_stride_bytes + head_bytes;
    // Scores.
    for (int key = warp; key < keys; key += T::warps) {
      const uint8_t* k_row = k_page + uint64_t(key) * row_bytes;
      float dots[T::q_heads];
#pragma unroll
      for (int h = 0; h < T::q_heads; ++h) dots[h] = 0.0f;
#pragma unroll
      for (int e = 0; e < T::lane_dims; ++e) {
        const int d = lane * T::lane_dims + e;
        const float k = element<T>(k_row, d);
#pragma unroll
        for (int h = 0; h < T::q_heads; ++h) dots[h] = fmaf(queries[h][d], k, dots[h]);
      }
#pragma unroll
      for (int h = 0; h < T::q_heads; ++h) {
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
          dots[h] += __shfl_xor_sync(0xffffffffu, dots[h], offset);
        if (lane == 0) scores[h][key] = __fmul_rn(dots[h], score_scale);
      }
    }
    __syncthreads();
    // The running softmax: warp h for query head h.
    if (warp < T::q_heads) {
      const int h = warp;
      float top = -INFINITY;
      for (int key = lane; key < keys; key += 32) top = fmaxf(top, scores[h][key]);
#pragma unroll
      for (int offset = 16; offset > 0; offset /= 2)
        top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, offset));
      const float next = fmaxf(maximum, top);
      const float alpha = maximum == -INFINITY ? 0.0f : exp_of(maximum - next);
      for (int key = lane; key < keys; key += 32) scores[h][key] = exp_of(scores[h][key] - next);
      __syncwarp();
      if (lane == 0) {
        float sum = __fmul_rn(denominator, alpha);
        for (int key = 0; key < keys; ++key) sum = __fadd_rn(sum, scores[h][key]);
        denominator = sum;
        alphas[h] = alpha;
      }
      maximum = next;
    }
    __syncthreads();
    // The numerators.
#pragma unroll
    for (int i = 0; i < T::thread_dims; ++i) {
      const int d = threadIdx.x + i * T::threads;
      if (d >= T::dim) break;
#pragma unroll
      for (int h = 0; h < T::q_heads; ++h) numerators[h][i] = __fmul_rn(numerators[h][i], alphas[h]);
      for (int key = 0; key < keys; ++key) {
        const float v = element<T>(v_page + uint64_t(key) * row_bytes, d);
#pragma unroll
        for (int h = 0; h < T::q_heads; ++h) numerators[h][i] = fmaf(scores[h][key], v, numerators[h][i]);
      }
    }
    __syncthreads();
  }
  if (warp < T::q_heads && lane == 0) {
    out_max[head0 + warp] = maximum;
    out_den[head0 + warp] = denominator;
  }
  const float v_scale = __uint_as_float(v_scale_bits);
#pragma unroll
  for (int i = 0; i < T::thread_dims; ++i) {
    const int d = threadIdx.x + i * T::threads;
    if (d >= T::dim) break;
#pragma unroll
    for (int h = 0; h < T::q_heads; ++h)
      out_num[(head0 + h) * T::dim + d] = __fmul_rn(numerators[h][i], v_scale);
  }
}

}  // namespace chop::paged_partials
