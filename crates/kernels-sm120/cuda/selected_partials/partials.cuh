#pragma once

// Selected attention partials (kernel-core
// `ContextParallelQuery::SelectedAttentionPartials`): block-sparse
// grouped-query attention over the keys of a row's selected blocks and its
// tail that one context-parallel group rank owns, written as the F32
// maximum, denominator and unnormalized numerator the partial merge folds.
// The arithmetic per key tile is the paged partials' (paged_partials/
// partials.cuh); only the key set differs. See README.md.

#include "paged_partials/partials.cuh"

namespace chop::selected_partials {

using paged_partials::exp_of;

template<int QHeads, int KvHeads, int Dim, int Page, int Block, int TopBlocks, bool Fp8>
struct SelectedPartials {
  static_assert(QHeads >= 1 && QHeads <= 8, "one warp per query head of a KV head");
  static_assert(KvHeads >= 1);
  static_assert(Dim % 32 == 0 && Dim <= 1024, "whole lanes of the head dimension");
  static_assert(Block >= 1 && Page % Block == 0, "a block never straddles a page");
  static_assert(TopBlocks >= 1 && TopBlocks <= 2048, "the selected list fits shared memory");
  static constexpr int q_heads = QHeads, kv_heads = KvHeads, dim = Dim, page = Page;
  static constexpr int block = Block, top = TopBlocks;
  static constexpr int heads = QHeads * KvHeads;
  static constexpr int threads = 256, warps = threads / 32;
  // Keys per tile of the running softmax.
  static constexpr int tile = 64;
  static constexpr int lane_dims = Dim / 32;
  static constexpr int thread_dims = (Dim + threads - 1) / threads;
  static constexpr int element_bytes = Fp8 ? 1 : 2;
  static constexpr bool fp8 = Fp8;
};

// Grid [rows, KvHeads], 256 threads. The CTA of (row, KV head) reads row
// `row`'s item `row_batch[row]`, position p = `positions[row]` and its c =
// (p + 1) / Block complete blocks. Its key set is the blocks
// `selected[row * TopBlocks + i]`, i < min(c, TopBlocks) (ascending), each
// as its Block positions, then the tail c * Block..=p. Warp 0 compacts the
// blocks whose page `b * Block / Page` the rank owns (page j on rank j %
// cp_size) in list order; the tail is kept when its page is owned. The kept
// keys are walked in tiles of 64, each key at the rank's local page
// `block_tables[item * max_pages + j / cp_size]`, slot t % Page:
//
// - warp w forms the scores of keys w, w + 8, ...: each lane FMA-sums its
//   `Dim / 32` dimensions from zero in order, the lanes fold by xor shuffles
//   (16, 8, 4, 2, 1), and `s = dot * (k_scale * scale)`;
// - warp h (one per query head) takes the tile's maximum, the new running
//   maximum `m'`, `alpha = exp(m - m')` (0 while m is -inf) and each key's
//   `p = exp(s - m')`; lane 0 updates `l = l * alpha + p_0 + p_1 + ...` in key
//   order;
// - each thread scales its dimensions' numerators by alpha and FMA-adds
//   `p * v` over the tile's keys in order.
//
// Writes `m`, `l` [rows, heads] and `o * v_scale` [rows, heads, Dim]; a row
// without a kept key writes -inf, +0 and +0. A listed block outside the
// row's complete blocks traps. Rows past `*live_rows` return after the PDL
// wait.
template<class T>
__device__ void selected_partials(
    const uint16_t* q, const uint8_t* k_cache, const uint8_t* v_cache,
    const int32_t* block_tables, const int32_t* row_batch, const int32_t* positions,
    const uint32_t* live_rows, const int32_t* selected, float* out_max, float* out_den,
    float* out_num, uint32_t max_pages, uint64_t k_page_stride_bytes,
    uint64_t v_page_stride_bytes, uint32_t scale_bits, uint32_t k_scale_bits,
    uint32_t v_scale_bits, uint32_t cp_rank, uint32_t cp_size) {
  ptx::griddep_launch_dependents();
  __shared__ float queries[T::q_heads][T::dim];
  __shared__ float scores[T::q_heads][T::tile];
  __shared__ float alphas[T::q_heads];
  __shared__ int32_t owned[T::top];
  __shared__ int32_t key_page[T::tile];
  __shared__ int32_t key_slot[T::tile];
  __shared__ int owned_count;
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
  const int64_t complete = position < 0 ? 0 : (position + 1) / T::block;
  const int count = int(min(complete, int64_t(T::top)));
  // The owned blocks in list order: one ballot per 32 entries.
  if (warp == 0) {
    int kept = 0;
    for (int base = 0; base < count; base += 32) {
      const int i = base + lane;
      const int32_t b = i < count ? selected[uint64_t(row) * T::top + i] : 0;
      if (i < count && (b < 0 || b >= complete)) __trap();
      const bool mine =
          i < count && (int64_t(b) * T::block / T::page) % cp_size == cp_rank;
      const uint32_t ballot = __ballot_sync(0xffffffffu, mine);
      if (mine) owned[kept + __popc(ballot & ((1u << lane) - 1u))] = b;
      kept += __popc(ballot);
    }
    if (lane == 0) owned_count = kept;
  }
  float maximum = -INFINITY, denominator = 0.0f;
  float numerators[T::q_heads][T::thread_dims];
#pragma unroll
  for (int h = 0; h < T::q_heads; ++h)
#pragma unroll
    for (int i = 0; i < T::thread_dims; ++i) numerators[h][i] = 0.0f;
  __syncthreads();
  const int64_t tail_first = complete * T::block;
  const bool tail_kept =
      position >= tail_first && (tail_first / T::page) % cp_size == cp_rank;
  const int listed = owned_count * T::block;
  const int keys = listed + (tail_kept ? int(position + 1 - tail_first) : 0);
  const uint64_t row_bytes = uint64_t(T::kv_heads) * T::dim * T::element_bytes;
  const uint64_t head_bytes = uint64_t(kv_head) * T::dim * T::element_bytes;
  for (int key0 = 0; key0 < keys; key0 += T::tile) {
    const int tile_keys = min(T::tile, keys - key0);
    for (int i = threadIdx.x; i < tile_keys; i += T::threads) {
      const int j = key0 + i;
      const int64_t t = j < listed ? int64_t(owned[j / T::block]) * T::block + j % T::block
                                   : tail_first + (j - listed);
      const uint64_t local = uint64_t(t / T::page) / cp_size;
      if (local >= max_pages) __trap();
      const int32_t physical = block_tables[uint64_t(item) * max_pages + local];
      if (physical < 0) __trap();
      key_page[i] = physical;
      key_slot[i] = int(t % T::page);
    }
    __syncthreads();
    // Scores.
    for (int key = warp; key < tile_keys; key += T::warps) {
      const uint8_t* k_row = k_cache + uint64_t(key_page[key]) * k_page_stride_bytes +
                             uint64_t(key_slot[key]) * row_bytes + head_bytes;
      float dots[T::q_heads];
#pragma unroll
      for (int h = 0; h < T::q_heads; ++h) dots[h] = 0.0f;
#pragma unroll
      for (int e = 0; e < T::lane_dims; ++e) {
        const int d = lane * T::lane_dims + e;
        const float k = paged_partials::element<T>(k_row, d);
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
      for (int key = lane; key < tile_keys; key += 32) top = fmaxf(top, scores[h][key]);
#pragma unroll
      for (int offset = 16; offset > 0; offset /= 2)
        top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, offset));
      const float next = fmaxf(maximum, top);
      const float alpha = maximum == -INFINITY ? 0.0f : exp_of(maximum - next);
      for (int key = lane; key < tile_keys; key += 32)
        scores[h][key] = exp_of(scores[h][key] - next);
      __syncwarp();
      if (lane == 0) {
        float sum = __fmul_rn(denominator, alpha);
        for (int key = 0; key < tile_keys; ++key) sum = __fadd_rn(sum, scores[h][key]);
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
      for (int h = 0; h < T::q_heads; ++h)
        numerators[h][i] = __fmul_rn(numerators[h][i], alphas[h]);
      for (int key = 0; key < tile_keys; ++key) {
        const uint8_t* v_row = v_cache + uint64_t(key_page[key]) * v_page_stride_bytes +
                               uint64_t(key_slot[key]) * row_bytes + head_bytes;
        const float v = paged_partials::element<T>(v_row, d);
#pragma unroll
        for (int h = 0; h < T::q_heads; ++h)
          numerators[h][i] = fmaf(scores[h][key], v, numerators[h][i]);
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

}  // namespace chop::selected_partials
