#pragma once

// Head-sum selection (README.md "Head-sum selection"): the selection of
// select.cuh scored as the block-sparse indexer scores its blocks
// (attention/sparse_indexer.cuh `block_scores`), so both select from the
// same F32 bits. Per head, two BF16 MMA chains from zero take the even and
// the odd k16 steps; `relu(even + odd)`; the heads are summed by the xor
// tree over lane bits 2 and 3; one multiply by `rsqrtf(dim)`. No head
// weights. The split merge, the per-rank pairs and the exact merge are
// select.cuh's, instantiated with these traits.

#include "pooled_indexer/select.cuh"

namespace chop::pooled_indexer {

template<int Heads, int Dim, int Group, int Page, int MaxTopK, bool Tail, bool PerRank>
struct HeadSumSelect {
  static_assert(Heads == 4, "four heads, rows 0 to 3 of one m16 tile, share the xor tree");
  static_assert(Dim == 128, "two chains of four k16 steps per head");
  static_assert(Group >= 2 && Group <= 16 && (Group & (Group - 1)) == 0);
  static_assert(Page % Group == 0 && MaxTopK % Group == 0);
  static constexpr int heads = Heads, dim = Dim, group = Group, page = Page;
  static constexpr int max_top_k = MaxTopK;
  static constexpr int max_kept = MaxTopK / Group;
  static constexpr bool tail = Tail, per_rank = PerRank;
  static constexpr int list_width = MaxTopK + (Tail ? Group - 1 : 0);
  static constexpr int page_pools = Page / Group;
  static constexpr int threads = 256, warps = threads / 32;
  // Pools per tile: one n8 column per warp.
  static constexpr int tile = 8 * warps;
  static_assert(tile <= max_kept, "a tile of offers fits beside the kept pools");
  static constexpr int row_bytes = Dim * 2, row_chunks = row_bytes / 16;
  // The query tile: one m16 tile of the heads, then zero rows.
  static constexpr int q_rows = 16;
  using TopKBuffer = TopK<max_kept, threads>;
  // Dynamic shared memory of `head_sum_split`: two key stages, the query
  // tile, the tile's scores, the stage valid flags and the top-k.
  static constexpr int key_stage = tile * row_bytes;
  static constexpr int smem = 2 * key_stage + q_rows * row_bytes + tile * 4 + 2 * tile * 4 +
                              TopKBuffer::smem;
};

// The scores of the stage's pools: warp w forms the 8 pools of n8 column w.
// MMA rows g = 0..3 are the heads. Lane bits 2 and 3 are g's low bits, so
// after the two xor steps the lanes with g = 0 hold `(h0 + h1) + (h2 + h3)`
// of pools 2c and 2c + 1 (c = lane), as `block_scores` sums them.
template<class T>
__device__ __forceinline__ void head_sums(uint8_t* q, uint8_t* stage, float* scores,
                                          float scale) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  ptx::D4 even{}, odd{};
#pragma unroll
  for (int ks = 0; ks < T::dim / 16; ks += 2) {
    uint32_t b[4];
    ptx::ldmatrix_b16<4>(b, chunk_at<T>(stage, warp * 8 + (lane & 7), 2 * ks + (lane >> 3)));
    ptx::A4 a[2];
#pragma unroll
    for (int step = 0; step < 2; ++step)
      ptx::ldmatrix_b16<4>(a[step].v, chunk_at<T>(q, lane & 15, 2 * (ks + step) + (lane >> 4)));
    ptx::mma_bf16(even, a[0], ptx::B2{{b[0], b[1]}});
    ptx::mma_bf16(odd, a[1], ptx::B2{{b[2], b[3]}});
  }
  float relu[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const float x = even.v[i] + odd.v[i];
    relu[i] = x > 0.0f ? x : 0.0f;
    relu[i] += __shfl_xor_sync(0xffffffff, relu[i], 4);
    relu[i] += __shfl_xor_sync(0xffffffff, relu[i], 8);
  }
  if (lane < 4) {
    scores[warp * 8 + 2 * lane] = relu[0] * scale;
    scores[warp * 8 + 2 * lane + 1] = relu[1] * scale;
  }
}

// `select_split` with head-sum scores: one CTA per (row, split) over the
// pools group rank `cp_rank` of `cp_size` owns; the same workspace.
template<class T>
__device__ void head_sum_split(const uint16_t* q, const uint8_t* pooled_keys,
                               const int32_t* page_tables, const int32_t* row_batch,
                               const int32_t* positions, const uint32_t* live_rows,
                               uint64_t* workspace, uint32_t max_pages,
                               uint64_t page_stride_bytes, uint32_t top_k, uint32_t splits,
                               uint32_t cp_rank, uint32_t cp_size) {
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* stages = smem;
  uint8_t* qs = stages + 2 * T::key_stage;
  float* scores = reinterpret_cast<float*>(qs + T::q_rows * T::row_bytes);
  uint32_t* valid = reinterpret_cast<uint32_t*>(scores + T::tile);
  typename T::TopKBuffer best(valid + 2 * T::tile);

  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x, split = blockIdx.y;
  if (row >= *live_rows) return;
  const uint32_t visible = visible_pools<T>(positions[row]);
  const uint32_t kept = top_k / T::group;
  if (!T::per_rank && visible <= kept) return;
  const uint32_t pools = rank_pools<T>(visible, cp_rank, cp_size);
  uint32_t begin, end;
  split_range<T>(pools, split, splits, begin, end);
  uint64_t* out = workspace + (size_t(row) * splits + split) * T::max_kept;
  if (begin >= end) {
    for (int i = threadIdx.x; i < T::max_kept; i += T::threads) out[i] = 0;
    return;
  }
  const int32_t* table = page_tables + size_t(row_batch[row]) * max_pages;

  // The heads' rows, then zero rows to the m16 tile.
  const uint8_t* query = reinterpret_cast<const uint8_t*>(q + size_t(row) * T::heads * T::dim);
  for (int c = threadIdx.x; c < T::q_rows * T::row_chunks; c += T::threads) {
    const bool live = c < T::heads * T::row_chunks;
    cp_async16(chunk_at<T>(qs, c / T::row_chunks, c % T::row_chunks),
               query + (live ? c * 16 : 0), live);
  }
  load_tile<T>(stages, valid, pooled_keys, table, page_stride_bytes, begin, end);
  best.init();
  const float scale = rsqrtf(float(T::dim));

  int stage = 0;
  for (uint32_t base = begin; base < end; base += T::tile, stage ^= 1) {
    if (base + T::tile < end) {
      load_tile<T>(stages + (stage ^ 1) * T::key_stage, valid + (stage ^ 1) * T::tile,
                   pooled_keys, table, page_stride_bytes, base + T::tile, end);
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();
    head_sums<T>(qs, stages + stage * T::key_stage, scores, scale);
    __syncthreads();
    if (threadIdx.x < T::tile && valid[stage * T::tile + threadIdx.x])
      best.offer(candidate_key(scores[threadIdx.x],
                               rank_pool<T>(base + threadIdx.x, cp_rank, cp_size)));
    best.make_room(kept, T::tile);
  }
  best.reduce(kept);
  const uint32_t count = *best.count;
  for (int i = threadIdx.x; i < T::max_kept; i += T::threads)
    out[i] = uint32_t(i) < count ? best.keys[i] : 0;
}

}  // namespace chop::pooled_indexer
