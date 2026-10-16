#pragma once

// Indexer score and selection (README.md "Selection"). `select_split`
// scores one row's pools in one split of its range with BF16 MMA and keeps
// the split's best pools in a streaming top-k; `select_merge` folds the
// splits' lists into the row's key list (`Global`) or its (score, pool)
// pairs (`PerRank`). `topk_merge` folds per-rank pairs into the key list,
// and `compact_selection` gathers key list rows.

#include "pooled_indexer/common.cuh"
#include "common/ptx_matrix.cuh"

namespace chop::pooled_indexer {

template<int Heads, int Dim, int Group, int Page, int MaxTopK, bool Tail, bool PerRank>
struct Select {
  static_assert(Heads % 16 == 0 && Heads <= 64, "heads fill m16 tiles");
  static_assert(Dim == 128, "one 128-K chain per head");
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
  static constexpr int relu_stride = Heads + 1;
  using TopKBuffer = TopK<max_kept, threads>;
  // Dynamic shared memory of `select_split`: two key stages, the query, the
  // ReLU dots, the head weights, the stage valid flags and the top-k.
  static constexpr int key_stage = tile * row_bytes;
  static constexpr int smem = 2 * key_stage + Heads * row_bytes + tile * relu_stride * 4 +
                              Heads * 4 + 2 * tile * 4 + TopKBuffer::smem;
};

// The XOR-swizzled 16-byte chunk `chunk` of shared row `row` (256-byte
// rows), so ldmatrix rows hit distinct banks.
template<class T>
__device__ __forceinline__ uint8_t* chunk_at(uint8_t* base, int row, int chunk) {
  return base + row * T::row_bytes + ((chunk ^ (row & 7)) << 4);
}

// Context-parallel pools (README.md "Pages"). Group rank `rank` of `size`
// holds physical pages j with j % size == rank, page j at stripe j / size
// of its table. Its owned pools, in sequence order, are its local pools:
// local pool l lies in stripe l / page_pools at row l % page_pools.

// Owned pools among the first `pools` pools of a sequence.
template<class T>
__device__ __forceinline__ uint32_t rank_pools(uint32_t pools, uint32_t rank, uint32_t size) {
  const uint32_t stripe = T::page_pools * size, tail = pools % stripe;
  const uint32_t before = rank * T::page_pools;
  const uint32_t partial = tail > before ? min(tail - before, uint32_t(T::page_pools)) : 0u;
  return pools / stripe * T::page_pools + partial;
}

// The sequence pool of local pool `local`.
template<class T>
__device__ __forceinline__ uint32_t rank_pool(uint32_t local, uint32_t rank, uint32_t size) {
  return (local / T::page_pools * size + rank) * T::page_pools + local % T::page_pools;
}

// The local range of split `split` among `splits`: tile-aligned chunks of
// the rank's `pools` owned candidates.
template<class T>
__device__ __forceinline__ void split_range(uint32_t pools, uint32_t split, uint32_t splits,
                                            uint32_t& begin, uint32_t& end) {
  const uint32_t chunk = (pools + splits - 1) / splits;
  const uint32_t aligned = (chunk + T::tile - 1) / T::tile * T::tile;
  begin = min(pools, split * aligned);
  end = min(pools, begin + aligned);
}

// Starts the copy of the keys of local pools [base, base + tile) below
// `end` into `stage`; pools past `end` are zero-filled and flagged invalid.
template<class T>
__device__ __forceinline__ void load_tile(uint8_t* stage, uint32_t* valid, const uint8_t* keys,
                                          const int32_t* table, uint64_t page_stride,
                                          uint32_t base, uint32_t end) {
  for (int c = threadIdx.x; c < T::tile * T::row_chunks; c += T::threads) {
    const int slot = c / T::row_chunks, chunk = c % T::row_chunks;
    const uint32_t local = base + slot;
    const bool live = local < end;
    const uint8_t* source =
        live ? keys + size_t(table[local / T::page_pools]) * page_stride +
                   size_t(local % T::page_pools) * T::row_bytes + chunk * 16
             : keys;
    cp_async16(chunk_at<T>(stage, slot, chunk), source, live);
    if (chunk == 0) valid[slot] = live;
  }
  cp_async_commit();
}

// `relu(q_h . K_g * dim^-0.5)` for the stage's pools: warp w forms the
// 8 pools of n8 column w for every head over one 128-K chain from zero.
template<class T>
__device__ __forceinline__ void relu_dots(uint8_t* q, uint8_t* stage, float* relu, float scale) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  ptx::D4 d[T::heads / 16] = {};
#pragma unroll
  for (int ks = 0; ks < T::dim / 16; ks += 2) {
    uint32_t b[4];
    ptx::ldmatrix_b16<4>(b, chunk_at<T>(stage, warp * 8 + (lane & 7), 2 * ks + (lane >> 3)));
#pragma unroll
    for (int step = 0; step < 2; ++step) {
      const ptx::B2 frag{{b[2 * step], b[2 * step + 1]}};
#pragma unroll
      for (int m = 0; m < T::heads / 16; ++m) {
        ptx::A4 a;
        ptx::ldmatrix_b16<4>(a.v, chunk_at<T>(q, m * 16 + (lane & 15),
                                              2 * (ks + step) + (lane >> 4)));
        ptx::mma_bf16(d[m], a, frag);
      }
    }
  }
#pragma unroll
  for (int m = 0; m < T::heads / 16; ++m) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int head = m * 16 + lane / 4 + (i >> 1) * 8;
      const int pool = warp * 8 + 2 * (lane & 3) + (i & 1);
      const float x = d[m].v[i] * scale;
      relu[pool * T::relu_stride + head] = x > 0.f ? x : 0.f;
    }
  }
}

template<class T>
__device__ __forceinline__ uint32_t visible_pools(int64_t position) {
  return uint32_t((position + 1) / T::group);
}

// One CTA per (row, split) over the pools group rank `cp_rank` of
// `cp_size` owns (`Global` takes 0 of 1). `workspace` is u64
// [rows][splits][max_kept]: the split's best keys descending, then zeros. A
// `Global` row with no more pools than it keeps has nothing to score.
template<class T>
__device__ void select_split(const uint16_t* q, const uint16_t* head_weights,
                             const uint8_t* pooled_keys, const int32_t* page_tables,
                             const int32_t* row_batch, const int32_t* positions,
                             const uint32_t* live_rows, uint64_t* workspace, uint32_t max_pages,
                             uint64_t page_stride_bytes, uint32_t top_k, uint32_t splits,
                             uint32_t cp_rank, uint32_t cp_size) {
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* stages = smem;
  uint8_t* qs = stages + 2 * T::key_stage;
  float* relu = reinterpret_cast<float*>(qs + T::heads * T::row_bytes);
  float* weights = relu + T::tile * T::relu_stride;
  uint32_t* valid = reinterpret_cast<uint32_t*>(weights + T::heads);
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

  const uint8_t* query = reinterpret_cast<const uint8_t*>(q + size_t(row) * T::heads * T::dim);
  for (int c = threadIdx.x; c < T::heads * T::row_chunks; c += T::threads)
    cp_async16(chunk_at<T>(qs, c / T::row_chunks, c % T::row_chunks), query + c * 16, true);
  load_tile<T>(stages, valid, pooled_keys, table, page_stride_bytes, begin, end);
  const float head_scale = __frsqrt_rn(float(T::heads));
  for (int h = threadIdx.x; h < T::heads; h += T::threads)
    weights[h] = bf16(head_weights[size_t(row) * T::heads + h]) * head_scale;
  best.init();
  const float scale = __frsqrt_rn(float(T::dim));

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
    relu_dots<T>(qs, stages + stage * T::key_stage, relu, scale);
    __syncthreads();
    if (threadIdx.x < T::tile && valid[stage * T::tile + threadIdx.x]) {
      const float* r = relu + threadIdx.x * T::relu_stride;
      float score = 0.f;
#pragma unroll
      for (int h = 0; h < T::heads; ++h) score = fmaf(weights[h], r[h], score);
      best.offer(candidate_key(score, rank_pool<T>(base + threadIdx.x, cp_rank, cp_size)));
    }
    best.make_room(kept, T::tile);
  }
  best.reduce(kept);
  const uint32_t count = *best.count;
  for (int i = threadIdx.x; i < T::max_kept; i += T::threads)
    out[i] = uint32_t(i) < count ? best.keys[i] : 0;
}

// Streams `lists` lists of `max_kept` keys, each descending with zero
// padding, into `best`; a list stops at its first key at or below the
// threshold.
template<class T, class Load>
__device__ __forceinline__ void fold_lists(typename T::TopKBuffer& best, uint32_t kept,
                                           uint32_t lists, Load load) {
  for (uint32_t list = 0; list < lists; ++list) {
    for (int base = 0; base < T::max_kept; base += T::threads) {
      if (load(list, base) <= *best.threshold) break;
      const int i = base + threadIdx.x;
      if (i < T::max_kept) best.offer(load(list, i));
      best.make_room(kept, T::threads < T::max_kept ? T::threads : T::max_kept);
    }
  }
}

// The key list of a row at `position` keeping `count` pools, `pool(j)`
// ascending: their positions, the always-selected tail, then -1.
template<class T, class Pool>
__device__ __forceinline__ void write_list(int32_t* list, int64_t position, uint32_t count,
                                           Pool pool) {
  const int64_t base = (position + 1) / T::group * T::group;
  const uint32_t tail = T::tail ? uint32_t(position + 1 - base) : 0;
  for (int e = threadIdx.x; e < T::list_width; e += T::threads) {
    const uint32_t j = e / T::group;
    int32_t value = -1;
    if (j < count) {
      value = int32_t(pool(j) * T::group + e % T::group);
    } else if (uint32_t(e) - count * T::group < tail) {
      value = int32_t(base + (e - count * T::group));
    }
    list[e] = value;
  }
}

// Orders the kept keys by pool ascending and writes the key list.
template<class T>
__device__ __forceinline__ void write_kept(typename T::TopKBuffer& best, int32_t* list,
                                           int64_t position) {
  const uint32_t count = *best.count;
  for (int i = threadIdx.x; i < T::max_kept; i += T::threads) {
    const uint64_t key = best.keys[i];
    best.keys[i] = uint32_t(i) < count ? key << 32 | key >> 32 : 0;
  }
  __syncthreads();
  sort_descending<T::max_kept, T::threads>(best.keys);
  write_list<T>(list, position, count,
                [&](uint32_t j) { return ~uint32_t(best.keys[j] >> 32); });
}

// One CTA per row. `Global` writes the key list `[rows, list_width]`;
// `PerRank` writes `[rows, max_kept]` (F32 score, i32 pool) pairs, best
// first, padded with (-inf, -1).
template<class T>
__device__ void select_merge(const uint64_t* workspace, const int32_t* positions,
                             const uint32_t* live_rows, void* output, uint32_t top_k,
                             uint32_t splits) {
  __shared__ __align__(16) uint8_t storage[T::TopKBuffer::smem];
  typename T::TopKBuffer best(storage);
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  const int64_t position = positions[row];
  const uint32_t pools = visible_pools<T>(position), kept = top_k / T::group;
  if constexpr (!T::per_rank) {
    if (pools <= kept) {
      write_list<T>(static_cast<int32_t*>(output) + size_t(row) * T::list_width, position, pools,
                    [](uint32_t j) { return j; });
      return;
    }
  }
  best.init();
  const uint64_t* lists = workspace + size_t(row) * splits * T::max_kept;
  fold_lists<T>(best, kept, splits,
                [&](uint32_t list, int i) { return lists[size_t(list) * T::max_kept + i]; });
  best.reduce(kept);
  if constexpr (T::per_rank) {
    uint32_t* pairs = static_cast<uint32_t*>(output) + size_t(row) * T::max_kept * 2;
    const uint32_t count = *best.count;
    for (int i = threadIdx.x; i < T::max_kept; i += T::threads) {
      const bool live = uint32_t(i) < count;
      pairs[2 * i] = live ? __float_as_uint(candidate_score(best.keys[i])) : 0xff800000u;
      pairs[2 * i + 1] = live ? candidate_pool(best.keys[i]) : 0xffffffffu;
    }
  } else {
    write_kept<T>(best, static_cast<int32_t*>(output) + size_t(row) * T::list_width, position);
  }
}

// The exact merge of `Ranks` per-rank pair lists, gathered rank-major as
// [Ranks][rank_rows][max_kept][2], into the row's key list.
template<class T, int Ranks>
__device__ void topk_merge(const uint32_t* pairs, const int32_t* positions,
                           const uint32_t* live_rows, int32_t* lists, uint32_t top_k,
                           uint32_t rank_rows) {
  static_assert(!T::per_rank && Ranks >= 2 && Ranks <= 8);
  __shared__ __align__(16) uint8_t storage[T::TopKBuffer::smem];
  typename T::TopKBuffer best(storage);
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  best.init();
  fold_lists<T>(best, top_k / T::group, Ranks, [&](uint32_t rank, int i) -> uint64_t {
    const uint32_t* pair = pairs + ((size_t(rank) * rank_rows + row) * T::max_kept + i) * 2;
    return int32_t(pair[1]) < 0 ? 0 : candidate_key(__uint_as_float(pair[0]), pair[1]);
  });
  best.reduce(top_k / T::group);
  write_kept<T>(best, lists + size_t(row) * T::list_width, positions[row]);
}

// `out[i] = lists[source[i]]` over i32 rows of `Width`; one CTA per row.
template<int Width>
__device__ void compact_selection(const int32_t* lists, const int32_t* source,
                                  const uint32_t* live_rows, int32_t* out) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  const int32_t* from = lists + size_t(source[row]) * Width;
  for (int e = threadIdx.x; e < Width; e += blockDim.x) out[size_t(row) * Width + e] = from[e];
}

}  // namespace chop::pooled_indexer
