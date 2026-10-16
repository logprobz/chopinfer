#pragma once

// One-split selection (README.md "One-split selection"): the selection of
// `select_split` and `select_merge` at one split per row, as one kernel. A
// CTA scores every pool of `Rows` rows and writes their key lists
// (`Global`) or (score, pool) pairs (`PerRank`) itself, so no split
// workspace or merge launch exists. Two rows of one sequence share each
// key tile.
//
// Every score is `select_split`'s instruction sequence: the same MMA
// operands and k order into one 128-K chain from zero per head, the same
// scale and ReLU, and the head-ordered F32 FMA chain. Candidates are the
// same distinct 64-bit keys, and the streaming top-k keeps the best
// `top_k / Group` of them whatever the order of offers and reductions, so
// the kept set, and with it the list or the pairs, are the two kernels'
// bit for bit.
//
// - Warp w scores n8 column w % 8 of each tile for CTA row w / 8, with its
//   row's query held as A fragments in registers, and sums its own 8
//   pools' head chains on lanes 0 to 7: the dots stay in the warp, so one
//   barrier per tile remains. With one row per CTA the dots go to the 2 KiB
//   of key rows the warp alone has just read; two rows' warps share their
//   key rows, so each has a buffer of its own.
// - `Stages` key stages keep `Stages - 1` tiles in flight, and each thread
//   reads the page table entries of its copies one tile before it copies.
// - Each warp counts its offers by ballot into a per-tile slot, so every
//   thread knows each row's buffer fill after the tile barrier without
//   another one; a buffer about to overflow is reduced there, both rows'
//   together.

#include "pooled_indexer/select.cuh"

namespace chop::pooled_indexer {

template<class S, int Rows, int Stages>
struct RowSelect {
  using Select = S;
  static_assert(Rows == 1 || Rows == 2);
  static_assert(Stages >= 2 && Stages <= 4);
  static_assert(S::heads <= 32, "a warp holds its row's query as fragments");
  static constexpr int rows = Rows, stages = Stages;
  // A row's threads: the selection's CTA.
  static constexpr int row_threads = S::threads;
  static constexpr int threads = Rows * row_threads, warps = threads / 32;
  static constexpr int m_tiles = S::heads / 16, k_steps = S::dim / 16;
  // 16-byte copies per thread and tile.
  static constexpr int copies = S::tile * S::row_chunks / threads;
  static_assert(S::tile * S::row_chunks % threads == 0);
  static constexpr int capacity = 2 * S::max_kept;
  // Each row thread holds capacity / row_threads of its buffer's keys in a
  // trim (`rows::Entry` refuses maxima below 128 kept pools).
  static_assert(capacity % row_threads == 0 && capacity >= row_threads,
                "a row's buffer spreads evenly over its threads");
  // The one-row list's bitmap: a bit per visible pool in the key stages,
  // which the list no longer needs; rows with more visible pools sort.
  static constexpr uint32_t bitmap_pools = uint32_t(Stages) * S::key_stage * 8;
  // A warp's ReLU dots: [8 pools][heads + 4] F32, 16-byte rows.
  static constexpr int relu_stride = S::heads + 4;
  static constexpr int relu_bytes = 8 * relu_stride * 4;
  static_assert(relu_bytes <= 8 * S::row_bytes, "a column's key rows hold its dots");
  static_assert(S::heads % 4 == 0, "the chains read their dots and weights by four");
  // A row's radix select: 256 digit counts, eight warp sums, and the
  // prefix, rank and state its passes leave.
  static constexpr int radix_bytes = 256 * 4 + 8 * 4 + 16;
  // Dynamic shared memory: the key stages, then per row its top-k buffer
  // (keys, count, threshold), then each warp's ReLU dots (two rows only),
  // each row's head weights, each warp's offer count for two tiles, and
  // each row's radix select.
  static constexpr int topk_bytes = capacity * 8 + 16;
  static constexpr int topk_offset = Stages * S::key_stage;
  static constexpr int relu_offset = topk_offset + Rows * topk_bytes;
  static constexpr int weights_offset = relu_offset + (Rows == 1 ? 0 : warps * relu_bytes);
  static constexpr int offers_offset = weights_offset + Rows * S::heads * 4;
  static constexpr int radix_offset = offers_offset + 2 * warps * 4;
  static constexpr int smem = radix_offset + Rows * radix_bytes;
  static_assert(topk_bytes % 16 == 0 && relu_bytes % 16 == 0 && radix_offset % 16 == 0);
};

// A CTA row's streaming top-k buffer (`TopK`'s layout and rules).
template<class T>
struct RowBuffer {
  uint64_t* keys;
  uint32_t* count;
  uint64_t* threshold;

  __device__ __forceinline__ RowBuffer(uint8_t* smem, int slot)
      : keys(reinterpret_cast<uint64_t*>(smem + T::topk_offset + slot * T::topk_bytes)),
        count(reinterpret_cast<uint32_t*>(keys + T::capacity)),
        threshold(keys + T::capacity + 1) {}
};

// Bitonic sort of `N` keys descending by the `Threads` threads of one CTA
// row, `local` this thread's index among them; every thread of the CTA
// takes part, each row sorting its own keys. `sort_descending`'s stages.
template<int N, int Threads>
__device__ __forceinline__ void sort_row(uint64_t* keys, int local) {
  static_assert((N & (N - 1)) == 0, "a bitonic sort takes a power of two");
  for (int size = 2; size <= N; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      for (int i = local; i < N / 2; i += Threads) {
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

// `write_list` by one CTA row's threads.
template<class S, class Pool>
__device__ __forceinline__ void write_row_list(int32_t* list, int64_t position, uint32_t count,
                                               Pool pool, int local) {
  const int64_t base = (position + 1) / S::group * S::group;
  const uint32_t tail = S::tail ? uint32_t(position + 1 - base) : 0;
  for (int e = local; e < S::list_width; e += S::threads) {
    const uint32_t j = e / S::group;
    int32_t value = -1;
    if (j < count) {
      value = int32_t(pool(j) * S::group + e % S::group);
    } else if (uint32_t(e) - count * S::group < tail) {
      value = int32_t(base + (e - count * S::group));
    }
    list[e] = value;
  }
}

// `write_list`'s entries after `count` kept pools: the always-selected
// tail, then -1, by one CTA row's threads.
template<class S>
__device__ __forceinline__ void write_row_tail(int32_t* list, int64_t position, uint32_t count,
                                               int local) {
  const int64_t base = (position + 1) / S::group * S::group;
  const uint32_t tail = S::tail ? uint32_t(position + 1 - base) : 0;
  for (int e = int(count) * S::group + local; e < S::list_width; e += S::threads) {
    const uint32_t after = uint32_t(e) - count * S::group;
    list[e] = after < tail ? int32_t(base + after) : -1;
  }
}

// `values[i]` for a row index known only at run time, without indexing a
// register array.
template<class V, int N>
__device__ __forceinline__ V pick(const V (&values)[N], int i) {
  V value = values[0];
#pragma unroll
  for (int j = 1; j < N; ++j)
    if (i == j) value = values[j];
  return value;
}

// The dot of `head` and pool `pool` (0 to 7) in a warp's ReLU dots: a
// fragment's stores (pools 2c, 2c + 1, heads g, g + 8, ...) and the chain
// lanes' 16-byte loads both hit distinct banks.
template<class T>
__device__ __forceinline__ int relu_at(int head, int pool) {
  return pool * T::relu_stride + head;
}

// Lab-only phase clocks (`CHOP_POOLED_ROWS_TEST_CLOCKS`): thread 0 adds
// the cycles since its previous mark to phase `phase` (0 copy wait, 1 tile
// barrier, 2 fill and reductions, 3 copy issue and page reads, 4 MMAs and
// dots, 5 head chains and offers, 6 prologue, 7 the final cut and list;
// 8 counts reductions, 9 tiles), which a row's list then carries.
#ifdef CHOP_POOLED_ROWS_TEST_CLOCKS
#define CHOP_ROWS_CLOCK(c, phase)                      \
  do {                                                 \
    if (threadIdx.x == 0) {                            \
      const long long now = clock64();                 \
      (c).clocks[phase] += now - (c).last;             \
      (c).last = now;                                  \
    }                                                  \
  } while (0)
#define CHOP_ROWS_COUNT(c, slot) \
  do {                           \
    if (threadIdx.x == 0) ++(c).clocks[slot]; \
  } while (0)
#else
#define CHOP_ROWS_CLOCK(c, phase) \
  do {                            \
  } while (0)
#define CHOP_ROWS_COUNT(c, slot) \
  do {                           \
  } while (0)
#endif

// What every thread of the CTA holds alike: each row's buffer fill, and the
// tile count whose parity picks the offer slots.
template<class T>
struct RowScan {
  uint8_t* smem;
  const uint8_t* pooled_keys;
  uint64_t page_stride_bytes;
  uint32_t kept, cp_rank, cp_size;
  float scale;
  uint32_t live[T::rows];
  uint32_t tick;
  bool pending;
#ifdef CHOP_POOLED_ROWS_TEST_CLOCKS
  long long last = 0;
  long long clocks[10] = {};
#endif
};

// Adds the previous tile's offers to each row's fill. Called after a
// barrier that follows the tile.
template<class T>
__device__ __forceinline__ void settle(RowScan<T>& c) {
  if (!c.pending) return;
  const uint32_t* offers =
      reinterpret_cast<const uint32_t*>(c.smem + T::offers_offset) + ((c.tick - 1) & 1) * T::warps;
#pragma unroll
  for (int r = 0; r < T::rows; ++r) {
    uint32_t sum = 0;
#pragma unroll
    for (int w = 0; w < 8; ++w) sum += offers[r * 8 + w];
    c.live[r] += sum;
  }
  c.pending = false;
}

// Keeps each row's best `kept` keys when it holds more, in any order, with
// the threshold a lower bound of its kept-th best key: an exact radix
// select from the top in 8-bit digits. Each pass counts the keys that match
// the prefix so far per digit with shared atomic adds (the counts do not
// depend on their order), a scan finds the digit holding the kept-th key,
// and the passes end when that digit's keys are exactly the rest to keep.
// The keys at or above the prefix (padded with zeros) are then the best
// `kept`, which the distinct keys make exactly `kept` of; they move to the
// front. Every thread takes part; called after a barrier that follows
// every offer.
template<class T>
__device__ __forceinline__ void trim_rows(RowScan<T>& c) {
  CHOP_ROWS_COUNT(c, 8);
  const int slot = threadIdx.x / T::row_threads, local = threadIdx.x % T::row_threads;
  const int lane = threadIdx.x & 31, warp = local >> 5;
  RowBuffer<T> best(c.smem, slot);
  uint8_t* radix = c.smem + T::radix_offset + slot * T::radix_bytes;
  uint32_t* counts = reinterpret_cast<uint32_t*>(radix);
  uint32_t* sums = counts + 256;
  uint64_t* prefix = reinterpret_cast<uint64_t*>(sums + 8);
  uint32_t* state = reinterpret_cast<uint32_t*>(prefix + 1);
  const uint32_t live = pick(c.live, slot);
  // A row with no more than it keeps is done before the first pass.
  if (local == 0) {
    *prefix = 0;
    state[0] = c.kept;              // the rank of the wanted key below the prefix
    state[1] = live <= c.kept;      // done
  }
  // Each thread's keys, read once (at most capacity / row threads).
  constexpr int held = T::capacity / T::row_threads;
  uint64_t keys[held];
#pragma unroll
  for (int j = 0; j < held; ++j) {
    const uint32_t i = local + j * T::row_threads;
    keys[j] = i < live ? best.keys[i] : 0;
  }
#pragma unroll 1
  for (int pass = 0; pass < 8; ++pass) {
    counts[local] = 0;
    __syncthreads();
    // The state the previous pass left; the deciding thread rewrites it only
    // after the next barrier.
    const bool done = state[1];
    const uint32_t rank = state[0];
    const uint64_t want = *prefix;
    bool all = done;
#pragma unroll
    for (int r = 0; r < T::rows; ++r) {
      const uint32_t* other =
          reinterpret_cast<const uint32_t*>(c.smem + T::radix_offset + r * T::radix_bytes + 256 * 4 +
                                            8 * 4 + 8);
      all &= other[1] != 0;
    }
    if (all) break;
    const int shift = 56 - 8 * pass;
    if (!done) {
#pragma unroll
      for (int j = 0; j < held; ++j) {
        const uint32_t i = local + j * T::row_threads;
        const bool match = pass == 0 || (keys[j] >> (shift + 8)) == (want >> (shift + 8));
        if (i < live && match) atomicAdd(&counts[(keys[j] >> shift) & 255], 1u);
      }
    }
    __syncthreads();
    // The keys of digits 255 - local and above, by an inclusive scan over
    // the reversed counts.
    const uint32_t mine = counts[255 - local];
    uint32_t total = mine;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      const uint32_t up = __shfl_up_sync(0xffffffffu, total, offset);
      if (lane >= offset) total += up;
    }
    if (lane == 31) sums[warp] = total;
    __syncthreads();
    for (int w = 0; w < warp; ++w) total += sums[w];
    if (!done && total - mine < rank && rank <= total) {
      const uint32_t left = rank - (total - mine);
      *prefix = want | uint64_t(255 - local) << shift;
      state[0] = left;
      state[1] = left == mine || pass == 7;
    }
    __syncthreads();
  }
  const bool trimmed = live > c.kept;
  const uint64_t bound = *prefix;
  if (local == 0) *best.count = 0;
  __syncthreads();
  if (trimmed) {
#pragma unroll
    for (int j = 0; j < held; ++j) {
      const uint32_t i = local + j * T::row_threads;
      if (i < live && keys[j] >= bound) best.keys[atomicAdd(best.count, 1u)] = keys[j];
    }
  }
  __syncthreads();
  if (local == 0) {
    if (trimmed) {
      *best.threshold = bound;
    } else {
      *best.count = live;
    }
  }
  __syncthreads();
#pragma unroll
  for (int r = 0; r < T::rows; ++r) c.live[r] = min(c.live[r], c.kept);
}

// Sorts each row's first `count` keys descending into its first
// `S::max_kept` slots, zeros after them. Every thread takes part.
template<class T>
__device__ __forceinline__ void sort_kept(RowScan<T>& c) {
  using S = typename T::Select;
  const int slot = threadIdx.x / T::row_threads, local = threadIdx.x % T::row_threads;
  RowBuffer<T> best(c.smem, slot);
  const uint32_t count = *best.count;
  for (int i = count + local; i < S::max_kept; i += T::row_threads) best.keys[i] = 0;
  __syncthreads();
  sort_row<S::max_kept, T::row_threads>(best.keys, local);
}

// A `Global` row's list from its first `count` kept keys: pool-major
// (`write_kept`), sorted, then written. Every thread of the CTA takes part.
template<class T>
__device__ __forceinline__ void sort_list(RowScan<T>& c, bool write, int32_t* list,
                                          int64_t position, uint32_t count) {
  using S = typename T::Select;
  const int slot = threadIdx.x / T::row_threads, local = threadIdx.x % T::row_threads;
  RowBuffer<T> best(c.smem, slot);
  for (int i = local; i < S::max_kept; i += T::row_threads) {
    const uint64_t key = best.keys[i];
    best.keys[i] = uint32_t(i) < count ? key << 32 | key >> 32 : 0;
  }
  __syncthreads();
  sort_row<S::max_kept, T::row_threads>(best.keys, local);
  if (write)
    write_row_list<S>(list, position, count,
                      [&](uint32_t j) { return ~uint32_t(best.keys[j] >> 32); }, local);
}

// A one-row CTA's list through a bitmap of the row's visible pools (at
// most `T::bitmap_pools`) in the free key stages: each kept pool sets its
// bit, and each thread writes the pools of its run of words at their rank,
// the set bits of the runs before it.
template<class T>
__device__ __forceinline__ void bitmap_list(RowScan<T>& c, bool write, int32_t* list,
                                            int64_t position, uint32_t count) {
  using S = typename T::Select;
  static_assert(T::rows == 1);
  const int local = threadIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  uint8_t* smem = c.smem;
  RowBuffer<T> best(smem, 0);
  uint32_t* bits = reinterpret_cast<uint32_t*>(smem);
  uint32_t* sums = reinterpret_cast<uint32_t*>(smem + T::radix_offset + 256 * 4);
  const uint32_t words = (visible_pools<S>(position) + 31) / 32;
  for (uint32_t w = local; w < words; w += T::row_threads) bits[w] = 0;
  __syncthreads();
  for (uint32_t i = local; i < count; i += T::row_threads) {
    const uint32_t pool = candidate_pool(best.keys[i]);
    atomicOr(&bits[pool >> 5], 1u << (pool & 31));
  }
  __syncthreads();
  const uint32_t per = (words + T::row_threads - 1) / T::row_threads;
  const uint32_t from = min(words, local * per), to = min(words, from + per);
  uint32_t ones = 0;
  for (uint32_t w = from; w < to; ++w) ones += __popc(bits[w]);
  uint32_t rank = ones;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const uint32_t up = __shfl_up_sync(0xffffffffu, rank, offset);
    if (lane >= offset) rank += up;
  }
  if (lane == 31) sums[warp] = rank;
  __syncthreads();
  for (int w = 0; w < warp; ++w) rank += sums[w];
  rank -= ones;
  if (write) {
    for (uint32_t w = from; w < to; ++w) {
      for (uint32_t word = bits[w]; word != 0; word &= word - 1, ++rank) {
        const uint32_t pool = w * 32 + __ffs(word) - 1;
#pragma unroll
        for (int g = 0; g < S::group; ++g) {
          const uint32_t e = rank * S::group + g;
          if (e < uint32_t(S::list_width)) list[e] = int32_t(pool * S::group + g);
        }
      }
    }
    write_row_tail<S>(list, position, count, local);
  }
}

// This thread's copy `i` of a tile: its pool slot and 16-byte chunk.
template<class T>
__device__ __forceinline__ int copy_slot(int i) {
  return (threadIdx.x + i * T::threads) / T::Select::row_chunks;
}

// The page table entries of this thread's copies of the tile at `base`.
template<class T>
__device__ __forceinline__ void tile_pages(int32_t (&pages)[T::copies], const int32_t* table,
                                           uint32_t base, uint32_t end) {
#pragma unroll
  for (int i = 0; i < T::copies; ++i) {
    const uint32_t local = base + copy_slot<T>(i);
    pages[i] = local < end ? table[local / T::Select::page_pools] : 0;
  }
}

// Starts the copy of the tile at `base` into `stage` from `pages`, pools
// past `end` zero-filled, and commits it; a tile wholly past `end` commits
// an empty group.
template<class T>
__device__ __forceinline__ void copy_tile(const RowScan<T>& c, uint8_t* stage,
                                          const int32_t (&pages)[T::copies], uint32_t base,
                                          uint32_t end) {
  using S = typename T::Select;
  if (base < end) {
#pragma unroll
    for (int i = 0; i < T::copies; ++i) {
      const int slot = copy_slot<T>(i);
      const int chunk = (threadIdx.x + i * T::threads) % S::row_chunks;
      const uint32_t local = base + slot;
      const bool live = local < end;
      const uint8_t* source =
          live ? c.pooled_keys + size_t(pages[i]) * c.page_stride_bytes +
                     size_t(local % S::page_pools) * S::row_bytes + chunk * 16
               : c.pooled_keys;
      cp_async16(chunk_at<S>(stage, slot, chunk), source, live);
    }
  }
  cp_async_commit();
}

// Scores the local pools `[0, max(end))` over `table` and offers row r's
// pools below `end[r]` to its buffer. Rows of one pass share the table.
template<class T>
__device__ __forceinline__ void scan_pools(RowScan<T>& c, const ptx::A4 (&query)[T::m_tiles][T::k_steps],
                                           const int32_t* table, const uint32_t (&end)[T::rows]) {
  using S = typename T::Select;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int slot = warp / 8, column = warp % 8;
  const uint32_t mine = pick(end, slot);
  uint32_t total = end[0];
#pragma unroll
  for (int r = 1; r < T::rows; ++r) total = max(total, end[r]);
  if (total == 0) return;
  uint8_t* stages = c.smem;
  const float* weights = reinterpret_cast<const float*>(c.smem + T::weights_offset) + slot * S::heads;
  uint32_t* offers = reinterpret_cast<uint32_t*>(c.smem + T::offers_offset);
  RowBuffer<T> best(c.smem, slot);

  int32_t ahead[T::copies];
#pragma unroll
  for (int s = 0; s < T::stages - 1; ++s) {
    tile_pages<T>(ahead, table, s * S::tile, total);
    copy_tile<T>(c, stages + s * S::key_stage, ahead, s * S::tile, total);
  }
  tile_pages<T>(ahead, table, (T::stages - 1) * S::tile, total);

  const uint32_t tiles = (total + S::tile - 1) / S::tile;
  for (uint32_t t = 0; t < tiles; ++t) {
    cp_async_wait<T::stages - 2>();
    CHOP_ROWS_CLOCK(c, 0);
    __syncthreads();
    CHOP_ROWS_CLOCK(c, 1);
    CHOP_ROWS_COUNT(c, 9);
    settle(c);
    bool full = false;
#pragma unroll
    for (int r = 0; r < T::rows; ++r) full |= c.live[r] + S::tile > T::capacity;
    if (full) trim_rows(c);
    CHOP_ROWS_CLOCK(c, 2);

    // The tile `stages - 1` ahead goes to the stage the previous tile
    // used, which every warp finished before the barrier.
    const uint32_t next = t + T::stages - 1;
    copy_tile<T>(c, stages + next % T::stages * S::key_stage, ahead, next * S::tile, total);
#ifdef CHOP_POOLED_ROWS_TEST_STALE_PAGE
    tile_pages<T>(ahead, table, next * S::tile, total);
#else
    tile_pages<T>(ahead, table, (next + 1) * S::tile, total);
#endif
    CHOP_ROWS_CLOCK(c, 3);

    const uint32_t base = t * S::tile;
    bool offered = false;
    if (base < mine) {
      uint8_t* stage = stages + t % T::stages * S::key_stage;
      // One row: the column's key rows, which no other warp reads and no
      // copy rewrites before the next barrier.
      float* relu = reinterpret_cast<float*>(
          T::rows == 1 ? stage + column * 8 * S::row_bytes
                       : c.smem + T::relu_offset + warp * T::relu_bytes);
      ptx::D4 d[T::m_tiles] = {};
#pragma unroll
      for (int ks = 0; ks < T::k_steps; ks += 2) {
        uint32_t b[4];
        ptx::ldmatrix_b16<4>(b, chunk_at<S>(stage, column * 8 + (lane & 7), 2 * ks + (lane >> 3)));
#pragma unroll
        for (int step = 0; step < 2; ++step) {
          const ptx::B2 frag{{b[2 * step], b[2 * step + 1]}};
#pragma unroll
          for (int m = 0; m < T::m_tiles; ++m) ptx::mma_bf16(d[m], query[m][ks + step], frag);
        }
      }
      __syncwarp();
#pragma unroll
      for (int m = 0; m < T::m_tiles; ++m) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int head = m * 16 + lane / 4 + (i >> 1) * 8;
          const int pool = 2 * (lane & 3) + (i & 1);
          const float x = d[m].v[i] * c.scale;
          relu[relu_at<T>(head, pool)] = x > 0.f ? x : 0.f;
        }
      }
      __syncwarp();
      CHOP_ROWS_CLOCK(c, 4);
      const uint32_t local = base + column * 8 + lane;
      if (lane < 8 && local < mine) {
        // The dots and weights by four, ahead of the head-ordered chain.
        const float4* dots = reinterpret_cast<const float4*>(relu + relu_at<T>(0, lane));
        const float4* w4 = reinterpret_cast<const float4*>(weights);
        float4 r[S::heads / 4], w[S::heads / 4];
#pragma unroll
        for (int q = 0; q < S::heads / 4; ++q) {
          r[q] = dots[q];
          w[q] = w4[q];
        }
        float score = 0.f;
#pragma unroll
        for (int q = 0; q < S::heads / 4; ++q) {
          score = fmaf(w[q].x, r[q].x, score);
          score = fmaf(w[q].y, r[q].y, score);
          score = fmaf(w[q].z, r[q].z, score);
          score = fmaf(w[q].w, r[q].w, score);
        }
        const uint64_t key = candidate_key(score, rank_pool<S>(local, c.cp_rank, c.cp_size));
        if (key > *best.threshold) {
          best.keys[atomicAdd(best.count, 1u)] = key;
          offered = true;
        }
      }
    }
    const uint32_t count = __popc(__ballot_sync(0xffffffffu, offered));
    if (lane == 0) offers[(c.tick & 1) * T::warps + warp] = count;
    ++c.tick;
    c.pending = true;
    CHOP_ROWS_CLOCK(c, 5);
  }
  cp_async_wait<0>();
  __syncthreads();
  settle(c);
}

// One CTA per `Rows` consecutive rows over the pools group rank `cp_rank`
// of `cp_size` owns (`Global` takes 0 of 1). `Global` writes the key lists
// `[rows, list_width]`, `PerRank` the `[rows, max_kept]` (F32 score, i32
// pool) pairs, best first, padded with (-inf, -1): `select_merge`'s
// outputs at one split. A `Global` row with no more pools than it keeps
// selects them all without scoring.
template<class T>
__device__ void select_rows(const uint16_t* q, const uint16_t* head_weights,
                            const uint8_t* pooled_keys, const int32_t* page_tables,
                            const int32_t* row_batch, const int32_t* positions,
                            const uint32_t* live_rows, void* output, uint32_t max_pages,
                            uint64_t page_stride_bytes, uint32_t top_k, uint32_t cp_rank,
                            uint32_t cp_size) {
  using S = typename T::Select;
  extern __shared__ __align__(128) uint8_t smem[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int slot = warp / 8, local = threadIdx.x % T::row_threads;

  ptx::griddep_launch_dependents();
#ifndef CHOP_POOLED_ROWS_TEST_LATE_WAIT
  ptx::griddep_wait();
#endif
#ifdef CHOP_POOLED_ROWS_TEST_CLOCKS
  const long long started = clock64();
#endif
  const uint32_t live = *live_rows, first = blockIdx.x * T::rows;
  if (first >= live) {
#ifdef CHOP_POOLED_ROWS_TEST_LATE_WAIT
    ptx::griddep_wait();
#endif
    return;
  }
  const uint32_t kept = top_k / S::group;
  // Whether a row writes after scoring: a live row, unless it is a `Global`
  // row that keeps every pool it sees.
  const auto scores = [&](uint32_t row, int64_t position) {
    return row < live && (S::per_rank || visible_pools<S>(position) > kept);
  };
  // Per row: its owned pools (0 when it does not score) and its sequence.
  uint32_t end[T::rows];
  int32_t batch[T::rows];
  bool any = false;
#pragma unroll
  for (int r = 0; r < T::rows; ++r) {
    const uint32_t row = first + r;
    const bool on = row < live;
    const int64_t position = on ? positions[row] : 0;
    batch[r] = on ? row_batch[row] : -1;
    const uint32_t visible = visible_pools<S>(position);
    const bool writes = scores(row, position);
    end[r] = writes ? rank_pools<S>(visible, cp_rank, cp_size) : 0;
    any |= writes;
    if constexpr (!S::per_rank) {
      if (on && !writes && slot == r)
        write_row_list<S>(static_cast<int32_t*>(output) + size_t(row) * S::list_width,
                          position, visible, [](uint32_t j) { return j; }, local);
    }
  }
  if (!any) {
#ifdef CHOP_POOLED_ROWS_TEST_LATE_WAIT
    ptx::griddep_wait();
#endif
    return;
  }

  // This warp's row's query as A fragments: rows g and g + 8 of m16 tile m,
  // columns 2c, 2c + 1 and 2c + 8, 2c + 9 of k16 step k (ldmatrix's
  // registers from the selection's shared query tile).
  ptx::A4 query[T::m_tiles][T::k_steps] = {};
  if (pick(end, slot) > 0) {
    const uint16_t* qr = q + size_t(first + slot) * S::heads * S::dim;
    const int g = lane / 4, c4 = lane % 4;
#pragma unroll
    for (int m = 0; m < T::m_tiles; ++m)
#pragma unroll
      for (int k = 0; k < T::k_steps; ++k) {
        const uint16_t* a = qr + (m * 16 + g) * S::dim + k * 16 + 2 * c4;
        query[m][k] = ptx::A4{{*reinterpret_cast<const uint32_t*>(a),
                               *reinterpret_cast<const uint32_t*>(a + 8 * S::dim),
                               *reinterpret_cast<const uint32_t*>(a + 8),
                               *reinterpret_cast<const uint32_t*>(a + 8 * S::dim + 8)}};
      }
  }
  float* weights = reinterpret_cast<float*>(smem + T::weights_offset);
  const float head_scale = __frsqrt_rn(float(S::heads));
  for (int i = threadIdx.x; i < T::rows * S::heads; i += T::threads) {
    const int r = i / S::heads;
    weights[i] = pick(end, r) > 0 ? bf16(head_weights[size_t(first + r) * S::heads + i % S::heads]) *
                                  head_scale
                            : 0.f;
  }
#ifdef CHOP_POOLED_ROWS_TEST_LATE_WAIT
  ptx::griddep_wait();
#endif
  if (local == 0) {
    RowBuffer<T> best(smem, slot);
    *best.count = 0;
    *best.threshold = 0;
  }

  RowScan<T> c{smem, pooled_keys, page_stride_bytes, kept, cp_rank, cp_size,
               __frsqrt_rn(float(S::dim)), {}, 0, false};
#ifdef CHOP_POOLED_ROWS_TEST_CLOCKS
  c.last = started;
  CHOP_ROWS_CLOCK(c, 6);
#endif
  // Rows of one sequence share a pass over its table; otherwise each row
  // that scores takes its own.
  const bool shared = T::rows == 2 && end[0] > 0 && end[T::rows - 1] > 0 &&
                      batch[0] == batch[T::rows - 1];
#pragma unroll
  for (int r = 0; r < T::rows; ++r) {
    if (shared && r > 0) break;
    uint32_t pass[T::rows];
#pragma unroll
    for (int j = 0; j < T::rows; ++j) pass[j] = shared || j == r ? end[j] : 0;
    if (pass[r] > 0) scan_pools<T>(c, query, page_tables + size_t(batch[r]) * max_pages, pass);
  }

  RowBuffer<T> best(smem, slot);
  const uint32_t row = first + slot;
  const int64_t position = row < live ? positions[row] : 0;
  const bool write = scores(row, position);
  if constexpr (S::per_rank) {
    trim_rows(c);
    sort_kept(c);
    if (write) {
      uint32_t* pairs = static_cast<uint32_t*>(output) + size_t(row) * S::max_kept * 2;
      const uint32_t count = *best.count;
      for (int i = local; i < S::max_kept; i += T::row_threads) {
        const bool held = uint32_t(i) < count;
        pairs[2 * i] = held ? __float_as_uint(candidate_score(best.keys[i])) : 0xff800000u;
        pairs[2 * i + 1] = held ? candidate_pool(best.keys[i]) : 0xffffffffu;
      }
    }
  } else {
    bool over = false;
#pragma unroll
    for (int r = 0; r < T::rows; ++r) over |= c.live[r] > kept;
#ifdef CHOP_POOLED_ROWS_TEST_KEEP_EXTRA
    ++c.kept;
#endif
    if (over) trim_rows(c);
    const uint32_t count = *best.count;
    int32_t* list = static_cast<int32_t*>(output) + size_t(row) * S::list_width;
    // One row per CTA writes the kept pools ascending through a bitmap of
    // its visible pools while they fit in it (to 1M tokens at the served
    // shape; the row decides for its whole CTA), else by a sort, as two
    // rows do.
    if constexpr (T::rows == 1) {
      if (visible_pools<S>(position) <= T::bitmap_pools)
        bitmap_list(c, write, list, position, count);
      else
        sort_list(c, write, list, position, count);
    } else {
      sort_list(c, write, list, position, count);
    }
#ifdef CHOP_POOLED_ROWS_TEST_CLOCKS
    __syncthreads();
    CHOP_ROWS_CLOCK(c, 7);
    if (threadIdx.x == 0 && write) {
      for (int i = 0; i < 10; ++i) {
        list[2 * i] = int32_t(uint64_t(c.clocks[i]));
        list[2 * i + 1] = int32_t(uint64_t(c.clocks[i]) >> 32);
      }
    }
#endif
  }
}

}  // namespace chop::pooled_indexer
