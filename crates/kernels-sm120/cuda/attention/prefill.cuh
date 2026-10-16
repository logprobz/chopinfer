#pragma once

// FlashAttention-2 style prefill. A CTA works on 128 grouped query rows
// (tokens x Q heads of one KV head) at a time and streams keys through two
// shared-memory stages. Each warp keeps one m16 row tile of Q in registers,
// forms S = QK^T with ldmatrix K fragments, applies the online softmax in
// registers, and reuses the probabilities as A fragments for O += PV with
// ldmatrix.trans V.

#include <math.h>
#include <type_traits>
#include "tile.cuh"

namespace chop::attention {

template<class T>
struct Prefill {
  // Grouped rows per token: a group that does not divide 16 is padded to
  // eight rows, whose padding holds zero queries and is never written.
  static constexpr int tile_group = 16 % T::group == 0 ? T::group : T::group_pad;
  static_assert(16 % tile_group == 0, "prefill packs whole GQA groups in m16 tiles");
  static constexpr int warps = 8;
  static constexpr int threads = warps * 32;
  static constexpr int tokens = 16 * warps / tile_group;
  static constexpr int keys = T::qk + T::v <= 320 ? 32 : 16;
  static constexpr int box = T::ring || T::page >= keys ? keys : T::page;
  static constexpr int stages = 2;
  // Cost of starting a query tile, in key tiles: loading Q and writing its
  // output.
  static constexpr int overhead = 2;
  // A paged row's keys split into at most max_pieces segments of a length
  // that is a multiple of segment_keys (see prefill_segments); the merge
  // combines them.
  static constexpr int max_pieces = 4;
  static constexpr int segment_keys = 256;
  static constexpr int k_bytes = keys * T::qk * 2;
  static constexpr int v_bytes = keys * T::v * 2;
  static constexpr int raw_bytes = keys * (T::qk + T::v);
  static constexpr int smem = T::fp8 ? stages * raw_bytes + k_bytes + v_bytes :
                                       stages * (k_bytes + v_bytes);
};

// Key tiles covering every visible key of query tile `tile`. Key tiles
// start at absolute multiples of the key tile size, so a row sees the same
// tiles whatever query tile holds it (tiles wholly outside its window are
// masked and change nothing), and a paged TMA box never crosses a page.
template<class T>
__device__ __forceinline__ void prefill_keys(
    int prefix, int len, int tile, uint32_t window, int& key_first, int& key_tiles) {
  using P = Prefill<T>;
  const int first_pos = prefix + tile * P::tokens;
  const int last_pos = prefix + min(len, (tile + 1) * P::tokens) - 1;
  const int first = window == UINT32_MAX ? 0 : max(0, first_pos - int(window));
  key_first = first / P::keys * P::keys;
  key_tiles = (last_pos - key_first) / P::keys + 1;
}

// A paged item's query tiles start at the absolute multiple of `tokens` at or
// before its prefix, so a row's tile depends on its position alone; the
// first `shift` rows of the first tile belong to the cached prefix. Ring
// tiles start at the chunk.
template<class T>
__device__ __forceinline__ void prefill_item(
    const int32_t* q_len, const int32_t* prefix_len, int b, int& base, int& span,
    int& shift) {
  const int prefix = prefix_len[b];
  shift = T::ring ? 0 : prefix % Prefill<T>::tokens;
  base = prefix - shift;
  span = q_len[b] + shift;
}

// The segments of every row of the paged query tile at absolute position
// `first_pos` (a multiple of `tokens`): the rows' keys split at multiples of
// a length that is the least multiple of segment_keys giving at most
// max_pieces segments to the tile's last row. The length and every row's
// segment count are the same for all rows of the tile, since both are
// multiples of `tokens`, so each depends on the row's position alone.
struct PrefillSegments {
  int tiles, count;
};

template<class T>
__device__ __forceinline__ PrefillSegments prefill_segments(int first_pos) {
  using P = Prefill<T>;
  constexpr int reach = P::segment_keys * P::max_pieces;
  const int keys = first_pos + P::tokens;
  const int length = P::segment_keys * ((keys + reach - 1) / reach);
  return {length / P::keys, (keys + length - 1) / length};
}

// Sum over i in [0, m) of floor((A i + c) / B) for c >= 0, where A and B are
// powers of two.
template<int A, int B>
__device__ __forceinline__ int64_t floor_sum(int64_t m, int64_t c) {
  if (m <= 0) return 0;
  if constexpr (B % A == 0) {
    constexpr int64_t r = B / A;
    auto below = [](int64_t x) {
      const int64_t k = x / r;
      return r * k * (k - 1) / 2 + k * (x - k * r);
    };
    return below(m + c / A) - below(c / A);
  } else {
    static_assert(A % B == 0);
    return int64_t(A / B) * m * (m - 1) / 2 + m * (c / B);
  }
}

// Paged prefill work is counted in units: one key tile of one query tile.
// Units of the first m query tiles of a batch, matching prefill_keys.
template<class T>
__device__ __forceinline__ int64_t prefill_units(int prefix, int len, uint32_t window, int m) {
  using P = Prefill<T>;
  const int count = (len + P::tokens - 1) / P::tokens;
  int64_t units = m + floor_sum<P::tokens, P::keys>(min(m, count - 1), prefix + P::tokens - 1);
  if (m == count) units += (prefix + len - 1) / P::keys;
  if (window != UINT32_MAX) {
    const int left = int(window);
    const int shifted = left <= prefix ? 0 : (left - prefix + P::tokens - 1) / P::tokens;
    if (m > shifted)
      units -= floor_sum<P::tokens, P::keys>(m - shifted, prefix + shifted * P::tokens - left);
  }
  return units;
}

// Paged prefill visits each batch's query tiles in zigzag order (first,
// last, second, second to last, ...) so every CTA's run mixes short and long
// causal tiles.
__device__ __forceinline__ int zigzag_tile(int count, int order) {
  return order % 2 == 0 ? order / 2 : count - 1 - order / 2;
}
__device__ __forceinline__ int zigzag_order(int count, int tile) {
  return tile < (count + 1) / 2 ? 2 * tile : 2 * (count - 1 - tile) + 1;
}

// Stream-K cost of the first `order` tiles of a batch in zigzag order: their
// key tiles plus the per-tile overhead. `all` is prefill_units(count).
template<class T>
__device__ __forceinline__ int64_t prefill_cost(
    int prefix, int len, uint32_t window, int count, int64_t all, int order) {
  return prefill_units<T>(prefix, len, window, (order + 1) / 2) + all -
         prefill_units<T>(prefix, len, window, count - order / 2) +
         int64_t(Prefill<T>::overhead) * order;
}

// Paged prefill work: the cost units (key tiles plus the per-tile overhead)
// of every live batch, in batch and zigzag tile order, are divided evenly
// over `ctas` CTAs per KV head. CTA j owns the segments whose first key tile
// lies in [start(j), start(j + 1)) and runs each whole; a row with one
// segment gets BF16 output, and each segment of a longer row leaves an F32
// partial that the merge kernel folds in segment order.
struct PrefillPlan {
  int64_t units;
  int ctas;
  __device__ __forceinline__ int64_t start(int cta) const { return units * cta / ctas; }
};

template<class T>
__device__ __forceinline__ PrefillPlan prefill_plan(
    const int32_t* q_len, const int32_t* prefix_len, int live, uint32_t ctas,
    uint32_t window) {
  using P = Prefill<T>;
  int64_t units = 0;
  for (int b = 0; b < live; ++b) {
    if (q_len[b] <= 0) continue;
    int base, span, shift;
    prefill_item<T>(q_len, prefix_len, b, base, span, shift);
    const int count = (span + P::tokens - 1) / P::tokens;
    units += prefill_units<T>(base, span, window, count) + int64_t(P::overhead) * count;
  }
  return {units, int(max(int64_t(1), min(int64_t(ctas), units)))};
}

// Position of one key tile: query tile `tile` of `batch`, visited `order`th,
// its key tile `key`, and that key tile's cost unit.
// `prefix` and `len` span the item's tiles: for paged items they start
// `shift` rows before the chunk (prefill_item).
template<class T>
struct PrefillCursor {
  int64_t unit;
  int batch, order, count, tile, key, keys, key_first, prefix, len, shift;
  PrefillSegments segments;

  __device__ __forceinline__ void load_tile(uint32_t window) {
    tile = T::ring ? tile : zigzag_tile(count, order);
    prefill_keys<T>(prefix, len, tile, window, key_first, keys);
    segments = T::ring ? PrefillSegments{keys, 1} :
                         prefill_segments<T>(prefix + tile * Prefill<T>::tokens);
  }
  // Whether the cursor is on the first key tile of a segment.
  __device__ __forceinline__ bool segment_start() const {
    return key % segments.tiles == 0;
  }
  // Moves to the first key tile of the next query tile of any live batch.
  __device__ __forceinline__ void next_tile(
      const int32_t* q_len, const int32_t* prefix_len, int live, uint32_t window) {
    unit += keys - key + Prefill<T>::overhead;
    key = 0;
    if (++order == count) {
      order = 0;
      while (++batch < live && q_len[batch] <= 0) {}
      if (batch >= live) return;
      prefill_item<T>(q_len, prefix_len, batch, prefix, len, shift);
      count = (len + Prefill<T>::tokens - 1) / Prefill<T>::tokens;
    }
    load_tile(window);
  }
  // Moves to the first key tile of the first segment at or after the cursor.
  __device__ __forceinline__ void next_segment(
      const int32_t* q_len, const int32_t* prefix_len, int live, uint32_t window) {
    const int segment = (key + segments.tiles - 1) / segments.tiles;
    if (segment < segments.count) {
      unit += segment * segments.tiles - key;
      key = segment * segments.tiles;
    } else {
      next_tile(q_len, prefix_len, live, window);
    }
  }
  __device__ __forceinline__ void next(
      const int32_t* q_len, const int32_t* prefix_len, int live, uint32_t window) {
    if (key + 1 == keys) {
      next_tile(q_len, prefix_len, live, window);
    } else {
      ++key;
      ++unit;
    }
  }
};

// The first key tile at or after cost unit `unit`.
template<class T>
__device__ __forceinline__ PrefillCursor<T> prefill_locate(
    int64_t unit, const int32_t* q_len, const int32_t* prefix_len, int live,
    uint32_t window) {
  using P = Prefill<T>;
  PrefillCursor<T> cursor{0, live, 0, 0, 0, 0, 0, 0, 0, 0, 0, {1, 1}};
  int64_t base = 0;
  for (int b = 0; b < live; ++b) {
    if (q_len[b] <= 0) continue;
    int prefix, len, shift;
    prefill_item<T>(q_len, prefix_len, b, prefix, len, shift);
    const int count = (len + P::tokens - 1) / P::tokens;
    const int64_t all = prefill_units<T>(prefix, len, window, count);
    const int64_t cost = all + int64_t(P::overhead) * count;
    if (unit >= base + cost) {
      base += cost;
      continue;
    }
    int low = 0, high = count - 1;
    while (low < high) {
      const int middle = (low + high + 1) / 2;
      if (base + prefill_cost<T>(prefix, len, window, count, all, middle) <= unit)
        low = middle;
      else
        high = middle - 1;
    }
    const int64_t first = base + P::overhead +
        prefill_cost<T>(prefix, len, window, count, all, low);
    cursor = {first, b, low, count, 0, 0, 0, 0, prefix, len, shift, {1, 1}};
    cursor.load_tile(window);
    cursor.key = int(max(int64_t(0), unit - first));
    cursor.unit += cursor.key;
    return cursor;
  }
  return cursor;
}

// Query tile rows owned by one thread: rows g and g + 8 of its warp's m16 tile.
// Tile rows before `lead` (a paged item's rows of the cached prefix) and from
// `tokens` on are not the chunk's: they load no query and write nothing.
template<class T>
struct PrefillRows {
  static constexpr int group = Prefill<T>::tile_group;
  int token[2], head[2], pos[2], first[2], terms[2];
  int pos_lo, pos_hi, first_lo, first_hi, lead, tokens;
  bool live;

  // Row r holds a query head of the group, not padding.
  __device__ __forceinline__ bool real(int r) const {
    return group == T::group || head[r] < T::group;
  }
  // Row r is one of the chunk's.
  __device__ __forceinline__ bool active(int r) const {
    return token[r] >= lead && token[r] < tokens && real(r);
  }

  // `pos0` is the absolute position of tile row 0.
  __device__ __forceinline__ void load(int pos0, int tile_lead, int tile_tokens,
                                       bool windowed, int window) {
    const int warp = threadIdx.x / 32;
    const int g = threadIdx.x % 32 / 4;
    lead = tile_lead;
    tokens = tile_tokens;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int row = warp * 16 + g + r * 8;
      token[r] = row / group;
      head[r] = row % group;
      pos[r] = pos0 + min(max(token[r], lead), tokens - 1);
      first[r] = windowed ? max(0, pos[r] - window) : 0;
      // The probability term count follows each row's own visible keys.
      terms[r] = probability_terms(min(pos[r], windowed ? window : INT32_MAX) + 1);
    }
    const int warp_first = warp * 16 / group;
    const int warp_last = (warp * 16 + 15) / group;
    live = warp_last >= lead && warp_first < tokens;
    pos_lo = pos0 + min(max(warp_first, lead), tokens - 1);
    pos_hi = pos0 + min(max(warp_last, lead), tokens - 1);
    first_lo = windowed ? max(0, pos_lo - window) : 0;
    first_hi = windowed ? max(0, pos_hi - window) : 0;
  }
};

// Paged layers divide segments of query tiles over persistent CTAs
// (PrefillPlan); ring layers give each CTA one query tile in each batch's
// reverse tile order. A row with one segment gets BF16 output; a longer
// row's segments write F32 [max, denominator, V numerator] partials in
// base-two logit units for the merge kernel. Each row carries its
// probability as as many BF16 terms as its own visible key count gives
// (probability_terms); a warp whose two tokens differ runs the larger count
// with the other row's extra terms zero.
template<class T>
__device__ void prefill(
    const __nv_bfloat16* q, const void* k_cache, const void* v_cache,
    const CUtensorMap* k_map, const CUtensorMap* v_map,
    const int32_t* table, const int32_t* slots, const int32_t* seq_lens,
    const int32_t* q_start, const int32_t* q_len, const int32_t* prefix_len,
    const int32_t* live_batch, const __nv_bfloat16* k_new,
    const __nv_bfloat16* v_new, float* workspace, uint32_t max_pages,
    uint64_t k_page_stride_bytes, uint64_t v_page_stride_bytes,
    uint32_t split_capacity, uint32_t split_target, uint32_t window_left,
    uint32_t scale_bits, uint32_t cap_bits, uint32_t k_scale_bits,
    uint32_t v_scale_bits, const __nv_bfloat16* sink, __nv_bfloat16* out) {
  using P = Prefill<T>;
  (void)seq_lens;
  (void)k_page_stride_bytes;
  (void)v_page_stride_bytes;
  (void)split_capacity;
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const int kv_head = blockIdx.y;
  const int live = *live_batch;
  const bool windowed = window_left != UINT32_MAX;
  const int window = windowed ? int(window_left) : 0;

  PrefillCursor<T> cursor{0, live, 0, 0, 0, 0, 0, 0, 0, 0, 0, {1, 1}};
  int64_t end = 0;
  if constexpr (T::ring) {
    const int work = blockIdx.x;
    for (int b = 0, base = 0; b < live && cursor.batch == live; ++b) {
      const int count = (max(q_len[b], 0) + P::tokens - 1) / P::tokens;
      if (work < base + count)
        cursor = {0, b, 0, count, base + count - 1 - work, 0, 0, 0, prefix_len[b], q_len[b], 0,
                  {1, 1}};
      base += count;
    }
    if (cursor.batch == live) return;
    cursor.load_tile(window_left);
    end = cursor.keys;
  } else {
    const PrefillPlan plan =
        prefill_plan<T>(q_len, prefix_len, live, split_target, window_left);
    if (int(blockIdx.x) >= plan.ctas) return;
    cursor = prefill_locate<T>(plan.start(blockIdx.x), q_len, prefix_len, live, window_left);
    if (!cursor.segment_start()) cursor.next_segment(q_len, prefix_len, live, window_left);
    end = plan.start(blockIdx.x + 1);
  }
  // A CTA starts the segments whose first key tile lies before its range
  // end and completes each one it starts.
  auto owns = [&](const PrefillCursor<T>& at) {
    return at.batch < live && (at.unit < end || !at.segment_start());
  };
  if (!owns(cursor)) return;

  const int lane = threadIdx.x % 32;
  const int c = lane % 4;
  __shared__ __align__(1024) uint8_t storage[P::smem];
  __shared__ __align__(8) uint64_t bar[P::stages];
  __shared__ PrefillCursor<T> ahead;
  const uint32_t phase = (ptx::shared_address(storage) / 128) % 8;
  uint8_t* raw = storage + (T::fp8 ? P::k_bytes + P::v_bytes : 0);
  auto k_tile = [&](int stage) {
    return storage + (T::fp8 ? 0 : stage * (P::k_bytes + P::v_bytes));
  };
  auto v_tile = [&](int stage) { return k_tile(stage) + P::k_bytes; };

  // Paged tiles arrive through TMA; FP8 tiles land unswizzled in a raw stage
  // and are widened to BF16 before use. Keys past the chunk's last token read
  // the last valid page and are masked.
  auto issue_tma = [&](const PrefillCursor<T>& at, int stage) {
    const int pos0 = at.key_first + at.key * P::keys;
    const int last = at.prefix + min(at.len, (at.tile + 1) * P::tokens) - 1;
    ptx::mbarrier_arrive_expect_tx(
        &bar[stage], P::keys * (T::qk + T::v) * (T::fp8 ? 1 : 2));
#pragma unroll
    for (int segment = 0; segment < P::keys / P::box; ++segment) {
      const int pos = pos0 + segment * P::box;
      const int page = table[uint64_t(at.batch) * max_pages + min(pos, last) / T::page];
      if constexpr (T::fp8) {
        uint8_t* stage_raw = raw + stage * P::raw_bytes;
        const int32_t coord[4] = {0, kv_head, pos % T::page, page};
        ptx::tma_load<4>(stage_raw + segment * P::box * T::qk, k_map, coord, &bar[stage]);
        ptx::tma_load<4>(stage_raw + P::keys * T::qk + segment * P::box * T::v,
                         v_map, coord, &bar[stage]);
      } else {
#pragma unroll
        for (int chunk = 0; chunk < T::qk / 64; ++chunk) {
          const int32_t coord[4] = {chunk * 64, kv_head, pos % T::page, page};
          ptx::tma_load<4>(k_tile(stage) + chunk * P::keys * 128 + segment * P::box * 128,
                           k_map, coord, &bar[stage]);
        }
#pragma unroll
        for (int chunk = 0; chunk < T::v / 64; ++chunk) {
          const int32_t coord[4] = {chunk * 64, kv_head, pos % T::page, page};
          ptx::tma_load<4>(v_tile(stage) + chunk * P::keys * 128 + segment * P::box * 128,
                           v_map, coord, &bar[stage]);
        }
      }
    }
  };

  // Ring tiles combine the cached prefix and the contiguous new chunk, so
  // every thread copies 16-byte sectors; keys past the chunk are zero filled.
  auto issue_ring = [&](int key, int stage) {
    const int pos0 = cursor.key_first + key * P::keys;
    const int last = cursor.prefix + min(cursor.len, (cursor.tile + 1) * P::tokens) - 1;
    const int slot = slots[cursor.batch];
    const int start = q_start[cursor.batch];
    auto copy = [&](uint8_t* dst, const __nv_bfloat16* cache,
                    const __nv_bfloat16* fresh, int dim, int i) {
      const int row = i / (dim / 8);
      const int sector = i % (dim / 8);
      const int pos = pos0 + row;
      const uint64_t token = pos >= cursor.prefix ?
          uint64_t(start + pos - cursor.prefix) * T::kv_heads + kv_head :
          (uint64_t(slot) * T::page + pos % T::page) * T::kv_heads + kv_head;
      const __nv_bfloat16* src =
          (pos >= cursor.prefix ? fresh : cache) + token * dim + sector * 8;
      cp_async_zfill(dst + tile_offset<P::keys>(row, sector * 8, phase),
                     pos <= last ? src : fresh, pos <= last);
    };
    for (int i = threadIdx.x; i < P::keys * T::qk / 8; i += P::threads)
      copy(k_tile(stage), static_cast<const __nv_bfloat16*>(k_cache), k_new, T::qk, i);
    for (int i = threadIdx.x; i < P::keys * T::v / 8; i += P::threads)
      copy(v_tile(stage), static_cast<const __nv_bfloat16*>(v_cache), v_new, T::v, i);
    cp_async_commit();
  };

  auto widen = [&](int stage) {
    const uint8_t* stage_raw = raw + stage * P::raw_bytes;
    widen_fp8<P::keys, T::qk, P::threads>(k_tile(0), stage_raw, phase);
    widen_fp8<P::keys, T::v, P::threads>(v_tile(0), stage_raw + P::keys * T::qk, phase);
  };

  if constexpr (T::ring) {
    issue_ring(0, 0);
    if (end > 1) issue_ring(1, 1);
  } else if (threadIdx.x == 0) {
    for (int stage = 0; stage < P::stages; ++stage) ptx::mbarrier_init(&bar[stage], 1);
    ptx::fence_mbarrier_init();
    ahead = cursor;
    for (int stage = 0; stage < P::stages && owns(ahead); ++stage) {
      issue_tma(ahead, stage);
      ahead.next(q_len, prefix_len, live, window_left);
    }
  }
  __syncthreads();
  // The chunk's Q rows of the query tile after `at`, fetched into L2 ahead
  // of use.
  auto prefetch_next_q = [&](PrefillCursor<T> at) {
    at.next_tile(q_len, prefix_len, live, window_left);
    if (!owns(at)) return;
    const int t0 = max(at.tile * P::tokens, at.shift);
    const int tokens = min((at.tile + 1) * P::tokens, at.len) - t0;
    const __nv_bfloat16* rows_q = q + (uint64_t(q_start[at.batch] + t0 - at.shift) * T::q_heads +
                                       kv_head * T::group) * T::qk;
    if constexpr (T::group == T::q_heads) {
      prefetch_l2(rows_q, tokens * T::q_heads * T::qk * 2);
    } else {
      for (int token = 0; token < tokens; ++token)
        prefetch_l2(rows_q + uint64_t(token) * T::q_heads * T::qk, T::group * T::qk * 2);
    }
  };

  const float scale = __uint_as_float(scale_bits) *
      (T::fp8 ? __uint_as_float(k_scale_bits) : 1.0f);
  const float cap = __uint_as_float(cap_bits);
  const float v_scale = T::fp8 ? __uint_as_float(v_scale_bits) : 1.0f;
  constexpr float log2e = 1.4426950408889634f;
  const float score_scale = T::softcap ? scale / cap : scale * log2e;
  const uint32_t* q_words = reinterpret_cast<const uint32_t*>(q);

  PrefillRows<T> rows;
  ptx::A4 query[T::qk / 16];
  float maximum[2];
  float denominator[2];
  ptx::D4 output[T::v / 8];
  // Packed index of tile row 0 (negative before a paged item's chunk).
  int packed = 0;
  int terms = 3;
  int pieces = 1;
  int piece = 0;

  auto accumulate = [&](auto terms_tag, const uint8_t* k_smem, const uint8_t* v_smem,
                        int key0) {
    constexpr int count = decltype(terms_tag)::value;
    ptx::D4 score[P::keys / 8];
#pragma unroll
    for (int n = 0; n < P::keys / 8; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) score[n].v[i] = 0.0f;
#pragma unroll
    for (int ks = 0; ks < T::qk / 16; ++ks)
#pragma unroll
      for (int np = 0; np < P::keys / 16; ++np) {
        const int row = np * 16 + (lane / 16) * 8 + lane % 8;
        const int dim = ks * 16 + ((lane / 8) % 2) * 8;
        uint32_t b[4];
        ptx::ldmatrix_b16<4>(b, k_smem + tile_offset<P::keys>(row, dim, phase));
        ptx::mma_bf16(score[2 * np], query[ks], ptx::B2{{b[0], b[1]}});
        ptx::mma_bf16(score[2 * np + 1], query[ks], ptx::B2{{b[2], b[3]}});
      }
#pragma unroll
    for (int n = 0; n < P::keys / 8; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float x = score[n].v[i] * score_scale;
        if constexpr (T::softcap) x = cap * tanhf(x) * log2e;
        score[n].v[i] = x;
      }
    if (key0 + P::keys - 1 > rows.pos_lo || key0 < rows.first_hi) {
#pragma unroll
      for (int n = 0; n < P::keys / 8; ++n)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int key = key0 + n * 8 + 2 * c + i % 2;
          const bool hidden = key > rows.pos[i / 2] || key < rows.first[i / 2];
          score[n].v[i] = hidden ? -INFINITY : score[n].v[i];
        }
    }
    float correction[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      float tile_max = -INFINITY;
#pragma unroll
      for (int n = 0; n < P::keys / 8; ++n)
        tile_max = fmaxf(tile_max, fmaxf(score[n].v[2 * r], score[n].v[2 * r + 1]));
      tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 1));
      tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 2));
      const float next = fmaxf(maximum[r], tile_max);
      const float base = next == -INFINITY ? 0.0f : next;
      correction[r] = exp2f(maximum[r] - base);
      maximum[r] = next;
      float sum = 0.0f;
#pragma unroll
      for (int n = 0; n < P::keys / 8; ++n)
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const float p = exp2f(score[n].v[2 * r + i] - base);
          score[n].v[2 * r + i] = p;
          sum += p;
        }
      denominator[r] = denominator[r] * correction[r] + sum;
    }
    // Row maxima settle quickly, so most tiles leave O unscaled. The 256-wide
    // V accumulator leaves no registers for the branch.
    if (T::v > 128 ||
        __any_sync(0xffffffff, correction[0] != 1.0f || correction[1] != 1.0f)) {
#pragma unroll
      for (int n = 0; n < T::v / 8; ++n) {
        output[n].v[0] *= correction[0];
        output[n].v[1] *= correction[0];
        output[n].v[2] *= correction[1];
        output[n].v[3] *= correction[1];
      }
    }
#pragma unroll
    for (int kk = 0; kk < P::keys / 16; ++kk) {
      ptx::A4 probability[count];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const ptx::D4& s = score[2 * kk + i / 2];
        float p0 = s.v[(i % 2) * 2];
        float p1 = s.v[(i % 2) * 2 + 1];
#pragma unroll
        for (int term = 0; term < count; ++term) {
          const uint32_t pair = bf16_pair(p0, p1);
          probability[term].v[i] = term < rows.terms[i % 2] ? pair : 0u;
          p0 -= bf16_low(pair);
          p1 -= bf16_high(pair);
        }
      }
#pragma unroll
      for (int vn = 0; vn < T::v / 16; ++vn) {
        const int row = kk * 16 + ((lane / 8) % 2) * 8 + lane % 8;
        const int dim = vn * 16 + (lane / 16) * 8;
        uint32_t b[4];
        ptx::ldmatrix_b16<4, true>(b, v_smem + tile_offset<P::keys>(row, dim, phase));
#pragma unroll
        for (int term = 0; term < count; ++term) {
          ptx::mma_bf16(output[2 * vn], probability[term], ptx::B2{{b[0], b[1]}});
          ptx::mma_bf16(output[2 * vn + 1], probability[term], ptx::B2{{b[2], b[3]}});
        }
      }
    }
  };

  for (int64_t t = 0; owns(cursor); ++t) {
    const int stage = int(t % P::stages);
    if (t == 0 || cursor.key == 0) {
      // A new query tile: load its Q fragments.
      const int t0 = cursor.tile * P::tokens;
      rows.load(cursor.prefix + t0, cursor.shift - t0, min(P::tokens, cursor.len - t0),
                windowed, window);
      packed = q_start[cursor.batch] + t0 - cursor.shift;
      terms = __reduce_max_sync(0xffffffff, max(rows.terms[0], rows.terms[1]));
      if (!T::ring && threadIdx.x == 0) prefetch_next_q(cursor);
      uint64_t q_row[2];
#pragma unroll
      for (int r = 0; r < 2; ++r)
        q_row[r] = ((uint64_t(packed) + rows.token[r]) * T::q_heads +
                    kv_head * T::group + rows.head[r]) * (T::qk / 2);
#pragma unroll
      for (int ks = 0; ks < T::qk / 16; ++ks)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int r = i % 2;
          query[ks].v[i] = rows.active(r) ? q_words[q_row[r] + ks * 8 + c + (i / 2) * 4] : 0u;
        }
    }
    if (t == 0 || cursor.segment_start()) {
      // A new segment: reset the softmax state.
      pieces = cursor.segments.count;
      piece = cursor.key / cursor.segments.tiles;
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        maximum[r] = -INFINITY;
        denominator[r] = 0.0f;
      }
#pragma unroll
      for (int n = 0; n < T::v / 8; ++n)
#pragma unroll
        for (int i = 0; i < 4; ++i) output[n].v[i] = 0.0f;
    }

    if constexpr (T::ring) {
      if (t + 1 < end) cp_async_wait<1>(); else cp_async_wait<0>();
      __syncthreads();
    } else {
      ptx::mbarrier_wait(&bar[stage], int(t / P::stages) & 1);
      if constexpr (T::fp8) {
        widen(stage);
        __syncthreads();
        if (threadIdx.x == 0 && owns(ahead)) {
          issue_tma(ahead, stage);
          ahead.next(q_len, prefix_len, live, window_left);
        }
      }
    }
    const int key0 = cursor.key_first + cursor.key * P::keys;
    if (rows.live && key0 <= rows.pos_hi && key0 + P::keys > rows.first_lo) {
      const uint8_t* k_smem = k_tile(T::fp8 ? 0 : stage);
      const uint8_t* v_smem = v_tile(T::fp8 ? 0 : stage);
      if (terms == 1) {
        accumulate(std::integral_constant<int, 1>{}, k_smem, v_smem, key0);
      } else if (terms == 2) {
        accumulate(std::integral_constant<int, 2>{}, k_smem, v_smem, key0);
      } else {
        accumulate(std::integral_constant<int, 3>{}, k_smem, v_smem, key0);
      }
    }
    __syncthreads();
    if constexpr (T::ring) {
      if (t + P::stages < end) issue_ring(int(t) + P::stages, stage);
    } else if constexpr (!T::fp8) {
      if (threadIdx.x == 0 && owns(ahead)) {
        issue_tma(ahead, stage);
        ahead.next(q_len, prefix_len, live, window_left);
      }
    }

    if (cursor.key + 1 == cursor.keys || (cursor.key + 1) % cursor.segments.tiles == 0) {
      // The segment is complete.
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        float total = denominator[r];
        total += __shfl_xor_sync(0xffffffff, total, 1);
        total += __shfl_xor_sync(0xffffffff, total, 2);
        if (!rows.active(r)) continue;
        const uint64_t token = uint64_t(packed) + rows.token[r];
        const int q_head = kv_head * T::group + rows.head[r];
        if (pieces == 1) {
          const float sink_logit =
              T::sink ? __bfloat162float(sink[q_head]) * log2e : -INFINITY;
          const float final_max = fmaxf(maximum[r], sink_logit);
          const float weight = exp2f(maximum[r] - final_max);
          const float normalizer = weight * v_scale /
              (total * weight + (T::sink ? exp2f(sink_logit - final_max) : 0.0f));
          uint32_t* destination = reinterpret_cast<uint32_t*>(out) +
              (token * T::q_heads + q_head) * (T::v / 2);
#pragma unroll
          for (int n = 0; n < T::v / 8; ++n)
            destination[n * 4 + c] = bf16_pair(output[n].v[2 * r] * normalizer,
                                               output[n].v[2 * r + 1] * normalizer);
        } else {
          float* destination = workspace +
              (((token * T::kv_heads + kv_head) * P::max_pieces + piece) * T::group +
               rows.head[r]) * (T::v + 2);
          if (c == 0) {
            destination[0] = maximum[r];
            destination[1] = total;
          }
#pragma unroll
          for (int n = 0; n < T::v / 8; ++n)
            *reinterpret_cast<float2*>(destination + 2 + n * 8 + 2 * c) = make_float2(
                output[n].v[2 * r] * v_scale, output[n].v[2 * r + 1] * v_scale);
        }
      }
    }
    cursor.next(q_len, prefix_len, live, window_left);
  }
}

// Merge CTA j combines the paged query tiles j, j + gridDim.x, ... of the
// step (in batch and tile order) whose rows have two or more segments,
// folding each row's segments in order and inserting the optional sink
// logit. Rows with one segment were written by the partial kernel. Each
// warp combines whole rows of grouped Q heads with lanes on adjacent V
// columns, so partial loads are contiguous.
template<class T, int Threads>
__device__ void prefill_merge(
    const float* workspace, const __nv_bfloat16* sink, const int32_t* q_start,
    const int32_t* q_len, const int32_t* prefix_len, const int32_t* live_batch,
    __nv_bfloat16* out, uint32_t split_capacity, uint32_t split_target,
    uint32_t window_left) {
  using P = Prefill<T>;
  constexpr int warps = Threads / 32;
  constexpr int pairs = (T::v + 63) / 64;
  constexpr int batch_rows = 2;
  (void)split_target;
  (void)window_left;
  if (T::ring || split_capacity == 1) return;
  const int kv_head = blockIdx.y;
  const int lane = threadIdx.x % 32;
  const uint64_t stride = uint64_t(T::group) * (T::v + 2);
  constexpr float log2e = 1.4426950408889634f;
  const int live = *live_batch;
  const int grid = gridDim.x;
  // Rows [first_token, first_token + tokens) of the step with `pieces`
  // segments each.
  auto merge_tile = [&](uint64_t first_token, int tokens, int pieces) {
    const int rows = tokens * T::group;
    for (int first = threadIdx.x / 32 * batch_rows; first < rows; first += warps * batch_rows) {
      float2 value[batch_rows][P::max_pieces][pairs];
      float weight[batch_rows][P::max_pieces];
      float denominator[batch_rows];
#pragma unroll
      for (int r = 0; r < batch_rows; ++r) {
        const int row = min(first + r, rows - 1);
        const uint64_t packed = first_token + row / T::group;
        const int h = row % T::group;
        const float* base = workspace +
            ((packed * T::kv_heads + kv_head) * P::max_pieces * T::group + h) * (T::v + 2);
        const float sink_logit = T::sink ?
            __bfloat162float(sink[kv_head * T::group + h]) * log2e : -INFINITY;
        float maximum = sink_logit;
#pragma unroll
        for (int s = 0; s < P::max_pieces; ++s) {
          weight[r][s] = s < pieces ? base[s * stride] : -INFINITY;
          maximum = fmaxf(maximum, weight[r][s]);
#pragma unroll
          for (int k = 0; k < pairs; ++k) {
            const int column = k * 64 + 2 * lane;
            value[r][s][k] = s < pieces && column < T::v ?
                *reinterpret_cast<const float2*>(base + s * stride + 2 + column) :
                make_float2(0.0f, 0.0f);
          }
        }
        denominator[r] = T::sink ? exp2f(sink_logit - maximum) : 0.0f;
#pragma unroll
        for (int s = 0; s < P::max_pieces; ++s) {
          weight[r][s] = exp2f(weight[r][s] - maximum);
          if (s < pieces) denominator[r] += base[s * stride + 1] * weight[r][s];
        }
      }
#pragma unroll
      for (int r = 0; r < batch_rows; ++r) {
        const int row = first + r;
        if (row >= rows) break;
        const uint64_t packed = first_token + row / T::group;
        uint32_t* destination = reinterpret_cast<uint32_t*>(out) +
            (packed * T::q_heads + kv_head * T::group + row % T::group) * (T::v / 2);
#pragma unroll
        for (int k = 0; k < pairs; ++k) {
          const int column = k * 64 + 2 * lane;
          if (column >= T::v) continue;
          float2 sum = make_float2(0.0f, 0.0f);
#pragma unroll
          for (int s = 0; s < P::max_pieces; ++s) {
            sum.x += value[r][s][k].x * weight[r][s];
            sum.y += value[r][s][k].y * weight[r][s];
          }
          destination[column / 2] =
              bf16_pair(sum.x / denominator[r], sum.y / denominator[r]);
        }
      }
    }
  };
  int first_tile = 0;
  for (int b = 0; b < live; ++b) {
    if (q_len[b] <= 0) continue;
    int base, span, shift;
    prefill_item<T>(q_len, prefix_len, b, base, span, shift);
    const int count = (span + P::tokens - 1) / P::tokens;
    for (int tile = (int(blockIdx.x) - first_tile % grid + grid) % grid; tile < count;
         tile += grid) {
      const PrefillSegments segments = prefill_segments<T>(base + tile * P::tokens);
      if (segments.count < 2) continue;
      const int t0 = max(tile * P::tokens, shift);
      const int t1 = min((tile + 1) * P::tokens, span);
      merge_tile(uint64_t(q_start[b]) + t0 - shift, t1 - t0, segments.count);
    }
    first_tile += count;
  }
}

} // namespace chop::attention
