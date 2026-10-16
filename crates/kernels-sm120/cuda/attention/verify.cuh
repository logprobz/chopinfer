#pragma once

// Multi-query decode ("verify") attention. Each batch item has up to Rows
// query tokens at positions prefix .. prefix + count - 1 (a verify step's
// current token and drafts, or a DFlash draft block). The item's keys run
// from the earliest position its first token's window reaches to its last
// token. Keys before the prefix are cached; the item's own rows come from
// the cache for paged layouts, which the row engine writes before
// attention, and from the contiguous `k_new`/`v_new` rows for ring layouts,
// which it writes after attention. Causal items show token t the keys up to
// prefix + t; bidirectional items (DFlash) show every token the whole item.
// A window bounds each token's keys from below at its own position minus
// `window_left`.
//
// The kernel is decode's row-local split: every live item's keys form units
// of `keys` positions aligned to absolute positions, and an item of u units
// is cut into pieces of `piece_units` units from its first unit, so its
// piece bounds depend on its own length and the bucket only. The step's
// units are divided evenly over persistent CTAs per KV head; a CTA computes
// every piece that starts in its share, each to its end from a fresh
// softmax state, streaming its units through two cp.async stages. Its warps
// each own 16 of the item's Rows * group grouped query rows (the MMA M
// dimension) and all V columns, with one online softmax per row, so K and V
// are read once per piece for all of the item's tokens. Each piece writes
// one F32 partial and `verify_merge` folds an item's pieces, counted from
// its own length, with the sink. With one split slot an item is one piece,
// and its CTA writes BF16 output with the sink directly. An item's output
// bits therefore do not depend on the other items of its step.

#include "decode.cuh"

namespace chop::attention {

template<class T, int Rows>
struct Verify {
  static constexpr int rows = Rows * T::group;
  static_assert(rows % 16 == 0 && rows <= 128, "an item's grouped rows fill whole m16 tiles");
  static constexpr int warps = rows / 16;
  static constexpr int threads = warps * 32;
  static constexpr int keys = T::qk + T::v <= 320 ? 32 : 16;
  // Wide heads read Q fragments from shared memory at each k-step: the
  // 256-wide V accumulator leaves no registers to hold them.
  static constexpr bool q_in_smem = T::qk + T::v > 320;
  static constexpr int stages = 2;
  static constexpr int k_bytes = keys * T::qk * 2;
  static constexpr int v_bytes = keys * T::v * 2;
  static constexpr int raw_bytes = keys * (T::qk + T::v);
  static constexpr int stage_bytes = T::fp8 ? stages * raw_bytes + k_bytes + v_bytes :
                                              stages * (k_bytes + v_bytes);
  static_assert(stage_bytes % 1024 == 0, "Q follows the stages at a swizzle boundary");
  static constexpr int q_bytes = rows * T::qk * 2;
  // The item table follows Q in the partial entry's dynamic shared memory,
  // which holds as many items as the launch's size leaves room for; the
  // entry has no static shared memory.
  static constexpr int table_offset = stage_bytes + q_bytes;
};

// An item's first cached key is its prefix; `start` is its first packed
// query token and `count` its token count.
struct VerifyItem {
  int prefix, start, count;
};

// One item of a verify step's shared table: its key range [first, last],
// its descriptor, and the step's units and query tokens through it (prefix
// sums in item order). 28 bytes, as the separate row and item tables were.
struct VerifyEntry {
  int first, last, unit0, units_end;
  int prefix, start, tokens_end;
};

// A verify step's items for the cursor, from their shared table.
struct VerifyRows {
  const VerifyEntry* entries;
  int live;
  __device__ __forceinline__ int end(int row) const { return entries[row].units_end; }
  // The step's query tokens before the item's, in item order.
  __device__ __forceinline__ int first_token(int row) const {
    return row == 0 ? 0 : entries[row - 1].tokens_end;
  }
  __device__ __forceinline__ DecodeRow geometry(int row) const {
    const VerifyEntry& entry = entries[row];
    return {entry.first, entry.last, entry.unit0,
            entry.units_end - int(row_first_unit(*this, row))};
  }
  __device__ __forceinline__ VerifyItem item(int row) const {
    const VerifyEntry& entry = entries[row];
    return {entry.prefix, entry.start, entry.tokens_end - first_token(row)};
  }
};

// Stages every live item's key range (as DecodeRow units), descriptor and
// prefix sums in shared memory with one parallel scan of the step inputs,
// with `scratch` for one int2 per warp.
template<class T, int Rows>
__device__ __forceinline__ StepTotals load_verify_items(
    VerifyEntry* entries, void* scratch, int live, const int32_t* q_start,
    const int32_t* q_len, const int32_t* prefix_len, uint32_t window) {
  constexpr int keys = Verify<T, Rows>::keys;
  return scan_rows(
      static_cast<int2*>(scratch), live,
      [&](int i) {
        const int prefix = prefix_len[i];
        const int count = min(max(q_len[i], 1), Rows);
        const int last = prefix + count - 1;
        const int first = window == UINT32_MAX ? 0 : max(0, prefix - int(window));
        VerifyEntry& entry = entries[i];
        entry.first = first;
        entry.last = last;
        entry.unit0 = first / keys;
        entry.prefix = prefix;
        entry.start = q_start[i];
        return make_int2(last / keys - first / keys + 1, count);
      },
      [&](int i, int2 sums) {
        entries[i].units_end = sums.x;
        entries[i].tokens_end = sums.y;
      });
}

// Most query tokens of one verify step: the partial workspace and the merge
// grid hold this many token rows (MAX_VERIFY_TOKENS in the planner).
constexpr int max_verify_tokens = 256;

// Partials are F32 [max, denominator, V numerator] per query token (in item
// order over the step), KV head, piece slot and grouped head, with the
// maximum in base-two logit units. Packing by token keeps the workspace at
// the step's token capacity however many items share it.
template<class T>
__device__ __forceinline__ uint64_t verify_partial(int token, int kv_head, uint32_t slots,
                                                   int piece, int h) {
  return (((uint64_t(token) * T::kv_heads + kv_head) * slots + piece) * T::group + h) *
      (T::v + 2);
}

template<class T, int Rows, bool Causal>
__device__ void verify(
    const __nv_bfloat16* q, const void* k_cache, const void* v_cache,
    const int32_t* table, const int32_t* slots, const int32_t* q_start,
    const int32_t* q_len, const int32_t* prefix_len, const int32_t* live_batch,
    const __nv_bfloat16* k_new, const __nv_bfloat16* v_new, float* workspace,
    uint32_t max_pages, uint64_t k_page_stride_bytes, uint64_t v_page_stride_bytes,
    uint32_t split_capacity, uint32_t split_target, uint32_t split_units,
    uint32_t window_left, uint32_t scale_bits, uint32_t cap_bits, uint32_t k_scale_bits,
    uint32_t v_scale_bits, const __nv_bfloat16* sink, __nv_bfloat16* out) {
  using D = Verify<T, Rows>;
  static_assert(!T::ring || !T::fp8, "ring caches hold BF16");
  ptx::griddep_launch_dependents();
  // Until griddepcontrol.wait the CTA overlaps its producers: it reads the
  // step inputs, which the caller uploads with stream-ordered copies before
  // the step's kernels, plans its units and copies cached keys before each
  // item's prefix. Q and the item's own rows come from the producers.
  const int kv_head = blockIdx.y;
  extern __shared__ __align__(1024) uint8_t verify_smem[];
  uint8_t* const storage = verify_smem;
  uint8_t* const q_smem = verify_smem + D::stage_bytes;
  VerifyEntry* const entries = reinterpret_cast<VerifyEntry*>(verify_smem + D::table_offset);
  // The table holds the items the launch's dynamic shared memory has room for.
  uint32_t dynamic_bytes;
  asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dynamic_bytes));
  const int capacity =
      min(max_decode_rows, (int(dynamic_bytes) - D::table_offset) / int(sizeof(VerifyEntry)));
  const int live = live_rows(live_batch, capacity);
  // The stages are free until the plan's first copy: the scan works in them.
  const StepTotals totals =
      load_verify_items<T, Rows>(entries, storage, live, q_start, q_len, prefix_len, window_left);
  // The workspace holds at most this many token rows: a step above it must
  // fail loudly before any partial lands past it.
  if (totals.tokens > min(capacity * Rows, max_verify_tokens)) __trap();
  const DecodePlan plan = decode_piece_plan(totals, split_target);
  const VerifyRows rows{entries, live};
  // One split slot: each item is one piece, written as BF16 output.
  const bool direct = split_capacity == 1;
  const int cta = blockIdx.x;
  if (cta >= plan.ctas) return;
  const int64_t end = plan.start(cta + 1);
  // Units of each piece of the item at `at`.
  auto piece_of = [&](const DecodeCursor& at) {
    return piece_units(at.geometry.units, split_capacity, split_target, split_units);
  };
  // Whether `at` continues a piece its CTA started: such a unit is computed
  // even past the CTA's range end.
  auto continues = [&](const DecodeCursor& at) { return at.key % piece_of(at) != 0; };
  // A CTA starts at the first piece that begins in its range.
  DecodeCursor cursor = decode_locate(plan.start(cta), rows);
  if (cursor.row < live && continues(cursor)) {
    const int next = (cursor.key / piece_of(cursor) + 1) * piece_of(cursor);
    if (next >= cursor.geometry.units) {
      decode_next_row(cursor, rows);
    } else {
      cursor.unit += next - cursor.key;
      cursor.key = next;
    }
  }
  // A CTA stops at its range end; a started piece completes.
  auto owns = [&](const DecodeCursor& at) {
    return at.row < live && (at.unit < end || continues(at));
  };
  if (!owns(cursor)) return;

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int g = lane / 4;
  const int c = lane % 4;
  constexpr int bytes = T::fp8 ? 1 : 2;
  const uint32_t phase = (ptx::shared_address(storage) / 128) % 8;
  const uint32_t q_phase = (ptx::shared_address(q_smem) / 128) % 8;
  uint8_t* raw = storage + (T::fp8 ? D::k_bytes + D::v_bytes : 0);
  auto k_tile = [&](int stage) {
    return storage + (T::fp8 ? 0 : stage * (D::k_bytes + D::v_bytes));
  };
  auto v_tile = [&](int stage) { return k_tile(stage) + D::k_bytes; };

  // Every thread copies 16-byte sectors of one unit's K and V rows; keys
  // outside the item's range are zero filled. `fresh` selects keys before
  // the prefix (0), the item's own rows (1), or both (2).
  auto issue = [&](const DecodeCursor& at, int stage, int fresh = 2) {
    const VerifyItem item = rows.item(at.row);
    const int pos0 = (at.geometry.unit0 + at.key) * D::keys;
    auto copy = [&](const void* cache, const __nv_bfloat16* rows_new, uint64_t stride,
                    uint8_t* dst, int dim, bool fp8_raw) {
      constexpr int sector = 16;
      const int sectors = dim * bytes / sector;
      for (int i = threadIdx.x; i < D::keys * sectors; i += D::threads) {
        const int key = i / sectors;
        const int part = i % sectors;
        const int pos = pos0 + key;
        const bool own = pos >= item.prefix;
        if (fresh != 2 && own != (fresh == 1)) continue;
        const bool valid = pos >= at.geometry.first && pos <= at.geometry.last;
        const uint8_t* source;
        if (T::ring && own && valid) {
          source = reinterpret_cast<const uint8_t*>(rows_new) +
              ((uint64_t(item.start) + (pos - item.prefix)) * T::kv_heads + kv_head) *
                  dim * 2;
        } else {
          source = static_cast<const uint8_t*>(cache) +
              cache_byte_offset<T>(table, slots, at.row, valid ? pos : at.geometry.last,
                                   kv_head, max_pages, dim, bytes, stride);
        }
        uint8_t* target = fp8_raw ? dst + key * dim + part * sector :
            dst + tile_offset<D::keys>(key, part * 8, phase);
        cp_async_zfill(target, source + part * sector, valid);
      }
    };
    if constexpr (T::fp8) {
      uint8_t* stage_raw = raw + stage * D::raw_bytes;
      copy(k_cache, k_new, k_page_stride_bytes, stage_raw, T::qk, true);
      copy(v_cache, v_new, v_page_stride_bytes, stage_raw + D::keys * T::qk, T::v, true);
    } else {
      copy(k_cache, k_new, k_page_stride_bytes, k_tile(stage), T::qk, false);
      copy(v_cache, v_new, v_page_stride_bytes, v_tile(stage), T::v, false);
    }
  };
  // The item's query rows, zero past its token count.
  auto stage_query = [&]() {
    const VerifyItem item = rows.item(cursor.row);
    const uint8_t* source = reinterpret_cast<const uint8_t*>(q);
    for (int i = threadIdx.x; i < D::rows * T::qk / 8; i += D::threads) {
      const int row = i / (T::qk / 8);
      const int part = i % (T::qk / 8);
      const int token = row / T::group;
      const bool valid = token < item.count;
      const uint64_t word = valid ?
          ((uint64_t(item.start) + token) * T::q_heads + kv_head * T::group +
           row % T::group) * T::qk * 2 : 0;
      cp_async_zfill(q_smem + tile_offset<D::rows>(row, part * 8, q_phase),
                     source + word + part * 16, valid);
    }
  };

  const float scale = __uint_as_float(scale_bits) *
      (T::fp8 ? __uint_as_float(k_scale_bits) : 1.0f);
  const float cap = __uint_as_float(cap_bits);
  const float v_scale = T::fp8 ? __uint_as_float(v_scale_bits) : 1.0f;
  constexpr float log2e = 1.4426950408889634f;
  const float score_scale = T::softcap ? scale / cap : scale * log2e;

  // This lane's two rows of the warp's m16 tile: 16 warp + g and 8 further.
  ptx::A4 query[D::q_in_smem ? 1 : T::qk / 16];
  float maximum[2];
  float denominator[2];
  ptx::D4 output[T::v / 8];
  int lo[2], hi[2];
  bool active = false;
  int terms = 3;
  int piece = 0;

  // A new piece: its item's key bounds and term count, its partial slot and
  // a fresh softmax state.
  auto begin_piece = [&]() {
    const VerifyItem item = rows.item(cursor.row);
    const DecodeRow& geometry = cursor.geometry;
    active = (16 * warp) / T::group < item.count;
    int fewest = 1 << 30;
    for (int t = 0; t < item.count; ++t) {
      const int top = Causal ? item.prefix + t : geometry.last;
      const int bottom = window_left == UINT32_MAX ? 0 :
          max(0, item.prefix + t - int(window_left));
      fewest = min(fewest, top - bottom + 1);
    }
    terms = probability_terms(fewest);
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int token = (16 * warp + g + 8 * r) / T::group;
      hi[r] = token < item.count ? (Causal ? item.prefix + token : geometry.last) : -1;
      lo[r] = window_left == UINT32_MAX ? 0 : item.prefix + token - int(window_left);
      maximum[r] = -INFINITY;
      denominator[r] = 0.0f;
    }
    piece = cursor.key / piece_of(cursor);
#pragma unroll
    for (int n = 0; n < T::v / 8; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) output[n].v[i] = 0.0f;
  };
  auto query_step = [&](int ks) {
    uint32_t a[4];
    ptx::ldmatrix_b16<4>(a, q_smem + tile_offset<D::rows>(
        16 * warp + ((lane / 8) % 2) * 8 + lane % 8, ks * 16 + (lane / 16) * 8, q_phase));
    return ptx::A4{{a[0], a[1], a[2], a[3]}};
  };
  auto read_query = [&]() {
    if constexpr (!D::q_in_smem) {
#pragma unroll
      for (int ks = 0; ks < T::qk / 16; ++ks) query[ks] = query_step(ks);
    }
  };

  DecodeCursor ahead = cursor;
#pragma unroll
  for (int stage = 0; stage + 1 < D::stages; ++stage) {
    if (owns(ahead)) issue(ahead, stage, 0);
    cp_async_commit();
  }
  begin_piece();
  ptx::griddep_wait();
  stage_query();
#pragma unroll
  for (int stage = 0; stage + 1 < D::stages; ++stage) {
    if (owns(ahead)) {
      issue(ahead, stage, 1);
      decode_next(ahead, rows);
    }
    cp_async_commit();
  }

  auto accumulate = [&](auto terms_tag, const uint8_t* k_smem, const uint8_t* v_smem,
                        int key0) {
    constexpr int count = decltype(terms_tag)::value;
    // Even and odd k-steps accumulate separately to halve the MMA chain.
    ptx::D4 score[D::keys / 8], odd[D::keys / 8];
#pragma unroll
    for (int n = 0; n < D::keys / 8; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) score[n].v[i] = odd[n].v[i] = 0.0f;
#pragma unroll
    for (int ks = 0; ks < T::qk / 16; ++ks) {
      const ptx::A4 q_step = D::q_in_smem ? query_step(ks) : query[D::q_in_smem ? 0 : ks];
#pragma unroll
      for (int np = 0; np < D::keys / 16; ++np) {
        uint32_t b[4];
        ptx::ldmatrix_b16<4>(b, k_smem + tile_offset<D::keys>(
            np * 16 + (lane / 16) * 8 + lane % 8, ks * 16 + ((lane / 8) % 2) * 8, phase));
        ptx::D4* sum = ks % 2 ? odd : score;
        ptx::mma_bf16(sum[2 * np], q_step, ptx::B2{{b[0], b[1]}});
        ptx::mma_bf16(sum[2 * np + 1], q_step, ptx::B2{{b[2], b[3]}});
      }
    }
#pragma unroll
    for (int n = 0; n < D::keys / 8; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float x = (score[n].v[i] + odd[n].v[i]) * score_scale;
        if constexpr (T::softcap) x = cap * tanhf(x) * log2e;
        const int key = key0 + n * 8 + 2 * c + i % 2;
        const int r = i / 2;
        score[n].v[i] = key < lo[r] || key > hi[r] ? -INFINITY : x;
      }
    float correction[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      float tile_max = -INFINITY;
#pragma unroll
      for (int n = 0; n < D::keys / 8; ++n)
        tile_max = fmaxf(tile_max, fmaxf(score[n].v[2 * r], score[n].v[2 * r + 1]));
      tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 1));
      tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 2));
      const float next = fmaxf(maximum[r], tile_max);
      const float base = next == -INFINITY ? 0.0f : next;
      correction[r] = exp2f(maximum[r] - base);
      maximum[r] = next;
      float sum = 0.0f;
#pragma unroll
      for (int n = 0; n < D::keys / 8; ++n)
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const float p = exp2f(score[n].v[2 * r + i] - base);
          score[n].v[2 * r + i] = p;
          sum += p;
        }
      denominator[r] = denominator[r] * correction[r] + sum;
    }
    if (__any_sync(0xffffffff, correction[0] != 1.0f || correction[1] != 1.0f)) {
#pragma unroll
      for (int n = 0; n < T::v / 8; ++n) {
        output[n].v[0] *= correction[0];
        output[n].v[1] *= correction[0];
        output[n].v[2] *= correction[1];
        output[n].v[3] *= correction[1];
      }
    }
#pragma unroll
    for (int kk = 0; kk < D::keys / 16; ++kk) {
      ptx::A4 probability[count];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const ptx::D4& s = score[2 * kk + i / 2];
        float p0 = s.v[(i % 2) * 2];
        float p1 = s.v[(i % 2) * 2 + 1];
#pragma unroll
        for (int term = 0; term < count; ++term) {
          const uint32_t pair = bf16_pair(p0, p1);
          probability[term].v[i] = pair;
          p0 -= bf16_low(pair);
          p1 -= bf16_high(pair);
        }
      }
#pragma unroll
      for (int vn = 0; vn < T::v / 16; ++vn) {
        uint32_t b[4];
        ptx::ldmatrix_b16<4, true>(b, v_smem + tile_offset<D::keys>(
            kk * 16 + ((lane / 8) % 2) * 8 + lane % 8, vn * 16 + (lane / 16) * 8, phase));
#pragma unroll
        for (int term = 0; term < count; ++term) {
          ptx::mma_bf16(output[2 * vn], probability[term], ptx::B2{{b[0], b[1]}});
          ptx::mma_bf16(output[2 * vn + 1], probability[term], ptx::B2{{b[2], b[3]}});
        }
      }
    }
  };

  for (int t = 0; owns(cursor); ++t) {
    const int stage = t % D::stages;
    const bool fresh_item = t > 0 && cursor.key == 0;
    if (t > 0 && !continues(cursor)) begin_piece();

    // Unit t has landed and every warp has left unit t - 1, whose stage now
    // takes unit t + stages - 1. A new item's Q replaces the previous one
    // once every warp has read that into registers.
    cp_async_wait<0>();
    __syncthreads();
    if (fresh_item) {
      stage_query();
      cp_async_commit();
      cp_async_wait<0>();
      __syncthreads();
    }
    if (t == 0 || fresh_item) read_query();
    if (owns(ahead)) {
      issue(ahead, (t + D::stages - 1) % D::stages);
      decode_next(ahead, rows);
    }
    cp_async_commit();
    if constexpr (T::fp8) {
      const uint8_t* stage_raw = raw + stage * D::raw_bytes;
      widen_fp8<D::keys, T::qk, D::threads>(k_tile(0), stage_raw, phase);
      widen_fp8<D::keys, T::v, D::threads>(v_tile(0), stage_raw + D::keys * T::qk, phase);
      __syncthreads();
    }
    const int key0 = (cursor.geometry.unit0 + cursor.key) * D::keys;
    const uint8_t* k_smem = k_tile(T::fp8 ? 0 : stage);
    const uint8_t* v_smem = v_tile(T::fp8 ? 0 : stage);
    if (active) {
      if (terms == 1) {
        accumulate(std::integral_constant<int, 1>{}, k_smem, v_smem, key0);
      } else if (terms == 2) {
        accumulate(std::integral_constant<int, 2>{}, k_smem, v_smem, key0);
      } else {
        accumulate(std::integral_constant<int, 3>{}, k_smem, v_smem, key0);
      }
    }

    if (active &&
        (cursor.key + 1 == cursor.geometry.units || (cursor.key + 1) % piece_of(cursor) == 0)) {
      // The piece is complete.
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        float total = denominator[r];
        total += __shfl_xor_sync(0xffffffff, total, 1);
        total += __shfl_xor_sync(0xffffffff, total, 2);
        const int row = 16 * warp + g + 8 * r;
        if (hi[r] < 0) continue;
        if (direct) {
          const VerifyItem item = rows.item(cursor.row);
          const int q_head = kv_head * T::group + row % T::group;
          float normalize = v_scale / total;
          if constexpr (T::sink) {
            // The sink joins the final maximum, as in prefill, so a sink far
            // above every key score cannot overflow the denominator.
            const float sink_logit = __bfloat162float(sink[q_head]) * log2e;
            const float final_max = fmaxf(maximum[r], sink_logit);
            const float weight = exp2f(maximum[r] - final_max);
            normalize = weight * v_scale /
                (total * weight + exp2f(sink_logit - final_max));
          }
          uint32_t* words = reinterpret_cast<uint32_t*>(
              out + ((uint64_t(item.start) + row / T::group) * T::q_heads + q_head) * T::v);
#pragma unroll
          for (int n = 0; n < T::v / 8; ++n)
            words[(n * 8 + 2 * c) / 2] = bf16_pair(output[n].v[2 * r] * normalize,
                                                  output[n].v[2 * r + 1] * normalize);
          continue;
        }
        float* destination = workspace +
            verify_partial<T>(rows.first_token(cursor.row) + row / T::group, kv_head,
                              split_capacity, piece, row % T::group);
        if (c == 0) {
          destination[0] = maximum[r];
          destination[1] = total;
        }
#pragma unroll
        for (int n = 0; n < T::v / 8; ++n)
          *reinterpret_cast<float2*>(destination + 2 + n * 8 + 2 * c) =
              make_float2(output[n].v[2 * r] * v_scale, output[n].v[2 * r + 1] * v_scale);
      }
    }
    decode_next(cursor, rows);
  }
  cp_async_wait<0>();
}

// One CTA per query token of the step (in item order), Q head and 64 V
// columns folds the token's piece partials, counted from its item's own
// length, with the sink into BF16 output at the token's packed row.
template<class T, int Rows, int MaxThreads>
__device__ void verify_merge(
    const float* workspace, const __nv_bfloat16* sink, const int32_t* q_start,
    const int32_t* q_len, const int32_t* prefix_len, const int32_t* live_batch,
    __nv_bfloat16* out, uint32_t split_capacity, uint32_t split_target,
    uint32_t split_units, uint32_t window_left) {
  const int token = blockIdx.z;
  __shared__ VerifyEntry entries[max_decode_rows];
  __shared__ int2 warp_sums[32];
  const int live = live_rows(live_batch, max_decode_rows);
  const StepTotals totals = load_verify_items<T, Rows>(entries, warp_sums, live, q_start, q_len,
                                                       prefix_len, window_left);
  // A step above the grid's tokens must fail loudly, never lose tokens.
  if (totals.tokens > int(gridDim.z)) __trap();
  if (token >= totals.tokens) return;
  const VerifyRows rows{entries, live};
  // The token's item: the first whose tokens end past it.
  int item = 0, high = live - 1;
  while (item < high) {
    const int middle = (item + high) / 2;
    if (entries[middle].tokens_end > token) {
      high = middle;
    } else {
      item = middle + 1;
    }
  }
  const int pieces =
      row_pieces(row_units(rows, item), split_capacity, split_target, split_units);
  const int q_head = blockIdx.y;
  constexpr float log2e = 1.4426950408889634f;
  const float sink_logit = T::sink ? __bfloat162float(sink[q_head]) * log2e : -INFINITY;
  ptx::griddep_wait();
  const int kv_head = q_head / T::group;
  const int position = entries[item].start + token - rows.first_token(item);
  merge_owners<T, MaxThreads>(
      workspace + verify_partial<T>(token, kv_head, split_capacity, 0, q_head % T::group),
      uint64_t(T::group) * (T::v + 2), pieces, sink_logit,
      out + (uint64_t(position) * T::q_heads + q_head) * T::v);
}

} // namespace chop::attention
