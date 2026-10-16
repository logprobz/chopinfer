#pragma once

// Stream-K split-KV decode. Every live row's visible keys are cut into units
// of `keys` positions aligned to absolute positions, and the units of all
// rows, in batch order, are divided evenly over the persistent CTAs of each
// KV head, so long and short rows share the GPU in proportion to their bytes.
// A CTA streams its units through two cp.async stages, issuing each unit's
// copy before computing the unit that precedes it. The keys are the MMA rows:
// warp w forms S^T = K Q^T for key tile w % tiles of 16 keys with the grouped
// Q heads as n8 columns, so no MMA row is padding, transposes P in registers,
// and accumulates O^T = V^T P^T for its share of the V columns, keeping an
// online softmax per key tile. Each CTA writes one F32 partial per row it
// touches, after folding its key tiles through shared memory; the merge
// combines the partials.

#include <math.h>
#include <type_traits>
#include "gated.cuh"
#include "tile.cuh"
#include "../common/rotary.cuh"

namespace chop::attention {

// RoPE and the KV write folded into decode. A folded kernel reads the packed
// QKV rows [row][q_heads * qk | kv_heads * qk | kv_heads * v] in place of
// rotated Q. It rotates NeoX pairs (d, d + Rotary / 2) of the first Rotary
// dimensions of Q and of the current token's K, scales V, rounds (or
// quantizes) both with the row engine's arithmetic, uses them for the current
// position and writes them to the cache. A decode row's current position is
// seq_lens - 1, which is the row engine's position for decode rows.
struct RopeArgs {
  void* k_cache;
  void* v_cache;
  float theta;
  float v_multiplier;
  const float* cos_sin = nullptr;
};

struct NoRope {
  static constexpr bool enabled = false, gated = false;
  using Args = RopeArgs;
};
template<int Rotary, bool Table = false>
struct NeoxRope {
  static constexpr bool enabled = true, gated = false;
  static constexpr int rotary = Rotary;
  static constexpr bool table = Table;
  using Args = RopeArgs;
};

// Gated attention folded into decode (decision 0084): the packed rows are
// gated.cuh's, each Q head's query followed by its gate. Q and the current K
// get the Gemma norm and NeoX M-RoPE of `gated::qk_norm_rope_kv`, with the
// same arithmetic, and V is written unscaled.
struct GatedRopeArgs {
  void* k_cache;
  void* v_cache;
  const uint16_t* q_norm;
  const uint16_t* k_norm;
  const int32_t* rope_positions;
  const float* cos_sin;
  float eps;
};
template<int Rotary, int SectionH, int SectionW>
struct GatedRope {
  static constexpr bool enabled = true, gated = true;
  static constexpr int rotary = Rotary, section_h = SectionH, section_w = SectionW;
  using Args = GatedRopeArgs;
};

// Elements from one Q head to the next, and from one row's Q to the next.
template<class T, class Fold>
struct QueryLayout {
  static constexpr int head = Fold::gated ? 2 * T::qk : T::qk;
  static constexpr int row =
      Fold::enabled ? T::q_heads * head + T::kv_heads * (T::qk + T::v) : T::q_heads * T::qk;
};

// Cosine and sine of rotary pair `lane` at `position`: a warp holds all of
// a position's pairs.
template<class R>
__device__ __forceinline__ float2 lane_angle(int position, const RopeArgs& rope) {
  static_assert(R::rotary == 64, "each lane holds one rotary pair");
  if constexpr (R::table) {
    return reinterpret_cast<const float2*>(rope.cos_sin)[uint64_t(position) * 32 + threadIdx.x % 32];
  } else {
    return rotary::angle<R::rotary>(threadIdx.x % 32, position, rope.theta);
  }
}

// Rotates Q^T B fragments in place: `low` holds dimensions d and d + 1 of a
// head and `high` dimensions d + Rotary / 2 and d + Rotary / 2 + 1. Every lane
// of the warp calls it; `cos_sin` is the lane's pair from lane_angle.
__device__ __forceinline__ void rotate_words(uint32_t& low, uint32_t& high, float2 cos_sin,
                                             int d) {
  auto pair = [&](int source) {
    return make_float2(__shfl_sync(0xffffffff, cos_sin.x, source),
                       __shfl_sync(0xffffffff, cos_sin.y, source));
  };
  const float2 first = pair(d);
  const float2 second = pair(d + 1);
  const float x0 = bf16_low(low), x1 = bf16_high(low);
  const float y0 = bf16_low(high), y1 = bf16_high(high);
  low = convert::bf16x2_from_f32(
      {rotary::hf_rotate(x0, y0, first, false), rotary::hf_rotate(x1, y1, second, false)});
  high = convert::bf16x2_from_f32(
      {rotary::hf_rotate(y0, x0, first, true), rotary::hf_rotate(y1, x1, second, true)});
}

// The current token's K and V row as one warp holds it: lane l has K
// dimensions l and l + 32 (a rotary pair), four K dimensions past the
// rotation and four V dimensions, plus the row's cache byte offsets.
template<class T, class R>
struct CurrentRow {
  static_assert(!R::gated, "MiMo BF16 rotation excludes gated Qwen");
  static_assert(R::rotary == 64 && T::qk - R::rotary == 128 && T::v == 128,
                "a lane holds one rotary pair and four other K and V dimensions");
  static constexpr int stride = T::q_heads * T::qk + T::kv_heads * (T::qk + T::v);
  uint32_t pair;
  uint2 k_rest, v_rest;
  uint64_t k_offset, v_offset;

  __device__ __forceinline__ void load(const __nv_bfloat16* qkv, int row, int kv_head,
                                       uint64_t k_offset_, uint64_t v_offset_) {
    const int lane = threadIdx.x % 32;
    const uint16_t* base = reinterpret_cast<const uint16_t*>(qkv) + uint64_t(row) * stride;
    const uint16_t* k = base + T::q_heads * T::qk + kv_head * T::qk;
    const uint16_t* v = base + (T::q_heads + T::kv_heads) * T::qk + kv_head * T::v;
    pair = uint32_t(k[lane]) | (uint32_t(k[lane + 32]) << 16);
    k_rest = *reinterpret_cast<const uint2*>(k + R::rotary + 4 * lane);
    v_rest = *reinterpret_cast<const uint2*>(v + 4 * lane);
    k_offset = k_offset_;
    v_offset = v_offset_;
  }

  // Writes the row as key `key` of a stage (BF16 tiles, or unswizzled FP8
  // [key][dim] rows) and to the cache; `cos_sin` is the lane's rotary pair.
  template<int Keys>
  __device__ __forceinline__ void store(uint8_t* k_stage, uint8_t* v_stage, uint32_t phase,
                                        int key, float2 cos_sin, const RopeArgs& rope,
                                        float k_scale, float v_scale) const {
    const int lane = threadIdx.x % 32;
    const float x = bf16_low(pair), y = bf16_high(pair);
    const float k_pair[2] = {rotary::hf_rotate(x, y, cos_sin, false),
                             rotary::hf_rotate(y, x, cos_sin, true)};
    float k[4], v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint32_t k_word = i < 2 ? k_rest.x : k_rest.y;
      const uint32_t v_word = i < 2 ? v_rest.x : v_rest.y;
      k[i] = i % 2 ? bf16_high(k_word) : bf16_low(k_word);
      v[i] = (i % 2 ? bf16_high(v_word) : bf16_low(v_word)) * rope.v_multiplier;
    }
    uint8_t* k_cache = static_cast<uint8_t*>(rope.k_cache) + k_offset;
    uint8_t* v_cache = static_cast<uint8_t*>(rope.v_cache) + v_offset;
    const int rest = R::rotary + 4 * lane;
    if constexpr (T::fp8) {
      auto code = [](float value, float scale) {
        return uint8_t(convert::pack_f32_e4m3({value / scale, 0.f}));
      };
      auto codes = [&](const float* values, float scale) {
        return uint32_t(code(values[0], scale)) | (uint32_t(code(values[1], scale)) << 8) |
               (uint32_t(code(values[2], scale)) << 16) | (uint32_t(code(values[3], scale)) << 24);
      };
      const uint8_t k0 = code(k_pair[0], k_scale), k1 = code(k_pair[1], k_scale);
      const uint32_t k_codes = codes(k, k_scale), v_codes = codes(v, v_scale);
      uint8_t* k_row = k_stage + key * T::qk;
      k_row[lane] = k_cache[lane] = k0;
      k_row[lane + 32] = k_cache[lane + 32] = k1;
      *reinterpret_cast<uint32_t*>(k_row + rest) = k_codes;
      *reinterpret_cast<uint32_t*>(k_cache + rest) = k_codes;
      *reinterpret_cast<uint32_t*>(v_stage + key * T::v + 4 * lane) = v_codes;
      *reinterpret_cast<uint32_t*>(v_cache + 4 * lane) = v_codes;
    } else {
      auto at = [&](uint8_t* stage, int d) {
        return stage + tile_offset<Keys>(key, d, phase) + (d % 8) * 2;
      };
      const uint16_t k0 = convert::bf16_from_f32(k_pair[0]);
      const uint16_t k1 = convert::bf16_from_f32(k_pair[1]);
      const uint2 k_bits = make_uint2(convert::bf16x2_from_f32({k[0], k[1]}),
                                      convert::bf16x2_from_f32({k[2], k[3]}));
      const uint2 v_bits = make_uint2(convert::bf16x2_from_f32({v[0], v[1]}),
                                      convert::bf16x2_from_f32({v[2], v[3]}));
      *reinterpret_cast<uint16_t*>(at(k_stage, lane)) = k0;
      *reinterpret_cast<uint16_t*>(at(k_stage, lane + 32)) = k1;
      *reinterpret_cast<uint16_t*>(k_cache + lane * 2) = k0;
      *reinterpret_cast<uint16_t*>(k_cache + (lane + 32) * 2) = k1;
      *reinterpret_cast<uint2*>(at(k_stage, rest)) = k_bits;
      *reinterpret_cast<uint2*>(k_cache + rest * 2) = k_bits;
      *reinterpret_cast<uint2*>(at(v_stage, 4 * lane)) = v_bits;
      *reinterpret_cast<uint2*>(v_cache + 4 * lane * 2) = v_bits;
    }
  }
};

// A gated fold's current K and V row as one warp holds it.
template<class T, class R, int Dim = T::qk>
struct GatedRow;

// Lane l has dimensions 8l to 8l + 7, as gated::qk_norm_rope_kv's K warp.
template<class T, class R>
struct GatedRow<T, R, 256> {
  static_assert(T::qk == 256 && T::v == 256, "a lane holds eight K and eight V dimensions");
  uint4 k_raw, v_raw;
  uint64_t k_offset, v_offset;

  __device__ __forceinline__ void load(const __nv_bfloat16* qkv, int row, int kv_head,
                                       uint64_t k_offset_, uint64_t v_offset_) {
    const int lane = threadIdx.x % 32;
    const uint16_t* base = reinterpret_cast<const uint16_t*>(qkv) +
        uint64_t(row) * QueryLayout<T, R>::row + T::q_heads * 2 * T::qk;
    k_raw = *reinterpret_cast<const uint4*>(base + kv_head * T::qk + 8 * lane);
    v_raw = *reinterpret_cast<const uint4*>(base + T::kv_heads * T::qk + kv_head * T::v +
                                            8 * lane);
    k_offset = k_offset_;
    v_offset = v_offset_;
  }

  // Norms and rotates K and writes the row as key `key` of a stage (BF16
  // tiles, or unswizzled FP8 [key][dim] rows) and to the cache; lane p's
  // `cos_sin` is rotary pair p of the row.
  template<int Keys>
  __device__ __forceinline__ void store(uint8_t* k_stage, uint8_t* v_stage, uint32_t phase,
                                        int key, float2 cos_sin, const GatedRopeArgs& rope,
                                        float k_scale, float v_scale) const {
    const int lane = threadIdx.x % 32;
    const uint32_t k_words[4] = {k_raw.x, k_raw.y, k_raw.z, k_raw.w};
    float k[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) k[i] = i % 2 ? bf16_high(k_words[i / 2]) : bf16_low(k_words[i / 2]);
    gated::norm_rotate_key<R::rotary>(k, rope.k_norm, rope.eps, [&](int pair) {
      return make_float2(__shfl_sync(0xffffffff, cos_sin.x, pair),
                         __shfl_sync(0xffffffff, cos_sin.y, pair));
    });
    uint8_t* k_cache = static_cast<uint8_t*>(rope.k_cache) + k_offset;
    uint8_t* v_cache = static_cast<uint8_t*>(rope.v_cache) + v_offset;
    if constexpr (T::fp8) {
      const uint32_t v_words[4] = {v_raw.x, v_raw.y, v_raw.z, v_raw.w};
      uint32_t k_codes[2] = {0, 0}, v_codes[2] = {0, 0};
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float v = i % 2 ? bf16_high(v_words[i / 2]) : bf16_low(v_words[i / 2]);
        k_codes[i / 4] |= uint32_t(gated::fp8_code(k[i], k_scale)) << (8 * (i % 4));
        v_codes[i / 4] |= uint32_t(gated::fp8_code(v, v_scale)) << (8 * (i % 4));
      }
      const uint2 k_bits = make_uint2(k_codes[0], k_codes[1]);
      const uint2 v_bits = make_uint2(v_codes[0], v_codes[1]);
      *reinterpret_cast<uint2*>(k_stage + key * T::qk + 8 * lane) = k_bits;
      *reinterpret_cast<uint2*>(k_cache + 8 * lane) = k_bits;
      *reinterpret_cast<uint2*>(v_stage + key * T::v + 8 * lane) = v_bits;
      *reinterpret_cast<uint2*>(v_cache + 8 * lane) = v_bits;
    } else {
      const uint4 k_bits = make_uint4(
          convert::bf16x2_from_f32({k[0], k[1]}), convert::bf16x2_from_f32({k[2], k[3]}),
          convert::bf16x2_from_f32({k[4], k[5]}), convert::bf16x2_from_f32({k[6], k[7]}));
      *reinterpret_cast<uint4*>(k_stage + tile_offset<Keys>(key, 8 * lane, phase)) = k_bits;
      *reinterpret_cast<uint4*>(k_cache + 16 * lane) = k_bits;
      *reinterpret_cast<uint4*>(v_stage + tile_offset<Keys>(key, 8 * lane, phase)) = v_raw;
      *reinterpret_cast<uint4*>(v_cache + 16 * lane) = v_raw;
    }
  }
};

// Head 128 holds four consecutive K and V dimensions per lane.
template<class T, class R>
struct GatedRow<T, R, 128> {
  static_assert(T::qk == 128 && T::v == 128, "a lane holds four K and four V dimensions");
  uint2 k_raw, v_raw;
  uint64_t k_offset, v_offset;

  __device__ __forceinline__ void load(const __nv_bfloat16* qkv, int row, int kv_head,
                                       uint64_t k_offset_, uint64_t v_offset_) {
    const int lane = threadIdx.x % 32;
    const uint16_t* base = reinterpret_cast<const uint16_t*>(qkv) +
        uint64_t(row) * QueryLayout<T, R>::row + T::q_heads * 2 * T::qk;
    k_raw = *reinterpret_cast<const uint2*>(base + kv_head * T::qk + 4 * lane);
    v_raw = *reinterpret_cast<const uint2*>(base + T::kv_heads * T::qk + kv_head * T::v +
                                            4 * lane);
    k_offset = k_offset_;
    v_offset = v_offset_;
  }

  template<int Keys>
  __device__ __forceinline__ void store(uint8_t* k_stage, uint8_t* v_stage, uint32_t phase,
                                        int key, float2 cos_sin, const GatedRopeArgs& rope,
                                        float k_scale, float v_scale) const {
    const int lane = threadIdx.x % 32;
    float k[4] = {bf16_low(k_raw.x), bf16_high(k_raw.x),
                  bf16_low(k_raw.y), bf16_high(k_raw.y)};
    gated::norm_rotate_key<T::qk, R::rotary>(k, rope.k_norm, rope.eps, [&](int pair) {
      return make_float2(__shfl_sync(0xffffffff, cos_sin.x, pair),
                         __shfl_sync(0xffffffff, cos_sin.y, pair));
    });
    uint8_t* k_cache = static_cast<uint8_t*>(rope.k_cache) + k_offset;
    uint8_t* v_cache = static_cast<uint8_t*>(rope.v_cache) + v_offset;
    if constexpr (T::fp8) {
      const uint32_t v_words[2] = {v_raw.x, v_raw.y};
      uint32_t k_codes = 0, v_codes = 0;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float v = i % 2 ? bf16_high(v_words[i / 2]) : bf16_low(v_words[i / 2]);
        k_codes |= uint32_t(gated::fp8_code(k[i], k_scale)) << (8 * i);
        v_codes |= uint32_t(gated::fp8_code(v, v_scale)) << (8 * i);
      }
      *reinterpret_cast<uint32_t*>(k_stage + key * T::qk + 4 * lane) = k_codes;
      *reinterpret_cast<uint32_t*>(k_cache + 4 * lane) = k_codes;
      *reinterpret_cast<uint32_t*>(v_stage + key * T::v + 4 * lane) = v_codes;
      *reinterpret_cast<uint32_t*>(v_cache + 4 * lane) = v_codes;
    } else {
      const uint2 k_bits = make_uint2(convert::bf16x2_from_f32({k[0], k[1]}),
                                      convert::bf16x2_from_f32({k[2], k[3]}));
      const int d = 4 * lane;
      *reinterpret_cast<uint2*>(k_stage + tile_offset<Keys>(key, d, phase) + (d % 8) * 2) = k_bits;
      *reinterpret_cast<uint2*>(k_cache + d * 2) = k_bits;
      *reinterpret_cast<uint2*>(v_stage + tile_offset<Keys>(key, d, phase) + (d % 8) * 2) = v_raw;
      *reinterpret_cast<uint2*>(v_cache + d * 2) = v_raw;
    }
  }
};

template<class T>
struct Decode {
  static constexpr int warps = 4;
  static constexpr int threads = warps * 32;
  static constexpr int keys = T::qk + T::v <= 320 ? 32 : 16;
  // Warp w owns key tile w % tiles and V column part w / tiles.
  static constexpr int tiles = keys / 16;
  static constexpr int parts = warps / tiles;
  static constexpr int columns = T::v / parts;
  static_assert(columns % 16 == 0, "each warp owns whole 16-column V blocks");
  static constexpr int stages = 2;
  static constexpr int k_bytes = keys * T::qk * 2;
  static constexpr int v_bytes = keys * T::v * 2;
  static constexpr int raw_bytes = keys * (T::qk + T::v);
  static constexpr int smem = T::fp8 ? stages * raw_bytes + k_bytes + v_bytes :
                                       stages * (k_bytes + v_bytes);
};

// Visible positions [first, last] of a decode row and its units.
struct DecodeRow {
  int first, last, unit0, units;
};

template<class T>
__device__ __forceinline__ DecodeRow decode_row(const int32_t* seq_lens, int row,
                                                uint32_t window) {
  constexpr int keys = Decode<T>::keys;
  const int last = seq_lens[row] - 1;
  const int first = window == UINT32_MAX ? 0 : max(0, last - int(window));
  return {first, last, first / keys, last / keys - first / keys + 1};
}

// Decode and verify steps hold at most this many rows (items). A CTA
// stages the prefix sums of the rows' units in shared memory with one
// parallel scan of the step inputs, and reads a row's geometry from the
// inputs when its cursor reaches it. A step above the tables traps rather
// than losing rows; planners refuse such buckets before any launch.
constexpr int max_decode_rows = 256;

// Totals of a step's scan: its units and its query tokens (verify).
struct StepTotals {
  int units, tokens;
};

// Inclusive prefix sums of `count(row)` (units and tokens) over rows
// [0, live), passed to `store(row, sums)`; every thread of the CTA calls it,
// and the stores are visible to all of them on return. Passes of blockDim.x
// rows scan within warps with shuffles and across warps through
// `warp_sums` (one per warp, in shared memory that is free until the scan
// returns), so a step of 256 rows costs a few barriers instead of a serial
// walk in every thread.
template<class Count, class Store>
__device__ __forceinline__ StepTotals scan_rows(int2* warp_sums, int live, Count count,
                                                Store store) {
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int warps = blockDim.x / 32;
  StepTotals totals{0, 0};
  for (int base = 0; base < live; base += blockDim.x) {
    const int row = base + threadIdx.x;
    const int2 own = row < live ? count(row) : make_int2(0, 0);
    int2 sum = own;
#pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
      const int units = __shfl_up_sync(0xffffffff, sum.x, offset);
      const int tokens = __shfl_up_sync(0xffffffff, sum.y, offset);
      if (lane >= offset) {
        sum.x += units;
        sum.y += tokens;
      }
    }
    if (lane == 31) warp_sums[warp] = sum;
    __syncthreads();
    int2 before = make_int2(totals.units, totals.tokens);
    for (int w = 0; w < warps; ++w) {
      const int2 part = warp_sums[w];
      if (w < warp) {
        before.x += part.x;
        before.y += part.y;
      }
      totals.units += part.x;
      totals.tokens += part.y;
    }
    if (row < live) store(row, make_int2(before.x + sum.x, before.y + sum.y));
    __syncthreads();
  }
  return totals;
}

// A decode step's rows for the cursor: `ends[r]` counts the units of rows
// [0, r], and each row's geometry comes from `seq_lens`.
template<class T>
struct DecodeRows {
  const int* ends;
  int live;
  const int32_t* seq_lens;
  uint32_t window;
  __device__ __forceinline__ int end(int row) const { return ends[row]; }
  __device__ __forceinline__ DecodeRow geometry(int row) const {
    return decode_row<T>(seq_lens, row, window);
  }
};

// Scans the live rows of a decode step into `ends`, with `scratch` for one
// int2 per warp.
template<class T>
__device__ __forceinline__ StepTotals load_decode_rows(
    int* ends, void* scratch, int live, const int32_t* seq_lens, uint32_t window) {
  return scan_rows(
      static_cast<int2*>(scratch), live,
      [&](int row) { return make_int2(decode_row<T>(seq_lens, row, window).units, 0); },
      [&](int row, int2 sums) { ends[row] = sums.x; });
}

// The live row count of a step, trapping above `capacity` rows.
__device__ __forceinline__ int live_rows(const int32_t* live_batch, int capacity) {
  const int live = *live_batch;
  if (live > capacity) __trap();
  return live;
}

// Exact verify rows: a verify step's token row run as the decode row it
// would be alone. Token `token` belongs to the item whose rows
// q_start .. q_start + q_len - 1 hold it, at position prefix + (token -
// q_start); `item` is -1 for a token past the step's rows.
struct VerifyRowsArgs {
  const int32_t* q_start = nullptr;
  const int32_t* q_len = nullptr;
  const int32_t* prefix_len = nullptr;
  const __nv_bfloat16* k_new = nullptr;
  const __nv_bfloat16* v_new = nullptr;
};

struct VerifyRow {
  int item, last, prefix, start;
};

__device__ __forceinline__ VerifyRow verify_row(const VerifyRowsArgs& rows,
                                                const int32_t* live_batch, int token) {
  const int items = live_rows(live_batch, max_decode_rows);
  for (int item = 0; item < items; ++item) {
    const int start = rows.q_start[item];
    if (token >= start && token < start + rows.q_len[item]) {
      const int prefix = rows.prefix_len[item];
      return {item, prefix + token - start, prefix, start};
    }
  }
  return {-1, 0, 0, 0};
}

// The global unit of a row's first unit, and its unit count.
template<class Rows>
__device__ __forceinline__ int64_t row_first_unit(const Rows& rows, int row) {
  return row == 0 ? 0 : rows.end(row - 1);
}

template<class Rows>
__device__ __forceinline__ int row_units(const Rows& rows, int row) {
  return rows.end(row) - int(row_first_unit(rows, row));
}

// A step's CTA ranges: CTA j of `ctas` covers units [start(j), start(j + 1)).
struct DecodePlan {
  int64_t units;
  int ctas;
  __device__ __forceinline__ int64_t start(int cta) const { return units * cta / ctas; }
};

// Position of one unit: its row, its index `key` among the row's units, and
// its global unit index.
struct DecodeCursor {
  int64_t unit;
  int row, key;
  DecodeRow geometry;
};

template<class Rows>
__device__ __forceinline__ void decode_next_row(DecodeCursor& cursor, const Rows& rows) {
  cursor.unit += cursor.geometry.units - cursor.key;
  cursor.key = 0;
  if (++cursor.row < rows.live) cursor.geometry = rows.geometry(cursor.row);
}

template<class Rows>
__device__ __forceinline__ void decode_next(DecodeCursor& cursor, const Rows& rows) {
  if (cursor.key + 1 == cursor.geometry.units) {
    decode_next_row(cursor, rows);
  } else {
    ++cursor.key;
    ++cursor.unit;
  }
}

// The cursor at `unit`: the first row whose units end past it, by binary
// search over the scanned ends.
template<class Rows>
__device__ __forceinline__ DecodeCursor decode_locate(int64_t unit, const Rows& rows) {
  int low = 0, high = rows.live;
  while (low < high) {
    const int middle = (low + high) / 2;
    if (rows.end(middle) > unit) {
      high = middle;
    } else {
      low = middle + 1;
    }
  }
  if (low == rows.live) return {unit, rows.live, 0, {0, 0, 0, 0}};
  return {unit, low, int(unit - row_first_unit(rows, low)), rows.geometry(low)};
}

// The workspace holds F32 [max, denominator, V numerator] partials per row,
// KV head, piece slot and grouped Q head, with the maximum in base-two logit
// units.
template<class T>
__device__ __forceinline__ uint64_t decode_partial(int row, int kv_head, uint32_t slots,
                                                   int piece, int h) {
  return (((uint64_t(row) * T::kv_heads + kv_head) * slots + piece) * T::group + h) *
      (T::v + 2);
}

// The row-local split of paged decode and verify: a row (verify: an item) of
// `units` units is cut into
// pieces of this many units from its first visible unit, at least the
// bucket's `split_units` and at most min(split slots, CTAs) pieces. The size
// depends on the row's own length and the bucket's constants only, so a
// row's partials, and its merged output, never depend on the other rows of
// its step.
__device__ __forceinline__ int piece_units(int units, uint32_t split_capacity,
                                           uint32_t split_target, uint32_t split_units) {
  const int most = int(min(split_capacity, split_target));
  return max(int(split_units), (units + most - 1) / most);
}

__device__ __forceinline__ int row_pieces(int units, uint32_t split_capacity,
                                          uint32_t split_target, uint32_t split_units) {
  const int piece = piece_units(units, split_capacity, split_target, split_units);
  return (units + piece - 1) / piece;
}

// A decode step's CTA ranges: the step's units cut evenly over the CTAs,
// each of which computes the pieces that start in its range, each to its
// end. Which CTA computes a piece does not change its bits.
__device__ __forceinline__ DecodePlan decode_piece_plan(const StepTotals& totals,
                                                       uint32_t split_target) {
  return {totals.units, int(max(int64_t(1), min(int64_t(split_target), int64_t(totals.units))))};
}

// With a NeoxRope or GatedRope `Fold`, `q` is the packed QKV rows and `rope`
// holds the writable caches; see NoRope and GatedRope. `Rows` (exact verify
// rows, `decode_rows`) also treats positions from `produced_from` on as
// written by the producers; the AR entries compile without it.
template<class T, class Fold = NoRope, bool Rows = false>
__device__ void decode(
    const __nv_bfloat16* q, const void* k_cache, const void* v_cache,
    const int32_t* table, const int32_t* slots, const int32_t* seq_lens,
    const int32_t* live_batch, float* workspace, uint32_t max_pages,
    uint64_t k_page_stride_bytes, uint64_t v_page_stride_bytes,
    uint32_t split_capacity, uint32_t split_target, uint32_t split_units, uint32_t window_left,
    uint32_t scale_bits, uint32_t cap_bits, uint32_t k_scale_bits,
    uint32_t v_scale_bits, typename Fold::Args rope = {}, int produced_from = 0) {
  using D = Decode<T>;
  constexpr bool folded = Fold::enabled;
  constexpr int q_stride = QueryLayout<T, Fold>::row;
  constexpr int q_head_stride = QueryLayout<T, Fold>::head;
  ptx::griddep_launch_dependents();
  // Until griddepcontrol.wait the CTA overlaps its producers: it reads the
  // step inputs (live batch, seq_lens, block tables and slots, which the
  // caller writes with stream-ordered copies before the step's kernels),
  // plans its units and starts copying their K and V rows for positions
  // before each row's current token. Q and the current token's K and V come
  // from the producers and are read after the wait.
  const int kv_head = blockIdx.y;
  __shared__ __align__(1024) uint8_t storage[D::smem];
  __shared__ int ends[max_decode_rows];
  const int live = live_rows(live_batch, max_decode_rows);
  // The stages are free until the plan's first copy: the scan works in them.
  const DecodePlan plan = decode_piece_plan(
      load_decode_rows<T>(ends, storage, live, seq_lens, window_left), split_target);
  const DecodeRows<T> rows{ends, live, seq_lens, window_left};
  const int cta = blockIdx.x;
  if (cta >= plan.ctas) return;
  const int64_t end = plan.start(cta + 1);
  // Units of each piece of the row at `at`.
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
  static_assert(T::group_pad == 8 || T::group_pad == 16,
                "grouped heads fill one or two m8 halves");
  __shared__ __align__(128) uint8_t q_smem[T::group_pad * T::qk * 2];
  const uint32_t phase = (ptx::shared_address(storage) / 128) % 8;
  const uint32_t q_phase = (ptx::shared_address(q_smem) / 128) % 8;
  uint8_t* raw = storage + (T::fp8 ? D::k_bytes + D::v_bytes : 0);
  auto k_tile = [&](int stage) {
    return storage + (T::fp8 ? 0 : stage * (D::k_bytes + D::v_bytes));
  };
  auto v_tile = [&](int stage) { return k_tile(stage) + D::k_bytes; };

  // Byte offset of a position's K or V row.
  auto row_offset = [&](const DecodeCursor& at, int pos, int dim, uint64_t stride) {
    return cache_byte_offset<T>(table, slots, at.row, pos, kv_head, max_pages, dim, bytes,
                                stride);
  };
  // A folded kernel's current K and V rows, held by warp 0 from the issue
  // of the unit that contains them until that unit is computed, and each
  // lane's rotary pair of the current row.
  std::conditional_t<folded,
                     std::conditional_t<Fold::gated, GatedRow<T, Fold>, CurrentRow<T, Fold>>,
                     int> current_row;
  std::conditional_t<folded, float2, int> cos_sin;
  // Lane p's rotary pair p at the current position of `at`'s row.
  auto fold_angle = [&](const DecodeCursor& at) {
    if constexpr (Fold::gated) {
      const int pair = threadIdx.x % 32;
      return gated::table_angle<Fold::rotary, Fold::section_h, Fold::section_w>(
          rope.cos_sin, rope.rope_positions + 3 * uint64_t(at.row), pair % (Fold::rotary / 2));
    } else if constexpr (folded) {
      return lane_angle<Fold>(at.geometry.last, rope);
    } else {
      return make_float2(0.0f, 0.0f);
    }
  };
  auto fetch_current = [&](const DecodeCursor& at) {
    if constexpr (folded) {
      if (warp == 0 && at.key + 1 == at.geometry.units)
        current_row.load(q, at.row, kv_head,
                         row_offset(at, at.geometry.last, T::qk, k_page_stride_bytes),
                         row_offset(at, at.geometry.last, T::v, v_page_stride_bytes));
    }
  };
  // Every thread copies 16-byte sectors of one unit's K and V rows; positions
  // outside the row's visible range are zero filled. `current` selects the
  // rows the producers write: skipped (0, before the PDL wait), only them
  // (1), or included (2). They are the current token's row and, with Rows
  // (exact verify rows), the step's own rows from `produced_from` on, which
  // the row engine writes in this step. With 1, `current_k` and `current_v` are the
  // current row's offsets. A folded kernel never reads the current row from
  // the cache; warp 0 loads it from the QKV rows instead.
  auto issue = [&](const DecodeCursor& at, int stage, int current = 2,
                   uint64_t current_k = 0, uint64_t current_v = 0) {
    if constexpr (folded) {
      if (current != 0) fetch_current(at);
      if (current == 1) return;
      current = 0;
    }
    const int pos0 = (at.geometry.unit0 + at.key) * D::keys;
    auto copy = [&](const void* cache, uint64_t stride, uint8_t* dst, int dim, bool fp8_raw,
                    uint64_t current_offset) {
      constexpr int sector = 16;
      const int sectors = dim * bytes / sector;
      for (int i = threadIdx.x; i < D::keys * sectors; i += D::threads) {
        const int key = i / sectors;
        const int part = i % sectors;
        const int pos = pos0 + key;
        if constexpr (Rows) {
          // The producers write the current row and the step's own rows
          // from `produced_from` on.
          const bool produced = pos == at.geometry.last || pos >= produced_from;
          if (current != 2 && produced != (current == 1)) continue;
        } else {
          if (current != 2 && (pos == at.geometry.last) != (current == 1)) continue;
        }
        const bool valid = pos >= at.geometry.first && pos <= at.geometry.last;
        // Only the current row has a precomputed offset.
        const bool at_current = Rows ? current == 1 && pos == at.geometry.last : current == 1;
        const uint64_t offset = at_current ? current_offset :
            row_offset(at, valid ? pos : at.geometry.last, dim, stride);
        uint8_t* target = fp8_raw ? dst + key * dim + part * sector :
            dst + tile_offset<D::keys>(key, part * 8, phase);
        cp_async_zfill(target, static_cast<const uint8_t*>(cache) + offset + part * sector,
                       valid);
      }
    };
    if constexpr (T::fp8) {
      uint8_t* stage_raw = raw + stage * D::raw_bytes;
      copy(k_cache, k_page_stride_bytes, stage_raw, T::qk, true, current_k);
      copy(v_cache, v_page_stride_bytes, stage_raw + D::keys * T::qk, T::v, true, current_v);
    } else {
      copy(k_cache, k_page_stride_bytes, k_tile(stage), T::qk, false, current_k);
      copy(v_cache, v_page_stride_bytes, v_tile(stage), T::v, false, current_v);
    }
  };

  const float scale = __uint_as_float(scale_bits) *
      (T::fp8 ? __uint_as_float(k_scale_bits) : 1.0f);
  const float cap = __uint_as_float(cap_bits);
  const float v_scale = T::fp8 ? __uint_as_float(v_scale_bits) : 1.0f;
  constexpr float log2e = 1.4426950408889634f;
  const float score_scale = T::softcap ? scale / cap : scale * log2e;
  const uint32_t* q_words = reinterpret_cast<const uint32_t*>(q);

  // Warp w's key tile, V columns, and head pairs: this lane's S^T and O^T
  // columns are heads n * 8 + 2c and n * 8 + 2c + 1 for each n8 tile n.
  // Padding heads past the group hold zero queries and are never written.
  constexpr int heads = T::group_pad / 8;
  constexpr int blocks = D::columns / 16;
  const int tile = warp % D::tiles;
  const int part = warp / D::tiles;
  ptx::B2 query[T::qk / 16][heads];
  float maximum[heads][2];
  float denominator[heads][2];
  ptx::D4 output[blocks][heads];
  int terms = 3;
  int piece = 0;

  // A new piece: its row's term count, its partial slot, and a fresh
  // softmax state.
  auto begin_piece = [&]() {
    const DecodeRow& geometry = cursor.geometry;
    terms = probability_terms(geometry.last - geometry.first + 1);
    piece = cursor.key / piece_of(cursor);
#pragma unroll
    for (int n = 0; n < heads; ++n)
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        maximum[n][r] = -INFINITY;
        denominator[n][r] = 0.0f;
      }
#pragma unroll
    for (int m = 0; m < blocks; ++m)
#pragma unroll
      for (int n = 0; n < heads; ++n)
#pragma unroll
        for (int i = 0; i < 4; ++i) output[m][n].v[i] = 0.0f;
  };
  // Loads the row's grouped Q heads, which the producers wrote, as Q^T B
  // fragments: head n * 8 + g, dimensions 2c and 2c + 8 of each k-step.
  auto load_row = [&]() {
#pragma unroll
    for (int n = 0; n < heads; ++n) {
      const bool head = T::group == T::group_pad || n * 8 + g < T::group;
      const uint32_t* words = q_words +
          (uint64_t(cursor.row) * q_stride +
           (kv_head * T::group + (head ? n * 8 + g : 0)) * q_head_stride) / 2;
#pragma unroll
      for (int ks = 0; ks < T::qk / 16; ++ks)
        query[ks][n] = head ? ptx::B2{{words[ks * 8 + c], words[ks * 8 + c + 4]}} :
                              ptx::B2{{0u, 0u}};
    }
  };
  // The first row's Q heads instead arrive with its first unit: every CTA
  // reads them at once, and 16-byte copies ask L2 for them in fewer requests.
  auto stage_query = [&]() {
    const uint8_t* source = reinterpret_cast<const uint8_t*>(q) +
        (uint64_t(cursor.row) * q_stride + kv_head * T::group * q_head_stride) * 2;
    for (int i = threadIdx.x; i < T::group_pad * T::qk / 8; i += D::threads) {
      const int h = i / (T::qk / 8);
      const int sector = i % (T::qk / 8);
      const bool head = T::group == T::group_pad || h < T::group;
      cp_async_zfill(q_smem + tile_offset<T::group_pad>(h, sector * 8, q_phase),
                     source + (head ? h : 0) * q_head_stride * 2 + sector * 16, head);
    }
  };
  auto read_query = [&]() {
    static_assert(T::qk % 32 == 0, "Q^T fragments load two k-steps at once");
#pragma unroll
    for (int n = 0; n < heads; ++n)
#pragma unroll
      for (int ks = 0; ks < T::qk / 16; ks += 2) {
        uint32_t b[4];
        ptx::ldmatrix_b16<4>(b, q_smem + tile_offset<T::group_pad>(
            n * 8 + lane % 8, ks * 16 + (lane / 8) * 8, q_phase));
        query[ks][n] = ptx::B2{{b[0], b[1]}};
        query[ks + 1][n] = ptx::B2{{b[2], b[3]}};
      }
  };
  // A folded kernel rotates the row's Q fragments: word w of k-step ks holds
  // dimensions 16 ks + 8 w + 2c and one more, and k-step ks + Rotary / 32
  // their pair mates.
  auto rotate_query = [&]() {
    if constexpr (Fold::gated) {
      static_assert(Fold::rotary == 32 || Fold::rotary == 64, "a warp holds the rotary pairs");
#pragma unroll
      for (int n = 0; n < heads; ++n) {
        uint32_t words[T::qk / 8];
#pragma unroll
        for (int ks = 0; ks < T::qk / 16; ++ks) {
          words[2 * ks] = query[ks][n].v[0];
          words[2 * ks + 1] = query[ks][n].v[1];
        }
        gated::norm_rotate_query<T::qk, Fold::rotary>(words, rope.q_norm, rope.eps, [&](int pair) {
          return make_float2(__shfl_sync(0xffffffff, cos_sin.x, pair),
                             __shfl_sync(0xffffffff, cos_sin.y, pair));
        });
#pragma unroll
        for (int ks = 0; ks < T::qk / 16; ++ks)
          query[ks][n] = ptx::B2{{words[2 * ks], words[2 * ks + 1]}};
      }
    } else if constexpr (folded) {
      constexpr int steps = Fold::rotary / 32;
#pragma unroll
      for (int n = 0; n < heads; ++n)
#pragma unroll
        for (int ks = 0; ks < steps; ++ks)
#pragma unroll
          for (int w = 0; w < 2; ++w)
            rotate_words(query[ks][n].v[w], query[ks + steps][n].v[w], cos_sin,
                         16 * ks + 8 * w + 2 * c);
    }
  };

  // The current row's offsets are found before the wait, so after it only
  // its copy remains.
  DecodeCursor ahead = cursor;
  uint64_t current_k[D::stages - 1], current_v[D::stages - 1];
#pragma unroll
  for (int stage = 0; stage + 1 < D::stages; ++stage) {
    if (owns(ahead)) {
      issue(ahead, stage, 0);
      if constexpr (!folded) {
        current_k[stage] = row_offset(ahead, ahead.geometry.last, T::qk, k_page_stride_bytes);
        current_v[stage] = row_offset(ahead, ahead.geometry.last, T::v, v_page_stride_bytes);
      }
    }
    cp_async_commit();
  }
  begin_piece();
  if constexpr (folded) cos_sin = fold_angle(cursor);
  ptx::griddep_wait();
  stage_query();
#pragma unroll
  for (int stage = 0; stage + 1 < D::stages; ++stage) {
    if (owns(ahead)) {
      issue(ahead, stage, 1, current_k[stage], current_v[stage]);
      decode_next(ahead, rows);
    }
    cp_async_commit();
  }

  auto accumulate = [&](auto terms_tag, const uint8_t* k_smem, const uint8_t* v_smem,
                        int key0) {
    constexpr int count = decltype(terms_tag)::value;
    // Rows of this lane's S^T: keys key0 + tile * 16 + g and 8 further.
    const int tile_key = tile * 16;
    // Even and odd k-steps accumulate separately to halve the MMA chain.
    ptx::D4 score[heads], odd[heads];
#pragma unroll
    for (int n = 0; n < heads; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) score[n].v[i] = odd[n].v[i] = 0.0f;
#pragma unroll
    for (int ks = 0; ks < T::qk / 16; ++ks) {
      uint32_t a[4];
      ptx::ldmatrix_b16<4>(a, k_smem + tile_offset<D::keys>(
          tile_key + ((lane / 8) % 2) * 8 + lane % 8, ks * 16 + (lane / 16) * 8, phase));
#pragma unroll
      for (int n = 0; n < heads; ++n)
        ptx::mma_bf16(ks % 2 ? odd[n] : score[n], ptx::A4{{a[0], a[1], a[2], a[3]}},
                      query[ks][n]);
    }
    const bool edge = key0 + tile_key < cursor.geometry.first ||
                      key0 + tile_key + 15 > cursor.geometry.last;
#pragma unroll
    for (int n = 0; n < heads; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float x = (score[n].v[i] + odd[n].v[i]) * score_scale;
        if constexpr (T::softcap) x = cap * tanhf(x) * log2e;
        const int key = key0 + tile_key + g + (i / 2) * 8;
        const bool hidden = edge && (key < cursor.geometry.first || key > cursor.geometry.last);
        score[n].v[i] = hidden ? -INFINITY : x;
      }
    // Maxima and sums over the tile's 16 keys run across the lanes' g.
    float correction[heads][2];
    bool rescale = false;
#pragma unroll
    for (int n = 0; n < heads; ++n)
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        float tile_max = fmaxf(score[n].v[r], score[n].v[r + 2]);
        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 4));
        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 8));
        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 16));
        const float next = fmaxf(maximum[n][r], tile_max);
        const float base = next == -INFINITY ? 0.0f : next;
        correction[n][r] = exp2f(maximum[n][r] - base);
        rescale |= correction[n][r] != 1.0f;
        maximum[n][r] = next;
        score[n].v[r] = exp2f(score[n].v[r] - base);
        score[n].v[r + 2] = exp2f(score[n].v[r + 2] - base);
        denominator[n][r] = denominator[n][r] * correction[n][r] +
            score[n].v[r] + score[n].v[r + 2];
      }
    if (__any_sync(0xffffffff, rescale)) {
#pragma unroll
      for (int m = 0; m < blocks; ++m)
#pragma unroll
        for (int n = 0; n < heads; ++n)
#pragma unroll
          for (int i = 0; i < 4; ++i) output[m][n].v[i] *= correction[n][i % 2];
    }
    // P^T as B fragments: movmatrix turns key g's heads 2c, 2c + 1 into head
    // g's keys 2c, 2c + 1.
    uint32_t probability[count][heads][2];
#pragma unroll
    for (int n = 0; n < heads; ++n)
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        float p0 = score[n].v[2 * half];
        float p1 = score[n].v[2 * half + 1];
#pragma unroll
        for (int term = 0; term < count; ++term) {
          const uint32_t pair = bf16_pair(p0, p1);
          p0 -= bf16_low(pair);
          p1 -= bf16_high(pair);
          probability[term][n][half] = movmatrix_trans(pair);
        }
      }
#pragma unroll
    for (int m = 0; m < blocks; ++m) {
      uint32_t a[4];
      ptx::ldmatrix_b16<4, true>(a, v_smem + tile_offset<D::keys>(
          tile_key + (lane / 16) * 8 + lane % 8,
          (part * blocks + m) * 16 + ((lane / 8) % 2) * 8, phase));
#pragma unroll
      for (int n = 0; n < heads; ++n)
#pragma unroll
        for (int term = 0; term < count; ++term)
          ptx::mma_bf16(output[m][n], ptx::A4{{a[0], a[1], a[2], a[3]}},
                        ptx::B2{{probability[term][n][0], probability[term][n][1]}});
    }
  };

  // Folding key tiles: the warps of tile 1 leave [max, denominator, V
  // numerator] per head in the computed unit's K tile, which no warp reads
  // again before the next unit's wait, and the warps of tile 0 fold them in.
  constexpr int fold_floats = 2 * T::group_pad + D::columns * T::group_pad;
  static_assert(D::tiles == 1 || (D::tiles == 2 && D::parts * fold_floats * 4 <= D::k_bytes),
                "key tile partials fit the K tile");
  for (int t = 0; owns(cursor); ++t) {
    const int stage = t % D::stages;
    if (t > 0 && !continues(cursor)) {
      begin_piece();
      if (cursor.key == 0) {
        if constexpr (folded) cos_sin = fold_angle(cursor);
        load_row();
      }
    }
    // The row's last unit holds its current token, which warp 0 forms in the
    // unit's stage, where no copy wrote it, and writes to the cache.
    if constexpr (folded) {
      if (warp == 0 && cursor.key + 1 == cursor.geometry.units) {
        const int key = cursor.geometry.last - (cursor.geometry.unit0 + cursor.key) * D::keys;
        if constexpr (T::fp8) {
          uint8_t* stage_raw = raw + stage * D::raw_bytes;
          current_row.template store<D::keys>(stage_raw, stage_raw + D::keys * T::qk, phase,
                                              key, cos_sin, rope, __uint_as_float(k_scale_bits),
                                              __uint_as_float(v_scale_bits));
        } else {
          current_row.template store<D::keys>(k_tile(stage), v_tile(stage), phase, key,
                                              cos_sin, rope, 1.0f, 1.0f);
        }
      }
    }

    // Unit t has landed and every warp has left unit t - 1, whose stage now
    // takes unit t + stages - 1.
    cp_async_wait<D::stages - 2>();
    __syncthreads();
    if (t == 0) read_query();
    if (t == 0 || cursor.key == 0) rotate_query();
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
    uint8_t* k_smem = k_tile(T::fp8 ? 0 : stage);
    const uint8_t* v_smem = v_tile(T::fp8 ? 0 : stage);
    if (terms == 1) {
      accumulate(std::integral_constant<int, 1>{}, k_smem, v_smem, key0);
    } else if (terms == 2) {
      accumulate(std::integral_constant<int, 2>{}, k_smem, v_smem, key0);
    } else {
      accumulate(std::integral_constant<int, 3>{}, k_smem, v_smem, key0);
    }

    const bool closes = cursor.key + 1 == cursor.geometry.units ||
        (cursor.key + 1) % piece_of(cursor) == 0;
    if (closes) {
      // The piece is complete.
#pragma unroll
      for (int n = 0; n < heads; ++n)
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          denominator[n][r] += __shfl_xor_sync(0xffffffff, denominator[n][r], 4);
          denominator[n][r] += __shfl_xor_sync(0xffffffff, denominator[n][r], 8);
          denominator[n][r] += __shfl_xor_sync(0xffffffff, denominator[n][r], 16);
        }
      if constexpr (D::tiles == 2) {
        float* fold = reinterpret_cast<float*>(k_smem) + part * fold_floats;
        __syncthreads();
        if (tile == 1) {
#pragma unroll
          for (int n = 0; n < heads; ++n) {
            if (g == 0) {
#pragma unroll
              for (int r = 0; r < 2; ++r) {
                fold[2 * (n * 8 + 2 * c + r)] = maximum[n][r];
                fold[2 * (n * 8 + 2 * c + r) + 1] = denominator[n][r];
              }
            }
#pragma unroll
            for (int m = 0; m < blocks; ++m)
#pragma unroll
              for (int half = 0; half < 2; ++half)
                *reinterpret_cast<float2*>(fold + 2 * T::group_pad +
                    (m * 16 + g + half * 8) * T::group_pad + n * 8 + 2 * c) =
                    make_float2(output[m][n].v[2 * half], output[m][n].v[2 * half + 1]);
          }
        }
        __syncthreads();
        if (tile == 1) {
          decode_next(cursor, rows);
          continue;
        }
#pragma unroll
        for (int n = 0; n < heads; ++n)
#pragma unroll
          for (int r = 0; r < 2; ++r) {
            const int h = n * 8 + 2 * c + r;
            const float other = fold[2 * h];
            const float next = fmaxf(maximum[n][r], other);
            // One of the tiles has seen a visible key, so `next` is finite.
            const float mine = exp2f(maximum[n][r] - next);
            const float theirs = exp2f(other - next);
            denominator[n][r] = denominator[n][r] * mine + fold[2 * h + 1] * theirs;
            maximum[n][r] = next;
#pragma unroll
            for (int m = 0; m < blocks; ++m)
#pragma unroll
              for (int half = 0; half < 2; ++half)
                output[m][n].v[2 * half + r] = output[m][n].v[2 * half + r] * mine +
                    fold[2 * T::group_pad + (m * 16 + g + half * 8) * T::group_pad + h] * theirs;
          }
      }
#pragma unroll
      for (int n = 0; n < heads; ++n)
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const int h = n * 8 + 2 * c + r;
          if (T::group < T::group_pad && h >= T::group) continue;
          float* destination =
              workspace + decode_partial<T>(cursor.row, kv_head, split_capacity, piece, h);
          if (part == 0 && g == 0) {
            destination[0] = maximum[n][r];
            destination[1] = denominator[n][r];
          }
#pragma unroll
          for (int m = 0; m < blocks; ++m)
#pragma unroll
            for (int half = 0; half < 2; ++half)
              destination[2 + (part * blocks + m) * 16 + g + half * 8] =
                  output[m][n].v[2 * half + r] * v_scale;
        }
    }
    decode_next(cursor, rows);
  }
  cp_async_wait<0>();
}

// Exact verify rows of paged decode: CTAs (x, kv head, token) run verify
// token row `token` as the lone row of a decode step at its position, over
// the item's pages (the step's row engine writes its rows before attention,
// so they are copied only after the PDL wait) and with the plan of a
// one-row step, so its partials are those of that step. Each
// token row's partials take one decode step's workspace, in token order;
// `decode_merge_rows` folds them.
template<class T>
__device__ void decode_rows(
    const __nv_bfloat16* q, const void* k_cache, const void* v_cache,
    const int32_t* table, const int32_t* slots, const int32_t* q_start,
    const int32_t* q_len, const int32_t* prefix_len, const int32_t* live_batch,
    float* workspace, uint32_t max_pages, uint64_t k_page_stride_bytes,
    uint64_t v_page_stride_bytes, uint32_t split_capacity, uint32_t split_target,
    uint32_t split_units, uint32_t window_left, uint32_t scale_bits, uint32_t cap_bits,
    uint32_t k_scale_bits, uint32_t v_scale_bits) {
  static_assert(!T::ring, "windowed rows run decode_window");
  __shared__ int32_t row_len, one_row;
  const int token = blockIdx.z;
  const VerifyRow row = verify_row({q_start, q_len, prefix_len}, live_batch, token);
  if (row.item < 0) return;
  if (threadIdx.x == 0) {
    row_len = row.last + 1;
    one_row = 1;
  }
  __syncthreads();
  decode<T, NoRope, true>(q + uint64_t(token) * QueryLayout<T, NoRope>::row, k_cache, v_cache,
            table + uint64_t(row.item) * max_pages, slots ? slots + row.item : nullptr,
            &row_len, &one_row,
            workspace + uint64_t(token) * decode_partial<T>(1, 0, split_capacity, 0, 0),
            max_pages, k_page_stride_bytes, v_page_stride_bytes, split_capacity,
            split_target, split_units, window_left, scale_bits, cap_bits, k_scale_bits,
            v_scale_bits, {}, row.prefix);
}

// Windowed decode: one CTA per row slot and KV head stages the row's whole
// window, at most Warps * 16 positions from its first visible one, and warp w
// owns 16 of them. It forms S^T = K Q^T with the eight grouped Q heads as the
// n8 columns, so no MMA row is padding, transposes P in registers, and
// accumulates O^T = V^T P^T over every V column. Each warp leaves its partial
// in its own keys' stage rows, and the warps' partials are folded with the
// optional sink into BF16 output, so no merge runs. With a NeoxRope `Fold`,
// `q` is the packed QKV rows and `rope` holds the writable ring; see NoRope.
//
// With Rows (exact verify rows) CTA x runs verify token row x of `verify`
// as its own decode row: its window's keys before the item's prefix come
// from the ring, which the step writes only after attention, and the rest
// from the step's `k_new`/`v_new` rows, so the row's bits are those of the
// unfused decode entry at its position.
template<class T, int Warps, class Fold = NoRope, bool Rows = false>
__device__ void decode_window(
    const __nv_bfloat16* q, const void* k_cache, const void* v_cache,
    const int32_t* slots, const int32_t* seq_lens, const int32_t* live_batch,
    uint32_t window_left, uint32_t scale_bits, const __nv_bfloat16* sink,
    __nv_bfloat16* out, RopeArgs rope = {}, VerifyRowsArgs verify = {}) {
  constexpr bool folded = Fold::enabled;
  static_assert(!(Rows && folded), "exact verify rows read rotated rows");
  constexpr int q_stride =
      folded ? T::q_heads * T::qk + T::kv_heads * (T::qk + T::v) : T::q_heads * T::qk;
  static_assert(T::ring && !T::fp8 && !T::softcap && T::group == 8,
                "windowed decode takes BF16 ring rows of eight grouped heads");
  constexpr int keys = Warps * 16;
  constexpr int threads = Warps * 32;
  constexpr int k_bytes = keys * T::qk * 2;
  static_assert(T::qk >= 192 && T::v == 128, "a warp's partial fits its K rows");
  ptx::griddep_launch_dependents();
  extern __shared__ __align__(1024) uint8_t window_stages[];
  __shared__ __align__(128) uint8_t q_smem[T::group * T::qk * 2];
  uint8_t* const k_smem = window_stages;
  uint8_t* const v_smem = window_stages + k_bytes;
  const uint32_t phase = (ptx::shared_address(window_stages) / 128) % 8;
  const uint32_t q_phase = (ptx::shared_address(q_smem) / 128) % 8;
  const int row = blockIdx.x;
  const int kv_head = blockIdx.y;
  // With Rows, positions at or past `prefix` are this step's k_new/v_new
  // rows from `start`.
  int last, prefix = 0, start = 0;
  uint64_t slot;
  if constexpr (Rows) {
    const VerifyRow token = verify_row(verify, live_batch, row);
    if (token.item < 0) return;
    last = token.last;
    prefix = token.prefix;
    start = token.start;
    slot = uint32_t(slots[token.item]);
  } else {
    // A step above the grid's rows must fail loudly, never lose rows.
    if (row >= live_rows(live_batch, gridDim.x)) return;
    last = seq_lens[row] - 1;
    slot = uint32_t(slots[row]);
  }
  const int first = max(0, last - int(window_left));
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int g = lane / 4;
  const int c = lane % 4;

  // Byte offset of a position's row, as cache_token.
  auto offset = [&](int pos, int dim) {
    return ((slot * T::page + pos % T::page) * T::kv_heads + kv_head) * uint64_t(dim) * 2;
  };
  // Local key j holds position first + j; keys past the row are zero filled.
  // The rows the producers write, the current token's and (exact verify
  // rows) the step's own, are copied only when `current` is set.
  auto copy = [&](const void* cache, const __nv_bfloat16* rows_new, uint8_t* dst, int dim,
                  bool current) {
    const int sectors = dim / 8;
    for (int i = threadIdx.x; i < keys * sectors; i += threads) {
      const int key = i / sectors;
      const int part = i % sectors;
      const int pos = first + key;
      if constexpr (Rows) {
        if ((pos == last || pos >= prefix) != current) continue;
        const bool valid = pos <= last;
        const int at = valid ? pos : last;
        const uint8_t* source = at >= prefix ?
            reinterpret_cast<const uint8_t*>(rows_new) +
                ((uint64_t(start) + (at - prefix)) * T::kv_heads + kv_head) * dim * 2 :
            static_cast<const uint8_t*>(cache) + offset(at, dim);
        cp_async_zfill(dst + tile_offset<keys>(key, part * 8, phase), source + part * 16, valid);
      } else {
        if ((pos == last) != current) continue;
        const bool valid = pos <= last;
        cp_async_zfill(dst + tile_offset<keys>(key, part * 8, phase),
                       static_cast<const uint8_t*>(cache) + offset(valid ? pos : last, dim) +
                           part * 16,
                       valid);
      }
    }
  };
  // Before the wait: the window's earlier rows, an L2 prefetch of the
  // current row (or a folded kernel's rotary pair), and the sink logit of
  // the head this thread folds.
  copy(k_cache, verify.k_new, k_smem, T::qk, false);
  copy(v_cache, verify.v_new, v_smem, T::v, false);
  cp_async_commit();
  std::conditional_t<folded, float2, int> cos_sin;
  if constexpr (folded) {
    cos_sin = lane_angle<Fold>(last, rope);
  } else if constexpr (!Rows) {
    if (threadIdx.x == 0)
      prefetch_l2(static_cast<const uint8_t*>(k_cache) + offset(last, T::qk), T::qk * 2);
    if (threadIdx.x == 32)
      prefetch_l2(static_cast<const uint8_t*>(v_cache) + offset(last, T::v), T::v * 2);
  }
  constexpr float log2e = 1.4426950408889634f;
  static_assert(threads % T::group == 0, "a thread folds one head");
  const int fold_head = threadIdx.x % T::group;
  const float sink_logit =
      T::sink ? __bfloat162float(sink[kv_head * T::group + fold_head]) * log2e : -INFINITY;

  // After it: the grouped Q heads and the current row, in one copy group. A
  // folded kernel's warp 0 loads the current row from the QKV rows instead,
  // then forms it in the stage and writes it to the ring.
  ptx::griddep_wait();
  const uint8_t* q_bytes = reinterpret_cast<const uint8_t*>(q) +
      (uint64_t(row) * q_stride + kv_head * T::group * T::qk) * 2;
  for (int i = threadIdx.x; i < T::group * T::qk / 8; i += threads) {
    const int h = i / (T::qk / 8);
    const int part = i % (T::qk / 8);
    cp_async_zfill(q_smem + tile_offset<T::group>(h, part * 8, q_phase),
                   q_bytes + h * T::qk * 2 + part * 16, true);
  }
  if constexpr (folded) {
    if (warp == 0) {
      CurrentRow<T, Fold> current_row;
      current_row.load(q, row, kv_head, offset(last, T::qk), offset(last, T::v));
      current_row.template store<keys>(k_smem, v_smem, phase, last - first, cos_sin, rope, 1.0f,
                                       1.0f);
    }
  } else {
    copy(k_cache, verify.k_new, k_smem, T::qk, true);
    copy(v_cache, verify.v_new, v_smem, T::v, true);
  }
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();
  // Q^T as B fragments: head g, dimensions 2c and 2c + 8 of each k-step.
  ptx::B2 query[T::qk / 16];
#pragma unroll
  for (int ks = 0; ks < T::qk / 16; ks += 2) {
    uint32_t b[4];
    ptx::ldmatrix_b16<4>(b, q_smem + tile_offset<T::group>(lane % 8, ks * 16 + (lane / 8) * 8,
                                                           q_phase));
    query[ks] = ptx::B2{{b[0], b[1]}};
    query[ks + 1] = ptx::B2{{b[2], b[3]}};
  }
  if constexpr (folded) {
    constexpr int steps = Fold::rotary / 32;
#pragma unroll
    for (int ks = 0; ks < steps; ++ks)
#pragma unroll
      for (int w = 0; w < 2; ++w)
        rotate_words(query[ks].v[w], query[ks + steps].v[w], cos_sin, 16 * ks + 8 * w + 2 * c);
  }

  // Rows of S^T and O^T's columns: keys key0 + g and key0 + g + 8, heads 2c
  // and 2c + 1.
  const int key0 = warp * 16;
  float maximum[2] = {-INFINITY, -INFINITY};
  float denominator[2] = {0.0f, 0.0f};
  ptx::D4 output[T::v / 16];
#pragma unroll
  for (int m = 0; m < T::v / 16; ++m)
#pragma unroll
    for (int i = 0; i < 4; ++i) output[m].v[i] = 0.0f;
  auto accumulate = [&](auto terms_tag) {
    constexpr int count = decltype(terms_tag)::value;
    // Even and odd k-steps accumulate separately to halve the MMA chain.
    ptx::D4 score{}, odd{};
#pragma unroll
    for (int ks = 0; ks < T::qk / 16; ++ks) {
      uint32_t a[4];
      ptx::ldmatrix_b16<4>(a, k_smem + tile_offset<keys>(
          key0 + ((lane / 8) % 2) * 8 + lane % 8, ks * 16 + (lane / 16) * 8, phase));
      ptx::mma_bf16(ks % 2 ? odd : score, ptx::A4{{a[0], a[1], a[2], a[3]}}, query[ks]);
    }
    const float score_scale = __uint_as_float(scale_bits) * log2e;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const bool hidden = first + key0 + g + (i / 2) * 8 > last;
      score.v[i] = hidden ? -INFINITY : (score.v[i] + odd.v[i]) * score_scale;
    }
    uint32_t probability[count][2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      float tile_max = fmaxf(score.v[h], score.v[h + 2]);
      tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 4));
      tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 8));
      tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 16));
      maximum[h] = tile_max;
      score.v[h] = exp2f(score.v[h] - tile_max);
      score.v[h + 2] = exp2f(score.v[h + 2] - tile_max);
      float sum = score.v[h] + score.v[h + 2];
      sum += __shfl_xor_sync(0xffffffff, sum, 4);
      sum += __shfl_xor_sync(0xffffffff, sum, 8);
      sum += __shfl_xor_sync(0xffffffff, sum, 16);
      denominator[h] = sum;
    }
    // P^T as B fragments: movmatrix turns key g's heads 2c, 2c + 1 into head
    // g's keys 2c, 2c + 1.
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      float p0 = score.v[2 * half];
      float p1 = score.v[2 * half + 1];
#pragma unroll
      for (int term = 0; term < count; ++term) {
        const uint32_t pair = bf16_pair(p0, p1);
        p0 -= bf16_low(pair);
        p1 -= bf16_high(pair);
        probability[term][half] = movmatrix_trans(pair);
      }
    }
#pragma unroll
    for (int m = 0; m < T::v / 16; ++m) {
      uint32_t a[4];
      ptx::ldmatrix_b16<4, true>(a, v_smem + tile_offset<keys>(
          key0 + (lane / 16) * 8 + lane % 8, m * 16 + ((lane / 8) % 2) * 8, phase));
#pragma unroll
      for (int term = 0; term < count; ++term)
        ptx::mma_bf16(output[m], ptx::A4{{a[0], a[1], a[2], a[3]}},
                      ptx::B2{{probability[term][0], probability[term][1]}});
    }
  };
  if (first + key0 <= last) {
    if (probability_terms(last - first + 1) == 2)
      accumulate(std::integral_constant<int, 2>{});
    else
      accumulate(std::integral_constant<int, 3>{});
  }

  // Each warp leaves its V numerator as [column][head] in its keys' rows of
  // the first two K column blocks and [max, denominator] per head in the
  // third; no other warp reads those rows. Idle warps leave a zero numerator
  // and an empty maximum.
  auto numerator = [&](int w, int column, int h) {
    return reinterpret_cast<float*>(k_smem + (column / 64) * keys * 128 + w * 16 * 128) +
        (column % 64) * T::group + h;
  };
  auto statistics = [&](int w) {
    return reinterpret_cast<float*>(k_smem + 2 * keys * 128 + w * 16 * 128);
  };
  __syncwarp();
  if (g == 0) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      statistics(warp)[2 * (2 * c + h)] = maximum[h];
      statistics(warp)[2 * (2 * c + h) + 1] = denominator[h];
    }
  }
#pragma unroll
  for (int m = 0; m < T::v / 16; ++m)
#pragma unroll
    for (int half = 0; half < 2; ++half)
      *reinterpret_cast<float2*>(numerator(warp, m * 16 + g + half * 8, 2 * c)) =
          make_float2(output[m].v[2 * half], output[m].v[2 * half + 1]);
  __syncthreads();
  // Thread i folds head i % 8 at columns 2 (i / 8) and 2 (i / 8) + 1.
  float weight[Warps];
  float total_max = sink_logit;
#pragma unroll
  for (int w = 0; w < Warps; ++w) {
    weight[w] = statistics(w)[2 * fold_head];
    total_max = fmaxf(total_max, weight[w]);
  }
  float total = T::sink ? exp2f(sink_logit - total_max) : 0.0f;
#pragma unroll
  for (int w = 0; w < Warps; ++w) {
    weight[w] = exp2f(weight[w] - total_max);
    total += statistics(w)[2 * fold_head + 1] * weight[w];
  }
  for (int column = 2 * (threadIdx.x / T::group); column < T::v;
       column += 2 * threads / T::group) {
    float sum0 = 0.0f, sum1 = 0.0f;
#pragma unroll
    for (int w = 0; w < Warps; ++w) {
      sum0 += *numerator(w, column, fold_head) * weight[w];
      sum1 += *numerator(w, column + 1, fold_head) * weight[w];
    }
    reinterpret_cast<uint32_t*>(out)[
        ((uint64_t(row) * T::q_heads + kv_head * T::group + fold_head) * T::v + column) / 2] =
        bf16_pair(sum0 / total, sum1 / total);
  }
}

// One CTA per output row and 64 V columns folds `pieces` owner partials
// [max, denominator, V numerator], `stride` floats apart from `base`, with
// the optional sink logit and writes BF16 `out` (the row's V columns). Each
// of the CTA's W warps (up to MaxThreads / 32) folds owners w, w + W, ...
// eight at a time, rescaling once per batch; each lane owns two columns.
// The warps' results then meet in a fixed shuffle tree.
template<class T, int MaxThreads>
__device__ __forceinline__ void merge_owners(
    const float* base, uint64_t stride, int pieces, float sink_logit,
    __nv_bfloat16* out) {
  const int warps = blockDim.x / 32;
  constexpr int batch = 8;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int column = min(blockIdx.x * 64 + 2 * lane, T::v - 2);
  float maximum = -INFINITY;
  float denominator = 0.0f;
  float2 value = make_float2(0.0f, 0.0f);
  for (int first = warp; first < pieces; first += warps * batch) {
    float2 head[batch];
    float2 pair[batch];
#pragma unroll
    for (int i = 0; i < batch; ++i) {
      const int p = first + i * warps;
      head[i] = p < pieces ? *reinterpret_cast<const float2*>(base + p * stride) :
                             make_float2(-INFINITY, 0.0f);
      pair[i] = p < pieces ?
          *reinterpret_cast<const float2*>(base + p * stride + 2 + column) :
          make_float2(0.0f, 0.0f);
    }
    // The batch's maximum first, so that its weights are independent.
    float next = maximum;
#pragma unroll
    for (int i = 0; i < batch; ++i) next = fmaxf(next, head[i].x);
    if (next == -INFINITY) continue;
    const float keep = exp2f(maximum - next);
    denominator *= keep;
    value.x *= keep;
    value.y *= keep;
#pragma unroll
    for (int i = 0; i < batch; ++i) {
      const float weight = head[i].x == -INFINITY ? 0.0f : exp2f(head[i].x - next);
      denominator += head[i].y * weight;
      value.x += pair[i].x * weight;
      value.y += pair[i].y * weight;
    }
    maximum = next;
  }
  __shared__ float maxima[MaxThreads / 32];
  __shared__ float denominators[MaxThreads / 32];
  __shared__ float2 values[MaxThreads / 32][32];
  if (lane == 0) {
    maxima[warp] = maximum;
    denominators[warp] = denominator;
  }
  values[warp][lane] = value;
  __syncthreads();
  // Thread t combines warp t % warps's partial of column pair t / warps;
  // the warps' partials of one pair meet in adjacent lanes.
  const int source = threadIdx.x % warps;
  const int pair_index = threadIdx.x / warps;
  const float source_max = maxima[source];
  float total_max = source_max;
  for (int offset = 1; offset < warps; offset *= 2)
    total_max = fmaxf(total_max, __shfl_xor_sync(0xffffffff, total_max, offset));
  total_max = fmaxf(total_max, sink_logit);
  const float weight = source_max == -INFINITY ? 0.0f : exp2f(source_max - total_max);
  float total = denominators[source] * weight;
  float2 sum = make_float2(values[source][pair_index].x * weight,
                           values[source][pair_index].y * weight);
  for (int offset = 1; offset < warps; offset *= 2) {
    total += __shfl_xor_sync(0xffffffff, total, offset);
    sum.x += __shfl_xor_sync(0xffffffff, sum.x, offset);
    sum.y += __shfl_xor_sync(0xffffffff, sum.y, offset);
  }
  const int out_column = blockIdx.x * 64 + 2 * pair_index;
  if (source != 0 || out_column >= T::v) return;
  if constexpr (T::sink) total += exp2f(sink_logit - total_max);
  reinterpret_cast<uint32_t*>(out)[out_column / 2] = bf16_pair(sum.x / total, sum.y / total);
}

// One CTA per row, Q head and 64 V columns combines the row's partials,
// counted from the row's own length, with `merge_owners`; the planner gives
// small batches, whose rows have more partials, more warps.
template<class T, int MaxThreads>
__device__ void decode_merge(
    const float* workspace, const __nv_bfloat16* sink, const int32_t* seq_lens,
    const int32_t* live_batch, __nv_bfloat16* out, uint32_t split_capacity,
    uint32_t split_target, uint32_t split_units, uint32_t window_left) {
  // Windowed rows are combined by decode_window. The wait keeps the entry a
  // link in the programmatic launch chain: kernels that wait on it also see
  // decode_window's output.
  if constexpr (T::ring) {
    ptx::griddep_wait();
    return;
  }
  const int row = blockIdx.z;
  // Before the wait, the row's piece count from its own length in the step
  // inputs. A step above the grid's rows must fail loudly, never lose rows.
  const int live = live_rows(live_batch, min(max_decode_rows, int(gridDim.z)));
  if (row >= live) return;
  const int pieces = row_pieces(decode_row<T>(seq_lens, row, window_left).units,
                                split_capacity, split_target, split_units);
  constexpr float log2e = 1.4426950408889634f;
  const float sink_logit = T::sink ? __bfloat162float(sink[blockIdx.y]) * log2e : -INFINITY;
  ptx::griddep_wait();
  const int q_head = blockIdx.y;
  const int kv_head = q_head / T::group;
  const int h = q_head % T::group;
  merge_owners<T, MaxThreads>(
      workspace + decode_partial<T>(row, kv_head, split_capacity, 0, h),
      uint64_t(T::group) * (T::v + 2), pieces, sink_logit,
      out + (uint64_t(row) * T::q_heads + q_head) * T::v);
}

// Folds `decode_rows`' partials: CTA (column block, Q head, token) merges
// token row `token` with the one-row plan of its position, as
// `decode_merge` merges a one-row step (launched with that step's threads).
template<class T, int MaxThreads>
__device__ void decode_merge_rows(
    const float* workspace, const __nv_bfloat16* sink, const int32_t* q_start,
    const int32_t* q_len, const int32_t* prefix_len, const int32_t* live_batch,
    __nv_bfloat16* out, uint32_t split_capacity, uint32_t split_target,
    uint32_t split_units, uint32_t window_left) {
  static_assert(!T::ring, "windowed rows need no merge");
  const int token = blockIdx.z;
  const VerifyRow row = verify_row({q_start, q_len, prefix_len}, live_batch, token);
  if (row.item < 0) return;
  const int32_t row_len = row.last + 1;
  const int pieces = row_pieces(decode_row<T>(&row_len, 0, window_left).units,
                                split_capacity, split_target, split_units);
  constexpr float log2e = 1.4426950408889634f;
  const float sink_logit = T::sink ? __bfloat162float(sink[blockIdx.y]) * log2e : -INFINITY;
  ptx::griddep_wait();
  const int q_head = blockIdx.y;
  const int kv_head = q_head / T::group;
  const int h = q_head % T::group;
  merge_owners<T, MaxThreads>(
      workspace + uint64_t(token) * decode_partial<T>(1, 0, split_capacity, 0, 0) +
          decode_partial<T>(0, kv_head, split_capacity, 0, h),
      uint64_t(T::group) * (T::v + 2), pieces, sink_logit,
      out + (uint64_t(token) * T::q_heads + q_head) * T::v);
}

} // namespace chop::attention
