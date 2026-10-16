#pragma once

// Exact verify rows of paged attention, a group of rows per CTA. A verify
// item's token rows t sit at positions prefix + t; each row alone would be a
// one-row decode step, whose plan cuts the row's units (Decode<T>::keys keys
// each, from position 0) into pieces (`piece_units`). Rows of one item with
// the same unit count (a class; an item of at most `keys` rows has at most
// two) share those pieces, so CTA j of a class computes piece j for all of
// them. A CTA takes one range and up to Group rows of a class and streams
// the range's K and V through decode's two stages once; each row has four
// warps of its own that run decode's arithmetic on the shared tiles (the
// same key tile and V columns per warp, MMA fragments, masks, probability
// terms, tile fold and partial slot), so every row's partials, and after
// `decode_merge_rows` its output, are bitwise those of the one-row decode
// step. A one-row step zero fills the keys past its row; here the tiles hold
// the group's rows up to its last, so a row zeroes the V fragments of keys
// past its own position before its P V products, which gives decode's
// operands exactly (their scores are masked as in decode). Q heads are read
// from shared memory at each k-step instead of held in registers, which
// changes no arithmetic and lets 4 * Group warps fit the register file.
//
// Grid: (group ordinal, CTA, KV head). Ordinals count the groups of the
// live items in order: item, class (its first unit count, then one more),
// Group rows at a time; items never hold more groups than token rows, and a
// range's groups are adjacent in launch order so their tiles meet in L2. The
// step's rows are written into the pages by this step's row engine, so
// every copy runs after the PDL wait.

#include "decode.cuh"

namespace chop::attention {

template<class T, int Group>
struct DecodeRowsGroup {
  using D = Decode<T>;
  static constexpr int warps = Group * D::warps;
  static constexpr int threads = warps * 32;
  static constexpr int q_bytes = T::group_pad * T::qk * 2;
  static constexpr int fold_floats = 2 * T::group_pad + D::columns * T::group_pad;
  // Decode's stages (and FP8 widen tile), then each row's Q stage. The rows'
  // tile partials fold through the stages once the range is done.
  static constexpr int q_offset = D::smem;
  static constexpr int smem = q_offset + Group * q_bytes;
  static_assert(D::smem % 1024 == 0 && q_bytes % 128 == 0, "stages keep their swizzle phases");
  static_assert(D::tiles == 1 || Group * D::parts * fold_floats * 4 <= D::smem,
                "the rows' key tile partials fit the stages");
};

template<class T, int Group>
__device__ void decode_rows_group(
    const __nv_bfloat16* q, const void* k_cache, const void* v_cache,
    const int32_t* table, const int32_t* q_start, const int32_t* q_len,
    const int32_t* prefix_len, const int32_t* live_batch, float* workspace,
    uint32_t max_pages, uint64_t k_page_stride_bytes, uint64_t v_page_stride_bytes,
    uint32_t split_capacity, uint32_t split_target, uint32_t split_units, uint32_t scale_bits,
    uint32_t cap_bits, uint32_t k_scale_bits, uint32_t v_scale_bits) {
  static_assert(!T::ring, "sliding-window rows run decode_window");
  using D = Decode<T>;
  using S = DecodeRowsGroup<T, Group>;
  constexpr int keys = D::keys;
  constexpr int q_stride = QueryLayout<T, NoRope>::row;
  constexpr int q_head_stride = QueryLayout<T, NoRope>::head;
  constexpr int bytes = T::fp8 ? 1 : 2;
  ptx::griddep_launch_dependents();
  // This block's group: item `item`, rows row0 .. row0 + rows - 1, all with
  // `units` units.
  int item = -1, units = 0, row0 = 0, rows = 0;
  {
    const int items = live_rows(live_batch, max_decode_rows);
    int ordinal = 0;
    for (int i = 0; i < items && item < 0; ++i) {
      const int tokens = q_len[i];
      const int prefix = prefix_len[i];
      if (tokens > keys) __trap();
      for (int cls = 0; cls < 2 && item < 0; ++cls) {
        // Rows r with (prefix + r) / keys + 1 == u.
        const int u = prefix / keys + 1 + cls;
        const int low = max(0, (u - 1) * keys - prefix);
        const int high = min(tokens, u * keys - prefix);
        for (int first = low; first < high; first += Group, ++ordinal) {
          if (ordinal == int(blockIdx.x)) {
            item = i;
            units = u;
            row0 = first;
            rows = min(Group, high - first);
          }
        }
      }
    }
  }
  if (item < 0) return;
  const int start = q_start[item];
  const int prefix = prefix_len[item];
  const int cta = blockIdx.y;
  // CTA j computes the one-row plan's piece j, so a CTA's partial slot is
  // its index.
  const int size = piece_units(units, split_capacity, split_target, split_units);
  const int unit_begin = cta * size;
  if (unit_begin >= units) return;
  const int unit_end = min(units, unit_begin + size);
  const int piece = cta;
  const int kv_head = blockIdx.z;
  // Positions past the group's last row are zero filled, as decode's are.
  const int last_max = prefix + row0 + rows - 1;

  extern __shared__ __align__(1024) uint8_t group_storage[];
  uint8_t* const q_stages = group_storage + S::q_offset;
  const uint32_t phase = (ptx::shared_address(group_storage) / 128) % 8;
  const uint32_t q_phase = (ptx::shared_address(q_stages) / 128) % 8;
  uint8_t* raw = group_storage + (T::fp8 ? D::k_bytes + D::v_bytes : 0);
  auto k_tile = [&](int stage) {
    return group_storage + (T::fp8 ? 0 : stage * (D::k_bytes + D::v_bytes));
  };
  auto v_tile = [&](int stage) { return k_tile(stage) + D::k_bytes; };

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int g = lane / 4;
  const int c = lane % 4;
  // Warp w runs row w / 4 of the group as decode's warp w % 4.
  const int slot = warp / D::warps;
  const int tile = (warp % D::warps) % D::tiles;
  const int part = (warp % D::warps) / D::tiles;
  const bool active = slot < rows;
  const int last = prefix + row0 + slot;
  const int token = start + row0 + slot;
  const uint8_t* const q_smem = q_stages + slot * S::q_bytes;

  auto offset = [&](int pos, int dim, uint64_t stride) {
    return cache_byte_offset<T>(table, nullptr, item, pos, kv_head, max_pages, dim, bytes,
                                stride);
  };
  // decode's copy of one unit's K and V rows into a stage, keys past the
  // group's last row zero filled.
  auto issue = [&](int unit, int stage) {
    const int pos0 = unit * keys;
    auto copy = [&](const void* cache, uint64_t stride, uint8_t* dst, int dim, bool fp8_raw) {
      constexpr int sector = 16;
      const int sectors = dim * bytes / sector;
      for (int i = threadIdx.x; i < keys * sectors; i += S::threads) {
        const int key = i / sectors;
        const int column = i % sectors;
        const int pos = pos0 + key;
        const bool valid = pos <= last_max;
        uint8_t* target = fp8_raw ? dst + key * dim + column * sector :
            dst + tile_offset<keys>(key, column * 8, phase);
        cp_async_zfill(target,
                       static_cast<const uint8_t*>(cache) +
                           offset(valid ? pos : last_max, dim, stride) + column * sector,
                       valid);
      }
    };
    if constexpr (T::fp8) {
      uint8_t* stage_raw = raw + stage * D::raw_bytes;
      copy(k_cache, k_page_stride_bytes, stage_raw, T::qk, true);
      copy(v_cache, v_page_stride_bytes, stage_raw + keys * T::qk, T::v, true);
    } else {
      copy(k_cache, k_page_stride_bytes, k_tile(stage), T::qk, false);
      copy(v_cache, v_page_stride_bytes, v_tile(stage), T::v, false);
    }
  };

  const float scale = __uint_as_float(scale_bits) *
      (T::fp8 ? __uint_as_float(k_scale_bits) : 1.0f);
  const float cap = __uint_as_float(cap_bits);
  const float v_scale = T::fp8 ? __uint_as_float(v_scale_bits) : 1.0f;
  constexpr float log2e = 1.4426950408889634f;
  const float score_scale = T::softcap ? scale / cap : scale * log2e;

  constexpr int heads = T::group_pad / 8;
  constexpr int blocks = D::columns / 16;
  float maximum[heads][2];
  float denominator[heads][2];
  ptx::D4 output[blocks][heads];
  // decode's begin_row.
  const int terms = probability_terms(last + 1);
#pragma unroll
  for (int n = 0; n < heads; ++n)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      maximum[n][h] = -INFINITY;
      denominator[n][h] = 0.0f;
    }
#pragma unroll
  for (int m = 0; m < blocks; ++m)
#pragma unroll
    for (int n = 0; n < heads; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) output[m][n].v[i] = 0.0f;

  // A BF16 pair of keys `key` and `key` + 1, each zeroed past this row.
  auto own_keys = [&](uint32_t pair, int key) {
    return (key > last ? 0u : pair & 0xffffu) | (key + 1 > last ? 0u : pair & 0xffff0000u);
  };
  // decode's accumulate over one unit of this warp's row, with the Q^T
  // fragments of each pair of k-steps read from the row's Q stage.
  auto accumulate = [&](auto terms_tag, const uint8_t* k_smem, const uint8_t* v_smem,
                        int key0) {
    constexpr int count = decltype(terms_tag)::value;
    const int tile_key = tile * 16;
    ptx::D4 score[heads], odd[heads];
#pragma unroll
    for (int n = 0; n < heads; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) score[n].v[i] = odd[n].v[i] = 0.0f;
    static_assert(T::qk % 32 == 0, "Q^T fragments load two k-steps at once");
#pragma unroll
    for (int ks = 0; ks < T::qk / 16; ks += 2) {
      ptx::B2 query[2][heads];
#pragma unroll
      for (int n = 0; n < heads; ++n) {
        uint32_t b[4];
        ptx::ldmatrix_b16<4>(b, q_smem + tile_offset<T::group_pad>(
            n * 8 + lane % 8, ks * 16 + (lane / 8) * 8, q_phase));
        query[0][n] = ptx::B2{{b[0], b[1]}};
        query[1][n] = ptx::B2{{b[2], b[3]}};
      }
#pragma unroll
      for (int step = 0; step < 2; ++step) {
        uint32_t a[4];
        ptx::ldmatrix_b16<4>(a, k_smem + tile_offset<keys>(
            tile_key + ((lane / 8) % 2) * 8 + lane % 8, (ks + step) * 16 + (lane / 16) * 8,
            phase));
#pragma unroll
        for (int n = 0; n < heads; ++n)
          ptx::mma_bf16(step ? odd[n] : score[n], ptx::A4{{a[0], a[1], a[2], a[3]}},
                        query[step][n]);
      }
    }
    const bool edge = key0 + tile_key + 15 > last;
#pragma unroll
    for (int n = 0; n < heads; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float x = (score[n].v[i] + odd[n].v[i]) * score_scale;
        if constexpr (T::softcap) x = cap * tanhf(x) * log2e;
        const int key = key0 + tile_key + g + (i / 2) * 8;
        const bool hidden = edge && key > last;
        score[n].v[i] = hidden ? -INFINITY : x;
      }
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
      ptx::ldmatrix_b16<4, true>(a, v_smem + tile_offset<keys>(
          tile_key + (lane / 16) * 8 + lane % 8,
          (part * blocks + m) * 16 + ((lane / 8) % 2) * 8, phase));
      if (edge) {
        // a[0] and a[1] hold keys 2c and 2c + 1 of the tile, a[2] and a[3]
        // keys 2c + 8 and 2c + 9, the lower key in the low half.
        const int key = key0 + tile_key + 2 * c;
        a[0] = own_keys(a[0], key);
        a[1] = own_keys(a[1], key);
        a[2] = own_keys(a[2], key + 8);
        a[3] = own_keys(a[3], key + 8);
      }
#pragma unroll
      for (int n = 0; n < heads; ++n)
#pragma unroll
        for (int term = 0; term < count; ++term)
          ptx::mma_bf16(output[m][n], ptx::A4{{a[0], a[1], a[2], a[3]}},
                        ptx::B2{{probability[term][n][0], probability[term][n][1]}});
    }
  };

  // Every copy reads rows this step's producers wrote.
  ptx::griddep_wait();
  {
    // decode's stage_query for each row of the group.
    constexpr int sectors = T::qk / 8;
    for (int i = threadIdx.x; i < rows * T::group_pad * sectors; i += S::threads) {
      const int row = i / (T::group_pad * sectors);
      const int h = (i / sectors) % T::group_pad;
      const int sector = i % sectors;
      const bool head = T::group == T::group_pad || h < T::group;
      const uint8_t* source = reinterpret_cast<const uint8_t*>(q) +
          (uint64_t(start + row0 + row) * q_stride + kv_head * T::group * q_head_stride) * 2;
      cp_async_zfill(q_stages + row * S::q_bytes +
                         tile_offset<T::group_pad>(h, sector * 8, q_phase),
                     source + (head ? h : 0) * q_head_stride * 2 + sector * 16, head);
    }
  }
  issue(unit_begin, 0);
  cp_async_commit();
  for (int unit = unit_begin; unit < unit_end; ++unit) {
    const int stage = (unit - unit_begin) % D::stages;
    // The unit has landed and every warp has left the one before, whose
    // stage now takes the next unit.
    cp_async_wait<0>();
    __syncthreads();
    if (unit + 1 < unit_end) issue(unit + 1, (stage + 1) % D::stages);
    cp_async_commit();
    if constexpr (T::fp8) {
      const uint8_t* stage_raw = raw + stage * D::raw_bytes;
      widen_fp8<keys, T::qk, S::threads>(k_tile(0), stage_raw, phase);
      widen_fp8<keys, T::v, S::threads>(v_tile(0), stage_raw + keys * T::qk, phase);
      __syncthreads();
    }
    if (active) {
      const uint8_t* k_smem = k_tile(T::fp8 ? 0 : stage);
      const uint8_t* v_smem = v_tile(T::fp8 ? 0 : stage);
      const int key0 = unit * keys;
      if (terms == 1) {
        accumulate(std::integral_constant<int, 1>{}, k_smem, v_smem, key0);
      } else if (terms == 2) {
        accumulate(std::integral_constant<int, 2>{}, k_smem, v_smem, key0);
      } else {
        accumulate(std::integral_constant<int, 3>{}, k_smem, v_smem, key0);
      }
    }
  }
  cp_async_wait<0>();

  // decode's end of a row's share: the tile fold and the partial write.
#pragma unroll
  for (int n = 0; n < heads; ++n)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      denominator[n][h] += __shfl_xor_sync(0xffffffff, denominator[n][h], 4);
      denominator[n][h] += __shfl_xor_sync(0xffffffff, denominator[n][h], 8);
      denominator[n][h] += __shfl_xor_sync(0xffffffff, denominator[n][h], 16);
    }
  bool writes = active;
  if constexpr (D::tiles == 2) {
    float* fold = reinterpret_cast<float*>(group_storage) +
        (slot * D::parts + part) * S::fold_floats;
    // Every warp has left the last unit's stage.
    __syncthreads();
    if (active && tile == 1) {
#pragma unroll
      for (int n = 0; n < heads; ++n) {
        if (g == 0) {
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            fold[2 * (n * 8 + 2 * c + h)] = maximum[n][h];
            fold[2 * (n * 8 + 2 * c + h) + 1] = denominator[n][h];
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
      writes = false;
    } else if (active) {
#pragma unroll
      for (int n = 0; n < heads; ++n)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int head = n * 8 + 2 * c + h;
          const float other = fold[2 * head];
          const float next = fmaxf(maximum[n][h], other);
          const float mine = exp2f(maximum[n][h] - next);
          const float theirs = exp2f(other - next);
          denominator[n][h] = denominator[n][h] * mine + fold[2 * head + 1] * theirs;
          maximum[n][h] = next;
#pragma unroll
          for (int m = 0; m < blocks; ++m)
#pragma unroll
            for (int half = 0; half < 2; ++half)
              output[m][n].v[2 * half + h] = output[m][n].v[2 * half + h] * mine +
                  fold[2 * T::group_pad + (m * 16 + g + half * 8) * T::group_pad + head] *
                      theirs;
        }
    }
  }
  if (writes) {
    float* row_workspace =
        workspace + uint64_t(token) * decode_partial<T>(1, 0, split_capacity, 0, 0);
#pragma unroll
    for (int n = 0; n < heads; ++n)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int head = n * 8 + 2 * c + h;
        if (T::group < T::group_pad && head >= T::group) continue;
        float* destination =
            row_workspace + decode_partial<T>(0, kv_head, split_capacity, piece, head);
        if (part == 0 && g == 0) {
          destination[0] = maximum[n][h];
          destination[1] = denominator[n][h];
        }
#pragma unroll
        for (int m = 0; m < blocks; ++m)
#pragma unroll
          for (int half = 0; half < 2; ++half)
            destination[2 + (part * blocks + m) * 16 + g + half * 8] =
                output[m][n].v[2 * half + h] * v_scale;
      }
  }
}

} // namespace chop::attention
