#pragma once
#include "sources_decode.cuh"

// Decode selected native KV once per row. Source tasks reuse those exact
// BF16 operands while retaining the reference QK, softmax, PV and split order.
namespace chop::latent::sources_cached {
using sources_decode::Index;
using sources_decode::mark;
using sources::partial_index;
using sources::exp_of;
using attention::tile_offset;
struct CacheIndex { uint32_t row, key_group; };

template<class T, bool Wrong>
__device__ void predecode(
    CacheIndex task, uint8_t* arena, uint16_t* decoded, int32_t* cached_valid,
    const int32_t* row_batch, const uint32_t* live_rows,
    const uint8_t* window_codes, const uint8_t* window_scales,
    const int32_t* window_item_slots, const int32_t* window_keys,
    const uint8_t* compressed_pool, const int32_t* compressed_keys,
    const int32_t* block_tables,
    uint64_t window_code_stride, uint64_t window_scale_stride,
    uint32_t window_slots, uint32_t window_sequences, uint32_t window_items,
    uint32_t window_key_stride, uint32_t window_budget,
    uint64_t page_stride, uint32_t max_pages, uint32_t compressed_pages,
    uint32_t compressed_sequences, uint32_t compressed_key_stride,
    uint32_t compressed_budget) {
  const uint32_t row = task.row;
  if (row >= *live_rows) return;
  const uint32_t j = task.key_group * 4 + threadIdx.x / 32;
  const uint32_t lane = threadIdx.x % 32;
  if (j >= T::max_keys) return;
  const int batch = row_batch[row];
  const int state_slot = batch >= 0 && uint32_t(batch) < window_items
      ? window_item_slots[batch] : -1;
  const uint32_t count = window_budget + (T::second ? compressed_budget : 0);
  int kind = 0;
  const uint8_t *codes = nullptr, *scales = nullptr;
  if (j < window_budget && state_slot >= 0 && uint32_t(state_slot) < window_sequences) {
    const int32_t position = window_keys[uint64_t(row) * window_key_stride + j];
    if (position >= 0 && uint32_t(position) < window_slots) {
      codes = window_codes + uint64_t(state_slot) * window_code_stride + uint64_t(position) * T::latent;
      scales = window_scales + uint64_t(state_slot) * window_scale_stride + uint64_t(position) * (T::latent / 32);
      kind = 1;
    }
  } else if constexpr (T::second) {
    if (j >= window_budget && j < count && batch >= 0 && uint32_t(batch) < compressed_sequences) {
      const int32_t position = compressed_keys[uint64_t(row) * compressed_key_stride + j - window_budget];
      if (position >= 0 && uint32_t(position) / T::page < max_pages) {
        const int32_t page = block_tables[uint64_t(batch) * max_pages + uint32_t(position) / T::page];
        if (page >= 0 && uint32_t(page) < compressed_pages) {
          const uint8_t* base = compressed_pool + uint64_t(page) * page_stride;
          const uint32_t slot = uint32_t(position) % T::page;
          codes = base + uint64_t(slot) * (T::latent / 2);
          scales = sources::NativeAddress::scale_row(base, compressed_pool, uint64_t(page) * page_stride,
                                                     uint64_t(T::page), slot, T::latent);
          kind = 2;
        }
      }
    }
  }
  const uint64_t at = uint64_t(row) * T::max_keys + j;
  if (lane == 0) cached_valid[at] = Wrong && row == 0 ? 0 : kind;
  for (uint32_t sector = lane; sector < T::latent / 8; sector += 32) {
    const uint32_t d = sector * 8;
    uint32_t pairs[4] = {};
    if (kind == 1) {
      const uint64_t values = sources_scheduled::load_eight(codes + d);
      const auto scale = sources_decode::decode_scale<false>(scales[d / 32]);
#pragma unroll
      for (int pair = 0; pair < 4; ++pair)
        pairs[pair] = sources_decode::widen_scaled<false, false>(uint16_t(values >> (pair * 16)), scale);
    } else if (kind == 2) {
      const uint32_t values = sources_scheduled::load_four(codes + d / 2);
      const auto scale = sources_decode::decode_scale<true>(scales[d / 16]);
#pragma unroll
      for (int pair = 0; pair < 4; ++pair)
        pairs[pair] = sources_decode::widen_scaled<false, true>(uint8_t(values >> (pair * 8)), scale);
    }
    *reinterpret_cast<uint4*>(decoded + at * T::latent + d) = make_uint4(pairs[0], pairs[1], pairs[2], pairs[3]);
  }
}

template<class T, bool Profile>
__device__ void task(
    Index task, uint8_t* arena, uint64_t* stamps,
    const uint16_t* decoded, const int32_t* cached_valid,
    const uint16_t* q, const int32_t* row_batch, const uint32_t* live_rows,
    const uint8_t* window_codes, const uint8_t* window_scales,
    const int32_t* window_item_slots, const int32_t* window_keys,
    const uint8_t* compressed_pool, const int32_t* compressed_keys,
    const int32_t* block_tables, float* workspace,
    uint64_t window_code_stride, uint64_t window_scale_stride,
    uint32_t window_slots, uint32_t window_sequences, uint32_t window_items,
    uint32_t window_key_stride, uint32_t window_budget,
    uint64_t page_stride, uint32_t max_pages, uint32_t compressed_pages,
    uint32_t compressed_sequences, uint32_t compressed_key_stride,
    uint32_t compressed_budget, uint32_t scale_bits) {
  constexpr int StageKeys = 64;
  const uint32_t row = task.row, split = task.split;
  if (row >= *live_rows) return;
  const uint64_t started = Profile && threadIdx.x == 0 ? clock64() : 0;
  uint64_t last = started, cycles[6] = {};
  const int head0 = task.head;
  const uint32_t count = window_budget + (T::second ? compressed_budget : 0);
  const uint32_t units = (count + T::keys - 1) / T::keys;
  const uint32_t begin = uint64_t(units) * split / T::splits;
  const uint32_t end = uint64_t(units) * (split + 1) / T::splits;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int g = lane / 4, c = lane % 4;
  constexpr int terms = 1, column_blocks = 8;
  constexpr int stage_keys = StageKeys;
  constexpr int tile_bytes = stage_keys * T::latent * 2;
  constexpr int exchange_bytes = T::blocks * T::keys * T::head_tile * 4;
  uint8_t* tile = arena;
  float* exchange = reinterpret_cast<float*>(arena + tile_bytes);
  int32_t* valid = reinterpret_cast<int32_t*>(arena + tile_bytes + exchange_bytes);
  const uint32_t phase = (ptx::shared_address(tile) / 128) % 8;
  // The same B fragments as the reference ldmatrix query loads, without
  // retaining a shared query tile for the whole CTA.
  ptx::B2 query[8][T::n_tiles];
#pragma unroll
  for (int n = 0; n < T::n_tiles; ++n)
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      const int head = head0 + n * 8 + g;
      const uint16_t* h = q + (uint64_t(row) * T::heads + head) * T::latent
          + warp * 128 + ks * 16 + 2 * c;
      query[ks][n] = head < T::heads && head < head0 + task.head_count
          ? ptx::B2{{uint32_t(h[0]) | uint32_t(h[1]) << 16,
                      uint32_t(h[8]) | uint32_t(h[9]) << 16}}
          : ptx::B2{{0, 0}};
    }
  const float scale = __uint_as_float(scale_bits);
  float maximum[T::n_tiles], denominator[T::n_tiles];
  ptx::D4 output[column_blocks][T::n_tiles];
#pragma unroll
  for (int n = 0; n < T::n_tiles; ++n) {
    maximum[n] = -INFINITY;
    denominator[n] = 0.0f;
  }
#pragma unroll
  for (int m = 0; m < column_blocks; ++m)
#pragma unroll
    for (int n = 0; n < T::n_tiles; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) output[m][n].v[i] = 0.0f;

  mark<Profile>(cycles, 0, last);
  for (uint32_t unit = begin; unit < end; ++unit) {
    __syncthreads();
    bool any_key = false;
    for (int key = threadIdx.x; key < T::keys; key += T::threads) {
      const uint32_t j = unit * T::keys + key;
      const int kind = j < count ? cached_valid[uint64_t(row) * T::max_keys + j] : 0;
      valid[key] = kind;
      any_key |= kind != 0;
    }
    const bool any = __syncthreads_or(any_key);
    mark<Profile>(cycles, 1, last);
    if (!any) {
      float correction[T::n_tiles];
#pragma unroll
      for (int n = 0; n < T::n_tiles; ++n) {
        const float next = fmaxf(maximum[n], -INFINITY);
        const float base = next == -INFINITY ? 0.0f : next;
        correction[n] = exp_of(maximum[n] - base);
        maximum[n] = next;
        denominator[n] = fmaf(denominator[n], correction[n], 0.0f);
      }
      float rescale[T::n_tiles][2];
#pragma unroll
      for (int n = 0; n < T::n_tiles; ++n)
#pragma unroll
        for (int r = 0; r < 2; ++r)
          rescale[n][r] = __shfl_sync(0xffffffff, correction[n], (2 * c + r) * 4);
#pragma unroll
      for (int m = 0; m < column_blocks; ++m)
#pragma unroll
        for (int n = 0; n < T::n_tiles; ++n)
#pragma unroll
          for (int i = 0; i < 4; ++i)
            output[m][n].v[i] = fmaf(output[m][n].v[i], rescale[n][i % 2], 0.0f);
      mark<Profile>(cycles, 4, last);
      continue;
    }

    // Reuse the decoded tile for QK and PV. Source addresses
    // and validity still describe the original complete 64-key softmax tile.
    auto stage = [&](int first_key) {
      __syncthreads();
      // Eight copies per committed group keep the asynchronous queue bounded.
      constexpr int sectors = stage_keys * T::latent / 8;
      for (int base = 0; base < sectors; base += T::threads * 8) {
#pragma unroll
        for (int item = 0; item < 8; ++item) {
          const int i = base + threadIdx.x + item * T::threads;
          if (i < sectors) {
            const int key = first_key + i / (T::latent / 8);
            const int d = (i % (T::latent / 8)) * 8;
            const uint16_t* source = decoded
                + (uint64_t(row) * T::max_keys + unit * T::keys + key) * T::latent + d;
            attention::cp_async_zfill(tile + tile_offset<stage_keys>(key - first_key, d, phase), source, true);
          }
        }
        attention::cp_async_commit();
      }
      attention::cp_async_wait<0>();
      __syncthreads();
    };
    // QK: one chain from zero over this warp's 128 latent dims per key tile.
#pragma unroll
    for (int key_half = 0; key_half < T::keys / stage_keys; ++key_half) {
      stage(key_half * stage_keys);
      mark<Profile>(cycles, 2, last);
#pragma unroll
      for (int kt = key_half * (stage_keys / 16); kt < (key_half + 1) * (stage_keys / 16); ++kt) {
        ptx::D4 chain[T::n_tiles];
#pragma unroll
        for (int n = 0; n < T::n_tiles; ++n)
#pragma unroll
          for (int i = 0; i < 4; ++i) chain[n].v[i] = 0.0f;
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          uint32_t a[4];
          ptx::ldmatrix_b16<4>(a, tile + tile_offset<stage_keys>(
              (kt % (stage_keys / 16)) * 16 + ((lane / 8) % 2) * 8 + lane % 8,
              warp * 128 + ks * 16 + (lane / 16) * 8, phase));
#pragma unroll
          for (int n = 0; n < T::n_tiles; ++n)
            ptx::mma_bf16(chain[n], ptx::A4{{a[0], a[1], a[2], a[3]}}, query[ks][n]);
        }
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int key = kt * 16 + g + half * 8;
#pragma unroll
          for (int n = 0; n < T::n_tiles; ++n) {
            float2 pair = make_float2(chain[n].v[2 * half], chain[n].v[2 * half + 1]);
            *reinterpret_cast<float2*>(
                exchange + (warp * T::keys + key) * T::head_tile + n * 8 + 2 * c) = pair;
          }
        }
      }
      __syncthreads();
      mark<Profile>(cycles, 3, last);
    }

    // Scores of keys kt * 16 + {2c, 2c + 1, 2c + 8, 2c + 9} for head
    // n * 8 + g: the block chains added in block order, times the scale.
    float p[T::key_tiles][T::n_tiles][4];
    float correction[T::n_tiles];
#pragma unroll
    for (int n = 0; n < T::n_tiles; ++n) {
      float tile_max = -INFINITY;
#pragma unroll
      for (int kt = 0; kt < T::key_tiles; ++kt)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int key = kt * 16 + 2 * c + (i % 2) + (i / 2) * 8;
          const float* chains = exchange + key * T::head_tile + n * 8 + g;
          float sum = chains[0];
#pragma unroll
          for (int w = 1; w < T::warps; ++w) sum += chains[w * T::keys * T::head_tile];
          const float s = valid[key] ? sum * scale : -INFINITY;
          p[kt][n][i] = s;
          tile_max = fmaxf(tile_max, s);
        }
      tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 1));
      tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 2));
      const float next = fmaxf(maximum[n], tile_max);
      const float base = next == -INFINITY ? 0.0f : next;
      correction[n] = exp_of(maximum[n] - base);
      maximum[n] = next;
      float sum = 0.0f;
#pragma unroll
      for (int kt = 0; kt < T::key_tiles; ++kt)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          p[kt][n][i] = exp_of(p[kt][n][i] - base);
          sum += p[kt][n][i];
        }
      denominator[n] = fmaf(denominator[n], correction[n], sum);
    }

    // P^T as B fragments: k = keys 2c, 2c + 1 (b0) and 2c + 8, 2c + 9 (b1),
    // n = head g, each value carried as `terms` BF16 parts.
    uint32_t probability[terms][T::key_tiles][T::n_tiles][2];
#pragma unroll
    for (int kt = 0; kt < T::key_tiles; ++kt)
#pragma unroll
      for (int n = 0; n < T::n_tiles; ++n)
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          float p0 = p[kt][n][2 * half];
          float p1 = p[kt][n][2 * half + 1];
#pragma unroll
          for (int term = 0; term < terms; ++term) {
            const uint32_t pair = attention::bf16_pair(p0, p1);
            p0 -= attention::bf16_low(pair);
            p1 -= attention::bf16_high(pair);
            probability[term][kt][n][half] = pair;
          }
        }
    // The rescale of output column 2c + r comes from the lanes of head
    // g = 2c + r.
    float rescale[T::n_tiles][2];
#pragma unroll
    for (int n = 0; n < T::n_tiles; ++n)
#pragma unroll
      for (int r = 0; r < 2; ++r)
        rescale[n][r] = __shfl_sync(0xffffffff, correction[n], (2 * c + r) * 4);

    mark<Profile>(cycles, 4, last);
    // Retain each column's chain across the two staged key halves. Every
    // chain still starts at zero and executes kt=0,1,2,3 in that order.
    ptx::D4 chain[column_blocks][T::n_tiles];
#pragma unroll
    for (int m = 0; m < column_blocks; ++m)
#pragma unroll
      for (int n = 0; n < T::n_tiles; ++n)
#pragma unroll
        for (int i = 0; i < 4; ++i) chain[m][n].v[i] = 0.0f;
#pragma unroll
    for (int key_half = 0; key_half < T::keys / stage_keys; ++key_half) {
      if constexpr (StageKeys != T::keys) stage(key_half * stage_keys);
      mark<Profile>(cycles, 2, last);
#pragma unroll
      for (int m = 0; m < column_blocks; ++m)
#pragma unroll
        for (int kt = key_half * (stage_keys / 16); kt < (key_half + 1) * (stage_keys / 16); ++kt) {
          uint32_t a[4];
          ptx::ldmatrix_b16<4, true>(a, tile + tile_offset<stage_keys>(
              (kt % (stage_keys / 16)) * 16 + (lane / 16) * 8 + lane % 8,
              warp * 128 + m * 16 + ((lane / 8) % 2) * 8, phase));
#pragma unroll
          for (int n = 0; n < T::n_tiles; ++n)
#pragma unroll
            for (int term = 0; term < terms; ++term)
              ptx::mma_bf16(chain[m][n], ptx::A4{{a[0], a[1], a[2], a[3]}},
                            ptx::B2{{probability[term][kt][n][0], probability[term][kt][n][1]}});
        }
      __syncthreads();
      mark<Profile>(cycles, 5, last);
    }
#pragma unroll
    for (int m = 0; m < column_blocks; ++m)
#pragma unroll
      for (int n = 0; n < T::n_tiles; ++n)
#pragma unroll
        for (int i = 0; i < 4; ++i)
          output[m][n].v[i] = fmaf(output[m][n].v[i], rescale[n][i % 2], chain[m][n].v[i]);
  }

  // Each head's denominator has four lane contributions. Every split
  // writes a complete partial, including empty splits, for poison replay.
#pragma unroll
  for (int n = 0; n < T::n_tiles; ++n) {
    denominator[n] += __shfl_xor_sync(0xffffffff, denominator[n], 1);
    denominator[n] += __shfl_xor_sync(0xffffffff, denominator[n], 2);
  }
#pragma unroll
  for (int n = 0; n < T::n_tiles; ++n)
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int head = head0 + n * 8 + 2 * c + r;
      const int source = (2 * c + r) * 4;
      const float m_head = __shfl_sync(0xffffffff, maximum[n], source);
      const float l_head = __shfl_sync(0xffffffff, denominator[n], source);
      if (head >= T::heads || head >= head0 + task.head_count) continue;
      float* dest = workspace + partial_index<T>(row, split, head);
      if (warp == 0 && g == 0) { dest[0] = m_head; dest[1] = l_head; }
#pragma unroll
      for (int m = 0; m < column_blocks; ++m)
#pragma unroll
        for (int half = 0; half < 2; ++half)
          dest[2 + warp * 128 + m * 16 + g + half * 8] = output[m][n].v[2 * half + r];
    }
  if constexpr (Profile) {
    if (threadIdx.x == 0) {
      const uint64_t at = ((uint64_t(row) * T::heads + task.head) * T::splits + split) * 8;
      for (int i = 0; i < 6; ++i) stamps[at + i] = cycles[i];
      stamps[at + 6] = clock64() - started;
      stamps[at + 7] = end - begin;
    }
  }

}



} // namespace chop::latent::sources_cached
