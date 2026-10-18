#pragma once

// Attention over an exact-decoded FP8 window and paged FP4 keys. This
// engine has its own source closure; the original latent entries keep
// their kernels and launch ABI. QK uses 128-channel chains from zero.

#include <math.h>
#include <stdint.h>
#include "../attention/tile.cuh"

namespace chop::latent::sources {

using attention::tile_offset;
__device__ __forceinline__ float exp_of(float x) {
  return exp2f(x * 1.4426950408889634f);
}

template<int Heads, int Latent, int Page, int MaxKeys, int Splits,
         bool Second, bool Sink, int Tail, bool Partial>
struct Attention {
  static_assert(Heads > 0 && Heads <= 128);
  static_assert(Latent > 0 && Latent % 128 == 0 && Latent <= 512);
  static_assert(Page > 0 && (Page & (Page - 1)) == 0);
  static_assert(Splits > 0 && Splits <= 64);
  static_assert(Tail == 0 || Tail == 64);
  static constexpr int heads = Heads, latent = Latent, page = Page, max_keys = MaxKeys;
  static constexpr int splits = Splits, tail = Tail;
  static constexpr bool second = Second, sink = Sink, partial = Partial;
  static constexpr int head_tile = 8, n_tiles = 1, keys = 64, key_tiles = 4;
  static constexpr int blocks = Latent / 128, warps = blocks, threads = warps * 32;
  static constexpr int tile_bytes = keys * Latent * 2;
  static constexpr int q_offset = tile_bytes;
  static constexpr int exchange_offset = q_offset + head_tile * Latent * 2;
  static constexpr int valid_offset = exchange_offset + blocks * keys * head_tile * 4;
  static constexpr int address_offset = valid_offset + keys * 4;
  static constexpr int smem = address_offset + keys * 2 * 8;
  static constexpr int partial_width = Latent + 2;
  static_assert(smem <= 101376);
};

template<class T>
__device__ __forceinline__ uint64_t partial_index(uint32_t row, uint32_t split, uint32_t head) {
  return ((uint64_t(row) * T::splits + split) * T::heads + head) * T::partial_width;
}

struct NativeAddress {
  __device__ __forceinline__ static const uint8_t* scale_row(
      const uint8_t* base, const uint8_t*, uint64_t,
      uint64_t rows, uint32_t slot, uint32_t width) {
    return base + rows * (width / 2) + uint64_t(slot) * (width / 16);
  }
};

// One CTA per row, 8-head group and fixed split. Both lists contribute to
// a single 64-key tile stream. Holes neither read codes nor read scales.
template<class T, class Address = NativeAddress, bool Bf16 = false>
__device__ void attention(
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
  ptx::griddep_launch_dependents();
  extern __shared__ __align__(1024) uint8_t native_latent_smem[];
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x, split = blockIdx.z;
  if (row >= *live_rows) return;
  const int head0 = blockIdx.y * T::head_tile;
  const int batch = row_batch[row];
  const int state_slot = batch >= 0 && uint32_t(batch) < window_items
      ? window_item_slots[batch] : -1;
  const uint32_t count = window_budget + (T::second ? compressed_budget : 0);
  const uint32_t units = (count + T::keys - 1) / T::keys;
  const uint32_t begin = uint64_t(units) * split / T::splits;
  const uint32_t end = uint64_t(units) * (split + 1) / T::splits;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int g = lane / 4, c = lane % 4;
  constexpr int terms = 1, column_blocks = 8;
  uint8_t* tile = native_latent_smem;
  uint8_t* q_smem = native_latent_smem + T::q_offset;
  float* exchange = reinterpret_cast<float*>(native_latent_smem + T::exchange_offset);
  int32_t* valid = reinterpret_cast<int32_t*>(native_latent_smem + T::valid_offset);
  const uint8_t** code_rows = reinterpret_cast<const uint8_t**>(native_latent_smem + T::address_offset);
  const uint8_t** scale_rows = code_rows + T::keys;
  const uint32_t phase = (ptx::shared_address(tile) / 128) % 8;
  const uint32_t q_phase = (ptx::shared_address(q_smem) / 128) % 8;
  for (int i = threadIdx.x; i < T::head_tile * T::latent; i += T::threads) {
    const int h = i / T::latent, d = i % T::latent;
    const uint16_t value = head0 + h < T::heads
        ? q[(uint64_t(row) * T::heads + head0 + h) * T::latent + d] : 0;
    *reinterpret_cast<uint16_t*>(q_smem + tile_offset<T::head_tile>(h, d, q_phase) + (d % 8) * 2) = value;
  }
  __syncthreads();
  ptx::B2 query[8][T::n_tiles];
#pragma unroll
  for (int n = 0; n < T::n_tiles; ++n)
#pragma unroll
    for (int ks = 0; ks < 8; ks += 2) {
      uint32_t b[4];
      ptx::ldmatrix_b16<4>(b, q_smem + tile_offset<T::head_tile>(
          n * 8 + lane % 8, warp * 128 + ks * 16 + (lane / 8) * 8, q_phase));
      query[ks][n] = ptx::B2{{b[0], b[1]}};
      query[ks + 1][n] = ptx::B2{{b[2], b[3]}};
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

  for (uint32_t unit = begin; unit < end; ++unit) {
    __syncthreads();
    for (int key = threadIdx.x; key < T::keys; key += T::threads) {
      const uint32_t j = unit * T::keys + key;
      int kind = 0;
      const uint8_t *codes = nullptr, *scales = nullptr;
      if (j < window_budget && state_slot >= 0 && uint32_t(state_slot) < window_sequences) {
        const int32_t position = window_keys[uint64_t(row) * window_key_stride + j];
        if (position >= 0 && uint32_t(position) < window_slots) {
          codes = window_codes + uint64_t(state_slot) * window_code_stride + uint64_t(position) * T::latent * (Bf16 ? 2 : 1);
          if constexpr (!Bf16) scales = window_scales + uint64_t(state_slot) * window_scale_stride + uint64_t(position) * (T::latent / 32);
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
              codes = base + uint64_t(slot) * (Bf16 ? T::latent * 2 : T::latent / 2);
              if constexpr (!Bf16) scales = Address::scale_row(base, compressed_pool, uint64_t(page) * page_stride, uint64_t(T::page), slot, T::latent);
              kind = 2;
            }
          }
        }
      }
      code_rows[key] = codes;
      scale_rows[key] = scales;
      valid[key] = kind;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < T::keys * T::latent / 2; i += T::threads) {
      const int key = i / (T::latent / 2), d = (i % (T::latent / 2)) * 2;
      uint32_t pair = 0;
      if constexpr (Bf16) {
        if (valid[key] != 0) {
          const auto* values = reinterpret_cast<const uint16_t*>(code_rows[key]);
          pair = uint32_t(values[d]) | uint32_t(values[d + 1]) << 16;
        }
      } else if (valid[key] == 1) {
        const auto code = *reinterpret_cast<const uint16_t*>(code_rows[key] + d);
        const auto value = convert::unpack_f32_e4m3(code);
        const float s = convert::unpack_f32_ue8m0(scale_rows[key][d / 32]).lo;
        pair = attention::bf16_pair(value.lo * s, value.hi * s);
      } else if (valid[key] == 2) {
        const auto value = convert::unpack_f32_e2m1(code_rows[key][d / 2]);
        const float s = convert::unpack_f32_e4m3(scale_rows[key][d / 16]).lo;
        pair = attention::bf16_pair(value.lo * s, value.hi * s);
      }
      *reinterpret_cast<uint32_t*>(tile + tile_offset<T::keys>(key, d, phase) + (d % 8) * 2) = pair;
    }
    __syncthreads();
    // QK: one chain from zero over this warp's 128 latent dims per key tile.
#pragma unroll
    for (int kt = 0; kt < T::key_tiles; ++kt) {
      ptx::D4 chain[T::n_tiles];
#pragma unroll
      for (int n = 0; n < T::n_tiles; ++n)
#pragma unroll
        for (int i = 0; i < 4; ++i) chain[n].v[i] = 0.0f;
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        uint32_t a[4];
        ptx::ldmatrix_b16<4>(a, tile + tile_offset<T::keys>(
            kt * 16 + ((lane / 8) % 2) * 8 + lane % 8,
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

    // PV: one chain from zero over the unit's keys per 16 columns, added
    // into the rescaled output.
#pragma unroll
    for (int m = 0; m < column_blocks; ++m) {
      ptx::D4 chain[T::n_tiles];
#pragma unroll
      for (int n = 0; n < T::n_tiles; ++n)
#pragma unroll
        for (int i = 0; i < 4; ++i) chain[n].v[i] = 0.0f;
#pragma unroll
      for (int kt = 0; kt < T::key_tiles; ++kt) {
        uint32_t a[4];
        ptx::ldmatrix_b16<4, true>(a, tile + tile_offset<T::keys>(
            kt * 16 + (lane / 16) * 8 + lane % 8,
            warp * 128 + m * 16 + ((lane / 8) % 2) * 8, phase));
#pragma unroll
        for (int n = 0; n < T::n_tiles; ++n)
#pragma unroll
          for (int term = 0; term < terms; ++term)
            ptx::mma_bf16(chain[n], ptx::A4{{a[0], a[1], a[2], a[3]}},
                          ptx::B2{{probability[term][kt][n][0], probability[term][kt][n][1]}});
      }
#pragma unroll
      for (int n = 0; n < T::n_tiles; ++n)
#pragma unroll
        for (int i = 0; i < 4; ++i)
          output[m][n].v[i] = fmaf(output[m][n].v[i], rescale[n][i % 2], chain[n].v[i]);
    }
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
      if (head >= T::heads) continue;
      float* dest = workspace + partial_index<T>(row, split, head);
      if (warp == 0 && g == 0) { dest[0] = m_head; dest[1] = l_head; }
#pragma unroll
      for (int m = 0; m < column_blocks; ++m)
#pragma unroll
        for (int half = 0; half < 2; ++half)
          dest[2 + warp * 128 + m * 16 + g + half * 8] = output[m][n].v[2 * half + r];
    }
}

// The final merge is the only sink owner. Rotation follows the mandatory
// BF16 store boundary and uses adjacent channel pairs, never extra dims.
template<class T, bool SinkPerSplit = false>
__device__ void merge(const float* workspace, const uint32_t* live_rows,
    const float* sinks, const float* cos_sin, uint16_t* out,
    float* out_max, float* out_den, float* out_num) {
  ptx::griddep_launch_dependents();
  __shared__ float weights[T::splits];
  __shared__ float totals[2];
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x, head = blockIdx.y;
  if (row >= *live_rows) return;
  auto part = [&](int split) { return workspace + partial_index<T>(row, split, head); };
  if (threadIdx.x == 0) {
    float top = -INFINITY;
    for (int split = 0; split < T::splits; ++split) top = fmaxf(top, part(split)[0]);
    float sink = -INFINITY;
    if constexpr (T::sink && !T::partial) { sink = sinks[head]; top = fmaxf(top, sink); }
    float den = 0.0f;
    for (int split = 0; split < T::splits; ++split) {
      weights[split] = part(split)[0] == -INFINITY ? 0.0f : exp_of(part(split)[0] - top);
      den = fmaf(weights[split], part(split)[1], den);
    }
    if constexpr (T::sink && !T::partial)
      den += exp_of(sink - top) * (SinkPerSplit ? float(T::splits) : 1.0f);
    totals[0] = top; totals[1] = den;
  }
  __syncthreads();
  const uint64_t pair = uint64_t(row) * T::heads + head;
  if constexpr (T::partial) {
    if (threadIdx.x == 0) { out_max[pair] = totals[0]; out_den[pair] = totals[1]; }
  }
  for (int d = threadIdx.x * 2; d < T::latent; d += blockDim.x * 2) {
    float x = 0.0f, y = 0.0f;
    for (int split = 0; split < T::splits; ++split) {
      x = fmaf(weights[split], part(split)[2 + d], x);
      y = fmaf(weights[split], part(split)[3 + d], y);
    }
    if constexpr (T::partial) {
      out_num[pair * T::latent + d] = x;
      out_num[pair * T::latent + d + 1] = y;
    } else {
      const float den = totals[1];
      const uint32_t rounded = attention::bf16_pair(den == 0.0f ? 0.0f : __fdiv_rn(x, den),
                                                   den == 0.0f ? 0.0f : __fdiv_rn(y, den));
      uint32_t value = rounded;
      if constexpr (T::tail != 0) {
        if (d >= T::latent - T::tail) {
          const float c = cos_sin[uint64_t(row) * T::tail + d - (T::latent - T::tail)];
          const float s = cos_sin[uint64_t(row) * T::tail + d - (T::latent - T::tail) + 1];
          const float a = attention::bf16_low(rounded), b = attention::bf16_high(rounded);
          value = attention::bf16_pair(fmaf(a, c, b * s), fmaf(b, c, -a * s));
        }
      }
      *reinterpret_cast<uint32_t*>(out + pair * T::latent + d) = value;
    }
  }
}

} // namespace chop::latent::sources
