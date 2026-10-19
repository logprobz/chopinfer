#pragma once

// Prefill staging candidate. The reference already uses BF16 tensor cores.
// Keep its 64-key softmax tiles, 128-channel QK chains, PV key order and
// partial ABI. Stage 32 decoded keys at a time and reread them for PV to
// halve the shared tile. The final merge is the unchanged reference entry.
#include "sources.cuh"

namespace chop::latent::sources_prefill {
using sources::NativeAddress;
using sources::partial_index;
using sources::exp_of;
using attention::tile_offset;

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
  constexpr int stage_keys = 32;
  constexpr int tile_bytes = stage_keys * T::latent * 2;
  constexpr int exchange_bytes = T::blocks * T::keys * T::head_tile * 4;
  uint8_t* tile = native_latent_smem;
  float* exchange = reinterpret_cast<float*>(native_latent_smem + tile_bytes);
  int32_t* valid = reinterpret_cast<int32_t*>(native_latent_smem + tile_bytes + exchange_bytes);
  const uint8_t** code_rows = reinterpret_cast<const uint8_t**>(valid + T::keys);
  const uint8_t** scale_rows = code_rows + T::keys;
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
      query[ks][n] = head < T::heads
          ? ptx::B2{{*reinterpret_cast<const uint32_t*>(h),
                      *reinterpret_cast<const uint32_t*>(h + 8)}}
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
    // Reuse the half-size decoded tile for QK and PV. Source addresses
    // and validity still describe the original complete 64-key softmax tile.
    auto stage = [&](int first_key) {
      __syncthreads();
      for (int i = threadIdx.x; i < stage_keys * T::latent / 2; i += T::threads) {
        const int local_key = i / (T::latent / 2), key = first_key + local_key, d = (i % (T::latent / 2)) * 2;
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
        *reinterpret_cast<uint32_t*>(tile + tile_offset<stage_keys>(local_key, d, phase) + (d % 8) * 2) = pair;
      }
      __syncthreads();
    };
    // QK: one chain from zero over this warp's 128 latent dims per key tile.
#pragma unroll
    for (int key_half = 0; key_half < 2; ++key_half) {
      stage(key_half * stage_keys);
#pragma unroll
      for (int kt = key_half * 2; kt < key_half * 2 + 2; ++kt) {
        ptx::D4 chain[T::n_tiles];
#pragma unroll
        for (int n = 0; n < T::n_tiles; ++n)
#pragma unroll
          for (int i = 0; i < 4; ++i) chain[n].v[i] = 0.0f;
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          uint32_t a[4];
          ptx::ldmatrix_b16<4>(a, tile + tile_offset<stage_keys>(
              (kt % 2) * 16 + ((lane / 8) % 2) * 8 + lane % 8,
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
    for (int key_half = 0; key_half < 2; ++key_half) {
      stage(key_half * stage_keys);
#pragma unroll
      for (int m = 0; m < column_blocks; ++m)
#pragma unroll
        for (int kt = key_half * 2; kt < key_half * 2 + 2; ++kt) {
          uint32_t a[4];
          ptx::ldmatrix_b16<4, true>(a, tile + tile_offset<stage_keys>(
              (kt % 2) * 16 + (lane / 16) * 8 + lane % 8,
              warp * 128 + m * 16 + ((lane / 8) % 2) * 8, phase));
#pragma unroll
          for (int n = 0; n < T::n_tiles; ++n)
#pragma unroll
            for (int term = 0; term < terms; ++term)
              ptx::mma_bf16(chain[m][n], ptx::A4{{a[0], a[1], a[2], a[3]}},
                            ptx::B2{{probability[term][kt][n][0], probability[term][kt][n][1]}});
        }
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


// Lab-only software profiling. Timed launches pass no stamp buffer; a
// separate sampled launch records warp zero's elapsed stage cycles.
enum class ProfileMode { Full, Loads, Resident };

__device__ __forceinline__ void profile_clock(bool sample, uint64_t* totals,
                                             int stage, uint64_t& last) {
  if (sample) {
    const uint64_t now = clock64();
    totals[stage] += now - last;
    last = now;
  }
}

template<ProfileMode Mode, class T, class Address = NativeAddress, bool Bf16 = false>
__device__ void profile_attention(
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
    uint32_t compressed_budget, uint32_t scale_bits, uint64_t* stamps) {
  ptx::griddep_launch_dependents();
  extern __shared__ __align__(1024) uint8_t native_latent_smem[];
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x, split = blockIdx.z;
  if (row >= *live_rows) return;
  const bool sample = stamps != nullptr && threadIdx.x == 0;
  const uint64_t started = sample ? clock64() : 0;
  uint64_t last = started, cycles[6] = {};
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
  ptx::B2 query[8][T::n_tiles];
  if constexpr (Mode != ProfileMode::Loads) {
  for (int i = threadIdx.x; i < T::head_tile * T::latent; i += T::threads) {
    const int h = i / T::latent, d = i % T::latent;
    const uint16_t value = head0 + h < T::heads
        ? q[(uint64_t(row) * T::heads + head0 + h) * T::latent + d] : 0;
    *reinterpret_cast<uint16_t*>(q_smem + tile_offset<T::head_tile>(h, d, q_phase) + (d % 8) * 2) = value;
  }
  __syncthreads();
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

  profile_clock(sample, cycles, 0, last);
  for (uint32_t unit = begin; unit < end; ++unit) {
    __syncthreads();
    // Resident is a diagnostic workload: load the first 64-key tile once,
    // then reuse it for the original number of math iterations. It is not
    // an attention implementation and is never registered for serving.
    if (Mode != ProfileMode::Resident || unit == begin) {
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
      if constexpr (Mode == ProfileMode::Loads) {
        // Volatile stores retain every gather/decode/stage iteration when
        // no later MMA reads the shared tile. No checksum ALU is added.
        *reinterpret_cast<volatile uint32_t*>(tile + tile_offset<T::keys>(key, d, phase) + (d % 8) * 2) = pair;
      } else {
        *reinterpret_cast<uint32_t*>(tile + tile_offset<T::keys>(key, d, phase) + (d % 8) * 2) = pair;
      }
    }
    __syncthreads();
    }
    profile_clock(sample, cycles, 1, last);
    if constexpr (Mode == ProfileMode::Loads) continue;
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

    profile_clock(sample, cycles, 2, last);

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

    profile_clock(sample, cycles, 3, last);

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
    profile_clock(sample, cycles, 4, last);
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
  profile_clock(sample, cycles, 5, last);
  if (sample) {
    uint32_t sm;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    const uint64_t cta = (uint64_t(row) * ((T::heads + 7) / 8) + blockIdx.y) * T::splits + split;
    uint64_t* record = stamps + cta * 9;
    record[0] = started; record[1] = last; record[2] = sm;
#pragma unroll
    for (int stage = 0; stage < 6; ++stage) record[3 + stage] = cycles[stage];
  }

}



// Eight decoded dimensions per thread make each shared store a full
// 16-byte sector. Keep the reference's F32 scale product and BF16 rounding.
struct Gather8 {
  uint4 bits;
  float scale;
  int kind;
};

template<bool Bf16>
__device__ __forceinline__ Gather8 load8(
    const uint8_t* codes, const uint8_t* scales, int kind, int dim) {
  Gather8 packet{make_uint4(0, 0, 0, 0), 0.0f, kind};
  if (kind == 0) return packet;
  if constexpr (Bf16) {
    const auto* values = reinterpret_cast<const uint16_t*>(codes) + dim;
    if ((reinterpret_cast<uintptr_t>(values) & 15) == 0) {
      packet.bits = *reinterpret_cast<const uint4*>(values);
    } else {
      uint32_t words[4];
#pragma unroll
      for (int i = 0; i < 4; ++i)
        words[i] = uint32_t(values[2 * i]) | uint32_t(values[2 * i + 1]) << 16;
      packet.bits = make_uint4(words[0], words[1], words[2], words[3]);
    }
  } else if (kind == 1) {
    const uint8_t* values = codes + dim;
    if ((reinterpret_cast<uintptr_t>(values) & 7) == 0) {
      const uint2 raw = *reinterpret_cast<const uint2*>(values);
      packet.bits.x = raw.x; packet.bits.y = raw.y;
    } else {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        packet.bits.x |= uint32_t(values[i]) << (8 * i);
        packet.bits.y |= uint32_t(values[i + 4]) << (8 * i);
      }
    }
    packet.scale = convert::unpack_f32_ue8m0(scales[dim / 32]).lo;
  } else {
    const uint8_t* values = codes + dim / 2;
    if ((reinterpret_cast<uintptr_t>(values) & 3) == 0) {
      packet.bits.x = *reinterpret_cast<const uint32_t*>(values);
    } else {
#pragma unroll
      for (int i = 0; i < 4; ++i) packet.bits.x |= uint32_t(values[i]) << (8 * i);
    }
    packet.scale = convert::unpack_f32_e4m3(scales[dim / 16]).lo;
  }
  return packet;
}

template<bool Bf16>
__device__ __forceinline__ uint4 decode8(const Gather8& packet) {
  if constexpr (Bf16) return packet.bits;
  uint32_t pairs[4] = {};
  if (packet.kind != 0) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const auto value = packet.kind == 1
          ? convert::unpack_f32_e4m3(uint16_t((i < 2 ? packet.bits.x : packet.bits.y) >> ((i % 2) * 16)))
          : convert::unpack_f32_e2m1(uint8_t(packet.bits.x >> (i * 8)));
      pairs[i] = attention::bf16_pair(value.lo * packet.scale, value.hi * packet.scale);
    }
  }
  return make_uint4(pairs[0], pairs[1], pairs[2], pairs[3]);
}

// Consecutive query rows share a decoded tile only after every source
// address, scale address and validity tag matches. Differing selections
// use two passes through the same tile. Each row executes one unchanged
// 64-key QK/softmax/PV unit, in its original unit order.
template<int Rows, class T, class Address = NativeAddress, bool Bf16 = false>
__device__ void reuse_attention(
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
  static_assert(Rows == 1 || Rows == 2);
  constexpr int storage_keys = T::keys + Rows - 1;
  constexpr int tile_bytes = storage_keys * T::latent * 2;
  constexpr int exchange_words = T::blocks * T::keys * T::head_tile;
  const uint32_t live = *live_rows, first_row = blockIdx.x * Rows, split = blockIdx.z;
  if (first_row >= live) return;
  const int group = threadIdx.x / T::threads, tid = threadIdx.x % T::threads;
  const uint32_t row = first_row + group;
  const bool row_live = row < live;
  const int head0 = blockIdx.y * T::head_tile;
  const int batch = row_live ? row_batch[row] : -1;
  const int state_slot = batch >= 0 && uint32_t(batch) < window_items
      ? window_item_slots[batch] : -1;
  const uint32_t count = window_budget + (T::second ? compressed_budget : 0);
  const uint32_t units = (count + T::keys - 1) / T::keys;
  const uint32_t begin = uint64_t(units) * split / T::splits;
  const uint32_t end = uint64_t(units) * (split + 1) / T::splits;
  const int warp = tid / 32, lane = tid % 32;
  const int g = lane / 4, c = lane % 4;
  constexpr int terms = 1, column_blocks = 8;
  uint8_t* tile = native_latent_smem;
  float* exchange_all = reinterpret_cast<float*>(tile + tile_bytes);
  float* exchange = exchange_all + group * exchange_words;
  int32_t* valid_all = reinterpret_cast<int32_t*>(exchange_all + Rows * exchange_words);
  const uint8_t** codes_all = reinterpret_cast<const uint8_t**>(valid_all + Rows * T::keys);
  const uint8_t** scales_all = codes_all + Rows * T::keys;
  int32_t* valid = valid_all + group * T::keys;
  const uint8_t** code_rows = codes_all + group * T::keys;
  const uint8_t** scale_rows = scales_all + group * T::keys;
  const uint32_t phase = (ptx::shared_address(tile) / 128) % 8;
  ptx::B2 query[8][T::n_tiles];
#pragma unroll
  for (int n = 0; n < T::n_tiles; ++n)
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      const int head = head0 + n * 8 + g;
      const uint16_t* h = row_live && head < T::heads ? q + (uint64_t(row) * T::heads + head) * T::latent
          + warp * 128 + ks * 16 + 2 * c : q;
      query[ks][n] = row_live && head < T::heads
          ? ptx::B2{{*reinterpret_cast<const uint32_t*>(h),
                      *reinterpret_cast<const uint32_t*>(h + 8)}}
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

  for (uint32_t unit = begin; unit < end; ++unit) {
    __syncthreads();
    for (int key = tid; key < T::keys; key += T::threads) {
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
    int shift = 0;
    if constexpr (Rows == 2) {
      const int key = threadIdx.x;
      const bool same_bad = key < T::keys &&
          (valid_all[key] != valid_all[T::keys + key] ||
           codes_all[key] != codes_all[T::keys + key] ||
           scales_all[key] != scales_all[T::keys + key]);
      const bool shifted_bad = key < T::keys - 1 &&
          (valid_all[key + 1] != valid_all[T::keys + key] ||
           codes_all[key + 1] != codes_all[T::keys + key] ||
           scales_all[key + 1] != scales_all[T::keys + key]);
      const int same = __syncthreads_count(same_bad);
      const int shifted = __syncthreads_count(shifted_bad);
      shift = same == 0 ? 0 : shifted == 0 ? 1 : -1;
    }
    const int passes = shift >= 0 ? 1 : Rows;
    for (int pass = 0; pass < passes; ++pass) {
      const int keys = T::keys + (shift == 1 ? 1 : 0);
      const int values = keys * (T::latent / 8);
      auto gather = [&](int i) {
        const int key = i / (T::latent / 8), dim = (i % (T::latent / 8)) * 8;
        const int source = shift < 0 ? pass * T::keys + key
            : key < T::keys ? key : T::keys * 2 - 1;
        return load8<Bf16>(codes_all[source], scales_all[source], valid_all[source], dim);
      };
      // Two register packets let the next coalesced load progress while
      // the current packet decodes and writes a 16-byte shared sector.
      int i = threadIdx.x;
      Gather8 current = gather(i);
      for (; i < values; i += Rows * T::threads) {
        Gather8 next{};
        if (i + Rows * T::threads < values) next = gather(i + Rows * T::threads);
        const int key = i / (T::latent / 8), dim = (i % (T::latent / 8)) * 8;
        *reinterpret_cast<uint4*>(tile + tile_offset<storage_keys>(key, dim, phase)) = decode8<Bf16>(current);
        current = next;
      }
      __syncthreads();
      const bool active = row_live && (shift >= 0 || group == pass);
      const int key_base = shift >= 0 ? group * shift : 0;
      if (active) {
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
            ptx::ldmatrix_b16<4>(a, tile + tile_offset<storage_keys>(
                key_base + kt * 16 + ((lane / 8) % 2) * 8 + lane % 8,
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
      }
      __syncthreads();
      if (active) {
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
            ptx::ldmatrix_b16<4, true>(a, tile + tile_offset<storage_keys>(
                key_base + kt * 16 + (lane / 16) * 8 + lane % 8,
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
      __syncthreads();
    }
  }
  if (!row_live) return;

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

} // namespace chop::latent::sources_prefill
