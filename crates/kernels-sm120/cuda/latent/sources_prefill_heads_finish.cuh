#pragma once
#include "sources_prefill_heads_async.cuh"
#include "sources_prefill_finish.cuh"
namespace chop::latent::sources_prefill {
template<class T, class Address = NativeAddress, bool Bf16 = false>
__device__ void head_shared_attention_finish(
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
    uint32_t compressed_budget, uint32_t scale_bits, const float* sinks, const float* cos_sin, uint16_t* out) {
  ptx::griddep_launch_dependents();
  extern __shared__ __align__(1024) uint8_t native_latent_smem[];
  ptx::griddep_wait();
  static_assert(T::latent == 512 && T::heads % 16 == 0 && T::n_tiles == 1);
  constexpr int storage_keys = T::keys;
  constexpr int tile_bytes = storage_keys * T::latent * 2;
  constexpr int exchange_words = T::blocks * T::keys * T::head_tile;
  const uint32_t live = *live_rows, first_row = blockIdx.x, split = blockIdx.z;
  if (first_row >= live) return;
  const int group = threadIdx.x / T::threads, tid = threadIdx.x % T::threads;
  const uint32_t row = first_row;
  const bool row_live = row < live;
  const int head0 = blockIdx.y * 16 + group * T::head_tile;
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
  int32_t* valid_all = reinterpret_cast<int32_t*>(exchange_all + 2 * exchange_words);
  const uint8_t** codes_all = reinterpret_cast<const uint8_t**>(valid_all + T::keys);
  const uint8_t** scales_all = codes_all + T::keys;
  int32_t* valid = valid_all;
  const uint8_t** code_rows = codes_all;
  const uint8_t** scale_rows = scales_all;
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
    for (int key = threadIdx.x; key < T::keys; key += 2 * T::threads) {
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
    // Both eight-head groups read the same ordered 64-key unit.
    {
      const int values = T::keys * (T::latent / 8);
      auto gather = [&](int i) {
        const int key = i / (T::latent / 8), dim = (i % (T::latent / 8)) * 8;
        return load8<Bf16>(codes_all[key], scales_all[key], valid_all[key], dim);
      };
      // Two register packets let the next coalesced load progress while
      // the current packet decodes and writes a 16-byte shared sector.
      int i = threadIdx.x;
      Gather8 current = gather(i);
      for (; i < values; i += 2 * T::threads) {
        Gather8 next{};
        if (i + 2 * T::threads < values) next = gather(i + 2 * T::threads);
        const int key = i / (T::latent / 8), dim = (i % (T::latent / 8)) * 8;
        *reinterpret_cast<uint4*>(tile + tile_offset<storage_keys>(key, dim, phase)) = decode8<Bf16>(current);
        current = next;
      }
      __syncthreads();
      const bool active = row_live;
      constexpr int key_base = 0;
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
  __syncthreads();

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
      float* dest = head_finish_partial<T>(tile, group * T::head_tile + n * 8 + 2 * c + r);
      float* audit = workspace ? workspace + partial_index<T>(row, split, head) : nullptr;
      if (warp == 0 && g == 0) {
        dest[0] = m_head; dest[1] = l_head;
        if (audit) { audit[0] = m_head; audit[1] = l_head; }
      }
#pragma unroll
      for (int m = 0; m < column_blocks; ++m)
#pragma unroll
        for (int half = 0; half < 2; ++half)
        {
          const int dim = warp * 128 + m * 16 + g + half * 8;
          dest[4 + dim] = output[m][n].v[2 * half + r];
          if (audit) audit[2 + dim] = output[m][n].v[2 * half + r];
        }
    }

  head_finish<T>(tile, row, blockIdx.y * 16, sinks, cos_sin, out);
}
template<bool Pipeline, class T, class Address = NativeAddress, bool Bf16 = false>
__device__ void head_async_attention_finish(
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
    uint32_t compressed_budget, uint32_t scale_bits, const float* sinks, const float* cos_sin, uint16_t* out) {
  ptx::griddep_launch_dependents();
  extern __shared__ __align__(1024) uint8_t native_latent_smem[];
  ptx::griddep_wait();
  static_assert(Bf16, "asynchronous key staging requires decoded BF16 operands");
  static_assert(T::latent == 512 && T::heads % 16 == 0 && T::n_tiles == 1);
  constexpr int storage_keys = T::keys;
  constexpr int tile_bytes = storage_keys * T::latent * 2;
  constexpr int exchange_words = T::blocks * T::keys * T::head_tile;
  const uint32_t live = *live_rows, first_row = blockIdx.x, split = blockIdx.z;
  if (first_row >= live) return;
  const int group = threadIdx.x / T::threads, tid = threadIdx.x % T::threads;
  const uint32_t row = first_row;
  const bool row_live = row < live;
  const int head0 = blockIdx.y * 16 + group * T::head_tile;
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
  int32_t* valid_all = reinterpret_cast<int32_t*>(exchange_all + 2 * exchange_words);
  const uint8_t** codes_all = reinterpret_cast<const uint8_t**>(valid_all + T::keys);
  const uint8_t** scales_all = codes_all + T::keys;
  int32_t* valid = valid_all;
  const uint8_t** code_rows = codes_all;
  const uint8_t** scale_rows = scales_all;
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

  auto resolve = [&](uint32_t unit) {
    __syncthreads();
    for (int key = threadIdx.x; key < T::keys; key += 2 * T::threads) {
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
  };
  if constexpr (Pipeline) {
    if (begin < end) {
      resolve(begin);
      stage_bf16_keys<64, 512>(tile, code_rows, valid, 0, 64, phase);
      attention::cp_async_wait<0>();
      __syncthreads();
    }
  }
  for (uint32_t unit = begin; unit < end; ++unit) {
    if constexpr (!Pipeline) resolve(unit);
    // Both eight-head groups read the same ordered 64-key unit.
    {
      if constexpr (!Pipeline) {
        stage_bf16_keys<64, 512>(tile, code_rows, valid, 0, 64, phase);
        attention::cp_async_wait<0>();
        __syncthreads();
      }
      const bool active = row_live;
      constexpr int key_base = 0;
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

        if constexpr (Pipeline) {
          // Softmax has consumed validity and addresses. All threads must
          // finish those reads before the next unit reuses the metadata.
          const bool more = unit + 1 < end;
          if (more) resolve(unit + 1);
          ptx::D4 chain[column_blocks][T::n_tiles];
#pragma unroll
          for (int m = 0; m < column_blocks; ++m)
#pragma unroll
            for (int n = 0; n < T::n_tiles; ++n)
#pragma unroll
              for (int i = 0; i < 4; ++i) chain[m][n].v[i] = 0.0f;
#pragma unroll
          for (int kt = 0; kt < T::key_tiles; ++kt) {
#pragma unroll
            for (int m = 0; m < column_blocks; ++m) {
              uint32_t a[4];
              ptx::ldmatrix_b16<4, true>(a, tile + tile_offset<storage_keys>(
                  kt * 16 + (lane / 16) * 8 + lane % 8,
                  warp * 128 + m * 16 + ((lane / 8) % 2) * 8, phase));
#pragma unroll
              for (int n = 0; n < T::n_tiles; ++n)
#pragma unroll
                for (int term = 0; term < terms; ++term)
                  ptx::mma_bf16(chain[m][n], ptx::A4{{a[0], a[1], a[2], a[3]}},
                                ptx::B2{{probability[term][kt][n][0], probability[term][kt][n][1]}});
            }
            if (kt == 1 && more) {
              // Every column in both groups has finished keys 0..31.
              __syncthreads();
              stage_bf16_keys<64, 512>(tile, code_rows, valid, 0, 32, phase);
            }
          }
#pragma unroll
          for (int m = 0; m < column_blocks; ++m)
#pragma unroll
            for (int n = 0; n < T::n_tiles; ++n)
#pragma unroll
              for (int i = 0; i < 4; ++i)
                output[m][n].v[i] = fmaf(output[m][n].v[i], rescale[n][i % 2], chain[m][n].v[i]);
          if (more) {
            // The second half is released only after every old PV reader.
            __syncthreads();
            stage_bf16_keys<64, 512>(tile, code_rows, valid, 32, 64, phase);
            attention::cp_async_wait<0>();
            __syncthreads();
          }
        } else {
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
      }
      __syncthreads();
    }
  }
  if (!row_live) return;
  __syncthreads();

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
      float* dest = head_finish_partial<T>(tile, group * T::head_tile + n * 8 + 2 * c + r);
      float* audit = workspace ? workspace + partial_index<T>(row, split, head) : nullptr;
      if (warp == 0 && g == 0) {
        dest[0] = m_head; dest[1] = l_head;
        if (audit) { audit[0] = m_head; audit[1] = l_head; }
      }
#pragma unroll
      for (int m = 0; m < column_blocks; ++m)
#pragma unroll
        for (int half = 0; half < 2; ++half)
        {
          const int dim = warp * 128 + m * 16 + g + half * 8;
          dest[4 + dim] = output[m][n].v[2 * half + r];
          if (audit) audit[2 + dim] = output[m][n].v[2 * half + r];
        }
    }

  head_finish<T>(tile, row, blockIdx.y * 16, sinks, cos_sin, out);
}
} // namespace chop::latent::sources_prefill
