#pragma once

// Latent attention in one split per row over FP8 pages, with each warp
// widening only the 16-column slices of its own latent block that its MMAs
// read (README.md "Slice-widened attention"). Per output element it runs
// `attention`'s operations in `attention`'s order with one split; only the
// staging differs, so two CTAs fit one SM:
//
// - the FP8 codes land in each stage with their 16-byte sectors swizzled
//   inside each 128-byte block, and a warp widens one [32 keys][16 columns]
//   slice at a time into its own 1 KiB buffer (twice per unit: for QK and
//   for PV), instead of the whole unit into a shared BF16 tile;
// - the queries load into their B fragments straight from global memory;
// - every thread follows one key of each unit: the key list entry three
//   units ahead and the page table entry two units ahead, so no copy waits
//   on a dependent load;
// - a unit whose keys are all holes skips its widening and MMAs and keeps
//   only `attention`'s state update: every score is -inf, so the maximum
//   stays, every probability term and PV chain is +0, and the denominator
//   and each output take `fmaf(x, correction, +0)` with the correction
//   `attention` computes. That update is not the identity: an output that
//   underflowed to -0 becomes +0.

#include "latent/attention.cuh"

namespace chop::latent {

template<int Heads, int Latent, int Page, int MaxKeys, bool Partial>
struct SliceAttention {
  static constexpr int heads = Heads, latent = Latent, page = Page, max_keys = MaxKeys;
  static constexpr bool partial = Partial;
  static_assert(Heads % 16 == 0, "heads fill whole 16-head tiles");
  static_assert(Latent % 128 == 0 && Latent <= 1024);
  static constexpr int head_tile = 16, n_tiles = 2;
  static constexpr int blocks = Latent / 128;
  static constexpr int warps = blocks, threads = warps * 32;
  // `attention`'s 32-key units, two 16-key tiles each.
  static constexpr int keys = 32, key_tiles = 2, stages = 2;
  static_assert(threads % keys == 0, "each key's copies split evenly");
  // A stage: FP8 codes [key][latent], each 128-byte block's sectors
  // swizzled by the key, then F32 scales [key][blocks].
  static constexpr int code_bytes = keys * Latent;
  static constexpr int stage_bytes = code_bytes + keys * blocks * 4;
  // A warp's widened slice: BF16 [key][16], its two 16-byte chunks
  // swizzled by bit 2 of the key.
  static constexpr int slice_bytes = keys * 16 * 2;
  static constexpr int exchange_bytes = warps * keys * head_tile * 4;
  static constexpr int slices_offset = stages * stage_bytes;
  static constexpr int exchange_offset = slices_offset + warps * slice_bytes;
  static constexpr int valid_offset = exchange_offset + exchange_bytes;
  static constexpr int smem = valid_offset + stages * keys * 4;
  static_assert(stage_bytes % 128 == 0 && slices_offset % 128 == 0);
};

// Sector `sector` (16 codes) of key `key`'s row in a stage.
template<class T>
__device__ __forceinline__ int code_offset(int key, int sector) {
  return key * T::latent + ((sector & ~7) | ((sector ^ key) & 7)) * 16;
}

// Chunk `chunk` (8 columns) of key `key`'s row in a widened slice.
__device__ __forceinline__ int slice_offset(int key, int chunk) {
  return key * 32 + ((chunk ^ ((key >> 2) & 1)) << 4);
}

// Lane `key` widens its key's 16 codes of sector `sector` (the warp's block,
// 16 columns) exactly to BF16 into `slice`; the warp synchronizes after.
template<class T>
__device__ __forceinline__ void widen_slice(uint8_t* slice, const uint8_t* codes, int sector,
                                            int key) {
  const uint4 bytes = *reinterpret_cast<const uint4*>(codes + code_offset<T>(key, sector));
  const uint32_t words[4] = {bytes.x, bytes.y, bytes.z, bytes.w};
  uint32_t wide[8];
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    wide[2 * w] = convert::unpack_bf16x2_e4m3(uint16_t(words[w]));
    wide[2 * w + 1] = convert::unpack_bf16x2_e4m3(uint16_t(words[w] >> 16));
  }
  *reinterpret_cast<uint4*>(slice + slice_offset(key, 0)) =
      make_uint4(wide[0], wide[1], wide[2], wide[3]);
  *reinterpret_cast<uint4*>(slice + slice_offset(key, 1)) =
      make_uint4(wide[4], wide[5], wide[6], wide[7]);
  __syncwarp();
}

// Grid [rows, heads / 16, 1], `T::threads` threads, two CTAs per SM. Row r
// reads entries [0, budget) of its key list in units of 32 keys: one split,
// written as `attention` writes it (BF16 `o / l`, or the F32 partial).
template<class T>
__device__ void attention_slices(
    const uint16_t* q, const uint8_t* pool, const int32_t* block_tables,
    const int32_t* row_batch, const int32_t* key_lists, const uint32_t* live_rows,
    uint16_t* out, float* out_max, float* out_den, float* out_num, uint32_t max_pages,
    uint64_t page_stride_bytes, uint32_t budget, uint32_t scale_bits) {
  ptx::griddep_launch_dependents();
  extern __shared__ __align__(1024) uint8_t latent_smem[];
#ifndef CHOP_LATENT_SLICES_TEST_LATE_WAIT
  ptx::griddep_wait();
#endif
  const int row = blockIdx.x;
  const int head0 = blockIdx.y * T::head_tile;
  if (row >= int(*live_rows)) return;
  const int count = min(int(budget), T::max_keys);
  const int units = (count + T::keys - 1) / T::keys;

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int g = lane / 4;
  const int c = lane % 4;
  constexpr int terms = 3;
  constexpr int column_blocks = 128 / 16;

  uint8_t* stages = latent_smem;
  uint8_t* slice = latent_smem + T::slices_offset + warp * T::slice_bytes;
  float* exchange = reinterpret_cast<float*>(latent_smem + T::exchange_offset);
  int32_t* valid_keys = reinterpret_cast<int32_t*>(latent_smem + T::valid_offset);

  // Warp w's queries over its latent block as B fragments: head n * 8 + g,
  // columns 2c, 2c + 1 (b0) and 2c + 8, 2c + 9 (b1) of each 16-column step.
  ptx::B2 query[8][T::n_tiles];
  {
    const uint16_t* q_row = q + (uint64_t(row) * T::heads + head0) * T::latent + warp * 128;
#pragma unroll
    for (int n = 0; n < T::n_tiles; ++n)
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        const uint16_t* h = q_row + (n * 8 + g) * T::latent + ks * 16 + 2 * c;
        query[ks][n] = ptx::B2{{*reinterpret_cast<const uint32_t*>(h),
                                *reinterpret_cast<const uint32_t*>(h + 8)}};
      }
  }
#ifdef CHOP_LATENT_SLICES_TEST_LATE_WAIT
  ptx::griddep_wait();
#endif
  const int batch = row_batch[row];

  // Thread t follows key t / lanes of every unit and copies part t % lanes
  // of its row.
  constexpr int lanes = T::threads / T::keys;
  const int key = threadIdx.x / lanes;
  const int part = threadIdx.x % lanes;
  const int32_t* list = key_lists + uint64_t(row) * T::max_keys;
  const int32_t* table = block_tables + uint64_t(batch) * max_pages;
  auto token_of = [&](int unit) {
    const int j = unit * T::keys + key;
    return j < count ? list[j] : -1;
  };
  auto page_of = [&](int token) {
    return token >= 0 ? uint32_t(table[token / T::page]) : 0u;
  };
  // The key's codes and scales of a unit into `stage`; holes and entries
  // past the budget are zero filled and marked invalid.
  auto issue = [&](int stage, int token, uint32_t page) {
    const bool valid = token >= 0;
    if (part == 0) valid_keys[stage * T::keys + key] = valid;
    const uint64_t slot = valid ? uint64_t(token % T::page) : 0;
    const uint8_t* page_base = pool + uint64_t(page) * page_stride_bytes;
    uint8_t* stage_base = stages + stage * T::stage_bytes;
    const uint8_t* codes = page_base + slot * T::latent;
    for (int sector = part; sector < T::latent / 16; sector += lanes)
      cp_async_zfill(stage_base + code_offset<T>(key, sector), codes + sector * 16, valid);
    const uint8_t* scales = page_base + uint64_t(T::page) * T::latent + slot * T::blocks * 4;
    for (int b = part; b < T::blocks; b += lanes)
      cp_async4_zfill(stage_base + T::code_bytes + (key * T::blocks + b) * 4, scales + b * 4,
                      valid);
  };

  // Unit u + 1's token and page, unit u + 2's token, and whether this
  // thread's key of unit u is live.
  int token_next = token_of(0);
  uint32_t page_next = page_of(token_next);
  bool live = token_next >= 0;
  if (units > 0) issue(0, token_next, page_next);
  cp_async_commit();
  token_next = token_of(1);
  int token_after = token_of(2);
  page_next = page_of(token_next);
#ifdef CHOP_LATENT_SLICES_TEST_STALE_PAGE
  uint32_t page_stale = page_of(token_of(0));
#endif

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

  for (int unit = 0; unit < units; ++unit) {
    const int stage = unit % T::stages;
    // Unit u has landed and every warp has left unit u - 1, whose stage now
    // takes unit u + 1.
    cp_async_wait<0>();
#ifdef CHOP_LATENT_SLICES_TEST_SKIP_FIRST
    const bool any = __syncthreads_or(key == 0 && live);
#else
    const bool any = __syncthreads_or(live);
#endif
    live = token_next >= 0;
    if (unit + 1 < units) {
#ifdef CHOP_LATENT_SLICES_TEST_STALE_PAGE
      issue((unit + 1) % T::stages, token_next, page_stale);
      page_stale = page_next;
#else
      issue((unit + 1) % T::stages, token_next, page_next);
#endif
    }
    cp_async_commit();
    // The loads for the next two units land under this unit's math.
    page_next = page_of(token_after);
    token_next = token_after;
    token_after = token_of(unit + 3);
    if (!any) {
      // An empty unit: the state update of a tile whose scores are all
      // -inf, whose probabilities and PV chains are +0.
      float correction[T::n_tiles];
#pragma unroll
      for (int n = 0; n < T::n_tiles; ++n) {
        const float base = maximum[n] == -INFINITY ? 0.0f : maximum[n];
        correction[n] = exp_of(maximum[n] - base);
        denominator[n] = fmaf(denominator[n], correction[n], 0.0f);
      }
#pragma unroll
      for (int n = 0; n < T::n_tiles; ++n)
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const float rescale = __shfl_sync(0xffffffff, correction[n], (2 * c + r) * 4);
#pragma unroll
          for (int m = 0; m < column_blocks; ++m)
#pragma unroll
            for (int half = 0; half < 2; ++half)
              output[m][n].v[2 * half + r] = fmaf(output[m][n].v[2 * half + r], rescale, 0.0f);
        }
      continue;
    }

    const uint8_t* codes = stages + stage * T::stage_bytes;
    const float* scales = reinterpret_cast<const float*>(codes + T::code_bytes);
    const int32_t* valid = valid_keys + stage * T::keys;

    // QK: one chain from zero over this warp's 128 latent dims per key
    // tile, its 16-column steps in order.
    ptx::D4 chain[T::key_tiles][T::n_tiles];
#pragma unroll
    for (int kt = 0; kt < T::key_tiles; ++kt)
#pragma unroll
      for (int n = 0; n < T::n_tiles; ++n)
#pragma unroll
        for (int i = 0; i < 4; ++i) chain[kt][n].v[i] = 0.0f;
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      widen_slice<T>(slice, codes, warp * 8 + ks, lane);
#pragma unroll
      for (int kt = 0; kt < T::key_tiles; ++kt) {
        uint32_t a[4];
        ptx::ldmatrix_b16<4>(a, slice + slice_offset(kt * 16 + ((lane / 8) % 2) * 8 + lane % 8,
                                                     lane / 16));
#pragma unroll
        for (int n = 0; n < T::n_tiles; ++n)
          ptx::mma_bf16(chain[kt][n], ptx::A4{{a[0], a[1], a[2], a[3]}}, query[ks][n]);
      }
      __syncwarp();
    }
#pragma unroll
    for (int kt = 0; kt < T::key_tiles; ++kt)
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int k = kt * 16 + g + half * 8;
        const float block_scale = scales[k * T::blocks + warp];
#pragma unroll
        for (int n = 0; n < T::n_tiles; ++n) {
          float2 pair = make_float2(chain[kt][n].v[2 * half], chain[kt][n].v[2 * half + 1]);
          pair.x *= block_scale;
          pair.y *= block_scale;
          *reinterpret_cast<float2*>(
              exchange + (warp * T::keys + k) * T::head_tile + n * 8 + 2 * c) = pair;
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
          const int k = kt * 16 + 2 * c + (i % 2) + (i / 2) * 8;
          const float* chains = exchange + k * T::head_tile + n * 8 + g;
          float sum = chains[0];
#pragma unroll
          for (int w = 1; w < T::warps; ++w) sum += chains[w * T::keys * T::head_tile];
          const float s = valid[k] ? sum * scale : -INFINITY;
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

    // P^T as B fragments, each value p times the key's block scale carried
    // as `terms` BF16 parts.
    uint32_t probability[terms][T::key_tiles][T::n_tiles][2];
#pragma unroll
    for (int kt = 0; kt < T::key_tiles; ++kt)
#pragma unroll
      for (int n = 0; n < T::n_tiles; ++n)
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int k = kt * 16 + 2 * c + half * 8;
          float p0 = p[kt][n][2 * half] * scales[k * T::blocks + warp];
          float p1 = p[kt][n][2 * half + 1] * scales[(k + 1) * T::blocks + warp];
#pragma unroll
          for (int term = 0; term < terms; ++term) {
            const uint32_t pair = attention::bf16_pair(p0, p1);
            p0 -= attention::bf16_low(pair);
            p1 -= attention::bf16_high(pair);
            probability[term][kt][n][half] = pair;
          }
        }
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
      widen_slice<T>(slice, codes, warp * 8 + m, lane);
      ptx::D4 pv[T::n_tiles];
#pragma unroll
      for (int n = 0; n < T::n_tiles; ++n)
#pragma unroll
        for (int i = 0; i < 4; ++i) pv[n].v[i] = 0.0f;
#pragma unroll
      for (int kt = 0; kt < T::key_tiles; ++kt) {
        uint32_t a[4];
        ptx::ldmatrix_b16<4, true>(
            a, slice + slice_offset(kt * 16 + (lane / 16) * 8 + lane % 8, (lane / 8) % 2));
#pragma unroll
        for (int n = 0; n < T::n_tiles; ++n)
#pragma unroll
          for (int term = 0; term < terms; ++term)
            ptx::mma_bf16(pv[n], ptx::A4{{a[0], a[1], a[2], a[3]}},
                          ptx::B2{{probability[term][kt][n][0], probability[term][kt][n][1]}});
      }
      __syncwarp();
#pragma unroll
      for (int n = 0; n < T::n_tiles; ++n)
#pragma unroll
        for (int i = 0; i < 4; ++i)
          output[m][n].v[i] = fmaf(output[m][n].v[i], rescale[n][i % 2], pv[n].v[i]);
    }
  }
  cp_async_wait<0>();

  // The denominator over the four lanes of each head, in a fixed order, then
  // the maximum and denominator of output column 2c + r.
#pragma unroll
  for (int n = 0; n < T::n_tiles; ++n) {
    denominator[n] += __shfl_xor_sync(0xffffffff, denominator[n], 1);
    denominator[n] += __shfl_xor_sync(0xffffffff, denominator[n], 2);
  }
#pragma unroll
  for (int n = 0; n < T::n_tiles; ++n)
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int source = (2 * c + r) * 4;
      const float m_head = __shfl_sync(0xffffffff, maximum[n], source);
      const float l_head = __shfl_sync(0xffffffff, denominator[n], source);
      const int head = head0 + n * 8 + 2 * c + r;
      const uint64_t pair = uint64_t(row) * T::heads + head;
      if constexpr (!T::partial) {
        uint16_t* destination = out + pair * T::latent + warp * 128;
#pragma unroll
        for (int m = 0; m < column_blocks; ++m)
#pragma unroll
          for (int half = 0; half < 2; ++half) {
            const float o = output[m][n].v[2 * half + r];
            destination[m * 16 + g + half * 8] =
                convert::bf16_from_f32(l_head == 0.0f ? 0.0f : __fdiv_rn(o, l_head));
          }
      } else {
        if (warp == 0 && g == 0) {
          out_max[pair] = m_head;
          out_den[pair] = l_head;
        }
        float* numerator = out_num + pair * T::latent;
#pragma unroll
        for (int m = 0; m < column_blocks; ++m)
#pragma unroll
          for (int half = 0; half < 2; ++half)
            numerator[warp * 128 + m * 16 + g + half * 8] = output[m][n].v[2 * half + r];
      }
    }
}

} // namespace chop::latent
