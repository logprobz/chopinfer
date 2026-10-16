#pragma once

// Latent attention (kernel-core `latent_attention`): every head attends over
// one shared latent row per key, which is both key and value, through a
// gathered key list with -1 holes. See README.md.

#include <math.h>
#include <stdint.h>
#include "../attention/tile.cuh"

namespace chop::latent {

using attention::cp_async_commit;
using attention::cp_async_wait;
using attention::cp_async_zfill;
using attention::tile_offset;

constexpr float kLog2e = 1.4426950408889634f;
// The widest split of one row; the merge's weight buffer.
constexpr int max_splits = 64;

// Copies 4 bytes, or zero fills them when `valid` is false.
__device__ __forceinline__ void cp_async4_zfill(void* dst, const void* src, bool valid) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;"
               :: "r"(ptx::shared_address(dst)), "l"(src), "r"(valid ? 4 : 0)
               : "memory");
}

// exp(x) as the engine computes it everywhere: exp2 of x * log2(e).
__device__ __forceinline__ float exp_of(float x) { return exp2f(x * kLog2e); }

template<int Heads, int Latent, int Page, int MaxKeys, bool Fp8, bool Partial, int Keys>
struct Attention {
  static constexpr int heads = Heads, latent = Latent, page = Page, max_keys = MaxKeys;
  static constexpr bool fp8 = Fp8, partial = Partial;
  // A CTA takes 16 heads (two n8 tiles of the MMA's N); warp w owns the
  // 128-wide latent block w for both QK and PV.
  static_assert(Heads % 16 == 0, "heads fill whole 16-head tiles");
  static_assert(Latent % 128 == 0 && Latent <= 1024);
  static constexpr int head_tile = 16, n_tiles = 2;
  static constexpr int blocks = Latent / 128;
  static constexpr int warps = blocks, threads = warps * 32;
  // A unit is `Keys` keys in 16-key tiles; a CTA streams units through two
  // cp.async stages.
  static_assert(Keys == 16 || Keys == 32);
  static_assert(threads % Keys == 0, "each key's copies split evenly");
  static constexpr int keys = Keys, key_tiles = Keys / 16, stages = 2;
  // The BF16 [key][latent] tile the MMAs read.
  static constexpr int tile_bytes = Keys * Latent * 2;
  // A stage: the BF16 tile, or FP8 codes [key][latent] then F32 scales
  // [key][blocks].
  static constexpr int stage_bytes = Fp8 ? Keys * Latent + Keys * blocks * 4 : tile_bytes;
  static constexpr int q_bytes = head_tile * Latent * 2;
  // Each warp's 128-wide score chain per (key, head).
  static constexpr int exchange_bytes = warps * Keys * head_tile * 4;
  static constexpr int valid_bytes = stages * Keys * 4;
  static constexpr int tile_offset_bytes = stages * stage_bytes;
  static constexpr int q_offset = tile_offset_bytes + (Fp8 ? tile_bytes : 0);
  static constexpr int exchange_offset = q_offset + q_bytes;
  static constexpr int valid_offset = exchange_offset + exchange_bytes;
  static constexpr int smem = valid_offset + valid_bytes;
  static_assert(stage_bytes % 128 == 0 && q_offset % 128 == 0);
  // A split partial: F32 [max, denominator, numerator] per (row, split, head).
  static constexpr int partial_width = Latent + 2;
};

template<class T>
__device__ __forceinline__ uint64_t partial_index(int row, int splits, int split, int head) {
  return ((uint64_t(row) * splits + split) * T::heads + head) * T::partial_width;
}

// Grid [rows, heads / 16, splits], `T::threads` threads. Row r reads entries
// [0, budget) of its key list, cut into units of `T::keys`; split s owns
// units [units s / splits, units (s + 1) / splits). Per unit, warp w forms
// the scores' 128-wide chain over latent block w (FP8: times the key's block
// scale), the warps' chains are added in block order through shared memory,
// and every warp runs the same online softmax on those scores, then
// accumulates O^T = C^T P^T for its own 128 columns: one chain from zero per
// unit, added into the running output after its rescale. Probabilities are
// carried as three BF16 terms (FP8: of p times the key's block scale).
// With one split the CTA writes the entry's output (BF16 `o / l`, or the F32
// partial); otherwise its split partial for `merge`.
template<class T>
__device__ void attention(
    const uint16_t* q, const uint8_t* pool, const int32_t* block_tables,
    const int32_t* row_batch, const int32_t* key_lists, const uint32_t* live_rows,
    float* workspace, uint16_t* out, float* out_max, float* out_den, float* out_num,
    uint32_t max_pages, uint64_t page_stride_bytes, uint32_t budget, uint32_t splits,
    uint32_t scale_bits) {
  ptx::griddep_launch_dependents();
  extern __shared__ __align__(1024) uint8_t latent_smem[];
  ptx::griddep_wait();
  const int row = blockIdx.x;
  const int head0 = blockIdx.y * T::head_tile;
  const int split = blockIdx.z;
  if (row >= int(*live_rows)) return;
  const int batch = row_batch[row];
  const int count = min(int(budget), T::max_keys);
  const int units = (count + T::keys - 1) / T::keys;
  const int unit_begin = int(int64_t(units) * split / splits);
  const int unit_end = int(int64_t(units) * (split + 1) / splits);

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int g = lane / 4;
  const int c = lane % 4;
  constexpr int terms = 3;
  constexpr int column_blocks = 128 / 16;

  uint8_t* stages = latent_smem;
  uint8_t* wide = latent_smem + T::tile_offset_bytes;
  uint8_t* q_smem = latent_smem + T::q_offset;
  float* exchange = reinterpret_cast<float*>(latent_smem + T::exchange_offset);
  int32_t* valid_keys = reinterpret_cast<int32_t*>(latent_smem + T::valid_offset);
  auto phase_of = [](const void* p) { return (ptx::shared_address(p) / 128) % 8; };
  auto tile_of = [&](int stage) -> uint8_t* {
    return T::fp8 ? wide : stages + stage * T::stage_bytes;
  };
  auto scales_of = [&](int stage) {
    return reinterpret_cast<const float*>(stages + stage * T::stage_bytes +
                                          T::keys * T::latent);
  };
  const uint32_t q_phase = phase_of(q_smem);

  // The 16 heads' queries, with the first unit's copies.
  const uint8_t* q_row = reinterpret_cast<const uint8_t*>(q) +
                         (uint64_t(row) * T::heads + head0) * T::latent * 2;
  for (int i = threadIdx.x; i < T::head_tile * T::latent / 8; i += T::threads) {
    const int h = i / (T::latent / 8);
    const int sector = i % (T::latent / 8);
    cp_async_zfill(q_smem + tile_offset<T::head_tile>(h, sector * 8, q_phase),
                   q_row + h * T::latent * 2 + sector * 16, true);
  }

  // Key j's latent row: `lanes` threads copy it in 16-byte sectors (FP8: and
  // its scales in 4-byte copies) after one page lookup. Holes and entries
  // past the budget are zero filled and marked invalid.
  auto issue = [&](int unit, int stage) {
    constexpr int lanes = T::threads / T::keys;
    const int key = threadIdx.x / lanes;
    const int part = threadIdx.x % lanes;
    const int j = unit * T::keys + key;
    const int t = j < count ? key_lists[uint64_t(row) * T::max_keys + j] : -1;
    const bool valid = t >= 0;
    if (part == 0) valid_keys[stage * T::keys + key] = valid;
    const uint64_t page =
        valid ? uint32_t(block_tables[uint64_t(batch) * max_pages + t / T::page]) : 0;
    const uint64_t slot = valid ? uint64_t(t % T::page) : 0;
    const uint8_t* page_base = pool + page * page_stride_bytes;
    uint8_t* stage_base = stages + stage * T::stage_bytes;
    if constexpr (T::fp8) {
      const uint8_t* codes = page_base + slot * T::latent;
      for (int sector = part; sector < T::latent / 16; sector += lanes)
        cp_async_zfill(stage_base + key * T::latent + sector * 16, codes + sector * 16, valid);
      const uint8_t* scales =
          page_base + uint64_t(T::page) * T::latent + slot * T::blocks * 4;
      for (int b = part; b < T::blocks; b += lanes)
        cp_async4_zfill(stage_base + T::keys * T::latent + (key * T::blocks + b) * 4,
                        scales + b * 4, valid);
    } else {
      const uint8_t* values = page_base + slot * T::latent * 2;
      const uint32_t phase = phase_of(stage_base);
      for (int sector = part; sector < T::latent / 8; sector += lanes)
        cp_async_zfill(stage_base + tile_offset<T::keys>(key, sector * 8, phase),
                       values + sector * 16, valid);
    }
  };

  const float scale = __uint_as_float(scale_bits);
  // Warp w's queries over its latent block, as B fragments.
  ptx::B2 query[8][T::n_tiles];
  // Running maximum and this lane's share of the denominator for head
  // n * 8 + g (the B-fragment layout of P).
  float maximum[T::n_tiles], denominator[T::n_tiles];
  // O^T for latent columns warp * 128 + m * 16 + (g, g + 8) and heads
  // n * 8 + 2c + (0, 1).
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

  if (unit_begin < unit_end) issue(unit_begin, 0);
  cp_async_commit();

  for (int unit = unit_begin; unit < unit_end; ++unit) {
    const int t = unit - unit_begin;
    const int stage = t % T::stages;
    // Unit t has landed and every warp has left unit t - 1, whose stage now
    // takes unit t + 1.
    cp_async_wait<0>();
    __syncthreads();
    if (t == 0) {
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
    if (unit + 1 < unit_end) issue(unit + 1, (t + 1) % T::stages);
    cp_async_commit();
    if constexpr (T::fp8) {
      attention::widen_fp8<T::keys, T::latent, T::threads>(wide, stages + stage * T::stage_bytes,
                                                phase_of(wide));
      __syncthreads();
    }
    const uint8_t* tile = tile_of(stage);
    const uint32_t phase = phase_of(tile);
    const float* scales = T::fp8 ? scales_of(stage) : nullptr;
    const int32_t* valid = valid_keys + stage * T::keys;

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
        const float block_scale = T::fp8 ? scales[key * T::blocks + warp] : 1.0f;
#pragma unroll
        for (int n = 0; n < T::n_tiles; ++n) {
          float2 pair = make_float2(chain[n].v[2 * half], chain[n].v[2 * half + 1]);
          if constexpr (T::fp8) {
            pair.x *= block_scale;
            pair.y *= block_scale;
          }
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
          if constexpr (T::fp8) {
            const int key = kt * 16 + 2 * c + half * 8;
            p0 *= scales[key * T::blocks + warp];
            p1 *= scales[(key + 1) * T::blocks + warp];
          }
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
      if (splits == 1 && !T::partial) {
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
        float* numerator;
        if (splits == 1) {
          if (warp == 0 && g == 0) {
            out_max[pair] = m_head;
            out_den[pair] = l_head;
          }
          numerator = out_num + pair * T::latent;
        } else {
          float* destination = workspace + partial_index<T>(row, splits, split, head);
          if (warp == 0 && g == 0) {
            destination[0] = m_head;
            destination[1] = l_head;
          }
          numerator = destination + 2;
        }
#pragma unroll
        for (int m = 0; m < column_blocks; ++m)
#pragma unroll
          for (int half = 0; half < 2; ++half)
            numerator[warp * 128 + m * 16 + g + half * 8] = output[m][n].v[2 * half + r];
      }
    }
}

// Grid [rows, heads], 128 threads: folds a row's split partials in split
// order with the context-parallel merge's arithmetic (`M = max m_s`, `a_s =
// exp(m_s - M)`, `L` and `O` FMA sums from zero) and writes the entry's
// output: BF16 `O / L` (zeros when every split is empty) or the F32 partial.
constexpr int merge_threads = 128;

template<class T>
__device__ void merge(const float* workspace, const uint32_t* live_rows, uint16_t* out,
                      float* out_max, float* out_den, float* out_num, uint32_t splits) {
  ptx::griddep_launch_dependents();
  __shared__ float weights[max_splits];
  __shared__ float totals[2];
  ptx::griddep_wait();
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  if (row >= int(*live_rows)) return;
  auto partial = [&](uint32_t split) {
    return workspace + partial_index<T>(row, splits, split, head);
  };
  if (threadIdx.x == 0) {
    float top = -INFINITY;
    for (uint32_t split = 0; split < splits; ++split) top = fmaxf(top, partial(split)[0]);
    float sum = 0.0f;
    for (uint32_t split = 0; split < splits; ++split) {
      const float m = partial(split)[0];
      const float weight = m == -INFINITY ? 0.0f : exp_of(m - top);
      weights[split] = weight;
      sum = fmaf(weight, partial(split)[1], sum);
    }
    totals[0] = top;
    totals[1] = sum;
  }
  __syncthreads();
  const uint64_t pair = uint64_t(row) * T::heads + head;
  const float total = totals[1];
  if (T::partial && threadIdx.x == 0) {
    out_max[pair] = totals[0];
    out_den[pair] = total;
  }
  for (int d = threadIdx.x; d < T::latent; d += merge_threads) {
    float numerator = 0.0f;
    for (uint32_t split = 0; split < splits; ++split)
      numerator = fmaf(weights[split], partial(split)[2 + d], numerator);
    if constexpr (T::partial) {
      out_num[pair * T::latent + d] = numerator;
    } else {
      out[pair * T::latent + d] =
          convert::bf16_from_f32(total == 0.0f ? 0.0f : __fdiv_rn(numerator, total));
    }
  }
}

} // namespace chop::latent
