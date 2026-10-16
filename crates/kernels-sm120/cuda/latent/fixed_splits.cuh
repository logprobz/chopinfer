#pragma once

// Latent attention over a fixed key partition for decode and verify (see
// README.md, "Fixed splits"): every row's list is cut into `splits` ranges
// whatever its bucket, and each range's partial is computed exactly as a
// one-split-per-CTA launch of `attention` computes it. The bucket only
// decides how many consecutive ranges one CTA takes, so a row's bits do not
// depend on the rows that share its step.

#include "attention.cuh"

namespace chop::latent {

// Grid [rows, heads / 16, groups], `T::threads` threads. Row r reads
// entries [0, budget) of its key list in units of `T::keys`; split s owns
// units [units s / splits, units (s + 1) / splits), and CTA z takes splits
// [splits z / groups, splits (z + 1) / groups) in order. Per split the CTA
// starts the online softmax from its initial state, runs `attention`'s
// unit loop over the split's units and writes the split's partial for
// `merge`; empty splits write the empty partial. The two cp.async stages
// run across split boundaries: the next unit's copies are issued before
// the current unit's math, whichever split it belongs to.
template<class T>
__device__ void attention_fixed_splits(
    const uint16_t* q, const uint8_t* pool, const int32_t* block_tables,
    const int32_t* row_batch, const int32_t* key_lists, const uint32_t* live_rows,
    float* workspace, uint32_t max_pages, uint64_t page_stride_bytes, uint32_t budget,
    uint32_t splits, uint32_t scale_bits) {
  ptx::griddep_launch_dependents();
  extern __shared__ __align__(1024) uint8_t latent_smem[];
  ptx::griddep_wait();
  const int row = blockIdx.x;
  const int head0 = blockIdx.y * T::head_tile;
  if (row >= int(*live_rows)) return;
  const int batch = row_batch[row];
  const int count = min(int(budget), T::max_keys);
  const int units = (count + T::keys - 1) / T::keys;
  auto unit_bound = [&](int split) { return int(int64_t(units) * split / splits); };
  const int split_first = int(int64_t(splits) * blockIdx.z / gridDim.z);
  const int split_last = int(int64_t(splits) * (blockIdx.z + 1) / gridDim.z);
  const int cta_begin = unit_bound(split_first);
  const int cta_end = unit_bound(split_last);

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

  // Key j's latent row, as in `attention`.
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
  ptx::B2 query[8][T::n_tiles];
  float maximum[T::n_tiles], denominator[T::n_tiles];
  ptx::D4 output[column_blocks][T::n_tiles];

  if (cta_begin < cta_end) issue(cta_begin, 0);
  cp_async_commit();

  for (int split = split_first; split < split_last; ++split) {
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

    const int unit_end = unit_bound(split + 1);
    for (int unit = unit_bound(split); unit < unit_end; ++unit) {
      // The CTA's units are contiguous: unit t of the CTA, across its
      // splits, picks the stage and loads the queries first.
      const int t = unit - cta_begin;
      const int stage = t % T::stages;
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
      if (unit + 1 < cta_end) issue(unit + 1, (t + 1) % T::stages);
      cp_async_commit();
      if constexpr (T::fp8) {
        attention::widen_fp8<T::keys, T::latent, T::threads>(
            wide, stages + stage * T::stage_bytes, phase_of(wide));
        __syncthreads();
      }
      const uint8_t* tile = tile_of(stage);
      const uint32_t phase = phase_of(tile);
      const float* scales = T::fp8 ? scales_of(stage) : nullptr;
      const int32_t* valid = valid_keys + stage * T::keys;

      // QK: one chain from zero over this warp's 128 latent dims per key
      // tile.
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

      // Scores, the online softmax, the three-term probabilities and PV, as
      // in `attention`.
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
      float rescale[T::n_tiles][2];
#pragma unroll
      for (int n = 0; n < T::n_tiles; ++n)
#pragma unroll
        for (int r = 0; r < 2; ++r)
          rescale[n][r] = __shfl_sync(0xffffffff, correction[n], (2 * c + r) * 4);

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

    // The split's partial: the denominator over the four lanes of each
    // head in a fixed order, then each column's maximum, denominator and
    // numerator.
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
        float* destination = workspace + partial_index<T>(row, splits, split, head);
        if (warp == 0 && g == 0) {
          destination[0] = m_head;
          destination[1] = l_head;
        }
#pragma unroll
        for (int m = 0; m < column_blocks; ++m)
#pragma unroll
          for (int half = 0; half < 2; ++half)
            destination[2 + warp * 128 + m * 16 + g + half * 8] = output[m][n].v[2 * half + r];
      }
  }
  cp_async_wait<0>();
}

} // namespace chop::latent
