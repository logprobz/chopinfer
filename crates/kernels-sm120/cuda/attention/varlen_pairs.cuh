#pragma once

// varlen.cuh's contiguous attention over BF16 pair operands with F32 output
// (contract attention.contiguous-varlen-bf16-pairs-f32.v1), for towers that
// keep F32 activations (`attention.varlen.pairs.v1`).
//
// Each packed token row holds six BF16 planes, the pairs hi = bf16(x) and
// lo = bf16(x - hi) of q, k and v (encoder/rows/rope_qkv_pairs.cuh):
// q hi, q lo, k hi, k lo, v hi, v lo. Scores take q hi k hi + q hi k lo +
// q lo k hi (the dropped q lo k lo is at most 2^-16 of each product), and
// every probability term multiplies both halves of v. Probabilities keep at
// least two BF16 terms (three below 128 keys), so the probability left out
// is below 2^-16 of each (2^-24 below 128 keys) and every product carries
// 16 bits of each operand. The slices, key intervals, online softmax and output
// normalization are varlen.cuh's; K and V stream through two cp.async stages
// of four tiles (k hi, k lo, v hi, v lo) in dynamic shared memory.

#include "varlen.cuh"

namespace chop::attention::varlen {

template<class T>
struct Pairs {
  static constexpr int q_width = T::q_heads * T::d;
  static constexpr int kv_width = T::kv_heads * T::d;
  // BF16 elements per packed row and the plane starts.
  static constexpr int row = 2 * (q_width + 2 * kv_width);
  static constexpr int q_lo = q_width;
  static constexpr int k_hi = 2 * q_width;
  static constexpr int smem = T::stages * 4 * T::tile_bytes;
};

template<class T>
__device__ void attention_pairs(
    const __nv_bfloat16* qkv, const int32_t* cu_seqlens, const float* sinks,
    float* output, const uint32_t* live_tokens, uint32_t live_sequences,
    uint32_t scale_bits) {
  using P = Pairs<T>;
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const int live = int(*live_tokens);
  const int sequences = int(live_sequences);
  const int t0 = int(blockIdx.x) * T::tokens;
  if (t0 >= live || sequences <= 0) return;
  const int kv_head = blockIdx.y;
  const int tokens = min(T::tokens, live - t0);
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int g = lane / 4;
  const int c = lane % 4;

  // Keys of the whole slice.
  int key_first, key_last, unused_hi, unused_start, unused_lo;
  visible<T>(cu_seqlens, sequences, t0, key_first, unused_hi, unused_start);
  visible<T>(cu_seqlens, sequences, t0 + tokens - 1, unused_lo, key_last, unused_start);
  const int key_tiles = (key_last - key_first) / T::keys + 1;

  // Tile `which` of a stage: 0 k hi, 1 k lo, 2 v hi, 3 v lo.
  extern __shared__ __align__(128) uint8_t storage[];
  auto tile = [&](int stage, int which) {
    return storage + (stage * 4 + which) * T::tile_bytes;
  };
  // Columns D..dk of both K tiles are the zero padding of the QK reduction;
  // cp.async never writes them.
  if constexpr (T::dk != T::d) {
    for (int row = threadIdx.x; row < T::stages * 2 * T::keys; row += T::threads) {
      const int stage = row / (2 * T::keys);
      uint8_t* base = tile(stage, row / T::keys % 2) + (row % T::keys) * T::stride;
      *reinterpret_cast<uint4*>(base + T::d * 2) = make_uint4(0, 0, 0, 0);
    }
  }

  // Each thread copies 16-byte sectors of key rows from the four planes;
  // keys past the slice are zero filled and masked.
  auto issue = [&](int key_tile, int stage) {
    constexpr int sectors = T::d / 8;
    constexpr int per_tile = T::keys * sectors;
    const int key0 = key_first + key_tile * T::keys;
    for (int i = threadIdx.x; i < 4 * per_tile; i += T::threads) {
      const int which = i / per_tile;
      const int j = i % per_tile;
      const int row = j / sectors;
      const int sector = j % sectors;
      const int key = key0 + row;
      const bool valid = key <= key_last;
      const __nv_bfloat16* src = qkv + uint64_t(valid ? key : key_first) * P::row +
          P::k_hi + which * P::kv_width + kv_head * T::d + sector * 8;
      cp_async_zfill(tile(stage, which) + row * T::stride + sector * 16, src, valid);
    }
    cp_async_commit();
  };
  issue(0, 0);
  if (key_tiles > 1) issue(1, 1);

  // This thread's rows g and g + 8 of the warp's m16 tile.
  int token[2], head[2], lo[2], hi[2], start[2];
  bool row_live[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int row = warp * 16 + g + r * 8;
    token[r] = t0 + row / T::group;
    head[r] = kv_head * T::group + row % T::group;
    row_live[r] = row / T::group < tokens;
    if (row_live[r]) {
      visible<T>(cu_seqlens, sequences, token[r], lo[r], hi[r], start[r]);
    } else {
      lo[r] = INT32_MAX;
      hi[r] = -1;
      start[r] = -1;
    }
  }
  int warp_lo = min(lo[0], lo[1]);
  int warp_hi = max(hi[0], hi[1]);
  int all_lo = max(row_live[0] ? lo[0] : INT32_MIN, row_live[1] ? lo[1] : INT32_MIN);
  int all_hi = min(row_live[0] ? hi[0] : INT32_MAX, row_live[1] ? hi[1] : INT32_MAX);
  int fewest = min(row_live[0] ? hi[0] - lo[0] + 1 : INT32_MAX,
                   row_live[1] ? hi[1] - lo[1] + 1 : INT32_MAX);
  int sink_lo = INT32_MAX, sink_hi = -1;
  if constexpr (T::sink) {
#pragma unroll
    for (int r = 0; r < 2; ++r)
      if (row_live[r] && lo[r] == start[r]) {
        sink_lo = min(sink_lo, start[r]);
        sink_hi = max(sink_hi, start[r]);
      }
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    warp_lo = min(warp_lo, __shfl_xor_sync(0xffffffff, warp_lo, offset));
    warp_hi = max(warp_hi, __shfl_xor_sync(0xffffffff, warp_hi, offset));
    all_lo = max(all_lo, __shfl_xor_sync(0xffffffff, all_lo, offset));
    all_hi = min(all_hi, __shfl_xor_sync(0xffffffff, all_hi, offset));
    fewest = min(fewest, __shfl_xor_sync(0xffffffff, fewest, offset));
    sink_lo = min(sink_lo, __shfl_xor_sync(0xffffffff, sink_lo, offset));
    sink_hi = max(sink_hi, __shfl_xor_sync(0xffffffff, sink_hi, offset));
  }
  const bool warp_live = warp_hi >= 0;
  const int terms = max(2, probability_terms(fewest));

  constexpr float log2e = 1.4426950408889634f;
  const float score_scale = __uint_as_float(scale_bits) * log2e;
  float sink_logit[2] = {0.0f, 0.0f};
  if constexpr (T::sink) {
#pragma unroll
    for (int r = 0; r < 2; ++r)
      if (row_live[r]) sink_logit[r] = sinks[head[r]] * log2e;
  }

  // Both halves of the Q fragments, zero past D and for rows past the slice.
  const uint32_t* q_words = reinterpret_cast<const uint32_t*>(qkv);
  ptx::A4 query[2][T::dk / 16];
#pragma unroll
  for (int half = 0; half < 2; ++half)
#pragma unroll
    for (int ks = 0; ks < T::dk / 16; ++ks)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int r = i % 2;
        const int column = ks * 16 + 2 * c + (i / 2) * 8;
        query[half][ks].v[i] = row_live[r] && column < T::d ?
            q_words[(uint64_t(token[r]) * P::row + half * P::q_lo +
                     head[r] * T::d + column) / 2] : 0u;
      }
  float maximum[2] = {-INFINITY, -INFINITY};
  float denominator[2] = {0.0f, 0.0f};
  ptx::D4 out[T::d / 8];
#pragma unroll
  for (int n = 0; n < T::d / 8; ++n)
#pragma unroll
    for (int i = 0; i < 4; ++i) out[n].v[i] = 0.0f;

  auto accumulate = [&](auto terms_tag, int stage, int key0) {
    constexpr int count = decltype(terms_tag)::value;
    const uint8_t* k_hi = tile(stage, 0);
    const uint8_t* k_lo = tile(stage, 1);
    const uint8_t* v_hi = tile(stage, 2);
    const uint8_t* v_lo = tile(stage, 3);
    ptx::D4 score[T::keys / 8];
#pragma unroll
    for (int n = 0; n < T::keys / 8; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) score[n].v[i] = 0.0f;
#pragma unroll
    for (int ks = 0; ks < T::dk / 16; ++ks)
#pragma unroll
      for (int np = 0; np < T::keys / 16; ++np) {
        const int row = np * 16 + (lane / 16) * 8 + lane % 8;
        const int offset = row * T::stride + (ks * 16 + ((lane / 8) % 2) * 8) * 2;
        uint32_t bh[4], bl[4];
        ptx::ldmatrix_b16<4>(bh, k_hi + offset);
        ptx::ldmatrix_b16<4>(bl, k_lo + offset);
#pragma unroll
        for (int n = 0; n < 2; ++n) {
          const ptx::B2 high{{bh[2 * n], bh[2 * n + 1]}};
          const ptx::B2 low{{bl[2 * n], bl[2 * n + 1]}};
          ptx::mma_bf16(score[2 * np + n], query[0][ks], high);
          ptx::mma_bf16(score[2 * np + n], query[0][ks], low);
          ptx::mma_bf16(score[2 * np + n], query[1][ks], high);
        }
      }
#pragma unroll
    for (int n = 0; n < T::keys / 8; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) score[n].v[i] *= score_scale;
    const bool cut = key0 < all_lo || key0 + T::keys - 1 > all_hi;
    const bool sunk = T::sink && key0 <= sink_hi && key0 + T::keys - 1 >= sink_lo;
    if (cut || sunk) {
#pragma unroll
      for (int n = 0; n < T::keys / 8; ++n)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int r = i / 2;
          const int key = key0 + n * 8 + 2 * c + i % 2;
          float x = score[n].v[i];
          if constexpr (T::sink) x += key == start[r] ? sink_logit[r] : 0.0f;
          score[n].v[i] = key < lo[r] || key > hi[r] ? -INFINITY : x;
        }
    }
    float correction[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      float tile_max = -INFINITY;
#pragma unroll
      for (int n = 0; n < T::keys / 8; ++n)
        tile_max = fmaxf(tile_max, fmaxf(score[n].v[2 * r], score[n].v[2 * r + 1]));
      tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 1));
      tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 2));
      const float next = fmaxf(maximum[r], tile_max);
      const float base = next == -INFINITY ? 0.0f : next;
      correction[r] = exp2f(maximum[r] - base);
      maximum[r] = next;
      float sum = 0.0f;
#pragma unroll
      for (int n = 0; n < T::keys / 8; ++n)
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
      for (int n = 0; n < T::d / 8; ++n) {
        out[n].v[0] *= correction[0];
        out[n].v[1] *= correction[0];
        out[n].v[2] *= correction[1];
        out[n].v[3] *= correction[1];
      }
    }
    // The tile's products accumulate from zero and join `out` in rounded
    // F32 adds: the tensor core's F32 accumulation truncates against its
    // accumulator, which over tens of thousands of keys biases a running sum.
    ptx::D4 tile[T::d / 8];
#pragma unroll
    for (int n = 0; n < T::d / 8; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) tile[n].v[i] = 0.0f;
#pragma unroll
    for (int kk = 0; kk < T::keys / 16; ++kk) {
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
      const int row = kk * 16 + ((lane / 8) % 2) * 8 + lane % 8;
      // Every probability term times both halves of v.
      auto multiply = [&](ptx::D4& o, const uint32_t* high, const uint32_t* low) {
#pragma unroll
        for (int term = 0; term < count; ++term) {
          ptx::mma_bf16(o, probability[term], ptx::B2{{high[0], high[1]}});
          ptx::mma_bf16(o, probability[term], ptx::B2{{low[0], low[1]}});
        }
      };
#pragma unroll
      for (int vn = 0; vn < T::d / 16; ++vn) {
        const int offset = row * T::stride + (vn * 16 + (lane / 16) * 8) * 2;
        uint32_t bh[4], bl[4];
        ptx::ldmatrix_b16<4, true>(bh, v_hi + offset);
        ptx::ldmatrix_b16<4, true>(bl, v_lo + offset);
        multiply(tile[2 * vn], bh, bl);
        multiply(tile[2 * vn + 1], bh + 2, bl + 2);
      }
      if constexpr (T::d % 16 != 0) {
        // The last eight V columns of D = 72.
        const int offset = row * T::stride + (T::d - 8) * 2;
        uint32_t bh[2], bl[2];
        ptx::ldmatrix_b16<2, true>(bh, v_hi + offset);
        ptx::ldmatrix_b16<2, true>(bl, v_lo + offset);
        multiply(tile[T::d / 8 - 1], bh, bl);
      }
    }
#pragma unroll
    for (int n = 0; n < T::d / 8; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) out[n].v[i] += tile[n].v[i];
  };

  for (int key_tile = 0; key_tile < key_tiles; ++key_tile) {
    const int stage = key_tile % T::stages;
    if (key_tile + 1 < key_tiles) cp_async_wait<1>(); else cp_async_wait<0>();
    __syncthreads();
    const int key0 = key_first + key_tile * T::keys;
    if (warp_live && key0 <= warp_hi && key0 + T::keys > warp_lo) {
      if (terms == 2) {
        accumulate(std::integral_constant<int, 2>{}, stage, key0);
      } else {
        accumulate(std::integral_constant<int, 3>{}, stage, key0);
      }
    }
    __syncthreads();
    if (key_tile + T::stages < key_tiles) issue(key_tile + T::stages, stage);
  }

#pragma unroll
  for (int r = 0; r < 2; ++r) {
    float total = denominator[r];
    total += __shfl_xor_sync(0xffffffff, total, 1);
    total += __shfl_xor_sync(0xffffffff, total, 2);
    if (!row_live[r]) continue;
    const float normalizer = 1.0f / total;
    float2* destination = reinterpret_cast<float2*>(output) +
        (uint64_t(token[r]) * T::q_heads + head[r]) * (T::d / 2);
#pragma unroll
    for (int n = 0; n < T::d / 8; ++n)
      destination[n * 4 + c] = make_float2(out[n].v[2 * r] * normalizer,
                                           out[n].v[2 * r + 1] * normalizer);
  }
}

} // namespace chop::attention::varlen
