#pragma once

// Gated attention rows (decision 0084, Qwen3.8 Flash): the per-head Gemma
// Q/K norm with interleaved M-RoPE and the paged KV write, and the sigmoid
// output gate. The arithmetic helpers are shared with the decode kernels
// that fold the norm, RoPE and KV write in, so both produce the same bits.
// Packed rows are [row][q_heads * 2 qk | kv_heads * qk | kv_heads * v]: each
// Q head's query is followed by its gate logits. Step rows follow the QSA
// kernels: `row_batch[row]` is the row's sequence (block-table row),
// `positions[row]` its cache position, `rope_positions[row]` its (T, H, W)
// RoPE positions, and `live_rows` the live count. The F32 table is
// `cos_sin[position][pair][cos, sin]`.

#include <cuda_bf16.h>
#include <stdint.h>
#include "tile.cuh"
#include "../common/rotary.cuh"
#include "../common/sigmoid.cuh"

namespace chop::attention::gated {

// The position axis (0 = T, 1 = H, 2 = W) of rotary pair `pair`.
template<int SectionH, int SectionW>
__device__ __forceinline__ int axis(int pair) {
  return pair % 3 == 1 && pair < 3 * SectionH ? 1 :
         pair % 3 == 2 && pair < 3 * SectionW ? 2 : 0;
}

// Cosine and sine of rotary pair `pair` for a row whose RoPE positions are
// `rope_position[0..3]`.
template<int Rotary, int SectionH, int SectionW>
__device__ __forceinline__ float2 table_angle(const float* cos_sin, const int32_t* rope_position,
                                              int pair) {
  const uint64_t position = uint32_t(rope_position[axis<SectionH, SectionW>(pair)]);
  return reinterpret_cast<const float2*>(cos_sin)[position * (Rotary / 2) + pair];
}

// `1 / sqrt(sum / Dim + eps)`, rounded alike wherever it runs.
template<int Dim>
__device__ __forceinline__ float inverse_rms(float sum, float eps) {
  return __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fmul_rn(sum, 1.0f / Dim), eps)));
}

// Gemma scaling of one element: `x * inverse * (1 + w)`.
__device__ __forceinline__ float normalize(float x, float inverse, uint16_t weight) {
  return __fmul_rn(__fmul_rn(x, inverse), __fadd_rn(1.0f, convert::bf16_to_f32(weight)));
}

__device__ __forceinline__ uint8_t fp8_code(float value, float scale) {
  return uint8_t(convert::pack_f32_e4m3({__fdiv_rn(value, scale), 0.f}));
}

// One Q head held by the four lanes of a quad (c = lane % 4), as decode's
// Q^T B fragments hold it: word 2 ks + w has dimensions 16 ks + 8 w + 2c and
// one more. Norms the head over the quad and rotates NeoX pairs (d, d +
// Rotary / 2) of the first Rotary dimensions, whose words are 2 Rotary / 32
// apart in the same lane; `angle(pair)` gives a pair's cosine and sine and
// is called by every lane alike. Every lane of the warp calls this.
template<int Qk, int Rotary, class Angle>
__device__ __forceinline__ void norm_rotate_query(uint32_t (&words)[Qk / 8], const uint16_t* weight,
                                                  float eps, Angle angle) {
  static_assert(Rotary % 32 == 0 && Rotary <= Qk);
  const int c = threadIdx.x % 4;
  float x[Qk / 4];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < Qk / 8; ++i) {
    x[2 * i] = bf16_low(words[i]);
    x[2 * i + 1] = bf16_high(words[i]);
    sum = __fmaf_rn(x[2 * i], x[2 * i], sum);
    sum = __fmaf_rn(x[2 * i + 1], x[2 * i + 1], sum);
  }
  sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, 1));
  sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, 2));
  const float inverse = inverse_rms<Qk>(sum, eps);
#pragma unroll
  for (int i = 0; i < Qk / 8; ++i) {
    const int d = 16 * (i / 2) + 8 * (i % 2) + 2 * c;
    const uint32_t w = *reinterpret_cast<const uint32_t*>(weight + d);
    x[2 * i] = normalize(x[2 * i], inverse, uint16_t(w));
    x[2 * i + 1] = normalize(x[2 * i + 1], inverse, uint16_t(w >> 16));
  }
  constexpr int mate = 2 * Rotary / 32;
#pragma unroll
  for (int i = 0; i < mate; ++i)
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const float2 cs = angle(16 * (i / 2) + 8 * (i % 2) + 2 * c + e);
      const float a = x[2 * i + e], b = x[2 * (i + mate) + e];
      x[2 * i + e] = rotary::rotate(a, b, cs, false);
      x[2 * (i + mate) + e] = rotary::rotate(b, a, cs, true);
    }
#pragma unroll
  for (int i = 0; i < Qk / 8; ++i) words[i] = convert::bf16x2_from_f32({x[2 * i], x[2 * i + 1]});
}

// One 256-wide K head held by a warp, lane l holding dimensions 8l to 8l + 7:
// the norm over the warp, then NeoX RoPE over the first Rotary dimensions,
// whose pair mates sit Rotary / 16 lanes apart. Every lane calls `angle`.
template<int Rotary, class Angle>
__device__ __forceinline__ void norm_rotate_key(float (&x)[8], const uint16_t* weight, float eps,
                                                Angle angle) {
  static_assert(Rotary % 16 == 0 && Rotary <= 256);
  const int lane = threadIdx.x % 32;
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) sum = __fmaf_rn(x[i], x[i], sum);
#pragma unroll
  for (int offset = 16; offset; offset /= 2)
    sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, offset));
  const float inverse = inverse_rms<256>(sum, eps);
  const uint4 w = *reinterpret_cast<const uint4*>(weight + 8 * lane);
  const uint32_t words[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
  for (int i = 0; i < 8; ++i)
    x[i] = normalize(x[i], inverse, uint16_t(words[i / 2] >> (16 * (i % 2))));
  constexpr int half = Rotary / 16;
  const bool rotated = lane < 2 * half;
  const bool second = (lane % (2 * half)) >= half;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float mate = __shfl_xor_sync(0xffffffff, x[i], half);
    const float2 cs = angle((8 * lane + i) % (Rotary / 2));
    if (rotated) x[i] = rotary::rotate(x[i], mate, cs, second);
  }
}

// Head 128 assigns four consecutive dimensions to each lane. Keep the
// eight-dimension path's arithmetic unchanged for head 256.
template<int Qk, int Rotary, class Angle>
__device__ __forceinline__ void norm_rotate_key(float (&x)[Qk / 32], const uint16_t* weight,
                                                float eps, Angle angle) {
  static_assert(Qk == 128 || Qk == 256);
  if constexpr (Qk == 256) {
    norm_rotate_key<Rotary>(x, weight, eps, angle);
  } else {
    static_assert(Rotary == 32 || Rotary == 64);
    const int lane = threadIdx.x % 32;
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) sum = __fmaf_rn(x[i], x[i], sum);
#pragma unroll
    for (int offset = 16; offset; offset /= 2)
      sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, offset));
    const float inverse = inverse_rms<Qk>(sum, eps);
    const uint2 w = *reinterpret_cast<const uint2*>(weight + 4 * lane);
    const uint32_t words[2] = {w.x, w.y};
#pragma unroll
    for (int i = 0; i < 4; ++i)
      x[i] = normalize(x[i], inverse, uint16_t(words[i / 2] >> (16 * (i % 2))));
    constexpr int half = Rotary / 8;
    const bool rotated = lane < 2 * half;
    const bool second = (lane % (2 * half)) >= half;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float mate = __shfl_xor_sync(0xffffffff, x[i], half);
      const float2 cs = angle((4 * lane + i) % (Rotary / 2));
      if (rotated) x[i] = rotary::rotate(x[i], mate, cs, second);
    }
  }
}

// One CTA of eight warps per row. Warp w < ceil(QHeads / 8) norms and
// rotates Q heads 8w to 8w + 7 (lane l: head 8w + l / 4, quarter l % 4) into
// packed BF16 `q[row][q_heads][qk]`; the next KvHeads warps each norm and
// rotate one K head and write it with its V row to the paged cache, BF16 or
// FP8 E4M3 divided by the per-layer scale.
template<int QHeads, int KvHeads, int Qk, int V, int Rotary, int SectionH, int SectionW,
         int Page, bool Fp8>
__device__ void qk_norm_rope_kv(
    const uint16_t* qkv, const uint16_t* q_norm, const uint16_t* k_norm,
    const int32_t* rope_positions, const float* cos_sin, const int32_t* row_batch,
    const int32_t* positions, const uint32_t* live_rows, const int32_t* block_tables,
    void* k_cache, void* v_cache, uint16_t* q, uint32_t max_pages,
    uint64_t k_page_stride_bytes, uint64_t v_page_stride_bytes, uint32_t eps_bits,
    uint32_t k_scale_bits, uint32_t v_scale_bits) {
  static_assert((Qk == 128 || Qk == 256) && V == Qk, "a warp holds one K or V head");
  constexpr int q_warps = (QHeads + 7) / 8;
  static_assert(q_warps + KvHeads <= 8, "one CTA holds a row's heads");
  constexpr int stride = QHeads * 2 * Qk + KvHeads * (Qk + V);
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const float eps = __uint_as_float(eps_bits);
  const int32_t* rope_position = rope_positions + 3 * uint64_t(row);
  auto angle = [&](int pair) {
    return table_angle<Rotary, SectionH, SectionW>(cos_sin, rope_position, pair);
  };
  const uint16_t* source = qkv + uint64_t(row) * stride;
  if (warp < q_warps) {
    const int head = warp * 8 + lane / 4;
    const bool live = head < QHeads;
    const uint32_t* words = reinterpret_cast<const uint32_t*>(source + (live ? head : 0) * 2 * Qk);
    uint32_t held[Qk / 8];
#pragma unroll
    for (int i = 0; i < Qk / 8; ++i) held[i] = live ? words[(i / 2) * 8 + (i % 2) * 4 + lane % 4] : 0u;
    norm_rotate_query<Qk, Rotary>(held, q_norm, eps, angle);
    if (!live) return;
    uint32_t* out = reinterpret_cast<uint32_t*>(q + (uint64_t(row) * QHeads + head) * Qk);
#pragma unroll
    for (int i = 0; i < Qk / 8; ++i) out[(i / 2) * 8 + (i % 2) * 4 + lane % 4] = held[i];
    return;
  }
  const int kv_head = warp - q_warps;
  if (kv_head >= KvHeads) return;
  if constexpr (Qk == 128) {
    const uint2 k_bits = *reinterpret_cast<const uint2*>(
        source + QHeads * 2 * Qk + kv_head * Qk + 4 * lane);
    const uint2 v_bits = *reinterpret_cast<const uint2*>(
        source + QHeads * 2 * Qk + KvHeads * Qk + kv_head * V + 4 * lane);
    float k[4] = {bf16_low(k_bits.x), bf16_high(k_bits.x),
                  bf16_low(k_bits.y), bf16_high(k_bits.y)};
    norm_rotate_key<Qk, Rotary>(k, k_norm, eps, angle);
    const int batch = row_batch[row];
    const int position = positions[row];
    const uint64_t page = uint32_t(block_tables[uint64_t(batch) * max_pages + position / Page]);
    const uint64_t within = uint64_t(position % Page) * KvHeads + kv_head;
    constexpr int bytes = Fp8 ? 1 : 2;
    uint8_t* k_row = static_cast<uint8_t*>(k_cache) + page * k_page_stride_bytes +
                     within * Qk * bytes + 4 * lane * bytes;
    uint8_t* v_row = static_cast<uint8_t*>(v_cache) + page * v_page_stride_bytes +
                     within * V * bytes + 4 * lane * bytes;
    if constexpr (Fp8) {
      const float k_scale = __uint_as_float(k_scale_bits);
      const float v_scale = __uint_as_float(v_scale_bits);
      const uint32_t v_words[2] = {v_bits.x, v_bits.y};
      uint32_t k_codes = 0, v_codes = 0;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float v = i % 2 ? bf16_high(v_words[i / 2]) : bf16_low(v_words[i / 2]);
        k_codes |= uint32_t(fp8_code(k[i], k_scale)) << (8 * i);
        v_codes |= uint32_t(fp8_code(v, v_scale)) << (8 * i);
      }
      *reinterpret_cast<uint32_t*>(k_row) = k_codes;
      *reinterpret_cast<uint32_t*>(v_row) = v_codes;
    } else {
      *reinterpret_cast<uint2*>(k_row) = make_uint2(
          convert::bf16x2_from_f32({k[0], k[1]}), convert::bf16x2_from_f32({k[2], k[3]}));
      *reinterpret_cast<uint2*>(v_row) = v_bits;
    }
  } else {
    const uint4 k_bits = *reinterpret_cast<const uint4*>(
        source + QHeads * 2 * Qk + kv_head * Qk + 8 * lane);
    const uint4 v_bits = *reinterpret_cast<const uint4*>(
        source + QHeads * 2 * Qk + KvHeads * Qk + kv_head * V + 8 * lane);
    float k[8];
    const uint32_t k_words[4] = {k_bits.x, k_bits.y, k_bits.z, k_bits.w};
#pragma unroll
    for (int i = 0; i < 8; ++i) k[i] = i % 2 ? bf16_high(k_words[i / 2]) : bf16_low(k_words[i / 2]);
    norm_rotate_key<Rotary>(k, k_norm, eps, angle);
    const int batch = row_batch[row];
    const int position = positions[row];
    const uint64_t page = uint32_t(block_tables[uint64_t(batch) * max_pages + position / Page]);
    const uint64_t within = uint64_t(position % Page) * KvHeads + kv_head;
    constexpr int bytes = Fp8 ? 1 : 2;
    uint8_t* k_row = static_cast<uint8_t*>(k_cache) + page * k_page_stride_bytes +
                     within * Qk * bytes + 8 * lane * bytes;
    uint8_t* v_row = static_cast<uint8_t*>(v_cache) + page * v_page_stride_bytes +
                     within * V * bytes + 8 * lane * bytes;
    if constexpr (Fp8) {
      const float k_scale = __uint_as_float(k_scale_bits);
      const float v_scale = __uint_as_float(v_scale_bits);
      const uint32_t v_words[4] = {v_bits.x, v_bits.y, v_bits.z, v_bits.w};
      uint32_t k_codes[2] = {0, 0}, v_codes[2] = {0, 0};
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float v = i % 2 ? bf16_high(v_words[i / 2]) : bf16_low(v_words[i / 2]);
        k_codes[i / 4] |= uint32_t(fp8_code(k[i], k_scale)) << (8 * (i % 4));
        v_codes[i / 4] |= uint32_t(fp8_code(v, v_scale)) << (8 * (i % 4));
      }
      *reinterpret_cast<uint2*>(k_row) = make_uint2(k_codes[0], k_codes[1]);
      *reinterpret_cast<uint2*>(v_row) = make_uint2(v_codes[0], v_codes[1]);
    } else {
      (void)k_scale_bits;
      (void)v_scale_bits;
      *reinterpret_cast<uint4*>(k_row) = make_uint4(
          convert::bf16x2_from_f32({k[0], k[1]}), convert::bf16x2_from_f32({k[2], k[3]}),
          convert::bf16x2_from_f32({k[4], k[5]}), convert::bf16x2_from_f32({k[6], k[7]}));
      *reinterpret_cast<uint4*>(v_row) = v_bits;
    }
  }
}

// `attention * sigmoid(gate)` for one row per CTA: attention is packed BF16
// [row][heads][dim] and the gate logits are the second half of each Q head
// in the packed rows. The output may alias the attention input.
template<int Heads, int KvHeads, int Dim, int V>
__device__ void output_gate(const uint16_t* attention, const uint16_t* qkv, uint16_t* out,
                            const uint32_t* live_rows) {
  constexpr int stride = Heads * 2 * Dim + KvHeads * (Dim + V);
  constexpr int vectors = Heads * Dim / 8;
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  for (int i = threadIdx.x; i < vectors; i += blockDim.x) {
    const int head = i / (Dim / 8);
    const int d = i % (Dim / 8) * 8;
    const uint4 x = *reinterpret_cast<const uint4*>(attention + (uint64_t(row) * Heads + head) * Dim + d);
    const uint4 z = *reinterpret_cast<const uint4*>(
        qkv + uint64_t(row) * stride + head * 2 * Dim + Dim + d);
    const uint32_t xs[4] = {x.x, x.y, x.z, x.w};
    const uint32_t zs[4] = {z.x, z.y, z.z, z.w};
    uint32_t ys[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      // Below the sigmoid's tail the product is formed in F64.
      auto gate = [](float a, float logit) {
        return logit >= chop::sigmoid::kTail
            ? __fmul_rn(a, __fdiv_rn(1.0f, __fadd_rn(1.0f, __expf(-logit))))
            : chop::sigmoid::tail_times(a, logit);
      };
      ys[j] = convert::bf16x2_from_f32({gate(bf16_low(xs[j]), bf16_low(zs[j])),
                                        gate(bf16_high(xs[j]), bf16_high(zs[j]))});
    }
    *reinterpret_cast<uint4*>(out + (uint64_t(row) * Heads + head) * Dim + d) =
        make_uint4(ys[0], ys[1], ys[2], ys[3]);
  }
}

} // namespace chop::attention::gated
