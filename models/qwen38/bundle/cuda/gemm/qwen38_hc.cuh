#pragma once

// Qwen3.8 decode hyper-connection input tail in one launch: the rank-320
// bottleneck, the up projection (K = 320) and the stream mix,
//   u[t][c] = sum_s sigmoid(up[t][s][c]) * xn[t][s][c] / 4,
// bitwise as chop_row_qwen38_hc_bottleneck_v1, chop_gemm_qwen38_hc_up_decode_v1
// and chop_row_qwen38_hc_mix_v1 in turn: the staged activation holds the
// bottleneck's rounded BF16 values, each warp runs the short-K kernel's MMA
// sequence over K for one stream's strip, rounds up to BF16 as its store
// does, and the mix sums the streams in order.

#include "qwen38.cuh"
#include "row/qwen38.cuh"

namespace chop::gemm::qwen38 {

// CTA x owns columns [16x, 16x + 16) of every stream: warp s computes strip
// s * 160 + x of the up weight for all Tokens rows; rows past the live count
// are zero and never stored.
template<int Tokens, bool Padded = false>
struct UpMixTraits {
  static_assert(Tokens == 32, "one token tile holds every decode bucket");
  static constexpr int k = CHOP_QWEN38_HC_RANK;
  static constexpr int k_tiles = k / 16;
  static constexpr int streams = CHOP_QWEN38_STREAMS;
  static constexpr int width = CHOP_QWEN38_HIDDEN;
  static constexpr int strips_per_stream = width / 16;
  static constexpr int row_bytes = k * 2 + 16;
  static constexpr int activation_bytes = Tokens * row_bytes;
  // Padding changes only shared addresses, never the stream reduction order.
  static constexpr int term_stride = Tokens + (Padded ? 1 : 0);
  static constexpr int term_bytes = streams * 16 * term_stride * 4;
  static constexpr int shared_bytes = activation_bytes + term_bytes;
};

template<int Tokens, bool Precomputed = false, bool Padded = false>
__device__ void run_hc_up_mix(
    const uint8_t* weight, const uint16_t* down, const uint16_t* normed,
    uint16_t* output, const uint32_t* live_rows) {
  using T = UpMixTraits<Tokens, Padded>;
  extern __shared__ __align__(16) unsigned char storage[];
  float* terms = reinterpret_cast<float*>(storage + T::activation_bytes);
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const uint32_t strip = warp * T::strips_per_stream + blockIdx.x;
  ptx::A4 a[T::k_tiles];
  const uint8_t* base = weight + uint64_t(strip) * T::k_tiles * 512 + lane * 16;
  ptx::griddep_launch_dependents();
#pragma unroll
  for (int t = 0; t < T::k_tiles; ++t) a[t] = load_weight_fragment(base + t * 512);
  ptx::griddep_wait();
  const uint32_t live = *live_rows;
  if (live > Tokens) asm volatile("trap;");
  constexpr uint32_t vectors = T::k / 8;
  for (uint32_t i = threadIdx.x; i < Tokens * vectors; i += blockDim.x) {
    const uint32_t token = i / vectors;
    uint4 value = make_uint4(0, 0, 0, 0);
    if (token < live) {
      const uint4 raw = reinterpret_cast<const uint4*>(down + uint64_t(token) * T::k)[i % vectors];
      if constexpr (Precomputed) value = raw;
      else {
      uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        uint16_t half[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const float v = row::bf(static_cast<uint16_t>(words[w] >> (16 * h))) / T::streams;
          half[h] = row::to_bf(chop::sigmoid::times(v, v));
        }
        words[w] = uint32_t(half[0]) | (uint32_t(half[1]) << 16);
      }
      value = make_uint4(words[0], words[1], words[2], words[3]);
      }
    }
    *reinterpret_cast<uint4*>(storage + token * T::row_bytes + (i % vectors) * 16) = value;
  }
  __syncthreads();
  const uint32_t b_row = (lane & 7) + ((lane >> 4) << 3);
  const uint32_t b_address = ptx::shared_address(storage) +
      b_row * T::row_bytes + ((lane >> 3) & 1) * 16;
  // run_short_k's order: every 8 k16 tiles and the last accumulate from
  // zero into `block`, then add into acc.
  ptx::D4 acc[Tokens / 8], block[Tokens / 8];
#pragma unroll
  for (int j = 0; j < Tokens / 8; ++j) acc[j] = block[j] = ptx::D4{};
#pragma unroll
  for (int t = 0; t < T::k_tiles; ++t) {
#pragma unroll
    for (int pair = 0; pair < Tokens / 16; ++pair) {
      ptx::B2 first, second;
      prefill::load_b2(b_address + pair * 16 * T::row_bytes + t * 32, 0, 0,
          first, second);
      ptx::mma_bf16(block[pair * 2], a[t], first);
      ptx::mma_bf16(block[pair * 2 + 1], a[t], second);
    }
    if (t % 8 == 7 || t + 1 == T::k_tiles) {
#pragma unroll
      for (int j = 0; j < Tokens / 8; ++j)
#pragma unroll
        for (int v = 0; v < 4; ++v) {
          acc[j].v[v] += block[j].v[v];
          block[j].v[v] = 0.0f;
        }
    }
  }
  // terms[stream][column][token]; accumulator v: column (lane >> 2) + 8 (v / 2),
  // token j * 8 + (lane & 3) * 2 + v % 2.
  const uint32_t first_column = blockIdx.x * 16;
#pragma unroll
  for (int j = 0; j < Tokens / 8; ++j)
#pragma unroll
    for (int v = 0; v < 4; ++v) {
      const uint32_t column = (lane >> 2) + 8 * (v / 2);
      const uint32_t token = j * 8 + (lane & 3) * 2 + v % 2;
      if (token >= live) continue;
      const uint16_t up = row::to_bf(acc[j].v[v]);
      const uint64_t i = uint64_t(token) * T::streams * T::width + warp * T::width +
          first_column + column;
      terms[(warp * 16 + column) * T::term_stride + token] =
          chop::sigmoid::times(row::bf(normed[i]), row::bf(up)) / T::streams;
    }
  __syncthreads();
  for (uint32_t index = threadIdx.x; index < 16 * live; index += blockDim.x) {
    const uint32_t token = index / 16;
    const uint32_t column = index % 16;
    float sum = 0.f;
#pragma unroll
    for (int s = 0; s < T::streams; ++s) sum += terms[(s * 16 + column) * T::term_stride + token];
    output[uint64_t(token) * T::width + first_column + column] = row::to_bf(sum);
  }
}

} // namespace chop::gemm::qwen38
