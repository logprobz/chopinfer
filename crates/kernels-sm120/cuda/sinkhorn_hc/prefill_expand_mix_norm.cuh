#pragma once

// prefill_expand_mix followed, in the same CTA, by the input norm and ceil
// quantizer of the rows it mixed. The pipeline writes the tile's collapsed
// rows; the tail reads them back while they are in L1/L2 and runs the BF16
// norm (rmsnorm's 256-lane tree and expressions), the optional widened copy
// and ceil_epilogue's quantizer and row status, row by row, as
// prefill_fused's tail does after its own collapse. Each lane keeps its
// collapsed columns in registers and the quantizer reads the stored norm
// output back, so only the codes stage in shared memory beside the
// pipeline's arena.
#include "prefill_expand_mix.cuh"
#include "../row/row.cuh"
#include "../row/ceil_parallel.cuh"

namespace chop::sinkhorn_hc::prefill_expand_mix_norm {

enum class Mutation { none, reverse_expansion, round_before_weight, clear_status };

template<int H> struct Tail {
  __align__(16) uint8_t codes[H];
  uint8_t scales[H / 32];
  float sums[kWarps];
  uint32_t flags[kWarps];
};

template<class M, int Rows, int Chains, uint32_t FloorBits, Mutation Mut>
__device__ void run(const uint16_t* x, const uint16_t* y, const float* post_in,
    const float* comb_in, uint16_t* expanded, const void* fn, const float* base,
    const float* scale, const float* pre_in, uint16_t* collapsed, float* pre,
    float* post, float* comb, const uint16_t* norm_weight, uint16_t* normed,
    float* widened, uint8_t* codes, uint8_t* scales, const uint32_t* live_rows,
    uint32_t* status, uint64_t status_offset, uint32_t norm_eps_bits) {
  constexpr int H = M::kWidth;
  static_assert(H % 16 == 0);
  // rmsnorm's lane t owns columns 8t + 2048k.
  constexpr int Passes = (H + kThreads * 8 - 1) / (kThreads * 8);
  constexpr auto Inner = Mut == Mutation::reverse_expansion
      ? prefill_expand_mix::Mutation::reverse_expansion
      : prefill_expand_mix::Mutation::none;
  __shared__ Tail<H> tail;
  prefill_expand_mix::run<M, Rows, Chains, Inner>(x, y, post_in, comb_in, expanded, fn, base,
                                                  scale, pre_in, collapsed, pre, post, comb,
                                                  live_rows);
  // The same live count as the mixer: a CTA it skipped skips the tail.
  const uint32_t first = blockIdx.x * Rows, live = *live_rows;
  if (first >= live) return;
  const int count = live - first < uint32_t(Rows) ? int(live - first) : Rows;
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const float norm_eps = __uint_as_float(norm_eps_bits);
  // Publish the pipeline's collapsed rows to every thread of the CTA.
  __threadfence_block();
  __syncthreads();
  for (int r = 0; r < count; ++r) {
    const uint32_t row = first + r;
    const uint64_t at = uint64_t(row) * H;
    float square = 0.f;
    uint4 kept[Passes];
    #pragma unroll
    for (int k = 0; k < Passes; ++k) {
      const int col = t * 8 + k * kThreads * 8;
      if (col < H) {
        kept[k] = *reinterpret_cast<const uint4*>(collapsed + at + col);
        const auto* in = reinterpret_cast<const uint16_t*>(&kept[k]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
          const float v = chop::row::bf(in[j]);
          square += v * v;
        }
      }
    }
    const float total = chop::row::sum_cta(square, tail.sums);
    const float inv = rsqrtf(total / H + norm_eps);
    #pragma unroll
    for (int k = 0; k < Passes; ++k) {
      const int col = t * 8 + k * kThreads * 8;
      if (col >= H) continue;
      uint4 packed = kept[k];
      auto* values = reinterpret_cast<uint16_t*>(&packed);
      const uint4 wp = *reinterpret_cast<const uint4*>(norm_weight + col);
      const auto* w = reinterpret_cast<const uint16_t*>(&wp);
      #pragma unroll
      for (int j = 0; j < 8; ++j) {
        float n = chop::row::bf(values[j]) * inv;
        if constexpr (Mut == Mutation::round_before_weight) n = chop::row::bf(chop::row::to_bf(n));
        values[j] = chop::row::to_bf(n * chop::row::bf(w[j]));
        if (widened) widened[at + col + j] = chop::row::bf(values[j]);
      }
      *reinterpret_cast<uint4*>(normed + at + col) = packed;
    }
    if constexpr (Mut == Mutation::clear_status)
      if (t == 0) status[status_offset + row] = 0;
    __syncthreads();
    uint32_t flags = 0;
    #pragma unroll 1
    for (int group = warp; group < H / 32; group += kWarps) {
      const int col = group * 32 + 2 * lane;
      const auto value =
          chop::row::block_quantize_parallel::quantize_group<0, false, false, FloorBits>(
              normed + at, col);
      flags |= value.flags;
      if (lane == 0) tail.scales[group] = value.scale_code;
      if (lane < 16) {
        tail.codes[col] = uint8_t(value.codes);
        tail.codes[col + 1] = uint8_t(value.codes >> 8);
      }
    }
    flags = chop::row::block_quantize_parallel::row_status<kWarps>(flags, tail.flags, status,
                                                                  status_offset + row);
    for (int col = t * 16; col < H; col += kThreads * 16)
      *reinterpret_cast<uint4*>(codes + at + col) =
          flags ? make_uint4(0, 0, 0, 0) : *reinterpret_cast<const uint4*>(tail.codes + col);
    for (int group = t; group < H / 32; group += kThreads)
      scales[uint64_t(row) * (H / 32) + group] = flags ? 127 : tail.scales[group];
    // The next row reuses the tail arena.
    __syncthreads();
  }
}

} // namespace chop::sinkhorn_hc::prefill_expand_mix_norm
