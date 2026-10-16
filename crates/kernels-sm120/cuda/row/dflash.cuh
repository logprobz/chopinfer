#pragma once

#include "argmax.cuh"
#include "rope_kv.cuh"

namespace chop::row {

template<int Heads>
__device__ void dflash_qk_norm(const uint16_t* input, const uint16_t* weight,
                               uint16_t* output, const uint32_t* live_tokens,
                               uint32_t input_stride, uint32_t output_stride,
                               float eps) {
  ptx::griddep_launch_dependents();
  __shared__ float warp_sums[8];
  ptx::griddep_wait();
  const uint32_t token = blockIdx.x, head = blockIdx.y;
  if (token >= *live_tokens || head >= Heads) return;
  const uint64_t in_base = uint64_t(token) * input_stride + head * 128;
  const uint64_t out_base = uint64_t(token) * output_stride + head * 128;
  float x = bf(input[in_base + threadIdx.x]);
  float square = x * x;
  float total = sum_cta(square, warp_sums);
  float inv = rsqrtf(total * (1.f / 128.f) + eps);
  output[out_base + threadIdx.x] = to_bf(x * inv * bf(weight[threadIdx.x]));
}

template<int Heads, bool Pos64, bool Table = false>
__device__ void dflash_rope(const uint16_t* input, uint16_t* output,
                           const void* positions, const uint32_t* live_tokens,
                           uint32_t input_stride, uint32_t output_stride,
                           const float* cos_sin = nullptr) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t token = blockIdx.x, head = blockIdx.y;
  if (token >= *live_tokens || head >= Heads) return;
  int64_t position = Pos64 ? reinterpret_cast<const int64_t*>(positions)[token]
                           : reinterpret_cast<const int32_t*>(positions)[token];
  const uint64_t in_base = uint64_t(token) * input_stride + head * 128;
  const uint64_t out_base = uint64_t(token) * output_stride + head * 128;
  output[out_base + threadIdx.x] = to_bf(rotated<64, true, Table>(
      input + in_base, threadIdx.x, position, 10000.f, cos_sin));
}

// Each rank writes one (score, global ID) pair per row, under the shared
// argmax's NaN rule (`argmax.cuh`). The caller gathers those pairs in rank
// order before launching dflash_argmax_final.
template<int Capacity>
__device__ void dflash_argmax_partial(const uint16_t* logits,
                                     float* scores_out, int32_t* ids_out,
                                     const uint32_t* live_rows,
                                     const uint32_t* live_width,
                                     uint32_t vocab_offset) {
  argmax::partial(logits, scores_out, ids_out, live_rows, live_width,
                  uint32_t(Capacity), vocab_offset);
}

// Each row's pairs at `[rows, Ranks]`, into `output[row]`.
template<int Ranks>
__device__ void dflash_argmax_final(const float* scores,
                                   const int32_t* ids, int32_t* output,
                                   const uint32_t* live_rows) {
  argmax::final_pairs(scores, ids, output, nullptr, nullptr, nullptr, nullptr,
                      live_rows, Ranks, Ranks, 1, 0, 0);
}

} // namespace chop::row
