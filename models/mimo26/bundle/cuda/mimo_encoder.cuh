#pragma once

// MiMo's own encoder row kernels (model-mimo26-contract encoders::rows): the
// residual quantizer and the 20-table speech embedding sum of its
// f32-pairs.v1 audio stages. One CTA per row; the live row count is read on the device.

#include "encoder/rows/common.cuh"

namespace chop::mimo_encoder {

using chop::multimodal::bf;

// Sum of per-code BF16 table rows in F32, in table order, kept in F32
// (`mimo26:embed.sum20-f32.v1`).
template<int Tables, int Vocab, int Width>
__device__ void embedding_sum_f32(const uint16_t* tables, const int32_t* codes,
                                  float* output, const uint32_t* live_rows) {
  chop::multimodal::begin();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  for (int col = threadIdx.x; col < Width; col += blockDim.x) {
    float sum = 0.f;
    for (int table = 0; table < Tables; ++table) {
      const int32_t code = codes[uint64_t(row) * Tables + table];
      if (code >= 0 && code < Vocab)
        sum = __fadd_rn(sum, bf(tables[(uint64_t(table) * Vocab + code) * Width + col]));
    }
    output[uint64_t(row) * Width + col] = sum;
  }
}

// Residual vector quantization of F32 rows over F32 codebooks
// (`mimo26:quantize.rvq20-f32.v1`). Per stage, the code nearest the
// remaining residual (F32 sum of squared differences, ties to the lower
// code), subtracted from it.
template<int Dim, int Stages, int MaxCodes>
__device__ void rvq_f32(const float* input, const float* codebooks,
                        const int32_t* offsets, int32_t* codes,
                        const uint32_t* live_rows) {
  ptx::griddep_launch_dependents();
  __shared__ float residual[Dim];
  __shared__ float best_distance[256];
  __shared__ int best_code[256];
  __shared__ int selected;
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  for (int col = threadIdx.x; col < Dim; col += blockDim.x)
    residual[col] = input[uint64_t(row) * Dim + col];
  __syncthreads();
  for (int stage = 0; stage < Stages; ++stage) {
    const int start = offsets[stage];
    const int count = offsets[stage + 1] - start;
    float best = INFINITY;
    int best_id = 0;
    for (int code = threadIdx.x; code < count && code < MaxCodes;
         code += blockDim.x) {
      const float* entry = codebooks + uint64_t(start + code) * Dim;
      float distance = 0.f;
      for (int col = 0; col < Dim; ++col) {
        const float delta = residual[col] - entry[col];
        distance = fmaf(delta, delta, distance);
      }
      if (distance < best || (distance == best && code < best_id)) {
        best = distance;
        best_id = code;
      }
    }
    best_distance[threadIdx.x] = best;
    best_code[threadIdx.x] = best_id;
    __syncthreads();
    for (int step = blockDim.x / 2; step; step >>= 1) {
      if (threadIdx.x < step) {
        const float other = best_distance[threadIdx.x + step];
        const int other_id = best_code[threadIdx.x + step];
        if (other < best_distance[threadIdx.x] ||
            (other == best_distance[threadIdx.x] &&
             other_id < best_code[threadIdx.x])) {
          best_distance[threadIdx.x] = other;
          best_code[threadIdx.x] = other_id;
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      selected = best_code[0];
      codes[uint64_t(row) * Stages + stage] = selected;
    }
    __syncthreads();
    for (int col = threadIdx.x; col < Dim; col += blockDim.x)
      residual[col] -= codebooks[uint64_t(start + selected) * Dim + col];
    __syncthreads();
  }
}

} // namespace chop::mimo_encoder
