#pragma once
#include "position_live.cuh"

// The compensated f64 score contract, implemented directly for lab replay.
// Explicit round-to-nearest operations prevent FMA contraction.
namespace chop::position_f64_probe {
struct Sum {
  double sum = 0, correction = 0;
  __device__ void add(double term) {
    const double next = __dadd_rn(sum,term);
    const double error = fabs(sum) >= fabs(term)
        ? __dadd_rn(__dsub_rn(sum,next),term)
        : __dadd_rn(__dsub_rn(term,next),sum);
    correction = __dadd_rn(correction,error);
    sum = next;
  }
  __device__ double value() const { return __dadd_rn(sum,correction); }
};
template<int Page, int Ratio>
__device__ void scores(const uint16_t* queries, const float* weights,
    const uint8_t* codes, const uint8_t* scales, const int32_t* tables,
    const int32_t* positions, const uint32_t* live, double* output,
    double* magnitudes, uint32_t width, uint64_t code_stride, uint64_t scale_stride) {
  const uint32_t row = blockIdx.y;
  if (row >= *live) return;
  __shared__ uint16_t q[4096];
  __shared__ double w[32];
  for (int i = threadIdx.x; i < 4096; i += blockDim.x) q[i] = queries[size_t(row) * 4096 + i];
  if (threadIdx.x < 32) w[threadIdx.x] = __dmul_rn(double(weights[size_t(row) * 32 + threadIdx.x]),0.015625);
  __syncthreads();
  __shared__ double terms[4][32], absolute_terms[4][32];
  const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const uint32_t count = min(position_indexer::visible(positions[row],Ratio),width);
  for (uint32_t base = blockIdx.x * 4; base < width; base += gridDim.x * 4) {
    const uint32_t index = base + warp;
    const bool admitted = position_indexer::admitted<Page,false>(index,count,nullptr,tables);
    Sum dot, magnitude;
    if (admitted) {
      const uint32_t page = uint32_t(tables[index / Page]);
      const uint8_t* k = codes + uint64_t(page) * code_stride + size_t(index % Page) * 64;
      const uint8_t* scale = scales + uint64_t(page) * scale_stride + size_t(index % Page) * 4;
#pragma unroll 1
      for (int c = 0; c < 128; c += 2) {
        const uint32_t pair = convert::scaled_mxfp4_bf16x2<false>(k[c / 2],scale[c / 32]);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const double key = double(position_indexer::bf16(uint16_t(pair >> (16 * i))));
          const double product = __dmul_rn(double(position_indexer::bf16(q[lane * 128 + c + i])),key);
          dot.add(product); magnitude.add(fabs(product));
        }
      }
    }
    terms[warp][lane] = __dmul_rn(fmax(dot.value(),0.0),w[lane]);
    absolute_terms[warp][lane] = __dmul_rn(magnitude.value(),fabs(w[lane]));
    __syncwarp();
    if (lane == 0 && index < width) {
      Sum score, absolute;
      if (admitted) {
#pragma unroll 1
        for (int h = 0; h < 32; ++h) { score.add(terms[warp][h]); absolute.add(absolute_terms[warp][h]); }
      }
      output[size_t(row) * width + index] = admitted ? score.value() : -__longlong_as_double(0x7ff0000000000000ll);
      magnitudes[size_t(row) * width + index] = absolute.value();
    }
    __syncwarp();
  }
}
} // namespace chop::position_f64_probe
