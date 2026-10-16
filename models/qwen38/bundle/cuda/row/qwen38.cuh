#pragma once
#include "row/row.cuh"
#include "common/sigmoid.cuh"

namespace chop::row::qwen38 {
constexpr int Width = CHOP_QWEN38_HIDDEN;
constexpr int Streams = CHOP_QWEN38_STREAMS;
constexpr int Wide = Width * Streams;

using chop::sigmoid::sigmoid;

// A stream's inject scale 2 sigmoid(z), z = logit / 4, with z kept for the
// sigmoid's tail.
struct InjectScale {
  float z, scale;
};
__device__ __forceinline__ InjectScale inject_scale(float logit) {
  const float z = logit / Streams;
  return {z, 2.f * sigmoid(z)};
}
// x + 2 sigmoid(z) y. In the sigmoid's tail the product is formed in F64
// (common/sigmoid.cuh).
__device__ __forceinline__ float inject(float x, float y, InjectScale s) {
  return s.z >= chop::sigmoid::kTail ? x + s.scale * y
                                     : x + 2.f * chop::sigmoid::tail_times(y, s.z);
}

__device__ void hc_norm(const uint16_t* x, const uint16_t* weight,
                       uint16_t* output, const uint32_t* live_rows) {
  begin();
  if (blockIdx.x >= *live_rows) return;
  __shared__ float sums[8];
  const uint64_t base = (uint64_t(blockIdx.x) * Streams + blockIdx.y) * Width;
  float square = 0.f;
  for (int c = threadIdx.x; c < Width; c += blockDim.x) {
    float v = bf(x[base + c]);
    square += v * v;
  }
  const float inv = rsqrtf(sum_cta(square, sums) / Width + 1e-6f);
  for (int c = threadIdx.x; c < Width; c += blockDim.x)
    output[base + c] = to_bf(bf(x[base + c]) * inv *
                            (1.f + bf(weight[blockIdx.y * Width + c])));
}

__device__ void hc_bottleneck(const uint16_t* x, uint16_t* output,
                             const uint32_t* live_rows) {
  begin();
  if (blockIdx.x >= *live_rows) return;
  for (int c = threadIdx.x; c < CHOP_QWEN38_HC_RANK; c += blockDim.x) {
    const uint64_t i = uint64_t(blockIdx.x) * CHOP_QWEN38_HC_RANK + c;
    const float v = bf(x[i]) / Streams;
    output[i] = to_bf(chop::sigmoid::times(v, v));
  }
}

__device__ void hc_mix(const uint16_t* up, const uint16_t* normed,
                      uint16_t* output, const uint32_t* live_rows) {
  begin();
  if (blockIdx.x >= *live_rows) return;
  // Column groups of blockDim.x across blockIdx.y keep every load of a
  // decode row in flight at once.
  for (int c = blockIdx.y * blockDim.x + threadIdx.x; c < Width; c += blockDim.x * gridDim.y) {
    float sum = 0.f;
    for (int s = 0; s < Streams; ++s) {
      const uint64_t i = uint64_t(blockIdx.x) * Wide + s * Width + c;
      sum += chop::sigmoid::times(bf(normed[i]), bf(up[i])) / Streams;
    }
    output[uint64_t(blockIdx.x) * Width + c] = to_bf(sum);
  }
}

__device__ void hc_inject(const uint16_t* x, const uint16_t* y,
                         const uint16_t* logits, uint16_t* output,
                         const uint32_t* live_rows) {
  begin();
  if (blockIdx.x >= *live_rows) return;
  const uint64_t row = uint64_t(blockIdx.x) * Streams + blockIdx.y;
  const InjectScale scale = inject_scale(bf(logits[row]));
  for (int c = threadIdx.x; c < Width; c += blockDim.x)
    output[row * Width + c] = to_bf(inject(bf(x[row * Width + c]),
        bf(y[uint64_t(blockIdx.x) * Width + c]), scale));
}
} // namespace chop::row::qwen38
