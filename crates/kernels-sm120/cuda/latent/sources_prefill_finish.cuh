#pragma once
#include "sources_prefill.cuh"

namespace chop::latent::sources_prefill {

// Four metadata words make each fixed-r store visit all 32 shared banks.
// The key tile is dead here, so these 33,024 bytes reuse its existing arena.
template<class T>
__device__ __forceinline__ float* head_finish_partial(uint8_t* tile, int head) {
  static_assert(T::latent == 512 && T::splits == 1 && !T::partial);
  return reinterpret_cast<float*>(tile) + head * (T::latent + 4);
}

// Keep the one-split reference merge's sink, FMA, division, BF16 store
// boundary and adjacent-pair rotation. Do not replace weight * value + 0
// with value: signed zero, empty selections and exceptional inputs matter.
template<class T>
__device__ void head_finish(uint8_t* tile, uint32_t row, uint32_t head_base,
    const float* sinks, const float* cos_sin, uint16_t* out) {
  __syncthreads();
  if (threadIdx.x < 16) {
    const uint32_t head = head_base + threadIdx.x;
    float* part = head_finish_partial<T>(tile, threadIdx.x);
    float top = -INFINITY;
    top = fmaxf(top, part[0]);
    float sink = -INFINITY;
    if constexpr (T::sink) { sink = sinks[head]; top = fmaxf(top, sink); }
    const float weight = part[0] == -INFINITY ? 0.0f : exp_of(part[0] - top);
    float den = 0.0f;
    den = fmaf(weight, part[1], den);
    if constexpr (T::sink) den += exp_of(sink - top) * 1.0f;
    part[2] = weight;
    part[3] = den;
  }
  __syncthreads();
  for (int index = threadIdx.x; index < 16 * (T::latent / 2); index += blockDim.x) {
    const int local_head = index / (T::latent / 2);
    const int d = (index % (T::latent / 2)) * 2;
    const uint32_t head = head_base + local_head;
    const float* part = head_finish_partial<T>(tile, local_head);
    const float2 numerator = *reinterpret_cast<const float2*>(part + 4 + d);
    float x = 0.0f, y = 0.0f;
    x = fmaf(part[2], numerator.x, x);
    y = fmaf(part[2], numerator.y, y);
    const float den = part[3];
    const uint32_t rounded = attention::bf16_pair(den == 0.0f ? 0.0f : __fdiv_rn(x, den),
                                                 den == 0.0f ? 0.0f : __fdiv_rn(y, den));
    uint32_t value = rounded;
    if constexpr (T::tail != 0) {
      if (d >= T::latent - T::tail) {
        const float c = cos_sin[uint64_t(row) * T::tail + d - (T::latent - T::tail)];
        const float s = cos_sin[uint64_t(row) * T::tail + d - (T::latent - T::tail) + 1];
        const float a = attention::bf16_low(rounded), b = attention::bf16_high(rounded);
        value = attention::bf16_pair(fmaf(a, c, b * s), fmaf(b, c, -a * s));
      }
    }
    const uint64_t pair = uint64_t(row) * T::heads + head;
    *reinterpret_cast<uint32_t*>(out + pair * T::latent + d) = value;
  }
}

} // namespace chop::latent::sources_prefill
