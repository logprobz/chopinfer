#pragma once
// Rank-ordered F32 partials are injected without intermediate rounding.
// Each stream keeps the 256-thread reduction and one final BF16 rounding.
#include "comm/comm.cuh"
#include "row/row.cuh"
#include "common/sigmoid.cuh"

namespace chop::comm::epilogue {

template <uint32_t Width, uint32_t Streams>
struct HcInjectNormF32 {
  static constexpr uint32_t kWidth = Width, kStreams = Streams;
  static constexpr uint32_t kThreads = 256;           // one CTA per (row, stream) group
  static constexpr uint32_t kPer = Width / kThreads;  // columns per thread
  static_assert(kPer * kThreads == Width, "whole columns per thread");
  static_assert(Streams == 4, "the inject scale divides the logit by four");

  // A group's per-row operand besides its residual values: the inject scale.
  struct Operand { float z, scale; };
  __device__ __forceinline__ static float inject(float x, float y, Operand s) {
    return s.z >= chop::sigmoid::kTail ? x + s.scale * y
        : x + 2.f * chop::sigmoid::tail_times(y, s.z);
  }

  const uint16_t* residual;
  const uint16_t* logits;
  const float* weight;
  uint16_t* residual_out;
  uint16_t* normed;

  // Group `group` is row * Streams + stream.
  __device__ __forceinline__ Operand scale(uint64_t group) const {
    const float z = row::bf(logits[group]) / Streams;
    return {z, 2.f * chop::sigmoid::sigmoid(z)};
  }
  __device__ __forceinline__ void operand(uint64_t group, uint16_t (&values)[kPer]) const {
#pragma unroll
    for (uint32_t j = 0; j < kPer; ++j) values[j] = residual[group * Width + threadIdx.x + j * kThreads];
  }
  // Legacy decode transports reserve BF16 parameter registers. F32 weights
  // are read by finish after the dependency wait, without narrowing.
  __device__ __forceinline__ void parameters(uint32_t, uint16_t (&)[kPer]) const {}

  // Injects the sum y of the group's token into the group's stream, whose
  // residual values and inject scale the caller loaded, and writes the
  // stream's norm. `sums` holds row::sum_cta's partials.
  __device__ __forceinline__ void finish(const float (&y)[kPer], const uint16_t (&residual)[kPer],
                                         Operand scale, uint64_t group,
                                         float* sums) const {
    float values[kPer];
    float square = 0.f;
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
      const uint32_t c = threadIdx.x + j * kThreads;
      const float v = inject(row::bf(residual[j]), y[j], scale);
      values[j] = v;
      square += v * v;
      residual_out[group * Width + c] = row::to_bf(v);
    }
    const float inv = rsqrtf(row::sum_cta(square, sums) / Width + 1e-6f);
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
      const uint32_t c = threadIdx.x + j * kThreads;
      normed[group * Width + c] = row::to_bf(values[j] * inv *
          (1.f + weight[(group % Streams) * Width + c]));
    }
  }

  __device__ __forceinline__ void finish(const float (&y)[kPer], const uint16_t (&residual)[kPer],
                                         Operand scale, uint64_t group,
                                         const uint16_t (&)[kPer], float* sums) const {
    finish(y, residual, scale, group, sums);
  }

  // Every stream of token `token`, one after another, by one CTA.
  __device__ __forceinline__ void finish_token(const float (&y)[kPer], uint64_t token,
                                               float* sums) const {
    for (uint32_t stream = 0; stream < Streams; ++stream) {
      const uint64_t group = token * Streams + stream;
      uint16_t values[kPer];
      operand(group, values);
      finish(y, values, scale(group), group, sums);
      __syncthreads();
    }
  }
};

} // namespace chop::comm::epilogue
