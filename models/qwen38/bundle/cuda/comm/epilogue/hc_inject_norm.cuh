#pragma once

// Qwen3.8's fused all-reduce epilogue: the hyper-connection inject and the
// per-stream Gemma norm. The transports (comm/qwen38*.cuh) sum the ranks'
// Width-wide partials of a token in rank order in F32 and never round the
// sum; this functor injects it into each of the token's Streams residual
// streams, x + 2 sigmoid(logit / 4) y, writes the BF16 residual stream and
// normalizes it with (1 + weight). The F32 residual stays in registers
// through the norm: the Q-4 fused oracle has no BF16 rounding boundary
// between inject and norm. One CTA of kThreads threads finishes one (row,
// stream) group, thread t holding columns t + kThreads j; every transport
// uses these expressions in this order, so their outputs match bit for bit.

#include "comm/comm.cuh"
#include "row/qwen38.cuh"

namespace chop::comm::epilogue {

template <uint32_t Width, uint32_t Streams>
struct HcInjectNorm {
  static constexpr uint32_t kWidth = Width, kStreams = Streams;
  static constexpr uint32_t kThreads = 256;           // one CTA per (row, stream) group
  static constexpr uint32_t kPer = Width / kThreads;  // columns per thread
  static_assert(kPer * kThreads == Width, "whole columns per thread");
  static_assert(Streams == row::qwen38::Streams, "the inject scale divides the logit by four");

  // A group's per-row operand besides its residual values: the inject scale.
  using Operand = row::qwen38::InjectScale;

  const uint16_t* residual;
  const uint16_t* logits;
  const uint16_t* weight;
  uint16_t* residual_out;
  uint16_t* normed;

  // Group `group` is row * Streams + stream.
  __device__ __forceinline__ Operand scale(uint64_t group) const {
    return row::qwen38::inject_scale(row::bf(logits[group]));
  }
  __device__ __forceinline__ void operand(uint64_t group, uint16_t (&values)[kPer]) const {
#pragma unroll
    for (uint32_t j = 0; j < kPer; ++j) values[j] = residual[group * Width + threadIdx.x + j * kThreads];
  }
  // The stream's norm weights, model constants a transport may read before
  // the PDL wait.
  __device__ __forceinline__ void parameters(uint32_t stream, uint16_t (&values)[kPer]) const {
#pragma unroll
    for (uint32_t j = 0; j < kPer; ++j) values[j] = weight[stream * Width + threadIdx.x + j * kThreads];
  }

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
      const float v = row::qwen38::inject(row::bf(residual[j]), y[j], scale);
      values[j] = v;
      square += v * v;
      residual_out[group * Width + c] = row::to_bf(v);
    }
    const float inv = rsqrtf(row::sum_cta(square, sums) / Width + 1e-6f);
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
      const uint32_t c = threadIdx.x + j * kThreads;
      normed[group * Width + c] = row::to_bf(values[j] * inv *
          (1.f + row::bf(weight[(group % Streams) * Width + c])));
    }
  }

  // finish with the stream's norm weights already in registers
  // (parameters): the same expressions in the same order, so the same bits.
  __device__ __forceinline__ void finish(const float (&y)[kPer], const uint16_t (&residual)[kPer],
                                         Operand scale, uint64_t group,
                                         const uint16_t (&weight)[kPer], float* sums) const {
    float values[kPer];
    float square = 0.f;
#pragma unroll
    for (uint32_t j = 0; j < kPer; ++j) {
      const uint32_t c = threadIdx.x + j * kThreads;
      const float v = row::qwen38::inject(row::bf(residual[j]), y[j], scale);
      values[j] = v;
      square += v * v;
      residual_out[group * Width + c] = row::to_bf(v);
    }
    const float inv = rsqrtf(row::sum_cta(square, sums) / Width + 1e-6f);
#pragma unroll
    for (uint32_t j = 0; j < kPer; ++j) {
      const uint32_t c = threadIdx.x + j * kThreads;
      normed[group * Width + c] = row::to_bf(values[j] * inv * (1.f + row::bf(weight[j])));
    }
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
