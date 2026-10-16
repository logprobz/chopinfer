#pragma once

// A hyper-connection epilogue for compile tests (tests/comm_extension.rs):
// every member the transports call, with the thread count Threads, as
// stub::Threads_<Threads>::Epilogue<Width, Streams>. Its arithmetic is a
// placeholder; nothing runs it.

#include <stdint.h>

namespace stub {
template <uint32_t Threads>
struct Threads_ {
  template <uint32_t Width, uint32_t Streams>
  struct Epilogue {
    static constexpr uint32_t kWidth = Width, kStreams = Streams, kThreads = Threads;
    static constexpr uint32_t kPer = Width / Threads;
    using Operand = float;
    const uint16_t* residual;
    const uint16_t* logits;
    const uint16_t* weight;
    uint16_t* residual_out;
    uint16_t* normed;
    __device__ Operand scale(uint64_t) const { return 0.f; }
    __device__ void operand(uint64_t, uint16_t (&)[kPer]) const {}
    // Marked with a PMTRIG in SASS, so tests can find where it runs.
    __device__ void parameters(uint32_t, uint16_t (&)[kPer]) const {
      asm volatile("pmevent 3;");
    }
    __device__ void finish(const float (&y)[kPer], const uint16_t (&)[kPer], Operand,
                           uint64_t group, float*) const {
      normed[group * Width + threadIdx.x] = uint16_t(y[0]);
    }
    __device__ void finish(const float (&y)[kPer], const uint16_t (&)[kPer], Operand,
                           uint64_t group, const uint16_t (&)[kPer], float*) const {
      normed[group * Width + threadIdx.x] = uint16_t(y[0]);
    }
    __device__ void finish_token(const float (&y)[kPer], uint64_t token, float*) const {
      normed[token * Width + threadIdx.x] = uint16_t(y[0]);
    }
  };
};
}
