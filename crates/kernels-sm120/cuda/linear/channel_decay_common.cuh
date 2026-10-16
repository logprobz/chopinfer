#pragma once

// Shared helpers of the per-channel-decay delta rule (a gated delta rule
// whose forget gate decays every key channel separately): the two gate
// forms, BF16 beta, the L2 norms and chunk addressing at absolute multiples
// of the chunk length. Contract: kernel_core::per_channel_decay_delta_rule.
// All arithmetic is F32 with IEEE FMA in a fixed order.

#include "common.cuh"

namespace chop::linear::channel_decay {

using chop::linear::bf;
using chop::linear::bf_hi;
using chop::linear::bf_lo;
using chop::linear::to_bf;
using chop::linear::warp_sum;

// Gate forms, the template parameter `Gate`.
constexpr int kBoundedSigmoid = 0;
constexpr int kSoftplus = 1;

// head_dim^-0.5 rounded once from the reference's double.
template<int Dim>
__device__ __forceinline__ float query_scale() {
  static_assert(Dim == 64 || Dim == 128);
  return Dim == 64 ? 0.125f : 0.08838834764831845f;
}

// The per-step log decay g of one channel from the gate projection output
// f, the channel's dt_bias and a = exp(A_log) of the head:
// bounded: lower_bound * sigmoid(a * (f + dt_bias));
// softplus: -a * softplus(f + dt_bias), with the reference's literal
// log(1 + exp(x)) below the threshold 20.
template<int Gate, uint32_t BoundBits>
__device__ __forceinline__ float log_decay(float f, float dt_bias, float a) {
  const float x = f + dt_bias;
  if constexpr (Gate == kBoundedSigmoid) {
    return __uint_as_float(BoundBits) * chop::sigmoid::sigmoid(a * x);
  } else {
    return -a * (x > 20.f ? x : logf(1.f + expf(x)));
  }
}

// sigmoid of the BF16 logit, rounded to BF16 as the reference computes it.
__device__ __forceinline__ float beta_of(uint16_t logit) {
  return bf(to_bf(chop::sigmoid::sigmoid(bf(logit))));
}

// sqrt(sum of squares + 1e-6) of a row whose lanes hold four elements each:
// lane-local FMA from zero, then the butterfly sum.
__device__ __forceinline__ float l2_norm4(const float (&x)[4]) {
  float squares = 0.f;
  #pragma unroll
  for (int j = 0; j < 4; ++j) squares = fmaf(x[j], x[j], squares);
  return sqrtf(warp_sum(squares) + 1e-6f);
}

// Prefill chunks start at absolute multiples of kChunk: a sequence whose
// first row sits at absolute position p (phase p % kChunk) has
// ceil((phase + rows) / kChunk) chunks, the first and last possibly short.
struct AlignedChunk {
  uint32_t sequence;
  int32_t first_row;  // absolute row of the chunk's first row in the buffers
  int32_t rows;       // live rows, 1..kChunk
  int32_t index;      // chunk index within its sequence
  int32_t count;      // chunks of the sequence
};

__device__ __forceinline__ int32_t aligned_chunks(int32_t phase, int32_t rows) {
  return rows > 0 ? (phase + rows + kChunk - 1) / kChunk : 0;
}

__device__ __forceinline__ int32_t phase_of(const int32_t* start_positions,
                                            uint32_t sequence) {
  return start_positions[sequence] & (kChunk - 1);
}

// Rows [begin, end) of chunk `index` of a sequence, relative to its start.
__device__ __forceinline__ void chunk_rows(int32_t phase, int32_t rows,
                                           int32_t index, int32_t& begin,
                                           int32_t& end) {
  begin = max(0, index * kChunk - phase);
  end = min(rows, (index + 1) * kChunk - phase);
}

// Global index of the first chunk of `sequence` over the live sequences.
__device__ __forceinline__ int32_t first_aligned_chunk(const int32_t* cu_seqlens,
                                                       const int32_t* start_positions,
                                                       uint32_t sequence) {
  int32_t base = 0;
  for (uint32_t s = 0; s < sequence; ++s)
    base += aligned_chunks(phase_of(start_positions, s), cu_seqlens[s + 1] - cu_seqlens[s]);
  return base;
}

// Finds global chunk `target`. Returns false past the last chunk.
__device__ __forceinline__ bool locate_aligned_chunk(const int32_t* cu_seqlens,
                                                     const int32_t* start_positions,
                                                     uint32_t sequences, int32_t target,
                                                     AlignedChunk& chunk) {
  int32_t base = 0;
  for (uint32_t s = 0; s < sequences; ++s) {
    const int32_t start = cu_seqlens[s];
    const int32_t rows = cu_seqlens[s + 1] - start;
    const int32_t phase = phase_of(start_positions, s);
    const int32_t count = aligned_chunks(phase, rows);
    if (target < base + count) {
      int32_t begin, end;
      chunk_rows(phase, rows, target - base, begin, end);
      chunk.sequence = s;
      chunk.index = target - base;
      chunk.first_row = start + begin;
      chunk.rows = end - begin;
      chunk.count = count;
      return true;
    }
    base += count;
  }
  return false;
}

// The local row of a sequence's prefix anchor: the last absolute multiple of
// kChunk strictly inside the call when the call ends off one, else -1.
__device__ __forceinline__ int32_t anchor_row(int32_t phase, int32_t rows) {
  const int32_t tail = (phase + rows) % kChunk;
  return tail != 0 && rows - tail > 0 ? rows - tail : -1;
}

} // namespace chop::linear::channel_decay
