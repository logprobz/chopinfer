#pragma once

// Shared helpers of the linear-attention engine (decision 0084): chunk
// addressing over variable-length sequences, BF16 access and the scalar
// activations of the gated delta rule. All arithmetic is F32 FMA; no
// operand is rounded below F32 before the final BF16 output store.

#include <stdint.h>
#include <math.h>
#include "../common/ptx.cuh"
#include "../common/convert.cuh"
#include "../common/sigmoid.cuh"

namespace chop::linear {

// Rows per prefill chunk (and per convolution prefill tile).
constexpr int kChunk = 64;

__device__ __forceinline__ void begin() {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
}

__device__ __forceinline__ float bf(uint16_t x) {
  return convert::bf16_to_f32(x);
}
__device__ __forceinline__ uint16_t to_bf(float x) {
  return convert::bf16_from_f32(x);
}
// Low and high BF16 halves of a packed pair, low address first.
__device__ __forceinline__ float bf_lo(uint32_t x) {
  return __uint_as_float(x << 16);
}
__device__ __forceinline__ float bf_hi(uint32_t x) {
  return __uint_as_float(x & 0xffff0000u);
}

using chop::sigmoid::sigmoid;
using chop::sigmoid::silu;
// PyTorch softplus with beta 1 and threshold 20.
__device__ __forceinline__ float softplus(float x) {
  return x > 20.f ? x : log1pf(expf(x));
}

// A chunk of at most kChunk rows that never crosses a sequence boundary.
struct Chunk {
  uint32_t sequence;
  int32_t first_row;  // absolute row of the chunk's first row
  int32_t rows;       // live rows, 1..kChunk
  int32_t index;      // chunk index within its sequence
  int32_t global;     // chunk index over all live sequences
};

__device__ __forceinline__ int32_t chunks_of(int32_t rows) {
  return (rows + kChunk - 1) / kChunk;
}

// Finds global chunk `target` over the live sequences of `cu_seqlens`.
// Returns false past the last chunk.
__device__ __forceinline__ bool locate_chunk(const int32_t* cu_seqlens,
                                             uint32_t sequences,
                                             int32_t target, Chunk& chunk) {
  int32_t base = 0;
  for (uint32_t s = 0; s < sequences; ++s) {
    const int32_t start = cu_seqlens[s];
    const int32_t rows = cu_seqlens[s + 1] - start;
    const int32_t count = chunks_of(rows);
    if (target < base + count) {
      chunk.sequence = s;
      chunk.index = target - base;
      chunk.first_row = start + chunk.index * kChunk;
      chunk.rows = min(kChunk, rows - chunk.index * kChunk);
      chunk.global = target;
      return true;
    }
    base += count;
  }
  return false;
}

// Global index of the first chunk of `sequence`.
__device__ __forceinline__ int32_t first_chunk(const int32_t* cu_seqlens,
                                               uint32_t sequence) {
  int32_t base = 0;
  for (uint32_t s = 0; s < sequence; ++s)
    base += chunks_of(cu_seqlens[s + 1] - cu_seqlens[s]);
  return base;
}

__device__ __forceinline__ float warp_sum(float x) {
  #pragma unroll
  for (int offset = 16; offset; offset >>= 1)
    x += __shfl_xor_sync(0xffffffffu, x, offset);
  return x;
}

} // namespace chop::linear
