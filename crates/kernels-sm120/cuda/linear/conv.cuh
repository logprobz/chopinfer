#pragma once

// Depthwise causal convolution without bias, BF16 in and out, with a BF16
// rolling state per sequence slot. Tap 0 multiplies the oldest input,
// History = (Kernel - 1) * Dilation rows back. A slot holds those History
// inputs as [History, Channels], oldest first. Each thread owns two adjacent
// channels, so every state element is read and written by one thread only.

#include "common.cuh"

namespace chop::linear {

template<int Channels, int Kernel>
struct ConvTaps {
  float w0[Kernel];
  float w1[Kernel];
  __device__ __forceinline__ void load(const uint16_t* weight, int c) {
    #pragma unroll
    for (int tap = 0; tap < Kernel; ++tap) {
      w0[tap] = bf(weight[c * Kernel + tap]);
      w1[tap] = bf(weight[(c + 1) * Kernel + tap]);
    }
  }
};

// Window of the previous History inputs of two channels, oldest first.
template<int History>
struct ConvWindow {
  float x0[History > 0 ? History : 1];
  float x1[History > 0 ? History : 1];
  __device__ __forceinline__ void push(float a, float b) {
    #pragma unroll
    for (int j = 0; j + 1 < History; ++j) {
      x0[j] = x0[j + 1];
      x1[j] = x1[j + 1];
    }
    x0[History - 1] = a;
    x1[History - 1] = b;
  }
};

template<int Kernel, int Dilation, bool Silu, int History>
__device__ __forceinline__ uint32_t conv_step(const float (&w0)[Kernel],
                                              const float (&w1)[Kernel],
                                              const ConvWindow<History>& window,
                                              float a, float b) {
  float y0 = w0[Kernel - 1] * a;
  float y1 = w1[Kernel - 1] * b;
  #pragma unroll
  for (int tap = 0; tap + 1 < Kernel; ++tap) {
    const int back = (Kernel - 1 - tap) * Dilation;
    y0 = fmaf(w0[tap], window.x0[History - back], y0);
    y1 = fmaf(w1[tap], window.x1[History - back], y1);
  }
  if constexpr (Silu) {
    y0 = silu(y0);
    y1 = silu(y1);
  }
  return uint32_t(to_bf(y0)) | (uint32_t(to_bf(y1)) << 16);
}

__device__ __forceinline__ uint32_t load_pair(const uint16_t* p) {
  return *reinterpret_cast<const uint32_t*>(p);
}
__device__ __forceinline__ void store_pair(uint16_t* p, uint32_t v) {
  *reinterpret_cast<uint32_t*>(p) = v;
}

// Grid: (Channels / (2 * blockDim.x), prefill chunk capacity). One CTA covers
// one chunk of one sequence. The CTA holding a sequence's first chunk is the
// only reader and writer of that sequence's slot: it reads the old state for
// its window and writes the new state, which is the sequence's last History
// inputs (older ones from the old state when the sequence is shorter).
// `anchor_ids[seq]` >= 0 names a pool entry that also receives the state as
// it stands after the sequence's last whole chunk, when a partial chunk
// follows it: the state a sequence ending there would write. The CTA of that
// whole chunk writes it; it holds History <= kChunk inputs of its own.
template<int Channels, int Kernel, int Dilation, bool Silu>
__device__ void conv_prefill(const uint16_t* input, uint32_t input_stride,
                             const uint16_t* weight, uint16_t* state,
                             uint16_t* output, const int32_t* cu_seqlens,
                             const int32_t* slot_ids, const int32_t* initial,
                             const int32_t* anchor_ids,
                             const uint32_t* live_sequences) {
  constexpr int History = (Kernel - 1) * Dilation;
  static_assert(Channels % 2 == 0 && History > 0);
  begin();
  Chunk chunk;
  if (!locate_chunk(cu_seqlens, *live_sequences, blockIdx.y, chunk)) return;
  const int c = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  if (c >= Channels) return;
  ConvTaps<Channels, Kernel> taps;
  taps.load(weight, c);
  const int32_t seq_start = cu_seqlens[chunk.sequence];
  const int32_t seq_rows = cu_seqlens[chunk.sequence + 1] - seq_start;
  const bool owner = chunk.index == 0;
  const bool carry = initial[chunk.sequence] != 0;
  uint16_t* slot = state + uint64_t(uint32_t(slot_ids[chunk.sequence])) *
                               History * Channels + c;
  ConvWindow<History> window;
  #pragma unroll
  for (int j = 0; j < History; ++j) {
    const int32_t row = chunk.first_row - History + j;
    uint32_t pair = 0;
    if (row >= seq_start)
      pair = load_pair(input + uint64_t(row) * input_stride + c);
    else if (carry)
      pair = load_pair(slot + (row - seq_start + History) * Channels);
    window.x0[j] = bf_lo(pair);
    window.x1[j] = bf_hi(pair);
  }
  // The owner's final state mixes old state rows only when the whole
  // sequence is shorter than History; capture those before any store.
  uint32_t tail[History];
  if (owner) {
    #pragma unroll
    for (int j = 0; j < History; ++j) {
      const int32_t row = seq_start + seq_rows - History + j;
      if (row >= seq_start)
        tail[j] = load_pair(input + uint64_t(row) * input_stride + c);
      else
        tail[j] = carry ? load_pair(slot + (row - seq_start + History) * Channels) : 0u;
    }
  }
  // Rows go in groups of eight so their loads are in flight together.
  constexpr int Group = 8;
  for (int r0 = 0; r0 < chunk.rows; r0 += Group) {
    uint32_t pairs[Group];
    #pragma unroll
    for (int g = 0; g < Group; ++g)
      if (r0 + g < chunk.rows)
        pairs[g] = load_pair(input + uint64_t(chunk.first_row + r0 + g) * input_stride + c);
    #pragma unroll
    for (int g = 0; g < Group; ++g) {
      if (r0 + g >= chunk.rows) break;
      const float a = bf_lo(pairs[g]), b = bf_hi(pairs[g]);
      store_pair(output + uint64_t(chunk.first_row + r0 + g) * Channels + c,
                 conv_step<Kernel, Dilation, Silu>(taps.w0, taps.w1, window, a, b));
      window.push(a, b);
    }
  }
  if (owner) {
    #pragma unroll
    for (int j = 0; j < History; ++j) store_pair(slot + j * Channels, tail[j]);
  }
  static_assert(History <= kChunk);
  const int32_t anchor = anchor_ids[chunk.sequence];
  if (anchor >= 0 && seq_rows % kChunk != 0 && chunk.index == seq_rows / kChunk - 1) {
    uint16_t* entry = state + uint64_t(uint32_t(anchor)) * History * Channels + c;
    #pragma unroll
    for (int j = 0; j < History; ++j)
      store_pair(entry + j * Channels,
                 load_pair(input + uint64_t(chunk.first_row + kChunk - History + j) *
                                       input_stride + c));
  }
}

// Grid: (Channels / (2 * blockDim.x), sequence capacity). One row per
// sequence; the slot state shifts in place.
template<int Channels, int Kernel, int Dilation, bool Silu>
__device__ void conv_decode(const uint16_t* input, uint32_t input_stride,
                            const uint16_t* weight, uint16_t* state,
                            uint16_t* output, const int32_t* slot_ids,
                            const uint32_t* live_sequences) {
  constexpr int History = (Kernel - 1) * Dilation;
  ptx::griddep_launch_dependents();
  const int c = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  ConvTaps<Channels, Kernel> taps;
  // Tap addresses depend only on the static channel, not on the producer.
  if (c < Channels) taps.load(weight, c);
  ptx::griddep_wait();
  const uint32_t seq = blockIdx.y;
  if (seq >= *live_sequences || c >= Channels) return;
  uint16_t* slot = state + uint64_t(uint32_t(slot_ids[seq])) * History * Channels + c;
  ConvWindow<History> window;
  uint32_t old[History];
  #pragma unroll
  for (int j = 0; j < History; ++j) {
    old[j] = load_pair(slot + j * Channels);
    window.x0[j] = bf_lo(old[j]);
    window.x1[j] = bf_hi(old[j]);
  }
  const uint32_t pair = load_pair(input + uint64_t(seq) * input_stride + c);
  store_pair(output + uint64_t(seq) * Channels + c,
             conv_step<Kernel, Dilation, Silu>(taps.w0, taps.w1, window,
                                               bf_lo(pair), bf_hi(pair)));
  #pragma unroll
  for (int j = 0; j + 1 < History; ++j) store_pair(slot + j * Channels, old[j + 1]);
  store_pair(slot + (History - 1) * Channels, pair);
}

// Grid: (Channels / (2 * blockDim.x), sequence capacity). Up to MaxRows rows
// per sequence from `cu_rows`. Reads the committed state from
// slot_ids[seq] and writes the state after row r to
// checkpoint_ids[seq * MaxRows + r]; the source slot is not modified unless a
// checkpoint aliases it.
template<int Channels, int Kernel, int Dilation, bool Silu, int MaxRows>
__device__ void conv_verify(const uint16_t* input, uint32_t input_stride,
                            const uint16_t* weight, uint16_t* state,
                            uint16_t* output, const int32_t* cu_rows,
                            const int32_t* slot_ids,
                            const int32_t* checkpoint_ids,
                            const uint32_t* live_sequences) {
  constexpr int History = (Kernel - 1) * Dilation;
  ptx::griddep_launch_dependents();
  const int c = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  ConvTaps<Channels, Kernel> taps;
  // Rank-image taps are immutable across every earlier graph launch.
  if (c < Channels) taps.load(weight, c);
  ptx::griddep_wait();
  const uint32_t seq = blockIdx.y;
  if (seq >= *live_sequences || c >= Channels) return;
  const int32_t first = cu_rows[seq];
  const int32_t rows = min(cu_rows[seq + 1] - first, MaxRows);
  const uint16_t* slot = state + uint64_t(uint32_t(slot_ids[seq])) * History * Channels + c;
  ConvWindow<History> window;
  uint32_t recent[History];
  #pragma unroll
  for (int j = 0; j < History; ++j) {
    recent[j] = load_pair(slot + j * Channels);
    window.x0[j] = bf_lo(recent[j]);
    window.x1[j] = bf_hi(recent[j]);
  }
  for (int r = 0; r < rows; ++r) {
    const uint64_t row = uint64_t(first + r);
    const uint32_t pair = load_pair(input + row * input_stride + c);
    const float a = bf_lo(pair), b = bf_hi(pair);
    store_pair(output + row * Channels + c,
               conv_step<Kernel, Dilation, Silu>(taps.w0, taps.w1, window, a, b));
    window.push(a, b);
    #pragma unroll
    for (int j = 0; j + 1 < History; ++j) recent[j] = recent[j + 1];
    recent[History - 1] = pair;
    uint16_t* checkpoint =
        state + uint64_t(uint32_t(checkpoint_ids[seq * MaxRows + r])) * History * Channels + c;
    #pragma unroll
    for (int j = 0; j < History; ++j) store_pair(checkpoint + j * Channels, recent[j]);
  }
}

} // namespace chop::linear
