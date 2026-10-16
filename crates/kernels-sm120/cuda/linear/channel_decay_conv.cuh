#pragma once

// The q, k and v short convolutions of the per-channel-decay delta rule:
// one depthwise causal convolution without bias over Channels = 3 * heads *
// head_dim BF16 inputs (q, then k, then v), then SiLU, rounded once to BF16.
// Each output is silu(sum_t w[t] * x[t]) with F32 weights, FMA from zero in
// tap order oldest first (tap 0 multiplies the input Kernel - 1 rows back).
// A slot holds the previous Kernel - 1 BF16 inputs [Kernel - 1, Channels],
// oldest first. Each thread owns two adjacent channels, so every state
// element is read and written by one thread only. Weights are one F32
// [Channels, Kernel] array (Separate false) or three F32 [Channels / 3,
// Kernel] arrays for q, k and v (Separate true).

#include "channel_decay_common.cuh"

namespace chop::linear::channel_decay {

template<int Channels, int Kernel, bool Separate>
struct ShortTaps {
  float w0[Kernel];
  float w1[Kernel];
  __device__ __forceinline__ void load(const float* weight_q, const float* weight_k,
                                       const float* weight_v, int c) {
    const float* weight = weight_q;
    int local = c;
    if constexpr (Separate) {
      constexpr int Part = Channels / 3;
      weight = c < Part ? weight_q : (c < 2 * Part ? weight_k : weight_v);
      local = c % Part;
    }
    #pragma unroll
    for (int tap = 0; tap < Kernel; ++tap) {
      w0[tap] = weight[local * Kernel + tap];
      w1[tap] = weight[(local + 1) * Kernel + tap];
    }
  }
};

// x[0..Kernel-2] hold the previous inputs, oldest first.
template<int Kernel>
struct ShortWindow {
  float x0[Kernel - 1];
  float x1[Kernel - 1];
  __device__ __forceinline__ void push(float a, float b) {
    #pragma unroll
    for (int j = 0; j + 2 < Kernel; ++j) {
      x0[j] = x0[j + 1];
      x1[j] = x1[j + 1];
    }
    x0[Kernel - 2] = a;
    x1[Kernel - 2] = b;
  }
};

template<int Channels, int Kernel, bool Separate>
__device__ __forceinline__ uint32_t short_step(const ShortTaps<Channels, Kernel, Separate>& taps,
                                               const ShortWindow<Kernel>& window, float a,
                                               float b) {
  float y0 = 0.f, y1 = 0.f;
  #pragma unroll
  for (int tap = 0; tap + 1 < Kernel; ++tap) {
    y0 = fmaf(taps.w0[tap], window.x0[tap], y0);
    y1 = fmaf(taps.w1[tap], window.x1[tap], y1);
  }
  y0 = fmaf(taps.w0[Kernel - 1], a, y0);
  y1 = fmaf(taps.w1[Kernel - 1], b, y1);
  return uint32_t(to_bf(chop::sigmoid::silu(y0))) |
         (uint32_t(to_bf(chop::sigmoid::silu(y1))) << 16);
}

__device__ __forceinline__ uint32_t load_pair(const uint16_t* p) {
  return *reinterpret_cast<const uint32_t*>(p);
}
__device__ __forceinline__ void store_pair(uint16_t* p, uint32_t v) {
  *reinterpret_cast<uint32_t*>(p) = v;
}

// Grid: (Channels / (2 * blockDim.x), prefill chunk capacity). One CTA
// covers one chunk of 64 rows from its sequence's start (the convolution
// keeps every row's bits under any split). The CTA of a sequence's first
// chunk is the only reader and writer of its slot. When `anchor_ids[seq]`
// >= 0 and the call ends off an absolute multiple of 64 with one inside it
// (`start_positions` gives each sequence's absolute first position), that
// CTA also writes the state as it stands at that multiple to the anchor
// entry: the state a call ending there would write.
template<int Channels, int Kernel, bool Separate>
__device__ void short_conv_prefill(const uint16_t* input, uint32_t input_stride,
                                   const float* weight_q, const float* weight_k,
                                   const float* weight_v, uint16_t* state,
                                   uint16_t* output, const int32_t* cu_seqlens,
                                   const int32_t* start_positions,
                                   const int32_t* slot_ids, const int32_t* initial,
                                   const int32_t* anchor_ids,
                                   const uint32_t* live_sequences) {
  constexpr int History = Kernel - 1;
  static_assert(Channels % 6 == 0 && Kernel >= 2 && Kernel <= 8 && History <= kChunk);
  chop::linear::begin();
  chop::linear::Chunk chunk;
  if (!chop::linear::locate_chunk(cu_seqlens, *live_sequences, blockIdx.y, chunk)) return;
  const int c = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  if (c >= Channels) return;
  ShortTaps<Channels, Kernel, Separate> taps;
  taps.load(weight_q, weight_k, weight_v, c);
  const int32_t seq_start = cu_seqlens[chunk.sequence];
  const int32_t seq_rows = cu_seqlens[chunk.sequence + 1] - seq_start;
  const bool owner = chunk.index == 0;
  const bool carry = initial[chunk.sequence] != 0;
  uint16_t* slot = state + uint64_t(uint32_t(slot_ids[chunk.sequence])) * History * Channels + c;
  // Input `row` (absolute) of this sequence, or the old state before it.
  auto history = [&](int32_t row) -> uint32_t {
    if (row >= seq_start) return load_pair(input + uint64_t(row) * input_stride + c);
    return carry ? load_pair(slot + (row - seq_start + History) * Channels) : 0u;
  };
  ShortWindow<Kernel> window;
  #pragma unroll
  for (int j = 0; j < History; ++j) {
    const uint32_t pair = history(chunk.first_row - History + j);
    window.x0[j] = bf_lo(pair);
    window.x1[j] = bf_hi(pair);
  }
  // The owner's states may mix old state rows; capture them before any store.
  uint32_t tail[History];
  uint32_t anchored[History];
  const int32_t anchor = anchor_ids[chunk.sequence];
  const int32_t anchor_at =
      anchor_row(phase_of(start_positions, chunk.sequence), seq_rows);
  if (owner) {
    #pragma unroll
    for (int j = 0; j < History; ++j) {
      tail[j] = history(seq_start + seq_rows - History + j);
      anchored[j] = anchor >= 0 && anchor_at > 0 ? history(seq_start + anchor_at - History + j)
                                                 : 0u;
    }
  }
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
                 short_step(taps, window, a, b));
      window.push(a, b);
    }
  }
  if (owner) {
    #pragma unroll
    for (int j = 0; j < History; ++j) store_pair(slot + j * Channels, tail[j]);
    if (anchor >= 0 && anchor_at > 0) {
      uint16_t* entry = state + uint64_t(uint32_t(anchor)) * History * Channels + c;
      #pragma unroll
      for (int j = 0; j < History; ++j) store_pair(entry + j * Channels, anchored[j]);
    }
  }
}

// Grid: (Channels / (2 * blockDim.x), sequence capacity). One row per
// sequence; the slot state shifts in place.
template<int Channels, int Kernel, bool Separate>
__device__ void short_conv_decode(const uint16_t* input, uint32_t input_stride,
                                  const float* weight_q, const float* weight_k,
                                  const float* weight_v, uint16_t* state,
                                  uint16_t* output, const int32_t* slot_ids,
                                  const uint32_t* live_sequences) {
  constexpr int History = Kernel - 1;
  ptx::griddep_launch_dependents();
  const int c = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  ShortTaps<Channels, Kernel, Separate> taps;
  // Weights are immutable across graph launches.
  if (c < Channels) taps.load(weight_q, weight_k, weight_v, c);
  ptx::griddep_wait();
  const uint32_t seq = blockIdx.y;
  if (seq >= *live_sequences || c >= Channels) return;
  uint16_t* slot = state + uint64_t(uint32_t(slot_ids[seq])) * History * Channels + c;
  ShortWindow<Kernel> window;
  uint32_t old[History];
  #pragma unroll
  for (int j = 0; j < History; ++j) {
    old[j] = load_pair(slot + j * Channels);
    window.x0[j] = bf_lo(old[j]);
    window.x1[j] = bf_hi(old[j]);
  }
  const uint32_t pair = load_pair(input + uint64_t(seq) * input_stride + c);
  store_pair(output + uint64_t(seq) * Channels + c,
             short_step(taps, window, bf_lo(pair), bf_hi(pair)));
  #pragma unroll
  for (int j = 0; j + 1 < History; ++j) store_pair(slot + j * Channels, old[j + 1]);
  store_pair(slot + (History - 1) * Channels, pair);
}

// Grid: (Channels / (2 * blockDim.x), sequence capacity). Up to MaxRows rows
// per sequence from `cu_rows`. Reads the committed state from slot_ids[seq]
// and writes the state after row r to checkpoint_ids[seq * MaxRows + r]; the
// source slot is not modified unless a checkpoint aliases it.
template<int Channels, int Kernel, bool Separate, int MaxRows>
__device__ void short_conv_verify(const uint16_t* input, uint32_t input_stride,
                                  const float* weight_q, const float* weight_k,
                                  const float* weight_v, uint16_t* state,
                                  uint16_t* output, const int32_t* cu_rows,
                                  const int32_t* slot_ids, const int32_t* checkpoint_ids,
                                  const uint32_t* live_sequences) {
  constexpr int History = Kernel - 1;
  ptx::griddep_launch_dependents();
  const int c = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  ShortTaps<Channels, Kernel, Separate> taps;
  if (c < Channels) taps.load(weight_q, weight_k, weight_v, c);
  ptx::griddep_wait();
  const uint32_t seq = blockIdx.y;
  if (seq >= *live_sequences || c >= Channels) return;
  const int32_t first = cu_rows[seq];
  const int32_t rows = min(cu_rows[seq + 1] - first, MaxRows);
  const uint16_t* slot = state + uint64_t(uint32_t(slot_ids[seq])) * History * Channels + c;
  ShortWindow<Kernel> window;
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
    store_pair(output + row * Channels + c, short_step(taps, window, a, b));
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

} // namespace chop::linear::channel_decay
