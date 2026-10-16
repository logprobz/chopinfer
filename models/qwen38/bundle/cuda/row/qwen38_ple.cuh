#pragma once
#include "qwen38.cuh"

namespace chop::row::qwen38 { static_assert(CHOP_QWEN38_NGRAM_SIZE == 3 && CHOP_QWEN38_NGRAM_HEADS == 2 * CHOP_QWEN38_NGRAM_HEADS_PER_ORDER, "two hash orders");
// Row t of a sequence chunk of count rows. History and state are immutable
// input; callers commit the separate output state after successful execution.
// A rank holds Local of the 16 hash heads (4 at TP4, 8 at TP2):
// grid y is its local head, global head rank * Local + y.
template<int Local = CHOP_QWEN38_NGRAM_DEFAULT_LOCAL_HEADS>
__device__ __forceinline__ void ngram_row(
    uint32_t t, uint32_t count, const uint32_t* tokens, const uint32_t* history,
    const uint64_t* multipliers, const uint64_t* sizes, const uint64_t* offsets,
    const uint8_t* table, const uint16_t* scale, uint16_t* output,
    uint64_t* row_ids, uint32_t* history_out, uint32_t rank,
    uint64_t table_start, uint64_t table_rows) {
  static_assert(Local >= 1 && CHOP_QWEN38_NGRAM_HEADS % Local == 0);
  const uint32_t head = rank * Local + blockIdx.y;
  if (rank >= CHOP_QWEN38_NGRAM_HEADS / Local) asm volatile("trap;");
  constexpr uint32_t Eos = CHOP_QWEN38_NGRAM_EOS;
  uint64_t mixed = uint64_t(tokens[t]) * multipliers[0];
  bool crossed = false;
  for (uint32_t shift = 1; shift < (head < CHOP_QWEN38_NGRAM_HEADS_PER_ORDER ? 2 : CHOP_QWEN38_NGRAM_SIZE); ++shift) {
    uint32_t id = t >= shift ? tokens[t - shift] : history[CHOP_QWEN38_NGRAM_SIZE - 1 + t - shift];
    if (crossed) id = Eos;
    crossed |= id == Eos;
    mixed ^= uint64_t(id) * multipliers[shift];
  }
  if (sizes[head] == 0) asm volatile("trap;");
  const uint64_t row = mixed % sizes[head] + offsets[head];
  if (row < table_start || row - table_start >= table_rows) asm volatile("trap;");
  if (threadIdx.x == 0) row_ids[uint64_t(t) * Local + blockIdx.y] = row;
  for (int c = threadIdx.x; c < CHOP_QWEN38_NGRAM_HEAD_WIDTH; c += blockDim.x) {
    const uint8_t code = table[(row - table_start) * CHOP_QWEN38_NGRAM_HEAD_WIDTH + c];
    const float v = convert::unpack_f32_e4m3(uint16_t(code)).lo;
    output[(uint64_t(t) * Local + blockIdx.y) * CHOP_QWEN38_NGRAM_HEAD_WIDTH + c] = to_bf(v * bf(*scale));
  }
  if (t == count - 1 && blockIdx.y == 0 && threadIdx.x == 0) {
    history_out[0] = count >= 2 ? tokens[count - 2] : history[1];
    history_out[1] = tokens[count - 1];
  }
}

// Each call covers one sequence chunk.
template<int Local = CHOP_QWEN38_NGRAM_DEFAULT_LOCAL_HEADS>
__device__ void ngram(const uint32_t* tokens, const uint32_t* history,
                     const uint64_t* multipliers, const uint64_t* sizes,
                     const uint64_t* offsets, const uint8_t* table,
                     const uint16_t* scale, uint16_t* output,
                     uint64_t* row_ids, uint32_t* history_out,
                     const uint32_t* live_rows, uint32_t rank,
                     uint64_t table_start, uint64_t table_rows) {
  begin();
  const uint32_t t = blockIdx.x, count = *live_rows;
  if (t >= count) return;
  ngram_row<Local>(t, count, tokens, history, multipliers, sizes, offsets, table, scale,
                   output, row_ids, history_out, rank, table_start, table_rows);
}

// Every decode item at once: item i (blockIdx.x) has q_len[i] rows, zero or
// one, at row i, and its staged history `entry` words after the previous
// item's, as one ngram call per item would read and write them.
template<int Local = CHOP_QWEN38_NGRAM_DEFAULT_LOCAL_HEADS>
__device__ void ngram_decode(const uint32_t* tokens, const uint32_t* history,
                            const uint64_t* multipliers, const uint64_t* sizes,
                            const uint64_t* offsets, const uint8_t* table,
                            const uint16_t* scale, uint16_t* output,
                            uint64_t* row_ids, uint32_t* history_out,
                            const uint32_t* q_len, uint32_t rank,
                            uint64_t table_start, uint64_t table_rows,
                            uint64_t entry) {
  begin();
  const uint32_t item = blockIdx.x, count = q_len[item];
  if (count == 0) return;
  if (count != 1) asm volatile("trap;");
  ngram_row<Local>(0, 1, tokens + item, history + item * entry, multipliers, sizes, offsets,
                   table, scale, output + uint64_t(item) * Local * CHOP_QWEN38_NGRAM_HEAD_WIDTH,
                   row_ids + uint64_t(item) * Local, history_out + item * entry, rank,
                   table_start, table_rows);
}

__device__ void ple_gate(const uint16_t* key, const uint16_t* query,
                        const uint16_t* value, uint16_t* output,
                        const uint32_t* live_rows) {
  begin();
  if (blockIdx.x >= *live_rows) return;
  __shared__ float sums[8];
  const uint64_t base = (uint64_t(blockIdx.x) * Streams + blockIdx.y) * Width;
  float dot = 0.f;
  for (int c = threadIdx.x; c < Width; c += blockDim.x)
    dot += bf(key[base + c]) * bf(query[base + c]);
  const float d = sum_cta(dot, sums) * rsqrtf(float(Width));
  const float gate = d == 0.f ? 0.5f : sigmoid(copysignf(sqrtf(fmaxf(fabsf(d), 1e-6f)), d));
  for (int c = threadIdx.x; c < Width; c += blockDim.x)
    output[base + c] = to_bf(gate * bf(value[uint64_t(blockIdx.x) * Width + c]));
}

__device__ __forceinline__ uint16_t conv_at(const uint16_t* input,
                                           const uint16_t* state,
                                           int64_t row, int channel) {
  return row < 0 ? state[(row + CHOP_QWEN38_PLE_CONV_ROWS) * Wide + channel] : input[row * Wide + channel];
}

__device__ __forceinline__ void ple_conv_row(
    uint32_t t, uint32_t count, int ch, const uint16_t* input,
    const uint16_t* weight, const uint16_t* state, uint16_t* output,
    uint16_t* state_out) {
  float sum = 0.f;
  for (int tap = 0; tap < CHOP_QWEN38_PLE_CONV_TAPS; ++tap)
    sum += bf(weight[ch * CHOP_QWEN38_PLE_CONV_TAPS + tap]) * bf(conv_at(input, state, int64_t(t) - (CHOP_QWEN38_PLE_CONV_TAPS - 1 - tap) * CHOP_QWEN38_PLE_CONV_DILATION, ch));
  output[uint64_t(t) * Wide + ch] = to_bf(sum * sigmoid(sum));
  if (t == 0)
    for (int r = 0; r < CHOP_QWEN38_PLE_CONV_ROWS; ++r)
      state_out[r * Wide + ch] = conv_at(input, state, int64_t(count) - CHOP_QWEN38_PLE_CONV_ROWS + r, ch);
}

__device__ void ple_conv(const uint16_t* input, const uint16_t* weight,
                        const uint16_t* state, uint16_t* output,
                        uint16_t* state_out, const uint32_t* live_rows) {
  begin();
  const uint32_t t = blockIdx.x, count = *live_rows;
  const int ch = blockIdx.y * 256 + threadIdx.x;
  if (t >= count || ch >= Wide) return;
  ple_conv_row(t, count, ch, input, weight, state, output, state_out);
}

// Every decode item at once, as ngram_decode: item i's row i and its staged
// state `entry` elements after the previous item's.
__device__ void ple_conv_decode(const uint16_t* input, const uint16_t* weight,
                               const uint16_t* state, uint16_t* output,
                               uint16_t* state_out, const uint32_t* q_len,
                               uint64_t entry) {
  begin();
  const uint32_t item = blockIdx.x, count = q_len[item];
  const int ch = blockIdx.y * 256 + threadIdx.x;
  if (count == 0 || ch >= Wide) return;
  if (count != 1) asm volatile("trap;");
  ple_conv_row(0, 1, ch, input + uint64_t(item) * Wide, weight, state + item * entry,
               output + uint64_t(item) * Wide, state_out + item * entry);
}
} // namespace chop::row::qwen38
