#pragma once
#include <stdint.h>
#include "common/ptx.cuh"

namespace chop::qwen38_hash { static_assert(CHOP_QWEN38_NGRAM_SIZE == 3 && CHOP_QWEN38_NGRAM_HEADS == 2 * CHOP_QWEN38_NGRAM_HEADS_PER_ORDER, "two hash orders");
// The family's hash and two-token history contract, independent of table
// encoding. A separate prepared shared lookup decodes only the selected rows.
__device__ __forceinline__ void row(
    uint32_t t, uint32_t count, const uint32_t* tokens, const uint32_t* history,
    const uint64_t* multipliers, const uint64_t* sizes, const uint64_t* offsets,
    uint64_t* ids, uint32_t* history_out, uint32_t rank,
    uint64_t table_start, uint64_t table_rows) {
  if (threadIdx.x != 0 || t >= count) return;
  if (rank >= CHOP_QWEN38_RANKS || blockIdx.y >= CHOP_QWEN38_NGRAM_LOCAL_HEADS) asm volatile("trap;");
  const uint32_t head = rank * CHOP_QWEN38_NGRAM_LOCAL_HEADS + blockIdx.y;
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
  const uint64_t selected = mixed % sizes[head] + offsets[head];
  if (selected < table_start || selected - table_start >= table_rows) asm volatile("trap;");
  ids[uint64_t(t) * CHOP_QWEN38_NGRAM_LOCAL_HEADS + blockIdx.y] = selected;
  if (t == count - 1 && blockIdx.y == 0) {
    history_out[0] = count >= 2 ? tokens[count - 2] : history[1];
    history_out[1] = tokens[count - 1];
  }
}

__device__ __forceinline__ void begin() {
  chop::ptx::griddep_launch_dependents();
  chop::ptx::griddep_wait();
}

__device__ void prefill(
    const uint32_t* tokens, const uint32_t* history, const uint64_t* multipliers,
    const uint64_t* sizes, const uint64_t* offsets, uint64_t* ids,
    uint32_t* history_out, const uint32_t* q_len, uint32_t rank,
    uint64_t table_start, uint64_t table_rows) {
  begin();
  row(blockIdx.x, *q_len, tokens, history, multipliers, sizes, offsets,
      ids, history_out, rank, table_start, table_rows);
}

__device__ void decode(
    const uint32_t* tokens, const uint32_t* history, const uint64_t* multipliers,
    const uint64_t* sizes, const uint64_t* offsets, uint64_t* ids,
    uint32_t* history_out, const uint32_t* q_len, uint32_t rank,
    uint64_t table_start, uint64_t table_rows, uint64_t entry_words) {
  begin();
  const uint32_t item = blockIdx.x, count = q_len[item];
  if (count == 0) {
    // The composed lookup can cover inactive slots inside a graph bucket.
    // Give those unused rows a valid address without changing their history.
    if (threadIdx.x == 0) ids[uint64_t(item) * CHOP_QWEN38_NGRAM_LOCAL_HEADS + blockIdx.y] = table_start;
    return;
  }
  if (count != 1) asm volatile("trap;");
  row(0, 1, tokens + item, history + item * entry_words, multipliers, sizes, offsets,
      ids + uint64_t(item) * CHOP_QWEN38_NGRAM_LOCAL_HEADS, history_out + item * entry_words, rank, table_start, table_rows);
}

__device__ void verify(
    const uint32_t* tokens, const uint32_t* history, const uint64_t* multipliers,
    const uint64_t* sizes, const uint64_t* offsets, uint64_t* ids,
    uint32_t* history_out, const int32_t* cu_rows, const uint32_t* q_len,
    const uint32_t* live_items, uint32_t rank, uint64_t table_start,
    uint64_t table_rows, uint64_t entry_words) {
  begin();
  const uint32_t item = blockIdx.z;
  if (item >= *live_items) return;
  const uint64_t start = uint64_t(cu_rows[item]);
  row(blockIdx.x, q_len[item], tokens + start, history + item * entry_words,
      multipliers, sizes, offsets, ids + start * CHOP_QWEN38_NGRAM_LOCAL_HEADS, history_out + item * entry_words,
      rank, table_start, table_rows);
}
} // namespace chop::qwen38_hash
