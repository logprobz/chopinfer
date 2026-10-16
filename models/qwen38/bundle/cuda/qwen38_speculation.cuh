#pragma once

// Qwen3.8 bundle rows for speculation: verify items of several rows through
// the per-sequence n-gram and PLE convolution kernels, the PLE state after
// each verify row, and the MTP drafter's input, residual and draft argmax.
//
// Grids depend only on prepared capacities; live counts and row offsets come
// from device arrays, so a verify or drafter graph replays for any items in
// its bucket.

#include <stdint.h>

#include "qwen38_bundle.cuh"
#include "row/qwen38_ple.cuh"

namespace chop::qwen38_bundle {

// Rows one verify item may carry.
constexpr uint32_t kVerifyRows = CHOP_QWEN38_MAX_VERIFY_ROWS;
constexpr uint32_t kWide = kHidden * kStreams;
// Bytes of one four-stream residual row.
constexpr uint64_t kWideBytes = uint64_t(kWide) * 2;
// The rank count, one rank's n-gram hash heads and vocabulary shard, which
// the build defines.
#if !defined(CHOP_QWEN38_RANKS) || !defined(CHOP_QWEN38_NGRAM_LOCAL_HEADS) || \
    !defined(CHOP_QWEN38_SHARD_VOCAB)
#error "CHOP_QWEN38_RANKS, CHOP_QWEN38_NGRAM_LOCAL_HEADS and CHOP_QWEN38_SHARD_VOCAB are required"
#endif
constexpr int kRanks = CHOP_QWEN38_RANKS;
constexpr int kNgramLocal = CHOP_QWEN38_NGRAM_LOCAL_HEADS;
// Vocabulary entries of one rank's logits shard.
constexpr uint32_t kShardVocab = CHOP_QWEN38_SHARD_VOCAB;

__device__ __forceinline__ float block_sum(float value, float* scratch) {
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  __syncthreads();
  if (lane == 0) scratch[warp] = value;
  __syncthreads();
  float total = 0.f;
  for (int w = 0; w < int(blockDim.x / 32); ++w) total += scratch[w];
  return total;
}

// The n-gram rows of every verify item: item z's rows start at cu_rows[z]
// and it has q_len[z] of them; its staged history sits `entry_words` words
// after the previous item's. Grid [kVerifyRows, kNgramLocal heads, items].
__device__ __forceinline__ void ngram_verify(
    const uint32_t* tokens, const uint32_t* history, const uint64_t* multipliers,
    const uint64_t* sizes, const uint64_t* offsets, const uint8_t* table,
    const uint16_t* scale, uint16_t* output, uint64_t* row_ids, uint32_t* history_out,
    const int32_t* cu_rows, const int32_t* q_len, const uint32_t* live_items, uint32_t rank,
    uint64_t table_start, uint64_t table_rows, uint64_t entry_words) {
  chop::ptx::griddep_launch_dependents();
  chop::ptx::griddep_wait();
  const uint32_t item = blockIdx.z;
  if (item >= *live_items) return;
  const uint64_t start = uint64_t(cu_rows[item]);
  chop::row::qwen38::ngram<kNgramLocal>(
      tokens + start, history + item * entry_words, multipliers, sizes, offsets, table, scale,
      output + start * kNgramLocal * CHOP_QWEN38_NGRAM_HEAD_WIDTH, row_ids + start * kNgramLocal,
      history_out + item * entry_words, reinterpret_cast<const uint32_t*>(q_len + item), rank,
      table_start, table_rows);
}

// The PLE convolution of every verify item, as ngram_verify. `entry` is the
// staged state stride in elements. Grid [kVerifyRows, ceil(kWide / 256) channel blocks, items].
__device__ __forceinline__ void ple_conv_verify(const uint16_t* input, const uint16_t* weight,
                                                const uint16_t* state, uint16_t* output,
                                                uint16_t* state_out, const int32_t* cu_rows,
                                                const int32_t* q_len,
                                                const uint32_t* live_items, uint64_t entry) {
  chop::ptx::griddep_launch_dependents();
  chop::ptx::griddep_wait();
  const uint32_t item = blockIdx.z;
  if (item >= *live_items) return;
  const uint64_t start = uint64_t(cu_rows[item]);
  chop::row::qwen38::ple_conv(input + start * kWide, weight, state + item * entry,
                              output + start * kWide, state_out + item * entry,
                              reinterpret_cast<const uint32_t*>(q_len + item));
}

// Writes to `to` the PLE state after row `row` of an item whose rows start
// at phase row `start`: what the n-gram and convolution kernels would write
// as their output state for a chunk ending at that row. `input` is the
// convolution input, `before` the item's state before the step. One CTA
// moves bytes [first, end) of the entry.
__device__ __forceinline__ void ple_state_after(const uint16_t* input, const uint32_t* tokens,
                                                const uint8_t* before, uint8_t* to,
                                                uint64_t start, uint32_t row, uint64_t first,
                                                uint64_t end) {
  for (uint64_t at = first + 16ull * threadIdx.x; at < end; at += 16ull * blockDim.x) {
    uint4 value;
    if (at < kPleConvBytes) {
      // History row r after `row + 1` inputs is input row row + 1 - history rows + r,
      // or row row + 1 + r of the staged history when that is negative.
      const int64_t r = int64_t(at / kWideBytes);
      const uint64_t column = at % kWideBytes;
      const int64_t source = int64_t(row) + 1 - CHOP_QWEN38_PLE_CONV_ROWS + r;
      const uint8_t* from =
          source < 0 ? before + uint64_t(source + CHOP_QWEN38_PLE_CONV_ROWS) * kWideBytes
                     : reinterpret_cast<const uint8_t*>(input + (start + uint64_t(source)) * kWide);
      value = *reinterpret_cast<const uint4*>(from + column);
    } else {
      value = *reinterpret_cast<const uint4*>(before + at);
      if (at == kPleConvBytes) {
        const uint32_t* history = reinterpret_cast<const uint32_t*>(before + kPleConvBytes);
        value.x = row >= 1 ? tokens[start + row - 1] : history[1];
        value.y = tokens[start + row];
      }
    }
    *reinterpret_cast<uint4*>(to + at) = value;
  }
}

// The PLE state after each verify row: entry item * kVerifyRows + row holds
// what the n-gram and convolution kernels would write as their output state
// for a chunk ending at that row. `input` is the convolution input the
// verify rows read, `staged` each item's state before the step. Grid
// [kVerifyRows, entry chunks, items].
__device__ __forceinline__ void ple_checkpoints(const uint16_t* input, const uint32_t* tokens,
                                                const uint8_t* staged, uint8_t* checkpoints,
                                                const int32_t* cu_rows, const int32_t* q_len,
                                                const uint32_t* live_items) {
  chop::ptx::griddep_launch_dependents();
  chop::ptx::griddep_wait();
  const uint32_t item = blockIdx.z, row = blockIdx.x;
  if (item >= *live_items || int32_t(row) >= q_len[item]) return;
  const uint64_t first = blockIdx.y * kChunk;
  if (first >= kPleEntryBytes) return;
  const uint64_t end = first + kChunk < kPleEntryBytes ? first + kChunk : kPleEntryBytes;
  ple_state_after(input, tokens, staged + uint64_t(item) * kPleEntryBytes,
                  checkpoints + (uint64_t(item) * kVerifyRows + row) * kPleEntryBytes,
                  uint64_t(cu_rows[item]), row, first, end);
}

// Indexed checkpoints share the committed pool. The host exchanges ownership
// after acceptance and returns the former committed entry to this workspace.
__device__ __forceinline__ void ple_checkpoints_indexed(
    const uint16_t* input, const uint32_t* tokens, const uint8_t* staged,
    uint8_t* pool, const int32_t* cu_rows, const int32_t* q_len,
    const uint32_t* live_items, const int32_t* checkpoint_ids, uint64_t pool_entries) {
  chop::ptx::griddep_launch_dependents();
  chop::ptx::griddep_wait();
  const uint32_t item = blockIdx.z, row = blockIdx.x;
  if (item >= *live_items || int32_t(row) >= q_len[item]) return;
  const uint64_t first = blockIdx.y * kChunk;
  if (first >= kPleEntryBytes) return;
  const uint64_t end = first + kChunk < kPleEntryBytes ? first + kChunk : kPleEntryBytes;
  const int32_t entry = checkpoint_ids[item * kVerifyRows + row];
  if (entry < 0 || uint64_t(entry) >= pool_entries) return;
  ple_state_after(input, tokens, staged + uint64_t(item) * kPleEntryBytes,
                  pool + uint64_t(entry) * kPleEntryBytes, uint64_t(cu_rows[item]), row, first, end);
}

// The PLE state a prefill item keeps at its anchor: entry anchor_slots[item]
// of `anchors` holds the state after the item's row before position
// anchor_ends[item], as ple_checkpoints for that row; items with a negative
// anchor slot write nothing. `staged` is each item's state before the step.
// Grid [entry chunks, items].
__device__ __forceinline__ void ple_anchor(const uint16_t* input, const uint32_t* tokens,
                                           const uint8_t* staged, uint8_t* anchors,
                                           const int32_t* cu_rows, const int32_t* prefix_len,
                                           const int32_t* anchor_slots,
                                           const int32_t* anchor_ends,
                                           const uint32_t* live_items) {
  chop::ptx::griddep_launch_dependents();
  chop::ptx::griddep_wait();
  const uint32_t item = blockIdx.y;
  if (item >= *live_items) return;
  const int32_t slot = anchor_slots[item];
  const int32_t rows = anchor_ends[item] - prefix_len[item];
  if (slot < 0 || rows < 1) return;
  const uint64_t first = blockIdx.x * kChunk;
  if (first >= kPleEntryBytes) return;
  const uint64_t end = first + kChunk < kPleEntryBytes ? first + kChunk : kPleEntryBytes;
  ple_state_after(input, tokens, staged + uint64_t(item) * kPleEntryBytes,
                  anchors + uint64_t(uint32_t(slot)) * kPleEntryBytes, uint64_t(cu_rows[item]),
                  uint32_t(rows - 1), first, end);
}

// The drafter's input rows: row i copies the target's four-stream X from
// `last` row sources[i] when that is at least zero, otherwise from retained
// entry -1 - sources[i]. When token_sources[i] is at least zero, the row's
// token is drafts[token_sources[i]]. Grid [rows].
__device__ __forceinline__ void mtp_gather(const uint16_t* last, const uint16_t* retained,
                                           const int32_t* sources, const int32_t* token_sources,
                                           const int32_t* drafts, uint16_t* x, int32_t* token_ids,
                                           const uint32_t* live_rows) {
  chop::ptx::griddep_launch_dependents();
  chop::ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  const int32_t source = sources[row];
  const uint4* from = reinterpret_cast<const uint4*>(
      source >= 0 ? last + uint64_t(source) * kWide : retained + uint64_t(-1 - source) * kWide);
  uint4* to = reinterpret_cast<uint4*>(x + uint64_t(row) * kWide);
  for (uint32_t vector = threadIdx.x; vector < kWide / 8; vector += blockDim.x)
    to[vector] = from[vector];
  if (threadIdx.x == 0 && token_sources[row] >= 0) token_ids[row] = drafts[token_sources[row]];
}

// The drafter's two input norms (Gemma RMSNorm, epsilon 1e-6): the target X
// normalized over all four streams together with the 10240-entry weight,
// and the next token's embedding with the 2560-entry weight. Grid [rows].
__device__ __forceinline__ void mtp_norms(const uint16_t* x, const uint16_t* hidden_weight,
                                          const uint16_t* embed, const uint16_t* embed_weight,
                                          uint16_t* x_normed, uint16_t* embed_normed,
                                          const uint32_t* live_rows) {
  chop::ptx::griddep_launch_dependents();
  chop::ptx::griddep_wait();
  __shared__ float scratch[32];
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  const uint16_t* xr = x + uint64_t(row) * kWide;
  float square = 0.f;
  for (uint32_t c = threadIdx.x; c < kWide; c += blockDim.x) {
    const float v = bf16_to_f32(xr[c]);
    square += v * v;
  }
  const float x_inv = rsqrtf(block_sum(square, scratch) / float(kWide) + 1e-6f);
  for (uint32_t c = threadIdx.x; c < kWide; c += blockDim.x)
    x_normed[uint64_t(row) * kWide + c] =
        f32_to_bf16(bf16_to_f32(xr[c]) * x_inv * (1.f + bf16_to_f32(hidden_weight[c])));
  const uint16_t* er = embed + uint64_t(row) * kHidden;
  square = 0.f;
  for (uint32_t c = threadIdx.x; c < kHidden; c += blockDim.x) {
    const float v = bf16_to_f32(er[c]);
    square += v * v;
  }
  const float e_inv = rsqrtf(block_sum(square, scratch) / float(kHidden) + 1e-6f);
  for (uint32_t c = threadIdx.x; c < kHidden; c += blockDim.x)
    embed_normed[uint64_t(row) * kHidden + c] =
        f32_to_bf16(bf16_to_f32(er[c]) * e_inv * (1.f + bf16_to_f32(embed_weight[c])));
}

// X[row][stream] = hidden[row][stream] + token[row], summed in F32 and
// rounded once. Grid [rows, streams].
__device__ __forceinline__ void mtp_combine(const uint16_t* hidden, const uint16_t* token,
                                            uint16_t* x, const uint32_t* live_rows) {
  chop::ptx::griddep_launch_dependents();
  chop::ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  const uint64_t base = (uint64_t(row) * kStreams + blockIdx.y) * kHidden;
  for (uint32_t c = threadIdx.x; c < kHidden; c += blockDim.x)
    x[base + c] = f32_to_bf16(bf16_to_f32(hidden[base + c]) +
                              bf16_to_f32(token[uint64_t(row) * kHidden + c]));
}

// This rank's best logit of each row: (F32 score bits, global id, 0, 0),
// the lowest id among equal scores. NaN never wins. Grid [rows].
__device__ __forceinline__ void draft_argmax_partial(const uint16_t* logits, uint32_t* best,
                                                     const uint32_t* live_rows, uint32_t rank) {
  chop::ptx::griddep_launch_dependents();
  chop::ptx::griddep_wait();
  __shared__ float scores[256];
  __shared__ uint32_t ids[256];
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  const uint16_t* values = logits + uint64_t(row) * kShardVocab;
  float score = -__int_as_float(0x7f800000);
  uint32_t id = 0xffffffffu;
  for (uint32_t c = threadIdx.x; c < kShardVocab; c += blockDim.x) {
    const float v = bf16_to_f32(values[c]);
    if (v > score) {
      score = v;
      id = c;
    }
  }
  scores[threadIdx.x] = score;
  ids[threadIdx.x] = id;
  __syncthreads();
  for (uint32_t half = blockDim.x / 2; half > 0; half >>= 1) {
    if (threadIdx.x < half) {
      const float other = scores[threadIdx.x + half];
      const uint32_t other_id = ids[threadIdx.x + half];
      if (other > scores[threadIdx.x] || (other == scores[threadIdx.x] && other_id < ids[threadIdx.x])) {
        scores[threadIdx.x] = other;
        ids[threadIdx.x] = other_id;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const uint32_t local = ids[0] == 0xffffffffu ? 0 : ids[0];
    uint32_t* out = best + uint64_t(row) * 4;
    out[0] = __float_as_uint(scores[0]);
    out[1] = rank * kShardVocab + local;
    out[2] = 0;
    out[3] = 0;
  }
}

// The draft of each row from every rank's best, [rows][kRanks][4 words]:
// the highest score, the lowest id on a tie. It becomes drafts[slots[row]]
// and, where targets[row] is at least zero, token_ids[targets[row]] of the
// verify rows. Grid [rows].
__device__ __forceinline__ void draft_argmax_final(const uint32_t* bests, const int32_t* slots,
                                                   const int32_t* targets, int32_t* drafts,
                                                   int32_t* token_ids,
                                                   const uint32_t* live_rows) {
  chop::ptx::griddep_launch_dependents();
  chop::ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows || threadIdx.x != 0) return;
  const uint32_t* candidates = bests + uint64_t(row) * (kRanks * 4);
  float score = __uint_as_float(candidates[0]);
  uint32_t id = candidates[1];
  for (int rank = 1; rank < kRanks; ++rank) {
    const float other = __uint_as_float(candidates[4 * rank]);
    const uint32_t other_id = candidates[4 * rank + 1];
    if (other > score || (other == score && other_id < id) || (score != score && other == other)) {
      score = other;
      id = other_id;
    }
  }
  drafts[slots[row]] = int32_t(id);
  if (targets[row] >= 0) token_ids[targets[row]] = int32_t(id);
}

}  // namespace chop::qwen38_bundle
