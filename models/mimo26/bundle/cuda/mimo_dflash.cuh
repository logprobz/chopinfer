#pragma once

// Bundle-owned row kernels of the DFlash drafter. Every kernel reads its live
// row count on the device, so the launches can be captured into the decode
// and verify graphs. Row r of a phase belongs to item token_batch[r] at
// position positions[r]; an item's first row sits at prefix_len[item].

#include <stdint.h>
#include "common/ptx.cuh"
#include <cuda_bf16.h>

// A rank's drafter KV heads and the rank count, which the build defines.
#if !defined(CHOP_MIMO_RANKS) || !defined(CHOP_MIMO_DFLASH_KV_HEADS) || !defined(CHOP_MIMO_DFLASH_CACHE_TOKENS) || !defined(CHOP_MIMO_DFLASH_HEAD_DIM) || !defined(CHOP_MIMO_DFLASH_SLOTS)
#error "CHOP_MIMO_RANKS, CHOP_MIMO_DFLASH_KV_HEADS, CHOP_MIMO_DFLASH_CACHE_TOKENS, CHOP_MIMO_DFLASH_HEAD_DIM and CHOP_MIMO_DFLASH_SLOTS are required"
#endif

namespace chop::mimo_dflash {

constexpr uint32_t kCacheTokens = CHOP_MIMO_DFLASH_CACHE_TOKENS;
constexpr uint32_t kKvHeads = CHOP_MIMO_DFLASH_KV_HEADS;
constexpr uint32_t kHeadDim = CHOP_MIMO_DFLASH_HEAD_DIM;
constexpr uint32_t kEntry = kKvHeads * kHeadDim;
// Most context cache slots an instance holds: one per item of its largest
// bucket, at most 256. The host packs slots below its served count.
constexpr uint32_t kSlots = CHOP_MIMO_DFLASH_SLOTS;
constexpr uint32_t kRanks = CHOP_MIMO_RANKS;

__device__ __forceinline__ uint4 scale_bf16x8(uint4 packed, float scale) {
  uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
  for (int i = 0; i < 4; ++i) {
    __nv_bfloat162 pair = *reinterpret_cast<__nv_bfloat162*>(&words[i]);
    float2 value = __bfloat1622float2(pair);
    pair = __floats2bfloat162_rn(value.x * scale, value.y * scale);
    words[i] = *reinterpret_cast<uint32_t*>(&pair);
  }
  return make_uint4(words[0], words[1], words[2], words[3]);
}

__device__ __forceinline__ bool draft_row(uint32_t row, const int32_t* positions,
                                          const int32_t* prefix_len,
                                          const int32_t* token_batch) {
  return positions[row] != prefix_len[token_batch[row]];
}

// Copies `width` BF16 columns of each live row, 16 bytes per thread step,
// scaling them unless `scale` is one. A null input writes zeros. Widths,
// strides and span offsets are multiples of eight elements.
__device__ void copy_rows(const uint16_t* input, uint16_t* output,
                          const uint32_t* live_rows, uint32_t width,
                          uint32_t input_stride, uint32_t output_stride,
                          uint32_t scale_bits) {
  chop::ptx::griddep_launch_dependents();
  chop::ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  const float scale = __uint_as_float(scale_bits);
  const uint32_t vectors = width / 8;
  uint4* target = reinterpret_cast<uint4*>(output + uint64_t(row) * output_stride);
  const uint4* source =
      input ? reinterpret_cast<const uint4*>(input + uint64_t(row) * input_stride) : nullptr;
  for (uint32_t i = threadIdx.x; i < vectors; i += blockDim.x) {
    uint4 value = source ? source[i] : make_uint4(0, 0, 0, 0);
    if (source && scale != 1.f) value = scale_bf16x8(value, scale);
    target[i] = value;
  }
}

// A draft block embeds its bonus token in row 0 and the mask embedding in
// every later row.
__device__ void mask_rows(uint16_t* activation, const uint16_t* mask,
                          const int32_t* positions, const int32_t* prefix_len,
                          const int32_t* token_batch, const uint32_t* live_rows) {
  chop::ptx::griddep_launch_dependents();
  chop::ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows || !draft_row(row, positions, prefix_len, token_batch)) return;
  uint4* target = reinterpret_cast<uint4*>(activation + uint64_t(row) * CHOP_MIMO_DFLASH_HIDDEN);
  const uint4* source = reinterpret_cast<const uint4*>(mask);
  for (uint32_t i = threadIdx.x; i < CHOP_MIMO_DFLASH_HIDDEN / 8; i += blockDim.x) target[i] = source[i];
}

// Writes one layer's context K (post norm and RoPE) and V into the per-slot
// circular cache [slot][position % kCacheTokens][head][kHeadDim]. A row more
// than the cache length behind its item's last position is overwritten by a
// later row of the same phase, so it is skipped.
__device__ void cache_write(const uint16_t* k, const uint16_t* v, uint16_t* k_cache,
                            uint16_t* v_cache, const int32_t* slot_ids,
                            const int32_t* token_batch, const int32_t* positions,
                            const int64_t* last_positions, const uint32_t* live_rows,
                            uint32_t k_stride, uint32_t v_stride, uint32_t v_scale_bits) {
  chop::ptx::griddep_launch_dependents();
  chop::ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  const int32_t item = token_batch[row];
  const int64_t position = positions[row];
  if (position < 0 || position + kCacheTokens <= last_positions[item]) return;
  const uint32_t slot = static_cast<uint32_t>(slot_ids[item]);
  // A slot past every cache is an input error, not a row to drop.
  if (slot >= kSlots) __trap();
  const uint64_t cell =
      (uint64_t(slot) * kCacheTokens + uint64_t(position) % kCacheTokens) * kEntry;
  const uint32_t vector = threadIdx.x % (kEntry / 8);
  if (threadIdx.x < kEntry / 8) {
    reinterpret_cast<uint4*>(k_cache + cell)[vector] =
        reinterpret_cast<const uint4*>(k + uint64_t(row) * k_stride)[vector];
  } else if (threadIdx.x < kEntry / 4) {
    const uint4 value = reinterpret_cast<const uint4*>(v + uint64_t(row) * v_stride)[vector];
    reinterpret_cast<uint4*>(v_cache + cell)[vector] =
        scale_bf16x8(value, __uint_as_float(v_scale_bits));
  }
}

// Every rank holds all ranks' (score, id) pairs after the all-gather:
// gathered[rank][2 * capacity] with scores in the first half and IDs in the
// second. Each draft row takes the best score, the lowest ID on ties, and
// stores it as the row's input token.
__device__ void drafts(const float* gathered, int32_t* token_ids, const int32_t* positions,
                       const int32_t* prefix_len, const int32_t* token_batch,
                       const uint32_t* live_rows, uint32_t capacity) {
  chop::ptx::griddep_launch_dependents();
  chop::ptx::griddep_wait();
  const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= *live_rows || row >= capacity || !draft_row(row, positions, prefix_len, token_batch))
    return;
  float best = -INFINITY;
  int32_t best_id = INT32_MAX;
  for (uint32_t rank = 0; rank < kRanks; ++rank) {
    const float* pairs = gathered + uint64_t(rank) * 2 * capacity;
    const float score = pairs[row];
    const int32_t id = __float_as_int(pairs[capacity + row]);
    if (score > best || (score == best && id < best_id)) {
      best = score;
      best_id = id;
    }
  }
  token_ids[row] = best_id;
}

}  // namespace chop::mimo_dflash
