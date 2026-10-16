#pragma once

// Qwen3.8 bundle-owned rows: stream expansion, the PLE residual, and the
// copies that stage and commit per-request state.
//
// Every entry publishes its dependents first and waits before touching
// memory, so PDL chains through it. Grids depend only on prepared
// capacities; live counts come from device scalars, so a decode graph
// replays for any live batch in its bucket.

#include <stdint.h>
#include "common/ptx.cuh"

namespace chop::qwen38_bundle {

constexpr uint32_t kHidden = CHOP_QWEN38_HIDDEN;
constexpr uint32_t kStreams = CHOP_QWEN38_STREAMS;
constexpr uint32_t kThreads = 256;
// One PLE state entry: the convolution history [9][10240] BF16, then the
// two n-gram history ids and padding to 128 bytes.
constexpr uint64_t kPleConvBytes = uint64_t(CHOP_QWEN38_PLE_CONV_ROWS) * kHidden * kStreams * 2;
constexpr uint64_t kPleEntryBytes = kPleConvBytes + 128;
constexpr uint32_t kEos = CHOP_QWEN38_NGRAM_EOS;
// Bytes one copy block moves.
constexpr uint64_t kChunk = 16 * 1024;

__device__ __forceinline__ float bf16_to_f32(uint16_t value) {
  return __uint_as_float(static_cast<uint32_t>(value) << 16);
}

__device__ __forceinline__ uint16_t f32_to_bf16(float value) {
  const uint32_t bits = __float_as_uint(value);
  if ((bits & 0x7fffffffu) > 0x7f800000u) return static_cast<uint16_t>((bits >> 16) | 0x40);
  return static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

// out = x + gated + conv over [rows][streams][2560], summed in F32 and
// rounded once. Grid [rows, streams].
__device__ __forceinline__ void ple_residual(const uint16_t* x, const uint16_t* gated,
                                             const uint16_t* conv, uint16_t* out,
                                             const uint32_t* live_rows) {
  chop::ptx::griddep_launch_dependents();
  chop::ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  const uint64_t base = (static_cast<uint64_t>(row) * kStreams + blockIdx.y) * kHidden;
  for (uint32_t column = threadIdx.x; column < kHidden; column += blockDim.x) {
    const uint64_t at = base + column;
    out[at] = f32_to_bf16(bf16_to_f32(x[at]) + bf16_to_f32(gated[at]) + bf16_to_f32(conv[at]));
  }
}

// Stages each live item's PLE state: its slot's committed entry when the
// item continues a sequence, otherwise a zero convolution history and an
// end-of-document n-gram history. Grid [items, entry chunks].
__device__ __forceinline__ void ple_stage(const uint8_t* pool, uint8_t* staging,
                                          const int32_t* slot_ids, const int32_t* prefix_len,
                                          const uint32_t* live_items) {
  chop::ptx::griddep_launch_dependents();
  chop::ptx::griddep_wait();
  const uint32_t item = blockIdx.x;
  if (item >= *live_items) return;
  const uint64_t first = blockIdx.y * kChunk;
  const uint64_t end = first + kChunk < kPleEntryBytes ? first + kChunk : kPleEntryBytes;
  uint8_t* to = staging + static_cast<uint64_t>(item) * kPleEntryBytes;
  if (prefix_len[item] > 0) {
    const uint8_t* from = pool + static_cast<uint64_t>(slot_ids[item]) * kPleEntryBytes;
    for (uint64_t at = first + 16ull * threadIdx.x; at < end; at += 16ull * blockDim.x)
      *reinterpret_cast<uint4*>(to + at) = *reinterpret_cast<const uint4*>(from + at);
    return;
  }
  for (uint64_t at = first + 16ull * threadIdx.x; at < end; at += 16ull * blockDim.x) {
    uint4 value = make_uint4(0, 0, 0, 0);
    if (at == kPleConvBytes) value.x = value.y = kEos;
    *reinterpret_cast<uint4*>(to + at) = value;
  }
}

// Copies entry src_ids[i] of every plane of `src` to entry dst_ids[i] of
// `dst` for each live item with a source (src_ids[i] >= 0). Planes are
// `plane_stride` bytes apart in both. Grid [items, planes, entry chunks].
__device__ __forceinline__ void copy_entries(const uint8_t* src, uint8_t* dst,
                                             const int32_t* src_ids, const int32_t* dst_ids,
                                             const uint32_t* live_items, uint64_t entry_bytes,
                                             uint64_t plane_stride) {
  chop::ptx::griddep_launch_dependents();
  chop::ptx::griddep_wait();
  const uint32_t item = blockIdx.x;
  if (item >= *live_items || src_ids[item] < 0) return;
  const uint64_t first = blockIdx.z * kChunk;
  if (first >= entry_bytes) return;
  const uint64_t end = first + kChunk < entry_bytes ? first + kChunk : entry_bytes;
  const uint64_t plane = static_cast<uint64_t>(blockIdx.y) * plane_stride;
  const uint8_t* from = src + plane + static_cast<uint64_t>(src_ids[item]) * entry_bytes;
  uint8_t* to = dst + plane + static_cast<uint64_t>(dst_ids[item]) * entry_bytes;
  for (uint64_t at = first + 16ull * threadIdx.x; at < end; at += 16ull * blockDim.x)
    *reinterpret_cast<uint4*>(to + at) = *reinterpret_cast<const uint4*>(from + at);
}

// Prefix-cache snapshot geometry (src/prefix.rs mirrors it). The linear
// state entries are one rank's: the build defines its convolution (3 x
// LINEAR_QKV BF16) and delta-rule (value heads x 128 x 128 F32) entry bytes.
#if !defined(CHOP_QWEN38_CONV_ENTRY_BYTES) || !defined(CHOP_QWEN38_DELTA_ENTRY_BYTES)
#error "CHOP_QWEN38_CONV_ENTRY_BYTES and CHOP_QWEN38_DELTA_ENTRY_BYTES are required"
#endif
constexpr uint64_t kLinearLayers = CHOP_QWEN38_LINEAR_LAYERS;
constexpr uint64_t kConvEntryBytes = CHOP_QWEN38_CONV_ENTRY_BYTES;
constexpr uint64_t kDeltaEntryBytes = CHOP_QWEN38_DELTA_ENTRY_BYTES;
constexpr uint64_t kRingKeySlot = CHOP_QWEN38_RING_SLOT_KEY_BYTES;
constexpr uint64_t kRingPositionSlot = CHOP_QWEN38_RING_SLOT_POSITION_BYTES;
constexpr uint64_t kRetainedSlot = uint64_t(CHOP_QWEN38_MAX_VERIFY_ROWS) * kHidden * kStreams * 2;
constexpr uint64_t kConvSnapshot = kLinearLayers * kConvEntryBytes;
constexpr uint64_t kDeltaSnapshot = kLinearLayers * kDeltaEntryBytes;
constexpr uint64_t kLinearSnapshot = kConvSnapshot + kDeltaSnapshot;

// Copies one slot's prefix-cache snapshot between its state and pool pages.
//
// The snapshot is, end to end: pool entry `index` (of `pool_entries` per
// plane) in every linear layer's convolution plane, then in every
// delta-rule plane, entry `slot` of the PLE entries, every one of
// `ring_layers` ring layers' keys and positions of slot `slot` in a ring of
// `ring_slots` slots, then, when `retained` is not null, entry `slot` of the
// retained drafter X rows. A slot's committed state or an anchor item's
// spares, which the host passes as these buffers. Byte `c` of the
// snapshot lives at `pool + pages[c / page_bytes] * page_bytes + c %
// page_bytes`. `to_pages` != 0 captures, 0 restores. One thread moves one
// 16-byte vector; every piece and page is a multiple of 16 bytes. Grid
// [ceil(snapshot bytes / 16 / 256)].
// The indexed entry names PLE state independently from the logical slot.
// The default preserves the original entry's ABI and addressing.
template<bool IndexedPle = false>
__device__ __forceinline__ void prefix_copy(uint8_t* conv_pool, uint8_t* delta_pool,
                                            uint8_t* ple_pool, uint8_t* ring, uint8_t* retained,
                                            uint8_t* pool, const uint64_t* pages,
                                            uint32_t ring_layers, uint64_t page_bytes,
                                            uint32_t slot, uint32_t index, uint32_t to_pages,
                                            uint32_t pool_entries, uint32_t ring_slots,
                                            uint32_t ple_entry = 0, uint32_t ple_entries = 0) {
  chop::ptx::griddep_launch_dependents();
  // The step that committed the slot, or the page-table upload, comes first.
  chop::ptx::griddep_wait();
  if (slot >= ring_slots || index >= pool_entries || page_bytes == 0 || page_bytes % 16 != 0)
    return;
  if constexpr (IndexedPle) {
    if (ple_entry >= ple_entries) return;
  }
  const uint64_t rings = kLinearSnapshot + kPleEntryBytes;
  const uint64_t ring_bytes = ring_layers * (kRingKeySlot + kRingPositionSlot);
  const uint64_t total = rings + ring_bytes + (retained != nullptr ? kRetainedSlot : 0);
  const uint64_t at = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 16;
  if (at >= total) return;
  const uint64_t ring_layer = ring_slots * (kRingKeySlot + kRingPositionSlot);
  uint8_t* held;
  if (at < kConvSnapshot) {
    held = conv_pool + ((at / kConvEntryBytes) * pool_entries + index) * kConvEntryBytes +
           at % kConvEntryBytes;
  } else if (at < kLinearSnapshot) {
    const uint64_t within = at - kConvSnapshot;
    held = delta_pool + ((within / kDeltaEntryBytes) * pool_entries + index) * kDeltaEntryBytes +
           within % kDeltaEntryBytes;
  } else if (at < rings) {
    held = ple_pool + static_cast<uint64_t>(IndexedPle ? ple_entry : slot) * kPleEntryBytes + (at - kLinearSnapshot);
  } else if (at < rings + ring_bytes) {
    const uint64_t within = at - rings;
    const uint64_t layer = within / (kRingKeySlot + kRingPositionSlot);
    const uint64_t piece = within % (kRingKeySlot + kRingPositionSlot);
    held = ring + layer * ring_layer +
           (piece < kRingKeySlot
                ? slot * kRingKeySlot + piece
                : ring_slots * kRingKeySlot + slot * kRingPositionSlot + (piece - kRingKeySlot));
  } else {
    held = retained + static_cast<uint64_t>(slot) * kRetainedSlot + (at - rings - ring_bytes);
  }
  uint8_t* page = pool + pages[at / page_bytes] * page_bytes + at % page_bytes;
  if (to_pages != 0) {
    *reinterpret_cast<uint4*>(page) = *reinterpret_cast<const uint4*>(held);
  } else {
    *reinterpret_cast<uint4*>(held) = *reinterpret_cast<const uint4*>(page);
  }
}

}  // namespace chop::qwen38_bundle
