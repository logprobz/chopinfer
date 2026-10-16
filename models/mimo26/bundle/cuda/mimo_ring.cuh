#pragma once

// Saves the SWA ring entries one MiMo phase is about to overwrite.
//
// Each rank's ring state holds a working half, which the attention and cache
// write kernels use, followed by a shadow half of the same layout. Working is
// [layer][K plane | V plane], and each plane is [slot][ring position][entry].
// Before a step writes working, this copies every entry the step touches into
// the shadow, so cancel can restore them; nothing else reads the shadow.
//
// One block handles one (item, recent position, layer) triple. The item's
// positions come from the phase's device inputs, so the launch has no
// per-step arguments and can be captured into a decode graph.
//
// The plane strides follow the ring's slot count, one slot per item of the
// instance's largest graph bucket. The default entries hold 32 slots; an
// instance serving another count runs the entries compiled for its slots
// (`Slots`), so a 32-slot instance keeps its launches.

#include <stdint.h>
#include "common/ptx.cuh"

// A rank's SWA KV heads, the ring state's working-half bytes and the ring
// blocks' threads follow the rank count, which the build defines them for.
#if !defined(CHOP_MIMO_SWA_KV_HEADS) || !defined(CHOP_MIMO_RING_WORKING_BYTES) || \
    !defined(CHOP_MIMO_RING_THREADS) || !defined(CHOP_MIMO_RING_TOKENS) || !defined(CHOP_MIMO_RING_SLOTS) || !defined(CHOP_MIMO_SWA_LAYERS) || !defined(CHOP_MIMO_SWA_QK_DIM) || !defined(CHOP_MIMO_SWA_V_DIM) || !defined(CHOP_MIMO_PREFIX_THREADS)
#error "CHOP_MIMO_SWA_KV_HEADS, CHOP_MIMO_RING_WORKING_BYTES CHOP_MIMO_RING_THREADS, CHOP_MIMO_RING_TOKENS, CHOP_MIMO_RING_SLOTS, CHOP_MIMO_SWA_LAYERS, CHOP_MIMO_SWA_QK_DIM, CHOP_MIMO_SWA_V_DIM and CHOP_MIMO_PREFIX_THREADS are required"
#endif

namespace chop::mimo_ring {

constexpr uint32_t kRingTokens = CHOP_MIMO_RING_TOKENS;
constexpr uint32_t kSlots = CHOP_MIMO_RING_SLOTS;
constexpr uint32_t kLayers = CHOP_MIMO_SWA_LAYERS;
constexpr uint64_t kKEntry = CHOP_MIMO_SWA_KV_HEADS * CHOP_MIMO_SWA_QK_DIM * 2;
constexpr uint64_t kVEntry = CHOP_MIMO_SWA_KV_HEADS * CHOP_MIMO_SWA_V_DIM * 2;
constexpr uint32_t kKVectors = kKEntry / 16;
constexpr uint32_t kVectors = (kKEntry + kVEntry) / 16;
constexpr uint32_t kThreads = CHOP_MIMO_RING_THREADS;
static_assert(kVectors <= kThreads, "one thread per 16-byte vector");

// The plane and half strides of a ring of `Slots` slots.
template<uint32_t Slots>
struct Planes {
  static_assert(Slots >= 1 && Slots <= 256 && (Slots & (Slots - 1)) == 0,
                "a power-of-two ring of at most 256 slots");
  static constexpr uint64_t kKLayer = static_cast<uint64_t>(Slots) * kRingTokens * kKEntry;
  static constexpr uint64_t kVLayer = static_cast<uint64_t>(Slots) * kRingTokens * kVEntry;
  static constexpr uint64_t kLayer = kKLayer + kVLayer;
  static constexpr uint64_t kWorking = kLayers * kLayer;
};
static_assert(Planes<kSlots>::kWorking == CHOP_MIMO_RING_WORKING_BYTES,
              "matches RING_STATE_BYTES_PER_RANK / 2");

// blockIdx.x = item * span + recent, blockIdx.y = layer. `span` is the number
// of recent positions a block row covers: 1 for decode, kRingTokens for
// prefill. An item touches its last min(q_len, kRingTokens) positions.
template<uint32_t Slots = kSlots>
__device__ __forceinline__ void backup(uint8_t* state, const int32_t* slot_ids,
                                       const int32_t* q_len, const int32_t* prefix_len,
                                       const uint32_t* live_batch, uint32_t span) {
  chop::ptx::griddep_launch_dependents();
  // The previous step's kernels wrote working; later ring writers wait for
  // this kernel through the dependent chain.
  chop::ptx::griddep_wait();
  if (span == 0 || blockDim.x != kThreads || blockIdx.y >= kLayers) return;
  const uint32_t item = blockIdx.x / span;
  const uint32_t recent = blockIdx.x % span;
  if (item >= live_batch[0]) return;
  const int32_t count = q_len[item];
  const uint32_t touched = count < static_cast<int32_t>(kRingTokens) ? count : kRingTokens;
  if (count <= 0 || recent >= touched) return;
  const uint32_t position = static_cast<uint32_t>(prefix_len[item] + count - 1) - recent;
  const uint64_t cell = static_cast<uint64_t>(slot_ids[item]) * kRingTokens +
                        position % kRingTokens;
  const uint32_t vector = threadIdx.x;
  if (vector >= kVectors) return;
  using P = Planes<Slots>;
  const uint64_t offset =
      blockIdx.y * P::kLayer + (vector < kKVectors
                                    ? cell * kKEntry + vector * 16ull
                                    : P::kKLayer + cell * kVEntry + (vector - kKVectors) * 16ull);
  *reinterpret_cast<uint4*>(state + P::kWorking + offset) =
      *reinterpret_cast<const uint4*>(state + offset);
}

// Copies a verify step's rejected ring positions back from the shadow.
//
// blockIdx.x = item * span + recent, blockIdx.y = layer. Item `item` restores
// positions starts[item] .. starts[item] + counts[item], each at most `span`
// long; the backup that ran before the step saved exactly these cells.
template<uint32_t Slots = kSlots>
__device__ __forceinline__ void restore(uint8_t* state, const int32_t* slot_ids,
                                        const int32_t* starts, const int32_t* counts,
                                        const uint32_t* live_items, uint32_t span) {
  chop::ptx::griddep_launch_dependents();
  // The step's ring writers must finish before their entries are replaced.
  chop::ptx::griddep_wait();
  if (span == 0 || blockDim.x != kThreads || blockIdx.y >= kLayers) return;
  const uint32_t item = blockIdx.x / span;
  const uint32_t recent = blockIdx.x % span;
  if (item >= live_items[0]) return;
  const int32_t count = counts[item];
  if (count <= 0 || recent >= static_cast<uint32_t>(count)) return;
  const uint32_t position = static_cast<uint32_t>(starts[item]) + recent;
  const uint64_t cell = static_cast<uint64_t>(slot_ids[item]) * kRingTokens +
                        position % kRingTokens;
  const uint32_t vector = threadIdx.x;
  if (vector >= kVectors) return;
  using P = Planes<Slots>;
  const uint64_t offset =
      blockIdx.y * P::kLayer + (vector < kKVectors
                                    ? cell * kKEntry + vector * 16ull
                                    : P::kKLayer + cell * kVEntry + (vector - kKVectors) * 16ull);
  *reinterpret_cast<uint4*>(state + offset) =
      *reinterpret_cast<const uint4*>(state + P::kWorking + offset);
}

constexpr uint64_t kKSlot = static_cast<uint64_t>(kRingTokens) * kKEntry;
constexpr uint64_t kSlot = static_cast<uint64_t>(kRingTokens) * (kKEntry + kVEntry);
constexpr uint32_t kPrefixThreads = CHOP_MIMO_PREFIX_THREADS;

// Copies one slot's prefix-cache snapshot between its state and pool pages.
//
// The snapshot is, end to end: every target layer's K then V entries of
// `slot` in the working ring, then `drafter_bytes` of the slot's drafter
// history starting at `drafter` (none when 0). Byte `c` of the snapshot lives
// at `pool + pages[c / page_bytes] * page_bytes + c % page_bytes`.
// `to_pages` != 0 captures, 0 restores. One thread moves one 16-byte vector;
// every piece and page is a multiple of 16 bytes.
template<uint32_t Slots = kSlots>
__device__ __forceinline__ void prefix_copy(uint8_t* state, uint8_t* drafter,
                                            uint64_t drafter_bytes, uint8_t* pool,
                                            const uint64_t* pages, uint32_t page_count,
                                            uint64_t page_bytes, uint32_t slot,
                                            uint32_t to_pages) {
  chop::ptx::griddep_launch_dependents();
  // The step that wrote the slot, or the table upload, precedes this kernel.
  chop::ptx::griddep_wait();
  using P = Planes<Slots>;
  if (blockDim.x != kPrefixThreads || slot >= Slots || page_bytes == 0 ||
      page_bytes % 16 != 0 || drafter_bytes % 16 != 0)
    return;
  const uint64_t target = kLayers * kSlot;
  const uint64_t at = (static_cast<uint64_t>(blockIdx.x) * kPrefixThreads + threadIdx.x) * 16;
  if (at >= target + drafter_bytes || at / page_bytes >= page_count) return;
  uint8_t* held;
  if (at < target) {
    const uint64_t within = at % kSlot;
    uint8_t* layer = state + (at / kSlot) * P::kLayer;
    held = within < kKSlot ? layer + slot * kKSlot + within
                           : layer + P::kKLayer + slot * (kSlot - kKSlot) + (within - kKSlot);
  } else {
    held = drafter + (at - target);
  }
  uint8_t* page = pool + pages[at / page_bytes] * page_bytes + at % page_bytes;
  if (to_pages != 0) {
    *reinterpret_cast<uint4*>(page) = *reinterpret_cast<const uint4*>(held);
  } else {
    *reinterpret_cast<uint4*>(held) = *reinterpret_cast<const uint4*>(page);
  }
}

}  // namespace chop::mimo_ring
