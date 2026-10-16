#pragma once

// Lab only: phase stamps of the split mix-collapse with the input norm, for
// the lab's critical-path breakdown (the probe copies
// mix_norm_exact_probe.cuh and mix_norm_batched_probe.cuh). A probe entry
// takes `stamps` [rows, groups, kSlots] u64 and records, per CTA (g, row),
// the global timer (ns) when the stamping thread passes each point below,
// and the SM clock at two of them; a slot a CTA never reaches keeps the
// value the lab wrote before the launch. A stamp after a barrier that
// covers every thread of a phase, or the maximum of per-warp stamps, bounds
// that phase's work in the CTA; a stamp after stores marks them issued, not
// drained. No serving cubin includes this header.

#include <stdint.h>

namespace chop::sinkhorn_hc {

struct PhaseProbe {
  // Every CTA, thread 0: kernel entry and after its dependency wait (that
  // thread's progress), and after the ticket's closing block barrier.
  static constexpr int kEntry = 0;
  static constexpr int kWaited = 1;
  static constexpr int kTicket = 2;
  // The finishing CTA, thread 0 after a block barrier: the streams and
  // chain sums staged with the sum of squares, the logits published.
  static constexpr int kStaged = 3;
  static constexpr int kLogits = 4;
  // Lane 0 of warp 0 after a warp barrier: comb stored (the Sinkhorn's
  // end).
  static constexpr int kSinkhorn = 5;
  // Worker 0 (thread 32) after the workers' named barrier: pre and post
  // published, the collapse stored, the squares stored; after its walk:
  // the ordered sum (the walker alone holds it); after the barrier: the
  // reciprocal published.
  static constexpr int kPrePost = 6;
  static constexpr int kCollapse = 7;
  static constexpr int kSquares = 8;
  static constexpr int kSum = 9;
  static constexpr int kInv = 10;
  // Thread 0's SM clock at kWaited and at kLogits (finishing CTA), which
  // calibrate the timer against cycles.
  static constexpr int kWaitedClock = 11;
  static constexpr int kLogitsClock = 12;
  // Every CTA: warp w's lane 0 after a warp barrier that follows the warp's
  // chain-sum stores, at kChainWarps + w (8 warps).
  static constexpr int kChainWarps = 16;
  // The finishing CTA: worker warp w's (1 to 7) lane 0 after a warp barrier
  // that follows the warp's normed-row stores, at kOutputWarps + w - 1.
  static constexpr int kOutputWarps = 24;
  static constexpr int kSlots = 32;
};

// CTA (group, row)'s stamps.
__device__ __forceinline__ uint64_t* phase_slots(uint64_t* stamps, uint32_t row, int group,
                                                 int groups) {
  return stamps + (uint64_t(row) * groups + group) * PhaseProbe::kSlots;
}

__device__ __forceinline__ void phase_time(uint64_t* slots, int slot) {
  uint64_t now;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now) :: "memory");
  slots[slot] = now;
}

__device__ __forceinline__ void phase_clock(uint64_t* slots, int slot) {
  uint64_t now;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(now) :: "memory");
  slots[slot] = now;
}

// The calling warp's stamp at `slot` by its lane 0, after a warp barrier,
// so it follows every lane's preceding work.
__device__ __forceinline__ void phase_warp_time(uint64_t* slots, int slot) {
  __syncwarp();
  if ((threadIdx.x & 31) == 0) phase_time(slots, slot);
}

} // namespace chop::sinkhorn_hc
