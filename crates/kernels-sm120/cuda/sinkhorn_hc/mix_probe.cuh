#pragma once

// Phase stamps of the split mix-collapse with the input norm, for the lab's
// critical-path breakdown. A probe entry takes `stamps` [rows, groups,
// kSlots] u64 and records, per CTA (g, row), the global timer (ns) at the
// phase ends below, and the SM clock at two of them; a slot a CTA never
// reaches keeps the value the lab wrote before the launch. Serving entries
// instantiate the kernels with Probe false, where every stamp compiles to
// nothing.

#include <stdint.h>

namespace chop::sinkhorn_hc {

struct MixProbe {
  // Every CTA, thread 0: kernel entry, after the dependency wait, after its
  // chain sums are stored, after the ticket.
  static constexpr int kEntry = 0;
  static constexpr int kWaited = 1;
  static constexpr int kChains = 2;
  static constexpr int kTicket = 3;
  // The finishing CTA. Thread 0: the streams and chain sums staged with
  // the sum of squares, the logits published.
  static constexpr int kStaged = 4;
  static constexpr int kLogits = 5;
  // Lane 0 of warp 0: comb written (the Sinkhorn's end).
  static constexpr int kSinkhorn = 6;
  // Thread 32 (worker 0): pre and post published, the collapse written,
  // the squares and weights ready, the ordered sum done, the reciprocal
  // published, the normed row written.
  static constexpr int kPrePost = 7;
  static constexpr int kCollapse = 8;
  static constexpr int kSquares = 9;
  static constexpr int kSum = 10;
  static constexpr int kInv = 11;
  static constexpr int kOutput = 12;
  // Thread 0's SM clock at kWaited and at kLogits (finishing CTA), which
  // calibrate the timer against cycles.
  static constexpr int kWaitedClock = 13;
  static constexpr int kLogitsClock = 14;
  static constexpr int kSlots = 16;
};

// CTA (group, row)'s stamps, or null without a probe.
template<bool Probe>
__device__ __forceinline__ uint64_t* probe_slots(uint64_t* stamps, uint32_t row, int group,
                                                 int groups) {
  if constexpr (Probe) {
    return stamps + (uint64_t(row) * groups + group) * MixProbe::kSlots;
  } else {
    return nullptr;
  }
}

template<bool Probe>
__device__ __forceinline__ void probe_time(uint64_t* slots, int slot) {
  if constexpr (Probe) {
    uint64_t now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now) :: "memory");
    slots[slot] = now;
  }
}

template<bool Probe>
__device__ __forceinline__ void probe_clock(uint64_t* slots, int slot) {
  if constexpr (Probe) {
    uint64_t now;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(now) :: "memory");
    slots[slot] = now;
  }
}

} // namespace chop::sinkhorn_hc
