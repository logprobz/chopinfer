#pragma once

// Low-latency rank-ordered F32 all-reduce of a few rows over CTAs: bitwise
// comm.cuh's allreduce_oneshot over F32 (every element summed from F32 zero
// in rank order, ((0 + x0) + x1) + ...), with each 4096-element tile of the
// one-shot spread over kSlices CTAs of kSliceElements elements, one 16-byte
// vector per thread. The round entry (Out = uint16_t) writes each sum
// rounded once to BF16 (cvt.rn.bf16.f32, allreduce_phased_f32.cuh's
// rounding): bitwise that file's rounding entries.
//
// Transport: push.cuh's low-latency protocol, recursive doubling of raw
// inputs. At step h CTA b sends every input vector it holds to CTA b of
// rank ^ 2^h as four 64-bit LL words per vector, each one F32 value and a
// flag word (push::ll_flag, both BF16 halves NaN patterns; the inbox holds
// a flag only in a word's upper half), so a matching flag proves its value
// arrived: no fences, no flag words, no acknowledgments. Receivers clear
// what they read; the inbox's slots alternate by epoch parity, and a slot's
// next writer comes two epochs later, after a call that needs this rank's
// input, which it sends only after the previous kernel (with its clears)
// completed. Power of two ranks only.
//
// Every CTA owns one slice and one epoch cell and exchanges with the CTAs of
// the same index on its peers only. It never waits on a CTA of its own
// grid, but it spins until its counterparts on the peer ranks have sent, so
// a call finishes only if every rank's grid can become resident: with too
// few free slots, resident CTAs on one rank can wait for counterparts that
// another rank never schedules. A caller must therefore admit the grid
// against the device's residency (a bundle checks its largest grid at
// create with `bundle_runtime::residency` and otherwise keeps the
// one-shot). PDL: the entry triggers its dependents first, so a
// dependent grid starts only once every CTA of this grid has started (or
// finished) and cannot hold the slots it needs; the grid this entry
// depends on never waits for it, so it finishes and frees its slots.
//
// Inbox (u64 words): [parity 2][source rank][CTA, capacity_ctas][thread
// 256][4], zeroed at allocation; epochs: one u32 per CTA of the capacity,
// zeroed at allocation. capacity_ctas fixes the layout and must be the same
// on every rank and every call over one state; any row count up to it runs
// on that state.

#include <stdint.h>
#include "comm.cuh"
#include "../common/convert.cuh"

namespace chop::comm::ll_f32 {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kSliceElements = 4 * kThreads;
// Words of one CTA's slot of one source: four per thread.
constexpr uint64_t kSlotWords = 4ull * kThreads;

// Test-only mutations, compiled into kernel-lab's mutant cubins: the
// dependency wait after the input loads (late wait) or none at all; and a
// stale-flag exchange whose flag ignores the epoch and whose receivers keep
// what they read, so a receiver can take a slot's previous payload. Serving
// cubins leave CHOP_COMM_LL_F32_TEST_WAIT and CHOP_COMM_LL_F32_TEST_STALE
// undefined.
#if defined(CHOP_COMM_LL_F32_TEST_WAIT)
constexpr int kWait = CHOP_COMM_LL_F32_TEST_WAIT;  // 1 late, 2 none
#else
constexpr int kWait = 0;
#endif
#if defined(CHOP_COMM_LL_F32_TEST_STALE)
constexpr bool kStale = true;
#else
constexpr bool kStale = false;
#endif

// push::ll_receive without its clear (the stale-flag mutant).
__device__ __forceinline__ Vector128 receive_kept(uint64_t* slot, uint32_t flag) {
  uint64_t a, b, c, d;
  do {
    asm volatile("ld.volatile.global.v2.b64 {%0,%1}, [%2];"
                 : "=l"(a), "=l"(b) : "l"(slot) : "memory");
    asm volatile("ld.volatile.global.v2.b64 {%0,%1}, [%2];"
                 : "=l"(c), "=l"(d) : "l"(slot + 2) : "memory");
  } while (static_cast<uint32_t>(a >> 32) != flag || static_cast<uint32_t>(b >> 32) != flag ||
           static_cast<uint32_t>(c >> 32) != flag || static_cast<uint32_t>(d >> 32) != flag);
  return {static_cast<uint32_t>(a), static_cast<uint32_t>(b),
          static_cast<uint32_t>(c), static_cast<uint32_t>(d)};
}

__device__ __forceinline__ float word_float(uint32_t word) { return __uint_as_float(word); }

// Two sums rounded once to BF16, the first in the low half.
__device__ __forceinline__ uint32_t rounded_pair(float lo, float hi) {
  return uint32_t(convert::bf16_from_f32(lo)) | (uint32_t(convert::bf16_from_f32(hi)) << 16);
}

template <typename Out, typename Inboxes>
__device__ void allreduce(const float* input, Out* output, Inboxes inboxes,
                          unsigned int* epochs, uint64_t elements, uint32_t capacity_ctas,
                          uint32_t rank, uint32_t ranks) {
  ptx::griddep_launch_dependents();
  if constexpr (kWait == 0) ptx::griddep_wait();
  const uint32_t cta = blockIdx.x;
  const uint64_t ctas = elements / kSliceElements;
  if (elements % kSliceElements != 0 || ctas > capacity_ctas || rank >= ranks || ranks == 0 ||
      ranks > kMaxRanks || (ranks & (ranks - 1)) != 0)
    asm volatile("trap;");
  if (cta >= ctas) return;
  const uint64_t vector = uint64_t(cta) * kThreads + threadIdx.x;
  Vector128 own = reinterpret_cast<const Vector128*>(input)[vector];
  if constexpr (kWait == 1) {
    ptx::griddep_wait();
  }
  const uint32_t epoch = epochs[cta] + 1;
  const uint32_t flag = push::ll_flag(kStale ? 1u : epoch);
  const uint64_t parity = (epoch & 1) * ranks;
  auto slot = [&](uint32_t owner, uint32_t source) {
    return reinterpret_cast<uint64_t*>(peer(inboxes, owner)) +
           ((parity + source) * capacity_ctas + cta) * kSlotWords + threadIdx.x * 4ull;
  };
  __shared__ Vector128 gathered[kMaxRanks][kThreads];
  gathered[rank][threadIdx.x] = own;
  for (uint32_t span = 1; span < ranks; span <<= 1) {
    const uint32_t partner = rank ^ span;
    const uint32_t mine = rank & ~(span - 1);
    const uint32_t theirs = partner & ~(span - 1);
    for (uint32_t i = 0; i < span; ++i)
      push::ll_send(slot(partner, mine + i), gathered[mine + i][threadIdx.x], flag);
    for (uint32_t i = 0; i < span; ++i) {
      if constexpr (kStale)
        gathered[theirs + i][threadIdx.x] = receive_kept(slot(rank, theirs + i), flag);
      else
        gathered[theirs + i][threadIdx.x] = push::ll_receive(slot(rank, theirs + i), flag);
    }
  }
  // allreduce_oneshot's sum: from F32 zero, ranks in order.
  float x = 0.0f, y = 0.0f, z = 0.0f, w = 0.0f;
  for (uint32_t source = 0; source < ranks; ++source) {
    const Vector128 v = gathered[source][threadIdx.x];
    x += word_float(v.x);
    y += word_float(v.y);
    z += word_float(v.z);
    w += word_float(v.w);
  }
  if constexpr (sizeof(Out) == 2) {
    reinterpret_cast<uint2*>(output)[vector] = make_uint2(rounded_pair(x, y), rounded_pair(z, w));
  } else {
    reinterpret_cast<float4*>(output)[vector] = make_float4(x, y, z, w);
  }
  // Every thread read the epoch before any writes it.
  __syncthreads();
  if (threadIdx.x == 0) epochs[cta] = epoch;
}

}  // namespace chop::comm::ll_f32
