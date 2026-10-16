#pragma once

// Low-latency rank-ordered all-reduce of BF16 partials from F32 inputs, for
// a few rows over CTAs: every rank rounds its F32 partial once to BF16
// (cvt.rn.bf16.f32), and every output element is the F32 sum of the ranks'
// BF16 values in rank order from F32 zero, ((0 + x_0) + x_1) + ..., rounded
// once to BF16: bitwise allreduce_phased_bf16.cuh's entries. Each CTA
// holds kSliceElements elements, one 16-byte vector of eight BF16 values
// per thread.
//
// Transport: allreduce_ll_f32.cuh's, with two BF16 values in each 64-bit LL
// word instead of one F32 value: at step h CTA b sends every vector it
// holds to CTA b of rank ^ 2^h as four words, each two BF16 values and a
// flag word (push::ll_flag), so a matching flag proves its values arrived:
// no fences, no flag words, no acknowledgments. Receivers clear what they
// read; the inbox's slots alternate by epoch parity, and a slot's next
// writer comes two epochs later, after a call that needs this rank's input,
// which it sends only after the previous kernel (with its clears)
// completed. Power of two ranks only.
//
// Every CTA owns one slice and one epoch cell and exchanges with the CTAs of
// the same index on its peers only. It never waits on a CTA of its own
// grid, but it spins until its counterparts on the peer ranks have sent, so
// a caller admits the grid against the device's residency first (see
// allreduce_ll_f32.cuh). PDL: the entry triggers its dependents first.
//
// Inbox (u64 words): [parity 2][source rank][CTA, capacity_ctas][thread
// 256][4], zeroed at allocation, allreduce_ll_f32.cuh's layout; epochs: one
// u32 per CTA of the capacity, zeroed at allocation. capacity_ctas fixes
// the layout and must be the same on every rank and every call over one
// state; any row count up to it runs on that state.

#include <stdint.h>
#include "comm.cuh"
#include "../common/convert.cuh"

namespace chop::comm::ll_bf16 {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kSliceElements = 8 * kThreads;
// Words of one CTA's slot of one source: four per thread.
constexpr uint64_t kSlotWords = 4ull * kThreads;

// Test-only mutations, compiled into kernel-lab's mutant cubins: the
// dependency wait after the input loads (late wait) or none at all; a
// stale-flag exchange whose flag ignores the epoch and whose receivers keep
// what they read; and a sum that adds this rank's own values unrounded.
// Serving cubins leave CHOP_COMM_LL_BF16_TEST_WAIT,
// CHOP_COMM_LL_BF16_TEST_STALE and CHOP_COMM_LL_BF16_TEST_OWN_F32 undefined.
#if defined(CHOP_COMM_LL_BF16_TEST_WAIT)
constexpr int kWait = CHOP_COMM_LL_BF16_TEST_WAIT;  // 1 late, 2 none
#else
constexpr int kWait = 0;
#endif
#if defined(CHOP_COMM_LL_BF16_TEST_STALE)
constexpr bool kStale = true;
#else
constexpr bool kStale = false;
#endif
#if defined(CHOP_COMM_LL_BF16_TEST_OWN_F32)
constexpr bool kOwnF32 = true;
#else
constexpr bool kOwnF32 = false;
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

__device__ __forceinline__ uint32_t pack(float lo, float hi) {
  return uint32_t(convert::bf16_from_f32(lo)) | (uint32_t(convert::bf16_from_f32(hi)) << 16);
}

template <typename Inboxes>
__device__ void allreduce(const float* input, uint16_t* output, Inboxes inboxes,
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
  const float4 a = reinterpret_cast<const float4*>(input)[2 * vector];
  const float4 b = reinterpret_cast<const float4*>(input)[2 * vector + 1];
  if constexpr (kWait == 1) {
    ptx::griddep_wait();
  }
  const Vector128 own{pack(a.x, a.y), pack(a.z, a.w), pack(b.x, b.y), pack(b.z, b.w)};
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
  // From F32 zero, ranks in order.
  float sums[8] = {};
  for (uint32_t source = 0; source < ranks; ++source) {
    if (kOwnF32 && source == rank) {
      const float values[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
      for (int k = 0; k < 8; ++k) sums[k] += values[k];
      continue;
    }
    const Vector128 v = gathered[source][threadIdx.x];
    const uint32_t words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      sums[2 * w] += __uint_as_float(words[w] << 16);
      sums[2 * w + 1] += __uint_as_float(words[w] & 0xffff0000u);
    }
  }
  reinterpret_cast<Vector128*>(output)[vector] = {pack(sums[0], sums[1]), pack(sums[2], sums[3]),
                                                  pack(sums[4], sums[5]), pack(sums[6], sums[7])};
  // Every thread read the epoch before any writes it.
  __syncthreads();
  if (threadIdx.x == 0) epochs[cta] = epoch;
}

}  // namespace chop::comm::ll_bf16
