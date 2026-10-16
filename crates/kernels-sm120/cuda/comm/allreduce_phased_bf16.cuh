#pragma once

// Rank-ordered all-reduce of BF16 partials in one launch, from F32 inputs:
// every rank rounds its F32 partial once to BF16 (cvt.rn.bf16.f32), the
// ranks exchange the BF16 values, and every output element is the F32 sum
// of the ranks' BF16 values in rank order from F32 zero, ((0 + x_0) + x_1)
// + ..., rounded once to BF16. allreduce_phased_f32.cuh's protocol, block
// geometry, flags and state layout, with half its payload: the inbox and
// the gathered output hold BF16.
//
// Two-shot: sending CTA s rounds blocks s, s + senders, ... of the span of
// the phase's destination into that rank's inbox, one destination at a time
// (rank + 1, rank + 2, ...), and releases one flag per block; worker CTAs
// (every CTA after the senders) sum each own block in rank order as soon as
// all its sources arrived, with this rank's own values rounded as sent, and
// the senders gather the rounded sums into every peer's `output`. With no
// workers each sending CTA sums its own blocks. One-shot (calls of up to
// kOneShotMaxElements): every rank sends its whole rounded partial to each
// peer in turn and sums every element itself; its slots and flags alternate
// by epoch parity in a region of their own.
//
// The state is allreduce_phased_f32.cuh's (kFlagWords flag words, one
// epoch word, the inbox's one-shot region then one largest-owner-span slot
// per source), counted in BF16 elements, so an inbox sized for the F32
// entries holds these calls of up to twice its F32 elements. Flags carry
// the call's tagged epoch and receivers compare for equality; no
// acknowledgments are needed (see allreduce_phased_f32.cuh). This header's
// two entries may share one state and follow each other in any order.
// A state serves one payload format: regions are placed by element, so a
// BF16 call's regions overlap an F32 call's in bytes (F32 one-shot source
// 0 of one parity starts where the BF16 two-shot region does), and a fast
// rank's BF16 call could overwrite F32 values a slow peer has not read. A
// state may pass from one header's entries to the other's only after every
// rank finished every call on it (all ranks quiescent).
// Every CTA spins, so the whole grid must be resident on every rank.
//
// With `live_rows`, a call reduces only the first *live_rows rows of
// `row_elements` each and writes nothing past them; null reduces all
// `elements`. A call with nothing to reduce returns without advancing the
// epoch, on every rank alike.

#include <stdint.h>
#include "allreduce_phased_f32.cuh"

namespace chop::comm::phased_bf16 {

using phased_f32::Span;
using phased_f32::Vector128;
using phased_f32::kGatherFlags;
using phased_f32::kMaxBlocks;
using phased_f32::kMaxRanks;
using phased_f32::kMaxSenders;
using phased_f32::kOneShotFlags;
using phased_f32::kOneShotMaxElements;
using phased_f32::kOneShotRegion;
using phased_f32::kReduced;
using phased_f32::kBarrier;
using phased_f32::kDone;
using phased_f32::kScatterFlags;
using phased_f32::kThreads;
using phased_f32::kGranule;

// Eight elements, one 16-byte BF16 vector, per thread step.
constexpr uint32_t kVector = 8;

__device__ __forceinline__ uint32_t pack(float lo, float hi) {
  return uint32_t(convert::bf16_from_f32(lo)) | (uint32_t(convert::bf16_from_f32(hi)) << 16);
}

// Eight F32 inputs rounded once to BF16.
__device__ __forceinline__ Vector128 rounded(const float* input, uint64_t vector) {
  const float4 a = reinterpret_cast<const float4*>(input)[2 * vector];
  const float4 b = reinterpret_cast<const float4*>(input)[2 * vector + 1];
  return {pack(a.x, a.y), pack(a.z, a.w), pack(b.x, b.y), pack(b.z, b.w)};
}

__device__ __forceinline__ float low(uint32_t word) { return __uint_as_float(word << 16); }
__device__ __forceinline__ float high(uint32_t word) { return __uint_as_float(word & 0xffff0000u); }

// One CTA rounds `count` F32 inputs (a multiple of kVector) into BF16 at
// `to`, with four vectors in flight per thread.
__device__ __forceinline__ void send_rounded(uint16_t* to, const float* from, uint64_t count) {
  constexpr uint32_t kUnroll = 4;
  auto* target = reinterpret_cast<Vector128*>(to);
  const uint64_t vectors = count / kVector;
  uint64_t v = threadIdx.x;
  for (; v + (kUnroll - 1) * kThreads < vectors; v += kUnroll * kThreads) {
    Vector128 values[kUnroll];
#pragma unroll
    for (uint32_t i = 0; i < kUnroll; ++i) values[i] = rounded(from, v + i * kThreads);
#pragma unroll
    for (uint32_t i = 0; i < kUnroll; ++i) target[v + i * kThreads] = values[i];
  }
  for (; v < vectors; v += kThreads) target[v] = rounded(from, v);
}

// Rank-ordered sums of [first, first + count) of the owner's span, rounded
// once: this rank's values rounded from `input`, every peer's from its
// inbox slot.
__device__ __forceinline__ void sum_block(const float* input, const uint16_t* inbox,
                                          uint16_t* output, uint64_t slot, uint64_t first,
                                          uint64_t count, uint32_t rank, uint32_t ranks) {
  for (uint64_t v = threadIdx.x; v < count / kVector; v += kThreads) {
    const uint64_t at = first / kVector + v;
    Vector128 parts[kMaxRanks];
#pragma unroll
    for (uint32_t source = 0; source < kMaxRanks; ++source) {
      if (source >= ranks) continue;
#ifdef CHOP_COMM_PHASED_BF16_TEST_OWN_F32
      // Lab mutant: this rank's own values stay F32 (read below).
      if (source == rank) continue;
#endif
      parts[source] = source == rank
                          ? rounded(input, at)
                          : reinterpret_cast<const Vector128*>(inbox + source * slot)[at];
    }
    float sums[kVector] = {};
    auto add = [&](uint32_t source) {
#ifdef CHOP_COMM_PHASED_BF16_TEST_OWN_F32
      if (source == rank) {
        for (uint32_t k = 0; k < kVector; ++k) sums[k] += input[at * kVector + k];
        return;
      }
#endif
      const uint32_t words[4] = {parts[source].x, parts[source].y, parts[source].z,
                                 parts[source].w};
#pragma unroll
      for (uint32_t w = 0; w < 4; ++w) {
        sums[2 * w] += low(words[w]);
        sums[2 * w + 1] += high(words[w]);
      }
    };
#ifdef CHOP_COMM_PHASED_BF16_TEST_REVERSED
    // Lab mutant: sources in descending order.
#pragma unroll
    for (int source = kMaxRanks - 1; source >= 0; --source)
      if (source < static_cast<int>(ranks)) add(source);
#else
#pragma unroll
    for (uint32_t source = 0; source < kMaxRanks; ++source)
      if (source < ranks) add(source);
#endif
    reinterpret_cast<Vector128*>(output)[at] = {pack(sums[0], sums[1]), pack(sums[2], sums[3]),
                                                pack(sums[4], sums[5]), pack(sums[6], sums[7])};
  }
}

// A geometry mismatch would leave peers waiting; fail the launch instead.
template <bool OneShot>
__device__ __forceinline__ void check(uint64_t elements, uint32_t row_elements,
                                      const uint32_t* live_rows, uint32_t rank,
                                      uint32_t ranks, uint32_t senders) {
  if (blockDim.x != kThreads || senders == 0 || senders > kMaxSenders ||
      gridDim.x < senders || ranks == 0 || ranks > kMaxRanks || rank >= ranks ||
      elements % kGranule || (live_rows && (row_elements == 0 || row_elements % kGranule)) ||
      (OneShot && elements > kOneShotMaxElements))
    __trap();
}

// The call's geometry, allreduce_phased_f32.cuh's in BF16 elements.
template <bool OneShot>
struct Geometry {
  uint64_t count;
  uint32_t rank, ranks, senders, blocks, parity;
  uint64_t slot;  // elements between sources' slots

  __device__ Geometry(uint64_t count, uint32_t rank, uint32_t ranks, uint32_t senders,
                      uint32_t epoch_value)
      : count(count), rank(rank), ranks(ranks), senders(senders),
        blocks(phased_f32::blocks_of(count, OneShot ? 1 : ranks, senders)),
        parity(epoch_value & 1),
        slot(OneShot ? kOneShotMaxElements : phased_f32::slot_elements(count, ranks)) {}

  __device__ Span span(uint32_t owner) const {
    return OneShot ? Span{0, count} : phased_f32::owner_span(count, owner, ranks);
  }
  // First element of the receiver's slots (source 0's slot).
  __device__ uint64_t region() const {
#ifdef CHOP_COMM_PHASED_BF16_TEST_SHARED_PARITY
    // Lab mutant: both one-shot parities share one inbox region (the flags
    // stay parity-separated), so a rank two calls ahead can overwrite
    // partials a slow peer has not read yet.
    return OneShot ? 0 : kOneShotRegion;
#else
    return OneShot ? uint64_t(parity) * kMaxRanks * kOneShotMaxElements : kOneShotRegion;
#endif
  }
  __device__ uint32_t scatter_flag(uint32_t source, uint32_t block) const {
    return OneShot ? kOneShotFlags + (parity * kMaxRanks + source) * kMaxBlocks + block
                   : kScatterFlags + source * kMaxBlocks + block;
  }
};

#ifdef CHOP_COMM_PHASED_BF16_TEST_SLOW_RANK
// Lab timing hook (no defect): on rank CHOP_COMM_PHASED_BF16_TEST_SLOW_RANK
// every CTA sleeps kSlowNanoseconds when it first reduces, after this
// rank's partials went out and before it reads its inbox or gathers. Peers
// then finish (a missing gather wait returns early) and start their next
// call (a shared one-shot parity overwrites unread values) inside the
// window, so those mutants must change outputs; the served entries wait.
constexpr uint64_t kSlowNanoseconds = 2000000;

__device__ __forceinline__ uint64_t global_nanoseconds() {
  uint64_t now;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
  return now;
}

__device__ __forceinline__ void slow_rank(uint32_t rank, bool& slept) {
  if (slept || rank != CHOP_COMM_PHASED_BF16_TEST_SLOW_RANK) return;
  slept = true;
  if (threadIdx.x != 0) return;
  const uint64_t start = global_nanoseconds();
  while (global_nanoseconds() - start < kSlowNanoseconds) __nanosleep(10000);
}
#endif

template <bool OneShot, typename Inboxes, typename Outputs, typename Flags>
__device__ void allreduce(const float* input, uint16_t* output, Inboxes inboxes,
                          Outputs outputs, Flags flags, unsigned int* epoch, uint64_t elements,
                          uint32_t row_elements, const uint32_t* live_rows, uint32_t rank,
                          uint32_t ranks, uint32_t senders) {
  using phased_f32::block_of;
  using phased_f32::peer;
  ptx::griddep_launch_dependents();
  check<OneShot>(elements, row_elements, live_rows, rank, ranks, senders);
#ifdef CHOP_COMM_PHASED_BF16_TEST_LATE_WAIT
  // Lab mutant: the first scatter phase reads the input before the PDL wait
  // (all elements: the mutant runs without live rows). It reads the epoch
  // early for the one-shot's parity; the lab runs it on a fresh state.
  const bool early = blockIdx.x < senders && ranks > 1 && live_rows == nullptr;
  if (early) {
    const Geometry<OneShot> g(elements, rank, ranks, senders, epoch[0] + 1);
    const uint32_t dest = (rank + 1) % ranks;
    const Span span = g.span(dest);
    for (uint32_t block = blockIdx.x; block < g.blocks; block += senders) {
      const Span part = block_of(span.count, block, g.blocks);
      if (part.count)
        send_rounded(peer(inboxes, dest) + g.region() + rank * g.slot + part.first,
                     input + span.first + part.first, part.count);
    }
  }
#endif
  ptx::griddep_wait();
  const uint64_t count =
      live_rows ? min(elements, uint64_t(*live_rows) * row_elements) : elements;
  if (count == 0) return;
  const uint32_t epoch_value = epoch[0] + 1;
  const uint32_t mark = phased_f32::tagged(epoch_value);
  const Geometry<OneShot> g(count, rank, ranks, senders, epoch_value);
  const uint32_t blocks = g.blocks;
  unsigned int* local = peer(flags, rank);
  const Span own = g.span(rank);
  const uint32_t workers = gridDim.x - senders;
  const bool sender = blockIdx.x < senders;
  // Sums block `block` of the own span once every source's block arrived.
#ifdef CHOP_COMM_PHASED_BF16_TEST_SLOW_RANK
  bool slept = false;
#endif
  auto reduce = [&](uint32_t block) {
    const Span part = block_of(own.count, block, blocks);
    if (part.count == 0) return;
#ifdef CHOP_COMM_PHASED_BF16_TEST_SLOW_RANK
    slow_rank(rank, slept);
#endif
#ifndef CHOP_COMM_PHASED_BF16_TEST_NO_SCATTER_WAIT
    if (threadIdx.x < ranks && threadIdx.x != rank)
      phased_f32::wait_system(local + g.scatter_flag(threadIdx.x, block), mark);
#endif
    __syncthreads();
    sum_block(input + own.first, peer(inboxes, rank) + g.region(), output + own.first, g.slot,
              part.first, part.count, rank, ranks);
    __syncthreads();
    if (!OneShot && workers && threadIdx.x == 0)
      phased_f32::release_gpu(local + kReduced + block, mark);
  };
  // The call completes only with every peer's gathered blocks in `output`.
  auto wait_gathered = [&](uint32_t first, uint32_t step) {
#ifndef CHOP_COMM_PHASED_BF16_TEST_NO_GATHER_WAIT
    for (uint32_t block = first; block < blocks; block += step) {
      const uint32_t source = threadIdx.x;
      if (source < ranks && source != rank &&
          block_of(phased_f32::owner_span(count, source, ranks).count, block, blocks).count)
        phased_f32::wait_system(local + kGatherFlags + source * kMaxBlocks + block, mark);
    }
#endif
  };
  if (sender) {
    const uint32_t s = blockIdx.x;
    uint32_t arrived = 0;
    // Every sending CTA issued its writes to this destination before any
    // writes to the next one.
    auto next_destination = [&]() {
      __syncthreads();
      if (senders == 1) return;
      if (threadIdx.x == 0) {
        arrived += senders;
        atomicAdd(local + kBarrier, 1u);
        uint32_t value;
        do {
          asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];"
                       : "=r"(value) : "l"(local + kBarrier) : "memory");
        } while (value < arrived);
      }
      __syncthreads();
    };
    for (uint32_t step = 1; step < ranks; ++step) {
      const uint32_t dest = (rank + step) % ranks;
      const Span span = g.span(dest);
      for (uint32_t block = s; block < blocks; block += senders) {
        const Span part = block_of(span.count, block, blocks);
        if (part.count == 0) continue;
#ifdef CHOP_COMM_PHASED_BF16_TEST_LATE_WAIT
        if (!(early && step == 1))
#endif
        send_rounded(peer(inboxes, dest) + g.region() + rank * g.slot + part.first,
                     input + span.first + part.first, part.count);
        __syncthreads();
        if (threadIdx.x == 0)
          phased_f32::release_system(peer(flags, dest) + g.scatter_flag(rank, block), mark);
      }
      if (!OneShot || step + 1 < ranks) next_destination();
    }
    if (workers == 0)
      for (uint32_t block = s; block < blocks; block += senders) reduce(block);
    if constexpr (!OneShot) {
      for (uint32_t step = 1; step < ranks; ++step) {
        const uint32_t dest = (rank + step) % ranks;
        for (uint32_t block = s; block < blocks; block += senders) {
          const Span part = block_of(own.count, block, blocks);
          if (part.count == 0) continue;
          if (step == 1 && workers) {
            if (threadIdx.x == 0) phased_f32::wait_gpu(local + kReduced + block, mark);
            __syncthreads();
          }
          phased_f32::copy_bytes(peer(outputs, dest) + own.first + part.first,
                                 output + own.first + part.first, part.count * 2);
          __syncthreads();
          if (threadIdx.x == 0)
            phased_f32::release_system(
                peer(flags, dest) + kGatherFlags + rank * kMaxBlocks + block, mark);
        }
        if (step + 1 < ranks) next_destination();
      }
      if (workers == 0) wait_gathered(s, senders);
    }
  } else {
    const uint32_t worker = blockIdx.x - senders;
    for (uint32_t block = worker; block < blocks; block += workers) reduce(block);
    if constexpr (!OneShot) wait_gathered(worker, workers);
  }
  // Every CTA arrives; the last clears the local words and advances the
  // epoch after all of them read it.
  __syncthreads();
  if (threadIdx.x != 0) return;
  __threadfence();
  if (atomicAdd(local + kDone, 1u) == gridDim.x - 1) {
    local[kBarrier] = 0;
    local[kDone] = 0;
    epoch[0] = epoch_value;
  }
}

} // namespace chop::comm::phased_bf16
