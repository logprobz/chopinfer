#pragma once

// Rank-ordered all-reduce of BF16 partials from F32 inputs whose writers
// take turns at each destination: allreduce_phased_bf16.cuh's two-shot
// (the same rounding, inbox slots, block flags and rank-order sums, so
// every output element is bitwise that entry's), with one more rule on
// when a rank starts each phase.
//
// A call runs 2 (ranks - 1) phases on each rank: the reduce-scatter's
// destinations rank + 1, ..., rank + ranks - 1, then the all-gather's in
// the same order. When every rank is in the same phase the destinations
// form a permutation, so each GPU sends to one peer and receives from one
// peer. When ranks start a call at different times (a rank's compute ran
// late), a phase-by-phase schedule that each rank keeps on its own clock
// lets two or three ranks write into one GPU at once, and they share its
// link. Here a rank starts phase t into destination d only after the
// rank that wrote into d in phase t - 1 finished issuing its writes: that
// writer, after its senders' barrier, releases a turn word in the next
// writer's flags (tagged with the call's epoch, compared for equality, as
// every flag). Turns order transfers only; every value still travels under
// the block flags of allreduce_phased_bf16.cuh, so a turn never makes a
// sum read early. Phase 1 waits for no turn (the previous call may have
// been another entry's). The writer into rank + k(t) in phase t - 1 is
// rank + k(t) - k(t - 1), which no turn waits on when it is this rank.
//
// The state is allreduce_phased_bf16.cuh's with kTurnWords more flag words
// after its kFlagWords (zeroed with them): that header's entries may share
// the state and follow this entry in any order. Every CTA spins, so the
// whole grid must be resident on every rank.
//
// Lab-only defines: CHOP_COMM_PHASED_TURNS_BF16_TEST_UNTURNED drops the
// turn waits (the transfer order of allreduce_phased_bf16.cuh's two-shot,
// for timing); CHOP_COMM_PHASED_TURNS_BF16_TEST_BLOCK_GRANULES sets the
// granules per block (default kBlockGranules);
// CHOP_COMM_PHASED_TURNS_BF16_TEST_LATE_SCATTER=r makes rank r sleep before
// its last scatter phase (a timing hook, no defect); the mutants
// CHOP_COMM_PHASED_TURNS_BF16_TEST_NO_SCATTER_WAIT, _NO_GATHER_WAIT and
// _LATE_WAIT drop a block wait or read the input before the PDL wait, and
// allreduce_phased_bf16.cuh's _REVERSED, _OWN_F32 and _SLOW_RANK defines
// act in the sums and reductions this header shares with it.

#include <stdint.h>
#include "allreduce_phased_bf16.cuh"

namespace chop::comm::phased_turns_bf16 {

using phased_f32::Span;
using phased_f32::kBarrier;
using phased_f32::kDone;
using phased_f32::kFlagWords;
using phased_f32::kGatherFlags;
using phased_f32::kGranule;
using phased_f32::kMaxBlocks;
using phased_f32::kMaxRanks;
using phased_f32::kOneShotRegion;
using phased_f32::kReduced;
using phased_f32::kScatterFlags;
using phased_f32::kThreads;

// Turn words [phase], local: phases 2 .. 2 (kMaxRanks - 1).
constexpr uint32_t kTurns = kFlagWords;
constexpr uint32_t kTurnWords = 2 * kMaxRanks;
constexpr uint32_t kTurnFlagWords = kTurns + kTurnWords;

#ifdef CHOP_COMM_PHASED_TURNS_BF16_TEST_BLOCK_GRANULES
constexpr uint32_t kBlockGranules = CHOP_COMM_PHASED_TURNS_BF16_TEST_BLOCK_GRANULES;
#else
constexpr uint32_t kBlockGranules = phased_f32::kBlockGranules;
#endif

// phased_f32::blocks_of with this header's granules per block.
__device__ __forceinline__ uint32_t blocks_of(uint64_t count, uint32_t ranks,
                                              uint32_t senders) {
  const uint64_t blocks = count / kGranule / ranks / kBlockGranules;
  const uint64_t at_least = blocks > senders ? blocks : senders;
  return static_cast<uint32_t>(at_least < kMaxBlocks ? at_least : kMaxBlocks);
}

#ifdef CHOP_COMM_PHASED_TURNS_BF16_TEST_LATE_SCATTER
// Lab timing hook (no defect): every sending CTA of rank
// CHOP_COMM_PHASED_TURNS_BF16_TEST_LATE_SCATTER sleeps 2 ms before its last
// scatter phase. No turn of that phase's destination waits for that data
// (its first gather turn follows other ranks' phases), so without the
// scatter flags the destination would sum the slot before it arrives.
constexpr uint64_t kLateScatterNanoseconds = 2000000;

__device__ __forceinline__ void late_scatter() {
  if (threadIdx.x == 0) {
    uint64_t start, now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    do {
      __nanosleep(10000);
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    } while (now - start < kLateScatterNanoseconds);
  }
  __syncthreads();
}
#endif

// The destination offset of phase t (1-based): rank + offset(t).
__device__ __forceinline__ uint32_t offset(uint32_t phase, uint32_t ranks) {
  return phase < ranks ? phase : phase - (ranks - 1);
}

template <typename Inboxes, typename Outputs, typename Flags>
__device__ void allreduce(const float* input, uint16_t* output, Inboxes inboxes,
                          Outputs outputs, Flags flags, unsigned int* epoch, uint64_t elements,
                          uint32_t row_elements, const uint32_t* live_rows, uint32_t rank,
                          uint32_t ranks, uint32_t senders) {
  using phased_f32::block_of;
  using phased_f32::owner_span;
  using phased_f32::peer;
  ptx::griddep_launch_dependents();
  phased_bf16::check<false>(elements, row_elements, live_rows, rank, ranks, senders);
#ifdef CHOP_COMM_PHASED_TURNS_BF16_TEST_LATE_WAIT
  // Lab mutant: the first scatter phase reads the input before the PDL wait
  // (all elements: the mutant runs without live rows).
  const bool early = blockIdx.x < senders && ranks > 1 && live_rows == nullptr;
  if (early) {
    const uint32_t dest = (rank + 1) % ranks;
    const Span span = owner_span(elements, dest, ranks);
    const uint32_t blocks = blocks_of(elements, ranks, senders);
    const uint64_t slot = phased_f32::slot_elements(elements, ranks);
    for (uint32_t block = blockIdx.x; block < blocks; block += senders) {
      const Span part = block_of(span.count, block, blocks);
      if (part.count)
        phased_bf16::send_rounded(peer(inboxes, dest) + kOneShotRegion + rank * slot + part.first,
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
  const uint32_t blocks = blocks_of(count, ranks, senders);
  const uint64_t slot = phased_f32::slot_elements(count, ranks);
  unsigned int* local = peer(flags, rank);
  const Span own = owner_span(count, rank, ranks);
  const uint32_t workers = gridDim.x - senders;
  const bool sender = blockIdx.x < senders;
#ifdef CHOP_COMM_PHASED_BF16_TEST_SLOW_RANK
  bool slept = false;
#endif
  // Sums block `block` of the own span once every source's block arrived.
  auto reduce = [&](uint32_t block) {
    const Span part = block_of(own.count, block, blocks);
    if (part.count == 0) return;
#ifdef CHOP_COMM_PHASED_BF16_TEST_SLOW_RANK
    phased_bf16::slow_rank(rank, slept);
#endif
#ifndef CHOP_COMM_PHASED_TURNS_BF16_TEST_NO_SCATTER_WAIT
    if (threadIdx.x < ranks && threadIdx.x != rank)
      phased_f32::wait_system(local + kScatterFlags + threadIdx.x * kMaxBlocks + block, mark);
#endif
    __syncthreads();
    phased_bf16::sum_block(input + own.first, peer(inboxes, rank) + kOneShotRegion,
                           output + own.first, slot, part.first, part.count, rank, ranks);
    __syncthreads();
    if (workers && threadIdx.x == 0) phased_f32::release_gpu(local + kReduced + block, mark);
  };
  // The call completes only with every peer's gathered blocks in `output`.
  auto wait_gathered = [&](uint32_t first, uint32_t step) {
#ifndef CHOP_COMM_PHASED_TURNS_BF16_TEST_NO_GATHER_WAIT
    for (uint32_t block = first; block < blocks; block += step) {
      const uint32_t source = threadIdx.x;
      if (source < ranks && source != rank &&
          block_of(owner_span(count, source, ranks).count, block, blocks).count)
        phased_f32::wait_system(local + kGatherFlags + source * kMaxBlocks + block, mark);
    }
#endif
  };
  if (sender) {
    const uint32_t s = blockIdx.x;
    const uint32_t phases = 2 * (ranks - 1);
    uint32_t arrived = 0;
    // Every sending CTA issued its writes of phase `phase` before any CTA
    // starts the next; then CTA 0 hands the destination's turn to the
    // rank that writes into it next, and every CTA waits for its own turn
    // at the next phase's destination.
    auto next_phase = [&](uint32_t phase) {
      __syncthreads();
      if (threadIdx.x == 0 && senders > 1) {
        arrived += senders;
        atomicAdd(local + kBarrier, 1u);
        uint32_t value;
        do {
          asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];"
                       : "=r"(value) : "l"(local + kBarrier) : "memory");
        } while (value < arrived);
      }
#ifndef CHOP_COMM_PHASED_TURNS_BF16_TEST_UNTURNED
      const uint32_t dest = (rank + offset(phase, ranks)) % ranks;
      const uint32_t next = (dest + ranks - offset(phase + 1, ranks)) % ranks;
      if (threadIdx.x == 0 && s == 0 && next != rank)
        phased_f32::release_system(peer(flags, next) + kTurns + phase + 1, mark);
      const uint32_t previous =
          (rank + offset(phase + 1, ranks) + ranks - offset(phase, ranks)) % ranks;
      if (threadIdx.x == 0 && previous != rank)
        phased_f32::wait_system(local + kTurns + phase + 1, mark);
#endif
      __syncthreads();
    };
    for (uint32_t phase = 1; phase < ranks; ++phase) {
      const uint32_t dest = (rank + phase) % ranks;
      const Span span = owner_span(count, dest, ranks);
#ifdef CHOP_COMM_PHASED_TURNS_BF16_TEST_LATE_SCATTER
      if (rank == CHOP_COMM_PHASED_TURNS_BF16_TEST_LATE_SCATTER && phase == ranks - 1)
        late_scatter();
#endif
      for (uint32_t block = s; block < blocks; block += senders) {
        const Span part = block_of(span.count, block, blocks);
        if (part.count == 0) continue;
#ifdef CHOP_COMM_PHASED_TURNS_BF16_TEST_LATE_WAIT
        if (!(early && phase == 1))
#endif
        phased_bf16::send_rounded(peer(inboxes, dest) + kOneShotRegion + rank * slot + part.first,
                                  input + span.first + part.first, part.count);
        __syncthreads();
        if (threadIdx.x == 0)
          phased_f32::release_system(
              peer(flags, dest) + kScatterFlags + rank * kMaxBlocks + block, mark);
      }
      next_phase(phase);
    }
    if (workers == 0)
      for (uint32_t block = s; block < blocks; block += senders) reduce(block);
    for (uint32_t step = 1; step < ranks; ++step) {
      const uint32_t phase = ranks - 1 + step;
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
      if (phase < phases) next_phase(phase);
    }
    if (workers == 0) wait_gathered(s, senders);
  } else {
    const uint32_t worker = blockIdx.x - senders;
    for (uint32_t block = worker; block < blocks; block += workers) reduce(block);
    wait_gathered(worker, workers);
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

} // namespace chop::comm::phased_turns_bf16
