#pragma once

// Rank-ordered F32 all-reduce of large row sets in one launch: a
// reduce-scatter, then an all-gather, each sending to one destination at a
// time (rank + 1, rank + 2, ...), as the phased pair in push.cuh does, but
// from `senders` CTAs instead of one 8-CTA cluster.
//
// Owner q holds granules [q G / ranks, (q + 1) G / ranks) of the call's
// G = count / kGranule granules. Each owner span splits into the call's
// `blocks` (blocks_of: about kBlockGranules granules each, at least one per
// sender, at most kMaxBlocks); sending CTA s moves blocks s, s + senders,
// ... of the span of the phase's destination and releases one flag per
// block, so a receiver sums a block as soon as all its sources arrived,
// while later blocks are in flight. Worker CTAs (every CTA after the
// senders) take blocks w, w + workers, ... and sum them in rank order from
// F32 zero, as phased::sum_range does, and write the sum (F32), or with
// Out = uint16_t
// the sum rounded once to BF16 (cvt.rn.bf16.f32, the round row's
// rounding), into the owner's span of `output`. The gather sends those
// blocks from `output` into every peer's `output`: the BF16 variant sends
// half the bytes of the F32 one.
//
// Between destinations the sending CTAs meet at a local counter barrier, so
// a GPU writes to one peer at a time (a GPU that writes to two or more peers
// at once collapses to about 2 GB/s, README "Measured PCIe behaviour"). The
// barrier orders nothing the receivers rely on: every block carries its
// own release flag.
//
// With `live_rows`, a call reduces only the first *live_rows rows of
// `row_elements` each (every rank reads the same count) and writes nothing
// past them; null reduces all `elements`. A call with nothing to reduce
// returns without advancing the epoch, on every rank alike.
//
// Flags of one rank's buffer (u32 words, kFlagWords): scatter flags
// [source][block], gather flags [source][block] and the one-shot's flags
// [parity][source][block] (see allreduce), which peers release with the
// call's tagged epoch and receivers compare for equality, then the local
// reduced flags [block], the sender barrier and the arrival counter.
// The last CTA to arrive clears the barrier and counter and advances
// epoch[0]. No acknowledgments are needed: a rank sends call k + 1's data
// only after its call k completed, which needed every peer's gathered
// blocks, each sent after the peer's reads of its inbox and flags for that
// block, and after this rank's stream ran every reader of call k's output.

#include <stdint.h>
#include "../common/convert.cuh"
#include "../common/ptx.cuh"

#ifndef CHOP_COMM_MAX_RANKS
#error "CHOP_COMM_MAX_RANKS is required"
#endif

// Self-contained (comm.cuh brings push.cuh and its tuning defines), so its
// own helpers repeat comm.cuh's.
namespace chop::comm::phased_f32 {

constexpr uint32_t kMaxRanks = CHOP_COMM_MAX_RANKS;
constexpr uint32_t kThreads = 256;
constexpr uint64_t kGranule = 1024;  // elements; blocks are whole granules
constexpr uint32_t kMaxSenders = 64;
constexpr uint32_t kBlockGranules = 64;  // 256 KiB of F32 per block
constexpr uint32_t kMaxBlocks = 256;
// The one-shot's largest call (256 granules) and its inbox region,
// [parity][source][kOneShotMaxElements], ahead of the two-shot's slots.
constexpr uint64_t kOneShotMaxElements = 256 * kGranule;
constexpr uint64_t kOneShotRegion = 2 * kMaxRanks * kOneShotMaxElements;
constexpr uint32_t kScatterFlags = 0;                                    // [source][block]
constexpr uint32_t kGatherFlags = kScatterFlags + kMaxRanks * kMaxBlocks;   // [source][block]
constexpr uint32_t kOneShotFlags = kGatherFlags + kMaxRanks * kMaxBlocks;   // [parity][source][block]
constexpr uint32_t kReduced = kOneShotFlags + 2 * kMaxRanks * kMaxBlocks;   // [block], local
constexpr uint32_t kBarrier = kReduced + kMaxBlocks;                        // local
constexpr uint32_t kDone = kBarrier + 1;                                    // local
constexpr uint32_t kFlagWords = kDone + 1;

struct Span { uint64_t first, count; };
struct alignas(16) Vector128 { uint32_t x, y, z, w; };

// Slot `rank` of a by-value pointer array of kMaxRanks slots.
template <typename Pointers>
__device__ __forceinline__ auto peer(Pointers pointers, uint32_t rank) {
  static_assert(kMaxRanks == 8, "peer names eight slots");
  static_assert(sizeof(Pointers) == kMaxRanks * sizeof(pointers.slot_0), "eight slots");
  switch (rank) {
    case 0: return pointers.slot_0;
    case 1: return pointers.slot_1;
    case 2: return pointers.slot_2;
    case 3: return pointers.slot_3;
    case 4: return pointers.slot_4;
    case 5: return pointers.slot_5;
    case 6: return pointers.slot_6;
    default: return pointers.slot_7;
  }
}

__device__ __forceinline__ uint32_t acquire_system(const unsigned int* address) {
  uint32_t value;
  asm volatile("ld.global.acquire.sys.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
  return value;
}

__device__ __forceinline__ void release_system(unsigned int* address, uint32_t value) {
  asm volatile("st.global.release.sys.u32 [%0], %1;" :: "l"(address), "r"(value) : "memory");
}

__device__ __forceinline__ uint32_t acquire_gpu(const unsigned int* address) {
  uint32_t value;
  asm volatile("ld.global.acquire.gpu.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
  return value;
}

__device__ __forceinline__ void release_gpu(unsigned int* address, uint32_t value) {
  asm volatile("st.global.release.gpu.u32 [%0], %1;" :: "l"(address), "r"(value) : "memory");
}

__device__ __forceinline__ uint32_t tagged(uint32_t epoch) { return 0x80000000u | epoch; }

__device__ __forceinline__ Span owner_span(uint64_t count, uint32_t owner, uint32_t ranks) {
  const uint64_t granules = count / kGranule;
  return {owner * granules / ranks * kGranule,
          ((owner + 1) * granules / ranks - owner * granules / ranks) * kGranule};
}

// Elements of one source's inbox slot: the largest owner span.
__device__ __forceinline__ uint64_t slot_elements(uint64_t count, uint32_t ranks) {
  return (count / kGranule + ranks - 1) / ranks * kGranule;
}

// Blocks per owner span of a call: the same on every rank, from the
// smallest span.
__device__ __forceinline__ uint32_t blocks_of(uint64_t count, uint32_t ranks, uint32_t senders) {
  const uint64_t blocks = count / kGranule / ranks / kBlockGranules;
  const uint64_t at_least = blocks > senders ? blocks : senders;
  return static_cast<uint32_t>(at_least < kMaxBlocks ? at_least : kMaxBlocks);
}

// Owner-relative range of block b of a span of `count` elements.
__device__ __forceinline__ Span block_of(uint64_t count, uint32_t block, uint32_t blocks) {
  const uint64_t granules = count / kGranule;
  return {block * granules / blocks * kGranule,
          ((block + 1) * granules / blocks - block * granules / blocks) * kGranule};
}

__device__ __forceinline__ void wait_system(const unsigned int* address, uint32_t mark) {
  uint32_t value;
  do {
    asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
  } while (value != mark);
  acquire_system(address);
}

__device__ __forceinline__ void wait_gpu(const unsigned int* address, uint32_t mark) {
  uint32_t value;
  do {
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
  } while (value != mark);
  acquire_gpu(address);
}

// One CTA copies `bytes` (a multiple of 16) with four 16-byte loads in
// flight per thread.
__device__ __forceinline__ void copy_bytes(void* to, const void* from, uint64_t bytes) {
  constexpr uint32_t kUnroll = 4;
  auto* target = reinterpret_cast<Vector128*>(to);
  const auto* source = reinterpret_cast<const Vector128*>(from);
  const uint64_t vectors = bytes / 16;
  uint64_t v = threadIdx.x;
  for (; v + (kUnroll - 1) * kThreads < vectors; v += kUnroll * kThreads) {
    Vector128 values[kUnroll];
#pragma unroll
    for (uint32_t i = 0; i < kUnroll; ++i) values[i] = source[v + i * kThreads];
#pragma unroll
    for (uint32_t i = 0; i < kUnroll; ++i) target[v + i * kThreads] = values[i];
  }
  for (; v < vectors; v += kThreads) target[v] = source[v];
}

template <typename Out>
__device__ __forceinline__ Out round_to(float value) {
  if constexpr (sizeof(Out) == 2) return convert::bf16_from_f32(value);
  else return value;
}

// Rank-ordered sums of [first, first + count) of the owner's span, four
// elements per thread step: ((0 + x_0) + x_1) + ... in F32.
template <typename Out>
__device__ __forceinline__ void sum_block(const float* input, const float* inbox, Out* output,
                                          uint64_t slot, uint64_t first, uint64_t count,
                                          uint32_t rank, uint32_t ranks) {
  for (uint64_t v = threadIdx.x; v < count / 4; v += kThreads) {
    const uint64_t at = first / 4 + v;
    Vector128 parts[kMaxRanks];
#pragma unroll
    for (uint32_t source = 0; source < kMaxRanks; ++source)
      if (source < ranks)
        parts[source] = reinterpret_cast<const Vector128*>(
            source == rank ? input : inbox + source * slot)[at];
    float sums[4] = {};
#ifdef CHOP_COMM_PHASED_F32_TEST_REVERSED
    // Lab mutant: sources in descending order.
#pragma unroll
    for (int source = kMaxRanks - 1; source >= 0; --source) {
      if (source >= static_cast<int>(ranks)) continue;
      const float* values = reinterpret_cast<const float*>(&parts[source]);
#pragma unroll
      for (uint32_t k = 0; k < 4; ++k) sums[k] += values[k];
    }
#else
#pragma unroll
    for (uint32_t source = 0; source < kMaxRanks; ++source) {
      if (source >= ranks) break;
      const float* values = reinterpret_cast<const float*>(&parts[source]);
#pragma unroll
      for (uint32_t k = 0; k < 4; ++k) sums[k] += values[k];
    }
#endif
    Out out[4];
#pragma unroll
    for (uint32_t k = 0; k < 4; ++k) out[k] = round_to<Out>(sums[k]);
    if constexpr (sizeof(Out) == 2) {
      reinterpret_cast<uint2*>(output)[at] =
          make_uint2(uint32_t(out[0]) | (uint32_t(out[1]) << 16),
                     uint32_t(out[2]) | (uint32_t(out[3]) << 16));
    } else {
      reinterpret_cast<float4*>(output)[at] = make_float4(out[0], out[1], out[2], out[3]);
    }
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

// The call's geometry: the span each destination receives and this rank's
// own span (two-shot: owner spans; one-shot: every element), the blocks
// per span, and where source `rank`'s data lands in a receiver's inbox.
template <bool OneShot>
struct Geometry {
  uint64_t count;
  uint32_t rank, ranks, senders, blocks, parity;
  uint64_t slot;  // elements between sources' slots

  __device__ Geometry(uint64_t count, uint32_t rank, uint32_t ranks, uint32_t senders,
                      uint32_t epoch_value)
      : count(count), rank(rank), ranks(ranks), senders(senders),
        blocks(blocks_of(count, OneShot ? 1 : ranks, senders)), parity(epoch_value & 1),
        slot(OneShot ? kOneShotMaxElements : slot_elements(count, ranks)) {}

  __device__ Span span(uint32_t owner) const {
    return OneShot ? Span{0, count} : owner_span(count, owner, ranks);
  }
  // First element of the receiver's slots (source 0's slot).
  __device__ uint64_t region() const {
#ifdef CHOP_COMM_PHASED_F32_TEST_SHARED_PARITY
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

// Two-shot (OneShot false): reduce-scatter then all-gather, as above.
// One-shot: every rank pushes its whole partial to each peer in turn, one
// destination at a time, and every rank sums every element itself, with no
// gather: fewer phases for small calls, (ranks - 1) times the payload sent.
// Its inbox slots and flags alternate by epoch parity in their own region
// (kOneShotRegion elements at the inbox start, kOneShotFlags words): a rank
// writes a slot again two calls later, after its intervening call needed
// the peer's data of that call, which the peer sends only after its stream
// finished the call that read the slot. The two-shot's regions follow, so
// calls of both protocols can share one state in any order.
//
// With no worker CTAs (gridDim.x == senders), each sending CTA sums its
// own blocks after the scatter and, in the two-shot, gathers them without a
// handoff: fewer steps for small calls.
template <typename Out, bool OneShot, typename Inboxes, typename Outputs, typename Flags>
__device__ void allreduce(const float* input, Out* output, Inboxes inboxes, Outputs outputs,
                          Flags flags, unsigned int* epoch, uint64_t elements,
                          uint32_t row_elements, const uint32_t* live_rows, uint32_t rank,
                          uint32_t ranks, uint32_t senders) {
  ptx::griddep_launch_dependents();
  check<OneShot>(elements, row_elements, live_rows, rank, ranks, senders);
#ifdef CHOP_COMM_PHASED_F32_TEST_LATE_WAIT
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
        copy_bytes(peer(inboxes, dest) + g.region() + rank * g.slot + part.first,
                   input + span.first + part.first, part.count * 4);
    }
  }
#endif
  ptx::griddep_wait();
  const uint64_t count =
      live_rows ? min(elements, uint64_t(*live_rows) * row_elements) : elements;
  if (count == 0) return;
  const uint32_t epoch_value = epoch[0] + 1;
  const uint32_t mark = tagged(epoch_value);
  const Geometry<OneShot> g(count, rank, ranks, senders, epoch_value);
  const uint32_t blocks = g.blocks;
  unsigned int* local = peer(flags, rank);
  const Span own = g.span(rank);
  const uint32_t workers = gridDim.x - senders;
  const bool sender = blockIdx.x < senders;
  // Sums block `block` of the own span once every source's block arrived.
  auto reduce = [&](uint32_t block) {
    const Span part = block_of(own.count, block, blocks);
    if (part.count == 0) return;
#ifndef CHOP_COMM_PHASED_F32_TEST_NO_SCATTER_WAIT
    if (threadIdx.x < ranks && threadIdx.x != rank)
      wait_system(local + g.scatter_flag(threadIdx.x, block), mark);
#endif
    __syncthreads();
    sum_block<Out>(input + own.first, peer(inboxes, rank) + g.region(), output + own.first,
                   g.slot, part.first, part.count, rank, ranks);
    __syncthreads();
    if (!OneShot && workers && threadIdx.x == 0) release_gpu(local + kReduced + block, mark);
  };
  // The call completes only with every peer's gathered blocks in `output`.
  auto wait_gathered = [&](uint32_t first, uint32_t step) {
#ifndef CHOP_COMM_PHASED_F32_TEST_NO_GATHER_WAIT
    for (uint32_t block = first; block < blocks; block += step) {
      const uint32_t source = threadIdx.x;
      if (source < ranks && source != rank &&
          block_of(owner_span(count, source, ranks).count, block, blocks).count)
        wait_system(local + kGatherFlags + source * kMaxBlocks + block, mark);
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
#ifdef CHOP_COMM_PHASED_F32_TEST_LATE_WAIT
        if (!(early && step == 1))
#endif
        copy_bytes(peer(inboxes, dest) + g.region() + rank * g.slot + part.first,
                   input + span.first + part.first, part.count * 4);
        __syncthreads();
        if (threadIdx.x == 0)
          release_system(peer(flags, dest) + g.scatter_flag(rank, block), mark);
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
            if (threadIdx.x == 0) wait_gpu(local + kReduced + block, mark);
            __syncthreads();
          }
          copy_bytes(peer(outputs, dest) + own.first + part.first,
                     output + own.first + part.first, part.count * sizeof(Out));
          __syncthreads();
          if (threadIdx.x == 0)
            release_system(peer(flags, dest) + kGatherFlags + rank * kMaxBlocks + block, mark);
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

} // namespace chop::comm::phased_f32
