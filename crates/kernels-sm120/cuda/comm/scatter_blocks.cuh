#pragma once
#include "comm/push.cuh"

// Split only the last destination into tile-aligned flagged blocks. The
// existing gather consumes the same owner spans, flags and epoch. Each
// element still uses phased::sum_range's ascending rank-order F32 sum.
namespace chop::comm::scatter_blocks {
using namespace phased;

template <uint32_t Blocks>
__device__ __forceinline__ uint32_t blocks_of(uint64_t count) {
  const uint64_t tiles = (count + kTile - 1) / kTile;
  return static_cast<uint32_t>(max(uint64_t{1}, min(uint64_t{Blocks}, tiles)));
}

template <uint32_t Blocks>
__device__ __forceinline__ Span block_of(uint64_t count, uint32_t block) {
  const uint64_t tiles = (count + kTile - 1) / kTile;
  const uint32_t blocks = blocks_of<Blocks>(count);
  const uint64_t first = min(block * tiles / blocks * kTile, count);
  const uint64_t end = min((block + 1) * tiles / blocks * kTile, count);
  return {first, end - first};
}

template <uint32_t Blocks, typename Inboxes, typename Flags>
__device__ void run(const float* input, float* output, Inboxes inboxes, Flags flags,
                               const unsigned int* epoch, uint64_t elements,
                               uint32_t rank, uint32_t ranks) {
  static_assert(Blocks > 0 && Blocks <= kCluster, "one last-phase block per sender at most");
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  if (!valid_geometry(rank, ranks) || elements == 0) return;
  check_grid();
  const uint32_t mark = push::tagged(epoch[0] + 1);
  const uint64_t slot = slot_elements(elements, ranks);
  unsigned int* local = peer(flags, rank);
  if (blockIdx.x < kCluster) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      local[kReady] = 0;
      local[kDone] = 0;
    }
    for (uint32_t step = 1; step < ranks; ++step) {
      const uint32_t dest = (rank + step) % ranks;
      const Span span = owner_span(elements, dest, ranks);
      float* target = push::region<float>(inboxes, dest, push::kPhasedBase) + rank * slot;
      if (step + 1 < ranks) {
        copy(target, input + span.first, span.count, blockIdx.x, kCluster);
        push::next_destination();
        if (blockIdx.x == 0 && threadIdx.x == 0)
          release_system(peer(flags, dest) + kScatterFlags + rank, mark);
        continue;
      }
      for (uint32_t block = blockIdx.x; block < blocks_of<Blocks>(span.count); block += kCluster) {
        const Span part = block_of<Blocks>(span.count, block);
        copy(target + part.first, input + span.first + part.first, part.count, 0, 1);
        __syncthreads();
        if (threadIdx.x == 0) release_system(peer(flags, dest) + kBlockFlags + block, mark);
      }
    }
    return;
  }
  const uint32_t worker = blockIdx.x - kCluster;
  const uint32_t workers = gridDim.x - kCluster;
  const uint32_t last = (rank + 1) % ranks;   // source of the last phase
  if (threadIdx.x < ranks && threadIdx.x != rank && threadIdx.x != last)
    push::wait_system(local + kScatterFlags + threadIdx.x, mark);
  __syncthreads();
  const Span span = owner_span(elements, rank, ranks);
  const float* inbox = push::region<const float>(inboxes, rank, push::kPhasedBase);
  for (uint32_t block = 0; block < blocks_of<Blocks>(span.count); ++block) {
    if (ranks > 1) {
      if (threadIdx.x == 0) push::wait_system(local + kBlockFlags + block, mark);
      __syncthreads();
    }
    const Span part = block_of<Blocks>(span.count, block);
    sum_range<float>(input + span.first, inbox, output + span.first,
                    nullptr, slot, part.first,
                    part.count, rank, ranks, worker, workers);
  }
}

}  // namespace chop::comm::scatter_blocks
