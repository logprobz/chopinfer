#pragma once

// The phased group gather (`chop_comm_group_phased_f32_v1`): every member of
// a context-parallel group sends one block of 32-bit words to every member,
// one destination at a time. See README.md "Phased group gather".
//
// On four RTX PRO 6000 GPUs a GPU that writes to two or more peers at once
// collapses to about 2 GB/s in total, while one destination at a time runs
// at 42-47 GB/s (cuda/comm/README.md). The shared all-gather
// (`chop_comm_allgather_f32_v1`) runs one CTA per 4096-word tile, each
// looping over every destination, so a group of four writes to all three
// peers at once. Here one cluster of kCtas CTAs sends to destination
// (rank + p) % ranks in phase p and synchronizes before the next, so in
// every phase each GPU sends to one peer and receives from one.
//
// Member `rank` sends `input + d * input_stride` (`words` words) to group
// rank d, where it lands at `outputs[d] + rank * words`: with input_stride 0
// the all-gather of one shard (every output holds the group's shards
// rank-major), with input_stride = words the exchange of a destination-major
// buffer (owner d receives block d of every member, source-major). The copy
// moves bits; nothing is converted.
//
// Protocol. `flags` (one pointer per member) holds 2 * ranks words per member,
// starting zeroed: data flags [source], then acknowledgements [ranks +
// receiver]; `epoch` one word. On entering call e every member stores
// tagged(e - 1) into its acknowledgement slot at every other member; a sender
// writes to d only after reading d's tagged(e - 1), so it never overwrites
// an output d's earlier work (stream-ordered before this call) may still read.
// The first acknowledgement, tagged(0), is not the zero the flags start
// with, so even a state's first call waits for its receiver. After its
// blocks to d, a sender stores tagged(e) into d's data flag [rank]; a member
// returns once every source's flag reads tagged(e). Only CTA 0's thread 0
// waits, flags and advances the epoch; the cluster barrier between phases
// orders every CTA's stores before the release.

#include <stdint.h>
#include "common/ptx_control.cuh"

namespace chop::context_parallel::phased {

// One cluster sends; at most eight CTAs form a portable cluster.
constexpr uint32_t kCtas = 8;
constexpr uint32_t kThreads = 256;
constexpr uint32_t kMaxRanks = 8;

struct alignas(16) Vector128 {
  uint32_t x, y, z, w;
};

__device__ __forceinline__ uint32_t tagged(uint32_t epoch) { return 0x80000000u | epoch; }

__device__ __forceinline__ void release_system(unsigned int* address, uint32_t value) {
  asm volatile("st.global.release.sys.u32 [%0], %1;" ::"l"(address), "r"(value) : "memory");
}

// Spins with relaxed loads, then acquires once.
__device__ __forceinline__ void wait_system(const unsigned int* address, uint32_t mark) {
  uint32_t value;
  do {
    asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
  } while (value != mark);
  asm volatile("ld.global.acquire.sys.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
}

__device__ __forceinline__ void cluster_sync() {
  ptx::cluster_arrive_release();
  ptx::cluster_wait_acquire();
}

// Slot `rank` of a by-value pointer array of kMaxRanks slots.
template <typename Pointers>
__device__ __forceinline__ auto slot(Pointers pointers, uint32_t rank) {
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

// This CTA's share of `words` words, interleaved over the cluster: 16-byte
// vectors with four loads in flight per thread when both ends allow them.
__device__ __forceinline__ void copy(uint32_t* to, const uint32_t* from, uint64_t words) {
  const uint64_t step = static_cast<uint64_t>(kCtas) * kThreads;
  const uint64_t first = static_cast<uint64_t>(blockIdx.x) * kThreads + threadIdx.x;
  const bool vectors = words % 4 == 0 &&
                       ((reinterpret_cast<uintptr_t>(to) | reinterpret_cast<uintptr_t>(from)) & 15) == 0;
  if (!vectors) {
    for (uint64_t i = first; i < words; i += step) to[i] = from[i];
    return;
  }
  auto* out = reinterpret_cast<Vector128*>(to);
  const auto* in = reinterpret_cast<const Vector128*>(from);
  const uint64_t count = words / 4;
  constexpr uint32_t kUnroll = 4;
  uint64_t v = first;
  for (; v + (kUnroll - 1) * step < count; v += kUnroll * step) {
    Vector128 values[kUnroll];
#pragma unroll
    for (uint32_t i = 0; i < kUnroll; ++i) values[i] = in[v + i * step];
#pragma unroll
    for (uint32_t i = 0; i < kUnroll; ++i) out[v + i * step] = values[i];
  }
  for (; v < count; v += step) out[v] = in[v];
}

// Grid [kCtas] as one cluster of kCtas, kThreads threads.
template <typename Outputs, typename Flags>
__device__ void gather(const uint32_t* input, uint64_t input_stride, Outputs outputs, Flags flags,
                       unsigned int* epoch, uint64_t words, uint32_t rank, uint32_t ranks) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  if (rank >= ranks || ranks == 0 || ranks > kMaxRanks || words == 0) return;
  if (blockDim.x != kThreads || gridDim.x != kCtas || ptx::cluster_nctarank() != kCtas) __trap();
  const bool leader = blockIdx.x == 0 && threadIdx.x == 0;
  const uint32_t value = leader ? epoch[0] + 1 : 0;
  unsigned int* local = slot(flags, rank);
  if (leader)
    for (uint32_t source = 0; source < ranks; ++source)
      if (source != rank) release_system(slot(flags, source) + ranks + rank, tagged(value - 1));
  for (uint32_t phase = 0; phase < ranks; ++phase) {
    const uint32_t destination = (rank + phase) % ranks;
    if (destination != rank) {
      if (leader) wait_system(local + ranks + destination, tagged(value - 1));
      cluster_sync();
    }
    copy(reinterpret_cast<uint32_t*>(slot(outputs, destination)) + rank * words,
         input + destination * input_stride, words);
    __syncthreads();
    cluster_sync();
    if (destination != rank && leader)
      release_system(slot(flags, destination) + rank, tagged(value));
  }
  if (leader) {
    for (uint32_t source = 0; source < ranks; ++source)
      if (source != rank) wait_system(local + source, tagged(value));
    epoch[0] = value;
  }
}

}  // namespace chop::context_parallel::phased
