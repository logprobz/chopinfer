#pragma once

// Lab probes for the phased group gather's race cases
// (tests/context_parallel_phased_races.rs): a delay a late sender runs
// before its transfer, a dependent consumer that snapshots a receive buffer
// on the device (after an optional hold, a late consumer of the previous
// generation), and lab-only mutants of the served
// `chop::context_parallel::phased::gather`.
//
// `gather<ReceiveWait, AcknowledgementWait>` is the served body line for
// line; the two lines marked `// mutation switch` put its incoming-data wait
// and its reuse acknowledgement wait behind switches. The test compares the
// two bodies' text with those lines removed, so the mutants stay the served
// code minus one wait. One difference is outside the body: in this
// namespace `wait_system` (which the body calls unqualified, so it hides the
// served helper) gives up after `kWaitBoundNs` and counts a timeout in
// `timeouts`. A mutant without a wait can deadlock (a receiver that does not
// wait for its sources can run a call ahead and overwrite the
// acknowledgement a late sender still waits for); bounded, it ends, and the
// test reports the timeouts as a deadlock outcome of their own, never as a
// detection.

#include <stdint.h>
#include "common/ptx_control.cuh"
#include "context_parallel/phased_gather.cuh"

namespace chop::probe::phased_races {

using namespace chop::context_parallel::phased;

__device__ __forceinline__ uint64_t now_ns() {
  uint64_t now;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
  return now;
}

// Waits the lab entries gave up on since `take_timeouts` last ran.
__device__ unsigned int timeouts;

// How long a lab entry's wait spins before it counts a timeout and goes on:
// far beyond every legitimate wait of the race cases (a 3 ms delay, a 5 ms
// hold, copies of a few milliseconds).
constexpr uint64_t kWaitBoundNs = 100000000;

// The served `wait_system` (relaxed spin, then one acquire), bounded.
__device__ __forceinline__ void wait_system(const unsigned int* address, uint32_t mark) {
  const uint64_t start = now_ns();
  uint32_t value;
  do {
    asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
    if (value == mark) {
      asm volatile("ld.global.acquire.sys.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
      return;
    }
  } while (now_ns() - start < kWaitBoundNs);
  atomicAdd(&timeouts, 1u);
}

// Moves the timeout count into out[0] and clears it. One thread.
__device__ void take_timeouts(unsigned int* out) {
  if (blockIdx.x == 0 && threadIdx.x == 0) out[0] = atomicExch(&timeouts, 0u);
}

// One thread spins `ns` nanoseconds. Launched without PDL and triggering no
// dependents, so the stream's next kernel starts after it.
__device__ void delay(uint64_t ns) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const uint64_t start = now_ns();
  while (now_ns() - start < ns) {}
}

// A dependent consumer of a transfer: after its dependency wait (and, with
// `hold_ns`, a hold in every CTA before it reads), copies `words` words of
// `source` into `target`.
__device__ void snapshot(const uint32_t* source, uint32_t* target, uint64_t words,
                         uint64_t hold_ns) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  if (hold_ns != 0) {
    if (threadIdx.x == 0) {
      const uint64_t start = now_ns();
      while (now_ns() - start < hold_ns) {}
    }
    __syncthreads();
  }
  const uint64_t step = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < words;
       i += step)
    target[i] = reinterpret_cast<const volatile uint32_t*>(source)[i];
}

// Grid [kCtas] as one cluster of kCtas, kThreads threads.
template <bool ReceiveWait, bool AcknowledgementWait, typename Outputs, typename Flags>
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
      if constexpr (AcknowledgementWait)  // mutation switch
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
    if constexpr (ReceiveWait)  // mutation switch
    for (uint32_t source = 0; source < ranks; ++source)
      if (source != rank) wait_system(local + source, tagged(value));
    epoch[0] = value;
  }
}

}  // namespace chop::probe::phased_races
