#pragma once
#include "mix_split.cuh"
#include "../common/ptx_control.cuh"

// Probe-only DSMEM traffic and exact ordered reduction. No projection or
// serving selection is implied by these transport latency measurements.
namespace chop::sinkhorn_hc::cluster_probe {
__device__ __forceinline__ void barrier() {
  ptx::cluster_arrive_release();
  ptx::cluster_wait_acquire();
}
__device__ __forceinline__ uint32_t remote_load(const uint32_t* local, uint32_t rank) {
  const uint32_t address = ptx::cluster_map_shared(local, rank);
  uint32_t value;
  asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(value) : "r"(address) : "memory");
  return value;
}
__device__ __forceinline__ uint64_t timer() {
  uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}
// Stamps are per CTA: SM, rank, cluster size, four local clock64 and four
// globaltimer stamps. Never subtract clock64 values from different CTAs.
template<class M, int Kind>
__device__ void task(const uint32_t* input, uint32_t* output, uint64_t* stamps,
    uint32_t words, uint32_t rounds, uint32_t rank, uint32_t row, uint32_t blocks,
    unsigned char* arena) {
  constexpr int Payload = M::kMixes * M::kChains + kThreads;
  constexpr int Padded = M::kMixes * (M::kChains + 1) + kThreads;
  const int t = threadIdx.x;
  const int shard = words / blocks;
  auto* packet = reinterpret_cast<uint32_t*>(arena);
  auto* gather = packet + Payload;
  auto* warp_sums = reinterpret_cast<float*>(gather + Padded);
  auto* result = reinterpret_cast<uint32_t*>(warp_sums + kWarps);
  uint64_t c[4], ns[4];
  // Broadcast needs a complete leader payload; gather uses disjoint shards.
  const int fill = Kind == 2 ? words : shard;
  for (int i = t; i < fill; i += kThreads) {
    int source = Kind == 2 ? i : rank * shard + i;
    if constexpr (Kind >= 3) {
      const int local_chains = M::kChains / blocks;
      source = i < M::kMixes * local_chains
          ? (i / local_chains) * M::kChains + rank * local_chains + i % local_chains
          : M::kMixes * M::kChains + rank * (kThreads / blocks) + i - M::kMixes * local_chains;
    }
    packet[i] = input[uint64_t(row) * Payload + source];
  }
  if (t < 32) result[t] = 0xa5a5a5a5;
  __syncthreads();
  if (t == 0) { c[0] = clock64(); ns[0] = timer(); }
  barrier();
  if (t == 0) { c[1] = clock64(); ns[1] = timer(); }
  if constexpr (Kind == 0) {
    for (uint32_t iteration = 0; iteration < rounds; ++iteration) barrier();
  } else if constexpr (Kind == 1 || Kind == 2) {
    if (Kind == 2 || rank == 0) {
      for (uint32_t iteration = 0; iteration < rounds; ++iteration) {
        for (int i = t; i < words; i += kThreads) {
          const int owner = Kind == 2 ? 0 : i / shard;
          const int offset = Kind == 2 ? i : i % shard;
          reinterpret_cast<volatile uint32_t*>(gather)[i] = remote_load(packet + offset, owner);
        }
        __syncthreads();
      }
    }
  } else {
    // Gather a complete [mix, K-chain] table and RMS lanes on the leader.
    // Padding retains the reference's distinct shared banks for each mix.
    if (rank == 0) {
      for (int i = t; i < Payload; i += kThreads) {
        const int dst = i < M::kMixes * M::kChains
            ? (i / M::kChains) * (M::kChains + 1) + i % M::kChains
            : M::kMixes * (M::kChains + 1) + i - M::kMixes * M::kChains;
        const int local_chains = M::kChains / blocks;
        const int owner = i < M::kMixes * M::kChains
            ? (i % M::kChains) / local_chains : (i - M::kMixes * M::kChains) / (kThreads / blocks);
        const int offset = i < M::kMixes * M::kChains
            ? (i / M::kChains) * local_chains + i % local_chains
            : M::kMixes * local_chains + (i - M::kMixes * M::kChains) % (kThreads / blocks);
        gather[dst] = remote_load(packet + offset, owner);
      }
      __syncthreads();
      if (t < M::kMixes) {
        float acc = 0.f;
        #pragma unroll 8
        for (int step = 0; step < M::kChains; ++step) {
          const int chain = Kind == 4 ? M::kChains - 1 - step : step;
          acc = __fadd_rn(acc, __uint_as_float(gather[t * (M::kChains + 1) + chain]));
        }
        result[t] = __float_as_uint(acc);
      }
      float square = __uint_as_float(gather[M::kMixes * (M::kChains + 1) + t]);
      square = warp_sum(square);
      if ((t & 31) == 0) warp_sums[t >> 5] = square;
      __syncthreads();
      if (t == 0) {
        float total = 0.f;
        #pragma unroll
        for (int w = 0; w < kWarps; ++w) total = __fadd_rn(total, warp_sums[w]);
        result[M::kMixes] = __float_as_uint(__frsqrt_rn(__fadd_rn(
            __fdiv_rn(total, float(M::kFlat)), __uint_as_float(M::kNormEpsBits))));
      }
      __syncthreads();
    }
  }
  if (t == 0) { c[2] = clock64(); ns[2] = timer(); }
  // Every peer stays alive through all remote reads. Ordered results are
  // published before peers broadcast them and before any CTA can exit.
  barrier();
  if constexpr (Kind >= 3) {
    if (t < 32) output[(uint64_t(row) * blocks + rank) * Payload + t] = remote_load(result + t, 0);
    barrier();
  }
  if (t == 0) { c[3] = clock64(); ns[3] = timer(); }
  if constexpr (Kind == 0) {
    if (t == 0) output[(uint64_t(row) * blocks + rank) * Payload] = packet[0];
  } else if constexpr (Kind == 1 || Kind == 2) {
    if (Kind == 2 || rank == 0)
      for (int i = t; i < words; i += kThreads)
        output[(uint64_t(row) * blocks + rank) * Payload + i] = gather[i];
  }
  if (t == 0) {
    uint32_t sm;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    const uint64_t at = (uint64_t(row) * blocks + rank) * 11;
    stamps[at] = sm; stamps[at+1] = rank; stamps[at+2] = blocks;
    #pragma unroll
    for (int i = 0; i < 4; ++i) { stamps[at+3+i] = c[i]; stamps[at+7+i] = ns[i]; }
  }
}
} // namespace chop::sinkhorn_hc::cluster_probe
