#pragma once
#include "ceil_cached.cuh"
#include "../common/ptx_control.cuh"

// Cluster form of block_quantize_cached::run for decode rows: each row's
// groups are split over a cluster of Parts CTAs (grid rows x Parts, cluster
// Parts), CTA p of a row's cluster quantizing groups [p G / Parts,
// (p + 1) G / Parts) with the same quantize_group, so every code and scale
// is the cached entry's. A one-CTA row runs its groups as eight-per-warp
// chains on one SM; split, each warp runs one or two.
//
// The row status protocol is the cached entry's single atomic OR. Each CTA
// reads the row's prior status before the cluster barrier (no other grid
// writes the row's status while this one runs, as the cached entry's
// atomic OR returning prior | flags already assumes), ORs its groups'
// flags, and publishes them in shared memory; after the barrier every CTA
// ORs every part's flags through distributed shared memory, so each part
// decides from prior | flags, and a flagged row stores the default codes
// and the unit scale in every part. Part 0 alone ORs the row's flags into
// the status, which ends as prior | flags. Every CTA arrives again after
// its reads and waits for the cluster before it exits, so no part's shared
// memory is read after it leaves. Every part of a row's cluster sees the
// same live count and leaves together.
namespace chop::row::block_quantize_split {
using namespace block_quantize;

template<int Width, int Format, bool F32Input, bool Fake, uint32_t FloorBits,
         bool EarlyRelease, int Parts>
__device__ void run(const void* input, void* output, uint8_t* scales, const uint32_t* live_rows,
                    uint32_t* status, uint64_t status_row_offset) {
  static_assert(Format == 0 && !Fake && Parts >= 2 && Parts <= 8);
  static_assert(Width > 0 && Width % (32 * Parts) == 0 && Width <= 8192);
  constexpr uint32_t Group = 32;
  constexpr uint32_t Groups = Width / Group;
  constexpr uint32_t PartGroups = Groups / Parts;
  constexpr uint32_t Warps = PartGroups < 32 ? PartGroups : 32;
  constexpr uint32_t PerWarp = (PartGroups + Warps - 1) / Warps;
  __shared__ uint32_t shared_flags[Warps];
  __shared__ uint32_t part_flags;
  __shared__ uint32_t row_flags;
  ptx::griddep_wait();
  if constexpr (EarlyRelease) ptx::griddep_launch_dependents();
  const uint32_t row = blockIdx.x / Parts;
  const uint32_t part = ptx::cluster_ctarank();
  if (row >= *live_rows) {
    if constexpr (!EarlyRelease) ptx::griddep_launch_dependents();
    return;
  }
  const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  constexpr uint32_t Bytes = F32Input ? 4 : 2;
  const uint32_t first = part * PartGroups;
  #pragma unroll
  for (uint32_t k = 0; k < PerWarp; ++k) {
    const uint32_t group = warp + k * Warps;
    if (group < PartGroups && lane < Group / 2) {
      const uint64_t at = uint64_t(row) * Width + uint64_t(first + group) * Group + 2 * lane;
      asm volatile("prefetch.global.L1 [%0];" ::"l"(static_cast<const char*>(input) + at * Bytes));
    }
  }
  uint32_t prior = 0;
  if (threadIdx.x == 0)
    prior = *reinterpret_cast<volatile const uint32_t*>(status + status_row_offset + row);
  // codes | scale_code << 16, the only fields a valid row stores.
  uint32_t packed[PerWarp];
  uint32_t flags = 0;
  #pragma unroll
  for (uint32_t k = 0; k < PerWarp; ++k) {
    const uint32_t group = warp + k * Warps;
    if (group < PartGroups) {
      const uint64_t at = uint64_t(row) * Width + uint64_t(first + group) * Group + 2 * lane;
      const GroupValue value =
          block_quantize_parallel::quantize_group<Format, F32Input, Fake, FloorBits>(input, at);
      packed[k] = uint32_t(value.codes) | uint32_t(value.scale_code) << 16;
      flags |= value.flags;
    }
  }
  for (int step = 16; step; step >>= 1) flags |= __shfl_xor_sync(0xffffffff, flags, step);
  if (lane == 0) shared_flags[warp] = flags;
  __syncthreads();
  if (threadIdx.x == 0) {
    uint32_t mine = 0;
    for (uint32_t w = 0; w < Warps; ++w) mine |= shared_flags[w];
    part_flags = mine;
  }
  ptx::cluster_arrive_release();
  ptx::cluster_wait_acquire();
  if (threadIdx.x == 0) {
    uint32_t all = 0;
    for (uint32_t p = 0; p < Parts; ++p) {
      uint32_t value;
      asm volatile("ld.shared::cluster.u32 %0, [%1];"
                   : "=r"(value) : "r"(ptx::cluster_map_shared(&part_flags, p)) : "memory");
      all |= value;
    }
    if (part == 0) StatusOr::update(status, status_row_offset + row, all);
    row_flags = prior | all;
  }
  __syncthreads();
  ptx::cluster_arrive_relaxed();
  const uint32_t decision = row_flags;
  #pragma unroll
  for (uint32_t k = 0; k < PerWarp; ++k) {
    const uint32_t group = warp + k * Warps;
    if (group < PartGroups) {
      const uint64_t at = uint64_t(row) * Width + uint64_t(first + group) * Group + 2 * lane;
      // A flagged row stores the default codes (zero) and scale 1.
      const uint32_t value = decision == 0 ? packed[k] : 127u << 16;
      if (lane == 0) scales[uint64_t(row) * Groups + first + group] = uint8_t(value >> 16);
      if (lane < Group / 2) {
        reinterpret_cast<uint8_t*>(output)[at] = uint8_t(value);
        reinterpret_cast<uint8_t*>(output)[at + 1] = uint8_t(value >> 8);
      }
    }
  }
  if constexpr (!EarlyRelease) ptx::griddep_launch_dependents();
  ptx::cluster_wait_acquire();
}
} // namespace chop::row::block_quantize_split
