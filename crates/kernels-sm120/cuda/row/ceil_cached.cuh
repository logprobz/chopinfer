#pragma once
#include "ceil_parallel.cuh"

// One-pass form of block_quantize_parallel::run for decode rows. The
// parallel entry computes every group twice, once for the row status and
// once for its outputs, each in a serial loop, so a warp's dependent loads
// run back to back. Here each warp first prefetches every group's operands
// into L1 (the shuffles keep the compiler from hoisting the loads), then
// computes its groups once in an unrolled loop, keeps each group's codes and scale code in
// one register across the row-status barriers, and stores them. Every group is
// block_quantize_parallel::quantize_group, which is deterministic, so the
// stored codes and scales are the parallel entry's; a flagged row stores
// the parallel entry's default codes and the unit scale.
namespace chop::row::block_quantize_cached {
using namespace block_quantize;

// EarlyRelease: see rmsnorm_quantize_cached_f32.cuh.
template<int Width, int Format, bool F32Input, bool Fake, uint32_t FloorBits,
         bool EarlyRelease = false>
__device__ void run(const void* input, void* output, uint8_t* scales, const uint32_t* live_rows,
                    uint32_t* status, uint64_t status_row_offset) {
  static_assert(Width > 0 && Width % 32 == 0 && Width <= 8192 && Format == 0 && !Fake);
  constexpr uint32_t Group = 32;
  constexpr uint32_t Groups = Width / Group;
  constexpr uint32_t Warps = Groups < 32 ? Groups : 32;
  constexpr uint32_t PerWarp = (Groups + Warps - 1) / Warps;
  __shared__ uint32_t shared_flags[Warps];
  ptx::griddep_wait();
  if constexpr (EarlyRelease) ptx::griddep_launch_dependents();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) {
    if constexpr (!EarlyRelease) ptx::griddep_launch_dependents();
    return;
  }
  const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  constexpr uint32_t Bytes = F32Input ? 4 : 2;
  #pragma unroll
  for (uint32_t k = 0; k < PerWarp; ++k) {
    const uint32_t group = warp + k * Warps;
    if (group < Groups && lane < Group / 2) {
      const uint64_t at = uint64_t(row) * Width + uint64_t(group) * Group + 2 * lane;
      asm volatile("prefetch.global.L1 [%0];" ::"l"(static_cast<const char*>(input) + at * Bytes));
    }
  }
  // codes | scale_code << 16, the only fields a valid row stores.
  uint32_t packed[PerWarp];
  uint32_t flags = 0;
  #pragma unroll
  for (uint32_t k = 0; k < PerWarp; ++k) {
    const uint32_t group = warp + k * Warps;
    if (group < Groups) {
      const uint64_t at = uint64_t(row) * Width + uint64_t(group) * Group + 2 * lane;
      const GroupValue value =
          block_quantize_parallel::quantize_group<Format, F32Input, Fake, FloorBits>(input, at);
      packed[k] = uint32_t(value.codes) | uint32_t(value.scale_code) << 16;
      flags |= value.flags;
    }
  }
  flags = block_quantize_parallel::row_status<Warps>(flags, shared_flags, status,
                                                     status_row_offset + row);
  #pragma unroll
  for (uint32_t k = 0; k < PerWarp; ++k) {
    const uint32_t group = warp + k * Warps;
    if (group < Groups) {
      const uint64_t at = uint64_t(row) * Width + uint64_t(group) * Group + 2 * lane;
      // A flagged row stores the default codes (zero) and scale 1.
      const uint32_t value = flags == 0 ? packed[k] : 127u << 16;
      if (lane == 0) scales[uint64_t(row) * Groups + group] = uint8_t(value >> 16);
      if (lane < Group / 2) {
        reinterpret_cast<uint8_t*>(output)[at] = uint8_t(value);
        reinterpret_cast<uint8_t*>(output)[at + 1] = uint8_t(value >> 8);
      }
    }
  }
  if constexpr (!EarlyRelease) ptx::griddep_launch_dependents();
}
} // namespace chop::row::block_quantize_cached
