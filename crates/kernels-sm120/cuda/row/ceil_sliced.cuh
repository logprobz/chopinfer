#pragma once
#include "ceil_parallel.cuh"

// Decode rows of block_quantize_parallel's E4M3 groups spread over Slices
// CTAs per row, for consumers that admit rows by status
// (gemm/mxfp8_fold_cluster_fused.cuh). The parallel entry runs one CTA per
// row, so at decode it occupies one SM per row. Here each warp prefetches
// its consecutive groups into L1 (as ceil_cached.cuh does), computes each
// once with block_quantize_parallel::quantize_group and stores its codes
// and scale code, and ORs its flags into the row status. A row whose status
// stays zero therefore stores exactly the parallel entry's codes and
// scales, and every row ends with the parallel entry's status bits. A row
// with a nonzero status keeps its computed codes instead of zeros and scale
// 1; they are dead, since its consumer fails the row by status.
namespace chop::row::block_quantize_sliced {
using namespace block_quantize;

template<int Width, int Slices, int Warps, bool F32Input, uint32_t FloorBits>
__device__ void run(const void* input, uint8_t* codes, uint8_t* scales,
                    const uint32_t* live_rows, uint32_t* status, uint64_t status_row_offset) {
  constexpr uint32_t Group = 32;
  constexpr uint32_t Groups = Width / Group / Slices;
  constexpr uint32_t PerWarp = Groups / Warps;
  static_assert(Width % (Group * Slices) == 0 && Groups % Warps == 0 && PerWarp >= 1);
  ptx::griddep_wait();
  // The consumer's own wait still covers this grid's stores.
  ptx::griddep_launch_dependents();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const uint32_t first = blockIdx.y * Groups + warp * PerWarp;
  constexpr uint32_t Bytes = F32Input ? 4 : 2;
#pragma unroll
  for (uint32_t k = 0; k < PerWarp; ++k) {
    if (lane < Group / 2) {
      const uint64_t at = uint64_t(row) * Width + uint64_t(first + k) * Group + 2 * lane;
      asm volatile("prefetch.global.L1 [%0];" ::"l"(static_cast<const char*>(input) + at * Bytes));
    }
  }
  uint32_t flags = 0;
#pragma unroll
  for (uint32_t k = 0; k < PerWarp; ++k) {
    const uint64_t at = uint64_t(row) * Width + uint64_t(first + k) * Group + 2 * lane;
    const GroupValue value =
        block_quantize_parallel::quantize_group<0, F32Input, false, FloorBits>(input, at);
    flags |= value.flags;
    if (lane == 0) scales[uint64_t(row) * (Width / Group) + first + k] = value.scale_code;
    if (lane < Group / 2) *reinterpret_cast<uint16_t*>(codes + at) = value.codes;
  }
  flags = __reduce_or_sync(0xffffffffu, flags);
  if (lane == 0 && flags) atomicOr(status + status_row_offset + row, flags);
}
} // namespace chop::row::block_quantize_sliced
