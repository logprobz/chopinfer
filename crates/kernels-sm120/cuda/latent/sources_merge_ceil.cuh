#pragma once

// The final merge of sources.cuh followed, in the same CTA, by the E4M3
// ceil groups of block_quantize_sliced over the head's columns, for
// consumers that admit rows by status (gemm/mxfp8_fold_cluster_fused.cuh).
// A CTA owns one row and one head, whose Latent columns are whole 32-column
// groups of the row of Heads * Latent BF16 columns. The merge stores every
// BF16 output first, exactly as merge does; after a block barrier each warp
// reads its groups back and computes each once with
// block_quantize_parallel::quantize_group, stores its codes and scale code
// and ORs its flags into the row status, as block_quantize_sliced does. So
// the BF16 output, the codes and scales of a row whose status stays zero,
// and every row's status bits are those of the merge and then the sliced
// quantizer. A row with a nonzero status keeps its computed codes, which
// its consumer never reads.
#include "sources.cuh"
#include "../row/ceil_parallel.cuh"

namespace chop::latent::sources {

template<class T, uint32_t FloorBits>
__device__ void merge_ceil(const float* workspace, const uint32_t* live_rows,
    const float* sinks, const float* cos_sin, uint16_t* out,
    uint8_t* codes, uint8_t* scales, uint32_t* status, uint64_t status_row_offset) {
  static_assert(!T::partial && T::latent % 32 == 0);
  merge<T>(workspace, live_rows, sinks, cos_sin, out, nullptr, nullptr, nullptr);
  const uint32_t row = blockIdx.x, head = blockIdx.y;
  // merge waited on its producer before this read; a row past it stored
  // nothing and quantizes nothing, in every thread of the CTA.
  if (row >= *live_rows) return;
  __syncthreads();
  constexpr uint32_t Group = 32, Width = T::heads * T::latent, Groups = T::latent / Group;
  const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
  uint32_t flags = 0;
  for (uint32_t k = warp; k < Groups; k += warps) {
    const uint32_t group = head * Groups + k;
    const uint64_t at = uint64_t(row) * Width + uint64_t(group) * Group + 2 * lane;
    const chop::row::block_quantize::GroupValue value =
        chop::row::block_quantize_parallel::quantize_group<0, false, false, FloorBits>(out, at);
    flags |= value.flags;
    if (lane == 0) scales[uint64_t(row) * (Width / Group) + group] = value.scale_code;
    if (lane < Group / 2) *reinterpret_cast<uint16_t*>(codes + at) = value.codes;
  }
  flags = __reduce_or_sync(0xffffffffu, flags);
  if (lane == 0 && flags) atomicOr(status + status_row_offset + row, flags);
}

} // namespace chop::latent::sources
