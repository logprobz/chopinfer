#pragma once
#include "position_blocks.cuh"

// Keep the compiled ID stride and pad unused slots. The default producer
// remains in position_blocks.cuh and has a separate symbol and cubin.
namespace chop::position_candidates_runtime {
using namespace chop::position_blocks;
template<int Block, int Kept>
__device__ void gather(
    const float* blocks, const uint32_t* live_rows, const uint64_t* tags,
    const int32_t* visible_counts, const int32_t* source_status,
    int32_t* ids, uint64_t* out_tags, int32_t* out_visible, int32_t* status,
    uint32_t width, uint32_t top_k_blocks) {
  __shared__ uint32_t histogram[threads], rank, scan[threads], written;
  __shared__ uint64_t prefix;
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  const uint32_t count = min(uint32_t(max(visible_counts[row], 0)), width);
  int32_t* out = ids + size_t(row) * Kept;
  for (int i = threadIdx.x; i < Kept; i += threads) out[i] = -1;
  const uint32_t block_count = (width + Block - 1) / Block;
  const float* values = blocks + size_t(row) * block_count;
  int invalid = source_status[row] != 0 || top_k_blocks == 0 || top_k_blocks > Kept;
  for (uint32_t i = threadIdx.x; i < block_count; i += threads) invalid |= isnan(values[i]);
  const int failed = __syncthreads_or(invalid) != 0;
  if (threadIdx.x == 0) {
    out_tags[row] = tags[row];
    out_visible[row] = int32_t(count);
    status[row] = failed;
  }
  if (failed) return;
  auto key = [&](uint32_t i) {
    return values[i] > -__int_as_float(0x7f800000) ? candidate_key(values[i], i) : uint64_t(0);
  };
  const uint64_t threshold = cutoff(block_count, top_k_blocks, key, histogram, &prefix, &rank);
  if (threadIdx.x == 0) written = 0;
  __syncthreads();
  for (uint32_t base = 0; base < block_count; base += threads) {
    const uint32_t i = base + threadIdx.x;
    const uint64_t value = i < block_count ? key(i) : 0;
    const bool keep = value != 0 && value >= threshold;
    scan[threadIdx.x] = keep;
    __syncthreads();
    for (int distance = 1; distance < threads; distance *= 2) {
      const uint32_t add = threadIdx.x >= distance ? scan[threadIdx.x - distance] : 0;
      __syncthreads();
      scan[threadIdx.x] += add;
      __syncthreads();
    }
    if (keep) out[written + scan[threadIdx.x] - 1] = int32_t(i);
    __syncthreads();
    if (threadIdx.x == 0) written += scan[threads - 1];
    __syncthreads();
  }
}
} // namespace chop::position_candidates_runtime
