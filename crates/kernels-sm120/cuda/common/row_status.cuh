#pragma once
#include "../row/block_quantize.cuh"

namespace chop::row_status {
using block_quantize = chop::row::block_quantize::StatusOr;
// One CTA validates a row before publishing its outputs. Preserve all prior
// bits through the shared quantizer's atomic OR protocol.
template<class Status = block_quantize>
__device__ uint32_t commit(uint32_t flags, uint32_t* status, uint64_t row) {
  __shared__ uint32_t combined;
  const uint32_t input = __syncthreads_or((flags & 1) != 0) ? 1 : 0;
  const uint32_t scale = __syncthreads_or((flags & 2) != 0) ? 2 : 0;
  const uint32_t output = __syncthreads_or((flags & 4) != 0) ? 4 : 0;
  if (threadIdx.x == 0) combined = Status::update(status, row, input | scale | output) | input | scale | output;
  __syncthreads();
  return combined;
}
} // namespace chop::row_status
