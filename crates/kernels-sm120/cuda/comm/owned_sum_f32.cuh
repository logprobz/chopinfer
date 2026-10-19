#pragma once
#include "allreduce_phased_f32.cuh"

namespace chop::comm::owned_sum_f32 {

// The transport has already copied peer partials into rank-indexed local
// slots. This entry performs no peer access, polling or protocol mutation.
__device__ __forceinline__ void run(const float* input, const float* inbox,
                                    uint16_t* output, uint64_t elements,
                                    uint64_t slot, uint32_t rank, uint32_t ranks) {
  constexpr uint64_t granule = 1024;
  for (uint64_t first = uint64_t(blockIdx.x) * granule; first < elements;
       first += uint64_t(gridDim.x) * granule) {
    const uint64_t count = elements - first < granule ? elements - first : granule;
    phased_f32::sum_block<uint16_t>(input, inbox, output, slot, first, count, rank, ranks);
  }
}

}  // namespace chop::comm::owned_sum_f32
