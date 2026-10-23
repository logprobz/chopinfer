#pragma once
#include "allreduce_phased_bf16.cuh"

// The arithmetic steps of a copy-engine transport of BF16 partials. Both
// entries call allreduce_phased_bf16.cuh's own functions, so the result is
// that header's bits: every rank rounds its F32 partial once to BF16
// (send_rounded), and the owner sums the ranks' BF16 values in F32 in rank
// order from F32 zero, its own values rounded as sent, and rounds once
// (sum_block). Neither entry performs peer access, polling or protocol
// mutation; the caller copies the rounded partials between ranks.
namespace chop::comm::owned_sum_bf16 {

constexpr uint64_t kGranule = 1024;

// Round `elements` F32 values once to BF16 for the transport to send.
__device__ __forceinline__ void round(const float* input, uint16_t* output, uint64_t elements) {
  for (uint64_t first = uint64_t(blockIdx.x) * kGranule; first < elements;
       first += uint64_t(gridDim.x) * kGranule) {
    const uint64_t count = elements - first < kGranule ? elements - first : kGranule;
    phased_bf16::send_rounded(output + first, input + first, count);
  }
}

// The owner's sums: its own F32 partial from `input`, every other rank's
// rounded partial from `inbox[source * slot]`.
__device__ __forceinline__ void run(const float* input, const uint16_t* inbox, uint16_t* output,
                                    uint64_t elements, uint64_t slot, uint32_t rank,
                                    uint32_t ranks) {
  for (uint64_t first = uint64_t(blockIdx.x) * kGranule; first < elements;
       first += uint64_t(gridDim.x) * kGranule) {
    const uint64_t count = elements - first < kGranule ? elements - first : kGranule;
    phased_bf16::sum_block(input, inbox, output, slot, first, count, rank, ranks);
  }
}

}  // namespace chop::comm::owned_sum_bf16
