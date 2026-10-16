#pragma once

// Qwen3.8's hyper-connection precompute epilogue for the shared cluster
// decode template (cluster_decode.cuh): the down projection (segment 0, the
// 320-row bottleneck; segment 1 of hc_in is the inject projection) is rounded
// to BF16 and stored, and the bottleneck activation sigmoid(v) * v for
// v = rounded / 4 is written to a [tokens, 320] BF16 buffer, exactly as
// chop_row_qwen38_hc_bottleneck_v1 computes it from the stored value. BF16
// output only.

#include "gemm/cluster_decode.cuh"
#include "common/convert.cuh"
#include "common/sigmoid.cuh"

namespace chop::gemm::qwen38 {

struct HcBottleneckStore {
  uint16_t* bottleneck;

  __device__ __forceinline__ void check(uint32_t output_f32) const {
    if (output_f32) asm volatile("trap;");
  }

  __device__ __forceinline__ void operator()(
      void* output, uint64_t index, uint32_t token, uint32_t row, int segment,
      float value, uint32_t output_f32) const {
    if (output_f32) {
      static_cast<float*>(output)[index] = value;
      return;
    }
    const uint16_t rounded = chop::convert::bf16_from_f32(value);
    static_cast<uint16_t*>(output)[index] = rounded;
    if (segment == 0) {
      const float v = chop::convert::bf16_to_f32(rounded) / CHOP_QWEN38_STREAMS;
      bottleneck[uint64_t(token) * CHOP_QWEN38_HC_RANK + row] =
          chop::convert::bf16_from_f32(chop::sigmoid::times(v, v));
    }
  }
};

} // namespace chop::gemm::qwen38
