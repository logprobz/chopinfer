#pragma once

#include "moe.cuh"

namespace chop::moe {

// Sums a split-K entry's partials in ascending split order and applies the
// epilogue, for the routed rows its compute entry wrote: those with a sorted
// slot in the expert ranges `run` visits (the experts up to the
// `active_experts`th with rows, below the padded total). Other routes keep
// their output, as the unsplit entry leaves them, so an expert range that
// omits a route never reads its unwritten partials.
template<bool GateUp, bool QuantizeOut, int NvGlobal, int TopK, int OutN,
         int TokensCapacity, int SplitK, int Experts, int Threads>
__device__ void reduce_owned(
    const float* partials, uint16_t* output, const float* topk_weights,
    const int32_t* live_tokens, const float* activation_f32_scales,
    const float* weight_f32_scales, uint8_t* quantized_output,
    uint8_t* quantized_scales, const int32_t* expert_offsets,
    const int32_t* sorted_rows, const int32_t* num_padded_rows,
    const int32_t* active_experts) {
  static_assert(SplitK > 1 && OutN % 32 == 0);
  // The routed rows one CTA's outputs span.
  constexpr int span = Threads / OutN + 2;
  __shared__ int owned[span];
  __shared__ int slot_end;
  if (threadIdx.x == 0) ptx::griddep_launch_dependents();
  __syncthreads();
  ptx::griddep_wait();
  const uint64_t first = uint64_t(blockIdx.x) * Threads;
  const int tokens = *live_tokens;
  const uint64_t live = min(uint64_t(TokensCapacity), uint64_t(max(tokens, 0))) * TopK * OutN;
  if (first >= live) return;
  const int first_route = int(first / OutN);
  for (int i = threadIdx.x; i < span; i += Threads) owned[i] = 0;
  if (threadIdx.x == 0) {
    // As `run`: nothing when a count is empty, else the ranges of the
    // experts up to the active_experts-th with rows.
    const int active = *active_experts;
    const int padded = *num_padded_rows;
    int end = expert_offsets[0];
    if (padded > 0 && active > 0) {
      int seen = 0;
      for (int e = 0; e < Experts && seen < active; ++e) {
        seen += expert_offsets[e + 1] > expert_offsets[e];
        end = expert_offsets[e + 1];
      }
    }
    slot_end = min(end, max(padded, 0));
  }
  __syncthreads();
  for (int slot = expert_offsets[0] + int(threadIdx.x); slot < slot_end; slot += Threads) {
    const int route = sorted_rows[slot] - first_route;
    if (route >= 0 && route < span) owned[route] = 1;
  }
  __syncthreads();
  const uint64_t index = first + threadIdx.x;
  if (index >= live) return;
  const int routed = int(index / OutN);
  if (!owned[routed - first_route]) return;
  float gate = 0.0f, up = 0.0f;
#pragma unroll
  for (int split = 0; split < SplitK; ++split) {
    const uint64_t partial_index =
        (uint64_t(split * tokens * TopK + routed) * OutN + index % OutN)
        * (GateUp ? 2 : 1);
    gate += partials[partial_index];
    if constexpr (GateUp) up += partials[partial_index + 1];
  }
  if constexpr (NvGlobal == 1) {
    const float scale = activation_f32_scales[0] * weight_f32_scales[0];
    gate *= scale;
    if constexpr (GateUp) up *= scale;
  } else if constexpr (NvGlobal == 2) {
    gate = __fmul_rn(gate, weight_f32_scales[0]);
    if constexpr (GateUp) up = __fmul_rn(up, weight_f32_scales[0]);
  }
  const float value = GateUp
      ? gate / (1.0f + expf(-gate)) * up
      : gate * topk_weights[routed];
  output[index] = convert::bf16_from_f32(value);
  if constexpr (QuantizeOut) {
    // A warp's 32 columns belong to one routed row (OutN % 32 == 0), so a
    // whole warp returns or quantizes together.
    float maximum = fabsf(value);
    for (int mask = 16; mask > 0; mask >>= 1)
      maximum = fmaxf(maximum, __uint_as_float(
          ptx::shuffle_xor(__float_as_uint(maximum), mask)));
    int scale_code = maximum == 0.0f ? 127 : ilogbf(maximum) - 8 + 127;
    scale_code = scale_code < 0 ? 0 : (scale_code > 254 ? 254 : scale_code);
    const float scale = ldexpf(1.0f, scale_code - 127);
    if ((threadIdx.x & 31) == 0)
      quantized_scales[uint64_t(routed) * (OutN / 32)
                       + (index % OutN) / 32] = uint8_t(scale_code);
    quantized_output[index] = uint8_t(convert::pack_f32_e4m3(
        {value / scale, value / scale}));
  }
}

} // namespace chop::moe
