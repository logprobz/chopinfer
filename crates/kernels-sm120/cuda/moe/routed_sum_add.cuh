#pragma once

#include <stdint.h>
#include "../common/ptx.cuh"

// A rank's routed sum (`chained_experts.cuh`'s `routed_sum`, F32 form)
// followed by one F32 add of a second F32 row of the same width, the
// `addend`: per token and column, the token's local routes added in top-k
// order from +0 (`__fadd_rn`), then `__fadd_rn(sum, addend)`, written as
// F32. The sum's adds are `routed_sum`'s, so the value before the last add
// is the F32 row `routed_sum` writes.
//
// A shared expert whose down projection runs beside the routed experts
// (its F32 output with nothing added) is the addend: the row equals that
// projection's residual epilogue over the routed sum (`acc + residual`, one
// add of the same two values, which commutes).
//
// One CTA of 256 threads per token, four consecutive columns per thread and
// pass; tokens past the live count (read after `griddepcontrol.wait`) write
// nothing.
//
// `M` (lab-only) is a deliberate defect kernel-lab instantiates to show its
// checks fail on it; served entries take `none`, the unmodified kernel.
namespace chop::moe::chained {

enum class SumAddMutation : int {
  none = 0,
  no_addend = 1,     // the addend never added
  addend_first = 2,  // the addend starts the sum instead of +0
  drop_wait = 3,     // no griddepcontrol.wait before the inputs are read
};

template<int TopK, int Width, int LocalExperts, SumAddMutation M = SumAddMutation::none>
__device__ void routed_sum_add(
    const float* routed, const int32_t* ids, const int32_t* live_tokens, uint32_t rank,
    const float* addend, float* output) {
  static_assert(Width % 1024 == 0, "four columns per thread and pass");
  if (blockDim.x != 256) __trap();
  ptx::griddep_launch_dependents();
  if constexpr (M != SumAddMutation::drop_wait) ptx::griddep_wait();
  const int live = *live_tokens;
  if (live > int(gridDim.x)) __trap();
  const int token = blockIdx.x;
  if (token >= live) return;
  bool local[TopK];
#pragma unroll
  for (int k = 0; k < TopK; ++k) {
    const int id = ids[uint64_t(token) * TopK + k];
    local[k] = id >= 0 && uint32_t(id) / LocalExperts == rank;
  }
#pragma unroll
  for (int pass = 0; pass < Width / 1024; ++pass) {
    const int col = pass * 1024 + threadIdx.x * 4;
    const float4 a = *reinterpret_cast<const float4*>(addend + uint64_t(token) * Width + col);
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    if constexpr (M == SumAddMutation::addend_first) {
      acc[0] = a.x;
      acc[1] = a.y;
      acc[2] = a.z;
      acc[3] = a.w;
    }
#pragma unroll
    for (int k = 0; k < TopK; ++k) {
      if (!local[k]) continue;
      const float4 v = *reinterpret_cast<const float4*>(
          routed + (uint64_t(token) * TopK + k) * Width + col);
      acc[0] = __fadd_rn(acc[0], v.x);
      acc[1] = __fadd_rn(acc[1], v.y);
      acc[2] = __fadd_rn(acc[2], v.z);
      acc[3] = __fadd_rn(acc[3], v.w);
    }
    if constexpr (M == SumAddMutation::none || M == SumAddMutation::drop_wait) {
      acc[0] = __fadd_rn(acc[0], a.x);
      acc[1] = __fadd_rn(acc[1], a.y);
      acc[2] = __fadd_rn(acc[2], a.z);
      acc[3] = __fadd_rn(acc[3], a.w);
    }
    *reinterpret_cast<float4*>(output + uint64_t(token) * Width + col) =
        make_float4(acc[0], acc[1], acc[2], acc[3]);
  }
}

}  // namespace chop::moe::chained
