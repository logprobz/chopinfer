#pragma once

#include <math.h>
#include <stdint.h>
#include "../common/sigmoid.cuh"

// The clamped SwiGLU epilogue of `kernel_core::gated_mlp::SwigluClamp`:
// `gate = min(gate, gate_max)` and `up = clamp(up, -up_limit, up_limit)` on
// the F32 sums, then `silu(gate) * up`. The clamps compare and select, so a
// NaN sum stays NaN (fminf and fmaxf would drop it). The gate has no lower
// bound: at gate >= sigmoid::kTail the unit is `gate / (1 + exp(-gate)) * up`
// in F32; below, where exp(-gate) overflows, `gate * up * exp(gate)` is
// formed in F64 and rounded once (`common/sigmoid.cuh`'s tail). Only the
// clamped grouped and dense instances include this header.
namespace chop::moe::clamp {

// Limits as F32 bits, fixed per instance; `Enabled = false` is the
// unclamped unit.
template<bool Enabled, uint32_t GateMaxBits, uint32_t UpLimitBits>
struct SwigluClamp {
  static constexpr bool enabled = Enabled;
  static constexpr uint32_t gate_max_bits = GateMaxBits;
  static constexpr uint32_t up_limit_bits = UpLimitBits;
};

template<class Clamp>
__device__ __forceinline__ float clamped_swiglu(float gate, float up) {
  if constexpr (Clamp::enabled) {
    const float gate_max = __uint_as_float(Clamp::gate_max_bits);
    const float up_limit = __uint_as_float(Clamp::up_limit_bits);
    gate = gate > gate_max ? gate_max : gate;
    up = up > up_limit ? up_limit : (up < -up_limit ? -up_limit : up);
  }
  if (gate >= sigmoid::kTail) return sigmoid::silu(gate) * up;
  return float(double(gate) * double(up) * exp(double(gate)));
}

}  // namespace chop::moe::clamp
