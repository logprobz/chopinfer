#pragma once

// Clamped SwiGLU in F32, one BF16 rounding, then E4M3/32 with a ceil UE8M0
// scale (kernel-core gated_mlp::quantized::SiluMulQuantizeQuery). The scale
// and code rule is block_quantize.cuh's: the BF16 product feeds the amax,
// the floor, the F32 reciprocal product's exponent-bit ceiling and the
// E4M3 RNE of x / scale.

#include <math.h>
#include <stdint.h>
#include "../common/convert.cuh"
#include "../common/ptx.cuh"
#include "block_quantize.cuh"

namespace chop::row::silu_mul_quantize {

// The op's exponential of the clamped gate, expf(-g). kernel-lab's expf
// error probe measures this function over every gate the op can see.
__device__ __forceinline__ float exponential(float g) { return expf(-g); }

// g = min(gate, gate_max), u = clamp(up, -limit, limit), then
// (g / (1 + expf(-g))) * u, each step rounded in F32, then BF16. Non-finite
// gate or up values, and a product that rounds to a BF16 infinity, trap:
// the contract refuses them, so they are never clamped away.
template<uint32_t GateMaxBits, uint32_t UpLimitBits>
__device__ __forceinline__ float product(float gate, float up) {
  if (!isfinite(gate) || !isfinite(up)) asm volatile("trap;");
  const float gate_max = __uint_as_float(GateMaxBits);
  const float limit = __uint_as_float(UpLimitBits);
  const float g = fminf(gate, gate_max);
  const float u = fminf(fmaxf(up, -limit), limit);
  const float silu = __fdiv_rn(g, __fadd_rn(1.f, exponential(g)));
  const float rounded = convert::bf16_to_f32(convert::bf16_from_f32(__fmul_rn(silu, u)));
  if (!isfinite(rounded)) asm volatile("trap;");
  return rounded;
}

// Grid (rows, Width / 32), 32 threads: lane l < 16 owns columns 2l and
// 2l + 1 of its group. gate and up are [rows, Width] F32, codes
// [rows, Width] E4M3 and scales [rows, Width / 32] UE8M0. Each column is a
// scalar load, so the operands need only F32 alignment.
template<int Width, uint32_t GateMaxBits, uint32_t UpLimitBits, uint32_t FloorBits>
__device__ void run(const float* gate, const float* up, uint8_t* codes, uint8_t* scales,
                    const uint32_t* live_rows) {
  static_assert(Width > 0 && Width % 32 == 0);
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  const uint32_t group = blockIdx.y;
  const uint32_t lane = threadIdx.x;
  const uint64_t at = uint64_t(row) * Width + uint64_t(group) * 32 + 2 * lane;
  float x = 0.f, y = 0.f;
  if (lane < 16) {
    x = product<GateMaxBits, UpLimitBits>(gate[at], up[at]);
    y = product<GateMaxBits, UpLimitBits>(gate[at + 1], up[at + 1]);
  }
  float maximum = fmaxf(fabsf(x), fabsf(y));
  for (int step = 16; step; step >>= 1)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, step));
  maximum = fmaxf(maximum, __uint_as_float(FloorBits));
  const uint8_t scale_code = block_quantize::ceil_scale(maximum, 1.f / 448.f);
  const float scale = __uint_as_float(uint32_t(scale_code) << 23);
  if (lane == 0) scales[uint64_t(row) * (Width / 32) + group] = scale_code;
  if (lane >= 16) return;
  const uint16_t pair =
      convert::pack_f32_e4m3(convert::Float2{__fdiv_rn(x, scale), __fdiv_rn(y, scale)});
  codes[at] = uint8_t(pair);
  codes[at + 1] = uint8_t(pair >> 8);
}

} // namespace chop::row::silu_mul_quantize
