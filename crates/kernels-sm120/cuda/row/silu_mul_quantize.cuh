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
// (g / (1 + expf(-g))) * u, each step rounded in F32, then BF16. Invalid
// operands never enter the product. In particular, -inf / expf(+inf)
// must not turn a non-finite input into a second product-overflow flag.
template<uint32_t GateMaxBits, uint32_t UpLimitBits, bool SkipInvalid = true>
__device__ __forceinline__ float product(float gate, float up, uint32_t& flags) {
  if (!isfinite(gate) || !isfinite(up)) {
    flags |= block_quantize::NonfiniteInput;
    if constexpr (SkipInvalid) return 0.f;
  }
  const float gate_max = __uint_as_float(GateMaxBits);
  const float limit = __uint_as_float(UpLimitBits);
  const float g = fminf(gate, gate_max);
  const float u = fminf(fmaxf(up, -limit), limit);
  const float silu = __fdiv_rn(g, __fadd_rn(1.f, exponential(g)));
  const float rounded = convert::bf16_to_f32(convert::bf16_from_f32(__fmul_rn(silu, u)));
  if (!isfinite(rounded)) {
    flags |= block_quantize::UnrepresentableOutput;
    return 0.f;
  }
  return rounded;
}

template<uint32_t GateMaxBits, uint32_t UpLimitBits, uint32_t FloorBits, bool SkipInvalid>
__device__ __forceinline__ block_quantize::GroupValue quantize_group(
    const float* gate, const float* up, uint64_t at) {
  const uint32_t lane = threadIdx.x;
  block_quantize::GroupValue result;
  float x = 0.f, y = 0.f;
  if (lane < 16) {
    x = product<GateMaxBits, UpLimitBits, SkipInvalid>(gate[at], up[at], result.flags);
    y = product<GateMaxBits, UpLimitBits, SkipInvalid>(gate[at + 1], up[at + 1], result.flags);
  }
  float maximum = fmaxf(fabsf(x), fabsf(y));
  for (int step = 16; step; step >>= 1)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, step));
  maximum = fmaxf(maximum, __uint_as_float(FloorBits));
  result.scale_code = block_quantize::ceil_scale(maximum, 1.f / 448.f);
  const float scale = __uint_as_float(uint32_t(result.scale_code) << 23);
  if (lane >= 16) return result;
  result.codes =
      convert::pack_f32_e4m3(convert::Float2{__fdiv_rn(x, scale), __fdiv_rn(y, scale)});
  auto decoded = convert::unpack_f32_e4m3(result.codes);
  decoded.lo = __fmul_rn(decoded.lo, scale);
  decoded.hi = __fmul_rn(decoded.hi, scale);
  const auto exact = convert::bf16x2_to_f32(convert::bf16x2_from_f32(decoded));
  if (!isfinite(exact.lo) || !isfinite(exact.hi) ||
      __float_as_uint(exact.lo) != __float_as_uint(decoded.lo) ||
      __float_as_uint(exact.hi) != __float_as_uint(decoded.hi))
    result.flags |= block_quantize::UnrepresentableOutput;
  return result;
}

// One warp per row validates every group before writing any output. This
// avoids races between groups when a late group fails. The second pass
// preserves the original per-group arithmetic and positive/negative zeros
// on passing rows. All data paths report status without trapping.
template<int Width, uint32_t GateMaxBits, uint32_t UpLimitBits, uint32_t FloorBits,
         class Status = block_quantize::StatusOr, bool SkipInvalid = true, bool WholeRow = true>
__device__ void run(const float* gate, const float* up, uint8_t* codes, uint8_t* scales,
                    const uint32_t* live_rows, uint32_t* status, uint64_t status_row_offset) {
  static_assert(Width > 0 && Width % 32 == 0);
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) { ptx::griddep_launch_dependents(); return; }
  const uint32_t lane = threadIdx.x;
  uint32_t flags = 0;
#pragma unroll 1
  for (uint32_t group = 0; group < Width / 32; ++group) {
    const uint64_t at = uint64_t(row) * Width + uint64_t(group) * 32 + 2 * lane;
    flags |= quantize_group<GateMaxBits,UpLimitBits,FloorBits,SkipInvalid>(gate,up,at).flags;
  }
  for (int step = 16; step; step >>= 1)
    flags |= __shfl_xor_sync(0xffffffff, flags, step);
  if (lane == 0) flags = Status::update(status,status_row_offset+row,flags);
  flags = __shfl_sync(0xffffffff,flags,0);
#pragma unroll 1
  for (uint32_t group = 0; group < Width / 32; ++group) {
    const uint64_t at = uint64_t(row) * Width + uint64_t(group) * 32 + 2 * lane;
    block_quantize::GroupValue value;
    if (flags == 0 || (!WholeRow && group != 0))
      value = quantize_group<GateMaxBits,UpLimitBits,FloorBits,SkipInvalid>(gate,up,at);
    else value.scale_code = 127; // fixed finite scale 1
    if (lane == 0) scales[uint64_t(row) * (Width / 32) + group] = value.scale_code;
    if (lane < 16) {
      codes[at] = uint8_t(value.codes);
      codes[at + 1] = uint8_t(value.codes >> 8);
    }
  }
  ptx::griddep_launch_dependents();
}

} // namespace chop::row::silu_mul_quantize
