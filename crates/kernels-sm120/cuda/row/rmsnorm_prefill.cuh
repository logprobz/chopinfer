#pragma once

// Prefill-only plain BF16 RMSNorm and ceil E4M3/32. The norm's 256-thread
// pack mapping, tree, rsqrt and BF16 store boundary match row.cuh. Publish
// every materialized norm output before releasing dependent consumers.
//
// With `Certify`, the norm also writes `certified[row]`: whether every BF16
// output element, widened, is an exact F32 projection's tensor operand
// (`exact_f32::tensor_operand`). The widened rows are exact by
// construction, so this is the flag `exact_f32::zero_prefill::prepare`
// derives as `single`, and `exact_f32::certified` reads the BF16 rows and
// the flag in its place. The other outputs are unchanged.
#include "row.cuh"
#include "ceil_parallel.cuh"
#include "../gemm/exact_f32.cuh"

namespace chop::row::rmsnorm_prefill {

// `certify_all` (lab-only) certifies every row, out-of-domain ones too.
enum class Mutation { none, round_before_weight, clear_status, certify_all };

template<int Width, uint32_t FloorBits, bool CacheGroups, Mutation M = Mutation::none,
         bool Certify = false>
__device__ void run(const void* input, const uint16_t* weight, void* normed,
                    float* widened, uint8_t* codes, uint8_t* scales,
                    const uint32_t* live_rows, uint32_t* status,
                    uint64_t status_row_offset, float eps,
                    uint32_t* certified = nullptr) {
  static_assert(Width > 0 && Width % 32 == 0);
  if (blockDim.x != 256) __trap();
  __shared__ float sums[8];
  __shared__ __align__(16) uint16_t staged[Width];
  __shared__ __align__(16) uint8_t cached_codes[CacheGroups ? Width : 1];
  __shared__ uint8_t cached_scales[CacheGroups ? Width / 32 : 1];
  __shared__ uint32_t shared_flags[8];
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) { ptx::griddep_launch_dependents(); return; }
  const auto* x = static_cast<const uint16_t*>(input);
  auto* y = static_cast<uint16_t*>(normed);
  const uint64_t base = uint64_t(row) * Width;
  float square = 0.f;
  for (int c = threadIdx.x * 8; c < Width; c += blockDim.x * 8) {
    const uint4 packed = *reinterpret_cast<const uint4*>(x + base + c);
    const auto* in = reinterpret_cast<const uint16_t*>(&packed);
    for (int j = 0; j < 8; ++j) {
      const float v = bf(in[j]);
      square += v * v;
    }
  }
  const float total = sum_cta(square, sums);
  const float inv = rsqrtf(total / Width + eps);
  bool operands = true;
  for (int c = threadIdx.x * 8; c < Width; c += blockDim.x * 8) {
    const uint4 packed = *reinterpret_cast<const uint4*>(x + base + c);
    const uint4 wpacked = *reinterpret_cast<const uint4*>(weight + c);
    const auto* in = reinterpret_cast<const uint16_t*>(&packed);
    const auto* weights = reinterpret_cast<const uint16_t*>(&wpacked);
    uint4 result;
    auto* out = reinterpret_cast<uint16_t*>(&result);
    for (int j = 0; j < 8; ++j) {
      const float w = bf(weights[j]);
      float n = bf(in[j]) * inv;
      if constexpr (M == Mutation::round_before_weight) n = bf(to_bf(n));
      out[j] = to_bf(n * w);
      if constexpr (Certify) operands &= exact_f32::tensor_operand(bf(out[j]));
    }
    *reinterpret_cast<uint4*>(y + base + c) = result;
    *reinterpret_cast<uint4*>(staged + c) = result;
    if (widened) {
      *reinterpret_cast<float4*>(widened + base + c) =
          make_float4(bf(out[0]), bf(out[1]), bf(out[2]), bf(out[3]));
      *reinterpret_cast<float4*>(widened + base + c + 4) =
          make_float4(bf(out[4]), bf(out[5]), bf(out[6]), bf(out[7]));
    }
  }
  if constexpr (M == Mutation::clear_status)
    if (threadIdx.x == 0) status[status_row_offset + row] = 0;
  if constexpr (Certify) {
    const int admitted = __syncthreads_and(operands);
    if (threadIdx.x == 0) certified[row] = M == Mutation::certify_all || admitted;
  } else {
    __syncthreads();
  }
  const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  uint32_t flags = 0;
  #pragma unroll 1
  for (uint32_t group = warp; group < Width / 32; group += 8) {
    const uint32_t at = group * 32 + 2 * lane;
    const auto value = block_quantize_parallel::quantize_group<0,false,false,FloorBits>(staged, at);
    flags |= value.flags;
    if constexpr (CacheGroups) {
      if (lane == 0) cached_scales[group] = value.scale_code;
      if (lane < 16) {
        cached_codes[at] = uint8_t(value.codes);
        cached_codes[at + 1] = uint8_t(value.codes >> 8);
      }
    }
  }
  // The integer OR also orders every provisional group store. No codes
  // become visible globally until the whole row, including sticky status,
  // has been admitted.
  flags = block_quantize_parallel::row_status<8>(flags, shared_flags, status, status_row_offset + row);
  if constexpr (CacheGroups) {
    for (uint32_t c = threadIdx.x * 16; c < Width; c += blockDim.x * 16)
      *reinterpret_cast<uint4*>(codes + base + c) = flags ? make_uint4(0,0,0,0)
          : *reinterpret_cast<const uint4*>(cached_codes + c);
    for (uint32_t group = threadIdx.x; group < Width / 32; group += blockDim.x)
      scales[uint64_t(row) * (Width / 32) + group] = flags ? 127 : cached_scales[group];
  } else {
    #pragma unroll 1
    for (uint32_t group = warp; group < Width / 32; group += 8) {
      const uint32_t at = group * 32 + 2 * lane;
      block_quantize::GroupValue value;
      if (!flags) value = block_quantize_parallel::quantize_group<0,false,false,FloorBits>(staged, at);
      else value.scale_code = 127;
      if (lane == 0) scales[uint64_t(row) * (Width / 32) + group] = value.scale_code;
      if (lane < 16) {
        codes[base + at] = uint8_t(value.codes);
        codes[base + at + 1] = uint8_t(value.codes >> 8);
      }
    }
  }
  ptx::griddep_launch_dependents();
}

} // namespace chop::row::rmsnorm_prefill
