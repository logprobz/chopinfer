#pragma once

#include <cuda.h>
#include "gemm/dense_mxfp4.cuh"
#include "moe/moe.cuh"
#include "moe/grouped.cuh"

namespace chop::probe {

// Each input word holds two packed codes, one scale at bit 16 and a
// fragment byte position at bit 24. Bit 31 adds an extreme scale in a
// different slot, selected by bits 29:28; bit 30 selects 254 instead of 253.
// Every thread writes four BF16 pairs: dense, packed grouped, fragment grouped and
// the served grouped operand expansion with its actual fragment scale selection.
// FP8 has no fragment converter, so its last two pairs repeat packed grouped.
template<bool Fp4>
__device__ void mx_operands(const uint32_t* input, uint32_t* output,
                            uint32_t count, uint32_t path) {
  __shared__ __align__(128) uint8_t tile[32 * 128];
  __shared__ uint8_t scales[32];
  const uint32_t i = blockIdx.x * 32 + threadIdx.x;
  if (i >= count) return;
  const int row = threadIdx.x;
  const uint32_t word = input[i];
  const uint8_t scale = uint8_t(word >> 16);
  scales[row] = scale;
  tile[smem::swizzle_128(tile, row * 128, 128)] = uint8_t(word);
  if constexpr (!Fp4)
    tile[smem::swizzle_128(tile, row * 128 + 1, 128)] = uint8_t(word >> 8);
  if (path == 0 || path == 1) {
    if constexpr (Fp4)
      output[4 * i] = gemm::packed_fp4_weight<256, gemm::WeightKind::mxfp4>(
          tile, row, 0, row, 0, 32, 32,
          reinterpret_cast<const float*>(scales), nullptr);
    else
      output[4 * i] = gemm::packed_fp8_weight<128, gemm::WeightKind::mxfp8>(
          tile, row, 0, row, 0, 32, 32,
          reinterpret_cast<const float*>(scales));
  }
  if (path == 0 || path == 2) {
    if constexpr (Fp4)
      output[4 * i + 1] = moe::scaled_mxfp4_bf16x2(uint8_t(word), scale);
    else
      output[4 * i + 1] = moe::scaled_mxfp8_bf16x2(uint16_t(word), scale);
  }
  if (path == 0 || path == 3) {
    if constexpr (Fp4)
      output[4 * i + 2] = moe::dequant(word, 0, uint32_t(scale) * 0x0101);
    else
      output[4 * i + 2] = moe::scaled_mxfp8_bf16x2(uint16_t(word), scale);
  }
  if (path == 0 || path >= 4) {
    if constexpr (Fp4) {
      // Place this pair in each of the 16 byte positions, against ordinary
      // scales in every other slot. Other slots may carry unrelated extreme scales.
      const int position = path == 0 ? (word >> 24) & 15 : path - 4;
      const int step = position / 4, byte = position % 4;
      const uint32_t codes = uint32_t(uint8_t(word)) << (8 * byte);
      const uint4 fragment{step == 0 ? codes : 0, step == 1 ? codes : 0,
                           step == 2 ? codes : 0, step == 3 ? codes : 0};
      const int shift = 8 * (2 * (step >> 1) + (byte & 1));
      uint32_t scale_word = (0x7f7f7f7f & ~(0xffu << shift))
                           | (uint32_t(scale) << shift);
      if (word & 0x80000000u) {
        const int other_shift = 8 * ((word >> 28) & 3);
        const uint32_t extreme = 253 + ((word >> 30) & 1);
        scale_word = (scale_word & ~(0xffu << other_shift)) | (extreme << other_shift);
      }
      output[4 * i + 3] = moe::grouped::mxfp4_operand(fragment, scale_word, step).v[byte];
    } else {
      output[4 * i + 3] = moe::scaled_mxfp8_bf16x2(uint16_t(word), scale);
    }
  }
}

} // namespace chop::probe
