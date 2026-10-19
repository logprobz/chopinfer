#pragma once
#include "row.cuh"
#include "ceil_parallel.cuh"

namespace chop::row::rmsnorm_quantize_tiles {
template<int Width> struct Arena {
  float sums[8];
  uint32_t flags[8];
  alignas(16) uint16_t normed[Width];
};

// Every CTA repeats the original row tree and validates the entire row.
// Only owned output groups are published, so CTAs never race norm stores.
// Input and weight must be read-only and disjoint from every output.
template<int Width, int Tiles, uint32_t FloorBits>
__device__ void task(const uint16_t* input, const uint16_t* weight,
    uint16_t* normed, float* widened, uint8_t* codes, uint8_t* scales,
    const uint32_t* live_rows, uint32_t* status, uint64_t status_row_offset,
    float eps, uint32_t tile, uint32_t row, unsigned char* arena) {
  if (row >= *live_rows) return;
  auto& shared = *reinterpret_cast<Arena<Width>*>(arena);
  const uint64_t base = uint64_t(row) * Width;
  float square = 0.f;
  for (int c = threadIdx.x * 8; c < Width; c += 256 * 8) {
    const uint4 packed = *reinterpret_cast<const uint4*>(input + base + c);
    const uint16_t* in = reinterpret_cast<const uint16_t*>(&packed);
    for (int j = 0; j < 8; ++j) {
      const float v = bf(in[j]);
      square += v * v;
    }
  }
  const float total = sum_cta(square, shared.sums);
  const float inv = rsqrtf(total / Width + eps);
  for (int c = threadIdx.x * 8; c < Width; c += 256 * 8) {
    const uint4 packed = *reinterpret_cast<const uint4*>(input + base + c);
    const uint4 wpacked = *reinterpret_cast<const uint4*>(weight + c);
    const uint16_t* in = reinterpret_cast<const uint16_t*>(&packed);
    const uint16_t* weights = reinterpret_cast<const uint16_t*>(&wpacked);
    uint4 result;
    uint16_t* out = reinterpret_cast<uint16_t*>(&result);
    for (int j = 0; j < 8; ++j) {
      const float w = bf(weights[j]);
      const float n = bf(in[j]) * inv;
      out[j] = to_bf(n * w);
    }
    *reinterpret_cast<uint4*>(shared.normed + c) = result;
    if ((c / 256) % Tiles == tile) {
      *reinterpret_cast<uint4*>(normed + base + c) = result;
      if (widened) for (int j = 0; j < 8; ++j) widened[base + c + j] = bf(out[j]);
    }
  }
  __syncthreads();
  const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  uint32_t flags = 0;
  #pragma unroll 1
  for (uint32_t group = warp; group < Width / 32; group += 8)
    flags |= block_quantize_parallel::quantize_group<0,false,false,FloorBits>(
        shared.normed, uint64_t(group) * 32 + 2 * lane).flags;
  // Identical full-row validation in every CTA makes the sticky OR and
  // whole-row fallback independent of which CTA reaches status first.
  flags = block_quantize_parallel::row_status<8>(flags, shared.flags, status, status_row_offset + row);
  #pragma unroll 1
  for (uint32_t group = tile * 8 + warp; group < Width / 32; group += Tiles * 8) {
    block_quantize::GroupValue value;
    if (flags == 0)
      value = block_quantize_parallel::quantize_group<0,false,false,FloorBits>(
          shared.normed, uint64_t(group) * 32 + 2 * lane);
    else value.scale_code = 127;
    if (lane == 0) scales[uint64_t(row) * (Width / 32) + group] = value.scale_code;
    if (lane < 16) {
      const uint64_t at = base + uint64_t(group) * 32 + 2 * lane;
      codes[at] = uint8_t(value.codes);
      codes[at + 1] = uint8_t(value.codes >> 8);
    }
  }
}
} // namespace chop::row::rmsnorm_quantize_tiles
