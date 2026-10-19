#pragma once
#include "row.cuh"
#include "ceil_parallel.cuh"

namespace chop::row::rmsnorm_quantize_cached {
template<int Width, int Warps> struct Arena {
  float sums[8];
  uint32_t flags[Warps];
  alignas(16) uint16_t normed[Width];
};

// Only the original 256 lanes accumulate RMS. Extra warps help normalize
// and quantize, without adding zero lanes to either floating-point tree.
template<int Width, int Warps, uint32_t FloorBits>
__device__ void task(const uint16_t* input, const uint16_t* weight,
    uint16_t* normed, float* widened, uint8_t* codes, uint8_t* scales,
    const uint32_t* live_rows, uint32_t* status, uint64_t status_row_offset,
    float eps, uint32_t tile, uint32_t row, unsigned char* arena) {
  (void)tile;
  if (row >= *live_rows) return;
  auto& shared = *reinterpret_cast<Arena<Width,Warps>*>(arena);
  const uint32_t t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const uint64_t base = uint64_t(row) * Width;
  if (t < 256) {
    float square = 0.f;
    for (int c = t * 8; c < Width; c += 256 * 8) {
      const uint4 packed = *reinterpret_cast<const uint4*>(input + base + c);
      const uint16_t* in = reinterpret_cast<const uint16_t*>(&packed);
      for (int j = 0; j < 8; ++j) {
        const float v = bf(in[j]);
        square += v * v;
      }
    }
    for (int offset = 16; offset; offset >>= 1)
      square += __shfl_down_sync(0xffffffff, square, offset);
    if (lane == 0) shared.sums[warp] = square;
  }
  __syncthreads();
  if (warp == 0) {
    float total = lane < 8 ? shared.sums[lane] : 0.f;
    for (int offset = 16; offset; offset >>= 1)
      total += __shfl_down_sync(0xffffffff, total, offset);
    if (lane == 0) shared.sums[0] = total;
  }
  __syncthreads();
  const float inv = rsqrtf(shared.sums[0] / Width + eps);
  for (int c = t * 8; c < Width; c += Warps * 32 * 8) {
    const uint4 packed = *reinterpret_cast<const uint4*>(input + base + c);
    const uint4 wpacked = *reinterpret_cast<const uint4*>(weight + c);
    const uint16_t* in = reinterpret_cast<const uint16_t*>(&packed);
    const uint16_t* weights = reinterpret_cast<const uint16_t*>(&wpacked);
    uint4 result;
    uint16_t* out = reinterpret_cast<uint16_t*>(&result);
    for (int j = 0; j < 8; ++j) {
      const float n = bf(in[j]) * inv;
      out[j] = to_bf(n * bf(weights[j]));
    }
    *reinterpret_cast<uint4*>(shared.normed + c) = result;
    *reinterpret_cast<uint4*>(normed + base + c) = result;
    if (widened) for (int j = 0; j < 8; ++j) widened[base + c + j] = bf(out[j]);
  }
  __syncthreads();
  constexpr int Slots = (Width / 32 + Warps - 1) / Warps;
  uint16_t saved_codes[Slots];
  uint8_t saved_scales[Slots];
  uint32_t flags = 0;
  #pragma unroll
  for (int slot = 0; slot < Slots; ++slot) {
    const uint32_t group = slot * Warps + warp;
    if (group < Width / 32) {
      const auto value = block_quantize_parallel::quantize_group<0,false,false,FloorBits>(
          shared.normed, uint64_t(group) * 32 + 2 * lane);
      flags |= value.flags;
      saved_codes[slot] = value.codes;
      saved_scales[slot] = value.scale_code;
    }
  }
  // All groups report before any codes or scales become observable. A bad
  // group still zeros the whole row, including groups with finite inputs.
  flags = block_quantize_parallel::row_status<Warps>(flags,shared.flags,status,status_row_offset+row);
  #pragma unroll
  for (int slot = 0; slot < Slots; ++slot) {
    const uint32_t group = slot * Warps + warp;
    if (group < Width / 32) {
      const uint16_t value = flags == 0 ? saved_codes[slot] : 0;
      if (lane == 0) scales[uint64_t(row) * (Width / 32) + group] = flags == 0 ? saved_scales[slot] : 127;
      if (lane < 16) {
        const uint64_t at = base + uint64_t(group) * 32 + 2 * lane;
        codes[at] = uint8_t(value);
        codes[at + 1] = uint8_t(value >> 8);
      }
    }
  }
}
} // namespace chop::row::rmsnorm_quantize_cached
