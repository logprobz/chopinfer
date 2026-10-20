#pragma once
#include "row.cuh"
#include "ceil_parallel.cuh"
#include "../common/row_status.cuh"

// Cached form of rmsnorm_quantize::run<Width, true, FloorBits> (rmsnorm_f32
// then ceil_epilogue) for F32 rows. The served entry stores the normalized
// row, then each warp reloads its groups from global memory twice, once for
// the row status and once for its outputs, in serial loops. Here the first
// 256 lanes keep their eight input values in registers, run rmsnorm_f32's
// expressions and RMS tree, and publish the stored row to shared memory as
// well; every warp then quantizes its groups once from shared memory. The
// norm and ceil status commits keep their order and atomic OR protocol.
namespace chop::row::rmsnorm_quantize_cached_f32 {
template<int Width, int Warps> struct Arena {
  float sums[8];
  uint32_t flags[Warps];
  alignas(16) float normed[Width];
};

template<int Width, int Warps, uint32_t FloorBits>
__device__ void run(const float* input, const uint16_t* weight, float* normed, uint8_t* codes,
                    uint8_t* scales, const uint32_t* live_rows, uint32_t* status,
                    uint64_t status_row_offset, float eps, unsigned char* arena) {
  static_assert(Width > 0 && Width % 32 == 0 && Width <= 2048);
  static_assert(Warps == 8 || Warps == 16 || Warps == 32);
  // rmsnorm_f32's thread t owns columns 8t..8t+7 at these widths; only the
  // first 256 lanes accumulate, so the tree matches the served 256-thread CTA.
  const uint32_t t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const int c = int(t) * 8;
  const bool owner = t < 256 && c < Width;
  auto& shared = *reinterpret_cast<Arena<Width, Warps>*>(arena);
  if ((reinterpret_cast<uintptr_t>(input) | reinterpret_cast<uintptr_t>(weight) |
       reinterpret_cast<uintptr_t>(normed)) & 15)
    __trap();
  // The weight is a model parameter, never a producer output.
  const uint4 weights = owner ? *reinterpret_cast<const uint4*>(weight + c) : uint4{};
  const uint16_t* w = reinterpret_cast<const uint16_t*>(&weights);
  ptx::griddep_wait();
  const uint32_t row = blockIdx.y;
  if (row >= *live_rows) { ptx::griddep_launch_dependents(); return; }
  const uint64_t base = uint64_t(row) * Width;
  float values[8];
  float square = 0.f;
  uint32_t flags = 0;
  if (owner) {
    const float4 lo = *reinterpret_cast<const float4*>(input + base + c);
    const float4 hi = *reinterpret_cast<const float4*>(input + base + c + 4);
    const float v[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      values[j] = v[j];
      if (!isfinite(v[j]) || !isfinite(bf(w[j]))) flags |= 1;
      square = __fmaf_rn(v[j], v[j], square);
    }
  }
  // sum_cta over the first eight warps, as the served 256-thread CTA.
  if (t < 256) {
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
  const float total = shared.sums[0];
  const float inv = rsqrtf(total / Width + eps);
  if (owner) {
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      values[j] = __fmul_rn(__fmul_rn(values[j], inv), bf(w[j]));
      if (!isfinite(total) || !isfinite(inv) || !isfinite(values[j])) flags |= 4;
    }
  }
  if (__syncthreads_or((flags & 1) != 0)) flags = 1;
  const uint32_t failed =
      chop::row_status::commit<block_quantize::StatusOr>(flags, status, status_row_offset + row);
  if (owner) {
    float y[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) y[j] = failed ? 0.f : values[j];
    const float4 lo = make_float4(y[0], y[1], y[2], y[3]);
    const float4 hi = make_float4(y[4], y[5], y[6], y[7]);
    *reinterpret_cast<float4*>(shared.normed + c) = lo;
    *reinterpret_cast<float4*>(shared.normed + c + 4) = hi;
    *reinterpret_cast<float4*>(normed + base + c) = lo;
    *reinterpret_cast<float4*>(normed + base + c + 4) = hi;
  }
  __syncthreads();
  constexpr int Groups = Width / 32;
  constexpr int Slots = (Groups + Warps - 1) / Warps;
  uint32_t packed[Slots];
  uint32_t group_flags = 0;
  #pragma unroll
  for (int slot = 0; slot < Slots; ++slot) {
    const uint32_t group = slot * Warps + warp;
    if (group < Groups) {
      const auto value = block_quantize_parallel::quantize_group<0, true, false, FloorBits>(
          shared.normed, uint64_t(group) * 32 + 2 * lane);
      group_flags |= value.flags;
      packed[slot] = uint32_t(value.codes) | uint32_t(value.scale_code) << 16;
    }
  }
  // All groups report before any codes or scales become observable; a
  // flagged row stores ceil_epilogue's default codes and the unit scale.
  group_flags = block_quantize_parallel::row_status<Warps>(group_flags, shared.flags, status,
                                                           status_row_offset + row);
  #pragma unroll
  for (int slot = 0; slot < Slots; ++slot) {
    const uint32_t group = slot * Warps + warp;
    if (group < Groups) {
      const uint32_t value = group_flags == 0 ? packed[slot] : 127u << 16;
      if (lane == 0) scales[uint64_t(row) * Groups + group] = uint8_t(value >> 16);
      if (lane < 16) {
        const uint64_t at = base + uint64_t(group) * 32 + 2 * lane;
        codes[at] = uint8_t(value);
        codes[at + 1] = uint8_t(value >> 8);
      }
    }
  }
  ptx::griddep_launch_dependents();
}
} // namespace chop::row::rmsnorm_quantize_cached_f32
