#pragma once
#include "row.cuh"
#include "ceil_parallel.cuh"

// Preserve rmsnorm_f32's pack-of-eight sums and 256-thread tree. Keep the
// F32 norm output and the quantizer's F32 -> BF16 -> E4M3 rounding points.
namespace chop::row::rmsnorm_f32_prefill {

enum class Mutation { none, round_before_weight, clear_status };

template<int Width>
struct Arena {
  float sums[8];
  uint32_t flags[8];
  __align__(16) float values[Width];
  __align__(16) uint8_t codes[Width];
  uint8_t scales[Width / 32];
};

// The task owns one virtual row and its arena. Entry wrappers own PDL.
template<int Width, uint32_t FloorBits, bool CacheInput, Mutation M>
__device__ void task(uint32_t row, Arena<Width>& arena,
    const float* x, const uint16_t* weight, float* y,
    uint8_t* codes, uint8_t* scales, const uint32_t* live_rows,
    uint32_t* status, uint64_t status_row_offset, float eps) {
  static_assert(Width > 0 && Width % 32 == 0);
  if (row >= *live_rows) return;
  const uint64_t base = uint64_t(row) * Width;
  if constexpr (M == Mutation::clear_status) {
    if (threadIdx.x == 0) status[status_row_offset + row] = 0;
    __syncthreads();
  }
  float square = 0.f;
  uint32_t flags = 0;
  for (int c = threadIdx.x * 8; c < Width; c += 256 * 8) {
    for (int j = 0; j < 8; ++j) {
      const float v = x[base + c + j];
      if (!isfinite(v) || !isfinite(bf(weight[c+j]))) flags |= 1;
      square = __fmaf_rn(v, v, square);
      if constexpr (CacheInput) arena.values[c+j] = v;
    }
  }
  const float total = sum_cta(square, arena.sums);
  const float inv = rsqrtf(total / Width + eps);
  for (int c = threadIdx.x; c < Width; c += 256) {
    const float v = CacheInput ? arena.values[c] : x[base+c];
    float normalized = __fmul_rn(v, inv);
    if constexpr (M == Mutation::round_before_weight)
      normalized = bf(to_bf(normalized));
    const float value = __fmul_rn(normalized, bf(weight[c]));
    if (!isfinite(total) || !isfinite(inv) || !isfinite(value)) flags |= 4;
    arena.values[c] = value;
  }
  // Input failures take precedence over arithmetic failures in rmsnorm_f32.
  if (__syncthreads_or((flags & 1) != 0)) flags = 1;
  const uint32_t failed = block_quantize_parallel::row_status<8>(
      flags, arena.flags, status, status_row_offset + row);
  for (int c = threadIdx.x; c < Width; c += 256) {
    const float value = failed ? 0.f : arena.values[c];
    arena.values[c] = value;
    y[base+c] = value;
  }
  __syncthreads();
  const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  flags = 0;
  #pragma unroll 1
  for (uint32_t group = warp; group < Width / 32; group += 8) {
    const uint32_t at = group * 32 + 2 * lane;
    const auto value = block_quantize_parallel::quantize_group<0,true,false,FloorBits>(
        arena.values, at);
    flags |= value.flags;
    if (lane == 0) arena.scales[group] = value.scale_code;
    if (lane < 16) {
      arena.codes[at] = uint8_t(value.codes);
      arena.codes[at+1] = uint8_t(value.codes >> 8);
    }
  }
  flags = block_quantize_parallel::row_status<8>(
      flags, arena.flags, status, status_row_offset + row);
  for (int c = threadIdx.x * 16; c < Width; c += 256 * 16)
    *reinterpret_cast<uint4*>(codes + base + c) = flags ? make_uint4(0,0,0,0)
        : *reinterpret_cast<const uint4*>(arena.codes + c);
  for (int group = threadIdx.x; group < Width / 32; group += 256)
    scales[uint64_t(row) * (Width / 32) + group] = flags ? 127 : arena.scales[group];
}

template<int Width, uint32_t FloorBits, bool CacheInput, Mutation M>
__device__ void run(const void* input, const uint16_t* weight, void* normed,
    float* widened, uint8_t* codes, uint8_t* scales, const uint32_t* live_rows,
    uint32_t* status, uint64_t status_row_offset, float eps) {
  if (blockDim.x != 256 || widened) __trap();
  __shared__ Arena<Width> arena;
  ptx::griddep_wait();
  task<Width,FloorBits,CacheInput,M>(blockIdx.x, arena,
      static_cast<const float*>(input), weight, static_cast<float*>(normed),
      codes, scales, live_rows, status, status_row_offset, eps);
  ptx::griddep_launch_dependents();
}

} // namespace chop::row::rmsnorm_f32_prefill
