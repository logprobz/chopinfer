#pragma once

#include "row.cuh"

namespace chop::row {

__device__ __forceinline__ float ue4m3_decode(int code) {
  int exponent = code >> 3, mantissa = code & 7;
  if (exponent == 0) return mantissa * (1.f / 512.f);
  return (1.f + mantissa * (1.f / 8.f)) * exp2f(float(exponent - 7));
}

__device__ __forceinline__ uint8_t ue4m3_encode(float x) {
  if (x <= 0.f) return 0;
  int best = 0;
  double error = INFINITY;
  for (int code = 0; code < 127; ++code) {
    double distance = fabs(double(x) - double(ue4m3_decode(code)));
    if (distance < error ||
        (distance == error && (code & 1) == 0 && (best & 1) != 0)) {
      error = distance;
      best = code;
    }
  }
  return uint8_t(best);
}

// Format 0/1: E4M3 with exact/power-of-two F32 scale per 128.
// Format 2: MXFP8. Format 3: MXFP4. Format 4: NVFP4.
template<int Width, int Format, bool AllGroups = false, bool NoPdl = false>
__device__ void quantize(const uint16_t* input, uint8_t* codes,
                         void* scales, const float* global_scale,
                         const uint32_t* live_rows) {
  if constexpr (!NoPdl) ptx::griddep_launch_dependents();
  constexpr int Group = Format < 2 ? 128 : (Format == 4 ? 16 : 32);
  constexpr int Groups = (Width + Group - 1) / Group;
  __shared__ float maxima[256];
  if constexpr (!NoPdl) ptx::griddep_wait();
  uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  for (uint32_t group = AllGroups ? 0 : blockIdx.y;
       group < (AllGroups ? Groups : blockIdx.y + 1) && group < Groups; ++group) {
  const uint32_t begin_col = group * Group;
  const uint32_t end_col = min(begin_col + Group, uint32_t(Width));
  float maximum = 0.f;
  for (uint32_t c = begin_col + threadIdx.x * 8; c < end_col; c += blockDim.x * 8) {
    uint64_t index = uint64_t(row) * Width + c;
    if (c + 8 <= end_col) {
      uint4 packed = reinterpret_cast<const uint4*>(input + index)[0];
      const uint16_t* half = reinterpret_cast<const uint16_t*>(&packed);
      for (int j = 0; j < 8; ++j) maximum = fmaxf(maximum, fabsf(bf(half[j])));
    } else {
      for (int j = 0; j < 8 && c + j < end_col; ++j)
        maximum = fmaxf(maximum, fabsf(bf(input[index + j])));
    }
  }
  maxima[threadIdx.x] = maximum;
  __syncthreads();
  for (int step = blockDim.x / 2; step; step >>= 1) {
    if (threadIdx.x < step) maxima[threadIdx.x] = fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + step]);
    __syncthreads();
  }
  maximum = maxima[0];
  float scale = 1.f;
  if constexpr (Format < 2) {
    if (maximum != 0.f) {
      scale = maximum / 448.f;
      if constexpr (Format == 1) scale = exp2f(ceilf(log2f(scale)));
    }
    if (threadIdx.x == 0)
      reinterpret_cast<float*>(scales)[uint64_t(row) * ((Width + 127) / 128) + group] = scale;
  } else if constexpr (Format == 2 || Format == 3) {
    int exponent = maximum == 0.f ? 127 : max(0, min(254, ilogbf(maximum) - (Format == 2 ? 8 : 2) + 127));
    if (threadIdx.x == 0)
      reinterpret_cast<uint8_t*>(scales)[uint64_t(row) * (Width / 32) + group] = uint8_t(exponent);
    scale = exp2f(float(exponent - 127));
  } else {
    scale = *global_scale;
    uint8_t local = maximum == 0.f ? 0 : ue4m3_encode((maximum / 6.f) / scale);
    if (threadIdx.x == 0)
      reinterpret_cast<uint8_t*>(scales)[uint64_t(row) * (Width / 16) + group] = local;
    scale *= ue4m3_decode(local);
  }
  for (uint32_t c = begin_col + threadIdx.x * 2; c < end_col; c += blockDim.x * 2) {
    uint64_t index = uint64_t(row) * Width + c;
    float x = bf(input[index]);
    float y = c + 1 < end_col ? bf(input[index + 1]) : 0.f;
    if constexpr (Format < 3) {
      uint16_t packed = convert::pack_f32_e4m3({float(double(x) / double(scale)), float(double(y) / double(scale))});
      codes[index] = uint8_t(packed);
      if (c + 1 < end_col) codes[index + 1] = uint8_t(packed >> 8);
    } else {
      float a = maximum == 0.f ? 0.f :
                (Format == 4 ? x / scale : float(double(x) / double(scale)));
      float b = maximum == 0.f ? 0.f :
                (Format == 4 ? y / scale : float(double(y) / double(scale)));
      uint16_t packed = convert::pack_f32_e2m1({a, b});
      codes[index / 2] = uint8_t(packed);
    }
  }
  __syncthreads();
  }
}

// Eight independent 32-value groups per warp. Four lanes load one aligned
// 128-bit transaction each, then encode and store eight contiguous bytes.
template<int Width>
__device__ void quantize_mxfp8_rows(const uint16_t* input, uint8_t* codes,
                                    uint8_t* scales, const uint32_t* live_rows) {
  begin();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  constexpr int Groups = Width / 32;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int subgroup = lane >> 2;
  const int lane_in_group = lane & 3;
  for (int group = warp * 8 + subgroup; group < Groups; group += 64) {
    const uint64_t index = uint64_t(row) * Width + group * 32 + lane_in_group * 8;
    uint4 packed = reinterpret_cast<const uint4*>(input + index)[0];
    const uint16_t* values = reinterpret_cast<const uint16_t*>(&packed);
    float maximum = 0.f;
    for (int j = 0; j < 8; ++j)
      maximum = fmaxf(maximum, fabsf(bf(values[j])));
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 2));
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 1));
    const int exponent = maximum == 0.f ? 127 :
        max(0, min(254, ilogbf(maximum) - 8 + 127));
    if (lane_in_group == 0)
      scales[uint64_t(row) * Groups + group] = uint8_t(exponent);
    const float scale = exp2f(float(exponent - 127));
    uint64_t encoded = 0;
    for (int j = 0; j < 8; j += 2) {
      uint16_t pair = convert::pack_f32_e4m3({
          float(double(bf(values[j])) / double(scale)),
          float(double(bf(values[j + 1])) / double(scale))});
      encoded |= uint64_t(pair) << (j * 8);
    }
    reinterpret_cast<uint64_t*>(codes + index)[0] = encoded;
  }
}

template<int Width, int Kind, bool Residual, int Format>
__device__ void rmsnorm_quant(const uint16_t* x, const uint16_t* weight,
                              const uint16_t* residual, uint16_t* residual_out,
                              uint16_t* y, uint8_t* codes, void* scales,
                              const float* global_scale, const uint32_t* live_rows,
                              float eps) {
  rmsnorm<Width,0,Kind,Residual>(x,weight,residual,residual_out,y,live_rows,eps);
  __syncthreads();
  if constexpr (Format < 2) {
    const uint32_t row = blockIdx.x;
    if (row >= *live_rows) return;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int half_warp = lane >> 4;
    const int local_lane = lane & 15;
    constexpr int Groups = Width / 128;
    for (int group = warp * 2 + half_warp; group < Groups; group += 16) {
      const uint64_t index = uint64_t(row) * Width + group * 128 + local_lane * 8;
      const uint4 packed = reinterpret_cast<const uint4*>(y + index)[0];
      const uint16_t* values = reinterpret_cast<const uint16_t*>(&packed);
      float maximum = 0.f;
      for (int j = 0; j < 8; ++j) maximum = fmaxf(maximum, fabsf(bf(values[j])));
      for (int step = 8; step; step >>= 1)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, step, 16));
      float scale = maximum == 0.f ? 1.f : maximum / 448.f;
      if constexpr (Format == 1)
        if (maximum != 0.f) scale = exp2f(ceilf(log2f(scale)));
      if (local_lane == 0)
        reinterpret_cast<float*>(scales)[uint64_t(row) * Groups + group] = scale;
      uint64_t encoded = 0;
      for (int j = 0; j < 8; j += 2) {
        const uint16_t pair = convert::pack_f32_e4m3({
            float(double(bf(values[j])) / double(scale)),
            float(double(bf(values[j + 1])) / double(scale))});
        encoded |= uint64_t(pair) << (j * 8);
      }
      reinterpret_cast<uint64_t*>(codes + index)[0] = encoded;
    }
  } else {
    quantize<Width,Format,true,true>(y,codes,scales,global_scale,live_rows);
  }
}

template<int Width>
__device__ void nvfp4_row_amax(const uint16_t* input, float* row_amax,
                               const uint32_t* live_rows) {
  ptx::griddep_launch_dependents();
  __shared__ float partial[256];
  ptx::griddep_wait();
  uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  float maximum = 0.f;
  for (int c = threadIdx.x; c < Width; c += blockDim.x)
    maximum = fmaxf(maximum, fabsf(bf(input[uint64_t(row) * Width + c])));
  partial[threadIdx.x] = maximum;
  __syncthreads();
  for (int step = blockDim.x / 2; step; step >>= 1) {
    if (threadIdx.x < step) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + step]);
    __syncthreads();
  }
  if (threadIdx.x == 0) row_amax[row] = partial[0];
}

__device__ void nvfp4_global_scale(const float* row_amax, float* scale,
                                   const uint32_t* live_rows) {
  ptx::griddep_launch_dependents();
  __shared__ float partial[256];
  ptx::griddep_wait();
  float maximum = 0.f;
  for (uint32_t row = threadIdx.x; row < *live_rows; row += blockDim.x)
    maximum = fmaxf(maximum, row_amax[row]);
  partial[threadIdx.x] = maximum;
  __syncthreads();
  for (int step = blockDim.x / 2; step; step >>= 1) {
    if (threadIdx.x < step) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + step]);
    __syncthreads();
  }
  if (threadIdx.x == 0) *scale = partial[0] == 0.f ? 1.f : partial[0] / 2688.f;
}

} // namespace chop::row
