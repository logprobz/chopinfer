#pragma once
// Row stages whose shapes are launch arguments: widths, strides, heads, the
// vocabulary bound and the routed-row divisor. Format, global-scale recipe,
// ID type, tiles and accumulation order stay compile-time. Each function
// repeats the loops and arithmetic of its compile-time template line for
// line (precision.cuh, quantize.cuh, router.cuh), so both forms write the
// same bits; kernel-lab's projection_row_shapes test compares them.
#include <cuda_bf16.h>
#include "row.cuh"
#include "quantize.cuh"

namespace chop::row::precision::launch {

// precision::copy_rows with its columns and (row, column, offset) strides.
__device__ void copy_rows(const uint16_t* input, uint16_t* output, const uint32_t* live_rows,
                          uint32_t columns, uint32_t source_row, uint32_t source_column,
                          uint32_t source_offset, uint32_t dest_row, uint32_t dest_column,
                          uint32_t dest_offset) {
  begin();
  const uint64_t row = blockIdx.x;
  if (row >= *live_rows) return;
  for (uint32_t c = threadIdx.x; c < columns; c += blockDim.x)
    output[row * dest_row + uint64_t(c) * dest_column + dest_offset] =
        input[row * source_row + uint64_t(c) * source_column + source_offset];
}

// precision::lookup with its width, heads and vocabulary bound.
template<int Format, int Global, typename Id>
__device__ void lookup(const uint8_t* table, const Id* ids, const uint8_t* local_scales,
                      const void* global_scale, uint16_t* output,
                      const uint32_t* live_rows, uint64_t row_offset, uint64_t shard_rows,
                      uint32_t width, uint32_t heads, uint64_t vocab) {
  begin();
  if (blockIdx.x >= *live_rows) return;
  const uint64_t row = uint64_t(blockIdx.x) * heads + blockIdx.y;
  const uint64_t id = uint64_t(ids[row]);
  const bool valid = id < vocab && id >= row_offset && id - row_offset < shard_rows;
  const uint64_t local = id - row_offset;
  float global = 1.f;
  if constexpr (Global == 1) global = bf(*reinterpret_cast<const uint16_t*>(global_scale));
  if constexpr (Global == 2) global = *reinterpret_cast<const float*>(global_scale);
  for (uint32_t c = threadIdx.x; c < width; c += blockDim.x) {
    uint16_t value = 0;
    if (valid) {
      if constexpr (Format == 0) {
        value = reinterpret_cast<const uint16_t*>(table)[local * width + c];
      } else if constexpr (Format == 1) {
        const float weight = convert::unpack_f32_e4m3(uint16_t(table[local * width + c])).lo;
        value = to_bf(weight * global);
      } else {
        const uint8_t packed = table[local * (width / 2) + c / 2];
        const uint8_t code = (c & 1) ? packed >> 4 : packed & 15;
        const float weight = convert::unpack_f32_e2m1(uint16_t(code)).lo;
        const float scale = convert::unpack_f32_e4m3(uint16_t(local_scales[local * (width / 16) + c / 16])).lo;
        value = __bfloat16_as_ushort(__double2bfloat16(double(weight * scale) * double(global)));
      }
    }
    output[row * width + c] = value;
  }
}

// row::quantize<Width, Format> (one group per CTA, PDL) with its width.
template<int Format>
__device__ void quantize(const uint16_t* input, uint8_t* codes,
                         void* scales, const float* global_scale,
                         const uint32_t* live_rows, uint32_t width_argument) {
  const int Width = int(width_argument);
  ptx::griddep_launch_dependents();
  constexpr int Group = Format < 2 ? 128 : (Format == 4 ? 16 : 32);
  const int Groups = (Width + Group - 1) / Group;
  __shared__ float maxima[256];
  ptx::griddep_wait();
  uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  for (uint32_t group = blockIdx.y; group < blockIdx.y + 1 && group < Groups; ++group) {
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

// row::quantize_mxfp8_rows<Width> with its width.
__device__ void quantize_mxfp8_rows(const uint16_t* input, uint8_t* codes,
                                    uint8_t* scales, const uint32_t* live_rows,
                                    uint32_t width_argument) {
  const int Width = int(width_argument);
  begin();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  const int Groups = Width / 32;
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

// row::nvfp4_row_amax<Width> with its width.
__device__ void nvfp4_row_amax(const uint16_t* input, float* row_amax,
                               const uint32_t* live_rows, uint32_t width_argument) {
  const int Width = int(width_argument);
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

// row::gather<Width, TopK> with its width and routed-row divisor.
__device__ void gather(const uint16_t* input, const int32_t* sorted_rows,
                       uint16_t* output, const int32_t* padded_rows,
                       uint32_t width_argument, uint32_t divisor) {
  const int Width = int(width_argument);
  const int TopK = int(divisor);
  begin();
  uint32_t row = blockIdx.x;
  if (row >= uint32_t(*padded_rows)) return;
  int32_t route = sorted_rows[row];
  for (int c = threadIdx.x * 8; c < Width; c += blockDim.x * 8) {
    const uint64_t dst = uint64_t(row) * Width + c;
    if (c + 8 <= Width) {
      uint4 v = route < 0 ? make_uint4(0, 0, 0, 0) :
                reinterpret_cast<const uint4*>(input + uint64_t(route / TopK) * Width + c)[0];
      reinterpret_cast<uint4*>(output + dst)[0] = v;
    } else {
      for (int j = 0; j < 8 && c + j < Width; ++j)
        output[dst + j] = route < 0 ? 0 : input[uint64_t(route / TopK) * Width + c + j];
    }
  }
}

// row::gather_mxfp8<Width, TopK> with its width and routed-row divisor.
__device__ void gather_mxfp8(const uint16_t* input, const int32_t* sorted_rows,
                             uint8_t* codes, uint8_t* scales,
                             const int32_t* padded_rows,
                             uint32_t width_argument, uint32_t divisor) {
  const int Width = int(width_argument);
  const int TopK = int(divisor);
  begin();
  uint32_t row = blockIdx.x;
  if (row >= uint32_t(*padded_rows)) return;
  int32_t route = sorted_rows[row];
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int group = warp; group < Width / 32; group += blockDim.x / 32) {
    uint64_t index = uint64_t(row) * Width + group * 32 + lane;
    float value = route < 0 ? 0.f :
        bf(input[uint64_t(route / TopK) * Width + group * 32 + lane]);
    float maximum = fabsf(value);
    for (int offset = 16; offset; offset >>= 1)
      maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
    int code = maximum == 0.f ? 127 : max(0, min(254, ilogbf(maximum) - 8 + 127));
    if (lane == 0) scales[uint64_t(row) * (Width / 32) + group] = uint8_t(code);
    float scale = exp2f(float(code - 127));
    uint16_t packed = convert::pack_f32_e4m3({float(double(value) / double(scale)), 0.f});
    codes[index] = uint8_t(packed);
  }
}

} // namespace chop::row::precision::launch
