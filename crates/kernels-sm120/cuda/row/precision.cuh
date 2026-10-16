#pragma once
#include <cuda_bf16.h>
#include "row.cuh"

namespace chop::row::precision {

template<uint32_t Columns, uint32_t SourceRow, uint32_t SourceColumn, uint32_t SourceOffset,
         uint32_t DestRow, uint32_t DestColumn, uint32_t DestOffset>
__device__ void copy_rows(const uint16_t* input, uint16_t* output, const uint32_t* live_rows) {
  begin();
  const uint64_t row = blockIdx.x;
  if (row >= *live_rows) return;
  for (uint32_t c = threadIdx.x; c < Columns; c += blockDim.x)
    output[row * DestRow + uint64_t(c) * DestColumn + DestOffset] =
        input[row * SourceRow + uint64_t(c) * SourceColumn + SourceOffset];
}

// Codes and local scales are row-major. IDs and outputs interleave Heads.
// Only selected rows are decoded, including when the table is mapped host memory.
template<uint32_t Width, uint32_t Heads, uint64_t Vocab, int Format, int Global, typename Id>
__device__ void lookup(const uint8_t* table, const Id* ids, const uint8_t* local_scales,
                      const void* global_scale, uint16_t* output,
                      const uint32_t* live_rows, uint64_t row_offset, uint64_t shard_rows) {
  begin();
  if (blockIdx.x >= *live_rows) return;
  const uint64_t row = uint64_t(blockIdx.x) * Heads + blockIdx.y;
  const uint64_t id = uint64_t(ids[row]);
  const bool valid = id < Vocab && id >= row_offset && id - row_offset < shard_rows;
  const uint64_t local = id - row_offset;
  float global = 1.f;
  if constexpr (Global == 1) global = bf(*reinterpret_cast<const uint16_t*>(global_scale));
  if constexpr (Global == 2) global = *reinterpret_cast<const float*>(global_scale);
  for (uint32_t c = threadIdx.x; c < Width; c += blockDim.x) {
    uint16_t value = 0;
    if (valid) {
      if constexpr (Format == 0) {
        value = reinterpret_cast<const uint16_t*>(table)[local * Width + c];
      } else if constexpr (Format == 1) {
        const float weight = convert::unpack_f32_e4m3(uint16_t(table[local * Width + c])).lo;
        value = to_bf(weight * global);
      } else {
        const uint8_t packed = table[local * (Width / 2) + c / 2];
        const uint8_t code = (c & 1) ? packed >> 4 : packed & 15;
        const float weight = convert::unpack_f32_e2m1(uint16_t(code)).lo;
        const float scale = convert::unpack_f32_e4m3(uint16_t(local_scales[local * (Width / 16) + c / 16])).lo;
        value = __bfloat16_as_ushort(__double2bfloat16(double(weight * scale) * double(global)));
      }
    }
    output[row * Width + c] = value;
  }
}

} // namespace chop::row::precision
