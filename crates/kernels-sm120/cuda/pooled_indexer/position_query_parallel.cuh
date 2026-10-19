#pragma once
#include "../row/ceil_parallel.cuh"

namespace chop::position_query_parallel {

// A warp owns each head's groups. Rotation uses the original separate RN
// multiply/add operations. Quantized pairs stay in registers until every
// group has contributed its flags and the row has been admitted once.
template<int Heads, int Dim, int Rotary, bool IgnoreRowStatus = false>
__device__ void run(const float* input, const float2* cos_sin,
    const int32_t* positions, const uint32_t* live_rows, uint16_t* output,
    uint32_t* status, uint64_t status_row_offset, uint32_t table_positions) {
  static_assert(Heads == 32 && Dim == 128 && Rotary == 64);
  constexpr int Width = Heads * Dim, Warps = 32, Groups = Width / 32 / Warps;
  __shared__ float rotated[Width];
  __shared__ uint32_t shared_flags[Warps];
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) { ptx::griddep_launch_dependents(); return; }
  const int32_t position = positions[row];
  const bool invalid = position < 0 || uint32_t(position) >= table_positions;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int pair = threadIdx.x; pair < Width / 2; pair += blockDim.x) {
    const int channel = (pair * 2) % Dim;
    float a = 0.f, b = 0.f;
    if (!invalid) {
      a = input[uint64_t(row) * Width + 2 * pair];
      b = input[uint64_t(row) * Width + 2 * pair + 1];
      if (channel >= Dim - Rotary) {
        const float2 cs = cos_sin[size_t(position) * (Rotary / 2) + (channel - (Dim - Rotary)) / 2];
        const float ra = __fsub_rn(__fmul_rn(a, cs.x), __fmul_rn(b, cs.y));
        b = __fadd_rn(__fmul_rn(a, cs.y), __fmul_rn(b, cs.x));
        a = ra;
      }
    }
    rotated[2 * pair] = a;
    rotated[2 * pair + 1] = b;
  }
  __syncthreads();
  row::block_quantize::GroupValue values[Groups];
  uint32_t flags = invalid ? 8u : 0u;
#pragma unroll
  for (int i = 0; i < Groups; ++i) {
    const int group = warp * Groups + i;
    values[i] = row::block_quantize_parallel::quantize_group<2,true,true,0>(rotated, group * 32 + 2 * lane);
    flags |= values[i].flags;
  }
  flags = row::block_quantize_parallel::row_status<Warps>(
      flags, shared_flags, status, status_row_offset + row);
  if constexpr (IgnoreRowStatus) flags = 0; // Memory-safe lab negative control.
#pragma unroll
  for (int i = 0; i < Groups; ++i) {
    if (lane < 16) {
      const uint64_t at = uint64_t(row) * Width + (warp * Groups + i) * 32 + 2 * lane;
      const uint32_t pair = flags ? 0u : values[i].pair;
      output[at] = uint16_t(pair);
      output[at + 1] = uint16_t(pair >> 16);
    }
  }
  ptx::griddep_launch_dependents();
}
} // namespace chop::position_query_parallel
