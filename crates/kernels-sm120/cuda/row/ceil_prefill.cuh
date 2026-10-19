#pragma once
#include "ceil_parallel.cuh"

// Prefill uses a bounded warp count per row. Group arithmetic and row
// admission are the shared parallel implementation's; only scheduling
// changes. Keeping this entry separate preserves the decode cubins.
namespace chop::row::block_quantize_prefill {
using namespace block_quantize_parallel;

template<int Width, int Format, bool F32Input, bool Fake, uint32_t FloorBits, int Warps>
__device__ void run(const void* input, void* output, uint8_t* scales,
                    const uint32_t* live_rows, uint32_t* status, uint64_t status_row_offset, uint32_t launch_width = 0) {
  static_assert(Width > 0 && Format == 0 && !Fake);
  static_assert(Warps == 4 || Warps == 8 || Warps == 16);
  __shared__ uint32_t shared_flags[Warps];
  constexpr uint32_t Group = Format == 1 ? 16 : 32;
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) { ptx::griddep_launch_dependents(); return; }
  const uint32_t width = Width ? Width : launch_width;
  const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  uint32_t flags = 0;
#pragma unroll 1
  for (uint32_t group = warp; group < width / Group; group += Warps) {
    const uint64_t at = uint64_t(row) * width + uint64_t(group) * Group + 2 * lane;
    flags |= block_quantize_parallel::quantize_group<Format,F32Input,Fake,FloorBits>(input,at).flags;
  }
  flags = block_quantize_parallel::row_status<Warps>(flags, shared_flags, status, status_row_offset + row);
#ifdef CHOP_CEIL_PREFILL_IGNORE_ROW_FLAGS
  // Lab-only negative control: valid groups escape a rejected row.
  flags = 0;
#endif
#pragma unroll 1
  for (uint32_t group = warp; group < width / Group; group += Warps) {
    const uint64_t at = uint64_t(row) * width + uint64_t(group) * Group + 2 * lane;
    GroupValue value;
    if (flags == 0) value = block_quantize_parallel::quantize_group<Format,F32Input,Fake,FloorBits>(input,at);
    else value.scale_code = Format == 1 ? 0x38 : 127; // scale 1
    if constexpr (!Fake) {
      if (lane == 0) scales[uint64_t(row) * (width / Group) + group] = value.scale_code;
    }
    if (lane < Group / 2) {
      if constexpr (Fake) {
        reinterpret_cast<uint16_t*>(output)[at] = uint16_t(value.pair);
        reinterpret_cast<uint16_t*>(output)[at+1] = uint16_t(value.pair >> 16);
      } else if constexpr (Format == 0) {
        reinterpret_cast<uint8_t*>(output)[at] = uint8_t(value.codes);
        reinterpret_cast<uint8_t*>(output)[at+1] = uint8_t(value.codes >> 8);
      } else reinterpret_cast<uint8_t*>(output)[at/2] = uint8_t(value.codes);
    }
  }
  ptx::griddep_launch_dependents();
}
} // namespace chop::row::block_quantize_prefill
