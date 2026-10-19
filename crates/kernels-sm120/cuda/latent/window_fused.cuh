#pragma once
#include "../common/ptx.cuh"

// Only fused entries include this header. Keep the original norm, rotation,
// and store bodies, but publish their completion after the final store.
namespace chop::ptx {
__device__ __forceinline__ void window_fused_defer_dependents() {}
}
#define griddep_launch_dependents window_fused_defer_dependents
#include "glue.cuh"
#undef griddep_launch_dependents
#include "../row/ceil_epilogue.cuh"

namespace chop::latent::window_fused {
template<bool Bf16>
__device__ void run(const float* input, const uint16_t* weight, float* latent,
    const float* cos_sin, const uint32_t* live_rows, uint32_t eps_bits,
    uint32_t* status, uint64_t status_row_offset,
    uint8_t* quantized_codes, uint8_t* quantized_scales,
    const int32_t* positions, const int32_t* row_batch,
    const int32_t* first_positions, const int32_t* last_positions,
    uint8_t* codes, uint8_t* scales, uint32_t sequences, uint32_t slots,
    uint32_t circular, uint64_t code_stride, uint64_t scale_stride) {
  ptx::griddep_wait();
  if (blockIdx.x >= *live_rows) { ptx::griddep_launch_dependents(); return; }
  glue::norm_rotate(input,weight,latent,cos_sin,live_rows,__uint_as_float(eps_bits),status,status_row_offset);
  __syncthreads();
  // The fixed floor is the existing latent-window contract's 1e-4.
  chop::row::ceil_epilogue<512,true,0x38d1b717u>(latent,quantized_codes,quantized_scales,status,status_row_offset);
  __syncthreads();
  if constexpr (Bf16)
    glue::store_bf16<>(quantized_codes,quantized_scales,positions,row_batch,
        first_positions,last_positions,codes,scales,live_rows,sequences,slots,
        circular,code_stride,scale_stride,status,status_row_offset);
  else
    glue::store(quantized_codes,quantized_scales,positions,row_batch,
        first_positions,last_positions,codes,scales,live_rows,sequences,slots,
        circular,code_stride,scale_stride,status,status_row_offset);
  ptx::griddep_launch_dependents();
}
} // namespace chop::latent::window_fused
