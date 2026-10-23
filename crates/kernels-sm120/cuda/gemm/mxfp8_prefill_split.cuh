#pragma once
#include "mxfp8_prefill.cuh"

// Keep full TMA tiles and the final fold tile in separate launches. Both
// bounds come from the device live count, so graph replay can shrink it.
// The two launches write disjoint rows and use no shared global workspace.
// PackedWeights reads mxfp8-fold-packed-8x128-v1 weight codes.
namespace chop::gemm::mxfp8_prefill_split {

template<int N, int K, int Groups, int RowStride, int GroupStride,
         bool F32Output, bool SingleChain = false, bool PackedWeights = false>
__device__ void full(const CUtensorMap& activation_map, const CUtensorMap& weight_map,
    const uint8_t* activations, const uint8_t* weights,
    const uint8_t* activation_scales, const uint8_t* weight_scales,
    void* output, const uint32_t* live_rows, const uint32_t* status,
    uint64_t status_offset, uint32_t rows) {
  __shared__ uint32_t full_rows;
  ptx::griddep_wait();
  if (threadIdx.x == 0) full_rows = min(rows, *live_rows) / 128 * 128;
  __syncthreads();
  mxfp8_prefill::run<N,K,Groups,RowStride,GroupStride,F32Output,SingleChain,PackedWeights>(
      activation_map,weight_map,activations,weights,activation_scales,weight_scales,
      output,&full_rows,status,status_offset,rows);
}

template<int N, int K, int Groups, int RowStride, int GroupStride,
         bool F32Output, bool Decode, bool SingleChain, int FoldK,
         bool ModuloScaleRows, uint32_t InvalidFlag, bool PackedWeights = false>
__device__ void tail(const uint8_t* activations, const uint8_t* activation_scales,
    const uint8_t* weights, const uint8_t* weight_scales, void* output,
    const uint32_t* live_rows, uint32_t* status, uint64_t status_offset, uint32_t rows) {
  static_assert(!Decode && !ModuloScaleRows && FoldK == 128);
  __shared__ uint32_t first, count;
  ptx::griddep_wait();
  if (threadIdx.x == 0) {
    const uint32_t live = min(rows, *live_rows);
    first = live / 128 * 128;
    count = live - first;
  }
  __syncthreads();
  const uint64_t input_offset = uint64_t(first) * RowStride;
  const uint64_t output_offset = uint64_t(first) * Groups * N * (F32Output ? 4 : 2);
  mxfp8_fold::run<N,K,Groups,RowStride,GroupStride,F32Output,false,
      SingleChain,128,false,InvalidFlag,PackedWeights>(
      activations + input_offset,activation_scales + input_offset / 32,
      weights,weight_scales,static_cast<uint8_t*>(output) + output_offset,
      &count,status,status_offset + first,count);
}
} // namespace chop::gemm::mxfp8_prefill_split
