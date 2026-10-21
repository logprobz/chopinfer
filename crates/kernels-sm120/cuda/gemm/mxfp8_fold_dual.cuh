#pragma once
#include "mxfp8_fold_stream.cuh"

namespace chop::gemm::mxfp8_fold_dual {
using mxfp8_fold_stream::Task;
using mxfp8_fold_stream::Shared;

template<int N, int K, int Groups, int RowStride, int GroupStride,
         int TileN, bool F32Output, bool DeferAdmission>
__device__ void run(uint32_t index, Shared& arena,
    const uint8_t* activations, const uint8_t* activation_scales,
    const uint8_t* packed, const uint8_t* weight_scales,
    void* output, uint8_t* workspace, uint32_t* state,
    const uint32_t* live_rows, uint32_t* status, uint64_t status_row_offset,
    uint32_t rows, uint32_t folds_per_cta) {
  constexpr int Folds = (K + 127) / 128, Nx = (N + TileN - 1) / TileN;
  const uint32_t ny = (max(rows, 1u) + 15) / 16;
  const uint32_t splits = (Folds + folds_per_cta - 1) / folds_per_cta;
  const Task task{index % Nx, (index / Nx) % ny, index / (Nx * ny), Nx, ny};
  const uint32_t first = (task.z % splits) * folds_per_cta;
  const uint32_t end = min(first + folds_per_cta, uint32_t(Folds));
  const auto weights = mxfp8_fold_stream::weight_fragments<N,K,Groups,TileN,false>(
      task,first,end,splits,packed,weight_scales,nullptr,nullptr);
  ptx::griddep_wait();
  // Each projection retains its own counters, partials and ordered finisher.
  // Shared row status is safe: both admissions inspect the same A bytes.
  mxfp8_fold_stream::run_task<N,K,Groups,RowStride,GroupStride,TileN,F32Output,false,DeferAdmission>(
      task,arena,weights,activations,activation_scales,packed,weight_scales,
      nullptr,nullptr,output,nullptr,workspace,state,live_rows,status,
      status_row_offset,rows,folds_per_cta);
}

template<int N0, int N1, int K, int Groups, int RowStride, int GroupStride,
         int TileN, bool F32Output, bool DeferAdmission = false>
__device__ void entry(const uint8_t* activations, const uint8_t* activation_scales,
    const uint8_t* packed0, const uint8_t* scales0, void* output0,
    uint8_t* workspace0, uint32_t* state0,
    const uint8_t* packed1, const uint8_t* scales1, void* output1,
    uint8_t* workspace1, uint32_t* state1,
    const uint32_t* live_rows, uint32_t* status, uint64_t status_row_offset,
    uint32_t rows, uint32_t folds0, uint32_t folds1) {
  constexpr int Folds = (K + 127) / 128;
  const uint32_t ny = (max(rows, 1u) + 15) / 16;
  const uint32_t first_ctas = ((N0 + TileN - 1) / TileN) * ny * Groups
      * ((Folds + folds0 - 1) / folds0);
  __shared__ Shared arena;
  ptx::griddep_launch_dependents();
  if (blockIdx.x < first_ctas)
    run<N0,K,Groups,RowStride,GroupStride,TileN,F32Output,DeferAdmission>(
        blockIdx.x,arena,activations,activation_scales,packed0,scales0,output0,
        workspace0,state0,live_rows,status,status_row_offset,rows,folds0);
  else
    run<N1,K,Groups,RowStride,GroupStride,TileN,F32Output,DeferAdmission>(
        blockIdx.x-first_ctas,arena,activations,activation_scales,packed1,scales1,output1,
        workspace1,state1,live_rows,status,status_row_offset,rows,folds1);
}
} // namespace chop::gemm::mxfp8_fold_dual
