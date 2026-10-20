#pragma once
#include "mxfp8_fold_stream.cuh"

namespace chop::gemm::mxfp8_fold_resident {

using mxfp8_fold_stream::Task;
using mxfp8_fold_stream::Shared;
using mxfp8_fold_stream::Fragments;

// Admission remains a row-wide operation before any output task. It also
// preserves sticky errors when a later replay presents valid operands.
template<int K, int Groups, int RowStride, int GroupStride>
__device__ void admit_task(uint32_t row, const uint8_t* codes,
    const uint8_t* scales, uint32_t* failed, const uint32_t* live_rows,
    uint32_t* status, uint64_t status_row_offset, uint32_t rows) {
  if (row >= min(rows, *live_rows)) return;
  bool invalid = false;
  for (int g = 0; g < Groups; ++g) {
    for (int k = threadIdx.x * 16; k < K; k += blockDim.x * 16) {
      const uint64_t at = uint64_t(row) * RowStride + uint64_t(g) * GroupStride + k;
      const uint4 packed = *reinterpret_cast<const uint4*>(codes + at);
      const uint8_t scale = scales[at / 32];
      invalid |= !(mxfp8_fold_ticket::admitted4(packed.x, scale) &&
                   mxfp8_fold_ticket::admitted4(packed.y, scale) &&
                   mxfp8_fold_ticket::admitted4(packed.z, scale) &&
                   mxfp8_fold_ticket::admitted4(packed.w, scale));
    }
  }
  const bool bad = __syncthreads_or(invalid);
  if (threadIdx.x == 0) {
    const uint32_t previous = atomicOr(status + status_row_offset + row, bad ? 1u : 0u);
    failed[row] = bad || previous != 0;
  }
}

// A column task retains all fold sums in registers. Each 128-K fold still
// starts at zero and uses the original four MMA steps. Only storage and
// scheduling change: the final additions remain fold 0, 1, ... from zero.
template<int N, int K, int Groups, int RowStride, int GroupStride,
         int TileN, bool Reverse = false>
__device__ void run_task(Task task, Shared& shared, Fragments current,
    const uint8_t* activations, const uint8_t* activation_scales,
    const uint8_t* packed, const uint8_t* weight_scales, float* output,
    const uint32_t* failed, const uint32_t* live_rows, uint32_t rows) {
  static_assert(TileN == 8 || TileN == 16 || TileN == 32);
  constexpr int Folds = (K + 127) / 128;
  constexpr int Threads = TileN * 4;
  const uint32_t live = min(rows, *live_rows);
  const uint32_t token_base = task.y * 16;
  if (token_base >= live) return;
  const uint32_t g = task.z;
  const uint32_t warp = threadIdx.x / 32;
  const uint32_t quad = (threadIdx.x & 31) / 4;
  const uint32_t lane4 = threadIdx.x & 3;
  ptx::D4 total{};
  for (uint32_t base = 0; base < Folds; base += 2) {
    Fragments next = mxfp8_fold_stream::weight_fragments<N, K, Groups, TileN, false>(
        task, base + 2, Folds, 1, packed, weight_scales, nullptr, nullptr);
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      for (uint32_t chunk = threadIdx.x; chunk < 16 * 8; chunk += Threads) {
        const uint32_t row = token_base + chunk / 8;
        const uint32_t k = (base + p) * 128 + (chunk % 8) * 16;
        const bool valid = base + p < Folds && row < live && k < K && !failed[row];
        const uint64_t at = uint64_t(row) * RowStride + uint64_t(g) * GroupStride + k;
        mxfp8_fold_ticket::copy16(shared.codes[p] + chunk * 16,
            activations + (valid ? at : 0), valid);
      }
      for (uint32_t i = threadIdx.x; i < 64; i += Threads) {
        const uint32_t row = token_base + i / 4;
        const uint32_t k = (base + p) * 128 + (i % 4) * 32;
        const bool valid = base + p < Folds && row < live && k < K && !failed[row];
        const uint64_t at = uint64_t(row) * RowStride + uint64_t(g) * GroupStride + k;
        shared.scales[p][i] = valid ? activation_scales[at / 32] : 127;
      }
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    ptx::D4 raw[2]{};
#pragma unroll
    for (int step = 0; step < 4; ++step) {
#pragma unroll
      for (int p = 0; p < 2; ++p) {
        ptx::A4 af{};
        ptx::B2 bf{};
#pragma unroll
        for (int i = 0; i < 4; ++i)
          af.v[i] = *reinterpret_cast<const uint32_t*>(shared.codes[p] +
              (quad + (i % 2) * 8) * 128 + step * 32 + (i / 2) * 16 + lane4 * 4);
        const uint4 w = step < 2 ? current.lo[p] : current.hi[p];
        bf.v[0] = (step & 1) ? w.z : w.x;
        bf.v[1] = (step & 1) ? w.w : w.y;
        const uint32_t sa = shared.scales[p][(quad + (lane4 & 1) * 8) * 4 + step];
        const uint32_t sb = (current.scales[p] >> (8 * step)) & 255;
        ptx::mma_mxf8f6f4<ptx::Element::e4m3, ptx::Element::e4m3>(raw[p], af, bf, sa, sb);
      }
    }
#pragma unroll
    for (int index = 0; index < 2; ++index) {
      const int p = Reverse ? 1 - index : index;
      if (base + p < Folds) {
#pragma unroll
        for (int v = 0; v < 4; ++v) total.v[v] = __fadd_rn(total.v[v], raw[p].v[v]);
      }
    }
    current = next;
    __syncthreads();
  }
#pragma unroll
  for (int v = 0; v < 4; ++v) {
    const uint32_t row = token_base + quad + (v / 2) * 8;
    const uint32_t col = task.x * TileN + warp * 8 + lane4 * 2 + v % 2;
    if (row < live && col < N)
      output[(uint64_t(row) * Groups + g) * N + col] = failed[row] ? 0.0f : total.v[v];
  }
}

} // namespace chop::gemm::mxfp8_fold_resident
