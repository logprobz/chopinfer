#pragma once

#include "mxfp8_fold.cuh"

namespace chop::gemm::mxfp8_fold_decode {

// Admission runs once per row, before any split reads the sticky status.
template<int K, int Groups, int RowStride, int GroupStride>
__device__ void admit(const uint8_t* codes, const uint8_t* scales, uint32_t* failed,
                     const uint32_t* live_rows, uint32_t* status,
                     uint64_t status_row_offset, uint32_t rows) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= min(rows, *live_rows)) return;
  bool invalid = false;
  for (int g = 0; g < Groups; ++g) {
    for (int k = threadIdx.x; k < K; k += 128) {
      const uint64_t at = uint64_t(row) * RowStride + uint64_t(g) * GroupStride + k;
      invalid |= !mxfp8_fold::admitted(codes[at], scales[at / 32]);
    }
  }
  const bool bad = __syncthreads_or(invalid);
  if (threadIdx.x == 0) {
    const uint32_t previous = atomicOr(status + status_row_offset + row, bad ? 1u : 0u);
    failed[row] = bad || previous != 0;
  }
}

__device__ __forceinline__ void copy16(uint8_t* dst, const uint8_t* src, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
               :: "r"(uint32_t(__cvta_generic_to_shared(dst))), "l"(src),
                  "r"(valid ? 16 : 0) : "memory");
}

template<int N, int K, int Groups, int RowStride, int GroupStride, bool Paired>
__device__ void partials(const uint8_t* activations, const uint8_t* activation_scales,
                        const uint8_t* weights, const uint8_t* weight_scales,
                        const uint8_t* weights2, const uint8_t* weight_scales2,
                        float* partials, const uint32_t* live_rows,
                        const uint32_t* status, uint64_t status_row_offset, uint32_t rows) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  constexpr int Folds = (K + 127) / 128;
  constexpr int Splits = (Folds + 1) / 2;
  const uint32_t bank = blockIdx.z / (Splits * Groups);
  const uint32_t g = (blockIdx.z / Splits) % Groups;
  const uint32_t first_fold = (blockIdx.z % Splits) * 2;
  const uint32_t token_base = blockIdx.y * 16;
  const uint32_t column_base = blockIdx.x * 16;
  const uint32_t live = min(rows, *live_rows);
  if (token_base >= live) return;
  const uint32_t* failed = reinterpret_cast<const uint32_t*>(partials);
  partials += 64; // The first aligned 256 bytes hold once-per-launch row flags.
  if constexpr (Paired) {
    if (bank) { weights = weights2; weight_scales = weight_scales2; }
  }
  partials += uint64_t(bank) * Folds * rows * Groups * N;
  __shared__ __align__(16) uint8_t codes[2][32 * 128];
  __shared__ uint8_t scales[2][32 * 4];
  // Each group contains one fold's operands. Waiting for the first group
  // leaves the second in flight while the first four MMA steps execute.
#pragma unroll
  for (int stage = 0; stage < 2; ++stage) {
    const uint32_t fold = first_fold + stage;
    for (int chunk = threadIdx.x; chunk < 32 * 8; chunk += 64) {
      const uint32_t r = chunk / 8;
      const uint32_t col = fold * 128 + (chunk % 8) * 16;
      const bool a = r < 16;
      const uint32_t row = a ? token_base + r : column_base + r - 16;
      bool valid = col < K && row < (a ? live : N);
      if (a && valid) valid = failed[row] == 0;
      const uint64_t at = a ? uint64_t(row) * RowStride + uint64_t(g) * GroupStride + col
                            : (uint64_t(g) * N + row) * K + col;
      const uint8_t* src = a ? activations : weights;
      copy16(codes[stage] + chunk * 16, src + (valid ? at : 0), valid);
    }
    for (int i = threadIdx.x; i < 32 * 4; i += 64) {
      const uint32_t r = i / 4;
      const uint32_t col = fold * 128 + (i % 4) * 32;
      const bool a = r < 16;
      const uint32_t row = a ? token_base + r : column_base + r - 16;
      bool valid = col < K && row < (a ? live : N);
      if (a && valid) valid = failed[row] == 0;
      const uint64_t at = a ? uint64_t(row) * RowStride + uint64_t(g) * GroupStride + col
                            : (uint64_t(g) * N + row) * K + col;
      scales[stage][i] = valid ? (a ? activation_scales : weight_scales)[at / 32] : 127;
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
  const uint32_t warp = threadIdx.x / 32;
  const uint32_t quad = (threadIdx.x & 31) / 4;
  const uint32_t lane4 = threadIdx.x & 3;
#pragma unroll
  for (int stage = 0; stage < 2; ++stage) {
    if (stage == 0) asm volatile("cp.async.wait_group 1;" ::: "memory");
    else asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    ptx::D4 partial{};
#pragma unroll
    for (int step = 0; step < 4; ++step) {
      ptx::A4 af{};
      ptx::B2 bf{};
#pragma unroll
      for (int i = 0; i < 4; ++i)
        af.v[i] = *reinterpret_cast<const uint32_t*>(codes[stage] +
            (quad + (i % 2) * 8) * 128 + step * 32 + (i / 2) * 16 + lane4 * 4);
#pragma unroll
      for (int i = 0; i < 2; ++i)
        bf.v[i] = *reinterpret_cast<const uint32_t*>(codes[stage] +
            (16 + warp * 8 + quad) * 128 + step * 32 + i * 16 + lane4 * 4);
      const uint32_t sa = scales[stage][(quad + (lane4 & 1) * 8) * 4 + step];
      const uint32_t sb = scales[stage][(16 + warp * 8 + quad) * 4 + step];
      ptx::mma_mxf8f6f4<ptx::Element::e4m3, ptx::Element::e4m3>(partial, af, bf, sa, sb);
    }
    if (first_fold + stage < Folds) {
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        const uint32_t row = token_base + quad + (v / 2) * 8;
        const uint32_t col = column_base + warp * 8 + lane4 * 2 + v % 2;
        if (row < live && col < N)
          partials[((uint64_t(first_fold + stage) * rows + row) * Groups + g) * N + col]
              = partial.v[v];
      }
    }
  }
}

template<int N, int K, int Groups, bool F32Output, bool Paired>
__device__ void merge(const float* partials, void* output, void* output2, const uint32_t* live_rows,
                     const uint32_t* status, uint64_t status_row_offset, uint32_t rows) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  if (!min(rows, *live_rows)) return;
  const uint32_t* failed = reinterpret_cast<const uint32_t*>(partials);
  partials += 64;
  const uint64_t flat = uint64_t(blockIdx.x) * 128 + threadIdx.x;
  const uint64_t bank_elements = uint64_t(rows) * Groups * N;
  const uint32_t bank = flat / bank_elements;
  if (bank >= (Paired ? 2 : 1)) return;
  const uint64_t at = flat % bank_elements;
  if constexpr (Paired) { if (bank) output = output2; }
  partials += uint64_t(bank) * ((K + 127) / 128) * bank_elements;
  const uint32_t row = at / (uint64_t(Groups) * N);
  if (row >= min(rows, *live_rows)) return;
  float total = 0.0f;
  if (!failed[row]) {
    for (int fold = 0; fold < (K + 127) / 128; ++fold)
      total = __fadd_rn(total, partials[uint64_t(fold) * rows * Groups * N + at]);
  }
  if constexpr (F32Output) reinterpret_cast<float*>(output)[at] = total;
  else reinterpret_cast<uint16_t*>(output)[at] = convert::bf16_from_f32(total);
}

} // namespace chop::gemm::mxfp8_fold_decode
