#pragma once
#include <stdint.h>
#include "non_dense_accumulation.cuh"

namespace chop::probe::grouped {

// One independent output strip, preserving explicit K ownership intervals.
// Decoded BF16 weights can chain over an entire interval. Raw FP8-converted
// BF16 weights always promote at a scale-block boundary; shorter runs use
// the SAME F32 post scale, never a scaled-and-rounded BF16 operand.
__device__ void chain(const uint16_t* weight, const uint16_t* activation,
    const float* scales, const uint32_t* offsets, float* output,
    uint32_t n, uint32_t m, uint32_t k, uint32_t parts,
    uint32_t post_scale, uint32_t run_k, float global) {
  const uint32_t column = blockIdx.x * 16;
  const uint32_t row = blockIdx.y * 8;
  if (column >= n || row >= m) return;
  const uint32_t lane = threadIdx.x & 31;
  const uint16_t* x = activation + uint64_t(row + lane / 4) * k + 2 * (lane % 4);
  ptx::D4 result{}, carry{};
  for (uint32_t part = 0; part < parts; ++part) {
    ptx::D4 total{}, run{};
    uint32_t count = 0;
    for (uint32_t offset = offsets[part]; offset < offsets[part + 1]; offset += 16) {
      const ptx::A4 a = non_dense::weight_fragment(weight, k, column, offset);
      const ptx::B2 b{{non_dense::pair(x + offset), non_dense::pair(x + offset + 8)}};
      ptx::mma_bf16(run, a, b);
      count += 16;
      const bool last = offset + 16 == offsets[part + 1];
      const bool scale_end = post_scale && (offset + 16) % 128 == 0;
      if (last || scale_end || (run_k != 0 && count == run_k)) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const uint32_t feature = column + lane / 4 + (i / 2) * 8;
          if (post_scale) {
            const float scale = scales[uint64_t(feature) * (k / 128) + offset / 128];
            total.v[i] = __fmaf_rn(run.v[i], scale, total.v[i]);
          } else total.v[i] = __fadd_rn(total.v[i], run.v[i]);
          run.v[i] = 0.f;
        }
        count = 0;
      }
    }
    // Ring and stream K ownership partials are merged in their K order.
    non_dense::promote(result, total, carry, false);
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) result.v[i] = __fmul_rn(result.v[i], global);
  non_dense::store(output, result, n, column, row);
}

// Sequential FMA within the same promotion and ownership boundaries. This
// controls mechanism/emulation; it is not a performance candidate.
__device__ void fma(const uint16_t* weight, const uint16_t* activation,
    const float* scales, const uint32_t* offsets, float* output,
    uint32_t n, uint32_t m, uint32_t k, uint32_t parts,
    uint32_t post_scale, uint32_t run_k, float global) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n * m) return;
  const uint32_t column = index % n;
  const uint32_t row = index / n;
  float result = 0.f;
  for (uint32_t part = 0; part < parts; ++part) {
    float total = 0.f, run = 0.f;
    uint32_t count = 0;
    for (uint32_t offset = offsets[part]; offset < offsets[part + 1]; ++offset) {
      const float w = __uint_as_float(uint32_t(weight[uint64_t(column) * k + offset]) << 16);
      const float x = __uint_as_float(uint32_t(activation[uint64_t(row) * k + offset]) << 16);
      run = __fmaf_rn(w, x, run);
      ++count;
      if (offset + 1 == offsets[part + 1] ||
          (post_scale && (offset + 1) % 128 == 0) ||
          (run_k != 0 && count == run_k)) {
        total = post_scale ? __fmaf_rn(run, scales[uint64_t(column) * (k / 128) + offset / 128], total)
                           : __fadd_rn(total, run);
        run = 0.f;
        count = 0;
      }
    }
    result = __fadd_rn(result, total);
  }
  output[index] = __fmul_rn(result, global);
}

// Freeze the baseline expert's intermediate once, before comparing down
// reductions. This is the grouped template's unclamped F32 SwiGLU and BF16
// storage rounding. Route weighting and final combine are separate studies.
__device__ void swiglu(const float* gate, const float* up,
                       uint16_t* output, uint32_t count) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const float g = gate[index];
  const float value = (g / (1.f + expf(-g))) * up[index];
  uint16_t hi, lo;
  pairs::split(value, hi, lo);
  output[index] = hi;
}
// Exercise the unchanged TMA consumer helper on its exact swizzled layout.
// This isolates arithmetic; it does not benchmark TMA loading/scheduling.
__device__ void tma_chunk(const uint16_t* weight, const uint16_t* activation,
                          float* output, uint32_t k) {
  __shared__ __align__(128) uint16_t a[16 * 128];
  __shared__ __align__(128) uint16_t b[8 * 128];
  ptx::D4 total{};
  for (uint32_t base = 0; base < k; base += 128) {
    for (uint32_t i = threadIdx.x; i < 16 * 128; i += blockDim.x)
      a[smem::swizzle_128(a, i * 2, 256) / 2] = weight[uint64_t(i / 128) * k + base + i % 128];
    for (uint32_t i = threadIdx.x; i < 8 * 128; i += blockDim.x)
      b[smem::swizzle_128(b, i * 2, 256) / 2] = activation[uint64_t(i / 128) * k + base + i % 128];
    __syncthreads();
    gemm::mma_chunk<gemm::WeightKind::bf16, true, 128>(total,
        reinterpret_cast<const uint8_t*>(a), reinterpret_cast<const uint8_t*>(b),
        0, 0, 0, 0, base / 128, 16, k, nullptr);
    __syncthreads();
  }
  non_dense::store(output, total, 16, 0, 0);
}
} // namespace chop::probe::grouped
