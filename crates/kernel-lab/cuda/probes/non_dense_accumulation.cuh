#pragma once
#include <stdint.h>
#include "../../../kernels-sm120/cuda/common/ptx.cuh"
#include "../../../kernels-sm120/cuda/common/bf16_pairs.cuh"
#include "../../../kernels-sm120/cuda/encoder/linear_pairs.cuh"

// Lab-only probes. The A operand owns sixteen output columns and the B
// operand owns eight query/token rows, matching encoder pairs. PV swaps the operands to use sixteen
// queries and eight features, as served prefill does. No served template includes this header.
namespace chop::probe::non_dense {

__device__ __forceinline__ uint32_t pair(const uint16_t* x) {
  return uint32_t(x[0]) | uint32_t(x[1]) << 16;
}
__device__ __forceinline__ ptx::A4 weight_fragment(
    const uint16_t* w, uint32_t k, uint32_t column, uint32_t offset) {
  const uint32_t lane = threadIdx.x & 31;
  const uint16_t* low = w + uint64_t(column + lane / 4) * k + offset + 2 * (lane % 4);
  const uint16_t* high = low + uint64_t(8) * k;
  return ptx::A4{{pair(low), pair(high), pair(low + 8), pair(high + 8)}};
}
__device__ __forceinline__ void promote(ptx::D4& total, ptx::D4& run,
                                       ptx::D4& carry, bool compensated) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    if (compensated) {
      const float addend = __fsub_rn(run.v[i], carry.v[i]);
      const float sum = __fadd_rn(total.v[i], addend);
      carry.v[i] = __fsub_rn(__fsub_rn(sum, total.v[i]), addend);
      total.v[i] = sum;
    } else {
      total.v[i] = __fadd_rn(total.v[i], run.v[i]);
    }
    run.v[i] = 0.f;
  }
}
__device__ __forceinline__ void store(float* out, ptx::D4 d,
                                    uint32_t n, uint32_t column, uint32_t row) {
  const uint32_t lane = threadIdx.x & 31;
#pragma unroll
  for (int i = 0; i < 4; ++i)
    out[uint64_t(row + 2 * (lane % 4) + (i & 1)) * n + column + lane / 4 + (i / 2) * 8] = d.v[i];
}

// Modes: 0 full hi/lo chain; 1 combined K64; 2 combined K128;
// 3 separate hi/lo K128; 4 Kahan combined K64. FMA is a separate entry.
__device__ void encoder(const uint16_t* weight, const float* activation,
                        float* output, uint32_t n, uint32_t m, uint32_t k,
                        uint32_t mode) {
  const uint32_t column = blockIdx.x * 16;
  const uint32_t row = blockIdx.y * 8;
  if (column >= n || row >= m) return;
  const uint32_t lane = threadIdx.x & 31;
  const float* x = activation + uint64_t(row + lane / 4) * k + 2 * (lane % 4);
  ptx::D4 total{}, run{}, low{}, carry{};
  for (uint32_t offset = 0; offset < k; offset += 16) {
    const ptx::A4 a = weight_fragment(weight, k, column, offset);
    ptx::B2 hi, lo;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const auto split = pairs::split(x[offset + h * 8], x[offset + h * 8 + 1]);
      hi.v[h] = split.hi;
      lo.v[h] = split.lo;
    }
    ptx::mma_bf16(run, a, hi);
    ptx::mma_bf16(mode == 3 ? low : run, a, lo);
    const uint32_t flush = (mode == 1 || mode == 4) ? 64 : 128;
    if (mode != 0 && ((offset + 16) % flush == 0 || offset + 16 == k)) {
      promote(total, run, carry, mode == 4);
      if (mode == 3) promote(total, low, carry, false);
    }
  }
  store(output, mode == 0 ? run : total, n, column, row);
}

__device__ void encoder_fma(const uint16_t* weight, const float* activation,
                            float* output, uint32_t n, uint32_t m, uint32_t k) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n * m) return;
  float total = 0.f;
  for (uint32_t offset = 0; offset < k; ++offset) {
    uint16_t hi, lo;
    pairs::split(activation[uint64_t(index / n) * k + offset], hi, lo);
    const float w = __uint_as_float(uint32_t(weight[uint64_t(index % n) * k + offset]) << 16);
    total = __fmaf_rn(w, __uint_as_float(uint32_t(hi) << 16), total);
    total = __fmaf_rn(w, __uint_as_float(uint32_t(lo) << 16), total);
  }
  output[index] = total;
}

// Unmodified served template, F32 store before bias/activation/residual.
// One 128-column strip allows exact long-chain parity on bounded cases.
__device__ void encoder_served(const uint16_t* weight, const float* activation,
    float* output, const uint32_t* live, uint32_t k) {
  using E = chop::encoder::LinearPairs<false, 0, false, float>;
#define CHOP_CASE(K) case K: chop::gemm::encoder_gemm_pairs<128, K, E>(activation, weight, nullptr, output, live); break
  switch (k) {
    CHOP_CASE(384); CHOP_CASE(1152); CHOP_CASE(1280); CHOP_CASE(1536);
    CHOP_CASE(2048); CHOP_CASE(3072); CHOP_CASE(4096); CHOP_CASE(4352);
    CHOP_CASE(4608); CHOP_CASE(5120); CHOP_CASE(16384);
  }
#undef CHOP_CASE
}

// Frozen online-softmax inputs: probability[row][key] is unnormalized P at
// its own 32-key tile's maximum; correction[row][tile] carries exp(m_old-m_new).
// Values are BF16 [column][key]. Each residual term uses the SAME chain.
// Modes: 0 persistent MMA; 1 K32 tile temporary; 2 at most eight MMA calls;
// 3 K128 temporary (flush before a nonidentity rescale); 4 sequential FMA.
__device__ void pv(const uint16_t* value, const float* probability,
                  const float* correction, float* output, uint32_t n,
                  uint32_t m, uint32_t k, uint32_t terms, uint32_t mode) {
  const uint32_t column = blockIdx.x * 8;
  const uint32_t row = blockIdx.y * 16;
  if (column >= n || row >= m) return;
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t q = 2 * (lane % 4);
  ptx::D4 total{}, run{}, carry{};
  uint32_t calls = 0;
  for (uint32_t offset = 0; offset < k; offset += 16) {
    if (offset % 32 == 0) {
      bool rescale = false;
#pragma unroll
      for (int half = 0; half < 2; ++half)
        rescale |= correction[uint64_t(row + lane / 4 + half * 8) * (k / 32) + offset / 32] != 1.f;
      if (mode != 0 && __any_sync(0xffffffff, rescale)) {
        promote(total, run, carry, false);
        calls = 0;
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint32_t token = row + lane / 4 + (i / 2) * 8;
        const float c = correction[uint64_t(token) * (k / 32) + offset / 32];
        if (mode == 0) run.v[i] = __fmul_rn(run.v[i], c);
        else total.v[i] = __fmul_rn(total.v[i], c);
      }
    }
    // P is A (sixteen queries), V is B (eight features), as prefill PV.
    const uint16_t* v = value + uint64_t(column + lane / 4) * k + offset + q;
    const ptx::B2 b{{pair(v), pair(v + 8)}};
    float residual[8];
#pragma unroll
    for (int half = 0; half < 2; ++half)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const float* p = probability + uint64_t(row + lane / 4 + half * 8) * k + offset + q + h * 8;
        residual[half * 2 + h * 4] = p[0];
        residual[half * 2 + h * 4 + 1] = p[1];
      }
    for (uint32_t term = 0; term < terms; ++term) {
      ptx::A4 a;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        a.v[i] = pairs::pack(residual[2 * i], residual[2 * i + 1]);
        residual[2 * i] = __fsub_rn(residual[2 * i], pairs::low(a.v[i]));
        residual[2 * i + 1] = __fsub_rn(residual[2 * i + 1], pairs::high(a.v[i]));
      }
      ptx::mma_bf16(run, a, b);
      ++calls;
      if (mode == 2 && calls == 8) {
        promote(total, run, carry, false);
        calls = 0;
      }
    }
    if ((mode == 1 && (offset + 16) % 32 == 0) ||
        (mode == 3 && (offset + 16) % 128 == 0)) {
      promote(total, run, carry, false);
      calls = 0;
    }
  }
  if (mode != 0) promote(total, run, carry, false);
  const ptx::D4 d = mode == 0 ? run : total;
#pragma unroll
  for (int i = 0; i < 4; ++i)
    output[uint64_t(row + lane / 4 + (i / 2) * 8) * n + column + q + (i & 1)] = d.v[i];
}

__device__ void pv_fma(const uint16_t* value, const float* probability,
                      const float* correction, float* output, uint32_t n,
                      uint32_t m, uint32_t k, uint32_t terms) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n * m) return;
  const uint32_t row = index / n, column = index % n;
  float total = 0.f;
  for (uint32_t offset = 0; offset < k; ++offset) {
    if (offset % 32 == 0)
      total = __fmul_rn(total, correction[uint64_t(row) * (k / 32) + offset / 32]);
    float residual = probability[uint64_t(row) * k + offset];
    const float v = __uint_as_float(uint32_t(value[uint64_t(column) * k + offset]) << 16);
    for (uint32_t term = 0; term < terms; ++term) {
      uint16_t hi, lo;
      pairs::split(residual, hi, lo);
      const float p = __uint_as_float(uint32_t(hi) << 16);
      residual = __fsub_rn(residual, p);
      total = __fmaf_rn(v, p, total);
    }
  }
  output[index] = total;
}
} // namespace chop::probe::non_dense
