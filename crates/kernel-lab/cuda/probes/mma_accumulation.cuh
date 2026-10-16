#pragma once
#include <stdint.h>
#include "../../../kernels-sm120/cuda/common/ptx.cuh"

// Accumulation schemes for BF16 mma.sync.m16n8k16 with F32 accumulators,
// compared against an f64 reference by tests/mma_accumulation.rs.
//
// Weights are row-major [n][k] BF16 and activations row-major [m][k] BF16;
// outputs are F32 [m][n], Y[token][row] = sum_k W[row][k] X[token][k]. As in
// the fragment GEMMs, weights are the MMA A operand (16 rows per strip) and
// activations the B operand (tokens on n8), with the logical PTX fragment
// layout that kernel_core::transform::fragment packs. MMAs issue in K order.

namespace chop::probe {

__device__ __forceinline__ uint32_t load_pair(const uint16_t* address) {
  return *reinterpret_cast<const uint32_t*>(address);
}

__device__ __forceinline__ float bf16_value(uint16_t bits) {
  return __uint_as_float(uint32_t(bits) << 16);
}

// One warp owns a 16-row weight strip by 32 tokens (four n8 tiles). With
// chain >= k every accumulator is one MMA chain over all of K from zero and is
// stored as the fragment GEMMs store theirs. Otherwise each run of chain / 16
// MMAs starts from zero (the last run may be shorter) and is added into an F32
// total in K order, plainly or with Kahan compensation. Requires n % 16 == 0,
// m % 32 == 0, k % 16 == 0 and chain % 16 == 0.
__device__ void mma_chain(const uint16_t* weight, const uint16_t* activation,
                          float* output, uint32_t n, uint32_t m, uint32_t k,
                          uint32_t chain, uint32_t compensated) {
  const uint32_t warp = threadIdx.x >> 5;
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t strip = blockIdx.x * (blockDim.x >> 5) + warp;
  const uint32_t token0 = blockIdx.y * 32;
  if (strip * 16 >= n || token0 >= m) return;
  const uint32_t group = lane >> 2;
  const uint32_t pair = (lane & 3) * 2;
  const uint16_t* row_low = weight + uint64_t(strip * 16 + group) * k + pair;
  const uint16_t* row_high = row_low + uint64_t(8) * k;
  const uint16_t* tokens[4];
#pragma unroll
  for (int j = 0; j < 4; ++j)
    tokens[j] = activation + uint64_t(token0 + j * 8 + group) * k + pair;
  const uint32_t steps = k / 16;
  const uint32_t run_steps = chain >= k ? steps : chain / 16;
  ptx::D4 run[4], total[4], carry[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    run[j] = ptx::D4{};
    total[j] = ptx::D4{};
    carry[j] = ptx::D4{};
  }
  for (uint32_t step = 0; step < steps; ++step) {
    const uint32_t offset = step * 16;
    ptx::A4 a;
    a.v[0] = load_pair(row_low + offset);
    a.v[1] = load_pair(row_high + offset);
    a.v[2] = load_pair(row_low + offset + 8);
    a.v[3] = load_pair(row_high + offset + 8);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      ptx::B2 b;
      b.v[0] = load_pair(tokens[j] + offset);
      b.v[1] = load_pair(tokens[j] + offset + 8);
      ptx::mma_bf16(run[j], a, b);
    }
    if (run_steps == steps || ((step + 1) % run_steps != 0 && step + 1 != steps))
      continue;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        if (compensated) {
          // b12x's compensated addition of each temporary.
          const float addend = __fsub_rn(run[j].v[v], carry[j].v[v]);
          const float sum = __fadd_rn(total[j].v[v], addend);
          carry[j].v[v] = __fsub_rn(__fsub_rn(sum, total[j].v[v]), addend);
          total[j].v[v] = sum;
        } else {
          total[j].v[v] = __fadd_rn(total[j].v[v], run[j].v[v]);
        }
        run[j].v[v] = 0.0f;
      }
  }
  const uint32_t column = strip * 16 + group;
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int v = 0; v < 4; ++v) {
      const uint32_t token = token0 + j * 8 + pair + (v & 1);
      output[uint64_t(token) * n + column + (v >> 1) * 8] =
          run_steps == steps ? run[j].v[v] : total[j].v[v];
    }
}

// Sequential F32 fused multiply-add from zero in K order, one thread per
// output.
__device__ void fma_chain(const uint16_t* weight, const uint16_t* activation,
                          float* output, uint32_t n, uint32_t m, uint32_t k) {
  const uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= uint64_t(n) * m) return;
  const uint16_t* row = weight + (index % n) * k;
  const uint16_t* token = activation + (index / n) * k;
  float total = 0.0f;
  for (uint32_t i = 0; i < k; ++i)
    total = __fmaf_rn(bf16_value(row[i]), bf16_value(token[i]), total);
  output[index] = total;
}

// One m16n8k16 per case (one warp per block): D = A B + C with A row-major
// [16][16], B stored as [8 tokens][16 k] and C, D row-major [16][8].
__device__ void mma_once(const uint16_t* a, const uint16_t* b, const float* c,
                         float* d) {
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t group = lane >> 2;
  const uint32_t pair = (lane & 3) * 2;
  const uint16_t* A = a + uint64_t(blockIdx.x) * 256;
  const uint16_t* B = b + uint64_t(blockIdx.x) * 128;
  const float* C = c + uint64_t(blockIdx.x) * 128;
  float* D = d + uint64_t(blockIdx.x) * 128;
  ptx::A4 fa;
  fa.v[0] = load_pair(A + group * 16 + pair);
  fa.v[1] = load_pair(A + (group + 8) * 16 + pair);
  fa.v[2] = load_pair(A + group * 16 + pair + 8);
  fa.v[3] = load_pair(A + (group + 8) * 16 + pair + 8);
  ptx::B2 fb;
  fb.v[0] = load_pair(B + group * 16 + pair);
  fb.v[1] = load_pair(B + group * 16 + pair + 8);
  ptx::D4 acc;
  acc.v[0] = C[group * 8 + pair];
  acc.v[1] = C[group * 8 + pair + 1];
  acc.v[2] = C[(group + 8) * 8 + pair];
  acc.v[3] = C[(group + 8) * 8 + pair + 1];
  ptx::mma_bf16(acc, fa, fb);
  D[group * 8 + pair] = acc.v[0];
  D[group * 8 + pair + 1] = acc.v[1];
  D[(group + 8) * 8 + pair] = acc.v[2];
  D[(group + 8) * 8 + pair + 1] = acc.v[3];
}

} // namespace chop::probe
