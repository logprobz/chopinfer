#pragma once
#include <stdint.h>
#include "../common/convert.cuh"
#include "../common/ptx.cuh"

namespace chop::latent::glue {

template<bool NarrowInput = false>
__device__ void norm_rotate(const float* input, const uint16_t* weight,
                            float* output, const float* cos_sin,
                            const uint32_t* live_rows, float eps) {
  ptx::griddep_launch_dependents();
  __shared__ float sums[8];
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  const uint32_t col = 2 * threadIdx.x;
  const uint64_t base = uint64_t(row) * 512 + col;
  float a = input[base], b = input[base + 1];
  if constexpr (NarrowInput) {
    a = convert::bf16_to_f32(convert::bf16_from_f32(a));
    b = convert::bf16_to_f32(convert::bf16_from_f32(b));
  }
  float total = __fmaf_rn(b, b, __fmul_rn(a, a));
  for (int d = 16; d; d >>= 1)
    total = __fadd_rn(total, __shfl_down_sync(0xffffffff, total, d));
  if ((threadIdx.x & 31) == 0) sums[threadIdx.x >> 5] = total;
  __syncthreads();
  total = threadIdx.x < 8 ? sums[threadIdx.x] : 0.f;
  for (int d = 16; d; d >>= 1)
    total = __fadd_rn(total, __shfl_down_sync(0xffffffff, total, d));
  if (threadIdx.x == 0) sums[0] = total;
  __syncthreads();
  const float mean = __fmul_rn(sums[0], 1.f / 512.f);
  const float inv = __fdiv_rn(1.f, __fsqrt_rn(__fadd_rn(mean, eps)));
  float x = __fmul_rn(__fmul_rn(a, inv), convert::bf16_to_f32(weight[col]));
  float y = __fmul_rn(__fmul_rn(b, inv), convert::bf16_to_f32(weight[col + 1]));
  if (col >= 448) {
    const uint64_t at = uint64_t(row) * 64 + col - 448;
    const float c = cos_sin[at], s = cos_sin[at + 1];
    const float next_x = __fsub_rn(__fmul_rn(x, c), __fmul_rn(y, s));
    y = __fadd_rn(__fmul_rn(x, s), __fmul_rn(y, c));
    x = next_x;
  }
  output[base] = x;
  output[base + 1] = y;
}

// F32 producers remain F32 through rotation. Only the query variant emits
// a BF16 operand; the KV path passes F32 to the shared trained quantizer.
template<bool Query, int Mutation = 0>
__device__ void rotate(float* values, uint16_t* native_q, const float* cos_sin,
                       const uint32_t* live_rows, uint32_t heads) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x, head = blockIdx.y;
  if (row >= *live_rows) return;
  const uint32_t column = 2 * threadIdx.x;
  const uint64_t at = (uint64_t(row) * heads + head) * 512 + column;
  float a = values[at], b = values[at + 1];
  constexpr uint32_t tail = Mutation == 1 ? 416 : 448;
  if (column >= tail && column < tail + 64) {
    if constexpr (Mutation == 2) { const float saved = a; a = b; b = saved; }
    if constexpr (Mutation == 3) {
      a = convert::bf16_to_f32(convert::bf16_from_f32(a));
      b = convert::bf16_to_f32(convert::bf16_from_f32(b));
    }
    const uint64_t angle = uint64_t(row) * 64 + column - tail;
    const float c = cos_sin[angle], s = cos_sin[angle + 1];
    const float x = __fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s));
    const float y = __fadd_rn(__fmul_rn(a, s), __fmul_rn(b, c));
    values[at] = a = x;
    values[at + 1] = b = y;
  }
  if constexpr (Query) {
    native_q[at] = convert::bf16_from_f32(a);
    native_q[at + 1] = convert::bf16_from_f32(b);
  }
}

// Quantization is already complete. Keep each plane's own sequence stride
// and never derive the scale pointer from the code allocation.
__device__ void store(const uint8_t* input_codes, const uint8_t* input_scales,
                      const int32_t* positions, const int32_t* row_batch,
                      const int32_t* first_positions, const int32_t* last_positions,
                      uint8_t* codes, uint8_t* scales, const uint32_t* live_rows,
                      uint32_t sequences, uint32_t slots, uint32_t circular,
                      uint64_t code_stride, uint64_t scale_stride) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  const int32_t p = positions[row], sequence = row_batch[row];
  if (p < 0 || sequence < 0 || uint32_t(sequence) >= sequences) return;
  const int32_t last = last_positions[sequence];
  if (p > last) return;
  if (circular && int64_t(p) + slots <= last) return;
  if (!circular && first_positions[sequence] < 0) return;
  const int64_t slot = circular ? uint32_t(p) % slots : int64_t(p) - first_positions[sequence];
  if (slot < 0 || slot >= slots) return;
  const uint64_t destination = uint64_t(sequence) * code_stride + uint64_t(slot) * 512;
  const uint32_t column = 2 * threadIdx.x;
  codes[destination + column] = input_codes[uint64_t(row) * 512 + column];
  codes[destination + column + 1] = input_codes[uint64_t(row) * 512 + column + 1];
  if (threadIdx.x < 16)
    scales[uint64_t(sequence) * scale_stride + uint64_t(slot) * 16 + threadIdx.x] =
        input_scales[uint64_t(row) * 16 + threadIdx.x];
}

template<bool ExcludeCurrent = false>
__device__ void keys(const int32_t* positions, const int32_t* row_batch,
                     const int32_t* first_positions, int32_t* output,
                     const uint32_t* live_rows, uint32_t sequences,
                     uint32_t window, uint32_t slots, uint32_t circular) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  const int32_t p = positions[row];
  int32_t first = 0;
  if (!circular) {
    const int32_t sequence = row_batch[row];
    if (sequence < 0 || uint32_t(sequence) >= sequences) return;
    first = first_positions[sequence];
  }
  const int64_t start = int64_t(p) + 1 - int64_t(window);
  const int64_t begin = p < 0 || start < 0 ? 0 : start;
  for (uint32_t i = threadIdx.x; i < window; i += blockDim.x) {
    const int64_t position = begin + i;
    int32_t slot = -1;
    // Only absolute positions inside the visible interval reach modulo.
    // Never wrap a stale or out-of-range physical candidate into a live cell.
    if (p >= 0 && first >= 0 && position <= int64_t(p) - (ExcludeCurrent ? 1 : 0)) {
      const int64_t offset = circular ? position % slots : position - first;
      if (offset >= 0 && offset < slots) slot = int32_t(offset);
    }
    output[uint64_t(row) * window + i] = slot;
  }
}

} // namespace chop::latent::glue
