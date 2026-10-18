#pragma once
#include "../../../kernels-sm120/cuda/row/silu_mul_quantize.cuh"

namespace chop::row::silu_mul_quantize::mutants {
struct DropNonfinite {
  __device__ static uint32_t update(uint32_t* status, uint64_t row, uint32_t flags) {
    flags &= ~block_quantize::NonfiniteInput;
    return atomicOr(status + row, flags) | flags;
  }
};
struct WrongRow {
  __device__ static uint32_t update(uint32_t* status, uint64_t row, uint32_t flags) {
    return atomicOr(status + (row ^ 1), flags) | flags;
  }
};
} // namespace chop::row::silu_mul_quantize::mutants
