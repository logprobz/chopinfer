#pragma once
#include "row/block_quantize.cuh"
namespace chop::row::block_quantize::mutants {
template<uint32_t Mask>
struct Drop {
  __device__ static uint32_t update(uint32_t* status,uint64_t row,uint32_t flags) {
    return StatusOr::update(status,row,flags & ~Mask);
  }
};
struct DropPriorScale {
  __device__ static uint32_t update(uint32_t* status,uint64_t row,uint32_t flags) {
    atomicAnd(status+row,~uint32_t(2));
    return StatusOr::update(status,row,flags);
  }
};
struct WrongRow {
  __device__ static uint32_t update(uint32_t* status,uint64_t row,uint32_t flags) {
    const uint64_t target = (row & 1) ? row-1 : (row+1 < gridDim.x ? row+1 : row);
    return StatusOr::update(status,target,flags);
  }
};
struct MissingOr {
  __device__ static uint32_t update(uint32_t* status,uint64_t row,uint32_t flags) {
    atomicExch(status+row,flags);
    return flags;
  }
};
}
