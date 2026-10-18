#pragma once
#include "gemm/exact_f32.cuh"
namespace chop::exact_f32::mutants {
struct DropLo {
  __device__ __forceinline__ static void split(float x, uint16_t (&terms)[3]) {
    Components::split(x,terms);
    terms[2] = 0;
  }
};
struct Tf32 {
  __device__ __forceinline__ static void split(float x, uint16_t (&terms)[3]) {
    uint32_t rounded;
    asm("cvt.rn.tf32.f32 %0,%1;" : "=r"(rounded) : "f"(x));
    Components::split(__uint_as_float(rounded),terms);
  }
};
struct WrongMerge : SeparateChains {
  __device__ __forceinline__ float merged(int v) const {
    return __fadd_rn(partial[0].v[v], __fadd_rn(partial[1].v[v], partial[2].v[v]));
  }
};
struct MergedChains {
  ptx::D4 partial{};
  __device__ __forceinline__ void step(const ptx::A4& a, const ptx::B2 (&b)[3]) {
#pragma unroll
    for (int c = 0; c < 3; ++c) ptx::mma_bf16(partial,a,b[c]);
  }
  __device__ __forceinline__ float merged(int v) const {return partial.v[v];}
};
}
