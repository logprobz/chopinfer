#pragma once

#include <stdint.h>

namespace chop::moe {

// One 16-byte load supplies the four A fragments for four K16 MMA steps.
__device__ __forceinline__ uint4 load_fragment(const uint8_t* source) {
  uint4 result;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 "
               "{%0,%1,%2,%3}, [%4];"
               : "=r"(result.x), "=r"(result.y), "=r"(result.z),
                 "=r"(result.w) : "l"(source) : "memory");
  return result;
}

__device__ __forceinline__ uint32_t fragment_word(uint4 value, int step) {
  return step == 0 ? value.x : step == 1 ? value.y
       : step == 2 ? value.z : value.w;
}

// Two E2M1 codes to BF16x2, scaled by one UE8M0 code replicated in both bytes.
__device__ __forceinline__ uint32_t dequant(uint32_t word, int byte,
                                            uint32_t scale_pair) {
  uint32_t result;
  const uint16_t codes = uint16_t((word >> (8 * byte)) & 0xff);
  asm("{ .reg .b8 c, unused; mov.b16 {c, unused}, %1; "
      "cvt.rn.scaled::n2::ue8m0.bf16x2.e2m1x2 %0, c, %2; }"
      : "=r"(result) : "h"(codes), "h"(uint16_t(scale_pair)));
  return result;
}

// Activation rows in a ring slot are padded by 16 bytes, so the eight rows
// read by one B fragment load fall in different banks.
constexpr int kFragmentRowBytes = 128 + 16;

__device__ __forceinline__ void cp_async_16(void* shared, const void* global) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
               :: "r"(ptx::shared_address(shared)), "l"(global) : "memory");
}

__device__ __forceinline__ void cp_async_4(void* shared, const void* global) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"
               :: "r"(ptx::shared_address(shared)), "l"(global) : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template<int Pending>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" :: "n"(Pending) : "memory");
}

} // namespace chop::moe
