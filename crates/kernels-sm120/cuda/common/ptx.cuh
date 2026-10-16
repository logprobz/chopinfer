#pragma once

// SM12x warp MMA and data movement. Inputs are per-lane register fragments.
// All shared addresses passed to PTX are converted from generic CUDA pointers.
// Cluster, multicast, gather4 and DSMEM entry points require hardware probes.

#include <stdint.h>

namespace chop::ptx {

struct A4 { uint32_t v[4]; };
struct B2 { uint32_t v[2]; };
struct D4 { float v[4]; };
struct A2 { uint32_t v[2]; };

enum class Element { e4m3, e5m2, e3m2, e2m3, e2m1 };

__device__ __forceinline__ uint32_t shared_address(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mma_bf16(D4& d, const A4& a, const B2& b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d.v[0]), "+f"(d.v[1]), "+f"(d.v[2]), "+f"(d.v[3])
               : "r"(a.v[0]), "r"(a.v[1]), "r"(a.v[2]), "r"(a.v[3]),
                 "r"(b.v[0]), "r"(b.v[1]));
}

__device__ __forceinline__ void mma_sparse_bf16(
    D4& d, const A2& a, const B2& b, uint32_t metadata) {
  asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%0,%1,%2,%3}, %8, 0;"
               : "+f"(d.v[0]), "+f"(d.v[1]), "+f"(d.v[2]), "+f"(d.v[3])
               : "r"(a.v[0]), "r"(a.v[1]), "r"(b.v[0]), "r"(b.v[1]),
                 "r"(metadata));
}

// FP4 and FP6 in this family use one byte container per value, shifted to
// the high bits. Native packed FP4 belongs to mma_mxf4nvf4 below.
#define CHOP_MMA_CASE(A, B, KIND, SCALE)                                        \
  if constexpr (EA == Element::A && EB == Element::B) {                         \
    asm volatile("mma.sync.aligned" KIND "m16n8k32.row.col.f32." #A "." #B    \
                 ".f32" SCALE " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "         \
                 "{%0,%1,%2,%3};"                                               \
                 : "+f"(d.v[0]), "+f"(d.v[1]), "+f"(d.v[2]), "+f"(d.v[3])       \
                 : "r"(a.v[0]), "r"(a.v[1]), "r"(a.v[2]), "r"(a.v[3]),            \
                   "r"(b.v[0]), "r"(b.v[1]));                                    \
  } else

template<Element EA, Element EB>
__device__ __forceinline__ void mma_f8f6f4(D4& d, const A4& a, const B2& b) {
  CHOP_MMA_CASE(e4m3, e4m3, ".kind::f8f6f4.", "")
  CHOP_MMA_CASE(e4m3, e5m2, ".kind::f8f6f4.", "")
  CHOP_MMA_CASE(e4m3, e3m2, ".kind::f8f6f4.", "")
  CHOP_MMA_CASE(e4m3, e2m3, ".kind::f8f6f4.", "")
  CHOP_MMA_CASE(e4m3, e2m1, ".kind::f8f6f4.", "")
  CHOP_MMA_CASE(e5m2, e4m3, ".kind::f8f6f4.", "")
  CHOP_MMA_CASE(e5m2, e5m2, ".kind::f8f6f4.", "")
  CHOP_MMA_CASE(e5m2, e3m2, ".kind::f8f6f4.", "")
  CHOP_MMA_CASE(e5m2, e2m3, ".kind::f8f6f4.", "")
  CHOP_MMA_CASE(e5m2, e2m1, ".kind::f8f6f4.", "")
  CHOP_MMA_CASE(e3m2, e4m3, ".kind::f8f6f4.", "")
  CHOP_MMA_CASE(e3m2, e5m2, ".kind::f8f6f4.", "")
  CHOP_MMA_CASE(e3m2, e3m2, ".kind::f8f6f4.", "")
  CHOP_MMA_CASE(e3m2, e2m3, ".kind::f8f6f4.", "")
  CHOP_MMA_CASE(e3m2, e2m1, ".kind::f8f6f4.", "")
  CHOP_MMA_CASE(e2m3, e4m3, ".kind::f8f6f4.", "")
  CHOP_MMA_CASE(e2m3, e5m2, ".kind::f8f6f4.", "")
  CHOP_MMA_CASE(e2m3, e3m2, ".kind::f8f6f4.", "")
  CHOP_MMA_CASE(e2m3, e2m3, ".kind::f8f6f4.", "")
  CHOP_MMA_CASE(e2m3, e2m1, ".kind::f8f6f4.", "")
  CHOP_MMA_CASE(e2m1, e4m3, ".kind::f8f6f4.", "")
  CHOP_MMA_CASE(e2m1, e5m2, ".kind::f8f6f4.", "")
  CHOP_MMA_CASE(e2m1, e3m2, ".kind::f8f6f4.", "")
  CHOP_MMA_CASE(e2m1, e2m3, ".kind::f8f6f4.", "")
  CHOP_MMA_CASE(e2m1, e2m1, ".kind::f8f6f4.", "")
  static_assert(EA != EA, "unsupported MMA element combination");
}
#undef CHOP_MMA_CASE

template<Element EA, Element EB, uint16_t ByteA = 0, uint16_t ThreadA = 0,
         uint16_t ByteB = 0, uint16_t ThreadB = 0>
__device__ __forceinline__ void mma_mxf8f6f4(
    D4& d, const A4& a, const B2& b, uint32_t scale_a, uint32_t scale_b) {
#define CHOP_SCALED_CASE(A, B)                                                  \
  if constexpr (EA == Element::A && EB == Element::B) {                         \
    asm volatile(                                                               \
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X."           \
        "m16n8k32.row.col.f32." #A "." #B ".f32.ue8m0 "                         \
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "                 \
        "{%10}, {%11,%12}, {%13}, {%14,%15};"                                    \
        : "+f"(d.v[0]), "+f"(d.v[1]), "+f"(d.v[2]), "+f"(d.v[3])                 \
        : "r"(a.v[0]), "r"(a.v[1]), "r"(a.v[2]), "r"(a.v[3]),                      \
          "r"(b.v[0]), "r"(b.v[1]), "r"(scale_a), "n"(ByteA), "n"(ThreadA),        \
          "r"(scale_b), "n"(ByteB), "n"(ThreadB));                                \
  } else
  CHOP_SCALED_CASE(e4m3, e4m3)
  CHOP_SCALED_CASE(e4m3, e5m2)
  CHOP_SCALED_CASE(e4m3, e3m2)
  CHOP_SCALED_CASE(e4m3, e2m3)
  CHOP_SCALED_CASE(e4m3, e2m1)
  CHOP_SCALED_CASE(e5m2, e4m3)
  CHOP_SCALED_CASE(e5m2, e5m2)
  CHOP_SCALED_CASE(e5m2, e3m2)
  CHOP_SCALED_CASE(e5m2, e2m3)
  CHOP_SCALED_CASE(e5m2, e2m1)
  CHOP_SCALED_CASE(e3m2, e4m3)
  CHOP_SCALED_CASE(e3m2, e5m2)
  CHOP_SCALED_CASE(e3m2, e3m2)
  CHOP_SCALED_CASE(e3m2, e2m3)
  CHOP_SCALED_CASE(e3m2, e2m1)
  CHOP_SCALED_CASE(e2m3, e4m3)
  CHOP_SCALED_CASE(e2m3, e5m2)
  CHOP_SCALED_CASE(e2m3, e3m2)
  CHOP_SCALED_CASE(e2m3, e2m3)
  CHOP_SCALED_CASE(e2m3, e2m1)
  CHOP_SCALED_CASE(e2m1, e4m3)
  CHOP_SCALED_CASE(e2m1, e5m2)
  CHOP_SCALED_CASE(e2m1, e3m2)
  CHOP_SCALED_CASE(e2m1, e2m3)
  CHOP_SCALED_CASE(e2m1, e2m1)
  static_assert(EA != EA, "unsupported MMA element combination");
#undef CHOP_SCALED_CASE
}

template<int ScaleVec, bool E4M3Scale = false,
         uint16_t ByteA = 0, uint16_t ThreadA = 0,
         uint16_t ByteB = 0, uint16_t ThreadB = 0>
__device__ __forceinline__ void mma_mxf4nvf4(
    D4& d, const A4& a, const B2& b, uint32_t scale_a, uint32_t scale_b) {
#define CHOP_FP4_MMA(VEC, SCALE)                                                \
  asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::"       \
               VEC ".m16n8k64.row.col.f32.e2m1.e2m1.f32." SCALE " "            \
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "          \
               "{%10}, {%11,%12}, {%13}, {%14,%15};"                             \
               : "+f"(d.v[0]), "+f"(d.v[1]), "+f"(d.v[2]), "+f"(d.v[3])        \
               : "r"(a.v[0]), "r"(a.v[1]), "r"(a.v[2]), "r"(a.v[3]),               \
                 "r"(b.v[0]), "r"(b.v[1]), "r"(scale_a), "n"(ByteA), "n"(ThreadA), \
                 "r"(scale_b), "n"(ByteB), "n"(ThreadB))
  if constexpr (ScaleVec == 2 && !E4M3Scale) {
    CHOP_FP4_MMA("2X", "ue8m0");
  } else if constexpr (ScaleVec == 4 && !E4M3Scale) {
    CHOP_FP4_MMA("4X", "ue8m0");
  } else if constexpr (ScaleVec == 4 && E4M3Scale) {
    CHOP_FP4_MMA("4X", "ue4m3");
  } else {
    static_assert(ScaleVec == 2 || ScaleVec == 4, "unsupported FP4 scale vector");
  }
#undef CHOP_FP4_MMA
}

template<uint16_t ByteA = 0, uint16_t ThreadA = 0,
         uint16_t ByteB = 0, uint16_t ThreadB = 0>
__device__ __forceinline__ void mma_mxf4(
    D4& d, const A4& a, const B2& b, uint32_t scale_a, uint32_t scale_b) {
  asm volatile("mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X."
               "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
               "{%10}, {%11,%12}, {%13}, {%14,%15};"
               : "+f"(d.v[0]), "+f"(d.v[1]), "+f"(d.v[2]), "+f"(d.v[3])
               : "r"(a.v[0]), "r"(a.v[1]), "r"(a.v[2]), "r"(a.v[3]),
                 "r"(b.v[0]), "r"(b.v[1]), "r"(scale_a), "n"(ByteA),
                 "n"(ThreadA), "r"(scale_b), "n"(ByteB), "n"(ThreadB));
}

__device__ __forceinline__ void mbarrier_init(uint64_t* bar, uint32_t arrivals) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
               :: "r"(shared_address(bar)), "r"(arrivals) : "memory");
}
__device__ __forceinline__ void mbarrier_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
               :: "r"(shared_address(bar)) : "memory");
}
__device__ __forceinline__ void mbarrier_arrive_expect_tx(
    uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
               :: "r"(shared_address(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbarrier_expect_tx(
    uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.expect_tx.shared::cta.b64 [%0], %1;"
               :: "r"(shared_address(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ bool mbarrier_try_wait(uint64_t* bar, uint32_t phase) {
  uint32_t ready;
  asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 "
               "p, [%1], %2; selp.u32 %0, 1, 0, p; }"
               : "=r"(ready) : "r"(shared_address(bar)), "r"(phase) : "memory");
  return ready != 0;
}
__device__ __forceinline__ void mbarrier_wait(uint64_t* bar, uint32_t phase) {
  while (!mbarrier_try_wait(bar, phase)) {}
}
__device__ __forceinline__ bool mbarrier_test_wait(
    uint64_t* bar, uint32_t phase) {
  uint32_t ready;
  asm volatile("{ .reg .pred p; mbarrier.test_wait.parity.shared::cta.b64 "
               "p, [%1], %2; selp.u32 %0, 1, 0, p; }"
               : "=r"(ready) : "r"(shared_address(bar)), "r"(phase) : "memory");
  return ready != 0;
}
__device__ __forceinline__ void mbarrier_inval(uint64_t* bar) {
  asm volatile("mbarrier.inval.shared::cta.b64 [%0];"
               :: "r"(shared_address(bar)) : "memory");
}
__device__ __forceinline__ void fence_mbarrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

template<int Rank>
__device__ __forceinline__ void tma_load(
    void* dst, const void* map, const int32_t (&coord)[Rank], uint64_t* bar) {
  const uint32_t s = shared_address(dst), m = shared_address(bar);
  const uint64_t g = reinterpret_cast<uint64_t>(map);
  if constexpr (Rank == 1)
    asm volatile("cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2}], [%3];" :: "r"(s), "l"(g), "r"(coord[0]), "r"(m) : "memory");
  else if constexpr (Rank == 2)
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2,%3}], [%4];" :: "r"(s), "l"(g), "r"(coord[0]), "r"(coord[1]), "r"(m) : "memory");
  else if constexpr (Rank == 3)
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2,%3,%4}], [%5];" :: "r"(s), "l"(g), "r"(coord[0]), "r"(coord[1]), "r"(coord[2]), "r"(m) : "memory");
  else if constexpr (Rank == 4)
    asm volatile("cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2,%3,%4,%5}], [%6];" :: "r"(s), "l"(g), "r"(coord[0]), "r"(coord[1]), "r"(coord[2]), "r"(coord[3]), "r"(m) : "memory");
  else if constexpr (Rank == 5)
    asm volatile("cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2,%3,%4,%5,%6}], [%7];" :: "r"(s), "l"(g), "r"(coord[0]), "r"(coord[1]), "r"(coord[2]), "r"(coord[3]), "r"(coord[4]), "r"(m) : "memory");
  else
    static_assert(Rank >= 1 && Rank <= 5, "TMA rank must be 1 to 5");
}

__device__ __forceinline__ void cp_async_bulk(
    void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
               "[%0], [%1], %2, [%3];"
               :: "r"(shared_address(dst)), "l"(src), "r"(bytes),
                  "r"(shared_address(bar)) : "memory");
}
__device__ __forceinline__ void cp_async_bulk_store(
    void* dst, const void* src, uint32_t bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group "
               "[%0], [%1], %2;"
               :: "l"(dst), "r"(shared_address(src)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void cp_reduce_add_f32(
    float* dst, const float* src, uint32_t bytes) {
  asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 "
               "[%0], [%1], %2;"
               :: "l"(dst), "r"(shared_address(src)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void cp_reduce_add_bf16(
    uint32_t* dst, const uint32_t* src, uint32_t bytes) {
  asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16 "
               "[%0], [%1], %2;"
               :: "l"(dst), "r"(shared_address(src)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void cp_bulk_commit() {
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}
template<int Pending = 0>
__device__ __forceinline__ void cp_bulk_wait() {
  asm volatile("cp.async.bulk.wait_group %0;" :: "n"(Pending) : "memory");
}
__device__ __forceinline__ void tma_store(
    const void* map, const int32_t (&coord)[2], const void* src) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group "
               "[%0, {%1,%2}], [%3];"
               :: "l"(map), "r"(coord[0]), "r"(coord[1]),
                  "r"(shared_address(src)) : "memory");
}
__device__ __forceinline__ void prefetch_tensor_l2(
    const void* map, const int32_t (&coord)[2]) {
  asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile "
               "[%0, {%1,%2}];"
               :: "l"(map), "r"(coord[0]), "r"(coord[1]));
}

__device__ __forceinline__ void griddep_wait() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}
__device__ __forceinline__ void griddep_launch_dependents() {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
__device__ __forceinline__ void elect_sync(uint32_t& elected) {
  asm volatile("{ .reg .pred p; elect.sync _|p, 0xffffffff; "
               "selp.u32 %0, 1, 0, p; }" : "=r"(elected));
}
template<int Registers>
__device__ __forceinline__ void setmaxnreg_inc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(Registers));
}
template<int Registers>
__device__ __forceinline__ void setmaxnreg_dec() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(Registers));
}

} // namespace chop::ptx
