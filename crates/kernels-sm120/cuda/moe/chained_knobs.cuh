#pragma once
#include "chained_streamed.cuh"

// Cache and schedule knobs of the direct MoE queues and finishers. Every
// knob keeps each loaded value, product, add and stored value of its body,
// and `Served` sets none, so the served entries compile as before.
// - Stream: weight code loads skip L1 and carry an L2 evict_first policy;
//   each weight byte is read once per step.
// - Keep: K-block partial stores carry an L2 evict_last policy.
// - Retire: whoever folds a partial line discards it once its warp has read
//   it (`discard.global.L2`), so the dead line is never written back. Each
//   line is written by one queue item and read by one warp of one step.
namespace chop::moe::knobs {
template<bool Stream, bool Keep, bool Retire>
struct Set {
  static constexpr bool stream = Stream, keep = Keep, retire = Retire;
};
using Served = Set<false, false, false>;

// A 16-byte weight load under Stream: the vector branch of streamed::load16
// with the policy created in place, so no policy register stays live.
__device__ __forceinline__ uint4 stream16(const uint8_t* p) {
  uint4 v;
  asm("{ .reg .b64 policy; createpolicy.fractional.L2::evict_first.b64 policy, 1.0; "
      "ld.global.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], policy; }"
      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

template<class K>
__device__ __forceinline__ uint4 weight16(const uint8_t* p) {
  if constexpr (K::stream)
    if ((reinterpret_cast<uintptr_t>(p) & 15u) == 0) return stream16(p);
  return streamed::load16(p);
}

template<class K>
__device__ __forceinline__ void store_partial(float* p, float value) {
  if constexpr (K::keep)
    asm volatile("{ .reg .b64 policy; createpolicy.fractional.L2::evict_last.b64 policy, 1.0; "
                 "st.global.L2::cache_hint.f32 [%0], %1, policy; }"
                 :: "l"(p), "f"(value) : "memory");
  else
    *p = value;
}

__device__ __forceinline__ void discard_line(const void* line) {
  asm volatile("discard.global.L2 [%0], 128;" :: "l"(line) : "memory");
}

// Retire for a gate finisher's warp: discards the lines it read of
// `route`'s partials, one per (half, K block) at its 32 columns from
// `column`. The whole warp calls this once every lane has used its loads.
template<int Blocks, int N>
__device__ __forceinline__ void retire_gate(const float* partial, int route, int column) {
  static_assert(N % 32 == 0, "a row of partials starts a line");
  if (reinterpret_cast<uintptr_t>(partial) & 127u) __trap();
  __syncwarp();
  for (int line = threadIdx.x & 31; line < 2 * Blocks; line += 32)
    discard_line(partial + ((uint64_t(route) * 2 + line % 2) * Blocks + line / 2) * N + column);
}
} // namespace chop::moe::knobs
