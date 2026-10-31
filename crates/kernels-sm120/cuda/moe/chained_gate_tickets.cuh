#pragma once
#include "chained_lanes_scale_tiles.cuh"

// The absorbing gate queue finishing its own expert strips, so no gate
// finisher launch reads the K-block partials back. After an item's last
// pass its CTA counts the item on the strip's ticket; the CTA whose item
// arrives last of the strip's K blocks then does the gate finisher's work
// for every routed row of the strip's 64 columns, reading the partials the
// strip's items have just written. Each output folds its gate and up
// chains by ascending K block from +0 with the finisher's adds, takes the
// same epilogue and BF16 rounding, and each 32-column group the same ceil
// quantizer, status flags and per-token sticky-error ticket, so every
// intermediate, code, scale and status is the finisher's.
//
// Each warp finishes its own routes of the strip (route i on warp i % 4):
// lane half h = lane / 16 folds the gate (h = 0) or up (h = 1) chains of
// four columns at 4 * (lane % 16) by ascending K block, ten blocks per
// batch of loads. Once every warp has stored its codes, the last warp
// arrives on the token tickets for all the strip's rows.
//
// The strip tickets (one word per expert and strip) start every launch at
// zero: they are zeroed at create, only this queue writes them, its
// launches are serialized on one stream, and a launch that does not finish
// ends the context. Every strip of an expert with routes receives exactly
// one arrival per K block, so its last arrival resets it; an arrival of at
// least the block count means corrupted state and traps. An arrival only
// releases the item's partials; the last arrival alone acquires the
// others', so the other items keep their L1 lines.
namespace chop::moe::lanes {
template<class C, uint32_t FloorBits, class K,
    class Clamp = typename C::clamp,
    class Quantization = chained_policy::CeilE4M3Group32<FloorBits>,
    class RouteWeight = chained_policy::BeforeQuantizeRouteWeight,
    class Output = chained_policy::Bf16IntermediateF32Output,
    class Reduction = chained_policy::OrderedF32>
struct StripFinish {
  static constexpr int Strips = C::n / Columns, Warps = Threads / 32;
  static constexpr int Blocks = Layout<C>::blocks, Span = 10;
  const float* route_weights;
  uint16_t* intermediate;
  uint8_t* codes;
  uint8_t* scales;
  uint32_t* status;
  uint32_t* tickets;
  uint32_t* strip_tickets;

  // Every thread calls this after the item's last pass and its barrier.
  __device__ __forceinline__ void strip(const float* partial, int expert, int column,
      int count, int first, const int32_t* sorted_rows) const {
    static_assert(C::gate_up && C::halves == 2 && Columns == 64 && Threads == 128 &&
                  C::n % Columns == 0 && C::n % 32 == 0 && Blocks % Span == 0,
                  "a strip is two 32-column groups and two lines of each partial row");
    __shared__ uint32_t last;
    __shared__ uint16_t values[Warps][Columns];
    if (threadIdx.x == 0) {
      uint32_t* ticket = strip_tickets + expert * Strips + column / Columns;
      uint32_t arrived;
      asm volatile("atom.add.release.gpu.u32 %0, [%1], 1;"
          : "=r"(arrived) : "l"(ticket) : "memory");
      if (arrived >= Blocks) __trap();
      last = arrived == Blocks - 1;
      if (last) {
        *ticket = 0;
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
      }
    }
    __syncthreads();
    if (!last) return;
    if (reinterpret_cast<uintptr_t>(partial) & (K::retire ? 127u : 15u)) __trap();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int h = lane >> 4, q = lane & 15;
    uint16_t* const own = values[warp];
    for (int i = warp; i < count; i += Warps) {
      const int route = sorted_rows[first + i];
      const float* row = partial + (uint64_t(route) * 2 + h) * Blocks * C::n + column + 4 * q;
      float sum[4] = {};
#pragma unroll 1
      for (int base = 0; base < Blocks; base += Span) {
        float4 value[Span];
#pragma unroll
        for (int b = 0; b < Span; ++b)
          value[b] = __ldcg(reinterpret_cast<const float4*>(row + uint64_t(base + b) * C::n));
#pragma unroll
        for (int b = 0; b < Span; ++b) {
          sum[0] = Reduction::add(sum[0], value[b].x);
          sum[1] = Reduction::add(sum[1], value[b].y);
          sum[2] = Reduction::add(sum[2], value[b].z);
          sum[3] = Reduction::add(sum[3], value[b].w);
        }
      }
      if constexpr (K::retire) {
        // The warp has used its loads: two lines of each (half, K block)
        // row of the route.
        __syncwarp();
        for (int line = lane; line < 4 * Blocks; line += 32)
          knobs::discard_line(partial + ((uint64_t(route) * 2 + line % 2) * Blocks + line / 4) *
                              C::n + column + 32 * (line / 2 % 2));
      }
      // Lanes 0-15 hold the gate sums of their four columns and lanes 16-31
      // the up sums of the same columns.
      float up[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) up[j] = __shfl_down_sync(0xffffffffu, sum[j], 16);
      if (h == 0) {
        const float weight = route_weights[route];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint16_t out = Output::intermediate(
              RouteWeight::template apply<Clamp>(sum[j], up[j], weight));
          own[4 * q + j] = out;
          intermediate[uint64_t(route) * C::n + column + 4 * q + j] = out;
        }
      }
      __syncwarp();
      uint32_t flags = 0;
#pragma unroll
      for (int g = 0; g < 2; ++g) {
        auto value = Quantization::quantize(own + 32 * g, lane * 2);
        flags |= value.flags;
        const int group = column / 32 + g;
        if (lane == 0) scales[uint64_t(route) * (C::n / 32) + group] = value.scale_code;
        if (lane < 16) {
          const uint64_t at = uint64_t(route) * C::n + group * 32 + lane * 2;
          codes[at] = uint8_t(value.codes);
          codes[at + 1] = uint8_t(value.codes >> 8);
        }
      }
#pragma unroll
      for (int step = 16; step; step >>= 1)
        flags |= __shfl_xor_sync(0xffffffffu, flags, step);
      if (lane == 0 && flags) atomicOr(status + route / C::top_k, flags);
      __syncwarp();
    }
    __syncthreads();
    if (warp != Warps - 1) return;
    // One arrival per routed row and strip once the strip's codes are
    // stored, the warp's lanes for different rows at once: the token's last
    // of its strips times its routes clears its whole quantized row on any
    // sticky error, as the gate finisher's last CTA does.
    for (int base = 0; base < count; base += 32) {
      int token = 0;
      bool clear = false;
      if (base + lane < count) {
        token = sorted_rows[first + base + lane] / C::top_k;
        uint32_t arrived;
        asm volatile("atom.add.acq_rel.gpu.u32 %0, [%1], 1;"
            : "=r"(arrived) : "l"(tickets + token) : "memory");
        clear = arrived == Strips * C::top_k - 1 && __ldcg(status + token);
      }
      uint32_t pending = __ballot_sync(0xffffffffu, clear);
      __syncwarp();
      while (pending) {
        const int cleared = __shfl_sync(0xffffffffu, token, __ffs(pending) - 1);
        pending &= pending - 1;
        for (int at = lane; at < C::top_k * C::n; at += 32)
          codes[uint64_t(cleared) * C::top_k * C::n + at] = 0;
        for (int at = lane; at < C::top_k * C::n / 32; at += 32)
          scales[uint64_t(cleared) * (C::top_k * C::n / 32) + at] = 127;
      }
    }
  }
};

// The absorbing gate queue with K::finish: its own arguments, then the gate
// finisher's outputs and tickets, then the strip tickets.
template<class C, class Scales, uint32_t FloorBits, class K>
__device__ void absorbing_finish(const uint8_t* input, const uint8_t* weight,
    const uint8_t* weight_scales, const uint8_t* activation_scales,
    const int32_t* offsets, const int32_t* sorted_rows, const int32_t* padded_rows,
    const int32_t* live_tokens, void* scratch, const float* route_weights,
    uint16_t* intermediate, uint8_t* codes, uint8_t* scales, uint32_t* status,
    uint32_t* tickets, uint32_t* strip_tickets) {
  static_assert(K::finish);
  absorbing<C, Scales, K>(input, weight, weight_scales, activation_scales, offsets,
      sorted_rows, padded_rows, live_tokens, scratch,
      StripFinish<C, FloorBits, K>{route_weights, intermediate, codes, scales, status,
                                   tickets, strip_tickets});
}
} // namespace chop::moe::lanes
