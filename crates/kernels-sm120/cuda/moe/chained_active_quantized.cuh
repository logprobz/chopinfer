#pragma once

#include "chained_quantized.cuh"
#include "chained_block_warps.cuh"

// Compact active-expert tiles, with four independent 128-K chains per CTA.
// Each chain uses the column engine's operands and eight ascending K16 MMAs.
// The four chains fold in K order. A persistent grid walks only live tiles;
// inactive experts contribute no items. All metadata follows the route wait.
namespace chop::moe::active_quantized {
using block_warps::cp_async16;
using block_warps::cp_async_commit;
using block_warps::cp_async_wait;
constexpr int Threads = 128, Warps = 4, TileN = 16, StageK = 32, Stages = 4;

template<class C> struct Layout {
  static constexpr int weight_stride = Warps * StageK / 2 + 16;
  static constexpr int row_stride = Warps * StageK + 16;
  static constexpr int weight_bytes = C::halves * TileN * weight_stride;
  static constexpr int stage_bytes = weight_bytes + C::row_tile * row_stride;
  static constexpr int blocks = C::loop_k / 128;
  static constexpr int rounds = (blocks + Warps - 1) / Warps;
  static constexpr int total = rounds * 4;
  static constexpr int elements = TileN * C::row_tile;
  static constexpr int per_thread = elements / Threads;
  static_assert(C::row_tile == 16 && C::quantized_before_down);
};

// The four-stage ring carries packed MXFP4 weights and MXFP8 activations.
// The last 128-K chain may have a zero 64-K tail. Entire absent chains are
// never folded, including their zero, so signed-zero behavior is preserved.
template<class C>
__device__ __forceinline__ void load_stage(
    uint8_t* stage, const uint8_t* input, const uint8_t* weight,
    const int32_t* rows, int expert, int n0, int s) {
  using L = Layout<C>;
  const int round = s / 4, sub = s % 4;
#pragma unroll
  for (int c = threadIdx.x; c < C::halves * TileN * Warps; c += Threads) {
    const int row = c / Warps, w = c % Warps;
    const int k = (round * Warps + w) * 128 + sub * StageK;
    const uint64_t global = uint64_t(expert) * C::weight_rows + block_warps::weight_row<C>(n0, row);
    const uint8_t* from = k < C::k ? weight + global * (C::k / 2) + k / 2 : weight;
    cp_async16(stage + row * L::weight_stride + w * 16, from, k < C::k);
  }
#pragma unroll
  for (int c = threadIdx.x; c < C::row_tile * Warps * 2; c += Threads) {
    const int row = c / (Warps * 2), part = c % (Warps * 2);
    const int w = part / 2, within = part % 2;
    const int k = (round * Warps + w) * 128 + sub * StageK + within * 16;
    const int route = rows[row];
    const int source = C::gate_up ? route / C::top_k : route;
    const bool valid = route >= 0 && k < C::k;
    const uint8_t* from = valid ? input + uint64_t(source) * C::k + k : input;
    cp_async16(stage + L::weight_bytes + row * L::row_stride + w * StageK + within * 16,
               from, valid);
  }
}

template<class C>
__device__ void run(
    const uint8_t* input, const uint8_t* weight, const uint8_t* weight_scales,
    const uint8_t* activation_scales, const int32_t* offsets, const int32_t* sorted_rows,
    const int32_t* padded_rows, const int32_t* live_tokens, const float* route_weights,
    void* output, const int32_t* active_experts) {
  using L = Layout<C>;
  constexpr int H = C::halves, F = C::row_tile / 8;
  __shared__ __align__(16) uint8_t stages[Stages][L::stage_bytes];
  __shared__ float chains[Warps][H][L::elements];
  __shared__ int32_t rows[C::row_tile], expert;
  if (blockDim.x != Threads) __trap();
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const int live = *live_tokens, padded = *padded_rows, active = active_experts[0];
  if (live < 0 || live > C::tokens_capacity || active < 0 || active > C::experts ||
      active > live * C::top_k || padded < 0 || padded % C::row_tile ||
      padded / C::row_tile > live * C::top_k) __trap();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  constexpr int Columns = C::n / TileN;
  const int items = (padded / C::row_tile) * Columns;
  for (int item = blockIdx.x; item < items; item += gridDim.x) {
    const int row0 = (item / Columns) * C::row_tile, n0 = (item % Columns) * TileN;
    if (threadIdx.x == 0) {
      // Offsets of the compact list are strictly ascending. A nonempty
      // expert can own several row tiles, all addressed by their offset.
      int lo = 0, hi = active;
      while (lo < hi) {
        const int mid = (lo + hi) / 2, e = active_experts[1 + mid];
        if (e < 0 || e >= C::experts) __trap();
        if (offsets[e] <= row0) lo = mid + 1; else hi = mid;
      }
      if (lo == 0) __trap();
      expert = active_experts[lo];
      if (row0 >= offsets[expert + 1]) __trap();
    }
    if (threadIdx.x < C::row_tile) {
      const int route = sorted_rows[row0 + threadIdx.x];
      if (route < -1 || route >= live * C::top_k) __trap();
      rows[threadIdx.x] = route;
    }
    __syncthreads();
#pragma unroll
    for (int s = 0; s < Stages - 1; ++s) {
      if (s < L::total) load_stage<C>(stages[s], input, weight, rows, expert, n0, s);
      cp_async_commit();
    }
    ptx::D4 chain[H][F];
    float sum[H][L::per_thread];
#pragma unroll
    for (int h = 0; h < H; ++h) {
#pragma unroll
      for (int j = 0; j < L::per_thread; ++j) sum[h][j] = 0.f;
#pragma unroll
      for (int f = 0; f < F; ++f)
#pragma unroll
        for (int i = 0; i < 4; ++i) chain[h][f].v[i] = 0.f;
    }
#pragma unroll 1
    for (int s = 0; s < L::total; ++s) {
      cp_async_wait<Stages - 2>();
      __syncthreads();
      const int next = s + Stages - 1;
      if (next < L::total)
        load_stage<C>(stages[next % Stages], input, weight, rows, expert, n0, next);
      cp_async_commit();
      const int round = s / 4, sub = s % 4;
      const int k = (round * Warps + warp) * 128 + sub * StageK;
      const uint8_t* stage = stages[s % Stages];
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        const int col = warp * StageK + kk * 16 + 2 * t;
        ptx::A4 a[H];
#pragma unroll
        for (int h = 0; h < H; ++h) {
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int row = h * TileN + g + ((i & 1) ? 8 : 0);
            const uint64_t global = uint64_t(expert) * C::weight_rows + block_warps::weight_row<C>(n0, row);
            const uint8_t scale = k < C::k ? weight_scales[global * (C::k / 32) + k / 32] : 127;
            a[h].v[i] = convert::scaled_mxfp4_bf16x2(
                stage[row * L::weight_stride + (col + (i >= 2 ? 8 : 0)) / 2], scale);
          }
        }
#pragma unroll
        for (int f = 0; f < F; ++f) {
          const int row = f * 8 + g, route = rows[row];
          const int source = C::gate_up ? route / C::top_k : route;
          const uint8_t scale = route >= 0 && k < C::k ?
              activation_scales[uint64_t(source) * (C::k / 32) + k / 32] : 127;
          ptx::B2 b;
#pragma unroll
          for (int i = 0; i < 2; ++i) {
            const uint16_t pair = *reinterpret_cast<const uint16_t*>(
                stage + L::weight_bytes + row * L::row_stride + col + i * 8);
            if ((pair & 0x7f) == 0x7f || ((pair >> 8) & 0x7f) == 0x7f) __trap();
            b.v[i] = chained::scaled_e4m3x2(pair, scale);
          }
#pragma unroll
          for (int h = 0; h < H; ++h) ptx::mma_bf16(chain[h][f], a[h], b);
        }
      }
      if (sub == 3) {
#pragma unroll
        for (int h = 0; h < H; ++h)
#pragma unroll
          for (int f = 0; f < F; ++f)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              chains[warp][h][(f * 4 + i) * 32 + lane] = chain[h][f].v[i];
              chain[h][f].v[i] = 0.f;
            }
        __syncthreads();
#pragma unroll
        for (int w = 0; w < Warps; ++w)
          if (round * Warps + w < L::blocks)
#pragma unroll
            for (int h = 0; h < H; ++h)
#pragma unroll
              for (int j = 0; j < L::per_thread; ++j)
                sum[h][j] = __fadd_rn(sum[h][j], chains[w][h][threadIdx.x + j * Threads]);
      }
    }
    cp_async_wait<0>();
#pragma unroll
    for (int j = 0; j < L::per_thread; ++j) {
      const int e = threadIdx.x + j * Threads;
      const int f = e / 128, i = (e / 32) % 4, owner = e % 32;
      const int route = rows[f * 8 + 2 * (owner & 3) + (i & 1)];
      if (route < 0) continue;
      const uint64_t at = uint64_t(route) * C::n + n0 + (owner >> 2) + (i >= 2 ? 8 : 0);
      if constexpr (C::gate_up) {
        const float value = chained::before_down<typename C::clamp>(sum[0][j], sum[H - 1][j], route_weights[route]);
        static_cast<uint16_t*>(output)[at] = convert::bf16_from_f32(value);
      } else {
        static_cast<float*>(output)[at] = sum[0][j];
      }
    }
    __syncthreads();
  }
}
} // namespace chop::moe::active_quantized
