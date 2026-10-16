#pragma once

#include <math.h>
#include <stdint.h>
#include "../common/convert.cuh"
#include "../common/ptx.cuh"
#include "swiglu_clamp.cuh"

// Grouped and dense SwiGLU MLP projections with 128-K accumulation chains
// (README "Chained expert GEMM"). One CTA of four warps owns 64 output
// columns of one row tile: a tile of `RowTile` routed rows of one expert
// (grouped) or `RowTile` tokens (dense). Each warp owns 16 output columns;
// gate/up warps hold the gate and the up rows of their columns, so the
// clamped SwiGLU epilogue needs no exchange.
//
// Arithmetic, fixed per instance:
// - MMA operands are exact BF16: an FP8 E4M3 code times the power of two
//   of its block scale; an E2M1 code times its UE4M3 scale and the power of
//   two of its expert global (`mul.rn.bf16x2`, at most six significant
//   bits); an E4M3 code scaled by its UE8M0 (`cvt.rn.scaled`, trapping on
//   an inexact product); or a stored BF16 value. An F32 scale below 1
//   splits into that power of two (at least 2^-100) and an exact residual;
//   larger scales stay whole in the residual, and a zero scale zeroes the
//   operand (`split_scale`). So a chain overflows only where the unsplit
//   chain would, and a zero-scale block adds exactly zero.
// - Every MMA chain starts from zero and spans one 128-K block (eight K16
//   steps, ascending K). The block's F32 temporary is then added into the
//   accumulator in K order: `fmaf(chain, residual, sum)` with the FP8
//   128 x 128 block scale's residual, `sum + chain` otherwise. No float
//   atomics; no split K.
// - NVFP4 global residuals multiply the full sums once (`__fmul_rn`) before
//   the epilogue.
// - Gate/up writes `BF16(clamped_swiglu(gate, up))`. Grouped down writes the
//   F32 `sum * route_weight` per routed row for the route sum
//   (`routed_sum`), which adds a token's routes in top-k order and rounds to
//   BF16 once, or writes the F32 sum for a later F32 add. Dense down writes
//   `BF16(sum)`.
//
// The grid is capacity-sized; every CTA reads the live counts after
// `griddepcontrol.wait` and CTAs past them exit. Routed row r (route index
// token * top_k + k) is read from and written to row r of the route-indexed
// buffers, so a launch writes exactly the live rows.
namespace chop::moe::chained {

enum class Weights : int {
  fp8_block128 = 0,  // E4M3 [rows, K], F32 [rows / 128, K / 128] per expert
  nvfp4 = 1,         // E2M1 pairs [rows, K / 2] (even K low), UE4M3 [rows, K / 16], F32 globals
  mxfp8 = 2,         // E4M3 [rows, K], UE8M0 [rows, K / 32]
  bf16 = 3,          // BF16 [rows, K]
};

enum class Projection : int { gate_up = 0, down = 1 };

constexpr int kThreads = 128;
constexpr int kTileN = 64;     // output columns per CTA
constexpr int kStepK = 64;     // K per shared-memory step
constexpr int kStride = 72;    // BF16 elements per shared row (conflict-free 32-bit loads)

// One instance's constants. `Experts` is 1 for a dense projection. Gate/up
// weights of an expert are [2 N, K]: N gate rows, then N up rows.
template<Weights W, Projection P, int Experts, int TopK, int N, int K, int RowTile,
         bool Grouped, class Clamp>
struct Config {
  static constexpr Weights weights = W;
  static constexpr bool gate_up = P == Projection::gate_up;
  static constexpr int experts = Experts;
  static constexpr int top_k = TopK;
  static constexpr int n = N;
  static constexpr int k = K;
  static constexpr int row_tile = RowTile;
  static constexpr bool grouped = Grouped;
  using clamp = Clamp;
  static constexpr int halves = gate_up ? 2 : 1;
  static constexpr int weight_rows = halves * N;
  static constexpr int tile_rows = halves * kTileN;
  static_assert(N % kTileN == 0 && K % 128 == 0, "64-column tiles and whole 128-K blocks");
  static_assert(RowTile == 16 || RowTile == 32 || RowTile == 64, "row tile");
  static_assert(Grouped || Experts == 1, "a dense projection has one weight");
  static_assert(gate_up || !Clamp::enabled, "the clamp belongs to gate/up");
  static_assert(W != Weights::fp8_block128 || N % 128 == 0, "whole 128-row scale blocks");
};

__device__ __forceinline__ uint32_t lds32(const uint16_t* p) {
  return *reinterpret_cast<const uint32_t*>(p);
}

__device__ __forceinline__ uint32_t mul_bf16x2(uint32_t a, uint32_t b) {
  uint32_t out;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(out) : "r"(a), "r"(b));
  return out;
}

// An F32 scale `s` as `2^e * residual` with `pow2` holding 2^e as a BF16
// pair for the operand:
// - Only scales below 1 fold: `e` is s's exponent clamped to [-100, 0]. A
//   fold only shrinks products and partial sums by an exact power of two,
//   so a chain overflows only where the unsplit chain (codes times
//   activations) would; scales of 1 and above stay whole in the residual,
//   which is the unsplit arithmetic.
// - A zero scale (either sign) gives a zero operand, so its block adds
//   exactly zero even against activations whose raw products overflow.
// - An infinite or NaN scale stays whole in the residual (admission refuses
//   them; `chained::admit_scales`).
// `residual = s * 2^-e` is exact: in [1, 2) for 2^-100 <= s < 1, s itself
// for s >= 1, and at least 2^-49 below 2^-100. An E4M3 code, or an E2M1
// code times a UE4M3 scale, times 2^e is an exact normal BF16 (at least
// 2^-110). The folded chain equals 2^e times the unsplit chain, bit for bit
// after the residual, while no product or partial sum leaves F32's normal
// range; a chain that underflows may round differently in subnormal bits.
struct SplitScale {
  uint32_t pow2;
  float residual;
};

__device__ __forceinline__ SplitScale split_scale(float s) {
  const uint32_t bits = __float_as_uint(s);
  const int biased = (bits >> 23) & 0xff;
  if ((bits & 0x7fffffffu) == 0) return {0u, s};
  const int e = biased == 255 ? 0 : min(max(biased - 127, -100), 0);
  const uint32_t pow2 = uint32_t(127 + e) << 7;
  return {pow2 | (pow2 << 16), __fmul_rn(s, __uint_as_float(uint32_t(127 - e) << 23))};
}

// Two E4M3 codes scaled by a UE8M0 into BF16; traps on an inexact or
// non-finite product, which only extreme scales can give.
__device__ __forceinline__ uint32_t scaled_e4m3x2(uint16_t codes, uint8_t scale) {
  uint32_t out;
  const uint16_t scale_pair = uint16_t(scale) * 0x0101;
  asm volatile("cvt.rn.scaled::n2::ue8m0.bf16x2.e4m3x2 %0, %1, %2;"
               : "=r"(out) : "h"(codes), "h"(scale_pair));
  if (scale < 3 || scale > 246) {
    const auto raw = convert::unpack_f32_e4m3(codes);
    const float factor = ldexpf(1.0f, int(scale) - 127);
    const float lo = raw.lo * factor, hi = raw.hi * factor;
    const auto decoded = convert::bf16x2_to_f32(out);
    if (scale == 255 || !isfinite(lo) || !isfinite(hi) || decoded.lo != lo || decoded.hi != hi)
      asm volatile("trap;");
  }
  return out;
}

// Weight row `row` of the CTA's shared tile in the expert's weight rows.
template<class C>
__device__ __forceinline__ int weight_row(int n0, int row) {
  if constexpr (C::gate_up)
    return (row >= kTileN ? C::n : 0) + n0 + (row & (kTileN - 1));
  else
    return n0 + row;
}

// The FP8 block scale of an expert's weight row and 128-K block.
template<class C>
__device__ __forceinline__ uint64_t block_scale_index(int expert, int row, int block) {
  return (uint64_t(expert) * (C::weight_rows / 128) + row / 128) * (C::k / 128) + block;
}

// Copies and decodes K step `step` of the CTA's weight rows into `tile`.
template<class C>
__device__ __forceinline__ void load_weights(
    uint16_t (*tile)[kStride], const uint8_t* weight, const float* block_scales,
    const uint8_t* local_scales, const float* globals, int expert, int n0, int step) {
  constexpr int K = C::k;
  const int k0 = step * kStepK;
  if constexpr (C::weights == Weights::bf16) {
    // 128 bytes per row: eight 16-byte chunks.
#pragma unroll
    for (int i = 0; i < C::tile_rows * 8 / kThreads; ++i) {
      const int chunk = threadIdx.x + i * kThreads;
      const int row = chunk >> 3, part = chunk & 7;
      const uint64_t global = uint64_t(expert) * C::weight_rows + weight_row<C>(n0, row);
      const uint4 v = *reinterpret_cast<const uint4*>(
          weight + (global * K + k0 + part * 8) * 2);
      *reinterpret_cast<uint4*>(&tile[row][part * 8]) = v;
    }
  } else if constexpr (C::weights == Weights::nvfp4) {
    // 32 bytes per row: four 8-byte chunks of 16 values, one UE4M3 each.
#pragma unroll
    for (int i = 0; i < C::tile_rows * 4 / kThreads; ++i) {
      const int chunk = threadIdx.x + i * kThreads;
      const int row = chunk >> 2, part = chunk & 3;
      const uint64_t global = uint64_t(expert) * C::weight_rows + weight_row<C>(n0, row);
      const uint2 codes = *reinterpret_cast<const uint2*>(
          weight + global * (K / 2) + (k0 + part * 16) / 2);
      const uint8_t scale_code = local_scales[global * (K / 16) + (k0 + part * 16) / 16];
      const uint32_t scale = convert::unpack_bf16x2_e4m3(uint16_t(scale_code) * 0x0101);
      const int half = C::gate_up && row >= kTileN ? 1 : 0;
      const uint32_t pow2 = split_scale(globals[uint64_t(expert) * C::halves + half]).pow2;
      uint32_t out[8];
#pragma unroll
      for (int b = 0; b < 8; ++b) {
        const uint32_t word = b < 4 ? codes.x : codes.y;
        out[b] = mul_bf16x2(
            mul_bf16x2(convert::unpack_bf16x2_e2m1(uint16_t((word >> (8 * (b & 3))) & 0xff)), scale),
            pow2);
      }
      *reinterpret_cast<uint4*>(&tile[row][part * 16]) = make_uint4(out[0], out[1], out[2], out[3]);
      *reinterpret_cast<uint4*>(&tile[row][part * 16 + 8]) =
          make_uint4(out[4], out[5], out[6], out[7]);
    }
  } else {
    // FP8 and MXFP8: 64 bytes per row, four 16-byte chunks.
#pragma unroll
    for (int i = 0; i < C::tile_rows * 4 / kThreads; ++i) {
      const int chunk = threadIdx.x + i * kThreads;
      const int row = chunk >> 2, part = chunk & 3;
      const uint64_t global = uint64_t(expert) * C::weight_rows + weight_row<C>(n0, row);
      const uint4 codes = *reinterpret_cast<const uint4*>(weight + global * K + k0 + part * 16);
      const uint32_t words[4] = {codes.x, codes.y, codes.z, codes.w};
      uint32_t out[8];
      if constexpr (C::weights == Weights::mxfp8) {
        const uint8_t scale = local_scales[global * (K / 32) + (k0 + part * 16) / 32];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
          out[2 * w] = scaled_e4m3x2(uint16_t(words[w]), scale);
          out[2 * w + 1] = scaled_e4m3x2(uint16_t(words[w] >> 16), scale);
        }
      } else {
        const uint32_t pow2 = split_scale(block_scales[block_scale_index<C>(
            expert, weight_row<C>(n0, row), k0 / 128)]).pow2;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
          out[2 * w] = mul_bf16x2(convert::unpack_bf16x2_e4m3(uint16_t(words[w])), pow2);
          out[2 * w + 1] = mul_bf16x2(convert::unpack_bf16x2_e4m3(uint16_t(words[w] >> 16)), pow2);
        }
      }
      *reinterpret_cast<uint4*>(&tile[row][part * 16]) = make_uint4(out[0], out[1], out[2], out[3]);
      *reinterpret_cast<uint4*>(&tile[row][part * 16 + 8]) =
          make_uint4(out[4], out[5], out[6], out[7]);
    }
  }
}

// The input row of slot `slot`: a token row of x for grouped gate/up, the
// routed row of the intermediate for grouped down, the token row when dense.
template<class C>
__device__ __forceinline__ const uint16_t* input_row(const uint16_t* input, int route) {
  const int source = C::grouped && C::gate_up ? route / C::top_k : route;
  return input + uint64_t(source) * C::k;
}

template<class C>
__device__ __forceinline__ void load_rows(
    uint16_t (*tile)[kStride], const uint16_t* input, const int32_t* rows, int step) {
#pragma unroll
  for (int i = 0; i < C::row_tile * 8 / kThreads; ++i) {
    const int chunk = threadIdx.x + i * kThreads;
    const int slot = chunk >> 3, part = chunk & 7;
    const int route = rows[slot];
    uint4 v = make_uint4(0, 0, 0, 0);
    if (route >= 0)
      v = *reinterpret_cast<const uint4*>(input_row<C>(input, route) + step * kStepK + part * 8);
    *reinterpret_cast<uint4*>(&tile[slot][part * 8]) = v;
  }
}

// One projection. Unused pointers of a format or mode may be null:
// `block_scales` (FP8), `local_scales` (NVFP4, MXFP8), `globals` (NVFP4:
// gate and up per expert for gate/up, one per expert for down), the route
// metadata (grouped), `live_tokens` (dense) and `route_weights` (grouped
// down). `output` is BF16, except F32 for grouped down.
template<class C>
__device__ void run(
    const uint16_t* input, const uint8_t* weight, const float* block_scales,
    const uint8_t* local_scales, const float* globals, const int32_t* offsets,
    const int32_t* sorted_rows, const int32_t* num_padded_rows, const int32_t* live_tokens,
    const float* route_weights, void* output) {
  constexpr int RT = C::row_tile;
  constexpr int Halves = C::halves;
  constexpr int Fragments = RT / 8;
  __shared__ __align__(16) uint16_t weights_tile[C::tile_rows][kStride];
  __shared__ __align__(16) uint16_t rows_tile[RT][kStride];
  __shared__ int32_t rows[RT];
  __shared__ int32_t tile_expert;
  if (blockDim.x != kThreads) __trap();
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const int row0 = blockIdx.x * RT;
  const int n0 = blockIdx.y * kTileN;
  if constexpr (C::grouped) {
    const int live_rows = *num_padded_rows;
    if (live_rows > int(gridDim.x) * RT || live_rows % RT != 0) __trap();
    if (row0 >= live_rows) return;
    // Expert ranges are padded to whole row tiles, so one expert owns the tile.
    for (int e = threadIdx.x; e < C::experts; e += kThreads)
      if (offsets[e] <= row0 && row0 < offsets[e + 1]) tile_expert = e;
    for (int i = threadIdx.x; i < RT; i += kThreads) rows[i] = sorted_rows[row0 + i];
  } else {
    const int live = *live_tokens;
    if (live > int(gridDim.x) * RT) __trap();
    if (row0 >= live) return;
    if (threadIdx.x == 0) tile_expert = 0;
    for (int i = threadIdx.x; i < RT; i += kThreads) rows[i] = row0 + i < live ? row0 + i : -1;
  }
  __syncthreads();
  const int expert = tile_expert;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;

  ptx::D4 chain[Halves][Fragments], sum[Halves][Fragments];
#pragma unroll
  for (int h = 0; h < Halves; ++h)
#pragma unroll
    for (int f = 0; f < Fragments; ++f)
#pragma unroll
      for (int i = 0; i < 4; ++i) chain[h][f].v[i] = sum[h][f].v[i] = 0.f;

#pragma unroll 1
  for (int step = 0; step < C::k / kStepK; ++step) {
    __syncthreads();
    load_weights<C>(weights_tile, weight, block_scales, local_scales, globals, expert, n0, step);
    load_rows<C>(rows_tile, input, rows, step);
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < kStepK / 16; ++kk) {
      const int col = kk * 16 + 2 * t;
      ptx::A4 a[Halves];
#pragma unroll
      for (int h = 0; h < Halves; ++h) {
        const int base = h * kTileN + warp * 16;
        a[h].v[0] = lds32(&weights_tile[base + g][col]);
        a[h].v[1] = lds32(&weights_tile[base + g + 8][col]);
        a[h].v[2] = lds32(&weights_tile[base + g][col + 8]);
        a[h].v[3] = lds32(&weights_tile[base + g + 8][col + 8]);
      }
#pragma unroll
      for (int f = 0; f < Fragments; ++f) {
        ptx::B2 b;
        b.v[0] = lds32(&rows_tile[f * 8 + g][col]);
        b.v[1] = lds32(&rows_tile[f * 8 + g][col + 8]);
#pragma unroll
        for (int h = 0; h < Halves; ++h) ptx::mma_bf16(chain[h][f], a[h], b);
      }
    }
    if (step & 1) {
      // The 128-K block ends: add its chain into the sums in K order.
      const int block = step >> 1;
#pragma unroll
      for (int h = 0; h < Halves; ++h) {
        float residual = 1.f;
        if constexpr (C::weights == Weights::fp8_block128)
          residual = split_scale(
              block_scales[block_scale_index<C>(expert, h * C::n + n0 + warp * 16, block)])
              .residual;
#pragma unroll
        for (int f = 0; f < Fragments; ++f)
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            if constexpr (C::weights == Weights::fp8_block128)
              sum[h][f].v[i] = fmaf(chain[h][f].v[i], residual, sum[h][f].v[i]);
            else
              sum[h][f].v[i] = __fadd_rn(sum[h][f].v[i], chain[h][f].v[i]);
            chain[h][f].v[i] = 0.f;
          }
      }
    }
  }

  if constexpr (C::weights == Weights::nvfp4) {
#pragma unroll
    for (int h = 0; h < Halves; ++h) {
      const float global = split_scale(globals[uint64_t(expert) * Halves + h]).residual;
#pragma unroll
      for (int f = 0; f < Fragments; ++f)
#pragma unroll
        for (int i = 0; i < 4; ++i) sum[h][f].v[i] = __fmul_rn(sum[h][f].v[i], global);
    }
  }
#pragma unroll
  for (int f = 0; f < Fragments; ++f)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int route = rows[f * 8 + 2 * t + (i & 1)];
      if (route < 0) continue;
      const uint64_t at = uint64_t(route) * C::n + n0 + warp * 16 + g + (i >= 2 ? 8 : 0);
      if constexpr (C::gate_up) {
        const float value =
            clamp::clamped_swiglu<typename C::clamp>(sum[0][f].v[i], sum[Halves - 1][f].v[i]);
        static_cast<uint16_t*>(output)[at] = convert::bf16_from_f32(value);
      } else if constexpr (C::grouped) {
        static_cast<float*>(output)[at] = __fmul_rn(sum[0][f].v[i], route_weights[route]);
      } else {
        static_cast<uint16_t*>(output)[at] = convert::bf16_from_f32(sum[0][f].v[i]);
      }
    }
}

// A token's routed rows summed in top-k order in F32 (`__fadd_rn` from
// +0), then rounded to BF16 once (`Out` uint16_t) or written as F32 (`Out`
// float): routes whose global expert lies outside this rank's
// `[rank * LocalExperts, (rank + 1) * LocalExperts)` add nothing, so the
// output is this rank's partial (zero for a token with no local route).
// One CTA of 256 threads per token, four consecutive columns per thread
// and pass.
template<int TopK, int Width, int LocalExperts, class Out>
__device__ void routed_sum(
    const float* routed, const int32_t* ids, const int32_t* live_tokens, uint32_t rank,
    Out* output) {
  static_assert(Width % 1024 == 0, "four columns per thread and pass");
  constexpr bool f32_output = sizeof(Out) == 4;
  static_assert(f32_output || sizeof(Out) == 2, "BF16 bits or F32 output");
  if (blockDim.x != 256) __trap();
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const int live = *live_tokens;
  if (live > int(gridDim.x)) __trap();
  const int token = blockIdx.x;
  if (token >= live) return;
  bool local[TopK];
#pragma unroll
  for (int k = 0; k < TopK; ++k) {
    const int id = ids[uint64_t(token) * TopK + k];
    local[k] = id >= 0 && uint32_t(id) / LocalExperts == rank;
  }
#pragma unroll
  for (int pass = 0; pass < Width / 1024; ++pass) {
    const int col = pass * 1024 + threadIdx.x * 4;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int k = 0; k < TopK; ++k) {
      if (!local[k]) continue;
      const float4 v = *reinterpret_cast<const float4*>(
          routed + (uint64_t(token) * TopK + k) * Width + col);
      acc[0] = __fadd_rn(acc[0], v.x);
      acc[1] = __fadd_rn(acc[1], v.y);
      acc[2] = __fadd_rn(acc[2], v.z);
      acc[3] = __fadd_rn(acc[3], v.w);
    }
    if constexpr (f32_output) {
      *reinterpret_cast<float4*>(output + uint64_t(token) * Width + col) =
          make_float4(acc[0], acc[1], acc[2], acc[3]);
    } else {
      const uint32_t lo = convert::bf16x2_from_f32({acc[0], acc[1]});
      const uint32_t hi = convert::bf16x2_from_f32({acc[2], acc[3]});
      *reinterpret_cast<uint2*>(output + uint64_t(token) * Width + col) = make_uint2(lo, hi);
    }
  }
}

}  // namespace chop::moe::chained
