#pragma once

#include "chained_block_warps.cuh"

// `chained_block_warps.cuh`'s block-warp schedule over NVFP4 weights
// (`chained::Schedule::BlockWarps` with `Weights::nvfp4`): every output
// element bitwise `chained::run`'s at the same inputs.
//
// As there, one CTA of four warps owns 16 output columns (gate and up rows
// for gate/up) of one 16-row tile, and warp w computes the 128-K blocks w,
// w + 4, ...: each block's MMA chain from zero over its eight K16 steps in
// ascending K, with `chained::run`'s operands and fragment layout. An NVFP4
// operand pair is `chained::load_weights`' expression: the E2M1 pair times
// its UE4M3 scale, then times the power of two of the expert's global for
// the half (`split_scale`), each a `mul.rn.bf16x2`. A K16 step spans one
// 16-value scale group, so each thread reads one UE4M3 code per weight row
// and step; the codes of a warp's next block load a round ahead. After
// every four blocks the warps hand their chains over shared memory and each
// thread adds its elements' four chains into its sums in K order with
// `__fadd_rn`, as `chained::run` does for every format without a block
// scale. The full sums are multiplied once by the global's residual
// (`__fmul_rn`), then `chained::run`'s epilogue follows.
//
// Loads: each warp's next 32 K of E2M1 codes (16 bytes per weight row) and
// activation rows go through a ring of four `cp.async` stages, as in
// `chained_block_warps.cuh`. No float atomics, no global workspace.
namespace chop::moe::block_warps_fp4 {

using block_warps::cp_async16;
using block_warps::cp_async_commit;
using block_warps::cp_async_wait;
using block_warps::kStageK;
using block_warps::kSubs;
using block_warps::kThreads;
using block_warps::kTileN;
using block_warps::kWarps;
using block_warps::Mutation;
using block_warps::weight_row;
using chained::Weights;

template<class C>
struct Layout {
  static constexpr int tile_rows = C::halves * kTileN;
  // E2M1 pairs: kStageK / 2 bytes per weight row and warp.
  static constexpr int warp_bytes = kStageK / 2;
  static constexpr int weight_stride = kWarps * warp_bytes + 16;
  static constexpr int row_stride = kWarps * kStageK * 2 + 16;
  static constexpr int weight_tile = tile_rows * weight_stride;
  static constexpr int stage_bytes = weight_tile + C::row_tile * row_stride;
  static constexpr int stages = 4;
  static constexpr int blocks = C::k / 128;
  static constexpr int rounds = blocks / kWarps;
  static constexpr int total = rounds * kSubs;
  static constexpr int elements = kTileN * C::row_tile;
  static constexpr int per_thread = elements / kThreads;
  static_assert(C::weights == Weights::nvfp4, "the NVFP4 block warps decode NVFP4 weights");
  static_assert(C::row_tile == 16, "block warps serve 16-row tiles");
  static_assert(blocks % kWarps == 0, "whole rounds of 128-K blocks");
  static_assert(C::n % kTileN == 0, "16-column tiles");
  static_assert(warp_bytes == 16, "one 16-byte copy per weight row and warp");
};

// Starts the copies of stage `s`: for each warp w, K [block * 128 + sub *
// kStageK, + kStageK) of block `round * kWarps + w`, E2M1 codes then
// activation rows, each warp's slice at column w * kStageK of the stage.
template<class C>
__device__ __forceinline__ void load_stage(
    uint8_t* stage, const uint16_t* input, const uint8_t* weight, const int32_t* rows,
    int expert, int n0, int s) {
  using L = Layout<C>;
  const int round = s / kSubs, sub = s % kSubs;
  constexpr int weight_total = L::tile_rows * kWarps;
#pragma unroll
  for (int c = threadIdx.x; c < weight_total; c += kThreads) {
    const int row = c / kWarps, w = c % kWarps;
    const int block = round * kWarps + w;
    const uint64_t global = uint64_t(expert) * C::weight_rows + weight_row<C>(n0, row);
    const uint8_t* source = weight + global * (C::k / 2) + (block * 128 + sub * kStageK) / 2;
    cp_async16(stage + row * L::weight_stride + w * L::warp_bytes, source, true);
  }
  constexpr int row_chunks = kStageK * 2 / 16;
  constexpr int row_total = C::row_tile * kWarps * row_chunks;
#pragma unroll
  for (int c = threadIdx.x; c < row_total; c += kThreads) {
    const int slot = c / (kWarps * row_chunks), part = c % (kWarps * row_chunks);
    const int w = part / row_chunks, within = part % row_chunks;
    const int block = round * kWarps + w;
    const int route = rows[slot];
    const uint16_t* source = route >= 0 ?
        chained::input_row<C>(input, route) + block * 128 + sub * kStageK + within * 8 : input;
    cp_async16(stage + L::weight_tile + slot * L::row_stride + (w * kStageK + within * 8) * 2,
               source, route >= 0);
  }
}

// One weight operand pair: columns `col` and `col + 1` (even) of tile row
// `row`, with the row's UE4M3 code for the pair's 16-value group.
template<class C>
__device__ __forceinline__ uint32_t weight_pair(const uint8_t* stage, int row, int col,
                                                uint8_t scale_code, uint32_t pow2) {
  using L = Layout<C>;
  const uint8_t codes = stage[row * L::weight_stride + col / 2];
  const uint32_t scale = convert::unpack_bf16x2_e4m3(uint16_t(scale_code) * 0x0101);
  return chained::mul_bf16x2(chained::mul_bf16x2(convert::unpack_bf16x2_e2m1(codes), scale), pow2);
}

// One projection: `chained::run`'s arguments and outputs.
template<class C, Mutation M = Mutation::none>
__device__ void run(
    const uint16_t* input, const uint8_t* weight, const float* block_scales,
    const uint8_t* local_scales, const float* globals, const int32_t* offsets,
    const int32_t* sorted_rows, const int32_t* num_padded_rows, const int32_t* live_tokens,
    const float* route_weights, void* output) {
  using L = Layout<C>;
  constexpr int RT = C::row_tile;
  constexpr int Halves = C::halves;
  constexpr int Fragments = RT / 8;
  (void)block_scales;
  __shared__ __align__(16) uint8_t stages[L::stages][L::stage_bytes];
  __shared__ float chains[kWarps][Halves][L::elements];
  __shared__ int32_t rows[RT];
  __shared__ int32_t tile_expert;
  if (blockDim.x != kThreads) __trap();
  ptx::griddep_launch_dependents();
  if constexpr (M != Mutation::drop_wait && M != Mutation::late_wait) ptx::griddep_wait();
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

#pragma unroll
  for (int s = 0; s < L::stages - 1; ++s) {
    if (s < L::total) load_stage<C>(stages[s], input, weight, rows, expert, n0, s);
    cp_async_commit();
  }

  // The power of two of each half's global goes into the operands; its
  // residual multiplies the full sums.
  uint32_t pow2[Halves];
  float global_residual[Halves];
#pragma unroll
  for (int h = 0; h < Halves; ++h) {
    const chained::SplitScale split = chained::split_scale(globals[uint64_t(expert) * Halves + h]);
    pow2[h] = split.pow2;
    global_residual[h] = split.residual;
  }

  // The UE4M3 codes of this thread's weight rows (g and g + 8 of each
  // half) over this warp's block of a round: eight codes, one per K16 step,
  // byte i of the pair holding step i's, loaded a round ahead (byte loads:
  // a scale view need not be 8-byte aligned).
  uint2 codes[Halves][2], next_codes[Halves][2];
  const auto load_codes = [&](int round) {
    const int block = round * kWarps + warp;
#pragma unroll
    for (int h = 0; h < Halves; ++h)
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const uint64_t global =
            uint64_t(expert) * C::weight_rows + weight_row<C>(n0, h * kTileN + g + 8 * r);
        const uint8_t* from = local_scales + global * (C::k / 16) + block * 8;
        uint32_t words[2] = {0u, 0u};
#pragma unroll
        for (int i = 0; i < 8; ++i) words[i / 4] |= uint32_t(from[i]) << (8 * (i & 3));
        next_codes[h][r] = make_uint2(words[0], words[1]);
      }
  };
  load_codes(0);

  ptx::D4 chain[Halves][Fragments];
  float sum[Halves][L::per_thread];
#pragma unroll
  for (int h = 0; h < Halves; ++h) {
#pragma unroll
    for (int j = 0; j < L::per_thread; ++j) sum[h][j] = 0.f;
#pragma unroll
    for (int f = 0; f < Fragments; ++f)
#pragma unroll
      for (int i = 0; i < 4; ++i) chain[h][f].v[i] = 0.f;
  }

#pragma unroll 1
  for (int s = 0; s < L::total; ++s) {
    cp_async_wait<L::stages - 2>();
    __syncthreads();
    {
      const int next = s + L::stages - 1;
      if (next < L::total)
        load_stage<C>(stages[next % L::stages], input, weight, rows, expert, n0, next);
      cp_async_commit();
    }
    const int round = s / kSubs, sub = s % kSubs;
    if (sub == 0) {
#pragma unroll
      for (int h = 0; h < Halves; ++h)
#pragma unroll
        for (int r = 0; r < 2; ++r) codes[h][r] = next_codes[h][r];
      if (round + 1 < L::rounds) load_codes(round + 1);
    }
    const uint8_t* stage = stages[s % L::stages];
#pragma unroll
    for (int kk = 0; kk < kStageK / 16; ++kk) {
      const int col = warp * kStageK + kk * 16 + 2 * t;
      // The step's 16-value group within the block: sub * 2 + kk.
      const int group = sub * (kStageK / 16) + kk;
      ptx::A4 a[Halves];
#pragma unroll
      for (int h = 0; h < Halves; ++h) {
        const int base = h * kTileN;
        const uint8_t low = uint8_t(
            (group < 4 ? codes[h][0].x : codes[h][0].y) >> (8 * (group & 3)));
        const uint8_t high = uint8_t(
            (group < 4 ? codes[h][1].x : codes[h][1].y) >> (8 * (group & 3)));
        a[h].v[0] = weight_pair<C>(stage, base + g, col, low, pow2[h]);
        a[h].v[1] = weight_pair<C>(stage, base + g + 8, col, high, pow2[h]);
        a[h].v[2] = weight_pair<C>(stage, base + g, col + 8, low, pow2[h]);
        a[h].v[3] = weight_pair<C>(stage, base + g + 8, col + 8, high, pow2[h]);
      }
#pragma unroll
      for (int f = 0; f < Fragments; ++f) {
        const uint8_t* row = stage + L::weight_tile + (f * 8 + g) * L::row_stride;
        ptx::B2 b;
        b.v[0] = *reinterpret_cast<const uint32_t*>(row + col * 2);
        b.v[1] = *reinterpret_cast<const uint32_t*>(row + (col + 8) * 2);
#pragma unroll
        for (int h = 0; h < Halves; ++h) ptx::mma_bf16(chain[h][f], a[h], b);
      }
    }
    if (sub == kSubs - 1) {
      // The round's blocks end: hand the chains over, then add them into
      // the sums in K order (warp w holds block round * kWarps + w).
#pragma unroll
      for (int h = 0; h < Halves; ++h)
#pragma unroll
        for (int f = 0; f < Fragments; ++f)
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            chains[warp][h][(f * 4 + i) * 32 + lane] = chain[h][f].v[i];
            chain[h][f].v[i] = 0.f;
          }
      __syncthreads();
#pragma unroll
      for (int i = 0; i < kWarps; ++i)
#pragma unroll
        for (int h = 0; h < Halves; ++h)
#pragma unroll
          for (int j = 0; j < L::per_thread; ++j) {
            const int w = M == Mutation::reversed_fold ? kWarps - 1 - i : i;
            sum[h][j] = __fadd_rn(sum[h][j], chains[w][h][threadIdx.x + j * kThreads]);
          }
    }
  }
  cp_async_wait<0>();
  if constexpr (M == Mutation::late_wait) ptx::griddep_wait();

#pragma unroll
  for (int h = 0; h < Halves; ++h)
#pragma unroll
    for (int j = 0; j < L::per_thread; ++j) sum[h][j] = __fmul_rn(sum[h][j], global_residual[h]);
#pragma unroll
  for (int j = 0; j < L::per_thread; ++j) {
    const int e = threadIdx.x + j * kThreads;
    const int f = e / 128, i = (e / 32) % 4, owner = e % 32;
    const int route = rows[f * 8 + 2 * (owner & 3) + (i & 1)];
    if (route < 0) continue;
    const uint64_t at = uint64_t(route) * C::n + n0 + (owner >> 2) + (i >= 2 ? 8 : 0);
    if constexpr (C::gate_up) {
      const float value =
          clamp::clamped_swiglu<typename C::clamp>(sum[0][j], sum[Halves - 1][j]);
      static_cast<uint16_t*>(output)[at] = convert::bf16_from_f32(value);
    } else if constexpr (C::grouped) {
      static_cast<float*>(output)[at] = __fmul_rn(sum[0][j], route_weights[route]);
    } else {
      static_cast<uint16_t*>(output)[at] = convert::bf16_from_f32(sum[0][j]);
    }
  }
}

}  // namespace chop::moe::block_warps_fp4
