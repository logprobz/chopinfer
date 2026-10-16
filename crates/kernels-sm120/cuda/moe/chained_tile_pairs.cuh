#pragma once

#include "chained_experts.cuh"

// The grouped chained expert projection of `chained_experts.cuh` at its
// 64-row route padding with two 64 x 64 output tiles per CTA of eight warps
// (`chained::Schedule::TilePairs`), so the two tiles share one operand:
// - Row pairs: an expert with at least `MinPairTiles` row tiles (a hot
//   expert; 0 never pairs) pairs its row tiles 2p and 2p + 1 under one
//   64-column weight tile, decoded once per K step for 128 rows. The CTA at
//   row tile 2p takes the pair's even column tile of its column pair, the
//   CTA at 2p + 1 the odd one.
// - Column pairs: every other row tile (a cold expert's, or a hot expert's
//   last tile of an odd count) takes the two column tiles of its column
//   pair under one activation tile.
// So each (row tile, column tile) of the column schedule has exactly one
// CTA, and the grid is the column schedule's with half the column tiles.
//
// Warps 0-3 compute the first tile and warps 4-7 the second, each warp 16
// output columns as `chained::run`'s warp does: the same exact BF16
// operands (an FP8 code times the power of two of its block scale,
// `mul.rn.bf16x2`), the same fragment layout, one MMA chain per 128-K block
// from zero over eight ascending K16 steps, `fmaf(chain, residual, sum)` in
// K order and `chained::run`'s epilogue. So every output element is bitwise
// `chained::run`'s at the same inputs; only the CTA that computes it and the
// load schedule differ. An 8-row fragment whose rows are all padding skips
// its MMAs: no written element reads them.
//
// Loads: each thread holds the next K step's raw weight codes, their block
// scales and its activation chunks in registers while the warps run the
// current step's MMAs, then decodes them into the shared tiles. No float
// atomics, no workspace.
//
// `M` (lab-only) is a deliberate defect kernel-lab instantiates to show its
// checks fail on it; served entries take `none`, the unmodified kernel.
namespace chop::moe::tile_pairs {

enum class Mutation : int {
  none = 0,
  drop_wait = 1,   // no griddepcontrol.wait
  late_wait = 2,   // the wait after the K loop, before the epilogue
  odd_tail = 3,    // a hot expert's odd last tile computes only its even column tile
  long_chain = 4,  // chains folded every 256 K
};

using chained::Weights;
using chained::kStepK;
using chained::kStride;
using chained::kTileN;

constexpr int kGroups = 2;                             // output tiles per CTA
constexpr int kThreads = kGroups * chained::kThreads;  // four warps per tile
constexpr int kRowTile = 64;

template<class C>
struct Layout {
  // Weight rows of one column tile: gate and up rows for gate/up.
  static constexpr int weight_rows = C::tile_rows;
  // Shared rows: a row pair holds one weight tile and 128 activation rows, a
  // column pair two weight tiles and 64 activation rows.
  static constexpr int row_pair_rows = weight_rows + kGroups * kRowTile;
  static constexpr int column_pair_rows = kGroups * weight_rows + kRowTile;
  static constexpr int rows =
      row_pair_rows > column_pair_rows ? row_pair_rows : column_pair_rows;
  // 16-byte chunks of one K step: four per weight row, eight per activation row.
  static constexpr int row_pair_chunks = weight_rows * 4 + kGroups * kRowTile * 8;
  static constexpr int column_pair_chunks = kGroups * weight_rows * 4 + kRowTile * 8;
  static constexpr int chunks =
      row_pair_chunks > column_pair_chunks ? row_pair_chunks : column_pair_chunks;
  static constexpr int per_thread = (chunks + kThreads - 1) / kThreads;
  static_assert(C::weights == Weights::fp8_block128, "tile pairs decode FP8 128x128 weights");
  static_assert(C::grouped && C::row_tile == kRowTile, "tile pairs serve 64-row routed tiles");
  static_assert(C::n % (kGroups * kTileN) == 0, "whole column pairs");
};

// One thread's chunks of a K step, held in registers: raw FP8 codes and the
// block scale of a weight chunk, or eight BF16 values of an activation row
// (zero for a padded slot).
template<class C>
struct Staged {
  uint4 v[Layout<C>::per_thread];
  float scale[Layout<C>::per_thread];
};

// Issues the loads of K step `step`: the first `weight_chunks` chunks are
// weight rows of the column tiles at `n0_first` and `n0_second` (row r of
// tile u at shared row u * weight_rows + r), the next `row_count * 8`
// activation rows.
template<class C>
__device__ __forceinline__ void fetch(
    Staged<C>& staged, const uint16_t* input, const uint8_t* weight, const float* block_scales,
    const int32_t* rows, int expert, int n0_first, int n0_second, int weight_chunks,
    int row_count, int step) {
  using L = Layout<C>;
  const int k0 = step * kStepK;
#pragma unroll
  for (int i = 0; i < L::per_thread; ++i) {
    const int c = threadIdx.x + i * kThreads;
    staged.v[i] = make_uint4(0, 0, 0, 0);
    staged.scale[i] = 0.f;
    if (c < weight_chunks) {
      const int row = c >> 2, part = c & 3;
      const bool second = row >= L::weight_rows;
      const int weight_row = chained::weight_row<C>(
          second ? n0_second : n0_first, row - (second ? L::weight_rows : 0));
      const uint64_t global = uint64_t(expert) * C::weight_rows + weight_row;
      staged.v[i] = *reinterpret_cast<const uint4*>(weight + global * C::k + k0 + part * 16);
      staged.scale[i] = block_scales[chained::block_scale_index<C>(expert, weight_row, k0 / 128)];
    } else if (c - weight_chunks < row_count * 8) {
      const int slot = (c - weight_chunks) >> 3, part = (c - weight_chunks) & 7;
      const int route = rows[slot];
      if (route >= 0)
        staged.v[i] = *reinterpret_cast<const uint4*>(
            chained::input_row<C>(input, route) + k0 + part * 8);
    }
  }
}

// Writes the staged chunks into the shared tiles, decoding the weights with
// `chained::load_weights`' FP8 arithmetic; activation rows start at shared
// row `weight_chunks / 4`.
template<class C>
__device__ __forceinline__ void stash(
    uint16_t (*tiles)[kStride], const Staged<C>& staged, int weight_chunks, int row_count) {
  using L = Layout<C>;
#pragma unroll
  for (int i = 0; i < L::per_thread; ++i) {
    const int c = threadIdx.x + i * kThreads;
    if (c < weight_chunks) {
      const int row = c >> 2, part = c & 3;
      const uint32_t pow2 = chained::split_scale(staged.scale[i]).pow2;
      const uint32_t words[4] = {staged.v[i].x, staged.v[i].y, staged.v[i].z, staged.v[i].w};
      uint32_t out[8];
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        out[2 * w] = chained::mul_bf16x2(convert::unpack_bf16x2_e4m3(uint16_t(words[w])), pow2);
        out[2 * w + 1] =
            chained::mul_bf16x2(convert::unpack_bf16x2_e4m3(uint16_t(words[w] >> 16)), pow2);
      }
      *reinterpret_cast<uint4*>(&tiles[row][part * 16]) =
          make_uint4(out[0], out[1], out[2], out[3]);
      *reinterpret_cast<uint4*>(&tiles[row][part * 16 + 8]) =
          make_uint4(out[4], out[5], out[6], out[7]);
    } else if (c - weight_chunks < row_count * 8) {
      const int slot = (c - weight_chunks) >> 3, part = (c - weight_chunks) & 7;
      *reinterpret_cast<uint4*>(&tiles[weight_chunks / 4 + slot][part * 8]) = staged.v[i];
    }
  }
}

// One projection: `chained::run`'s arguments and outputs (grouped FP8:
// `local_scales`, `globals` and `live_tokens` are unread).
template<class C, int MinPairTiles, Mutation M = Mutation::none>
__device__ void run(
    const uint16_t* input, const uint8_t* weight, const float* block_scales,
    const uint8_t* local_scales, const float* globals, const int32_t* offsets,
    const int32_t* sorted_rows, const int32_t* num_padded_rows, const int32_t* live_tokens,
    const float* route_weights, void* output) {
  using L = Layout<C>;
  constexpr int Halves = C::halves;
  constexpr int Fragments = kRowTile / 8;
  (void)local_scales;
  (void)globals;
  (void)live_tokens;
  __shared__ __align__(16) uint16_t tiles[L::rows][kStride];
  __shared__ int32_t rows[kGroups * kRowTile];
  __shared__ int32_t tile_expert, expert_begin, expert_end;
  __shared__ uint32_t live_fragments[kGroups];
  if (blockDim.x != kThreads) __trap();
  ptx::griddep_launch_dependents();
  if constexpr (M != Mutation::drop_wait && M != Mutation::late_wait) ptx::griddep_wait();
  const int row0 = blockIdx.x * kRowTile;
  const int live_rows = *num_padded_rows;
  if (live_rows > int(gridDim.x) * kRowTile || live_rows % kRowTile != 0) __trap();
  if (row0 >= live_rows) return;
  // Expert ranges are padded to whole row tiles, so one expert owns the tile.
  for (int e = threadIdx.x; e < C::experts; e += kThreads)
    if (offsets[e] <= row0 && row0 < offsets[e + 1]) {
      tile_expert = e;
      expert_begin = offsets[e];
      expert_end = offsets[e + 1];
    }
  if (threadIdx.x < kGroups) live_fragments[threadIdx.x] = 0;
  __syncthreads();
  const int expert = tile_expert;
  const int tiles_of_expert = (expert_end - expert_begin) / kRowTile;
  const int tile = (row0 - expert_begin) / kRowTile;
  const int pair = tile & ~1;
  const bool hot = MinPairTiles > 0 && tiles_of_expert >= MinPairTiles;
  // A row pair needs both tiles; the odd_tail mutant pairs a hot expert's
  // odd last tile with no rows instead of giving it a column pair.
  const bool row_pair =
      hot && (pair + 1 < tiles_of_expert || (M == Mutation::odd_tail && tile == pair));
  const bool lone_tail = row_pair && pair + 1 >= tiles_of_expert;
  // The first padded row of `rows` and the two tiles' first columns.
  int first_row, n0_first, n0_second;
  if (row_pair) {
    first_row = expert_begin + pair * kRowTile;
    n0_first = n0_second = (blockIdx.y * kGroups + (tile & 1)) * kTileN;
  } else {
    first_row = row0;
    n0_first = blockIdx.y * kGroups * kTileN;
    n0_second = n0_first + kTileN;
  }
  const int row_count = row_pair ? kGroups * kRowTile : kRowTile;
  const int weight_chunks = (row_pair ? 1 : kGroups) * L::weight_rows * 4;
  for (int i = threadIdx.x; i < row_count; i += kThreads) {
    const int route = lone_tail && i >= kRowTile ? -1 : sorted_rows[first_row + i];
    rows[i] = route;
    if (route >= 0) atomicOr(&live_fragments[i / kRowTile], 1u << ((i % kRowTile) / 8));
  }
  __syncthreads();

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int group = warp / 4, column_warp = warp % 4;
  const int g = lane >> 2, t = lane & 3;
  // This warp's tile: its weight rows, activation rows, route rows, columns.
  const int weight_base = row_pair ? 0 : group * L::weight_rows;
  const int row_base = weight_chunks / 4 + (row_pair ? group * kRowTile : 0);
  const int32_t* slots = rows + (row_pair ? group * kRowTile : 0);
  const uint32_t live = live_fragments[row_pair ? group : 0];
  const int col0 = group ? n0_second : n0_first;

  ptx::D4 chain[Halves][Fragments], sum[Halves][Fragments];
#pragma unroll
  for (int h = 0; h < Halves; ++h)
#pragma unroll
    for (int f = 0; f < Fragments; ++f)
#pragma unroll
      for (int i = 0; i < 4; ++i) chain[h][f].v[i] = sum[h][f].v[i] = 0.f;

  Staged<C> staged;
  fetch<C>(staged, input, weight, block_scales, rows, expert, n0_first, n0_second, weight_chunks,
           row_count, 0);
  constexpr int fold_mask = M == Mutation::long_chain ? 3 : 1;
#pragma unroll 1
  for (int step = 0; step < C::k / kStepK; ++step) {
    __syncthreads();
    stash<C>(tiles, staged, weight_chunks, row_count);
    __syncthreads();
    if (step + 1 < C::k / kStepK)
      fetch<C>(staged, input, weight, block_scales, rows, expert, n0_first, n0_second,
               weight_chunks, row_count, step + 1);
#pragma unroll
    for (int kk = 0; kk < kStepK / 16; ++kk) {
      const int col = kk * 16 + 2 * t;
      ptx::A4 a[Halves];
#pragma unroll
      for (int h = 0; h < Halves; ++h) {
        const int base = weight_base + h * kTileN + column_warp * 16;
        a[h].v[0] = chained::lds32(&tiles[base + g][col]);
        a[h].v[1] = chained::lds32(&tiles[base + g + 8][col]);
        a[h].v[2] = chained::lds32(&tiles[base + g][col + 8]);
        a[h].v[3] = chained::lds32(&tiles[base + g + 8][col + 8]);
      }
#pragma unroll
      for (int f = 0; f < Fragments; ++f) {
        if (!(live >> f & 1)) continue;
        ptx::B2 b;
        b.v[0] = chained::lds32(&tiles[row_base + f * 8 + g][col]);
        b.v[1] = chained::lds32(&tiles[row_base + f * 8 + g][col + 8]);
#pragma unroll
        for (int h = 0; h < Halves; ++h) ptx::mma_bf16(chain[h][f], a[h], b);
      }
    }
    if ((step & fold_mask) == fold_mask) {
      // The 128-K block ends: add its chain into the sums in K order.
      const int block = step >> 1;
#pragma unroll
      for (int h = 0; h < Halves; ++h) {
        const float residual = chained::split_scale(
            block_scales[chained::block_scale_index<C>(
                expert, h * C::n + col0 + column_warp * 16, block)])
            .residual;
#pragma unroll
        for (int f = 0; f < Fragments; ++f)
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            sum[h][f].v[i] = fmaf(chain[h][f].v[i], residual, sum[h][f].v[i]);
            chain[h][f].v[i] = 0.f;
          }
      }
    }
  }
  if constexpr (M == Mutation::late_wait) ptx::griddep_wait();

#pragma unroll
  for (int f = 0; f < Fragments; ++f)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int route = slots[f * 8 + 2 * t + (i & 1)];
      if (route < 0) continue;
      const uint64_t at = uint64_t(route) * C::n + col0 + column_warp * 16 + g + (i >= 2 ? 8 : 0);
      if constexpr (C::gate_up) {
        const float value =
            clamp::clamped_swiglu<typename C::clamp>(sum[0][f].v[i], sum[Halves - 1][f].v[i]);
        static_cast<uint16_t*>(output)[at] = convert::bf16_from_f32(value);
      } else {
        static_cast<float*>(output)[at] = __fmul_rn(sum[0][f].v[i], route_weights[route]);
      }
    }
}

}  // namespace chop::moe::tile_pairs
