#pragma once

#include "chained_block_warps.cuh"

// The grouped chained expert projection of `chained_experts.cuh` for 16-row
// decode tiles, with each warp streaming its own K blocks
// (`chained::Schedule::WarpStreams`).
//
// Work: an item is one 16-row tile of one expert and 16 output columns (gate
// and up rows for gate/up), as one `block_warps::run` CTA. Items are numbered
// tile-major over the live tiles (columns fastest, so the CTAs at work read
// neighbouring rows of few experts). With `Persistent`, a fixed grid of
// persistent CTAs takes items b, b + G, b + 2G, ... (G = gridDim.x), so no
// CTA starts on a dead tile and a CTA's next item loads under its current
// one; otherwise a [columns / 16, tile capacity] grid runs one item per
// CTA, and CTAs past the live tiles exit.
//
// Arithmetic, per item, exactly `block_warps::run`'s: warp w computes the
// 128-K blocks w, w + 4, w + 8, ...; each block's MMA chain starts from zero
// and runs its eight K16 steps in ascending K with `chained::run`'s exact
// BF16 operands (an E4M3 code times the power of two of its block scale,
// `mul.rn.bf16x2`) and fragment layout. After every four blocks (a round) the
// warps hand their chains over shared memory and each thread adds its
// elements' four chains into its sums in K order with `fmaf(chain, residual,
// sum)`. The epilogue is `chained::run`'s. So every written element is
// bitwise `chained::run`'s at the same inputs; only the CTA that computes it
// and the load schedule differ.
//
// Loads: each warp streams only its own blocks' weight rows, `StageK`
// contiguous bytes per row and stage, through its own ring of `Stages`
// `cp.async` stages, so a K step needs no CTA barrier; the CTA meets only at
// round ends, around the chain handover. With `Swizzle`, the 16-byte chunk c
// of row r of a stage is stored at chunk c ^ (r % 8) (StageK = 128, no row
// padding); otherwise rows are padded by 16 bytes. Activation fragments are
// read from global memory straight into registers one step ahead (routed rows
// stay in L2). An 8-row fragment whose routes are all padding runs no MMAs:
// none of its rows is written.
//
// A CTA keeps two item slots (route rows and expert) in shared memory; item
// q + 1's slot, item q - 1's before, is written at item q's first round end
// and published by that round's closing barrier, before item q + 1's first
// load. No float atomics, no global workspace. Every CTA reads the live
// counts after `griddepcontrol.wait`.
//
// `M` (lab-only) is a deliberate defect kernel-lab instantiates to show its
// checks fail on it; served entries take `none`, the unmodified kernel.
namespace chop::moe::warp_streams {

enum class Mutation : int {
  none = 0,
  drop_wait = 1,       // no griddepcontrol.wait
  late_wait = 2,       // the wait before the first item's epilogue
  reversed_fold = 3,   // a round's chains added in descending K order
  stale_slot = 4,      // items after the second keep their slot's old rows and expert
  dead_fragment = 5,   // the second 8-row fragment never runs its MMAs
};

constexpr int kWarps = 4;
constexpr int kThreads = 32 * kWarps;
constexpr int kTileN = 16;

template<class C, int StageK, bool Swizzle>
struct Layout {
  static constexpr int tile_rows = C::halves * kTileN;
  static constexpr int row_bytes = Swizzle ? StageK : StageK + 16;
  static constexpr int slice_bytes = tile_rows * row_bytes;  // one warp's stage
  static constexpr int subs = 128 / StageK;
  static constexpr int blocks = C::k / 128;
  static constexpr int rounds = blocks / kWarps;
  static constexpr int total = rounds * subs;             // K steps per warp and item
  static constexpr int elements = kTileN * C::row_tile;   // per half
  static constexpr int per_thread = elements / kThreads;
  static constexpr int chunks = tile_rows * StageK / 16;  // 16-byte copies per slice
  static_assert(C::weights == chained::Weights::fp8_block128 && C::grouped,
                "warp streams serve grouped FP8 128x128 projections");
  static_assert(C::row_tile == 16, "warp streams serve 16-row tiles");
  static_assert(C::n % kTileN == 0 && blocks % kWarps == 0, "whole items and rounds");
  static_assert(StageK == 32 || StageK == 64 || StageK == 128, "whole 16-byte chunks of a block");
  static_assert(!Swizzle || StageK == 128, "the swizzle covers 128-byte rows");
  static_assert(chunks % 32 == 0, "whole copies per lane");

  // Byte offset of byte `col` of row `row` in a slice.
  static __device__ __forceinline__ int at(int row, int col) {
    if constexpr (Swizzle)
      return row * row_bytes + (((col >> 4) ^ (row & 7)) << 4) + (col & 15);
    else
      return row * row_bytes + col;
  }
};

template<class C, int StageK, int Stages, bool Swizzle, bool Persistent, int MaxTiles,
         Mutation M = Mutation::none>
__device__ void run(
    const uint16_t* input, const uint8_t* weight, const float* block_scales,
    const uint8_t* local_scales, const float* globals, const int32_t* offsets,
    const int32_t* sorted_rows, const int32_t* num_padded_rows, const int32_t* live_tokens,
    const float* route_weights, void* output) {
  using L = Layout<C, StageK, Swizzle>;
  constexpr int RT = C::row_tile;
  constexpr int Halves = C::halves;
  constexpr int Fragments = RT / 8;
  constexpr int Columns = C::n / kTileN;
  constexpr int Total = L::total;
  constexpr int KK = StageK / 16;
  static_assert(Stages >= 2 && L::rounds >= 2 && Total - L::subs >= Stages - 1,
                "an item's slot is filled at its predecessor's first round end, before its first load");
  (void)local_scales;
  (void)globals;
  (void)live_tokens;
  __shared__ __align__(16) uint8_t stages[Stages][kWarps][L::slice_bytes];
  __shared__ float chains[kWarps][Halves][L::elements];
  __shared__ float residuals[kWarps][Halves];
  __shared__ int32_t rows[2][RT];
  __shared__ int32_t item_expert[2];
  __shared__ int32_t expert_offsets[C::experts + 1];
  if (blockDim.x != kThreads) __trap();
  ptx::griddep_launch_dependents();
  if constexpr (M != Mutation::drop_wait && M != Mutation::late_wait) ptx::griddep_wait();
  const int live_rows = *num_padded_rows;
  if (live_rows < 0 || live_rows > MaxTiles * RT || live_rows % RT != 0) __trap();
  const int items = live_rows / RT * Columns;
  if (!Persistent && (gridDim.x != Columns || live_rows > int(gridDim.y) * RT)) __trap();
  const int first = Persistent ? int(blockIdx.x) : int(blockIdx.x + blockIdx.y * Columns);
  const int step = Persistent ? int(gridDim.x) : items;
  if (first >= items) return;
  const int count = (items - 1 - first) / step + 1;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g4 = lane >> 2, t = lane & 3;
  for (int e = tid; e <= C::experts; e += kThreads) expert_offsets[e] = offsets[e];
  __syncthreads();
  const auto item_tile = [&](int q) { return (first + q * step) / Columns; };
  const auto item_n0 = [&](int q) { return (first + q * step) % Columns * kTileN; };
  // The expert owning a tile: expert ranges pad to whole tiles.
  const auto find_expert = [&](int slot, int tile) {
    const int row0 = tile * RT;
    for (int e = tid; e < C::experts; e += kThreads)
      if (expert_offsets[e] <= row0 && row0 < expert_offsets[e + 1]) item_expert[slot] = e;
  };
  for (int q = 0; q < 2 && q < count; ++q) {
    find_expert(q, item_tile(q));
    if (tid < RT) rows[q][tid] = sorted_rows[item_tile(q) * RT + tid];
  }
  __syncthreads();

  const int loads = count * Total;
  // This warp's slice of its step g: K [block * 128 + sub * StageK, +
  // StageK) of block round * 4 + warp, every weight row of the item's tile.
  const auto issue = [&](int g) {
    const int q = g / Total, s = g % Total;
    const int block = (s / L::subs) * kWarps + warp, sub = s % L::subs;
    const int expert = item_expert[q & 1], n0 = item_n0(q);
    uint8_t* slice = stages[g % Stages][warp];
    constexpr int per_row = StageK / 16;
#pragma unroll
    for (int i = 0; i < L::chunks / 32; ++i) {
      const int c = lane + 32 * i;
      const int row = c / per_row, within = c % per_row;
      const uint8_t* source =
          weight + (uint64_t(expert) * C::weight_rows + block_warps::weight_row<C>(n0, row)) * C::k +
          block * 128 + sub * StageK + within * 16;
      block_warps::cp_async16(slice + L::at(row, within * 16), source, true);
    }
  };
#pragma unroll
  for (int g = 0; g < Stages - 1; ++g) {
    if (g < loads) issue(g);
    block_warps::cp_async_commit();
  }

  // Activation fragments of step g: B[k][n] is route row n of the tile at K
  // block * 128 + sub * StageK + kk * 16 + {2t, 2t + 1} and {2t + 8, 2t + 9}.
  const auto load_b = [&](int g, uint32_t (&b)[KK][Fragments][2], bool (&live)[Fragments]) {
    const int q = g / Total, s = g % Total;
    const int k0 = ((s / L::subs) * kWarps + warp) * 128 + (s % L::subs) * StageK + 2 * t;
#pragma unroll
    for (int f = 0; f < Fragments; ++f) {
      const int route = rows[q & 1][f * 8 + g4];
      live[f] = __any_sync(0xffffffffu, route >= 0) &&
                !(M == Mutation::dead_fragment && f == 1);
      const uint16_t* row = route >= 0 ? chained::input_row<C>(input, route) + k0 : nullptr;
#pragma unroll
      for (int kk = 0; kk < KK; ++kk) {
        b[kk][f][0] = row ? *reinterpret_cast<const uint32_t*>(row + kk * 16) : 0u;
        b[kk][f][1] = row ? *reinterpret_cast<const uint32_t*>(row + kk * 16 + 8) : 0u;
      }
    }
  };

  // This warp's block scales of a round, loaded a round ahead.
  float next_scale[Halves];
  const auto load_scales = [&](int q, int round) {
    const int expert = item_expert[q & 1], n0 = item_n0(q);
#pragma unroll
    for (int h = 0; h < Halves; ++h)
      next_scale[h] = block_scales[chained::block_scale_index<C>(
          expert, h * C::n + n0, round * kWarps + warp)];
  };
  load_scales(0, 0);

  ptx::D4 chain[Halves][Fragments];
  float sum[Halves][L::per_thread];
  uint32_t pow2[Halves];
  float residual[Halves];
#pragma unroll
  for (int h = 0; h < Halves; ++h) {
#pragma unroll
    for (int j = 0; j < L::per_thread; ++j) sum[h][j] = 0.f;
#pragma unroll
    for (int f = 0; f < Fragments; ++f)
#pragma unroll
      for (int i = 0; i < 4; ++i) chain[h][f].v[i] = 0.f;
  }
  uint32_t b[KK][Fragments][2], b_next[KK][Fragments][2];
  bool live[Fragments], live_next[Fragments];
  load_b(0, b, live);
  // The route rows of the item after next, for its slot fill.
  int pending_route = -1;
  if (count > 2 && tid < RT) pending_route = sorted_rows[item_tile(2) * RT + tid];

#pragma unroll 1
  for (int g = 0; g < loads; ++g) {
    const int q = g / Total, s = g % Total;
    const int round = s / L::subs, sub = s % L::subs;
    block_warps::cp_async_wait<Stages - 2>();
    __syncwarp();
    {
      const int next = g + Stages - 1;
      if (next < loads) issue(next);
      block_warps::cp_async_commit();
    }
    if (g + 1 < loads) load_b(g + 1, b_next, live_next);
    if (sub == 0) {
#pragma unroll
      for (int h = 0; h < Halves; ++h) {
        const chained::SplitScale split = chained::split_scale(next_scale[h]);
        pow2[h] = split.pow2;
        residual[h] = split.residual;
      }
      if (round + 1 < L::rounds) load_scales(q, round + 1);
      else if (q + 1 < count) load_scales(q + 1, 0);
    }
    const uint8_t* slice = stages[g % Stages][warp];
#pragma unroll
    for (int kk = 0; kk < KK; ++kk) {
      const int col = kk * 16 + 2 * t;
#pragma unroll
      for (int h = 0; h < Halves; ++h) {
        const int base = h * kTileN;
        const auto pair = [&](int r, int c) {
          const uint16_t codes = *reinterpret_cast<const uint16_t*>(slice + L::at(r, c));
          return chained::mul_bf16x2(convert::unpack_bf16x2_e4m3(codes), pow2[h]);
        };
        ptx::A4 a;
        a.v[0] = pair(base + g4, col);
        a.v[1] = pair(base + g4 + 8, col);
        a.v[2] = pair(base + g4, col + 8);
        a.v[3] = pair(base + g4 + 8, col + 8);
#pragma unroll
        for (int f = 0; f < Fragments; ++f) {
          if (!live[f]) continue;
          ptx::B2 bf;
          bf.v[0] = b[kk][f][0];
          bf.v[1] = b[kk][f][1];
          ptx::mma_bf16(chain[h][f], a, bf);
        }
      }
    }
#pragma unroll
    for (int kk = 0; kk < KK; ++kk)
#pragma unroll
      for (int f = 0; f < Fragments; ++f) {
        b[kk][f][0] = b_next[kk][f][0];
        b[kk][f][1] = b_next[kk][f][1];
      }
#pragma unroll
    for (int f = 0; f < Fragments; ++f) live[f] = live_next[f];
    if (sub != L::subs - 1) continue;

    // Round end: hand the chains over, then add them into the sums in K
    // order (warp w holds block round * 4 + w).
#pragma unroll
    for (int h = 0; h < Halves; ++h) {
#pragma unroll
      for (int f = 0; f < Fragments; ++f)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          chains[warp][h][(f * 4 + i) * 32 + lane] = chain[h][f].v[i];
          chain[h][f].v[i] = 0.f;
        }
      if (lane == 0) residuals[warp][h] = residual[h];
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < kWarps; ++i)
#pragma unroll
      for (int h = 0; h < Halves; ++h)
#pragma unroll
        for (int j = 0; j < L::per_thread; ++j) {
          const int w = M == Mutation::reversed_fold ? kWarps - 1 - i : i;
          sum[h][j] = fmaf(chains[w][h][tid + j * kThreads], residuals[w][h], sum[h][j]);
        }
    // Item q + 1's slot was item q - 1's, whose epilogue ran before this
    // round's barrier.
    if (round == 0 && q >= 1 && q + 1 < count && M != Mutation::stale_slot) {
      find_expert((q + 1) & 1, item_tile(q + 1));
      if (tid < RT) rows[(q + 1) & 1][tid] = pending_route;
    }
    if (round == L::rounds - 1) {
      if constexpr (M == Mutation::late_wait) {
        if (q == 0) ptx::griddep_wait();
      }
      const int n0 = item_n0(q);
      const int32_t* item_rows = rows[q & 1];
#pragma unroll
      for (int j = 0; j < L::per_thread; ++j) {
        const int e = tid + j * kThreads;
        const int f = e / 128, i = (e / 32) % 4, owner = e % 32;
        const int route = item_rows[f * 8 + 2 * (owner & 3) + (i & 1)];
        if (route >= 0) {
          const uint64_t at = uint64_t(route) * C::n + n0 + (owner >> 2) + (i >= 2 ? 8 : 0);
          if constexpr (C::gate_up) {
            const float value =
                clamp::clamped_swiglu<typename C::clamp>(sum[0][j], sum[Halves - 1][j]);
            static_cast<uint16_t*>(output)[at] = convert::bf16_from_f32(value);
          } else {
            static_cast<float*>(output)[at] = __fmul_rn(sum[0][j], route_weights[route]);
          }
        }
#pragma unroll
        for (int h = 0; h < Halves; ++h) sum[h][j] = 0.f;
      }
    }
    __syncthreads();
    if (round == 0 && q >= 1 && q + 2 < count && tid < RT)
      pending_route = sorted_rows[item_tile(q + 2) * RT + tid];
  }
  block_warps::cp_async_wait<0>();
}

}  // namespace chop::moe::warp_streams
