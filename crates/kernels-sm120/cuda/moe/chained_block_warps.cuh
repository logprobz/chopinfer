#pragma once

#include "chained_experts.cuh"

// The chained expert projection of `chained_experts.cuh` with its K blocks
// spread over the warps of a CTA (`chained::Schedule::BlockWarps`), for
// 16-row tiles. One CTA of four warps owns 16 output columns (one MMA
// strip; gate and up rows for gate/up) of one row tile, so a projection
// launches four times the CTAs of `chained::run`. Warp w computes the
// 128-K blocks w, w + 4, w + 8, ...: each block's MMA chain from zero over
// its eight K16 steps in ascending K, with `chained::run`'s exact BF16
// operands (an FP8 code decoded with the power of two of its block scale,
// `mul.rn.bf16x2`, or a stored BF16 value) and its fragment layout. After
// every four blocks the warps hand their chain temporaries over shared
// memory and each thread adds its elements' four chains into its sums in K
// order with `chained::run`'s operation: `fmaf(chain, residual, sum)` with
// the FP8 block scale's residual, `sum + chain` for BF16. The epilogue is
// `chained::run`'s. So every output element is bitwise `chained::run`'s at
// the same inputs; only the CTA that computes it and the load schedule
// differ.
//
// Loads: weights and activation rows of each warp's next 32 K go through a
// ring of `Stages` shared-memory stages by `cp.async` (rows of a padded
// slot are zero-filled), and each warp loads the next round's block scales
// a round ahead. No float atomics, no global workspace.
//
// `M` (lab-only) is a deliberate defect kernel-lab instantiates to show its
// checks fail on it; served entries take `none`, the unmodified kernel.
namespace chop::moe::block_warps {

enum class Mutation : int {
  none = 0,
  drop_wait = 1,      // no griddepcontrol.wait
  late_wait = 2,      // the wait after the K loop, before the epilogue
  reversed_fold = 3,  // a round's chains added in descending K order
};

using chained::Weights;

constexpr int kWarps = 4;               // warps, each on its own 128-K blocks
constexpr int kThreads = 32 * kWarps;
constexpr int kTileN = 16;              // output columns per CTA
constexpr int kStageK = 32;             // K per warp and stage
constexpr int kSubs = 128 / kStageK;    // stages per 128-K block

template<class C>
struct Layout {
  static constexpr int weight_bytes = C::weights == Weights::bf16 ? 2 : 1;
  static constexpr int tile_rows = C::halves * kTileN;
  // Padded strides keep the fragment loads free of bank conflicts.
  static constexpr int weight_stride = kWarps * kStageK * weight_bytes + 16;
  static constexpr int row_stride = kWarps * kStageK * 2 + 16;
  static constexpr int weight_tile = tile_rows * weight_stride;
  static constexpr int stage_bytes = weight_tile + C::row_tile * row_stride;
  static constexpr int stages = weight_bytes == 2 ? 3 : 4;
  static constexpr int blocks = C::k / 128;
  static constexpr int rounds = blocks / kWarps;
  static constexpr int total = rounds * kSubs;
  // Output elements of one half per CTA, in fragment order
  // ((f * 4 + i) * 32 + lane).
  static constexpr int elements = kTileN * C::row_tile;
  static constexpr int per_thread = elements / kThreads;
  static_assert(C::weights == Weights::fp8_block128 || C::weights == Weights::bf16,
                "block warps decode FP8 128x128 and BF16 weights");
  static_assert(C::row_tile == 16, "block warps serve 16-row tiles");
  static_assert(blocks % kWarps == 0, "whole rounds of 128-K blocks");
  static_assert(C::n % kTileN == 0, "16-column tiles");
};

__device__ __forceinline__ void cp_async16(void* target, const void* source, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
               :: "r"(ptx::shared_address(target)), "l"(source), "r"(valid ? 16 : 0)
               : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template<int Pending>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" :: "n"(Pending) : "memory");
}

// The expert weight row of row `row` of the CTA's tile: gate rows, then up
// rows for gate/up.
template<class C>
__device__ __forceinline__ int weight_row(int n0, int row) {
  if constexpr (C::gate_up)
    return (row >= kTileN ? C::n : 0) + n0 + (row & (kTileN - 1));
  else
    return n0 + row;
}

// Starts the copies of stage `s`: for each warp w, K [block * 128 + sub *
// kStageK, + kStageK) of block `round * kWarps + w`, weight rows then
// activation rows, each warp's slice at column w * kStageK of the stage.
template<class C>
__device__ __forceinline__ void load_stage(
    uint8_t* stage, const uint16_t* input, const uint8_t* weight, const int32_t* rows,
    int expert, int n0, int s) {
  using L = Layout<C>;
  const int round = s / kSubs, sub = s % kSubs;
  constexpr int weight_chunks = kStageK * L::weight_bytes / 16;  // per warp slice
  constexpr int weight_total = L::tile_rows * kWarps * weight_chunks;
#pragma unroll
  for (int c = threadIdx.x; c < weight_total; c += kThreads) {
    const int row = c / (kWarps * weight_chunks), part = c % (kWarps * weight_chunks);
    const int w = part / weight_chunks, within = part % weight_chunks;
    const int block = round * kWarps + w;
    const uint64_t global = uint64_t(expert) * C::weight_rows + weight_row<C>(n0, row);
    const uint8_t* source = weight + (global * C::k + block * 128 + sub * kStageK) *
        L::weight_bytes + within * 16;
    cp_async16(stage + row * L::weight_stride + (w * kStageK) * L::weight_bytes + within * 16,
               source, true);
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

// One weight operand pair: columns `col` and `col + 1` of tile row `row`.
template<class C>
__device__ __forceinline__ uint32_t weight_pair(const uint8_t* stage, int row, int col,
                                                uint32_t pow2) {
  using L = Layout<C>;
  if constexpr (C::weights == Weights::bf16) {
    return *reinterpret_cast<const uint32_t*>(stage + row * L::weight_stride + col * 2);
  } else {
    const uint16_t codes = *reinterpret_cast<const uint16_t*>(
        stage + row * L::weight_stride + col);
    return chained::mul_bf16x2(convert::unpack_bf16x2_e4m3(codes), pow2);
  }
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
  constexpr bool fp8 = C::weights == Weights::fp8_block128;
  (void)local_scales;
  (void)globals;
  __shared__ __align__(16) uint8_t stages[L::stages][L::stage_bytes];
  __shared__ float chains[kWarps][Halves][L::elements];
  __shared__ float residuals[kWarps][Halves];
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

  // This warp's block scales for the next round, loaded a round ahead.
  float next_scale[Halves];
  const auto load_scales = [&](int round) {
    if constexpr (fp8) {
#pragma unroll
      for (int h = 0; h < Halves; ++h)
        next_scale[h] = block_scales[chained::block_scale_index<C>(
            expert, h * C::n + n0, round * kWarps + warp)];
    }
  };
  load_scales(0);

  ptx::D4 chain[Halves][Fragments];
  float sum[Halves][L::per_thread];
  uint32_t pow2[Halves];
  float residual[Halves];
#pragma unroll
  for (int h = 0; h < Halves; ++h) {
    pow2[h] = 0x3f803f80u;  // BF16 1.0 pairs; FP8 sets its own per round
    residual[h] = 1.f;
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
    if constexpr (fp8) {
      if (sub == 0) {
#pragma unroll
        for (int h = 0; h < Halves; ++h) {
          const chained::SplitScale split = chained::split_scale(next_scale[h]);
          pow2[h] = split.pow2;
          residual[h] = split.residual;
        }
        if (round + 1 < L::rounds) load_scales(round + 1);
      }
    }
    const uint8_t* stage = stages[s % L::stages];
#pragma unroll
    for (int kk = 0; kk < kStageK / 16; ++kk) {
      const int col = warp * kStageK + kk * 16 + 2 * t;
      ptx::A4 a[Halves];
#pragma unroll
      for (int h = 0; h < Halves; ++h) {
        const int base = h * kTileN;
        a[h].v[0] = weight_pair<C>(stage, base + g, col, pow2[h]);
        a[h].v[1] = weight_pair<C>(stage, base + g + 8, col, pow2[h]);
        a[h].v[2] = weight_pair<C>(stage, base + g, col + 8, pow2[h]);
        a[h].v[3] = weight_pair<C>(stage, base + g + 8, col + 8, pow2[h]);
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
            const float value = chains[w][h][threadIdx.x + j * kThreads];
            if constexpr (fp8)
              sum[h][j] = fmaf(value, residuals[w][h], sum[h][j]);
            else
              sum[h][j] = __fadd_rn(sum[h][j], value);
          }
    }
  }
  cp_async_wait<0>();
  if constexpr (M == Mutation::late_wait) ptx::griddep_wait();

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

}  // namespace chop::moe::block_warps
