#pragma once

// The Sinkhorn mix-collapse with rows tiled over a CTA: bitwise
// sinkhorn_hc.cuh's one-CTA mix_collapse at every row count. A CTA takes
// Rows consecutive rows. Warp w owns mixes w, w + 8, ... of every row of
// the tile, as mix_collapse's warps own them of one row, so each lane loads
// its four fn values of a chain once for all Rows rows. Per (row, mix) and
// chain the lane products are mix_collapse's (k = 128 c + 4 l .. + 3, fmul
// then three fma from zero) and their sum is the value lane 0 of
// mix_collapse's butterfly stores, the same additions with the same
// operands in the same order: the warp sums its Rows x PerWarp chains at
// once by halving the set at the first levels (lanes whose level bit is set
// keep the upper half) and finishing the remaining levels as warp_sum, and
// every addition puts the lower lanes' partial sum first, as lane 0's
// expression does. Each sum is added into its accumulator in K order. The sum of squares keeps
// mix_collapse's thread mapping (thread t FMAs packs t, t + 256, ... in
// order), the per-warp butterfly and the serial add over warps; rstd,
// pre/post, the Sinkhorn normalizations and the collapse are mix_collapse's
// expressions, the Sinkhorn per row on lanes of one warp that exchange
// cells by shuffles. See README.md ("Tiled mix-collapse").

#include <stdint.h>
#include <math.h>
#include "sinkhorn_hc.cuh"

namespace chop::sinkhorn_hc {

namespace tiled_detail {

__device__ __forceinline__ void cp_async16(void* shared, const void* global) {
  const uint32_t at = static_cast<uint32_t>(__cvta_generic_to_shared(shared));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(at), "l"(global) : "memory");
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}
template<int Pending>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(Pending) : "memory");
}

// warp_sum over Count values at once: the first Levels levels (offsets 16,
// 8, ...) halve the set, a lane with the level's bit set keeping the upper
// half; the remaining levels add every kept value as warp_sum does.
// Afterwards lane l holds, in v[0 .. Count >> Levels), the sums of block
// l >> (5 - Levels) of Count >> Levels consecutive values, and the block's
// first lane (its lane bits below 5 - Levels clear) holds exactly the
// expression lane 0 of warp_sum computes: at every level the lower lanes'
// partial sum is the first operand (a halving lane with the level bit set
// adds its partner's copy first; at the remaining levels the first lane's
// own value is the lower one). The lab-only
// CHOP_MIX_TILED_TEST_LEVELS_REVERSED runs the levels after the halvings in
// reverse order (offset 1 first): the same rows and lanes, another tree.
template<int Count, int Levels>
__device__ __forceinline__ void reduce(float (&v)[Count], int lane) {
  static_assert(Levels >= 0 && Levels <= 5 && Count % (1 << Levels) == 0, "whole halvings");
  #pragma unroll
  for (int level = 0; level < Levels; ++level) {
    const int offset = 16 >> level;
    const bool upper = (lane & offset) != 0;
    const int half = Count >> (level + 1);
    #pragma unroll
    for (int k = 0; k < half; ++k) {
      const float low = v[k];
      const float high = v[k + half];
      const float send = upper ? low : high;
      const float keep = upper ? high : low;
      const float peer = __shfl_xor_sync(0xffffffffu, send, offset);
      v[k] = upper ? __fadd_rn(peer, keep) : __fadd_rn(keep, peer);
    }
  }
  #pragma unroll
  for (int level = Levels; level < 5; ++level) {
#ifdef CHOP_MIX_TILED_TEST_LEVELS_REVERSED
    const int offset = 1 << (level - Levels);
#else
    const int offset = 16 >> level;
#endif
    #pragma unroll
    for (int k = 0; k < (Count >> Levels); ++k) {
      v[k] = __fadd_rn(v[k], __shfl_xor_sync(0xffffffffu, v[k], offset));
    }
  }
}

}  // namespace tiled_detail

template<class M, int Rows>
struct Tiled {
  static constexpr int kRows = Rows;
  // Halving levels of the warp sums: log2(Rows).
  static constexpr int kLevels = Rows == 4 ? 2 : Rows == 8 ? 3 : Rows == 16 ? 4 : -1;
  static_assert(kLevels > 0, "4, 8 or 16 rows per CTA");
  // Lanes of one group after the halvings; group g holds row g.
  static constexpr int kGroupLanes = 32 >> kLevels;
  // One staged chunk of the tile's streams: 16 KiB, double-buffered.
  static constexpr int kChunkChains = 64 / Rows;
  static constexpr int kChunk = kChunkChains * kChain;
  static constexpr int kChunks = M::kChains / kChunkChains;
  // 16-byte packs of a row's chunk; the sum of squares' thread t takes
  // pack t of every 256, so it reads chunk b when b % kPeriod == t / kPacks.
  static constexpr int kPacks = kChunk / 8;
  static constexpr int kPeriod = kThreads / kPacks;
  static_assert(M::kChains % kChunkChains == 0, "whole staged chunks");
  static_assert(kThreads % kPacks == 0, "chunks tile the squares' 256-pack period");
  // The Sinkhorn: one group of kCellLanes lanes per row.
  static constexpr int kCellLanes = M::kCells <= 4 ? 4 : M::kCells <= 8 ? 8 : M::kCells <= 16 ? 16 : 32;
  static_assert(M::kCells <= 32, "every Sinkhorn cell of a row in one warp");
  static constexpr int kRowsPerWarp = 32 / kCellLanes;
  static constexpr int kSinkhornWarps =
      (Rows + kRowsPerWarp - 1) / kRowsPerWarp < kWarps ? (Rows + kRowsPerWarp - 1) / kRowsPerWarp
                                                        : kWarps;
  // The collapse runs on the warps without Sinkhorn rows, or on all of
  // them after the Sinkhorn when every warp has rows.
  static constexpr int kCollapseFirst = kSinkhornWarps < kWarps ? kSinkhornWarps * 32 : 0;
  static_assert(M::kWidth % 8 == 0, "16-byte collapse packs");
};

// MixCollapse with Rows rows per CTA (grid: ceil(row capacity / Rows)), 256
// threads; mix_collapse's ABI and outputs, its probes with Probe.
template<class M, int Rows, bool Probe>
__device__ void mix_collapse_tiled(const uint16_t* x, const void* fn, const float* base,
                                   const float* scale, uint16_t* collapsed, uint16_t* post,
                                   uint16_t* comb, float* projected, float* rstd_out,
                                   float* cells_out, const uint32_t* live_rows) {
  using T = Tiled<M, Rows>;
  constexpr int N = M::kStreams;
  constexpr int H = M::kWidth;
  constexpr int Flat = M::kFlat;
  constexpr int Mixes = M::kMixes;
  constexpr int Cells = M::kCells;
  constexpr int PerWarp = M::kPerWarp;
  constexpr int ElementBytes = M::kF32Fn ? 4 : 2;
  __shared__ __align__(16) uint16_t stage[2][Rows][T::kChunk];
  __shared__ float partials[kWarps][Rows];
  __shared__ float rstds[Rows];
  __shared__ float logits[Rows][Mixes];
  __shared__ float pre[Rows][N];
  const int t = threadIdx.x;
  const int lane = t & 31;
  const int warp = t >> 5;

  ptx::griddep_launch_dependents();
  // fn is a weight: warm L2 with it before waiting on the producer. The
  // launch's CTAs prefetch disjoint 128-byte lines covering fn.
  {
    constexpr uint64_t Lines = (uint64_t(Mixes) * Flat * ElementBytes + 127) / 128;
    for (uint64_t line = uint64_t(blockIdx.x) * kThreads + t; line < Lines;
         line += uint64_t(gridDim.x) * kThreads) {
      asm volatile("prefetch.global.L2 [%0];" ::"l"(static_cast<const char*>(fn) + line * 128));
    }
  }
  const uint32_t first = blockIdx.x * Rows;
  auto stage_chunk = [&](int chunk, int count) {
    for (int i = t; i < count * T::kPacks; i += kThreads) {
      const int r = i / T::kPacks;
      const int pack = i % T::kPacks;
      tiled_detail::cp_async16(
          &stage[chunk & 1][r][pack * 8],
          x + (uint64_t(first + r) * Flat + uint64_t(chunk) * T::kChunk + pack * 8));
    }
    tiled_detail::cp_async_commit();
  };
#ifdef CHOP_MIX_TILED_TEST_LATE_WAIT
  // Lab-only mutant: the live count and the first chunk read before the
  // dependency wait.
  const uint32_t live = *live_rows;
  if (first >= live) return;
  const int count = live - first < uint32_t(Rows) ? int(live - first) : Rows;
  stage_chunk(0, count);
  ptx::griddep_wait();
#else
  ptx::griddep_wait();
  const uint32_t live = *live_rows;
  if (first >= live) return;
  const int count = live - first < uint32_t(Rows) ? int(live - first) : Rows;
  stage_chunk(0, count);
#endif

  // The tile's chains, chunk by chunk. Lane group g of warp w accumulates
  // row g's mixes w + 8 p; every thread FMAs the squares of its packs.
  float squares[Rows];
  #pragma unroll
  for (int r = 0; r < Rows; ++r) squares[r] = 0.f;
  float acc[PerWarp];
  #pragma unroll
  for (int p = 0; p < PerWarp; ++p) acc[p] = 0.f;
  auto load_chain = [&](int c, float (&w)[PerWarp][4]) {
    #pragma unroll
    for (int p = 0; p < PerWarp; ++p) {
      const int m = warp + p * kWarps;
      if (m < Mixes) {
        load_fn<M::kF32Fn>(fn, uint64_t(m) * Flat + c * kChain + lane * 4, w[p]);
      } else {
        w[p][0] = w[p][1] = w[p][2] = w[p][3] = 0.f;
      }
    }
  };
  float next[PerWarp][4];
  load_chain(0, next);
  for (int chunk = 0; chunk < T::kChunks; ++chunk) {
    if (chunk + 1 < T::kChunks) {
      stage_chunk(chunk + 1, count);
      tiled_detail::cp_async_wait<1>();
    } else {
      tiled_detail::cp_async_wait<0>();
    }
    __syncthreads();
    const uint16_t (*xs)[T::kChunk] = stage[chunk & 1];
    if (t / T::kPacks == chunk % T::kPeriod) {
      const int pack = t % T::kPacks;
      #pragma unroll
      for (int r = 0; r < Rows; ++r) {
        const uint4 packed = *reinterpret_cast<const uint4*>(&xs[r][pack * 8]);
        const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float lo = bf_lo(words[j]);
          const float hi = bf_hi(words[j]);
          squares[r] = fmaf(lo, lo, squares[r]);
          squares[r] = fmaf(hi, hi, squares[r]);
        }
      }
    }
    #pragma unroll 1
    for (int cc = 0; cc < T::kChunkChains; ++cc) {
      const int c = chunk * T::kChunkChains + cc;
      float w[PerWarp][4];
      #pragma unroll
      for (int p = 0; p < PerWarp; ++p) {
        #pragma unroll
        for (int e = 0; e < 4; ++e) w[p][e] = next[p][e];
      }
      if (c + 1 < M::kChains) load_chain(c + 1, next);
      float sums[Rows * PerWarp];
      #pragma unroll
      for (int r = 0; r < Rows; ++r) {
        const uint2 xp = *reinterpret_cast<const uint2*>(&xs[r][cc * kChain + lane * 4]);
        const float xv[4] = {bf_lo(xp.x), bf_hi(xp.x), bf_lo(xp.y), bf_hi(xp.y)};
        #pragma unroll
        for (int p = 0; p < PerWarp; ++p) {
          float chain = __fmul_rn(w[p][0], xv[0]);
          chain = fmaf(w[p][1], xv[1], chain);
          chain = fmaf(w[p][2], xv[2], chain);
          chain = fmaf(w[p][3], xv[3], chain);
          sums[r * PerWarp + p] = chain;
        }
      }
      tiled_detail::reduce<Rows * PerWarp, T::kLevels>(sums, lane);
      #pragma unroll
      for (int p = 0; p < PerWarp; ++p) acc[p] = __fadd_rn(acc[p], sums[p]);
    }
    __syncthreads();
  }

  // rstd per row: the squares' butterfly per warp, then the warps in order.
  tiled_detail::reduce<Rows, T::kLevels>(squares, lane);
  const int group = lane / T::kGroupLanes;
  const bool leader = lane % T::kGroupLanes == 0;
  if (leader) partials[warp][group] = squares[0];
  __syncthreads();
  if (t < Rows) {
    float total = 0.f;
    #pragma unroll
    for (int w = 0; w < kWarps; ++w) total = __fadd_rn(total, partials[w][t]);
    rstds[t] =
        __frsqrt_rn(__fadd_rn(__fdiv_rn(total, float(Flat)), __uint_as_float(M::kNormEpsBits)));
  }
  __syncthreads();
  if (leader && group < count) {
    #pragma unroll
    for (int p = 0; p < PerWarp; ++p) {
      const int m = warp + p * kWarps;
      if (m < Mixes) {
        logits[group][m] = __fmul_rn(acc[p], rstds[group]);
        if constexpr (Probe) projected[uint64_t(first + group) * Mixes + m] = acc[p];
      }
    }
  }
  if constexpr (Probe) {
    if (t < count) rstd_out[first + t] = rstds[t];
  }
  __syncthreads();

  // pre and post.
  const float eps = __uint_as_float(M::kEpsBits);
  for (int i = t; i < count * N; i += kThreads) {
    const int r = i / N;
    const int s = i % N;
    pre[r][s] = __fadd_rn(chop::sigmoid::sigmoid(fmaf(logits[r][s], scale[0], base[s])), eps);
    const float placed = chop::sigmoid::sigmoid(fmaf(logits[r][N + s], scale[1], base[N + s]));
    post[uint64_t(first + r) * N + s] = to_bf(__fmul_rn(2.f, placed));
  }
  __syncthreads();

  if (warp < T::kSinkhornWarps) {
    // Sinkhorn: lane group of kCellLanes lanes per row, lane cell (i, j) =
    // (cell / N, cell % N); every sum runs in index order from zero over
    // the cells its line holds, read from their lanes.
    constexpr int L = T::kCellLanes;
    for (int row0 = warp * T::kRowsPerWarp; row0 < Rows; row0 += kWarps * T::kRowsPerWarp) {
      const int r = row0 + lane / L;
      const int cell = lane % L;
      const int row_lane = lane - cell;
      const bool owner = cell < Cells && r < count;
      const int i = cell / N;
      const int j = cell % N;
      auto at = [&](float value, int index) {
        return __shfl_sync(0xffffffffu, value, row_lane + index);
      };
      float value = owner ? fmaf(logits[r][2 * N + cell], scale[2], base[2 * N + cell]) : 0.f;
      float max = at(value, i * N);
      #pragma unroll
      for (int k = 1; k < N; ++k) max = fmaxf(max, at(value, i * N + k));
      if (owner) value = expf(__fadd_rn(value, -max));
      float sum = 0.f;
      #pragma unroll
      for (int k = 0; k < N; ++k) sum = __fadd_rn(sum, at(value, i * N + k));
      if (owner) {
        value = __fadd_rn(__fdiv_rn(value, sum), eps);
        if constexpr (Probe) cells_out[uint64_t(first + r) * Cells + cell] = value;
      }
      auto normalize = [&](bool rows) {
        float line = 0.f;
        #pragma unroll
        for (int k = 0; k < N; ++k) line = __fadd_rn(line, at(value, rows ? i * N + k : k * N + j));
        if (owner) value = __fdiv_rn(value, __fadd_rn(line, eps));
      };
      normalize(false);
      for (int iteration = 1; iteration < M::kIterations; ++iteration) {
        normalize(true);
        normalize(false);
      }
      if (owner) comb[uint64_t(first + r) * Cells + cell] = to_bf(value);
    }
  }

  // The collapse: per column an FMA chain from zero in stream order.
  if (t >= T::kCollapseFirst) {
    constexpr int Packs = H / 8;
    for (int item = t - T::kCollapseFirst; item < count * Packs;
         item += kThreads - T::kCollapseFirst) {
      const int r = item / Packs;
      const int col = (item % Packs) * 8;
      const uint16_t* row = x + uint64_t(first + r) * Flat + col;
      float out[8];
      #pragma unroll
      for (int e = 0; e < 8; ++e) out[e] = 0.f;
      #pragma unroll
      for (int s = 0; s < N; ++s) {
        const uint4 packed = *reinterpret_cast<const uint4*>(row + s * H);
        const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
          out[2 * e] = fmaf(pre[r][s], bf_lo(words[e]), out[2 * e]);
          out[2 * e + 1] = fmaf(pre[r][s], bf_hi(words[e]), out[2 * e + 1]);
        }
      }
      uint4 packed;
      packed.x = pack(out[0], out[1]);
      packed.y = pack(out[2], out[3]);
      packed.z = pack(out[4], out[5]);
      packed.w = pack(out[6], out[7]);
      *reinterpret_cast<uint4*>(collapsed + uint64_t(first + r) * H + col) = packed;
    }
  }
}

} // namespace chop::sinkhorn_hc
