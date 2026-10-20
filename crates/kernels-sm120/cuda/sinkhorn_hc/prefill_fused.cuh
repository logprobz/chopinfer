#pragma once
#include "expand_mix.cuh"
#include "mix_tiled.cuh"
#include "../row/row.cuh"
#include "../row/ceil_parallel.cuh"

namespace chop::sinkhorn_hc::prefill_fused {

enum class Mutation { none, reverse_expansion, round_before_weight, clear_status };

template<class M, int Rows, int Chains>
struct Tile {
  static constexpr int kChunkChains = Chains;
  static constexpr int kChunk = Chains * kChain;
  static constexpr int kChunks = M::kChains / Chains;
  static constexpr int kPacks = kChunk / 8;
  static constexpr int kPeriod = kThreads / kPacks;
  static constexpr int kLevels = Rows == 1 ? 0 : Rows == 4 ? 2 : 3;
  static constexpr int kGroupLanes = 32 >> kLevels;
  static constexpr int kCellLanes = M::kCells <= 4 ? 4 : M::kCells <= 8 ? 8 : 16;
  static constexpr int kRowsPerWarp = 32 / kCellLanes;
  static constexpr int kSinkhornWarps = (Rows + kRowsPerWarp - 1) / kRowsPerWarp;
  static_assert(Rows == 1 || Rows == 4 || Rows == 8);
  static_assert(M::kCells <= 16 && M::kWidth % 32 == 0);
  static_assert(M::kChains % Chains == 0 && kThreads % kPacks == 0);
};

template<class M, int Rows, int Chains>
struct Arena {
  using T = Tile<M,Rows,Chains>;
  struct Projection {
    __align__(16) uint16_t stage[2][Rows][T::kChunk];
    float partials[kWarps][Rows], rstds[Rows], logits[Rows][M::kMixes];
    float incoming_post[Rows][M::kStreams];
    float incoming_comb[Rows][M::kCells];
  };
  struct Tail {
    __align__(16) uint16_t values[M::kWidth];
    __align__(16) uint8_t codes[M::kWidth];
    uint8_t scales[M::kWidth / 32];
    float sums[kWarps];
    uint32_t flags[kWarps];
  };
  union Work { Projection projection; Tail tail; } work;
  // These remain live across the arena overlay and any aliased pre output.
  float weights[Rows][M::kStreams];
};

template<class M, Mutation Mut>
__device__ __forceinline__ uint32_t expanded_pair(
    const uint16_t* x, const uint16_t* y, const float* post,
    const float* comb, uint32_t row, int flat) {
  if constexpr (Mut != Mutation::reverse_expansion)
    return expand_mix_pair<M>(x,y,post,comb,row,flat);
  constexpr int N = M::kStreams, H = M::kWidth;
  const int j = flat / H, col = flat % H;
  float lo = 0.f, hi = 0.f;
  for (int i = N - 1; i >= 0; --i) {
    const uint32_t pair = *reinterpret_cast<const uint32_t*>(
        x + uint64_t(row) * N * H + i * H + col);
    const float w = comb[uint64_t(row) * N * N + i * N + j];
    lo = fmaf(w, bf_lo(pair), lo);
    hi = fmaf(w, bf_hi(pair), hi);
  }
  const uint32_t ys = *reinterpret_cast<const uint32_t*>(y + uint64_t(row) * H + col);
  const float p = post[uint64_t(row) * N + j];
  return pack(__fadd_rn(__fmul_rn(p,bf_lo(ys)),lo),
              __fadd_rn(__fmul_rn(p,bf_hi(ys)),hi));
}

// A virtual tile owns every output column of its rows. No global scratch,
// arrival tickets or grid dependency instructions occur in this task.
template<class M, int Rows, int Chains, uint32_t FloorBits, Mutation Mut>
__device__ void task(uint32_t tile, Arena<M,Rows,Chains>& arena,
    const uint16_t* x, const uint16_t* y, const float* post_in, const float* comb_in,
    uint16_t* expanded, const void* fn, const float* base, const float* scale,
    const float* pre_in, uint16_t* collapsed, float* pre, float* post, float* comb,
    const uint16_t* norm_weight, uint16_t* normed, float* widened,
    uint8_t* codes, uint8_t* scales, const uint32_t* live_rows,
    uint32_t* status, uint64_t status_offset, float norm_eps) {
  using T = Tile<M,Rows,Chains>;
  constexpr int N = M::kStreams, H = M::kWidth, Flat = M::kFlat;
  constexpr int Mixes = M::kMixes, Cells = M::kCells, PerWarp = M::kPerWarp;
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const uint32_t first = tile * Rows, live = *live_rows;
  if (first >= live) return;
  const int count = live - first < uint32_t(Rows) ? int(live - first) : Rows;
  auto& partials = arena.work.projection.partials;
  auto& rstds = arena.work.projection.rstds;
  auto& logits = arena.work.projection.logits;
  auto& weights = arena.weights;
  auto& incoming_post = arena.work.projection.incoming_post;
  auto& incoming_comb = arena.work.projection.incoming_comb;
  for (int i = t; i < count * N; i += kThreads) {
    incoming_post[i/N][i%N] = post_in[uint64_t(first) * N + i];
    weights[i/N][i%N] = pre_in[uint64_t(first) * N + i];
  }
  for (int i = t; i < count * Cells; i += kThreads)
    incoming_comb[i/Cells][i%Cells] = comb_in[uint64_t(first) * Cells + i];
  __syncthreads();
  auto stage_chunk = [&](int chunk, int rows) {
    for (int item = t; item < Rows * T::kPacks; item += kThreads) {
      const int r = item / T::kPacks, col = item % T::kPacks * 8;
      uint4 value = make_uint4(0,0,0,0);
      if (r < rows) {
        uint32_t* words = reinterpret_cast<uint32_t*>(&value);
        #pragma unroll
        for (int j = 0; j < 4; ++j)
          words[j] = expanded_pair<M,Mut>(x + uint64_t(first) * Flat,
              y + uint64_t(first) * H, &incoming_post[0][0], &incoming_comb[0][0],
              r, chunk * T::kChunk + col + 2*j);
        *reinterpret_cast<uint4*>(expanded + uint64_t(first+r) * Flat
            + chunk * T::kChunk + col) = value;
      }
      *reinterpret_cast<uint4*>(&arena.work.projection.stage[chunk & 1][r][col]) = value;
    }
  };
  stage_chunk(0, count);

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

    // This barrier completes the current copy and every previous panel read.
    // Only then can next-panel copies overwrite the other buffer.
    __syncthreads();
    if (chunk+1<T::kChunks) stage_chunk(chunk+1,count);
    const uint16_t (*xs)[T::kChunk] = arena.work.projection.stage[chunk & 1];
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

      }
    }
  }
  __syncthreads();

  // pre and post.
  const float eps = __uint_as_float(M::kEpsBits);
  for (int i = t; i < count * N; i += kThreads) {
    const int r = i / N;
    const int s = i % N;
    pre[uint64_t(first + r) * N + s] = __fadd_rn(chop::sigmoid::sigmoid(fmaf(logits[r][s], scale[0], base[s])), eps);
    const float placed = chop::sigmoid::sigmoid(fmaf(logits[r][N + s], scale[1], base[N + s]));
    post[uint64_t(first + r) * N + s] = __fmul_rn(2.f, placed);
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
      if (owner) comb[uint64_t(first + r) * Cells + cell] = value;
    }
  }


  // All projection and Sinkhorn readers retire before Tail reuses their arena.
  __syncthreads();
  auto& tail = arena.work.tail;
  for (int r = 0; r < count; ++r) {
    const uint32_t row = first + r;
    const uint64_t at = uint64_t(row) * H;
    float square = 0.f;
    for (int col = t * 8; col < H; col += kThreads * 8) {
      float values[8] = {};
      #pragma unroll
      for (int stream = 0; stream < N; ++stream) {
        const uint4 input = *reinterpret_cast<const uint4*>(
            expanded + uint64_t(row) * Flat + stream * H + col);
        const uint32_t words[4] = {input.x,input.y,input.z,input.w};
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          values[2*j] = fmaf(weights[r][stream],bf_lo(words[j]),values[2*j]);
          values[2*j+1] = fmaf(weights[r][stream],bf_hi(words[j]),values[2*j+1]);
        }
      }
      uint4 result;
      auto* rounded = reinterpret_cast<uint16_t*>(&result);
      #pragma unroll
      for (int j = 0; j < 8; ++j) {
        rounded[j] = chop::row::to_bf(values[j]);
        const float v = chop::row::bf(rounded[j]);
        square += v * v;
      }
      *reinterpret_cast<uint4*>(collapsed + at + col) = result;
      *reinterpret_cast<uint4*>(tail.values + col) = result;
    }
    const float total = chop::row::sum_cta(square, tail.sums);
    const float inv = rsqrtf(total / H + norm_eps);
    for (int col = t * 8; col < H; col += kThreads * 8) {
      uint4 packed = *reinterpret_cast<const uint4*>(tail.values + col);
      auto* values = reinterpret_cast<uint16_t*>(&packed);
      const uint4 wp = *reinterpret_cast<const uint4*>(norm_weight + col);
      const auto* w = reinterpret_cast<const uint16_t*>(&wp);
      #pragma unroll
      for (int j = 0; j < 8; ++j) {
        float n = chop::row::bf(values[j]) * inv;
        if constexpr (Mut == Mutation::round_before_weight) n = chop::row::bf(chop::row::to_bf(n));
        values[j] = chop::row::to_bf(n * chop::row::bf(w[j]));
        if (widened) widened[at + col + j] = chop::row::bf(values[j]);
      }
      *reinterpret_cast<uint4*>(normed + at + col) = packed;
      *reinterpret_cast<uint4*>(tail.values + col) = packed;
    }
    if constexpr (Mut == Mutation::clear_status)
      if (t == 0) status[status_offset + row] = 0;
    __syncthreads();
    uint32_t flags = 0;
    #pragma unroll 1
    for (int group = warp; group < H/32; group += kWarps) {
      const int col = group*32 + 2*lane;
      const auto value = chop::row::block_quantize_parallel::quantize_group<0,false,false,FloorBits>(
          tail.values,col);
      flags |= value.flags;
      if (lane == 0) tail.scales[group] = value.scale_code;
      if (lane < 16) {
        tail.codes[col] = uint8_t(value.codes);
        tail.codes[col+1] = uint8_t(value.codes >> 8);
      }
    }
    flags = chop::row::block_quantize_parallel::row_status<8>(
        flags,tail.flags,status,status_offset+row);
    for (int col = t*16; col < H; col += kThreads*16)
      *reinterpret_cast<uint4*>(codes+at+col) = flags ? make_uint4(0,0,0,0)
          : *reinterpret_cast<const uint4*>(tail.codes+col);
    for (int group = t; group < H/32; group += kThreads)
      scales[uint64_t(row)*(H/32)+group] = flags ? 127 : tail.scales[group];
    __syncthreads();
  }
}

template<class M, int Rows, int Chains, uint32_t FloorBits, Mutation Mut>
__device__ void run(const uint16_t* x, const uint16_t* y, const float* post_in,
    const float* comb_in, uint16_t* expanded, const void* fn, const float* base,
    const float* scale, const float* pre_in, uint16_t* collapsed, float* pre,
    float* post, float* comb, const uint16_t* norm_weight, uint16_t* normed,
    float* widened, uint8_t* codes, uint8_t* scales, const uint32_t* live_rows,
    uint32_t* status, uint64_t status_offset, uint32_t norm_eps_bits) {
  if (blockDim.x != kThreads) __trap();
  __shared__ Arena<M,Rows,Chains> arena;
  ptx::griddep_wait();
  task<M,Rows,Chains,FloorBits,Mut>(blockIdx.x,arena,x,y,post_in,comb_in,expanded,
      fn,base,scale,pre_in,collapsed,pre,post,comb,norm_weight,normed,widened,
      codes,scales,live_rows,status,status_offset,__uint_as_float(norm_eps_bits));
  ptx::griddep_launch_dependents();
}
} // namespace chop::sinkhorn_hc::prefill_fused
