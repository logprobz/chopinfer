#pragma once

#include "fold_rows.cuh"
#include "mxfp8_fold.cuh"
#include <type_traits>

namespace chop::gemm::dense_decode {

enum class Finish { ticket, merge };
struct Knobs {
  int tile_n;
  int row_groups;
  int stages;
  // Weights and scales must persist and be ready before any PDL predecessor.
  bool weight_prefetch;
  Finish finish;
};

// These policies describe operands already quantized at the trained point.
struct E4m3CeilUe8m0 {
  static __device__ __forceinline__ bool admitted(uint8_t code, uint8_t scale) {
    return mxfp8_fold::admitted(code, scale);
  }
};
struct Ue8m0PerK32 {};
struct FourK32 {
  static __device__ __forceinline__ void mma(ptx::D4& d, ptx::A4 a, ptx::B2 b,
                                            uint32_t sa, uint32_t sb) {
    ptx::mma_mxf8f6f4<ptx::Element::e4m3, ptx::Element::e4m3>(d, a, b, sa, sb);
  }
};

using mxfp8_fold_common::AscendingF32;

template<Knobs S, class Activation, class Scale, class Chain, class Reduction>
struct Policies {
  static_assert(std::is_same_v<Activation, E4m3CeilUe8m0>);
  static_assert(std::is_same_v<Scale, Ue8m0PerK32>);
  static_assert(std::is_same_v<Chain, FourK32>);
  static_assert(std::is_same_v<Reduction, AscendingF32>);
  static_assert(S.finish == Finish::ticket || S.finish == Finish::merge);
  // Ticket slot reuse, prefetch distance and wait_group require exactly 3 stages.
  static_assert(S.finish == Finish::merge
      ? S.tile_n == 16 && S.row_groups == 1 && S.stages == 2 && !S.weight_prefetch
      : (S.tile_n == 32 || S.tile_n == 64) &&
        (S.row_groups == 1 || S.row_groups == 2) && S.stages == 3 &&
        (!S.weight_prefetch || S.row_groups == 1));
  static constexpr bool valid = true;
};

using mxfp8_fold_common::admitted4;
using mxfp8_fold_common::copy16;
using mxfp8_fold_common::copy_packed_chunk;
using mxfp8_fold_common::packed_fragment;
using mxfp8_fold_common::split_fragment;

// Task and packed helpers are shared with the other fold schedules.
struct Task { uint32_t x, y, z, nx, ny; };
template<Knobs S> struct alignas(16) Arena {
  uint8_t codes[S.stages][(16 * S.row_groups + S.tile_n) * 128];
  uint8_t scales[S.stages][(16 * S.row_groups + S.tile_n) * 4];
  uint32_t failed[16 * S.row_groups];
  bool last;
};
template<int N, int Groups, Knobs S, bool Paired>
__device__ __forceinline__ uint32_t tiles(uint32_t rows) {
  constexpr int Rows = 16 * S.row_groups;
  return ((N + S.tile_n - 1) / S.tile_n) * Groups * (Paired ? 2 : 1) *
      ((max(rows, 1u) + Rows - 1) / Rows);
}
// The workspace opens with one failed-row word per row, in whole 256-byte
// blocks: 64 rows for the decode entries, more for an entry that names a
// wider row bound. The tile tickets follow, then the fold partials.
template<int FailedRows>
inline constexpr uint32_t FailedBytes = (uint32_t(FailedRows) * 4 + 255) / 256 * 256;
template<int N, int Groups, Knobs S, bool Paired, int FailedRows = 64>
__device__ __forceinline__ uint32_t header(uint32_t rows) {
  static_assert(FailedRows >= 64 && FailedRows % 64 == 0);
  return FailedBytes<FailedRows> + ((tiles<N, Groups, S, Paired>(rows) * 4 + 255) / 256) * 256;
}

template<int N, int K, int Groups, int RowStride, int GroupStride,
         Knobs S, bool F32Output, bool Paired, bool PackedWeights,
         class Activation, class Scale, class Chain, class Reduction, int FailedRows = 64>
__device__ void initialize(const uint8_t* codes, const uint8_t* scales, uint8_t* workspace,
                          const uint32_t* live_rows, uint32_t* status,
                          uint64_t status_row_offset, uint32_t rows) {
  static_assert(Policies<S, Activation, Scale, Chain, Reduction>::valid);
  static_assert(S.finish == Finish::ticket);
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  auto* tickets = reinterpret_cast<uint32_t*>(workspace + FailedBytes<FailedRows>);
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < tiles<N, Groups, S, Paired>(rows); i += gridDim.x * blockDim.x)
    tickets[i] = 0;
  if constexpr (S.weight_prefetch || S.row_groups == 2) {
    mxfp8_fold_common::admit_task<K,Groups,RowStride,GroupStride>(
        blockIdx.x,codes,scales,reinterpret_cast<uint32_t*>(workspace),
        live_rows,status,status_row_offset,rows);
  } else {
    const uint32_t row = blockIdx.x;
    if (row >= min(rows, *live_rows)) return;
    bool invalid = false;
    for (int g = 0; g < Groups; ++g) {
      for (int k = threadIdx.x * 4; k < K; k += blockDim.x * 4) {
        const uint64_t at = uint64_t(row) * RowStride + uint64_t(g) * GroupStride + k;
        invalid |= !admitted4(*reinterpret_cast<const uint32_t*>(codes + at), scales[at / 32]);
      }
    }
    const bool bad = __syncthreads_or(invalid);
    if (threadIdx.x == 0) {
      const uint32_t previous = atomicOr(status + status_row_offset + row, bad ? 1u : 0u);
      reinterpret_cast<uint32_t*>(workspace)[row] = bad || previous != 0;
    }
  }
}

template<int N, int K, int Groups, int RowStride, int GroupStride,
         Knobs S, bool F32Output, bool Paired, bool PackedWeights,
         class Activation, class Scale, class Chain, class Reduction, int FailedRows = 64>
__device__ void task(Task task, uint8_t (&codes)[S.stages][(16 * S.row_groups + S.tile_n) * 128],
                   uint8_t (&scales)[S.stages][(16 * S.row_groups + S.tile_n) * 4],
                   uint32_t (&row_failed)[16 * S.row_groups], bool& last, const uint8_t* activations, const uint8_t* activation_scales,
                   const uint8_t* weights, const uint8_t* weight_scales,
                   const uint8_t* weights2, const uint8_t* weight_scales2,
                   void* output, void* output2, uint8_t* workspace,
                   const uint32_t* live_rows, uint32_t rows, uint32_t folds_per_cta) {
  static_assert(Policies<S, Activation, Scale, Chain, Reduction>::valid);
  static_assert(S.finish == Finish::ticket);
  static_assert(S.row_groups == 1 || F32Output);
  constexpr int TileN = S.tile_n;
  constexpr int Rows = 16 * S.row_groups;
  constexpr int Folds = (K + 127) / 128;
  constexpr int Threads = TileN * 4;
  const uint32_t splits = (Folds + folds_per_cta - 1) / folds_per_cta;
  const uint32_t bank = (S.row_groups == 2 ? task.z : blockIdx.z) / (splits * Groups);
  const uint32_t g = ((S.row_groups == 2 ? task.z : blockIdx.z) / splits) % Groups;
  const uint32_t first = ((S.row_groups == 2 ? task.z : blockIdx.z) % splits) * folds_per_cta;
  const uint32_t end = min(first + folds_per_cta, uint32_t(Folds));
  const uint32_t token_base = (S.row_groups == 2 ? task.y : blockIdx.y) * Rows;
  const uint32_t column_base = (S.row_groups == 2 ? task.x : blockIdx.x) * TileN;
  // Keep coordinate and live-count reads in their original schedule order.
  const uint32_t initial_live = S.weight_prefetch ? 0 : min(rows, *live_rows);
  if constexpr (!S.weight_prefetch) {
    if (token_base >= initial_live) return;
  }
  const auto* failed = reinterpret_cast<const uint32_t*>(workspace);
  auto* tickets = reinterpret_cast<uint32_t*>(workspace + FailedBytes<FailedRows>);
  auto* partials = reinterpret_cast<float*>(workspace + header<N, Groups, S, Paired, FailedRows>(rows));
  partials += uint64_t(bank) * Folds * rows * Groups * N;
  if constexpr (Paired) {
    if (bank) { weights = weights2; weight_scales = weight_scales2; output = output2; }
  }
  const auto load_weights = [&](uint32_t stage, uint32_t fold) {
    for (uint32_t chunk = threadIdx.x; chunk < TileN * 8; chunk += Threads) {
      const uint32_t row = column_base + chunk / 8;
      const uint32_t col = fold * 128 + (chunk % 8) * 16;
      const bool valid = fold < end && row < N && col < K;
      const uint64_t at = (uint64_t(g) * N + row) * K + col;
      if constexpr (PackedWeights) {
        copy_packed_chunk<N, K>(codes[stage] + 16 * 128,
            weights, g, column_base, chunk, fold, end);
      } else {
      copy16(codes[stage] + fold_row_chunk(16 + chunk / 8, chunk % 8), weights + (valid ? at : 0), valid);
      }
    }
    if (threadIdx.x < TileN) {
      const uint32_t row = column_base + threadIdx.x;
      const bool valid = fold < end && row < N;
      auto* dst = scales[stage] + (16 + threadIdx.x) * 4;
      const uint64_t at = (uint64_t(g) * N + row) * K + fold * 128;
      if (valid && fold * 128 + 128 <= K && K % 128 == 0) {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"
            :: "r"(uint32_t(__cvta_generic_to_shared(dst))), "l"(weight_scales + at / 32) : "memory");
      } else {
#pragma unroll
        for (int i = 0; i < 4; ++i)
          dst[i] = valid && fold * 128 + i * 32 < K ? weight_scales[at / 32 + i] : 127;
      }
    }
  };
  if constexpr (S.weight_prefetch) {
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    load_weights(i, first + i);
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
  // Warm the remaining immutable row segments in L2 while admission runs,
  // using the same bulk-prefetch instruction as the weight-prefetch engine.
  if constexpr (PackedWeights) {
    // Each lane warms the remaining folds of one complete eight-row strip.
    if (threadIdx.x < TileN / 8 && column_base + threadIdx.x * 8 < N && first + 3 < end) {
      const uint32_t begin = (first + 3) * 128;
      const uint32_t bytes = 8 * (min(end * 128, uint32_t(K)) - begin);
      const uint64_t at = mxfp8_fold_packed::fold_offset(
          (g * N + column_base) / 8 + threadIdx.x, first + 3, K);
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
          :: "l"(weights + at), "r"(bytes) : "memory");
    }
  } else {
  if (threadIdx.x < TileN && column_base + threadIdx.x < N && first + 3 < end) {
    const uint32_t begin = (first + 3) * 128;
    const uint32_t bytes = min(end * 128, uint32_t(K)) - begin;
    const uint64_t at = (uint64_t(g) * N + column_base + threadIdx.x) * K + begin;
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
        :: "l"(weights + at), "r"(bytes) : "memory");
  }
  }
  ptx::griddep_wait();
  if (token_base >= min(rows, *live_rows)) {
    // Do not release CTA storage with asynchronous writes still pending.
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    return;
  }
  }
  const uint32_t live = S.weight_prefetch ? min(rows, *live_rows) : initial_live;
  if (threadIdx.x < Rows)
    row_failed[threadIdx.x] = token_base + threadIdx.x < live ? failed[token_base + threadIdx.x] : 1;
  __syncthreads();
  const auto load_plain = [&](uint32_t stage, uint32_t fold) {
    for (uint32_t chunk = threadIdx.x; chunk < (Rows + TileN) * 8; chunk += Threads) {
      const uint32_t r = chunk / 8;
      const uint32_t col = fold * 128 + (chunk % 8) * 16;
      const bool a = r < Rows;
      const uint32_t row = a ? token_base + r : column_base + r - Rows;
      const bool valid = fold < end && col < K && row < (a ? live : N) && (!a || !row_failed[r]);
      const uint64_t at = a ? uint64_t(row) * RowStride + uint64_t(g) * GroupStride + col
                            : (uint64_t(g) * N + row) * K + col;
      if constexpr (PackedWeights) {
        if (!a) {
          copy_packed_chunk<N, K>(codes[stage] + Rows * 128,
              weights, g, column_base, chunk - Rows * 8, fold, end);
          continue;
        }
      }
      copy16(codes[stage] + fold_row_chunk(r, chunk % 8), (a ? activations : weights) + (valid ? at : 0), valid);
    }
    for (uint32_t r = threadIdx.x; r < Rows + TileN; r += Threads) {
      const bool a = r < Rows;
      const uint32_t row = a ? token_base + r : column_base + r - Rows;
      const bool valid = fold < end && row < (a ? live : N) && (!a || !row_failed[r]);
      const uint64_t at = a ? uint64_t(row) * RowStride + uint64_t(g) * GroupStride + fold * 128
                            : (uint64_t(g) * N + row) * K + fold * 128;
      auto* dst = scales[stage] + r * 4;
      const auto* src = a ? activation_scales : weight_scales;
      // Full aligned scale quads travel asynchronously with the codes.
      // Partial K and padded group pitches retain byte-exact scalar loads.
      if (valid && fold * 128 + 128 <= K &&
          (a ? (RowStride % 128 == 0 && GroupStride % 128 == 0) : K % 128 == 0)) {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"
            :: "r"(uint32_t(__cvta_generic_to_shared(dst))), "l"(src + at / 32) : "memory");
      } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) dst[i] = valid && fold * 128 + i * 32 < K ? src[at / 32 + i] : 127;
      }
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  };
  const auto load_activations = [&](uint32_t stage, uint32_t fold) {
    for (uint32_t chunk = threadIdx.x; chunk < 16 * 8; chunk += Threads) {
      const uint32_t r = chunk / 8;
      const uint32_t row = token_base + r;
      const uint32_t col = fold * 128 + (chunk % 8) * 16;
      const bool valid = fold < end && row < live && col < K && !row_failed[r];
      const uint64_t at = uint64_t(row) * RowStride + uint64_t(g) * GroupStride + col;
      copy16(codes[stage] + fold_row_chunk(r, chunk % 8), activations + (valid ? at : 0), valid);
    }
    if (threadIdx.x < 16) {
      const uint32_t row = token_base + threadIdx.x;
      const bool valid = fold < end && row < live && !row_failed[threadIdx.x];
      auto* dst = scales[stage] + threadIdx.x * 4;
      const uint64_t at = uint64_t(row) * RowStride + uint64_t(g) * GroupStride + fold * 128;
      if (valid && fold * 128 + 128 <= K && RowStride % 128 == 0 && GroupStride % 128 == 0) {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"
            :: "r"(uint32_t(__cvta_generic_to_shared(dst))), "l"(activation_scales + at / 32) : "memory");
      } else {
#pragma unroll
        for (int i = 0; i < 4; ++i)
          dst[i] = valid && fold * 128 + i * 32 < K ? activation_scales[at / 32 + i] : 127;
      }
    }
  };
  const auto load = [&](uint32_t stage, uint32_t fold) {
    if constexpr (S.weight_prefetch) {
      load_weights(stage, fold);
      load_activations(stage, fold);
      asm volatile("cp.async.commit_group;" ::: "memory");
    } else load_plain(stage, fold);
  };
#pragma unroll
  for (int i = 0; i < S.stages; ++i) {
    if constexpr (S.weight_prefetch) {
      load_activations(i, first + i);
      asm volatile("cp.async.commit_group;" ::: "memory");
    } else load(i, first + i);
  }
  const uint32_t warp = threadIdx.x / 32;
  const uint32_t quad = (threadIdx.x & 31) / 4;
  const uint32_t lane4 = threadIdx.x & 3;
  for (uint32_t base = first; base < end; base += 2) {
    asm volatile("cp.async.wait_group 1;" ::: "memory");
    __syncthreads();
    ptx::D4 raw[2][S.row_groups]{};
    uint4 packed_b[2];
#pragma unroll
    for (int step = 0; step < 4; ++step) {
#pragma unroll
      for (int pair = 0; pair < 2; ++pair) {
#pragma unroll
        for (int group = 0; group < S.row_groups; ++group) {
          if constexpr (S.row_groups > 1) {
            if (token_base + group * 16 >= live) continue;
          }
        const uint32_t slot = (base - first + pair) % S.stages;
        ptx::A4 af{};
        ptx::B2 bf{};
#pragma unroll
        for (int i = 0; i < 4; ++i)
          af.v[i] = *reinterpret_cast<const uint32_t*>(codes[slot] +
              fold_row_chunk(group * 16 + quad + (i % 2) * 8, step * 2 + i / 2) + lane4 * 4);
        if constexpr (PackedWeights) {
          if ((step & 1) == 0 && group == 0)
            packed_b[pair] = packed_fragment<K>(codes[slot] + Rows * 128,
                warp, quad, lane4, step, base + pair);
          const uint4 w = packed_b[pair];
          bf.v[0] = (step & 1) ? w.z : w.x;
          bf.v[1] = (step & 1) ? w.w : w.y;
        } else {
#pragma unroll
        for (int i = 0; i < 2; ++i)
          bf.v[i] = *reinterpret_cast<const uint32_t*>(codes[slot] +
              fold_row_chunk(Rows + warp * 8 + quad, step * 2 + i) + lane4 * 4);
        }
        const uint32_t sa = scales[slot][(group * 16 + quad + (lane4 & 1) * 8) * 4 + step];
        const uint32_t sb = scales[slot][(Rows + warp * 8 + quad) * 4 + step];
        Chain::mma(raw[pair][group], af, bf, sa, sb);
      }
      }
    }
#pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
      if (base + pair < end) {
#pragma unroll
        for (int group = 0; group < S.row_groups; ++group) {
#pragma unroll
        for (int v = 0; v < 4; ++v) {
          const uint32_t row = token_base + group * 16 + quad + (v / 2) * 8;
          const uint32_t col = column_base + warp * 8 + lane4 * 2 + v % 2;
          if (row < live && col < N)
            partials[((uint64_t(base + pair) * rows + row) * Groups + g) * N + col] = raw[pair][group].v[v];
        }
        }
      }
    }
    __syncthreads();
    if (base + 2 < end) {
      load((base - first) % S.stages, base + 3);
      load((base - first + 1) % S.stages, base + 4);
    }
  }
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  // Every producer thread publishes its own stores before the CTA ticket.
  // The last integer arrival acquires all stores, but never changes fold order.
  __threadfence();
  __syncthreads();
  const uint32_t tile = ((bank * Groups + g) * (S.row_groups == 2 ? task.ny : gridDim.y) + (S.row_groups == 2 ? task.y : blockIdx.y)) * (S.row_groups == 2 ? task.nx : gridDim.x) + (S.row_groups == 2 ? task.x : blockIdx.x);
  if (threadIdx.x == 0) {
    uint32_t arrived;
    asm volatile("atom.add.acq_rel.gpu.u32 %0, [%1], 1;"
        : "=r"(arrived) : "l"(tickets + tile) : "memory");
    last = arrived == splits - 1;
  }
  __syncthreads();
  if (!last) return;
  for (uint32_t i = threadIdx.x; i < Rows * TileN; i += Threads) {
    const uint32_t row = token_base + i / TileN;
    const uint32_t col = column_base + i % TileN;
    if (row >= live || col >= N) continue;
    float total = 0.0f;
    if (!row_failed[i / TileN]) {
      const uint64_t at = (uint64_t(row) * Groups + g) * N + col;
      for (uint32_t base = 0; base < Folds; base += 16) {
        float chain[16];
#pragma unroll
        for (int j = 0; j < 16; ++j)
          if (base + j < Folds) chain[j] = __ldcg(partials + uint64_t(base + j) * rows * Groups * N + at);
#pragma unroll
        for (int j = 0; j < 16; ++j)
          if (base + j < Folds) total = Reduction::add(total, chain[j]);
      }
    }
    const uint64_t at = (uint64_t(row) * Groups + g) * N + col;
    if constexpr (F32Output) reinterpret_cast<float*>(output)[at] = total;
    else reinterpret_cast<uint16_t*>(output)[at] = convert::bf16_from_f32(total);
  }
  __syncthreads();
  if (threadIdx.x == 0) atomicExch(tickets + tile, 0);
}

template<int N, int K, int Groups, int RowStride, int GroupStride,
         Knobs S, bool F32Output, bool Paired, bool PackedWeights,
         class Activation, class Scale, class Chain, class Reduction, int FailedRows = 64>
__device__ void run(const uint8_t* activations, const uint8_t* activation_scales,
                   const uint8_t* weights, const uint8_t* weight_scales,
                   const uint8_t* weights2, const uint8_t* weight_scales2,
                   void* output, void* output2, uint8_t* workspace,
                   const uint32_t* live_rows, uint32_t rows, uint32_t folds_per_cta) {
  const Task work = S.row_groups == 2
      ? Task{blockIdx.x, blockIdx.y, blockIdx.z, gridDim.x, gridDim.y} : Task{};
  ptx::griddep_launch_dependents();
  if constexpr (!S.weight_prefetch) ptx::griddep_wait();
  if constexpr (S.row_groups == 1) {
    __shared__ __align__(16) uint8_t codes[S.stages][(16 + S.tile_n) * 128];
    __shared__ __align__(4) uint8_t scales[S.stages][(16 + S.tile_n) * 4];
    __shared__ uint32_t row_failed[16];
    __shared__ bool last;
    task<N,K,Groups,RowStride,GroupStride,S,F32Output,Paired,PackedWeights,
         Activation,Scale,Chain,Reduction,FailedRows>(work,codes,scales,row_failed,last,
        activations,activation_scales,weights,weight_scales,weights2,weight_scales2,
        output,output2,workspace,live_rows,rows,folds_per_cta);
  } else {
    __shared__ Arena<S> arena;
    task<N,K,Groups,RowStride,GroupStride,S,F32Output,Paired,PackedWeights,
         Activation,Scale,Chain,Reduction,FailedRows>(work,arena.codes,arena.scales,arena.failed,arena.last,
        activations,activation_scales,weights,weight_scales,weights2,weight_scales2,
        output,output2,workspace,live_rows,rows,folds_per_cta);
  }
}

template<int N, int K, int Groups, int RowStride, int GroupStride,
         Knobs S, bool F32Output, bool Paired, bool PackedWeights,
         class Activation, class Scale, class Chain, class Reduction>
__device__ void admit(const uint8_t* codes, const uint8_t* scales, uint32_t* failed,
                     const uint32_t* live_rows, uint32_t* status,
                     uint64_t status_row_offset, uint32_t rows) {
  static_assert(Policies<S, Activation, Scale, Chain, Reduction>::valid);
  static_assert(S.finish == Finish::merge);
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= min(rows, *live_rows)) return;
  bool invalid = false;
  for (int g = 0; g < Groups; ++g) {
    for (int k = threadIdx.x; k < K; k += 128) {
      const uint64_t at = uint64_t(row) * RowStride + uint64_t(g) * GroupStride + k;
      invalid |= !Activation::admitted(codes[at], scales[at / 32]);
    }
  }
  const bool bad = __syncthreads_or(invalid);
  if (threadIdx.x == 0) {
    const uint32_t previous = atomicOr(status + status_row_offset + row, bad ? 1u : 0u);
    failed[row] = bad || previous != 0;
  }
}

template<int N, int K, int Groups, int RowStride, int GroupStride,
         Knobs S, bool F32Output, bool Paired, bool PackedWeights,
         class Activation, class Scale, class Chain, class Reduction>
__device__ void partials(const uint8_t* activations, const uint8_t* activation_scales,
                        const uint8_t* weights, const uint8_t* weight_scales,
                        const uint8_t* weights2, const uint8_t* weight_scales2,
                        float* partials, const uint32_t* live_rows,
                        const uint32_t* status, uint64_t status_row_offset, uint32_t rows) {
  static_assert(Policies<S, Activation, Scale, Chain, Reduction>::valid);
  static_assert(S.finish == Finish::merge);
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  constexpr int Folds = (K + 127) / 128;
  constexpr int Splits = (Folds + 1) / 2;
  const uint32_t bank = blockIdx.z / (Splits * Groups);
  const uint32_t g = (blockIdx.z / Splits) % Groups;
  const uint32_t first_fold = (blockIdx.z % Splits) * 2;
  const uint32_t token_base = blockIdx.y * 16;
  const uint32_t column_base = blockIdx.x * 16;
  const uint32_t live = min(rows, *live_rows);
  if (token_base >= live) return;
  const uint32_t* failed = reinterpret_cast<const uint32_t*>(partials);
  partials += 64; // The first aligned 256 bytes hold once-per-launch row flags.
  if constexpr (Paired) {
    if (bank) { weights = weights2; weight_scales = weight_scales2; }
  }
  partials += uint64_t(bank) * Folds * rows * Groups * N;
  __shared__ __align__(16) uint8_t codes[2][32 * 128];
  __shared__ uint8_t scales[2][32 * 4];
  // Each group contains one fold's operands. Waiting for the first group
  // leaves the second in flight while the first four MMA steps execute.
#pragma unroll
  for (int stage = 0; stage < 2; ++stage) {
    const uint32_t fold = first_fold + stage;
    if constexpr (PackedWeights) {
      // Fixed trips keep activation and weight copies out of operand branches.
#pragma unroll 1
      for (int i = 0; i < 2; ++i) {
        const uint32_t chunk = threadIdx.x + i * 64;
        const uint32_t r = chunk / 8;
        const uint32_t col = fold * 128 + (chunk % 8) * 16;
        const uint32_t row = token_base + r;
        bool valid = col < K && row < live;
        if (valid) valid = failed[row] == 0;
        const uint64_t at = uint64_t(row) * RowStride + uint64_t(g) * GroupStride + col;
        copy16(codes[stage] + fold_row_chunk(r, chunk % 8),
            activations + (valid ? at : 0), valid);
      }
#pragma unroll 1
      for (int i = 0; i < 2; ++i) {
        const uint32_t chunk = threadIdx.x + i * 64;
        copy_packed_chunk<N, K>(codes[stage] + 16 * 128, weights, g, column_base,
            chunk, fold, Folds);
      }
    } else {
    for (int chunk = threadIdx.x; chunk < 32 * 8; chunk += 64) {
      const uint32_t r = chunk / 8;
      const uint32_t col = fold * 128 + (chunk % 8) * 16;
      const bool a = r < 16;
      const uint32_t row = a ? token_base + r : column_base + r - 16;
      bool valid = col < K && row < (a ? live : N);
      if (a && valid) valid = failed[row] == 0;
      const uint64_t at = a ? uint64_t(row) * RowStride + uint64_t(g) * GroupStride + col
                            : (uint64_t(g) * N + row) * K + col;
      const uint8_t* src = a ? activations : weights;
      copy16(codes[stage] + fold_row_chunk(r, chunk % 8), src + (valid ? at : 0), valid);
    }
    }
    for (int i = threadIdx.x; i < 32 * 4; i += 64) {
      const uint32_t r = i / 4;
      const uint32_t col = fold * 128 + (i % 4) * 32;
      const bool a = r < 16;
      const uint32_t row = a ? token_base + r : column_base + r - 16;
      bool valid = col < K && row < (a ? live : N);
      if (a && valid) valid = failed[row] == 0;
      const uint64_t at = a ? uint64_t(row) * RowStride + uint64_t(g) * GroupStride + col
                            : (uint64_t(g) * N + row) * K + col;
      scales[stage][i] = valid ? (a ? activation_scales : weight_scales)[at / 32] : 127;
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
  const uint32_t warp = threadIdx.x / 32;
  const uint32_t quad = (threadIdx.x & 31) / 4;
  const uint32_t lane4 = threadIdx.x & 3;
#pragma unroll
  for (int stage = 0; stage < 2; ++stage) {
    if (stage == 0) asm volatile("cp.async.wait_group 1;" ::: "memory");
    else asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    ptx::D4 partial{};
    uint4 packed_b;
#pragma unroll
    for (int step = 0; step < 4; ++step) {
      ptx::A4 af{};
      ptx::B2 bf{};
#pragma unroll
      for (int i = 0; i < 4; ++i)
        af.v[i] = *reinterpret_cast<const uint32_t*>(codes[stage] +
            fold_row_chunk(quad + (i % 2) * 8, step * 2 + i / 2) + lane4 * 4);
      if constexpr (PackedWeights) {
        if ((step & 1) == 0)
          packed_b = split_fragment<K>(codes[stage] + 16 * 128,
              warp, quad, lane4, step, first_fold + stage);
        bf.v[0] = (step & 1) ? packed_b.z : packed_b.x;
        bf.v[1] = (step & 1) ? packed_b.w : packed_b.y;
      } else {
#pragma unroll
      for (int i = 0; i < 2; ++i)
        bf.v[i] = *reinterpret_cast<const uint32_t*>(codes[stage] +
            fold_row_chunk(16 + warp * 8 + quad, step * 2 + i) + lane4 * 4);
      }
      const uint32_t sa = scales[stage][(quad + (lane4 & 1) * 8) * 4 + step];
      const uint32_t sb = scales[stage][(16 + warp * 8 + quad) * 4 + step];
      Chain::mma(partial, af, bf, sa, sb);
    }
    if (first_fold + stage < Folds) {
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        const uint32_t row = token_base + quad + (v / 2) * 8;
        const uint32_t col = column_base + warp * 8 + lane4 * 2 + v % 2;
        if (row < live && col < N)
          partials[((uint64_t(first_fold + stage) * rows + row) * Groups + g) * N + col]
              = partial.v[v];
      }
    }
  }
}

template<int N, int K, int Groups, int RowStride, int GroupStride,
         Knobs S, bool F32Output, bool Paired, bool PackedWeights,
         class Activation, class Scale, class Chain, class Reduction>
__device__ void merge(const float* partials, void* output, void* output2, const uint32_t* live_rows,
                     const uint32_t* status, uint64_t status_row_offset, uint32_t rows) {
  static_assert(Policies<S, Activation, Scale, Chain, Reduction>::valid);
  static_assert(S.finish == Finish::merge);
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  if (!min(rows, *live_rows)) return;
  const uint32_t* failed = reinterpret_cast<const uint32_t*>(partials);
  partials += 64;
  const uint64_t flat = uint64_t(blockIdx.x) * 128 + threadIdx.x;
  const uint64_t bank_elements = uint64_t(rows) * Groups * N;
  const uint32_t bank = flat / bank_elements;
  if (bank >= (Paired ? 2 : 1)) return;
  const uint64_t at = flat % bank_elements;
  if constexpr (Paired) { if (bank) output = output2; }
  partials += uint64_t(bank) * ((K + 127) / 128) * bank_elements;
  const uint32_t row = at / (uint64_t(Groups) * N);
  if (row >= min(rows, *live_rows)) return;
  float total = 0.0f;
  if (!failed[row]) {
    for (int fold = 0; fold < (K + 127) / 128; ++fold)
      total = Reduction::add(total, partials[uint64_t(fold) * rows * Groups * N + at]);
  }
  if constexpr (F32Output) reinterpret_cast<float*>(output)[at] = total;
  else reinterpret_cast<uint16_t*>(output)[at] = convert::bf16_from_f32(total);
}

} // namespace chop::gemm::dense_decode
