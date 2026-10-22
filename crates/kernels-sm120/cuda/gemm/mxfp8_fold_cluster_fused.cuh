#pragma once
#include "mxfp8_fold_cluster.cuh"
#include "../row/ceil_parallel.cuh"

namespace chop::gemm::mxfp8_fold_cluster_fused {
using mxfp8_fold_cluster::Layout;
using mxfp8_fold_cluster::dynamic_shared_bytes;
using mxfp8_fold_cluster::wait_newer;

// mxfp8_fold_cluster::run without its admission launch and, with Quantize,
// with the next projection's block quantization in the merge. The folds,
// the fold partials and their increasing-K sums are unchanged.
//
// Admission: a live row fails exactly when its status, ORed with the
// Pending flag bytes a fused producer left for the row, is nonzero after the
// producer wait. For operands of an E4M3 ceil block quantizer with an amax
// floor this is the admission kernel's result: the floor and finite BF16
// inputs keep every scale code in [floor, 247], where a group the quantizer
// passes holds only admitted codes, and a flagged row has a nonzero status.
// The admission kernel then never sets a bit and fails exactly the rows
// with a nonzero status. The first CTA ORs each row's pending flags into its
// status, so the status ends as the row quantizer would leave it.
// IgnoreStatus is the lab control that admits every live row.
//
// Quantize: the 32 outputs a warp merges are one 32-column block of one row
// (whole blocks per tile and per rank share). The warp stages them in its
// CTA's freed operand slots and quantizes them with the row quantizer's
// group function (F32 input rounded to BF16, E4M3 codes, ceil scale, amax
// floor FloorBits), and stores the codes, the scale code and the group's
// flags byte as computed. Every live row's every block is stored by one
// warp, so no flag needs clearing, and no status this grid reads changes
// while it runs. A row any block flags is failed by the next fold, which
// ORs the flags into its status: the row quantizer's status bits. Its
// stored codes, unlike the row quantizer's zeros, are then dead.
template<int N, int K, int Groups, int RowStride, int GroupStride, int TileN,
         int Rows, int Splits, int Stages, int Pending, bool Quantize,
         uint32_t FloorBits, bool IgnoreStatus>
__device__ void run(const uint8_t* activations, const uint8_t* activation_scales,
    const uint8_t* weights, const uint8_t* weight_scales, float* output,
    uint8_t* codes_out, uint8_t* scales_out, uint8_t* flags_out,
    const uint8_t* pending, uint8_t* workspace,
    const uint32_t* live_rows, uint32_t* status, uint64_t status_row_offset,
    uint32_t rows) {
  constexpr int Folds = K / 128;
  constexpr int Per = (Folds + Splits - 1) / Splits;
  constexpr int Threads = TileN * 4;
  constexpr int RowGroups = Rows / 16;
  constexpr int Outputs = Rows * TileN / Splits;
  using L = Layout<Rows, TileN, Stages>;
  static_assert(K % 128 == 0 && N % TileN == 0 && (TileN == 16 || TileN == 32 || TileN == 64));
  static_assert(Rows == 16 || Rows == 32);
  static_assert(Splits >= 2 && Splits <= 8 && (Splits - 1) * Per < Folds);
  static_assert((Rows * TileN) % Splits == 0 && Per <= 8);
  static_assert(Stages >= 2 && Stages <= Per);
  static_assert(!Quantize || (TileN % 32 == 0 && Outputs % 32 == 0 && Threads * 4 <= L::scales));
  static_assert(Pending % 16 == 0);
  extern __shared__ __align__(16) uint8_t storage[];
  // A launch with less shared memory than the layout is a planning fault.
  if (dynamic_shared_bytes() < L::bytes) __trap();
  const auto codes = [&](int f) { return storage + (f % Stages) * L::stage; };
  const auto scales = [&](int f) { return storage + L::scales + (f % Stages) * L::scale_stage; };
  uint32_t* failed = reinterpret_cast<uint32_t*>(storage + L::failed);
  const uint32_t split = ptx::cluster_ctarank();
  const uint32_t g = blockIdx.z / Splits;
  const uint32_t column_base = blockIdx.x * TileN;
  const uint32_t first = split * Per;
  // Fold f of this CTA's split, if the split holds it.
  const auto valid_fold = [&](int f) { return f < Per && first + f < Folds; };
  const auto load_weights = [&](int f) {
    const uint32_t fold = first + f;
    for (uint32_t chunk = threadIdx.x; chunk < TileN * 8; chunk += Threads) {
      const uint32_t row = column_base + chunk / 8;
      const bool valid = valid_fold(f);
      const uint64_t at = (uint64_t(g) * N + row) * K + fold * 128 + (chunk % 8) * 16;
      mxfp8_fold_ticket::copy16(codes(f) + fold_row_chunk(Rows + chunk / 8, chunk % 8),
                                weights + (valid ? at : 0), valid);
    }
    if (threadIdx.x < TileN) {
      auto* dst = scales(f) + (Rows + threadIdx.x) * 4;
      if (valid_fold(f)) {
        const uint64_t at = (uint64_t(g) * N + column_base + threadIdx.x) * (K / 32) + fold * 4;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"
            :: "r"(uint32_t(__cvta_generic_to_shared(dst))), "l"(weight_scales + at) : "memory");
      } else *reinterpret_cast<uint32_t*>(dst) = 0x7f7f7f7fu;
    }
  };
  const auto load_activations = [&](int f) {
    const uint32_t fold = first + f;
    for (uint32_t chunk = threadIdx.x; chunk < Rows * 8; chunk += Threads) {
      const uint32_t row = chunk / 8;
      const bool valid = valid_fold(f) && !failed[row];
      const uint64_t at = uint64_t(row) * RowStride + uint64_t(g) * GroupStride
          + fold * 128 + (chunk % 8) * 16;
      mxfp8_fold_ticket::copy16(codes(f) + fold_row_chunk(row, chunk % 8),
                                activations + (valid ? at : 0), valid);
    }
    for (uint32_t row = threadIdx.x; row < Rows; row += Threads) {
      auto* dst = scales(f) + row * 4;
      if (valid_fold(f) && !failed[row]) {
        const uint64_t at = (uint64_t(row) * RowStride + uint64_t(g) * GroupStride) / 32 + fold * 4;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"
            :: "r"(uint32_t(__cvta_generic_to_shared(dst))), "l"(activation_scales + at) : "memory");
      } else *reinterpret_cast<uint32_t*>(dst) = 0x7f7f7f7fu;
    }
  };
  ptx::griddep_launch_dependents();
  // Only immutable weights enter private shared memory before the wait.
#pragma unroll
  for (int f = 0; f < Stages; ++f) {
    load_weights(f);
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
  ptx::griddep_wait();
  const uint32_t live = min(rows, *live_rows);
  // Every CTA of a cluster reads the same count, so a whole cluster leaves
  // here or none of it does.
  if (live == 0) {
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    return;
  }
  for (uint32_t r = threadIdx.x; r < Rows; r += Threads) {
    uint32_t flags = 0;
    if (r < live) {
#pragma unroll
      for (int b = 0; b < Pending; b += 16) {
        const uint4 v = __ldcg(reinterpret_cast<const uint4*>(pending + uint64_t(r) * Pending + b));
        flags |= v.x | v.y | v.z | v.w;
      }
      flags = (flags | flags >> 8 | flags >> 16 | flags >> 24) & 0xffu;
      if (Pending && flags && blockIdx.x == 0 && blockIdx.z == 0)
        atomicOr(status + status_row_offset + r, flags);
    }
    failed[r] = r >= live ? 1u
        : IgnoreStatus ? 0u : flags | __ldcg(status + status_row_offset + r);
  }
  __syncthreads();
#pragma unroll
  for (int f = 0; f < Stages; ++f) {
    load_activations(f);
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
  const uint32_t warp = threadIdx.x / 32, quad = (threadIdx.x & 31) / 4;
  const uint32_t lane4 = threadIdx.x & 3;
  ptx::D4 raw[Per][RowGroups];
#pragma unroll
  for (int f = 0; f < Per; ++f) {
    if (f == 0) wait_newer<Stages - 1>();
    else wait_newer<Stages - 2>();
    __syncthreads();
    if (f > 0) {
      if (f - 1 + Stages < Per) {
        load_weights(f - 1 + Stages);
        load_activations(f - 1 + Stages);
      }
      asm volatile("cp.async.commit_group;" ::: "memory");
    }
#pragma unroll
    for (int group = 0; group < RowGroups; ++group) raw[f][group] = ptx::D4{};
#pragma unroll
    for (int step = 0; step < 4; ++step) {
      ptx::B2 bf{};
#pragma unroll
      for (int v = 0; v < 2; ++v)
        bf.v[v] = *reinterpret_cast<const uint32_t*>(codes(f)
            + fold_row_chunk(Rows + warp * 8 + quad, step * 2 + v) + lane4 * 4);
      const uint32_t sb = scales(f)[(Rows + warp * 8 + quad) * 4 + step];
#pragma unroll
      for (int group = 0; group < RowGroups; ++group) {
        if (group * 16 >= live) continue;
        ptx::A4 af{};
#pragma unroll
        for (int v = 0; v < 4; ++v)
          af.v[v] = *reinterpret_cast<const uint32_t*>(codes(f)
              + fold_row_chunk(group * 16 + quad + (v % 2) * 8, step * 2 + v / 2) + lane4 * 4);
        const uint32_t sa = scales(f)[(group * 16 + quad + (lane4 & 1) * 8) * 4 + step];
        ptx::mma_mxf8f6f4<ptx::Element::e4m3, ptx::Element::e4m3>(raw[f][group], af, bf, sa, sb);
      }
    }
  }
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  // Slab of fold partials [f][row][column] for this CTA.
  constexpr uint32_t SlabFloats = Per * Rows * TileN;
  float* slab = reinterpret_cast<float*>(workspace)
      + (uint64_t(blockIdx.z) * gridDim.x + blockIdx.x) * SlabFloats;
#pragma unroll
  for (int f = 0; f < Per; ++f) {
#pragma unroll
    for (int group = 0; group < RowGroups; ++group) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const uint32_t row = group * 16 + quad + h * 8;
        if (row < live && first + f < Folds)
          *reinterpret_cast<float2*>(slab + (f * Rows + row) * TileN + warp * 8 + lane4 * 2)
              = make_float2(raw[f][group].v[h * 2], raw[f][group].v[h * 2 + 1]);
      }
    }
  }
  ptx::cluster_arrive_release();
  ptx::cluster_wait_acquire();
  // Rank `split` owns outputs [split * Outputs, (split + 1) * Outputs) of the
  // tile. Issue every load before the fixed-order sums.
  constexpr int PerThread = (Outputs + Threads - 1) / Threads;
  float values[PerThread][Folds];
#pragma unroll
  for (int e = 0; e < PerThread; ++e) {
    const uint32_t index = threadIdx.x + e * Threads;
    const uint32_t row = (split * Outputs + index) / TileN;
    const uint32_t column = (split * Outputs + index) % TileN;
    if (index >= Outputs || row >= live || failed[row]) continue;
#pragma unroll
    for (int source = 0; source < Splits; ++source) {
#pragma unroll
      for (int f = 0; f < Per; ++f) {
        if (source * Per + f >= Folds) continue;
        const uint32_t at = (f * Rows + row) * TileN + column;
        const float* base = reinterpret_cast<const float*>(workspace)
            + (uint64_t(blockIdx.z - split + source) * gridDim.x + blockIdx.x) * SlabFloats;
        values[e][source * Per + f] = __ldcg(base + at);
      }
    }
  }
  // Every warp of this CTA passed its last operand read before the barrier.
  float* stage = reinterpret_cast<float*>(storage);
#pragma unroll
  for (int e = 0; e < PerThread; ++e) {
    const uint32_t index = threadIdx.x + e * Threads;
    const uint32_t row = (split * Outputs + index) / TileN;
    const uint32_t column = (split * Outputs + index) % TileN;
    // Warp-uniform with Quantize: a warp's outputs are one block of one row.
    if (index >= Outputs || row >= live) continue;
    float total = 0.0f;
    if (!failed[row]) {
#pragma unroll
      for (int fold = 0; fold < Folds; ++fold) total = __fadd_rn(total, values[e][fold]);
    }
    const uint64_t at = (uint64_t(row) * Groups + g) * N + column_base + column;
    output[at] = total;
    if constexpr (Quantize) {
      const uint32_t lane = threadIdx.x & 31;
      stage[threadIdx.x] = total;
      __syncwarp();
      const auto value = chop::row::block_quantize_parallel::quantize_group<0, true, false, FloorBits>(
          stage + (threadIdx.x - lane), 2 * lane);
      const uint32_t flags = __reduce_or_sync(0xffffffffu, value.flags);
      const uint64_t block = at - lane;
      if (lane == 0) {
        scales_out[block / 32] = value.scale_code;
        flags_out[block / 32] = uint8_t(flags);
      }
      if (lane < 16)
        *reinterpret_cast<uint16_t*>(codes_out + block + 2 * lane) = value.codes;
      __syncwarp();
    }
  }
}

} // namespace chop::gemm::mxfp8_fold_cluster_fused
