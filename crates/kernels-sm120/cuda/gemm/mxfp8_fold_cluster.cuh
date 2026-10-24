#pragma once
#include "fold_rows.cuh"
#include "mxfp8_fold_local_rows.cuh"
#include "../common/ptx_control.cuh"

namespace chop::gemm::mxfp8_fold_cluster {

// cp.async.wait_group with a compile-time count.
template<int Pending>
__device__ __forceinline__ void wait_newer() {
  asm volatile("cp.async.wait_group %0;" :: "n"(Pending) : "memory");
}

// Dynamic shared memory of one CTA: Stages fold slots of Rows activation
// rows then TileN weight rows of 128 bytes, their scale quads, and row flags.
template<int Rows, int TileN, int Stages>
struct Layout {
  static constexpr uint32_t stage = (Rows + TileN) * 128;
  static constexpr uint32_t scale_stage = (Rows + TileN) * 4;
  static constexpr uint32_t scales = Stages * stage;
  static constexpr uint32_t failed = scales + Stages * scale_stage;
  static constexpr uint32_t bytes = failed + Rows * 4;
};

__device__ __forceinline__ uint32_t dynamic_shared_bytes() {
  uint32_t bytes;
  asm volatile("mov.u32 %0, %dynamic_smem_size;" : "=r"(bytes));
  return bytes;
}

// The K splits of one Rows x TileN output tile form a thread-block cluster.
// A CTA owns Per consecutive 128-K folds of its split and keeps Stages of
// them in shared memory at once (Stages == Per keeps all of them): the first
// Stages folds' weights are issued before the dependency wait and their
// activations after it, one commit group per fold, and each later fold
// refills the slot its predecessor Stages folds back left. Each fold is still
// four k32 MMAs from zero with the activations as the m16 operand and one
// weight stage shared by both sixteen-row groups. The fold partials go to
// the CTA's workspace slab (or, Distributed, stay in its shared memory);
// after one cluster barrier each rank sums its share of the tile's outputs
// over every fold in increasing K order from +0. No ticket, fence or last-CTA
// merge orders the sum; no arrival order reaches an output.
template<int N, int K, int Groups, int RowStride, int GroupStride, int TileN,
         int Rows, int Splits, int Stages, bool Distributed, bool OmitFirst>
__device__ void run(const uint8_t* activations, const uint8_t* activation_scales,
    const uint8_t* weights, const uint8_t* weight_scales, float* output,
    uint8_t* workspace, const uint32_t* live_rows, uint32_t rows) {
  constexpr int Folds = K / 128;
  constexpr int Per = (Folds + Splits - 1) / Splits;
  constexpr int Threads = TileN * 4;
  constexpr int RowGroups = Rows / 16;
  constexpr int Outputs = Rows * TileN / Splits;
  using L = Layout<Rows, TileN, Stages>;
  static_assert(K % 128 == 0 && N % TileN == 0 && (TileN == 16 || TileN == 32 || TileN == 64));
  static_assert(Rows == 16 || Rows == 32 || Rows == 64);
  static_assert(Splits >= 2 && Splits <= 8 && (Splits - 1) * Per < Folds);
  static_assert((Rows * TileN) % Splits == 0 && Per <= 8);
  static_assert(Stages >= 2 && Stages <= Per);
  static_assert(!Distributed || Per * Rows * TileN * 4 <= L::scales);
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
  for (uint32_t r = threadIdx.x; r < Rows; r += Threads)
    failed[r] = r < live ? reinterpret_cast<const uint32_t*>(workspace)[r] : 1;
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
    // Groups newer than fold f's: the later prologue activation groups and
    // the refills of earlier iterations, Stages - 1 for fold 0 and Stages - 2
    // for every later fold (all weight groups are older than fold 0's).
    if (f == 0) wait_newer<Stages - 1>();
    else wait_newer<Stages - 2>();
    __syncthreads();
    // Every warp has finished fold f - 1, so its slot takes fold f - 1 + Stages.
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
  float* slab;
  if constexpr (Distributed) {
    // Every warp has read its last operands before they are overwritten.
    __syncthreads();
    slab = reinterpret_cast<float*>(storage);
  } else {
    slab = reinterpret_cast<float*>(workspace + 256)
        + (uint64_t(blockIdx.z) * gridDim.x + blockIdx.x) * SlabFloats;
  }
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
        if constexpr (Distributed) {
          const uint32_t remote = ptx::cluster_map_shared(slab + at, source);
          asm volatile("ld.shared::cluster.f32 %0, [%1];"
              : "=f"(values[e][source * Per + f]) : "r"(remote) : "memory");
        } else {
          const float* base = reinterpret_cast<const float*>(workspace + 256)
              + (uint64_t(blockIdx.z - split + source) * gridDim.x + blockIdx.x) * SlabFloats;
          values[e][source * Per + f] = __ldcg(base + at);
        }
      }
    }
  }
#pragma unroll
  for (int e = 0; e < PerThread; ++e) {
    const uint32_t index = threadIdx.x + e * Threads;
    const uint32_t row = (split * Outputs + index) / TileN;
    const uint32_t column = (split * Outputs + index) % TileN;
    if (index >= Outputs || row >= live) continue;
    float total = 0.0f;
    if (!failed[row]) {
#pragma unroll
      for (int fold = 0; fold < Folds; ++fold)
        total = __fadd_rn(total, OmitFirst && fold == 0 ? 0.0f : values[e][fold]);
    }
    output[(uint64_t(row) * Groups + g) * N + column_base + column] = total;
  }
  if constexpr (Distributed) {
    // No CTA leaves while another may still read its shared memory.
    ptx::cluster_arrive_release();
    ptx::cluster_wait_acquire();
  }
}

} // namespace chop::gemm::mxfp8_fold_cluster
