#pragma once
#include "mxfp8_fold_ticket.cuh"
#include "mxfp8_fold_resident.cuh"

namespace chop::gemm::mxfp8_fold_reuse {
using mxfp8_fold_ticket::copy16;
struct Task { uint32_t x, y, z, nx, ny; };
template<int TileN> struct alignas(16) Arena {
  uint8_t codes[3][(32 + TileN) * 128];
  uint8_t scales[3][(32 + TileN) * 4];
  uint32_t failed[32];
  bool last;
};
template<int N, int Groups, int TileN, bool Paired>
__device__ __forceinline__ uint32_t tiles(uint32_t rows) {
  return ((N + TileN - 1) / TileN) * Groups * (Paired ? 2 : 1) * ((max(rows, 1u) + 31) / 32);
}
template<int N, int Groups, int TileN, bool Paired>
__device__ __forceinline__ uint32_t header(uint32_t rows) {
  return 256 + ((tiles<N, Groups, TileN, Paired>(rows) * 4 + 255) / 256) * 256;
}
template<int N, int K, int Groups, int RowStride, int GroupStride, int TileN, bool Paired>
__device__ void admit_task(uint32_t cta, uint32_t ctas, const uint8_t* codes,
    const uint8_t* scales, uint8_t* workspace, const uint32_t* live_rows,
    uint32_t* status, uint64_t status_row_offset, uint32_t rows) {
  auto* tickets = reinterpret_cast<uint32_t*>(workspace + 256);
  for (uint32_t i = cta * blockDim.x + threadIdx.x;
       i < tiles<N, Groups, TileN, Paired>(rows); i += ctas * blockDim.x) tickets[i] = 0;
  mxfp8_fold_resident::admit_task<K,Groups,RowStride,GroupStride>(cta,codes,scales,
      reinterpret_cast<uint32_t*>(workspace),live_rows,status,status_row_offset,rows);
}

// A weight stage serves two independent sixteen-row MMA groups. Each
// 128-K fold and the final ascending fold merge keep their original order.
template<int N, int K, int Groups, int RowStride, int GroupStride,
         int TileN, bool F32Output, bool Paired>
__device__ void task(Task task, Arena<TileN>& shared, const uint8_t* activations, const uint8_t* activation_scales,
                   const uint8_t* weights, const uint8_t* weight_scales,
                   const uint8_t* weights2, const uint8_t* weight_scales2,
                   void* output, void* output2, uint8_t* workspace,
                   const uint32_t* live_rows, uint32_t rows, uint32_t folds_per_cta) {
  static_assert(TileN == 32 || TileN == 64);
  constexpr int Folds = (K + 127) / 128;
  constexpr int Threads = TileN * 4;
  const uint32_t splits = (Folds + folds_per_cta - 1) / folds_per_cta;
  const uint32_t bank = task.z / (splits * Groups);
  const uint32_t g = (task.z / splits) % Groups;
  const uint32_t first = (task.z % splits) * folds_per_cta;
  const uint32_t end = min(first + folds_per_cta, uint32_t(Folds));
  const uint32_t token_base = task.y * 32;
  const uint32_t column_base = task.x * TileN;
  const uint32_t live = min(rows, *live_rows);
  if (token_base >= live) return;
  const auto* failed = reinterpret_cast<const uint32_t*>(workspace);
  auto* tickets = reinterpret_cast<uint32_t*>(workspace + 256);
  auto* partials = reinterpret_cast<float*>(workspace + header<N, Groups, TileN, Paired>(rows));
  partials += uint64_t(bank) * Folds * rows * Groups * N;
  if constexpr (Paired) {
    if (bank) { weights = weights2; weight_scales = weight_scales2; output = output2; }
  }
  auto& codes = shared.codes;
  auto& scales = shared.scales;
  auto& row_failed = shared.failed;
  auto& last = shared.last;
  if (threadIdx.x < 32)
    row_failed[threadIdx.x] = token_base + threadIdx.x < live ? failed[token_base + threadIdx.x] : 1;
  __syncthreads();
  const auto load = [&](uint32_t stage, uint32_t fold) {
    for (uint32_t chunk = threadIdx.x; chunk < (32 + TileN) * 8; chunk += Threads) {
      const uint32_t r = chunk / 8;
      const uint32_t col = fold * 128 + (chunk % 8) * 16;
      const bool a = r < 32;
      const uint32_t row = a ? token_base + r : column_base + r - 32;
      const bool valid = fold < end && col < K && row < (a ? live : N) && (!a || !row_failed[r]);
      const uint64_t at = a ? uint64_t(row) * RowStride + uint64_t(g) * GroupStride + col
                            : (uint64_t(g) * N + row) * K + col;
      copy16(codes[stage] + chunk * 16, (a ? activations : weights) + (valid ? at : 0), valid);
    }
    for (uint32_t r = threadIdx.x; r < 32 + TileN; r += Threads) {
      const bool a = r < 32;
      const uint32_t row = a ? token_base + r : column_base + r - 32;
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
#pragma unroll
  for (int i = 0; i < 3; ++i) load(i, first + i);
  const uint32_t warp = threadIdx.x / 32;
  const uint32_t quad = (threadIdx.x & 31) / 4;
  const uint32_t lane4 = threadIdx.x & 3;
  for (uint32_t base = first; base < end; base += 2) {
    asm volatile("cp.async.wait_group 1;" ::: "memory");
    __syncthreads();
    ptx::D4 raw[2][2]{};
#pragma unroll
    for (int step = 0; step < 4; ++step) {
#pragma unroll
      for (int pair = 0; pair < 2; ++pair) {
#pragma unroll
        for (int group = 0; group < 2; ++group) {
          if (token_base + group * 16 >= live) continue;
        const uint32_t slot = (base - first + pair) % 3;
        ptx::A4 af{};
        ptx::B2 bf{};
#pragma unroll
        for (int i = 0; i < 4; ++i)
          af.v[i] = *reinterpret_cast<const uint32_t*>(codes[slot] +
              (group * 16 + quad + (i % 2) * 8) * 128 + step * 32 + (i / 2) * 16 + lane4 * 4);
#pragma unroll
        for (int i = 0; i < 2; ++i)
          bf.v[i] = *reinterpret_cast<const uint32_t*>(codes[slot] +
              (32 + warp * 8 + quad) * 128 + step * 32 + i * 16 + lane4 * 4);
        const uint32_t sa = scales[slot][(group * 16 + quad + (lane4 & 1) * 8) * 4 + step];
        const uint32_t sb = scales[slot][(32 + warp * 8 + quad) * 4 + step];
        ptx::mma_mxf8f6f4<ptx::Element::e4m3, ptx::Element::e4m3>(raw[pair][group], af, bf, sa, sb);
      }
      }
    }
#pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
      if (base + pair < end) {
#pragma unroll
        for (int group = 0; group < 2; ++group) {
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
      load((base - first) % 3, base + 3);
      load((base - first + 1) % 3, base + 4);
    }
  }
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  // Every producer thread publishes its own stores before the CTA ticket.
  // The last integer arrival acquires all stores, but never changes fold order.
  __threadfence();
  __syncthreads();
  const uint32_t tile = ((bank * Groups + g) * task.ny + task.y) * task.nx + task.x;
  if (threadIdx.x == 0) {
    uint32_t arrived;
    asm volatile("atom.add.acq_rel.gpu.u32 %0, [%1], 1;"
        : "=r"(arrived) : "l"(tickets + tile) : "memory");
    last = arrived == splits - 1;
  }
  __syncthreads();
  if (!last) return;
  for (uint32_t i = threadIdx.x; i < 32 * TileN; i += Threads) {
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
          if (base + j < Folds) total = __fadd_rn(total, chain[j]);
      }
    }
    const uint64_t at = (uint64_t(row) * Groups + g) * N + col;
    if constexpr (F32Output) reinterpret_cast<float*>(output)[at] = total;
    else reinterpret_cast<uint16_t*>(output)[at] = convert::bf16_from_f32(total);
  }
  __syncthreads();
  if (threadIdx.x == 0) atomicExch(tickets + tile, 0);
}

} // namespace chop::gemm::mxfp8_fold_reuse
