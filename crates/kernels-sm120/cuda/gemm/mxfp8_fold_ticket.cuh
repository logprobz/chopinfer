#pragma once

#include "fold_rows.cuh"
#include "mxfp8_fold.cuh"

namespace chop::gemm::mxfp8_fold_ticket {

__device__ __forceinline__ bool admitted4(uint32_t codes, uint8_t scale) {
  // Every finite E4M3 code is BF16-exact in this UE8M0 interval. Detect
  // magnitude 127 in four bytes without decoding or cross-byte carries.
  if (scale >= 3 && scale <= 246)
    return (((codes & 0x7f7f7f7fu) + 0x01010101u) & 0x80808080u) == 0;
  bool valid = true;
#pragma unroll
  for (int i = 0; i < 4; ++i) valid &= mxfp8_fold::admitted(uint8_t(codes >> (i * 8)), scale);
  return valid;
}

template<int N, int Groups, int TileN, bool Paired>
__device__ __forceinline__ uint32_t tiles(uint32_t rows) {
  return ((N + TileN - 1) / TileN) * Groups * (Paired ? 2 : 1) * ((max(rows, 1u) + 15) / 16);
}

template<int N, int Groups, int TileN, bool Paired>
__device__ __forceinline__ uint32_t header(uint32_t rows) {
  return 256 + ((tiles<N, Groups, TileN, Paired>(rows) * 4 + 255) / 256) * 256;
}

template<int N, int K, int Groups, int RowStride, int GroupStride, int TileN, bool Paired>
__device__ void initialize(const uint8_t* codes, const uint8_t* scales, uint8_t* workspace,
                          const uint32_t* live_rows, uint32_t* status,
                          uint64_t status_row_offset, uint32_t rows) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  auto* tickets = reinterpret_cast<uint32_t*>(workspace + 256);
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < tiles<N, Groups, TileN, Paired>(rows); i += gridDim.x * blockDim.x)
    tickets[i] = 0;
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

__device__ __forceinline__ void copy16(uint8_t* dst, const uint8_t* src, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
      :: "r"(uint32_t(__cvta_generic_to_shared(dst))), "l"(src), "r"(valid ? 16 : 0) : "memory");
}

// A packed strip keeps its contiguous fold in one 1024-byte shared slot.
// A final half fold occupies its first 512 bytes; the rest is zero-filled.
template<int N, int K>
__device__ __forceinline__ void copy_packed_chunk(uint8_t* dst, const uint8_t* weights,
    uint32_t group, uint32_t column_base, uint32_t chunk, uint32_t fold, uint32_t end) {
  const uint32_t strip = chunk / 64;
  const uint32_t byte = (chunk % 64) * 16;
  const bool valid = fold < end && column_base + strip * 8 < N &&
      byte < mxfp8_fold_packed::fold_bytes(fold, K);
  const uint64_t at = mxfp8_fold_packed::fold_offset(
      (group * N + column_base) / 8 + strip, fold, K) + byte;
  copy16(dst + fold_row_chunk(chunk / 8, chunk % 8), weights + (valid ? at : 0), valid);
}

// One aligned vector supplies two unchanged MMA steps. Keep the vector
// shared load explicit so the packed lane runs use the swizzle together.
template<int K>
__device__ __forceinline__ uint4 packed_fragment(const uint8_t* codes,
    uint32_t warp, uint32_t quad, uint32_t lane, uint32_t step, uint32_t fold) {
  const bool half = K % 128 != 0 && fold == K / 128;
  if (half && step >= 2) return uint4{};
  const uint32_t at = warp * 1024 +
      mxfp8_fold_packed::segment_word(quad, step * 32 + lane * 4, half);
  const uint8_t* address = codes + fold_row_chunk(at / 128, (at % 128) / 16);
  uint4 value;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
      : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
      : "r"(uint32_t(__cvta_generic_to_shared(address))));
  return value;
}

template<int N, int K, int Groups, int RowStride, int GroupStride,
         int TileN, bool F32Output, bool Paired, bool PackedWeights = false>
__device__ void run(const uint8_t* activations, const uint8_t* activation_scales,
                   const uint8_t* weights, const uint8_t* weight_scales,
                   const uint8_t* weights2, const uint8_t* weight_scales2,
                   void* output, void* output2, uint8_t* workspace,
                   const uint32_t* live_rows, uint32_t rows, uint32_t folds_per_cta) {
  static_assert(TileN == 32 || TileN == 64);
  constexpr int Folds = (K + 127) / 128;
  constexpr int Threads = TileN * 4;
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t splits = (Folds + folds_per_cta - 1) / folds_per_cta;
  const uint32_t bank = blockIdx.z / (splits * Groups);
  const uint32_t g = (blockIdx.z / splits) % Groups;
  const uint32_t first = (blockIdx.z % splits) * folds_per_cta;
  const uint32_t end = min(first + folds_per_cta, uint32_t(Folds));
  const uint32_t token_base = blockIdx.y * 16;
  const uint32_t column_base = blockIdx.x * TileN;
  const uint32_t live = min(rows, *live_rows);
  if (token_base >= live) return;
  const auto* failed = reinterpret_cast<const uint32_t*>(workspace);
  auto* tickets = reinterpret_cast<uint32_t*>(workspace + 256);
  auto* partials = reinterpret_cast<float*>(workspace + header<N, Groups, TileN, Paired>(rows));
  partials += uint64_t(bank) * Folds * rows * Groups * N;
  if constexpr (Paired) {
    if (bank) { weights = weights2; weight_scales = weight_scales2; output = output2; }
  }
  __shared__ __align__(16) uint8_t codes[3][(16 + TileN) * 128];
  __shared__ __align__(4) uint8_t scales[3][(16 + TileN) * 4];
  __shared__ uint32_t row_failed[16];
  __shared__ bool last;
  if (threadIdx.x < 16)
    row_failed[threadIdx.x] = token_base + threadIdx.x < live ? failed[token_base + threadIdx.x] : 1;
  __syncthreads();
  const auto load = [&](uint32_t stage, uint32_t fold) {
    for (uint32_t chunk = threadIdx.x; chunk < (16 + TileN) * 8; chunk += Threads) {
      const uint32_t r = chunk / 8;
      const uint32_t col = fold * 128 + (chunk % 8) * 16;
      const bool a = r < 16;
      const uint32_t row = a ? token_base + r : column_base + r - 16;
      const bool valid = fold < end && col < K && row < (a ? live : N) && (!a || !row_failed[r]);
      const uint64_t at = a ? uint64_t(row) * RowStride + uint64_t(g) * GroupStride + col
                            : (uint64_t(g) * N + row) * K + col;
      if constexpr (PackedWeights) {
        if (!a) {
          copy_packed_chunk<N, K>(codes[stage] + 16 * 128, weights, g, column_base,
              chunk - 16 * 8, fold, end);
          continue;
        }
      }
      copy16(codes[stage] + fold_row_chunk(r, chunk % 8), (a ? activations : weights) + (valid ? at : 0), valid);
    }
    for (uint32_t r = threadIdx.x; r < 16 + TileN; r += Threads) {
      const bool a = r < 16;
      const uint32_t row = a ? token_base + r : column_base + r - 16;
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
    ptx::D4 raw[2]{};
    uint4 packed_b[2];
#pragma unroll
    for (int step = 0; step < 4; ++step) {
#pragma unroll
      for (int pair = 0; pair < 2; ++pair) {
        const uint32_t slot = (base - first + pair) % 3;
        ptx::A4 af{};
        ptx::B2 bf{};
#pragma unroll
        for (int i = 0; i < 4; ++i)
          af.v[i] = *reinterpret_cast<const uint32_t*>(codes[slot] +
              fold_row_chunk(quad + (i % 2) * 8, step * 2 + i / 2) + lane4 * 4);
        if constexpr (PackedWeights) {
          if ((step & 1) == 0)
            packed_b[pair] = packed_fragment<K>(codes[slot] + 16 * 128,
                warp, quad, lane4, step, base + pair);
          const uint4 w = packed_b[pair];
          bf.v[0] = (step & 1) ? w.z : w.x;
          bf.v[1] = (step & 1) ? w.w : w.y;
        } else {
#pragma unroll
        for (int i = 0; i < 2; ++i)
          bf.v[i] = *reinterpret_cast<const uint32_t*>(codes[slot] +
              fold_row_chunk(16 + warp * 8 + quad, step * 2 + i) + lane4 * 4);
        }
        const uint32_t sa = scales[slot][(quad + (lane4 & 1) * 8) * 4 + step];
        const uint32_t sb = scales[slot][(16 + warp * 8 + quad) * 4 + step];
        ptx::mma_mxf8f6f4<ptx::Element::e4m3, ptx::Element::e4m3>(raw[pair], af, bf, sa, sb);
      }
    }
#pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
      if (base + pair < end) {
#pragma unroll
        for (int v = 0; v < 4; ++v) {
          const uint32_t row = token_base + quad + (v / 2) * 8;
          const uint32_t col = column_base + warp * 8 + lane4 * 2 + v % 2;
          if (row < live && col < N)
            partials[((uint64_t(base + pair) * rows + row) * Groups + g) * N + col] = raw[pair].v[v];
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
  const uint32_t tile = ((bank * Groups + g) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  if (threadIdx.x == 0) {
    uint32_t arrived;
    asm volatile("atom.add.acq_rel.gpu.u32 %0, [%1], 1;"
        : "=r"(arrived) : "l"(tickets + tile) : "memory");
    last = arrived == splits - 1;
  }
  __syncthreads();
  if (!last) return;
  for (uint32_t i = threadIdx.x; i < 16 * TileN; i += Threads) {
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

} // namespace chop::gemm::mxfp8_fold_ticket
