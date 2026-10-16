#pragma once

#include "dense.cuh"

namespace chop::gemm {

template<bool Interleaved>
__device__ __forceinline__ uint32_t gate_up_row(
    uint32_t logical, uint32_t half, int projection) {
  if constexpr (Interleaved) return logical * 2 + projection;
  return logical + projection * half;
}

// A work unit owns both gate and up tiles. Its last split reduces each in
// split order, applies optional residuals to the raw columns, then writes
// SiLU(gate) * up. One semaphore belongs to each logical output tile.
template<WeightKind W, bool Decode, bool Residual, bool Interleaved,
         int TileM, int TileN, int TileK, int Stages = 2>
__device__ void run_a16_swiglu(
    const CUtensorMap& map_a, const CUtensorMap& map_b,
    void* output, float* partial, uint32_t* semaphores,
    uint32_t live_m, uint32_t n, uint32_t k, uint32_t splits,
    uint32_t output_f32, const float* scales, const void* residual,
    const float* weight_global_scale) {
  using T = A16Traits<W, Decode, TileM, TileN, TileK, Stages>;
  static_assert(T::shared_bytes + 1152 <= 101376);
  __shared__ TmaPipeline<Stages, T::math_warps> pipe;
  extern __shared__ __align__(128) unsigned char storage[];
  auto* aligned = align_tma_storage(storage);
  auto* a = reinterpret_cast<uint8_t (*)[TileM][T::a_row_bytes]>(aligned);
  auto* b = reinterpret_cast<uint8_t (*)[TileN][T::b_row_bytes]>(
      aligned + Stages * TileM * T::a_row_bytes);
  __shared__ uint32_t last;
  const int warp = int(threadIdx.x / 32) - 4;
  const uint32_t half = n / 2;
  const uint32_t rows_a = Decode ? half : live_m;
  const uint32_t rows_b = Decode ? live_m : half;
  const uint32_t tiles_a = (rows_a + TileM - 1) / TileM;
  const uint32_t tiles_b = (rows_b + TileN - 1) / TileN;
  const uint64_t tile_count = uint64_t(tiles_a) * tiles_b;
  const uint64_t tile_elements = uint64_t(TileM) * TileN;
  const uint32_t initial_tile = blockIdx.x / splits;
  ptx::griddep_launch_dependents();
  if (threadIdx.x == 0 && initial_tile < tile_count) {
    const uint32_t logical = Decode
        ? (initial_tile / tiles_b) * TileM
        : (initial_tile % tiles_b) * TileN;
    const int32_t weight_coord[2] = {
        0, int32_t(gate_up_row<Interleaved>(logical, half, 0))};
    ptx::prefetch_tensor_l2(Decode ? &map_a : &map_b, weight_coord);
  }
  ptx::griddep_wait();
  if (warp < 0) configure_producer_warp();
  else configure_math_warp();

  for (uint64_t work = blockIdx.x; work < tile_count * splits;
       work += gridDim.x) {
    const uint32_t tile = uint32_t(work / splits);
    const uint32_t split = uint32_t(work % splits);
    const uint64_t scratch_unit = splits == 1
        ? blockIdx.x : uint64_t(tile) * splits + split;
    const uint32_t logical_a = (tile / tiles_b) * TileM;
    const uint32_t logical_b = (tile % tiles_b) * TileN;
    const uint32_t chunks = (k + TileK - 1) / TileK;
    const uint32_t first = (uint64_t(chunks) * split) / splits;
    const uint32_t end = (uint64_t(chunks) * (split + 1)) / splits;
    const uint64_t policy = ptx::l2_policy<ptx::L2Evict::first>();

    for (int projection = 0; projection < 2; ++projection) {
      const uint32_t row_a = Decode
          ? gate_up_row<Interleaved>(logical_a, half, projection)
          : logical_a;
      const uint32_t row_b = Decode
          ? logical_b
          : gate_up_row<Interleaved>(logical_b, half, projection);
      if (threadIdx.x == 0) pipe.init();
      __syncthreads();
      if (warp >= 0 && (threadIdx.x & 31) == 0) pipe.prime_empty();
      for (int wave = 0; wave < T::waves; ++wave) {
        ptx::D4 acc0{}, acc1{}, acc2{}, acc3{}, acc4{}, acc5{}, acc6{}, acc7{}, acc8{}, acc9{}, acc10{}, acc11{}, acc12{}, acc13{}, acc14{}, acc15{};
        for (uint32_t chunk = first; chunk < end; ++chunk) {
          const uint32_t step = wave * (end - first) + chunk - first;
          const int stage = step % Stages;
          if (threadIdx.x == 0) {
            pipe.producer_acquire(step);
            uint64_t* bar = pipe.producer_commit(
                step, T::shared_bytes / Stages);
            const int32_t ca[2] = {int32_t(chunk * TileK), int32_t(row_a)};
            const int32_t cb[2] = {int32_t(chunk * TileK), int32_t(row_b)};
            ptx::tma_load_hint<2>(a[stage], &map_a, ca, bar, policy);
            ptx::tma_load_hint<2>(b[stage], &map_b, cb, bar, policy);
          }
          if (warp >= 0) {
            pipe.consumer_wait(step);
#define CHOP_SWIGLU_CONSUME(I, ACC)                                             \
            if constexpr (T::per_warp > I) {                                    \
              const int subtile = wave * T::math_warps * T::per_warp +          \
                                  warp + I * T::math_warps;                     \
              if (subtile < T::subtiles) {                                       \
                const int sm = (subtile / (TileN / 8)) * 16;                     \
                const int sn = (subtile % (TileN / 8)) * 8;                      \
                mma_chunk<W, Decode, TileK, Interleaved ? 2 : 1>(             \
                    ACC, &a[stage][0][0], &b[stage][0][0], sm, sn,             \
                    row_a, row_b, chunk, n, k, scales, weight_global_scale);    \
              }                                                                  \
            }
            CHOP_SWIGLU_CONSUME(0, acc0)
            CHOP_SWIGLU_CONSUME(1, acc1)
            CHOP_SWIGLU_CONSUME(2, acc2)
            CHOP_SWIGLU_CONSUME(3, acc3)
            CHOP_SWIGLU_CONSUME(4, acc4)
            CHOP_SWIGLU_CONSUME(5, acc5)
            CHOP_SWIGLU_CONSUME(6, acc6)
            CHOP_SWIGLU_CONSUME(7, acc7)
            CHOP_SWIGLU_CONSUME(8, acc8)
            CHOP_SWIGLU_CONSUME(9, acc9)
            CHOP_SWIGLU_CONSUME(10, acc10)
            CHOP_SWIGLU_CONSUME(11, acc11)
            CHOP_SWIGLU_CONSUME(12, acc12)
            CHOP_SWIGLU_CONSUME(13, acc13)
            CHOP_SWIGLU_CONSUME(14, acc14)
            CHOP_SWIGLU_CONSUME(15, acc15)
#undef CHOP_SWIGLU_CONSUME
            if ((threadIdx.x & 31) == 0) pipe.consumer_release(step);
          }
        }
        if (warp >= 0) {
          const int group = (threadIdx.x & 31) >> 2;
          const int lane4 = threadIdx.x & 3;
#define CHOP_SWIGLU_WRITE(I, ACC)                                               \
          if constexpr (T::per_warp > I) {                                      \
            const int subtile = wave * T::math_warps * T::per_warp +            \
                                warp + I * T::math_warps;                       \
            if (subtile < T::subtiles) {                                         \
              const uint32_t sm = (subtile / (TileN / 8)) * 16;                 \
              const uint32_t sn = (subtile % (TileN / 8)) * 8;                  \
              _Pragma("unroll")                                                  \
              for (int v = 0; v < 4; ++v) {                                      \
                const uint32_t local_a = sm + group + (v / 2) * 8;             \
                const uint32_t local_b = sn + lane4 * 2 + (v % 2);             \
                const uint32_t token = Decode                                   \
                    ? logical_b + local_b : logical_a + local_a;               \
                const uint32_t column = Decode                                  \
                    ? logical_a + local_a : logical_b + local_b;               \
                if (token < live_m && column < half) {                           \
                  const uint64_t index = uint64_t(local_a) * TileN + local_b;  \
                  partial[(scratch_unit * 2 + projection) *                    \
                      tile_elements + index] = ACC.v[v];                       \
                }                                                               \
              }                                                                 \
            }                                                                   \
          }
          CHOP_SWIGLU_WRITE(0, acc0)
          CHOP_SWIGLU_WRITE(1, acc1)
          CHOP_SWIGLU_WRITE(2, acc2)
          CHOP_SWIGLU_WRITE(3, acc3)
          CHOP_SWIGLU_WRITE(4, acc4)
          CHOP_SWIGLU_WRITE(5, acc5)
          CHOP_SWIGLU_WRITE(6, acc6)
          CHOP_SWIGLU_WRITE(7, acc7)
          CHOP_SWIGLU_WRITE(8, acc8)
          CHOP_SWIGLU_WRITE(9, acc9)
          CHOP_SWIGLU_WRITE(10, acc10)
          CHOP_SWIGLU_WRITE(11, acc11)
          CHOP_SWIGLU_WRITE(12, acc12)
          CHOP_SWIGLU_WRITE(13, acc13)
          CHOP_SWIGLU_WRITE(14, acc14)
          CHOP_SWIGLU_WRITE(15, acc15)
#undef CHOP_SWIGLU_WRITE
        }
      }
      __syncthreads();
      if (threadIdx.x == 0) pipe.invalidate();
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      if (splits == 1) last = 1;
      else {
        __threadfence();
        last = atomicAdd(&semaphores[tile], 1) == splits - 1;
      }
    }
    __syncthreads();
    if (last) {
      for (uint32_t index = threadIdx.x; index < tile_elements;
           index += blockDim.x) {
        const uint32_t logical_a = (tile / tiles_b) * TileM + index / TileN;
        const uint32_t logical_b = (tile % tiles_b) * TileN + index % TileN;
        const uint32_t token = Decode ? logical_b : logical_a;
        const uint32_t column = Decode ? logical_a : logical_b;
        if (token >= live_m || column >= half) continue;
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t s = 0; s < splits; ++s) {
          const uint64_t unit = splits == 1
              ? blockIdx.x : uint64_t(tile) * splits + s;
          const uint64_t base = unit * 2 * tile_elements + index;
          gate += partial[base];
          up += partial[base + tile_elements];
        }
        if constexpr (W == WeightKind::nvfp4) {
          gate = __fmul_rn(gate, *weight_global_scale);
          up = __fmul_rn(up, *weight_global_scale);
        }
        if constexpr (Residual) {
          gate += residual_value(residual,
              uint64_t(token) * n + gate_up_row<Interleaved>(column, half, 0),
              output_f32);
          up += residual_value(residual,
              uint64_t(token) * n + gate_up_row<Interleaved>(column, half, 1),
              output_f32);
        }
        const float value = (gate / (1.0f + expf(-gate))) * up;
        const uint64_t out = uint64_t(token) * half + column;
        if (output_f32) static_cast<float*>(output)[out] = value;
        else static_cast<__nv_bfloat16*>(output)[out] = __float2bfloat16(value);
      }
      __syncthreads();
      if (threadIdx.x == 0 && splits > 1) {
        __threadfence();
        atomicExch(&semaphores[tile], 0);
      }
    }
    __syncthreads();
  }
}

} // namespace chop::gemm
