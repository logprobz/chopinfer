#pragma once

// `run_a16_warps` (tiled_warps.cuh, `DenseKernel::TiledChains` above 32
// rows) with its ring stages released only after every read of them, as
// `run_a16_strips_ordered` (strip_chains_ordered.cuh) does for strip
// chains: that kernel's lane 0 releases a stage right after its own last
// reads, with no fence between the warp's generic-proxy shared reads
// (ldmatrix, and the FP8 weight codes) and the producer's next TMA write of
// the stage, which is in the async proxy, and ptxas issues the release
// before those reads return. Here each math lane runs
// `fence.proxy.async.shared::cta` after its last read of the stage, the
// warp synchronizes, and then lane 0 releases. Everything else, and so
// every output bit, is `run_a16_warps`'.

#include "tiled_warps.cuh"
#include "../common/ptx_control.cuh"

namespace chop::gemm {

template<WeightKind W, bool Residual, int TileM, int TileN, int TileK,
         int Stages = 2, ChainsMutation Mutation = ChainsMutation::none>
__device__ void run_a16_warps_ordered(
    const CUtensorMap& map_a, const CUtensorMap& map_b,
    void* output, float* partial, uint32_t* semaphores,
    uint32_t live_m, uint32_t n, uint32_t k, uint32_t splits,
    uint32_t output_f32, const float* scales, const void* residual,
    const float* weight_global_scale) {
  using T = A16Traits<W, false, TileM, TileN, TileK, Stages>;
  using P = WarpTiles<W, TileM, TileN, TileK>;
  static_assert(T::shared_bytes <= 101376);
  static_assert(Mutation == ChainsMutation::none || Mutation == ChainsMutation::drop_wait ||
                    Mutation == ChainsMutation::late_wait,
                "warp tiles keep no temporaries or tickets");
  (void)partial;
  (void)semaphores;
  (void)weight_global_scale;
  __shared__ TmaPipeline<Stages, T::math_warps> pipe;
  extern __shared__ __align__(128) unsigned char storage[];
  auto* aligned = align_tma_storage(storage);
  auto* a = reinterpret_cast<uint8_t (*)[TileM][T::a_row_bytes]>(aligned);
  auto* b = reinterpret_cast<uint8_t (*)[TileN][T::b_row_bytes]>(
      aligned + Stages * TileM * T::a_row_bytes);
  if (splits != 1) __trap();
  const int warp = int(threadIdx.x / 32) - 4;
  const uint32_t tiles_a = (live_m + TileM - 1) / TileM;
  const uint32_t tiles_b = (n + TileN - 1) / TileN;
  const uint64_t tile_count = uint64_t(tiles_a) * tiles_b;
  const uint32_t chunks = (k + TileK - 1) / TileK;
  ptx::griddep_launch_dependents();
  // One tile per CTA (the grid covers the capacity's tiles); CTAs past the
  // live tiles exit before touching the pipeline.
  const uint64_t tile = blockIdx.x;
  if (tile >= tile_count) return;
  const uint32_t row_a = uint32_t(tile / tiles_b) * TileM;
  const uint32_t row_b = uint32_t(tile % tiles_b) * TileN;
  if (threadIdx.x == 0) {
    const int32_t weight_coord[2] = {0, int32_t(row_b)};
    ptx::prefetch_tensor_l2(&map_b, weight_coord);
    pipe.init();
  }
  __syncthreads();
  if (warp >= 0 && (threadIdx.x & 31) == 0) pipe.prime_empty();
  if constexpr (Mutation == ChainsMutation::none) ptx::griddep_wait();
  // The producer and the math warps run separate loops over the tile's
  // chunks, so no code runs on both sides of the register split.
  if (warp < 0) {
    configure_producer_warp();
    if (threadIdx.x == 0) {
      const uint64_t policy = ptx::l2_policy<ptx::L2Evict::first>();
      for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
        const int stage = chunk % Stages;
        pipe.producer_acquire(chunk);
        uint64_t* bar = pipe.producer_commit(chunk, T::shared_bytes / Stages);
        const int32_t ca[2] = {int32_t(chunk * TileK), int32_t(row_a)};
        const int32_t cb[2] = {int32_t(chunk * TileK), int32_t(row_b)};
        ptx::tma_load_hint<2>(a[stage], &map_a, ca, bar, policy);
        ptx::tma_load_hint<2>(b[stage], &map_b, cb, bar, policy);
      }
    }
  } else {
    configure_math_warp();
    const int warp_m = warp / P::warps_n;
    const int warp_n = warp % P::warps_n;
    const int group = (threadIdx.x & 31) >> 2;
    const int lane4 = threadIdx.x & 3;
    ptx::D4 acc[P::m16][P::n8];
#pragma unroll
    for (int i = 0; i < P::m16; ++i)
#pragma unroll
      for (int j = 0; j < P::n8; ++j) acc[i][j] = ptx::D4{};
    for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
      const int stage = chunk % Stages;
      pipe.consumer_wait(chunk);
      const uint8_t* tile_a = &a[stage][0][0];
      const uint8_t* tile_b = &b[stage][0][0];
#pragma unroll
      for (int block = 0; block < P::blocks; ++block) {
        // `mma_chunk`'s chain: from zero over the block's K16 steps.
        ptx::D4 raw[P::m16][P::n8];
#pragma unroll
        for (int i = 0; i < P::m16; ++i)
#pragma unroll
          for (int j = 0; j < P::n8; ++j) raw[i][j] = ptx::D4{};
#pragma unroll
        for (int kk = block * 128; kk < block * 128 + 128 && kk < TileK; kk += 16) {
          ptx::A4 af[P::m16];
#pragma unroll
          for (int i = 0; i < P::m16; ++i)
            af[i] = load_a_strip<TileK, P::swizzled_a>(tile_a, warp_m * 32 + i * 16, kk);
#pragma unroll
          for (int j = 0; j < P::n8; ++j) {
            const ptx::B2 bf = load_b_strip<W, TileK>(tile_b, warp_n * P::warp_n + j * 8,
                kk, int(row_b), int(chunk * TileK), int(n), int(k), scales);
#pragma unroll
            for (int i = 0; i < P::m16; ++i) ptx::mma_bf16(raw[i][j], af[i], bf);
          }
        }
#pragma unroll
        for (int j = 0; j < P::n8; ++j) {
          if constexpr (W == WeightKind::fp8_block128) {
            const int weight_row = int(row_b) + warp_n * P::warp_n + j * 8;
            const float scale = weight_row < int(n) ?
                scales[(weight_row / 128) * ((int(k) + 127) / 128) + chunk] : 0.0f;
#pragma unroll
            for (int i = 0; i < P::m16; ++i)
#pragma unroll
              for (int v = 0; v < 4; ++v) acc[i][j].v[v] += raw[i][j].v[v] * scale;
          } else {
#pragma unroll
            for (int i = 0; i < P::m16; ++i)
#pragma unroll
              for (int v = 0; v < 4; ++v) acc[i][j].v[v] += raw[i][j].v[v];
          }
        }
      }
      // Every lane's shared reads of the stage are performed before the
      // async proxy may refill it.
      ptx::fence_async_shared();
      __syncwarp();
      if ((threadIdx.x & 31) == 0) pipe.consumer_release(chunk);
    }
    if constexpr (Mutation == ChainsMutation::late_wait) ptx::griddep_wait();
#pragma unroll
    for (int i = 0; i < P::m16; ++i)
#pragma unroll
      for (int j = 0; j < P::n8; ++j)
#pragma unroll
        for (int v = 0; v < 4; ++v) {
          const uint32_t token = row_a + warp_m * 32 + i * 16 + group + (v / 2) * 8;
          const uint32_t column = row_b + warp_n * P::warp_n + j * 8 + lane4 * 2 + (v % 2);
          if (token < live_m && column < n) {
            float value = acc[i][j].v[v];
            const uint64_t offset = uint64_t(token) * n + column;
            if constexpr (Residual) value += residual_value(residual, offset, output_f32);
            if (output_f32) static_cast<float*>(output)[offset] = value;
            else static_cast<__nv_bfloat16*>(output)[offset] = __float2bfloat16(value);
          }
        }
  }
  __syncthreads();
  if (threadIdx.x == 0) pipe.invalidate();
}

} // namespace chop::gemm
