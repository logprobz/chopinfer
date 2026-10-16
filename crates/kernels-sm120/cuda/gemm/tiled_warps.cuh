#pragma once

// The tiled A16 prefill GEMM (`run_a16` in dense.cuh at one split, tokens
// as the MMA's M) with warp tiles (`DenseKernel::TiledChains` above 32
// rows). Each of the eight math warps owns a 32-row by (TileN / WarpsN)
// column block of the tile for the whole K loop, so every chunk is loaded
// once (run_a16 reloads it once per wave of 16 x 8 subtiles), and BF16
// fragments come from `ldmatrix` at the same swizzled addresses
// `packed_bf16` reads. The operands, the MMA calls of each 16 x 8 subtile
// and their order are run_a16's: per chunk and 128-K block, a chain from
// zero over its K16 steps in ascending K, then `acc += chain` (BF16) or
// `acc += chain * scale` with the FP8 128x128 block scale of the subtile's
// first weight row (zero past N), the same expressions as `mma_chunk`.
// FP8 weights decode as `packed_fp8_weight` does, zero past N or K. The
// epilogue is run_a16's one-split epilogue. Each CTA computes one tile (the
// grid covers every tile at the capacity), so the pipeline serves one tile
// from its initialization. So each output element is
// bitwise run_a16's at one split, at every live M and grid. `Mutation`
// (`ChainsMutation`, lab-only) drops the dependency wait or moves it after
// the K loop; served entries take `none`.

#include "../common/ptx_matrix.cuh"
#include "tiled_chains.cuh"

namespace chop::gemm {

template<WeightKind W, int TileM, int TileN, int TileK>
struct WarpTiles {
  static constexpr int math_warps = 8;
  static constexpr int warps_m = TileM / 32;
  static constexpr int warps_n = math_warps / warps_m;
  static constexpr int warp_n = TileN / warps_n;
  static constexpr int m16 = 2;            // 16-row strips per warp
  static constexpr int n8 = warp_n / 8;    // 8-column strips per warp
  static constexpr int blocks = (TileK + 127) / 128;
  static constexpr bool swizzled_a = W == WeightKind::bf16;
  static_assert(W == WeightKind::bf16 || W == WeightKind::fp8_block128,
                "warp tiles take BF16 or FP8 128x128 weights");
  static_assert(TileM == 64 || TileM == 128, "32-row warp blocks");
  static_assert(warps_m * warps_n == math_warps && TileN % (warps_n * 8) == 0);
  static_assert(n8 >= 1 && m16 * n8 <= 16, "accumulators per warp");
  static_assert(W != WeightKind::fp8_block128 || TileK == 128);
};

// One lane's A fragment of the 16-row strip at `row` and K16 step `k`:
// `ldmatrix.x4` of the swizzled BF16 tile (rows row..row+15, K k..k+15),
// or the four pairs `packed_bf16` reads from an unswizzled tile.
template<int TileK, bool Swizzled>
__device__ __forceinline__ ptx::A4 load_a_strip(const uint8_t* tile, int row, int k) {
  const int lane = threadIdx.x & 31;
  ptx::A4 a;
  if constexpr (Swizzled) {
    const int r = row + (lane & 7) + ((lane >> 3) & 1) * 8;
    const int kk = k + (lane >> 4) * 8;
    const uint32_t offset = smem::swizzle_128(tile, r * TileK * 2 + kk * 2, TileK * 2);
    uint32_t out[4];
    ptx::ldmatrix_b16<4, false>(out, tile + offset);
    for (int i = 0; i < 4; ++i) a.v[i] = out[i];
  } else {
    const int group = lane >> 2, lane4 = lane & 3;
    const uint16_t* base = reinterpret_cast<const uint16_t*>(tile);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      a.v[i] = packed_bf16<TileK, false>(base, row + group + (i % 2) * 8,
                                         k + (i / 2) * 8 + lane4 * 2);
  }
  return a;
}

// One lane's B fragment of the 8-column strip at `col` and K16 step `k`:
// `ldmatrix.x2` of a swizzled BF16 weight tile, or `packed_fp8_weight`'s
// decoded pairs of an FP8 tile.
template<WeightKind W, int TileK>
__device__ __forceinline__ ptx::B2 load_b_strip(
    const uint8_t* tile, int col, int k, int row_b, int chunk_k, int total_n,
    int total_k, const float* scales) {
  const int lane = threadIdx.x & 31;
  ptx::B2 b;
  if constexpr (W == WeightKind::bf16) {
    const int r = col + (lane & 7);
    const int kk = k + ((lane >> 3) & 1) * 8;
    const uint32_t offset = smem::swizzle_128(tile, r * TileK * 2 + kk * 2, TileK * 2);
    uint32_t out[2];
    ptx::ldmatrix_b16<2, false>(out, tile + offset);
    b.v[0] = out[0];
    b.v[1] = out[1];
  } else {
    const int group = lane >> 2, lane4 = lane & 3;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int local_row = col + group;
      const int local_k = k + i * 8 + lane4 * 2;
      b.v[i] = packed_fp8_weight<TileK, W, true>(tile, local_row, local_k,
          row_b + local_row, chunk_k + local_k, total_n, total_k, scales);
    }
  }
  return b;
}

template<WeightKind W, bool Residual, int TileM, int TileN, int TileK,
         int Stages = 2, ChainsMutation Mutation = ChainsMutation::none>
__device__ void run_a16_warps(
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
