#pragma once

// The tiled A16 prefill GEMM over BF16 weights (`DenseKernel::TiledChains`
// above 32 rows at `gemm::staged_tile_knobs`) in the decoded tiles' layout
// (`decoded_tiles.cuh`) without the decode: warp 0's first lane loads
// 64-K stages of activations (128 tokens) and weights (128 weight rows),
// both BF16 with the 128-byte swizzle, by TMA into a ring of three (warps
// 1-3 only complete the producer warpgroup's register split); warps 4-11,
// each over 32 tokens by 64 weight rows, read both operands with
// `ldmatrix.x4` (`load_decoded_fragments`), the next K16 step's fragments
// loading before this step's MMAs. The warp tiles (`tiled_warps.cuh`) at
// 64 x 128 give each math warp 32 x 32 and load each 8-column weight strip
// with its own `ldmatrix.x2`.
//
// One item per CTA, the final partial wave over `Ctas` SMs split into
// 64-row halves when they fit them (`common/balanced_tiles.cuh`); a 64-row
// item loads the 128-row activation box from its first row and the warps of
// its lower 64 rows skip their MMAs. Every
// hand-off is an mbarrier: full on TMA bytes, empty on the math warps'
// arrivals, primed once; each math lane fences its reads to the async proxy
// (`fence.proxy.async.shared::cta`) before its warp releases a stage
// (FIX-SC1).
//
// Per output element the arithmetic is the warp tiles' (and `run_a16`'s at
// one split): the same `mma.m16n8k16` BF16 operands, per 64-K chunk a chain
// from zero over its four K16 steps in ascending K, then `acc += chain` in
// K order, then the residual add and the output conversion
// (`store_warp_tile`). Which CTA, item, warp or stage holds an element
// never enters its value, so each output element is bitwise the warp tiles'
// at every live count and grid. `n` is a multiple of 128 and `k` of 64 (the
// builder refuses others). `DecodedMutation` (lab-only) drops the
// dependency wait, moves it after the K loop, or adds 128-K chains.

#include "decoded_balanced.cuh"

namespace chop::gemm {

struct StagedTiles {
  static constexpr int stages = 3;
  static constexpr int a_bytes = DecodedTiles::a_bytes;        // 128 tokens x 64 K
  static constexpr int b_bytes = DecodedTiles::decoded_bytes;  // 128 rows x 64 K
  static constexpr int threads = 12 * 32;  // a loader warpgroup and eight math warps
  // `gemm::tile_shared_bytes` of a staged-tile entry: the stages and 1024
  // bytes to align them for the 128-byte swizzle.
  static constexpr int shared_bytes = stages * (a_bytes + b_bytes) + 1024;
  static_assert(shared_bytes == 99328);
};

template<bool Residual, uint32_t Ctas, DecodedMutation Mutation = DecodedMutation::none>
__device__ void run_a16_staged(
    const CUtensorMap& map_a, const CUtensorMap& map_b,
    void* output, float* partial, uint32_t* semaphores,
    uint32_t live_m, uint32_t n, uint32_t k, uint32_t splits,
    uint32_t output_f32, const float* scales, const void* residual,
    const float* weight_global_scale) {
  using P = DecodedTiles;
  using S = StagedTiles;
  (void)partial;
  (void)semaphores;
  (void)scales;
  (void)weight_global_scale;
  __shared__ uint64_t full[S::stages], empty[S::stages];
  extern __shared__ __align__(128) unsigned char storage[];
  uint8_t* stage_a = storage + ((1024 - (ptx::shared_address(storage) & 1023)) & 1023);
  uint8_t* stage_b = stage_a + S::stages * S::a_bytes;
  if (splits != 1 || n % 128 != 0 || k % 64 != 0 || blockDim.x != S::threads) __trap();
  const int warp = int(threadIdx.x / 32);
  const int lane = int(threadIdx.x & 31);
  const uint32_t steps = k / P::step_k;
  ptx::griddep_launch_dependents();
  // CTAs past the items exit before touching the pipeline.
  balanced::Item item;
  if (!balanced::Tiles<Ctas>(live_m, n / P::tile_n).item(blockIdx.x, item)) return;
  const uint32_t row_b = item.block * P::tile_n;
  if (threadIdx.x == 0) {
    const int32_t weight_coord[2] = {0, int32_t(row_b)};
    ptx::prefetch_tensor_l2(&map_b, weight_coord);
    for (int i = 0; i < S::stages; ++i) {
      ptx::mbarrier_init(&full[i], 1);
      ptx::mbarrier_init(&empty[i], P::math_warps);
    }
    ptx::fence_mbarrier_init();
  }
  __syncthreads();
  // Each math warp frees every stage once, so the first round's waits pass.
  if (lane == 0 && warp >= 4)
    for (int i = 0; i < S::stages; ++i) ptx::mbarrier_arrive(&empty[i]);
  if constexpr (Mutation != DecodedMutation::drop_wait &&
                Mutation != DecodedMutation::late_wait)
    ptx::griddep_wait();
  if (warp < 4) {
    configure_producer_warp();
    if (warp == 0 && lane == 0) {
      const uint64_t policy = ptx::l2_policy<ptx::L2Evict::normal>();
      for (uint32_t s = 0; s < steps; ++s) {
        const int x = int(s % S::stages);
        ptx::mbarrier_wait(&empty[x], (s / S::stages) & 1);
        ptx::mbarrier_arrive_expect_tx(&full[x], S::a_bytes + S::b_bytes);
        const int32_t cb[2] = {int32_t(s * P::step_k), int32_t(row_b)};
        ptx::tma_load_hint<2>(stage_b + x * S::b_bytes, &map_b, cb, &full[x], policy);
        const int32_t ca[2] = {int32_t(s * P::step_k), int32_t(item.row)};
        ptx::tma_load_hint<2>(stage_a + x * S::a_bytes, &map_a, ca, &full[x], policy);
      }
    }
  } else {
    configure_math_warp();
    const int math = warp - 4;
    const int row = (math / P::warps_n) * P::warp_m;
    const int col = (math % P::warps_n) * P::warp_n;
    const uint32_t first_row = item.row + uint32_t(row);
    const uint32_t live_end = live_m < item.row + item.rows ? live_m : item.row + item.rows;
    const int live_strips = live_strip_count(first_row, live_end);
    ptx::D4 acc[P::m16][P::n8];
    ptx::D4 chain[P::m16][P::n8];
#pragma unroll
    for (int i = 0; i < P::m16; ++i)
#pragma unroll
      for (int j = 0; j < P::n8; ++j) {
        acc[i][j] = ptx::D4{};
        chain[i][j] = ptx::D4{};
      }
    constexpr uint32_t chain_steps = Mutation == DecodedMutation::long_chain ? 2 : 1;
    for (uint32_t s = 0; s < steps; ++s) {
      const int x = int(s % S::stages);
      const bool fold = s % chain_steps == chain_steps - 1 || s + 1 == steps;
      ptx::mbarrier_wait(&full[x], (s / S::stages) & 1);
      if (live_strips > 0) {
        const uint8_t* tile_a = stage_a + x * S::a_bytes;
        const uint8_t* tile_b = stage_b + x * S::b_bytes;
        // Fragments of the next K16 step load before this step's MMAs.
        DecodedFragments f[2];
        load_decoded_fragments(f[0], tile_a, tile_b, row, col, 0, live_strips);
#pragma unroll
        for (int t = 0; t < P::step_k / 16; ++t) {
          if (t + 1 < P::step_k / 16)
            load_decoded_fragments(f[(t + 1) & 1], tile_a, tile_b, row, col, (t + 1) * 16,
                                   live_strips);
#pragma unroll
          for (int j = 0; j < P::n8; ++j)
#pragma unroll
            for (int i = 0; i < P::m16; ++i)
              if (i < live_strips) ptx::mma_bf16(chain[i][j], f[t & 1].a[i], f[t & 1].b[j]);
        }
      }
      // Every lane's reads of the stage are performed before the async
      // proxy may refill it (FIX-SC1).
      ptx::fence_async_shared();
      __syncwarp();
      if (lane == 0) ptx::mbarrier_arrive(&empty[x]);
      if (fold) {
        // `mma_chunk`'s BF16 sum: the chunk's chain added in K order.
#pragma unroll
        for (int i = 0; i < P::m16; ++i)
#pragma unroll
          for (int j = 0; j < P::n8; ++j)
#pragma unroll
            for (int v = 0; v < 4; ++v) {
              acc[i][j].v[v] = __fadd_rn(acc[i][j].v[v], chain[i][j].v[v]);
              chain[i][j].v[v] = 0.0f;
            }
      }
    }
    if constexpr (Mutation == DecodedMutation::late_wait) ptx::griddep_wait();
    store_warp_tile<Residual>(acc, first_row, live_end, row_b + uint32_t(col), n, output_f32,
                              output, residual);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 0; i < S::stages; ++i) {
      ptx::mbarrier_inval(&full[i]);
      ptx::mbarrier_inval(&empty[i]);
    }
  }
}

} // namespace chop::gemm
