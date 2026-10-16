#pragma once

// Decoded weight tiles (`decoded_tiles.cuh`) whose final partial wave over
// the SMs is split into 64-row halves when they fit them
// (`gemm::balanced_decoded_tile_knobs`, `common/balanced_tiles.cuh`). One
// tile per CTA leaves a whole wave for a few tiles past a multiple of the
// SM count: at 2,048 rows a 1,536 x 4,096 projection runs 192 tiles in two
// waves, where a wave of tiles and one of halves take about 1.7 tile times.
//
// One item per CTA; roles and stages are `run_a16_decoded`'s (warp 0's
// first lane loads, warps 1-3 decode, warps 4-11 run the MMAs).
// Activations load in 64-row TMA boxes, one per half: two fill an
// activation stage as the 128-row box does (the 128-byte swizzle depends on
// the 1024-byte aligned address only). Each math warp owns 32 tokens by 64
// weight rows of the item, as there; in a 64-row item the warps of the
// lower 64 rows skip their MMAs, as they do in a half-live tile there.
//
// Per output element the arithmetic is `run_a16_decoded`'s: the same
// `mma.m16n8k16` operands, per 128-K block a chain from zero over its eight
// K16 steps in ascending K, `acc = fma(chain, scale, acc)` in K order, then
// the residual add and the output conversion (`store_warp_tile`). Which CTA,
// item, warp or strip computes an element never enters its value, so each
// output element is bitwise the decoded tiles' (and the warp tiles') at
// every live count and grid. `DecodedMutation` (lab-only) drops the
// dependency wait, moves it after the K loop, or folds 256-K chains.

#include "../common/balanced_tiles.cuh"
#include "decoded_tiles.cuh"

namespace chop::gemm {

// `run_a16_decoded`'s epilogue of one math warp's 32 x 64 accumulators
// (tokens from `first_row`, output columns from `column0`): the rows before
// `live_end`, each pair of columns plus its residual (F32, or BF16 widened)
// with one round-to-nearest add, then stored as F32 or rounded to BF16.
template<bool Residual>
__device__ __forceinline__ void store_warp_tile(
    const ptx::D4 (&acc)[DecodedTiles::m16][DecodedTiles::n8], uint32_t first_row,
    uint32_t live_end, uint32_t column0, uint32_t n, uint32_t output_f32, void* output,
    const void* residual) {
  using P = DecodedTiles;
  const int lane = int(threadIdx.x & 31);
  const int group = lane >> 2;
  const int lane4 = lane & 3;
#pragma unroll
  for (int i = 0; i < P::m16; ++i)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint32_t token = first_row + uint32_t(i * 16 + group + h * 8);
      if (token >= live_end) continue;
#pragma unroll
      for (int j = 0; j < P::n8; ++j) {
        const uint32_t column = column0 + uint32_t(j * 8 + lane4 * 2);
        const uint64_t offset = uint64_t(token) * n + column;
        float lo = acc[i][j].v[2 * h];
        float hi = acc[i][j].v[2 * h + 1];
        if (output_f32) {
          if constexpr (Residual) {
            const float2 add = *reinterpret_cast<const float2*>(
                static_cast<const float*>(residual) + offset);
            lo = __fadd_rn(lo, add.x);
            hi = __fadd_rn(hi, add.y);
          }
          *reinterpret_cast<float2*>(static_cast<float*>(output) + offset) = make_float2(lo, hi);
        } else {
          if constexpr (Residual) {
            const __nv_bfloat162 add = *reinterpret_cast<const __nv_bfloat162*>(
                static_cast<const __nv_bfloat16*>(residual) + offset);
            lo = __fadd_rn(lo, __bfloat162float(add.x));
            hi = __fadd_rn(hi, __bfloat162float(add.y));
          }
          __nv_bfloat162 pair;
          pair.x = __float2bfloat16(lo);
          pair.y = __float2bfloat16(hi);
          *reinterpret_cast<__nv_bfloat162*>(static_cast<__nv_bfloat16*>(output) + offset) = pair;
        }
      }
    }
}

// The 16-row strips (of a math warp's two) whose first row, from
// `first_row`, is before `live_end`.
__device__ __forceinline__ int live_strip_count(uint32_t first_row, uint32_t live_end) {
  if (first_row >= live_end) return 0;
  const uint32_t strips = (live_end - first_row + 15) / 16;
  return strips < uint32_t(DecodedTiles::m16) ? int(strips) : DecodedTiles::m16;
}

template<bool Residual, uint32_t Ctas, DecodedMutation Mutation = DecodedMutation::none>
__device__ void run_a16_decoded_balanced(
    const CUtensorMap& map_a, const CUtensorMap& map_b,
    void* output, float* partial, uint32_t* semaphores,
    uint32_t live_m, uint32_t n, uint32_t k, uint32_t splits,
    uint32_t output_f32, const float* scales, const void* residual,
    const float* weight_global_scale) {
  using P = DecodedTiles;
  constexpr int half_bytes = P::a_bytes / 2;  // one 64-row activation box
  (void)partial;
  (void)semaphores;
  (void)weight_global_scale;
  __shared__ uint64_t full_a[P::a_stages], empty_a[P::a_stages];
  __shared__ uint64_t full_raw[P::raw_stages], empty_raw[P::raw_stages];
  __shared__ uint64_t full_decoded[P::decoded_stages], empty_decoded[P::decoded_stages];
  extern __shared__ __align__(128) unsigned char storage[];
  uint8_t* stage_a = storage + ((1024 - (ptx::shared_address(storage) & 1023)) & 1023);
  uint8_t* stage_raw = stage_a + P::a_stages * P::a_bytes;
  uint8_t* stage_decoded = stage_raw + P::raw_stages * P::raw_bytes;
  if (splits != 1 || n % 128 != 0 || k % 128 != 0) __trap();
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
    for (int i = 0; i < P::a_stages; ++i) {
      ptx::mbarrier_init(&full_a[i], 1);
      ptx::mbarrier_init(&empty_a[i], P::math_warps);
    }
    for (int i = 0; i < P::raw_stages; ++i) {
      ptx::mbarrier_init(&full_raw[i], 1);
      ptx::mbarrier_init(&empty_raw[i], P::decode_warps);
    }
    for (int i = 0; i < P::decoded_stages; ++i) {
      ptx::mbarrier_init(&full_decoded[i], P::decode_warps * 32);
      ptx::mbarrier_init(&empty_decoded[i], P::math_warps);
    }
    ptx::fence_mbarrier_init();
  }
  __syncthreads();
  // Each consumer frees every stage once, so the first round's waits pass.
  if (lane == 0 && warp >= 4) {
    for (int i = 0; i < P::a_stages; ++i) ptx::mbarrier_arrive(&empty_a[i]);
    for (int i = 0; i < P::decoded_stages; ++i) ptx::mbarrier_arrive(&empty_decoded[i]);
  }
  if (lane == 0 && warp >= 1 && warp < 4)
    for (int i = 0; i < P::raw_stages; ++i) ptx::mbarrier_arrive(&empty_raw[i]);
  if constexpr (Mutation != DecodedMutation::drop_wait &&
                Mutation != DecodedMutation::late_wait)
    ptx::griddep_wait();
  if (warp < 4) {
    configure_producer_warp();
    if (warp == 0) {
      if (lane == 0) {
        const uint64_t policy = ptx::l2_policy<ptx::L2Evict::normal>();
        const int boxes = int(item.rows / 64);
        for (uint32_t s = 0; s < steps; ++s) {
          // Raw codes first: the decoders run a stage ahead of the MMAs.
          const int r = int(s % P::raw_stages);
          ptx::mbarrier_wait(&empty_raw[r], (s / P::raw_stages) & 1);
          ptx::mbarrier_arrive_expect_tx(&full_raw[r], P::raw_bytes);
          const int32_t cb[2] = {int32_t(s * P::step_k), int32_t(row_b)};
          ptx::tma_load_hint<2>(stage_raw + r * P::raw_bytes, &map_b, cb, &full_raw[r], policy);
          const int a = int(s % P::a_stages);
          ptx::mbarrier_wait(&empty_a[a], (s / P::a_stages) & 1);
          ptx::mbarrier_arrive_expect_tx(&full_a[a], uint32_t(boxes * half_bytes));
          for (int box = 0; box < boxes; ++box) {
            const int32_t ca[2] = {int32_t(s * P::step_k), int32_t(item.row + box * 64)};
            ptx::tma_load_hint<2>(stage_a + a * P::a_bytes + box * half_bytes, &map_a, ca,
                                  &full_a[a], policy);
          }
        }
      }
    } else {
      const int thread = int(threadIdx.x) - 32;
      for (uint32_t s = 0; s < steps; ++s) {
        const int r = int(s % P::raw_stages);
        const uint32_t phase = (s / P::raw_stages) & 1;
        ptx::mbarrier_wait(&full_raw[r], phase);
        ptx::mbarrier_wait(&empty_decoded[r], phase);
        decode_stage(stage_raw + r * P::raw_bytes, stage_decoded + r * P::decoded_bytes, thread,
                     P::decode_warps * 32);
        // Every lane's reads of the raw stage are performed before the
        // async proxy may refill it (FIX-SC1).
        ptx::fence_async_shared();
        __syncwarp();
        if (lane == 0) ptx::mbarrier_arrive(&empty_raw[r]);
        ptx::mbarrier_arrive(&full_decoded[r]);
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
    const float* scale_row = scales + uint64_t(item.block) * (k / 128);
    ptx::D4 acc[P::m16][P::n8];
    ptx::D4 chain[P::m16][P::n8];
#pragma unroll
    for (int i = 0; i < P::m16; ++i)
#pragma unroll
      for (int j = 0; j < P::n8; ++j) {
        acc[i][j] = ptx::D4{};
        chain[i][j] = ptx::D4{};
      }
    constexpr uint32_t chain_steps = Mutation == DecodedMutation::long_chain ? 4 : 2;
    for (uint32_t s = 0; s < steps; ++s) {
      const int a = int(s % P::a_stages);
      const int d = int(s % P::decoded_stages);
      const bool fold = s % chain_steps == chain_steps - 1 || s + 1 == steps;
      const float scale = fold ? scale_row[s / 2] : 0.0f;
      ptx::mbarrier_wait(&full_a[a], (s / P::a_stages) & 1);
      ptx::mbarrier_wait(&full_decoded[d], (s / P::decoded_stages) & 1);
      if (live_strips > 0) {
        const uint8_t* tile_a = stage_a + a * P::a_bytes;
        const uint8_t* tile_b = stage_decoded + d * P::decoded_bytes;
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
      // Every lane's reads of the activation stage are performed before the
      // async proxy may refill it (FIX-SC1).
      ptx::fence_async_shared();
      __syncwarp();
      if (lane == 0) {
        ptx::mbarrier_arrive(&empty_a[a]);
        ptx::mbarrier_arrive(&empty_decoded[d]);
      }
      if (fold) {
        // `mma_chunk`'s promotion: the block's chain times its scale, added
        // in K order.
#pragma unroll
        for (int i = 0; i < P::m16; ++i)
#pragma unroll
          for (int j = 0; j < P::n8; ++j)
#pragma unroll
            for (int v = 0; v < 4; ++v) {
              acc[i][j].v[v] = __fmaf_rn(chain[i][j].v[v], scale, acc[i][j].v[v]);
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
    for (int i = 0; i < P::a_stages; ++i) {
      ptx::mbarrier_inval(&full_a[i]);
      ptx::mbarrier_inval(&empty_a[i]);
    }
    for (int i = 0; i < P::raw_stages; ++i) {
      ptx::mbarrier_inval(&full_raw[i]);
      ptx::mbarrier_inval(&empty_raw[i]);
    }
    for (int i = 0; i < P::decoded_stages; ++i) {
      ptx::mbarrier_inval(&full_decoded[i]);
      ptx::mbarrier_inval(&empty_decoded[i]);
    }
  }
}

} // namespace chop::gemm
