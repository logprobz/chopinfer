#pragma once

// The tiled A16 prefill GEMM over FP8 128x128 block weights
// (`DenseKernel::TiledChains` above 32 rows at `gemm::decoded_tile_knobs`)
// with each weight code decoded once per CTA. Warp tiles (`tiled_warps.cuh`)
// decode every B fragment from the raw codes in each math warp that reads
// it; here three decode warps turn each 64-K stage of raw codes into a
// swizzled BF16 tile that the math warps read with `ldmatrix`, as BF16
// weights are read.
//
// Roles of the 384 threads (one tile per CTA, the grid covering every tile
// at the capacity, so each pipeline serves one tile from its
// initialization): warp 0's first lane loads 64-K stages of activations
// (128 tokens, BF16, 128-byte swizzle; three stages) and raw weight codes
// (128 weight rows, unswizzled; two stages) by TMA; warps 1-3 decode raw
// stages into two BF16 stages; warps 4-11 run the MMAs, each over 32 tokens
// by 64 weight rows. Every hand-off is an mbarrier: full on TMA bytes or
// decoder arrivals, empty on consumer arrivals, primed once; a consumer of a
// TMA-written stage fences its reads to the async proxy
// (`fence.proxy.async.shared::cta`) in every lane before releasing it.
//
// Per output element the arithmetic is the warp tiles' (and `run_a16`'s at
// one split): the same `mma.m16n8k16` BF16 operands (activations as
// stored; weights `cvt.rn.bf16x2.e4m3x2` of the codes, exactly the
// `packed_fp8_weight` values), per 128-K block a chain from zero over its
// eight K16 steps in ascending K (two stages), then `acc = fma(chain,
// scale, acc)` with the block's F32 scale, blocks in K order, then the
// residual add and the output conversion. Which warp, fragment or stage
// holds an element never enters its value, so each output element is
// bitwise the warp tiles' at every live count and grid. Rows of 16-row
// strips past the live count skip their MMAs; no written element reads
// them. `n` and `k` are multiples of 128 (the builder refuses others).
//
// `DecodedMutation` (lab-only) drops the dependency wait, moves it after the
// K loop, or folds 256-K chains; served entries take `none`.

#include "../common/ptx_control.cuh"
#include "../common/ptx_matrix.cuh"
#include "dense.cuh"

namespace chop::gemm {

enum class DecodedMutation : int {
  none = 0,
  drop_wait = 1,   // no griddepcontrol.wait
  late_wait = 2,   // the math warps wait after the K loop, the loaders never
  long_chain = 3,  // each chain spans two scale blocks before its fold
};

struct DecodedTiles {
  static constexpr int tile_m = 128;  // tokens
  static constexpr int tile_n = 128;  // weight rows (output columns)
  static constexpr int step_k = 64;   // K per stage
  static constexpr int a_stages = 3;
  static constexpr int raw_stages = 2;
  static constexpr int decoded_stages = 2;
  static constexpr int decode_warps = 3;
  static constexpr int math_warps = 8;
  static constexpr int warps_n = 2;   // math warps across the weight rows
  static constexpr int warp_m = 32;
  static constexpr int warp_n = 64;
  static constexpr int m16 = warp_m / 16;
  static constexpr int n8 = warp_n / 8;
  static constexpr int a_bytes = tile_m * step_k * 2;
  static constexpr int raw_bytes = tile_n * step_k;
  static constexpr int decoded_bytes = tile_n * step_k * 2;
  static constexpr int units = raw_bytes / 16;  // 16-byte raw chunks a stage
  // `gemm::tile_shared_bytes` of a decoded-tile entry: the stages and 1024
  // bytes to align them for the 128-byte swizzle.
  static constexpr int shared_bytes = a_stages * a_bytes + raw_stages * raw_bytes +
                                      decoded_stages * decoded_bytes + 1024;
  static_assert((math_warps / warps_n) * warp_m == tile_m && warps_n * warp_n == tile_n);
  static_assert(shared_bytes == 99328);
};

// The byte offset of 16-byte chunk `chunk` of 128-byte row `row` in a
// 1024-byte aligned tile with the TMA 128-byte swizzle.
__device__ __forceinline__ uint32_t swizzled_chunk(int row, int chunk) {
  return uint32_t(row) * 128 + uint32_t((chunk ^ (row & 7)) * 16);
}

// Decodes one stage: each 16-byte chunk of codes (weight row u / 4, K
// (u % 4) * 16 + 0..15) becomes two 16-byte chunks of BF16 pairs, the lower
// K in each pair's low half as `packed_fp8_weight` packs them.
__device__ __forceinline__ void decode_stage(const uint8_t* raw, uint8_t* decoded, int thread,
                                             int threads) {
  using P = DecodedTiles;
  for (int u = thread; u < P::units; u += threads) {
    const uint4 codes = *reinterpret_cast<const uint4*>(raw + u * 16);
    const uint32_t words[4] = {codes.x, codes.y, codes.z, codes.w};
    uint32_t pairs[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      pairs[2 * i] = convert::unpack_bf16x2_e4m3(uint16_t(words[i]));
      pairs[2 * i + 1] = convert::unpack_bf16x2_e4m3(uint16_t(words[i] >> 16));
    }
    const int row = u >> 2;
    const int chunk = (u & 3) * 2;
    *reinterpret_cast<uint4*>(decoded + swizzled_chunk(row, chunk)) =
        make_uint4(pairs[0], pairs[1], pairs[2], pairs[3]);
    *reinterpret_cast<uint4*>(decoded + swizzled_chunk(row, chunk + 1)) =
        make_uint4(pairs[4], pairs[5], pairs[6], pairs[7]);
  }
}

// One math warp's fragments of K16 step `kk` of a stage: the A strips
// (`load_a_strip`'s `ldmatrix.x4` lanes) and the B strips in pairs (one
// `ldmatrix.x4` gives two 8-row strips, each K 0-7 then 8-15, as
// `load_b_strip`'s `ldmatrix.x2` gives one).
struct DecodedFragments {
  ptx::A4 a[DecodedTiles::m16];
  ptx::B2 b[DecodedTiles::n8];
};

__device__ __forceinline__ void load_decoded_fragments(
    DecodedFragments& f, const uint8_t* tile_a, const uint8_t* tile_b, int row, int col,
    int kk, int live_strips) {
  using P = DecodedTiles;
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int i = 0; i < P::m16; ++i) {
    if (i < live_strips) {
      const int r = row + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
      uint32_t out[4];
      ptx::ldmatrix_b16<4, false>(out, tile_a + swizzled_chunk(r, kk / 8 + (lane >> 4)));
#pragma unroll
      for (int v = 0; v < 4; ++v) f.a[i].v[v] = out[v];
    }
  }
#pragma unroll
  for (int j = 0; j < P::n8; j += 2) {
    const int r = col + j * 8 + (lane & 7) + ((lane >> 4) & 1) * 8;
    uint32_t out[4];
    ptx::ldmatrix_b16<4, false>(out, tile_b + swizzled_chunk(r, kk / 8 + ((lane >> 3) & 1)));
    f.b[j].v[0] = out[0];
    f.b[j].v[1] = out[1];
    f.b[j + 1].v[0] = out[2];
    f.b[j + 1].v[1] = out[3];
  }
}

template<bool Residual, DecodedMutation Mutation = DecodedMutation::none>
__device__ void run_a16_decoded(
    const CUtensorMap& map_a, const CUtensorMap& map_b,
    void* output, float* partial, uint32_t* semaphores,
    uint32_t live_m, uint32_t n, uint32_t k, uint32_t splits,
    uint32_t output_f32, const float* scales, const void* residual,
    const float* weight_global_scale) {
  using P = DecodedTiles;
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
  const uint32_t tiles_a = (live_m + P::tile_m - 1) / P::tile_m;
  const uint32_t tiles_b = n / P::tile_n;
  const uint64_t tile_count = uint64_t(tiles_a) * tiles_b;
  const uint32_t steps = k / P::step_k;
  ptx::griddep_launch_dependents();
  // One tile per CTA (the grid covers the capacity's tiles); CTAs past the
  // live tiles exit before touching the pipeline.
  const uint64_t tile = blockIdx.x;
  if (tile >= tile_count) return;
  const uint32_t row_a = uint32_t(tile / tiles_b) * P::tile_m;
  const uint32_t row_b = uint32_t(tile % tiles_b) * P::tile_n;
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
        for (uint32_t s = 0; s < steps; ++s) {
          // Raw codes first: the decoders run a stage ahead of the MMAs.
          const int r = int(s % P::raw_stages);
          ptx::mbarrier_wait(&empty_raw[r], (s / P::raw_stages) & 1);
          ptx::mbarrier_arrive_expect_tx(&full_raw[r], P::raw_bytes);
          const int32_t cb[2] = {int32_t(s * P::step_k), int32_t(row_b)};
          ptx::tma_load_hint<2>(stage_raw + r * P::raw_bytes, &map_b, cb, &full_raw[r], policy);
          const int a = int(s % P::a_stages);
          ptx::mbarrier_wait(&empty_a[a], (s / P::a_stages) & 1);
          ptx::mbarrier_arrive_expect_tx(&full_a[a], P::a_bytes);
          const int32_t ca[2] = {int32_t(s * P::step_k), int32_t(row_a)};
          ptx::tma_load_hint<2>(stage_a + a * P::a_bytes, &map_a, ca, &full_a[a], policy);
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
        // Every lane's reads of the raw stage are performed before the async
        // proxy may refill it (FIX-SC1).
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
    const int group = lane >> 2;
    const int lane4 = lane & 3;
    // 16-row strips whose first row is live (all, some or none).
    const uint32_t first_row = row_a + uint32_t(row);
    const int live_strips = first_row >= live_m ? 0 :
        int((live_m - first_row + 15) / 16) < P::m16 ? int((live_m - first_row + 15) / 16) :
        P::m16;
    const float* scale_row = scales + uint64_t(row_b / 128) * (k / 128);
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
#pragma unroll
    for (int i = 0; i < P::m16; ++i)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const uint32_t token = first_row + uint32_t(i * 16 + group + h * 8);
        if (token >= live_m) continue;
#pragma unroll
        for (int j = 0; j < P::n8; ++j) {
          const uint32_t column = row_b + uint32_t(col + j * 8 + lane4 * 2);
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
            *reinterpret_cast<__nv_bfloat162*>(static_cast<__nv_bfloat16*>(output) + offset) =
                pair;
          }
        }
      }
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
