#pragma once

// Dense FP8 128x128 gate/up at prefill (`chained::Schedule::DecodedTiles`)
// with each weight code decoded once per CTA. The column schedule
// (`chained_experts.cuh`) decodes a 64-K step of its weight rows into shared
// memory between two CTA barriers, then runs its MMAs; here the loads, the
// decode and the MMAs run in separate warps behind mbarrier rings, as the
// decoded-tile GEMM (`gemm/decoded_tiles.cuh`) does for TiledChains.
//
// A CTA owns 128 tokens by 64 output columns: the 64 gate rows and the
// matching 64 up rows of the weights, so each math warp holds the gate and
// the up sums of its columns and the clamped SwiGLU needs no exchange.
// Roles of the 384 threads (one tile per CTA, the grid covering the
// capacity's tiles): warp 0 loads 64-K stages of activations (128 token
// rows of BF16 in 128-byte rows, swizzled; three stages) with `cp.async`,
// each lane arriving on the stage's full barrier when its copies land;
// warps 1-3 copy their own 16-byte chunks of raw codes (two private stages
// per thread, `cp.async` groups) and decode them into two swizzled BF16
// stages; warps 4-11 run the MMAs, each over 32 gate and the matching 32 up
// rows by 32 tokens. Every shared hand-off is an mbarrier: full on loader
// or decoder arrivals, empty on consumer arrivals, primed once; the math
// warps fence their stage reads (`fence.proxy.async.shared::cta`) in every
// lane before releasing them.
//
// Per output element the arithmetic is the column schedule's: the operands
// of `mma.m16n8k16` are the weight rows (A: `mul.rn.bf16x2` of the code's
// exact BF16 pair by the power of two of its block scale, `split_scale`)
// and the token rows as stored (B); per 128-K block a chain from zero over
// its eight K16 steps in ascending K (two stages), then `fmaf(chain,
// residual, sum)` with the block scale's residual, blocks in K order; then
// `clamped_swiglu` of the gate and up sums and one BF16 rounding. Which
// warp, fragment or stage holds an element never enters its value, so each
// output is bitwise the column schedule's at every live count. Token rows
// past the live count are neither loaded nor written; 8-token strips wholly
// past it skip their MMAs, and a stale row only reaches its own (unwritten)
// output column.
//
// `Mutation` (lab-only) drops the dependency wait, moves it after the K
// loop of the math warps (the loaders never wait), or folds 256-K chains;
// served entries take `none`.

#include "../common/pipeline.cuh"
#include "../common/ptx_control.cuh"
#include "../common/ptx_matrix.cuh"
#include "chained_experts.cuh"

namespace chop::moe::decoded_tiles {

enum class Mutation : int {
  none = 0,
  drop_wait = 1,   // no griddepcontrol.wait
  late_wait = 2,   // the math warps wait after the K loop, the loaders never
  long_chain = 3,  // each chain spans two scale blocks before its fold
};

struct Layout {
  static constexpr int threads = 384;
  static constexpr int tile_tokens = 128;
  static constexpr int tile_columns = chained::kTileN;  // output columns
  static constexpr int tile_rows = 2 * tile_columns;     // gate rows, then up rows
  static constexpr int step_k = 64;                      // K per stage
  static constexpr int decode_threads = 96;  // warps 1-3
  static constexpr int math_warps = 8;
  static constexpr int warps_n = 2;          // math warps across the output columns
  static constexpr int warp_columns = 32;    // per math warp: 32 gate and 32 up rows
  static constexpr int warp_tokens = 32;
  static constexpr int m16 = 2 * warp_columns / 16;  // gate strips, then up strips
  static constexpr int n8 = warp_tokens / 8;
  static constexpr int a_bytes = tile_tokens * step_k * 2;
  static constexpr int raw_bytes = tile_rows * step_k;
  static constexpr int decoded_bytes = tile_rows * step_k * 2;
  static constexpr int units = raw_bytes / 16;  // 16-byte code chunks a stage
  // Stage rings: activations, raw codes (private to each decode thread) and
  // decoded weights. Kernel-lab timed 2 or 3 activation, 2 to 4 raw and 2
  // or 3 decoded stages within 1% of each other.
  static constexpr int a_stages = 3;
  static constexpr int raw_stages = 2;
  static constexpr int decoded_stages = 2;
  // `chained::Projection::plan`'s dynamic shared bytes: the stages and 1024
  // bytes to align them.
  static constexpr int shared_bytes = a_stages * a_bytes + raw_stages * raw_bytes +
                                      decoded_stages * decoded_bytes + 1024;
  static_assert(warps_n * warp_columns == tile_columns);
  static_assert((math_warps / warps_n) * warp_tokens == tile_tokens);
  static_assert(shared_bytes == 99328);
};

// The byte offset of 16-byte chunk `chunk` of 128-byte row `row` in a
// 1024-byte aligned stage, swizzled so that eight rows' chunks at one
// column fall in distinct banks.
__device__ __forceinline__ uint32_t swizzled(int row, int chunk) {
  return uint32_t(row) * 128 + uint32_t((chunk ^ (row & 7)) * 16);
}

__device__ __forceinline__ void cp_async16(void* target, const void* source) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
               :: "r"(ptx::shared_address(target)), "l"(source) : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template<int Pending>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" :: "n"(Pending) : "memory");
}

__device__ __forceinline__ void cp_async_arrive(uint64_t* bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];"
               :: "r"(ptx::shared_address(bar)) : "memory");
}

// One math warp's fragments of K16 step `kk` of a stage: four weight strips
// (`ldmatrix.x4`, the column schedule's A fragment: rows g and g + 8, K 2t
// and 2t + 8) and the token strips in pairs (one `ldmatrix.x4` gives two
// 8-token strips, each K 0-7 then 8-15).
struct Fragments {
  ptx::A4 a[Layout::m16];
  ptx::B2 b[Layout::n8];
};

__device__ __forceinline__ void load_fragments(
    Fragments& f, const uint8_t* weights, const uint8_t* tokens, int column, int token, int kk,
    int live_n8) {
  using L = Layout;
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int i = 0; i < L::m16; ++i) {
    // Strips 0 and 1 are gate rows, 2 and 3 the matching up rows.
    const int base = (i >= L::m16 / 2 ? L::tile_columns : 0) + column + (i & 1) * 16;
    const int r = base + (lane & 7) + ((lane >> 3) & 1) * 8;
    uint32_t out[4];
    ptx::ldmatrix_b16<4, false>(out, weights + swizzled(r, kk / 8 + (lane >> 4)));
#pragma unroll
    for (int v = 0; v < 4; ++v) f.a[i].v[v] = out[v];
  }
#pragma unroll
  for (int j = 0; j < L::n8; j += 2) {
    if (j < live_n8) {
      const int r = token + j * 8 + (lane & 7) + ((lane >> 4) & 1) * 8;
      uint32_t out[4];
      ptx::ldmatrix_b16<4, false>(out, tokens + swizzled(r, kk / 8 + ((lane >> 3) & 1)));
      f.b[j].v[0] = out[0];
      f.b[j].v[1] = out[1];
      f.b[j + 1].v[0] = out[2];
      f.b[j + 1].v[1] = out[3];
    }
  }
}

template<class C, Mutation M = Mutation::none>
__device__ void run(
    const uint16_t* input, const uint8_t* weight, const float* block_scales,
    const uint8_t* local_scales, const float* globals, const int32_t* offsets,
    const int32_t* sorted_rows, const int32_t* num_padded_rows, const int32_t* live_tokens,
    const float* route_weights, void* output) {
  using L = Layout;
  static_assert(C::weights == chained::Weights::fp8_block128 && C::gate_up && !C::grouped,
                "decoded tiles serve dense FP8 128x128 gate/up");
  static_assert(C::k % 128 == 0 && C::n % 128 == 0);
  (void)local_scales;
  (void)globals;
  (void)offsets;
  (void)sorted_rows;
  (void)num_padded_rows;
  (void)route_weights;
  __shared__ uint64_t full_a[L::a_stages], empty_a[L::a_stages];
  __shared__ uint64_t full_decoded[L::decoded_stages], empty_decoded[L::decoded_stages];
  // The F32 block scales of the tile's gate rows and up rows, per 128-K block.
  __shared__ float scales[2][C::k / 128];
  extern __shared__ __align__(128) unsigned char storage[];
  uint8_t* stage_a = storage + ((1024 - (ptx::shared_address(storage) & 1023)) & 1023);
  uint8_t* stage_raw = stage_a + L::a_stages * L::a_bytes;
  uint8_t* stage_decoded = stage_raw + L::raw_stages * L::raw_bytes;
  if (blockDim.x != L::threads) __trap();
  constexpr int K = C::k;
  constexpr int N = C::n;
  constexpr uint32_t steps = K / L::step_k;
  ptx::griddep_launch_dependents();
  if constexpr (M == Mutation::none || M == Mutation::long_chain) ptx::griddep_wait();
  const int live = *live_tokens;
  if (live > int(gridDim.x) * L::tile_tokens) __trap();
  const int row0 = int(blockIdx.x) * L::tile_tokens;
  if (row0 >= live) return;
  const int n0 = int(blockIdx.y) * L::tile_columns;
  const int warp = int(threadIdx.x / 32);
  const int lane = int(threadIdx.x & 31);
  if (threadIdx.x == 0) {
    for (int i = 0; i < L::a_stages; ++i) {
      ptx::mbarrier_init(&full_a[i], 32);
      ptx::mbarrier_init(&empty_a[i], L::math_warps);
    }
    for (int i = 0; i < L::decoded_stages; ++i) {
      ptx::mbarrier_init(&full_decoded[i], L::decode_threads);
      ptx::mbarrier_init(&empty_decoded[i], L::math_warps);
    }
    ptx::fence_mbarrier_init();
  }
  for (int i = int(threadIdx.x); i < 2 * K / 128; i += L::threads) {
    const int half = i / (K / 128), block = i % (K / 128);
    scales[half][block] =
        block_scales[chained::block_scale_index<C>(0, half * N + n0, block)];
  }
  __syncthreads();
  // Each consumer frees every stage once, so the first round's waits pass.
  if (lane == 0 && warp >= 4) {
    for (int i = 0; i < L::a_stages; ++i) ptx::mbarrier_arrive(&empty_a[i]);
    for (int i = 0; i < L::decoded_stages; ++i) ptx::mbarrier_arrive(&empty_decoded[i]);
  }
  if (warp < 4) {
    configure_producer_warp();
    if (warp == 0) {
      // A warp instruction copies four whole 128-byte token rows.
      for (uint32_t s = 0; s < steps; ++s) {
        const int a = int(s % L::a_stages);
        ptx::mbarrier_wait(&empty_a[a], (s / L::a_stages) & 1);
        uint8_t* tile = stage_a + a * L::a_bytes;
#pragma unroll 4
        for (int i = 0; i < L::tile_tokens / 4; ++i) {
          const int row = i * 4 + lane / 8;
          const int chunk = lane & 7;
          if (row0 + row < live)
            cp_async16(tile + swizzled(row, chunk),
                       input + uint64_t(row0 + row) * K + s * L::step_k + chunk * 8);
        }
        cp_async_arrive(&full_a[a]);
      }
    } else {
      // Each thread copies and decodes the same chunks of every stage: unit
      // u is weight row u / 4 of the tile (gate rows, then up rows), K
      // (u % 4) * 16 + 0..15 of the stage.
      const int thread = int(threadIdx.x) - 32;
      const auto issue = [&](uint32_t s) {
        if (s < steps) {
          uint8_t* raw = stage_raw + (s % L::raw_stages) * L::raw_bytes;
          for (int u = thread; u < L::units; u += L::decode_threads) {
            const int row = chained::weight_row<C>(n0, u >> 2);
            cp_async16(raw + u * 16, weight + uint64_t(row) * K + s * L::step_k + (u & 3) * 16);
          }
        }
        cp_async_commit();
      };
      for (uint32_t s = 0; s + 1 < L::raw_stages; ++s) issue(s);
      for (uint32_t s = 0; s < steps; ++s) {
        issue(s + L::raw_stages - 1);
        cp_async_wait<L::raw_stages - 1>();
        const uint32_t block = s * L::step_k / 128;
        const uint32_t pow2[2] = {chained::split_scale(scales[0][block]).pow2,
                                  chained::split_scale(scales[1][block]).pow2};
        const int d = int(s % L::decoded_stages);
        ptx::mbarrier_wait(&empty_decoded[d], (s / L::decoded_stages) & 1);
        const uint8_t* raw = stage_raw + (s % L::raw_stages) * L::raw_bytes;
        uint8_t* decoded = stage_decoded + d * L::decoded_bytes;
        for (int u = thread; u < L::units; u += L::decode_threads) {
          const uint4 codes = *reinterpret_cast<const uint4*>(raw + u * 16);
          const uint32_t words[4] = {codes.x, codes.y, codes.z, codes.w};
          const int row = u >> 2;
          const uint32_t scale = pow2[row >= L::tile_columns ? 1 : 0];
          uint32_t out[8];
#pragma unroll
          for (int w = 0; w < 4; ++w) {
            out[2 * w] = chained::mul_bf16x2(convert::unpack_bf16x2_e4m3(uint16_t(words[w])),
                                             scale);
            out[2 * w + 1] = chained::mul_bf16x2(
                convert::unpack_bf16x2_e4m3(uint16_t(words[w] >> 16)), scale);
          }
          const int chunk = (u & 3) * 2;
          *reinterpret_cast<uint4*>(decoded + swizzled(row, chunk)) =
              make_uint4(out[0], out[1], out[2], out[3]);
          *reinterpret_cast<uint4*>(decoded + swizzled(row, chunk + 1)) =
              make_uint4(out[4], out[5], out[6], out[7]);
        }
        ptx::mbarrier_arrive(&full_decoded[d]);
      }
      cp_async_wait<0>();
    }
  } else {
    configure_math_warp();
    const int math = warp - 4;
    const int column = (math % L::warps_n) * L::warp_columns;
    const int token = (math / L::warps_n) * L::warp_tokens;
    const int group = lane >> 2;
    const int lane4 = lane & 3;
    // 8-token strips whose first token is live (all, some or none).
    const int first = row0 + token;
    const int live_n8 = first >= live ? 0 : min((live - first + 7) / 8, L::n8);
    ptx::D4 sum[L::m16][L::n8];
    ptx::D4 chain[L::m16][L::n8];
#pragma unroll
    for (int i = 0; i < L::m16; ++i)
#pragma unroll
      for (int j = 0; j < L::n8; ++j)
#pragma unroll
        for (int v = 0; v < 4; ++v) sum[i][j].v[v] = chain[i][j].v[v] = 0.f;
    constexpr uint32_t chain_steps = M == Mutation::long_chain ? 4 : 2;
    for (uint32_t s = 0; s < steps; ++s) {
      const int a = int(s % L::a_stages);
      const int d = int(s % L::decoded_stages);
      const bool fold = s % chain_steps == chain_steps - 1 || s + 1 == steps;
      ptx::mbarrier_wait(&full_a[a], (s / L::a_stages) & 1);
      ptx::mbarrier_wait(&full_decoded[d], (s / L::decoded_stages) & 1);
      if (live_n8 > 0) {
        const uint8_t* tokens = stage_a + a * L::a_bytes;
        const uint8_t* weights = stage_decoded + d * L::decoded_bytes;
        // Fragments of the next K16 step load before this step's MMAs.
        Fragments f[2];
        load_fragments(f[0], weights, tokens, column, token, 0, live_n8);
#pragma unroll
        for (int t = 0; t < L::step_k / 16; ++t) {
          if (t + 1 < L::step_k / 16)
            load_fragments(f[(t + 1) & 1], weights, tokens, column, token, (t + 1) * 16,
                           live_n8);
#pragma unroll
          for (int j = 0; j < L::n8; ++j)
#pragma unroll
            for (int i = 0; i < L::m16; ++i)
              if (j < live_n8) ptx::mma_bf16(chain[i][j], f[t & 1].a[i], f[t & 1].b[j]);
        }
      }
      // Every lane's reads of the stages are performed before they are
      // released for refilling (FIX-SC1's rule for refilled rings).
      ptx::fence_async_shared();
      __syncwarp();
      if (lane == 0) {
        ptx::mbarrier_arrive(&empty_a[a]);
        ptx::mbarrier_arrive(&empty_decoded[d]);
      }
      if (fold) {
        // The 128-K block ends: add its chain into the sums in K order, the
        // gate strips with the gate rows' block scale, the up strips with
        // the up rows'.
        const uint32_t block = s * L::step_k / 128;
        const float residual[2] = {chained::split_scale(scales[0][block]).residual,
                                   chained::split_scale(scales[1][block]).residual};
#pragma unroll
        for (int i = 0; i < L::m16; ++i)
#pragma unroll
          for (int j = 0; j < L::n8; ++j)
#pragma unroll
            for (int v = 0; v < 4; ++v) {
              sum[i][j].v[v] =
                  fmaf(chain[i][j].v[v], residual[i >= L::m16 / 2 ? 1 : 0], sum[i][j].v[v]);
              chain[i][j].v[v] = 0.f;
            }
      }
    }
    if constexpr (M == Mutation::late_wait) ptx::griddep_wait();
#pragma unroll
    for (int i = 0; i < L::m16 / 2; ++i)
#pragma unroll
      for (int j = 0; j < L::n8; ++j)
#pragma unroll
        for (int v = 0; v < 4; ++v) {
          const int t = first + j * 8 + 2 * lane4 + (v & 1);
          if (t >= live) continue;
          const int at = n0 + column + i * 16 + group + (v >= 2 ? 8 : 0);
          const float value = clamp::clamped_swiglu<typename C::clamp>(
              sum[i][j].v[v], sum[i + L::m16 / 2][j].v[v]);
          static_cast<uint16_t*>(output)[uint64_t(t) * N + at] = convert::bf16_from_f32(value);
        }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 0; i < L::a_stages; ++i) {
      ptx::mbarrier_inval(&full_a[i]);
      ptx::mbarrier_inval(&empty_a[i]);
    }
    for (int i = 0; i < L::decoded_stages; ++i) {
      ptx::mbarrier_inval(&full_decoded[i]);
      ptx::mbarrier_inval(&empty_decoded[i]);
    }
  }
}

}  // namespace chop::moe::decoded_tiles
