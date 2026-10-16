#pragma once

#include <math.h>
#include <stdint.h>
#include "prefill_mainloop.cuh"
#include "../row/row.cuh"
#include "../common/gelu.cuh"

namespace chop::gemm {

// Encoder BF16 x BF16 GEMM with an epilogue functor. Weights and
// activations are plain row-major [rows][K] BF16, as the MM-7 ABI passes
// them. A persistent CTA owns 128 tokens (fewer in a split tail wave) by 128
// weight rows; SwiGLU CTAs
// stage 64 gate rows and the matching 64 up rows so each warp holds both
// halves of its outputs. A producer warpgroup copies each 64-K stage of both
// operands with 16-byte cp.async into 128-byte swizzled rows (rows past live
// M are zero-filled) and signals the stage's full barrier with
// cp.async.mbarrier.arrive.noinc. Eight math warps read both operands with
// conflict-free ldmatrix.x4 and release the stage through an empty barrier.
//
// The epilogue functor `Epilogue` is the only part a linear variant chooses
// (cuda/encoder/linear_*.cuh): `bias` says whether the GEMM adds a BF16 bias,
// `swiglu` whether the weight rows are SwiGLU gate and up halves, and
// `apply` maps each output's biased F32 accumulator (with its up half's for
// SwiGLU) to the value stored as BF16. A new epilogue is a new functor, never
// an edit here.
template<int N, int K, bool Swiglu>
struct EncoderGemmTraits {
  static_assert(N % 128 == 0 && K % 64 == 0);
  static constexpr int tokens = 128;
  static constexpr int rows = 128;
  static constexpr int math_warps = 8;
  static constexpr int producer_threads = 128;
  static constexpr int threads = math_warps * 32 + producer_threads;
  static constexpr int operand_bytes = 128 * prefill::swizzle_row_bytes;
  static constexpr int stage_bytes = 2 * operand_bytes;
  static constexpr int stages = 3;
  static constexpr int shared_bytes = stages * stage_bytes + 1024;
  static constexpr int width = Swiglu ? N / 2 : N;
  static constexpr int output_rows = Swiglu ? 64 : 128;
  static constexpr int feature_tiles = width / output_rows;
  static constexpr int k_stages = K / prefill::stage_k;
};

__device__ __forceinline__ void cp_async_zfill16(
    uint32_t shared, const void* global, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
      :: "r"(shared), "l"(global), "r"(valid ? 16 : 0) : "memory");
}

__device__ __forceinline__ void cp_async_arrive_noinc(uint64_t* bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];"
      :: "r"(ptx::shared_address(bar)) : "memory");
}

// A fragment (16 weight rows by 16 K) from a 128-byte swizzled row tile.
__device__ __forceinline__ ptx::A4 load_a_swizzled(
    uint32_t tile, uint32_t first_row, uint32_t chunk_base) {
  const int lane = threadIdx.x & 31;
  const uint32_t row = first_row + (lane & 15);
  const uint32_t chunk = chunk_base + (lane >> 4);
  const uint32_t address = tile + row * prefill::swizzle_row_bytes +
      ((chunk ^ (row & 7)) << 4);
  ptx::A4 a;
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
      : "=r"(a.v[0]), "=r"(a.v[1]), "=r"(a.v[2]), "=r"(a.v[3])
      : "r"(address));
  return a;
}

template<int Gelu>
__device__ __forceinline__ float encoder_activation(float x) {
  if constexpr (Gelu == 1)
    return chop::gelu_erf(x);
  if constexpr (Gelu == 2) {
    // tanh(u) = 1 - 2 / (exp(2u) + 1), which saturates correctly and keeps
    // the call inline under the math warps' register budget.
    const float u = 0.7978845608028653559f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (2.f - 2.f * __frcp_rn(expf(2.f * u) + 1.f));
  }
  return x;
}

// Tail split: when the last wave of 128-token tiles would leave SMs idle,
// its tiles are cut into `split` token slices (64 or 32 tokens). The cost
// weights (full tile 10, half 6, quarter 4) are measured on SM120, where the
// smaller slices reread the same weight stage per token and are bound by L2.
__device__ __forceinline__ uint32_t encoder_tail_split(
    uint32_t tail, uint32_t grid) {
  if (tail == 0) return 1;
  uint32_t best = 1;
  uint32_t best_cost = 10 * ((tail + grid - 1) / grid);
  const uint32_t cost2 = 6 * ((tail * 2 + grid - 1) / grid);
  if (cost2 < best_cost) { best = 2; best_cost = cost2; }
  const uint32_t cost4 = 4 * ((tail * 4 + grid - 1) / grid);
  if (cost4 < best_cost) best = 4;
  return best;
}

// Walks this CTA's work items: full 128-token tiles first, then the tail
// tiles cut into `split` token slices. Calls `body.template run<Tokens>` with
// the slice's feature tile and first token; slices past live M are skipped
// identically by the producer and math warps.
template<int FeatureTiles, class Body>
__device__ __forceinline__ void encoder_for_each_tile(
    uint32_t live, Body& body) {
  const uint32_t token_tiles = (live + 127) / 128;
  const uint32_t tiles = token_tiles * FeatureTiles;
  const uint32_t full = tiles - tiles % gridDim.x;
  const uint32_t split = encoder_tail_split(tiles - full, gridDim.x);
  const uint32_t items = full + (tiles - full) * split;
  for (uint32_t item = blockIdx.x; item < items; item += gridDim.x) {
    uint32_t tile = item, part = 0;
    if (item >= full) {
      tile = full + (item - full) / split;
      part = (item - full) % split;
    }
    const uint32_t feature_tile = tile / token_tiles;
    const uint32_t token0 = (tile % token_tiles) * 128 + part * (128 / split);
    if (token0 >= live) continue;
    if (split == 1 || item < full)
      body.template run<128>(feature_tile, token0);
    else if (split == 2)
      body.template run<64>(feature_tile, token0);
    else
      body.template run<32>(feature_tile, token0);
  }
}

template<int N, int K, bool Swiglu>
struct EncoderProducer {
  using T = EncoderGemmTraits<N, K, Swiglu>;
  const uint16_t* activation;
  const uint16_t* weight;
  uint64_t* full;
  uint64_t* empty;
  uint32_t storage_address;
  uint32_t live;
  uint32_t producer;
  uint32_t iteration;

  // Staged weight row r of a feature tile.
  __device__ __forceinline__ static uint32_t weight_row(
      uint32_t feature_tile, uint32_t r) {
    if constexpr (Swiglu)
      return r < 64 ? feature_tile * 64 + r
                    : T::width + feature_tile * 64 + r - 64;
    return feature_tile * 128 + r;
  }

  // Each thread copies 8 of the stage's 1024 weight chunks and Tokens / 16
  // of its activation chunks.
  template<int Tokens>
  __device__ __forceinline__ void run(uint32_t feature_tile, uint32_t token0) {
#pragma unroll 1
    for (int stage = 0; stage < T::k_stages; ++stage, ++iteration) {
      const uint32_t slot = iteration % T::stages;
      prefill::wait_empty(empty, iteration, T::stages);
      const uint32_t base = storage_address + slot * T::stage_bytes;
      // A partial unroll keeps the copy addresses within 40 registers.
#pragma unroll 2
      for (int i = 0; i < 8; ++i) {
        const uint32_t chunk_index = producer + i * T::producer_threads;
        const uint32_t r = chunk_index >> 3;
        const uint32_t c = chunk_index & 7;
        const uint32_t swizzled = r * prefill::swizzle_row_bytes +
            ((c ^ (r & 7)) << 4);
        if (i < Tokens / 16) {
          const uint32_t token = token0 + r;
          const bool valid = token < live;
          cp_async_zfill16(base + swizzled,
              activation + uint64_t(valid ? token : 0) * K +
                  stage * prefill::stage_k + c * 8,
              valid);
        }
        cp_async_zfill16(base + T::operand_bytes + swizzled,
            weight + uint64_t(weight_row(feature_tile, r)) * K +
                stage * prefill::stage_k + c * 8,
            true);
      }
      cp_async_arrive_noinc(&full[slot]);
    }
  }

  // Keeps the warpgroup until its copies land.
  __device__ __forceinline__ void finish() {
    asm volatile("cp.async.wait_all;" ::: "memory");
  }
};

template<int N, int K, class Epilogue>
struct EncoderMath {
  static constexpr bool Swiglu = Epilogue::swiglu;
  using T = EncoderGemmTraits<N, K, Swiglu>;
  const uint16_t* bias;
  uint16_t* output;
  uint64_t* full;
  uint64_t* empty;
  uint32_t storage_address;
  uint32_t live;
  uint32_t iteration;

  // Eight math warps split a tile as two feature halves of 64 rows by four
  // token quarters of Tokens / 4.
  template<int Tokens>
  __device__ __forceinline__ void run(uint32_t feature_tile, uint32_t token0) {
    constexpr int warp_tokens = Tokens / 4;
    constexpr int n8 = warp_tokens / 8;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warp_f = warp >> 2;
    const int warp_t = warp & 3;
    // Staged weight strips this warp multiplies: four rows of 16 for plain
    // outputs; two gate strips and the two matching up strips for SwiGLU.
    uint32_t strips[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
      strips[i] = Swiglu ? (i < 2 ? warp_f * 2 + i : 4 + warp_f * 2 + i - 2)
                         : warp_f * 4 + i;
    const uint32_t b_row =
        warp_t * warp_tokens + (lane & 7) + ((lane >> 4) << 3);
    const uint32_t b_half = (lane >> 3) & 1;
    const uint32_t b_xor = lane & 7;
    const bool active = token0 + warp_t * warp_tokens < live;
    ptx::D4 acc[4][n8];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int j = 0; j < n8; ++j) acc[i][j] = ptx::D4{};
#pragma unroll 1
    for (int stage = 0; stage < T::k_stages; ++stage, ++iteration) {
      const uint32_t slot = iteration % T::stages;
      prefill::wait_full(full, iteration, T::stages);
      const uint32_t activation_tile = storage_address + slot * T::stage_bytes;
      const uint32_t weight_tile = activation_tile + T::operand_bytes;
      const uint32_t b_address =
          activation_tile + b_row * prefill::swizzle_row_bytes;
      if (active) {
#pragma unroll
        for (int step = 0; step < 4; ++step) {
          ptx::A4 a[4];
#pragma unroll
          for (int i = 0; i < 4; ++i)
            a[i] = load_a_swizzled(weight_tile, strips[i] * 16, step * 2);
          const uint32_t chunk = step * 2 + b_half;
          ptx::B2 b[n8 < 2 ? 2 : n8];
          if constexpr (n8 == 1) {
            // Lanes 0-15 address both K halves of the warp's eight tokens.
            const uint32_t address =
                b_address + ((chunk ^ b_xor) << 4);
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                : "=r"(b[0].v[0]), "=r"(b[0].v[1]) : "r"(address));
          } else {
#pragma unroll
            for (int j = 0; j < n8; j += 2)
              prefill::load_b2(
                  b_address + j * 8 * prefill::swizzle_row_bytes, chunk,
                  b_xor, b[j], b[j + 1]);
          }
#pragma unroll
          for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < n8; ++j) ptx::mma_bf16(acc[i][j], a[i], b[j]);
        }
      }
      // Every lane releases its own reads, which racecheck can follow
      // (it does not order other lanes' reads through a lane-0 arrive).
      ptx::mbarrier_arrive(&empty[slot]);
    }
    if (!active) return;

    // Epilogue: bias, activation, BF16 pairs of adjacent features.
    const uint32_t group = lane >> 2;
    const bool even = (group & 1) == 0;
    const uint32_t token = token0 + warp_t * warp_tokens + (lane & 3) * 2;
#pragma unroll
    for (int i = 0; i < (Swiglu ? 2 : 4); ++i) {
      const uint32_t feature = (Swiglu
          ? feature_tile * 64 + (warp_f * 2 + i) * 16
          : feature_tile * 128 + strips[i] * 16) + group;
      float bias_value[2] = {0.f, 0.f}, up_bias[2] = {0.f, 0.f};
      if constexpr (Epilogue::bias) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          bias_value[h] = chop::row::bf(bias[feature + h * 8]);
          if constexpr (Swiglu)
            up_bias[h] = chop::row::bf(bias[T::width + feature + h * 8]);
        }
      }
#pragma unroll
      for (int j = 0; j < n8; ++j) {
        float c[4];
#pragma unroll
        for (int v = 0; v < 4; ++v) {
          const float x = acc[i][j].v[v] + bias_value[v / 2];
          if constexpr (Swiglu)
            c[v] = Epilogue::apply(x, acc[i + 2][j].v[v] + up_bias[v / 2]);
          else
            c[v] = Epilogue::apply(x);
        }
        // Lanes swap with their g^1 neighbour so each store covers two
        // adjacent features of one token.
        const float x0 = __shfl_xor_sync(0xffffffff, even ? c[1] : c[0], 4);
        const float x1 = __shfl_xor_sync(0xffffffff, even ? c[3] : c[2], 4);
        const uint32_t row = token + j * 8 + (even ? 0 : 1);
        const uint32_t column = even ? feature : feature - 1;
        if (row < live) {
          const uint64_t offset = uint64_t(row) * T::width + column;
          const uint32_t first =
              uint32_t(chop::row::to_bf(even ? c[0] : x0)) |
              uint32_t(chop::row::to_bf(even ? x0 : c[1])) << 16;
          const uint32_t second =
              uint32_t(chop::row::to_bf(even ? c[2] : x1)) |
              uint32_t(chop::row::to_bf(even ? x1 : c[3])) << 16;
          *reinterpret_cast<uint32_t*>(output + offset) = first;
          *reinterpret_cast<uint32_t*>(output + offset + 8) = second;
        }
      }
    }
  }
};

template<int N, int K, class Epilogue>
__device__ void encoder_gemm(const uint16_t* activation,
                             const uint16_t* weight,
                             const uint16_t* bias, uint16_t* output,
                             const uint32_t* live_rows) {
  constexpr bool Swiglu = Epilogue::swiglu;
  using T = EncoderGemmTraits<N, K, Swiglu>;
  __shared__ uint64_t full[T::stages];
  __shared__ uint64_t empty[T::stages];
  extern __shared__ unsigned char raw[];
  unsigned char* storage = prefill::align_swizzle_storage(raw);
  const uint32_t storage_address = ptx::shared_address(storage);
  const int warp = threadIdx.x >> 5;
  if (threadIdx.x == 0) {
    for (int stage = 0; stage < T::stages; ++stage) {
      ptx::mbarrier_init(&full[stage], T::producer_threads);
      ptx::mbarrier_init(&empty[stage], T::math_warps * 32);
    }
    ptx::fence_mbarrier_init();
  }
  __syncthreads();
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t live = *live_rows;

  if (warp >= T::math_warps) {
    ptx::setmaxnreg_dec<40>();
    EncoderProducer<N, K, Swiglu> producer{activation, weight, full, empty,
        storage_address, live, threadIdx.x - T::math_warps * 32u, 0};
    encoder_for_each_tile<T::feature_tiles>(live, producer);
    producer.finish();
    return;
  }

  ptx::setmaxnreg_inc<232>();
  EncoderMath<N, K, Epilogue> math{bias, output, full, empty,
      storage_address, live, 0};
  encoder_for_each_tile<T::feature_tiles>(live, math);
}

} // namespace chop::gemm
