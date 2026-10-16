#pragma once

// Encoder GEMM over F32 activations and row-major BF16 weights, for towers
// that keep F32 activations (`linear.pairs.v1`). The activation tile is
// staged in F32; the math warps split each value into its BF16 pair
// (common/bf16_pairs.cuh) as they load it, and both halves multiply the same
// weight fragment with F32 accumulation, so the product carries 16 bits of
// the activation instead of 8. The epilogue functor (cuda/encoder/
// linear_pairs.cuh) states the output type, F32 or BF16, whether a BF16 bias
// is added and whether the F32 output adds in place to the rows it
// overwrites (the residual stream), and maps each biased accumulator to its
// output value. A new epilogue is a new functor, never an edit here.
//
// Tiles, the persistent schedule and the tail split are encoder_gemm.cuh's:
// 128 tokens (or a 64- or 32-token tail slice) by 128 weight rows. A stage
// holds 64 K of both operands, 32 KiB of F32 activations and 16 KiB of
// swizzled weights, and two stages fit the 99 KiB budget. Activation rows
// are 256 bytes; their 32-byte column groups are XOR-swizzled by the row so
// the math warps' 8-byte fragment loads are conflict free.

#include "encoder_gemm.cuh"
#include "../common/bf16_pairs.cuh"

namespace chop::gemm {

template<int N, int K>
struct PairsGemmTraits {
  static_assert(N % 128 == 0 && K % 64 == 0);
  static constexpr int math_warps = 8;
  static constexpr int producer_threads = 128;
  static constexpr int threads = math_warps * 32 + producer_threads;
  static constexpr int activation_bytes = 128 * prefill::stage_k * 4;
  static constexpr int weight_bytes = 128 * prefill::swizzle_row_bytes;
  static constexpr int stage_bytes = activation_bytes + weight_bytes;
  static constexpr int stages = 2;
  static constexpr int shared_bytes = stages * stage_bytes + 1024;
  static constexpr int feature_tiles = N / 128;
  static constexpr int k_stages = K / prefill::stage_k;
};

// Byte offset of 16-byte chunk `c` (F32 columns 4c..4c+3) of activation row
// `r` in a stage.
__device__ __forceinline__ uint32_t pairs_activation_offset(uint32_t r, uint32_t c) {
  return r * (prefill::stage_k * 4) + (((c >> 1) ^ (r & 7)) << 5) + ((c & 1) << 4);
}

template<int N, int K>
struct PairsProducer {
  using T = PairsGemmTraits<N, K>;
  const float* activation;
  const uint16_t* weight;
  uint64_t* full;
  uint64_t* empty;
  uint32_t storage_address;
  uint32_t live;
  uint32_t producer;
  uint32_t iteration;

  // Each thread copies Tokens / 8 of the stage's activation chunks (16 per
  // row) and 8 of its 1024 weight chunks.
  template<int Tokens>
  __device__ __forceinline__ void run(uint32_t feature_tile, uint32_t token0) {
#pragma unroll 1
    for (int stage = 0; stage < T::k_stages; ++stage, ++iteration) {
      const uint32_t slot = iteration % T::stages;
      prefill::wait_empty(empty, iteration, T::stages);
      const uint32_t base = storage_address + slot * T::stage_bytes;
#pragma unroll 2
      for (int i = 0; i < Tokens / 8; ++i) {
        const uint32_t chunk = producer + i * T::producer_threads;
        const uint32_t r = chunk >> 4;
        const uint32_t c = chunk & 15;
        const uint32_t token = token0 + r;
        const bool valid = token < live;
        cp_async_zfill16(base + pairs_activation_offset(r, c),
            activation + uint64_t(valid ? token : 0) * K +
                stage * prefill::stage_k + c * 4,
            valid);
      }
#pragma unroll 2
      for (int i = 0; i < 8; ++i) {
        const uint32_t chunk = producer + i * T::producer_threads;
        const uint32_t r = chunk >> 3;
        const uint32_t c = chunk & 7;
        cp_async_zfill16(base + T::activation_bytes +
                r * prefill::swizzle_row_bytes + ((c ^ (r & 7)) << 4),
            weight + uint64_t(feature_tile * 128 + r) * K +
                stage * prefill::stage_k + c * 8,
            true);
      }
      cp_async_arrive_noinc(&full[slot]);
    }
  }

  __device__ __forceinline__ void finish() {
    asm volatile("cp.async.wait_all;" ::: "memory");
  }
};

// The BF16 pairs of a B fragment (eight tokens by 16 K) at `step` of a
// stage's F32 activation tile: K columns step * 16 + 2q, + 1 and + 8, + 9 of
// token row `row`, with q = lane % 4.
__device__ __forceinline__ void load_b_pairs(
    uint32_t tile, uint32_t row, int step, ptx::B2& hi, ptx::B2& lo) {
  const uint32_t q = threadIdx.x & 3;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const uint32_t chunk = step * 4 + (q >> 1) + h * 2;
    float x, y;
    asm volatile("ld.shared.v2.f32 {%0,%1}, [%2];" : "=f"(x), "=f"(y)
        : "r"(tile + pairs_activation_offset(row, chunk) + ((q & 1) << 3)));
    const pairs::Packed packed = pairs::split(x, y);
    hi.v[h] = packed.hi;
    lo.v[h] = packed.lo;
  }
}

template<int N, int K, class Epilogue>
struct PairsMath {
  using T = PairsGemmTraits<N, K>;
  using Out = typename Epilogue::Out;
  static_assert(!Epilogue::residual || sizeof(Out) == 4, "the residual stream is F32");
  const uint16_t* bias;
  Out* output;
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
    const uint32_t b_row = warp_t * warp_tokens + (lane >> 2);
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
      const uint32_t weight_tile = activation_tile + T::activation_bytes;
      if (active) {
#pragma unroll
        for (int step = 0; step < 4; ++step) {
          ptx::A4 a[4];
#pragma unroll
          for (int i = 0; i < 4; ++i)
            a[i] = load_a_swizzled(weight_tile, (warp_f * 4 + i) * 16, step * 2);
#pragma unroll
          for (int j = 0; j < n8; ++j) {
            ptx::B2 hi, lo;
            load_b_pairs(activation_tile, b_row + j * 8, step, hi, lo);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              ptx::mma_bf16(acc[i][j], a[i], hi);
              ptx::mma_bf16(acc[i][j], a[i], lo);
            }
          }
        }
      }
      ptx::mbarrier_arrive(&empty[slot]);
    }
    if (!active) return;

    // Epilogue: bias, activation, then pairs of adjacent features per token.
    const uint32_t group = lane >> 2;
    const bool even = (group & 1) == 0;
    const uint32_t token = token0 + warp_t * warp_tokens + (lane & 3) * 2;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint32_t feature = feature_tile * 128 + (warp_f * 4 + i) * 16 + group;
      float bias_value[2] = {0.f, 0.f};
      if constexpr (Epilogue::bias) {
#pragma unroll
        for (int h = 0; h < 2; ++h)
          bias_value[h] = chop::row::bf(bias[feature + h * 8]);
      }
#pragma unroll
      for (int j = 0; j < n8; ++j) {
        float c[4];
#pragma unroll
        for (int v = 0; v < 4; ++v)
          c[v] = Epilogue::apply(acc[i][j].v[v] + bias_value[v / 2]);
        // Lanes swap with their g^1 neighbour so each store covers two
        // adjacent features of one token.
        const float x0 = __shfl_xor_sync(0xffffffff, even ? c[1] : c[0], 4);
        const float x1 = __shfl_xor_sync(0xffffffff, even ? c[3] : c[2], 4);
        const uint32_t row = token + j * 8 + (even ? 0 : 1);
        const uint32_t column = even ? feature : feature - 1;
        if (row >= live) continue;
        const uint64_t offset = uint64_t(row) * N + column;
        float2 first = even ? make_float2(c[0], x0) : make_float2(x0, c[1]);
        float2 second = even ? make_float2(c[2], x1) : make_float2(x1, c[3]);
        if constexpr (sizeof(Out) == 4) {
          float2* out = reinterpret_cast<float2*>(output + offset);
          float2* out8 = reinterpret_cast<float2*>(output + offset + 8);
          if constexpr (Epilogue::residual) {
            const float2 r0 = *out;
            const float2 r8 = *out8;
            first = make_float2(r0.x + first.x, r0.y + first.y);
            second = make_float2(r8.x + second.x, r8.y + second.y);
          }
          *out = first;
          *out8 = second;
        } else {
          *reinterpret_cast<uint32_t*>(output + offset) =
              uint32_t(chop::row::to_bf(first.x)) |
              uint32_t(chop::row::to_bf(first.y)) << 16;
          *reinterpret_cast<uint32_t*>(output + offset + 8) =
              uint32_t(chop::row::to_bf(second.x)) |
              uint32_t(chop::row::to_bf(second.y)) << 16;
        }
      }
    }
  }
};

// `Epilogue::Out` is float or uint16_t (BF16).
template<int N, int K, class Epilogue>
__device__ void encoder_gemm_pairs(const float* activation,
                                   const uint16_t* weight,
                                   const uint16_t* bias,
                                   typename Epilogue::Out* output,
                                   const uint32_t* live_rows) {
  using T = PairsGemmTraits<N, K>;
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
    PairsProducer<N, K> producer{activation, weight, full, empty,
        storage_address, live, threadIdx.x - T::math_warps * 32u, 0};
    encoder_for_each_tile<T::feature_tiles>(live, producer);
    producer.finish();
    return;
  }

  ptx::setmaxnreg_inc<232>();
  PairsMath<N, K, Epilogue> math{bias, output, full, empty,
      storage_address, live, 0};
  encoder_for_each_tile<T::feature_tiles>(live, math);
}

} // namespace chop::gemm
