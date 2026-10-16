#pragma once

#include <cuda_bf16.h>
#include <stdint.h>
#include "../common/pipeline.cuh"
#include "../common/ptx_tma.cuh"
#include "../common/smem.cuh"
#include "../common/convert.cuh"

namespace chop::gemm {

__device__ __forceinline__ unsigned char* align_tma_storage(
    unsigned char* storage) {
  return reinterpret_cast<unsigned char*>(
      (reinterpret_cast<uintptr_t>(storage) + 127) & ~uintptr_t(127));
}

// Both operands are row-major in K. Decode exchanges their MMA roles:
// weight rows become M and token rows become N.
enum class WeightKind {
  bf16, fp8_tensor, fp8_channel, fp8_block128, mxfp8, mxfp4, nvfp4,
  // FP8 with one F32 scale per 64 weight rows and 128 K values: regrouped
  // rows (MiMo's QKV at TP2 and TP8) whose 128-row blocks do not align.
  fp8_row64
};

// Fragment kernels scale each 16-row strip by one F32 per 128-K block; this
// many strips share a scale row.
template<WeightKind W>
constexpr bool fragment_fp8 =
    W == WeightKind::fp8_block128 || W == WeightKind::fp8_row64;
template<WeightKind W>
constexpr int fragment_scale_strips = W == WeightKind::fp8_row64 ? 4 : 8;

__device__ __forceinline__ float ue4m3_scale(uint8_t code) {
  const int exponent = code >> 3;
  const int mantissa = code & 7;
  if (exponent == 0) return mantissa * (1.0f / 512.0f);
  return (1.0f + mantissa * (1.0f / 8.0f)) *
         ldexpf(1.0f, exponent - 7);
}

__device__ __forceinline__ float residual_value(
    const void* residual, uint64_t offset, uint32_t output_f32) {
  if (output_f32) return static_cast<const float*>(residual)[offset];
  return __bfloat162float(static_cast<const __nv_bfloat16*>(residual)[offset]);
}

template<WeightKind W, bool Decode, int TileM, int TileN, int TileK,
         int Stages = 2>
struct A16Traits {
  static_assert(TileM % 16 == 0 && TileN % 8 == 0 && TileK % 16 == 0);
  static_assert(W == WeightKind::bf16 ||
      ((W == WeightKind::mxfp4 || W == WeightKind::nvfp4)
          ? TileK == 256 : TileK == 128));
  static constexpr int math_warps = 8;
  static constexpr int subtiles = (TileM / 16) * (TileN / 8);
  static constexpr int per_warp =
      (subtiles + math_warps - 1) / math_warps < (Decode ? 8 : 4) ?
      (subtiles + math_warps - 1) / math_warps : (Decode ? 8 : 4);
  static constexpr int waves =
      (subtiles + math_warps * per_warp - 1) / (math_warps * per_warp);
  static constexpr int weight_row_bytes =
      W == WeightKind::bf16 ? TileK * 2 :
      (W == WeightKind::mxfp4 || W == WeightKind::nvfp4) ? TileK / 2 : TileK;
  static constexpr int a_row_bytes = Decode ? weight_row_bytes : TileK * 2;
  static constexpr int b_row_bytes = Decode ? TileK * 2 : weight_row_bytes;
  static constexpr int shared_bytes =
      Stages * (TileM * a_row_bytes + TileN * b_row_bytes);
};

template<int TileK, bool Swizzled = true>
__device__ __forceinline__ uint32_t packed_bf16(
    const uint16_t* base, int row, int k) {
  if constexpr (!Swizzled)
    return uint32_t(base[row * TileK + k]) |
        (uint32_t(base[row * TileK + k + 1]) << 16);
  const uint32_t off = smem::swizzle_128(base, row * TileK * 2 + k * 2,
                                        TileK * 2) / 2;
  const uint32_t hi = smem::swizzle_128(base, row * TileK * 2 + (k + 1) * 2,
                                       TileK * 2) / 2;
  return uint32_t(base[off]) | (uint32_t(base[hi]) << 16);
}

template<int TileK, WeightKind W, bool PromoteBlock = true>
__device__ __forceinline__ uint32_t packed_fp8_weight(
    const uint8_t* base, int local_row, int k, int global_row,
    int global_k, int total_n, int total_k, const float* scales) {
  if (global_row >= total_n || global_k >= total_k) return 0;
  const uint32_t lo = smem::swizzle_128(base, local_row * TileK + k,
                                       TileK);
  const uint32_t hi = smem::swizzle_128(base, local_row * TileK + k + 1,
                                       TileK);
  const uint16_t raw = uint16_t(base[lo]) | (uint16_t(base[hi]) << 8);
  if constexpr (W == WeightKind::fp8_block128 && PromoteBlock)
    return convert::unpack_bf16x2_e4m3(raw);
  const convert::Float2 values = convert::bf16x2_to_f32(
      convert::unpack_bf16x2_e4m3(raw));
  float scale;
  if constexpr (W == WeightKind::fp8_tensor) scale = scales[0];
  else if constexpr (W == WeightKind::fp8_channel) scale = scales[global_row];
  else if constexpr (W == WeightKind::mxfp8) {
    const uint8_t code = reinterpret_cast<const uint8_t*>(scales)[
        uint64_t(global_row) * (total_k / 32) + global_k / 32];
    scale = ldexpf(1.0f, int(code) - 127);
  } else {
    const int scale_k = (total_k + 127) / 128;
    scale = scales[(global_row / 128) * scale_k + global_k / 128];
  }
  const convert::Float2 products{values.lo * scale, values.hi * scale};
  const uint32_t packed = convert::bf16x2_from_f32(products);
  if constexpr (W == WeightKind::mxfp8) {
    // Cached images can predate exact-operand admission. Refuse any lossy
    // conversion, including BF16 subnormal rounding, instead of using it.
    if (scale < 0x1p-124f || scale > 0x1p119f) {
      const auto decoded = convert::bf16x2_to_f32(packed);
      if (!isfinite(products.lo) || !isfinite(products.hi) ||
          decoded.lo != products.lo || decoded.hi != products.hi)
        asm volatile("trap;");
    }
  }
  return packed;
}

template<int TileK, WeightKind W>
__device__ __forceinline__ uint32_t packed_fp4_weight(
    const uint8_t* base, int local_row, int k, int global_row,
    int global_k, int total_n, int total_k, const float* scales,
    const float* global_scale) {
  static_assert(W == WeightKind::mxfp4 || W == WeightKind::nvfp4);
  if (global_row >= total_n || global_k >= total_k) return 0;
  const uint32_t off = smem::swizzle_128(
      base, local_row * (TileK / 2) + k / 2, TileK / 2);
  const convert::Float2 values = convert::unpack_f32_e2m1(base[off]);
  const uint8_t* codes = reinterpret_cast<const uint8_t*>(scales);
  float scale;
  if constexpr (W == WeightKind::mxfp4) {
    const uint8_t code =
        codes[uint64_t(global_row) * (total_k / 32) + global_k / 32];
    scale = ldexpf(1.0f, int(code) - 127);
  } else {
    const uint8_t code =
        codes[uint64_t(global_row) * (total_k / 16) + global_k / 16];
    scale = ue4m3_scale(code);
  }
  return convert::bf16x2_from_f32({values.lo * scale, values.hi * scale});
}

template<WeightKind W, bool Decode, int TileK, int WeightRowStride = 1>
__device__ __forceinline__ void mma_tile(ptx::D4& d,
    const uint8_t* a, const uint8_t* b, int m, int n, int k,
    int row_a, int row_b, int chunk_k, int total_n, int total_k,
    const float* scales, const float* global_scale = nullptr) {
  const int group = (threadIdx.x & 31) >> 2;
  const int lane4 = threadIdx.x & 3;
  ptx::A4 af{};
  ptx::B2 bf{};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int local_row = m + group + (i % 2) * 8;
    const int local_k = k + (i / 2) * 8 + lane4 * 2;
    if constexpr (Decode && (W == WeightKind::mxfp4 || W == WeightKind::nvfp4))
      af.v[i] = packed_fp4_weight<TileK, W>(a, local_row, local_k,
          row_a + local_row * WeightRowStride, chunk_k + local_k,
          total_n, total_k, scales, global_scale);
    else if constexpr (Decode && W != WeightKind::bf16)
      af.v[i] = packed_fp8_weight<TileK, W, WeightRowStride == 1>(a, local_row, local_k,
          row_a + local_row * WeightRowStride, chunk_k + local_k,
          total_n, total_k, scales);
    else
      af.v[i] = packed_bf16<TileK, W == WeightKind::bf16>(
                                  reinterpret_cast<const uint16_t*>(a),
                                  local_row, local_k);
  }
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int local_row = n + group;
    const int local_k = k + i * 8 + lane4 * 2;
    if constexpr (!Decode && (W == WeightKind::mxfp4 || W == WeightKind::nvfp4))
      bf.v[i] = packed_fp4_weight<TileK, W>(b, local_row, local_k,
          row_b + local_row * WeightRowStride, chunk_k + local_k,
          total_n, total_k, scales, global_scale);
    else if constexpr (!Decode && W != WeightKind::bf16)
      bf.v[i] = packed_fp8_weight<TileK, W, WeightRowStride == 1>(b, local_row, local_k,
          row_b + local_row * WeightRowStride, chunk_k + local_k,
          total_n, total_k, scales);
    else
      bf.v[i] = packed_bf16<TileK, W == WeightKind::bf16>(
                                  reinterpret_cast<const uint16_t*>(b),
                                  local_row, local_k);
  }
  ptx::mma_bf16(d, af, bf);
}

// A block128 scale is constant over this MMA tile and its 128-K chunk.
// Accumulate raw FP8-converted fragments first, then promote four results.
template<WeightKind W, bool Decode, int TileK, int WeightRowStride = 1>
__device__ __forceinline__ void mma_chunk(ptx::D4& accum,
    const uint8_t* a, const uint8_t* b, int sm, int sn,
    int row_a, int row_b, int chunk, int total_n, int total_k,
    const float* scales, const float* global_scale = nullptr) {
  if constexpr (W == WeightKind::fp8_block128 && WeightRowStride == 1) {
    ptx::D4 raw{};
    for (int kk = 0; kk < TileK; kk += 16)
      mma_tile<W, Decode, TileK, WeightRowStride>(raw, a, b, sm, sn, kk,
          row_a, row_b, chunk * TileK, total_n, total_k, scales, global_scale);
    const int weight_row = Decode ? row_a + sm * WeightRowStride :
                                    row_b + sn * WeightRowStride;
    const float scale = weight_row < total_n ?
        scales[(weight_row / 128) * ((total_k + 127) / 128) + chunk] : 0.0f;
#pragma unroll
    for (int v = 0; v < 4; ++v) accum.v[v] += raw.v[v] * scale;
  } else {
    // Keep each MMA chain within 128 K, including 256-K FP4 staging tiles.
    // The tensor-global NVFP4 scale remains in the final F32 epilogue.
    for (int block = 0; block < TileK; block += 128) {
      ptx::D4 raw{};
      for (int kk = block; kk < block + 128 && kk < TileK; kk += 16)
        mma_tile<W, Decode, TileK, WeightRowStride>(raw, a, b, sm, sn, kk,
            row_a, row_b, chunk * TileK, total_n, total_k, scales, global_scale);
#pragma unroll
      for (int v = 0; v < 4; ++v) accum.v[v] += raw.v[v];
    }
  }
}

template<WeightKind W, bool Decode, bool Residual, int TileM, int TileN, int TileK,
         int Stages = 2>
__device__ void run_a16(
    const CUtensorMap& map_a, const CUtensorMap& map_b,
    void* output, float* partial, uint32_t* semaphores,
    uint32_t live_m, uint32_t n, uint32_t k, uint32_t splits,
    uint32_t output_f32, const float* scales, const void* residual,
    const float* weight_global_scale) {
  using T = A16Traits<W, Decode, TileM, TileN, TileK, Stages>;
  static_assert(T::shared_bytes <= 101376);
  __shared__ TmaPipeline<Stages, T::math_warps> pipe;
  extern __shared__ __align__(128) unsigned char storage[];
  auto* aligned = align_tma_storage(storage);
  auto* a = reinterpret_cast<uint8_t (*)[TileM][T::a_row_bytes]>(aligned);
  auto* b = reinterpret_cast<uint8_t (*)[TileN][T::b_row_bytes]>(
      aligned + Stages * TileM * T::a_row_bytes);
  __shared__ uint32_t last;
  const int warp = int(threadIdx.x / 32) - 4;
  const uint32_t rows_a = Decode ? n : live_m;
  const uint32_t rows_b = Decode ? live_m : n;
  const uint32_t tiles_a = (rows_a + TileM - 1) / TileM;
  const uint32_t tiles_b = (rows_b + TileN - 1) / TileN;
  const uint64_t tile_count = uint64_t(tiles_a) * tiles_b;
  const uint32_t initial_tile = blockIdx.x / splits;
  ptx::griddep_launch_dependents();
  if (threadIdx.x == 0 && initial_tile < tile_count) {
    const int32_t weight_coord[2] = {
        0, int32_t(Decode ? (initial_tile / tiles_b) * TileM :
                              (initial_tile % tiles_b) * TileN)};
    ptx::prefetch_tensor_l2(Decode ? &map_a : &map_b, weight_coord);
  }
  ptx::griddep_wait();
  if (warp < 0) configure_producer_warp();
  else configure_math_warp();
  for (uint64_t work = blockIdx.x; work < tile_count * splits;
       work += gridDim.x) {
  const uint32_t tile = uint32_t(work / splits);
  const uint32_t split = uint32_t(work % splits);
  if (threadIdx.x == 0) pipe.init();
  __syncthreads();
  if (warp >= 0 && (threadIdx.x & 31) == 0) pipe.prime_empty();
  const uint32_t row_a = (tile / tiles_b) * TileM;
  const uint32_t row_b = (tile % tiles_b) * TileN;
  const uint32_t chunks = (k + TileK - 1) / TileK;
  const uint32_t first = (uint64_t(chunks) * split) / splits;
  const uint32_t end = (uint64_t(chunks) * (split + 1)) / splits;
  const uint64_t policy = ptx::l2_policy<ptx::L2Evict::first>();
  const uint64_t tile_elements = uint64_t(TileM) * TileN;
  for (int wave = 0; wave < T::waves; ++wave) {
  ptx::D4 acc0{}, acc1{}, acc2{}, acc3{}, acc4{}, acc5{}, acc6{}, acc7{}, acc8{}, acc9{}, acc10{}, acc11{}, acc12{}, acc13{}, acc14{}, acc15{};
  for (uint32_t chunk = first; chunk < end; ++chunk) {
    const uint32_t step = wave * (end - first) + chunk - first;
    const int stage = step % Stages;
    if (threadIdx.x == 0) {
      pipe.producer_acquire(step);
      uint64_t* bar = pipe.producer_commit(step, T::shared_bytes / Stages);
      const int32_t ca[2] = {int32_t(chunk * TileK), int32_t(row_a)};
      const int32_t cb[2] = {int32_t(chunk * TileK), int32_t(row_b)};
      ptx::tma_load_hint<2>(a[stage], &map_a, ca, bar, policy);
      ptx::tma_load_hint<2>(b[stage], &map_b, cb, bar, policy);
    }
    if (warp >= 0) {
      pipe.consumer_wait(step);
#define CHOP_CONSUME(I, ACC)                                                   \
      if constexpr (T::per_warp > I) {                                         \
        const int subtile = wave * T::math_warps * T::per_warp +               \
                            warp + I * T::math_warps;                          \
        if (subtile < T::subtiles) {                                            \
          const int sm = (subtile / (TileN / 8)) * 16;                           \
          const int sn = (subtile % (TileN / 8)) * 8;                            \
          mma_chunk<W, Decode, TileK>(ACC, &a[stage][0][0],                   \
              &b[stage][0][0], sm, sn, row_a, row_b, chunk, n, k,             \
              scales, weight_global_scale);                                     \
        }                                                                       \
      }
      CHOP_CONSUME(0, acc0)
      CHOP_CONSUME(1, acc1)
      CHOP_CONSUME(2, acc2)
      CHOP_CONSUME(3, acc3)
      CHOP_CONSUME(4, acc4)
      CHOP_CONSUME(5, acc5)
      CHOP_CONSUME(6, acc6)
      CHOP_CONSUME(7, acc7)
      CHOP_CONSUME(8, acc8)
      CHOP_CONSUME(9, acc9)
      CHOP_CONSUME(10, acc10)
      CHOP_CONSUME(11, acc11)
      CHOP_CONSUME(12, acc12)
      CHOP_CONSUME(13, acc13)
      CHOP_CONSUME(14, acc14)
      CHOP_CONSUME(15, acc15)
#undef CHOP_CONSUME
      if ((threadIdx.x & 31) == 0) pipe.consumer_release(step);
    }
  }
  if (warp >= 0) {
    const int group = (threadIdx.x & 31) >> 2;
    const int lane4 = threadIdx.x & 3;
#define CHOP_WRITE(I, ACC)                                                     \
    if constexpr (T::per_warp > I) {                                           \
      const int subtile = wave * T::math_warps * T::per_warp +                 \
                          warp + I * T::math_warps;                            \
      if (subtile < T::subtiles) {                                              \
        const uint32_t sm = (subtile / (TileN / 8)) * 16;                       \
        const uint32_t sn = (subtile % (TileN / 8)) * 8;                        \
        _Pragma("unroll")                                                       \
        for (int v = 0; v < 4; ++v) {                                           \
          const uint32_t ma = row_a + sm + group + (v / 2) * 8;                \
          const uint32_t nb = row_b + sn + lane4 * 2 + (v % 2);               \
          const uint32_t token = Decode ? nb : ma;                             \
          const uint32_t column = Decode ? ma : nb;                            \
          if (token < live_m && column < n) {                                   \
            float value = ACC.v[v];                                           \
            if (splits > 1) {                                                   \
              const uint64_t offset = (uint64_t(tile) * splits + split) *      \
                  tile_elements + (sm + group + (v / 2) * 8) * TileN +          \
                  sn + lane4 * 2 + (v % 2);                                    \
              partial[offset] = value;                                         \
            } else {                                                            \
              if constexpr (W == WeightKind::nvfp4)                         \
                value = __fmul_rn(value, *weight_global_scale);               \
              const uint64_t offset = uint64_t(token) * n + column;           \
              if constexpr (Residual) value += residual_value(               \
                  residual, offset, output_f32);                              \
              if (output_f32) static_cast<float*>(output)[offset] = value;    \
              else static_cast<__nv_bfloat16*>(output)[offset] =              \
                  __float2bfloat16(value);                                      \
            }                                                                   \
          }                                                                     \
        }                                                                       \
      }                                                                         \
    }
    CHOP_WRITE(0, acc0)
    CHOP_WRITE(1, acc1)
    CHOP_WRITE(2, acc2)
    CHOP_WRITE(3, acc3)
    CHOP_WRITE(4, acc4)
    CHOP_WRITE(5, acc5)
    CHOP_WRITE(6, acc6)
    CHOP_WRITE(7, acc7)
    CHOP_WRITE(8, acc8)
    CHOP_WRITE(9, acc9)
    CHOP_WRITE(10, acc10)
    CHOP_WRITE(11, acc11)
    CHOP_WRITE(12, acc12)
    CHOP_WRITE(13, acc13)
    CHOP_WRITE(14, acc14)
    CHOP_WRITE(15, acc15)
#undef CHOP_WRITE
  }
  }
  __syncthreads();
  if (splits > 1) {
    if (threadIdx.x == 0) {
      __threadfence();
      last = atomicAdd(&semaphores[tile], 1) == splits - 1;
    }
    __syncthreads();
    if (last) {
      for (uint32_t index = threadIdx.x; index < tile_elements; index += blockDim.x) {
        const uint32_t ma = row_a + index / TileN;
        const uint32_t nb = row_b + index % TileN;
        const uint32_t token = Decode ? nb : ma;
        const uint32_t column = Decode ? ma : nb;
        if (token < live_m && column < n) {
          float sum = 0;
          for (uint32_t s = 0; s < splits; ++s)
            sum += partial[(uint64_t(tile) * splits + s) * tile_elements + index];
          if constexpr (W == WeightKind::nvfp4)
            sum = __fmul_rn(sum, *weight_global_scale);
          const uint64_t out_offset = uint64_t(token) * n + column;
          if constexpr (Residual)
            sum += residual_value(residual, out_offset, output_f32);
          if (output_f32) static_cast<float*>(output)[out_offset] = sum;
          else static_cast<__nv_bfloat16*>(output)[out_offset] = __float2bfloat16(sum);
        }
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        __threadfence();
        atomicExch(&semaphores[tile], 0);
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) pipe.invalidate();
  __syncthreads();
  }
}

} // namespace chop::gemm
