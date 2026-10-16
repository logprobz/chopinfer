#pragma once

#include "dense.cuh"

namespace chop::gemm {

template<WeightKind W, bool Decode, int TileM, int TileN, int TileK,
         int Stages = 2>
struct Fp8Traits {
  static constexpr int tile_k = TileK;
  static constexpr int math_warps = 8;
  static constexpr int subtiles = (TileM / 16) * (TileN / 8);
  static constexpr int per_warp =
      (subtiles + math_warps - 1) / math_warps < 4 ?
      (subtiles + math_warps - 1) / math_warps : 4;
  static constexpr int waves =
      (subtiles + math_warps * per_warp - 1) / (math_warps * per_warp);
  static constexpr int a_row_bytes =
      Decode && W == WeightKind::mxfp4 ? TileK / 2 : TileK;
  static constexpr int b_row_bytes =
      !Decode && W == WeightKind::mxfp4 ? TileK / 2 : TileK;
  static constexpr int shared_bytes =
      Stages * (TileM * a_row_bytes + TileN * b_row_bytes);
};

template<int RowBytes, bool Swizzled = true>
__device__ __forceinline__ uint32_t packed_fp8(
    const uint8_t* base, int row, int k) {
  uint32_t result = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t offset = Swizzled ? smem::swizzle_128(
        base, row * RowBytes + k + i, RowBytes) : row * RowBytes + k + i;
    result |= uint32_t(base[offset]) << (i * 8);
  }
  return result;
}

__device__ __forceinline__ void mma_tile_fp8(
    ptx::D4& d, const uint8_t* a, const uint8_t* b,
    int m, int n, int k) {
  const int group = (threadIdx.x & 31) >> 2;
  const int lane4 = threadIdx.x & 3;
  ptx::A4 af{};
  ptx::B2 bf{};
#pragma unroll
  for (int i = 0; i < 4; ++i)
    af.v[i] = packed_fp8<128>(a, m + group + (i % 2) * 8,
                         k + (i / 2) * 16 + lane4 * 4);
#pragma unroll
  for (int i = 0; i < 2; ++i)
    bf.v[i] = packed_fp8<128>(b, n + group, k + i * 16 + lane4 * 4);
  ptx::mma_f8f6f4<ptx::Element::e4m3, ptx::Element::e4m3>(d, af, bf);
}

// The block-scale instruction selects A scales from lanes 0/1 in each quad
// and B scales from lane 0. One MMA spans one 32-K MXFP8 scale block.
template<bool Decode>
__device__ __forceinline__ void mma_tile_mxfp8(
    ptx::D4& d, const uint8_t* a, const uint8_t* b, int m, int n, int k,
    uint32_t row_a, uint32_t row_b, uint32_t k_block, uint32_t k_blocks,
    uint32_t live_m, uint32_t output_n, const uint8_t* weight_scales,
    const uint8_t* activation_scales) {
  const int group = (threadIdx.x & 31) >> 2;
  const int lane4 = threadIdx.x & 3;
  ptx::A4 af{};
  ptx::B2 bf{};
#pragma unroll
  for (int i = 0; i < 4; ++i)
    af.v[i] = packed_fp8<128>(a, m + group + (i % 2) * 8,
                         k + (i / 2) * 16 + lane4 * 4);
#pragma unroll
  for (int i = 0; i < 2; ++i)
    bf.v[i] = packed_fp8<128>(b, n + group, k + i * 16 + lane4 * 4);
  const uint8_t* scale_a = Decode ? weight_scales : activation_scales;
  const uint8_t* scale_b = Decode ? activation_scales : weight_scales;
  const uint32_t max_a = Decode ? output_n : live_m;
  const uint32_t max_b = Decode ? live_m : output_n;
  const uint32_t scale_row_a = row_a + m + group + (lane4 & 1) * 8;
  const uint32_t scale_row_b = row_b + n + group;
  const uint32_t sa = scale_row_a < max_a && k_block < k_blocks ?
      scale_a[uint64_t(scale_row_a) * k_blocks + k_block] : 0;
  const uint32_t sb = scale_row_b < max_b && k_block < k_blocks ?
      scale_b[uint64_t(scale_row_b) * k_blocks + k_block] : 0;
  ptx::mma_mxf8f6f4<ptx::Element::e4m3, ptx::Element::e4m3>(d, af, bf,
                                                                sa, sb);
}

template<int RowBytes>
__device__ __forceinline__ uint32_t packed_fp4_as_byte4(
    const uint8_t* base, int row, int k) {
  uint32_t result = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int element = k + i;
    const uint32_t offset = smem::swizzle_128(
        base, row * RowBytes + element / 2, RowBytes);
    const uint8_t code = (base[offset] >> ((element & 1) * 4)) & 15;
    // f8f6f4 MMA stores each E2M1 value in bits [5:2] of its byte.
    result |= uint32_t(code << 2) << (i * 8);
  }
  return result;
}

template<bool Decode, int TileK>
__device__ __forceinline__ void mma_tile_mxfp4_a8(
    ptx::D4& d, const uint8_t* a, const uint8_t* b, int m, int n, int k,
    uint32_t row_a, uint32_t row_b, uint32_t k_block, uint32_t k_blocks,
    uint32_t live_m, uint32_t output_n, const uint8_t* weight_scales,
    const uint8_t* activation_scales) {
  const int group = (threadIdx.x & 31) >> 2;
  const int lane4 = threadIdx.x & 3;
  ptx::A4 af{};
  ptx::B2 bf{};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int row = m + group + (i % 2) * 8;
    const int col = k + (i / 2) * 16 + lane4 * 4;
    if constexpr (Decode)
      af.v[i] = packed_fp4_as_byte4<TileK / 2>(a, row, col);
    else
      af.v[i] = packed_fp8<TileK, false>(a, row, col);
  }
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int row = n + group;
    const int col = k + i * 16 + lane4 * 4;
    if constexpr (Decode)
      bf.v[i] = packed_fp8<TileK, false>(b, row, col);
    else
      bf.v[i] = packed_fp4_as_byte4<TileK / 2>(b, row, col);
  }
  const uint8_t* scale_a = Decode ? weight_scales : activation_scales;
  const uint8_t* scale_b = Decode ? activation_scales : weight_scales;
  const uint32_t max_a = Decode ? output_n : live_m;
  const uint32_t max_b = Decode ? live_m : output_n;
  const uint32_t scale_row_a = row_a + m + group + (lane4 & 1) * 8;
  const uint32_t scale_row_b = row_b + n + group;
  const uint32_t sa = scale_row_a < max_a && k_block < k_blocks ?
      scale_a[uint64_t(scale_row_a) * k_blocks + k_block] : 0;
  const uint32_t sb = scale_row_b < max_b && k_block < k_blocks ?
      scale_b[uint64_t(scale_row_b) * k_blocks + k_block] : 0;
  if constexpr (Decode)
    ptx::mma_mxf8f6f4<ptx::Element::e2m1, ptx::Element::e4m3>(
        d, af, bf, sa, sb);
  else
    ptx::mma_mxf8f6f4<ptx::Element::e4m3, ptx::Element::e2m1>(
        d, af, bf, sa, sb);
}

template<WeightKind W, bool Decode>
__device__ __forceinline__ void promote_fp8(
    ptx::D4& accum, const ptx::D4& raw, int subtile,
    uint32_t row_a, uint32_t row_b, uint32_t k_block, uint32_t k_blocks,
    const float* activation_scales, const float* weight_scales,
    uint32_t live_m, uint32_t n, int tile_n) {
  const int group = (threadIdx.x & 31) >> 2;
  const int lane4 = threadIdx.x & 3;
  const int sm = (subtile / (tile_n / 8)) * 16;
  const int sn = (subtile % (tile_n / 8)) * 8;
#pragma unroll
  for (int v = 0; v < 4; ++v) {
    const uint32_t ma = row_a + sm + group + (v / 2) * 8;
    const uint32_t nb = row_b + sn + lane4 * 2 + (v % 2);
    const uint32_t token = Decode ? nb : ma;
    const uint32_t column = Decode ? ma : nb;
    if (token >= live_m || column >= n) continue;
    const float a_scale = activation_scales[uint64_t(token) * k_blocks + k_block];
    float w_scale;
    if constexpr (W == WeightKind::fp8_tensor) w_scale = weight_scales[0];
    else if constexpr (W == WeightKind::fp8_channel) w_scale = weight_scales[column];
    else w_scale = weight_scales[uint64_t(column / 128) * k_blocks + k_block];
    const float scaled = raw.v[v] * (a_scale * w_scale);
    accum.v[v] += scaled;
  }
}

template<WeightKind W, bool Decode, bool Residual, int TileM, int TileN,
         int TileK,
         int Stages = 2>
__device__ void run_fp8_a8(
    const CUtensorMap& map_a, const CUtensorMap& map_b,
    void* output, float* partial, uint32_t* semaphores,
    uint32_t live_m, uint32_t n, uint32_t k, uint32_t splits,
    uint32_t output_f32, const float* weight_scales,
    const float* activation_scales, const void* residual) {
  using T = Fp8Traits<W, Decode, TileM, TileN, TileK, Stages>;
  static_assert(TileM % 16 == 0 && TileN % 8 == 0);
  static_assert(T::shared_bytes + 1152 <= 101376);
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
    ptx::D4 acc0{}, acc1{}, acc2{}, acc3{};
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
#define CHOP_A8_CONSUME(I, ACC)                                                \
        if constexpr (T::per_warp > I) {                                       \
          const int subtile = wave * T::math_warps * T::per_warp +             \
                              warp + I * T::math_warps;                        \
          if (subtile < T::subtiles) {                                          \
            const int sm = (subtile / (TileN / 8)) * 16;                        \
            const int sn = (subtile % (TileN / 8)) * 8;                         \
            if constexpr (W == WeightKind::mxfp8) {                           \
              for (int kk = 0; kk < TileK; kk += 32)                          \
                mma_tile_mxfp8<Decode>(ACC, &a[stage][0][0],                  \
                    &b[stage][0][0], sm, sn, kk, row_a, row_b,                \
                    chunk * (TileK / 32) + kk / 32, k / 32, live_m, n,        \
                    reinterpret_cast<const uint8_t*>(weight_scales),         \
                    reinterpret_cast<const uint8_t*>(activation_scales));     \
            } else if constexpr (W == WeightKind::mxfp4) {                    \
              for (int kk = 0; kk < TileK; kk += 32)                          \
                mma_tile_mxfp4_a8<Decode, TileK>(ACC, &a[stage][0][0],       \
                    &b[stage][0][0], sm, sn, kk, row_a, row_b,                \
                    chunk * (TileK / 32) + kk / 32, k / 32, live_m, n,        \
                    reinterpret_cast<const uint8_t*>(weight_scales),         \
                    reinterpret_cast<const uint8_t*>(activation_scales));     \
            } else {                                                            \
              ptx::D4 raw{};                                                   \
              for (int kk = 0; kk < TileK; kk += 32)                          \
                mma_tile_fp8(raw, &a[stage][0][0], &b[stage][0][0], sm, sn, kk);\
              promote_fp8<W, Decode>(ACC, raw, subtile, row_a, row_b, chunk,  \
                  chunks, activation_scales, weight_scales, live_m, n, TileN); \
            }                                                                   \
          }                                                                     \
        }
        CHOP_A8_CONSUME(0, acc0)
        CHOP_A8_CONSUME(1, acc1)
        CHOP_A8_CONSUME(2, acc2)
        CHOP_A8_CONSUME(3, acc3)
#undef CHOP_A8_CONSUME
        if ((threadIdx.x & 31) == 0) pipe.consumer_release(step);
      }
    }
    if (warp >= 0) {
      const int group = (threadIdx.x & 31) >> 2;
      const int lane4 = threadIdx.x & 3;
#define CHOP_A8_WRITE(I, ACC)                                                  \
      if constexpr (T::per_warp > I) {                                         \
        const int subtile = wave * T::math_warps * T::per_warp +               \
                            warp + I * T::math_warps;                          \
        if (subtile < T::subtiles) {                                            \
          const uint32_t sm = (subtile / (TileN / 8)) * 16;                     \
          const uint32_t sn = (subtile % (TileN / 8)) * 8;                      \
          _Pragma("unroll")                                                     \
          for (int v = 0; v < 4; ++v) {                                         \
            const uint32_t ma = row_a + sm + group + (v / 2) * 8;              \
            const uint32_t nb = row_b + sn + lane4 * 2 + (v % 2);             \
            const uint32_t token = Decode ? nb : ma;                           \
            const uint32_t column = Decode ? ma : nb;                          \
            if (token < live_m && column < n) {                                 \
              float value = ACC.v[v];                                         \
              if (splits > 1) {                                                 \
                const uint64_t offset = (uint64_t(tile) * splits + split) *    \
                    tile_elements + (sm + group + (v / 2) * 8) * TileN +        \
                    sn + lane4 * 2 + (v % 2);                                  \
                partial[offset] = value;                                       \
              } else {                                                          \
                const uint64_t offset = uint64_t(token) * n + column;         \
                if constexpr (Residual) value += residual_value(             \
                    residual, offset, output_f32);                            \
                if (output_f32) static_cast<float*>(output)[offset] = value;  \
                else static_cast<__nv_bfloat16*>(output)[offset] =            \
                    __float2bfloat16(value);                                    \
              }                                                                 \
            }                                                                   \
          }                                                                     \
        }                                                                       \
      }
      CHOP_A8_WRITE(0, acc0)
      CHOP_A8_WRITE(1, acc1)
      CHOP_A8_WRITE(2, acc2)
      CHOP_A8_WRITE(3, acc3)
#undef CHOP_A8_WRITE
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
