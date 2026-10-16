#pragma once

#include "dense.cuh"

namespace chop::gemm {

// Each lane reads its complete 16-byte MMA A fragment from a contiguous
// fragment-major weight tile. Weight bytes never visit shared memory.
__device__ __forceinline__ ptx::A4 load_weight_fragment(const uint8_t* address) {
  ptx::A4 result{};
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
      : "=r"(result.v[0]), "=r"(result.v[1]), "=r"(result.v[2]), "=r"(result.v[3])
      : "l"(address) : "memory");
  return result;
}

// Keep four independent vector loads in one PTX block. Otherwise the CUDA
// optimizer compacts adjacent source loads into a serialized address loop.
__device__ __forceinline__ void load_weight_group4(
    const uint8_t* address, ptx::A4& a, ptx::A4& b,
    ptx::A4& c, ptx::A4& d) {
  asm volatile(
      "ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%16];\n\t"
      "ld.global.nc.L1::no_allocate.v4.u32 {%4,%5,%6,%7}, [%16+512];\n\t"
      "ld.global.nc.L1::no_allocate.v4.u32 {%8,%9,%10,%11}, [%16+1024];\n\t"
      "ld.global.nc.L1::no_allocate.v4.u32 {%12,%13,%14,%15}, [%16+1536];"
      : "=r"(a.v[0]), "=r"(a.v[1]), "=r"(a.v[2]), "=r"(a.v[3]),
        "=r"(b.v[0]), "=r"(b.v[1]), "=r"(b.v[2]), "=r"(b.v[3]),
        "=r"(c.v[0]), "=r"(c.v[1]), "=r"(c.v[2]), "=r"(c.v[3]),
        "=r"(d.v[0]), "=r"(d.v[1]), "=r"(d.v[2]), "=r"(d.v[3])
      : "l"(address) : "memory");
}

template<int TileN, int ActivationStride = 128>
__device__ __forceinline__ ptx::B2 activation_fragment(
    const uint16_t* staged, int token_tile, int k_step) {
  const int group = (threadIdx.x & 31) >> 2;
  const int lane4 = threadIdx.x & 3;
  ptx::B2 b{};
  const uint16_t* first = staged + (token_tile * 8 + group) * ActivationStride +
      k_step + lane4 * 2;
  const uint32_t address = ptx::shared_address(first);
  asm volatile("ld.shared.b32 %0, [%1];" : "=r"(b.v[0]) : "r"(address) : "memory");
  asm volatile("ld.shared.b32 %0, [%1+16];" : "=r"(b.v[1]) : "r"(address) : "memory");
  return b;
}

template<WeightKind W, int TileN, int Sub, bool RegisterB = false,
         int ActivationStride = 128>
__device__ __forceinline__ void consume_weight_fragment(
    ptx::A4 packed, const uint16_t* staged,
    ptx::D4& acc0, ptx::D4& acc1, ptx::D4& acc2, ptx::D4& acc3,
    ptx::D4& block0, ptx::D4& block1, ptx::D4& block2, ptx::D4& block3,
    const ptx::B2* cached_b = nullptr) {
  if constexpr (W == WeightKind::bf16) {
#define CHOP_FRAGMENT_MMA(I, ACC)                                            \
    if constexpr (TileN / 8 > I) {                                            \
      const ptx::B2 b = RegisterB ? cached_b[Sub * (TileN / 8) + I] :       \
          activation_fragment<TileN, ActivationStride>(staged, I, Sub * 16); \
      ptx::mma_bf16(ACC, packed, b);                                          \
    }
    CHOP_FRAGMENT_MMA(0, acc0)
    CHOP_FRAGMENT_MMA(1, acc1)
    CHOP_FRAGMENT_MMA(2, acc2)
    CHOP_FRAGMENT_MMA(3, acc3)
#undef CHOP_FRAGMENT_MMA
  } else {
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const uint32_t low = packed.v[half * 2];
      const uint32_t high = packed.v[half * 2 + 1];
      ptx::A4 a{};
      a.v[0] = convert::unpack_bf16x2_e4m3(uint16_t(low));
      a.v[1] = convert::unpack_bf16x2_e4m3(uint16_t(low >> 16));
      a.v[2] = convert::unpack_bf16x2_e4m3(uint16_t(high));
      a.v[3] = convert::unpack_bf16x2_e4m3(uint16_t(high >> 16));
#define CHOP_FRAGMENT_FP8_MMA(I, ACC)                                        \
      if constexpr (TileN / 8 > I) {                                          \
        const ptx::B2 b = RegisterB ?                                        \
            cached_b[(Sub * 2 + half) * (TileN / 8) + I] :                  \
            activation_fragment<TileN, ActivationStride>(                  \
                staged, I, Sub * 32 + half * 16);                            \
        ptx::mma_bf16(ACC, a, b);                                             \
      }
      CHOP_FRAGMENT_FP8_MMA(0, block0)
      CHOP_FRAGMENT_FP8_MMA(1, block1)
      CHOP_FRAGMENT_FP8_MMA(2, block2)
      CHOP_FRAGMENT_FP8_MMA(3, block3)
#undef CHOP_FRAGMENT_FP8_MMA
    }
  }
}

template<WeightKind W, int TileN, bool Residual = false, bool Swiglu = false,
         int TileRows = 128>
__device__ void run_fragment_decode(
    const uint8_t* weight, const uint16_t* activation,
    void* output, float* partial, uint32_t* semaphores,
    uint32_t live_m, uint32_t n, uint32_t k, uint32_t splits,
    uint32_t output_f32, const float* scales, const void* residual) {
  static_assert(W == WeightKind::bf16 || fragment_fp8<W>);
  static_assert(TileN == 8 || TileN == 16 || TileN == 32);
  static_assert(TileRows == 32 || TileRows == 64 || TileRows == 128);
  constexpr int tile_m = TileRows;
  constexpr int lanes = 32;
  constexpr int warps = TileRows / 16;
  constexpr int weight_tile_k = W == WeightKind::bf16 ? 16 : 32;
  constexpr int weight_tiles_per_block = 128 / weight_tile_k;
  extern __shared__ __align__(16) unsigned char storage[];
  auto* staged = reinterpret_cast<uint16_t*>(storage);
  __shared__ uint32_t last;
  const int warp = threadIdx.x / lanes;
  const int lane = threadIdx.x & 31;
  const uint32_t output_rows = Swiglu ? n / 2 : n;
  const uint32_t row_tiles = (output_rows + tile_m - 1) / tile_m;
  const uint32_t k_blocks = k / 128;
  const uint32_t k_tiles = k / weight_tile_k;
  const uint32_t initial_tile = blockIdx.x / splits;
  const uint32_t initial_split = blockIdx.x % splits;
  const uint32_t initial_first = uint64_t(k_blocks) * initial_split / splits;
  const uint32_t initial_strip = initial_tile * warps + warp;
  ptx::A4 w0{}, w1{}, w2{}, w3{}, w4{}, w5{}, w6{}, w7{};
#define CHOP_LOAD_SET(STRIP, BLOCK) do {                               \
    const uint64_t base = (uint64_t(STRIP) * k_tiles +              \
        uint64_t(BLOCK) * weight_tiles_per_block) * 512 + lane * 16; \
    load_weight_group4(weight + base, w0, w1, w2, w3);              \
    if constexpr (W == WeightKind::bf16) {                          \
      load_weight_group4(weight + base + 2048, w4, w5, w6, w7);     \
    }                                                               \
  } while (0)
  ptx::griddep_launch_dependents();
  if (initial_tile < row_tiles && initial_strip < (output_rows + 15) / 16) {
    CHOP_LOAD_SET(initial_strip, initial_first);
  }
  ptx::griddep_wait();

  for (uint64_t work = blockIdx.x; work < uint64_t(row_tiles) * splits;
       work += gridDim.x) {
    const uint32_t tile = uint32_t(work / splits);
    const uint32_t split = uint32_t(work % splits);
    const uint32_t first = uint64_t(k_blocks) * split / splits;
    const uint32_t end = uint64_t(k_blocks) * (split + 1) / splits;
    const uint32_t strip = tile * warps + warp;
    const bool valid_strip = strip < (output_rows + 15) / 16;
    for (int projection = 0; projection < (Swiglu ? 2 : 1); ++projection) {
    const uint32_t weight_strip = strip + projection * (output_rows / 16);
    ptx::D4 acc0{}, acc1{}, acc2{}, acc3{};
    if (valid_strip && (work != blockIdx.x || projection != 0))
      CHOP_LOAD_SET(weight_strip, first);

    for (uint32_t kb = first; kb < end; ++kb) {
      if (valid_strip && kb != first)
        CHOP_LOAD_SET(weight_strip, kb);
      // All warps stage the small activation slab once. The K-contiguous
      // vector copy keeps the weight path independent of tensor-map boxes.
      for (uint32_t vector = threadIdx.x; vector < TileN * 16;
           vector += blockDim.x) {
        const uint32_t token = vector / 16;
        const uint32_t byte_in_row = (vector % 16) * 16;
        const uint64_t source = uint64_t(token) * k * 2 + kb * 256 + byte_in_row;
        auto* dst = reinterpret_cast<uint4*>(storage + vector * 16);
        if (token < live_m) {
          const auto* src = reinterpret_cast<const uint4*>(
              reinterpret_cast<const uint8_t*>(activation) + source);
          *dst = *src;
        } else {
          *dst = make_uint4(0, 0, 0, 0);
        }
      }
      __syncthreads();

      // Each 128-K block accumulates from zero into block0..3, so no MMA
      // chain spans more than 128 K; FP8 promotes it by its F32 scale and
      // BF16 adds it. consume_weight_fragment accumulates BF16 into its first
      // four arguments (the router passes its own accumulators there).
      ptx::D4 block0{}, block1{}, block2{}, block3{};
      if (valid_strip) {
        const float scale = W == WeightKind::bf16 ? 1.0f :
            scales[uint64_t(weight_strip / fragment_scale_strips<W>) * k_blocks + kb];
 #define CHOP_CONSUME(SUB, VALUE) \
        if constexpr (W == WeightKind::bf16) \
          consume_weight_fragment<W, TileN, SUB>(VALUE, staged, \
              block0, block1, block2, block3, block0, block1, block2, block3); \
        else \
          consume_weight_fragment<W, TileN, SUB>(VALUE, staged, \
              acc0, acc1, acc2, acc3, block0, block1, block2, block3)
        CHOP_CONSUME(0, w0);
        CHOP_CONSUME(1, w1);
        CHOP_CONSUME(2, w2);
        CHOP_CONSUME(3, w3);
        if constexpr (W == WeightKind::bf16) {
          CHOP_CONSUME(4, w4);
          CHOP_CONSUME(5, w5);
          CHOP_CONSUME(6, w6);
          CHOP_CONSUME(7, w7);
        }
#undef CHOP_CONSUME
#pragma unroll
        for (int v = 0; v < 4; ++v) {
          acc0.v[v] += block0.v[v] * scale;
          if constexpr (TileN >= 16) acc1.v[v] += block1.v[v] * scale;
          if constexpr (TileN >= 24) acc2.v[v] += block2.v[v] * scale;
          if constexpr (TileN >= 32) acc3.v[v] += block3.v[v] * scale;
        }
      }
      __syncthreads();
    }

    if (valid_strip) {
      const int group = lane >> 2;
      const int lane4 = lane & 3;
#define CHOP_FRAGMENT_WRITE(I, ACC)                                          \
      if constexpr (TileN / 8 > I) {                                          \
        for (int v = 0; v < 4; ++v) {                                          \
          const uint32_t column = strip * 16 + group + (v / 2) * 8;          \
          const uint32_t token = I * 8 + lane4 * 2 + (v % 2);                 \
          if (column < output_rows && token < live_m) {                        \
            float value = ACC.v[v];                                           \
            if (splits > 1 || Swiglu) {                                        \
              const uint32_t index = (warp * 16 + group + (v / 2) * 8) *     \
                  TileN + I * 8 + lane4 * 2 + (v % 2);                         \
              partial[((uint64_t(tile) * splits + split) *                     \
                  (Swiglu ? 2 : 1) + projection) * tile_m * TileN +          \
                  index] = value;                                             \
            } else {                                                           \
              const uint64_t out = uint64_t(token) * n + column;             \
              if constexpr (Residual) value += residual_value(              \
                  residual, out, output_f32);                                 \
              if (output_f32) static_cast<float*>(output)[out] = value;      \
              else static_cast<__nv_bfloat16*>(output)[out] =               \
                  __float2bfloat16(value);                                     \
            }                                                                  \
          }                                                                    \
        }                                                                      \
      }
      CHOP_FRAGMENT_WRITE(0, acc0)
      CHOP_FRAGMENT_WRITE(1, acc1)
      CHOP_FRAGMENT_WRITE(2, acc2)
      CHOP_FRAGMENT_WRITE(3, acc3)
#undef CHOP_FRAGMENT_WRITE
    }
    __syncthreads();
    }
    if (splits > 1 || Swiglu) {
      if (threadIdx.x == 0) {
        if (splits == 1) last = 1;
        else {
          __threadfence();
          last = atomicAdd(&semaphores[tile], 1) == splits - 1;
        }
      }
      __syncthreads();
      if (last) {
        for (uint32_t index = threadIdx.x; index < tile_m * TileN;
             index += blockDim.x) {
          const uint32_t column = tile * tile_m + index / TileN;
          const uint32_t token = index % TileN;
          if (column >= output_rows || token >= live_m) continue;
          float value = 0;
          if constexpr (Swiglu) {
            float gate = 0;
            float up = 0;
            for (uint32_t s = 0; s < splits; ++s) {
              const uint64_t base = ((uint64_t(tile) * splits + s) * 2) *
                  tile_m * TileN + index;
              gate += partial[base];
              up += partial[base + tile_m * TileN];
            }
            value = (gate / (1.0f + expf(-gate))) * up;
          } else {
            for (uint32_t s = 0; s < splits; ++s)
              value += partial[(uint64_t(tile) * splits + s) * tile_m * TileN + index];
          }
          const uint64_t out = uint64_t(token) * output_rows + column;
          if constexpr (Residual) value += residual_value(residual, out, output_f32);
          if (output_f32) static_cast<float*>(output)[out] = value;
          else static_cast<__nv_bfloat16*>(output)[out] = __float2bfloat16(value);
        }
        __syncthreads();
        if (threadIdx.x == 0 && splits > 1) {
          __threadfence();
          atomicExch(&semaphores[tile], 0);
        }
      }
    }
    __syncthreads();
  }
#undef CHOP_LOAD_SET
}

} // namespace chop::gemm
