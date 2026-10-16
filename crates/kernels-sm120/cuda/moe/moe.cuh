#pragma once

#include <stdint.h>
#include "../common/convert.cuh"
#include "../common/ptx.cuh"
#include "../common/ptx_control.cuh"
#include "../common/ptx_matrix.cuh"
#include "../common/ptx_tma.cuh"
#include "route.cuh"

namespace chop::moe {

enum Format : int { mxfp4 = 0, fp8_block = 1, nvfp4 = 2, mxfp8 = 3 };

__device__ __forceinline__ uint32_t scaled_mxfp4_bf16x2(
    uint8_t codes, uint8_t scale) {
  uint32_t result;
  const uint16_t code_pair = codes;
  const uint16_t scale_pair = uint16_t(scale) * 0x0101;
  asm volatile("{ .reg .b8 c, unused; mov.b16 {c, unused}, %1; "
               "cvt.rn.scaled::n2::ue8m0.bf16x2.e2m1x2 %0, c, %2; }"
               : "=r"(result) : "h"(code_pair), "h"(scale_pair));
  return result;
}

__device__ __forceinline__ uint32_t scaled_mxfp8_bf16x2(
    uint16_t codes, uint8_t scale) {
  uint32_t result;
  const uint16_t scale_pair = uint16_t(scale) * 0x0101;
  asm volatile("cvt.rn.scaled::n2::ue8m0.bf16x2.e4m3x2 %0, %1, %2;"
               : "=r"(result) : "h"(codes), "h"(scale_pair));
  if (scale < 3 || scale > 246) {
    const auto raw = convert::unpack_f32_e4m3(codes);
    const float factor = ldexpf(1.0f, int(scale) - 127);
    const float lo = raw.lo * factor, hi = raw.hi * factor;
    const auto decoded = convert::bf16x2_to_f32(result);
    if (scale == 255 || !isfinite(lo) || !isfinite(hi) ||
        decoded.lo != lo || decoded.hi != hi)
      asm volatile("trap;");
  }
  return result;
}

__device__ __forceinline__ uint32_t pack_expanded_fp4(uint32_t low,
                                                        uint32_t high) {
  uint32_t packed = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    packed |= ((low >> (i * 8)) & 15) << (i * 4);
    packed |= ((high >> (i * 8)) & 15) << ((i + 4) * 4);
  }
  return packed;
}

// ldmatrix expands four FP4 values into each register. Repack two adjacent
// fragments into the eight consecutive FP4 values expected by mxf4nvf4 MMA.
__device__ __forceinline__ void load_fp4_fragment(
    uint32_t* out, const uint8_t* padded, int lane, int row_group) {
  uint32_t expanded[4];
  const int address_row = row_group * 8 + lane % 8;
  const int matrix = lane / 8;
  ptx::ldmatrix_b8x16<4, ptx::PackedTile::b4x16_p64>(
      expanded, padded + address_row * 64 + matrix * 16);
  const int source = (lane & ~3) + 2 * (lane & 1);
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const uint32_t low0 = __shfl_sync(0xffffffff, expanded[2 * half], source);
    const uint32_t low1 = __shfl_sync(0xffffffff, expanded[2 * half + 1], source);
    const uint32_t high0 = __shfl_sync(0xffffffff, expanded[2 * half], source + 1);
    const uint32_t high1 = __shfl_sync(0xffffffff, expanded[2 * half + 1], source + 1);
    const uint32_t low = (lane & 2) ? low1 : low0;
    const uint32_t high = (lane & 2) ? high1 : high0;
    out[half] = pack_expanded_fp4(low, high);
  }
}

__device__ __forceinline__ float ue8m0(uint8_t code) {
  return ldexpf(1.0f, int(code) - 127);
}

__device__ __forceinline__ float ue4m3(uint8_t code) {
  const int exponent = code >> 3;
  const int mantissa = code & 7;
  if (exponent == 15 && mantissa == 7) return __int_as_float(0x7fc00000);
  if (exponent == 0) return ldexpf(float(mantissa), -9);
  return ldexpf(1.0f + float(mantissa) * 0.125f, exponent - 7);
}

__device__ __forceinline__ uint64_t packed_scale_offset(
    int row, int k_block, int k_blocks) {
  const int k_tiles = (k_blocks + 7) / 8;
  const int logical = (row & 15) * 8 + (k_block & 7);
  return uint64_t((row / 16) * k_tiles + k_block / 8) * 128
       + (logical & 31) * 4 + logical / 32;
}

template<bool FragmentScales, int K>
__device__ __forceinline__ uint64_t scale_offset(int row, int k_block) {
  if constexpr (FragmentScales) {
    const int strip = row / 16;
    const int g = row & 7;
    return uint64_t(strip * (K / 64) + k_block / 2) * 32
         + g * 4 + (k_block & 1) * 2 + ((row & 15) >= 8);
  } else {
    return packed_scale_offset(row, k_block, K / 32);
  }
}

template<int F, int Experts, int N, int K,
         bool PerExpertGlobals = false, bool GateUp = false, bool Interleaved = false>
__device__ __forceinline__ float weight_value(
    const uint8_t* tile, int row_stride, int row, int local_k, int expert, int column,
    int global_k, const uint8_t* scales, const float* f32_scales) {
  constexpr int bytes = F == mxfp4 || F == nvfp4 ? 16 : 32;
  uint8_t code = tile[row * row_stride + (F == mxfp4 || F == nvfp4
                                          ? local_k / 2 : local_k)];
  float value;
  if constexpr (F == mxfp4 || F == nvfp4) {
    code = (local_k & 1) ? code >> 4 : code & 15;
    value = convert::unpack_f32_e2m1(uint16_t(code)).lo;
  } else {
    value = convert::unpack_f32_e4m3(uint16_t(code)).lo;
  }
  if constexpr (F == fp8_block) {
    constexpr int n_blocks = N / 128;
    constexpr int k_blocks = K / 128;
    const int offset = (expert * n_blocks + column / 128) * k_blocks + global_k / 128;
    return value * f32_scales[offset];
  } else {
    const int global_row = expert * N + column;
    if constexpr (F == nvfp4) {
      const uint64_t offset = packed_scale_offset(global_row, global_k / 16,
                                                   K / 16);
      return value * ue4m3(scales[offset]);
    } else {
      const uint64_t offset = packed_scale_offset(global_row, global_k / 32,
                                                   K / 32);
      return value * ue8m0(scales[offset]);
    }
  }
}

template<int F, bool FragmentScales, int TopK, bool Pregather, bool GateUp, bool Interleaved,
         bool QuantizeOut,
         int Experts, int N, int K,
         int RowBlock, int Warps, int SplitK,
         int DecodeWeightBytes = 0, bool PerExpertGlobals = false>
__device__ void run(
    const uint8_t* activation, const uint8_t* activation_scales,
    const float* activation_f32_scales, const uint8_t* weight_scales,
    const float* weight_f32_scales, uint16_t* output,
    const float* topk_weights, const int32_t* expert_offsets,
    const int32_t* sorted_rows, const int32_t* num_padded_rows,
    const int32_t* live_tokens, const int32_t* active_experts,
    uint8_t* quantized_output, uint8_t* quantized_scales,
    float* partials, const uint8_t* weight_ptr, const void* weight_map) {
  static_assert(RowBlock == Warps * 8);
  static_assert(K % 32 == 0 && N % 16 == 0);
  static_assert(SplitK > 0 && K % SplitK == 0);
  static_assert(RowBlock == 16 || SplitK == 1);
  static_assert(!PerExpertGlobals || (F == nvfp4 && !Pregather && SplitK == 1));
  static_assert(DecodeWeightBytes == 0 || DecodeWeightBytes == 32
      || DecodeWeightBytes == 64 || DecodeWeightBytes == 128);
  static_assert(!QuantizeOut || (GateUp && Pregather
      && (F == mxfp4 || F == mxfp8) && !Interleaved));
  constexpr bool fragment_format = FragmentScales;
  constexpr int bytes = fragment_format ? 32
      : F == mxfp4 || (F == nvfp4 && !Pregather) ? 16 : 32;
  constexpr int k_step = fragment_format || (F == nvfp4 && Pregather) ? 64 : 32;
  constexpr int out_n = GateUp ? N / 2 : N;
  constexpr int output_tile = QuantizeOut ? 32 : 16;
  constexpr int subtiles = QuantizeOut ? 2 : 1;
  constexpr int n_tiles = out_n / output_tile;
  constexpr bool decode = RowBlock == 16;
  constexpr int weight_bytes = 16 * bytes * (GateUp ? 2 : 1);
  constexpr int split_chunks = K / SplitK / k_step;
  constexpr int group_chunks = decode ? (DecodeWeightBytes ? DecodeWeightBytes
                                              : K == 384 ? 64 : 128) / bytes
                               : 1;
  constexpr int stage_step = k_step;
  constexpr int weight_row_stride = bytes * group_chunks;
  static_assert(!decode || K % (SplitK * k_step) == 0);
  static_assert(!decode || split_chunks % group_chunks == 0);
  __shared__ alignas(128) uint8_t weights[32 * weight_row_stride];
  __shared__ alignas(128) uint8_t fragments[GateUp ? 1024 : 512];
  extern __shared__ __align__(128) uint8_t weight_cache[];
  __shared__ alignas(128) uint16_t activation_tile[RowBlock * (fragment_format ? 64 : 32)];
  __shared__ alignas(128) uint8_t fp4_weight_padded[
      F == nvfp4 && Pregather ? (GateUp ? 2048 : 1024) : 1];
  __shared__ alignas(128) uint8_t fp4_activation_padded[
      F == nvfp4 && Pregather ? Warps * 512 : 1];
  __shared__ alignas(8) uint64_t barrier;

  if (threadIdx.x == 0) {
    ptx::griddep_launch_dependents();
    if constexpr (!fragment_format) if (blockIdx.x == 0) {
      const int32_t first[2] = {0, 0};
      ptx::prefetch_tensor_l2(weight_map, first);
    }
  }
  __syncthreads();
  ptx::griddep_wait();
  const int tokens = *live_tokens;
  const int padded = *num_padded_rows;
  if (tokens <= 0 || padded <= 0 || *active_experts <= 0) return;

  const int active = *active_experts;
  uint64_t total = 0;
  int seen = 0;
  for (int e = 0; e < Experts && seen < active; ++e) {
    const int rows = expert_offsets[e + 1] - expert_offsets[e];
    seen += rows > 0;
    if constexpr (decode)
      total += rows > 0 ? uint64_t(n_tiles) * SplitK : 0;
    else
      total += uint64_t((rows + RowBlock - 1) / RowBlock) * n_tiles;
  }
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const uint64_t policy = ptx::l2_policy<ptx::L2Evict::first>();
  for (uint64_t work = blockIdx.x; work < total; work += gridDim.x) {
    uint64_t local = work;
    int expert = 0;
    for (; expert < Experts; ++expert) {
      const int rows = expert_offsets[expert + 1] - expert_offsets[expert];
      const uint64_t units = decode
          ? (rows > 0 ? uint64_t(n_tiles) * SplitK : 0)
          : uint64_t((rows + RowBlock - 1) / RowBlock) * n_tiles;
      if (local < units) break;
      local -= units;
    }
    if (expert == Experts) break;
    const int split = decode ? int(local % SplitK) : 0;
    const int base_column = int((decode ? local / SplitK : local) % n_tiles)
                          * output_tile;
    const int row_tiles = decode
        ? (expert_offsets[expert + 1] - expert_offsets[expert] + RowBlock - 1)
              / RowBlock
        : 1;
    if constexpr (decode && !fragment_format) {
      if (threadIdx.x == 0) {
        ptx::mbarrier_init(&barrier, 1);
        ptx::fence_mbarrier_init();
      }
      __syncthreads();
      if (threadIdx.x == 0)
        ptx::mbarrier_arrive_expect_tx(
            &barrier, subtiles * split_chunks * weight_bytes);
      __syncthreads();
      if (threadIdx.x == 0) {
        for (int subtile = 0; subtile < subtiles; ++subtile) {
          const int column = base_column + subtile * 16;
          for (int chunk = 0; chunk < split_chunks; chunk += group_chunks) {
            const int k_base = (split * split_chunks + chunk) * k_step;
            uint8_t* tile = weight_cache
                          + (subtile * split_chunks + chunk) * weight_bytes;
            const int32_t coord_gate[2] = {
                k_base * bytes / k_step,
                expert * N + (Interleaved ? column * 2 : column)};
            ptx::tma_load_hint<2>(tile, weight_map, coord_gate,
                                  &barrier, policy);
            if constexpr (GateUp && !Interleaved) {
              const int32_t coord_up[2] = {
                  k_base * bytes / k_step, expert * N + column + N / 2};
              ptx::tma_load_hint<2>(tile + 16 * weight_row_stride, weight_map,
                                    coord_up, &barrier, policy);
            }
          }
        }
      }
      ptx::mbarrier_wait(&barrier, 0);
      __syncthreads();
      if (threadIdx.x == 0) ptx::mbarrier_inval(&barrier);
      __syncthreads();
    }
    for (int row_tile = 0; row_tile < row_tiles; ++row_tile) {
    const int first_row = expert_offsets[expert]
                        + (decode ? row_tile : int(local / n_tiles)) * RowBlock;
    ptx::D4 gate_tiles[subtiles], up_tiles[subtiles];
    for (int subtile = 0; subtile < subtiles; ++subtile) {
    const int column = base_column + subtile * 16;
    ptx::D4 gate{{0.0f, 0.0f, 0.0f, 0.0f}};
    ptx::D4 up{{0.0f, 0.0f, 0.0f, 0.0f}};
    for (int stage_base = split * (K / SplitK);
         stage_base < (split + 1) * (K / SplitK); stage_base += stage_step) {
      if constexpr (fragment_format || !decode || !Pregather) {
      if (threadIdx.x == 0) {
        ptx::mbarrier_init(&barrier, 1);
        ptx::fence_mbarrier_init();
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        ptx::mbarrier_arrive_expect_tx(&barrier,
            (fragment_format ? 512 * (GateUp ? 2 : 1)
                             : decode ? 0 : weight_bytes * group_chunks)
                + (Pregather ? 0 : RowBlock * (fragment_format ? 128 : 64)));
      }
      __syncthreads();
      }
      if constexpr (fragment_format) {
        if (threadIdx.x == 0) {
          const int strips_k = K / 64;
          const int strip = (expert * N + column) / 16;
          ptx::cp_async_bulk(fragments,
              weight_ptr + (uint64_t(strip) * strips_k + stage_base / 64) * 512,
              512, &barrier);
          if constexpr (GateUp)
            ptx::cp_async_bulk(fragments + 512,
                weight_ptr + (uint64_t(strip + N / 32) * strips_k
                              + stage_base / 64) * 512,
                512, &barrier);
        }
      } else if constexpr (!decode) {
      if (threadIdx.x == 0) {
        const int32_t coord_gate[2] = {stage_base * bytes / k_step,
                                       expert * N + (Interleaved ? column * 2 : column)};
        ptx::tma_load_hint<2>(weights, weight_map, coord_gate,
                              &barrier, policy);
        if constexpr (GateUp && !Interleaved) {
          const int32_t coord_up[2] = {stage_base * bytes / k_step,
                                       expert * N + column + N / 2};
          ptx::tma_load_hint<2>(weights + 16 * weight_row_stride, weight_map, coord_up,
                                &barrier, policy);
        }
      }
      }
      if constexpr (!Pregather) {
        if (threadIdx.x < RowBlock) {
          const int row_slot = first_row + threadIdx.x;
          const int routed = row_slot < expert_offsets[expert + 1]
                                  && row_slot < padded ? sorted_rows[row_slot] : -1;
          const int input_row = routed >= 0 && routed < tokens * TopK
                                    ? (GateUp ? routed / TopK : routed) : 0;
          ptx::cp_async_bulk(activation_tile
                  + threadIdx.x * (fragment_format ? 64 : 32),
              activation + (uint64_t(input_row) * K + stage_base) * 2,
              fragment_format ? 128 : 64, &barrier);
        }
      }
      if constexpr (fragment_format || !decode || !Pregather) {
        ptx::mbarrier_wait(&barrier, 0);
        __syncthreads();
      }
      if constexpr (fragment_format) {
        if (threadIdx.x < 32) {
          const int fragment_lane = threadIdx.x;
          const int g = fragment_lane >> 2;
          const int t = fragment_lane & 3;
          for (int projection = 0; projection < (GateUp ? 2 : 1); ++projection) {
            const uint4 fragment = *reinterpret_cast<const uint4*>(
                fragments + projection * 512 + fragment_lane * 16);
            const uint32_t words[4] = {
                fragment.x, fragment.y, fragment.z, fragment.w};
#pragma unroll
            for (int byte = 0; byte < 16; ++byte) {
              const int row = g + ((byte & 1) ? 8 : 0) + projection * 16;
              const int k_pair = (byte / 4) * 16
                  + ((byte / 2) & 1) * 8 + t * 2;
              weights[row * weight_row_stride + k_pair / 2] =
                  uint8_t(words[byte / 4] >> ((byte & 3) * 8));
            }
          }
        }
        __syncthreads();
      }
      for (int substep = 0; substep < (fragment_format ? 2 : 1); ++substep) {
      const int k_base = stage_base + substep * (fragment_format ? 32 : k_step);
      const uint8_t* weight_tile = fragment_format ? weights + substep * 16 : decode
          ? weight_cache + (subtile * split_chunks
                            + ((k_base / k_step) % split_chunks / group_chunks)
                                * group_chunks) * weight_bytes
                            + ((k_base / k_step) % group_chunks) * bytes
          : weights + substep * bytes;
      if constexpr (Pregather && F == nvfp4) {
        ptx::A4 a0, a1;
        ptx::B2 b;
        if (threadIdx.x < 64) {
          const int row = threadIdx.x / 4;
          const int group = threadIdx.x & 3;
          const int gate_source = (Interleaved ? 2 * row : row) * weight_row_stride + group * 8;
          const int up_source = (Interleaved ? 2 * row + 1 : 16 + row) * weight_row_stride
                              + group * 8;
#pragma unroll
          for (int byte = 0; byte < 16; ++byte) {
            fp4_weight_padded[row * 64 + group * 16 + byte] =
                byte < 8 ? weight_tile[gate_source + byte] : 0;
            if constexpr (GateUp)
              fp4_weight_padded[1024 + row * 64 + group * 16 + byte] =
                  byte < 8 ? weight_tile[up_source + byte] : 0;
          }
        }
        {
          const int row_slot = first_row + warp * 8 + lane / 4;
          const int routed = row_slot < expert_offsets[expert + 1]
                                  && row_slot < padded ? sorted_rows[row_slot] : -1;
          const int group = lane & 3;
          const int dest = warp * 512 + (lane / 4) * 64 + group * 16;
#pragma unroll
          for (int byte = 0; byte < 16; ++byte) {
            fp4_activation_padded[dest + byte] =
                byte < 8 && routed >= 0 && routed < tokens * TopK
                    ? activation[uint64_t(row_slot) * (K / 2) + k_base / 2
                                 + group * 8 + byte]
                    : 0;
          }
        }
        __syncthreads();
        uint32_t gate_low[2], gate_high[2];
        load_fp4_fragment(gate_low, fp4_weight_padded, lane, 0);
        load_fp4_fragment(gate_high, fp4_weight_padded, lane, 1);
        a0.v[0] = gate_low[0];
        a0.v[1] = gate_high[0];
        a0.v[2] = gate_low[1];
        a0.v[3] = gate_high[1];
        if constexpr (GateUp) {
          uint32_t up_low[2], up_high[2];
          load_fp4_fragment(up_low, fp4_weight_padded + 1024, lane, 0);
          load_fp4_fragment(up_high, fp4_weight_padded + 1024, lane, 1);
          a1.v[0] = up_low[0];
          a1.v[1] = up_high[0];
          a1.v[2] = up_low[1];
          a1.v[3] = up_high[1];
        }
        load_fp4_fragment(b.v, fp4_activation_padded + warp * 512, lane, 0);
        const int scale_row = lane / 4 + 8 * (lane & 1);
        const int scale_slot = first_row + warp * 8 + lane / 4;
        const int scale_routed = scale_slot < expert_offsets[expert + 1]
                                    && scale_slot < padded
                                    ? sorted_rows[scale_slot] : -1;
        uint32_t scale_gate = 0, scale_up = 0, scale_b = 0;
        for (int part = 0; part < 4; ++part) {
          const int kb = k_base / 16 + part;
          if (lane % 4 < 2) {
            scale_gate |= uint32_t(weight_scales[packed_scale_offset(
                expert * N + (Interleaved ? 2 * (column + scale_row)
                                           : column + scale_row), kb, K / 16)]) << (part * 8);
            if constexpr (GateUp)
              scale_up |= uint32_t(weight_scales[packed_scale_offset(
                  expert * N + (Interleaved ? 2 * (column + scale_row) + 1
                                             : column + N / 2 + scale_row),
                  kb, K / 16)]) << (part * 8);
          }
          if (lane % 4 == 0 && scale_routed >= 0 && scale_routed < tokens * TopK)
            scale_b |= uint32_t(activation_scales[
                uint64_t(scale_slot) * (K / 16) + kb]) << (part * 8);
        }
        ptx::mma_mxf4nvf4<4, true>(gate, a0, b, scale_gate, scale_b);
        if constexpr (GateUp)
          ptx::mma_mxf4nvf4<4, true>(up, a1, b, scale_up, scale_b);
      } else if constexpr (Pregather) {
        ptx::A4 a0, a1;
        ptx::B2 b;
        for (int i = 0; i < 4; ++i) {
          const int weight_row = lane / 4 + 8 * (i & 1);
          const int k = (lane & 3) * 4 + 16 * (i / 2);
          uint32_t packed_gate = 0, packed_up = 0;
          for (int byte = 0; byte < 4; ++byte) {
            uint8_t raw_gate = weight_tile[(Interleaved ? 2 * weight_row : weight_row) * weight_row_stride
                                       + (F == mxfp4 ? (k + byte) / 2 : k + byte)];
            if constexpr (F == mxfp4) {
              raw_gate = ((k + byte) & 1) ? raw_gate >> 4 : raw_gate & 15;
              raw_gate <<= 2;
            }
            packed_gate |= uint32_t(raw_gate) << (byte * 8);
            if constexpr (GateUp) {
              uint8_t raw_up = weight_tile[(Interleaved ? 2 * weight_row + 1 : 16 + weight_row) * weight_row_stride
                                         + (F == mxfp4 ? (k + byte) / 2 : k + byte)];
              if constexpr (F == mxfp4) {
                raw_up = ((k + byte) & 1) ? raw_up >> 4 : raw_up & 15;
                raw_up <<= 2;
              }
              packed_up |= uint32_t(raw_up) << (byte * 8);
            }
          }
          a0.v[i] = packed_gate;
          if constexpr (GateUp) a1.v[i] = packed_up;
        }
        for (int i = 0; i < 2; ++i) {
          const int row_slot = first_row + warp * 8 + lane / 4;
          const int routed = row_slot < expert_offsets[expert + 1]
                                  && row_slot < padded ? sorted_rows[row_slot] : -1;
          const int k = k_base + (lane & 3) * 4 + 16 * i;
          uint32_t packed = 0;
          if (routed >= 0 && routed < tokens * TopK) {
            for (int byte = 0; byte < 4; ++byte)
              packed |= uint32_t(activation[uint64_t(row_slot) * K + k + byte])
                        << (byte * 8);
          }
          b.v[i] = packed;
        }
        if constexpr (F == fp8_block) {
          ptx::D4 partial_gate{{0.0f, 0.0f, 0.0f, 0.0f}};
          ptx::D4 partial_up{{0.0f, 0.0f, 0.0f, 0.0f}};
          ptx::mma_f8f6f4<ptx::Element::e4m3,
                           ptx::Element::e4m3>(partial_gate, a0, b);
          if constexpr (GateUp)
            ptx::mma_f8f6f4<ptx::Element::e4m3,
                             ptx::Element::e4m3>(partial_up, a1, b);
          for (int i = 0; i < 4; ++i) {
            const int row_slot = first_row + warp * 8 + (lane & 3) * 2 + (i & 1);
            const int routed = row_slot < expert_offsets[expert + 1]
                                    && row_slot < padded ? sorted_rows[row_slot] : -1;
            if (routed < 0 || routed >= tokens * TopK) continue;
            const int out_col = column + lane / 4 + 8 * (i / 2);
            const int gate_col = Interleaved ? 2 * out_col : out_col;
            const int weight_scale = (expert * (N / 128) + gate_col / 128)
                                     * (K / 128) + k_base / 128;
            const float act_scale = activation_f32_scales[
                uint64_t(row_slot) * (K / 128) + k_base / 128];
            gate.v[i] += partial_gate.v[i] * weight_f32_scales[weight_scale]
                         * act_scale;
            if constexpr (GateUp) {
              const int up_scale = (expert * (N / 128)
                                   + (Interleaved ? gate_col + 1 : out_col + N / 2) / 128)
                                   * (K / 128) + k_base / 128;
              up.v[i] += partial_up.v[i] * weight_f32_scales[up_scale]
                         * act_scale;
            }
          }
        } else {
          const int scale_row = lane / 4 + 8 * (lane & 1);
          const uint32_t scale_a = lane % 4 < 2
              ? uint32_t(weight_scales[scale_offset<FragmentScales, K>(
                    expert * N + (Interleaved ? 2 * (column + scale_row)
                                               : column + scale_row), k_base / 32)])
              : 0;
          const int scale_slot = first_row + warp * 8 + lane / 4;
          const int scale_routed = scale_slot < expert_offsets[expert + 1]
                                      && scale_slot < padded
                                      ? sorted_rows[scale_slot] : -1;
          const uint32_t scale_b = lane % 4 == 0 && scale_routed >= 0
                                      && scale_routed < tokens * TopK
              ? uint32_t(activation_scales[
                    uint64_t(scale_slot) * (K / 32) + k_base / 32]) : 127;
          if constexpr (F == mxfp4)
            ptx::mma_mxf8f6f4<ptx::Element::e2m1,
                               ptx::Element::e4m3>(gate, a0, b, scale_a, scale_b);
          else
            ptx::mma_mxf8f6f4<ptx::Element::e4m3,
                               ptx::Element::e4m3>(gate, a0, b, scale_a, scale_b);
          if constexpr (GateUp) {
            const uint32_t up_scale = lane % 4 < 2
                ? uint32_t(weight_scales[scale_offset<FragmentScales, K>(
                      expert * N + (Interleaved ? 2 * (column + scale_row) + 1
                                                 : column + N / 2 + scale_row),
                      k_base / 32)])
                : 0;
            if constexpr (F == mxfp4)
              ptx::mma_mxf8f6f4<ptx::Element::e2m1,
                                 ptx::Element::e4m3>(up, a1, b, up_scale, scale_b);
            else
              ptx::mma_mxf8f6f4<ptx::Element::e4m3,
                                 ptx::Element::e4m3>(up, a1, b, up_scale, scale_b);
          }
        }
      } else {
      uint8_t gate_scales[2]{}, up_scales[2]{};
      if constexpr (decode && (F == mxfp4 || F == mxfp8)) {
#pragma unroll
        for (int part = 0; part < 2; ++part) {
          const int row = lane / 4 + part * 8;
          const int gate_column = Interleaved ? 2 * (column + row)
                                                : column + row;
          gate_scales[part] = weight_scales[scale_offset<FragmentScales, K>(
              expert * N + gate_column, k_base / 32)];
          if constexpr (GateUp) {
            const int up_column = Interleaved ? gate_column + 1
                                              : gate_column + N / 2;
            up_scales[part] = weight_scales[scale_offset<FragmentScales, K>(
                expert * N + up_column, k_base / 32)];
          }
        }
      }
      for (int half = 0; half < 2; ++half) {
        ptx::A4 a0, a1;
        ptx::B2 b;
        for (int i = 0; i < 4; ++i) {
          const int weight_row = lane / 4 + 8 * (i & 1);
          const int local_k = half * 16 + (lane & 3) * 2 + 8 * (i / 2);
          const int global_k = k_base + local_k;
          const int gate_column = Interleaved ? 2 * (column + weight_row)
                                                : column + weight_row;
          const int gate_row = Interleaved ? 2 * weight_row : weight_row;
          if constexpr (F == mxfp4 || F == mxfp8) {
            uint8_t scale;
            if constexpr (decode) scale = gate_scales[i & 1];
            else scale = weight_scales[scale_offset<FragmentScales, K>(
                expert * N + gate_column, global_k / 32)];
            if constexpr (F == mxfp4)
              a0.v[i] = scaled_mxfp4_bf16x2(
                  weight_tile[gate_row * weight_row_stride + local_k / 2], scale);
            else {
              const int index = gate_row * weight_row_stride + local_k;
              a0.v[i] = scaled_mxfp8_bf16x2(
                  uint16_t(weight_tile[index]) | uint16_t(weight_tile[index + 1]) << 8,
                  scale);
            }
          } else {
            const float w0 = weight_value<F, Experts, N, K, PerExpertGlobals, GateUp, Interleaved>(
                weight_tile, weight_row_stride, gate_row,
                local_k, expert, gate_column,
                global_k, weight_scales, weight_f32_scales);
            const float w1 = weight_value<F, Experts, N, K, PerExpertGlobals, GateUp, Interleaved>(
                weight_tile, weight_row_stride, gate_row,
                local_k + 1, expert, gate_column,
                global_k + 1, weight_scales, weight_f32_scales);
            a0.v[i] = convert::bf16x2_from_f32({w0, w1});
          }
          if constexpr (GateUp) {
            const int up_column = Interleaved ? gate_column + 1
                                              : gate_column + N / 2;
            const int up_row = Interleaved ? 2 * weight_row + 1 : 16 + weight_row;
            if constexpr (F == mxfp4 || F == mxfp8) {
              uint8_t scale;
              if constexpr (decode) scale = up_scales[i & 1];
              else scale = weight_scales[scale_offset<FragmentScales, K>(
                  expert * N + up_column, global_k / 32)];
              if constexpr (F == mxfp4)
                a1.v[i] = scaled_mxfp4_bf16x2(
                    weight_tile[up_row * weight_row_stride + local_k / 2], scale);
              else {
                const int index = up_row * weight_row_stride + local_k;
                a1.v[i] = scaled_mxfp8_bf16x2(
                    uint16_t(weight_tile[index]) | uint16_t(weight_tile[index + 1]) << 8,
                    scale);
              }
            } else {
              const float u0 = weight_value<F, Experts, N, K, PerExpertGlobals, GateUp, Interleaved>(
                  weight_tile, weight_row_stride, up_row,
                  local_k, expert, up_column,
                  global_k, weight_scales, weight_f32_scales);
              const float u1 = weight_value<F, Experts, N, K, PerExpertGlobals, GateUp, Interleaved>(
                  weight_tile, weight_row_stride, up_row,
                  local_k + 1, expert, up_column,
                  global_k + 1, weight_scales, weight_f32_scales);
              a1.v[i] = convert::bf16x2_from_f32({u0, u1});
            }
          }
        }
        for (int i = 0; i < 2; ++i) {
          const int row_slot = first_row + warp * 8 + lane / 4;
          const int routed = row_slot < expert_offsets[expert + 1]
                                  && row_slot < padded ? sorted_rows[row_slot] : -1;
          const int k = k_base + half * 16 + (lane & 3) * 2 + i * 8;
          uint32_t packed_activation = 0;
          if (routed >= 0 && routed < tokens * TopK) {
            const int tile_index = (row_slot - first_row)
                    * (fragment_format ? 64 : 32) + k - stage_base;
            packed_activation = uint32_t(activation_tile[tile_index])
                | (uint32_t(activation_tile[tile_index + 1]) << 16);
          }
          b.v[i] = packed_activation;
        }
        ptx::mma_bf16(gate, a0, b);
        if constexpr (GateUp) ptx::mma_bf16(up, a1, b);
      }
      }
      }
      __syncthreads();
      if constexpr (fragment_format || !decode || !Pregather) {
        if (threadIdx.x == 0) ptx::mbarrier_inval(&barrier);
        __syncthreads();
      }
    }
    gate_tiles[subtile] = gate;
    up_tiles[subtile] = up;
    }
    if constexpr (SplitK > 1) {
      for (int subtile = 0; subtile < subtiles; ++subtile) {
        for (int i = 0; i < 4; ++i) {
          const int row_slot = first_row + warp * 8 + (lane & 3) * 2 + (i & 1);
          const int routed = row_slot < expert_offsets[expert + 1]
                                  && row_slot < padded ? sorted_rows[row_slot] : -1;
          if (routed < 0 || routed >= tokens * TopK) continue;
          const int out_column = base_column + subtile * 16
                               + lane / 4 + 8 * (i / 2);
          const uint64_t index =
              (uint64_t(split * tokens * TopK + routed) * out_n + out_column)
              * (GateUp ? 2 : 1);
          partials[index] = gate_tiles[subtile].v[i];
          if constexpr (GateUp) partials[index + 1] = up_tiles[subtile].v[i];
        }
      }
    } else {
    float values[subtiles][4];
    for (int subtile = 0; subtile < subtiles; ++subtile) {
      for (int i = 0; i < 4; ++i) {
        const int row_slot = first_row + warp * 8 + (lane & 3) * 2 + (i & 1);
        const int routed = row_slot < expert_offsets[expert + 1]
                                && row_slot < padded ? sorted_rows[row_slot] : -1;
        float value = gate_tiles[subtile].v[i];
        if constexpr (F == nvfp4 && Pregather)
          value *= weight_f32_scales[0] * activation_f32_scales[0];
        if constexpr (F == nvfp4 && !Pregather)
          value = __fmul_rn(value, weight_f32_scales[PerExpertGlobals ? expert * (GateUp ? 2 : 1) : 0]);
        if constexpr (GateUp) {
          float up_value = up_tiles[subtile].v[i];
          if constexpr (F == nvfp4 && Pregather)
            up_value *= weight_f32_scales[0] * activation_f32_scales[0];
          if constexpr (F == nvfp4 && !Pregather)
            up_value = __fmul_rn(up_value, weight_f32_scales[PerExpertGlobals ? expert * 2 + 1 : 0]);
          value = value / (1.0f + expf(-value)) * up_value;
        } else if (routed >= 0 && routed < tokens * TopK) {
          value *= topk_weights[routed];
        }
        values[subtile][i] = value;
        if (routed < 0 || routed >= tokens * TopK) continue;
        const int out_column = base_column + subtile * 16
                             + lane / 4 + 8 * (i / 2);
        output[uint64_t(routed) * out_n + out_column] =
            convert::bf16_from_f32(value);
      }
    }
    if constexpr (QuantizeOut) {
      for (int parity = 0; parity < 2; ++parity) {
        float maximum = 0.0f;
        for (int subtile = 0; subtile < 2; ++subtile) {
          maximum = fmaxf(maximum, fabsf(values[subtile][parity]));
          maximum = fmaxf(maximum, fabsf(values[subtile][parity + 2]));
        }
        for (int mask = 4; mask <= 16; mask <<= 1) {
          const uint32_t other = ptx::shuffle_xor(__float_as_uint(maximum), mask);
          maximum = fmaxf(maximum, __uint_as_float(other));
        }
        int scale_code = maximum == 0.0f ? 127 : ilogbf(maximum) - 8 + 127;
        scale_code = scale_code < 0 ? 0 : (scale_code > 254 ? 254 : scale_code);
        const float scale = ldexpf(1.0f, scale_code - 127);
        const int row_slot = first_row + warp * 8 + (lane & 3) * 2 + parity;
        const int routed = row_slot < expert_offsets[expert + 1]
                                && row_slot < padded ? sorted_rows[row_slot] : -1;
        if (routed < 0 || routed >= tokens * TopK) continue;
        if (lane / 4 == 0)
          quantized_scales[uint64_t(routed) * (out_n / 32)
                           + base_column / 32] = uint8_t(scale_code);
        for (int subtile = 0; subtile < 2; ++subtile) {
          for (int half = 0; half < 2; ++half) {
            const int i = parity + half * 2;
            const int out_column = base_column + subtile * 16
                                 + lane / 4 + 8 * half;
            quantized_output[uint64_t(routed) * out_n + out_column] =
                uint8_t(convert::pack_f32_e4m3({values[subtile][i] / scale,
                                                 values[subtile][i] / scale}));
          }
        }
      }
    }
    }
    }
  }
}

template<bool GateUp, bool QuantizeOut, int NvGlobal,
         int TopK, int OutN, int TokensCapacity, int SplitK>
__device__ void reduce(
    const float* partials, uint16_t* output, const float* topk_weights,
    const int32_t* live_tokens, const float* activation_f32_scales,
    const float* weight_f32_scales, uint8_t* quantized_output,
    uint8_t* quantized_scales) {
  static_assert(SplitK > 1 && OutN % 32 == 0);
  if (threadIdx.x == 0) ptx::griddep_launch_dependents();
  __syncthreads();
  ptx::griddep_wait();
  const uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= uint64_t(TokensCapacity) * TopK * OutN) return;
  const int tokens = *live_tokens;
  if (index >= uint64_t(tokens) * TopK * OutN) return;
  const int routed = int(index / OutN);
  float gate = 0.0f, up = 0.0f;
#pragma unroll
  for (int split = 0; split < SplitK; ++split) {
    const uint64_t partial_index =
        (uint64_t(split * tokens * TopK + routed) * OutN + index % OutN)
        * (GateUp ? 2 : 1);
    gate += partials[partial_index];
    if constexpr (GateUp) up += partials[partial_index + 1];
  }
  if constexpr (NvGlobal == 1) {
    const float scale = activation_f32_scales[0] * weight_f32_scales[0];
    gate *= scale;
    if constexpr (GateUp) up *= scale;
  } else if constexpr (NvGlobal == 2) {
    gate = __fmul_rn(gate, weight_f32_scales[0]);
    if constexpr (GateUp) up = __fmul_rn(up, weight_f32_scales[0]);
  }
  const float value = GateUp
      ? gate / (1.0f + expf(-gate)) * up
      : gate * topk_weights[routed];
  output[index] = convert::bf16_from_f32(value);
  if constexpr (QuantizeOut) {
    float maximum = fabsf(value);
    for (int mask = 16; mask > 0; mask >>= 1)
      maximum = fmaxf(maximum, __uint_as_float(
          ptx::shuffle_xor(__float_as_uint(maximum), mask)));
    int scale_code = maximum == 0.0f ? 127 : ilogbf(maximum) - 8 + 127;
    scale_code = scale_code < 0 ? 0 : (scale_code > 254 ? 254 : scale_code);
    const float scale = ldexpf(1.0f, scale_code - 127);
    if ((threadIdx.x & 31) == 0)
      quantized_scales[uint64_t(routed) * (OutN / 32)
                       + (index % OutN) / 32] = uint8_t(scale_code);
    quantized_output[index] = uint8_t(convert::pack_f32_e4m3(
        {value / scale, value / scale}));
  }
}

} // namespace chop::moe

#include "moe/fragments.cuh"
