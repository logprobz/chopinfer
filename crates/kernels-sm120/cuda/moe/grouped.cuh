#pragma once

#include <stdint.h>
#include "prefill_staging.cuh"

// One grouped-MoE template for every model (README "Grouped MoE template").
// A kernel is instantiated per model at its own shape, weight format and
// schedule; nothing is decided at run time that the per-model engines fix
// at compile time. Two schedules share the pieces below:
//
// - `grouped_ring.cuh`: per-warp cp.async rings, one CTA per (active
//   expert, output tile) over the full K, K split across the CTA's warps.
// - `grouped_prefill.cuh`: a persistent CTA per SM over (expert, feature
//   tile, 128-row token tile) items with producer rings.
//
// Every format feeds the BF16 MMA exact operands: an FP8 or FP4 code, or
// a code times a power-of-two or UE4M3 local scale, is representable in
// BF16. Any F32 scale is applied to MMA sums: FP8 block scales per 128-K
// block, NVFP4 globals once per output before the epilogue. The epilogues
// round to BF16 once, after SwiGLU or the route weight.
namespace chop::moe::grouped {

using prefill::cp_async16;
using prefill::cp_async_arrive;
using prefill::pack_bf16;
using prefill::row_chunk;
using prefill::Ring;
using prefill::RouteSlot;
using prefill::Tile;

// Weight formats. `mxfp4` and `fp8_block128` are served (MiMo and Qwen).
// The others are named for the parameterization in the README; their
// instances come later and every schedule refuses them until then.
enum class Weights : int {
  mxfp4 = 0,         // E2M1 codes, UE8M0 per K32, scale folded in the operand
  fp8_block128 = 1,  // E4M3 codes, F32 per 128 x 128 block after the MMA
  nvfp4 = 2,         // E2M1 x UE4M3 per K16 operand, F32 expert global after
  mxfp8 = 3,         // E4M3 x UE8M0 per K32 operand
  bf16 = 4,          // BF16 operand
};

// One rank's expert shape.
struct Shape {
  int experts;       // local experts
  int top_k;         // routes per token; routed row r = token * top_k + k
  int hidden;
  int intermediate;  // per-rank expert intermediate
};

template<Weights W>
constexpr bool served = W == Weights::mxfp4 || W == Weights::fp8_block128;

// Bytes of one 16-row strip's K64 codes in the fragment-major layouts:
// one 512-byte 16 x K64 tile of E2M1, or two 512-byte 16 x K32 tiles of E4M3.
template<Weights W>
constexpr int kStripStepBytes = W == Weights::mxfp4 ? 512 : 1024;

__host__ __device__ constexpr int log2_exact(int value) {
  return value <= 1 ? 0 : 1 + log2_exact(value / 2);
}

// Four K16 MMA A fragments of an E2M1 fragment word, scaled per K32 by the
// lane's UE8M0 word (the two row groups of both K32 blocks).
__device__ __forceinline__ ptx::A4 mxfp4_operand(uint4 codes, uint32_t scale, int k16) {
  const uint32_t word = fragment_word(codes, k16);
  const int block = k16 >> 1;
  const uint32_t low = __byte_perm(scale, 0, 0x22 * block);
  const uint32_t high = __byte_perm(scale, 0, 0x22 * block + 0x11);
  ptx::A4 a;
  a.v[0] = dequant(word, 0, low);
  a.v[1] = dequant(word, 1, high);
  a.v[2] = dequant(word, 2, low);
  a.v[3] = dequant(word, 3, high);
  return a;
}

// Two E4M3 bytes per packed conversion; one 16-byte fragment covers two
// K16 steps (`half` selects the step).
__device__ __forceinline__ ptx::A4 fp8_operand(uint4 fragment, int half) {
  const uint32_t low = half ? fragment.z : fragment.x;
  const uint32_t high = half ? fragment.w : fragment.y;
  ptx::A4 a;
  a.v[0] = convert::unpack_bf16x2_e4m3(uint16_t(low));
  a.v[1] = convert::unpack_bf16x2_e4m3(uint16_t(low >> 16));
  a.v[2] = convert::unpack_bf16x2_e4m3(uint16_t(high));
  a.v[3] = convert::unpack_bf16x2_e4m3(uint16_t(high >> 16));
  return a;
}

// ---------------------------------------------------------------------
// Pieces of the persistent prefill schedule: a K64 weight stage of 16
// strips, its consumption by a math warp, and the staged BF16 epilogue.

// A weight stage: 16 strips' codes, then their scales. MXFP4 carries the
// 16 compact 32-byte scale tiles; FP8 carries the item's two F32 block
// scales (gate and up, or the two 128-row halves of a down item).
template<Weights W>
constexpr int kStageCodeBytes = prefill::kWeightTiles * kStripStepBytes<W>;
template<Weights W>
constexpr int kStageBytes = kStageCodeBytes<W> + (W == Weights::mxfp4 ? prefill::kScaleBytes : 16);

// Weight tile j of an item. Gate/up tiles 0..7 are gate rows and 8..15
// the matching up rows of 128 output columns; down tiles are 256
// adjacent rows.
template<bool GateUp, int N>
__device__ __forceinline__ int stage_strip(const Tile& tile, int j) {
  if constexpr (GateUp)
    return (tile.expert * N + (j >= 8 ? N / 2 : 0) + tile.feature_tile * 128 + (j & 7) * 16) / 16;
  else
    return (tile.expert * N + tile.feature_tile * prefill::kWeightRows + j * 16) / 16;
}

// Warp-uniform sources of one item's weight stages.
template<Weights W>
struct StageSource {
  const uint8_t* strips[2];  // tiles 0-7 and 8-15, each eight consecutive strips
  const void* scale;
};

template<Weights W, bool GateUp, int N, int K>
__device__ __forceinline__ StageSource<W> stage_source(
    const Tile& tile, const uint8_t* weight_bytes, const uint8_t* weight_scales,
    const float* weight_f32_scales, int lane) {
  static_assert(served<W>);
  constexpr int Ktiles = K / 64;
  StageSource<W> source;
#pragma unroll
  for (int half = 0; half < 2; ++half)
    source.strips[half] = weight_bytes
        + uint64_t(stage_strip<GateUp, N>(tile, 8 * half)) * Ktiles * kStripStepBytes<W> + lane * 16;
  if constexpr (W == Weights::mxfp4)
    // Lane l copies half l & 1 of strip l >> 1's scale tile.
    source.scale = weight_scales
        + uint64_t(stage_strip<GateUp, N>(tile, lane >> 1)) * Ktiles * 32 + (lane & 1) * 16;
  else
    // Lanes 0 and 1 copy the scale of 128-row block (tiles 0-7, 8-15).
    source.scale = weight_f32_scales
        + uint64_t(stage_strip<GateUp, N>(tile, 8 * (lane & 1)) / 8) * (K / 128);
  return source;
}

// Copies K64 stage kt of an item with the calling warp. `Bulk` issues one
// 1024-byte bulk copy per FP8 strip against the stage's full barrier.
template<Weights W, bool GateUp, int N, int K, bool Bulk>
__device__ __forceinline__ void produce_stage(
    const StageSource<W>& source, const Tile& tile, uint8_t* stage, uint64_t* full,
    const uint8_t* weight_bytes, int kt, int lane) {
  constexpr int Ktiles = K / 64;
  if constexpr (W == Weights::mxfp4) {
#pragma unroll
    for (int j = 0; j < prefill::kWeightTiles; ++j)
      cp_async16(stage + j * 512 + lane * 16,
                 source.strips[j >> 3] + (uint64_t(j & 7) * Ktiles + kt) * 512);
    cp_async16(stage + kStageCodeBytes<W> + lane * 16,
               static_cast<const uint8_t*>(source.scale) + kt * 32);
  } else {
    static_assert(!Bulk || GateUp, "bulk stage copies cover gate/up strips");
    if constexpr (Bulk) {
      if (lane == 0) ptx::mbarrier_expect_tx(full, kStageCodeBytes<W>);
      __syncwarp();
      if (lane < prefill::kWeightTiles)
        ptx::cp_async_bulk(stage + lane * 1024,
            weight_bytes + (uint64_t(stage_strip<true, N>(tile, lane)) * (K / 32) + 2 * kt) * 512,
            1024, full);
    } else {
#pragma unroll
      for (int j = 0; j < prefill::kWeightTiles; ++j)
#pragma unroll
        for (int t = 0; t < 2; ++t)
          cp_async16(stage + j * 1024 + t * 512 + lane * 16,
                     source.strips[j >> 3] + (uint64_t(j & 7) * (K / 32) + 2 * kt + t) * 512);
    }
    if (lane < 2)
      cp_async_4(stage + kStageCodeBytes<W> + lane * 4,
                 static_cast<const float*>(source.scale) + kt / 2);
  }
  cp_async_arrive(full);
}

// Warp 8 streams K64 stages [kt0, kt1) of one item's weights through the
// weight ring.
template<Weights W, bool GateUp, int N, int K, int Stages, bool Bulk>
__device__ __forceinline__ void produce_weights(
    const Tile& tile, const Ring& ring, uint32_t& step, int kt0, int kt1,
    const uint8_t* weight_bytes, const uint8_t* weight_scales, const float* weight_f32_scales,
    int lane) {
  const StageSource<W> source =
      stage_source<W, GateUp, N, K>(tile, weight_bytes, weight_scales, weight_f32_scales, lane);
#pragma unroll 1
  for (int kt = kt0; kt < kt1; ++kt, ++step) {
    const int slot = step % Stages;
    ptx::mbarrier_wait(&ring.empty[slot], ((step / Stages) & 1) ^ 1);
    produce_stage<W, GateUp, N, K, Bulk>(source, tile, ring.slots + slot * kStageBytes<W>,
                                         &ring.full[slot], weight_bytes, kt, lane);
  }
}

// Per-warp F32 block scales of the persistent FP8 schedules: the scale of
// each of the warp's two strips, read at the first K64 stage of a block.
template<Weights W>
struct BlockScales {
  float value[W == Weights::fp8_block128 ? 2 : 1];
};

// A math warp's 32 weight rows (strips m = 0, 1) against up to `Groups`
// 16-row groups of routed rows, for one K64 stage; full items (128-row
// tiles) run fully unrolled. `block_start` marks the first stage of a
// 128-K block.
//
// MXFP4 accumulates each K16 step into the tile. FP8 keeps the tile in
// output units: each K32 half's products of one 16-row group accumulate
// into a temporary that is promoted by its block's F32 scale with one
// fused multiply-add; a block whose scale is zero is skipped.
template<Weights W, bool GateUp, int Groups>
__device__ __forceinline__ void consume_stage(
    const uint8_t* stage, const uint8_t* rows, ptx::D4 (&accum)[2][2 * Groups],
    BlockScales<W>& scales, int groups, bool block_start, int lane, int warp) {
  if constexpr (W == Weights::mxfp4) {
    uint4 fragment[2];
    uint32_t scale_word[2];
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      const int j = GateUp ? warp + 8 * m : 2 * warp + m;
      fragment[m] = *reinterpret_cast<const uint4*>(stage + j * 512 + lane * 16);
      scale_word[m] = *reinterpret_cast<const uint32_t*>(
          stage + kStageCodeBytes<W> + j * 32 + (lane >> 2) * 4);
    }
#pragma unroll
    for (int k16 = 0; k16 < 4; ++k16) {
      ptx::A4 a[2];
#pragma unroll
      for (int m = 0; m < 2; ++m) a[m] = mxfp4_operand(fragment[m], scale_word[m], k16);
      auto group = [&](int p) {
        const int row = 16 * p + (lane & 7) + 8 * (lane >> 4);
        const int chunk = 2 * k16 + ((lane >> 3) & 1);
        uint32_t b[4];
        ptx::ldmatrix_b16<4>(b, rows + row_chunk(row, chunk));
        const ptx::B2 low_rows{{b[0], b[1]}};
        const ptx::B2 high_rows{{b[2], b[3]}};
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          ptx::mma_bf16(accum[m][2 * p], a[m], low_rows);
          ptx::mma_bf16(accum[m][2 * p + 1], a[m], high_rows);
        }
      };
      if (groups == Groups) {
#pragma unroll
        for (int p = 0; p < Groups; ++p) group(p);
      } else {
#pragma unroll
        for (int p = 0; p < Groups; ++p) {
          if (p >= groups) break;
          group(p);
        }
      }
    }
  } else {
    static_assert(W == Weights::fp8_block128);
    if (block_start)
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        const int j = GateUp ? warp + 8 * m : 2 * warp + m;
        scales.value[m] = *reinterpret_cast<const float*>(stage + kStageCodeBytes<W> + (j >> 3) * 4);
      }
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      ptx::A4 a[2][2];
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        const int j = GateUp ? warp + 8 * m : 2 * warp + m;
        const uint4 fragment =
            *reinterpret_cast<const uint4*>(stage + j * 1024 + half * 512 + lane * 16);
        a[m][0] = fp8_operand(fragment, 0);
        a[m][1] = fp8_operand(fragment, 1);
      }
      auto group = [&](int p) {
        ptx::D4 part[2][2] = {};
#pragma unroll
        for (int k = 0; k < 2; ++k) {
          const int row = 16 * p + (lane & 7) + 8 * (lane >> 4);
          const int chunk = 4 * half + 2 * k + ((lane >> 3) & 1);
          uint32_t b[4];
          ptx::ldmatrix_b16<4>(b, rows + row_chunk(row, chunk));
          const ptx::B2 low_rows{{b[0], b[1]}};
          const ptx::B2 high_rows{{b[2], b[3]}};
#pragma unroll
          for (int m = 0; m < 2; ++m) {
            if (scales.value[m] == 0.f) continue;
            ptx::mma_bf16(part[m][0], a[m][k], low_rows);
            ptx::mma_bf16(part[m][1], a[m][k], high_rows);
          }
        }
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          if (scales.value[m] == 0.f) continue;
#pragma unroll
          for (int q = 0; q < 2; ++q)
#pragma unroll
            for (int i = 0; i < 4; ++i)
              accum[m][2 * p + q].v[i] =
                  fmaf(part[m][q].v[i], scales.value[m], accum[m][2 * p + q].v[i]);
        }
      };
      if (groups == Groups) {
#pragma unroll
        for (int p = 0; p < Groups; ++p) group(p);
      } else {
#pragma unroll
        for (int p = 0; p < Groups; ++p) {
          if (p >= groups) break;
          group(p);
        }
      }
    }
  }
}

// The staged epilogue of a math warp: SwiGLU of the gate and up tiles, or
// the route weight, is rounded to BF16 once; each 16-row group is
// transposed through the warp's 1 KB staging buffer so a lane stores 16
// contiguous bytes of one routed row.
template<bool GateUp, int Groups, int OutN>
__device__ __forceinline__ void store_tile(
    const ptx::D4 (&accum)[2][2 * Groups], int groups, const int32_t* route_rows,
    const float* route_weights, uint8_t* staging, uint16_t* output, int column_base, int lane) {
  constexpr int WarpColumns = GateUp ? 16 : 32;
  constexpr int Chunks = WarpColumns / 8;
  // The XOR spreads the eight rows of each 8x8 matrix over all banks.
  auto staged = [&](int token, int chunk) {
    const int swizzled = GateUp ? chunk ^ ((token >> 2) & 1) : chunk ^ ((token >> 1) & 3);
    return staging + token * WarpColumns * 2 + swizzled * 16;
  };
#pragma unroll
  for (int p = 0; p < Groups; ++p) {
    if (p >= groups) break;
#pragma unroll
    for (int m = 0; m < (GateUp ? 1 : 2); ++m) {
      uint32_t packed[4];
#pragma unroll
      for (int q = 0; q < 2; ++q) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          float v0 = accum[m][2 * p + q].v[2 * h];
          float v1 = accum[m][2 * p + q].v[2 * h + 1];
          if constexpr (GateUp) {
            v0 = v0 / (1.0f + expf(-v0)) * accum[1][2 * p + q].v[2 * h];
            v1 = v1 / (1.0f + expf(-v1)) * accum[1][2 * p + q].v[2 * h + 1];
          } else {
            const int token = 16 * p + 8 * q + 2 * (lane & 3);
            v0 *= route_weights[token];
            v1 *= route_weights[token + 1];
          }
          packed[2 * q + h] = pack_bf16(v0, v1);
        }
      }
      const int matrix = lane >> 3;
      ptx::stmatrix_trans_b16<4>(
          staged((matrix >> 1) * 8 + (lane & 7), 2 * m + (matrix & 1)), packed);
    }
    __syncwarp();
#pragma unroll
    for (int it = 0; it < 16 * Chunks / 32; ++it) {
      const int index = it * 32 + lane;
      const int token = index / Chunks;
      const int chunk = index % Chunks;
      const uint4 value = *reinterpret_cast<const uint4*>(staged(token, chunk));
      const int routed = route_rows[16 * p + token];
      if (routed >= 0)
        *reinterpret_cast<uint4*>(output + uint64_t(routed) * OutN + column_base + chunk * 8) =
            value;
    }
    __syncwarp();
  }
}

} // namespace chop::moe::grouped
