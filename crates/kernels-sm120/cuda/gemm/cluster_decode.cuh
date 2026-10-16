#pragma once

// Shared dense cluster decode template for M <= 32 over fragment-major
// weights: BF16, or FP8 E4M3 with F32 scales per 128-K block. It covers
// run_fragment_decode_cluster (fragment_decode_cluster.cuh) and Qwen3.8's
// run_cluster_decode (qwen38.cuh) with one arithmetic:
//
// A CTA owns Strips 16-row strips over one K chunk. Its eight warps split as
// Strips strips by 8 / Strips K parts; each warp holds WarpBlocks 128-K blocks
// of fragments in registers, streamed two groups of four tiles ahead of its
// MMAs and started before the PDL wait. After the wait one thread bulk-copies
// the chunk's activation rows into padded shared rows read with ldmatrix. Each
// 128-K block accumulates from zero (a chain of at most eight MMAs), then BF16
// adds it into the warp's F32 accumulator and FP8 adds it times its scale.
// The CTAs covering one row tile's K form a cluster of at most eight; each
// publishes its F32 partial tile to its workspace slice (distributed shared
// memory loads are slower than L2 here), and after one cluster barrier each
// rank sums ceil(rows / splits) output rows over ranks and K parts in K
// order, from zero. No semaphore is used and the sum order is fixed.
//
// Up to four weights that read the same activation run as one launch
// (Segments): their strips are numbered consecutively and each writes its
// own output rows. With Swiglu the single segment holds rows[0] / 2 gate rows
// then as many up rows; a CTA's first Strips / 2 strips are gate strips and
// the rest the matching up strips, and each rank writes SiLU(gate) * up. With
// Passes = 2 the slab holds half the token tile and the second half is staged
// after the first is consumed, reusing the weight registers. The Epilogue
// stores each finished value (Store below by default).

#include "fragment_decode.cuh"
#include "prefill_mainloop.cuh"
#include "../common/ptx_control.cuh"

namespace chop::gemm::cluster {

template<WeightKind W, int TileN, int KBlocks, int Strips, int WarpBlocks,
         int Passes>
struct Traits {
  static_assert(W == WeightKind::bf16 || fragment_fp8<W>);
  static_assert(TileN == 8 || TileN == 16 || TileN == 32);
  static_assert(Passes == 1 || (Passes == 2 && TileN >= 16));
  static_assert(Strips == 1 || Strips == 2 || Strips == 4 || Strips == 8);
  static constexpr int pass_rows = TileN / Passes;
  static constexpr int warps = 8;
  static constexpr int k_parts = warps / Strips;
  static constexpr int tiles_per_block = W == WeightKind::bf16 ? 8 : 4;
  static constexpr int tiles = WarpBlocks * tiles_per_block;
  static_assert(tiles % 4 == 0 && tiles <= 32, "whole groups of four in registers");
  static constexpr int chunk_blocks = WarpBlocks * k_parts;
  static_assert(KBlocks % chunk_blocks == 0);
  static constexpr int splits = KBlocks / chunk_blocks;
  static_assert(splits <= 8);
  static constexpr int rows = Strips * 16;
  // Padding the row by 16 bytes spreads eight token rows over all banks.
  static constexpr int activation_stride = chunk_blocks * 256 + 16;
  static constexpr int activation_bytes = pass_rows * activation_stride;
  static constexpr int partial_bytes = warps * 16 * TileN * 4;
  static constexpr int shared_bytes = activation_bytes + partial_bytes;
};

// Segment s has rows[s] output rows in ceil(rows[s] / 16) fragment strips of
// weight[s] and writes output[s][token * stride[s] + row].
template<int Count>
struct Segments {
  static_assert(Count >= 1 && Count <= 4);
  const uint8_t* weight[Count];
  void* output[Count];
  uint32_t rows[Count];
  uint32_t stride[Count];

  __device__ __forceinline__ uint32_t strips() const {
    uint32_t total = 0;
#pragma unroll
    for (int i = 0; i < Count; ++i) total += (rows[i] + 15) / 16;
    return total;
  }

  // The segment of global strip g and its first global strip.
  __device__ __forceinline__ int find(uint32_t g, uint32_t& first) const {
    int segment = 0;
    uint32_t start = 0;
    first = 0;
#pragma unroll
    for (int i = 0; i < Count; ++i) {
      if (g >= start) {
        segment = i;
        first = start;
      }
      start += (rows[i] + 15) / 16;
    }
    return segment;
  }
};

// The default epilogue: F32, or BF16 rounded to nearest even.
struct Store {
  __device__ __forceinline__ void check(uint32_t) const {}
  __device__ __forceinline__ void operator()(
      void* output, uint64_t index, uint32_t, uint32_t, int, float value,
      uint32_t output_f32) const {
    if (output_f32) static_cast<float*>(output)[index] = value;
    else static_cast<__nv_bfloat16*>(output)[index] = __float2bfloat16(value);
  }
};

// `scales` holds FP8 weights' F32 scales, one per fragment_scale_strips<W>
// strips and 128-K block; BF16 ignores it.
template<WeightKind W, int TileN, int KBlocks, int Strips, int WarpBlocks,
         int Passes, int Count, bool Swiglu, class Epilogue = Store>
__device__ void run(
    const Segments<Count> segments, const uint16_t* activation,
    float* workspace, uint32_t live_m, uint32_t output_f32,
    const float* scales = nullptr, const Epilogue epilogue = {}) {
  using T = Traits<W, TileN, KBlocks, Strips, WarpBlocks, Passes>;
  constexpr bool fp8 = fragment_fp8<W>;
  static_assert(!Swiglu || (Count == 1 && Strips >= 2));
  static_assert(!fp8 || Count == 1, "one scale grid per launch");
  epilogue.check(output_f32);
  constexpr int R = T::pass_rows;
  constexpr bool clustered = T::splits > 1;
  extern __shared__ __align__(16) unsigned char storage[];
  auto* partials = reinterpret_cast<float*>(storage + T::activation_bytes);
  const uint32_t k = KBlocks * 128;
  const uint32_t rank = clustered ? ptx::cluster_ctarank() : 0;
  const uint32_t row_tile = blockIdx.x / T::splits;
  const uint32_t first_block = rank * T::chunk_blocks;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int local_strip = warp % Strips;
  const int k_part = warp / Strips;
  constexpr int half = Strips / 2;
  const uint32_t gate_strips = segments.strips() / 2;
  const uint32_t strip = !Swiglu ? row_tile * Strips + local_strip
      : local_strip < half ? row_tile * half + local_strip
      : gate_strips + row_tile * half + local_strip - half;
  const bool valid = Swiglu
      ? row_tile * half + local_strip % half < gate_strips
      : strip < segments.strips();
  const uint32_t warp_block = first_block + k_part * WarpBlocks;
  uint32_t segment_first;
  const int segment = segments.find(strip, segment_first);
  const uint8_t* weight = segments.weight[0];
#pragma unroll
  for (int i = 1; i < Count; ++i)
    if (segment == i) weight = segments.weight[i];

  ptx::A4 cached[T::tiles];
  __shared__ uint64_t activation_ready;
  const uint8_t* lane_base = weight +
      (uint64_t(strip - segment_first) * KBlocks + warp_block) *
          T::tiles_per_block * 512 +
      lane * 16;
  constexpr int groups = T::tiles / 4;
  // Two groups of four fragment tiles per warp stay in flight. Deeper
  // prefetch queues the activation copy behind weight traffic.
  constexpr int lookahead = 2 < groups ? 2 : groups;
  constexpr int block_groups = T::tiles_per_block / 4;
  const auto load_weights = [&](int first, int end) {
    if (!valid) return;
#pragma unroll
    for (int group = first; group < end; ++group)
      load_weight_group4(lane_base + group * 2048, cached[group * 4],
          cached[group * 4 + 1], cached[group * 4 + 2], cached[group * 4 + 3]);
  };
  // Rows past live M read as zero.
  const auto zero_rows = [&](uint32_t live) {
    for (uint32_t i = threadIdx.x; i < (R - live) * T::chunk_blocks * 16;
         i += blockDim.x)
      *reinterpret_cast<uint4*>(storage +
          (live + i / (T::chunk_blocks * 16)) * T::activation_stride +
          (i % (T::chunk_blocks * 16)) * 16) = make_uint4(0, 0, 0, 0);
  };
  const auto pass_live = [&](int pass) {
    return live_m > uint32_t(pass * R) ? min(live_m - pass * R, uint32_t(R)) : 0u;
  };
  const auto stage = [&](int pass) {
    const uint32_t live = pass_live(pass);
    ptx::mbarrier_arrive_expect_tx(&activation_ready,
        live * T::chunk_blocks * 256);
    for (uint32_t row = 0; row < live; ++row)
      ptx::cp_async_bulk(storage + row * T::activation_stride,
          activation + uint64_t(pass * R + row) * k + first_block * 128,
          T::chunk_blocks * 256, &activation_ready);
  };
  zero_rows(pass_live(0));
  if (threadIdx.x == 0) {
    ptx::mbarrier_init(&activation_ready, 1);
    ptx::fence_mbarrier_init();
  }
  __syncthreads();
  ptx::griddep_launch_dependents();
  if (warp != 0) load_weights(0, lookahead);
  ptx::griddep_wait();
  if (threadIdx.x == 0) stage(0);
  if (warp == 0) load_weights(0, lookahead);

  ptx::D4 acc[TileN / 8];
#pragma unroll
  for (int j = 0; j < TileN / 8; ++j) acc[j] = ptx::D4{};
  const uint32_t b_row = (lane & 7) + ((R >= 16 ? lane >> 4 : 0) << 3);
  const uint32_t b_address = ptx::shared_address(storage) +
      b_row * T::activation_stride + ((lane >> 3) & 1) * 16 +
      k_part * WarpBlocks * 256;
#pragma unroll
  for (int pass = 0; pass < Passes; ++pass) {
    if (pass > 0) {
      if (pass_live(pass) == 0) break;
      // Every warp has consumed the previous half before it is replaced.
      __syncthreads();
      zero_rows(pass_live(pass));
      __syncthreads();
      if (threadIdx.x == 0) stage(pass);
    }
    ptx::mbarrier_wait(&activation_ready, pass & 1);
    if (!valid) continue;
#pragma unroll
    for (int local = 0; local < WarpBlocks; ++local) {
      // Keep lookahead groups in flight ahead of the block being consumed.
      if (pass == 0) {
        const int first = local * block_groups + lookahead;
        const int end = (local + 1) * block_groups + lookahead;
        load_weights(first < groups ? first : groups, end < groups ? end : groups);
      }
      ptx::D4 block[R / 8];
#pragma unroll
      for (int j = 0; j < R / 8; ++j) block[j] = ptx::D4{};
#pragma unroll
      for (int step = 0; step < 8; ++step) {
        const uint32_t column = (local * 8 + step) * 32;
        ptx::B2 b[R / 8];
        if constexpr (R == 8) {
          asm volatile(
              "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
              : "=r"(b[0].v[0]), "=r"(b[0].v[1])
              : "r"(b_address + column));
        } else {
#pragma unroll
          for (int pair = 0; pair < R / 16; ++pair)
            prefill::load_b2(b_address + pair * 16 * T::activation_stride +
                column, 0, 0, b[pair * 2], b[pair * 2 + 1]);
        }
        ptx::A4 a;
        if constexpr (fp8) {
          const ptx::A4& packed = cached[local * 4 + step / 2];
          a = (step & 1) ? prefill::fp8_to_bf16<1>(packed)
                         : prefill::fp8_to_bf16<0>(packed);
        } else {
          a = cached[local * 8 + step];
        }
#pragma unroll
        for (int j = 0; j < R / 8; ++j) ptx::mma_bf16(block[j], a, b[j]);
      }
      if constexpr (fp8) {
        const float scale = scales[uint64_t(strip / fragment_scale_strips<W>) *
            KBlocks + warp_block + local];
#pragma unroll
        for (int j = 0; j < R / 8; ++j)
#pragma unroll
          for (int v = 0; v < 4; ++v)
            acc[pass * (R / 8) + j].v[v] += block[j].v[v] * scale;
      } else {
#pragma unroll
        for (int j = 0; j < R / 8; ++j)
#pragma unroll
          for (int v = 0; v < 4; ++v) acc[pass * (R / 8) + j].v[v] += block[j].v[v];
      }
    }
  }

  // partials[k_part][tile row][token], tile row = local strip * 16 + row.
  float* published = clustered
      ? workspace + uint64_t(blockIdx.x) * T::partial_bytes / 4
      : partials;
  const uint32_t group = lane >> 2;
  const uint32_t lane4 = lane & 3;
#pragma unroll
  for (int j = 0; j < TileN / 8; ++j)
#pragma unroll
    for (int h = 0; h < 2; ++h)
      *reinterpret_cast<float2*>(published +
          ((k_part * T::rows + local_strip * 16 + group + h * 8) * TileN +
           j * 8 + lane4 * 2)) =
          make_float2(acc[j].v[h * 2], acc[j].v[h * 2 + 1]);
  if constexpr (clustered) {
    ptx::cluster_arrive_release();
    ptx::cluster_wait_acquire();
  } else {
    __syncthreads();
  }

  // Each rank sums its output rows over ranks, then K parts, in K order.
  constexpr int out_rows = Swiglu ? T::rows / 2 : T::rows;
  constexpr int rank_rows = (out_rows + T::splits - 1) / T::splits;
  constexpr int outputs = rank_rows * TileN;
  constexpr int per_thread_outputs = (outputs + 255) / 256;
  constexpr int parts = T::splits * T::k_parts;
  float values[per_thread_outputs][parts];
  float ups[per_thread_outputs][Swiglu ? parts : 1];
  // Issue every load before the fixed-order sums.
#pragma unroll
  for (int e = 0; e < per_thread_outputs; ++e) {
    const uint32_t index = threadIdx.x + e * 256;
    const uint32_t tile_row = rank * rank_rows + index / TileN;
    const uint32_t token = index % TileN;
    if (index >= outputs || tile_row >= out_rows) continue;
#pragma unroll
    for (int source = 0; source < T::splits; ++source)
#pragma unroll
      for (int part = 0; part < T::k_parts; ++part) {
        const uint32_t offset = (part * T::rows + tile_row) * TileN + token;
        const float* base = clustered
            ? workspace + uint64_t(blockIdx.x - rank + source) * T::partial_bytes / 4
            : partials;
        values[e][source * T::k_parts + part] = clustered
            ? __ldcg(base + offset) : base[offset];
        if constexpr (Swiglu)
          ups[e][source * T::k_parts + part] = clustered
              ? __ldcg(base + offset + out_rows * TileN)
              : base[offset + out_rows * TileN];
      }
  }
#pragma unroll
  for (int e = 0; e < per_thread_outputs; ++e) {
    const uint32_t index = threadIdx.x + e * 256;
    const uint32_t tile_row = rank * rank_rows + index / TileN;
    const uint32_t token = index % TileN;
    if (index >= outputs || tile_row >= out_rows || token >= live_m) continue;
    int output_segment = 0;
    void* output = segments.output[0];
    uint32_t row, rows = segments.rows[0], stride = segments.stride[0];
    if constexpr (Swiglu) {
      row = row_tile * out_rows + tile_row;
      rows /= 2;
    } else {
      const uint32_t global = row_tile * Strips + tile_row / 16;
      if (global >= segments.strips()) continue;
      uint32_t first;
      const int s = segments.find(global, first);
      output_segment = s;
      row = (global - first) * 16 + tile_row % 16;
#pragma unroll
      for (int i = 1; i < Count; ++i)
        if (s == i) {
          output = segments.output[i];
          rows = segments.rows[i];
          stride = segments.stride[i];
        }
    }
    if (row >= rows) continue;
    float value = 0.0f, up = 0.0f;
#pragma unroll
    for (int part = 0; part < parts; ++part) {
      value += values[e][part];
      if constexpr (Swiglu) up += ups[e][part];
    }
    if constexpr (Swiglu) value = (value / (1.0f + expf(-value))) * up;
    epilogue(output, uint64_t(token) * stride + row, token, row, output_segment,
        value, output_f32);
  }
}

} // namespace chop::gemm::cluster
