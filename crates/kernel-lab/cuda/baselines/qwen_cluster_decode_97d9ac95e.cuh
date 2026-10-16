// Lab-only reference: Qwen3.8's cluster decode as served at main 97d9ac95e
// (kernels-sm120/cuda/gemm/qwen38.cuh), before the shared cluster decode
// template (gemm/cluster_decode.cuh) replaced it. Kernel-lab builds it as
// the chop_gemm_eng_qwen38_* cubins and compares the served template entries
// with it bit for bit (tests/dense_template.rs). Keep it unchanged: it is the
// evidence that the template reproduces the kernel, not code to maintain.
#pragma once

#include "gemm/fragment_decode.cuh"
#include "gemm/fragment_prefill.cuh"
#include "common/ptx_control.cuh"
#include "common/convert.cuh"
#include "common/sigmoid.cuh"

namespace chop::gemm::qwen38 {

// BF16 decode for M <= 32, generalized from run_fragment_decode_cluster in
// fragment_decode_cluster.cuh. Up to four weights that read the same
// activation run as one launch (DecodeSegments): their strips are numbered
// consecutively and each writes its own output rows. A CTA owns Strips 16-row strips over one K
// chunk; its eight warps split as Strips strips by 8 / Strips K parts, and
// each warp holds WarpBlocks 128-K blocks (8 fragment tiles each) in
// registers, loaded before the PDL wait. One thread bulk-copies the chunk's
// activation rows into padded shared rows read by ldmatrix. The CTAs that
// cover one row tile's K form a cluster of at most eight; each publishes its
// F32 partial tile to its workspace slice, and after one cluster barrier
// each rank sums ceil(rows / splits) output rows over ranks and K parts in K
// order. MiMo's kernel fixes WarpBlocks at 2 and needs splits to divide the
// tile rows, which no Qwen K allows. With Passes = 2 the slab holds half the
// tokens and the second half is staged after the first is consumed, reusing
// the weight registers, so a long K chunk fits shared memory.
template<int TileN, int KBlocks, int Strips, int WarpBlocks, int Passes>
struct ClusterDecodeTraits {
  static_assert(TileN == 8 || TileN == 16 || TileN == 32);
  static_assert(Passes == 1 || (Passes == 2 && TileN >= 16));
  static constexpr int pass_rows = TileN / Passes;
  static_assert(Strips == 1 || Strips == 2 || Strips == 4 || Strips == 8);
  static_assert(WarpBlocks >= 1 && WarpBlocks <= 4);
  static constexpr int warps = 8;
  static constexpr int k_parts = warps / Strips;
  static constexpr int tiles = WarpBlocks * 8;
  static constexpr int chunk_blocks = WarpBlocks * k_parts;
  static_assert(KBlocks % chunk_blocks == 0);
  static constexpr int splits = KBlocks / chunk_blocks;
  static_assert(splits <= 8);
  static constexpr int rows = Strips * 16;
  static constexpr int rank_rows = (rows + splits - 1) / splits;
  // Padding the row by 16 bytes spreads eight token rows over all banks.
  static constexpr int activation_stride = chunk_blocks * 256 + 16;
  static constexpr int activation_bytes = pass_rows * activation_stride;
  static constexpr int partial_bytes = warps * 16 * TileN * 4;
  static constexpr int shared_bytes = activation_bytes + partial_bytes;
};

// Segment s has rows[s] output rows in ceil(rows[s] / 16) fragment strips of
// weight[s] and writes output[s][token * stride[s] + row].
template<int Count>
struct DecodeSegments {
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

// With Swiglu the single segment holds rows[0] / 2 gate rows then as many up
// rows (concatenated halves). A CTA's first Strips / 2 strips are gate
// strips and the rest the matching up strips, so each rank writes
// SiLU(gate) * up for its output rows, rows[0] / 2 wide.
template<int TileN, int KBlocks, int Strips, int WarpBlocks, int Passes,
         int Count, bool Swiglu = false, bool HcBottleneck = false>
__device__ void run_cluster_decode(
    const DecodeSegments<Count> segments, const uint16_t* activation,
    float* workspace, uint32_t live_m, uint32_t output_f32,
    uint16_t* bottleneck = nullptr) {
  static_assert(!HcBottleneck || (KBlocks == 80 && !Swiglu && (Count == 1 || Count == 2)));
  if constexpr (HcBottleneck) if (output_f32) asm volatile("trap;");
  using T = ClusterDecodeTraits<TileN, KBlocks, Strips, WarpBlocks, Passes>;
  static_assert(!Swiglu || (Count == 1 && Strips >= 2));
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
      (uint64_t(strip - segment_first) * KBlocks + warp_block) * 8 * 512 +
      lane * 16;
  constexpr int groups = T::tiles / 4;
  constexpr int lookahead = 2 < groups ? 2 : groups;
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
      if (pass == 0) {
        const int first = local * 2 + lookahead;
        const int end = (local + 1) * 2 + lookahead;
        load_weights(first < groups ? first : groups, end < groups ? end : groups);
      }
      // Each 128-K block accumulates from zero, then adds into acc.
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
#pragma unroll
        for (int j = 0; j < R / 8; ++j)
          ptx::mma_bf16(block[j], cached[local * 8 + step], b[j]);
      }
#pragma unroll
      for (int j = 0; j < R / 8; ++j)
#pragma unroll
        for (int v = 0; v < 4; ++v) acc[pass * (R / 8) + j].v[v] += block[j].v[v];
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
    const uint64_t out = uint64_t(token) * stride + row;
    if (output_f32) static_cast<float*>(output)[out] = value;
    else if constexpr (HcBottleneck) {
      // Round the down projection before activation, exactly as its consumer.
      const uint16_t rounded = chop::convert::bf16_from_f32(value);
      static_cast<uint16_t*>(output)[out] = rounded;
      if (output_segment == 0) {
        const float v = chop::convert::bf16_to_f32(rounded) / 4;
        bottleneck[uint64_t(token) * 320 + row] =
            chop::convert::bf16_from_f32(chop::sigmoid::times(v, v));
      }
    } else static_cast<__nv_bfloat16*>(output)[out] = __float2bfloat16(value);
  }
}

} // namespace chop::gemm::qwen38
