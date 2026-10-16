#pragma once

#include "fragment_decode.cuh"
#include "prefill_mainloop.cuh"
#include "../common/ptx_control.cuh"

namespace chop::gemm {

// Decode for M <= 32. A CTA owns Strips 16-row weight strips over one K
// chunk. Its eight warps split as Strips strips by 8 / Strips K parts; each
// warp streams 16 fragment tiles into registers, two groups of four ahead
// of the MMAs, starting before the PDL wait. After the wait one thread
// copies the chunk's activation rows with the bulk-copy engine into padded
// shared rows, read with conflict-free ldmatrix, so all Strips strips share
// one activation slab. The splits CTAs that cover one row tile's K form a
// thread-block cluster (splits may be 1). Each clustered CTA writes its F32
// partial tile to its own workspace slice (distributed shared memory loads
// are much slower than L2 on this part); after one cluster barrier every
// rank sums a slice of output rows over all ranks and K parts in K order.
// No semaphore is used and the sum order is fixed.
template<WeightKind W, int TileN, bool Swiglu, int KBlocks, int Strips>
struct ClusterDecodeTraits {
  static_assert(W == WeightKind::bf16 || fragment_fp8<W>);
  static_assert(TileN == 8 || TileN == 16 || TileN == 32);
  static_assert(Strips == 1 || Strips == 2 || Strips == 4 || Strips == 8);
  static_assert(!Swiglu || Strips >= 2);
  static constexpr int warps = 8;
  static constexpr int k_parts = warps / Strips;
  static constexpr int tiles_per_block = W == WeightKind::bf16 ? 8 : 4;
  static constexpr int tiles = 16;
  static constexpr int warp_blocks = tiles / tiles_per_block;
  static constexpr int chunk_blocks = warp_blocks * k_parts;
  static_assert(KBlocks % chunk_blocks == 0);
  static constexpr int splits = KBlocks / chunk_blocks;
  static_assert(splits <= 8);
  static constexpr int rows = Strips * 16;
  static constexpr int output_rows = Swiglu ? rows / 2 : rows;
  static_assert(output_rows % splits == 0);
  static constexpr int rank_rows = output_rows / splits;
  // Padding the row by 16 bytes spreads eight token rows over all banks.
  static constexpr int activation_stride = chunk_blocks * 256 + 16;
  static constexpr int activation_bytes = TileN * activation_stride;
  static constexpr int partial_bytes = warps * 16 * TileN * 4;
  static constexpr int shared_bytes = activation_bytes + partial_bytes;
};

template<WeightKind W, int TileN, bool Swiglu, int KBlocks, int Strips>
__device__ void run_fragment_decode_cluster(
    const uint8_t* weight, const uint16_t* activation,
    void* output, float* workspace, uint32_t* semaphores,
    uint32_t live_m, uint32_t n, uint32_t k, uint32_t splits,
    uint32_t output_f32, const float* scales, const void* residual) {
  using T = ClusterDecodeTraits<W, TileN, Swiglu, KBlocks, Strips>;
  constexpr bool fp8 = fragment_fp8<W>;
  constexpr bool clustered = T::splits > 1;
  (void)semaphores;
  (void)residual;
  (void)splits;
  extern __shared__ __align__(16) unsigned char storage[];
  auto* partials = reinterpret_cast<float*>(storage + T::activation_bytes);
  const uint32_t output_rows = Swiglu ? n / 2 : n;
  const uint32_t strips_total = n / 16;
  const uint32_t rank = clustered ? ptx::cluster_ctarank() : 0;
  const uint32_t row_tile = blockIdx.x / T::splits;
  const uint32_t first_block = rank * T::chunk_blocks;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int local_strip = warp % Strips;
  const int k_part = warp / Strips;
  // Swiglu tiles pair Strips / 2 gate strips with the matching up strips.
  constexpr int half = Strips / 2;
  const uint32_t strip = Swiglu
      ? (local_strip < half ? row_tile * half + local_strip
                            : output_rows / 16 + row_tile * half +
                              local_strip - half)
      : row_tile * Strips + local_strip;
  const bool valid = strip < strips_total && k == KBlocks * 128;
  const uint32_t warp_block = first_block + k_part * T::warp_blocks;

  ptx::A4 cached[T::tiles];
  __shared__ uint64_t activation_ready;
  const uint8_t* lane_base = weight +
      (uint64_t(strip) * KBlocks + warp_block) * T::tiles_per_block * 512 +
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
  const uint32_t live_rows = min(live_m, uint32_t(TileN));
  for (uint32_t i = threadIdx.x; i < (TileN - live_rows) * T::chunk_blocks * 16;
       i += blockDim.x)
    *reinterpret_cast<uint4*>(storage +
        (live_rows + i / (T::chunk_blocks * 16)) * T::activation_stride +
        (i % (T::chunk_blocks * 16)) * 16) = make_uint4(0, 0, 0, 0);
  if (threadIdx.x == 0) {
    ptx::mbarrier_init(&activation_ready, 1);
    ptx::fence_mbarrier_init();
  }
  __syncthreads();
  ptx::griddep_launch_dependents();
  if (warp != 0) load_weights(0, lookahead);
  ptx::griddep_wait();
  if (threadIdx.x == 0) {
    ptx::mbarrier_arrive_expect_tx(&activation_ready,
        live_rows * T::chunk_blocks * 256);
    for (uint32_t token = 0; token < live_rows; ++token)
      ptx::cp_async_bulk(storage + token * T::activation_stride,
          activation + uint64_t(token) * k + first_block * 128,
          T::chunk_blocks * 256, &activation_ready);
  }
  if (warp == 0) load_weights(0, lookahead);
  ptx::mbarrier_wait(&activation_ready, 0);

  ptx::D4 acc[TileN / 8];
#pragma unroll
  for (int j = 0; j < TileN / 8; ++j) acc[j] = ptx::D4{};
  if (valid) {
    const uint32_t b_row = (lane & 7) + ((TileN >= 16 ? lane >> 4 : 0) << 3);
    const uint32_t b_address = ptx::shared_address(storage) +
        b_row * T::activation_stride + ((lane >> 3) & 1) * 16 +
        k_part * T::warp_blocks * 256;
#pragma unroll
    for (int local = 0; local < T::warp_blocks; ++local) {
      // Keep lookahead groups in flight ahead of the block being consumed.
      {
        const int first = local * block_groups + lookahead;
        const int end = (local + 1) * block_groups + lookahead;
        load_weights(first < groups ? first : groups, end < groups ? end : groups);
      }
      ptx::D4 block[TileN / 8];
#pragma unroll
      for (int j = 0; j < TileN / 8; ++j) block[j] = ptx::D4{};
#pragma unroll
      for (int step = 0; step < 8; ++step) {
        const uint32_t column = (local * 8 + step) * 32;
        ptx::B2 b[TileN / 8];
        if constexpr (TileN == 8) {
          asm volatile(
              "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
              : "=r"(b[0].v[0]), "=r"(b[0].v[1])
              : "r"(b_address + column));
        } else {
#pragma unroll
          for (int pair = 0; pair < TileN / 16; ++pair)
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
        for (int j = 0; j < TileN / 8; ++j) ptx::mma_bf16(block[j], a, b[j]);
      }
      // Each 128-K block accumulates from zero into `block`; FP8 promotes it
      // by its F32 scale and BF16 adds it.
      if constexpr (fp8) {
        const float scale = scales[uint64_t(strip / fragment_scale_strips<W>) * KBlocks +
            warp_block + local];
#pragma unroll
        for (int j = 0; j < TileN / 8; ++j)
#pragma unroll
          for (int v = 0; v < 4; ++v) acc[j].v[v] += block[j].v[v] * scale;
      } else {
#pragma unroll
        for (int j = 0; j < TileN / 8; ++j)
#pragma unroll
          for (int v = 0; v < 4; ++v) acc[j].v[v] += block[j].v[v];
      }
    }
  }

  // partials[k_part][tile row][token], tile row = local strip * 16 + row.
  // Clustered CTAs exchange them through their global workspace slice: on
  // this part distributed shared memory loads are much slower than L2.
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
  constexpr int outputs = T::rank_rows * TileN;
  constexpr int per_thread_outputs = (outputs + 255) / 256;
  constexpr int parts = T::splits * T::k_parts;
  float gate_parts[per_thread_outputs][parts];
  float up_parts[per_thread_outputs][Swiglu ? parts : 1];
  // Issue every load before the fixed-order sums.
#pragma unroll
  for (int e = 0; e < per_thread_outputs; ++e) {
    const uint32_t index = threadIdx.x + e * 256;
    if (index >= outputs) continue;
    const uint32_t tile_row = rank * T::rank_rows + index / TileN;
    const uint32_t token = index % TileN;
#pragma unroll
    for (int source = 0; source < T::splits; ++source) {
#pragma unroll
      for (int part = 0; part < T::k_parts; ++part) {
        const uint32_t offset =
            ((part * T::rows + tile_row) * TileN + token) * 4;
        const uint32_t up_offset = offset + T::output_rows * TileN * 4;
        if constexpr (clustered) {
          const float* peer = workspace + uint64_t(blockIdx.x - rank + source) *
              T::partial_bytes / 4;
          gate_parts[e][source * T::k_parts + part] = __ldcg(peer + offset / 4);
          if constexpr (Swiglu)
            up_parts[e][source * T::k_parts + part] =
                __ldcg(peer + up_offset / 4);
        } else {
          gate_parts[e][part] = partials[offset / 4];
          if constexpr (Swiglu) up_parts[e][part] = partials[up_offset / 4];
        }
      }
    }
  }
#pragma unroll
  for (int e = 0; e < per_thread_outputs; ++e) {
    const uint32_t index = threadIdx.x + e * 256;
    const uint32_t token = index % TileN;
    const uint32_t row = row_tile * T::output_rows + rank * T::rank_rows +
        index / TileN;
    if (index >= outputs || token >= live_m || row >= output_rows) continue;
    float value = 0.0f, up = 0.0f;
#pragma unroll
    for (int part = 0; part < parts; ++part) {
      value += gate_parts[e][part];
      if constexpr (Swiglu) up += up_parts[e][part];
    }
    if constexpr (Swiglu) value = (value / (1.0f + expf(-value))) * up;
    const uint64_t out = uint64_t(token) * output_rows + row;
    if (output_f32) static_cast<float*>(output)[out] = value;
    else static_cast<__nv_bfloat16*>(output)[out] = __float2bfloat16(value);
  }
}

} // namespace chop::gemm
