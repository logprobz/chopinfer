#pragma once

// The tiled A16 GEMM (`run_a16` in dense.cuh) with its K chunks split
// across CTAs and every MMA chain kept (`DenseKernel::TiledChains`).
//
// Split s of a tile runs chunks [chunks * s / splits, chunks * (s + 1) /
// splits) through the same TMA pipeline, warp tiles and `mma_tile` calls as
// `run_a16`, and stores each chain's temporary (one per 128-K block of a
// chunk: 64 K for BF16's 64-K chunks, 128 K for FP8; the raw F32
// chain from zero, before an FP8 block scale) to the workspace, one slot
// per (tile, chunk, block) whatever the split count. The last CTA of a tile
// (an integer ticket, no float atomics) then adds the temporaries of every
// chunk and block into a zero accumulator in K order with `run_a16`'s
// operation: `acc += chain * scale`, one FMA, with the FP8 128x128 block
// scale, else `acc += chain`. NVFP4's global scale, the residual and the
// output conversion follow as in `run_a16`. So each output element is
// bitwise `run_a16`'s at one split, at every split count, live row count
// and grid; splits only spread the chains over more CTAs.

#include "dense.cuh"

namespace chop::gemm {

// Deliberate defects that kernel-lab instantiates to show its PDL, workspace
// and ticket checks fail on them (`tiled_chain_splits`). Served entries take
// the default, `none`, whose code is the unmodified kernel.
enum class ChainsMutation : int {
  none = 0,
  drop_wait = 1,     // no griddepcontrol.wait
  late_wait = 2,     // the wait after the K loop, before the ticket
  shared_slots = 3,  // every tile writes tile 0's temporaries
  keep_ticket = 4,   // the ticket is never reset
};

template<WeightKind W, bool Decode, bool Residual, int TileM, int TileN, int TileK,
         int Stages = 2, ChainsMutation Mutation = ChainsMutation::none>
__device__ void run_a16_chains(
    const CUtensorMap& map_a, const CUtensorMap& map_b,
    void* output, float* chains, uint32_t* semaphores,
    uint32_t live_m, uint32_t n, uint32_t k, uint32_t splits,
    uint32_t output_f32, const float* scales, const void* residual,
    const float* weight_global_scale) {
  using T = A16Traits<W, Decode, TileM, TileN, TileK, Stages>;
  static_assert(T::shared_bytes <= 101376);
  // `mma_chunk`'s 128-K blocks of one chunk; FP8 128x128 chunks are one
  // block, whose scale multiplies its chain.
  constexpr int Blocks = (TileK + 127) / 128;
  static_assert(W != WeightKind::fp8_block128 || TileK == 128);
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
  const uint32_t chunks = (k + TileK - 1) / TileK;
  const uint64_t tile_elements = uint64_t(TileM) * TileN;
  const uint32_t initial_tile = blockIdx.x / splits;
  ptx::griddep_launch_dependents();
  if (threadIdx.x == 0 && initial_tile < tile_count) {
    const uint32_t first = (uint64_t(chunks) * (blockIdx.x % splits)) / splits;
    const int32_t weight_coord[2] = {
        int32_t(first * TileK),
        int32_t(Decode ? (initial_tile / tiles_b) * TileM :
                         (initial_tile % tiles_b) * TileN)};
    ptx::prefetch_tensor_l2(Decode ? &map_a : &map_b, weight_coord);
  }
  if constexpr (Mutation != ChainsMutation::drop_wait &&
                Mutation != ChainsMutation::late_wait)
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
  const uint32_t first = (uint64_t(chunks) * split) / splits;
  const uint32_t end = (uint64_t(chunks) * (split + 1)) / splits;
  const uint64_t policy = ptx::l2_policy<ptx::L2Evict::first>();
  float* tile_chains = chains +
      uint64_t(Mutation == ChainsMutation::shared_slots ? 0 : tile) * chunks * Blocks *
          tile_elements;
  for (int wave = 0; wave < T::waves; ++wave) {
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
      const int group = (threadIdx.x & 31) >> 2;
      const int lane4 = threadIdx.x & 3;
#pragma unroll
      for (int i = 0; i < T::per_warp; ++i) {
        const int subtile = wave * T::math_warps * T::per_warp + warp +
                            i * T::math_warps;
        if (subtile < T::subtiles) {
          const int sm = (subtile / (TileN / 8)) * 16;
          const int sn = (subtile % (TileN / 8)) * 8;
#pragma unroll
          for (int block = 0; block < Blocks; ++block) {
            // `mma_chunk`'s chain: from zero over the block's K16 steps.
            ptx::D4 raw{};
            for (int kk = block * 128; kk < block * 128 + 128 && kk < TileK; kk += 16)
              mma_tile<W, Decode, TileK>(raw, &a[stage][0][0], &b[stage][0][0],
                  sm, sn, kk, row_a, row_b, chunk * TileK, n, k, scales,
                  weight_global_scale);
            float* slot = tile_chains + (uint64_t(chunk) * Blocks + block) * tile_elements;
#pragma unroll
            for (int v = 0; v < 4; ++v)
              slot[(sm + group + (v / 2) * 8) * TileN + sn + lane4 * 2 + (v % 2)] =
                  raw.v[v];
          }
        }
      }
      if ((threadIdx.x & 31) == 0) pipe.consumer_release(step);
    }
  }
  }
  if constexpr (Mutation == ChainsMutation::late_wait) ptx::griddep_wait();
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    last = atomicAdd(&semaphores[tile], 1) == splits - 1;
  }
  __syncthreads();
  if (last) {
    __threadfence();
    for (uint32_t index = threadIdx.x; index < tile_elements; index += blockDim.x) {
      const uint32_t ma = row_a + index / TileN;
      const uint32_t nb = row_b + index % TileN;
      const uint32_t token = Decode ? nb : ma;
      const uint32_t column = Decode ? ma : nb;
      if (token < live_m && column < n) {
        // `mma_chunk` scales by its 16-row strip's first weight row, which
        // shares this row's 128-row scale block.
        const float* row_scales = W == WeightKind::fp8_block128 ?
            scales + uint64_t(Decode ? ma : nb) / 128 * ((k + 127) / 128) : nullptr;
        const uint32_t temporaries = chunks * Blocks;
        float sum = 0.0f;
        // Loads run a batch ahead of the adds, which keep K order.
        constexpr uint32_t Batch = 16;
        for (uint32_t base = 0; base < temporaries; base += Batch) {
          float chain[Batch];
          float scale[Batch];
#pragma unroll
          for (uint32_t i = 0; i < Batch; ++i) {
            if (base + i < temporaries) {
              chain[i] = __ldcg(tile_chains + uint64_t(base + i) * tile_elements + index);
              if constexpr (W == WeightKind::fp8_block128) scale[i] = row_scales[base + i];
            }
          }
#pragma unroll
          for (uint32_t i = 0; i < Batch; ++i) {
            if (base + i < temporaries) {
              if constexpr (W == WeightKind::fp8_block128)
                sum = __fmaf_rn(chain[i], scale[i], sum);
              else
                sum = __fadd_rn(sum, chain[i]);
            }
          }
        }
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
      if constexpr (Mutation != ChainsMutation::keep_ticket)
        atomicExch(&semaphores[tile], 0);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) pipe.invalidate();
  __syncthreads();
  }
}

} // namespace chop::gemm
