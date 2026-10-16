#pragma once

// The tiled A16 decode GEMM over BF16 weights with `run_a16`'s one-split
// arithmetic, in a schedule for 1 to 32 rows (`DenseKernel::StripChains`).
//
// A CTA holds `Strips` 16-row weight strips (16 * Strips output columns),
// every token row of the bucket (TileN, one tile) and one of `splits` K
// partitions: chunks [chunks * s / splits, chunks * (s + 1) / splits) of
// 64 K. A producer warp streams its chunks through a `Stages`-deep TMA ring,
// each stage one chunk's weight rows and activation rows in `run_a16`'s
// swizzled tiles. It issues the weight loads of the first stages before the
// grid dependency wait (no earlier kernel writes the weights) and every
// activation load after it.
//
// Math warp w owns strip w. Per chunk and 8-token group it runs `mma_tile`'s
// chain: four m16n8k16 MMAs from zero over the chunk's K16 steps in order,
// with the operand registers `mma_tile` packs, read here with ldmatrix from
// the same swizzled tiles (weight rows are the MMA's M, tokens its N).
//
// At one split each warp adds its chains into a zero accumulator in K order
// (`run_a16`'s `accum += raw`) and writes `run_a16`'s epilogue. Over more
// splits each chain is stored to the workspace, one slot per (tile, chunk)
// whatever the split count, and the tile's last CTA (an integer ticket, no
// float atomics) adds them from zero in K order and writes the same
// epilogue, as `run_a16_chains` does. Either way each output element is
// bitwise `run_a16`'s at one split, at every split count, strip count, live
// row count and grid.

#include "dense.cuh"
#include "../common/ptx_matrix.cuh"

namespace chop::gemm {

// Deliberate defects that kernel-lab instantiates to show its PDL, workspace
// and ticket checks fail on them (`strip_chain_mutants`). Served entries take
// the default, `none`, whose code is the unmodified kernel.
enum class StripsMutation : int {
  none = 0,
  drop_wait = 1,     // no griddepcontrol.wait
  late_wait = 2,     // activation loads before the wait, after the K loop
  shared_slots = 3,  // every tile writes tile 0's temporaries
  keep_ticket = 4,   // the ticket is never reset
};

template<int Strips, int TileN, int Stages>
struct StripTraits {
  static_assert(Strips == 1 || Strips == 2 || Strips == 4);
  static_assert(TileN == 8 || TileN == 16 || TileN == 32);
  static_assert(Stages >= 2 && Stages <= 16);
  static constexpr int tile_m = 16 * Strips;
  static constexpr int tile_k = 64;
  // One BF16 row of a chunk: the 128-byte swizzle span.
  static constexpr int row_bytes = tile_k * 2;
  static constexpr int groups = TileN / 8;
  static constexpr int weight_bytes = tile_m * row_bytes;
  static constexpr int activation_bytes = TileN * row_bytes;
  static constexpr int stage_bytes = weight_bytes + activation_bytes;
  static constexpr int shared_bytes = Stages * stage_bytes;
  static constexpr int threads = (Strips + 1) * 32;
};

// The 16-byte sector of a swizzled 128-byte-row tile that holds `row`'s
// eight BF16 values from `k` (a multiple of 8): where `packed_bf16` reads them.
__device__ __forceinline__ const unsigned char* strip_sector(
    const unsigned char* tile, int row, int k) {
  return tile + smem::swizzle_128(tile, row * 128 + k * 2, 128);
}

template<bool Residual, int Strips, int TileN, int Stages,
         StripsMutation Mutation = StripsMutation::none>
__device__ void run_a16_strips(
    const CUtensorMap& map_a, const CUtensorMap& map_b,
    void* output, float* chains, uint32_t* semaphores,
    uint32_t live_m, uint32_t n, uint32_t k, uint32_t splits,
    uint32_t output_f32, const float* scales, const void* residual,
    const float* weight_global_scale) {
  using T = StripTraits<Strips, TileN, Stages>;
  static_assert(T::shared_bytes <= 101376);
  (void)scales;
  (void)weight_global_scale;
  __shared__ TmaPipeline<Stages, Strips> pipe;
  __shared__ uint32_t last;
  extern __shared__ __align__(128) unsigned char storage[];
  unsigned char* weights = align_tma_storage(storage);
  unsigned char* activations = weights + Stages * T::weight_bytes;
  const int warp = int(threadIdx.x / 32);
  const int lane = int(threadIdx.x & 31);
  const uint32_t tiles = (n + T::tile_m - 1) / T::tile_m;
  const uint32_t tile = blockIdx.x / splits;
  const uint32_t split = blockIdx.x % splits;
  ptx::griddep_launch_dependents();
  if (tile >= tiles) return;
  const uint32_t row_a = tile * T::tile_m;
  const uint32_t chunks = (k + T::tile_k - 1) / T::tile_k;
  const uint32_t first = (uint64_t(chunks) * split) / splits;
  const uint32_t end = (uint64_t(chunks) * (split + 1)) / splits;
  const uint32_t count = end - first;
  const uint32_t ahead = count < uint32_t(Stages) ? count : uint32_t(Stages);
  const bool producer = threadIdx.x == 0;
  if (producer) pipe.init();
  __syncthreads();
  if (warp > 0 && lane == 0) pipe.prime_empty();
  const uint64_t streamed = ptx::l2_policy<ptx::L2Evict::first>();
  const uint64_t shared_rows = ptx::l2_policy<ptx::L2Evict::normal>();
  auto load_weights = [&](uint32_t step, uint64_t* bar) {
    const int32_t coord[2] = {int32_t((first + step) * T::tile_k), int32_t(row_a)};
    ptx::tma_load_hint<2>(weights + (step % Stages) * T::weight_bytes, &map_a,
                          coord, bar, streamed);
  };
  auto load_activations = [&](uint32_t step) {
    const int32_t coord[2] = {int32_t((first + step) * T::tile_k), 0};
    ptx::tma_load_hint<2>(activations + (step % Stages) * T::activation_bytes,
                          &map_b, coord, &pipe.full[step % Stages], shared_rows);
  };
  if (producer) {
    for (uint32_t step = 0; step < ahead; ++step) {
      pipe.producer_acquire(step);
      load_weights(step, pipe.producer_commit(step, T::stage_bytes));
      if constexpr (Mutation == StripsMutation::late_wait) load_activations(step);
    }
  }
  if constexpr (Mutation == StripsMutation::none ||
                Mutation == StripsMutation::shared_slots ||
                Mutation == StripsMutation::keep_ticket)
    ptx::griddep_wait();
  float* tile_chains = chains +
      uint64_t(Mutation == StripsMutation::shared_slots ? 0 : tile) * chunks *
          T::tile_m * TileN;
  if (producer) {
    if constexpr (Mutation != StripsMutation::late_wait)
      for (uint32_t step = 0; step < ahead; ++step) load_activations(step);
    for (uint32_t step = ahead; step < count; ++step) {
      pipe.producer_acquire(step);
      load_weights(step, pipe.producer_commit(step, T::stage_bytes));
      load_activations(step);
    }
  } else if (warp > 0) {
    const int strip = (warp - 1) * 16;
    const int group = lane >> 2;
    const int lane4 = lane & 3;
    // ldmatrix rows: x4 lanes 0-15 address rows 0-15 at K 0, lanes 16-31
    // the same rows at K 8 (a0..a3 of `mma_tile`); x2 lanes 0-7 address the
    // group's tokens at K 0, lanes 8-15 at K 8 (b0, b1).
    const int a_row = strip + (lane & 15);
    const int a_k = (lane >> 4) * 8;
    const int b_row = lane & 7;
    const int b_k = ((lane >> 3) & 1) * 8;
    ptx::D4 acc[T::groups];
#pragma unroll
    for (int j = 0; j < T::groups; ++j)
      acc[j] = ptx::D4{};
    for (uint32_t step = 0; step < count; ++step) {
      const int stage = int(step % Stages);
      const unsigned char* a = weights + stage * T::weight_bytes;
      const unsigned char* b = activations + stage * T::activation_bytes;
      pipe.consumer_wait(step);
      ptx::D4 raw[T::groups];
#pragma unroll
      for (int j = 0; j < T::groups; ++j)
        raw[j] = ptx::D4{};
#pragma unroll
      for (int kk = 0; kk < T::tile_k; kk += 16) {
        ptx::A4 af;
        ptx::ldmatrix_b16<4>(af.v, strip_sector(a, a_row, kk + a_k));
#pragma unroll
        for (int j = 0; j < T::groups; ++j) {
          ptx::B2 bf;
          ptx::ldmatrix_b16<2>(bf.v, strip_sector(b, j * 8 + b_row, kk + b_k));
          ptx::mma_bf16(raw[j], af, bf);
        }
      }
      __syncwarp();
      if (lane == 0) pipe.consumer_release(step);
      if (splits == 1) {
#pragma unroll
        for (int j = 0; j < T::groups; ++j)
#pragma unroll
          for (int v = 0; v < 4; ++v)
            acc[j].v[v] = __fadd_rn(acc[j].v[v], raw[j].v[v]);
      } else {
        float* slot = tile_chains + uint64_t(first + step) * T::tile_m * TileN;
#pragma unroll
        for (int j = 0; j < T::groups; ++j)
#pragma unroll
          for (int half = 0; half < 2; ++half)
            *reinterpret_cast<float2*>(
                slot + (strip + group + half * 8) * TileN + j * 8 + lane4 * 2) =
                make_float2(raw[j].v[half * 2], raw[j].v[half * 2 + 1]);
      }
    }
    if (splits == 1) {
#pragma unroll
      for (int j = 0; j < T::groups; ++j)
#pragma unroll
        for (int v = 0; v < 4; ++v) {
          const uint32_t column = row_a + strip + group + (v / 2) * 8;
          const uint32_t token = j * 8 + lane4 * 2 + (v % 2);
          if (token < live_m && column < n) {
            float value = acc[j].v[v];
            const uint64_t offset = uint64_t(token) * n + column;
            if constexpr (Residual) value += residual_value(residual, offset, output_f32);
            if (output_f32) static_cast<float*>(output)[offset] = value;
            else static_cast<__nv_bfloat16*>(output)[offset] = __float2bfloat16(value);
          }
        }
    }
  }
  if constexpr (Mutation == StripsMutation::late_wait) ptx::griddep_wait();
  if (splits > 1) {
    __threadfence();
    __syncthreads();
    if (producer) last = atomicAdd(&semaphores[tile], 1) == splits - 1;
    __syncthreads();
    if (last) {
      __threadfence();
      constexpr uint32_t tile_elements = T::tile_m * TileN;
      for (uint32_t index = threadIdx.x; index < tile_elements; index += T::threads) {
        const uint32_t column = row_a + index / TileN;
        const uint32_t token = index % TileN;
        if (token < live_m && column < n) {
          float sum = 0.0f;
          // Loads run a batch ahead of the adds, which keep K order.
          constexpr uint32_t Batch = 32;
          for (uint32_t base = 0; base < chunks; base += Batch) {
            float chain[Batch];
#pragma unroll
            for (uint32_t i = 0; i < Batch; ++i)
              if (base + i < chunks)
                chain[i] = __ldcg(tile_chains + uint64_t(base + i) * tile_elements + index);
#pragma unroll
            for (uint32_t i = 0; i < Batch; ++i)
              if (base + i < chunks) sum = __fadd_rn(sum, chain[i]);
          }
          const uint64_t offset = uint64_t(token) * n + column;
          if constexpr (Residual) sum += residual_value(residual, offset, output_f32);
          if (output_f32) static_cast<float*>(output)[offset] = sum;
          else static_cast<__nv_bfloat16*>(output)[offset] = __float2bfloat16(sum);
        }
      }
      __syncthreads();
      if (producer) {
        __threadfence();
        if constexpr (Mutation != StripsMutation::keep_ticket)
          atomicExch(&semaphores[tile], 0);
      }
    }
  }
  __syncthreads();
  if (producer) pipe.invalidate();
}

} // namespace chop::gemm
