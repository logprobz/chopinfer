#pragma once

// `run_a16_strips` (strip_chains.cuh, `DenseKernel::StripChains` over BF16
// weights) with its ring stages released only after every read of them.
//
// That kernel's math warps release a stage (`consumer_release`, an mbarrier
// arrive) right after issuing the step's ldmatrix reads, with no fence
// between those generic-proxy shared reads and the producer's next TMA
// write of the stage, which is in the async proxy. ptxas schedules the
// arrive before the last ldmatrix reads have returned (the MMAs that wait
// on them follow it), so the producer can refill the stage while a warp
// still reads it, and that warp's chain for the step takes data of a later
// chunk. Two-split entries at the vocabulary head's shape (BF16 38720 x
// 4096, 1210 CTAs) stored such chains in most launches (FIX-SC1).
//
// Here each math lane runs `fence.proxy.async.shared::cta` after its last
// read of the stage and before the warp's release, as the PTX memory model
// requires between generic accesses and async-proxy accesses of the same
// shared memory; the fence waits for the reads. Everything else, and so
// every output bit, is `run_a16_strips`': the same chains, slots, ticket
// and fold, and the same mutations for kernel-lab.

#include "strip_chains.cuh"
#include "../common/ptx_control.cuh"

namespace chop::gemm {

template<bool Residual, int Strips, int TileN, int Stages,
         StripsMutation Mutation = StripsMutation::none>
__device__ void run_a16_strips_ordered(
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
      // Every lane's ldmatrix reads of the stage are performed before the
      // async proxy may refill it.
      ptx::fence_async_shared();
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
