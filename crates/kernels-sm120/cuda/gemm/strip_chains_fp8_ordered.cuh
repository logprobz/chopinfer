#pragma once

// `run_a16_strips_fp8` (strip_chains_fp8.cuh, `DenseKernel::StripChains`
// over FP8 128x128 weights) with its ring stages released only after every
// read of them, as `run_a16_strips_ordered` (strip_chains_ordered.cuh) does
// for BF16 weights: each math lane runs `fence.proxy.async.shared::cta`
// after its last shared read of a stage and before the warp's release, so
// the producer's next TMA write of the stage cannot overtake those reads.
// Everything else, and so every output bit, is `run_a16_strips_fp8`'.

#include "strip_chains_fp8.cuh"
#include "../common/ptx_control.cuh"

namespace chop::gemm {

template<bool Residual, int Strips, int TileN, int Stages,
         Fp8StripsMutation Mutation = Fp8StripsMutation::none>
__device__ void run_a16_strips_fp8_ordered(
    const CUtensorMap& map_a, const CUtensorMap& map_b,
    void* output, float* chains, uint32_t* semaphores,
    uint32_t live_m, uint32_t n, uint32_t k, uint32_t splits,
    uint32_t output_f32, const float* scales, const void* residual,
    const float* weight_global_scale) {
  using T = Fp8StripTraits<Strips, TileN, Stages>;
  static_assert(T::shared_bytes <= 101376);
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
      if constexpr (Mutation == Fp8StripsMutation::late_wait) load_activations(step);
    }
  }
  if constexpr (Mutation == Fp8StripsMutation::none ||
                Mutation == Fp8StripsMutation::shared_slots ||
                Mutation == Fp8StripsMutation::keep_ticket)
    ptx::griddep_wait();
  const uint32_t scale_k = (k + 127) / 128;
  float* tile_chains = chains +
      uint64_t(Mutation == Fp8StripsMutation::shared_slots ? 0 : tile) * chunks *
          T::tile_m * TileN;
  if (producer) {
    if constexpr (Mutation != Fp8StripsMutation::late_wait)
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
    // `packed_fp8_weight`'s bytes: rows strip + group and strip + group + 8
    // (a0/a2 and a1/a3), K kk + 2 * lane4 (a0, a1) and kk + 8 + 2 * lane4
    // (a2, a3), in 16-byte sector kk / 16 of the 128-byte row, which the
    // TMA swizzle moves to sector (kk / 16) ^ ((row + base phase) & 7). The
    // two rows differ by 8 and so share the phase.
    const uint32_t row0 = uint32_t(strip + group);
    const uint32_t weights_address = ptx::shared_address(weights);
    const uint32_t phase = (row0 + weights_address / 128) & 7;
    const uint32_t a_lo = weights_address + row0 * T::weight_row_bytes + 2 * lane4;
    const uint32_t a_hi = a_lo + 8 * T::weight_row_bytes;
    // ldmatrix x2 rows: lanes 0-7 address the group's tokens at K 0, lanes
    // 8-15 at K 8 (b0, b1), in the unswizzled 256-byte activation rows.
    const uint32_t b_offset = uint32_t(lane & 7) * T::activation_row_bytes +
                              uint32_t((lane >> 3) & 1) * 16;
    // `mma_chunk` scales by the strip's first weight row, zero past the
    // matrix (whose outputs are never written).
    const uint32_t scale_row = row_a + strip;
    const float* row_scales = scales + uint64_t(scale_row / 128) * scale_k;
    const bool scaled = scale_row < n;
    ptx::D4 acc[T::groups];
#pragma unroll
    for (int j = 0; j < T::groups; ++j)
      acc[j] = ptx::D4{};
    for (uint32_t step = 0; step < count; ++step) {
      const int stage = int(step % Stages);
      const uint32_t a = uint32_t(stage * T::weight_bytes);
      const unsigned char* b = activations + stage * T::activation_bytes;
      const float scale = scaled ? row_scales[first + step] : 0.0f;
      pipe.consumer_wait(step);
      ptx::D4 raw[T::groups];
#pragma unroll
      for (int j = 0; j < T::groups; ++j)
        raw[j] = ptx::D4{};
#pragma unroll
      for (int kk = 0; kk < T::tile_k; kk += 16) {
        const uint32_t sector = uint32_t((kk / 16) ^ phase) * 16;
        ptx::A4 af;
        af.v[0] = convert::unpack_bf16x2_e4m3(fp8_pair_shared(a + a_lo + sector));
        af.v[1] = convert::unpack_bf16x2_e4m3(fp8_pair_shared(a + a_hi + sector));
        af.v[2] = convert::unpack_bf16x2_e4m3(fp8_pair_shared(a + a_lo + sector + 8));
        af.v[3] = convert::unpack_bf16x2_e4m3(fp8_pair_shared(a + a_hi + sector + 8));
#pragma unroll
        for (int j = 0; j < T::groups; ++j) {
          ptx::B2 bf;
          ptx::ldmatrix_b16<2>(bf.v, b + j * 8 * T::activation_row_bytes +
                                         b_offset + kk * 2);
          ptx::mma_bf16(raw[j], af, bf);
        }
      }
      // Every lane's shared reads of the stage (weight codes and ldmatrix
      // activations) are performed before the async proxy may refill it.
      ptx::fence_async_shared();
      __syncwarp();
      if (lane == 0) pipe.consumer_release(step);
      if (splits == 1) {
#pragma unroll
        for (int j = 0; j < T::groups; ++j)
#pragma unroll
          for (int v = 0; v < 4; ++v)
            acc[j].v[v] = __fmaf_rn(raw[j].v[v], scale, acc[j].v[v]);
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
  if constexpr (Mutation == Fp8StripsMutation::late_wait) ptx::griddep_wait();
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
          // The column's 128-row scale block, which its strip's first row
          // shares.
          const float* column_scales = scales + uint64_t(column / 128) * scale_k;
          float sum = 0.0f;
          // Loads run a batch ahead of the adds, which keep K order.
          constexpr uint32_t Batch = 16;
          for (uint32_t base = 0; base < chunks; base += Batch) {
            float chain[Batch];
            float scale[Batch];
#pragma unroll
            for (uint32_t i = 0; i < Batch; ++i)
              if (base + i < chunks) {
                chain[i] = __ldcg(tile_chains + uint64_t(base + i) * tile_elements + index);
                scale[i] = column_scales[base + i];
              }
#pragma unroll
            for (uint32_t i = 0; i < Batch; ++i)
              if (base + i < chunks) sum = __fmaf_rn(chain[i], scale[i], sum);
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
        if constexpr (Mutation != Fp8StripsMutation::keep_ticket)
          atomicExch(&semaphores[tile], 0);
      }
    }
  }
  __syncthreads();
  if (producer) pipe.invalidate();
}

} // namespace chop::gemm
