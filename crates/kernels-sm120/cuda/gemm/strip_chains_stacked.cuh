#pragma once

// Several strip-chain GEMMs over BF16 weights that read the same activation
// rows, in one launch (`gemm::stacked`): each part's output elements are
// bitwise `run_a16_strips`' (so `run_a16`'s at one split) of the part alone.
//
// Part p is a GEMM of n_p weight rows (its own tensor map, `run_a16_strips`'
// tiles) whose first width_p output columns land at
// output_p[token * stride_p + column]: in place of the part's contiguous
// `[rows, n_p]` output and a layout copy of those columns into a strided
// buffer (several parts writing thirds of one buffer, or a part whose
// weight rows the format pads writing its rank's columns only). The parts'
// tiles are numbered in part order; CTA b takes tile b / splits and K split
// b % splits, and loads, multiplies, adds and folds its chains exactly as
// `run_a16_strips` does for that tile of that part alone (same chunks, K
// order, ldmatrix fragments, MMA chain from zero, `__fadd_rn` from zero at
// one split, one chain slot per (tile, chunk) and a last-CTA ticket fold
// over more), so the split count, strip count, part order and live row
// count change no output bit.
//
// `Prefetch`: before the grid dependency wait, the producer also asks L2
// for every weight chunk of its K range that the ring does not stage ahead
// (`cp.async.bulk.prefetch.tensor`, the loads' own boxes), so weights
// stream from DRAM while the kernels this launch depends on still run. A
// prefetch moves no data the kernel reads and changes no output bit.

#include "strip_chains.cuh"

namespace chop::gemm {

// Deliberate defects that kernel-lab instantiates to show its checks fail
// on them (`strip_chains_stacked`'s mutants). Served entries take `none`.
enum class StackedMutation : int {
  none = 0,
  drop_wait = 1,     // no griddepcontrol.wait
  late_wait = 2,     // activation loads before the wait, after the K loop
  shared_slots = 3,  // every tile writes tile 0's temporaries
  keep_ticket = 4,   // the ticket is never reset
  first_map = 5,     // every part reads the first part's weights
};

// The parts' runtime geometry, from the entry's parameters.
template<int Parts>
struct StackedParts {
  static_assert(Parts >= 1 && Parts <= 6);
  const CUtensorMap* maps[Parts];
  void* outputs[Parts];
  uint32_t n[Parts];
  uint32_t width[Parts];
  uint32_t stride[Parts];
};

// The part that holds stacked tile `tile` and its first tile, by unrolled
// selects (no local-memory indexing).
template<int Parts, int TileM, StackedMutation Mutation>
struct StackedPart {
  const CUtensorMap* map = nullptr;
  void* output = nullptr;
  uint32_t n = 0, width = 0, stride = 0, first = 0;
  bool found = false;

  __device__ __forceinline__ StackedPart(const StackedParts<Parts>& parts, uint32_t tile) {
    uint32_t start = 0;
#pragma unroll
    for (int p = 0; p < Parts; ++p) {
      const uint32_t tiles = (parts.n[p] + TileM - 1) / TileM;
      if (!found && tile < start + tiles) {
        map = Mutation == StackedMutation::first_map ? parts.maps[0] : parts.maps[p];
        output = parts.outputs[p];
        n = parts.n[p];
        width = parts.width[p];
        stride = parts.stride[p];
        first = start;
        found = true;
      }
      start += tiles;
    }
  }
};

template<int Parts, int Strips, int TileN, int Stages, bool Prefetch,
         StackedMutation Mutation = StackedMutation::none>
__device__ void run_a16_strips_stacked(
    const StackedParts<Parts>& parts, const CUtensorMap& map_b,
    float* chains, uint32_t* semaphores,
    uint32_t live_m, uint32_t k, uint32_t splits, uint32_t output_f32) {
  using T = StripTraits<Strips, TileN, Stages>;
  static_assert(T::shared_bytes <= 101376);
  __shared__ TmaPipeline<Stages, Strips> pipe;
  __shared__ uint32_t last;
  extern __shared__ __align__(128) unsigned char storage[];
  unsigned char* weights = align_tma_storage(storage);
  unsigned char* activations = weights + Stages * T::weight_bytes;
  const int warp = int(threadIdx.x / 32);
  const int lane = int(threadIdx.x & 31);
  const uint32_t tile = blockIdx.x / splits;
  const uint32_t split = blockIdx.x % splits;
  ptx::griddep_launch_dependents();
  const StackedPart<Parts, T::tile_m, Mutation> part(parts, tile);
  if (!part.found) return;
  const CUtensorMap* map_a = part.map;
  const uint32_t row_a = (tile - part.first) * T::tile_m;
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
    ptx::tma_load_hint<2>(weights + (step % Stages) * T::weight_bytes, map_a,
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
      if constexpr (Mutation == StackedMutation::late_wait) load_activations(step);
    }
    if constexpr (Prefetch) {
      for (uint32_t step = ahead; step < count; ++step) {
        const int32_t coord[2] = {int32_t((first + step) * T::tile_k), int32_t(row_a)};
        ptx::prefetch_tensor_l2(map_a, coord);
      }
    }
  }
  if constexpr (Mutation != StackedMutation::drop_wait &&
                Mutation != StackedMutation::late_wait)
    ptx::griddep_wait();
  float* tile_chains = chains +
      uint64_t(Mutation == StackedMutation::shared_slots ? 0 : tile) * chunks *
          T::tile_m * TileN;
  if (producer) {
    if constexpr (Mutation != StackedMutation::late_wait)
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
    // ldmatrix rows as `run_a16_strips` addresses them.
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
          if (token < live_m && column < part.width) {
            const float value = acc[j].v[v];
            const uint64_t offset = uint64_t(token) * part.stride + column;
            if (output_f32) static_cast<float*>(part.output)[offset] = value;
            else static_cast<__nv_bfloat16*>(part.output)[offset] = __float2bfloat16(value);
          }
        }
    }
  }
  if constexpr (Mutation == StackedMutation::late_wait) ptx::griddep_wait();
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
        if (token < live_m && column < part.width) {
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
          const uint64_t offset = uint64_t(token) * part.stride + column;
          if (output_f32) static_cast<float*>(part.output)[offset] = sum;
          else static_cast<__nv_bfloat16*>(part.output)[offset] = __float2bfloat16(sum);
        }
      }
      __syncthreads();
      if (producer) {
        __threadfence();
        if constexpr (Mutation != StackedMutation::keep_ticket)
          atomicExch(&semaphores[tile], 0);
      }
    }
  }
  __syncthreads();
  if (producer) pipe.invalidate();
}

} // namespace chop::gemm
