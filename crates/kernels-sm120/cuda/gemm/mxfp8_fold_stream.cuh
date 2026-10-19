#pragma once
#include "mxfp8_fold_ticket.cuh"
#include "weight_prefetch.cuh"

namespace chop::gemm::mxfp8_fold_stream {

// A scheduler supplies virtual coordinates and its own shared arena. Only
// the generated entry wrappers use physical CTA coordinates or griddep.
struct Task { uint32_t x, y, z, nx, ny; };
struct __align__(16) Shared {
  uint8_t codes[2][16 * 128];
  uint8_t scales[2][16 * 4];
  bool last_tile, last_row_tile, admitted;
};
struct Fragments { uint4 lo[2], hi[2]; uint32_t scales[2]; };

__device__ __forceinline__ uint32_t acquire(const uint32_t* address) {
  uint32_t value;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
  return value;
}
__device__ __forceinline__ uint32_t arrive(uint32_t* address) {
  uint32_t value;
  asm volatile("atom.add.acq_rel.gpu.u32 %0, [%1], 1;" : "=r"(value) : "l"(address) : "memory");
  return value;
}

template<int N, int K, int Groups, bool Paired>
__device__ void initialize_task(uint32_t cta, uint32_t ctas,
    const uint8_t* weights, const uint8_t* weights2, uint8_t* packed,
    uint8_t* packed2, uint32_t* state, uint32_t state_words) {
  constexpr int Folds = (K + 127) / 128;
  constexpr uint64_t Words = uint64_t(Groups) * N * K / 4;
  const uint64_t tid = uint64_t(cta) * blockDim.x + threadIdx.x;
  const uint64_t stride = uint64_t(ctas) * blockDim.x;
  for (uint64_t i = tid; i < Words; i += stride) {
    const uint32_t within = i % (2 * K);
    const uint32_t fold = within / 256;
    const uint32_t words_per_lane = K % 128 && fold == Folds - 1 ? 4 : 8;
    const uint32_t word = (within % 256) % words_per_lane;
    const uint32_t lane = (within % 256) / words_per_lane;
    const uint32_t tile = i / (2 * K);
    const uint32_t row = tile * 8 + lane / 4;
    const uint32_t k = fold * 128 + (word / 2) * 32 + (word % 2) * 16 + (lane % 4) * 4;
    const uint64_t at = uint64_t(row) * K + k;
    reinterpret_cast<uint32_t*>(packed)[i] = k < K ? *reinterpret_cast<const uint32_t*>(weights + at) : 0;
    if constexpr (Paired)
      reinterpret_cast<uint32_t*>(packed2)[i] = k < K ? *reinterpret_cast<const uint32_t*>(weights2 + at) : 0;
  }
  for (uint64_t i = tid; i < state_words; i += stride) state[i] = 0;
}

// Keep immutable fragment reads before the wrapper's dependency wait.
// A readonly intrinsic can otherwise be scheduled after that wait.
__device__ __forceinline__ uint4 fragment16(const uint8_t* address) {
  uint4 value;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
      : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
      : "l"(address) : "memory");
  return value;
}

template<int N, int K, int Groups, int TileN, bool Paired>
__device__ __forceinline__ Fragments weight_fragments(Task task, uint32_t base,
    uint32_t end, uint32_t splits, const uint8_t* packed, const uint8_t* scales,
    const uint8_t* packed2, const uint8_t* scales2) {
  constexpr int Folds = (K + 127) / 128;
  const uint32_t bank = task.z / (splits * Groups);
  const uint32_t g = (task.z / splits) % Groups;
  const uint32_t warp = threadIdx.x / 32;
  const uint32_t lane = threadIdx.x % 32;
  const uint32_t tile = task.x * (TileN / 8) + warp;
  const uint32_t row = tile * 8 + lane / 4;
  if constexpr (Paired) if (bank) { packed = packed2; scales = scales2; }
  Fragments result{};
#pragma unroll
  for (int p = 0; p < 2; ++p) {
    result.scales[p] = 0x7f7f7f7fu;
    if (base + p < end && row < N) {
      const bool tail = K % 128 && base + p == Folds - 1;
      const uint64_t at = (uint64_t(g) * (N / 8) + tile) * (8 * K) + (base + p) * 1024 + lane * (tail ? 16 : 32);
      result.lo[p] = fragment16(packed + at);
      if (!tail) result.hi[p] = fragment16(packed + at + 16);
      const uint64_t s = (uint64_t(g) * N + row) * (K / 32) + (base + p) * 4;
      if constexpr (K % 128 == 0) result.scales[p] = __ldg(reinterpret_cast<const uint32_t*>(scales + s));
      else {
#pragma unroll
        for (int j = 0; j < 4; ++j) if ((base + p) * 128 + j * 32 < K)
          result.scales[p] = (result.scales[p] & ~(255u << (8 * j))) | (uint32_t(scales[s + j]) << (8 * j));
      }
    }
  }
  return result;
}

template<int N, int K, int Groups, int TileN, bool F32Output, bool Paired>
__device__ void merge_task(Task task, uint32_t bank, uint32_t g,
    void* output, void* output2, uint8_t* workspace, uint32_t live, uint32_t rows) {
  constexpr int Folds = (K + 127) / 128;
  constexpr int Threads = TileN * 4;
  const uint32_t token_base = task.y * 16;
  const uint32_t column_base = task.x * TileN;
  const auto* failed = reinterpret_cast<const uint32_t*>(workspace);
  auto* partials = reinterpret_cast<float*>(workspace + mxfp8_fold_ticket::header<N, Groups, TileN, Paired>(rows));
  partials += uint64_t(bank) * Folds * rows * Groups * N;
  if constexpr (Paired) if (bank) output = output2;
  for (uint32_t i = threadIdx.x; i < 16 * TileN; i += Threads) {
    const uint32_t row = token_base + i / TileN;
    const uint32_t col = column_base + i % TileN;
    if (row >= live || col >= N) continue;
    float total = 0.0f;
    if (!failed[row]) {
      const uint64_t at = (uint64_t(row) * Groups + g) * N + col;
      for (uint32_t base = 0; base < Folds; base += 16) {
        float chain[16];
#pragma unroll
        for (int j = 0; j < 16; ++j)
          if (base + j < Folds) chain[j] = __ldcg(partials + uint64_t(base + j) * rows * Groups * N + at);
#pragma unroll
        for (int j = 0; j < 16; ++j)
          if (base + j < Folds) total = __fadd_rn(total, chain[j]);
      }
    }
    const uint64_t at = (uint64_t(row) * Groups + g) * N + col;
    if constexpr (F32Output) reinterpret_cast<float*>(output)[at] = total;
    else reinterpret_cast<uint16_t*>(output)[at] = convert::bf16_from_f32(total);
  }
}

template<int N, int K, int Groups, int RowStride, int GroupStride,
         int TileN, bool F32Output, bool Paired, bool DeferAdmission = false>
__device__ void run_task(Task task, Shared& shared, Fragments current,
    const uint8_t* activations, const uint8_t* activation_scales,
    const uint8_t* packed, const uint8_t* weight_scales,
    const uint8_t* packed2, const uint8_t* weight_scales2,
    void* output, void* output2, uint8_t* workspace, uint32_t* state,
    const uint32_t* live_rows, uint32_t* status, uint64_t status_row_offset,
    uint32_t rows, uint32_t folds_per_cta) {
  constexpr int Folds = (K + 127) / 128;
  constexpr int Threads = TileN * 4;
  constexpr int TilesPerRow = ((N + TileN - 1) / TileN) * Groups * (Paired ? 2 : 1);
  const uint32_t splits = (Folds + folds_per_cta - 1) / folds_per_cta;
  const uint32_t bank = task.z / (splits * Groups);
  const uint32_t g = (task.z / splits) % Groups;
  const uint32_t first = (task.z % splits) * folds_per_cta;
  const uint32_t end = min(first + folds_per_cta, uint32_t(Folds));
  const uint32_t live = min(rows, *live_rows);
  const uint32_t token_base = task.y * 16;
  if (token_base >= live) return;
  auto* ready = state + task.y * 16;
  auto* done = ready + 1;
  auto* tickets = state + 64;
  auto* failed = reinterpret_cast<uint32_t*>(workspace);
  auto* partials = reinterpret_cast<float*>(workspace + mxfp8_fold_ticket::header<N, Groups, TileN, Paired>(rows));
  partials += uint64_t(bank) * Folds * rows * Groups * N;
  // Exactly one virtual tile per row tile checks all groups. No other
  // task reads its flags until the release below is acquired. Computation
  // may run ahead; rejected rows are discarded before any output store.
  if (task.x == 0 && task.z == 0) {
    if constexpr (DeferAdmission) {
      const uint64_t begin = clock64();
      while (clock64() - begin < 1000000) __nanosleep(64);
    }
    for (uint32_t row = token_base; row < min(token_base + 16, live); ++row) {
      bool invalid = false;
      for (int group = 0; group < Groups; ++group) {
        for (uint32_t k = threadIdx.x * 16; k < K; k += Threads * 16) {
          const uint64_t at = uint64_t(row) * RowStride + uint64_t(group) * GroupStride + k;
          const uint4 code = *reinterpret_cast<const uint4*>(activations + at);
          const uint8_t scale = activation_scales[at / 32];
          invalid |= !(mxfp8_fold_ticket::admitted4(code.x, scale) && mxfp8_fold_ticket::admitted4(code.y, scale) &&
                       mxfp8_fold_ticket::admitted4(code.z, scale) && mxfp8_fold_ticket::admitted4(code.w, scale));
        }
      }
      const bool bad = __syncthreads_or(invalid);
      if (threadIdx.x == 0) {
        const uint32_t previous = atomicOr(status + status_row_offset + row, bad ? 1u : 0u);
        failed[row] = bad || previous != 0;
      }
    }
    if (threadIdx.x == 0) {
      __threadfence();
      asm volatile("st.release.gpu.global.u32 [%0], 1;" :: "l"(ready) : "memory");
    }
  }
  const uint32_t warp = threadIdx.x / 32;
  const uint32_t quad = (threadIdx.x & 31) / 4;
  const uint32_t lane4 = threadIdx.x & 3;
  for (uint32_t base = first; base < end; base += 2) {
    // Four independent 16-byte fragment loads per lane overlap the current
    // fold pair's activation copies and MMA instructions.
    Fragments next = weight_fragments<N, K, Groups, TileN, Paired>(task, base + 2, end, splits,
        packed, weight_scales, packed2, weight_scales2);
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      for (uint32_t chunk = threadIdx.x; chunk < 16 * 8; chunk += Threads) {
        const uint32_t row = token_base + chunk / 8;
        const uint32_t k = (base + p) * 128 + (chunk % 8) * 16;
        const bool valid = base + p < end && row < live && k < K;
        const uint64_t at = uint64_t(row) * RowStride + uint64_t(g) * GroupStride + k;
        mxfp8_fold_ticket::copy16(shared.codes[p] + chunk * 16, activations + (valid ? at : 0), valid);
      }
      for (uint32_t i = threadIdx.x; i < 64; i += Threads) {
        const uint32_t row = token_base + i / 4;
        const uint32_t k = (base + p) * 128 + (i % 4) * 32;
        const bool valid = base + p < end && row < live && k < K;
        const uint64_t at = uint64_t(row) * RowStride + uint64_t(g) * GroupStride + k;
        shared.scales[p][i] = valid ? activation_scales[at / 32] : 127;
      }
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    ptx::D4 raw[2]{};
#pragma unroll
    for (int step = 0; step < 4; ++step) {
#pragma unroll
      for (int p = 0; p < 2; ++p) {
        ptx::A4 af{};
        ptx::B2 bf{};
#pragma unroll
        for (int i = 0; i < 4; ++i)
          af.v[i] = *reinterpret_cast<const uint32_t*>(shared.codes[p] +
              (quad + (i % 2) * 8) * 128 + step * 32 + (i / 2) * 16 + lane4 * 4);
        const uint4 w = step < 2 ? current.lo[p] : current.hi[p];
        bf.v[0] = (step & 1) ? w.z : w.x;
        bf.v[1] = (step & 1) ? w.w : w.y;
        const uint32_t sa = shared.scales[p][(quad + (lane4 & 1) * 8) * 4 + step];
        const uint32_t sb = (current.scales[p] >> (8 * step)) & 255;
        ptx::mma_mxf8f6f4<ptx::Element::e4m3, ptx::Element::e4m3>(raw[p], af, bf, sa, sb);
      }
    }
#pragma unroll
    for (int p = 0; p < 2; ++p) if (base + p < end) {
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        const uint32_t row = token_base + quad + (v / 2) * 8;
        const uint32_t col = task.x * TileN + warp * 8 + lane4 * 2 + v % 2;
        if (row < live && col < N)
          partials[((uint64_t(base + p) * rows + row) * Groups + g) * N + col] = raw[p].v[v];
      }
    }
    current = next;
    __syncthreads();
  }
  __threadfence();
  __syncthreads();
  const uint32_t tile = ((bank * Groups + g) * task.ny + task.y) * task.nx + task.x;
  if (threadIdx.x == 0) shared.last_tile = arrive(tickets + tile) == splits - 1;
  __syncthreads();
  if (!shared.last_tile) return;
  if (threadIdx.x == 0) shared.admitted = acquire(ready) != 0;
  __syncthreads();
  if (shared.admitted) {
    merge_task<N, K, Groups, TileN, F32Output, Paired>(task, bank, g, output, output2, workspace, live, rows);
    __syncthreads();
    if (threadIdx.x == 0) atomicExch(tickets + tile, 0);
  } else if (threadIdx.x == 0) {
    atomicOr(ready + 2, 1u);
  }
  // No CTA waits for an unscheduled admission task. A tile arriving early
  // leaves its completed ticket as a pending merge. The last completed
  // output tile acquires all producers and drains any pending merges.
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) shared.last_row_tile = arrive(done) == TilesPerRow - 1;
  __syncthreads();
  if (!shared.last_row_tile) return;
  if (threadIdx.x == 0) shared.admitted = acquire(ready + 2) != 0;
  __syncthreads();
  if (shared.admitted) for (uint32_t part = 0; part < TilesPerRow; ++part) {
    const uint32_t x = part % task.nx;
    const uint32_t bg = part / task.nx;
    const uint32_t index = (bg * task.ny + task.y) * task.nx + x;
    if (acquire(tickets + index)) {
      merge_task<N, K, Groups, TileN, F32Output, Paired>(Task{x, task.y, 0, task.nx, task.ny},
          bg / Groups, bg % Groups, output, output2, workspace, live, rows);
      __syncthreads();
      if (threadIdx.x == 0) atomicExch(tickets + index, 0);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) { *ready = 0; *done = 0; ready[2] = 0; }
}

// Read-only prefetch for a later task's immutable packed codes and scales.
// The schedule supplies the virtual CTA and keeps these ranges live through
// the later consumer. It must qualify DRAM traffic as well as elapsed time.
__device__ void prefetch_task(const PrefetchRanges& ranges, uint32_t cta, uint32_t ctas) {
  constexpr uint32_t Chunk = 16384;
  const uint64_t thread = uint64_t(cta) * blockDim.x + threadIdx.x;
  const uint64_t threads = uint64_t(ctas) * blockDim.x;
  uint64_t start = 0;
  for (int r = 0; r < 4; ++r) {
    const uint64_t bytes = ranges.bytes[r];
    const uint64_t pieces = (bytes + Chunk - 1) / Chunk;
    for (uint64_t p = (thread + threads - start % threads) % threads; p < pieces; p += threads) {
      const uint64_t offset = p * Chunk;
      const uint32_t size = uint32_t(min(uint64_t(Chunk), bytes - offset));
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
          :: "l"(ranges.base[r] + offset), "r"(size) : "memory");
    }
    start += pieces;
  }
}

// The entry wrapper alone supplies physical coordinates and PDL ordering.
template<int N, int K, int Groups, int RowStride, int GroupStride,
         int TileN, bool F32Output, bool Paired, bool DeferAdmission = false>
__device__ void entry(const uint8_t* activations, const uint8_t* activation_scales,
    const uint8_t* packed, const uint8_t* weight_scales,
    const uint8_t* packed2, const uint8_t* weight_scales2,
    void* output, void* output2, uint8_t* workspace, uint32_t* state,
    const uint32_t* live_rows, uint32_t* status, uint64_t status_row_offset,
    uint32_t rows, uint32_t folds_per_cta) {
  constexpr int Folds = (K + 127) / 128;
  const uint32_t splits = (Folds + folds_per_cta - 1) / folds_per_cta;
  const uint32_t first = (blockIdx.z % splits) * folds_per_cta;
  const uint32_t end = min(first + folds_per_cta, uint32_t(Folds));
  const Task task{blockIdx.x, blockIdx.y, blockIdx.z, gridDim.x, gridDim.y};
  __shared__ Shared arena;
  ptx::griddep_launch_dependents();
  Fragments first_weights = weight_fragments<N, K, Groups, TileN, Paired>(task, first, end, splits,
      packed, weight_scales, packed2, weight_scales2);
  ptx::griddep_wait();
  run_task<N, K, Groups, RowStride, GroupStride, TileN, F32Output, Paired, DeferAdmission>(task, arena, first_weights,
      activations, activation_scales, packed, weight_scales, packed2, weight_scales2,
      output, output2, workspace, state, live_rows, status, status_row_offset, rows, folds_per_cta);
}

} // namespace chop::gemm::mxfp8_fold_stream
