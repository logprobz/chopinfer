#pragma once

// Expert-grouped BF16 MMA over exact decoded MX operands. Queue tickets
// choose independent output tiles; they never choose an arithmetic order.
#include "chained_streamed.cuh"
#include "../common/ptx_matrix.cuh"

namespace chop::moe::grouped_bf16 {
__device__ __forceinline__ void copy16(void* dst, const void* src, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
    :: "r"(ptx::shared_address(dst)), "l"(src), "r"(valid ? 16 : 0) : "memory");
}
__device__ __forceinline__ void copy_scale(uint32_t* dst, const uint8_t* src,
                                         uint64_t at, uint64_t bytes, bool valid) {
  const uint64_t aligned = at & ~uint64_t(3);
  const int size = valid ? int(min(uint64_t(4),bytes-aligned)) : 0;
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;"
    :: "r"(ptx::shared_address(dst)), "l"(src+(valid ? aligned : 0)), "r"(size) : "memory");
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template<int Pending> __device__ __forceinline__ void wait() {
  asm volatile("cp.async.wait_group %0;" :: "n"(Pending) : "memory");
}


template<class C, int Rows> struct Queue {
  static constexpr int slots = (C::tokens_capacity * C::top_k + Rows - 1) / Rows + C::experts;
};

template<class C, int Rows>
__device__ __forceinline__ void pack(const int* counts, const int* offsets, const int* active,
                    const int* padded, const int* live_ptr, int* table) {
  const int lane = threadIdx.x, live = *live_ptr, total = active[0];
  if (blockDim.x != 32 || gridDim.x != 1 || live < 0 || live > C::tokens_capacity ||
      total < 0 || total > C::experts || total > live * C::top_k || *padded < 0 ||
      *padded % 64 || *padded > C::tokens_capacity * C::top_k + C::experts * 64) __trap();
  int carry = 0;
  for (int base = 0; base < total; base += 32) {
    const bool valid = base + lane < total;
    const int expert = valid ? active[base + lane + 1] : 0;
    if (expert < 0 || expert >= C::experts) __trap();
    const int count = valid ? counts[expert] : 0, first = valid ? offsets[expert] : 0;
    if (valid && (count <= 0 || count > live || first < 0 || first % 64 ||
                  first + count > *padded || (base + lane > 0 && expert <= active[base + lane]))) __trap();
    const int tiles = (count + Rows - 1) / Rows;
    int prefix = tiles;
#pragma unroll
    for (int delta = 1; delta < 32; delta *= 2) {
      const int previous = __shfl_up_sync(0xffffffffu, prefix, delta);
      if (lane >= delta) prefix += previous;
    }
    const int begin = carry + prefix - tiles;
    if (begin + tiles > Queue<C,Rows>::slots) __trap();
    for (int tile = 0; tile < tiles; ++tile) {
      const int at = 2 + 3 * (begin + tile);
      table[at] = expert;
      table[at + 1] = first + tile * Rows;
      table[at + 2] = min(Rows, count - tile * Rows);
    }
    carry += __shfl_sync(0xffffffffu, prefix, 31);
  }
  if (lane == 0) { table[0] = carry; table[1] = 0; }
}

template<class C, int Rows, int Cols, int Stages, bool ChainTile> struct Shared {
  static constexpr int decoded_k = ChainTile ? 128 : 64;
  static constexpr int stride = decoded_k + 8;
  struct alignas(16) Raw {
    uint8_t weights[C::halves * Cols][32];
    uint8_t input[Rows][64];
    uint32_t scales[C::halves * Cols + Rows];
  } raw[Stages];
  alignas(16) uint16_t decoded[C::halves * Cols + Rows][stride];
  int routes[Rows];
  int item;
};

template<class C, int Rows, int Cols, int Stages, bool ChainTile>
__device__ __forceinline__ void enqueue(Shared<C,Rows,Cols,Stages,ChainTile>& s,
    int slot, int step, int expert, int column, int live,
    const uint8_t* input, const uint8_t* weight,
    const uint8_t* weight_scales, const uint8_t* activation_scales) {
  auto& raw = s.raw[slot];
  const int k0 = step * 64;
  for (int v = threadIdx.x; v < C::halves * Cols * 2; v += blockDim.x) {
    const int r = v / 2, offset = (v % 2) * 16, col = column + r % Cols;
    const uint64_t row = uint64_t(expert) * C::weight_rows + (r / Cols) * C::n + col;
    const bool valid = col < C::n && k0 < C::k;
    copy16(raw.weights[r] + offset, weight + (valid ? row * (C::k / 2) + k0 / 2 + offset : 0), valid);
  }
  for (int v = threadIdx.x; v < Rows * 4; v += blockDim.x) {
    const int r = v / 4, offset = (v % 4) * 16, route = s.routes[r];
    const int source = C::gate_up ? route / C::top_k : route;
    const bool valid = route >= 0 && k0 < C::k;
    copy16(raw.input[r] + offset, input + (valid ? uint64_t(source) * C::k + k0 + offset : 0), valid);
  }
  for (int r = threadIdx.x; r < C::halves * Cols + Rows; r += blockDim.x) {
    const bool w = r < C::halves * Cols;
    const int col = column + r % Cols;
    const int route = w ? 0 : s.routes[r - C::halves * Cols];
    const int source = C::gate_up ? route / C::top_k : route;
    const bool valid = k0 < C::k && (w ? col < C::n : route >= 0);
    const uint64_t row = w ? uint64_t(expert) * C::weight_rows + (r / Cols) * C::n + col : source;
    const uint64_t at = row * (C::k / 32) + k0 / 32;
    const uint64_t bytes = uint64_t(w ? C::experts * C::weight_rows : live * (C::gate_up ? 1 : C::top_k)) * (C::k / 32);
    copy_scale(&raw.scales[r], w ? weight_scales : activation_scales, at, bytes, valid);
  }
  commit();
}

template<class C, int Rows, int Cols, int Stages, bool ChainTile>
__device__ __forceinline__ void decode(Shared<C,Rows,Cols,Stages,ChainTile>& s,
                                      int slot, int step, int expert, int column) {
  auto& raw = s.raw[slot];
  for (int v = threadIdx.x; v < (C::halves * Cols + Rows) * 8; v += blockDim.x) {
    const int r = v / 8, pair0 = (v % 8) * 4;
    const bool w = r < C::halves * Cols;
    const int route = w ? 0 : s.routes[r - C::halves * Cols];
    const int source = C::gate_up ? route / C::top_k : route;
    const int half = ChainTile ? (step & 1) * 64 : 0;
    if (step * 64 >= C::k || (w ? column + r % Cols >= C::n : route < 0)) {
      *reinterpret_cast<uint4*>(&s.decoded[r][half + pair0 * 2]) = make_uint4(0,0,0,0);
      continue;
    }
    const uint64_t row = w ? uint64_t(expert) * C::weight_rows + (r / Cols) * C::n + column + r % Cols : source;
    uint32_t scales = raw.scales[r] >> (8 * ((row * (C::k / 32) + step * 2) & 3));
    uint32_t words[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int pair = pair0 + i;
      const uint8_t scale = uint8_t(scales >> (pair / 16 * 8));
      if (w) {
        const uint8_t code = raw.weights[r][pair];
        convert::check_mxfp4(code, scale);
        words[i] = convert::scaled_mxfp4_bf16x2<false>(code, scale);
      } else {
        const uint16_t code = *reinterpret_cast<const uint16_t*>(raw.input[r - C::halves * Cols] + pair * 2);
        if ((code & 0x7f) == 0x7f || ((code >> 8) & 0x7f) == 0x7f) __trap();
        words[i] = chained::scaled_e4m3x2(code, scale);
      }
    }
    *reinterpret_cast<uint4*>(&s.decoded[r][half + pair0 * 2]) = make_uint4(words[0],words[1],words[2],words[3]);
  }
}

template<class C, int Rows, int Cols, int Stages, bool ChainTile, bool OmitFirst>
__device__ __forceinline__ void run(void* arena, const uint8_t* input, const uint8_t* weight,
    const uint8_t* weight_scales, const uint8_t* activation_scales,
    const int* sorted, const int* padded_ptr, const int* live_ptr,
    int* table, const float* route_weights, void* output) {
  constexpr int H = C::halves, F = 4, WarpRows = 32;
  constexpr int Threads = (Rows / WarpRows) * (Cols / 16) * 32;
  constexpr int Steps = C::loop_k / 64, Group = 2;
  using S = Shared<C,Rows,Cols,Stages,ChainTile>;
  static_assert(Rows >= 64 && Rows <= 256 && Rows % 64 == 0);
  static_assert(Cols >= 32 && Cols <= 128 && Cols % 32 == 0);
  static_assert(Stages >= 2 && Stages <= 3 && C::k % 64 == 0);
  static_assert(sizeof(S) <= 101376 && Threads <= 1024);
  auto& s = *static_cast<S*>(arena);
  const int live = *live_ptr, padded = *padded_ptr, tiles = table[0];
  if (blockDim.x != Threads || live < 0 || live > C::tokens_capacity ||
      tiles < 0 || tiles > Queue<C,Rows>::slots || padded < 0 || padded % 64 ||
      padded > C::tokens_capacity * C::top_k + C::experts * 64) __trap();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int row_base = (warp / (Cols / 16)) * WarpRows, column_warp = warp % (Cols / 16);
  const int columns = (C::n + Cols - 1) / Cols;
  for (;;) {
    if (threadIdx.x == 0) s.item = atomicAdd(table + 1, 1);
    __syncthreads();
    const int item = s.item;
    if (item >= tiles * columns) break;
    const int at = 2 + 3 * (item / columns), column = (item % columns) * Cols;
    const int expert = table[at], first = table[at + 1], count = table[at + 2];
    if (expert < 0 || expert >= C::experts || first < 0 || first % 64 ||
        count <= 0 || count > Rows || count > live || first + count > padded) __trap();
    for (int r = threadIdx.x; r < Rows; r += Threads) {
      const int route = r < count ? sorted[first + r] : -1;
      if (r < count && (route < 0 || route >= live * C::top_k)) __trap();
      s.routes[r] = route;
    }
    __syncthreads();
    ptx::D4 sum[H][F], partial[H][ChainTile ? Group : F];
#pragma unroll
    for (int h = 0; h < H; ++h)
#pragma unroll
      for (int f = 0; f < F; ++f)
#pragma unroll
        for (int v = 0; v < 4; ++v) {
          sum[h][f].v[v] = 0.f;
          if constexpr (!ChainTile) partial[h][f].v[v] = 0.f;
        }
#pragma unroll
    for (int step = 0; step < Stages; ++step)
      enqueue<C,Rows,Cols,Stages,ChainTile>(s,step,step,expert,column,live,input,weight,weight_scales,activation_scales);
#pragma unroll 1
    for (int step = 0; step < Steps; ++step) {
      wait<Stages - 1>();
      __syncthreads();
      decode<C,Rows,Cols,Stages,ChainTile>(s,step % Stages,step,expert,column);
      __syncthreads();
      // The raw slot is free after decode. Start its next load before MMA.
      if (step + Stages < Steps)
        enqueue<C,Rows,Cols,Stages,ChainTile>(s,step % Stages,step + Stages,expert,column,live,input,weight,weight_scales,activation_scales);
      else commit();
      if constexpr (ChainTile) {
        if (step % 2 == 0) continue;
#pragma unroll
        for (int first_f = 0; first_f < F; first_f += Group) {
#pragma unroll
          for (int h = 0; h < H; ++h)
#pragma unroll
            for (int f = 0; f < Group; ++f)
#pragma unroll
              for (int v = 0; v < 4; ++v) partial[h][f].v[v] = 0.f;
#pragma unroll
          for (int kk = 0; kk < 8; ++kk) {
            ptx::A4 a[H];
#pragma unroll
            for (int h = 0; h < H; ++h)
              ptx::ldmatrix_b16<4>(a[h].v, &s.decoded[h * Cols + column_warp * 16 + lane % 16][kk * 16 + (lane / 16) * 8]);
#pragma unroll
            for (int f = 0; f < Group; ++f) {
              if (row_base + (first_f + f) * 8 < count) {
                ptx::B2 b;
                ptx::ldmatrix_b16<2>(b.v, &s.decoded[H * Cols + row_base + (first_f + f) * 8 + lane % 8][kk * 16 + ((lane / 8) % 2) * 8]);
#pragma unroll
                for (int h = 0; h < H; ++h) ptx::mma_bf16(partial[h][f], a[h], b);
              }
            }
          }
#pragma unroll
          for (int h = 0; h < H; ++h)
#pragma unroll
            for (int f = 0; f < Group; ++f)
#pragma unroll
              for (int v = 0; v < 4; ++v)
                if (!OmitFirst || step > 1) sum[h][first_f + f].v[v] = __fadd_rn(sum[h][first_f + f].v[v], partial[h][f].v[v]);
        }
      } else {
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
          ptx::A4 a[H];
#pragma unroll
          for (int h = 0; h < H; ++h)
            ptx::ldmatrix_b16<4>(a[h].v, &s.decoded[h * Cols + column_warp * 16 + lane % 16][kk * 16 + (lane / 16) * 8]);
#pragma unroll
          for (int f = 0; f < F; ++f) {
            if (row_base + f * 8 < count) {
              ptx::B2 b;
              ptx::ldmatrix_b16<2>(b.v, &s.decoded[H * Cols + row_base + f * 8 + lane % 8][kk * 16 + ((lane / 8) % 2) * 8]);
#pragma unroll
              for (int h = 0; h < H; ++h) ptx::mma_bf16(partial[h][f], a[h], b);
            }
          }
        }
        if (step % 2 == 1) {
#pragma unroll
          for (int h = 0; h < H; ++h)
#pragma unroll
            for (int f = 0; f < F; ++f)
#pragma unroll
              for (int v = 0; v < 4; ++v) {
                if (!OmitFirst || step > 1) sum[h][f].v[v] = __fadd_rn(sum[h][f].v[v],partial[h][f].v[v]);
                partial[h][f].v[v] = 0.f;
              }
        }
      }
      __syncthreads();
    }
    wait<0>();
#pragma unroll
    for (int f = 0; f < F; ++f)
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        const int route = s.routes[row_base + f * 8 + 2 * t + (v & 1)];
        const int col = column + column_warp * 16 + g + (v >= 2 ? 8 : 0);
        if (route < 0 || col >= C::n) continue;
        const uint64_t at = uint64_t(route) * C::n + col;
        if constexpr (C::gate_up)
          static_cast<uint16_t*>(output)[at] = convert::bf16_from_f32(chained::before_down<typename C::clamp>(sum[0][f].v[v],sum[1][f].v[v],route_weights[route]));
        else static_cast<float*>(output)[at] = sum[0][f].v[v];
      }
    __syncthreads();
  }
}
} // namespace chop::moe::grouped_bf16
