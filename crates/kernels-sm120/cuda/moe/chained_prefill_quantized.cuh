#pragma once

#include "chained_streamed.cuh"

// Group expert rows into small, medium and full tiles. Every output keeps
// the old eight-MMA 128-K chain and ascending-K fold, entirely in registers.
namespace chop::moe::prefill_quantized {
using streamed::load16;
using streamed::check;
using streamed::shuffle;
constexpr int Threads = 128, Columns = 64, Stride = 136;

template<class C> struct Layout {
  static constexpr int blocks = C::loop_k / 128;
  static constexpr int slots = (C::tokens_capacity * C::top_k + 63) / 64 + C::experts;
  static_assert(C::row_tile == 64 && C::tokens_capacity <= 8192);
};

// Each queue contains (expert, first padded row, live rows) triples. A
// warp prefix sum preserves expert order without atomic arrival order.
template<class C>
__device__ void pack(const int32_t* counts, const int32_t* offsets,
                    const int32_t* active, const int32_t* live_tokens, void* scratch) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const int lane = threadIdx.x, live = *live_tokens, total = active[0];
  if (blockDim.x != 32 || gridDim.x != 1 || live < 0 || live > C::tokens_capacity ||
      total < 0 || total > C::experts || total > live * C::top_k) __trap();
  int* table = static_cast<int*>(scratch);
  int carry[3] = {};
  for (int base = 0; base < total; base += 32) {
    const bool valid = base + lane < total;
    const int expert = valid ? active[base + lane + 1] : 0;
    if (expert < 0 || expert >= C::experts) __trap();
    const int count = valid ? counts[expert] : 0, first = valid ? offsets[expert] : 0;
    if (valid && (count <= 0 || count > live || first < 0 || first % 64)) __trap();
    const int full = count / 64, tail = count % 64;
    const int sizes[3] = {int(tail > 0 && tail <= 8), int(tail > 8 && tail <= 32), full + int(tail > 32)};
#pragma unroll
    for (int q = 0; q < 3; ++q) {
      int prefix = sizes[q];
#pragma unroll
      for (int delta = 1; delta < 32; delta *= 2) {
        const int previous = __shfl_up_sync(0xffffffffu, prefix, delta);
        if (lane >= delta) prefix += previous;
      }
      const int begin = carry[q] + prefix - sizes[q];
      if (begin + sizes[q] > Layout<C>::slots) __trap();
      for (int tile = 0; tile < sizes[q]; ++tile) {
        const int row = q == 2 ? tile * 64 : full * 64;
        const int at = 3 + 3 * (q * Layout<C>::slots + begin + tile);
        table[at] = expert;
        table[at + 1] = first + row;
        table[at + 2] = min(64,count-row);
      }
      carry[q] += __shfl_sync(0xffffffffu, prefix, 31);
    }
  }
  if (lane == 0) for (int q = 0; q < 3; ++q) table[q] = carry[q];
}

template<class C, int Rows, bool OmitFirstChain = false>
__device__ void compute(const uint8_t* input, const uint8_t* weight,
    const uint8_t* weight_scales, const uint8_t* activation_scales,
    const int32_t* sorted_rows, const int32_t* padded_rows, const int32_t* live_tokens,
    const void* scratch, const float* route_weights, void* output) {
  using L = Layout<C>;
  constexpr int H = C::halves, Fragments = (Rows == 64 ? 32 : Rows) / 8;
  constexpr int Threads = Rows == 64 ? 256 : 128;
  constexpr int Queue = Rows == 8 ? 0 : (Rows == 32 ? 1 : 2);
  static_assert(Rows == 8 || Rows == 32 || Rows == 64);
  __shared__ __align__(16) uint16_t activations[Rows][Stride];
  __shared__ int32_t rows[Rows];
  if (blockDim.x != Threads) __trap();
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const int live = *live_tokens, padded = *padded_rows;
  const int* table = static_cast<const int*>(scratch);
  const int tiles = table[Queue];
  if (live < 0 || live > C::tokens_capacity || tiles < 0 || tiles > L::slots ||
      padded < 0 || padded % 64 || padded > C::tokens_capacity*C::top_k+C::experts*64) __trap();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int row_base = Rows == 64 ? (warp / 4) * 32 : 0, column_warp = warp % 4;
  const int items = tiles * (C::n / Columns);
  for (int item = blockIdx.x; item < items; item += gridDim.x) {
    const int column = item % (C::n / Columns) * Columns;
    const int at = 3 + 3 * (Queue * L::slots + item / (C::n / Columns));
    const int expert = table[at], first = table[at+1], count = table[at+2];
    if (expert < 0 || expert >= C::experts || first < 0 || first % 64 ||
        count <= 0 || count > Rows || count > live || first + count > padded) __trap();
    for (int r = threadIdx.x; r < Rows; r += Threads) {
      const int route = r < count ? sorted_rows[first+r] : -1;
      if (r < count && (route < 0 || route >= live*C::top_k)) __trap();
      rows[r] = route;
    }
    __syncthreads();
    ptx::D4 sum[H][Fragments];
#pragma unroll
    for (int h = 0; h < H; ++h)
#pragma unroll
      for (int f = 0; f < Fragments; ++f)
#pragma unroll
        for (int i = 0; i < 4; ++i) sum[h][f].v[i] = 0.f;
#pragma unroll 1
    for (int block = 0; block < L::blocks; ++block) {
    // Four neighboring lanes own successive 32-K groups of each weight
    // row. Each issues a 16-byte load and one scale load, once per chain.
    uint4 owned[H][2];
    uint32_t scales[H][2];
    const int k = block * 128 + t * 32;
#pragma unroll
    for (int h = 0; h < H; ++h)
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const uint64_t wr = uint64_t(expert) * C::weight_rows + h * C::n + column + column_warp * 16 + g + r * 8;
        const bool valid = k < C::k;
        owned[h][r] = valid ? load16(weight + wr * (C::k / 2) + k / 2) : make_uint4(0,0,0,0);
        scales[h][r] = valid ? weight_scales[wr * (C::k / 32) + k / 32] : 127;
        check(owned[h][r], uint8_t(scales[h][r]));
      }

    // Two wide loads per 32-code activation group, decoded once into the
    // shared operand strip and reused by all 64 output columns.
    for (int c = threadIdx.x; c < Rows * 4; c += Threads) {
      const int r = c / 4, group = c % 4, k0 = block * 128 + group * 32;
      const int route = rows[r], source = C::gate_up ? route / C::top_k : route;
      const bool valid = route >= 0 && k0 < C::k;
      const uint8_t scale = valid ? activation_scales[uint64_t(source) * (C::k / 32) + k0 / 32] : 127;
#pragma unroll
      for (int v = 0; v < 2; ++v) {
        const uint4 raw = valid ? load16(input + uint64_t(source) * C::k + k0 + v * 16) : make_uint4(0,0,0,0);
        const uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
        for (int w = 0; w < 4; ++w)
#pragma unroll
          for (int pair = 0; pair < 2; ++pair) {
            const uint16_t bits = uint16_t(words[w] >> (16 * pair));
            if ((bits & 0x7f) == 0x7f || ((bits >> 8) & 0x7f) == 0x7f) __trap();
            *reinterpret_cast<uint32_t*>(&activations[r][group * 32 + v * 16 + w * 4 + pair * 2]) = chained::scaled_e4m3x2(bits, scale);
          }
      }
    }
    __syncthreads();
    ptx::D4 chain[H][Fragments];
#pragma unroll
    for (int h = 0; h < H; ++h)
#pragma unroll
      for (int f = 0; f < Fragments; ++f)
#pragma unroll
        for (int i = 0; i < 4; ++i) chain[h][f].v[i] = 0.f;
#pragma unroll
    for (int group = 0; group < 4; ++group) {
      const int owner = (lane & ~3) + group;
      uint4 raw[H][2];
      uint32_t scale[H][2];
#pragma unroll
      for (int h = 0; h < H; ++h)
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          raw[h][r] = make_uint4(shuffle(owned[h][r].x, owner), shuffle(owned[h][r].y, owner),
                                shuffle(owned[h][r].z, owner), shuffle(owned[h][r].w, owner));
          scale[h][r] = shuffle(scales[h][r], owner);
        }
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        ptx::A4 a[H];
#pragma unroll
        for (int h = 0; h < H; ++h)
#pragma unroll
          for (int r = 0; r < 2; ++r) {
            const uint32_t low = half == 0 ? raw[h][r].x : raw[h][r].z;
            const uint32_t high = half == 0 ? raw[h][r].y : raw[h][r].w;
            a[h].v[r] = convert::scaled_mxfp4_bf16x2<false>(uint8_t(low >> (8 * t)), uint8_t(scale[h][r]));
            a[h].v[r + 2] = convert::scaled_mxfp4_bf16x2<false>(uint8_t(high >> (8 * t)), uint8_t(scale[h][r]));
          }
        const int col = group * 32 + half * 16 + 2 * t;
#pragma unroll
        for (int f = 0; f < Fragments; ++f) {
          if (row_base + f * 8 < count) {
            ptx::B2 b;
            b.v[0] = chained::lds32(&activations[row_base + f * 8 + g][col]);
            b.v[1] = chained::lds32(&activations[row_base + f * 8 + g][col + 8]);
#pragma unroll
            for (int h = 0; h < H; ++h) ptx::mma_bf16(chain[h][f], a[h], b);
          }
        }
      }
    }
#pragma unroll
      for (int h = 0; h < H; ++h)
#pragma unroll
        for (int f = 0; f < Fragments; ++f)
#pragma unroll
          for (int i = 0; i < 4; ++i)
            if (!OmitFirstChain || block != 0)
              sum[h][f].v[i] = __fadd_rn(sum[h][f].v[i],chain[h][f].v[i]);
      __syncthreads();
    }
#pragma unroll
    for (int f = 0; f < Fragments; ++f)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int route = rows[row_base+f*8+2*t+(i&1)];
        if (route < 0) continue;
        const uint64_t at = uint64_t(route)*C::n+column+column_warp*16+g+(i>=2 ? 8 : 0);
        if constexpr (C::gate_up)
          static_cast<uint16_t*>(output)[at] = convert::bf16_from_f32(
              chained::before_down<typename C::clamp>(sum[0][f].v[i],sum[1][f].v[i],route_weights[route]));
        else static_cast<float*>(output)[at] = sum[0][f].v[i];
      }
    __syncthreads();
  }
}
} // namespace chop::moe::prefill_quantized
