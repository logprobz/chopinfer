#pragma once
#include "chained_policies.cuh"

// Weight staging changes neither the independent 128-K chains nor their stores.
namespace chop::moe::pipeline {
using namespace streamed;
using namespace chained_policy;
template<class C, int Rows, int Stages, int Ahead,
    class Activation = ExactE4M3Activation, class Scale = ExactMxFp4Scale,
    class Chain = Bf16Chain128, class Reduction = OrderedF32>
__device__ void compute(const uint8_t* input, const uint8_t* weight,
    const uint8_t* weight_scales, const uint8_t* activation_scales,
    const int32_t* offsets, const int32_t* sorted_rows, const int32_t* padded_rows,
    const int32_t* live_tokens, void* scratch) {
  using L = Layout<C>;
  static_assert(Stages >= 0 && Stages <= 8 && Ahead >= 0 && (Stages == 0 || Ahead == 0));
  constexpr int H = C::halves, Fragments = Rows / 8;
  static_assert(Rows == 8 || Rows == 64);
  __shared__ __align__(16) uint16_t activations[Rows][Stride];
  __shared__ int32_t rows[Rows];
  __shared__ __align__(16) uint8_t raw_weight[Stages > 0 ? Stages : 1][H * Columns][64];
  __shared__ uint8_t raw_scale[Stages > 0 ? Stages : 1][H * Columns][4];
  if (blockDim.x != Threads) __trap();
  ptx::griddep_launch_dependents();
  // The metadata producer has no early launch signal. Both queues are
  // complete here. Down reuses that immutable header across gate's merge.
  const int32_t* table = static_cast<const int32_t*>(scratch);
  const int experts = table[Rows == 8 ? 0 : 1];
  const int items = experts * (C::n / Columns) * L::blocks;
  auto fetch = [&](int item, int stage) {
    if (item < items) {
      const int block = item % L::blocks;
      const int column = (item / L::blocks) % (C::n / Columns) * Columns;
      const int slot = item / (L::blocks * (C::n / Columns));
      const int expert = table[2 + (Rows == 8 ? 0 : 2 * C::experts) + 2 * slot];
      if (expert < 0 || expert >= C::experts) __trap();
      for (int at = threadIdx.x; at < H * Columns * 4; at += Threads) {
        const int row = at / 4, group = at % 4, k = block * 128 + group * 32;
        const uint64_t wr = uint64_t(expert) * C::weight_rows +
            (row / Columns) * C::n + column + row % Columns;
        const bool valid = k < C::k;
        const uint8_t* src = valid ? weight + wr * (C::k / 2) + k / 2 : weight;
        // Device allocations and packed row strides are 16-byte aligned.
        // The planner checks that contract before dispatch.
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
            :: "r"(ptx::shared_address(&raw_weight[stage][row][group * 16])),
               "l"(src), "r"(valid ? 16 : 0) : "memory");
        raw_scale[stage][row][group] = valid ? weight_scales[wr * (C::k / 32) + k / 32] : 127;
      }
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  };
  auto prefetch = [&](int item) {
    if (item >= items || threadIdx.x >= H * Columns) return;
    const int block = item % L::blocks;
    const int column = (item / L::blocks) % (C::n / Columns) * Columns;
    const int slot = item / (L::blocks * (C::n / Columns));
    const int expert = table[2 + (Rows == 8 ? 0 : 2 * C::experts) + 2 * slot];
    if (expert < 0 || expert >= C::experts) __trap();
    const uint64_t wr = uint64_t(expert) * C::weight_rows +
        (threadIdx.x / Columns) * C::n + column + threadIdx.x % Columns;
    const uint8_t* pointer = weight + wr * (C::k / 2) + block * 64;
    asm volatile("prefetch.global.L2 [%0];" :: "l"(pointer) : "memory");
  };
  if constexpr (Stages > 0) {
#pragma unroll
    for (int stage = 0; stage < Stages; ++stage)
      fetch(blockIdx.x + stage * gridDim.x, stage);
  } else {
#pragma unroll
    for (int i = 0; i < Ahead; ++i) prefetch(blockIdx.x + i * gridDim.x);
  }
  // Only immutable weights and completed route metadata precede the wait.
  // Activations and chain scratch remain behind it, including down's alias.
  ptx::griddep_wait();
  const int live = *live_tokens, padded = *padded_rows;
  if (live < 0 || live > C::tokens_capacity || experts < 0 || experts > C::experts ||
      padded < 0 || padded % 64 || padded > C::tokens_capacity * C::top_k + C::experts * 64) __trap();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  float* out = partials<C>(scratch);
  int iteration = 0;
  for (int item = blockIdx.x; item < items; item += gridDim.x, ++iteration) {
    const int stage = Stages > 0 ? iteration % Stages : 0;
    if constexpr (Stages > 0) {
      asm volatile("cp.async.wait_group %0;" :: "n"(Stages - 1) : "memory");
      __syncthreads();
    } else {
      prefetch(item + Ahead * gridDim.x);
    }
    const int block = item % L::blocks;
    const int column = (item / L::blocks) % (C::n / Columns) * Columns;
    const int slot = item / (L::blocks * (C::n / Columns));
    const int at = 2 + (Rows == 8 ? 0 : 2 * C::experts) + 2 * slot;
    const int expert = table[at], count = table[at + 1];
    if (expert < 0 || expert >= C::experts || count <= 0 || count > live || count > Rows ||
        (Rows == 64 && count <= 8)) __trap();
    const int first = offsets[expert];
    if (first < 0 || first % 64 || first + count > padded) __trap();
    for (int r = threadIdx.x; r < Rows; r += Threads) {
      const int route = r < count ? sorted_rows[first + r] : -1;
      if (r < count && (route < 0 || route >= live * C::top_k)) __trap();
      rows[r] = route;
    }
    __syncthreads();

    // Four neighboring lanes own successive 32-K groups of each weight
    // row. Each issues a 16-byte load and one scale load, once per chain.
    uint4 owned[H][2];
    uint32_t scales[H][2];
    const int k = block * 128 + t * 32;
#pragma unroll
    for (int h = 0; h < H; ++h)
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        if constexpr (Stages > 0) {
          const int wr = h * Columns + warp * 16 + g + r * 8;
          owned[h][r] = *reinterpret_cast<const uint4*>(&raw_weight[stage][wr][t * 16]);
          scales[h][r] = raw_scale[stage][wr][t];
        } else {
          const uint64_t wr = uint64_t(expert) * C::weight_rows + h * C::n + column + warp * 16 + g + r * 8;
          const bool valid = k < C::k;
          owned[h][r] = valid ? load16(weight + wr * (C::k / 2) + k / 2) : make_uint4(0,0,0,0);
          scales[h][r] = valid ? weight_scales[wr * (C::k / 32) + k / 32] : 127;
        }
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
            *reinterpret_cast<uint32_t*>(&activations[r][group * 32 + v * 16 + w * 4 + pair * 2]) = Activation::decode(bits, scale);
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
            a[h].v[r] = Scale::decode(uint8_t(low >> (8 * t)), uint8_t(scale[h][r]));
            a[h].v[r + 2] = Scale::decode(uint8_t(high >> (8 * t)), uint8_t(scale[h][r]));
          }
        const int col = group * 32 + half * 16 + 2 * t;
#pragma unroll
        for (int f = 0; f < Fragments; ++f) {
          if (f * 8 < count) {
            ptx::B2 b;
            b.v[0] = chained::lds32(&activations[f * 8 + g][col]);
            b.v[1] = chained::lds32(&activations[f * 8 + g][col + 8]);
#pragma unroll
            for (int h = 0; h < H; ++h) Chain::mma(chain[h][f], a[h], b);
          }
        }
      }
    }
#pragma unroll
    for (int f = 0; f < Fragments; ++f)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int route = rows[f * 8 + 2 * t + (i & 1)];
        if (route < 0) continue;
        const int n = column + warp * 16 + g + (i >= 2 ? 8 : 0);
#pragma unroll
        for (int h = 0; h < H; ++h)
          out[((uint64_t(route) * H + h) * L::blocks + block) * C::n + n] = chain[h][f].v[i];
      }
    __syncthreads();
    if constexpr (Stages > 0) fetch(item + Stages * gridDim.x, stage);
  }
  if constexpr (Stages > 0) asm volatile("cp.async.wait_group 0;" ::: "memory");
}

} // namespace chop::moe::pipeline
