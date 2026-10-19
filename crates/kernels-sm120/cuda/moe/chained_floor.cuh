#pragma once
#include "chained_streamed.cuh"
#include "../row/ceil_parallel.cuh"

// A separate entry preserves the locked streamed consumers. Every weight
// stage is one independent 128-K chain; no ring stage changes a fold.
namespace chop::moe::floor {
using namespace streamed;
template<class C, int Rows, int Stages = 3>
__device__ void compute(const uint8_t* input, const uint8_t* weight,
    const uint8_t* weight_scales, const uint8_t* activation_scales,
    const int32_t* offsets, const int32_t* sorted_rows, const int32_t* padded_rows,
    const int32_t* live_tokens, void* scratch) {
  using L = Layout<C>;
  constexpr int H = C::halves, Fragments = Rows / 8;
  static_assert(Rows == 8 || Rows == 64);
  __shared__ __align__(16) uint16_t activations[Rows][Stride];
  __shared__ int32_t rows[Rows];
  __shared__ __align__(16) uint8_t raw_weight[Stages][H * Columns][64];
  __shared__ uint8_t raw_scale[Stages][H * Columns][4];
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
#pragma unroll
  for (int stage = 0; stage < Stages; ++stage)
    fetch(blockIdx.x + stage * gridDim.x, stage);
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
    const int stage = iteration % Stages;
    asm volatile("cp.async.wait_group %0;" :: "n"(Stages - 1) : "memory");
    __syncthreads();
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
#pragma unroll
    for (int h = 0; h < H; ++h)
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const int wr = h * Columns + warp * 16 + g + r * 8;
        owned[h][r] = *reinterpret_cast<const uint4*>(&raw_weight[stage][wr][t * 16]);
        scales[h][r] = raw_scale[stage][wr][t];
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
          if (f * 8 < count) {
            ptx::B2 b;
            b.v[0] = chained::lds32(&activations[f * 8 + g][col]);
            b.v[1] = chained::lds32(&activations[f * 8 + g][col + 8]);
#pragma unroll
            for (int h = 0; h < H; ++h) ptx::mma_bf16(chain[h][f], a[h], b);
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
    fetch(item + Stages * gridDim.x, stage);
  }
  asm volatile("cp.async.wait_group 0;" ::: "memory");
}

// Each warp finishes one 32-value ceil group. The last integer ticket for
// a token clears its entire quantized row on any sticky error, just as the
// unfused row quantizer does. The ticket publishes bytes, never float sums.
template<class C, uint32_t FloorBits>
__device__ void gate_finish(const void* scratch, const int32_t* live_tokens,
    const float* route_weights, uint16_t* intermediate, uint8_t* codes,
    uint8_t* scales, uint32_t* status, uint32_t* tickets) {
  static_assert(C::gate_up && C::n % 32 == 0);
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const int live = *live_tokens, route = blockIdx.y;
  if (live < 0 || live > C::tokens_capacity || blockDim.x != 128) __trap();
  if (route >= live * C::top_k) return;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int column0 = blockIdx.x * 128, group = column0 / 32 + warp;
  const int token = route / C::top_k;
  __shared__ uint16_t values[128];
  __shared__ uint32_t last;
  const float* input = partials<C>(const_cast<void*>(scratch));
  if (lane < 16) {
#pragma unroll
    for (int v = 0; v < 2; ++v) {
      const int column = group * 32 + lane * 2 + v;
      float sum[2] = {};
      if (column < C::n) {
#pragma unroll 1
        for (int block = 0; block < Layout<C>::blocks; ++block)
#pragma unroll
          for (int h = 0; h < 2; ++h)
            sum[h] = __fadd_rn(sum[h], __ldcg(input +
                ((uint64_t(route) * 2 + h) * Layout<C>::blocks + block) * C::n + column));
        const uint16_t value = convert::bf16_from_f32(
            chained::before_down<typename C::clamp>(sum[0], sum[1], route_weights[route]));
        values[warp * 32 + lane * 2 + v] = value;
        intermediate[uint64_t(route) * C::n + column] = value;
      } else values[warp * 32 + lane * 2 + v] = 0;
    }
  }
  __syncthreads();
  if (group < C::n / 32) {
    auto value = row::block_quantize_parallel::quantize_group<0,false,false,FloorBits>(
        values, warp * 32 + lane * 2);
    uint32_t flags = value.flags;
#pragma unroll
    for (int step = 16; step; step >>= 1)
      flags |= __shfl_xor_sync(0xffffffffu, flags, step);
    if (lane == 0) {
      if (flags) atomicOr(status + token, flags);
      scales[uint64_t(route) * (C::n / 32) + group] = value.scale_code;
    }
    if (lane < 16) {
      const uint64_t at = uint64_t(route) * C::n + group * 32 + lane * 2;
      codes[at] = uint8_t(value.codes);
      codes[at + 1] = uint8_t(value.codes >> 8);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    uint32_t arrived;
    asm volatile("atom.add.acq_rel.gpu.u32 %0, [%1], 1;"
        : "=r"(arrived) : "l"(tickets + token) : "memory");
    last = arrived == ((C::n + 127) / 128) * C::top_k - 1;
  }
  __syncthreads();
  if (last && __ldcg(status + token)) {
    for (int at = threadIdx.x; at < C::top_k * C::n; at += 128)
      codes[uint64_t(token) * C::top_k * C::n + at] = 0;
    for (int at = threadIdx.x; at < C::top_k * C::n / 32; at += 128)
      scales[uint64_t(token) * (C::top_k * C::n / 32) + at] = 127;
  }
}

// Route weights were applied before intermediate quantization. Fold each
// down output in K order, add routes in selection order, then add shared.
// The caller places this launch after the shared-expert join.
template<class C, bool OmitFirst = false>
__device__ void down_finish(const void* scratch, const int32_t* live_tokens,
    const int32_t* ids, const float* shared, float* routed, float* output) {
  static_assert(!C::gate_up);
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const int live = *live_tokens;
  if (live < 0 || live > C::tokens_capacity) __trap();
  const int token = blockIdx.y, column = blockIdx.x * 128 + threadIdx.x;
  if (token >= live || column >= C::n) return;
  const float* input = partials<C>(const_cast<void*>(scratch));
  float total = 0.f;
#pragma unroll
  for (int k = 0; k < C::top_k; ++k) {
    const int route = token * C::top_k + k;
    const int expert = ids[route];
    if (expert < 0 || expert >= C::experts) continue;
    float value = 0.f;
#pragma unroll
    for (int block = OmitFirst ? 1 : 0; block < Layout<C>::blocks; ++block)
      value = __fadd_rn(value, __ldcg(input +
          (uint64_t(route) * Layout<C>::blocks + block) * C::n + column));
    routed[uint64_t(route) * C::n + column] = value;
    total = __fadd_rn(total, value);
  }
  output[uint64_t(token) * C::n + column] =
      __fadd_rn(total, shared[uint64_t(token) * C::n + column]);
}
} // namespace chop::moe::floor
