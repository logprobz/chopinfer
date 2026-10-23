#pragma once
#include "chained_absorb.cuh"
#include "chained_lanes_checks.h"

// The direct gate small and wide queues, the resident8 gate and the
// absorbing gate small queue reading their bank in quad-lane row tiles
// (`row-tiles-64x128-lanes`): the 64-row by 128-K tiles of
// `row-tiles-64x128-groups`, with each row's 64-byte K block reordered so
// lane t of a quad loads exactly the bytes it converts. Word g of the lane's
// 16-byte load holds bytes 16g + 4q + t (q = 0..3) of the block, and one
// 4-byte load holds the row's four group scales. The row-tiled queues give
// each lane one 32-K group, then shuffle every word to the quad and shift
// out the lane's byte; here each byte converts in place with the
// conversion's byte selector. Every A operand, MMA, chain, K order and
// partial address is the row-tiled queue's, so the partials are its bit
// for bit.
//
// The activation staging checks four E4M3 codes per instruction and the
// group's scale once (`chained_lanes_checks.h`). The same codes and scales
// trap, and the same conversion writes the same operand strip.
namespace chop::moe::lanes {
using namespace streamed;

// Byte K of `word` (two E2M1 codes) scaled by `pair`: the conversion
// convert::scaled_mxfp4_bf16x2 issues for that byte. K is a constant once
// the caller's loops unroll, so ptxas reads the byte in place.
__device__ __forceinline__ uint32_t scaled_byte(uint32_t word, int k, uint16_t pair) {
  uint32_t out;
  switch (k) {
  case 0:
    asm("{ .reg .b8 c0, c1, c2, c3; mov.b32 {c0, c1, c2, c3}, %1; "
        "cvt.rn.scaled::n2::ue8m0.bf16x2.e2m1x2 %0, c0, %2; }" : "=r"(out) : "r"(word), "h"(pair));
    break;
  case 1:
    asm("{ .reg .b8 c0, c1, c2, c3; mov.b32 {c0, c1, c2, c3}, %1; "
        "cvt.rn.scaled::n2::ue8m0.bf16x2.e2m1x2 %0, c1, %2; }" : "=r"(out) : "r"(word), "h"(pair));
    break;
  case 2:
    asm("{ .reg .b8 c0, c1, c2, c3; mov.b32 {c0, c1, c2, c3}, %1; "
        "cvt.rn.scaled::n2::ue8m0.bf16x2.e2m1x2 %0, c2, %2; }" : "=r"(out) : "r"(word), "h"(pair));
    break;
  default:
    asm("{ .reg .b8 c0, c1, c2, c3; mov.b32 {c0, c1, c2, c3}, %1; "
        "cvt.rn.scaled::n2::ue8m0.bf16x2.e2m1x2 %0, c3, %2; }" : "=r"(out) : "r"(word), "h"(pair));
  }
  return out;
}

// Group `group`'s scale byte of a row's scale word, doubled into the
// conversion's scale pair.
__device__ __forceinline__ uint16_t scale_pair(uint32_t scales, int group) {
  return uint16_t(__byte_perm(scales, 0, 0x4400 | (group * 0x11)));
}

// Two E4M3 codes scaled by a UE8M0 scale into BF16, for a scale at which
// every finite code is exact: chained::scaled_e4m3x2's conversion.
__device__ __forceinline__ uint32_t exact_e4m3x2(uint16_t codes, uint16_t pair) {
  uint32_t out;
  asm("cvt.rn.scaled::n2::ue8m0.bf16x2.e4m3x2 %0, %1, %2;" : "=r"(out) : "h"(codes), "h"(pair));
  return out;
}

// One item's weights: rows (h, r) of the lane, 16 code bytes and the four
// group scales each.
template<int H>
struct Weights {
  uint4 codes[H][2];
  uint32_t scales[H][2];
};

// Loads the lane's bytes of the item's tile: the row-tiled queues' tile,
// row and K block, at the lane's quad-lane offset.
template<class C, int H>
__device__ __forceinline__ Weights<H> load(const uint8_t* weight, const uint8_t* weight_scales,
    int expert, int column, int block, int warp, int g, int t) {
  using L = Layout<C>;
  Weights<H> w;
#pragma unroll
  for (int h = 0; h < H; ++h)
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const uint64_t tile = (uint64_t(expert) * C::weight_rows + h * C::n + column) / 64 * L::blocks + block;
      const int row = warp * 16 + g + r * 8;
      w.codes[h][r] = load16(weight + tile * 4096 + row * 64 + t * 16);
      w.scales[h][r] = *reinterpret_cast<const uint32_t*>(weight_scales + tile * 256 + row * 4);
      const uint32_t words[4] = {w.codes[h][r].x, w.codes[h][r].y, w.codes[h][r].z, w.codes[h][r].w};
      if (!checks::weights_valid(words, w.scales[h][r])) __trap();
    }
  return w;
}

// The A operands of `group`, `half`: the registers the row-tiled queue
// converts from the owner lane's words 2 * half and 2 * half + 1.
template<int H>
__device__ __forceinline__ void operands(const Weights<H>& w, int group, int half, ptx::A4 (&a)[H]) {
#pragma unroll
  for (int h = 0; h < H; ++h)
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const uint4 c = w.codes[h][r];
      const uint32_t word = group == 0 ? c.x : group == 1 ? c.y : group == 2 ? c.z : c.w;
      const uint16_t pair = scale_pair(w.scales[h][r], group);
      a[h].v[r] = scaled_byte(word, 2 * half, pair);
      a[h].v[r + 2] = scaled_byte(word, 2 * half + 1, pair);
    }
}

// The row-tiled queues' staging loop: thread c decodes route `rows[c / 4]`,
// group c % 4 of the item's K block into the shared operand strip.
template<class C, int Rows>
__device__ __forceinline__ void stage(const uint8_t* input, const uint8_t* activation_scales,
    const int32_t* rows, int block, uint16_t (*activations)[Stride]) {
  for (int c = threadIdx.x; c < Rows * 4; c += Threads) {
    const int r = c / 4, group = c % 4, k0 = block * 128 + group * 32;
    const int route = rows[r], source = C::gate_up ? route / C::top_k : route;
    const bool valid = route >= 0 && k0 < C::k;
    const uint8_t scale = valid ? activation_scales[uint64_t(source) * (C::k / 32) + k0 / 32] : 127;
    uint32_t* strip = reinterpret_cast<uint32_t*>(&activations[r][group * 32]);
    const bool extreme = checks::extreme_activation_scale(scale);
    const uint16_t pair = uint16_t(scale) * 0x0101;
#pragma unroll
    for (int v = 0; v < 2; ++v) {
      const uint4 raw = valid ? load16(input + uint64_t(source) * C::k + k0 + v * 16) : make_uint4(0,0,0,0);
      const uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
      if (checks::e4m3_nan(raw.x) | checks::e4m3_nan(raw.y) | checks::e4m3_nan(raw.z) |
          checks::e4m3_nan(raw.w)) __trap();
      if (extreme) {
#pragma unroll
        for (int i = 0; i < 8; ++i)
          strip[8 * v + i] = chained::scaled_e4m3x2(uint16_t(words[i / 2] >> (16 * (i % 2))), scale);
      } else {
#pragma unroll
        for (int i = 0; i < 8; ++i)
          strip[8 * v + i] = exact_e4m3x2(uint16_t(words[i / 2] >> (16 * (i % 2))), pair);
      }
    }
  }
}

// The item's MMAs over the staged strip, in the row-tiled queue's group,
// half and fragment order, into `chain`.
template<int H, int Fragments>
__device__ __forceinline__ void multiply(const Weights<H>& w, uint16_t (*activations)[Stride],
    int count, int g, int t, ptx::D4 (&chain)[H][Fragments]) {
#pragma unroll
  for (int group = 0; group < 4; ++group)
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      ptx::A4 a[H];
      operands<H>(w, group, half, a);
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

// streamed::compute<C, Rows, true> over the quad-lane bank.
template<class C, int Rows>
__device__ void compute(const uint8_t* input, const uint8_t* weight,
    const uint8_t* weight_scales, const uint8_t* activation_scales,
    const int32_t* offsets, const int32_t* sorted_rows, const int32_t* padded_rows,
    const int32_t* live_tokens, void* scratch) {
  using L = Layout<C>;
  constexpr int H = C::halves, Fragments = Rows / 8;
  static_assert(Rows == 8 || Rows == 64);
  static_assert(C::k % 128 == 0 && C::loop_k == C::k && C::n % 64 == 0 &&
                C::weight_rows % 64 == 0 && Columns == 64 && Threads == 128,
                "quad-lane row tiles need whole 64x128 tiles, one per item");
  __shared__ __align__(16) uint16_t activations[Rows][Stride];
  __shared__ int32_t rows[Rows];
  if (blockDim.x != Threads) __trap();
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const int live = *live_tokens, padded = *padded_rows;
  const int32_t* table = static_cast<const int32_t*>(scratch);
  const int experts = table[Rows == 8 ? 0 : 1];
  if (live < 0 || live > C::tokens_capacity || experts < 0 || experts > C::experts ||
      padded < 0 || padded % 64 || padded > C::tokens_capacity * C::top_k + C::experts * 64) __trap();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int items = experts * (C::n / Columns) * L::blocks;
  float* out = partials<C>(scratch);
  for (int item = blockIdx.x; item < items; item += gridDim.x) {
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
    const Weights<H> w = load<C, H>(weight, weight_scales, expert, column, block, warp, g, t);
    stage<C, Rows>(input, activation_scales, rows, block, activations);
    __syncthreads();
    ptx::D4 chain[H][Fragments];
#pragma unroll
    for (int h = 0; h < H; ++h)
#pragma unroll
      for (int f = 0; f < Fragments; ++f)
#pragma unroll
        for (int i = 0; i < 4; ++i) chain[h][f].v[i] = 0.f;
    multiply<H, Fragments>(w, activations, count, g, t, chain);
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
  }
}

// absorb::compute<C, true> over the quad-lane bank.
template<class C>
__device__ void absorbing(const uint8_t* input, const uint8_t* weight,
    const uint8_t* weight_scales, const uint8_t* activation_scales,
    const int32_t* offsets, const int32_t* sorted_rows, const int32_t* padded_rows,
    const int32_t* live_tokens, void* scratch) {
  using L = Layout<C>;
  constexpr int H = C::halves, Rows = ::chop::moe::absorb::Rows;
  static_assert(C::k % 128 == 0 && C::loop_k == C::k && C::n % 64 == 0 &&
                C::weight_rows % 64 == 0 && Columns == 64 && Threads == 128,
                "quad-lane row tiles need whole 64x128 tiles, one per item");
  __shared__ __align__(16) uint16_t activations[Rows][Stride];
  __shared__ int32_t rows[Rows];
  if (blockDim.x != Threads) __trap();
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const int live = *live_tokens, padded = *padded_rows;
  const int32_t* table = static_cast<const int32_t*>(scratch);
  const int small = table[0], wide = table[1];
  if (live < 0 || live > C::tokens_capacity || small < 0 || wide < 0 || small + wide > C::experts ||
      padded < 0 || padded % 64 || padded > C::tokens_capacity * C::top_k + C::experts * 64) __trap();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int items = (small + wide) * (C::n / Columns) * L::blocks;
  float* out = partials<C>(scratch);
  for (int item = blockIdx.x; item < items; item += gridDim.x) {
    const int block = item % L::blocks;
    const int column = (item / L::blocks) % (C::n / Columns) * Columns;
    const int slot = item / (L::blocks * (C::n / Columns));
    const bool is_wide = slot < wide;
    const int at = is_wide ? 2 + 2 * C::experts + 2 * slot : 2 + 2 * (slot - wide);
    const int expert = table[at], count = table[at + 1];
    if (expert < 0 || expert >= C::experts || count <= 0 || count > live ||
        (is_wide ? count <= 8 || count > 64 : count > 8)) __trap();
    const int first = offsets[expert];
    if (first < 0 || first % 64 || first + count > padded) __trap();
    const Weights<H> w = load<C, H>(weight, weight_scales, expert, column, block, warp, g, t);
    for (int base = 0; base < count; base += Rows) {
      for (int r = threadIdx.x; r < Rows; r += Threads) {
        const int route = base + r < count ? sorted_rows[first + base + r] : -1;
        if (base + r < count && (route < 0 || route >= live * C::top_k)) __trap();
        rows[r] = route;
      }
      __syncthreads();
      stage<C, Rows>(input, activation_scales, rows, block, activations);
      __syncthreads();
      ptx::D4 chain[H][1];
#pragma unroll
      for (int h = 0; h < H; ++h)
#pragma unroll
        for (int i = 0; i < 4; ++i) chain[h][0].v[i] = 0.f;
      multiply<H, 1>(w, activations, Rows, g, t, chain);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int route = rows[2 * t + (i & 1)];
        if (route < 0) continue;
        const int n = column + warp * 16 + g + (i >= 2 ? 8 : 0);
#pragma unroll
        for (int h = 0; h < H; ++h)
          out[((uint64_t(route) * H + h) * L::blocks + block) * C::n + n] = chain[h][0].v[i];
      }
      __syncthreads();
    }
  }
}
} // namespace chop::moe::lanes
