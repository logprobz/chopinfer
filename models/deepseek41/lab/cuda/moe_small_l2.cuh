#pragma once
#include "moe/chained_lanes_scale_tiles.cuh"
#include "moe/chained_down_scale_tiles.cuh"
#include "moe/chained_bandwidth.cuh"

// Lab candidate: the served direct MoE queues and finishers of the 9- to
// 64-row steps (absorbing gate queue, bandwidth gate finisher, two-strip
// down queue, floor down finisher) with L2 cache-policy knobs. Every load,
// conversion, MMA, add, store address and value is the served body's; only
// the cache policy of the weight loads and of the K-block partials changes.
//
// The queues write every K-block partial as F32 and the finishers read each
// one back once. At 32 to 64 rows that is a quarter to a half of the weight
// bytes again in each direction, and the 256 MiB of streamed weights evict
// the partials before their finisher runs.
// - Stream: weight code loads skip L1 and carry an L2 evict_first policy.
//   Each weight byte is read once per step, so streaming them leaves the
//   partials resident.
// - Keep: partial stores carry an L2 evict_last policy.
// - Retire: a finisher discards each 128-byte partial line once its warp
//   has read it (discard.global.L2), so a dead line is never written back.
//   Every partial a finisher reads was written by this step's queue, and
//   each line is read by exactly one warp.
// - Loads: the gate finisher issues every K-block partial load before its
//   first add, as the served gate_finish_loads.cuh does, instead of groups
//   of four; the adds and their order are unchanged.
// - Fold: one down item covers a strip pair's five K blocks in ascending
//   order and folds their chains with the finisher's adds (+0, then each
//   block's chain), so it stores one F32 per route and column instead of
//   five, and the down finisher reads that value where it read the five.
// - Early: each queue item issues its weight code loads before the
//   tile-scale decode and its barrier, rather than after, so the two
//   latencies overlap. The codes, scales and checks are the same.
namespace chop::moe::lab_l2 {
using namespace streamed;

template<bool Stream, bool Keep, bool Retire, bool Loads = false, bool Fold = false,
    bool Early = false>
struct Policy {
  static constexpr bool stream = Stream, keep = Keep, retire = Retire, loads = Loads, fold = Fold,
      early = Early;
};

// Each hinted access creates its policy in place, so no policy register
// stays live across the queues' loops (they run at 64 registers).

// streamed::load16 with the Stream policy on its vector branch.
template<class P>
__device__ __forceinline__ uint4 weight16(const uint8_t* p) {
  if constexpr (P::stream) {
    if ((reinterpret_cast<uintptr_t>(p) & 15u) == 0) {
      uint4 v;
      asm("{ .reg .b64 policy; createpolicy.fractional.L2::evict_first.b64 policy, 1.0; "
          "ld.global.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], policy; }"
          : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
      return v;
    }
  }
  return load16(p);
}

template<class P>
__device__ __forceinline__ void store_partial(float* p, float value) {
  if constexpr (P::keep)
    asm volatile("{ .reg .b64 policy; createpolicy.fractional.L2::evict_last.b64 policy, 1.0; "
                 "st.global.L2::cache_hint.f32 [%0], %1, policy; }"
                 :: "l"(p), "f"(value) : "memory");
  else
    *p = value;
}

__device__ __forceinline__ void discard_line(const float* line) {
  asm volatile("discard.global.L2 [%0], 128;" :: "l"(line) : "memory");
}

// lanes::load with weight16.
template<class C, int H, class Scales, class P>
__device__ __forceinline__ lanes::Weights<H> load(const uint8_t* weight, const uint8_t* weight_scales,
    int expert, int column, int block, int warp, int g, int t) {
  using L = Layout<C>;
  lanes::Weights<H> w;
#pragma unroll
  for (int h = 0; h < H; ++h)
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const uint64_t tile = (uint64_t(expert) * C::weight_rows + h * C::n + column) / 64 * L::blocks + block;
      const int row = warp * 16 + g + r * 8;
      w.codes[h][r] = weight16<P>(weight + tile * 4096 + row * 64 + t * 16);
      w.scales[h][r] = Scales::template load<C>(weight_scales, expert, h, column, block, row, tile);
      const uint32_t words[4] = {w.codes[h][r].x, w.codes[h][r].y, w.codes[h][r].z, w.codes[h][r].w};
      if (!lanes::checks::weights_valid(words, w.scales[h][r])) __trap();
    }
  return w;
}

// load's code loads alone, for Early.
template<class C, int H, class P>
__device__ __forceinline__ void load_codes(lanes::Weights<H>& w, const uint8_t* weight,
    int expert, int column, int block, int warp, int g, int t) {
  using L = Layout<C>;
#pragma unroll
  for (int h = 0; h < H; ++h)
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const uint64_t tile = (uint64_t(expert) * C::weight_rows + h * C::n + column) / 64 * L::blocks + block;
      w.codes[h][r] = weight16<P>(weight + tile * 4096 + (warp * 16 + g + r * 8) * 64 + t * 16);
    }
}

// load's scale words and checks after load_codes, for Early.
template<class C, int H, class Scales>
__device__ __forceinline__ void load_scales(lanes::Weights<H>& w, const uint8_t* weight_scales,
    int expert, int column, int block, int warp, int g) {
  using L = Layout<C>;
#pragma unroll
  for (int h = 0; h < H; ++h)
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const uint64_t tile = (uint64_t(expert) * C::weight_rows + h * C::n + column) / 64 * L::blocks + block;
      const int row = warp * 16 + g + r * 8;
      w.scales[h][r] = Scales::template load<C>(weight_scales, expert, h, column, block, row, tile);
      const uint32_t words[4] = {w.codes[h][r].x, w.codes[h][r].y, w.codes[h][r].z, w.codes[h][r].w};
      if (!lanes::checks::weights_valid(words, w.scales[h][r])) __trap();
    }
}

// lanes::absorbing with the policy.
template<class C, class Scales, class P>
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
        (is_wide ? count <= 8 || count > L::expert_rows : count > 8)) __trap();
    const int first = offsets[expert];
    if (first < 0 || first % 64 || first + count > padded) __trap();
    lanes::Weights<H> w;
    if constexpr (P::early) {
      load_codes<C, H, P>(w, weight, expert, column, block, warp, g, t);
      const uint8_t* item_scales = Scales::template prepare<C>(weight_scales, expert, column, block);
      if constexpr (Scales::staged) __syncthreads();
      load_scales<C, H, Scales>(w, item_scales, expert, column, block, warp, g);
    } else {
      const uint8_t* item_scales = Scales::template prepare<C>(weight_scales, expert, column, block);
      if constexpr (Scales::staged) __syncthreads();
      w = load<C, H, Scales, P>(weight, item_scales, expert, column, block, warp, g, t);
    }
    for (int base = 0; base < count; base += Rows) {
      for (int r = threadIdx.x; r < Rows; r += Threads) {
        const int route = base + r < count ? sorted_rows[first + base + r] : -1;
        if (base + r < count && (route < 0 || route >= live * C::top_k)) __trap();
        rows[r] = route;
      }
      __syncthreads();
      lanes::stage<C, Rows>(input, activation_scales, rows, block, activations);
      __syncthreads();
      ptx::D4 chain[H][1];
#pragma unroll
      for (int h = 0; h < H; ++h)
#pragma unroll
        for (int i = 0; i < 4; ++i) chain[h][0].v[i] = 0.f;
      lanes::multiply<H, 1>(w, activations, Rows, g, t, chain);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int route = rows[2 * t + (i & 1)];
        if (route < 0) continue;
        const int n = column + warp * 16 + g + (i >= 2 ? 8 : 0);
#pragma unroll
        for (int h = 0; h < H; ++h)
          store_partial<P>(&out[((uint64_t(route) * H + h) * L::blocks + block) * C::n + n], chain[h][0].v[i]);
      }
      __syncthreads();
    }
  }
}

// strip2 with Fold: item (slot, strip pair); each pass of eight rows walks
// the five K blocks in order with the served item's loads, staging and MMAs
// per block, and adds each block's chain to the pass's running sums. Each
// thread keeps its eight sums in its own shared-memory column, since eight
// more live registers spill at the queue's 64. A slot of more than eight
// rows reloads its weights per pass with the default policy, so the later
// passes can hit L2.
template<class C, class Scales, class P, class Activation, class Scale, class Chain,
    class Reduction = chained_policy::OrderedF32>
__device__ void strip2_fold(const uint8_t* input, const uint8_t* weight,
    const uint8_t* weight_scales, const uint8_t* activation_scales,
    const int32_t* offsets, const int32_t* sorted_rows, const int32_t* padded_rows,
    const int32_t* live_tokens, void* scratch) {
  using L = Layout<C>;
  constexpr int Strips = 2, Rows = ::chop::moe::absorb::Rows;
  constexpr int H = C::halves * Strips;
  __shared__ __align__(16) uint16_t activations[Rows][Stride];
  __shared__ int32_t rows[Rows];
  __shared__ float sums[H * 4][Threads];
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
  constexpr int Pairs = C::n / (Strips * Columns);
  const int items = (small + wide) * Pairs;
  float* out = partials<C>(scratch);
  for (int item = blockIdx.x; item < items; item += gridDim.x) {
    const int column = item % Pairs * (Strips * Columns);
    const int slot = item / Pairs;
    const bool is_wide = slot < wide;
    const int at = is_wide ? 2 + 2 * C::experts + 2 * slot : 2 + 2 * (slot - wide);
    const int expert = table[at], count = table[at + 1];
    if (expert < 0 || expert >= C::experts || count <= 0 || count > live ||
        (is_wide ? count <= 8 || count > L::expert_rows : count > 8)) __trap();
    const int first = offsets[expert];
    if (first < 0 || first % 64 || first + count > padded) __trap();
    for (int base = 0; base < count; base += Rows) {
      for (int r = threadIdx.x; r < Rows; r += Threads) {
        const int route = base + r < count ? sorted_rows[first + base + r] : -1;
        if (base + r < count && (route < 0 || route >= live * C::top_k)) __trap();
        rows[r] = route;
      }
#pragma unroll
      for (int j = 0; j < H * 4; ++j) sums[j][threadIdx.x] = 0.f;
#pragma unroll 1
      for (int block = 0; block < L::blocks; ++block) {
        uint4 owned[H][2];
        uint32_t scales[H][2];
        const int k = block * 128 + t * 32;
        const bool valid = k < C::k;
        const auto codes = [&] {
#pragma unroll
          for (int h = 0; h < H; ++h)
#pragma unroll
            for (int r = 0; r < 2; ++r) {
              const uint64_t wr = uint64_t(expert) * C::weight_rows + (h / Strips) * C::n + (h % Strips) * Columns + column + warp * 16 + g + r * 8;
              const uint8_t* at_weight = weight + wr * (C::k / 2) + k / 2;
              owned[h][r] = valid ? (is_wide ? load16(at_weight) : weight16<P>(at_weight)) : make_uint4(0,0,0,0);
            }
        };
        if constexpr (P::early) codes();
        const uint32_t* scale_words = Scales::template prepare<C, Strips>(weight_scales, expert, column, block);
        __syncthreads();
        if constexpr (!P::early) codes();
#pragma unroll
        for (int h = 0; h < H; ++h)
#pragma unroll
          for (int r = 0; r < 2; ++r) {
            scales[h][r] = valid ? ((scale_words[h * Columns + warp * 16 + g + r * 8] >> (8 * t)) & 255) : 127;
            check(owned[h][r], uint8_t(scales[h][r]));
          }
        for (int c = threadIdx.x; c < Rows * 4; c += Threads) {
          const int r = c / 4, group = c % 4, k0 = block * 128 + group * 32;
          const int route = rows[r], source = route;
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
        ptx::D4 chain[H];
#pragma unroll
        for (int h = 0; h < H; ++h)
#pragma unroll
          for (int i = 0; i < 4; ++i) chain[h].v[i] = 0.f;
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
            ptx::B2 b;
            b.v[0] = chained::lds32(&activations[g][col]);
            b.v[1] = chained::lds32(&activations[g][col + 8]);
#pragma unroll
            for (int h = 0; h < H; ++h) Chain::mma(chain[h], a[h], b);
          }
        }
        // The down finisher's fold of this block, in its order.
#pragma unroll
        for (int h = 0; h < H; ++h)
#pragma unroll
          for (int i = 0; i < 4; ++i)
            sums[h * 4 + i][threadIdx.x] = Reduction::add(sums[h * 4 + i][threadIdx.x], chain[h].v[i]);
        __syncthreads();
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int route = rows[2 * t + (i & 1)];
        if (route < 0) continue;
        const int n = column + warp * 16 + g + (i >= 2 ? 8 : 0);
#pragma unroll
        for (int h = 0; h < H; ++h)
          store_partial<P>(&out[uint64_t(route) * C::n + n + (h % Strips) * Columns], sums[h * 4 + i][threadIdx.x]);
      }
      __syncthreads();
    }
  }
}

// absorb::compute<C, false, Scales, 2> (strip2::compute) with the policy.
template<class C, class Scales, class P,
    class Activation = chained_policy::ExactE4M3Activation,
    class Scale = chained_policy::ExactMxFp4Scale,
    class Chain = chained_policy::Bf16Chain128>
__device__ void strip2(const uint8_t* input, const uint8_t* weight,
    const uint8_t* weight_scales, const uint8_t* activation_scales,
    const int32_t* offsets, const int32_t* sorted_rows, const int32_t* padded_rows,
    const int32_t* live_tokens, void* scratch) {
  if constexpr (P::fold) {
    strip2_fold<C, Scales, P, Activation, Scale, Chain>(input, weight, weight_scales,
        activation_scales, offsets, sorted_rows, padded_rows, live_tokens, scratch);
    return;
  }
  using L = Layout<C>;
  constexpr int Strips = 2, Rows = ::chop::moe::absorb::Rows;
  constexpr int H = C::halves * Strips;
  static_assert(!C::gate_up && Scales::staged);
  static_assert(C::n % (Strips * Columns) == 0, "whole strip items");
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
  const int items = (small + wide) * (C::n / (Strips * Columns)) * L::blocks;
  float* out = partials<C>(scratch);
  for (int item = blockIdx.x; item < items; item += gridDim.x) {
    const int block = item % L::blocks;
    const int column = (item / L::blocks) % (C::n / (Strips * Columns)) * (Strips * Columns);
    const int slot = item / (L::blocks * (C::n / (Strips * Columns)));
    const bool is_wide = slot < wide;
    const int at = is_wide ? 2 + 2 * C::experts + 2 * slot : 2 + 2 * (slot - wide);
    const int expert = table[at], count = table[at + 1];
    if (expert < 0 || expert >= C::experts || count <= 0 || count > live ||
        (is_wide ? count <= 8 || count > L::expert_rows : count > 8)) __trap();
    const int first = offsets[expert];
    if (first < 0 || first % 64 || first + count > padded) __trap();

    uint4 owned[H][2];
    uint32_t scales[H][2];
    const int k = block * 128 + t * 32;
    if constexpr (P::early) {
      const bool valid = k < C::k;
#pragma unroll
      for (int h = 0; h < H; ++h)
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const uint64_t wr = uint64_t(expert) * C::weight_rows + (h / Strips) * C::n + (h % Strips) * Columns + column + warp * 16 + g + r * 8;
          owned[h][r] = valid ? weight16<P>(weight + wr * (C::k / 2) + k / 2) : make_uint4(0,0,0,0);
        }
      const uint32_t* scale_words = Scales::template prepare<C, Strips>(weight_scales, expert, column, block);
      __syncthreads();
#pragma unroll
      for (int h = 0; h < H; ++h)
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          scales[h][r] = valid ? ((scale_words[h * Columns + warp * 16 + g + r * 8] >> (8 * t)) & 255) : 127;
          check(owned[h][r], uint8_t(scales[h][r]));
        }
    } else {
      const uint32_t* scale_words = Scales::template prepare<C, Strips>(weight_scales, expert, column, block);
      __syncthreads();
#pragma unroll
      for (int h = 0; h < H; ++h)
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const uint64_t wr = uint64_t(expert) * C::weight_rows + (h / Strips) * C::n + (h % Strips) * Columns + column + warp * 16 + g + r * 8;
          const bool valid = k < C::k;
          owned[h][r] = valid ? weight16<P>(weight + wr * (C::k / 2) + k / 2) : make_uint4(0,0,0,0);
          scales[h][r] = valid ? ((scale_words[h * Columns + warp * 16 + g + r * 8] >> (8 * t)) & 255) : 127;
          check(owned[h][r], uint8_t(scales[h][r]));
        }
    }

    for (int base = 0; base < count; base += Rows) {
      for (int r = threadIdx.x; r < Rows; r += Threads) {
        const int route = base + r < count ? sorted_rows[first + base + r] : -1;
        if (base + r < count && (route < 0 || route >= live * C::top_k)) __trap();
        rows[r] = route;
      }
      __syncthreads();
      for (int c = threadIdx.x; c < Rows * 4; c += Threads) {
        const int r = c / 4, group = c % 4, k0 = block * 128 + group * 32;
        const int route = rows[r], source = route;
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
      ptx::D4 chain[H];
#pragma unroll
      for (int h = 0; h < H; ++h)
#pragma unroll
        for (int i = 0; i < 4; ++i) chain[h].v[i] = 0.f;
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
          ptx::B2 b;
          b.v[0] = chained::lds32(&activations[g][col]);
          b.v[1] = chained::lds32(&activations[g][col + 8]);
#pragma unroll
          for (int h = 0; h < H; ++h) Chain::mma(chain[h], a[h], b);
        }
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int route = rows[2 * t + (i & 1)];
        if (route < 0) continue;
        const int n = column + warp * 16 + g + (i >= 2 ? 8 : 0);
#pragma unroll
        for (int h = 0; h < H; ++h)
          store_partial<P>(&out[((uint64_t(route) * C::halves + h / Strips) * L::blocks + block) * C::n + n + (h % Strips) * Columns], chain[h].v[i]);
      }
      __syncthreads();
    }
  }
}

// bandwidth::gate_finish with the policy.
template<class C, uint32_t FloorBits, class P, class Clamp = typename C::clamp,
    class Quantization = chained_policy::CeilE4M3Group32<FloorBits>,
    class RouteWeight = chained_policy::BeforeQuantizeRouteWeight,
    class Output = chained_policy::Bf16IntermediateF32Output,
    class Reduction = chained_policy::OrderedF32>
__device__ void gate_finish(const void* scratch, const int32_t* live_tokens,
    const float* route_weights, uint16_t* intermediate, uint8_t* codes,
    uint8_t* scales, uint32_t* status, uint32_t* tickets) {
  static_assert(C::gate_up && C::n % 32 == 0 && C::n * 4 % 128 == 0);
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
  if constexpr (P::retire)
    if (reinterpret_cast<uintptr_t>(input) & 127u) __trap();
  const int column = column0 + threadIdx.x;
  if (column < C::n && P::loads) {
    constexpr int Blocks = Layout<C>::blocks;
    const float weight = route_weights[route];
    float value[Blocks][2];
#pragma unroll
    for (int b = 0; b < Blocks; ++b)
#pragma unroll
      for (int h = 0; h < 2; ++h)
        value[b][h] = __ldcg(input + ((uint64_t(route) * 2 + h) * Blocks + b) * C::n + column);
    float sum[2] = {};
#pragma unroll
    for (int b = 0; b < Blocks; ++b)
#pragma unroll
      for (int h = 0; h < 2; ++h) sum[h] = Reduction::add(sum[h], value[b][h]);
    const uint16_t out = Output::intermediate(
        RouteWeight::template apply<Clamp>(sum[0], sum[1], weight));
    values[threadIdx.x] = out;
    intermediate[uint64_t(route) * C::n + column] = out;
  } else if (column < C::n) {
    float sum[2] = {};
#pragma unroll 1
    for (int base = 0; base < Layout<C>::blocks; base += 4) {
      float value[4][2];
#pragma unroll
      for (int b = 0; b < 4; ++b)
#pragma unroll
        for (int h = 0; h < 2; ++h)
          value[b][h] = base + b < Layout<C>::blocks ? __ldcg(input +
              ((uint64_t(route) * 2 + h) * Layout<C>::blocks + base + b) * C::n + column) : 0.f;
#pragma unroll
      for (int b = 0; b < 4; ++b)
        if (base + b < Layout<C>::blocks)
#pragma unroll
          for (int h = 0; h < 2; ++h) sum[h] = Reduction::add(sum[h], value[b][h]);
    }
    const uint16_t value = Output::intermediate(
        RouteWeight::template apply<Clamp>(sum[0], sum[1], route_weights[route]));
    values[threadIdx.x] = value;
    intermediate[uint64_t(route) * C::n + column] = value;
  } else values[threadIdx.x] = 0;
  if constexpr (P::retire) {
    // The warp's 32 columns of each (half, K block) row are one line, all
    // valid or all past C::n.
    __syncwarp();
    if (column0 + 32 * warp < C::n)
      for (int line = lane; line < 2 * Layout<C>::blocks; line += 32)
        discard_line(input + ((uint64_t(route) * 2 + line % 2) * Layout<C>::blocks + line / 2) * C::n +
                     column0 + 32 * warp);
  }
  __syncthreads();
  if (group < C::n / 32) {
    auto value = Quantization::quantize(
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

// floor::down_finish<C, false> with the policy.
template<class C, class P,
    class Reduction = chained_policy::OrderedF32,
    class Output = chained_policy::Bf16IntermediateF32Output>
__device__ void down_finish(const void* scratch, const int32_t* live_tokens,
    const int32_t* ids, const float* shared, float* routed, float* output) {
  static_assert(!C::gate_up && C::n % 128 == 0);
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const int live = *live_tokens;
  if (live < 0 || live > C::tokens_capacity) __trap();
  const int token = blockIdx.y, column = blockIdx.x * 128 + threadIdx.x;
  if (token >= live || column >= C::n) return;
  const float* input = partials<C>(const_cast<void*>(scratch));
  if constexpr (P::retire)
    if (reinterpret_cast<uintptr_t>(input) & 127u) __trap();
  // With Fold the queue already folded each route's K blocks in this
  // order; its one value per route and column stands where block 0's was.
  constexpr int Blocks = P::fold ? 1 : Layout<C>::blocks;
  float total = 0.f;
#pragma unroll
  for (int k = 0; k < C::top_k; ++k) {
    const int route = token * C::top_k + k;
    const int expert = ids[route];
    if (expert < 0 || expert >= C::experts) continue;
    float value;
    if constexpr (P::fold) {
      value = __ldcg(input + uint64_t(route) * C::n + column);
    } else {
      value = 0.f;
#pragma unroll
      for (int block = 0; block < Blocks; ++block)
        value = Reduction::add(value, __ldcg(input +
            (uint64_t(route) * Blocks + block) * C::n + column));
    }
    routed[uint64_t(route) * C::n + column] = Output::output(value);
    total = Reduction::add(total, value);
  }
  output[uint64_t(token) * C::n + column] =
      Output::output(Reduction::add(total, shared[uint64_t(token) * C::n + column]));
  if constexpr (P::retire) {
    // Whole CTAs pass the early return together (C::n % 128 == 0), and the
    // warp's 32 columns of each (route, K block) row are one line.
    __syncwarp();
    const int lane = threadIdx.x & 31, column0 = column - lane;
    for (int line = lane; line < C::top_k * Blocks; line += 32) {
      const int route = token * C::top_k + line / Blocks;
      const int expert = ids[route];
      if (expert < 0 || expert >= C::experts) continue;
      discard_line(input + (uint64_t(route) * Blocks + line % Blocks) * C::n + column0);
    }
  }
}
// Lab timing utility, not a candidate: reads `words` 16-byte words, so the
// L2 then holds only clean lines of this buffer. A write flush leaves its
// own dirty lines, which the next timed launch must write back.
__device__ void read_flush(const uint4* data, uint64_t words, uint32_t* sink) {
  uint32_t folded = 0;
  for (uint64_t i = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x; i < words;
       i += uint64_t(gridDim.x) * blockDim.x) {
    const uint4 v = data[i];
    folded ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  // The buffer holds zeros; the store only keeps the loads.
  if (folded == 0x9e3779b9u) *sink = folded;
}
} // namespace chop::moe::lab_l2
