#pragma once
#include "chained_streamed.cuh"

// The streamed small queue that also runs the wide queue's experts, so no
// wide launch follows it. Wide experts take the first slots, so their longer
// items start first. A wide item loads its weight strip into registers once,
// as every item does, then runs the small queue's 8-row fragment over each
// successive 8 rows of the expert. Each pass stages, decodes, multiplies
// from +0 and stores exactly as the wide queue's fragment of those rows, so
// every chain and partial is unchanged and the merge and finishers are the
// served ones.
//
// With Tiled the weights are stored in 64-row by 128-K tiles
// (`row-tiles-64x128-groups`): tile (strip, block) holds the 64 rows of one
// 64-row strip over one 128-K block, 64 code bytes and 4 scale bytes per
// row, tiles in K order within each strip. An item's loads are then one
// contiguous 4096-byte code tile and one 256-byte scale tile; each lane
// still loads the same 16 code bytes and scale byte.
namespace chop::moe::absorb {
using namespace streamed;
constexpr int Rows = 8;

template<class C, bool Tiled = false, class Scales = RawScales>
__device__ void compute(const uint8_t* input, const uint8_t* weight,
    const uint8_t* weight_scales, const uint8_t* activation_scales,
    const int32_t* offsets, const int32_t* sorted_rows, const int32_t* padded_rows,
    const int32_t* live_tokens, void* scratch) {
  using L = Layout<C>;
  constexpr int H = C::halves;
  static_assert(!Tiled || (C::k % 128 == 0 && C::loop_k == C::k && C::n % 64 == 0 &&
                           C::weight_rows % 64 == 0 && Columns == 64 && Threads == 128),
                "row tiles need whole 64x128 tiles, one per item");
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

    const uint32_t* scale_words = nullptr;
    if constexpr (Scales::staged) {
      scale_words = Scales::template prepare<C, 1>(weight_scales, expert, column, block);
      __syncthreads();
    }

    // The small and wide queues' lane ownership: one 16-byte load and one
    // scale load per lane, row and half, once per item.
    uint4 owned[H][2];
    uint32_t scales[H][2];
    const int k = block * 128 + t * 32;
#pragma unroll
    for (int h = 0; h < H; ++h)
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const uint64_t wr = uint64_t(expert) * C::weight_rows + h * C::n + column + warp * 16 + g + r * 8;
        const bool valid = k < C::k;
        if constexpr (Tiled) {
          // The item's 64 columns are one strip, so wr's tile is the item's
          // and its row in the tile is the lane's.
          const uint64_t tile = (uint64_t(expert) * C::weight_rows + h * C::n + column) / 64 * L::blocks + block;
          const int row = warp * 16 + g + r * 8;
          owned[h][r] = valid ? load16(weight + tile * 4096 + row * 64 + t * 16) : make_uint4(0,0,0,0);
          scales[h][r] = valid ? weight_scales[tile * 256 + row * 4 + t] : 127;
        } else {
          owned[h][r] = valid ? load16(weight + wr * (C::k / 2) + k / 2) : make_uint4(0,0,0,0);
          if constexpr (Scales::staged) {
            static_assert(!C::gate_up && H == 1);
            scales[h][r] = valid ? ((scale_words[warp * 16 + g + r * 8] >> (8 * t)) & 255) : 127;
          } else {
            scales[h][r] = valid ? weight_scales[wr * (C::k / 32) + k / 32] : 127;
          }
        }
        check(owned[h][r], uint8_t(scales[h][r]));
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
              a[h].v[r] = convert::scaled_mxfp4_bf16x2<false>(uint8_t(low >> (8 * t)), uint8_t(scale[h][r]));
              a[h].v[r + 2] = convert::scaled_mxfp4_bf16x2<false>(uint8_t(high >> (8 * t)), uint8_t(scale[h][r]));
            }
          const int col = group * 32 + half * 16 + 2 * t;
          ptx::B2 b;
          b.v[0] = chained::lds32(&activations[g][col]);
          b.v[1] = chained::lds32(&activations[g][col + 8]);
#pragma unroll
          for (int h = 0; h < H; ++h) ptx::mma_bf16(chain[h], a[h], b);
        }
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int route = rows[2 * t + (i & 1)];
        if (route < 0) continue;
        const int n = column + warp * 16 + g + (i >= 2 ? 8 : 0);
#pragma unroll
        for (int h = 0; h < H; ++h)
          out[((uint64_t(route) * H + h) * L::blocks + block) * C::n + n] = chain[h].v[i];
      }
      __syncthreads();
    }
  }
}
} // namespace chop::moe::absorb
