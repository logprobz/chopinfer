#pragma once

#include "chained_quantized.cuh"

// One expert's weight strip is read once per 128-K chain and reused for
// every routed row of that expert. The small queue is the GEMV/narrow-M
// case; the wide queue handles experts with 9..64 rows. Only the merge
// adds chains, in ascending K order, before the unchanged epilogue.
//
// With Tiled the weights are stored in 64-row by 128-K tiles
// (`row-tiles-64x128-groups`), as in chained_absorb.cuh: an item's loads
// are one contiguous 4096-byte code tile and one 256-byte scale tile, and
// each lane still loads the same 16 code bytes and scale byte.
namespace chop::moe::streamed {
constexpr int Threads = 128, Columns = 64, Stride = 136;

template<class C> struct Layout {
  static constexpr int blocks = C::loop_k / 128;
  static constexpr int header_bytes = ((2 + 4 * C::experts) * 4 + 255) / 256 * 256;
  static_assert(C::row_tile == 64 && C::tokens_capacity <= 64);
};

template<class C>
__device__ __forceinline__ float* partials(void* scratch) {
  return reinterpret_cast<float*>(static_cast<uint8_t*>(scratch) + Layout<C>::header_bytes);
}

// All lanes of one warp scan the already ordered active list. Each queue
// stores (expert, count); unused records are never read, including at M=0.
template<class C>
__device__ void pack(const int32_t* counts, const int32_t* active, const int32_t* live_tokens, void* scratch) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const int live = *live_tokens, total = active[0], lane = threadIdx.x;
  if (blockDim.x != 32 || gridDim.x != 1 || live < 0 || live > C::tokens_capacity ||
      total < 0 || total > C::experts || total > live * C::top_k) __trap();
  int32_t* table = static_cast<int32_t*>(scratch);
  int small_carry = 0, wide_carry = 0;
  for (int base = 0; base < total; base += 32) {
    const bool valid = base + lane < total;
    const int expert = valid ? active[1 + base + lane] : 0;
    if (expert < 0 || expert >= C::experts) __trap();
    const int count = valid ? counts[expert] : 0;
    if (valid && (count <= 0 || count > live)) __trap();
    const int small = valid && count <= 8, wide = valid && count > 8;
    int s = small, w = wide;
#pragma unroll
    for (int delta = 1; delta < 32; delta *= 2) {
      const int a = __shfl_up_sync(0xffffffffu, s, delta);
      const int b = __shfl_up_sync(0xffffffffu, w, delta);
      if (lane >= delta) { s += a; w += b; }
    }
    if (small) {
      const int at = 2 + 2 * (small_carry + s - 1);
      table[at] = expert; table[at + 1] = count;
    }
    if (wide) {
      const int at = 2 + 2 * C::experts + 2 * (wide_carry + w - 1);
      table[at] = expert; table[at + 1] = count;
    }
    small_carry += __shfl_sync(0xffffffffu, s, 31);
    wide_carry += __shfl_sync(0xffffffffu, w, 31);
  }
  if (lane == 0) { table[0] = small_carry; table[1] = wide_carry; }
}

// Preserve the unaligned byte-view domain. Packed serving matrices take
// the vector branch: every row stride and K strip is a multiple of 16.
__device__ __forceinline__ uint4 load16(const uint8_t* p) {
  if ((reinterpret_cast<uintptr_t>(p) & 15u) == 0) return *reinterpret_cast<const uint4*>(p);
  uint32_t words[4] = {};
#pragma unroll
  for (int i = 0; i < 16; ++i) words[i / 4] |= uint32_t(p[i]) << (8 * (i % 4));
  return make_uint4(words[0], words[1], words[2], words[3]);
}

// Check all 32 E2M1 values once at load, rather than at each MMA operand.
// UE8M0 253 forbids magnitudes 6/7; 254 forbids the magnitude's bit 2.
__device__ __forceinline__ void check(uint4 codes, uint8_t scale) {
  if (scale == 255) __trap();
  if (scale > 252) {
    uint32_t bad;
    if (scale == 254) bad = codes.x | codes.y | codes.z | codes.w;
    else bad = (codes.x & (codes.x << 1)) | (codes.y & (codes.y << 1)) |
               (codes.z & (codes.z << 1)) | (codes.w & (codes.w << 1));
    if (bad & 0x44444444u) __trap();
  }
}

__device__ __forceinline__ uint32_t shuffle(uint32_t word, int source) {
  return __shfl_sync(0xffffffffu, word, source);
}

template<class C, int Rows, bool Tiled = false>
__device__ void compute(const uint8_t* input, const uint8_t* weight,
    const uint8_t* weight_scales, const uint8_t* activation_scales,
    const int32_t* offsets, const int32_t* sorted_rows, const int32_t* padded_rows,
    const int32_t* live_tokens, void* scratch) {
  using L = Layout<C>;
  constexpr int H = C::halves, Fragments = Rows / 8;
  static_assert(Rows == 8 || Rows == 64);
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

    // Four neighboring lanes own successive 32-K groups of each weight
    // row. Each issues a 16-byte load and one scale load, once per chain.
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
  }
}

template<class C>
__device__ void merge(const void* scratch, const int32_t* live_tokens,
    const float* route_weights, void* output) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const int live = *live_tokens;
  if (live < 0 || live > C::tokens_capacity) __trap();
  const float* input = partials<C>(const_cast<void*>(scratch));
  const uint64_t elements = uint64_t(live) * C::top_k * C::n;
  for (uint64_t at = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; at < elements;
       at += uint64_t(gridDim.x) * blockDim.x) {
    const int route = at / C::n, column = at % C::n;
    float sum[C::halves] = {};
#pragma unroll 1
    for (int block = 0; block < Layout<C>::blocks; ++block)
#pragma unroll
      for (int h = 0; h < C::halves; ++h)
        sum[h] = __fadd_rn(sum[h], input[((uint64_t(route) * C::halves + h) * Layout<C>::blocks + block) * C::n + column]);
    if constexpr (C::gate_up)
      static_cast<uint16_t*>(output)[at] = convert::bf16_from_f32(chained::before_down<typename C::clamp>(sum[0], sum[1], route_weights[route]));
    else static_cast<float*>(output)[at] = sum[0];
  }
}
} // namespace chop::moe::streamed
