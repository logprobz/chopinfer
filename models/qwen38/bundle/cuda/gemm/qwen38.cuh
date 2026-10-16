#pragma once

// Qwen3.8 Flash dense GEMMs over fragment-major BF16 weights that the shared
// families refuse: a short-K entry for K below one 128-K block multiple (the
// hyper-connection up projection, K = 320) and a skinny entry for outputs of
// at most 32 rows per weight (in_proj_a with in_proj_b, and the inject
// projection). Its cluster decode entries run the shared template
// (cluster_decode.cuh). Only the Qwen cubins include this header, so the
// MiMo sources are unchanged.

#include "gemm/fragment_decode.cuh"
#include "gemm/fragment_prefill.cuh"
#include "common/ptx_control.cuh"
#include "common/convert.cuh"
#include "common/sigmoid.cuh"

namespace chop::gemm::qwen38 {

// Work items are (token tile, block of Warps strips), token tile major. CTA
// x takes the contiguous item range [x * items / grid, (x + 1) * items /
// grid), so every CTA gets the same share and restages its Tokens
// activation rows (whole K, padded rows for conflict-free ldmatrix) only
// when the token tile changes. Warp w owns strip w of each item and holds
// its K / 16 fragment tiles in registers; it loads the next item's strip
// before storing the current one.
template<int K, int Tokens, int Warps>
struct ShortKTraits {
  static_assert(K % 16 == 0 && K <= 512, "a strip's K stays in registers");
  static_assert(Tokens == 32 || Tokens == 64);
  static_assert(Warps == 4 || Warps == 8);
  static constexpr int k_tiles = K / 16;
  static constexpr int warps = Warps;
  static constexpr int row_bytes = K * 2 + 16;
  static constexpr int shared_bytes = Tokens * row_bytes;
};

template<int K, int Tokens, int Warps>
__device__ void run_short_k(
    const uint8_t* weight, const uint16_t* activation, void* output,
    uint32_t live_m, uint32_t n, uint32_t output_f32) {
  using T = ShortKTraits<K, Tokens, Warps>;
  extern __shared__ __align__(16) unsigned char storage[];
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const uint32_t blocks = n / (16 * T::warps);
  const uint32_t items = (live_m + Tokens - 1) / Tokens * blocks;
  uint32_t item = uint64_t(items) * blockIdx.x / gridDim.x;
  const uint32_t end = uint64_t(items) * (blockIdx.x + 1) / gridDim.x;
  ptx::A4 a[T::k_tiles];
  const auto load = [&](uint32_t it) {
    const uint32_t strip = it % blocks * T::warps + warp;
    const uint8_t* base = weight + uint64_t(strip) * T::k_tiles * 512 + lane * 16;
#pragma unroll
    for (int t = 0; t < T::k_tiles; ++t) a[t] = load_weight_fragment(base + t * 512);
  };
  ptx::griddep_launch_dependents();
  if (item < end) load(item);
  ptx::griddep_wait();
  const uint32_t b_row = (lane & 7) + ((lane >> 4) << 3);
  const uint32_t b_address = ptx::shared_address(storage) +
      b_row * T::row_bytes + ((lane >> 3) & 1) * 16;
  uint32_t staged = ~0u;
  for (; item < end; ++item) {
    const uint32_t tile = item / blocks;
    if (tile != staged) {
      __syncthreads();
      constexpr uint32_t vectors = K / 8;
      for (uint32_t i = threadIdx.x; i < Tokens * vectors; i += blockDim.x) {
        const uint32_t row = i / vectors;
        const uint32_t token = tile * Tokens + row;
        uint4 value = make_uint4(0, 0, 0, 0);
        if (token < live_m)
          value = reinterpret_cast<const uint4*>(
              activation + uint64_t(token) * K)[i % vectors];
        *reinterpret_cast<uint4*>(storage + row * T::row_bytes +
            (i % vectors) * 16) = value;
      }
      __syncthreads();
      staged = tile;
    }
    // Every 8 k16 tiles (128 K) and the last accumulate from zero into
    // `block`, then add into acc; run_hc_up_mix repeats this order.
    ptx::D4 acc[Tokens / 8], block[Tokens / 8];
#pragma unroll
    for (int j = 0; j < Tokens / 8; ++j) acc[j] = block[j] = ptx::D4{};
#pragma unroll
    for (int t = 0; t < T::k_tiles; ++t) {
#pragma unroll
      for (int pair = 0; pair < Tokens / 16; ++pair) {
        ptx::B2 first, second;
        prefill::load_b2(b_address + pair * 16 * T::row_bytes + t * 32, 0, 0,
            first, second);
        ptx::mma_bf16(block[pair * 2], a[t], first);
        ptx::mma_bf16(block[pair * 2 + 1], a[t], second);
      }
      if (t % 8 == 7 || t + 1 == T::k_tiles) {
#pragma unroll
        for (int j = 0; j < Tokens / 8; ++j)
#pragma unroll
          for (int v = 0; v < 4; ++v) {
            acc[j].v[v] += block[j].v[v];
            block[j].v[v] = 0.0f;
          }
      }
    }
    const uint32_t feature = (item % blocks * T::warps + warp) * 16 + (lane >> 2);
    if (item + 1 < end) load(item + 1);
    const uint32_t token = tile * Tokens + (lane & 3) * 2;
#pragma unroll
    for (int j = 0; j < Tokens / 8; ++j) {
      const float c[4] = {acc[j].v[0], acc[j].v[1], acc[j].v[2], acc[j].v[3]};
      if (output_f32)
        store_fragment<true>(output, c, feature, token + j * 8, live_m, n);
      else
        store_fragment<false>(output, c, feature, token + j * 8, live_m, n);
    }
  }
}

// Each weight is one or two 16-row fragment strips (Strips) whose first Rows
// rows are live; the kernel accumulates Units = Weights * Strips strips, unit
// u being strip u % Strips of weight u / Strips. With Rows <= 16 a unit is a
// weight. CTA x owns 32 tokens (x / splits) and K split x % splits, a whole
// number of 256-K chunks. One thread bulk-copies each chunk's live token rows
// into one of two padded shared buffers while the eight warps consume the
// other: warp w takes the chunk's fragment tiles 2w and 2w + 1, loading its
// weight fragments from global memory and activation fragments with
// ldmatrix. Warp partials are summed in warp order in shared memory; with
// several splits the last CTA of a token tile sums the split partials in
// split order, so the result does not depend on arrival order. Output row t
// holds weight w's Rows values at columns [w * Rows, (w + 1) * Rows) of a
// row that is output_stride elements long.
template<int K, int Rows, int Weights>
struct SkinnyTraits {
  static_assert(K % 256 == 0 && Rows >= 1 && Rows <= 32);
  static_assert(Weights == 1 || Weights == 2);
  static constexpr int strips = (Rows + 15) / 16;
  static constexpr int units = Weights * strips;
  static constexpr int warps = 8;
  static constexpr int tokens = 32;
  static constexpr int chunk = 256;
  static constexpr int chunks = K / chunk;
  static constexpr int row_bytes = chunk * 2 + 16;
  static constexpr int stage_bytes = tokens * row_bytes;
  static constexpr int outputs = units * 16 * tokens;
  static constexpr int sum_bytes = warps * outputs * 4;
  static constexpr int shared_bytes =
      2 * stage_bytes > sum_bytes ? 2 * stage_bytes : sum_bytes;
};

template<int K, int Rows, int Weights>
__device__ void run_skinny(
    const uint8_t* weight0, const uint8_t* weight1, const uint16_t* activation,
    void* output, float* partial, uint32_t* semaphores, uint32_t live_m,
    uint32_t output_stride, uint32_t splits, uint32_t output_f32) {
  using T = SkinnyTraits<K, Rows, Weights>;
  extern __shared__ __align__(16) unsigned char storage[];
  float* sums = reinterpret_cast<float*>(storage);
  __shared__ uint64_t full[2];
  __shared__ uint32_t last;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const uint32_t tile = blockIdx.x / splits;
  const uint32_t split = blockIdx.x % splits;
  const uint32_t first_token = tile * T::tokens;
  const uint32_t live_rows = min(live_m - first_token, uint32_t(T::tokens));
  const uint32_t begin = uint32_t(T::chunks) * split / splits;
  const uint32_t end = uint32_t(T::chunks) * (split + 1) / splits;
  constexpr int Units = T::units;
  // Unit u's strip: strip u % Strips of weight u / Strips, K / 16 fragment
  // tiles after the weight's previous strip.
  const uint8_t* weights[Units];
#pragma unroll
  for (int u = 0; u < Units; ++u)
    weights[u] = (u / T::strips == 0 ? weight0 : Weights == 2 ? weight1 : weight0) +
                 uint64_t(u % T::strips) * (K / 16) * 512 + lane * 16;
  const auto load_a = [&](uint32_t c, ptx::A4 (&a)[2][Units]) {
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int w = 0; w < Units; ++w)
        a[i][w] = load_weight_fragment(weights[w] +
            (uint64_t(c) * (T::chunk / 16) + warp * 2 + i) * 512);
  };
  // Rows past live M read as zero in both buffers.
  for (uint32_t i = threadIdx.x; i < 2 * (T::tokens - live_rows) * (T::chunk / 8);
       i += blockDim.x) {
    const uint32_t per_stage = (T::tokens - live_rows) * (T::chunk / 8);
    const uint32_t row = live_rows + i % per_stage / (T::chunk / 8);
    *reinterpret_cast<uint4*>(storage + i / per_stage * T::stage_bytes +
        row * T::row_bytes + i % (T::chunk / 8) * 16) = make_uint4(0, 0, 0, 0);
  }
  if (threadIdx.x == 0) {
    ptx::mbarrier_init(&full[0], 1);
    ptx::mbarrier_init(&full[1], 1);
    ptx::fence_mbarrier_init();
  }
  __syncthreads();
  ptx::A4 a[2][Units];
  ptx::griddep_launch_dependents();
  load_a(begin, a);
  ptx::griddep_wait();
  const auto issue = [&](uint32_t c) {
    uint64_t* bar = &full[(c - begin) & 1];
    unsigned char* stage = storage + ((c - begin) & 1) * T::stage_bytes;
    ptx::mbarrier_arrive_expect_tx(bar, live_rows * T::chunk * 2);
    for (uint32_t row = 0; row < live_rows; ++row)
      ptx::cp_async_bulk(stage + row * T::row_bytes,
          activation + uint64_t(first_token + row) * K + c * T::chunk,
          T::chunk * 2, bar);
  };
  if (threadIdx.x == 0) {
    issue(begin);
    if (begin + 1 < end) issue(begin + 1);
  }
  // Every four chunks (8 MMAs per accumulator) and the split's last chunk
  // accumulate from zero into `block`, then add into acc.
  ptx::D4 acc[Units][4], block[Units][4];
#pragma unroll
  for (int w = 0; w < Units; ++w)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[w][j] = block[w][j] = ptx::D4{};
  const uint32_t b_row = (lane & 7) + ((lane >> 4) << 3);
  const uint32_t b_offset = b_row * T::row_bytes + ((lane >> 3) & 1) * 16 +
      warp * 2 * 32;
  for (uint32_t c = begin; c < end; ++c) {
    const uint32_t slot = (c - begin) & 1;
    ptx::mbarrier_wait(&full[slot], ((c - begin) >> 1) & 1);
    const uint32_t b_address = ptx::shared_address(storage) +
        slot * T::stage_bytes + b_offset;
    ptx::A4 current[2][Units];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int w = 0; w < Units; ++w) current[i][w] = a[i][w];
    if (c + 1 < end) load_a(c + 1, a);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      ptx::B2 b[4];
      prefill::load_b2(b_address + i * 32, 0, 0, b[0], b[1]);
      prefill::load_b2(b_address + 16 * T::row_bytes + i * 32, 0, 0, b[2], b[3]);
#pragma unroll
      for (int w = 0; w < Units; ++w)
#pragma unroll
        for (int j = 0; j < 4; ++j) ptx::mma_bf16(block[w][j], current[i][w], b[j]);
    }
    if ((c - begin) % 4 == 3 || c + 1 == end) {
#pragma unroll
      for (int w = 0; w < Units; ++w)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int v = 0; v < 4; ++v) {
            acc[w][j].v[v] += block[w][j].v[v];
            block[w][j].v[v] = 0.0f;
          }
    }
    __syncthreads();
    if (threadIdx.x == 0 && c + 2 < end) issue(c + 2);
  }

  // sums[warp][unit][row][token], over the drained stage buffers.
  const int group = lane >> 2;
  const int lane4 = lane & 3;
#pragma unroll
  for (int w = 0; w < Units; ++w)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        const int row = group + (v / 2) * 8;
        const int token = j * 8 + lane4 * 2 + (v % 2);
        sums[((warp * Units + w) * 16 + row) * T::tokens + token] =
            acc[w][j].v[v];
      }
  __syncthreads();
  const auto store = [&](uint32_t index, float value) {
    const uint32_t unit = index / (16 * T::tokens);
    const uint32_t w = unit / T::strips;
    const uint32_t row = unit % T::strips * 16 + index / T::tokens % 16;
    const uint32_t token = first_token + index % T::tokens;
    if (row >= Rows || token >= live_m) return;
    const uint64_t out = uint64_t(token) * output_stride + w * Rows + row;
    if (output_f32) static_cast<float*>(output)[out] = value;
    else static_cast<__nv_bfloat16*>(output)[out] = __float2bfloat16(value);
  };
  for (uint32_t index = threadIdx.x; index < T::outputs; index += blockDim.x) {
    float value = 0.0f;
#pragma unroll
    for (int w = 0; w < T::warps; ++w) value += sums[w * T::outputs + index];
    if (splits == 1) store(index, value);
    else partial[(uint64_t(tile) * splits + split) * T::outputs + index] = value;
  }
  if (splits == 1) return;
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0)
    last = atomicAdd(&semaphores[tile], 1) == splits - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  for (uint32_t index = threadIdx.x; index < T::outputs; index += blockDim.x) {
    float value = 0.0f;
    for (uint32_t s = 0; s < splits; ++s)
      value += __ldcg(partial + (uint64_t(tile) * splits + s) * T::outputs + index);
    store(index, value);
  }
  __syncthreads();
  if (threadIdx.x == 0) atomicExch(&semaphores[tile], 0);
}

} // namespace chop::gemm::qwen38
