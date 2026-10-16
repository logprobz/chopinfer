#pragma once

// Head-batched BF16 GEMM (kernel-core `HeadBatchedGemmQuery`): for every row
// and head h, y[row, h, :] = W_h x[row, h, :], with head h's weight at a
// launch-supplied element stride from the first. Latent attention absorbs
// its up projection's key half into the query and its value half into the
// output with it.

#include <stdint.h>
#include "../attention/tile.cuh"

namespace chop::latent {

template<int Heads, int N, int K, bool KByN, bool OutF32>
struct HeadGemm {
  static constexpr int heads = Heads, n = N, k = K;
  static constexpr bool k_by_n = KByN, out_f32 = OutF32;
  // A CTA computes 64 rows x 64 columns of one head; warp w takes rows
  // 16w..16w + 15. K streams in 128-wide chunks, one MMA chain each.
  static constexpr int tile_m = 64, tile_n = 64, chunk = 128;
  static constexpr int warps = 4, threads = warps * 32, stages = 2;
  static_assert(N % tile_n == 0 && K % chunk == 0);
  static constexpr int chunks = K / chunk;
  static constexpr int x_bytes = tile_m * chunk * 2;
  static constexpr int w_bytes = chunk * tile_n * 2;
  static constexpr int stage_bytes = x_bytes + w_bytes;
  static constexpr int smem = stages * stage_bytes;
};

// Grid [ceil(prepared rows / 64), N / 64, heads], 128 threads. The first
// weight chunk is prefetched before the PDL wait. Each 128-K chunk is one
// chain of eight MMAs from zero, added into the accumulator in K order; the
// output rounds once. Rows past the live count are zero filled and not
// written.
template<class T>
__device__ void head_gemm(const uint16_t* x, const uint16_t* w, const uint32_t* live_rows,
                          void* y, uint64_t head_stride) {
  using attention::cp_async_commit;
  using attention::cp_async_wait;
  using attention::cp_async_zfill;
  using attention::tile_offset;
  ptx::griddep_launch_dependents();
  extern __shared__ __align__(1024) uint8_t head_gemm_smem[];
  const int m0 = blockIdx.x * T::tile_m;
  const int n0 = blockIdx.y * T::tile_n;
  const int head = blockIdx.z;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int g = lane / 4;
  const int c = lane % 4;
  const uint16_t* weight = w + uint64_t(head) * head_stride;
  auto x_tile = [&](int stage) { return head_gemm_smem + stage * T::stage_bytes; };
  auto w_tile = [&](int stage) { return x_tile(stage) + T::x_bytes; };
  auto phase_of = [](const void* p) { return (ptx::shared_address(p) / 128) % 8; };

  // KByN stores [k][n] chunk rows of 128 bytes; NByK stores [n][k] rows of
  // 256 bytes. Both use the attention tiles' 64-element swizzled blocks.
  auto issue_w = [&](int chunk, int stage) {
    uint8_t* tile = w_tile(stage);
    const uint32_t phase = phase_of(tile);
    for (int i = threadIdx.x; i < T::chunk * T::tile_n / 8; i += T::threads) {
      if constexpr (T::k_by_n) {
        const int kk = i / (T::tile_n / 8);
        const int sector = i % (T::tile_n / 8);
        cp_async_zfill(tile + tile_offset<T::chunk>(kk, sector * 8, phase),
                       weight + uint64_t(chunk * T::chunk + kk) * T::n + n0 + sector * 8, true);
      } else {
        const int nn = i / (T::chunk / 8);
        const int sector = i % (T::chunk / 8);
        cp_async_zfill(tile + tile_offset<T::tile_n>(nn, sector * 8, phase),
                       weight + uint64_t(n0 + nn) * T::k + chunk * T::chunk + sector * 8, true);
      }
    }
  };
  auto issue_x = [&](int chunk, int stage, int live) {
    uint8_t* tile = x_tile(stage);
    const uint32_t phase = phase_of(tile);
    for (int i = threadIdx.x; i < T::tile_m * T::chunk / 8; i += T::threads) {
      const int r = i / (T::chunk / 8);
      const int sector = i % (T::chunk / 8);
      const bool valid = m0 + r < live;
      const uint16_t* source =
          valid ? x + (uint64_t(m0 + r) * T::heads + head) * T::k + chunk * T::chunk + sector * 8
                : x;
      cp_async_zfill(tile + tile_offset<T::tile_m>(r, sector * 8, phase), source, valid);
    }
  };

  issue_w(0, 0);
  ptx::griddep_wait();
  const int live = int(*live_rows);
  if (m0 >= live) {
    cp_async_commit();
    cp_async_wait<0>();
    return;
  }
  issue_x(0, 0, live);
  cp_async_commit();

  ptx::D4 acc[T::tile_n / 8];
#pragma unroll
  for (int t = 0; t < T::tile_n / 8; ++t)
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[t].v[i] = 0.0f;

  for (int chunk = 0; chunk < T::chunks; ++chunk) {
    const int stage = chunk % T::stages;
    if (chunk + 1 < T::chunks) {
      issue_w(chunk + 1, (chunk + 1) % T::stages);
      issue_x(chunk + 1, (chunk + 1) % T::stages, live);
    }
    cp_async_commit();
    cp_async_wait<1>();
    __syncthreads();
    const uint8_t* xs = x_tile(stage);
    const uint8_t* ws = w_tile(stage);
    const uint32_t x_phase = phase_of(xs);
    const uint32_t w_phase = phase_of(ws);
    ptx::D4 part[T::tile_n / 8];
#pragma unroll
    for (int t = 0; t < T::tile_n / 8; ++t)
#pragma unroll
      for (int i = 0; i < 4; ++i) part[t].v[i] = 0.0f;
#pragma unroll
    for (int ks = 0; ks < T::chunk / 16; ++ks) {
      uint32_t a[4];
      ptx::ldmatrix_b16<4>(a, xs + tile_offset<T::tile_m>(
          warp * 16 + ((lane / 8) % 2) * 8 + lane % 8, ks * 16 + (lane / 16) * 8, x_phase));
#pragma unroll
      for (int np = 0; np < T::tile_n / 16; ++np) {
        uint32_t b[4];
        if constexpr (T::k_by_n) {
          ptx::ldmatrix_b16<4, true>(b, ws + tile_offset<T::chunk>(
              ks * 16 + ((lane / 8) % 2) * 8 + lane % 8, np * 16 + (lane / 16) * 8, w_phase));
        } else {
          ptx::ldmatrix_b16<4>(b, ws + tile_offset<T::tile_n>(
              np * 16 + (lane / 16) * 8 + lane % 8, ks * 16 + ((lane / 8) % 2) * 8, w_phase));
        }
        ptx::mma_bf16(part[2 * np], ptx::A4{{a[0], a[1], a[2], a[3]}}, ptx::B2{{b[0], b[1]}});
        ptx::mma_bf16(part[2 * np + 1], ptx::A4{{a[0], a[1], a[2], a[3]}},
                      ptx::B2{{b[2], b[3]}});
      }
    }
#pragma unroll
    for (int t = 0; t < T::tile_n / 8; ++t)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[t].v[i] += part[t].v[i];
    // Every warp has left this stage before the next chunk's copies land in
    // it.
    __syncthreads();
  }

#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const int r = m0 + warp * 16 + g + half * 8;
    if (r >= live) continue;
    const uint64_t base = (uint64_t(r) * T::heads + head) * T::n + n0;
#pragma unroll
    for (int t = 0; t < T::tile_n / 8; ++t) {
      const float v0 = acc[t].v[2 * half], v1 = acc[t].v[2 * half + 1];
      if constexpr (T::out_f32) {
        *reinterpret_cast<float2*>(static_cast<float*>(y) + base + t * 8 + 2 * c) =
            make_float2(v0, v1);
      } else {
        *reinterpret_cast<uint32_t*>(static_cast<uint16_t*>(y) + base + t * 8 + 2 * c) =
            convert::bf16x2_from_f32({v0, v1});
      }
    }
  }
}

} // namespace chop::latent
