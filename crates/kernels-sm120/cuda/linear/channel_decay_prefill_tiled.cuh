#pragma once

// A faster schedule of the per-channel-decay delta rule's chunked prefill
// (channel_decay_prefill.cuh) with the same workspace and the same
// arithmetic: every stored value is the same chain of IEEE F32 operations in
// the same order, so prepare, scan and both mixed pairings give the staged
// entries' bits. Only the work assignment changes:
//
// `prepare_tiled` (one CTA per aligned chunk and head, 256 threads):
//   - q, k and beta rows (warps 4-7) and the running sums of g (warps 0-3,
//     the f loads issued before the ordered adds) at once;
//   - the decayed products in 2 x 2 blocks of (i, j) pairs per thread, four
//     x at a time from 16-byte shared loads, with the 64 pairs of the last
//     16 blocks as single pairs on warps 0-1;
//   - the forward substitution by columns: lane j of warps 0-1 owns column j
//     of T and walks four rows at a time, each element still the chain
//     sum_{m=j+1}^{i-1} A[i][m] T[m][j] in ascending m added to A[i][j],
//     with no barrier; warps 2-7 form beta k exp(G) and beta v meanwhile;
//   - W and U in 4 x 4 register tiles per thread, the two row blocks of a
//     thread balanced over the triangle.
// `scan_tiled` (16 value rows of a head per CTA, 256 threads): three phases
// per chunk with one barrier each, the next operand streaming by cp.async
// into the other of two 64 x 128 tiles while the current one is used
// (W, then QD and P, then KD and EGL); V' and O in 2 x 2 (row, value)
// register tiles, the state update in 2 x 4 (key, value) tiles.
//
// Shared tiles permute the 16-byte chunks of each row (XOR swizzle) so the
// tiled accesses are free of bank conflicts.
//
// Test-only mutants, each compiled into its own kernel-lab cubin (serving
// cubins define none of them), which kernel-lab must catch:
//   CHOP_CHANNEL_DECAY_TILED_TEST_LATE_WAIT: the dependency wait moves after
//     the first input reads of both stages;
//   CHOP_CHANNEL_DECAY_TILED_TEST_NO_TILE_WAIT: the scan reads its streamed
//     tiles without waiting for the copies;
//   CHOP_CHANNEL_DECAY_TILED_TEST_REVERSED_CHAIN: the scan's V' chains run
//     in descending x;
//   CHOP_CHANNEL_DECAY_TILED_TEST_SHORT_TRIANGLE: the forward substitution
//     drops each element's last term;
//   CHOP_CHANNEL_DECAY_TILED_TEST_NO_SINGLES: the 64 single pairs of rows 62
//     and 63 are never formed.

#include "channel_decay_prefill.cuh"

namespace chop::linear::channel_decay {
namespace tiled {

constexpr int kThreads = 256;
constexpr int kDim = 128;

// Float offset of element x of row `row` in a [rows][Cols] F32 tile whose
// 16-byte chunks are XOR-permuted by `swizzle` (less than Cols / 4).
template<int Cols>
__device__ __forceinline__ int at(int row, int x, int swizzle) {
  return row * Cols + ((((x >> 2) ^ swizzle) << 2) | (x & 3));
}
// Float offset of chunk c (four elements) of row `row`.
template<int Cols>
__device__ __forceinline__ int quad(int row, int c, int swizzle) {
  return row * Cols + ((c ^ swizzle) << 2);
}

__device__ __forceinline__ float lane4(const float4& v, int e) {
  return e == 0 ? v.x : e == 1 ? v.y : e == 2 ? v.z : v.w;
}

__device__ __forceinline__ void cp_async16(float* shared, const float* global) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
               :: "r"(ptx::shared_address(shared)), "l"(global) : "memory");
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}
__device__ __forceinline__ void cp_async_wait_all() {
#ifndef CHOP_CHANNEL_DECAY_TILED_TEST_NO_TILE_WAIT
  asm volatile("cp.async.wait_all;" ::: "memory");
#endif
}

// The PDL entry: release dependents, then wait for the producer (the late-
// wait mutant waits in `late_wait` instead).
__device__ __forceinline__ void begin() {
#ifdef CHOP_CHANNEL_DECAY_TILED_TEST_LATE_WAIT
  ptx::griddep_launch_dependents();
#else
  chop::linear::begin();
#endif
}
__device__ __forceinline__ void late_wait() {
#ifdef CHOP_CHANNEL_DECAY_TILED_TEST_LATE_WAIT
  ptx::griddep_wait();
#endif
}

// ---------------------------------------------------------------- prepare --

// Dynamic shared memory of the tiled prepare (floats): K and G [64][128]
// with chunk swizzle (row >> 1) & 7, beta [64], then Q [64][128]; once Q is
// dead its space holds A (the strictly lower products) [64][64] and TT
// [64][68], TT[j][i] = T[i][j].
struct PrepareTiles {
  static constexpr int Tile = kChunk * kDim;
  static constexpr int K = 0;
  static constexpr int G = Tile;
  static constexpr int Beta = 2 * Tile;
  static constexpr int Q = Beta + kChunk;
  static constexpr int A = Q;
  static constexpr int TTStride = kChunk + 4;
  static constexpr int TT = A + kChunk * kChunk;
  static constexpr int Floats = TT + kChunk * TTStride;
  static_assert(Floats >= Q + Tile && Q % 4 == 0 && TT % 4 == 0);
  static_assert(Floats * 4 <= 101376);
};

__device__ __forceinline__ int key_swizzle(int row) { return (row >> 1) & 7; }

// The decayed products of rows i0.. (NI) and j0.. (NJ) over the 128 key
// channels, in ascending x: term = k_j exp(G_i - G_j), acc_a += k_i term,
// acc_p += q_i term.
template<int NI, int NJ>
__device__ __forceinline__ void decayed_products(const float* K, const float* Q,
                                                 const float* G, int i0, int j0,
                                                 float (&acc_a)[NI][NJ],
                                                 float (&acc_p)[NI][NJ]) {
  #pragma unroll
  for (int a = 0; a < NI; ++a)
    #pragma unroll
    for (int b = 0; b < NJ; ++b) acc_a[a][b] = acc_p[a][b] = 0.f;
  #pragma unroll 2
  for (int c = 0; c < kDim / 4; ++c) {
    float4 gi[NI], ki[NI], qi[NI], gj[NJ], kj[NJ];
    #pragma unroll
    for (int a = 0; a < NI; ++a) {
      const int at4 = quad<kDim>(i0 + a, c, key_swizzle(i0 + a));
      gi[a] = *reinterpret_cast<const float4*>(G + at4);
      ki[a] = *reinterpret_cast<const float4*>(K + at4);
      qi[a] = *reinterpret_cast<const float4*>(Q + at4);
    }
    #pragma unroll
    for (int b = 0; b < NJ; ++b) {
      const int at4 = quad<kDim>(j0 + b, c, key_swizzle(j0 + b));
      gj[b] = *reinterpret_cast<const float4*>(G + at4);
      kj[b] = *reinterpret_cast<const float4*>(K + at4);
    }
    #pragma unroll
    for (int e = 0; e < 4; ++e)
      #pragma unroll
      for (int a = 0; a < NI; ++a)
        #pragma unroll
        for (int b = 0; b < NJ; ++b) {
          const float term =
              __fmul_rn(lane4(kj[b], e), expf(__fsub_rn(lane4(gi[a], e), lane4(gj[b], e))));
          acc_a[a][b] = fmaf(lane4(ki[a], e), term, acc_a[a][b]);
          acc_p[a][b] = fmaf(lane4(qi[a], e), term, acc_p[a][b]);
        }
  }
}

// Block b of the 2 x 2 pair blocks (I, J), J <= I, in row-major order.
__device__ __forceinline__ void pair_block(int b, int& I, int& J) {
  int i = int((sqrtf(8.f * float(b) + 1.f) - 1.f) * 0.5f);
  while ((i + 1) * (i + 2) / 2 <= b) ++i;
  while (i * (i + 1) / 2 > b) --i;
  I = i;
  J = b - i * (i + 1) / 2;
}

// Grid, operands and workspace as channel_delta_prepare; block kThreads,
// dynamic shared PrepareTiles::Floats * 4 bytes.
template<int Heads, int Dim, int Gate, uint32_t BoundBits>
__device__ void channel_delta_prepare_tiled(const uint16_t* qkv, uint32_t qkv_stride,
                                            const uint16_t* f, uint32_t f_stride,
                                            const uint16_t* b, uint32_t b_stride,
                                            const float* a_log, const float* dt_bias,
                                            float* workspace, const int32_t* cu_seqlens,
                                            const int32_t* start_positions,
                                            const uint32_t* live_sequences) {
  static_assert(Dim == kDim && kPrefillThreads == kThreads);
  using Work = ChunkWork<Dim>;
  using L = PrepareTiles;
  extern __shared__ float4 channel_decay_tiled_smem4[];
  float* smem = reinterpret_cast<float*>(channel_decay_tiled_smem4);
  float* K = smem + L::K;
  float* G = smem + L::G;
  float* beta = smem + L::Beta;
  float* Q = smem + L::Q;
  float* A = smem + L::A;
  float* TT = smem + L::TT;
  begin();
  AlignedChunk chunk;
  if (!locate_aligned_chunk(cu_seqlens, start_positions, *live_sequences, blockIdx.y, chunk))
    return;
  const int head = blockIdx.x;
  const int len = chunk.rows;
  const int t = threadIdx.x;
  const int lane = t & 31, warp = t >> 5;
  const float a = expf(a_log[head]);
  float* work = workspace + (uint64_t(uint32_t(blockIdx.y)) * Heads + head) * Work::Floats;

  if (warp < 4) {
    // Running sums of g per channel x = t in row order; constant past the
    // end. g rounds to F32 before the add (no contraction into an FMA).
    const int x = t;
    const float bias = dt_bias[head * Dim + x];
    const uint16_t* column = f + uint64_t(chunk.first_row) * f_stride + head * Dim + x;
    float running = 0.f;
    #pragma unroll
    for (int r0 = 0; r0 < kChunk; r0 += 32) {
      uint16_t raw[32];
      #pragma unroll
      for (int r = 0; r < 32; ++r)
        raw[r] = r0 + r < len ? column[uint64_t(r0 + r) * f_stride] : uint16_t(0);
      #pragma unroll
      for (int r = 0; r < 32; ++r) {
        if (r0 + r < len)
          running = __fadd_rn(running, log_decay<Gate, BoundBits>(bf(raw[r]), bias, a));
        G[at<kDim>(r0 + r, x, key_swizzle(r0 + r))] = running;
      }
    }
  } else {
    // Normalized q (scaled) and k rows r = warp - 4 + 4n, and beta; zero
    // past the chunk's end. Lane l holds elements 4l..4l+3.
    #pragma unroll 1
    for (int n0 = 0; n0 < kChunk / 4; n0 += 4) {
      uint2 qp[4], kp[4];
      uint16_t braw[4];
      #pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int r = warp - 4 + 4 * (n0 + u);
        qp[u] = kp[u] = make_uint2(0u, 0u);
        braw[u] = 0;
        if (r < len) {
          const uint16_t* base = qkv + uint64_t(chunk.first_row + r) * qkv_stride;
          qp[u] = *reinterpret_cast<const uint2*>(base + head * Dim + lane * 4);
          kp[u] = *reinterpret_cast<const uint2*>(base + (Heads + head) * Dim + lane * 4);
          braw[u] = b[uint64_t(chunk.first_row + r) * b_stride + head];
        }
      }
      #pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int r = warp - 4 + 4 * (n0 + u);
        float qn[4] = {0.f, 0.f, 0.f, 0.f}, kn[4] = {0.f, 0.f, 0.f, 0.f};
        float rb = 0.f;
        if (r < len) {
          float q[4] = {bf_lo(qp[u].x), bf_hi(qp[u].x), bf_lo(qp[u].y), bf_hi(qp[u].y)};
          float k[4] = {bf_lo(kp[u].x), bf_hi(kp[u].x), bf_lo(kp[u].y), bf_hi(kp[u].y)};
          const float q_norm = l2_norm4(q);
          const float k_norm = l2_norm4(k);
          #pragma unroll
          for (int j = 0; j < 4; ++j) {
            qn[j] = (q[j] / q_norm) * query_scale<Dim>();
            kn[j] = k[j] / k_norm;
          }
          rb = beta_of(braw[u]);
        }
        const int at4 = quad<kDim>(r, lane, key_swizzle(r));
        *reinterpret_cast<float4*>(Q + at4) = make_float4(qn[0], qn[1], qn[2], qn[3]);
        *reinterpret_cast<float4*>(K + at4) = make_float4(kn[0], kn[1], kn[2], kn[3]);
        if (lane == 0) beta[r] = rb;
      }
    }
  }
  late_wait();
  __syncthreads();

  // Decayed q and k rows, the chunk's total decay, and the zero upper
  // blocks of P.
  for (int e = t; e < kChunk * Dim / 4; e += kThreads) {
    const int r = e >> 5, c = e & 31;
    const float4 q = *reinterpret_cast<const float4*>(Q + quad<kDim>(r, c, key_swizzle(r)));
    const float4 k = *reinterpret_cast<const float4*>(K + quad<kDim>(r, c, key_swizzle(r)));
    const float4 g = *reinterpret_cast<const float4*>(G + quad<kDim>(r, c, key_swizzle(r)));
    const float4 last =
        *reinterpret_cast<const float4*>(G + quad<kDim>(len - 1, c, key_swizzle(len - 1)));
    float4 qd, kd;
    qd.x = q.x * expf(g.x);
    qd.y = q.y * expf(g.y);
    qd.z = q.z * expf(g.z);
    qd.w = q.w * expf(g.w);
    kd.x = k.x * expf(last.x - g.x);
    kd.y = k.y * expf(last.y - g.y);
    kd.z = k.z * expf(last.z - g.z);
    kd.w = k.w * expf(last.w - g.w);
    *reinterpret_cast<float4*>(work + Work::QD + r * Dim + 4 * c) = qd;
    *reinterpret_cast<float4*>(work + Work::KD + r * Dim + 4 * c) = kd;
  }
  if (t < Dim / 4) {
    const float4 last =
        *reinterpret_cast<const float4*>(G + quad<kDim>(len - 1, t, key_swizzle(len - 1)));
    *reinterpret_cast<float4*>(work + Work::EGL + 4 * t) =
        make_float4(expf(last.x), expf(last.y), expf(last.z), expf(last.w));
  }
  for (int e = t; e < kChunk * kChunk / 2; e += kThreads) {
    const int i = e >> 5, J = e & 31;
    if (J > (i >> 1))
      *reinterpret_cast<float2*>(work + Work::P + i * kChunk + 2 * J) = make_float2(0.f, 0.f);
  }

  // Pairwise decayed products: blocks t and t + 256 of the 528 2 x 2 blocks
  // with J <= I, then the last 16 blocks (rows 62 and 63, columns 32..63)
  // as single pairs on threads 0..63. A stays in registers until Q is dead.
  float strict[2][2][2];
  #pragma unroll
  for (int round = 0; round < 2; ++round) {
    int I, J;
    pair_block(t + kThreads * round, I, J);
    const int i0 = 2 * I, j0 = 2 * J;
    float acc_a[2][2], acc_p[2][2];
    if (i0 < len) {
      decayed_products<2, 2>(K, Q, G, i0, j0, acc_a, acc_p);
    } else {
      #pragma unroll
      for (int u = 0; u < 2; ++u)
        #pragma unroll
        for (int v = 0; v < 2; ++v) acc_a[u][v] = acc_p[u][v] = 0.f;
    }
    #pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int i = i0 + u;
      float p[2];
      #pragma unroll
      for (int v = 0; v < 2; ++v) {
        const int j = j0 + v;
        strict[round][u][v] = i < len && j < i ? __fmul_rn(-beta[i], acc_a[u][v]) : 0.f;
        p[v] = i < len && j <= i ? acc_p[u][v] : 0.f;
      }
      *reinterpret_cast<float2*>(work + Work::P + i * kChunk + j0) = make_float2(p[0], p[1]);
    }
  }
  float strict_single = 0.f;
  const int single_i = 62 + (t >> 5), single_j = 32 + lane;
#ifdef CHOP_CHANNEL_DECAY_TILED_TEST_NO_SINGLES
#define CHOP_CHANNEL_DECAY_TILED_SINGLES(t) false
#else
#define CHOP_CHANNEL_DECAY_TILED_SINGLES(t) ((t) < 64)
#endif
  if (CHOP_CHANNEL_DECAY_TILED_SINGLES(t)) {
    float acc_a[1][1], acc_p[1][1];
    acc_a[0][0] = acc_p[0][0] = 0.f;
    if (single_i < len) decayed_products<1, 1>(K, Q, G, single_i, single_j, acc_a, acc_p);
    const bool live = single_i < len;
    strict_single =
        live && single_j < single_i ? __fmul_rn(-beta[single_i], acc_a[0][0]) : 0.f;
    work[Work::P + single_i * kChunk + single_j] =
        live && single_j <= single_i ? acc_p[0][0] : 0.f;
  }
  __syncthreads();

  // A (strictly lower, row-major) and TT = I in Q's space.
  for (int e = t; e < kChunk * L::TTStride; e += kThreads) {
    const int j = e / L::TTStride, i = e % L::TTStride;
    TT[e] = i == j ? 1.f : 0.f;
  }
  #pragma unroll
  for (int round = 0; round < 2; ++round) {
    int I, J;
    pair_block(t + kThreads * round, I, J);
    #pragma unroll
    for (int u = 0; u < 2; ++u)
      *reinterpret_cast<float2*>(A + (2 * I + u) * kChunk + 2 * J) =
          make_float2(strict[round][u][0], strict[round][u][1]);
  }
  if (t < 64) A[single_i * kChunk + single_j] = strict_single;
  __syncthreads();

  if (warp < 2) {
    // Forward substitution by columns: lane j owns column j, T[i][j] =
    // A[i][j] + sum_{m=j+1}^{i-1} A[i][m] T[m][j] in ascending m, rows
    // i0..i0+3 sharing the loads of their common terms m < i0.
    const int j = t;
    float* own = TT + j * L::TTStride;
    #pragma unroll 1
    for (int i0 = 0; i0 < len; i0 += 4) {
      if (i0 + 3 <= j) continue;
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
      #pragma unroll 1
      for (int m0 = 0; m0 < i0; m0 += 4) {
        const float4 tm = *reinterpret_cast<const float4*>(own + m0);
        float4 am[4];
        #pragma unroll
        for (int r = 0; r < 4; ++r)
          am[r] = *reinterpret_cast<const float4*>(A + (i0 + r) * kChunk + m0);
        #pragma unroll
        for (int e = 0; e < 4; ++e)
          if (m0 + e > j) {
            #pragma unroll
            for (int r = 0; r < 4; ++r)
              acc[r] = fmaf(lane4(am[r], e), lane4(tm, e), acc[r]);
          }
      }
      #pragma unroll
      for (int r = 0; r < 4; ++r) {
        const int i = i0 + r;
        if (i > j && i < len) {
#ifdef CHOP_CHANNEL_DECAY_TILED_TEST_SHORT_TRIANGLE
          for (int m = max(j + 1, i0); m < i - 1; ++m)
#else
          for (int m = max(j + 1, i0); m < i; ++m)
#endif
            acc[r] = fmaf(A[i * kChunk + m], own[m], acc[r]);
          own[i] = __fadd_rn(A[i * kChunk + j], acc[r]);
        }
      }
    }
  } else {
    // beta k exp(G) over K and beta v over G, each element as the staged
    // prepare forms it.
    for (int e = t - 64; e < kChunk * Dim / 4; e += kThreads - 64) {
      const int r = e >> 5, c = e & 31;
      const int at4 = quad<kDim>(r, c, key_swizzle(r));
      const float rb = beta[r];
      float v[4] = {0.f, 0.f, 0.f, 0.f};
      if (r < len) {
        const uint2 vp = *reinterpret_cast<const uint2*>(
            qkv + uint64_t(chunk.first_row + r) * qkv_stride + (2 * Heads + head) * Dim + 4 * c);
        v[0] = bf_lo(vp.x);
        v[1] = bf_hi(vp.x);
        v[2] = bf_lo(vp.y);
        v[3] = bf_hi(vp.y);
      }
      const float4 k = *reinterpret_cast<const float4*>(K + at4);
      const float4 g = *reinterpret_cast<const float4*>(G + at4);
      *reinterpret_cast<float4*>(K + at4) =
          make_float4(__fmul_rn(__fmul_rn(rb, k.x), expf(g.x)),
                      __fmul_rn(__fmul_rn(rb, k.y), expf(g.y)),
                      __fmul_rn(__fmul_rn(rb, k.z), expf(g.z)),
                      __fmul_rn(__fmul_rn(rb, k.w), expf(g.w)));
      *reinterpret_cast<float4*>(G + at4) = make_float4(
          __fmul_rn(rb, v[0]), __fmul_rn(rb, v[1]), __fmul_rn(rb, v[2]), __fmul_rn(rb, v[3]));
    }
  }
  __syncthreads();

  // W = T (beta k exp(G)) and U = T (beta v): thread (x quad, block pair)
  // takes rows 4p..4p+3 and 4(15 - p)..+3, each chain over j <= i in
  // ascending j.
  {
    const int c = t & 31, pair = t >> 5;
    const int i0[2] = {4 * pair, 4 * (15 - pair)};
    float w[2][4][4], u[2][4][4];
    #pragma unroll
    for (int s = 0; s < 2; ++s)
      #pragma unroll
      for (int r = 0; r < 4; ++r)
        #pragma unroll
        for (int e = 0; e < 4; ++e) w[s][r][e] = u[s][r][e] = 0.f;
    const int last = i0[1] + 3;
    #pragma unroll 1
    for (int j = 0; j <= last; ++j) {
      const int at4 = quad<kDim>(j, c, key_swizzle(j));
      const float4 kq = *reinterpret_cast<const float4*>(K + at4);
      const float4 gq = *reinterpret_cast<const float4*>(G + at4);
      #pragma unroll
      for (int s = 0; s < 2; ++s) {
        if (j > i0[s] + 3) continue;
        const float4 tq = *reinterpret_cast<const float4*>(TT + j * L::TTStride + i0[s]);
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
          if (j > i0[s] + r) continue;
          const float tij = lane4(tq, r);
          #pragma unroll
          for (int e = 0; e < 4; ++e) {
            w[s][r][e] = fmaf(tij, lane4(kq, e), w[s][r][e]);
            u[s][r][e] = fmaf(tij, lane4(gq, e), u[s][r][e]);
          }
        }
      }
    }
    #pragma unroll
    for (int s = 0; s < 2; ++s)
      #pragma unroll
      for (int r = 0; r < 4; ++r) {
        const int i = i0[s] + r;
        *reinterpret_cast<float4*>(work + Work::W + i * Dim + 4 * c) =
            make_float4(w[s][r][0], w[s][r][1], w[s][r][2], w[s][r][3]);
        *reinterpret_cast<float4*>(work + Work::U + i * Dim + 4 * c) =
            make_float4(u[s][r][0], u[s][r][1], u[s][r][2], u[s][r][3]);
      }
  }
}

// ------------------------------------------------------------------- scan --

// Dynamic shared memory of the tiled scan (floats): two operand tiles
// [64][128] and P [64][64] (chunk swizzle row & 7), U [64][16], EGL [128],
// the state slice S [16][128] and V'^T [16][64] (chunk swizzle v & 7).
struct ScanTiles {
  static constexpr int Tile = kChunk * kDim;
  static constexpr int Buffer0 = 0;
  static constexpr int Buffer1 = Tile;
  static constexpr int P = 2 * Tile;
  static constexpr int U = P + kChunk * kChunk;
  static constexpr int EGL = U + kChunk * kScanRows;
  static constexpr int S = EGL + kDim;
  static constexpr int VT = S + kScanRows * kDim;
  static constexpr int Floats = VT + kScanRows * kChunk;
  static_assert(Floats * 4 <= 101376 && U % 4 == 0 && S % 4 == 0 && VT % 4 == 0);
};

// Streams a [64][128] workspace tile into `tile` (chunk swizzle row & 7).
__device__ __forceinline__ void load_tile(float* tile, const float* source) {
  #pragma unroll
  for (int e = threadIdx.x; e < kChunk * kDim / 4; e += kThreads) {
    const int row = e >> 5, c = e & 31;
    cp_async16(tile + quad<kDim>(row, c, row & 7), source + 4 * e);
  }
}
__device__ __forceinline__ void load_triangle(float* P, const float* source) {
  #pragma unroll
  for (int e = threadIdx.x; e < kChunk * kChunk / 4; e += kThreads) {
    const int row = e >> 4, c = e & 15;
    cp_async16(P + quad<kChunk>(row, c, row & 7), source + 4 * e);
  }
}

// Grid, operands and state rules as channel_delta_scan; block kThreads,
// dynamic shared ScanTiles::Floats * 4 bytes.
template<int Heads, int Dim>
__device__ void channel_delta_scan_tiled(const float* workspace, float* state, uint16_t* output,
                                         const int32_t* cu_seqlens,
                                         const int32_t* start_positions,
                                         const int32_t* slot_ids, const int32_t* initial,
                                         const int32_t* anchor_ids,
                                         const uint32_t* live_sequences) {
  static_assert(Dim == kDim && kScanRows == 16 && kPrefillThreads == kThreads);
  using Work = ChunkWork<Dim>;
  using L = ScanTiles;
  constexpr uint64_t HeadElements = uint64_t(Dim) * Dim;
  extern __shared__ float4 channel_decay_tiled_smem4[];
  float* smem = reinterpret_cast<float*>(channel_decay_tiled_smem4);
  float* P = smem + L::P;
  float* U = smem + L::U;
  float* EGL = smem + L::EGL;
  float* S = smem + L::S;
  float* VT = smem + L::VT;
  begin();
  const uint32_t seq = blockIdx.z;
  if (seq >= *live_sequences) return;
  const int head = blockIdx.y;
  const int v0 = blockIdx.x * kScanRows;
  const int t = threadIdx.x;
  const int lane = t & 31, warp = t >> 5;
  const int32_t seq_start = cu_seqlens[seq];
  const int32_t rows = cu_seqlens[seq + 1] - seq_start;
  const int32_t phase = phase_of(start_positions, seq);
  const int32_t count = aligned_chunks(phase, rows);
  const int32_t first = first_aligned_chunk(cu_seqlens, start_positions, seq);
  const int32_t anchor = anchor_ids[seq];
  const int32_t anchor_at = anchor_row(phase, rows);
  const uint64_t slice = uint64_t(head) * HeadElements + uint64_t(v0) * Dim;
  float* slot = state + uint64_t(uint32_t(slot_ids[seq])) * Heads * HeadElements + slice;
  const bool carry = initial[seq] != 0;
  for (int e = t; e < kScanRows * Dim / 4; e += kThreads) {
    const int v = e >> 5, c = e & 31;
    float4 value = make_float4(0.f, 0.f, 0.f, 0.f);
    if (carry) value = *reinterpret_cast<const float4*>(slot + 4 * e);
    *reinterpret_cast<float4*>(S + quad<kDim>(v, c, v & 7)) = value;
  }
  auto stage_rows = [&](const float* work) {
    load_triangle(P, work + Work::P);
    {
      const int i = t >> 2, c = t & 3;
      cp_async16(U + i * kScanRows + 4 * c, work + Work::U + i * Dim + v0 + 4 * c);
    }
  };
  if (count > 0) {
    const float* work = chunk_work(workspace, first, head, Heads, Work::Floats);
    load_tile(smem + L::Buffer0, work + Work::W);
    stage_rows(work);
    cp_async_commit();
    cp_async_wait_all();
  }
  late_wait();
  // V' and O: thread (rows a = ip and b = 63 - ip, values vp and vp + 8).
  const int vp = t & 7, ip = t >> 3;
  const int ra = ip, rb = kChunk - 1 - ip;
  // The state update: thread (keys 2xp, 2xp + 1, values vq + 4k).
  const int xp = 16 * (warp & 3) + (lane & 15), vq = 2 * (warp >> 2) + (lane >> 4);
  #pragma unroll 1
  for (int c = 0; c < count; ++c) {
    const float* work = chunk_work(workspace, first + c, head, Heads, Work::Floats);
    float* X = smem + ((c & 1) ? L::Buffer1 : L::Buffer0);
    float* Y = smem + ((c & 1) ? L::Buffer0 : L::Buffer1);
    int32_t begin, end;
    chunk_rows(phase, rows, c, begin, end);

    // V' = U - W S^T.
    cp_async_wait_all();
    __syncthreads();
    load_tile(Y, work + Work::QD);
    cp_async_commit();
    {
      float acc[2][2] = {{0.f, 0.f}, {0.f, 0.f}};
      #pragma unroll 8
      for (int step = 0; step < Dim / 4; ++step) {
#ifdef CHOP_CHANNEL_DECAY_TILED_TEST_REVERSED_CHAIN
        const int x4 = Dim / 4 - 1 - step;
#else
        const int x4 = step;
#endif
        const float4 wa = *reinterpret_cast<const float4*>(X + quad<kDim>(ra, x4, ra & 7));
        const float4 wb = *reinterpret_cast<const float4*>(X + quad<kDim>(rb, x4, rb & 7));
        const float4 s0 = *reinterpret_cast<const float4*>(S + quad<kDim>(vp, x4, vp));
        const float4 s1 = *reinterpret_cast<const float4*>(S + quad<kDim>(vp + 8, x4, vp));
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
          acc[0][0] = fmaf(lane4(wa, e), lane4(s0, e), acc[0][0]);
          acc[0][1] = fmaf(lane4(wa, e), lane4(s1, e), acc[0][1]);
          acc[1][0] = fmaf(lane4(wb, e), lane4(s0, e), acc[1][0]);
          acc[1][1] = fmaf(lane4(wb, e), lane4(s1, e), acc[1][1]);
        }
      }
      VT[at<kChunk>(vp, ra, vp)] = __fsub_rn(U[ra * kScanRows + vp], acc[0][0]);
      VT[at<kChunk>(vp + 8, ra, vp)] = __fsub_rn(U[ra * kScanRows + vp + 8], acc[0][1]);
      VT[at<kChunk>(vp, rb, vp)] = __fsub_rn(U[rb * kScanRows + vp], acc[1][0]);
      VT[at<kChunk>(vp + 8, rb, vp)] = __fsub_rn(U[rb * kScanRows + vp + 8], acc[1][1]);
    }

    // O = QD S^T + P V'.
    cp_async_wait_all();
    __syncthreads();
    load_tile(X, work + Work::KD);
    if (t < Dim / 4) cp_async16(EGL + 4 * t, work + Work::EGL + 4 * t);
    cp_async_commit();
    {
      float inter[2][2] = {{0.f, 0.f}, {0.f, 0.f}};
      #pragma unroll 8
      for (int x4 = 0; x4 < Dim / 4; ++x4) {
        const float4 qa = *reinterpret_cast<const float4*>(Y + quad<kDim>(ra, x4, ra & 7));
        const float4 qb = *reinterpret_cast<const float4*>(Y + quad<kDim>(rb, x4, rb & 7));
        const float4 s0 = *reinterpret_cast<const float4*>(S + quad<kDim>(vp, x4, vp));
        const float4 s1 = *reinterpret_cast<const float4*>(S + quad<kDim>(vp + 8, x4, vp));
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
          inter[0][0] = fmaf(lane4(qa, e), lane4(s0, e), inter[0][0]);
          inter[0][1] = fmaf(lane4(qa, e), lane4(s1, e), inter[0][1]);
          inter[1][0] = fmaf(lane4(qb, e), lane4(s0, e), inter[1][0]);
          inter[1][1] = fmaf(lane4(qb, e), lane4(s1, e), inter[1][1]);
        }
      }
      float intra[2][2] = {{0.f, 0.f}, {0.f, 0.f}};
      #pragma unroll 4
      for (int j4 = 0; j4 < kChunk / 4; ++j4) {
        if (4 * j4 > rb) break;
        const float4 t0 = *reinterpret_cast<const float4*>(VT + quad<kChunk>(vp, j4, vp));
        const float4 t1 = *reinterpret_cast<const float4*>(VT + quad<kChunk>(vp + 8, j4, vp));
        const float4 pb = *reinterpret_cast<const float4*>(P + quad<kChunk>(rb, j4, rb & 7));
        #pragma unroll
        for (int e = 0; e < 4; ++e)
          if (4 * j4 + e <= rb) {
            intra[1][0] = fmaf(lane4(pb, e), lane4(t0, e), intra[1][0]);
            intra[1][1] = fmaf(lane4(pb, e), lane4(t1, e), intra[1][1]);
          }
        if (4 * j4 <= ra) {
          const float4 pa = *reinterpret_cast<const float4*>(P + quad<kChunk>(ra, j4, ra & 7));
          #pragma unroll
          for (int e = 0; e < 4; ++e)
            if (4 * j4 + e <= ra) {
              intra[0][0] = fmaf(lane4(pa, e), lane4(t0, e), intra[0][0]);
              intra[0][1] = fmaf(lane4(pa, e), lane4(t1, e), intra[0][1]);
            }
        }
      }
      const int live = end - begin;
      #pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int i = u == 0 ? ra : rb;
        if (i < live) {
          uint16_t* row = output + (uint64_t(seq_start + begin + i) * Heads + head) * Dim + v0;
          row[vp] = to_bf(__fadd_rn(inter[u][0], intra[u][0]));
          row[vp + 8] = to_bf(__fadd_rn(inter[u][1], intra[u][1]));
        }
      }
    }

    // S = S diag(EGL) + V'^T KD.
    cp_async_wait_all();
    __syncthreads();
    if (c + 1 < count) {
      const float* next = chunk_work(workspace, first + c + 1, head, Heads, Work::Floats);
      load_tile(Y, next + Work::W);
      stage_rows(next);
      cp_async_commit();
    }
    {
      const int x = 2 * xp, x4 = xp >> 1, half = 2 * (xp & 1);
      float acc[4][2];
      #pragma unroll
      for (int k = 0; k < 4; ++k) acc[k][0] = acc[k][1] = 0.f;
      #pragma unroll 2
      for (int i4 = 0; i4 < kChunk / 4; ++i4) {
        float4 vt[4];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
          const int v = vq + 4 * k;
          vt[k] = *reinterpret_cast<const float4*>(VT + quad<kChunk>(v, i4, v & 7));
        }
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int i = 4 * i4 + e;
          const float2 kd =
              *reinterpret_cast<const float2*>(X + quad<kDim>(i, x4, i & 7) + half);
          #pragma unroll
          for (int k = 0; k < 4; ++k) {
            acc[k][0] = fmaf(kd.x, lane4(vt[k], e), acc[k][0]);
            acc[k][1] = fmaf(kd.y, lane4(vt[k], e), acc[k][1]);
          }
        }
      }
      const float2 egl = *reinterpret_cast<const float2*>(EGL + x);
      const bool at_anchor = anchor >= 0 && anchor_at > 0 && c == count - 2;
      #pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int v = vq + 4 * k;
        float2* s = reinterpret_cast<float2*>(S + quad<kDim>(v, x4, v & 7) + half);
        const float2 old = *s;
        const float2 next = make_float2(fmaf(old.x, egl.x, acc[k][0]), fmaf(old.y, egl.y, acc[k][1]));
        *s = next;
        if (at_anchor) {
          float* entry = state + uint64_t(uint32_t(anchor)) * Heads * HeadElements + slice;
          *reinterpret_cast<float2*>(entry + uint64_t(v) * Dim + x) = next;
        }
      }
    }
  }
  __syncthreads();
  for (int e = t; e < kScanRows * Dim / 4; e += kThreads) {
    const int v = e >> 5, c = e & 31;
    *reinterpret_cast<float4*>(slot + 4 * e) =
        *reinterpret_cast<const float4*>(S + quad<kDim>(v, c, v & 7));
  }
}

} // namespace tiled
} // namespace chop::linear::channel_decay
