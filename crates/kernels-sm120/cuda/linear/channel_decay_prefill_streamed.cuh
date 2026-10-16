#pragma once

// A third schedule of the per-channel-decay delta rule's chunked prefill
// (channel_decay_prefill.cuh, channel_decay_prefill_tiled.cuh) with the same
// workspace and the same arithmetic: every stored value is the same chain of
// IEEE F32 operations in the same order, so either stage pairs with any
// schedule of the other and keeps every bit. Only the work assignment
// changes:
//
// `prepare_streamed` (one CTA per aligned chunk and head, 256 threads): the
//   tiled prepare with its latency-bound phases spread over every warp:
//   - every input load first (f of two channels over 16 rows, q, k and the
//     beta logit of 8 rows per thread), then g for all of them, stored
//     unsummed, and the q and k norms of 8 rows per warp at once;
//   - the running sums of g as a separate pass over shared memory;
//   - the decayed products as the tiled prepare (2 x 2 blocks, singles);
//   - the forward substitution on four threads per column j: thread s owns
//     the rows i = s + 4r; step m finishes T[m][j] = A[m][j] + sum_{m'=j+1}
//     ^{m-1} A[m][m'] T[m'][j] on the row's owner, passes it to the column's
//     other threads by a warp shuffle and adds A[i][m] T[m][j] to each row
//     i > m, so every chain still runs in ascending m from zero; steps run
//     in quads whose A loads come first;
//   - beta k exp(G) and beta v after it (v loaded before), then W and U in
//     4 x 4 tiles as the tiled prepare, the j range split where a row block
//     stops taking terms instead of predicating them off.
// `scan_streamed` (16 value rows of a head per CTA, 128 threads): per chunk
//   two key halves, then two row halves, one barrier each:
//   - keys 0..63, then 64..127: V' = U - W S^T and QD S^T on one load of the
//     state per key, rows ra = p and rb = 63 - p by values 4q..4q + 3 per
//     thread, each chain continuing in registers over both halves;
//   - rows 0..31, then 32..63 of the state update S = S diag(EGL) + V'^T KD
//     (keys 4x..4x + 3 by values 4w..4w + 3 per thread) and of P V';
//   - every phase loads the operands of its next one or two steps ahead
//     into registers;
//   - W and QD stream in as 64 x 64 key halves and KD as 32 x 128 row halves
//     through four 16 KB slots, each load issued one or two half phases
//     before its use.
//
// Shared tiles permute the 16-byte chunks of each row (XOR swizzle) so the
// tiled accesses are free of bank conflicts.
//
// Test-only mutants, each compiled into its own kernel-lab cubin (serving
// cubins define none of them), which kernel-lab must catch:
//   CHOP_CHANNEL_DECAY_STREAMED_TEST_LATE_WAIT: the dependency wait moves
//     after the first input reads of both stages;
//   CHOP_CHANNEL_DECAY_STREAMED_TEST_NO_TILE_WAIT: the scan reads its half
//     tiles without waiting for the copies;
//   CHOP_CHANNEL_DECAY_STREAMED_TEST_REVERSED_CHAIN: the scan's V' and output
//     chains run each key half in descending order;
//   CHOP_CHANNEL_DECAY_STREAMED_TEST_SHORT_TRIANGLE: the forward substitution
//     drops each element's last term;
//   CHOP_CHANNEL_DECAY_STREAMED_TEST_WRONG_OWNER: a column's finished element
//     comes from the next row's thread;
//   CHOP_CHANNEL_DECAY_STREAMED_TEST_NO_SINGLES: the 64 single pairs of rows
//     62 and 63 are never formed.

#include "channel_decay_prefill_tiled.cuh"

namespace chop::linear::channel_decay {
namespace streamed {

using tiled::at;
using tiled::decayed_products;
using tiled::key_swizzle;
using tiled::lane4;
using tiled::pair_block;
using tiled::quad;

constexpr int kThreads = 256;
constexpr int kScanThreads = 128;
constexpr int kDim = 128;
// Floats of a half tile: 64 x 64 (a key half) or 32 x 128 (a row half).
constexpr int kHalf = kChunk * kChunk;

#ifdef CHOP_CHANNEL_DECAY_STREAMED_TEST_REVERSED_CHAIN
constexpr bool kReversedChain = true;
#else
constexpr bool kReversedChain = false;
#endif
#ifdef CHOP_CHANNEL_DECAY_STREAMED_TEST_SHORT_TRIANGLE
constexpr bool kShortTriangle = true;
#else
constexpr bool kShortTriangle = false;
#endif
#ifdef CHOP_CHANNEL_DECAY_STREAMED_TEST_WRONG_OWNER
constexpr int kOwnerShift = 1;
#else
constexpr int kOwnerShift = 0;
#endif
#ifdef CHOP_CHANNEL_DECAY_STREAMED_TEST_NO_SINGLES
constexpr bool kSingles = false;
#else
constexpr bool kSingles = true;
#endif

__device__ __forceinline__ void cp_async16(float* shared, const float* global) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
               :: "r"(ptx::shared_address(shared)), "l"(global) : "memory");
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}
// Waits until at most N of the committed copy groups are in flight.
template<int N>
__device__ __forceinline__ void cp_async_wait() {
#ifndef CHOP_CHANNEL_DECAY_STREAMED_TEST_NO_TILE_WAIT
  asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory");
#endif
}

// The PDL entry: release dependents, then wait for the producer (the late-
// wait mutant waits in `late_wait` instead).
__device__ __forceinline__ void begin() {
#ifdef CHOP_CHANNEL_DECAY_STREAMED_TEST_LATE_WAIT
  ptx::griddep_launch_dependents();
#else
  chop::linear::begin();
#endif
}
__device__ __forceinline__ void late_wait() {
#ifdef CHOP_CHANNEL_DECAY_STREAMED_TEST_LATE_WAIT
  ptx::griddep_wait();
#endif
}

// ---------------------------------------------------------------- prepare --

// Dynamic shared memory of the streamed prepare (floats): K, G, beta and Q
// as the tiled prepare; once Q is dead its space holds A [64][68] (strictly
// lower) and TT [64][68], TT[j][i] = T[i][j].
struct PrepareTiles {
  static constexpr int Tile = kChunk * kDim;
  static constexpr int K = 0;
  static constexpr int G = Tile;
  static constexpr int Beta = 2 * Tile;
  static constexpr int Q = Beta + kChunk;
  static constexpr int Stride = kChunk + 4;
  static constexpr int A = Q;
  static constexpr int TT = A + kChunk * Stride;
  static constexpr int Floats = TT + kChunk * Stride;
  static_assert(Floats >= Q + Tile && Q % 4 == 0 && TT % 4 == 0 && Stride % 4 == 0);
  static_assert(Floats * 4 <= 101376);
};

// Grid, operands and workspace as channel_delta_prepare; block kThreads,
// dynamic shared PrepareTiles::Floats * 4 bytes.
template<int Heads, int Dim, int Gate, uint32_t BoundBits>
__device__ void channel_delta_prepare_streamed(const uint16_t* qkv, uint32_t qkv_stride,
                                               const uint16_t* f, uint32_t f_stride,
                                               const uint16_t* b, uint32_t b_stride,
                                               const float* a_log, const float* dt_bias,
                                               float* workspace, const int32_t* cu_seqlens,
                                               const int32_t* start_positions,
                                               const uint32_t* live_sequences) {
  static_assert(Dim == kDim && kPrefillThreads == kThreads);
  using Work = ChunkWork<Dim>;
  using L = PrepareTiles;
  extern __shared__ float4 channel_decay_streamed_smem4[];
  float* smem = reinterpret_cast<float*>(channel_decay_streamed_smem4);
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
  float* work = workspace + (uint64_t(uint32_t(blockIdx.y)) * Heads + head) * Work::Floats;

  // Every input load first: f of channels 2p, 2p + 1 over rows rg + 4n; q,
  // k and the beta logit of rows warp + 8n (lane l holds elements 4l..4l+3,
  // lane n < 8 the logit of row warp + 8n).
  const int p = t & 63, rg = t >> 6;
  uint32_t fraw[16];
  #pragma unroll
  for (int n = 0; n < 16; ++n) {
    const int r = rg + 4 * n;
    fraw[n] = r < len ? *reinterpret_cast<const uint32_t*>(
                            f + uint64_t(chunk.first_row + r) * f_stride + head * Dim + 2 * p)
                      : 0u;
  }
  uint2 qp[8], kp[8];
  #pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int r = warp + 8 * n;
    qp[n] = kp[n] = make_uint2(0u, 0u);
    if (r < len) {
      const uint16_t* base = qkv + uint64_t(chunk.first_row + r) * qkv_stride;
      qp[n] = *reinterpret_cast<const uint2*>(base + head * Dim + lane * 4);
      kp[n] = *reinterpret_cast<const uint2*>(base + (Heads + head) * Dim + lane * 4);
    }
  }
  const int brow = warp + 8 * lane;
  const uint16_t braw = lane < 8 && brow < len
                            ? b[uint64_t(chunk.first_row + brow) * b_stride + head]
                            : uint16_t(0);
  late_wait();
  const float a = expf(a_log[head]);
  const float bias0 = dt_bias[head * Dim + 2 * p];
  const float bias1 = dt_bias[head * Dim + 2 * p + 1];

  // g of each element, stored unsummed.
  #pragma unroll
  for (int n = 0; n < 16; ++n) {
    const int r = rg + 4 * n;
    float2 g = make_float2(0.f, 0.f);
    if (r < len) {
      g.x = log_decay<Gate, BoundBits>(bf(uint16_t(fraw[n] & 0xffffu)), bias0, a);
      g.y = log_decay<Gate, BoundBits>(bf(uint16_t(fraw[n] >> 16)), bias1, a);
    }
    *reinterpret_cast<float2*>(G + at<kDim>(r, 2 * p, key_swizzle(r))) = g;
  }
  // Normalized q (scaled) and k rows as l2_norm4 forms them, and beta; zero
  // past the chunk's end.
  {
    float q[8][4], k[8][4], sq[8], sk[8];
    #pragma unroll
    for (int n = 0; n < 8; ++n) {
      q[n][0] = bf_lo(qp[n].x);
      q[n][1] = bf_hi(qp[n].x);
      q[n][2] = bf_lo(qp[n].y);
      q[n][3] = bf_hi(qp[n].y);
      k[n][0] = bf_lo(kp[n].x);
      k[n][1] = bf_hi(kp[n].x);
      k[n][2] = bf_lo(kp[n].y);
      k[n][3] = bf_hi(kp[n].y);
      sq[n] = sk[n] = 0.f;
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        sq[n] = fmaf(q[n][j], q[n][j], sq[n]);
        sk[n] = fmaf(k[n][j], k[n][j], sk[n]);
      }
    }
    #pragma unroll
    for (int n = 0; n < 8; ++n) {
      sq[n] = warp_sum(sq[n]);
      sk[n] = warp_sum(sk[n]);
    }
    #pragma unroll
    for (int n = 0; n < 8; ++n) {
      const int r = warp + 8 * n;
      float qn[4] = {0.f, 0.f, 0.f, 0.f}, kn[4] = {0.f, 0.f, 0.f, 0.f};
      if (r < len) {
        const float q_norm = sqrtf(sq[n] + 1e-6f);
        const float k_norm = sqrtf(sk[n] + 1e-6f);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          qn[j] = (q[n][j] / q_norm) * query_scale<Dim>();
          kn[j] = k[n][j] / k_norm;
        }
      }
      const int at4 = quad<kDim>(r, lane, key_swizzle(r));
      *reinterpret_cast<float4*>(Q + at4) = make_float4(qn[0], qn[1], qn[2], qn[3]);
      *reinterpret_cast<float4*>(K + at4) = make_float4(kn[0], kn[1], kn[2], kn[3]);
    }
    if (lane < 8) beta[brow] = brow < len ? beta_of(braw) : 0.f;
  }
  __syncthreads();

  // Running sums of g per channel in row order, constant past the end; g
  // rounded to F32 before each add. The other warps zero P's upper blocks.
  if (t < Dim) {
    float g[kChunk];
    #pragma unroll
    for (int r = 0; r < kChunk; ++r) g[r] = G[at<kDim>(r, t, key_swizzle(r))];
    float running = 0.f;
    #pragma unroll
    for (int r = 0; r < kChunk; ++r) {
      if (r < len) running = __fadd_rn(running, g[r]);
      G[at<kDim>(r, t, key_swizzle(r))] = running;
    }
  } else {
    for (int e = t - Dim; e < kChunk * kChunk / 2; e += kThreads - Dim) {
      const int i = e >> 5, J = e & 31;
      if (J > (i >> 1))
        *reinterpret_cast<float2*>(work + Work::P + i * kChunk + 2 * J) = make_float2(0.f, 0.f);
    }
  }
  __syncthreads();

  // Decayed q and k rows and the chunk's total decay.
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
      float pv[2];
      #pragma unroll
      for (int v = 0; v < 2; ++v) {
        const int j = j0 + v;
        strict[round][u][v] = i < len && j < i ? __fmul_rn(-beta[i], acc_a[u][v]) : 0.f;
        pv[v] = i < len && j <= i ? acc_p[u][v] : 0.f;
      }
      *reinterpret_cast<float2*>(work + Work::P + i * kChunk + j0) = make_float2(pv[0], pv[1]);
    }
  }
  float strict_single = 0.f;
  const int single_i = 62 + (t >> 5), single_j = 32 + lane;
  if (kSingles && t < 64) {
    float acc_a[1][1], acc_p[1][1];
    acc_a[0][0] = acc_p[0][0] = 0.f;
    if (single_i < len) decayed_products<1, 1>(K, Q, G, single_i, single_j, acc_a, acc_p);
    const bool live = single_i < len;
    strict_single = live && single_j < single_i ? __fmul_rn(-beta[single_i], acc_a[0][0]) : 0.f;
    work[Work::P + single_i * kChunk + single_j] =
        live && single_j <= single_i ? acc_p[0][0] : 0.f;
  }
  __syncthreads();

  // A (strictly lower) and TT = I in Q's space; v for beta v loads now.
  for (int e = t; e < kChunk * L::Stride; e += kThreads) {
    const int j = e / L::Stride, i = e % L::Stride;
    TT[e] = i == j ? 1.f : 0.f;
  }
  #pragma unroll
  for (int round = 0; round < 2; ++round) {
    int I, J;
    pair_block(t + kThreads * round, I, J);
    #pragma unroll
    for (int u = 0; u < 2; ++u)
      *reinterpret_cast<float2*>(A + (2 * I + u) * L::Stride + 2 * J) =
          make_float2(strict[round][u][0], strict[round][u][1]);
  }
  if (t < 64) A[single_i * L::Stride + single_j] = strict_single;
  uint2 vp[8];
  #pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int r = (t >> 5) + 8 * n, c = lane;
    vp[n] = r < len ? *reinterpret_cast<const uint2*>(qkv + uint64_t(chunk.first_row + r) * qkv_stride +
                                                     (2 * Heads + head) * Dim + 4 * c)
                    : make_uint2(0u, 0u);
  }
  __syncthreads();

  // Forward substitution by columns, four threads per column j: thread s
  // owns rows i = s + 4r. Step m finishes T[m][j] on its owner and passes it
  // to the column's threads, which add A[i][m] T[m][j] to every row i > m:
  // each element is the chain sum_{m=j+1}^{i-1} A[i][m] T[m][j] in
  // ascending m added to A[i][j]. Steps run in quads, the quad's A loads
  // first; warp w's columns 8w.. start at quad 2w.
  {
    const int j = t >> 2, s = t & 3;
    const int group = lane & ~3;
    float acc[kChunk / 4];
    #pragma unroll
    for (int r = 0; r < kChunk / 4; ++r) acc[r] = 0.f;
    #pragma unroll
    for (int q = 0; q < kChunk / 4; ++q) {
      if (q < 2 * warp || 4 * q >= len) continue;
      float4 a_q[kChunk / 4];
      #pragma unroll
      for (int r = q; r < kChunk / 4; ++r)
        a_q[r] = *reinterpret_cast<const float4*>(A + (s + 4 * r) * L::Stride + 4 * q);
      float a_j[4];
      #pragma unroll
      for (int d = 0; d < 4; ++d) a_j[d] = A[(4 * q + d) * L::Stride + j];
      #pragma unroll
      for (int d = 0; d < 4; ++d) {
        const int m = 4 * q + d;
        const float done = __fadd_rn(a_j[d], acc[q]);
        const float tm = __shfl_sync(0xffffffffu, done, group | ((m + kOwnerShift) & 3));
        if (m > j && s == d && m < len) TT[j * L::Stride + m] = done;
        #pragma unroll
        for (int r = q; r < kChunk / 4; ++r) {
          const int i = s + 4 * r;
          if (m > j && i > m && !(kShortTriangle && i == m + 1))
            acc[r] = fmaf(lane4(a_q[r], d), tm, acc[r]);
        }
      }
    }
  }
  __syncthreads();

  // beta k exp(G) over K and beta v over G, each element as the staged
  // prepare forms it.
  #pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int r = (t >> 5) + 8 * n, c = lane;
    const int at4 = quad<kDim>(r, c, key_swizzle(r));
    const float rb = beta[r];
    const float v[4] = {bf_lo(vp[n].x), bf_hi(vp[n].x), bf_lo(vp[n].y), bf_hi(vp[n].y)};
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
  __syncthreads();

  // W = T (beta k exp(G)) and U = T (beta v): thread (x quad c, block pair
  // p) takes rows 4p..4p+3 (block 0) and 60-4p..63-4p (block 1), each chain
  // over j <= i in ascending j, the j range split where a block's rows stop
  // taking terms.
  {
    const int c = t & 31, pair = t >> 5;
    const int i0 = 4 * pair, i1 = 4 * (15 - pair);
    float w[2][4][4], u[2][4][4];
    #pragma unroll
    for (int b = 0; b < 2; ++b)
      #pragma unroll
      for (int r = 0; r < 4; ++r)
        #pragma unroll
        for (int e = 0; e < 4; ++e) w[b][r][e] = u[b][r][e] = 0.f;
    // Block b's rows r >= from take term j.
    auto term = [&](int j, int b, int base, int from) {
      const int at4 = quad<kDim>(j, c, key_swizzle(j));
      const float4 kq = *reinterpret_cast<const float4*>(K + at4);
      const float4 gq = *reinterpret_cast<const float4*>(G + at4);
      const float4 tq = *reinterpret_cast<const float4*>(TT + j * L::Stride + base);
      #pragma unroll
      for (int r = 0; r < 4; ++r) {
        if (r < from) continue;
        const float tij = lane4(tq, r);
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
          w[b][r][e] = fmaf(tij, lane4(kq, e), w[b][r][e]);
          u[b][r][e] = fmaf(tij, lane4(gq, e), u[b][r][e]);
        }
      }
    };
    #pragma unroll 2
    for (int j = 0; j < i0; ++j) {
      term(j, 0, i0, 0);
      term(j, 1, i1, 0);
    }
    #pragma unroll
    for (int d = 0; d < 4; ++d) {
      term(i0 + d, 0, i0, d);
      term(i0 + d, 1, i1, 0);
    }
    #pragma unroll 2
    for (int j = i0 + 4; j < i1; ++j) term(j, 1, i1, 0);
    #pragma unroll
    for (int d = 0; d < 4; ++d) term(i1 + d, 1, i1, d);
    #pragma unroll
    for (int b = 0; b < 2; ++b)
      #pragma unroll
      for (int r = 0; r < 4; ++r) {
        const int i = (b == 0 ? i0 : i1) + r;
        *reinterpret_cast<float4*>(work + Work::W + i * Dim + 4 * c) =
            make_float4(w[b][r][0], w[b][r][1], w[b][r][2], w[b][r][3]);
        *reinterpret_cast<float4*>(work + Work::U + i * Dim + 4 * c) =
            make_float4(u[b][r][0], u[b][r][1], u[b][r][2], u[b][r][3]);
      }
  }
}

// ------------------------------------------------------------------- scan --

// Chunk swizzle of value row v: rows 4q + k for q = 0..3 fall on distinct
// bank groups.
__device__ __forceinline__ int value_swizzle(int v) { return ((v >> 2) + 2 * (v & 3)) & 7; }

// Dynamic shared memory of the streamed scan (floats): four half-tile slots,
// P [64][64] (chunk swizzle row & 7, lower quads only), U [64][16], EGL
// [128], the state slice S [16][128] and V'^T [16][64] (chunk swizzle
// value_swizzle(v)).
struct ScanTiles {
  static constexpr int P = 4 * kHalf;
  static constexpr int U = P + kChunk * kChunk;
  static constexpr int EGL = U + kChunk * kScanRows;
  static constexpr int S = EGL + kDim;
  static constexpr int VT = S + kScanRows * kDim;
  static constexpr int Floats = VT + kScanRows * kChunk;
  static_assert(Floats * 4 <= 101376 && U % 4 == 0 && S % 4 == 0 && VT % 4 == 0);
};

// Key half h of a [64][128] workspace tile into a [64][64] slot (chunk
// swizzle row & 7).
__device__ __forceinline__ void load_key_half(float* slot, const float* source, int h) {
  #pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int e = threadIdx.x + kScanThreads * n;
    const int row = e >> 4, c = e & 15;
    cp_async16(slot + row * kChunk + ((c ^ (row & 7)) << 2), source + row * kDim + 64 * h + 4 * c);
  }
}
// Row half h of a [64][128] workspace tile into a [32][128] slot.
__device__ __forceinline__ void load_row_half(float* slot, const float* source, int h) {
  #pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int e = threadIdx.x + kScanThreads * n;
    cp_async16(slot + 4 * e, source + 32 * h * kDim + 4 * e);
  }
}
// P's quads on and below the diagonal, the U slice and EGL of a chunk.
template<int Dim>
__device__ __forceinline__ void load_rows(float* P, float* U, float* EGL, const float* work,
                                          int v0) {
  using Work = ChunkWork<Dim>;
  #pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int e = threadIdx.x + kScanThreads * n;
    const int row = e >> 4, c = e & 15;
    if (c <= (row >> 2))
      cp_async16(P + row * kChunk + ((c ^ (row & 7)) << 2), work + Work::P + row * kChunk + 4 * c);
  }
  #pragma unroll
  for (int n = 0; n < 2; ++n) {
    const int e = threadIdx.x + kScanThreads * n;
    const int row = e >> 2, c = e & 3;
    cp_async16(U + row * kScanRows + 4 * c, work + Work::U + row * Dim + v0 + 4 * c);
  }
  if (threadIdx.x < Dim / 4) cp_async16(EGL + 4 * threadIdx.x, work + Work::EGL + 4 * threadIdx.x);
}

// The operands of one key quad c of a key half: W and QD of rows ra and
// rb, the state of values 4vq..4vq + 3.
struct KeyQuad {
  float4 wa, wb, qa, qb, s[4];
};

__device__ __forceinline__ void load_key_quad(KeyQuad& o, const float* Wt, const float* Qt,
                                              const float* S, int h, int c, int ra, int rb,
                                              int vq) {
  o.wa = *reinterpret_cast<const float4*>(Wt + ra * kChunk + ((c ^ (ra & 7)) << 2));
  o.wb = *reinterpret_cast<const float4*>(Wt + rb * kChunk + ((c ^ (rb & 7)) << 2));
  o.qa = *reinterpret_cast<const float4*>(Qt + ra * kChunk + ((c ^ (ra & 7)) << 2));
  o.qb = *reinterpret_cast<const float4*>(Qt + rb * kChunk + ((c ^ (rb & 7)) << 2));
  #pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int v = 4 * vq + k;
    o.s[k] = *reinterpret_cast<const float4*>(S + v * kDim + (((16 * h + c) ^ value_swizzle(v)) << 2));
  }
}

// W S^T and QD S^T over key half h for rows ra and rb and values 4vq + k,
// in ascending key order (descending under the reversed-chain mutant); the
// operands of the next two quads load ahead.
__device__ __forceinline__ void key_half(const float* Wt, const float* Qt, const float* S, int h,
                                         int ra, int rb, int vq, float (&aw)[2][4],
                                         float (&aq)[2][4]) {
  auto quad_of = [](int step) { return kReversedChain ? 15 - step : step; };
  KeyQuad o[3];
  load_key_quad(o[0], Wt, Qt, S, h, quad_of(0), ra, rb, vq);
  load_key_quad(o[1], Wt, Qt, S, h, quad_of(1), ra, rb, vq);
  #pragma unroll
  for (int step = 0; step < 16; ++step) {
    if (step + 2 < 16) load_key_quad(o[(step + 2) % 3], Wt, Qt, S, h, quad_of(step + 2), ra, rb, vq);
    const KeyQuad& x = o[step % 3];
    #pragma unroll
    for (int e = 0; e < 4; ++e)
      #pragma unroll
      for (int k = 0; k < 4; ++k) {
        aw[0][k] = fmaf(lane4(x.wa, e), lane4(x.s[k], e), aw[0][k]);
        aw[1][k] = fmaf(lane4(x.wb, e), lane4(x.s[k], e), aw[1][k]);
        aq[0][k] = fmaf(lane4(x.qa, e), lane4(x.s[k], e), aq[0][k]);
        aq[1][k] = fmaf(lane4(x.qb, e), lane4(x.s[k], e), aq[1][k]);
      }
  }
}

// The operands of one row quad n of row half h: V' of the update's values
// 4vw + k and of the output's values 4vq + k, KD rows 4n..4n + 3 at keys
// 4kq.., and P of rows ra and rb.
struct RowQuad {
  float4 vu[4], vo[4], kd[4], pa, pb;
};

template<int H>
__device__ __forceinline__ void load_row_quad(RowQuad& o, const float* Kt, const float* VT,
                                              const float* P, int n, int kq, int vw, int vq,
                                              int ra, int rb) {
  const int q = 8 * H + n;
  #pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int v = 4 * vw + k;
    o.vu[k] = *reinterpret_cast<const float4*>(VT + v * kChunk + ((q ^ value_swizzle(v)) << 2));
    const int w = 4 * vq + k;
    o.vo[k] = *reinterpret_cast<const float4*>(VT + w * kChunk + ((q ^ value_swizzle(w)) << 2));
    o.kd[k] = *reinterpret_cast<const float4*>(Kt + (4 * n + k) * kDim + 4 * kq);
  }
  o.pb = *reinterpret_cast<const float4*>(P + rb * kChunk + ((q ^ (rb & 7)) << 2));
  if (H == 0) o.pa = *reinterpret_cast<const float4*>(P + ra * kChunk + ((q ^ (ra & 7)) << 2));
}

// Row half H: V'^T KD over rows 32H..32H + 31 (row half slot Kt) for keys
// 4kq + x and values 4vw + k, and P V' over the columns of the same rows
// for rows ra < 32 <= rb and values 4vq + k, both in ascending row order;
// the next quad's operands load ahead. P is read only on and below the
// diagonal.
template<int H>
__device__ __forceinline__ void row_half(const float* Kt, const float* VT, const float* P, int kq,
                                         int vw, int vq, int ra, int rb, float (&as)[4][4],
                                         float (&in)[2][4]) {
  RowQuad o[2];
  load_row_quad<H>(o[0], Kt, VT, P, 0, kq, vw, vq, ra, rb);
  #pragma unroll
  for (int n = 0; n < 8; ++n) {
    if (n + 1 < 8) load_row_quad<H>(o[(n + 1) & 1], Kt, VT, P, n + 1, kq, vw, vq, ra, rb);
    const RowQuad& x = o[n & 1];
    const int j0 = 4 * (8 * H + n);
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
      #pragma unroll
      for (int y = 0; y < 4; ++y)
        #pragma unroll
        for (int k = 0; k < 4; ++k) as[y][k] = fmaf(lane4(x.kd[e], y), lane4(x.vu[k], e), as[y][k]);
      if (j0 + e <= rb)
        #pragma unroll
        for (int k = 0; k < 4; ++k) in[1][k] = fmaf(lane4(x.pb, e), lane4(x.vo[k], e), in[1][k]);
      if (H == 0 && j0 + e <= ra)
        #pragma unroll
        for (int k = 0; k < 4; ++k) in[0][k] = fmaf(lane4(x.pa, e), lane4(x.vo[k], e), in[0][k]);
    }
  }
}

// Grid, operands and state rules as channel_delta_scan; block kScanThreads,
// dynamic shared ScanTiles::Floats * 4 bytes. Chunk c uses slots (0, 1) or
// (2, 3) for its low key halves of W and QD and the other pair for the high
// ones, then its low slots for the row halves of KD; copy groups commit in
// the order: high QD with P, U and EGL (at the low phase), KD (at the high
// phase), the next chunk's low halves (first row phase), its high W (second
// row phase).
template<int Heads, int Dim>
__device__ void channel_delta_scan_streamed(const float* workspace, float* state, uint16_t* output,
                                            const int32_t* cu_seqlens,
                                            const int32_t* start_positions,
                                            const int32_t* slot_ids, const int32_t* initial,
                                            const int32_t* anchor_ids,
                                            const uint32_t* live_sequences) {
  static_assert(Dim == kDim && kScanRows == 16);
  using Work = ChunkWork<Dim>;
  using L = ScanTiles;
  constexpr uint64_t HeadElements = uint64_t(Dim) * Dim;
  extern __shared__ float4 channel_decay_streamed_smem4[];
  float* smem = reinterpret_cast<float*>(channel_decay_streamed_smem4);
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
  for (int e = t; e < kScanRows * Dim / 4; e += kScanThreads) {
    const int v = e >> 5, c = e & 31;
    float4 value = make_float4(0.f, 0.f, 0.f, 0.f);
    if (carry) value = *reinterpret_cast<const float4*>(slot + 4 * e);
    *reinterpret_cast<float4*>(S + v * Dim + ((c ^ value_swizzle(v)) << 2)) = value;
  }
  if (count > 0) {
    const float* work = chunk_work(workspace, first, head, Heads, Work::Floats);
    load_key_half(smem, work + Work::W, 0);
    load_key_half(smem + kHalf, work + Work::QD, 0);
    cp_async_commit();
    load_key_half(smem + 2 * kHalf, work + Work::W, 1);
    cp_async_commit();
  }
  late_wait();
  // V', QD S^T and P V': rows ra = p and rb = 63 - p, values 4vq + k.
  const int vq = t & 3, rp = t >> 2;
  const int ra = rp, rb = kChunk - 1 - rp;
  // The state update: keys 4kq + x, values 4vw + k.
  const int kq = t & 31, vw = t >> 5;
  #pragma unroll 1
  for (int c = 0; c < count; ++c) {
    const float* work = chunk_work(workspace, first + c, head, Heads, Work::Floats);
    float* low_w = smem + ((c & 1) ? 2 : 0) * kHalf;
    float* low_q = low_w + kHalf;
    float* high_w = smem + ((c & 1) ? 0 : 2) * kHalf;
    float* high_q = high_w + kHalf;
    int32_t begin, end;
    chunk_rows(phase, rows, c, begin, end);
    float aw[2][4], aq[2][4];
    #pragma unroll
    for (int u = 0; u < 2; ++u)
      #pragma unroll
      for (int k = 0; k < 4; ++k) aw[u][k] = aq[u][k] = 0.f;

    // Keys 0..63 (the high W may still be in flight); QD's high half, P, U
    // and EGL stream.
    cp_async_wait<1>();
    __syncthreads();
    load_key_half(high_q, work + Work::QD, 1);
    load_rows<Dim>(P, U, EGL, work, v0);
    cp_async_commit();
    key_half(low_w, low_q, S, 0, ra, rb, vq, aw, aq);

    // Keys 64..127, then V' = U - W S^T; KD streams into the low slots.
    cp_async_wait<0>();
    __syncthreads();
    load_row_half(low_w, work + Work::KD, 0);
    load_row_half(low_q, work + Work::KD, 1);
    cp_async_commit();
    key_half(high_w, high_q, S, 1, ra, rb, vq, aw, aq);
    #pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int i = u == 0 ? ra : rb;
      const float4 uu = *reinterpret_cast<const float4*>(U + i * kScanRows + 4 * vq);
      #pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int v = 4 * vq + k;
        VT[v * kChunk + (((i >> 2) ^ value_swizzle(v)) << 2) + (i & 3)] =
            __fsub_rn(lane4(uu, k), aw[u][k]);
      }
    }

    // Rows 0..31 of the state update and P V' over j < 32; the next chunk's
    // low key halves stream into the high slots.
    cp_async_wait<0>();
    __syncthreads();
    const bool more = c + 1 < count;
    const float* next = chunk_work(workspace, first + c + 1, head, Heads, Work::Floats);
    if (more) {
      load_key_half(high_w, next + Work::W, 0);
      load_key_half(high_q, next + Work::QD, 0);
      cp_async_commit();
    }
    float as[4][4], in[2][4];
    #pragma unroll
    for (int x = 0; x < 4; ++x)
      #pragma unroll
      for (int k = 0; k < 4; ++k) as[x][k] = 0.f;
    #pragma unroll
    for (int u = 0; u < 2; ++u)
      #pragma unroll
      for (int k = 0; k < 4; ++k) in[u][k] = 0.f;
    row_half<0>(low_w, VT, P, kq, vw, vq, ra, rb, as, in);

    // Rows 32..63; the next chunk's high W streams into the first low slot.
    __syncthreads();
    if (more) {
      load_key_half(low_w, next + Work::W, 1);
      cp_async_commit();
    }
    row_half<1>(low_q, VT, P, kq, vw, vq, ra, rb, as, in);
    {
      const float4 egl = *reinterpret_cast<const float4*>(EGL + 4 * kq);
      const bool at_anchor = anchor >= 0 && anchor_at > 0 && c == count - 2;
      #pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int v = 4 * vw + k;
        float4* s = reinterpret_cast<float4*>(S + v * Dim + ((kq ^ value_swizzle(v)) << 2));
        const float4 old = *s;
        const float4 updated =
            make_float4(fmaf(old.x, egl.x, as[0][k]), fmaf(old.y, egl.y, as[1][k]),
                        fmaf(old.z, egl.z, as[2][k]), fmaf(old.w, egl.w, as[3][k]));
        *s = updated;
        if (at_anchor) {
          float* entry = state + uint64_t(uint32_t(anchor)) * Heads * HeadElements + slice;
          *reinterpret_cast<float4*>(entry + uint64_t(v) * Dim + 4 * kq) = updated;
        }
      }
    }
    const int live = end - begin;
    #pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int i = u == 0 ? ra : rb;
      if (i < live) {
        uint32_t o[2];
        #pragma unroll
        for (int k = 0; k < 2; ++k)
          o[k] = uint32_t(to_bf(__fadd_rn(aq[u][2 * k], in[u][2 * k]))) |
                 (uint32_t(to_bf(__fadd_rn(aq[u][2 * k + 1], in[u][2 * k + 1]))) << 16);
        uint16_t* row = output + (uint64_t(seq_start + begin + i) * Heads + head) * Dim + v0;
        *reinterpret_cast<uint2*>(row + 4 * vq) = make_uint2(o[0], o[1]);
      }
    }
  }
  __syncthreads();
  for (int e = t; e < kScanRows * Dim / 4; e += kScanThreads) {
    const int v = e >> 5, c = e & 31;
    *reinterpret_cast<float4*>(slot + 4 * e) =
        *reinterpret_cast<const float4*>(S + v * Dim + ((c ^ value_swizzle(v)) << 2));
  }
}

} // namespace streamed
} // namespace chop::linear::channel_decay
