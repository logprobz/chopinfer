#pragma once

// Chunked prefill of the per-channel-decay delta rule in two kernels over
// chunks that start at absolute multiples of kChunk (64). Decay enters only
// as differences within a chunk: with G_i the F32 running sum of g per key
// channel from the chunk's first row, the factors are exp(G_i - G_j) for
// j <= i, exp(G_i) and exp(G_last - G_j), every exponent at most zero (a
// running sum of nonpositive steps never increases under rounding). With a
// lower bound of -5 G reaches -320 per chunk, so a factor exp(-G) would
// overflow F32.
//
// `prepare` (one CTA per chunk and head) forms, all F32:
//   A[i][j] = -beta_i sum_x k_i[x] k_j[x] exp(G_i[x] - G_j[x])  (j < i)
//   P[i][j] = sum_x q_i[x] k_j[x] exp(G_i[x] - G_j[x])           (j <= i)
//   T = (I - A)^-1 by forward substitution (the reference's row loop)
//   W = T (beta k exp(G)), U = T (beta v),
//   QD = q exp(G), KD = k exp(G_last - G), EGL = exp(G_last)
// and stores W, U, QD, KD, P and EGL in the workspace; rows past a short
// chunk's end are zero.
// `scan` (one CTA per 16 value rows of a head of a sequence) walks the
// sequence's chunks with the state slice in shared memory:
//   V' = U - W S^T; O = QD S^T + P V'; S = S diag(EGL) + V'^T KD.
// Every reduction is one F32 FMA chain from zero in ascending index order
// over at most 128 terms (the 128-K rule holds with no split), so a row's
// bits depend only on its chunk's rows and the state the chunk starts from:
// any split of a prompt into calls that end on multiples of 64 keeps every
// bit.

#include "channel_decay_common.cuh"

namespace chop::linear::channel_decay {

// Value rows of one head per scan CTA.
constexpr int kScanRows = 16;
constexpr int kPrefillThreads = 256;
// Shared row strides: one bank apart per row.
constexpr int kKeyStride = 129;
constexpr int kTriangleStride = 65;

// F32 workspace layout of one (chunk, head).
template<int Dim>
struct ChunkWork {
  static constexpr int W = 0;
  static constexpr int U = W + kChunk * Dim;
  static constexpr int QD = U + kChunk * Dim;
  static constexpr int KD = QD + kChunk * Dim;
  static constexpr int P = KD + kChunk * Dim;
  static constexpr int EGL = P + kChunk * kChunk;
  static constexpr int Floats = EGL + Dim;
};

__device__ __forceinline__ const float* chunk_work(const float* workspace, int32_t chunk,
                                                   int head, int heads, int floats) {
  return workspace + (uint64_t(uint32_t(chunk)) * heads + head) * floats;
}

// Dynamic shared memory: K, Q (then T) and G at kKeyStride, and beta.
template<int Dim>
struct PrepareLayout {
  static constexpr int K = 0;
  static constexpr int Q = K + kChunk * kKeyStride;
  static constexpr int G = Q + kChunk * kKeyStride;
  static constexpr int Beta = G + kChunk * kKeyStride;
  static constexpr int Floats = Beta + kChunk;
  static_assert(Dim < kKeyStride && kChunk * kTriangleStride <= kChunk * kKeyStride);
};

// Grid: (Heads, prefill chunk capacity), block kPrefillThreads, dynamic
// shared PrepareLayout<Dim>::Floats * 4 bytes. Operands as for the
// recurrent entries; `start_positions[s]` is the absolute position of
// sequence s's first row.
template<int Heads, int Dim, int Gate, uint32_t BoundBits>
__device__ void channel_delta_prepare(const uint16_t* qkv, uint32_t qkv_stride,
                                      const uint16_t* f, uint32_t f_stride,
                                      const uint16_t* b, uint32_t b_stride,
                                      const float* a_log, const float* dt_bias,
                                      float* workspace, const int32_t* cu_seqlens,
                                      const int32_t* start_positions,
                                      const uint32_t* live_sequences) {
  static_assert(Dim == 128, "lanes hold four keys of a 128-wide head");
  using Work = ChunkWork<Dim>;
  using Layout = PrepareLayout<Dim>;
  extern __shared__ float channel_decay_smem[];
  float* K = channel_decay_smem + Layout::K;
  float* Q = channel_decay_smem + Layout::Q;
  float* T = channel_decay_smem + Layout::Q;  // after Q's last read
  float* G = channel_decay_smem + Layout::G;
  float* beta = channel_decay_smem + Layout::Beta;
  chop::linear::begin();
  AlignedChunk chunk;
  if (!locate_aligned_chunk(cu_seqlens, start_positions, *live_sequences, blockIdx.y, chunk))
    return;
  const int head = blockIdx.x;
  const int len = chunk.rows;
  const int t = threadIdx.x;
  const int lane = t & 31, warp = t >> 5;
  const float a = expf(a_log[head]);
  float* work = workspace + (uint64_t(uint32_t(blockIdx.y)) * Heads + head) * Work::Floats;

  // Normalized q (scaled) and k rows, and beta; zero past the chunk's end.
  for (int r = warp; r < kChunk; r += kPrefillThreads / 32) {
    float qn[4] = {0.f, 0.f, 0.f, 0.f}, kn[4] = {0.f, 0.f, 0.f, 0.f};
    float rb = 0.f;
    if (r < len) {
      const uint16_t* base = qkv + uint64_t(chunk.first_row + r) * qkv_stride;
      const uint2 qp = *reinterpret_cast<const uint2*>(base + head * Dim + lane * 4);
      const uint2 kp = *reinterpret_cast<const uint2*>(base + (Heads + head) * Dim + lane * 4);
      float q[4] = {bf_lo(qp.x), bf_hi(qp.x), bf_lo(qp.y), bf_hi(qp.y)};
      float k[4] = {bf_lo(kp.x), bf_hi(kp.x), bf_lo(kp.y), bf_hi(kp.y)};
      const float q_norm = l2_norm4(q);
      const float k_norm = l2_norm4(k);
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        qn[j] = (q[j] / q_norm) * query_scale<Dim>();
        kn[j] = k[j] / k_norm;
      }
      rb = beta_of(b[uint64_t(chunk.first_row + r) * b_stride + head]);
    }
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      Q[r * kKeyStride + lane * 4 + j] = qn[j];
      K[r * kKeyStride + lane * 4 + j] = kn[j];
    }
    if (lane == 0) beta[r] = rb;
  }
  // Running sums of g per channel, in row order; constant past the end. g
  // rounds to F32 before the add (no contraction into an FMA).
  if (t < Dim) {
    const float bias = dt_bias[head * Dim + t];
    float running = 0.f;
    for (int r = 0; r < kChunk; ++r) {
      if (r < len)
        running = __fadd_rn(
            running, log_decay<Gate, BoundBits>(
                         bf(f[uint64_t(chunk.first_row + r) * f_stride + head * Dim + t]), bias, a));
      G[r * kKeyStride + t] = running;
    }
  }
  __syncthreads();

  // Pairwise decayed products: warp w takes rows i = w + 8n, lanes the
  // columns j = lane and lane + 32. A stays in registers until Q is dead.
  float strict[kChunk / 8][2];
  #pragma unroll
  for (int n = 0; n < kChunk / 8; ++n) {
    const int i = warp + 8 * n;
    #pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int j = lane + 32 * half;
      float acc_a = 0.f, acc_p = 0.f;
      if (i < len && j <= i) {
        const float* ki = K + i * kKeyStride;
        const float* qi = Q + i * kKeyStride;
        const float* gi = G + i * kKeyStride;
        const float* kj = K + j * kKeyStride;
        const float* gj = G + j * kKeyStride;
        for (int x = 0; x < Dim; ++x) {
          const float term = kj[x] * expf(gi[x] - gj[x]);
          acc_a = fmaf(ki[x], term, acc_a);
          acc_p = fmaf(qi[x], term, acc_p);
        }
      }
      strict[n][half] = i < len && j < i ? -beta[i] * acc_a : 0.f;
      work[Work::P + i * kChunk + j] = i < len && j <= i ? acc_p : 0.f;
    }
  }
  // Decayed q and k rows and the chunk's total decay.
  {
    const int x = t & (Dim - 1);
    const float last = G[(len - 1) * kKeyStride + x];
    for (int r = t / Dim; r < kChunk; r += kPrefillThreads / Dim) {
      const float g = G[r * kKeyStride + x];
      work[Work::QD + r * Dim + x] = Q[r * kKeyStride + x] * expf(g);
      work[Work::KD + r * Dim + x] = K[r * kKeyStride + x] * expf(last - g);
    }
    if (t < Dim) work[Work::EGL + x] = expf(last);
  }
  __syncthreads();
  #pragma unroll
  for (int n = 0; n < kChunk / 8; ++n) {
    const int i = warp + 8 * n;
    #pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int j = lane + 32 * half;
      T[i * kTriangleStride + j] = j == i ? 1.f : strict[n][half];
    }
  }
  __syncthreads();
  // Forward substitution, one row at a time:
  // T[i][j] = A[i][j] + sum_{m=j+1}^{i-1} A[i][m] T[m][j].
  for (int i = 1; i < len; ++i) {
    float value = 0.f;
    if (t < i) {
      float acc = 0.f;
      for (int m = t + 1; m < i; ++m)
        acc = fmaf(T[i * kTriangleStride + m], T[m * kTriangleStride + t], acc);
      value = T[i * kTriangleStride + t] + acc;
    }
    __syncthreads();
    if (t < i) T[i * kTriangleStride + t] = value;
    __syncthreads();
  }
  // beta k exp(G) over K and beta v over G, each element by its own thread.
  {
    const int x = t & (Dim - 1);
    for (int r = t / Dim; r < kChunk; r += kPrefillThreads / Dim) {
      const float rb = beta[r];
      const float v =
          r < len ? bf(qkv[uint64_t(chunk.first_row + r) * qkv_stride + (2 * Heads + head) * Dim + x])
                  : 0.f;
      K[r * kKeyStride + x] = rb * K[r * kKeyStride + x] * expf(G[r * kKeyStride + x]);
      G[r * kKeyStride + x] = rb * v;
    }
  }
  __syncthreads();
  {
    const int x = t & (Dim - 1);
    for (int i = t / Dim; i < kChunk; i += kPrefillThreads / Dim) {
      float w = 0.f, u = 0.f;
      for (int j = 0; j <= i; ++j) {
        const float tij = T[i * kTriangleStride + j];
        w = fmaf(tij, K[j * kKeyStride + x], w);
        u = fmaf(tij, G[j * kKeyStride + x], u);
      }
      work[Work::W + i * Dim + x] = w;
      work[Work::U + i * Dim + x] = u;
    }
  }
}

// Dynamic shared memory of the scan: the state slice S, V', a staged
// 64 x Dim operand tile, P and EGL.
template<int Dim>
struct ScanLayout {
  static constexpr int S = 0;
  static constexpr int V = S + kScanRows * kKeyStride;
  static constexpr int Tile = V + kChunk * kScanRows;
  static constexpr int P = Tile + kChunk * kKeyStride;
  static constexpr int EGL = P + kChunk * kTriangleStride;
  static constexpr int Floats = EGL + Dim;
};

// Copies a [64, Dim] F32 workspace tile to shared memory at kKeyStride.
template<int Dim>
__device__ __forceinline__ void stage_tile(float* tile, const float* source) {
  const float4* from = reinterpret_cast<const float4*>(source);
  for (int e = threadIdx.x; e < kChunk * Dim / 4; e += kPrefillThreads) {
    const float4 value = from[e];
    const int row = (e * 4) / Dim, col = (e * 4) % Dim;
    float* to = tile + row * kKeyStride + col;
    to[0] = value.x;
    to[1] = value.y;
    to[2] = value.z;
    to[3] = value.w;
  }
}

// Grid: (Dim / kScanRows, Heads, sequence capacity), block
// kPrefillThreads, dynamic shared ScanLayout<Dim>::Floats * 4 bytes.
// Reads slot_ids[s] when initial[s] is nonzero (zero state otherwise) and
// writes the final state back to the same slot. When anchor_ids[s] >= 0 and
// the call ends off an absolute multiple of 64 with one inside it, the
// state at that multiple also goes to that entry.
template<int Heads, int Dim>
__device__ void channel_delta_scan(const float* workspace, float* state, uint16_t* output,
                                   const int32_t* cu_seqlens, const int32_t* start_positions,
                                   const int32_t* slot_ids, const int32_t* initial,
                                   const int32_t* anchor_ids,
                                   const uint32_t* live_sequences) {
  static_assert(Dim == 128 && kPrefillThreads == 2 * Dim && kPrefillThreads == 16 * kScanRows);
  using Work = ChunkWork<Dim>;
  using Layout = ScanLayout<Dim>;
  constexpr uint64_t HeadElements = uint64_t(Dim) * Dim;
  extern __shared__ float channel_decay_smem[];
  float* S = channel_decay_smem + Layout::S;
  float* V = channel_decay_smem + Layout::V;
  float* tile = channel_decay_smem + Layout::Tile;
  float* P = channel_decay_smem + Layout::P;
  float* EGL = channel_decay_smem + Layout::EGL;
  chop::linear::begin();
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
  // Thread (x, v) pairs for the state: x = t % Dim, v = t / Dim + 2r.
  const int sx = t & (Dim - 1), sv = t / Dim;
  const bool carry = initial[seq] != 0;
  for (int v = sv; v < kScanRows; v += kPrefillThreads / Dim)
    S[v * kKeyStride + sx] = carry ? slot[uint64_t(v) * Dim + sx] : 0.f;
  // Thread (i, v) pairs for V' and O: v = t % 16, i = t / 16 + 16r.
  const int ov = t & (kScanRows - 1), oi = t / kScanRows;
  for (int c = 0; c < count; ++c) {
    const float* work = chunk_work(workspace, first + c, head, Heads, Work::Floats);
    int32_t begin, end;
    chunk_rows(phase, rows, c, begin, end);
    stage_tile<Dim>(tile, work + Work::W);
    for (int e = t; e < kChunk * kChunk; e += kPrefillThreads)
      P[(e / kChunk) * kTriangleStride + e % kChunk] = work[Work::P + e];
    if (t < Dim) EGL[t] = work[Work::EGL + t];
    __syncthreads();
    // V' = U - W S^T.
    for (int i = oi; i < kChunk; i += kPrefillThreads / kScanRows) {
      float acc = 0.f;
      for (int x = 0; x < Dim; ++x)
        acc = fmaf(tile[i * kKeyStride + x], S[ov * kKeyStride + x], acc);
      V[i * kScanRows + ov] = work[Work::U + i * Dim + v0 + ov] - acc;
    }
    __syncthreads();
    stage_tile<Dim>(tile, work + Work::QD);
    __syncthreads();
    // O = QD S^T + P V'.
    for (int i = oi; i < kChunk; i += kPrefillThreads / kScanRows) {
      float inter = 0.f, intra = 0.f;
      for (int x = 0; x < Dim; ++x)
        inter = fmaf(tile[i * kKeyStride + x], S[ov * kKeyStride + x], inter);
      for (int j = 0; j <= i; ++j)
        intra = fmaf(P[i * kTriangleStride + j], V[j * kScanRows + ov], intra);
      if (i < end - begin)
        output[(uint64_t(seq_start + begin + i) * Heads + head) * Dim + v0 + ov] =
            to_bf(inter + intra);
    }
    __syncthreads();
    stage_tile<Dim>(tile, work + Work::KD);
    __syncthreads();
    // S = S diag(EGL) + V'^T KD.
    for (int v = sv; v < kScanRows; v += kPrefillThreads / Dim) {
      float acc = 0.f;
      for (int i = 0; i < kChunk; ++i)
        acc = fmaf(tile[i * kKeyStride + sx], V[i * kScanRows + v], acc);
      S[v * kKeyStride + sx] = fmaf(S[v * kKeyStride + sx], EGL[sx], acc);
    }
    __syncthreads();
    if (anchor >= 0 && anchor_at > 0 && c == count - 2) {
      float* entry = state + uint64_t(uint32_t(anchor)) * Heads * HeadElements + slice;
      for (int v = sv; v < kScanRows; v += kPrefillThreads / Dim)
        entry[uint64_t(v) * Dim + sx] = S[v * kKeyStride + sx];
    }
  }
  for (int v = sv; v < kScanRows; v += kPrefillThreads / Dim)
    slot[uint64_t(v) * Dim + sx] = S[v * kKeyStride + sx];
}

} // namespace chop::linear::channel_decay
