#pragma once

// Chunked gated delta rule prefill in two kernels.
//
// Within a chunk of L <= 64 rows of one sequence and value head, with raw
// BF16 rows Kr, Qr, Vr, inverse norms ik and iq (iq includes 1/sqrt(Dim)),
// beta, and G the cumulative sum of g inside the chunk:
//   A[i][j] = beta_i ik_i ik_j (Kr_i . Kr_j) exp(G_i - G_j), j < i
//   T = (I + A)^-1,  T' = T diag(beta)
//   P[i][j] = iq_i ik_j (Qr_i . Kr_j) exp(G_i - G_j), j <= i
//   c1_i = exp(G_i) ik_i,  c2_i = exp(G_i) iq_i,  c3_i = exp(G_L - G_i) ik_i
// and for the state S0 [value, key] at the chunk start:
//   U = T' (Vr - diag(c1) Kr S0^T)            (the delta rows)
//   O = diag(c2) Qr S0^T + P U                (the outputs)
//   S = exp(G_L) S0 + U^T diag(c3) Kr         (the state after the chunk)
// which is the token recurrence rewritten exactly.
//
// `gated_delta_prepare` computes T', P and the c vectors for every chunk and
// value head in parallel into a workspace. `gated_delta_scan` then walks the
// chunks of each sequence in order; each CTA owns kScanRows value rows of one
// head and keeps them in registers. Everything is F32 FMA on BF16-exact
// inputs, and G is accumulated in F64 so that exp(G_i - G_j) stays accurate
// for fast-decaying heads.

#include "common.cuh"
#include "../common/ptx_control.cuh"
#include "../common/ptx_matrix.cuh"

namespace chop::linear {

constexpr int kScanRows = 16;
constexpr int kPrefillThreads = 256;
// T' and P are lower triangular and stored packed: row i holds columns
// 0 .. 4 (i / 4) + 3 (zeros past i), so rows stay 16-byte aligned.
__host__ __device__ constexpr int tri_row(int i) {
  return 4 * (i / 4 + 1) * (2 * (i / 4) + i % 4);
}
__host__ __device__ constexpr int tri_width(int i) { return 4 * (i / 4 + 1); }
constexpr int kTri = tri_row(kChunk);
// Workspace floats per (chunk, value head): packed T', packed P, then c1,
// c2, c3 and exp(G_L) in the first element of a fourth 64-float vector.
constexpr int kWorkT = 0;
constexpr int kWorkP = kTri;
constexpr int kWorkC = 2 * kTri;
constexpr int kWorkFloats = 2 * kTri + 4 * kChunk;
static_assert(kTri == 2176 && kWorkFloats == 4608);

// Workspace of one chunk, in bytes: the per-value-head T', P and scales, then
// per key head the K and Q rows in the scan's padded shared layout, then per
// value head the value columns grouped by scan slice ([Dim / kScanRows][64]
// [kScanRows]). Rows past the chunk are zero, so the scan copies whole blocks.
template<int KeyHeads, int ValueHeads, int Dim>
struct ChunkWork {
  static constexpr int kKqBytes = 2 * kChunk * (Dim + 8) * 2;
  static constexpr int kVBytes = kChunk * Dim * 2;
  static constexpr uint64_t kMatrices = 0;
  static constexpr uint64_t kKq = kMatrices + uint64_t(ValueHeads) * kWorkFloats * 4;
  static constexpr uint64_t kV = kKq + uint64_t(KeyHeads) * kKqBytes;
  static constexpr uint64_t kBytes = kV + uint64_t(ValueHeads) * kVBytes;
  __device__ static unsigned char* chunk(float* workspace, int32_t global) {
    return reinterpret_cast<unsigned char*>(workspace) + uint64_t(global) * kBytes;
  }
  __device__ static const unsigned char* chunk(const float* workspace, int32_t global) {
    return reinterpret_cast<const unsigned char*>(workspace) + uint64_t(global) * kBytes;
  }
};

// Shared row strides: BF16 rows padded by 16 bytes, F32 rows by 16 bytes.
template<int Dim>
struct PrefillLayout {
  static constexpr int kRow = Dim + 8;        // BF16 elements per K/Q row
  static constexpr int kMat = kChunk + 4;     // F32 elements per 64-wide row
  static constexpr int kKqBytes = 2 * kChunk * kRow * 2;
  static constexpr int kMatBytes = kChunk * kMat * 4;
};

template<int Dim>
__device__ __forceinline__ void load_chunk_kq(uint16_t* kq, const uint16_t* qkv,
                                              uint32_t qkv_stride, int key_offset,
                                              int query_offset, int first_row,
                                              int rows) {
  using L = PrefillLayout<Dim>;
  constexpr int Vectors = Dim / 8;
  for (int index = threadIdx.x; index < 2 * kChunk * Vectors; index += blockDim.x) {
    const int r = index / Vectors;
    const int col = (index % Vectors) * 8;
    const int local = r & (kChunk - 1);
    uint4 value = make_uint4(0, 0, 0, 0);
    if (local < rows)
      value = *reinterpret_cast<const uint4*>(
          qkv + uint64_t(first_row + local) * qkv_stride +
          (r < kChunk ? key_offset : query_offset) + col);
    *reinterpret_cast<uint4*>(kq + r * L::kRow + col) = value;
  }
}

__device__ __forceinline__ void unpack8(const uint4& p, float (&x)[8]) {
  x[0] = bf_lo(p.x); x[1] = bf_hi(p.x);
  x[2] = bf_lo(p.y); x[3] = bf_hi(p.y);
  x[4] = bf_lo(p.z); x[5] = bf_hi(p.z);
  x[6] = bf_lo(p.w); x[7] = bf_hi(p.w);
}

// Grid: (KeyHeads, prefill chunk capacity), block kPrefillThreads. One CTA
// forms the Gram matrices of one chunk and key head once, then prepares the
// value heads that share the key head.
template<int KeyHeads, int ValueHeads, int Dim>
__device__ void gated_delta_prepare(const uint16_t* qkv, uint32_t qkv_stride,
                                    const uint16_t* a, const uint16_t* b,
                                    uint32_t gate_stride, const float* a_log,
                                    const float* dt_bias, float* workspace,
                                    const int32_t* cu_seqlens,
                                    const uint32_t* live_sequences) {
  using L = PrefillLayout<Dim>;
  constexpr int Group = ValueHeads / KeyHeads;
  static_assert(Group <= 3 && Dim % 8 == 0);
  extern __shared__ __align__(16) unsigned char smem[];
  uint16_t* kq = reinterpret_cast<uint16_t*>(smem);
  float* kk = reinterpret_cast<float*>(smem + L::kKqBytes);
  float* qk = kk + kChunk * L::kMat;
  float* inv = qk + kChunk * L::kMat;              // [2][64]: ik, iq
  float* beta = inv + 2 * kChunk;                  // [Group][64]
  double* cumulative = reinterpret_cast<double*>(beta + Group * kChunk);  // [Group][64]
  double* totals = cumulative + Group * kChunk;    // [Group]
  float* g_hi = reinterpret_cast<float*>(totals + Group);  // [Group][64]
  float* g_lo = g_hi + Group * kChunk;                      // [Group][64]
  begin();
  Chunk chunk;
  if (!locate_chunk(cu_seqlens, *live_sequences, blockIdx.y, chunk)) return;
  const int key_head = blockIdx.x;
  const int rows = chunk.rows;
  using W = ChunkWork<KeyHeads, ValueHeads, Dim>;
  unsigned char* work_chunk = W::chunk(workspace, chunk.global);
  // Load the chunk's K and Q rows and the value columns of this key head's
  // value heads in one batch, then place K and Q in shared memory and stage
  // them, as laid out there, and the values grouped by scan slice in the
  // workspace for the scan's bulk copies.
  {
    constexpr int RowVectors = Dim / 8;
    constexpr int KqLoads = 2 * kChunk * RowVectors / kPrefillThreads;
    constexpr int VLoads = Group * kChunk * RowVectors / kPrefillThreads;
    static_assert(KqLoads * kPrefillThreads == 2 * kChunk * RowVectors &&
                  VLoads * kPrefillThreads == Group * kChunk * RowVectors);
    uint4 kq_regs[KqLoads], v_regs[VLoads];
    #pragma unroll
    for (int n = 0; n < KqLoads; ++n) {
      const int index = threadIdx.x + n * kPrefillThreads;
      const int r = index / RowVectors, col = index % RowVectors * 8;
      const int local = r & (kChunk - 1);
      kq_regs[n] = make_uint4(0, 0, 0, 0);
      if (local < rows)
        kq_regs[n] = *reinterpret_cast<const uint4*>(
            qkv + uint64_t(chunk.first_row + local) * qkv_stride +
            (r < kChunk ? KeyHeads + key_head : key_head) * Dim + col);
    }
    #pragma unroll
    for (int n = 0; n < VLoads; ++n) {
      const int index = threadIdx.x + n * kPrefillThreads;
      const int head = key_head * Group + index / (kChunk * RowVectors);
      const int row = index / RowVectors % kChunk, col = index % RowVectors * 8;
      v_regs[n] = make_uint4(0, 0, 0, 0);
      if (row < rows)
        v_regs[n] = *reinterpret_cast<const uint4*>(
            qkv + uint64_t(chunk.first_row + row) * qkv_stride + (2 * KeyHeads + head) * Dim + col);
    }
    uint4* kq_out = reinterpret_cast<uint4*>(work_chunk + W::kKq + uint64_t(key_head) * W::kKqBytes);
    #pragma unroll
    for (int n = 0; n < KqLoads; ++n) {
      const int index = threadIdx.x + n * kPrefillThreads;
      const int offset = index / RowVectors * L::kRow + index % RowVectors * 8;
      *reinterpret_cast<uint4*>(kq + offset) = kq_regs[n];
      kq_out[offset / 8] = kq_regs[n];
    }
    #pragma unroll
    for (int n = 0; n < VLoads; ++n) {
      const int index = threadIdx.x + n * kPrefillThreads;
      const int head = key_head * Group + index / (kChunk * RowVectors);
      const int row = index / RowVectors % kChunk, col = index % RowVectors * 8;
      reinterpret_cast<uint4*>(work_chunk + W::kV + uint64_t(head) * W::kVBytes)
          [((col / kScanRows) * kChunk + row) * (kScanRows / 8) + col % kScanRows / 8] = v_regs[n];
    }
  }
  __syncthreads();
  if (threadIdx.x < 2 * kChunk) {
    const uint16_t* row = kq + threadIdx.x * L::kRow;
    float squares = 0.f;
    for (int col = 0; col < Dim; col += 8) {
      float x[8];
      unpack8(*reinterpret_cast<const uint4*>(row + col), x);
      #pragma unroll
      for (int j = 0; j < 8; ++j) squares = fmaf(x[j], x[j], squares);
    }
    const float scale = threadIdx.x < kChunk ? 1.f : rsqrtf(float(Dim));
    inv[threadIdx.x] = rsqrtf(squares + 1e-6f) * scale;
  }
  // Gram matrices on the tensor cores (exact BF16 products, F32 sums): warp
  // w forms rows 16 (w % 4) .. + 16 of K K^T (w < 4) or Q K^T (w >= 4).
  {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t4 = lane & 3;
    const int row0 = (warp >> 2) * kChunk + 16 * (warp & 3);
    ptx::D4 acc[8] = {};
    #pragma unroll
    for (int k = 0; k < Dim; k += 16) {
      ptx::A4 a;
      ptx::ldmatrix_b16<4>(a.v, kq + (row0 + (lane & 15)) * L::kRow + k + (lane >> 4) * 8);
      #pragma unroll
      for (int n = 0; n < 8; n += 2) {
        uint32_t b[4];
        ptx::ldmatrix_b16<4>(
            b, kq + (8 * n + (lane & 7) + ((lane >> 4) << 3)) * L::kRow + k + ((lane >> 3) & 1) * 8);
        ptx::mma_bf16(acc[n], a, ptx::B2{{b[0], b[1]}});
        ptx::mma_bf16(acc[n + 1], a, ptx::B2{{b[2], b[3]}});
      }
    }
    float* gram = warp < 4 ? kk : qk;
    #pragma unroll
    for (int n = 0; n < 8; ++n)
      #pragma unroll
      for (int e = 0; e < 4; ++e)
        gram[(16 * (warp & 3) + g + 8 * (e >> 1)) * L::kMat + 8 * n + 2 * t4 + (e & 1)] =
            acc[n].v[e];
  }
  // Decay and beta per (value head, row); F64 inclusive scan of g.
  if (threadIdx.x < Group * kChunk) {
    const int r = threadIdx.x / kChunk, i = threadIdx.x % kChunk;
    const int head = key_head * Group + r;
    double g = 0.0;
    float bt = 0.f;
    if (i < rows) {
      const uint64_t row = uint64_t(chunk.first_row + i);
      g = double(-expf(a_log[head]) * softplus(bf(a[row * gate_stride + head]) + dt_bias[head]));
      bt = sigmoid(bf(b[row * gate_stride + head]));
    }
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      const double other = __shfl_up_sync(0xffffffffu, g, offset);
      if ((threadIdx.x & 31) >= offset) g += other;
    }
    if (i == 31) totals[r] = g;
    beta[r * kChunk + i] = bt;
    cumulative[r * kChunk + i] = g;
  }
  __syncthreads();
  // G as an F32 hi + lo pair: exp(G_i - G_j) then needs only F32
  // subtractions, each within 2^-24 of the difference.
  if (threadIdx.x < Group * kChunk) {
    const int r = threadIdx.x / kChunk;
    double total = cumulative[threadIdx.x];
    if (threadIdx.x % kChunk >= 32) {
      total += totals[r];
      cumulative[threadIdx.x] = total;
    }
    const float hi = float(total);
    g_hi[threadIdx.x] = hi;
    g_lo[threadIdx.x] = float(total - double(hi));
  }
  __syncthreads();
  // A for every value head. The first two reuse the K/Q rows' space, the
  // last overwrites the key Gram matrix after the others have read it.
  float* const amat = reinterpret_cast<float*>(smem);
  static_assert(2 * L::kMatBytes <= L::kKqBytes);
  const float* ik = inv;
  const float* iq = inv + kChunk;
  auto matrices = [&](int r) {
    return reinterpret_cast<float*>(work_chunk + W::kMatrices) +
           (key_head * Group + r) * kWorkFloats;
  };
  auto decay_of = [&](int r, int i, int j) {
    const float* hi = g_hi + r * kChunk;
    const float* lo = g_lo + r * kChunk;
    return expf((hi[i] - hi[j]) + (lo[i] - lo[j]));
  };
  for (int index = threadIdx.x; index < kChunk * kChunk; index += blockDim.x) {
    const int i = index / kChunk, j = index % kChunk;
    const float gram = kk[i * L::kMat + j];
    const float qdot = qk[i * L::kMat + j];
    #pragma unroll
    for (int r = 0; r < Group; ++r) {
      // Above the diagonal only P's padding needs zeros; the solve reads
      // A below the diagonal only.
      if (j > i) {
        if (j < tri_width(i)) matrices(r)[kWorkP + tri_row(i) + j] = 0.f;
        continue;
      }
      const float decay = decay_of(r, i, j);
      if (r < 2)
        amat[r * kChunk * L::kMat + i * L::kMat + j] =
            j < i ? beta[r * kChunk + i] * ik[i] * ik[j] * gram * decay : 0.f;
      matrices(r)[kWorkP + tri_row(i) + j] = i < rows ? iq[i] * ik[j] * qdot * decay : 0.f;
    }
  }
  if constexpr (Group == 3) {
    __syncthreads();
    for (int index = threadIdx.x; index < kChunk * kChunk; index += blockDim.x) {
      const int i = index / kChunk, j = index % kChunk;
      if (j >= i) continue;
      float* entry = kk + i * L::kMat + j;
      *entry = beta[2 * kChunk + i] * ik[i] * ik[j] * *entry * decay_of(2, i, j);
    }
  }
  // Per-row scale vectors.
  if (threadIdx.x < Group * kChunk) {
    const int r = threadIdx.x / kChunk, i = threadIdx.x % kChunk;
    const double* gr = cumulative + r * kChunk;
    const double last = gr[rows - 1];
    float* c = matrices(r) + kWorkC;
    const bool live = i < rows;
    const float gamma = expf(float(gr[i]));
    c[i] = live ? gamma * ik[i] : 0.f;
    c[kChunk + i] = live ? gamma * iq[i] : 0.f;
    c[2 * kChunk + i] = live ? expf(float(last - gr[i])) * ik[i] : 0.f;
    c[3 * kChunk + i] = i == 0 ? expf(float(last)) : 0.f;
  }
  __syncthreads();
  // Forward substitution, one column of T per thread: t_i = e_c - sum_{j<i}
  // A_ij t_j. Rows past the chunk have A = 0 and beta = 0.
  if (threadIdx.x < Group * kChunk) {
    const int r = threadIdx.x / kChunk, column = threadIdx.x % kChunk;
    const float* am = r < 2 ? amat + r * kChunk * L::kMat : kk;
    // Four partial sums keep the dependent chain short: t_i waits on t_{i-1}
    // through one FMA and the final adds, not through i FMAs.
    float t[kChunk];
    #pragma unroll
    for (int i = 0; i < kChunk; ++i) {
      float part[4] = {i == column ? 1.f : 0.f, 0.f, 0.f, 0.f};
      #pragma unroll
      for (int j = 0; j < i; ++j) part[(i - 1 - j) & 3] = fmaf(-am[i * L::kMat + j], t[j], part[(i - 1 - j) & 3]);
      t[i] = (part[0] + part[1]) + (part[2] + part[3]);
    }
    const float bc = beta[r * kChunk + column];
    float* tw = matrices(r) + kWorkT;
    #pragma unroll
    for (int i = 0; i < kChunk; ++i)
      if (column < tri_width(i)) tw[tri_row(i) + column] = t[i] * bc;
  }
}

// Splits x into BF16 hi + lo with |x - hi - lo| <= 2^-16 |x|: hi is x
// rounded, lo is the exact F32 remainder rounded.
__device__ __forceinline__ void split_bf16(float x, uint16_t& hi, uint16_t& lo) {
  hi = to_bf(x);
  lo = to_bf(x - bf(hi));
}

// Tensor-core layout of the scan (mma.m16n8k16, BF16 in, F32 accumulate):
//   step 1  [Kr; Qr] (128 x Dim, exact BF16) x S^T (Dim x 16), with S split
//           into BF16 hi + lo, one 16-row m-tile per warp;
//   step 4  S (16 x Dim) += (c3 U)^T (16 x 64, split hi + lo) x Kr (64 x Dim),
//           two 8-key n-tiles per warp, the state held in the accumulators.
// The split leaves an error of at most 2^-16 of each S or U element, well
// inside the oracle's bound for a key_dim reduction per row. Steps 2 and 3
// (64 x 64 triangular products with F32 T' and P) run on F32 FMA.
//
// Operands arrive by TMA bulk copies, which keep the load/store pipe free for
// the math. Each buffer is refilled for the next chunk as soon as its last
// reader is done: Q and V after step 1, T' after step 2, P after steps 3 and
// 4; K and the scales are double buffered because steps 1 and 4 both use them.
template<int Dim>
struct ScanLayout {
  static constexpr int kRow = Dim + 8;              // BF16 K/Q and S split rows
  static constexpr int kURow = kChunk + 8;          // BF16 U split rows
  static constexpr int kZRow = kScanRows + 1;       // F32 Z/X/U rows
  static constexpr int kRows = kChunk * kRow * 2;   // one 64-row K or Q buffer
  static constexpr int kK = 0;
  static constexpr int kQ = kK + 2 * kRows;
  static constexpr int kV = kQ + kRows;
  static constexpr int kT = kV + kChunk * kScanRows * 2;
  static constexpr int kP = kT + kTri * 4;
  static constexpr int kC = kP + kTri * 4;
  static constexpr int kSplit = kC + 2 * 4 * kChunk * 4;
  static constexpr int kUSplit = kSplit + 2 * kScanRows * kRow * 2;
  static constexpr int kZx = kUSplit + 2 * kScanRows * kURow * 2;
  static constexpr int kU = kZx + 2 * kChunk * kZRow * 4;
  static constexpr int kBar = kU + kChunk * kZRow * 4;
  static constexpr int kBytes = kBar + 5 * 8;
};

// acc[m] += sum_j M[i][j] x[j] for rows i = orow + 16 m of a packed lower
// triangular M and one column x of a [64][ZRow] matrix. Fully unrolled, so
// the row predicates are the only runtime control.
template<int ZRow>
__device__ __forceinline__ void triangular_times(float (&acc)[4], const float* matrix,
                                                 const float* x, int orow, int column) {
  #pragma unroll
  for (int j = 0; j < kChunk; j += 4) {
    const float v[4] = {x[j * ZRow + column], x[(j + 1) * ZRow + column],
                        x[(j + 2) * ZRow + column], x[(j + 3) * ZRow + column]};
    #pragma unroll
    for (int m = 0; m < 4; ++m) {
      const int i = orow + 16 * m;
      if (16 * m + 15 < j || i < j) continue;
      const float4 t = *reinterpret_cast<const float4*>(matrix + tri_row(i) + j);
      acc[m] = fmaf(t.x, v[0], acc[m]);
      acc[m] = fmaf(t.y, v[1], acc[m]);
      acc[m] = fmaf(t.z, v[2], acc[m]);
      acc[m] = fmaf(t.w, v[3], acc[m]);
    }
  }
}

// Grid: (Dim / kScanRows, ValueHeads, sequence capacity), block
// kPrefillThreads. Walks the chunks of one sequence for kScanRows value rows
// of one head. `initial[seq]` zero starts from a zero state instead of the
// slot's; the final state is written back to the slot. `anchor_ids[seq]` >= 0
// names a pool entry that also receives the state after the sequence's last
// whole chunk when a partial chunk follows it: the state a sequence ending
// there would write, from the same accumulators.
template<int KeyHeads, int ValueHeads, int Dim>
__device__ void gated_delta_scan(const uint16_t* qkv, uint32_t qkv_stride,
                                 const float* workspace, float* state,
                                 uint16_t* output, const int32_t* cu_seqlens,
                                 const int32_t* slot_ids, const int32_t* initial,
                                 const int32_t* anchor_ids,
                                 const uint32_t* live_sequences) {
  using L = ScanLayout<Dim>;
  constexpr int Group = ValueHeads / KeyHeads;
  constexpr int ZRow = L::kZRow;
  static_assert(Dim == 128 && kPrefillThreads == 256 && kScanRows == 16);
  static_assert(L::kBytes <= 101376);
  extern __shared__ __align__(16) unsigned char smem[];
  uint16_t* kbuf = reinterpret_cast<uint16_t*>(smem + L::kK);         // [2][64][kRow]
  uint16_t* qbuf = reinterpret_cast<uint16_t*>(smem + L::kQ);         // [64][kRow]
  uint16_t* vbuf = reinterpret_cast<uint16_t*>(smem + L::kV);         // [64][16]
  float* tmat = reinterpret_cast<float*>(smem + L::kT);               // packed T'
  float* pmat = reinterpret_cast<float*>(smem + L::kP);               // packed P
  float* cbuf = reinterpret_cast<float*>(smem + L::kC);               // [2][4][64]
  uint16_t* s_split = reinterpret_cast<uint16_t*>(smem + L::kSplit);  // [2][16][kRow]
  uint16_t* u_split = reinterpret_cast<uint16_t*>(smem + L::kUSplit); // [2][16][kURow]
  float* zx = reinterpret_cast<float*>(smem + L::kZx);                // [128][ZRow]
  float* u = reinterpret_cast<float*>(smem + L::kU);                  // [64][ZRow]
  uint64_t* bar = reinterpret_cast<uint64_t*>(smem + L::kBar);        // K/c x2, Q+V, T', P
  begin();
  const uint32_t seq = blockIdx.z;
  if (seq >= *live_sequences) return;
  const int head = blockIdx.y;
  const int key_head = head / Group;
  const int v0 = blockIdx.x * kScanRows;
  const int32_t seq_start = cu_seqlens[seq];
  const int32_t seq_rows = cu_seqlens[seq + 1] - seq_start;
  const int32_t chunk0 = first_chunk(cu_seqlens, seq);
  const int chunks = chunks_of(seq_rows);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t4 = lane & 3;
  if (threadIdx.x == 0) {
    for (int i = 0; i < 5; ++i) ptx::mbarrier_init(&bar[i], 1);
    ptx::fence_mbarrier_init();
  }
  __syncthreads();
  auto chunk_rows = [&](int n) { return min(kChunk, seq_rows - n * kChunk); };
  // Whole-block copies from the chunk's workspace, one thread and one
  // expect_tx arrival per barrier phase.
  using W = ChunkWork<KeyHeads, ValueHeads, Dim>;
  constexpr int RowsBytes = kChunk * L::kRow * 2;
  auto matrices = [&](int n) {
    return W::chunk(workspace, chunk0 + n) + W::kMatrices + uint64_t(head) * kWorkFloats * 4;
  };
  auto kq_rows = [&](int n) {
    return W::chunk(workspace, chunk0 + n) + W::kKq + uint64_t(key_head) * W::kKqBytes;
  };
  auto issue_kc = [&](int n) {
    if (threadIdx.x != 0) return;
    const int b = n & 1;
    ptx::fence_async_shared();
    ptx::mbarrier_arrive_expect_tx(&bar[b], RowsBytes + 4 * kChunk * 4);
    ptx::cp_async_bulk(kbuf + b * kChunk * L::kRow, kq_rows(n), RowsBytes, &bar[b]);
    ptx::cp_async_bulk(cbuf + b * 4 * kChunk, matrices(n) + kWorkC * 4, 4 * kChunk * 4, &bar[b]);
  };
  auto issue_qv = [&](int n) {
    if (threadIdx.x != 0) return;
    ptx::fence_async_shared();
    ptx::mbarrier_arrive_expect_tx(&bar[2], RowsBytes + kChunk * kScanRows * 2);
    ptx::cp_async_bulk(qbuf, kq_rows(n) + RowsBytes, RowsBytes, &bar[2]);
    ptx::cp_async_bulk(vbuf,
                       W::chunk(workspace, chunk0 + n) + W::kV + uint64_t(head) * W::kVBytes +
                           uint64_t(blockIdx.x) * kChunk * kScanRows * 2,
                       kChunk * kScanRows * 2, &bar[2]);
  };
  auto issue_matrix = [&](int n, float* dst, int offset, int barrier) {
    if (threadIdx.x != 0) return;
    ptx::fence_async_shared();
    ptx::mbarrier_arrive_expect_tx(&bar[barrier], kTri * 4);
    ptx::cp_async_bulk(dst, matrices(n) + offset * 4, kTri * 4, &bar[barrier]);
  };
  // Step 4 ownership: warp w holds S for value rows g, g + 8 and keys
  // 16w + 8n + 2 t4 (+1), n = 0, 1, in mma accumulator order.
  float* slot = state + (uint64_t(uint32_t(slot_ids[seq])) * ValueHeads + head) * Dim * Dim +
                uint64_t(v0) * Dim;
  const int32_t anchor = anchor_ids[seq];
  const int anchor_chunk = anchor >= 0 && seq_rows % kChunk != 0 ? seq_rows / kChunk - 1 : -1;
  float* anchor_state =
      state + (uint64_t(uint32_t(max(anchor, 0))) * ValueHeads + head) * Dim * Dim +
      uint64_t(v0) * Dim;
  ptx::D4 s_acc[2];
  const bool carry = initial[seq] != 0;
  #pragma unroll
  for (int n = 0; n < 2; ++n) {
    const int key = 16 * warp + 8 * n + 2 * t4;
    float2 top = make_float2(0.f, 0.f), bottom = top;
    if (carry) {
      top = *reinterpret_cast<const float2*>(slot + g * Dim + key);
      bottom = *reinterpret_cast<const float2*>(slot + (g + 8) * Dim + key);
    }
    s_acc[n].v[0] = top.x; s_acc[n].v[1] = top.y;
    s_acc[n].v[2] = bottom.x; s_acc[n].v[3] = bottom.y;
  }
  auto publish_state = [&]() {
    #pragma unroll
    for (int n = 0; n < 2; ++n) {
      const int key = 16 * warp + 8 * n + 2 * t4;
      #pragma unroll
      for (int half = 0; half < 2; ++half) {
        uint16_t h0, l0, h1, l1;
        split_bf16(s_acc[n].v[2 * half], h0, l0);
        split_bf16(s_acc[n].v[2 * half + 1], h1, l1);
        const int row = g + 8 * half;
        *reinterpret_cast<uint32_t*>(s_split + row * L::kRow + key) =
            uint32_t(h0) | (uint32_t(h1) << 16);
        *reinterpret_cast<uint32_t*>(s_split + (kScanRows + row) * L::kRow + key) =
            uint32_t(l0) | (uint32_t(l1) << 16);
      }
    }
  };
  if (chunks > 0) {
    issue_kc(0);
    issue_qv(0);
    issue_matrix(0, tmat, kWorkT, 3);
    issue_matrix(0, pmat, kWorkP, 4);
  }
  publish_state();
  __syncthreads();
  // FMA ownership for steps 2 and 3: value column ov, rows orow + 16 m.
  const int ov = threadIdx.x % 16, orow = threadIdx.x / 16;
  for (int c = 0; c < chunks; ++c) {
    const int32_t first_row = seq_start + c * kChunk;
    const int rows = chunk_rows(c);
    const int b = c & 1;
    const uint16_t* kc = kbuf + b * kChunk * L::kRow;
    const float* cvec = cbuf + b * 4 * kChunk;
    ptx::mbarrier_wait(&bar[b], (c >> 1) & 1);
    ptx::mbarrier_wait(&bar[2], c & 1);
    // Step 1: warp w computes rows 16w..16w+15 of [Kr; Qr] S^T.
    {
      const uint16_t* a_rows =
          warp < 4 ? kc + 16 * warp * L::kRow : qbuf + 16 * (warp - 4) * L::kRow;
      ptx::D4 acc[2] = {};
      #pragma unroll
      for (int k = 0; k < Dim; k += 16) {
        ptx::A4 a;
        ptx::ldmatrix_b16<4>(a.v, a_rows + (lane & 15) * L::kRow + k + (lane >> 4) * 8);
        #pragma unroll
        for (int part = 1; part >= 0; --part) {
          uint32_t bf[4];
          ptx::ldmatrix_b16<4>(
              bf, s_split + (part * kScanRows + (lane & 7) + ((lane >> 4) << 3)) * L::kRow + k +
                      ((lane >> 3) & 1) * 8);
          ptx::mma_bf16(acc[0], a, ptx::B2{{bf[0], bf[1]}});
          ptx::mma_bf16(acc[1], a, ptx::B2{{bf[2], bf[3]}});
        }
      }
      #pragma unroll
      for (int n = 0; n < 2; ++n)
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int row = 16 * warp + g + 8 * (e >> 1);
          const int col = 8 * n + 2 * t4 + (e & 1);
          float value = acc[n].v[e];
          if (row < kChunk) value = fmaf(-cvec[row], value, bf(vbuf[row * kScanRows + col]));
          zx[row * ZRow + col] = value;
        }
    }
    __syncthreads();
    if (c + 1 < chunks) {
      issue_kc(c + 1);
      issue_qv(c + 1);
    }
    ptx::mbarrier_wait(&bar[3], c & 1);
    // Step 2: U = T' Z, and (c3 U)^T split for step 4. T' and P are lower
    // triangular, so row i stops at column i.
    {
      float acc[4] = {};
      triangular_times<ZRow>(acc, tmat, zx, orow, ov);
      #pragma unroll
      for (int m = 0; m < 4; ++m) {
        const int i = orow + 16 * m;
        u[i * ZRow + ov] = acc[m];
        uint16_t hi, lo;
        split_bf16(cvec[2 * kChunk + i] * acc[m], hi, lo);
        u_split[ov * L::kURow + i] = hi;
        u_split[(kScanRows + ov) * L::kURow + i] = lo;
      }
    }
    __syncthreads();
    if (c + 1 < chunks) issue_matrix(c + 1, tmat, kWorkT, 3);
    ptx::mbarrier_wait(&bar[4], c & 1);
    // Step 3: O = diag(c2) X + P U.
    {
      float acc[4];
      #pragma unroll
      for (int m = 0; m < 4; ++m) {
        const int i = orow + 16 * m;
        acc[m] = cvec[kChunk + i] * zx[(kChunk + i) * ZRow + ov];
      }
      triangular_times<ZRow>(acc, pmat, u, orow, ov);
      #pragma unroll
      for (int m = 0; m < 4; ++m) {
        const int i = orow + 16 * m;
        if (i < rows)
          output[(uint64_t(first_row + i) * ValueHeads + head) * Dim + v0 + ov] = to_bf(acc[m]);
      }
    }
    // Step 4: S = exp(G_L) S + (c3 U)^T Kr on the tensor cores.
    {
      const float gl = cvec[3 * kChunk];
      #pragma unroll
      for (int n = 0; n < 2; ++n)
        #pragma unroll
        for (int e = 0; e < 4; ++e) s_acc[n].v[e] *= gl;
      #pragma unroll
      for (int k = 0; k < kChunk; k += 16) {
        uint32_t bf[4];
        ptx::ldmatrix_b16<4, true>(
            bf, kc + (k + (lane & 15)) * L::kRow + 16 * warp + (lane >> 4) * 8);
        #pragma unroll
        for (int part = 1; part >= 0; --part) {
          ptx::A4 a;
          ptx::ldmatrix_b16<4>(
              a.v, u_split + (part * kScanRows + (lane & 15)) * L::kURow + k + (lane >> 4) * 8);
          ptx::mma_bf16(s_acc[0], a, ptx::B2{{bf[0], bf[1]}});
          ptx::mma_bf16(s_acc[1], a, ptx::B2{{bf[2], bf[3]}});
        }
      }
    }
    if (c == anchor_chunk) {
      #pragma unroll
      for (int n = 0; n < 2; ++n) {
        const int key = 16 * warp + 8 * n + 2 * t4;
        *reinterpret_cast<float2*>(anchor_state + g * Dim + key) =
            make_float2(s_acc[n].v[0], s_acc[n].v[1]);
        *reinterpret_cast<float2*>(anchor_state + (g + 8) * Dim + key) =
            make_float2(s_acc[n].v[2], s_acc[n].v[3]);
      }
    }
    __syncthreads();
    if (c + 1 < chunks) issue_matrix(c + 1, pmat, kWorkP, 4);
    publish_state();
    __syncthreads();
  }
  #pragma unroll
  for (int n = 0; n < 2; ++n) {
    const int key = 16 * warp + 8 * n + 2 * t4;
    *reinterpret_cast<float2*>(slot + g * Dim + key) = make_float2(s_acc[n].v[0], s_acc[n].v[1]);
    *reinterpret_cast<float2*>(slot + (g + 8) * Dim + key) =
        make_float2(s_acc[n].v[2], s_acc[n].v[3]);
  }
}

} // namespace chop::linear
