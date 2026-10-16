#pragma once

// Token-by-token gated delta rule for decode and speculative verify.
// Per row t and value head h, with q and k L2-normalized (eps 1e-6) and q
// scaled by 1/sqrt(Dim):
//   decay = exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b)
//   d = beta * (v - decay * S k);  S = decay * S + d k^T;  o = S q
// The F32 slot state is [ValueHeads, Dim (value), Dim (key)]. Each state row
// (one value index) is independent, so a CTA owns kRowsPerCta value rows of
// one head of one sequence and keeps them in registers across rows. Each
// warp owns kRowsPerWarp rows, each lane four consecutive keys of a row.

#include "common.cuh"

namespace chop::linear {

constexpr int kRowsPerWarp = 4;
constexpr int kRecurrentWarps = 4;
constexpr int kRowsPerCta = kRowsPerWarp * kRecurrentWarps;

struct RecurrentRow {
  float q[4];
  float k[4];
  float v[kRowsPerWarp];
  float a;
  float b;
};

template<int KeyHeads, int ValueHeads, int Dim>
__device__ __forceinline__ void load_recurrent_row(const uint16_t* qkv,
                                                   uint32_t qkv_stride,
                                                   const uint16_t* a,
                                                   const uint16_t* b,
                                                   uint32_t gate_stride,
                                                   uint64_t row, int key_head,
                                                   int head, int v0, int lane,
                                                   RecurrentRow& out) {
  const uint16_t* base = qkv + row * qkv_stride;
  const uint2 qp = *reinterpret_cast<const uint2*>(base + key_head * Dim + lane * 4);
  const uint2 kp =
      *reinterpret_cast<const uint2*>(base + (KeyHeads + key_head) * Dim + lane * 4);
  const uint2 vp =
      *reinterpret_cast<const uint2*>(base + (2 * KeyHeads + head) * Dim + v0);
  out.q[0] = bf_lo(qp.x); out.q[1] = bf_hi(qp.x);
  out.q[2] = bf_lo(qp.y); out.q[3] = bf_hi(qp.y);
  out.k[0] = bf_lo(kp.x); out.k[1] = bf_hi(kp.x);
  out.k[2] = bf_lo(kp.y); out.k[3] = bf_hi(kp.y);
  out.v[0] = bf_lo(vp.x); out.v[1] = bf_hi(vp.x);
  out.v[2] = bf_lo(vp.y); out.v[3] = bf_hi(vp.y);
  out.a = bf(a[row * gate_stride + head]);
  out.b = bf(b[row * gate_stride + head]);
}

// Advances the warp's rows by one token and returns lane-local outputs for
// row `lane` (lanes 0..kRowsPerWarp-1 hold valid values).
template<int Dim>
__device__ __forceinline__ float recurrent_step(float4 (&s)[kRowsPerWarp],
                                                const RecurrentRow& in,
                                                float neg_a, float dt_bias,
                                                int lane) {
  float qq = 0.f, kk = 0.f, qk = 0.f;
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    qq = fmaf(in.q[j], in.q[j], qq);
    kk = fmaf(in.k[j], in.k[j], kk);
    qk = fmaf(in.q[j], in.k[j], qk);
  }
  qq = warp_sum(qq);
  kk = warp_sum(kk);
  qk = warp_sum(qk);
  const float q_scale = rsqrtf(qq + 1e-6f) * rsqrtf(float(Dim));
  const float k_scale = rsqrtf(kk + 1e-6f);
  float qn[4], kn[4];
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    qn[j] = in.q[j] * q_scale;
    kn[j] = in.k[j] * k_scale;
  }
  const float kq = qk * q_scale * k_scale;
  const float decay = expf(neg_a * softplus(in.a + dt_bias));
  const float beta = sigmoid(in.b);
  float sk[kRowsPerWarp], sq[kRowsPerWarp];
  #pragma unroll
  for (int i = 0; i < kRowsPerWarp; ++i) {
    sk[i] = s[i].x * kn[0];
    sk[i] = fmaf(s[i].y, kn[1], sk[i]);
    sk[i] = fmaf(s[i].z, kn[2], sk[i]);
    sk[i] = fmaf(s[i].w, kn[3], sk[i]);
    sq[i] = s[i].x * qn[0];
    sq[i] = fmaf(s[i].y, qn[1], sq[i]);
    sq[i] = fmaf(s[i].z, qn[2], sq[i]);
    sq[i] = fmaf(s[i].w, qn[3], sq[i]);
  }
  #pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
    #pragma unroll
    for (int i = 0; i < kRowsPerWarp; ++i) {
      sk[i] += __shfl_xor_sync(0xffffffffu, sk[i], offset);
      sq[i] += __shfl_xor_sync(0xffffffffu, sq[i], offset);
    }
  }
  float out = 0.f;
  #pragma unroll
  for (int i = 0; i < kRowsPerWarp; ++i) {
    const float delta = beta * (in.v[i] - decay * sk[i]);
    const float o = fmaf(delta, kq, decay * sq[i]);
    if (lane == i) out = o;
    s[i].x = fmaf(kn[0], delta, decay * s[i].x);
    s[i].y = fmaf(kn[1], delta, decay * s[i].y);
    s[i].z = fmaf(kn[2], delta, decay * s[i].z);
    s[i].w = fmaf(kn[3], delta, decay * s[i].w);
  }
  return out;
}

// Grid: (Dim / kRowsPerCta, ValueHeads, sequence capacity), block
// 32 * kRecurrentWarps. Decode (MaxRows 0): row `seq` of every operand, and
// the slot state advances in place. Verify (MaxRows > 0): rows
// cu_rows[seq]..cu_rows[seq + 1] (at most MaxRows); the committed state is
// read from slot_ids[seq] and the state after row r is written to slot
// checkpoint_ids[seq * MaxRows + r]. Output is [rows, ValueHeads, Dim] BF16.
template<int KeyHeads, int ValueHeads, int Dim, int MaxRows>
__device__ void gated_delta_recurrent(const uint16_t* qkv, uint32_t qkv_stride,
                                      const uint16_t* a, const uint16_t* b,
                                      uint32_t gate_stride, const float* a_log,
                                      const float* dt_bias, float* state,
                                      uint16_t* output, const int32_t* cu_rows,
                                      const int32_t* slot_ids,
                                      const int32_t* checkpoint_ids,
                                      const uint32_t* live_sequences) {
  static_assert(Dim == 128 && ValueHeads % KeyHeads == 0);
  constexpr bool Verify = MaxRows > 0;
  constexpr int Rows = Verify ? MaxRows : 1;
  constexpr uint64_t HeadElements = uint64_t(Dim) * Dim;
  float staged_log_a = 0.f, staged_bias = 0.f;
  if constexpr (Verify) {
    // Serving also uses this entry for AR. Keep its original load order:
    // early constants regressed the eight-row C32 graph measurement.
    begin();
  } else {
    ptx::griddep_launch_dependents();
    staged_log_a = a_log[blockIdx.y];
    staged_bias = dt_bias[blockIdx.y];
    ptx::griddep_wait();
  }
  const uint32_t seq = blockIdx.z;
  if (seq >= *live_sequences) return;
  const int head = blockIdx.y;
  const int key_head = head / (ValueHeads / KeyHeads);
  const int lane = threadIdx.x & 31;
  const int v0 = blockIdx.x * kRowsPerCta + (threadIdx.x >> 5) * kRowsPerWarp;
  int32_t first = seq, rows = 1;
  if constexpr (Verify) {
    first = cu_rows[seq];
    rows = min(cu_rows[seq + 1] - first, MaxRows);
  }
  const uint64_t head_offset = uint64_t(head) * HeadElements + uint64_t(v0) * Dim + lane * 4;
  float* source =
      state + uint64_t(uint32_t(slot_ids[seq])) * ValueHeads * HeadElements + head_offset;
  float4 s[kRowsPerWarp];
  #pragma unroll
  for (int i = 0; i < kRowsPerWarp; ++i)
    s[i] = *reinterpret_cast<const float4*>(source + i * Dim);
  RecurrentRow in[Rows];
  #pragma unroll
  for (int r = 0; r < Rows; ++r)
    if (r < rows)
      load_recurrent_row<KeyHeads, ValueHeads, Dim>(qkv, qkv_stride, a, b, gate_stride,
                                                    uint64_t(first + r), key_head, head,
                                                    v0, lane, in[r]);
  const float neg_a = -expf(Verify ? a_log[head] : staged_log_a);
  const float bias = Verify ? dt_bias[head] : staged_bias;
  #pragma unroll
  for (int r = 0; r < Rows; ++r) {
    if (r >= rows) break;
    const float out = recurrent_step<Dim>(s, in[r], neg_a, bias, lane);
    if (lane < kRowsPerWarp)
      output[(uint64_t(first + r) * ValueHeads + head) * Dim + v0 + lane] = to_bf(out);
    if constexpr (Verify) {
      float* checkpoint =
          state + uint64_t(uint32_t(checkpoint_ids[seq * MaxRows + r])) * ValueHeads * HeadElements +
          head_offset;
      #pragma unroll
      for (int i = 0; i < kRowsPerWarp; ++i)
        *reinterpret_cast<float4*>(checkpoint + i * Dim) = s[i];
    }
  }
  if constexpr (!Verify) {
    #pragma unroll
    for (int i = 0; i < kRowsPerWarp; ++i)
      *reinterpret_cast<float4*>(source + i * Dim) = s[i];
  }
}

} // namespace chop::linear
