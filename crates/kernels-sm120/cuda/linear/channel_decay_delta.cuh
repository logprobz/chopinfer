#pragma once

// Token-by-token per-channel-decay delta rule for decode and speculative
// verify. Per row and head, with q = q / sqrt(sum(q^2) + 1e-6) then scaled
// by Dim^-0.5, k = k / sqrt(sum(k^2) + 1e-6), beta = bf16(sigmoid(b)) and g
// per key channel (channel_decay_common.cuh):
//   S[v][k] *= exp(g[k]); u = S k; delta = (v - u) * beta;
//   S += k delta^T; o = S q
// The F32 slot state is [Heads, Dim (value), Dim (key)]. A CTA owns
// kRowsPerCta value rows of one head of one sequence and keeps them in
// registers across rows; each warp owns kRowsPerWarp rows, each lane four
// consecutive keys of a row. Reductions over keys are lane-local FMA from
// zero followed by the butterfly sum: one fixed order for every row.

#include "channel_decay_common.cuh"

namespace chop::linear::channel_decay {

constexpr int kRowsPerWarp = 4;
constexpr int kRecurrentWarps = 4;
constexpr int kRowsPerCta = kRowsPerWarp * kRecurrentWarps;

// One row's operands for a lane: q and k normalized (q scaled), the lane's
// four channel decays exp(g), beta, and the warp's kRowsPerWarp values.
struct ChannelRow {
  float q[4];
  float k[4];
  float decay[4];
  float v[kRowsPerWarp];
  float beta;
};

template<int Heads, int Dim, int Gate, uint32_t BoundBits>
__device__ __forceinline__ void load_channel_row(const uint16_t* qkv, uint32_t qkv_stride,
                                                 const uint16_t* f, uint32_t f_stride,
                                                 const uint16_t* b, uint32_t b_stride,
                                                 const float (&dt_bias)[4], float a,
                                                 uint64_t row, int head, int v0, int lane,
                                                 ChannelRow& out) {
  const uint16_t* base = qkv + row * qkv_stride;
  const uint2 qp = *reinterpret_cast<const uint2*>(base + head * Dim + lane * 4);
  const uint2 kp = *reinterpret_cast<const uint2*>(base + (Heads + head) * Dim + lane * 4);
  const uint2 vp = *reinterpret_cast<const uint2*>(base + (2 * Heads + head) * Dim + v0);
  const uint2 fp = *reinterpret_cast<const uint2*>(f + row * f_stride + head * Dim + lane * 4);
  float q[4] = {bf_lo(qp.x), bf_hi(qp.x), bf_lo(qp.y), bf_hi(qp.y)};
  float k[4] = {bf_lo(kp.x), bf_hi(kp.x), bf_lo(kp.y), bf_hi(kp.y)};
  const float fg[4] = {bf_lo(fp.x), bf_hi(fp.x), bf_lo(fp.y), bf_hi(fp.y)};
  out.v[0] = bf_lo(vp.x); out.v[1] = bf_hi(vp.x);
  out.v[2] = bf_lo(vp.y); out.v[3] = bf_hi(vp.y);
  const float q_norm = l2_norm4(q);
  const float k_norm = l2_norm4(k);
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    out.q[j] = (q[j] / q_norm) * query_scale<Dim>();
    out.k[j] = k[j] / k_norm;
    out.decay[j] = expf(log_decay<Gate, BoundBits>(fg[j], dt_bias[j], a));
  }
  out.beta = beta_of(b[row * b_stride + head]);
}

// Advances the warp's rows by one token and returns lane-local outputs for
// row `lane` (lanes 0..kRowsPerWarp-1 hold valid values).
__device__ __forceinline__ float channel_step(float4 (&s)[kRowsPerWarp], const ChannelRow& in,
                                              int lane) {
  float u[kRowsPerWarp];
  #pragma unroll
  for (int i = 0; i < kRowsPerWarp; ++i) {
    s[i].x *= in.decay[0];
    s[i].y *= in.decay[1];
    s[i].z *= in.decay[2];
    s[i].w *= in.decay[3];
    u[i] = s[i].x * in.k[0];
    u[i] = fmaf(s[i].y, in.k[1], u[i]);
    u[i] = fmaf(s[i].z, in.k[2], u[i]);
    u[i] = fmaf(s[i].w, in.k[3], u[i]);
  }
  #pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
    #pragma unroll
    for (int i = 0; i < kRowsPerWarp; ++i) u[i] += __shfl_xor_sync(0xffffffffu, u[i], offset);
  }
  float o[kRowsPerWarp];
  #pragma unroll
  for (int i = 0; i < kRowsPerWarp; ++i) {
    const float delta = (in.v[i] - u[i]) * in.beta;
    s[i].x = fmaf(in.k[0], delta, s[i].x);
    s[i].y = fmaf(in.k[1], delta, s[i].y);
    s[i].z = fmaf(in.k[2], delta, s[i].z);
    s[i].w = fmaf(in.k[3], delta, s[i].w);
    o[i] = s[i].x * in.q[0];
    o[i] = fmaf(s[i].y, in.q[1], o[i]);
    o[i] = fmaf(s[i].z, in.q[2], o[i]);
    o[i] = fmaf(s[i].w, in.q[3], o[i]);
  }
  #pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
    #pragma unroll
    for (int i = 0; i < kRowsPerWarp; ++i) o[i] += __shfl_xor_sync(0xffffffffu, o[i], offset);
  }
  float out = 0.f;
  #pragma unroll
  for (int i = 0; i < kRowsPerWarp; ++i)
    if (lane == i) out = o[i];
  return out;
}

// Grid: (Dim / kRowsPerCta, Heads, sequence capacity), block
// 32 * kRecurrentWarps. Operands: qkv [rows, qkv_stride] BF16 (q, k, v of
// Heads x Dim each), the gate projection f [rows, f_stride] BF16
// (Heads x Dim), beta logits b [rows, b_stride] BF16 (Heads), A_log [Heads]
// and dt_bias [Heads * Dim] F32. Decode (MaxRows 0): row `seq` of every
// operand, and the slot state advances in place. Verify (MaxRows > 0): rows
// cu_rows[seq]..cu_rows[seq + 1] (at most MaxRows); the committed state is
// read from slot_ids[seq] and the state after row r is written to
// checkpoint_ids[seq * MaxRows + r]. Output is [rows, Heads, Dim] BF16.
template<int Heads, int Dim, int Gate, uint32_t BoundBits, int MaxRows>
__device__ void channel_delta_recurrent(const uint16_t* qkv, uint32_t qkv_stride,
                                        const uint16_t* f, uint32_t f_stride,
                                        const uint16_t* b, uint32_t b_stride,
                                        const float* a_log, const float* dt_bias,
                                        float* state, uint16_t* output,
                                        const int32_t* cu_rows, const int32_t* slot_ids,
                                        const int32_t* checkpoint_ids,
                                        const uint32_t* live_sequences) {
  static_assert(Dim == 128, "lanes hold four keys of a 128-wide head");
  constexpr bool Verify = MaxRows > 0;
  constexpr uint64_t HeadElements = uint64_t(Dim) * Dim;
  const int head = blockIdx.y;
  const int lane = threadIdx.x & 31;
  ptx::griddep_launch_dependents();
  // Parameters are immutable across graph launches.
  const float a = expf(a_log[head]);
  float bias[4];
  #pragma unroll
  for (int j = 0; j < 4; ++j) bias[j] = dt_bias[head * Dim + lane * 4 + j];
  ptx::griddep_wait();
  const uint32_t seq = blockIdx.z;
  if (seq >= *live_sequences) return;
  const int v0 = blockIdx.x * kRowsPerCta + (threadIdx.x >> 5) * kRowsPerWarp;
  int32_t first = seq, rows = 1;
  if constexpr (Verify) {
    first = cu_rows[seq];
    rows = min(cu_rows[seq + 1] - first, MaxRows);
  }
  const uint64_t head_offset = uint64_t(head) * HeadElements + uint64_t(v0) * Dim + lane * 4;
  float* source = state + uint64_t(uint32_t(slot_ids[seq])) * Heads * HeadElements + head_offset;
  float4 s[kRowsPerWarp];
  #pragma unroll
  for (int i = 0; i < kRowsPerWarp; ++i)
    s[i] = *reinterpret_cast<const float4*>(source + i * Dim);
  for (int r = 0; r < rows; ++r) {
    ChannelRow in;
    load_channel_row<Heads, Dim, Gate, BoundBits>(qkv, qkv_stride, f, f_stride, b, b_stride,
                                                  bias, a, uint64_t(first + r), head, v0,
                                                  lane, in);
    const float out = channel_step(s, in, lane);
    if (lane < kRowsPerWarp)
      output[(uint64_t(first + r) * Heads + head) * Dim + v0 + lane] = to_bf(out);
    if constexpr (Verify) {
      float* checkpoint =
          state + uint64_t(uint32_t(checkpoint_ids[seq * MaxRows + r])) * Heads * HeadElements +
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

} // namespace chop::linear::channel_decay
