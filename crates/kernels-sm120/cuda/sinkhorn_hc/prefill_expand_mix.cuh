#pragma once

// Prefill expansion then lagged mixing in one launch. Each CTA first expands
// its own Rows rows once, with expand_f32's expressions, into `expanded`, then
// runs the unchanged prefill pipeline over those rows while they are still in
// L2. The projection must consume the flat row in ascending K order, which is
// stream-major, so the expansion cannot be interleaved per column block
// without rereading every input stream for each output stream.
#include "lagged.cuh"
#include "prefill_pipeline.cuh"

namespace chop::sinkhorn_hc::prefill_expand_mix {

enum class Mutation { none, reverse_expansion, fused_post, round_mixed };

// One 8-column pack of one row: every output stream j of columns
// col..col+7 is bf16(fl(post_j * y) + sum_i comb[i][j] * X_i), the stream sum
// an FMA chain from zero in stream order, as expand_f32 computes it.
template<class M, Mutation Mut>
__device__ __forceinline__ void expand_pack(const uint16_t* x, const uint16_t* y, const float* post,
                                            const float* comb, uint16_t* out, uint32_t row,
                                            int col) {
  constexpr int N = M::kStreams;
  constexpr int H = M::kWidth;
  const uint64_t base = uint64_t(row) * N * H + col;
  uint4 streams[N];
  #pragma unroll
  for (int i = 0; i < N; ++i)
    streams[i] = *reinterpret_cast<const uint4*>(x + base + uint64_t(i) * H);
  const uint4 ys = *reinterpret_cast<const uint4*>(y + uint64_t(row) * H + col);
  const uint32_t yw[4] = {ys.x, ys.y, ys.z, ys.w};
  const float* weights = comb + uint64_t(row) * N * N;
  #pragma unroll
  for (int j = 0; j < N; ++j) {
    const float p = post[uint64_t(row) * N + j];
    uint32_t result[4];
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
      float mixed_lo = 0.f, mixed_hi = 0.f;
      #pragma unroll
      for (int k = 0; k < N; ++k) {
        const int i = Mut == Mutation::reverse_expansion ? N - 1 - k : k;
        const uint32_t xw[4] = {streams[i].x, streams[i].y, streams[i].z, streams[i].w};
        const float w = weights[i * N + j];
        mixed_lo = fmaf(w, bf_lo(xw[e]), mixed_lo);
        mixed_hi = fmaf(w, bf_hi(xw[e]), mixed_hi);
      }
      if constexpr (Mut == Mutation::round_mixed) {
        const uint32_t rounded = pack(mixed_lo, mixed_hi);
        mixed_lo = bf_lo(rounded);
        mixed_hi = bf_hi(rounded);
      }
      float out_lo, out_hi;
      if constexpr (Mut == Mutation::fused_post) {
        out_lo = fmaf(p, bf_lo(yw[e]), mixed_lo);
        out_hi = fmaf(p, bf_hi(yw[e]), mixed_hi);
      } else {
        out_lo = __fadd_rn(__fmul_rn(p, bf_lo(yw[e])), mixed_lo);
        out_hi = __fadd_rn(__fmul_rn(p, bf_hi(yw[e])), mixed_hi);
      }
      result[e] = pack(out_lo, out_hi);
    }
    *reinterpret_cast<uint4*>(out + base + uint64_t(j) * H) =
        make_uint4(result[0], result[1], result[2], result[3]);
  }
}

// Grid ceil(row capacity / Rows), 256 threads. x, y, post_in and comb_in are
// expand_f32's operands; expanded is its output and the mixer's input (it
// must not alias x). The mixer outputs may alias post_in and comb_in: every
// coefficient read of a row finishes before the barrier that precedes the
// pipeline, and only this CTA reads or writes its rows.
template<class M, int Rows, int Chains, Mutation Mut>
__device__ void run(const uint16_t* x, const uint16_t* y, const float* post_in,
                    const float* comb_in, uint16_t* expanded, const void* fn, const float* base,
                    const float* scale, const float* pre_in, uint16_t* collapsed, float* pre,
                    float* post, float* comb, const uint32_t* live_rows) {
  static_assert(M::kWidth % 8 == 0 && Rows >= 1 && Rows <= 8);
  if (blockDim.x != kThreads) __trap();
  constexpr int Packs = M::kWidth / 8;
  constexpr int ElementBytes = M::kF32Fn ? 4 : 2;
  // fn is a weight: warm L2 with it before waiting on the producer, as the
  // pipeline does.
  {
    constexpr uint64_t Lines = (uint64_t(M::kMixes) * M::kFlat * ElementBytes + 127) / 128;
    for (uint64_t line = uint64_t(blockIdx.x) * kThreads + threadIdx.x; line < Lines;
         line += uint64_t(gridDim.x) * kThreads) {
      asm volatile("prefetch.global.L2 [%0];" ::"l"(static_cast<const char*>(fn) + line * 128));
    }
  }
  ptx::griddep_wait();
  const uint32_t first = blockIdx.x * Rows;
  const uint32_t live = *live_rows;
  if (first >= live) return;
  const int count = live - first < uint32_t(Rows) ? int(live - first) : Rows;
  // Two independent packs per iteration keep two rows' loads in flight.
  for (int item = threadIdx.x; item < count * Packs; item += 2 * kThreads) {
    expand_pack<M, Mut>(x, y, post_in, comb_in, expanded, first + item / Packs,
                        item % Packs * 8);
    const int next = item + kThreads;
    if (next < count * Packs)
      expand_pack<M, Mut>(x, y, post_in, comb_in, expanded, first + next / Packs,
                          next % Packs * 8);
  }
  // The pipeline's asynchronous copies and collapse read these rows.
  __threadfence_block();
  __syncthreads();
  mix_collapse_prefill_pipeline<M, Rows, Chains, false>(expanded, fn, base, scale, pre_in,
                                                        collapsed, pre, post, comb, nullptr,
                                                        nullptr, nullptr, live_rows);
}

} // namespace chop::sinkhorn_hc::prefill_expand_mix
