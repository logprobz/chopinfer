#pragma once
#include "mix_split.cuh"

// Lab ablations of the locked split mixer. Only full mode is a serving
// equivalence candidate; the other modes deliberately omit work.
namespace chop::sinkhorn_hc::breakdown {
enum class Mode : int { full = 0, loads = 1, resident = 2, sinkhorn = 3, ticket = 4 };
constexpr int kMarks = 9;
constexpr int kFields = 12; // SM, last-arrival flag, mode, nine timestamps.

template<bool Sampled> struct Clock {
  uint64_t time[kMarks];
  bool enabled;
  __device__ explicit Clock(uint64_t* output) : enabled(Sampled && output != nullptr && threadIdx.x < 32) {}
  __device__ __forceinline__ void mark(int stage) {
    if constexpr (Sampled) if (enabled) time[stage] = clock64();
  }
  __device__ void end(uint64_t* output, uint32_t tile, uint32_t row, uint32_t groups,
                      Mode kind, bool last, bool finished) {
    if constexpr (Sampled) if (enabled) {
      if (!finished) {
        #pragma unroll
        for (int stage = 4; stage < kMarks; ++stage) time[stage] = time[3];
      }
      uint32_t sm;
      asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
      const uint64_t at = ((uint64_t(row)*groups+tile)*32+threadIdx.x)*kFields;
      output[at] = sm; output[at+1] = last; output[at+2] = int(kind);
      #pragma unroll
      for (int stage = 0; stage < kMarks; ++stage) output[at+3+stage] = time[stage];
    }
  }
};

template<class M> struct Arena {
  // Ablation packets are dead before a finishing CTA reuses this arena.
  union {
    float packet[kThreads*4];
    struct {
      float sums[M::kMixes*(M::kChains+1)];
      float partials[kWarps];
      float logits[M::kMixes];
      float weights[M::kStreams];
      float cells[M::kCells];
    } finish;
  };
  uint32_t last;
};

// The full arm calls the locked projection helper without modification.
// Loads uses volatile shared stores to retain every original vector load.
// Resident loads one packet once, then reads volatile shared operands for
// each original mix so the compiler cannot hoist or eliminate its math.
template<class M, Mode Kind>
__device__ void projection(const uint16_t* x, const void* fn, float* parts,
                           float* loaded, uint32_t tile, uint32_t row, unsigned char* arena) {
  static_assert(M::kF32Fn, "this diagnostic covers F32 projection weights");
  const int t = threadIdx.x, lane = t & 31;
  const int chain = tile*kWarps+(t>>5);
  const int k = chain*kChain+lane*4;
  const uint16_t* source = x+uint64_t(row)*M::kFlat;
  if constexpr (Kind == Mode::full) {
    chain_sums<M>(source,fn,chain,parts);
  } else if constexpr (Kind == Mode::loads || Kind == Mode::resident) {
    volatile float* packet = reinterpret_cast<Arena<M>*>(arena)->packet;
    if constexpr (Kind == Mode::loads) {
      #pragma unroll
      for (int m = 0; m < M::kMixes; ++m) {
        float w[4]; load_fn<true>(fn,uint64_t(m)*M::kFlat+k,w);
        #pragma unroll
        for (int j = 0; j < 4; ++j) packet[4*t+j] = w[j];
        if (loaded != nullptr) {
          *reinterpret_cast<float4*>(loaded+(uint64_t(row)*M::kMixes+m)*M::kFlat+k) =
              make_float4(w[0],w[1],w[2],w[3]);
        }
      }
    } else {
      float initial[4]; load_fn<true>(fn,k,initial);
      #pragma unroll
      for (int j = 0; j < 4; ++j) packet[4*t+j] = initial[j];
      const uint2 xs = *reinterpret_cast<const uint2*>(source+k);
      const float xv[4] = {bf_lo(xs.x),bf_hi(xs.x),bf_lo(xs.y),bf_hi(xs.y)};
      #pragma unroll
      for (int m = 0; m < M::kMixes; ++m) {
        float sum = __fmul_rn(packet[4*t],xv[0]);
        sum = fmaf(packet[4*t+1],xv[1],sum);
        sum = fmaf(packet[4*t+2],xv[2],sum);
        sum = fmaf(packet[4*t+3],xv[3],sum);
        sum = warp_sum(sum);
        if (lane == 0) parts[m*M::kChains+chain] = sum;
      }
    }
  }
}

__device__ bool arrive(uint32_t* ticket, uint32_t groups, uint32_t& last) {
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    uint32_t arrived;
    asm volatile("atom.add.acq_rel.gpu.u32 %0, [%1], 1;"
                 : "=r"(arrived) : "l"(ticket) : "memory");
    if (arrived >= groups) asm volatile("trap;");
    last = arrived == groups-1;
    if (last) *ticket = 0;
  }
  __syncthreads();
  return last;
}

template<class M, Mode Kind, bool Sampled>
__device__ void finish(const uint16_t* x, const void* fn, const float* base,
                                    const float* scale, const float* pre_in,
                                    uint16_t* collapsed, float* pre, float* post, float* comb,
                                    const float* seed_logits, float* observed_logits,
                                    const float* chain_partials, uint32_t row, unsigned char* arena, Clock<Sampled>& clock) {
  constexpr int N = M::kStreams;
  constexpr int H = M::kWidth;
  constexpr int Flat = M::kFlat;
  constexpr int Mixes = M::kMixes;
  constexpr int Cells = M::kCells;
  constexpr int Stride = M::kChains + 1;
  const uint16_t* streams = x + uint64_t(row) * Flat;
  auto& shared = reinterpret_cast<Arena<M>*>(arena)->finish;
  auto& sums = shared.sums;
  auto& partials = shared.partials;
  auto& logits = shared.logits;
  auto& weights = shared.weights;
  auto& cells = shared.cells;
  const int t = threadIdx.x;
  const int lane = t & 31;
  const int warp = t >> 5;

  if constexpr (Kind == Mode::sinkhorn) {
    clock.mark(4);
    if (t < Mixes) logits[t] = seed_logits[uint64_t(row)*Mixes+t];
    __syncthreads();
  } else {
  // The collapse weights the previous sublayer stored (or the one-hot
  // weights before the first), published by the barrier below.
  if (t < N) weights[t] = pre_in[uint64_t(row) * N + t];

  // Stage the row and sum its squares: mix_collapse's order.
  const uint4* source = reinterpret_cast<const uint4*>(x + uint64_t(row) * Flat);
  float squares = 0.f;
  for (int v = t; v < Flat / 8; v += kThreads) {
    const uint4 packed = __ldcg(source + v);
    const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float lo = bf_lo(words[j]);
      const float hi = bf_hi(words[j]);
      squares = fmaf(lo, lo, squares);
      squares = fmaf(hi, hi, squares);
    }
  }
  for (int i = t; i < Mixes * M::kChains; i += kThreads) {
    sums[(i / M::kChains) * Stride + i % M::kChains] = __ldcg(chain_partials + i);
  }
  squares = warp_sum(squares);
  if (lane == 0) partials[warp] = squares;
  __syncthreads();
  float total = 0.f;
  #pragma unroll
  for (int w = 0; w < kWarps; ++w) total = __fadd_rn(total, partials[w]);
  const float rstd =
      __frsqrt_rn(__fadd_rn(__fdiv_rn(total, float(Flat)), __uint_as_float(M::kNormEpsBits)));

  clock.mark(4);

  // Fold each mix's chain sums in the original increasing K order.
  if (t < Mixes) {
    float acc = 0.f;
    #pragma unroll 8
    for (int step = 0; step < M::kChains; ++step) {
      #ifdef CHOP_HC_LAGGED_REVERSE_CHAINS
      const int c = M::kChains - 1 - step;
      #else
      const int c = step;
      #endif
      acc = __fadd_rn(acc, sums[t * Stride + c]);
    }
    logits[t] = __fmul_rn(acc, rstd);
    if (observed_logits != nullptr) observed_logits[uint64_t(row) * Mixes + t] = logits[t];
  }
  __syncthreads();

  }
  clock.mark(5);

  // pre and post, stored in F32 for the next sublayer and the expand.
  const float eps = __uint_as_float(M::kEpsBits);
  if (t < N) {
    pre[uint64_t(row) * N + t] =
        __fadd_rn(chop::sigmoid::sigmoid(fmaf(logits[t], scale[0], base[t])), eps);
    const float placed = chop::sigmoid::sigmoid(fmaf(logits[N + t], scale[1], base[N + t]));
    post[uint64_t(row) * N + t] = __fmul_rn(2.f, placed);
  }
  clock.mark(6);
  // Sinkhorn: mix_collapse's cells, sums and normalizations.
  const int i = t / N;
  const int j = t % N;
  const bool owner = t < Cells;
  float cell = owner ? fmaf(logits[2 * N + t], scale[2], base[2 * N + t]) : 0.f;
  if (owner) cells[t] = cell;
  __syncthreads();
  if (owner) {
    float max = cells[i * N];
    #pragma unroll
    for (int k = 1; k < N; ++k) max = fmaxf(max, cells[i * N + k]);
    cell = expf(__fadd_rn(cell, -max));
  }
  __syncthreads();
  if (owner) cells[t] = cell;
  __syncthreads();
  if (owner) {
    float sum = 0.f;
    #pragma unroll
    for (int k = 0; k < N; ++k) sum = __fadd_rn(sum, cells[i * N + k]);
    cell = __fadd_rn(__fdiv_rn(cell, sum), eps);
  }
  auto normalize = [&](bool rows) {
    __syncthreads();
    if (owner) cells[t] = cell;
    __syncthreads();
    if (owner) {
      float sum = 0.f;
      #pragma unroll
      for (int k = 0; k < N; ++k) sum = __fadd_rn(sum, cells[rows ? i * N + k : k * N + j]);
      cell = __fdiv_rn(cell, __fadd_rn(sum, eps));
    }
  };
  normalize(false);
  for (int iteration = 1; iteration < M::kIterations; ++iteration) {
    normalize(true);
    normalize(false);
  }
  if (owner) comb[uint64_t(row) * Cells + t] = cell;

  clock.mark(7);
  if constexpr (Kind != Mode::sinkhorn) {
  // The collapse with the given weights: per column an FMA chain from zero
  // in stream order.
  const uint64_t out = uint64_t(row) * H;
  for (int col = 2 * t; col < H; col += kColumns) {
    float lo = 0.f, hi = 0.f;
    #pragma unroll
    for (int s = 0; s < N; ++s) {
      const uint32_t pair = *reinterpret_cast<const uint32_t*>(streams + s * H + col);
      lo = fmaf(weights[s], bf_lo(pair), lo);
      hi = fmaf(weights[s], bf_hi(pair), hi);
    }
    *reinterpret_cast<uint32_t*>(collapsed + out + col) = pack(lo, hi);
  }
  }
  clock.mark(8);
}

template<class M, Mode Kind, bool Sampled>
__device__ void task(
    const uint16_t* x, const void* fn, const float* base, const float* scale,
    const float* pre_in, uint16_t* collapsed, float* pre, float* post, float* comb,
    float* partials, uint32_t* tickets, const uint32_t* live_rows,
    const float* seed_logits, float* observed_logits, float* loaded, uint64_t* stamps,
    uint32_t tile, uint32_t row, unsigned char* arena, Clock<Sampled>& clock) {
  if (row >= *live_rows) return;
  constexpr int Groups = M::kChains/kWarps;
  float* parts = partials+uint64_t(row)*M::kMixes*M::kChains;
  projection<M,Kind>(x,fn,parts,loaded,tile,row,arena);
  clock.mark(2);
  const bool last = arrive(tickets+row,Groups,reinterpret_cast<Arena<M>*>(arena)->last);
  clock.mark(3);
  constexpr bool HasFinish = Kind == Mode::full || Kind == Mode::resident || Kind == Mode::sinkhorn;
  if constexpr (HasFinish) {
    if (last) finish<M,Kind,Sampled>(x,fn,base,scale,pre_in,collapsed,pre,post,comb,
                                   seed_logits,observed_logits,parts,row,arena,clock);
  }
  clock.end(stamps,tile,row,Groups,Kind,last,last && HasFinish);
}
} // namespace chop::sinkhorn_hc::breakdown
