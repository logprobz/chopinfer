#pragma once
#include "router_finish_r2.cuh"

// The wide cached producer receives independent domain and partial spans.
// Keep the r2 chain order, parallel scores and elected metadata publication.
namespace chop::moe::router_finish64 {

template<int K>
__device__ void prepare(const float* input, const uint32_t* live,
    uint32_t rows, uint32_t* row_domain, uint32_t* arrived) {
  ptx::griddep_wait();
  if (rows < 33 || rows > 64 || gridDim.x != rows || blockDim.x != 128) __trap();
  if (blockIdx.x == 0 && threadIdx.x == 0) *arrived = 0;
  exact_f32::classify_rows<K>(input, live, rows, row_domain);
}

template<int Experts, int TopK, int K>
__device__ void finish(const float* partials, float* logits, const int32_t* live_tokens,
    uint32_t rows, const float* bias, const float* alternate_bias,
    const uint8_t* bias_select, float* scores, int32_t* ids, float* weights,
    float scaling, int32_t* counts, int32_t* offsets, int32_t* sorted,
    int32_t* padded, int32_t* active, void* scratch, uint32_t* gate_tickets,
    uint32_t* arrived) {
  constexpr int Threads = Experts > 64 * TopK ? Experts : 64 * TopK;
  constexpr int Chains = (K + 127) / 128;
  static_assert(Experts % 32 == 0 && TopK <= 8);
  if (rows < 33 || rows > 64 || gridDim.x != rows || blockDim.x != Threads) __trap();
  ptx::griddep_wait();
  const int live = *live_tokens, token = blockIdx.x;
  if (live < 0 || live > int(rows)) __trap();
  __shared__ float row_logits[Experts], row_unbiased[Experts], row_rank[Experts];
  __shared__ uint32_t last;
  if (token < live) {
    for (int expert = threadIdx.x; expert < Experts; expert += Threads) {
      float value[Chains];
#pragma unroll
      for (int chain = 0; chain < Chains; ++chain)
        value[chain] = partials[(uint64_t(chain) * rows + token) * Experts + expert];
      float sum = 0.f;
#pragma unroll
      for (int chain = 0; chain < Chains; ++chain) sum = __fadd_rn(sum, value[chain]);
      row_logits[expert] = sum;
      logits[uint64_t(token) * Experts + expert] = sum;
    }
  }
  __syncthreads();
  floor::RowScore<Experts> score{{bias,alternate_bias,bias_select,scores,scaling},token};
  if (token < live && threadIdx.x < Experts) {
    const float value = score.transform(row_logits[threadIdx.x]);
    row_unbiased[threadIdx.x] = value;
    row_rank[threadIdx.x] = score.rank(value, threadIdx.x);
  }
  __syncthreads();
  if (token < live && threadIdx.x < 32)
    router_finish_r2::select_scored<Experts,TopK>(row_unbiased,row_rank,0,score,
        ids + token * TopK, weights + token * TopK);
  __syncthreads();
  if (threadIdx.x == 0) {
    uint32_t previous;
    asm volatile("atom.add.acq_rel.gpu.u32 %0, [%1], 1;"
        : "=r"(previous) : "l"(arrived) : "memory");
    last = previous == rows - 1;
  }
  __syncthreads();
  if (last)
    router_finish_r2::metadata_parallel<Experts,TopK,64>(ids,counts,offsets,sorted,
        padded,live_tokens,active,static_cast<int32_t*>(scratch),gate_tickets);
  // Consumers may inspect queue headers before their wait. Publish only
  // after the elected CTA finishes metadata, including an empty live set.
}
} // namespace chop::moe::router_finish64
