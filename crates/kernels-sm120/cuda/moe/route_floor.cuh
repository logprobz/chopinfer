#pragma once
#include "route_row_bias.cuh"
#include "../gemm/exact_f32_split.cuh"

namespace chop::moe::floor {
// The inherited policy keeps transforms, tie-breaking and serial top-k
// normalization. A split router CTA has an explicit row index.
template<int Experts>
struct RowScore : SqrtSoftplusRowBias<Experts> {
  int token;
  __device__ float rank(float value, int expert) const {
    const uint8_t selected = this->bias_select[token];
    if (selected > 1) __trap();
    const float correction = (selected ? this->alternate_bias : this->bias)[expert];
    if (!isfinite(correction)) __trap();
    this->own_scores[uint64_t(token) * Experts + expert] = value;
    const float key = __fadd_rn(value, correction);
    if (!isfinite(key)) __trap();
    return key;
  }
};

template<int Experts, int K>
__device__ void router_prepare(const float* input, const uint32_t* live,
    uint32_t rows, void* scratch, uint32_t* row_tickets) {
  exact_f32::split_v2::prepare<Experts,K>(input, live, rows, scratch);
  const int at = blockIdx.x * 128 + threadIdx.x;
  if (at < (rows + 7) / 8) row_tickets[at] = 0;
}

template<int Experts, int TopK, int K>
__device__ void router(const float* input, const uint16_t* weight, void* scratch,
    float* logits, const uint32_t* live, uint32_t rows, const uint32_t* weight_domain,
    uint32_t* row_tickets, const float* bias, const float* alternate_bias,
    const uint8_t* bias_select, float* scores, float scaling, int32_t* ids, float* weights) {
  exact_f32::split_v2::Workspace<Experts,K> work(scratch, rows);
  exact_f32::split_v2::compute<Experts,K,false>(input,weight,work.partial,logits,live,
      rows,weight_domain,work.domain,work.components,work.tickets);
  // Count all split CTAs after their own ordered reduction, including the
  // ones that did not win an output tile's ticket. No cross-CTA spin wait.
  __shared__ uint32_t last;
  __shared__ float row_logits[4][Experts];
  __syncthreads();
  if (threadIdx.x == 0) {
    uint32_t arrived;
    asm volatile("atom.add.acq_rel.gpu.u32 %0, [%1], 1;"
        : "=r"(arrived) : "l"(row_tickets + blockIdx.y) : "memory");
    last = arrived == gridDim.x * gridDim.z - 1;
  }
  __syncthreads();
  if (last) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int r = warp; r < 8; r += 4) {
      const int token = blockIdx.y * 8 + r;
      if (token >= min(*live, rows)) continue;
      for (int e = lane; e < Experts; e += 32)
        row_logits[warp][e] = __ldcg(logits + uint64_t(token) * Experts + e);
      __syncwarp();
      RowScore<Experts> score{{bias,alternate_bias,bias_select,scores,scaling},token};
      select_route_warp<Experts,TopK>(row_logits[warp],0,score,
          ids + token * TopK,weights + token * TopK);
      __syncwarp();
    }
  }
}

// Ordered route metadata and both expert queues are produced together.
// There is deliberately no early dependent-launch signal: consumers may
// prefetch immutable weights using this header before their own PDL wait.
template<int Experts, int TopK, int Capacity>
__device__ void metadata(const int32_t* ids, int32_t* counts, int32_t* offsets,
    int32_t* sorted_rows, int32_t* padded, const int32_t* live_tokens,
    int32_t* active, int32_t* table, uint32_t* gate_tickets) {
  static_assert(Experts % 32 == 0 && Capacity <= 64 && TopK <= 8);
  constexpr int Groups = (Capacity * TopK + 31) / 32;
  constexpr int Threads = Experts > Groups * 32 ? Experts : Groups * 32;
  __shared__ uint16_t group_start[Groups][Experts];
  __shared__ int32_t expert_count[Experts], expert_first[Experts], active_ids[Experts];
  __shared__ int active_count;
  if (blockDim.x != Threads || gridDim.x != 1) __trap();
  ptx::griddep_wait();
  const int live = *live_tokens;
  if (live < 0 || live > Capacity) __trap();
  const int routes = live * TopK, groups = (routes + 31) / 32;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int at = threadIdx.x; at < groups * Experts; at += Threads)
    (&group_start[0][0])[at] = 0;
  for (int at = threadIdx.x; at < Capacity; at += Threads) gate_tickets[at] = 0;
  __syncthreads();
  int expert = -1, before = 0;
  if (warp < groups) {
    const int route = warp * 32 + lane;
    expert = route < routes ? ids[route] : -1;
    if (expert >= Experts || expert < -1) __trap();
    const unsigned same = __match_any_sync(0xffffffffu, expert >= 0 ? expert : -1 - lane);
    before = __popc(same & ((1u << lane) - 1u));
    if (expert >= 0 && before == 0) group_start[warp][expert] = uint16_t(__popc(same));
  }
  __syncthreads();
  for (int e = threadIdx.x; e < Experts; e += Threads) {
    int count = 0;
    for (int g = 0; g < groups; ++g) {
      const int n = group_start[g][e];
      group_start[g][e] = uint16_t(count);
      count += n;
    }
    if (count > live) __trap();
    expert_count[e] = count;
    counts[e] = count;
  }
  __syncthreads();
  if (warp == 0) {
    int carry = 0, small = 0, wide = 0;
    for (int base = 0; base < Experts; base += 32) {
      const int e = base + lane, count = expert_count[e];
      const int valid = count > 0, is_small = valid && count <= 8, is_wide = count > 8;
      int a = valid, s = is_small, w = is_wide;
#pragma unroll
      for (int d = 1; d < 32; d *= 2) {
        const int ua = __shfl_up_sync(0xffffffffu, a, d);
        const int us = __shfl_up_sync(0xffffffffu, s, d);
        const int uw = __shfl_up_sync(0xffffffffu, w, d);
        if (lane >= d) { a += ua; s += us; w += uw; }
      }
      expert_first[e] = offsets[e] = (carry + a - valid) * 64;
      if (valid) active[1 + carry + a - 1] = active_ids[carry + a - 1] = e;
      if (is_small) {
        const int at = 2 + 2 * (small + s - 1);
        table[at] = e; table[at + 1] = count;
      }
      if (is_wide) {
        const int at = 2 + 2 * Experts + 2 * (wide + w - 1);
        table[at] = e; table[at + 1] = count;
      }
      carry += __shfl_sync(0xffffffffu, a, 31);
      small += __shfl_sync(0xffffffffu, s, 31);
      wide += __shfl_sync(0xffffffffu, w, 31);
    }
    if (lane == 0) {
      offsets[Experts] = *padded = carry * 64;
      active[0] = active_count = carry;
      table[0] = small; table[1] = wide;
    }
  }
  __syncthreads();
  if (expert >= 0)
    sorted_rows[expert_first[expert] + group_start[warp][expert] + before] = warp * 32 + lane;
  for (int at = threadIdx.x; at < active_count * 64; at += Threads) {
    const int e = active_ids[at / 64], row = at % 64;
    if (row >= expert_count[e]) sorted_rows[expert_first[e] + row] = -1;
  }
}
} // namespace chop::moe::floor
