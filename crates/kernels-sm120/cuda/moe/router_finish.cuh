#pragma once
#include "route_floor.cuh"

// Keep the cached exact-F32 partial producer. A row CTA folds its chains
// in order and selects that row's routes. The final release/acquire ticket
// elects the metadata writer without a cross-CTA spin or float atomics.
namespace chop::moe::router_finish {

template<int K>
__device__ void prepare(const float* input, const uint32_t* live,
    uint32_t rows, float* workspace, uint32_t* arrived) {
  ptx::griddep_wait();
  if (rows == 0 || rows > 32 || gridDim.x != rows || blockDim.x != 128) __trap();
  if (blockIdx.x == 0 && threadIdx.x == 0) *arrived = 0;
  exact_f32::classify_rows<K>(input, live, rows, reinterpret_cast<uint32_t*>(workspace));
}

// The standalone metadata engine's integer scan and stable scatter, in
// the elected row CTA. Bypass L1 for IDs published by other CTAs. The
// acquire and following block barrier precede every metadata read.
template<int Experts, int TopK, int Capacity>
__device__ void metadata_after_acquire(const int32_t* ids, int32_t* counts, int32_t* offsets,
    int32_t* sorted_rows, int32_t* padded, const int32_t* live_tokens,
    int32_t* active, int32_t* table, uint32_t* gate_tickets) {
  static_assert(Experts % 32 == 0 && Capacity <= 64 && TopK <= 8);
  constexpr int Groups = (Capacity * TopK + 31) / 32;
  constexpr int Threads = Experts > Groups * 32 ? Experts : Groups * 32;
  __shared__ uint16_t group_start[Groups][Experts];
  __shared__ int32_t expert_count[Experts], expert_first[Experts], active_ids[Experts];
  __shared__ int active_count;
  if (blockDim.x != Threads) __trap();
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
    expert = route < routes ? __ldcg(ids + route) : -1;
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

template<int Experts, int TopK, int K>
__device__ void finish(const float* workspace, float* logits, const int32_t* live_tokens,
    uint32_t rows, const float* bias, const float* alternate_bias,
    const uint8_t* bias_select, float* scores, int32_t* ids, float* weights,
    float scaling, int32_t* counts, int32_t* offsets, int32_t* sorted,
    int32_t* padded, int32_t* active, void* scratch, uint32_t* gate_tickets,
    uint32_t* arrived) {
  constexpr int Threads = Experts > 64 * TopK ? Experts : 64 * TopK;
  constexpr int Chains = (K + 127) / 128;
  static_assert(Experts % 32 == 0 && TopK <= 8);
  if (rows == 0 || rows > 32 || gridDim.x != rows || blockDim.x != Threads) __trap();
  ptx::griddep_wait();
  const int live = *live_tokens, token = blockIdx.x;
  if (live < 0 || live > int(rows)) __trap();
  __shared__ float row_logits[Experts];
  __shared__ uint32_t last;
  const float* partial = workspace + 32;
  if (token < live) {
    for (int expert = threadIdx.x; expert < Experts; expert += Threads) {
      float sum = 0.f;
#pragma unroll 1
      for (int base = 0; base < Chains; base += 4) {
        float value[4];
#pragma unroll
        for (int b = 0; b < 4; ++b)
          value[b] = base + b < Chains
              ? partial[(uint64_t(base+b)*rows+token)*Experts+expert] : 0.f;
#pragma unroll
        for (int b = 0; b < 4; ++b)
          if (base + b < Chains) sum = __fadd_rn(sum, value[b]);
      }
      row_logits[expert] = sum;
      logits[uint64_t(token)*Experts+expert] = sum;
    }
  }
  __syncthreads();
  if (token < live && threadIdx.x < 32) {
    floor::RowScore<Experts> score{{bias,alternate_bias,bias_select,scores,scaling},token};
    select_route_warp<Experts,TopK>(row_logits,0,score,
        ids + token * TopK, weights + token * TopK);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    uint32_t previous;
    asm volatile("atom.add.acq_rel.gpu.u32 %0, [%1], 1;"
        : "=r"(previous) : "l"(arrived) : "memory");
    last = previous == rows - 1;
  }
  __syncthreads();
  if (last)
    metadata_after_acquire<Experts,TopK,64>(ids,counts,offsets,sorted,padded,
        live_tokens,active,static_cast<int32_t*>(scratch),gate_tickets);
  // No early launch signal: consumers may read the completed queue header
  // before their PDL wait. Empty-live graphs still publish an empty header.
}
} // namespace chop::moe::router_finish
