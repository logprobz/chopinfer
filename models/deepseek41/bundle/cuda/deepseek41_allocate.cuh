#pragma once
#include <stdint.h>
// Verify allocation (deepseek41.verify-allocation.f32-rn.v1): one block
// computes every item's verified drafts and the reserved/computed row maps,
// as model_deepseek41_contract::own_ops::allocation::allocate_ranked does,
// then compacts the target's per-row inputs in place.
// Every float operation is an explicitly rounded intrinsic, so no
// contraction changes a bit against runtime_bundle_sdk::sigmoid_v1.
namespace chop::deepseek41 {
constexpr uint32_t kAllocateBlock = 5;
constexpr uint32_t kAllocateItems = 32;
constexpr uint32_t kAllocateItemWords = 4;
constexpr uint32_t kAllocateNone = 0xffffffffu;
constexpr uint32_t kAllocateRows = kAllocateItems * (kAllocateBlock + 1);
constexpr uint32_t kAllocateArrays = 5;

__device__ __forceinline__ float allocate_sigmoid(float z) {
  if (z != z) return 0.0f;
  z = fminf(fmaxf(z, -30.0f), 30.0f);
  const float x = -z;
  const float n = floorf(__fadd_rn(__fmul_rn(x, __int_as_float(0x3fb8aa3b)), 0.5f));
  const float r = __fsub_rn(__fsub_rn(x, __fmul_rn(n, __int_as_float(0x3f318000))),
                            __fmul_rn(n, __int_as_float(0xb95e8083)));
  float t = __int_as_float(0x3ab60b61);
  t = __fadd_rn(__fmul_rn(t, r), __int_as_float(0x3c088889));
  t = __fadd_rn(__fmul_rn(t, r), __int_as_float(0x3d2aaaab));
  t = __fadd_rn(__fmul_rn(t, r), __int_as_float(0x3e2aaaab));
  t = __fadd_rn(__fmul_rn(t, r), __int_as_float(0x3f000000));
  t = __fadd_rn(__fmul_rn(__fadd_rn(__fmul_rn(t, r), 1.0f), r), 1.0f);
  const float power = __int_as_float((int(n) + 127) << 23);
  return __fdiv_rn(1.0f, __fadd_rn(1.0f, __fmul_rn(t, power)));
}

// items [count, 4]: reserved drafts, kind, confidence row, first reserved
// row. scores [score_rows, 5]. rule: width, floor, then (scale, offset)
// bits per position for greedy and then sampled items.
// token_ids, positions, token_batch, window_rows and index_visible hold
// the reserved rows and are compacted in place; q_start takes each item's
// computed first row; text (text_rows bytes, zero when absent) is
// compacted like the 4-byte rows.
__device__ __forceinline__ void verify_allocate(const uint32_t* items, const float* scores,
    const uint32_t* rule, const uint32_t* live_items, uint32_t* status, uint32_t* verified,
    uint32_t* starts, uint32_t* rows, uint32_t* expand, uint32_t* token_ids,
    uint32_t* positions, uint32_t* token_batch, uint32_t* window_rows, uint32_t* index_visible,
    uint32_t* q_start, uint8_t* text, uint32_t item_capacity, uint32_t score_rows,
    uint32_t reserved_capacity, uint32_t width_capacity, uint32_t text_rows) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  __shared__ float survival[kAllocateItems * kAllocateBlock];
  __shared__ uint32_t candidate[kAllocateItems * kAllocateBlock];
  __shared__ uint32_t counts[kAllocateItems];
  __shared__ uint32_t first[kAllocateItems];
  __shared__ uint32_t reserved_values[kAllocateArrays][kAllocateRows];
  __shared__ uint8_t reserved_text[kAllocateRows];
  __shared__ uint32_t bad, count, floor_drafts, left, reserved_total;
  const uint32_t tid = threadIdx.x;
  if (tid == 0) {
    const uint32_t n = *live_items;
    uint32_t next = 0, drafting = 0;
    bool invalid = n == 0 || n > item_capacity || n > kAllocateItems;
    for (uint32_t i = 0; !invalid && i < n; ++i) {
      const uint32_t* item = items + i * kAllocateItemWords;
      const uint32_t reserved = item[0];
      invalid = reserved > kAllocateBlock || item[1] > 1 || item[3] != next
          || (reserved == 0 ? item[2] != kAllocateNone
                            : (item[2] != drafting || item[2] >= score_rows));
      drafting += reserved > 0;
      next += 1 + reserved;
    }
    const uint32_t width = rule[0];
    invalid = invalid || width < n || width > next || next > reserved_capacity
        || width > width_capacity || (text_rows != 0 && text_rows < next);
    uint32_t f = 0, used = 0;
    if (!invalid) {
      // The largest floor up to the rule's at which every item fits; a
      // floor past the block reserves no more, and at zero every item fits.
      for (f = min(rule[1], kAllocateBlock) + 1; f-- > 0;) {
        used = 0;
        for (uint32_t i = 0; i < n; ++i) used += 1 + min(f, items[i * kAllocateItemWords]);
        if (used <= width) break;
      }
    }
    bad = invalid;
    count = n;
    reserved_total = next;
    floor_drafts = f;
    left = invalid ? 0 : width - used;
  }
  __syncthreads();
  if (bad) {
    if (tid == 0) status[0] = 1;
    return;
  }
  const uint32_t n = count, f = floor_drafts;
  for (uint32_t t = tid; t < n * kAllocateBlock; t += blockDim.x) {
    const uint32_t i = t / kAllocateBlock, j = t % kAllocateBlock;
    const uint32_t* item = items + i * kAllocateItemWords;
    float s = 0.0f;
    if (j < item[0]) {
      const uint32_t* pairs = rule + 2 + item[1] * 2 * kAllocateBlock;
      const float* row = scores + item[2] * kAllocateBlock;
      s = 1.0f;
      for (uint32_t k = 0; k <= j; ++k) {
        const float z = __fadd_rn(__fmul_rn(__uint_as_float(pairs[2 * k]), row[k]),
                                  __uint_as_float(pairs[2 * k + 1]));
        s = __fmul_rn(s, allocate_sigmoid(z));
      }
    }
    survival[t] = s;
    candidate[t] = j < item[0] && j >= f;
  }
  for (uint32_t i = tid; i < n; i += blockDim.x) counts[i] = min(f, items[i * kAllocateItemWords]);
  uint32_t* const arrays[kAllocateArrays] = {token_ids, positions, token_batch, window_rows,
                                             index_visible};
  for (uint32_t r = tid; r < reserved_total; r += blockDim.x) {
    for (uint32_t a = 0; a < kAllocateArrays; ++a) reserved_values[a][r] = arrays[a][r];
    if (text_rows != 0) reserved_text[r] = text[r];
  }
  __syncthreads();
  for (uint32_t t = tid; t < n * kAllocateBlock; t += blockDim.x) {
    if (!candidate[t]) continue;
    const float s = survival[t];
    uint32_t rank = 0;
    for (uint32_t u = 0; u < n * kAllocateBlock; ++u) {
      rank += candidate[u] && (survival[u] > s || (survival[u] == s && u < t));
    }
    if (rank < left) atomicAdd(&counts[t / kAllocateBlock], 1u);
  }
  __syncthreads();
  if (tid == 0) {
    uint32_t start = 0;
    for (uint32_t i = 0; i < n; ++i) {
      first[i] = start;
      start += 1 + counts[i];
    }
    status[0] = 0;
  }
  __syncthreads();
  for (uint32_t i = tid; i < n; i += blockDim.x) {
    const uint32_t* item = items + i * kAllocateItemWords;
    const uint32_t k = counts[i], start = first[i], row = item[3];
    verified[i] = k;
    starts[i] = start;
    q_start[i] = start;
    for (uint32_t o = 0; o <= k; ++o) {
      rows[start + o] = row + o;
      for (uint32_t a = 0; a < kAllocateArrays; ++a) arrays[a][start + o] = reserved_values[a][row + o];
      if (text_rows != 0) text[start + o] = reserved_text[row + o];
    }
    for (uint32_t o = 0; o <= item[0]; ++o) expand[row + o] = start + min(o, k);
  }
}
} // namespace chop::deepseek41
