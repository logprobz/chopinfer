#pragma once
// The shared sharded argmax (`kernel_core::contract::RowQuery`'s
// `ShardedArgmaxPartial` and `ShardedArgmaxFinal`, `row::argmax`). One CTA
// per row of the launch; rows at or past the device live count return.
//
// A logit ranks by `rank`: IEEE order with -0 as +0, every NaN (any sign or
// payload) zero, below -inf. A candidate's key holds its rank in the high
// word and its inverted global id in the low, so the larger key wins and
// equal ranks go to the lower id. Keys are a total order over the
// candidates, so every reduction order picks the same one.
#include "row.cuh"

namespace chop::row::argmax {

// Reads bits only, so no flag or compiler option changes it.
__device__ __forceinline__ uint32_t rank(float value) {
  uint32_t bits = __float_as_uint(value);
  if ((bits & 0x7fffffffu) > 0x7f800000u) return 0;
  if (bits == 0x80000000u) bits = 0;
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ uint64_t key(float value, uint32_t id) {
  return (uint64_t(rank(value)) << 32) | (0xffffffffu - id);
}

__device__ __forceinline__ float logit(const float* row, uint32_t col) {
  return row[col];
}
// BF16 widens by its bits, so a NaN keeps its sign and payload.
__device__ __forceinline__ float logit(const uint16_t* row, uint32_t col) {
  return __uint_as_float(uint32_t(row[col]) << 16);
}

// The largest of every thread's `best`, in thread 0. Zero is below every
// candidate's key (ids are below 2^31), so a thread without columns
// contributes nothing.
__device__ __forceinline__ uint64_t max_cta(uint64_t best) {
  __shared__ uint64_t warps[32];
  for (int offset = 16; offset > 0; offset >>= 1) {
    const uint64_t other = __shfl_xor_sync(0xffffffffu, best, offset);
    best = other > best ? other : best;
  }
  if ((threadIdx.x & 31) == 0) warps[threadIdx.x >> 5] = best;
  __syncthreads();
  if (threadIdx.x == 0)
    for (uint32_t w = 1; w < blockDim.x / 32; ++w)
      best = warps[w] > best ? warps[w] : best;
  return best;
}

// Each row's pair over one rank's logits `[rows, vocab]` (F32, or BF16
// widened by its bits): the winning logit's F32 bits, reread at its column,
// and its global id `vocab_offset + column`. The row reads
// `max(min(*live_width, vocab), 1)` columns, so every id is in the shard.
template <typename Logit>
__device__ void partial(const Logit* logits, float* scores, int32_t* ids,
                        const uint32_t* live_rows, const uint32_t* live_width,
                        uint32_t vocab, uint32_t vocab_offset) {
  begin();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  const uint32_t width = max(min(*live_width, vocab), 1u);
  const Logit* in = logits + uint64_t(row) * vocab;
  uint64_t best = 0;
  for (uint32_t col = threadIdx.x; col < width; col += blockDim.x) {
    const uint64_t candidate = key(logit(in, col), vocab_offset + col);
    best = candidate > best ? candidate : best;
  }
  best = max_cta(best);
  if (threadIdx.x != 0) return;
  const uint32_t id = 0xffffffffu - uint32_t(best);
  scores[row] = logit(in, id - vocab_offset);
  ids[row] = int32_t(id);
}

// Each row's global argmax over `ranks` pairs, rank `r`'s at
// `row * row_stride + r * rank_stride` (row-major: `ranks` and 1;
// rank-major: 1 and the gathered stride). The id goes to `output[row]`
// when `output` is not null, and to `out_a[rows_a[row]]` and
// `out_b[rows_b[row]]` when those are not null and the index is below
// `len_a` or `len_b` (a negative index, read unsigned, is past it).
__device__ void final_pairs(const float* scores, const int32_t* ids,
                            int32_t* output, const int32_t* rows_a,
                            int32_t* out_a, const int32_t* rows_b,
                            int32_t* out_b, const uint32_t* live_rows,
                            uint32_t ranks, uint32_t row_stride,
                            uint32_t rank_stride, uint32_t len_a,
                            uint32_t len_b) {
  begin();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows || threadIdx.x != 0) return;
  uint64_t best = 0;
  for (uint32_t r = 0; r < ranks; ++r) {
    const uint64_t at = uint64_t(row) * row_stride + uint64_t(r) * rank_stride;
    const uint64_t candidate = key(scores[at], uint32_t(ids[at]));
    best = candidate > best ? candidate : best;
  }
  const int32_t id = int32_t(0xffffffffu - uint32_t(best));
  if (output) output[row] = id;
  if (out_a && uint32_t(rows_a[row]) < len_a) out_a[rows_a[row]] = id;
  if (out_b && uint32_t(rows_b[row]) < len_b) out_b[rows_b[row]] = id;
}

} // namespace chop::row::argmax
