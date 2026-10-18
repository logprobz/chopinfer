#pragma once

#include "gated_pooling/pool.cuh"
#include "gated_pooling/key_write.cuh"

namespace chop::pooling_write_tests {

// Run the real prefill, but discard the tail carried out of its four-row
// prefix. The six-row whole launch is unchanged, so the split-inside-pair
// acceptance check must observe this fault in the next launch's latent.
template<int Width>
__device__ void drop_split_tail(const float* key, const float* gate,
    const int32_t* positions, const int32_t* row_batch,
    const uint32_t* live_rows, const int32_t* slot_ids,
    const int32_t* checkpoint_ids, float* state, float* pooled,
    int32_t* output_positions, uint32_t items, uint32_t entries) {
  gated_pooling::pool2<Width,0>(key,gate,positions,row_batch,live_rows,
      slot_ids,checkpoint_ids,state,pooled,output_positions,items,entries);
  __syncthreads();
  const uint32_t row = blockIdx.x, live = *live_rows;
  if (live != 4 || row >= live || positions[row] < 0) return;
  const int32_t item = row_batch[row];
  if (item < 0 || uint32_t(item) >= items) return;
  for (uint32_t previous = 0; previous < row; ++previous)
    if (row_batch[previous] == item && positions[previous] >= 0) return;
  uint32_t last = row;
  for (uint32_t next = row + 1; next < live && row_batch[next] == item; ++next)
    if (positions[next] >= 0) last = next;
  if (positions[last] & 1) return;
  const int32_t slot = slot_ids[item];
  if (slot < 0 || uint32_t(slot) >= entries) return;
  // Clear the open tail to finite values. The same write map is satisfied;
  // the next split launch must expose the missing projected key and score.
  for (uint32_t d = threadIdx.x; d < 2 * Width; d += blockDim.x)
    state[uint64_t(slot) * 2 * Width + d] = 0.f;
}

// Called after the real writer in its owning CTA, after a block barrier.
// Valid rows have distinct cache slots, as in the serving contract.
template<int Width, int Ratio, int Page>
__device__ void swap_nibbles(uint8_t* codes, const int32_t* positions,
    const int32_t* row_batch, const uint32_t* live_rows,
    const int32_t* tables, uint32_t items, uint32_t pages,
    uint32_t max_pages, uint64_t stride, uint32_t table_rows) {
  __syncthreads();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  const int32_t position = positions[row], item = row_batch[row];
  if (position < 0 || item < 0 || uint32_t(item) >= items) return;
  const uint32_t logical = uint32_t(position) / Page;
  if (logical >= max_pages || uint64_t(position) * Ratio >= table_rows) return;
  const int32_t page = tables[uint64_t(item) * max_pages + logical];
  if (page < 0 || uint32_t(page) >= pages) return;
  uint8_t* target = codes + uint64_t(page) * stride + uint64_t(position % Page) * (Width / 2);
  for (uint32_t pair = threadIdx.x; pair < Width / 2; pair += blockDim.x) {
    const uint8_t value = target[pair];
    target[pair] = uint8_t((value << 4) | (value >> 4));
  }
}

}  // namespace chop::pooling_write_tests
