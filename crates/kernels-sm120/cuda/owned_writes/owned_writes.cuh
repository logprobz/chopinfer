#pragma once

// Owner-routed cache writes under context parallelism (README.md): the
// gated Q/K norm, M-RoPE and KV write, and the block-sparse indexer's key
// write, with every K/V row and block key stored only by the group rank
// that owns its page, at that rank's local page. The arithmetic is the
// served kernels' own (attention/gated.cuh `qk_norm_rope_kv`,
// attention/sparse_indexer.cuh `indexer_key_write`), so the bits are theirs.

#include "attention/gated.cuh"
#include "attention/sparse_indexer.cuh"

namespace chop::owned_writes {

// `gated::qk_norm_rope_kv` over group-local cache positions:
// `local_positions[row]` is the row's position on this rank's pages (a
// `LocalizeKeys` list of width 1), -1 when another rank owns its page.
// Every row's Q heads are written; the K/V warps of a row the rank does not
// own return before the shared body, so it stores no K or V. An owned row
// stores K and V at `block_tables[batch * max_pages + l / Page]`, slot
// `l % Page`, through the rank's local table.
template<int QHeads, int KvHeads, int Qk, int V, int Rotary, int SectionH, int SectionW,
         int Page, bool Fp8>
__device__ void owned_qk_norm_rope_kv(
    const uint16_t* qkv, const uint16_t* q_norm, const uint16_t* k_norm,
    const int32_t* rope_positions, const float* cos_sin, const int32_t* row_batch,
    const int32_t* local_positions, const uint32_t* live_rows, const int32_t* block_tables,
    void* k_cache, void* v_cache, uint16_t* q, uint32_t max_pages,
    uint64_t k_page_stride_bytes, uint64_t v_page_stride_bytes, uint32_t eps_bits,
    uint32_t k_scale_bits, uint32_t v_scale_bits) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  constexpr int q_warps = (QHeads + 7) / 8;
  if (int(threadIdx.x / 32) >= q_warps && local_positions[row] < 0) return;
  attention::gated::qk_norm_rope_kv<QHeads, KvHeads, Qk, V, Rotary, SectionH, SectionW, Page,
                                    Fp8>(
      qkv, q_norm, k_norm, rope_positions, cos_sin, row_batch, local_positions, live_rows,
      block_tables, k_cache, v_cache, q, max_pages, k_page_stride_bytes, v_page_stride_bytes,
      eps_bits, k_scale_bits, v_scale_bits);
}

// `sparse::indexer_key_write` with owner-routed block keys: block b lies in
// physical page j = b / page_blocks, which group rank `j % cp_size` holds at
// its local page `block_tables[batch * max_pages + j / cp_size]`, row
// `b % page_blocks`. That rank pools and stores the block; the others skip
// it. The raw-key ring and the anchor ring are written as the served kernel
// writes them, on every rank. The body is the served kernel's, line for
// line, with only the block-key row lookup replaced.
template<class I>
__device__ void owned_index_key_write(
    const uint16_t* projection, const uint16_t* weight, const int32_t* rope_positions,
    const float* cos_sin, const int32_t* row_batch, const int32_t* positions,
    const uint32_t* live_rows, const int32_t* slot_ids, uint16_t* ring_keys,
    int32_t* ring_positions, const int32_t* anchor_slots, const int32_t* anchor_ends,
    uint16_t* anchor_keys, int32_t* anchor_positions, const int32_t* block_tables,
    uint8_t* block_keys, uint32_t max_pages, uint64_t block_page_stride_bytes,
    uint32_t eps_bits, uint32_t cp_rank, uint32_t cp_size) {
  using attention::sparse::load4;
  using attention::sparse::write_block;
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  if (cp_size == 0 || cp_rank >= cp_size) __trap();
  const int warps = blockDim.x / 32;
  const int row = blockIdx.x * warps + threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int live = int(*live_rows);
  if (row >= live) return;
  const float eps = __uint_as_float(eps_bits);
  const int batch = row_batch[row];
  const int position = positions[row];
  // The rank's row of block b, or null when another rank owns its page.
  auto owned_row = [&](int b) -> uint8_t* {
    const uint32_t physical = uint32_t(b) / I::page_blocks;
    if (physical % cp_size != cp_rank) return nullptr;
    const uint64_t page =
        uint32_t(block_tables[uint64_t(batch) * max_pages + physical / cp_size]);
    return block_keys + page * block_page_stride_bytes +
           uint64_t(uint32_t(b) % I::page_blocks) * I::dim * 2;
  };
  auto raw_key = [&](int r) {
    return projection + uint64_t(r) * I::width + I::heads * I::dim + 4 * lane;
  };
  auto add_key = [&](double (&sum)[4], const uint16_t* source) {
    float x[4];
    load4(x, source);
#pragma unroll
    for (int i = 0; i < 4; ++i) sum[i] += x[i];
  };
  if (position % I::block == I::block - 1) {
    const int first = row - (I::block - 1);
    uint8_t* destination = owned_row(position / I::block);
    if (destination && first >= 0 && row_batch[first] == batch &&
        positions[first] == position - (I::block - 1)) {
      double sum[4] = {0.0, 0.0, 0.0, 0.0};
      for (int r = first; r <= row; ++r) add_key(sum, raw_key(r));
      write_block<I>(sum, weight, rope_positions + 3 * uint64_t(first), cos_sin, eps,
                     destination);
    }
  }
  if (row > 0 && row_batch[row - 1] == batch) return;

  // The leader: the sequence's rows are [row, end).
  int low = row, high = live;
  while (high - low > 1) {
    const int middle = (low + high) / 2;
    if (row_batch[middle] == batch) low = middle; else high = middle;
  }
  const int end = high;
  const uint64_t slot = uint32_t(slot_ids[batch]);
  uint16_t* ring = ring_keys + slot * I::ring * I::dim;
  int32_t* ring_rope = ring_positions + slot * I::ring * 3;
  const int start = position - position % I::block;
  const int completing = start + I::block - 1;
  uint8_t* straddling = owned_row(start / I::block);
  if (straddling && start < position && completing - position < end - row) {
    double sum[4] = {0.0, 0.0, 0.0, 0.0};
    for (int p = start; p <= completing; ++p)
      add_key(sum, p < position ? ring + (p % I::ring) * I::dim + 4 * lane
                                : raw_key(row + (p - position)));
    write_block<I>(sum, weight, ring_rope + (start % I::ring) * 3, cos_sin, eps, straddling);
  }
  __syncwarp();
  for (int r = max(row, end - I::ring); r < end; ++r) {
    const int slot_row = (position + (r - row)) % I::ring;
    *reinterpret_cast<uint2*>(ring + slot_row * I::dim + 4 * lane) =
        *reinterpret_cast<const uint2*>(raw_key(r));
    if (lane < 3) ring_rope[slot_row * 3 + lane] = rope_positions[3 * uint64_t(r) + lane];
  }
  const int32_t anchor = anchor_slots[batch];
  if (anchor < 0) return;
  const int anchor_end = anchor_ends[batch];
  uint16_t* kept = anchor_keys + uint64_t(uint32_t(anchor)) * I::ring * I::dim;
  int32_t* kept_rope = anchor_positions + uint64_t(uint32_t(anchor)) * I::ring * 3;
  for (int p = max(position, anchor_end - I::ring); p < anchor_end && p - position < end - row;
       ++p) {
    const int r = row + (p - position);
    const int slot_row = p % I::ring;
    *reinterpret_cast<uint2*>(kept + slot_row * I::dim + 4 * lane) =
        *reinterpret_cast<const uint2*>(raw_key(r));
    if (lane < 3) kept_rope[slot_row * 3 + lane] = rope_positions[3 * uint64_t(r) + lane];
  }
}

}  // namespace chop::owned_writes
