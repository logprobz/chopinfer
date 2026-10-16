#pragma once

// SM120 attention engine: traits, KV cache addressing, and the prefill,
// decode and merge entries. See README.md.

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace chop::attention {

template<int Qk, int V, int QHeads, int KvHeads, int Page, bool Ring,
         bool Fp8, bool Sink, bool Softcap>
struct Traits {
  static_assert(Qk == 64 || Qk == 128 || Qk == 192 || Qk == 256);
  static_assert(V == 64 || V == 128 || V == 192 || V == 256);
  static_assert(QHeads > 0 && KvHeads > 0 && QHeads % KvHeads == 0);
  static_assert(QHeads / KvHeads <= 16);
  static_assert(Page == 16 || Page == 32 || Page == 64 ||
                (Ring && (Page == 128 || Page == 2048)));
  static constexpr int qk = Qk, v = V, q_heads = QHeads, kv_heads = KvHeads;
  static constexpr int page = Page, group = QHeads / KvHeads;
  // Decode's grouped heads are MMA n8 columns; a group below eight is padded
  // to one tile of columns that no output reads.
  static constexpr int group_pad = group < 8 ? 8 : group;
  static constexpr bool ring = Ring, fp8 = Fp8, sink = Sink, softcap = Softcap;
};

template<class T>
__device__ __forceinline__ uint64_t cache_token(
    const int32_t* table, const int32_t* slots, int batch, int position,
    int kv_head, int max_pages) {
  if constexpr (T::ring) {
    return ((uint64_t(slots[batch]) * T::page + (position % T::page)) *
            T::kv_heads + kv_head);
  } else {
    const int32_t physical = table[uint64_t(batch) * max_pages + position / T::page];
    return ((uint64_t(physical) * T::page + position % T::page) *
            T::kv_heads + kv_head);
  }
}

template<class T>
__device__ __forceinline__ uint64_t cache_byte_offset(
    const int32_t* table, const int32_t* slots, int batch, int position,
    int kv_head, int max_pages, int dim, int bytes_per,
    uint64_t page_stride_bytes) {
  if constexpr (T::ring) {
    return cache_token<T>(table, slots, batch, position, kv_head, max_pages) *
        uint64_t(dim) * bytes_per;
  } else {
    const uint64_t page = uint32_t(
        table[uint64_t(batch) * max_pages + position / T::page]);
    const uint64_t within = uint64_t(position % T::page) * T::kv_heads + kv_head;
    return page * page_stride_bytes + within * dim * bytes_per;
  }
}

} // namespace chop::attention

#include "prefill.cuh"
#include "decode.cuh"
#include "decode_rows.cuh"
#include "verify.cuh"

namespace chop::attention {

// Merge entries share one parameter list; each phase ignores the arrays it
// does not use. Entries without a `split_units` parameter (prefill, and the
// windowed decode merge, which only waits) pass 0.
template<class T, bool Decode>
__device__ void merge(
    const float* workspace, const __nv_bfloat16* sink,
    const int32_t* seq_lens, const int32_t* q_start, const int32_t* q_len,
    const int32_t* prefix_len, const int32_t* live_batch,
    __nv_bfloat16* out, uint32_t split_capacity, uint32_t split_target,
    uint32_t split_units, uint32_t window_left) {
  ptx::griddep_launch_dependents();
  if constexpr (Decode) {
    (void)q_start;
    (void)q_len;
    (void)prefix_len;
    decode_merge<T, 512>(workspace, sink, seq_lens, live_batch, out, split_capacity,
                         split_target, split_units, window_left);
  } else {
    (void)seq_lens;
    (void)split_units;
    ptx::griddep_wait();
    prefill_merge<T, 256>(workspace, sink, q_start, q_len, prefix_len, live_batch,
                          out, split_capacity, split_target, window_left);
  }
}

// The exact verify rows merge entry shares the merge parameter list.
template<class T>
__device__ void merge_rows(
    const float* workspace, const __nv_bfloat16* sink,
    const int32_t* seq_lens, const int32_t* q_start, const int32_t* q_len,
    const int32_t* prefix_len, const int32_t* live_batch,
    __nv_bfloat16* out, uint32_t split_capacity, uint32_t split_target,
    uint32_t split_units, uint32_t window_left) {
  (void)seq_lens;
  ptx::griddep_launch_dependents();
  decode_merge_rows<T, 512>(workspace, sink, q_start, q_len, prefix_len, live_batch, out,
                            split_capacity, split_target, split_units, window_left);
}

// The verify merge entry shares the merge parameter list.
template<class T, int Rows>
__device__ void merge_verify(
    const float* workspace, const __nv_bfloat16* sink,
    const int32_t* seq_lens, const int32_t* q_start, const int32_t* q_len,
    const int32_t* prefix_len, const int32_t* live_batch,
    __nv_bfloat16* out, uint32_t split_capacity, uint32_t split_target,
    uint32_t split_units, uint32_t window_left) {
  (void)seq_lens;
  ptx::griddep_launch_dependents();
  verify_merge<T, Rows, 512>(workspace, sink, q_start, q_len, prefix_len, live_batch, out,
                             split_capacity, split_target, split_units, window_left);
}

} // namespace chop::attention
