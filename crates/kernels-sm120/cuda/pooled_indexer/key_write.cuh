#pragma once

// Indexer key write (README.md "Key write"): the key LayerNorm, the tail
// state of each sequence's incomplete group and the pooled key of every
// group a row completes. A warp owns one row; lane l holds channels 4l to
// 4l + 3.
//
// A pool warp is a row that completes a group whose tokens are all rows of
// this launch: it normalizes them and pools. A sequence warp is the first
// row of a sequence: it pools the group that straddles the launch start
// (its earlier tokens are live tail slots) and writes the tail state, so
// every tail read and write of a sequence happens in one warp, in order.

#include "pooled_indexer/common.cuh"

namespace chop::pooled_indexer {

// `MaxRows` is 0 for prefill and decode, which leave the state after the
// last row in the source entry, and the verify row bound otherwise, which
// writes one checkpoint entry per row.
template<int Width, int Group, int Page, bool PositionF32, int MaxRows>
struct KeyWrite {
  static_assert(Width == 128, "a warp holds one 128-wide row");
  static_assert(Group >= 2 && Group <= 8 && (Group & (Group - 1)) == 0);
  static_assert(Page % Group == 0 && MaxRows >= 0 && MaxRows <= 16);
  static constexpr int width = Width, group = Group, page = Page, max_rows = MaxRows;
  static constexpr bool position_f32 = PositionF32;
  static constexpr int bank = (Group - 1) * Width;
  static constexpr int state_elements = 2 * bank;
  static constexpr int warps = 4;
};

struct KeyWriteArgs {
  const uint16_t* key;
  uint32_t key_stride;
  const uint16_t* gate;
  uint32_t gate_stride;
  const uint16_t* norm_weight;
  const uint16_t* norm_bias;
  const void* position;
  const int32_t* row_batch;
  const int32_t* positions;
  uint32_t live;
  const int32_t* slot_ids;
  const int32_t* checkpoint_ids;
  uint16_t* tail_state;
  const int32_t* page_tables;
  uint8_t* pooled_keys;
  uint32_t max_pages;
  uint64_t page_stride_bytes;
  float eps;
  // Context-parallel group rank and size (README.md "Pages").
  uint32_t cp_rank;
  uint32_t cp_size;
};

// The BF16 normalized key of `row`, as F32 values of the stored BF16.
template<class T>
__device__ __forceinline__ void normalize(const KeyWriteArgs& a, int row, float (&y)[4]) {
  const int lane = threadIdx.x & 31;
  float x[4], w[4], b[4];
  load4(x, a.key + size_t(row) * a.key_stride + 4 * lane);
  load4(w, a.norm_weight + 4 * lane);
  load4(b, a.norm_bias + 4 * lane);
  const float mean = warp_sum(((x[0] + x[1]) + x[2]) + x[3]) / float(T::width);
  float squares = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    x[i] -= mean;
    squares = fmaf(x[i], x[i], squares);
  }
  const float rstd = __frsqrt_rn(warp_sum(squares) / float(T::width) + a.eps);
#pragma unroll
  for (int i = 0; i < 4; ++i) y[i] = bf16(to_bf16(fmaf(x[i] * rstd, w[i], b[i])));
}

template<class T>
__device__ __forceinline__ void load_gate(const KeyWriteArgs& a, int row, float (&g)[4]) {
  load4(g, a.gate + size_t(row) * a.gate_stride + 4 * (threadIdx.x & 31));
}

// Softmax over the group of (gate + position) per channel, then the
// weighted key sum in offset order, rounded once and stored to the page
// row of `pool` when this group rank holds its page: physical page j at
// stripe j / cp_size of the table, on group rank j % cp_size.
template<class T>
__device__ __forceinline__ void pool_group(const KeyWriteArgs& a, int batch, int64_t pool,
                                           const float (&keys)[T::group][4],
                                           const float (&gates)[T::group][4]) {
  const int lane = threadIdx.x & 31;
  float out[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int channel = 4 * lane + i;
    float logit[T::group], m = -INFINITY;
#pragma unroll
    for (int t = 0; t < T::group; ++t) {
      const float p = T::position_f32
          ? static_cast<const float*>(a.position)[t * T::width + channel]
          : bf16(static_cast<const uint16_t*>(a.position)[t * T::width + channel]);
      logit[t] = gates[t][i] + p;
      m = fmaxf(m, logit[t]);
    }
    float e[T::group], total = 0.f;
#pragma unroll
    for (int t = 0; t < T::group; ++t) {
      e[t] = expf(logit[t] - m);
      total += e[t];
    }
    float sum = 0.f;
#pragma unroll
    for (int t = 0; t < T::group; ++t) sum = fmaf(e[t] / total, keys[t][i], sum);
    out[i] = sum;
  }
  const int64_t token = pool * T::group, physical = token / T::page;
  if (uint32_t(physical % a.cp_size) != a.cp_rank) return;
  const int32_t page = a.page_tables[size_t(batch) * a.max_pages + size_t(physical / a.cp_size)];
  uint8_t* row = a.pooled_keys + size_t(page) * a.page_stride_bytes +
                 size_t(token % T::page / T::group) * T::width * 2;
  store4(reinterpret_cast<uint16_t*>(row) + 4 * lane, out);
}

// Rows of `batch` from `row` on (a sequence's rows are contiguous).
__device__ __forceinline__ int run_length(const KeyWriteArgs& a, int row, int batch) {
  const int lane = threadIdx.x & 31;
  int rows = 0;
  for (int base = row;; base += 32) {
    const int j = base + lane;
    const uint32_t same = __ballot_sync(0xffffffffu, j < int(a.live) && a.row_batch[j] == batch);
    if (same != 0xffffffffu) return rows + __ffs(~same) - 1;
    rows += 32;
  }
}

// Writes tail slot `slot` of `state` from launch row `row`.
template<class T>
__device__ __forceinline__ void write_slot(const KeyWriteArgs& a, uint16_t* state, int slot,
                                           int row) {
  float y[4], g[4];
  normalize<T>(a, row, y);
  load_gate<T>(a, row, g);
  const int lane = threadIdx.x & 31;
  store4(state + slot * T::width + 4 * lane, y);
  store4(state + T::bank + slot * T::width + 4 * lane, g);
}

template<class T>
__device__ void key_write(KeyWriteArgs a, const uint32_t* live_rows) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  a.live = *live_rows;
  const int row = int(blockIdx.x) * T::warps + int(threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (row >= int(a.live)) return;
  const int batch = a.row_batch[row];
  const int64_t position = a.positions[row];
  const int offset = int(position % T::group);

  // Pool warp: a group whose tokens are all rows of this launch.
  const int first = row - (T::group - 1);
  if (offset == T::group - 1 && first >= 0 && a.row_batch[first] == batch) {
    float keys[T::group][4], gates[T::group][4];
#pragma unroll
    for (int t = 0; t < T::group; ++t) {
      normalize<T>(a, first + t, keys[t]);
      load_gate<T>(a, first + t, gates[t]);
    }
    pool_group<T>(a, batch, position / T::group, keys, gates);
    return;
  }
  if (row > 0 && a.row_batch[row - 1] == batch) return;

  // Sequence warp. The source entry's live slots 0..offset hold the
  // group's tokens before this launch.
  const int rows = run_length(a, row, batch);
  const uint16_t* source = a.tail_state + size_t(a.slot_ids[batch]) * T::state_elements;
  if (offset > 0 && rows >= T::group - offset) {
    float keys[T::group][4], gates[T::group][4];
#pragma unroll
    for (int t = 0; t < T::group; ++t) {
      if (t < offset) {
        load4(keys[t], source + t * T::width + 4 * lane);
        load4(gates[t], source + T::bank + t * T::width + 4 * lane);
      } else {
        normalize<T>(a, row + t - offset, keys[t]);
        load_gate<T>(a, row + t - offset, gates[t]);
      }
    }
    pool_group<T>(a, batch, position / T::group, keys, gates);
  }
  __syncwarp();
  if constexpr (T::max_rows == 0) {
    // The state after the last row, in place: slots of tokens before this
    // launch stay; the incomplete group's launch rows fill the rest.
    const int64_t last = position + rows - 1;
    const int tail = int((last + 1) % T::group);
    const int64_t base = last + 1 - tail;
    uint16_t* target = a.tail_state + size_t(a.slot_ids[batch]) * T::state_elements;
    for (int slot = 0; slot < tail; ++slot)
      if (base + slot >= position) write_slot<T>(a, target, slot, row + int(base + slot - position));
  } else {
    // One checkpoint per row, each the whole tail after that row. A
    // checkpoint entry never aliases the source entry.
    for (int r = 0; r < rows && r < T::max_rows; ++r) {
      const int64_t at = position + r;
      const int tail = int((at + 1) % T::group);
      const int64_t base = at + 1 - tail;
      uint16_t* target =
          a.tail_state + size_t(a.checkpoint_ids[batch * T::max_rows + r]) * T::state_elements;
      for (int slot = 0; slot < tail; ++slot) {
        if (base + slot >= position) {
          write_slot<T>(a, target, slot, row + int(base + slot - position));
        } else {
          float k[4], g[4];
          load4(k, source + slot * T::width + 4 * lane);
          load4(g, source + T::bank + slot * T::width + 4 * lane);
          store4(target + slot * T::width + 4 * lane, k);
          store4(target + T::bank + slot * T::width + 4 * lane, g);
        }
      }
    }
  }
}

}  // namespace chop::pooled_indexer
