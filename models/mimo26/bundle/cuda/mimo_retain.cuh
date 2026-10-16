#pragma once

// Retains target rows for a lazily caught-up drafter, and gathers them back
// when it catches up. Each slot owns a
// circular history of `capacity` rows; row r of a decode or verify phase
// belongs to item token_batch[r] at sequence position
// prefix_len[item] + r - q_start[item] and lands in history row
// slot_ids[item] * capacity + position % capacity. The kernel reads its live
// row count on the device, so it is captured with the decode and verify
// graphs.

#include <stdint.h>
#include "common/ptx.cuh"

namespace chop::mimo_retain {

// Copies `width` BF16 columns of each live row, 16 bytes per thread step.
// Widths, strides and span offsets are multiples of eight elements. Rows of
// an item outside `slots` are skipped.
__device__ void retain_rows(const uint16_t* input, uint16_t* history,
                            const int32_t* slot_ids, const int32_t* token_batch,
                            const int32_t* q_start, const int32_t* prefix_len,
                            const uint32_t* live_rows, uint32_t width,
                            uint32_t input_stride, uint32_t history_stride,
                            uint32_t capacity, uint32_t slots) {
  chop::ptx::griddep_launch_dependents();
  chop::ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  const int32_t item = token_batch[row];
  const int32_t slot = slot_ids[item];
  const int64_t position = int64_t(prefix_len[item]) + row - q_start[item];
  if (slot < 0 || uint32_t(slot) >= slots || position < 0) return;
  const uint64_t cell = uint64_t(slot) * capacity + uint64_t(position) % capacity;
  const uint4* source =
      reinterpret_cast<const uint4*>(input + uint64_t(row) * input_stride);
  uint4* target = reinterpret_cast<uint4*>(history + cell * history_stride);
  for (uint32_t i = threadIdx.x; i < width / 8; i += blockDim.x) target[i] = source[i];
}

// The reverse for a catch-up: row r of the phase is item token_batch[r]'s
// sequence position positions[r], read from history row
// slot_ids[item] * capacity + positions[r] % capacity into output row r.
__device__ void gather_rows(const uint16_t* history, uint16_t* output,
                            const int32_t* slot_ids, const int32_t* token_batch,
                            const int32_t* positions, const uint32_t* live_rows,
                            uint32_t width, uint32_t history_stride,
                            uint32_t output_stride, uint32_t capacity, uint32_t slots) {
  chop::ptx::griddep_launch_dependents();
  chop::ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  const int32_t item = token_batch[row];
  const int32_t slot = slot_ids[item];
  const int32_t position = positions[row];
  if (slot < 0 || uint32_t(slot) >= slots || position < 0) return;
  const uint64_t cell = uint64_t(slot) * capacity + uint64_t(position) % capacity;
  const uint4* source = reinterpret_cast<const uint4*>(history + cell * history_stride);
  uint4* target = reinterpret_cast<uint4*>(output + uint64_t(row) * output_stride);
  for (uint32_t i = threadIdx.x; i < width / 8; i += blockDim.x) target[i] = source[i];
}

}  // namespace chop::mimo_retain
