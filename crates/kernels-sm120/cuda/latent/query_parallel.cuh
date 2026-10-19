#pragma once
#include "glue.cuh"

namespace chop::latent::query_parallel {

// Four head groups share the row's validation barrier. A failed row still
// preserves every F32 value and writes zero to every native query head.
__device__ void rotate(float* values, uint16_t* native_q, const float* cos_sin,
    const uint32_t* live_rows, uint32_t heads, uint32_t* status,
    uint64_t status_row_offset) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  const uint32_t column = 2 * (threadIdx.x % 256);
  const uint32_t group = threadIdx.x / 256;
  const float* table = cos_sin + uint64_t(row) * 64;
  uint32_t flags = 0;
  for (uint32_t head = group; head < heads; head += 4) {
    const uint64_t at = (uint64_t(row) * heads + head) * 512 + column;
    float x, y;
    uint32_t pair_flags = 0;
    glue::rotated_pair<0>(values[at], values[at + 1], column, table, x, y, pair_flags);
    flags |= pair_flags;
  }
  const uint32_t failed = chop::row_status::commit(flags, status, status_row_offset + row);
  for (uint32_t head = group; head < heads; head += 4) {
    const uint64_t at = (uint64_t(row) * heads + head) * 512 + column;
    float x = 0.f, y = 0.f;
    uint32_t ignored = 0;
    if (!failed) {
      glue::rotated_pair<0>(values[at], values[at + 1], column, table, x, y, ignored);
      if (column >= 448) { values[at] = x; values[at + 1] = y; }
    }
    native_q[at] = convert::bf16_from_f32(x);
    native_q[at + 1] = convert::bf16_from_f32(y);
  }
}

} // namespace chop::latent::query_parallel
