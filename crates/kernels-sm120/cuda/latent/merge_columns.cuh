#pragma once

// Column-parallel split merge (README.md "Column merge"): `merge`'s outputs
// bit for bit. Every output takes the same operations on the same values in
// the same order: the maximum over the splits in split order, each split's
// weight `exp_of(m_s - M)` (0 for an empty split), the denominator and each
// numerator column as FMA sums from zero in split order, then `O / L`. Only
// the schedule differs: a CTA takes 128 columns of one row and head, each
// thread one column, and every split's value of that column is loaded
// before the fold, where `merge` folds four columns a thread with one
// dependent load per split and step.

#include "attention.cuh"

// Lab-only mutants (README.md "Column merge"); each is 0 when served.
#ifndef CHOP_LATENT_MERGE_COLUMNS_TEST_SKIP_LAST
#define CHOP_LATENT_MERGE_COLUMNS_TEST_SKIP_LAST 0
#endif
#ifndef CHOP_LATENT_MERGE_COLUMNS_TEST_DESCENDING
#define CHOP_LATENT_MERGE_COLUMNS_TEST_DESCENDING 0
#endif
#ifndef CHOP_LATENT_MERGE_COLUMNS_TEST_LATE_WAIT
#define CHOP_LATENT_MERGE_COLUMNS_TEST_LATE_WAIT 0
#endif

namespace chop::latent {

// Columns, and threads, of one column merge CTA.
constexpr int merge_columns = 128;

// Grid [rows, heads, latent / 128], 128 threads: CTA (r, h, c) writes
// columns [128 c, 128 c + 128) of row r and head h; CTA c = 0 also writes
// the partial's maximum and denominator.
template<class T>
__device__ void merge_columns_of(const float* workspace, const uint32_t* live_rows, uint16_t* out,
                                 float* out_max, float* out_den, float* out_num,
                                 uint32_t splits) {
  static_assert(T::latent % merge_columns == 0, "column CTAs cover the latent row");
  static_assert(max_splits <= merge_columns, "a thread loads each split's max and denominator");
  ptx::griddep_launch_dependents();
  __shared__ float maxima[max_splits];
  __shared__ float denominators[max_splits];
  __shared__ float weights[max_splits];
  __shared__ float totals[2];
  if (!CHOP_LATENT_MERGE_COLUMNS_TEST_LATE_WAIT) ptx::griddep_wait();
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  if (row >= int(*live_rows)) return;
  const int d = blockIdx.z * merge_columns + threadIdx.x;
  const uint32_t folded = splits - CHOP_LATENT_MERGE_COLUMNS_TEST_SKIP_LAST;
  auto partial = [&](uint32_t split) {
    return workspace + partial_index<T>(row, splits, split, head);
  };
  // Every split's value of this thread's column, and thread s's split s
  // maximum and denominator, all requested before any is used.
  float column[max_splits];
#pragma unroll
  for (int split = 0; split < max_splits; ++split)
    column[split] = uint32_t(split) < folded ? partial(split)[2 + d] : 0.0f;
  if (threadIdx.x < folded) {
    maxima[threadIdx.x] = partial(threadIdx.x)[0];
    denominators[threadIdx.x] = partial(threadIdx.x)[1];
  }
  if (CHOP_LATENT_MERGE_COLUMNS_TEST_LATE_WAIT) ptx::griddep_wait();
  __syncthreads();
  if (threadIdx.x == 0) {
    float top = -INFINITY;
    for (uint32_t split = 0; split < folded; ++split) top = fmaxf(top, maxima[split]);
    totals[0] = top;
  }
  __syncthreads();
  if (threadIdx.x < folded) {
    const float m = maxima[threadIdx.x];
    weights[threadIdx.x] = m == -INFINITY ? 0.0f : exp_of(m - totals[0]);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float sum = 0.0f;
    for (uint32_t split = 0; split < folded; ++split)
      sum = fmaf(weights[split], denominators[split], sum);
    totals[1] = sum;
  }
  __syncthreads();
  float numerator = 0.0f;
#pragma unroll
  for (int step = 0; step < max_splits; ++step) {
    const int split = CHOP_LATENT_MERGE_COLUMNS_TEST_DESCENDING ? max_splits - 1 - step : step;
    if (uint32_t(split) < folded) numerator = fmaf(weights[split], column[split], numerator);
  }
  const uint64_t pair = uint64_t(row) * T::heads + head;
  const float total = totals[1];
  if constexpr (T::partial) {
    if (blockIdx.z == 0 && threadIdx.x == 0) {
      out_max[pair] = totals[0];
      out_den[pair] = total;
    }
    out_num[pair * T::latent + d] = numerator;
  } else {
    out[pair * T::latent + d] =
        convert::bf16_from_f32(total == 0.0f ? 0.0f : __fdiv_rn(numerator, total));
  }
}

} // namespace chop::latent
