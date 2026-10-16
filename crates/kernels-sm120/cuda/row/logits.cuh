#pragma once

#include "row.cuh"

namespace chop::row {

// The live vocabulary width excludes padded entries. A single CTA owns the
// row, so every tie is resolved by the lower vocabulary index.
template<int Capacity, int K>
__device__ void logits_topk(const float* logits, int32_t* ids,
                            float* log_probs, const uint32_t* live_rows,
                            const uint32_t* live_width) {
  ptx::griddep_launch_dependents();
  __shared__ float scores[1024];
  __shared__ int32_t indices[1024];
  __shared__ int32_t chosen[K];
  ptx::griddep_wait();
  uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  uint32_t width = min(*live_width, uint32_t(Capacity));
  float maximum = -INFINITY;
  for (uint32_t c = threadIdx.x; c < width; c += blockDim.x)
    maximum = fmaxf(maximum, logits[uint64_t(row) * Capacity + c]);
  scores[threadIdx.x] = maximum;
  __syncthreads();
  for (int step = blockDim.x / 2; step; step >>= 1) {
    if (threadIdx.x < step) scores[threadIdx.x] = fmaxf(scores[threadIdx.x], scores[threadIdx.x + step]);
    __syncthreads();
  }
  maximum = scores[0];
  // Every warp reads the maximum before thread 0 reuses scores[0] for a sum.
  __syncthreads();
  float sum = 0.f;
  for (uint32_t c = threadIdx.x; c < width; c += blockDim.x)
    sum += __expf(logits[uint64_t(row) * Capacity + c] - maximum);
  scores[threadIdx.x] = sum;
  __syncthreads();
  for (int step = blockDim.x / 2; step; step >>= 1) {
    if (threadIdx.x < step) scores[threadIdx.x] += scores[threadIdx.x + step];
    __syncthreads();
  }
  // Only thread 0 consumes the log sum, reading it before its own next write.
  float logsum = logf(scores[0]) + maximum;
  for (int k = 0; k < K; ++k) {
    float best = -INFINITY;
    int32_t best_id = INT32_MAX;
    for (uint32_t c = threadIdx.x; c < width; c += blockDim.x) {
      bool selected = false;
      for (int j = 0; j < k; ++j) selected |= chosen[j] == int32_t(c);
      if (selected) continue;
      float score = logits[uint64_t(row) * Capacity + c];
      if (score > best || (score == best && int32_t(c) < best_id)) {
        best = score;
        best_id = c;
      }
    }
    scores[threadIdx.x] = best;
    indices[threadIdx.x] = best_id;
    __syncthreads();
    for (int step = blockDim.x / 2; step; step >>= 1) {
      if (threadIdx.x < step) {
        float candidate = scores[threadIdx.x + step];
        int32_t id = indices[threadIdx.x + step];
        if (candidate > scores[threadIdx.x] ||
            (candidate == scores[threadIdx.x] && id < indices[threadIdx.x])) {
          scores[threadIdx.x] = candidate;
          indices[threadIdx.x] = id;
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      chosen[k] = indices[0];
      ids[uint64_t(row) * K + k] = indices[0];
      if (log_probs) log_probs[uint64_t(row) * K + k] = scores[0] - logsum;
    }
    __syncthreads();
  }
}

} // namespace chop::row
