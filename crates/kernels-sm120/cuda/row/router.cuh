#pragma once

#include "row.cuh"

namespace chop::row {

template<int Experts, int TopK>
__device__ void router_sigmoid_warp(const float* logits, const float* bias,
                                    int32_t* ids, float* weights,
                                    const uint32_t* live_rows,
                                    float scaling_factor) {
  static_assert(Experts == 256 && TopK == 8);
  begin();
  const uint32_t token = blockIdx.x;
  if (token >= *live_rows) return;
  const int lane = threadIdx.x;
  float scores[8], unbiased[8], selected[8];
  #pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int expert = lane + j * 32;
    float sigmoid = 1.f / (1.f + expf(-logits[uint64_t(token) * Experts + expert]));
    unbiased[j] = sigmoid;
    scores[j] = sigmoid + bias[expert];
  }
  float selected_sum = 0.f;
  #pragma unroll
  for (int k = 0; k < TopK; ++k) {
    float best_score = -INFINITY;
    int best_id = INT32_MAX;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int expert = lane + j * 32;
      if (scores[j] > best_score || (scores[j] == best_score && expert < best_id)) {
        best_score = scores[j];
        best_id = expert;
      }
    }
    for (int offset = 16; offset; offset >>= 1) {
      float candidate = __shfl_down_sync(0xffffffff, best_score, offset);
      int candidate_id = __shfl_down_sync(0xffffffff, best_id, offset);
      if (candidate > best_score || (candidate == best_score && candidate_id < best_id)) {
        best_score = candidate;
        best_id = candidate_id;
      }
    }
    best_id = __shfl_sync(0xffffffff, best_id, 0);
    float local_weight = 0.f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      if (best_id == lane + j * 32) {
        local_weight = unbiased[j];
        scores[j] = -INFINITY;
      }
    }
    float chosen_weight = __shfl_sync(0xffffffff, local_weight, best_id & 31);
    if (lane == 0) {
      ids[uint64_t(token) * TopK + k] = best_id;
      selected[k] = chosen_weight;
      selected_sum += chosen_weight;
    }
  }
  if (lane == 0) {
    #pragma unroll
    for (int k = 0; k < TopK; ++k)
      weights[uint64_t(token) * TopK + k] =
          selected[k] / (selected_sum + 1e-20f) * scaling_factor;
  }
}

// Kind 0: softmax, 1: sigmoid with correction bias, 2: grouped top-k.
template<int Experts, int TopK, int Kind, bool Renorm, int Groups, int Take>
__device__ void router(const float* logits, const float* bias, int32_t* ids,
                       float* weights, const uint32_t* live_rows,
                       float scaling_factor) {
  ptx::griddep_launch_dependents();
  __shared__ float scores[Experts];
  __shared__ float unbiased[Experts];
  __shared__ float group_scores[Groups];
  __shared__ float selected_values[TopK];
  ptx::griddep_wait();
  uint32_t token = blockIdx.x;
  if (token >= *live_rows) return;
  for (int e = threadIdx.x; e < Experts; e += blockDim.x) {
    float raw = logits[uint64_t(token) * Experts + e];
    float score = Kind == 1 ? 1.f / (1.f + expf(-raw)) : raw;
    unbiased[e] = score;
    scores[e] = score + (Kind == 1 ? bias[e] : 0.f);
  }
  __syncthreads();
  if (threadIdx.x != 0) return;
  if constexpr (Kind == 2) {
    constexpr int GroupSize = Experts / Groups;
    for (int g = 0; g < Groups; ++g) {
      float first = -INFINITY, second = -INFINITY;
      for (int e = g * GroupSize; e < (g + 1) * GroupSize; ++e) {
        float v = scores[e];
        if (v > first) { second = first; first = v; }
        else if (v > second) second = v;
      }
      group_scores[g] = first + (GroupSize > 1 ? second : 0.0);
    }
    for (int i = 0; i < Take; ++i) {
      int best = -1;
      for (int g = 0; g < Groups; ++g)
        if (best < 0 || group_scores[g] > group_scores[best]) best = g;
      group_scores[best] = -INFINITY;
    }
    for (int e = 0; e < Experts; ++e)
      if (group_scores[e / GroupSize] != -INFINITY) scores[e] = -INFINITY;
  }
  float max_score = -INFINITY;
  if constexpr (Kind == 0)
    for (int e = 0; e < Experts; ++e) max_score = fmax(max_score, scores[e]);
  float total = 0.f;
  if constexpr (Kind == 0)
    for (int e = 0; e < Experts; ++e) total += expf(scores[e] - max_score);
  float chosen_sum = 0.f;
  for (int k = 0; k < TopK; ++k) {
    int best = -1;
    for (int e = 0; e < Experts; ++e)
      if (best < 0 || scores[e] > scores[best]) best = e;
    ids[token * TopK + k] = best;
    float w = Kind == 1 ? unbiased[best] :
              (Kind == 0 ? expf(scores[best] - max_score) / total : scores[best]);
    selected_values[k] = w;
    chosen_sum += w;
    scores[best] = -INFINITY;
  }
  if constexpr (Kind == 2) {
    float top_max = -INFINITY;
    for (int k = 0; k < TopK; ++k)
      top_max = fmax(top_max, selected_values[k]);
    chosen_sum = 0.f;
    for (int k = 0; k < TopK; ++k) {
      float w = expf(selected_values[k] - top_max);
      selected_values[k] = w;
      chosen_sum += w;
    }
  }
  float denominator = Kind == 1 || Kind == 2 || Renorm ?
                      chosen_sum + (Kind == 1 ? 1e-20f : 0.f) : 1.f;
  for (int k = 0; k < TopK; ++k)
    weights[token * TopK + k] = selected_values[k] / denominator *
                                (Kind == 1 ? scaling_factor : 1.f);
}

template<int Experts, int TopK, int Block>
__device__ void permute_count(const int32_t* ids, int32_t* counts,
                              const uint32_t* live_rows) {
  begin();
  int expert = blockIdx.x;
  if (expert >= Experts) return;
  int count = 0;
  for (uint32_t r = threadIdx.x; r < *live_rows * TopK; r += blockDim.x)
    count += ids[r] == expert;
  __shared__ int partial[1024];
  partial[threadIdx.x] = count;
  __syncthreads();
  for (int step = blockDim.x / 2; step; step >>= 1) {
    if (threadIdx.x < step) partial[threadIdx.x] += partial[threadIdx.x + step];
    __syncthreads();
  }
  if (threadIdx.x == 0) counts[expert] = partial[0];
}

template<int Experts, int Block>
__device__ void permute_scan(const int32_t* counts, int32_t* offsets,
                             int32_t* padded_rows) {
  begin();
  if (threadIdx.x == 0) {
    int total = 0;
    for (int e = 0; e < Experts; ++e) {
      offsets[e] = total;
      total += (counts[e] + Block - 1) / Block * Block;
    }
    offsets[Experts] = total;
    *padded_rows = total;
  }
}

// One CTA per expert owns its output range. Its ascending source scan is
// stable and uses no atomics. A following grouped GEMM reads padded_rows.
template<int Experts, int TopK, int Block>
__device__ void permute_scatter(const int32_t* ids, const int32_t* offsets,
                                int32_t* sorted_rows, const uint32_t* live_rows) {
  ptx::griddep_launch_dependents();
  __shared__ int warp_counts[8];
  ptx::griddep_wait();
  int expert = blockIdx.x;
  if (expert >= Experts) return;
  const int start = offsets[expert];
  int written = 0;
  const uint32_t routes = *live_rows * TopK;
  for (uint32_t chunk = 0; chunk < routes; chunk += blockDim.x) {
    uint32_t route = chunk + threadIdx.x;
    bool matches = route < routes && ids[route] == expert;
    unsigned mask = __ballot_sync(0xffffffff, matches);
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) warp_counts[warp] = __popc(mask);
    __syncthreads();
    int before_warp = 0, in_chunk = 0;
    for (int w = 0; w < warp; ++w) before_warp += warp_counts[w];
    for (int w = 0; w < blockDim.x / 32; ++w) in_chunk += warp_counts[w];
    if (matches)
      sorted_rows[start + written + before_warp + __popc(mask & ((1u << lane) - 1))] = route;
    written += in_chunk;
    __syncthreads();
  }
  for (int slot = start + written + threadIdx.x; slot < offsets[expert + 1]; slot += blockDim.x)
    sorted_rows[slot] = -1;
}

template<int Width, int TopK>
__device__ void gather(const uint16_t* input, const int32_t* sorted_rows,
                       uint16_t* output, const int32_t* padded_rows) {
  begin();
  uint32_t row = blockIdx.x;
  if (row >= uint32_t(*padded_rows)) return;
  int32_t route = sorted_rows[row];
  for (int c = threadIdx.x * 8; c < Width; c += blockDim.x * 8) {
    const uint64_t dst = uint64_t(row) * Width + c;
    if (c + 8 <= Width) {
      uint4 v = route < 0 ? make_uint4(0, 0, 0, 0) :
                reinterpret_cast<const uint4*>(input + uint64_t(route / TopK) * Width + c)[0];
      reinterpret_cast<uint4*>(output + dst)[0] = v;
    } else {
      for (int j = 0; j < 8 && c + j < Width; ++j)
        output[dst + j] = route < 0 ? 0 : input[uint64_t(route / TopK) * Width + c + j];
    }
  }
}

// Prefill routes are partitioned into 256-route tiles. Counting may use
// atomics because only totals are stored; scatter computes each route's
// stable rank from warp masks and deterministic preceding-warp counts.
template<int Experts, int TopK, int Tiles>
__device__ void permute_tile_count(const int32_t* ids, int32_t* tile_counts,
                                   const uint32_t* live_rows) {
  ptx::griddep_launch_dependents();
  __shared__ int histogram[Experts];
  ptx::griddep_wait();
  const uint32_t tile = blockIdx.x;
  const uint32_t route = tile * 256 + threadIdx.x;
  const uint32_t routes = *live_rows * TopK;
  const int expert = route < routes ? ids[route] : -1;
  histogram[threadIdx.x] = 0;
  __syncthreads();
  if (expert >= 0 && expert < Experts) atomicAdd(&histogram[expert], 1);
  __syncthreads();
  tile_counts[uint64_t(tile) * Experts + threadIdx.x] = histogram[threadIdx.x];
}

__device__ __forceinline__ int scan256_exclusive(int value, int* warp_bases) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  int inclusive = value;
  for (int step = 1; step < 32; step <<= 1) {
    const int prior = __shfl_up_sync(0xffffffff, inclusive, step);
    if (lane >= step) inclusive += prior;
  }
  if (lane == 31) warp_bases[warp] = inclusive;
  __syncthreads();
  if (warp == 0) {
    const int warp_count = lane < 8 ? warp_bases[lane] : 0;
    int prefix = warp_count;
    for (int step = 1; step < 32; step <<= 1) {
      const int prior = __shfl_up_sync(0xffffffff, prefix, step);
      if (lane >= step) prefix += prior;
    }
    if (lane < 8) warp_bases[lane] = prefix - warp_count;
  }
  __syncthreads();
  return warp_bases[warp] + inclusive - value;
}

template<int Experts, int TopK, int Tiles>
__device__ void permute_tile_prefix(const int32_t* tile_counts, int32_t* counts,
                                    int32_t* tile_offsets,
                                    const uint32_t* live_rows) {
  ptx::griddep_launch_dependents();
  __shared__ int warp_bases[8];
  ptx::griddep_wait();
  const int expert = blockIdx.x;
  const int tile = threadIdx.x;
  const uint32_t live_tiles = (*live_rows * TopK + 255) / 256;
  const int count = tile < live_tiles ? tile_counts[uint64_t(tile) * Experts + expert] : 0;
  const int before = scan256_exclusive(count, warp_bases);
  if (tile < live_tiles) tile_offsets[uint64_t(tile) * Experts + expert] = before;
  if (tile == Tiles - 1) counts[expert] = before + count;
}

template<int Experts, int Block>
__device__ void permute_tile_scan(const int32_t* counts, int32_t* offsets,
                                  int32_t* sorted_rows, int32_t* padded_rows) {
  ptx::griddep_launch_dependents();
  __shared__ int warp_bases[8];
  ptx::griddep_wait();
  const int expert = threadIdx.x;
  const int count = counts[expert];
  const int padded = (count + Block - 1) / Block * Block;
  const int before = scan256_exclusive(padded, warp_bases);
  offsets[expert] = before;
  if (expert == Experts - 1) {
    offsets[Experts] = before + padded;
    *padded_rows = before + padded;
  }
  __syncthreads();
  for (int slot = before + count; slot < before + padded; ++slot)
    sorted_rows[slot] = -1;
}

template<int Experts, int TopK, int Tiles>
__device__ void permute_tile_scatter(const int32_t* ids, const int32_t* tile_offsets,
                                     const int32_t* offsets, int32_t* sorted_rows,
                                     const uint32_t* live_rows) {
  ptx::griddep_launch_dependents();
  __shared__ int warp_counts[8][Experts];
  ptx::griddep_wait();
  const uint32_t tile = blockIdx.x;
  const uint32_t route = tile * 256 + threadIdx.x;
  const uint32_t routes = *live_rows * TopK;
  if (tile * 256 >= routes) return;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  for (int e = lane; e < Experts; e += 32) warp_counts[warp][e] = 0;
  __syncthreads();
  const int expert = route < routes ? ids[route] : -1;
  if (expert >= 0 && expert < Experts) atomicAdd(&warp_counts[warp][expert], 1);
  __syncthreads();
  const unsigned matches = __match_any_sync(0xffffffff, expert);
  if (expert < 0 || expert >= Experts) return;
  int before_warp = 0;
  for (int w = 0; w < warp; ++w) before_warp += warp_counts[w][expert];
  const int before_lane = __popc(matches & ((1u << lane) - 1));
  const uint64_t slot = uint64_t(offsets[expert]) +
      tile_offsets[uint64_t(tile) * Experts + expert] + before_warp + before_lane;
  sorted_rows[slot] = route;
}

template<int Width, int TopK>
__device__ void gather_mxfp8(const uint16_t* input, const int32_t* sorted_rows,
                             uint8_t* codes, uint8_t* scales,
                             const int32_t* padded_rows) {
  begin();
  uint32_t row = blockIdx.x;
  if (row >= uint32_t(*padded_rows)) return;
  int32_t route = sorted_rows[row];
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int group = warp; group < Width / 32; group += blockDim.x / 32) {
    uint64_t index = uint64_t(row) * Width + group * 32 + lane;
    float value = route < 0 ? 0.f :
        bf(input[uint64_t(route / TopK) * Width + group * 32 + lane]);
    float maximum = fabsf(value);
    for (int offset = 16; offset; offset >>= 1)
      maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
    int code = maximum == 0.f ? 127 : max(0, min(254, ilogbf(maximum) - 8 + 127));
    if (lane == 0) scales[uint64_t(row) * (Width / 32) + group] = uint8_t(code);
    float scale = exp2f(float(code - 127));
    uint16_t packed = convert::pack_f32_e4m3({float(double(value) / double(scale)), 0.f});
    codes[index] = uint8_t(packed);
  }
}

} // namespace chop::row
