#pragma once

#include <cuda_bf16.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include "common/ptx.cuh"

namespace chop::sampling {

// The host uploads these naturally aligned, little-endian records. Offsets
// index the flat counts, biases, and EOS arrays for this row.
struct TokenCount {
  uint32_t token_id;
  uint32_t prompt_seen;
  uint32_t output_count;
  uint32_t reserved;
};
struct TokenBias {
  uint32_t token_id;
  uint32_t reserved;
  double bias;
};
struct RowMeta {
  uint64_t seed;
  uint64_t generated;
  uint64_t counts_offset;
  uint64_t biases_offset;
  uint64_t eos_offset;
  uint32_t counts_len;
  uint32_t biases_len;
  uint32_t eos_len;
  uint32_t top_k;
  uint32_t top_logprobs;
  uint32_t mask_eos;
  double temperature;
  double top_p;
  double min_p;
  double repetition_penalty;
  double frequency_penalty;
  double presence_penalty;
};
struct RowOutput {
  uint32_t token_id;
  uint32_t status;
  double logprob;
  uint32_t top_ids[20];
  double top_logprobs[20];
};
struct SimplePartial {
  double raw_max;
  double raw_sum;
  double best_score;
  uint32_t token_id;
  uint32_t invalid;
  uint32_t top_ids[20];
};

static_assert(sizeof(TokenCount) == 16 && sizeof(TokenBias) == 16,
              "host sampling list records must match device layout");
static_assert(sizeof(RowMeta) == 112 && offsetof(RowMeta, temperature) == 64 &&
              offsetof(RowMeta, presence_penalty) == 104,
              "host sampling metadata must match device layout");
static_assert(sizeof(RowOutput) == 256 && offsetof(RowOutput, top_ids) == 16 &&
              offsetof(RowOutput, top_logprobs) == 96,
              "host sampled output must match device layout");
static_assert(sizeof(SimplePartial) == 112,
              "simple sampling partials must match the planner");

__device__ __forceinline__ float load_logit(const float* row, uint32_t token) {
  return row[token];
}
__device__ __forceinline__ float load_logit(const __nv_bfloat16* row,
                                             uint32_t token) {
  return __bfloat162float(row[token]);
}

__device__ __forceinline__ bool better(double a, uint32_t aid, double b,
                                       uint32_t bid) {
  return a > b || (a == b && aid < bid);
}

__device__ __forceinline__ TokenCount find_count(const TokenCount* counts,
                                                   uint32_t length,
                                                   uint32_t token) {
  uint32_t lo = 0, hi = length;
  while (lo < hi) {
    uint32_t mid = lo + (hi - lo) / 2;
    if (counts[mid].token_id < token) lo = mid + 1;
    else hi = mid;
  }
  if (lo < length && counts[lo].token_id == token) return counts[lo];
  return TokenCount{token, 0, 0, 0};
}

__device__ __forceinline__ double find_bias(const TokenBias* biases,
                                             uint32_t length, uint32_t token) {
  uint32_t lo = 0, hi = length;
  while (lo < hi) {
    uint32_t mid = lo + (hi - lo) / 2;
    if (biases[mid].token_id < token) lo = mid + 1;
    else hi = mid;
  }
  return lo < length && biases[lo].token_id == token ? biases[lo].bias : 0.0;
}

__device__ __forceinline__ bool is_eos(const uint32_t* eos, uint32_t length,
                                        uint32_t token) {
  for (uint32_t i = 0; i < length; ++i) if (eos[i] == token) return true;
  return false;
}

// A constraint mask admits token t through bit t % 32 of word t / 32; a row
// without a mask admits every token.
__device__ __forceinline__ bool admitted(const uint32_t* mask, uint32_t token) {
  return mask == nullptr || ((mask[token >> 5] >> (token & 31)) & 1u) != 0;
}

__device__ __forceinline__ void philox(uint64_t seed, uint64_t generated,
                                        uint32_t token, uint32_t (&x)[4]) {
  x[0] = token;
  x[1] = static_cast<uint32_t>(generated);
  x[2] = static_cast<uint32_t>(generated >> 32);
  x[3] = 0;
  uint32_t k0 = static_cast<uint32_t>(seed);
  uint32_t k1 = static_cast<uint32_t>(seed >> 32);
  #pragma unroll
  for (int round = 0; round < 10; ++round) {
    uint64_t a = 0xd2511f53ULL * x[0];
    uint64_t b = 0xcd9e8d57ULL * x[2];
    uint32_t y0 = static_cast<uint32_t>(b >> 32) ^ x[1] ^ k0;
    uint32_t y1 = static_cast<uint32_t>(b);
    uint32_t y2 = static_cast<uint32_t>(a >> 32) ^ x[3] ^ k1;
    uint32_t y3 = static_cast<uint32_t>(a);
    x[0] = y0; x[1] = y1; x[2] = y2; x[3] = y3;
    k0 += 0x9e3779b9U;
    k1 += 0xbb67ae85U;
  }
}

__device__ __forceinline__ double gumbel(uint64_t seed, uint64_t generated,
                                           uint32_t token) {
  uint32_t x[4];
  philox(seed, generated, token, x);
  uint64_t bits = (static_cast<uint64_t>(x[0]) << 21) | (x[1] >> 11);
  double unit = (static_cast<double>(bits) + 0.5) *
                (1.0 / 9007199254740992.0);
  return -log(-log(unit));
}

// The approximate score only prunes distant candidates. This polynomial
// avoids two SFU logarithms for each vocabulary element. Possible winners
// are recomputed with the f64 score above before choosing a token.
__device__ __forceinline__ float fast_log_positive(float value) {
  uint32_t bits = __float_as_uint(value);
  int exponent = static_cast<int>(bits >> 23) - 127;
  float mantissa = __uint_as_float((bits & 0x007fffffU) | 0x3f800000U);
  float t = (mantissa - 1.5f) * (2.0f / 3.0f);
  float polynomial = 1.0f / 11.0f;
  polynomial = fmaf(polynomial, t, -1.0f / 10.0f);
  polynomial = fmaf(polynomial, t, 1.0f / 9.0f);
  polynomial = fmaf(polynomial, t, -1.0f / 8.0f);
  polynomial = fmaf(polynomial, t, 1.0f / 7.0f);
  polynomial = fmaf(polynomial, t, -1.0f / 6.0f);
  polynomial = fmaf(polynomial, t, 1.0f / 5.0f);
  polynomial = fmaf(polynomial, t, -1.0f / 4.0f);
  polynomial = fmaf(polynomial, t, 1.0f / 3.0f);
  polynomial = fmaf(polynomial, t, -0.5f);
  polynomial = fmaf(polynomial, t, 1.0f);
  return fmaf(static_cast<float>(exponent), 0.69314718056f,
              fmaf(t, polynomial, 0.40546510811f));
}

__device__ __forceinline__ float approximate_gumbel(uint64_t seed,
    uint64_t generated, uint32_t token) {
  uint32_t x[4];
  philox(seed, generated, token, x);
  float unit = static_cast<float>(x[0]) * (1.0f / 4294967296.0f);
  if (unit < 0.00001f || unit > 0.999f) {
    uint64_t bits = (static_cast<uint64_t>(x[0]) << 21) | (x[1] >> 11);
    double exact_unit = (static_cast<double>(bits) + 0.5) *
                        (1.0 / 9007199254740992.0);
    return static_cast<float>(-log(-log(exact_unit)));
  }
  return -fast_log_positive(-fast_log_positive(unit));
}

__device__ __forceinline__ void sift(uint32_t* order, const double* adjusted,
                                      uint32_t start, uint32_t end) {
  uint32_t root = start;
  while (root * 2 + 1 <= end) {
    uint32_t child = root * 2 + 1;
    if (child + 1 <= end && better(adjusted[order[child + 1]], order[child + 1],
                                   adjusted[order[child]], order[child])) ++child;
    if (!better(adjusted[order[child]], order[child],
                adjusted[order[root]], order[root])) return;
    uint32_t tmp = order[root]; order[root] = order[child]; order[child] = tmp;
    root = child;
  }
}

__device__ __forceinline__ void sort_candidates(uint32_t* order,
                                                  const double* adjusted,
                                                  uint32_t count) {
  if (count < 2) return;
  for (uint32_t start = count / 2; start > 0;) {
    --start;
    sift(order, adjusted, start, count - 1);
  }
  for (uint32_t end = count - 1; end > 0; --end) {
    uint32_t tmp = order[end]; order[end] = order[0]; order[0] = tmp;
    sift(order, adjusted, 0, end - 1);
  }
}

// Masked tokens are excluded from selection and from the logprob normalizer,
// as when a constraint mask is written into the logits.
template <typename Logit>
__device__ __forceinline__ void sample_row(const Logit* row, uint32_t live,
                                            const RowMeta& meta,
                                            const TokenCount* counts,
                                            const TokenBias* biases,
                                            const uint32_t* eos,
                                            const uint32_t* mask,
                                            double* adjusted, uint32_t* order,
                                            RowOutput& output) {
  output.status = 0;
  if (meta.top_logprobs > 20 || live == 0 ||
      !(meta.temperature >= 0.0) || !(meta.top_p > 0.0) ||
      !(meta.min_p >= 0.0)) { output.status = 3; return; }
  double raw_max = -INFINITY;
  for (uint32_t token = 0; token < live; ++token) {
    double value = static_cast<double>(load_logit(row, token));
    if (!isfinite(value)) { output.status = 1; return; }
    if (value > raw_max && admitted(mask, token)) raw_max = value;
  }
  double raw_sum = 0.0;
  uint32_t ranked = 0;
  uint32_t count = 0;
  double largest = -INFINITY;
  uint32_t greedy = 0;
  for (uint32_t token = 0; token < live; ++token) {
    if (!admitted(mask, token)) {
      adjusted[token] = -INFINITY;
      continue;
    }
    double raw = static_cast<double>(load_logit(row, token));
    raw_sum += exp(raw - raw_max);
    if (meta.top_logprobs != 0) {
      uint32_t place = 0;
      while (place < ranked &&
             !better(raw, token,
                     static_cast<double>(load_logit(row, output.top_ids[place])),
                     output.top_ids[place])) ++place;
      if (place < meta.top_logprobs) {
        if (ranked < meta.top_logprobs) ++ranked;
        for (uint32_t j = ranked - 1; j > place; --j)
          output.top_ids[j] = output.top_ids[j - 1];
        output.top_ids[place] = token;
      }
    }
    if (meta.mask_eos && is_eos(eos + meta.eos_offset, meta.eos_len, token)) {
      adjusted[token] = -INFINITY;
      continue;
    }
    TokenCount seen = find_count(counts + meta.counts_offset,
                                 meta.counts_len, token);
    double value = raw + find_bias(biases + meta.biases_offset,
                                   meta.biases_len, token);
    if (seen.prompt_seen || seen.output_count) {
      if (value > 0.0) value /= meta.repetition_penalty;
      else value *= meta.repetition_penalty;
    }
    value -= meta.frequency_penalty * seen.output_count;
    if (seen.output_count) value -= meta.presence_penalty;
    if (value > largest) { largest = value; greedy = token; }
    if (meta.temperature != 0.0) value /= meta.temperature;
    adjusted[token] = value;
    order[count++] = token;
  }
  if (count == 0 || !isfinite(raw_sum) || raw_sum <= 0.0) {
    output.status = 2; return;
  }
  double raw_log_sum = log(raw_sum);
  for (uint32_t i = 0; i < ranked; ++i) {
    uint32_t token = output.top_ids[i];
    output.top_logprobs[i] =
        static_cast<double>(load_logit(row, token)) - raw_max - raw_log_sum;
  }
  if (meta.temperature == 0.0) {
    output.token_id = greedy;
  } else {
    double max_adjusted = -INFINITY;
    for (uint32_t i = 0; i < count; ++i)
      if (adjusted[order[i]] > max_adjusted)
        max_adjusted = adjusted[order[i]];
    if (meta.min_p > 0.0) {
      double cutoff = log(meta.min_p);
      uint32_t kept = 0;
      for (uint32_t i = 0; i < count; ++i) {
        uint32_t token = order[i];
        if (adjusted[token] - max_adjusted >= cutoff) order[kept++] = token;
      }
      count = kept;
    }
    sort_candidates(order, adjusted, count);
    uint32_t start = meta.top_k && count > meta.top_k ? count - meta.top_k : 0;
    uint32_t accepted = count - start;
    if (meta.top_p < 1.0) {
      double total = 0.0;
      // The sorted list is worst to best; the best candidate is at its end.
      double max_kept = adjusted[order[count - 1]];
      for (uint32_t i = count; i > start; --i)
        total += exp(adjusted[order[i - 1]] - max_kept);
      double prefix = 0.0;
      accepted = 0;
      for (uint32_t i = count; i > start; --i) {
        prefix += exp(adjusted[order[i - 1]] - max_kept);
        ++accepted;
        if (prefix >= meta.top_p * total) break;
      }
    }
    double best = -INFINITY;
    uint32_t chosen = order[count - 1];
    for (uint32_t i = count; i > count - accepted; --i) {
      uint32_t token = order[i - 1];
      double score = (adjusted[token] - max_adjusted) +
                     gumbel(meta.seed, meta.generated, token);
      if (score > best || (score == best && token < chosen)) {
        best = score; chosen = token;
      }
    }
    output.token_id = chosen;
  }
  output.logprob = static_cast<double>(load_logit(row, output.token_id)) -
                   raw_max - raw_log_sum;
}

template <typename Logit>
__device__ __forceinline__ void simple_row(const Logit* row, uint32_t live,
                                            const RowMeta& meta, RowOutput& output) {
  uint32_t lane = threadIdx.x;
  double maximum = -INFINITY;
  double best = -INFINITY;
  uint32_t chosen = 0xffffffffU;
  bool invalid = false;
  for (uint32_t token = lane; token < live; token += 32) {
    double value = static_cast<double>(load_logit(row, token));
    invalid |= !isfinite(value);
    if (value > maximum) maximum = value;
  }
  uint32_t bad = __ballot_sync(0xffffffffU, invalid);
  for (int delta = 16; delta > 0; delta >>= 1)
    maximum = fmax(maximum, __shfl_down_sync(0xffffffffU, maximum, delta));
  maximum = __shfl_sync(0xffffffffU, maximum, 0);
  double sum = 0.0;
  for (uint32_t token = lane; token < live; token += 32) {
    double value = static_cast<double>(load_logit(row, token));
    sum += exp(value - maximum);
    double score = meta.temperature == 0.0 ? value :
                   (value - maximum) / meta.temperature +
                   gumbel(meta.seed, meta.generated, token);
    if (better(score, token, best, chosen)) { best = score; chosen = token; }
  }
  for (int delta = 16; delta > 0; delta >>= 1) {
    double other_best = __shfl_down_sync(0xffffffffU, best, delta);
    uint32_t other_id = __shfl_down_sync(0xffffffffU, chosen, delta);
    if (better(other_best, other_id, best, chosen)) {
      best = other_best; chosen = other_id;
    }
    sum += __shfl_down_sync(0xffffffffU, sum, delta);
  }
  if (lane == 0) {
    if (bad || !isfinite(sum) || sum <= 0.0) {
      output.status = 1;
      return;
    }
    output.status = 0;
    output.token_id = chosen;
    output.logprob = static_cast<double>(load_logit(row, chosen)) -
                     maximum - log(sum);
  }
}

static constexpr uint32_t SIMPLE_TILE = 1024;

template <typename Logit>
__device__ __forceinline__ float simple_term(
    const Logit* values, uint32_t token, uint32_t tile, double tile_max,
    const RowMeta& settings, const float* approx_scores,
    float tile_approx_best, double& best, uint32_t& best_id) {
  double raw = static_cast<double>(load_logit(values, token));
  if (!isfinite(raw)) return 0.0f;
  if (settings.temperature != 0.0) {
    float approximate = approx_scores[token - tile * SIMPLE_TILE];
    float margin = 0.1f + 1.0e-5f * fabsf(tile_approx_best);
    if (!isfinite(approximate) || !isfinite(tile_approx_best) ||
        approximate >= tile_approx_best - margin) {
      double score = (raw - tile_max) / settings.temperature +
                     gumbel(settings.seed, settings.generated, token);
      if (better(score, token, best, best_id)) {
        best = score;
        best_id = token;
      }
    }
  }
  return __expf(static_cast<float>(raw - tile_max));
}

template <typename Logit, uint32_t Tile>
__device__ __forceinline__ void greedy_partials(
    const Logit* logits, uint64_t stride, uint32_t live, uint32_t rows,
    const RowMeta* meta, SimplePartial* partials);

// Sorts one 1024-token tile's raw logits by descending value and ascending
// token ID with 256 threads, then stores the first `count` IDs.
template <typename Logit>
__device__ __forceinline__ void tile_top_ids(const Logit* values, uint32_t tile,
                                             uint32_t live, uint32_t count,
                                             uint32_t* top_ids,
                                             const uint32_t* mask = nullptr) {
  uint32_t tid = threadIdx.x;
  __shared__ uint64_t keys[SIMPLE_TILE];
  for (uint32_t i = 0; i < 4; ++i) {
    uint32_t token = tile * SIMPLE_TILE + tid * 4 + i;
    uint64_t key = 0;
    if (token < live && admitted(mask, token)) {
      float raw = load_logit(values, token);
      if (isfinite(raw)) {
        uint32_t bits = raw == 0.0f ? 0 : __float_as_uint(raw);
        uint32_t ordered = (bits & 0x80000000U) ? ~bits :
                           (bits ^ 0x80000000U);
        key = (static_cast<uint64_t>(ordered) << 32) |
              (0xffffffffU - token);
      }
    }
    keys[tid * 4 + i] = key;
  }
  __syncthreads();
  for (uint32_t size = 2; size <= SIMPLE_TILE; size <<= 1) {
    for (uint32_t half = size >> 1; half != 0; half >>= 1) {
      for (uint32_t part = 0; part < 2; ++part) {
        uint32_t pair = tid * 2 + part;
        uint32_t first = (pair & (half - 1)) |
                         ((pair & ~(half - 1)) << 1);
        uint32_t second = first | half;
        uint64_t a = keys[first], b = keys[second];
        bool descending = (first & size) == 0;
        if ((descending && a < b) || (!descending && a > b)) {
          keys[first] = b;
          keys[second] = a;
        }
      }
      __syncthreads();
    }
  }
  if (tid < count) top_ids[tid] = ~static_cast<uint32_t>(keys[tid]);
}


template <typename Logit, bool Top = false>
__device__ __forceinline__ void simple_partials(
    const Logit* logits, uint64_t stride, uint32_t live, uint32_t rows,
    const RowMeta* meta, SimplePartial* partials) {
  if (threadIdx.x == 0) chop::ptx::griddep_launch_dependents();
  chop::ptx::griddep_wait();
  uint32_t row = blockIdx.y;
  if (row >= rows) return;
  uint32_t tile = blockIdx.x;
  uint32_t tiles = (live + SIMPLE_TILE - 1) / SIMPLE_TILE;
  uint32_t tid = threadIdx.x;
  uint32_t lane = tid & 31;
  uint32_t warp = tid >> 5;
  uint32_t warps = blockDim.x >> 5;
  const Logit* values = logits + static_cast<uint64_t>(row) * stride;
  const RowMeta& settings = meta[row];
  if (settings.temperature == 0.0 && settings.top_logprobs <= 1) {
    greedy_partials<Logit, SIMPLE_TILE>(logits, stride, live, rows, meta, partials);
    return;
  }
  float inverse_temperature = settings.temperature == 0.0 ? 0.0f :
                              static_cast<float>(1.0 / settings.temperature);
  double local_max = -INFINITY;
  uint32_t local_max_id = 0xffffffffU;
  float approx_best = -INFINITY;
  bool invalid = false;
  uint32_t begin = tile * SIMPLE_TILE + tid;
  uint32_t end = min((tile + 1) * SIMPLE_TILE, live);
  __shared__ float approx_scores[SIMPLE_TILE], tile_approx_best;
  for (uint32_t token = begin; token < end; token += blockDim.x) {
    double raw = static_cast<double>(load_logit(values, token));
    if (!isfinite(raw)) {
      invalid = true;
      continue;
    }
    if (better(raw, token, local_max, local_max_id)) {
      local_max = raw;
      local_max_id = token;
    }
  }
  uint32_t bad = __ballot_sync(0xffffffffU, invalid);
  for (int delta = 16; delta > 0; delta >>= 1) {
    double other_max = __shfl_down_sync(0xffffffffU, local_max, delta);
    uint32_t other_max_id = __shfl_down_sync(0xffffffffU, local_max_id, delta);
    if (better(other_max, other_max_id, local_max, local_max_id)) {
      local_max = other_max;
      local_max_id = other_max_id;
    }
  }
  __shared__ double warp_max[32], warp_best[32], warp_sum[32], tile_max;
  __shared__ float warp_approx_best[32];
  __shared__ uint32_t warp_id[32], warp_raw_id[32], warp_bad[32];
  if (lane == 0) {
    warp_max[warp] = local_max;
    warp_raw_id[warp] = local_max_id;
    warp_bad[warp] = bad != 0;
  }
  __syncthreads();
  if (tid == 0) {
    double maximum = -INFINITY;
    uint32_t maximum_id = 0xffffffffU;
    uint32_t any_bad = 0;
    for (uint32_t i = 0; i < warps; ++i) {
      if (better(warp_max[i], warp_raw_id[i], maximum, maximum_id)) {
        maximum = warp_max[i];
        maximum_id = warp_raw_id[i];
      }
      any_bad |= warp_bad[i];
    }
    tile_max = maximum;
    SimplePartial& partial = partials[static_cast<uint64_t>(row) * tiles + tile];
    partial.raw_max = maximum;
    partial.top_ids[0] = maximum_id;
    partial.invalid = any_bad;
    if (settings.temperature == 0.0) {
      partial.best_score = maximum;
      partial.token_id = maximum_id;
    }
  }
  __syncthreads();
  if (settings.temperature != 0.0) {
    for (uint32_t token = begin; token < end; token += blockDim.x) {
      double raw = static_cast<double>(load_logit(values, token));
      float score = static_cast<float>(raw - tile_max) * inverse_temperature +
                    approximate_gumbel(settings.seed, settings.generated, token);
      if (!isfinite(score)) score = NAN;
      approx_scores[token - tile * SIMPLE_TILE] = score;
      approx_best = fmaxf(approx_best, score);
    }
    for (int delta = 16; delta > 0; delta >>= 1)
      approx_best = fmaxf(approx_best, __shfl_down_sync(0xffffffffU, approx_best, delta));
    if (lane == 0) warp_approx_best[warp] = approx_best;
    __syncthreads();
    if (tid == 0) {
      float winner = -INFINITY;
      for (uint32_t i = 0; i < warps; ++i) winner = fmaxf(winner, warp_approx_best[i]);
      tile_approx_best = winner;
    }
    __syncthreads();
  }
  float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
  double best = -INFINITY;
  uint32_t best_id = 0xffffffffU;
  for (uint32_t token = begin; token < end; token += blockDim.x * 4) {
    sum0 += simple_term(values, token, tile, tile_max, settings,
                        approx_scores, tile_approx_best, best, best_id);
    if (token + blockDim.x < end)
      sum1 += simple_term(values, token + blockDim.x, tile, tile_max,
                          settings, approx_scores, tile_approx_best, best, best_id);
    if (token + blockDim.x * 2 < end)
      sum2 += simple_term(values, token + blockDim.x * 2, tile, tile_max,
                          settings, approx_scores, tile_approx_best, best, best_id);
    if (token + blockDim.x * 3 < end)
      sum3 += simple_term(values, token + blockDim.x * 3, tile, tile_max,
                          settings, approx_scores, tile_approx_best, best, best_id);
  }
  double sum = static_cast<double>(sum0) + static_cast<double>(sum1) +
               static_cast<double>(sum2) + static_cast<double>(sum3);
  if (settings.temperature != 0.0) {
    for (int delta = 16; delta > 0; delta >>= 1) {
      double other_best = __shfl_down_sync(0xffffffffU, best, delta);
      uint32_t other_id = __shfl_down_sync(0xffffffffU, best_id, delta);
      if (better(other_best, other_id, best, best_id)) {
        best = other_best;
        best_id = other_id;
      }
    }
  }
  if (lane == 0 && settings.temperature != 0.0) {
    warp_best[warp] = best;
    warp_id[warp] = best_id;
  }
  for (int delta = 16; delta > 0; delta >>= 1)
    sum += __shfl_down_sync(0xffffffffU, sum, delta);
  if (lane == 0) warp_sum[warp] = sum;
  __syncthreads();
  if (tid == 0) {
    double total = 0.0;
    double winner = -INFINITY;
    uint32_t chosen = 0xffffffffU;
    for (uint32_t i = 0; i < warps; ++i) {
      total += warp_sum[i];
      if (settings.temperature != 0.0 &&
          better(warp_best[i], warp_id[i], winner, chosen)) {
        winner = warp_best[i]; chosen = warp_id[i];
      }
    }
    SimplePartial& partial = partials[static_cast<uint64_t>(row) * tiles + tile];
    partial.raw_sum = total;
    if (settings.temperature != 0.0) {
      partial.best_score = winner;
      partial.token_id = chosen;
    }
  }
  if constexpr (Top) {
    if (settings.top_logprobs > 1)
      tile_top_ids(values, tile, live, settings.top_logprobs,
                   partials[static_cast<uint64_t>(row) * tiles + tile].top_ids);
  }
}

// Raw maximum, lowest maximal ID, max-subtracted sum and invalid flag for
// one tile. The raw maximum is also stored as the tile's greedy choice, and
// the tile's raw logits remain in `raw_cache`, with -inf for tokens outside
// `mask`, which take no part in the maximum or sum.
template <typename Logit, uint32_t Tile>
__device__ __forceinline__ void raw_tile_stats(const Logit* values,
                                               uint32_t begin, uint32_t end,
                                               SimplePartial& partial,
                                               float* raw_cache,
                                               const uint32_t* mask = nullptr) {
  uint32_t tid = threadIdx.x;
  uint32_t lane = tid & 31;
  uint32_t warp = tid >> 5;
  uint32_t warps = blockDim.x >> 5;
  __shared__ float warp_max[32], tile_max;
  __shared__ double warp_sum[32];
  __shared__ uint32_t warp_id[32], warp_bad[32];
  float best = -INFINITY;
  uint32_t best_id = 0xffffffffU;
  bool invalid = false;
  for (uint32_t token = begin + tid; token < end; token += blockDim.x) {
    float raw = load_logit(values, token);
    if (!isfinite(raw)) invalid = true;
    if (!admitted(mask, token)) raw = -INFINITY;
    raw_cache[token - begin] = raw;
    if (!isfinite(raw)) continue;
    if (raw > best || (raw == best && token < best_id)) {
      best = raw;
      best_id = token;
    }
  }
  uint32_t bad = __ballot_sync(0xffffffffU, invalid);
  for (int delta = 16; delta > 0; delta >>= 1) {
    float other = __shfl_down_sync(0xffffffffU, best, delta);
    uint32_t other_id = __shfl_down_sync(0xffffffffU, best_id, delta);
    if (other > best || (other == best && other_id < best_id)) {
      best = other;
      best_id = other_id;
    }
  }
  if (lane == 0) {
    warp_max[warp] = best;
    warp_id[warp] = best_id;
    warp_bad[warp] = bad != 0;
  }
  __syncthreads();
  if (tid == 0) {
    float maximum = -INFINITY;
    uint32_t chosen = 0xffffffffU;
    uint32_t any_bad = 0;
    for (uint32_t i = 0; i < warps; ++i) {
      if (warp_max[i] > maximum || (warp_max[i] == maximum && warp_id[i] < chosen)) {
        maximum = warp_max[i];
        chosen = warp_id[i];
      }
      any_bad |= warp_bad[i];
    }
    tile_max = maximum;
    partial.raw_max = maximum;
    partial.best_score = maximum;
    partial.token_id = chosen;
    partial.top_ids[0] = chosen;
    partial.invalid = any_bad;
  }
  __syncthreads();
  // A masked token adds nothing, even when the whole tile is masked.
  auto term = [&](uint32_t offset) {
    float raw = raw_cache[offset];
    return raw == -INFINITY ? 0.0f : __expf(raw - tile_max);
  };
  float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
  for (uint32_t offset = tid; offset < Tile; offset += blockDim.x * 4) {
    if (begin + offset < end) sum0 += term(offset);
    if (begin + offset + blockDim.x < end) sum1 += term(offset + blockDim.x);
    if (begin + offset + blockDim.x * 2 < end) sum2 += term(offset + blockDim.x * 2);
    if (begin + offset + blockDim.x * 3 < end) sum3 += term(offset + blockDim.x * 3);
  }
  double sum = static_cast<double>(sum0) + static_cast<double>(sum1) +
               static_cast<double>(sum2) + static_cast<double>(sum3);
  for (int delta = 16; delta > 0; delta >>= 1)
    sum += __shfl_down_sync(0xffffffffU, sum, delta);
  if (lane == 0) warp_sum[warp] = sum;
  __syncthreads();
  if (tid == 0) {
    double total = 0.0;
    for (uint32_t i = 0; i < warps; ++i) total += warp_sum[i];
    partial.raw_sum = total;
  }
}

template <typename Logit, uint32_t Tile>
__device__ __forceinline__ void greedy_partials(
    const Logit* logits, uint64_t stride, uint32_t live, uint32_t rows,
    const RowMeta* meta, SimplePartial* partials) {
  if (threadIdx.x == 0) chop::ptx::griddep_launch_dependents();
  chop::ptx::griddep_wait();
  uint32_t row = blockIdx.y;
  if (row >= rows) return;
  uint32_t tile = blockIdx.x;
  uint32_t tiles = (live + Tile - 1) / Tile;
  uint32_t begin = tile * Tile;
  __shared__ float raw_cache[Tile];
  raw_tile_stats<Logit, Tile>(logits + static_cast<uint64_t>(row) * stride,
                              begin, min(begin + Tile, live),
                              partials[static_cast<uint64_t>(row) * tiles + tile],
                              raw_cache);
}

// Filtered rows may choose a processed winner that is not the raw maximum,
// so they always merge raw top IDs instead of reusing the greedy choice.
template <typename Logit, uint32_t Tile = SIMPLE_TILE, bool Filtered = false>
__device__ __forceinline__ void simple_finish(
    const Logit* logits, uint64_t stride, uint32_t live, uint32_t rows,
    const RowMeta* meta, const SimplePartial* partials, RowOutput* output) {
  if (threadIdx.x == 0) chop::ptx::griddep_launch_dependents();
  chop::ptx::griddep_wait();
  uint32_t row = blockIdx.x;
  if (row >= rows) return;
  uint32_t lane = threadIdx.x;
  uint32_t tiles = (live + Tile - 1) / Tile;
  const SimplePartial* pieces = partials + static_cast<uint64_t>(row) * tiles;
  double maximum = -INFINITY;
  double best = -INFINITY;
  uint32_t chosen = 0xffffffffU;
  uint32_t invalid = 0;
  for (uint32_t tile = lane; tile < tiles; tile += 32) {
    maximum = fmax(maximum, pieces[tile].raw_max);
    invalid |= pieces[tile].invalid;
  }
  for (int delta = 16; delta > 0; delta >>= 1) {
    maximum = fmax(maximum, __shfl_down_sync(0xffffffffU, maximum, delta));
    invalid |= __shfl_down_sync(0xffffffffU, invalid, delta);
  }
  maximum = __shfl_sync(0xffffffffU, maximum, 0);
  double sum = 0.0;
  for (uint32_t tile = lane; tile < tiles; tile += 32) {
    SimplePartial piece = pieces[tile];
    double score = piece.best_score;
    if (!Filtered && meta[row].temperature != 0.0)
      score += (piece.raw_max - maximum) / meta[row].temperature;
    if (better(score, piece.token_id, best, chosen)) {
      best = score; chosen = piece.token_id;
    }
    sum += piece.raw_sum * static_cast<double>(
        __expf(static_cast<float>(piece.raw_max - maximum)));
  }
  for (int delta = 16; delta > 0; delta >>= 1) {
    double other_best = __shfl_down_sync(0xffffffffU, best, delta);
    uint32_t other_id = __shfl_down_sync(0xffffffffU, chosen, delta);
    if (better(other_best, other_id, best, chosen)) {
      best = other_best; chosen = other_id;
    }
    sum += __shfl_down_sync(0xffffffffU, sum, delta);
  }
  __shared__ double raw_log_sum;
  if (lane == 0) {
    RowOutput& result = output[row];
    // A valid row without a candidate (every admitted token masked as EOS,
    // or no admitted token) fails only its request.
    if (invalid || meta[row].top_logprobs > 20) {
      result.status = 1;
    } else if (chosen >= live) {
      result.status = 2;
    } else if (!isfinite(sum) || sum <= 0.0) {
      result.status = 1;
    } else {
      raw_log_sum = log(sum);
      result.status = 0;
      result.token_id = chosen;
      result.logprob = static_cast<double>(load_logit(
          logits + static_cast<uint64_t>(row) * stride, chosen)) -
          maximum - raw_log_sum;
    }
  }
  __syncwarp();
  uint32_t top_count = min(meta[row].top_logprobs, live);
  if (top_count == 0 || output[row].status != 0) return;
  if (!Filtered && top_count == 1 && meta[row].temperature == 0.0) {
    if (lane == 0) {
      output[row].top_ids[0] = output[row].token_id;
      output[row].top_logprobs[0] = output[row].logprob;
    }
    return;
  }
  __shared__ uint8_t cursor[1024];
  for (uint32_t tile = lane; tile < tiles; tile += 32) cursor[tile] = 0;
  __syncwarp();
  const Logit* values = logits + static_cast<uint64_t>(row) * stride;
  for (uint32_t rank = 0; rank < top_count; ++rank) {
    double candidate = -INFINITY;
    uint32_t candidate_id = 0xffffffffU;
    for (uint32_t tile = lane; tile < tiles; tile += 32) {
      uint32_t next = cursor[tile];
      if (next >= top_count) continue;
      uint32_t token = pieces[tile].top_ids[next];
      if (token >= live) continue;
      double value = static_cast<double>(load_logit(values, token));
      if (better(value, token, candidate, candidate_id)) {
        candidate = value;
        candidate_id = token;
      }
    }
    for (int delta = 16; delta > 0; delta >>= 1) {
      double other = __shfl_down_sync(0xffffffffU, candidate, delta);
      uint32_t other_id = __shfl_down_sync(0xffffffffU, candidate_id, delta);
      if (better(other, other_id, candidate, candidate_id)) {
        candidate = other;
        candidate_id = other_id;
      }
    }
    if (lane == 0 && candidate_id < live) {
      output[row].top_ids[rank] = candidate_id;
      output[row].top_logprobs[rank] = candidate - maximum - raw_log_sum;
      ++cursor[candidate_id / Tile];
    }
    __syncwarp();
  }
}

// `masks` holds `mask_words` words per row when `mask_words` is nonzero.
template <bool Bf16, typename Logit>
__device__ __forceinline__ void run(const Logit* logits, uint64_t stride,
                                    uint32_t live, uint32_t rows,
                                    const RowMeta* meta,
                                    const TokenCount* counts,
                                    const TokenBias* biases,
                                    const uint32_t* eos,
                                    double* adjusted, uint32_t* order,
                                    RowOutput* output, const uint32_t* masks,
                                    uint32_t mask_words) {
  if (threadIdx.x == 0) chop::ptx::griddep_launch_dependents();
  chop::ptx::griddep_wait();
  uint32_t row = blockIdx.x;
  if (row >= rows) return;
  const RowMeta& settings = meta[row];
  const uint32_t* mask =
      mask_words == 0 ? nullptr
                      : masks + static_cast<uint64_t>(row) * mask_words;
  if (mask == nullptr && settings.counts_len == 0 && settings.biases_len == 0 &&
      (!settings.mask_eos || settings.eos_len == 0) &&
      settings.top_k == 0 && settings.top_p == 1.0 &&
      settings.min_p == 0.0 && settings.top_logprobs == 0 &&
      settings.repetition_penalty == 1.0 &&
      settings.frequency_penalty == 0.0 &&
      settings.presence_penalty == 0.0 &&
      settings.temperature >= 0.0 && live > 0) {
    simple_row(logits + static_cast<uint64_t>(row) * stride, live,
               settings, output[row]);
    return;
  }
  if (threadIdx.x != 0) return;
  sample_row(logits + static_cast<uint64_t>(row) * stride, live, meta[row],
             counts, biases, eos, mask,
             adjusted + static_cast<uint64_t>(row) * live,
             order + static_cast<uint64_t>(row) * live, output[row]);
}

}  // namespace chop::sampling
