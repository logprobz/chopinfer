#pragma once

// Tiled sampling for rows with penalties, logit bias, EOS masking, min-p,
// top-k or top-p. The kept set is found without sorting the vocabulary:
//
// 1. stats: raw logprob partials and each tile's largest processed logit.
// 2. histogram: survivors of min-p are binned by distance from the maximum.
//    The last tile picks the bins that can hold the top-k and top-p cutoffs.
// 3. collect: exact processed values and IDs of tokens in those bins are
//    appended to the row workspace.
// 4. resolve: one block per row sorts the candidates and evaluates the cutoff
//    in f64. A top-p cutoff is accepted only if the prefix sums on both sides
//    are farther from the threshold than a bound on this evaluation's and
//    the CPU oracle's rounding error. Otherwise all survivors are sorted and
//    the oracle's sequential algorithm runs.
// 5. gumbel: kept tokens compete with the Philox Gumbel score of the simple
//    path; `simple_finish` then merges tiles and raw logprobs.
//
// FP64 runs at 1/64 of the FP32 rate on SM120, so per-token work uses an f32
// key: `RN32(RN64(value * T))`, with T taken as 1 for greedy rows. Rounding
// is monotone, so a larger processed value never has a smaller key, and a
// token without penalty or bias has its raw logit as key because
// `RN32(RN64(RN64(raw / T) * T)) == raw`. Exact f64 values are computed only
// where keys tie or an f32 estimate is too close to a decision.
//
// Every kernel runs after `griddepcontrol.wait`, so each sees the previous
// kernel's writes; the stats kernel resets the per-row records.

#include "sampling/sampling.cuh"

namespace chop::sampling {

static constexpr uint32_t FILTER_BINS = 2048;
static constexpr float FILTER_BINS_PER_NAT = 32.0f;
// Bins below 26 nats use f64 masses; beyond, f32 masses err by less than
// 2^-53 each relative to the maximum's unit mass.
static constexpr uint32_t FILTER_NEAR_BINS = 26 * 32;
static constexpr uint32_t FILTER_SHARED_SORT = 2048;

enum : uint32_t { FILTER_KEEP_SURVIVORS = 0, FILTER_WINDOW = 1, FILTER_THRESHOLD = 2 };

struct FilterBin {
  uint32_t count;
  uint32_t reserved;
  double mass;
};
struct FilterState {
  uint32_t hist_ticket;
  uint32_t reserved_ticket;
  uint32_t candidates;
  uint32_t mode;
  uint32_t bin_lo;
  uint32_t bin_hi;
  uint32_t bin_k;
  uint32_t threshold_id;
  uint32_t survivors;
  uint32_t exact_fallback;
  uint32_t head_count;
  uint32_t k_head_count;
  uint32_t window_count;
  uint32_t k_count;
  uint32_t reserved[2];
  double maximum;
  double threshold;
  double head_mass;
  double k_head_mass;
  double tail_mass;
  double reserved_mass[3];
};
struct FilterRow {
  FilterState state;
  FilterBin bins[FILTER_BINS];
};

static_assert(sizeof(FilterBin) == 16 && sizeof(FilterState) == 128 &&
              sizeof(FilterRow) == 128 + 16 * FILTER_BINS,
              "filtered sampling records must match the planner");

template <typename Entry>
__device__ __forceinline__ uint32_t lower_token(const Entry* entries,
                                                uint32_t length, uint32_t token) {
  uint32_t lo = 0, hi = length;
  while (lo < hi) {
    uint32_t mid = lo + (hi - lo) / 2;
    if (entries[mid].token_id < token) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

// Penalties for a seen token, in the oracle's order. The explicit intrinsics
// keep the frequency product and subtraction separately rounded.
__device__ __forceinline__ double penalized(double value, const TokenCount& seen,
                                            const RowMeta& settings) {
  if (value > 0.0) value = __ddiv_rn(value, settings.repetition_penalty);
  else value = __dmul_rn(value, settings.repetition_penalty);
  value = __dsub_rn(value, __dmul_rn(settings.frequency_penalty,
                                     static_cast<double>(seen.output_count)));
  if (seen.output_count) value = __dsub_rn(value, settings.presence_penalty);
  return value;
}

// Per-row constants for keys and bins.
struct RowScale {
  double temperature;  // 1 for greedy rows
  double inverse;
  float inverse32;
  float bin_scale;     // bins per key unit
  bool divide;         // temperature is neither 0 nor 1
  double log_min_p;
  float log_min_p32;
};

__device__ __forceinline__ RowScale row_scale(const RowMeta& settings) {
  RowScale scale;
  scale.temperature = settings.temperature == 0.0 ? 1.0 : settings.temperature;
  scale.inverse = 1.0 / scale.temperature;
  scale.inverse32 = __double2float_rn(scale.inverse);
  scale.bin_scale = __double2float_rn(FILTER_BINS_PER_NAT / scale.temperature);
  scale.divide = settings.temperature != 0.0 && settings.temperature != 1.0;
  scale.log_min_p = settings.min_p > 0.0 ? log(settings.min_p) : -INFINITY;
  scale.log_min_p32 = __double2float_rn(scale.log_min_p);
  return scale;
}

// Computes the row constants once per block; every thread must call this.
__device__ __forceinline__ RowScale shared_row_scale(const RowMeta& settings) {
  __shared__ RowScale scale;
  if (threadIdx.x == 0) scale = row_scale(settings);
  __syncthreads();
  return scale;
}

__device__ __forceinline__ float processed_key(double value, const RowScale& scale) {
  return __double2float_rn(__dmul_rn(value, scale.temperature));
}

// One tile's keys (NaN for masked EOS tokens and non-finite raw logits) and
// exact values of the tokens that have a penalty or bias.
struct TileView {
  float key[SIMPLE_TILE];
  double exact[SIMPLE_TILE];
  uint8_t adjusted[SIMPLE_TILE];
};

// The oracle's processed value of a token with a count entry, or NaN for a
// non-finite raw logit.
template <typename Logit>
__device__ __forceinline__ double counted_value(const Logit* values,
                                                const TokenCount& entry,
                                                const RowMeta& settings,
                                                const TokenBias* bias) {
  double value = static_cast<double>(load_logit(values, entry.token_id));
  if (!isfinite(value)) return NAN;
  uint32_t b = lower_token(bias, settings.biases_len, entry.token_id);
  if (b < settings.biases_len && bias[b].token_id == entry.token_id)
    value = __dadd_rn(value, bias[b].bias);
  value = penalized(value, entry, settings);
  return settings.temperature != 0.0 ? __ddiv_rn(value, settings.temperature) : value;
}

// The same for a biased token without a count entry.
template <typename Logit>
__device__ __forceinline__ double biased_value(const Logit* values,
                                               const TokenBias& entry,
                                               const RowMeta& settings) {
  double value = static_cast<double>(load_logit(values, entry.token_id));
  if (!isfinite(value)) return NAN;
  value = __dadd_rn(value, entry.bias);
  return settings.temperature != 0.0 ? __ddiv_rn(value, settings.temperature) : value;
}

__device__ __forceinline__ bool has_count(const TokenCount* seen, uint32_t length,
                                          uint32_t token) {
  uint32_t c = lower_token(seen, length, token);
  return c < length && seen[c].token_id == token;
}

template <typename Logit>
__device__ __forceinline__ void tile_view(
    const Logit* values, uint32_t begin, uint32_t end, const RowMeta& settings,
    const RowScale& scale, const TokenCount* counts, const TokenBias* biases,
    const uint32_t* eos, const uint32_t* mask, TileView& view) {
  uint32_t tid = threadIdx.x;
  for (uint32_t token = begin + tid; token < end; token += blockDim.x) {
    float raw = load_logit(values, token);
    view.key[token - begin] = isfinite(raw) ? raw : NAN;
    view.adjusted[token - begin] = 0;
  }
  __syncthreads();
  const TokenCount* seen = counts + settings.counts_offset;
  const TokenBias* bias = biases + settings.biases_offset;
  uint32_t first = lower_token(seen, settings.counts_len, begin);
  uint32_t last = lower_token(seen, settings.counts_len, end);
  for (uint32_t i = first + tid; i < last; i += blockDim.x) {
    TokenCount entry = seen[i];
    double value = counted_value(values, entry, settings, bias);
    if (isnan(value)) continue;
    uint32_t at = entry.token_id - begin;
    view.exact[at] = value;
    view.key[at] = processed_key(value, scale);
    view.adjusted[at] = 1;
  }
  first = lower_token(bias, settings.biases_len, begin);
  last = lower_token(bias, settings.biases_len, end);
  for (uint32_t i = first + tid; i < last; i += blockDim.x) {
    TokenBias entry = bias[i];
    if (has_count(seen, settings.counts_len, entry.token_id)) continue;
    double value = biased_value(values, entry, settings);
    if (isnan(value)) continue;
    uint32_t at = entry.token_id - begin;
    view.exact[at] = value;
    view.key[at] = processed_key(value, scale);
    view.adjusted[at] = 1;
  }
  __syncthreads();
  if (settings.mask_eos) {
    for (uint32_t i = tid; i < settings.eos_len; i += blockDim.x) {
      uint32_t token = eos[settings.eos_offset + i];
      if (token >= begin && token < end) view.key[token - begin] = NAN;
    }
  }
  if (mask != nullptr) {
    for (uint32_t token = begin + tid; token < end; token += blockDim.x)
      if (!admitted(mask, token)) view.key[token - begin] = NAN;
  }
  __syncthreads();
}

// The row's constraint mask, or null when the batch has none.
__device__ __forceinline__ const uint32_t* row_mask(const uint32_t* masks,
                                                    uint32_t mask_words,
                                                    uint32_t row) {
  return mask_words == 0 ? nullptr : masks + static_cast<uint64_t>(row) * mask_words;
}

// The oracle's processed value of an unmasked tile token.
__device__ __forceinline__ double exact_value(const TileView& view, uint32_t at,
                                              const RowMeta& settings,
                                              const RowScale& scale) {
  if (view.adjusted[at]) return view.exact[at];
  double raw = static_cast<double>(view.key[at]);
  return scale.divide ? __ddiv_rn(raw, settings.temperature) : raw;
}

// Exact min-p test. The f32 estimate of `value - maximum` errs by at most
// 2^-22 (|key| + |max key|) / T; the band below is far wider, and only
// tokens inside it compute the exact value.
__device__ __forceinline__ bool min_p_survivor(const TileView& view, uint32_t at,
                                               float max_key, double maximum,
                                               const RowMeta& settings,
                                               const RowScale& scale) {
  float key = view.key[at];
  if (isnan(key)) return false;
  if (settings.min_p == 0.0) return true;
  float estimate = __fmul_rn(__fsub_rn(key, max_key), scale.inverse32);
  float band = 1.0e-5f * (fabsf(key) + fabsf(max_key)) * scale.inverse32 +
               1.0e-6f * (1.0f + fabsf(scale.log_min_p32));
  if (estimate > scale.log_min_p32 + band) return true;
  if (estimate < scale.log_min_p32 - band) return false;
  return exact_value(view, at, settings, scale) - maximum >= scale.log_min_p;
}

// Monotone in the key, so a larger processed value never has a later bin.
__device__ __forceinline__ uint32_t filter_bin(float key, float max_key,
                                               const RowScale& scale) {
  float scaled = __fmul_rn(__fsub_rn(max_key, key), scale.bin_scale);
  return scaled >= static_cast<float>(FILTER_BINS - 1) ?
      FILTER_BINS - 1 : static_cast<uint32_t>(scaled);
}

// exp(x) for x <= 0 with relative error near 1e-15, or zero below -64:
// x = k ln2/32 + r, exp(x) = 2^(k/32) * p(r) with a degree-5 polynomial on
// |r| <= 0.011. Masses need only be within the certification margin;
// `powers` holds 2^(j/32) for j < 32.
__device__ __forceinline__ double mass_exp(double x, const double* powers) {
  if (x < -64.0) return 0.0;
  int k = __float2int_rn(__double2float_rn(x) * 46.16624130844683f);
  double scaled = static_cast<double>(k);
  double r = fma(-scaled, 0x1.62e42fefa3000p-6, x);
  r = fma(-scaled, 0x1.3de6af278ece6p-47, r);
  double p = 1.0 / 120.0;
  p = fma(p, r, 1.0 / 24.0);
  p = fma(p, r, 1.0 / 6.0);
  p = fma(p, r, 0.5);
  p = fma(p, r, 1.0);
  p = fma(p, r, 1.0);
  double v = p * powers[k & 31];
  return __hiloint2double(__double2hiint(v) + ((k >> 5) << 20), __double2loint(v));
}

// Error-free sum of two floats: hi + lo == a + b exactly.
__device__ __forceinline__ void two_sum(float a, float b, float& hi, float& lo) {
  hi = __fadd_rn(a, b);
  float bb = __fsub_rn(hi, a);
  lo = __fadd_rn(__fsub_rn(a, __fsub_rn(hi, bb)), __fsub_rn(b, bb));
}

// The same when |a| >= |b|.
__device__ __forceinline__ void fast_two_sum(float a, float b, float& hi, float& lo) {
  hi = __fadd_rn(a, b);
  lo = __fsub_rn(b, __fsub_rn(hi, a));
}

// round(exp(key - max_key) * 2^52) for f32 keys at most 26 nats below
// max_key, in f32 float-float arithmetic (FP64 runs at 1/64 of the FP32 rate).
// The argument is exact as a two-sum; x = k ln2/32 + r with ln2/32 split in
// three parts so that r errs by less than 1e-16. exp(r) and 2^(j/32) carry
// about 44 bits: the result is within 1e-13 relative plus 0.5 + 2^-23 units
// of the exact product. `powers` holds 2^(j/32) as float-float pairs.
__device__ __forceinline__ unsigned long long mass_fixed_ff(float key, float max_key,
                                                            const float2* powers) {
  float xh, xl;
  two_sum(key, -max_key, xh, xl);
  int k = __float2int_rn(__fmul_rn(xh, 46.16624130844683f));
  float kf = static_cast<float>(k);
  // kf * C1 is exact (|k| < 2^11, C1 has 12 bits) and so is the subtraction
  // (Sterbenz): |xh - kf * C1| < 0.011.
  float a = __fsub_rn(xh, __fmul_rn(kf, 0x1.62ep-6f));
  float p2 = __fmul_rn(kf, 0x1.0bfbe8p-20f);
  float p2_error = __fmaf_rn(kf, 0x1.0bfbe8p-20f, -p2);
  float s, s_error, rh, r_error;
  two_sum(a, xl, s, s_error);
  two_sum(s, -p2, rh, r_error);
  float rl = __fsub_rn(__fsub_rn(__fadd_rn(s_error, r_error), p2_error),
                       __fmul_rn(kf, 0x1.cf79acp-45f));
  fast_two_sum(rh, rl, rh, rl);
  // exp(r) = 1 + r + r^2/2 + r^2 u, u = r/6 + r^2/24 + r^3/120; the omitted
  // r^6/720 is below 2.3e-15.
  float square = __fmul_rn(rh, rh);
  float square_error = __fadd_rn(__fmaf_rn(rh, rh, -square), __fmul_rn(2.0f, __fmul_rn(rh, rl)));
  float u = __fmul_rn(rh, __fmaf_rn(rh, __fmaf_rn(rh, 1.0f / 120.0f, 1.0f / 24.0f), 1.0f / 6.0f));
  float cubic = __fmul_rn(square, u);
  float cubic_error = __fmaf_rn(square, u, -cubic);
  float eh, el, t, t_error;
  two_sum(1.0f, rh, eh, el);
  two_sum(eh, __fmul_rn(0.5f, square), eh, t);
  el = __fadd_rn(el, t);
  two_sum(eh, cubic, eh, t_error);
  el = __fadd_rn(__fadd_rn(el, t_error),
                 __fadd_rn(__fadd_rn(rl, __fmul_rn(0.5f, square_error)), cubic_error));
  fast_two_sum(eh, el, eh, el);
  // Times 2^(j/32) and 2^(k >> 5); k >= -1201 keeps both parts normal.
  float2 power = powers[k & 31];
  float mh = __fmul_rn(eh, power.x);
  float ml = __fadd_rn(__fmaf_rn(eh, power.x, -mh),
                       __fadd_rn(__fmul_rn(eh, power.y), __fmul_rn(el, power.x)));
  float scale = __int_as_float((127 + 52 + (k >> 5)) << 23);
  float high = __fmul_rn(mh, scale);
  float low = __fmul_rn(ml, scale);
  // high - floor(high) is exact; the sum with `low` stays below 2 in
  // magnitude when high has a fraction.
  float whole = floorf(high);
  return __float2ull_rz(whole) +
         static_cast<long long>(__float2ll_rn(__fadd_rn(__fsub_rn(high, whole), low)));
}

__device__ __forceinline__ bool filter_active(const RowMeta& settings) {
  return settings.temperature != 0.0 &&
         (settings.top_p < 1.0 || settings.top_k != 0);
}

// Block-wide maximum; every thread receives the result. The first warp
// combines the warp results, so each thread does a fixed amount of work
// (FP64 runs at 1/64 of the FP32 rate).
template <typename T>
__device__ __forceinline__ T block_max(T value) {
  __shared__ T warp_values[32];
  __shared__ T result;
  for (int delta = 16; delta > 0; delta >>= 1)
    value = max(value, __shfl_xor_sync(0xffffffffU, value, delta));
  if ((threadIdx.x & 31) == 0) warp_values[threadIdx.x >> 5] = value;
  __syncthreads();
  if (threadIdx.x < 32) {
    value = warp_values[threadIdx.x < (blockDim.x >> 5) ? threadIdx.x : 0];
    for (int delta = 16; delta > 0; delta >>= 1)
      value = max(value, __shfl_xor_sync(0xffffffffU, value, delta));
    if (threadIdx.x == 0) result = value;
  }
  __syncthreads();
  value = result;
  __syncthreads();
  return value;
}

// Block-wide f64 sum in a fixed order; every thread receives the result.
__device__ __forceinline__ double block_sum(double value) {
  __shared__ double warp_values[32];
  __shared__ double result;
  for (int delta = 16; delta > 0; delta >>= 1)
    value += __shfl_xor_sync(0xffffffffU, value, delta);
  if ((threadIdx.x & 31) == 0) warp_values[threadIdx.x >> 5] = value;
  __syncthreads();
  if (threadIdx.x < 32) {
    value = threadIdx.x < (blockDim.x >> 5) ? warp_values[threadIdx.x] : 0.0;
    for (int delta = 16; delta > 0; delta >>= 1)
      value += __shfl_xor_sync(0xffffffffU, value, delta);
    if (threadIdx.x == 0) result = value;
  }
  __syncthreads();
  value = result;
  __syncthreads();
  return value;
}

// Inclusive and exclusive block prefix sums of one value per thread.
__device__ __forceinline__ void block_scan(double value, double& inclusive,
                                           double& exclusive, double& total) {
  __shared__ double warp_values[32];
  uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  uint32_t warps = blockDim.x >> 5;
  double scan = value;
  for (int delta = 1; delta < 32; delta <<= 1) {
    double other = __shfl_up_sync(0xffffffffU, scan, delta);
    if (lane >= static_cast<uint32_t>(delta)) scan += other;
  }
  double before = __shfl_up_sync(0xffffffffU, scan, 1);
  if (lane == 0) before = 0.0;
  if (lane == 31) warp_values[warp] = scan;
  __syncthreads();
  if (warp == 0) {
    double carry = lane < warps ? warp_values[lane] : 0.0;
    for (int delta = 1; delta < 32; delta <<= 1) {
      double other = __shfl_up_sync(0xffffffffU, carry, delta);
      if (lane >= static_cast<uint32_t>(delta)) carry += other;
    }
    if (lane < warps) warp_values[lane] = carry;
  }
  __syncthreads();
  double offset = warp == 0 ? 0.0 : warp_values[warp - 1];
  inclusive = offset + scan;
  exclusive = offset + before;
  total = warp_values[warps - 1];
  __syncthreads();
}

// Sorts (value, ID) pairs by descending value and ascending ID with a bitonic
// network whose merges first compare mirrored pairs, so every merge keeps one
// direction. Elements past `count` act as the worst keys and never move
// before a real element, so no padding is stored. Global pairs written by
// other blocks are read from L2.
template <bool Global>
__device__ __forceinline__ void sort_step(double* value, uint32_t* id,
                                          uint32_t count, uint32_t pairs,
                                          uint32_t size, uint32_t half) {
  for (uint32_t pair = threadIdx.x; pair < pairs; pair += blockDim.x) {
    uint32_t first, second;
    if (half == size >> 1) {
      first = (pair / half) * size + pair % half;
      second = (pair / half) * size + size - 1 - pair % half;
    } else {
      first = (pair / half) * 2 * half + pair % half;
      second = first + half;
    }
    if (second >= count) continue;
    double a = Global ? __ldcg(value + first) : value[first];
    double b = Global ? __ldcg(value + second) : value[second];
    uint32_t a_id = Global ? __ldcg(id + first) : id[first];
    uint32_t b_id = Global ? __ldcg(id + second) : id[second];
    if (better(b, b_id, a, a_id)) {
      value[first] = b; value[second] = a;
      id[first] = b_id; id[second] = a_id;
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void sort_pairs(double* value, uint32_t* id,
                                           uint32_t count) {
  uint32_t padded = 1;
  while (padded < count) padded <<= 1;
  for (uint32_t size = 2; size <= padded; size <<= 1)
    for (uint32_t half = size >> 1; half > 0; half >>= 1)
      sort_step<false>(value, id, count, padded / 2, size, half);
}

// The same order for global pairs. Steps whose pairs fall inside one aligned
// chunk of FILTER_SHARED_SORT elements run in the shared `chunk` buffers:
// chunks are first sorted whole, and each later merge makes only its steps
// wider than a chunk in global memory.
__device__ __forceinline__ void sort_global_pairs(double* value, uint32_t* id,
                                                  uint32_t count,
                                                  double* chunk_value,
                                                  uint32_t* chunk_id) {
  constexpr uint32_t CHUNK = FILTER_SHARED_SORT;
  uint32_t padded = 1;
  while (padded < count) padded <<= 1;
  auto in_chunks = [&](uint32_t size) {
    for (uint32_t base = 0; base < count; base += CHUNK) {
      uint32_t n = min(CHUNK, count - base);
      for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        chunk_value[i] = __ldcg(value + base + i);
        chunk_id[i] = __ldcg(id + base + i);
      }
      __syncthreads();
      if (size == 0) {
        sort_pairs(chunk_value, chunk_id, n);
      } else {
        for (uint32_t half = CHUNK / 2; half > 0; half >>= 1)
          sort_step<false>(chunk_value, chunk_id, n, CHUNK / 2, size, half);
      }
      for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        value[base + i] = chunk_value[i];
        id[base + i] = chunk_id[i];
      }
      __syncthreads();
    }
  };
  in_chunks(0);
  for (uint32_t size = 2 * CHUNK; size <= padded; size <<= 1) {
    for (uint32_t half = size >> 1; half >= CHUNK; half >>= 1)
      sort_step<true>(value, id, count, padded / 2, size, half);
    in_chunks(size);
  }
}

template <typename Logit>
__device__ __forceinline__ void filtered_stats(
    const Logit* logits, uint64_t stride, uint32_t live, uint32_t rows,
    const RowMeta* meta, const TokenCount* counts, const TokenBias* biases,
    const uint32_t* eos, SimplePartial* partials, FilterRow* filter,
    const uint32_t* masks, uint32_t mask_words) {
  if (threadIdx.x == 0) chop::ptx::griddep_launch_dependents();
  chop::ptx::griddep_wait();
  uint32_t row = blockIdx.y;
  if (row >= rows) return;
  const uint32_t* mask = row_mask(masks, mask_words, row);
  uint32_t tile = blockIdx.x;
  uint32_t tiles = (live + SIMPLE_TILE - 1) / SIMPLE_TILE;
  uint32_t begin = tile * SIMPLE_TILE;
  uint32_t end = min(begin + SIMPLE_TILE, live);
  const Logit* values = logits + static_cast<uint64_t>(row) * stride;
  const RowMeta& settings = meta[row];
  FilterRow& record = filter[row];
  for (uint32_t bin = tile * blockDim.x + threadIdx.x; bin < FILTER_BINS;
       bin += tiles * blockDim.x)
    record.bins[bin] = FilterBin{0, 0, 0.0};
  if (tile == 0 && threadIdx.x == 0) record.state = FilterState{};
  // Tokens with a penalty, bias or EOS mask; the rest take their raw value.
  __shared__ uint32_t special[SIMPLE_TILE / 32];
  __shared__ float raw_cache[SIMPLE_TILE];
  if (threadIdx.x < SIMPLE_TILE / 32) special[threadIdx.x] = 0;
  SimplePartial& partial = partials[static_cast<uint64_t>(row) * tiles + tile];
  raw_tile_stats<Logit, SIMPLE_TILE>(values, begin, end, partial, raw_cache, mask);
  if (settings.top_logprobs > 1)
    tile_top_ids(values, tile, live, min(settings.top_logprobs, 20U),
                 partial.top_ids, mask);
  RowScale scale = shared_row_scale(settings);
  const TokenCount* seen = counts + settings.counts_offset;
  const TokenBias* bias = biases + settings.biases_offset;
  const uint32_t* stop = eos + settings.eos_offset;
  uint32_t seen_first = lower_token(seen, settings.counts_len, begin);
  uint32_t seen_last = lower_token(seen, settings.counts_len, end);
  uint32_t bias_first = lower_token(bias, settings.biases_len, begin);
  uint32_t bias_last = lower_token(bias, settings.biases_len, end);
  auto mark = [&](uint32_t token) {
    uint32_t at = token - begin;
    atomicOr(&special[at >> 5], 1U << (at & 31));
  };
  for (uint32_t i = seen_first + threadIdx.x; i < seen_last; i += blockDim.x)
    mark(seen[i].token_id);
  for (uint32_t i = bias_first + threadIdx.x; i < bias_last; i += blockDim.x)
    mark(bias[i].token_id);
  if (settings.mask_eos)
    for (uint32_t i = threadIdx.x; i < settings.eos_len; i += blockDim.x)
      if (stop[i] >= begin && stop[i] < end) mark(stop[i]);
  __syncthreads();
  float plain = -INFINITY;
  for (uint32_t token = begin + threadIdx.x; token < end; token += blockDim.x) {
    uint32_t at = token - begin;
    float raw = raw_cache[at];
    if (!(special[at >> 5] >> (at & 31) & 1U) && isfinite(raw)) plain = fmaxf(plain, raw);
  }
  plain = block_max(plain);
  // Exact values: the plain maximum once, then each unmasked special token.
  double maximum = -INFINITY;
  if (threadIdx.x == 0 && plain > -INFINITY)
    maximum = scale.divide ? __ddiv_rn(static_cast<double>(plain), settings.temperature) :
                             static_cast<double>(plain);
  auto masked = [&](uint32_t token) {
    if (!admitted(mask, token)) return true;
    if (!settings.mask_eos) return false;
    for (uint32_t i = 0; i < settings.eos_len; ++i) if (stop[i] == token) return true;
    return false;
  };
  for (uint32_t i = seen_first + threadIdx.x; i < seen_last; i += blockDim.x) {
    double value = counted_value(values, seen[i], settings, bias);
    if (!isnan(value) && !masked(seen[i].token_id)) maximum = fmax(maximum, value);
  }
  for (uint32_t i = bias_first + threadIdx.x; i < bias_last; i += blockDim.x) {
    if (has_count(seen, settings.counts_len, bias[i].token_id)) continue;
    double value = biased_value(values, bias[i], settings);
    if (!isnan(value) && !masked(bias[i].token_id)) maximum = fmax(maximum, value);
  }
  maximum = block_max(maximum);
  // The Gumbel kernel replaces this with the tile's winning score.
  if (threadIdx.x == 0) partial.best_score = maximum;
}

// Chooses candidate bins from a shared copy of the merged histogram. Lane l
// owns 64 bins.
__device__ __forceinline__ void select_window(FilterState& state,
                                              const RowMeta& settings,
                                              const uint32_t* bin_count,
                                              const double* bin_mass) {
  uint32_t lane = threadIdx.x & 31;
  uint32_t first_bin = lane * (FILTER_BINS / 32);
  uint32_t lane_count = 0;
  double lane_mass = 0.0;
  for (uint32_t i = 0; i < FILTER_BINS / 32; ++i) {
    lane_count += bin_count[first_bin + i];
    lane_mass += bin_mass[first_bin + i];
  }
  uint32_t count_scan = lane_count;
  double mass_scan = lane_mass;
  for (int delta = 1; delta < 32; delta <<= 1) {
    uint32_t other_count = __shfl_up_sync(0xffffffffU, count_scan, delta);
    double other_mass = __shfl_up_sync(0xffffffffU, mass_scan, delta);
    if (lane >= static_cast<uint32_t>(delta)) {
      count_scan += other_count;
      mass_scan += other_mass;
    }
  }
  uint32_t count_before = __shfl_up_sync(0xffffffffU, count_scan, 1);
  double mass_before = __shfl_up_sync(0xffffffffU, mass_scan, 1);
  if (lane == 0) { count_before = 0; mass_before = 0.0; }
  uint32_t survivors = __shfl_sync(0xffffffffU, count_scan, 31);
  double total = __shfl_sync(0xffffffffU, mass_scan, 31);
  if (survivors == 0) {
    if (lane == 0) state.mode = FILTER_KEEP_SURVIVORS;
    return;
  }
  bool top_k = settings.top_k != 0 && settings.top_k < survivors;
  uint32_t limit = top_k ? settings.top_k : survivors;
  // Bin holding rank `limit`, with the count and mass before it.
  uint32_t owner = __ffs(__ballot_sync(0xffffffffU,
      count_before < limit && limit <= count_scan)) - 1;
  uint32_t bin_k = 0, k_head_count = 0;
  double k_head_mass = 0.0;
  if (lane == owner) {
    uint32_t count = count_before;
    double mass = mass_before;
    for (uint32_t i = 0; i < FILTER_BINS / 32; ++i) {
      if (count + bin_count[first_bin + i] >= limit) {
        bin_k = first_bin + i;
        k_head_count = count;
        k_head_mass = mass;
        break;
      }
      count += bin_count[first_bin + i];
      mass += bin_mass[first_bin + i];
    }
  }
  bin_k = __shfl_sync(0xffffffffU, bin_k, owner);
  k_head_count = __shfl_sync(0xffffffffU, k_head_count, owner);
  k_head_mass = __shfl_sync(0xffffffffU, k_head_mass, owner);
  uint32_t bin_lo = bin_k, bin_hi = bin_k;
  if (settings.top_p < 1.0) {
    double low = total, high = total;
    if (top_k) {
      // Bin edges widen by 1e-3 nat for the f32 distances used for binning.
      double taken = static_cast<double>(limit - k_head_count);
      double near_edge = bin_k / static_cast<double>(FILTER_BINS_PER_NAT) - 1.0e-3;
      double far_edge = (bin_k + 1) / static_cast<double>(FILTER_BINS_PER_NAT) + 1.0e-3;
      low = k_head_mass + taken * (bin_k == FILTER_BINS - 1 ? 0.0 : exp(-far_edge));
      high = k_head_mass + taken * exp(-fmax(near_edge, 0.0));
    }
    // The slack only widens the window; exactness is checked later.
    double targets[2] = {settings.top_p * low * (1.0 - 1.0e-7),
                         settings.top_p * high * (1.0 + 1.0e-7)};
    uint32_t found[2];
    for (uint32_t t = 0; t < 2; ++t) {
      uint32_t ballot = __ballot_sync(0xffffffffU, mass_scan >= targets[t]);
      uint32_t result = bin_k;
      if (ballot != 0) {
        uint32_t holder = __ffs(ballot) - 1;
        if (lane == holder) {
          // Rounding may differ from the lane scan; stop at the lane's end.
          result = first_bin + FILTER_BINS / 32 - 1;
          double mass = mass_before;
          for (uint32_t i = 0; i < FILTER_BINS / 32; ++i) {
            mass += bin_mass[first_bin + i];
            if (mass >= targets[t]) { result = first_bin + i; break; }
          }
        }
        result = __shfl_sync(0xffffffffU, result, holder);
      }
      found[t] = min(result, bin_k);
    }
    bin_lo = found[0];
    bin_hi = found[1];
  } else if (!top_k) {
    if (lane == 0) {
      state.mode = FILTER_KEEP_SURVIVORS;
      state.survivors = survivors;
    }
    return;
  }
  // Counts and masses before `bin_lo`, inside [bin_lo, bin_hi], after bin_hi.
  uint32_t head_count = 0, window_count = 0;
  double head_mass = 0.0, through_mass = 0.0;
  for (uint32_t i = 0; i < FILTER_BINS / 32; ++i) {
    uint32_t bin = first_bin + i;
    if (bin < bin_lo) {
      head_count += bin_count[bin];
      head_mass += bin_mass[bin];
    } else if (bin <= bin_hi) {
      window_count += bin_count[bin];
    }
    if (bin <= bin_hi) through_mass += bin_mass[bin];
  }
  for (int delta = 16; delta > 0; delta >>= 1) {
    head_count += __shfl_xor_sync(0xffffffffU, head_count, delta);
    window_count += __shfl_xor_sync(0xffffffffU, window_count, delta);
    head_mass += __shfl_xor_sync(0xffffffffU, head_mass, delta);
    through_mass += __shfl_xor_sync(0xffffffffU, through_mass, delta);
  }
  if (lane == 0) {
    state.mode = FILTER_WINDOW;
    state.survivors = survivors;
    state.bin_lo = bin_lo;
    state.bin_hi = bin_hi;
    state.bin_k = bin_k;
    state.head_count = head_count;
    state.head_mass = head_mass;
    state.k_head_count = k_head_count;
    state.k_head_mass = k_head_mass;
    state.window_count = window_count;
    state.k_count = bin_count[bin_k];
    state.tail_mass = fmax(total - through_mass, 0.0);
  }
}

template <typename Logit>
__device__ __forceinline__ void filtered_histogram(
    const Logit* logits, uint64_t stride, uint32_t live, uint32_t rows,
    const RowMeta* meta, const TokenCount* counts, const TokenBias* biases,
    const uint32_t* eos, SimplePartial* partials, FilterRow* filter,
    const uint32_t* masks, uint32_t mask_words) {
  if (threadIdx.x == 0) chop::ptx::griddep_launch_dependents();
  chop::ptx::griddep_wait();
  uint32_t row = blockIdx.y;
  if (row >= rows) return;
  const uint32_t* mask = row_mask(masks, mask_words, row);
  uint32_t tiles = (live + SIMPLE_TILE - 1) / SIMPLE_TILE;
  const RowMeta& settings = meta[row];
  FilterRow& record = filter[row];
  const SimplePartial* pieces = partials + static_cast<uint64_t>(row) * tiles;
  double maximum = -INFINITY;
  for (uint32_t i = threadIdx.x; i < tiles; i += blockDim.x)
    maximum = fmax(maximum, pieces[i].best_score);
  maximum = block_max(maximum);
  if (blockIdx.x == 0 && threadIdx.x == 0) record.state.maximum = maximum;
  if (!filter_active(settings)) return;
  __shared__ uint32_t bin_count[FILTER_BINS];
  // Tile masses accumulate as integer multiples of 2^-52 (exact and order
  // independent) in two 32-bit words, since 64-bit shared atomics are
  // compare-and-swap loops here. Flushing every four tiles keeps a bin below
  // 4096 unit masses, within 64 bits. The merged row histogram later reuses
  // the storage as f64.
  __shared__ union {
    struct {
      uint32_t low[FILTER_BINS];
      uint32_t high[FILTER_BINS];
    } fixed;
    double mass[FILTER_BINS];
  } bin_mass;
  __shared__ double powers[32];
  __shared__ float2 powers_ff[32];
  __shared__ TileView view;
  __shared__ bool last;
  for (uint32_t bin = threadIdx.x; bin < FILTER_BINS; bin += blockDim.x) {
    bin_count[bin] = 0;
    bin_mass.fixed.low[bin] = 0;
    bin_mass.fixed.high[bin] = 0;
  }
  if (threadIdx.x < 32) {
    double power = exp2(threadIdx.x / 32.0);
    float high = __double2float_rn(power);
    powers[threadIdx.x] = power;
    powers_ff[threadIdx.x] = make_float2(high, __double2float_rn(power - high));
  }
  RowScale scale = shared_row_scale(settings);
  float max_key = processed_key(maximum, scale);
  // Unscaled rows whose maximum is an f32 key take f32 float-float masses.
  bool plain = !scale.divide && static_cast<double>(max_key) == maximum;
  auto flush = [&]() {
    for (uint32_t bin = threadIdx.x; bin < FILTER_BINS; bin += blockDim.x) {
      if (bin_count[bin] == 0) continue;
      atomicAdd(&record.bins[bin].count, bin_count[bin]);
      unsigned long long fixed =
          (static_cast<unsigned long long>(bin_mass.fixed.high[bin]) << 32) |
          bin_mass.fixed.low[bin];
      atomicAdd(&record.bins[bin].mass, static_cast<double>(fixed) * 0x1p-52);
      bin_count[bin] = 0;
      bin_mass.fixed.low[bin] = 0;
      bin_mass.fixed.high[bin] = 0;
    }
  };
  // With many rows, each block covers several tiles to share the bin reset,
  // flush and ticket.
  uint32_t covered = 0;
  for (uint32_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    uint32_t begin = tile * SIMPLE_TILE;
    uint32_t end = min(begin + SIMPLE_TILE, live);
    tile_view(logits + static_cast<uint64_t>(row) * stride, begin, end, settings,
              scale, counts, biases, eos, mask, view);
    for (uint32_t token = begin + threadIdx.x; token < end; token += blockDim.x) {
      uint32_t at = token - begin;
      if (!min_p_survivor(view, at, max_key, maximum, settings, scale)) continue;
      float key = view.key[at];
      uint32_t bin = filter_bin(key, max_key, scale);
      unsigned long long fixed;
      if (bin < FILTER_NEAR_BINS && plain && !view.adjusted[at]) {
        // key <= max_key: the maximum is a plain key or a larger exact value.
        fixed = mass_fixed_ff(key, max_key, powers_ff);
      } else {
        double mass;
        if (bin < FILTER_NEAR_BINS) {
          // Temperature-scaled keys use a product with 1/T: the argument then
          // errs by a few ulps of |value| + |maximum|, covered by the margin.
          double x = view.adjusted[at] ? view.exact[at] - maximum :
                     scale.divide ? fma(static_cast<double>(key), scale.inverse, -maximum) :
                                    static_cast<double>(key) - maximum;
          mass = mass_exp(fmin(x, 0.0), powers);
        } else {
          mass = __expf(-__fmul_rn(__fsub_rn(max_key, key), scale.inverse32));
        }
        fixed = __double2ull_rn(mass * 0x1p52);
      }
      atomicAdd(&bin_count[bin], 1U);
      uint32_t low = static_cast<uint32_t>(fixed);
      uint32_t high = static_cast<uint32_t>(fixed >> 32);
      uint32_t before = atomicAdd(&bin_mass.fixed.low[bin], low);
      high += before + low < before;
      if (high != 0) atomicAdd(&bin_mass.fixed.high[bin], high);
    }
    __syncthreads();
    if (++covered % 4 == 0) {
      flush();
      __syncthreads();
    }
  }
  if (covered % 4 != 0) flush();
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0)
    last = atomicAdd(&record.state.hist_ticket, 1U) == gridDim.x - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  for (uint32_t bin = threadIdx.x; bin < FILTER_BINS; bin += blockDim.x) {
    bin_count[bin] = __ldcg(&record.bins[bin].count);
    bin_mass.mass[bin] = __ldcg(&record.bins[bin].mass);
  }
  __syncthreads();
  if (threadIdx.x < 32)
    select_window(record.state, settings, bin_count, bin_mass.mass);
}

// Exact fallback: sorts every survivor and runs the oracle's sequential top-k
// and top-p evaluation. Returns the index of the last kept token.
template <typename Logit>
__device__ __forceinline__ uint32_t exact_cutoff(
    const Logit* values, uint32_t live, const RowMeta& settings,
    const TokenCount* counts, const TokenBias* biases, const uint32_t* eos,
    const uint32_t* mask, double maximum, double* value, uint32_t* id,
    TileView& view, double* terms,
    uint32_t* chunk_id) {
  __shared__ uint32_t slot, kept;
  __shared__ double running[2];
  if (threadIdx.x == 0) {
    slot = 0;
    running[0] = 0.0;
    running[1] = 0.0;
  }
  RowScale scale = shared_row_scale(settings);
  float max_key = processed_key(maximum, scale);
  for (uint32_t begin = 0; begin < live; begin += blockDim.x) {
    uint32_t end = min(begin + blockDim.x, live);
    tile_view(values, begin, end, settings, scale, counts, biases, eos, mask, view);
    uint32_t token = begin + threadIdx.x;
    if (token < end &&
        min_p_survivor(view, threadIdx.x, max_key, maximum, settings, scale)) {
      uint32_t at = atomicAdd(&slot, 1U);
      value[at] = exact_value(view, threadIdx.x, settings, scale);
      id[at] = token;
    }
    __syncthreads();
  }
  uint32_t count = slot;
  if (count == 0) return 0;
  sort_global_pairs(value, id, count, terms, chunk_id);
  uint32_t limit = settings.top_k != 0 && settings.top_k < count ?
                   settings.top_k : count;
  if (settings.top_p >= 1.0) return limit - 1;
  double top = __ldcg(value);
  // Pass 0 sums the kept terms; pass 1 stops at the first prefix reaching
  // top_p times that sum. Thread 0 adds in rank order like the oracle.
  if (threadIdx.x == 0) kept = limit;
  for (uint32_t pass = 0; pass < 2; ++pass) {
    for (uint32_t begin = 0; begin < limit; begin += blockDim.x) {
      uint32_t index = begin + threadIdx.x;
      if (index < limit) terms[threadIdx.x] = exp(__ldcg(value + index) - top);
      __syncthreads();
      if (threadIdx.x == 0) {
        uint32_t n = min(blockDim.x, limit - begin);
        double sum = running[pass];
        double threshold = settings.top_p * running[0];
        for (uint32_t i = 0; i < n; ++i) {
          sum += terms[i];
          if (pass == 1 && sum >= threshold) { kept = begin + i + 1; break; }
        }
        running[pass] = sum;
      }
      __syncthreads();
      if (pass == 1 && kept != limit) break;
    }
  }
  return kept - 1;
}

// Last collect block: resolves the cutoff from the sorted window candidates.
template <typename Logit>
__device__ __forceinline__ void resolve_cutoff(
    const Logit* values, uint32_t live, const RowMeta& settings,
    const TokenCount* counts, const TokenBias* biases, const uint32_t* eos,
    const uint32_t* mask, FilterState& state, double* value, uint32_t* id,
    TileView& view) {
  __shared__ double shared_value[FILTER_SHARED_SORT];
  __shared__ uint32_t shared_id[FILTER_SHARED_SORT];
  __shared__ uint32_t found;
  __shared__ bool exact;
  uint32_t count = __ldcg(&state.candidates);
  double maximum = state.maximum;
  uint32_t survivors = state.survivors;
  bool top_k = settings.top_k != 0 && settings.top_k < survivors;
  bool split = state.bin_k > state.bin_hi;
  uint32_t window = state.window_count;
  bool consistent = count == window + (split ? state.k_count : 0);
  bool in_shared = consistent && count <= FILTER_SHARED_SORT;
  if (in_shared) {
    for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
      shared_value[i] = __ldcg(value + i);
      shared_id[i] = __ldcg(id + i);
    }
    __syncthreads();
    sort_pairs(shared_value, shared_id, count);
  } else if (consistent) {
    sort_global_pairs(value, id, count, shared_value, shared_id);
  }
  auto value_at = [&](uint32_t i) {
    return in_shared ? shared_value[i] : __ldcg(value + i);
  };
  // Candidate index of the top-k cutoff, when top-k truncates.
  uint32_t k_index = 0xffffffffU;
  if (top_k)
    k_index = split ? window + (settings.top_k - 1 - state.k_head_count) :
                      settings.top_k - 1 - state.head_count;
  uint32_t cutoff = k_index;
  if (threadIdx.x == 0) exact = !consistent;
  __syncthreads();
  if (consistent && settings.top_p < 1.0) {
    // Top-p among the top-k survivors, or among all survivors.
    double partial = 0.0;
    uint32_t from = split && top_k ? window : 0;
    uint32_t to = top_k ? k_index + 1 : window;
    for (uint32_t i = from + threadIdx.x; i < to; i += blockDim.x)
      partial += exp(value_at(i) - maximum);
    partial = block_sum(partial);
    double total = top_k ? (split ? state.k_head_mass : state.head_mass) + partial :
                           state.head_mass + partial + state.tail_mass;
    double threshold = settings.top_p * total;
    // The first term is at least three times a first-order bound,
    // (10n + 300) * 2^-53 * total, on the combined error of the oracle's
    // sequential sums, the histogram's fixed-point and unordered sums, the
    // tail subtraction, and exp results within one ulp of the CPU's. The
    // others cover histogram mass arguments, the f64 polynomial or the f32
    // float-float mass (1e-13 relative) and f32 masses past 26 nats, for all
    // terms on each side. `total` is at least the maximum's unit mass.
    // Only the thread at the cutoff evaluates it.
    auto margin = [&]() {
      bool divide = settings.temperature != 1.0;
      double argument = divide ? 0x1p-51 * (2.0 * fabs(maximum) + 64.0) :
                                 0x1p-52 * (fabs(maximum) + 64.0);
      // f32 masses past 26 nats: at most e^-25.99 times their relative error.
      double far = 5.2e-12 * (0x1p-21 * (2.0 * fabs(maximum) + 64.0) + 0x1p-20);
      return total * ((static_cast<double>(survivors) + FILTER_BINS + 64.0) * 0x1p-48 +
                      2.0 * (argument + 1.0e-13)) +
             2.0 * survivors * far;
    };
    uint32_t limit = top_k && !split ? k_index + 1 : window;
    if (threadIdx.x == 0) found = 0xffffffffU;
    double carry = state.head_mass;
    for (uint32_t begin = 0; begin < limit; begin += blockDim.x) {
      uint32_t index = begin + threadIdx.x;
      double term = index < limit ? exp(value_at(index) - maximum) : 0.0;
      double inclusive, exclusive, chunk;
      block_scan(term, inclusive, exclusive, chunk);
      double reached = carry + inclusive;
      if (index < limit && reached >= threshold) atomicMin(&found, index);
      __syncthreads();
      uint32_t first = found;
      if (first != 0xffffffffU) {
        if (index == first) {
          double before = carry + exclusive;
          bool first_overall = index == 0 && state.head_count == 0;
          double bound = margin();
          exact = !(reached - threshold > bound &&
                    (first_overall || threshold - before > bound));
        }
        break;
      }
      carry += chunk;
    }
    __syncthreads();
    if (found == 0xffffffffU) exact = true;
    __syncthreads();
    cutoff = found;
  }
  if (exact) {
    // Uniform across the block: every thread reads the same shared flag.
    cutoff = exact_cutoff(values, live, settings, counts, biases, eos, mask, maximum,
                          value, id, view, shared_value, shared_id);
    in_shared = false;
  }
  if (threadIdx.x == 0) {
    state.threshold = in_shared ? shared_value[cutoff] : __ldcg(value + cutoff);
    state.threshold_id = in_shared ? shared_id[cutoff] : __ldcg(id + cutoff);
    state.exact_fallback = exact;
    state.mode = FILTER_THRESHOLD;
  }
}

template <typename Logit>
__device__ __forceinline__ void filtered_collect(
    const Logit* logits, uint64_t stride, uint32_t live, uint32_t rows,
    const RowMeta* meta, const TokenCount* counts, const TokenBias* biases,
    const uint32_t* eos, const SimplePartial* partials, FilterRow* filter,
    double* candidate_value, uint32_t* candidate_id, const uint32_t* masks,
    uint32_t mask_words) {
  if (threadIdx.x == 0) chop::ptx::griddep_launch_dependents();
  chop::ptx::griddep_wait();
  uint32_t row = blockIdx.y;
  if (row >= rows) return;
  const uint32_t* mask = row_mask(masks, mask_words, row);
  FilterState& state = filter[row].state;
  if (state.mode != FILTER_WINDOW) return;
  uint32_t tile = blockIdx.x;
  uint32_t tiles = (live + SIMPLE_TILE - 1) / SIMPLE_TILE;
  const RowMeta& settings = meta[row];
  RowScale scale = shared_row_scale(settings);
  double maximum = state.maximum;
  float max_key = processed_key(maximum, scale);
  // Every token in this tile is at least as far as its largest one.
  double tile_max = partials[static_cast<uint64_t>(row) * tiles + tile].best_score;
  if (!(tile_max > -INFINITY) ||
      filter_bin(processed_key(tile_max, scale), max_key, scale) > state.bin_k)
    return;
  const Logit* values = logits + static_cast<uint64_t>(row) * stride;
  double* value = candidate_value + static_cast<uint64_t>(row) * live;
  uint32_t* id = candidate_id + static_cast<uint64_t>(row) * live;
  __shared__ TileView view;
  uint32_t begin = tile * SIMPLE_TILE;
  uint32_t end = min(begin + SIMPLE_TILE, live);
  tile_view(values, begin, end, settings, scale, counts, biases, eos, mask, view);
  uint32_t lane = threadIdx.x & 31;
  for (uint32_t at = threadIdx.x; at < SIMPLE_TILE; at += blockDim.x) {
    bool take = false;
    if (begin + at < end &&
        min_p_survivor(view, at, max_key, maximum, settings, scale)) {
      uint32_t bin = filter_bin(view.key[at], max_key, scale);
      take = (bin >= state.bin_lo && bin <= state.bin_hi) || bin == state.bin_k;
    }
    uint32_t votes = __ballot_sync(0xffffffffU, take);
    if (votes == 0) continue;
    uint32_t leader = __ffs(votes) - 1;
    uint32_t base = 0;
    if (lane == leader) base = atomicAdd(&state.candidates, __popc(votes));
    base = __shfl_sync(0xffffffffU, base, leader);
    if (take) {
      uint32_t slot = base + __popc(votes & ((1U << lane) - 1));
      value[slot] = exact_value(view, at, settings, scale);
      id[slot] = begin + at;
    }
  }
}

template <typename Logit>
__device__ __forceinline__ void filtered_resolve(
    const Logit* logits, uint64_t stride, uint32_t live, uint32_t rows,
    const RowMeta* meta, const TokenCount* counts, const TokenBias* biases,
    const uint32_t* eos, FilterRow* filter, double* candidate_value,
    uint32_t* candidate_id, const uint32_t* masks, uint32_t mask_words) {
  if (threadIdx.x == 0) chop::ptx::griddep_launch_dependents();
  chop::ptx::griddep_wait();
  uint32_t row = blockIdx.x;
  if (row >= rows) return;
  FilterState& state = filter[row].state;
  if (state.mode != FILTER_WINDOW) return;
  __shared__ TileView view;
  resolve_cutoff(logits + static_cast<uint64_t>(row) * stride, live, meta[row],
                 counts, biases, eos, row_mask(masks, mask_words, row), state,
                 candidate_value + static_cast<uint64_t>(row) * live,
                 candidate_id + static_cast<uint64_t>(row) * live, view);
}

template <typename Logit>
__device__ __forceinline__ void filtered_gumbel(
    const Logit* logits, uint64_t stride, uint32_t live, uint32_t rows,
    const RowMeta* meta, const TokenCount* counts, const TokenBias* biases,
    const uint32_t* eos, SimplePartial* partials, const FilterRow* filter,
    const uint32_t* masks, uint32_t mask_words) {
  if (threadIdx.x == 0) chop::ptx::griddep_launch_dependents();
  chop::ptx::griddep_wait();
  uint32_t row = blockIdx.y;
  if (row >= rows) return;
  const uint32_t* mask = row_mask(masks, mask_words, row);
  uint32_t tile = blockIdx.x;
  uint32_t tiles = (live + SIMPLE_TILE - 1) / SIMPLE_TILE;
  const RowMeta& settings = meta[row];
  const FilterState& state = filter[row].state;
  __shared__ TileView view;
  __shared__ float approx_scores[SIMPLE_TILE];
  __shared__ double warp_best[32];
  __shared__ uint32_t warp_id[32];
  RowScale scale = shared_row_scale(settings);
  bool threshold = state.mode == FILTER_THRESHOLD;
  double cut = state.threshold;
  SimplePartial& partial = partials[static_cast<uint64_t>(row) * tiles + tile];
  // A tile whose largest processed value is below the cutoff keeps nothing.
  if (settings.temperature != 0.0 && threshold && partial.best_score < cut) {
    __syncthreads();
    if (threadIdx.x == 0) {
      partial.best_score = -INFINITY;
      partial.token_id = 0xffffffffU;
    }
    return;
  }
  uint32_t begin = tile * SIMPLE_TILE;
  uint32_t end = min(begin + SIMPLE_TILE, live);
  tile_view(logits + static_cast<uint64_t>(row) * stride, begin, end, settings,
            scale, counts, biases, eos, mask, view);
  double maximum = state.maximum;
  float max_key = processed_key(maximum, scale);
  uint32_t cut_id = state.threshold_id;
  float cut_key = processed_key(cut, scale);
  double best = -INFINITY;
  uint32_t best_id = 0xffffffffU;
  if (settings.temperature == 0.0) {
    // Greedy: the largest key, then exact values among equal keys.
    float best_key = -INFINITY;
    for (uint32_t token = begin + threadIdx.x; token < end; token += blockDim.x) {
      float key = view.key[token - begin];
      if (!isnan(key)) best_key = fmaxf(best_key, key);
    }
    best_key = block_max(best_key);
    for (uint32_t token = begin + threadIdx.x; token < end; token += blockDim.x) {
      if (view.key[token - begin] != best_key) continue;
      double v = exact_value(view, token - begin, settings, scale);
      if (better(v, token, best, best_id)) { best = v; best_id = token; }
    }
  } else {
    float approx_best = -INFINITY;
    for (uint32_t token = begin + threadIdx.x; token < end; token += blockDim.x) {
      uint32_t at = token - begin;
      float score = -INFINITY;
      bool keep = min_p_survivor(view, at, max_key, maximum, settings, scale);
      if (keep && threshold) {
        float key = view.key[at];
        if (key == cut_key) {
          double v = exact_value(view, at, settings, scale);
          keep = v > cut || (v == cut && token <= cut_id);
        } else {
          keep = key > cut_key;
        }
      }
      if (keep) {
        score = static_cast<float>(exact_value(view, at, settings, scale) - maximum) +
                approximate_gumbel(settings.seed, settings.generated, token);
        // -INFINITY marks a rejected token, so an estimate that left the F32
        // range (a tiny temperature) becomes NaN: never pruned, and always
        // evaluated exactly.
        if (!isfinite(score)) score = NAN;
        approx_best = fmaxf(approx_best, score);
      }
      approx_scores[at] = score;
    }
    float tile_best = block_max(approx_best);
    float margin = 0.1f + 1.0e-5f * fabsf(tile_best);
    for (uint32_t token = begin + threadIdx.x; token < end; token += blockDim.x) {
      float approximate = approx_scores[token - begin];
      if (approximate == -INFINITY ||
          (isfinite(tile_best) && approximate < tile_best - margin)) continue;
      double score = (exact_value(view, token - begin, settings, scale) - maximum) +
                     gumbel(settings.seed, settings.generated, token);
      if (better(score, token, best, best_id)) { best = score; best_id = token; }
    }
  }
  for (int delta = 16; delta > 0; delta >>= 1) {
    double other = __shfl_down_sync(0xffffffffU, best, delta);
    uint32_t other_id = __shfl_down_sync(0xffffffffU, best_id, delta);
    if (better(other, other_id, best, best_id)) { best = other; best_id = other_id; }
  }
  if ((threadIdx.x & 31) == 0) {
    warp_best[threadIdx.x >> 5] = best;
    warp_id[threadIdx.x >> 5] = best_id;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (uint32_t i = 1; i < (blockDim.x >> 5); ++i)
      if (better(warp_best[i], warp_id[i], best, best_id)) {
        best = warp_best[i];
        best_id = warp_id[i];
      }
    partial.best_score = best;
    partial.token_id = best_id;
  }
}

}  // namespace chop::sampling
