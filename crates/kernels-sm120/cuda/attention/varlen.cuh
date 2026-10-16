#pragma once

// Variable-length attention over contiguous Q, K and V for the vision and
// audio encoders (decision 0083, contract attention.contiguous-varlen-bf16.v1).
//
// Q is [T][q_heads][D], K and V are [T][kv_heads][D] BF16, and cu_seqlens
// splits the T packed tokens into sequences. A CTA owns one slice of 128
// grouped query rows (128 / group consecutive tokens of one KV head) on the
// packed token axis, so the grid depends only on the prepared capacity. A
// slice may cross sequence boundaries: every row's visible keys form one
// interval [lo, hi] of packed token indices, built from its sequence bounds,
// the causal flag and the windows. Both ends are nondecreasing in the token
// index, so the slice's keys are [lo(first row), hi(last row)].
//
// The inner loop follows the FlashAttention-2 prefill in prefill.cuh: each
// warp holds its m16 row tile of Q in registers, keys stream through two
// cp.async shared-memory stages, S = QK^T and O += PV use BF16
// mma.sync.m16n8k16, the online softmax runs in base-two units, and the
// probabilities feed the PV MMA as A fragments without leaving registers.
// Rows are stored at a padded stride instead of the TMA swizzle, which keeps
// ldmatrix conflict free for D = 64 and for D = 72, whose QK reduction is
// padded to 80 with zeros.

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>
#include <type_traits>
#include "../common/ptx.cuh"
#include "../common/ptx_matrix.cuh"

namespace chop::attention::varlen {

// Window distances are inclusive maximum index distances; -1 means none.
template<int D, int QHeads, int KvHeads, bool Causal, int WindowLeft, int WindowRight,
         bool Sink>
struct Traits {
  static_assert(D == 64 || D == 72, "contract head dimensions");
  static_assert(QHeads > 0 && KvHeads > 0 && QHeads % KvHeads == 0);
  static constexpr int group = QHeads / KvHeads;
  static_assert(16 % group == 0, "an m16 row tile holds whole GQA groups");
  static constexpr int d = D, q_heads = QHeads, kv_heads = KvHeads;
  static constexpr bool causal = Causal, sink = Sink;
  static constexpr int window_left = WindowLeft, window_right = WindowRight;
  // The QK reduction rounds D up to the MMA K of 16; V keeps D columns.
  static constexpr int dk = (D + 15) / 16 * 16;
  // Shared row stride in bytes: 8 extra BF16 columns put the eight rows an
  // ldmatrix reads on distinct bank groups.
  static constexpr int stride = (dk + 8) * 2;
  static constexpr int warps = 8;
  static constexpr int threads = warps * 32;
  static constexpr int tokens = 16 * warps / group;
  static constexpr int keys = 64;
  static constexpr int stages = 2;
  static constexpr int tile_bytes = keys * stride;
  static constexpr int smem = stages * 2 * tile_bytes;
  static_assert(smem <= 48 * 1024, "static shared memory");
};

__device__ __forceinline__ void cp_async_zfill(void* dst, const void* src, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
               :: "r"(ptx::shared_address(dst)), "l"(src), "r"(valid ? 16 : 0)
               : "memory");
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}
template<int Pending>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" :: "n"(Pending) : "memory");
}

__device__ __forceinline__ float bf16_low(uint32_t pair) {
  return __uint_as_float(pair << 16);
}
__device__ __forceinline__ float bf16_high(uint32_t pair) {
  return __uint_as_float(pair & 0xffff0000u);
}
__device__ __forceinline__ uint32_t bf16_pair(float low, float high) {
  const __nv_bfloat162 pair = __floats2bfloat162_rn(low, high);
  return *reinterpret_cast<const uint32_t*>(&pair);
}

// BF16 fragments per probability (the rounded value and its rounded
// residuals) for a row that sees `keys` keys. The dropped residual is at most
// 2^(-8 terms) of each |p v| product, within half of the decision-0081 term
// 2^-22 K S at these thresholds. Same rule as the prefill engine.
__device__ __forceinline__ int probability_terms(int keys) {
  return keys >= 32768 ? 1 : keys >= 128 ? 2 : 3;
}

// The sequence holding packed token t: cu_seqlens[s] <= t < cu_seqlens[s + 1].
__device__ __forceinline__ int sequence_of(const int32_t* cu_seqlens, int sequences, int t) {
  int low = 0, high = sequences - 1;
  while (low < high) {
    const int middle = (low + high + 1) / 2;
    if (cu_seqlens[middle] <= t) low = middle; else high = middle - 1;
  }
  return low;
}

// The visible key interval of packed token t, and its sequence start.
template<class T>
__device__ __forceinline__ void visible(const int32_t* cu_seqlens, int sequences, int t,
                                        int& lo, int& hi, int& start) {
  const int s = sequence_of(cu_seqlens, sequences, t);
  start = cu_seqlens[s];
  const int end = cu_seqlens[s + 1];
  lo = T::window_left >= 0 ? max(start, t - T::window_left) : start;
  hi = T::causal ? t : end - 1;
  if constexpr (T::window_right >= 0) hi = min(hi, t + T::window_right);
}

template<class T>
__device__ void attention(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    const int32_t* cu_seqlens, const float* sinks, __nv_bfloat16* output,
    const uint32_t* live_tokens, uint32_t live_sequences, uint32_t scale_bits) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const int live = int(*live_tokens);
  const int sequences = int(live_sequences);
  const int t0 = int(blockIdx.x) * T::tokens;
  if (t0 >= live || sequences <= 0) return;
  const int kv_head = blockIdx.y;
  const int tokens = min(T::tokens, live - t0);
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int g = lane / 4;
  const int c = lane % 4;

  // Keys of the whole slice.
  int key_first, key_last, unused_hi, unused_start, unused_lo;
  visible<T>(cu_seqlens, sequences, t0, key_first, unused_hi, unused_start);
  visible<T>(cu_seqlens, sequences, t0 + tokens - 1, unused_lo, key_last, unused_start);
  const int key_tiles = (key_last - key_first) / T::keys + 1;

  __shared__ __align__(128) uint8_t storage[T::smem];
  auto k_tile = [&](int stage) { return storage + stage * 2 * T::tile_bytes; };
  auto v_tile = [&](int stage) { return k_tile(stage) + T::tile_bytes; };
  // Columns D..dk of K are the zero padding of the QK reduction; cp.async
  // never writes them.
  if constexpr (T::dk != T::d) {
    for (int row = threadIdx.x; row < T::stages * T::keys; row += T::threads) {
      uint8_t* base = k_tile(row / T::keys) + (row % T::keys) * T::stride;
      *reinterpret_cast<uint4*>(base + T::d * 2) = make_uint4(0, 0, 0, 0);
    }
  }

  // Each thread copies 16-byte sectors of key rows; keys past the slice are
  // zero filled and masked.
  auto issue = [&](int tile, int stage) {
    constexpr int sectors = T::d / 8;
    const int key0 = key_first + tile * T::keys;
    for (int i = threadIdx.x; i < T::keys * sectors * 2; i += T::threads) {
      const bool is_v = i >= T::keys * sectors;
      const int j = is_v ? i - T::keys * sectors : i;
      const int row = j / sectors;
      const int sector = j % sectors;
      const int key = key0 + row;
      const bool valid = key <= key_last;
      const __nv_bfloat16* base = is_v ? v : k;
      const __nv_bfloat16* src =
          base + (uint64_t(valid ? key : key_first) * T::kv_heads + kv_head) * T::d + sector * 8;
      uint8_t* dst = (is_v ? v_tile(stage) : k_tile(stage)) + row * T::stride + sector * 16;
      cp_async_zfill(dst, src, valid);
    }
    cp_async_commit();
  };
  issue(0, 0);
  if (key_tiles > 1) issue(1, 1);

  // This thread's rows g and g + 8 of the warp's m16 tile.
  int token[2], head[2], lo[2], hi[2], start[2];
  bool row_live[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int row = warp * 16 + g + r * 8;
    token[r] = t0 + row / T::group;
    head[r] = kv_head * T::group + row % T::group;
    row_live[r] = row / T::group < tokens;
    if (row_live[r]) {
      visible<T>(cu_seqlens, sequences, token[r], lo[r], hi[r], start[r]);
    } else {
      lo[r] = INT32_MAX;
      hi[r] = -1;
      start[r] = -1;
    }
  }
  // Warp-wide bounds: the union of the rows' keys, the keys every live row
  // sees, and the smallest visible count, which sets the probability terms.
  int warp_lo = min(lo[0], lo[1]);
  int warp_hi = max(hi[0], hi[1]);
  int all_lo = max(row_live[0] ? lo[0] : INT32_MIN, row_live[1] ? lo[1] : INT32_MIN);
  int all_hi = min(row_live[0] ? hi[0] : INT32_MAX, row_live[1] ? hi[1] : INT32_MAX);
  int fewest = min(row_live[0] ? hi[0] - lo[0] + 1 : INT32_MAX,
                   row_live[1] ? hi[1] - lo[1] + 1 : INT32_MAX);
  int sink_lo = INT32_MAX, sink_hi = -1;
  if constexpr (T::sink) {
#pragma unroll
    for (int r = 0; r < 2; ++r)
      if (row_live[r] && lo[r] == start[r]) {
        sink_lo = min(sink_lo, start[r]);
        sink_hi = max(sink_hi, start[r]);
      }
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    warp_lo = min(warp_lo, __shfl_xor_sync(0xffffffff, warp_lo, offset));
    warp_hi = max(warp_hi, __shfl_xor_sync(0xffffffff, warp_hi, offset));
    all_lo = max(all_lo, __shfl_xor_sync(0xffffffff, all_lo, offset));
    all_hi = min(all_hi, __shfl_xor_sync(0xffffffff, all_hi, offset));
    fewest = min(fewest, __shfl_xor_sync(0xffffffff, fewest, offset));
    sink_lo = min(sink_lo, __shfl_xor_sync(0xffffffff, sink_lo, offset));
    sink_hi = max(sink_hi, __shfl_xor_sync(0xffffffff, sink_hi, offset));
  }
  const bool warp_live = warp_hi >= 0;
  const int terms = probability_terms(fewest);

  constexpr float log2e = 1.4426950408889634f;
  const float score_scale = __uint_as_float(scale_bits) * log2e;
  float sink_logit[2] = {0.0f, 0.0f};
  if constexpr (T::sink) {
#pragma unroll
    for (int r = 0; r < 2; ++r)
      if (row_live[r]) sink_logit[r] = sinks[head[r]] * log2e;
  }

  // Q fragments, zero past D and for rows past the slice.
  const uint32_t* q_words = reinterpret_cast<const uint32_t*>(q);
  ptx::A4 query[T::dk / 16];
#pragma unroll
  for (int ks = 0; ks < T::dk / 16; ++ks)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int r = i % 2;
      const int column = ks * 16 + 2 * c + (i / 2) * 8;
      query[ks].v[i] = row_live[r] && column < T::d ?
          q_words[(uint64_t(token[r]) * T::q_heads + head[r]) * (T::d / 2) + column / 2] : 0u;
    }
  float maximum[2] = {-INFINITY, -INFINITY};
  float denominator[2] = {0.0f, 0.0f};
  ptx::D4 out[T::d / 8];
#pragma unroll
  for (int n = 0; n < T::d / 8; ++n)
#pragma unroll
    for (int i = 0; i < 4; ++i) out[n].v[i] = 0.0f;

  auto accumulate = [&](auto terms_tag, const uint8_t* k_smem, const uint8_t* v_smem,
                        int key0) {
    constexpr int count = decltype(terms_tag)::value;
    ptx::D4 score[T::keys / 8];
#pragma unroll
    for (int n = 0; n < T::keys / 8; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) score[n].v[i] = 0.0f;
#pragma unroll
    for (int ks = 0; ks < T::dk / 16; ++ks)
#pragma unroll
      for (int np = 0; np < T::keys / 16; ++np) {
        const int row = np * 16 + (lane / 16) * 8 + lane % 8;
        const int dim = ks * 16 + ((lane / 8) % 2) * 8;
        uint32_t b[4];
        ptx::ldmatrix_b16<4>(b, k_smem + row * T::stride + dim * 2);
        ptx::mma_bf16(score[2 * np], query[ks], ptx::B2{{b[0], b[1]}});
        ptx::mma_bf16(score[2 * np + 1], query[ks], ptx::B2{{b[2], b[3]}});
      }
#pragma unroll
    for (int n = 0; n < T::keys / 8; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) score[n].v[i] *= score_scale;
    // Masks and the sink apply only on tiles that cut a row's interval or
    // hold a sequence's first key.
    const bool cut = key0 < all_lo || key0 + T::keys - 1 > all_hi;
    const bool sunk = T::sink && key0 <= sink_hi && key0 + T::keys - 1 >= sink_lo;
    if (cut || sunk) {
#pragma unroll
      for (int n = 0; n < T::keys / 8; ++n)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int r = i / 2;
          const int key = key0 + n * 8 + 2 * c + i % 2;
          float x = score[n].v[i];
          if constexpr (T::sink) x += key == start[r] ? sink_logit[r] : 0.0f;
          score[n].v[i] = key < lo[r] || key > hi[r] ? -INFINITY : x;
        }
    }
    float correction[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      float tile_max = -INFINITY;
#pragma unroll
      for (int n = 0; n < T::keys / 8; ++n)
        tile_max = fmaxf(tile_max, fmaxf(score[n].v[2 * r], score[n].v[2 * r + 1]));
      tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 1));
      tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 2));
      const float next = fmaxf(maximum[r], tile_max);
      const float base = next == -INFINITY ? 0.0f : next;
      correction[r] = exp2f(maximum[r] - base);
      maximum[r] = next;
      float sum = 0.0f;
#pragma unroll
      for (int n = 0; n < T::keys / 8; ++n)
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const float p = exp2f(score[n].v[2 * r + i] - base);
          score[n].v[2 * r + i] = p;
          sum += p;
        }
      denominator[r] = denominator[r] * correction[r] + sum;
    }
    if (__any_sync(0xffffffff, correction[0] != 1.0f || correction[1] != 1.0f)) {
#pragma unroll
      for (int n = 0; n < T::d / 8; ++n) {
        out[n].v[0] *= correction[0];
        out[n].v[1] *= correction[0];
        out[n].v[2] *= correction[1];
        out[n].v[3] *= correction[1];
      }
    }
#pragma unroll
    for (int kk = 0; kk < T::keys / 16; ++kk) {
      ptx::A4 probability[count];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const ptx::D4& s = score[2 * kk + i / 2];
        float p0 = s.v[(i % 2) * 2];
        float p1 = s.v[(i % 2) * 2 + 1];
#pragma unroll
        for (int term = 0; term < count; ++term) {
          const uint32_t pair = bf16_pair(p0, p1);
          probability[term].v[i] = pair;
          p0 -= bf16_low(pair);
          p1 -= bf16_high(pair);
        }
      }
      const int row = kk * 16 + ((lane / 8) % 2) * 8 + lane % 8;
#pragma unroll
      for (int vn = 0; vn < T::d / 16; ++vn) {
        const int dim = vn * 16 + (lane / 16) * 8;
        uint32_t b[4];
        ptx::ldmatrix_b16<4, true>(b, v_smem + row * T::stride + dim * 2);
#pragma unroll
        for (int term = 0; term < count; ++term) {
          ptx::mma_bf16(out[2 * vn], probability[term], ptx::B2{{b[0], b[1]}});
          ptx::mma_bf16(out[2 * vn + 1], probability[term], ptx::B2{{b[2], b[3]}});
        }
      }
      if constexpr (T::d % 16 != 0) {
        // The last eight V columns of D = 72.
        uint32_t b[2];
        ptx::ldmatrix_b16<2, true>(b, v_smem + row * T::stride + (T::d - 8) * 2);
#pragma unroll
        for (int term = 0; term < count; ++term)
          ptx::mma_bf16(out[T::d / 8 - 1], probability[term], ptx::B2{{b[0], b[1]}});
      }
    }
  };

  for (int tile = 0; tile < key_tiles; ++tile) {
    const int stage = tile % T::stages;
    if (tile + 1 < key_tiles) cp_async_wait<1>(); else cp_async_wait<0>();
    __syncthreads();
    const int key0 = key_first + tile * T::keys;
    if (warp_live && key0 <= warp_hi && key0 + T::keys > warp_lo) {
      if (terms == 1) {
        accumulate(std::integral_constant<int, 1>{}, k_tile(stage), v_tile(stage), key0);
      } else if (terms == 2) {
        accumulate(std::integral_constant<int, 2>{}, k_tile(stage), v_tile(stage), key0);
      } else {
        accumulate(std::integral_constant<int, 3>{}, k_tile(stage), v_tile(stage), key0);
      }
    }
    __syncthreads();
    if (tile + T::stages < key_tiles) issue(tile + T::stages, stage);
  }

#pragma unroll
  for (int r = 0; r < 2; ++r) {
    float total = denominator[r];
    total += __shfl_xor_sync(0xffffffff, total, 1);
    total += __shfl_xor_sync(0xffffffff, total, 2);
    if (!row_live[r]) continue;
    const float normalizer = 1.0f / total;
    uint32_t* destination = reinterpret_cast<uint32_t*>(output) +
        (uint64_t(token[r]) * T::q_heads + head[r]) * (T::d / 2);
#pragma unroll
    for (int n = 0; n < T::d / 8; ++n)
      destination[n * 4 + c] =
          bf16_pair(out[n].v[2 * r] * normalizer, out[n].v[2 * r + 1] * normalizer);
  }
}

} // namespace chop::attention::varlen
