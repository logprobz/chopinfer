#pragma once

// QSA block-sparse attention (decision 0084) over the paged KV cache. A row
// at position p attends to the tokens of its selected blocks (ascending, from
// the indexer's top-k) and then its incomplete tail, `(p + 1) / block * block`
// to p. All query heads share the key set. See SPARSE.md.

#include <math.h>
#include <type_traits>
#include "sparse_indexer.cuh"

namespace chop::attention::sparse {

template<int QHeads, int KvHeads, int Dim, int Page, int Block, int TopBlocks, bool Fp8,
         int Keys>
struct Attention {
  static_assert(QHeads % KvHeads == 0);
  static constexpr int q_heads = QHeads, kv_heads = KvHeads, dim = Dim, page = Page;
  static constexpr int block = Block, top = TopBlocks;
  static constexpr bool fp8 = Fp8;
  static constexpr int group = QHeads / KvHeads;
  // Grouped heads fill one or two n8 tiles; the padded heads of a partial
  // tile are zero filled and never written.
  static_assert(group <= 16, "grouped heads fill one or two n8 tiles");
  static_assert((Dim == 128 || Dim == 256) && Page % Block == 0);
  // The grouped heads padded to whole n8 tiles.
  static constexpr int n_tiles = (group + 7) / 8;
  static constexpr int padded = n_tiles * 8;
  static constexpr int warps = 4, threads = warps * 32;
  // A unit is `Keys` keys in 16-key tiles. Warp w forms S^T for key tile
  // w % tiles and accumulates O^T for V column part w / tiles.
  static_assert(Keys == 16 || Keys == 32);
  static constexpr int keys = Keys;
  static constexpr int tiles = Keys / 16, parts = warps / tiles;
  static constexpr int columns = Dim / parts;
  static constexpr int stages = 2;
  static constexpr int k_bytes = keys * Dim * 2;
  static constexpr int v_bytes = keys * Dim * 2;
  static constexpr int raw_bytes = keys * Dim * 2;
  static constexpr int q_bytes = padded * Dim * 2;
  static constexpr int list_bytes = TopBlocks * 4;
  static constexpr int smem = (Fp8 ? stages * raw_bytes + k_bytes + v_bytes
                                   : stages * (k_bytes + v_bytes)) +
                              q_bytes + list_bytes;
  // F32 [max, denominator, V numerator] per row, KV head, split and head.
  static constexpr int partial = Dim + 2;
};

template<class T>
__device__ __forceinline__ uint64_t partial_index(int row, int kv_head, int splits, int split,
                                                  int h) {
  return (((uint64_t(row) * T::kv_heads + kv_head) * splits + split) * T::group + h) *
         T::partial;
}

// Grid [rows, KV heads, splits]. A row's key list is cut into units of 32
// keys, and split s owns units [units * s / splits, units * (s + 1) /
// splits). With one split the CTA writes normalized BF16 output; otherwise it
// writes its F32 partial, empty when it owns no unit, for `merge`.
template<class T>
__device__ void attention(
    const uint16_t* q, const void* k_cache, const void* v_cache,
    const int32_t* block_tables, const int32_t* row_batch, const int32_t* positions,
    const uint32_t* live_rows, const int32_t* selected, float* workspace, uint16_t* out,
    uint32_t max_pages, uint64_t k_page_stride_bytes, uint64_t v_page_stride_bytes,
    uint32_t splits, uint32_t scale_bits, uint32_t k_scale_bits, uint32_t v_scale_bits) {
  ptx::griddep_launch_dependents();
  extern __shared__ __align__(1024) uint8_t attention_smem[];
  ptx::griddep_wait();
  const int row = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int split = blockIdx.z;
  if (row >= int(*live_rows)) return;
  const int batch = row_batch[row];
  const int position = positions[row];
  const int complete = (position + 1) / T::block;
  const int count = min(complete, T::top);
  const int listed = count * T::block;
  const int keys = listed + position + 1 - complete * T::block;
  const int units = (keys + T::keys - 1) / T::keys;
  const int unit_begin = int(int64_t(units) * split / splits);
  const int unit_end = int(int64_t(units) * (split + 1) / splits);

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int g = lane / 4;
  const int c = lane % 4;
  const int tile = warp % T::tiles;
  const int part = warp / T::tiles;
  constexpr int blocks = T::columns / 16;
  constexpr int bytes = T::fp8 ? 1 : 2;

  if (unit_begin == unit_end) {
    // An empty partial: the merge weighs it by exp2(-inf) = 0.
    for (int h = warp; h < T::group; h += T::warps) {
      float* destination = workspace + partial_index<T>(row, kv_head, splits, split, h);
      for (int i = lane; i < T::partial; i += 32)
        destination[i] = i == 0 ? -INFINITY : 0.0f;
    }
    return;
  }

  uint8_t* storage = attention_smem;
  uint8_t* q_smem = storage + (T::fp8 ? T::stages * T::raw_bytes + T::k_bytes + T::v_bytes
                                      : T::stages * (T::k_bytes + T::v_bytes));
  int32_t* list = reinterpret_cast<int32_t*>(q_smem + T::q_bytes);
  const uint32_t phase = (ptx::shared_address(storage) / 128) % 8;
  const uint32_t q_phase = (ptx::shared_address(q_smem) / 128) % 8;
  uint8_t* raw = storage + (T::fp8 ? T::k_bytes + T::v_bytes : 0);
  auto k_tile = [&](int stage) {
    return storage + (T::fp8 ? 0 : stage * (T::k_bytes + T::v_bytes));
  };
  auto v_tile = [&](int stage) { return k_tile(stage) + T::k_bytes; };

  // This row's selected blocks, then the grouped Q heads (zero padded) as
  // 16-byte copies with the first unit.
  for (int i = threadIdx.x; i < count; i += T::threads)
    list[i] = selected[uint64_t(row) * T::top + i];
  const uint8_t* q_row = reinterpret_cast<const uint8_t*>(q) +
                         (uint64_t(row) * T::q_heads + kv_head * T::group) * T::dim * 2;
  for (int i = threadIdx.x; i < T::padded * T::dim / 8; i += T::threads) {
    const int h = i / (T::dim / 8);
    const int sector = i % (T::dim / 8);
    cp_async_zfill(q_smem + tile_offset<T::padded>(h, sector * 8, q_phase),
                   q_row + (h < T::group ? h : 0) * T::dim * 2 + sector * 16, h < T::group);
  }
  __syncthreads();

  // Key j of the row: a listed block's token or a tail token.
  auto token = [&](int j) {
    return j < listed ? list[j / T::block] * T::block + j % T::block
                      : complete * T::block + (j - listed);
  };
  // Each key's K and V rows are copied by `lanes` threads in 16-byte
  // sectors, so a thread looks up its key's page once per unit; keys past
  // the list are zero filled.
  auto issue = [&](int unit, int stage) {
    constexpr int lanes = T::threads / T::keys;
    constexpr int sectors = T::dim * bytes / 16;
    const int key = threadIdx.x / lanes;
    const int j = unit * T::keys + key;
    const bool valid = j < keys;
    const int t = token(valid ? j : 0);
    const uint64_t page = uint32_t(block_tables[uint64_t(batch) * max_pages + t / T::page]);
    const uint64_t within = (uint64_t(t % T::page) * T::kv_heads + kv_head) * T::dim * bytes;
    const uint8_t* k_row = static_cast<const uint8_t*>(k_cache) + page * k_page_stride_bytes + within;
    const uint8_t* v_row = static_cast<const uint8_t*>(v_cache) + page * v_page_stride_bytes + within;
    uint8_t* stage_raw = raw + stage * T::raw_bytes;
#pragma unroll
    for (int sector = threadIdx.x % lanes; sector < sectors; sector += lanes) {
      if constexpr (T::fp8) {
        cp_async_zfill(stage_raw + key * T::dim + sector * 16, k_row + sector * 16, valid);
        cp_async_zfill(stage_raw + (T::keys + key) * T::dim + sector * 16, v_row + sector * 16,
                       valid);
      } else {
        cp_async_zfill(k_tile(stage) + tile_offset<T::keys>(key, sector * 8, phase),
                       k_row + sector * 16, valid);
        cp_async_zfill(v_tile(stage) + tile_offset<T::keys>(key, sector * 8, phase),
                       v_row + sector * 16, valid);
      }
    }
  };

  const float score_scale = __uint_as_float(scale_bits) *
                            (T::fp8 ? __uint_as_float(k_scale_bits) : 1.0f) *
                            1.4426950408889634f;
  const float v_scale = T::fp8 ? __uint_as_float(v_scale_bits) : 1.0f;
  const int terms = probability_terms(keys);

  ptx::B2 query[T::dim / 16][T::n_tiles];
  float maximum[T::n_tiles][2];
  float denominator[T::n_tiles][2];
  ptx::D4 output[blocks][T::n_tiles];
#pragma unroll
  for (int n = 0; n < T::n_tiles; ++n)
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      maximum[n][r] = -INFINITY;
      denominator[n][r] = 0.0f;
    }
#pragma unroll
  for (int m = 0; m < blocks; ++m)
#pragma unroll
    for (int n = 0; n < T::n_tiles; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) output[m][n].v[i] = 0.0f;

#pragma unroll
  for (int stage = 0; stage + 1 < T::stages; ++stage) {
    if (unit_begin + stage < unit_end) issue(unit_begin + stage, stage);
    cp_async_commit();
  }

  auto accumulate = [&](auto terms_tag, const uint8_t* k_smem, const uint8_t* v_smem,
                        int key0) {
    constexpr int count_terms = decltype(terms_tag)::value;
    const int tile_key = tile * 16;
    ptx::D4 score[T::n_tiles], odd[T::n_tiles];
#pragma unroll
    for (int n = 0; n < T::n_tiles; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) score[n].v[i] = odd[n].v[i] = 0.0f;
#pragma unroll
    for (int ks = 0; ks < T::dim / 16; ++ks) {
      uint32_t a[4];
      ptx::ldmatrix_b16<4>(a, k_smem + tile_offset<T::keys>(
          tile_key + ((lane / 8) % 2) * 8 + lane % 8, ks * 16 + (lane / 16) * 8, phase));
#pragma unroll
      for (int n = 0; n < T::n_tiles; ++n)
        ptx::mma_bf16(ks % 2 ? odd[n] : score[n], ptx::A4{{a[0], a[1], a[2], a[3]}},
                      query[ks][n]);
    }
    const bool edge = key0 + tile_key + 15 >= keys;
#pragma unroll
    for (int n = 0; n < T::n_tiles; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float x = (score[n].v[i] + odd[n].v[i]) * score_scale;
        const bool hidden = edge && key0 + tile_key + g + (i / 2) * 8 >= keys;
        score[n].v[i] = hidden ? -INFINITY : x;
      }
    float correction[T::n_tiles][2];
    bool rescale = false;
#pragma unroll
    for (int n = 0; n < T::n_tiles; ++n)
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        float tile_max = fmaxf(score[n].v[r], score[n].v[r + 2]);
        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 4));
        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 8));
        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 16));
        const float next = fmaxf(maximum[n][r], tile_max);
        const float base = next == -INFINITY ? 0.0f : next;
        correction[n][r] = exp2f(maximum[n][r] - base);
        rescale |= correction[n][r] != 1.0f;
        maximum[n][r] = next;
        score[n].v[r] = exp2f(score[n].v[r] - base);
        score[n].v[r + 2] = exp2f(score[n].v[r + 2] - base);
        denominator[n][r] = denominator[n][r] * correction[n][r] + score[n].v[r] +
                            score[n].v[r + 2];
      }
    if (__any_sync(0xffffffff, rescale)) {
#pragma unroll
      for (int m = 0; m < blocks; ++m)
#pragma unroll
        for (int n = 0; n < T::n_tiles; ++n)
#pragma unroll
          for (int i = 0; i < 4; ++i) output[m][n].v[i] *= correction[n][i % 2];
    }
    // P^T as B fragments: movmatrix turns key g's heads 2c, 2c + 1 into head
    // g's keys 2c, 2c + 1. Each probability is carried as `count_terms` BF16
    // parts (the value and its rounded residuals).
    uint32_t probability[count_terms][T::n_tiles][2];
#pragma unroll
    for (int n = 0; n < T::n_tiles; ++n)
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        float p0 = score[n].v[2 * half];
        float p1 = score[n].v[2 * half + 1];
#pragma unroll
        for (int term = 0; term < count_terms; ++term) {
          const uint32_t pair = bf16_pair(p0, p1);
          p0 -= bf16_low(pair);
          p1 -= bf16_high(pair);
          probability[term][n][half] = movmatrix_trans(pair);
        }
      }
#pragma unroll
    for (int m = 0; m < blocks; ++m) {
      uint32_t a[4];
      ptx::ldmatrix_b16<4, true>(a, v_smem + tile_offset<T::keys>(
          tile_key + (lane / 16) * 8 + lane % 8,
          (part * blocks + m) * 16 + ((lane / 8) % 2) * 8, phase));
#pragma unroll
      for (int n = 0; n < T::n_tiles; ++n)
#pragma unroll
        for (int term = 0; term < count_terms; ++term)
          ptx::mma_bf16(output[m][n], ptx::A4{{a[0], a[1], a[2], a[3]}},
                        ptx::B2{{probability[term][n][0], probability[term][n][1]}});
    }
  };

  for (int unit = unit_begin; unit < unit_end; ++unit) {
    const int t = unit - unit_begin;
    const int stage = t % T::stages;
    // Unit t has landed and every warp has left unit t - 1, whose stage now
    // takes unit t + 1.
    cp_async_wait<T::stages - 2>();
    __syncthreads();
    if (t == 0) {
#pragma unroll
      for (int n = 0; n < T::n_tiles; ++n)
#pragma unroll
        for (int ks = 0; ks < T::dim / 16; ks += 2) {
          uint32_t b[4];
          ptx::ldmatrix_b16<4>(b, q_smem + tile_offset<T::padded>(
              n * 8 + lane % 8, ks * 16 + (lane / 8) * 8, q_phase));
          query[ks][n] = ptx::B2{{b[0], b[1]}};
          query[ks + 1][n] = ptx::B2{{b[2], b[3]}};
        }
    }
    if (unit + T::stages - 1 < unit_end)
      issue(unit + T::stages - 1, (t + T::stages - 1) % T::stages);
    cp_async_commit();
    if constexpr (T::fp8) {
      const uint8_t* stage_raw = raw + stage * T::raw_bytes;
      widen_fp8<T::keys, T::dim, T::threads>(k_tile(0), stage_raw, phase);
      widen_fp8<T::keys, T::dim, T::threads>(v_tile(0), stage_raw + T::keys * T::dim, phase);
      __syncthreads();
    }
    const uint8_t* k_smem = k_tile(T::fp8 ? 0 : stage);
    const uint8_t* v_smem = v_tile(T::fp8 ? 0 : stage);
    const int key0 = unit * T::keys;
    if (terms == 2) {
      accumulate(std::integral_constant<int, 2>{}, k_smem, v_smem, key0);
    } else {
      accumulate(std::integral_constant<int, 3>{}, k_smem, v_smem, key0);
    }
  }
  cp_async_wait<0>();

  // Fold the key tiles: tile 1's warps leave [max, denominator, numerator]
  // per head in the first K stage, and tile 0's warps fold them in. Tile 0
  // always holds the unit's first key, so its maximum is finite.
#pragma unroll
  for (int n = 0; n < T::n_tiles; ++n)
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      denominator[n][r] += __shfl_xor_sync(0xffffffff, denominator[n][r], 4);
      denominator[n][r] += __shfl_xor_sync(0xffffffff, denominator[n][r], 8);
      denominator[n][r] += __shfl_xor_sync(0xffffffff, denominator[n][r], 16);
    }
  if constexpr (T::tiles == 2) {
    constexpr int fold_floats = 2 * T::padded + T::columns * T::padded;
    static_assert(T::parts * fold_floats * 4 <= T::k_bytes + T::v_bytes, "fold fits a stage");
    float* fold = reinterpret_cast<float*>(k_tile(0)) + part * fold_floats;
    __syncthreads();
    if (tile == 1) {
#pragma unroll
      for (int n = 0; n < T::n_tiles; ++n) {
        if (g == 0) {
#pragma unroll
          for (int r = 0; r < 2; ++r) {
            fold[2 * (n * 8 + 2 * c + r)] = maximum[n][r];
            fold[2 * (n * 8 + 2 * c + r) + 1] = denominator[n][r];
          }
        }
#pragma unroll
        for (int m = 0; m < blocks; ++m)
#pragma unroll
          for (int half = 0; half < 2; ++half)
            *reinterpret_cast<float2*>(fold + 2 * T::padded +
                                       (m * 16 + g + half * 8) * T::padded + n * 8 + 2 * c) =
                make_float2(output[m][n].v[2 * half], output[m][n].v[2 * half + 1]);
      }
    }
    __syncthreads();
    if (tile == 1) return;
#pragma unroll
    for (int n = 0; n < T::n_tiles; ++n)
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const int h = n * 8 + 2 * c + r;
        const float other = fold[2 * h];
        const float next = fmaxf(maximum[n][r], other);
        const float mine = exp2f(maximum[n][r] - next);
        const float theirs = exp2f(other - next);
        denominator[n][r] = denominator[n][r] * mine + fold[2 * h + 1] * theirs;
        maximum[n][r] = next;
#pragma unroll
        for (int m = 0; m < blocks; ++m)
#pragma unroll
          for (int half = 0; half < 2; ++half)
            output[m][n].v[2 * half + r] =
                output[m][n].v[2 * half + r] * mine +
                fold[2 * T::padded + (m * 16 + g + half * 8) * T::padded + h] * theirs;
      }
  }

#pragma unroll
  for (int n = 0; n < T::n_tiles; ++n)
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int h = n * 8 + 2 * c + r;
      if (h >= T::group) continue;
      if (splits == 1) {
        const float normalizer = v_scale / denominator[n][r];
        uint16_t* destination =
            out + (uint64_t(row) * T::q_heads + kv_head * T::group + h) * T::dim +
            part * T::columns;
#pragma unroll
        for (int m = 0; m < blocks; ++m)
#pragma unroll
          for (int half = 0; half < 2; ++half)
            destination[m * 16 + g + half * 8] =
                to_bf16(output[m][n].v[2 * half + r] * normalizer);
      } else {
        float* destination = workspace + partial_index<T>(row, kv_head, splits, split, h);
        if (part == 0 && g == 0) {
          destination[0] = maximum[n][r];
          destination[1] = denominator[n][r];
        }
#pragma unroll
        for (int m = 0; m < blocks; ++m)
#pragma unroll
          for (int half = 0; half < 2; ++half)
            destination[2 + part * T::columns + m * 16 + g + half * 8] =
                output[m][n].v[2 * half + r] * v_scale;
      }
    }
}

// Grid [rows, Q heads], eight warps: folds a row's split partials and writes
// BF16 output. Warp 0 loads every split's maximum and denominator at once
// (at most 64 splits) and reduces them with a fixed shuffle tree. Warp w then
// sums the V columns of splits w, w + 8, ..., all loads in flight at once,
// and the warps' sums fold in warp order. The fold order is fixed, so the
// output is deterministic.
constexpr int merge_warps = 8;

template<class T>
__device__ void merge(const float* workspace, const uint32_t* live_rows, uint16_t* out,
                      uint32_t splits) {
  static_assert(T::dim % 32 == 0);
  ptx::griddep_launch_dependents();
  __shared__ float weights[64];
  __shared__ float normalizer;
  __shared__ float sums[merge_warps][T::dim];
  ptx::griddep_wait();
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  if (row >= int(*live_rows)) return;
  const int kv_head = head / T::group;
  const int h = head % T::group;
  auto partial = [&](uint32_t split) {
    return workspace + partial_index<T>(row, kv_head, splits, split, h);
  };
  if (threadIdx.x < 32) {
    const uint32_t lane = threadIdx.x;
    float maximum[2], denominator[2];
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const uint32_t split = lane + 32 * k;
      maximum[k] = split < splits ? partial(split)[0] : -INFINITY;
      denominator[k] = split < splits ? partial(split)[1] : 0.0f;
    }
    float top = fmaxf(maximum[0], maximum[1]);
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
      top = fmaxf(top, __shfl_xor_sync(0xffffffff, top, offset));
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const float weight = exp2f(maximum[k] - top);
      if (lane + 32 * k < 64) weights[lane + 32 * k] = weight;
      sum += denominator[k] * weight;
    }
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, offset);
    if (lane == 0) normalizer = 1.0f / sum;
  }
  __syncthreads();
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  constexpr int columns = T::dim / 32;
  float sum[columns] = {};
#pragma unroll
  for (int k = 0; k < 64 / merge_warps; ++k) {
    const uint32_t split = warp + merge_warps * k;
    if (split >= splits) break;
    const float* numerator = partial(split) + 2;
    const float weight = weights[split];
#pragma unroll
    for (int c = 0; c < columns; ++c) sum[c] += numerator[lane + 32 * c] * weight;
  }
#pragma unroll
  for (int c = 0; c < columns; ++c) sums[warp][lane + 32 * c] = sum[c];
  __syncthreads();
  for (int d = threadIdx.x; d < T::dim; d += blockDim.x) {
    float numerator = 0.0f;
#pragma unroll
    for (int w = 0; w < merge_warps; ++w) numerator += sums[w][d];
    out[(uint64_t(row) * T::q_heads + head) * T::dim + d] = to_bf16(numerator * normalizer);
  }
}

} // namespace chop::attention::sparse
