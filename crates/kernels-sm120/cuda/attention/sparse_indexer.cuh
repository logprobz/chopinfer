#pragma once

// QSA indexer (decision 0084): query norm and M-RoPE, the raw-key ring and
// block-key cache writes, block scoring and top-k block selection. See
// SPARSE.md. Every entry takes the packed step rows: `row_batch[row]` names
// the row's sequence (its block-table row and slot), `positions[row]` its
// cache position, and `live_rows` the live row count. A sequence's rows are
// contiguous with consecutive positions.

#include <cuda_bf16.h>
#include <stdint.h>
#include "tile.cuh"

namespace chop::attention::sparse {

template<int Heads, int Dim, int Rotary, int SectionH, int SectionW, int Block,
         int Ring, int Page, int TopBlocks>
struct Indexer {
  static_assert((Heads == 4 || Heads == 8 || Heads == 16) && Dim == 128 &&
                (Rotary == 32 || Rotary == 64),
                "a warp holds one 128-wide head; 4, 8 or 16 heads share a score");
  static_assert(Block > 1 && Ring >= Block && Ring % Block == 0);
  static_assert(Page % Block == 0 && TopBlocks > 0);
  static constexpr int heads = Heads, dim = Dim, rotary = Rotary, pairs = Rotary / 2;
  static constexpr int block = Block, ring = Ring, page = Page;
  static constexpr int top = TopBlocks;
  static constexpr int page_blocks = Page / Block;
  // A projection row holds the query heads, then the raw key.
  static constexpr int width = (Heads + 1) * Dim;

  // The position axis (0 = T, 1 = H, 2 = W) of rotary pair `pair`.
  static __device__ __forceinline__ int axis(int pair) {
    return pair % 3 == 1 && pair < 3 * SectionH ? 1 :
           pair % 3 == 2 && pair < 3 * SectionW ? 2 : 0;
  }
};

__device__ __forceinline__ float bf16(uint16_t x) { return __uint_as_float(uint32_t(x) << 16); }
__device__ __forceinline__ uint16_t to_bf16(float x) {
  const __nv_bfloat16 value = __float2bfloat16_rn(x);
  return *reinterpret_cast<const uint16_t*>(&value);
}

__device__ __forceinline__ void load4(float (&x)[4], const uint16_t* source) {
  const uint2 packed = *reinterpret_cast<const uint2*>(source);
  x[0] = __uint_as_float(packed.x << 16);
  x[1] = __uint_as_float(packed.x & 0xffff0000u);
  x[2] = __uint_as_float(packed.y << 16);
  x[3] = __uint_as_float(packed.y & 0xffff0000u);
}
__device__ __forceinline__ void store4(uint16_t* target, const float (&x)[4]) {
  *reinterpret_cast<uint2*>(target) =
      make_uint2(uint32_t(to_bf16(x[0])) | uint32_t(to_bf16(x[1])) << 16,
                 uint32_t(to_bf16(x[2])) | uint32_t(to_bf16(x[3])) << 16);
}

// Gemma RMSNorm `x * rsqrt(mean(x^2) + eps) * (1 + w)` of one head held by a
// warp (lane l holds dimensions 4l to 4l + 3), then NeoX RoPE over the first
// rotary dimensions with M-RoPE axes. The table is F32 [position][pair][cos, sin].
template<class I>
__device__ __forceinline__ void norm_rotate(float (&x)[4], const uint16_t* weight,
                                            const int32_t* rope_position,
                                            const float* cos_sin, float eps) {
  const int lane = threadIdx.x % 32;
  float square = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
#pragma unroll
  for (int offset = 16; offset; offset >>= 1)
    square += __shfl_xor_sync(0xffffffff, square, offset);
  const float inverse = 1.0f / sqrtf(square / I::dim + eps);
  float w[4];
  load4(w, weight + 4 * lane);
#pragma unroll
  for (int i = 0; i < 4; ++i) x[i] = x[i] * inverse * (1.0f + w[i]);
  // Each half of the rotary pairs occupies rotary / 8 lanes.
  float mate[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) mate[i] = __shfl_xor_sync(0xffffffff, x[i], I::rotary / 8);
  if (lane < I::rotary / 4) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int pair = (4 * lane + i) % I::pairs;
      const uint64_t position = uint32_t(rope_position[I::axis(pair)]);
      const float2 angle =
          reinterpret_cast<const float2*>(cos_sin)[position * I::pairs + pair];
      x[i] = lane < I::rotary / 8 ? x[i] * angle.x - mate[i] * angle.y
                      : x[i] * angle.x + mate[i] * angle.y;
    }
  }
}

// One CTA per row, one warp per head: normalized, rotated BF16 queries
// [rows][heads][dim].
template<class I>
__device__ void indexer_query(const uint16_t* projection, const uint16_t* weight,
                              const int32_t* rope_positions, const float* cos_sin,
                              const uint32_t* live_rows, uint16_t* q, uint32_t eps_bits) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  const int head = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (row >= *live_rows) return;
  float x[4];
  load4(x, projection + uint64_t(row) * I::width + head * I::dim + 4 * lane);
  norm_rotate<I>(x, weight, rope_positions + 3 * uint64_t(row), cos_sin,
                 __uint_as_float(eps_bits));
  store4(q + (uint64_t(row) * I::heads + head) * I::dim + 4 * lane, x);
}

// Block b of a sequence is row b % page_blocks of the block-key page that
// holds its tokens, through the KV block table.
template<class I>
__device__ __forceinline__ uint8_t* block_key_row(uint8_t* block_keys, const int32_t* table,
                                                  int batch, int b, uint32_t max_pages,
                                                  uint64_t page_stride_bytes) {
  const uint64_t page = uint32_t(table[uint64_t(batch) * max_pages + b / I::page_blocks]);
  return block_keys + page * page_stride_bytes + uint64_t(b % I::page_blocks) * I::dim * 2;
}

// Pools the block's raw keys (a warp holds 4 dimensions per lane): the F64
// sum in position order, as the oracle adds them, so cancellation between
// keys of different magnitudes loses no term, times 1 / block, rounded to F32
// and then BF16. Then the norm and RoPE at the block's first token, written
// as BF16.
template<class I>
__device__ __forceinline__ void write_block(const double (&sum)[4], const uint16_t* weight,
                                            const int32_t* first_rope, const float* cos_sin,
                                            float eps, uint8_t* destination) {
  const int lane = threadIdx.x % 32;
  float x[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) x[i] = bf16(to_bf16(float(sum[i] * (1.0 / I::block))));
  norm_rotate<I>(x, weight, first_rope, cos_sin, eps);
  store4(reinterpret_cast<uint16_t*>(destination) + 4 * lane, x);
}

// A warp per row. A row that completes a block whose tokens are all in this
// launch pools them from the projection rows. The first row of each sequence
// (its leader) pools a block that began before this launch from the ring,
// and only after that writes the ring rows of the sequence's last `ring`
// positions, so no other warp reads or writes that sequence's ring.
//
// `anchor_slots[b]` >= 0 names a slot of the anchor ring (`anchor_keys`,
// `anchor_positions`, laid out as the ring) that also receives the ring a
// sequence ending before position `anchor_ends[b]` would leave: its rows of
// positions [anchor_ends[b] - ring, anchor_ends[b]), which the caller keeps
// within the sequence's rows of this launch.
template<class I>
__device__ void indexer_key_write(
    const uint16_t* projection, const uint16_t* weight, const int32_t* rope_positions,
    const float* cos_sin, const int32_t* row_batch, const int32_t* positions,
    const uint32_t* live_rows, const int32_t* slot_ids, uint16_t* ring_keys,
    int32_t* ring_positions, const int32_t* anchor_slots, const int32_t* anchor_ends,
    uint16_t* anchor_keys, int32_t* anchor_positions, const int32_t* block_tables,
    uint8_t* block_keys, uint32_t max_pages, uint64_t block_page_stride_bytes,
    uint32_t eps_bits) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const int warps = blockDim.x / 32;
  const int row = blockIdx.x * warps + threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int live = int(*live_rows);
  if (row >= live) return;
  const float eps = __uint_as_float(eps_bits);
  const int batch = row_batch[row];
  const int position = positions[row];
  auto raw_key = [&](int r) {
    return projection + uint64_t(r) * I::width + I::heads * I::dim + 4 * lane;
  };
  auto add_key = [&](double (&sum)[4], const uint16_t* source) {
    float x[4];
    load4(x, source);
#pragma unroll
    for (int i = 0; i < 4; ++i) sum[i] += x[i];
  };
  if (position % I::block == I::block - 1) {
    const int first = row - (I::block - 1);
    if (first >= 0 && row_batch[first] == batch && positions[first] == position - (I::block - 1)) {
      double sum[4] = {0.0, 0.0, 0.0, 0.0};
      for (int r = first; r <= row; ++r) add_key(sum, raw_key(r));
      write_block<I>(sum, weight, rope_positions + 3 * uint64_t(first), cos_sin, eps,
                     block_key_row<I>(block_keys, block_tables, batch, position / I::block,
                                      max_pages, block_page_stride_bytes));
    }
  }
  if (row > 0 && row_batch[row - 1] == batch) return;

  // The leader: the sequence's rows are [row, end).
  int low = row, high = live;
  while (high - low > 1) {
    const int middle = (low + high) / 2;
    if (row_batch[middle] == batch) low = middle; else high = middle;
  }
  const int end = high;
  const uint64_t slot = uint32_t(slot_ids[batch]);
  uint16_t* ring = ring_keys + slot * I::ring * I::dim;
  int32_t* ring_rope = ring_positions + slot * I::ring * 3;
  const int start = position - position % I::block;
  const int completing = start + I::block - 1;
  if (start < position && completing - position < end - row) {
    double sum[4] = {0.0, 0.0, 0.0, 0.0};
    for (int p = start; p <= completing; ++p)
      add_key(sum, p < position ? ring + (p % I::ring) * I::dim + 4 * lane
                                : raw_key(row + (p - position)));
    write_block<I>(sum, weight, ring_rope + (start % I::ring) * 3, cos_sin, eps,
                   block_key_row<I>(block_keys, block_tables, batch, start / I::block,
                                    max_pages, block_page_stride_bytes));
  }
  __syncwarp();
  for (int r = max(row, end - I::ring); r < end; ++r) {
    const int slot_row = (position + (r - row)) % I::ring;
    *reinterpret_cast<uint2*>(ring + slot_row * I::dim + 4 * lane) =
        *reinterpret_cast<const uint2*>(raw_key(r));
    if (lane < 3) ring_rope[slot_row * 3 + lane] = rope_positions[3 * uint64_t(r) + lane];
  }
  const int32_t anchor = anchor_slots[batch];
  if (anchor < 0) return;
  const int anchor_end = anchor_ends[batch];
  uint16_t* kept = anchor_keys + uint64_t(uint32_t(anchor)) * I::ring * I::dim;
  int32_t* kept_rope = anchor_positions + uint64_t(uint32_t(anchor)) * I::ring * 3;
  for (int p = max(position, anchor_end - I::ring); p < anchor_end && p - position < end - row;
       ++p) {
    const int r = row + (p - position);
    const int slot_row = p % I::ring;
    *reinterpret_cast<uint2*>(kept + slot_row * I::dim + 4 * lane) =
        *reinterpret_cast<const uint2*>(raw_key(r));
    if (lane < 3) kept_rope[slot_row * 3 + lane] = rope_positions[3 * uint64_t(r) + lane];
  }
}

// The exclusive prefix sum of one value per thread over a CTA of `Threads`.
template<int Threads>
__device__ __forceinline__ uint32_t exclusive_scan(uint32_t value, uint32_t* warp_totals) {
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  uint32_t inclusive = value;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const uint32_t other = __shfl_up_sync(0xffffffff, inclusive, offset);
    if (lane >= offset) inclusive += other;
  }
  if (lane == 31) warp_totals[warp] = inclusive;
  __syncthreads();
  if (warp == 0) {
    uint32_t sum = lane < Threads / 32 ? warp_totals[lane] : 0;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      const uint32_t other = __shfl_up_sync(0xffffffff, sum, offset);
      if (lane >= offset) sum += other;
    }
    if (lane < Threads / 32) warp_totals[lane] = sum;
  }
  __syncthreads();
  const uint32_t before = warp == 0 ? 0 : warp_totals[warp - 1];
  __syncthreads();
  return before + inclusive - value;
}

// Block scores: `sum_h relu(q_h . key_b) / sqrt(dim)` for every complete
// block b of every row with more than `top` complete blocks (rows with fewer
// keep every block and skip scoring). A CTA takes `Rows` consecutive rows and
// one fixed chunk of blocks; rows of the same sequence share each
// block-key stage. With `ByRun`, CTA g instead takes the rows of the step's
// g-th sequence (a verify item), in groups of `Rows`, so an item's block keys
// are read once. The MMA rows are the (row, head) columns and the blocks
// are n8 columns: warp w takes `warp_m_tiles` m16 tiles of the columns and
// 1 / groups of each 64-block stage, so wide row groups load each block key
// once for 256 head rows (64, 32 or 16 queries). LiveChunks partitions live
// stage counts across CTAs; each CTA can visit several single-row tiles.
template<class I, int Rows>
struct Scores {
  static constexpr int warps = 8, threads = warps * 32;
  static constexpr int columns = Rows * I::heads;
  static_assert(Rows <= 256 / I::heads, "score tiles hold at most 256 heads");
  static constexpr int m_tiles = (columns + 15) / 16;
  static_assert(m_tiles == 1 || m_tiles == 2 || m_tiles == 4 || m_tiles == 8 || m_tiles == 16);
  static constexpr int warp_m_tiles = m_tiles > warps ? m_tiles / warps : 1;
  static constexpr int m_groups = m_tiles / warp_m_tiles;
  static constexpr int groups = warps / m_groups;
  static constexpr int stage_blocks = 64;
  static constexpr int warp_blocks = stage_blocks / groups;
  static constexpr int n_tiles = warp_blocks / 8;
  static constexpr int stage_bytes = stage_blocks * I::dim * 2;
  static constexpr int q_bytes = m_tiles * 16 * I::dim * 2;
  // Two stages beside 256 head rows, and for single rows so two CTAs share
  // an SM; three otherwise.
  static constexpr int stages = q_bytes > 32768 || Rows == 1 ? 2 : 3;
  // Single-row CTAs take smaller chunks so a small decode batch fills the
  // SMs.
  static constexpr int chunk_blocks = Rows == 1 ? 64 : 1024;
  static constexpr int smem = stages * stage_bytes + q_bytes;
};

template<class I, int Rows, bool ByRun, bool LiveChunks = false>
__device__ void block_scores(const uint16_t* q, const uint8_t* block_keys,
                             const int32_t* block_tables, const int32_t* row_batch,
                             const int32_t* positions, const uint32_t* live_rows, float* scores,
                             uint32_t max_pages, uint64_t block_page_stride_bytes,
                             uint32_t score_stride) {
  using S = Scores<I, Rows>;
  ptx::griddep_launch_dependents();
  extern __shared__ __align__(1024) uint8_t score_smem[];
  __shared__ int row_batches[Rows], row_complete[Rows];
  static_assert(!LiveChunks || (Rows == 1 && !ByRun));
  __shared__ int bounds[2];
  __shared__ uint32_t stage_prefix[LiveChunks ? S::threads + 1 : 1];
  __shared__ uint32_t warp_totals[32];
  ptx::griddep_wait();
  const int live = int(*live_rows);
  int stage_first = 0, stage_last = 0;
  int first_row = blockIdx.x * Rows, end_row = min(first_row + Rows, live);
  if constexpr (ByRun) {
    // Row t starts a run when its sequence differs from row t - 1's; the
    // runs are ranked by a CTA scan over at most `threads` live rows.
    const int t = threadIdx.x;
    const bool start = t < live && (t == 0 || row_batch[t] != row_batch[t - 1]);
    if (t == 0) bounds[0] = bounds[1] = -1;
    __syncthreads();
    const uint32_t rank = exclusive_scan<S::threads>(start, warp_totals);
    uint32_t run = blockIdx.x;
    if (start && rank == run) bounds[0] = t;
    if (start && rank == run + 1) bounds[1] = t;
    __syncthreads();
    first_row = bounds[0];
    end_row = bounds[1] < 0 ? live : bounds[1];
    if (first_row < 0) return;
  }
  if constexpr (LiveChunks) {
    // Flatten live 64-block stages across all rows. Short or inactive rows
    // cannot reserve CTAs that a long row needs. Captured grids stay fixed.
    if (live < 0 || live > S::threads) __trap();
    const int t = threadIdx.x;
    const int complete = t < live ? (positions[t] + 1) / I::block : 0;
    const uint32_t stages = complete > I::top
        ? (complete + S::stage_blocks - 1) / S::stage_blocks : 0;
    const uint32_t prefix = exclusive_scan<S::threads>(stages, warp_totals);
    stage_prefix[t] = prefix;
    if (t == S::threads - 1) stage_prefix[S::threads] = prefix + stages;
    __syncthreads();
    const int total = stage_prefix[S::threads];
    const int cta = blockIdx.x + gridDim.x * blockIdx.y;
    const int ctas = gridDim.x * gridDim.y;
    stage_first = total * cta / ctas;
    stage_last = total * (cta + 1) / ctas;
    if (stage_first == stage_last) return;
    // Upper bound of the starting stage skips rows with no score work.
    int lo = 0, hi = live;
    while (lo < hi) {
      const int mid = (lo + hi) / 2;
      if (stage_prefix[mid + 1] <= stage_first) lo = mid + 1;
      else hi = mid;
    }
    first_row = lo;
    end_row = live;
  }
  if (first_row >= live) return;
  const int chunk0 = blockIdx.y * S::chunk_blocks;
  uint8_t* stage_base = score_smem;
  uint8_t* q_smem = score_smem + S::stages * S::stage_bytes;
  const uint32_t phase = (ptx::shared_address(stage_base) / 128) % 8;
  const uint32_t q_phase = (ptx::shared_address(q_smem) / 128) % 8;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int g = lane / 4;
  const int c = lane % 4;
  const int m_tile0 = (warp % S::m_groups) * S::warp_m_tiles;
  const int group = warp / S::m_groups;
  const float scale = rsqrtf(float(I::dim));

  for (int row0 = first_row; row0 < end_row; row0 += Rows) {
    if constexpr (LiveChunks) {
      if (stage_prefix[row0] >= stage_last) break;
    }
    const int rows = min(Rows, end_row - row0);
    __syncthreads();
    for (int r = threadIdx.x; r < Rows; r += S::threads) {
      row_batches[r] = r < rows ? row_batch[row0 + r] : -1;
      row_complete[r] = r < rows ? (positions[row0 + r] + 1) / I::block : 0;
    }
    __syncthreads();
    for (int run = 0; run < rows;) {
      const int batch = row_batches[run];
      int run_end = run + 1;
      int blocks = row_complete[run] > I::top ? row_complete[run] : 0;
      while (run_end < rows && row_batches[run_end] == batch) {
        if (row_complete[run_end] > I::top) blocks = max(blocks, row_complete[run_end]);
        ++run_end;
      }
      const int first = LiveChunks
          ? max(stage_first - int(stage_prefix[row0]), 0) * S::stage_blocks : chunk0;
      const int last = LiveChunks
          ? min((stage_last - int(stage_prefix[row0])) * S::stage_blocks, blocks)
          : min(first + S::chunk_blocks, blocks);
      if (first < last) {
        // Q columns (row - row0) * heads + h of this run; the others are zero.
        __syncthreads();
        for (int i = threadIdx.x; i < S::m_tiles * 16 * I::dim / 8; i += S::threads) {
          const int column = i / (I::dim / 8);
          const int sector = i % (I::dim / 8);
          const int r = column / I::heads;
          const bool valid = r >= run && r < run_end;
          cp_async_zfill(q_smem + tile_offset<S::m_tiles * 16>(column, sector * 8, q_phase),
                         q + (uint64_t(row0 + (valid ? r : run)) * I::heads + column % I::heads) *
                                 I::dim + sector * 8,
                         valid);
        }
        // Four threads copy each block key, looking its page up once.
        auto issue = [&](int stage, int b0) {
          static_assert(S::threads == 4 * S::stage_blocks);
          uint8_t* target = stage_base + stage * S::stage_bytes;
          const int key = threadIdx.x / 4;
          const int b = b0 + key;
          const bool valid = b < last;
          const uint8_t* source = block_key_row<I>(const_cast<uint8_t*>(block_keys), block_tables,
                                                   batch, valid ? b : first, max_pages,
                                                   block_page_stride_bytes);
#pragma unroll
          for (int sector = threadIdx.x % 4; sector < I::dim / 8; sector += 4)
            cp_async_zfill(target + tile_offset<S::stage_blocks>(key, sector * 8, phase),
                           source + sector * 16, valid);
        };
        const int steps = (last - first + S::stage_blocks - 1) / S::stage_blocks;
#pragma unroll
        for (int s = 0; s + 1 < S::stages; ++s) {
          if (s < steps) issue(s, first + s * S::stage_blocks);
          cp_async_commit();
        }
        ptx::A4 query[S::warp_m_tiles][I::dim / 16];
        for (int step = 0; step < steps; ++step) {
          cp_async_wait<S::stages - 2>();
          __syncthreads();
          if (step == 0) {
#pragma unroll
            for (int mt = 0; mt < S::warp_m_tiles; ++mt)
#pragma unroll
              for (int ks = 0; ks < I::dim / 16; ++ks)
                ptx::ldmatrix_b16<4>(query[mt][ks].v, q_smem + tile_offset<S::m_tiles * 16>(
                    (m_tile0 + mt) * 16 + (lane % 16), ks * 16 + (lane / 16) * 8, q_phase));
          }
          if (step + S::stages - 1 < steps)
            issue((step + S::stages - 1) % S::stages,
                  first + (step + S::stages - 1) * S::stage_blocks);
          cp_async_commit();
          const uint8_t* keys = stage_base + (step % S::stages) * S::stage_bytes;
          const int b0 = first + step * S::stage_blocks + group * S::warp_blocks;
#pragma unroll
          for (int n = 0; n < S::n_tiles; ++n) {
            ptx::D4 even[S::warp_m_tiles], odd[S::warp_m_tiles];
#pragma unroll
            for (int mt = 0; mt < S::warp_m_tiles; ++mt) even[mt] = odd[mt] = ptx::D4{};
#pragma unroll
            for (int ks = 0; ks < I::dim / 16; ks += 2) {
              uint32_t b[4];
              ptx::ldmatrix_b16<4>(b, keys + tile_offset<S::stage_blocks>(
                  group * S::warp_blocks + n * 8 + lane % 8, ks * 16 + (lane / 8) * 8, phase));
#pragma unroll
              for (int mt = 0; mt < S::warp_m_tiles; ++mt) {
                ptx::mma_bf16(even[mt], query[mt][ks], ptx::B2{{b[0], b[1]}});
                ptx::mma_bf16(odd[mt], query[mt][ks + 1], ptx::B2{{b[2], b[3]}});
              }
            }
#pragma unroll
            for (int mt = 0; mt < S::warp_m_tiles; ++mt) {
              // MMA row 16 m + g + 8 half is (query row, head). Four heads
              // span lane bits 2/3; eight span 2/3/4. Sixteen also fold the
              // two register halves, after each head's ReLU.
              float relu[4];
#pragma unroll
              for (int i = 0; i < 4; ++i) {
                const float x = even[mt].v[i] + odd[mt].v[i];
                relu[i] = x > 0.0f ? x : 0.0f;
                if constexpr (I::heads <= 8) {
                  relu[i] += __shfl_xor_sync(0xffffffff, relu[i], 4);
                  relu[i] += __shfl_xor_sync(0xffffffff, relu[i], 8);
                  if constexpr (I::heads == 8)
                    relu[i] += __shfl_xor_sync(0xffffffff, relu[i], 16);
                }
              }
              if constexpr (I::heads == 16) {
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                  relu[i] += relu[i + 2];
                  relu[i] += __shfl_xor_sync(0xffffffff, relu[i], 4);
                  relu[i] += __shfl_xor_sync(0xffffffff, relu[i], 8);
                  relu[i] += __shfl_xor_sync(0xffffffff, relu[i], 16);
                }
              }
              if (g % I::heads == 0) {
                const int b = b0 + n * 8 + 2 * c;
#pragma unroll
                for (int half = 0; half < (I::heads == 16 ? 1 : 2); ++half) {
                  const int r = (m_tile0 + mt) * (16 / I::heads) +
                                g / I::heads + (8 / I::heads) * half;
                  if (r < run || r >= run_end || row_complete[r] <= I::top) continue;
                  float* destination = scores + uint64_t(row0 + r) * score_stride + b;
                  if (b + 1 < row_complete[r]) {
                    *reinterpret_cast<float2*>(destination) =
                        make_float2(relu[2 * half] * scale, relu[2 * half + 1] * scale);
                  } else if (b < row_complete[r]) {
                    destination[0] = relu[2 * half] * scale;
                  }
                }
              }
            }
          }
        }
        cp_async_wait<0>();
      }
      run = run_end;
    }
  }
}

// Top-k block selection: one CTA per row. Rows with at most `top` complete
// blocks keep them all. Otherwise a radix select over the scores' bits
// (nonnegative F32, so they order as unsigned integers) finds the `top`-th
// largest value T. A histogram of the top 11 bits finds T's bin; a second
// pass marks every block above that bin in a bitmask and gathers the bin's
// blocks into shared memory, where digits of 11 and 10 bits find T. Blocks
// above T are kept, then the lowest-index blocks equal to T, written in
// ascending order from the bitmasks. A bin too large to gather (heavy ties)
// runs its digit passes over global memory instead.
template<class I>
struct TopK {
  static constexpr int threads = 512;
  static constexpr int bins = 2048;
  static constexpr int max_blocks = 65536;
  static constexpr int words = max_blocks / 32;
  // Bitmask words each thread owns in the output pass.
  static constexpr int thread_words = words / threads;
  static constexpr int gather = 8192;
  static constexpr int smem = (bins + 2 * words + 2 * gather) * 4;
};

// After the histogram is complete, finds the digit d with
// count(> d) < want <= count(>= d): found = {d, count(> d)}. Thread t sums
// the t-th slice of digits from the top, and one CTA scan ranks the slices.
template<int Threads>
__device__ __forceinline__ void find_digit(const uint32_t* histogram, uint32_t digits,
                                           uint32_t want, uint32_t* found,
                                           uint32_t* warp_totals) {
  __syncthreads();
  const int span = (digits + Threads - 1) / Threads;
  const int top_digit = int(digits) - 1 - int(threadIdx.x) * span;
  uint32_t sum = 0;
  for (int j = 0; j < span && top_digit - j >= 0; ++j) sum += histogram[top_digit - j];
  uint32_t running = exclusive_scan<Threads>(sum, warp_totals);
  if (running < want && running + sum >= want) {
    for (int j = 0; j < span; ++j) {
      const uint32_t count = histogram[top_digit - j];
      if (running + count >= want) {
        found[0] = top_digit - j;
        found[1] = running;
        break;
      }
      running += count;
    }
  }
  __syncthreads();
}

// Counts a valid digit. Plain shared atomics measured faster than
// aggregating equal digits with __match_any_sync.
__device__ __forceinline__ void add_digit(uint32_t* histogram, uint32_t digit, bool valid) {
  if (valid) atomicAdd(&histogram[digit], 1u);
}

template<class I>
__device__ void block_topk(const float* scores, const int32_t* positions,
                           const uint32_t* live_rows, int32_t* selected,
                           uint32_t score_stride) {
  using K = TopK<I>;
  ptx::griddep_launch_dependents();
  extern __shared__ __align__(16) uint32_t topk_smem[];
  __shared__ uint32_t warp_totals[32];
  __shared__ uint32_t found[2];
  __shared__ uint32_t gathered;
  ptx::griddep_wait();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  const int complete = (positions[row] + 1) / I::block;
  int32_t* out = selected + uint64_t(row) * I::top;
  if (complete <= I::top) {
    for (int b = threadIdx.x; b < complete; b += K::threads) out[b] = b;
    return;
  }
  uint32_t* histogram = topk_smem;
  uint32_t* chosen = histogram + K::bins;
  uint32_t* tied = chosen + K::words;
  uint32_t* gathered_value = tied + K::words;
  uint32_t* gathered_block = gathered_value + K::gather;
  const uint32_t* bits = reinterpret_cast<const uint32_t*>(scores) + uint64_t(row) * score_stride;
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  constexpr int warps = K::threads / 32;
  auto clear_histogram = [&]() {
    for (int i = threadIdx.x; i < K::bins; i += K::threads) histogram[i] = 0;
  };
  // Warp w visits tiles of 32 * loads blocks w, w + warps, ...; lane l
  // loads blocks tile + 32 i + l, so word tile / 32 + i of a bitmask is one
  // ballot of load i. A single CTA reads a whole row, so every load of a
  // tile is in flight before any is used.
  constexpr int loads = 16;
  auto each_load = [&](auto&& visit_tile) {
    for (int base = warp * 32 * loads; base < complete; base += warps * 32 * loads) {
      uint32_t v[loads];
#pragma unroll
      for (int i = 0; i < loads; ++i) {
        const int b = base + 32 * i + lane;
        v[i] = b < complete ? bits[b] : 0;
      }
      visit_tile(base, v);
    }
  };
  auto each_tile = [&](auto&& visit) {
    each_load([&](int base, const uint32_t (&v)[loads]) {
#pragma unroll
      for (int i = 0; i < loads; ++i)
        visit(base + 32 * i, v[i], base + 32 * i + lane < complete);
    });
  };

  clear_histogram();
  for (int i = threadIdx.x; i < K::words; i += K::threads) chosen[i] = tied[i] = 0;
  if (threadIdx.x == 0) gathered = 0;
  __syncthreads();
  each_tile([&](int, uint32_t v, bool valid) { add_digit(histogram, v >> 21, valid); });
  find_digit<K::threads>(histogram, 2048, I::top, found, warp_totals);
  const uint32_t bin = found[0];
  uint32_t prefix = bin << 21, mask = 0x7ffu << 21, want = I::top - found[1];
  // Blocks above the bin are kept; the bin's blocks are gathered, with one
  // reservation per warp tile.
  each_load([&](int base, const uint32_t (&v)[loads]) {
    uint32_t members[loads], total = 0;
#pragma unroll
    for (int i = 0; i < loads; ++i) {
      const bool valid = base + 32 * i + lane < complete;
      const uint32_t above = __ballot_sync(0xffffffff, valid && (v[i] >> 21) > bin);
      members[i] = __ballot_sync(0xffffffff, valid && (v[i] >> 21) == bin);
      if (lane == i) chosen[(base + 32 * i) / 32] = above;
      total += __popc(members[i]);
    }
    if (total == 0) return;
    uint32_t slot = 0;
    if (lane == 0) slot = atomicAdd(&gathered, total);
    slot = __shfl_sync(0xffffffff, slot, 0);
#pragma unroll
    for (int i = 0; i < loads; ++i) {
      const uint32_t at = slot + __popc(members[i] & ((1u << lane) - 1));
      if ((members[i] >> lane) & 1 && at < K::gather) {
        gathered_value[at] = v[i];
        gathered_block[at] = base + 32 * i + lane;
      }
      slot += __popc(members[i]);
    }
  });
  __syncthreads();
  const uint32_t count = gathered;
  const bool local = count <= K::gather;
#pragma unroll
  for (int pass = 0; pass < 2; ++pass) {
    const int shift = pass == 0 ? 10 : 0;
    const uint32_t digits = pass == 0 ? 2048 : 1024;
    clear_histogram();
    __syncthreads();
    if (local) {
      for (uint32_t i0 = warp * 32; i0 < count; i0 += K::threads) {
        const uint32_t i = i0 + lane;
        const uint32_t v = i < count ? gathered_value[i] : 0;
        add_digit(histogram, (v >> shift) & (digits - 1), i < count && (v & mask) == prefix);
      }
    } else {
      each_tile([&](int, uint32_t v, bool valid) {
        add_digit(histogram, (v >> shift) & (digits - 1), valid && (v & mask) == prefix);
      });
    }
    find_digit<K::threads>(histogram, digits, want, found, warp_totals);
    prefix |= found[0] << shift;
    mask |= (digits - 1) << shift;
    want -= found[1];
  }
  // T is `prefix`: the bin's blocks above T are kept and those equal to T
  // are ties.
  if (local) {
    for (uint32_t i = threadIdx.x; i < count; i += K::threads) {
      const uint32_t v = gathered_value[i], b = gathered_block[i];
      if (v >= prefix) atomicOr(&(v > prefix ? chosen : tied)[b / 32], 1u << (b % 32));
    }
  } else {
    each_tile([&](int word_base, uint32_t v, bool valid) {
      const bool inside = valid && (v >> 21) == bin;
      const uint32_t above = __ballot_sync(0xffffffff, inside && v > prefix);
      const uint32_t equal = __ballot_sync(0xffffffff, inside && v == prefix);
      if (lane == 0) {
        chosen[word_base / 32] |= above;
        tied[word_base / 32] = equal;
      }
    });
  }
  __syncthreads();
  // The first `want` ties in block order join the kept blocks, in place in
  // `chosen`; each thread then writes its words' blocks at their ascending
  // ranks.
  const int word0 = threadIdx.x * K::thread_words;
  uint32_t ties = 0;
  for (int w = word0; w < word0 + K::thread_words; ++w) ties += __popc(tied[w]);
  uint32_t tie_rank = exclusive_scan<K::threads>(ties, warp_totals);
  uint32_t kept = 0;
  for (int w = word0; w < word0 + K::thread_words; ++w) {
    uint32_t word = chosen[w];
    for (uint32_t pending = tied[w]; pending; pending &= pending - 1)
      if (tie_rank++ < want) word |= pending & (0u - pending);
    chosen[w] = word;
    kept += __popc(word);
  }
  uint32_t rank = exclusive_scan<K::threads>(kept, warp_totals);
  for (int w = word0; w < word0 + K::thread_words; ++w)
    for (uint32_t bitset = chosen[w]; bitset; bitset &= bitset - 1)
      out[rank++] = w * 32 + __ffs(bitset) - 1;
}

} // namespace chop::attention::sparse
