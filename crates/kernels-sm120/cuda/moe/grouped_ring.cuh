#pragma once

#include "grouped.cuh"

// Ring decode schedule of the grouped-MoE template. A CTA owns one (active
// expert, output tile) item over the full K, and the grid covers every
// item of the bucket's most active experts. Its warps take adjacent 16-row
// output strips and, for gate/up, `split_k` K splits of each strip. Each
// warp streams its fragments, and above one token each K64 step's routed
// rows (at most one per token and expert), through its own cp.async ring
// `depth` steps deep; at one token the CTA stages its single row once. Split
// warps leave their F32 sums in their rings and the first split's warp adds
// them in split order, so no workspace or reduction launch is used.
//
// Gate/up waits for its predecessor before it launches dependents, so the
// route metadata is complete when down starts: down starts its weight
// copies before its own wait, and only the gate/up rows follow the wait.
namespace chop::moe::grouped {

struct RingSchedule {
  int split_k;  // K splits per strip; 1 for down
  int warps;    // CTA warps: warps / split_k strips of 16 output columns
  int depth;    // ring slots per warp, in K64 steps
  // Staging choices that leave the arithmetic unchanged.
  bool early_row;    // one-token gate/up loads token 0's row before the route metadata
  bool reload_rows;  // above one token, each chunk reloads its routed row from sorted_rows
};

// A ring slot: each projection part's K64 step, then above one token the
// step's activation rows. MXFP4 parts are a lane's 16-byte code fragment
// and 4-byte scale word; FP8 parts are the strip's two 512-byte K32 tiles.
template<Weights W>
__host__ __device__ constexpr int ring_part_bytes() {
  return W == Weights::mxfp4 ? 512 + 128 : 1024;
}

template<Weights W>
__host__ __device__ constexpr int ring_slot_bytes(int parts, int tokens_capacity) {
  return parts * ring_part_bytes<W>()
       + (tokens_capacity > 1 ? tokens_capacity * kFragmentRowBytes : 0);
}

template<Weights W, Shape S, bool GateUp, int TokensCapacity, RingSchedule R>
__device__ void ring_decode(
    const uint8_t* activation, const uint8_t* weight_bytes, const uint8_t* weight_scales,
    const float* weight_f32_scales, uint16_t* output, const float* topk_weights,
    const int32_t* expert_offsets, const int32_t* sorted_rows, const int32_t* num_padded_rows,
    const int32_t* live_tokens, const int32_t* active_experts) {
  static_assert(served<W>, "instances of this weight format come later");
  constexpr bool Fp8 = W == Weights::fp8_block128;
  constexpr int N = GateUp ? 2 * S.intermediate : S.hidden;
  constexpr int K = GateUp ? S.hidden : S.intermediate;
  constexpr int SplitK = R.split_k;
  constexpr int Warps = R.warps;
  constexpr int Depth = R.depth;
  constexpr int Experts = S.experts;
  constexpr int TopK = S.top_k;
  static_assert(K % (SplitK * (Fp8 ? 128 : 64)) == 0 && N % (Fp8 ? 256 : 64) == 0);
  static_assert(TokensCapacity >= 1 && TokensCapacity <= 32);
  static_assert(Warps % SplitK == 0 && (GateUp || SplitK == 1));
  constexpr int OutN = GateUp ? N / 2 : N;
  constexpr int Ntile = Warps / SplitK * 16;
  static_assert(OutN % Ntile == 0);
  constexpr int Ntiles = OutN / Ntile;
  constexpr int Kpart = K / SplitK;
  constexpr int KpartTiles = Kpart / 64;
  constexpr int Ktiles = K / 64;
  constexpr int KpartBlocks = Fp8 ? Kpart / 128 : 1;
  constexpr int Kblocks = K / 128;
  constexpr int Parts = GateUp ? 2 : 1;
  constexpr int PartBytes = ring_part_bytes<W>();
  constexpr int Rows = TokensCapacity;
  constexpr int TokenTiles = (Rows + 7) / 8;
  constexpr int SlotBytes = ring_slot_bytes<W>(Parts, TokensCapacity);
  constexpr int MaxActive = TopK * TokensCapacity < Experts ? TopK * TokensCapacity : Experts;
  // 16-byte row chunks per slot, spread over the lanes.
  constexpr int RowCopies = (Rows * 8 + 31) / 32;
  static_assert(Depth >= 1 && Depth <= KpartTiles);
  static_assert(Depth * SlotBytes >= 32 * Parts * TokenTiles * 16);
  static_assert(!R.early_row || GateUp);
  extern __shared__ __align__(16) uint8_t ring_shared[];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int split = warp % SplitK;
  uint8_t* ring = ring_shared + warp * Depth * SlotBytes;
  if constexpr (GateUp) {
    ptx::griddep_wait();
    if (threadIdx.x == 0) ptx::griddep_launch_dependents();
  } else {
    if (threadIdx.x == 0) ptx::griddep_launch_dependents();
  }
  const int active = min(*active_experts, MaxActive);
  const int work = blockIdx.x;
  if (active <= 0 || work >= active * Ntiles) return;
  const int expert = active_experts[1 + work / Ntiles];
  if (expert < 0 || expert >= Experts) return;
  const int column = work % Ntiles * Ntile + warp / SplitK * 16;

  // The warp's code (and MXFP4 scale) sources: its strip and K split, in
  // 512-byte code tiles (16 x K64 of E2M1, 16 x K32 of E4M3).
  constexpr int Tiles = kStripStepBytes<W> / 512;
  const uint8_t* codes[Parts];
  const uint8_t* scale_words[Parts];
#pragma unroll
  for (int part = 0; part < Parts; ++part) {
    const uint64_t tile = uint64_t((expert * N + column + part * (N / 2)) / 16) * (Ktiles * Tiles)
                        + split * (KpartTiles * Tiles);
    codes[part] = weight_bytes + tile * 512 + lane * 16;
    scale_words[part] = weight_scales + tile * 32 + (lane >> 2) * 4;
  }
  auto issue_weights = [&](int kt) {
#pragma unroll
    for (int part = 0; part < Parts; ++part) {
      uint8_t* slot = ring + (kt % Depth) * SlotBytes + part * PartBytes;
      if constexpr (Fp8) {
        cp_async_16(slot + lane * 16, codes[part] + kt * 1024);
        cp_async_16(slot + 512 + lane * 16, codes[part] + kt * 1024 + 512);
      } else {
        cp_async_16(slot + lane * 16, codes[part] + kt * 512);
        cp_async_4(slot + 512 + lane * 4, scale_words[part] + kt * 32);
      }
    }
  };
#pragma unroll
  for (int kt = 0; kt < Depth; ++kt) {
    issue_weights(kt);
    cp_async_commit();
  }
  // FP8: the F32 scale of each 128-K block of the warp's K split.
  float scales[Parts][KpartBlocks];
  if constexpr (Fp8) {
#pragma unroll
    for (int part = 0; part < Parts; ++part) {
      const float* block_scales = weight_f32_scales
          + uint64_t((expert * N + column + part * (N / 2)) / 128) * Kblocks
          + split * KpartBlocks;
#pragma unroll
      for (int b = 0; b < KpartBlocks; ++b) scales[part][b] = __ldg(block_scales + b);
    }
  }

  const uint8_t* row_source[RowCopies];
  auto issue_rows = [&](int kt) {
    uint8_t* rows = ring + (kt % Depth) * SlotBytes + Parts * PartBytes;
#pragma unroll
    for (int c = 0; c < RowCopies; ++c) {
      const int chunk = lane + 32 * c;
      if (chunk < Rows * 8 && row_source[c] != nullptr)
        cp_async_16(rows + chunk / 8 * kFragmentRowBytes + chunk % 8 * 16,
                    row_source[c] + kt * 128);
    }
  };
  // At one token the CTA stages its row once, after the rings.
  constexpr bool OneRow = TokensCapacity == 1;
  uint16_t* cta_row = reinterpret_cast<uint16_t*>(ring_shared + Warps * Depth * SlotBytes);
  constexpr int RowChunks = K / 8;
  constexpr int ChunksPerThread = (RowChunks + Warps * 32 - 1) / (Warps * 32);
  uint4 row_chunks[OneRow ? ChunksPerThread : 1];
  auto load_row = [&](int row) {
#pragma unroll
    for (int i = 0; i < ChunksPerThread; ++i) {
      const int chunk = threadIdx.x + i * Warps * 32;
      if (chunk < RowChunks)
        row_chunks[i] = *reinterpret_cast<const uint4*>(
            activation + uint64_t(row) * K * 2 + chunk * 16);
    }
  };
  // One-token gate/up reads token 0, which needs no route metadata.
  if constexpr (OneRow && R.early_row) load_row(0);

  // The expert's routed rows: slot j of the tile is its j-th routed row.
  const int first = expert_offsets[expert];
  const int end = min(expert_offsets[expert + 1], *num_padded_rows);
  int routed[Rows];
#pragma unroll
  for (int j = 0; j < Rows; ++j) routed[j] = first + j < end ? sorted_rows[first + j] : -1;
  if constexpr (!GateUp) ptx::griddep_wait();
  const int routes = *live_tokens * TopK;
#pragma unroll
  for (int j = 0; j < Rows; ++j)
    if (routed[j] >= routes) routed[j] = -1;
  if (routed[0] < 0) return;
  if constexpr (OneRow) {
    if constexpr (!R.early_row) load_row(GateUp ? routed[0] / TopK : routed[0]);
#pragma unroll
    for (int i = 0; i < ChunksPerThread; ++i) {
      const int chunk = threadIdx.x + i * Warps * 32;
      if (chunk < RowChunks) reinterpret_cast<uint4*>(cta_row)[chunk] = row_chunks[i];
    }
    __syncthreads();
  } else if constexpr (R.reload_rows) {
#pragma unroll
    for (int c = 0; c < RowCopies; ++c) {
      // The chunk's routed row, reloaded so that the row array stays in
      // registers.
      const int chunk = lane + 32 * c;
      int row = chunk / 8 < Rows && first + chunk / 8 < end ? sorted_rows[first + chunk / 8] : -1;
      if (row >= routes) row = -1;
      if (GateUp && row >= 0) row /= TopK;
      row_source[c] = row >= 0
          ? activation + (uint64_t(row) * K + split * Kpart) * 2 + (chunk % 8) * 16
          : nullptr;
    }
    // One group per step, so step kt's rows complete Depth - 1 groups
    // before the newest, like its weights.
#pragma unroll
    for (int kt = 0; kt < Depth; ++kt) {
      issue_rows(kt);
      cp_async_commit();
    }
  } else {
    // The chunk's routed row, selected from the row array.
    auto issue_first_rows = [&](auto row_of) {
#pragma unroll
      for (int c = 0; c < RowCopies; ++c) {
        const int chunk = lane + 32 * c;
        const int row = row_of(chunk / 8);
        row_source[c] = row >= 0
            ? activation + (uint64_t(row) * K + split * Kpart) * 2 + (chunk % 8) * 16
            : nullptr;
      }
#pragma unroll
      for (int kt = 0; kt < Depth; ++kt) {
        issue_rows(kt);
        cp_async_commit();
      }
    };
    issue_first_rows([&](int j) {
      int row = -1;
#pragma unroll
      for (int i = 0; i < Rows; ++i)
        if (i == j && routed[i] >= 0) row = GateUp ? routed[i] / TopK : routed[i];
      return row;
    });
  }

  // Token tiles past the last routed row are skipped. MXFP4 fragments
  // accumulate directly; FP8 accumulates each 128-K block into `block` and
  // promotes it once by the block's F32 scale.
  int tiles = 0;
#pragma unroll
  for (int p = 0; p < TokenTiles; ++p)
    if (routed[8 * p] >= 0) tiles = p + 1;
  ptx::D4 accum[Parts][TokenTiles], block[Fp8 ? Parts : 1][TokenTiles];
#pragma unroll
  for (int part = 0; part < Parts; ++part)
#pragma unroll
    for (int p = 0; p < TokenTiles; ++p)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        accum[part][p].v[i] = 0.f;
        if constexpr (Fp8) block[part][p].v[i] = 0.f;
      }
#pragma unroll
  for (int kt = 0; kt < KpartTiles; ++kt) {
    cp_async_wait<Depth - 1>();
    __syncwarp();
    const uint8_t* slot = ring + (kt % Depth) * SlotBytes;
    uint4 frag[Parts][Fp8 ? 2 : 1];
    uint32_t scale_word[Parts];
#pragma unroll
    for (int part = 0; part < Parts; ++part) {
      if constexpr (Fp8) {
#pragma unroll
        for (int t = 0; t < 2; ++t)
          frag[part][t] = *reinterpret_cast<const uint4*>(slot + part * 1024 + t * 512 + lane * 16);
      } else {
        frag[part][0] = *reinterpret_cast<const uint4*>(slot + part * PartBytes + lane * 16);
        scale_word[part] =
            *reinterpret_cast<const uint32_t*>(slot + part * PartBytes + 512 + lane * 4);
      }
    }
    uint32_t b_words[TokenTiles][8];
#pragma unroll
    for (int p = 0; p < TokenTiles; ++p) {
      if (p >= tiles) break;
      const int b_row = 8 * p + lane / 4 < Rows ? 8 * p + lane / 4 : 0;
      const uint8_t* row = OneRow
          ? reinterpret_cast<const uint8_t*>(cta_row + split * Kpart + kt * 64) + (lane & 3) * 4
          : slot + Parts * PartBytes + b_row * kFragmentRowBytes + (lane & 3) * 4;
#pragma unroll
      for (int i = 0; i < 8; ++i)
        b_words[p][i] = *reinterpret_cast<const uint32_t*>(row + i * 16);
    }
#pragma unroll
    for (int step = 0; step < 4; ++step) {
      ptx::A4 a[Parts];
#pragma unroll
      for (int part = 0; part < Parts; ++part) {
        if constexpr (Fp8)
          a[part] = fp8_operand(frag[part][step >> 1], step & 1);
        else
          a[part] = mxfp4_operand(frag[part][0], scale_word[part], step);
      }
#pragma unroll
      for (int p = 0; p < TokenTiles; ++p) {
        if (p >= tiles) break;
        const ptx::B2 b{{b_words[p][2 * step], b_words[p][2 * step + 1]}};
#pragma unroll
        for (int part = 0; part < Parts; ++part) {
          if constexpr (Fp8)
            ptx::mma_bf16(block[part][p], a[part], b);
          else
            ptx::mma_bf16(accum[part][p], a[part], b);
        }
      }
    }
    if constexpr (Fp8) {
      if (kt % 2 == 1) {
#pragma unroll
        for (int part = 0; part < Parts; ++part)
#pragma unroll
          for (int p = 0; p < TokenTiles; ++p)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              accum[part][p].v[i] =
                  fmaf(block[part][p].v[i], scales[part][kt / 2], accum[part][p].v[i]);
              block[part][p].v[i] = 0.f;
            }
      }
    }
    // Every lane has read the slot before any lane refills it.
    __syncwarp();
    if (kt + Depth < KpartTiles) {
      issue_weights(kt + Depth);
      if constexpr (!OneRow) issue_rows(kt + Depth);
    }
    cp_async_commit();
  }

  if constexpr (SplitK > 1) {
    // Every copy into the ring has landed, so each warp reuses its own
    // ring for its sums.
    cp_async_wait<0>();
    float4* sums = reinterpret_cast<float4*>(ring) + lane * Parts * TokenTiles;
    if (split > 0) {
#pragma unroll
      for (int part = 0; part < Parts; ++part)
#pragma unroll
        for (int p = 0; p < TokenTiles; ++p)
          sums[part * TokenTiles + p] = make_float4(
              accum[part][p].v[0], accum[part][p].v[1], accum[part][p].v[2], accum[part][p].v[3]);
    }
    __syncthreads();
    if (split > 0) return;
#pragma unroll
    for (int later = 1; later < SplitK; ++later) {
      const float4* in = reinterpret_cast<const float4*>(
          ring_shared + (warp + later) * Depth * SlotBytes) + lane * Parts * TokenTiles;
#pragma unroll
      for (int part = 0; part < Parts; ++part)
#pragma unroll
        for (int p = 0; p < TokenTiles; ++p) {
          const float4 value = in[part * TokenTiles + p];
          accum[part][p].v[0] += value.x;
          accum[part][p].v[1] += value.y;
          accum[part][p].v[2] += value.z;
          accum[part][p].v[3] += value.w;
        }
    }
  }
  // C column j of token tile p holds routed row 8p + j: c0 and c2 are
  // column 2t, c1 and c3 column 2t + 1, rows g and g + 8.
#pragma unroll
  for (int p = 0; p < TokenTiles; ++p) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int j = 8 * p + (lane & 3) * 2 + (i & 1);
      int row = -1;
#pragma unroll
      for (int r = 0; r < Rows; ++r)
        if (r == j) row = routed[r];
      if (row < 0) continue;
      float result = accum[0][p].v[i];
      if constexpr (GateUp)
        result = result / (1.0f + expf(-result)) * accum[1][p].v[i];
      else
        result *= topk_weights[row];
      output[uint64_t(row) * OutN + column + lane / 4 + 8 * (i / 2)] =
          convert::bf16_from_f32(result);
    }
  }
}

} // namespace chop::moe::grouped
