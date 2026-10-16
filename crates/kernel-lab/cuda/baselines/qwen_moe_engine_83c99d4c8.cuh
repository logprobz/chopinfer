// Lab-only reference: the Qwen3.8 grouped-MoE expert engine as served at
// main 83c99d4c8 (kernels-sm120/cuda/moe/qwen.cuh), before the grouped-MoE
// template replaced it. Kernel-lab builds it as the chop_moe_eng_qwen38_*
// cubins and compares the served template entries with it bit for bit
// (tests/moe_template.rs). Keep it unchanged: it is the evidence that the
// template reproduces the engine, not code to maintain.
//
// Grouped FP8 decode and prefill over this rank's experts: FP8 E4M3 weights
// in the fragment-major layout of `kernel_core::transform::FragmentWeightLayout`
// (16 rows by K32 per 512-byte tile), one F32 scale per 128 x 128 block. The
// MMAs add exact BF16 products of codes and activations in F32, and the
// scales apply to those sums per block.
#pragma once

#include "moe/qwen.cuh"

namespace chop::moe::qwen {

// Steps of the prefill item search over the local experts: log2(kExperts).
constexpr int kExpertSearchSteps =
    kExperts == 256 ? 8 : kExperts == 128 ? 7 : kExperts == 64 ? 6 : 5;

// Two E4M3 bytes per packed conversion; one 16-byte fragment covers two
// k16 steps (`Half` selects the step).
__device__ __forceinline__ ptx::A4 fp8_step(uint4 fragment, int half) {
  const uint32_t low = half ? fragment.z : fragment.x;
  const uint32_t high = half ? fragment.w : fragment.y;
  ptx::A4 a;
  a.v[0] = convert::unpack_bf16x2_e4m3(uint16_t(low));
  a.v[1] = convert::unpack_bf16x2_e4m3(uint16_t(low >> 16));
  a.v[2] = convert::unpack_bf16x2_e4m3(uint16_t(high));
  a.v[3] = convert::unpack_bf16x2_e4m3(uint16_t(high >> 16));
  return a;
}

// A decode ring slot: one K64 step per projection part (two 512-byte K32
// fragment tiles) and, above one token, the step's activation rows.
__host__ __device__ constexpr int decode_slot_bytes(int parts, int tokens_capacity) {
  return parts * 1024 + (tokens_capacity > 1 ? tokens_capacity * kFragmentRowBytes : 0);
}

// Grouped FP8 decode, the MiMo ring kernel's schedule (fragments.cuh): a CTA
// owns one (expert, output tile) item over the full K. Its warps take
// adjacent 16-row strips and, for gate/up, K splits of each strip. Each warp
// streams its fragments, and above one token the step's routed rows, through
// its own cp.async ring `Depth` K64 steps deep. Every K split covers whole
// 128-K scale blocks: a block accumulates into a temporary fragment and is
// promoted once by its F32 scale. The first split's warp adds the other
// splits' sums in split order, so no workspace or reduction launch is used.
template<bool GateUp, int N, int K, int SplitK, int TokensCapacity, int Warps,
         int Depth>
__device__ void run_decode(
    const uint8_t* activation, const uint8_t* weight_bytes,
    const float* weight_scales, uint16_t* output, const float* topk_weights,
    const int32_t* expert_offsets, const int32_t* sorted_rows,
    const int32_t* num_padded_rows, const int32_t* live_tokens,
    const int32_t* active_experts) {
  static_assert(K % (SplitK * 128) == 0 && N % 256 == 0);
  static_assert(TokensCapacity >= 1 && TokensCapacity <= 32);
  static_assert(Warps % SplitK == 0);
  constexpr int OutN = GateUp ? N / 2 : N;
  constexpr int Ntile = Warps / SplitK * 16;
  static_assert(OutN % Ntile == 0);
  constexpr int Ntiles = OutN / Ntile;
  constexpr int Kpart = K / SplitK;
  constexpr int KpartTiles = Kpart / 64;
  constexpr int KpartBlocks = Kpart / 128;
  constexpr int Kblocks = K / 128;
  constexpr int Parts = GateUp ? 2 : 1;
  constexpr int Rows = TokensCapacity;
  constexpr int TokenTiles = (Rows + 7) / 8;
  constexpr int SlotBytes = decode_slot_bytes(Parts, TokensCapacity);
  constexpr int MaxActive = kTopK * TokensCapacity < kExperts ? kTopK * TokensCapacity : kExperts;
  constexpr int RowCopies = (Rows * 8 + 31) / 32;
  static_assert(Depth >= 1 && Depth <= KpartTiles);
  static_assert(Depth * SlotBytes >= 32 * Parts * TokenTiles * 16);
  extern __shared__ __align__(16) uint8_t decode_shared[];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int split = warp % SplitK;
  uint8_t* ring = decode_shared + warp * Depth * SlotBytes;
  // Gate/up waits before it launches dependents, so down starts after the
  // route metadata is complete and may stream weights before its own wait.
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
  if (expert < 0 || expert >= kExperts) return;
  const int column = work % Ntiles * Ntile + warp / SplitK * 16;

  const uint8_t* codes[Parts];
  float scales[Parts][KpartBlocks];
#pragma unroll
  for (int part = 0; part < Parts; ++part) {
    const int row = expert * N + column + part * (N / 2);
    codes[part] = weight_bytes
        + (uint64_t(row / 16) * (K / 32) + split * (Kpart / 32)) * 512 + lane * 16;
  }
  auto issue_weights = [&](int kt) {
#pragma unroll
    for (int part = 0; part < Parts; ++part) {
      uint8_t* slot = ring + (kt % Depth) * SlotBytes + part * 1024;
      cp_async_16(slot + lane * 16, codes[part] + kt * 1024);
      cp_async_16(slot + 512 + lane * 16, codes[part] + kt * 1024 + 512);
    }
  };
#pragma unroll
  for (int kt = 0; kt < Depth; ++kt) {
    issue_weights(kt);
    cp_async_commit();
  }
#pragma unroll
  for (int part = 0; part < Parts; ++part) {
    const float* block_scales = weight_scales
        + uint64_t((expert * N + column + part * (N / 2)) / 128) * Kblocks
        + split * KpartBlocks;
#pragma unroll
    for (int b = 0; b < KpartBlocks; ++b) scales[part][b] = __ldg(block_scales + b);
  }

  const uint8_t* row_source[RowCopies];
  auto issue_rows = [&](int kt) {
    uint8_t* rows = ring + (kt % Depth) * SlotBytes + Parts * 1024;
#pragma unroll
    for (int c = 0; c < RowCopies; ++c) {
      const int chunk = lane + 32 * c;
      if (chunk < Rows * 8 && row_source[c] != nullptr)
        cp_async_16(rows + chunk / 8 * kFragmentRowBytes + chunk % 8 * 16,
                    row_source[c] + kt * 128);
    }
  };
  constexpr bool OneRow = TokensCapacity == 1;
  uint16_t* cta_row = reinterpret_cast<uint16_t*>(decode_shared + Warps * Depth * SlotBytes);
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

  const int first = expert_offsets[expert];
  const int end = min(expert_offsets[expert + 1], *num_padded_rows);
  int routed[Rows];
#pragma unroll
  for (int j = 0; j < Rows; ++j) routed[j] = first + j < end ? sorted_rows[first + j] : -1;
  if constexpr (!GateUp) ptx::griddep_wait();
  const int routes = *live_tokens * kTopK;
#pragma unroll
  for (int j = 0; j < Rows; ++j)
    if (routed[j] >= routes) routed[j] = -1;
  if (routed[0] < 0) return;
  if constexpr (OneRow) {
    load_row(GateUp ? routed[0] / kTopK : routed[0]);
#pragma unroll
    for (int i = 0; i < ChunksPerThread; ++i) {
      const int chunk = threadIdx.x + i * Warps * 32;
      if (chunk < RowChunks) reinterpret_cast<uint4*>(cta_row)[chunk] = row_chunks[i];
    }
    __syncthreads();
  } else {
#pragma unroll
    for (int c = 0; c < RowCopies; ++c) {
      // The chunk's routed row, reloaded so that the row array stays in
      // registers.
      const int chunk = lane + 32 * c;
      int row = chunk / 8 < Rows && first + chunk / 8 < end ? sorted_rows[first + chunk / 8] : -1;
      if (row >= routes) row = -1;
      if (GateUp && row >= 0) row /= kTopK;
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
  }

  int tiles = 0;
#pragma unroll
  for (int p = 0; p < TokenTiles; ++p)
    if (routed[8 * p] >= 0) tiles = p + 1;
  ptx::D4 accum[Parts][TokenTiles], block[Parts][TokenTiles];
#pragma unroll
  for (int part = 0; part < Parts; ++part)
#pragma unroll
    for (int p = 0; p < TokenTiles; ++p)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        accum[part][p].v[i] = 0.f;
        block[part][p].v[i] = 0.f;
      }
#pragma unroll
  for (int kt = 0; kt < KpartTiles; ++kt) {
    cp_async_wait<Depth - 1>();
    __syncwarp();
    const uint8_t* slot = ring + (kt % Depth) * SlotBytes;
    uint4 frag[Parts][2];
#pragma unroll
    for (int part = 0; part < Parts; ++part)
#pragma unroll
      for (int t = 0; t < 2; ++t)
        frag[part][t] = *reinterpret_cast<const uint4*>(slot + part * 1024 + t * 512 + lane * 16);
    uint32_t b_words[TokenTiles][8];
#pragma unroll
    for (int p = 0; p < TokenTiles; ++p) {
      if (p >= tiles) break;
      const int b_row = 8 * p + lane / 4 < Rows ? 8 * p + lane / 4 : 0;
      const uint8_t* row = OneRow
          ? reinterpret_cast<const uint8_t*>(cta_row + split * Kpart + kt * 64) + (lane & 3) * 4
          : slot + Parts * 1024 + b_row * kFragmentRowBytes + (lane & 3) * 4;
#pragma unroll
      for (int i = 0; i < 8; ++i)
        b_words[p][i] = *reinterpret_cast<const uint32_t*>(row + i * 16);
    }
#pragma unroll
    for (int step = 0; step < 4; ++step) {
      ptx::A4 a[Parts];
#pragma unroll
      for (int part = 0; part < Parts; ++part) a[part] = fp8_step(frag[part][step >> 1], step & 1);
#pragma unroll
      for (int p = 0; p < TokenTiles; ++p) {
        if (p >= tiles) break;
        const ptx::B2 b{{b_words[p][2 * step], b_words[p][2 * step + 1]}};
#pragma unroll
        for (int part = 0; part < Parts; ++part) ptx::mma_bf16(block[part][p], a[part], b);
      }
    }
    if (kt % 2 == 1) {
#pragma unroll
      for (int part = 0; part < Parts; ++part)
#pragma unroll
        for (int p = 0; p < TokenTiles; ++p)
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            accum[part][p].v[i] = fmaf(block[part][p].v[i], scales[part][kt / 2], accum[part][p].v[i]);
            block[part][p].v[i] = 0.f;
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
    cp_async_wait<0>();
    float4* sums = reinterpret_cast<float4*>(ring) + lane * Parts * TokenTiles;
    if (split > 0) {
#pragma unroll
      for (int part = 0; part < Parts; ++part)
#pragma unroll
        for (int p = 0; p < TokenTiles; ++p)
          sums[part * TokenTiles + p] = make_float4(
              accum[part][p].v[0], accum[part][p].v[1],
              accum[part][p].v[2], accum[part][p].v[3]);
    }
    __syncthreads();
    if (split > 0) return;
#pragma unroll
    for (int later = 1; later < SplitK; ++later) {
      const float4* in = reinterpret_cast<const float4*>(
          decode_shared + (warp + later) * Depth * SlotBytes) + lane * Parts * TokenTiles;
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
  // C column j of token tile p holds routed row 8p + j.
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

// Grouped FP8 prefill, the MiMo persistent schedule (prefill.cuh) with FP8
// weight stages: a K64 stage holds each strip's two 512-byte K32 fragment
// tiles and the item's two block scales (gate and up, or the two 128-row
// halves of a down item). A math warp's 32 x 128 tile fills its registers,
// so instead of promoting a temporary per block it keeps the tile in units
// of the current block's scale: at each 128-K block it multiplies the tile
// by the previous scale over the new one, and the epilogue multiplies by the
// last scale. The MMAs still add exact code-by-activation products, and a
// block whose scale is zero adds nothing.
namespace prefill = chop::moe::prefill;

constexpr int kPrefillWeightBytes = prefill::kWeightTiles * 1024;
constexpr int kPrefillWeightSlotBytes = kPrefillWeightBytes + 16;
constexpr int kPrefillTableBytes = (4 * kExperts + 4) * 4;

__host__ __device__ constexpr int prefill_shared_bytes(int weight_stages,
                                                       int activation_stages) {
  return weight_stages * (kPrefillWeightSlotBytes + 16)
       + activation_stages * (prefill::kActivationBytes + 16) + kPrefillTableBytes
       + prefill::kRouteSlots * prefill::kRouteSlotBytes + prefill::kStagingBytes;
}

struct Tables {
  int* first;
  int* live;
  int* full_start;
  int* partial;
  int* partial_count;
};

template<int FeatureTiles>
__device__ __forceinline__ prefill::Tile decode_item(const Tables& tables, uint64_t work) {
  constexpr int kTile = prefill::kTokenTile;
  if (work < uint64_t(tables.full_start[kExperts])) {
    int low = 0, high = kExperts - 1;
#pragma unroll
    for (int step = 0; step < kExpertSearchSteps; ++step) {
      const int middle = (low + high + 1) / 2;
      if (uint64_t(tables.full_start[middle]) <= work) low = middle;
      else high = middle - 1;
    }
    const int local = int(work) - tables.full_start[low];
    const int token_tiles = tables.live[low] / kTile;
    return {low, local / token_tiles, tables.first[low] + local % token_tiles * kTile, kTile};
  }
  const int local = int(work) - tables.full_start[kExperts];
  const int expert = tables.partial[local / FeatureTiles];
  const int live = tables.live[expert];
  return {expert, local % FeatureTiles, tables.first[expert] + live / kTile * kTile, live % kTile};
}

// Gate/up strips 0..7 are gate rows and 8..15 the matching up rows of 128
// output columns; down strips are 256 adjacent rows.
template<bool GateUp, int N>
__device__ __forceinline__ int fp8_strip_of(const prefill::Tile& tile, int j) {
  if constexpr (GateUp)
    return (tile.expert * N + (j >= 8 ? N / 2 : 0) + tile.feature_tile * 128 + (j & 7) * 16) / 16;
  else
    return (tile.expert * N + tile.feature_tile * prefill::kWeightRows + j * 16) / 16;
}

// Warp 8 streams an item's weight fragments and its two block scales.
template<bool GateUp, int N, int K, int Stages>
__device__ __forceinline__ void produce_fp8_weights(
    const prefill::Tile& tile, const prefill::Ring& ring, uint32_t& step,
    const uint8_t* weight_bytes, const float* weight_scales) {
  constexpr int Ktiles = K / 64;
  const int lane = threadIdx.x & 31;
  const uint8_t* strips[2];
#pragma unroll
  for (int half = 0; half < 2; ++half)
    strips[half] = weight_bytes
        + uint64_t(fp8_strip_of<GateUp, N>(tile, 8 * half)) * (K / 32) * 512 + lane * 16;
  const float* scale = weight_scales
      + uint64_t(fp8_strip_of<GateUp, N>(tile, 8 * (lane & 1)) / 8) * (K / 128);
#pragma unroll 1
  for (int kt = 0; kt < Ktiles; ++kt, ++step) {
    const int slot = step % Stages;
    ptx::mbarrier_wait(&ring.empty[slot], ((step / Stages) & 1) ^ 1);
    uint8_t* stage = ring.slots + slot * kPrefillWeightSlotBytes;
    if constexpr (GateUp && kExperts == 128) {
      if (lane == 0) ptx::mbarrier_expect_tx(&ring.full[slot], kPrefillWeightBytes);
      __syncwarp();
      if (lane < prefill::kWeightTiles)
        ptx::cp_async_bulk(stage + lane * 1024,
            weight_bytes + (uint64_t(fp8_strip_of<true, N>(tile, lane)) * (K / 32) + 2 * kt) * 512,
            1024, &ring.full[slot]);
    } else {
#pragma unroll
      for (int j = 0; j < prefill::kWeightTiles; ++j)
#pragma unroll
        for (int t = 0; t < 2; ++t)
          cp_async_16(stage + j * 1024 + t * 512 + lane * 16,
                      strips[j >> 3] + (uint64_t(j & 7) * (K / 32) + 2 * kt + t) * 512);
    }
    if (lane < 2) cp_async_4(stage + kPrefillWeightBytes + lane * 4, scale + kt / 2);
    prefill::cp_async_arrive(&ring.full[slot]);
  }
}

// Warps 9-11 fill one item's route slot, then stream its routed rows.
template<bool GateUp, int K, int Stages>
__device__ __forceinline__ void produce_fp8_rows(
    const prefill::Tile& tile, prefill::RouteSlot& route, const prefill::Ring& ring,
    uint32_t& step, const uint8_t* activation, const float* topk_weights,
    const int32_t* sorted_rows, int padded, int routes) {
  constexpr int Ktiles = K / 64;
  const int t = threadIdx.x - (prefill::kMathWarps + 1) * 32;
  for (int row = t; row < prefill::kTokenTile; row += prefill::kRowThreads) {
    const int slot = tile.first_row + row;
    int routed = row < tile.rows && slot < padded ? sorted_rows[slot] : -1;
    if (routed >= routes) routed = -1;
    route.rows[row] = routed;
    if (!GateUp && routed >= 0) route.weights[row] = topk_weights[routed];
  }
  prefill::row_producer_sync();
#pragma unroll 1
  for (int kt = 0; kt < Ktiles; ++kt, ++step) {
    const int slot = step % Stages;
    ptx::mbarrier_wait(&ring.empty[slot], ((step / Stages) & 1) ^ 1);
    uint8_t* rows = ring.slots + slot * prefill::kActivationBytes;
    for (int index = t; index < (GateUp && kExperts == 128 ? tile.rows : prefill::kTokenTile) * 8; index += prefill::kRowThreads) {
      const int row = index >> 3;
      const int chunk = index & 7;
      const int routed = route.rows[row];
      if (routed >= 0)
        prefill::cp_async16(rows + prefill::row_chunk(row, chunk),
                            activation + (uint64_t(GateUp ? routed / kTopK : routed) * K
                                          + kt * 64 + chunk * 8) * 2);
    }
    prefill::cp_async_arrive(&ring.full[slot]);
    if (t == 0) ptx::mbarrier_arrive(&ring.full[slot]);
  }
}

template<bool GateUp, int N, int K, int WeightStages, int ActivationStages>
__device__ void run_prefill(
    const uint8_t* activation, const uint8_t* weight_bytes,
    const float* weight_scales, uint16_t* output, const float* topk_weights,
    const int32_t* expert_offsets, const int32_t* sorted_rows,
    const int32_t* num_padded_rows, const int32_t* live_tokens) {
  using prefill::kMathWarps;
  using prefill::kTokenTile;
  static_assert(K % 128 == 0);
  static_assert(GateUp ? N % 256 == 0 : N % prefill::kWeightRows == 0);
  static_assert(WeightStages >= 2 && ActivationStages >= 2);
  constexpr int OutN = GateUp ? N / 2 : N;
  constexpr int FeatureTiles = GateUp ? OutN / 128 : N / prefill::kWeightRows;
  constexpr int Ktiles = K / 64;
  constexpr int WarpColumns = GateUp ? 16 : 32;
  constexpr int Chunks = WarpColumns / 8;
  extern __shared__ __align__(128) uint8_t shared[];
  uint8_t* activation_slots = shared + WeightStages * kPrefillWeightSlotBytes;
  uint64_t* barriers = reinterpret_cast<uint64_t*>(
      activation_slots + ActivationStages * prefill::kActivationBytes);
  const prefill::Ring weight_ring{shared, barriers, barriers + WeightStages};
  const prefill::Ring row_ring{activation_slots, barriers + 2 * WeightStages,
                               barriers + 2 * WeightStages + ActivationStages};
  uint8_t* tables_base = reinterpret_cast<uint8_t*>(
      barriers + 2 * (WeightStages + ActivationStages));
  int* table = reinterpret_cast<int*>(tables_base);
  const Tables tables{table, table + kExperts, table + 2 * kExperts,
                      table + 3 * kExperts + 1, table + 4 * kExperts + 1};
  prefill::RouteSlot* route_slots =
      reinterpret_cast<prefill::RouteSlot*>(tables_base + kPrefillTableBytes);
  uint8_t* staging_base =
      tables_base + kPrefillTableBytes + prefill::kRouteSlots * prefill::kRouteSlotBytes;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (threadIdx.x == 0) ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const int tokens = *live_tokens;
  const int padded = *num_padded_rows;
  if (tokens <= 0 || padded <= 0) return;
  const int routes = tokens * kTopK;

  {
    __shared__ int warp_totals[kExperts / 32];
    int* keys = reinterpret_cast<int*>(staging_base);
    const int expert = threadIdx.x;
    int units = 0, through = 0;
    bool partial = false;
    if (expert < kExperts) {
      const int first = min(expert_offsets[expert], padded);
      const int rows = max(min(expert_offsets[expert + 1], padded) - first, 0);
      // Padding trails the routed rows; find the last routed row from the end.
      int live = 0;
#pragma unroll 1
      for (int block = rows - 16; live == 0 && block > -16; block -= 16) {
        int32_t routed[16];
#pragma unroll
        for (int i = 0; i < 16; ++i)
          routed[i] = block + i >= 0 ? sorted_rows[first + block + i] : -1;
#pragma unroll
        for (int i = 0; i < 16; ++i)
          if (routed[i] >= 0 && routed[i] < routes) live = block + i + 1;
      }
      units = live / kTokenTile * FeatureTiles;
      const int rest = live % kTokenTile;
      partial = rest > 0;
      keys[expert] = partial ? (8 - (rest + 15) / 16) * kExperts + expert : INT32_MAX;
      through = units;
#pragma unroll
      for (int delta = 1; delta < 32; delta *= 2) {
        const int previous = __shfl_up_sync(0xffffffff, through, delta);
        if (lane >= delta) through += previous;
      }
      if (lane == 31) warp_totals[warp] = through;
      tables.first[expert] = first;
      tables.live[expert] = live;
    }
    if (threadIdx.x < WeightStages) {
      ptx::mbarrier_init(&weight_ring.full[threadIdx.x], 32);
      ptx::mbarrier_init(&weight_ring.empty[threadIdx.x], kMathWarps);
    }
    if (threadIdx.x < ActivationStages) {
      ptx::mbarrier_init(&row_ring.full[threadIdx.x], prefill::kRowThreads + 1);
      ptx::mbarrier_init(&row_ring.empty[threadIdx.x], kMathWarps);
    }
    ptx::fence_mbarrier_init();
    const int partials = __syncthreads_count(partial);
    if (expert < kExperts) {
#pragma unroll
      for (int w = 0; w < kExperts / 32; ++w)
        if (w < warp) through += warp_totals[w];
      tables.full_start[expert] = through - units;
      if (expert == kExperts - 1) tables.full_start[kExperts] = through;
      if (partial) {
        int rank = 0;
#pragma unroll 8
        for (int other = 0; other < kExperts; ++other) rank += keys[other] < keys[expert];
        tables.partial[rank] = expert;
      }
      if (expert == 0) *tables.partial_count = partials;
    }
    __syncthreads();
  }
  const uint64_t total = uint64_t(tables.full_start[kExperts])
                       + uint64_t(*tables.partial_count) * FeatureTiles;
  if (blockIdx.x >= total) return;

  if (warp >= kMathWarps) {
    ptx::setmaxnreg_dec<40>();
    uint32_t step = 0;
    int item = 0;
#pragma unroll 1
    for (int round = 0; uint64_t(round) * gridDim.x < total; ++round) {
      const uint64_t work = prefill::snake(round);
      if (work >= total) continue;
      const prefill::Tile tile = decode_item<FeatureTiles>(tables, work);
      if (warp == kMathWarps)
        produce_fp8_weights<GateUp, N, K, WeightStages>(
            tile, weight_ring, step, weight_bytes, weight_scales);
      else
        produce_fp8_rows<GateUp, K, ActivationStages>(
            tile, route_slots[item % prefill::kRouteSlots], row_ring, step, activation,
            topk_weights, sorted_rows, padded, routes);
      ++item;
    }
    return;
  }

  ptx::setmaxnreg_inc<232>();
  uint8_t* staging = staging_base + warp * 1024;
  ptx::D4 accum[2][16];
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int n = 0; n < 16; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) accum[m][n].v[i] = 0.f;
  // The tile is in output units: each K32 half step's products accumulate in
  // a temporary per token group and are promoted by their block's F32 scale,
  // as decode promotes each block, so no scale ratio rescales the sums.
  float scale[2] = {0.f, 0.f};

  uint32_t step = 0;
  int item = 0;
#pragma unroll 1
  for (int round = 0; uint64_t(round) * gridDim.x < total; ++round) {
    const uint64_t work = prefill::snake(round);
    if (work >= total) continue;
    const prefill::Tile tile = decode_item<FeatureTiles>(tables, work);
    const prefill::RouteSlot& route = route_slots[item++ % prefill::kRouteSlots];
    const int groups = min(8, (tile.rows + 15) / 16);
#pragma unroll 1
    for (int kt = 0; kt < Ktiles; ++kt, ++step) {
      const int weight_slot = step % WeightStages;
      const int row_slot = step % ActivationStages;
      ptx::mbarrier_wait(&weight_ring.full[weight_slot], (step / WeightStages) & 1);
      ptx::mbarrier_wait(&row_ring.full[row_slot], (step / ActivationStages) & 1);
      const uint8_t* stage = weight_ring.slots + weight_slot * kPrefillWeightSlotBytes;
      const uint8_t* rows = row_ring.slots + row_slot * prefill::kActivationBytes;
      if (kt % 2 == 0)
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          const int j = GateUp ? warp + 8 * m : 2 * warp + m;
          scale[m] = *reinterpret_cast<const float*>(stage + kPrefillWeightBytes + (j >> 3) * 4);
        }
      // One K32 half at a time; a block whose scale is zero is skipped.
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        ptx::A4 a[2][2];
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          const int j = GateUp ? warp + 8 * m : 2 * warp + m;
          const uint4 fragment =
              *reinterpret_cast<const uint4*>(stage + j * 1024 + half * 512 + lane * 16);
          a[m][0] = fp8_step(fragment, 0);
          a[m][1] = fp8_step(fragment, 1);
        }
        auto group = [&](int p) {
          ptx::D4 part[2][2] = {};
#pragma unroll
          for (int k = 0; k < 2; ++k) {
            const int row = 16 * p + (lane & 7) + 8 * (lane >> 4);
            const int chunk = 4 * half + 2 * k + ((lane >> 3) & 1);
            uint32_t b[4];
            ptx::ldmatrix_b16<4>(b, rows + prefill::row_chunk(row, chunk));
            const ptx::B2 low_rows{{b[0], b[1]}};
            const ptx::B2 high_rows{{b[2], b[3]}};
#pragma unroll
            for (int m = 0; m < 2; ++m) {
              if (scale[m] == 0.f) continue;
              ptx::mma_bf16(part[m][0], a[m][k], low_rows);
              ptx::mma_bf16(part[m][1], a[m][k], high_rows);
            }
          }
#pragma unroll
          for (int m = 0; m < 2; ++m) {
            if (scale[m] == 0.f) continue;
#pragma unroll
            for (int q = 0; q < 2; ++q)
#pragma unroll
              for (int i = 0; i < 4; ++i)
                accum[m][2 * p + q].v[i] =
                    fmaf(part[m][q].v[i], scale[m], accum[m][2 * p + q].v[i]);
          }
        };
        if (groups == 8) {
#pragma unroll
          for (int p = 0; p < 8; ++p) group(p);
        } else {
#pragma unroll
          for (int p = 0; p < 8; ++p) {
            if (p >= groups) break;
            group(p);
          }
        }
      }
      __syncwarp();
      if (lane == 0) {
        ptx::mbarrier_arrive(&weight_ring.empty[weight_slot]);
        ptx::mbarrier_arrive(&row_ring.empty[row_slot]);
      }
    }

    const int column_base = GateUp
        ? tile.feature_tile * 128 + warp * WarpColumns
        : tile.feature_tile * prefill::kWeightRows + warp * WarpColumns;
    auto staged = [&](int token, int chunk) {
      const int swizzled = GateUp ? chunk ^ ((token >> 2) & 1) : chunk ^ ((token >> 1) & 3);
      return staging + token * WarpColumns * 2 + swizzled * 16;
    };
#pragma unroll
    for (int p = 0; p < 8; ++p) {
      if (p >= groups) break;
#pragma unroll
      for (int m = 0; m < (GateUp ? 1 : 2); ++m) {
        uint32_t packed[4];
#pragma unroll
        for (int q = 0; q < 2; ++q) {
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            float v0 = accum[m][2 * p + q].v[2 * h];
            float v1 = accum[m][2 * p + q].v[2 * h + 1];
            if constexpr (GateUp) {
              v0 = v0 / (1.0f + expf(-v0)) * accum[1][2 * p + q].v[2 * h];
              v1 = v1 / (1.0f + expf(-v1)) * accum[1][2 * p + q].v[2 * h + 1];
            } else {
              const int token = 16 * p + 8 * q + 2 * (lane & 3);
              v0 *= route.weights[token];
              v1 *= route.weights[token + 1];
            }
            packed[2 * q + h] = prefill::pack_bf16(v0, v1);
          }
        }
        const int matrix = lane >> 3;
        ptx::stmatrix_trans_b16<4>(
            staged((matrix >> 1) * 8 + (lane & 7), 2 * m + (matrix & 1)), packed);
      }
      __syncwarp();
#pragma unroll
      for (int it = 0; it < 16 * Chunks / 32; ++it) {
        const int index = it * 32 + lane;
        const int token = index / Chunks;
        const int chunk = index % Chunks;
        const uint4 value = *reinterpret_cast<const uint4*>(staged(token, chunk));
        const int routed = route.rows[16 * p + token];
        if (routed >= 0)
          *reinterpret_cast<uint4*>(output + uint64_t(routed) * OutN + column_base + chunk * 8) =
              value;
      }
      __syncwarp();
    }
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
      for (int n = 0; n < 16; ++n)
#pragma unroll
        for (int i = 0; i < 4; ++i) accum[m][n].v[i] = 0.f;
  }
}

} // namespace chop::moe::qwen
