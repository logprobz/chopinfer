// Lab-only reference: the MiMo V2.6 grouped-MoE expert engines as served at
// main 83c99d4c8 (kernels-sm120/cuda/moe/mimo26_fragments.cuh and
// prefill_staging.cuh), before the grouped-MoE template replaced them; its
// stream-K decode engine left with the stream-K schedule. Kernel-lab builds
// them as the chop_moe_eng_mimo_* cubins and compares the served template
// entries with them bit for bit (tests/moe_template.rs). Keep them unchanged: they are the evidence that
// the template reproduces the engines, not code to maintain. The shared
// helpers they call stay in the served headers included below.
#pragma once

#include "moe/prefill_staging.cuh"

namespace chop::moe {

// A ring slot holds one K64 tile per projection part (a lane's 16-byte code
// fragment and 4-byte scale word each) and, above one token, the tile's
// activation rows: at most one routed row per token and expert.
__host__ __device__ constexpr int fragment_slot_bytes(int parts, int tokens_capacity) {
  return parts * (512 + 128) + (tokens_capacity > 1 ? tokens_capacity * kFragmentRowBytes : 0);
}

// Ring depth: the rings of all warps, and at one token the 8 KB row, share
// 49 KB, so two CTAs and their reserved shared memory fit in an SM.
__host__ __device__ constexpr int fragment_ring_tiles(
    int k_part_tiles, int warps, int parts, int tokens_capacity) {
  const int fit = (50176 - (tokens_capacity == 1 ? 8192 : 0))
      / (warps * fragment_slot_bytes(parts, tokens_capacity));
  return k_part_tiles < fit ? k_part_tiles : fit;
}

// MiMo grouped decode. Each expert has at most one routed row per token,
// so its rows form up to four n8 token tiles. A CTA owns one (expert,
// output tile) item over the full K, and the grid covers every item. Its warps take adjacent 16-row output strips and, for gate/up, K
// splits of each strip. As soon as the expert is known, each warp streams
// its fragments through its own cp.async ring several tiles ahead of the
// MMAs, and each tile's activation rows travel in the same ring slot. A
// lane reads back its own fragment copies; the activation rows are read
// across the warp after a warp barrier. Missing rows repeat row 0 and only
// feed MMA columns the epilogue skips. Split warps leave their F32 sums in
// their rings and the first split's warp adds them in split order, so no
// partial workspace or reduction launch is needed.
//
// Gate/up waits for its predecessor before it launches dependents, so the
// route metadata is complete when down starts: down starts its weight
// copies before its own wait, and only the gate/up rows follow the wait.
template<bool GateUp, int Experts, int N, int K, int SplitK,
         int TokensCapacity, int Warps>
__device__ void run_fragment_decode(
    const uint8_t* activation, const uint8_t* weight_bytes,
    const uint8_t* weight_scales, uint16_t* output,
    const float* topk_weights, const int32_t* expert_offsets,
    const int32_t* sorted_rows, const int32_t* num_padded_rows,
    const int32_t* live_tokens, const int32_t* active_experts) {
  static_assert(K % (SplitK * 64) == 0 && N % 64 == 0);
  static_assert(TokensCapacity >= 1 && TokensCapacity <= 32);
  static_assert(Warps % SplitK == 0);
  constexpr int OutN = GateUp ? N / 2 : N;
  constexpr int Ntile = Warps / SplitK * 16;
  constexpr int Ntiles = OutN / Ntile;
  constexpr int Kpart = K / SplitK;
  constexpr int KpartTiles = Kpart / 64;
  constexpr int Ktiles = K / 64;
  constexpr int Parts = GateUp ? 2 : 1;
  constexpr int Rows = TokensCapacity;
  constexpr int TokenTiles = (Rows + 7) / 8;
  constexpr int SlotBytes = fragment_slot_bytes(Parts, TokensCapacity);
  constexpr int Depth = fragment_ring_tiles(KpartTiles, Warps, Parts, TokensCapacity);
  constexpr int MaxActive = 8 * TokensCapacity < Experts ? 8 * TokensCapacity : Experts;
  // 16-byte row chunks per slot, spread over the lanes.
  constexpr int RowCopies = (Rows * 8 + 31) / 32;
  static_assert(Depth >= 1 && Depth * SlotBytes >= 32 * Parts * TokenTiles * 16);
  extern __shared__ __align__(16) uint8_t fragment_shared[];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int split = warp % SplitK;
  uint8_t* ring = fragment_shared + warp * Depth * SlotBytes;
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

  const uint8_t* codes[Parts];
  const uint8_t* scales[Parts];
#pragma unroll
  for (int part = 0; part < Parts; ++part) {
    const uint64_t tile = uint64_t((expert * N + column + part * (N / 2)) / 16)
        * Ktiles + split * KpartTiles;
    codes[part] = weight_bytes + tile * 512 + lane * 16;
    scales[part] = weight_scales + tile * 32 + (lane >> 2) * 4;
  }
  auto issue_weights = [&](int kt) {
#pragma unroll
    for (int part = 0; part < Parts; ++part) {
      uint8_t* slot = ring + (kt % Depth) * SlotBytes + part * 640;
      cp_async_16(slot + lane * 16, codes[part] + kt * 512);
      cp_async_4(slot + 512 + lane * 4, scales[part] + kt * 32);
    }
  };
#pragma unroll
  for (int kt = 0; kt < Depth; ++kt) {
    issue_weights(kt);
    cp_async_commit();
  }

  const uint8_t* row_source[RowCopies];
  auto issue_rows = [&](int kt) {
    uint8_t* rows = ring + (kt % Depth) * SlotBytes + Parts * 640;
#pragma unroll
    for (int c = 0; c < RowCopies; ++c) {
      const int chunk = lane + 32 * c;
      if (chunk < Rows * 8 && row_source[c] != nullptr)
        cp_async_16(rows + chunk / 8 * kFragmentRowBytes + chunk % 8 * 16,
                    row_source[c] + kt * 128);
    }
  };
  auto issue_first_rows = [&](auto row_of) {
#pragma unroll
    for (int c = 0; c < RowCopies; ++c) {
      const int chunk = lane + 32 * c;
      const int row = row_of(chunk / 8);
      row_source[c] = row >= 0
          ? activation + (uint64_t(row) * K + split * Kpart) * 2 + (chunk % 8) * 16
          : nullptr;
    }
    // One group per tile, so tile kt's rows complete Depth - 1 groups
    // before the newest, like its weights.
#pragma unroll
    for (int kt = 0; kt < Depth; ++kt) {
      issue_rows(kt);
      cp_async_commit();
    }
  };
  // At one token the CTA stages its row once, after the rings; gate/up
  // reads token 0, which needs no route metadata.
  constexpr bool OneRow = TokensCapacity == 1;
  uint16_t* cta_row = reinterpret_cast<uint16_t*>(fragment_shared + Warps * Depth * SlotBytes);
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
  if constexpr (OneRow && GateUp) load_row(0);

  // The expert's routed rows: slot j of the tile is its j-th routed row.
  const int first = expert_offsets[expert];
  const int end = min(expert_offsets[expert + 1], *num_padded_rows);
  int routed[Rows];
#pragma unroll
  for (int j = 0; j < Rows; ++j) {
    routed[j] = first + j < end ? sorted_rows[first + j] : -1;
  }
  if constexpr (!GateUp) ptx::griddep_wait();
  const int routes = *live_tokens * 8;
#pragma unroll
  for (int j = 0; j < Rows; ++j)
    if (routed[j] >= routes) routed[j] = -1;
  if (routed[0] < 0) return;
  if constexpr (OneRow) {
    if constexpr (!GateUp) load_row(routed[0]);
#pragma unroll
    for (int i = 0; i < ChunksPerThread; ++i) {
      const int chunk = threadIdx.x + i * Warps * 32;
      if (chunk < RowChunks) reinterpret_cast<uint4*>(cta_row)[chunk] = row_chunks[i];
    }
    __syncthreads();
  } else {
    issue_first_rows([&](int j) {
      int row = -1;
#pragma unroll
      for (int i = 0; i < Rows; ++i)
        if (i == j && routed[i] >= 0) row = GateUp ? routed[i] / 8 : routed[i];
      return row;
    });
  }

  // The scaled conversion applies each K32 block's power of two, as in
  // the stream-K kernel, so fragments accumulate directly. Token tiles past
  // the last routed row are skipped.
  int tiles = 0;
#pragma unroll
  for (int p = 0; p < TokenTiles; ++p)
    if (routed[8 * p] >= 0) tiles = p + 1;
  ptx::D4 accum[Parts][TokenTiles];
#pragma unroll
  for (int part = 0; part < Parts; ++part)
#pragma unroll
    for (int p = 0; p < TokenTiles; ++p)
#pragma unroll
      for (int i = 0; i < 4; ++i) accum[part][p].v[i] = 0.0f;
#pragma unroll
  for (int kt = 0; kt < KpartTiles; ++kt) {
    cp_async_wait<Depth - 1>();
    __syncwarp();
    const uint8_t* slot = ring + (kt % Depth) * SlotBytes;
    uint4 frag[Parts];
    uint32_t scale[Parts];
#pragma unroll
    for (int part = 0; part < Parts; ++part) {
      frag[part] = *reinterpret_cast<const uint4*>(slot + part * 640 + lane * 16);
      scale[part] = *reinterpret_cast<const uint32_t*>(slot + part * 640 + 512 + lane * 4);
    }
    uint32_t b_words[TokenTiles][8];
#pragma unroll
    for (int p = 0; p < TokenTiles; ++p) {
      if (p >= tiles) break;
      const int b_row = 8 * p + lane / 4 < Rows ? 8 * p + lane / 4 : 0;
      const uint8_t* row = OneRow
          ? reinterpret_cast<const uint8_t*>(cta_row + split * Kpart + kt * 64) + (lane & 3) * 4
          : slot + Parts * 640 + b_row * kFragmentRowBytes + (lane & 3) * 4;
#pragma unroll
      for (int i = 0; i < 8; ++i)
        b_words[p][i] = *reinterpret_cast<const uint32_t*>(row + i * 16);
    }
#pragma unroll
    for (int step = 0; step < 4; ++step) {
      const int block = step >> 1;
      ptx::A4 a[Parts];
#pragma unroll
      for (int part = 0; part < Parts; ++part) {
        const uint32_t word = fragment_word(frag[part], step);
        const uint32_t low = __byte_perm(scale[part], 0, 0x22 * block);
        const uint32_t high = __byte_perm(scale[part], 0, 0x22 * block + 0x11);
        a[part].v[0] = dequant(word, 0, low);
        a[part].v[1] = dequant(word, 1, high);
        a[part].v[2] = dequant(word, 2, low);
        a[part].v[3] = dequant(word, 3, high);
      }
#pragma unroll
      for (int p = 0; p < TokenTiles; ++p) {
        if (p >= tiles) break;
        const ptx::B2 b{{b_words[p][2 * step], b_words[p][2 * step + 1]}};
#pragma unroll
        for (int part = 0; part < Parts; ++part)
          ptx::mma_bf16(accum[part][p], a[part], b);
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
              accum[part][p].v[0], accum[part][p].v[1],
              accum[part][p].v[2], accum[part][p].v[3]);
    }
    __syncthreads();
    if (split > 0) return;
#pragma unroll
    for (int later = 1; later < SplitK; ++later) {
      const float4* in = reinterpret_cast<const float4*>(
          fragment_shared + (warp + later) * Depth * SlotBytes)
          + lane * Parts * TokenTiles;
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

} // namespace chop::moe

// MiMo fragment-major MXFP4 grouped prefill. A persistent CTA walks a static
// list of (expert, feature tile, token tile) work items. Each item is a 256
// weight-row by up to 128 routed-row tile, reduced over K in K64 stages.
// Items with 128 live rows come first, then the partial last tile of each
// expert, largest first; CTAs take items in snake order across rounds, so
// partial items spread evenly.
//
// Warps 8-11 produce into two rings of K64 stages with 16-byte cp.async,
// each thread arriving once per stage on the slot's full barrier. Warp 8
// copies sixteen 512-byte weight fragment tiles and their 32-byte scale
// tiles into a deep weight ring; warps 9-11 copy the 128-byte BF16 row of
// every routed row into a shallow activation ring. Items with few routed
// rows are bound by weight streaming, so only that ring is deep. Warps 0-7
// consume: each owns 32 weight rows (for gate/up, 16 gate and the matching
// 16 up rows) and all routed rows of the item, and releases each slot
// through its empty barrier. No CTA-wide barrier runs after setup; the
// producer warpgroup gives most of its registers to the math warpgroups.
//
// Before an item's first stage the activation producers write the item's
// routed rows and route weights into one of two route slots; the activation
// full barrier of that stage publishes them.
// Activation rows are XOR-swizzled in 16-byte chunks so the eight rows of
// each ldmatrix 8x8 matrix fall in distinct banks.
namespace chop::moe::prefill {

constexpr int kExperts = 256;

constexpr int kTableBytes = (4 * kExperts + 4) * 4;

__host__ __device__ constexpr int shared_bytes(int weight_stages,
                                               int activation_stages) {
  return weight_stages * (kWeightSlotBytes + 16)
       + activation_stages * (kActivationBytes + 16) + kTableBytes
       + kRouteSlots * kRouteSlotBytes + kStagingBytes;
}

struct Tables {
  int* first;         // [kExperts] first padded slot of each expert
  int* live;          // [kExperts] rows up to the last routed row
  int* full_start;    // [kExperts + 1] first full item of each expert, then total
  int* partial;       // [kExperts] experts with a partial tile, largest first
  int* partial_count; // number of those experts
};

// Full items are expert-major, then feature tile, then token tile, so CTAs
// running adjacent items share weight tiles in L2. Each expert's partial
// items follow each other in feature tile order.
template<int FeatureTiles>
__device__ __forceinline__ Tile decode(const Tables& tables, uint64_t work) {
  if (work < uint64_t(tables.full_start[kExperts])) {
    int low = 0, high = kExperts - 1;
#pragma unroll
    for (int step = 0; step < 8; ++step) {
      const int middle = (low + high + 1) / 2;
      if (uint64_t(tables.full_start[middle]) <= work) low = middle;
      else high = middle - 1;
    }
    const int local = int(work) - tables.full_start[low];
    const int token_tiles = tables.live[low] / kTokenTile;
    return {low, local / token_tiles,
            tables.first[low] + local % token_tiles * kTokenTile, kTokenTile};
  }
  const int local = int(work) - tables.full_start[kExperts];
  const int expert = tables.partial[local / FeatureTiles];
  const int live = tables.live[expert];
  return {expert, local % FeatureTiles,
          tables.first[expert] + live / kTokenTile * kTokenTile, live % kTokenTile};
}

// Weight tile j of a CTA tile. Gate/up tiles 0..7 are gate rows and 8..15 the
// matching up rows of 128 output columns; down tiles are 256 adjacent rows.
template<bool GateUp, int N>
__device__ __forceinline__ int strip_of(const Tile& tile, int j) {
  if constexpr (GateUp)
    return (tile.expert * N + (j >= 8 ? N / 2 : 0) + tile.feature_tile * 128
            + (j & 7) * 16) / 16;
  else
    return (tile.expert * N + tile.feature_tile * kWeightRows + j * 16) / 16;
}

// Warp 8 streams one item's weight fragments and scales.
template<bool GateUp, int N, int K, int Stages>
__device__ __forceinline__ void produce_weights(
    const Tile& tile, const Ring& ring, uint32_t& step,
    const uint8_t* weight_bytes, const uint8_t* weight_scales) {
  constexpr int Ktiles = K / 64;
  const int lane = threadIdx.x & 31;
  // Lane l copies 16 bytes at l, l + 32, ... of each 512-byte strip tile and
  // one 16-byte half of a scale tile.
  // Tiles 0-7 and 8-15 are each eight consecutive strips.
  const uint8_t* strips[2];
#pragma unroll
  for (int half = 0; half < 2; ++half)
    strips[half] = weight_bytes
        + uint64_t(strip_of<GateUp, N>(tile, 8 * half)) * Ktiles * 512 + lane * 16;
  const uint8_t* scale = weight_scales
      + uint64_t(strip_of<GateUp, N>(tile, lane >> 1)) * Ktiles * 32 + (lane & 1) * 16;
#pragma unroll 1
  for (int kt = 0; kt < Ktiles; ++kt, ++step) {
    const int slot = step % Stages;
    ptx::mbarrier_wait(&ring.empty[slot], ((step / Stages) & 1) ^ 1);
    uint8_t* stage = ring.slots + slot * kWeightSlotBytes;
#pragma unroll
    for (int j = 0; j < kWeightTiles; ++j)
      cp_async16(stage + j * 512 + lane * 16,
                 strips[j >> 3] + (uint64_t(j & 7) * Ktiles + kt) * 512);
    cp_async16(stage + kWeightBytes + lane * 16, scale + kt * 32);
    cp_async_arrive(&ring.full[slot]);
  }
}

// Warps 9-11 fill one item's route slot, then stream its routed rows.
template<bool GateUp, int K, int Stages, bool TrimGather>
__device__ __forceinline__ void produce_rows(
    const Tile& tile, RouteSlot& route, const Ring& ring, uint32_t& step,
    const uint8_t* activation, const float* topk_weights,
    const int32_t* sorted_rows, int padded, int routes) {
  constexpr int Ktiles = K / 64;
  const int t = threadIdx.x - (kMathWarps + 1) * 32;
  for (int row = t; row < kTokenTile; row += kRowThreads) {
    const int slot = tile.first_row + row;
    int routed = row < tile.rows && slot < padded ? sorted_rows[slot] : -1;
    if (routed >= routes) routed = -1;
    route.rows[row] = routed;
    if (!GateUp && routed >= 0) route.weights[row] = topk_weights[routed];
  }
  // The rows are read back by other threads, and the first stage's barrier
  // arrival below must follow every route slot write.
  row_producer_sync();
#pragma unroll 1
  for (int kt = 0; kt < Ktiles; ++kt, ++step) {
    const int slot = step % Stages;
    ptx::mbarrier_wait(&ring.empty[slot], ((step / Stages) & 1) ^ 1);
    uint8_t* rows = ring.slots + slot * kActivationBytes;
    for (int index = t; index < (TrimGather ? tile.rows : kTokenTile) * 8; index += kRowThreads) {
      const int row = index >> 3;
      const int chunk = index & 7;
      const int routed = route.rows[row];
      if (routed >= 0)
        cp_async16(rows + row_chunk(row, chunk),
                   activation + (uint64_t(GateUp ? routed / 8 : routed) * K
                                 + kt * 64 + chunk * 8) * 2);
    }
    cp_async_arrive(&ring.full[slot]);
    if (t == 0) ptx::mbarrier_arrive(&ring.full[slot]);
  }
}

template<bool GateUp, int Experts, int N, int K, int WeightStages,
         int ActivationStages>
__device__ void run_fragment(
    const uint8_t* activation, const uint8_t* weight_bytes,
    const uint8_t* weight_scales, uint16_t* output,
    const float* topk_weights, const int32_t* expert_offsets,
    const int32_t* sorted_rows, const int32_t* num_padded_rows,
    const int32_t* live_tokens, const int32_t* active_experts) {
  static_assert(Experts == kExperts && K % 64 == 0);
  static_assert(GateUp ? N % 256 == 0 : N % kWeightRows == 0);
  static_assert(WeightStages >= 2 && ActivationStages >= 2);
  constexpr int OutN = GateUp ? N / 2 : N;
  constexpr int FeatureTiles = GateUp ? OutN / 128 : N / kWeightRows;
  constexpr int Ktiles = K / 64;
  // Output columns per warp and 16-byte chunks per staged token row.
  constexpr int WarpColumns = GateUp ? 16 : 32;
  constexpr int Chunks = WarpColumns / 8;
  extern __shared__ __align__(128) uint8_t shared[];
  uint8_t* activation_slots = shared + WeightStages * kWeightSlotBytes;
  uint64_t* barriers = reinterpret_cast<uint64_t*>(
      activation_slots + ActivationStages * kActivationBytes);
  const Ring weight_ring{shared, barriers, barriers + WeightStages};
  const Ring row_ring{activation_slots, barriers + 2 * WeightStages,
                      barriers + 2 * WeightStages + ActivationStages};
  uint8_t* tables_base = reinterpret_cast<uint8_t*>(
      barriers + 2 * (WeightStages + ActivationStages));
  int* table = reinterpret_cast<int*>(tables_base);
  const Tables tables{table, table + kExperts, table + 2 * kExperts,
                      table + 3 * kExperts + 1, table + 4 * kExperts + 1};
  RouteSlot* route_slots = reinterpret_cast<RouteSlot*>(tables_base + kTableBytes);
  uint8_t* staging_base = tables_base + kTableBytes + kRouteSlots * kRouteSlotBytes;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (threadIdx.x == 0) ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const int tokens = *live_tokens;
  const int padded = *num_padded_rows;
  // Expert offsets alone define the work; the active expert list is unused.
  if (tokens <= 0 || padded <= 0) return;
  const int routes = tokens * 8;

  {
    __shared__ int warp_totals[kExperts / 32];
    // Sort keys of experts with a partial tile; the staging buffers are idle.
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
      ptx::mbarrier_init(&row_ring.full[threadIdx.x], kRowThreads + 1);
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
        for (int other = 0; other < kExperts; ++other)
          rank += keys[other] < keys[expert];
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
      const uint64_t work = snake(round);
      if (work >= total) continue;
      const Tile tile = decode<FeatureTiles>(tables, work);
      if (warp == kMathWarps)
        produce_weights<GateUp, N, K, WeightStages>(
            tile, weight_ring, step, weight_bytes, weight_scales);
      else
        produce_rows<GateUp, K, ActivationStages, GateUp && N == 1024>(
            tile, route_slots[item % kRouteSlots], row_ring, step, activation,
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

  uint32_t step = 0;
  int item = 0;
#pragma unroll 1
  for (int round = 0; uint64_t(round) * gridDim.x < total; ++round) {
    const uint64_t work = snake(round);
    if (work >= total) continue;
    const Tile tile = decode<FeatureTiles>(tables, work);
    const RouteSlot& route = route_slots[item++ % kRouteSlots];
    const int groups = min(8, (tile.rows + 15) / 16);
#pragma unroll 1
    for (int kt = 0; kt < Ktiles; ++kt, ++step) {
      const int weight_slot = step % WeightStages;
      const int row_slot = step % ActivationStages;
      ptx::mbarrier_wait(&weight_ring.full[weight_slot], (step / WeightStages) & 1);
      ptx::mbarrier_wait(&row_ring.full[row_slot], (step / ActivationStages) & 1);
      const uint8_t* stage = weight_ring.slots + weight_slot * kWeightSlotBytes;
      const uint8_t* rows = row_ring.slots + row_slot * kActivationBytes;
      uint4 fragment[2];
      uint32_t scale_word[2];
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        const int j = GateUp ? warp + 8 * m : 2 * warp + m;
        fragment[m] = *reinterpret_cast<const uint4*>(stage + j * 512 + lane * 16);
        scale_word[m] = *reinterpret_cast<const uint32_t*>(
            stage + kWeightBytes + j * 32 + (lane >> 2) * 4);
      }
#pragma unroll
      for (int k16 = 0; k16 < 4; ++k16) {
        ptx::A4 a[2];
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          const uint32_t word = fragment_word(fragment[m], k16);
          const int block = k16 >> 1;
          const uint32_t low = __byte_perm(scale_word[m], 0, 0x22 * block);
          const uint32_t high = __byte_perm(scale_word[m], 0, 0x22 * block + 0x11);
          a[m].v[0] = dequant(word, 0, low);
          a[m].v[1] = dequant(word, 1, high);
          a[m].v[2] = dequant(word, 2, low);
          a[m].v[3] = dequant(word, 3, high);
        }
        auto group = [&](int p) {
          const int row = 16 * p + (lane & 7) + 8 * (lane >> 4);
          const int chunk = 2 * k16 + ((lane >> 3) & 1);
          uint32_t b[4];
          ptx::ldmatrix_b16<4>(b, rows + row_chunk(row, chunk));
          const ptx::B2 low_rows{{b[0], b[1]}};
          const ptx::B2 high_rows{{b[2], b[3]}};
#pragma unroll
          for (int m = 0; m < 2; ++m) {
            ptx::mma_bf16(accum[m][2 * p], a[m], low_rows);
            ptx::mma_bf16(accum[m][2 * p + 1], a[m], high_rows);
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

    // Each 16-row group is transposed through the warp's staging buffer, so
    // a lane stores 16 contiguous bytes of one routed row.
    const int column_base = GateUp
        ? tile.feature_tile * 128 + warp * WarpColumns
        : tile.feature_tile * kWeightRows + warp * WarpColumns;
    // The XOR spreads the eight rows of each 8x8 matrix over all banks.
    auto staged = [&](int token, int chunk) {
      const int swizzled = GateUp ? chunk ^ ((token >> 2) & 1)
                                  : chunk ^ ((token >> 1) & 3);
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
            packed[2 * q + h] = pack_bf16(v0, v1);
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
          *reinterpret_cast<uint4*>(output + uint64_t(routed) * OutN
                                    + column_base + chunk * 8) = value;
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

} // namespace chop::moe::prefill
