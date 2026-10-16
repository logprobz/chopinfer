#pragma once

#include "grouped.cuh"

// Persistent prefill schedule of the grouped-MoE template. A CTA per SM
// walks a static list of (expert, feature tile, token tile) items: 256
// weight rows (for gate/up, 128 gate rows and the matching up rows) by up
// to 128 routed rows, reduced over K in K64 stages. Items with 128 live
// rows come first, expert-major, then each expert's partial tile, largest
// first; CTAs take items in snake order across rounds.
//
// Warp 8 streams each stage's weight strips and scales through the weight
// ring; warps 9-11 fill the item's route slot, then gather its routed BF16
// rows into the activation ring (XOR-swizzled 16-byte chunks). Warps 0-7
// each own 32 weight rows across all routed rows of the item and release
// slots through empty barriers. The producer warpgroup gives its registers
// to the math warpgroups.
namespace chop::moe::grouped {

struct PrefillSchedule {
  int weight_stages;
  int activation_stages;
  // Staging choices that leave the arithmetic unchanged.
  bool trim_gather;   // gather only the item's routed rows, not all 128
  bool bulk_weights;  // FP8 gate/up: one bulk copy per strip and stage
};

template<Weights W>
constexpr int kPrefillWeightSlotBytes = kStageBytes<W>;

__host__ __device__ constexpr int prefill_table_bytes(int experts) {
  return (4 * experts + 4) * 4;
}

template<Weights W>
__host__ __device__ constexpr int prefill_shared_bytes(int weight_stages, int activation_stages,
                                                       int experts) {
  return weight_stages * (kPrefillWeightSlotBytes<W> + 16)
       + activation_stages * (prefill::kActivationBytes + 16) + prefill_table_bytes(experts)
       + prefill::kRouteSlots * prefill::kRouteSlotBytes + prefill::kStagingBytes;
}

struct PrefillTables {
  int* first;          // [experts] first padded slot of each expert
  int* live;           // [experts] rows up to the last routed row
  int* full_start;     // [experts + 1] first full item of each expert, then total
  int* partial;        // [experts] experts with a partial tile, largest first
  int* partial_count;  // number of those experts
};

// Full items are expert-major, then feature tile, then token tile, so CTAs
// running adjacent items share weight tiles in L2. Each expert's partial
// items follow each other in feature tile order.
template<int Experts, int FeatureTiles>
__device__ __forceinline__ Tile prefill_item(const PrefillTables& tables, uint64_t work) {
  constexpr int kTile = prefill::kTokenTile;
  if (work < uint64_t(tables.full_start[Experts])) {
    int low = 0, high = Experts - 1;
#pragma unroll
    for (int step = 0; step < log2_exact(Experts); ++step) {
      const int middle = (low + high + 1) / 2;
      if (uint64_t(tables.full_start[middle]) <= work) low = middle;
      else high = middle - 1;
    }
    const int local = int(work) - tables.full_start[low];
    const int token_tiles = tables.live[low] / kTile;
    return {low, local / token_tiles, tables.first[low] + local % token_tiles * kTile, kTile};
  }
  const int local = int(work) - tables.full_start[Experts];
  const int expert = tables.partial[local / FeatureTiles];
  const int live = tables.live[expert];
  return {expert, local % FeatureTiles, tables.first[expert] + live / kTile * kTile,
          live % kTile};
}

// Warps 9-11 fill one item's route slot, then stream its routed rows.
template<bool GateUp, int K, int TopK, int Stages, bool TrimGather>
__device__ __forceinline__ void prefill_rows(
    const Tile& tile, RouteSlot& route, const Ring& ring, uint32_t& step,
    const uint8_t* activation, const float* topk_weights, const int32_t* sorted_rows, int padded,
    int routes) {
  constexpr int Ktiles = K / 64;
  const int t = threadIdx.x - (prefill::kMathWarps + 1) * 32;
  for (int row = t; row < prefill::kTokenTile; row += prefill::kRowThreads) {
    const int slot = tile.first_row + row;
    int routed = row < tile.rows && slot < padded ? sorted_rows[slot] : -1;
    if (routed >= routes) routed = -1;
    route.rows[row] = routed;
    if (!GateUp && routed >= 0) route.weights[row] = topk_weights[routed];
  }
  // The rows are read back by other threads, and the first stage's barrier
  // arrival below must follow every route slot write.
  prefill::row_producer_sync();
#pragma unroll 1
  for (int kt = 0; kt < Ktiles; ++kt, ++step) {
    const int slot = step % Stages;
    ptx::mbarrier_wait(&ring.empty[slot], ((step / Stages) & 1) ^ 1);
    uint8_t* rows = ring.slots + slot * prefill::kActivationBytes;
    for (int index = t; index < (TrimGather ? tile.rows : prefill::kTokenTile) * 8;
         index += prefill::kRowThreads) {
      const int row = index >> 3;
      const int chunk = index & 7;
      const int routed = route.rows[row];
      if (routed >= 0)
        cp_async16(rows + row_chunk(row, chunk),
                   activation + (uint64_t(GateUp ? routed / TopK : routed) * K
                                 + kt * 64 + chunk * 8) * 2);
    }
    cp_async_arrive(&ring.full[slot]);
    if (t == 0) ptx::mbarrier_arrive(&ring.full[slot]);
  }
}

template<Weights W, Shape S, bool GateUp, PrefillSchedule P>
__device__ void persistent_prefill(
    const uint8_t* activation, const uint8_t* weight_bytes, const uint8_t* weight_scales,
    const float* weight_f32_scales, uint16_t* output, const float* topk_weights,
    const int32_t* expert_offsets, const int32_t* sorted_rows, const int32_t* num_padded_rows,
    const int32_t* live_tokens) {
  using prefill::kMathWarps;
  using prefill::kTokenTile;
  static_assert(served<W>, "instances of this weight format come later");
  constexpr int N = GateUp ? 2 * S.intermediate : S.hidden;
  constexpr int K = GateUp ? S.hidden : S.intermediate;
  constexpr int Experts = S.experts;
  constexpr int TopK = S.top_k;
  constexpr int WeightStages = P.weight_stages;
  constexpr int ActivationStages = P.activation_stages;
  constexpr int SlotBytes = kPrefillWeightSlotBytes<W>;
  constexpr int TableBytes = prefill_table_bytes(Experts);
  static_assert(K % (W == Weights::fp8_block128 ? 128 : 64) == 0);
  static_assert(GateUp ? N % 256 == 0 : N % prefill::kWeightRows == 0);
  static_assert(WeightStages >= 2 && ActivationStages >= 2);
  static_assert(Experts >= 32 && Experts <= 256 && (Experts & (Experts - 1)) == 0);
  static_assert(!P.bulk_weights || (GateUp && W == Weights::fp8_block128));
  constexpr int OutN = GateUp ? N / 2 : N;
  constexpr int FeatureTiles = GateUp ? OutN / 128 : N / prefill::kWeightRows;
  constexpr int Ktiles = K / 64;
  // The row producer writes an item's route slot before waiting on any ring
  // slot, and the math warps read their item's slot until its epilogue ends.
  // With kRouteSlots slots, a producer that runs ActivationStages stages
  // ahead reaches item n + kRouteSlots, and overwrites item n's slot during
  // its epilogue, unless that takes more than ActivationStages stages.
  static_assert(ActivationStages < (prefill::kRouteSlots - 1) * Ktiles,
                "the activation ring would let the row producer reuse a route slot in use");
  constexpr int WarpColumns = GateUp ? 16 : 32;
  extern __shared__ __align__(128) uint8_t prefill_shared[];
  uint8_t* activation_slots = prefill_shared + WeightStages * SlotBytes;
  uint64_t* barriers = reinterpret_cast<uint64_t*>(
      activation_slots + ActivationStages * prefill::kActivationBytes);
  const Ring weight_ring{prefill_shared, barriers, barriers + WeightStages};
  const Ring row_ring{activation_slots, barriers + 2 * WeightStages,
                      barriers + 2 * WeightStages + ActivationStages};
  uint8_t* tables_base = reinterpret_cast<uint8_t*>(
      barriers + 2 * (WeightStages + ActivationStages));
  int* table = reinterpret_cast<int*>(tables_base);
  const PrefillTables tables{table, table + Experts, table + 2 * Experts,
                             table + 3 * Experts + 1, table + 4 * Experts + 1};
  RouteSlot* route_slots = reinterpret_cast<RouteSlot*>(tables_base + TableBytes);
  uint8_t* staging_base = tables_base + TableBytes + prefill::kRouteSlots * prefill::kRouteSlotBytes;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (threadIdx.x == 0) ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const int tokens = *live_tokens;
  const int padded = *num_padded_rows;
  // Expert offsets alone define the work; the active expert list is unused.
  if (tokens <= 0 || padded <= 0) return;
  const int routes = tokens * TopK;

  {
    __shared__ int warp_totals[Experts / 32];
    // Sort keys of experts with a partial tile; the staging buffers are idle.
    int* keys = reinterpret_cast<int*>(staging_base);
    const int expert = threadIdx.x;
    int units = 0, through = 0;
    bool partial = false;
    if (expert < Experts) {
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
      keys[expert] = partial ? (8 - (rest + 15) / 16) * Experts + expert : INT32_MAX;
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
    if (expert < Experts) {
#pragma unroll
      for (int w = 0; w < Experts / 32; ++w)
        if (w < warp) through += warp_totals[w];
      tables.full_start[expert] = through - units;
      if (expert == Experts - 1) tables.full_start[Experts] = through;
      if (partial) {
        int rank = 0;
#pragma unroll 8
        for (int other = 0; other < Experts; ++other) rank += keys[other] < keys[expert];
        tables.partial[rank] = expert;
      }
      if (expert == 0) *tables.partial_count = partials;
    }
    __syncthreads();
  }
  const uint64_t total = uint64_t(tables.full_start[Experts])
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
      const Tile tile = prefill_item<Experts, FeatureTiles>(tables, work);
      if (warp == kMathWarps) {
        produce_weights<W, GateUp, N, K, WeightStages, P.bulk_weights>(
            tile, weight_ring, step, 0, Ktiles, weight_bytes, weight_scales, weight_f32_scales, lane);
      } else {
        prefill_rows<GateUp, K, TopK, ActivationStages, P.trim_gather>(
            tile, route_slots[item % prefill::kRouteSlots], row_ring, step, activation,
            topk_weights, sorted_rows, padded, routes);
      }
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
  BlockScales<W> scales{};

  uint32_t step = 0;
  int item = 0;
#pragma unroll 1
  for (int round = 0; uint64_t(round) * gridDim.x < total; ++round) {
    const uint64_t work = prefill::snake(round);
    if (work >= total) continue;
    const Tile tile = prefill_item<Experts, FeatureTiles>(tables, work);
    const RouteSlot& route = route_slots[item++ % prefill::kRouteSlots];
    const int groups = min(8, (tile.rows + 15) / 16);
#pragma unroll 1
    for (int kt = 0; kt < Ktiles; ++kt, ++step) {
      const int weight_slot = step % WeightStages;
      const int row_slot = step % ActivationStages;
      ptx::mbarrier_wait(&weight_ring.full[weight_slot], (step / WeightStages) & 1);
      ptx::mbarrier_wait(&row_ring.full[row_slot], (step / ActivationStages) & 1);
      consume_stage<W, GateUp, 8>(
          weight_ring.slots + weight_slot * SlotBytes,
          row_ring.slots + row_slot * prefill::kActivationBytes, accum, scales, groups, kt % 2 == 0, lane, warp);
      __syncwarp();
      if (lane == 0) {
        ptx::mbarrier_arrive(&weight_ring.empty[weight_slot]);
        ptx::mbarrier_arrive(&row_ring.empty[row_slot]);
      }
    }

    const int column_base = GateUp
        ? tile.feature_tile * 128 + warp * WarpColumns
        : tile.feature_tile * prefill::kWeightRows + warp * WarpColumns;
    store_tile<GateUp, 8, OutN>(accum, groups, route.rows, route.weights, staging, output,
                                column_base, lane);
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
      for (int n = 0; n < 16; ++n)
#pragma unroll
        for (int i = 0; i < 4; ++i) accum[m][n].v[i] = 0.f;
  }
}

} // namespace chop::moe::grouped
