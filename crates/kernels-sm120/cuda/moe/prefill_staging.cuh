#pragma once

#include <stdint.h>
#include "moe.cuh"

// Pieces of the persistent grouped-MoE schedule (`grouped_prefill.cuh`):
// the K64 weight stage and 128-row activation stage geometry, cp.async
// helpers and mbarrier arrivals, work tiles, snake order, route slots and
// the XOR-swizzled activation rows. The MiMo prefill engine that introduced them is a
// kernel-lab reference (kernel-lab/cuda/baselines).
namespace chop::moe::prefill {

constexpr int kMathWarps = 8;
constexpr int kRowWarps = 3;
constexpr int kRowThreads = kRowWarps * 32;
constexpr int kThreads = (kMathWarps + 1 + kRowWarps) * 32;
constexpr int kTokenTile = 128;
constexpr int kWeightRows = 256;
constexpr int kWeightTiles = kWeightRows / 16;
constexpr int kWeightBytes = kWeightTiles * 512;
constexpr int kScaleBytes = kWeightTiles * 32;
constexpr int kWeightSlotBytes = kWeightBytes + kScaleBytes;
constexpr int kActivationBytes = kTokenTile * 128;
constexpr int kRouteSlots = 2;
constexpr int kRouteSlotBytes = kTokenTile * 8;
constexpr int kStagingBytes = kMathWarps * 1024;

__device__ __forceinline__ void cp_async16(void* shared, const void* global) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
               :: "r"(ptx::shared_address(shared)), "l"(global) : "memory");
}

// Arrives on the barrier once this thread's earlier cp.async copies land.
__device__ __forceinline__ void cp_async_arrive(uint64_t* bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];"
               :: "r"(ptx::shared_address(bar)) : "memory");
}

__device__ __forceinline__ void row_producer_sync() {
  asm volatile("bar.sync 1, %0;" :: "n"(kRowThreads) : "memory");
}

__device__ __forceinline__ uint32_t pack_bf16(float low, float high) {
  uint32_t result;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(result) : "f"(high), "f"(low));
  return result;
}

struct Tile {
  int expert;
  int feature_tile;
  int first_row;
  int rows;
};

// Item of a CTA in a round: CTA order alternates direction between rounds.
__device__ __forceinline__ uint64_t snake(int round) {
  return uint64_t(round) * gridDim.x
       + (round & 1 ? gridDim.x - 1 - blockIdx.x : blockIdx.x);
}

// A route slot: routed row per tile row (-1 when absent), route weights.
struct RouteSlot {
  int32_t rows[kTokenTile];
  float weights[kTokenTile];
};

// Byte offset of 16-byte chunk c of activation row r in a ring slot.
__device__ __forceinline__ int row_chunk(int row, int chunk) {
  return row * 128 + ((chunk ^ (row & 7)) << 4);
}

struct Ring {
  uint8_t* slots;
  uint64_t* full;
  uint64_t* empty;
};

} // namespace chop::moe::prefill
