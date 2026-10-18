#pragma once
#include <stdint.h>
#include "../common/ptx.cuh"

namespace chop::latent {

template<int Mutation = 0>
__device__ void window_stage(const uint8_t* ring_codes, const uint8_t* ring_scales,
    const int32_t* item_slots, const int32_t* first_positions, const int32_t* last_positions,
    uint8_t* codes, uint8_t* scales, const uint32_t* live_items,
    uint32_t items, uint32_t sequences, uint32_t window, uint32_t ring_slots,
    uint64_t read_codes, uint64_t read_scales, uint64_t write_codes, uint64_t write_scales) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t item = blockIdx.x, row = blockIdx.y;
  if (item >= items || item >= *live_items) return;
  const int32_t slot = item_slots[item], first = first_positions[item];
  if (slot < 0 || uint32_t(slot) >= sequences || first < 0) return;
  const int32_t last = last_positions[slot];
  if (last < 0) return;
  int64_t start = int64_t(first) - window + 1;
  if constexpr (Mutation != 4) { if (start < 0) start = 0; }
  if constexpr (Mutation == 1) ++start;
  const int64_t position = start + row;
  // Validate absolute visibility before mapping any physical ring cell.
  if (position < 0 || position >= first || position > last || position + ring_slots <= last) return;
  uint32_t source = uint64_t(position) % ring_slots;
  if constexpr (Mutation == 3) source = (source + 1) % ring_slots;
  if constexpr (Mutation == 2) read_scales = read_codes / 32;
  const uint64_t from_codes = uint64_t(slot) * read_codes + uint64_t(source) * 512;
  const uint64_t from_scales = uint64_t(slot) * read_scales + uint64_t(source) * 16;
  const uint64_t to_codes = uint64_t(item) * write_codes + uint64_t(row) * 512;
  const uint64_t to_scales = uint64_t(item) * write_scales + uint64_t(row) * 16;
  for (uint32_t col = threadIdx.x; col < 512; col += blockDim.x)
    codes[to_codes + col] = ring_codes[from_codes + col];
  if (threadIdx.x < 16) scales[to_scales + threadIdx.x] = ring_scales[from_scales + threadIdx.x];
}

} // namespace chop::latent

namespace chop::latent {
// The BF16 comparison layout has one plane, already in native code order.
// Keep the exact same absolute-position guards as the native gather.
template<int Mutation = 0>
__device__ void window_stage_bf16(const uint8_t* ring_codes, const uint8_t* ring_scales,
    const int32_t* item_slots, const int32_t* first_positions, const int32_t* last_positions,
    uint8_t* codes, uint8_t* scales, const uint32_t* live_items,
    uint32_t items, uint32_t sequences, uint32_t window, uint32_t ring_slots,
    uint64_t read_codes, uint64_t read_scales, uint64_t write_codes, uint64_t write_scales) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t item = blockIdx.x, row = blockIdx.y;
  if (item >= items || item >= *live_items) return;
  const int32_t slot = item_slots[item], first = first_positions[item];
  if (slot < 0 || uint32_t(slot) >= sequences || first < 0) return;
  const int32_t last = last_positions[slot];
  if (last < 0) return;
  int64_t start = int64_t(first) - window + 1;
  if constexpr (Mutation != 4) { if (start < 0) start = 0; }
  if constexpr (Mutation == 1) ++start;
  const int64_t position = start + row;
  if (position < 0 || position >= first || position > last || position + ring_slots <= last) return;
  uint32_t source = uint64_t(position) % ring_slots;
  if constexpr (Mutation == 3) source = (source + 1) % ring_slots;
  if constexpr (Mutation == 2) read_codes = read_codes / 2;
  const uint64_t from = uint64_t(slot)*read_codes + uint64_t(source)*1024;
  const uint64_t to = uint64_t(item)*write_codes + uint64_t(row)*1024;
  for (uint32_t col = threadIdx.x; col < 1024; col += blockDim.x) codes[to+col] = ring_codes[from+col];
}
} // namespace chop::latent
