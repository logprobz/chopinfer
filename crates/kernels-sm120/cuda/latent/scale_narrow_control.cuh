#pragma once

// Lab-only mutant: code addresses stay wide; only the scale offset narrows.
namespace chop::latent::scale_narrow_control {
struct Address {
  __device__ __forceinline__ static const uint8_t* scale_row(
      const uint8_t*, const uint8_t* pool, uint64_t page_offset,
      uint64_t rows, uint32_t slot, uint32_t width) {
    const uint32_t offset = uint32_t(page_offset + rows * (width / 2) + uint64_t(slot) * (width / 16));
    return pool + offset;
  }
};
}
