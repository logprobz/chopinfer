#pragma once
#include "../attention/tile.cuh"

namespace chop::latent::sources_prefill {

// Copy logical key rows into the dimension-major swizzle. A 32-key half
// is not a contiguous half of the allocation. The caller releases all old
// readers first, then waits for copies and establishes CTA visibility.
template<int Keys, int Width>
__device__ __forceinline__ void stage_bf16_keys(
    uint8_t* tile, const uint8_t* const* rows, const int32_t* valid,
    int first, int end, uint32_t phase) {
  static_assert(Width % 8 == 0);
  for (int i = first * (Width / 8) + threadIdx.x;
       i < end * (Width / 8); i += blockDim.x) {
    const int key = i / (Width / 8), dim = i % (Width / 8) * 8;
    void* dest = tile + attention::tile_offset<Keys>(key, dim, phase);
    const uint8_t* source = valid[key] ? rows[key] + dim * 2 : nullptr;
    if ((reinterpret_cast<uintptr_t>(source) & 15) == 0) {
      attention::cp_async_zfill(dest, source, valid[key] != 0);
    } else {
      // Generic BF16 views may be only two-byte aligned. Preserve their
      // exact bits; decoded-cache rows take the aligned asynchronous arm.
      const auto* words = reinterpret_cast<const uint16_t*>(source);
      uint32_t pairs[4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
        pairs[j] = uint32_t(words[2 * j]) | uint32_t(words[2 * j + 1]) << 16;
      *reinterpret_cast<uint4*>(dest) = make_uint4(pairs[0], pairs[1], pairs[2], pairs[3]);
    }
  }
  attention::cp_async_commit();
}

} // namespace chop::latent::sources_prefill
