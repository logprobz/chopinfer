#pragma once

// The latent KV write (kernel-core `LatentAttentionQuery::KvWrite`): stores
// each row's BF16 latent at its token position through the sequence's page
// table, as BF16 or as E4M3 codes with one F32 scale per 128 elements.

#include <stdint.h>
#include "../common/convert.cuh"
#include "../common/ptx.cuh"

namespace chop::latent {

template<int Latent, int Page, bool Fp8>
struct KvWrite {
  static_assert(Latent % 128 == 0 && Latent <= 1024);
  static constexpr int latent = Latent, page = Page;
  static constexpr bool fp8 = Fp8;
  static constexpr int blocks = Latent / 128;
  // One warp per row, one 128-element block at a time: lane l owns elements
  // [4 l, 4 l + 4) of each block, so every width in 128s partitions exactly.
  static constexpr int per_lane = 128 / 32;
  static constexpr int rows_per_cta = 4, threads = rows_per_cta * 32;
};

// Grid [ceil(rows / 4)], 128 threads. Rows past the live count and rows with
// a negative position write nothing. FP8: a block's scale is its maximum
// magnitude (a full-warp butterfly) divided by 448 in F32 (1 for an all-zero
// block), and each code is `E4M3(x / scale)` with the F32 division and
// `cvt.rn.satfinite`; lane 0 stores the block's scale.
template<class T>
__device__ void kv_write(const uint16_t* latent, const int32_t* row_batch,
                         const int32_t* positions, const uint32_t* live_rows,
                         const int32_t* block_tables, uint8_t* pool, uint32_t max_pages,
                         uint64_t page_stride_bytes) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const int lane = threadIdx.x % 32;
  const int row = blockIdx.x * T::rows_per_cta + threadIdx.x / 32;
  if (row >= int(*live_rows)) return;
  const int position = positions[row];
  if (position < 0) return;
  const uint64_t page =
      uint32_t(block_tables[uint64_t(row_batch[row]) * max_pages + position / T::page]);
  const uint64_t slot = position % T::page;
  uint8_t* base = pool + page * page_stride_bytes;
#pragma unroll
  for (int block = 0; block < T::blocks; ++block) {
    const int first = block * 128 + lane * T::per_lane;
    const uint2 words =
        *reinterpret_cast<const uint2*>(latent + uint64_t(row) * T::latent + first);
    if constexpr (!T::fp8) {
      *reinterpret_cast<uint2*>(base + (slot * T::latent + first) * 2) = words;
    } else {
      const convert::Float2 low = convert::bf16x2_to_f32(words.x);
      const convert::Float2 high = convert::bf16x2_to_f32(words.y);
      float amax = fmaxf(fmaxf(fabsf(low.lo), fabsf(low.hi)),
                         fmaxf(fabsf(high.lo), fabsf(high.hi)));
#pragma unroll
      for (int offset = 1; offset < 32; offset *= 2)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset));
      const float scale = amax == 0.0f ? 1.0f : __fdiv_rn(amax, 448.0f);
      const uint32_t codes =
          uint32_t(convert::pack_f32_e4m3(
              {__fdiv_rn(low.lo, scale), __fdiv_rn(low.hi, scale)})) |
          uint32_t(convert::pack_f32_e4m3(
              {__fdiv_rn(high.lo, scale), __fdiv_rn(high.hi, scale)})) << 16;
      *reinterpret_cast<uint32_t*>(base + slot * T::latent + first) = codes;
      if (lane == 0) {
        reinterpret_cast<float*>(base + uint64_t(T::page) * T::latent)[slot * T::blocks + block] =
            scale;
      }
    }
  }
}

} // namespace chop::latent
