#pragma once
#include <stdint.h>

// Addressing of E4M3 weight codes stored as `mxfp8-fold-packed-8x128-v1`
// (`kernel_core::weight_transform::mxfp8_fold`): the layout the stream and
// direct folds read, and the one every fold reader moves to so the row-major
// copy can go.
//
// Rows are counted across groups (row `g * N + n`); N is a multiple of 8 and
// K a multiple of 64. Each eight-row strip is 8 * K bytes: its 128-K folds in
// order, each 1024 bytes, then a final 512-byte fold when K % 128 == 64.
// Inside a fold, row r of the strip owns one contiguous 128-byte segment
// (64 bytes in the final half fold). The four-byte code word holding K bytes
// [16 * c + 4 * l, 16 * c + 4 * l + 4) of the fold sits at byte 32 * l + 4 * c
// of that segment (16 * l + 4 * c in a half fold): lane l of the row's four
// lanes owns words c = 0..7 (0..3), every fourth word of the row.
//
// A row-major reader that loads a K word becomes a packed reader by changing
// only that address; a reader that stages 16-byte row chunks stages the
// strip's contiguous fold instead and reads its words at `segment_word`.
// Neither changes arithmetic or accumulation order.
namespace chop::gemm::mxfp8_fold_packed {

// Bytes of fold `fold` of one strip: 1024, or 512 for a final 64-K fold.
__device__ __host__ __forceinline__ constexpr uint32_t fold_bytes(uint32_t fold, uint32_t k) {
  return (k % 128 != 0 && fold == k / 128) ? 512u : 1024u;
}

// Byte offset, within the strip's fold `fold`, of the word holding K bytes
// [local, local + 4) of the fold for strip row `r` (local % 4 == 0).
__device__ __host__ __forceinline__ constexpr uint32_t segment_word(
    uint32_t r, uint32_t local, bool half) {
  const uint32_t chunk = local / 16, lane = (local % 16) / 4;
  return half ? r * 64 + lane * 16 + chunk * 4 : r * 128 + lane * 32 + chunk * 4;
}

// Byte offset of strip `strip`'s fold `fold` in a matrix of K-wide rows.
__device__ __host__ __forceinline__ constexpr uint64_t fold_offset(
    uint32_t strip, uint32_t fold, uint32_t k) {
  return uint64_t(strip) * 8 * k + uint64_t(fold) * 1024;
}

// Byte offset of the word holding K bytes [k, k + 4) of row `row`, which a
// row-major reader finds at row * K + k (k % 4 == 0).
__device__ __host__ __forceinline__ constexpr uint64_t word_offset(
    uint32_t row, uint32_t k, uint32_t width) {
  const uint32_t fold = k / 128;
  const bool half = width % 128 != 0 && fold == width / 128;
  return fold_offset(row / 8, fold, width) + segment_word(row % 8, k % 128, half);
}

// The word itself, for a reader that loads four codes at once.
__device__ __forceinline__ uint32_t load_word(const uint8_t* packed, uint32_t row, uint32_t k,
                                              uint32_t width) {
  return *reinterpret_cast<const uint32_t*>(packed + word_offset(row, k, width));
}

}  // namespace chop::gemm::mxfp8_fold_packed
