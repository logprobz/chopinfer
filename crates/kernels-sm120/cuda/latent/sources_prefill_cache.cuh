#pragma once
#include "sources_prefill.cuh"

namespace chop::latent::sources_prefill {

// These producers run with ordinary stream ordering. Clearing page marks,
// marking, decoding, attention and merge are ordered on the same stream.
__device__ void clear_pages(uint32_t* pages, uint32_t count) {
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += uint64_t(gridDim.x) * blockDim.x) pages[i] = 0;
}

template<int Page>
__device__ void mark_pages(const int32_t* row_batch, const uint32_t* live_rows,
    const int32_t* keys, const int32_t* tables, uint32_t* marked,
    uint32_t capacity, uint32_t key_stride, uint32_t budget,
    uint32_t max_pages, uint32_t pages, uint32_t sequences) {
  const uint32_t live = min(*live_rows, capacity);
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < uint64_t(live) * budget; i += uint64_t(gridDim.x) * blockDim.x) {
    const uint32_t row = i / budget, key = i % budget;
    const int batch = row_batch[row];
    if (batch < 0 || uint32_t(batch) >= sequences) continue;
    const int position = keys[uint64_t(row) * key_stride + key];
    if (position < 0 || uint32_t(position) / Page >= max_pages) continue;
    const int page = tables[uint64_t(batch) * max_pages + uint32_t(position) / Page];
    if (page >= 0 && uint32_t(page) < pages) atomicOr(marked + page, 1u);
  }
}

template<int Width>
__device__ void decode_window(const uint8_t* codes, const uint8_t* scales,
    uint16_t* decoded, uint32_t slots, uint32_t sequences,
    uint64_t code_stride, uint64_t scale_stride) {
  const uint64_t vectors = uint64_t(slots) * sequences * (Width / 8);
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < vectors; i += uint64_t(gridDim.x) * blockDim.x) {
    const uint64_t row = i / (Width / 8);
    const int dim = (i % (Width / 8)) * 8;
    const auto packet = load8<false>(
        codes + (row / slots) * code_stride + (row % slots) * Width,
        scales + (row / slots) * scale_stride + (row % slots) * (Width / 32), 1, dim);
    *reinterpret_cast<uint4*>(decoded + i * 8) = decode8<false>(packet);
  }
}

template<int Width, int Page>
__device__ void decode_pages(const uint8_t* pool, const uint32_t* marked,
    uint16_t* decoded, uint32_t pages, uint64_t page_stride) {
  const uint64_t vectors = uint64_t(pages) * Page * (Width / 8);
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < vectors; i += uint64_t(gridDim.x) * blockDim.x) {
    const uint64_t row = i / (Width / 8), page = row / Page;
    if (marked[page] == 0) continue;
    const int slot = row % Page, dim = (i % (Width / 8)) * 8;
    const uint8_t* base = pool + page * page_stride;
    const auto packet = load8<false>(base + slot * (Width / 2),
        sources::NativeAddress::scale_row(base, pool, page * page_stride, Page, slot, Width), 2, dim);
    *reinterpret_cast<uint4*>(decoded + i * 8) = decode8<false>(packet);
  }
}
} // namespace chop::latent::sources_prefill
