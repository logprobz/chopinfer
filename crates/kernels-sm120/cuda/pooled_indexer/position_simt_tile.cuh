#pragma once
#include "position_live.cuh"

namespace chop::position_simt_tile {
constexpr int Rows = 4;
constexpr int Columns = 64;
constexpr int HeadTile = 4;

// Each thread owns two complete scores. The virtual lane array preserves
// the original four-FMA chains and the contributing shuffle-down pairings.
// No floating-point operation crosses threads. Packed BF16 key pairs are
// transposed in shared memory, so a warp reads consecutive key positions.
template<int Heads, int Dim, int Page, int Ratio, bool Mask, bool Bf16,
         bool ReverseTree = false>
__device__ void scores(const uint16_t* queries, const float* weights,
    const uint8_t* codes, const uint8_t* scales, const int32_t* tables,
    const int32_t* batches, const int32_t* positions, const uint32_t* live_rows,
    const uint8_t* mask, float* output, uint32_t width, uint32_t max_pages,
    uint64_t code_stride, uint64_t scale_stride, float weight_scale,
    uint32_t launch_rows) {
  static_assert(Heads == 32 && Dim == 128);
  __shared__ uint32_t keys[Dim / 2][Columns];
  __shared__ uint32_t qtile[Rows][HeadTile][Dim / 2];
  __shared__ float wt[Rows][HeadTile];
  __shared__ uint32_t key_bad[Columns];
  __shared__ uint32_t count[Rows];
  __shared__ int32_t batch[Rows];
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t first = blockIdx.y * Rows;
  const uint32_t live = min(*live_rows, launch_rows);
  if (first >= live) return;
  const int r = threadIdx.x / 32, column = threadIdx.x & 31;
  const uint32_t row = first + r;
  if (threadIdx.x < Rows) {
    const uint32_t at = first + threadIdx.x;
    count[threadIdx.x] = at < live ? min(position_indexer::visible(positions[at],Ratio),width) : 0;
    batch[threadIdx.x] = at < live ? batches[at] : -1;
  }
  __syncthreads();
  if (row < live) {
    for (uint32_t i = count[r] + blockIdx.x * 32 + column; i < width; i += gridDim.x * 32)
      output[size_t(row) * width + i] = -__int_as_float(0x7f800000);
  }
  // Prefill rows usually share one sequence. Decode may contain four
  // different sequences; stage each distinct sequence once, without
  // assuming that neighboring rows share a page table or visibility.
  for (int owner = 0; owner < Rows; ++owner) {
    bool leader = first + owner < live;
    for (int prior = 0; prior < owner; ++prior) leader &= batch[prior] != batch[owner];
    if (!leader) continue;
    uint32_t extent = 0;
    for (int i = 0; i < Rows; ++i)
      if (first + i < live && batch[i] == batch[owner]) extent = max(extent,count[i]);
    const bool own = row < live && batch[r] == batch[owner];
    const int32_t* table = tables + size_t(batch[owner]) * max_pages;
    for (uint32_t base = blockIdx.x * Columns; base < extent; base += gridDim.x * Columns) {
      if (threadIdx.x < Columns) key_bad[threadIdx.x] = 0;
      __syncthreads();
      for (int i = threadIdx.x; i < (Dim / 2) * Columns; i += blockDim.x) {
        const int pair = i / Columns, slot = i % Columns;
        const uint32_t index = base + slot;
        uint32_t value = 0;
        if (position_indexer::admitted<Page,false>(index,extent,nullptr,table)) {
          const uint32_t page = uint32_t(table[index / Page]);
          if constexpr (Bf16) {
            const uint16_t* source = reinterpret_cast<const uint16_t*>(codes + uint64_t(page)*code_stride)
                + size_t(index % Page)*Dim + 2*pair;
            value = uint32_t(source[0]) | (uint32_t(source[1]) << 16);
          } else {
            const uint8_t scale = scales[uint64_t(page)*scale_stride + size_t(index % Page)*(Dim/32) + pair/16];
            const uint8_t code = codes[uint64_t(page)*code_stride + size_t(index % Page)*(Dim/2) + pair];
            value = convert::scaled_mxfp4_bf16x2<false>(code,scale);
            if (scale == 255) atomicOr(key_bad + slot,1u);
          }
        }
        keys[pair][slot] = value;
      }
      bool active[2];
#pragma unroll
      for (int p = 0; p < 2; ++p) {
        const uint32_t index = base + column + p*32;
        const uint8_t* row_mask = Mask ? mask + size_t(row)*width : nullptr;
        active[p] = own && position_indexer::admitted<Page,Mask>(index,count[r],row_mask,table);
      }
      float result[2] = {};
      bool invalid[2] = {};
      for (int head_base = 0; head_base < Heads; head_base += HeadTile) {
        for (int i = threadIdx.x; i < Rows*HeadTile*(Dim/2); i += blockDim.x) {
          const int pair = i % (Dim/2), h = (i/(Dim/2)) % HeadTile, qr = i/(HeadTile*(Dim/2));
          uint32_t value = 0;
          if (first + qr < live && batch[qr] == batch[owner]) {
            const uint16_t* source = queries + (size_t(first+qr)*Heads + head_base+h)*Dim + 2*pair;
            value = uint32_t(source[0]) | (uint32_t(source[1]) << 16);
          }
          qtile[qr][h][pair] = value;
        }
        if (threadIdx.x < Rows*HeadTile) {
          const int qr = threadIdx.x/HeadTile, h = threadIdx.x%HeadTile;
          wt[qr][h] = first+qr < live && batch[qr] == batch[owner]
              ? __fmul_rn(weights[size_t(first+qr)*Heads + head_base+h],weight_scale) : 0.f;
        }
        __syncthreads();
#pragma unroll 1
        for (int h = 0; h < HeadTile; ++h) {
          if (active[0] || active[1]) {
            float lanes[2][32];
#pragma unroll
            for (int v = 0; v < 32; ++v) {
              float chain[2] = {};
#pragma unroll
              for (int part = 0; part < 4; ++part) {
                const int c = part*32+v, shift = (v&1)*16;
                const float q = position_indexer::bf16(uint16_t(qtile[r][h][c/2] >> shift));
#pragma unroll
                for (int p = 0; p < 2; ++p) {
                  const float k = position_indexer::bf16(uint16_t(keys[c/2][column+p*32] >> shift));
                  chain[p] = __fmaf_rn(q,k,chain[p]);
                }
              }
#pragma unroll
              for (int p = 0; p < 2; ++p) lanes[p][v] = chain[p];
            }
            // The mutant uses an adjacent-pair tree to demonstrate that
            // the fixture detects changed associations, not just products.
#pragma unroll
            for (int level = 0; level < 5; ++level) {
              constexpr bool reverse = ReverseTree;
              const int offset = reverse ? (1 << level) : (16 >> level);
#pragma unroll
              for (int v = 0; v < 32; ++v) {
                if ((!reverse && v < offset) || (reverse && v % (2*offset) == 0)) {
#pragma unroll
                  for (int p = 0; p < 2; ++p)
                    lanes[p][v] = __fadd_rn(lanes[p][v],lanes[p][v+offset]);
                }
              }
            }
#pragma unroll
            for (int p = 0; p < 2; ++p) {
              const float dot = __fadd_rn(0.f,lanes[p][0]);
              // A nonfinite lane FMA or contributing tree node cannot
              // become finite later in that dot. Check before ReLU, which
              // would otherwise hide a negative infinity or NaN.
              invalid[p] |= !isfinite(dot);
              result[p] = __fmaf_rn(dot > 0.f ? dot : 0.f,wt[r][h],result[p]);
            }
          }
        }
        __syncthreads();
      }
#pragma unroll
      for (int p = 0; p < 2; ++p) {
        const uint32_t index = base + column + p*32;
        if (own && index < count[r]) {
          float value = -__int_as_float(0x7f800000);
          if (active[p]) value = invalid[p] || key_bad[column+p*32] || !isfinite(result[p])
              ? __int_as_float(0x7fc00000) : result[p];
          output[size_t(row)*width+index] = value;
        }
      }
      __syncthreads();
    }
  }
}
} // namespace chop::position_simt_tile
