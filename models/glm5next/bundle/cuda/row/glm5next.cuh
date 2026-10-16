#pragma once
// GLM 5.3 Flash's own row kernels beside the shared row engine: the single
// rounding of an all-reduced F32 row to BF16, the widening of a BF16 row
// into an F32 rank partial, the gather of the rows that need logits, the
// drafter's residual add and the gather of its row inputs. One CTA per row;
// rows at or past the device live count return.
#include "row/row.cuh"

namespace chop::row::glm5next {

// output[row] = bf16(input[row]), round to nearest even.
template<int Width>
__device__ void round_rows(const float* input, uint16_t* output, const uint32_t* live_rows) {
  begin();
  static_assert(Width % 4 == 0);
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  for (int c = threadIdx.x * 4; c < Width; c += blockDim.x * 4) {
    const float4 v = reinterpret_cast<const float4*>(input + uint64_t(row) * Width + c)[0];
    uint2 bits;
    bits.x = uint32_t(to_bf(v.x)) | (uint32_t(to_bf(v.y)) << 16);
    bits.y = uint32_t(to_bf(v.z)) | (uint32_t(to_bf(v.w)) << 16);
    reinterpret_cast<uint2*>(output + uint64_t(row) * Width + c)[0] = bits;
  }
}

// output[row] = f32(input[row]), exact.
template<int Width>
__device__ void widen_rows(const uint16_t* input, float* output, const uint32_t* live_rows) {
  begin();
  static_assert(Width % 4 == 0);
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  for (int c = threadIdx.x * 4; c < Width; c += blockDim.x * 4) {
    const uint2 bits = reinterpret_cast<const uint2*>(input + uint64_t(row) * Width + c)[0];
    const float4 v = make_float4(bf(uint16_t(bits.x)), bf(uint16_t(bits.x >> 16)),
                                 bf(uint16_t(bits.y)), bf(uint16_t(bits.y >> 16)));
    reinterpret_cast<float4*>(output + uint64_t(row) * Width + c)[0] = v;
  }
}

// x[row] = bf16(x[row] + y[row]), F32 add, one rounding: the drafter's
// single residual stream.
template<int Width>
__device__ void residual_add(uint16_t* x, const uint16_t* y, const uint32_t* live_rows) {
  begin();
  static_assert(Width % 2 == 0);
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  for (int c = threadIdx.x * 2; c < Width; c += blockDim.x * 2) {
    const uint64_t at = uint64_t(row) * Width + c;
    const uint32_t a = reinterpret_cast<const uint32_t*>(x + at)[0];
    const uint32_t b = reinterpret_cast<const uint32_t*>(y + at)[0];
    const float lo = bf(uint16_t(a)) + bf(uint16_t(b));
    const float hi = bf(uint16_t(a >> 16)) + bf(uint16_t(b >> 16));
    reinterpret_cast<uint32_t*>(x + at)[0] = uint32_t(to_bf(lo)) | (uint32_t(to_bf(hi)) << 16);
  }
}

// output[i] = input[row_indices[i]] for the first `*count` rows.
template<int Width>
__device__ void gather_rows(const uint16_t* input, const uint32_t* row_indices,
                            const uint32_t* count, uint16_t* output) {
  begin();
  static_assert(Width % 8 == 0);
  const uint32_t row = blockIdx.x;
  if (row >= *count) return;
  const uint64_t source = row_indices[row];
  for (int c = threadIdx.x * 8; c < Width; c += blockDim.x * 8)
    reinterpret_cast<uint4*>(output + uint64_t(row) * Width + c)[0] =
        reinterpret_cast<const uint4*>(input + source * Width + c)[0];
}

// The MTP drafter's row inputs: hidden row `row` from a slot's retained
// rows (code < 0: entry -1 - code) or a row the previous program left
// (code >= 0), and its token kept from the run's inputs (code -1) or taken
// from the drafting item's latest draft (code >= 0). `codes` holds the
// pair per row.
template<int Width>
__device__ void draft_gather(const uint16_t* retained, const uint16_t* last,
                             const int32_t* codes, const uint32_t* drafts,
                             const uint32_t* live_rows, uint16_t* output, uint32_t* tokens) {
  begin();
  static_assert(Width % 8 == 0);
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  const int32_t hidden = codes[2 * row];
  const int32_t token = codes[2 * row + 1];
  const uint16_t* from = hidden < 0 ? retained + uint64_t(-1 - int64_t(hidden)) * Width
                                    : last + uint64_t(hidden) * Width;
  for (int c = threadIdx.x * 8; c < Width; c += blockDim.x * 8)
    reinterpret_cast<uint4*>(output + uint64_t(row) * Width + c)[0] =
        reinterpret_cast<const uint4*>(from + c)[0];
  if (threadIdx.x == 0 && token >= 0) tokens[row] = drafts[token];
}

}  // namespace chop::row::glm5next
