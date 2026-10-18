#pragma once

// Lab-only address mutants reuse the serving attention arithmetic. The
// fixture backs every mutated read with finite guard bytes; no trap runs.
namespace chop::latent::page_controls {
template<int Rows, int ScaleRows, int Bias>
struct Page {
  __host__ __device__ constexpr explicit operator uint64_t() const { return ScaleRows; }
  __device__ __forceinline__ friend uint32_t operator/(uint32_t position, Page) {
    return position / Rows + Bias;
  }
  __device__ __forceinline__ friend uint32_t operator%(uint32_t position, Page) {
    return position % Rows;
  }
};
template<class T, int Rows, int ScaleRows, int Bias>
struct Geometry : T {
  static constexpr Page<Rows, ScaleRows, Bias> page{};
};
}
