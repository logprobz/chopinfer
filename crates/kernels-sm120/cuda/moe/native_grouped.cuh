#pragma once

#include "chained_experts.cuh"
#include "native_grouped_layout.h"

// Native block-scaled expert tiles for prefill: MXFP4 weights by the
// trained E4M3/32 ceil-UE8M0 activations through mma.sync kind::mxf8f6f4
// block_scale, with no decode to BF16. This is a within-bound path, not a
// bitwise one: every product is exact, each 128-K chain starts from +0 and
// is added into the F32 accumulator in ascending K order as in the exact
// BF16 engine, and only the instruction's own reduction inside a 32-K
// group differs. Admission traps on the codes and scales the exact engine
// traps on, and the gate/up epilogue is the exact engine's.
//
// One launch packs the active experts into row tiles; a persistent launch
// takes (tile, column strip) items from an atomic ticket. Which CTA runs an
// item never changes its arithmetic.
//
// With Config's Tiles the weights are stored in 64-row by 128-K tiles
// (`row-tiles-64x128-groups`): a weight row's 128-K stage is one 64-byte
// tile row and its four scale bytes one aligned word, so each copy moves
// the same bytes into the same shared slot. With Lanes too the tile rows'
// bytes are in quad-lane order (`row-tiles-64x128-lanes`): a thread copies
// whole tile rows and, after its own wait, restores their row-tiled order
// in place (`untangle`) before the stage barrier, so every warp reads the
// same shared bytes.
namespace chop::moe::native_grouped {
namespace lay = layout;

template<int Experts, int TopK, int K, int N, int Halves, int Capacity, class Clamp,
         bool Tiles = false, bool Lanes = false>
struct Config {
  static constexpr int experts = Experts, top_k = TopK, k = K, n = N, halves = Halves;
  static constexpr bool tiles = Tiles, lanes = Lanes;
  static constexpr int capacity = Capacity;
  static constexpr bool gate_up = Halves == 2;
  using clamp = Clamp;
  static constexpr int weight_rows = Halves * N;  // gate rows, then up rows
  static constexpr int groups = K / lay::GroupK;   // scale bytes per row
  static constexpr int blocks = (K + lay::StageK - 1) / lay::StageK;
  static constexpr int tail_groups = (K % lay::StageK) / lay::GroupK;
  // Rows whose scale bytes do not start on a word take two words.
  static constexpr int scale_words = groups % 4 ? 2 : 1;
  static_assert(K > 0 && K % lay::GroupK == 0 && N > 0 && N % 16 == 0);
  static_assert(Halves == 1 || Halves == 2);
  static_assert(gate_up || !Clamp::enabled, "the clamp belongs to gate/up");
  static_assert(int64_t(Capacity) * TopK < (int64_t(1) << 31));
  static_assert(!Tiles || (K % lay::StageK == 0 && weight_rows % 64 == 0 && scale_words == 1),
                "row tiles need whole 64x128 tiles");
  static_assert(!Lanes || Tiles, "quad lanes order the bytes of row tiles");
};

template<int RowWarps, int ColWarps, int Stages>
struct Tile {
  static constexpr int row_warps = RowWarps, col_warps = ColWarps, stages = Stages;
  static constexpr int fragments = 8;  // routed fragments of eight rows per warp
  static constexpr int rows = RowWarps * fragments * 8;
  static constexpr int threads = RowWarps * ColWarps * 32;
  static_assert(RowWarps >= 1 && ColWarps >= 1 && Stages >= 2 && Stages <= 4);
};

// Gate/up warps own 16 output columns (one gate and one up fragment),
// down warps 32 (two fragments). Either way a warp holds two A fragments.
template<class C> constexpr int warp_cols = C::gate_up ? 16 : 32;
template<class C, class T> constexpr int tile_cols = T::col_warps * warp_cols<C>;
template<class C, class T> constexpr int tile_weight_rows = C::halves * tile_cols<C, T>;
template<class C, class T> constexpr int strips = (C::n + tile_cols<C, T> - 1) / tile_cols<C, T>;

// table[0] tiles, table[1] the item ticket, then (expert, first, rows).
template<class C, class T> struct Queue {
  static constexpr int header = 4;
  static constexpr int slots = (C::capacity * C::top_k + T::rows - 1) / T::rows + C::experts;
  static constexpr int words = header + 3 * slots;
};

template<class C, class T> struct Shared {
  struct alignas(16) Stage {
    uint8_t act[T::rows * lay::ActRowBytes];
    uint8_t weight[tile_weight_rows<C, T> * lay::WeightRowBytes];
    uint32_t act_scale[T::rows][C::scale_words];
    uint32_t weight_scale[tile_weight_rows<C, T>][C::scale_words];
  } stage[T::stages];
  int routes[T::rows];
  int sources[T::rows];
  int item;
};

__device__ __forceinline__ void copy16(void* dst, const void* src, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
    :: "r"(ptx::shared_address(dst)), "l"(src), "r"(valid ? 16 : 0) : "memory");
}
// The aligned word `word` of a row's stage scales; bytes past `total`
// are zero filled.
__device__ __forceinline__ void copy_scale(uint32_t* dst, const uint8_t* scales,
                                         uint64_t at, int word, uint64_t total, bool valid) {
  const uint64_t aligned = (at & ~uint64_t(3)) + 4 * uint64_t(word);
  const int size = valid && aligned < total ? int(min(uint64_t(4), total - aligned)) : 0;
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;"
    :: "r"(ptx::shared_address(dst)), "l"(scales + (size ? aligned : 0)), "r"(size) : "memory");
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template<int Pending> __device__ __forceinline__ void wait() {
  asm volatile("cp.async.wait_group %0;" :: "n"(Pending) : "memory");
}
__device__ __forceinline__ uint32_t lds32(const uint8_t* p) {
  return *reinterpret_cast<const uint32_t*>(p);
}
__device__ __forceinline__ uint2 lds64(const uint8_t* p) {
  return *reinterpret_cast<const uint2*>(p);
}

// The exact engine's MXFP4 admission (streamed::check) over eight codes.
__device__ __forceinline__ void check_weights(uint32_t codes, uint32_t scale) {
  if (scale == 255) __trap();
  if (scale > 252) {
    const uint32_t bad = scale == 254 ? codes : codes & (codes << 1);
    if (bad & 0x44444444u) __trap();
  }
}
// The exact engine's E4M3 admission: NaN codes always, and the exact BF16
// product check at scales outside 3..246.
__device__ __forceinline__ void check_activations(uint2 codes, uint32_t scale) {
  if (scale < 3 || scale > 246) {
    (void)chained::scaled_e4m3x2(uint16_t(codes.x), uint8_t(scale));
    (void)chained::scaled_e4m3x2(uint16_t(codes.x >> 16), uint8_t(scale));
    (void)chained::scaled_e4m3x2(uint16_t(codes.y), uint8_t(scale));
    (void)chained::scaled_e4m3x2(uint16_t(codes.y >> 16), uint8_t(scale));
  }
}

// One warp's packed active experts. A warp prefix sum keeps expert order
// without atomic arrival order; the ticket restarts at zero.
template<class C, class T>
__device__ void pack(const int* counts, const int* offsets, const int* active,
                     const int* live_tokens, int* table) {
  using Q = Queue<C, T>;
  const int lane = threadIdx.x, live = *live_tokens, total = active[0];
  if (blockDim.x != 32 || gridDim.x != 1 || live < 0 || live > C::capacity ||
      total < 0 || total > C::experts || total > live * C::top_k) __trap();
  int carry = 0;
  for (int base = 0; base < total; base += 32) {
    const bool valid = base + lane < total;
    const int expert = valid ? active[base + lane + 1] : 0;
    if (expert < 0 || expert >= C::experts) __trap();
    const int count = valid ? counts[expert] : 0, first = valid ? offsets[expert] : 0;
    if (valid && (count <= 0 || count > live || first < 0 || first % 64)) __trap();
    const int tiles = valid ? lay::tiles(count, T::rows) : 0;
    int prefix = tiles;
#pragma unroll
    for (int d = 1; d < 32; d *= 2) {
      const int previous = __shfl_up_sync(0xffffffffu, prefix, d);
      if (lane >= d) prefix += previous;
    }
    const int begin = carry + prefix - tiles;
    if (begin + tiles > Q::slots) __trap();
    for (int tile = 0; tile < tiles; ++tile) {
      const int at = Q::header + 3 * (begin + tile);
      table[at] = expert;
      table[at + 1] = first + tile * T::rows;
      table[at + 2] = min(T::rows, count - tile * T::rows);
    }
    carry += __shfl_sync(0xffffffffu, prefix, 31);
  }
  if (lane < Q::header) table[lane] = lane == 0 ? carry : 0;
}

template<class C, class T>
__device__ __forceinline__ void issue(Shared<C, T>& s, int slot, int block, int expert,
    int column, uint64_t act_scale_bytes, const uint8_t* input, const uint8_t* weight,
    const uint8_t* weight_scales, const uint8_t* activation_scales) {
  constexpr int WR = tile_weight_rows<C, T>, TC = tile_cols<C, T>;
  auto& st = s.stage[slot];
  const int k0 = block * lay::StageK;
  for (int v = threadIdx.x; v < T::rows * 8; v += T::threads) {
    const int r = v >> 3, c = v & 7, source = s.sources[r], k = k0 + c * 16;
    const bool valid = source >= 0 && k < C::k;
    copy16(st.act + lay::act_store(r, c),
           input + (valid ? uint64_t(source) * C::k + k : 0), valid);
  }
  if constexpr (C::lanes) {
    // Whole tile rows per thread, lane chunk q into chunk q's slot; the
    // copying thread untangles them (`untangle`, the same assignment).
    for (int w = threadIdx.x; w < WR; w += T::threads) {
      const int col = column + w % TC;
      const bool valid = col < C::n;
      const uint64_t row = uint64_t(expert) * C::weight_rows + (w / TC) * C::n + col;
      const uint64_t at = (row / 64 * C::blocks + block) * 4096 + row % 64 * 64;
#pragma unroll
      for (int q = 0; q < 4; ++q)
        copy16(st.weight + lay::weight_store(w, q), weight + (valid ? at + q * 16 : 0), valid);
    }
  } else {
    for (int v = threadIdx.x; v < WR * 4; v += T::threads) {
      const int w = v >> 2, q = v & 3, col = column + w % TC, byte = k0 / 2 + q * 16;
      const bool valid = col < C::n && byte < C::k / 2;
      const uint64_t row = uint64_t(expert) * C::weight_rows + (w / TC) * C::n + col;
      if constexpr (C::tiles) {
        const uint64_t at = (row / 64 * C::blocks + block) * 4096 + row % 64 * 64 + q * 16;
        copy16(st.weight + lay::weight_store(w, q), weight + (valid ? at : 0), valid);
      } else {
        copy16(st.weight + lay::weight_store(w, q),
               weight + (valid ? row * (C::k / 2) + byte : 0), valid);
      }
    }
  }
  constexpr int SW = C::scale_words;
  constexpr uint64_t weight_scale_bytes = uint64_t(C::experts) * C::weight_rows * C::groups;
  for (int v = threadIdx.x; v < (T::rows + WR) * SW; v += T::threads) {
    const int r = v / SW, word = v % SW;
    if (r < T::rows) {
      const int source = s.sources[r];
      const uint64_t at = uint64_t(source < 0 ? 0 : source) * C::groups + k0 / lay::GroupK;
      copy_scale(&st.act_scale[r][word], activation_scales, at, word, act_scale_bytes, source >= 0);
    } else {
      const int w = r - T::rows, col = column + w % TC;
      const uint64_t row = uint64_t(expert) * C::weight_rows + (w / TC) * C::n + col;
      if constexpr (C::tiles) {
        const uint64_t at = (row / 64 * C::blocks + block) * 256 + row % 64 * 4;
        copy_scale(&st.weight_scale[w][word], weight_scales, at, word, weight_scale_bytes, col < C::n);
      } else {
        copy_scale(&st.weight_scale[w][word], weight_scales, row * C::groups + k0 / lay::GroupK,
                   word, weight_scale_bytes, col < C::n);
      }
    }
  }
}

// Restores the row-tiled order of the quad-lane tile rows this thread
// copied into stage `slot` (`issue`'s assignment), once its own copies have
// landed: word g of the four lane chunks, transposed as bytes, is row-tiled
// chunk g. No other thread reads those rows before the stage barrier.
template<class C, class T>
__device__ __forceinline__ void untangle(Shared<C, T>& s, int slot) {
  constexpr int WR = tile_weight_rows<C, T>;
  auto& st = s.stage[slot];
  for (int w = threadIdx.x; w < WR; w += T::threads) {
    uint4 lane[4];
#pragma unroll
    for (int q = 0; q < 4; ++q)
      lane[q] = *reinterpret_cast<const uint4*>(st.weight + lay::weight_store(w, q));
    uint4 chunk[4] = {
        make_uint4(lane[0].x, lane[1].x, lane[2].x, lane[3].x),
        make_uint4(lane[0].y, lane[1].y, lane[2].y, lane[3].y),
        make_uint4(lane[0].z, lane[1].z, lane[2].z, lane[3].z),
        make_uint4(lane[0].w, lane[1].w, lane[2].w, lane[3].w)};
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      lay::transpose_bytes(chunk[g].x, chunk[g].y, chunk[g].z, chunk[g].w);
      *reinterpret_cast<uint4*>(st.weight + lay::weight_store(w, g)) = chunk[g];
    }
  }
}

// The served inner loop's policy. The lab's tuning variants change one
// part of it: `lean` takes each MMA's scale byte by the instruction's byte
// selector and checks activation NaN codes in each row group's first
// column warp only (bitwise the served outputs and admission); the others
// are timing ablations whose outputs are not meaningful: no checks, no
// MMA, no global loads after the prologue, or MMAs on register operands
// with no operand loads. `nan = false` leaves the activation NaN check to
// `nan_scan`, which must run on the same input before the tiles; `acts =
// false` also leaves the activation scale admission to `admission_scan`.
struct Served {
  static constexpr bool lean = false, checks = true, mma = true, global = true, loads = true,
                        nan = true, acts = true;
};

// One MMA whose A and B scales are byte `q` of the given words, by the
// instruction's byte selectors; `q` folds once the group loop unrolls.
template<int Q>
__device__ __forceinline__ void byte_mma(ptx::D4& d, const ptx::A4& a, const ptx::B2& b,
                                         uint32_t scale_a, uint32_t scale_b) {
  ptx::mma_mxf8f6f4<ptx::Element::e2m1, ptx::Element::e4m3, Q, 0, Q, 0>(d, a, b, scale_a, scale_b);
}
__device__ __forceinline__ void byte_mma(int q, ptx::D4& d, const ptx::A4& a, const ptx::B2& b,
                                         uint32_t scale_a, uint32_t scale_b) {
  switch (q) {
    case 0: byte_mma<0>(d, a, b, scale_a, scale_b); break;
    case 1: byte_mma<1>(d, a, b, scale_a, scale_b); break;
    case 2: byte_mma<2>(d, a, b, scale_a, scale_b); break;
    default: byte_mma<3>(d, a, b, scale_a, scale_b); break;
  }
}

// The row's four group scales of the stage, from its copied words.
template<class C, int Words>
__device__ __forceinline__ uint32_t stage_scales(const uint32_t (&words)[Words], uint64_t row) {
  if constexpr (Words == 1) return words[0];
  else return lay::scale_word(words[0], words[1], int((row * C::groups) & 3));
}

// Whether any of a stage's `Groups` scale bytes leaves [lo, hi].
template<int Groups>
__device__ __forceinline__ bool outside(uint32_t scales, uint32_t lo, uint32_t hi) {
  bool out = false;
#pragma unroll
  for (int q = 0; q < Groups; ++q) {
    const uint32_t b = (scales >> (8 * q)) & 255;
    out |= b < lo || b > hi;
  }
  return out;
}

// One 128-K stage of `Groups` 32-K groups: per routed fragment, a chain
// from +0 over the groups, then one ordered F32 add into the accumulator.
// The exact engines' admission runs off the MMA path: its slow checks
// execute only when a scale leaves the range where every code is exact.
template<class C, class T, int Groups, bool Omit, class Tune = Served>
__device__ __forceinline__ void block(const Shared<C, T>& s, int slot, int expert, int column,
    int count, ptx::D4 (&sum)[2][T::fragments]) {
  constexpr int TC = tile_cols<C, T>, F = T::fragments;
  const auto& st = s.stage[slot];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int row_base = (warp / T::col_warps) * F * 8, col_warp = warp % T::col_warps;
  ptx::A4 a[2][Groups];
  uint32_t scale_a[2][Groups];
#pragma unroll
  for (int cf = 0; cf < 2; ++cf) {
    const int w = C::gate_up ? cf * TC + col_warp * 16 : col_warp * 32 + cf * 16;
    const int col = column + (C::gate_up ? col_warp * 16 : col_warp * 32 + cf * 16);
    const uint64_t row = uint64_t(expert) * C::weight_rows + (C::gate_up ? cf * C::n : 0) + col;
    if constexpr (!Tune::loads) {
#pragma unroll
      for (int q = 0; q < Groups; ++q) {
#pragma unroll
        for (int i = 0; i < 4; ++i) a[cf][q].v[i] = (lane * 0x01010101u + q + i) & 0x3c3c3c3cu;
        scale_a[cf][q] = 127;
      }
      continue;
    }
    const uint32_t s0 = stage_scales<C>(st.weight_scale[w + g], row + g);
    const uint32_t s1 = stage_scales<C>(st.weight_scale[w + g + 8], row + g + 8);
#pragma unroll
    for (int q = 0; q < Groups; ++q) {
      const uint32_t lo = lds32(st.weight + lay::weight_load(w + g, q, t));
      const uint32_t hi = lds32(st.weight + lay::weight_load(w + g + 8, q, t));
      lay::expand_e2m1x8(lo, a[cf][q].v[0], a[cf][q].v[2]);
      lay::expand_e2m1x8(hi, a[cf][q].v[1], a[cf][q].v[3]);
      // Lean keeps the stage's four scale bytes and selects one per MMA.
      scale_a[cf][q] = Tune::lean ? (t & 1 ? s1 : s0) : ((t & 1 ? s1 : s0) >> (8 * q)) & 255;
    }
    if (Tune::checks && (outside<Groups>(s0, 0, 252) || outside<Groups>(s1, 0, 252))) {
#pragma unroll 1
      for (int q = 0; q < Groups; ++q) {
        check_weights(lds32(st.weight + lay::weight_load(w + g, q, t)), (s0 >> (8 * q)) & 255);
        check_weights(lds32(st.weight + lay::weight_load(w + g + 8, q, t)), (s1 >> (8 * q)) & 255);
      }
    }
  }
  // Activation scales of each live fragment row, and the slow admission.
  uint32_t sb[F];
  bool slow = false;
#pragma unroll
  for (int f = 0; f < F; ++f) {
    const int r = row_base + f * 8 + g, source = s.sources[r];
    sb[f] = Tune::loads ? stage_scales<C>(st.act_scale[r], uint64_t(source < 0 ? 0 : source))
                        : 0x7f7f7f7fu;
    slow |= Tune::checks && Tune::acts && row_base + f * 8 < count && source >= 0 &&
            outside<Groups>(sb[f], 3, 246);
  }
  if (Tune::checks && Tune::acts && __any_sync(0xffffffffu, slow)) {
#pragma unroll 1
    for (int f = 0; f < F; ++f) {
      const int r = row_base + f * 8 + g, source = s.sources[r];
      if (row_base + f * 8 >= count || source < 0) continue;
      const uint32_t scales = stage_scales<C>(st.act_scale[r], uint64_t(source));
#pragma unroll 1
      for (int q = 0; q < Groups; ++q)
        check_activations(lds64(st.act + lay::act_load(r, q, t)), (scales >> (8 * q)) & 255);
    }
  }
  // Fragments run in pairs, four independent chains per warp; a pair's
  // second fragment may be padding, whose zero rows are discarded.
  static_assert(F % 2 == 0);
  uint32_t nan = 0;
  // Lean: one column warp of each row group checks the NaN codes, which
  // its loads cover for every live row and K of the stage.
  const bool check_nan = Tune::checks && Tune::nan && (!Tune::lean || col_warp == 0);
#pragma unroll
  for (int f0 = 0; f0 < F; f0 += 2) {
    if (row_base + f0 * 8 >= count) continue;  // warp-uniform: only padding remains
    ptx::D4 chain[2][2] = {};
#pragma unroll
    for (int q = 0; q < Groups; ++q) {
#pragma unroll
      for (int p = 0; p < 2; ++p) {
        const int f = f0 + p, r = row_base + f * 8 + g;
        const uint2 codes = Tune::loads ? lds64(st.act + lay::act_load(r, q, t))
                                        : make_uint2(lane * 0x01010101u + q, f + q);
        if (check_nan) nan |= lay::e4m3_nan(codes.x) | lay::e4m3_nan(codes.y);
        const ptx::B2 b{{codes.x, codes.y}};
        if constexpr (Tune::lean) {
#pragma unroll
          for (int cf = 0; cf < 2; ++cf)
            byte_mma(q, chain[p][cf], a[cf][q], b, scale_a[cf][q], sb[f]);
        } else if constexpr (Tune::mma) {
          const uint32_t scale_b = (sb[f] >> (8 * q)) & 255;
#pragma unroll
          for (int cf = 0; cf < 2; ++cf)
            ptx::mma_mxf8f6f4<ptx::Element::e2m1, ptx::Element::e4m3>(
                chain[p][cf], a[cf][q], b, scale_a[cf][q], scale_b);
        } else {
          // Keep the operands live without the tensor instruction.
          asm volatile("" :: "r"(codes.x), "r"(codes.y), "r"(sb[f]), "r"(scale_a[0][q]),
                       "r"(scale_a[1][q]));
#pragma unroll
          for (int cf = 0; cf < 2; ++cf)
            asm volatile("" :: "r"(a[cf][q].v[0]), "r"(a[cf][q].v[1]), "r"(a[cf][q].v[2]),
                         "r"(a[cf][q].v[3]));
        }
      }
    }
    if constexpr (!Omit) {
#pragma unroll
      for (int p = 0; p < 2; ++p)
#pragma unroll
        for (int cf = 0; cf < 2; ++cf)
#pragma unroll
          for (int i = 0; i < 4; ++i)
            sum[cf][f0 + p].v[i] = __fadd_rn(sum[cf][f0 + p].v[i], chain[p][cf].v[i]);
    }
  }
  if (nan) __trap();
}

// No quantizer in the gate/up epilogue; `native_ceil.cuh` supplies one.
struct NoCeil {
  static constexpr bool enabled = false;
};

// The persistent tile loop. Lab-only variants: `Omit` drops the first
// 128-K chain, a negative control the oracle checks must reject, and
// `Raw` also writes gate/up's F32 gate and up sums to `raw` (route rows of
// n gate then n up columns) for the oracle. `Ceil` (gate/up) also
// quantizes each tile's BF16 outputs, staged in shared memory.
template<class C, class T, bool Omit = false, bool Raw = false, class Ceil = NoCeil,
         class Tune = Served>
__device__ void compute(void* arena, const uint8_t* input, const uint8_t* weight,
    const uint8_t* weight_scales, const uint8_t* activation_scales, const int* sorted_rows,
    const int* padded_rows, const int* live_tokens, int* table, const float* route_weights,
    void* output, float* raw = nullptr, const Ceil& ceil = Ceil{}) {
  static_assert(!Raw || C::gate_up, "down's output is already its F32 sum");
  using Q = Queue<C, T>;
  constexpr int S = T::stages, TC = tile_cols<C, T>, Strips = strips<C, T>;
  auto& s = *static_cast<Shared<C, T>*>(arena);
  const int live = *live_tokens, padded = *padded_rows, tiles = table[0];
  if (blockDim.x != T::threads || live < 0 || live > C::capacity || tiles < 0 ||
      tiles > Q::slots || padded < 0 || padded % 64 ||
      padded > C::capacity * C::top_k + C::experts * 64) __trap();
  const uint64_t act_scale_bytes = uint64_t(live) * (C::gate_up ? 1 : C::top_k) * C::groups;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int row_base = (warp / T::col_warps) * T::fragments * 8, col_warp = warp % T::col_warps;
  const int items = tiles * Strips;
  for (;;) {
    __syncthreads();  // the previous item's stages and routes are consumed
    if (threadIdx.x == 0) s.item = atomicAdd(table + 1, 1);
    __syncthreads();
    const int item = s.item;
    if (item >= items) break;
    const int column = (item % Strips) * TC, at = Q::header + 3 * (item / Strips);
    const int expert = table[at], first = table[at + 1], count = table[at + 2];
    if (expert < 0 || expert >= C::experts || first < 0 || first % 64 || count <= 0 ||
        count > T::rows || count > live * C::top_k || first + count > padded) __trap();
    for (int r = threadIdx.x; r < T::rows; r += T::threads) {
      const int route = r < count ? sorted_rows[first + r] : -1;
      if (r < count && (route < 0 || route >= live * C::top_k)) __trap();
      s.routes[r] = route;
      s.sources[r] = route < 0 ? -1 : (C::gate_up ? route / C::top_k : route);
    }
    __syncthreads();
    ptx::D4 sum[2][T::fragments] = {};
#pragma unroll
    for (int b = 0; b < S - 1; ++b) {
      if (b < C::blocks)
        issue(s, b, b, expert, column, act_scale_bytes, input, weight, weight_scales,
              activation_scales);
      commit();
    }
#pragma unroll 1
    for (int b = 0; b < C::blocks; ++b) {
      wait<S - 2>();
      if constexpr (C::lanes) untangle(s, b % S);
      __syncthreads();
      // The slot of block b - 1, which every thread has finished reading.
      if (Tune::global && b + S - 1 < C::blocks)
        issue(s, (b + S - 1) % S, b + S - 1, expert, column, act_scale_bytes, input, weight,
              weight_scales, activation_scales);
      commit();  // empty groups keep wait_group's count exact at the tail
      const bool omit = Omit && b == 0;
      if (C::tail_groups == 0 || b + 1 < C::blocks) {
        if (omit) block<C, T, 4, true, Tune>(s, b % S, expert, column, count, sum);
        else block<C, T, 4, false, Tune>(s, b % S, expert, column, count, sum);
      } else {
        constexpr int G = C::tail_groups == 0 ? 4 : C::tail_groups;
        if (omit) block<C, T, G, true, Tune>(s, b % S, expert, column, count, sum);
        else block<C, T, G, false, Tune>(s, b % S, expert, column, count, sum);
      }
    }
    wait<0>();
    // The staged outputs reuse the operand stages, once every warp has
    // finished reading them.
    [[maybe_unused]] uint16_t* staged = reinterpret_cast<uint16_t*>(&s.stage[0]);
    if constexpr (Ceil::enabled) {
      static_assert(sizeof(s.stage) >= sizeof(uint16_t) * T::rows * TC);
      __syncthreads();
    }
#pragma unroll
    for (int f = 0; f < T::fragments; ++f) {
      if (row_base + f * 8 >= count) continue;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int r = row_base + f * 8 + lay::d_route(t, i);
        const int route = s.routes[r];
        if (route < 0) continue;
        if constexpr (C::gate_up) {
          const int col = column + col_warp * 16 + lay::d_weight_row(g, i);
          if (col >= C::n) continue;
          const uint16_t bits = convert::bf16_from_f32(
              chained::before_down<typename C::clamp>(sum[0][f].v[i], sum[1][f].v[i],
                                                       route_weights[route]));
          static_cast<uint16_t*>(output)[uint64_t(route) * C::n + col] = bits;
          if constexpr (Ceil::enabled) staged[r * TC + (col - column)] = bits;
          if constexpr (Raw) {
            raw[uint64_t(route) * 2 * C::n + col] = sum[0][f].v[i];
            raw[uint64_t(route) * 2 * C::n + C::n + col] = sum[1][f].v[i];
          }
        } else {
#pragma unroll
          for (int cf = 0; cf < 2; ++cf) {
            const int col = column + col_warp * 32 + cf * 16 + lay::d_weight_row(g, i);
            if (col < C::n)
              static_cast<float*>(output)[uint64_t(route) * C::n + col] = sum[cf][f].v[i];
          }
        }
      }
    }
    if constexpr (Ceil::enabled) {
      __syncthreads();
      ceil.template tile<C, T>(staged, s.routes, column);
    }
  }
}

// The lab's tuning variants of `Served`.
struct Lean : Served { static constexpr bool lean = true; };
struct NoChecks : Served { static constexpr bool checks = false; };
struct NoMma : Served { static constexpr bool mma = false; };
struct NoGlobal : Served { static constexpr bool global = false, checks = false; };
struct MmaOnly : Served { static constexpr bool global = false, checks = false, loads = false; };
struct NoNan : Served { static constexpr bool lean = true, nan = false; };
// The lean loop with the whole activation admission in `admission_scan`.
struct Scanned : Served { static constexpr bool lean = true, nan = false, acts = false; };

// Traps on any E4M3 NaN code in the live input rows (tokens for gate/up,
// routes for down): every code the tiles read, so a `nan = false` tile
// behind it refuses exactly the inputs the served tile refuses.
template<class C>
__device__ void nan_scan(const uint8_t* input, const int* live_tokens) {
  static_assert(C::k % 16 == 0);
  const int live = *live_tokens;
  if (live < 0 || live > C::capacity) __trap();
  const uint64_t words = uint64_t(live) * (C::gate_up ? 1 : C::top_k) * (C::k / 16);
  const uint4* codes = reinterpret_cast<const uint4*>(input);
  uint32_t nan = 0;
  for (uint64_t i = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x; i < words;
       i += uint64_t(gridDim.x) * blockDim.x) {
    const uint4 v = codes[i];
    nan |= lay::e4m3_nan(v.x) | lay::e4m3_nan(v.y) | lay::e4m3_nan(v.z) | lay::e4m3_nan(v.w);
  }
  if (nan) __trap();
}

// The tiles' whole activation admission over the live input rows (tokens
// for gate/up, routes for down), one thread per row and 32-K group: any
// E4M3 NaN code traps, and a group whose scale leaves 3..246 runs the
// exact engine's product check (`chained::scaled_e4m3x2`) on its 32 codes,
// as `check_activations` does in the tile. Every live row is routed (all
// experts are local at the native geometries), so the tiles read exactly
// these rows and groups; a `Scanned` tile behind this pass refuses exactly
// the inputs the served tile refuses.
template<class C>
__device__ void admission_scan(const uint8_t* input, const uint8_t* activation_scales,
                               const int* live_tokens) {
  static_assert(C::k % lay::GroupK == 0 && C::k / lay::GroupK == C::groups);
  const int live = *live_tokens;
  if (live < 0 || live > C::capacity) __trap();
  const uint64_t items = uint64_t(live) * (C::gate_up ? 1 : C::top_k) * C::groups;
  uint32_t nan = 0;
  for (uint64_t i = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x; i < items;
       i += uint64_t(gridDim.x) * blockDim.x) {
    const uint4* codes = reinterpret_cast<const uint4*>(input + i * lay::GroupK);
    const uint4 v0 = codes[0], v1 = codes[1];
    nan |= lay::e4m3_nan(v0.x) | lay::e4m3_nan(v0.y) | lay::e4m3_nan(v0.z) | lay::e4m3_nan(v0.w) |
           lay::e4m3_nan(v1.x) | lay::e4m3_nan(v1.y) | lay::e4m3_nan(v1.z) | lay::e4m3_nan(v1.w);
    const uint32_t scale = activation_scales[i];
    if (scale < 3 || scale > 246) {
      check_activations(make_uint2(v0.x, v0.y), scale);
      check_activations(make_uint2(v0.z, v0.w), scale);
      check_activations(make_uint2(v1.x, v1.y), scale);
      check_activations(make_uint2(v1.z, v1.w), scale);
    }
  }
  if (nan) __trap();
}

// Lab probe of one instruction from +0: lane l passes A registers
// a[4l..4l+3], B registers b[2l..2l+1] and scales scales[2l] (A) and
// scales[2l+1] (B), and writes its accumulators to d[4l..4l+3].
__device__ void probe(const uint32_t* a, const uint32_t* b, const uint32_t* scales, float* d) {
  const int lane = threadIdx.x;
  if (blockDim.x != 32 || gridDim.x != 1) __trap();
  const ptx::A4 af{{a[4 * lane], a[4 * lane + 1], a[4 * lane + 2], a[4 * lane + 3]}};
  const ptx::B2 bf{{b[2 * lane], b[2 * lane + 1]}};
  ptx::D4 acc{};
  ptx::mma_mxf8f6f4<ptx::Element::e2m1, ptx::Element::e4m3>(
      acc, af, bf, scales[2 * lane], scales[2 * lane + 1]);
#pragma unroll
  for (int i = 0; i < 4; ++i) d[4 * lane + i] = acc.v[i];
}

} // namespace chop::moe::native_grouped
