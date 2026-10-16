#pragma once

#include "grouped.cuh"

// Ring decode of the grouped-MoE template over the packed row-major layout
// of NVFP4 and MXFP8 expert banks (README "Packed NVFP4 and MXFP8 ring
// decode"). The bank keeps the layout the packed engine (`moe.cuh`) reads:
// codes row-major, K/2 bytes per row (even K in the low nibble) or K bytes,
// and UE4M3 per K16 or UE8M0 per K32 local scales in 16-row by 8-block
// 128-byte tiles. The persistent images need no repacking.
//
// A CTA owns one (active expert, output tile) item over the full K; its
// `strips` 16-column strips each have one warp, or for gate/up with
// `split_parts` a gate warp and an up warp. A warp streams its strip's K64
// steps (codes, local scales and above one token the step's routed rows)
// through its own cp.async ring `depth` steps deep; at one token the CTA
// stages its single row once. K is never split: every accumulator adds its
// K16 MMAs in ascending K from zero, as the packed engine does, so outputs
// are bitwise the engine's. An expert with more routed rows than the
// bucket's tokens (never in routes the models write) takes further passes.
//
// Operands are exact BF16: NVFP4 is BF16(E2M1) times BF16(UE4M3), a product
// of at most six significant bits; MXFP8 uses the engine's scaled E4M3
// conversion, which traps on an inexact product. NVFP4 F32 globals
// multiply each full F32 sum with `__fmul_rn` before SwiGLU or the route
// weight (`ExactLocalThenGlobal`), and the epilogue rounds to BF16 once.
//
// PDL follows `grouped_ring.cuh`: gate/up waits before it launches its
// dependents, so down reads the route metadata and starts its weight copies
// before its own wait.
namespace chop::moe::grouped {

struct PackedRingSchedule {
  int strips;           // 16-column output strips per CTA
  int depth;            // ring slots per warp, in K64 steps
  bool split_parts;     // gate/up: gate and up of a strip on separate warps
  bool expert_globals;  // NVFP4: F32 global per expert and projection, else one scalar
  int prefetch = 0;     // L2 prefetch size of each weight copy in bytes: 0, 64, 128 or 256
};

// A strip's K64 step: 16 rows of codes, then their local scales. NVFP4 has
// four UE4M3 blocks per row (64 bytes), MXFP8 two UE8M0 blocks (32 bytes).
template<Weights W>
__host__ __device__ constexpr int packed_code_bytes() {
  return W == Weights::nvfp4 ? 512 : 1024;
}

template<Weights W>
__host__ __device__ constexpr int packed_part_bytes() {
  return packed_code_bytes<W>() + (W == Weights::nvfp4 ? 64 : 32);
}

template<Weights W>
__host__ __device__ constexpr int packed_slot_bytes(int parts, int tokens_capacity) {
  return parts * packed_part_bytes<W>()
       + (tokens_capacity > 1 ? tokens_capacity * kFragmentRowBytes : 0);
}

// A 16-byte copy that also asks L2 to fetch the `Prefetch`-byte block
// around it: the packed rows are read 32 or 64 bytes per K64 step, so the
// following steps of the row then come from L2. Only data movement changes.
template<int Prefetch>
__device__ __forceinline__ void cp_async_16_prefetch(void* shared, const void* global) {
  static_assert(Prefetch == 0 || Prefetch == 64 || Prefetch == 128 || Prefetch == 256);
  if constexpr (Prefetch == 0)
    cp_async_16(shared, global);
  else if constexpr (Prefetch == 64)
    asm volatile("cp.async.cg.shared.global.L2::64B [%0], [%1], 16;"
                 :: "r"(ptx::shared_address(shared)), "l"(global) : "memory");
  else if constexpr (Prefetch == 128)
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;"
                 :: "r"(ptx::shared_address(shared)), "l"(global) : "memory");
  else
    asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;"
                 :: "r"(ptx::shared_address(shared)), "l"(global) : "memory");
}

__device__ __forceinline__ void cp_async_8(void* shared, const void* global) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;"
               :: "r"(ptx::shared_address(shared)), "l"(global) : "memory");
}

// The low byte's two E2M1 codes as BF16x2, even K in the low half.
__device__ __forceinline__ uint32_t e2m1x2_bf16x2(uint32_t codes) {
  uint32_t result;
  asm("{ .reg .b8 c, unused; .reg .b16 w; cvt.u16.u32 w, %1; mov.b16 {c, unused}, w; "
      "cvt.rn.bf16x2.e2m1x2 %0, c; }"
      : "=r"(result) : "r"(codes));
  return result;
}

// Both halves of a UE4M3 local scale as BF16x2. A UE4M3 code below 0x80
// is the positive E4M3 value with that code; admission rejects the others.
__device__ __forceinline__ uint32_t ue4m3_pair_bf16x2(uint32_t pair) {
  uint32_t result;
  asm("{ .reg .b16 w; cvt.u16.u32 w, %1; cvt.rn.bf16x2.e4m3x2 %0, w; }"
      : "=r"(result) : "r"(pair));
  return result;
}

// An exact product: both factors are BF16 and it has at most six
// significant bits, well inside the BF16 exponent range.
__device__ __forceinline__ uint32_t mul_bf16x2(uint32_t a, uint32_t b) {
  uint32_t result;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(result) : "r"(a), "r"(b));
  return result;
}

template<Weights W, Shape S, bool GateUp, int TokensCapacity, PackedRingSchedule R>
__device__ void packed_ring_decode(
    const uint8_t* activation, const uint8_t* weight_bytes, const uint8_t* weight_scales,
    const float* weight_f32_scales, uint16_t* output, const float* topk_weights,
    const int32_t* expert_offsets, const int32_t* sorted_rows, const int32_t* num_padded_rows,
    const int32_t* live_tokens, const int32_t* active_experts) {
  static_assert(W == Weights::nvfp4 || W == Weights::mxfp8,
                "the packed ring reads the NVFP4 and MXFP8 packed layouts");
  constexpr bool Nv = W == Weights::nvfp4;
  constexpr int N = GateUp ? 2 * S.intermediate : S.hidden;
  constexpr int K = GateUp ? S.hidden : S.intermediate;
  constexpr int OutN = GateUp ? N / 2 : N;
  constexpr int Experts = S.experts;
  constexpr int TopK = S.top_k;
  constexpr bool Split = R.split_parts;
  static_assert(!Split || GateUp, "only gate/up has two parts to split");
  static_assert(!R.expert_globals || Nv, "F32 globals are NVFP4's");
  constexpr int Strips = R.strips;
  constexpr int Warps = Strips * (Split ? 2 : 1);
  // Projection parts each warp accumulates.
  constexpr int Parts = GateUp && !Split ? 2 : 1;
  constexpr int Ntile = Strips * 16;
  static_assert(K % 64 == 0 && OutN % Ntile == 0);
  static_assert(TokensCapacity >= 1 && TokensCapacity <= 32);
  constexpr int Ntiles = OutN / Ntile;
  constexpr int Ktiles = K / 64;
  constexpr int RowBytes = Nv ? K / 2 : K;
  constexpr int ScaleTiles = (K / (Nv ? 16 : 32) + 7) / 8;
  constexpr int CodeBytes = packed_code_bytes<W>();
  constexpr int PartBytes = packed_part_bytes<W>();
  constexpr int Rows = TokensCapacity;
  constexpr int TokenTiles = (Rows + 7) / 8;
  constexpr int SlotBytes = packed_slot_bytes<W>(Parts, TokensCapacity);
  constexpr int Depth = R.depth;
  constexpr int MaxActive = TopK * TokensCapacity < Experts ? TopK * TokensCapacity : Experts;
  constexpr int RowCopies = (Rows * 8 + 31) / 32;
  static_assert(Depth >= 1 && Depth <= Ktiles);
  static_assert(SlotBytes % 16 == 0);
  // Up warps leave their sums in their own ring for the gate warps.
  static_assert(!Split || Depth * SlotBytes >= 32 * TokenTiles * 16);
  extern __shared__ __align__(16) uint8_t ring_shared[];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int strip = warp % Strips;
  const int first_part = Split ? warp / Strips : 0;
  uint8_t* ring = ring_shared + warp * Depth * SlotBytes;
  if constexpr (GateUp) {
    ptx::griddep_wait();
    if (threadIdx.x == 0) ptx::griddep_launch_dependents();
  } else {
    if (threadIdx.x == 0) ptx::griddep_launch_dependents();
  }
  const int active = min(*active_experts, MaxActive);
  const int work = blockIdx.x;
  if (active <= 0 || work >= active * Ntiles) return;
  const int expert = active_experts[1 + work / Ntiles];
  if (expert < 0 || expert >= Experts) return;
  const int column = work % Ntiles * Ntile + strip * 16;

  // Each lane copies 16-byte code chunks of the strip's rows: NVFP4 row
  // lane / 2, half lane % 2; MXFP8 rows lane / 4 and lane / 4 + 8, quarter
  // lane % 4. Chunks are XOR-swizzled by row so the eight rows one operand
  // load reads fall in distinct banks. Lanes 0-3 copy the scale words of
  // rows lane, lane + 4, lane + 8 and lane + 12 (one word per K block).
  constexpr int CodeCopies = Nv ? 1 : 2;
  const int code_row = Nv ? lane >> 1 : lane >> 2;
  const int code_chunk = Nv ? lane & 1 : lane & 3;
  const int code_swizzle = Nv ? (code_row >> 2) & 1 : (code_row >> 1) & 3;
  const int code_row_bytes = Nv ? 32 : 64;
  const int code_target = code_row * code_row_bytes + ((code_chunk ^ code_swizzle) << 4);
  const uint8_t* codes[Parts];
  const uint8_t* scale_tiles[Parts];
#pragma unroll
  for (int part = 0; part < Parts; ++part) {
    const int row0 = expert * N + column + (first_part + part) * (N / 2);
    codes[part] = weight_bytes + uint64_t(row0 + code_row) * RowBytes + code_chunk * 16;
    scale_tiles[part] = weight_scales + uint64_t(row0 / 16) * ScaleTiles * 128 + lane * 32;
  }
  auto issue_weights = [&](int kt) {
#pragma unroll
    for (int part = 0; part < Parts; ++part) {
      uint8_t* slot = ring + (kt % Depth) * SlotBytes + part * PartBytes;
#pragma unroll
      for (int c = 0; c < CodeCopies; ++c)
        cp_async_16_prefetch<R.prefetch>(slot + code_target + c * 8 * code_row_bytes,
                    codes[part] + uint64_t(c) * 8 * RowBytes + kt * code_row_bytes);
      if (lane < 4) {
        if constexpr (Nv)
          cp_async_16_prefetch<(R.prefetch > 128 ? 128 : R.prefetch)>(slot + CodeBytes + lane * 16,
                      scale_tiles[part] + (kt >> 1) * 128 + (kt & 1) * 16);
        else
          cp_async_8(slot + CodeBytes + lane * 8,
                     scale_tiles[part] + (kt >> 2) * 128 + (kt & 3) * 8);
      }
    }
  };
  auto issue_first_weights = [&] {
#pragma unroll
    for (int kt = 0; kt < Depth; ++kt) {
      issue_weights(kt);
      cp_async_commit();
    }
  };
  issue_first_weights();

  const uint8_t* row_source[RowCopies];
  auto issue_rows = [&](int kt) {
    uint8_t* rows = ring + (kt % Depth) * SlotBytes + Parts * PartBytes;
#pragma unroll
    for (int c = 0; c < RowCopies; ++c) {
      const int chunk = lane + 32 * c;
      if (chunk < Rows * 8 && row_source[c] != nullptr)
        cp_async_16(rows + chunk / 8 * kFragmentRowBytes + chunk % 8 * 16,
                    row_source[c] + kt * 128);
    }
  };
  // At one token the CTA stages its row once per pass, after the rings.
  constexpr bool OneRow = TokensCapacity == 1;
  uint16_t* cta_row = reinterpret_cast<uint16_t*>(ring_shared + Warps * Depth * SlotBytes);
  constexpr int RowChunks = K / 8;
  constexpr int ChunksPerThread = (RowChunks + Warps * 32 - 1) / (Warps * 32);

  // The expert's routed rows: slot j of a pass is its (base + j)-th slot.
  const int first = expert_offsets[expert];
  const int end = min(expert_offsets[expert + 1], *num_padded_rows);
  int routed[Rows];
  auto load_routes = [&](int base) {
#pragma unroll
    for (int j = 0; j < Rows; ++j) routed[j] = base + j < end ? sorted_rows[base + j] : -1;
  };
  load_routes(first);
  if constexpr (!GateUp) ptx::griddep_wait();
  const int routes = *live_tokens * TopK;
  float gate_global = 1.f, up_global = 1.f;
  if constexpr (Nv) {
    gate_global = weight_f32_scales[R.expert_globals ? expert * (GateUp ? 2 : 1) : 0];
    if constexpr (GateUp) up_global = weight_f32_scales[R.expert_globals ? expert * 2 + 1 : 0];
  }

  // The lane's operand rows g and g + 8 and its K pair t.
  const int g = lane >> 2;
  const int t = lane & 3;
  const int g_swizzle = Nv ? (g >> 2) & 1 : (g >> 1) & 3;
  // NVFP4: byte t of a word holds the K pair (2t, 2t + 1) of its eight K.
  // MXFP8: bytes 2t and 2t + 1 of a 16-byte chunk's first or second half.
  const uint32_t code_selector = Nv ? uint32_t(t) : uint32_t(2 * t) | uint32_t(2 * t + 1) << 4;
  // Scale bytes of rows g and g + 8 in their word.
  const int scale_byte = g >> 2;
  const uint32_t scale_selector[2] = {uint32_t(scale_byte) * 0x11, uint32_t(scale_byte + 2) * 0x11};

  // Routes the model writes give an expert at most one row per token, so
  // one pass covers it. Any further slots take more passes, each streaming
  // the strip's weights again; every slot of the expert's range is
  // computed as the packed engine computes it, whatever the metadata.
  for (int base = first; base < end; base += Rows) {
    if (base != first) {
      // The previous pass's staged row and up sums have been read, and its
      // copies into the rings have landed.
      cp_async_wait<0>();
      __syncthreads();
      load_routes(base);
    }
    bool any = false;
#pragma unroll
    for (int j = 0; j < Rows; ++j) {
      if (routed[j] >= routes) routed[j] = -1;
      any |= routed[j] >= 0;
    }
    if (!any) continue;
    if (base != first) issue_first_weights();
    if constexpr (OneRow) {
      const int row = GateUp ? routed[0] / TopK : routed[0];
      uint4 row_chunks[ChunksPerThread];
#pragma unroll
      for (int i = 0; i < ChunksPerThread; ++i) {
        const int chunk = threadIdx.x + i * Warps * 32;
        if (chunk < RowChunks)
          row_chunks[i] = *reinterpret_cast<const uint4*>(activation + uint64_t(row) * K * 2 + chunk * 16);
      }
#pragma unroll
      for (int i = 0; i < ChunksPerThread; ++i) {
        const int chunk = threadIdx.x + i * Warps * 32;
        if (chunk < RowChunks) reinterpret_cast<uint4*>(cta_row)[chunk] = row_chunks[i];
      }
      __syncthreads();
    } else {
#pragma unroll
      for (int c = 0; c < RowCopies; ++c) {
        // The chunk's routed row, reloaded so that the row array stays in
        // registers.
        const int chunk = lane + 32 * c;
        int row = chunk / 8 < Rows && base + chunk / 8 < end ? sorted_rows[base + chunk / 8] : -1;
        if (row >= routes) row = -1;
        if (GateUp && row >= 0) row /= TopK;
        row_source[c] = row >= 0 ? activation + uint64_t(row) * K * 2 + (chunk % 8) * 16 : nullptr;
      }
      // One group per step, so step kt's rows complete Depth - 1 groups
      // before the newest, like its weights.
#pragma unroll
      for (int kt = 0; kt < Depth; ++kt) {
        issue_rows(kt);
        cp_async_commit();
      }
    }

    // Token tiles without a live row are skipped.
    int tiles = 0;
#pragma unroll
    for (int p = 0; p < TokenTiles; ++p) {
      bool live = false;
#pragma unroll
      for (int j = 8 * p; j < 8 * p + 8 && j < Rows; ++j) live |= routed[j] >= 0;
      if (live) tiles = p + 1;
    }
    ptx::D4 accum[Parts][TokenTiles];
#pragma unroll
    for (int part = 0; part < Parts; ++part)
#pragma unroll
      for (int p = 0; p < TokenTiles; ++p)
#pragma unroll
        for (int i = 0; i < 4; ++i) accum[part][p].v[i] = 0.f;
#pragma unroll 1
    for (int kt = 0; kt < Ktiles; ++kt) {
      cp_async_wait<Depth - 1>();
      __syncwarp();
      const uint8_t* slot = ring + (kt % Depth) * SlotBytes;
      uint32_t b_words[TokenTiles][8];
#pragma unroll
      for (int p = 0; p < TokenTiles; ++p) {
        if (p >= tiles) break;
        const int b_row = 8 * p + g < Rows ? 8 * p + g : 0;
        const uint8_t* row = OneRow
            ? reinterpret_cast<const uint8_t*>(cta_row + kt * 64) + t * 4
            : slot + Parts * PartBytes + b_row * kFragmentRowBytes + t * 4;
#pragma unroll
        for (int i = 0; i < 8; ++i)
          b_words[p][i] = *reinterpret_cast<const uint32_t*>(row + i * 16);
      }
#pragma unroll
      for (int step = 0; step < 4; ++step) {
        ptx::A4 a[Parts];
#pragma unroll
        for (int part = 0; part < Parts; ++part) {
          const uint8_t* codes_in = slot + part * PartBytes;
          const uint8_t* scales_in = codes_in + CodeBytes + (g & 3) * (Nv ? 16 : 8);
#pragma unroll
          for (int half = 0; half < 2; ++half) {
            // Row g + 8 * half; its code chunks start 8 rows further on.
            const uint8_t* row = codes_in + (g + 8 * half) * code_row_bytes;
            if constexpr (Nv) {
              // Step `step` is the row's bytes [8 step, 8 step + 8): chunk
              // step / 2, half step % 2.
              const uint2 words = *reinterpret_cast<const uint2*>(
                  row + (((step >> 1) ^ g_swizzle) << 4) + (step & 1) * 8);
              const uint32_t scale_word =
                  *reinterpret_cast<const uint32_t*>(scales_in + step * 4);
              const uint32_t scale =
                  ue4m3_pair_bf16x2(__byte_perm(scale_word, 0, scale_selector[half]));
              a[part].v[half] =
                  mul_bf16x2(e2m1x2_bf16x2(__byte_perm(words.x, 0, code_selector)), scale);
              a[part].v[2 + half] =
                  mul_bf16x2(e2m1x2_bf16x2(__byte_perm(words.y, 0, code_selector)), scale);
            } else {
              // Step `step` is the row's chunk `step`.
              const uint4 words = *reinterpret_cast<const uint4*>(row + ((step ^ g_swizzle) << 4));
              const uint32_t scale_word =
                  *reinterpret_cast<const uint32_t*>(scales_in + (step >> 1) * 4);
              const uint8_t scale = uint8_t(__byte_perm(scale_word, 0, scale_selector[half]));
              a[part].v[half] = scaled_mxfp8_bf16x2(
                  uint16_t(__byte_perm(words.x, words.y, code_selector)), scale);
              a[part].v[2 + half] = scaled_mxfp8_bf16x2(
                  uint16_t(__byte_perm(words.z, words.w, code_selector)), scale);
            }
          }
        }
#pragma unroll
        for (int p = 0; p < TokenTiles; ++p) {
          if (p >= tiles) break;
          const ptx::B2 b{{b_words[p][2 * step], b_words[p][2 * step + 1]}};
#pragma unroll
          for (int part = 0; part < Parts; ++part) ptx::mma_bf16(accum[part][p], a[part], b);
        }
      }
      // Every lane has read the slot before any lane refills it.
      __syncwarp();
      if (kt + Depth < Ktiles) {
        issue_weights(kt + Depth);
        if constexpr (!OneRow) issue_rows(kt + Depth);
      }
      cp_async_commit();
    }

    // The full F32 sums of each part: gate/up with split parts takes the up
    // sums from the strip's up warp, which writes no output.
    ptx::D4 up[GateUp ? TokenTiles : 1];
    if constexpr (Split) {
      // Every copy into the ring has landed, so each up warp reuses its own
      // ring for its sums.
      cp_async_wait<0>();
      float4* sums = reinterpret_cast<float4*>(ring) + lane * TokenTiles;
      if (first_part == 1) {
#pragma unroll
        for (int p = 0; p < TokenTiles; ++p)
          sums[p] = make_float4(accum[0][p].v[0], accum[0][p].v[1], accum[0][p].v[2], accum[0][p].v[3]);
      }
      __syncthreads();
      if (first_part == 1) continue;
      const float4* in = reinterpret_cast<const float4*>(
          ring_shared + (warp + Strips) * Depth * SlotBytes) + lane * TokenTiles;
#pragma unroll
      for (int p = 0; p < TokenTiles; ++p) {
        const float4 value = in[p];
        up[p] = ptx::D4{{value.x, value.y, value.z, value.w}};
      }
    } else if constexpr (GateUp) {
#pragma unroll
      for (int p = 0; p < TokenTiles; ++p) up[p] = accum[Parts - 1][p];
    }
    // C column j of token tile p holds routed row 8p + j: c0 and c2 are
    // column 2t, c1 and c3 column 2t + 1, rows g and g + 8.
#pragma unroll
    for (int p = 0; p < TokenTiles; ++p) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int j = 8 * p + t * 2 + (i & 1);
        int row = -1;
#pragma unroll
        for (int r = 0; r < Rows; ++r)
          if (r == j) row = routed[r];
        if (row < 0) continue;
        float result = accum[0][p].v[i];
        if constexpr (Nv) result = __fmul_rn(result, gate_global);
        if constexpr (GateUp) {
          float up_value = up[p].v[i];
          if constexpr (Nv) up_value = __fmul_rn(up_value, up_global);
          result = result / (1.0f + expf(-result)) * up_value;
        } else {
          result *= topk_weights[row];
        }
        output[uint64_t(row) * OutN + column + g + 8 * (i / 2)] = convert::bf16_from_f32(result);
      }
    }
  }
}

// ---------------------------------------------------------------------
// Staged packed decode at one token (`packed_staged_decode`). A CTA owns one
// (active expert, 16-column strip) item. `producers` warps each stream the
// strip's K64 steps kt with kt % producers == their index through their own
// cp.async ring, build the step's exact BF16 A fragments for every part
// (the ring's conversions) and store them, lane-major, in a ring of
// `stages` converted steps. Warp 0 consumes the converted steps in
// ascending K and issues every MMA: each accumulator adds its K16 products
// in ascending K from zero, as the ring and the packed engine do, so the
// outputs are bitwise theirs. The split moves the operand conversion off
// the MMA chain, which bounds a one-token strip.

struct PackedStagedSchedule {
  int producers;        // converting warps per CTA
  int depth;            // producer ring slots, in K64 steps
  int stages;           // converted-step ring slots
  bool expert_globals;  // NVFP4: F32 global per expert and projection, else one scalar
};

// One K64 step's A fragments of one part: 4 K16 steps x 32 lanes x 16 bytes.
constexpr int kStagedPartBytes = 4 * 32 * 16;

template<Weights W>
__device__ __forceinline__ void packed_fragments(
    const uint8_t* codes_in, int g, int g_swizzle, uint32_t code_selector,
    const uint32_t (&scale_selector)[2], ptx::A4 (&a)[4]) {
  constexpr bool Nv = W == Weights::nvfp4;
  constexpr int CodeBytes = packed_code_bytes<W>();
  constexpr int code_row_bytes = Nv ? 32 : 64;
  const uint8_t* scales_in = codes_in + CodeBytes + (g & 3) * (Nv ? 16 : 8);
#pragma unroll
  for (int step = 0; step < 4; ++step) {
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const uint8_t* row = codes_in + (g + 8 * half) * code_row_bytes;
      if constexpr (Nv) {
        const uint2 words = *reinterpret_cast<const uint2*>(
            row + (((step >> 1) ^ g_swizzle) << 4) + (step & 1) * 8);
        const uint32_t scale_word = *reinterpret_cast<const uint32_t*>(scales_in + step * 4);
        const uint32_t scale = ue4m3_pair_bf16x2(__byte_perm(scale_word, 0, scale_selector[half]));
        a[step].v[half] = mul_bf16x2(e2m1x2_bf16x2(__byte_perm(words.x, 0, code_selector)), scale);
        a[step].v[2 + half] = mul_bf16x2(e2m1x2_bf16x2(__byte_perm(words.y, 0, code_selector)), scale);
      } else {
        const uint4 words = *reinterpret_cast<const uint4*>(row + ((step ^ g_swizzle) << 4));
        const uint32_t scale_word = *reinterpret_cast<const uint32_t*>(scales_in + (step >> 1) * 4);
        const uint8_t scale = uint8_t(__byte_perm(scale_word, 0, scale_selector[half]));
        a[step].v[half] = scaled_mxfp8_bf16x2(
            uint16_t(__byte_perm(words.x, words.y, code_selector)), scale);
        a[step].v[2 + half] = scaled_mxfp8_bf16x2(
            uint16_t(__byte_perm(words.z, words.w, code_selector)), scale);
      }
    }
  }
}

template<Weights W, Shape S, bool GateUp, PackedStagedSchedule R>
__device__ void packed_staged_decode(
    const uint8_t* activation, const uint8_t* weight_bytes, const uint8_t* weight_scales,
    const float* weight_f32_scales, uint16_t* output, const float* topk_weights,
    const int32_t* expert_offsets, const int32_t* sorted_rows, const int32_t* num_padded_rows,
    const int32_t* live_tokens, const int32_t* active_experts) {
  static_assert(W == Weights::nvfp4 || W == Weights::mxfp8,
                "the staged schedule reads the NVFP4 and MXFP8 packed layouts");
  static_assert(!R.expert_globals || W == Weights::nvfp4, "F32 globals are NVFP4's");
  constexpr bool Nv = W == Weights::nvfp4;
  constexpr int N = GateUp ? 2 * S.intermediate : S.hidden;
  constexpr int K = GateUp ? S.hidden : S.intermediate;
  constexpr int OutN = GateUp ? N / 2 : N;
  constexpr int Experts = S.experts;
  constexpr int TopK = S.top_k;
  constexpr int Parts = GateUp ? 2 : 1;
  constexpr int Producers = R.producers;
  constexpr int Warps = 1 + Producers;
  constexpr int Depth = R.depth;
  constexpr int Stages = R.stages;
  static_assert(K % 64 == 0 && OutN % 16 == 0);
  constexpr int Ntiles = OutN / 16;
  constexpr int Ktiles = K / 64;
  constexpr int RowBytes = Nv ? K / 2 : K;
  constexpr int ScaleTiles = (K / (Nv ? 16 : 32) + 7) / 8;
  constexpr int CodeBytes = packed_code_bytes<W>();
  constexpr int PartBytes = packed_part_bytes<W>();
  constexpr int SlotBytes = Parts * PartBytes;
  // A producer handles ceil(Ktiles / Producers) steps at most.
  constexpr int ProducerSteps = (Ktiles + Producers - 1) / Producers;
  static_assert(Producers >= 1 && Warps <= 16);
  static_assert(Depth >= 1 && Depth <= ProducerSteps);
  static_assert(Stages >= 1 && Stages <= Ktiles);
  // Within a pass each converted slot then has one producer, which fills it
  // in step order, so no producer can wait on a phase parity that repeats.
  static_assert(Stages % Producers == 0);
  constexpr int MaxActive = TopK < Experts ? TopK : Experts;
  extern __shared__ __align__(16) uint8_t ring_shared[];
  // Layout: producer rings, converted steps, the staged row, barriers.
  uint8_t* converted = ring_shared + Producers * Depth * SlotBytes;
  uint16_t* cta_row =
      reinterpret_cast<uint16_t*>(converted + Stages * Parts * kStagedPartBytes);
  uint64_t* full = reinterpret_cast<uint64_t*>(cta_row + K);
  uint64_t* empty = full + Stages;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if constexpr (GateUp) {
    ptx::griddep_wait();
    if (threadIdx.x == 0) ptx::griddep_launch_dependents();
  } else {
    if (threadIdx.x == 0) ptx::griddep_launch_dependents();
  }
  const int active = min(*active_experts, MaxActive);
  const int work = blockIdx.x;
  if (active <= 0 || work >= active * Ntiles) return;
  const int expert = active_experts[1 + work / Ntiles];
  if (expert < 0 || expert >= Experts) return;
  const int column = work % Ntiles * 16;
  if (threadIdx.x == 0) {
    for (int s = 0; s < Stages; ++s) {
      ptx::mbarrier_init(&full[s], 32);
      ptx::mbarrier_init(&empty[s], 32);
    }
    ptx::fence_mbarrier_init();
  }

  // Producer p's code and scale sources, as the ring's.
  const int producer = warp - 1;
  const int code_row = Nv ? lane >> 1 : lane >> 2;
  const int code_chunk = Nv ? lane & 1 : lane & 3;
  const int code_swizzle = Nv ? (code_row >> 2) & 1 : (code_row >> 1) & 3;
  const int code_row_bytes = Nv ? 32 : 64;
  const int code_target = code_row * code_row_bytes + ((code_chunk ^ code_swizzle) << 4);
  constexpr int CodeCopies = Nv ? 1 : 2;
  const uint8_t* codes[Parts];
  const uint8_t* scale_tiles[Parts];
#pragma unroll
  for (int part = 0; part < Parts; ++part) {
    const int row0 = expert * N + column + part * (N / 2);
    codes[part] = weight_bytes + uint64_t(row0 + code_row) * RowBytes + code_chunk * 16;
    scale_tiles[part] = weight_scales + uint64_t(row0 / 16) * ScaleTiles * 128 + lane * 32;
  }
  uint8_t* ring = ring_shared + (producer < 0 ? 0 : producer) * Depth * SlotBytes;
  // Producer step j is K64 step producer + j * Producers.
  auto issue = [&](int j) {
    const int kt = producer + j * Producers;
    if (kt < Ktiles) {
#pragma unroll
      for (int part = 0; part < Parts; ++part) {
        uint8_t* slot = ring + (j % Depth) * SlotBytes + part * PartBytes;
#pragma unroll
        for (int c = 0; c < CodeCopies; ++c)
          cp_async_16(slot + code_target + c * 8 * code_row_bytes,
                      codes[part] + uint64_t(c) * 8 * RowBytes + kt * code_row_bytes);
        if (lane < 4) {
          if constexpr (Nv)
            cp_async_16(slot + CodeBytes + lane * 16,
                        scale_tiles[part] + (kt >> 1) * 128 + (kt & 1) * 16);
          else
            cp_async_8(slot + CodeBytes + lane * 8,
                       scale_tiles[part] + (kt >> 2) * 128 + (kt & 3) * 8);
        }
      }
    }
    cp_async_commit();
  };
  auto issue_first = [&] {
    if (producer >= 0) {
#pragma unroll
      for (int j = 0; j < Depth; ++j) issue(j);
    }
  };
  issue_first();

  const int first = expert_offsets[expert];
  const int end = min(expert_offsets[expert + 1], *num_padded_rows);
  int routed = first < end ? sorted_rows[first] : -1;
  if constexpr (!GateUp) ptx::griddep_wait();
  const int routes = *live_tokens * TopK;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int g_swizzle = Nv ? (g >> 2) & 1 : (g >> 1) & 3;
  const uint32_t code_selector = Nv ? uint32_t(t) : uint32_t(2 * t) | uint32_t(2 * t + 1) << 4;
  const int scale_byte = g >> 2;
  const uint32_t scale_selector[2] = {uint32_t(scale_byte) * 0x11, uint32_t(scale_byte + 2) * 0x11};
  float gate_global = 1.f, up_global = 1.f;
  if constexpr (Nv) {
    gate_global = weight_f32_scales[R.expert_globals ? expert * (GateUp ? 2 : 1) : 0];
    if constexpr (GateUp) up_global = weight_f32_scales[R.expert_globals ? expert * 2 + 1 : 0];
  }
  constexpr int RowChunks = K / 8;
  constexpr int ChunksPerThread = (RowChunks + Warps * 32 - 1) / (Warps * 32);
  // Converted steps are numbered across passes, so barrier phases continue.
  int base_step = 0;
  // Routes the model writes give an expert at most one row per token; any
  // further slots take further passes, as in the ring.
  for (int slot_index = first; slot_index < end; ++slot_index) {
    if (slot_index != first) {
      cp_async_wait<0>();
      __syncthreads();
      routed = sorted_rows[slot_index];
    }
    if (routed >= routes) routed = -1;
    if (routed < 0) continue;
    if (slot_index != first) issue_first();
    {
      const int row = GateUp ? routed / TopK : routed;
      uint4 row_chunks[ChunksPerThread];
#pragma unroll
      for (int i = 0; i < ChunksPerThread; ++i) {
        const int chunk = threadIdx.x + i * Warps * 32;
        if (chunk < RowChunks)
          row_chunks[i] =
              *reinterpret_cast<const uint4*>(activation + uint64_t(row) * K * 2 + chunk * 16);
      }
#pragma unroll
      for (int i = 0; i < ChunksPerThread; ++i) {
        const int chunk = threadIdx.x + i * Warps * 32;
        if (chunk < RowChunks) reinterpret_cast<uint4*>(cta_row)[chunk] = row_chunks[i];
      }
    }
    __syncthreads();
    if (producer >= 0) {
#pragma unroll 1
      for (int j = 0; producer + j * Producers < Ktiles; ++j) {
        const int kt = producer + j * Producers;
        cp_async_wait<Depth - 1>();
        __syncwarp();
        const int step = base_step + kt;
        const int s = step % Stages;
        ptx::mbarrier_wait(&empty[s], ((step / Stages) & 1) ^ 1);
        const uint8_t* slot = ring + (j % Depth) * SlotBytes;
#pragma unroll
        for (int part = 0; part < Parts; ++part) {
          ptx::A4 a[4];
          packed_fragments<W>(slot + part * PartBytes, g, g_swizzle, code_selector,
                              scale_selector, a);
          uint4* out = reinterpret_cast<uint4*>(converted + (s * Parts + part) * kStagedPartBytes);
#pragma unroll
          for (int k16 = 0; k16 < 4; ++k16)
            out[k16 * 32 + lane] = make_uint4(a[k16].v[0], a[k16].v[1], a[k16].v[2], a[k16].v[3]);
        }
        ptx::mbarrier_arrive(&full[s]);
        // Every lane has read the slot before any lane refills it.
        __syncwarp();
        issue(j + Depth);
      }
    } else {
      ptx::D4 accum[Parts];
#pragma unroll
      for (int part = 0; part < Parts; ++part)
#pragma unroll
        for (int i = 0; i < 4; ++i) accum[part].v[i] = 0.f;
#pragma unroll 1
      for (int kt = 0; kt < Ktiles; ++kt) {
        const int step = base_step + kt;
        const int s = step % Stages;
        ptx::mbarrier_wait(&full[s], (step / Stages) & 1);
        const uint8_t* row = reinterpret_cast<const uint8_t*>(cta_row + kt * 64) + t * 4;
        uint32_t b_words[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) b_words[i] = *reinterpret_cast<const uint32_t*>(row + i * 16);
#pragma unroll
        for (int k16 = 0; k16 < 4; ++k16) {
          const ptx::B2 b{{b_words[2 * k16], b_words[2 * k16 + 1]}};
#pragma unroll
          for (int part = 0; part < Parts; ++part) {
            const uint4 v = reinterpret_cast<const uint4*>(
                converted + (s * Parts + part) * kStagedPartBytes)[k16 * 32 + lane];
            const ptx::A4 a{{v.x, v.y, v.z, v.w}};
            ptx::mma_bf16(accum[part], a, b);
          }
        }
        ptx::mbarrier_arrive(&empty[s]);
      }
      // Rows g and g + 8, token column 0: c0 and c2 of lanes with t == 0.
      if (t == 0) {
#pragma unroll
        for (int i = 0; i < 4; i += 2) {
          float result = accum[0].v[i];
          if constexpr (Nv) result = __fmul_rn(result, gate_global);
          if constexpr (GateUp) {
            float up_value = accum[Parts - 1].v[i];
            if constexpr (Nv) up_value = __fmul_rn(up_value, up_global);
            result = result / (1.0f + expf(-result)) * up_value;
          } else {
            result *= topk_weights[routed];
          }
          output[uint64_t(routed) * OutN + column + g + 8 * (i / 2)] =
              convert::bf16_from_f32(result);
        }
      }
    }
    base_step += Ktiles;
  }
}

} // namespace chop::moe::grouped
