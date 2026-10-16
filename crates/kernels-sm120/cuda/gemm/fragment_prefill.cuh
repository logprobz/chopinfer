#pragma once

#include "prefill_mainloop.cuh"

namespace chop::gemm {

// Fragment-major prefill: a persistent CTA owns Tokens tokens by eight
// 16-row weight strips. SwiGLU CTAs stage four gate strips and the matching
// four up strips, so each warp holds gate and up accumulators for the same
// outputs.
template<WeightKind W, bool Swiglu, int Tokens>
struct FragmentPrefillTraits {
  static_assert(W == WeightKind::bf16 || fragment_fp8<W>);
  static_assert(Tokens == 64 || Tokens == 128);
  static constexpr int tokens = Tokens;
  static constexpr int quarters = Tokens / 32;
  static constexpr int strips = 8;
  static constexpr int math_warps = 8;
  // Math warps are feature groups by 32-token quarters; a warp owns
  // warp_strips strips by four n8 token tiles.
  static constexpr int groups = math_warps / quarters;
  static constexpr int warp_strips = strips / groups;
  static constexpr int output_strips = Swiglu ? warp_strips / 2 : warp_strips;
  // Warps 8-11 form the producer warpgroup; it donates registers to the
  // math warpgroups with setmaxnreg (24 to 240 each: the math warps hold a
  // tile's total, segment and block accumulators) and one lane issues every
  // copy.
  static constexpr int threads = (math_warps + 4) * 32;
  static constexpr int strip_stage_bytes = W == WeightKind::bf16 ? 2048 : 1024;
  static constexpr int activation_bytes = tokens * prefill::swizzle_row_bytes;
  static constexpr int stage_bytes =
      activation_bytes + strips * strip_stage_bytes;
  static constexpr int stages = 96 * 1024 / stage_bytes;
  static constexpr int output_rows = Swiglu ? 64 : 128;
  static constexpr int fragments = warp_strips * 4;
  static constexpr int partial_floats = math_warps * fragments * 32 * 4;
  static constexpr int shared_bytes = stages * stage_bytes + 1024;
};

// Every tile's K splits into the same prefill_segments(k_blocks, splits)
// segments of whole 128-K blocks, at any live M (`splits` is the entry's
// fixed segment count, `gemm::prefill_segments`). An output element is the
// sum, from zero in segment order, of its segments' sums, each the sum from
// zero in K order of its blocks' MMA chains (FP8 blocks by their F32
// scale). So its bits depend on neither live M, the grid, nor the element's
// place in the step: only where the work splits does, and it splits at
// segment bounds.
constexpr uint32_t kMaxSegments = 8;

__device__ __forceinline__ uint32_t prefill_segments(
    uint32_t k_blocks, uint32_t splits) {
  return max(1u, min(min(kMaxSegments, splits), k_blocks));
}

__device__ __forceinline__ uint32_t segment_first_block(
    uint32_t k_blocks, uint32_t segments, uint32_t segment) {
  return k_blocks * segment / segments;
}

// Whole tiles are dealt round robin in full waves. When the remaining tiles
// fill less than 90% of a wave, their segments are instead divided evenly
// over the CTAs (stream-K over segments). The CTA holding a split tile's
// first segment sums its segments into one partial; every later CTA of the
// tile leaves each of its segments' sums as a partial. Once all have
// arrived, the tile's CTAs finish it together, each adding the partials of
// its own strips in segment order.
struct PrefillSchedule {
  uint32_t token_tiles, segments, grid;
  // Stream-K counts stay in 32 bits: at most one wave of tiles times the
  // segments, times the grid.
  uint32_t dp_tiles, sk_ctas, sk_iters;

  __device__ __forceinline__ uint32_t sk_begin(uint32_t cta) const {
    return cta * sk_iters / sk_ctas;
  }
  // The CTA whose stream-K range holds segment unit x.
  __device__ __forceinline__ uint32_t sk_owner(uint32_t x) const {
    return ((x + 1) * sk_ctas + sk_iters - 1) / sk_iters - 1;
  }
};

__device__ __forceinline__ PrefillSchedule prefill_schedule(
    uint32_t tiles, uint32_t token_tiles, uint32_t segments, uint32_t grid) {
  PrefillSchedule s;
  s.token_tiles = token_tiles;
  s.segments = segments;
  s.grid = grid;
  uint32_t tail = tiles >= grid ? tiles % grid : tiles;
  if (segments == 1 || uint64_t(tail) * 10 >= uint64_t(grid) * 9) tail = 0;
  s.dp_tiles = tiles - tail;
  s.sk_ctas = min(grid, tail * segments);
  s.sk_iters = tail * segments;
  return s;
}

struct PrefillWork {
  uint32_t tile, feature_tile, token_tile;
  // Segments [first_segment, end_segment) of the tile.
  uint32_t first_segment, end_segment;
  bool partial;
};

// Enumerates one CTA's work: its whole tiles, then its stream-K range.
struct PrefillWorkIterator {
  uint32_t next_dp, sk_next, sk_end;

  __device__ __forceinline__ PrefillWorkIterator(
      const PrefillSchedule& s, uint32_t cta) {
    next_dp = cta;
    sk_next = cta < s.sk_ctas ? s.sk_begin(cta) : 0;
    sk_end = cta < s.sk_ctas ? s.sk_begin(cta + 1) : 0;
  }

  __device__ __forceinline__ bool next(
      const PrefillSchedule& s, PrefillWork& w) {
    if (next_dp < s.dp_tiles) {
      w.tile = next_dp;
      next_dp += s.grid;
      w.first_segment = 0;
      w.end_segment = s.segments;
      w.partial = false;
    } else if (sk_next < sk_end) {
      const uint32_t tail_tile = sk_next / s.segments;
      w.first_segment = sk_next % s.segments;
      w.end_segment = min(s.segments, w.first_segment + (sk_end - sk_next));
      w.tile = s.dp_tiles + tail_tile;
      w.partial = w.end_segment - w.first_segment != s.segments;
      sk_next += w.end_segment - w.first_segment;
    } else {
      return false;
    }
    // Feature tiles are outer so concurrent CTAs share one weight tile and
    // stream the activation tiles, which stay L2 resident.
    w.feature_tile = w.tile / s.token_tiles;
    w.token_tile = w.tile % s.token_tiles;
    return true;
  }
};

template<bool Swiglu>
__device__ __forceinline__ uint32_t prefill_slot_strip(
    uint32_t slot, uint32_t feature_tile, uint32_t half_strips) {
  if constexpr (Swiglu)
    return slot < 4 ? feature_tile * 4 + slot
                    : half_strips + feature_tile * 4 + slot - 4;
  return feature_tile * 8 + slot;
}

template<bool F32>
__device__ __forceinline__ void store_pair(
    void* output, uint64_t offset, float lo, float hi) {
  if constexpr (F32) {
    *reinterpret_cast<float2*>(static_cast<float*>(output) + offset) =
        make_float2(lo, hi);
  } else {
    *reinterpret_cast<__nv_bfloat162*>(
        static_cast<__nv_bfloat16*>(output) + offset) =
        __floats2bfloat162_rn(lo, hi);
  }
}

// Stores one 16-feature by 8-token C fragment as Y[token][feature]. Lanes
// exchange values with their g^1 neighbour so each store covers two features.
template<bool F32>
__device__ __forceinline__ void store_fragment(
    void* output, const float (&c)[4], uint32_t feature, uint32_t token,
    uint32_t live_m, uint32_t row_stride) {
  const int lane = threadIdx.x & 31;
  const bool even = ((lane >> 2) & 1) == 0;
  const float x0 = __shfl_xor_sync(0xffffffff, even ? c[1] : c[0], 4);
  const float x1 = __shfl_xor_sync(0xffffffff, even ? c[3] : c[2], 4);
  const uint32_t row = even ? token : token + 1;
  const uint32_t column = even ? feature : feature - 1;
  if (row < live_m) {
    const uint64_t offset = uint64_t(row) * row_stride + column;
    store_pair<F32>(output, offset, even ? c[0] : x0, even ? x0 : c[1]);
    store_pair<F32>(output, offset + 8, even ? c[2] : x1, even ? x1 : c[3]);
  }
}

// Tokens = 64 serves the medium-M entries (gemm::mid): it stages and keeps
// half the activation rows of a 128-token tile, which a live M of 64 or
// less leaves unused, and so holds a deeper ring.
template<WeightKind W, bool Swiglu, int Tokens>
__device__ void run_fragment_prefill(
    const CUtensorMap& activation_map, const uint8_t* weight,
    void* output, float* partial, uint32_t* semaphores,
    uint32_t live_m, uint32_t n, uint32_t k, uint32_t splits,
    uint32_t output_f32, const float* scales, const void* residual) {
  using T = FragmentPrefillTraits<W, Swiglu, Tokens>;
  constexpr bool fp8 = fragment_fp8<W>;
  constexpr int WS = T::warp_strips;
  // Strips of one FP8 scale among a warp's slots: gate and up strips never
  // share one, and a warp's 4 (or 2) gate or plain strips always do.
  constexpr int scale_strips = (Swiglu ? WS / 2 : WS) >= 2 ? 2 : 1;
  (void)residual;
  __shared__ uint64_t full[T::stages];
  __shared__ uint64_t empty[T::stages];
  extern __shared__ unsigned char raw[];
  unsigned char* storage = prefill::align_swizzle_storage(raw);
  const uint32_t storage_address = ptx::shared_address(storage);
  const uint32_t output_rows = Swiglu ? n / 2 : n;
  const uint32_t strips_total = n / 16;
  const uint32_t half_strips = output_rows / 16;
  const uint32_t feature_tiles =
      (output_rows + T::output_rows - 1) / T::output_rows;
  const uint32_t token_tiles = (live_m + T::tokens - 1) / T::tokens;
  const uint32_t tiles = feature_tiles * token_tiles;
  const uint32_t k_blocks = k / 128;
  const uint32_t k_stages = k / prefill::stage_k;
  const uint32_t segments = prefill_segments(k_blocks, splits);
  // The schedule depends on the launch grid and live M; the arithmetic of
  // an output element does not.
  const PrefillSchedule schedule =
      prefill_schedule(tiles, token_tiles, segments, gridDim.x);
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
#ifdef CHOP_GEMM_CLOCK_PROFILE
  uint64_t profile_wait = 0, profile_math = 0, profile_issue = 0;
  uint64_t profile_epilogue = 0, profile_prologue = 0;
  uint64_t profile_stages = 0, profile_units = 0;
  const uint64_t profile_start = clock64();
#endif

  if (threadIdx.x == 0) {
    for (int stage = 0; stage < T::stages; ++stage) {
      ptx::mbarrier_init(&full[stage], 1);
      ptx::mbarrier_init(&empty[stage], T::math_warps);
    }
    ptx::fence_mbarrier_init();
  }
  __syncthreads();
  ptx::griddep_launch_dependents();

  const auto stage_range = [&](const PrefillWork& u, uint32_t& first,
                               uint32_t& end) {
    first = 2 * segment_first_block(k_blocks, segments, u.first_segment);
    end = 2 * segment_first_block(k_blocks, segments, u.end_segment);
  };

  if (warp >= T::math_warps) {
    ptx::setmaxnreg_dec<24>();
    if (warp != T::math_warps || lane != 0) return;
    // Producer. Weights never depend on the previous kernel, so the first
    // unit's weight copies are issued before the PDL wait.
    const auto issue_weights = [&](const PrefillWork& u, uint32_t stage,
                                   uint32_t slot) {
      unsigned char* destination = storage + slot * T::stage_bytes +
          T::activation_bytes;
      for (uint32_t s = 0; s < T::strips; ++s) {
        const uint32_t strip =
            prefill_slot_strip<Swiglu>(s, u.feature_tile, half_strips);
        if (strip >= strips_total) break;
        ptx::cp_async_bulk(destination + s * T::strip_stage_bytes,
            weight + (uint64_t(strip) * k_stages + stage) *
                T::strip_stage_bytes,
            T::strip_stage_bytes, &full[slot]);
      }
    };
    const auto stage_bytes = [&](const PrefillWork& u) {
      uint32_t valid = 0;
      for (uint32_t s = 0; s < T::strips; ++s)
        valid += prefill_slot_strip<Swiglu>(s, u.feature_tile, half_strips) <
            strips_total;
      return uint32_t(T::activation_bytes) + valid * T::strip_stage_bytes;
    };
    const auto issue_activation = [&](const PrefillWork& u, uint32_t stage,
                                      uint32_t slot) {
      const int32_t coord[2] = {int32_t(stage * prefill::stage_k),
                                int32_t(u.token_tile * T::tokens)};
      ptx::tma_load<2>(storage + slot * T::stage_bytes, &activation_map,
          coord, &full[slot]);
    };
    uint32_t prefetched = 0;
    PrefillWork u;
    PrefillWorkIterator first(schedule, blockIdx.x);
    if (first.next(schedule, u)) {
      uint32_t first_stage, end_stage;
      stage_range(u, first_stage, end_stage);
      const uint32_t bytes = stage_bytes(u);
      prefetched = min(uint32_t(T::stages), end_stage - first_stage);
      for (uint32_t i = 0; i < prefetched; ++i) {
        ptx::mbarrier_arrive_expect_tx(&full[i], bytes);
        issue_weights(u, first_stage + i, i);
      }
    }
    ptx::griddep_wait();
    uint32_t iteration = 0;
    PrefillWorkIterator work(schedule, blockIdx.x);
    while (work.next(schedule, u)) {
      uint32_t first_stage, end_stage;
      stage_range(u, first_stage, end_stage);
      const uint32_t bytes = stage_bytes(u);
      for (uint32_t stage = first_stage; stage < end_stage;
           ++stage, ++iteration) {
        const uint32_t slot = iteration % T::stages;
        if (iteration < prefetched) {
          issue_activation(u, stage, slot);
          continue;
        }
#ifdef CHOP_GEMM_CLOCK_PROFILE
        const uint64_t marker = clock64();
#endif
        prefill::wait_empty(empty, iteration, T::stages);
#ifdef CHOP_GEMM_CLOCK_PROFILE
        profile_issue += clock64() - marker;
#endif
        ptx::mbarrier_arrive_expect_tx(&full[slot], bytes);
        issue_activation(u, stage, slot);
        issue_weights(u, stage, slot);
      }
    }
#ifdef CHOP_GEMM_CLOCK_PROFILE
    auto* samples = reinterpret_cast<uint64_t*>(semaphores + 2048) +
        uint64_t(blockIdx.x) * 8;
    samples[2] = profile_issue;
#endif
    return;
  }

  // Math warps: warp_f selects warp_strips of the staged strips and warp_t
  // a 32-token quarter.
  ptx::setmaxnreg_inc<240>();
  ptx::griddep_wait();
  const int warp_f = warp / T::quarters;
  const int warp_t = warp % T::quarters;
  uint32_t slots[WS];
#pragma unroll
  for (int i = 0; i < WS; ++i)
    slots[i] = Swiglu
        ? (i < WS / 2 ? warp_f * (WS / 2) + i
                      : 4 + warp_f * (WS / 2) + i - WS / 2)
        : warp_f * WS + i;
  const uint32_t b_row = warp_t * 32 + (lane & 7) + ((lane >> 4) << 3);
  const uint32_t b_half = (lane >> 3) & 1;
  const uint32_t b_xor = lane & 7;
  uint32_t iteration = 0;
#ifdef CHOP_GEMM_CLOCK_PROFILE
  if (threadIdx.x == 0) profile_prologue = clock64() - profile_start;
#endif

  // Partial slots per CTA: its first segments' sum for the tile its range
  // ends in, then one per segment of the tile its range starts in.
  const auto partial_slot = [&](uint32_t cta, uint32_t slot) {
    return partial + (uint64_t(cta) * kMaxSegments + slot) * T::partial_floats +
        (warp * T::fragments * 32 + lane) * 4;
  };
  const auto write_partial = [&](uint32_t slot, const ptx::D4 (&values)[WS][4]) {
    float* own = partial_slot(blockIdx.x, slot);
#pragma unroll
    for (int f = 0; f < T::fragments; ++f)
      *reinterpret_cast<float4*>(own + f * 32 * 4) = make_float4(
          values[f / 4][f % 4].v[0], values[f / 4][f % 4].v[1],
          values[f / 4][f % 4].v[2], values[f / 4][f % 4].v[3]);
  };

  // Whether virtual warp vw's strips exist and its token quarter holds a
  // live row; other warps skip their MMAs and stores.
  const auto live_warp = [&](int vw, uint32_t feature_tile,
                             uint32_t token_tile) {
    const uint32_t first_slot = Swiglu ? (vw / T::quarters) * (WS / 2)
                                       : (vw / T::quarters) * WS;
    return prefill_slot_strip<Swiglu>(first_slot, feature_tile,
               half_strips) < strips_total &&
        token_tile * T::tokens + (vw % T::quarters) * 32 < live_m;
  };
  // Stores output strip i of virtual warp vw's fragments (with the matching
  // up strip for SwiGLU).
  const auto store_strip = [&](int vw, uint32_t feature_tile,
                               uint32_t token_tile, int i,
                               const ptx::D4 (&gate)[4],
                               const ptx::D4 (&up)[4]) {
    const int vw_f = vw / T::quarters;
    const int vw_t = vw % T::quarters;
    const uint32_t token = token_tile * T::tokens + vw_t * 32 +
        (lane & 3) * 2;
    const uint32_t feature = (Swiglu
        ? feature_tile * 4 + vw_f * (WS / 2) + i
        : prefill_slot_strip<false>(vw_f * WS + i, feature_tile,
              half_strips)) * 16 + (lane >> 2);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float c[4];
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        if constexpr (Swiglu) {
          const float g = gate[j].v[v];
          c[v] = (g / (1.0f + expf(-g))) * up[j].v[v];
        } else {
          c[v] = gate[j].v[v];
        }
      }
      if (output_f32)
        store_fragment<true>(output, c, feature, token + j * 8, live_m,
            output_rows);
      else
        store_fragment<false>(output, c, feature, token + j * 8, live_m,
            output_rows);
    }
  };

  // `total` sums the segments of a tile the CTA holds from its first
  // segment, `sum` the blocks of the current segment, `block` one 128-K
  // block's MMA chain from zero (so no chain spans more than 128 K; FP8
  // promotes it by its F32 scale, BF16 adds it).
  ptx::D4 total[WS][4];
  // A CTA holds at most two split tiles, so it helps finish at most two.
  uint32_t pending_first = 0, pending_second = 0, pending_count = 0;
  PrefillWorkIterator work(schedule, blockIdx.x);
  PrefillWork u;
  while (work.next(schedule, u)) {
    const bool valid = live_warp(warp, u.feature_tile, u.token_tile);
    uint32_t scale_rows[WS / scale_strips];
#pragma unroll
    for (int p = 0; p < WS / scale_strips; ++p)
      scale_rows[p] = prefill_slot_strip<Swiglu>(
          slots[p * scale_strips], u.feature_tile, half_strips) /
          fragment_scale_strips<W>;
    ptx::D4 block[WS][4], sum[WS][4];
#pragma unroll
    for (int i = 0; i < WS; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        total[i][j] = ptx::D4{};
        sum[i][j] = ptx::D4{};
        block[i][j] = ptx::D4{};
      }
    float scale[WS / scale_strips];
#pragma unroll
    for (int p = 0; p < WS / scale_strips; ++p) scale[p] = 0.0f;
    // The tile's first segment is this CTA's: its segments add into total.
    const bool first = u.first_segment == 0;
    uint32_t segment = u.first_segment;
    uint32_t segment_end = 2 * segment_first_block(k_blocks, segments, segment + 1);
    uint32_t first_stage, end_stage;
    stage_range(u, first_stage, end_stage);
#ifdef CHOP_GEMM_CLOCK_PROFILE
    if (threadIdx.x == 0) ++profile_units;
#endif
    for (uint32_t stage = first_stage; stage < end_stage;
         ++stage, ++iteration) {
      const uint32_t slot = iteration % T::stages;
      if constexpr (fp8) {
        if (valid && (stage & 1) == 0) {
          const uint32_t kb = stage / 2;
#pragma unroll
          for (int p = 0; p < WS / scale_strips; ++p)
            scale[p] = __ldg(scales + uint64_t(scale_rows[p]) * k_blocks + kb);
        }
      }
#ifdef CHOP_GEMM_CLOCK_PROFILE
      uint64_t marker = threadIdx.x == 0 ? clock64() : 0;
#endif
      prefill::wait_full(full, iteration, T::stages);
#ifdef CHOP_GEMM_CLOCK_PROFILE
      if (threadIdx.x == 0) {
        profile_wait += clock64() - marker;
        marker = clock64();
        ++profile_stages;
      }
#endif
      const uint32_t activation = storage_address + slot * T::stage_bytes;
      const uint32_t b_address = activation + b_row * prefill::swizzle_row_bytes;
      const uint32_t weights = activation + T::activation_bytes + lane * 16;
      if (valid) {
        if constexpr (fp8) {
#pragma unroll
          for (int pair = 0; pair < 2; ++pair) {
            ptx::A4 packed[WS];
#pragma unroll
            for (int i = 0; i < WS; ++i)
              packed[i] = prefill::load_a(
                  weights + slots[i] * T::strip_stage_bytes + pair * 512);
#pragma unroll
            for (int half = 0; half < 2; ++half) {
              const uint32_t chunk = (pair * 2 + half) * 2 + b_half;
              ptx::B2 b[4];
              prefill::load_b2(b_address, chunk, b_xor, b[0], b[1]);
              prefill::load_b2(b_address + 16 * prefill::swizzle_row_bytes,
                  chunk, b_xor, b[2], b[3]);
#pragma unroll
              for (int i = 0; i < WS; ++i) {
                const ptx::A4 a = half == 0
                    ? prefill::fp8_to_bf16<0>(packed[i])
                    : prefill::fp8_to_bf16<1>(packed[i]);
#pragma unroll
                for (int j = 0; j < 4; ++j) ptx::mma_bf16(block[i][j], a, b[j]);
              }
            }
          }
        } else {
#pragma unroll
          for (int step = 0; step < 4; ++step) {
            ptx::A4 a[WS];
#pragma unroll
            for (int i = 0; i < WS; ++i)
              a[i] = prefill::load_a(
                  weights + slots[i] * T::strip_stage_bytes + step * 512);
            const uint32_t chunk = step * 2 + b_half;
            ptx::B2 b[4];
            prefill::load_b2(b_address, chunk, b_xor, b[0], b[1]);
            prefill::load_b2(b_address + 16 * prefill::swizzle_row_bytes,
                chunk, b_xor, b[2], b[3]);
#pragma unroll
            for (int i = 0; i < WS; ++i)
#pragma unroll
              for (int j = 0; j < 4; ++j) ptx::mma_bf16(block[i][j], a[i], b[j]);
          }
        }
      }
      __syncwarp();
      if (lane == 0) ptx::mbarrier_arrive(&empty[slot]);
      if ((stage & 1) == 1) {
        if (valid) {
#pragma unroll
          for (int i = 0; i < WS; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
              for (int v = 0; v < 4; ++v) {
                if constexpr (fp8)
                  sum[i][j].v[v] += block[i][j].v[v] * scale[i / scale_strips];
                else sum[i][j].v[v] += block[i][j].v[v];
                block[i][j].v[v] = 0.0f;
              }
        }
        if (stage + 1 == segment_end) {
          // The segment is complete: into total (which the tile's first
          // segment starts), or its own partial.
          if (!first) {
            write_partial(1 + segment - u.first_segment, sum);
          } else {
#pragma unroll
            for (int i = 0; i < WS; ++i)
#pragma unroll
              for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int v = 0; v < 4; ++v) total[i][j].v[v] += sum[i][j].v[v];
          }
#pragma unroll
          for (int i = 0; i < WS; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) sum[i][j] = ptx::D4{};
          ++segment;
          segment_end = 2 * segment_first_block(k_blocks, segments, segment + 1);
        }
      }
#ifdef CHOP_GEMM_CLOCK_PROFILE
      if (threadIdx.x == 0) profile_math += clock64() - marker;
#endif
    }
#ifdef CHOP_GEMM_CLOCK_PROFILE
    const uint64_t epilogue_start = threadIdx.x == 0 ? clock64() : 0;
#endif

    if (u.partial) {
      if (first) write_partial(0, total);
      __threadfence();
      prefill::named_barrier(1, T::math_warps * 32);
      const uint32_t tail_tile = u.tile - schedule.dp_tiles;
      if (threadIdx.x == 0) atomicAdd(&semaphores[2 * tail_tile], 1);
      // Every contributor helps finish the tile, after its own segments.
      if (pending_count == 0) pending_first = u.tile;
      else pending_second = u.tile;
      ++pending_count;
    } else if (valid) {
#pragma unroll
      for (int i = 0; i < T::output_strips; ++i)
        store_strip(warp, u.feature_tile, u.token_tile, i, total[i],
            total[Swiglu ? i + T::output_strips : i]);
    }
#ifdef CHOP_GEMM_CLOCK_PROFILE
    if (threadIdx.x == 0) profile_epilogue += clock64() - epilogue_start;
#endif
  }
  // Finish split tiles cooperatively. Once all of a tile's CTAs have
  // arrived, each sums a disjoint set of strips: the first CTA's partial,
  // then every later segment's in order, and stores them. All stream-K CTAs
  // are resident (one per SM, grid within the SM count) and finish their own
  // segments before waiting, so the wait cannot deadlock. The last CTA to
  // finish a tile resets both of its counters.
  constexpr int units = T::math_warps * T::output_strips;
  constexpr int strips_per_warp = T::output_strips;
  for (uint32_t p = 0; p < pending_count; ++p) {
    const uint32_t tile = p == 0 ? pending_first : pending_second;
    const uint32_t tail_tile = tile - schedule.dp_tiles;
    const uint32_t tile_first = tail_tile * segments;
    const uint32_t first_cta = schedule.sk_owner(tile_first);
    const uint32_t last_cta = schedule.sk_owner(tile_first + segments - 1);
    const uint32_t owners = last_cta - first_cta + 1;
    const uint32_t position = blockIdx.x - first_cta;
    const uint32_t feature_tile = tile / token_tiles;
    const uint32_t token_tile = tile % token_tiles;
    if (threadIdx.x == 0) {
      uint32_t arrived;
      do {
        asm volatile("ld.acquire.gpu.u32 %0, [%1];"
            : "=r"(arrived) : "l"(&semaphores[2 * tail_tile]) : "memory");
        if (arrived < owners) __nanosleep(64);
      } while (arrived < owners);
    }
    prefill::named_barrier(1, T::math_warps * 32);
    const uint32_t begin = position * units / owners;
    const uint32_t end = (position + 1) * units / owners;
    for (uint32_t unit = begin + warp; unit < end; unit += T::math_warps) {
      const int vw = int(unit) / strips_per_warp;
      const int i = int(unit) % strips_per_warp;
      if (!live_warp(vw, feature_tile, token_tile)) continue;
      ptx::D4 gate[4], up[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        gate[j] = ptx::D4{};
        up[j] = ptx::D4{};
      }
      // The first CTA's partial, then each later CTA's segments in order.
      for (uint32_t cta = first_cta; cta <= last_cta; ++cta) {
        const uint32_t from = max(tile_first, schedule.sk_begin(cta));
        const uint32_t to = min(tile_first + segments, schedule.sk_begin(cta + 1));
        for (uint32_t x = from; x < to; ++x) {
          if (cta == first_cta && x != from) break;
          const float* part = partial +
              (uint64_t(cta) * kMaxSegments + (cta == first_cta ? 0 : 1 + x - from)) *
                  T::partial_floats +
              (vw * T::fragments * 32 + lane) * 4;
          float4 values[8];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            values[j] = __ldcg(reinterpret_cast<const float4*>(
                part + (i * 4 + j) * 32 * 4));
            if constexpr (Swiglu)
              values[4 + j] = __ldcg(reinterpret_cast<const float4*>(
                  part + ((i + T::output_strips) * 4 + j) * 32 * 4));
          }
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            gate[j].v[0] += values[j].x;
            gate[j].v[1] += values[j].y;
            gate[j].v[2] += values[j].z;
            gate[j].v[3] += values[j].w;
            if constexpr (Swiglu) {
              up[j].v[0] += values[4 + j].x;
              up[j].v[1] += values[4 + j].y;
              up[j].v[2] += values[4 + j].z;
              up[j].v[3] += values[4 + j].w;
            }
          }
        }
      }
      store_strip(vw, feature_tile, token_tile, i, gate, up);
    }
    prefill::named_barrier(1, T::math_warps * 32);
    if (threadIdx.x == 0 &&
        atomicAdd(&semaphores[2 * tail_tile + 1], 1) == owners - 1) {
      semaphores[2 * tail_tile] = 0;
      semaphores[2 * tail_tile + 1] = 0;
    }
  }
#ifdef CHOP_GEMM_CLOCK_PROFILE
  if (threadIdx.x == 0) {
    auto* samples = reinterpret_cast<uint64_t*>(semaphores + 2048) +
        uint64_t(blockIdx.x) * 8;
    samples[0] = profile_wait;
    samples[1] = profile_math;
    samples[3] = profile_epilogue;
    samples[4] = profile_prologue;
    samples[5] = clock64() - profile_start;
    samples[6] = profile_stages;
    samples[7] = profile_units;
  }
#endif
}

} // namespace chop::gemm
