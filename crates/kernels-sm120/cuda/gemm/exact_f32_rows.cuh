#pragma once
#include "exact_f32_prefill.cuh"
#include "../common/ptx_matrix.cuh"

namespace chop::exact_f32::rows {

// One CTA owns Features x Rows outputs over the K of its slice. Every
// output keeps the original arithmetic: each 128-K chain runs its hi, mid
// and lo components from +0 through the same eight k16 MMAs (weights as
// A, the prepared components as B) and merges them in that order. With one
// slice the chains add into the output from +0 in K order and no partial
// leaves the CTA; with several, each chain's value goes to its own plane
// (`Partials`) and the cached reducer adds them from +0 in K order. Warps
// own WarpFeatures x WarpRows outputs and read ldmatrix fragments from a
// ring of Stages swizzled planes of PlaneK: the weight lines, then each
// component's row lines. Row tiles are the grid's fastest axis, so the
// CTAs that read one weight tile run together and share it in L2.
template<int Features, int Rows, int WarpFeatures, int WarpRows, int Stages, int PlaneK>
struct Tile {
  static constexpr int Warps = (Features/WarpFeatures)*(Rows/WarpRows);
  static constexpr int Threads = Warps*32;
  static constexpr int Lines = Features + 3*Rows;
  static constexpr int LineChunks = PlaneK/8;
  static constexpr int Plane = Lines*PlaneK;
  static constexpr int Chunks = Lines*LineChunks;
  static constexpr int PerThread = (Chunks + Threads - 1)/Threads;
  static constexpr int ChainPlanes = 128/PlaneK;
  static constexpr int FeatureTiles = WarpFeatures/16;
  static constexpr int RowTiles = WarpRows/8;
  static_assert(Features%WarpFeatures == 0 && Rows%WarpRows == 0);
  static_assert(WarpFeatures%16 == 0 && WarpRows%8 == 0 && Rows%8 == 0);
  static_assert(RowTiles == 1 || RowTiles%2 == 0);
  static_assert(PlaneK == 32 || PlaneK == 64 || PlaneK == 128);
  static_assert(Threads%LineChunks == 0 && Threads <= 1024 && Stages >= 2);
};

// Rotate each 16-byte vector of a line so the eight lines an ldmatrix
// phase reads hit distinct banks: by the line's low three bits for lines
// of at least 128 bytes, by bits one and two for 64-byte lines.
template<int PlaneK>
__device__ __forceinline__ int swizzle(int line, int k) {
  constexpr int Mask = PlaneK >= 64 ? 7 : 3;
  const int rotate = PlaneK >= 64 ? (line&7) : ((line>>1)&3);
  return line*PlaneK + ((((k/8)^rotate)&Mask) | ((k/8)&~Mask))*8 + k%8;
}

// The position of set bit `n` (from 0) of `mask`, lowest first.
__device__ __forceinline__ int nth_bit(uint32_t mask, int n) {
  for (; n > 0; --n) mask &= mask - 1;
  return __ffs(mask) - 1;
}

// The chain planes of a sliced projection follow the preparation's row
// domain and components in the workspace, 16-byte aligned:
// partial[(chain*rows + row)*N + feature], the cached reducer's layout.
template<int K>
__device__ __forceinline__ float* partials(void* workspace, uint32_t rows) {
  return reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(workspace)
      + (uint64_t(rows)+3)/4*16 + uint64_t(rows)*K*6);
}

// The original SIMT chains of the CTA's K slice for `Width` features from
// `feature0` (those of the 16-feature tiles set in `tiles`) at `count`
// rows: rows `row0` on, or with `lost_rows` the rows set in `lost` from
// its set bit `first`. Each output runs FMAs from +0 in K order within
// each 128-K chain and adds its chains from +0 in K order (or stores each
// to its plane), as the original entry does. Weight and activation lines
// reach shared memory through a ring of Stages panels of 16-byte copies,
// the deepest that fits the CTA's ring; thread `t` owns feature
// `t % Width` at RowsPer rows `t / Width` apart.
template<int Width, int Bytes, int Threads, int RowsPer>
struct SimtPanel {
  static constexpr int Slots = Threads/Width;
  static constexpr int Lines = Slots*RowsPer;
  static constexpr int buffer(int k) { return Width*(k + 8)*2 + Lines*(k + 4)*4; }
  static constexpr bool fits(int k, int stages) { return stages*buffer(k) <= Bytes; }
  // The panel K and stage count with the most K in flight.
  static constexpr int K = fits(128, 4) ? 128 : fits(128, 3) ? 128 : fits(64, 4) ? 64
      : fits(64, 3) ? 64 : fits(128, 2) ? 128 : fits(32, 4) ? 32 : fits(64, 2) ? 64 : 32;
  static constexpr int Stages = fits(K, 4) ? 4 : fits(K, 3) ? 3 : 2;
  static constexpr int WeightBytes = Width*(K + 8)*2, Buffer = buffer(K);
  static_assert(Threads%Width == 0 && fits(K, Stages));
};

template<int N, int K, int Rows, int Threads, int Width, int RowsPer, int Slices, int Bytes,
    int RowWords = (Rows + 31)/32>
__device__ __forceinline__ void simt(uint16_t* storage, const float* activation,
    const uint16_t* weight, void* workspace, float* output, uint32_t rows, int k0,
    uint32_t feature0, uint32_t tiles, uint32_t row0, int count,
    const uint32_t (&lost)[RowWords], bool lost_rows, int first) {
  using P = SimtPanel<Width, Bytes, Threads, RowsPer>;
  constexpr int SliceK = K/Slices, Panels = SliceK/P::K, ChainPanels = 128/P::K;
  uint8_t* bytes = reinterpret_cast<uint8_t*>(storage);
  auto row_of = [&](int r) {
    if (!lost_rows) return row0 + r;
    int n = first + r, base = 0;
#pragma unroll
    for (int w = 0; w < RowWords; ++w) {
      const int c = __popc(lost[w]);
      if (n < c) return row0 + base + nth_bit(lost[w], n);
      n -= c;
      base += 32;
    }
    return row0;
  };
  // Every stage commits one group, empty past the last panel.
  auto stage = [&](int panel) {
    if (panel < Panels) {
      uint8_t* buffer = bytes + panel%P::Stages*P::Buffer;
      uint16_t* w = reinterpret_cast<uint16_t*>(buffer);
      float* a = reinterpret_cast<float*>(buffer + P::WeightBytes);
      const int k = k0 + panel*P::K;
      for (int at = threadIdx.x; at < Width*P::K/8; at += Threads) {
        const int f = at/(P::K/8), c = at%(P::K/8)*8;
        const bool valid = (tiles >> (f/16)) & 1;
        copy16(w + f*(P::K + 8) + c,
            valid ? weight + uint64_t(feature0 + f)*K + k + c : weight, valid);
      }
      for (int at = threadIdx.x; at < P::Lines*P::K/4; at += Threads) {
        const int r = at/(P::K/4), c = at%(P::K/4)*4;
        const bool valid = r < count;
        copy16(a + r*(P::K + 4) + c,
            valid ? activation + uint64_t(row_of(r))*K + k + c : activation, valid);
      }
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  };
  const int f = threadIdx.x%Width, slot = threadIdx.x/Width;
  const bool feature = (tiles >> (f/16)) & 1;
  float total[RowsPer]{}, partial[RowsPer]{};
#pragma unroll
  for (int panel = 0; panel < P::Stages - 1; ++panel) stage(panel);
#pragma unroll 1
  for (int panel = 0; panel < Panels; ++panel) {
    asm volatile("cp.async.wait_group %0;" :: "n"(P::Stages - 2) : "memory");
    // This panel is visible, and every warp has finished the panel whose
    // slot the next copy refills.
    __syncthreads();
    stage(panel + P::Stages - 1);
    const uint8_t* buffer = bytes + panel%P::Stages*P::Buffer;
    const uint16_t* w = reinterpret_cast<const uint16_t*>(buffer) + f*(P::K + 8);
    const float* a = reinterpret_cast<const float*>(buffer + P::WeightBytes) + slot*(P::K + 4);
#pragma unroll 2
    for (int k = 0; k < P::K; k += 8) {
      const uint4 v = *reinterpret_cast<const uint4*>(w + k);
      const uint32_t words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int j = 0; j < RowsPer; ++j) {
        const float4 x0 = *reinterpret_cast<const float4*>(a + j*P::Slots*(P::K + 4) + k);
        const float4 x1 = *reinterpret_cast<const float4*>(a + j*P::Slots*(P::K + 4) + k + 4);
        const float x[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
        for (int e = 0; e < 8; ++e)
          partial[j] = __fmaf_rn(x[e], widen(uint16_t(words[e/2] >> (e%2*16))), partial[j]);
      }
    }
    if ((panel + 1)%ChainPanels == 0) {
#pragma unroll
      for (int j = 0; j < RowsPer; ++j) {
        const int r = slot + j*P::Slots;
        if constexpr (Slices == 1) {
          total[j] = __fadd_rn(total[j], partial[j]);
        } else if (feature && r < count) {
          const int chain = (k0 + panel*P::K)/128;
          partials<K>(workspace, rows)[(uint64_t(chain)*rows + row_of(r))*N + feature0 + f]
              = partial[j];
        }
        partial[j] = 0.f;
      }
    }
  }
  // The ring is free for the next call.
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncthreads();
  if constexpr (Slices == 1)
#pragma unroll
    for (int j = 0; j < RowsPer; ++j) {
      const int r = slot + j*P::Slots;
      if (feature && r < count) output[uint64_t(row_of(r))*N + feature0 + f] = total[j];
    }
}

// `simt` for one 16-feature tile at `count` rows from `row0`, when those
// rows leave thread slots idle: the slots split into `groups` (at most
// Groups) of `ceil(count / RowsPer)` slots, and group `z` runs the chains
// `z`, `z + groups`, ... of the slice, each from +0 in K order. After every
// round of `groups` chains the first group adds the round's chain values
// to its totals from the first chain on (or each group stores its own to
// their planes), so the sums keep the original order.
template<int Bytes, int Threads, int RowsPer>
struct SplitPanel {
  static constexpr int Slots = Threads/16;
  static constexpr int Lines = Slots*RowsPer;
  static constexpr int Values = Lines*16*4;
  static constexpr int buffer(int k, int groups) {
    return groups*16*(k + 8)*2 + Lines*(k + 4)*4;
  }
  static constexpr bool fits(int k, int stages, int groups) {
    return stages*buffer(k, groups) + Values <= Bytes;
  }
  static constexpr int most() {
    int groups = Slots;
    while (groups > 1 && !fits(32, 2, groups)) --groups;
    return groups;
  }
  static constexpr int Groups = most();
  static constexpr int K = fits(128, 3, Groups) ? 128 : fits(64, 3, Groups) ? 64
      : fits(128, 2, Groups) ? 128 : fits(64, 2, Groups) ? 64 : 32;
  static constexpr int Stages = fits(K, 3, Groups) ? 3 : 2;
  static constexpr int Region = 16*(K + 8)*2, Buffer = buffer(K, Groups);
  static_assert(Threads%16 == 0 && fits(K, Stages, Groups));
};

template<int N, int K, int Threads, int RowsPer, int Slices, int Bytes>
__device__ __forceinline__ void simt_split(uint16_t* storage, const float* activation,
    const uint16_t* weight, void* workspace, float* output, uint32_t rows, int k0,
    uint32_t feature0, uint32_t row0, int count, int groups) {
  using P = SplitPanel<Bytes, Threads, RowsPer>;
  constexpr int Chains = K/Slices/128, ChainPanels = 128/P::K;
  const int width = (count + RowsPer - 1)/RowsPer, lines = width*RowsPer;
  const int rounds = (Chains + groups - 1)/groups, panels = rounds*ChainPanels;
  uint8_t* bytes = reinterpret_cast<uint8_t*>(storage);
  float* values = reinterpret_cast<float*>(bytes + P::Stages*P::Buffer);
  // Every stage commits one group, empty past the last panel.
  auto stage = [&](int panel) {
    if (panel < panels) {
      uint8_t* buffer = bytes + panel%P::Stages*P::Buffer;
      const int round = panel/ChainPanels, part = panel%ChainPanels*P::K;
      for (int at = threadIdx.x; at < groups*16*P::K/8; at += Threads) {
        const int z = at/(16*P::K/8), f = at/(P::K/8)%16, c = at%(P::K/8)*8;
        const int chain = round*groups + z;
        const bool valid = chain < Chains;
        copy16(reinterpret_cast<uint16_t*>(buffer + z*P::Region) + f*(P::K + 8) + c,
            valid ? weight + uint64_t(feature0 + f)*K + k0 + chain*128 + part + c : weight,
            valid);
      }
      for (int at = threadIdx.x; at < groups*lines*P::K/4; at += Threads) {
        const int z = at/(lines*P::K/4), r = at/(P::K/4)%lines, c = at%(P::K/4)*4;
        const int chain = round*groups + z;
        const bool valid = chain < Chains && r < count;
        copy16(reinterpret_cast<float*>(buffer + P::Groups*P::Region)
            + (z*lines + r)*(P::K + 4) + c,
            valid ? activation + uint64_t(row0 + r)*K + k0 + chain*128 + part + c
                : activation, valid);
      }
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  };
  const int f = threadIdx.x%16, slot = threadIdx.x/16;
  const int u = slot%width, z = slot/width;
  const bool active = z < groups;
  float total[RowsPer]{}, partial[RowsPer]{};
#pragma unroll
  for (int panel = 0; panel < P::Stages - 1; ++panel) stage(panel);
#pragma unroll 1
  for (int panel = 0; panel < panels; ++panel) {
    asm volatile("cp.async.wait_group %0;" :: "n"(P::Stages - 2) : "memory");
    // This panel is visible, and every warp has finished the panel whose
    // slot the next copy refills.
    __syncthreads();
    stage(panel + P::Stages - 1);
    const uint8_t* buffer = bytes + panel%P::Stages*P::Buffer;
    const uint16_t* w = reinterpret_cast<const uint16_t*>(buffer + z*P::Region)
        + f*(P::K + 8);
    const float* a = reinterpret_cast<const float*>(buffer + P::Groups*P::Region)
        + (z*lines + u)*(P::K + 4);
    if (active) {
#pragma unroll 2
      for (int k = 0; k < P::K; k += 8) {
        const uint4 v = *reinterpret_cast<const uint4*>(w + k);
        const uint32_t words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int j = 0; j < RowsPer; ++j) {
          const float4 x0 = *reinterpret_cast<const float4*>(a + j*width*(P::K + 4) + k);
          const float4 x1 = *reinterpret_cast<const float4*>(a + j*width*(P::K + 4) + k + 4);
          const float x[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
          for (int e = 0; e < 8; ++e)
            partial[j] = __fmaf_rn(x[e], widen(uint16_t(words[e/2] >> (e%2*16))), partial[j]);
        }
      }
    }
    if ((panel + 1)%ChainPanels == 0) {
      const int chain = panel/ChainPanels*groups + z;
#pragma unroll
      for (int j = 0; j < RowsPer; ++j) {
        const int r = u + j*width;
        if constexpr (Slices == 1) {
          if (active) values[(z*lines + r)*16 + f] = partial[j];
        } else if (active && chain < Chains && r < count) {
          partials<K>(workspace, rows)[(uint64_t(k0/128 + chain)*rows + row0 + r)*N
              + feature0 + f] = partial[j];
        }
        partial[j] = 0.f;
      }
      if constexpr (Slices == 1) {
        __syncthreads();
        if (z == 0)
          for (int g = 0; g < groups && panel/ChainPanels*groups + g < Chains; ++g)
#pragma unroll
            for (int j = 0; j < RowsPer; ++j)
              total[j] = __fadd_rn(total[j], values[(g*lines + u + j*width)*16 + f]);
        __syncthreads();
      }
    }
  }
  // The ring is free for the next call.
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncthreads();
  if constexpr (Slices == 1)
    if (z == 0)
#pragma unroll
      for (int j = 0; j < RowsPer; ++j) {
        const int r = u + j*width;
        if (r < count) output[uint64_t(row0 + r)*N + feature0 + f] = total[j];
      }
}

template<int N, int K, int Features, int Rows, int WarpFeatures, int WarpRows, int Stages,
    int PlaneK, int Slices = 1>
__device__ void run(const float* activation, const uint16_t* weight, void* workspace,
    float* output, const uint32_t* live_rows, uint32_t rows, const uint32_t* weight_domain) {
  using T = Tile<Features, Rows, WarpFeatures, WarpRows, Stages, PlaneK>;
  static_assert(N > 0 && N%16 == 0 && K > 0 && K%128 == 0);
  static_assert(uint64_t(N)*K < (uint64_t(1) << 32));
  static_assert(Slices >= 1 && (K/128)%Slices == 0);
  constexpr int SliceK = K/Slices;
  constexpr int Planes = SliceK/PlaneK;
  const int k0 = blockIdx.z*SliceK;
  if (blockDim.x != T::Threads) __trap();
  extern __shared__ __align__(16) uint16_t storage[];
  const int thread = threadIdx.x, warp = thread/32, lane = thread%32;
  const int row0 = blockIdx.x*Rows, feature0 = blockIdx.y*Features;
  // Each thread copies one 16-byte column of PerThread fixed lines.
  const int column = (thread%T::LineChunks)*8;
  uint32_t source[T::PerThread];
  bool valid[T::PerThread];
#pragma unroll
  for (int j = 0; j < T::PerThread; ++j) {
    const int line = thread/T::LineChunks + j*(T::Threads/T::LineChunks);
    valid[j] = line < Features && feature0+line < N;
    source[j] = valid[j] ? uint32_t(feature0+line)*K + column : 0;
  }
  // Weights are immutable after the create-time classification, so the
  // first planes' weight lines load before the predecessor completes.
  auto stage_weights = [&](int plane, int slot) {
#pragma unroll
    for (int j = 0; j < T::PerThread; ++j) {
      const int line = thread/T::LineChunks + j*(T::Threads/T::LineChunks);
      if (line < Features)
        copy16(storage + slot*T::Plane + swizzle<PlaneK>(line, column),
            valid[j] ? weight + source[j] + k0 + plane*PlaneK : weight, valid[j]);
    }
  };
#pragma unroll
  for (int plane = 0; plane < Stages-1; ++plane)
    if (plane < Planes) stage_weights(plane, plane);
  ptx::griddep_wait();
  const uint32_t live = min(*live_rows, rows);
  if (row0 >= live) {
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
    ptx::griddep_launch_dependents();
    return;
  }
  prefill::Workspace<K> work(workspace, rows);
#pragma unroll
  for (int j = 0; j < T::PerThread; ++j) {
    const int line = thread/T::LineChunks + j*(T::Threads/T::LineChunks);
    if (line >= Features && line < T::Lines) {
      const int c = (line-Features)/Rows, row = row0 + (line-Features)%Rows;
      valid[j] = row < live && work.domain[row];
      source[j] = valid[j] ? (uint32_t(row)*3+c)*K + column : 0;
    }
  }
  auto stage_rows = [&](int plane, int slot) {
#pragma unroll
    for (int j = 0; j < T::PerThread; ++j) {
      const int line = thread/T::LineChunks + j*(T::Threads/T::LineChunks);
      if (line >= Features && line < T::Lines)
        copy16(storage + slot*T::Plane + swizzle<PlaneK>(line, column),
            valid[j] ? work.components + source[j] + k0 + plane*PlaneK : work.components,
            valid[j]);
    }
  };
#pragma unroll
  for (int plane = 0; plane < Stages-1; ++plane) {
    if (plane < Planes) stage_rows(plane, plane);
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
  const int wf = (warp%(Features/WarpFeatures))*WarpFeatures;
  const int wr = (warp/(Features/WarpFeatures))*WarpRows;
  ptx::D4 total[T::FeatureTiles][T::RowTiles]{};
  int slot = 0;
#pragma unroll 1
  for (int base = 0; base < SliceK; base += 128) {
    SeparateChains chains[T::FeatureTiles][T::RowTiles]{};
#pragma unroll
    for (int part = 0; part < T::ChainPlanes; ++part) {
      const int plane = base/PlaneK + part;
      asm volatile("cp.async.wait_group %0;" :: "n"(Stages-2) : "memory");
      // Every copy of this plane is visible, and every warp has finished
      // reading the slot the next copy refills.
      __syncthreads();
      const int next = plane + Stages-1;
      const int refill = slot == 0 ? Stages-1 : slot-1;
      if (next < Planes) {
        stage_weights(next, refill);
        stage_rows(next, refill);
      }
      asm volatile("cp.async.commit_group;" ::: "memory");
      const uint16_t* tile = storage + slot*T::Plane;
#pragma unroll
      for (int step = 0; step < PlaneK/16; ++step) {
        ptx::A4 a[T::FeatureTiles];
#pragma unroll
        for (int i = 0; i < T::FeatureTiles; ++i)
          ptx::ldmatrix_b16<4>(a[i].v, tile + swizzle<PlaneK>(
              wf + i*16 + lane%8 + ((lane/8)&1)*8, step*16 + (lane/16)*8));
        ptx::B2 b[3][T::RowTiles];
#pragma unroll
        for (int c = 0; c < 3; ++c) {
          if constexpr (T::RowTiles == 1) {
            ptx::ldmatrix_b16<2>(b[c][0].v, tile + swizzle<PlaneK>(
                Features + c*Rows + wr + lane%8, step*16 + ((lane/8)&1)*8));
          } else {
#pragma unroll
            for (int j = 0; j < T::RowTiles; j += 2) {
              uint32_t v[4];
              ptx::ldmatrix_b16<4>(v, tile + swizzle<PlaneK>(
                  Features + c*Rows + wr + j*8 + lane%8 + (lane/16)*8,
                  step*16 + ((lane/8)&1)*8));
              b[c][j].v[0] = v[0];
              b[c][j].v[1] = v[1];
              b[c][j+1].v[0] = v[2];
              b[c][j+1].v[1] = v[3];
            }
          }
        }
#pragma unroll
        for (int i = 0; i < T::FeatureTiles; ++i)
#pragma unroll
          for (int j = 0; j < T::RowTiles; ++j)
#pragma unroll
            for (int c = 0; c < 3; ++c)
              ptx::mma_bf16(chains[i][j].partial[c], a[i], b[c][j]);
      }
      slot = slot == Stages-1 ? 0 : slot+1;
    }
#pragma unroll
    for (int i = 0; i < T::FeatureTiles; ++i)
#pragma unroll
      for (int j = 0; j < T::RowTiles; ++j)
#pragma unroll
        for (int v = 0; v < 4; ++v) {
          if constexpr (Slices == 1) {
            total[i][j].v[v] = __fadd_rn(total[i][j].v[v], chains[i][j].merged(v));
          } else {
            const uint32_t row = row0 + wr + j*8 + (lane&3)*2 + (v&1);
            const uint32_t feature = feature0 + wf + i*16 + (lane>>2) + (v>>1)*8;
            if (row < live && feature < N)
              partials<K>(workspace, rows)[(uint64_t((k0+base)/128)*rows + row)*N + feature]
                  = chains[i][j].merged(v);
          }
        }
  }
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  bool fallback = false;
#pragma unroll
  for (int i = 0; i < T::FeatureTiles; ++i)
#pragma unroll
    for (int j = 0; j < T::RowTiles; ++j)
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        const uint32_t row = row0 + wr + j*8 + (lane&3)*2 + (v&1);
        const uint32_t feature = feature0 + wf + i*16 + (lane>>2) + (v>>1)*8;
        if (row >= live || feature >= N) continue;
        if (!weight_domain[feature/16] || !work.domain[row])
          fallback = true;
        else if constexpr (Slices == 1)
          output[uint64_t(row)*N + feature] = total[i][j].v[v];
      }
  // A row or weight tile outside the tensor domain keeps the original
  // SIMT chains (`simt`): first each tile outside the weight domain at
  // every live row, then the rows outside the row domain at the other
  // tiles' features, a CTA's worth of rows at a time.
  if (__syncthreads_or(fallback)) {
    static_assert(Features <= 32*16 && Features <= T::Threads);
    constexpr int RowWords = (Rows + 31)/32;
    uint32_t outside = 0, inside = 0, lost[RowWords]{};
#pragma unroll
    for (int g = 0; g < Features/16; ++g)
      if (feature0 + g*16 < N) {
        if (weight_domain[(feature0 + g*16)/16]) inside |= 1u << g;
        else outside |= 1u << g;
      }
    const int span = min(uint32_t(Rows), live - row0);
    int lost_rows = 0;
#pragma unroll
    for (int w = 0; w < RowWords; ++w) {
      for (int r = w*32; r < min(span, w*32 + 32); ++r)
        if (!work.domain[row0 + r]) lost[w] |= 1u << (r - w*32);
      lost_rows += __popc(lost[w]);
    }
    constexpr int RowsPer = (Rows*16 + T::Threads - 1)/T::Threads;
    const int groups = min(T::Threads/16/((span + RowsPer - 1)/RowsPer),
        SplitPanel<Stages*T::Plane*2, T::Threads, RowsPer>::Groups);
    for (int i = 0; i < __popc(outside); ++i) {
      const uint32_t tile = feature0 + nth_bit(outside, i)*16;
      if (groups > 1)
        simt_split<N, K, T::Threads, RowsPer, Slices, Stages*T::Plane*2>(storage, activation,
            weight, workspace, output, rows, k0, tile, row0, span, groups);
      else
        simt<N, K, Rows, T::Threads, 16, RowsPer, Slices, Stages*T::Plane*2>(storage,
            activation, weight, workspace, output, rows, k0, tile, 1u, row0, span, lost,
            false, 0);
    }
    if (inside != 0)
      for (int first = 0; first < lost_rows; first += T::Threads/Features)
        simt<N, K, Rows, T::Threads, Features, 1, Slices, Stages*T::Plane*2>(storage,
            activation, weight, workspace, output, rows, k0, feature0, inside, row0,
            min(T::Threads/Features, lost_rows - first), lost, true, first);
  }
  ptx::griddep_launch_dependents();
}

}  // namespace chop::exact_f32::rows
