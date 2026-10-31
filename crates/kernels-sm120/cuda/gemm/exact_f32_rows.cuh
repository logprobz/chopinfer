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

// The chain planes of a sliced projection follow the preparation's row
// domain and components in the workspace, 16-byte aligned:
// partial[(chain*rows + row)*N + feature], the cached reducer's layout.
template<int K>
__device__ __forceinline__ float* partials(void* workspace, uint32_t rows) {
  return reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(workspace)
      + (uint64_t(rows)+3)/4*16 + uint64_t(rows)*K*6);
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
  bool simt = false;
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
          simt = true;
        else if constexpr (Slices == 1)
          output[uint64_t(row)*N + feature] = total[i][j].v[v];
      }
  // A row or weight tile outside the tensor domain keeps the original
  // SIMT chains: FMAs from +0 in K order within each 128-K chain, added
  // into the output from +0 in chain order.
  if (__syncthreads_or(simt)) {
#pragma unroll 1
    for (int at = thread; at < Features*Rows; at += T::Threads) {
      const uint32_t row = row0 + at/Features, feature = feature0 + at%Features;
      if (row >= live || feature >= N || (weight_domain[feature/16] && work.domain[row]))
        continue;
      float result = 0.f;
#pragma unroll 1
      for (int base = k0; base < k0+SliceK; base += 128) {
        float partial = 0.f;
#pragma unroll 1
        for (int k = base; k < base+128; ++k)
          partial = __fmaf_rn(activation[uint64_t(row)*K + k],
              widen(weight[uint64_t(feature)*K + k]), partial);
        if constexpr (Slices == 1)
          result = __fadd_rn(result, partial);
        else
          partials<K>(workspace, rows)[(uint64_t(base/128)*rows + row)*N + feature] = partial;
      }
      if constexpr (Slices == 1) output[uint64_t(row)*N + feature] = result;
    }
  }
  ptx::griddep_launch_dependents();
}

}  // namespace chop::exact_f32::rows
