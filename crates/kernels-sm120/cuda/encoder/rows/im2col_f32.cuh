#pragma once

// Rows of a 1D convolution's taps over F32 frames, kept in F32
// (`conv1d.im2col.v1` under `f32-pairs.v1`).

#include "common.cuh"

namespace chop::multimodal {

template<int Channels, int Kernel, int Stride, int PadLeft, int PadRight>
__device__ void im2col_f32(const float* input, float* output,
                           const uint32_t* live_frames) {
  begin();
  const uint32_t frames = *live_frames;
  const uint32_t rows = (frames + PadLeft + PadRight - Kernel) / Stride + 1;
  const uint32_t row = blockIdx.x;
  if (row >= rows) return;
  for (int col = threadIdx.x; col < Kernel * Channels; col += blockDim.x) {
    const int frame = int(row * Stride + col / Channels) - PadLeft;
    output[uint64_t(row) * Kernel * Channels + col] =
        frame >= 0 && uint32_t(frame) < frames ?
            input[uint64_t(frame) * Channels + col % Channels] : 0.f;
  }
}

} // namespace chop::multimodal
