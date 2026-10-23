#pragma once
// The target sampler's Gumbel noise on one rank's shard of F32 drafter
// logits (`kernel_core::draft_noise`, `row::draft_noise`), in place. The
// grid's x index is the row and its y index a tile of `Tile` columns; rows
// at or past the device live count, rows whose temperature is not positive
// and columns at or past the live width keep their bits.
//
// A sampled row's column becomes `x / T + g(t)` for global token
// `t = vocab_offset + column`, with the device sampler's pruning noise
// (`sampling::approximate_gumbel`) at the generated index
// `generated + position`: the noise the target samples that draft position
// with, to `kernel_core::draft_noise::NOISE_ERROR`.
#include "row.cuh"
#include "../sampling/sampling.cuh"

namespace chop::row::draft_noise {

// Little endian, as `kernel_core::draft_noise::NoiseRecord::encode`
// writes it.
struct Noise {
  uint64_t seed;
  uint64_t generated;
  double temperature;
  uint64_t reserved;
};
static_assert(sizeof(Noise) == 32, "draft noise records match the host's");

template <uint32_t Tile>
__device__ void perturb(float* logits, const Noise* noise,
                        const uint32_t* live_rows, const uint32_t* live_width,
                        uint32_t vocab, uint32_t vocab_offset,
                        uint32_t position) {
  begin();
  const uint32_t row = blockIdx.x;
  if (row >= *live_rows) return;
  const Noise record = noise[row];
  if (!(record.temperature > 0.0)) return;
  const uint32_t width = max(min(*live_width, vocab), 1u);
  const uint32_t first = blockIdx.y * Tile;
  if (first >= width) return;
  const uint32_t end = min(first + Tile, width);
  const uint64_t generated = record.generated + position;
  const float temperature = static_cast<float>(record.temperature);
  float* in = logits + uint64_t(row) * vocab;
  for (uint32_t col = first + threadIdx.x; col < end; col += blockDim.x) {
    const float scaled = in[col] / temperature;
    in[col] = scaled + sampling::approximate_gumbel(record.seed, generated,
                                                    vocab_offset + col);
  }
}

} // namespace chop::row::draft_noise
