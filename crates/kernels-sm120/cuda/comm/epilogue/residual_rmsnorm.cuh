#pragma once

// MiMo's fused all-reduce epilogue: the residual add and HF RMSNorm of
// BF16 rows of Width columns, one residual stream per row. The transport
// (comm.cuh, push.cuh) sums every rank's input in rank order in F32 and
// hands each row to this functor:
//   carry      adds the residual to the sum and rounds to BF16; the
//              transport keeps that pre-norm row in residual_output (keep,
//              kept) and may exchange it between ranks
//   statistic  the row statistic the transport sums in a fixed order: the
//              squares of the rounded values
//   finish     normalizes with the summed statistic (scale), rounding to
//              BF16 before the weight multiply, and writes `output`
// The norm weights are the row's constant parameters, read before the PDL
// wait. Every rank applies the same expressions in the same order, so the
// outputs are bitwise identical on all ranks. The scalar `_element` forms
// serve the rank-generic v1 entry and the local decode control.

#include "../comm.cuh"

namespace chop::comm::epilogue {

template <uint32_t Width, uint32_t Streams>
struct ResidualRmsNorm {
  static_assert(Streams == 1, "one residual stream per row");
  static constexpr uint32_t kWidth = Width;

  uint16_t* output;
  const uint16_t* residual;
  const uint16_t* weights;
  uint32_t eps_bits;
  uint16_t* residual_output;

  // Vector forms: eight BF16 columns per Vector128, `index` counts vectors
  // from the first row. The row's parameters are its norm weights.
  __device__ __forceinline__ const Vector128* parameters() const {
    return reinterpret_cast<const Vector128*>(weights);
  }
  __device__ __forceinline__ Vector128 operand(uint64_t index) const {
    return reinterpret_cast<const Vector128*>(residual)[index];
  }

  // Adds the residual after the rank-ordered sum and rounds to BF16.
  __device__ __forceinline__ static Vector128 carry(const float (&sums)[8], Vector128 residual,
                                                    float& square_sum) {
    const uint32_t words[4] = {residual.x, residual.y, residual.z, residual.w};
    uint32_t out[4];
#pragma unroll
    for (int word = 0; word < 4; ++word) {
      const uint16_t low = encode<uint16_t>(sums[2 * word] + decode(static_cast<uint16_t>(words[word])));
      const uint16_t high = encode<uint16_t>(sums[2 * word + 1] +
                                             decode(static_cast<uint16_t>(words[word] >> 16)));
      square_sum += decode(low) * decode(low) + decode(high) * decode(high);
      out[word] = static_cast<uint32_t>(low) | (static_cast<uint32_t>(high) << 16);
    }
    return {out[0], out[1], out[2], out[3]};
  }

  __device__ __forceinline__ void keep(uint64_t index, Vector128 value) const {
    reinterpret_cast<Vector128*>(residual_output)[index] = value;
  }
  __device__ __forceinline__ Vector128 kept(uint64_t index) const {
    return reinterpret_cast<const Vector128*>(residual_output)[index];
  }

  __device__ __forceinline__ static float statistic(Vector128 value) {
    const uint32_t words[4] = {value.x, value.y, value.z, value.w};
    float sum = 0.0f;
#pragma unroll
    for (int word = 0; word < 4; ++word) {
      const float low = decode(static_cast<uint16_t>(words[word]));
      const float high = decode(static_cast<uint16_t>(words[word] >> 16));
      sum += low * low + high * high;
    }
    return sum;
  }

  __device__ __forceinline__ static float inverse(float square_sum, float eps) {
    return 1.0f / sqrtf(square_sum / static_cast<float>(Width) + eps);
  }
  __device__ __forceinline__ float scale(float square_sum) const {
    return inverse(square_sum, __uint_as_float(eps_bits));
  }

  // HF RMSNorm: round the normalized value to BF16 before the weight multiply.
  __device__ __forceinline__ static Vector128 normalize(Vector128 value, Vector128 weight,
                                                        float inv) {
    const uint32_t words[4] = {value.x, value.y, value.z, value.w};
    const uint32_t scale[4] = {weight.x, weight.y, weight.z, weight.w};
    uint32_t out[4];
#pragma unroll
    for (int word = 0; word < 4; ++word) {
      const float low = decode(encode<uint16_t>(decode(static_cast<uint16_t>(words[word])) * inv));
      const float high = decode(encode<uint16_t>(decode(static_cast<uint16_t>(words[word] >> 16)) * inv));
      out[word] = static_cast<uint32_t>(encode<uint16_t>(low * decode(static_cast<uint16_t>(scale[word])))) |
                  (static_cast<uint32_t>(encode<uint16_t>(high * decode(static_cast<uint16_t>(scale[word] >> 16)))) << 16);
    }
    return {out[0], out[1], out[2], out[3]};
  }
  __device__ __forceinline__ void finish(uint64_t index, Vector128 value, Vector128 weight,
                                         float inv) const {
    reinterpret_cast<Vector128*>(output)[index] = normalize(value, weight, inv);
  }

  // The rounding of `carry` for one column whose residual is already loaded.
  __device__ __forceinline__ static uint16_t carry_scalar(float sum, uint16_t residual) {
    return encode<uint16_t>(sum + decode(residual));
  }

  // Scalar forms by column `i` from the first row: an absent residual adds
  // nothing and an absent residual_output keeps nothing.
  __device__ __forceinline__ uint16_t carry_element(float value, uint64_t i,
                                                    float& square_sum) const {
    if (residual) value += decode(residual[i]);
    const uint16_t rounded = encode<uint16_t>(value);
    if (residual_output) residual_output[i] = rounded;
    const float x = decode(rounded);
    square_sum += x * x;
    return rounded;
  }
  // `col` is the column within the row, which picks the weight.
  __device__ __forceinline__ void finish_element(uint64_t i, uint32_t col, uint16_t value,
                                                 float inv) const {
    const float normalized = decode(encode<uint16_t>(decode(value) * inv));
    output[i] = encode<uint16_t>(normalized * decode(weights[col]));
  }
};

} // namespace chop::comm::epilogue
