// DeepSeek V4.1's own kernels: the Engram gate and the RoPE cosine and sine
// rows. Contracts and CPU oracles: model-deepseek41-contract's own_ops
// module. Self-contained: it includes no shared kernel header.
#pragma once

#include <cuda_bf16.h>
#include <stdint.h>

// The RoPE frequencies' F32 bits, pair i of the window table as DS41_RWi
// and of the compressed table as DS41_RCi, and the position domain
// DS41_ROPE_POSITIONS, defined by the build from the contract.
#if !defined(DS41_RW31) || !defined(DS41_RC31) || !defined(DS41_ROPE_POSITIONS)
#error "the build defines DS41_RW0..31, DS41_RC0..31 and DS41_ROPE_POSITIONS"
#endif

namespace chop {
namespace deepseek41 {

constexpr int OWN_THREADS = 256;

// Every own kernel launches with programmatic stream serialization, as the
// step's other launches do: it waits for the previous grid's results before
// it reads or writes anything. Without that attribute the wait returns at
// once.
__device__ __forceinline__ void wait_for_previous_grid() {
    asm volatile("griddepcontrol.wait;" ::: "memory");
}

__device__ __forceinline__ float bf16_bits_to_f32(uint16_t bits) {
    return __uint_as_float(uint32_t(bits) << 16);
}

__device__ __forceinline__ uint16_t f32_to_bf16_bits(float value) {
    return __bfloat16_as_ushort(__float2bfloat16_rn(value));
}

// Sums three per-thread values over the block in one fixed order: a warp
// butterfly (every lane ends with the same sum, since F32 addition
// commutes), then the warps' sums in warp order.
__device__ __forceinline__ void block_sum3(float& a, float& b, float& c) {
    __shared__ float partial[3][OWN_THREADS / 32];
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        a = __fadd_rn(a, __shfl_xor_sync(0xffffffffu, a, mask));
        b = __fadd_rn(b, __shfl_xor_sync(0xffffffffu, b, mask));
        c = __fadd_rn(c, __shfl_xor_sync(0xffffffffu, c, mask));
    }
    const int warp = threadIdx.x / 32;
    if ((threadIdx.x & 31) == 0) {
        partial[0][warp] = a;
        partial[1][warp] = b;
        partial[2][warp] = c;
    }
    __syncthreads();
    a = partial[0][0];
    b = partial[1][0];
    c = partial[2][0];
#pragma unroll
    for (int w = 1; w < OWN_THREADS / 32; ++w) {
        a = __fadd_rn(a, partial[0][w]);
        b = __fadd_rn(b, partial[1][w]);
        c = __fadd_rn(c, partial[2][w]);
    }
}

// The Engram gate: one block per (row, stream), grid [rows, STREAMS].
// Every thread reads its elements of the stream before the block sum and
// writes them after it, so `output` may be `x`; neither is `__restrict__`.
template <int WIDTH, int STREAMS, bool MASKED>
__device__ __forceinline__ void engram_gate(const uint16_t* x_in,
                                            const float* __restrict__ kv,
                                            const uint16_t* __restrict__ q_weight,
                                            const uint16_t* __restrict__ k_weight,
                                            const uint8_t* __restrict__ text,
                                            const uint32_t* __restrict__ live_rows,
                                            uint16_t* output) {
    static_assert(WIDTH % OWN_THREADS == 0, "width is a multiple of the block");
    constexpr int PER = WIDTH / OWN_THREADS;
    wait_for_previous_grid();
    const uint32_t row = blockIdx.x;
    const uint32_t stream = blockIdx.y;
    if (row >= *live_rows) {
        return;
    }
    const uint16_t* x = x_in + (size_t(row) * STREAMS + stream) * WIDTH;
    const float* kv_row = kv + size_t(row) * (STREAMS + 1) * WIDTH;
    const float* key = kv_row + size_t(stream) * WIDTH;
    const float* value = kv_row + size_t(STREAMS) * WIDTH;
    const uint16_t* qw = q_weight + size_t(stream) * WIDTH;
    const uint16_t* kw = k_weight + size_t(stream) * WIDTH;

    float h[PER];
    float hh = 0.0f, kk = 0.0f, dot = 0.0f;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int i = threadIdx.x + j * OWN_THREADS;
        h[j] = bf16_bits_to_f32(x[i]);
        const float k = key[i];
        const float q = bf16_bits_to_f32(qw[i]);
        const float kw_i = bf16_bits_to_f32(kw[i]);
        // Non-finite gate inputs poison only this stream's reduction.
        if (!isfinite(h[j]) || !isfinite(k) || !isfinite(q) || !isfinite(kw_i)) {
            hh = __int_as_float(0x7fffffff);
        }
        const float w = __fmul_rn(q, kw_i);
        hh = __fmaf_rn(h[j], h[j], hh);
        kk = __fmaf_rn(k, k, kk);
        dot = __fmaf_rn(__fmul_rn(h[j], w), k, dot);
    }
    block_sum3(hh, kk, dot);

    const float n = float(WIDTH);
    const float r = __fmul_rn(__frcp_rn(__fsqrt_rn(__fadd_rn(__fdiv_rn(hh, n), 1e-20f))),
                              __frcp_rn(__fsqrt_rn(__fadd_rn(__fdiv_rn(kk, n), 1e-20f))));
    const float scaled = __fmul_rn(__fmul_rn(dot, r), __frsqrt_rn(n));
    // fmaxf would turn NaN into the finite clamp and invent an open gate.
    const float z = isnan(scaled) ? scaled
        : copysignf(__fsqrt_rn(fmaxf(fabsf(scaled), 1e-6f)), scaled);
    float gate = __frcp_rn(__fadd_rn(1.0f, expf(-z)));
    if (MASKED && text[row] == 0) {
        gate = 0.0f;
    }
    uint16_t* out = output + (size_t(row) * STREAMS + stream) * WIDTH;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int i = threadIdx.x + j * OWN_THREADS;
        out[i] = f32_to_bf16_bits(__fmaf_rn(gate, value[i], h[j]));
    }
}

__constant__ uint32_t rope_window_bits[32] = {
    DS41_RW0, DS41_RW1, DS41_RW2, DS41_RW3, DS41_RW4, DS41_RW5, DS41_RW6, DS41_RW7,
    DS41_RW8, DS41_RW9, DS41_RW10, DS41_RW11, DS41_RW12, DS41_RW13, DS41_RW14,
    DS41_RW15, DS41_RW16, DS41_RW17, DS41_RW18, DS41_RW19, DS41_RW20, DS41_RW21,
    DS41_RW22, DS41_RW23, DS41_RW24, DS41_RW25, DS41_RW26, DS41_RW27, DS41_RW28,
    DS41_RW29, DS41_RW30, DS41_RW31};
__constant__ uint32_t rope_compressed_bits[32] = {
    DS41_RC0, DS41_RC1, DS41_RC2, DS41_RC3, DS41_RC4, DS41_RC5, DS41_RC6, DS41_RC7,
    DS41_RC8, DS41_RC9, DS41_RC10, DS41_RC11, DS41_RC12, DS41_RC13, DS41_RC14,
    DS41_RC15, DS41_RC16, DS41_RC17, DS41_RC18, DS41_RC19, DS41_RC20, DS41_RC21,
    DS41_RC22, DS41_RC23, DS41_RC24, DS41_RC25, DS41_RC26, DS41_RC27, DS41_RC28,
    DS41_RC29, DS41_RC30, DS41_RC31};

// The vendor's F32 angle, then its cosine and sine in f64 rounded once to
// F32: correctly rounded at every position of the contract's domain.
template <bool COMPRESSED>
__device__ __forceinline__ void rope_pair(uint32_t position, uint32_t pair, float* out) {
    const uint32_t bits = COMPRESSED ? rope_compressed_bits[pair] : rope_window_bits[pair];
    const float angle = __fmul_rn(__uint2float_rn(position), __uint_as_float(bits));
    double s, c;
    sincos(double(angle), &s, &c);
    out[0] = __double2float_rn(c);
    out[1] = __double2float_rn(s);
}

// Rows `first + r` for r below `rows`; one thread per (row, pair).
template <bool COMPRESSED>
__device__ __forceinline__ void rope_range(uint32_t first, uint32_t rows, float* output) {
    wait_for_previous_grid();
    const uint32_t t = blockIdx.x * OWN_THREADS + threadIdx.x;
    const uint32_t row = t / 32;
    if (row >= rows) {
        return;
    }
    rope_pair<COMPRESSED>(first + row, t % 32, output + size_t(t) * 2);
}

// Rows at `positions[r]` for r below `*live_rows`. A position past the
// contract's domain writes NaN cosine and sine.
template <bool COMPRESSED>
__device__ __forceinline__ void rope_positions(const uint32_t* __restrict__ positions,
                                               const uint32_t* __restrict__ live_rows,
                                               float* output) {
    wait_for_previous_grid();
    const uint32_t t = blockIdx.x * OWN_THREADS + threadIdx.x;
    const uint32_t row = t / 32;
    if (row >= *live_rows) {
        return;
    }
    const uint32_t position = positions[row];
    if (position >= DS41_ROPE_POSITIONS) {
        output[size_t(t) * 2] = __int_as_float(0x7fffffff);
        output[size_t(t) * 2 + 1] = __int_as_float(0x7fffffff);
        return;
    }
    rope_pair<COMPRESSED>(position, t % 32, output + size_t(t) * 2);
}

}  // namespace deepseek41
}  // namespace chop
