# Exact native position scores

The guarded scorer uses the native dot only when `exact_scales` accepts every
head for that row and slot. Otherwise a whole warp recomputes that score with
the original four sequential FMAs per lane, shuffle-down offsets 16, 8, 4, 2,
1, and ascending head FMAs. Selection, holes, candidate provenance and tail
stores keep the existing behavior. Query repacking must still reproduce every
BF16 bit; a failed repack uses the original SIMT row entry.

For one head let `s[g] = query_scale[g] + key_scale[g]`, for four groups of
32 channels. UE8M0 bytes in [2, 252] give finite, normal decoded nonzero
operands, each exactly an E2M1 integer in [-12, 12] times `2^(scale-128)`.
Every product is therefore an integer times `u = 2^(min(s)-256)`.
With `max(s)-min(s) <= 9`, the absolute coefficient sum is at most
`128 * 144 * 2^9 = 9,437,184 < 2^24`. Every partial sum, in any order, is an
exactly representable F32 multiple of u. Cancellation does not weaken this
bound. Requiring `min(s) >= 130` keeps u normal. Requiring `max(s) <= 369`
bounds the sum of absolute products by `4608 * 2^115 < F32_MAX`.

Thus products and all intermediate sums are exact, normal or zero and finite.
The native dot and the SIMT dot have the same value. A zero's sign cannot affect
the score because both use `dot > 0 ? dot : +0` before the unchanged head FMA.
This is a sufficient domain, not a necessary one. Scale bytes 0, 1, 253, 254
and 255, wide spreads, and extreme products all use SIMT. Invalid inputs and
intermediate overflow retain the SIMT failure propagation.

The proof does not assume a particular tensor-core pairing or rounding mode.
The [PTX MMA specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma)
guarantees at least single-precision accumulation but leaves its order,
rounding and subnormal handling unspecified. The accepted domain requires no
rounding and no subnormal arithmetic. The 9-bit spread is deliberately more
conservative than a magnitude-dependent test near 11 bits.

Only new symbols and a new lock family use this source. The old native entry
remains available to the lab as the negative control and timing baseline.
The optional diagnostic pointer counts admitted scores and SIMT repairs;
serving passes a compile-time null pointer and contains no counter atomics.
