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
The instruction is `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X`
with shape `m16n8k64`, E2M1 operands, UE8M0 scales and F32 input/output
accumulators. Two K64 operations share the same F32 accumulator. The
[CUDA 13.2 PTX MMA specification](https://docs.nvidia.com/cuda/archive/13.2.0/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma)
explicitly includes E2M1 in its at-least-single-precision accumulation rule.
Order, rounding and subnormal-input handling remain unspecified. The
[block-scaling definition](https://docs.nvidia.com/cuda/archive/13.2.0/parallel-thread-execution/index.html#warp-level-matrix-instructions-block-scaling)
applies scales to operands before multiplication.

Exact scaled E2M1 products are the proof's separate premise. Their integer
significands fit in F32, but the specification does not name an intermediate
product format, and raw E2M1 magnitude 0.5 is subnormal even when its scaled
value is normal. GPU probes therefore include every pair of E2M1 codes at
the scale edges, including both signs of 0.5. This is a hardware qualification
obligation, not an additional guarantee inferred from the accumulation rule.

The unguarded `2^25 + 1 - 2^25` witness is outside this domain. Reordering
full-F32 additions can produce either zero or one, so the witness does not
demonstrate reduced accumulator width. Within this domain no such rounding
is needed. Edge probes use a positive single-head dot with unit effective
weight, compare its raw F32 word to an integer/f64 oracle, and assert that
accepted cases do not use SIMT repair. Spreads 7, 8 and 9 test the admitted
boundary; adjacent rejected scales test repair. A separate spread-10 probe
whose coefficient fits in 24 bits tests the accumulator premise without
expanding serving admission. The nine-bit bound remains unchanged.

Only new symbols and a new lock family use this source. The old native entry
remains available to the lab as the negative control and timing baseline.
The optional diagnostic pointer counts admitted scores and SIMT repairs;
serving passes a compile-time null pointer and contains no counter atomics.
