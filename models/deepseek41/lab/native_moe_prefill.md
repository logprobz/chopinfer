# Native MoE prefill qualification

This candidate changes bits. It runs DeepSeek V4.1 Flash TP4's routed
expert tiles (gate/up and down, 384 experts, top 6) in every prefill step
through native `mma.sync kind::mxf8f6f4.block_scale` MMA on the trained
operands: MXFP4 weights (E2M1, UE8M0 per 32) by the E4M3/32 ceil-UE8M0
activations the serving quantizers already produce. Every product is exact.
Each 128-K chain starts from +0 and is added into the F32 accumulator in
ascending K order, the same fold structure as the exact BF16 engine; inside
a 32-K group the instruction does its own reduction. Routing, the gate/up
epilogue (clamp, SiLU, route weight, BF16), the intermediate ceil quantizer
and the routed sum are unchanged. It is opt-in
(`CHOPINFER_DEEPSEEK41_NATIVE_MOE_PREFILL=1`) and qualifies on the
within-bound track of docs/numerics.md, "A speedup that changes bits".

The kernels are model-neutral (`kernels_sm120::moe::native_grouped`): any
expert count, top-k, K and N (K and N multiples of 32 and 16), with an
active-expert tile queue and a persistent grid taking items from a ticket.

## Decision on the no-larger-error predicate (2026-10-20)

Recorded before any result of this candidate exists.

An earlier native-down candidate declared, before measurement, that per
layer, rank and prompt its RMSE and maximum absolute error against f64
must be no larger than the exact down's, with no tolerance, for the routed
F32 output and for the ordered top-k sum with the real shared addend, and
that an unresolved failure blocks promotion. That declared criterion blocks
that candidate: a partial replay over layers 0 to 19 (400 cases) found 81
ordered-sum cases with a larger RMSE (largest ratio 1.000117, no larger
maximum error or ULP), while its native down itself was never worse. The
criterion is neither relaxed nor removed for that candidate, which stays
unselected, and its pending jobs would add evidence only about a kernel
this candidate replaces.

A different accumulation structure cannot pass that predicate honestly.
That design already keeps the 128-K fold boundaries with F32 partials
(four native 32-K instructions per chain, then one ordered F32 add). The
failures appear one stage later: the ordered sum rounds in F32, and any
change in a routed output's low bits moves those roundings, so the sum's
error against f64 rises or falls by rounding noise whether or not down
became more accurate. Native down was at least as accurate in all 400
cases and the sum still failed in 20% of them. Only arithmetic that
reproduces the exact bits is sure to pass, and native MMA cannot. Trying
variants until one happens to pass would select on the result.

So the native path is a separate within-bound candidate. Its criteria are
the within-bound track of docs/numerics.md, fixed before that replay's
result and made concrete below. The no-larger-error counts remain reported
for every stage as a diagnostic, so nothing measured is hidden; they are
not a gate for this candidate.

## Criteria, fixed before measurement

Every GPU check runs with `CHOPINFER_KERNEL_REQUIRE=1` on idle GPUs, prints
its executed cases, and fails on zero executed tests or cases.

1. **Instruction exactness.** One instruction from +0 with one nonzero
   product per output: every finite E4M3 code (including subnormals) by
   every E2M1 code, at several UE8M0 scale pairs and in each of the 32 K
   slots, equals the exactly representable product bit for bit, except
   that a zero product may come back as either signed zero (both compare
   equal and add identically into a +0 chain). Sums of small integers
   whose exact sum is an F32 value equal it bit for bit.
   This also checks the fragment layout the host emulation
   (`crates/kernels-sm120/tests/native_grouped_layout.cpp`) assumes.
2. **f64 oracle bound** `|y - y_ref| <= 2^-22 * K * S` (docs/numerics.md,
   zero BF16 allowance since the checked values are F32) on every element
   of the sampled rows, for the gate and up F32 sums (the lab's raw entry)
   and the down F32 output. Sampled rows: 32 evenly spaced live tokens
   (first and last included, every row below 32) with all their routes and
   all output columns. Zero failures. Per-route cosine against f64 at
   least 0.9999. Cases: synthetic E384/top-6 balanced and skewed routing
   and E128/top-3 skewed at 8192 rows, and real captured operands from the
   published capture: layers 0, 13, 14 and 39 (the layers whose rank
   banks were extracted and sealed), all four ranks, the 8192-row chunk
   of prompt input-0 and the 65- and 257-row tails of prompts input-1 and
   input-2. Each case first reproduces the captured exact intermediate,
   codes and sampled down rows bit for bit, which authenticates its banks.
3. **Serving entries.** The serving gate/up entry's BF16 intermediate
   equals the raw lab entry's bit for bit, and every compiled tile variant
   gives the same bits for every output.
4. **Invariance, bitwise.** Live-row prefixes (8192, 4097, 65 and 1 rows
   of the same inputs) give the same bits for every common row; graph and
   eager launches and poisoned repeats give the same bits.
5. **Negative control.** The omit-first-chain entry must fail check 2.
6. **Memory safety.** compute-sanitizer memcheck reports zero errors on
   both devices over the synthetic routing at 1, 65, 513 and 2048 live
   rows of the 8192-row capacity, every tile variant and the probe.
7. **Report only, not a gate:** RMSE, maximum absolute error and the
   no-larger-error counts against the exact path at down, the full chain
   and the ordered top-k sum with the shared addend, by layer and rank.
8. Then, in order: a report-only serving A/B (`"base_new": "report-only"`,
   repeats and AR against speculative still bitwise), and paired_teacher
   against the stored 2026-10-20 baseline for tp4-ar and tp4-dspark with
   the fixed 0.005-nat margins, exit 0 for both profiles.

No limit, margin, case list or reference changes after a result. Timings
count only after checks 1 to 6 pass.

## Floor and expected time per 8192-row chunk

Per rank and chunk (40 layers): useful expert work is 34.79 TFLOP (gate/up
23.19, down 11.60). At the FP8-class rate of the 1.6 GHz reference clock
(616 TFLOP/s; the mixed E2M1 x E4M3 form runs at the FP8 rate, not the FP4
one) the MMA-only floor is 56.5 ms (gate/up 37.6, down 18.8). Memory: the
weight banks are 1.20 GB (gate/up) and 0.60 GB (down) per layer with
scales, read once when every expert is active, and down writes 1.01 GB of
F32 routed rows per layer; with their inputs and outputs, at 1.63 TB/s
that is 32 ms for gate/up and 40 ms for down per chunk. Gate/up is
compute-bound and down is memory-bound, so the combined floor is about
37.6 + 40 = 78 ms.

Expected, before measurement: gate/up at about 55% of the FP8 reference
rate (1.7 ms per layer, 68 ms per chunk) and down at about 75% of the
memory rate (1.3 ms per layer, 53 ms per chunk), about 120 ms per full
8192-row chunk; the plausible range is 90 to 170 ms. The current serving
kernels take 137.6 ms for gate/up (89.5, 33.1, 8.8 and 6.2 ms in its four
stages, including the activation quantizer this candidate keeps) and 97.7
ms for down per 8192 rows in the GPU-exclusive profile of chunk 0 of a
cold 32K prompt on the round 7 head (2026-10-20), 235 ms together, so the
expected saving is about 110 ms per full chunk. Paired lab timings of the
same kernels on captured layers 14 and 39 (5.15 to 5.62 ms gate/up and
3.20 to 3.48 ms down per layer) are higher than their profile share and
are not the baseline here. Another implementation of the same E2M1 x E4M3
instruction with live-tile queues takes 128 ms for routed experts (gate/up
76, down 46, activation quantize 5.3) per 8192 rows on this hardware,
which supports the estimate's order. Upper layers skipped at a prompt's
last chunk scale both sides. The router, ceil and combine are unchanged.
These are estimates from the floor model and those measurements, not
results of this candidate; the timing job measures it, and its timings
count only after criteria 1 to 6 pass.
