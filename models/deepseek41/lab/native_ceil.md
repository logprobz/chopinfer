# Fused ceil qualification

This candidate keeps every bit. It removes the separate intermediate
quantizer launch after DeepSeek V4.1 Flash TP4's native gate/up tiles
(`chop_<tag>_moe_ceil_w3456_v1`, E4M3 per 32 columns with a ceil UE8M0
scale and a 1e-4 amax floor) by quantizing each tile's BF16 outputs in the
tiles' epilogue (`kernels_sm120::moe::native_grouped::Ceil`,
`cuda/moe/native_ceil.cuh`). The quantizer decides a whole token row
(top_k routes of 576 columns) at once: any group's flag, or a status bit
already set, gives the row zero codes and scale 127. A tile holds part of a
row, so it writes each group as if the row were clean and ORs the group's
flags into the token's status; a fixup launch after the tiles rewrites
every live token whose status is set. It is on the bitwise track: the
evidence is equality with the native tiles followed by the serving
quantizer.

The native tiles are the within-bound candidate of native_moe_prefill.md;
this qualification is relative to them and does not change their track.

This tree carries the fused entries, their lock and the serving binding
(`sm120::moe_native`, steps of at most 2048 rows); the qualification tests
and job runner build on the lab's CPU emulation of the native tiles and
land separately. This file records their criteria and result.

## Criteria, fixed before measurement

Every GPU check runs with `CHOPINFER_KERNEL_REQUIRE=1` on idle GPUs, prints
its executed cases, and fails on zero executed tests or cases.

1. **Bitwise.** For every case and both prior status patterns (all clear;
   every 97th token from the fifth set to 1, as an earlier quantizer's
   flag leaves it), the fused arm (pack, fused tiles, fixup) and the native
   arm (pack, tiles, serving quantizer) write the same BF16 intermediate
   rows, codes, scales and status on every live token, and the fused arm
   writes no intermediate row, code or scale, and no status word, past the
   live tokens.
2. **Order independence.** A poisoned eager repeat and a graph replay of
   the fused arm (other item orders) give the same bits, and so does one
   more run after the negative control.
3. **Negative control.** The fused tiles without the fixup must differ from
   the native arm's codes or scales exactly when a live token's final
   status is set, and the flagged pattern must report at least its prior
   flags as set.
4. **Memory safety.** compute-sanitizer memcheck reports zero errors on
   both devices over the synthetic cases at 1, 65, 513 and 2048 live rows
   of the 8192-row capacity.
5. **Coverage and authentication.** Synthetic: the serving quantizer and
   router on E384/top-6 balanced and skewed routing at 8192 and 2048 live
   rows. Captured: the published r7 capture's gate/up inputs (quantized
   rows, routes and route weights) with the sealed rank banks, layers 0,
   13, 14 and 39, all four ranks, the 8192-row chunk of prompt input-0 and
   the 65- and 257-row tails of input-1 and input-2 (24 cases per worker).
   Each captured case first runs the exact gate/up and requires its
   intermediate rows, codes and scales to match the capture's row hashes,
   which authenticates the bank and the inputs.
6. **Timing, only after 1 to 4 pass.** Paired graphs (5 rounds of 3
   samples) of the native arm against the fused arm on the captured
   8192-row cases and on synthetic steps of 8192, 2048 and 512 rows.

No case list, pattern or reference changes after a result.

## Expected time

In the round 7 GPU-exclusive profile of chunk 0 of a cold 32K prompt (Flash
TP4, 250 W) the quantizer takes 9.5 ms per 8192-row chunk over 40 launches,
about 0.24 ms per full layer: one 32-thread CTA per token row, which reads
the 56.6 MB intermediate twice (validate, then write) and writes 28.3 MB of
codes. In round 4's paired timing the native gate/up arm with this
quantizer took about 2.8 ms per captured 8192-row layer. The fused epilogue
adds one warp per 32-column group to each tile, on BF16 values already in
shared memory, and the fixup is a launch that reads 32 KB of status.

Expected, before measurement: the fused arm at 0.92 to 0.99 of the native
arm per full layer (a saving of 0.03 to 0.2 ms), at most the 9.5 ms per
chunk the quantizer takes. A ratio above 1 means the epilogue costs the
tiles more than the pass it replaces. These are estimates; the timing
phase measures them.

Before the job was queued, every phase also ran once on CPU through the
job runner's emulation mode (a host emulation of each launch's reads,
writes and traps, with the quantizer's host form on emulated tile values).
It is a check of the job, not evidence for any criterion.

## Result (2026-10-21, Flash TP4, two RTX PRO 6000 at 250 W)

Criteria 1 to 6 passed on both workers: 56 bitwise records (8 synthetic,
48 captured), memcheck clean, captured cases authenticated. The clear
pattern flagged no token; the flagged pattern flagged 85, 22, 3 and 1
tokens before the fixup, and every unfixed control differed.

Timing (fused / separate quantizer, median of paired rounds): captured
8192-row layers 2.953 / 2.908 ms (1.015); synthetic 8192 rows 1.010, 2048
rows 0.947, 512 rows 0.855. Per 8192-row chunk the full layers lose 0.88 ms
and the upper-skip layers gain 1.23 ms. The 8192-row ratio is above the
expected 0.92 to 0.99: the epilogue costs the full tiles more than the
separate pass. The bundle therefore fuses only steps of at most 2048 rows,
where it measured faster; the bits do not depend on that choice.
