# Native MoE prefill qualification

The served native prefill path changes bits. It runs DeepSeek V4.1 Flash TP4's routed
expert tiles (gate/up and down, 384 experts, top 6) in every prefill step
through native `mma.sync kind::mxf8f6f4.block_scale` MMA on the trained
operands: MXFP4 weights (E2M1, UE8M0 per 32) by the E4M3/32 ceil-UE8M0
activations the serving quantizers already produce. Every product is exact.
Each 128-K chain starts from +0 and is added into the F32 accumulator in
ascending K order, the same fold structure as the exact BF16 engine; inside
a 32-K group the instruction does its own reduction. Routing, the gate/up
epilogue (clamp, SiLU, route weight, BF16), the intermediate ceil quantizer
and the routed sum are unchanged. In steps of at most 2048 rows the
quantizer runs in the gate/up tiles' epilogue with a fixup launch, bit for
bit its separate launch (native_ceil.md); the lab's native arm launches
the same fused entries there. Native MoE prefill is adopted for Flash TP4;
its experiment switch is removed. Decode, verify, the
drafter and other topologies keep their existing expert paths. This
path qualifies on the within-bound track of docs/numerics.md,
"A speedup that changes bits".

The kernels are model-neutral (`kernels_sm120::moe::native_grouped`): any
expert count, top-k, K and N (K and N multiples of 32 and 16), with an
active-expert tile queue and a persistent grid taking items from a ticket.

The f64 gate checks original decoded inputs with the unchanged
`2^-22 * K * S` bound and per-route cosine at least 0.9999. Chunk prefixes,
poisoned replay and graph execution must retain the native path's bits.
An omitted-chain control must fail the bound. Changes to the native path
also require the model's absolute and paired teacher gates.

The earlier no-larger-error down experiment remains unselected: although
its down output was no worse, ordered-sum rounding increased RMSE in 81 of
400 cases, up to a ratio of 1.000117. Its predicate was not relaxed. The
adopted path follows the separately declared within-bound contract;
no-larger-error comparisons remain diagnostics.

## Results (2026-10-20, r4)

Flash TP4 lab on one GPU pair at 250 W, no Xids; every case below ran on
both workers and the executed counts match the plan.

- **1, instruction exactness:** 17,408 probe outputs per run, 0 failures,
  in both the memcheck and oracle runs. Report only: 88 of 128 small-sum
  outputs round to nearest.
- **2, f64 oracle bound:** 0 failures of 52,101,120 down elements and
  11,722,752 gate/up elements over 10,176 sampled route rows (3 synthetic
  cases and 24 captured cases per worker, all captured cases authenticated
  against the capture). Worst bound ratio 0.00013 (gate/up) and 0.0016
  (down and chain down); minimum cosine above 0.9999999999999.
- **3, serving entries:** the serving gate/up equals the raw entry and all
  three other tile variants are bitwise equal in every case.
- **4, invariance:** prefixes 4097, 65 and 1, graph replay and poisoned
  repeat bitwise for all three synthetic cases on both workers.
- **5, negative control:** the omit-first entry failed the bound in 54 of
  54 cases (at least 490,300 failing elements each).
- **6, memory safety:** "ERROR SUMMARY: 0 errors" on both workers over the
  12 synthetic cases (three routings at 1, 65, 513 and 2048 live rows), the
  tile variants and the probe.
- **7, report only:** native down's RMSE is never larger than the exact
  down's (54 of 54 cases, ratio 0.85 to 1.0); the ordered top-k sum with
  the shared addend is no larger in 44 of 54 cases (largest RMSE ratio
  1.000062, no larger maximum error), the rounding noise the decision above
  anticipated.

Timing (paired, 5 rounds of 3 samples, captured 8192-row chunks of layers
0, 13, 14 and 39 on all ranks): exact 5.58 ms gate/up and 4.14 ms down per
layer, native 2.83 and 1.77 ms (median ratios 0.51 and 0.43; both together
10.48 to 5.07 ms, ratio 0.48). Over 40 full 8192-row layers that is 419 ms
against 203 ms. Later tuning adopted the bitwise r2w6s3 gate/up tile; see
[native tuning](native_tune.md).

## The row-tiled bank (2026-10-23)

Flash TP4 serves its backbone gate/up bank in 64-row by 128-K row tiles,
each 64-byte block of the codes in quad-lane order, and the bundle embeds
the fused ceil and scanned gate/up tiles and the column gate/up only as
their builds for that layout. The harnesses upload a copy of the gate/up
bank in the served layout (`pipeline::gate_tiles`) and run both arms'
launches of those entries as those builds over it. The target
bank's wrapper check compares the lab's launches with the served wrapper
(`moe_row_tiles_op`) over the tiled bank. The lab and tune entries, the
raw entry and the tile variants keep the row-major bank, so the serving
checks still compare the served bits with them.

## Captured inputs

The captured GPU tests read two inputs that are not in the
repository:
- `CHOPINFER_NATIVE_MOE_READY`: the ready record (path and digest) of the
  sealed r7 prefill capture. The capture holds inputs, routes and the
  exact producer's outputs at layers 0, 13, 14 and 39 on all four ranks,
  for the chunks under criterion 2.
- `CHOPINFER_NATIVE_MOE_BANKS`: the sealed rank banks of those layers,
  each rank's gate/up and down weights and scales (64 files, 28.9 GB),
  with `BANKS.json` naming each file's bytes and digest.

The capture and banks behind the 2026-10-20 results were in temporary
storage and are gone; only their manifests remain. The test that wrote the
banks from the checkpoint lives only in an earlier lane's source snapshot
(the L10 grouped-BF16 lane), not in this repository. Both must be
regenerated before these tests run again. Without them, each test fails
before any GPU work and names the missing input. The captured
harnesses share their path over the row-tiled bank with the synthetic
harness: the tiled upload, both arms' remap and the wrapper check. The
synthetic harness runs that path on GPU, and the captured and tune
harnesses keep CPU coverage until their inputs exist.
