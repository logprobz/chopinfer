# Native MoE tuning pair

Lab only; serving is unchanged. Flash TP4, target bank, 8192 rows.

## Where the tiles stand

Per full layer and rank at 8192 rows, gate/up is 0.580 TFLOP and down
0.290 TFLOP of useful E2M1 x E4M3 work. At the FP8-class rate of the
1.6 GHz reference clock (616 TFLOP/s) the MMA floors are 0.94 ms (gate/up)
and 0.47 ms (down). Memory floors at 1.63 TB/s are about 0.8 ms (gate/up)
and 1.0 ms (down, whose 1.01 GB of F32 route rows alone take 0.62 ms).

Paired lab timings (2026-10-20 and 2026-10-21) put the served gate/up
tiles at 2.6 to 2.97 ms per layer (r2w4s3) and 2.36 to 2.77 ms (r2w6s3),
32 to 40% of the MMA floor, and the down tiles at about 1.77 ms, 56% of
the memory floor and 27% of the MMA floor. Another implementation of the
same instruction takes about 1.9 ms (gate/up) and 1.15 ms (down) per layer.

The served loop runs one CTA per SM (gate/up r2w6s3: 12 warps at 168
registers; down r2w4s3: 8 warps at about 230), three cp.async stages of
128 K, and per MMA spends integer work on the activation NaN check, the
scale byte extraction and the admission checks. Shared-memory traffic is
about a third of the MMA time per stage at gate/up's tile. Down has 4.5
stages of K per tile, so its per-tile pipeline fill and F32 epilogue are a
larger share than gate/up's.

## Arms

Each arm times against the served tile of the same projection on the same
inputs as paired CUDA graphs (5 rounds, 3 samples per arm). Bitwise arms
must write the served output bytes; ablation outputs are not meaningful.

| Arm | Change | Kind |
|---|---|---|
| lean | scale bytes by the instruction's byte selector; NaN codes checked by the first column warp of each row group only | bitwise |
| no_checks | no NaN, admission or slow-path checks | ablation |
| no_mma | every load and check, no tensor instruction | ablation |
| no_global | no global loads after the prologue (no checks) | ablation |
| mma_only | MMAs on register operands: no global or shared loads, no checks | ablation |
| paired | 64-row tiles (gate/up r1w6s2, down r1w4s2), registers bounded for two CTAs per SM | bitwise |
| paired_lean | the paired tile with the lean loop | bitwise |

The ablations name the limiter: mma_only near the floor and no_global near
served means the shared loads and integer work limit; no_global well below
served means the global pipeline (depth or L2) limits; no_checks and lean
measure the integer overhead; paired measures what overlapping one CTA's
barriers, fill and epilogue with another's mainloop recovers.

## Cases

Worker 0: the balanced synthetic e384 case and captured layer 14 on rank 0.
Worker 1: the skewed synthetic e384 case and captured layer 39 on rank 1.
Both projections per case; down reads the served gate/up's quantized
intermediate. The job fails on any missing, duplicate or failed case or
arm, any bitwise arm that differs, zero executed tests, a power cap other
than 250 W or an Xid. Timings are report-only.

## Results (2026-10-21, r1)

One GPU pair at 250 W, no Xids, 16 s. Both workers executed their two
cases, each with all 7 arms of both projections timed (5 rounds of 3
samples per arm). Every bitwise arm wrote the served bytes on all four
cases. Median round ratios, arm over served tile (balanced, captured
layer 14, skewed, captured layer 39):

| Arm | Gate/up | Down |
|---|---|---|
| lean | 0.952, 0.950, 0.946, 0.950 | 0.978, 0.977, 0.973, 0.980 |
| paired | 1.071, 0.988, 0.969, 1.010 | 0.994, 0.976, 1.000, 0.950 |
| paired_lean | 1.012, 0.945, 0.924, 0.968 | 0.971, 0.952, 0.977, 0.926 |
| no_checks | 0.856, 0.838, 0.836, 0.840 | 0.876, 0.873, 0.866, 0.872 |
| no_mma | 0.567, 0.562, 0.523, 0.578 | 0.683, 0.700, 0.679, 0.715 |
| no_global | 0.513, 0.564, 0.646, 0.540 | 0.731, 0.730, 0.762, 0.722 |
| mma_only | 0.320, 0.359, 0.390, 0.341 | 0.514, 0.546, 0.541, 0.543 |

Every round of every arm agrees with its median within 0.03. The served
tile's own time depends on the arm it is paired with (gate/up balanced
2.62 ms beside lean, 3.19 ms beside no_mma), so the ablations are read as
ratios only.

Bitwise and faster: lean on gate/up (5.0%) and the two-CTA lean tile on
down (4.3%, 6.1% on the captured cases). The two-CTA gate/up tile is not
faster than the served one.

Limiter. Gate/up's MMA-only loop runs at about the MMA floor (0.9 to 1.05
ms against 0.94), and the loop without MMAs takes 56% of the served time,
so the MMAs and everything else barely overlap: the served time is close
to their sum. Of the rest, global loads and their waits are about 27% of
the served time (no_global, which also drops the checks, against
no_checks), checks about 16% (lean recovers 5 of it) and shared loads and
E2M1 expansion the remainder. Down's MMA-only loop still takes 1.0 ms,
about twice its MMA floor: the F32 route-row epilogue (0.62 ms at the
memory rate) does not overlap the mainloop, which is what the two-CTA
tile partly recovers.
