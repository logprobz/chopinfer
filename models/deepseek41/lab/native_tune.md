# Native MoE tuning results

Flash TP4 native prefill uses the adopted tile and loop lineage recorded in
`moe_native_entries`: gate/up uses r2w6s3, down uses r1w4s2, and both select
`Tune::Scanned`. The locked lab recipes retain historical entries as equality
controls for the served checks in `tests/sm120/native_moe_prefill`.

The earlier paired experiment compared the lean instruction loop and
64-row paired tiles with the then-served tile on balanced, skewed and two
captured 8192-row inputs. These are kernel measurements, not whole-serving
performance evidence.

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

Every round of every arm agreed with its median within 0.03. The bitwise
lean gate/up loop improved the paired median by 5.0%; the paired lean down
tile improved it by 4.3%, or 6.1% on captured cases. The two-CTA gate/up tile
did not improve the paired median.
