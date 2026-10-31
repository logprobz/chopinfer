# Wide verify HC fusion

With the `wide_verify` option, a Flash TP4 verify step of 65 to 192 rows
runs its HC boundaries as a 64-row step does. The column-block producer
defers the all-reduce round, and the split finish writes the expansion, the
next mixer and its input norm, widening and quantizer. `HcOps` prepares the fused finish norm once per wide verify bucket
(`verify_bounds::WIDE`) from the wide boundary's own backend
(`sm120::hc_wide`). Every symbol, cubin and argument is the one the decode
route launches, and the grids follow the step's rows.

The [layout declaration](wide-verify-hc.json) declares exact positional
operation edits in two journals and nothing else: no allocation, handle,
module, resolve or activation changes. In the C16 odd-width allocation
fixture (verify steps of 65 rows) and the C32 maximum fixture (192 rows),
both with the option, each of the two wide verify steps fuses 77 of each
rank's 80 boundaries (308 per step):

| Launch | Per journal |
|---|---|
| removed `row_rounding_round_deepseek41_w5120_v1` | 616 |
| removed `sinkhorn_hc_deepseek41_expand_f32_v1` | 616 |
| removed `sinkhorn_hc_deepseek41_lagged_mix_collapse_v1` | 616 |
| removed `row_rmsnorm_deepseek41_w5120_v1` | 616 |
| removed `row_rounding_widen_deepseek41_w5120_v1` | 376 |
| removed `deepseek41_moe_ceil_w5120_cached_v1` | 616 |
| added `deepseek41_hc_round_cols_r4_tx_v1` | 616 |
| added `deepseek41_hc_finish_split_w32_v1` | 616 |

Each journal loses 2224 lines: C16 odd-width from 190434 to 188210 and
C32 maximum from 343172 to 340948. Every retained operation keeps its text
and order. The three standalone mixers per rank and step keep the one-CTA
entry, and every other journal, create range and device history is
identical.

On 2026-10-31 the journal proof of the HC fusion commit against its parent
(the expand-mix capacity commit), with this declaration, was PROVEN with
full history coverage: 23 journals identical, 2 proven, 25 of 25 fixture
histories compared.
