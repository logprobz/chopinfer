# Wide verify HC fusion and queries over the dense folds

The merge of the [HC fusion](wide-verify-hc.md) and the
[queries](wide-verify-queries.md) into the [wide dense folds](wide-verify-dense-folds.md)
changes the two allocation fixtures with the `wide_verify` option (the C16
odd-width fixture at 65 rows and the C32 maximum fixture at 192 rows). Its
journals are proven against each parent; this is the proof against the
first, the dense fold branch.

The [layout declaration](wide-verify-merge-hc.json) declares exact
positional operation edits in those two journals and nothing else: no
allocation, handle, module, resolve or activation changes. Its edits are
the HC and query declarations' launches on the dense fold journals, the
same count per symbol and side in each journal (5520 edits):

| Launch | Per journal |
|---|---|
| removed `row_rounding_round_deepseek41_w5120_v1` | 616 |
| removed `sinkhorn_hc_deepseek41_expand_f32_v1` | 616 |
| removed `sinkhorn_hc_deepseek41_lagged_mix_collapse_v1` | 616 |
| removed `row_rmsnorm_deepseek41_w5120_v1` | 616 |
| removed `row_rounding_widen_deepseek41_w5120_v1` | 376 |
| removed `deepseek41_moe_ceil_w5120_cached_v1` | 616 |
| removed `deepseek41_latent_query_rope_v1` | 320 |
| removed `position_query_deepseek41_rotate_v3` | 64 |
| removed `position_query_deepseek41_mxfp4_v2` | 64 |
| added `deepseek41_hc_round_cols_r4_tx_v1` | 616 |
| added `deepseek41_hc_finish_split_w32_v1` | 616 |
| added `deepseek41_latent_query_rope_heads_v1` | 320 |
| added `position_query_deepseek41_parallel_v1` | 64 |

Each journal loses 2288 lines: C16 odd-width from 193206 to 190918 and
C32 maximum from 345944 to 343656. Every retained operation, the dense fold
launches included, keeps its text and order, and every other journal,
create range and device history is identical.

On 2026-10-31 the journal proof of the merge commit against the dense
fold branch tip, with this declaration, was PROVEN with full history
coverage: 23 journals identical, 2 proven, 25 of 25 fixture histories
compared.
