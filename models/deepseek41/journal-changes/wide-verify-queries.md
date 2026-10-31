# Wide verify latent and index queries

With the `wide_verify` option, a Flash TP4 verify step of 65 to 192 rows
rotates its latent queries with the per-head entry that serves 49 to 64
rows and computes its index queries with the fused parallel position query
that serves 33 to 64 rows. `GlueOps` and `IndexerOps` prepare each once per
wide verify bucket (`verify_bounds::WIDE`), with the decode entry's symbol
and cubin, and both grid the step's rows.

The [layout declaration](wide-verify-queries.json) declares exact
positional operation edits in two journals and nothing else: no
allocation, handle, module, resolve or activation changes. In the C16
odd-width allocation fixture (verify steps of 65 rows) and the C32 maximum
fixture (192 rows), both with the option, each of the two wide verify steps
changes 160 latent query and 32 index query launches:

| Launch | Per journal |
|---|---|
| removed `deepseek41_latent_query_rope_v1` | 320 |
| added `deepseek41_latent_query_rope_heads_v1` | 320 |
| removed `position_query_deepseek41_rotate_v3` | 64 |
| removed `position_query_deepseek41_mxfp4_v2` | 64 |
| added `position_query_deepseek41_parallel_v1` | 64 |

Each journal loses 64 lines: C16 odd-width from 188210 to 188146 and C32
maximum from 340948 to 340884. Every retained operation keeps its text and
order, and every other journal, create range and device history is
identical.
