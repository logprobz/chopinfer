# Wide verify exact F32 row tiles

With the `wide_verify` option, Flash TP4 verify steps of 65 to 192 rows run
the vocabulary head, the MoE router and the compressor projections in the
measured whole-K row tiles of `f32.rows.verify` instead of the ordered
split route. The tiles' module (`f32_verify_rows_deepseek41_sm120`) was
locked unchanged one commit earlier and no instance loaded it.

The [layout declaration](wide-verify-f32-rows.json) changes two journals,
the C16 odd-width (verify steps of 65 rows) and C32 maximum (192 rows)
allocation fixtures with the option on:

- `module_activations` and positional `handles` add one load of the locked
  module on each of the four ranks with its payload's locked SHA-256, its
  nine resolves and the configure notes of the three computes: 90112 bytes
  of shared memory for the head's tile and 65536 for the router's and the
  compressor's.
- `operations` replace, in each fixture's two verify cycles, every split
  launch pair of the three projections with the row tile launches, 1952
  exact edits per fixture:

| Projection | Removed per fixture | Added per fixture |
|---|---|---|
| Head, n32320 | 8 `split_prepare_v2`, 8 `split_v2` | 8 `rows_prepare_v1`, 8 `rows_f256_r32_w32x32_s2_k64_v1` |
| Router, n384 | 320 `split_prepare_v2`, 320 `split_v2` | 320 each of `rows_prepare_v1`, `rows_f64_r64_w32x16_s2_k64_z10_v1`, `rows_reduce_v1` |
| Compressor, n512 | 56 `split_prepare_v2`, 56 `split_v2` | 56 each of `rows_prepare_v1`, `rows_f64_r64_w32x16_s2_k64_z10_v1`, `rows_reduce_v1` |

  The 40 added `release` lines are the module and its resolves released at
  destroy. Each added launch reads the split launch's activation, weight,
  weight domain, live-row count and output spans, and the shared workspace.

Every allocation, every other launch, resolve and module, and the order of
the retained operations are unchanged, and every other journal, create
range and device history is identical. `tools/journal_proof/specs/operation_edits.py`
generated the edits from sealed captures of the module commit and the
routing commit and checked them through the fixture prover; they were
reviewed against the symbol, grid and launch counts above. Both forms
compute every output with the same components, chains and order, so the
logits, router logits and compressor rows are unchanged.
