# Exact F32 row tiles by default

The `f32_row_tiles` flip makes Flash TP4 decode steps and verify steps of
64 rows or fewer run the vocabulary head in the measured whole-K exact F32
row tiles: `f32.rows.decode` (module `f32_decode_rows_deepseek41_sm120`)
through 32 rows and `f32.rows.verify` (`f32_verify_rows_deepseek41_sm120`)
above. The DSpark drafter's head runs the same tiles at its own rows, and
verify steps of 33 to 64 rows run the compressor in its row tiles and the
index weights on the split route. Both modules were locked unchanged before
the flip, and the preceding change
([f32-row-tiles-narrow-configure](f32-row-tiles-narrow-configure.md))
already configures the compressor's 33- to 64-row tile wherever the verify
module loads, so this change configures nothing on a retained resolve.

The [layout declaration](f32-row-tiles-adoption.json) changes the 32 Flash
TP4 device fixtures; the TP1 and TP2 journals stay identical:

- `module_activations` and positional `handles` add one load of the decode
  tile module on each of the four ranks of every Flash TP4 fixture, with
  its four resolves and the configure note of its 256 by 32 compute, and
  one load of the verify tile module with its nine resolves and the
  configure notes of the head's tile and the compressor's 33- to 64-row
  tile in the 13 allocation fixtures that admit more than 32 verify rows
  and did not load it before. Each payload matches its lock's SHA-256.
- positional graph `handles` replace the 128 captured graphs whose node
  count the new launches change with their recaptures, in capture order.
- `operations` hold 2592 exact edits, and every changed launch is a
  vocabulary head launch, the target's or the drafter's:

| Launch | Removed | Added |
|---|---|---|
| `head_f32_..._cached_rows_v1` | 128 | |
| `head_f32_..._cached_reduce_v1` | 80 | |
| `head_f32_..._cached_decode_v1` | 32 | |
| `head_f32_..._cached_decode_v1_last_m8_s1_p1_r0_v1` and its prepare | 8 + 8 | |
| `head_f32_..._cached_decode_v1_last_m32_s1_p1_r0_v1` and its prepare | 24 + 24 | |
| `head_f32_..._cached_decode_v1_reuse32_v1` | 48 | |
| `head_f32_..._cached_decode_v1_reuse32_m60_compute_v1`, `_reduce_v1` | 48 + 48 | |
| `decode_f32_n32320_k5120_rows_prepare_v1` | | 112 |
| `decode_f32_n32320_k5120_rows_f96_r8_w32x8_s3_k64_v1` | | 40 |
| `decode_f32_n32320_k5120_rows_f96_r16_w32x16_s2_k64_v1` | | 28 |
| `decode_f32_n32320_k5120_rows_f256_r32_w32x32_s2_k64_v1` | | 44 |
| `verify_f32_n32320_k5120_rows_prepare_v1` | | 48 |
| `verify_f32_n32320_k5120_rows_f256_r32_w32x32_s2_k64_v1` | | 48 |

  The other edits are the recaptured graphs' 280 replays, named by their
  new IDs in place of the old ones, 112 releases of old graphs, and 1152
  added releases of the activated modules and resolves and of the
  recaptured graphs.
  Each added launch reads its head's activation, weight, weight domain,
  live-row count and logits spans, and the shared workspace.

No pinned fixture runs a target verify step of 33 to 64 rows, so the
compressor's tiles and the index weights' split route appear in no journal;
`dspark::verify_rows` checks their launch counts and grids. Every
allocation, every other launch, resolve and module, and the order of the
retained operations are unchanged, and every other journal, create range
and device history is identical. `tools/journal_proof/specs/operation_edits.py`
generated the edits from sealed captures of the preceding commit and this
one and checked them through the fixture prover; they were reviewed against
the symbol counts above. The row tiles keep each output's components,
chains and order, so the logits and proposals keep their bits.

The long-context template and failed-template tests pin direct digests
outside the sealed fixture set. Private captures of both revisions with
those tests checking their journals through `check_journal` changed the
same way: each Flash TP4 lifecycle loads the decode tile module on every
rank (20 records), the AR long-context decode heads (4 in each verification
mode) and the DSpark ones (8, with 8 recaptured graphs) take the decode
tiles, and the failed-template journals change by the module load alone.
