# Exact F32 row tile fallback fix over the merged wide verify branch

The merge of the [exact F32 row tile fallback fix](f32-row-tile-fallback.md)
and the decode row tiles into the wide verify branch changes no journal:
the verify row tile module `f32_verify_rows_deepseek41_sm120` changes its
payload only, and the decode row tile module, which only the
`f32_row_tiles` option loads, is new. No fixture sets that option, so no
fixture loads the decode module.

On 2026-10-31 the journal proof of the merge commit against the first
parent, the wide verify branch, was PROVEN with full history coverage: 29
journals identical, 29 of 29 fixture histories compared. Its specification
was the identity specification of the first parent's capture with the fix's
authenticated module replacement from
[f32-row-tile-fallback.json](f32-row-tile-fallback.json): the verify row
tile module's lock SHA-256 from the previous lock's to the new one's, with
no renamed or removed symbol and the locked resolve order.
