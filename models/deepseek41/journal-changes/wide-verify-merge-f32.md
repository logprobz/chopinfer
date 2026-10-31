# Exact F32 row tiles over the merged wide verify branch

The merge of the [exact F32 row tiles](wide-verify-f32-rows.md) into the
wide verify branch (dense folds with their tuned schedules, HC fusion,
queries and the wide direct MoE) changes the two allocation fixtures with
the `wide_verify` option (the C16 odd-width fixture at 65 rows and the C32
maximum fixture at 192 rows). This is its proof against the first parent,
the wide verify branch.

The [layout declaration](wide-verify-merge-f32.json) restates the F32
change's declaration for this base. Its first parent does not lock
`f32_verify_rows_deepseek41_sm120`, so the module is an added module rather
than an activation of a locked one: it loads exactly where
`dense_decode_wide_deepseek41_sm120_v1` loads, once per rank, and its nine
resolves carry the configure values of the three computes (90112 bytes of
shared memory for the head's tile and 65536 for the router's and the
compressor's). `tools/journal_proof/specs/operation_edits.py` regenerated
the operation edits from captures of the first parent and of the merge:
1952 exact edits in each of the two journals, 768 removed and 1144 added
launches and 40 added releases. Per symbol and side these are the counts
of the F32 change's own declaration on its parent; only their positions
differ. Each journal grows by 464 lines: C16 odd-width from 191338 to
191802 and C32 maximum from 344076 to 344540. Every other journal, create
range and device history is identical.

On 2026-10-31 the journal proof of the merge commit against the first
parent, with this declaration read at the commit that adds it, was
PROVEN with full history coverage: 23 journals identical, 2 proven (768
removed and 1184 added operations, 36 added resolves and 12 added
configures each), 25 of 25 fixture histories compared.
