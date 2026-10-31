# Wide verify dense folds over the HC fusion and queries

The merge of the [HC fusion](wide-verify-hc.md) and the
[queries](wide-verify-queries.md) into the [wide dense folds](wide-verify-dense-folds.md)
changes the two allocation fixtures with the `wide_verify` option (the C16
odd-width fixture at 65 rows and the C32 maximum fixture at 192 rows). Its
journals are proven against each parent; this is the proof against the
second, the HC and query branch.

The [change specification](wide-verify-merge-dense-folds.json) is the
dense fold change's specification generated on the HC and query journals:
the same 36 expansion rules (2640 launches in each journal), the same 4
workspace resizes and 16 added resolves per rank's wide module load, and
the 68 added releases at destroy, which the HC and query edits move 2288
lines earlier. Every other fixture keeps the identity specification. The
HC and query launches are identical on both sides.

On 2026-10-31 the journal proof of the merge commit against the HC and
query branch tip was PROVEN with full history coverage: 23 journals
identical, 2 proven, 25 of 25 fixture histories compared.
