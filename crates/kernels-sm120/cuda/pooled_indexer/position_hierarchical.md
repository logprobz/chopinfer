# Hierarchical ordered position selection

This lab candidate replaces only the final merge of 32 shard lists. The
scorer, shard selector and their scratch remain unchanged. Four CTAs each
read eight contiguous lists and retain at most the runtime top-k count.
One final CTA selects from their union. At k=512, its bound drops from
16,384 keys to 2,048 keys.

Every intermediate entry is the original 64-bit score/index key. If a key
is among the global top k, fewer than k keys in its own group can exceed
it. Therefore it survives its group's top-k filter. The final union has
exactly the same global top-k set. The total order includes the original
index tie key; no score conversion or new floating-point operation occurs.

Each group scans contiguous original shard lists and emits its survivors
in their original order. The final CTA scans groups in that same order.
Shard lists already follow increasing positions, including valid compact
candidate IDs, so final outputs retain the original increasing-position
order. Zero padding is excluded, and every output slot is initialized to
-1. Each group ORs its eight original failure flags, and the final CTA ORs
all four group flags before emitting any ID. Inactive rows remain untouched.

When each shard contributes at most 64 entries, the group launch returns
without reading or writing its intermediate buffer. The final launch uses
the original merge body and its failure check. This also covers empty rows
and small runtime top-k counts. The extra no-op launch still has overhead;
the pair measures short contexts as well as 32K and 128K.

The lab allocates a separate intermediate buffer of
`rows * 4 * (512 + 1) * 8` bytes, 16,416 bytes per row. It is used on the
same stream after the shard selector and before the final merge. No
cross-stream workspace is shared. Serving admission and allocation are
unchanged until this candidate is qualified.

The negative-control kernel retains only k/4 entries per group. A tied
fixture concentrates all global winners in the first group and requires
that mutant to disagree. Other cases cover random keys, signed zero,
holes, poisoned tails, nonfinite scores, empty/one-position rows, malformed
compact IDs, stale tags, inactive rows and runtime k=1/128/511/512.
