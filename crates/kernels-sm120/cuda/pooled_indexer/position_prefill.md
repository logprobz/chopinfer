The four-row scorer assigns one query row and eight key positions to each
warp. Rows with one sequence share a staged key tile. Mixed sequences have
separate banks. Every row uses its own visible count and every warp reaches
the CTA barriers, including the last partial row tile.

Each warp retains the existing two ordered m16n8k64 MMAs for each group of
16 heads. The coarse scale certificate and the per-head certificate are
unchanged. Rejected slots use the original SIMT dot and head chains. A failed
query repack also recomputes each admitted slot from the original BF16 query.
The F32 head FMAs keep their original order. Moving a row across a tile or
partition changes data reuse, not its operands, guard domain or arithmetic.

The cached local selector stores each thread's score/index keys in registers.
Four radix passes find the boundary score and the number of equal-score keys
still required. Input traversal already follows increasing original position,
so a ballot prefix keeps exactly the lowest positions tied at the boundary.
The emitted 64-bit keys, list padding and failure slots retain the original
layout. The existing final merge remains unchanged. Shards wider than the
register cache use the existing selector, with the decision made from the
device-visible count.

These are separately compiled candidate families. Their declarations do not
change a serving binding. Qualification must compare raw scores before the
selector's nonfinite cleanup, then compare final scores, status and ordered
IDs, including candidate publication. Partition checks must retain absolute
positions while changing local query-row origins. Real sampled rows must be
reported as samples; duplicating them does not establish a full real prefill
chunk. Packed-query scratch belongs to each timed score batch, so pack stays
inside a multi-batch score timing graph. Workspaces are disjoint between arms
and reused only along the existing sequential stage dependencies.
