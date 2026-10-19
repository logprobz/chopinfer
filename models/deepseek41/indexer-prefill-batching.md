# Eager prefill indexer batching

The guarded full-index scorer currently runs in 64-row blocks because its
score stride is the cache capacity. A ratio-2 row reserves 524,288 scores;
a ratio-1 row reserves 1,048,576. A cold 8192-row chunk has at most 4096 or
8192 visible scores. The invisible suffix is written as negative infinity
and never enters selection.

`contract::indexer::prefill::batches` plans an eager candidate using the exact
host positions that are uploaded for the call. It rounds the maximum causal
visible length to a key page, and packs as many rows as the score budget and
caller-supplied row limit permit. It takes the maximum over all rows, so
multiple items with different prefixes and upper-layer suffixes work without
assuming sorted positions. Negative positions have zero visibility, and the
same original position-capacity clamp applies. Even an entirely empty batch
has a positive, page-aligned stride.

The lab caps ratio-2 batches at 4096 rows and the layer-20 candidate source at
512. Both the score allocation (256 MiB) and the largest backend workspace
(546,848,768 bytes, already required by 4096 compact rows) stay within the
guarded Flash allocations. At cold 8K, the three ratio-2 layers need six score
and six merge launches altogether, instead of 384 of each. The 2496-row
layer-20 suffix needs five batches instead of 39. The merge grid has one CTA
per row, with up to 4096 rows in a launch. These are planned launch counts,
not measured speedups.

## Exactness

Every row's bounded width is at least its original clamped causal count.
The unchanged guarded scorer therefore admits exactly the same positions,
uses the same page table and physical key addresses, and performs the same
arithmetic and SIMT repair. Query, weights and position slices start at the
same original row. Only the address stride of temporary scores changes.

Both selectors use each row's device-side visible count, which is unchanged.
Their shard partition, integer score/index keys, cutoffs and ordered merge
are consequently identical. Holes stay negative infinity. Stored scores,
including the stored invisible suffix, must match the guarded baseline bit
for bit. The omitted suffix has implicit negative-infinity semantics; the
comparison independently checks every baseline word in that suffix.

The layer-20 producer consumes scores before they are reused. The bounded
stride is page aligned and hence also eight-position-block aligned. Its
visible block maxima, newest-block pin, runtime keep count and ordered IDs
do not change. Tags, visible counts and statuses retain the original packed
row coordinates. Later compact consumers retain the original position
capacity; no score stride is stored in their candidate record.

No workspace crosses a stream fork. A single stream finishes selection and
candidate production before the next batch reuses scores or scratch.

## Qualification and integration boundary

The candidate currently lives in the contract planner and model lab. Serving
continues to use the guarded 64-row path pending GPU evidence. No serving
kernel, existing lock, launch journal or model option changes here.

A serving integration must bind the exact host position slice used by the
upload, immutable per-batch live counts, local transient query rows and
original carried candidate rows. It must prepare the bounded-width ops and
verify their scratch requirements against the existing allocations. It must
not reinterpret the existing 64/4096 block-live table for a new batch size.
The plan is for eager prefill; a graph replay with changed positions requires
a new validated bound. Decode graph planning stays unchanged.

The pair covers 54 shapes, with 8192/8191/4096-row chunks and
2496/1984/448/65-row suffixes at 8K, 32K and 128K context endpoints. Both
ratios compare every score, ordered selection and failure status; layer 20
also compares every retained candidate plane. Edge cases include ties,
unsorted rows, empty/one-position contexts, page holes, wide exponent repair,
NaN/Inf/overflow, smaller runtime selection counts and inactive launch rows.
Inputs, outputs and workspace are poisoned or seeded; each case must reject
a one-bit ordered-ID corruption. The candidate runs through captured graphs.

Timing encloses complete schedules or valid producer-inclusive prefixes on
one stream, so PDL overlap is counted once. It alternates A/B order with
independent cache eviction, one warmup and three measured rounds. It saves
all raw timing rounds and requires exclusive device receipts. Selector-only
timing over stale reused scratch is deliberately excluded. GPU equality and
performance remain unproven until the staged pair passes.
