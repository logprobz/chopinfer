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

## Serving and qualification

Native Flash with `text_config.indexer_mma=true` uses this plan for eager
full-index sources. Normal, upper-skip and micro-prefill pass the exact host
position slice used by their upload. Their transient query rows remain local;
carried candidate planes keep the original packed coordinates. Each distinct
batch size is prepared once within the eager lowering, with its workspace
checked against the existing allocation. BF16 keys and the SIMT model option
retain their prior schedules. Compact reindex and decode graphs are unchanged.

An immutable table of counts 0..=8192 adds 32,772 metadata bytes per rank.
It follows the existing mutable 64/4096 block-live counts in the same
allocation and is uploaded once at instance creation. Per-step uploads touch
only the original prefix. Both the native text and native media journals pin
the allocation growth, initialization and bounded prefill argument changes;
BF16 journals and every existing kernel lock are byte-identical.

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
timing over stale reused scratch is deliberately excluded.

On 2026-10-19, the pair passed all 54 bitwise cases and corruption controls,
with 105 timing pairs and all 630 raw measured samples present. Both GPUs
were exclusive at 250 W, with no Xids. The two shards contained 13.75 and
15.02 seconds of measured GPU intervals, plus warmups and fixture work,
consistent with the 31-second run.

Complete guarded score/select schedules for 8192 rows, paired medians in ms:

| Source | Context | Previous | Batched |
| --- | ---: | ---: | ---: |
| Ratio 2 | 8K | 22.954 | 6.912 |
| Ratio 2 | 32K | 59.486 | 30.960 |
| Ratio 2 | 128K | 181.884 | 129.221 |
| Layer 20, including candidate publication | 8K | 39.835 | 12.922 |
| Layer 20, including candidate publication | 32K | 108.421 | 62.637 |
| Layer 20, including candidate publication | 128K | 333.582 | 258.057 |

The 2496-row layer-20 suffix at cold 8K improved from 13.839 to 5.512 ms.
All 42 complete-schedule medians improved; the median paired new/old ratio
was 0.585, ranging from 0.301 to 0.984. The smallest difference is within
timing noise. These are synthetic kernel-chain measurements, not served
TTFT or token-throughput measurements. Integrated serving comparison remains
part of the integration round's qualification.
