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

## Long-context scoring and selection

Past the short tiled band (`text_config.indexer_prefill_tile`, every position
below 32768), the batches above still ran the coarse guarded scorer at 192
CTAs per row and the live selector (or the 128K cached
selector). A profile on 2026-11-04 of tp4-dspark 512-row mixed steps beside 8
and 16 streaming decoders put full-source scoring and selection at 11.0 to
11.3% of the 200K step and at all of the step's growth from 100K to 200K.

A phase whose largest position is at least 32768 and whose page-rounded extent
is at most 262144 now scores each full-source batch with the tile4 scorer at the
CTA count the indexer table gives its rows, and selects with the early-stop
selector and merge:

| Source | Rows | CTAs |
| --- | --- | ---: |
| Ratio 2 | 32-191 | 192 |
| Ratio 2 | 192-1535 | 32 |
| Ratio 2 | 1536-4032 | 16 |
| Layer 20 | 32-447 | 192 |
| Layer 20 | 448-512 | 32 |

Other rows keep the previous route. A ratio-2 batch exceeds 4032 rows only when
the phase's largest position is exactly 32768; no measured context produced
that shape. Every batch of a phase shares its extent, so a phase that straddles
32768 takes the band for all of its rows. Compact reindex batches, the layer-20
three-stage producer, decode and verify, BF16 keys and the short band are
unchanged.

The tile4 scorer is the guarded fast scorer's body with a four-tile position
loop: the same packed query, FP4 MMA, exactness predicate, coarse bound, SIMT
repair and ordered 32-head chain. Its position loop and its negative-infinity
tail are grid-stride over the grid's x size, so any CTA count writes every
score once with the same value. The early-stop selector differs from the live
one only in stopping its radix cutoff once the boundary bucket is kept whole;
it keeps the integer keys, ordered ties and ordered merge. The cached selector
lost to it at every measured batch, the 128K extent included, so the table
chooses only the CTA count.

The kernel-lab pair compared the previous route with tile4 at 8, 16, 32 and 192
CTAs, each with the early-stop or the cached selector, at contexts of 32832,
65536, 100544, 131072, 200512 and 262144, over the batch sizes the planner
produces for 512-row mixed steps, 8192-row chunks and upper-skip and adaptive
tails, and on both sides of every row-range edge. On 2026-11-04 it compared raw
scores before cleanup, final scores, statuses, ordered IDs and layer-20
candidate records bit for bit, 15872 plane comparisons in its complete run, on
seeded inputs and on ties, repair scale edges, NaN and off-grid queries,
non-finite weights, scale bytes, page holes, inactive rows and packed short and
long rows; each case rejected a one-bit ordered-ID corruption. Timed with cold
caches, one warmup and three paired rounds per visit and three visits per
batch, every admitted range beat its in-job A/A spread in all 474 visits
(spread median 0.27%, largest 3.70%), with minimum gains of 17.8 to 34.6% per
batch size. Projected per rank, a 512-row mixed step saves 3.9 ms at 100K and
6.7 ms at 200K, and an 8192-row chunk 62 ms at 100K and 105 ms at 200K.

Served evidence, 2026-11-05, one install with the candidate option off (A) and
on (B), 300 W enforced on all four GPUs, nothing set:

- Exactness: on tp4-ar and tp4-dspark, 28 requests each matched bit for bit in
  token IDs, text, top-20 logprobs, finish reasons, usage and cached tokens:
  cold greedy prompts of 40000, 100003 and 200003 tokens, the top-20 prompt
  logprobs of every row of a cold 40000-token prompt, seeded sampled prompts of
  60000 and 150003 tokens, prefix-hit continuations of 100032- and 200000-token
  prompts beside eight streaming decoders and their lone restores, and eight
  packed cold prompts (five past 32K) and their restores.
- Step time on tp4-dspark, visits A1, B1, B2, A2, each a fresh keeper-borrowed
  process, Engram residency 92.7 to 100% and 12 to 57 major faults per visit.
  The primary metric is the median step time less its Engram wait (synchronous
  mapped gathers plus the prepared gathers' exposed waits); the raw median is
  secondary. Gains are mean (spread) over the two pairs:

| Cell | Samples per visit | Adjusted gain | Raw gain |
| --- | ---: | ---: | ---: |
| 512-row mixed step, 200K, 16 decoders | 12 | 2.58% (0.06) | 3.22% (1.07) |
| 512-row mixed step, 200K, 8 decoders | 12 | 2.96% (2.09) | 3.42% (2.99) |
| 512-row mixed step, 100K, 16 decoders | 12 | 1.66% (0.52) | 1.21% (1.21) |
| 512-row mixed step, 100K, 8 decoders | 12 | 2.07% (0.09) | 1.98% (0.25) |
| Cold 200003-token prompt, 8192-row chunks past 32K | 40 | 6.30% (0.18) | 5.91% (0.36) |
| The same chunks from 32K to 96K | 16 | 4.98% (0.12) | 4.65% (0.40) |
| The same chunks from 96K | 24 | 7.92% (0.18) | 7.52% (0.03) |

The decode-only steps of the same windows moved by -0.04 to 0.19% (adjusted),
within their spreads. A repeat of A1 with the same seeds differed from it by at
most 0.64% in any cell.
