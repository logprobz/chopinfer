# Exact F32 prefill routes

The qualified family covers the five full-row projection shapes declared
by `f32_dense` and `dspark_router`. Serving selects the 45 measured winning
shape/row pairs in `sm120::f32_prefill`, whose selector records the full
paired timing table. The vocabulary head gathers smaller row sets and
retains its split path. Existing CUDA entries and locks remain unchanged;
the qualified prefill lock is copied byte for byte into the bundle family.

Two tile sizes cover 32 token rows and either 32 or 64 output features.
They use eight or sixteen warps and 64 or 80 KiB of dynamic shared memory.
The first launch classifies each complete activation row and stores its
three exact BF16 components. The second asynchronously stages alternating
128-K buffers while computing the current chain. Warps share the weights
and activation components within the CTA.

Each warp retains the original 16-feature by 8-token MMA mapping and the
same eight k16 steps per chain. The three component chains restart at each
128 K, merge hi then mid then lo, and enter the outer sum from zero in K
order. That outer sum stays in registers. There are no global partial
planes or completion counters. The original whole-row and 16-feature
weight-domain decisions govern fallback, including when one half of a
larger tile is ineligible. Fallback uses the original F32 FMA order.

The CPU planner checks both tile sizes and bounded scratch. The GPU matrix
compares every candidate shape at 257, 511, 512, 513, 2047, 2048, 2049,
8191 and 8192 rows. Every production case compares eager and captured
execution after full, partial, zero-live and repeated invocations, with
poisoned scratch and unchanged caller status. Edge cases cover K tails,
all BF16 weight codes, raw F32 weights, non-finite inputs, signed zero,
underflow, overflow, component loss, component merge order and 128-K
boundaries. The two GPUs execute disjoint case-ID shards.

The pair job reports each launch, the full sequence and the weight-only
bandwidth floor using stored BF16 bytes at 1.6 TB/s. Its checker recognizes
markers anywhere in a line. A measured ratio at most 0.97 qualifies a
candidate at that tested row count; the report chooses the best qualifying
tile and states the first measured win. Serving now uses those 45 points,
without interpolating a winning range. Other row counts, including 4096,
retain the original path. At SM120, 250 W (2026-10-19), the 2048-row router
measured 2399.232 to 675.872 us and wkv measured 3389.440 to 692.224 us.
All 934 prefill bitwise cases passed across the two GPUs.

The compact decode comparison also completed all 1001 bitwise cases,
501 and 500 per GPU, in 46.68 and 60.84 seconds. Its former fivefold gate
rejected the TP1 head despite gains of 2.69-3.19x at every measured bucket
from 33 through 256 rows. The selector records that table and retains the
winning split route. A routed bucket fails only for a paired median ratio
above 1.03; fivefold remains a reported goal, met by 30 of 48 measured
verify buckets. The revised job also compares all 45 promoted prefill
routes through the serving registration.

# Served weight-domain assumptions

Rank arena upload synchronizes each rank stream before returning
(`runtime-host::services::upload_rank_arenas`). Instance creation then
classifies the resolved weight spans separately for each rank, allocates
persistent flag tables and synchronizes before execution. DSpark stage
binding copies the complete checkpoint-name suffix, so the weight and its
`.f32_domain` view move together. The target head and drafter head share
one immutable rank-local vocabulary weight and its classification.

The cache has no invalidation for later weight mutation or view rebinding.
Every projection invocation must retain exclusive access to its shared
workspace until its final stage finishes. Kernel-level comparisons do not
exercise real compiled-arena contents, later mutations by other operations,
or an integrated graph's buffer overlap. Fake-host tests check host bindings
and lifetime but do not execute those device writes. No defect in upload
ordering or rank mapping was found in this inspection.

Large dense prefill bypasses cached flags and split scratch. Its equality
therefore does not qualify those decode dependencies. The gathered head
already uses the small-row path during prefill: if equality includes its
final logits, a persistent head-flag error is less consistent with that
result. A served audit should compare each cached table with the predicate
over its actual rank-local weight span and check scratch ownership across
adjacent graph stages.

# Bounded comparison jobs

Both matrices use the old GPU kernel for every output bit. The small CPU
chain/oracle suite remains in the dry run; large head outputs never use a
CPU GEMM reference. The independent weight-domain predicate checks every
tile in small fixtures and a deterministic sample of at most 32 tiles in
large fixtures, including the first and last. Special-value fixtures keep
exhaustive predicate checks.

Decode coverage enumerates the serving declarations, with rows 1, 2, 4, 8,
16, 31, 32, 33, 48, 64, 128, 192, 255 and 256, plus four distinct seeded
samples. Dense entries also cover every promoted prefill count, including 512,
2048 and their maximum 8192 rows.
The requested boundaries and every registered implementation remain covered.
All numerical witnesses run at rows 1, 32, 33 and 256, with all K tails,
BF16 codes and exponent/sign/mantissa boundaries retained.

Each case logs host wall time for fixtures, allocation/capture, poison,
execution/readback/comparison and kernel timing. Shared weight setup has
its own record. Wall time includes GPU waits. The staged host fixture and
harness crates use test-profile optimization level 2; the kernel-core
profile and every CUDA compiler flag and existing lock stay unchanged.

Poison still fills every intermediate and output before each comparison.
For a one-byte pattern, it uploads at most 16 MiB and expands that prefix
with ordered, disjoint device copies, then synchronizes. Each GPU first
reads back an unaligned fill larger than 16 MiB and verifies all bytes and
both guards. The pair gate requires that check and a host record for every
bitwise case. Each compact worker retains a 900-second timeout.
