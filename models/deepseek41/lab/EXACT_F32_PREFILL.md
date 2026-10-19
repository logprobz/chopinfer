# Exact F32 prefill candidates

The candidate family covers the five full-row projection shapes declared
by `f32_dense` and `dspark_router`. It does not enter the serving inventory.
The vocabulary head gathers smaller row sets and retains its current path.
Existing CUDA entries, locks and serving row selection remain unchanged.

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
tile and states the first measured win. It neither interpolates a winning
range nor promotes serving. Measurements must precede any routing change.
The current 2048-row baselines are 2.45 ms for the router and 3.48 ms for
wkv at SM120, 250 W (2026-10-19); candidate gains remain unmeasured.

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
