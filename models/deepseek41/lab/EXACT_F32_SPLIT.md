# Exact F32 split projection

The `f32_split` family routes rows 1 through 256 to two launches. The first
classifies each complete activation row, writes its exact three BF16
components, and resets every completion counter. The second computes four
independent 128-K chains per CTA. Its final CTA for each output tile loads
every chain and adds from zero in K order. Atomic arrival order selects
the reducer but never determines arithmetic order. Whole-row and whole
16-feature weight-tile refusal retain the original F32 FMA fallback.

The persistent weight-domain table still initializes once per immutable
weight view. Each graph invocation initializes its own transient scratch,
including invocations with zero live rows. No replay depends on old
counters. The head uses this same entry and does not rescan its weights.
The original `f32_decode` and baseline families and their locks remain
unchanged. Rows above 256 retain the baseline kernel.

The shared scratch allocation includes the DSpark vocabulary head at its
160-row maximum. This requirement must participate in the drafter's
workspace maximum; its previous omission was hidden by the 32-row split
limit. Without a drafter, shared scratch grows by about 63.5 MB per rank
at TP4 (125.0 MB at TP2 and 248.1 MB at TP1).

The lab enumerates registered entries and tests every row count 1 through
256, all special-value witnesses at tile boundaries, and dense rows 2048
and 8192. Every routed row count compares eager and captured execution;
timed buckets and edge witnesses also exercise inactive rows, zero live
rows, poisoned scratch and repeated replay. Static weights and their
classification are reused across the production row sweep.

The staged pair job reports cold-weight paired times, individual launches,
and time divided by the weight-bandwidth floor. The stated floor assumes
1.6 TB/s. Its numerator uses the actual BF16 storage bytes; the coverage
manifest also records F32-equivalent bytes. It is a weight-only lower
bound, excluding activation traffic, scratch traffic, arithmetic and
launch overhead. Fixed timing gates require new/old <= 0.2 for measured
rows 33 through 256 and <= 1.03 elsewhere. The timing table is written
even if a performance gate fails. GPU acceptance remains pending that
job; CPU compilation and oracle checks are not performance evidence.

## Prefill follow-up

The job measures the original dense entries at 2048 rows per chunk and
at maximum capacity. There is no 2048-row result from the earlier pair
jobs. Forty router calls contribute `40 * router_old_us / 1000` ms per
chunk. A percentage of prefill also needs a measured full-forward time.

The proposed exact prefill path uses a 32-feature by 32- or 64-token CTA
tile with several warps sharing asynchronously staged weights and the
three precomputed activation components. Each warp retains the original
16-feature by 8-token MMA mapping. It restarts the three component
accumulators at every 128 K, merges hi then mid then lo, and immediately
adds that chain to its output accumulator in increasing K order. Two
shared-memory buffers overlap the next 128-K load with current arithmetic.
This reuses weights across more tokens and keeps chain partials in the
CTA instead of allocating a full split plane for every prefill row.

Keep the whole-row domain check, cached 16-feature weight flags and
per-row SIMT fallback. Add this as a separately locked family and compare
it bitwise at 257, 512, 2048 and 8192 rows, including chunk and live-row
boundaries, before routing prefill to it. Select its tile size from paired
measurements; the current change leaves prefill arithmetic and routing
unchanged above 256 rows.
