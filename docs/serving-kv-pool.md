# Host-managed serving KV pool

A bundle opts in with the `chop.serving-kv.v1` required capability. The host
allocates one pool on each declared rank, using the manifest's KV page size,
alignment, page token count, and rank count. Logical page ID `p` addresses
`pool_base[rank] + p * page_bytes` on every rank. The host builds the engine's
logical `KvPool` with exactly the acquired page capacity.

During `create`, the bundle first allocates weights, workspaces, and persistent
state. It then calls `acquire_serving_kv(layout_id)`. The host measures free
device memory on every rank at that point, subtracts 1 GiB from the tightest
rank, takes the configured fraction (default 0.9), divides by page bytes, and
applies `--kv-pages` and the logical pool's maximum. Zero capacity and a rank
with less than the reserve are refused. The host logs the resulting page and
token capacity. The bundle calls `resolve_serving_kv_pool(rank)` for each rank
and may capture decode graphs using those device buffer views. They remain
stable for the process lifetime and have no bundle release callback.

For each engine batch, the host zeros newly allocated pages and applies any
copy-on-write copy on every rank before invoking `execute`. A copy covers the
whole shared page, because the page layout is the bundle's and one token's
rows need not be contiguous in it. An append copies a shared partly filled
last page. When the manifest's `batch_limits.rewrites_last_page` is true (a
drafter whose row for a position needs the next token, as Qwen3.8's MTP),
it also copies a shared full last page, since the next step rewrites
committed rows there; a prefix hit then reserves a page for that copy
(`KvPool::with_last_page_rewrites`,
`SchedulerConfig::with_last_page_rewrites`). The V2 batch wire
carries each item's previous and new token counts, full post-append page table,
and copy record. The bundle maps the table's page IDs into the acquired pool
and writes the appended tokens. After sampling, the engine commits or rolls
back its prepared logical batch. The bundle's `commit` or `cancel` callback
still commits or restores bundle-owned state.

The per-request KV services (`allocate_physical_kv`, `create_kv_request`)
remain for the conformance fixtures; native bundles use the serving pool.

With prefix caching enabled,
the logical pool also holds anchors: references to the pages of a cached
prefix plus pages the bundle wrote a state snapshot into. They use only pages
no live sequence needs and are evicted least recently used when an append or a
new snapshot needs pages, so admission reservations do not change. Occupancy
reports pages held only by anchors separately from used pages.

## Where prefix anchors are captured

A request resumes from the longest anchor of its own prompt (same tokens,
media and `cache_salt`) that ends before its last prompt token: it always
prefills that token, for its first logits. A prompt captures anchors at:

- its end, when it is at least one page long. This serves later prompts that
  extend it, such as the next turn of a conversation or an agent loop, never
  the same prompt again. With a prefill granule (Qwen3.8's 64) the anchor sits
  at the last multiple of it at or before the end, which a repeat of a prompt
  that does not end on the granule can use. A bundle that declares
  `batch_limits.anchor_items` keeps its state there inside the prompt's last
  chunk, for up to that many such chunks a step, which run past the anchor
  (`BatchItem::anchor_tokens`); the others end a chunk there;
- the start of its final prefill chunk (with a granule, its last multiple of
  it), when the prompt is longer than one chunk (`--prefill-chunk-tokens`,
  2,048 in the profiles). This anchor keeps the configured chunk boundary
  even when a lone request with a declared prefill capacity can use the whole
  step budget, so it can end an
  expanded chunk early. An identical repeat, or a prompt that diverges inside
  that chunk, resumes there and prefills at most one configured chunk;
- the end of the leading full pages an earlier prompt already produced (with a
  granule, their last multiple of it), when that lies past the request's hit
  and no anchor ends there yet, unless the request resumed within one chunk of
  its end: then the anchor it resumed from serves the prompts that share those
  pages. A chunk ends there, which can add a prefill step.

So a prompt that extends an earlier one hits at the earlier prompt's end, and
a prompt longer than one chunk hits from its second use, at its final chunk.
A prompt of one chunk, or one that diverges from an earlier one before its
final chunk, misses on its second use, which captures its shared pages, and
hits from its third. A capture at shared pages ends a chunk; since prefill
rows do not depend on chunk boundaries ([numerics](numerics.md)), its logits
and later hits equal the first send bit for bit.

`chop_prefix_cache_recomputed_seen_tokens_total` counts, at commit, prompt
tokens a prefill computed although an earlier committed prefill had already
computed the same leading full pages. A steady rise next to a low
`chop_prefix_cache_reused_tokens_total` means repeats or shared prefixes are
missing the cache.

## Formats, caching and offload

The selected layout determines page bytes and format on every rank. BF16 and
FP8 layouts have distinct identities; `--kv-cache-dtype auto` uses the package
preference, currently BF16. FP8 quantizes only the declared KV regions, leaving
other state exact. Prefix keys include token and media identity. Model and
layout identity are implicit in the cache's lifetime within one engine.
Anchors restore bundle state as well as logical KV pages.

Prompt-logprob requests cannot use cached prefixes to skip logits they must
report. For MiMo V2.6 and Qwen3.8 restored rows must equal a one-shot cold
prefill bit for bit. A bundle whose prefill arithmetic follows fixed position
blocks declares `batch_limits.prefill_granule`, and the scheduler then
captures and looks up anchors only on its multiples (Qwen3.8: 64).

Offload to host memory is disabled by default; `serve --kv-offload-gib N`
enables it. A transfer holds every page reference until completion on all ranks;
swap-in restores into newly reserved pages before admission proceeds. Resume
tries a GPU anchor, then a host anchor, then recomputation. Refused capture or
restore may fall back only when device completion is confirmed; unconfirmed
work must not release or reuse its buffers.

This path swaps preempted requests. `--kv-offload-prefix on` is refused:
a reusable host prefix cache is deferred. Host copies are not
deduplicated by device allocation identity: every capture copies all of
its pages. A prefix anchor ending inside a full page resumes with that
page as its tail, so an append, or a last-page rewrite, can change its
bytes without changing that identity.

The full correctness suite's `kv_offload` check compares offload-off and
forced-swap servers in two scenarios: plain AR without prefix caching, and
a small KV pool with prefix caching and the configuration's fixed draft
depth. C1 uses 36 prompts, compares token IDs, token strings, logprob bits,
top alternatives, finish reasons and usage after exact host resumes, and
requires every request beyond the first swap point to be preempted. At
least 18 requests must qualify for comparison. Plain runs allow no fallback;
pressure runs list and exclude capacity fallbacks. Concurrent requests must
complete, exercise both copy directions, leave the host tier idle and
produce no stalls. Pressure also requires prefix evictions and, when
speculating, verified items. Before its concurrent cohort, the pressure
record fills the cache one distinct prompt at a time until every earlier
anchor and one filler anchor are evicted; each filler is shorter than the
cohort's longest prompt by more than the pages a request holds beyond its
anchor, so that prompt cannot fit beside the fillers and the cohort evicts
whatever swaps return meanwhile (`tools/qual/kv_offload.py`). Concurrent
equality and an off/off control are
reported as information. The reference host arena is 16 GiB; this check
keeps offload disabled in the default serving profile.

## Media arena

Encoder output, one row per media token, lives in a GPU media arena beside
the KV pool. The arena is a cache keyed by each item's content, not a
reservation: a request is admitted without arena pages. A prefill chunk whose
items would exceed the arena is shortened at a prefill granule, and a step
takes chunks only while their items fit the arena together. A media prompt
starts prefilling only while its step demand (the pages its smallest chunk
covers, capped by the pages of the media still ahead of its prefill) fits
beside those of the prompts already prefilling, in arrival order, so
concurrent prompts rarely evict each other's items. A turn whose images all
lie in its restored prefix demands nothing, so this rule never holds it back.
A step pins the
items its chunks read, encodes those not already present and unpins them when
it commits. Eviction is least
recently used and spares items a live prefill still needs while others
remain; an item evicted between its chunks is encoded again and counted in
`chop_media_reencodes_total` (beside `chop_media_encodes_total`,
`chop_media_cache_hits_total` and `chop_media_evictions_total`). A request
resuming from a prefix anchor places only the items after the anchor, and
once prefill passes an item its host payload is released. Requests with media
reserve their worst-case KV and are not preempted, so a passed item is never
needed again.

## Context parallelism

`serve --context-parallel-size N` (default 1) splits every sequence's KV
pages across groups of N ranks, so KV capacity grows with N where the KV
would otherwise be replicated. N must divide the topology's rank count, and
the bundle must declare it: a manifest topology lists the sizes above 1 it
serves in `context_parallel_sizes` (left out of the canonical JSON when
empty). The host refuses any other size before loading, naming the declared
sizes.

Ownership rule. Groups are consecutive ranks, and rank `t` is group rank
`t % N`. Physical page `j` of a sequence, which holds positions
`j * page_tokens` onward, lives on group rank `j % N`. The host allocates
pages in stripes: one logical pool page is one page on every group rank, all
at the same local page index, so it covers `N * page_tokens` positions and
the logical block table is every rank's own table of local pages. Group rank
`r` holds positions `(i * N + r) * page_tokens` onward in local page
`table[i]`; `runtime_kv::ContextParallelPagesV1` computes the pages, token
counts and positions each rank holds. At N = 1 a stripe is one page and
nothing changes.

The rule covers both kinds of replicated KV. KV that every rank holds (a
latent cache the attention heads share) can split over any N that divides
the rank count. KV with fewer heads than ranks is replicated across
consecutive ranks, so a group must lie inside one replica set: the bundle
declares only sizes that divide the ranks per KV head.

Capacity accounting. The pool is acquired as before: every rank allocates
the same number of local pages, sized by the tightest rank. The engine's
logical pool has that many stripes, so its token capacity is pages per rank
times `N * page_tokens`, counted across the group. Admission reserves and
the pool allocates stripes; a stripe holds at most one page on each rank, so
a reservation bounds every rank's pages and no rank runs out first.
`--kv-pages` caps the pages per rank, which is the stripe count. A
sequence's last stripe may leave up to `N * page_tokens - 1` positions
unused. Startup logs the pages per rank and the group's token capacity.

The prefix cache, snapshots, preemption and offload work on stripes without
change. Prefix digests and the discovery of shared blocks use stripe-sized
blocks, but anchors still sit where the scheduler captures them (prompt and
chunk ends on the prefill granule), so a hit can restore inside a stripe, for
example at token 13 with eight-token stripes. Its next append then copies the
shared partial stripe, one page on every group rank. A snapshot's
state pages are logical pages, so each rank writes its own state into its
local page. Zeroing a new page, the copy of a shared last page and offload
transfers already act on the local page on every rank, which for a stripe
is each rank's share of it.

Wire format. The batch wire does not change: every append carries the
logical stripe table, in the bytes a size-1 pool would send for the same
table. The bundle learns the group size once, from the create record: a
`chop.context-parallel.v1` tag followed by a little-endian `u32` size, after
the served context limit and before the model options. It is sent only above
1, so a size-1 record, every batch wire and every journal stay byte for byte
what they were.

The bundle's kernels then attend each rank over the positions it holds:
the stripe table with the rank's token count as its sequence length, and
the position map for causal masks. Partial results merge across the group in
fixed rank order, and each appended row is written by the rank that holds
its position.
