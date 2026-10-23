# Prefill Engram layer gathers

`CHOPINFER_DEEPSEEK41_PREFILL_ENGRAM=1` opts AR text prefill into prepared
per-layer mapped gathers. The default is off. Both gathers enqueue before
compute; every rank waits immediately before that layer's Engram row decoder.
Layer 14 can gather while layers 1 through 13 execute. Hashing still uses the
current chunk's tokens and committed Engram history. Commit and cancel retain
their existing history rules.

The candidate uses the existing asynchronous mapped-gather host service and
supports ordinary eager and upper-skip prefill. Media, diagnostic text taps,
DSpark, and alternate prefill communication schedules retain their existing
path. No runtime ABI, kernel, arithmetic order, reduction boundary, output
layout, or lock changes are required.

## Cross-stream audit

Each layer owns a separate device ID span. The IDs are uploaded in the same
order as the synchronous gather, with the row-major interleaved hash result
split into contiguous per-layer inputs. Both ID uploads precede both enqueues.
The host records producer readiness after the uploads. It records destination
owner readiness before any current-chunk compute is submitted, so the private
upload streams cannot wait behind a consumer that is waiting for them.

The destination code and scale spans are the existing per-layer slices of the
prefill scratch buffers. A prepared-plan audit rejects any earlier launch that
references a pending layer's ID or destination span, including direct device
arguments, tensor maps and pointer arrays. The audit checks all rank spans and
requires exactly one ordered consumer boundary for each layer. Each split also
passes the existing collective sequence check before any upload or launch.
The per-rank launch sequence remains unchanged.

The private gather streams read only the dedicated ID buffers and mapped
tables. They write only the audited layer destinations. Every main stream
waits for its layer's upload event before reading those destinations. Existing
row-status readbacks complete all consumers before nonblocking gather
completion is checked. Callback failure or an incomplete transfer refuses the
step. A failed or pending gather cannot be reused by another execute call.
Destruction releases and fences gathers before releasing their referenced
allocations; existing poisoned-collective handling remains in force.

Flash TP4 allocates two 786,432-byte device ID buffers. Each prepared layer
owns 198.75 MiB of pinned staging, within the host service's 256 MiB per-handle
limit. No additional device output buffers are allocated. Each transfer uses
only the current row count, up to 8192.

## Qualification

CPU fixtures compare complete per-stream launch journals with the candidate
off and on, check uploaded ID bytes against the hash oracle, and exercise
TP1, TP2, TP4, both KV layouts, partial rows, handle reuse, cancellation and
allocation failure. Plan tests reject early readers of either layer, reordered
or repeated layer boundaries, mismatched extents and a pending prior gather.
These tests establish scheduling invariants, not GPU numerical equivalence.

The served comparison uses one diagnostic install with the candidate off and
on. Untimed requests capture committed state and full-vocabulary logits at
chunk boundaries; generated token IDs and logprob words are also compared.
Separate prefix-cold 8K and 32K timings run in ABBA order with capture disabled,
host-load samples and per-process minor and major fault counters. The
diagnostic build disables the ready-event result fast path in both arms.

The `prefill-engram` record reports host gather nanoseconds and
`exposed_ns` per layer. The latter is the maximum rank interval between events
immediately before and after its upload wait. It includes event and submission
overhead; it is not summed rank time or a whole-request critical-path measure.
The served result is needed before selecting this candidate or deciding whether
remaining exposed gather time warrants a cross-chunk interface extension.

## Cold large gathers

A chunk of more than 2,730 tokens gathers at least 65,536 rows per plane
(24 per token), which skips the resident row cache. Before runtime-host read
such gathers through its reader threads, the prepared gather copied them from
the mapping with no read-ahead (asynchronous read-ahead is off unless
`CHOPINFER_MAPPED_ASYNC_READ_AHEAD=1`), one page fault at a time. Measured on
CPU on 2026-10-23 on the serving host's drive, one 8,192-token chunk of one
layer for four ranks (196,608 rows per plane, codes and scales cold, quiet
drive) took 18 to 22 s that way, about 10 s of it the codes plane; the
synchronous gather this candidate replaces took 1.9 to 2.6 s. A chunk
gathers two layers. With the reader threads, the default since then, the
same prepared chunk took 0.30 to 0.36 s per layer, in the same bytes.

The old copy returns with `CHOPINFER_MAPPED_LARGE_GATHER_FETCH=0`,
`CHOPINFER_MAPPED_MISS_START=0` or `CHOPINFER_MAPPED_MISS_FETCH=0`. Do not
time or select this candidate under any of them without
`CHOPINFER_MAPPED_ASYNC_READ_AHEAD=1`. The served comparison must drop the
Engram tables' pages before each timed run: repeated prompts find the rows
in the page cache, and the cold cost then does not show.
