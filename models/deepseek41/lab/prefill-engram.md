# Prefill Engram layer gathers

AR and DSpark text prefill use prepared per-layer mapped gathers by default
on supported schedules. Both gathers enqueue before
compute; every rank waits immediately before that layer's Engram row decoder.
Layer 14 can gather while layers 1 through 13 execute. Hashing still uses the
current chunk's tokens and committed Engram history. Commit and cancel retain
their existing history rules.

The route uses the existing asynchronous mapped-gather host service and
supports ordinary eager and upper-skip prefill without DMA communication. The default TP4 upper-skip profiles use phased DMA
from 512 rows and keep synchronous gathers there, including at 8K and 32K.
Full-prefill profiles and short upper-skip chunks can use prepared gathers.
Media, diagnostic text taps, missing host services and unsupported schedules
retain their existing fallback reasons. DSpark joins each layer after submitting its target producer work and
finishes the gathers after consolidated status reads. No runtime ABI, kernel,
arithmetic order, reduction boundary, output layout, or lock changes are
required.

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

Flash TP4 reserves two 786,432-byte device ID buffers on rank 0 before
serving KV sizing. Each prepared layer
owns 198.75 MiB of pinned staging, within the host service's 256 MiB per-handle
limit. No additional device output buffers are allocated. Each transfer uses
only the current row count, up to 8192.

## Qualification

CPU fixtures compare complete per-stream launch journals and committed state
with immediate and delayed gather completion, check uploaded ID bytes against
the hash oracle, and exercise
TP1, TP2, TP4, both KV layouts, partial rows, handle reuse, cancellation and
allocation failure. Plan tests reject early readers of either layer, reordered
or repeated layer boundaries, mismatched extents and a pending prior gather.
These tests establish scheduling invariants, not GPU numerical equivalence.

The adoption comparison on 2026-10-26 used one diagnostic install with the
previous default and prepared gathers, with paired greedy and sampled seeds
at 511, 8192 and 32768 prompt tokens. All six pairs generated 256 tokens.
Full-vocabulary logits, generated token IDs, logprob words and committed
state, KV, chunk and prefix snapshots matched bitwise (2,758,849,272 compared
bytes per arm). Both full-prefill diagnostic arms set the bundle chunk size
to 8192 and disabled upper skip to enable boundary capture. A native CPU
fixture proved that this capture setting preserves all eight stream launch
sequences at 8K and 32K. The diagnostic build disabled the ready-event result
fast path in both arms.

Separate cold-Engram timings used the original profiles with capture off,
one normal install, concurrency one, full KV pools, paired prompts and seeds,
fresh servers in ABBA order, host-load samples and process fault counters.
The arms differed only in the former gather switch. Matched token IDs and
logprob words also passed on these original profiles. All 12 timing windows
executed with zero replay, no prefix hits, preemption or eviction, and zero
measured table residency after cold advice. KV page counts were equal across
arms: 329219 for full-prefill and 343946 for short default AR. The historical
full-schedule configuration used `prefill_chunk_tokens=0`,
`prefill_upper_skip=false` and a 1280 MiB graph memory reserve.

For each metric, d1 = A1/B1 - 1, d2 = A2/B2 - 1, and spread = |d1 - d2|.
Positive values are inverse-latency gains. Every row passed the new-route
threshold: both pairs positive, mean above spread, and mean at least 1.5%.
The 32K prefill value is the median step time of its four 8192-token chunks.

| Configuration | Tokens | Metric | d1 | d2 | Mean | Spread |
|---|---:|---|---:|---:|---:|---:|
| AR, full schedule | 8192 | TTFT | 21.5644% | 23.0753% | 22.3198% | 1.5109% |
| AR, full schedule | 8192 | Prefill step | 22.1022% | 23.2546% | 22.6784% | 1.1524% |
| AR, full schedule | 32768 | TTFT | 20.6364% | 22.2626% | 21.4495% | 1.6262% |
| AR, full schedule | 32768 | Prefill step | 19.6303% | 20.8057% | 20.2180% | 1.1754% |
| tp4-ar | 511 | TTFT | 9.0102% | 9.1789% | 9.0946% | 0.1687% |
| tp4-ar | 511 | Prefill step | 9.5489% | 9.5507% | 9.5498% | 0.0017% |

Energy is provisional. J/token values in visit order and relative changes
are below; a positive change means more energy. The full-prefill mean was
+0.7919%, and the short mean was -0.7525%. Neither reached the 3% veto.

| Tokens | A1 | B1 | B2 | A2 | d1 | d2 | Mean | Spread |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 250.7428 | 254.9324 | 258.5471 | 258.8253 | 1.6709% | -0.1075% | 0.7817% | 1.7783% |
| 32768 | 394.3390 | 404.6377 | 411.2416 | 415.4260 | 2.6116% | -1.0073% | 0.8022% | 3.6189% |
| 511 | 224.6464 | 223.8887 | 219.0953 | 221.6837 | -0.3373% | -1.1676% | -0.7525% | 0.8304% |

Enforced GPU limits stayed at 275/250/250/250 W throughout both jobs.
No limit or clock was set, and no Xid appeared. Recorded SM clocks in MHz,
in rank order, include the start, end and each visit boundary:

| Boundary | Full-prefill | Short default AR |
|---|---|---|
| Start | 1890/1890/1890/1852 | 1890/1890/1890/742 |
| A1 | 1890/1882/1890/1882 | 180/180/1890/1882 |
| B1 | 1890/1882/1890/1882 | 180/742/1650/1882 |
| B2 | 1890/1890/1890/1882 | 180/180/1815/1882 |
| A2 | 1890/1890/1890/1890 | 180/1882/1890/1597 |
| End | 1890/1890/1890/1890 | 180/1882/1890/1597 |

The timing guard admitted four existing full-prefill fallback entries and
40 short-profile entries, bound to exact symbols, row counts and phases,
with equal counters in each pair. The full-prefill entries are existing
8192-row mxfp8 admission and packed fallbacks for n4096/k1280 and n5120/k576;
the short entries match existing route allowances. This adoption removes
the AR prefill allowance only. DMA integration and those independent kernel
admission gaps remain outside this result.

The `prefill-engram` record reports host gather nanoseconds and
`exposed_ns` per layer. The latter is the maximum rank interval between events
immediately before and after its upload wait. It includes event and submission
overhead; it is not summed rank time or a whole-request critical-path measure.
The served result accepts the existing prepared route. It does not qualify
a cross-chunk interface extension or integration with the DMA schedule.

## Cold large gathers

The bulk-read path starts at 8,192 rows per plane (342 tokens at 24 rows
per token). The earlier threshold was 65,536 rows, reached above 2,730
tokens; these large gathers skip the resident row cache. Before runtime-host read
such gathers through its reader threads, the prepared gather copied them from
the mapping with no read-ahead (the asynchronous default at that time),
one page fault at a time. Measured on
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

Both synchronous and asynchronous read-ahead now default on. The shared
`CHOPINFER_MAPPED_READ_AHEAD=0` switch disables both unless the async-specific
`CHOPINFER_MAPPED_ASYNC_READ_AHEAD` override selects otherwise; its `=0`
value disables async advice alone. The [caller audit](../../../docs/mapped-row-read-ahead.md)
confirms that no in-tree production consumer depends on advice being off.
