# Mapped-row read-ahead

Synchronous and asynchronous mapped gathers both use adaptive read-ahead by
default. `CHOPINFER_MAPPED_READ_AHEAD=0` disables both paths, while `=1`
enables both. `CHOPINFER_MAPPED_ASYNC_READ_AHEAD=0` or `=1` overrides only the
asynchronous path. The explicit measurement-control modes `off`, `sync`,
`async` and `both` keep their existing meanings.

The [policy](../crates/runtime-host/src/host_rows/gather/policy.rs) controls
I/O advice, not gather bytes, row order, cache admission or synchronization.
Queued descriptor reads already fetch missing rows and ignore the advice
boolean. The legacy mmap path advises unread rows after a slow group using
`MADV_WILLNEED`; advice errors are ignored. See the
[gather implementation](../crates/runtime-host/src/device/mapped_gather.rs)
and [mapped-row hints](../crates/runtime-host/src/host_rows.rs).

## Caller audit

The source audit on 2026-10-28 includes these production async consumers. None
reads the policy setting or relies on advice being off.

| Consumer | Async gather use |
| --- | --- |
| DeepSeek AR decode | [Per-layer gathers for decode buckets](../models/deepseek41/bundle/src/instance/ar_engram.rs), with synchronous fallback. |
| DeepSeek DSpark target and allocated verification | [Target layer gathers](../models/deepseek41/bundle/src/instance/dspark/cycle.rs). |
| DeepSeek DSpark first-target-row prefetch | Separate early gathers in the same target-cycle module. |
| DeepSeek DSpark decode without drafts | Reuses the target layer gathers with deferred host-hashed IDs. |
| DeepSeek eager text prefill | [Prepared per-layer gathers](../models/deepseek41/bundle/src/instance/prefill_engram.rs), including DSpark prefill. |
| DeepSeek DMA text prefill | Prepared per-layer gathers by default on the TP4 native-KV single-item upper-skip path at 512-8192 rows. |
| GLM | No mapped-gather caller; text and encoder plans have no special mapped views. |
| MiMo | No mapped-gather caller; acquisition places its views in the device arena. |
| Qwen | No mapped-gather caller. Its optional n-gram table uses pinned host placement read by kernels, a separate route. |

Shared callers are the ABI wrappers and trampolines, host-service forwarding,
CUDA and fake-device gather implementations, and conformance accounting
forwarders. The API has no read-ahead parameter. DeepSeek's model fake host
emulates enqueue/wait/completion separately from this runtime policy.

Shared test callers cover the async-gather and layer-gather host fixtures,
ABI forwarding and unsupported service tails, direct mapped gathers,
resident-cache gathers and cache routing. They assert bytes, validation,
ordering, accounting or routing. Tests of advice being disabled select that
setting explicitly. DeepSeek's verify-width lab sets the shared switch to
`0`, which continues to disable both paths. No in-tree production caller
requires the old async default; external bundle consumers were not audited.

## Engram route guard

The `engram_gather` family records host calls in the selected-entry `route`
object of `chopinfer.step.v1` when `CHOPINFER_ROUTE_COUNTERS=1`. The field
named `launches` counts host calls for this family, not CUDA launches or
rows. One synchronous call copies one layer's codes or scales on one rank.
One prepared enqueue can cover both planes and every rank for a layer.
Neither count measures I/O time, page faults or read-ahead activity.

Prepared calls are Fast with symbol `host.enqueue_mapped_gather` and entries
`engram/prepared/ar_decode`, `engram/prepared/prefill`,
`engram/prepared/dspark_target`, `engram/prepared/dspark_early` and
`engram/prepared/dspark_undrafted`. Synchronous calls are Baseline with
symbol `host.gather_mapped_rows` and entry `engram/sync/<reason>`:

| Reason | Synchronous selection |
| --- | --- |
| `media` | Media flags and token hashing retain the synchronous staging boundary. |
| `taps` | Text taps retain their diagnostic stage and readback order. |
| `no_async_service` | The host lacks the optional prepared mapped-gather services. |
| `mixed_target` | A host target combines input kinds without a prepared layer plan. |
| `host_target` | Supplied target/verify rows retain their host hashing boundary. |
| `no_prepared_gather` | The phase has no prepared gather state or hashes. |
| `gather_capacity` | Undrafted hashes exceed the prepared gather capacity. |

Prefill reasons take precedence in this order: media, taps, missing async
service, then unavailable preparation.
Reasons are carried from preparation state or the selected phase, never
inferred from journal text. A graph reservation failure keeps the prepared
gather and executes eagerly, so it is not a synchronous fallback.

The shared route sweep independently counts synchronous host calls in every
fake-host journal mode and resets the count before each measured execute and
commit. DeepSeek's prepared-wait emulator does not add synchronous evidence.
Raw `host.gather_mapped_rows` counts must exactly match selected entries,
including when no tag was declared. Missing and duplicated selections fail
before allowance checks. Fast metadata on this reserved synchronous symbol is
invalid, even when its counts match. Every synchronous shape needs an allowance with a
reason and removal task, including a constant synchronous route. Other models with no mapped gathers need no tags or allowances.

The [profile checker](../tools/route_guard/check.py) accepts `host.` symbols
from step records without Nsight kernel evidence. It still validates entry
metadata, family totals and allowances, and retains raw-evidence checks for
kernels in a mixed capture. The raw synchronous count is a fake-host sweep
check, not an independent counter in serving step records. No synthetic CUDA
launch or new journal line is introduced by this guard.

## DMA prefill

Prepared Engram gathers are the default for reductions of 512 through 8192
rows on the existing TP4 native-KV single-item DMA path. Both layers' IDs are
uploaded into dedicated buffers and their gathers enqueue before compute.
Each layer joins every target stream immediately before its Engram decoder.
Collective order, row intervals and arithmetic stay the same. Completion is
checked after consolidated status readbacks, before publishing results.

With `CHOPINFER_ROUTE_COUNTERS=1`, submitted DMA owner sums appear in the
entries with family `prefill_dma` and their `chop_comm_owned_sum_` symbols.
Fast entries do not increment the fallback-only family totals. Require this
execution witness in every measured prefill step, plus two prepared gathers
and a completion record per prepared chunk. Its `waits` field is the configured
rank count times layer count, not an independent event counter. The completion
record establishes that both joins reached the result boundary. A selected route without submitted owner sums does not witness executed DMA
arithmetic.

Preparation audits pending IDs and destinations against the original launches
(including tensor maps and pointer arrays), DMA transfer spans, generated
rounding and sum launches, and the producers and consumers moved into owner
intervals. No pending gather span may be touched by that work. DMA owner
joins still precede subsequent compute; eager prefill has no graph side-stream
fork. Failed gathers cannot be reused. Destruction drains compute, DMA and
side streams, then releases gather handles, whose host service fences pending
callbacks and transfers before any referenced destination allocation is freed.
A refused status or result sink settles healthy pending gathers: join remaining
layers, synchronize every target stream, then complete before another step can
reuse their rows. AR then serves the next request. DSpark retains its existing
terminal host-error policy; its target settles healthy gathers before returning
the error that poisons the drafter runtime. Failed callbacks or handles remain
terminal. An unmatched
collective keeps resources retained and must not be synchronized.

Plan-time overlap or collective-balance audit failures refuse the request before
enqueue. Preparation fails closed instead of silently selecting synchronous gathers. Its owner-sum tags change route metadata when counters are enabled;
journal identity does not claim route-tag identity.

Two default-off candidates target the host waits of a DeepSeek DSpark mixed
step, a DMA prefill beside verify rows. `CHOPINFER_PEER_COPY_BACKPRESSURE=event`,
read once when the CUDA host backend starts, makes a full peer-copy queue wait
for its oldest copy's completion event instead of synchronizing that copy's
stream; `stream`, the default, keeps the stream wait. The DeepSeek model
option `dspark_mixed_prepared_engram` replaces the synchronous gather of the
verify rows (`engram/sync/mixed_target`) with prepared gathers
(`engram/prepared/dspark_mixed_target`) that begin after the prefill's last
Engram join. It admits only text-only single-request DMA prefills of 512
through 8192 rows beside graphed verify phases of at most 64 rows, on
instances that serve images too; every other mixed step keeps the synchronous
gather. Both are candidate switches with CPU evidence
only; GPU exactness and step-time acceptance decide adoption.

The default-off DeepSeek option `dspark_early_prefill_engram` moves a DSpark
step's prefill gathers earlier: the bundle hashes the prefill rows, whose ids
read only their own tokens and committed history, and begins their prepared
gathers before catch-up and drafting rather than just before the prefill's
launch. The step then reuses those ids for its target. Only where the target
takes the prepared gathers (no encoder rows, no text taps) does it begin
early; the previous step's gathers settled at execute entry, and no work the
step issues before the prefill's joins touches the gathers' ids or rows. A
step refused after the early begin settles its healthy gathers before it
returns, as a refused issue does: it joins them on every target stream,
synchronizes those streams and completes them, so the refusal that poisons
the runtime leaves no callback pending. The ids, the gathered rows, every
launch and every result are unchanged; CPU evidence only, as for the
candidates above.

Two default-off host knobs shorten a prepared gather's callback.
`CHOPINFER_MAPPED_GATHER_INIT` chooses when the callback zeroes its pinned
outputs, whose uploads copy each plane's live rows: unset or `extents`, the
default, zeroes each stream's output before every gather up to its last
plane's live end, which for a DeepSeek layer includes its codes plane's
whole 8192-row capacity, about 48 MiB a rank; `failure` zeroes the live
rows only after a gather that fails, since one that succeeds writes every
byte of them. The outputs are zeroed once when they are allocated either
way, so the callback never sees uninitialized bytes. `CHOPINFER_MAPPED_FANOUT_COPIERS`, 1
(the default) to 16, copies the other ranks' planes of a large leader's
rows on that many threads, which only copy host bytes and finish before
the callback returns; the fetch snapshot's `parallel_fanouts` counts the
gathers that used them. Routes, reads and bytes stay the same. On
2026-11-06, a CPU benchmark of a four-rank DeepSeek layer gather (24 ids
a row, a warm page cache, a shared build host) spent 7.8, 14.5 and 20.1 ms
on the leading rank's reads at 1536, 2890 and 4096 prefill rows, 2.3 ms
zeroing by default, and
0.7, 1.7 and 2.5 ms copying the other three ranks' 28, 52 and 74 MiB of
fanout on one thread (0.4, 1.3 and 2.1 ms on eight). CPU evidence only; served step time decides adoption.

Cold-row experiments must keep async read-ahead enabled. The shared defaults
already enable it; `CHOPINFER_MAPPED_ASYNC_READ_AHEAD=1` makes the setting
explicit even when another read-ahead control was inherited. The bundle
does not override host policy. Reset the mapped-row cache and advise only the
sealed Engram file ranges before each cold visit, never host-wide caches.
CPU ordering tests do not establish cold-row latency or GPU equality. Media
is the next possible caller to investigate if operator traffic uses it often;
its hashing and destination lifetimes need a separate audit.

## Adoption evidence

On 2026-10-28, TP4 native-KV AR and DSpark matched full-vocabulary logits,
complete chunk state and sampled output bits with prepared DMA gathers on
and off, including a 25345-token prompt with a 769-row final DMA chunk.
The odd-width exactness runs disabled prefix caching to avoid the rounded-end
cut. Timing kept prefix caching enabled and compared 32 output tokens per
request at temperature 1 with matched seeds and fresh cache salts.

Warm TTFT ABBA reduced latency in every measured cell. For d1=B1/A1-1 and
d2=B2/A2-1, both deltas favored prepared gathers, and each mean reduction
exceeded its pair spread and the 1.5% new-route threshold:

| Prompt tokens | AR reduction / spread | DSpark reduction / spread |
| --- | --- | --- |
| 512 | 6.797% / 3.387 percentage points | 1.839% / 0.452 percentage points |
| 8192 | 12.840% / 1.015 percentage points | 11.769% / 0.685 percentage points |
| 25345 | 11.123% / 1.384 percentage points | 10.728% / 0.002 percentage points |

All matched output IDs and logprob bits agreed. Every warm measurement and
its cache check had zero major faults. GPU power limits remained unchanged
through each job; sampled clocks were recorded per visit. The incomplete
DSpark long-prompt visit that failed its quiet-host check was excluded in
full. Deliberately cold TTFT was diagnostic only.

The synchronous host-call timer fell to zero on prepared DMA chunks, while
layer-1 join time remained exposed. This does not imply that all gather work
was hidden. Media, taps, host-target rows, missing async service and the other
listed preparation fallbacks still use their synchronous path. The candidate
option and DMA-only synchronous fallback are removed. Default tests exercise
the adopted path. The short-prefill graph candidate was removed after
regressing continuation-prefill timing; see the
[removal record](../models/deepseek41/README.md#removed-short-prefill-graph-candidate).
