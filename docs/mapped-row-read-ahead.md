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

The source audit includes these production async consumers. None
reads the policy setting or relies on advice being off.

| Consumer | Async gather use |
| --- | --- |
| DeepSeek AR decode | [Per-layer gathers for decode buckets](../models/deepseek41/bundle/src/instance/ar_engram.rs), with synchronous fallback. |
| DeepSeek DSpark target and allocated verification | [Target layer gathers](../models/deepseek41/bundle/src/instance/dspark/cycle.rs). |
| DeepSeek DSpark first-target-row prefetch | Separate early gathers in the same target-cycle module. |
| DeepSeek DSpark decode without drafts | Reuses the target layer gathers with deferred host-hashed IDs. |
| DeepSeek eager text prefill | [Prepared per-layer gathers](../models/deepseek41/bundle/src/instance/prefill_engram.rs), including DSpark prefill. |
| DeepSeek DMA text prefill | Prepared per-layer gathers when the default-off candidate `prefill_dma_engram=true` is selected, on the TP4 native-KV single-item upper-skip path. |
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
| `prefill_schedule` | Upper-skip DMA prefill with `prefill_dma_engram=false` (the default). |
| `no_async_service` | The host lacks the optional prepared mapped-gather services. |
| `mixed_target` | A host target combines input kinds without a prepared layer plan. |
| `host_target` | Supplied target/verify rows retain their host hashing boundary. |
| `no_prepared_gather` | The phase has no prepared gather state or hashes. |
| `gather_capacity` | Undrafted hashes exceed the prepared gather capacity. |

Prefill reasons take precedence in this order: media, taps, default DMA
schedule, missing async service, then unavailable preparation.
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

## DMA prefill candidate

`--model-option prefill_dma_engram=true` enables the candidate for reductions
of 512 through 8192 rows on the existing DMA path. Both layers' IDs are
uploaded into dedicated buffers and their gathers enqueue before compute.
Each layer joins every target stream immediately before its Engram decoder.
Collective order, row intervals and arithmetic stay the same. Completion is
checked after consolidated status readbacks, before publishing results.

Preparation audits pending IDs and destinations against the original launches
(including tensor maps and pointer arrays), DMA transfer spans, generated
rounding and sum launches, and the producers and consumers moved into owner
intervals. No pending gather span may be touched by that work. DMA owner
joins still precede subsequent compute; eager prefill has no graph side-stream
fork. Failed gathers cannot be reused. Destruction drains compute, DMA and
side streams, then releases gather handles, whose host service fences pending
callbacks and transfers before any referenced destination allocation is freed.
A terminal execution or cancellation failure keeps resources retained.

Cold-row experiments must keep async read-ahead enabled. The shared defaults
already enable it; `CHOPINFER_MAPPED_ASYNC_READ_AHEAD=1` makes the setting
explicit even when another read-ahead control was inherited. The candidate
does not override host policy. Reset the mapped-row cache and advise only the
sealed Engram file ranges before each cold visit, never host-wide caches.
CPU ordering tests do not establish cold-row latency or GPU equality. Keep
the candidate off until full-logits/state exactness and cold/warm TTFT ABBA
support adoption. Media is the next possible caller to investigate if operator
traffic uses it often; its hashing and destination lifetimes need a separate
audit.
