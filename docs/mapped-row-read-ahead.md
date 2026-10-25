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

The 2026-10-24 source audit found these production async consumers. None
reads the policy setting or relies on advice being off.

| Consumer | Async gather use |
| --- | --- |
| DeepSeek AR decode | [Per-layer gathers for decode buckets](../models/deepseek41/bundle/src/instance/ar_engram.rs), with synchronous fallback. |
| DeepSeek DSpark target and allocated verification | [Target layer gathers](../models/deepseek41/bundle/src/instance/dspark/cycle.rs). |
| DeepSeek DSpark first-target-row prefetch | Separate early gathers in the same target-cycle module. |
| DeepSeek DSpark decode without drafts | Reuses the target layer gathers with deferred host-hashed IDs. |
| DeepSeek eager text prefill | [Prepared per-layer gathers](../models/deepseek41/bundle/src/instance/prefill_engram.rs), including DSpark prefill. |
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
| `ar_prefill` | AR prefill preparation is off by default. |
| `media` | A prefill has media. |
| `taps` | A prefill records text taps. |
| `prefill_schedule` | A prefill uses an alternate DMA, micro or overlap schedule. |
| `no_async_service` | The host lacks prepared mapped-gather services. |
| `ar_decode_disabled` | The AR decode preparation switch is off. |
| `prefill_disabled` | DSpark prefill preparation is off. |
| `undrafted_disabled` | DSpark undrafted decode preparation is off. |
| `mixed_target` | A host target combines different input kinds. |
| `host_target` | Other host-target phases, including supplied verify rows. |
| `no_prepared_gather` | No prepared gather state serves the phase. |
| `gather_capacity` | Undrafted hashes exceed the prepared gather capacity. |

Prefill reasons take precedence in this order: media, taps, alternate
schedule, missing async service, then disabled or unavailable preparation.
Reasons are carried from preparation state or the selected phase, never
inferred from journal text. A graph reservation failure keeps the prepared
gather and executes eagerly, so it is not a synchronous fallback.

The shared route sweep independently counts synchronous host calls in every
fake-host journal mode and resets the count before each measured execute and
commit. DeepSeek's prepared-wait emulator does not add synchronous evidence.
Raw `host.gather_mapped_rows` counts must exactly match selected entries,
including when no tag was declared. Missing and duplicated selections fail
before allowance checks. Every synchronous shape needs an allowance with a
reason and removal task, including a constant synchronous route. AR prefill
uses `ar-prefill-engram`; old test-default routes name the chunk that removes
them. Other models with no mapped gathers need no tags or allowances.

The [profile checker](../tools/route_guard/check.py) accepts `host.` symbols
from step records without Nsight kernel evidence. It still validates entry
metadata, family totals and allowances, and retains raw-evidence checks for
kernels in a mixed capture. The raw synchronous count is a fake-host sweep
check, not an independent counter in serving step records. No synthetic CUDA
launch or new journal line is introduced by this guard.
