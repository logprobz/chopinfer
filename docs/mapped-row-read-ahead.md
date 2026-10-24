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
