# Per-request recurrent state

Some models carry state from token to token that is not KV: a linear-attention
state matrix, a short convolution history, a rolling token window. Chopinfer
treats this as per-request state that the bundle owns and the host accounts
for. The runtime crates never learn what the state means; they reserve room
for it, name its slot in every batch, and tell the bundle when a request is
gone. The [Qwen bundle](bundle-qwen38.md) uses it; the recurrent fixture
(`bundle-recurrent-fixture`) is the conformance bundle for it.

## Manifest

A manifest lists its request state in `recurrent_state_layouts`, one
`BundleStateLayoutV1` per layout, sorted by ID:

```json
"recurrent_state_layouts": [
  {"id": "conv", "bytes": 1024, "alignment_bytes": 64, "lifetime": "request"},
  {"id": "scan", "bytes": 65536, "alignment_bytes": 128, "lifetime": "request"}
]
```

- `bytes` is the size of one complete copy of that layout for one sequence
  on one rank, over every layer that uses it. A model with a scan state of
  `[value_heads, value_dim, key_dim]` F32 per layer and a conv history
  declares two layouts, or one if it packs them.
- Every layout has `request` lifetime, and IDs are unique. Validation also
  refuses a total that overflows `u64`.
- A manifest without the field is unchanged: the field is omitted from
  canonical JSON when empty, and such a bundle sees no new wire or callback.

A bundle that declares layouts also requires `chop.recurrent-state.v1` and
`chop.request-retirement.v1`. The host refuses to serve it without retirement,
because a slot is freed only when the bundle has dropped the request.

## Slots and admission

The host builds a `runtime_host::recurrent::StateSlotPoolV1` with one slot per
sequence the engine may run (`--max-sequences`). Each slot reserves
`total_layout_bytes * (rows + 1)` per rank: the committed state plus one
checkpoint for every row one item may execute before a commit. `rows` is 1
without speculation and `maximum_draft_tokens + 1` with it.

- The pool's total reservation is subtracted from free device memory before
  the serving KV pool is sized, so KV pages never overlap state bytes. This
  applies to bundles on the host-managed serving KV pool
  ([serving KV pool](serving-kv-pool.md)).
- A bundle that allocates storage for every slot during `create`, before it
  calls `acquire_serving_kv`, also requires
  `chop.recurrent-state.preallocated.v1`. Its state is then already outside the
  free memory the pool is sized from, so the host reserves nothing for it;
  subtracting it again would charge the state twice.
- The engine calls `ModelExecutor::admit_request_v1` for each request the
  scheduler admits, before any prefix restore or execution. The host assigns
  the lowest free slot. If that fails, the engine cancels the request.
- A slot returns to the pool only after the bundle's `retire_request`
  callback succeeds. A failed retirement keeps the slot owned, so no later
  request can be given bytes that the bundle may still use.
- Cancellation, finishing and execution errors all end in retirement.

The bundle allocates storage and maps each logical slot to its physical state
entries. Slot IDs are dense, in `0..capacity`; physical entry IDs may change at
commit. Retirement leaves the mapping owned by the logical slot, so reuse
initializes or restores its current entries rather than assuming entry equals
slot. The Qwen bundle exchanges conv, delta and PLE entries at verify commit;
it retains the same checkpoint capacity and state bytes.

## Batch wire

When state is declared, every batch is `chopbatch-v4`: the magic, the length
of an enclosed V3 batch, the V3 batch, then one `u64` slot ID per item in item
order. Slot IDs within one batch are unique. `runtime_bundle_sdk::BatchCodecV1`
decodes it, and `BatchRequestV1::state_slots()` returns the IDs. A bundle
without declared state keeps receiving V1, V2 or V3 unchanged.

## Commit, cancel and speculation

Execution never changes the committed state; it writes tentative
checkpoints.

- A prefill or decode item leaves one checkpoint, the state after its last
  input. `commit` makes it the committed state.
- A verify item with `R` rows leaves `R` checkpoints, the state after each
  row. `commit_accepted_v1` names each verify item's `kept` rows, and the
  bundle commits checkpoint `kept - 1`. It must not recompute the prefix at
  commit time. A bundle may make the kept checkpoint its committed physical
  entry by exchanging indices. It must return the former committed entry to
  the writable checkpoint workspace, and that workspace must never contain
  any slot's current committed entry.
- `cancel` drops tentative checkpoints without exchanging their ownership.
  A verify item whose source is separate keeps the previous committed bytes.
  Bundles with in-place AR or prefill updates declare invalidation on cancel;
  those requests must retire and reinitialize or restore before executing again.

## Prefix caching

With prefix caching on ([serving kv pool](serving-kv-pool.md)), an anchor
must restore request state exactly. The bundle writes the committed state for
a given token count into the snapshot pages it asked for in
`describe_prefix_state_v1`, and reads it back in `restore_prefix_state_v1`.

- Capture happens after a commit at the committed token count, or at an
  interior boundary retained for a declared `anchor_items` capture. The bundle
  must retain the exact state for that boundary until capture completes.
- Restore arrives after admission, so the request already owns a slot. The
  restore callback is keyed by request, and the next batch carries the slot
  ID. A bundle whose state lives in device slots cannot wait for that batch,
  because the snapshot pages are free again when the callback returns. It
  requires `chop.prefix-cache.state-slot.v1` and exports
  `restore_prefix_state_in_slot_v1`; the host then restores through it with
  the slot it assigned at admission, and the bundle writes the snapshot into
  that slot's current mapped entries before returning.
- A refused restore must leave no request state; the engine then prefills
  from zero.

## Limits

- The state reservation is charged only against the host-managed serving KV
  pool. A bundle that allocates KV itself must budget its state in its own
  resource requirements.
- The pool assigns IDs only and never moves state between slots. A preempted
  request retires its slot, and takes a free one when it is
  readmitted and recomputes its state from its tokens or from its anchor.

## Qwen indexed commit qualification

On 2026-10-04, Qwen TP2 qualification compared indexed conv, delta and PLE
commits with the checkpoint-copy implementation. Kernel tests found bitwise
identical committed state and continuation outputs for every acceptance count,
including cancel, repeated verify, decode after verify, prefix capture and
restore, slot reuse, graph replay and PDL. Tests poisoned outputs between
launches. C1 greedy equality, repeated determinism and spec-verify checks passed
at K1 and K3 with token-identical outputs; same-policy logprob and top-token
records also matched exactly.

These GPU results used the indexed implementation as committed on
2026-10-04 and the checkpoint-copy build of 2026-10-03, before a rebase
onto main of 2026-10-04. The implementation patch was unchanged by that
rebase.

The initial C16 trace attributed 5.59% of the K1 step and 1.84% of the K3 step
to recurrent commit copies. The implementation threshold was at least 3% at
C16 for either depth. Adoption also required measured serving benefit and no
K3 C32 regression beyond observed repeat noise.

Two interleaved native serving windows per arm measured the following
throughput. Each window includes prefill and request tails; C32 predominantly
ran decode under the 32-row budget.

| Configuration | Checkpoint copies, tokens/s | Indexed commit, tokens/s | Change |
|---|---:|---:|---:|
| K1 C16 | 990.73 | 1031.45 | +4.11% |
| K1 C32 | 1171.12 | 1177.70 | +0.56% |
| K3 C16 | 813.50 | 812.48 | -0.13% |
| K3 C32, controlled repeat | 1182.62 | 1189.56 | +0.59% |

The first K3 C32 sample included a 387 ms tail graph capture in one indexed
window and averaged -1.75%. A controlled repeat warmed the bucket-8, single-item
tail graph on both ranks before timing. It ran the arms in copy/index/index/copy
order, retained all raw windows, and excluded a window only if it captured a
new graph. All four accepted windows were capture-free. Paired throughput
changes were +0.37% and +0.80%; observed repeat spreads were 0.80% for copies
and 0.37% for indices. No K3 C32 regression was observed. These two samples per
arm do not establish a statistical confidence interval or a general K3 gain.
The qualification and measured-outcome requirements for phase 1 passed.
