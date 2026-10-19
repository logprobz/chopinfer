# DSpark lifecycle integration

`state` owns committed row ranges, slot leases and a projection cursor for
each drafter stage. `execution::Runtime` orders the synchronous host work;
`execution::Host` connects it to the bundle's prepared program and OwnOps.
The program supplies geometry, vocabulary and projection-pass capacity.

Raw feature rows concatenate the target layers' attention inputs in geometry
order. For Flash these are layers 37, 38 and 39, averaged over the hyper
streams before attention, as in the checkpoint reference. They are not
attention outputs or projected drafter KV. Target execution stages every
row separately from committed history.

For each step:

1. Pass all prefill IDs, or one input/bonus ID per decode item, to `execute`.
   Before drafting it projects each drafting item's missing committed rows
   through every stage. Items that do not draft defer their rows to their
   next drafting step, and rows older than the newest `window` are skipped
   because later rows overwrite their ring cells. Deferred rows of all
   drafting items share launches of at most 32 rows, where both dense
   projections are one-row exact and the other operations are row-local,
   and one main projection serves all stages. A commit wider than 32 rows (a prefill chunk or restored window)
   catches up at the item's next step in the per-stage passes of an eager
   catch-up, so every projected row keeps its eager bits.
2. `Host::draft` computes all five block positions, the sequential Markov
   head and confidence scores. Use the shared `WindowBlockVisibility`
   contract with `DraftBlock::main_rows` as its projected-row count. Draft
   keys occupy separate scratch after the main ring. K changes only the
   emitted prefix and the target's K+1 rows.
3. Bounded synchronous readback returns five IDs and five F32 confidence
   scores per drafting item. CPU Engram hash/gather uses the completed
   target ID rows, then upload finishes before target verification starts.
   Hash/gather uses provisional history; commit publishes only kept rows.
4. After host-ledger validation, `commit` checks all retained-row counts
   before invoking `Host::commit_features`. The callback finishes every
   rank's work before returning. Without early termination, acceptance
   depths 0 through 5 keep 1 through 6 target rows. An accepted stop token or
   token limit can end verification without the extra row, so the retained
   count is not an exact accepted-proposal count.

Diagnostics record all five scores. The commit callback exposes only retained
rows, not a stop reason or an independent acceptance count. The leading
`kept_rows - 1` proposals are provably accepted; later emitted proposals are
`Unknown`, including a possible accepted stop token. Un-emitted and cancelled
proposals remain distinct. Unknown outcomes must be excluded from confidence
calibration until independent verifier evidence resolves them.

`calibration::record` encodes these diagnostics for the coordinator's host
log callback. It preserves raw score bits and records the numerical contract
and acceptance evidence. The [confidence record format](../../../dspark-confidence.md)
defines the JSON fields and external calibration artifacts. The full program
must connect this callback when its execution adapter is available.

Each slot's raw history has `window + 6` rows, indexed by absolute position
modulo that capacity. Short commits retain every accepted row. A prefill
chunk larger than the history capacity projects all accepted rows directly
from staging through each stage, in position order, before retaining its
last window. `CommitItem::retain_offset` names the first staged row to copy.
This keeps history bounded while preserving every accepted row's projection.

`transfers` lowers those plans to ordered row ranges. Projection chunks split
at both history and KV-ring wraps and at the prepared operation's capacity,
so a launch never writes two positions to one ring cell. Retention and
snapshot copies split at history wraps. Offsets are rows; the bound device
layouts supply byte strides. Submit every chunk in order on every rank
before acknowledging its projection pass or completing its commit.

`features` binds those raw-row plans to checked device spans. It derives the
BF16 row width from the geometry, rejects overlapping staging and retained
roles, and binds the entire copy list before any IO. The state declaration
uses the same `window + 6` feature stride and `window` committed KV stride;
target rollback headroom is separate. Prefix copies stay chronological when
restored into a different slot.

Cancellation before commit discards provisional target rows and keeps
completed catch-up projections. A failed host operation poisons the instance;
after commit copies begin, cancellation cannot restore overwritten state.

For prefix capture or preemption, catch up every stage, call `snapshot`, and
copy the raw window before retiring the request. For restore, validate the
snapshot and destination slot before any copy, restore that raw window on
every rank, then call `restore_after_copy`. The stage cursors restart at the
window's first position. No derived ring or cursor from a prior slot owner
survives restore. A failed copy must poison the runtime.

The configured position capacity must hold the full five-position block
even when K is smaller. If it cannot, admission refuses that draft step; the
caller can schedule K=0. Confidence values never change this choice or the
emitted prefix.

The CPU tests cover state transitions, retained contents, projection order
and the synchronous host-call journal. Their synthetic logits check budget
and ordering behavior. Real model logits, confidence-head accuracy and GPU
execution require the model program and kernel tests.

Vendor acceptance parity cannot be established from the pinned reference:
`generate.py` runs only AR through `forward`, and `model.py` explicitly leaves
the speculative decoding loop out of scope. Its `forward_spec` defines the
drafting math, but supplies no reference acceptance loop.

The Markov and confidence projections have model OwnOps contracts and F64
oracles in `contract/src/own_ops/dspark`. SM120 prepares them through the
same OwnOps interface as the other model kernels. Each head keeps F32
outputs and uses ordered F32 chains of at most 128 products, with no TF32.
Markov completes its projection before adding the base logits, which may
be updated in place. Confidence reads the F32 collapsed hidden row before
vocabulary RMSNorm without BF16 rounding, and the predecessor token's
exactly decoded BF16 Markov embedding.

One launch processes one position across block items. Use a stride of five
vocabulary rows for logits, five hidden rows for confidence inputs and five
elements for confidence scores, with views starting at the chosen position.
The device live count is the number of drafting items. Execute all five
positions regardless of the scheduler's emitted prefix. The five Markov
steps remain sequential because each consumes the previous sampled token.

`model-deepseek41-lab` checks preparation and locked cubins with the
`head_cpu` filter. Its `head_gpu_markov_and_confidence_match_oracles_and_replay`
test checks F64 bounds, the exact chain order, in-place logits, live tails,
stride gaps, relaunch, graph replay and delayed input producers. Queued GPU
runs supply prebuilt head and test-producer cubins through
`CHOPINFER_DEEPSEEK41_HEAD_CUBIN` and `CHOPINFER_DEEPSEEK41_PROBE_CUBIN`;
the job verifies the snapshot hashes before executing. The test producer
lives only in the lab and is never included in the serving bundle.
The GPU test defaults to `synthetic_small` and refuses a binary without its
12-layer geometry. Compile the lab with `CHOPINFER_DEEPSEEK41_GEOMETRY_DIRS`
pointing to `fixtures/deepseek41/geometries/synthetic-small`. Later checks
select another compiled geometry with `CHOPINFER_DEEPSEEK41_HEAD_GEOMETRY`.

On 2026-10-17 the SM120 head tests passed on the 12-layer synthetic geometry
(hidden width 1024, Markov rank 256) at a 250 W power cap. All 24 cases
passed F64 bounds and exact chain-order checks, including live counts
0, 1, 3 and 7, in-place logits, stride gaps, graph replay and delayed
producers. Maximum bound use was 0.06178. A separate memcheck replay passed
the same 24 cases with zero errors and no Xids. This verifies the two head
kernels under the original input contract. The confidence hidden input has
since been corrected to F32; those confidence results are superseded until
the corrected cubin passes its GPU checks at those shapes.

On 2026-10-18 the corrected F32 head cubin passed 28 normal and 28 memcheck
cases at a 250 W power cap, with zero memcheck errors and no Xids. These
cover production-width 5120 confidence, including the unrounded hidden
counterexample, and the production TP4 Markov matrix. Maximum bound use was
0.061771. Corrected width-1024 confidence, full-stage parity and
real-checkpoint execution remain separate integration gates.

The [synthetic lowering interface](../lower/dspark/README.md) binds catch-up,
block attention and MoE, and the five-position head chain. `staging` connects
bounded draft readback to the mapped Engram rows before verification. The
program still needs to connect these modules to its instance lifecycle.
