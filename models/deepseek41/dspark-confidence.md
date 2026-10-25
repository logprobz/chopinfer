DSpark confidence records use the schema
`chopinfer.deepseek41.drafter-confidence.v1`. One record describes one
draft block after commit, or its cancellation. K=0 steps produce no block
record. Every record keeps all five computed positions, including positions
outside the scheduler's emitted prefix.

The record carries opaque instance, execute-step and request IDs, the number
of committed main rows, the emitted budget K, K+1 verify rows, five proposal
token IDs, five raw F32 confidence bit patterns and five outcome labels.
Instance and step IDs distinguish slot reuse and retries at the same token
position. Confidence is an unnormalized score, not a probability. Integer
IEEE-754 bit patterns preserve NaN, infinity and signed zero without invalid
JSON numbers or a storage-rounding step.

The outcome labels are `accepted`, `unknown`, `not_emitted` and `cancelled`.
The current commit callback proves only the leading `kept_rows - 1`
acceptances. Later emitted proposals remain unknown: an accepted EOS or a
token limit can stop verification without its extra row. Unknown outcomes
must never be treated as rejections. Cancellation and un-emitted positions
are excluded from calibration. The current records are partially labelled;
fitting a rejection threshold also needs independent verifier evidence.

The model-owned `dspark::calibration::record` encoder accepts completed
lifecycle diagnostics. The staging adapter connects `Host::record_diagnostics`
to it on the coordinator, using `|message| services.log(message)` as its sink.
The coordinator assigns a unique instance ID
within the run and a new step ID per execute attempt, including retries.
The model-neutral `drafter-confidence.v1` suffix selects the host's bounded,
best-effort routine logger. It enqueues `bundle: ` followed by the complete
JSON object without waiting for stderr. When the queue fills, the producer
drops whole records and increments the existing host-wide dropped-line
counter. The writer reports accumulated drops when output resumes and at
shutdown; the bounded flush result also exposes the total. These counts
include other routine diagnostics, so they do not identify which blocks
were lost.

Logging failures do not change serving. Custom sinks must also avoid waiting
for output progress. The calibration runner must check recording coverage,
uniqueness of `(instance_id, step_id, request_id)` and dropped-line reports;
it must not treat an incomplete capture as a complete calibration set.
Serving never reads these JSON records. With `--speculation-confidence`,
the device allocator reads the live confidence scores and ranks draft
prefixes within the balanced plan's fixed verify width. It multiplies the
calibrated conditional probabilities through each position, then assigns
spare rows by that prefix survival. The floor is a minimum draft count per
item, reduced when it cannot fit, not a confidence threshold. Allocation
fills the width even when every score is low. It is bypassed when only one
item can compete for rows, when the floor fills the width, or when all
reserved drafts already fit.

The drafter computes five proposals and scores per block. The default
profile already verifies up to five drafts at C1. At C4 its 16-row width
leaves 12 draft rows to distribute, while the default 32-row capacity can
reserve all five drafts per item. At C8 the same capacity fits only three
reserved drafts per item; reserving five requires 48 rows, even when the
target still computes 32. Request output and KV limits can reduce these
reservations. See [serving profiles](../../docs/serving-profiles.md#drafter-confidence)
for the calibration and [depth tuning](../../docs/depth-tuning.md#confidence-calibration)
for fitting it from sampled runs without allocation. Records from allocated
steps cannot distinguish unverified drafts from rejected ones and must not
be used to fit a calibration.

Calibration runs keep the server log outside the repository and collect the
records as newline-delimited JSON at
`$CHOPINFER_DATA/deepseek41/dspark-confidence/<run-id>/blocks.jsonl`.
A sibling `manifest.json` identifies the source log digest, checkpoint
identity, bundle digest, revision, geometry, rank count, profile, workload
digest, sampling parameters and confidence numerical contract. The runner
supplies these facts from the actual run; no placeholder provenance is
accepted for calibration. Different runs retain separate manifests.

For example, a committed K=3 block with one proven accepted proposal is:

```json
{"schema":"chopinfer.deepseek41.drafter-confidence.v1","confidence_contract":"deepseek41.confidence.hidden-f32.markov-bf16.acc-f32-chain128.out-f32.v1","instance_id":"1","step_id":"9","request_id":"17","main_rows":128,"computed_positions":5,"emitted_positions":3,"verify_rows":4,"completion":"committed","acceptance_evidence":"retained_rows_lower_bound","draft_ids":[11,12,13,14,15],"confidence_f32_bits":[1065385984,1065353216,1056964608,0,3212836864],"outcomes":["accepted","unknown","unknown","not_emitted","not_emitted"]}
```

Decode each score as `struct.unpack('<f', struct.pack('<I', bits))[0]` in
Python. Preserve the original bits, and exclude non-finite scores from a
threshold fit. Keep proposal positions in order when testing a prefix cutoff.
