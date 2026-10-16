# Diagnose the first divergence

Start with the smallest existing comparison that preserves the failure's
inputs and state. A changed generated token alone does not establish a
regression: changed batching and floating-point reductions can change tokens.
Keep the checkpoint, token inputs, schedule and numerical contract attached to
every result.

| Question | Comparison to choose | What it establishes |
|---|---|---|
| Did rendering or tokenization change? | `tools/qual/frontend_goldens.py` on frontend captures | Equality of the compared token IDs, with missing and omitted scope reported |
| Did a package or dispatch change fixed-schedule results? | `tools/dev/fixed-batch-logits.sh prepare-compare`, `run-compare`, then `tools/dev/diagnostics.py` | Recorded bitwise logits and sampling comparison under one host build |
| Does one decode bucket's arithmetic depend on served capacity, slot placement or the other rows' lengths? | `tools/dev/fixed-state-replay.py prepare`, `run`, `compare` | Bitwise repeats, capacity and placement pairs of one fixed state, and the probe-row logprob change from reordered filler lengths |
| Which service call changed? | Fake-host journal export and `diagnostics.py --mode journal` | The first structural event or launch argument difference, not GPU arithmetic |
| Did a refactor only rename kernel entries? | Journal captures at both commits, `tools/journal_proof/proof.py` with a change spec | Every pinned fake-host journal identical or changed only as the spec declares |
| Which layer or operation accumulated error? | The model's oracle, activation comparator or kernel-lab test | Its declared bound on matched inputs, shapes and numerical settings |

For upstream GPU comparisons, use the taps and `tapdiff` of the
[reference runner](reference-runner.md).

All diagnostic tools share one outcome scheme:

| Status | Exit | Meaning |
|---|---|---|
| `match` | 0 | The named comparison completed for its recorded scope |
| `diverged` | 1 | A completed first-difference record identifies the observed failure; later steps were not compared |
| `incompatible` | 2 | Required identities differ, or a package change was not declared |
| `insufficient` | 3 | Evidence is absent, incomplete, stale, unsupported or over budget |

These are diagnostic outcomes, not release qualification. Store inputs,
packages, reports and payloads under `$CHOPINFER_DATA` or another external
directory; repository paths, symlinks and overwrites are refused. Each output
directory must be fresh.

## Try the CPU fixture

Use the same `local.env` as `ci/cpu.sh`, locked offline Cargo and an external
target directory. Build dependencies may need CUDA 13.2 even though this
fixture uses no device.

```bash
source "${CHOPINFER_LOCAL_ENV:?set the local environment file}"
export CARGO_NET_OFFLINE=true
export CARGO_TARGET_DIR="$CHOPINFER_DATA/diagnostic-target"
DIAG_ROOT="$CHOPINFER_DATA/diagnostics/first-example"
mkdir -p "$(dirname "$DIAG_ROOT")"
CHOPINFER_DIAGNOSTIC_FIXTURE_DIR="$DIAG_ROOT" \
  cargo test -p runtime-host --lib --locked --offline \
  executor::fixed_batch::tests::diagnostic_fixture_reports_first_divergence \
  -- --exact --nocapture
python3 -B tools/dev/diagnostics.py compare --mode fixed-schedule \
  --reference "$DIAG_ROOT/baseline" --candidate "$DIAG_ROOT/matching"
python3 -B tools/dev/diagnostics.py compare --mode fixed-schedule \
  --reference "$DIAG_ROOT/baseline" --candidate "$DIAG_ROOT/mismatch"
```

The fixture writes through the real recorder. The mismatch names step 1,
request 1, row 0, item 0, state slot 7, F32 shape `[4]`, element 2: expected
`0x80000000` (negative zero), observed `0x3fa00000` (1.25). The report
includes a reproduction command that reruns the fixture into a fresh
directory; run it from the checkout that built the test binary. Replace
`mismatch` with `incompatible`, `incomplete`, `over-budget`, `stale`,
`truncated` or `trailing` to see the other outcomes.

## Reproduce one device case

Compare two direct packages of one checkpoint (for example each build's
`chopinfer-model-<m> prepare --direct` output, which the redeploy scripts'
`--prepare-only` prints) on otherwise idle GPUs. Both arms load their weights
from `--checkpoint`. Commit the source before preparation.

```bash
DIAG_GPU_ROOT="$CHOPINFER_DATA/diagnostics/fixed-verify"
bash tools/dev/fixed-batch-logits.sh prepare-compare \
  --root "$DIAG_GPU_ROOT" --ranks 4 --topology qwen38-flash-tp4 \
  --base-bundle "$BASE_BUNDLE" --tip-bundle "$TIP_BUNDLE" \
  --checkpoint "$CHOPINFER_QWEN38_CHECKPOINT" \
  --base-env CHOPINFER_SPECULATIVE_WIDE_ROWS=1 --tip-env CHOPINFER_SPECULATIVE_WIDE_ROWS=1 \
  --concurrency 16 --max-sequences 32 --verify-rows-per-item 8
bash tools/dev/fixed-batch-logits.sh run-compare 0,1,2,3 \
  --root "$DIAG_GPU_ROOT" --ranks 4 --minutes 25
```

Both packages need their drafter (Qwen3.8 `--drafter mtp`, MiMo `--drafter
dflash`) and must verify every live request's rows in one step; Qwen3.8 needs
`CHOPINFER_SPECULATIVE_WIDE_ROWS=1` for that above 32 rows, as above.
Preparation refuses packages that cannot replay the trace and lists their
differences: declare each intended one with a repeated `--allow-difference
FIELD`, and numerical switches with `--base-env`/`--tip-env KEY=VALUE`. Trace,
vocabulary, host build, topology, capacities and sampling contract must match.
`run-compare` and `replay` refuse the written `prepared.json` if the recorder
binary or a package manifest changed since.

The report names the first observed difference (scheduled input,
transaction, sampling input, logits element or sampled record); it cannot
identify an earlier unrecorded model operation. To rerun one case:

```bash
bash tools/dev/fixed-batch-logits.sh replay 0,1,2,3 --ranks 4 \
  --prepared "$PREPARED" --trace "$CASE/trace.json" \
  --concurrency 16 --arm tip --reference "$CASE/base" \
  --control "$CASE/control" --output "$CASE/tip-replay"
```

Decode comparisons take `--concurrency` 1, 16 or 32; C1 covers the one-
and two-row decode steps. For fixed prefill batches, prepare with
`--prefill-only --concurrency N`
and use `replay` with an explicit version-4 trace. That trace contains one
greedy prefill step, 1 to 32 items and at most 8192 input tokens in total.
Each item may set `anchor: true`; omitted anchors remain false. This mode
requires no drafter and compares one logits row per item. `run-compare`
continues to require the automatic 13-step verification trace.

Supported: C16/C32, TP2/TP4, verify widths 2 to 8 and capacities
up to 32. The 13-step trace rejects all drafts, so partial draft acceptance is
not tested. Prepared diagnostic descriptions carry `limits.logits_bytes`,
`limits.step_bytes` and `limits.metadata_bytes` (defaults: 4 GiB, 256 MiB
and 16 MiB; the metadata reader ceiling is 256 MiB). Journal exports allow
64 MiB. The trace is never shrunk to fit.

## Separate capacity, placement and composition at one bucket

`tools/dev/fixed-state-replay.py` forces a teacher reference through one
decode bucket: its first texts are probes, interleaved with fillers cut
from its longest text at evenly spread lengths. Every arm replays the same
trace bytes and KV page count. Arms: a repeat at the bucket's capacity and
at twice it (determinism), the two capacities (capacity), twice the
capacity with idle holders in the low slots (placement), and the fillers'
lengths reversed (composition, at both capacities). `compare` reports
bitwise equality of every shared row and, for probe rows, the mean absolute
target logprob change per decode step and each arm's error against the
reference.

```bash
ROOT="$CHOPINFER_DATA/diagnostics/fixed-state"
python3 -B tools/dev/fixed-state-replay.py prepare --root "$ROOT" \
  --reference "$REFERENCE" --bundle "$PACKAGE" --checkpoint "$CHECKPOINT" \
  --topology "$TOPOLOGY" --vocabulary "$VOCABULARY" --page-tokens "$PAGE_TOKENS"
python3 -B tools/dev/fixed-state-replay.py run --prepared "$ROOT"/prepared-*/prepared.json \
  0,1,2,3 --minutes 22 --compare-minutes 5
```

`prepare` builds the recorder from the committed source unless `--binary`
names one, and dry-runs every arm in it without a device
(`CHOPINFER_BATCH_REPLAY_DRY_RUN=1`): package, checkpoint, trace, capacity
and pages are checked before any GPU time. Each arm records logits in
memory order with no reference, so `compare` can rerun on CPU.

## Inspect model journals and deeper probes

Export the Qwen mixed prefill/decode/cancel journal:

```bash
CHOPINFER_QWEN38_JOURNAL_DIR="$CHOPINFER_DATA/diagnostics/journal-reference" \
  cargo test -p bundle-qwen38 --lib --locked --offline \
  fake_host_tests::qwen38_launch_journal_is_unchanged -- --exact --nocapture
```

`CHOPINFER_QWEN38_GEOMETRY_JOURNAL_DIR` exports one journal per listed
geometry topology from
`fake_host_tests::every_geometry_topology_runs_the_fake_host_journal_and_lifecycle`.
Set `CHOPINFER_QWEN38_GEOMETRY_DIRS` at build time to include fixture geometries.

Export a second execution under another root and compare with
`diagnostics.py compare --mode journal --reference DIR --candidate DIR`; it
names the first changed launch, upload digest, argument, allocation or
lifecycle event. Do not set a switch that makes a journal assertion return
early. These are CPU structural tests.

For an input or rendering problem, capture the real frontend without running
model arithmetic (set `CHOPINFER_QWEN38_TOKENIZER_DIR` or
`CHOPINFER_QWEN38_CHECKPOINT` first):

```bash
export QWEN38_OFFLINE_CAPTURE_DIR="$CHOPINFER_DATA/diagnostics/frontend"
cargo test -p model-qwen38 --test frontend_replay --locked --offline -- --nocapture
python3 -B tools/qual/frontend_goldens.py "$QWEN38_OFFLINE_CAPTURE_DIR" \
  --required-file chat-renders.json --required-file tokenizer-golden.json
```

An input-unavailable early return is not coverage; missing requested files
and empty comparisons are insufficient evidence.

For a numerical failure in Qwen, compare a bundle dump with the reference
runner's taps (`qwen38-oracle batch-drift --reference`, see the
[Qwen dump checks](qwen38-oracle.md)) or with `tools/qual/qwen38_layers.py`,
and check QSA selections and prepared projections operation by operation with
`qwen38-oracle qsa-capture` and `prepared-capture`. Qwen dumps force eager
execution and synchronize, so they cannot clear a graph-only failure. For
MiMo, the kernel-lab `mimo_layers::real_weight_layers` test compares real
layers and persistent KV history against the layer oracle. Use idle devices
and keep receipts outside the repository. Keep the models'
[numerical contracts](numerics.md) distinct: do not borrow a reference
checkpoint's expectations for fine-tuned weights or introduce a tolerance
after a bitwise failure.

## Prove a refactor only renames entries

Use the [journal proofs](../tools/journal_proof/README.md). Capture the
model's pinned fake-host journals at the base and the changed commit, write
a change spec that declares the renames (`specs/build.py` derives one from a
short change declaration), then prove every journal changed only that way:

```bash
python3 -B tools/journal_proof/capture.py --repo "$REPOSITORY" --model mimo26 \
  --revision "$BASE_COMMIT" --out "$CHOPINFER_DATA/journal-proof/base"
python3 -B tools/journal_proof/capture.py --repo "$REPOSITORY" --model mimo26 \
  --revision "$HEAD_COMMIT" --out "$CHOPINFER_DATA/journal-proof/head"
python3 -B tools/journal_proof/proof.py --repo "$REPOSITORY" --model mimo26 \
  --base "$BASE_COMMIT" --head "$HEAD_COMMIT" --spec "$CHANGE_SPEC" \
  --base-capture-index "$CHOPINFER_DATA/journal-proof/base/index.json" \
  --head-capture-index "$CHOPINFER_DATA/journal-proof/head/index.json" \
  --report "$CHOPINFER_DATA/journal-proof/proof.json"
```

Exit 0 (PROVEN) means every journal is identical or changed only as the
spec declares; 1 means not proven and 2 invalid input. This is structure
only; numerics need SASS identity or GPU captures.

## Prove a declared served-layout change

A change that adds allocations or collectives or regrids kernels (a TP
split, a new KV format) is declared in a
[layout declaration](../tools/journal_proof/README.md#layout-declarations)
committed with the change under `models/<model>/journal-changes/`. Write the
spec's witness from the two captures, then prove it:

```bash
python3 -B tools/journal_proof/specs/layout_spec.py --repo "$REPOSITORY" --model glm5next \
  --base "$BASE_COMMIT" --head "$HEAD_COMMIT" \
  --base-index "$CHOPINFER_DATA/journal-proof/base/index.json" \
  --head-index "$CHOPINFER_DATA/journal-proof/head/index.json" \
  --declaration models/glm5next/journal-changes/CHANGE.json --out "$CHANGE_SPEC"
python3 -B tools/journal_proof/proof.py --repo "$REPOSITORY" --model glm5next \
  --base "$BASE_COMMIT" --head "$HEAD_COMMIT" --spec "$CHANGE_SPEC" \
  --base-capture-index "$CHOPINFER_DATA/journal-proof/base/index.json" \
  --head-capture-index "$CHOPINFER_DATA/journal-proof/head/index.json" \
  --report "$CHOPINFER_DATA/journal-proof/proof.json"
```

A proof offered to land the change must report
`layout_declaration.read_at_head` true; `--declaration-revision` is only for
replaying a change that has already landed. Anything the declaration does not
cover still fails. The report's
`coverage.unbounded_host_span_lengths` counts declared span lengths in
host-provided buffers, which no allocation bounds.

## Replay the recorded mixed cancellation lifecycle

A private Qwen lifecycle adapter captures full inputs for one CPU scenario
(TP4, BF16 KV, 32 sequences): it prefills two requests, commits a decode,
then cancels a mixed batch with a new request, failing rank 1's restore copy
in the fault arm.

```bash
LIFECYCLE_ROOT="$CHOPINFER_DATA/diagnostics/mixed-cancellation"
CHOPINFER_QWEN38_LIFECYCLE_DIR="$LIFECYCLE_ROOT" \
  cargo test -p bundle-qwen38 --lib --locked --offline \
  fake_host_tests::lifecycle::recorded_mixed_cancellation -- --exact --nocapture
python3 -B tools/dev/diagnostics.py compare --mode journal \
  --reference "$LIFECYCLE_ROOT/baseline" --candidate "$LIFECYCLE_ROOT/matching"
python3 -B tools/dev/diagnostics.py compare --mode journal \
  --reference "$LIFECYCLE_ROOT/baseline" --candidate "$LIFECYCLE_ROOT/failure"
```

Use a fresh root with an existing parent. Expected exits are 0 for matching
and 1 for the failure, whose first difference names the cancel action, copy
result, rank and slot mapping. The report's `reproduction` command replays
the frozen `input.json` with the pinned test binary (rebuilding it changes its
digest); set `CHOPINFER_QWEN38_LIFECYCLE_OUTPUT` to a fresh path for another
replay. This is CPU structural evidence; device execution is `not_run`.

## Reuse model and operation captures

Oracle captures use the same envelope with `--mode oracle`. To narrow one,
copy the `scope` object from its `scope.json` into an external file and keep
the probes (in order) and row fields you need.

```bash
python3 -B tools/dev/diagnostics.py reuse --mode oracle \
  --candidate "$REFERENCE_CAPTURE" --scope "$DIAGNOSTIC_SCOPE"
python3 -B tools/dev/diagnostics.py compare --mode oracle \
  --reference "$REFERENCE_CAPTURE" --candidate "$CANDIDATE_CAPTURE" \
  --scope "$DIAGNOSTIC_SCOPE"
```

`reuse` executes nothing; it lists reusable probes, missing dependencies and
the producer command. The comparison separates `chained` and `op_isolated`
results: a chained failure with a passing matched-input operation means the
operation agrees on the operands it received, so look upstream. A `match`
needs complete, nonzero gated coverage, and a focused result cannot satisfy a
model's release gate.
