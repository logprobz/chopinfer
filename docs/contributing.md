# Contributing

Follow [AGENTS.md](../AGENTS.md). For kernel or bundle changes, read
[the boundary](../BUNDLE_ARCHITECTURE.md), [kernel contribution rules](kernel-library.md)
and [numerical requirements](numerics.md).

## Setup

Use Linux x86_64, the Rust toolchain pinned in
[rust-toolchain.toml](../rust-toolchain.toml), Python 3.11+, Git and a C linker.
Full CPU checks also compile device code with CUDA 13.2.86, but need no GPU.
Set `CHOPINFER_CUDA_HOME` for a toolkit outside `/usr/local/cuda`.
Kernel locks require the pinned nvcc binary, not just the same version text.

Copy [local.env.example](../local.env.example) to `local.env` and edit it.
Shell tools source the checkout's file, otherwise `CHOPINFER_LOCAL_ENV`
(default `~/.config/chopinfer/local.env`). Its exports replace shell values;
the qualification Python loader reads `export CHOPINFER_*=value` lines and
keeps variables already exported. Keep caches, targets, logs and references
outside the tree, and keep `TMPDIR` existing, writable and at most 64 bytes.

```sh
CARGO_NET_OFFLINE=false cargo fetch --locked
export CARGO_TARGET_DIR="${CHOPINFER_DATA:?set an external data directory}/target"
export TMPDIR=/tmp
```

The full gate uses cargo-nextest 0.9.146. Install its verified prebuilt binary
from the [pinned release](https://github.com/nextest-rs/nextest/releases/tag/cargo-nextest-0.9.146)
in `$CARGO_HOME/bin`, checking the archive's SHA-256. `ci/quick.sh` falls back
to Cargo's runner; `CHOPINFER_RUST_TEST_RUNNER=cargo` selects it explicitly.

## First success without weights or CUDA

```sh
bash ci/quick.sh --contributor qwen38 --doctor
bash ci/quick.sh --contributor qwen38
```

Use `sdk`, `mimo26` or `glm5next` for those fixtures. Model contributor modes
are declared in `model.toml`; `--list` shows their commands. They exercise
config admission, tensor mapping, frontend and weightless packaging with
bounded inputs. They do not qualify real weights or device execution.
Each selected test must execute and pass without ignored cases. Start a new
model with the [external example](../examples/external-model/README.md).

## Checks

```sh
bash ci/quick.sh          # changed packages, dependents and affected checks
bash ci/quick.sh --report # remaining validation obligations, without running
bash ci/cpu.sh            # full CPU gate on a clean committed HEAD
```

Each quick run checks formatting and runs the unsafe audit
(`cargo run -p xtask -- check-unsafe`). The audit accepts unsafe Rust only in
a file listed in `REVIEWED_UNSAFE_FILES` (`crates/xtask/src/lib.rs`), with a
`// SAFETY-REVIEWED:` comment starting at most three lines above each unsafe
line.

`--base REF` selects a comparison point; `--all` selects the whole workspace.
`bash ci/cpu.sh --only STEP` runs the named steps and the setup they need.
It refuses a step that reads another step's outputs, such as the kernel
report reader, unless that step is selected too or `--with-producers` adds it.
`ci/check_inputs.py` derives each Python or shell check's inputs from its
code. A walk (rglob, os.walk or a Git listing) that reads only directories
outside the tree, or only some repository directories, says so on its line
with `# check-inputs: walks outside the repository` or
`# check-inputs: walks DIR`, so source edits elsewhere do not select it.
The gate runs format, Clippy, Rust tests, boundary and layout guards,
Python/shell checks, synthetic bundles, SDK recipes and leak checks. Its
recipe tests use detached checkouts of HEAD: commit first and keep HEAD
fixed until the gate finishes. `bash ci/cpu.sh --preflight` checks prerequisites.
Both scripts build Rust with `ci/rust-profile.env`: opt-level 1 for every
package, kernel-core included, with debug assertions and overflow checks on;
release builds are unchanged. kernel-core's one test that needs opt-level 0
runs in the gate's `kernel-core-unoptimized` step. The gate's steps share
`--jobs`, and no step counts for more than half of it, so the workspace tests
start beside the long recipe steps whatever `--test-threads` says.
For memory-limited runs, set `CHOPINFER_CI_SERIAL_LIBRARIES=1` to run the
libraries marked `lib-tests-in-one-process` one at a time before nextest.
`RUST_TEST_THREADS` still limits tests within each such library; nextest's
`--test-threads` limit does not control those separate processes.

### Landing and release gates

A change lands when these pass. The CPU checks run on its final tip:

- the tests of each touched crate and its dependents, and the Python and
  shell checks the change affects (`bash ci/quick.sh --base REF` selects
  them against the landing base);
- `cargo check --locked --offline --workspace --tests --keep-going`;
- the unsafe audit, `cargo run --locked --offline -p xtask -- check-unsafe`,
  which `bash ci/quick.sh` runs;
- `python3 -B tests/leak_guard.py`, `bash tests/bundle-boundary.sh` and
  `bash tests/model-neutral-host.sh`;
- the pre-push leak check over the change's commits, every file version
  and message: `bash ci/quick.sh --base REF` runs
  `python3 -B tests/leak_guard.py --rev HEAD --messages HEAD --base REF`
  before its other checks;
- when the change touches a model's bundle, kernels or routes, that
  model's quick and CPU [route sweep tiers](kernel-library.md#route-classes-and-served-envelopes)
  and journal pins;
- when the change can reach device execution or served output, GPU
  exactness for each affected model: bitwise equality with the previous
  output, or the [within-bound gates](numerics.md#a-speedup-that-changes-bits).

GPU exactness and timing may come from the frozen base the change was
measured on when main moved meanwhile; do not rebuild for the rebase.
Rebase the range for landing, show with `git range-diff` that its patches
are unchanged, and run the CPU checks above on the rebased tip. If the
rebase resolves a conflict in serving code, say so; the integrator decides
whether the GPU evidence must be rerun.

Commit messages, merge messages included, must not name internal session
branches, worktrees or review labels. Write each merge message yourself
(`git merge -m`); git's default subject names the branches.

The full CPU gate (`bash ci/cpu.sh`) is optional per batch; the integrator
decides whether a batch needs it. It is required before a [release](#release).

### CPU phases and inventories

The CPU gate runs `portable-build`, `portable-lint`, `portable-test`, then
`high-memory-tests`. The first three call xtask's `check-portable-build`,
`check-portable-lint` and `check-portable-test` commands. `--only portable`
selects all four; `--only portable-test --with-producers` includes its
build/lint prerequisites. The quick gate keeps its own selection policy.

The exact `high-memory-*` nextest overrides declare the exceptional tests
once. The CPU runner excludes them from ordinary tests, then runs each
serially in an exclusive gate slot after the ordinary phase. That slot
does not lift an enclosing supervisor or cgroup cap. The historical
16 GiB and 30-minute limits came from an outer supervisor, not from the
heavy wrapper or this phase split.

The 2026-10-25 phased-gate comparison preserved the baseline Rust test,
doctest and route inventories. Measured wall/CPU seconds and peak RSS were:

| Phase | Wall seconds | CPU seconds | Peak RSS GiB |
| --- | ---: | ---: | ---: |
| portable-build | 997.418 | 4594.563 | 2.184 |
| portable-lint | 167.929 | 716.873 | 1.738 |
| portable-test | 1414.386 | 10325.714 | 36.635 |
| high-memory-tests | 146.883 | 147.178 | 32.567 |

RSS is the process/descendant high-water mark from `wait4`, not aggregate
cgroup memory. Independent steps overlap, so phase times do not sum to gate
wall time. The ordinary phase also exceeded 16 GiB; separating the two
exceptional tests does not make the rest fit that cap.

Each gate retains `results.json`, `executed-tests.json`, JUnit reports and
per-case `high-memory-results.json` under its external work directory.
To reconstruct or compare inventories:

```sh
python3 -B ci/test_inventory.py --work "$CI_WORK" --summary "$CI_SUMMARY" \
  --junit "$CI_JUNIT" --output "$CI_INVENTORY"
python3 -B ci/test_inventory.py --compare "$CI_BEFORE" "$CI_AFTER" \
  --output "$CI_COMPARISON"
```

Repeat `--junit` for historical reports. Only explicitly reviewed added
Python cases may use `--allow-added-python-step STEP`; removals still fail.
Older logs without identities compare exact counts, skips and verdicts,
which the inventory marks separately from identity comparisons.

Every CPU route sweep covers every boundary, packed and chunk case; dense
single-prompt probes rotate by seed. `CHOPINFER_ROUTE_SEED` selects an explicit
seed for the gate, which otherwise uses the UTC day. The CI runner prints
each model's seed and reproduction command. There are no per-model counters
or seed maps. See [route classes and served envelopes](kernel-library.md#route-classes-and-served-envelopes)
for the rotation periods and full-tier coverage. CPU checks are not GPU evidence.

## Full CPU check prerequisites

`ci/cpu.sh` passes only with all of these in place, even for a change that
touches no model:

- The toolchain above: Rust 1.89.0, Python 3.11 or newer, gcc, Git and
  `nvcc` 13.2 under `CHOPINFER_CUDA_HOME`.
- A clean committed HEAD. The `geometry-recipe`, `geometry-recipe-contract`
  and `export-recipe` steps build a detached worktree of HEAD and fail on any
  uncommitted change or untracked file, so commit before running the gate.
- The pinned Qwen3.8 and MiMo tokenizer files (`tokenizer.json` and
  `tokenizer_config.json`) in `CHOPINFER_QWEN38_TOKENIZER_DIR` and
  `CHOPINFER_MIMO_TOKENIZER_DIR`. When only `CHOPINFER_QWEN38_CHECKPOINT` or
  `CHOPINFER_MIMO_CHECKPOINT` is set, the script uses that directory as the
  tokenizer directory. The geometry and export recipes always run and fail
  without them ([geometry recipe](contributing.md#checks)).
- MiMo's pinned DFlash `mask_embedding.pt`, in `CHOPINFER_MIMO_DFLASH_DIR` or
  in the `dflash/` subdirectory of the MiMo tokenizer or checkpoint directory.
  The MiMo geometry recipe fails without it.
- A reference-runner Python with torch, transformers and numpy in
  `CHOPINFER_REFRUN_PYTHON`
  ([reference runner](../tools/refrun/README.md#prepare-and-run)). The
  `reference-prerequisites` step fails without it ([Contributing](contributing.md#checks)).
- The pinned references in `CHOPINFER_QWEN38_LONG_REFERENCE`,
  `CHOPINFER_QWEN38_LAYER_REFERENCE` and `CHOPINFER_GLM5NEXT_LAYER_REFERENCE`
  ([local.env.example](../local.env.example)), and a short, existing, writable `TMPDIR` (at
  most 64 bytes). The gate checks them before its first step and refuses to
  start without them; `bash ci/cpu.sh --preflight` runs only that check
  ([Contributing](contributing.md#checks)). Neither the repository
  nor a release includes these references: they are GPU
  [reference runner](../tools/refrun/README.md) outputs on the pinned checkpoints,
  and `models/<model>/qual-references.lock.json` pins their bytes. Ask the
  maintainers for copies. Until you have them, run `bash ci/quick.sh`,
  which needs them only when a change selects the release gate,
  `runtime-eval` or the whole workspace, and say in the pull request that
  `ci/cpu.sh` did not run.
- The reference workloads each model's `qual.json` keeps under
  `CHOPINFER_DATA`: Qwen's and GLM's image-probe workloads
  (`qwen38-refs/refrun/image-probe-references.workload.json` and
  `glm5next/refrun/image-probe-references.workload.json`) and MiMo's mixed
  media workload (`refrun/g9-min8192/answers/combined-workload.json`). The
  `runtime-eval` reference tests verify them whenever `CHOPINFER_DATA` is
  set and fail without them; get them with the references.
- A HEAD that stays put: the gate fails when the checkout's HEAD moves while
  it runs, so commit, rebase or switch branches in another worktree.

`bash ci/quick.sh --doctor` checks the toolchain, the reference-runner
Python and the reference variables. Other checkpoint-dependent tests skip while their inputs are unset; a
skipped test does not establish model correctness. Tests that read checkpoint
weights run only with `CHOPINFER_CI_CHECKPOINT_TESTS=1`.


## Contribution tiers

Contributors report the commands, executed cases and results, including
missing prerequisites. CPU acceptance and a skipped GPU test establish no
device correctness. Select only idle GPUs with `CHOPINFER_KERNEL_GPU` or
`CHOPINFER_KERNEL_GPUS` and set `CHOPINFER_KERNEL_REQUIRE=1` for required
kernel checks. Choose the affected tests from [kernel-lab](../crates/kernel-lab/README.md).
Changes that alter bits follow the [numerical evidence rules](numerics.md#a-speedup-that-changes-bits)
before measuring speed. Served timings require an idle host as well as idle
GPUs; [pair-lab kernel timing](#pair-lab-kernel-timing) needs only idle GPUs.

Maintainers produce pinned-toolchain locks, sealed [journal proofs](../tools/journal_proof/README.md),
[serving smokes](../tools/smoke/README.md), [serving A/Bs](../tools/ab/serving/README.md)
and the affected models' [qualification runs](../tools/qual/README.md#correctness-runner).
Never regenerate locks or goldens to hide a mismatch. Reference capture setup
is in [tools/refrun](../tools/refrun/README.md).

## Shared build hosts

`tools/dev/heavy.sh` coordinates expensive jobs through `CHOPINFER_HEAVY_DIR`.
Use its `--help` and `--status`; keep queue state and per-worktree Cargo
targets outside the repository. It does not reserve GPUs. Development
redeploy helpers and Nsight wrappers live under `tools/dev` and `tools/profile`;
their help describes inputs. Preserve matched workloads and configurations
when comparing timings, and keep raw evidence outside the tree.

## Release

Run the full CPU gate and affected [correctness suites](../tools/qual/README.md#correctness-runner)
before a release. Set the workspace version and refresh Cargo.lock, preserving
ABI/schema versions. Build the archive with `tools/release/build.sh` into a
fresh external directory; its builder verifies pinned kernel locks and emits
checksums and licenses. Verify the archive before publishing it. The archive
contains binaries, libraries and declarations, not weights. Installation
instructions are [shipped with it](../tools/release/INSTALL.md).

The builder uses a fresh private Cargo target below `CARGO_TARGET_DIR`,
which is a scratch root, and prints a `Prebuilt:` directory. It disables
Rust compiler wrappers, forces normal release settings and all lock/profile
bypasses off, and verifies every selected kernel lock record, including child
selections and both host samplers. Shared nvcc cache hits still compare the
pinned locks. If the requested scratch root is inside the source context
(as in the Docker command), the private target goes under `TMPDIR` instead.

The prebuilt directory contains `release-build.json`, a portable producer
record binding each artifact's bytes to the revision, actual source inputs,
release configuration, kernel identities and compiler freshness. Preserve
that directory to stage the same verified build again:

```sh
bash tools/release/build.sh --stage-from "$PREBUILT" "$CHOPINFER_DATA/restaged-release"
```

Unrecorded directories must be rebuilt through the producer. Staging checks
the record and every artifact before executing staged code, copies through
bounded no-follow reads, and includes `BUILD_PROVENANCE.json` under the
archive's checksums. `CHOPINFER_BUILD_REVISION`, when set, must equal HEAD.
Native Git worktrees and Docker contexts retaining HEAD and loose or packed
refs are supported. Retained refs establish the declared revision and supplied
source bytes, not proof that those bytes equal a clean Git tree. The record
trusts the controlled producer; it is not a signature authenticating another
builder. These provenance checks do not replace the full CPU gate or affected
model qualification before publishing.

## Pull requests

See [accepted contributions](../.github/CONTRIBUTING.md#what-is-accepted).
Use imperative commit subjects and explain changed behavior and verification.
Install the repository's pre-push check in this checkout:

```sh
hook=$(git rev-parse --git-path hooks/pre-push)
printf '#!/bin/sh\nexec bash "$(git rev-parse --show-toplevel)/tools/dev/pre-push.sh" "$@"\n' > "$hook"
chmod +x "$hook"
```

It checks outgoing history, messages and tags for data leaks and refuses
protected-ref deletion or unrelated history. [Leak guard](../tests/leak_guard.py)
also checks Markdown links and optional local deny-list patterns.

## Shared-source guard exceptions

`tests/model-neutral-host.sh` checks shared code and tools against exact
vocabulary counts in `tests/model-neutral-allowlist.json`. Remove entries or
reduce counts when references disappear. Do not grow the allowlist to admit
new model behavior in shared code. `cargo tree --locked --offline` and the
bundle boundary checks expose dependency violations. Review new dependencies
against their owning layer.

## Diagnostics

Reproduce a divergence under a fixed request/batch schedule first. Use
`tools/dev/fixed-batch-logits.sh` and [serving A/B](../tools/ab/serving/README.md)
for exact comparisons, [journal proofs](../tools/journal_proof/README.md)
for structural claims, and [reference taps](../tools/refrun/README.md#taps-and-tapdiff)
for matched-input arithmetic. Find the first differing stage before changing
precision or tolerance. Eager synchronized dumps cannot clear a graph-only
failure. Keep captures outside the repository and bound to the producer.

A host built with `linux-cuda-host` also exposes
`chopinfer fixed-batch replay --config FILE`. It runs the same recorder and
comparison checks as the fixed-batch device test, using the native executable's
identity. This lets a replay use that executable's weight-keeper content proof.
The JSON configuration accepts `weight_keeper`, `weight_keeper_policy` and
`weight_keeper_identity`, with the same meanings and validation as the serving
options. Stage the proof with the actual package, checkpoint and instance
configuration; do not reuse a proof from another executable. The optional
`temperature` field controls sampled rows and is recorded as part of comparison
identity. Omission preserves 0.8; greedy rows remain at zero. Keep GPU selection,
trace limits, baseline repeat controls and capture locations unchanged.

A plain replay is one arm of the gather/owned logits A/B: `owned: false`
expects every decode row on rank 0 and `owned: true` the owner blocks, after a
gathered repeat control. A package without a rank-0 gather switch sets
`default_owners: true` (with `owned: false` and no `control`) instead: each
step's rows must then sit all on rank 0 or all in the owner blocks, one layout
for the whole trace, and the receipt records which. The mode is part of the
comparison identity, so such a capture only serves as the reference of another
default-owner replay, which must take the same layout, and never as a gathered
baseline or the owned arm's control. An ordinary replay refuses any reference
or control whose receipt records default owners, including receipts written
before the identity bound the mode. Ownership is never numerical evidence;
the rows themselves still match the reference bit for bit.

For a bounded drafting transition, supply a version-6 trace and set
`transitions: true`, `trace_input: true`, `verify: true`, explicit `kv_pages`
and diagnostic capture limits in the replay configuration. Such traces permit
2 to 1,024 steps, up to 32 live requests and 256 rows per step, within the
package's limits. Finish all prompt prefills before decoding. An absent
request keeps its committed state. Each item's serialized `history` is empty:
the recorder derives it from committed inputs and accepted bundle proposals.
`kept` includes the input row and any accepted draft rows; `emitted` counts
the current history's generated tokens before the proposed rows. Every
proposal, resulting history, sampling record and logits byte remains part of
the exact comparison. Transition traces forward the native seed, generated
index and temperature envelope when the package declares sampled drafting.
Their comparison invariants bind that wire behavior. Versions 1 through 5
retain their existing rules and wire records.

An allocated replay accepts `model_options` as a map of option names to
string values and `allocation` as `{ "width": N, "confidence": CALIBRATION }`,
using the serving calibration schema. An optional `capacity` in `allocation`
sets the instance's maximum verify rows; omit it to use the trace's maximum.
Use the serving capacity and `max_sequences` when comparing padded layouts.
It requires a version-6 trace with
prefills followed by one final drafted step, using greedy, seeded or top-logprob
sampling. The width must cover the calibration's floor and be smaller than
the reserved rows. The recorder submits serving's allocation and device-result
metadata, records every reserved row and proposal, and commits the actual
device acceptance instead of the trace's prescribed `kept` counts. With
`draft_confidence: true` it also reads each drafted request's raw F32 drafter
confidence, one word per computed token, beside its proposals and compares
every word bit for bit; it refuses proposals without confidence and a replay
that records none. The mode is part of the comparison identity, so its
reference sets it too. Configurations
without `allocation` keep the existing replay path. Model options must also
be included in the executable's keeper proof.

Version 7 adds bounded request lifetimes to transition traces. Its optional
`lifecycle` list contains ordered, one-based `before_step` boundaries, each
with `actions` and an optional `cancel: true`. Actions are `capture {slot}`,
`retire {slot}`, and `restore {slot, from}`, encoded with an `action` tag.
The slot identifies a logical request; physical state slots come from the
normal host allocator. A canceled step still compares all logits, proposals
and samples, then cancels the bundle, rolls back provisional KV and retires
the touched requests. A later restore or prefill starts a new lifetime.
Capture retains real KV anchors and bundle snapshot pages; restore validates
their ownership and uses the bundle's state-slot restore API. All requests
must be prefilled, and the trace must exercise capture, retirement, restore,
cancellation and drafting. At most 16 boundaries and eight saved prefixes
are allowed. Version 6 retains its original validation and wire records.
Snapshot page requirements are checked after instance creation; a CPU dry
run alone does not establish that the snapshot allocation fits.

## Measuring a change

First establish [accuracy](numerics.md#a-speedup-that-changes-bits). Use the
affected kernel-lab benchmark and serving A/B with identical checkpoints,
profiles, workloads and output lengths on idle GPUs and, except for
[pair-lab kernel timing](#pair-lab-kernel-timing), an idle host. Report
paired timings and variability, not a best run. Shared-device results are
diagnostic only. [Tracing and profiling](operations-observability.md#trace-a-slow-request)
helps locate the cost; profiling bypasses do not qualify release artifacts.

### Step-time acceptance

Use the in-tree `tools/tune` `mode: ab` for the following paired method.
It judges kernel and route changes. Scheduler and prefix-cache changes use
[stateful replay acceptance](#stateful-replay-acceptance); a change that
does both needs both verdicts. Arm A is the baseline and arm B is the
candidate. Only served full-graph timing of the affected bucket decides
adoption of a kernel or route change. [Pair-lab kernel timings](#pair-lab-kernel-timing)
with an in-job A/A help locate work to improve.

- Use sampled coding and chat prompts at temperature 1.0.
- Visit A1, B1, B2, A2 in that order, with 15-second measured windows.
  A1/B1 share one seed set and B2/A2 share another. Each DSpark visit runs
  in a fresh process borrowing the weight keeper, so learned acceptance
  state cannot carry between visits. Assert zero cached prompt tokens in
  measured windows.
- Record major faults per measured window and SM clocks per window. A
  visit whose measured windows sum to more than 1,000 major faults is
  invalid and must be rerun before a verdict. Startup, priming and drain
  are not counted.
- Report each cell and weight cells by served shapes: decode C1, C2, C4,
  C8, C16 and C32, and DSpark verify widths 6, 8, 16, 32 and 64.

For a prefill route, declare `prefill_rows` by concurrency in the tuner
model, for example `{"1": [192, 256]}`, and name the affected target as
`prefill:c1:r192` in `target_cells`. These cells use complete step execution
times, with exactly the declared prefill item and token counts and no
decode or verify work in the step. Every declared width, including a
control, must have sufficient samples in all four visits. They use the
same minimum sample/time-share policy, paired spread, regression veto and
1.5% table-row threshold as other cells. Synthetic dry runs never qualify
prefill timing. Mixed steps need a separate declared measurement protocol.

Report full-graph step times and paired served rates. In the formulas below,
A1, B1, B2 and A2 are higher-is-better rates; use inverse step time when
judging a step-time reduction. For each cell, compute:

```text
d1 = B1 / A1 - 1
d2 = B2 / A2 - 1
mean = (d1 + d2) / 2
spread = abs(d1 - d2)
```

A speedup is resolved only when d1 and d2 have the same sign and the mean
exceeds the spread. A new table row or variant also needs a mean of at least
1.5% in its target cells. Reject a simplification (a removed row, merged
variant or deleted path) if any weighted cell has both d1 and d2 negative
and a mean below `-max(spread, 0.5%)`. Keep cell verdicts separate so a gain
elsewhere cannot hide that regression. An A1/A2 or B1/B2 comparison counts
as A/A only when both visits used identical seeds and prompts.

### Stateful replay acceptance

Step-time acceptance asserts zero cached prompt tokens, so it cannot see a
change that acts through serving state across steps and turns: prefix-cache
eviction order, cache points captured on cancel or stop, adaptive
prefill chunk cuts, or CUDA graphs for short prefill continuations. Judge
such scheduler and prefix-cache changes on stateful multi-turn traffic.

Establish exactness before any timing:

- Greedy and fixed-seed sampled output of single requests and fixed-batch
  replays must equal the previous output bit for bit
  ([batch shape](numerics.md#batch-shape)).
- A change that moves prefix anchors, cache points or chunk cuts must
  pass `tools/qual/prefix_exact.py` and the correctness runner's
  determinism screen ([prefix-cache accuracy](numerics.md#prefix-cache-accuracy)).
- A change to preemption capture or restore must pass
  `preempt_compare.py compare --exact` ([preemption resume](numerics.md#preemption-resume)).

Concurrent multi-turn arms may still diverge once batch composition
differs. Report how many matched turns agree (`agentic_compare.py`
reports it); that divergence is not an accuracy failure.

Method:

- Replay sessions with `tools/qual/agentic_mixed.py` at temperature 1.0
  with per-session cache salts (`--cache-salt session`), so each turn can
  hit only its own session's prefix.
  Its `--workload trace --trace FILE` mode accepts content-free length,
  dependency, seed and pacing declarations generated by
  `tools/qual/agentic_trace.py`. Use canonical synthetic histories for
  matched inputs, or `--trace-history generated` to measure response
  re-rendering and thinking-prefix loss. Shared-prefix diagnostics use
  `--cache-salt group`, prime each group once, and report that stratum
  separately from session-isolated acceptance. Idle returns may wait for
  logical token-page demand from intervening completed traffic; the receipt
  states the page size and formula. This is a pressure proxy, not physical
  allocation telemetry. Keep server eviction and reuse counters beside it.
- Run one untimed warm-up process on the same install and workload, then
  visit A1, B1, B2, A2 in that order. Each visit is a fresh server
  process (it may borrow the [weight keeper](weight-keeper.md)), so no
  cache or learned policy state carries between visits. A1/B1 share one
  seed and B2/A2 share another.
- Keep the profile, KV pool size, concurrency levels, window length and
  session limits identical in every visit. Record preemptions and
  prefix-cache evictions per visit instead of asserting zero; eviction
  and cache-point changes act through them.
- Record the [power and clock evidence](#power-and-energy) and major
  faults of step-time acceptance. The 1,000 major-fault budget applies to
  each visit's measured windows.
- Run on an idle host and idle GPUs.

Before running, declare the metrics and concurrencies the change should
improve. Report each concurrency (a cell) for every visit:

- the cache-hit rate (cached over prompt tokens) and cached prompt tokens;
- TTFT p50 and p95;
- decode event-gap p50 and p95; stalls (gaps above the stall threshold,
  100 ms by default) per 1,000 gaps and stall time per 1,000 gaps, with
  the stall count, total stall time and eligible gaps;
- output tokens per second and valid sessions per minute.

Arms complete different numbers of turns in a fixed window, so judge
rates, not visit totals. Report raw cached prompt tokens, the stall
count and total stall time as evidence; do not judge them.

For each metric and cell compute d1, d2, mean and spread with the
step-time formulas, inverting lower-is-better metrics (TTFT, event gaps)
so a positive d is an improvement. Three rates can be zero in the
baseline, so use differences for them, positive when better:

- cache-hit rate, in percentage points (d1 = B1 - A1, d2 = B2 - A2);
- stalls per 1,000 gaps and stall time per 1,000 gaps
  (d1 = A1 - B1, d2 = A2 - B2).

A metric is unavailable in a cell when either visit of a pair lacks its
samples: a percentile below the harness's supported sample count (20
for p95), or a rate with no prompt tokens or no eligible gaps. An
unavailable metric gives no verdict in that cell; report it as
unavailable, never as a tie.

- A gain is resolved when d1 and d2 have the same sign and the mean
  exceeds the spread, in a declared metric and cell.
- A regression is resolved when d1 and d2 are both negative and the mean
  is below `-max(spread, 0.5%)` for ratio metrics, or below `-spread`
  for difference metrics, in any metric and cell.
- Keep metric and cell verdicts separate so a gain elsewhere cannot hide
  a regression.

A change with a resolved gain and no resolved regression replaces the
old behavior. A change that resolves gains and regressions in different
metrics or cells is a tradeoff: it may be adopted only as a
[serving option value](kernel-library.md#serving-options-and-candidate-switches).

### Pair-lab kernel timing

Kernel-lab timing measured with CUDA events on the job's own two idle GPUs
does not need an idle host. The job runs for at most five minutes and
interleaves an in-job A/A control with identical operands in both arms,
and reports the A/A spread beside the result. A lab gain counts only when
it exceeds that spread. Report an unusually wide A/A spread and rerun the
job on an idle host. Pair-lab timings locate and rank work; they never
decide adoption. Served timing, host overhead included, still requires an
idle host and idle GPUs.

### Power and energy

Power limits and clocks are operator-controlled. Never set, lock or assume
them. Every timed job records each GPU's enforced power limit and SM clocks
at the start and end of the job and each visit. Limits may differ between
GPUs and between jobs, but each GPU's limit must stay constant within one
job. A job whose limits change mid-run is void: rerun it and never pool its
measurements with valid jobs.

Nothing is keyed by power or clock, including route choices, tuning tables,
profiles and depth schedules. A candidate whose gain plausibly comes from
power-cap behavior, such as lower draw without less work, needs a second
verdict at a different limit before adoption. The operator controls that
limit too.

Check energy (J/token) separately from the speed verdict, and compare it
only between arms under identical per-GPU power limits. Until a pooled noise
model exists, a J/token increase vetoes adoption only when it exceeds 3% and
agrees in sign across the concurrencies where the candidate is active;
otherwise report it as informational. Energy does not replace the timing or
bitwise gates.

### Multi-GPU host configuration

The reference four-GPU RTX PRO 6000 host runs the IOMMU in passthrough and
disables ACS redirection on each GPU's root port, the configuration
commonly used for peer-to-peer GPU hosts. On GRUB, add both to
`GRUB_CMDLINE_LINUX_DEFAULT`, escaping the semicolons between root ports,
and run `update-grub`:

```sh
GRUB_CMDLINE_LINUX_DEFAULT="iommu=pt pci=disable_acs_redir=<port>\;<port>\;<port>\;<port>"
```

Each `<port>` is the PCI address of a GPU's parent root port, for example
`readlink -f /sys/bus/pci/devices/<gpu>/..`. After rebooting, check that
`/proc/cmdline` carries both parameters, that the kernel log reports
`Default domain type: Passthrough` and `Configured ACS to 0x0000` for each
port, and that every GPU's IOMMU group reports `identity` in
`/sys/kernel/iommu_groups/<group>/type`. The `disable-acs` script's
`--verify` reads the same state; its `--apply` refuses on Ubuntu, whose
recovery entries carry only `GRUB_CMDLINE_LINUX`.

On the reference host (GPUs behind separate root ports of one CPU), these
settings did not change measured peer traffic on 2026-10-31: one destination
still runs at 42 to 54 GB/s and a GPU driving two or more peers at once
still collapses, as [SM120 PCIe communication](../crates/kernels-sm120/cuda/comm/README.md#measured-pcie-behaviour)
records. Record the host's IOMMU mode and kernel command line with timing
evidence, and compare timings only between runs on the same configuration.
