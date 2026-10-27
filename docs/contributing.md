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
- `python3 -B tests/leak_guard.py`, `bash tests/bundle-boundary.sh` and
  `bash tests/model-neutral-host.sh`;
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
Arm A is the baseline and arm B is the candidate. Only served full-graph
timing of the affected bucket decides adoption. [Pair-lab kernel timings](#pair-lab-kernel-timing)
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
