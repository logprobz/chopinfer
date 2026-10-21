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
before measuring speed. Timings require an idle host as well as idle GPUs.

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

## Measuring a change

First establish [accuracy](numerics.md#a-speedup-that-changes-bits). Use the
affected kernel-lab benchmark and serving A/B with identical checkpoints,
profiles, workloads and output lengths on an idle host and GPUs. Report
paired timings and variability, not a best run. Shared-device results are
diagnostic only. [Tracing and profiling](operations-observability.md#trace-a-slow-request)
helps locate the cost; profiling bypasses do not qualify release artifacts.
