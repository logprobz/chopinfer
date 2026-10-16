# Contributing

Contributor rules (data hygiene, accuracy first, boundaries, commits) are in
[AGENTS.md](../AGENTS.md). Read [BUNDLE_ARCHITECTURE.md](../BUNDLE_ARCHITECTURE.md),
[kernel-library.md](kernel-library.md) and [numerics](numerics.md) before
changing kernel or bundle boundaries. For a first edit without weights, CUDA
or a GPU, use the
[contributor walkthrough](development-loop.md#first-success-without-weights-or-cuda).

## Build and verify

Use Linux x86_64, Rust 1.89.0 (see `rust-toolchain.toml`), Python 3.11 or
newer, gcc, and CUDA 13.2 at `CHOPINFER_CUDA_HOME` (default `/usr/local/cuda`). Kernel locks pin `nvcc`
13.2.86; the CPU gate compiles device code, so it needs the toolkit but no
GPU. Configure paths with `local.env.example` as described in
[toolchain.md](toolchain.md). On a new machine, populate the locked Cargo
cache once with network access:

```sh
CARGO_NET_OFFLINE=false cargo fetch --locked
```

The gates run Rust tests with cargo-nextest 0.9.146. Install the prebuilt
Linux x86_64 binary from the
[upstream release](https://github.com/nextest-rs/nextest/releases/tag/cargo-nextest-0.9.146)
once, after checking its `.sha256`; Cargo finds it in `$CARGO_HOME/bin`:

```sh
release=https://github.com/nextest-rs/nextest/releases/download/cargo-nextest-0.9.146
archive=cargo-nextest-0.9.146-x86_64-unknown-linux-gnu
curl -fLO "$release/$archive.tar.gz" -fLO "$release/$archive.sha256"
sha256sum -c "$archive.sha256"
mkdir -p "$CARGO_HOME/bin" && tar -xzf "$archive.tar.gz" -C "$CARGO_HOME/bin" cargo-nextest
```

`ci/cpu.sh` requires it: without it the gate stops before its first step.
`ci/quick.sh` falls back to `cargo test` without it, which runs the same tests
one test binary at a time and more slowly; set
`CHOPINFER_RUST_TEST_RUNNER=cargo` to run the gate that way too.

Then, from the workspace root:

```sh
export CARGO_NET_OFFLINE=true
bash ci/quick.sh          # while iterating: changed packages and dependents
bash ci/cpu.sh            # full CPU checks
```

`ci/quick.sh` runs format checking, Clippy and tests (through
`ci/rust_tests.py`, as `ci/cpu.sh` does) for the packages a change touches
plus their workspace dependents, and `ci/cpu.sh`'s Python and shell checks
whose inputs the change touches ([testing.md](testing.md#affected-checks)).
`--base REF` compares against a branch point, `--list` prints the selection
without compiling, `--all` selects the whole workspace and every such check,
`--report` explains which checks a change needs without
running anything (see [testing.md](testing.md)), and `--contributor
sdk|qwen38|mimo26|glm5next` runs the bounded fixtures without CUDA (add `--doctor` to
check prerequisites only).

`ci/cpu.sh` runs `cargo run -p xtask -- check-portable` (unsafe-code
audit, `cargo fmt --check`, the portable runtime policy, Clippy with
`-D warnings` and the workspace tests through `ci/rust_tests.py`),
`tests/bundle-boundary.sh`, `tests/model-neutral-host.sh`, the
synthetic bundle fixtures, the binary builds, the Python and shell tests under
`tests/` and `tools/`, the external SDK smoke, `tests/leak_guard.py` and the
fake CUDA driver tests. The leak guard also reads an untracked deny-list of
machine-specific patterns (your hostname, local tool directories), one
regular expression per line, from `$CHOPINFER_LEAK_DENYLIST` (default
`~/.config/chopinfer/leak-denylist`) and `leak-denylist.local` in the
checkout. It also rejects internal review and work-batch labels, review
notes that cite a commit, wording that attributes a decision to the
operator, plan steps and `/root`, `/mnt` or `/srv` paths: state a result
with its date and configuration instead. Its `PENDING` list names the files not yet
rewritten. `ci/cpu.sh` runs these checks as parallel steps in dependency order
([how the gate runs](testing.md#how-the-cpu-gate-runs)). On a 48-core
machine shared with other work (2026-10-05) it took about 6 minutes with a
warm target and 11 with a cold one, against 48 minutes for the previous
serial script, for about 30% more CPU time. Its host memory use peaked near
35 GB, about the same as the serial script's on that machine, and its
temporary and target directories near 40 GB.
Both scripts source `local.env` (see [toolchain.md](toolchain.md)).

After CPU acceptance, kernel changes also need the affected `kernel-lab`
test on an idle GPU ([testing.md](testing.md)). For example, a sampling
change uses:

```sh
CHOPINFER_KERNEL_GPU=0 cargo test -p kernel-lab --test ops --locked --offline sampling::
```

Choose the test for the changed operation, inspect its executed cases, and
keep its numerical receipts outside the tree. Never infer GPU correctness or
performance from CPU checks or a skipped GPU run.

### Full CPU check prerequisites

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
  without them ([geometry recipe](testing.md#geometry-recipe)).
- MiMo's pinned DFlash `mask_embedding.pt`, in `CHOPINFER_MIMO_DFLASH_DIR` or
  in the `dflash/` subdirectory of the MiMo tokenizer or checkpoint directory.
  The MiMo geometry recipe fails without it.
- A reference-runner Python with torch, transformers and numpy in
  `CHOPINFER_REFRUN_PYTHON`
  ([reference runner](reference-runner.md#prepare-and-run)). The
  `reference-prerequisites` step fails without it ([testing.md](testing.md)).
- The pinned references in `CHOPINFER_QWEN38_LONG_REFERENCE`,
  `CHOPINFER_QWEN38_LAYER_REFERENCE` and `CHOPINFER_GLM5NEXT_LAYER_REFERENCE`
  ([toolchain.md](toolchain.md)), and a short, existing, writable `TMPDIR` (at
  most 64 bytes). The gate checks them before its first step and refuses to
  start without them; `bash ci/cpu.sh --preflight` runs only that check
  ([testing.md](testing.md#how-the-cpu-gate-runs)). Neither the repository
  nor a release includes these references: they are GPU
  [reference runner](reference-runner.md) outputs on the pinned checkpoints,
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

## Dependencies

`Cargo.lock` pins registry sources; keep it current and use
`cargo tree --locked --offline` to inspect a dependency closure. Review a new
dependency against the owning crate's boundary: shared runtime crates take no
model or bundle dependencies, and a bundle may depend on model-neutral
contracts and kernel crates but never links `runtime-cuda`. The host owns the
CUDA driver boundary and native module loading.

## Pull requests

Use imperative commit subjects, matching recent history, such as
"Fix prefix restore accounting". Keep each change reviewable and explain the
behavior and verification in the pull request.

## Contribution tiers

There is no CI. GitHub-hosted runners have no SM120 GPU, so they cannot run
the GPU tests or serve a model, and `ci/cpu.sh` needs pinned tokenizer
files, MiMo's DFlash mask embedding, a reference-runner Python with torch
and about 35 GB of memory ([prerequisites](#full-cpu-check-prerequisites)). A
hosted run would skip or fail exactly the checks that matter, and a skipped
GPU test reports as passed. Instead each pull request carries its own
evidence, and the maintainers run the rest.

**Contributors** work on their own SM120 GPU (the tested part is the RTX
PRO 6000 Blackwell 96 GB; see [hardware](performance.md#hardware)) and
attach to the pull request each command they ran, the commit, the GPU and
driver, and the output or a summary of it (receipts stay outside the
tree):

- `bash ci/quick.sh` on the change, and `bash ci/cpu.sh` when its
  prerequisites are in place;
- the affected kernel-lab oracle tests with `CHOPINFER_KERNEL_REQUIRE=1`
  and `CHOPINFER_KERNEL_GPU` on an idle GPU, with the executed cases;
- for a speed change, the benchmarks and paired timings in
  [performance](performance.md), and for a change that alters bits, the
  oracle evidence of the
  [checklist](numerics.md#a-speedup-that-changes-bits).

A kernel source change changes its cubins, so the build reports a lock
mismatch until `KERNELS_LOCK_UPDATE=1` rewrites the lock. Lock records hold
the nvcc binary's SHA-256, and a lock matches only cubins built by the same
nvcc 13.2.86 binary (NVIDIA's `cuda-nvcc-13-2` 13.2.86-1 package, which
the Docker build installs). With another binary every kernel build reports
a mismatch; a contributor in that position builds and tests with a local
lock update, does not commit the rewritten locks, and leaves them to the
maintainers.

**Maintainers** produce the evidence that needs the pinned toolchain, the
four-GPU machines or the sealing key, and merge the change:

- regenerate kernel locks with the pinned nvcc binary;
- run [journal proofs](../tools/journal_proof/README.md) for every model
  whose bundle launches change; captures are sealed with an HMAC key held
  on the maintainers' machine, so captures made elsewhere do not verify;
- run the four-GPU [serving smokes](../tools/smoke/README.md) and
  [serving A/B](../tools/ab/serving/README.md), base against tip, for each
  affected model and profile;
- requalify each affected model against its pinned HF references
  ([correctness runner](../tools/qual/README.md#correctness-runner)).

## Shared-source guard exceptions

`tests/model-neutral-host.sh` scans shared crates, including kernel sources and
tests, and `tools/`, `ci/`, and `deploy/`. It checks paths and text in tracked or unignored source files, including CUDA,
C, Python and deployment configuration.
Model names also come from `models/*/model.toml`; compiler and contract checks
use their declared package names.

`tests/model-neutral-allowlist.json` records exact files, matched vocabulary
counts, and reasons for existing model integrations and pending kernel
extraction. No directory is exempt. Remove an entry when its last reference
moves, and reduce counts when references disappear. Deleted files and changed
counts fail the guard. Do not regenerate this list to admit new model behavior
in shared code. Counts prevent silent growth but do not replace code review:
replacing one reference with another identical spelling needs semantic review.
Unlabelled numeric shapes also require review; a vocabulary scan cannot infer
which model owns an arbitrary integer.
