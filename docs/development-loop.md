# Development loop

Setup and the full CPU checks are in [contributing.md](contributing.md). For a
reproducible mismatch, start with the [first-divergence workflow](debugging.md).

## First success without weights or CUDA

If you are adding a model, start with the SDK's
[independent example](../examples/external-model/README.md): it packages a
seed as bundle metadata, then shows a real input edit and a refused input.
For Qwen, MiMo or GLM compiler work, the contributor fixtures below exercise the
production compiler's config, mapping, frontend and package code with small
inputs. They load no production weights and need no GPU or `nvcc`.

You need Linux x86_64, Rust 1.89.0 with rustfmt and Clippy, Python 3.11 or
newer, Git, a C linker and a provisioned Cargo cache
([toolchain.md](toolchain.md)). From the workspace root, with external output
paths:

```sh
export CARGO_NET_OFFLINE=true
export CARGO_TARGET_DIR="${CHOPINFER_DATA:?set an external data directory}/contributor/target"
export CHOPINFER_SCRATCH="$CHOPINFER_DATA/contributor/tmp"
export TMPDIR="$CHOPINFER_SCRATCH"
mkdir -p "$CHOPINFER_SCRATCH"
bash ci/quick.sh --contributor qwen38 --doctor   # or sdk, mimo26, glm5next
bash ci/quick.sh --contributor qwen38
```

Contributor modes come from each model's `model.toml` `contributor` section;
`sdk` remains the shared exercise. Declare bounded fixtures, package roots and
typed test selections there when adding a mode, then use `--list` to inspect
its commands. Recipes do not add shell snippets or per-model CI branches.

`--doctor` checks that the selected dependency graph needs no CUDA build and
that the cache and fixtures are present; a missing piece is a setup error with
a remedy. Make sure your `local.env` does not override the paths above.
`ci/quick.sh` refuses a `TMPDIR` longer than 64 bytes, so with a long
`CHOPINFER_DATA` path, set `CHOPINFER_SCRATCH` to a shorter directory. The
fixture run prints each executed check and its passing test count, then
`Contributor fixtures passed`, followed by the outstanding obligations from
the validation report (which it does not mark satisfied). Contributor mode
clears checkpoint/tokenizer variables, checkpoint-index keys, all
`CHOPINFER_PACKAGE_*` keys and model-declared extra keys so its inputs stay
bounded. Executed test selections must pass at least one test with no ignored
cases. A cold contributor run spends under a minute compiling on an
eight-thread machine; repeats take seconds.

**Qwen config.** Change `seed` in
`models/qwen38/compiler/fixtures/contributor-config.json` and repeat the Qwen
command; the compatibility test accepts this metadata, while another test
refuses execution geometry and quantization changes.

**Qwen mapping.** Change the norm's shape from `[128]` to `[127]` in
`models/qwen38/compiler/fixtures/contributor-tensors.json`, then run:

```sh
cargo test -p model-qwen38 --lib --locked --offline \
  catalog::tests::contributor_tensor_names_shapes_scales_and_pairs -- --exact
```

The test fails on the changed tensor. Restore `[128]` and run the contributor
command again. The same per-tensor rule as checkpoint admission rejects
unknown names, wrong scale formats and missing scale pairs.

**Qwen frontend.** Edit `models/qwen38/compiler/src/frontend.rs` or the
request cases in `models/qwen38/compiler/tests/reasoning_controls.rs`, then
repeat the Qwen command. It renders the existing chat goldens and tests
reasoning and tool output parsing. Token-ID equality needs the real tokenizer
and belongs to the full checks; do not regenerate goldens to hide a mismatch.

**MiMo config.** `models/mimo26/compiler/fixtures/config.json` supplies the
complete text and encoder configuration without weights. Change
`hidden_size` from `4096` to `4095` and run:

```sh
cargo test -p model-mimo26 --lib --locked --offline \
  config::tests::portable_config_admits_the_text_and_encoder_profile -- --exact
```

The test fails with `unsupported vision_config.out_hidden_size: expected
4095, received 4096`: the vision encoder's output width must equal the text
`hidden_size`. Restore `4096` and repeat the MiMo contributor command. For a frontend edit,
add a request case to `models/mimo26/compiler/tests/reasoning_controls.rs`.

The package fixtures emit a weightless package and plan its images at every
topology, using a small Rust-built stand-in module that cannot execute the
model. They cover package structure and verification, not whether a complete
model loads or produces correct tokens.

`python3 -B tools/dev/benchmark-loop.py --contributor qwen38 --output FILE
--repetitions 3` times the contributor phases into a new external file.

## Iterate on compatible checkpoints

Use `chopinfer serve --model "$CHOPINFER_QWEN38_CHECKPOINT" --tensor-parallel-size 4`
after installing the host, the Qwen helper and the native module once. The
helper emits a small package into the model cache (reusing an identical
published one), and the host loads every weight from the checkpoint on each
start. See [compatible local checkpoints](compatible-checkpoints.md) and
[direct weight loading](direct-loading.md).

For edits, use `bash ci/quick.sh` ([options](contributing.md#build-and-verify));
it does not replace the shell, Python, ABI, feature-variant and policy checks
in `ci/cpu.sh`. Model `checks` inventory sections select reference tests,
Python compile inputs and profile-build targets; model-specific CPU gates
remain explicit. `bash ci/quick.sh --report` explains which checks a change
needs ([testing.md](testing.md#explain-a-change-before-running-checks)).

## Landing a change

When kernels or launches change, capture the base and tip journals and run
[journal proof](../tools/journal_proof/README.md) for identity or the reviewed
change spec. State its claim and coverage in the commit message. For serving
changes, run the affected model's
[base-against-tip smoke](../tools/smoke/README.md) and check that every leg
passes.

Contributors then open a pull request, and maintainers merge it to main. When
main has moved since a smoke, its `paths.txt` tells whether the smoke still
applies ([serving smoke](../tools/smoke/README.md#build)). Keep captures, smoke
outputs and logs outside the repository under `$CHOPINFER_DATA`.

## Shared build hosts

On a host that several people or agents build on, run every heavy command
through [`tools/dev/heavy.sh`](../tools/dev/heavy.sh): each cargo build or
test, `ci/quick.sh`, `ci/cpu.sh` and qualification run. It keeps one queue
for the whole host, so the host runs a known number of them at once:

```bash
export CHOPINFER_HEAVY_DIR="$CHOPINFER_DATA/heavy"     # shared by the host's users
export CHOPINFER_TARGET_ROOT="$CHOPINFER_DATA/targets" # optional: a target per worktree
bash tools/dev/heavy.sh --label "my change" bash ci/quick.sh
bash tools/dev/heavy.sh --status
```

- Runs wait in arrival order for one of `CHOPINFER_HEAVY_SLOTS` slots
  (default 4), or `CHOPINFER_HEAVY_SLOTS_DURING_GATE` (default 2) while a
  gate run holds the host. `heavy.sh --gate COMMAND` runs a command ahead of
  the queue, such as a full `ci/cpu.sh` before a release: it waits only for
  another gate run, and running jobs carry on beside it.
- Everyone who shares a state directory sets the same slot and thread values
  (in the host's shared local.env) and belongs to one Unix group: the launcher
  makes the files and directories it creates group-writable.
- An admitted command runs on its slot's own `CHOPINFER_HEAVY_THREADS` CPUs
  (default 8), at `nice -n 19` and idle I/O priority, with Cargo's jobs, the
  test threads, `CHOPINFER_KERNEL_BUILD_JOBS` and the OpenMP, BLAS and Rayon
  pools capped at the same count. A command started inside an admitted one
  runs at once instead of queueing for a second slot. Slots bound CPUs, not
  memory: a full `ci/cpu.sh` peaks near 35 GB and a single-process bundle
  suite near 30 GB, so size the slot count to the host's memory too.
- Run `tools/journal_proof/capture.py` unwrapped, with
  `CHOPINFER_HEAVY_WRAPPER` pointing at `tools/dev/heavy.sh`: it then holds a
  slot only for its build and test run, not while it waits for its capture
  lock.
- `heavy.sh --pause REASON` holds new admissions, for an idle-host
  measurement for example, until `heavy.sh --resume`; running commands carry
  on.
- Give every worktree its own Cargo target. Cargo names workspace members the
  same in every checkout and judges freshness by source mtimes, so a target
  shared between worktrees can silently test another checkout's code. With
  `CHOPINFER_TARGET_ROOT` set and `CARGO_TARGET_DIR` unset, the launcher uses
  `wt-<name>-<hash of the worktree path>` under the root, and it refuses a
  target under the root that another worktree uses. Reusing a worktree for
  the next task, switching its branch in place, keeps its target warm; a new
  worktree starts a cold build. The launcher never deletes a target:
  `--status` reports the root's size, with a warning above
  `CHOPINFER_TARGET_WARN_GB` (default 250).
- Each finished run is one line of `$CHOPINFER_HEAVY_DIR/runs.jsonl` (its
  wait, run time and exit status); `--status` also prints the median wait of
  the last day. A run that a signal ends while it still waits, such as a
  caller's timeout, is a line too, marked `"abandoned": true`, and `--status`
  counts those separately.

Kernel builds in fresh targets can share cubins through the
[cubin cache](../crates/kernel-build/README.md#shared-cubin-cache). Never
share `*.stage` directories or one `RUNTIME_KERNELS_CUBIN_DIR` between
worktrees.

## Redeploy a dev server

`tools/dev/mimo-redeploy.sh` and `tools/dev/qwen38-redeploy.sh` build the
CUDA-feature host and the model's inventory-selected helper and bundle module,
install them privately below `--work-dir` (default `$CHOPINFER_DATA/dev-redeploy`), prepare
the direct package in `--model-cache-dir` (default
`$CHOPINFER_DATA/model-cache`), then run `chopinfer serve --model` with the
model's profile as defaults and wait for `/health` and `/v1/models`:

```sh
tools/dev/mimo-redeploy.sh --dry-run --port 8311   # print the sequence only
tools/dev/mimo-redeploy.sh --port 8311 --smoke     # redeploy and send one request
```

`--prepare-only` builds, installs and prepares without a GPU and prints
`prepared <package dir>`; `--help` lists the other options (checkpoint, rank
count, devices, drafter, profile overrides, `--serve-arg ARG` for other host
flags). On a later run for the same port, the script stops only the server it
started and waits for its GPUs to be released. It needs `nvidia-smi` and the
CUDA driver.

## Opt-in development startup

`CHOPINFER_DEV_SKIP_WARMUP=1` skips generation warmup before listening (the
host prints a development warning); graph capture then happens on the first
requests. Verification, weight planning and the checkpoint check still run.
`tools/dev/startup-timing-ab.sh --help` describes a startup timing run.

## Benchmark loop

`tools/dev/benchmark-loop.py` rebuilds through the redeploy scripts, serves,
and compares a speculation cohort against a saved baseline. Run it one at a
time on otherwise idle GPUs; it does not check for other GPU users and leaves
the development server running.

```bash
python3 -B tools/dev/benchmark-loop.py --model mimo --cohort spec-c1 \
  --baseline "$CHOPINFER_DATA/dev-loop/spec-c1.json" --save-baseline
# After an edit, repeat without --save-baseline to compare.
python3 -B tools/dev/benchmark-loop.py --model mimo --cohort spec-c1 \
  --baseline "$CHOPINFER_DATA/dev-loop/spec-c1.json"
```

Cohorts are `spec-c1`, `spec-c16` and `spec-c32`; `--model qwen38` uses its
MTP drafter; `--dry-run` prints commands only; redeploy paths go after `--`.
Baselines are never overwritten, and results are development measurements,
not qualification receipts.
