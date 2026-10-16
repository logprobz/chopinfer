# Contributor and coding-agent instructions

These rules apply to human contributors and AI coding agents alike. Setup,
build and checks are in [docs/contributing.md](docs/contributing.md).
Read [BUNDLE_ARCHITECTURE.md](BUNDLE_ARCHITECTURE.md),
[docs/kernel-library.md](docs/kernel-library.md) and
[docs/numerics.md](docs/numerics.md) before kernel or bundle work.

## Checks

- `bash ci/cpu.sh` runs the full CPU checks. Use `bash ci/quick.sh` while
  iterating.
- GPU tests run only when `CHOPINFER_KERNEL_GPU` or `CHOPINFER_KERNEL_GPUS`
  names idle GPUs. CPU results and shared-device measurements are not GPU
  correctness or performance evidence.

## Keep data and personal information out of the repository

- Write evaluation data, qualification and kernel-lab receipts, benchmark
  outputs, traces, profiles, logs, model weights, packages and model caches
  under `$CHOPINFER_DATA` (or another directory outside the tree), never into
  the repository. `fixtures/` holds only small inputs that a test reads.
- Never commit personal or machine information: names, emails, hostnames,
  usernames, local absolute paths, container names, GPU UUIDs or PCI bus IDs.
  Refer to local locations through variables such as `$CHOPINFER_DATA` or
  `$CHOPINFER_MIMO_CHECKPOINT` (see `local.env.example`).
- Docs state results (numbers, configuration, date) in prose. Do not link
  receipts or local logs.
- `tests/leak_guard.py` (part of `ci/cpu.sh`) enforces the mechanical part of
  these rules.

## Engineering rules

- Accuracy first. A kernel or serving change that can alter numerics needs
  bitwise-equality or fixed-batch evidence against the previous output or an
  f64 oracle before any performance claim.
- Prefer shared, model-neutral kernels and runtime code. Keep shared
  `runtime-*` crates free of model and bundle dependencies; bundles do not link
  `runtime-cuda`. Model-specific work lives in the owning model directory.
- Never regenerate a kernel lock or golden file to hide a mismatch. Find and
  fix the cause.
- Preserve C ABI symbols, schema and bundle IDs, locks, no-follow file checks
  and manifest validation. A model has one bundle ID; one package serves
  every rank count it supports, each as a topology with its own resources,
  so never add a per-rank-count bundle ID.
- Stage the specific files you changed (no `git add -A`), never force push and
  never use `--no-verify`. Use imperative commit subjects.
- House style for code comments, docs and messages: plain and direct, no
  em-dashes, straight quotes, no emoji.
