# Toolchain

Use Rust 1.89.0 (`rust-toolchain.toml`), CUDA 13.2 (`nvcc` 13.2.86 from
`CHOPINFER_CUDA_HOME`, default `/usr/local/cuda`) and Python 3.11 or newer (CPU checks and the
Docker build helper use the standard-library `tomllib`). Run Cargo from the
workspace root with `--locked --offline`; a fresh Cargo cache needs
`CARGO_NET_OFFLINE=false cargo fetch --locked` once. Kernel builds target
`sm_120f`; compiling device code needs no GPU. Run kernel-lab, benchmarks and
serving only on otherwise idle GPUs ([testing.md](testing.md)).

## local.env

`ci/cpu.sh`, `ci/quick.sh` and the shell tools source `local.env` from the
workspace root when it exists, and otherwise `$CHOPINFER_LOCAL_ENV` (default
`~/.config/chopinfer/local.env`), so every checkout and worktree on a machine
can share one file. The workspace file is gitignored. Create either from
[`local.env.example`](../local.env.example) and edit the paths:

```sh
cp local.env.example local.env
```

| Variable | Meaning |
|---|---|
| `CHOPINFER_DATA` | Model cache, private installs, qualification runs and logs |
| `CHOPINFER_MIMO_CHECKPOINT` | MiMo V2.6 Flash checkpoint directory |
| `CHOPINFER_MIMO_FIXTURES` | MiMo functional and GSM8K input fixtures |
| `CHOPINFER_MIMO_TOKENIZER_DIR` | MiMo tokenizer files; defaults to `CHOPINFER_MIMO_CHECKPOINT`. `ci/cpu.sh` needs one of them ([prerequisites](contributing.md#full-cpu-check-prerequisites)) |
| `CHOPINFER_MIMO_DFLASH_DIR` | Directory holding MiMo's DFlash `mask_embedding.pt`; defaults to the tokenizer or checkpoint directory's `dflash/`. `ci/cpu.sh` needs the file |
| `CHOPINFER_QWEN38_CHECKPOINT` | `Qwen/Qwen3.8-Flash-Next-FP8` snapshot, revision `236dfdf285828023ca3bcd3f37366c58a3469b13` |
| `CHOPINFER_QWEN38_TOKENIZER_DIR` | Qwen tokenizer files; defaults to `CHOPINFER_QWEN38_CHECKPOINT`. `ci/cpu.sh` needs one of them |
| `CHOPINFER_GLM5NEXT_CHECKPOINT` | `zai-org/GLM-5.3-Flash` checkpoint, revision `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`; `CHOPINFER_GLM5NEXT_EXTRA_CHECKPOINTS` names further checkpoint directories of any layout (`:` separated), each checked against its own layout ([checkpoints](deploy-glm5next.md#checkpoints)) |
| `CHOPINFER_QWEN38_LONG_REFERENCE` | Directory holding the pinned 128K `reference.json` and `reference.workload.json` of Qwen's `long_context_128k` check ([model-qwen38.md](model-qwen38.md)). `ci/cpu.sh` needs it |
| `CHOPINFER_QWEN38_LAYER_REFERENCE` | Directory of the reference runner's pinned Qwen layer taps (`run.json`, `f32.json`, `bf16.json` and `taps/`) that the layers check compares with. `ci/cpu.sh` needs it |
| `CHOPINFER_GLM5NEXT_LAYER_REFERENCE` | The same for GLM: two pinned runner runs of one workload ([models/glm5next/README.md](../models/glm5next/README.md)). `ci/cpu.sh` needs it |
| `CHOPINFER_CUDA_HOME` | CUDA 13.2 toolkit root for `nvcc` and headers; default `/usr/local/cuda` |
| `CHOPINFER_SCRATCH` | Temporary space for local tools |
| `CHOPINFER_REFRUN_PYTHON` | Reference-runner Python ([reference-runner.md](reference-runner.md#prepare-and-run)); `ci/cpu.sh` runs the reference checks with it on CPU and refuses to pass without it. Contributor modes do not need it |
| `CARGO_HOME`, `TMPDIR`, `PATH` | Set according to your Rust installation. Keep `TMPDIR` short: tests bind Unix sockets under it, and a socket path must stay under 108 bytes. `ci/cpu.sh` refuses a `TMPDIR` longer than 64 bytes |

Qualification Python tools read the file's `export CHOPINFER_*=value` lines;
variables already set in the shell take precedence. The shell scripts source
the file after your exports, so its plain `export` lines replace them; the
example fills `TMPDIR` and `CHOPINFER_SCRATCH` only when they are unset or
empty.

The CI scripts pass only each checkpoint's tokenizer to CPU tests (as
`CHOPINFER_MIMO_TOKENIZER_DIR` or `CHOPINFER_QWEN38_TOKENIZER_DIR`). Tests that
read checkpoint weights are slow and run only with
`CHOPINFER_CI_CHECKPOINT_TESTS=1`. Qwen3.8 metadata tests read configuration,
tokenizer assets and safetensors headers only, and skip when the checkpoint is
unset. Hugging Face snapshot links are resolved only to regular files in that
cache's `blobs` directory.

## Preflight

`bash ci/quick.sh --contributor qwen38 --doctor` checks the bounded
contributor setup (no CUDA, checkpoint or Torch needed); `bash ci/quick.sh
--doctor` also checks the native toolchain and the reference-runner
Python. Neither fetches missing prerequisites,
and both set `RUSTUP_AUTO_INSTALL=0` so a missing compiler stays a setup error.

First-time install and build-output paths are in the
[developer preview quickstart](developer-preview.md); the full CPU checks are in
[contributing.md](contributing.md); kernel build and lock mechanics are in
[kernel-library.md](kernel-library.md); building on a shared host is in
[development-loop.md](development-loop.md#shared-build-hosts).
