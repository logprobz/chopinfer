# Compatible local checkpoints

```sh
chopinfer serve --model "$CHOPINFER_QWEN38_CHECKPOINT" --tensor-parallel-size 4
chopinfer serve --model "$CHOPINFER_MIMO_CHECKPOINT" --tensor-parallel-size 4
```

`serve --model DIR` loads the weights straight from the checkpoint's
safetensors files into device memory through the bundle's weight plan
([direct weight loading](direct-loading.md)) and writes no copy of them. The
host reads the architectures `config.json` names and asks each installed
family helper whether it accepts the checkpoint at the requested rank count;
exactly one must. That helper emits a small package into the model cache (the
native module, tokenizer, frontend and precision metadata, with every weight
image planned from the checkpoint), or reuses one published for the same
inputs. The host then refuses the checkpoint if its safetensors headers or
metadata files changed since the helper read them, and loads the weights.

The helper also names the serving profile (an entry of
`models/<model>/profiles.json`). Its speculation, scheduling, limits and device
map become defaults under explicit options;
`--speculative-tokens 0` serves autoregressively. Command-line options are in
the [command reference](cli.md#local-checkpoint-serving).

| Family | Checkpoints | Topologies | Default profile | Served name |
| --- | --- | --- | --- | --- |
| Qwen (`chopinfer-model-qwen38`) | Complete Qwen3.8 Flash checkpoints in a layout the family serves: the official FP8 export, its revisions and fine-tunes, and the mixed NVFP4, MXFP8 and BF16 domain of [quantized checkpoints](quantized-checkpoints.md#qwen-coverage) | TP2, TP4 | `tp4-mtp`, `tp2-mtp`; autoregressive when the MTP layer stores a format it has no composition for; vision encoders when the checkpoint has a vision tower | `local-<checkpoint identity>`, the pinned revision included |
| MiMo (`chopinfer-model-mimo26`) | The pinned MOPD and RL checkpoints, and any other MiMo V2.6 Flash checkpoint in the encodings MiMo serves ([MiMo checkpoints](#mimo-v26-flash-checkpoints)) | TP2, TP4 | `tp4-dflash` when the checkpoint's DFlash drafter checks (MOPD), else `tp4-mtp`; `tp2-ar`; vision and audio encoders | The pinned checkpoint's name, else `local-<metadata identity>` |
| GLM (`chopinfer-model-glm5next`) | Any GLM-5.3 Flash checkpoint whose formats the projection declaration serves for each projection class, the official `zai-org/GLM-5.3-Flash` FP8 checkpoint included ([deploy GLM](deploy-glm5next.md#checkpoints)) | TP4 | `tp4-mtp` for every layout; vision encoder | `GLM-5.3-Flash` |

Sizes come from `config.json` and must match a compiled geometry in
`models/<m>/geometries/supported.json`. Runtime geometry selection uses
`<model>-<geometry>-tp<n>` topology IDs, with one bundle ID per family.
Tensor formats, semantics, vocabulary, special tokens and chat rendering
must match the family. Add another geometry with the
[same-architecture recipe](adding-a-model.md#same-architecture-checkpoints).
LoRA adapters, new architectures, encodings outside the family's declaration
and checkpoint code are not loaded. A checkpoint no helper accepts is refused
with each helper's reason. A Qwen vision tower that fails the BF16 shape check
refuses the start.

`serve --model` does not run Cargo, nvcc, evaluations or qualification.
Packages report `compatible-not-evaluated`; `--served-model-name` changes
display names only. Starting another checkpoint means starting another
serving process.

### What serving checks, and what `verify` adds

The checkpoint identity is the SHA-256 of every safetensors file's name,
length and header. It binds each tensor's name, dtype, shape and offsets, not
the tensor bytes: a payload rewritten in place with the same length still
loads. `chopinfer verify --model DIR` hashes every file, offline, against the
digest Hugging Face recorded for it (SHA-256 for LFS files and manifest
entries, git blob SHA-1 for the hub's other small files). It reads the
recorded digests from:

- a hub cache snapshot (`.../snapshots/<revision>/`), whose blob names are the
  digests;
- a directory from `hf download --local-dir`, whose
  `.cache/huggingface/download/<file>.metadata` records each file's etag;
- any directory, with `--manifest FILE[#ENTRY]`: a JSON object from relative
  path to `{"sha256": ..., "byte_length": ...}` (`byte_length` optional),
  optionally under the named top-level entry.

A plain copy without that metadata needs a manifest. MiMo's source lock is
one:

```sh
chopinfer verify --model "$CHOPINFER_MIMO_CHECKPOINT" \
  --manifest models/mimo26/checkpoint-files.json#mopd --allow-unrecorded
```

`verify` reads only regular files and does not follow links, except a hub
snapshot's links into its own `blobs/`. A file added, removed or rewritten
during the run fails it. It reports every mismatch, missing shard and file
without a recorded digest, and exits nonzero unless every file verifies;
`--allow-unrecorded` accepts unrecorded files other than safetensors files and
their indexes (MiMo's lock omits the README, images and Python files).

## Serve your own checkpoint

Another checkpoint of a supported model (a fine-tune, another revision or
another quantization) needs no JSON and no change to the repository. The
family helper reads the checkpoint's `config.json`, its quantization metadata
and its safetensors headers, and admits it by its layout when:

- its sizes match a compiled geometry;
- every tensor the family declares is present with a dtype, shape and
  encoding that the family's projection declaration
  (`models/<m>/contract/projections.json`) serves for that tensor's role, and
  a supplied kernel serves it at the requested rank count;
- its tokenizer configuration and chat template state the official
  release's. The excerpt each family checks against is
  `models/<m>/compiler/fixtures/tokenizer-semantics.json`. Keys that record
  provenance (`name_or_path`, `_name_or_path`, `transformers_version`,
  `is_local`) are ignored, and the files may be serialized differently.

Each family then names the checkpoint by what it read:

- Qwen serves every checkpoint as `local-<checkpoint identity>` with its own
  tokenizer files, and refuses a generation config that asks for behavior it
  does not implement.
- MiMo serves a checkpoint that matches a registered source lock under that
  checkpoint's name and FP8 KV calibration. It serves any other checkpoint
  under a local identity with BF16 KV only, and with its DFlash drafter only
  when that drafter checks ([MiMo checkpoints](#mimo-v26-flash-checkpoints)).
- GLM derives the checkpoint's layout from the formats its
  `quantization_config` gives each projection class and from the dtypes of
  its KDA convolution and mHC vector tensors. Every layout serves with
  `tp4-mtp`. Tokenizer files other than the pinned revision's are packaged
  under a frontend that names their content (tokenizer repository `local`).

A refused checkpoint is refused before any weight is read. The reason names
the tensor, its format and the formats the family serves for its role, or
the tokenizer key or chat template that differs.

Admission is not qualification. To gate a checkpoint against the model's own
checks and limits, declare it as an
[external variant](../tools/qual/README.md#external-variants) in a directory
outside the tree, such as `$CHOPINFER_DATA/qual-variants`. The repository
declares only the model authors' checkpoints.

## Install the implementation once

Release archives contain `bin/chopinfer`, `bin/chopinfer-host`, the family
helpers `bin/chopinfer-model-qwen38`, `bin/chopinfer-model-mimo26` and
`bin/chopinfer-model-glm5next`, the packagers `bin/qwen38-package` and
`bin/mimo26-package`, and `lib/libbundle_qwen38.so`,
`lib/libbundle_mimo26.so` and `lib/libbundle_glm5next.so`. To build the same
layout from source, follow the
[quickstart](developer-preview.md#obtain-and-verify-a-checkpoint).

Helpers are discovered only beside the running host executable (not on PATH,
in the checkpoint or in the working directory). The directory and files must
not be group or world writable, and helper symlinks are refused, so install
Cargo outputs into this layout first. Each helper finds its native module
beside itself or in the adjacent `lib/` directory.

## Startup and package reuse

Startup prints `startup provider_discovery`, `startup direct_package` (package,
checkpoint and metadata identities, the profile, and whether the package was
emitted, reused or replaced), `startup direct_checkpoint`, `startup
direct_profile`, `startup weight_plan`, `startup direct_load` and `startup
readiness`. A package is reused when the helper executable, native module,
rank count, activation policy, served modalities and checkpoint identities
match; otherwise it is emitted again (seconds, no weights read). A reused package is
still verified in full; a damaged one is moved aside as `q-...` and replaced.
`tools/dev/startup-timing-ab.sh local` measures `serve --model` startup.

## Prepare without serving

The helper's `prepare` emits the package that `serve --model` would select and
prints its record (directory, identities and serving profile):

```sh
chopinfer-model-qwen38 prepare --model "$CHOPINFER_QWEN38_CHECKPOINT" \
  --cache-dir "$CHOPINFER_DATA/model-cache" --tensor-parallel-size 4
chopinfer-model-qwen38 profile --model "$CHOPINFER_QWEN38_CHECKPOINT" --tensor-parallel-size 4
```

Serve the package explicitly with its checkpoint and profile, for example
`chopinfer serve --bundle "$PACKAGE" --checkpoint
"$CHOPINFER_QWEN38_CHECKPOINT" --profile models/qwen38/profiles.json#tp4-mtp`.
The package directory is `p-<manifest sha256>` under the cache root
(`--model-cache-dir`, or the helper's `--cache-dir`; see the
[command reference](cli.md#local-checkpoint-serving) for the default). The Qwen
helper refuses a native module that does not declare the contracts it
expects; install a matching helper and module together.

## Inspect and prune

```sh
chopinfer-model-qwen38 cache list --cache-dir "$CHOPINFER_DATA/model-cache"
chopinfer-model-qwen38 cache prune --cache-dir "$CHOPINFER_DATA/model-cache"
chopinfer-model-qwen38 cache prune --cache-dir "$CHOPINFER_DATA/model-cache" --apply
```

Prune previews by default; `--apply` removes. `--identity KEY` (repeatable)
narrows the selection. The last selection and packages leased by running
servers are protected, and pruning defers while a preparation holds the cache
gate. Prune also removes prepared weights (`w-`) and snapshots (`s-`) left by
older builds.

## API and independent implementations

`model_compiler_sdk::preparation::direct::prepare_direct` returns the package
record and its cache lease (keep it while the package is in use). A family
implements `helper::Provider` (name, profiles, metadata probe, `emit_direct`)
and its installed helper calls `helper::run`; Qwen exposes
`model_qwen38::preparation::QwenFamily`. `runtime_bundle::preparation` owns the
helper protocol (newline-delimited JSON records of at most 1 MiB on
stdin/stdout, version 2). The host links neither the compiler SDK nor model
crates; helper code is installed trusted code. The
[independent-model SDK workflow](external-models.md) uses the same protocol.

## MiMo V2.6 Flash checkpoints

`chopinfer-model-mimo26` serves the pinned MOPD and RL checkpoints, matched
by the source lock (`models/mimo26/checkpoint-files.json`) of their
`config.json`, `model.safetensors.index.json`, `tokenizer.json` and
`tokenizer_config.json`. A checkpoint that matches no lock is served under a
local identity, named by the SHA-256 of a lock of those four files measured
at preparation: repository `local`, the digest's first 40 hex digits as its
revision, and `local-<digest>` as its served name. Its tokenizer
configuration must state the MOPD release's, and its tokenizer files must
build the declared tokenizer. Its DFlash drafter is served only when the
drafter's config matches the geometry and its mask archive is the checked
one; otherwise its packages draft with MTP at TP4.

The weight plan admits only encodings MiMo's kernels serve: BF16, FP8 E4M3
128x128 blocks with F32 inverse scales, and MXFP4 routed experts; the DFlash
drafter and encoders stay BF16. Other quantizations, including NVFP4 or MXFP8
experts, are refused before any weight is read, naming the tensor and the
formats MiMo serves (see
[MiMo coverage](quantized-checkpoints.md#mimo-v26-flash-coverage)).

FP8 KV calibration belongs to the authenticated checkpoint, not its execution
geometry. Its `checkpoints.json` entry must declare `fp8_kv_scales` matching
the compiled FP8 layout. A registered checkpoint without calibration, and
every local identity, advertise BF16 KV only, even when they share
`v26-flash`. An explicit `--kv-cache-dtype fp8` request is refused with the
package's supported KV formats. The existing MOPD and RL scale tables are
preserved; registering a checkpoint does not establish calibration or
qualification.

Preparation authenticates metadata and headers, not weight payloads. A
fine-tune that keeps all four metadata files byte-identical to MOPD's or RL's
therefore matches that lock, and keeps its served name and FP8 KV
calibration. `chopinfer verify` with the source lock as its manifest (above)
tells the released weights apart.

A MiMo package carries the matched checkpoint's served name or its local
name; `--served-model-name` changes the display name. Only the released MOPD
checkpoint has been checked on this path; any other checkpoint is
compatible-not-evaluated.

For MiMo geometries without encoder kernels, `--modalities auto` serves text.
Explicit media requests are refused before preparation. The host's v1 audio recipe requires 24 kHz, a 600-second bound within the
encoder's 60,001-frame cap (hop 240), and 6,000-frame segments. Preparation
refuses audio selections with other values. The bound is independent of the
checkpoint training crop duration. Flash retains these values.
