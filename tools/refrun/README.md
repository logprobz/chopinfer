# Reference runner

`tools/refrun` executes a model-owned reference adapter under declared
numerics, binds scores to authenticated inputs and emits gate fixtures.
Use `python3 -m tools.refrun --help` for commands. CPU adapter tests prove
mechanics; reference accuracy evidence requires the intended GPU execution.

## Prepare and run

Set `CHOPINFER_DATA`, checkpoint paths and `CHOPINFER_REFRUN_PYTHON` through
[local.env.example](../../local.env.example). Create a Python 3.12 environment
outside the repository with the exact hashed distributions in
[requirements-cu130.txt](requirements-cu130.txt):

```sh
python3.12 -m venv "$CHOPINFER_DATA/refrun-venv"
"$CHOPINFER_DATA/refrun-venv/bin/pip" install -r tools/refrun/requirements-cu130.txt
export CHOPINFER_REFRUN_PYTHON="$CHOPINFER_DATA/refrun-venv/bin/python"
```

Extra packages change the environment digest and prevent cache-key equality.
Use idle GPUs. Pin checkpoint bytes, upstream code, adapter and mode policy
before a capture. A workload holds ordered prompt/answer token IDs and score
starts; position p is scored by logits at p-1. Media files are already
processed tensors with shapes, dtypes and processor identities.

```sh
"$CHOPINFER_REFRUN_PYTHON" -B -m tools.refrun.stamps \
  --checkpoint "$CHECKPOINT" --lock "$CHECKPOINT_LOCK"
"$CHOPINFER_REFRUN_PYTHON" -B -m tools.refrun run \
  --adapter "$ADAPTER" --checkpoint "$CHECKPOINT" --lock "$CHECKPOINT_LOCK" \
  --workload "$WORKLOAD" --modes f32,bf16,f64 \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 --output "$RUN"
```

Keyed checkpoint locks also need `--lock-variant`; mode dequantization comes
from the model's declared reference policy. Use fresh output directories.
`python3 -m tools.refrun.dry_run run ...` validates the same arguments and
placement on CPU before reading weights or initializing CUDA. The runner
places whole model units across devices, not tensor-parallel reference
computation; there is no layer-major streaming for a model that does not fit.

F32 is the usual `exact` gate variant; BF16/f64 provide reference spread.
Each mode repeats its first computed sequence and requires identical scores.
Top-one ties go to lower token IDs. Near-tie gaps use logits, not differences
of rounded logprobs. Checkpoint stamps bind file identity and digests and
reject files changed during a run.

## Cache and fixture bindings

The cache key covers checkpoint and code digests, token/media inputs, score
policy, adapter/runner numerics, output selection, library/runtime versions,
device architecture and package environment. Names, categories, placement,
driver and timing are provenance. A new semantic input gets a new key;
recomputing an existing key must produce identical score bytes or fail.

`emit` joins modes in workload order into a skeleton, with a workload and
refset binding. `verify-binding` verifies the copied scores and inputs without
GPU execution. Never edit a reference to match a candidate. Adopt a changed
reference only after verifying the runner binding and reviewing the change:

```sh
python3 -m tools.refrun emit --skeleton "$SKELETON" --runs "$RUN" \
  --workload "$WORKLOAD" --lock "$CHECKPOINT_LOCK" \
  --modes f32=exact,bf16,f64 --output "$FIXTURE"
python3 -m tools.refrun verify-binding "$FIXTURE" --workload "$WORKLOAD"
python3 -m tools.qual refs status --model MODEL
python3 -m tools.qual refs adopt --model MODEL --replace-sha256 "$OLD_LOCK_DIGEST"
python3 -m tools.qual refs verify --model MODEL
```

The reference lock pins every input and directory member scope. Commit its
reviewed update together with the fixture and declaration. No numerical
reference gates without authenticated runner provenance. `cache gc` lists
unreferenced entries and deletes only with `--delete`; `diff-entries` compares
keys and scores. These commands do not grant new qualification.

### Current bound fixtures

MiMo's bound fixtures are `fixtures/mimo26/teacher-exact-reference.json`,
`fixtures/mimo26/multimodal/exact-reference.json` and
`fixtures/mimo26/multimodal/mixed-reference.json`. Qwen's are
`fixtures/qwen38/teacher-reference.json` (F32 `exact`, BF16 and f64 over the
teacher texts), `fixtures/qwen38/long-context-reference.json` (the last 1,024
positions of the frozen 32,768-token prompt) and
`fixtures/qwen38/image-probe-references.json`, whose workload and media stay
outside the repository at the location its declaration names. An unbound
fixture is reported `unbound`; a present binding must verify.

`fixtures/mimo26/multimodal/mixed-reference.json` holds single-image,
two-image, resized two-image, image/video and 32-image sequences. MiMo's
audio is served but not qualified in-tree: no reference holds an audio item,
so test audio with your own inputs. The adapter keeps image order and
accepts up to 32 images, one video and one audio item per sequence.
`tools/refrun/mixed_answers.py` generates each of the first four answers
greedily in F32 (a 60 to 100 word paragraph, at most 256 tokens, ending at a
checkpoint stop ID, at least 32 text positions), then forces every generated position in F32, BF16 and F64.
The media gate forces those answers with exact served prompt token IDs and
scores them with the usual F32 gate and BF16/F64 spread rule. These check
agreement with the reference, not labeled accuracy. The 32-image case is
encoder-capacity coverage. The workload and input tensors stay outside the
repository, at `references.media_mixed.workload` in the declaration.

## Taps and tapdiff

Add `--taps` to a GPU run to capture each forced sequence under
`$RUN/taps/<label>/step<NNNN>`. Taps are not cached: capture recomputes even
hits and requires the same score bytes. The first-sequence repeat has no
taps, checking that capture leaves scores unchanged.

Names use `<scope>.<site>`: `embed`, `L<nn>`, `final`, `enc.<tower>.B<nn>`,
`enc.<tower>` or `mtp.L<n>` scopes with canonical sites from
[taps.py](taps.py), or model-specific `x.<...>` sites, for
example `L03.attn.out`, `enc.vision.out`, `L05.x.hc.pre`. `--tap-sites`
takes site globs or `all` (default `taps.DEFAULT_SITES`), `--tap-rows`
selects text positions, and `--tap-bytes` bounds each sequence (default
4 GiB). Default text sampling is every position below 2048, then every
eighth and the last 16.

Encoder rows align by (request, media kind, item, row), not by token
position or storage order. An adapter declares the encoder axis with a
`Hook.rows` tuple of media kinds (one call per kind, each split into items
by its `grid_thw`) or wraps a one-item-per-call producer in
`taps.item(kind)`. Chunked prefill wraps each call in
`taps.chunk(start, stop)` with contiguous, ordered text ranges; taps must
cover the full sequence. The wrapper is inert outside capture, so taps do not
change the chunk schedule.

Each sequence gets one step holding `<name>.r0.bin` rows in the tapped
tensor's dtype (BF16, F32 or f64; integers as u32, booleans as u8),
`index.jsonl`, `items.json`, `capture.json` (`chopinfer.taps.v1`) and
`completion.json`. Encoder entries carry `"axis": "encoder"`, with their
segments under `encoder` in `items.json`.

**Engine captures** use the shared
[tap writer](../../crates/runtime-bundle-sdk/src/taps.rs) and the model's tap
declaration ([MiMo](../../models/mimo26/taps.json),
[Qwen](../../models/qwen38/taps.json)), which map engine sites to the
adapter's names. Build the bundle with its `diagnostic-taps` feature (for
example `bundle-mimo26/diagnostic-taps`) and set `CHOPINFER_TAPS_DIR` before
serving; release bundles refuse tap requests. `CHOPINFER_TAPS_LAYERS`,
`CHOPINFER_TAPS_SITES`, `CHOPINFER_TAPS_FROM`, `CHOPINFER_TAPS_BYTES`
(default 2 GiB per step) and `CHOPINFER_TAPS_SCOPE_JSON` select rows, sites,
budget and scope. The diagnostic path keeps serving's prefill schedule and
runs decode and verify graph kernels stage by stage; it adds
synchronizations and copies, so it cannot clear graph-only failures or
measure performance. For gates, bind the capture to its process and request
([tap captures](../qual/README.md#tap-captures)).

Compare on CPU with the runner Python (numpy):

```sh
"$CHOPINFER_REFRUN_PYTHON" -B -m tools.qual.tapdiff \
  --engine "$ENGINE_TAPS" --f32 "$RUN/taps/f32" \
  --bf16 "$RUN/taps/bf16" --f64 "$RUN/taps/f64"
```

Use `--request ENGINE=SEQUENCE` if request names differ. Text rows align by
request and position; encoder rows by (request, kind, item, row), so a runner
that encodes images before videos matches an engine that encodes them in
prompt order. Rank shards combine according to the declarations. Tapdiff
prints median and p90 relative error against F32 (`e`), the BF16-vs-F32
spread (`floor`), their ratio, and the F32-vs-f64 distance (`num`). A dense
row's excess is its error over three times its positive finite BF16 floor
(or over 0.05 without one); a dense tap fails when its median row excess
exceeds one. Integer set taps compare disagreement rates against three times
BF16's (or 0.05); router weights align by expert ID. Missing kind or item
metadata makes a capture incomplete, missing encoder rows are reported
incomplete, and extra engine rows are incompatible. The first-failure
histogram separates a broad failure at one operation from routing flips
followed by downstream error. `--ratio` and `--limit` override 3 and 0.05;
`--json` writes a receipt and refuses to overwrite it. A passing tapdiff is
diagnostic evidence, not a substitute for the serving gates.

| Exit | Verdict |
|---|---|
| 0 | Match |
| 1 | A tap diverged |
| 2 | Incompatible inputs |
| 3 | Incomplete capture or no rows compared |

## Encoder reference

`python3 -m tools.refrun.encoder --help` describes execution of the model's
encoder declarations. Capture exact, policy and reference-BF16 modes against
the same media. [Encoder parity](../qual/README.md#encoder-parity) defines
coverage and thresholds, and distinguishes serving-bound taps from lab runs.

## Frontend preprocessing

`python3 -m tools.refrun preprocess images --checkpoint CHECKPOINT --output
OUTPUT` runs the checkpoint's `AutoImageProcessor` in the main runner
environment (pinned transformers, Pillow and torchvision versions) and records
the processor class, configurations, package versions and source digests.
`--declaration models/<model>/qual.json` applies the declaration's
`media_preprocessing.PROCESSOR_OVERRIDES`: MiMo's ImageNet normalization,
8,192-pixel minimum and 1,048,576-pixel cap, all declared deviations from the
checkpoint. Qwen uses its checkpoint defaults. The check covers resize,
normalization and patch order of decoded RGB arrays, not decode, EXIF,
alpha, video frame selection or token caps.

`python3 -m tools.refrun preprocess audio` runs in a separate CPU environment:
`pip install --require-hashes -r tools/refrun/requirements-audio-reference.txt`
(CPython 3.12, torch and torchaudio CPU wheels). Never install it into the GPU
runner environment. It reconstructs MiMo-Audio's `load_audio` and
`mel_spectrogram` at the source revision the model's `media_audio_source`
declaration pins (Hann sinc resampling, HTK unnormalized mel, log floor
1e-7). Every reference runs twice and must be identical.

The `media_preprocessing` check builds `cargo build -p runtime-host --example
media-preprocessing --locked --offline`, runs it on the raw reference inputs
with the served recipes, and requires exact integer resize and patch values
and each floating stage within twice that stage's F32/F64 reference spread
(mel relative to each frame's peak). Adopt the
`media_preprocessing_reference` input with `tools.qual.refs adopt` before it
gates; a `--reference` override is reported unbound and cannot pass.

## Add an adapter

Start from [adapter.py](adapter.py), the two model adapters
and [tiny.py](tiny.py). An adapter module defines:

- `MODEL`: the model name in keys and workloads.
- `reference_code(checkpoint) -> {label: path}`: every upstream source file
  it executes, keyed by SHA-256. Label a checkpoint file `<model>/<file>`
  (rechecked by `verify-binding --checkpoint`), another repository file the
  adapter loads by its path `models/<model>/<file>` (rechecked against the
  working tree, like the adapter) and a package file `<package>/<path>`.
- `units(checkpoint) -> [place.Unit]`: placed modules in model order, each
  with the tensors it owns; a `host=True` unit stays on the host and is read
  with `place.HostRows`.
- `build(config, placement) -> adapter.Reference` for one mode; `config`
  carries the checkpoint, mode policy, dtype and the shared `Store`.
- optional `EXTRA_SHARDS = {file: prefix}`: locked safetensors outside the
  shard index (an audio tokenizer's own file).

`Reference` records every stored tensor's destination (module, parameter,
buffer or `host:<use>`; an unmapped tensor is refused), layout corrections
(for example a QKV projection stored in TP chunks), kernel replacements,
`media_inputs` and the tap map, with optional `forward` and `decode`
overrides for a vendor schedule. Use `store.linear`/`store.tensor` for
quantized weights (one `QuantTensor` per stored weight, shared by every
mode), distinct scratch slots for weights live at the same time,
`permute_rows` to reorder stored rows with their scales, and `HostRows` for
large tables. A tap-map module path selects its output (the first element of
a tuple); a `Hook` selects input or output, an argument and an optional
transform. Declare encoder row axes explicitly. A change to the adapter or
to the runner files in `keys.NUMERICS` changes reference keys.

Follow [Qwen's tiny test](../../models/qwen38/test_reference.py) and
[MiMo's tiny test](../../models/mimo26/test_reference.py): build a small
synthetic checkpoint, compare adapter scores with a direct upstream forward
bitwise, cover every stored tensor, verify cache replay without loading, and
include negative controls for layout corrections. These CPU tests prove
adapter mechanics, not full-model GPU parity.

The runner places a whole model on the selected GPUs; it has no layer-major
streaming for models that do not fit.

Reference adapter source bytes participate in cache identity, including
comments. Historical guide citations in pinned adapters retain their original
spelling; use this guide and numerics for the current contracts. A prose
cleanup must not silently invalidate the reference producer.
