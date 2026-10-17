# GPU reference runner

Use the reference runner when adding a model, rebuilding a gate fixture or
finding the first numerical divergence. It runs a model's upstream reference
code (HF transformers or the vendor's PyTorch modeling file) on GPUs and
scores sequences under a declared numerics policy; qualification gates
compare the engine against those scores. The shared runner in
[tools/refrun/](../tools/refrun/README.md) owns loading, placement, exact
dequantization, modes, scoring and cache identity. A per-model adapter,
`models/<model>/reference.py`, owns everything model-specific.

Reference evaluation is GPU-only. CPU `--help`, binding checks and tiny
synthetic unit tests are fine; full-model CPU reference runs are not.

The current adapters are [MiMo](../models/mimo26/reference.py) and
[Qwen](../models/qwen38/reference.py). Read their layout and replacement
records before interpreting a mismatch: upstream code does not validate our
checkpoint-layout corrections.

## Numerics and gate decisions

| Mode | Computation | Default weight rule | Log-softmax |
|---|---|---|---|
| `f32` | IEEE F32, TF32 off | `exact` | F32 |
| `f64` | f64 | `exact` | f64 |
| `bf16` | BF16 with full F32 reductions | `hf-bf16` | F32 |

Every mode disables reduced-precision reductions, uses deterministic
algorithms and the math SDPA backend, and runs one sequence per forward,
with no torch compilation or CUDA graphs. The runner sets
`NVIDIA_TF32_OVERRIDE=0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, matmul
precision `highest` and `fp32_precision="ieee"`, deregisters torch's eager
DSL kernels and refuses to run if any remain. Do not mix torch's legacy
`allow_tf32` API with the precision API. Scores are serialized as f64; that
does not turn an F32 computation into f64.

| Dequantization rule | Meaning and use |
|---|---|
| `exact` | Decode the stored codes and scales without losing the product. F32 refuses a product it cannot represent exactly; f64 supports them. Use for BF16-scaled FP8 and representable power-of-two MXFP4 scales. |
| `rounded` | Round the exact product once to the target dtype. MiMo's F32-scaled FP8 projections need this for F32. |
| `hf-bf16` | Form the F32 product, round to BF16, then widen. Reproduces upstream BF16 loading. |

These rules are distinct cache inputs: `f32:hf-bf16` is an F32 computation
on BF16-rounded weights, not a BF16 computation. Upstream internal casts
(F32 rotary tables, HF's eager softmax in F32) still round in the f64 mode
unless the adapter replaces them. Model-defined activation or KV
quantization stays in all modes.

Gates use the runner's F32 output as the gate variant (fixture variant
`exact`) and its BF16 and f64 runs as the reference's own spread; a
fixture's binding records each variant's mode and weight rule. MiMo uses
`rounded` F32 dequantization, Qwen `exact`. The limits derived from that
spread, and the near-tie rule for top-1 flips, are in
[numerics](numerics.md#serving-qualification-gates); the exact computation
is in [check kinds](check-kinds.md). Gates score every position; do not
sample scored positions.

## Prepare and run

Run commands from the repository root with `CHOPINFER_DATA`, the checkpoint
variables and `CHOPINFER_REFRUN_PYTHON` set in your `local.env`
([local.env.example](../local.env.example)).

**Environment.** `CHOPINFER_REFRUN_PYTHON` is the Python of a CUDA virtual
environment, outside the repository, holding exactly the distributions
pinned in [requirements-cu130.txt](../tools/refrun/requirements-cu130.txt)
(torch 2.14.0+cu130 and transformers 5.16.1 among them). The file pins
every distribution by hash and names the torch wheel index as an extra
index. Create the environment with Python 3.12 and install the pins:

```sh
python3.12 -m venv REFRUN_VENV
REFRUN_VENV/bin/pip install -r tools/refrun/requirements-cu130.txt
```

Avoid extra packages: cache hits and reproducing a bound fixture's keys
require the same torch, CUDA runtime, cuBLAS, device architecture and
installed package set (one extra distribution changes `env_freeze_sha256`).
CPU binding verification needs none of this. Job scripts may set
`TORCH_DISABLE_NATIVE_JIT=1` before importing torch as a backstop, and
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for offline runs.

**Workloads.** A workload (`chopinfer.refrun.workload.v1`,
[workload.py](../tools/refrun/workload.py)) holds the model and ordered
sequences, each with name, category, prompt `token_ids`, `score_from` and an
answer policy. Forced answer IDs are appended to the prompt; a greedy answer
(`max_new_tokens`, `stop_ids`) is generated in the first mode, then forced in
every mode. Position p is scored from the logits at p-1. Media inputs are
already-processed little-endian tensor files relative to the workload, with
dtype, shape, optional grid and processor code/config hashes; the runner
does not run the processor.

**Run.** First populate checkpoint stamps on CPU, without loading a model,
then run the workload on idle GPUs. The examples use `$WORKLOAD`, `$RUN`,
`$SKELETON` and `$FIXTURE` for caller-supplied inputs and fresh output
locations outside the repository.

```sh
"$CHOPINFER_REFRUN_PYTHON" -B -m tools.refrun.stamps \
  --checkpoint "$CHOPINFER_MIMO_CHECKPOINT" \
  --lock models/mimo26/checkpoint-files.json --lock-variant mopd

"$CHOPINFER_REFRUN_PYTHON" -B -m tools.refrun run \
  --adapter models/mimo26/reference.py \
  --checkpoint "$CHOPINFER_MIMO_CHECKPOINT" \
  --lock models/mimo26/checkpoint-files.json --lock-variant mopd \
  --workload "$WORKLOAD" --modes f32,bf16,f64 --dequant f32=rounded \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 --output "$RUN"
```

For Qwen, use `models/qwen38/reference.py`,
`models/qwen38/checkpoint-files.json` and `$CHOPINFER_QWEN38_CHECKPOINT`, and
omit `--lock-variant` and `--dequant`. `--modes` defaults to `f32,bf16,f64`
and `--devices` to the first four GPUs. A `MODE:RULE` label (written as
`MODE-RULE.json`) compares two weight rules for one mode in one load;
`--dequant MODE=RULE` overrides a mode's rule. The output directory must be
fresh. It gets one file per mode (sequence name, category, key document,
`key_sha256` and scores per entry) and `run.json` (placement, load rates,
timings, cache hits and determinism checks); per-sequence times go to stderr
as JSON lines. Each mode repeats its first computed sequence and requires
byte-identical score documents.

`python3 -m tools.refrun.dry_run run ...` takes the same arguments on CPU: the
runner's own parser and every step before the first weight is read (adapter,
workload, modes and rules, source lock and stamps, units, placement, with the
devices checked against `CUDA_VISIBLE_DEVICES`). It creates no output and
initializes no CUDA, so a queued job's exact command line can be checked
before it takes GPUs.

**Scores.** Position p >= `score_from` is scored by the logits at p - 1: the
forced token's logit and logprob, top-1, the margin (top-1 minus top-2
logit, subtracted in f64) and the top 20 IDs with logits and logprobs. Ranks
come from a stable descending sort of the logits, so equal logits go to the
lower token ID and top-1 is the greedy argmax even where log-softmax rounds
two logits to one logprob. Gates judging how close the reference came to a
produced token use `margins.gap` on logits, never differences of
`topk_logprobs`.

**Placement and loading.** Units are placed in model order over the
selected GPUs by stored bytes; activations move at boundaries. This is one
process, not tensor-parallel reference computation. The first cache miss
triggers one checkpoint load; each mode builds and drops its own model over
shared stored tensors. The loader runs up to eight shard readers with
bounded pinned staging. Quantized weights stay as stored codes and scales and
dequantize per call into reusable scratch; host units gather only requested
table rows (`HostRows`). Split long f64 workloads into separate bounded
jobs: small expert GEMMs and f64 vision attention run far below peak rates.
[calibrate.py](../tools/refrun/calibrate.py) measures rates on a GPU.

**Checkpoint integrity.** Before reading, `config.json`, the shard index and
every shard it names (and any extra shards the adapter declares) must match
the lock's sizes and SHA-256s, and files must remain unchanged until outputs
are written. The stamp cache keys on resolved path, size, mtime, ctime and
inode and rehashes on change; a file changed and restored during a run is
refused ([stamps.py](../tools/refrun/stamps.py) states its filesystem and
clock assumptions). Vendor code outside the lock is read through
`Checkpoint.pinned` with an adapter-pinned digest.

## Cache and fixture bindings

The cache (`$CHOPINFER_DATA/refcache/v1`; `--cache DIR` elsewhere,
`--no-cache` to disable) keeps every scored (sequence, mode), so a score is
reused only when its inputs and numerics match.

**Keys.** `keys.document` builds the `chopinfer.refcache.key.v1` document of
one (sequence, mode, outputs); `key_sha256` is the SHA-256 of its canonical
JSON. Keyed: the model; the checkpoint source lock and files read; the
sequence (prompt and answer token IDs, `score_from`, answer policy, media
tensor content hashes, dtypes, shapes and processor identity); the upstream
code hashes; the transformers version and a digest of every installed
transformers Python file; the adapter and the runner files that compute
scores (`keys.NUMERICS`); the mode policy; torch, CUDA runtime and cuBLAS
versions, device architecture and the installed package set; and the
outputs. Sequence names, categories, raw media digests, driver, GPU names,
timings, placement and git HEAD are provenance only, as are runner files
that do not compute scores (stamps, taps, keys).

**Layout.**

```
entries/<k[0:2]>/<key_sha256>/   key.json, scores.json, entry.json (seal and provenance)
refsets/<refset_sha256>.json     model, workload digest, per mode the ordered
                                 [name, key_sha256] pairs, assembled fixture digest
stamps/                          the checkpoint stamp cache
tmp/                             staging; entries and refsets appear by one rename or link
keep                             optional: git refs (one per line) whose fixtures gc keeps
```

Writes are create-only: a second producer of a key must write the same
`scores.json` bytes or fail (`Nondeterministic`); `run --recompute`
computes every hit again to check exactly that. `entry.json` seals the
byte length and SHA-256 of `key.json` and `scores.json`. There is no
invalidation: a semantic change gives a new key.

```sh
python3 -m tools.refrun cache gc [--ref REF ...] [--delete]
python3 -m tools.refrun diff-entries A B
python3 -m tools.refrun verify-binding FIXTURE ... [--checkpoint DIR] [--workload WORKLOAD.json]
```

`cache gc` lists entries and refsets older than 14 days that no fixture
under `fixtures/` binds on `main`, on a ref in `keep` or on a `--ref`, plus
stale `tmp/` items, and deletes them only with `--delete` (never `stamps/`).
`diff-entries` compares two entry keys or two refset digests (differing key
sections, top-1 changes, largest logprob and margin change).

**Assemble a bound fixture** on CPU after collecting all GPU runs:

```sh
"$CHOPINFER_REFRUN_PYTHON" -B -m tools.refrun emit \
  --skeleton "$SKELETON" --runs "$RUN" --workload "$WORKLOAD" \
  --lock models/mimo26/checkpoint-files.json --lock-variant mopd \
  --modes f32=exact,bf16,f64 --topk 5 --output "$FIXTURE"
```

The skeleton supplies schema, provenance and named items without scores.
`emit` joins each mode's runs from all `--runs` directories in workload
order and copies forced scores into the variants (`--modes LABEL=VARIANT`,
default the label); `--fields`, `--topk` (default 5) and `--greedy-fields`
select the copied columns. It writes `<stem>.workload.json` and the media
beside the fixture, adds the binding, writes the refset and prints the
fixture SHA-256. (`emit` is dispatched separately, so the root `--help`
omits it.) Commit the fixture, workload and media together with the new
pins in `models/<model>/qual.json` (`references.<name>.sha256`, and
`workload_sha256` when the workload changed), then adopt the inputs (below).

**Binding fields.** The `binding` block holds `refset_sha256`,
`assembled_fixture_sha256` (the fixture's canonical JSON without the
block), `modes` (per run label the mode, fixture variant, dequant rule,
policy digest and ordered `[name, sequence index, key_sha256]` entries),
`key` (sections every entry shares) and `sequences`, `digests`, `lock`
(path and variant) and `workload_sha256` (a digest of the workload's model
and ordered sequences, stable under reformatting). A generated answer is
bound by its greedy entry and the forced entry of that answer, and stated as
the item's `answer_token_ids`. `cache.verify` raises `StaleReference` naming
the first differing field:

| Field | Check |
|---|---|
| `schema`, `lock`, `reference_code`, `adapter`, `environment`, `mode_policy`, `key_sha256`, `refset_sha256` | the block agrees with itself: sub-digests with sections, keys with their rebuilt documents, the refset with its digest |
| `lock` | the lock file (and variant) in the working tree |
| `adapter` | every adapter and runner numerics file (`keys.NUMERICS`) in the working tree |
| `mode_policy` | `modes.policy` in the working tree |
| `reference_code` | files labelled `models/<model>/<file>` (repository code an adapter loads) against the working tree; with `--checkpoint`, files labelled `<model>/<file>` against the checkpoint directory; package files only through the key |
| `workload` | with `--workload`, the workload has the bound digest and every mode scores its sequences in order |
| `sequence` | each bound name is a fixture item whose `prompt_token_ids`, `score_from` and `answer_token_ids` match |
| `scores` | with a cache, every copied scores field equals the entry's; absent entries are counted, not refused |
| `assembled_fixture` | the fixture's content still has the bound digest |

**Gate checks.** A gate's `common.pinned_reference` verifies the declared
byte, schema and checkpoint pin, and that the binding names the gate's
checkpoint lock; `chopinfer eval`, which captures the teacher checks, reads
pinned references by the same rules. The pinned bytes stand for the rest of the binding, which
is not verified again on each read. A reference pinned without a spread
variant (floor-only) is the exception: each read verifies its whole binding,
resolving the canonical sibling or declared workload (including
environment-backed locations). Adoption verifies a candidate's whole
binding before pinning it; available cache entries must match and absent
ones are counted. A mismatch refuses with `stale_reference: FIELD: ...`:
repair the input or regenerate and repin the reference, never remove a
binding to bypass it. Numerical gate inputs, including exact and MTP
references, require a runner binding: adoption pins one only after verifying
it, and gate admission refuses a reference lock object that does not record
it. The cache reader can report `unbound`, but such fixtures cannot qualify
numerical gates.

`verify-binding` checks a fixture standalone against the working lock,
adapter and numerics files, mode policies, sequences and fixture digest. It
does not enforce the declaration's pin; pass `--workload` and
`--checkpoint` (`$CHOPINFER_MIMO_CHECKPOINT` or
`$CHOPINFER_QWEN38_CHECKPOINT`) to check those too:

```sh
"$CHOPINFER_REFRUN_PYTHON" -B -m tools.refrun verify-binding "$FIXTURE" \
  --workload "$WORKLOAD" --checkpoint "$CHOPINFER_MIMO_CHECKPOINT"
```

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

### Adopt and verify gate inputs

A binding proves the runner inputs and copied scores. The model's reference
lock, `models/<model>/qual-references.lock.json`, additionally pins the bytes
gates read, each directory with the member scope its entry records. Before
replacing fixture bytes or changing the declaration, record the current lock
digest; after emission and declaration updates, adopt on CPU:

```sh
python3 -B -m tools.qual refs status --model mimo26   # prints the lock digest
python3 -B -m tools.qual refs adopt --model mimo26 --replace-sha256 "$OLD_LOCK_DIGEST"
python3 -B -m tools.qual refs verify --model mimo26
```

Omit `--replace-sha256` for an initial lock; unchanged adoption is
idempotent. Adoption verifies runner bindings before recording
`runner-binding` and `runner-workload` provenance; gate admission requires
a numerical reference's `runner-binding` record. `verify` and `status`
need every declared input, including environment-backed data. Commit the
lock with the fixture, workload and declaration changes. See
[adopted qualification inputs](../tools/qual/README.md#adopted-qualification-inputs).

## Taps and tapdiff

Add `--taps` to a GPU run to capture each forced sequence under
`$RUN/taps/<label>/step<NNNN>`. Taps are not cached: capture recomputes even
hits and requires the same score bytes. The first-sequence repeat has no
taps, checking that capture leaves scores unchanged.

Names use `<scope>.<site>`: `embed`, `L<nn>`, `final`, `enc.<tower>.B<nn>`,
`enc.<tower>` or `mtp.L<n>` scopes with canonical sites from
[taps.py](../tools/refrun/taps.py), or model-specific `x.<...>` sites, for
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
[tap writer](../crates/runtime-bundle-sdk/src/taps.rs) and the model's tap
declaration ([MiMo](../models/mimo26/taps.json),
[Qwen](../models/qwen38/taps.json)), which map engine sites to the
adapter's names. Build the bundle with its `diagnostic-taps` feature (for
example `bundle-mimo26/diagnostic-taps`) and set `CHOPINFER_TAPS_DIR` before
serving; release bundles refuse tap requests. `CHOPINFER_TAPS_LAYERS`,
`CHOPINFER_TAPS_SITES`, `CHOPINFER_TAPS_FROM`, `CHOPINFER_TAPS_BYTES`
(default 2 GiB per step) and `CHOPINFER_TAPS_SCOPE_JSON` select rows, sites,
budget and scope. The diagnostic path keeps serving's prefill schedule and
runs decode and verify graph kernels stage by stage; it adds
synchronizations and copies, so it cannot clear graph-only failures or
measure performance. For gates, bind the capture to its process and request
([tap captures](../tools/qual/README.md#tap-captures)).

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

`tools/refrun/encoder.py` evaluates an encoder declaration
(`crates/encoder-spec`, `models/<model>/contract/encoders/*.json`) in torch,
stage by stage, over every media item of a workload the declaration takes.
It is the reference the engine's encoders are compared with
([encoder parity](../tools/qual/README.md#encoder-parity)), independent of
the upstream modeling code.

```sh
"$CHOPINFER_REFRUN_PYTHON" -B -m tools.refrun.encoder \
  --declaration models/<model>/contract/encoders/<name>.json \
  --adapter models/<model>/reference.py --checkpoint "$CHECKPOINT" \
  --lock models/<model>/checkpoint-files.json [--lock-variant V] \
  --workload "$WORKLOAD" --output "$RUN" [--blocks] [--tap-bytes N]
```

| Mode (default `exact,policy,reference-bf16`) | Arithmetic |
|---|---|
| `exact` | every value and product in f64; no output rounding |
| `policy` | the declaration's precision policy: `f32-pairs.v1` (F32 rows, GEMM and attention operands as BF16 hi+lo pairs, one BF16 rounding at the output) or `bf16-legacy.v1` (BF16 rows, F32 accumulators) |
| `policy:<id>` | the same stages under another policy, labelled `policy-<id>` |
| `reference-bf16` | unfused BF16 PyTorch, as HF or vLLM run an encoder in BF16 |

Each component variant id maps to one implementation: shared ones in
`encoder_ops.py` (which lists every rounding point), model-specific ones
(`<model>:...`) in `models/<model>/encoder_reference.py`. An id, patch order
or output layout without one is refused by name. Weights come through the
declaration's weight mapping from the checkpoint the runner verifies against
its lock. Each mode repeats its first item and requires the same bytes.

Outputs are `<label>.json` per mode (`chopinfer.refrun.encoder-rows.v1`:
per sequence its key and each item's row digest), `run.json`, and taps in the
engine format under `taps/<label>/step<NNNN>`: `enc.<tower>.out` (the rows
the language model reads), `enc.<tower>.x.unrounded` (before the output
rounding), `enc.<tower>.x.codes` (a quantize stage's codes) and, with
`--blocks`, `enc.<tower>.B<nn>.out` for every transformer block, always in
the rows' own order. Encoder keys (`chopinfer.refcache.encoder-key.v1`)
include the declaration digest and the encoder code
(`keys.ENCODER_NUMERICS`), kept separate from `keys.NUMERICS` so text keys do
not change with encoder code.

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

Start from [adapter.py](../tools/refrun/adapter.py), the two model adapters
and [tiny.py](../tools/refrun/tiny.py). An adapter module defines:

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

Follow [Qwen's tiny test](../models/qwen38/test_reference.py) and
[MiMo's tiny test](../models/mimo26/test_reference.py): build a small
synthetic checkpoint, compare adapter scores with a direct upstream forward
bitwise, cover every stored tensor, verify cache replay without loading, and
include negative controls for layout corrections. These CPU tests prove
adapter mechanics, not full-model GPU parity.

The runner places a whole model on the selected GPUs; it has no layer-major
streaming for models that do not fit.
