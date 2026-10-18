# Deploy GLM 5.3 Flash

**Served and released.** The `chopinfer-model-glm5next` helper emits the
package of a checkpoint of any admitted layout (text, and images by
default), and `serve` runs it on four GPUs. Four of its five profiles are
qualified through 32 sequences on the official checkpoint. The release
archive and the image built from it ship the helper and the bundle library,
with the container image default and the systemd unit below.

**Release status.** The final release gate's results were adopted on
2026-10-11 (see the [model page](model-glm5next.md#final-release-gate-2026-10-11)).
The default configuration's soak failed the plain memory rule; it passes
the soak's declared warm-up and cache allowance.
The full requalification through the release preview (stage 5,
2026-10-14 to 2026-10-15) qualified it, with accepted rows
in two configurations (`taps` and `dcp4`)
([release qualification](model-glm5next.md#release-qualification-stage-5);
the paired-teacher policies are pinned to accepted arms of the final
gate's install), and `models/glm5next/model.toml` declares
`[release] enabled = true` with its release targets (the helper and the
bundle library) and kernel lock directory. This page records, from the code,
how to serve it; [common deployment operations](deploy-common.md)
covers prerequisites, the image, health, drain and rollback for every model.
Status and evidence are on the [model page](model-glm5next.md), shapes on
the [topology page](topology-glm5next.md).

## Checkpoints

| Checkpoint | Revision | Size | Layout | Test variable |
|---|---|---:|---|---|
| `zai-org/GLM-5.3-Flash` | `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a` | 328.4 GB | `attn-fp8.dense-fp8.experts-fp8.mtp-fp8.conv-bf16.hc-f32` | `CHOPINFER_GLM5NEXT_CHECKPOINT` |

The size is the sum of `models/glm5next/checkpoint-files.json`, which pins
every file's length and SHA-256.
Download a checkpoint at its revision into a plain directory with
`hf download --local-dir`, as in the [quickstart](developer-preview.md).

Any other checkpoint of the flash geometry is admitted by its layout, the
stored encoding of each projection class
([checkpoint layouts](../models/glm5next/README.md#checkpoint-layouts)):

- latent attention and shared experts: block-128 FP8 or BF16;
- dense MLPs: block-128 FP8, BF16 or NVFP4;
- routed experts: block-128 FP8 or NVFP4;
- MTP routed experts: block-128 FP8, MXFP8 or BF16;
- KDA convolution weights and mHC base and scale vectors: BF16 or F32.

Support follows the projection declaration (`contract/projections.json`)
and the kernels the bundle supplies for it. Admission is not
qualification: `qual.json` qualifies the official checkpoint, and a
checkpoint of another layout is qualified as an
[external variant](../tools/qual/README.md#external-variants).

The test variable points CPU tests and reference runs at the official
checkpoint, and `CHOPINFER_GLM5NEXT_EXTRA_CHECKPOINTS` at further
checkpoint directories of any layout (`:` separated), each checked against
expectations derived from its own layout. Serving takes the checkpoint
directory itself (`serve --model DIR`). Every layout serves the same
package, `glm5next.v1`, at the same topology, `glm5next-tp4`. A checkpoint
whose tokenizer files are not the pinned revision's is packaged when its
`tokenizer_config.json` and chat template state the pinned revision's,
under a frontend whose tokenizer repository is `local` at a content
revision; the served model name stays `GLM-5.3-Flash`.

## Serve

Once a package is emitted:

```sh
chopinfer serve --model /path/to/GLM-5.3-Flash --tensor-parallel-size 4
```

prepares the package from the checkpoint and serves it with the profile the
preparation provider selects, `tp4-mtp`, for every layout. Explicit options
override the profile.
TP4 needs exactly four distinct visible devices
([device visibility](deploy-common.md#device-visibility-and-rank-selection));
preparation refuses any other rank count.

## Run as a service

`deploy/systemd/chopinfer-glm5next.service` pins
`CHOPINFER_EXPECTED_BUNDLE_ID=glm5next.v1` and reads
`/etc/chopinfer/glm5next.env`; start from
`deploy/systemd/glm5next.env.example`, which names the checkpoint directory
and the model cache and serves on all four visible GPUs. GLM's deployment
policy (`models/glm5next/deploy/policy.json`, carried in its package) admits
the `CHOPINFER_GLM5NEXT_*` switches and the MTP drafter. Wait for readiness,
then check the served identity:

```sh
URL=http://127.0.0.1:8000
timeout 600s curl --retry 600 --retry-connrefused --retry-all-errors --retry-delay 1 \
  --max-time 3 -fsS "$URL/health"
curl -fsS "$URL/v1/models"
```

A successful `/health` means construction and warmup succeeded, not that
the release is qualified.

## GSM8K screen

After a deploy, screen the server on the 200 frozen GSM8K items at the
floor adopted for GLM on 2026-10-11, 193 of 200
([GSM8K floor](model-glm5next.md#gsm8k-floor)), which each declaration's
`preview.gsm8k_floor` states. From a checkout:

```sh
CHOPINFER_QUAL_CONFIG=models/glm5next/qual.json \
  python3 -B tools/qual/gsm8k.py --url http://127.0.0.1:8101 --model GLM-5.3-Flash
```

The screen refuses a server whose checkpoint is not the declaration's pin,
so for another checkpoint set `CHOPINFER_QUAL_CONFIG` to its external
variant's declaration, as `chopinfer eval declaration --model
models/glm5next --variant V --variant-dir DIR` prints it
([external variants](../tools/qual/README.md#external-variants)).

## Profiles

`models/glm5next/profiles.json` declares five TP4
[serving profiles](serving-profiles.md), each with 32 sequences, 8,192
batch tokens, 2,048-token prefill chunks, KV fraction 0.9, KV dtype `auto`
and the prefix cache on. Each serves every checkpoint layout. Four are
qualified through 32 sequences on the official checkpoint by the adopted
final release gate (2026-10-11); the soak of `tp4-mtp` passes under the
declared warm-up rule ([model README](../models/glm5next/README.md)).
`tp4-mtp-k5` stays experimental.

| Profile | Gated on | Speculation | Context-parallel size | Status |
|---|---|---|---:|---|
| `tp4-mtp` (default) | FP8 | MTP, 3 draft tokens, adaptive | 1 | qualified through 32 |
| `tp4-ar` | FP8 | none | 1 | qualified through 32 |
| `tp4-mtp-dcp2` | FP8 | MTP, 3 draft tokens, adaptive | 2 | qualified through 32 |
| `tp4-mtp-dcp4` | FP8 | MTP, 3 draft tokens, adaptive | 4 | qualified through 32 |
| `tp4-mtp-k5` | FP8 | MTP, 5 draft tokens, adaptive (benchmark only) | 1 | experimental |

`serve --model` never picks the autoregressive or context-parallel profiles
by itself: pass `--speculative-tokens 0` for AR, `--speculative-tokens 5`
for five drafts (the package declares five; `tp4-mtp-k5` is that arm) and
`--context-parallel-size 2` or `4` for DCP, or name the profile with `--profile FILE#NAME` on an
explicit package. A server started with explicit options keeps the name
of the profile the provider selected (`tp4-mtp`) and reports those values
in its served identity. No profile sets the checkpoint layout (below), so
each serves a package of any layout. The DCP profiles are qualified on their own declarations
([context-parallel attention](context-parallel.md)); five drafts are served
on the text package only (`--speculative-tokens 5 --modalities text`).

## Model options

The package declares three `--model-option NAME=VALUE` options (list them with
`--list-model-options`):

| Option | Default | Values |
|---|---|---|
| `checkpoint.quantization_layout` | the package's layout | that layout only, for example `attn-fp8.dense-fp8.experts-fp8.mtp-fp8.conv-bf16.hc-f32` |
| `runtime.sinkhorn_comm_epilogue` | `false` | `true` or `false` |
| `text_config.index_topk` | the checkpoint's 2,048 | a multiple of 4 from 4 to 2,048 |

`checkpoint.quantization_layout` records the layout the package was prepared
for, as its id `attn-<e>.dense-<e>.experts-<e>.mtp-<e>.conv-<d>.hc-<d>`.
The manifest declares it as the option's only allowed value, so an explicit
value other than the package's fails the manifest check at startup; the
shipped profiles do not set it. Create reads it back. `text_config.index_topk` is the indexer's key budget; a value other
than the checkpoint's is unqualified
([serving profiles](serving-profiles.md#model-options)).
`runtime.sinkhorn_comm_epilogue=true` changes the target model's sublayer
boundaries that a next sublayer follows (89 of 90; the last boundary and the
drafter are unchanged): after the unchanged all-reduce (the one-shot or the
phased reduce-scatter and all-gather), one launch split over 16 CTAs per row
replaces the BF16 round, the Sinkhorn expand, the next mix-collapse and the
next input norm. On four GPUs it gave the same bits as the default path and
was not faster, so it stays off; a server with it on reports unqualified
model options.

## Serving switches

The bundle reads four `CHOPINFER_GLM5NEXT_*` environment variables once,
when serve creates the instance. They are not model options, and the
serving profiles set none of them; the image's deployment policy admits
them. None changes the served bits: the shared overlap was served bitwise
equal to the default when it was introduced, and the diagnostics change only
scheduling or when a graph is captured. A value the table does
not list keeps the default.

| Variable | Default | Values | Use |
|---|---|---|---|
| `CHOPINFER_GLM5NEXT_SHARED_OVERLAP` | every graph bucket | `0` turns it off; a positive N limits it to decode and verify graph buckets of at most N rows | Production speed: decode, verify and drafter graphs run each MoE sublayer's shared expert on a second stream beside the routed experts. `0` is the off-ramp. |
| `CHOPINFER_GLM5NEXT_SYNC_STAGES` | off | `1` | Bring-up diagnostic: every rank synchronizes and logs after each eager segment and graph replay, so a hang names its stage. Prefills run eagerly. It serializes the ranks. |
| `CHOPINFER_GLM5NEXT_PHASED_RESET_RUNS` | 2^20 runs | 1 to 2^20 - 1 | Diagnostic: zeroes every rank's phased all-reduce state while idle every N runs, to exercise the reset in a short run. |
| `CHOPINFER_GLM5NEXT_LAST_CHUNK_GRAPHS` | off | `1` | Internal diagnostic: create captures the programs of all 64 last-prompt-chunk lengths instead of each at its first run. Skipped at DCP 2 and 4 and under `SYNC_STAGES`. Holding all 64 made AR decode steps up to about 1% slower. |

`CHOPINFER_GLM5NEXT_COMM_EPILOGUE` is retired: the model option
`runtime.sinkhorn_comm_epilogue` replaced it, and setting it does nothing.
The row-parts prefill and its `PREFILL_OVERLAP`, `OVERLAP_COMM_CTAS` and
`PREFILL_PARTS` switches were removed; setting them does nothing.
The [model README](../models/glm5next/README.md) describes each switch's
mechanism and measurements, and the variables that only tests and tools
read.

## Activation policy

Every projection runs BF16 activations (A16). `--activation-policy auto`
(the default) and `bf16` run every layout's weights under BF16 activations
(W4A16 for NVFP4 weights). `checkpoint` is refused for a checkpoint that
declares quantized activations, which no GLM engine implements; the
official checkpoint declares dynamic FP8
([quantized checkpoints](quantized-checkpoints.md)).

## Modalities

The frontend declares one image modality (patch 14, 16 to 8,000 tokens per
image, at most 64 images per request) and no video, so the default `auto`
modalities are text and image: the package plans the encoder image and
requires media inputs, and each rank also holds the encoder's image and
buffers (3.31 GiB), which come out of the KV pool
([image cost](model-glm5next.md#images)). Use `--modalities text` to serve
without the encoder. A video part is refused with a message naming it;
video is out of scope.

Sampling defaults copy the checkpoint's explicit `generation_config.json`
fields: temperature 1.0, top-p 0.95 and its three EOS ids as stop tokens;
top-k stays off and reasoning defaults to on at effort `max`
([deploy defaults](model-glm5next.md#serving-profiles)).

## KV formats and planned capacity

Each 64-token page holds, per latent sparse layer, the KV latent rows and the
indexer's pooled keys; the `-mtp` layouts add the MTP layer as a twelfth.

| `--kv-cache-dtype` | Layout | Page-layer bytes | Page bytes (11 layers) |
|---|---|---:|---:|
| `auto` or `fp8` (default) | `glm5next.latent-paged-fp8.v1`: FP8 E4M3 rows with per-128 F32 scales | 37,888 | 416,768 |
| `bf16` | `glm5next.latent-paged-bf16.v1` | 69,632 | 765,952 |

FP8 latent KV is the package's preference, so `auto` selects it; BF16 is the
comparison format. KV tokens per pool, from the contract's capacity plan for
32 slots, of a text package (`--modalities text`) and of a package serving
images (`auto` or `text,image`), which also holds the encoder's image (1.05
GiB) and buffers (2.26 GiB) on each rank:

| Checkpoint | KV | MTP drafts | Package | DCP 1 | DCP 2 | DCP 4 |
|---|---|---:|---|---:|---:|---:|
| FP8 | FP8 | 3 | text | 665,792 | 1,218,560 | 2,249,472 |
| FP8 | FP8 | 3 | images | 215,744 | 318,464 | 449,280 |
| FP8 | FP8 | 0 | text | 1,648,064 | 3,172,864 | 6,140,928 |
| FP8 | FP8 | 0 | images | 1,157,120 | 2,190,976 | 4,177,152 |
| FP8 | FP8 | 5 | text | 366,400 | 619,776 | 1,051,904 |
| FP8 | FP8 | 5 | images | does not fit | does not fit | does not fit |
| FP8 | BF16 | 3 | text | 362,240 | 663,040 | 1,223,936 |
| FP8 | BF16 | 3 | images | 117,376 | 173,312 | 244,480 |

The default package serves images, so the default `tp4-mtp` profile (FP8
checkpoint, FP8 KV, three drafts, 32 slots) plans 215,744 KV tokens at DCP
1, 318,464 at DCP 2 and 449,280 at DCP 4; with `--modalities text` it plans
665,792, 1,218,560 and 2,249,472. The difference is the encoder's 3.31 GiB
per rank (its image and buffers), which create allocates before the host
sizes the KV pool.
`tp4-mtp-k5` (five drafts) does not fit 32 slots in a package serving
images; serve it with `--modalities text`.

**These are planned capacity, not measured**; the pools serving measured
are on the [model page](model-glm5next.md#context-parallelism). The plan
(`model_glm5next_contract::capacity`) assumes 94 GiB usable per GPU (the
board reports 95.59 GiB), then subtracts the host's 1 GiB reserve, the weights (the text
rank image holds no vision tensor), the encoder's 3.31 GiB with images, 5.46 GiB of
workspaces (the named engine buffers plus a 2 GiB allowance for buffers the
bundle has not listed yet), the context-parallel buffers (0.42 GiB at DCP
2, 0.76 GiB at DCP 4) and the request state, and takes 0.9 of the rest for
pages. The plan counts the state as create allocates it before the KV
pool, which the host then reserves nothing for: one copy of the KDA state
and indexer tail per slot without drafts, `rows + 1` with them, and sixteen
copies for the anchor spares (0.55 GiB); the indexer tail is in fact a
fixed working tail plus six checkpoints, so the state term is an
approximation. The totals are shared by every slot; one sequence is
limited to the model's 1,048,576-token context. With FP8 weights, 64 slots
fit without drafts but not with three, and with images 32 slots with five
drafts do not fit. A layout with NVFP4 weights holds less per rank and
plans a larger pool; the plan is computed per layout. A ModelOpt
checkpoint's static FP8 KV hint (`kv_cache_scheme`) imports nothing: the
latent KV format is `--kv-cache-dtype`.
Print the whole plan for the representative layouts, or for the layout ids
given after `--`, with

```sh
cargo run -p model-glm5next-contract --example capacity-table --locked --offline
```
