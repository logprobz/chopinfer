# Deploy GLM 5.3 Flash

**Served and released.** The `chopinfer-model-glm5next` helper emits the
package of the FP8 checkpoint and of both NVFP4 exports (text, and images by
default), and `serve` runs it on four GPUs. Six of its seven profiles are
qualified through 32 sequences. The release archive and the image built
from it ship the helper and the bundle library, with the container image
default and the systemd unit below.

**Release status.** The final release gate's results were adopted on
2026-10-11 (see the [model page](model-glm5next.md#final-release-gate-2026-10-11)).
Two soaks failed the plain memory rule, the default configuration's and the
local-inference-lab NVFP4 export's; they pass the soak's declared warm-up
and cache allowance.
The full requalification through the release preview (stage 5,
2026-10-14 to 2026-10-15) qualified it, with accepted rows
in three configurations
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
| `zai-org/GLM-5.3-Flash` (default) | `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a` | 328.4 GB | `fp8-block128` | `CHOPINFER_GLM5NEXT_CHECKPOINT` |
| `local-inference-lab/GLM-5.3-Flash-NVFP4` | `175ae8ce3b5af842b0d0140dbeb43e9cfc557c49` | 198.1 GB | `nvfp4-routed` | `CHOPINFER_GLM5NEXT_NVFP4_CHECKPOINT` |
| `nvidia/GLM-5.3-Flash-NVFP4` | `da920bb0b9f4a06727223a349e55468e38352348` | 204.5 GB | `nvfp4-dense-routed` | `CHOPINFER_GLM5NEXT_NVIDIA_NVFP4_CHECKPOINT` |

Sizes are the sums of `models/glm5next/checkpoint-files.json`,
`checkpoint-files-nvfp4.json` and `checkpoint-files-nvidia-nvfp4.json`,
which pin every file's length and SHA-256.
Download a checkpoint at its revision into a plain directory with
`hf download --local-dir`, as in the [quickstart](developer-preview.md). The
layouts differ in what is quantized
([model README](../models/glm5next/README.md)): `nvfp4-routed` stores NVFP4
routed experts, MXFP8 MTP experts and BF16 elsewhere; `nvfp4-dense-routed`
stores NVFP4 dense MLPs and routed experts, BF16 MTP experts and BF16
elsewhere.

The test variables point CPU tests and reference runs at a checkpoint;
serving takes the checkpoint directory itself (`serve --model DIR`). All
three checkpoints serve the same package, `glm5next.v1`, at the same
topology, `glm5next-tp4`. The local-inference-lab export shares the FP8
checkpoint's tokenizer, chat template, processor and generation config byte
for byte. NVIDIA's shares the tokenizer and processor configuration, but its
chat template is an older revision, and its generation config, although
not byte-identical, states the same temperature, top-p and EOS ids.

## Serve

Once a package is emitted:

```sh
chopinfer serve --model /path/to/GLM-5.3-Flash --tensor-parallel-size 4
```

prepares the package from the checkpoint and serves it with the profile the
preparation provider selects for its layout: `tp4-mtp` for the FP8
checkpoint, `tp4-mtp-nvfp4` for the local-inference-lab export and
`tp4-mtp-nvfp4-nvidia` for NVIDIA's. Explicit options override the profile.
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
so for an NVFP4 export set `CHOPINFER_QUAL_CONFIG` to that configuration's
declaration, as `chopinfer eval declaration --model models/glm5next
--variant nvfp4` (or `nvfp4-nvidia`) prints it.

## Profiles

`models/glm5next/profiles.json` declares seven TP4
[serving profiles](serving-profiles.md), each with 32 sequences, 8,192
batch tokens, 2,048-token prefill chunks, KV fraction 0.9, KV dtype `auto`
and the prefix cache on. Six are qualified through 32 sequences on the
adopted final release gate (2026-10-11); the soaks of `tp4-mtp` and
`tp4-mtp-nvfp4` pass under the declared warm-up rule
([model README](../models/glm5next/README.md)).
`tp4-mtp-k5` stays experimental.

| Profile | Checkpoint | Speculation | Context-parallel size | Status |
|---|---|---|---:|---|
| `tp4-mtp` (default) | FP8 | MTP, 3 draft tokens, adaptive | 1 | qualified through 32 |
| `tp4-ar` | FP8 | none | 1 | qualified through 32 |
| `tp4-mtp-dcp2` | FP8 | MTP, 3 draft tokens, adaptive | 2 | qualified through 32 |
| `tp4-mtp-dcp4` | FP8 | MTP, 3 draft tokens, adaptive | 4 | qualified through 32 |
| `tp4-mtp-k5` | FP8 | MTP, 5 draft tokens, adaptive (benchmark only) | 1 | experimental |
| `tp4-mtp-nvfp4` | local-inference-lab NVFP4 | MTP, 3 draft tokens, adaptive | 1 | qualified through 32 |
| `tp4-mtp-nvfp4-nvidia` | NVIDIA NVFP4 | MTP, 3 draft tokens, adaptive | 1 | qualified through 32 |

`serve --model` never picks the autoregressive or context-parallel profiles
by itself: pass `--speculative-tokens 0` for AR, `--speculative-tokens 5`
for five drafts (the package declares five; `tp4-mtp-k5` is that arm) and
`--context-parallel-size 2` or `4` for DCP, or name the profile with `--profile FILE#NAME` on an
explicit package. A server started with explicit options keeps the name
of the profile the provider selected (`tp4-mtp` for the FP8 checkpoint)
and reports those values in its served identity. Each profile
pins its checkpoint layout (below), so a profile of another layout is
refused. The DCP profiles are qualified on their own declarations
([context-parallel attention](context-parallel.md)); five drafts are served
on the text package only (`--speculative-tokens 5 --modalities text`).

## Model options

The package declares three `--model-option NAME=VALUE` options (list them with
`--list-model-options`):

| Option | Default | Values |
|---|---|---|
| `checkpoint.quantization_layout` | the package's layout | that layout only: `fp8-block128`, `nvfp4-routed` or `nvfp4-dense-routed` |
| `runtime.sinkhorn_comm_epilogue` | `false` | `true` or `false` |
| `text_config.index_topk` | the checkpoint's 2,048 | a multiple of 4 from 4 to 2,048 |

`checkpoint.quantization_layout` records the layout the package was prepared
for. Every profile sets it, and `--profile` checks the profile's value
against the package before any explicit option applies; an explicit value
other than the package's fails the manifest check at startup. Create reads it
back. `text_config.index_topk` is the indexer's key budget; a value other
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
(the default) and `bf16` run the FP8 checkpoint's block-FP8 weights and the
NVFP4 exports' NVFP4 weights (W4A16) under BF16 activations. `checkpoint` is
refused for all three, since each declares quantized activations no GLM
engine implements: dynamic FP8 for the FP8 checkpoint and static NVFP4
inputs for both NVFP4 exports
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
| local-inference-lab NVFP4 | FP8 | 3 | text | 4,782,656 | 9,452,288 | 18,716,928 |
| local-inference-lab NVFP4 | FP8 | 3 | images | 4,332,608 | 8,552,192 | 16,916,736 |
| NVIDIA NVFP4 | FP8 | 3 | text | 4,580,672 | 9,048,320 | 17,908,992 |
| NVIDIA NVFP4 | FP8 | 3 | images | 4,130,624 | 8,148,224 | 16,108,800 |

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
drafts do not fit; with either NVFP4 export 64 slots with three drafts fit
(4,034,176 and 3,832,192 tokens at DCP 1 for text, 3,584,128 and 3,382,208
with images), though both NVFP4 profiles declare 32. NVIDIA's export ships an FP8 KV hint
(`kv_cache_scheme`), which is ignored: the latent KV format is
`--kv-cache-dtype`.
Print the whole plan with

```sh
cargo run -p model-glm5next-contract --example capacity-table --locked --offline
```
