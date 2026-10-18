# Serving MiMo V2.6 Flash

Dimensions come from the checkpoint's `config.json`, matched to a compiled
entry in `models/mimo26/geometries/supported.json`. Numeric shapes and memory
figures below describe the shipped geometry. Other admitted geometries derive
their own per-rank shapes, catalogs and resources. Topology IDs follow
`<model>-<geometry>-tp<n>`; see the
[same-architecture recipe](adding-a-model.md#same-architecture-checkpoints).

This guide serves MiMo V2.6 Flash with `chopinfer` on one Linux host with four
RTX PRO 6000 Blackwell GPUs. Run commands from the workspace root. The
[quickstart](developer-preview.md) covers installation, checkpoint download and
a first request; [direct loading](direct-loading.md) explains how the weights
are planned and streamed from the checkpoint.

## MOPD alpha checkpoint

The default checkpoint is `XiaomiMiMo/MiMo-V2.6-Flash-MOPD`, revision
`2479e2d0029eca9a34cc7e7f55a121925f81908e`. As of `v0.1.0-alpha.2`
(2026-10-03), validation passed at TP4 (DFlash and AR) and TP2 (AR); see the
[validation summary](../README.md#what-it-runs) and
[measured speed](serving-profiles.md#measured-speed). The
[correctness runner](../tools/qual/README.md#correctness-runner) gates MOPD's
teacher, exact-verify and multimodal checks against GPU reference runner F32
references of the MOPD checkpoint and requires GSM8K of at least 193/200 in
each arm. In 34 stability runs MOPD
scored 193 or 194; the extra misses were near-tie flips (items 85 and 87).

The family helper serves either pinned checkpoint (MOPD or RL) whose config,
index and tokenizer files match `models/mimo26/checkpoint-files.json` under
that checkpoint's name. It serves any other MiMo V2.6 Flash checkpoint in the
encodings MiMo serves under a local identity, `local-<digest>`, with BF16 KV
only ([MiMo checkpoints](compatible-checkpoints.md#mimo-v26-flash-checkpoints)).
The DFlash drafter is
`dflash/dflash_draft_model.safetensors` inside the checkpoint. Serving checks
safetensors headers and lengths; only `chopinfer verify --model` checks tensor
payloads (see
[what verify adds](compatible-checkpoints.md#what-serving-checks-and-what-verify-adds)).

## Start the server

Prerequisites are those of the [quickstart](developer-preview.md). Keep the
checkpoint at `${CHOPINFER_MIMO_CHECKPOINT}` on fast local storage: every
start reads the weights from it.

From the source tree, `tools/quickstart.sh` builds and installs the host,
`chopinfer-model-mimo26` and `libbundle_mimo26.so` into a private directory,
optionally verifies the checkpoint, and serves the chosen profile:

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 tools/quickstart.sh mimo26 tp4-dflash "$CHOPINFER_DATA" \
  --checkpoint "$CHOPINFER_MIMO_CHECKPOINT" --verify --listen 127.0.0.1:8311
```

With an existing install `$INSTALL`:

```sh
"$INSTALL/bin/chopinfer" verify --model "$CHOPINFER_MIMO_CHECKPOINT" \
  --manifest models/mimo26/checkpoint-files.json#mopd --allow-unrecorded
"$INSTALL/bin/chopinfer" serve --model "$CHOPINFER_MIMO_CHECKPOINT" \
  --model-cache-dir "$CHOPINFER_DATA/model-cache" --listen 127.0.0.1:8311
```

At TP4 the helper picks DFlash when the checkpoint has `dflash/`, else MTP
when it has `model_mtp.safetensors`, else no drafter. The default package plans
the vision tower for text and image requests; `--modalities all` also serves
video and audio, at some KV capacity
([capacity](serving-profiles.md#served-modalities-and-memory)). Audio is
served but not qualified in-tree: the repository holds no audio reference
input, so test audio with your own inputs. The TP4 profiles use devices 0-3,
32 sequences (AR and DFlash are qualified through 64), with
`--max-batch-tokens 8192 --prefill-chunk-tokens 2048 --kv-memory-fraction 0.9`
(`models/mimo26/profiles.json`). See the
[C64 qualification](serving-profiles.md#tp4-concurrency-qualification-2026-10-04).
`--tensor-parallel-size 2` serves TP2 autoregressively;
[deployment](deploy-common.md) covers running MiMo and Qwen TP2 instances side
by side. For a container or systemd install, follow the
[deployment runbook](deploy-mimo26.md).

For a choice the helper does not make, emit an explicit package into a new
directory and serve it with the checkpoint and a profile:

```sh
cargo build -p model-mimo26 --release --locked --offline --bin mimo26-package
PACKAGE_DIR=${CHOPINFER_DATA}/packages/mimo26-tp4-mtp  # must not exist yet
target/release/mimo26-package --checkpoint "$CHOPINFER_MIMO_CHECKPOINT" \
  --module "$INSTALL/lib/libbundle_mimo26.so" --output "$PACKAGE_DIR" \
  --drafter mtp --encoders on
"$INSTALL/bin/chopinfer" serve --bundle "$PACKAGE_DIR" --checkpoint "$CHOPINFER_MIMO_CHECKPOINT" \
  --profile models/mimo26/profiles.json#tp4-mtp --listen 127.0.0.1:8311
```

Without `--profile`, an explicit package serves with host defaults and no
speculation.

Packages declare the most sequences whose KV holds an 8K average context: 256
at TP4, 64 at TP2 (32 with the encoders, 16 with DFlash and the
encoders), at most 32 with the MTP drafter ([device
memory](topology-mimo.md)). Per-sequence state is sized for the served count
rounded up to a power of two, so serving more sequences leaves fewer KV pages.
The KV pool takes the requested fraction of the tightest rank's free memory
after loading and a 1 GiB reserve, then any `--kv-pages` cap
([serving KV pool](serving-kv-pool.md)).

Diagnostic switches, read when the instance is created:
`CHOPINFER_MIMO_F32_LOGITS=0` uses the BF16 vocabulary head instead of F32;
`CHOPINFER_MIMO_ROW_OWNED_LOGITS=0` or `CHOPINFER_SAMPLING_DEFER_VIEWS=0`
restores the logits gather or serial sampler; `CHOPINFER_MIMO_WIDE_COMM=0`
uses the phased reduce stages for verify steps above 32 rows.

Decode all-reduces take the plain decode v2 entry, whose low-latency rows
sum the norm statistic asynchronously, and the decode combine entry. They
serve through 12 rows; the low-latency protocol serves only one and two
rows.

## Speculative decoding

| Drafter | Profile | Explicit package | Serve options | Status |
|---|---|---|---|---|
| DFlash block drafter | `tp4-dflash` | `mimo26-package --drafter dflash` | `--speculative-tokens 7 --speculation adaptive --draft-cost per-block` | Default drafter; serves 32 sequences by default, qualified through 64 (`--max-sequences 64`) with the speculative accuracy gate waived ([why](serving-profiles.md#accuracy-waiver-2026-10-06)) |
| MTP layer 0 | `tp4-mtp` | `mimo26-package --drafter mtp` | `--speculative-tokens 3 --speculation adaptive` | Supported alternative; gains less |
| None (AR) | `tp4-ar` | no `--drafter` | `--speculative-tokens 0` | Baseline |

`serve --model`, `tools/dev/mimo-redeploy.sh` and the release container serve
adaptive DFlash K7 from a checkpoint with `dflash/`. Pass
`--speculative-tokens 0` for AR. MOPD carries `dflash/`, so its MTP drafter
needs an explicit `--drafter mtp` package with the `tp4-mtp` profile
(`tools/dev/mimo-redeploy.sh --drafter mtp` does this).

Adaptive speculation estimates draft overhead from measured step time and
batch shape, and drafts only when it pays; DFlash uses `--draft-cost
per-block` because one forward pass produces the whole block. `--speculation
fixed` drafts the full depth on every step. A DFlash package lets one step
verify 256 rows; `--speculative-rows 32` keeps steps within the 32-row decode
batch.

The policy changes speed, not output: verify steps of up to eight rows (every
C1 step) use exact rows whose bits equal the one-row AR step, so C1 greedy
DFlash and MTP serve AR's tokens and logprobs
([exact verify rows](adaptive-speculation.md)).
`CHOPINFER_MIMO_VERIFY_EXACT=0` restores the verify kernels for A/B runs.

The DFlash drafter runs inside the target's rank program and is captured with
the decode and verify graphs. Per rank it adds 720 MiB of weights, a 320 MiB
context cache for 32 slots and 120 MiB of context-feature rows. MTP retains at
most eight target hidden rows per slot (2 MiB per rank) plus its weights and
scratch; `CHOPINFER_MIMO_MTP_EAGER=1` runs all MTP work eagerly for
comparison.

### Speculation at C16 and C32 against vLLM

Historical RL checkpoint comparison (2026-09-27): adaptive DFlash measured
0.4-22.7% higher throughput than the vLLM DFlash K7 reference at C1, C4,
C16 and C32
(64-output-token greedy chat and GSM8K cohorts, TP4, BF16 KV; chat C16's
margin was 0.8%). AR and adaptive both scored 195/200 on GSM8K. This is
workload-specific evidence, not production qualification.

### Known limits

- Speculation gains depend on the text. `bench.py`'s repetitive text accepts
  nearly every draft and overstates the gain; use `spec_bench.py` for
  realistic numbers.
- A cancelled prefill chunk longer than 1025 rows can lower DFlash acceptance
  (not correctness) until the overwritten positions leave the drafter window.
- Release qualification with the FP8 global KV layout has not been run with
  speculation.
- MiMo does not name the color of a uniform image; see
  [solid-color images](#solid-color-images).

## Prefix caching

Prefix reuse is on by default; `--prefix-cache off` disables it. A follow-up
chat turn hits at the end of the previous prompt, a repeated long prompt from
its second use, and a repeated system prompt from its third
([where anchors are captured](serving-kv-pool.md)). Requests that ask for
prompt logprobs never resume, and a request's `cache_salt` limits reuse to
requests with the same salt. A hit's tokens and logprobs are bitwise equal to a
cold prefill's ([numerics](numerics.md)); drafter acceptance after a hit can
differ. Anchors use only KV pages no running request needs, so admission
capacity does not change. Responses report
`usage.prompt_tokens_details.cached_tokens`, and `/metrics` exports
`chop_prefix_cache_*` and `vllm:prefix_cache_{queries,hits}_total`
([observability](operations-observability.md)).

## Development redeploy

`tools/dev/mimo-redeploy.sh --port 8311 --smoke` rebuilds from the source
tree, installs privately under `--work-dir` (default
`${CHOPINFER_DATA}/dev-redeploy`), restarts only a server it started for that
port, waits for health and sends a short greedy completion. It serves adaptive
DFlash at TP4 unless given `--drafter none|mtp` or `--ranks 2`.
`--prepare-only` needs no GPU and `--dry-run` prints the steps; `--help` lists
the rest. See [redeploy a dev server](development-loop.md#redeploy-a-dev-server).

## Registering a checkpoint

A checkpoint added through the
[shared recipe](adding-a-model.md#same-architecture-checkpoints) also needs:

- A geometry file stripped of export metadata such as `quantization_config`,
  so its execution-field leaves match
  `models/mimo26/geometries/v26-flash.json`. The `engine_domain` section of
  `models/mimo26/geometries/supported.json` records MiMo's current engine
  restrictions.
- For an official release, its identity in `models/mimo26/checkpoints.json`,
  which serves it under its own name; without one it is served under a local
  identity with BF16 KV only. To serve FP8 KV, add `fp8_kv_scales`: one `[K, V]` pair of finite positive f32 scales per
  global-attention layer, in layer order. Compute them with
  `kernel_core::kv_fp8::calibrated_scale` on 16 times the per-layer K/V amax
  from an f64 oracle over representative teacher texts. The shipped MOPD/RL
  table used 33 texts; its provenance is beside `LEGACY_SCALES` in
  [the calibration module](../models/mimo26/contract/src/kv_calibration.rs).
  Calibration defaults to `v26-flash`; set `fp8_kv_geometry` to the compiled
  geometry ID for another shape, then rebuild the compiler and bundle. Each
  distinct table gets a deterministic layout ID; the shipped table keeps
  `mimo.ga-paged-fp8.v2`. This step needs no Rust, CUDA or kernel lock
  edits. Preparation advertises only the authenticated identity's FP8 table,
  and BF16 alone when calibration is absent
  ([compatible checkpoints](compatible-checkpoints.md#mimo-v26-flash-checkpoints)).
  Registration admits metadata; it does not qualify new weights, calibration
  or geometry.

## Qualification and benchmarking

Plain decode v2 serving qualification on 2026-10-04 preserved C1 outputs
bit for bit and matched fixed-batch C16 outputs at TP4 bitwise. Real-host
create succeeded, with the LL and bulk plans requiring 2 of 188 and 3 of 46
resident clusters. Across two C1 runs, time per output token relative to the
replaced v1 entry was 0.987 and 0.991 at TP2, and 0.990 and 0.954 at TP4.
C16 and C32 timing differences were within measurement noise.

### Release gate

Release qualification uses the
[correctness runner](../tools/qual/README.md#correctness-runner)
(`tools/qual/release_preview.sh run --model mimo26 --suite full`), which
starts its own servers and records every input a check reads. Use
[the release gate](../tools/qual/README.md#release-gates) against an
already-running server: `python3 tools/qual/release_gate.py --model mimo26
--url URL --profile quick` is a bounded accuracy and serving screen, and
`--profile full --inputs INPUTS.json` runs the full cohort with the records
and earlier receipts its checks read, against a server started with
`--modalities all`: the media checks judge each modality the server serves
and report the others not applicable, which cannot qualify. Qwen uses
`--model qwen38`. The gate never starts or stops a server or touches a GPU.

Keep the run directory (`summary.json`, `summary.md`, child receipts and logs)
with the release. Qualification requires exit zero and `qualified: true`.
Quick and CPU mechanics results cannot qualify a release; missing inputs,
skipped coverage, failed thresholds and timeouts fail the gate. GPU memory
growth and performance are outside its coverage.

### Per-layer taps

To find the first layer where the engine leaves the reference, export
`CHOPINFER_TAPS_DIR=<dir>` before `tools/dev/mimo-redeploy.sh`, which then
builds the bundle with `--features bundle-mimo26/diagnostic-taps` (a serving
bundle refuses to start while the variable is set). It writes
`<dir>/step<NNNN>/` with the taps of `models/mimo26/taps.json` under the
reference runner's names (`L05.in`, `final.norm`, `enc.vision.out`, ...);
`CHOPINFER_TAPS_LAYERS` and the other `CHOPINFER_TAPS_*` variables work as for
[Qwen3.8](bundle-qwen38.md). Send one prompt with `max_tokens` 1, run the
reference runner with `--taps` on the same token IDs, and compare with
`tools/qual/tapdiff.py`. Taps run graphs stage by stage, so they cannot clear
graph-only failures.

### Vision merger normalization

The vision merger normalizes each row with an RMSNorm over
`visual.merger.ln_q.weight` (eps 1e-6), as vLLM and SGLang do; the vendor code
declares a LayerNorm with biases the checkpoint does not store. Paired F32
reference probes found LayerNorm no better overall and worse on OCR, so
RMSNorm stays.

### Solid-color images

MiMo does not name the color of a uniform image: it answers "white" for solid
colors, and reads a solid red frame as blue, under either merger norm. This is
model behavior, not preprocessing. Normalization is a declared deviation: the
served recipe keeps ImageNet mean and std, which a paired F32 study of 349
probes favored over the checkpoint's CLIP statistics.

### Image resize reference

Images use the checkpoint processor's bicubic resize with a minimum of 8,192
pixels. The served recipe keeps ImageNet mean and std on 0..255 values, where
the checkpoint names CLIP constants, and caps an image at 1,048,576 pixels;
the qualification declaration records the normalization, the minimum and the
cap as explicit deviations from the checkpoint processor. A paired F32 GPU
comparison (2026-10-02) found no significant difference between bicubic and
bilinear, and the 3,136-pixel checkpoint minimum lost accuracy on small
images, so the 8,192 minimum stays. Video keeps its 8,192-pixel minimum and
262,144-pixel per-frame cap, with bicubic resize and ImageNet normalization
as a serving choice (the checkpoint has no video processor configuration).
The image reference is transformers' AutoImageProcessor on the GPU
[reference runner](reference-runner.md).

### Production qualification status (2026-09-25)

Historical status: API, structured-output, multimodal and soak runs passed;
full production qualification remained open. See the
[alpha.2 status](#mopd-alpha-checkpoint) for the later preview validation.
FP8 KV with speculation remains unqualified. A bounded TP4 KV preemption
swap smoke on 2026-10-12 compared all 36 C1 requests bit for bit in plain
and prefix-pressure scenarios, with a 16 GiB host arena. The full suite's
[offload check](serving-kv-pool.md#formats-caching-and-offload) supplies the
configuration qualification; reusable host-prefix caching is deferred.

## Troubleshooting

- Startup writes phase timings to stderr (`startup weight_plan`, `startup
  direct_load`, bundle creation, KV pool, graph capture). A slow direct load
  usually means a cold page cache or slow checkpoint storage; see
  [direct loading](direct-loading.md) for `--load-threads`.
- If GPU allocation fails, check free memory on all four ranks; the tightest
  rank sizes the KV pool. Lower `--kv-memory-fraction`, cap `--kv-pages`, or
  reduce `--max-sequences`.
- If requests are refused for context or capacity, compare their token counts
  with the declared limit, the batch budget and the logged KV capacity.
- Otherwise, [report a problem](developer-preview.md#report-a-problem).
