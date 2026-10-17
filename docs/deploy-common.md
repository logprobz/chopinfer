# Common deployment operations

One image and entrypoint serve [MiMo V2.6 Flash](deploy-mimo26.md) or
[Qwen3.8 Flash](deploy-qwen38.md) from a mounted Hugging Face checkpoint. The
entrypoint prepares a small package without weights in a writable model cache
and runs `chopinfer serve --model`, which loads every weight from the
checkpoint ([direct loading](direct-loading.md)). The same entrypoint runs
under Docker or systemd.

For Docker and native first-response commands, use the
[quickstart](developer-preview.md). [Operator commands](cli.md) is the
reference for `serve --model`, `--modalities` and `verify`;
[serving profiles](serving-profiles.md) lists the profiles and their limits;
[the request surface](serving-request-surface.md) covers endpoints, request
parameters and queueing.

## Prerequisites

- Linux x86-64, one RTX PRO 6000 Blackwell GPU per rank (four by default) and
  a driver compatible with CUDA 13.2.
- Docker with the NVIDIA container toolkit and driver capabilities
  `compute,utility,video`: `libcuda.so.1` and NVDEC's `libnvcuvid.so.1` come
  from the host driver. Link-only CUDA stubs must never be on the serving
  library path; check with `ldd` inside the image.
- The native modules and model helpers run as trusted code. Keep their
  directories not group or world writable, and do not let the serving account
  modify the checkpoint ([security boundary](security-boundary.md)).
- Image builds install Rust 1.89.0 and CUDA 13.2.86 inside the builder; no
  host toolchain or GPU is needed. Never regenerate kernel locks to hide a mismatch.

## Build the image

From a clone of a published tag, with network access:

```sh
docker build -f deploy/docker/Dockerfile -t chopinfer:v0.1.0-alpha.3 .
docker run --rm chopinfer:v0.1.0-alpha.3 --version
```

The Dockerfile pins the CUDA devel and base images and the frontend by
digest. It installs the locked compiler and uses `tools/release/build.sh`.
Only the release archive and runtime tools enter the final image. The base
image sufficed for the TP2 GPU smoke; the NVIDIA runtime supplies driver
libraries. The default user is `chopinfer` (UID 10001).

Standalone clone builds validate identity against Git HEAD. For a Git URL
context, retain Git metadata so the build validates the fetched HEAD too:

```sh
TAG=v0.1.0-alpha.3
docker build -f deploy/docker/Dockerfile \
  --build-arg BUILDKIT_CONTEXT_KEEP_GIT_DIR=1 -t "chopinfer:$TAG" \
  "https://github.com/logprobz/chopinfer.git#$TAG"
```

Both paths derive the revision from the context's HEAD. A supplied
`CHOPINFER_BUILD_REVISION` must match it; missing HEAD is refused even with
that argument. Archive-only contexts and Git worktree contexts are unsupported. `--version` reads
the release-verified build identity without needing driver libraries. For
`verify` without GPU access, use `--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=none`.

`python3 deploy/docker/build.py` remains available for explicit digest
inputs from a standalone clone. For offline Cargo, pass
`--build-context cargo-cache=DIR --build-arg CARGO_OFFLINE=true`, using a
registry cache without credentials. `/usr/share/chopinfer/build-inputs/`
records compiler versions, the lock hash and source revision. Its `rustc.txt`
and `Cargo.lock.sha256` are authoritative; these are not duplicated as image labels.

## Checkpoint, model cache and package selection

The entrypoint reads the checkpoint at `CHOPINFER_MODEL` (default `/data/model`)
and writes packages to `CHOPINFER_MODEL_CACHE` (default `/data/model-cache`).

- Mount the checkpoint read-only. It must be a complete directory (config,
  tokenizer, index and every safetensors shard), for example from
  `hf download --local-dir`. For a hub cache snapshot, mount the whole
  repository cache, since snapshots link into its `blobs/`, and point
  `CHOPINFER_MODEL` at the snapshot. UID 10001 needs read access to every
  file and traverse access to directories. The read-only mount does not grant
  these permissions. For a private checkpoint, run as its owner with
  `--user "$(id -u):$(id -g)"` and use a bind-mounted model cache created by
  that user (`install -d -m 700 "$CACHE"`), replacing the named cache mount
  with `--mount type=bind,src="$CACHE",dst=/data/model-cache`. Keep the checkpoint mount read-only;
  do not grant write permissions to checkpoint files.
- Use a named volume for the model cache; it inherits UID 10001 ownership.
  A bind-mounted cache must be writable by the chosen `--user` and empty or an
  existing preparation cache. Each package is about 40 MB; keep the cache
  across restarts so later starts reuse it. Several servers can share one
  cache.
- Serving checks safetensors headers and lengths, not tensor bytes. Run
  `chopinfer verify --model DIR` once per checkpoint before relying on it, or
  set `CHOPINFER_VERIFY_MODEL=1` to verify at every start
  ([compatible checkpoints](compatible-checkpoints.md)).
  `CHOPINFER_VERIFY_MANIFEST=FILE#ENTRY` names a file of recorded checksums
  instead; every file in the directory must then be recorded.

The entrypoint runs each installed helper
(`chopinfer-model-<model> prepare --direct`) until one prepares the
checkpoint at the configured rank count, and fails with every helper's
refusal otherwise. Native `serve --model` takes its profile from the helper
that prepares the package: the helper selects it for the checkpoint and rank
count from the copy of the model's `profiles.json` it embeds. The image passes
only explicit serving overrides.
Each installed helper reports its model's deployment policy (`chopinfer-model-<model>
deployment-policy`), so every installed model's variables are validated
before preparation, and the prepared package carries the policy of the bundle
it serves: its drafters and its variables. The image owns no per-model
file. Set
`CHOPINFER_EXPECTED_BUNDLE_ID` (the model-specific units and examples do) to
refuse a checkpoint that prepares a different model. Record the checkpoint
revision and the startup `direct_package` line with each release.

## Environment and arguments

| Variable | Default | Meaning |
|---|---|---|
| `CHOPINFER_MODEL`, `CHOPINFER_MODEL_CACHE` | `/data/model`, `/data/model-cache` | Checkpoint and package cache |
| `CHOPINFER_TENSOR_PARALLEL_SIZE` | 4 | Rank count: 2 or 4 |
| `CHOPINFER_LISTEN` | `0.0.0.0:8000` | Listener; bracket IPv6, for example `[::]:9000` |
| `CHOPINFER_MODALITIES` | `auto` (text and image) | Served modalities: `auto`, `all` or a list; `--modalities` overrides it ([CLI](cli.md)) |
| `CHOPINFER_MAX_SEQUENCES`, `CHOPINFER_MAX_BATCH_TOKENS`, `CHOPINFER_PREFILL_CHUNK_TOKENS` | 32, 8192, 2048 | Scheduler sizes |
| `CHOPINFER_MAX_CONTEXT_TOKENS` | unset | Refuse startup unless KV holds one request of this length |
| `CHOPINFER_QUEUE_DEPTH` | unset (unbounded) | Opt-in waiting queue bound |
| `CHOPINFER_KV_MEMORY_FRACTION` | native profile (currently 0.9) | Fraction of free device memory for KV |
| `CHOPINFER_KV_CACHE_DTYPE` | `auto` (BF16) | `auto`, `bf16` or `fp8` ([KV pool](serving-kv-pool.md)) |
| `CHOPINFER_SPECULATIVE_TOKENS`, `CHOPINFER_SPECULATION`, `CHOPINFER_DRAFT_COST` | native profile | `CHOPINFER_SPECULATIVE_TOKENS=0` serves autoregressively |
| `CHOPINFER_DRAIN_TIMEOUT_MS` | 30000 | Drain budget on SIGTERM |
| `CHOPINFER_VERIFY_MODEL`, `CHOPINFER_VERIFY_MANIFEST` | unset | Checkpoint content check before serving |
| `CHOPINFER_API_KEY` | unset | Bearer key for API paths |
| `CHOPINFER_SERVE_ARGS` | empty | systemd only: extra `serve` arguments |

Unknown or empty `CHOPINFER_*` variables fail before preparation, and so do the
variables of the old package deployment (`CHOPINFER_PACKAGE`,
`CHOPINFER_INSTALL_ROOT`, `CHOPINFER_MODULE` and the verify cache variables);
the error names the replacement. Model-specific variables are accepted only
for their model, drafter settings only with a matching drafter, and
speculation policy only when speculation is on. Do not pass a development
`local.env` to the serving process. `CHOPINFER_PREFILL_CHUNK_TOKENS` must be a
multiple of the package's `batch_limits.prefill_granule` (64 for Qwen3.8).

Arguments after the image name override environment values and native serving
defaults, except that Docker listener changes must use `CHOPINFER_LISTEN`.
Do not pass `--listen` after the image: the health check reads the environment
and cannot discover that override. Publish the same container port, for example
`-e CHOPINFER_LISTEN=0.0.0.0:9000 -p 127.0.0.1:8600:9000`. Use `CHOPINFER_TENSOR_PARALLEL_SIZE` for the rank count, not a second
`--tensor-parallel-size` or `--topology`. `--profile` belongs to explicit
packages, so use `CHOPINFER_SPECULATIVE_TOKENS=0` for AR through this entrypoint.
`verify`, `--help` and `--version` run the CLI without preparing a serving
package. Under systemd, `CHOPINFER_SERVE_ARGS` uses systemd argument splitting.

The container listener defaults to `0.0.0.0:8000` so published ports work;
keep the host port on loopback. This is the image-specific serving default.
The TP2 smoke needed neither `--ipc=host`, `--shm-size` nor a `nofile` override.
Keep the pinned-memory limit described below.

## Package topology and separate instances

A model has one bundle ID (`mimo.v26.flash.v2`, `qwen38.flash.v2`), and one
package serves every rank count it supports. Its manifest declares one
topology per rank count, each with its own minimum device memory, KV and state
layouts, sequence maximum and graph buckets. `CHOPINFER_TENSOR_PARALLEL_SIZE`
selects the topology and the matching serving profile. Per-rank shapes and
memory are in [MiMo topology](topology-mimo.md) and
[Qwen topology](topology-qwen.md).

Two TP2 servers can share a four-GPU host. For systemd, install
`deploy/systemd/chopinfer@.service` into `/etc/systemd/system/` and copy
`mimo26-tp2.env.example` and `qwen38-tp2.env.example` from `deploy/systemd/`
to `/etc/chopinfer/mimo26-tp2.env` and `/etc/chopinfer/qwen38-tp2.env`. They
select `CUDA_VISIBLE_DEVICES=0,1` and `2,3`, ports 8311 and 8431, and
`CHOPINFER_TENSOR_PARALLEL_SIZE=2`. Edit the checkpoint paths, create
`/var/lib/chopinfer/model-cache` owned by `chopinfer`, reload systemd and
start `chopinfer@mimo26-tp2` and `chopinfer@qwen38-tp2`. For Docker, give each
container its own `--gpus` set; each container sees its GPUs as ordinals 0
and 1. Do not run containers and systemd units on the same devices. Check
`/health` and `/v1/models` on both ports before traffic.

To watch the instances live, run `chopinfer-top`, which ships in the same
release and image. `docker exec -it CONTAINER chopinfer-top` works inside the
image: it finds the container's serve process. From the host it finds the
containers whose process list shows serve when the user can open the Docker
socket, and reaches them through the published port. The systemd units run as
the `chopinfer` user, so pass `--target 127.0.0.1:8311 --target
127.0.0.1:8431`, or run it as root. See
[watch running instances with chopinfer-top](operations-observability.md#watch-running-instances-with-chopinfer-top).

## Device visibility and rank selection

Set `CUDA_VISIBLE_DEVICES` (systemd) or `--gpus` / `NVIDIA_VISIBLE_DEVICES`
(Docker) to the GPUs assigned to the server. CUDA renumbers the visible set
from zero; ordinals are not `nvidia-smi` indices. Without `--devices`, `serve`
uses every visible device and refuses startup when the count differs from the
rank count. `--devices 1,0` selects and orders visible ordinals, one distinct
ordinal per rank; duplicates are refused. It does not hide the other visible
GPUs.

Startup checks peer access between every selected pair before loading
weights, and each rank logs its visible ordinal, PCI bus ID and UUID. Video
decoding (NVDEC) uses rank 0's device. These checks do not prove the devices
are idle.

Rank folding and `--fold-ranks` have been removed.

## Pinned host memory

Qwen3.8 TP2 keeps its n-gram table in page-locked host memory
([host-placed views](direct-loading.md)): 51.2 GB in total, 25.6 GB per rank,
pinned until the server exits. A KV offload arena is also pinned for the
server's lifetime: `--kv-offload-gib 16`, the qualification reference size,
adds 16 GiB across the ranks ([KV pool](serving-kv-pool.md)). Count it with
the n-gram table and other pinned use before setting memlock and container
limits. Every server also pins two 64 MiB upload buffers per rank while
loading weights. A model with host-mapped views also pins a 64 MiB gather
ring per stream that gathers rows; the mapped checkpoint bytes themselves are
not pinned and live in the page cache
([host-mapped views](direct-loading.md#host-mapped-views)), so count them
toward the host RAM a warm cache needs rather than toward memlock.

- Keep `LimitMEMLOCK=infinity` (systemd units) and `--ulimit memlock=-1:-1`
  (Docker examples). Where a finite limit is required, size it for the pinned
  total, with headroom for other pinned allocations.
- A container `--memory` limit or a unit `MemoryMax=` must exceed the pinned
  total plus the process's resident size; pinned pages cannot be reclaimed.
- If the allocation fails, the server exits during startup and never falls
  back to device memory. Raise the limits, or serve TP4, which keeps
  the table on the GPU.

## Concurrency and prefill policy

Every profile admits 32 sequences by default. MiMo and Qwen TP4 AR and MiMo
TP4 DFlash are qualified through 64 (DFlash with its speculative accuracy gate
waived; see
[serving profiles](serving-profiles.md#accuracy-waiver-2026-10-06)).
Disabling speculation retains the selected profile's sequence limit; it does
not select the AR profile.
`CHOPINFER_MAX_SEQUENCES` (or `--max-sequences`) may raise this up to the
package manifest's `batch_limits.maximum_sequences`; the host refuses larger
values. A larger package maximum does not qualify a workload at that
concurrency ([alpha limits](serving-profiles.md)).

The prefill budget comes from the profile: MiMo `tp4-mtp` and `tp4-dflash`
use `--prefill-budget tbt --decode-tbt-target-ms 150`; the other profiles use
`fixed`. `--prefill-budget fixed` or `tbt` overrides it, through arguments
after the image name or `CHOPINFER_SERVE_ARGS`. See
[mixed scheduling](mixed-scheduling.md) for when to change it.
`--prefill-budget decode-aware` is experimental.

## Health, readiness, metrics and shutdown

The listener opens after engine construction and generation warmup. Warmup
waves share one 180-second deadline; keys left unwarmed are logged and
captured on first use. A failed worker or failed cleanup prevents listening.
`/health` reports worker readiness, not complete graph coverage; there is no
separate readiness endpoint. `--verify-warmup off` skips the verify waves and
context ladder.

Allow 600 seconds from process launch to readiness. The image health check
(start period 600 s) and the runbooks' `timeout 600s` waits use this budget.
It covers optional checkpoint verification, package preparation, loading
the weights (a cold page cache reads the whole checkpoint from disk), graph
capture and warmup. On slow storage, run `chopinfer verify --model` before
starting instead of `CHOPINFER_VERIFY_MODEL=1`. The image health check
probes `CHOPINFER_LISTEN` (loopback for a wildcard bind). Docker marks the
container unhealthy but does not restart it for that alone.

### What `/health` reports

Besides `status` (and `reason` when not ready) and the served `modalities`,
`/health` has a `serving` member that binds results to a package and
checkpoint:

```json
{"status": "ready", "modalities": ["text", "image"], "serving": {
  "bundle_id": "...", "package_identity": "<sha256>",
  "checkpoint_identity": "<sha256>", "metadata_identity": "<sha256>",
  "startup_nonce": "<128-bit hex>",
  "served_model_names": ["..."],
  "profile": {"name": "...", "tensor_parallel_size": 4,
              "speculation": {"mode": "adaptive", "tokens": 3, "drafter": "mtp"},
              "kv_cache_dtype": "bf16", "prefix_cache": true, "max_sequences": 32},
  "kv_page_tokens": 64,
  "version": "0.1.0-alpha.3", "source_commit": "<git sha>"}}
```

`package_identity` and `metadata_identity` match the startup
`direct_package` log line (`null` for an explicit `--bundle`), and
`checkpoint_identity` covers safetensors names, lengths and headers (`null`
without a direct checkpoint). `startup_nonce` is random per process; the
startup log repeats the member as `startup serving_identity: {...}`. `profile`
is what the server resolved after explicit options override the profile. Its
`context_parallel_size` (after `tensor_parallel_size`) appears when the served
topology declares context-parallel groups, 1 included, and is absent otherwise.
`kv_page_tokens` is the package's KV page on one rank; a context-parallel
group finds shared prefixes and captures prompt ends in stripes of
`kv_page_tokens * context_parallel_size` tokens
([KV pool](serving-kv-pool.md#context-parallelism)).

Every HTTP response, refusals included, carries `x-chopinfer-startup-nonce`
and, with a directly loaded checkpoint, `x-chopinfer-checkpoint-identity`.
`/ping` returns the same body as `/health`. Neither needs an API key, so anyone
who can reach the port can read the identities.

### Drain and shutdown

On SIGTERM or SIGINT, admission closes and `/health` becomes unready. While
admitted requests drain, the listener stays open: `/health` and new requests
return 503 with `Retry-After: 1`; `/metrics` stays available. Admitted requests,
including those still waiting for the scheduler, run to completion, subject to
`--queue-wait-ms` if set. When the work finishes or the drain budget
(`--drain-timeout-ms`, default 30 s; 0 cancels at once) expires, the listener
closes and later probes get a connection failure, not 503. At the deadline,
unfinished streams receive an error and `[DONE]` within a 100 ms flush window.
The process then releases the GPUs and exits 0. Keep the supervisor stop
timeout above the drain budget (45 s in the examples). A second signal exits
immediately.

A fatal engine or rank error makes health unavailable and exits nonzero
without native teardown, so the supervisor restarts a fresh process.
Configure the load balancer to drop unready endpoints and retry on another
replica.

Scrape `GET /metrics` every 15 seconds on the private listener and alert as
described in [metrics](operations-observability.md).

## Drain and restart checks

With a streaming request in flight (`"stream_options":{"include_usage":true}`),
send `docker kill --signal=TERM $CONTAINER`. New requests and `/health` should
get 503 with `Retry-After: 1`, and the stream should finish with usage and
`data: [DONE]` before the process exits. A manual kill or stop does not
trigger the restart policy, so run `docker start $CONTAINER` afterwards and
repeat health, model discovery and generation. To check the restart policy,
use a disposable container that exits nonzero on its own:

```sh
docker run -d --name chopinfer-restart-check --restart unless-stopped \
  --entrypoint /bin/sh $IMAGE -c 'sleep 12; exit 1'
sleep 15
docker inspect --format '{{.RestartCount}} {{.State.Status}}' chopinfer-restart-check
docker rm -f chopinfer-restart-check
```

Finish with `docker stop --time 45 $CONTAINER` and check that GPU memory is
released.

## Release records and rollback

Before routing traffic, record the image digest, source revision, checkpoint
revision, the startup `direct_package` line, serve options and model cache
location, and keep the previous image and supervisor configuration.

To roll back, drain and stop the new deployment, confirm its GPU memory is
released, then start the previous container or restore the previous systemd
environment file and restart the unit. The previous image prepares its own
package in the model cache; never edit a prepared package or the checkpoint.
Repeat `/health`, `/v1/models` and a short generation before restoring
traffic.

## Common HTTP options

**CORS** is off by default. `--allowed-origins`, `--allowed-methods`,
`--allowed-headers` (JSON arrays) or `--allow-credentials` enable it, and
unspecified lists then default to `["*"]`. `--allowed-origins '["*"]'` gives
vLLM's default browser access. Prefer explicit origins with credentials.

**Authentication.** `--api-key KEY` or `CHOPINFER_API_KEY` (then
`VLLM_API_KEY`) requires a bearer key on paths under `/v1`, `/v2`,
`/inference` and `/cohere`; wrong or missing keys get
`401 {"error":"Unauthorized"}`. `/health`, `/ping`, `/metrics`, `/version`,
the tokenizer routes and OPTIONS are exempt, as in vLLM. One key is
supported. Prefer the environment so the key is not in argv.

**Request IDs.** `--enable-request-id-headers` echoes `X-Request-Id`, or
generates one, on every response. Client IDs must be nonempty ASCII of at most
512 bytes; invalid ones are replaced.

**Model names.** `--served-model-name NAME [NAME ...]` replaces the API model
names without repackaging. The first is the default for requests that omit
`model` and labels the `vllm:*` metrics; `/v1/models` lists all. Responses echo
the requested alias (vLLM returns the primary name).

**Context.** `--max-context-tokens N` (`CHOPINFER_MAX_CONTEXT_TOKENS`) refuses
startup when KV cannot hold one request of N tokens. Without it, the context
may shrink to the available KV; check `max_model_len` in `/v1/models`.

## Migrating from vLLM

This maps configuration intent. It does not imply equal memory policy,
numerics or scheduling performance.

| vLLM argument | Chopinfer |
|---|---|
| `serve MODEL` | `serve --model DIR` (the entrypoint serves `CHOPINFER_MODEL`) |
| `--tensor-parallel-size N` | Same flag; the entrypoint sets it from `CHOPINFER_TENSOR_PARALLEL_SIZE` |
| `--served-model-name` | Same flag |
| `--max-model-len N` | `--max-context-tokens N` |
| `--max-num-seqs N` | `--max-sequences N` |
| `--max-num-batched-tokens N` | `--max-batch-tokens N` |
| `--enable-chunked-prefill` | Always on; `--prefill-chunk-tokens N` |
| `--enable-prefix-caching` | `--prefix-cache on` (default) |
| `--gpu-memory-utilization F` | `--kv-memory-fraction F`, a fraction of free memory after weights, not of total memory |
| `--kv-cache-dtype fp8_e4m3` | Same flag (`fp8` or `fp8_e4m3`) |
| `--speculative-config` | `--speculative-tokens N --speculation fixed\|adaptive --draft-cost per-token\|per-block`; the drafter comes from the package |
| `--allowed-origins`, `--allowed-methods`, `--allowed-headers`, `--allow-credentials`, `--api-key` | Same flags; CORS is off by default |
| `--trust-remote-code`, `--dtype`, backend selection, reasoning and tool parsers | No equivalent needed: the package fixes kernels, precision and the model's frontend |
| `--generation-config`, `--override-generation-config` | Sampling defaults come from the package's frontend descriptor |
| `--limit-mm-per-prompt` | Images: `--max-images-per-request N` (default 64); video and audio counts are fixed per model by the package's media recipes |

Review the [differences from vLLM](serving-request-surface.md) before
migrating clients.
