# Build and serve

Use the [supported hardware and configurations](../README.md). Build from a
chosen checkout; weights are downloaded separately. Docker supplies the
pinned build tools. The native route uses the same release builder.

## Checkpoints

Download a complete Hugging Face snapshot with `hf download REPOSITORY
--revision REVISION --local-dir DIRECTORY`, retaining its download metadata.
The model's `checkpoint-files.json` records the pinned source and revision:
[MiMo](../models/mimo26/checkpoint-files.json),
[Qwen](../models/qwen38/checkpoint-files.json),
[GLM](../models/glm5next/checkpoint-files.json).
Set `CHECKPOINT` to the local directory. Allow space for weights, build
output and host memory outside the checkout.

Compatible fine-tunes and quantizations are admitted by their config and
tensor layout, not by a repository-name whitelist. A matching layout does
not qualify its output. Installed model helpers reject unsupported geometry,
missing tensors, scale layouts and precision combinations before serving.
`chopinfer verify --model DIR` checks payload digests; serving itself checks
headers and lengths. For a copy without download metadata, use
`verify --manifest FILE#ENTRY` with the complete recorded checksums.

## Docker

Install Docker, the NVIDIA container toolkit and a CUDA 13.2-compatible
driver. From the selected checkout:

```sh
IMAGE=chopinfer:local
docker build -f deploy/docker/Dockerfile -t "$IMAGE" .
docker run --rm "$IMAGE" --version
# Set CHECKPOINT to your complete local snapshot.
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=none \
  --mount type=bind,src="$CHECKPOINT",dst=/data/model,readonly \
  "$IMAGE" verify --model /data/model
docker run -d --name chopinfer --gpus all --stop-timeout 45 \
  --ulimit memlock=-1:-1 -p 127.0.0.1:8600:8000 \
  --mount type=bind,src="$CHECKPOINT",dst=/data/model,readonly \
  --mount type=volume,src=chopinfer-cache,dst=/data/model-cache \
  -e CHOPINFER_TENSOR_PARALLEL_SIZE=4 "$IMAGE"
```

For TP2, use `--gpus '"device=0,1"'` and
`CHOPINFER_TENSOR_PARALLEL_SIZE=2`. Separate servers need disjoint GPU sets
and ports. Visible devices are renumbered from zero; each rank needs a
distinct visible ordinal. Startup checks peer access between all ranks.

### Checkpoint and cache permissions

The image runs as UID 10001. Checkpoint files must be readable and their
directories traversable by that user. A read-only mount grants no extra
permissions. For a private checkpoint, run as its owner with
`--user "$(id -u):$(id -g)"` and use a bind-mounted cache that owner can write.
Keep checkpoint files immutable and mounted read-only.

For a hub cache snapshot, mount the whole repository cache, including
`blobs/`, and set `CHOPINFER_MODEL` to the snapshot inside that mount.
The model cache defaults to `/data/model-cache`; keep it across restarts.
Every start still reads checkpoint weights. `CHOPINFER_VERIFY_MODEL=1`
verifies at startup; `CHOPINFER_VERIFY_MANIFEST=FILE#ENTRY` supplies explicit
checksums. Model-specific service examples are in [deploy/systemd](../deploy/systemd).

### Pinned host memory

Qwen TP2 pins 51.2 GB of host memory for n-grams until exit. KV offload adds
the configured host arena across ranks, and weight loading pins two 64 MiB
upload buffers per rank. A model with checkpoint-mapped views also pins a
64 MiB gather ring per gathering stream; the mapped bytes stay in the page
cache and count toward host RAM, not memlock. Set Docker memlock or systemd
`LimitMEMLOCK` for that total with headroom. Container memory limits must also cover the
resident process. A failed pinned allocation stops startup; it does not
fall back to device memory. Qwen TP4 keeps the n-gram table on the GPU.

## Native build

Use the toolchain and external paths in [contributing](contributing.md#setup).
From the selected checkout, with fresh archive and install directories:

```sh
cp local.env.example local.env
# Edit local.env for your installation and external data directory.
source local.env
export CARGO_TARGET_DIR="$CHOPINFER_DATA/build-target"
CARGO_NET_OFFLINE=false cargo fetch --locked
bash tools/release/build.sh "$CHOPINFER_DATA/release-archive"
mkdir -p "$CHOPINFER_DATA/release-install"
tar -xzf "$CHOPINFER_DATA"/release-archive/chopinfer-*-linux-x86_64-cuda13.2.tar.gz \
  -C "$CHOPINFER_DATA/release-install" --strip-components=1
export PATH="$CHOPINFER_DATA/release-install/bin:$PATH"
chopinfer verify --model "$CHECKPOINT"
CUDA_VISIBLE_DEVICES=0,1,2,3 chopinfer serve --model "$CHECKPOINT" \
  --tensor-parallel-size 4 --listen 127.0.0.1:8600
```

`serve --model` selects the family's profile and refuses `--profile`;
`--speculative-tokens 0` selects AR. Explicit packages accept
[`--profile`](serving-profiles.md#selecting-a-profile). Use `serve --help`
for options and `profiles list` for declared settings. Text and images are
default inputs; `--modalities text` omits encoders and `--modalities all`
enables the package's additional media. See the
[qualification limits](serving-profiles.md#qualification-scope).

## Send a request and stop

Allow 600 seconds for cold checkpoint verification, preparation, loading
and warmup. In another shell:

```sh
URL=http://127.0.0.1:8600
timeout 600s curl --retry 600 --retry-connrefused --retry-all-errors \
  --retry-delay 1 --max-time 3 -fsS "$URL/health"
MODEL=$(curl -fsS "$URL/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[1],"messages":[{"role":"user","content":"Say hello in one short sentence."}],"max_tokens":256,"temperature":0}))' "$MODEL" |
  curl -fsS "$URL/v1/chat/completions" -H 'Content-Type: application/json' --data-binary @-
```

Thinking consumes output tokens; raise `max_tokens` when the finish reason
is `length`. Stop with `docker stop --time 45 chopinfer`, or Ctrl-C for the
native server. Keep ports on loopback unless an authenticated proxy protects
them. [Operations](operations-observability.md) covers readiness and draining;
the [HTTP reference](serving-request-surface.md) covers client settings.

## Report a problem

[Open an issue](https://github.com/logprobz/chopinfer/issues) with the version,
GPU type/count, driver, commands and a short reproduction. Include startup
package/checkpoint identities, removing credentials and private inputs.
Report vulnerabilities through [SECURITY.md](../SECURITY.md).
