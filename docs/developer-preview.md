# Quickstart: checkpoint to first response

Build from source, verify a checkpoint, then serve it on Linux x86_64 with
RTX PRO 6000 Blackwell 96 GB GPUs. Docker is recommended; the native path is
below. Read the [supported checkpoints](compatible-checkpoints.md) and
[alpha limits](serving-profiles.md#alpha-limits). Weights are not included.

The commands below use the preview tag `v0.1.0-alpha.3`. Both paths
record the checked-out revision. Results stated for `v0.1.0-alpha.2`
(2026-10-03) belong to that earlier preview. GLM 5.3 Flash joined the
release after it, with its 2026-10-14 to 2026-10-15 requalification. To
follow later fixes, clone `main` instead (drop `--branch v0.1.0-alpha.3`
and choose your own image tag). GLM serves only at TP4, on four GPUs.
Contributors work on `main` ([contributing](contributing.md)).

## Obtain and verify a checkpoint

Download the complete snapshot with `hf download REPOSITORY --revision
REVISION --local-dir DIRECTORY`, retaining its download metadata:

| Model | Repository | Revision |
|---|---|---|
| MiMo | `XiaomiMiMo/MiMo-V2.6-Flash-MOPD` | `2479e2d0029eca9a34cc7e7f55a121925f81908e` |
| Qwen | `Qwen/Qwen3.8-Flash-Next-FP8` | `236dfdf285828023ca3bcd3f37366c58a3469b13` |
| GLM (TP4 only) | `zai-org/GLM-5.3-Flash` | `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a` |

Other checkpoints of these models, such as fine-tunes or other
quantizations, are admitted by layout
([compatible checkpoints](compatible-checkpoints.md#serve-your-own-checkpoint)).
Set `CHECKPOINT` to that directory's absolute path. Allow about 166 GB for
MiMo, 173 GB for Qwen or 328 GB for GLM, plus build space and host RAM.
Qwen TP2 pins another 51.2 GB of host RAM per server. See [deployment memory](deploy-common.md#pinned-host-memory).
A Hugging Face cache snapshot contains symlinks: mount its whole repository
cache, as described in [checkpoint mounts](deploy-common.md#checkpoint-model-cache-and-package-selection).

Verification hashes every file. Serving checks headers and lengths only.
A plain MiMo copy without download metadata needs the
[source-lock verification options](deploy-mimo26.md#checkpoint).

## Docker (recommended)

Install Docker and the NVIDIA container toolkit on a host with a CUDA
13.2-compatible driver. The build installs the pinned Rust and CUDA tools;
you do not need them on the host. The checkpoint must be readable and its
directories traversable by UID 10001; see the
[private-checkpoint permission option](deploy-common.md#checkpoint-model-cache-and-package-selection).
See [common deployment operations](deploy-common.md)
and the [MiMo](deploy-mimo26.md) or [Qwen](deploy-qwen38.md) runbook.

```sh
git clone --branch v0.1.0-alpha.3 https://github.com/logprobz/chopinfer.git
cd chopinfer
docker build -f deploy/docker/Dockerfile -t chopinfer:v0.1.0-alpha.3 .
IMAGE=chopinfer:v0.1.0-alpha.3
docker run --rm "$IMAGE" --version
CHECKPOINT=/absolute/path/to/checkpoint
# Mount driver libraries for the CLI, without exposing GPU devices.
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=none \
  --mount type=bind,src="$CHECKPOINT",dst=/data/model,readonly \
  "$IMAGE" verify --model /data/model
```

Choose one of the following layouts. The image runs as UID 10001; named
volumes receive the correct cache ownership automatically. Keep the volume
across restarts. Every start still reads the checkpoint weights.

TP4 on a dedicated four-GPU host:

```sh
docker run -d --name chopinfer-tp4 --gpus all --stop-timeout 45 \
  --ulimit memlock=-1:-1 -p 127.0.0.1:8600:8000 \
  --mount type=bind,src="$CHECKPOINT",dst=/data/model,readonly \
  --mount type=volume,src=chopinfer-cache,dst=/data/model-cache \
  -e CHOPINFER_TENSOR_PARALLEL_SIZE=4 "$IMAGE"
```

TP2 on one pair (MiMo and Qwen only; GLM serves TP4 only):

```sh
docker run -d --name chopinfer-pair-a --gpus '"device=0,1"' --stop-timeout 45 \
  --ulimit memlock=-1:-1 -p 127.0.0.1:8600:8000 \
  --mount type=bind,src="$CHECKPOINT",dst=/data/model,readonly \
  --mount type=volume,src=chopinfer-cache,dst=/data/model-cache \
  -e CHOPINFER_TENSOR_PARALLEL_SIZE=2 "$IMAGE"
```

For two TP2 servers side by side (MiMo and Qwen only), keep pair A running
and add pair B on the other GPUs and port. Both can reuse the same
checkpoint and package cache:

```sh
docker run -d --name chopinfer-pair-b --gpus '"device=2,3"' --stop-timeout 45 \
  --ulimit memlock=-1:-1 -p 127.0.0.1:8601:8000 \
  --mount type=bind,src="$CHECKPOINT",dst=/data/model,readonly \
  --mount type=volume,src=chopinfer-cache,dst=/data/model-cache \
  -e CHOPINFER_TENSOR_PARALLEL_SIZE=2 "$IMAGE"
```

The image uses native serving defaults. Add
`-e CHOPINFER_SPECULATIVE_TOKENS=0` before the image name for AR, or
`-e CHOPINFER_MODALITIES=text` to omit media encoders. Text and images are
served by default, just as with native `serve --model`;
`-e CHOPINFER_MODALITIES=all` adds MiMo's video and audio. No host IPC or
enlarged shared-memory mount was needed by the TP2 smoke. Keep ports on loopback unless an
authenticated proxy protects them.

## Native build from a tag

Install [Rust 1.89.0, CUDA 13.2.86, gcc and Python 3.11+](toolchain.md),
with the NVIDIA driver available when running the binaries. Build the same
release layout used by Docker:

```sh
git clone --branch v0.1.0-alpha.3 https://github.com/logprobz/chopinfer.git
cd chopinfer
cp local.env.example local.env   # set paths, including CHOPINFER_DATA
source local.env
export CARGO_TARGET_DIR="$CHOPINFER_DATA/preview-target"
CARGO_NET_OFFLINE=false cargo fetch --locked
bash tools/release/build.sh "$CHOPINFER_DATA/preview-archive"
mkdir -p "$CHOPINFER_DATA/preview-install"
tar -xzf "$CHOPINFER_DATA/preview-archive/chopinfer-0.1.0-alpha.3-linux-x86_64-cuda13.2.tar.gz" \
  -C "$CHOPINFER_DATA/preview-install" --strip-components=1
export PATH="$CHOPINFER_DATA/preview-install/bin:$PATH"
chopinfer --version
CHECKPOINT=/absolute/path/to/checkpoint
chopinfer verify --model "$CHECKPOINT"
CUDA_VISIBLE_DEVICES=0,1,2,3 chopinfer serve --model "$CHECKPOINT" \
  --tensor-parallel-size 4 --listen 127.0.0.1:8600
```

Use a fresh archive output directory for each build. `serve --model` selects
the family's profile; use `--speculative-tokens 0` for AR, not `--profile`.
The small package cache defaults to `$CHOPINFER_DATA/model-cache`.

Users only build, verify and serve. `bash ci/cpu.sh` is the contributor merge
gate. On a 48-core machine (2026-10-05) its host memory use peaked near 35 GB
and it took about 6 minutes with a warm target and 11 with a cold one. See
[contributing](contributing.md).

## Send a request and stop

In another shell, for either path:

```sh
URL=http://127.0.0.1:8600
timeout 600s curl --retry 600 --retry-connrefused --retry-all-errors \
  --retry-delay 1 --max-time 3 -fsS "$URL/health"
curl -fsS "$URL/v1/models"
MODEL=$(curl -fsS "$URL/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[1],"messages":[{"role":"user","content":"Say hello in one short sentence."}],"max_tokens":256,"temperature":0}))' "$MODEL" |
  curl -fsS "$URL/v1/chat/completions" -H 'Content-Type: application/json' --data-binary @-
```

Expect a healthy server, its model ID and a chat completion. Thinking can
consume output tokens; increase `max_tokens` if the response ends with
`finish_reason: length`. Stop your container with `docker stop --time 45 NAME`,
or press Ctrl-C in the native server terminal. Run checks on both ports for
two TP2 servers. See [qualification tooling](../tools/qual/README.md) for
contributor gates and benchmarks.

## Client settings for reasoning models

Use each model card's sampling: MiMo `temperature: 1.0`, `top_p: 0.95`;
Qwen `temperature: 1.0`, `top_p: 0.95`, `top_k: 20`. Never use greedy
decoding for agentic or coding work: it makes these reasoning models loop.
Keep MiMo reasoning on. Set `max_tokens` explicitly to
`min(context / 2, 131072)`, and not below 65,536; clients such as Aider
send no `max_tokens`. Use a context of at least 131,072 tokens for that budget.

A request served alone (C1) is bit-reproducible for the same
configuration, with or without speculation, when it is greedy or repeats the
same explicit `seed` and sampling parameters; a sampled request without a
seed draws a fresh one, and MiMo and Qwen sample at temperature 1.0 by
default. Outputs under concurrent load are not bit-reproducible across batch
compositions: the batch bucket can change the order of F32 summation, which
shifts logprobs and can change a greedy token, within the accuracy gates
([batch shape](numerics.md#batch-shape)).
Streamed tool-call arguments arrive whole when each call closes.

## Report a problem

[Open an issue](https://github.com/logprobz/chopinfer/issues) with the version,
GPU type and count, driver version, exact commands, a short reproduction and
the startup log. Include package and checkpoint identities from
`startup direct_package`. Remove credentials and private inputs. Report
security issues through [SECURITY.md](../SECURITY.md).
