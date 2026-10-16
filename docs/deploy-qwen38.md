# Deploy Qwen3.8 Flash

This runbook serves Qwen3.8 Flash from the shared image. Prerequisites, the
image build, environment variables, health, drain and rollback are in
[common deployment operations](deploy-common.md). For tag builds and TP2 examples, use the [quickstart](developer-preview.md).

## Checkpoint

Download `Qwen/Qwen3.8-Flash-Next-FP8` at revision
`236dfdf285828023ca3bcd3f37366c58a3469b13` (about 173 GB) into a plain
directory with `hf download --local-dir`, as in the
[quickstart](developer-preview.md). Serving writes no copy of the weights; the
model cache holds only small packages.

`chopinfer serve --model DIR --tensor-parallel-size 4` serves the `tp4-mtp`
profile with the MTP drafter, and image inputs through the planned vision
encoders when the checkpoint has the vision tower. Serving checks only
safetensors headers and lengths, so run `chopinfer verify --model DIR` once
(or start with `CHOPINFER_VERIFY_MODEL=1`) before relying on a checkpoint.

See the [Qwen bundle](bundle-qwen38.md) for current text and media evidence.
Run the [release gate](../tools/qual/README.md) on the exact image
and checkpoint before promotion; to compare AR, rerun with
`CHOPINFER_SPECULATIVE_TOKENS=0` sequentially at the same memory and batch
settings, never on the same GPUs at once.

## Start on dedicated GPUs

Run this only on GPUs no other service is using. TP4 needs exactly four
visible devices; `--gpus all` assumes a dedicated four-GPU machine (see
[device visibility](deploy-common.md#device-visibility-and-rank-selection)).

```sh
IMAGE=chopinfer:v0.1.0-alpha.3
CHECKPOINT=/path/to/Qwen3.8-Flash-Next-FP8
docker run -d --name chopinfer-qwen38-test --gpus all \
  -e CHOPINFER_EXPECTED_BUNDLE_ID=qwen38.flash.v2 \
  --restart unless-stopped --stop-timeout 45 \
  --ulimit memlock=-1:-1 \
  -p 127.0.0.1:8431:8000 \
  --mount type=bind,src="$CHECKPOINT",dst=/data/model,readonly \
  --mount type=volume,src=chopinfer-cache,dst=/data/model-cache \
  "$IMAGE"

URL=http://127.0.0.1:8431
timeout 600s curl --retry 600 --retry-connrefused --retry-all-errors --retry-delay 1 \
  --max-time 3 -fsS "$URL/health"
curl -fsS "$URL/v1/models"
# Use the served identity returned by the package.
MODEL=$(curl -fsS "$URL/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[1],"prompt":"Hello.","temperature":0,"max_tokens":8}))' "$MODEL" |
  curl -fsS "$URL/v1/completions" -H 'Content-Type: application/json' --data-binary @-
curl -fsS "$URL/metrics"
```

Bind externally only behind an authenticated proxy or on a trusted network.
A successful `/health` means construction and warmup succeeded, not that the
release is qualified. During an MTP workload, the `vllm:spec_decode_*`
counters must increase. Then run the
[drain and restart checks](deploy-common.md#drain-and-restart-checks) with
`CONTAINER=chopinfer-qwen38-test`.

## Defaults

The image inherits native `serve --model` defaults from the selected
[serving profile](serving-profiles.md): 32 sequences (TP4 AR is qualified
through 64), 8192 batch tokens, 2048 prefill tokens, KV fraction 0.9, KV dtype `auto` and prefix
caching on. Set only the [environment overrides](deploy-common.md#environment-and-arguments)
you need. `CHOPINFER_SPECULATIVE_TOKENS=0` selects AR; otherwise the native
profile selects the checkpoint's drafter. CLI arguments win over environment
values.

The default package serves text and images, as native `serve --model` does.
The checkpoint provides no video or audio, so `CHOPINFER_MODALITIES=all` serves
the same package; `CHOPINFER_MODALITIES=text` omits the vision encoder. Local media
files need a read-only mount and `--allowed-local-media-path DIR`.

## TP2

`CHOPINFER_TENSOR_PARALLEL_SIZE=2` selects TP2. TP2 and TP4 are the
served topologies
([alpha limits](serving-profiles.md)). TP2 keeps the n-gram table in 51.2 GB
of [pinned host memory](deploy-common.md#pinned-host-memory). See
[Qwen topology](topology-qwen.md) and
[separate instances](deploy-common.md#package-topology-and-separate-instances).

## systemd

Install the image's `chopinfer` and both `chopinfer-model-*` helpers under
`/opt/chopinfer/bin` and the modules under `/opt/chopinfer/lib` (neither group
or world writable), the executable entrypoint and `deploy/docker/deployment.py`
under `/opt/chopinfer/bin` as in the image, and Python 3. Use a `chopinfer` system account with NVIDIA device access. Put the
checkpoint at `/var/lib/chopinfer/models/qwen3.8-flash`, readable but not
writable by that account, and create `/var/lib/chopinfer/model-cache` owned by
it with mode 700.

```sh
sudo install -d -m 755 /etc/chopinfer
sudo install -m 640 deploy/systemd/qwen38.env.example /etc/chopinfer/qwen38.env
sudo install -m 644 deploy/systemd/chopinfer-qwen38.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now chopinfer-qwen38
sudo journalctl -u chopinfer-qwen38 -f
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
```

Edit the environment file for the checkpoint path and listener. `Type=exec`
reports process start, so wait for `/health` before traffic. Do not run Docker
and systemd services on the same GPUs.
