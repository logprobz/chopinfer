# Deploy MiMo V2.6 Flash

This runbook serves MiMo V2.6 Flash from the shared image. Prerequisites, the
image build, environment variables, health, drain and rollback are in
[common deployment operations](deploy-common.md). For tag builds and TP2 examples, use the [quickstart](developer-preview.md).

## Checkpoint

The default checkpoint is `XiaomiMiMo/MiMo-V2.6-Flash-MOPD` at revision
`2479e2d0029eca9a34cc7e7f55a121925f81908e` (about 166 GB). See
[serving MiMo](serving-mimo26.md) for current qualification and release gates. Serving writes no copy of the weights;
the model cache holds only small packages.

`chopinfer serve --model DIR --tensor-parallel-size 4` serves the `tp4-dflash`
profile, reading the DFlash drafter from `dflash/dflash_draft_model.safetensors`
inside the checkpoint, plus image and audio inputs through the encoders
planned from the checkpoint. Video is opt-in. A checkpoint without `dflash/` serves MTP. The
package's served name is `MiMo-V2.6-Flash-MOPD`.

Serving checks only safetensors headers and lengths, so check the contents
once. A plain MiMo copy without Hugging Face metadata needs the source lock:

```sh
chopinfer verify --model "$CHECKPOINT" \
  --manifest models/mimo26/checkpoint-files.json#mopd --allow-unrecorded
```

## Start and test on dedicated GPUs

Run this only on GPUs no other service is using. TP4 needs exactly four
visible devices; `--gpus all` assumes a dedicated four-GPU machine (see
[device visibility](deploy-common.md#device-visibility-and-rank-selection)).

```sh
IMAGE=chopinfer:v0.1.0-alpha.3
CHECKPOINT=/path/to/MiMo-V2.6-Flash-MOPD
docker run -d --name chopinfer-mimo26-test --gpus all \
  -e CHOPINFER_EXPECTED_BUNDLE_ID=mimo.v26.flash.v2 \
  --restart unless-stopped --stop-timeout 45 \
  --ulimit memlock=-1:-1 \
  -p 127.0.0.1:8312:8000 \
  --mount type=bind,src="$CHECKPOINT",dst=/data/model,readonly \
  --mount type=volume,src=chopinfer-cache,dst=/data/model-cache \
  "$IMAGE"

URL=http://127.0.0.1:8312
# Retry while the package is prepared, weights load and warmup runs.
timeout 600s curl --retry 600 --retry-connrefused --retry-all-errors --retry-delay 1 \
  --max-time 3 -fsS "$URL/health"
curl -fsS "$URL/v1/models"
MODEL=$(curl -fsS "$URL/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[1],"prompt":"Hello.","temperature":0,"max_tokens":8}))' "$MODEL" |
  curl -fsS "$URL/v1/completions" -H 'Content-Type: application/json' --data-binary @-
curl -fsS "$URL/metrics"
chopinfer eval run --model models/mimo26 --url "$URL" --served-model "$MODEL" --check api
```

Bind externally only behind an authenticated proxy or on a trusted network.
Then run the [drain and restart checks](deploy-common.md#drain-and-restart-checks)
with `CONTAINER=chopinfer-mimo26-test`.

## Defaults

The image inherits native `serve --model` defaults from the selected
[serving profile](serving-profiles.md): 32 sequences (TP4 AR and DFlash are
qualified through 64), 8192 batch tokens, 2048 prefill tokens, KV fraction 0.9, KV dtype `auto` and prefix
caching on. Set only the [environment overrides](deploy-common.md#environment-and-arguments)
you need. `CHOPINFER_SPECULATIVE_TOKENS=0` selects AR; otherwise the native
profile selects the checkpoint's drafter. CLI arguments win over environment
values.

The default package serves text and images, as native `serve --model` does.
`CHOPINFER_MODALITIES=all` adds video and audio, at some KV capacity
([capacity](serving-profiles.md#served-modalities-and-memory)), and
`CHOPINFER_MODALITIES=text` omits media encoders. Local media
files need a read-only mount and `--allowed-local-media-path DIR`.

## TP2

`CHOPINFER_TENSOR_PARALLEL_SIZE=2` selects TP2. TP2 and TP4 are the
served topologies
([alpha limits](serving-profiles.md)). MiMo TP2 with the media encoders
admits 32 sequences (16 with DFlash) and declares a per-rank minimum of 92 to
94 GiB. A device with less free memory than the minimum still serves, with a
smaller KV pool and a `startup memory warning` line. See
[MiMo topology](topology-mimo.md) and
[separate instances](deploy-common.md#package-topology-and-separate-instances).

## systemd

Install the image's `chopinfer` and both `chopinfer-model-*` helpers under
`/opt/chopinfer/bin` and the modules under `/opt/chopinfer/lib` (neither group
or world writable), the entrypoint there with mode 755,
and `deploy/docker/deployment.py` as `/opt/chopinfer/bin/deployment.py`.
Install Python 3.
Use a system account `chopinfer` with access to the NVIDIA device nodes. Put
the checkpoint at `/var/lib/chopinfer/models/mimo-v2.6-flash`, readable but not
writable by that account, and create `/var/lib/chopinfer/model-cache` owned by
it with mode 700.

```sh
sudo install -d -m 755 /etc/chopinfer
sudo install -m 640 deploy/systemd/mimo26.env.example /etc/chopinfer/mimo26.env
sudo install -m 644 deploy/systemd/chopinfer-mimo26.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now chopinfer-mimo26
sudo journalctl -u chopinfer-mimo26 -f
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
```

Edit the environment file for the checkpoint, listener and serve options; the
defaults above apply through the same variables. `Type=exec` reports process
start, so wait for `/health` before traffic. Do not run Docker and systemd
deployments on the same GPUs.
