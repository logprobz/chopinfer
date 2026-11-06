# Build and serve

Use the [supported hardware and configurations](../README.md). Build from a
chosen checkout; weights are downloaded separately. Docker supplies the
pinned build tools. The native route uses the same release builder.

## Checkpoints

Download a complete Hugging Face snapshot with `hf download REPOSITORY
--revision REVISION --local-dir DIRECTORY`, retaining its download metadata.
The model's `checkpoint-files.json` records the pinned source and revision:
[Qwen](../models/qwen38/checkpoint-files.json),
[GLM](../models/glm5next/checkpoint-files.json),
[DeepSeek](../models/deepseek41/checkpoint-files.json).
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
the configured host arena across ranks (`--kv-offload-gib`); the RAM prefix
tier (`--kv-offload-prefix on`) lives in that arena and pins nothing more.
Weight loading pins two 64 MiB upload buffers per rank. A model with
checkpoint-mapped views also pins a 64 MiB gather ring per gathering
stream; the mapped bytes stay in the page
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
`--speculative-tokens 0` runs the AR baseline, a correctness control rather
than a serving profile. Explicit packages accept
[`--profile`](#selecting-a-profile). Use `serve --help`
for options and `profiles list` for declared settings. Text and images are
default inputs; `--modalities text` omits encoders and `--modalities all`
enables the package's additional media. See the
[qualification limits](#qualification-scope).

The release builder prints a recorded prebuilt directory for later
`tools/release/build.sh --stage-from PREBUILT OUTPUT` use. Raw directories
from older builds need rebuilding. Native and Docker builds use the same
[producer and staging checks](contributing.md#release); provenance does not
qualify a model's numerical output.

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

## Selecting a profile

Each model owns `profiles.json`; `chopinfer profiles list FILE` and
`chopinfer profiles show FILE NAME` print settings and recorded status and
validate profiles without a GPU. Package manifests remain authoritative for
topology, limits and capabilities.

Production serves the speculative profiles: DeepSeek `tp4-dspark`, GLM
`tp4-mtp` and its DCP variants, and Qwen `tp4-mtp`. Each model's AR profile
is the baseline and correctness control that speculative output is checked
against, and a debugging aid. `tp4-dspark` drafts nothing from 32 decode
items, so falling back to AR needs no profile switch. MiMo is outside the
release scope.

`serve --model DIR` selects a profile through the installed model helper and
refuses `--profile`. Explicit packages accept a packaged name or `FILE#NAME`:

```sh
chopinfer serve --bundle "$PACKAGE" --checkpoint "$CHECKPOINT" --profile tp4-mtp
chopinfer profiles list models/qwen38/profiles.json
```

Four commands describe these choices as JSON, for a launcher such as
`choptop`:

- `chopinfer models --json` lists each installed model's profiles with their
  status, rank count, drafter, sequence limits, qualification text and the
  effective settings `serve --model` resolves from each.
- `chopinfer models pick --model DIR --json` names, for each rank count the
  model's profiles serve, the profile `serve --model` picks for that
  checkpoint or the helper's refusal, and which other profiles are reachable
  by explicit options.
- `chopinfer models command --model DIR --tensor-parallel-size N --profile
  NAME -- OPTIONS` prints the `serve --model` arguments that serve a profile
  with the given options.
- `chopinfer serve ... --check` runs every startup check that needs neither a
  device nor the package's native module, in serving's order, and prints the
  profile, the effective settings and every model option with its value and
  default. A failure exits nonzero with the message serving would give; a
  start can still fail on what only it decides (the native module, devices,
  their memory and peer access, KV capacity and the listen address).

`serve --describe-options` prints every serve option with its value shape,
group and help, and the environment variables serve reads.

An explicit option overrides a profile default. `--speculative-tokens 0`
disables speculation but retains the profile's sequence cap. A profile
cannot grant qualification to an override. `max_sequences` is the serving
cap; `qualified_max_sequences` is the largest concurrency its evidence
covers, or null for experimental profiles, and startup warns above it. The
shared default is 32. Production deployments should set `--queue-depth N`
or `--queue-wait-ms N` ([queueing](serving-request-surface.md#queueing-and-admission)):
profiles set neither, so by default waiting requests queue without bound
and overload never returns 429.

## Serving options

### GPU memory

`--gpu-memory-utilization F` (profile field `gpu_memory_utilization`, image
variable `CHOPINFER_GPU_MEMORY_UTILIZATION`) is the share of each GPU's
total memory this server's own allocations may take: weights, workspaces,
encoders, drafter and request state, the sampler, the KV pool and its graph
memory reserve. Every profile uses 1.0, which gives the pool all free memory
but the reserve; lower it to leave memory for other programs on the same
GPUs. The CUDA context and loaded kernel code come on top. When F leaves no
room for one KV page, startup refuses and logs the smallest utilization that
fits.

CUDA graphs embed the pool's addresses, so they are captured after it
exists. The host keeps a reserve per rank out of the pool for them,
`--graph-memory-reserve-mib N` (profile field `graph_memory_reserve_mib`),
1 GiB when neither names one; each profile sets its own from measured graph
memory. Startup logs each rank's free memory after warmup and warns below
256 MiB; raise the reserve or lower `--max-sequences` when a rank warns. A
configuration that raises its graph budget must measure its reserve again.
A GLM deployment that serves no images gets the vision encoder's memory back
for its KV pool with the text package (`--modalities text`).
`CHOPINFER_WEIGHT_CENSUS=0` plans every weight section raw instead of in its
smaller lossless encoding, which costs KV pool memory.

### Speculation capacity

- `--speculative-rows N` (`speculation.rows`), the token capacity: the most
  verify rows one step may carry, one row per decoding item plus every
  draft. Startup refuses rows at or below `--speculative-tokens` or above
  the package's maximum.
- `--speculation-graph-rows N` (`speculation.graph_rows`), the graph budget:
  the widest verify step captured in a CUDA graph, for a package that
  declares the `verify_graph_rows` model option. Wider steps run without a
  graph.
- `--speculation-confidence JSON|off` (`speculation.confidence`) lets a
  balanced plan leave the choice of which drafts to verify to a package that
  declares `chop.verify-allocation.v1`. Every emitted token is still
  verified. The calibration names the package's `confidence_contract`, a
  `floor` of drafts every item verifies, and `greedy` and `sampled`
  `[scale, offset]` pairs per draft position. A step with a request whose
  sampling depends on its history (penalties or a grammar constraint) is
  verified on the host without allocation.

`--list-model-options` checks these against the package without opening a
device. Each graph captured above the default rows also takes memory from
the graph memory reserve.

### Prefill policy

`--prefill-policy fixed|tbt|adaptive` (profile field `prefill_policy`,
default `fixed`) selects how prefill shares a step with decoding requests:

| Policy | Mixed-step sizing | Timing cuts |
| --- | --- | --- |
| `fixed` | Configured token and chunk limits | Off |
| `tbt` | Time-between-token step model | Off |
| `adaptive` | TBT cap sized from prefill demand | On with a grid and negotiated cuts; off without a grid |

`--decode-tbt-target-ms` applies to `tbt` and `adaptive`, defaults to
100 ms and accepts 1 through 60,000 ms; specifying it with `fixed` is an
error. DeepSeek's `tp4-dspark` uses `adaptive` at 750 ms; every other
shipped profile uses `fixed`. Under `adaptive`, a sustained backlog sizes
steps to keep the predicted step within the target, and a transient one may
run to four thirds of it; the target is a cadence goal, not a latency bound.
A grid bundle without negotiated prefill cuts refuses `adaptive`. Startup
prints `prefill policy: fixed|tbt|adaptive`.

The retired flags `--prefill-budget`, `--prefill-timing-cuts`,
`--prefill-itl-target-ms`, `--prefill-decode-steps`,
`--mixed-min-prefill-tokens` and `--mixed-decode-steps` (and their profile
fields) fail at startup with guidance; `tbt` with adaptive timing cuts is
now `adaptive`, and decode-aware budgets and fixed timing cuts have no exact
replacement.

### KV offload and the RAM prefix tier

`--kv-offload-gib N` (profile field `kv_offload_gib`) pins N GiB of host
memory, split evenly across the ranks, as an arena for KV pages copied off
the GPU; 0, the default, turns offload off. It needs optimistic admission
and a bundle that declares `chop.kv-offload.v1`. A preempted request swaps
its state into the arena and resumes from it. `--kv-offload-min-tokens N`
(default 64) is the fewest tokens a request swaps or the tier keeps, and
`--kv-offload-gbps F` caps the copy rate per link. Memory preflight requires
the arena plus a 16 GiB reserve to fit within MemAvailable, and the arena
counts toward the memlock limit ([pinned host memory](#pinned-host-memory)).

`--kv-offload-prefix on` (profile field `kv_offload_prefix`, default off)
keeps prefix anchors the GPU evicts in a RAM tier inside the same arena and
restores them from there ([design](numerics.md#kv-offload-and-prefix-state)).
It needs a positive `--kv-offload-gib`, `--prefix-cache on` and optimistic
admission. The tier is process memory: a restart, worker restart or deploy
clears it. Startup logs `prefix RAM tier: on` with its bounds, or why it is
off. `--debug-prefix-ram` is for exactness runs only. No profile sets either
field.

### Host allocator memory

On Linux with glibc, `serve` sets `M_ARENA_MAX` to 8 at startup.
`CHOPINFER_MALLOC_ARENA_MAX` chooses another positive limit, and `0` leaves
glibc's policy unchanged; invalid values refuse startup. The generation
worker returns unused allocator pages with `malloc_trim(0)` at most once a
minute after activity drains.

## Qualification scope

Qualification is tied to actual checkpoint tensors, package, host, topology,
settings and workload. A compatible checkpoint needs its own evidence.
Eight-rank, multi-node, MPS and other GPU SKUs are unqualified. TP2 profiles
remain experimental. Text and image are default inputs; MiMo can also serve
video and audio, but audio is not qualified in-tree. Offload and the RAM
prefix tier are opt-in and not qualified for every profile. Different batch
buckets can change output bits under concurrency
([batch shape](numerics.md#batch-shape)); single-request and fixed-batch
guarantees do not extend to arbitrary concurrent batches. Each model's
README states what its profiles are qualified for.

## Report a problem

[Open an issue](https://github.com/logprobz/chopinfer/issues) with the version,
GPU type/count, driver, commands and a short reproduction. Include startup
package/checkpoint identities, removing credentials and private inputs.
Report vulnerabilities through [SECURITY.md](../SECURITY.md).
