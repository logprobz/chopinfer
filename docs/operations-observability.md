# Metrics, request traces and step profiles

`chopinfer serve` publishes Prometheus text at `/metrics` without
authentication. Scrape each process separately, every 15 seconds on the
private listener. Native engine and KV metrics come from the worker that owns
their state; HTTP refusal and intake metrics come from the listener. Label
domains are bounded and request IDs never become labels. Counters advance only
after the corresponding transition commits and reset on restart. For
behavioral or numerical diagnosis, use the [first-divergence workflow](contributing.md#diagnostics).

## vLLM-compatible metrics

The [contract fixture](../crates/runtime-engine/tests/vllm_metrics.json)
records names, labels and default histogram bounds. Histograms publish
cumulative `_bucket` samples with `le` (integer bounds keep `.0`, as the
Prometheus Python client prints them), plus `_sum` and `_count`. Every series
below carries `model_name` (the first served name) and `engine="0"`.

| Published series | Type and meaning | Additional labels |
|---|---|---|
| `vllm:request_success_total` | Terminal engine sequences, including cancellation | `finished_reason`: `stop`, `length`, `abort` |
| `vllm:num_requests_running` | Live sequences that have been scheduled | None |
| `vllm:num_requests_waiting` | Live sequences not yet scheduled | None |
| `vllm:prompt_tokens_total` | Full prompt tokens, including cache reuse, counted on first generated output | None |
| `vllm:prompt_tokens_cached_total` | Reused tokens in those prompts | None |
| `vllm:prompt_tokens_by_source_total` | Those prompt tokens partitioned by origin | `source`: `local_compute`, `local_cache_hit`, `external_kv_transfer` (zero) |
| `vllm:generation_tokens_total` | Committed generated tokens, including hidden stop/reasoning tokens | None |
| `vllm:time_to_first_token_seconds` | Engine admission to first generated token | None |
| `vllm:inter_token_latency_seconds` | Interval between generated outputs, once per decode step | None |
| `vllm:e2e_request_latency_seconds` | Engine admission to terminal transition | None |
| `vllm:request_queue_time_seconds` | Admission to first scheduling, observed at completion | None |
| `vllm:request_prefill_time_seconds` | First scheduling to first generated token | None |
| `vllm:request_decode_time_seconds` | First to last generated token | None |
| `vllm:request_inference_time_seconds` | First scheduling to last generated token | None |
| `vllm:request_time_per_output_token_seconds` | Decode duration divided by output tokens minus one; zero for at most one token | None |
| `vllm:request_prompt_tokens` | Full prompt length per terminal sequence | None |
| `vllm:request_generation_tokens` | Generated tokens per terminal sequence | None |
| `vllm:request_params_max_tokens` | Output limit per terminal sequence | None |
| `vllm:request_prefill_kv_computed_tokens` | Prompt length minus reused tokens per terminal sequence | None |
| `vllm:iteration_tokens_total` | Histogram of computed prefill tokens plus generated tokens per committed batch; despite its name, not a counter | None |
| `vllm:num_preemptions_total` | Requests returned to the queue to recompute; zero under `--admission reserved` | None |
| `vllm:request_num_preemptions` | Preemptions of each terminal sequence | None |
| `vllm:kv_cache_usage_perc` | Occupied KV blocks divided by capacity, a fraction from 0 to 1 | None |
| `vllm:prefix_cache_queries_total` | Queried prompt tokens | None |
| `vllm:prefix_cache_hits_total` | Successfully reused prompt tokens | None |
| `vllm:spec_decode_num_drafts_total` | Verify items that checked drafts | None |
| `vllm:spec_decode_num_draft_tokens_total` | Tokens verified as drafts | None |
| `vllm:spec_decode_num_accepted_tokens_total` | Accepted drafts after stop/length cuts | None |
| `vllm:spec_decode_num_draft_tokens_per_pos_total` | Verified drafts at each position | `position`, starting at `0` |
| `vllm:spec_decode_num_accepted_tokens_per_pos_total` | Accepted drafts at each position | `position`, starting at `0` |

Scope matters when comparing dashboards with vLLM. Each generated choice is an
engine sequence. Timings exclude HTTP body intake, tokenization and network
delivery. A cancelled sequence leaves TTFT unobserved. Speculative output
reports one inter-token interval per step, even when the step emits several
tokens. Iteration counts follow committed Chopinfer batches. Some reference
series have no Chopinfer producer and are absent (fairness pressure, external
KV transfer, multimodal frontend cache, sleep modes, LoRA, corrupted-logit
counters, `request_params_n`, Python collector metrics).

## Chopinfer metrics

Native prompt counters count at admission, while the vLLM prompt counter waits
for the first generated output. Native prefix-cache query and hit counters
count lookups; the vLLM ones count tokens.

| Series | Meaning |
|---|---|
| `chop_admission_refusals_total{reason}` | Generation and stored-response HTTP errors by status class: `invalid_request`, `unknown_model`, `not_found`, `intake_timeout`, `capacity`, `unavailable`, `internal`, `other`. `unknown_model` counts generation 404s; `not_found` counts missing stored response IDs on read and cancel. Dashboards that treated all 404s as `unknown_model` should include `not_found`. Actual model-name refusals emit bounded `INFO unknown_model refusal` lines with the requested name, socket peer and served names. Drain middleware refusals also count. Errors after SSE headers are outside this counter. |
| `chop_intake_budget_bytes` | Configured listener budget for request bodies |
| `chop_intake_used_bytes` | Currently charged bytes, including the 16 KiB minimum per request |
| `chop_instance_poisoned{model_name,engine}` | Sticky 1 after model commit/rollback, retirement or prefix-state failure makes reuse unsafe |
| `chop_generation_worker_{starting,ready,failed,stopped}` | One-hot worker lifecycle state |
| `chop_speculation_k_total{model_name,engine,k}` | Scheduled decode/verify items by selected K, including K=0; counts attempted work even if execution fails |
| `chop_step_gpu_seconds{model_name,engine}` | Histogram of executor-reported kernel time per measured execution, absent when no executor measurement exists |
| `chop_batches_total`, `chop_{prefill_only,decode_only,mixed}_batches_total` | Actual batch composition |
| `chop_step_execution_seconds{model_name,engine,phase}` | Histogram of engine wall time from execute through model commit per committed step; `phase` is `decode`, `mixed` or `prefill` by whether the batch had prefill items and decode or verify items |
| `chop_step_rows_total{model_name,engine,decode_rows,verify_rows,row_capacity}` | Exact count of committed decode/verify steps with this target-row geometry; excludes any prefill-bearing step. Capacity is `unknown` when the executor cannot report it |
| `chop_step_rows_dropped_total{model_name,engine}` | Observations of new geometries beyond the 4096-bin process limit; existing bins continue counting |
| `chop_step_prefill_tokens{model_name,engine}` | Histogram of prompt tokens per prefill-bearing step |
| `chop_prefill_budget_selected_tokens{model_name,engine}` | Decode-aware and tbt modes only: histogram of the prefill cap selected for each step beside decoding requests, whether or not prefill used it; prefill-only steps select none |
| `chop_prefill_budget_decode_estimate_seconds{model_name,engine,rows_le}`, `..._samples` | Decode-aware mode only: the decode step estimate per power-of-two row bucket and its sample count; two samples qualify it |
| `chop_prefill_budget_prefill_token_seconds{model_name,engine,tokens_le}` | Decode-aware mode only: estimated prefill time per token per power-of-two step-token bucket |
| `chop_mixed_step_predicted_seconds{model_name,engine}`, `chop_mixed_step_actual_over_predicted{model_name,engine}` | Tbt mode only: histograms of each mixed step's predicted time, for the batch as packed, and of its execution time over that prediction |
| `chop_mixed_budget_term_seconds{model_name,engine,term}`, `chop_mixed_budget_samples`, `chop_mixed_budget_trained`, `chop_mixed_budget_decode_estimate_seconds{items_le}` | Tbt mode only: the fitted fixed, mixed, per-token and per-attention-pair terms, prefill steps in the fit window, whether the fit sets budgets, and the decode estimate per power-of-two item bucket ([mixed scheduling](serving-profiles.md#prefill-policy)) |
| `chop_prefix_cache_{queries,hits}_total`, `chop_prefix_cache_{queried,reused}_tokens_total` | Prefix-cache lookups and restored hits, and their prompt and reused tokens |
| `chop_prefix_cache_recomputed_seen_tokens_total` | Prompt tokens a committed prefill computed although an earlier request's committed prefill had computed the same leading full pages, short of the last prompt token: repeats and shared prefixes no anchor covered ([where anchors are captured](numerics.md#kv-offload-and-prefix-state)) |
| `chop_prefix_cache_{captures,capture_failures,restore_failures,evictions}_total`, `chop_prefix_cache_{anchors,blocks}` | Anchor captures, captures and restores the bundle refused, evictions, and the anchors and pages held now |
| `chop_errors_total`, `chop_cancellations_total` | Engine errors and cancelled sequences |
| `chop_requests_submitted_total` | Engine sequences submitted, counted at admission |
| `chop_kv_tokens`, `chop_kv_capacity_tokens` | Tokens the KV page pool holds now, and the tokens it can hold: its pages times the tokens of a page, a context-parallel group's stripe. `vllm:kv_cache_usage_perc` is the same pool's used pages over its pages |
| `chop_request_decode_tokens_total{model_name,engine}`, `chop_request_decode_seconds_total{model_name,engine}` | Generated tokens after each sequence's first, and the seconds between each sequence's consecutive outputs, counted as outputs commit. Their ratio is the time-weighted decode speed per sequence: the terms `vllm:request_time_per_output_token_seconds` divides, without waiting for sequences to finish |
| `chop_spec_{verify_items,emitted_tokens}_total` | Verify items that checked drafts, and the tokens they emitted after stop and length cuts; published once a draft is verified |
| `vllm:batch_refusals_total{model_name,engine}` | Batches safely refused before device work, including the refusal that trips the loop guard |
| `vllm:batch_refused_requests_total{model_name,engine}` | Requests failed in those refused batches, counted once per sequence |
| `chop_requests_waiting`, `chop_requests_waiting_by_stage` | Requests waiting before they run, by stage ([request surface](serving-request-surface.md)) |
| `chop_transport_buffer_budget_bytes`, `chop_transport_buffer_reserved_bytes`, `chop_transport_read_bytes_total` | Connection read-buffer budget, its reservations, and bytes read |
| `chop_media_host_{limit,preparing,payload}_bytes`, `chop_media_host_cached_idle_{images,bytes}` | The process's media preparation budget, the bytes preparations hold beyond their payloads, the bytes prepared payloads keep (cached ones included), and the cached images no request holds. They bound accounted media buffers, not resident memory ([request surface](serving-request-surface.md)) |
| `chop_media_refusals_total{reason}` | Media requests refused by the backend: `context` (a prompt of inline images over the context, before decoding), `memory` (media over the request's preparation memory) and `codec` (a source that could not be read, probed or decoded, or outside its modality's limits) |
| `chop_process_uptime_seconds` | Seconds on the server's monotonic metrics clock, which starts when the listener opens; it resets when the server restarts |
| `chop_device_memory_{total,own,kv,reserve}_bytes{rank}` | Each rank's device memory when the serving KV pool was sized: the device's total, what this process's own allocations held before the pool, the pool, and the reserve kept for what is allocated after it ([GPU memory](serving-profiles.md#gpu-memory)) |
| `chop_device_memory_free_bytes{rank}`, `chop_device_memory_free_low_bytes{rank}` | Each rank's free device memory after the latest CUDA graph capture (or the sizing), and the lowest seen since the sizing. Graphs captured later come out of it |

`chop_step_gpu_seconds` is absent with the native bundle executor, which
reports no device time; host wall time is never substituted.

At startup the host logs one `serving KV memory rank R` line per rank as it
sizes the pool, and one `device memory after warmup rank R` line per rank
when warmup ends: the pool, the memory warmup's graphs took since the sizing,
and what is left. A rank left with less than 256 MiB also logs a warning,
since a graph captured later (a batch shape warmup missed, a GLM prefill
program) needs room there; `--graph-memory-reserve-mib` raises the room.

## Refused batches

A bundle can refuse a batch with `INVALID_ARGUMENT` before any device work. The
host then fails only that batch's requests, rolls back their KV and retires
their state; other requests keep running. Non-streaming callers receive HTTP
500 with up to 512 bytes of the bundle's last log; a stream that already sent
headers gets the same error as an SSE event and `[DONE]`. Any other execute
failure stops the worker for a supervisor restart.

By default the fourth consecutive refusal, or the eleventh within a rolling
60 seconds, also stops the worker. A committed batch resets the consecutive
count. Configure the limits before startup:

| Variable | Default | Range |
|---|---|---|
| `CHOPINFER_BATCH_REFUSAL_MAX_CONSECUTIVE` | 3 | 0 through 10000 |
| `CHOPINFER_BATCH_REFUSAL_MAX_IN_WINDOW` | 10 | 0 through 10000 |
| `CHOPINFER_BATCH_REFUSAL_WINDOW_SECONDS` | 60 | 1 through 86400 |

Zero for either count stops on the first refusal. Invalid values prevent
startup.

## Weight-loading counters

`startup direct_load` reports `preparation` (arena allocation and filled
host views) and `upload` (device image production and transfer) separately,
each with elapsed seconds and counter deltas. The top-level and `cumulative`
counters cover the image's lifetime, so do not divide them by upload seconds.
`CHOPINFER_DIRECT_LOAD_TRACK_RANGES=1` adds exact fetched-range accounting
(`unique_source_bytes`) at the cost of a shared lock per read, and
`CHOPINFER_DIRECT_LOAD_TIMING=1` adds lock and wait timings (`shared_timing`,
`startup weight_pipeline`) that overlap the other counters; leave both off
when comparing startup times. `tools/dev/startup_timing.py` summarizes a
startup log. Validated weight plans, metadata only, are reused from the model
cache's `weight-plans-v1` directory; `CHOPINFER_WEIGHT_PLAN_CACHE` names
another absolute private directory, or `0` disables reuse. Tensor payloads
are still read and validated on every start. Views mapped
from the checkpoint stay in the page cache, not pinned memory; `mapped_bytes`
and the `prewarm_*` fields state them, and `host-ram-short` warns when
`MemAvailable` cannot hold the prewarmed views plus a 16 GiB margin
(`CHOPINFER_PREWARM_MARGIN_BYTES`).

On 2026-10-21, the loader changes were compared with the round-8 serving
baseline using DeepSeek V4.1 Flash, TP4, release binaries, four GPUs capped at
250 W each, 48 CPUs and `--load-threads 16`. Both arms used prefix caching,
32 maximum sequences and 8192-token prefill chunks. Each condition used an
ABBA sequence with two starts per arm on an idle host and idle selected GPUs.
Readiness includes the same model construction and warmup in both arms.

| Profile | Checkpoint cache condition | Baseline mean (range), seconds | Loader mean (range), seconds | Mean reduction |
|---|---|---:|---:|---:|
| `tp4-ar` | Cold | 105.9 (104.3-107.5) | 94.4 (90.0-98.9) | 10.8% |
| `tp4-ar` | Recent restart | 106.3 (104.8-107.8) | 96.9 (95.9-98.0) | 8.8% |
| `tp4-dspark` | Cold | 145.1 (141.9-148.3) | 135.5 (130.0-141.1) | 6.6% |
| `tp4-dspark` | Recent restart | 148.3 (144.8-151.7) | 131.8 (131.4-132.2) | 11.1% |

Cold runs evicted only checkpoint files with `POSIX_FADV_DONTNEED` and
recorded resulting residency. Recent restarts followed a priming startup;
the text loader still read almost all of its approximately 339 GB through
direct I/O. Both conditions retained validated metadata-plan cache entries.
These are paired observations with two samples per arm, including the
variation shown above. The earlier 95.9-second diagnostic startup used
different cache conditions and is not the baseline for these percentages.

Mean text-weight production and upload fell from 80.9 to 73.1 seconds for
cold AR, 81.3 to 75.8 for recent AR, 83.6 to 77.5 for cold DSpark and 87.4
to 74.2 for recent DSpark. Cached text planning and validation took 1.8-1.9
seconds, versus 5.4 seconds without reuse. DSpark warmup remained about
38 seconds. The largest server peak RSS observed by readiness was 9.26
versus 8.91 GiB for AR and 15.09 versus 14.78 GiB for DSpark. Reading,
preparing and uploading text weights remains the largest loading phase.

Separate diagnostic starts matched all eight arena digests in each profile
and all token IDs and log-probability bits for three fixed 32-token responses
per profile. An unequal-rank upload probe checked all 208,666,687 bytes on
four GPUs. Diagnostic readback and detailed timing counters were disabled in
the performance runs. Every job passed its before/after power and Xid checks
and released the GPUs. The quick and full CPU gates also passed.

### Package reuse and packed-input follow-up

A second comparison on 2026-10-21 used the round-9 serving baseline, including
its wider 97-shape DSpark warmup. Both arms used the same serving code and
the TP4 settings above. The follow-up reuses a verified direct package before
emission, shares an owned checkpoint-header snapshot during preparation,
reuses scale inputs across packed sections, omits unused packed outputs and
reads cold input windows directly into owned aligned storage. Exact BF16
operand admission remains mandatory, including code-dependent checks for
extreme scales. Ordinary exact scale ranges avoid expanding a temporary
floating-point grid.

| Profile | Cache condition | Baseline mean (range), seconds | Follow-up mean (range), seconds | Mean reduction |
|---|---|---:|---:|---:|
| `tp4-dspark` | Cold payload, reused metadata | 137.5 (137.2-137.8) | 117.5 (117.4-117.6) | 14.5% |
| `tp4-dspark` | Recent restart | 140.4 (137.3-143.4) | 120.3 (120.3-120.4) | 14.3% |
| `tp4-dspark` | First package and plan, cold payload | 141.0 (139.6-142.4) | 132.7 (132.5-132.8) | 5.9% |
| `tp4-ar` | Recent restart | 94.5 (93.4-95.6) | 76.6 (76.5-76.8) | 18.9% |

Each condition used four starts in ABBA order on a quiet host, two per arm.
Cold and first-preparation starts recorded zero resident checkpoint pages
before launch after file-scoped eviction. Recent restarts followed a priming
start and had 52-84 MiB of checkpoint pages resident, far less than the model.
The first-preparation condition used a new private package and metadata-plan
cache for each start. Diagnostic readbacks and detailed timing counters were
disabled. These percentages compare only this matched serving baseline;
the earlier round-8 figures used different warmup coverage.

The recent-restart DSpark means break down as follows. The two residual rows
subtract nested timers from their enclosing elapsed timer; they do not name
individual functions. Reading, packing and transfers overlap within the
weight pipeline and cannot be added as separate elapsed phases.

| Startup work | Baseline, seconds | Follow-up, seconds |
|---|---:|---:|
| Package preparation/reuse | 12.1 | 0.5 |
| Other host startup, residual | 2.3 | 1.6 |
| Weight planning and validation | 1.9 | 2.0 |
| Text-weight read, prepare and upload | 75.7 | 67.5 |
| Encoder weight upload | 0.5 | 0.6 |
| Other model initialization, residual | 2.3 | 2.3 |
| Generation warmup | 45.5 | 45.9 |
| Total readiness | 140.4 | 120.3 |

First package preparation fell from 12.0 to 7.8 seconds; uncached host weight
planning still took about 6.2 seconds in both arms. A valid package hit skips
emission and topology planning, but still verifies its inputs and package.
One package continues to serve its supported TP2 and TP4 topologies. Actual
checkpoint dimensions, headers and precision contracts determine the layout.

Text-weight source reads fell from 339.0 to 321.7 GB, and loader requests from
143,948 to 96,995. The largest retained shared-input/output allocation across
these timing runs rose from 0.58 to 0.80 GiB, within the unchanged 1 GiB cap.
The idle buffer pool stayed within 64 MiB, and two 64 MiB pinned upload buffers
per rank remain the limit. The largest server peak RSS observed by readiness
was 16.59 versus 16.85 GiB for DSpark and 8.92 versus 8.79 GiB for AR.
These are server process peaks, not aggregate peaks of exited preparation
helpers. Live-process major faults were 1,127-1,128 in cold DSpark starts and
zero in the recent-restart comparisons.

Separate diagnostic starts matched all eight weight-arena digests and all
1,088 token IDs and selected/top log-probability positions per profile,
including 62 concurrent requests. DSpark captured the same 1,536 startup
graph keys in both arms. The baseline graph cache was full: both arms refused
100 capture attempts covering the same 13 keys. Concurrent serving formed
different batches and therefore different later refusal sets, including
between two executions of the unchanged baseline. Those dynamic refusals
were recorded separately; exact startup coverage, no new serving captures,
served bits, capture/release failures and pressure checks remained required.
AR matched all 24 startup graph keys. The unequal-rank upload probe and five
shared-format loading fixtures also passed. Every job checked the 250 W caps,
Xid records and device release.

Shorter per-shape warmup requests were rejected after GPU testing: some
requests ended before an accepted draft exercised the required catch-up
graphs. The retained change preserves the original output budgets and full
warmup work. Larger source-read windows were also not retained: 4, 8 and
16 MiB windows each delivered about 6.8 GB/s in a bounded 16 GiB source-only
probe with four readers. This is a sampled storage rate, not an end-to-end
startup guarantee.

A later same-candidate ABBA sweep compared 16 with eight loading workers per
rank. Readiness averaged 109.1 versus 111.9 seconds; eight workers were 2.6%
slower despite using about 0.9 GiB less peak server memory. All arena hashes,
startup graph coverage and served bits matched. The 16-worker default for
this CPU/rank configuration is retained. These runs used a separate quiet
window, so their absolute times should not replace the paired baseline
comparison above.

## Readiness and shutdown

The listener opens after engine construction and warmup. Warmup has one
180-second deadline; remaining graphs are captured on first use. `/health`
means worker readiness, not complete graph coverage. Allow 600 seconds from
process launch for cold verification, preparation, loading and warmup.
Docker marks an unhealthy container but does not restart it for that alone.

### Serving identity

`/health` and `/ping` expose status, modalities and a `serving` object with
bundle, package, checkpoint and metadata identities, startup nonce and the
effective profile. Checkpoint identity covers safetensors names, headers
and lengths; byte verification is separate. `kv_page_tokens` and the declared
context-parallel size determine prefix reuse stripes. When the profile's
`kv_cache_dtype` is `mixed`, `kv_cache_planes` lists each plane's name, code
(`bf16`, `fp8_e4m3` or `fp4_e2m1`), block scale and bytes per token; other
elements omit it. Startup logs repeat
the identity. Responses carry the startup nonce and, for direct checkpoints,
checkpoint identity in headers. Health/ping do not require an API key.

### Drain and shutdown

SIGTERM/SIGINT stops admission and makes health unready. During draining,
health and new requests return 503 with `Retry-After: 1`; metrics remain
available. Admitted work runs until completion or the drain deadline
(`--drain-timeout-ms`, default 30 seconds). At the deadline, unfinished streams
receive an error and `[DONE]` within a 100 ms flush window, then the process
releases GPUs and exits. A second signal exits immediately. Keep the
supervisor stop timeout above the drain budget, as in the 45-second examples.
A fatal engine/rank error exits nonzero without native teardown; restart a
fresh process and remove unready endpoints from load balancing.

## Watch running instances with chopinfer-top

`chopinfer-top --help` describes discovery, explicit URLs and display options.
It reads health and metrics; unavailable queries must not be interpreted as
zero usage. Its GPU discovery is local and does not establish that a device
is idle. Use the served identity to match an instance to its configuration.

The monitor (`m`) shows the local host beside its GPUs: CPU, memory, pressure,
disk and network I/O from `/proc`, sensors from `/sys/class/hwmon`, and the
used share of each local filesystem. In a container, CPU, memory, disk I/O
and sensors are the host's, while filesystems are the container's mounts and
network I/O is the container's namespace, shown as not visible when it has no
physical interface. CPU package power needs read access to RAPL, which is
usually root only.

## Example alerts

Tune thresholds to the model and traffic; keep `model_name` when grouping a
multi-model fleet.

```yaml
groups:
  - name: chopinfer
    rules:
      - alert: ChopinferWorkerFailed
        expr: increase(chopinfer_supervisor_restarts_total{service="chopinfer"}[10m]) > 0
      - alert: ChopinferWorkerFailureObserved
        expr: chop_generation_worker_failed == 1 or chop_instance_poisoned == 1
      - alert: ChopinferUnavailable
        expr: up{job="chopinfer"} == 0 or absent(up{job="chopinfer"})
        for: 10m
      - alert: ChopinferStartupDeadline
        expr: chopinfer_supervisor_ready{service="chopinfer"} == 0 and (time() - chopinfer_supervisor_start_time_seconds{service="chopinfer"} > 600)
      - alert: ChopinferSupervisorCollectorMissing
        expr: absent(chopinfer_supervisor_restarts_total{service="chopinfer"})
        for: 1m
      - alert: ChopinferQueueBacklog
        expr: vllm:num_requests_waiting > 32
        for: 2m
      - alert: ChopinferSlowFirstToken
        expr: histogram_quantile(0.99, sum by (le, model_name) (rate(vllm:time_to_first_token_seconds_bucket[5m]))) > 5
        for: 5m
      - alert: ChopinferAdmissionRefusals
        expr: sum by (instance, reason) (rate(chop_admission_refusals_total{reason="capacity"}[5m])) > 1
        for: 2m
      - alert: ChopinferIntakePressure
        expr: chop_intake_used_bytes / chop_intake_budget_bytes > 0.9
        for: 1m
```

`ChopinferWorkerFailed` requires an independent, always-running supervisor
collector. The `chopinfer_supervisor_*` series above are site collector
contracts, not metrics served by Chopinfer. Export a durable restart counter
(including failed starts, retained across child restarts and container
replacement), the current start timestamp, and readiness from `/health`, with
one stable service label per deployment. Collect Docker restart state or
systemd restart events outside the worker, and alert on collector failure too.

The worker gauges are best-effort: they can vanish before a scrape and cannot
reliably detect a fatal exit. `up` catches an unreachable target; `absent`
catches a missing job. The 600-second startup allowance matches the
deployment readiness budget ([deployment](developer-preview.md)).

Prefix hit rate is `rate(vllm:prefix_cache_hits_total[5m]) /
rate(vllm:prefix_cache_queries_total[5m])`. Speculative acceptance rate is
`rate(vllm:spec_decode_num_accepted_tokens_total[5m]) /
rate(vllm:spec_decode_num_draft_tokens_total[5m])`. Both are undefined without
traffic; do not alert on that.

## Trace a slow request

Set `CHOPINFER_REQUEST_TRACE=1` before starting the server. Each terminal
engine sequence then writes one JSON line to stderr with
`schema="chopinfer.request.v1"`:

```json
{"schema":"chopinfer.request.v1","request_id":"req-123_0","engine_request_id":7,"prompt_tokens":100,"cached_tokens":80,"queue_seconds":0.01,"ttft_seconds":0.04,"e2e_seconds":0.2,"generation_tokens":10,"finish_reason":"length","speculation":{"draft_tokens":12,"accepted_tokens":6}}
```

Search for the response's `x-request-id` with its choice suffix (`_0`, `_1`,
...). Records contain no prompt text, generated text or token IDs; token
counts include hidden tokens. TTFT is null when no token was emitted, and
error records have `finish_reason="error"`. A request refused before engine
admission has no trace. Logging is synchronous, so enable it only when its
per-request cost is acceptable.

## Measure target row usage

`chop_step_rows_total` counts committed decode and verify steps by target-row
geometry, with no device timing or per-step logging. `verify_rows` sums each
verify item's realized draft length plus its seed row, rejected drafts
included; `decode_rows` counts co-batched AR rows; live rows are their sum.
`row_capacity` is the executed graph bucket, or `unknown`. Bundle diagnostic
bucket overrides (for example `CHOPINFER_MIMO_VERIFY_BUCKET`) are not reported
in the capacity label, so leave them unset for these measurements.
Prefill-bearing steps are excluded.

`tools/qual/spec_bench.py` records the histogram's delta for each cohort in
`speculation.step_rows`; run the cohort after warmup with no other traffic.

## Trace scheduler steps

Set `CHOPINFER_STEP_TRACE=1` before starting `chopinfer serve` to write one JSON
line per committed step to stderr, schema `chopinfer.step.v1`. It is meant for
short tuning runs. Trace lines go through a bounded queue (4,096 lines): when
it is full, new lines are dropped and counted, never waited on, so a trace
with drops is incomplete. Startup, error and fatal messages are never dropped.
Step and request logs are not replay inputs; they omit token contents.

```json
{"schema":"chopinfer.step.v1","batch":812,"prefill_tokens":2048,"prefill_items":1,"decode_rows":31,"verify_items":0,"verify_rows":0,"drafted_tokens":0,"accepted_tokens":0,"verify_emitted_tokens":0,"live_rows":31,"row_capacity":null,"prefill_cap":2048,"execution_us":176210,"completed_unix_ns":1792224301138000123,"predicted_us":175300,"decode_estimate_us":16400,"trained":true,"host":{"outside_us":61,"schedule_us":38,"execute_us":176020,"commit_us":152,"gap_us":270,"period_us":176390,"boundary_us":431,"device_wait_us":175640,"upload_us":24,"launch_us":95,"sampling_us":180,"deferred_us":0}}
```

- `prefill_tokens`, `prefill_items`, `decode_rows`, `verify_items`,
  `verify_rows`, `live_rows`, `row_capacity`: the step's composition, with
  the row definitions above; capacity is null for prefill-bearing steps.
- `drafted_tokens`, `accepted_tokens`, `verify_emitted_tokens`: summed over
  the step's verify items with at least one proposal: proposals verified,
  proposals accepted, and tokens emitted (accepted proposals plus the bonus
  token, unless a stop or the length limit ended the run first). All three
  are 0 in steps without verification.
- `execution_us`: the wall time `chop_step_execution_seconds` records.
- `completed_unix_ns`: when the step completed, read on the engine thread
  (nanoseconds since the Unix epoch). Lines reach stderr through the queue,
  so align steps with client timings by this field, not by when a line is
  read.
- `prefill_cap`, `predicted_us`, `decode_estimate_us`, `trained`: the
  [mixed scheduling](serving-profiles.md#prefill-policy) cap and estimates; null under the
  fixed budget or before an estimate exists.
- `host`: the engine thread's wall time per phase in microseconds
  (scheduling, execute, commit, gaps between executor calls, and
  `boundary_us`, an upper bound on device idle time between steps). Phases
  nest, so do not sum them.
- Within `host.commit_us`, `scheduler_commit_us` measures scheduler preflight
  and apply, `model_commit_us` measures model transaction finalization
  (including kept-row conversion), and `publication_us` measures applying
  prepared tokens and publishing engine events and counters. Output checks,
  sample staging, event preflight and later estimator updates remain outside
  those three sub-phases. `sampling_us` includes sampling both inside and
  after execute, so it is not wholly contained in `commit_us`.
  The new scopes read clocks only while step tracing is enabled. They add
  no device synchronization and do not change tokens or event order.

## Inspect sampled steps

Set `CHOPINFER_STEP_PROFILE_EVERY=N` before startup to sample every Nth engine
step (0 or unset disables it). `CHOPINFER_STEP_PROFILE_CAPACITY` (default 256,
1 through 4096) bounds the retained samples. `GET /debug/step-profile` returns
schema `chopinfer.step_profile.v1` with the rate, capacity and samples oldest
first, or 404 when disabled. Each sample has `step`, `total_us`,
`host_scheduling_us`, `upload_us`, `graph_launch_us`, `device_wait_us`,
`sampling_us` and `deferred_us`, all host wall microseconds; graph launch is
the host API call, not graph execution. Values nest and must not be summed.
Sampling adds no device synchronization.
The samples also expose `scheduler_commit_us`, `model_commit_us` and
`publication_us`; these three fields are populated only when
`CHOPINFER_STEP_TRACE=1` is enabled alongside sampling.

## Capture a serving window with Nsight Systems

On idle GPUs, run the server under `nsys profile` with a delayed, finite
capture window. Set the shell array `server_command` to the server's usual
command and environment, choose a delay longer than loading and warmup, and
send the measured requests during the window:

```bash
profile_prefix="$CHOPINFER_SCRATCH/serving-$(date +%Y%m%dT%H%M%S)"
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node \
  --sample=none --cpuctxsw=none --delay=120 --duration=30 \
  --kill=sigterm --export=sqlite --output="$profile_prefix" \
  "${server_command[@]}"
```

The server is terminated when the window ends; check that request and kernel
activity fall inside the window before reading timings.
