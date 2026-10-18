# Metrics, request traces and step profiles

`chopinfer serve` publishes Prometheus text at `/metrics` without
authentication. Scrape each process separately, every 15 seconds on the
private listener. Native engine and KV metrics come from the worker that owns
their state; HTTP refusal and intake metrics come from the listener. Label
domains are bounded and request IDs never become labels. Counters advance only
after the corresponding transition commits and reset on restart. For
behavioral or numerical diagnosis, use the [first-divergence workflow](debugging.md).

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
| `chop_mixed_budget_term_seconds{model_name,engine,term}`, `chop_mixed_budget_samples`, `chop_mixed_budget_trained`, `chop_mixed_budget_decode_estimate_seconds{items_le}` | Tbt mode only: the fitted fixed, mixed, per-token and per-attention-pair terms, prefill steps in the fit window, whether the fit sets budgets, and the decode estimate per power-of-two item bucket ([mixed scheduling](mixed-scheduling.md)) |
| `chop_prefix_cache_{queries,hits}_total`, `chop_prefix_cache_{queried,reused}_tokens_total` | Prefix-cache lookups and restored hits, and their prompt and reused tokens |
| `chop_prefix_cache_recomputed_seen_tokens_total` | Prompt tokens a committed prefill computed although an earlier request's committed prefill had computed the same leading full pages, short of the last prompt token: repeats and shared prefixes no anchor covered ([where anchors are captured](serving-kv-pool.md#where-prefix-anchors-are-captured)) |
| `chop_prefix_cache_{captures,capture_failures,restore_failures,evictions}_total`, `chop_prefix_cache_{anchors,blocks}` | Anchor captures, captures and restores the bundle refused, evictions, and the anchors and pages held now |
| `chop_errors_total`, `chop_cancellations_total` | Engine errors and cancelled sequences |
| `vllm:batch_refusals_total{model_name,engine}` | Batches safely refused before device work, including the refusal that trips the loop guard |
| `vllm:batch_refused_requests_total{model_name,engine}` | Requests failed in those refused batches, counted once per sequence |
| `chop_requests_waiting`, `chop_requests_waiting_by_stage` | Requests waiting before they run, by stage ([request surface](serving-request-surface.md)) |
| `chop_transport_buffer_budget_bytes`, `chop_transport_buffer_reserved_bytes`, `chop_transport_read_bytes_total` | Connection read-buffer budget, its reservations, and bytes read |
| `chop_process_uptime_seconds` | Seconds on the server's monotonic metrics clock, which starts when the listener opens; it resets when the server restarts |

`chop_step_gpu_seconds` is absent with the native bundle executor, which
reports no device time; host wall time is never substituted.

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

`startup direct_load` separates `preparation` (arena allocation and filled
host views) from `upload` (device image production and transfer). Each has
elapsed seconds and counter deltas. The legacy top-level counters and the
`cumulative` object cover the image's lifetime; do not divide those totals
by upload seconds when preparation also read weights. Directly mapped
checkpoint views and optional prewarming are separate from filled host views.

`requested_source_bytes` includes retained input hits; `source_bytes` counts
requested ranges fetched from files, and `file_read_bytes` also includes
direct-I/O alignment. Buffered reads may come from the page cache.
`CHOPINFER_DIRECT_LOAD_TRACK_RANGES=1` enables exact fetched-range accounting
for diagnostics. It takes a shared lock per source read and is off by default.
`source_ranges_tracked` reports whether it is enabled; `unique_source_bytes`
is zero when disabled. When enabled, phase deltas count newly encountered
bytes, not visits to an earlier phase's ranges. Source-cache hits
and shared-output reproductions distinguish input reuse from repeated packing.
Read and production seconds sum worker time and overlap each other.

Packed inputs share the configured retention budget across ranks. Idle
packing buffers retain at most 64 MiB, capped further by that budget;
`retained_peak_bytes` and `pooled_peak_bytes` report these separate lifetime
high-water marks. Active worker temporaries and pinned upload buffers are
additional memory. The loader releases retained inputs and idle buffers
after upload.

## Watch running instances with chopinfer-top

`chopinfer-top` is a terminal view of the `chopinfer serve` instances on a
host: what each serves, whether it is ready, and how it is doing. It is a
separate binary, shipped in `bin/` beside `chopinfer` and in the image. It is
a client of `/health` and `/metrics` only, which need no API key, and it links
no serving code. The flags are in [operator commands](cli.md#live-view-chopinfer-top).

```text
chopinfer-top                          # find instances and watch them
chopinfer-top --target HOST:8311       # also watch a server it cannot find
chopinfer-top --list --json            # one reading, for scripts
```

### Where instances come from

| Source | What it finds | Limits |
|---|---|---|
| Explicit targets | `--target HOST:PORT` (repeatable; also `http://HOST:PORT`, or `:PORT` for 127.0.0.1), `--targets-file FILE` (one per line, `#` starts a comment) and the comma-separated `CHOPINFER_TOP_TARGETS` | Stay listed while unreachable and are retried every 5 s. `--api-key KEY` (default `CHOPINFER_API_KEY`) sends a bearer token to these only, for a server behind an authenticating proxy |
| `/proc` | `chopinfer` and `chopinfer-host` processes whose command line runs `serve`, owned by this user (every user's when run as root) and in chopinfer-top's network namespace | Processes in another namespace, such as a container's, are left to Docker. The listening sockets come from the process's descriptors; a wildcard bind is reached on loopback |
| Docker | Running containers whose process list shows `chopinfer serve` or `chopinfer-host serve`. The daemon lists the processes on its host side, so any image works and nothing runs inside the container | Needs a daemon socket this user can open. Only `/_ping`, the container list and each container's process list are read |

Discovery runs alongside explicit targets; `--no-docker` turns off containers
and `--no-discover` turns off both scanners. Rows that report the same startup
nonce are one server and are shown once: an explicit target first, then a
process, then a container, with the others listed as "also seen as".

Docker discovery uses `DOCKER_HOST` alone when it is a `unix://` socket. Any
other `DOCKER_HOST` scheme turns container discovery off, with a note.
Without `DOCKER_HOST`, it tries the rootless socket
`$XDG_RUNTIME_DIR/docker.sock` and the system socket `/var/run/docker.sock`,
and uses each one that answers `/_ping`. A daemon that does not answer, or
stops answering, is tried again every 60 seconds. A container that does not
run serve is checked on every scan for its first two minutes, then every five
minutes.

A container is reached through the `--listen` address on its serve command
line (`127.0.0.1:8310` when absent): with host networking at that address,
otherwise at a host port published for the listen port, and failing that at
the container's own network addresses on the listen port (IPv4 for an IPv4
listener, IPv6 for an IPv6 one, both for `[::]`). A server that listens on
the container's loopback cannot be reached from the host; the row shows as
unreachable and an event says so once. Its process list is read again every
minute, so the row fills in once serve listens where the host can reach it.
Listen on `0.0.0.0` and publish the port, as the image does.

The systemd units run serve as the `chopinfer` user, so another user's `/proc`
scan does not count them. Pass `--target 127.0.0.1:PORT` for each unit, or
run chopinfer-top as root.

### When it scans

A full scan runs at startup, before the first frame, and then every
`--rescan` seconds (default 15, from 2 to 300). An instance that stops
answering or restarts asks for a scan sooner, at most one every 5 seconds. A
process found without a listener has only its own sockets checked every 2
seconds, so its row fills in soon after it opens one. `r` runs a full scan at
once. Discovery only has to find new instances: a restart or a different model
at an address already watched shows up in that instance's poller within one
poll, through the new startup nonce. At most 64 instances are polled at once;
an event reports when there are more, and the rest are polled as others go.

### States

serve opens its listener only after loading and warmup, so a discovered
process appears as starting well before it answers. The header counts rows by
state.

| State | Meaning |
|---|---|
| connecting | An explicit target not polled yet |
| starting | Discovered, not answering yet: loading weights, warming up, or listening for under 30 s without a valid answer |
| not answering | A process has listened for over 30 s without a valid answer |
| ready | `/health` is ready and requests are running or waiting |
| ready idle | `/health` is ready with nothing running or waiting |
| not ready | `/health` is not ready or unreadable; the summary, the details header and the events show its reason |
| unreachable | An instance that answered before stopped answering, an explicit target never answered, or a container has no address the host can reach |
| not chopinfer | Something answers without the `x-chopinfer-startup-nonce` header: another server, or a proxy that strips it |
| failed | `chop_generation_worker_failed` is 1 |
| poisoned | `chop_instance_poisoned` is 1 |

### What the numbers are

Each poll reads `/health`, then `/metrics`, and keeps the metrics only when
both carry the same startup nonce, so a restart between the two requests
cannot mix two processes' counters. The list shows one row per
instance and a five-minute decode sparkline; Enter opens the details of one
instance. The selected instance's decode and prefill plots under the list are
scaled to a round number at or above their largest value, shown at their top
right, so a small burst does not fill the plot. The details fit the terminal:
a smaller one shows fewer panels, and throughput, load, latency and health
stay the longest. Identity (bundle, profile, ranks, KV dtype, served names, version,
checkpoint and startup nonce) comes from the `serving` member of `/health`.

| Shown | Series |
|---|---|
| Decode tok/s | Rate of `vllm:generation_tokens_total` |
| Prefill tok/s | Rate of `chop_step_prefill_tokens_sum` (prompt tokens computed by committed prefill-bearing steps) |
| Running, waiting | `vllm:num_requests_running`, `vllm:num_requests_waiting`; sequences are shown against `max_sequences` from `/health` |
| KV | `vllm:kv_cache_usage_perc` |
| Intake | `chop_intake_used_bytes` over `chop_intake_budget_bytes` |
| Requests/s | Rate of the sum of `vllm:request_success_total` |
| Prefix: reused / queried tokens | `vllm:prefix_cache_hits_total` over `vllm:prefix_cache_queries_total` (the list's PREFIX column) |
| Prefix: lookups that hit | `chop_prefix_cache_hits_total` over `chop_prefix_cache_queries_total` |
| Prefix: reused tok/s, recompute seen tok/s, evictions/s | Rates of `vllm:prefix_cache_hits_total`, `chop_prefix_cache_recomputed_seen_tokens_total`, `chop_prefix_cache_evictions_total` |
| Prefix: anchors / pages, capture / restore failures | `chop_prefix_cache_anchors`, `chop_prefix_cache_blocks`; `chop_prefix_cache_capture_failures_total`, `chop_prefix_cache_restore_failures_total` |
| Accepted drafts | `vllm:spec_decode_num_accepted_tokens_total` over `vllm:spec_decode_num_draft_tokens_total` (the list's SPEC column) |
| Acceptance by position | `vllm:spec_decode_num_accepted_tokens_per_pos_total` over `vllm:spec_decode_num_draft_tokens_per_pos_total` at each `position` |
| K chosen | Shares of `chop_speculation_k_total` by `k` |
| Latency p50, p90, p99 and count | `vllm:time_to_first_token_seconds` (the list's TTFT50 column), `vllm:inter_token_latency_seconds`, `vllm:e2e_request_latency_seconds`, `vllm:request_queue_time_seconds`, `vllm:request_prefill_time_seconds`, `vllm:request_time_per_output_token_seconds` |
| Batches/s and batch mix | Rate of `chop_batches_total`; `chop_decode_only_batches_total`, `chop_mixed_batches_total`, `chop_prefill_only_batches_total` |
| Step time p50, p99 | `chop_step_execution_seconds` by `phase` (`decode`, `mixed`, `prefill`) |
| Requests since start | `vllm:request_success_total` by `finished_reason`, `chop_cancellations_total`, `vllm:num_preemptions_total`, `chop_errors_total`, `chop_admission_refusals_total` by `reason` |
| Worker, poisoned, batch refusals | `chop_generation_worker_{starting,ready,failed,stopped}`, `chop_instance_poisoned`, `vllm:batch_refusals_total` |
| Transport buffer | `chop_transport_buffer_reserved_bytes` over `chop_transport_buffer_budget_bytes` |
| Uptime | The process age from `/proc` for a discovered process, else `chop_process_uptime_seconds` |

Rates span the last three polls. Ratios, shares, the batch mix and latency
quantiles cover the selected window: 60 s, 5 m or since start, which `w`
cycles. They come from counter and histogram bucket deltas between the newest
poll and the one at the window's start, timed on `chop_process_uptime_seconds`
when every poll has it. History holds five minutes of polls. When the server
has run for less than the window, the figures cover its whole life; when
chopinfer-top has watched it for less than the window, they cover what it has
seen, and the window label says how much (`window 5m (40s of data)`). A new
startup nonce, or a counter or the server clock going backwards, is a restart:
the history is dropped, the counters are re-baselined and the restart is
counted.

### Keys and scripted use

| Key | Action |
|---|---|
| Up, Down, `j`, `k` | Select an instance (list) |
| Enter, double click | Open its details; a click selects |
| Esc, Backspace, `h` | Back to the list |
| Right, Left, Tab, Shift-Tab, `l` | Next or previous instance (details) |
| `w` | Window: 60 s, 5 m, since start |
| `r` | Rescan for processes and containers now |
| `p` | Pause the screen; polls keep their latest result |
| `?` | Help; any key closes it |
| `q`, Ctrl-C | Quit |

The interactive view needs a terminal; without one chopinfer-top exits with
status 2 and points to `--list`. `--list` runs one discovery pass and two
polls one interval apart, prints a table (source, address, model, state,
decode and prefill tok/s, running/waiting, KV) and writes warnings to
stderr. `--list --json` prints a `chopinfer.top.list.v1` document: an
`instances` array (source, pid or container or target, address, state and
detail, model, uptime, the `/health` identity, rates, gauges, ratios and
counter totals) and the `events` log. Its ratios use the 60 s window, which
for a server older than a minute covers only the interval between the two
polls.

### Polling cost

Each poll of an instance is one `/health` and one `/metrics` request. The
`/metrics` snapshot briefly takes the engine's metrics locks. The default
`--interval` is 1 s and the floor 0.5 s; an instance that does not answer is
polled every 5 s. A few dashboards on one server cost nothing measurable.
chopinfer-top keeps no history beyond five minutes and raises no alerts:
scrape with Prometheus for alerting and dashboards, as below.

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
deployment readiness budget ([deployment](deploy-common.md)).

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
{"schema":"chopinfer.step.v1","batch":812,"prefill_tokens":2048,"prefill_items":1,"decode_rows":31,"verify_items":0,"verify_rows":0,"live_rows":31,"row_capacity":null,"prefill_cap":2048,"execution_us":176210,"completed_unix_ns":1792224301138000123,"predicted_us":175300,"decode_estimate_us":16400,"trained":true,"host":{"outside_us":61,"schedule_us":38,"execute_us":176020,"commit_us":152,"gap_us":270,"period_us":176390,"boundary_us":431,"device_wait_us":175640,"upload_us":24,"launch_us":95,"sampling_us":180,"deferred_us":0}}
```

- `prefill_tokens`, `prefill_items`, `decode_rows`, `verify_items`,
  `verify_rows`, `live_rows`, `row_capacity`: the step's composition, with
  the row definitions above; capacity is null for prefill-bearing steps.
- `execution_us`: the wall time `chop_step_execution_seconds` records.
- `completed_unix_ns`: when the step completed, read on the engine thread
  (nanoseconds since the Unix epoch). Lines reach stderr through the queue,
  so align steps with client timings by this field, not by when a line is
  read.
- `prefill_cap`, `predicted_us`, `decode_estimate_us`, `trained`: the
  [mixed scheduling](mixed-scheduling.md) cap and estimates; null under the
  fixed budget or before an estimate exists.
- `host`: the engine thread's wall time per phase in microseconds
  (scheduling, execute, commit, gaps between executor calls, and
  `boundary_us`, an upper bound on device idle time between steps). Phases
  nest, so do not sum them.

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
