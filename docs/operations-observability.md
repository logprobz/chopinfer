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
| `chop_admission_refusals_total{reason}` | Generation and stored-response HTTP errors by status class: `invalid_request`, `unknown_model`, `not_found`, `intake_timeout`, `capacity`, `unavailable`, `internal`, `other`. `unknown_model` counts generation 404s; `not_found` counts missing stored response IDs on read and cancel. Dashboards that treated all 404s as `unknown_model` should include `not_found`. Each refusal also writes one [request-failure line](#logs-and-log-levels); an unknown-model line names the requested model, the socket peer and the served names. Drain middleware refusals also count. Errors after SSE headers are outside this counter. |
| `chop_intake_budget_bytes` | Configured listener budget for request bodies |
| `chop_intake_used_bytes` | Currently charged bytes, including the 16 KiB minimum per request |
| `chop_instance_poisoned{model_name,engine}` | Sticky 1 after model commit/rollback, retirement or prefix-state failure makes reuse unsafe |
| `chop_generation_worker_{starting,ready,failed,stopped,kv_stalled}` | One-hot worker lifecycle state; `kv_stalled` is temporarily unready while overdue transfers retain their leases, and recovers when they complete. Engine summaries use the same state name |
| `chop_speculation_k_total{model_name,engine,k}` | Scheduled decode/verify items by selected K, including K=0; counts attempted work even if execution fails |
| `chop_step_gpu_seconds{model_name,engine}` | Histogram of executor-reported kernel time per measured execution, absent when no executor measurement exists |
| `chop_batches_total`, `chop_{prefill_only,decode_only,mixed}_batches_total` | Actual batch composition |
| `chop_step_execution_seconds{model_name,engine,phase}` | Histogram of engine wall time from execute through model commit per committed step; `phase` is `decode`, `mixed` or `prefill` by whether the batch had prefill items and decode or verify items |
| `chop_step_rows_total{model_name,engine,decode_rows,verify_rows,row_capacity}` | Exact count of committed decode/verify steps with this target-row geometry; excludes any prefill-bearing step. Capacity is `unknown` when the executor cannot report it |
| `chop_step_rows_dropped_total{model_name,engine}` | Observations of new geometries beyond the 4096-bin process limit; existing bins continue counting |
| `chop_prefill_policy_info{model_name,engine,policy}` | Gauge 1 for the canonical negotiated policy, before any step; omitted for unsupported library-only combinations |
| `chop_step_prefill_tokens{model_name,engine}` | Histogram of prompt tokens per prefill-bearing step |
| `chop_prefill_budget_selected_tokens{model_name,engine}` | `tbt` and `adaptive` only: histogram of the prefill cap selected for each step beside decoding requests, whether or not prefill used it; prefill-only steps select none |
| `chop_mixed_step_predicted_seconds{model_name,engine}`, `chop_mixed_step_actual_over_predicted{model_name,engine}` | `tbt` and `adaptive` only: histograms of each mixed step's predicted time, for the batch as packed, and of its execution time over that prediction |
| `chop_prefill_timing_cut_steps_total{model_name,engine}`, `chop_prefill_timing_cut_chunks_total{model_name,engine,end}`, `chop_prefill_timing_cut_forced_steps_total`, `chop_prefill_timing_cut_over_target_steps_total` | Timing cuts only (`--prefill-policy adaptive` on a grid bundle): steps whose grid chunks were sized against the tbt cap while requests decoded, those chunks by where they ended (`grid`, `capture`, `prompt`, `timing`, `floor` or `progress`), steps where a chunk exceeded the cap only to progress (neither the cap nor the floor reached a granule multiple within the step's room), and steps whose predicted time exceeded the target |
| `chop_mixed_budget_term_seconds{model_name,engine,term}`, `chop_mixed_budget_samples`, `chop_mixed_budget_trained`, `chop_mixed_budget_decode_estimate_seconds{items_le}` | `tbt` and `adaptive` only: the fitted fixed, mixed, per-token and per-attention-pair terms, prefill steps in the fit window, whether the fit sets budgets, and the decode estimate per power-of-two item bucket ([prefill policy](developer-preview.md#prefill-policy)) |
| `chop_prefix_cache_{queries,hits}_total`, `chop_prefix_cache_{queried,reused}_tokens_total` | Prefix-cache lookups and restored hits, and their prompt and reused tokens |
| `chop_prefix_cache_recomputed_seen_tokens_total` | Prompt tokens a committed prefill computed although an earlier request's committed prefill had computed the same leading full pages, short of the last prompt token: repeats and shared prefixes no anchor covered ([where anchors are captured](numerics.md#kv-offload-and-prefix-state)) |
| `chop_prefix_cache_recomputed_generated_tokens_total` | Whole generated pages from earlier completed replies that a later prompt successfully prefills; requires the optional reply-history observer |
| `chop_reply_history_{enabled,allocated_bytes,entries}` | Collection status, all observer capacities, and completed histories retained; disabled collection is not a zero-match observation |
| `chop_reply_history_observations_total{outcome}`, `chop_reply_history_rows_total{kind,context}` | Fixed-label reply comparison, coverage, finish and committed-row counts, described below |
| `chop_prefix_cache_{captures,capture_failures,restore_failures}_total`, `chop_prefix_cache_{anchors,blocks}` | Anchor captures, captures and restores the bundle refused, and the anchors and pages held now |
| `chop_prefix_cache_evictions_total{class}` | Public anchor evictions by `superseded`, `retained` or `maximal` class; sum all classes for the total |
| `chop_prefix_cache_public_anchors{class}` | Public anchors currently in each class |
| `chop_prefix_cache_snapshot_pages{class}` | Snapshot pages owned by those anchors; excludes pins surviving anchor removal |
| `chop_prefix_cache_pages_freed_total{class}` | Pages actually returned to the free list at eviction, including KV pages; shared or pinned pages count only when their last reference disappears |
| `chop_prefix_cache_private_anchors` | Owned private preemption anchors |
| `chop_prefix_cache_maximal_eviction_age_seconds` | Histogram of monotonic seconds since capture for public maximal-tier victims; bounds 1, 10, 60, 300, 1800, 3600, 7200, 86400 and infinity |
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
| `chop_device_memory_{total,own,kv,reserve}_bytes{rank}` | Each rank's device memory when the serving KV pool was sized: the device's total, what this process's own allocations held before the pool, the pool, and the reserve kept for what is allocated after it ([GPU memory](developer-preview.md#gpu-memory)) |
| `chop_device_memory_free_bytes{rank}`, `chop_device_memory_free_low_bytes{rank}` | Each rank's free device memory after the latest CUDA graph capture (or the sizing), and the lowest seen since the sizing. Graphs captured later come out of it |
| `chop_mapped_view_bytes` | Gauge of live mapped views in whole pages, including partial edge pages, not raw tensor bytes |
| `chop_mapped_view_resident_bytes` | Gauge of those whole pages resident in the host page cache; absent when residency is hidden |
| `chop_mapped_view_residency_visible` | Gauge: 1 when every mapped view's residency is visible, 0 when any is hidden |
| `chop_mapped_view_residency_sample_seconds` | Gauge of the latest successful residency sampling pass's elapsed seconds |

The four mapped-view gauges have no labels. They are absent without mapped
views or before the first sample. `chop_mapped_view_resident_bytes` is also
absent while residency is hidden; hidden residency never reports full
residency. The two byte gauges count whole pages, so their ratio is bounded
by 1.

`chop_step_gpu_seconds` is absent with the native bundle executor, which
reports no device time; host wall time is never substituted.

At startup the host logs one `serving KV memory rank R` line per rank as it
sizes the pool, and one `device memory after warmup rank R` line per rank
when warmup ends: the pool, the memory warmup's graphs took since the sizing,
and what is left. A rank left with less than 256 MiB also logs a warning,
since a graph captured later (a batch shape warmup missed, a GLM prefill
program) needs room there; `--graph-memory-reserve-mib` raises the room.

### Prefix cache accounting

These prefix-cache series are emitted in both eviction policies. The labelled
`chop_prefix_cache_evictions_total{class}` replaces the unlabelled series;
migrate dashboards to `sum(chop_prefix_cache_evictions_total)`. There is no
additional unlabelled total to double-count, and these series have no host
labels. With the candidate off, or for digest-only captures whose ancestry
is unknown, public anchors are counted in the maximal class. Private anchors
are excluded from the public class counters and gauges. With the candidate
on, the class is a label, not the eviction order: `superseded` anchors are
the first tier, `retained` covers both spaced ancestors (the second tier)
and shared or not yet passed ancestors (ordered by recency with the
`maximal` frontiers).

The total anchor gauge includes private anchors. The cached-block gauge
counts unique pages held only by anchors; class snapshot counts do not
partition physical occupancy. Live sequences, restore pins and transfer
leases can retain pages after an anchor disappears. `CHOPINFER_PREFIX_ANCESTRY=1`
enables the default-off ancestry eviction candidate, a candidate switch
rather than a serving option ([KV state](numerics.md#kv-offload-and-prefix-state)).

### KV offload and the RAM prefix tier

The `chop_kv_offload_*` series below, but for the byte counter and the
pending transfer gauge, appear only when the server copies through an
offload arena (`--kv-offload-gib` above 0 with a bundle that declares
`chop.kv-offload.v1`). The RAM prefix tier's series,
`chop_kv_offload_bytes_total` and `chop_kv_offload_pending_transfers` are
exported from startup. The tier's stay zero while it is off and the gauge
stays zero without an arena; the byte counter and the gauge count swaps too
([option and bounds](developer-preview.md#kv-offload-and-the-ram-prefix-tier)).
`chop_prefix_cache_hits_total` and `chop_prefix_cache_reused_tokens_total`
count every hit, RAM hits included; recomputation stays
`chop_prefix_cache_recomputed_seen_tokens_total`.

| Series | Meaning |
|---|---|
| `chop_kv_offload_swap_out_total`, `chop_kv_offload_swap_in_total` | Completed swap copies out of and into the GPU |
| `chop_kv_offload_copy_seconds{direction}` | Histogram of every completed copy's time, swaps and the tier's demotions (`out`) and promotions (`in`); bounds 0.0001, 0.001, 0.01, 0.1, 1 and infinity seconds |
| `chop_kv_offload_fallback_total{reason}` | Swaps that fell back to recomputation: `tier_full` (no host room), `capture` (no swap capture), `out_refused` and `in_refused` (the executor refused the copy), `out_failed` and `in_failed` (the copy failed with every rank idle), `restore` (the bundle refused a reloaded state), `unconfirmed` (a copy's completion could not be confirmed, which poisons the instance), `staging_capacity` (the restore needs more pages than the pool holds) and `unsupported` (counted by nothing now) |
| `chop_kv_offload_host_pages{kind}` | Host pages per rank: `used` and `capacity` |
| `chop_kv_offload_swapped_requests` | Requests whose state is swapped to the arena |
| `chop_kv_offload_pending_transfers` | Copies submitted and not yet finished, swaps and the tier's demotions and promotions; a cancelled demotion counts until its cancel settles. It changes with every submission, completion and settled cancel, so a reader that waits for the copies to end waits for it to read 0 |
| `chop_kv_offload_oldest_pending_seconds` | Age of the oldest copy still pending, measured when the engine polls its copies: a copy submitted since then is not counted, so 0 does not mean none is pending |
| `chop_kv_offload_stalled_transfers`, `chop_kv_offload_stalls_total` | Copies pending past twice the drain timeout now, and copies that ever became overdue; an overdue copy keeps its leases and makes the worker `kv_stalled` until it completes |
| `chop_kv_offload_bytes_total{direction,kind}` | Bytes completed copies moved across every rank, `direction` `out` or `in`, `kind` `swap` or `prefix` (the tier's demotions and promotions) |
| `chop_prefix_cache_ram_hits_total`, `chop_prefix_cache_ram_reused_tokens_total` | Hits restored from an anchor promoted from RAM, and their reused tokens |
| `chop_prefix_ram_demotions_total{result}` | Demotions by result: `ok`, `failed` (the copy failed), `cancelled` (an allocation needed the pages its lease held), `refused` (the executor refused the copy) and `tier_full` (no host room older than the anchor) |
| `chop_prefix_ram_promotions_total{result}` | Settled promotions: `ok` (published), `failed` (copy failed or never started), `invalidated` (a recapture, resume or descendant of the prefix made the copy stale) and `gpu_resident` (a GPU anchor of the prefix appeared first) |
| `chop_prefix_ram_missed_evictions_total{reason}` | Public GPU evictions without a ready RAM copy: `headroom` (the demotion start rule refused it, or its demotion was cancelled for an allocation), `in_flight` (another demotion was outstanding), `tier_full`, `failed`, `below_minimum` (shorter than `--kv-offload-min-tokens`), `not_frontier` (it had a resident descendant) and `unplanned` (outside the demotion lead when last planned) |
| `chop_prefix_ram_eviction_log_discarded_total` | Evictions the pool's bounded eviction log dropped before the engine read them, so counted neither covered nor missed |
| `chop_prefix_ram_headroom_waits_total` | Waits for a demotion's cancel that an allocation needed; the demotion counts `cancelled` when the cancel took effect and `ok` when its copy finished first |
| `chop_prefix_ram_cancel_timeouts_total` | Of those waits, cancels still pending at the executor's drain timeout; their pages stay leased until the copy drains |
| `chop_prefix_ram_fallbacks_total{reason}` | Requests that preferred a ready RAM match and resumed from their GPU match or recomputed: `pool_exhausted`, `capacity`, `failed`, `invalidated` and `gpu_resident` |
| `chop_prefix_ram_evictions_total` | RAM entries the tier evicted to make room |
| `chop_prefix_ram_anchors`, `chop_prefix_ram_bytes` | RAM entries now, and the bytes they hold across every rank |
| `chop_prefix_ram_restore_seconds` | Histogram of a RAM hit's time from the admission decision to its restore; bounds 0.001, 0.01, 0.1, 1, 10 and infinity seconds |

A covered eviction is one whose anchor had a ready RAM copy; the misses
above are the rest of the public evictions. Headroom waits and cancel
timeouts measure what demotion cost the foreground; acceptance reports them
beside event gaps and stalls.

### Generated reply observations

`serve --reply-history-observer on` enables bounded digest accounting. The
optional profile field `reply_history_observer` is a boolean; an explicit
flag overrides it, and omission resolves to off. Every shipped profile
keeps its serving defaults. Collection needs no request or step tracing,
no GPU reservation and no replay. Docker forwards the flag after its image;
systemd can carry it in the existing `CHOPINFER_SERVE_ARGS` argument list.
There is no new deployment environment variable.

This is observational: public anchors remain prefill-only. Prefix capture
boundaries, KV retention, scheduler admission and scheduling rules, sampling
arithmetic and output semantics are unchanged. Identical inputs and batch
schedules preserve output bytes. Additional host latency is measured, not
claimed absent. Neither this metric nor an exact match proves that reply-end
KV reuse would be correct.

The registry retains at most 256 completed histories. Its total capacity,
including at most 256 active observations and 256 delivery receipts, temporary prompt material,
annotations, indices and reserved buffers, is capped at 64 MiB per engine.
Oldest completed histories leave first, independently of KV eviction. A
request whose observation cannot fit continues without it. Capacity and
allocation drops are counted. Completed entries contain salted token/media
digests, lengths, completion order and numeric region masks. They retain no
text, token IDs, raw salts, prompt token storage or KV handles. The active
partial page is discarded at completion. Warmup does not enter live history.

`chop_reply_history_rows_total` counts committed prompt rows by `kind`
(`candidate`, `eligible`, `repeated`, `rounding`, `unobserved`, `ambiguous`
and `prefilled`) and context bucket (`0_32k`, `32_64k`, `64_128k`,
`128_192k`, `192k_plus`): rows of earlier generated replies that a later
prompt prefills again, and whether reuse at an exact reply end would have
been eligible. `chop_reply_history_observations_total{outcome}` classifies
each consumer (`no_history`, `scope_without_ancestry`, `before_reply`,
`inside_reply`, `final_token`, `complete_reply`, `shorter_consumer` and
finish and mismatch counts). All labels are fixed, with no salt hashes,
request IDs, token values or prompt-derived strings. The generated counter
and `chop_prefix_cache_recomputed_seen_tokens_total` may overlap and must
not be added. Collection has measured host cost, so it stays off by
default.

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
and the `prewarm_*` fields describe them. `prewarm_resident_bytes` is omitted
when residency is hidden. `host-ram-short` warns when
`MemAvailable` cannot hold the prewarmed views plus a 16 GiB margin
(`CHOPINFER_PREWARM_MARGIN_BYTES`).

`serve --prewarm-mapped-views off|on|auto` defaults to `off`. `on` and `auto`
warm mapped views in the background after the `listening` line. They do not
delay readiness, so early traffic can still meet cold rows. `auto` warms
only when `MemAvailable` holds the mapped bytes plus the margin, including
an exact fit; it declines if that value is unreadable. `on` warms anyway,
logging `mapped prewarm: warning host-ram-short: ...; warming anyway (on)`
when RAM is short. This option needs no `--checkpoint`.

A job waiting for the background warm should wait for the first
`mapped prewarm:` line other than the `warming anyway (on)` warning, then
handle its ending:

- `mapped prewarm: {...}` is the JSON done line, with `mode`, `bytes`,
  `seconds`, `resident_bytes`, `available_bytes`, `margin_bytes` and `stopped`.
  `resident_bytes` is null when hidden or stopped. A visible result below
  95% residency logs at warning level. Check `stopped: false` before treating
  the warm as complete.
- `mapped prewarm: warning host-ram-short: ...; not warming (auto)` means
  the warm was declined. An unreadable RAM value produces
  `mapped prewarm: warning host-ram-short: MemAvailable is unreadable; not warming (auto)`.
- `mapped prewarm: failed: ...; serving continues` means the warm failed.
- `mapped prewarm: could not start worker: ...; serving continues` means
  the background thread could not start, so no warm ran.
- `mapped prewarm: no mapped views` means an enabled mode found nothing
  to warm.

Decline, failure and worker spawn failure do not establish warm tables.
When residency is visible, a job can instead wait for its required ratio
of `chop_mapped_view_resident_bytes / chop_mapped_view_bytes`. When hidden,
only the done line can signal completion. Warming does not require visible
residency. Shutdown stops the background work and waits at most one second
for the workers before leaving a blocked operation to finish.

Every server that maps views samples residency immediately after readiness
and then every 60 seconds, whatever the option. The sampler reads no file
bytes; it uses bounded `mincore` calls. It logs the first sample as
`mapped residency: ...`, with residency and sample duration. After a visible
sample reaches 95%, a drop below it logs
`mapped residency: warning resident-low: ...` once; recovery logs
`mapped residency: resident-recovered: ...`. The first hidden sample logs
`mapped residency: warning residency-hidden: ...`. Other sampling errors
keep the previous sample and log `mapped residency: sampling failed: ...`.

Residency is visible only if the process owns the file on a mount that is
neither idmapped nor overlay (its filesystem UID equals the file owner,
and the owner is not the overflow UID), or may write the file on a
read-write mount. A writer on a read-only mount and root relying only on
`CAP_FOWNER` are conservatively reported hidden even when the kernel would
show them. The [documented container](developer-preview.md#checkpoint-and-cache-permissions)
runs as UID 10001 with a read-only checkpoint it does not own, so residency
is hidden. Running it as the checkpoint's owner with `--user` makes residency
visible on the documented read-only bind mount; keep that mount read-only.

Keep the option `off` for cold-cache measurements. Since the sampler runs
even with `off`, timing A/Bs should build both arms from a base that includes
it. `CHOPINFER_MAPPED_READ_AHEAD=0` disables adaptive read-ahead advice for
synchronous and asynchronous mapped gathers (`=1` enables both), and
`CHOPINFER_MAPPED_ASYNC_READ_AHEAD` overrides only the asynchronous path;
the advice never changes gather bytes, row order or synchronization.

Two default-off candidate switches, not serving options, change how a
large mapped gather (8192 or more table rows, such as a prefill layer's
prepared gather) reads its rows. `CHOPINFER_MAPPED_READER_FILES=1` reads
each mapping through open file descriptions of its own, one per reader,
instead of the one descriptor the mapping retains: the retained
descriptor's `/proc/self/fd` link reopened read-only, kept only when it
names the same device and inode, at most 24 per mapping and
`min(256, RLIMIT_NOFILE soft / 16)` in the process, and closed with the
mapping. `CHOPINFER_MAPPED_LARGE_GATHER_DIRECT=1`, which needs the first,
reads the rows on scoped workers (the CPUs the process may use less two,
from 1 to 24) straight into the gather's outputs and writes the other
ranks' copies in the same pass. Bytes, routes, events and ordering are
unchanged. Each logs one `chopinfer.mapped-row-fetch.v1` line when it
first engages (`reader-files` with the description count and budget,
`direct-gather` with the worker count). The process's first description
that fails to open logs one `reader-files` line with a `reason`; that
mapping opens no more, and its gathers read through the descriptions it
has or the retained one. The `miss_fetch` object of the mapped-row reports
counts `reader_file_gathers`, `reader_files_held`, `reader_file_failures`,
the `direct` gathers and workers, and the `nowait` probe classes (`full`,
`short`, `would_block`, `unsupported`, `other`). On 2026-11-07, a CPU
benchmark of a four-rank layer gather of resident rows on 24 CPUs of a
shared host took 26.3, 9.6 and 5.9 ms at 2890 prefill rows with neither
switch, the first, and both; served step-time acceptance is pending.

### Weight keeper

A keeper owns immutable rank images after a serving process exits, so a
restarted server maps them read-only instead of loading weights again:

```sh
chopinfer weights keep --keeper-dir "$KEEPER_DIR" \
  --model "$CHECKPOINT" --tensor-parallel-size 4
chopinfer serve --model "$CHECKPOINT" --tensor-parallel-size 4 \
  --weight-keeper "$KEEPER_DIR" --weight-keeper-policy require
```

A borrower holds a lease until its mappings are released; stopping it never
stops the keeper. Without a content proof, the recipe identity (checkpoint
identity, encoded weight plan and transform implementations) decides
whether images match. To lend across builds, each build prepares its own CPU
proof with `chopinfer weights identity --content --output FILE --model
"$CHECKPOINT" --tensor-parallel-size 4` and passes it to both keeper and
server with `--weight-keeper-identity FILE`; the keeper verifies it against
the uploaded allocation before lending. `prefer` loads refused images
normally, `require` fails instead, and a malformed or stale proof is an
input error. Cross-install reuse still requires the same unchanged
checkpoint files. Record loaded versus borrowed bytes with the startup
receipt.

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

## Watch running instances with choptop

`choptop --help` describes discovery, explicit URLs and display options.
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

### Start and stop serve processes from choptop

The model loader (`l`) starts `chopinfer serve` processes of this user and
stops the ones they replace; `X` stops the selected instance. `--read-only`
turns both off. choptop never stops a container or a server on another host:
either blocks a placement, and `X` refuses it with the reason. It uses the
Docker API only to read.

Each start passes three checks, any of which refuses it with the reason:

1. `S` takes a fresh GPU process reading and a discovery scan, even while the
   screen is paused. A picked GPU that became unknown or is newly held blocks
   the start.
2. `chopinfer serve ARGS --check` runs with the launch's environment. A
   refusal shows serve's message. The settings and model options it resolves
   must equal the selected profile's, as `chopinfer models --json` gives
   them, with the form's values applied. Only keys both sides govern are
   compared, and each difference is named.
3. The confirm dialog lists the instances with their ports and logs, what
   stops first, the profile's qualification, warnings (evidence limits,
   unqualified model options, other processes on the picked GPUs) and host
   memory. A failure that needs a GPU, such as memory fit or peer access, can
   still happen after the stop.

On Enter, choptop records each serve process it stops: pid, start time, uid,
command line (which must be `chopinfer serve`) and that its cgroup names no
container. It sends one SIGTERM through a pidfd, and only while the process
still matches that record; a process that changed is left alone, and the
launches waiting for it are cancelled. The stopping event keeps the stopped
server's command line. Each launch waits until the owners of its GPUs and the
holder of its port have exited, checks the port once more, and starts, or
reports the port in use. A new start cancels a launch that is still waiting
and stops a starting one through its process group first. serve drains for
up to `--drain-timeout-ms` before it exits.

A launch runs in a process group of its own with stdin closed and its output
in a new log under `$XDG_STATE_HOME/choptop/logs`, else
`~/.local/state/choptop/logs`, readable by this user only. It keeps running
after choptop exits. `launches.json` beside the logs keeps up to 20
launches, evicting completed history first: their choices, GPU UUIDs, port,
pid, start time, log and exit status. Active records are always kept; if
they alone exceed 20, choptop reports that it is keeping them all.
It feeds the loader's recent launches and lets a restarted choptop
take up the launches still running. A launch that exits during startup stays
listed as `exited N` with the end of its log until `X` dismisses it; `l` on
it reopens the loader with its choices.

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
      - alert: ChopinferMappedViewsEvicted
        expr: chop_mapped_view_resident_bytes / chop_mapped_view_bytes < 0.95
        for: 5m
      - alert: ChopinferMappedResidencyHidden
        expr: chop_mapped_view_residency_visible == 0
```

Drop `ChopinferMappedResidencyHidden` where hidden residency is expected,
such as a container that does not own the checkpoint.

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

## Logs and log levels

`chopinfer serve` writes its log to stderr, except the
`{"status":"listening",...}` line and a few startup lines about the KV cache,
which go to stdout. `--log-level error|warn|info|debug`, or
`CHOPINFER_LOG_LEVEL` when the flag is absent, sets how much it writes on
both; the default is `info`, and the listening line is always written. At `info` and `debug` the log
says which level wrote it: `startup logging: level=info`.

| Level | Adds |
|---|---|
| `error` | Server faults and requests that fail after admission: 5xx lines, `generation request failed`, worker and rank failures and fatal exits. Every bundle message the host does not recognize is written at every level, because the bundle log has no severity and such a message can be a failure's only detail. |
| `warn` | Client refusals (4xx), drain refusals, requests cut off at the drain deadline, connections closed at `--max-connections` (default 1,024 accepted sockets, including buffer waiters; overflow closes without an HTTP response), transient accept resource exhaustion, and degraded paths such as fallbacks and warmup warnings. |
| `info` | Startup and shutdown summaries, client cancellations, the graph-cache capacity, capture failure, eviction and release events, and the summary line. |
| `debug` | Routine traces: bundle step and commit shapes, drafter catch-up and confidence records, route records, per-capture graph-cache events, mapped-row samples and warmup progress. With these, the output is what serve wrote before it had levels, plus the request and summary lines. |

Host traces that an environment variable turns on, such as
`CHOPINFER_STEP_TRACE` and `CHOPINFER_REQUEST_TRACE`, print at every level.
Bundle records are leveled by their schema whatever enabled them, so a
model's diagnostic step-log option (for example `--model-option
diagnostic_step_logs=true`) shows its records only at `debug`. `tools/tune`
serves at `debug`; launchers that read startup lines set `info`.
`chopinfer fixed-batch replay` also reads `CHOPINFER_LOG_LEVEL`, so a
replay at `debug` with that option writes the bundle's records, such as each
drafted block's proposals and confidence bits, beside its exact comparison.

### Request failure lines

Every request that fails writes exactly one line,
`schema="chopinfer.request-failure.v1"`:

```json
{"schema":"chopinfer.request-failure.v1","time":"2026-10-25T14:03:11.482Z","level":"warn","request_id":"req_18e3fd2a0c1b4e00_000000000000002a","client_request_id":null,"method":"POST","path":"/v1/chat/completions","model":"missing-model","status":404,"error_type":"NotFoundError","code":null,"param":"model","message":"model is not available","served":["example-model"],"stage":"preparation","prompt_tokens":null,"generated_tokens":null,"elapsed_seconds":0.000412,"peer":"127.0.0.1:51234"}
```

- `level` is `error` for 5xx responses and failures after admission, and
  `warn` for 4xx refusals, drain refusals (`server is draining`), requests the
  drain deadline ended (`code="shutdown_timeout"`) and connections closed at
  the connection limit.
- `request_id` is the server's ID for the request, with the choice suffix
  (`_0`, `_1`, ...) once the request is admitted: the ID that
  `chopinfer.request.v1` and `generation request failed: request_id=` lines
  carry, and the response's `x-request-id`. `client_request_id` is the
  client's `X-Client-Request-Id` or `X-Request-Id`. With
  `--enable-request-id-headers` the response's `x-request-id` is instead the
  client's `X-Request-Id`, or an ID the server generated when the client sent
  none or an unusable one, and `client_request_id` carries that ID.
- `message` is a fixed text. A refusal whose client message is built from the
  request, such as a JSON error that quotes the body, an out-of-vocabulary
  token ID or the model name, logs its kind instead: `invalid request`,
  `invalid JSON request`, `backend failure` and the like. A body that did not
  parse adds `json_error` with serde's `category` (`syntax`, `data`, `eof` or
  `io`), `line` and `column`. An unknown model adds `served`, the first eight
  served names.
- A decoder failure (`code="invalid_model_output"`) adds `output_refusal`
  with `reason`, `token_ordinal` (zero-based) and `token_id`. The token fields
  are null when unknown, such as a refusal at completion finish. The reason
  is a known fixed refusal, or only its family followed by `refusal text
  withheld`; an unknown family becomes `unrecognized refusal`. It is bounded
  to 256 bytes on one line and carries no prompt or generated text. The
  client's error body and the line's `message` and `code` stay unchanged.
- `stage` is how far the request got: `transport` (refused before a handler:
  a head that did not parse or arrive, the connection limit, an unknown
  route, a missing API key), `intake`, `preparation`, `admission` or
  `generation`. A head that did not arrive is one a connection started and
  did not finish within the header deadline; a kept-alive connection that
  the deadline closes while idle writes no line. Fields that are not known yet are null; method and path are
  null for a head that did not parse or arrive and for a closed connection.
  `generated_tokens` counts what every choice of the request generated
  before it ended; with several choices, `request_id` names the one that
  failed.
- No line carries prompt or generated text. Only `output_refusal` carries a
  token ID: the one refused token. Fields the client
  controls are cut to a byte limit (method 16, path 256, model and request
  IDs 128, `param` 64), so a line stays under 4 KiB.

A readiness probe (`GET` or `HEAD` on `/health` or `/ping`, or `POST
/ping`) that hears 503 while the server starts or drains writes no line.
Any other refusal on those paths, such as a wrong method or a refused CORS
preflight, does.

A client that leaves, by closing its connection, dropping a stream or
abandoning a body upload, writes `schema="chopinfer.request-cancel.v1"` at
`info` with `reason="client_disconnected"` and no outcome fields. Cancelling a
background Responses request writes `reason="cancelled"`. If the drain
deadline has passed, the request instead writes a `shutdown_timeout` failure
line.

One thread writes these lines from a 1,024-line queue, so a handler never
waits on stderr. Each level has its own budget: bursts of 200 then 50 a
second for `error`, 100 then 20 for `warn`, and 50 then 10 for `info`. A line
over its budget, or one that finds the queue full, is dropped and counted.
Once a second, and when serving stops, a
`schema="chopinfer.request-failure-dropped.v1"` line reports the counts:

```json
{"schema":"chopinfer.request-failure-dropped.v1","time":"2026-10-25T14:03:12.001Z","dropped":{"error":0,"warn":312,"info":0}}
```

### Summary line

Every 60 seconds (`CHOPINFER_LOG_SUMMARY_SECONDS` sets another, 1 to 86400) the
server writes `schema="chopinfer.summary.v1"` at `info`. Counters are deltas
since the previous line; `running`, `waiting`, `kv` and `worker_state` are
current values:

```json
{"schema":"chopinfer.summary.v1","time":"2026-10-25T14:04:00.000Z","interval_seconds":60.0,"http_requests":1830,"failures":{"error":0,"warn":3,"cancelled":2},"refusals":{"invalid_request":2,"unknown_model":1,"not_found":0,"intake_timeout":0,"capacity":0,"unavailable":0,"internal":0,"other":0},"requests":{"submitted":1825,"completed":1823,"failed":0,"cancelled":2},"tokens":{"prompt":2310442,"generated":402118,"drafted":288320,"accepted":205427},"acceptance":0.7125,"preemptions":0,"ttft_seconds":{"count":1825,"p50":0.041,"p95":0.118},"itl_seconds":{"count":398470,"p50":0.0182,"p95":0.0265},"running":12,"waiting":0,"kv":{"used_tokens":431104,"capacity_tokens":1069952,"blocks_used":26944,"block_capacity":66872,"usage":0.4029},"worker_state":"ready"}
```

`acceptance` is accepted over drafted tokens in the interval, null when
nothing was drafted. `kv.usage` is occupied pages over pages, as
`vllm:kv_cache_usage_perc` reports it. `kv.used_tokens` counts the tokens of
every live sequence, so a prefix several sequences share counts once for
each and can exceed `kv.capacity_tokens` while pages remain. TTFT and ITL percentiles come from the interval's
histogram bucket counts, interpolated linearly within a bucket as Prometheus
`histogram_quantile` does, and are null when the interval observed nothing.

### Volume, retention and loss

Output stays on stderr, so capping and retention belong to the supervisor:
journald (`SystemMaxUse=`, and `LogRateLimitIntervalSec=` with
`LogRateLimitBurst=` per unit) or Docker (`--log-driver local`, or
`json-file` with `--log-opt max-size=` and `--log-opt max-file=`). Sustained
failures, cancellations or unrecognized bundle output can write much more
than ordinary traffic.

Lines can be lost: request lines over their budget or a full queue (counted
in the dropped-count line), and routine traces when their queue is full
(counted in `diagnostic writer: dropped` lines). When serving stops, the
server waits at most half a second for queued request lines. A second
signal, and a stalled worker's watchdog exit, end the process without that
wait.

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
admission has no trace; its [request-failure line](#logs-and-log-levels)
carries the same request ID with the stage, status and error type. Logging
is synchronous, so enable it only when its per-request cost is acceptable.

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
with drops is incomplete. Routine bundle records, written at `--log-level
debug`, share this queue. Startup, error and fatal messages are never
dropped; request-failure lines have their own queue and rate limits.
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
- `prefill_cap`, `predicted_us`: the
  [prefill policy](developer-preview.md#prefill-policy) cap and TBT prediction;
  null under the fixed budget or before an estimate exists.
- `decode_estimate_us` is always null and `trained` is always false. These
  retired decode-aware slots retain their names and types in `chopinfer.step.v1`.
- `timing_cuts`: present only when grid chunks were sized against the tbt
  cap (`--prefill-policy adaptive` on a grid bundle): the
  cap, the progress floor (limited to the room beside the decodes), the
  target, whether a chunk
  exceeded the cap only to progress because neither the cap nor the floor
  reached a granule multiple within that room (`forced`), whether
  `predicted_us` exceeded the target, and each such chunk's request,
  start, end and `end_reason` (`grid`, `capture`, `prompt`, `timing`,
  `floor` or `progress`). A
  `floor` chunk ends at the first granule multiple at or past the floor,
  so it may exceed the cap by less than one granule.
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
- `host.mapped_gather_us` is the host time of synchronous mapped-row
  gathers. `host.mapped_gather_exposed_us` is the device-measured wait on
  prepared ones: for each prepared gather the step's execution completed,
  the longest time one of its consumer streams waited for it after the
  stream's earlier work, summed over those gathers. A gather counts in the
  step whose execution completes it, which for in-tree callers is the step
  that joined it; a refused step's gathers count in the next step. Waits
  on one stream never overlap, but a sum over gathers whose longest waits
  fall on different ranks can, so the field bounds the critical-path wait
  from above. It is 0 when no prepared gather completes. Recording reads
  the stats completion already returns and adds no synchronization.
- The Engram-adjusted step time subtracts each step's Engram wait,
  `host.mapped_gather_us` plus `host.mapped_gather_exposed_us`, from the
  step's completion-to-completion cycle time (from the previous step's
  completion to its own) for decode and verify cells, and from its
  `execution_us` for prefill cells. It removes only the Engram wait on the
  step's critical path, not the CPU or I/O contention that page faults
  cause, and it is invalid for a change that alters Engram timing itself;
  judge those on raw time.

In DeepSeek's DSpark phase timings in `chopinfer.step.v1`, legacy prefill
planning is charged to `upload`, since it runs inside the staging upload
timer. A DSpark prefill's Engram hashing and the begin of its prepared
gathers run before catch-up and count in `gather` instead of
`target_upload`; a step refused after that begin settles its gathers before
it returns. A DSpark prefill planned while the drafter runs counts in
`prefill_plan`, outside `target`.
An upper-skip prefill that no kept plan or template serves is lowered while
it issues; that lowering and the keeping of its plan count in
`target_submit`.
`target_lower` keeps the target's own planning and the freeing of the
templates the step's planning dropped, which runs after the prefill's
launch; a step after a failed one frees that step's held templates at its
entry, outside every field. The phase fields stay disjoint and the target's
four fields stay within `target`. The DSpark step-shape record's
`early_prefill_begin`, `early_prefill_plan` and `streamed_lowering` say
whether the step's prefill began its gathers before catch-up, planned while
the drafter ran and was lowered while it issued. Its `prefill_templates` holds the instance's prefill plan
templates after the step: `templates` kept of at most `limit`, and the
growing counts of `hits`, `misses`, `busy`, `refused`, `kept` and
`dropped`, so two consecutive records give the later step's template reuse,
miss and drop.

The served A/B tuner (`tools/tune`) records per-window clocks, per-visit
major faults and job and visit enforced power limits; a changing limit voids
the whole job. A/B jobs can decide on the Engram-adjusted step time above.

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

### Timing-cut serving option

`--prefill-policy fixed|tbt|adaptive` resolves from `prefill_policy` with an
explicit flag taking precedence. `profiles show` and `serve --check` report
that identity. Retain the resolved settings and negotiated startup marker
`prefill policy: fixed|tbt|adaptive` with replay receipts, together with the
TBT target. Serving uses a 512-token floor and zero extra decode-only steps.
Adaptive needs cut support only on a bundle with a prefill grid. Its grid
and no-grid startup markers remain `prefill timing cuts on (adaptive): ...`.

Use `chop_prefill_policy_info` for remote identity. TBT model metrics and
selected-cap histograms are shared by `tbt` and `adaptive` and cannot tell
them apart. Current qualification rejects missing, conflicting or retired
identity. Historical receipt readers retain old budget evidence without
claiming it identifies a current policy. The three decode-aware estimator
gauges have been removed; step, selected-cap, timing-cut and TBT model
metrics retain their names and meanings.

A mixed step's `prefill_cap` is its chosen cap, or null when uncapped. The
cut counters and step records show whether eligible mixed steps actually
cut chunks.
