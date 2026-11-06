# Check kinds: the qualification contract

This is the contract of a model's release gate and of the shared check
kinds: `teacher_forced`, `media_forced`, `greedy_pin`, `determinism`, `api`,
`structured` and `frontend_goldens`. `chopinfer eval` (`crates/runtime-eval`)
implements their shared verification, scoring and judging. `eval run`
captures the supported kinds against a server (media remains Python-only),
and the Python suite (`tools/qual/kinds.py`, `tools/qual/checks.py`) scores
them by calling `eval rescore`. The conformance vectors in
`tests/qual-vectors/` pin the verdicts. Other checks (`receipt` kind) are
judged by their declared predicates alone. Numerical limits live in
[numerics](numerics.md#serving-qualification-gates); running the gate is in
[contributing](contributing.md#correctness-runner).

Each kind has a version (`teacher_forced.v1`, ...). A change of what a kind
sends, requires, computes or decides is a new version with new vectors.

## Release gates

A model's `qual.json` (schema `chopinfer-qual-model/v1`) is the one source
of its release gate: cohorts, named checks with their limits, required
inputs, reference pins, capabilities and optional configurations.
`chopinfer eval declaration` owns validation, and the Python loader calls
it; unknown fields and mistyped constants are refused, and
`tools/qual/modules.json` records the allowed modules and constant types.
`preview` binds the runner to the serving profile and its speculation
settings. No default model is inferred by model-neutral tools. Within v1,
fields are only added, as optional fields whose absence keeps the previous
behavior; any change of meaning (a field becoming required, removed,
renamed, retyped, or changing how a gate judges or selects checks) takes a
new schema version.

The gate consumes an already-running server and prepared inputs. A quick
cohort is a screen; full qualification requires every declared check and
configuration that is not opt-in. Missing metrics, empty selections,
nonfinite values, timeouts and child failures fail closed. Structured output
must be valid without refusal or truncation. Inputs must belong to the
candidate; an input path or a successful rescore does not prove which binary
produced it. A missing input is insufficient, and a failed worker stays
failed even if its partial receipt can be scored.

A different topology, KV format, draft depth, checkpoint or build needs its
own configuration and server. The configuration selects checks and
overrides declared serving facts, pins and rules. Its waivers are its own,
not inherited permission to waive the default's failures. Locks preserve
member scopes; an unbound declaration cannot vouch for another
declaration's inputs. A configuration declared `"opt_in": true` runs only
when selected with `--config NAME`: a full run without `--config` leaves it
out, and last good does not require it.

### External variants

An external checkpoint uses `DIR/MODEL/VARIANT.json` with its own
checkpoint, reference and overlay locks outside the tree, selected with
`--variant-dir` or `CHOPINFER_QUAL_VARIANTS`; `python3 -m tools.qual
variant` supplies lock, pin, skeleton and adopt commands. Its overlay may
narrow media and capabilities and select base checks, but cannot change
limits, add waivers, change how the model serves or reuse an in-tree
configuration name. Inherited references must still bind its own
checkpoint, and nothing it owns is read unauthenticated. A full pass
qualifies that checkpoint only and never advances the in-tree last-good
baseline. Preserve the overlay and lock identities in receipts.

## Capabilities and required evidence

`capabilities` describes the served profile. It is optional for older v1
files; when present every field below except `not_served` is required.
Both loaders refuse missing coverage, including checks omitted from
`cohorts.full.checks` and checks marked informational. Module names identify
the required checks, so each model chooses its own IDs.

| Capability field | Value and required hard check modules |
|---|---|
| `text` | Boolean; `teacher`. |
| `frontend` | Boolean; `frontend_goldens`. |
| `api` | Boolean; `api`. |
| `structured` | Boolean; `structured`. |
| `media` | Unique list of `image`, `audio`, `video`; each modality needs its own nonempty reference inventory. Case-format media declares `qualification.media.coverage`, mapping each modality to reference item IDs; image probes cover images only. |
| `speculation` | `none`, `lookup`, `mtp`, `dflash` or `dspark`; enabled speculation requires `spec_verify` and `verify_exact` with distinct control and candidate inputs, mandatory predicates, positive coverage and observed drafting. |
| `prefix_cache` | Boolean; `prefix_exact` scores bitwise equality. |
| `preemption` | Boolean; `preempt_compare compare --exact`, without `--baseline`. |
| `long_context` | Boolean; `needle` with the declared context lengths. |
| `stability` | Boolean; a local `soak` with timed host-memory and device-memory samples, bounded growth and traffic in every enabled mix. A `--remote-only` soak cannot satisfy it. The media mix reads the `soak_media_fixtures` input when the declaration states one, else `media_fixtures`. |
| `kv_offload` | Boolean. True requires model-level `kv_offload` facts and hard plain/pressure checks in the full suite. Serving offload remains off by default. |
| `not_served` | Optional. Maps a boolean capability declared false, one the model does not serve, to its reason. The correctness runner reports each false boolean capability as a not-applicable row with its reason, never a pass, and fails one without a reason. |

A capability the server does not serve is `not_applicable`, never a
silently skipped check. Capability coverage is a declaration check; the gate
run must still prove every predicate and input.

### Soak receipts

The soak must run the declared duration and concurrency, with samples
spanning the interval (no gap over 60 seconds) and at most 512 MiB growth in
both host and device memory. A server that fills a bounded cache while it
warms up declares the soak check's constants `WARMUP_SECONDS` and
`CACHE_ALLOWANCE_MIB` (host and device, the cache's bound): the growth from
the first sample at or after the warm-up must stay within the limit, and
the growth over the whole soak within the limit plus the allowance
(`soak.memory_rule`; the warm-up is capped at a quarter of the duration).

Host memory is VmRSS (`rss`) unless the constants declare
`HOST_MEMORY_SCOPE` `rss_less_mapped_views` and `MAPPED_VIEW_FILES` (file
names relative to the served checkpoint directory), for a server that maps
large read-only checkpoint files: the limit then applies to VmRSS less the
Rss of those files' shared, read-only, clean mappings that the serving tree
held at the first sample and in every sample since, with the first sample's
permissions and no Anonymous, Private_Dirty, Shared_Dirty or Swap pages,
attributed per mapping from `/proc/PID/smaps`. A mapping first seen later, a
retired one, a private, writable or dirty one, and every other mapping
count. Each sample records the judged host memory, the exempt Rss, its bound
and the exempt mappings, and the receipt names each declared file's
identity in the serving process's namespace. The Python `soak` kind
recomputes the judged series from each sample; a receipt judged under
another warm-up, allowance, scope or files, or with missing, inconsistent or
non-integer mapping evidence, is insufficient. A shorter saved soak is
diagnostic only.

### Preprocessing receipts

The Python `media_preprocessing` kind admits the CPU exporter's nested
`references` reports. Every adopted reference, case and stage must occur
exactly once, with the pinned binding and frontend digest. The scorer checks
shape and integer observations and recomputes the verdict from each recorded
error and the unchanged reference-spread bound. Rejudging a legacy receipt's
recorded fields is not a fresh tensor comparison.

## Common rules

### The receipt envelope

Every kind writes one JSON object (`chopinfer-qualification/v1`) with these
fields and the kind's own:

| Field | Type | Meaning |
|---|---|---|
| `schema` | `"chopinfer-qualification/v1"` | |
| `check` | string | the receipt ID: `teacher`, `multimodal`, `greedy-equality-<arm>`, `determinism-<arm>`, `api`, `structured` |
| `at` | string | UTC stamp `YYYYMMDDTHHMMSSffffffZ`; also the file name prefix, `<at>-<check>.json` |
| `server` | object | identity: `model`, `url`, `health`, `models`, and `checkpoint` when the server reports it |
| `mechanics_only` | bool | a mechanics run; never qualifies and is not coverage-checked |
| `passed` | bool | the kind's own verdict |
| `errors` | list | one entry per failed item; nonempty fails the check |
| `skipped` | list | coverage the run skipped; nonempty fails the check outside a mechanics run |
| `host_load` | object | host load samples that bound the check's timed work and their verdict (`common.HOST_LOAD_RULE`); `timing` `informational` marks throughput and latency figures as not performance evidence and never changes the verdict |

Field order is free; JSON numbers are IEEE 754 doubles. Receipts and vectors
are standard JSON, which has no NaN or infinity; a reader that accepts such
tokens must treat them as non-finite values, which make a receipt or
reference `insufficient`.

### Verdict procedure

A stored receipt is judged in four steps (`kinds.rescore`):

1. Admit: the receipt is an object, not mechanics-only and without
   `skipped` (unless the run was a mechanics run), and `errors` is empty.
   Otherwise the status is `fail`.
2. Verify: the receipt holds every field and item the kind requires below,
   and its counts agree with each other and with the reference. Otherwise
   the status is `insufficient`. A mechanics-only receipt skips this step.
3. Score: the metrics are the kind's metric fields and every field a
   declared predicate reads. The fields marked recomputed are computed again
   from the receipt's rows and the reference; the others are read as
   recorded and listed in the report's `recorded_fields`.
4. Judge: the status is `pass` when every declared predicate holds over the
   metrics, else `waived` when the check's declared waivers cover every
   failure, else `fail`. A predicate whose path the metrics lack fails.
   `greedy_pin` and `determinism` have fixed rules instead of predicates.

Statuses: `pass`, `fail`, `waived` (failed only where declared waivers cover
it; counts as passing and qualifies, and the report lists each applied
waiver), `informational` (a declared `informational_reason`: the check is
neither run nor judged; or `greedy_pin` under rule `informational`),
`insufficient`, `not_run` (a gate stopped before the check) and
`not_applicable` (validated unserved media, never a pass). A gate passes
when no selected check fails and at least one passes or is waived.

A live run (`release_gate.py`, `checks.py check`) applies steps 1, 3 and 4
to the receipt its own module just wrote; `checks.py rescore` applies all
four to stored receipts. In a stored gate run, a check whose worker exited
nonzero is `fail` whatever its receipt holds; the receipt's own verdict is
reported as `receipt_status`. A single stored receipt is judged as the check
its `check` field names (after a declared rename), else as the check whose
`records/<ID>.json` it is, else as the one check whose receipt pattern its
file name matches; `rescore --check ID` may name the check only within those
constraints, and anything else is a usage error.

A stored `checks.py check` run (its `check.json`, `chopinfer-check-run/v1`)
is judged as the one row of a gate run under the policy it recorded
(`--limits recorded`). Its receipt is the file the row names, or the log for
a check whose module prints its receipt (`stdout_json`), and must keep the
row's digest; every input its input rules read must keep its recorded
digest, else the row is `insufficient`. A run refused before its judge
(`"row": null`) is judged from its recorded inputs only when its receipt and
reference record are among them, every input its arguments name is recorded
and unchanged, and its recorded server reports qualified model options; its
report names the refusal `unjudged`, and a refused run whose receipt its
module writes stays `fail`. An input rule's `{model}` is the served model the
run recorded, or for an offline check run its `--served-model` name; for a
run recorded without it, `rescore --served-model NAME` supplies the name,
and without one such a rule is `insufficient`.
`release_preview.sh rescore RUN` replays a correctness run this way with the
current code and declarations. It only reads the run and writes its judgment
under `RUN/rescore/<UTC>/` or `--out DIR`. A regression row the run recorded
as informational for want of a baseline stays so; any other regression row
whose check run is missing fails.

A partial media receipt must declare the server's served modalities,
checkpoint identity and startup nonce. Its `not_applicable` reasons and
every unserved row must agree with that identity and the pinned reference's
coverage; a failing served row remains `fail`.

### Waivers

A check may declare `waivers` (a configuration declares its own under
`configurations.NAME.waivers.CHECK`; the base check's waivers are the
default configuration's alone). Each is a recorded decision with a `reason`
and a `decision` (who decided, when, where it is recorded), and covers one
of:

- a predicate: `predicate` names a declared rule by `path` and `op`, and
  `ceiling` is a value the rule refuses (looser than its bound, or for `eq`
  another value). The failing predicate is covered when it holds with the
  ceiling in place of its value, so a result worse than the accepted one
  still fails;
- a teacher category: `category`, `metric` (one of the spread metrics) and a
  positive `ceiling`, on a check that declares `passed` true. A failed
  teacher verdict is covered when the pooled scores are within their limits
  and every failed category fails only its waived metric, at most the
  ceiling; the `passed` and `failed_categories` predicates are then covered.

One validator (runtime-eval's `declaration::validate_waivers`) refuses a
waiver of an undeclared rule, a ceiling the rule already accepts, a category
waiver of a check without categories, an empty list, and a waiver whose
reason or decision is empty or blank. Every path that can make a check
`waived` applies it first: the declaration loader, `eval rescore`, eval's
judge (`kind::waive`) and, in tools/qual, `kinds.validated_waivers`, which
every judge and `release_gate.validate` call. Each applied waiver is
recorded on the predicate it covers (`waiver`: the measured value, ceiling,
reason and decision) and listed in the report's `waivers`.

### Predicates

A declared predicate is `{path, op, value, source}`; `source` cites the rule
or evidence its limit comes from. `path` selects values: each key descends
one level; `*` takes every child of an object or list.

| `op` | Holds when |
|---|---|
| `eq` | every selected value has the type of `value` and equals it |
| `le`, `ge` | every selected value is a finite number (not a bool) `<=` or `>=` `value` |
| `count_ge` | every selected value has at least `value` elements |
| `true_count_ge` | at least `value` selected values are `true` |

An empty selection does not hold.

### References and bindings

A reference is a file the declaration pins (`references.<name>` in
`models/<m>/qual.json`): its `sha256`, `schema`, checkpoint provenance
(`checkpoint_keys`, default repository, revision and served model name) and
`variants`. A kind reads a reference only when its bytes have the pinned
`sha256`, its `schema` and provenance equal the pin's, and every pinned
variant is present for every item; a reference with other bytes, even
reformatted, is refused. A fixture with a `binding` block must verify
against the reference cache ([fixture bindings](../tools/refrun/README.md#cache-and-fixture-bindings)).
Numerical gate inputs require a verified binding; an `unbound` reference
cannot qualify a numerical gate. A reference pinned without a spread variant
(a floor-only reference) is verified whole, workload included, on each read.

Reference scores: except for the explicit floor-only teacher policy, the
kind needs the gate variant and at least one spread variant, and for every
item every pinned variant's scores of its tokens, one entry per position, in
either layout (`tools/qual/exact_reference.py`): `logprob`, `top1`, `margin`
and `top5` (five `[token ID, logprob]` pairs), or the reference runner's
`logprob`, `top1`, `margin`, `topk_ids`, `topk_logits` and `topk_logprobs`.
Every logprob is finite and at most 1e-9; the top-k rows hold five distinct
token IDs in non-increasing order; `top1` is the first; `margin` is the
first gap within 1e-9; the token's `logprob` is at most the first top-k
logprob and equals its own top-k entry when listed. A reference that breaks
any of this, or lacks a pinned variant on an item, leaves the receipt
`insufficient`. A receipt records the reference it scored
(`reference_sha256` or `exact_reference_sha256`), and a stored receipt whose
reference digest differs from the reference read for it is `insufficient`.
`rescore --limits recorded` reads the references at the gate run's recorded
source revision.

`qualification.checkpoint` pins the checkpoint: lock, repository, revision,
served model name, source-lock digest and the direct-load
`checkpoint_identity`, the SHA-256 of the safetensors names, lengths and
headers that `serve` logs. Every tool that scores a server binds it through
its `/health` identity: it must name the pinned checkpoint identity and a
startup nonce and serve the pinned model name, and the tool refuses the run
if `/health` changed or any response's `x-chopinfer-checkpoint-identity` or
`x-chopinfer-startup-nonce` named another process. The identity covers file
names, lengths and headers, not tensor bytes; `chopinfer verify --model`
checks those.

`models/<model>/qual-references.lock.json` (with each configuration's own
entries under `configurations`; an external variant's lock beside its
overlay) pins the bytes gates read. `python3 -B -m tools.qual refs verify`
authenticates every declared input, `status` also prints the lock digest,
and `adopt` records new or changed inputs (a refresh requires the old lock
digest; `--entry KEY` refreshes only the named entries). A directory entry's
`scope` selects only the files gates read. Adoption verifies a bound
reference's runner binding, workload and preprocessing records before
pinning their bytes, and every numerical reference needs a runner binding,
which `adopt` and gate admission require. Never invent pins, and never
remove a binding to make a check pass.

### Exit statuses

| Command | Exit | Meaning |
|---|---|---|
| `check` | 0 | pass, waived, informational or not applicable |
| `check` | 1 | fail |
| `check` | 2 | usage or validation error |
| `rescore` | 0 | every verdict passes |
| `rescore` | 1 | a verdict fails |
| `rescore` | 2 | usage or validation error |
| `rescore` | 3 | insufficient evidence and no failure |

`check` has no insufficient-evidence status: a live run writes its own
receipt, and a missing prerequisite, an unreachable server or a worker that
exits nonzero is a failure (1) recorded in `check.json`. Usage and
validation errors (2) are an undeclared check ID, an unreadable or invalid
declaration, a malformed `--inputs` file or `--with` value, a PATH that is
neither a gate run, a check run nor a check receipt, a check run of
another model, variant or check than `--model` and `--check` name, or whose
recorded `policy.json` is missing or changed, and a `--served-model` other
than an offline check run's recorded one. With `--require-recorded`,
`rescore` exits 1 unless every verdict, metric and gate result equals the
recorded one. The recorded metrics are a receipt's own metric fields, or for
a check scored against a reference record (`reference_file`), whose receipt
is one arm's record, the comparison its module printed to its log (with the
two records it compared as `left` and `right`), against this code's
comparison of the same records in the same order. Metrics the run did not
record, such as a pair without that log or a run refused before its judge,
are never equal.

### Conformance vectors

`tests/qual-vectors/<kind>.json` (`chopinfer-qual-vectors/v1`) lists
vectors: `name`, `check` (the declared predicates, or `{"rule": ...}` for
`greedy_pin`), `reference` (the reference as the kind reads it, the baseline
record for `greedy_pin`, the other arm's record for `determinism`, or null),
`receipt` and `expected`: `status`, `metrics` when scoring completed, and
`error`. An implementation passes a vector when its status equals
`expected.status` and its metrics equal `expected.metrics`, with floats
equal within a relative 1e-12. `tools/qual/vectors.py` writes them;
`cargo test -p runtime-eval` checks the implementation against them exactly
(floats bit for bit), and `tools/qual/test_qual_vectors.py` checks the
Python suite's calls into it. Each vector's `covers` names the contract
clauses it pins, and every boundary has a vector on each side.
`tests/qual-vectors/references.json` pins the reference reading rules with
fixture bytes the same way.

### The spread rule (variant-spread.v2)

The limits of a set of `n` positions scored against a reference's gate
variant `G` (`tools/qual/limits.py`):

```
spread_v   = summary of variant v scored like a server against G, for each other pinned variant v
mean_limit = max(mean_floor(n), max_v rounded_up(spread_v.mean_abs_dlogprob * factor, 0.05))
decisive_limit = max(0.08, max_v spread_v.top1_decisive)
near_tie_limit = max(0.08, max_v rounded_up(spread_v.top1_near_tie, 0.01))
total_limit    = max(0.10, max_v spread_v.top1_disagreement)
mean_floor(n) = 0.10 if n >= 1000 else 0.15
factor     = 1 unless the check declares teacher.SPREAD_MEAN_FACTOR
```

The limits object is `{mean_abs_dlogprob, top1_decisive, top1_near_tie,
top1_disagreement}`; the total limit, never rounded, holds all flips
together, so the decisive and near-tie allowances cannot add up past the
reference's own disagreement or the 10% floor.

Category power (v2; v1 was the limits alone). A pooled set is judged by its
limits alone. A teacher category or media case is too, except a top-1
metric (decisive, near tie or total) whose limit has no power over the set:

```
p          = the reference bf16 variant's rate of that metric over the set's n positions
has_power  = p == 0 or limit - p >= 2 * sqrt(p * (1 - p) / n)
```

Such a metric also passes by the paired rule's top-1 clause against the
bf16 variant on the same positions: with `test_only` the positions where the
served rows flip with that verdict and the bf16 variant does not, and
`reference_only` the reverse, `test_only - reference_only <= 2 *
sqrt(test_only + reference_only)` (`limits.paired_top1`, the constant
`limits.CATEGORY_SIGMAS`). Without a bf16 variant every metric is held to
its limit. The mean limit never falls back. Each category and case row
records `spread_rule`, `passed_by`, `power` and, where the fallback was
tried, `paired_reference`.

`rounded_up(x, step)` reads both binary64 numbers as their shortest
round-trip decimal strings (Python's `repr`) and rounds up, in decimal, to a
multiple of `step`: 0.15 stays 0.15, 0.15000000000000002 becomes 0.2. The
spread values are computed in binary64 first. A scored set is within its
limits when its mean |dlogprob|, decisive rate, near-tie rate and total
top-1 disagreement are each at most (`<=`) their limit, compared in
binary64. The floors and the steps are the module constants a declaration
may override (`constants`). The product `spread * factor` is a binary64
product, so a factor of 1 leaves every limit unchanged.

A summary of positions is `scored` (count), `mean_abs_dlogprob` (mean of
|served logprob - reference logprob| of the scored token),
`max_abs_dlogprob`, `top1_disagreements` (positions whose served top-1 token
differs from the reference's, the flips) and `top1_disagreement` (that count
divided by `scored`), `top1_near_ties` and `top1_decisive_flips` (the flips
of each verdict under the near-tie rule) and their rates `top1_near_tie` and
`top1_decisive`. The top-1 token of a served row is the entry with the
lowest `rank`, then the highest logprob, then the lowest token ID.

### The near-tie rule (top1-near-tie.v1)

At a scored position whose served top-1 token `s` differs from the gate
variant's top-1 `t` (a flip), the gap is the reference's own: `value(t) -
value(s)` over its stored top-k row (`tools/qual/exact_reference.py`
`gap_top`: the runner layout's `topk_logits`, the CPU layout's `top5`
logprobs), computed in binary64. The flip is a near tie when the gap is at
most (`<=`) `NEAR_TIE_NATS = 0.3` (`limits.near_tie`, the boundary every
near-tie comparison uses), and decisive when the gap is larger or `s` is
not in the stored row. A reference variant scored like a server is judged by
the same rule against the gate variant. A receipt records each flip as
`top1_flips` entries (`offset` among the set's scored positions, `served`,
`gap`, `decisive`), and keeps the served rows the flips come from:
`served_rows`, per scored position the server's prompt-logprob row as
[token ID, logprob, rank] entries in top-1 order, and `served_rows_sha256`,
the SHA-256 of `served_rows` as compact JSON (`common.rows_digest`). The
kinds check the digest, that each row is well formed and holds the scored
token, take each position's served top-1 from its row, and require the
recorded flips to be exactly the positions and tokens where it differs from
the gate variant's top-1. They judge those flips again and recompute the
counts and rates from them. A receipt without its flips or served rows is
insufficient.

## teacher_forced (teacher_forced.v1)

Inputs: the pinned `teacher` reference. Per text: `name`, `category`,
`prompt_token_ids`, `score_from` (the first scored position) and, per
pinned variant, one entry per scored position in either score layout. The
gate variant is `exact`; the others set the spread. The text order is the
frozen `teacher-texts.json` order.

A check may explicitly set `teacher.ALLOW_FLOOR_ONLY` to `true` for a
floor-limited F32-only long-context reference. It requires exactly the
`exact` variant, authenticated producer and workload bindings to F32 with
exact dequantization, complete finite scores, and at least 1,000 scored
positions; the limits stay mean 0.10, decisive and near-tie rates 0.08, and
total disagreement 0.10. Missing or malformed evidence fails; it never
selects this policy. Every other check keeps the requirement for a spread
variant.

Request, one per text, in order: `POST /v1/completions` with `prompt` the
token IDs, `echo: true`, `logprobs: 5`, `prompt_logprobs: 5`,
`return_token_ids: true`, `max_tokens: 1`, `temperature: 0`. The response's
`prompt_token_ids` must equal the request's and the first echoed token's
logprob must be null. The served logprob of position `p` is the row
`prompt_logprobs[p]` entry of token `prompt_token_ids[p]`.

Receipt fields: `rows` (per text: `name`, `prompt_tokens`, `scored`,
`latency_seconds`, `served_rows` and `served_rows_sha256`), the pooled
summary against the gate variant and `vs_<variant>` for each spread variant,
`top1_flips` (with `text` and `category`), `by_category` (per category its
summary, `limits` and `passed`), `limits`, `reference_spread`,
`failed_categories`, `reference_sha256`, `reference_binding` and `server`.

Coverage: the rows name the reference's texts in order with their counts;
the pooled and category summaries are consistent and add up; each
`vs_<variant>` scores the pooled count; each row's served rows match their
digest and hold the scored tokens; `top1_flips` holds exactly the flips of
the served rows; and every reference text has valid token IDs,
`score_from`, a category and valid scores for every pinned variant.

Metrics recomputed from the served rows and the reference: the flips and
their verdicts, the pooled and per-category near-tie, decisive and total
counts and rates, `limits`, `reference_spread`, each category's `limits` and
`passed`, `failed_categories` and `passed` (pooled within its limits and no
failed category). Recorded: the served summaries.

Verdict: the declared predicates. Declarations require `passed` true,
pooled `mean_abs_dlogprob` within the pooled limit, `top1_decisive` and
`top1_near_tie` each at most 0.08, `top1_disagreement` at most 0.10,
`failed_categories` empty, `scored` at least the reference's positions, the
pinned checkpoint identity and the pinned `reference_sha256`. A receipt of
the retired prompt-logprobs mode names no exact reference, so rescoring it
is insufficient.

## media_forced (media_forced.v1)

Inputs: the pinned media reference (`references.media`) and the capture it
scores, whose manifest pins both files and the media fixtures. Per item, in
order: `id`, `kind` (`image`, `video`, `audio`, `text`),
`prompt_token_ids`, `forced_prompt_token_ids`, `answer_token_start`,
`answer` and, per pinned variant, one entry per answer position in either
score layout. The gate variant is `exact`.

Requests, per item: `POST /v1/chat/completions` with the item's messages,
`chat_template_kwargs` disabling thinking, `temperature: 0`,
`max_tokens: 256`, `return_token_ids: true` (where the frontend refuses
disabled thinking, no thinking switch and `max_tokens: 4096`); then the same
messages plus an assistant message holding the reference answer, with
`continue_final_message: true`, `max_tokens: 1`, `prompt_logprobs: 20` and
`return_token_ids: true` (teacher-forced). The forced response's
`prompt_token_ids` must equal the item's `forced_prompt_token_ids`.

Receipt fields: `rows`, one per reference item in order, with the answer,
expectation and token-count matches, `logprobs` (the item's summary,
`top1_flips`, served rows and their digest, `limits`, `passed`), `greedy`
(`passed`, `exact`, `near_tie_limit`, `reference_margin`, and on divergence
`divergence`, `served_token_id`, `reference_token_id`, `reason`) and
`passed`; then `limits`, `reference_sha256`, `exact_reference_sha256`,
`reference_binding` and `server`.

Greedy walk: from the first answer position, each served token must be in
the gate variant's top 5 at that position, and its gap, the top-1's logprob
minus its own in binary64, must be at most (`<=`) the near-tie limit:
`limits.NEAR_TIE_NATS` (0.3) or, for audio, 2.3. The walk ends at the first
served token that differs from the reference answer, or passes at the
reference answer's end; a served answer that ends inside the reference
answer fails. A served answer equal to the reference answer after whitespace
normalization is walked on the reference tokens. `reference_margin` is the
largest gap walked.

Coverage, metrics and verdict: every row is complete and consistent with
its served rows and the reference, as for `teacher_forced`; the flips,
rates, limits, greedy walk and `passed` (every item's greedy, expectation,
token count and logprobs passed) are recomputed. The declared predicates
require `passed` true and the pinned `exact_reference_sha256`.

## greedy_pin (greedy_pin.v1)

Inputs: the pinned exact reference's texts and a baseline record: the same
run's AR arm (the correctness runner's C1 greedy screen, recorded by
`chopinfer eval capture --kind greedy_pin`).

Requests, one per reference text, at the record's `concurrency`:
`POST /v1/completions` with `prompt` the text's first
`min(prompt_tokens, len - 1)` token IDs, `max_tokens`, `temperature: 0`,
`top_p: 1`, `return_token_ids: true`, `skip_special_tokens: false` and a
fresh `cache_salt`; the response must report `cached_tokens: 0`.

Receipt fields: `arm`, `k` (drafts), `concurrency`, `prompt_tokens`,
`max_tokens`, `cold_cache`, `reference_sha256`, `requests` (text name to
token IDs) and `finish_reasons`. Records of different workloads fail.
Metrics (recomputed): `equal`, `first_differences` and
`finish_differences`. Verdict: rule `exact` passes when `equal`, else
fails; rule `informational` passes when `equal`, else is informational.

## determinism (determinism.v1)

Inputs: none beyond the fixed request set of `tools/qual/determinism.py`;
the image request when the package serves image inputs. Two records: this
arm's and the other arm's, which a gate check names as `json_file` and
`reference_file`.

Requests, one at a time, greedy with 48 output tokens and top-5 logprobs,
each with a fresh `cache_salt` and reporting `cached_tokens: 0`: four short
completions; a long completion over two prefill chunks and its cold repeat;
with images, one chat request with a generated PNG; the long prompt as token
IDs; and per anchor A in 1024 and 2000, a primer of the first A IDs and the
whole ID prompt under the primer's salt, reporting `0 < cached_tokens <= A`.

Coverage: each record passes `determinism.check`: the request names in
order, a distinct salt per request (each hit its primer's), the cache rules
above, token IDs matching usage, complete and finite logprobs, and a long
prompt longer than one 2048-token chunk. Two answers are equal when their
`token_ids`, `tokens`, `logprobs` and `top_logprobs` are equal position by
position (floats by their IEEE 754 bits), then `text`, `reasoning` and
`finish_reason`.

Verdict: pass when every request is equal across the arms, the two cold long
sends are equal within each arm, and each prefix hit equals the cold
`long-ids` send within each arm; else fail. Diagnostic records of
nondefault model options (`determinism.py record --experimental` with a
sealed pin) report `qualified: false`, and release scoring rejects them.

## api (api.v1)

Inputs: the declared `api_samples` (a captured completions response with
logprobs), `chat_renders` (its `tools-00` conversation) and the frontend
declaration's thinking keyword.

Requests: the 40 cases of `tools/qual/api.py`, in order, against the served
model: single and multiple choices with and without streaming and usage,
generated token IDs, `/tokenize` and `/detokenize`, batched prompts,
`add_special_tokens`, a stop token, `add_generation_prompt: false`, hidden
reasoning, `ignore_eos`, nullable flags, an unknown extension, sampling
extensions, chat token-limit precedence, continuous and stream usage, a stop
string, completion logprobs against the captured sample, reasoning on and
off (where the frontend refuses disabled thinking, that request must be
refused with HTTP 400), streamed and whole tool calls whose names and
parsed arguments agree as typed JSON, a tool result, and the vLLM-compatible
`/metrics` series.

Receipt fields: `rows`, one per case in order (`name`, `passed`,
`latency_seconds` when it passed, and case facts), `errors` and
`stream_text`. Coverage: the row names are exactly the 40 cases in order,
each with a boolean `passed`. Metrics: `rows`, and `passed` recomputed as
every row passed. Verdict: the declared predicate `passed` true.

## structured (structured.v1)

Inputs: the 12 cases of `tools/qual/structured.py`, their schemas, choices,
regex and tools, and the settings: `repeat`, `stream`, `max_tokens` (512),
`reasoning_max_tokens` (4096), `temperature` (1.0) and `seed`.

Requests: per case, `repeat` times, the case's request with the settings'
sampling; with `stream`, each once more streamed with usage. An HTTP 400 is
`refused`; finish `length` is `truncated`; otherwise the output is
validated locally (JSON Schema subset, choice membership, regex full match,
offered tools with valid arguments) and is `passed` or `failed`.

Coverage: `settings.cases` are the 12 cases in order, `repeat` positive,
`stream` boolean, and the rows are exactly the requests those settings run,
each with a known status. Metrics recomputed from the rows: `summary` (per
case and overall status counts and `validity_rate = passed / (passed +
failed)`) and `passed` (no failed row). Verdict: the declared predicates:
`passed` true, overall `validity_rate` at least 1.0, and no refused or
truncated request.

## Declared media capture formats

`qualification.media` selects a capture format, reference name and receipt
ID; model names select no code path. `cases` pairs a captured prompt/answer
set with its exact variants and requires `capture_schema`, `expectations`
and, for audio, a positive `audio_near_tie`. `image-probes` uses captured
image requests and the pinned reference's per-probe `short` or `free-text`
rule, and requires `stop_token`, `expected`, `fixture_images` and `items`.

For image probes, rows must cover every declared probe in the captured
request order. Short answers use the reference's greedy near-tie threshold.
Free-text answers use the reference answer teacher-forced, with each probe's
spread-derived limits, floors, step and cap unchanged. Content expectations
and prompt token counts gate both groups. Every disagreement must be
identified by position and served token ID, and its gap and rates are
recomputed; missing or inconsistent flip identities are insufficient.
Media captures, their sidecars and the named exact reference are read
through the locked fixture readers, and regenerated image bytes must match
the captured image digests.

## frontend_goldens (frontend_goldens.v1)

The check compares a served capture with the declared goldens:
`frontend_capture/chat-renders.json` for chat and the declared
`tokenizer_goldens` for the tokenizer. Each served file compares
`token_ids` by row index, with exact IDs and equal row counts. A check may
declare `tokenizer_refusals` (`{sha256, special_token, reason}`): each
listed row must return HTTP 400 with exactly `prompt contains a special
token the model does not admit in prompts: <token>`; unlisted rows still
require exact IDs.

The served capture is a directory a capture script wrote, or one the check
captures itself with `--url` and `--capture-out` through the server's
`/tokenize` in reference order, so it can run in a quick cohort. The report
binds each compared golden file by name, input, SHA-256 and row count, and
rescore refuses missing or unequal bindings. Worker exit codes are 0 for
match, 1 for diverged and 3 for insufficient (zero gated rows, missing
required files or missing predicate evidence); a declaration without a
frontend check is a usage error (exit 2). Release qualification requires
both golden files, plus every declared predicate.

## Multimodal serving

Match checkpoint binding, prompt token IDs and usage to the reference.
Declared media formats determine coverage and scoring. Case answers are
teacher-forced under per-case spread limits, and a greedy walk through the
first departure must remain within 0.3 nats of the reference top-one (2.3
for audio). Image-probe descriptions use per-probe spread bounds with mean
error at least 0.15 and at most 0.30. Preprocessing and encoder checks are
separate obligations. Served but unqualified modalities remain explicitly
declared as such (`capabilities.unqualified_media`).

## Tap captures

A tap gate (`encoder_parity`, `qwen38_layers`, `text_layers`) counts engine
taps only from a capture bound to the process and request that wrote them.
Serve a diagnostic-taps bundle with `CHOPINFER_TAPS_DIR` set,
`CHOPINFER_REQUEST_TRACE=1`, prefix caching off and nothing else in flight,
then capture with `tools/qual/tap_binding.py capture`. Each request is sent
once under a fresh request ID and cache salt, and only the new steps that
name the live startup nonce and that request are kept; the trace, the steps
and the manifest must name it by its integer engine request ID. A prefix-hit
continuation must report `0 < cached_tokens <=` its primer's prompt tokens.
The manifest keeps the server's complete `model_options`, which must be the
qualified defaults; a capture with diagnostic option overrides records
`qualified: false`, and no gate admits it. The gate admits a capture only if
the candidate still serves with the capture's `/health` identity, startup
nonce included, and every step holds its recorded digests.

## Text layers

The `text_layers` check compares one reference sequence, captured from a
diagnostic-taps server, against the reference runner's F32 taps with its
BF16 taps as the floor, through `tapdiff.py`, by the
[layer rules](numerics.md#layer-comparisons). The
reference run is the declared `layer_reference` input, which the reference
lock pins; each mode's cache keys must name that mode's adapter and the
runner numerics files with their current digests, and the check's `MODES`
and `ADAPTERS` constants say which dequantization and adapter each mode
uses. A declaration's `REQUIRED_TAPS` must all be compared, else the status
is `insufficient`. A `ROUTER` constant (`score`, `top_k`, `n_group`,
`topk_group` and the `bias` input, a `router_bias.py extract` sidecar the
reference lock authenticates) has tapdiff judge routing flips by
[route-near-tie.v2](numerics.md#routing-near-ties); the receipt keeps the
flips, margins and removed rows.

## Encoder parity

The `encoder_parity` check compares every encoder output row the engine
computes, and MiMo's RVQ codes, with the generic encoder reference over the
same media, through `tapdiff.py`. The encoder declarations in
`models/<model>/contract/encoders/*.json` define the taps. Its input is a
`chopinfer.encoder-parity-run.v1` directory holding a bound
[tap capture](#tap-captures) and one reference run per declaration in the
`exact`, `policy` and `reference-bf16` modes. Tapdiff takes `exact` as the
reference and `reference-bf16` as the spread:

- `enc.vision.out`, per item. Direction: with d = |engine/|engine| -
  exact/|exact|| per row and its floor the same for reference-bf16, every
  row needs d <= 3 x max(floor, 2^-8) and the median d / (3 x floor) at most
  1. Scale (largest row-norm error per item): an F32-policy encoder
  (`f32-pairs.v1`) is held to its policy run's own error, at least 0.1% and
  rounded up to 0.1%; a BF16-policy encoder (`bf16-legacy.v1`) to the larger
  of reference-bf16's and the policy run's, rounded up to 0.5%
  ([why](numerics.md#mimo-vision-row-scale)).
- MiMo's `enc.audio.out`: relative L2 per row by the same rules, over rows
  whose RVQ codes all equal the exact run's.
- The audio encoder's `enc.audio.x.codes`: agreement with the exact codes,
  overall and in the first stage, at least reference-bf16's less 3
  percentage points.

An encoder whose every input kind the declaration lists in
`capabilities.unqualified_media` has no required rows and pins no
reference. Every tap must hold compared rows and tapdiff must report a
match. A lab capture of the encoder alone (`encoder_parity.py lab`) is
evidence for the encoder kernels and weights, not this check.

## Prefix caching

Hit and chunked rows must equal cold rows bitwise under the model's granule
and the context-parallel stripe rules
([prefix-cache accuracy](numerics.md#prefix-cache-accuracy),
[KV state](numerics.md#kv-offload-and-prefix-state)).
`prefix_exact.py record --ram-hit` is the RAM-hit arm: its server runs the
RAM prefix tier with `--debug-prefix-ram`, so every hit restores an anchor
copied back from the host tier. Its record meets the GPU-hit arm's gate,
drafter acceptance included, and each request that reports cached tokens
must raise `chop_prefix_cache_ram_hits_total` once. A model whose bundle
runs the tier may declare the opt-in `ram-forced` configuration, which
serves every arm that way; its prefix_exact input predicates require
RAM-hit records. `kv_offload.py`'s `kv-offload-prefix-pressure` arm runs the
pressure workload with the tier on and no forced swaps, and is judged
against the pressure off arm: complete captures, every C1 request bitwise
equal, promotions and RAM hits above zero, and no pending transfer, swapped
request or stall at the end.

## KV layout equality

A comparison KV layout that stores exactly the values the native formats
decode to must serve the same bits. In a model's BF16 comparison
configuration (`kv-bf16`), `kv_layout_equality.py compare` requires the
native and BF16 layouts' determinism records and forced records (on the
pinned exact and long-context references, the long text forced through 1
prefill and 1,023 decode rows) to be bit-identical in every text, finish
reason, token ID, token and selected and top logprob, with complete typed
rows. Each pair must name one checkpoint, metadata, source commit, bundle,
version, served model, effective profile and complete `model_options` map;
the layout fields are validated, not compared. `kv_layout_equality.py taps`
compares two tap captures, one of each layout, bit for bit in logical
representation, and requires the native capture's raw stored codes and
scales to decode exactly to its logical values. The check does not qualify
production accuracy; the default configuration's gates do.

## Capture coverage

Model declarations pin their capture inventory in addition to the shared
kind rules. Qwen's frozen frontend capture requires at least three needle
rows, 15 API rows, 75 chat-render rows and 81 tokenizer rows. Those minima
do not permit missing declared cases, and an unrelated diagnostic capture
does not replace the authenticated frontend or serving inputs.
