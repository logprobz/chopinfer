# Check kinds: the qualification contract

This is the contract of the six shared check kinds: `teacher_forced`,
`media_forced`, `greedy_pin`, `determinism`, `api` and `structured`.
`chopinfer eval` (`crates/runtime-eval`) implements their shared verification,
scoring and judging. `eval run` captures the supported kinds against a
server (media remains Python-only), and the Python suite
(`tools/qual/kinds.py`, `tools/qual/checks.py`) scores them by calling `eval
rescore` ([CLI](cli.md)). The conformance vectors in `tests/qual-vectors/`
pin the verdicts. Other checks (`receipt` kind) are judged by their declared
predicates alone and are not part of this contract.

Each kind has a version (`teacher_forced.v1`, ...). A change of what a kind
sends, requires, computes or decides is a new version with new vectors.

## Preprocessing receipts

The Python `media_preprocessing` kind admits the CPU exporter's nested
`references` reports. Every adopted reference, case and stage must occur
exactly once, with the pinned binding and frontend digest. The scorer checks
shape and integer observations and recomputes the verdict from each recorded
error and the unchanged reference-spread bound. The generic receipt kind still
requires nonempty item coverage.

Legacy preprocessing receipts do not preserve candidate tensors. Their tensor
errors, shapes and integer equality remain recorded fields. Rejudging these
fields is not a fresh tensor comparison and cannot establish strict numerical
recovery of a failed run.

## Common rules

### The receipt envelope

Every kind writes one JSON object (`chopinfer-qualification/v1`) with these
fields:

| Field | Type | Meaning |
|---|---|---|
| `schema` | `"chopinfer-qualification/v1"` | |
| `check` | string | the receipt ID: `teacher`, `multimodal`, `greedy-equality-<arm>`, `determinism-<arm>`, `api`, `structured` |
| `at` | string | UTC stamp `YYYYMMDDTHHMMSSffffffZ`; also the file name prefix, `<at>-<check>.json` |
| `server` | object | identity: `model`, `url`, `health`, `models`, and `checkpoint` (`checkpoint_identity`, ...) when the server reports it |
| `mechanics_only` | bool | a mechanics run; never qualifies and is not coverage-checked |
| `passed` | bool | the kind's own verdict |
| `errors` | list | one entry per failed item; nonempty fails the check |
| `skipped` | list | coverage the run skipped; nonempty fails the check outside a mechanics run |
| `host_load` | object | host load over the check's timed work, to the receipt (tools/qual/README.md, `common.HOST_LOAD_RULE`); `timing` `informational` marks throughput and latency figures as not performance evidence and never changes the verdict |

and the kind's fields below. Field order is free; JSON numbers are IEEE 754
doubles.

### Verdict procedure

A stored receipt is judged in four steps (`kinds.rescore`):

1. Admit: the receipt is an object, not mechanics-only and without
   `skipped` (unless the run was a mechanics run), and `errors` is empty.
   Otherwise the status is `fail`.
2. Verify: the receipt holds every field and item the kind requires below,
   and its counts agree with each other and with the reference. Otherwise
   the status is `insufficient`. A mechanics-only receipt skips this step.
3. Score: the metrics are the kind's metric fields and every field a
   declared predicate reads, nested as in the receipt. The fields marked
   "recomputed" are computed again from the receipt's rows and the reference;
   the others are read as recorded and listed in the report's
   `recorded_fields`.
4. Judge: the status is `pass` when every declared predicate holds over the
   metrics, else `waived` when the check's declared waivers cover every
   failure (below), else `fail`. A predicate whose path the metrics lack
   fails. `greedy_pin` and `determinism` have fixed rules instead of
   predicates.

Statuses: `pass`, `fail`, `waived` (failed only where declared waivers
cover it; counts as passing and qualifies, and the report lists each applied
waiver), `informational` (a declared `informational_reason`: the check is
neither run nor judged; or `greedy_pin` under rule `informational`),
`insufficient`, `not_run` (a gate stopped before the check) and
`not_applicable` (validated unserved media, never a pass).

### Waivers

A check may declare `waivers` (a configuration declares its own under
`configurations.NAME.waivers.CHECK`; the base check's waivers are the
default configuration's alone). Each is a recorded decision with a
`reason` and a `decision` (who decided, when, where it is recorded), and
covers one of:

- a predicate: `predicate` names a declared rule by `path` and `op`, and
  `ceiling` is a value the rule refuses (looser than its bound, or for `eq`
  another value). The failing predicate is covered when it holds with the
  ceiling in place of its value, so a result worse than the accepted one
  still fails;
- a teacher category: `category`, `metric` (one of the spread metrics) and a
  positive `ceiling`, on a check that declares `passed` true. A failed
  teacher verdict is covered when the pooled scores are within their limits
  and every failed category fails only its waived metric (a metric the
  paired fallback passed is not failing), at most the ceiling; the `passed`
  and `failed_categories` predicates are then covered.

One validator (runtime-eval's `declaration::validate_waivers`) refuses a
waiver of an undeclared rule, a ceiling the rule already accepts, a
category waiver of a check without categories, an empty list, and a waiver
whose reason or decision is empty or blank. Every path that can make a
check `waived` applies it first: the declaration loader; `eval rescore`,
for every check it is sent and through its `waivers` call; eval's judge
(`kind::waive`), where waivers the validator refuses cover nothing; and in
tools/qual, `kinds.validated_waivers`, which every judge and
`release_gate.validate` (each `chopinfer-release-gates/v1` config, a
compiled declaration or another) call, refusing the check. Each applied
waiver is recorded on the predicate it covers (`waiver`: the measured value,
ceiling, reason and decision) and listed in the report's `waivers`. A gate
passes when no selected check fails and at least one passes or is waived.

A live run (`release_gate.py`, `checks.py check`) applies steps 1, 3 and 4 to
the receipt its own module just wrote; `checks.py rescore` applies all four
to stored receipts. In a stored gate run, a check whose worker exited
nonzero is `fail` whatever its receipt holds, whether the row links the
receipt or the receipt was left in the run; the receipt's own verdict is
reported as `receipt_status`.

A single stored receipt is judged as the check its `check` field names
(after a declared rename), else as the check whose `records/<ID>.json` it
is, else as the one check whose receipt pattern its file name matches.
`rescore --check ID` names the check instead, as the correctness runner's
replay does for a recorded row: two checks may share a receipt pattern (GLM's
NVFP4 `evals_compare` and `gsm8k_regression` both write `evals-compare`), or
names the check of a capture nothing else identifies. The receipt's own
identity is resolved first, in the order above, and the named check must be
one it names when it names any. A declared (or renamed) check ID in the
receipt, then its `records/<ID>.json` name, takes priority: such a receipt is
judged as that check whatever its file name. Only when neither names the
receipt does its file name constrain the choice: the named check must be one
whose receipt pattern matches it (so a check without a receipt pattern cannot
take a receipt another check's pattern names), and when no pattern matches,
only a check without a receipt pattern can be named. Anything else is a
usage error.

A partial media receipt must declare the server's served modalities, checkpoint
identity and startup nonce. Its sorted `not_applicable` reasons and every
unserved row must agree with that identity and the pinned reference's coverage.
The scorer recomputes served rows before returning `not_applicable`; a failing
served row remains `fail`. Missing or inconsistent evidence is `insufficient`.

### Predicates

A declared predicate is `{path, op, value, source}`. `path` selects values:
each key descends one level; `*` takes every child of an object or list.

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
variant is present for every item. A fixture with a `binding` block must
verify against the reference cache (`tools/refrun/cache.py`): its lock and
workload are recomputed from the repository. Numerical gate inputs require
a verified binding. The cache reader can report an absent binding as
`unbound`; that status cannot qualify a numerical gate.

Reference scores: except for the explicit floor-only teacher policy below,
the kind needs the gate variant and at least one spread
variant, and for every item every pinned variant's scores of its tokens, one
entry per position, in either layout (`tools/qual/exact_reference.py`):
`logprob`, `top1`, `margin` and `top5` (five `[token ID, logprob]` pairs), or
the reference runner's `logprob`, `top1`, `margin`, `topk_ids`,
`topk_logits` and `topk_logprobs`. Every logprob is finite and at most 1e-9;
the top-k rows hold five distinct token IDs in non-increasing order; `top1`
is the first; `margin` is the first gap (of logprobs, or of logits in the
runner layout) within 1e-9; the token's `logprob` is at most the first
top-k logprob and equals its own top-k entry when listed. A reference that
breaks any of this, or lacks a pinned variant on an item, leaves the
receipt `insufficient`.

`tests/qual-vectors/references.json` pins the reading rules with fixture
bytes: each vector's `pin` (`sha256`, `schema`, `variants`), `checkpoint`
(taken as verified against its lock) and `fixture` (the exact file text)
are read as `common.pinned_reference` and `pinned_binding` read them, and
`expected` is `accepted` with the binding status, or `refused` with the
reason.

A receipt records the reference it scored (`reference_sha256` or
`exact_reference_sha256`). A stored receipt whose reference digest differs
from the reference read for it is `insufficient`. `rescore --limits recorded`
reads the references at the gate run's recorded source revision.

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
declaration, a malformed `--inputs` file or `--with` value, and a PATH that
is neither a gate run nor a check receipt. With `--require-recorded`,
`rescore` exits 1 unless every verdict, metric and gate result equals the
recorded one.

### Conformance vectors

`tests/qual-vectors/<kind>.json` (`chopinfer-qual-vectors/v1`) lists vectors:
`name`, `check` (the declared predicates, or `{"rule": ...}` for
`greedy_pin`), `reference` (the reference as the kind reads it, the baseline
record for `greedy_pin`, the other arm's record for `determinism`, or null),
`receipt` and `expected`: `status`, `metrics` when scoring completed, and
`error`. An implementation passes a vector when its status equals
`expected.status` and its metrics equal `expected.metrics`, with floats equal
within a relative 1e-12. `tools/qual/vectors.py` writes them;
`cargo test -p runtime-eval` checks the implementation against them
exactly (floats bit for bit), and `tools/qual/test_qual_vectors.py` checks
the Python suite's calls into it. Each
vector's `covers` names the contract clauses it pins; every boundary has a
vector on each side: the decimal step of `rounded_up`, the inclusive limits,
both near-tie limits, the walk's outcomes, both score layouts, a missing
variant, invalid reference scores and impossible receipt scores, and for the
near-tie rule: a flip at exactly 0.3 nats and one just above, a served token
outside the stored top-k, the decisive, near-tie and total rate bounds (the
latter two at their inclusive edges), each raised by a spread variant, the
total cap at a variant's exact rate (21 of 200 passes, 22 fails), at a
near-tie-heavy shape that passes and a decisive-heavy one that fails,
recorded flip verdicts recomputed, recorded
served tokens checked against the served rows, served rows bound by digest,
and receipts without their flips or served rows.

Receipts and vectors are standard JSON, which has no NaN or infinity. A
reader that accepts such tokens (Python's `json` does) must treat them as
non-finite values, which make a receipt or reference `insufficient`.

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
records `spread_rule`, `passed_by` (per top-1 metric `limit`,
`paired_reference` or null), `power` (per metric `rate`, `limit`,
`positions`, `standard_error`, `has_power`) and, where the fallback was
tried, `paired_reference` (per metric `test_only`, `baseline_only`,
`margin`, `passed`, and the `variant`).

`rounded_up(x, step)` reads both binary64 numbers as their shortest
round-trip decimal strings (Python's `repr`) and rounds up, in decimal, to a
multiple of `step`: 0.15 stays 0.15, 0.15000000000000002 becomes 0.2. The
spread values are computed in binary64 first: |-0.75 - -0.5| is 0.25 and
gives a 0.25 limit, while |-0.55 - -0.3| is 0.25000000000000006 and gives
0.3. A scored set is within its limits when its mean |dlogprob|, decisive
rate, near-tie rate and total top-1 disagreement are each at most (`<=`)
their limit, compared in binary64. The floors and the steps are the module constants a declaration may
override (`constants`). The product `spread * factor` is a binary64 product,
so a factor of 1 leaves every limit unchanged; a factor above 1 keeps a
headroom proportional to the spread where rounding alone could leave the
limit at the spread itself (GLM's chat check declares 1.25,
[numerics](numerics.md#serving-qualification-gates)).

A summary of positions is `scored` (count), `mean_abs_dlogprob` (mean of
|served logprob - reference logprob| of the scored token),
`max_abs_dlogprob`, `top1_disagreements` (positions whose served top-1 token
differs from the reference's, the flips) and `top1_disagreement` (that count
divided by `scored`), `top1_near_ties` and `top1_decisive_flips` (the flips of
each verdict under the near-tie rule) and their rates `top1_near_tie` and
`top1_decisive`. The top-1 token of a served row is the entry with the lowest
`rank`, then the highest logprob, then the lowest token ID.

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
[token ID, logprob, rank] entries (rank null when the server gave none) in
top-1 order, and `served_rows_sha256`, the SHA-256 of `served_rows` as
compact JSON (`common.rows_digest`: `json.dumps` with `(",", ":")`
separators). The kinds check the digest, that each row is well formed and
holds the scored token, take each position's served top-1 from its row (the
entry with the lowest rank, then the highest logprob, then the lowest token
ID), and require the recorded flips to be exactly the positions and tokens
where it differs from the gate variant's top-1. They judge those flips
again (`gap`, `decisive`) and recompute the counts and rates from them. A
receipt without its flips or served rows is insufficient.

## teacher_forced (teacher_forced.v1)

Inputs: the pinned `teacher` reference. Per text: `name`, `category`,
`prompt_token_ids`, `score_from` (the first scored position) and, per
pinned variant, one entry per scored position
(`len(prompt_token_ids) - score_from` entries) in either layout of the
media reference below; the kind reads `logprob`, `top1` and the top-k gaps
of the gate variant. The gate variant is `exact`; the others set the spread (both
models: the reference runner's `bf16` and `f64` runs). The text order is the
frozen `teacher-texts.json` order.

A check may explicitly set `teacher.ALLOW_FLOOR_ONLY` to `true` for a
floor-limited F32-only long-context reference. It requires exactly the
`exact` variant, authenticated producer and workload bindings to F32 with
exact dequantization, complete finite scores, and at least 1,000 scored
positions. Capture, admission and offline scoring enforce the same policy.
For the approved 1,024-position check, the unchanged limits are mean 0.10,
decisive and near-tie rates 0.08, and total disagreement 0.10. Missing or
malformed evidence fails; it never selects this policy. Every existing check
keeps the default requirement for a spread variant.

Request, one per text, in order: `POST /v1/completions` with `prompt` the
token IDs, `echo: true`, `logprobs: 5`, `prompt_logprobs: 5`,
`return_token_ids: true`, `max_tokens: 1`, `temperature: 0`. The response's
`prompt_token_ids` must equal the request's and the first echoed token's
logprob must be null. The served logprob of position `p` is the row
`prompt_logprobs[p]` entry of token `prompt_token_ids[p]`.

Receipt fields:

- `rows`: one per reference text, in order: `name`, `prompt_tokens` (the
  text's token count), `scored` (its scored positions), `latency_seconds`,
  `served_rows` and `served_rows_sha256` (its scored positions' served rows,
  above).
- the pooled summary (`scored`, `mean_abs_dlogprob`, `max_abs_dlogprob`,
  `top1_disagreements`, `top1_disagreement`, `top1_near_ties`,
  `top1_near_tie`, `top1_decisive_flips`, `top1_decisive`) against the gate
  variant, and `vs_<variant>` for each spread variant;
- `top1_flips`: every flip against the gate variant, with `text`,
  `category`, `offset` (among the text's scored positions), `served`, `gap`
  and `decisive`, and `near_tie_nats`;
- `by_category`: per reference category its summary, `limits` and `passed`;
- `limits`, `reference_spread`, `failed_categories`, `reference_sha256`,
  `reference_binding`, `server`; older receipts may also hold an
  `informational` block, which is not scored.

Coverage: the rows name the reference's texts in order, each with the text's
`prompt_tokens` and `scored`; the pooled `scored` is their sum; the
categories are exactly the reference's, each with its positions; every
summary is consistent (`top1_disagreement == top1_disagreements / scored`
within 1e-9 relative, mean at most max); the category summaries add up to
the pooled one (counts exactly, the mean weighted by `scored` within 1e-9
relative, the max exactly); each `vs_<variant>` scores the pooled count;
each row's served rows match their digest, are well formed and hold the
scored tokens; `top1_flips` names reference texts with their categories and
holds, per text, exactly the offsets and served tokens where the served
rows' top-1 differs from the gate variant's, as many per category and pooled
as the recorded `top1_disagreements`; and every reference text has valid
token IDs, `score_from`, a category and valid scores for every pinned variant
(see references above).

Metrics: the receipt's fields above. Recomputed from the served rows and
the reference: the flips, each flip's `gap` and `decisive` (the near-tie
rule), the pooled and per-category near-tie, decisive and total counts and
rates from them, `limits` (the spread rule
over all texts), `reference_spread`, each category's `limits` and `passed`
(the spread rule over its texts),
`failed_categories` and `passed` (pooled within its limits and no failed
category; a receipt with `diagnostic_passed` keeps `passed` false and
recomputes `diagnostic_passed`). Recorded: the served summaries, since the
receipt keeps no served logprobs of the scored tokens beyond its rows'
entries, which this version does not re-sum.

Verdict: the declared predicates. Both models declare `passed` true, pooled
`mean_abs_dlogprob` within the pooled limit for their position count,
`top1_decisive` and `top1_near_tie` each at most 0.08, `top1_disagreement`
at most 0.10, `failed_categories`
empty, `scored` at least the reference's positions, the pinned checkpoint
identity and the pinned `reference_sha256`.

`chopinfer eval run` captures the check against an exact reference only. A
receipt of the retired prompt-logprobs mode (`vs_vllm`, a gate whose
`teacher_reference_kind` is `prompt-logprobs`) names no exact reference, so
rescoring it is insufficient.

## media_forced (media_forced.v1)

Inputs: the pinned media reference (`references.media`) and the capture it
scores (`fixtures/mimo26/multimodal/reference.json`), whose manifest pins
both files and the media fixtures. Per item, in order: `id`, `kind`
(`image`, `video`, `audio`, `text`), `prompt_token_ids`,
`forced_prompt_token_ids`, `answer_token_start`, `answer` (the reference
answer text) and, per pinned variant, one entry per answer position
(`forced_prompt_token_ids` from `answer_token_start` on) in one of two
layouts (`tools/qual/exact_reference.py`): `logprob`, `top1`, `margin` and
`top5` ([token ID, logprob] pairs), or the reference runner's `logprob`,
`top1`, `margin`, `topk_ids`, `topk_logits` and `topk_logprobs`, ranked by
logit. The gate variant is `exact`.

Requests, per item: `POST /v1/chat/completions` with the item's messages,
`chat_template_kwargs` disabling thinking, `temperature: 0`,
`max_tokens: 256`, `return_token_ids: true` (where the frontend refuses
disabled thinking, `reasoning.allow_disabled: false`, no thinking switch and
`max_tokens: 4096`, so the answer can follow the reasoning); then the same
messages plus an assistant message holding the reference answer, with
`continue_final_message: true`, `max_tokens: 1`, `prompt_logprobs: 20` and
`return_token_ids: true` (teacher-forced). The forced response's
`prompt_token_ids` must equal the item's `forced_prompt_token_ids`.

Receipt fields: `rows`, one per reference item in order: `id`, `kind`,
`answer`, `reference_answer`, `fixture_expectation_match` (the answer
contains the fixture's expected clauses in order, or the item has none),
`prompt_tokens`, `reference_prompt_tokens`, `token_count_match`,
`logprobs` (the item's summary over its answer positions: `scored`,
`mean_abs_dlogprob`, `max_abs_dlogprob`, `top1_disagreements`,
`top1_disagreement`, `top1_near_ties`, `top1_near_tie`,
`top1_decisive_flips`, `top1_decisive`, `top1_flips` (`offset`, `served`,
`gap`, `decisive` per flip), `served_rows` and `served_rows_sha256` (the
answer positions' served rows, see the near-tie rule), `limits`, `passed`;
or `{passed: false, error}`), `greedy` (`passed`, `exact`,
`near_tie_limit`, `reference_margin`, and on divergence `divergence`,
`served_token_id`, `reference_token_id`, `reason`) and `passed`; then
`limits` (the floors and near-tie
limits), `reference_sha256` (the capture), `exact_reference_sha256`,
`reference_binding`, `server`.

Greedy walk: from the first answer position, each served token must be in
the gate variant's top 5 at that position, and its gap, the top-1's logprob
minus its own computed in binary64, must be at most (`<=`) the near-tie
limit (`limits.near_tie`): `NEAR_TIE_LIMIT = limits.NEAR_TIE_NATS` (the
binary64 value 0.3) or, for audio, 2.3. The walk ends at the first served token that differs from the
reference answer, or passes at the reference answer's end; a served answer
that ends inside the reference answer fails. A served answer equal to the
reference answer after whitespace normalization is walked on the reference
tokens. The receipt records the walk: no `divergence` when the served
answer followed the whole reference answer; `divergence` and
`served_token_id` where it departed; `divergence` alone where it ended; and
`reason: "missing generated token IDs"` or an `error` when there was nothing
to walk (a failure). `reference_margin` is the largest gap walked, up to and
including a gap beyond the limit; a token outside the top 5 adds none.

Coverage: one row per reference item in order with its `kind`; each row has
`logprobs`, `greedy` (with a boolean `passed`), `fixture_expectation_match`
and `token_count_match` as booleans; a row whose `logprobs` has an `error`
fails; any other scored exactly the item's answer positions, with finite
`0 <= mean_abs_dlogprob <= max_abs_dlogprob` (1e-9 tolerance) and a
`top1_disagreement` in [0, 1] that is a whole number of positions divided by
`scored` (within 1e-9), with served rows that match their digest, are well
formed and hold the answer tokens, and with that many `top1_flips`, exactly
the answer offsets and served tokens where the served rows' top-1 differs
from the gate variant's; a
recorded `divergence` lies inside the answer, and a
`served_token_id` comes with its `divergence`; the receipt's
`exact_reference_sha256` is the reference's; and the reference's scores are
valid (see references above).

Metrics recomputed from the served rows and the reference: each item's
flips and its near-tie, decisive and total counts and rates (the near-tie
rule), `logprobs.limits` (the
spread rule over that item's variants), `logprobs.passed`, its greedy walk
(`greedy.passed`, `greedy.reference_margin`, `greedy.near_tie_limit`) from
the recorded walk fields, the item's `passed` (greedy, expectation, token
count and logprobs all passed) and the receipt's `passed` (every item
passed). Recorded: the served scores, the walk fields and the expectation
and token-count matches.

Verdict: the declared predicates: `passed` true and the pinned
`exact_reference_sha256`.

## greedy_pin (greedy_pin.v1)

Inputs: the pinned exact reference's texts; a baseline record: the same
run's AR arm (the [correctness runner](../tools/qual/README.md#correctness-runner)'s
C1 greedy screen, recorded by `chopinfer eval capture --kind greedy_pin`).

Requests, one per reference text, at the record's `concurrency`:
`POST /v1/completions` with `prompt` the text's first
`min(prompt_tokens, len - 1)` token IDs, `max_tokens`, `temperature: 0`,
`top_p: 1`, `return_token_ids: true`, `skip_special_tokens: false` and a
fresh `cache_salt`; the response must report `cached_tokens: 0`.

Receipt fields: `arm`, `k` (drafts), `concurrency`, `prompt_tokens`,
`max_tokens`, `cold_cache`, `reference_sha256`, `requests` (text name to
token IDs) and `finish_reasons` (text name to `stop` or `length`).

Coverage: both records hold a nonempty `requests` of non-negative token ID
lists, `finish_reasons` for exactly those requests, the workload integers and
`reference_sha256`. Records of different workloads (`reference_sha256`,
`prompt_tokens`, `max_tokens`, `concurrency`, request names) fail.

Metrics (recomputed): `equal`, `first_differences` (per differing request:
`request`, `position`, `ar_token`, `other_token`, a missing token as null)
and `finish_differences` (`request`, `ar`, `other`).

Verdict: rule `exact`: pass when `equal`, else fail. Rule `informational`:
pass when `equal`, else informational.

## determinism (determinism.v1)

Inputs: none beyond the fixed request set; the image request when the
package serves image inputs. Two records: this arm's and the other arm's.
A release gate check names them as `json_file` (this arm's record) and
`reference_file` (the other arm's), so the gate scores the one against the
other ([declarations](../tools/qual/README.md#model-declarations)).

Requests, one at a time, greedy with 48 output tokens and top-5 logprobs,
each with a fresh `cache_salt` and reporting `cached_tokens: 0`: four short
completions (`capital`, `haiku`, `fibonacci`, `arithmetic`); a long
completion over two prefill chunks (`long`) and its cold repeat
(`long-cold-repeat`); with images, one chat request with a generated PNG
(`image`); the long prompt as token IDs (`long-ids`); per anchor A in 1024
and 2000, a primer of the first A IDs with one output token
(`long-primer-A`) and the whole ID prompt under the primer's salt
(`long-hit-A`, reporting `0 < cached_tokens <= A`). `tools/qual/determinism.py`
holds the prompts and the PNG.

Receipt fields: `arm`, `run_id`, `concurrency` (1), `sampling`,
`request_set` (`names`, `sha256`, `long_prompt_tokens`,
`long_prompt_token_ids_sha256`), `image` (`included`, `png_sha256`,
`reason`), `cold_repeat`, `prefix_hit`, and `requests`: per request in order
`name`, `cache_salt`, `text`, `finish_reason`, `token_ids`, `tokens`,
`logprobs`, `top_logprobs` (ordered `[token, logprob]` pairs) and `usage`.

Coverage: each record passes `determinism.check`: the request names in
order, a distinct salt per request (each hit its primer's), the cache rules
above, token IDs matching usage, complete and finite logprobs, and the long
prompt longer than one 2048-token chunk.

Metrics (recomputed): `passed`, `left_arm`, `right_arm`, `requests`,
`tokens`, `equal`, `first_difference` (`request`, `field`, `position`,
`left`, `right`), `max_abs_difference`, `cold_repeat` and `prefix_hit` (per
arm `equal`, `first_difference`, `max_abs_difference`). Two answers are
equal when their `token_ids`, `tokens`, `logprobs` and `top_logprobs` are
equal position by position (floats by their IEEE 754 bits), then `text`,
`reasoning` and `finish_reason`.

Verdict: pass when every request is equal across the arms, the two cold long
sends are equal within each arm, and each prefix hit equals the cold
`long-ids` send within each arm; else fail.

## api (api.v1)

Inputs: the declared `api_samples` (a captured completions response with
logprobs), `chat_renders` (its `tools-00` conversation) and the frontend
declaration's thinking keyword.

Requests: the 40 cases below, in order. The model is the served one; every
request is non-streaming unless it says `stream`. `prompt` is `"Hello"` and
chat messages are one user turn `"Hello"` unless stated.

| Case | Request | Passes when |
|---|---|---|
| `choice_single_False` | completion, `n: 1`, `seed: 41`, `temperature: 0.7`, `max_tokens: 3`, `ignore_eos` | one choice |
| `multiple_choices_False_False`, `_True` | the same with `n: 3`; `_True` streams with usage | choice indexes 0-2, each finishing `length`; usage prompt equal to the single case's, completion 9, total their sum; a stream ends with a usage-only chunk and one response ID |
| `choice_single_True`, `multiple_choices_True_False`, `_True_True` | the chat forms of the three above | as above |
| `generated_token_ids_<chat>_<stream>` | completion or chat, `return_token_ids`, `temperature: 0`, `max_tokens: 3`, `ignore_eos`, streamed or not | prompt token IDs present; generated IDs non-negative, as many as `completion_tokens` |
| `tokenize` | `POST /tokenize` `"Hello"`, no special tokens | `count` equals the token list's length |
| `batch_single` (twice) | completion of one prompt, then of its token IDs, `max_tokens: 1` | a choice |
| `batch_False_False`, `batch_False_True`, `batch_True_False`, `batch_True_True` | completion of two equal prompts (text, then token IDs), `n: 2`, `max_tokens: 3`, streamed or not | 4 choices; usage prompt twice the single prompt, completion 12 |
| `add_special_tokens_<chat>_<special>` | completion or chat with `add_special_tokens`, `return_token_ids`, `max_tokens: 1` | prompt token IDs present |
| `detokenize` | `POST /detokenize` of the `tokenize` IDs | `"Hello"` |
| `stop_token_ids` | completion biased (+100) to the first `tokenize` ID, which is a stop token, `max_tokens: 3` | one completion token, empty text, finish `stop` |
| `generation_prompt_control` | chat, `add_generation_prompt: false`, `max_tokens: 1` | a choice |
| `hidden_reasoning` | chat "What is 17 times 23?" with thinking on, `include_reasoning: false`, `max_tokens: 8` | no reasoning field |
| `ignore_eos` | completion, `max_tokens: 3`, `ignore_eos`, `stop_token_ids: []` | 3 tokens, finish `length` |
| `nullable_flags` | chat with `stream`, `logprobs` null and `top_logprobs: 0` | a choice |
| `unknown_extension` | completion with an unknown field | a choice |
| `sampling_extensions` | completion, `min_tokens: 2`, `max_tokens: 3`, `seed: -1`, `top_k: -1`, `user` | 2-3 completion tokens |
| `chat_token_limit_precedence` | chat, `max_tokens: 1`, `max_completion_tokens: 3`, `min_tokens: 2` | 2-3 completion tokens |
| `continuous_usage` | streamed completion with `continuous_usage_stats` | usage on every chunk, non-decreasing, totals consistent, a final usage-only chunk |
| `stream_usage` | streamed chat "Say hello.", `max_tokens: 24`, with usage | a chat chunk and a usage-only chunk |
| `stop` | chat asking for a sentence ending with END, `stop: ["END"]`, `max_tokens: 48` | no `END` in the content |
| `completion_logprobs` | completion "Paris is", `max_tokens: 5`, `logprobs: 5` | its logprobs fields include the captured sample's |
| `reasoning_True`, `reasoning_False` | chat "What is 17 times 23?", thinking on or off, `max_tokens: 128` | reasoning present exactly when thinking is on; where the frontend refuses disabled thinking (`reasoning.allow_disabled: false`), `reasoning_False` must be refused with HTTP 400 (the row's `refused`: the first 200 characters of the error) |
| `stream_tool_arguments` | streamed chat of `tools-00` with its tools, `tool_choice: auto`, `max_tokens: 256`; then the same request not streamed | every chunk's choice has index 0 and tool indexes count up from 0; finish `tool_calls`; each call has an ID (a nonempty string, unique among the calls) and one name, sent before any arguments and never repeated; its argument deltas, in one part or several and all before the finish chunk, join into a JSON object; the calls' names and parsed arguments equal the unstreamed response's as JSON of the same types (an integer never equals a float, a boolean never a number; key order and spacing are ignored; IDs are not compared) |
| `tool_call` | the same, not streamed, `max_tokens: 128` | finish `tool_calls` with tool calls |
| `tool_result` | the conversation plus the call and a tool result (18, cloudy), `max_tokens: 1024` | the answer mentions 18 or cloudy |
| `metrics_compatibility` | `GET /metrics` | the vLLM-compatible counters, gauges and histograms with `model_name` and `engine` labels, a finish reason label and `+Inf` buckets |

Receipt fields: `rows`, one per case in the order above (`name`, `passed`,
`latency_seconds` when it passed, and case facts such as `finish_reason`),
`errors` (per failed case: `name`, `error`, the rejected `response`),
`stream_text`.

Coverage: the row names are exactly the 40 cases in order, each with a
boolean `passed`.

Metrics: `rows`, and `passed` recomputed as every row passed.

Verdict: the declared predicate `passed` true.

## structured (structured.v1)

Inputs: the 12 cases of `tools/qual/structured.py` (`chat_json_object`,
`chat_json_schema_strict`, `chat_json_schema_loose`, `chat_so_json`,
`chat_so_choice`, `chat_so_regex`, `chat_so_json_object`,
`completion_so_json`, `completion_json_schema`, `tool_required`,
`tool_named`, `reasoning_json_schema`), their schemas, choices, regex and
tools, and the settings: `repeat`, `stream`, `max_tokens` (512),
`reasoning_max_tokens` (4096), `temperature` (1.0) and `seed`.
The chat cases other than `reasoning_json_schema` disable thinking; where
the frontend refuses that (`reasoning.allow_disabled: false`) they send no
thinking switch and take `reasoning_max_tokens` unless the case sets its own.

Requests: per case, `repeat` times, the case's request with the settings'
sampling; with `stream`, each once more streamed with usage
(`<case>+stream`). An HTTP 400 is `refused`; finish `length` is
`truncated`; otherwise the output is validated locally (JSON Schema subset,
choice membership, regex full match, offered tools with valid arguments) and
is `passed` or `failed`.

Receipt fields: `settings` (`cases`, `repeat`, `stream`, ...), `rows` (per
request: `name`, `stream`, `status`, `latency_seconds`,
`completion_tokens`, `tokens_per_second`, `finish_reason`, and `error` and
`output` when not passed), `summary` (`overall` and per case: the four
status counts, `validity_rate = passed / (passed + failed)`, null when
none, rounded to 4 places, latency p50 and p90, mean tokens per second),
`errors` (the failed rows).

Coverage: `settings.cases` are the 12 cases in order, `repeat` positive,
`stream` boolean, and the rows are exactly the requests those settings run
in order, each with a known status.

Metrics recomputed from the rows: `summary` and `passed` (no failed row).

Verdict: the declared predicates: `passed` true, overall `validity_rate`
at least 1.0, and no refused or truncated request.


## Declared media capture formats

`qualification.media` selects a capture format, reference name and receipt ID.
`cases` pairs a captured prompt/answer set with its exact variants. `image-probes`
uses captured image requests and the pinned reference's per-probe `short` or
`free-text` rule. These formats share `media_forced`; model names select no code
path. Teacher reference variants and reference names are already declared data.

For image probes, rows must cover every declared probe in the captured
request order (the normalized reference's explicit `order` list, independent
of JSON object key order). Short answers use the reference's greedy
near-tie threshold. Free-text answers use the
reference answer teacher-forced, with each probe's spread-derived limits,
floors, step and cap unchanged. Their greedy comparison and near-tie reports
are context. Content expectations and prompt token counts gate both groups.
Scores must be finite, nonnegative and cover the reference answer positions;
top-1 disagreement must represent a whole number of those positions. The
recorded `near_tie_report.flips` must identify every disagreement by position
and served token ID, with the reference top-1 matching the pinned reference.
Their gaps, decisive/near-tie rates and total rate are recomputed with the
same rule as live capture; the report's diagnostic classifications are
context. Missing or inconsistent flip identities are insufficient.

Legacy image-probe receipts store the first divergence and its reference
margin, rather than the divergent token bytes. Rescore checks the margin
against the pinned top-5 gaps, then recomputes the inclusive threshold test and recomputes forced-score
limits. It does not claim to reconstruct missing token bytes. Reference digest,
probe coverage and all boolean expectation/count results are required.
Teacher and case-format media receipts require the bound `served_rows` of
the near-tie rule; their absence is insufficient.

## frontend_goldens (frontend_goldens.v1)

The check compares a served capture with the declared goldens.
`frontend_capture/chat-renders.json` is the chat reference; the separately
declared `tokenizer_goldens` is the tokenizer reference. Each served file
compares `token_ids` by row index, with exact IDs and equal row counts. Row
identifiers are not matching keys.

The served capture is either a directory a capture script wrote (the
positional argument) or one the check captures itself: with `--url` and
`--capture-out`, each reference chat row (its `messages`, with tool-call
arguments sent as JSON text, `tools`, `chat_template_kwargs` and
`add_generation_prompt` true) and each tokenizer text (`add_special_tokens`
false) goes to the server's `/tokenize` in reference order. The served IDs, or
the status and error body of a refused request, are written to `--capture-out`
with the server's identity in `capture-meta.json`, then compared as a capture
directory. This form needs no external capture, so it can run in a quick
cohort.

The report's `reference.files` maps each compared golden filename to `{input,
file, sha256, rows}`. `input` names the declaration input; `file` is
`chat-renders.json` under `frontend_capture`, or empty for the direct
`tokenizer_goldens` file. `sha256` hashes the exact reference bytes; `rows` is
the reference JSON array length. Rescore loads these declared files and refuses
missing or unequal bindings. Historical unbound reports are insufficient.
`--reference` supplies a diagnostic override; its report qualifies when its digest
and row count match the declared files, regardless of the override path
spelling. With a reference lock, an override must resolve to the declared file;
a different directory is refused before comparison. Environment-backed inputs are resolved through the same declaration
mechanism. Each golden report contains `rows` (served count), `reference_rows`,
`mismatches` (the first 20 mismatch indexes), `media_refused` (refused indexes),
`key` (filename) and `agrees` (equal counts and no mismatches). Media refusals
never bypass equality. Rescore checks `reference_rows` against the bound file
length, then validates counts, indexes and consistency; individual served IDs
are not retained, so equality is recorded evidence.

`coverage.files` counts compared rows (minimum of the two file counts).
`required` lists `--required-file` arguments; `missing` lists requested files
without compared rows. `omitted` lists absent or empty files. `compared` sums
all compared rows; `gated` sums files read by declared predicates or explicitly
required. Worker exit codes are 0 for match, 1 for diverged and 3 for
insufficient (zero gated rows, missing required files or missing predicate
evidence). A declaration without a frontend check is refused as a usage error
(exit 2). Only declared predicates gate; incidental needle/API failures remain
context. An explicit partial scope selects predicates rooted at the requested
filenames. Release qualification requires both golden files, plus every declared
predicate.

The Cargo tokenizer/chat tests use the distinct validation evidence category
`frontend_cargo_goldens`. This offline capture kind is developer tooling,
outside the portable eval subset. `frontend_goldens.py` compares HF
references, including the declared API answer shapes.

### Media declaration and historical policy data

`qualification.media` requires `format`, `reference` and `receipt`. `cases` also
requires `capture_schema` (string) and `expectations` (object); `image-probes`
requires `stop_token`, `expected` (nonempty string lists), `fixture_images`
(path strings) and `items` (`short` or `free-text`). `audio_near_tie` is a
required positive number for `cases`. Expectation entries under `files` or
`kinds` require a nonempty text `field` and boolean `single`. `expected` and
`fixture_images` keys must name declared `items`. MiMo declares 2.3 nats from
its measured encoder spread. The named reference must exist; case references
require variants, image references store their variants in probe data and do not
require a declaration variant list.

Stored policies lacking `qualification.media` receive this format block and
missing inputs from today's declaration, not from their historical revision.
Their recorded reference pins and thresholds remain authoritative. A short probe
margin at its divergence must equal a gap to a non-reference top-5 token within
the shared float tolerance; otherwise it is insufficient. Valid margins use
`limits.near_tie`, including its boundary tolerance. Free-text probe flips
retain token identities and are judged against pinned reference top-5 rows.

`dev/encoder_parallel.py` requires `CHOPINFER_QUAL_CONFIG` set to the
model's `qual.json` for declared media paths.

### Authenticated media and frontend inputs

The unified workers consume bytes authenticated by the reference lock. Media
captures, their sidecars and the named exact reference use the locked fixture
readers. Recorded-revision rescoring exports sidecars from their declared
source paths and checks those copied bytes against the adopted inputs before
scoring. Functional media files are read through the `media_fixtures` entry and
those returned bytes are passed to message construction. Regenerated image
bytes must match the captured image digests. `media_cases` and `media_images`
are code inputs, following the existing exemption for fixture helper code.

Frontend context and chat-render files use scoped `frontend_capture` members;
tokenizer goldens use their own file entry. Parsing, counts and report digests
use the authenticated bytes. Lock refusals exit with a `refused:` message
and no comparison verdict or traceback.
