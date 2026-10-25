# Tuning speculative decoding

A drafting profile sets how many draft tokens each item may verify at each
concurrency with a balanced depth schedule: the profile's `speculation.depths`
(or `--speculation-depths`), for example `1:5,8:3,16:0`, gives depth 5 from
one live item, 3 from eight and none from sixteen. Plans never read step
times, so identical processes serve identical step shapes. The depth tuner
measures every depth at every concurrency on the real install and picks the
schedule, and measures what each verify width costs. Use it for every model
with a drafter, and again when a batch changes drafting or verify cost.

## Settings a deployment can tune

Each is a serve flag and a profile field under `speculation`, so a profile
sets the default and a flag overrides it:

- **Depth schedule:** `--speculation-depths` (`depths`), as above.
- **Verify widths:** `--speculation-widths` (`widths`), for example
  `6,8,16,32`. A balanced plan verifies the widest listed width between its
  decode rows and the rows its depth caps reach, filled by acceptance, so
  steps land on widths the package runs at its prepared speed.
- **Token capacity:** `--speculative-rows` (`rows`), the most verify rows of
  one step: one row per decoding item plus every draft. A package that
  declares `chop.speculative-requested-rows.v1` serves its default (32 for
  DeepSeek) unless asked for more, up to its manifest maximum (192 for
  DeepSeek, every item at its deepest block), and prepares the rows it is
  asked for.
- **Graph budget:** `--speculation-graph-rows` (`graph_rows`), the widest
  verify step captured in a graph. It reaches a package that declares the
  `verify_graph_rows` model option; wider steps run without a graph. Startup
  warms every width up to the package's default rows and, above them, only
  the listed widths up to the graph budget.

Raising the capacity or graph budget only pays where the wider steps cost
less than the tokens their drafts add. Measure the step costs first (the
`costs` job below), then widen the widths and depths that use them.

## Inputs

- `models/<model>/depth-tune.json`: the served model name, checkpoint
  variable, tensor parallel size, test KV pages and the bundle's KV log line,
  serve options, speculation mode, deepest depth (the drafter's published
  block), verify row capacity, concurrencies, timed repeats per arm, tokens
  per window, prompt groups per concurrency, kinds (sampled only: speed,
  tuning and costs are measured at the serving temperature 1.0, and greedy
  equality is an exactness test, not a measurement), sampling, selection
  rule, power limit and work budget, planning
  figures for the time estimate, and named `jobs` that split the sweep or
  measure costs. See `models/deepseek41/depth-tune.json`.
- A workload under `tools/tune/workloads/`. `coding.json` weights code
  completion and code edits over a general chat share; prompts are built from
  the install's own source snapshot, so a revision always gives the same
  prompts.

## Running

1. Seal a stage outside the tree, on a frozen integration install
   (install-NAME, cache-NAME, source-NAME, logs/):
   `python3 -I tools/tune/stage.py --model models/<model>/depth-tune.json
   --install-dir ROUND_DIR [--job NAME] $CHOPINFER_DATA/.../stages/NAME`
   with `local.env` sourced. It prints the job's SHA256 and estimated
   minutes, and writes `keeper-spec.json`. Keep each job under about 30
   minutes: split by concurrency (a low job over every depth, a high job over
   the depths that can still change its rows) and check the estimate against
   the last run's timings.
2. `python3 -I STAGE/harness/preflight.py STAGE` parses every arm's server
   options with the frozen executable, without CUDA, and checks the keeper
   spec.
3. `python3 -B STAGE/harness/recorded.py RECORDED_RUN OUT` runs the job's
   parsers (KV, memory, weight upload and counter lines, engine steps, the
   full-concurrency and cost tables) on an earlier run on real servers and
   fails unless every server and window parses.
4. Dry run on the CPU stub (with `local.env` sourced):
   `python3 -B STAGE/harness/tune.py --output STAGE/validation/dry
   --payload-sha PAYLOAD --dry-run`. Stub timings are synthetic; the dry run
   checks mechanics and failures only. The stub's controls
   (`TUNE_STUB_PERTURB`, `TUNE_STUB_PREEMPT`, `TUNE_STUB_IGNORE_DEPTHS`,
   `TUNE_DRY_POWER_W`) must each fail the job.
5. Queue `STAGE/tune.job` on an otherwise idle machine.

## What the job does and checks

For sweep, costs and exact modes, a depth schedule is fixed for a server's lifetime, so the job runs one
server per arm and reuses it for every window: by default one arm per depth,
the deepest first. Each server borrows its weights from the weight keeper
that the wrapper's `# keeper use keeper-spec.json SHA256` header asks the
queue runner for, and loads them when no keeper answers. The first arm runs
an untimed reference pass through every window, which also warms the page
cache, and keeps its outputs. Each arm then runs `repeats` passes, in
alternating order, of one fixed-seed sampled window (as the `kinds`
select) at every concurrency the depth can change (C x (depth + 1)
within the verify rows, every C at depth 0). Each window is one batched
request per prompt group, so all streams start in the same step; low
concurrencies serve several groups in turn so one prompt does not decide
them. Warm windows run the same kind. An arm starts only while its reserve of the work budget remains. Every
output must equal the reference bit for bit: draft depth never changes an
emitted token. Servers run the test KV mode with zero preemptions and
evictions, report their speculation mode, record memory after warmup and GPU
memory at readiness, after each warm window and at the end, and never draft
past their depth. Power limits are recorded and constant; SM clocks are
recorded, never asserted.

## Selection

`tune.md` lists per concurrency and depth: request and full-concurrency
tok/s with spreads over the repeats, J/token, stream p10, drafts per item,
accepted over drafted by draft position (not conditional on the earlier
positions) and whether every repeat served the same shapes. A depth's score
is its request tok/s over depth 0's (a weighted geometric mean when a
config measures more than one kind).
Among stable depths (identical shapes, spread within the limit) the
shallowest within the margin of the best wins (the larger of 1.5% and the
best depth's spread). `python3 -I tools/tune/report.py RUN_LOW RUN_HIGH`
merges split jobs into one schedule. The emitted schedule goes into the
profile's `depths` with the run cited by date and numbers in the model
README, and its landing batch passes the stability gate like any drafting
change.

## Step costs per verify width

A job with `mode` `costs` measures what each verify step shape costs. Its
`arms` set the token capacity and graph budget under test and list verify
widths that hold every row count the arm's depth reaches (C x (depth + 1)):
a profile's own widths would otherwise land those plans on a narrower width.
A depth 0 arm at the profile defaults gives the default graph memory.
`costs.md` and `costs.json` (`report.py --costs RUN`) list, for every
(items, rows) step shape, the median and p10/p90 cycle time from the previous
step's completion and the median execution time, the windows' tok/s and
J/token, and per arm its rows, graph rows and widths with each rank's memory
after warmup (MiB taken since KV sizing, MiB free) and its GPU memory
samples. With the KV pool fixed, the free memory of an arm with a wider
graph budget less the default arm's is what the wider graphs and buffers
cost. DeepSeek's costs job measures 16 to 48 rows at C8 and 32 to 96 rows
at C16 (96 runs without a graph).

## Confidence calibration

`--speculation-confidence` (profile `speculation.confidence`, see
[serving profiles](serving-profiles.md#drafter-confidence)) needs a
calibration of the drafter's raw confidence, fitted on CPU from the
drafter records a sampled run's servers log (`*.drafter-confidence.v1`):

```sh
python3 -B tools/tune/calibrate.py --sampled "$RUN" \
    --out "$CHOPINFER_DATA/calibration/<model>/calibration.json"
```

`$RUN` is a tuner run whose config measures only sampled windows, such as
the costs job above, or server logs. Position j's samples are the committed
blocks whose drafts before j were accepted and whose draft j was verified,
labelled by whether it was accepted, so the fit is the chance a draft is
accepted when every earlier one was. A record proves only its accepted
drafts; the draft verification stopped on counts as rejected only when the
same request drafts again later (an accepted stop token or a token limit
also stops verification, and ends the request), and is otherwise left out.
Repeated records are refused, and dropped log lines mark the capture
incomplete in the report. Each position needs `--min-samples` (200) of them,
so record arms that draft as deep as the speculative tokens. Fit on runs
served without a calibration: the fitter refuses logs whose startup line
says confidence is on. It drops records before the server's readiness line
so greedy startup warmup cannot contaminate sampled calibration. Extracted
record-only logs remain supported. Use recorded sampled measurement windows
when untimed warmup prompts differ from the workload. The report beside the output lists per
position the samples, accepted given earlier, accepted/drafted (the share of
drafted tokens accepted), the fit, and log loss in sample, held out over two
folds by request, and for a constant rate, AUC and the expected calibration
error; a fit that does not beat the constant held out carries no
information. Greedy pairs repeat the sampled fit unless `--greedy` runs are
given.

## Confidence exactness and A/B

The `confidence-exact` job runs greedy and fixed-seed sampled checks against
the first arm's reference, including an AR arm, at C8 and C16. These are
exactness checks; performance selection uses sampled temperature 1.0 only.
The `confidence-ab` job compares declared baseline/treatment `pairs` on one
install. Each calibration is one JSON argv element after
`--speculation-confidence`. The shipped arms carry the sampled fit approved
on 2026-10-24, with identical pairs at floors 0 and 1. Position 1 has ECE
0.0103 on 19,835 measured sampled labels; all five positions have at least
925 labels and held-out log loss below their constant baseline. The greedy
pairs reuse that fit for exactness checks.

Confidence arms must report `startup speculation_confidence: on`; a
refusal or missing line fails. A server reporting on without a requested
calibration also fails. Legacy non-confidence logs may omit the line.
Every measured confidence window where a common reservation can exceed
the balanced width must execute allocated steps. Non-confidence windows
must execute none. Depth checks use the verified width. Allocated step
keys include items, width, verify items and reserved rows, for example
`i8-r32-v8-R48`; unallocated keys retain their existing form.

A/B runs one ABBA block, two fresh processes per arm. Declare each baseline
before its treatments. The tuner runs the declared arms, then reverses them;
overlapping pairs share visits while each pair retains A1, B1, B2, A2 order.
Speculative visits require keeper borrowing, including in the CPU stub.
Each slot has a seed sequence: A1/B1 share one set and B2/A2 share another.
Sampling uses temperature 1.0. A single-repeat concurrency remains a short
control and receives no decision verdict.

Each measured window lasts 15 seconds. One worker per slot immediately
replaces completed requests. Only tokens streamed inside the interval enter
the throughput and energy measurements; in-flight requests drain afterward.
Every response must report zero cached prompt tokens. Full-concurrency
step cells exclude prefill and boundary intervals. Complete measured fault
deltas must sum to at most 1,000 major faults per visit. Startup, warmup and
drain faults do not enter that sum.

After all timed visits, additional baseline processes replay every distinct
prompt, seed and output length. Token IDs and selected logprob bits must
match. Deferring this pass preserves unseen prompts for cold measurements.
The stage estimate includes replay and process startup costs. A missing
exactness result, repeat, visit, clock record or fault counter fails the report.

`report.py --ab RUN` writes `ab.json` and `ab.md`. Each decision cell reports
`d1 = B1/A1 - 1`, `d2 = B2/A2 - 1`, their mean and `abs(d1-d2)` spread.
Request cells use throughput; matched step cells use inverse median cycle
time. A resolved gain has two positive deltas and mean above spread.
A resolved regression has two negative deltas and mean below
`-max(spread, 0.005)`. Other results are neutral. Set model `change` to
`speedup` (default), `variant`, `table_row` or `simplification`. Variants and
table rows also require a 1.5% mean; simplifications reject any weighted
resolved regression.

Default weights are one for measured decode C1, C2, C4, C8, C16 and C32,
and matched verify widths 6, 8, 16, 32 and 64. Model `cell_weights` overrides
these using keys such as `decode:c8` and `verify:r32`. Declare required
target shapes in `required_shapes`, for example `{"8": ["i8-r32"]}`.
Use `target_cells`, for example `["decode:c8", "c8/i8-r32"]`, to require
gains only in the intended cells while retaining regression checks on all
weighted cells. Without this field, every weighted cell is a gain target.
Missing required cells fail. Unmatched incidental shapes are displayed
without a paired verdict; an unmatched served cell prevents acceptance.
The report states the number and total weight of
its decision cells; a weighted average cannot hide a cell regression.
Short controls do not establish coverage of the unmeasured cells.

Energy is reported separately. The job records each GPU's enforced power
limit and SM clocks at the start and end of the job and every visit, with
clocks and power samples inside each window. Limits may differ between GPUs
and jobs, but a change within the job voids the entire result, without
pooling it. Compare J/token only at identical limits. Leave
`EXPECTED_POWER_W` and `--expected-power-w` unset; a legacy model config's
`expected_power_w` does not impose a cap. The tuner never changes limits or
clocks, and neither is a tuning key.

## Cold Engram workload

Select `tools/tune/workloads/cold-engram.json` with `mode: ab`. Its coding
and chat pool supplies prompts once per job, including warmups and request
replacements. Exhaustion fails instead of reusing a prompt. Increase the
workload's pool for larger runs. Exactness replays occur only after timing.

Supply `cold_engram` in the model configuration with `warm_fraction`
(default 0.1) and an `images` array. Each entry names an absolute `path`,
its expected `bytes`, and `views`: objects with `name`, `offset` and `bytes`
for each Engram view. These are measurement inputs, sealed with the stage;
keep the configuration outside the repository. For direct loading, identify
the actual checkpoint backing files and view ranges rather than assuming
that a logical rank image exists on disk. The package's manifest and host
validation still govern serving. The harness refuses symlinks and invalid
file sizes or view ranges.

Before each timed window, while its requests are idle, the tuner publishes
a new epoch to its private `CHOPINFER_MAPPED_ROW_CONTROL` file with
`reset_cache: true`, `release_page_tables: true` and read-ahead `mode: both`.
It requires the host's applied acknowledgement, then issues
`POSIX_FADV_DONTNEED` for every declared image source. Untouched mappings and
`mincore` measure each Engram view before and after advice and after the
window. If any post-advice resident fraction reaches `warm_fraction`, the
window is labelled `partly warm`; file advice alone is not proof of eviction.

The report retains residency, TTFT, prefill execution time and step.v1
`host.mapped_gather_us`. This phase measures synchronous mapped gathers;
prepared reads need their own trace to attribute asynchronous cost. CPU
stub residency and clocks are synthetic and prove no GPU performance or
physical eviction. `TUNE_STUB_RESIDENCY` exercises partly warm labelling.

## Replay a historical closed-loop run

Run `python3 -B tools/tune/recorded.py --closed-loop RECORDED_RUN OUTPUT_DIR`
on CPU. The adapter reconstructs timed rates from request events, checks
paired seeds, cached-token counts, faults, keeper sources and recorded power,
and applies the same paired verdict function as `report.ab`. It requires
agreement with the historical harness's classifications. Outputs and source
digests stay outside the repository.

Historical runs remain labelled `historical replay`, with admission gaps
listed. A replay does not retroactively certify fresh processes, 15-second
windows or exact output evidence that its source never recorded.

Before queueing, run the sealed wrapper end to end with `env -i` and
`--dry-run`. The stage writer removes group/other write permissions from
every stage file, including the keeper spec. Repeat `chmod -R go-w` on the
stage after adding validation outputs, and on its install before queueing.
Confidence jobs require the keeper and fail before server startup if it is
absent; they never silently replace borrowing with a normal load.
Test `TUNE_STUB_IGNORE_CONFIDENCE` (claims on but never
allocates) and `TUNE_STUB_CLAIM_CONFIDENCE` (claims on without the flag)
against the sealed harness directly; the queue wrapper clears these test
controls. Both must fail. Stub results check the harness only.
