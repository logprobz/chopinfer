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
   `TUNE_DRY_POWER_CHANGE_LABEL`) must each fail the job. `TUNE_DRY_POWER_W`
   selects a synthetic limit and does not impose a required wattage.
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

A/B runs one ABBA block, two fresh keeper-borrowed processes per arm.
Declare each baseline before its treatments. The tuner runs the declared
arms, then reverses them, retaining A1, B1, B2, A2 order for every pair.
A1/B1 share one set of per-slot seed sequences; B2/A2 share another.
Sampling uses temperature 1.0. Single-repeat concurrencies are short controls
and receive no decision verdict.

Before A1, a separate baseline process runs an untimed closed-loop pass at
every concurrency with both seed sets and the same prompts. This warms the
OS page cache for prompt and generated-token Engram rows. The process is
stopped before timed visits, so its learned acceptance prior is discarded.
`summary.json` records `page_cache_warmup`; a standard report requires the
complete warm-up before the first measurement. Cold jobs skip this pass.

Each window opens only after every slot completes its first request and at
least four seconds of priming have elapsed. Workers continue replacing
requests during standard priming. The 15-second measurement counts stream
tokens within its boundaries; drain tokens and time stay outside it.
Every response must report zero cached prompt tokens. Windows retain their
priming timestamps, replacement submissions, prefill execution share,
clocks and complete fault deltas. At most 1,000 measured major faults are
allowed per visit; startup, priming and drain are excluded from that sum.

Exactness compares token IDs and selected logprob bits across matched arms
only within one concurrency. Keys include prompt, seed and output length;
keys from different concurrencies are never merged because reduction
buckets can change the result (see [Batch shape](numerics.md#batch-shape)).
Only requests lacking a matched counterpart need an additional baseline
replay, at their original concurrency. Replay starts after an idle check.
Baseline padding requests keep every slot occupied until the unmatched
targets finish; padding outputs do not add exactness comparisons.
All cold requests are unmatched, so their exactness replay remains deferred
until all timed visits finish.

The four confidence comparisons are separate jobs: `confidence-ab`
(default against floor 1), `confidence-ab-cap48` (capacity 48 against floor 1),
`confidence-ab-floor0` (default against floor 0), and `confidence-ab-decode`
(AR against depth 1/floor 0). Each has a 28-minute work budget. The stage
estimates startup, warm-up, priming, measurement, drain and unmatched replay,
and refuses an estimate above `work_seconds`. Standard replay planning
reserves a cohort plus 25% of measured output per pair for unmatched tails;
this is a budget estimate, not an observed performance claim.

`report.py --ab RUN` writes `ab.json` and `ab.md`. Per decision cell it gives
`d1 = B1/A1 - 1`, `d2 = B2/A2 - 1`, their mean and `abs(d1-d2)` spread.
Rates are higher-is-better: request throughput or inverse median cycle time
of the served full graph. A resolved gain has two positive deltas and mean
above spread. A resolved regression has two negative deltas and mean below
`-max(spread, 0.005)`. Other results are neutral.

| Measurement | Target name | Weight key | Default weight |
|---|---|---|---:|
| Request throughput at C8 | `decode:c8` | `decode:c8` | 1 |
| Decode step `i8-r8` | `decode_step:c8` or `c8/i8-r8` | `decode_step:c8` | 1 |
| Verify step `i8-r32` | `c8/i8-r32` | `verify:r32` or `c8/i8-r32` | 1 |

Defaults cover measured decode C1, C2, C4, C8, C16 and C32 and verify widths
6, 8, 16, 32 and 64. Step cells must have full concurrency and at least
`min_cell_steps` (default 20) in every visit. Optional `min_cell_time_share`
can admit a cell instead when it occupies that fraction of measured time
in every visit. Other shapes are informational with zero decision weight.
Model `cell_weights` supplies nonnegative overrides.

Set `target_cells` to the intended gain targets. Every declared target must
be present in all four visits, sufficiently sampled and positively weighted;
a missing target cannot disappear from acceptance. Without explicit targets,
every weighted cell is a target. `required_shapes`, for example
`{"8": ["i8-r32"]}`, is a hard coverage and sampling requirement.

Model `change` is `speedup` (default), `variant`, `table_row` or
`simplification`. Variants and table rows also need a 1.5% mean gain in
their target cells. A simplification rejects only a resolved regression in
a weighted cell. Unmatched incidental cells are reported as coverage gaps,
not simplification rejections; required shapes and targets remain mandatory.
Other change classes cannot pass with unmatched served coverage.

The weighted section reports its descriptive verdict, mean, cell count,
coverage gaps and `timing_accepted`. Its verdict describes all decision
cells; timing acceptance checks the declared targets and rejects any
weighted regression. A neutral descriptive verdict can therefore accompany
a successful simplification or a gain limited to declared targets. The
weighted mean never hides a cell regression.

Energy is separate under `energy_check`. Only identical per-GPU enforced
limits permit comparison. Pending a pooled noise model, the equal-weight
mean paired J/token increase across active decision concurrencies vetoes
adoption only above 3%, with a positive mean at every such concurrency.
Mixed signs and smaller changes are informational. This provisional energy
check does not replace `timing_accepted` or exactness.

The job records each GPU's enforced limit and SM clocks at job and visit
boundaries and within each window. Limits may differ between GPUs and jobs,
but a change within a job makes it void. Void jobs cannot be pooled by A/B,
costs, sweep or recorded adapters. Leave `EXPECTED_POWER_W` and
`--expected-power-w` unset. The tuner never changes limits or clocks, and
neither is a tuning key.

## Cold Engram workload

Select `tools/tune/workloads/cold-engram.json` with `mode: ab`. Cold cells are
labelled **cold, unmatched prompts**, excluded from automatic timing and
energy acceptance. They report each arm's TTFT, prefill execution time,
step.v1 `host.mapped_gather_us` and residency. Metric spread is max/min minus
one across visit medians. Priming is excluded; cold request latencies and
prefill steps can include drain when a prefill outlasts the measurement.
Such a window can have zero timed output tokens and no J/token result.

The tokenized pool is assigned deterministically by length and original
index, with equal length histograms per arm, then split across visits.
Incomplete length groups stay unused. Request index determines its prompt,
so thread arrival order cannot change the assignment. Each prompt is used
once before deferred exactness replay. The report records actual used
lengths, because arms may complete different numbers of requests.

At staging, `cold_engram.request_budgets` reserves a priming and in-flight
cohort plus twice the planned 15-second request count for each window.
Warmup prompts are included. Staging refuses an undersized pool; after
tokenization, the runtime also refuses insufficient balanced pools before
timing. Exceeding a reserved window budget fails instead of reusing inputs.

Supply `cold_engram` with `warm_fraction` (default 0.1) and `images`. Each
entry names an absolute `path`, its expected `bytes`, and Engram `views`
with `name`, `offset` and `bytes`. Keep this configuration outside the
repository. For direct packages, identify the actual checkpoint backing
files rather than assuming logical rank images exist on disk. Serving
manifest validation is unchanged. The stage seals path, size, device,
inode and nanosecond mtime, checked without rehashing multi-GB files at job
startup and before advice. Symlinks and invalid sizes/ranges are refused.

Cold workers pause after priming. While idle, the tuner publishes a new
epoch to its private `CHOPINFER_MAPPED_ROW_CONTROL` file with
`reset_cache: true`, `release_page_tables: true` and read-ahead `mode: both`.
It requires the applied acknowledgement, then issues
`POSIX_FADV_DONTNEED` on every declared image source before timed requests
resume with unseen prompts. Untouched mappings and `mincore` measure each
view before/after advice and after the window. Any post-advice resident
fraction at least `warm_fraction` labels it `partly warm`; advice alone
never proves eviction. `mapped_gather_us` covers synchronous gathers;
prepared asynchronous reads require their own attribution trace.

CPU clocks and residency are synthetic. `TUNE_STUB_RESIDENCY` tests partly
warm labeling, and `TUNE_STUB_PREFILL_DELAY_SECONDS` tests prefill drain.
The CPU stub has one scheduler emitting batched steps for all live requests;
its standard A/B tests include C8 and C16.

## Replay a historical closed-loop run

Run `python3 -B tools/tune/recorded.py --closed-loop RECORDED_RUN OUTPUT_DIR`
on CPU. It reconstructs timed rates from events, validates paired inputs,
faults, keeper sources and power, then uses the paired verdict calculator.
Historical labels, mapped verdicts and disagreements are retained. Add
`--require-agreement` to fail when any classification differs or is unresolved.
The legacy tuner adapter rejects void sources before adaptation and retains
the source validity status and summary digest.

Historical results remain labelled `historical replay`, with admission gaps
listed. They do not certify fresh processes, 15-second windows or exact
output proofs missing from the source. Evidence stays outside the repository.

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
