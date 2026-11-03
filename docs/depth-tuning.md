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
  variable, tensor parallel size, KV pool and the bundle's KV log line,
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

A job may replace the generated workload with `prompt_file`, an absolute
path under `$CHOPINFER_DATA` outside the repository. It accepts a JSON array
or `.jsonl` file, one literal prompt per record. Records may be strings or
objects with exactly one of `prompt` (a string) and `messages` (literal
role/content strings), plus optional `kind` and `target_length`:

```json
[
  {"prompt": "Literal coding prompt...", "kind": "coding", "target_length": 8192},
  {"messages": [{"role": "user", "content": "Literal chat prompt..."}],
   "kind": "chat", "target_length": 32768}
]
```

The examples show the format; replace their text with the actual prompts.
`workload` may be omitted when `prompt_file` is set. Staging keeps the
literal content, seals the file's absolute path, byte size and SHA256, and
refuses symlinks or a file that changes while being read. The job verifies
the seal again, tokenizes with the served tokenizer and chat template, and
checks lengths before timing. In matched cold mode every external prompt
must fit a declared length cell within its tolerance, including any
explicit `target_length`. With no matched mode, an explicit target length
is exact. Out-of-range external prompts are refused rather than discarded.
Matched pair sets are drawn from these validated inputs. Keep the prompt
file and all generated evidence outside the repository.

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

## Compare two revisions

Build both revisions into one external round directory. Set `ROUND` under
`$CHOPINFER_DATA`, `MODEL` to the model directory name, `CHECKPOINT` to the
checkpoint, `RANKS` to the tensor parallel size, and `BASE_REV` / `HEAD_REV`
to the two Git revisions. Source `local.env` and use the pinned build setup
from [contributing](contributing.md#setup). In each revision's source
snapshot, build the host, model helper and bundle together:

```sh
set -euo pipefail
mkdir -p "$ROUND/logs"
for arm in base head; do
    if [ "$arm" = base ]; then ref=$BASE_REV; else ref=$HEAD_REV; fi
    revision=$(git rev-parse "$ref^{commit}")
    mkdir -p "$ROUND/source-$arm"
    git archive "$revision" | tar -x -C "$ROUND/source-$arm"
    printf '%s\n' "$revision" > "$ROUND/source-$arm/.commit"
    (
        cd "$ROUND/source-$arm"
        export CHOPINFER_BUILD_REVISION="$revision"
        export CARGO_TARGET_DIR="$ROUND/target-$arm"
        bash tools/dev/heavy.sh cargo build --release --locked --offline \
            -p runtime-host -p "model-$MODEL" -p "bundle-$MODEL" \
            --features runtime-host/linux-cuda-host --lib \
            --bin chopinfer --bin "chopinfer-model-$MODEL"
        source tools/dev/redeploy-common.sh
        install_serving_binaries "$CARGO_TARGET_DIR/release" "$ROUND/install-$arm" "$MODEL"
    )
    "$ROUND/install-$arm/bin/chopinfer" --version > "$ROUND/logs/version-$arm.txt"
    bash tools/dev/heavy.sh "$ROUND/install-$arm/bin/chopinfer-model-$MODEL" \
        prepare --direct --model "$CHECKPOINT" --tensor-parallel-size "$RANKS" \
        --cache-dir "$ROUND/cache-$arm" > "$ROUND/logs/prepare-$arm.json"
    find "$ROUND/install-$arm" -type f -print0 | sort -z | xargs -0 sha256sum \
        > "$ROUND/logs/install-$arm.sha256"
    chmod -R go-w "$ROUND/install-$arm" "$ROUND/cache-$arm"
done
```

Use absolute paths. Keep each helper's `prepared-direct` JSON object in
`logs/prepare-NAME.json`; if the helper also prints diagnostics, save those
separately. The package must be directly inside `cache-NAME`. Both builds
must use the same checkpoint files and package preparation options. Add any
model-specific preparation flags required by the serve configuration to
both helper commands.

In an external model config with `mode: ab`, select the installs per arm:

```json
"arms": [
  {"label": "a", "install": "base", "depth": 3},
  {"label": "b", "install": "head", "depth": 3}
],
"pairs": [["a", "b"]]
```

An arm may carry `depths` in `--speculation-depths` form (for example a
profile's default schedule `"1:5,8:3,16:3,32:0"`) instead of one uniform
`depth`, so a comparison runs the production schedule. Its entries start at
one item and increase; each step's depth check uses the entry for its item
count. It needs `speculation_mode: "balanced"` and cannot be combined with
`depth`.

Seal with `--install-dir "$ROUND" --install-name base --install-name head`.
An arm without `install` uses the first name on the command line. Unknown
names, unused sealed installs and multiple installs outside `mode: ab` or
`mode: exact` are refused. Sweep and costs jobs use one install. With one install, the seal
keeps its existing shape; omitting `--install-name` still means `head`.
Prompts and reporting helpers come from the first sealed source snapshot,
so both arms receive the same inputs. Preflight, warmup, every timed visit,
and unmatched baseline replay use the selected arm's binary and cache.
Records and reports name each arm's install and revision. A report with
install metadata requires a complete, consistent arm map; deleting the map
does not make the run legacy. Matched-seed ABBA
order and token/logprob-bit exactness comparisons are unchanged.

For a build comparison with greedy and fixed-seed sampled batches, use the
same named installs and arms with:

```json
"mode": "exact",
"kinds": {"greedy": 0.5, "sampled": 0.5},
"sampling": {"temperature": 1.0, "top_p": 1.0, "seed": 9100},
"concurrencies": [8, 16],
"repeats": 2
```

Exact mode needs no `pairs`. Its first declared arm captures one reference
per kind and concurrency, then checks every arm's repeated windows against
that reference. Greedy requests use temperature 0. Prompts are tokenized
once by the reference arm; all arms receive those token IDs. Equality
compares emitted token IDs and selected-logprob bits, with the first
different stream and token position retained for every differing window.
Output mismatches fail the job after its planned windows; invalid evidence
and server failures still stop it. The existing costs report separates
multi-install exact results by arm and labels reference repeats separately
from cross-install comparisons. Its timings do not establish a speed verdict.

Set `required_allocated_concurrencies`, for example `[8, 16]`, when the
comparison must exercise confidence allocation at those full concurrencies.
Every arm's reference and measured windows at each named concurrency must
execute allocated steps with more reserved than executed rows. Allocation
only in smaller drain cohorts does not qualify. This explicit requirement
also applies to A/B measured windows and supplements the inferred allocation
check described below.

Normal-load controls are separate single-install exact jobs with `keeper:
false` and `require_keeper: false`, queued before the keeper-backed block.
For a reference-arm control, use identical prompts, sampling and windows;
compare `prompt-tokens.json` first, then the reference and window `exact`
arrays across the normal and borrowed runs. A successful control alone does
not compare builds. Multi-install exact jobs always require keeper borrowing
and each install's content proof.

A single-install job may also pass `--content-proof NAME=FILE` for its one
build; the proof is checked against every arm's fresh recipe the same way.

Multi-install jobs require one shared keeper and each build's own
[content proof](weight-keeper.md#identity-across-builds). Supply existing
lane-staged `content.json` files with `--content-proof base="$BASE_PROOF"
--content-proof head="$HEAD_PROOF"`. The stage copies and seals them, so the
job no longer depends on the original proof paths. Without these flags it
reads `logs/weight-content-base.json` and `logs/weight-content-head.json`.

To produce proofs with the native host command, run the following once for
each install before staging. Set the shell array `SERVE_OPTIONS` to that
install's actual model and arm options, including its KV pool, profile or
model options and speculation settings. The lane `stage_keeper.py` helper
also runs this command and publishes a reusable `content.json`.

```sh
for arm in base head; do
    bash tools/dev/heavy.sh "$ROUND/install-$arm/bin/chopinfer" weights identity \
        --content --output "$ROUND/logs/weight-content-$arm.json" \
        --model "$CHECKPOINT" --model-cache-dir "$ROUND/cache-$arm" \
        --tensor-parallel-size "$RANKS" "${SERVE_OPTIONS[@]}"
done
```

Proof generation hashes produced weight bytes on CPU and can be expensive.
Run it sequentially before reserving GPUs. Output files must be new; keep
them owned and protected from group/other writes. Unchanged proofs can be
reused: tuner staging only runs fresh CPU `weights identity` planning for
every arm, using that arm's binary, cache and options. It checks each proof's
executable hash, checkpoint identity/stamps, recipe, layout, length and
complete rank coverage against that fresh plan. Missing, stale, unsafe or
mismatched proofs are refused. Serve options must match apart from
speculation options; checkpoint, cache and keeper overrides in argument
lists are refused. Run staging through `tools/dev/heavy.sh` too.

The cross-install `content_key` matches the lane helper: checkpoint identity
and stamps, every rank's content hash, physical layout and size, and the
device selection. Producer executable hashes and recipes may differ across
builds, but remain bound by each build's own proof. This permits a changed
cubin or tuning table to change the bundle module and recipe without
changing the shared weight bytes. Different content keys are refused.

The keeper spec uses the first arm's proof and seals its `identity_sha256`.
Every borrower, including warmup and replay, receives its own install's
`--weight-keeper-identity` and requires borrowing. The native keeper still
hashes its actual uploaded allocation; native borrowers still validate
their executable, fresh recipe and checkpoint before requesting equal
read-only content and layout. No runtime check is bypassed. The CPU stub
exercises different recipes with matching proofs and proof refusals; it
proves harness mechanics, not GPU exactness or performance.

## What the job does and checks

Set `kv_pages` to a positive page count for the fixed test pool, or to
`"default"` for the full pool selected by the native profile. Default-pool
servers and keeper specs receive no `--kv-pages`; explicit size or memory
fraction overrides are refused. The tuner records the actual startup page
counts and requires equality across all arms, warmup and baseline replay.
Both modes require zero preemptions and zero prefix-cache evictions. Cold
Engram jobs require `kv_pages: "default"`; a fixed test pool does not admit
a cold-memory measurement. `TUNE_STUB_DEFAULT_KV_PAGES` supplies a synthetic
default and `TUNE_STUB_KV_CHANGE_LABEL` tests a mismatched visit.

Native AR jobs declare `speculation_mode: "off"`, use `mode: ab`, `exact`
or `costs`, and omit arm `depth` and model `depths`. Omit `max_depth` or set
it to zero. AR uses the install's sealed prepared package with `--bundle`
and `--checkpoint`; select its AR profile in `serve_arguments`, for example
`["--profile", "tp4-ar"]`. This avoids a speculative default from direct
model preparation. No speculation or draft flags are appended or accepted
in AR argument lists. Health must report mode off and zero draft tokens.
AR dry runs require a frozen install and parse every launch with its native
`--list-model-options` before starting the CPU stub. The stub also rejects
draft flags in AR mode. Speculative jobs retain their existing arguments.

Arms can declare model environment controls as `env`, with strings to set
values and `null` to unset them. Use only controls the compared build still
reads. For an AR comparison of two served revisions, use the named-install
form instead: `{"label": "a", "install": "base"}` against
`{"label": "b", "install": "head"}`, with both installs sealed as above.
The controls apply to parser checks, CPU identity probes, warmup, timed
visits and reference replay, and are recorded in launch receipts. Only
`CHOPINFER_*` model controls are accepted; checkpoint, tracing, cold-control
and launcher settings cannot be overridden. The first arm supplies the
keeper baseline and may only unset controls, since the queue's keeper spec
does not carry an environment. Later arms may set controls. Their native
keeper identity checks remain mandatory.

An arm can also carry its own `allow_fallbacks`, in the job's form: in that
arm's windows only, each family it names replaces the job's allowance for the
family, or adds the family if the job has none. Use it for a candidate whose
expected route takes a fallback the baseline must not (for example a norm
entry in verify steps), so the same launches still fail the baseline's
windows.

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
emitted token. Servers use the declared KV pool with zero preemptions and
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
best depth's spread). `python3 -B -I tools/tune/report.py RUN_LOW RUN_HIGH`
merges split jobs into one schedule. The emitted schedule goes into the
profile's `depths` with the run cited by date and numbers in the model
README, and its landing batch passes the stability gate like any drafting
change.

Split recommendations also require disjoint concurrency ranges and the same
frozen install and package bytes, checkpoint, prompts, sampling, selection
policy and resolved serving target. Declare `adoption_target` with the frozen
package's `profile`, explicit non-depth `arguments` and optional `env` model
controls. Preflight resolves the target and every arm through `serve --check`
without CUDA; a real job repeats this check and binds live health identities
and launch arguments to it. Equivalent explicit and default flags may match.
The report keeps each partition's arguments, environment and resolved settings.
Its `verify_rows` remains a planning bound, not proof of runtime capacity.

For the profile-matched split jobs, with `local.env` sourced and `TUNE_ROUND`
pointing to a frozen integration install:

```sh
for job in profile-low profile-c8 profile-c16c32; do
  stage="$CHOPINFER_DATA/depth/stages/$job"
  python3 -B -I tools/tune/stage.py --model models/deepseek41/depth-tune.json \
    --install-dir "$TUNE_ROUND" --job "$job" "$stage"
  python3 -B -I "$stage/harness/preflight.py" "$stage"
done
# Queue each stage's tune.job on designated idle devices, then report its runs.
python3 -B -I tools/tune/report.py \
  "$CHOPINFER_DATA/depth/runs/profile-low" \
  "$CHOPINFER_DATA/depth/runs/profile-c8" \
  "$CHOPINFER_DATA/depth/runs/profile-c16c32"
```

These jobs inherit `tp4-dspark` settings from the frozen package and explicitly
declare the fixed test KV pool. They do not change or qualify the profile.
The existing `low`, `high-c8` and `high-c16c32` experiments retain their
different widths, disabled confidence and fixed prefill policy. Their combined
reports are diagnostic; use `--diagnostic` to inspect them. A capture without
a resolved sealed target cannot provide an adoptable schedule. Combined JSON
and Markdown are saved as `merged-tune.json` and `merged-tune.md` in the first
run directory, with the same refusal behavior as individual reports.

Recommendations require a passed depth sweep with every declared timed
window, repeat, workload kind and eligible depth present, including depth
zero. Failed, running, historical and synthetic captures are non-adoptable.
Use `python3 -B -I tools/tune/report.py --diagnostic RUN` to inspect their
measurements without a schedule. Regenerating an invalid report removes
the old recommendation. A complete experiment with no stable depth reports
a depth-zero fallback and does not claim a measured win. CPU dry runs test
the runner's mechanics and never attach a recommended schedule.

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
drafter records a sampled run's servers log (`*.drafter-confidence.v1`).
The records are routine traces that serve writes only at `--log-level
debug`, with the model's diagnostic step-log option on; the tuner serves at
`debug`, and the fitter's refusal of logs without records names the level
each served at:

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
every concurrency with its measured seed sets and the same prompts. A
single-repeat concurrency warms only its first seed set. This warms the
OS page cache for prompt and generated-token Engram rows. The process is
stopped before timed visits, so its learned acceptance prior is discarded.
`summary.json` records `page_cache_warmup`; a standard report requires the
complete warm-up before the first measurement. Cold jobs skip this pass.

The validity check also looks for an A1 offset on supported control shapes.
By default these are undrafted `iC-rC` steps. If the change affects decode,
declare untouched full-concurrency shapes with `control_shapes`, for example
`{"8": ["i8-r32"]}`. Each control needs the same minimum step count or time
share in all four visits as a decision cell. A1's median cycle time must lie
outside the B1/B2/A2 range and differ from their median by more than both
their max/min spread and 0.5% to flag an offset. Either direction invalidates
the job's timing verdict. `a1_control_check` records the medians and threshold;
missing or thin controls are labelled `unavailable`, not cleared. This checks
for a first-visit confound, without attributing its cause to page residency.
Where a job's design shows such an offset, `"lead_visit": true` runs one
more visit before A1: the first declared arm in a fresh process on A1's
windows and seed set. It is held to every runtime check (faults, prompt
caching, power, exactness) and recorded in `summary.json`, but no pair,
control check or verdict reads its windows.

Each window opens only after every slot completes its first request and at
least four seconds of priming have elapsed. Workers continue replacing
requests during standard priming. The 15-second measurement counts stream
tokens within its boundaries; drain tokens and time stay outside it.
Every response must report zero cached prompt tokens. Windows retain their
priming timestamps, replacement submissions, prefill execution share,
clocks and complete fault deltas. At most 1,000 measured major faults are
allowed per visit; startup, priming and drain are excluded from that sum.
`window_seconds` and `request_seconds` use the observed interval through the
final fault snapshot. A real window must last 15 seconds with at most 0.15
seconds of polling/snapshot overrun. Dry runs measure their actual shortened
interval and remain mechanics-only evidence. A cold slot whose priming pause
times out fails the job instead of submitting requests during preparation.

Exactness compares token IDs and selected logprob bits across matched arms
only within one concurrency. Keys include prompt, seed and output length;
keys from different concurrencies are never merged because reduction
buckets can change the result (see [Batch shape](numerics.md#batch-shape)).
Only requests lacking a matched counterpart need an additional baseline
replay, at their original concurrency. Replay starts after an idle check.
Treatment-only keys compare against baseline output. Baseline-only keys
replay against the baseline itself as a reproducibility control; they add
no cross-arm evidence.
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
The repository confidence jobs take about 20 minutes in this estimate.
Served timing jobs queued under the lane rules target 15 minutes.

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
Model `cell_weights` supplies nonnegative overrides using the key forms
above, with measured concurrencies and positive row widths. Unknown key
forms or unmeasured concurrencies are rejected.

Set `target_cells` to the intended gain targets. Every declared target must
be present in all four visits, sufficiently sampled and positively weighted;
a missing target cannot disappear from acceptance. Without explicit targets,
every weighted cell is a target. `required_shapes`, for example
`{"8": ["i8-r32"]}`, is a hard coverage and sampling requirement.
Required shapes also need all four visits; a single-repeat control cannot
satisfy that requirement.

Model `change` is `speedup` (default), `variant`, `table_row` or
`simplification`. Variants and table rows also need a 1.5% mean gain in
their target cells. A simplification rejects only a resolved regression in
a weighted cell. A single job's simplification rejection is provisional
until the cell's regression reproduces in a confirmation job.

Support is checked before pairing. Shapes below the threshold in every
visit are informational. Shapes supported in both visits of only one arm
are also informational, labelled `arm-specific shape`, since different
depths can serve disjoint decode and verify shapes. Other unmatched cells
are coverage gaps. These cells have zero decision weight and do not reject
a speedup or simplification. Declared targets and required shapes remain
hard requirements even when a shape would otherwise be informational.

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

For matched latency acceptance, opt in with:

```json
"cold_engram": {
  "matched": {"lengths": [8192, 32768], "tolerance": 128},
  "warm_fraction": 0.1,
  "images": []
}
```

Populate `images` with the backing files and Engram views described below.
Supply enough distinct workload prompts at each target length. Lengths are
checked after native tokenization, including the chat template; the example
accepts 8064-8320 and 32640-32896 tokens. Ranges must not overlap. Staging
checks the total prompt budget, and the runtime refuses a missing length
pool before timing. Every measured concurrency must have all four visits.

A1/B1 share one prompt set and B2/A2 another, disjoint set. Each cell first
primes one cohort, then pauses all workers for the acknowledged mapped-row
reset, page-table release and `POSIX_FADV_DONTNEED`. It submits exactly one
timed cohort of C new prompts after that eviction. Paired arms therefore
complete the same requests even when their speeds differ. Each request has
a distinct cache salt, and zero cached prompt tokens is mandatory. Extra
warmup requests remain untimed. The observation window stays at 15 seconds;
TTFT and prefill steps can finish in drain. This is a fixed-cohort latency
measurement, not a saturated throughput measurement.

Every length cell receives its own eviction. If any post-advice resident
fraction reaches `warm_fraction`, the entire run is void, retains the
residency evidence, and cannot be pooled. Missing eviction evidence, length
violations, differing paired prompts/seeds, reused salts or overlapping
pair sets also prevent acceptance. All visits must be cold.

`ab.json` and `ab.md` report `d1`, `d2`, mean and spread separately for TTFT
and prefill step execution time at each concurrency and target length.
Values use inverse visit medians, so a positive delta means lower latency.
The prefill metric uses full prefill `step.v1` execution times, including
chunks when a long prompt spans several steps. Every declared length and
both metrics are required targets with weight one, using the same gain,
regression and change-class thresholds as warm cells. Warm `target_cells`,
`required_shapes` and `cell_weights` overrides are refused in this mode.
Synthetic runs remain mechanics-only and cannot establish contract A
acceptance. `TUNE_STUB_TOKEN_LENGTHS` selects synthetic tokenizer lengths;
`TUNE_STUB_RESIDENCY_LABEL` confines a synthetic warm fraction to one visit.

Without `matched`, the existing unmatched cold behavior below is unchanged.

Select `tools/tune/workloads/cold-engram.json` with `mode: ab`. Cold cells are
labelled **cold, unmatched prompts**, excluded from automatic timing and
energy acceptance. They report each arm's TTFT, prefill execution time,
step.v1 `host.mapped_gather_us` and residency. Metric spread is max/min minus
one across visit medians. Priming is excluded; cold request latencies and
prefill steps can include drain when a prefill outlasts the measurement.
Such a window can have zero timed output tokens and no J/token result.

The tokenized pool is sorted by length and original index, then consecutive
nearest-length groups assign one prompt to each arm. Only the final group
is discarded if it has fewer prompts than arms. Each arm restores original
index order before splitting its prompts across visits, so window order is
not sorted by length. `cold_prompt_assignment` records indices, lengths and
each group's longest-minus-shortest token count. Request index determines its prompt,
so thread arrival order cannot change the assignment. Each prompt is used
once before deferred exactness replay. The report records actual used
lengths, because arms may complete different numbers of requests.

At staging, `cold_engram.request_budgets` reserves a priming and in-flight
cohort plus twice the planned 15-second request count for each window.
Warmup prompts are included. Staging computes the balanced capacity of each
visit and refuses an undersized pool before server startup; after
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
`TUNE_STUB_TOKEN_BYTES=4` exercises varied, more realistic token lengths.
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
