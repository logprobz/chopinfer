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
  per window, prompt groups per concurrency, kinds (greedy and sampled
  weights), sampling, selection rule, power limit and work budget, planning
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
3. Dry run on the CPU stub (with `local.env` sourced):
   `python3 -B STAGE/harness/tune.py --output STAGE/validation/dry
   --payload-sha PAYLOAD --dry-run`. Stub timings are synthetic; the dry run
   checks mechanics and failures only. The stub's controls
   (`TUNE_STUB_PERTURB`, `TUNE_STUB_PREEMPT`, `TUNE_STUB_IGNORE_DEPTHS`,
   `TUNE_DRY_POWER_W`) must each fail the job.
4. Queue `STAGE/tune.job` on an otherwise idle machine.

## What the job does and checks

A depth schedule is fixed for a server's lifetime, so the job runs one
server per arm and reuses it for every window: by default one arm per depth,
the deepest first. Each server borrows its weights from the weight keeper
that the wrapper's `# keeper use keeper-spec.json SHA256` header asks the
queue runner for, and loads them when no keeper answers. The first arm runs
an untimed reference pass through every window, which also warms the page
cache, and keeps its outputs. Each arm then runs `repeats` passes, in
alternating order, of one greedy and one fixed-seed sampled window (as the
`kinds` select) at every concurrency the depth can change (C x (depth + 1)
within the verify rows, every C at depth 0). Each window is one batched
request per prompt group, so all streams start in the same step; low
concurrencies serve several groups in turn so one prompt does not decide
them. An arm starts only while its reserve of the work budget remains. Every
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
is its request tok/s over depth 0's, weighted over greedy and sampled.
Among stable depths (identical shapes, spread within the limit) the
shallowest within the margin of the best wins (the larger of 1.5% and the
best depth's spread). `python3 -I tools/tune/report.py RUN_LOW RUN_HIGH`
merges split jobs into one schedule. The emitted schedule goes into the
profile's `depths` with the run cited by date and numbers in the model
README, and its landing batch passes the stability gate like any drafting
change.

## Step costs per verify width

A job with `mode` `costs` runs the same arms without verify widths, so
every depth verifies its exact rows (C x (depth + 1)), with the token
capacity and graph budget under test in its `extra_arguments`. `costs.md`
and `costs.json` (`report.py --costs RUN`) list, for every (items, rows)
step shape, the median and p10/p90 cycle time from the previous step's
completion and the median execution time, the windows' tok/s and J/token,
and each arm's GPU memory. The memory after the warm windows less the memory
at readiness bounds the graphs those windows captured. A model config's
`arms` can instead name other settings per server, such as two graph
budgets, to sweep them in one job.
