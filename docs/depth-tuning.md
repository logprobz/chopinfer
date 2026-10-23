# Tuning speculative draft depth

A drafting profile sets how many draft tokens each item may verify at each
concurrency with a balanced depth schedule: the profile's `speculation.depths`
(or `--speculation-depths`), for example `1:5,8:3,16:0`, gives depth 5 from
one live item, 3 from eight and none from sixteen. Plans never read step
times, so identical processes serve identical step shapes. The depth tuner
measures every depth at every concurrency on the real install and picks the
schedule. Use it for every model with a drafter, and again when a batch
changes drafting or verify cost.

## Inputs

- `models/<model>/depth-tune.json`: the served model name, checkpoint
  variable, tensor parallel size, test KV pages and the bundle's KV log line,
  serve options, speculation mode, deepest depth (the drafter's published
  block), verify row capacity, concurrencies, processes per depth, tokens per
  window, sampling, selection rule, power limit and work budget, and planning
  rates for the time estimate. See `models/deepseek41/depth-tune.json`.
- A workload under `tools/tune/workloads/`. `coding.json` weights code
  completion and code edits over a general chat share; prompts are built from
  the install's own source snapshot, so a revision always gives the same
  prompts.

## Running

1. Seal a stage outside the tree, on a frozen integration install
   (install-NAME, cache-NAME, source-NAME, logs/):
   `python3 -I tools/tune/stage.py --model models/<model>/depth-tune.json
   --install-dir ROUND_DIR $CHOPINFER_DATA/.../stages/NAME`. It prints the
   job's SHA256 and estimated minutes.
2. `python3 -I STAGE/harness/preflight.py STAGE` parses every depth's server
   options with the frozen executable, without CUDA.
3. Dry run on the CPU stub (with `local.env` sourced):
   `python3 -B STAGE/harness/tune.py --output STAGE/validation/dry
   --payload-sha PAYLOAD --dry-run`. Stub timings are synthetic; the dry run
   checks mechanics and failures only. The stub's controls
   (`TUNE_STUB_PERTURB`, `TUNE_STUB_PREEMPT`, `TUNE_STUB_IGNORE_DEPTHS`,
   `TUNE_DRY_POWER_W`) must each fail the job.
4. Queue `STAGE/tune.job` on an otherwise idle machine.

## What the job does and checks

An untimed warm-up process runs the deepest depth through every window and
keeps the reference outputs. Then `processes_per_depth` fresh servers per
depth, in alternating order, run one greedy and one fixed-seed sampled window
at every concurrency the depth can change (C x (depth + 1) within the verify
rows, every C at depth 0). Each window is one batched request, so all streams
start in the same step. Every output must equal the warm-up's bit for bit:
draft depth never changes an emitted token. Servers run the test KV mode with
zero preemptions and evictions, report their speculation mode, record memory
after warmup, and never draft past their depth. Power limits are recorded and
constant; SM clocks are recorded, never asserted.

## Selection

`tune.md` lists per concurrency and depth: request and full-concurrency
tok/s with spreads, J/token, stream p10, drafts per item, acceptance by
draft position and whether every process served the same shapes. A depth's
score is its request tok/s over depth 0's, weighted over greedy and sampled.
Among stable depths (identical shapes, spread within the limit) the
shallowest within the margin of the best wins (the larger of 1.5% and the
best depth's spread); a depth that serves the same plans as a shallower one
counts as that one. The emitted schedule goes into the profile's `depths`
with the run cited by date and numbers in the model README, and its landing
batch passes the stability gate like any drafting change.
