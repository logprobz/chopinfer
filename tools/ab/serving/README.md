# Counterbalanced serving A/B

`run.py` serves two installs of one model (`base` and `new`) in owned,
counterbalanced arms on the same GPUs, captures a fixed workload from each
arm, and refuses incomplete evidence. It answers one question: does `new`
serve the configured workload bit for bit as `base` does, with no lower
speculative acceptance? It needs Python 3.11 or later, Bash and Linux with
`/proc` and pidfds, and uses only the standard library. It does not build,
install or prepare a model, and never stops a service it did not start.

```sh
python3 -B tools/ab/serving/run.py "$RUN_FILE" --dry-run
# Only inside an admitted GPU job, after the dry run succeeds:
python3 -B tools/ab/serving/run.py "$RUN_FILE" --out "$RESULTS/UNIQUE_RUN"
```

Exit 0 means every configured gate passed; 1 means a gate or the run failed
(`result.json` names the errors and whether ownership is retained); 2 means it
refused before any arm started. The result directory must be new and outside
a repository. The dry run checks the installs, caches, configuration and
phases, prints the resolved plan and starts nothing.

## Run file

A JSON object with `"schema": "chopinfer.ab-serving.v1"`. Relative paths
resolve against the run file; path strings expand environment variables and
`~`. Argv and environment values are literal strings.

- `model`: the model ID. The runner reads `models/<model>/model.toml` and
  `profiles.json` (or under `models`, another models directory) for the files
  an install must hold (`bin/chopinfer`, `bin/chopinfer-model-<model>` and
  the bundle cdylib under `lib/`), the bundle ID and the declared drafters.
- `checkpoint` and `checkpoint_identity`: the checkpoint directory and the
  prepared identity `/health` reports.
- `installs.base` and `installs.new`: `path` (an install with `INSTALL.json`:
  a 40-character `commit` and a `files` map of every file under `bin/` and
  `lib/` to its SHA-256) and `cache` (a model cache prepared by that
  install). Optional `commit` (the receipt must name it), `serve_args`, `env`
  and `expect` per install.
- `port`, `gpus` (indices or UUIDs the queue admitted).
- `serve_profile`: optional; a profile of the model's `profiles.json`. Its
  options (tensor-parallel and context-parallel sizes, limits, KV dtype,
  prefix cache, scheduling, draft cost and model options) are passed to both
  installs explicitly, as the host expands a profile (`serve --model`
  refuses `--profile`); the variant sets speculation. `/health` must report
  the profile's tensor-parallel size, sequence limit, prefix cache, explicit
  KV dtype, declared context-parallel size and model options, and the
  selected package must declare the profile's topology, context-parallel
  size and option values. `serve_args` may add options, not repeat them.
  Without a profile `serve_args` defaults to `--tensor-parallel-size` with
  the GPU count.
- `variants`: `ar` (`"mode": "off", "tokens": 0`) and optionally `spec`
  (`"mode": "fixed"` or `"adaptive"`, `tokens` 1 to 16, `drafter`, default
  the model's one declared drafter). A speculative variant needs the AR
  control. Default: `ar` only.
- `order`: `["base", "new", "new", "base"]` (default) or BAAB; a single AB
  pair is refused.
- `phases`: `standard`, `short` or a list of catalog names (`evidence.py`):
  prefills of 32K, 9K and 3K tokens, teacher forcing at 64, 1040 and 2060
  tokens, a decode prefix warmup, decode at concurrency 1, 4, 8 and 16 with
  128 tokens each (and `decode-c32`, which only a list names: `standard`
  and `short` leave it out), `gsm8k-c1` and `gsm8k-c8` (eight frozen questions, which
  need `client_assets` naming `gsm8k-examples.json` and
  `gsm8k-manifest.json`), and `determinism`: `tools/qual/determinism.py
  record` against the arm's server (with its image request when `/health`
  lists the image modality), scored for both arms by the new install's
  `chopinfer`, matching the shared qualification tools; two arms' records
  are compared with `determinism.py compare` using that same scorer.
  `qual_tools` names another `tools/qual` directory; its files are pinned in
  the plan.
- `smoke`: `true` serves one variant as one AB (or BA) pair; the gates are
  base against new on every phase and, with speculation, the acceptance
  floor. Its phases include `determinism`, `warm` and `decode-c1`
  (`evidence.SMOKE`), and `base_new` must be `bitwise`. `tools/smoke` writes
  such run files.
- `base_new`: `bitwise` (default) or `report-only` for a declared numeric
  change; the repeat and AR/speculative comparisons stay bitwise gates.
- `acceptance.max_rate_drop`: absolute accepted-rate allowance (default 0;
  a nonzero value needs a `reason`).
- Optional `serve_args`, `env`, `expect` (a subset of `/health` `serving`),
  `seed`, `budget_seconds` (at most 1740), `timeouts`, `phase_timeout`,
  `telemetry_interval`, `query_command` (default `nvidia-smi`), `client` and
  `profile` (an `nsys` command and arguments).

## What it checks

- Installs: the receipt covers exactly the files under `bin/` and `lib/`, no
  symlinks or group- or other-writable entries, digests match. Each arm runs
  from a private read-only snapshot, rechecked before and after the arm, and
  the running executable is checked through `/proc`.
- Packages: the cache holds a complete prepared package for the checkpoint
  whose native module is the install's bundle; `/health` must name that
  package, the install's commit, the bundle ID, the checkpoint identity, the
  variant's speculation and drafter, the package's checkpoint layout when the
  package declares one, and this launch's logged startup identity and nonce.
- Prompts are sized through the server's `/tokenize` to the largest word
  count at most each phase's token target, so every model gets the same
  prompt lengths; the catalog's prompt-token bounds follow from that.
- Every prefill, teacher and decode request records its generated token IDs
  (one per completion position) and finish reason; a decode stream must
  also end with `[DONE]` and carry no error event. The decode prefix warmup
  must complete too (one choice with its finish reason and token). GSM8K records answers,
  generated token IDs and counts (no logprobs), compared as equal values.
- `compare.py` validates complete coverage on both sides and compares gated
  outputs (token IDs, text and logprobs) by IEEE bits
  (with equal JSON types at every node): base
  against new in each order, each install's two repeats, and with
  speculation the new install's AR against speculative arms.
- Speculation counters before and after each phase: complete, consistent
  totals, position bins and effective-K histogram. A fixed-K arm must verify
  at its configured K in every decode and GSM8K phase. Per phase, new's
  accepted rate must be at least base's minus the allowance.
- Ownership: every arm gets an empty GPU query and a free port first; each
  launch is its own session, tracked by birth identity, and is stopped and
  shown released (an empty GPU query) before the next arm. A failed release
  stops the run and keeps the ownership receipts.

Timing (per-arm phase times, new/base ratios in both orders, GPU clocks and
power) is report-only.

## Self-tests

`test_serving.py` (stub server, client, GPU query and profiler on loopback,
dummy processes) and `test_compare.py` (synthetic captures) run in
`ci/cpu.sh` as the `ab-serving` step. They use no GPU.
