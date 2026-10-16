# Base-against-tip serving smoke

`smoke.py` checks that a change serves one model bit for bit as before: it
builds the model at a base and a tip commit and serves each leg as one AB
pair through the serving runner's smoke mode
([tools/ab/serving](../ab/serving/README.md)). Any model with a
`model.toml` and `profiles.json` works; nothing is model-specific.

```sh
python3 -B tools/smoke/smoke.py legs MODEL
python3 -B tools/smoke/smoke.py build MODEL BASE TIP --out "$RESULTS/SMOKE" \
  [--legs PROFILE,...] [--checkpoint DIR] [--job-prefix NAME]
# then queue the job files in $RESULTS/SMOKE/jobs/; each job runs
python3 -B "$RESULTS/SMOKE/runner/tools/smoke/smoke.py" run "$RESULTS/SMOKE" LEG...
```

## Legs

A leg is a serve profile of `models/MODEL/profiles.json`. The defaults are
the TP4 autoregressive, TP4 speculative and TP2 autoregressive profiles the
model declares, a qualified one first; `--legs` names others (for example a
context-parallel profile). A leg serves its whole profile through the
runner's `serve_profile`: every numerical option (sizes, limits, KV dtype,
prefix cache, scheduling and model options) is passed explicitly, `/health`
must report the profile's values (a declared context-parallel size
included) and the prepared package must declare its topology and option
values. A speculative profile serves at its draft depth in fixed mode, so its
verify shapes do not depend on timings.

Each leg's phases are the `tools/qual/determinism.py` request set (four short
prompts, a long prompt sent cold twice, as token IDs and as prefix-cache hits
at two anchors, and the image chat request when the server serves images), a
decode prefix warmup, and single-stream decode of 128 tokens. A leg passes
when the two records are identical by `determinism.py compare`, the decode is
bit-identical, and with speculation the tip's accepted rate is not lower than
the base's.

Concurrent streams are not in the default set: their batch composition
follows arrival timing, and a model need not answer them bit for bit alike
across runs. On 2026-10-12 one TP4 install answered eight concurrent decode
streams differently in two runs, while its single-stream answers were
identical. `build --phases` adds such phases (any catalog names of the
runner) to the default set for a model that serves them reproducibly; the
runner refuses a smoke run file without the default set.

## Build

`build` runs on the CPU. Per commit, under `OUT/arms/COMMIT/`:

- `src/`: `git archive` of the commit, extracted with `tar --touch` so a
  reused target cannot keep stale outputs;
- `target/`: the arm's own release target for `runtime-host`, the model's
  compiler and its bundle (with `CHOPINFER_BUILD_REVISION` set to the commit,
  and the shared cubin cache when `CHOPINFER_NVCC_CACHE_DIR` is set);
- `install/`: `bin/chopinfer`, `bin/chopinfer-model-MODEL` and the bundle
  library with an `INSTALL.json` receipt;
- `cache/`: a direct package per tensor-parallel size, prepared by the arm's
  own compiler;
- `READY`, written after a second build compiled nothing.

A READY arm is kept only when READY and its install receipt name the
requested commit, the model, the checkpoint and the build command, its
install passes the runner's install check and it prepared every needed TP;
otherwise it is rebuilt. Each run file binds each install to its requested
commit, and the server must report it. The checkpoint
is `--checkpoint` or the model's `CHOPINFER_<checkpoint_env>_CHECKPOINT`.

Qualification reads `OUT/qualification/TIP/`, a separate detached Git
worktree at the exact requested tip with its authentic history. Build exports
have no Git metadata and cannot validate reference adoption ancestry. The
run's `models`, `qual_tools`, model root and qualification config all name
the pinned checkout, including when both build arms are already READY.
The CPU build preflights its actual frontend and reference-lock loaders
with the new install's scorer before writing queue jobs. `pair.json` records
the source commit, tree and path. Each leg checks that identity and a clean
checkout (including ignored files) before and after serving. It refuses a
missing, changed or shallow source, or an arm overriding its qualification
paths. It does not read qualification from the mutable parent checkout.

Then `build` copies the runner and this tool into `OUT/runner/`, writes one
run file per leg in `OUT/legs/` and dry-runs it, writes `OUT/paths.txt`, and
writes job files in `OUT/jobs/`: legs of one device count per job, as many as
fit 1740 s at `--start-seconds` (default 120) per server start, with
`# timeout 1800` and `# gpus N`.

`OUT/paths.txt` lists the smoke-staleness pathspecs of the tip: the local
packages (inside the repository, workspace members or not) that the server,
the model's compiler and its bundle build from (normal and build
dependencies, from `cargo metadata` with the build's features), files
outside those packages that they embed (`include_str!` and the like) or that
the tip build read (its dep-info and build-script `rerun-if-changed` files
and directories), the workspace manifests and lock, `.cargo/`, the
toolchain files, `tools/nvcc_cache/` (the compiler wrapper the build selects)
and `models/MODEL/`. `smoke.py paths MODEL [--target DIR]` prints the list for
the current checkout, with the dep-info of a build of it when given.

To check whether a smoke of BASE still applies after main moved, pass each
line as one pathspec (entries may contain spaces); any output means the
smoke is stale:

```sh
mapfile -t specs < <(grep -v '^[[:space:]]*\(#\|$\)' OUT/paths.txt)
git diff --name-only BASE main -- "${specs[@]}"
```

## Run

`run` serves the named legs in order on the GPUs the queue exported
(`CUDA_VISIBLE_DEVICES`), each on a free port, with the job's remaining time
shared by its remaining legs. Results go to `OUT/runs/LEG-UTC-PID/` (the
runner's result directory) and `OUT/runs/UTC-PID.summary`, one `PASS LEG` or
`FAIL LEG: reason` line per leg. It exits 0 only when every leg passes.

`test_smoke.py` covers the legs, run files, job packing, paths and `run` on
the runner's CPU fixtures; it runs in `ci/cpu.sh` as the `smoke-tools` step.

## Existing build arms and source cleanup

For a pair built before qualification sources were pinned, run on CPU from
the current tools checkout, with no job using that output directory:

```sh
python3 -B tools/smoke/smoke.py prepare-qualification "$RESULTS/SMOKE"
```

This uses the existing `pair.json` tip, authenticates the existing new
install, loads the real frontend and lock, and dry-runs each proposed leg.
Only after those pass does it repin the runner and qualification paths.
Previous pair, leg and runner files are saved under
`OUT/qualification-updates/`; `arms/`, READY files, `paths.txt` and previous
`runs/` are untouched. It does not rebuild a model or enqueue a GPU job.

The qualification worktree stays locked while queued jobs may need it.
After the smoke artifacts no longer need that checkout, unlock and remove
it through Git before removing the output directory. `TIP` is the full
commit in `pair.json`:

```sh
git -C "$REPOSITORY" worktree unlock "$RESULTS/SMOKE/qualification/$TIP"
git -C "$REPOSITORY" worktree remove "$RESULTS/SMOKE/qualification/$TIP"
```
