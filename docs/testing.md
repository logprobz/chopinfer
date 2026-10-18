# Testing

`bash ci/cpu.sh` is the single CPU acceptance script; what
it runs is listed in [contributing.md](contributing.md#build-and-verify). Use
the Rust 1.89.0 offline Cargo environment in [toolchain.md](toolchain.md). A
zero-test or skipped run is not evidence for a required check.

The reference checks (`tools/refrun`, each model's declared reference tests
and tapdiff) are required. They run with `CHOPINFER_REFRUN_PYTHON`, which the
`reference-prerequisites` step checks first: unset, or without torch,
transformers or numpy, it fails and names the remedy. `ci/quick.sh` and
`ci/cpu.sh --only` bring that step along with any reference step they run.
Each reference step must execute at least one test body (unittest counts
skipped cases as run; they do not count here), end with unittest's OK
verdict and skip only cases whose reason names the opt-in input that enables
them, a checkpoint variable (`CHOPINFER_CI_CHECKPOINT_TESTS=1`) or a
`CHOPINFER_*_CUDA=1` round trip, and no missing dependency (torch,
transformers, numpy or the reference runner). Its log ends with the executed
and skipped counts and each unexpected skip; a step that exits 0 with an
unexpected skip fails.

CPU fixture results establish ABI, lifecycle, transaction, and policy
behavior. They do not establish native model serving, GPU correctness, or
performance. The two fixture bundles (`bundle-tp4-fixture`,
`bundle-recurrent-fixture`) use synthetic output.

## How the CPU gate runs

`ci/cpu.sh` sets up the environment and runs `ci/cpu.py`, which runs every
check as a supervised step: each in its own process group, with its own log
printed when the step ends. A failed step stops the run (`--keep-going` runs
every step the failure does not block), and SIGINT or SIGTERM stops every step
with its child processes. `--only STEP` runs selected steps, and `--summary
FILE` writes each step's status, time, CPU time and peak memory as JSON.

Before its first step a run checks, in under a second, what its steps need
from the environment (`ci/preflight.py`), and refuses with exit 2 and each
problem named: `TMPDIR` must be an existing, writable absolute directory of at
most 64 bytes, and when the selection includes `portable` or `release-gate`,
every variable that holds a pinned reference must name a directory: each
variable an entry of `models/*/qual-references.lock.json` is kept under with
no default, other than the data root `CHOPINFER_DATA` (today
`CHOPINFER_QWEN38_LONG_REFERENCE`, `CHOPINFER_QWEN38_LAYER_REFERENCE` and
`CHOPINFER_GLM5NEXT_LAYER_REFERENCE`). With the data root set, it must also
hold every reference workload a model's `qual.json` keeps under it (today
Qwen's and GLM's image-probe workloads and MiMo's mixed media workload):
`runtime-eval`'s reference tests verify them whenever the data root is set,
and expect those references to be insufficient only while it is unset.
`--preflight` runs only these checks.
`ci/quick.sh` checks the same before it runs anything, the references only
when it runs the release gate or the tests of `runtime-eval` or the whole
workspace, and `ci/quick.sh --doctor` reports them. In a Git checkout a
`ci/cpu.sh` run also fails when HEAD moves while it runs: it checks HEAD every
10 s, stops its steps once HEAD reads as another commit, and checks again at
the end, since its steps would not all have tested one commit.

Steps share one worker budget, `CHOPINFER_CI_JOBS` (default: the CPU count).
The Rust tests take `CHOPINFER_CI_TEST_THREADS` (default 16) of it, the
geometry recipe and each qualification run reserve about a sixth (their
builds and module workers burst above that), and every other step takes one.
Lower the budget on a shared machine; `--serial` runs one step at a time.

A step starts only when its share fits beside the steps already running, and
the workspace tests (`portable`), the gate's longest step, become ready while
the recipes, the clean qualification gate and the long Python checks still
hold theirs. A test share near the whole budget therefore waits for those
steps: in 27 gates from 2026-10-13 to 10-15 run on a 48-core host with a
budget of 8 and 8 test threads, the workspace tests waited 625 to 1,451 s
(median 752 s) to start, about a third of the median 38.5-minute gate.
Replaying those gates' step times through the runner, a budget of 48 with 16
test threads starts them as soon as they are ready in 26 of the 27 gates
(101 s later in the other), while 24 threads makes them wait a median 270 s
for the long checks to finish. A machine that runs only the gate should keep
the defaults (its CPU count and 16 threads) rather than a small budget with a
matching thread count. The thread count changes only how many tests run at
once, never which tests run.

`ci/cpu.sh`, `ci/quick.sh` and the clean qualification gate build Rust code
with the profile in `ci/rust-profile.env`: opt-level 1 and line-tables-only
debug information, with debug assertions and overflow checks kept on. On
2026-10-05 it ran a Qwen geometry leg in 140 s instead of 260 s and the MiMo
bundle's library tests in 20 s instead of 103 s, for about 17% more compile
time, and the target directory shrank from 47 GB to 27 GB. Release builds are
unchanged. Plain `cargo build` and `cargo test` use the default dev profile,
so alternating them with the gates in one target directory rebuilds the
workspace each time; export the file's variables or give each its own
`CARGO_TARGET_DIR`.

Each Cargo feature variant replaces the binaries in `target/debug`, so the
steps that build into or read the shared target directory follow an order.
The profile CLI build runs first, and its binaries are copied for their
readers (qualification, model preparation, the external SDK smoke, serve
profiles and the reference checks, which score through the copy's
`chopinfer` as `CHOPINFER_EVAL`), which then run beside everything else. The portable checks and
workspace tests (`xtask check-portable`) follow the build; then the fixture
feature variants; then the kernel report integration and the feature variants
that build only test executables; and last, one at a time, the per-package
variants that rebuild `target/debug` binaries, each with its readers. The
geometry recipe and the clean qualification gate build in their own
temporary targets and run alongside from the start, as do the Python and
shell checks that need no build.

On 2026-10-05, on a 48-core machine shared with other work, the gate took
5.75 minutes with a warm target, 5.3 after an isolated Rust change, 6.0 after
a kernel source change with its lock update, and 10.9 with a cold target. The
previous serial script took 48.2 minutes with a warm target on the same
inputs, and the new gate with `--serial` 14.3. The geometry legs, which build
in fresh targets on every run, and the shared-target steps now take about
equally long. Ordinary qualification takes about 1.5 minutes instead of 12.

Kernel builds in fresh targets, including the geometry legs, can share cubins
through the content-addressed cache in `tools/nvcc_cache`: export
`CHOPINFER_NVCC_WRAPPER` and `CHOPINFER_NVCC_CACHE_DIR` as the
[kernel-build README](../crates/kernel-build/README.md#shared-cubin-cache)
shows. A hit replaces only the nvcc call, so every cubin is still audited and
compared with its lock.
On 2026-10-06, kernel-lab's 212 cubins built into a fresh target in 73 s
with an empty cache and 22 s from a filled one, with identical cubins and
verified locks.

### Affected checks

`ci/quick.sh` runs the gate's own Python and shell steps that a change
affects, from the step definitions in `ci/cpu.py`; there is no second list.
`ci/check_inputs.py` derives each step's inputs from its command: the scripts,
modules and discovered test directories it names, then the Python modules
they import or load, the scripts they name and the files, directories and
globs they name in string literals. A file that loads modules from computed
paths is taken to load any module in the directories it names. A changed file
among a step's inputs selects the step. The qualification run is limited to
the `tools/qual` test modules whose own inputs include a changed file. Quick
runs the profile build and copy first when it runs the qualification or
reference checks, which score with the copy as in the gate. The steps that
build Rust in fresh targets of their own (the geometry and export recipes,
the clean qualification gate and the CUDA toolchain check) are listed but
left to `ci/cpu.sh`. Unowned tooling is a file under `ci/`, `docs/`,
`tests/` or `tools/`, or a root file other than `Cargo.toml`, `Cargo.lock`
and the `rust-toolchain` files. When a Rust source names such a file, or its
directory, in a string literal, the change selects that source's package and
its dependents. Otherwise, when a selected step other than a Python compile
step covers it and neither a Rust include nor `ci/source_dependencies.toml`
reads it, it selects no Rust packages. Any other unowned file still selects
the whole workspace. The scan sees only literal paths and names, so only
`ci/cpu.sh` runs every check.

### Rust tests

`ci/rust_tests.py` runs the workspace tests for `xtask check-portable` and the
selected packages for `ci/quick.sh`. With cargo-nextest
([setup](contributing.md#build-and-verify)) each test runs in its own process
and tests from different binaries run at once. `.config/nextest.toml` groups
the tests that must not overlap: `shared-target` runs one at a time the tests
that rebuild fixture libraries, `build-*-fixture` programs or `chopinfer-host`
in the target directory, or run what those builds replace, and `nvcc` allows
two tests that compile CUDA at once. Add a new test of either kind to its
group. A package that sets
`package.metadata.chopinfer.lib-tests-in-one-process` in its `Cargo.toml`
keeps its library tests in one libtest process beside nextest, from the same
build; `bundle-mimo26` does, because its fake-host tests build plan caches
once per process. Doctests follow with `cargo test --doc`. With
`CHOPINFER_RUST_TEST_RUNNER=cargo`, or under `ci/quick.sh` without nextest,
the runner uses `cargo test`, which runs the same tests one binary at a time.
`ci/cpu.sh` runs the workspace tests under nextest unless that variable says
`cargo` or `auto`, and its preflight refuses to start when nextest is missing.

### Time budget

The `ci` nextest profile, which `ci/rust_tests.py` uses for the gate and
`ci/quick.sh`, stops any test after 10 minutes and fails the run: a test is
reported slow every 60 s and terminated at the tenth period. A test that
needs longer gets an override in `.config/nextest.toml`, with a comment
giving its reason and measured time and its own `terminate-after`, reviewed
like any other change: under `profile.ci.overrides` for a test the gate
runs, and under `profile.exhaustive.overrides` for an exhaustive test, so
running that test under `ci` keeps the 10-minute limit. First make the
gate's version cheaper, for example by comparing representative layers.
Five gate tests whose time follows host load (CPU oracles over large buffers
and nvcc builds) have a 20-minute `ci` override: full gates stopped them at 10
minutes at load 87 to 250 on a 48-core host, and the limit is at least twice
the longest passing gate run each had from 2026-10-08 to 10-15.

Exhaustive comparisons, such as port proofs and full-graph comparisons, stay
`#[ignore]`d out of the gate. `bash ci/cpu.sh --exhaustive` (or
`CHOPINFER_CI_EXHAUSTIVE=1`) adds the `exhaustive-comparisons` step after the
shared-target steps, and `bash ci/cpu.sh --exhaustive --only
exhaustive-comparisons` runs it alone. It runs the tests named in the
`exhaustive` profile's `default-filter`, under the `ci` budget and overrides
and the profile's own; add each new exhaustive test there. Qwen's full-graph
comparison has the one exhaustive override (60 minutes; it took 2,523 s in a
full gate on 2026-10-06), and the others took 3 to 8 minutes.

The tests that run under libtest rather than nextest have no per-test limit,
because stable libtest has none: the feature-variant `cargo test` steps, the
one-process library tests, the doctests and the `cargo test` fallback. On
2026-10-03 each of those steps took under 40 s.

At the end of a run, `ci/cpu.py` prints the ten slowest tests from the JUnit
reports nextest wrote during it (`target/nextest/ci` and
`target/nextest/exhaustive`). A full default gate, without `--only` or
`--exhaustive`, that takes longer than 15 minutes prints a warning; on
2026-10-06 the warm gate took 473 s, and no test reached 100 s.

### Qualification

`ci/qualification.sh` runs `ci/qualification.py`: each `tools/qual/test_*.py`
module in its own process under every gate config (`models/*/qual.json` and
the mechanics fixture), with its own temporary and receipt directories. A
cell passes only when its module ran tests and unittest reported OK.

A module that pins its gate config in `setUpModule` through
`tools/qual/qual_test_config.py` and declares `QUALIFICATION_RUNS_ONCE =
"reason"` runs once instead. Declare it only after an audit shows that
nothing in the module (imports, decorators, setup, tests and the child
processes they start) reads, copies or passes on the outer
`CHOPINFER_QUAL_CONFIG`, and that its test outcomes match under every config;
`python3 -B ci/qualification.py --matrix full --module NAME` runs it under
each. The runner refuses a declaration in a module that no longer pins its
config, and every undeclared module keeps the full matrix. `--discover` runs
the previous single-process discovery for comparison.

## GPU tests

Kernel-lab GPU tests run only when `CHOPINFER_KERNEL_GPU=<ordinal>` (or
`CHOPINFER_KERNEL_GPUS=<list>` for multi-GPU tests) names otherwise idle GPUs;
without it they print a skip notice. Run GPU benchmarks and serving work on
idle GPUs as well. A shared-device measurement is not performance evidence.
GPU receipts are written under `$CHOPINFER_KERNEL_RECEIPTS`, else
`$CHOPINFER_DATA/kernel-lab`, else a temporary directory, never into
`fixtures/`. MiMo kernel-lab tests plan their weights from
`CHOPINFER_MIMO_CHECKPOINT` in memory, through the bundle's weight plan and
the host's direct-load executor; nothing is written to disk.

## Explain a change before running checks

`bash ci/quick.sh --report` explains, for a change, which packages and
artifacts may change, which model profiles and contracts may be affected, and
which checks establish those contracts. It reads Git and Cargo metadata and
inspects files; it does not build, run tests, contact servers, use GPUs or
prepare packages. Exit zero means a report was produced, not that checks
passed. The report selects checks; `bash ci/cpu.sh` runs all of them.

```sh
bash ci/quick.sh --report --base main
bash ci/quick.sh --report --base main --format json \
  --output "$CHOPINFER_DATA/validation/change.json"
bash ci/quick.sh --report --all
```

`--output` writes a new file outside the repository; stdout is the default.
Commands in the report are templates: replace their named placeholders with
explicit inputs. A check requested but not covered by the suggested cohort
stays listed as an unresolved obligation.

Selection follows Cargo dependents (normal, build and dev edges, all
features) plus declared hidden readers, generated CUDA headers and contract
references. The inputs are:

- `ci/source_dependencies.toml`: hidden readers of source and data files.
- `ci/validation.toml`: shared change classes and check scopes.
- `models/<m>/validation.toml` and `models/<m>/qual.json`: each model's
  classes, profiles and the gate checks each change class selects.

Unknown ownership or incomplete metadata widens the selection, and unowned
tooling follows the rule in [Affected checks](#affected-checks), as in
`ci/quick.sh`. Ordinary documentation selects only `ci/cpu.sh`.

To include kernel consumers, pass existing kernel-build reports with
repeatable `--kernel-reports PATH` (a report file or a directory of reports,
such as `$CARGO_TARGET_DIR`) and `--previous-kernel-reports PATH` for reports
from before a deletion or dependency change. The reader uses actual `nvcc`
dependency membership from the
[producer contract](../crates/kernel-build/README.md) and checks current
sources, compiler, recipes, cubins and locks. Missing, stale or incomplete
reports keep the expected consumers in scope; they never shrink it. The
reader follows no report-supplied symlinks and bounds its reads.

## Evidence applicability

Repeatable `--receipt FILE` and an optional `--context FILE` let the report
check whether existing evidence applies to the current change. The context is
a JSON object of expected `candidate` (and, for comparisons, `baseline`)
identities; it is an expectation, not an observed receipt. The reader binds
the current source revision plus changed and untracked content, and for gate
checks the current policy and check implementation.

Each obligation reports selection (`selected`, `not_applicable`), execution
(`not_run`, `executed`), recorded result (`passed`, `failed`, `skipped` or
absent) and applicability (`current`, `missing`, `stale`). Missing
identities, partial goldens, zero comparisons, skips, failed subprocesses and
incomplete traces cannot satisfy an obligation, and a historical pass can be
stale. CPU/fake-host mechanics, GPU correctness, fixed-batch exactness, model
quality and exclusive-device performance are separate claims; an unchanged
cubin cannot clear a changed host, and one checkpoint's accuracy evidence
does not transfer to another.

The report grants no release qualification or gate waiver. Serving,
model-quality and multi-profile GPU claims currently remain `missing` until
producers record complete per-profile identities.

`bash tests/encoder-membership.sh` runs the CPU encoder membership gate.
It uses the bundle build's shared symbol deduplication and geometry membership
derivation, including the shared-audio fixture, to check encoder availability
at every supported rank count without a GPU.

## Evidence tools

Use [journal proofs](../tools/journal_proof/README.md) when kernels or launches
change: capture the model's pinned fake-host journals at the base and tip
commits, then prove identity or only the changes a reviewed spec declares.
A served-layout change (added allocations, new collectives, changed grids) is
proven from a [layout declaration](../tools/journal_proof/README.md#layout-declarations)
committed with it. State the report's claim and coverage in the commit
message.

Use [serving smoke](../tools/smoke/README.md) for a base-against-tip check of
one model through the serving runner's smoke mode. Its `paths.txt` lists the
pathspecs that later changes to main must not touch for the smoke to still
apply.

Use [serving A/B](../tools/ab/serving/README.md) to compare two prepared
installs on a fixed workload in counterbalanced order (ABBA or BAAB). By
default it requires bitwise equality of token IDs and logprobs and no drop in
speculative acceptance; timing is report-only.

Use [`tools/qual`'s `release_preview`](../tools/qual/README.md) as the one
validation entry point for a release of any model; see
[releasing](releasing.md).

## Geometry recipe

Each model declares its legs in its `model.toml` `[geometry_recipe]` table,
one per fixture under `fixtures/<model>/geometries/`
([in-tree maintenance](adding-a-model.md#in-tree-maintenance)).
`bash tests/model-geometry-recipe.sh --ci` runs the bundle-stage legs.
`--contract` runs the legs of models whose bundle cannot yet build and serve
an added geometry, which stop at the contract stage; CI runs them as the `geometry-recipe-contract`
step. A model name runs that model's legs. Each uses a temporary worktree at a clean committed HEAD and the
offline CUDA build environment. The legs run in lanes at once
(`CHOPINFER_GEOMETRY_LANES`, default 2): a model's legs share one lane and
temporary target, so the second reuses the first's dependency builds. One
lane runs every leg in turn; three give each leg its own target. Each lane
gets an equal share of `CHOPINFER_GEOMETRY_JOBS` (default: the CPU count) for
Cargo jobs and, through `CHOPINFER_KERNEL_BUILD_JOBS`, kernel-build's nvcc
workers. The
[data-only recipe](adding-a-model.md#same-architecture-checkpoints) assigns a
fresh geometry ID and kernel tag, permits only the appended geometry and
architecture alias in `supported.json`, checks byte-identical existing lock
records and the new records' tag, and exercises catalogs, independent encoder
plan sizes (including Flash's vision MLP padding), text view sizes,
nonoverlapping destinations, helper admission, package emission
and fake-host lifecycles at TP2 and TP4. The MiMo leg also checks data-only FP8
calibration and BF16-only preparation without calibration.

Qwen requires pinned tokenizer files in `CHOPINFER_QWEN38_TOKENIZER_DIR`, with
`CHOPINFER_QWEN38_CHECKPOINT` as fallback. MiMo uses
`CHOPINFER_MIMO_TOKENIZER_DIR` or `CHOPINFER_MIMO_CHECKPOINT` and additionally
requires `mask_embedding.pt` through `CHOPINFER_MIMO_DFLASH_DIR` or the
checkpoint/tokenizer directory's `dflash/` subdirectory. The `--ci` wrapper
preserves CI's configured environment. No GPU is used, and these checks do not
establish numerical qualification. `python3 -B tests/test_model_geometry_recipe.py`
checks the recipe's guard mechanics.

## Export recipe

`bash tests/model-export-recipe.sh --ci` checks that a quantized export whose
format mix no tested layout uses is served with data alone. In a temporary
worktree at a clean committed HEAD it synthesizes a sparse mixed export of a
compiled geometry from the geometry generator's layout, rebuilds the bundle
with `KERNELS_LOCK_UPDATE=1` and regenerates the projection adoption ledger.
It then probes the export at TP2 and TP4, prepares it at TP2 with the MTP
drafter, checks every operation the package's precision components record
against the export's quantization rules and creates each topology on the fake
device with its MTP profile's speculation, as `serve` would; at TP2 a create
without speculation shows the drafter's runtime buffers. The tree, ignored
files included, may then differ from HEAD only in the kernel lock and the
adoption ledger, with every existing record byte-identical and the schedule
rows unchanged. Negative controls in the same tree (a source edit, new source
files including under ignored paths, a rewritten lock or adoption record, a
changed schedule row) must each fail that assertion. It uses the shared worktree and ledger checks of the
geometry recipe (`tests/data_only_recipe.py`), the same pinned tokenizer files
and no GPU, and it does not establish numerical qualification.
`python3 -B tests/test_model_export_recipe.py` checks its synthesis,
recorded-format and data-only guards.
