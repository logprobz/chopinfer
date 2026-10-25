# Kernel library

Chopinfer's kernels live in one base per NVIDIA ISA family. The first family
is SM120, built as `sm_120f` cubins. The kernels are tuned for and tested
only on the RTX PRO 6000 Blackwell (96 GB, 188 SMs: `SM_COUNT` in
`kernels-sm120`). The runtime refuses a device that is not compute
capability 12.0 exactly (`runtime-cuda`), so SM121 parts such as DGX Spark
are refused; other compute capability 12.0 parts, such as GeForce RTX 50
series cards, are untested, and packages plan 94 GiB of device memory per
rank, which smaller cards do not have ([hardware](../README.md)).
The numerical gates are in [numerics](numerics.md); for independent model workspaces, see
[external models](adding-a-model.md#external-sdk-workflow).

## Crates

| Crate | Role |
|---|---|
| `kernel-core` | Op contracts, typed queries, instantiation specs, launch plans, codecs, reference quantizers, weight transforms, f64 oracles, exactness properties and the `Backend` trait. No CUDA. |
| `kernel-build` | Build-script library: nvcc driver, cubin audits, source membership and ownership reports, entry codegen, kernels.lock. |
| `kernels-sm120` | CUDA C++ engines under `cuda/`, launch planners, instantiation catalogs, tuning tables and the `Sm120` backend (`src/backend/`). |
| `kernel-harness` | The model-free GPU test harness: sessions and GPU selection, launches, poisoned replays, conformance runs, the conformance kit and encoder conformance. It names no model and builds no cubin; a consumer passes its cubins (`Cubins`), catalogs (`&Catalogs`) and cases. Links `runtime-cuda`. |
| `kernel-lab` | GPU probes, correctness tests and benchmarks of the in-tree models' catalogs and the lab's own cubins, on `kernel-harness`. New models do not register here; they test from their own crates on the harness. |

Bundles depend on `kernel-core`, `kernel-build` (as a build dependency) and
`kernels-sm120`. Bundles never link `runtime-cuda`; the host loads their
cubins and launches through the bundle ABI.

## Kernel locks

`kernel-build` accepts two lock forms.

- **Lock file** (`build`, `build_with_context`, `KernelSelection::build`): one
  sorted `kernels.lock` per consumer. Every record gates on its full identity,
  including `source_closure_sha256`, and the whole file must match. Every
  existing lock uses this form and keeps its bytes until it is migrated.
- **Lock directory** (`build_in_families`, `KernelSelection::build_in_families`):
  a `LockDir` assigns each built cubin, by name, to exactly one named *source
  family*. Each family owns `<family>.lock` (gating records) and
  `<family>.provenance.json` (source-closure digests) in the directory. A build
  compares, or under `KERNELS_LOCK_UPDATE=1` rewrites, only the families of the
  cubins it built, each on its own, and never opens another family's files.
  Branches that add different families therefore merge without conflict. One
  build call produces all of a family's cubins.

In the directory form a record is the compiled cubin plus its ABI: name,
toolchain version, nvcc hash, flags, defines, entry ABI (symbol, parameters,
launch bounds, local-byte allowance), `cubin_sha256` and the resource report
audited from that cubin. The source closure is provenance. A header-only edit
that leaves every cubin byte-identical passes with a Cargo warning naming the
family and cubins, and `KERNELS_LOCK_UPDATE=1` refreshes the provenance file.
Any gating difference fails, naming the family, cubin and field (for example
`tiny.cubin_sha256` or `tiny.entries[tiny_v1].max_threads`); a cubin change
still needs bitwise kernel-lab evidence before an update. A missing family file
fails with the update hint. Profiling builds bypass both forms.

New engines and new bundle locks use the directory form. A crate keeps its
existing `kernels.lock` for existing specs and builds new families in a second
call with its own output subdirectory (one build report per output directory),
for example `crates/kernel-lab/kernels.lock.d/`. Existing locks migrate later in
one reviewed commit that shows every `cubin_sha256` unchanged.

### Adding a kernel-lab lock family

Each family of `crates/kernel-lab/kernels.lock.d/` is one file of
`crates/kernel-lab/families/` (crate `kernel-lab-families`), whose build script
lists that directory. kernel-lab's build script builds every listed family in
its one lock directory build. Adding a family adds files and edits no shared
list, so branches that add different families merge without conflict:

1. Add `crates/kernel-lab/families/<family>.rs`. The name is lowercase ASCII
   letters, digits and `_`, starting with a letter. The file defines
   `pub const FAMILY: &str = "<family>";` (a constant assertion holds it to the
   file name) and `pub fn cubins() -> Vec<kernel_build::CubinSpec>`, every
   cubin the family's lock gates. Keep the family's lab profile, geometry and
   cubin names in the same file. Return cubins whose names tests load through
   `crate::declared(&[CUBIN, ...], specs)`, which fails the build when the
   built names differ.
2. Tests reach the family's items as `kernel_lab::families::<family>`, for
   example `session.module(kernel_lab::families::<family>::CUBIN)`. Do not
   include the file with `#[path]`: it names `crate::declared`.
3. Create the family's lock pair with
   `KERNELS_LOCK_UPDATE=1 cargo test -p kernel-lab --no-run --locked --offline`,
   which writes `kernels.lock.d/<family>.lock` and `<family>.provenance.json`.
   Commit them with the family file and the kernel's evidence.
4. `cargo test -p kernel-lab --lib family_tests` fails on a lock directory file
   with no family file and on a family missing either file. To remove a
   family, delete its file and its lock pair in one commit.

The older families still include their `<x>_shapes.rs` files beside
kernel-lab's build script, which their tests include by path.

## Adding an op

An op is model math first, then each backend's implementation of it.

1. Contract and oracle. Add the contract (a requirement or query that
   implements `Validate`) and its f64 oracle to `kernel-core`, with a
   numerical contract ID, and a record of its buffers by role declared with
   `kernel_core::scratch_record!` (`RmsNormBuffers` is a small example).
   `Validate` checks math only: hardware limits such as a shared-memory
   bound, a head-dimension set or a compiled width are backend refusals. Pin
   numerics (a reduction order, a rounding point) only where a model needs
   them to match its reference; otherwise the backend chooses and is judged
   against the oracle. Bit-exact ops get exact tests; others use the
   numerical contract bound.
2. `Backend` method. Add one method to `kernel_core::backend::Backend`, in
   its family's section, that returns a `Prepared` op over the role record
   of `View`s and refuses with `Refused::Unsupported` by default, so no
   existing backend breaks. Its rustdoc says which `Counts` fields its
   launches read and what bounds the entries of any device table they
   index. Add the method to the default-refusal tests (`TESTED`);
   `every_method_has_a_default_refusal_test` fails until it is listed.
3. SM120 implementation. Add the device code to the owning engine in
   `kernels-sm120/cuda/` as a constexpr-specialized variant, not a new
   engine, when one fits, with its typed planner and builder beside the
   engine. Implement the method in `crates/kernels-sm120/src/backend/<family>.rs`:
   register entries on `Sm120` with a `with_*` method, choose among them by
   the rule below, apply SM120's limits at prepare (`backend::check_plan`
   adds the shared-memory bound to `LaunchPlan::validate_geometry`), and
   return an op whose `launches` only plans and packs arguments.
4. Kernel-lab. Add a conformance-kit case (`kernel_harness::kit::cases`: the
   op's roles over `Spec`s and its oracle; see
   [running the kit](#running-the-conformance-kit)) and a benchmark case.
   A case whose rows are items of several rows (a verify step's attention
   queries or recurrent sequences) declares `item_rows` and gives the
   `Decoded` mapping of its verify rows to the decode op's one-row
   launches; a collective gets a multi-rank case (`kit::collective`).
   Declare the gate before running on a GPU. Its relaunch, graph and PDL
   checks poison the outputs and workspaces before every compared launch
   with `kernel_harness::poison` (see `crates/kernel-lab/README.md`).

An op that only one model runs stays with that model, behind its `OwnOps`
trait ([model-owned ops](#model-owned-ops)), and moves to `kernel-core` when
a second model needs it.

### How SM120 chooses an entry

A model's `sm120` module only registers entries on `Sm120` (`with_*`):
the entries its build compiles, as data. It never picks a symbol for an op;
the backend does, by one rule for every family, and the choice does not
depend on the order the entries were registered in
(`crates/kernels-sm120/src/backend/select.rs`):

- The family decides which entries match the op from the op's own fields:
  its contract record and, where the entry depends on how the step runs, its
  use (phase, capacity, graph).
- Of those, the entries that hold `need.capacity` rows and whose
  declarations claim `need.exact` remain. The least capacity wins, then the
  least rank the family gives.
- Entries left that would launch the same count as one. The op claims the
  meet of their declarations, and the least whole declaration names it, so
  claims and the implementation name are order-independent too.
- Two entries left that would launch differently are a tie the op does not
  resolve: prepare refuses, naming both.

"Would launch the same" compares a launch key each family writes by hand
from the launch's own parts: symbols, instantiation, plan and argument
shape, but not declared claims. Key guards keep the keys honest. A test
destructures every field of the family's entry type (`field_names!`), so a
new field fails to compile until it is classified, and
`keys_follow_launches` changes each field alone and checks that it behaves
as its class says: it changes the launches (`Launch`), is a limit the key
holds (`Bound`), changes nothing (`Inert`, `Unread`) or no longer matches
the op (`Selects`).

## Adding a GPU backend

Implement `kernel_core::backend::Backend` for the target. Generic model code
uses that trait and its own `OwnOps`; target modules register compiled
entries. Contract methods and refusal rules are documented in
`crates/kernel-core/src/backend.rs` and `properties.rs`.

Prepare performs selection, tuning and validation once. Per-step `launches`
only checks buffers/counts and packs launch arguments. `Need` records capacity
and required invariance; `OpInfo` declares implementation identity,
guarantees, role sizes, symbols, weight layouts, persistent state and scratch.
Scratch is uninitialized and not preserved between steps. Never claim
properties the kernel does not keep. Missing families refuse by default;
unsupported shapes or precision refuse at prepare.

The host sampler and device admission also need target support. They are
outside the backend trait: inspect `runtime-kernels`, the host sampler and
`runtime-cuda` before claiming a second backend serves a model. Packages
must declare supported architectures and capabilities; integrity validation
does not establish correctness of arbitrary native kernels.

### Model-owned ops

An operation used by only one model keeps its contract, oracle and kernel in
that model's `OwnOps` implementation. Shared runtime crates remain model-free.
Tuning is target-owned, selected at prepare and represented in the chosen
implementation identity.

### Running the conformance kit

Use `kernel_harness::kit::conform` for prepared operations and
`kit::collective::conform` for multi-rank operations. Mark required ops
`.required()` and assert the expected prepared count. Refusing every op is
not a passing backend. Read `Report::failures` and `Report::not_exercised`.
Without a GPU session only CPU declaration/need checks run.

```sh
CHOPINFER_KERNEL_GPU=0 CHOPINFER_KERNEL_REQUIRE=1 \
  cargo test -p kernel-lab --release --locked --offline --test kit _on_a_gpu \
  -- --test-threads=1 --nocapture
```

Select idle devices first. Collectives use `CHOPINFER_KERNEL_GPUS` with one
device per rank and the `kit_collectives` suite. Verify operations over
multi-row items need a `Decoded` mapping to one-row decode launches, so the
kit can prove `Rows::OneRow`. A missing mapping or untested property is not
equality evidence. A conformance pass covers the tested operation shapes,
not a whole model, topology or performance.

## Declarative conformance runs

`kernel_harness::conformance` runs declared inputs, outputs, scratch and
state through an oracle, poisoned relaunch, graph replay and PDL checks.
`Case::validate` must pass on the CPU for every configuration. Declare zero,
partial and full live counts, bounded write maps, exact ID/high-ID/hole
witnesses and every output the oracle must judge. Numeric similarity alone
cannot witness an indexing change. `Case::refusing_runs` supplies CPU
negative controls; a null launch must fail wherever something must write.

A chain's metadata only shows that each later stage reads some earlier
write. It does not prove every input was initialized or every PDL wait was
placed correctly. Output checks see the final chain, so a later write can
hide an earlier overrun. PDL overlap is not guaranteed; a passing run is not
a proof that early reads are impossible. Keep GPU negative controls and
inspect which phase actually detected them. See the conformance API and
[kernel-lab checks](../crates/kernel-lab/README.md#poisoned-replay-checks).

## Model-owned selections

Keep mathematical requirements, selected implementations and qualification
separate. A shape match or coarse numerical ID is insufficient. Shared
builders derive declarations from typed requirements; model-owned entries
supply their own declarations and correctness cases. Build validation proves
recipe membership, not arbitrary CUDA's mathematical truth.

## Generated projection offers

Derive offers from a model's [projection declaration](../crates/projection-spec/README.md)
through [generated precision offers](../crates/kernels-sm120/src/precision/README.md).
Symbols follow content; existing aliases are kept in append-only adoption
ledgers. A migration to generated offers preserves existing lock records.
New entries require oracle and invariance evidence. Execution binds precision
handles once and retains the selected implementation.

## Routed experts split over the ranks

Declare expert splits and scale-group boundaries in the model's projection
specification. Shared routing, projection and combine operations do not
choose a model's intermediate dtype or rounding order. MiMo rounds weighted
routes to BF16, sums in top-k order and hands BF16 partials to its hidden
all-reduce. GLM keeps routed and shared-expert partials in F32 through its
sum and rounded all-reduce. Sharing those policies would change bits.
Keep phase/bucket policy and shared-expert placement model-owned; see
[numerics](numerics.md#batch-shape) before changing their order.

### Encoded expert scale banks

A weight plan may store a routed expert bank's one-byte local scales as
`scale-row-offset-v1`: per row, a base byte and 1-, 2- or 4-bit offsets,
then the exceptions as `(index << 8) | value` records. Each bank is an
offset table followed by one encoded block per matrix, and its view's
layout is `scale-row-offset-v1.w{width}`. `kernel_core::precision::scale_offset`
holds the encoder and the CPU oracle. `kernels_sm120::moe::scale_expand`
decodes the matrices of the experts with routes into a raw bank in scratch,
one bank per entry or a layer's two banks in one launch, and the consumers
bind that raw bank unchanged. Encoding reads a matrix's scales as rows, so
a bank stored in `row-tiles-64x128-groups` keeps its encoded scales
untiled; the tiled paired entry (`PAIR_TILES_SYMBOL`) writes the first
bank's raw matrices as row tiles for consumers that read them. The launch
uses PDL and
waits for its predecessor before it reads or writes. A consumer that loads
weight scales before its own `griddepcontrol.wait` must keep binding raw
banks: in DeepSeek, that is the one-row floor kernel, which serves only the
drafter's banks.

`scale-row-tiles-offset-v1.w{width}` is a separate, lossless scale format.
It uses the same row bases and exception decisions, but groups packed
offsets and exceptions by 64-row by 4-group tiles. Strip starts are `u32`,
tile starts are strip-relative `u16`, and each exception is a `u16`
`(local << 8) | value` record. The matrix bank table remains unchanged.
`kernel_core::precision::scale_tiles` holds the codec and CPU oracle;
`kernels_sm120::moe::scale_tiles` holds the device readers. Consumers must
declare this exact format and width before binding an encoded view. The
host checks the format's own minimum size, including partial final tiles,
and preserves the format in packed transforms and weight-keeper identity.

`kernels_sm120::moe::scale_tiles_expand` expands tile-encoded banks for
consumers that still read raw scales, including native prefill. Its paired
entries retain the row-offset pair's pointer ABI and PDL ordering. A model
may omit expansion only when every scale reader in the selected projection
has a compatible in-place entry; other arguments and launch geometry stay
those of the prepared schedule. DeepSeek's one-row drafter floor continues
to bind raw banks.

### Packed FP8 fold layout

The MXFP8 fold kernels read E4M3 codes in `mxfp8-fold-packed-8x128-v1`:
eight-row tiles of 128-K folds, 32 lanes each, with every lane owning one
row and every fourth 4-byte word of it. The closed-set weight transform
`MXFP8_FOLD_PACKED` (`kernel_core::weight_transform::mxfp8_fold`) writes
that permutation at load from the checkpoint's codes, for N divisible by 8
and K divisible by 64, and its CPU oracle and a lab parity test pin it to
the stream folds' `*_prepare_v1` kernels, which built the same copy at
create before any plan stored it.

Every row-major fold reader has a packed entry: the v2, ticket, prefetch,
split decode and reuse folds, the TMA, split, compact, matrix-load and
staged-scale prefills, and the fused cluster and wide row-local status
folds. `kernels-sm120/cuda/gemm/mxfp8_fold_packed.cuh` holds the device
addressing (`Mxfp8FoldLayout::word_offset` is its host twin), and a packed
entry is its row-major entry with only the code address changed: the same
parameters, plan, shared memory and arithmetic, named by the row-major
symbol plus `_packed_v1` (`kernels_sm120::gemm::mxfp8_fold_packed`). A TMA
prefill launch also takes a 4D weight map over the packed strips. Each
family's kernel-lab `<family>_packed` cubin holds row-major twins beside
the packed entries, and `kernel_harness::packed_fold` requires every byte a
packed entry writes to equal the row-major entry's on codes whose words are
all distinct; see
[the porting notes](../crates/kernels-sm120/cuda/gemm/mxfp8_fold_packed.md).
The TMA matrix prefill and the row-local and cluster folds have no packed
entries.

`Sm120::with_packed_fold_weights` serves every fold op over packed codes:
it renames each weight-reading launch to its packed entry (moving a TMA
prefill to the packed weight map), declares the packed layout for every
weight role, refuses an entry with no packed form, and refuses the stream
folds' create-time prepare, since those folds bind the planned view as
their packed bank. `packed_op` does the same for a fold op a model
prepares itself. DeepSeek V4.1 stores every MXFP8 dense projection's codes
in this layout, compiles only packed fold readers into its fold cubins and
refuses at create a rank index that stores a dense codes view in another
layout than its ops read.

Measured on Flash SM120 at 300 W on 2026-10-23:

- Parity: every packed compute symbol DeepSeek serves (367 across TP1,
  TP2 and TP4, in 506 production-shape comparisons) wrote the same bytes
  as its row-major twin, and moving one eight-row strip failed each
  family's comparison.
- Prefill, paired lab timings of packed against row-major entries at 448
  to 8192 rows:
  - compact 0.83 to 0.92;
  - TMA 0.85 to 0.89;
  - split 0.86 to 0.97;
  - matrix-load 0.88 to 1.00;
  - staged-scale 0.98 to 1.00.
- Decode: at the decode routes DeepSeek serves, the split, ticket,
  prefetch and reuse folds measured 0.96 to 1.02. Each sample was at
  least 1 ms of cold repetitions.
- Serving, against the previous layout with its create-time packed
  copies:
  - outputs were bitwise equal in tp4-ar and tp4-dspark;
  - greedy throughput was within the A/A spread or faster: tp4-ar 0.9997,
    1.0028 and 1.0103 at C1, C8 and C32, and tp4-dspark 1.0020, 1.0067
    and 1.0039;
  - each rank had 1922 MiB more free memory at the first graph capture
    (2116 MiB with the drafter), about 35,400 more KV pages of 56,960
    bytes.

## Ownership and source changes

`kernel-build-report.json` beside the cubins records the transitive
`nvcc -M` membership (generated wrappers and headers included), source-root
relative ownership, ordered recipes and entry declarations, on fresh builds and
cache hits. `BuildReport::verify_current` checks source, compiler, lock and
artifact digests; ownership and declaration completeness are checked
separately, and reports do not grant qualification. Schema and invalidation
rules are in [`kernel-build`](../crates/kernel-build/README.md).

A change to shared row, PTX or conversion headers can affect the independent
example as well as production catalogs; `ci/cpu.sh` builds its `cuda-probe`,
verifies locks and runs selection tests. Follow the
[example's template lock procedure](../examples/external-model/README.md)
for an intentional source update. Never repair a pinned consumer by bypassing
its lock. A structural refactor must preserve every existing record; an
arithmetic change needs numerical evidence before a lock update.

## GPU access

Run GPU work only on otherwise idle GPUs. kernel-lab GPU tests run only when
`CHOPINFER_KERNEL_GPU=<ordinal>` is set. The harness refuses to run when
device memory is short and records co-resident compute processes; a benchmark
taken with another process on the device is labelled shared-device and is not
performance evidence. See [kernel-lab](../crates/kernel-lab/README.md).

## Contribution checks

- Target native `sm_120f` cubins with warp-level MMA; SM120 has no WGMMA,
  TCGEN05 or TMEM. Shared memory is bounded by 101,376 bytes per block.
- Use `shared::cta` TMA. On SM12x, cluster TMA can lower to an undefined
  `__cuda_syscall_cp_async_bulk_tensor_*` call; the cubin audit rejects it.
  Keep unproven cluster features disabled until a hardware probe passes.
- Compile keys contain static geometry, capacity, formats and target; live
  counts are runtime scalars. Capacity only sizes a row stage's grid and is
  not in its key. Layout copies also take their widths and strides at launch
  from a geometry-shared cubin, so a new geometry or rank count adds no copy
  entry; lookups, quantizers and gathers keep compiled shapes, since their
  launch-shape forms measured slower
  ([generated projections](../crates/kernels-sm120/src/precision/README.md)).
  Planners never allocate, kernels support graph capture, and qualified
  symbols are append-only.
- Pool-scaled offsets are 64-bit. The kernel-lab `high_ids` tests hold pool
  ids past the 32-bit lines; a kernel that scales a new id into a pool offset
  adds a case there and a row to `crates/kernel-lab/tests/common/pool_offsets.rs`.
- The cubin audit rejects local or stack bytes above the entry's allowance
  (zero by default); compare register counts in the lock when reviewing build
  changes. Never regenerate a lock to hide a numerical or artifact mismatch.

Grouped MoE ring entries retain compile-time shapes and measured per-shape
schedules. Replacing them with runtime dimensions requires fixed-batch
accuracy and serving performance evidence for every affected configuration.

## Untuned geometries

An admitted geometry without measured tuning tries compatible measured rows
in shape-distance order, then model or engine defaults. Each inherited row
and default must pass the complete typed engine checks, including resource
limits and reduction order. A failure refuses catalog generation; fallback
never changes operation semantics or establishes qualification. Build reports
label the source row or default for tunable entries.

The exact dense, MoE and schedule rules live with the tables:
[Qwen tuning](../models/qwen38/contract/tuning/README.md) and
[MiMo tuning](../models/mimo26/contract/tuning/README.md).

## Activation policy

The serving baseline is A16, BF16 activation operands and F32 accumulation.
`bf16` means BF16 inside kernels as well as at their input pointers. `auto`
may choose another policy only from the model's qualified inventory; a
checkpoint's quantization recipe alone cannot enable it. Forced `native`
execution remains subject to complete format, scale-order, calibration and
quality checks. Weight-only reuse and activation-quantized execution have
separate numerical contracts. A trained activation quantization point must
be preserved explicitly; omitting it is a different evaluated computation.
Payload validation rejects nonfinite or unrepresentable decoded operands.
See [numerics](numerics.md) and the typed format/precision declarations.

## Route classes and served envelopes

Every model declares the shapes it serves through the required
`conformance::Startup::envelope` method. `conformance::suite!` emits one
`route_sweep` test per profile. The CPU fake host executes those shapes
through the bundle ABI, including lowering-time choices and graph replays.
The quick tier selects an autoregressive and a speculating profile per
topology; the CPU tier includes every profile, packed prompts and chunk
continuations. CPU single-prompt probes retain every admission boundary and
its neighbours in every profile. Every secondary mode keeps fixture probes
and its smallest and largest shapes. Complete boundary triples rotate among
eligible secondary profile/mode combinations; extra dense probes rotate
among single-prompt profiles. No triple is split between combinations.

The CPU plan repeats after the printed `rotation_period`: currently 36 runs
for DeepSeek and Qwen, 30 for GLM, and 24 for MiMo. Every eligible combination
of every boundary triple is covered within that many consecutive seeds for a
fixed envelope. CI assigns consecutive seeds per model using counters under
`$CHOPINFER_DATA/route-guard/rotation`. It prints the seed, selected combinations
and a one-command reproduction. An explicit `CHOPINFER_ROUTE_SEED` reproduces
a run without advancing the counter. Direct Cargo runs default to the UTC
day, so use explicit consecutive seeds when exercising a rotation manually.

`CHOPINFER_ROUTE_SWEEP=full` checks every single-prompt prefill row count and
adds the untagged-symbol island backstop. Its secondary modes retain every
admission boundary and fixture probe in every profile, plus seeded dense
probes. The full tier does not use the CPU rotation.
`CHOPINFER_ROUTE_SWEEP=exhaustive` checks the full Cartesian set, including
all rows in every packed and chunk mode. Run the full tier before release
qualification, before landing a route change, and when adding a model or a
new served shape; the exhaustive tier is opt-in.

The shared CI Rust runner runs route sweeps once per selected model in a
separate libtest process, and excludes them from its ordinary Rust test pass.
It discovers models through their required allowlist file. Quick sweeps run
up to four models in parallel. The runner warns loudly when a model exceeds
20 seconds in quick or 60 seconds in CPU, and prints the route stage's total
added wall time. The CPU stage runs models serially. Its measured added time
on eight admitted CPUs is 184 to 188 seconds: DeepSeek 85 to 86, GLM 27 to
28, MiMo 33 and Qwen 40 seconds (2026-10-25). The model warnings remain active. `ci/quick.sh`
always selects quick; `ci/cpu.sh` selects CPU. Missing executed cases or peak
RSS reports fail the stage.

Full and exhaustive sweeps run only in bounded pieces. Build the bundle test
executable through `tools/dev/heavy.sh`, then run the coordinator outside a
heavy slot, with one exact profile test at a time:

Run the coordinator once for every emitted `route_sweep` profile when
qualifying the complete model envelope.

```sh
python3 -B tools/route_guard/sweep.py --binary "$ROUTE_TEST_BINARY" \
  --test "$ROUTE_TEST_NAME" --out "$ROUTE_RESULTS"
```

Use the exact test name from the binary's `--list` output. The coordinator
copies the executable into its private output directory, so a later Cargo
build cannot replace the code between pieces. It admits each piece separately through `heavy.sh`, runs shapes sequentially,
and releases the slot between pieces. Each piece runs in a
`systemd-run --user --scope` with `MemoryMax=24G` and `MemorySwapMax=0`.
The test enforces a 16 GiB peak RSS budget and checks elapsed time after each
shape. The wrapper terminates at 880 seconds and kills after another 10
seconds, keeping each admission under 15 minutes. Both the test and
`/usr/bin/time -v` report peak RSS, including failed runs when available.
Missing cgroup limits or RSS evidence fail the run. Increase `--parts`
(default 32) if a piece reaches its time limit. `--tier exhaustive` selects
the Cartesian sweep. Use a new external output directory for each run.

Pieces retain route counts and deduplicated metadata, never fake device
buffers or launch argument history. A piece passing is not a coverage
verdict. The final audit requires every expected shape from the same
executable and seed, then checks neighbours, islands across piece boundaries
and stale allowlist entries over the complete profile.

`kernel_core::route::RouteClass` distinguishes Fast, Baseline and Reference.
Tags are registration metadata. `Launch`, `OpInfo`, kernel locks and the
bundle ABI carry no new field. A wrapper reports its actual selected branch
through `PreparedOp::route`; lowering carries that entry in the dispatch
bindings. A symbol can be Fast at one call site and Baseline in a wrapper's
fallback branch. Those are distinct entry identities, never a global
Baseline label on the reused symbol. Conflicting classes for one entry
are refused.

Nested MXFP8 wrappers use the canonical family `mxfp8_prefill`. The registered
admitted rows are combined, and reports retain the wrapper path: compact,
matrix, TMA-matrix and TMA-scales. The selected inner fast route remains
Fast when an outer wrapper delegates to it. Keep allowlist entries specific
to the selected entry, symbol and row range.

The sweep compares each fallback entry's launch counts across shapes in the
same mode and phase. It reports a gap when nearby shapes have fewer launches
of that entry and fewer fallbacks overall, or shapes on both sides have fewer
launches of the entry. Constant fallbacks are reported. The full tier also
refuses untagged symbols present only in runs of at most three prefill
counts. A broad route with short fragments at its boundaries is not an island.

Each bundle keeps `route-allowlist.toml`, including when it is empty. An entry
names `profiles`, `phase`, `family`, `entry`, `symbol`, inclusive `rows`, a
nonempty `reason`, and a tracking `task`. Profiles and symbols may use `*`.
Use one assignment per line, double-quoted strings and JSON-compatible
arrays. Full-line comments are allowed. A full sweep refuses stale entries.
Keep independent replacements in separate entries so a qualified route can
remove its exception without changing unrelated ones.

Existing gaps may be recorded as temporary, exact-symbol/profile/range
exceptions with a follow-up task that removes them. Every sweep prints each
exception active in its sampled shapes, including its reason and task.
Unlisted gaps fail. Full sweeps enforce staleness over the complete profile;
sparse quick and CPU probes cannot establish full-envelope staleness.

```toml
[[allow]]
profiles = ["tp4-*"]
phase = "prefill"
family = "projection"
entry = "projection/general/example_projection"
symbol = "example_projection"
rows = [[258, 8191]]
reason = "The row-range replacement is awaiting qualification"
task = "projection-range"
```

Adding a model:

1. Derive `Startup::envelope` from the contract's serving constants. Include
   the maximum prefill, packing and chunk offsets. `Envelope::served` derives
   decode and verify shapes from the host's warmup policy. Size the fake KV
   pool for the longest continuation plus its final chunk, with disjoint
   page ranges for packed requests. In `symbols_only` mode, keep current
   simulated state and graph commands, but omit diagnostic upload and launch
   histories. Use `Startup::finish_route_case` to drop retired-request
   simulator state, preserving graph commands and resident block-table bytes.
   Reuse or release all per-shape tensor backing.
2. Register shared routed families through `Sm120::with_*`. Give every
   model-owned row selector an entry tag with its family, class and admitted
   rows. Forward the chosen entry through wrappers and the model's lowering
   bindings, including cached programs. A fallback reusing a fast symbol
   needs its own selected-branch identity. Additional execution exports use
   `bundle_runtime::entry::with_route_counter`, including negotiated prefill.
   Override `Startup::execute_route_step` when serving uses that extension;
   preserve non-final prompt boundaries while setting up continuation probes.
3. Run `cargo test -p bundle-MODEL --lib route_sweep --locked --offline`
   through `tools/dev/heavy.sh`. Read the profile, mode, phase, family, entry,
   symbol, row ranges, wrapper path and admitted rows. Qualify and admit a
   row range, or add a justified exception. Repeat with
   `CHOPINFER_ROUTE_SWEEP=cpu`, then the bounded full-tier coordinator above.
4. Check the existing journal pins. Route metadata must preserve their
   default bytes and every locked kernel. Metadata is not numerical or GPU
   performance evidence.
5. For served jobs, set `CHOPINFER_ROUTE_COUNTERS=1` and enable step tracing.
   The optional `step.v1` `route` object reports fallback launches, families
   and selected entries. Counts include graph replays and add no device
   synchronization or readback. The tune runner enables these counters and
   rejects missing evidence or a nonzero family without a reason in the
   job's `allow_fallbacks` object.

For profile review, set `CHOPINFER_ROUTE_TAGS_OUT` to a fresh existing directory
outside the repository when running a sweep. It writes one tag JSON per
bundle/profile. Use tags from the profiled source revision:

```sh
python3 -B tools/route_guard/check.py --classes "$ROUTE_CLASSES" \
  --nsys "$PROFILE_SQLITE" --steps "$STEP_LOG" --allow "$JOB_ALLOWLIST"
```

The checker also accepts `--journal`. Symbol-only evidence fails for an
unallowed Baseline-only or Reference-only symbol. A symbol shared by Fast
and fallback entries is ambiguous: provide selected-entry step records to
gate its exact fallback counts. Its device time remains unattributed.
Exit codes are 0 for covered evidence, 1 for unallowed fallbacks and 2 for
invalid, missing or ambiguous evidence. A model's range allowlist also
needs `--profile`, `--phase` and `--rows` where those constraints apply;
a job's family list instead supplies explicit reasons for that workload.
The serving-step profile report accepts `--route-classes` and `--route-allow`
to run this check after reading the export.
