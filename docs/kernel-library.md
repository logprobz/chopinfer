# Kernel library

Chopinfer's kernels live in one base per NVIDIA ISA family. The first family
is SM120, built as `sm_120f` cubins. The kernels are tuned for and tested
only on the RTX PRO 6000 Blackwell (96 GB, 188 SMs: `SM_COUNT` in
`kernels-sm120`). The runtime refuses a device that is not compute
capability 12.0 exactly (`runtime-cuda`), so SM121 parts such as DGX Spark
are refused; other compute capability 12.0 parts, such as GeForce RTX 50
series cards, are untested, and packages plan 94 GiB of device memory per
rank, which smaller cards do not have ([hardware](performance.md#hardware)).
The numerical gates are in [numerics](numerics.md); for independent model workspaces, see
[external models](external-models.md).

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

## Engines

The serving standard is A16: every projection runs BF16 activation operands
on the A16 engines. FP8- and FP4-activation MMA rows are optional performance
variants, enabled by a model only after its own accuracy gate passes.
An engine's supported format list alone does not qualify that variant.
The model's supplied inventory exposes only variants that passed its gate.
Activation policy `auto` can select non-BF16 operands only from that gated
inventory; the checkpoint's declared recipe alone cannot enable them.

Checkpoints trained with activation quantization must retain their trained
quantization points. The planned A16 path applies an exact rounding stage at
each such point, placing BF16 activations on the trained quantization grid
before the projection. This stage is not implemented; it is planned with the
first model that needs it. Weight reuse without that stage has its own
numerical contract and qualification.

- GEMM: dense and grouped (MoE) forms, decode swap-AB and prefill paths.
  Activations BF16, FP8 in 1 x 128 groups, MXFP8 or NVFP4. Weights BF16, FP8
  (per-tensor, per-channel or 128 x 128 block), MXFP8, MXFP4 or NVFP4.
  Epilogues: residual add, SwiGLU, output quantization, weighted MoE combine.
- Attention: paged KV in BF16 or FP8; GQA, MQA or MHA; head dims 64, 128,
  192 and 256 including a qk dim that differs from the v dim; causal, sliding
  window, sinks and softcap; prefill (variable length, chunked over a prefix)
  and decode (split-KV plus merge).
- Row: RMSNorm variants, RoPE fused with the KV write, activation quantizers,
  routers, expert permute and combine, embedding gather, logits reductions and
  sampling.
- Comm: PCIe P2P one-shot and two-shot all-reduce with fused residual add and
  RMSNorm, plus all-gather.
- Linear attention: depthwise causal convolution with slot state, the gated
  delta rule (chunked prefill scan, decode, and verify with a state checkpoint
  per row) over F32 state pools addressed by slot id, and the per-head gated
  RMSNorm. See `crates/kernels-sm120/cuda/linear/README.md`.

Direct dense attention accepts per-rank GQA groups 1 through 8 or 16.
Prefill keeps groups that divide 16 and pads groups 3, 5, 6 and 7 to 8;
groups 9 through 15 are not padded and fail its m16 tile assertion. At head
dim 256 with FP8 KV, groups 2 and 4 also fail the zero-stack audit (16 bytes
of stack and 32 bytes of spill stores with CUDA 13.2.86). The admissible
groups there are 1, 3, 5, 6, 7, 8 and 16. BF16 at head 256 and both KV
formats at head 128 pass that audit for groups 1 through 8 and 16.

QSA uses a separate attention kernel that pads every group up to 16 to
whole n8 tiles. Qwen's served TP2 group 12 pads to 16 and TP4 group 6 pads
to 8. Its catalog adds direct dense entries only at TP4. Admission checks
each served topology with the padding of the entries it selects. Paged
decode at head dim 128 refuses group 16: its fold needs 8,448 bytes in an
8,192-byte K tile. Both KV formats therefore require groups 1 through 8
for a complete direct dense head-128 profile.

The Qwen engine domain admits indexer heads 4, 8 or 16 at dimension 128,
attention heads of dimension 128 or 256, and rotary widths 32 or 64 for
both the indexer and attention. M-RoPE sections are [6,5,5] at rotary 32 and [11,11,10] at rotary
64; other triples instantiate unqualified code and are refused. The bundle builds its table from that width and the geometry's theta;
theta must remain finite and positive in F32, with no pinned-value list.
`HfF32AngleBf16CosSin` pins Torch 2.14 inverse-frequency bits only for theta
1e4 and 1e7 at rotary 32 or 64. Other thetas, including synthetic-wide's
5e6, use Rust F32 `powf` and reciprocal. They have no Torch bit-parity
guarantee and require reference-table evidence before checkpoint qualification.
Gated rows require `ceil(local Q heads / 8) + local KV heads <= 8`.
The bundle requires one KV head per rank at both TP2 and TP4, using split
or replicated checkpoint heads, and checks every topology's group limits.
Direct verify requires an even local group: its eight query rows must
fill m16 tiles. With both topologies, that admits 8, 16, 24 or 32 total Q
heads at dimension 128; dimension 256 admits 24. TP4's head-256 FP8
prefill spills at groups 2 and 4; group 8 has no qualified dense
prefill/verify instance and is refused. Checkpoint KV heads may be 1 or 2.
QSA block size is 4 and compiled kept-block counts are 256 or 512.
The compiler intersects structural limits with
`kernels_sm120::attention::qualification`; a lab test checks every table
tuple against the compiled rows, dense, fold, sparse and indexer catalogs.

Linear key and value dimensions remain equal to 128, with an integral
value/key head ratio from 1 through 3. The delta rule instantiates one
key/value dimension and the norm requires 128. At key dimension 256, the prefill scan
would use 157,480 shared bytes and prepare at least 104,224 bytes, both
above SM120's 101,376-byte limit. See the
[linear engine](../crates/kernels-sm120/cuda/linear/README.md).

These limits describe engine compatibility. `synthetic-wide` is an opt-in
CPU catalog, build and serving fixture. It does not qualify a checkpoint.

GPU qualification is instance-specific. The existing lab evidence covers
head-128 dense prefill/decode/verify at group 6; gated rows and folds at
(head, rotary) (128,32), (128,64), (256,32), each with six local Q heads;
and head-128 QSA groups 6 and 16 with 512 kept blocks. Indexer evidence
covers heads 4/8/16 crossed with rotary 32/64, at dimension 128, block 4,
ring 16 and 512 kept blocks. The production head-256, rotary-64 path has
rows at Q6/Q12, dense/fold group 6 and QSA groups 6/12.

The following additional instances passed GPU oracle and invariance jobs
on 2026-10-04, with all 1,057 required coverage items reconciled.
All entries below use one KV head, BF16 and FP8 KV, block 4 and page 64.

| Engine | GPU-qualified additional instances |
| --- | --- |
| Dense head 128 | Prefill and decode groups 1..8; verify groups 2/4/6/8 |
| Gated rows, head 128 | Rotary 32, local Q1..16; rotary 64, Q12/Q16 |
| Gated fold, head 128 | Rotary 32, groups 1..8 |
| Gated rows, head 256 | Rotary 32, Q1/2/3/5/7/8/10/12/14/16 |
| Gated fold, head 256 | Rotary 32, groups 1/3/5/7/8 |
| QSA head 128 | Groups 6/12, 256 kept blocks, rotary 32, theta 5e6 |
| Indexer dimension 128 | Eight heads, rotary 32, 256 kept blocks, ring 16, theta 5e6 |

Head-128 rotary-32 rows and folds also use theta 5e6 and sections [6,5,5];
other new rows use theta 1e7, with [6,5,5] at rotary 32 and [11,11,10] at
rotary 64. The sparse pair matches synthetic-wide at TP4 and TP2, including
its 1024-token indexer budget. Fold tests require bitwise equality with
rows plus ordinary decode and check both paths against their oracles.
All existing declared batch/chunk/rows/anchors/ties claims remain required.
The following closure instances passed GPU oracle and declared invariance
jobs on 2026-10-04. Reconciliation qualified all 3,067 required W7D coverage
items: rows and folds, 14 sparse geometries in both KV formats across all
phases and boundary contexts, and five kept-256 indexers, each with 33 matrix
and 253 adversarial cases. This is kernel instance qualification, not model
quality or serving performance evidence.

| Engine | GPU-qualified closure instances |
| --- | --- |
| Gated rows | Head 128 rotary 64 Q2/4/8; head 256 rotary 64 Q8/16 |
| Gated folds | Head 128 rotary 64 groups 2/4/8; head 256 rotary 64 group 8 |
| QSA head 128 | Groups 2/4/8/16 at kept 256; groups 2/4/8/12 at kept 512 |
| QSA head 256 | Groups 6/8/12/16 at kept 256; groups 8/16 at kept 512 |
| Indexer dimension 128 | Kept 256: heads 4 rotary 32/64, heads 8 rotary 64, heads 16 rotary 32/64 |

These use theta 1e7, the section triples above, block 4, ring 16 and page
64. Sparse attention and indexer coverage factor independently because
attention does not consume indexer heads or rotary parameters. All lab
sparse declarations now require four exact verify rows, matching the
bundle. The top-k ties fixture retains its original 300-block row at kept
512 and uses 255 blocks at kept 256. Future table additions require matching
lab instances and GPU evidence in the same change, or proven SASS identity
to covered instances. RoPE tables additionally need reference parity
before checkpoint qualification.

## How a bundle consumes kernels

1. The bundle's `build.rs` lists the instantiations it needs from a
   model-owned selection, shared typed builders, offers generated from its
   projection declaration or legacy `kernels-sm120` catalogs.
2. `kernel-build` generates one `extern "C"` entry translation unit per
   cubin, compiles it with `nvcc -arch=sm_120f -cubin`, audits the result and
   computes the source-closure digest.
3. The build compares every cubin identity with the bundle's lock (see
   [Kernel locks](#kernel-locks)). A mismatch fails the build; investigate it
   before any deliberate lock update. External SDK builds reject update and
   profiling overrides.
4. The bundle prepares each op once, at instance creation, through
   `kernel_core::backend` or a typed planner. Per step it only plans launches
   (grid, block, shared memory, workspace, cluster, PDL) and packs arguments
   using only ABI argument kinds.

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

A backend is a crate that implements `kernel_core::backend::Backend` for
its kernels. `kernels-sm120` (`Sm120`) is the production backend and today
the only one; an op family a backend does not implement keeps the trait's
default refusal. Read the module docs of `kernel_core::backend` and
`kernel_core::properties` first; the method rustdoc is the reference, and
this section only maps it.

### What a model hands a backend

A model reaches a backend only through `&dyn Backend` and its own
`&dyn OwnOps`, which it gets from its target module: today the `sm120`
module of each model crate, which registers that model's compiled entries,
catalogs and tuning on `Sm120` and implements `OwnOps` over its own cubins.
The rest of the model states math and names no SM120 item
([the seam](adding-a-model.md#the-seam-definition-of-done)). A second
backend adds its own target module beside `sm120` in each model it serves,
with the same two values; no generic model code changes.

### What a backend implements

- One method per op, grouped into one section per contract family in the
  `Backend` trait (`crates/kernel-core/src/backend.rs`). Implement the
  families your kernels cover; the defaults refuse the rest. A model
  prepares its ops when an instance is created, so a refused op fails
  creation, never a step.
- Prepare once, launch per step. A method takes the contract op and a
  `Need` and does all selection, tuning lookup and validation. It returns a
  `Prepared<B>`, a boxed `PreparedOp` over the family's role record `B` of
  `View`s. Per step the host calls only `launches(buffers, workspace,
  counts)`. It checks the views and counts against the prepared op and
  returns `Launch`es: symbol, recorded rows, `LaunchPlan`, arguments in
  parameter order, and whether the launch is `collective` (waits for peer
  ranks) or `persistent` (occupies every SM). No search, catalog lookup or
  environment read happens per step, and buffers arrive by role, never by
  kernel parameter position.
- `Need`. `need.capacity` is the most rows one launch processes;
  `need.exact` is the exactness the model relies on, as an `Invariance`:
  verify rows equal to decode rows (`Rows::OneRow`), chunk splits, batch
  independence, tie and anchor rules. `OpInfo::guarantees` must hold for
  every launch and must not claim more than the kernels keep; the kit
  checks claims on a GPU. Call `Need::check` before returning an op. An op
  that sends each row count to its own entry composes its claim with
  `Invariance::routed`.
- `Refused`. `Unsupported` for a family the backend does not implement,
  `Inexact { needed, guaranteed }` when it cannot keep `need.exact`, and
  `Rule(reason)` for a contract rule or a backend limit (a shape outside the
  compiled set, a plan past the shared-memory bound). Refuse at prepare;
  `launches` refuses only buffers and counts that do not fit the prepared
  op.
- `OpInfo`, what the backend chose:
  - `implementation`: a versioned name of the numerics choices. Ops with
    equal implementations and contracts compute equal bits.
  - `guarantees`: the exactness every launch keeps.
  - `bytes`: the least bytes of each role the op sizes, by role path
    (`scratch.partial`). The host checks each listed view before binding it.
  - `workspace`: backend scratch beyond the roles, passed to `launches` as
    one view aligned to at least `WORKSPACE_ALIGNMENT` (256) bytes and named
    by the reserved role path `backend_workspace` (`WORKSPACE_ROLE`). It is
    uninitialized before the first launch and not preserved between steps.
  - `state`: persistent regions a collective keeps on every rank, which the
    host allocates and zeroes once.
  - `weight_layouts`: the layout of each weight role whose layout the
    backend chooses. The model's weight preparation writes that layout.
  - `symbols`: every symbol the op may launch, with the dynamic shared
    memory the host configures it for at load.
- Projections also return the implementation identity the model's precision
  document authenticates (`precision::execution::Binding`).

### Model-owned ops

Kernels that only one model runs (ring backups, history copies, drafter
rows) have their contracts and CPU oracles in that model's contract crate,
behind its `OwnOps` trait: the same shape as `Backend`, one default-refusing
method per op. The SM120 implementations live in the model's bundle `sm120`
module. MiMo's are the example: `models/mimo26/contract/src/own_ops/` and
`models/mimo26/bundle/src/sm120/own_ops.rs`. A new backend implements the
methods it has kernels for; a model whose required op is refused does not
create an instance on that backend.

### Tuning

Tuning is the backend's and is read when an op is prepared: K splits,
prefill segments, schedules, ring depths, tile shapes. The backend reports
the result through `OpInfo::implementation`. `Sm120` takes its tuned choices
from the entries a model registers from its SM120 selection (its catalogs
and `contract/tuning/` tables). A new backend keeps its own tables and
applies them in its prepare methods, never per step.

### Host kernels a backend must also supply

The host launches one kernel set itself, not through `Backend`: the device
sampler, an SM120 cubin that `runtime-kernels` builds and the host loads by
symbol, so another target's version also needs a host change. Its contract is
`kernel_core::sampling`: the order of operations (bias, penalties,
temperature, min-p, top-k, top-p), constraint and EOS masks, Philox and Gumbel
sampling, tie rules, logprob and top logprob semantics, the row records and
the `SamplingBuffers` roles. The oracle is the CPU sampler,
`runtime_sampling::oracle::sample_row`, and `kernel_core::sampling::check`
judges a device row against it. SM120's entries are `kernels_sm120::sampling`,
launched by `crates/runtime-host/src/sampling.rs`; their GPU tests are in
`crates/kernel-lab/tests/ops/sampling.rs`.

### Host admission

What the host checks before any kernel runs:

- The device. `runtime_cuda::Driver::retain_primary_context` admits only a
  device of compute capability 12.0 with memory pools and refuses others
  (`CudaError::UnsupportedDevice`, `MemoryPoolsUnsupported`). A backend for another
  architecture therefore needs this admission widened in `runtime-cuda`,
  which is outside the backend's crate.
- The bundle. Before it opens the native module, the host resolves every
  operation the manifest's graph buckets reference for its selected
  architecture (`--architecture`, default `sm_120f`) against its own
  catalog; implementations declared for other architectures are never
  candidates (`crates/runtime-host/src/install.rs`). It then checks the
  module's length, digest and entry symbol. Cubins the bundle loads later
  go to the driver through the host's module service, which loads any
  non-empty module the driver accepts.
- The build. `kernel-build` compiles and audits `sm_120f` cubins. A
  different target needs its own build and audit.

### Running the conformance kit

`kernel_harness::kit::conform(backend, modules, session, ops)` (re-exported
as `kernel_lab::kit`) runs a list of
ops against any backend. Each `Required` names an op, its `Need`, how a
backend prepares it (the contract op through the `Backend` method) and its
case. A refusal is only a result unless the op is marked `.required()`, so
mark every op your backend must serve, and also assert that the ops you
expect were prepared (`Report::prepared`; `must_prepare` in the test below
is the example): a backend that refuses everything otherwise reports no
failures. Implement `kit::Modules` to load your backend's symbols, as
kernel-lab's `LabModules` does for SM120's lab cubins. The case families'
SM120 selections take the catalogs (`&Catalogs`) or entries they select
from, and each family's `checks` module holds its case self-checks over
them. `crates/kernel-lab/tests/kit.rs` runs SM120 over one set from the
in-tree catalogs:

```
cargo test -p kernel-lab --locked --offline --test kit
CHOPINFER_KERNEL_GPU=<ordinal> CHOPINFER_KERNEL_REQUIRE=1 \
  cargo test -p kernel-lab --release --locked --offline --test kit _on_a_gpu \
  -- --test-threads=1 --nocapture
```

Without a session the kit prepares every op, runs the CPU checks and
reports the rest as not exercised. For each op the report says prepared or
refused (with the reason) and the result of each check:

- `declaration` (CPU): the case declares an oracle and validates, and the
  launches match `OpInfo`: every symbol is listed, within its listed shared
  memory, and every role the info sizes is bound with at least those bytes.
- `need` (CPU): the op's guarantees cover the required `Need`.
- `conformance` (GPU): the checks of
  [declarative conformance runs](#declarative-conformance-runs) over the
  op's launches, judged against the contract's oracle.
- One property check per property the op claims or the need requests
  (rows, batch, chunk), or the reason the kit cannot check it.

Verify rows equal to decode rows (`Rows::OneRow`) is a property like the
others. For a row op the rows check runs the op at one row. For an op over
items of several rows, a verify step, the one-row entry is a different op,
the decode op of the same query: such a case declares `item_rows` and gives
a `kit::Decoded` mapping from the verify step's items to one-row launches of
the decode op (`kit/decode.rs`). The check then runs the verify step over
every item of up to the claimed rows and each mapped decode launch alone,
and requires the bytes the mapping names to be equal. A case of such items
that claims or needs one-row bits without a mapping does not declare, so
the check never passes vacuously. The attention and recurrent cases
(`kit/cases/attention.rs`, `kit/cases/recurrent.rs`) are the examples.

Collectives run on one GPU per rank through the multi-rank runner,
`kit::collective::conform(backend, modules, ranks, sessions, ops)`. Each
rank's role record names its own buffers, persistent state regions
(`OpInfo::state`) and peers' buffers by path. The runner prepares the op on
every rank, launches every rank's `k`th launch before any rank's next, and
checks the declaration per rank (and that every rank launches the same
symbols, plans and collective flags), the need, two steps in a row at each
live count (including both sides of the comm protocols' row thresholds)
against the oracle with equal bits on every rank, and the claimed
properties over all ranks' rows. `crates/kernel-lab/tests/kit_collectives.rs`
prepares the collectives at two and four ranks on the CPU and runs them at
the rank count `CHOPINFER_KERNEL_GPUS` names:

```
cargo test -p kernel-lab --locked --offline --test kit_collectives
CHOPINFER_KERNEL_GPUS=<ordinals> CHOPINFER_KERNEL_REQUIRE=1 \
  cargo test -p kernel-lab --release --locked --offline --test kit_collectives _on_a_gpu \
  -- --test-threads=1 --nocapture
```

The runner does not exercise PDL order, graph replay, the null launch or a
device live count below the rows (the collectives read `Counts::rows`
alone).

`Report::failures` lists failed checks, undeclared cases and refused
`.required()` ops, and `Report::not_exercised` what did not run; neither
counts as a pass. A family without a case generator in
`kit/cases.rs` cannot run yet: add one, with the family's roles and oracle,
before claiming conformance for it. A passing run shows that the prepared
ops meet their contracts and claims on the tested shapes; it does not
qualify a model, a topology or performance.

### Cleanup candidates when a second backend arrives

These neutral interface items exist to keep SM120's launches byte-identical
through the migration. They stay until a second backend shows what it
needs; revisit each then:

- `AttentionOp::plan_at_capacity`: an op plans the same grid whatever items
  a step serves.
- `Counts::padded_rows`: row ops grid a step's padded bucket while their
  arrays are checked at the live rows.
- `RmsNormReduction::WarpTree { threads }`: a reduction order expressed in
  SM120 thread terms.
- `Launch::persistent`: a launch that occupies every SM, which a bundle's
  scheduler uses for overlap decisions (it reserves SMs for communication).
- Dynamic shared memory in `OpInfo` (`symbols`): an SM120 launch fact
  surfaced through the neutral info.

## Declarative conformance runs

`kernel_harness::conformance` runs one entry, or a chain of entries, through
the checks every engine owes, so a new engine gets them by declaring its
buffers. The entry
implements `kernel_harness::lab_entry::LabEntry`: its symbol, parameters,
`plan(prepared_rows)` and, when its ABI differs from the parameter kinds,
`pack_arguments`. kernel-lab implements it for the shared engines whose
planners take prepared rows; a test wraps a bundle's own kernel in a
test-local type.

A test then declares a `Case::new(&entry, capacity)`. Buffers bind to the
entry's parameters by name; each has a dtype and dims (`Rows::Live`, the
capacity, or `Rows::Fixed(n)` rows, times a row width) and one role:

| Builder | Role |
|---|---|
| `input(name, dtype, dims, image)` | Read-only input. `produced(name)` marks the one a PDL producer rewrites. |
| `ids(name, dims, ids, pools, holes)` | I32 IDs (-1 for a hole) into rows of `(pool, stride bytes)`. `Holes::LastLive`, or `Holes::Row(n)` for fixed inputs such as page tables, names a row the entry reads, rerun with every ID -1. |
| `output(name, dtype, dims, write)` | Poisoned output that must write `Write::LiveRows`, `Write::All` (every element at a positive count, for fixed rows) or `Write::Map(rule)`, a `WriteMap::of(dtype)` built from the images and the live count. Whatever the rule, a map writes nothing at live 0, and a `Rows::Live` buffer's map and updates stay below the live rows. `output_with` takes another fill. |
| `scratch(name, bytes, may_write)` | Poisoned scratch that may write only the given byte ranges. |
| `state(name, dtype, dims, initial, write_only, updated)` | In-place state: `write_only` is poisoned and must be written; `updated` may be written but need not be; bytes outside both keep `initial`. |

A chain, such as split partials and their merge, adds entries with
`Case::new(&first, capacity).then(&second)`; `lab_entry::SplitMerge(&entry)`
is the merge kernel of a latent attention or pooled indexer selection
entry, which plans only a bucket the entry itself plans (so a merge
launched alone keeps the entry's contract and capacity checks) and has a
merge launch. Every entry plans at the capacity and binds the buffers and
scalars its parameters name, and every buffer and scalar must bind to some
entry.

The link rule is metadata validation of the entries' declared parameters,
not instrumentation of what the kernels read or write. A pointer parameter
whose C++ type is not `const` counts as a write and must name an output,
scratch or state. Each later entry must have a `const` pointer parameter
naming at least one buffer an earlier entry writes. The rule does not check
the producer of every read: an entry with one such link that also reads,
say, scratch no entry writes still validates. Such an intermediate is
declared once, as scratch (poisoned before each launch, bounded by
`may_write`, contents unchecked) or as an output the oracle covers (checked
like any other output). The phases treat the chain as one launch: they
prepare the buffers before the first entry and check them after the last,
so a check sees what the chain leaves, not which entry wrote it, and a
write out of bounds that a later entry overwrites goes unseen. Per entry the
run checks only that it plans, binds, launches, and that the chain without
it fails (the null launch drops each entry in turn, eagerly).

Eager and relaunch phases launch the entries in stream order without PDL.
The graph and PDL phases launch each entry with PDL, when its plan allows,
behind the one before it, the first behind the producer. No check shows
that a later entry waits for its predecessor before it reads an
intermediate: an early read (one placed before the entry's PDL wait) is
caught only when it happens before the predecessor writes that location and
changes a checked result, and no case delays an intermediate producer to
make that overlap certain. A passing chain therefore does not show that its
waits are placed correctly. The runner adds no synchronization between
entries, since that would hide the ordering under test.

`Case::refusing_runs(mutant)` lists the runs (each live count, and the
high-id and hole reruns) whose exact checks refuse a host model of a wrong
launch, comparing the outputs it gives with the oracle's bits inside their
write maps and the in-place states it gives whole, with the oracle's bits
or, where a state writes nothing, its initial bytes. A CPU test asserts
that a mutant the declaration cannot rule out is refused, so a fixture
that stops telling it apart fails before a GPU run.

`scalar(name, value)` binds a scalar parameter and `live(name)` the device
live count (default `live_rows`). `live_counts(list)` replaces
`poison::live_counts(capacity)`; a list keeps 0, a partial count and the
capacity. Every output and state must write at every positive live count;
`no_write(name, live, reason)` declares a count at which one writes nothing,
such as a superseded row. `oracle(|images, live| ...)` returns, for every
output and state that writes at that count, `Expected::Bits` of the whole
buffer or `Expected::Values` with one value per element.

`Case::validate()` checks the declaration on the CPU: bindings both ways,
image sizes, IDs within their pools, map dtypes and bounds, the write and
no-write declarations, the oracle's coverage, and that each rerun can fail
a launch that ignores what it changes. It judges that on exact results
alone: for an output whose oracle result is `Expected::Bits`, its fill with
those bits in its write map; for a state, its bits; for a buffer that writes
nothing, its fill or initial bytes. Update ranges only permit writes and a
numeric reference (`Expected::Values`) can share accepted values with
another, so neither counts. At the largest live count, the high-ID rerun's
results, with the pools moved back, must match the first run's, and one
output or state exact in the first run, the rerun and the old IDs read over
the moved pools must be the same in the first two and differ in the last;
the hole row must change the results. A high-ID or hole claim therefore
needs exact bits, in every run, for an output or state the IDs select, IDs
in the rows the entry reads, and moved pool rows that differ from the rows
they displace; a case that cannot give them declares the IDs without pools
or `Holes::Never`, and `run` reports that rerun as not exercised. Call
`validate` from a CPU test for every configuration.
`Case::run(session, &module, Some(&lab))`, where `lab` is the latent lab
module holding the PDL copy producer, calls it, opts each entry's function
in to its plan's dynamic shared memory above 48 KiB, and then checks, with
one graph captured at the capacity and replayed at every live count:

- eager, relaunch, graph and PDL launches on poisoned buffers match bit for
  bit; outputs and in-place states keep their poison or initial bytes at
  live 0, and past the live count of a `Rows::Live` buffer; scratch stays
  within its `may_write` ranges at every count, live 0 included;
- the eager bytes match the oracle;
- PDL behind a producer that holds the device live count at zero until its
  tail and, for a produced input, behind the copy producer and then with that
  input held as well. These phases probe for reads before the PDL wait; they
  are not guaranteed detection. A misplaced read fails only when it lands
  while the producer holds the count or input. CUDA does not promise that
  the launch overlaps the producer, so a launch that starts after the
  producer's tail, or reaches the read after the producer restores the count
  and input, passes. The GPU negative controls
  (`latent_split_merge_conformance.rs`, lock family `pdl_misorder`) run the
  latent decode split merge chain with the split's, then the merge's,
  dependency wait dropped: on 2026-10-12 each failed 10 of 10 runs, every
  time in the graph phase, the first that launches a stage behind its
  predecessor, while eager and relaunch passed. A run stops at its first
  failing phase, so these mutants never reached the PDL phases: the controls
  show that the runner catches the misordered chains, not that the PDL
  producers do;
- a launch that writes nothing fails at every count where something writes;
- ID inputs with pools rerun with their IDs and pool rows moved to the top
  rows and match bit for bit, the pools moved back; the hole row reruns
  against the oracle.

`run` returns the evidence as JSON for a receipt. The `conformance::tests`
module declares a small gather case. The page-indexed engines' cases
(`tests/*_conformance.rs` in kernel-lab) get exact attention outputs from
one-hot keys (`tests/common/hot_keys.rs`): integer queries and keys whose
scores differ by more than any exp survives, so the best key's value row
is the numerator bit for bit. The oracle refuses keys outside that domain:
non-finite elements, dots F32 cannot sum exactly and finitely in every
order, F32 scores closer than the margin, and scores so far apart that the
F32 exponent argument overflows. Each engine also has a wide case over a
pool of 65636 pages, whose high-id rerun moves its pages to IDs past 16
bits; its oracle shows that reading the IDs narrowed to 16 bits changes an
exact output. Engines that fold splits with a second kernel run both ways:
with one split and no merge (latent prefill, a pooled bucket of one
split), and as a chain over many splits (`latent_split_merge_conformance.rs`,
the 64 fixed splits of every decode and verify row, at 4 rows of one split
per CTA and at 24 rows of four consecutive splits per CTA; and
`pooled_indexer_merge_conformance.rs`, 64 splits of which three score),
whose oracles give the split partials and the merged outputs exact bits.
The latent chains run over the one-hot keys and over twin keys whose best
keys tie across two splits, so the merge sums two contributions of weight
1; the pooled chain's selected pools come from two or three splits. Their
witness tests require a merge that copies the greatest-max split, or folds
only split 0's list, to fail.

FP8 E4M3 KV entries share the page addressing with 1-byte elements, so the
paged, selected and latent cases have FP8 forms (`*_fp8_conformance.rs`)
over the same rows, tables and strides; their wide reruns pass byte offsets
of 2^31 (2^32 for latent pages). A host model of a launch that computes
page byte offsets in unsigned 32 bits fails the latent wide rerun on the
CPU; the paged and selected offsets lie below 2^32, where only signed
32-bit offsets go wrong, as addresses outside the pool. E4M3 holds integers exactly only up to 16,
so their one-hot keys (`tests/common/hot_fp8.rs`) spell a code in three
digits of 16 under query weights 2^20, 2^16 and 2^12, at a K scale of 2
(scores at least 512 apart) and a V scale of 1/4. Latent FP8 pages carry
per-token block scales, powers of two that vary by token past the block a
query reads, so the outputs show that a launch reads each key's own
scales. The oracle refuses non-finite codes, inexact dots or scaled block
sums, latent block scales that are not powers of two, and scores within
the margin (`tests/hot_fp8_refusal.rs`). The pooled indexer has no FP8 KV
entry, and the partial merge reads no KV.

The RoPE and KV write family (`kit/cases/rope_kv.rs`, run on a GPU by
`sm120_passes_the_rope_kv_family_on_a_gpu` in `tests/kit.rs`) covers every
catalog `rope_kv`, `dflash_rope` and `ring_write` entry. Its ring planes,
of the ring write and of a storing ring RoPE KV write, are those
`oracle::ring_kv_write` leaves: exact bits, except a fused write's rotated
K, which keeps the rotation's bound. Each computed-angle entry also runs an
angle op (`... angles`) whose rotated pairs are (1, 0). The rotation turns
each pair into the device's cosine and sine exactly, but the output is
BF16, so the outputs hold the BF16-rounded device cosines and sines;
`rotary::check_computed`, whose allowance includes that rounding, judges
each, the RoPE alone at positions up to 2^20 - 1
([numerics](numerics.md#text-rope-arithmetic-and-qualification)).

Two more KV writers have cases of their own: the latent KV write
(`latent_kv_write_conformance.rs`) and the gated norm, RoPE and KV write,
served and owner-routed (`gated_kv_write_conformance.rs`), each in BF16
and FP8 E4M3. Their caches are in-place state (`tests/common/kv_pool.rs`):
pages of old contents and decoy bytes, of which a launch may change only
the slots of its live rows, so untouched pages, decoys and the slots of
rows past the live count keep their bytes. Their high-id reruns move the
page tables' pages, and their hole reruns set a position to -1 where the
writer skips one (a latent row's position, an owner-routed row's local
position). Each writer also has wide cases over 65636 pages, run at live
0, half and the capacity. They cover both gated address branches (heads
128 and 256), and their byte offsets pass 2^31 but stay below 2^32, so
they do not tell an unsigned 32-bit offset from a 64-bit one. A CPU test
requires a writer that narrows page IDs to 16 bits to fail the state
checks of every wide case's high-id rerun. The fixtures keep every step exact: FP8 codes are E4M3
values times a power-of-two scale, a latent block's largest magnitude is
448 times its scale, and the gated norm sees elements of +-8 (mean square
64, which absorbs the epsilon) and a RoPE table of 0 and +-1, so the
oracle's emulation of the kernel's F32 steps gives its bits. The oracle's
checks are narrower than that domain: they refuse a sum of squares whose
terms are not integers or that reaches 2^24 (the bounds within which every
reduction order is exact), an epsilon that changes the mean square, a
latent scale that is not a power of two, and any value BF16 or E4M3 does
not hold exactly.
GPU runs use the kernel-lab harness:

```
CHOPINFER_KERNEL_GPU=<ordinal> CHOPINFER_KERNEL_REQUIRE=1 \
  cargo test -p kernel-lab --release --locked --offline --test <file> <test>_gpu \
  -- --test-threads=1 --nocapture
```

## Model-owned selections

Keep four declarations separate:

| Declaration | Owner and meaning |
|---|---|
| Mathematical requirement | Model requests formats, layouts, scaling, epilogues, reduction/rounding and state behavior. `kernel-core` holds reusable typed contracts. |
| Selection | Model's list of shared and native instantiations, explicit symbols, capacity and source ownership. |
| Implementation | Target-specific builder or local CUDA, with hardware/resource requirements and a versioned identity. |
| Qualification profile | Evidence for exact artifacts, topology and serving limits. Building a selection does not qualify it. |

`kernel_core::rmsnorm::RmsNormRequirement` and
`kernels_sm120::row::rmsnorm::RmsNorm` provide a profile-free row builder.
It supports the existing unquantized BF16 Plain, Gemma, per-head QK and ordered
HF variants. It validates contiguous aligned rows, epsilon, residual behavior,
accumulation/rounding, capacity and SM120f bounds. Its argument packer takes epsilon
from that requirement. `require` compares the complete contract: an HF request
cannot bind to a plain kernel at the same width. Unsupported formats, layouts or
rounding rules return errors before code generation.

Compose `SelectedCubin` values into `KernelSelection` with a `BuildContext`.
Shared builders derive `EntryDeclaration` from validated requirements. A native
entry supplies its own declaration, CUDA and model-owned correctness cases. The
driver checks declaration coverage and recipe validity, not arbitrary CUDA's
mathematical truth. New cubins use versioned names; `SelectedCubin::existing`
attaches declarations to a locked legacy name without changing its wrapper.
Entry, parameter, include and define order remains author-controlled.

Legacy catalog cubins reach a selection as `arithmetic::DeclaredCubin`. Its
producer states, from the instantiation it built each entry from, the arithmetic
that launch arguments or `moe::Variant` fields decide: dense K splits, prefill K
segments, QSA attention splits, and a grouped entry's route weight point, clamps
and schedule. `DeclaredCubin::select` adds it to the entry's declaration, so the
contract identity changes with it even when the cubin does not. An entry that
takes `splits` at launch cannot be declared as compiling its arithmetic in.

See the external example's `sm120` module
(`examples/external-model/bundle/src/sm120/`): `selection.rs` selects shared
1280-wide RMSNorm and its local sigmoid probe without a new `Profile` or any
production program edits, and `precision_selection.rs` generates the
precision exercise's offers from the example's projection declaration and
keeps its locked names through `bundle/projection-adoptions.json`. The
optional `cuda-probe` and `precision-exercise` features compile them ahead
of time. The ops themselves are kernel-core contracts and the example's own
`OwnOps` probe (`contract/src/ops.rs`), prepared through `&dyn Backend`;
their cases live in `bundle/kernel_cases.rs` and the SM120 build checks in
`bundle/tests/sm120/`. The example's own test crate (`lab`) loads the
exported report and cubins and launches the prepared ops on the kernel
harness. The example remains
mechanics-only.

`gemm::DenseRequest`, `moe::GroupedRequest` and `row::precision` builders also
construct projections, quantizers, gathers, layout copies and lookups without a
`Profile`. Legal knob derivation and explicit overrides use the same validation.
For NVFP4 and MXFP8 grouped decode with BF16 activations, knobs from
`moe::packed_ring_knobs` select the grouped-MoE template's packed ring over the
same packed bank, entry ABI and arithmetic as the packed engine
([grouped MoE README](../crates/kernels-sm120/cuda/moe/README.md)).
`precision::KernelOffer` publishes complete operation capabilities from actual
selected entries. Preparation binds a source/effective precision contract once;
execution retains its typed compute and conversion handles. See
[quantized checkpoints](quantized-checkpoints.md) for the format and activation
domain.

## Generated projection offers

Quantized projections are not listed by hand. A model's projection
declaration (`contract/projections.json`, see
[projection-spec](../crates/projection-spec/README.md)) supplies each
projection's kind, shape, operands, epilogues and uses, and
`kernels_sm120::precision::generated` turns one rank-local operation into an
offer ([generated offers](../crates/kernels-sm120/src/precision/README.md)):

- `generated::offer(Request, adoptions)` takes the `OperationSpec` (already
  sliced and padded for the rank, from a derivation or `generated::domain`),
  the format plug-in id, the use and an explicit `Implementation` (dense
  kernel and capacity, grouped phase, scales and route weight, or lookup
  addressing). It checks the plug-in's decode order against the operation,
  builds the entries through `DenseRequest`, `GroupedRequest` and
  `LookupRequest`, and binds the resulting capability before returning it.
  `generated::layout_copy` covers exact copies.
- `generated::schedule::offer` picks the implementation from
  `schedule::default` or a matching record of the bundle's
  `projection-schedules.json`.
- Symbols derive from content: `chop_proj_<format>_<kind>_n<rows>_k<k>_<digest>_v1`,
  where the digest covers the neutral entry code, ABI, headers, defines,
  arithmetic and launch declarations. Equal shapes across models share one
  body; phase labels and tensor names never create another symbol. A row
  stage built with launch shapes (`ShapeBinding::Launch`, used by layout
  copies) holds no geometry and is named
  `chop_proj_<format>_<kind>_<digest>_v1`; one entry serves every layout and
  rank geometry.
- `projection-adoptions.json` keeps existing names. Each record maps a
  locked `symbol` to its `generated` identity and `entry_sha256`, with an
  optional cubin `placement` that fills a slot in an existing catalog cubin.
  A record with `replaced_by` retires its symbol for the named active entry.
  `generated::adopt` proves an initial alias against an independently built
  offer; a changed entry digest is refused rather than silently renamed. The
  ledger is append only (`tests/projection-adoptions.py`).

`kernels.lock` stays the authority for cubin bytes and source closures:
adopting a symbol changes neither. A port that moves hand lists onto
generated offers must leave every lock record byte-identical; new instances
append records and need kernel-lab oracle and invariance evidence, which the
lab enumerates from the same generated inventories (`kernel_lab::projection_cases`).

At execution, `bundle_runtime::projection` decodes the package's
precision component into retained handles, and `projection::launch(handle,
phase, input, output, scratch)` returns the gather, activation quantization,
compute, split-K reduction and output-copy launches for one bound use.
Registered entries launch through their own `projection::Launcher`: a
bundle's generated recipes run the shared sequence, and its registered
entries, up to every use, run their own launchers. See
[bound projections](bundle-runtime.md#bound-projections).

Legacy `Profile` catalogs remain supported. Attention, linear and comm keep
their existing construction paths where no typed builder exists. A shape match
or coarse numerical ID is insufficient. Model programs, fusion boundaries and
numerical choices remain with their model.

A served model's catalog profile is `Model::at(ranks)`: the model and its
tensor-parallel rank count, a number the model declares (2 and 4 for the
in-tree models). Shared 8-rank communication support remains for a model
validated on eight GPUs. `Model` and `Profile` are data a model declares
(`Model::new` with `profile_names!` and its `ModelConfig`); no shared enum
lists models or rank counts.
`kernel_core::rank_shape::RankShape` derives every per-rank size from the
model's `ModelConfig` and the rank count, with no CUDA dependency, so kernel
catalogs, bundle builds and host placement share one derivation. The
configuration lists the model's feed-forward block classes (a name, the decoder
layers that run it, and a dense MLP or a mixture of experts), and the grouped
MoE catalog generates entries per MoE class. Catalogs list each operation once
and evaluate it on the derived shape for every rank count. The first rank
count a model declares is its base (`Model::base`, TP4 in-tree): its profile
keeps the model's tag and also holds the rank-invariant kernels; the others
append `_tp{ranks}`.

A model's catalog entries live in its own contract crate
(`model_<name>_contract::kernels`, or the contract's `sm120` module): a
`static` `kernels_sm120::ModelCatalog` (GEMMs, grouped experts and routes,
attention, rows, linear attention, comm, extra invariance declarations and
the grouped-template schedule table `tuning/schedules.json`) built with the
shared engines' entry builders. A
consumer builds one immutable `kernels_sm120::Catalogs` from the models it
serves or tests and passes it to every reader (cubin generation, invariance
declarations, template instances, lab selectors). Construction refuses any
collision in the collection's namespaces (`Catalogs::cubins`): two models of
one name, two profiles of one cubin tag, two cubins of one name, an entry
symbol in two cubins or two different declarations of one symbol. Each
contract's `kernels::catalogs()` (`sm120::catalogs()` where the catalogs live
in the contract's sm120 module) holds the kernel base and that model, which
its bundle reads; `kernel_lab::catalogs()` holds every model listed in
`CATALOGS` (`crates/kernel-lab/models.rs`). Only the in-tree models are
listed, as legacy evidence: each sets `kernel_lab.catalog = true` in
`model.toml`, appends its catalog to `CATALOGS` and adds its contract to both
normal and build dependencies of the lab's `Cargo.toml`.
`python3 -B ci/kernel_lab_inventory.py` prints each missing or extra
registration line; it does not edit them. A new model does not register: it
tests its catalog from its own test crate on `kernel-harness`. Shared engines are named by
what they compute
(`gemm::fused`, `moe::expert_parallel`, `moe::mxfp4_fragment`,
`moe::sigmoid_route`); kernel symbols and cubin names keep their model names.
`kernels-sm120` depends on no model.

Model geometry, configuration and profile identities live in each contract's
`kernels::config` (or `sm120::config`): `CONFIG`, `MODEL`, the TP4 `PROFILE`, and
`profile(ranks)`.
Model CUDA lives under each bundle's `cuda/` (Qwen's fused GEMMs, routing,
hyper-connection rows and inject-norm epilogue; MiMo's fixed-shape fragment
down-combine), bound through catalog source callbacks. Parameterized
hyper-connection transports, prefill staging, stream workspaces and fragment
helpers stay shared under operation names, as do the attention engines. CUDA
namespaces, entry symbols and tuning macros keep their model names to
preserve every serving and lab cubin; renaming them is a separate qualified
change.

Encoder declarations own their cubins through `encoder::catalog` and
`encoder::cubin_specs`. The [encoder kernel guide](../crates/kernels-sm120/src/encoder/README.md)
describes their numerical contract. Shared runtime taps capture served encoder
stages, `tools/refrun/encoder.py` produces reference taps, and
`tools/qual/tapdiff.py` compares them.

Some specializations carry implementation contracts a shape match does not
show: the router GEMM refuses SwiGLU in decode and prefill (it writes router
logits), and fused all-reduce/RMSNorm is 4096-wide HF BF16-intermediate with a
required residual for v2/combine, whose tree reductions are not the ordered HF
row reduction (the v1 argument API keeps its optional residual).

## Routed experts split over the ranks

MiMo V2.6 Flash and GLM 5.3 Flash split every routed expert's intermediate
over the ranks: each rank holds a slice of every expert and runs every route
over it ([MiMo shapes](topology-mimo.md#per-rank-shapes),
[GLM collectives](topology-glm5next.md#collectives-and-precision)). As of
2026-10-15 this split stays in each model's bundle and has no shared split
path. The part that does not depend on the weight format is already shared,
one level down. The parts above it differ between the two models in
numerics or policy, so a shared path could carry them only as behavior
switches.

A model that splits its experts builds on the shared layer:

- **Slicing.** Declare the expert bank's split in the projection
  declaration: class `expert_sharded`, gate/up on `rows` and down on `k`,
  with a tile that holds whole scale groups of the format. Then derive the
  rank's records with [projection-spec](../crates/projection-spec/README.md).
  The format plug-in's `FormatPlugin::slice` (`kernel_core::precision::format`)
  keeps each slice's source scale origins. MiMo's image is built this way.
  GLM packs its expert views by hand in the chained engine's layout
  (`models/glm5next/contract/src/direct.rs`).
- **Routes.** Use `Backend::moe_route`, or `Backend::moe_projected_route`,
  which also projects the router logits. Give it a `RowQuery::Router` and a
  `MoeQuery::LocalRoutes` that has every expert local, with rank 0 on every
  rank. Pad the routes to the expert op's route tile
  (`PreparedExperts::route_tile`).
- **Expert projections.** Use `Backend::gated_mlp_projection` with a
  `GatedMlpProjectionQuery`. The weight format is its `gemm.weight`, the
  clamp its `swiglu_clamp` and the route weight point its `route_weight`.
  The model's `sm120` module registers its entries (`Sm120::with_experts`
  for grouped template entries, `Sm120::with_chained_projection` for the
  chained engine), and the backend chooses among them
  ([how SM120 chooses an entry](#how-sm120-chooses-an-entry)). A new weight
  format needs a new backend entry but no change to this shared projection
  and split interface. The bundle may still need its own changes to use the
  format: registering the entry, packing the weights and binding their
  views (GLM's chained views name each format's tensors and refuse others,
  and MiMo's expert binding supplies local scales but no global scales).
- **Sums.** Use `Backend::moe_expert_combine` (`ExpertCombineOp`) or
  `Backend::moe_routed_sum` (`MoeQuery::LocalRoutedSum`, with an optional
  F32 addend).

The model keeps everything else:

- **Combine order and dtype.** MiMo rounds each weighted route to BF16. It
  then sums a token's routes in top-k order in F32 into one BF16 row; in
  decode and verify buckets of up to 32 rows, the MLP all-reduce does that
  sum. GLM keeps its weighted routes in F32 and sums them, unrounded, with
  the shared expert's F32 output. Each choice is part of a qualified
  numerical contract, so sharing either one changes the other model's bits.
- **The all-reduce hand-off.** MiMo hands a BF16 rank partial to its hidden
  all-reduce, which adds the residual and applies the next RMSNorm
  ([MiMo prefill all-reduce](numerics.md#mimo-prefill-all-reduce)). GLM
  hands an F32 partial to `CommQuery::AllReduceRounded` before the
  hyper-connection expansion.
- **Decode and prefill policy.** MiMo routes decode and verify buckets above
  16 rows as prefill does (`DECODE_EXPERT_ROWS` in
  `models/mimo26/contract/src/serving.rs`). It also runs exact entries for
  small verify buckets. GLM prepares one op per phase kind and leaves each
  row count's schedule to the backend. Both choices are measured per model
  ([batch shape](numerics.md#batch-shape)).
- **The shared expert,** where a model has one, and where it meets the
  routed sum.

The two bundles' MoE code, 400 to 600 lines each, is the template for a
third model: `models/mimo26/bundle/src/ops/moe.rs` and `ops/experts.rs`, and
`models/glm5next/bundle/src/ops/moe.rs` and `lower/moe.rs`.

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

Grouped MoE entries stay compile-time. The ring decode asserts its shape,
sizes a register array by K (a warp's FP8 block scales) and fully unrolls its
K loop, and the schedule table picks its ring depth and warps per rank shape
and bucket. A kernel-lab measurement on 2026-10-07 built the ring decode with
N, K, local experts and top-k as launch arguments (the experiment's code
was removed afterwards). The 50 served ring decode entries
need 39 such entries, since the per-shape schedules leave few shared knob
sets. None spills, and all 1,362 cases (every live count, skewed, hot and
crowded routes, real experts) were bitwise equal. At capacity, the median pair
ratio of runtime-shape to served entry ranged from 0.89 to 1.41 (median 1.04).
39 of the 50 served entries ran more than 1% faster than their runtime-shape
entry, including every MXFP4 shape but one and small-bucket FP8 down (up to
23%), while some FP8 gate/up buckets ran up to 11% slower. That blocks a
served change: it would remove 11 entries at up to 41% decode cost.

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
