# Adding a model

The host never names a model: it loads a verified package (manifest, native
module, frontend declaration) and serves the chat format the module exports.
It fills the rank arenas from the checkpoint by
the bundle's weight plan. Everything model-specific lives in the model's own
directory. Accuracy gates come before performance work.

A model lives in its own directory: `models/<m>` in this repository, or its
own repository built against an SDK exported at a pinned revision
([external models](external-models.md#the-workflow) gives that workflow).
Either way the model owns:

- its contract, compiler and bundle crates, against the SDK's
  [supported surface](external-models.md#supported-source-surface);
- its kernels, selections and kernel locks, created and updated with
  [`relock`](external-models.md#kernel-locks);
- its chat frontend: a Rust port of its chat template in its contract crate,
  written on the opt-in `frontend-format` toolkit, the `format.parser`
  section its completions read by, and the render, tokenizer and streaming
  vectors that `chopinfer bundle check-frontend` checks on the built package
  ([chat frontend](#chat-frontend)). An in-tree model also keeps its HF
  differential corpus (a cases module, a divergence table and the committed
  subset). The corpus tools (`tools/refrun/frontend_corpus.py`) read
  `models/<m>/` and `fixtures/<m>/frontend/` under this checkout and take no
  model root, so a model outside the tree checks its frontend with its
  vectors;
- its serving profiles and deployment policy, which the compiler emits into
  every package
  ([profiles and policy](external-models.md#serving-profiles-and-deployment-policy));
- its kernel GPU tests, in its own test crate on `kernel-harness`;
- its qualification declaration and modules at its model root
  ([qualification](#qualification)).

The rules below apply to both. A model changes shared code only for a shared
capability ([what still touches shared code](#what-still-touches-shared-code));
an in-tree model also adds itself to the lists in
[in-tree maintenance](#in-tree-maintenance).

Start with the [contributor exercise](development-loop.md#first-success-without-weights-or-cuda)
before acquiring production weights or GPUs. Background:
[architecture](architecture.md#contributor-invariants), the
[external example](../examples/external-model/README.md) and the
[Qwen3.8 bundle](bundle-qwen38.md) worked example.

## Directory contract

Every model directory follows this layout: the root of the model's own
repository, or `models/<m>/` in this tree.

| Path | Required | Owns |
|---|---|---|
| `README.md` | yes | Build, package and test in one screen |
| `model.toml` | yes | Role targets, release selections, contributor recipes, CPU inputs and lab opt-in |
| `bundle/` | yes | `bundle-<m>` (`rlib` + `cdylib`): program and lowering, state, ABI entries, the chat frontend export, bundle-owned CUDA (`cuda/`), `build.rs`, `kernels.lock`, encoder adoptions, projection adoption and schedule ledgers when it keeps locked symbols or overrides schedules (`projection-adoptions.json`, `projection-schedules.json`) |
| `compiler/` | yes | `model-<m>`: config, tensor catalog, tokenizer contract, frontend, placement, packager, preparation helper, operation oracles, goldens, small `fixtures/` |
| `contract/` | no (every in-tree model has one) | `model-<m>-contract`: IDs, geometry, views, weight plan and kernel declarations that the compiler, bundle and `build.rs` all need; the chat format (`contract/src/frontend.rs`, a `frontend_format::ChatFormatV1` ported from the chat template); the projection declaration `contract/projections.json`, encoder declarations in `contract/encoders/*.json`, tuning tables in `contract/tuning/` |
| `frontend_cases.py`, `frontend-divergences.json` | in-tree, with a chat frontend | The model's cases for the HF differential corpus (template pin, transport translation, generation endings, its own feature axes and cases) and its divergence table; the committed corpus is `fixtures/<m>/frontend/corpus*.json` ([chat frontend](#chat-frontend)) |
| `validation.toml` | in-tree | Change classes, locks and kernel commands for `ci/quick.sh --report` ([testing](testing.md)) |
| `profiles.json` | yes once serving | Serving configurations ([serving-profiles.md](serving-profiles.md)); the compiler emits them into every package (`package_draft::serve_profiles`) |
| `deploy/policy.json` | for the container image | Deployment policy (`chopinfer-deployment-policy-v1`, `model_compiler_sdk::deployment_policy`): the single bundle ID, supported drafters, knob prefixes, and path, boolean, presence, numeric, multiple, tap, drafter-only and disabled-by rules for the model's own container variables; no prefix may cover a shared variable. The compiler emits it into every package (`package_draft::deployment_policy`) and the helper reports it (`Provider::deployment_policy`) |
| A kernel test crate | for model-owned kernels | GPU correctness tests of the model's own kernels and encoder stages on `kernel-harness`, outside the default members since it links the CUDA driver (the example's `lab/`) |
| `checkpoint-files.json` | for qualification | Source lock: byte length and SHA-256 of every checkpoint file |
| `reference.py`, `test_reference.py` | for qualification | GPU reference runner adapter and its CPU tests |
| `taps.json` | for qualification | Per-layer taps the bundle captures, read by `tapdiff`; required once `reference.py` has a `taps(model)` map |
| `qual.json` | for qualification | Check declarations, reference pins and release cohorts |
| `qual-references.lock.json` | for qualification | SHA-256 pins of adopted qualification inputs (a directory entry's `scope` selects the files gates read) and each object's proof record ([adopted inputs](../tools/qual/README.md#adopted-qualification-inputs)) |
| `diagnostic.py`, `agentic_markup.py` | no | Qualification modules `tools/qual` loads from the model root: scoped oracle diagnostic validation and payload checks, and the token-evidence re-parse of a format without a library tool grammar |
| `test_inventory.py` | no | The model's fixed consumer requirements (exact fixtures, release names, CPU reference steps and profile targets), a `ModelRequirements` subclass from `tests/model_requirements.py` that `tests/test_model_inventory.py` discovers |

The workspace globs `models/*/{bundle,compiler,contract}`. A crate's role is
its directory; `model.toml` records role targets and operational selections:

```toml
schema = "chopinfer-model-inventory-v1"
checkpoint_env = "M3"        # optional: CHOPINFER_M3_CHECKPOINT in CPU CI
publisher = "m3-org"         # optional: the only owner of repositories its JSON names

[bundle]
package = "bundle-m3"
cdylib = ["bundle_m3"]

[compiler]
package = "model-m3"
bin = ["chopinfer-model-m3", "m3-package"]

[contract]
package = "model-m3-contract"
```

`publisher` names the model's author. Every `repository` value in the
model's JSON (under its directory and its fixture root) must then be one of
that publisher's repositories, so the tree names only the author's
checkpoints; another checkpoint is admitted by its layout and qualified as an
[external variant](../tools/qual/README.md#external-variants). Each in-tree
model declares one.

Operational sections are optional and reject unknown keys. Declare them when
the model participates in the corresponding workflow:

| Section | Model supplies |
|---|---|
| `release` | Explicit `enabled` opt-in, selected `bins` and `cdylibs`, and extra `assets` with source/destination pairs. Declaring a Cargo target alone does not ship it. |
| `contributor` | Own-role package `roots`, bounded workspace-relative `fixtures`, extra `clear_env` keys and `tests` with package, phase, lib/bin/suite selections and exact cases or compile-only intent. Recipes are data, not shell commands. |
| `checks` | Model-relative `reference_tests`, bounded `compile_inputs` glob patterns and typed `profile_targets` (package, kind, target). |
| `kernel_lab` | In-tree legacy: `catalog = true` requires the kernel-lab registration of the in-tree models ([in-tree maintenance](#in-tree-maintenance)). A new model does not set it. |

Inventory validation checks Cargo targets, files, confined paths, symlinks and
release destination collisions. Locks use shared release paths, and profiles
ship inside each package; eval-file discovery includes declared qualification inputs. Docker and the
shared redeploy installer consume the same selected artifacts. Qualification
obligations in `validation.toml` and `qual.json` remain separate.

`tests/bundle-boundary.sh` checks the inventory against `cargo metadata` and refuses
unknown keys. A bundle may link `bundle-runtime`, `kernels-sm120`,
`kernel-core`, `encoder-spec`, `runtime-bundle-abi`, `runtime-bundle-sdk`,
`frontend-format` and `runtime-grammar`, never `runtime-cuda`, `runtime-engine`, `runtime-host` or `runtime-server`.
Shared runtime crates must not depend on model or bundle crates.

To serve with `serve --model`, the compiler implements
`model_compiler_sdk::preparation::helper::Provider` (`probe` routes a
checkpoint by metadata; `emit_direct` admits it and emits a package whose
images are planned components) and ships it as the `chopinfer-model-<m>`
binary. The host discovers installed helpers beside itself
([security boundary](security-boundary.md),
[compatible checkpoints](compatible-checkpoints.md)).

## What still touches shared code

A new model touches shared code for three reasons, and nothing else:

- a new shared kernel the model needs and other models can use;
- a model-neutral capability the host lacks, such as a media expansion (for
  example DeepSeek V4.1's image-span expansion) or a tokenizer component the
  host's reader does not implement;
- a chat frontend piece `frontend-format` lacks: a completion syntax neither
  shared parser family reads, or a Python semantic the toolkit does not
  implement.

Two proofs back this. On 2026-10-14 a scratch model built outside the tree,
with its own Rust chat, reasoning and tool-call format and a tokenizer
policy no in-tree model declares, passed `chopinfer bundle check-frontend`
with 29 vectors and served chat requests over HTTP with fake host services,
with no chopinfer edits. On 2026-10-15 GLM 5.3 Flash, exported from this tree
and built against the SDK, served the same output as the in-tree build under
the same host binary at TP4 (`tp4-ar` and `tp4-mtp`): the determinism records
were equal over 12 requests and 482 tokens, the maximum logprob difference was
0.0, and MTP acceptance was 95 of 95 in both builds.

| Shared location | When a model touches it, and why |
|---|---|
| `kernels-sm120`, `kernel-core` contracts, a kernel-lab case and lock family | A new shared kernel or builder that other models can use. Prefer profile-free builders and audit each stage of an existing adapter before assuming it fits; add generic builders with oracle evidence and a kernel-lab conformance case ([recipe](kernel-library.md#adding-an-op), [lock families](kernel-library.md#adding-a-kernel-lab-lock-family)). A kernel only this model runs stays in its tree behind `OwnOps` ([kernels](#kernels-through-kernel_corebackend)). A model registers its own entries through its own catalog, so the catalog-driven builders that name Qwen (`gemm::fused`, `row::hyper_connection`, `moe::expert_parallel`) need no edit. |
| `runtime-media`, `runtime-frontend` media checks | A model-neutral media capability the host lacks, such as a new expansion of media spans in the prompt. Keep it a declared, model-neutral rule with tests. |
| `runtime-frontend-contract`, `runtime-tokenizer` | A tokenizer component the host's reader lacks: add the policy field or value in `runtime-frontend-contract`, implement it in `runtime-tokenizer` with tests, and keep it model-neutral ([tokenizer](#chat-frontend)). |
| `frontend-format`, `tools/qual/agentic_mixed.py`, `tools/refrun/frontend_cases.py` | A completion syntax neither shared parser family reads: a new named family that any model may then declare, with its `FrontendToolGrammarV1` variant (`src/descriptor.rs`), its decoder, its tool-choice grammar (`src/tool_grammar.rs`) and tests, the qualification re-parser's reading of it (`TOOL_GRAMMARS` in `agentic_mixed.py`) and its delimiter cases in the corpus generator. A Python semantic a ported template relies on that the toolkit lacks: a `frontend_format::toolkit` helper, with its rows written by `tools/refrun/python_semantics.py` into `tests/python_semantics.json` ([renderer](#chat-frontend)). |

A chat format, reasoning rule, serving profile, deployment rule, kernel
lock, kernel test or qualification module is model-owned and changes no
shared file: the chat format is the model's own Rust, written on the
`frontend-format` toolkit. The two `frontend-format` changes above are the
only shared frontend changes. Never add a flag to an existing family for one
model.

## In-tree maintenance

The models under `models/` are also registered in lists that only this
repository's own build, tests and CI read. A model in its own repository
never edits them. An in-tree model updates these rows, each recorded rather
than moved because an external model never needs it:

| Location | In-tree action | Why it stays central |
|---|---|---|
| `Cargo.toml` and `Cargo.lock` | Member globs discover the three role directories; add workspace dependency aliases if other crates use the model's crates, and refresh the lock offline | This repository's workspace; an external model has its own lock |
| `models/<m>/model.toml` `release`, `contributor`, `checks` and `geometry_recipe` sections | Opt in to the release archive, declare contributor recipes, CPU checks and geometry recipe legs; pin them, and the deployment policy with `deployment_policy()`, in the model's `test_inventory.py`. Keep preparation arguments, build features and serving choices in a model-owned redeploy wrapper when they differ. Model-specific geometry recipe checks (refusal messages, checkpoint registration) are added by hand in `tests/model_geometry_recipe.py` | Read by this repository's release plan, `ci/contributor.py`, `ci/quick.py`, `ci/model_checks.py` and `tests/model-geometry-recipe.sh` |
| `tests/model-neutral-allowlist.json` | A registration this table requires in a shared file raises that file's model-word count; update the entry in the same commit, with its reason. Never raise a count to admit model behavior in shared runtime code | Counts of this repository's shared files |
| `crates/xtask/src/lib.rs` `REVIEWED_UNSAFE_FILES` | Only for new custom unsafe code, with its safety review; `runtime_bundle_sdk::embed_note!` needs none | This repository's lint policy |
| `deploy/systemd/chopinfer-<m>.service`, `<m>*.env.example`, `tests/test_deploy_artifacts.py` | Add a unit and examples | In-tree convenience units; an external model ships its own |
| `tests/test_deploy_entrypoint.py`, `tests/model_geometry_recipe.py`, `tests/model_export_recipe.py`, `tests/data_only_recipe.py`, `tests/test_ci_runner.py` | Add the model's helper discovery, checkpoint refusal, profile default and deployment rule cases where these list models | In-tree test lists and branches |
| `ci/cpu.py`, `ci/source_dependencies.toml`, `ci/validation.toml` | Keep model-specific CPU gates (such as redeploy-lock or topology-lab checks) and source edges explicit | In-tree CI only |
| The closed model sets of the `tools/qual` tests, Qwen's own qualification modules, `tools/quickstart.py` | Extend them when the model joins those tools | In-tree tool tests and helpers; an external model extends qualification through its model-root modules |
| kernel-lab `models.rs`, `Cargo.toml`, build probes, model tests, `tests/encoder_conformance*.rs`, its `kernels.lock` records, `ci/kernel_lab_inventory.py` | None for a new model: it tests from its own crate on `kernel-harness`, including encoder conformance (`kernel_harness::encoder`) | Kept as legacy evidence of the in-tree models |
| Model-named symbols and tags in `kernels-sm120` | None | Renaming changes symbols or source closures, and therefore locks |
| `tools/dev`, `tools/profile`, `tools/ab`, `tools/refrun/calibrate.py`, `tools/load-proto` | Extend when needed | In-tree development tooling |
| `docs/README.md` | List new docs | In-tree documentation |

The `geometry_recipe` section declares `stage` (`contract` while the bundle
cannot build and serve an added geometry, because it has no program yet or
is compiled for fixed geometries, then `bundle`), `fixtures` (directories under
`fixtures/<m>/geometries/`, each with `config.json`, `geometry.json` and
`supported.json`), the `architecture` alias the added geometry registers, and
at the bundle stage the contract crate's `contract_tests` suites.
`bash tests/model-geometry-recipe.sh --ci` (bundle stage) or `--contract`
then runs one leg per fixture; pin the legs with `geometry_legs()` in the
model's `test_inventory.py`. A contract-stage leg runs the contract crate's
`geometry_recipe` test with `CHOPINFER_<M>_RECIPE_CONFIG` and
`CHOPINFER_<M>_RECIPE_GEOMETRY`. A bundle-stage leg runs the shared steps
(lock update build, `projection_assembly`, the declared suites, the
`geometry-checkpoint` example, the bundle's
`every_geometry_topology_runs_the_fake_host_journal_and_lifecycle` test,
helper probe and `prepare`).

The CPU plan (`ci/model_checks.py`), `ci/qualification.py`,
`tests/test_validation.py`, `tests/test_serve_profiles.py`, the released and
tap sets of the `tools/qual` tests, `tools/qual/test_release_gate.py`, the
`tools/qual` workers and the `crates/runtime-eval` tests read the inventory
or the declarations and need no edit. Check shared lists again when landing,
since a list may have become inventory-driven. Run the inventory, boundary
and deployment tests, the new compiler and bundle suites, qualification
declaration validation and the portable gate, and use the new binaries'
`--help` to verify every command the model README documents.

## Model implementation

Keep tensor and projection declarations in the existing model contract. The
bundle supplies an explicit forward program: stage order, phase selection,
launch lowering, fusion boundaries and numerical choices remain model-owned.

The contract's `build.rs` generates its geometry catalog with
`model_compiler_sdk::geometry_build`: `Catalog` reads `geometries/supported.json`
and the directories its environment variable lists, and `Schema` checks each
document against the model's field and group tables and renders them. The
model keeps those tables, its structure check and the items it renders.

Declare admission as data where the shared code reads it, rather than in model
code. `model_compiler_sdk::geometry::admit` checks a checkpoint config's
semantic rules and quantization recipe class, taking a recipe without a named
method as the `quant_method_default` the geometry declares. The runtime limits
the engine supports go in the geometry's `engine_domain`: each config path
names one limit or a list of them (`values`, a range with `min`, `max` and
`multiple_of`, or `max_length`, each with its `reason`), and an optional
`phase` places a limit before or after the model's own relation checks
(`geometry::check_engine_phase`), so refusal order stays the model's choice.
Kernel tuning for a shape without its own entry is inherited with
`kernel_build::tuning::{nearest_first, inherit}`, and a package's device
budget starts from `model_compiler_sdk::device::{DEVICE_BYTES,
HOST_RESERVE_BYTES}`; the model adds only what it keeps outside the KV pool.

### Projection declarations

Quantized projections are data. `contract/projections.json` declares every
projection (checkpoint name pattern, `[rows, k]` per operand, kind, TP split
class, epilogues, numerical contract and uses), the fusion groups, the layer
stems (decoder, MTP and drafter layers alike) and the side tensors such as
norms and routers, with their admitted dtypes. The contract parses it with
`projection_spec::parse` and a `Registry` of the model's registered variants,
and a test pins its digest. From that one file:

- the compiler binds checkpoint headers (`Declaration::bind`), so unknown
  tensors and missing companions are refused by name, and the weight plan
  takes each rank's slices from the declaration (`derive::rank_selection`, or
  a full `derive::derive` turned into records by the SDK's
  `quantization::projection`);
- the bundle's build generates kernel offers with
  `kernels_sm120::precision::generated`. A bundle that keeps existing locked
  symbol names records them in `projection-adoptions.json`, and one that
  overrides default schedules records them in `projection-schedules.json`
  (both append only, checked by `tests/projection-adoptions.py` and
  `tests/projection-schedules.py`); a bundle with neither has no ledger;
- the bundle binds the package's precision component into
  `bundle_runtime::projection` handles at create and lowers each use
  with `projection::launch`.

Register a variant only for arithmetic the shared schema cannot express,
such as a source transform, a stateful stage or an existing fused entry kept
for byte identity, and list the model's variants in its own guide. What a
new layout or rank count needs, and when model code is still required, is in
[quantized checkpoints](quantized-checkpoints.md#adding-a-family-or-layout);
[the external example](../examples/external-model/README.md) shows a
declaration with no registered variants.

Declare the scratch record once with `kernel_core::scratch_record!`, fill its
`Allocation` leaves with model size formulas, and use `Traversal` for ordered
allocation, named rows and admission sums. `bundle_runtime::scratch::bind`
maps that declaration to host spans through the caller's allocator. The model
still defines aliases, zeroing and create order.

Choose request-state, cancellation, ring and checkpoint semantics explicitly.
The shared ledger provides validated commit plans; optional entry maps manage
checkpoint ownership, and `HostRankStreams` adapts snapshot copies. These
helpers do not choose device effects or rollback policy. Supply a model adapter
and independent expectations for the shared fake-host conformance scenarios,
and a `Startup` adapter with one serving per profile for the serving startup
scenario ([bundle runtime](bundle-runtime.md#modules)).
See [bundle runtime](bundle-runtime.md) for the APIs and boundaries.

Start from the shared steps instead of copying another bundle's versions. Each
takes the model's choices as hooks or parameters:

- `transaction` ends a step (commit, accepted-row commit, cancel, retire) in
  one order over the ledger; the bundle supplies its device effects and the
  status each refusal returns.
- `acquire::rank_images` reads, admits and authenticates every rank index
  before the image opens; `index::index_role` names the rank index roles and
  `index::encoder` reads and writes a BF16 encoder image index.
- `drafter::Drafter<P>` keeps an MTP drafter's rows, catch-up and retained-row
  copies when the drafter runs beside the target.
- `fake_host::backend::Backend` runs the fake host for tests over a model's
  `FakePackage` (its components, images and failure hooks), with the device
  journal, handle accounting and failure injection shared.

### Kernels through `kernel_core::backend`

A model reaches kernels through contracts, not symbols. The backend side is
in [adding a GPU backend](kernel-library.md#adding-a-gpu-backend).

- State each op as a kernel-core contract (a requirement or query) and a
  `Need`: the most rows one launch processes, and the exactness the model
  relies on, such as verify rows equal to decode rows
  (`Invariance::one_row`) or rows independent of the prefill chunk split
  (`Invariance::PREFILL`). A backend that cannot keep the need refuses.
- Prepare every op once, at instance creation, through `&dyn Backend`, per
  rank and launch key (phase and row bucket), and keep the prepared ops in
  the instance. Read environment switches there too. Per step, build each
  op's role record over the step's buffers and call `launches`; schedule by
  `Launch::collective` and `Launch::persistent`, not by symbol names or SM
  counts.
- Keep SM120 selection in `sm120` modules. Model code names `kernels_sm120`
  only in an `sm120` module (a directory or file named `sm120`) of its
  crates. There the model registers its entries, catalogs and tuning on
  `Sm120` and hands it out as a `Box<dyn Backend>`; the rest of the model
  states math and prepares through the trait. Files that named
  `kernels_sm120` before this rule are frozen in
  `models/<m>/kernels_sm120_imports.txt`, and the list only shrinks: remove
  a file when it stops naming `kernels_sm120`, never add one. A new model
  has no list. `bash tests/bundle-boundary.sh` enforces this
  (`check_model_kernel_imports` in `tools/model_sdk/core_policy.py`).
  Nor does the rest of the model reach SM120 items another way. Outside
  `sm120` modules and test code, these fail (`check_model_sm120_reexports`):
  a `pub use` (any visibility but `pub(self)`) through an `sm120` module or
  `kernels_sm120`, a glob re-export of `crate`, `super` or `self`, a module
  whose `#[path]` or `#[cfg_attr(.., path = ..)]` points into an `sm120`
  module, and an `include!`, `include_str!` or `include_bytes!` of such a
  path. So do the other names of an `sm120` module outside it
  (`check_model_sm120_uses`): a private or `pub(self)` `use
  crate::sm120::..`, or a path such as `crate::sm120::backend()` in code.
  Test code is `#[cfg(test)]` items, files that only test code declares as
  modules, and a crate's `tests/` and `benches/`. Names that predate these
  rules are frozen as `<file> <item>` lines in
  `models/<m>/sm120_reexports.txt` and `models/<m>/sm120_uses.txt`, whose
  baselines hold exactly the listed lines' hashes: remove a line and its
  hash together, never add either. The checks are lexical and catch
  accidental coupling, not code written to hide it: they do not see paths
  that `concat!` builds from pieces that do not name `sm120` themselves,
  macros that expand to such paths, uses through an alias of an `sm120`
  module (the alias is listed), globs of other modules that re-export one,
  or `#![cfg(test)]` files (production unless test code declares them);
  `#[cfg(any(test, ..))]` items count as production. Known gaps, where a
  check can miss a name or fail on test code:
  - a module with several `cfg_attr` paths keeps one, and a production
    file that a test module also declares by path can count as test code;
  - children of a `#[path]` module outside `mod.rs` are looked for at
    `parent/child.rs`;
  - helpers that `tests/` or `benches/` files declare by `#[path]` are not
    exempt;
  - a `#[cfg(test)]` item ends at its first `;`, also one in an array type;
  - path attributes on inline modules are ignored;
  - a production `include!` of a test-declared file keeps it test code;
  - `include! { .. }` with braces is not seen;
  - nested block comments are not handled.
- Give the model's own kernels contracts. A kernel no other model runs gets
  a contract type (its math: layouts, addressing, skip rules), a role
  record and a CPU oracle that computes what the kernel computes, in the
  model's contract crate, plus a method on the model's `OwnOps` trait there,
  refusing by default. Implement `OwnOps` for SM120 in the bundle's `sm120`
  module, where SM120 choices (threads, vector widths, compiled constants)
  are refusals, not contract rules. MiMo is the example
  (`models/mimo26/contract/src/own_ops/`,
  `models/mimo26/bundle/src/sm120/own_ops.rs`). An op moves to
  `kernel-core` when a second model needs it.
- Run encoders on the shared encoder runtime (`bundle_runtime::encoder`)
  created with `Encoder::with_backend`: the model's `Backend` (for SM120,
  `Sm120::with_encoders` over its encoder catalog) and a resolver of each
  declaration slot (stage, part, block) to the query its catalog entry was
  generated from, or, for a registered stage, to the model's own op prepared
  through `OwnOps`. The runtime prepares every slot through the multimodal
  encoder methods when it lowers the declarations and checks every image view
  against its stage at creation. The walk from declarations to queries is in
  `kernels_sm120::encoder` (`slots`), so the resolver sits in SM120 code; a
  second backend takes its queries from the same walk.

New kernels get a declarative conformance case
([kernel library](kernel-library.md#declarative-conformance-runs)), including
a chain of entries for split-and-merge kernels and FP8 KV variants where the
kernel has them: a model's own kernels and encoder stages in its own test
crate on `kernel-harness` (`kernel_harness::conformance` and
`kernel_harness::encoder`), as the example's `lab` does, and a shared kernel
in kernel-lab. A model's required ops can also run, against any backend,
in the [conformance kit](kernel-library.md#running-the-conformance-kit).

Supply the chat frontend, authenticated checkpoint identities, reference
adapters and qualification metadata.

### Chat frontend

The compiler emits `frontend.json` (`chop-frontend-openai-v2`) into the
package, and the bundle exports the format that renders and parses it
([frontend export](../BUNDLE_ARCHITECTURE.md#frontend-export)). The host
reads the model-neutral sections and passes `format` to the bundle unread.

- **Renderer.** Port the chat template to Rust in the contract crate as
  `contract/src/frontend.rs`, a `frontend_format::ChatFormatV1`: one
  function per template macro or block, each with a comment citing the
  template lines it ports, written with `frontend_format::toolkit` (prompt
  bounds, reserved-marker screening, Python's JSON and whitespace, tool
  objects, reasoning controls and refusals); a Python semantic the toolkit
  lacks is added to it by the port that needs it, with its rows in
  `tools/refrun/python_semantics.py`; a model outside this tree keeps such a
  helper in its own contract crate until an SDK revision carries it. The
  renderer writes its own media
  labels. Its `open` accepts only what it follows: the parser section its
  template writes, kept beside it as `parser_declaration()` and compared
  whole, the reasoning controls its template reads, and media kinds it has
  labels and pads for. An export test refuses a changed field of each.
  The bundle exports it with `frontend_format::export_chat_format_v1!(Format)`
  at its crate root, and the compiler's tests reach it through the contract
  crate they already link ([its README](../crates/frontend-format/README.md)).
- **Parser section.** Declare in `format.parser` how the model's
  completions read: one of the shared tool-call families
  (`xml-function-parameters-v1` or `xml-key-value-arguments-v1`), the JSON
  encoding, the reasoning and tool-call markers as the template writes them,
  the turn-opening boundary tokens and the assistant header. The export
  parses completions and builds tool-choice grammars from it with the shared
  parsers, so the model writes no parser. When its completions fit neither
  family, an in-tree model adds a new named family to `frontend-format`
  ([what still touches shared code](#what-still-touches-shared-code)). A
  model outside this tree whose pinned SDK has no family for them may instead
  implement `runtime_bundle_abi::frontend::FrontendV1` itself, export it with
  `export_frontend_v1!` and supply `agentic_markup.py`
  ([marker, effort and parser rules](external-models.md#chat-frontend)).
- **Corpus** (in-tree models, as the ownership list at the top of this page
  states). Write the model's cases module (`frontend_cases.py`: the
  template pin, the transport translation the HF input follows, the
  generation endings of each turn shape, how a history turn reads back, and
  the model's own feature axes and cases) and its divergence table
  (`frontend-divergences.json`: template lines no accepted request reaches,
  and requests whose outcome differs from HF's on purpose, each with its
  reason). A checkpoint that ships a Python encoder instead of a chat template pins
  the encoder (`PIN["source"]` and `source_sha256`) and maps the HF input
  onto it (`encoder_render`), and the corpus traces the encoder's function
  lines as it would a template's. Then run `python3 -m tools.refrun.frontend_corpus generate --model
  <m>`, which renders every case with HF at the pinned template (or with the
  pinned encoder) into
  `$CHOPINFER_DATA/frontend-corpus/<m>/`; run the model's Rust corpus test
  over that full corpus with `CHOPINFER_FRONTEND_CORPUS_DIR` set; `minimize`
  it into `fixtures/<m>/frontend/corpus*.json`, which keeps every template
  line, feature tag, refusal class, divergence case and round-trip shape;
  and run `check`, the coverage and divergence gate. The committed corpus
  runs as raw requests through the served frontend: renders, token IDs,
  refusals, and round trips at every byte split.
- **Corpus test.** The Rust corpus test,
  `compiler/tests/frontend_corpus.rs`, is a few lines over the shared
  `runtime_host::frontend_corpus` (`runtime-host` with its `test-support`
  feature as a dev-dependency): it fills a `FrontendCorpusV1` with the
  bundle's export, the served component the corpus renders with, the
  committed directory and the tokenizer directory from the model's
  variables, and its three tests call `check_committed`, `check_full` (the
  corpus `CHOPINFER_FRONTEND_CORPUS_DIR` names) and `check_transport` (the
  coverage's `transport` against the export's open answer). With
  `CHOPINFER_FRONTEND_CORPUS_DUMP` naming a file, the selected corpus's
  outcomes are written there, so two runs compare. How a history turn
  reads back stays in the cases module's `expected_turn`.
- **Round trips.** The HF corpus (`tools/refrun/frontend_corpus.py`,
  [its README](../tools/refrun/README.md)) reads each history turn back
  through the parser as a round trip, which holds within its lossless
  domain: the history render less the generation prompt is the completion,
  every argument has the one type its schema names (a string where none
  types it, a number for an integer), no value the parser scans as raw text
  spells a marker that ends it, and the tokenizer's normalizer leaves the
  completion as it is. A template that writes a union-typed value, a string
  that spells its own close marker, or text the tokenizer normalizes (such
  as NFC) is lossy there without escaping, and the renderer must not invent
  escaping the template does not write. Declare each such loss the corpus
  finds as a `no-round-trip` entry naming its class (`round-trip:union`,
  `round-trip:delimiter`, `round-trip:tokenizer` and so on); the gate
  refuses a failed round trip without one, and an entry no committed case
  fails as it names. A tokenizer loss still yields a round trip that
  expects the normalized turn, so the parser is checked on the text it
  reads.
- **Tokenizer.** Declare the files and the component policy in
  `tokenizer.policy`: the normalizer, the byte-level flags of the
  pretokenizer, post-processor and decoder, merge result ids and record
  shape, ignore-merges, base and config token forms, the `add_bos_token`,
  `bos_token` and `unk_token` fields, the tokenizer class, subword affixes,
  `add_eos_token`, `legacy` and an empty `sp_model_kwargs`. The reader refuses a policy it does not implement
  (`unsupported tokenizer policy: ...`). A tokenizer component the host
  lacks is a shared capability: add the policy field or value in
  `runtime-frontend-contract`, implement it in `runtime-tokenizer` with
  tests, and keep it model-neutral.
- **Reasoning.** Declare `reasoning.end_markers`, the boundary tokens that
  end a reasoning section; constraints wait for one when the render leaves
  reasoning open.
- **Vectors.** Keep render, tokenizer and streaming vectors from HF
  `apply_chat_template` or the model's reference encoder as the model's
  goldens, and check a built package with `chopinfer bundle check-frontend`
  ([frontend vectors](external-models.md#frontend-vectors)).

#### Updating the chat template

When the checkpoint's chat template changes, the corpus finds what an
in-tree model's renderer must change. A model outside this tree regenerates
its render and streaming vectors from HF `apply_chat_template` at the new
template and reruns `chopinfer bundle check-frontend`. The in-tree steps:

1. Bump the pin in `frontend_cases.py` (`PIN`: the revision and the
   template's SHA-256).
2. Run `python3 -m tools.refrun.frontend_corpus generate --model <m>`.
3. Read the cases that now differ, with the model's Rust corpus test over
   the new corpus (`CHOPINFER_FRONTEND_CORPUS_DIR`), and the template lines
   no case runs any more or that are newly uncovered.
4. Port the changed lines in the model's `frontend.rs`, citing the new line
   numbers.
5. Update `frontend-divergences.json`: the gate reports an entry whose lines
   a case now runs, or whose cases now match HF, as stale.
6. Run `minimize` into `fixtures/<m>/frontend/`, then `check`.

`generate --template PATH` runs an unpinned template instead, which is how
the procedure is rehearsed against an older template revision before a pin
moves.

The generic BPE artifact test in
`runtime-host/tests/paired_merge_artifacts.rs` covers verified artifact
loading, encode/decode and malformed inputs; it does not qualify a model's
tokenizer or chat template.

### The seam: definition of done

A model reaches kernels through the backend interface when all of these
hold:

- Generic model code names no SM120 item beyond its two construction lines.
  Outside `sm120` modules nothing names `kernels_sm120`, test code included;
  outside `sm120` modules and test code nothing re-exports an SM120 item,
  and only the backend and own-ops construction names a path through an
  `sm120` module. The guard lists `kernels_sm120_imports.txt` and
  `sm120_reexports.txt` are empty and deleted. Each shipped model's
  `sm120_uses.txt` keeps exactly those two lines
  (`contract/src/backend.rs crate::sm120::Sm120` and
  `bundle/src/instance.rs crate::sm120::own_ops::Sm120Own::new`), which
  `tools/model_sdk/core_policy.py` checks. A new model that constructs its
  kernels in its `sm120` module, as the external example does, has no
  lists.
- The `sm120` modules are mostly data: the entries the build compiles
  (selections, catalogs, tuning tables, adoption ledgers), their
  registration on `Sm120` with its `with_*` methods, and the `OwnOps`
  implementation over the model's own cubins, whose launches only plan and
  pack arguments. They choose no entry for an op.
- Selection lives in the backend: `Sm120` picks among the registered
  entries by the op's contract, use, capacity and needed exactness
  (`crates/kernels-sm120/src/backend/select.rs`,
  [how SM120 chooses](kernel-library.md#how-sm120-chooses-an-entry)). A
  model that needs another entry changes its contract or its `Need`, never
  a symbol.
- Ops are prepared once, at instance creation, through `&dyn Backend` and
  the model's `&dyn OwnOps`, which the target module supplies without
  generic code naming it. The external example declares
  `struct Kernels { backend, own }` in generic code and implements
  `Kernels::new` in its `sm120` module
  (`examples/external-model/bundle/src/sm120/mod.rs`); its build script,
  which only compiles and locks SM120 cubins, lives there too
  (`build = "src/sm120/build.rs"` in its `Cargo.toml`).
- The conformance kit runs the model's required ops on the backend, marked
  `.required()`, with every op prepared and, on a GPU, every conformance
  run and needed property passing: verify rows against decode launches for
  ops over items of several rows, and collectives through the multi-rank
  runner ([running the kit](kernel-library.md#running-the-conformance-kit)).

The guard lists shrink and never grow. Each listed line has its SHA-256 in
a baseline (`tools/model_sdk/kernels_sm120_imports.sha256`,
`sm120_reexports.sha256`, `sm120_uses.sha256`), and
`bash tests/bundle-boundary.sh` fails on a new name outside the lists, on a
listed line the code no longer has and on a line whose hash is not in the
baseline. Moving a file's SM120 code into the `sm120` module, or behind
`kernel_core::backend`, removes its line; the same commit removes the line's
hash, and a list that empties is deleted. The external example's list
emptied this way.

## Same-architecture checkpoints

A same-architecture checkpoint uses the family's admitted aliases, semantic
rules and quantization recipes in `models/<m>/geometries/supported.json`.
Sizes come from `config.json`; layer kinds, activations, projection layout,
RoPE semantics and tensor encodings must remain supported. Shapes must fit
both the model's engine domain and a compiled geometry
([engine contracts](kernel-library.md#engines)). An alias registers a
spelling, not a new execution path. Each model's page lists its domain and
the extra steps this recipe needs for it:
[Qwen3.8](model-qwen38.md#same-architecture-checkpoints),
[MiMo V2.6](serving-mimo26.md#registering-a-checkpoint) and
[GLM 5.3](model-glm5next.md#geometries).

1. Add the checkpoint's execution config as `models/<m>/geometries/<id>.json`
   (test-only geometries live under `fixtures/<m>/geometries/` instead). Add
   its `id` and `file` entry to `supported.json`, and an alias only if the checkpoint uses a new spelling
   of the same architecture. Choose a unique `kernel_tag`; reserve
   `pinned_names` for historical identities. Catalogs, resources and capacity
   buckets derive from the geometry and the shared row list. Runtime selects
   it through topology ID `<model>-<geometry>-tp<n>`, within the existing
   single bundle ID.
2. Pin authenticated source files in `checkpoint-files.json` and add the
   checkpoint's identity or goldens as the model's page describes. KV
   calibration, if the model has one, belongs to that identity.
3. Rebuild the compiler and bundle with Cargo's `--locked --offline` flags.
   When adding a geometry, build the owning bundle with
   `KERNELS_LOCK_UPDATE=1`. New kernel instances compile at build time.
   Review the appended `kernels.lock` records and preserve existing records;
   never refresh a lock to hide a mismatch. Measured tuning is geometry-owned
   under `contract/tuning/<id>/`. Untuned geometries use the
   [validated fallback](kernel-library.md#untuned-geometries), with no source
   edits or runtime compilation.
4. Run `bash tests/model-geometry-recipe.sh <m>`, the catalog and compiler
   checks, and the [full CPU checks](testing.md). Then produce pinned
   [GPU references](reference-runner.md) and rerun the owning model's
   [qualification gates](#qualification) for every intended topology and
   serving mode. Keep outputs under `$CHOPINFER_DATA`. Synthetic CPU evidence
   proves admission and lifecycle mechanics; each checkpoint needs its own
   numerical requalification before accuracy or performance claims.

Refusals name the field, expected rule and received value, for example
`unsupported text_config.hidden_act: expected ..., received ...`.
An unmatched size reports `unsupported geometry`, the differing fields and
`supported ids`; an engine restriction names the shape constraint. These
checks happen before weight loading. Runtime fields remain bounded by the
compiled geometry; a field the kernels compile stays non-overridable, as
Qwen3.8's `indexer_budget` does ([its page](model-qwen38.md#shape-domain)).

## Where the details are

Bundle mechanics: [bundle-runtime.md](bundle-runtime.md). Weight plans and
packages: [direct-loading.md](direct-loading.md). Kernels, selections and
locks: [kernel-library.md](kernel-library.md). Numerics:
[numerics.md](numerics.md). References: [reference-runner.md](reference-runner.md)
and [qwen38-oracle.md](qwen38-oracle.md). State, speculation, media and KV:
[serving-recurrent-state.md](serving-recurrent-state.md),
[adaptive-speculation.md](adaptive-speculation.md),
[serving-request-surface.md](serving-request-surface.md),
[serving-kv-pool.md](serving-kv-pool.md). Tests: [testing.md](testing.md).
Gates: [tools/qual/README.md](../tools/qual/README.md). Topology and
deployment: [deploy-common.md](deploy-common.md).

Read each kernel's implementation contract before selecting it; shape alone
does not establish compatibility (for example, the fused all-reduce/RMSNorm
builder requires width 4096, HF BF16-intermediate semantics and a residual).
Store only small test inputs in `fixtures/`; keep captures and qualification
outputs under `$CHOPINFER_DATA`.

## Qualification

A new checkpoint needs a model-owned package, frontend, checkpoint identity
and reference adapter before qualification can pass. Declarations select
existing check implementations; they do not implement a model or produce
reference scores.
A model outside this tree passes its declaration and its model root to the
release gate, which loads its qualification modules from that root
([external models](external-models.md#packages-updates-and-qualification));
the in-tree commands below take `--model` instead.

Start from the [qual.json template](../tools/qual/templates/qual.json) and
its field reference in [tools/qual/README.md](../tools/qual/README.md). Its
top-level fields are:

| Field | Holds |
|---|---|
| `schema`, `model` | `chopinfer-qual-model/v1` and the model name |
| `cohorts` | Ordered `quick` and `full` check lists |
| `change_classes` | Which checks each change class selects |
| `checks` | Per check: `module` (an existing worker), `args`, `timeout_seconds`, `receipt` pattern and `rules` |
| `references` | Pinned reference files: `path`, `sha256`, `schema`, `variants`, `workload`, `checkpoint_keys` |
| `inputs` | Input locations, directly or as `{"env": ..., "path": ...}` under `CHOPINFER_DATA` |
| `qualification` | `checkpoint` pin (`lock`, `repository`, `revision`, `served_model_name`, `source_lock_sha256`, `checkpoint_identity`), `mechanics_only`, teacher and `media` settings |
| `receipts` | Where receipts are written |
| `limitations` | Known limits of the declaration, in prose |
| `capabilities` | Served behavior: `text`, `frontend`, `api`, `structured`, `prefix_cache`, `preemption`, `long_context`, `stability`, `media`, `speculation`, `kv_offload`, and optionally `unqualified_media` |
| `preview` | Optional: the correctness runner's serving profile, speculation, GSM8K floor and screen rules |
| `configurations` | Optional: each other serving configuration or checkpoint, by name, expanded onto the base declaration ([configurations](../tools/qual/README.md#configurations)) |

Every declared capability requires its hard checks in the `full` cohort. For
case-format media, `qualification.media.coverage` lists reference IDs per
modality, and the scorer verifies their kinds against the authenticated
reference. A modality the model serves without in-tree reference evidence
goes in `unqualified_media`, mapped to the reason, never only left out of
`media`, and holds no coverage; encoder parity then requires no rows of an
encoder whose every input kind is unqualified. Unsupported media is
`not_applicable` at serving time; it must not silently skip required
evidence. Do not use `informational_reason` to count missing evidence as a
pass. Unknown fields and modules are errors.

All commands run from the workspace root. The examples use the MiMo
declaration; substitute your model. Set `CHOPINFER_DATA` to an output
directory outside the checkout and `URL` to an already-running candidate
endpoint with the checkpoint and profile being qualified. The gate does not
start servers or allocate GPUs.

1. Pin the checkpoint and declare inputs, references, checks, predicates,
   timeouts and cohorts. Use `python3 -B -m tools.qual list` and
   [check-kinds.md](check-kinds.md) to choose existing workers. Inspect the
   declaration without a device:

   ```sh
   cargo run --locked --offline -p runtime-host --bin chopinfer -- eval list --model models/mimo26 --json
   python3 -B -m tools.qual gate --model mimo26 --profile full --plan
   ```

2. Implement `reference.py` following the
   [reference runner](reference-runner.md) adapter contract. Run the f32, bf16
   and f64 variants on idle GPUs (never a full-model reference on CPU) with
   `python3 -B -m tools.refrun run`, then assemble bound fixtures with
   `tools.refrun emit` (`stamps` and `verify-binding` check checkpoint and
   fixture bindings; each has `--help`). Update declaration paths, schemas,
   variant names and digests together. Measure limits against reference
   spread; template bounds are not qualification evidence.

3. Adopt and authenticate the inputs on CPU:

   ```sh
   python3 -B -m tools.qual refs adopt --model mimo26
   python3 -B -m tools.qual refs verify --model mimo26
   cargo run --locked --offline -p runtime-host --bin chopinfer -- eval verify-references --model models/mimo26 --json
   ```

   The first `adopt` creates the lock. To replace inputs later, record the
   current digest from `refs status --model mimo26` as `$OLD_LOCK_DIGEST`,
   update the inputs and declaration, then run
   `refs adopt --model mimo26 --replace-sha256 "$OLD_LOCK_DIGEST"` and
   `refs verify`. Commit the declaration and lock with any small tracked
   fixtures. Never update a pin merely to silence a mismatch.

4. Run the portable accuracy checks, then the full release gate:

   ```sh
   cargo run --locked --offline -p runtime-host --bin chopinfer -- eval run \
     --model models/mimo26 --url "$URL" --suite accuracy \
     --report "$CHOPINFER_DATA/qual/new-model-eval" --json
   python3 -B -m tools.qual gate --model mimo26 --list-inputs
   python3 -B -m tools.qual gate --model mimo26 --profile full --url "$URL" \
     --inputs "$GATE_INPUTS" --receipts "$CHOPINFER_DATA/qual/new-model-release"
   ```

   `GATE_INPUTS` is an external JSON file mapping the plan's runtime
   prerequisites (reference URL, server log and evidence paths) to actual
   inputs; see [gate inputs](../tools/qual/README.md#gate-inputs). The gate
   writes policy, input hashes, per-check evidence and `summary.json`. Require
   `passed: true` and `qualified: true` with complete intended coverage. A
   plan, mechanics run or informational check is never release evidence.

5. With a `preview` section, the
   [correctness runner](../tools/qual/README.md#correctness-runner) serves
   every configuration itself and runs the whole suite against the
   references and the stored baselines:

   ```sh
   bash tools/qual/release_preview.sh prepare --model mimo26
   bash tools/qual/release_preview.sh run --model mimo26 --suite full
   ```

   Keep every qualification fact of the model (configurations, references,
   limits, producers) in its own `models/<model>/` directory; shared tools
   take them from the declaration.

`eval` exits 0 for pass, 1 for a failed predicate, 2 for usage or declaration
errors, 3 for insufficient evidence and 4 for environment errors; the Python
gate exits 2 for configuration errors and 1 for an unsuccessful run (read its
summary). Answer missing evidence with evidence, not looser thresholds, and
localize numerical failures against the pinned reference
([debugging](debugging.md)).

## Checklist

Model directory (beyond the [directory contract](#directory-contract)):

- [ ] Config, catalog and tokenizer checked against the checkpoint
- [ ] Frontend goldens: renders, decode, tool calls, capture replay; frontend vectors pass `chopinfer bundle check-frontend` on the built package
- [ ] Chat format ported in `contract/src/frontend.rs`
- [ ] In-tree: the committed corpus passes as raw requests and `frontend_corpus check` passes the coverage and divergence gate
- [ ] The bundle exports its chat frontend (`chop.frontend.v1`) and every topology requires it
- [ ] Test pinning IDs and geometry; fake-host launch journal per topology
- [ ] Fake-host conformance and serving startup adapters
- [ ] Shared bundle steps (`transaction`, `acquire::rank_images`, `drafter`, `fake_host::backend`) and declared admission (`engine_domain`, `quant_method_default`) instead of model copies
- [ ] Ops prepared through `Backend` at instance creation, with the `Need` the model relies on
- [ ] SM120 selection only in `sm120` modules, with no guard list ([the seam](#the-seam-definition-of-done)); own kernels behind `OwnOps` with contracts and CPU oracles
- [ ] Correctness cases and a reviewed `kernels.lock`, created and updated with `relock` out of tree
- [ ] `profiles.json` and, for the image, `deploy/policy.json` emitted into every package
- [ ] `contract/projections.json` with a pinned digest, and the bundle's projection adoption and schedule ledgers if it keeps locked symbols or overrides schedules
- [ ] Taps captured through `runtime_bundle_sdk::taps`; bound reference fixtures
- [ ] In-tree: `model.toml` `checks.profile_targets` declares profile-build packages and targets

Qualification (GPU):

- [ ] Correctness and benchmark case per new op: the model's own kernels in its test crate on `kernel-harness`, a shared kernel in kernel-lab
- [ ] Per-layer activations within the reference spread
- [ ] Teacher-forced log-probabilities within the gate limits
- [ ] Fixed-batch replay for exactness claims (`tools/dev/fixed-batch-logits.sh`)
- [ ] Portable `chopinfer eval` checks and the full release gate pass

Shared changes, only when needed ([what still touches shared code](#what-still-touches-shared-code)):

- [ ] New shared operation contracts or builders only when another consumer needs them
- [ ] A model-neutral media capability the host lacks
- [ ] A tokenizer component the host lacks: a policy field in `runtime-frontend-contract` and its reader in `runtime-tokenizer`
- [ ] A completion syntax neither shared parser family reads: a new named family in `frontend-format` with its tool-choice grammar, qualification re-parser and tests

An in-tree model also walks [in-tree maintenance](#in-tree-maintenance).

The profile-free RMSNorm and native CUDA paths, prepared through `Backend`
and `OwnOps`, are demonstrated in the external example's `sm120` module
(`examples/external-model/bundle/src/sm120/`); see
[model-owned selections](kernel-library.md#model-owned-selections).

A shared change leaves other models' journals and goldens and every other
bundle's `kernels.lock` unchanged; confirm it with `bash ci/cpu.sh` and the
[journal proofs](../tools/journal_proof/README.md).
