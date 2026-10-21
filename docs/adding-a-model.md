# Adding a model

The model owns its compiler, contract, native bundle, frontend and
qualification. The host owns transport, scheduling and device services.
Read [BUNDLE_ARCHITECTURE.md](../BUNDLE_ARCHITECTURE.md) for the boundary.

## Directory contract

An in-tree model lives under `models/<model>/{compiler,contract,bundle}`;
an independent model uses the same roles against a pinned SDK checkout.
Start with the executable [external example](../examples/external-model/README.md).
It contains the export, lock, build, package and exercise commands.

`model.toml` declares Cargo role targets, optional release selection,
contributor fixtures and check inputs. Its schema is validated by
[the inventory reader](../tools/model_sdk/inventory.py). A Cargo target is
shipped only when explicitly selected by an enabled release declaration.
`publisher` restricts checkpoint names committed by the model to its author;
other checkpoints use [external qualification variants](../tools/qual/README.md#external-variants).

Keep geometry, projections, encoders, tuning, frontend assets, serving
profiles, deployment policy, kernel locks and qualification declarations
with the model. Use existing parsers and emitters instead of copying schemas
into a new guide. Small test inputs belong in fixtures; weights and captured
evidence stay outside the tree.

## External SDK workflow

Pin the SDK's full revision and content identity in `sdk-pin.json`, reconcile
and commit Cargo.lock, and keep a real nested `sdk/` checkout. The
`tools.model_sdk` workflow checks role dependency closures, locks, exports,
compatibility matrices and test execution. A model's kernel tests use
`kernel-harness` in its own test crate; new models do not register in
`kernel-lab`. The [example](../examples/external-model/README.md) is the
command reference, including intentional kernel relocking.

Compatibility has four parts: Rust SDK source, native ABI, package/frontend
schemas and host capabilities. Pin and test all four. The ABI is a draft;
matching C symbols alone does not promise compatibility across releases.
Test independently built bundles against the intended prebuilt hosts.
Bundles cannot link `runtime-cuda`, `runtime-host`, `runtime-engine` or
`runtime-server`. Shared runtime code cannot depend on models or bundles.

## Model implementation

The compiler implements `model_compiler_sdk::preparation::helper::Provider`:
`probe` routes metadata and `emit_direct` validates the checkpoint and emits
a canonical package with planned weight components. The installed
`chopinfer-model-<model>` helper is discovered beside the host executable.
Package emission takes an explicitly built native module; it does not build
one. Each model has one bundle ID and one package containing every supported
rank topology and its resources.

Declare tensor names, shapes, split rules and precision in the model's
[projection specification](../crates/projection-spec/README.md). Derive
rank-local views from the checkpoint, preserving scale-group boundaries and
source scale origins. Define new weight formats as shared typed contracts
when reusable. Layout admission, payload validation and quality qualification
are separate checks. A format supported by an engine does not qualify every
activation policy or checkpoint that uses it.

## Kernels through the backend

Generic model code requests math through `&dyn Backend` and model-owned
`&dyn OwnOps`. Target modules register compiled entries and prepare them at
instance creation. They do not select symbols in the execution loop.
Selection belongs to the backend; request the required precision and
invariance in the contract and `Need`. Unsupported operations refuse at
prepare. Model-only operations keep their contracts, oracles and kernels
with that model, moving to shared code when another model needs them.

Follow [kernel contribution rules](kernel-library.md#adding-an-op). Required
conformance cases must prepare and run, with no refused or unexercised case
counted as a pass. Verify-row equality needs a decode mapping; collectives
need multi-rank cases. Preserve qualified symbols and lock bytes during a
structural refactor. Geometry support is separate from measured tuning and
from qualification.

## Chat frontend

The bundle exports `chop.frontend.v1`. Implement its renderer in model-owned
Rust with `frontend-format`, or implement the native frontend contract.
The host authenticates the declaration and tokenizer, checks every frontend
answer, tokenizes under declared policies and owns stop/UTF-8 handling.
Checkpoint chat templates are reference inputs, not executed server code.

Port the upstream template or encoder literally, citing source lines.
Keep HF render, tokenization and streaming vectors as goldens and run
`chopinfer bundle check-frontend` on the built package. Include reasoning,
tools, role handling, media and every byte split of streaming output.
Use the HF differential corpus to distinguish real divergence from
intentionally unsupported or lossy template behavior. A `no-round-trip`
declaration must identify the actual union, delimiter or tokenizer loss;
the renderer must not invent escaping absent from the source template.

The tokenizer reader refuses unimplemented policies. A missing shared
tokenizer or media capability needs a model-neutral contract and tests.
A completion syntax absent from `frontend-format` needs a reusable parser
family, grammar and qualification/corpus support. Model-specific behavior
belongs in the model, not a flag added to a shared family.

### Updating the chat template

1. Update the revision and template digest pin in `frontend_cases.py`.
2. Generate the corpus with `python3 -m tools.refrun.frontend_corpus generate --model MODEL`.
3. Inspect changed cases and coverage with the Rust corpus test using `CHOPINFER_FRONTEND_CORPUS_DIR`.
4. Port changed lines in the model renderer and update justified divergence declarations.
5. Run corpus `minimize` into the model's frontend fixtures, then `check`.

External models regenerate vectors from their pinned reference and rerun
`bundle check-frontend`. Never regenerate a golden merely to hide a mismatch.

## Qualification

Declare bounded contributor fixtures and exact test selectors in
`model.toml`; run [contributor checks](contributing.md#first-success-without-weights-or-cuda).
Use `qual.json`, the reference lock, `profiles.json` and deployment policy
for operational settings. [Qualification tools](../tools/qual/README.md)
explain the runner and declaration semantics; [check kinds](check-kinds.md)
and [numerics](numerics.md) own the acceptance contracts. Add a model-owned
reference adapter and authenticate its outputs through
[the reference runner](../tools/refrun/README.md).

Do not enable a release or mark a profile qualified based on CPU fixtures,
successful packaging or a skipped GPU run. Record real-device oracle,
resource, lifecycle, frontend and serving evidence for the actual artifacts,
checkpoint, topology and settings. Keep model-owned exceptions and their
rationale with the qualification declaration or its cited contract.
