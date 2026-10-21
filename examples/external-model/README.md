# Independent model workspace

This maintained example has three roles: `bundle` implements a bounded
synthetic state machine, `compiler` packages a checkpoint with an already
built native module, and `contract` owns their IDs and layout. It has no core
model registration. Execution uses fake host services and proves boundary
mechanics, not model quality. Compatibility rules and release policy are in
the SDK's `docs/adding-a-model.md#external-sdk-workflow`.

The first exercise builds a small Rust bundle and compiler, then packages a
seed as authenticated bundle metadata. The bundle holds no weights; a model
with weights plans them from its checkpoint (see the SDK's
`BUNDLE_ARCHITECTURE.md#direct-weight-loading`). It needs Rust 1.89.0, Python 3.11 or newer, Git, a
C linker and a preseeded Cargo registry cache, and no `nvcc`, model weights,
host binary or GPU.

## Export, build and package

A new model starts from this template. `export --model models/<m>` exists
only to move an in-tree model out of the SDK's repository; the SDK's
`docs/adding-a-model.md#external-sdk-workflow` states its limits.

Export this template before building. The exporter places one complete pinned
SDK under the ignored `sdk/` directory and creates `sdk-pin.json`; `lock`
creates `Cargo.lock`. Commit both in your model repository. From a clean SDK
checkout, set these paths to unrelated directories:

```sh
export SDK_SOURCE=/path/to/clean/sdk
export MODEL_SOURCE=/path/to/new/model
export MODEL_OUTPUT=/path/to/generated/artifacts
export SDK_REVISION=$(git -C "$SDK_SOURCE" rev-parse HEAD)
export CARGO_NET_OFFLINE=true
export PYTHONPATH="$SDK_SOURCE"
python3 -B -m tools.model_sdk export --source "$SDK_SOURCE" --output "$MODEL_SOURCE" --revision "$SDK_REVISION"
python3 -B -m tools.model_sdk lock --workspace "$MODEL_SOURCE"
python3 -B -m tools.model_sdk check --workspace "$MODEL_SOURCE" --release
python3 -B -m tools.model_sdk build --workspace "$MODEL_SOURCE" --target "$MODEL_OUTPUT/target" --logs "$MODEL_OUTPUT/build" --release-pin
python3 -B -m tools.model_sdk package --workspace "$MODEL_SOURCE" --compiler "$MODEL_OUTPUT/target/debug/example-compile" --module "$MODEL_OUTPUT/target/debug/libexample_bundle.so" --checkpoint "$MODEL_SOURCE/fixtures/checkpoint.json" --output "$MODEL_OUTPUT/package"
```

The package command prints `example.stateful.tp1.v1`. Canonical emission
verifies the package before publishing it. The build receipt records
`nvcc: null` and `cuda_build_owners: []` for this path.

`lock` reconciles the SDK's lock once and refuses to replace an existing
one. After changing the model's own dependencies, `lock --update` resolves
the existing lock again offline, keeping the SDK's pins, and reports the
packages added, removed or changed. Every later Cargo command is locked and
offline, and missing cache entries are setup failures.

For a first edit, change the exported model's small input and package it into
a new directory. The SDK checkout under `sdk/` stays read-only:

```sh
printf '{"seed":8}\n' > "$MODEL_SOURCE/fixtures/checkpoint.json"
python3 -B -m tools.model_sdk package --workspace "$MODEL_SOURCE" --compiler "$MODEL_OUTPUT/target/debug/example-compile" --module "$MODEL_OUTPUT/target/debug/libexample_bundle.so" --checkpoint "$MODEL_SOURCE/fixtures/checkpoint.json" --output "$MODEL_OUTPUT/package-8"
python3 -c 'import pathlib, sys; print(int.from_bytes(pathlib.Path(sys.argv[1]).read_bytes(), "little"))' "$MODEL_OUTPUT/package-8/checkpoint.bin"
```

The last command prints `8`: the package ID stays the same and the
authenticated checkpoint bytes change. Set `seed` to a string and repeat with
another new output directory to see `checkpoint requires unsigned seed` and a
nonzero exit.

When editing the core SDK, use `bash ci/quick.sh` while iterating and
`bash ci/cpu.sh` for the full checks; example success does not replace them.

## Serve the package

Point at explicitly selected existing host binaries. `exercise` verifies and
serves the package over HTTP and exercises commit, cancel/replay, cleanup and
retirement. It checks both host hashes and never rebuilds a host:

```sh
python3 -B -m tools.model_sdk exercise --host "$CHOPINFER_HOST" --conformance-host "$CHOPINFER_CONFORMANCE_HOST" --package "$MODEL_OUTPUT/package" --output "$MODEL_OUTPUT/host-check"
```

The equivalent deployment commands are:

```sh
"$CHOPINFER_HOST" bundle verify --bundle "$MODEL_OUTPUT/package"
"$CHOPINFER_HOST" serve --bundle "$MODEL_OUTPUT/package" --topology tp1 --fake-host-services --max-sequences 4 --max-context-tokens 1024 --prefill-chunk-tokens 512 --kv-pages 512 --listen 127.0.0.1:8101
```

The synthetic model's logit rows are +1 for one token and -1 for every
other, so default sampling picks other tokens too. Send `"temperature": 0`
in chat and completion requests for output that follows the model.

While serving, run the model-owned mechanics declaration through the shared
gate:

```sh
export CHOPINFER_DATA="$MODEL_OUTPUT/evidence"
export CHOPINFER_EXAMPLE_PACKAGE="$MODEL_OUTPUT/package"
python3 -B "$SDK_SOURCE/tools/qual/release_gate.py" --config "$MODEL_SOURCE/qualification.json" --model-root "$MODEL_SOURCE" --url http://127.0.0.1:8101 --profile quick
```

The declaration sets `qualification.mechanics_only: true`, so a successful run
reports `passed: true`, `mechanics_only: true`, `qualified: false`. A real
architecture must add numerical and service qualification.

`python3 -B -m tools.model_sdk.smoke --help` describes the full maintained
demonstration: admission refusals, two releases with the same ID, rollback
and the qualification declaration. Use `--release-pin` only with a clean SDK
revision; development pins are never release evidence.

## Kernels through the backend interface (`cuda-probe`)

The example reaches kernels the way every model does: it states its ops as
contracts and prepares them through `&dyn kernel_core::backend::Backend`,
and its SM120 code lives in one `sm120` module, which nothing else names.

- `contract/src/ops.rs` holds the math: a plain BF16 RMSNorm at width 1280
  and capacity 4 (a kernel-core requirement that leaves the reduction order
  to the backend), and the example's own sigmoid probe, a contract with a
  role record, a CPU oracle and a method of the example's `OwnOps` trait.
  `Ops::prepare` prepares both on a backend or names the first refusal.
- `bundle/src/sm120/` is the SM120 module, mostly data. `selection.rs`
  selects the shared RMSNorm builder and the local `bundle/cuda/probe.cuh`
  kernel for the build and registers the RMSNorm entry on `Sm120`;
  `own_ops.rs` implements `OwnOps` over the probe entry; `mod.rs` supplies
  both as the generic `Kernels` (`Kernels::new`). The build script,
  `build.rs`, also lives there, since it only compiles and locks SM120
  cubins.
- With `cuda-probe`, instance creation prepares the ops and refuses an
  instance whose kernels cannot serve them. The synthetic serving program
  launches neither.

`--feature cuda-probe` compiles both entries ahead of time under
`bundle/kernels.lock` and needs CUDA 13.2 with `nvcc` 13.2.86. Neither entry
is added to a central `Profile`. `--feature sm120-helpers` exercises the
shared Rust helpers without CUDA. The `omit-retirement` feature builds a
negative artifact that cannot serve a package declaring request retirement.

`bundle/kernel_cases.rs` owns inputs, live-row cases, f64 references and
bounds; `bundle/tests/sm120/` checks the locked artifacts, the refusals and
that the prepared ops launch the selected entries. The example's own test
crate, `lab`, runs them on a GPU through the SDK's model-free kernel harness
(`kernel-harness`); nothing registers in the SDK's kernel-lab. It is not a
default workspace member, since it links the CUDA driver. Set
`CHOPINFER_MODEL_SELECTION_REPORT` to the absolute path of the exported
build's `kernel-build-report.json`, then from the model workspace run:

```sh
cargo test --manifest-path "$MODEL_SOURCE/Cargo.toml" -p example-lab --test selection --locked --offline -- --nocapture
```

This fails if the report is absent or stale. It launches the ops as the
bundle prepares them. Without an idle GPU assigned through
`CHOPINFER_KERNEL_GPU`, it performs CPU declaration and artifact checks and
reports the GPU oracle, padding and graph cases as unrun.

## Quantized checkpoint consumer (`precision-exercise`)

The optional `precision-exercise` feature is a small independent
architecture (a projection, four top-2 experts with fused gate/up, and an
embedding table) that consumes ModelOpt NVFP4 checkpoints through the shared
SDK importers and typed kernel constructors. Activation policies: `auto` uses
supported checkpoint recipes and reports fallbacks, `checkpoint` refuses
anything that cannot keep the source recipe, and `bf16` binds no activation
quantizer; the parameter overrides `CHOPINFER_ACTIVATION_POLICY`.

The architecture is declared as data in `contract/projections.json`, the same
schema the in-tree models use (see the SDK's
`crates/projection-spec/README.md`). The declaration names each projection's
checkpoint tensor, its `[rows, k]` per operand, its kind (dense, expert bank
with fused `gate` and `up` operands, lookup table), its format, epilogues and
its `served` use profile. `contract/src/precision.rs` parses it with an empty
variant registry and supplies the geometry values it names; a test pins its
digest.

- The compiler admits each declared source through the SDK importer
  (`import_operand`), then binds the checkpoint against the declaration
  (`Declaration::bind` with `quantization::declared::AdmittedRules`). Binding
  claims every tensor for a declared projection, static input scales
  included, so an undeclared tensor, a missing companion, or a shape or
  format the declaration does not admit is refused, naming the tensor.
- `quantization::declared::imported` builds each projection's import, the
  stacked expert bank included, and `projection_spec::derive::derive` derives
  the rank's operations. The compiler keeps the source shape for the matrix
  and table operations (`GeometryMode::SourceShape`), which is why it calls
  these two steps itself rather than `declared::derive`, applies the
  checkpoint's activation recipes (`declared::admitted_activations`) and
  names each operation by its projection. A compiler test executes the
  derived operations' records over the checkpoint and compares them with the
  bytes the example's packing writes.
- The bundle's build generates its kernel offers from the same declaration
  with `kernels_sm120::precision::generated::{domain, offer}`, choosing only
  the activation variants and schedules (`bundle/src/sm120/precision_selection.rs`).
  `bundle/projection-adoptions.json` maps the generated identities to the
  symbol names `bundle/precision.lock` records, so the locked cubins keep
  their names and bytes. The ledger is append only
  (`tests/projection-adoptions.py` in the SDK checks it).
- `bundle/precision_program.rs` prepares each operation of a package once,
  at each of its uses, through `&dyn Backend`
  (`Backend::projection`), and refuses a package whose recorded
  implementation the backend does not bind. The `sm120` module supplies
  `Sm120` over the offers and the entries the module embeds
  (`PreparedExercise::new`).

The weight plan (`contract/src/weight_plan.rs`) still records its own
packing per operation and plans one rank.

The compiler prepares direct packages:

```sh
example-compile prepare --direct --model DIR --tensor-parallel-size 1 \
  [--cache-dir DIR] [--activation-policy auto|checkpoint|bf16]
```

It reads only metadata and safetensors headers and publishes the package in
the cache as `p-<manifest sha256>`, reusing it for the same inputs. The host
packs the weights by the bundle's plan (`contract/src/weight_plan.rs`) at
load. The helper packages the `libexample_bundle.so` installed beside it or
in `../lib`; install the module built with `precision-exercise` there.

Build and test the feature with all lock overrides unset:

```sh
python3 -B -m tools.model_sdk build --workspace "$MODEL_SOURCE" \
  --target "$MODEL_TARGET" --logs "$MODEL_LOGS" --feature precision-exercise
python3 -B examples/external-model/tests/precision_preparation.py \
  --compiler "$MODEL_TARGET/debug/example-compile" \
  --module "$MODEL_TARGET/debug/libexample_bundle.so" \
  --work "$CHOPINFER_DATA/precision-preparation"
```

The build's CPU test executes the weight plan over synthetic checkpoints and
compares every section with the SDK's prepared-weight builder. The
preparation test checks policies, refusals and package reuse, and writes
`packages.json` for the GPU adapter. Then set
`CHOPINFER_PRECISION_SELECTION_REPORT` to the build's
`out/precision/kernel-build-report.json` and `CHOPINFER_PRECISION_PACKAGES` to
that `packages.json`, and from the model workspace run:

```sh
cargo test --manifest-path "$MODEL_SOURCE/Cargo.toml" -p example-lab --test precision \
  --locked --offline -- --nocapture
```

The adapter packs each package's weights through the host's direct-load
executor, launches the prepared operations and checks them against exact
codes, the f64 oracle and bitwise lookups across graph replays. Assign an idle GPU through
`CHOPINFER_KERNEL_GPU`; without one the GPU cases are reported as unrun.

## Updating the model's kernel locks

`bundle/kernels.lock` and `bundle/precision.lock` are model-owned: the build
script locks them through `kernel_build::ModelLock`. Ordinary SDK builds
verify them and refuse lock-update and profiling bypasses, and a pinned
consumer must not repair an incompatible SDK by replacing a lock. After a
reviewed change to the model's kernel sources or its SDK revision, create or
update both with `KERNELS_LOCK_UPDATE` and `CHOPINFER_KERNEL_LINEINFO` unset:

```sh
python3 -B -m tools.model_sdk relock --workspace "$MODEL_SOURCE" \
  --target "$MODEL_TARGET" --logs "$MODEL_LOGS/relock" \
  --feature cuda-probe --feature precision-exercise
```

`relock` builds once with each model lock redirected to a candidate, refuses
any lock outside the model's own tree (SDK locks are never rewritten),
replaces only changed lock files and rebuilds against them. A failure
restores the old locks; the SDK's `docs/adding-a-model.md#external-sdk-workflow` describes its
recovery state. Its `relock-report.json` lists each lock's added, removed and
changed records. A changed record names its fields and is `cubin` when its cubin bytes changed,
`recipe` when another gating field changed, or `provenance` when only the
source closure digest changed (source paths or the SDK revision moved with
identical cubins). Explain every change in review; a later `build` must
report `Verified`, which releases require. Never relock merely to silence a
failing smoke.

For a reviewed shared-source change, a core maintainer relocks a disposable
export of the candidate SDK and copies the reviewed locks back into
`examples/external-model/bundle/`.

The default CPU CI smoke also builds and tests `precision-exercise` with its
lock verified, then runs the build-tool-free preparation checks above.
