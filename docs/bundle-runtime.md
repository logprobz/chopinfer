# Shared bundle runtime

`crates/bundle-runtime` holds the model- and target-neutral mechanics
of native bundles. A bundle lowers its own program into prepared launches;
this crate binds them to host buffers, issues them eagerly or as cached
graphs, owns the mixed-step input format and its device layout, and runs the
shared create, snapshot and encoder machinery. Despite the similar name,
`runtime-bundle` is another crate: this one runs inside a bundle's native
module, while `runtime-bundle` is the host's parser and verifier of bundle
package directories. Both native bundles,
[bundle-qwen38](bundle-qwen38.md) and `bundle-mimo26`, use it. For
independent model workspaces, see [external models](external-models.md).

## Boundary

The crate depends on `runtime-bundle-abi`, `runtime-bundle-sdk`,
`kernel-core`, `encoder-spec` and `serde_json`. It names no kernel set: a
bundle prepares its ops on a `kernel_core::backend::Backend` and supplies
them (`projection::Recipe::Prepared`, encoder slots), and its target-specific
code (an SM120 module in each model) states the rest, such as the device a
fake host models. `kernels-sm120` is a dev-dependency only, for tests that
compare the backend path with SM120 offers and catalogs; the boundary
guard refuses it in production source (any path or alias) and in the
crate's normal or build dependencies. It has no dependency on a `model-*` or
other `bundle-*` crate, or on the host crates (`runtime-cuda`,
`runtime-host`, `runtime-engine`, `runtime-server`).
`tests/bundle-boundary.sh` enforces these ownership boundaries. Shapes,
tensor names and entry symbols come from callers. `dispatch` translates
launch plans and tensor-map contracts into the host ABI.

`runtime-bundle-sdk` is the layer below: wire reading, host errors, stream
transactions, rank indexes, taps and diagnostics.

## The seams a bundle implements

| Seam | What the bundle supplies |
|---|---|
| `BundleGeometry` (`geometry`) | Associated constants: `PAGE_TOKENS`, `PREFILL_TOKENS`, `DECODE_BATCH` (package maximum of items, a power of two up to `MAX_ITEMS`, 256), `MAX_VERIFY_ROWS`, `TOKEN_GRANULE`, and optionally `VERIFY_TOKENS` and `NARROW_VERIFY_TOKENS`. |
| `ModelLimits` (`geometry`) | Validated vocabulary and maximum context, with a fixed page size and block-table row stride. `ModelLimits::new` refuses zero dimensions, partial pages and page counts above `u32::MAX`. Bundles currently pass compile-time contract limits; the create record's served context is a separate limit, checked against the model maximum by `CreateConfig::validate_context`. |
| `entry::Bundle` | The instance behind the ABI entry bodies: its thread-local instance table, `create`, `resources`, `release_graphs`, `execute`, `commit`, `commit_accepted`, `cancel`, `retire_request`. |
| `table::Live` | Access to the instance's `pipeline::Collectives`, which a failed submission poisons. |
| `prefix::PrefixInstance` | `state_pages`, `capture` and `restore` for prefix-cache snapshots. |
| `snapshot::HostRankStreams` (or `RankStreams`) | Borrowed host services and rank streams, plus launch and error conversion closures for snapshot copies. |
| `encoder::Registered` | Rotary rows and stage launches of the bundle's model-namespaced encoder variants (`encoder::Shared` when it has none). |
| `state` declarations | `RequestState`: per-request `StateBuffer`s and `PagedLayers` of `KvStream`s for one instance configuration. |
| A program | Lists of `pipeline::PreparedLaunch` (symbol, launch plan, arguments, buffer bindings). There is no program trait: each bundle writes its own stage enum, selection and lowering and hands launch lists to `pipeline` and `graphs`. |

A bundle exports `entry::create::<Instance>` and the other bodies through
`runtime_bundle_abi::export_bundle_api_v1!` (see
`models/qwen38/bundle/src/lib.rs`).

## Rank and item counts

The rank count is a runtime `Ranks`, one of `SUPPORTED_RANKS` (1, 2, 4, 8),
read from the create record; `MAX_RANKS` (8, the comm kernels'
`MAX_COMM_RANKS`) bounds per-rank arrays. One native module serves every rank
count its bundle lists.

The served item count is a runtime `Items`, from 1 to `DECODE_BATCH`.
`config::decode::<G>` reads it from the create record's last field (sent only
to packages that require `chop.served-items.v1`, filled from
`--max-sequences`); a record without it serves `DECODE_BATCH`. The shared
code sizes by it through the `_within` forms (`InputLayout::decode_within`,
`PackedStep::pack_within`, `VerifyLimits::enabled_within`,
`GraphCache::with_items`): slot tables and block-table rows hold `Items::get`
entries, and per-row arrays a graph reads hold `Items::capacity`, the count
rounded up to a power of two (the largest decode bucket).

With `chop.state-per-item.v1`, the manifest's state layout declares bytes per
state slot and the host allocates `bytes * state_slots`, with `state_slots`
equal to `Items::capacity` (`runtime_bundle_abi::raw_state_slots_v1`).

## Modules

Execution:

- `dispatch`: buffer bindings, launch configurations and tensor-map
  translation into the host ABI.
- `pipeline`: prepared launches, eager and traced issue, and graph-or-eager
  rank runs. `launch_all` resolves a whole launch list on the host before
  submitting any of it, and `run_ranks` lowers every rank's eager program or
  captures every rank's missing graphs before launching any rank. Collectives
  cannot be aborted, so a failure after submission began (through
  `pipeline::Unit`, or a panic inside `run_ranks`) poisons the instance's
  `Collectives`: it refuses every later call, destroy included, and keeps its
  resources for process-level recovery.
- `graphs`: `GraphKey { rank, bucket, items }` and `GraphCache<G>`, the
  per-rank decode and verify graph cache. `decode_bucket::<G>` rounds a live
  count up to a power-of-two bucket, `pieces` splits a program into capture
  pieces (a short first piece so every rank starts early), and
  `replay_piece_major` replays every rank's pieces piece-major. A cache
  releases its graphs newest first in capture order; extra caches built with
  `GraphCache::sharing` share that ledger. `release_all` releases the whole
  ledger; `release_if_newest` releases one cache's graphs alone while they are
  the ledger's newest, so a cache of evictable programs captured after the
  decode graphs releases without them.
- `lanes`: a two-lane (compute and comm) list scheduler for pipelined
  prefill. The bundle groups its stages into a `Program` of whole-chunk
  stages, a per-part prologue, blocks and an epilogue.
- `phase`: the step phase (`Decode`, `Verify`, `Prefill`), its token
  capacity at the geometry's buckets, and the prefill row-grid policy. Entry
  selection per phase stays with the bundle.
- `residency`: checks that spinning comm grids can be resident
  (`query_cluster_residency_v1`).
- `recorded` and `submission`: immutable device programs whose uploads are
  staged in bounded upload windows before compute, submitted to independent
  ranks round-robin, with every rank that may have queued work drained on
  failure.

Inputs and outputs:

- `inputs`: mixed-step validation and packing (`PackedStep`), verify limits
  (`VerifyLimits`) and the prefill split. Slot lookup is a closure; Qwen
  passes the batch's `state_slots()`.
- `upload`: one packed input arena per rank and phase (`InputLayout`) and the
  resident block tables (`BlockTableMirror`), which upload only changed
  entries.
- `logits`: device logits views in the original item order, one row per
  verify input; `logits::exchange` distributes vocabulary blocks across ranks.
- `pages`: the layer-interleaved view of the host-owned serving page pool
  ([serving KV pool](serving-kv-pool.md)).
- `kv`: serving KV element formats. A bundle lists each layout as a
  `KvFormat` (layout ID, page bytes, BF16 or FP8 E4M3, per-layer K and V
  scales) and `select`s the one the host declared from `--kv-cache-dtype`.
- `media`: the batch wire's media section validated against the bundle's
  arena (`MediaArena`), the prefill rows each inject overrides, and the page
  runs an encoder writes through.

Create and lifetime:

- `config`: the `chopbundle-config-v1` create-record reader. It checks the
  topology against the bundle's `Topology { id, ranks }` list and the item
  count against the package maximum, and returns the rest raw for the bundle
  to check (layout IDs, page geometry, component roles).
- `acquire`: the steps of a create, each recording what it acquires in the
  caller's `Held` list: authenticated component reads, rank images
  (`rank_images` reads every rank index, admits it with the bundle's checks
  and binds it to the identity the host serves for that rank's image,
  `chop.compiled-image-identity.v1/<image role>/<rank>`, before the image is
  opened), rank contexts, image upload, view placement, peer grants, module loading, zeroed request state,
  the serving KV pool and the closing synchronize. Views are placed in the
  device arena, in pinned host memory (`chop.host-resident-views.v1`), or as
  this rank's rows of a row-sharded table; `place_views_mapped` also cuts
  out the views a plan maps from the checkpoint
  (`chop.host-mapped-gather.v1`), whose rows the bundle copies to its own
  buffers with `gather_mapped_rows` instead. The bundle chooses the order.
- `scratch`: `bind` maps a model's `Traversal<Allocation>` to typed host
  spans in declaration order, stopping at the first error. Declare the record
  once with `kernel_core::scratch_record!`; arrays, optional leaves and nested
  records share its `try_map`, `visit` and `visit_named` traversal. `Allocation`
  separates physical bytes from visible span bytes. Binding retains handles
  and base offsets, checks backing ranges and applies visible lengths. The
  caller records allocations in `Held` for cleanup. Size formulas, admission
  policy, aliases and initialization remain model-owned; binding does not pack
  an arena or infer them.
- `resources` (`Held`, reverse release), `table` (generation-checked instance
  handles; `usable` refuses stale, re-entered and poisoned instances),
  `entry` (the ABI entry bodies), `index` (rank index parsing and identity
  authentication; `index::encoder` writes and parses a BF16 encoder image's
  index against the model's layout), `startup` (`startup <stage>: <seconds>s` timing lines).

Request state and snapshots:

- `state`: the declarations. `RequestState::validate` checks the rules every
  model must meet and `validate_extent` one instance's slot, verify-row and
  checkpoint counts. The piece list, checkpoint entries, cancel regions and
  page offsets follow from the declaration.
- `ledger`: one transaction ledger mapping each request to its slot and
  committed tokens (`begin`, `cancel`, `restore`, `retire`). `plan_commit`
  resolves accepted rows into ordered `CommitPlan::items()` without mutation;
  consuming `apply_commit` rejects foreign or stale plans with `StalePlan`
  before mutation. `commit` remains a compatibility wrapper. Slots are
  ledger-placed or host-placed (`Placement`). The bundle keeps its device and
  drafter effects in their existing order and chooses the ledger apply point.
- `transaction`: the order in which a step ends over the ledger, for a bundle
  that implements `Steps`. `commit` refuses a step with verify items (they
  commit only through `commit_accepted`); a commit then runs `begin`,
  refuses when no step is open, plans the kept rows (each record's
  `(request_id, kept_rows)`), runs `prepare` with the plan, applies it, runs
  `ended` with how each item ended (`ring::tails` of them are the rejected
  tails) and returns `finish_taps(true)`. `cancel` runs `begin`, the open
  check, `restore` with the open step, the ledger cancel, `ended` and
  `finish_taps(false)`. `retire` refuses during any open step under
  `Retire::DuringNoStep` (the ledger itself refuses only a step that holds
  the request), then runs `before_retire`, the ledger retire and `retired`.
  The first failing hook or ledger call ends the call; hooks return their own
  status, and `Steps::status` maps each refusal (`Failure`, with the ledger
  `Stage` that refused) to the status the bundle returns.
- `drafter`: MTP drafter bookkeeping for a drafter whose rows run beside
  the target's (`Drafter<P>`): which rows each step runs (catch-up, first
  draft, chained drafts, a prefill chunk's completed rows), where each row's
  hidden row and token come from, and the retained-row copies commit makes.
  `Params` names the slot, verify-row, retained-row and run-row counts and
  the token the row at position 0 embeds. The device runs and copies stay
  with the bundle.
- `entries`: optional `EntryMap` ownership of slot and checkpoint workspace
  entries, with inverse lookup and duplicate-choice rejection. `prepare`
  produces checked `Exchanges`; `commit_pair` checks both maps for staleness
  before applying either. Checkpoint selection, initialization and device
  rollback remain model-owned.
- `ring`: `WindowRing`, a declared sliding-window ring's cell addressing, the
  cells a step touches (shadow save) and the cells of rejected tails.
- `snapshot`: `Table` derives one descriptor per declared piece of a slot or
  anchor holder; `Pages` checks a snapshot's page list and splits ranges at
  page boundaries. Callers construct and validate every rank's `Action`s,
  including spans and launch bindings, before submission. `copy_prefix` checks
  the rank-plan count and maximum read length, submits through `RankStreams`
  and drains every rank that received work, with drain failure taking precedence. `HostRankStreams`
  supplies the host-backed adapter using borrowed services/streams and caller
  launch/error closures. Actions, headers and poisoning policy remain with
  the bundle. A bundle copies pieces with device copies
  (`Table::actions`) or with its own copy kernel that mirrors the table.
- `prefix`: snapshot headers (`HeaderLayout`, an 8-byte magic plus 64-bit
  words; `HeaderLayout::V1` is the shared layout, while MiMo and Qwen keep
  their own legacy layouts so their snapshot bytes do not change),
  `layout_digest`, `may_restore` (the drafter-frontier and owner rule) and
  the prefix-state entry bodies.

Encoders (`encoder`):

The shared encoder runtime takes a bundle's `encoder-spec` declarations, the
views of each rank's encoder image and the catalog entry per declared slot.
At create it builds the buffer plan, allocates every rank's regions and
lowers each declaration to one canonical launch list (`encoder::lower`). The
declaration's precision policy (`f32-pairs.v1`; the retired `bf16-legacy.v1`
is refused by the loader) fixes the arithmetic. Per media record it admits the record against the declaration's
recipes, computes side inputs on the host (`encoder::side`), records one
program against rank 0 and submits it relocated to every rank (encoders are
replicated). A device failure drains every stream before the step fails; a
stream that does not drain poisons the instance. Qwen3.8 runs its vision
tower on it, MiMo its vision tower and audio encoder.

| Variable | Effect |
|---|---|
| `CHOPINFER_ENCODER_SERIAL=1` | Each rank records and runs its own program, one rank after another. |
| `CHOPINFER_ENCODER_PARALLEL` | Parallel policy; only `replicated` is accepted. |
| `CHOPINFER_ENCODER_CAPTURE_DIR` | Rank 0 writes immutable `step-<n>/` captures of encoded and injected rows (`encoder::capture`). A capture's file-system failure disables it and never fails a step. |
| `CHOPINFER_ENCODER_ARENA_DIR` | Every rank's arena rows of every encoded record. |

Testing (feature `test-support`):

- `fake_host`: the core of a bundle's CPU fake host. `FakeDevice` hands out
  handles, records the device journal, tracks streams, events, graph captures
  and queued collectives, answers residency
  (from a caller's `DeviceModel` of a device and its entries' locked
  resources, from fixed reports, or else from the package's
  `FakePackage::device_residency`) and batch-limit queries and holds fault
  switches (`Countdown`) and the
  snapshot record. `journal_digest` and `check_journal` pin journals;
  `fake_host::dump` replays snapshot copies to show where each byte came
  from. A bundle's fake host derefs to `FakeDevice` and keeps its package
  answers and scenarios. `fake_host::conformance` runs shared lifecycle,
  allocation/release, malformed-commit, poisoned-client, cancellation and slot
  reuse scenarios through a model `Adapter`. The adapter supplies requests,
  callbacks, configuration, release accounting, fault hooks, cancellation
  policy and assertions that poisoned clients refuse prefix calls. Keep prefix
  round-trip, preemption and independent snapshot-copy expectations in the
  model suites. These CPU scenarios do not establish numerical correctness or
  device qualification.
- A bundle runs these scenarios only through its one
  `fake_host::conformance::suite!` invocation, which generates one test per
  scenario line and a `serving_startup` module: one test per profile, listed
  in the invocation as `tp4_mtp: "tp4-mtp",` (macros cannot read
  profiles.json), and a `profiles` test that requires the list and the
  adapter's configurations to be exactly the profiles of profiles.json, each
  test named after its profile. `Adapter` names the model's startup adapter
  (`type Startup`), which must be the suite's, and
  tests/test_model_inventory.py requires one invocation in every model's
  bundle crate, so startup cannot be left out and each profile starts once
  per suite in any test runner, including nextest's one process per test,
  where the profiles start in parallel.
- `fake_host::conformance::serving_startup` runs what `serve` does before
  readiness, per profile, with the host's own generators
  (`runtime_bundle_sdk::warmup`, which runtime-host calls too): create with
  the modalities the host serves by default (`auto`, every profile's: text
  and image of those the package provides, `runtime_bundle_sdk::modalities`,
  which runtime-bundle's direct preparation calls too), so the create record
  names the vision encoder's components when the package provides images;
  then `describe_prefix_state` when the profile caches prefixes, then
  - the decode waves (`waves`): each prefills and decodes every sequence;
  - with speculation, one wave per verify key the host warms
    (`verify_shapes`): each request pins its drafts, so the step after the
    prefill decodes the zero-draft requests beside the drafting ones, at every
    reachable (row bucket, item bucket), including shallower draft depths;
  - the context ladder (`context_ladder`): one request per prompt length,
    whose first drafting step at the deepest pin catches the drafter up;
  - when the package limits drafting items below what one step could hold,
    a step with one more drafting item, which the bundle must refuse
    (`InvalidArgument`), so the declared maximum is the bundle's own.

  Every drafting step keeps all its rows (`commit_accepted`); every request
  retires and destroy must then release everything. The host's verify limits
  come from package facts through the host's own rules (`served_verify_rows`,
  `narrows_rows`, `item_rows`). The scenario does not model the tokenizer,
  sampling, repeated decode or verify steps of one warmup request, request
  admission, timeouts or the readiness barrier. Its steps are text only, as
  the host's warmup is: neither encodes media. Its model `Startup` adapter
  provides:
  - `profiles()`: the model's profiles.json, usually
    `include_str!("../../profiles.json")`.
  - `servings()`: one `Serving` per profile. Tensor and context parallel
    size, sequences, drafter, draft tokens, prefix cache and model options
    must equal the profile. The adapter adds the modalities the package
    provides (`provides`: "text", then those its encoders serve), an explicit
    `--modalities` list only where its fake host cannot yet create the
    default package (`selection`, `None` for the host's default), and the
    package's speculation declaration (`PackageSpeculation`:
    maximum draft tokens, bundle drafter, maximum verify tokens, row buckets,
    maximum draft items, the narrow-rows capability), taken from the same
    contract values the package builder uses.
  - `new(&Serving)`: a fake host configured as the served package
    (topology, served count, host-resident views, drafter images) that
    reports the draft tokens and `Serving::verify_rows()` when the profile
    speculates.
  - `create()`, whose record names the components of the encoders
    `Serving::modalities()` needs (with `Serving::serves_images()`, the
    vision encoder's image and rank indexes), `step(&[Item])` (one state slot and enough KV pages per
    item, a decode or verify item of any depth; a bundle that drafts takes the
    bonus token alone), `token(position)` and `assert_released()`, as for
    `Adapter`.

## What stays in the bundle

- Geometry values, the stage enum and order, stage lowering, kernel entry
  selection per phase, launch builders and catalogs.
- Rank buffers (scratch), rank view names and sizes, and the create order.
- Drafters (Qwen's MTP, MiMo's MTP and DFlash): they plan their own launch
  lists over the shared graphs, inputs and upload, and keep their per-slot
  drafter records.
- The bytes behind the state declarations, the snapshot copy kernel and its
  addressing, and the legacy header layout.
- Encoder registered variants and the prefill injection stage.
- MiMo's ring byte layout (`RingGeometry`, checked cell for cell against the
  declared `WindowRing`) and its pipelined-prefill stage grouping (`overlap`).

When a change to shared code must not change a bundle's behavior, the
bundles' fake-host launch journals (for example
`models/mimo26/bundle/src/launch_journal_tests.rs`) and snapshot dumps
(`models/qwen38/bundle/src/fake_host_tests/snapshot_dump.rs`) pin every host
call and snapshot byte.

## Two-lane prefill execution

`bundle_runtime::lanes` schedules token parts over compute and
communication lanes. The bundle groups its stages into whole-chunk setup,
per-part prologue, ordered lane blocks and whole-chunk epilogue. Events
order each part across lane changes; compute-stream issue order carries
state from an earlier part to a later part.

`Settings` parses bundle-supplied environment names and defaults, rejecting
zero SM or communication cluster counts. `Config` retains the SM count.
The bundle classifies launches as persistent compute or phased communication.
`adjust_grid` uses the saved count, clamps reservations to leave at least one
SM in a persistent compute grid, and applies the communication CTA override.
Symbol names and numerical eligibility remain bundle-owned.

`Executor` adapts a stage, part and lane to bundle lowering and submission,
and maps marks to host event record/wait calls. Lower all ranks before
submitting any rank. Keep collective bookkeeping and diagnostic taps in
that adapter. `execute` stops on the first error. The caller uses `drain`
before rollback or buffer reuse, including preparation errors before the
executor starts. Pass the communication and compute stream lists separately;
the helper synchronizes every communication stream before any compute stream,
without relying on failed schedule events. Stop and poison after the first
drain failure; leave an already poisoned instance alone.

The bundle owns half inputs, row and workspace views, recurrent state
handoff, media row mappings, prefix captures and head gathering. Splits
must meet the bundle's numerical granule. A schedule alone does not prove
bitwise chunk invariance or qualify concurrent device execution.

Qwen's rank-tuned default prefill overlap uses this executor with whole-chunk PLE
staging, per-half recurrent history handoff, disjoint row views and
part-aware tap capture. Its input split preserves scored rows and assigns
anchors using bundle-supplied absolute frontiers. The legacy `split_at`
keeps its refusal policy and existing consumers' journals. The Qwen switch
and resource settings are documented in [the model guide](model-qwen38.md#prefill-overlap).

## Bound projections

`projection::Store::decode` reads an authenticated precision component and
rebinds every use through the caller's `ImplementationProvider<Handle =
Recipe>`. It compares the complete supplied identity, including dependency,
layouts, activations, arithmetic and stage order, with the component. The
provider owns admission and physical layout validation; the view resolver
supplies the persistent codes, local scales, weight globals and supplied
activation calibration. Dynamic activation globals use caller scratch. The bound calibration
chooses the source: supplied globals require persistent views, tensor-amax
requires scratch and refuses persistent globals, and absent recipes ignore
unrelated global and amax scratch. Decode refuses mismatched persistent views.

`projection::Authenticated` splits that work for multi-rank instances.
`Authenticated::decode` (or `new` over an already decoded document)
rebinds every use once and retains the adapters, launchers and needs;
`Authenticated::bind` then attaches one rank's views and returns its
`Store`. Its resolver may return `None` for a selection the rank does not
serve, which leaves that selection out of the rank's store and document.
`Registry::for_document` registers one launcher per distinct identity of a
document.

`Store::handle(id, usage)` returns a retained handle. `Handle::needs` reports
capacity workspace, temporary output, gathered input, quantizer codes,
scales and dynamic-global scratch sizes. `Handle::recipe` exposes the
immutable typed entry planners for acquisition and shared-memory setup.
Caller arenas own these regions and their initialization. Lowering does
not allocate device memory, upload or submit device work.

`projection::launch(handle, phase, input, output, scratch)` returns ordered
`PreparedLaunch` descriptors for the existing pipeline or graph recorder.
It performs input gather, activation quantization, compute, a separate
grouped split-K reduction when required, and the bound output layout copy.
Dense split-K reductions remain inside their selected compute entry. The
phase must match the authenticated phase, capacity and graph mode.

Legacy recipes and stateful prefixes use `projection::Registry`. A bundle
registers a `Launcher` under the complete `ImplementationIdentity`; no
shared code matches an entry's namespace. The callback receives the phase,
buffers, and optional side spans and scalars interpreted by that
implementation. It reports its capacity needs and returns launches only
for its authenticated stages. Callbacks may declare per-record repetition counts for each authenticated
stage, such as a stateful prefix run per item. Stage order and multiplicity
are checked exactly; repeated entries in the identity are never collapsed. A
bundle lowers its declared projections through these APIs and may bind any
use, or every use, to a registered launcher.

Scratch spans must be disjoint from one another. Active writable scratch
also must be disjoint from input, output, persistent weights and side spans.
The handle reports external GEMM semaphore bytes separately; initialize
these counters to zero before first use. Fragment prefill uses a sufficient
capacity bound across all live row counts. Dense layout-copy temporary
output must be separate from final output. Grouped and lookup output views
must cover prepared capacity; layout copies on those kinds are refused.
Registered launchers receive the caller's `Scratch` in both `writes` and
`launch`, after its regions satisfy their declared `Needs`. They declare
written regions using these reserved spans. Prefixes of prepared recipes may
write only to declared side spans, which remain disjoint from the op's
scratch. Complete
registered recipes may also declare output and scratch writes.

`projection::ScratchBuffers<T>` is one shared typed declaration for these
regions. Its bound `Scratch` alias holds spans; the same declaration can hold
`Allocation` leaves and use `scratch::bind`. Region checks traverse that
declaration in field order, including every present optional region, except
`unused`: it only fills kernel arguments the recipe never reads, so it may
name any live region, takes no part in overlap checks and is never a writable
region. A registered recipe that reads a model constant receives it as a side
input.
