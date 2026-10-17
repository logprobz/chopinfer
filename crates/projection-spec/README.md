# projection-spec

Projection declarations. Each model declares its quantized projections once,
as data, in `models/<m>/contract/projections.json` (an external model keeps
the same file in its own contract, as `examples/external-model` does). Every
consumer reads it through this crate: the compiler binds checkpoints against
it, the weight plan derives rank slices and records from it, the bundle's
build generates kernel offers from it, and the bundle binds the resulting
precision component at create. The crate has no CUDA, no checkpoint IO and no
model names (`tests/model-neutral-host.sh` checks it). It depends only on
kernel-core (with `precision-serde`), serde and sha2; model-compiler-sdk depends
on it for imports and weight-plan records.

Schema v1 (`chopinfer.projection.v1`) is frozen. Later changes are additive:
the optional `drafters`, `publishes_as` and `offers` fields came after it, and
a declaration that omits them parses, validates and digests as before.

## Workflow

1. Declare projections, fusion groups, layer stems, side tensors and uses in
   `projections.json`. Shapes name geometry keys, so one declaration serves
   every compiled geometry of the model.
2. `parse(text, geometry, registry)` refuses unknown fields everywhere,
   decodes registered variant parameters and runs `Declaration::validate`.
3. `Declaration::bind` claims every checkpoint tensor from header metadata
   through a `CompanionRules` import hook, usually the SDK's
   `HeaderCompanionRules` or `declared::AdmittedRules`.
4. `derive::derive` produces one rank's operations, slices, required views
   and variant records as pure functions of the declaration, geometry,
   topology and imported operands.
5. The SDK turns derived operations into weight-plan records
   (`model_compiler_sdk::quantization::projection::{records, side_records}`,
   `persistent::rank_records`).
6. The bundle's build generates offers with
   `kernels_sm120::precision::generated` and keeps existing symbol names
   through its adoption ledger.
7. At create, `bundle_runtime::projection` decodes the precision
   component into bound handles, and `projection::launch` lowers each use.

## A declaration

| Field | Holds |
|---|---|
| `schema`, `namespace` | `chopinfer.projection.v1` and the model's variant namespace |
| `use_profiles` | Named, ordered use schedules that projections and groups reference |
| `layer_stems` | Checkpoint layer families: name template, materialized indices, optional execution indices and aliases, support rules |
| `projections` | Logical matrices: sources, `[rows, k]` per operand, kind, TP split class, epilogues, numerical contract, uses |
| `fusion_groups` | Ordered members sharing one input, with eligibility, fallback and output segments |
| `side_tensors` | Non-projection tensors (norms, gates, routers, convolutions, state, biases, sinks), their admitted dtypes, placement and consumer |
| `external_tensors` | Tensors another declaration owns and validates, such as encoder weights, kept in closed coverage |
| `drafters` | Optional drafters: an `id`, the stems and stemless projections each owns, and the operations it records all or none |

### Projections

A projection has an `id`, a checkpoint `source` (a name pattern, with an
optional `stem`), a logical `shape` of `[rows, k]` per operand, and `kind`:
`dense`, `expert_bank` (with `experts` and `top_k`), `lookup_table` or
`grouped_dense` (with `groups`). A grouped dense projection is `groups`
independent `shape` matrices stacked along the source's rows, without routing
or a sum across groups: group `g` reads its own K-wide input slice and writes
its own output columns, a rank owns whole groups (`expert_owned`), and the
derived operation maps onto the grouped fold GEMM contract
(`kernel_core::mxfp8_gemm`) rather than a dense kernel. `operands` names fused operands such as gate and up; without `{operand}` or
`{expert}` in the source name, one tensor stacks operands along rows and
experts along its leading axis. `parts` are row chunks that keep their own
scale-grid origin. `format` optionally pins a format plug-in; without it the
importer chooses from the export metadata. `composition` maps source formats
to numerical recipes and registered implementations, with ordered `pre` and
`post` stages. `assembly` builds a projection from members of other bound
projections; `lookup` declares lookup shards and head vocabularies.
`publishes_as` serves the projection through another projection's published
port, and `offers` lists the formats its port publishes through generated
kernel offers (see [Drafters and published ports](#drafters-and-published-ports)
and [Kernel offers and ledgers](#kernel-offers-and-ledgers)).

### Tensor-parallel split classes

`split` states how ranks own a projection and the tile its split axis pads
to: `replicate`, `column` (weight rows), `row` (K), `head_grouped` (heads of
a given dimension, divided exactly), `segmented_heads` (heterogeneous,
possibly discontiguous head segments with an explicit gather order),
`expert_owned` (whole experts per rank), `expert_sharded` (each expert's
intermediate axis) and `vocab_sharded` (table rows). `operand_splits`
overrides the axis for one named operand. Format plug-ins own split legality
at scale-group boundaries and the physical padding.

### Layer stems, MTP and drafters

A stem's `template` contains `{i}` once and `indices` lists the materialized
layers (a range or an explicit list). Decoder layers, multi-token prediction
(MTP) layers and drafter layers are all ordinary stems; `drafters` only
records which stems and projections a drafter owns. `execution` selects the
executed indices separately from the materialized ones and adds `aliases`,
which run an executed layer from another stem's source layer without claiming
its tensors twice. A stem's `support` rules gate rank counts, formats and
phases, optionally conditioned on the formats a checkpoint actually binds;
binding resolves those conditions once and `admit_checkpoint` applies them
before operation admission.

### Drafters and published ports

`Declaration::drafters` lists optional drafters. A `Drafter` owns its layer
`stems` (with every projection they own) and the stemless `projections` it
names. Fusion groups belong to the drafter that owns their members, and side
tensors to the drafter that owns their consumer or source stem. `operations`
names the member projections whose prepared operations a component records all
or none. Stems and projections belong to at most one drafter, and a fusion group
may not mix owners. `omit_drafter` removes a drafter with every member, for a
geometry that has no drafter layers.

A drafter layer that executes after the target's layers reuses execution
aliases: its stems declare the executed index and the source index it reads.
`LayerStem::executed` lists the executed indices and `LayerStem::source_index`
maps an executed index back to its stem's materialized one.

`Projection::publishes_as` names another projection whose published port,
offers and entries serve this one, for example a drafter layer's experts that
run the main layers' grouped entries. The projection keeps its own sources,
stem and format admission. Validation requires the same kind, shape, operands,
parts, split, epilogues, numerical id, format and expanded uses, and a target
that publishes its own port (no chains or cycles). Its composition may admit
fewer formats than the target's, never another format or recipe.
`Declaration::published` resolves the reference.

### Uses and profiles

A use is a `phase` (`Prefill`, `Decode` or `Verify`), a requested `capacity`
and a `graph` flag, with an optional contract: compiled capacity, row
multiplier, chunk rows, per-item and aggregate verify limits, output dtype,
numerical contract, exact rows, support rules and an explicit
implementation. Requested buckets and compiled capacity are separate.
Projections usually reference a `use_profiles` entry, optionally with
overrides selected by `when`; `expanded_profiles` writes each projection's
uses out.

### Fusion groups and side tensors

A fusion group lists members that read one input. `eligibility` decides when
they fuse (by default when their bound formats agree) and `fallback` runs
them `separate`ly or refuses. `outputs` places each member's columns in named
output buffers.

A side tensor has a `kind` (`norm`, `gate`, `router`, `conv`, `state`,
`bias`, `sink`, `auxiliary`), a source and shape, admitted source `dtypes`,
an independent placement across ranks, a `consumer` (a declared projection or
group, or a named kernel) and optional consumer `views`. A view preserves the
stored bits or widens BF16 to F32 exactly; narrowing is refused. A `norm`
side may carry a shared `NormContract`; a kernel-owned norm omits it.

## Validation and binding

`validate` checks names and patterns, geometry expressions (exact division,
checked overflow, instance limits), split legality, profile references and
overrides, fusion and output overlap, side placement, and registered variant
parameters and ports. Unknown fields, dimensions, format ids and variant ids
fail.

`bind(geometry, registry, metadata, rules)` reads `CheckpointMetadata`
(tensor names, dtypes, shapes and files from safetensors headers, plus the
export's `quantization_config`), never payload bytes. For each source the
`CompanionRules` hook returns the imported weight, its parts, any extra
companions and static calibrations. Binding claims every tensor once: a
missing tensor, a shape or dtype mismatch, a repeated claim or an unknown
checkpoint tensor fails with the tensor named. Fusion decisions, assemblies,
layer aliases, side views and external owners are checked in the same pass.
File authentication and no-follow checks stay with the checkpoint reader.

## Derivation

`derive::derive(DerivationInput)` takes the declaration, the bound result,
the geometry, the registry, a `Topology { ranks, rank }`, the imported
projections and a `VariantDeriver`, and returns a `RankDerivation`: per-use
`DerivedOperation`s (the rank-local `OperationSpec`, logical and padded
geometry, source slices, packing plan, required views, stages and variant
records), fusion results and side views. `rank_selection` gives one
projection's rank geometry without imports; a model's offer generation uses
it. `RankDerivation::operation_specs` lists the portable operations.

A grouped projection keeps one scalar weight global for the whole bank only
when every operand reads the same scalar tensor. Operands with distinct
scalar globals, such as separate stacked gate and up sources that each carry
their own, pack one global per expert and operand.

The SDK's `quantization::declared` module is the convenience adapter for
callers that import each source with the SDK importer. `AdmittedRules` binds
admitted operands and claims their static input calibrations per source
frame when the calibration tensor is shown in the caller's
`CheckpointMetadata`. A claimed calibration must be its whole checkpoint
tensor (offset zero, the header's dtype and shape, the exact byte length);
frames are sliced from it only after that check, so fused operands or
several sources can share one scalar or per-expert calibration tensor. A
caller ignores a calibration it does not use, for example a static input
scale of an export served with BF16 activations, by leaving the tensor out
of the metadata it shows: binding then neither claims it nor requires it as
a companion, and coverage closes over the shown tensors. `declared::imported`
builds the imports, from separate per-expert sources or one stacked expert
bank whose operands stack along rows, and for each layer of an assembly from
its materialized member. `declared::admitted_activations` restores the
checkpoint's activation recipes on derived operations, assemblies over a
stacked bank included. `declared::derive` composes binding,
`declared::imported` and `derive::derive`. It always imports rank-local
geometry: a caller that needs another `GeometryMode` calls
`declared::imported` and `derive::derive` itself. `declared::imported` does not read a stacked bank with parts or a
dense projection whose fused operands share one tensor; such callers bind
with `HeaderCompanionRules` or their own hook and construct the imported
projections for `derive::derive` explicitly.

## Registered variants

Shared code never branches on a model. Genuine model arithmetic (a source
transform such as a scale regroup, a stateful stage before a lookup, or an
existing fused entry kept for byte identity) is a registered variant: a
`<namespace>/<name>` id with typed ports, strict parameters and an immutable
numerical contract. The model registers a `registry::Provider` (its
namespace and `Registration`s, each with a decoder) in a `Registry`, and
implements the behavior in a `derive::hooks::VariantDeriver`. A declaration
with no variants uses `Registry::default()` and `hooks::NoVariants`; missing
behavior refuses rather than falling back. Shape never selects an
implementation.

## Digests and goldens

`canonical_json` sorts object keys and emits compact typed JSON, and `digest`
is its lowercase SHA-256, so key order, whitespace and omitted defaults do
not change it. A model pins its declaration digest in its contract tests and
keeps its derived outputs (locks, journals, weight plans, load digests) as
the byte-identity evidence for a port. This crate's synthetic fixture has an
independently computed digest; regressions cover every schema revision item,
source checks and exact rank bytes. Never update a pin to hide a mismatch.

## Kernel offers and ledgers

Offers come from `kernels_sm120::precision::generated`
(see `crates/kernels-sm120/src/precision/README.md`): a caller passes a
rank-local `OperationSpec` (from a derivation or `generated::domain`), a
format plug-in id, a use and an explicit implementation schedule, and
`generated::offer` builds the entries through the typed builders and binds
the resulting capability. Generated symbols are
`chop_proj_<format>_<kind>_n<rows>_k<k>_<digest>_v1`, so equal shapes share
one body across models; a row stage built with launch shapes drops `n` and
`k` and serves every geometry.

`Projection::offers` lists, in order, the formats a projection's port
publishes through shared generated kernel offers; registered implementations
publish their own formats. Each offered format maps to the parameterless weight
format its plug-in accepts (`offered_weight`, `Projection::offered_weights`),
so block FP8 has no generated offer. A non-lookup offer needs the composition's
numerical recipe for that format; a lookup uses its own numerical id. Two
offers of one weight format and offers on a projection that publishes as
another are refused. Whether the shared kernels build an offered format is
checked when the model generates its offers, not by declaration validation.

Two optional per-bundle data files keep generated code stable: a bundle
needs the adoption ledger only to keep existing locked symbols and the
schedule ledger only to record schedule overrides. They live beside the
bundle's `kernels.lock`, which stays the authority for cubin bytes:

| File | Records |
|---|---|
| `projection-adoptions.json` | `symbol`, `generated`, `entry_sha256` and optional cubin `placement` (`cubin`, `position`): a generated identity keeps its existing locked symbol, and placed entries fill their recorded slot in an existing catalog cubin. A record with `replaced_by` instead retires its symbol for the named active entry and has no placement |
| `projection-schedules.json` | Per-instance schedule overrides keyed by the portable execution inputs (`generated::schedule::Override`); other instances use `generated::schedule::default` |

Both are append-only canonical JSON arrays; a later record supersedes an
earlier one for the same key. `tests/projection-adoptions.py` and
`tests/projection-schedules.py` refuse rewrites, removals and unchanged
duplicates against the approved base. `generated::adopt` proves an initial
alias by comparing a generated offer with the existing offer's capability,
declarations, compilation inputs and entry code.

## Reference: compact declarations

`Declaration::use_profiles` maps names to ordered `Use` schedules. A projection
or fusion group sets `use_profile` to an id or
`{"profile":"served","overrides":[...]}`. An override selects optional phase,
capacity and graph fields through `when` and sets output dtype or a registered
implementation. Overlapping overrides for the same field and unmatched
selectors fail. Inline `uses` and `use_profile` are mutually exclusive.
`expanded_profiles` returns a derived copy; the canonical digest retains the
compact wire representation and ordered arrays.

Consumer references may name a declaration id or `{"kernel":"attention.query"}`. Kernel-owned norm sides omit `norm`; the bound view dtype selects the consumer's weight variant without approximating its arithmetic.

Layer stems declare materialized indices separately from executed indices and
aliases. Dimensions are positive literals, geometry keys or checked sums,
products and exact quotients. Offset expressions admit zero. An alias may name
another `source_stem`. `BoundLayerAlias` preserves both source and execution
identities. Checkpoint-wide support conditions resolve once during binding;
`admit_checkpoint` applies those snapshots before operation admission.

## Reference: operands and assemblies

A projection declares `[rows,k]` per named operand. The default operand is
`weight`. `{operand}` selects its name; `{expert}` selects an expert index.
Without those selectors, sources stack operands along rows and experts along
the leading bank axis. Explicit lookup shards declare source names and logical
row ranges separately from ordered head vocabularies and TP ownership.

A projection's `stem` can supply execution context independently of its source.
Assemblies reuse source projections with a named operand and a layer selector:
`"current"`, a literal, or a Dim expression containing `current`. Each member
specifies global or rank-local coordinates and `inherit_split`. Rank-local rows
are already owned by the rank and bypass destination row splitting; K and expert
ownership follow `inherit_split`. Expert-bank assembly members must agree on
inheritance so one local slot retains one source expert. Concrete source
claims must be unique; explicit assemblies and declared side consumers can
reuse authenticated sources. Two claims naming the same tensor fail with both
claimants, regardless of stem prefix relationships.

## Reference: typed contracts

| Module | Public contracts |
|---|---|
| Root | `Declaration`, `Projection`, `Part`, `FusionGroup`, `SideTensor`, `Dim`, `Geometry`, `LayerStem`, `SplitClass`, `Use`, `Variant`, `Drafter` |
| `profiles`, `layers`, `inputs` | Profile references/overrides, current-layer selectors, activation bindings and row/rank-K segments |
| `contracts` | Composition/stages/ports, numerical recipes, lookup assemblies/scopes, source members, norm/view contracts, aliases/support and external owners |
| `segments`, `side` | Whole-head segment ownership and gather order, arbitrary-rank side placement and exact byte repacking |
| `registry` | Namespaced providers, registrations and strict parameter decoders |
| `bind` | Header/index metadata, import hooks, checked operand/side/assembly/fusion/alias facts |
| `derive` | Per-rank operations, logical/padded matrix geometry, source slices, required views and variant hooks |

Shared dtypes and epilogues come from kernel-core. Format references use
`precision::format::FormatId`; split axes use its `SliceAxis`. Zero padding is
implied. Unknown fields, dimensions, format ids and registered contracts fail.

Registered variants carry typed input/output ports, strict provider parameters
and an immutable numerical contract. Integer ports may annotate
`interpretation: unsigned` while retaining I32 or I64 storage. Source transforms
state input and effective output formats. Ordered stages declare phases and
state mutations with matching output state. Genuine arithmetic and explicit
legacy implementations remain registered; shape never selects an implementation.

`Composition::implementations` selects registered implementations by bound source format. `Composition::implementation` is its fallback.
`UseContract::implementation` overrides either selection for one use. Repeating the default as
an override fails. `Composition::numerical_contracts` maps bound source formats
to numerical ids and kernel-core `A16Recipe`s, before source transforms. A mapped
id supersedes the ordinary base/per-use numerical id; an explicit implementation
must match its immutable id. Binding refuses incompatible source selections for a fused group.
`Composition::numerical_for_sources` resolves the authoritative declaration map
for binding and derivation; bound operands and fusions retain source facts.

## Reference: ownership, views and numerical contracts

Splits include replicate, column/row, head-grouped, segmented heads, expert-owned,
expert-sharded intermediate and vocabulary ownership. Head-grouped ownership
requires explicit head count/dimension and exact division. Segmented ownership
allows heterogeneous dimensions and discontiguous source ranges within each
head group, with source or explicit segment gather order. Parts declare logical
row ranges and independent scale-grid origins.

Side tensors have arbitrary rank, admitted source dtypes and an independent
axis/split placement. Views apply after placement, preserving stored bits or
widening BF16 to F32 exactly; narrowing fails. Shared `NormContract` declares
reduction width, group count, shared/per-group affine storage, reduction order,
F32 epsilon bits, `(1+w)` convention and admitted BF16/F32 weights. Affine storage
is checked against the source shape. The current grouped norm kernel fixes
1e-6 epsilon, `(1+w)` and per-group affine indexing. Declared arithmetic outside
that contract is refused until an admitted lowering exists. Kernel-owned sides
may omit the arithmetic contract.

Requested buckets are separate from compiled capacity, row multiplier, chunking
and per-item/aggregate verification limits. Each use can select its output dtype.
Fusion groups declare eligibility, separate/refuse fallback and logical segments
in named output buffers. Overlap is checked within a buffer; format plug-ins own
physical padding and strides.

Lookup shards accept explicit lists or `{count,source,rows}` templates. Templates expand `{shard}` and a scoped `shard` Dim for row ranges; variable head vocabularies remain explicit. Resolve the `LookupShards` type before iterating.

Lookup kind implies weight-only import. Projection activation quantizers and
input calibration cannot apply to integer lookup indices. Lookup companion
scope has a default and format-keyed local/global overrides; binding checks the
actual imported tensors for table sharing or shard ownership. Optional external
owners are absent or complete, and every present tensor requires successful
validation attested by its owning contract. Coverage closes over all owners.

## Reference: shared metadata imports and derivation boundary

`model_compiler_sdk::quantization::projection::metadata::HeaderCompanionRules`
adapts header/index facts plus quantization documents through the SDK importer
and format plug-ins. Optional `HeaderRole`s declare target aliases, explicit
companion names per format and scalar sharing for independent frames. Tensor
spans retain source dtypes and tensor-relative offsets; calibration tensors
remain provenance requirements. `ImportedOperand::calibrations` retains the
resolved expert/operand/part coordinates and bounded F32 scalar spans. Explicit
scalar sharing is validated before source frames are admitted. Unknown recipes, unexplained companions,
invalid cardinality and incomplete code/scale coverage fail.

Independent sources import validated 2D frames in expert-major, operand-major,
part-major order. Each frame retains declared grid origins, separate code/local
scale offsets and explicit global/calibration sharing. The full descriptor
retains raw source shapes, while parts validate the restarted grids. No uniform
grid is fabricated for a compound source.

Shared rank derivations cover execution aliases resolved through declared source
roles, inherited/rank-local assemblies, and activation owner-domain intersections.
An activation segment's `origin` and `length` define its physical owner's domain;
active intersections produce owner-relative columns, and inactive owners are
skipped. Domain segments share one logical selection and must cover live K once.
The CPU gather preserves source bits and zeroes physical padding.

`VariantDeriver` provides source/part transforms, ordered stages, explicit legacy
implementations and fusion hooks. Context retains bound source identity and rank
selections; `ordinary()` checks topology, declaration/use membership and source
authentication before supplying a baseline. Fusion context includes its
owning group and selected numerical/A16 recipe. No implementation is inferred
from shape. Environment-based implementation policy and legacy input arithmetic
remain owned by explicitly registered implementations.

Source hooks return `SourceTransformResult::Applied(imported)` or `Bypass`.
A bypass preserves the entire input. Applied transforms retain authenticated
`SourceMatrix` values and supply effective weights separately. Changed weights
require `ImportedProjection::preparation`: an explicit topology, physical rank
geometry and preparation records. Derivation preserves that carrier in
`DerivedOperation::source_preparation`, including when a legacy adapter replaces
its final packing records. Ordinary operations use those preparation records.
Per-part hooks retain their source window; compatible preparations combine.

`DerivedStage::adapter` carries a `StageAdapter` owned by the registered stage
contract. `StageExecution` selects a standalone symbol or fusion into an owning
operation. Typed `AdapterValue` facts retain addressing, integer values, F32 bits
and runtime `ControlBinding` values without changing logical ports. Providers
validate field meanings; shared checks retain provider identity and byte-aligned
control storage. Fused execution does not request an additional stage launch.

`VariantRecord::view_offset` is a byte offset within its required view. The SDK
checks bounds, overlap and complete coverage across fragments, and deduplicates
complete record sets across uses. Arena placement stays with the consumer.
`DerivedOperation::record_sources` retains authenticated aggregate and part
spans for contributing tensors. Preparation records can read the whole bound
tensors while preserving validated part mappings. Persistent descriptors
preserve both view offsets and bounded source offsets;
`persistent::rank_records` deduplicates complete resident recipes.

SDK persistent descriptors retain authenticated spans and placement for existing
host ABI transforms. Offset/multipart source frames require explicit provider
records. There is no additional multipart recipe or executor.
