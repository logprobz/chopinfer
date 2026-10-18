# encoder-spec

Encoder declarations. Each vision or audio encoder is one JSON file,
`models/<m>/contract/encoders/<name>.json`, that every consumer (host,
compiler, bundle, reference, tools) reads through this crate. The crate has no
CUDA, no device code and no model names (`tests/model-neutral-host.sh` checks
`src/`). The kernel catalogs are generated from it (kernels-sm120
`encoder::catalog`, see its `src/encoder/README.md`) and the shared encoder
runtime lowers and runs it (bundle-runtime `encoder`, see
`docs/bundle-runtime.md`). The GPU reference runner evaluates the same files
in torch (`tools/refrun/encoder.py`, one implementation per variant id),
reading the JSON itself; the declarations it runs are the ones this crate
validates.

A declaration holds:

| Field | Meaning |
|---|---|
| `schema` | `chopinfer.encoder.v1` |
| `model`, `name` | the owning model directory and the frontend's `encoders[].name` |
| `precision` | the precision policy (below) |
| `checkpoint` | tensor-name `prefix`, optional `files` (name prefix to safetensors file), and `config` pins: values the checkpoint config must hold at a JSON pointer of a file |
| `input` | host recipes with the limits that bound their rows, patch row order (`patchify`), payload element and columns, the kernels' row capacity, and optional `chunking` |
| `sequences` | how rows form attention sequences: `item`, `temporal-grid`, `segments` |
| `image_layout` | how the per-rank encoder image is laid out: `program-order-stacked.v1` or `view-order.v1` |
| `stages` | ordered components, each `{"op": "<id>", <parameters>, "weights": {<role>: <source>}}` |
| `output` | the layout into the language model: hidden size, position rule, media mask flag, and learned rows if any |
| `parallel` | admissible parallel policies, `replicated` first |

`parse(text, &environment)` refuses unknown fields everywhere, decodes
model-namespaced variants with the environment's registry, and runs
`Declaration::validate`.

## Components

A component id is `[<model>:]<kind>.<name>.v<N>` (`id.rs`). Each kind's
shared variants are a closed enum in `src/components/<kind>.rs`; a variant's
numeric parameters (widths, heads, eps, theta, windows) are data, and its
numeric behavior is fixed by its id.

| Kind | Shared variants | Policies |
|---|---|---|
| `linear` | `linear.bf16.v1`, `linear.bf16-round-linear.v1` | legacy |
| | `linear.pairs.v1` | pairs |
| `conv1d` | `conv1d.im2col.v1` | both |
| `position` | `position.learned-grid.bilinear-align-corners.f32.v1` | pairs |
| `norm` | `norm.layer.v1`, `norm.rms.v1` | legacy |
| | `norm.layer-f32.v1`, `norm.rms-f32.v1` | pairs |
| `transformer` | `transformer.pre-norm.v1` with slots `norm`, `attention`, `rotary`, `mlp` and per-block `overrides`, `reorder`, `carry`, `sequences` | both |
| `attention` (slot) | `attention.varlen.bf16.v1` | legacy |
| | `attention.varlen.pairs.v1`, `attention.varlen.qk-rms.pairs.v1` (per-head q and k RMSNorm before the rotary, `qk_norm_eps`) | pairs |
| `rotary` (slot) | `rotary.axial-2d.v1` (table `table.hf-f32-angle.v1`) | pairs |
| `mlp` (slot) | `mlp.gelu.v1`, `mlp.swiglu.v1` | both |
| | `mlp.swiglu-clamped.v1` (gate at most `limit`, up within `[-limit, limit]`) | pairs |
| `reorder` (slot) | `reorder.merge-group-columns.v1` | both |
| `merge` | `merge.consecutive.v1`, `merge.space-to-depth.v1` | both |
| `group` | `group.repeat-last.v1` | both |
| `cast` | `cast.bf16.v1` | pairs |
| `pad` | `pad.zero-columns.v1`: F32 rows widened with zero columns to `columns`, for a GEMM whose K steps the input's width does not fill (the weight takes `pad_cols` to match) | pairs |
| `projector` | `projector.norm-gelu-mlp.v1`: `mlp(gelu(norm(x)))` with no residual, slots `norm` and `mlp` | pairs |
| `quantize`, `embed` | none; models register their own | |
| `layout` (output) | `layout.tokens.v1`, `layout.learned-delimiters.v1` | both |

A `factor x factor` convolution whose stride equals its kernel, over
`merge-blocks.v1` rows, needs no variant of its own: `merge.consecutive.v1`
(`rows: factor * factor`) then `linear.pairs.v1` with the weight transforms
`["flatten", {"space_to_depth_cols": factor}]`, as `tests/vision_components.rs`
declares.

Every component states, through `Component` (or `RotaryComponent` for the
rotary slot):
- the rows it writes for the rows it reads (`Boundary`): data (rows of a
  width and element, or codes with per-code bounds), the most rows one run
  holds, the multiple every item's rows come in, whether they follow a grid,
  and whether they are still split into segments;
- its weight roles (`Role`): the shape and element the kernel reads (BF16
  unless the role states otherwise), and for a stacked role each part's
  shape in order;
- the precision policies it implements and its parallel class;
- one item's rows after it, for exact geometry (`Declaration::trace`);
- for a rotary variant, what a row rotates by (`axes`: its grid row and
  column, or its index inside its sequence);
- for a model's registered variant only, the scratch regions its lowering
  needs (`scratch`) and its cost estimate (`cost`).

## Model providers

A model's bespoke variants live in the model's crate, not here. The model
defines each as a serde enum tagged by `op` with an id in its namespace
(`<model>:<kind>.<name>.v<N>`), implements `Component` or
`RotaryComponent` for it, and lists it in a `registry::Provider`:

```rust
pub static PROVIDER: Provider = Provider {
    namespace: "<model>",
    registrations: &[Registration {
        id: "<model>:quantize.<name>.v1",
        decoder: Decoder::Stage(registry::stage::<MyQuantizer>),
    }],
};
```

The caller registers the provider in the `Environment` it loads that model's
declarations with. `Registry::register` refuses ids outside the namespace,
repeated ids, a decoder for the wrong kind and a namespace registered twice.
Validation decodes each namespaced id with its registration (unknown
parameters refused), refuses an id no provider of the owning model registers
or that belongs to another model, and admits a decoded variant only if the
environment's registry holds the provider that decoded it. A registered
variant's id never changes meaning; a change is a new id. Registering a
provider leaves every other declaration's digest unchanged. MiMo's provider
is `models/mimo26/contract/src/encoders/spec.rs`.

## Precision policies

- `f32-pairs.v1`, the default for every new declaration: F32 payload, F32
  rows and residual stream, activation operands of tensor-core products as
  BF16 hi+lo pairs, F32 norms, the reference's F32 host tables, and one BF16
  rounding at the rows the language model reads.

The declaration loader refuses the retired `bf16-legacy.v1` policy with an
explicit error. No precision permission can enable it.

The gate follows the rows' element, not the last op's kind: under
`f32-pairs.v1` no stage may read BF16 rows, and the language model reads BF16
rows, so exactly one rounding happens at that boundary, fused into the last
GEMM (`linear.pairs.v1` with `output: bf16`) or as `cast.bf16.v1`.

## Validation

`Declaration::validate` refuses, with a message naming the stage:

- a declaration of another model than the environment's owner; unknown ids,
  kinds, parameters and weight roles; a slot kind used as a stage;
  unregistered or foreign model-namespaced ids;
- recipe limits that disagree with the input (columns, patch order,
  capacity). Without `chunking`, `capacity_rows` must be the largest run
  (from the limits and `sequences`) rounded up to a multiple of 1,024 rows;
  with `chunking: whole-sequences.v1`, runs pack whole sequences (or groups)
  up to `capacity_rows`, a multiple of 1,024 that holds the largest one;
- shape mismatches between stages, grid-dependent stages on rows with no
  grid, merges of rows that are not whole groups or still segmented, codes
  outside an embedding's vocabulary, and derived dimensions that overflow;
- scalars (eps, scale, theta) whose F32 value is zero, infinite or negative;
- weight mappings that do not map exactly the component's roles, wrong
  padding or stacked tensor counts, misplaced block-indexed names (`{i}`),
  and tensors no `checkpoint.files` entry holds;
- policy violations: a variant not defined under the declared policy, the
  retired `bf16-legacy.v1` policy, `narrow_bf16` or `round_linear` under `f32-pairs.v1`,
  and rounding before the language model's boundary;
- parallel lists that do not start with `replicated`, repeat a policy, or
  admit `sequence-split` where a stage needs a halo, reorders rows, joins
  segments or sets its own sequences, or where sequences are not one per item.

`Declaration::bind(lookup)` then checks the mapping against checkpoint
metadata: every tensor exists in its declared file, every transform applies
(widening needs BF16, narrowing F32, unfolds need whole cells), and every
role gets exactly its shape and element, block by block and, for a stacked
role, part by part.
`Declaration::check_config(file, &config)` checks the pins against a parsed
config file.

## Derived from a declaration

Pure functions of a declaration and an input, with no model names. Each
model's tests compare them with the code they replace (below).

- **Geometry** (`geometry.rs`). `Declaration::geometry(recipe, media)`
  takes an item as its recipe emits it (`Media::Grid([t, h, w])` patches or
  `Media::Frames(n)`), refuses one outside the recipe's declared limits, and
  gives the item's rows after every stage (`trace`) and the runs it is cut
  into (one item, segment or temporal grid, or whole groups up to
  `capacity_rows` after a joining stage), with each transformer's
  `cu_seqlens`, rotary positions and reorder map. `admit_record` checks a
  wire record against the declared limits, which are wider than what a
  frontend emits.
- **Placement** (`placement.rs`). `Declaration::place(geometry, start)` lays
  the encoded rows into the language model's sequence from position
  `start`: `layout.tokens.v1` gives one piece per temporal grid (text such as
  video timestamps may sit between), `layout.learned-delimiters.v1` adds
  start, newline and end rows. `grid-3axis.v1` gives row `k` of a merged
  `[t, h, w]` grid `(s + k / hw, s + (k % hw) / w, s + k % w)` and advances
  by `max(t, h, w)`, as the engine's `PositionAxesV1` does; `sequential`
  uses running positions.
- **Buffer plan** (`buffers.rs`). `buffers::plan(declarations, rows,
  serving)` sizes one rank's encoder regions for runs of up to `rows` rows
  (`serving` adds the arena, prefill source rows and a recorded driver's
  staging). `f32-pairs.v1` keeps the residual stream in `stream` and
  alternates other outputs between `a` and `b`. Side inputs get regions of
  their own (`Region::owner`, `BufferPlan::owned`), or for a staged
  declaration (`Declaration::staged`: an item can take several runs) shared
  regions filled in command order. A stage never writes the region it reads
  (in-place adds and rotary aside); a declaration whose stages would is
  refused.
- **Cost** (`cost.rs`). `Declaration::cost(run)` estimates each stage's
  FLOPs and bytes for scheduling; it ignores the extra tensor-core products
  of pairs and every device rate.

Each model checks the derived functions against its own code:
`models/<m>/contract/tests/encoder_media.rs` (the frontend recipes' grids and
token counts), `models/<m>/bundle/tests/encoder_spec_geometry.rs` (runs,
rows, `cu_seqlens`, rotary positions, reorders and group gathers against the
bundle's geometry), `models/<m>/bundle/tests/encoder_spec_buffers.rs` (the
plan against every encoder allocation of the bundle), the bundles' fake-host
tests (live row counts, sequence bounds and arena writes), and
`tests/cost.rs` here for the cost arithmetic.

## Digest

`Declaration::digest()` is the SHA-256 of `canonical_json()`: the typed
declaration, registered variants included, serialized with object keys sorted
at every level and no whitespace. It identifies the typed structure:
- key order, spacing and the spelling of typed numbers (`10000` or
  `10000.0` for an f64 parameter) do not change it;
- optional parameters at their defaults are not written, and validation
  refuses an explicit default where the digest would otherwise split
  (`output: f32`, a scale equal to `head_dim^-0.5` in F32), a block set
  naming every block, and a list spelling of `even` or `odd`;
- config pin values are JSON values, compared and hashed as written: `27` and
  `27.0` differ, as they do in a checkpoint config.

## Adding a component variant

A new architecture adds variants; it never edits an existing one.

1. A variant only one model needs goes in that model's crate under its
   namespace, registered by its provider (above). No file here changes.
2. A shared variant goes in its kind's enum with
   `#[serde(rename = "<kind>.<name>.v1")]` and its own parameter struct with
   `#[serde(deny_unknown_fields)]`, plus its arms in the kind's `Typed` and
   `Component` impls: the policies it is defined under, its boundary rule,
   its roles. Every existing arm stays as it is. If no kind fits, add
   `src/components/<kind>.rs`, a `Kind` entry in `id.rs` and, for a stage
   kind, an arm in `StageOp`.
3. Add refusal tests for its rules.
4. Run `cargo test -p encoder-spec` and the model contract tests. The digest
   goldens of every committed declaration must hold: a change to an existing
   variant's parameters, defaults or serialization shows up there.

Changing a variant's math means a new version (`.v2`) beside the old one. An
existing declaration moves to it only through its own edit, which changes its
digest.

## The committed declarations

Each model's contract crate tests its declarations
(`models/<m>/contract/tests/encoder_declarations.rs`): they parse, validate
and keep their digests, bind to the contract's tensor metadata (MiMo
`encoders::expected_for(geometry)`, Qwen `encoder::views::layout()`), hold their pins
against the committed checkpoint configs, and agree with the frontend
records and recipe limits.

- MiMo vision and audio: `models/mimo26/contract/encoders/mimo-{vision,audio}.json`,
  both `f32-pairs.v1`. `models/mimo26/contract/src/encoders.rs` checks tensor
  metadata and derives the image order from the declarations; the shared
  runtime owns geometry and launches. MiMo's F32 RVQ and embedding sum live
  in its `bundle/cuda/mimo_encoder.cuh`.
- Qwen3.8 vision: `models/qwen38/contract/encoders/qwen38-vision.json` (F32
  payloads, BF16 pair arithmetic, the learned position table, rotary and
  merger rules); its weight views are in
  `models/qwen38/contract/src/encoder/views.rs`.
- `tests/fixtures/dsv41-vision.json` is a test fixture only (DeepSeek V4.1
  Flash vision, not a served model). It exercises `norm.rms-f32.v1`,
  `merge.space-to-depth.v1`, `layout.learned-delimiters.v1`, the centered
  pad recipe (`image.centered-pad.v1`) and row-major patchify, and `tests/declarations.rs` binds it to its
  checkpoint headers (`tests/fixtures/tensors/dsv41-vision.txt`).

Generated kernel entries and their numerical contracts are described in the
[encoder kernel guide](../kernels-sm120/src/encoder/README.md). The
declaration-driven GPU reference is `tools/refrun/encoder.py`; compare
served and reference taps with `tools/qual/tapdiff.py`.
