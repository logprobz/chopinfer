# Qwen3.8 Flash weight plan and package

The full correctness suite includes [KV preemption swap qualification](serving-kv-pool.md#formats-caching-and-offload); offload stays disabled by default.

Dimensions come from the checkpoint's `config.json`, matched to a compiled
entry in `models/qwen38/geometries/supported.json`. Numeric shapes and memory
figures below describe the shipped geometry. Other admitted geometries derive
their own per-rank shapes, catalogs and resources. Topology IDs follow
`<model>-<geometry>-tp<n>`; see the
[same-architecture recipe](adding-a-model.md#same-architecture-checkpoints).

`models/qwen38/compiler` (crate `model-qwen38`) places the pinned FP8
checkpoint into per-rank arenas at TP2 or TP4 and emits the package of
[bundle-qwen38](bundle-qwen38.md) that serves them. The package holds no
weights: the bundle's weight plan (`models/qwen38/contract/src/weight_plan.rs`,
built from the same placement) produces every arena from the checkpoint on
each load ([direct weight loading](direct-loading.md)). One package serves
every topology the checkpoint's precision serves (the FP8 checkpoint and both
mixed NVFP4 exports serve TP2 and TP4); at TP2 the n-gram table is kept in
host memory by the device capacity rule. Vision tensors form a separate encoder image; the MTP
drafter's tensors are placed with the main layers.

## Views

`placement::ImagePlan` lists each rank's views: 4096-byte aligned spans with
a name, dtype, shape and layout ID. Names follow the checkpoint, except that
the gate and up halves of an expert or shared expert form one `gate_up_proj`
view. `model_qwen38_contract::views::validate` states what the program reads;
the plan passes it on every rank, for the synthetic catalog and for the real
checkpoint's headers.

Every transformation is exact:

| View | Stored as | From the checkpoint |
|---|---|---|
| dense 2-D `.weight` | `bf16_mma_fragments_16x16_v1`, rows padded to 16 | row slice of this rank |
| routed experts | `fp8_e4m3_mma_fragments_16x32_v1`, `[128, N, K]` | local experts in order, gate rows then up rows |
| expert block scales | F32 `f32_block_scale_128x128_v1`, `[128, N / 128, K / 128]` | BF16 `weight_scale_inv`, widened |
| `A_log`, `dt_bias` | F32 `f32_row_major_v1` | BF16, widened |
| shared expert `gate_up_proj` | BF16 fragments `[512, 2560]` | 160 gate rows, 96 zero rows, 160 up rows, 96 zero rows |
| shared expert `down_proj` | BF16 fragments `[2560, 256]` | 160 columns, then 96 zero columns |
| `shared_expert_gate` | `bf16_row_major_v1` | as is |
| embedding, norms, convolutions | row-major | as is |
| n-gram table | `fp8_e4m3_ngram_head_rows_160_v1` | this rank's four hash heads |

Widening BF16 to F32 appends zero mantissa bits. The shared expert padding is
exact: `SiLU(0) * 0 = 0` for the padded intermediate columns, and the zero
down columns multiply those zeros. Shapes above are TP4's;
[topology-qwen.md](topology-qwen.md) lists the per-rank shapes of every
topology.

Each full attention layer's `q_proj` (six heads, each query then gate),
`k_proj` and `v_proj` are placed back to back, each a whole number of 16-row
strips and of 4096 bytes. The bytes from `q_proj`'s offset are therefore the
fragment view of one `[3584, 2560]` weight whose output is the packed q|k|v
row, so one prefill GEMM writes it. `views::validate` refuses an index where
they are not contiguous.

### The MTP drafter

The drafter's decoder layer `mtp.layers.0` is placed exactly like a main
full-attention layer: every `mtp.layers.0.X` view has the shape, dtype,
layout ID and size of `model.language_model.layers.3.X` on each rank. It runs
on the main layers' kernels and lowering as a 49th full layer
(`contract::MTP_LAYER`) with its own pages and ring. `mtp.fc_hidden`,
`mtp.fc_embedding`, `mtp.pre_fc_norm_*` and `mtp.hyper_connection_mixer.*`
are replicated. The `mtp.*` names sort after every other view, so the
drafter's views are the tail of each image; a test asserts it.

## Encoder images

Image inputs run the vision tower from a separate image per rank, a second
planned component beside the text image. The images are identical:
`model_qwen38_contract::encoder::views::layout` fixes every view as
row-major BF16 (`bf16_row_major_v1`) at a 4096-byte aligned offset. Two views
change shape, both exactly:

- `linear_fc1` weight rows and bias, and `linear_fc2` columns, are
  zero-padded from 4304 to 4352 in every block. The padded fc1 columns are
  GELU(0) = 0, and the padded fc2 rows multiply those zeros.
- The patch convolution `[1152, 3, 2, 16, 16]` is stored as `[1152, 1536]`,
  the same bytes.

Each source tensor must be BF16 with the shape its view is taken from. Each
image is about 0.9 GB. The weight plan copies each `model.visual.*` tensor
whole into its view and supplies one runtime index per rank (schema
`chopinfer.qwen38.encoder-image.v1`), which the bundle parses with the contract's
`encoder::views::parse`.

## Projections and precision

`models/qwen38/contract/projections.json` declares the quantized projections
and their side tensors as data: the linear-attention, full-attention, MLP,
expert and hyper-connection projections of the decoder layers, the MTP
layer's projections as ordinary layer stems, the embedding, head, MTP input
and PLE projections, and the n-gram lookup table, whose 128 checkpoint
sources a shard template expands. The MTP layer is data too: the drafter
`mtp` owns its stems, its stemless projections and the operations a
component records all or none; its stems' execution aliases run it after
the main layers from source layer 0; its dense and expert projections
publish through the main layers' ports (`publishes_as`); and each port's
`offers` lists the formats its generated kernels publish (BF16 and MXFP8
for dense ports, NVFP4 for the routed experts and the n-gram table). Norms, convolutions and the n-gram tables'
metadata are side tensors, and the vision tower's tensors are external
tensors owned by the encoder declaration. `model_qwen38_contract::projections`
parses it with four registered variants: `qwen38/ngram-hash.v1` (the hash
stage before the n-gram lookup), `qwen38/legacy-input-fusion.v1` and
`qwen38/legacy-fused-dense.v1` (the existing fused BF16 entries and their
input arithmetic) and `qwen38/legacy-fragment-experts.v1` (the FP8 fragment
experts). The contract
tests pin its digest (`contract/tests/fixtures/projections/declaration.sha256`).

- Admission: `projections::ports` resolves each projection's rank selection
  and checkpoint source bindings from the declaration. `SourcePrecision`
  (`compiler/src/precision_source.rs`) imports every bound source through the
  SDK importer and refuses any tensor nothing accounts for. Checkpoint-wide
  refusals are the declaration's support rules, applied by
  `projections::support`.
- Offers: `contract/src/sm120/generated.rs` generates the precision
  offers from the declaration for every geometry and topology through
  `kernels_sm120::precision::generated`. `bundle/projection-adoptions.json`
  keeps the `qwen38_precision_*` symbols recorded in `kernels.lock`, with
  their cubin placements, and `bundle/projection-schedules.json` holds the
  schedule overrides. `contract/examples/sm120/qwen38_projection_adoptions.rs`
  prints the ledger with any new discoveries appended.
- Execution: `bundle/src/projections.rs` authenticates the precision
  component once per instance and binds each rank's views;
  `bundle/src/lower/projection.rs` lowers every declared projection through
  `bundle_runtime::projection::launch`. Generated recipes run the shared
  launch sequence and the registered variants their own launchers; an
  embedding or n-gram lookup without a bound precision use runs the existing
  lookup launcher (`projections::legacy_lookup`).

The port onto the declaration was byte-identical (2026-10-06): `kernels.lock`,
the fake-host journals and snapshots for the FP8 checkpoint and NVIDIA's
mixed export at TP4, and the `qwen38-load-direct digest` ledgers did not
change.

An unused format mix inside the declared offers is data only, as the
[export recipe](testing.md#export-recipe) proves. The prepared profile's
weight contract in the native note follows content, not source files: the
fixed views, a table of view alternatives and the packed layouts' ids and
versions (`prepared::admitted`). For each operation port and each generated
or registered capability of the native inventory that binding can match to
it, the table holds the views of the port's operation without a static
activation scale and, where binding accepts one, with it; it also holds the
F32 form of each norm an export may store in F32. A change to any of these,
a layout version alone included, changes the weight contract, and an edit
that changes none of them leaves it unchanged. Tests check the table
against both mixed exports and crafted components, not against every
component create admits. The drafter's expert operations, their ports and
the MTP layer's BF16-only dense compositions are declaration data. Model
code still holds the view derivation and binding checks in
`contract/src/prepared.rs`, the registered fragment expert adapter that
serves FP8 block experts (`precision::legacy_expert_projection`), the
drafter's view membership in `contract/src/views.rs` and
`compiler/src/placement.rs`, and the registered n-gram hash builder: they
describe Qwen's rank-image layout and registered kernels, which the
declaration does not. CPU tests generate MXFP8 grouped and NVFP4 dense
offers from a declaration that offers them; neither has lock records,
adoptions or GPU evidence.

Both mixed NVFP4 exports serve at TP2 and TP4 with MTP. Their formats,
F32 norm weights and qualification results are in
[quantized checkpoints](quantized-checkpoints.md#qwen-coverage).

## Packaging

```sh
qwen38-package --checkpoint "$CHOPINFER_QWEN38_CHECKPOINT" \
  --module target/release/libbundle_qwen38.so --output DIR \
  [--drafter none|mtp] \
  [--encoders on|off [--image-positions grid-3axis|sequential]] \
  [--activation-policy auto|checkpoint|bf16] [--precision-component FILE[,FILE...]]
```

The package is computed from the checkpoint's headers and documents and the
native module. The only tensor payloads it reads are F32 norm weights, which
preparation checks are finite (`CheckpointSource::import`) and refuses by
tensor and element otherwise; the weight plan that loads the arenas reads
headers only. `chopinfer-model-qwen38`
emits the same package for `chopinfer serve --model DIR`, choosing MTP where
the rank count has an MTP profile and the precision executes it, and the
encoders when the checkpoint has the vision tower. Serve an explicit package
with `chopinfer serve --bundle DIR --checkpoint C [--profile NAME]`.

One package, bundle ID `qwen38.flash.v2`, serves every topology
`qwen38-flash-tp{N}` the checkpoint's precision serves, and the packager
prints each topology it leaves out (`left out ...`). A topology is left out
when its operations are valid and admitted but the supplied kernels have no
composition of them at that rank count (for example an expert format whose
grouped kernels are built for TP4 alone); an invalid or inadmissible
operation, or a
composition missing only some phase or capacity, fails the package.
Requesting a left-out topology fails (`qwen38-flash-tp2 is not served with
this checkpoint's precision`). The package declares, from
`model_qwen38_contract`:

- the text image as a planned component (`rank-images`, role
  `qwen38.rank-images.v1`, format `qwen38.rank-image.v1`); the weight plan
  supplies it and one runtime index per rank
  (`chop.bundle-metadata.qwen38.rank-index.r{rank}.v1`), which the bundle
  checks at create against the identity the host serves for that rank's
  image (`chop.compiled-image-identity.v1`) before opening it, and the
  package requires `chop.weight-plan.v1`;
- per topology, the precision contract resolved from the checkpoint under
  `--activation-policy`, or the one of `--precision-component` that records
  that topology (`chop.bundle-metadata.qwen38.precision.v1`, component
  `precision.qwen38-flash-tp{N}`); with supplied components only the
  topologies they record are served;
- the paged KV layouts `qwen38.qsa-paged-bf16.v1` (preferred) and
  `qwen38.qsa-paged-fp8.v1`, selected with `serve --kv-cache-dtype`, and the
  QSA ring `qwen38.qsa-ring.v1` as state per slot;
- request state per slot and rank: `qwen38.linear-conv.v1`,
  `qwen38.linear-delta.v1` and `qwen38.ple.v1`, with
  `chop.recurrent-state.v1` and `chop.request-retirement.v1`;
- `chop.serving-kv.v1`, 8192 prefill tokens per step, prefix caching with KV
  offload, and a sequence maximum per topology
  (`model_qwen38_contract::package::maximum_sequences`, at most 256);
- `chop.prompt-rows.v2`: F32 prompt-row logits, up to 512 per prefill phase;
- `chop.speculative.v1`, up to 7 draft tokens in at most 32 drafting items.
  With `--drafter none` (the default) the host drafts, which serving enables
  only with `--speculative-tokens`;
- at TP2, `chop.host-resident-views.v1` for the n-gram table;
- a minimum device memory per rank and topology
  (`model_qwen38_contract::package::minimum_device_memory`).

With `--drafter mtp` the package serves at most 128 sequences
(`MTP_QUALIFIED_SEQUENCES`) and declares instead the KV layouts
`qwen38.qsa-paged-bf16-mtp.v1` and `qwen38.qsa-paged-fp8-mtp.v1`, whose pages
and ring also hold the drafter's layer, and the bundle as drafter with the
drafter component `rank-images` (the drafter's weights are views of the text
image). Mixed precision runs the drafter when its MTP layer has a composition
for every format it stores, BF16 dense projections and experts the main
layers' grouped compositions bind, as both mixed NVFP4 exports do. Otherwise
`--drafter mtp` is refused, naming the tensor and its format, and
`chopinfer-model-qwen38` selects the autoregressive profile. The bundle
checks the drafter's views at create (`views::validate_drafter`).

With `--encoders on`, packaging refuses a checkpoint without the vision tower
or with a quantized or differently shaped vision tensor. The package then also
declares the encoder image as a planned component (`encoder-rank-images`,
role `qwen38.encoder-rank-images.v1`, format `qwen38.encoder-rank-image.v1`,
one index per rank `chop.bundle-metadata.qwen38.encoder-rank-index.r{r}.v1`),
`chop.media-inputs.v1` and `chop.position-axes.v1`, the frontend's
`multimodal` section, and the encoder image, buffers and media arena in its
minimum device memory (`package::vision_bytes`). Without `--encoders on` the
package is exactly the text package.

Image rows take HF's 3-axis grid positions (`grid-3axis.v1`). The pinned vLLM
gives every axis the token index instead; `--image-positions sequential`
packages that behavior for comparison, without `chop.position-axes.v1`.

`qwen38-load-direct digest --image encoders` records the planned encoder
bytes, and `qwen38-load-direct verify --image encoders --reference LEDGER`
shows a change left them unchanged (see [direct loading](direct-loading.md)).

### Prefill overlap

Prefill overlap is on by default at TP2 and TP4. The per-rank defaults are:

| Ranks | Reserved SMs | Communication grid | Minimum phase tokens |
| --- | ---: | --- | ---: |
| TP2 | 24 | Planned grid, no clamp | 4096 |
| TP4 | 16 | 80 CTAs | 1024 |

Other rank counts stay off by default. `CHOPINFER_QWEN38_PREFILL_OVERLAP=0`
disables overlap. `=1` forces it on; untuned rank counts use the TP4
settings. Unset or other values select the rank-specific default.

Each eligible chunk splits near its middle; crossings require an aligned
prefix and whole 64-row delta chunks. Unaligned continuing items stay
intact, or the phase runs eagerly. Attention, linear attention, PLE and MoE
run on compute, while each complete HC reduce-scatter/gather pair runs on
a separate communication stream.

`CHOPINFER_QWEN38_OVERLAP_RESERVED_SMS`, `CHOPINFER_QWEN38_OVERLAP_COMM_CTAS`
and `CHOPINFER_QWEN38_OVERLAP_MIN_TOKENS` override the defaults. Reserved
SMs are removed only from verified persistent fragment-prefill GEMM and
MoE grids. `COMM_CTAS=0` keeps the planned grid; a positive clamp must be
an 8-CTA cluster multiple greater than 8 and pass residency checks.
The token threshold applies to the whole phase, including aggregated
requests. Smaller or unsplittable phases use the eager path. Scored prompt
phases retain the 512-row limit and stay eager under these defaults.

On October 4, 2026, TP4 engine prefill throughput improved 17-30% at C1
and 27-29% at C8 across 2K, 8K and 32K prompts; TTFT fell 15-23%.
TP2's 4096-token threshold preserved 2K C1 throughput, improved 8K/32K C1
by 10%/7%, and improved C8 by about 5%; TTFT was unchanged to 9% lower.
Paired C1 outputs matched. The tuned settings avoid the original 24-CTA
clamp, which limited the HC inject/norm workers and regressed throughput.

The enabled path doubles comm-visible row backing and the final head input, and
allocates two input/side arenas, a PLE history
spare and per-rank communication streams/events. Row views remain disjoint
while a reduction reads them; compute-only expert workspaces and collective
epochs remain shared. PLE history and linear state continue from half 0 to
half 1. Each anchor is captured by its containing half. Splits avoid anchor
frontiers so the linear kernels write the anchor state; a refused split runs
the whole chunk. Media source rows
and diagnostic taps retain their whole-chunk row mapping. The final mixer
runs per half; head gathering and the drafter run once over the original
rows. Failures drain both lanes before rollback, or poison the instance if
draining fails. Device bitwise and timing jobs are required to qualify this
path; CPU journals establish schedule and resource invariants only.
For a diagnostic bundle with taps enabled, `CHOPINFER_QWEN38_OVERLAP_EVIDENCE=1`
adds raw prompt-logit shards and the live items' KV pages, QSA ring, PLE,
convolution and delta-state bytes, including their prompt-ending anchor entries under each tap step. These synchronized
captures serve bitwise comparisons and are excluded from timing runs.

### Frontend

`src/frontend.json` is the text frontend and stays byte-identical.
`src/frontend-multimodal.json` holds the image section and its labels;
`frontend::multimodal_component()` appends the section as `multimodal` and
the labels as the format's `media`. Together they declare:

| Field | Value |
|---|---|
| arena | 65,536 rows of 2560 BF16 in pages of 64: four of the largest images at once; a cache, not a per-request limit |
| encoder | `qwen38-vision`, F32 patch rows of 1536 columns, at most 402,653,184 bytes (65,536 rows) |
| image | `image_url`, at most 64 items per request (`--max-images-per-request`), 64 MiB each, 16,384 tokens each |
| recipe | `image.patch-grid.v1` as `fixtures/qwen38/image-processor.json`: factor 32, 65,536 to 16,777,216 pixels, no small-side upscale, `bicubic-antialias`, patch 16, temporal 2, merge 2, 1/255 scale, mean and std 0.5 |
| render | `<|vision_start|>`, `<|image_pad|>` per token, `<|vision_end|>`; with `add_vision_id`, `Picture N: ` before each image |
| positions | `grid-3axis.v1` |

The host decodes images of up to 89,478,485 pixels (PIL's
`MAX_IMAGE_PIXELS`) and the recipe resizes them to its budget, as vLLM does
([request surface](serving-request-surface.md)).

## Long-context qualification

The full correctness run includes `long_context_128k` at the default and DCP 2
configurations, alongside the existing long-context gate. Its frozen prompt
has 131,072 token IDs and scores the final 1,024 positions against HF F32 with
exact dequantization. Set `CHOPINFER_QWEN38_LONG_REFERENCE` to the directory
containing its byte-pinned `reference.json` and `reference.workload.json`.
The reference completed on 2026-10-12 in 2,570.9 seconds with a 1,024-token
prefill chunk. Producer, checkpoint, workload and mode bindings authenticate
its complete finite score rows. This is reference evidence; candidate serving
still has to pass the check.

Only this check enables the explicit floor-only teacher policy. It uses the
existing 1,024-position limits: mean absolute logprob error 0.10, decisive and
near-tie disagreement 0.08, and total disagreement 0.10. The reference contains
only the exact F32 variant. Existing references, spread policies and accepted
seed bytes are unchanged. The DCP 2 check also requires the existing TP4,
BF16 KV and context-parallel-size-2 serving predicates.

## Context parallelism

At TP4 ranks {0, 1} hold KV head 0 and ranks {2, 3} KV head 1, so at size 1
each pair stores the same K, V and QSA block keys twice. The TP4 topology
declares context-parallel size 2 ([context-parallel attention](context-parallel.md)):
each pair is a group, and a sequence's pages alternate between its two
ranks. TP2 ranks hold distinct KV heads and declare no size above 1. Only
geometries listed in `context_parallel::GEOMETRIES` (flash) declare
size 2; any other geometry builds and serves size 1.

Per full-attention layer, MTP layer included (`bundle/src/lower/context_parallel.rs`),
only a row's owner writes its K, V and block key, each rank selects over its
own block keys and the pair merges the selections into QSA's block list, the
pair gathers the queries, each rank attends both ranks' heads over its own
pages, and owner gathers and a merge give each rank its heads' output. The
entries are the model-neutral owner-routed writes, head-sum selection, block
merge, selected partials and partial merge, built in their own lock family
(`bundle/kernels.lock.d/context_parallel.lock`). The raw-key ring, linear and
PLE state stay per slot on both ranks. Create serves size 2 since the gate
below passed (`bundle/src/context_parallel.rs`, `pub const SERVED: bool =
true`); turning the switch off makes create refuse sizes above 1 again. DCP 2
is served and gated, not release-qualified: `qual.json`'s other checks have
not run at size 2.

`profiles.json` declares `tp4-ar-dcp2` and `tp4-mtp-dcp2`: `tp4-ar` and
`tp4-mtp` with `context_parallel_size` 2, experimental, 32 sequences and no
qualified sequence count. `serve --model DIR --context-parallel-size 2` serves
the same settings (with `--speculative-tokens 0` for AR).

Each rank of a group allocates the group buffers (`RankBytes`, 113,731,012
bytes: gathered queries, pairs per bucket, two send and two received partial
blocks of 2,048 rows, the selection workspace and the gathers' flags) and
drops QSA's score rows, which the group's selection does not read (512 MiB at
size 1, `buffers::score_bytes`), so it allocates 423,139,644 bytes less than
at size 1 and every page holds 128 tokens of the group. Measured pages per
rank (the startup log's `serving KV capacity`, TP4, BF16 KV, KV memory
fraction 0.9, 32 sequences, both sizes from one install, 2026-10-07):

| Profile | DCP 1 pages | DCP 1 tokens | DCP 2 pages | DCP 2 tokens | Tokens, DCP 2 / DCP 1 |
|---|---:|---:|---:|---:|---:|
| AR (`tp4-ar-dcp2`) | 42,464 | 2,717,696 | 43,496 | 5,567,488 | 2.049 |
| MTP (`tp4-mtp-dcp2`) | 38,761 | 2,480,704 | 39,759 | 5,089,152 | 2.051 |

The capacity planner (`size_two_drops_the_score_rows_and_holds_over_twice_the_tokens`)
stays conservative by 5,000 to 6,600 pages per rank at both sizes. FP8 KV and 64
sequences are not measured at size 2.

The `dcp2` configuration of `qual.json` (`--model qwen38.dcp2`) declares
the DCP 2 gate: `qual.json`'s teacher, long_context, spec_verify,
verify_exact, prefix_exact and preemption_exact checks with their limits and
adopted references, the 128K long-context check (added after the
2026-10-07 run below), and the shared determinism check, each requiring
TP4, BF16 KV and context-parallel size 2 of the server and every record.
The default configuration's cohorts do not run determinism. Its preemption
pair records at concurrency 1, where
[preemption resume](numerics.md#preemption-resume) promises bit-exact
resumes, as Qwen3.8's DCP 1 qualification did; at 16 the decode bits also
follow batch composition, at size 1 as at size 2.

The full cohort passed on 2026-10-07 against the adopted references (one
install and qualification run, MTP server at 32 sequences unless noted):

| Check | Result |
|---|---|
| teacher | 4,176 positions, mean abs dlogprob 0.055 against exact (limit 0.1), top-1 decisive 0.0065 (limit 0.08); the BF16 reference spread is 0.066 |
| long_context | 1,024 positions, mean 0.021, top-1 decisive 0.0049 |
| spec_verify | copy and natural sets 16/16 requests exact (961 and 701 positions), mean abs dlogprob 0.0 |
| verify_exact | k3 pooled decode against AR: 4,106 positions, row against AR gap 0.0, no top-1 flips |
| prefix_exact | 2,084 positions and 1,939 prompt-end positions bitwise equal to cold, every cached count as the 128-token stripe gives it, acceptance and media pass |
| preemption_exact | 36/36 requests bit-identical, 1,873 positions after the first of 59 preemptions, every one restored |
| determinism | AR against MTP, cold repeats and prefix hits equal over 11 requests and 392 tokens |

## Tests

Compiler tests cover strict views, placement, the weight plan, packaging and
frontend goldens; run the [CPU gate](testing.md). Model quality and
performance need separate GPU qualification.

Package emission derives checkpoint defaults and compiled limits through
`Geometry::model_options(checkpoint_runtime)`. Flash emits no model options
and retains its package manifest bytes. The current bundle does not consume
model options at create, so geometry declarations must keep `overridable`
false.

## Same-architecture checkpoints

A checkpoint added through the
[shared recipe](adding-a-model.md#same-architecture-checkpoints) also adds its
per-checkpoint goldens under
`models/qwen38/compiler/fixtures/checkpoints/<name>/`.

### Shape domain

The compiler admits indexer heads 4, 8 or 16 at dimension 128, attention head
dimensions 128 or 256, and rotary widths 32 or 64 with M-RoPE sections
[6,5,5] or [11,11,10] respectively. QSA uses block 4 and 256 or 512 kept
blocks. Linear key and value dimensions stay 128: the delta rule instantiates
one dimension, the norm requires 128, and key-dimension-256 prefill exceeds
shared memory. Direct dense verify requires an even local group, so the
served domain admits 8, 16, 24 or 32 total Q heads at head 128 and 24 at head
256; head-256 group-8 dense prefill and verify are outside qualification.
Every served topology must also satisfy the per-rank group, fold-scratch and
FP8-prefill limits in the [kernel library](kernel-library.md#engines).
Theta must be finite and positive in F32. Outside 1e4 and 1e7 it follows Rust
F32 `powf` and reciprocal with no Torch bit-parity guarantee, so qualify its
RoPE table against the reference separately.

Admission in the domain does not add a shape to a package; it also needs a
compiled geometry. `fixtures/qwen38/geometries/synthetic-wide` (head 128,
rotary 32, eight indexer heads, the real vocabulary) has opt-in catalog and
build checks and is not GPU qualification.

The [kernel library's instance tables](kernel-library.md#engines) list the
GPU-qualified instances. `kernels_sm120::attention::qualification` bounds
attention admission, and a lab test requires each of its tuples in the
compiled lab catalog. Adding a tuple needs GPU oracle and declared invariance
evidence, or SASS identity to an already qualified entry, in the same change;
a CPU catalog build does not establish it.

`text_config.indexer_budget` is runtime-tunable but not overridable: the
kernels compile its keep count, so `models/qwen38/contract/build.rs` refuses
an overridable declaration and package emission writes no override for it
(see [Tests](#tests)).
