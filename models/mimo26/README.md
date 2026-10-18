# mimo26

The full correctness suite includes [KV preemption swap qualification](../../docs/serving-kv-pool.md#formats-caching-and-offload); offload stays disabled by default.

MiMo V2.6 Flash. `bundle/` (crate `bundle-mimo26`) owns native execution, model
CUDA specializations and `kernels.lock`. `compiler/` (crate `model-mimo26`,
binaries `chopinfer-model-mimo26` and `mimo26-package`) owns checkpoint
reading, image sizes, packaging and frontend tests. `contract/` owns the shared
tensor and topology definitions and the chat format (`contract/src/frontend.rs`,
the checkpoint's chat template ported to Rust), which the bundle exports. `profiles.json` holds the
[serving profiles](../../docs/serving-profiles.md), `qual.json` the
qualification declaration, and `taps.json` the diagnostic taps. Shared kernels
live in `crates/kernels-sm120`, fixtures in `fixtures/mimo26`. Run `nice -n 19
bash ci/cpu.sh` from the workspace root.

`checkpoints.json` names the pinned checkpoints (MOPD, the default, and RL);
registration does not imply numerical qualification. A checkpoint whose
metadata matches no registered lock is served under a local identity with
BF16 KV only
([compatible checkpoints](../../docs/compatible-checkpoints.md#mimo-v26-flash-checkpoints)). `checkpoint-files.json`
records each source file's size and SHA-256, taken from the Hugging Face tree
API at the pinned revision (for ordinary Git blobs, after checking the local
bytes against the Git blob ID). Do not refresh it from an unverified local
directory.

FP8 KV calibration is checkpoint data in `checkpoints.json`. Add
`fp8_kv_scales` to the registered identity, with one finite positive f32
`[K, V]` pair per global-attention layer in layer order, then rebuild the
compiler and bundle. `fp8_kv_geometry` defaults to `v26-flash`; set it to the
compiled geometry ID for another shape. The build validates and deduplicates
tables. The shipped MOPD/RL table keeps `mimo.ga-paged-fp8.v2`; other tables
use that prefix followed by the SHA-256 of their little-endian f32 scale bits.
Preparation advertises only the authenticated checkpoint's table, or BF16
alone without calibration. Create accepts any compiled table of the geometry
by layout ID: its create record carries neither checkpoint identity nor the
package's declared formats. The prepare manifest binds the authenticated
checkpoint to its table. The host selects only a manifest-declared layout and
requires physical KV requests to match it. Scales
reach kernels as launch arguments, so calibration additions need no Rust,
CUDA or kernel lock edits. Derive each scale with
`kernel_core::kv_fp8::calibrated_scale` applied to 16 times the per-layer K or V
amax from an f64 oracle over representative teacher texts. The shipped table
used 33 texts; its provenance is recorded beside `LEGACY_SCALES` in
[the calibration module](contract/src/kv_calibration.rs).
Registration does not qualify the calibration.

A package carries no weights: its text, drafter and encoder images are planned
from the checkpoint. Packaging authenticates the checkpoint's config, index and
tokenizer files against the lock and records the selected identity.
Each topology also carries a precision component built from admitted checkpoint
operands and the projection declaration, which every listed geometry
instantiates with its own layers and QKV layout. Its file digest enters the
checkpoint binding. The bundle authenticates every implementation once before
opening rank contexts, binds each rank's validated views, and lowers main and
MTP projections only through registered native adapters: every use binds a
registered launcher of `bundle_runtime::projection`. The declaration
(`contract/projections.json`) registers five variants:
`mimo26/qkv-scale-regroup.v1`, `mimo26/dflash-mask.v1`,
`mimo26/legacy-fused-dense.v1`, `mimo26/legacy-routing.v1` and
`mimo26/legacy-combine.v1`. Its MTP and DFlash drafter layers are ordinary
stems, and drafter membership stays prefix based (`mtp`, `dflash.`) instead
of the schema's `drafters` field: import pruning keeps drafter side tensors
the kernels consume, such as the DFlash norms, which no drafter field owns.
The inexact and
medium-M alternatives are bound wherever the kernels offer them; the existing
environment switches choose between them at execution. A package without the
component is refused at create; `serve` and `package` rebuild it from the
checkpoint. `bundle/projection-adoptions.json` records the generated identity
and catalog cubin placement of each locked projection entry, and
`bundle/projection-schedules.json` the schedules the generated inventory
(`contract/src/capabilities/sm120.rs`) reads;
`contract/tests/sm120/projection_placements.rs` checks the active adoptions against
that inventory and the cubin positions in `kernels.lock`. See
[serving MiMo](../../docs/serving-mimo26.md),
[adding a model](../../docs/adding-a-model.md) and
[direct loading](../../docs/direct-loading.md).

## Chat frontend

`contract/src/frontend.rs` (`Mimo26Format`) renders the prompt: a Rust port
of the checkpoint's chat template (revision 2479e2d0), one function per
template block, each citing the lines it ports. It writes each image, video
and audio item between its kind's labels, opens over every encoder
selection a package can serve, and refuses `tool_reference` parts. The
bundle exports it with `frontend_format::export_chat_format_v1!`.
`compiler/src/frontend.json` declares the model-neutral sections and
`format.parser`: the function-parameter tool-call family
(`<tool_call><function=name><parameter=k>v</parameter></function></tool_call>`),
the `<think>` markers and `<|im_start|>` as the turn opener.

The HF differential corpus (`frontend_cases.py`, `frontend-divergences.json`
and the committed subset `fixtures/mimo26/frontend/corpus*.json`) runs
through `compiler/tests/frontend_corpus.rs` as raw requests: renders and
token ids, refusals, and round trips parsed at every byte split, with the
checkpoint's tokenizer (`CHOPINFER_MIMO_TOKENIZER_DIR` or
`CHOPINFER_MIMO_CHECKPOINT`). `python3 -m tools.refrun.frontend_corpus check
--model mimo26` is the coverage and divergence gate, and a template update
follows
[updating the chat template](../../docs/adding-a-model.md#updating-the-chat-template).

## Checkpoint geometry

`geometries/` holds one config per compiled geometry and `supported.json`.
Its `aliases` admit architecture/type spellings, `semantic_rules` constrain
execution semantics, and `quantization_recipe_classes` list admitted exports.
Each `geometries` entry has an `id`, config `file` and `kernel_tag`;
`pinned_names` retains historical kernel identities. `encoders` enables encoder
execution; cubin membership shares matching encoder entries across geometries.
`engine_domain` records current shape restrictions. A geometry config must have
Flash's exact leaf set, with `quantization_config` stripped.
Sizes come from the checkpoint's `config.json`. A package selects geometry
through its topology ID and keeps one bundle ID across rank counts.
Follow the [same-architecture recipe](../../docs/adding-a-model.md#same-architecture-checkpoints).
From the workspace root, run:

```sh
bash tests/model-geometry-recipe.sh mimo26
```

The recipe requires a clean committed HEAD, the offline CUDA build environment,
pinned tokenizer files in `CHOPINFER_MIMO_TOKENIZER_DIR` (or
`CHOPINFER_MIMO_CHECKPOINT`), and `mask_embedding.pt` in
`CHOPINFER_MIMO_DFLASH_DIR` (or the checkpoint/tokenizer directory's `dflash/`).
It checks text, DFlash and encoder plans, helper admission and preparation,
calibrated and uncalibrated KV manifests, and TP2/TP4 fake-host lifecycles.
It uses no GPU and does not establish numerical qualification.

Geometry files must have the same execution-field leaves as `v26-flash.json`.
Strip export metadata such as `quantization_config` from the geometry file;
the checkpoint's original config still carries it for admission.
