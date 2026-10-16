# qwen38

Qwen3.8 Flash. `bundle/` (crate `bundle-qwen38`) owns native execution, model
CUDA specializations and `kernels.lock`. `compiler/` (crate `model-qwen38`,
binaries `chopinfer-model-qwen38`, `qwen38-package` and `qwen38-oracle`) owns
checkpoint reading, packaging and frontend tests. `contract/` owns the
shared tensor and topology definitions and the chat format
(`contract/src/frontend.rs`, ported from the pinned chat template), which the
bundle exports. `profiles.json` holds the
[serving profiles](../../docs/serving-profiles.md), `qual.json` the
qualification declaration, and `checkpoint-files.json` the pinned checkpoint's
file digests. Shared kernels live in `crates/kernels-sm120`, fixtures in
`fixtures/qwen38`. Run `nice -n 19 bash ci/cpu.sh` from the workspace root.
See [adding a model](../../docs/adding-a-model.md) and
[direct weight loading](../../docs/direct-loading.md).

The [model guide](../../docs/model-qwen38.md#tests) describes package
defaults and the current create contract.

## Chat frontend

`contract/src/frontend.rs` (`Qwen38Format`) renders the prompt: a Rust port
of the pinned chat template (revision 236dfdf2), one function per template
block, each citing the lines it ports. It writes each image as
`<|vision_start|>`, the pad and `<|vision_end|>`, after a numbered
`Picture N: ` label when the request sets `add_vision_id`
(`compiler/tests/vision_labels.rs`), and refuses `tool_reference` parts. The
bundle exports it with `frontend_format::export_chat_format_v1!`.
`compiler/src/frontend.json` (with `frontend-multimodal.json` for a package
with the encoder images) declares the model-neutral sections and
`format.parser`: the function-parameter tool-call family
(`<tool_call>\n<function=name>\n<parameter=k>\nv\n</parameter>\n</function>\n</tool_call>`),
the `<think>` markers and `<|im_start|>` as the turn opener.

The HF differential corpus (`frontend_cases.py`, `frontend-divergences.json`
and the committed subset `fixtures/qwen38/frontend/corpus*.json`) runs
through `compiler/tests/frontend_corpus.rs` as raw requests: renders and
token ids, refusals, and round trips parsed at every byte split, with the
checkpoint's tokenizer. `python3 -m tools.refrun.frontend_corpus check
--model qwen38` is the coverage and divergence gate, and a template update
follows
[updating the chat template](../../docs/adding-a-model.md#updating-the-chat-template).

## Checkpoint geometry

`geometries/` holds one config per compiled geometry and `supported.json`.
Its `aliases` admit architecture/type spellings, `semantic_rules` constrain
execution semantics, and `quantization_recipe_classes` list admitted exports.
Each `geometries` entry has an `id`, config `file` and `kernel_tag`;
`pinned_names` retains historical kernel identities. `tuning` selects measured
`contract/tuning/<id>/` tables, `limits` bounds runtime fields, and
`runtime_fields` declares bounded checkpoint runtime fields.
Sizes come from the checkpoint's `config.json`. A package selects geometry
through its topology ID and keeps one bundle ID across rank counts.
Follow the [same-architecture recipe](../../docs/adding-a-model.md#same-architecture-checkpoints).
From the workspace root, run:

```sh
bash tests/model-geometry-recipe.sh qwen38
```

The [export recipe](../../docs/testing.md#export-recipe) admits a synthesized
mixed export that neither committed export matches: MXFP8 full-attention,
indexer and shared-expert projections, BF16 linear-attention projections,
NVFP4 experts and n-gram lookup, and weight-only NVFP4 MTP experts. It needs
no tree change:

```sh
bash tests/model-export-recipe.sh qwen38
```
