# Qwen3.8 Flash metadata

This CPU-only crate (`model-qwen38`) owns the pinned configuration, the
complete tensor catalog, TP2/TP4 placement and expert repack recipes, the
tokenizer contract, the text frontend and its image section, the package
emitter (`qwen38-package`, `chopinfer-model-qwen38`) and the CPU dump checks
(`qwen38-oracle`). It reads no tensor payloads and does not qualify a GPU
bundle. Revision: `236dfdf285828023ca3bcd3f37366c58a3469b13`.

The config validates all typed fields against the pinned fixture. Missing
`norm_topk_prob` and `seed` use the HF defaults, true and 1234. Unknown fields
and changed values are refused.

The catalog checks 152,089 names, shapes and dtypes against the shard headers
and the index, without reading tensor data. Every routed expert keeps all 640
intermediate channels; gate and up concatenate into [1280, 2560] with
[10, 20] BF16 inverse scales, and down stays [2560, 640] with [20, 5] scales.
N-gram heads have unequal prime row counts from the HF prime algorithm;
slices cross storage shards and the last rank keeps padding.

The text descriptor serves 262,144 tokens, defaults to xhigh reasoning,
preserves history reasoning, supports medium/low and disabled thinking, and
stops on 248046 or 248044. It declares no media inputs; a package that plans
the vision encoder (`qwen38-package --encoders on`) appends the image section
of `src/frontend-multimodal.json`. The chat format writes the labels around
each image itself; the declaration's `format.parser` section holds only what
the shared parsers read.

Tokenizer goldens load the pinned `tokenizer.json` unchanged through HF
`PreTrainedTokenizerFast`, because the cached Transformers `AutoTokenizer`
reconstructs an older Qwen2 regex that changes combining-mark tokenization.
See [fixtures/README.md](fixtures/README.md) and `fixtures/checkpoints/qwen3.8-flash-next-fp8/artifact-identities.json`
for capture provenance.

Set `CHOPINFER_QWEN38_CHECKPOINT` to the pinned snapshot to run the real
config, header and tokenizer checks. Without it only those file-dependent
tests skip; synthetic catalog accounting and chat text goldens always run.

## Planned images and packaging

A package carries no weights. `qwen38-package` emits one from the checkpoint
and the native module alone, serving 2 and 4 ranks (mixed or NVFP4
precision at TP4 only). The text image and, with `--encoders on`, the vision
encoder image are planned components: the weight plan
(`../contract/src/weight_plan.rs`) places them from the checkpoint with the
layout below, and the host loads them. Commands, flags and package
declarations are in [docs/model-qwen38.md](../../../docs/model-qwen38.md).

Dense BF16 weights use `bf16_mma_fragments_16x16_v1` and routed FP8 weights
`fp8_e4m3_mma_fragments_16x32_v1`; skinny matrices pad their last 16-row
strip, and logical shapes exclude that padding. Expert views have leading
dimension 128, ordered by local expert, gate then up within each expert.
Scale views are expert-major 128x128 block grids widened exactly from BF16 to
F32, as are `A_log` and `dt_bias`. Convolution, normalization, embedding and
hash metadata stay row-major.

N-gram checkpoint shards concatenate by global row into each rank's
contiguous four-head range, 160 FP8 bytes per row, kept in host memory where
the device capacity rule places it (TP2). The consumer subtracts
`ngram_rank_rows_for(geometry, rank, ranks).start` from global hash rows. Q/K/V linear
projections concatenate rank-local Q, K and V rows; attention Q/gate rows keep
their per-head interleaving, and KV heads are replicated on rank pairs.
Recurrent state is allocated by the bundle, not stored in the images.

## Dump checks

`src/oracle.rs` and `src/oracle/` check bundle dumps operation by operation
on the CPU; see [docs/qwen38-oracle.md](../../../docs/qwen38-oracle.md).
Whole-model references come from the GPU reference runner (`../reference.py`).

Direct emission inputs now include the selected geometry. After upgrading,
existing warm direct packages are emitted once with the new reuse key.
