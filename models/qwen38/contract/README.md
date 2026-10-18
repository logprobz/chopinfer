# Qwen projection declaration

projections.json declares Flash checkpoint tensor ownership, source shapes and
formats, named operands, TP placement, layer and MTP materialization, use
profiles, fusion output buffers, lookup companion scopes, encoder ownership and
checkpoint-wide admission. Metadata tests bind the FP8 export and two mixed
layouts synthesized from it (nvfp4-routed-fp8-mtp and mxfp8-dense-nvfp4)
through the shared SDK header importer. The MXFP8 layout's 100 F32 norm
sources retain F32 identity views. Serving continues through the existing model
code; the providers decode typed descriptors.

Shared NormContract records cover HC/PLE standalone norms with shuffle_down and
MTP pre-FC norms with xor_then_serial. The records retain their exact epsilon,
affine layout and admitted weight dtypes. When an HC weight is consumed by the
fused inject-norm epilogue, that kernel owns the arithmetic and selects the
weight variant from the bound view dtype.

Attention Q/K and indexer norms, and the linear gated norm, declare their owning
consumer, source shape, dtype admission and placement. Their consumers own the
reduction, reciprocal, intermediate rounding, RoPE and sigmoid gating. Those
side records carry no shared normalization arithmetic.

The legacy fused implementations retain phase-dependent input schedules and
sigmoid/stream-mean input arithmetic. The n-gram provider retains wrapping
unsigned hash and history addressing. FP8 fragment arithmetic remains an
explicit legacy implementation. Shared fields carry geometry, placement,
output buffers, source-format numerical recipes and uses; the existing model
kernels own the remaining implementation policy.

A shard template expands the n-gram table's 128 physical sources while keeping
variable vocabulary head ranges explicit. Format-keyed maps select FP8 fragment
experts and the BF16 linear-output implementation; NVFP4 experts and MXFP8
linear output retain their native source-format path. Shape matching never
selects a legacy implementation.
