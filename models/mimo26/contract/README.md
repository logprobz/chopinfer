# MiMo projection declaration

projections.json declares Flash MOPD text, MTP and DFlash tensor ownership,
source shapes and formats, named operands, TP and side placement, segmented
QKV heads, layer materialization and execution aliases, use profiles,
assemblies, activation inputs, numerical recipes and encoder-owned tensors.
Metadata tests bind these roles through the shared SDK header importer.
Serving continues through the existing model code; the providers decode typed
descriptors.

DFlash Q/K head norms declare their owning kernel, replicated 128-element
source weights and BF16 admission. Their kernels multiply by shared plain
affine weights across heads, which the shared GroupedRmsNorm cannot lower.
Standalone row norms and fused residual norms also declare their owning
consumer, source shape, dtype admission and placement. These kernels own the
reductions, epsilon, affine application, reciprocal and intermediate rounding.
No MiMo norm side carries shared normalization arithmetic.

The model retains its exact/stream environment-switch implementation policy
in src/serving.rs and the existing kernel selectors. Profiles retain requested
buckets and exact-row requirements. Legacy fused dense, routing and
combine descriptors retain their arithmetic and ordering. QKV scale regroup
remains a source transform with an explicit topology policy. The DFlash mask
remains an external PT archive stage whose values are outside header evidence.

MTP aliases reuse authenticated source roles without duplicate checkpoint
claims. EH intersects each rank's logical K range with the embedding and hidden
owner domains, then uses owner-relative columns. Format-keyed implementations
select legacy entries from the bound source format. Expert source identity stays
mimo26.source-recipe.v1, matching the current precision builder. Admitted expert
weights use MXFP4; no FP8 fragment-expert implementation is registered. The
existing compiler baseline remains the byte-exact comparison target.

Legacy combine belongs to routed expert-down and MLP reduction. The draft-only
embedding alias owns the mask stage after lookup, before input norm, for Verify
blocks. Its typed adapter preserves signed I32 positions, prefix_len and
token_batch, plus unsigned I32 live_tokens, and resolves the existing native
symbol for each topology. Logical ports and control metadata describe the launch;
behavior qualification remains with the owning provider.
