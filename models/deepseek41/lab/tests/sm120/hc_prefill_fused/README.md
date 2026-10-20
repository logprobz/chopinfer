# Exact prefill HC composition

The matrix now qualifies the served expansion-then-mix tiles with the input
norm and ceil quantizer folded into the same CTA
(`kernels_sm120::sinkhorn_hc::prefill_expand_mix_norm`, lab family
`hc_prefill_expand_mix_norm`, tiles r4c4 and r8c8, one launch) against the
served expand, mixer and norm chain (three launches). Case labels name the
tile as `r{rows}c{chains}n`. Its controls reverse the stream sum, round the
normalized value to BF16 before the weight multiply, and clear the prior
row status. The counts scale with the tiles: 64 cases, 4 invariances and
24 timing records per tile, two memcheck cases per tile, three controls.

The previous matrix qualified the expansion-then-mix entries
(`prefill_expand_mix`, lab family `hc_prefill_expand_mix`, tiles r4c4, r8c4
and r8c8, two launches with the served norm). Each CTA expands its own
rows once with the F32 expand's expressions into the expanded output, then
runs the unchanged prefill pipeline over those rows while they are still in
L2. The projection consumes the flat row in ascending K order, which is
stream-major, so an exact schedule cannot interleave output streams per
column block; the earlier composition below regenerated each panel from
all four input streams and lost at full chunks (d2-0451a).

## Earlier composition (lab only, not in the matrix)

This lab candidate replaces the BF16 stream expansion, lagged mixer and
BF16 input norm/ceil with one kernel. It is not registered for serving.
The query norm fusion is a separate, measured serving change.

Each CTA owns 1, 4 or 8 complete rows. It generates a panel of four 128-K
chains in shared memory, writes the original expanded BF16 output and
projects that same rounded panel. Projection products, four-element FMA
chains, butterfly levels and ascending K fold retain the reference order.
The flattened HC RMS sum retains the original logical 256-thread pack
mapping. Sinkhorn uses the existing prefill pipeline expressions.

After projection and Sinkhorn finish, the CTA overlays its panel arena
with a single-row norm arena. Collapse still starts from positive zero,
folds streams in order with FMA and rounds once to BF16. The input norm
consumes that rounded value, preserves its original square chain and
256-thread tree, multiplies the weight before BF16 rounding, and widens
only when requested. Ceil consumes the rounded BF16 operand, retains its
trained scale grid and commits codes only after whole-row sticky-status
admission. Every intermediate remains observable.

The task takes a virtual tile index and a caller-supplied shared arena.
Only the entry wrapper calls grid dependency instructions. There is no
global workspace, counter or last-arrival state. A CTA caches incoming
pre/post/comb coefficients before publishing potentially aliased outputs.
The panel-to-norm overlay has a CTA barrier, and each row retires before
the norm arena is reused. Concurrent tests use disjoint device allocations.

The frozen corpus contains real checkpoint projection, bias, scale and
norm weights from three authenticated HC boundaries. Seeded prefill
streams and residual rows use captured activation magnitudes. They are
synthetic prefill inputs, not a claimed capture of an 8K serving step.
Coverage includes 1/7-row edges, cone buckets, 8191/8192 rows, optional
widening, exceptional operands, sticky failure state, poisoned outputs,
graph replay, chunk partitions, changed batch positions and coefficient
aliasing. Three mutations reverse expansion order, insert a norm rounding
point or clear sticky status. Both GPUs execute the complete matrix.

The matrix requires 192 case records, 12 invariance checks, three rejected
mutations and 72 paired timing records per GPU. Memcheck requires six
full-8K comparisons. Parsers reject zero-test runs, missing or duplicate
cases, missing timing pairs, numerical failures and cross-GPU digest
differences. Small control rows use the same norm arithmetic even where
that prefill entry is not selected in serving. Timing rows are qualified
prefill buckets and use the current selected mixer and norm entries.

The round 7 cold 32K profile charges approximately 241 ms to HC and
144 ms to norm/quantize, about 60 and 36 ms per chunk. Fusion removes two
launches per eligible boundary. Its intended traffic saving is the
projection's reread of expanded BF16 rows and the norm's reread of
collapsed BF16 rows. Expanded, collapsed, normalized and quantized
outputs still have to be written. Extra expansion operand loads and CTA
lifetime may offset those savings, particularly for the one-row tile.
Only the pair results can establish a win or justify a serving route.
