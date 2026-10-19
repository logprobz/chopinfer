# Decode producer fusion

## Inventory, 2026-10-19

The Flash TP4 decode profile contains 25 complete forwards on each of four
GPUs, with 1,713 launches per forward per GPU after excluding sampling and
the truncated final forward. Its 40 layers contribute 1,700 launches;
initialization contributes nine and the head/logits suffix four.

The profile predates the split FP8 projections, paired shared gate/up,
cached exact-F32 projections, active expert schedule, split HC mixer and
parallel ceil entries. The first combined decode binding
had 2,382 launches at one row per GPU. The later ticket projections and
streamed experts reduce its unfused total to 2,332. Historical kernel
durations cannot predict fusion savings: PDL overlaps launches, and their arithmetic and
scheduling have since changed. No GPU speedup is claimed here.

The chain below follows `bundle/src/program.rs`. Each arrow is a launch
boundary in the historical profile unless a row states otherwise.

| Part | Per-layer chain |
| --- | --- |
| Attention input | Lagged HC mix/collapse -> BF16 RMSNorm -> optional F32 widen -> ceil E4M3/32 |
| Query | `wq_a` -> F32 q_norm -> ceil -> `wq_b` -> tail RoPE and BF16 query |
| Window key | `wkv` -> F32 norm and tail RoPE -> ceil -> ring store |
| Full source | Compressor key/gate projections -> pooling/norm -> index key projection/write -> compressed latent write |
| Full or reindex | q_norm requantization -> index query projection/preparation; index-weight projection -> score/select |
| Attention output | Attention split/merge with inverse tail RoPE -> ceil -> grouped `wo_a` -> ceil -> `wo_b` |
| Attention residual | Reduce-scatter -> all-gather -> BF16 round -> HC expand |
| MLP input | Lagged HC mix/collapse -> BF16 RMSNorm -> F32 widen -> ceil |
| Routed MLP | Exact-F32 router -> routes -> gate/up with weighted SwiGLU -> ceil -> down |
| Shared MLP | Gate/up -> SwiGLU with ceil -> down; current decode pairs gate/up and overlaps the routed branch |
| MLP residual | Routed sum with shared partial -> reduce-scatter -> all-gather -> BF16 round -> HC expand |
| Engram | Table decode -> ceil -> `wkv` -> residual-stream gate, before the attention mixer |

| Layers | Role | Historical launches per layer |
| --- | --- | ---: |
| 0 | Window | 40 |
| 1 | Window with Engram | 44 |
| 2, 8 | Full source, ratio 2 | 55 |
| 14 | Full source, ratio 2, with Engram | 59 |
| 20 | Full source, ratio 1, candidate publication | 55 |
| 24, 28, 32, 36 | Reindex | 48 |
| Remaining 30 | Reuse | 40 |

There are 290 standalone ceil launches: 80 input, 48 query, 40 window,
40 attention output, 40 output-A, 40 routed intermediate and two Engram.
There are 80 input norms, 40 query norms, 49 widens (48 norm outputs and
one embedding), 81 rounds, 80 HC expansions and 80 lagged mixers.

The live-position indexer variants serve four Full sources and four
Reindex sources. Index metadata comes from one `index_upload` per step
and one upload per rank. Reuse layers perform no new index query or
selection.

## Fused chains

`kernels-sm120` supplies the shared implementations; the model registers
them in independent `norm_quantize` and `window_fused` lock families.
Every pre-existing lock remains unchanged. Only decode selects the fused
entries; prefill and verify keep their previous chains.

| Chain | Previous launches | Fused launches | Saved per Flash step/GPU |
| --- | ---: | ---: | ---: |
| Input RMSNorm, optional widen, ceil | 208 | 80 | 128 |
| Query RMSNorm and its first ceil | 80 | 40 | 40 |
| Window norm/RoPE, ceil, store | 120 | 40 | 80 |
| Total | 408 | 160 | 248 |

The norm/window stage lowers the one-row TP4 journal from 2,332 to 2,084 launches per
GPU in both KV layouts. The six complete topology/layout journals change
only the declared fusion launch families, plus their module and symbol
registration. The new launch-count regression fails on the previous binding.

The wrappers reuse the existing device norm and store bodies, keeping
their reduction order, F32 operations, BF16 boundaries and status commits.
The original norm's early PDL signal is deferred inside the new translation
unit until every epilogue finishes. The ceil epilogue validates all groups
before publishing codes; a failure writes positive-zero codes with unit
scales. Norm outputs and optional widened rows keep the old chain's bytes,
including failed BF16 norm outputs. Each fused entry has zero backend
workspace; its only new scratch is CTA-local shared memory.

Window storage retains both physical layouts, independent plane strides,
circular overwrite guards, invalid-position skips and preserved failed
cells. Quantization still sees F32 rotated values and rounds them to BF16
before amax. The BF16 comparison ring stores the exact decoded trained
codes, not unquantized normalized values.

Eight later indexer query quantizations remain. Other producers can flag
status between the first query use and the indexer, so reusing earlier
codes would change the quantizer's failure outputs.

## Expansion and the next lagged mixer

Decode now composes 77 uninterrupted expansion/mixer pairs. Its first
kernel computes the original BF16 expanded values and the next projection's
independent 128-element chain sums. It recomputes expanded values for the
RMS statistic so each original lane retains its complete sequential FMA
chain. Its second kernel folds projection chains in ascending order and
runs the original RMS butterfly and Sinkhorn expressions in each of ten
512-column CTAs. Each CTA writes its own collapsed columns; only CTA zero
publishes the shared `pre`, `post` and `comb` outputs.

The first attention, the two Engram boundaries and the final head retain
the original standalone operations. Prefill and verify are unchanged.
Every fused pair still takes two launches. The purpose is to distribute
work from the original single finishing CTA; no measured latency gain is
claimed. This stage keeps the one-row TP4 count at 2,084 launches per GPU,
248 below the composed baseline.

The new `sinkhorn_hc_expand_mix` lock family preserves every existing
lock. It uses 16,384 bytes of temporary workspace per row, rewritten on
every call, with no tickets or atomic protocol. The previous `post` and
`comb` may share storage with the next values because only the first
kernel reads them and only the second writes them. All other roles and
workspace must be distinct. Both cross-stream workspace audits pass with
the composed decode schedule.

The staged GPU matrix compares all five outputs against the old expansion
and split mixer over six row buckets, eleven input patterns, zero/partial
live counts, eager calls and repeated graphs. It poisons outputs and
workspace, checks inactive rows and the old ticket reset, and includes
byte mutations and changed collapse weights. Production comparisons reuse
the `post`/`comb` buffers. Timings use immutable previous weights and first
check that this layout gives the same bytes, so timing contains only the
two kernels. All mismatching cases are collected before the test fails.
A two-GPU, 250 W comparison on 2026-10-19 passed all 66 cases, 30 byte
controls and six changed-weight controls per GPU. At one row the original
two-kernel pair took 32.736/32.768 us and the new pair 30.720/30.720 us,
a 2.016/2.048 us saving. Across 77 boundaries that extrapolates to
0.155/0.158 ms per step. These are whole-pair event timings; individual
kernel exclusive times and serving-step savings were not measured. All
timing samples flagged SM clock drift, and the four-row result on one
GPU showed almost no gain.

## Rounded expansion and replicated embedding

The next decode form folds the all-reduce's final BF16 round into the
expansion/partial task at 77 boundaries. A typed deferred-round interface
returns the original collective prefix and its F32 sum view. The two
collective launches, their state bindings, ordered F32 rank sum and refusal
checks remain unchanged. The fused task still writes the original rounded
intermediate. Its 16-cell Sinkhorn matrix uses warp shuffles to read the
previous iteration's cells; every row/column sum and division keeps its
original order. Larger matrices retain the shared-memory form.

Decode embedding reads the owning immutable vocabulary shard directly on
every rank, retaining the existing weight layout and peer grants. One task
writes the local gather, its F32 widening, the rank-ordered sum rounded to
BF16, and the four residual streams. Signed zero and NaNs retain the old
collective semantics, checked against the oracle for every BF16 encoding
at every owner position for one, two, four and eight ranks. The embedding
stage boundaries remain in the program. Prefill and verify keep the old
chain, including image injection before stream broadcast.

The round task removes 77 launches and the embedding task removes five.
The one-row TP4 journal has 2,002 launches per GPU in both KV layouts, 330
below the composed 2,332-launch baseline. Both kernels use virtual CTA
indices and caller-owned shared arenas; only entry wrappers wait for or
signal grid dependencies. Existing locks remain unchanged, with separate
`sinkhorn_hc_round_expand_mix` and `embedding_replicated` families.
GPU parity and timings for these new tasks remain pending. The staged job
compares the original round/expand/mixer chain, and the actual two-GPU
embedding collective chain, against the fused tasks. Dedicated Nsight
runs attribute exclusive time after sorting kernel starts, advancing the
frontier by each kernel's end. Component savings extrapolated by boundary
count are reported separately from a serving-step measurement.

## Remaining boundaries

| Candidate | Opportunity | Constraint |
| --- | --- | --- |
| Attention merge -> ceil | 40 launches | Preserve BF16 output boundary and whole-row status before publishing any group |
| `wo_a` merge -> ceil | 40 launches | Current merge CTAs own slices; one bad slice must zero the complete quantized row |
| Routed weighted SwiGLU -> ceil (L5) | 40 launches | Streamed merge folds gate/up chains in order; quantizer failure covers every route of a token |
| Shared gate/up merge -> SwiGLU+ceil (L5) | 40 launches | SwiGLU already includes ceil; paired banks need common ownership and row-wide validation |
| Engram decode -> ceil | Two launches | Preserve decoded BF16 operands and row status |
| All-gather -> BF16 round | 81 launches including embedding | Keep F32 rank-order sum and the final BF16 boundary, with peer completion before consumers |
| BF16 round -> HC expand | 80 launches | Preserve rounded sublayer output before residual mixing |
| Query projection merge -> RoPE (L12) | 40 launches | Row-wide validation spans all heads while merge CTAs own output slices |

HC expansion already performs the residual addition. Its stream-order FMA
sum, separately rounded `post*y` multiplication and addition, and final
BF16 conversion must stay in order. Engram interrupts two expand/mix
boundaries; the final expansion feeds the head. A two-kernel mixer also
needs the original projection chains, RMS fold, 20 Sinkhorn iterations,
publication fences and ticket reset semantics.

## Verification

CPU tests check preparation, alias and extent refusals, exact argument
composition, independent window planes, zero norm/window workspace and
separate HC partials. CUDA compile audits require zero spill/local and stack bytes. Fake-host tests count the
removed launches and compare physical router/shared-expert workspace
ranges at every captured MoE fork. That safety test is separate from the
kernel comparisons and must pass before serving qualification.

The model lab stages old-chain versus fused GPU comparisons for rows
1, 2, 4, 8, 16 and 32. They compare every intermediate, final output and
status bit under randomized and special-value inputs, sticky flags,
inactive rows, eager launches and poisoned graph replays. Window tests
also cover both storage layouts, circular and staged addressing, distinct
strides, invalid positions and retained cells. Byte perturbations of every
output role and a changed epsilon must fail the equality gate.

The pair job uses frozen binaries and cubins, verifies every declared
export before execution, checks the 250 W power cap before and after,
and records kernel Xids. Timing uses the original kernel-lab `probe_sm120`
module authenticated against its lock. These tests have been compiled and
staged; GPU bitwise equality, performance and serving qualification still
require the idle-GPU run.

The first pair run exposed an invalid comparison fixture. Its baseline
RMSNorm used the prepared 32-row grid with a one-row allocation and an
eight-row live count. Excess CTAs accessed beyond those buffers, invalidating
the comparison. The fixture now narrows the baseline grid to its allocated
row bucket, matching the production binding. CPU admission rejects zero or
oversized buckets, live counts above the allocation, and mismatched grids
before any GPU session starts. Replays exercise full, zero, partial and
restored full live counts within the allocation. Unused widening and BF16
scale outputs remain poisoned guards, checked on both arms.

Parity checks record every mismatching role and case, including input
pattern, live count and eager or graph replay. Both suites finish their
case matrices before failing; driver errors still abort the affected suite.
The pair worker attempts both suites even when the first fails. Timings are
omitted for case/row combinations with a mismatch. The invalid first run
provides no GPU correctness or performance qualification.

The composed binding passes both workspace isolation audits: all 160 TP4
router/shared-expert forks have disjoint scratch, and the complete branch
span audit passes at TP2 and TP4. The tests recognize the paired ticket
admission and the fused MLP input quantizer. Rebased text and media journals
were audited by expanding the norm/window and HC fusions: every other
launch retained its grid, scalar arguments and span extents under a
consistent bijection of host-issued handles.

The subsequent [wide mixer measurements](wide-mixer-results.md) pass bitwise
checks but regress at every tested bucket. They supersede the experimental
wide mixer's pending status above; its entries remain excluded from serving.
