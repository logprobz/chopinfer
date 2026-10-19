# Decode producer fusion

## Inventory, 2026-10-19

The Flash TP4 decode profile contains 25 complete forwards on each of four
GPUs, with 1,713 launches per forward per GPU after excluding sampling and
the truncated final forward. Its 40 layers contribute 1,700 launches;
initialization contributes nine and the head/logits suffix four.

The profile predates the split FP8 projections, paired shared gate/up,
cached exact-F32 projections, active expert schedule, split HC mixer and
parallel ceil entries. The CPU journal of their combined decode binding
has 2,382 launches at one row per GPU. Historical kernel durations cannot
predict fusion savings: PDL overlaps launches, and their arithmetic and
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

The one-row TP4 fake-host journal falls from 2,382 to 2,134 launches per
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

## Remaining boundaries

| Candidate | Opportunity | Constraint |
| --- | --- | --- |
| Attention merge -> ceil | 40 launches | Preserve BF16 output boundary and whole-row status before publishing any group |
| `wo_a` merge -> ceil | 40 launches | Current merge CTAs own slices; one bad slice must zero the complete quantized row |
| Routed weighted SwiGLU -> ceil | 40 launches | Keep route multiplication before BF16 rounding; active tiles own 16 columns, quantization groups 32 |
| Shared gate/up merge -> SwiGLU+ceil | 40 launches | SwiGLU already includes ceil; paired banks need common ownership and row-wide validation |
| Engram decode -> ceil | Two launches | Preserve decoded BF16 operands and row status |
| All-gather -> BF16 round | 81 launches including embedding | Keep F32 rank-order sum and the final BF16 boundary, with peer completion before consumers |
| BF16 round -> HC expand | 80 launches | Preserve rounded sublayer output before residual mixing |
| HC expand -> next lagged mix | 77 uninterrupted boundaries | Keep expansion's BF16 output, previous `pre`, alternating buffers and fixed mixer/Sinkhorn orders |
| Query projection merge -> RoPE | 40 launches | Row-wide validation spans all heads while merge CTAs own output slices |

HC expansion already performs the residual addition. Its stream-order FMA
sum, separately rounded `post*y` multiplication and addition, and final
BF16 conversion must stay in order. Engram interrupts two expand/mix
boundaries; the final expansion feeds the head. A two-kernel mixer also
needs the original projection chains, RMS fold, 20 Sinkhorn iterations,
publication fences and ticket reset semantics.

## Verification

CPU tests check preparation, alias and extent refusals, exact argument
composition, independent window planes, and zero workspace. CUDA compile
audits require zero spill/local and stack bytes. Fake-host tests count the
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
