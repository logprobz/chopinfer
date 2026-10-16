# GLM 5.3 Flash at TP4

Dimensions come from the checkpoint's `config.json`, matched to the compiled
`flash` geometry in `models/glm5next/geometries/supported.json`. This page
gives the per-rank shapes, the collectives and their precision, the
context-parallel groups, package memory, IDs and device selection of the one
topology, `glm5next-tp4`.

**Served, qualified and released** ([status](model-glm5next.md)): every stage
below lowers for all three checkpoint layouts and runs on four GPUs, and
create serves context-parallel sizes 1, 2 and 4.

## Per-rank shapes

GLM 5.3 Flash: hidden 4096, 45 layers (34 KDA layers and 11 latent sparse
attention layers at 3, 7, ..., 43), dense MLPs in layers 0-2 (intermediate
12,288) and MoE MLPs after them (288 routed experts, top 8, intermediate
2048, one shared expert of 2048), four BF16 hyper-connection streams mixed
by Sinkhorn mHC around every sublayer, vocabulary 154,880 with an untied
head, and one MTP layer (layer 45).

The per-rank profile (`contract/src/kernels.rs`) divides heads,
intermediate columns and vocabulary rows by the rank count and refuses a
dimension that does not divide; the package serves 4 ranks only. The catalog
(`contract/src/catalog.rs`) places each tensor as replicated or split on a
dimension; each routed expert's gate/up is split on its intermediate rows
and its down on its intermediate columns, so every rank holds a quarter of
every expert.

| Per rank | TP4 |
|---|---:|
| Latent attention query heads (qk 256, v 256, no rotary part) | 16 |
| `q_a_proj` `[1536, 4096]`, `kv_a_proj_with_mqa`, their norms | replicated |
| `q_b_proj` rows (`[16384, 1536]` split on rows) | 4,096 |
| `kv_b_proj` rows (`[32768, 512]`, BF16) | 8,192 |
| Latent `o_proj` K (`[4096, 16384]` split on columns) | 4,096 |
| Indexer (32 heads of 128, pool of 4, top 2,048) | replicated |
| KDA heads (dim 128) | 16 |
| KDA `q`, `k`, `v` projection rows, each | 2,048 |
| KDA convolution channels (q, k and v, 4 taps) | 6,144 |
| KDA `b_proj`, `A_log` | 16 |
| KDA `dt_bias`, `f_b_proj` and `g_b_proj` rows | 2,048 |
| KDA `f_a_proj`, `g_a_proj` `[128, 4096]`, `o_norm` | replicated |
| KDA `o_proj` K (`[4096, 8192]` split on columns) | 2,048 |
| Dense MLP columns (layers 0-2) | 3,072 |
| Routed experts (every expert, a quarter of intermediate 2048) | 288 x 512 |
| Shared expert columns | 512 |
| Router `gate` `[288, 4096]` and score correction bias | replicated |
| mHC `hc_*_fn` `[24, 16384]`, base and scale, per sublayer | replicated |
| Embedding and `lm_head` rows (vocabulary shard) | 38,720 |

The MTP layer has the main layers' latent attention and MoE splits, a
replicated `eh_proj` `[4096, 8192]`, and no hyper-connections: it runs one
stream. Its latent KV is a twelfth sparse layer of the MTP KV formats.

The vision tower (24 blocks of width 1,024) is replicated on every rank
([images](model-glm5next.md#images)).

## State and KV per rank

| Per rank | Bytes |
|---|---:|
| KDA convolution state per layer and slot (BF16) | 36,864 |
| KDA recurrent state per layer and slot (F32, 16 x 128 x 128) | 1,048,576 |
| One state copy over the 34 KDA layers (both) | 36,904,960 |
| Indexer pool tail per sparse layer, slot and verify checkpoint | 1,536 |
| KV page layer, FP8 latent (64 rows of 528 B plus 4,096 B of pooled keys) | 37,888 |
| KV page layer, BF16 latent (64 rows of 1,024 B plus 4,096 B) | 69,632 |

A 64-token page holds 11 sparse layers (12 with the MTP layer). The latent
cache and the indexer state are replicated: every rank holds the same KV, so
KV capacity does not grow with the rank count unless the pages are split by
[context parallelism](#context-parallel-groups). Create allocates the KDA
state before the KV pool: one copy per slot without drafts, `rows + 1` with
them (`rows` verify rows), and sixteen copies for the anchor spares, while
the indexer tail is per-item state of a fixed size: per sparse layer its
working tail plus six checkpoints, one per possible verify row. At three
drafts that is five KDA copies but seven tail copies
([KDA and indexer state](model-glm5next.md#kda-and-indexer-state)).

## Collectives and precision

Every sublayer runs: the mHC mix and collapse of the four streams, the input
norm, the body, one all-reduce, and the mHC expansion back into the streams.

- **Partials and the all-reduce.** The projections that end a sublayer are
  split on their K dimension (KDA `o_proj`, latent `o_proj`, dense
  `down_proj`, and in a MoE sublayer the routed experts' and shared
  expert's down projections) and write F32 partials `[rows, 4096]`. The
  attention and dense MLP sublayers' all-reduce reduces them as BF16
  partials: each rank rounds its F32 partial once to
  BF16, the ranks exchange the BF16 values (half the bytes of F32), and
  every element is the F32 sum of the ranks' BF16 values over source ranks
  in fixed order from F32 zero (identically on every rank), rounded once
  into the BF16 row the expansion reads. The entry writes that row itself.
  The row count picks the entry (SM120's table,
  `kernels_sm120::backend::bf16_tuned`) in every phase: one row runs the
  low-latency all-reduce, two CTAs that exchange with their counterparts
  on the peers; 2 to 4 rows the phased all-reduce's one-shot entry, 5 to
  1,023 rows its two-shot entry, one launch of 16 sending CTAs (with 32
  more summing CTAs above 32 rows), and from
  `kernels_sm120::backend::BF16_TURNS_ROWS` (1,024) rows the two-shot
  whose writers take turns at each destination (same CTAs, transfers and
  sums). Prefill
  ranks start their all-reduces at different times (the routed experts'
  loads differ), and the turns keep one writer per receiving PCIe link
  where the two-shot lets two or three share one. All three entries
  are admitted against device residency at create; an instance whose
  device cannot vouch for them composes the same numerics from the round
  row, its widening, the comm engine's F32 reduce-scatter and all-gather
  pair and the round row again. All of them sum in the same rank order.
  The phased entries share one state whose flags carry the call's epoch
  and stay set, so an instance zeroes every rank's phased flags and epoch
  while every rank is idle every `engines::PHASED_RESET_RUNS` runs, long
  before a flag tag could repeat.
- **The MoE all-reduce.** A MoE sublayer's (and the drafter's) F32
  partials are summed in F32 over source ranks in fixed order from zero
  and rounded once into the BF16 row (`CommQuery::AllReduceRounded`): the
  rank partials carry no BF16 rounding of their own, which keeps the MoE
  output about 17% closer to an f64 oracle than BF16 partials (chosen for
  the split routed experts on 2026-10-12). SM120's
  table runs one row through the low-latency round entry and more through
  the phased F32 one-shot round and round two-shot, admitted at create
  with the BF16 entries; without them the comm engine's F32 pair and the
  round row compose it. Its phased state is its own (a state serves one
  payload format, about 144 MiB per rank) and is zeroed with the BF16
  one. The comm epilogue fuses only the BF16 boundaries.
- **MoE.** The router is replicated and scores in F32. Every rank holds a
  quarter of every routed expert's intermediate (tensor parallel, as the
  shared expert) and runs every route over its slice: gate/up writes 512
  of each expert's 2,048 SwiGLU columns, the grouped down projection
  reduces their K of 512 (four 128-K chains in order) into F32 rank
  partials, and the rank's weighted routes are summed per token in top-k
  order in F32 and left unrounded. Every rank does the same work at every
  row count, so no rank waits on a busier one. The shared expert's down projection (split on
  K) adds that sum through its F32 residual epilogue and writes the F32
  partial the all-reduce rounds and sums, so routed and shared outputs meet
  the collective once. In the decode and verify graphs the shared expert
  runs beside the routed experts instead, and the routed sum adds its F32
  output: the same add of the same two values.
- **mHC.** The four streams are BF16 and replicated on every rank. The
  mix-collapse of each row is split over 16 CTAs whose chain sums add in a
  fixed order, bitwise the one-CTA kernel. The
  expansion computes `bf16(bf16(post_j * y) + bf16(sum_i comb[i][j] X_i))`
  from the reduced BF16 row `y`, with `post` and `comb` rounded to BF16
  once; the head collapses the streams by `bf16(sum_i X_i / 4)`.
- **Embedding.** Each rank looks up the rows of its vocabulary shard (zeros
  elsewhere); the rows are widened to F32 and all-reduced, which is exact
  because one rank holds each row (its rounding to BF16 is too).
- **Head.** `lm_head` is split by vocabulary row and writes F32 logits for
  its 38,720 rows. Decode and verify steps exchange the F32 shards to the
  rows' owners; a prefill gathers its logits rows to rank 0.
- **MTP.** The drafter's sublayers end in the same all-reduce, then add into
  its single BF16 stream. Each draft token is the best of every rank's
  candidate over its logits shard, gathered by an exact F32 all-reduce of
  the zero-padded candidates.

Every stage lowers: the projections and the head
(with the F32 outputs above) through the shared projection runtime over the
package's precision component, the all-reduce, the rounding, the mHC, the
embedding, the routed experts and their sum, the logits moves and the
drafter's gather and argmax. The dense GEMMs and the all-reduce compute
every row of a decode or verify bucket. Their rows stay independent through
row-local chains, such as the KDA gates' down then up projections or a gate-up
then its down projection, and each chain ends at a consumer that reads only
the live rows by the device count (a copy, a row kernel, an engine, or the
rounding after the all-reduce) or at the logits exchange, whose inactive rows
the sink never reads. The
NVFP4 layouts lower the same stages: the instance checks its acquired image
against the package's layout and binds the NVFP4, MXFP8 and BF16 expert and
projection formats that layout stores (`bundle/src/lower/formats.rs`).

The latent attention partitions a row by its kind (one split for a target
prompt row and every drafter row, 64 fixed splits for a generated row), not
by the step's bucket, and the dense projections' chain splits and the MoE
block warps are bitwise their one-split and column forms. A target row
never changes GEMM layout; the drafter's catch-up pass still does with its
row count, which is not shown to keep bits
([batch invariance](model-glm5next.md#batch-invariance)).

## Context-parallel groups

The topology declares context-parallel sizes 1, 2 and 4
([context-parallel attention](context-parallel.md)). A group is N
consecutive ranks: rank `t` is group rank `t % N` of group `t / N`, so DCP 2
pairs ranks 0-1 and 2-3, and DCP 4 is one group of all four. Heads stay
split at 16 per rank. Each sparse layer's latent KV pages are striped across
the group, only a row's owner writes it, the indexer selects per rank and
merges the global key list exactly, the absorbed queries are all-gathered,
and each rank merges its own heads' partials in fixed rank order before the
value projection and the all-reduce, which are unchanged. The KDA state and
the indexer tail are not split.

Create serves sizes 2 and 4 (`bundle/src/context_parallel.rs`,
`pub const SERVED: bool = true`). **DCP 2 and 4 are served, not
qualified**; each size has its own gate declaration. Planned capacity per
size is in [deploy](deploy-glm5next.md#kv-formats-and-planned-capacity).

## Device memory

The package declares 32 sequences, decode graphs of 1 to 32 items, verify
graphs of up to 32 rows (one to six rows per item), prefill of up to 8,192
tokens with a granule of 64 (eager, replayed from captured graphs once its
exact shape repeats, except a prefill injecting encoder rows, any prefill at
DCP 2 or 4 and the stage-sync diagnostic), and 64-token pages. Its minimum device
memory per rank is the checkpoint's weights per GPU plus a 1 GiB host
reserve (`compiler/src/package.rs`):

| Checkpoint | Weights per GPU (with MTP and vision) | Minimum device memory |
|---|---:|---:|
| FP8 (`fp8-block128`) | 83,370,801,144 B (77.65 GiB) | 84,444,542,968 B |
| local-inference-lab NVFP4 (`nvfp4-routed`) | 50,875,133,880 B (47.38 GiB) | 51,948,875,704 B |
| NVIDIA NVFP4 (`nvfp4-dense-routed`) | 52,469,325,024 B (48.87 GiB) | the weights plus 1 GiB |

The planned KV pool is in
[deploy](deploy-glm5next.md#kv-formats-and-planned-capacity) and the
measured one on the [model page](model-glm5next.md#context-parallelism).

## IDs

One package, `glm5next.v1`, serves one topology, `glm5next-tp4`, from each
of the three checkpoint layouts; the package records the layout it was prepared for as the
model option `checkpoint.quantization_layout`. Preparation refuses any other
rank count ("unsupported tensor parallel size N: glm5next serves [4]"). At
TP2 the FP8 weights would take 153.69 GiB per GPU, and TP2 and TP8 serving
are out of scope (`contract/src/fit.rs`).

## Device selection

`--tensor-parallel-size 4` (or `CHOPINFER_TENSOR_PARALLEL_SIZE=4`) selects
the topology, and `--devices` selects and orders visible CUDA ordinals
([device visibility](deploy-common.md#device-visibility-and-rank-selection)).
Each rank requires one distinct device; duplicate ordinals are refused.
