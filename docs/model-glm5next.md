# GLM 5.3 Flash

The full correctness suite of the default, `dcp2` and `dcp4`
configurations includes [KV preemption swap qualification](serving-kv-pool.md#formats-caching-and-offload);
offload stays disabled by default. These checks joined after the
qualification below and have not run on GPUs yet.

**Status: served, qualified and released.** GLM 5.3 Flash
(`glm5next`) serves at TP4 on GPUs from the official FP8 checkpoint: MTP
with three or five drafts or autoregressive, context-parallel sizes 1, 2
and 4, images, and FP8 or BF16 latent KV. It admits a checkpoint of any
supported layout ([checkpoint layouts](#checkpoint-layouts)). The final
release gate's results were adopted on 2026-10-11, and the default
configuration's soak failure passes the soak's declared warm-up and cache
allowance, so
every gate declaration but the diagnostic-taps one qualifies. Its full
requalification through the release preview (stage 5,
2026-10-14 to 2026-10-15, on one release install) then qualified it, with
accepted rows in two configurations (`taps` and `dcp4`)
([release qualification](#release-qualification-stage-5)), and
`models/glm5next/model.toml` declares `[release] enabled = true`: the
release archive and the image built from it ship its helper, bundle
library and kernel locks. Its serving profiles travel inside the packages
the helper builds, as their `serve-profiles` component.
[References and qualification](#references-and-qualification) lists the
results.
[The model's README](../models/glm5next/README.md) holds the details each
section summarizes.

| Area | State |
|---|---|
| Contract, catalog, weight plan (TP4) | Built and checked against the checkpoint headers |
| Text program | Every stage lowers for every admitted layout, the MTP drafter's included; runs on four GPUs |
| Package emission, `serve` | The helper emits the package (text, or text and images when the checkpoint stores the vision tower) for a checkpoint of any admitted layout; `serve --model DIR` prepares and serves it |
| FP8 and BF16 latent KV | Both served and qualified; FP8 is the default, BF16 was requalified on 2026-10-16 ([BF16 KV](#bf16-kv)) |
| MTP | Three drafts (`tp4-mtp`, the default) or five (`tp4-mtp-k5`, text package only) |
| Context parallelism (DCP 2 and 4) | Served (create accepts every checkpoint layout; run on GPUs with the FP8 checkpoint); DCP 2 and 4 have their own gate declarations |
| Images | Served by default (`auto` modalities); video is refused |
| NVFP4 and mixed layouts | Admitted by layout and served (W4A16) with `tp4-mtp`; another checkpoint is qualified as an [external variant](../tools/qual/README.md#external-variants) |
| Serving profiles | Five declared, each serving every layout; four qualified through 32 sequences on the official checkpoint, `tp4-mtp-k5` experimental (see [Serving profiles](#serving-profiles)) |
| Qualification | `qual.json`: the default configuration and five others (BF16 KV, held on 2026-10-14, declared again on 2026-10-15 and requalified on 2026-10-16); the final release gate (two parts, completed 2026-10-11) passed every check but two soaks under the plain memory rule, which the declared soak warm-up now passes, and its results were adopted on 2026-10-11; only `taps` stays `mechanics_only` |

## Model

`zai-org/GLM-5.3-Flash` at revision `eb9eb208` (Hugging Face `model_type`
`glm5_next`; file sizes and digests in `models/glm5next/checkpoint-files.json`).
The text model has 45 layers over a hidden size of 4,096 and a 154,880-token
vocabulary, with mHC: four residual streams per position, mixed into and out
of each sublayer by hyper-connections.

- 34 Kimi Delta Attention (KDA) layers, 64 heads of 128, with a short
  convolution of 4 (layers 0-2, 4-6, ..., 40-42 and 44).
- 11 latent sparse attention layers (3, 7, ..., 43): a 512-wide KV latent, 64
  query heads of 256, and an indexer that selects 2,048 keys in pools of 4.
- Dense MLPs in layers 0-2, then mixture-of-experts layers: 288 routed
  experts (top 8, sigmoid scores) plus one shared expert, with clamped
  SwiGLU (limit 10).
- One MTP layer (layer 45).
- A 24-block vision tower (see [Images](#images)).

The catalog has 76,108 tensors (328.3 GB); at TP4 each GPU holds 77.65 GiB
of weights, 70.89 of them routed experts.

## Geometries

A geometry follows the
[same-architecture recipe](adding-a-model.md#same-architecture-checkpoints)
with GLM's files: the config's execution part in
`models/glm5next/geometries/<id>.json` plus a `supported.json` entry whose
`limits` bound `text_config.index_topk`. Layer counts and kinds may change
within the layer rules (`models/glm5next/contract/src/structure.rs`), every
rank-split size must be divisible by four, and the per-rank shapes must fit the engine
domains (latent attention takes local heads in 16s; the contract's
`geometry_recipe` test checks the latent attention and KDA planners). The
schedule table is `models/glm5next/contract/tuning/<id>/schedules.json`; an
untuned geometry inherits flash's legal rows with a label naming them
([tuning](../models/glm5next/contract/tuning/README.md)).

`bash tests/model-geometry-recipe.sh glm5next` stops at the contract stage
(`[geometry_recipe] stage = "contract"` in `models/glm5next/model.toml`): it adds
`fixtures/glm5next/geometries/synthetic-small` as data, then the contract
crate admits its config, builds kernel profiles, the catalog, the TP4 weight
plan, tuning decisions and the capacity table
(`model_glm5next_contract::capacity`), and refuses changed copies by field.
CI runs it as the `geometry-recipe-contract` step
(`bash tests/model-geometry-recipe.sh --contract`).

The recipe stops there because the bundle serves only flash.
`Shape::TP4` (`models/glm5next/contract/src/contract.rs`) fixes the flash
geometry at TP4 for the bundle program and for the cubins
`bundle/src/sm120/build.rs` compiles. An added geometry is therefore
admitted, cataloged and planned by the contract, but serving it first needs
the bundle to build and select its shape per geometry. The recipe moves to
the bundle stage with that work.

## Bundle and TP4 topology

The package `glm5next.v1` serves one topology, `glm5next-tp4`
(`contract/src/contract.rs`). `fit.rs` refuses TP2 (the weights do not fit)
and TP8 by name. Per rank, KDA and latent attention run 16 heads; the latent,
query-down, indexer and router projections are replicated; each rank holds
a quarter of every routed expert (512 of its 2,048 intermediate columns),
as the dense and shared MLPs split by column; the embedding and head split
by vocabulary row. The package declares 32 sequences, 8,192 prefill tokens
per step, a prefill granule of 64 (the KDA chunk), sixteen anchor items per
step and 64-token pages.

A prompt that does not end on the granule runs its last rows in the same
chunk as the rows before them, instead of ending a chunk at the last
multiple of 64 and running the rest as a step of its own. Up to sixteen such
prompt-ending items per step keep their state at that anchor in spare
entries (`contract/src/state.rs`): the KDA convolution and scan write their
spares after the checkpoints, and with the MTP drafter the step copies the
target's normalized rows from the indexer pool before the anchor into the
retained rows' spare. A capture at the anchor copies the spares, and its
header records that pool boundary as the drafter's frontier: the step ran
the drafter past the anchor and kept no incomplete pool, so a continuation
runs the pool's drafter rows again (`bundle/src/drafter.rs`,
`anchor_frontier`). The indexer tail's spares are never written, because an
anchor ends a pool. The scheduler ends the chunk of any further item at its
anchor, as before.

`bundle/src/program.rs` lists every layer's stages and `launch.rs` turns
each into a launch record; the fake-host journals in `journal_tests.rs` pin
the program. The instance creates and acquires the planned text image (and
the encoder image when the package serves images), the KDA state and the
KV pool. A step runs its prefill, then its decode or verify phase as captured
graphs per bucket, then the drafter. Every stage lowers, the projections and
the LM head through the shared projection runtime over the package's
precision component. The instance checks its image against the package's
layout, binds each projection and expert in the format its layout stores
(`bundle/src/lower/formats.rs`), and admits the image's scale payloads on
the device at create (`contract/src/payload.rs`), refusing an invalid code
by view and element.

The weights load directly from the checkpoint through the bundle's weight
plan (`contract/src/direct.rs`, [direct loading](direct-loading.md)); the
package holds no weights.

Per-rank shapes, collectives and their precision: [GLM 5.3 Flash at TP4](topology-glm5next.md).

### Step cost

Each of these keeps the served bits of the schedule it replaced; fake-host
journals and kernel-lab bitwise cases pin them.

- **Host lowering.** A rank's program is lowered only when one of its
  bucket's graphs is missing, and projection shapes are shared per shape
  and layout, so a step whose graphs are cached does no lowering.
- **Prefill.** An eager prefill whose exact shape (rows and everything else
  its programs depend on) ran before replays the graphs its second run
  captured (`bundle/src/instance/prefill_graphs.rs`). Nothing is bucketed:
  a prompt length seen once stays eager, except a request's last prompt
  chunk (one item's rows after its last anchor, at most the 64-row
  granule, scoring its last row), which captures at its first run and
  stays captured, at most 64 programs, so later requests' last chunks of
  that length replay on their way to the first token; it captures again at
  its next run only after a release of every graph (a failed capture or
  release, or the store's fallback when its programs fill up under other
  graphs). `CHOPINFER_GLM5NEXT_LAST_CHUNK_GRAPHS=1` is an internal
  diagnostic switch that captures all 64 at create, before any other graph
  (not on a context-parallel instance; not a serving tunable; served bits
  are the same either way). It is off because holding all 64 lengthened
  AR decode steps by up to about 1% on the GPU (host sampling and device
  wait), against one eager-cost capture per length at its first run. Two kinds of
  prefill always stay eager (`prefill_key` in
  `bundle/src/instance/execute.rs`): one that injects encoder rows (a
  text-only prefill of a package serving images still replays), and every
  prefill under the `CHOPINFER_GLM5NEXT_SYNC_STAGES` diagnostic or the
  text taps of a diagnostic build.
- **Dense projections.** Over BF16 and block-FP8 weights, a decode or
  verify bucket whose tiles would leave most SMs idle spreads its MMA
  chains over CTAs (64 K per chain over BF16 weights, one 128-K block over
  FP8) and adds their temporaries in K order (`DenseKernel::TiledChains`,
  bitwise the one-split GEMM); prefill runs one split in warp tiles, and
  the block-FP8 latent q-a, kv-a, q-b and output, dense down and shared
  down projections in decoded weight tiles
  (`gemm::balanced_decoded_tile_knobs`: each weight code decoded to BF16
  once per CTA, bitwise the warp tiles, the final partial wave split into
  64-row halves when they fit the SMs; their own cubin and lock family,
  `projections_decoded`; the wide verify buckets keep the warp tiles;
  `contract/src/sm120/precision.rs`); the BF16 linear attention q, k, v and
  output, its gates' up projections, the indexer's query and the
  drafter's input projection run staged weight tiles at prefill
  (`gemm::staged_tile_knobs`: 128 x 128 tiles that one lane stages by TMA
  for eight `ldmatrix.x4` math warps, the final wave split as above,
  bitwise the warp tiles; cubin and lock family `projections_staged`).
  Projections
  over NVFP4 weights run the
  one-split tiled GEMM; the BF16 projections of a mixed layout take the
  chain splits and warp tiles like any other. The decode and verify graph
  buckets up to 32 rows run each KDA layer's projections of its normed
  input (q, k, v, forget-a, beta and gate-a) as one stacked launch
  (`projections_stacked`, `contract/src/sm120/stacked.rs`): each part keeps
  its own chains, so its bits are its own entry's, and writes where the
  layout copies would put its rows; each CTA asks L2 for its later weight
  chunks before it waits on the launch before it, and releases each ring
  stage only after a proxy fence orders its reads, as the ordered strip
  chains do. The precision component
  binds those uses to the stacked entry, under a dependency over its lock
  and the part's place in it. Prefill and wider verify buckets keep the six
  GEMMs and four copies.
- **MoE.** The chained MoE's block-FP8 dense MLP and shared expert run as
  block warps at decode, bitwise the column schedule with four times its
  CTAs (`contract/src/sm120/engines.rs`), and their gate/up as decoded
  tiles at prefill (`chained_decoded.cuh`: each weight code decoded once
  per CTA of 128 tokens, bitwise the column schedule). The routed experts,
  split over the ranks (`chained::split::forms`), run their block-FP8
  gate/up in warp streams at decode and their down in block warps up to
  four tokens and in columns above; at prefill every routed projection
  runs in columns and the block-FP8 ones in tile pairs from the backend's
  tile-pair rows. In the other layouts the NVFP4 and BF16 routed experts
  and a BF16 dense MLP and shared expert run as block warps at decode;
  MXFP8 routed experts and an NVFP4 dense gate/up keep the column
  schedule.
- **mHC.** The mix-collapse of each sublayer is split over 16 CTAs per row,
  with chain sums in fixed order (`sinkhorn_hc_split`), bitwise the one-CTA
  kernel. Prefill launches from 144 rows run it with 4 rows per CTA, and
  from 1,505 rows with 8 (`sinkhorn_hc_tiled`), also bitwise the one-CTA
  kernel. The decode and verify graphs run each main layer's mix-collapse
  and input RMSNorm as one launch (`sinkhorn_hc_mix_norm_exact`), whose norm
  adds its squares by an exact parallel method with the serial sum's bits
  (`row/ordered_sum.cuh`); prefill keeps the two launches.
- **Latent attention.** Every FP8 attention launch in prefill's one split
  per row runs the slice-widened form (`latent_attention_slices`): each warp
  widens only the FP8 slices its MMAs read and the page lookups run ahead, so
  two CTAs share an SM, bitwise the one-split entry.
- **Indexer selection.** A pooled indexer selection whose bucket takes one
  split per row (376 rows or more: prompt pieces, and the context-parallel
  per-rank selections) runs its one-split form (`pooled_indexer_rows`): one
  launch scores every pool and writes the key lists or pairs itself, in
  place of the split kernel and its merge, bitwise the pair.
- **KDA prefill.** The chunk prepare and the scan run the streamed schedule
  (`channel_decay_streamed`: the prepare's loads, gate sums, norms and
  forward substitution spread over every warp; a four-warp scan sharing
  each state load between V' and the output, its half tiles streamed by
  cp.async), bitwise the staged kernels.

The Sinkhorn mHC comm epilogue (`--model-option
runtime.sinkhorn_comm_epilogue=true`, default false) folds each sublayer
boundary's round, expand, next mix-collapse and next input norm into one
launch after the unchanged all-reduce. On four GPUs it gave the same bits as
the default path and was not faster, so it stays off; a server with it on
reports unqualified model options.

The performance work's single-request GPU jobs (2026-10-07, on a build
before the release-candidate install) measured autoregressive decode at 28.4 ms/token,
MTP with three drafts at 20.1 ms/token, and a time to first token of 103 ms
for a 55-token prompt.

### Measured speed

One benchmark pass of an unqualified build (installed 2026-10-07, run
2026-10-07 and 08), four GPUs, one server per configuration started with the
profile's settings (32 sequences, FP8 KV, text package unless noted),
greedy decoding with end-of-sequence ignored. Every declaration was still
`mechanics_only` true. Decode is `bench.py` over diverse prompts, 512
output tokens per request, aggregate output tokens per second at
concurrency C; prefill is one 32,726-token prompt at C1.

| Profile | Decode C1 tok/s (ms/token) | C8 tok/s | C32 tok/s | 32K prefill tok/s (TTFT) |
|---|---:|---:|---:|---:|
| `tp4-mtp` | 77.0 (13.0) | 220.6 | 325.9 | 3,464 (9.4 s) |
| `tp4-ar` | 35.3 (28.3) | 146.1 | 329.5 | 3,557 (9.2 s) |
| `tp4-mtp-k5` | 78.7 (12.7) | 218.1 | 325.8 | 3,470 (9.4 s) |
| `tp4-mtp-dcp2` | 77.1 (13.0) | 219.3 | 308.1 | 3,050 (10.7 s) |
| `tp4-mtp-dcp4` | 69.5 (14.4) | 188.3 | 271.9 | 1,374 (23.8 s) |

What the pass showed:

- **Adaptive speculation verified nothing at C32 in the measured
  cohorts.** `spec_bench.py`'s GSM8K and chat cohorts at C32 counted no
  verify step at DCP 1 for three and five drafts (see
  [MTP speculation](#mtp-speculation)); DCP 2 and 4 have no
  such count. Separately, `bench.py`'s diverse prompts decode at about the
  same rate with and without MTP at C32 (325.9 against 329.5 tok/s), while
  MTP is 2.18 times faster at C1 and 1.51 times at C8.
- **Five drafts do not pay.** Against three: about 2% faster at C1 (78.7
  against 77.0 tok/s), 1-7% slower at C8 depending on the prompts, for 34%
  less KV. Three drafts stay the default; `tp4-mtp-k5` stays a benchmark
  profile.
- **DCP 4 costs long prefill.** The 32K prompt takes 23.8 s at DCP 4
  against 9.4 s at DCP 1 and 10.7 s at DCP 2, and C32 decode drops 17%. DCP 2 costs 5% at C32
  and 12% in prefill for 1.87 times the KV.
- **Decode utilization.** In the saturation windows below, decode kept the
  SMs 92.4-97.5% active with 1.4-4.9% tensor activity and 10.9-20.2% of
  DRAM read bandwidth. These are window averages, not a time breakdown of a decode
  step; they suggest, without showing, that decode is limited by neither
  tensor throughput nor DRAM bandwidth.

Saturation windows (same pass and install; means over each window of
`bench/saturation_run.py`, decode at C1, C8 and C32, prefill at 8,192-token
prompts at C1 and C8). Each cell lists C1 / C8 / C32:

| Profile | SM active % | Tensor active % | DRAM read % | Power % of limit | Prefill 8K C1 / C8 tok/s |
|---|---|---|---|---|---:|
| `tp4-mtp` | 92.4 / 95.3 / 97.5 | 2.5 / 3.4 / 4.9 | 13.5 / 16.1 / 20.2 | 58.1 / 61.6 / 73.1 | 3,453 / 3,695 |
| `tp4-ar` | 95.8 / 96.6 / 97.5 | 1.4 / 3.2 / 4.7 | 10.9 / 16.1 / 20.0 | 53.4 / 61.0 / 73.3 | 3,563 / 3,835 |

## KV formats

A 64-token page holds, per latent attention layer, the KV latent rows and
the indexer's pooled keys (`contract/src/contract.rs`):

| Layout | Latent rows | Page-layer bytes | Page bytes (11 layers) |
|---|---|---:|---:|
| `glm5next.latent-paged-fp8.v1` (preferred) | FP8 E4M3 with per-128 F32 scales | 37,888 | 416,768 |
| `glm5next.latent-paged-bf16.v1` | BF16 | 69,632 | 765,952 |

The `-mtp` variants add the MTP layer's pages (12 layers). The host picks
the format with `--kv-cache-dtype`; the profiles leave it at `auto` (FP8).
BF16 KV was requalified on 2026-10-16 ([BF16 KV](#bf16-kv)).

## KDA and indexer state

Each request slot keeps per KDA layer its convolution history and its
recurrent state, the recurrent state layouts, and
per latent attention layer the indexer's incomplete-pool tail, as per-item
state (`contract/src/state.rs`):

| State | Per slot |
|---|---:|
| `glm5next.kda-conv.v1` | 1,253,376 B (34 x 36,864) |
| `glm5next.kda-state.v1` | 35,651,584 B (34 x 1,048,576) |
| `glm5next.indexer-tail.v1` | 1,536 B per layer and verify checkpoint |

One copy of a slot's KDA state is 36,904,960 B per rank. Create allocates
the KDA pools before the KV pool and the package declares them preallocated
(`chop.recurrent-state.preallocated.v1`), so the host reserves no copies of
its own and sizes the pool from what create leaves. A slot holds one KDA
copy without drafts and `rows + 1` with them, one per verify row plus one,
so a speculative step can keep any accepted prefix: three drafts take five
copies and five drafts seven. The pools hold sixteen copies more, the
anchor spares of up to sixteen prompt-ending items per step, which with the
indexer tail's spares take 590,774,272 B per rank with the MTP layer. The
indexer tail is per-item state of a fixed size whatever the drafts: per
sparse layer its working tail plus six checkpoints, one per possible verify
row (129,024 B per slot and rank with the MTP layer). At three drafts a
slot takes 184,653,824 B per rank (184.7 MB). The capacity plan counts the
state as those KDA copies of both, as the capacity table's state copies
column lists, plus the sixteen anchor copies; the tail's fixed size makes it
an approximation.

## MTP speculation

The checkpoint has one MTP layer. The contract allows up to five draft
tokens (verify buckets of 2 to 6 rows). The package declares the bundle's
MTP drafter at that maximum of five drafts (`compiler/src/package.rs`),
with host drafts as the alternative, and the served `--speculative-tokens`
picks how many a server drafts: three for `tp4-mtp`, five for
`tp4-mtp-k5`. Create allocates KDA state copies and the host captures verify
graphs for the served count, and the drafter drafts the rows the host allots, so
`tp4-mtp`'s capacity is that of a three-draft package.

Measured acceptance (the 2026-10-07 benchmark pass, unqualified;
`spec_bench.py`, adaptive speculation, 256 output tokens, FP8 checkpoint):

| Drafts | Prompts | C1 acceptance (tokens per verify step) | C8 acceptance | C32 |
|---:|---|---:|---:|---|
| 3 | GSM8K | 0.83 (3.48) | 0.79 (3.35) | no verify steps (adaptive serves C32 autoregressively) |
| 3 | chat | 0.58 (2.71) | 0.60 (2.80) | no verify steps |
| 5 | GSM8K | 0.81 (3.53) | 0.77 (3.46) | no verify steps |
| 5 | chat | 0.57 (2.77) | 0.59 (2.80) | no verify steps |

Five drafts are served on the text package only (`--modalities text`): the
capacity plan leaves no KV pool for seven KDA copies per slot at 32 slots
once the encoder's image and buffers are placed. A five-draft package
serving images has not been tried on a GPU.

## Batch invariance

The latent attention partitions each row by what it is, not by the step's
bucket: a target prompt row runs prefill's one split, a generated row
(decode or verify) always 64 fixed splits, and every drafter row one split,
whichever path its pass takes (`bundle/src/lower.rs`, `attention_phase`).
The dense chain splits and MoE block warps are bitwise their one-split and
column forms at every split count. Target decode and verify launches never
exceed 32 rows and target prefill always runs the plain GEMM layout, so a
target row does not change GEMM layout with its step.

This is not a proof that every row's bits are independent of its step. The
drafter still switches a catch-up pass from the verify path (at most 32
rows, the decode GEMM layout with swapped MMA operands) to the prefill path
(more rows, the plain layout), and bit equality across that switch is not
established. It can change draft tokens and so which drafts are accepted;
under greedy decoding the accepted tokens are still the target's own.
End-to-end exactness is what the
gates observed: before the fixed splits a preempted request with more than
2,048 context keys changed bits at concurrency 16; afterwards the c16
preemption pair at DCP 1 was identical (36 of 36 requests), and the
`preemption_exact`, `verify_exact` and `determinism` gates check it at each
configuration. Context-parallel output is not bitwise equal to DCP 1
([context-parallel attention](context-parallel.md#numerics-and-determinism)),
so each size is gated on its own.

## Context parallelism

The topology declares context-parallel sizes 1, 2 and 4 (DCP): the latent
attention's KV pages are split across ranks and the partial attention rows
merged. Create serves sizes 2 and 4 (`bundle/src/context_parallel.rs`,
`pub const SERVED: bool = true`) for every checkpoint layout; a four-GPU
kernel-lab chain (`crates/kernel-lab/tests/glm5next_context_parallel_chain.rs`)
runs one sparse layer through the bundle's lowering at DCP 2 and 4 against
DCP 1. `/health` reports `kv_page_tokens` beside the context-parallel size,
from which `tools/qual/prefix_exact.py` derives the prefix reuse unit (one
stripe, 128 tokens at DCP 2 and 256 at DCP 4). DCP 2 and 4 are gated by
the `dcp2` and `dcp4` configurations of `qual.json`.
The model-neutral design is in [context-parallel attention](context-parallel.md).

`model_glm5next_contract::capacity` plans each rank by formula (94 GiB
usable with the host's 1 GiB reserve and 0.9 memory fraction, the text
weights, with images the encoder's image and buffers (1.05 and 2.26 GiB),
named workspaces plus a 2 GiB allowance, the context-parallel buffers (0.42
GiB at DCP 2, 0.76 GiB at DCP 4), the state, and the KV pool from what
remains). KV tokens scale with the DCP size; state does not. For the FP8
checkpoint with FP8 KV, a text package and one serving images:

| Slots | MTP drafts | State GiB | Text DCP 1 | DCP 2 | DCP 4 | Images DCP 1 | DCP 2 | DCP 4 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0.58 | 1,806,272 | 3,489,152 | 6,773,760 | 1,315,328 | 2,507,392 | 4,809,984 |
| 32 | 0 | 1.65 | 1,648,064 | 3,172,864 | 6,140,928 | 1,157,120 | 2,190,976 | 4,177,152 |
| 32 | 3 | 6.05 | 665,792 | 1,218,560 | 2,249,472 | 215,744 | 318,464 | 449,280 |
| 64 | 0 | 2.75 | 1,484,800 | 2,846,208 | 5,487,872 | 993,856 | 1,864,320 | 3,524,096 |
| 64 | 3 | does not fit | | | | | | |

These are planned pool totals shared by every slot; one sequence is limited
to the model's 1,048,576-token context. The 94 GiB figure is an assumption
(the board reports 95.59 GiB), and the plan's 2 GiB allowance stands for
buffers it does not list, so serving's own pool differs from the plan.
Measured on an empty pool at 32 slots by capacity runs on 2026-10-14
(a build that allocates the KDA state and its anchor
spares before the KV pool), as the pool the host allocated at startup: each
server's startup line checked against `/metrics` and `/health`
(`tools/qual/kv_capacity.py`):

| Profile | KV | Planned, text | Measured, text | Planned, images | Measured, images |
|---|---|---:|---:|---:|---:|
| `tp4-mtp` | FP8 | 665,792 | 811,584 | 215,744 | 360,960 |
| `tp4-ar` | FP8 | 1,648,064 | 1,444,992 | 1,157,120 | 994,368 |
| `tp4-mtp-dcp2` | FP8 | 1,218,560 | 1,504,128 | 318,464 | 611,456 |
| `tp4-mtp-dcp4` | FP8 | 2,249,472 | 2,821,376 | 449,280 | 1,035,776 |
| `tp4-mtp-k5` | FP8 | 366,400 | 509,760 | does not fit | not measured |
| `tp4-mtp` | BF16 | 362,240 | 441,600 | 117,376 | 196,416 |

Measured over planned: 1.22 to 1.39 for the MTP profiles' text packages and
1.67 to 2.31 with images; `tp4-ar` 0.88 (text) and 0.86 (images). Serving
images costs 450,624 tokens measured against 450,048 planned at DCP 1,
892,672 tokens measured against 900,096 planned at DCP 2, 1,785,600 tokens
measured against 1,800,192 planned at DCP 4. The plan overstates `tp4-ar`:
its autoregressive row leaves out the MTP layer's weights, which the
autoregressive server, serving the MTP package with no drafts, likely still
holds (not verified). Five drafts with images was not measured; the plan's
"does not fit" there is a planning statement. The earlier measurement
(2026-10-07, before the TP split and this change) held 892,736 tokens
for `tp4-mtp` with FP8 KV as a text package and 442,112 with images.

Print the whole planned table (1 to 64 slots, 0, 3 and 5 drafts, FP8 and
BF16 KV, the representative layouts or the layout ids given after `--`)
with

```sh
cargo run -p model-glm5next-contract --example capacity-table --locked --offline
```

## Chat frontend

GLM renders its own prompt. `models/glm5next/contract/src/frontend.rs`
(`Glm53Format`) is a Rust port of the pinned `chat_template.jinja` (revision
eb9eb208), one function per template block with the lines it ports, written
on the `frontend-format` toolkit; the bundle exports it with
`frontend_format::export_chat_format_v1!`. `compiler/src/frontend.json`
declares the model-neutral sections and `format.parser`, which the shared
parsers read completions and build tool-choice grammars from: the key-value
tool-call family
(`<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`),
the `<think>` markers and the four role tokens that open a turn. No template
runs at serve time.

The format renders `tool_reference` results as the template does, by name,
including references to unknown or unoffered tools and to deferred tools.
What the template accepts but GLM refuses on purpose (other reasoning
efforts, disabled reasoning, developer messages, a message's own tools, text
beside a tool message's references) is listed with its reason in
`models/glm5next/frontend-divergences.json`. The HF differential corpus in
`fixtures/glm5next/frontend/` is gated on running every template line a
request can reach, and runs as raw requests through the served frontend
(`compiler/tests/frontend_corpus.rs`); the
[model README](../models/glm5next/README.md#chat-frontend) gives the
commands, and [adding a model](adding-a-model.md#updating-the-chat-template)
the template-update procedure.

## Images

The frontend declares one image modality (`compiler/src/frontend.json`):
recipe `image.zero-padded-token-budget.v1` (patch 14, temporal patch 2,
merge 2, 16 to 8,000 tokens, at most 64 images per request), which the chat
format writes as `<|begin_of_image|>`, the `<|image|>` pad repeated per
token and `<|end_of_image|>`. There is no video modality.

`contract/encoders/glm5next-vision.json` declares the tower for the shared
encoder runtime: 1,176-wide patch rows zero-padded to 1,216 columns, the
patch GEMM, 24 pre-norm blocks of width 1,024 (per-head q/k RMSNorm, axial
rotary, clamped SwiGLU), a post norm, the 2 x 2 merge and downsample, and a
LayerNorm-GELU-clamped-SwiGLU projector to the 4,096-wide rows the prompt
reads. It runs under the `f32-pairs.v1` policy. Its working memory is
R x 57,600 + 16 bytes for R patch rows (1.89 GB at the 32,768-row capacity),
beside the 65,536-row media arena. The encoder image (1.13 GB, replicated)
is a planned component the weight plan builds from the checkpoint
(`contract/src/encoder_plan.rs`), including two bundle-owned transforms:
padded patch rows and the downsample's space-to-depth column order.

**Serving images.** A checkpoint storing the vision tower serves images by
default: the helper reads its headers, checks the tower against the vision
declaration and plans the encoder image (`compiler/src/encoder_image.rs`),
which makes the package require `chop.media-inputs.v1`. The host plans the
encoder image and its rank indexes, create acquires them before the KV pool,
each step runs the encoder before the step opens and the prefill injects its
rows. `--modalities text` leaves the encoder out; a request with a video or
audio part is refused, naming it.

**Image cost.** An image costs its tokens in the prompt (16 to 8,000, by
its size) and an encoder run before its step. Memory is the larger cost:
every rank holds the encoder's image and buffers (3.31 GiB), which create
places before the KV pool, so a package serving images plans about 450,000
fewer KV tokens than a text package at three drafts and DCP 1 (215,744
against 665,792 at 32 slots). Serve `--modalities text` when no request
carries an image.

Evidence:

- `crates/kernel-lab/tests/encoder_conformance_glm5next.rs`: every encoder
  entry the declaration uses, from the bundle's serving cubins, against its
  oracle, with replays on poisoned outputs.
- `crates/kernel-lab/tests/glm5next_encoder_tower.rs` (opt-in GPU): the
  whole tower from the real checkpoint's planned image over the six probe
  images, every block's rows and the output rows compared with the GPU
  encoder reference (`tools/refrun/encoder.py`, exact, policy and
  reference-bf16 modes) under the `encoder_parity` gate's tapdiff rules
  (`tools/qual/encoder_parity.py lab`). It matched: 24 blocks of 9,440 rows
  and 2,360 output rows.
- `models/glm5next/taps.json` declares the encoder and text taps.
  bundle-glm5next's `diagnostic-taps` build writes them
  (`bundle/src/taps.rs`): with `CHOPINFER_TAPS_DIR` set, each image step
  writes the rows each request injects (`enc.vision.out`) and every
  transformer block's rows (`enc.vision.B<nn>.in` and `.out`), under the
  host's producer nonce, published by its commit. A serving build refuses
  to create while a taps variable is set. The `taps` configuration gates
  `encoder_parity` on that build.
- Text taps: while `CHOPINFER_TAPS_SITES` selects one, every target
  prefill runs eagerly, stage by stage across the ranks (collective by
  collective within a stage), and reads the selected taps back from rank
  0's scratch after each stage (`bundle/src/taps/text.rs`): the residual
  streams, each sublayer's collapsed input (`x.hc.attn`, `x.hc.mlp`) and
  normed input, the all-reduced sublayer outputs (`attn.out`, `mlp.out`,
  the BF16 sum the expansion reads), the normalized latents (`x.latent`,
  before the FP8 write), the key lists (`x.indexer.ids`), the router's
  logits, weights and ids, the head's collapse and `final.norm`. The
  launches are the untapped prefill's (a fake-host test compares them).
  Decode and verify graphs and the MTP drafter are not tapped; text taps
  refuse a context-parallel instance and the comm epilogue.
  The `taps` configuration's `text_layers` (`tools/qual/text_layers.py`) compares
  one sequence's taps with the reference runner's F32 run, its floor the
  runner's BF16 run with the served FP8 latent
  ([layer reference](../models/glm5next/README.md#layer-reference));
  `tools.qual.tapdiff` compares any capture.

## Checkpoint layouts

A checkpoint is admitted by its layout, the stored encoding of each
projection class, read from its `config.json` and safetensors headers
(`contract/src/quantization.rs`):

- latent attention and shared experts: block-128 FP8 or BF16;
- dense MLPs: block-128 FP8, BF16 or NVFP4;
- routed experts: block-128 FP8 or NVFP4;
- MTP routed experts: block-128 FP8, MXFP8 or BF16;
- KDA convolution weights and mHC base and scale vectors: BF16 or F32.

The layouts derive from the per-class formats that
`contract/projections.json` declares, so support follows the projection
declaration and the kernels the bundle supplies for it; no layout or
export needs a file of its own. A layout id reads
`attn-<e>.dense-<e>.experts-<e>.mtp-<e>.conv-<d>.hc-<d>`; the official
checkpoint's is `attn-fp8.dense-fp8.experts-fp8.mtp-fp8.conv-bf16.hc-f32`.
Every projection runs BF16 activations (W4A16 for NVFP4 weights;
`--activation-policy checkpoint` is refused for a checkpoint that declares
quantized activations), and a ModelOpt checkpoint's static FP8 KV hint
(`kv_cache_scheme`) is accepted and imports nothing. Every layout serves the
same package and engines with `tp4-mtp` at TP4
([Serving profiles](#serving-profiles)), with the NVFP4, MXFP8 and BF16
formats bound per layout. NVFP4 projections run the one-split tiled GEMM
(see [Step cost](#step-cost)).

Admission is not qualification. `qual.json` qualifies the official
checkpoint; a checkpoint of another layout is qualified out of tree as an
[external variant](../tools/qual/README.md#external-variants), with its own
checkpoint lock, pin and teacher reference, under the model's checks and
limits. Its paired GSM8K compares it with the FP8 arm `qual.json`'s gate
run records ([GSM8K floor](#gsm8k-floor)). The
[model README](../models/glm5next/README.md#checkpoint-layouts) gives the
admission rules and the tests, which `CHOPINFER_GLM5NEXT_EXTRA_CHECKPOINTS`
points at further checkpoints of any layout.

## Serving profiles

`models/glm5next/profiles.json` declares five TP4 profiles of the
`glm5next.v1` package ([serving profiles](serving-profiles.md)), each
serving every [checkpoint layout](#checkpoint-layouts). Every one has 32
sequences, `auto` (FP8) KV, the prefix cache on, 8,192 batch tokens,
2,048-token prefill chunks and a KV memory fraction of 0.9; its
qualification text says which declaration gates it. Four are `qualified`
through 32 sequences on the official checkpoint by the adopted final gate
([final release gate](#final-release-gate-2026-10-11)); the soak of
`tp4-mtp` passes under the declared soak warm-up. `tp4-mtp-k5`, the
five-draft benchmark arm, passed its declaration but stays `experimental`.
The KV columns are the capacity plan's pool estimates for 32
slots, of a text package and of one serving images (see
[Context parallelism](#context-parallelism)).

| Profile | Gated on | Speculation | DCP | KV tokens, text | With images | Configuration |
|---|---|---|---:|---:|---:|---|
| `tp4-mtp` (default) | FP8 | MTP, 3 drafts | 1 | 665,792 | 215,744 | `default` |
| `tp4-ar` | FP8 | none | 1 | 1,648,064 | 1,157,120 | the AR arms of each configuration |
| `tp4-mtp-dcp2` | FP8 | MTP, 3 drafts | 2 | 1,218,560 | 318,464 | `dcp2` |
| `tp4-mtp-dcp4` | FP8 | MTP, 3 drafts | 4 | 2,249,472 | 449,280 | `dcp4` |
| `tp4-mtp-k5` | FP8 | MTP, 5 drafts | 1 | 366,400 | does not fit | `k5` |

The profile settings are the package's serving defaults: 32 sequences and
8,192 batch tokens are the package's decode batch and prefill tokens per
step (`compiler/src/package.rs`), and the draft counts are those the
package's MTP drafter serves within its maximum of five. BF16 KV has no
profile; its comparison arm is `tp4-mtp` with `--kv-cache-dtype bf16`.

**How a profile is chosen.** The package declares the model option
`checkpoint.quantization_layout` with the admitted layout's id as its one
allowed value (`compiler/src/package.rs`), so an explicit value of another
layout fails the manifest check at startup; create reads the option back.
No shipped profile sets the option, so each serves every layout, and
`serving_profile` (`compiler/src/preparation.rs`) still refuses a profile
whose option names another layout, naming the field and both layouts. The
preparation provider (`select_profile`) picks `tp4-mtp` as the profile a
direct package carries for every layout. The autoregressive, five-draft and
context-parallel profiles are never picked by default; they are named, or
their options given explicitly (`--speculative-tokens 0` or `5`,
`--context-parallel-size`). A server started that way keeps the name of the
profile the provider selected (`tp4-mtp`) and reports the explicit values
in its served identity; a profile named with `--profile` keeps its own
name.

A released model's `tp4-ar` is qualified through the smaller of 64 and its
package maximum; GLM's (`preview.package_max_sequences`) is 32, so `tp4-ar`
stays at 32.

**Deploy defaults.** Each sampling default is taken from the checkpoint's
`generation_config.json`, which states only `temperature`, `top_p`,
`eos_token_id` and `pad_token_id`:

| Default | Value | Source |
|---|---|---|
| temperature | 1.0 | `generation_config.json` `temperature` |
| top-p | 0.95 | `generation_config.json` `top_p` |
| stop tokens | 154820, 154827, 154829 | `generation_config.json` `eos_token_id` (`vocabulary.stop_token_ids`) |
| top-k | off | not stated; Chopinfer's default |
| min-p, repetition penalty | off, neutral | not stated; Chopinfer's defaults, as in HF |
| reasoning | on, effort `max` | `frontend.json` `reasoning`, following `chat_template.jinja` (which renders Max); not the generation config |

`compiler/src/frontend.json` holds the first three as `generation_config`
and `vocabulary.stop_token_ids`. Where the file is silent Chopinfer's
defaults apply, not HF's effective ones: HF generation fills an omitted
`top_k` with 50, and its bare `generate` defaults to greedy decoding
(`do_sample=False`), neither of which the served API follows. Requests
override the sampling defaults and may pick another declared reasoning
effort (`low` or `high`), as on every model
([request surface](serving-request-surface.md)); a request that disables
reasoning is refused (the frontend declares `allow_disabled: false`).
`compiler/tests/frontend_goldens.rs` checks the declared fields and
omissions against the file of the official checkpoint
(`CHOPINFER_GLM5NEXT_CHECKPOINT`) and of each checkpoint
`CHOPINFER_GLM5NEXT_EXTRA_CHECKPOINTS` names, when set (a file may also
state `do_sample: true`, which the served API follows anyway); it does not
claim parity with HF's effective generation settings.

## GSM8K floor

**Adopted: 193 of 200 on the screen** (2026-10-11).
`qual.json` states it as `preview.gsm8k_floor`,
which `tools/qual/gsm8k.py` and the release preview read. At stage 5 the
full-split paired comparison against the previous gate's FP8 arm decides;
the screen floor is advisory where the two disagree.

- **Screen (200 items, `tools/qual/gsm8k.py`).** A bring-up prerequisite
  of the release. The floor is MiMo's on the same frozen items, the first
  200 of the test split. The final release gate's whole-split answers
  (2026-10-11) score
  those items at 196 for the FP8 arm; the first bring-up screen scored 197 (2026-10-07, before the fixed latent
  splits). Treating each item as an independent trial, an unchanged model
  at the final gate's FP8 rate (97.27%) scores below 193 with probability about
  18%, and at the earlier screen's 98.5% about 1.1%. A screen below the
  floor after a numerics change is therefore checked on the whole split
  before it is read as a regression.
- **Whole test split (1,319 items, `evals_gsm8k` arm `fp8`).** `qual.json`
  records it as the FP8 arm an external variant's paired GSM8K compares
  against (McNemar, p < 0.05), with no absolute floor. The final gate at
  C32: FP8 (`tp4-mtp`, FP8 KV, DCP 1) **1,283 of 1,319** (97.27%). The
  release-candidate run (2026-10-07) also scored 1,283. An
  absolute whole-split floor (1,269 of 1,319, the count an unchanged model
  falls below with probability under 1%) was proposed and not adopted.

## References and qualification

GPU references exist from the reference runner
([reference runner](reference-runner.md)): `models/glm5next/reference.py`
runs HF `glm5_next` (transformers 5.16.1) for the teacher, chat,
long-context (131,072 tokens) and image-probe references that
`models/glm5next/qual-references.lock.json` pins, and
`tools/refrun/encoder.py` runs the vision declaration in the exact, policy
and reference-bf16 modes. An external variant brings its own teacher
reference.
`models/glm5next/taps.json` declares the taps the engine and the reference
share, for tapdiff.

**Qualified (2026-10-11).** `qual.json` declares the default configuration's 16
checks (teacher, chat, the FP8 GSM8K arm, long context, needle, API,
structured output, frontend goldens, soak, multimodal, speculation, verify,
prefix and preemption exactness, determinism and media preprocessing), one
gate run against one `tp4-mtp` server on the release build. A TP4 server takes all four GPUs, so each
other configuration is declared in `qual.json`'s `configurations` and gated
against its own server: `dcp2` and `dcp4` (DCP 2 and 4), `k5` (five drafts
on the text package), `taps` (encoder parity on the diagnostic-taps build)
and `kv-bf16` (BF16 KV), which passed the final gate, was held on 2026-10-14, declared
again on 2026-10-15 and requalified on 2026-10-16 ([below](#bf16-kv)). Until 2026-10-13 each was a
declaration file beside `qual.json` (`qual.<name>.json`), which the
configurations reproduce. The final gate's results (below) were
adopted on 2026-10-11, so `mechanics_only`
is false in every configuration except `taps`, whose `text_layers` check
still needs its pinned layer reference
([layer reference](../models/glm5next/README.md#layer-reference)).

### Release qualification (stage 5)

The full requalification ran through the release preview
(`tools/qual/release_preview.sh run --model glm5next --suite full`) from
one release install built from main of 2026-10-14 13:11 UTC, each
configuration against its own four-GPU server, from 2026-10-14 14:40 UTC to
2026-10-15 00:03 UTC. That build holds the TP-split adoption
(2026-10-14) plus the evals_compare fix, KDA state and anchor spares
allocated before the KV pool, the layer check floored on HF BF16 with the
served FP8 latent, and release_preview's `--input-value`. `k5` was rerun
alone on 2026-10-14 with one more fix, which stops the determinism
records from sending an image to its text-only server. Later landings
were bound to this run by a reviewed check that their served paths and
GLM's declarations are unchanged. The run selected every configuration of
the official checkpoint then declared; the BF16 KV configuration was held
then and was not run
([below](#bf16-kv)). Release was enabled after the
run was reviewed.

Three configurations pass (default, k5 and dcp2). taps and dcp4 are
OPERATOR (the release preview's
status for a row that needs a reviewer's decision), each with one accepted
row and every other check passing. The accepted rows carry their numbers
below; none is a waiver and no limit changed. GSM8K
counts are of the whole test split (1,319 items); "against the seed"
pairs them with the arm in the stored seed.

| Configuration (server) | Result |
|---|---|
| `default` (`tp4-mtp`, FP8 KV, DCP 1, images) | PASS: teacher 0.0607 (decisive 0.77%, near tie 2.09%, all 2.86%, 3,781 positions); chat 0.190 against 0.25; long_context 0.0118 (1,536 positions); GSM8K FP8 arm 1,287 of 1,319 (against the seed: 6 right only here, 2 only there, McNemar p 0.289); soak host growth 345.4 MiB, GPU 184 MiB (42.7 and 16 MiB after warm-up), 4,811 requests; preemption_exact exact (36 requests, 2,248 positions); api, structured (validity 1.0), multimodal, media_preprocessing, needle, frontend_goldens, spec_verify, verify_exact, prefix_exact, determinism and both kv_offload checks pass |
| `k5` (`tp4-mtp-k5`) | PASS (rerun on 2026-10-14): teacher 0.0607 (all 2.86%); chat 0.190; spec_verify, verify_exact, determinism pass; preemption_exact exact (2,248 positions); 4.07 and 3.46 tokens per step (copy and natural sets) |
| `taps` (diagnostic-taps build) | OPERATOR: encoder_parity passes. text_layers is a row accepted on review (2026-10-14): 5 of 329 taps exceed their bound, first `L16.mlp.out`, on the whole-tap aggregate only, all from one of 31 rows where layer 16's routing is a near tie (the F32 reference routes expert 260, the TP build 92, at a 3.0e-5 gap between the 8th and 9th selection scores); without that row the five aggregates are 0.23 to 0.34 of the bound |
| `dcp2` (`tp4-mtp-dcp2`) | PASS: teacher 0.0603 (all 2.86%); chat 0.193; long_context 0.0105; spec_verify, verify_exact, prefix_exact, determinism and both kv_offload checks pass; preemption_exact exact (2,248 positions) |
| `dcp4` (`tp4-mtp-dcp4`) | OPERATOR: teacher 0.0603 (decisive 0.71%, near tie 2.12%, all 2.83%); chat 0.193; long_context 0.0127; spec_verify, verify_exact, prefix_exact, determinism and kv_offload_plain pass; preemption_exact exact (2,248 positions). kv_offload_pressure is a row accepted on review (2026-10-15): its one failing gate is `c16_prefix_evictions`, 0 against at least 1, and its other 24 gates pass (C1 36 of 36 gated, 0 bitwise differences, 107 swap-ins, 0 missed swaps, 0 accounting errors; C16 0 errors, 991 verified items). The scenario pins the device KV pool at 128 pages per rank, so a C16 prefix eviction is a pressure indicator with a narrow timing margin, not a correctness gate |

Soak limits: host growth 3,072 MiB and GPU 1,856 MiB, each 512 MiB after
the declared warm-up.

Model screens on `default`: GSM8K 200 scored 196 on the speculative and
the autoregressive arm, against the floor of 193; C1 greedy AR against
speculative matched (0 of 35 requests differ); determinism, speculative
against its repeat, matched (12 requests); the AR soak passed (host growth
258.7 MiB, GPU 136 MiB; 2.5 and 0 MiB after warm-up). A reseed moved
GLM's seed and last-good stores to this run's TP numerics on 2026-10-15.

### BF16 KV

BF16 latent KV is qualified: its requalification passed on 2026-10-16
(below). It has no profile of its own; serve it with `--kv-cache-dtype
bf16` on `tp4-mtp`. FP8 KV stays the default.

It was held on 2026-10-14 after the
TP-split candidate's `kv-bf16` teacher run (2026-10-14, split routed experts
with F32 MoE all-reduce partials) failed one absolute category gate:
prose-frozen top-1 disagreement 16 of 141 positions (0.1135) against the
0.10 limit. The BF16 reference's own spread in that category is 14 of 141,
the final gate's BF16 KV run had 11, and the pooled teacher was 0.0619 against 0.10
with every other category passing. The `kv-bf16` configuration and the
`teacher_arms` check only it ran were removed from `qual.json`, so stage 5
covered the other configurations. FP8 KV, the default, was
unaffected, and `--kv-cache-dtype bf16` still served BF16 KV, unqualified.

The configuration was declared again on 2026-10-15, unchanged,
now that small categories are judged by variant-spread.v2
([small categories](numerics.md#small-categories)): the 0.10 total limit of the 141-position prose-frozen category sits within two
binomial standard errors of the BF16 reference's own 14 of 141, so the
category also passes when the candidate's flips exceed the BF16 run's on
the same positions by at most two standard deviations of the discordant
count. Rescored this way, the 2026-10-14 run's 16 flips (9 shared with the
BF16 run, 7 of its own against the run's 5) pass. That run predates the
TP-split landing and its `teacher_arms` did not run, so BF16 KV stayed
unqualified until its own requalification on the landed rule passed.

The requalification ran on 2026-10-16 against `tp4-mtp` servers started
with `--kv-cache-dtype bf16`. The teacher, chat and `teacher_arms` checks
ran from a release install built from main of 2026-10-16 05:41 UTC, and
`teacher_arms` compared the BF16 arm with the default configuration's FP8
teacher arm, run from the same install that morning. prefix_exact ran from
an install built from main of 2026-10-16 08:42 UTC.

- teacher: passed in every category, 3,781 positions scored. Pooled mean
  |dlogprob| 0.0619 (limit 0.10); top-1 disagreement 2.83% (107
  positions), near tie 1.98%, decisive 0.85%. Prose-frozen (141 positions)
  had top-1 disagreement 0.1135 (16 of 141) against the 0.10 limit. That
  limit sits within two binomial standard errors of the BF16 reference's
  own rate there (0.0993, standard error 0.0252), so variant-spread.v2's
  paired rule judged it: 7 flips of its own against the BF16 run's 5, an
  excess of 2 within the margin of 6.93 (two standard deviations of the 12
  discordant positions). Its near tie (0.0780) and decisive (0.0355) rates are within
  their 0.08 limits.
- chat: passed, 4,340 positions. Mean |dlogprob| 0.1817 (limit 0.25),
  top-1 disagreement 6.59% (limit 10%), near tie 3.39% and decisive 3.20%
  (limits 8%).
- teacher_arms: passed by its declared rules (a passing comparison over at
  least 3,781 scored positions). Against the FP8 arm, the pooled mean
  |logprob difference| is 0.0645 and top-1 differs at 2.86% of positions;
  the highest category rates are prose-frozen's 6.38% and chat's 6.25%. It was judged on CPU
  from the run's two teacher captures by the same declared rules, because
  the release preview's own `teacher_arms` then resolved the BF16 capture
  to the wrong directory. That was fixed on main on 2026-10-16.
- prefix_exact: passed on 2026-10-16. Over 1,745 scored positions of 27
  texts, the prefix-cache hit rows and the chunked-prefill rows equal the
  cold rows bit for bit (raw logprob and top logprobs). Over 1,616
  positions of 25 texts, the rows that resume a prompt-end anchor also
  equal their cold rows. Cached-token counts followed the anchor rules at
  a reuse unit of 64 tokens (one KV page), with no cache errors, and the
  media and acceptance checks passed.

<a id="final-gate-gates4"></a>

### Final release gate (2026-10-11)

The final release gate ran in two parts, completed on 2026-10-11: part A,
the jobs that never draft (the autoregressive arms, encoder parity), and
part B, every job that serves the drafter plus the autoregressive control
arms. Part A ran on a build of main from 2026-10-11 08:17 UTC. Part B ran
on a build from 09:56 UTC the same day, which differs in the drafter: it
drafts GLM's tokens on the shared sharded argmax kernels instead of GLM's
own draft argmax kernels (bundle lowering, CUDA source and kernel locks).
Part A's jobs never draft. Main after part B's build only deleted tests and
renamed a crate.
Each configuration ran on its own four-GPU server. Part A passed every
job. In part B every job of the configurations below passed except the
default configuration's soak, which failed the host-memory limit under the
plain rule, from the bounded warm-up of pinned last-chunk programs. It was
waived on 2026-10-11; the soak now declares that warm-up (180 s) and the
programs' bound (2,560 MiB host, 1,344 MiB GPU), and under that rule it
passes (models/glm5next/README.md, "Qualification checks", gives the
numbers and the GPU headroom).

| Configuration (server) | Result |
|---|---|
| `qual.json` (`tp4-mtp`, FP8 KV, DCP 1, images) | teacher 0.0629 (decisive 0.98%, near tie 1.96%, all 2.94%, 3,781 positions); long_context 0.0026 (1,536 positions); api, structured (validity 1.0), multimodal, media_preprocessing, spec_verify, verify_exact, determinism and prefix_exact pass; preemption_exact exact (36 requests, 2,196 positions); GSM8K FP8 arm 1,283 of 1,319; soak host growth 1,077.5 MiB, GPU 440 MiB (43.6 and 24 MiB after 180 s): failed the plain 512 MiB host limit, passes the declared warm-up rule |
| `taps` (diagnostic-taps build) | encoder_parity pass (part A, autoregressive server); `text_layers` not run |
| `k5` (`tp4-mtp-k5`) | teacher 0.0629; spec_verify, verify_exact, determinism pass; preemption_exact exact (2,196 positions); 4.11 and 3.72 tokens per step (copy and natural sets) |
| `kv-bf16` (BF16 KV) | teacher 0.0620 (decisive 0.74%, near tie 2.17%, all 2.91%); teacher_arms reported against the FP8 arm |
| `dcp2` (`tp4-mtp-dcp2`) | gate passed: teacher 0.0640 (all 2.80%), long_context 0.0029, spec_verify, verify_exact, prefix_exact, determinism; preemption_exact exact (2,202 positions) |
| `dcp4` (`tp4-mtp-dcp4`) | gate passed: teacher 0.0658 (decisive 0.90%, near tie 1.96%, all 2.86%; chat 0.185 against 0.20), long_context 0.0025, spec_verify, verify_exact, prefix_exact, determinism; preemption_exact exact (2,196 positions) |

The MTP drafter at three drafts made 3.30 and 3.11 tokens per step (copy
and natural sets), with acceptance 0.78 and 0.73.

### Coverage checks (2026-10-11)

Three checks the final gate did not run were added afterwards and passed
on 2026-10-11, on the build that adopted the gate's results, one server per
configuration:

- `frontend_goldens`: a served `/tokenize` against transformers' render
  of the pinned chat template and tokenizer (31 chat rows: role tags,
  reasoning, efforts low, high and max, tool definitions, tool calls with
  non-string arguments and tool results, the chat teacher conversations;
  27 tokenizer texts).
- `chat`: 17 chat texts, 4,340 positions, teacher-forced against their own
  HF reference, limits from its BF16 spread with headroom factor 1.25.
- `needle`: three needles at total contexts of 8,192, 131,072 and 262,144
  tokens (the served `tp4-mtp` reports `max_model_len` 360,960;
  429,120 on the 2026-10-07 build).

The long-context reference was extended from 65,536 to 131,072 tokens at
the same time, and `long_context` ran again against it.

| Server | frontend_goldens | chat: mean (limit), decisive, all top-1 | needle | long_context (131,072) |
|---|---|---|---|---|
| `tp4-mtp`, DCP 1 | 31 + 27 rows equal | 0.1915 (0.25), 3.2%, 6.3% | 3 of 3 | 0.0114, all top-1 0.39% |
| DCP 2 | | 0.1935 (0.25), 3.8%, 6.8% | 3 of 3 | 0.0108, 0.52% |
| DCP 4 | | 0.2015 (0.25), 3.7%, 6.8% | 3 of 3 | 0.0123, 0.33% |
| five drafts, text | | 0.1915 (0.25) | | |
| BF16 KV | | 0.1899 (0.25), 3.6%, 6.8% | | |

The chat reference's own BF16 run sits at 0.180, so the served builds are
at most 1.12 times its BF16 spread; the tool-call conversations carry most
of the error. Every needle answer gave the three values after a short reasoning
(`</think>`). The 262,144-token rows took 43 to 56 s to their first token,
reusing the 131,072-token row's cached prefix. `capabilities.frontend` and
`capabilities.long_context` are true in `qual.json`; the DCP 2 and 4 declarations claim `long_context` through
their needle checks. `logprobs_parity` stays undeclared: it compares a
reference OpenAI server's logprobs with the candidate's, and GLM's
reference is HF through the reference runner, not a server.

### Earlier gate runs

The first gate runs (2026-10-07, before the fixed latent splits)
passed the default configuration's teacher (pooled mean |dlogprob| 0.065
against 0.10), long context, API, structured output, speculation and verify
exactness, and failed preemption exactness at concurrency 16, the batch
variance [Batch invariance](#batch-invariance) describes. They are
superseded by the final runs on the release-candidate install (built
2026-10-07, run 2026-10-07 and 2026-10-08, four GPUs, one server per
configuration). Every check below ran under its declaration with
`mechanics_only` true; the final release gate above supersedes them.

| Configuration (server) | Check | Result |
|---|---|---|
| `qual.json` (`tp4-mtp`, FP8 KV, DCP 1, images) | teacher | pass: pooled mean \|dlogprob\| 0.0653 (limit 0.10), top-one decisive 0.71% (8%), near tie 2.04% (8%), all 2.75% (10%), 3,781 positions, no category failed |
| | long_context | pass: mean 0.0024, decisive 0.07%, 1,536 positions of the 64K reference |
| | api, structured | pass; structured validity 1.0, none refused or truncated |
| | multimodal, media_preprocessing | pass |
| | spec_verify | pass: copy set 1,024 scored positions, MTP equal to autoregressive |
| | verify_exact | pass: three-draft verify against the `tp4-ar` arm's forced tokens, full coverage |
| | determinism | pass: the MTP record equals the `tp4-ar` record bit for bit, image request included |
| | preemption_exact | pass: c16, 36 requests, 2,210 positions after the first preemption, exact |
| | prefix_exact | pass |
| | soak | pass: 1,380 s at C32, 2,765 requests, device memory growth 0 MiB, host 247 MiB (limit 512) |
| | evals_gsm8k (FP8 arm) | pass: 1,283 of 1,319 at C32 (see [GSM8K floor](#gsm8k-floor)) |
| `taps` (diagnostic-taps build) | encoder_parity | pass |
| `k5` (`tp4-mtp` at five drafts, text package) | teacher | pass: the same scores as DCP 1 (the teacher scores prompt tokens, which no draft touches) |
| | spec_verify, verify_exact, determinism | pass, at exactly five drafts |
| | preemption_exact | pass: c16, 36 requests, exact |
| `kv-bf16` (`tp4-mtp`, BF16 KV) | teacher | pass: pooled mean 0.0569, decisive 0.61%, near tie 1.85%, all 2.46% |
| | teacher_arms (reported, not gated) | FP8 minus BF16 mean \|dlogprob\|: 0.0084 pooled; chat 0.020, prose 0.032, prose-frozen -0.016 |
| `dcp2` (`tp4-mtp`, DCP 2) | teacher | pass: pooled mean \|dlogprob\| 0.0673 (limit 0.10), decisive 0.69%, near tie 2.12%, all 2.80%, 3,781 positions, no category failed |
| | long_context | pass: mean 0.0021, decisive 0.07%, 1,536 positions of the 64K reference |
| | spec_verify | pass: copy set 1,024 scored positions, MTP equal to autoregressive |
| | verify_exact | pass: three-draft verify against the AR arm's forced tokens, full coverage |
| | determinism | pass: the MTP record equals the AR record bit for bit, text requests |
| | preemption_exact | pass: c16, 36 requests, 2,201 positions after the first preemption, exact |
| | prefix_exact | pass |
| `dcp4` (`tp4-mtp`, DCP 4) | teacher | fail: chat mean 0.2025 (limit 0.20); pooled mean 0.0688 passes (limit 0.10), decisive 0.66%, near tie 2.09%, all 2.75%, 3,781 positions, no other category failed |
| | long_context | pass: mean 0.0019, decisive 0.07%, 1,536 positions of the 64K reference |
| | spec_verify | pass: copy set 1,024 scored positions, MTP equal to autoregressive |
| | verify_exact | pass: three-draft verify against the AR arm's forced tokens, full coverage |
| | determinism | pass: the MTP record equals the AR record bit for bit, text requests |
| | preemption_exact | pass: c16, 36 requests, 2,195 positions after the first preemption, exact |
| | prefix_exact | pass: 2026-10-08 21:29 UTC, after the media sizing fix |

On 2026-10-08, DCP 2 passed 7 of 7 gate checks. DCP 4's gate stopped at the teacher failure;
its five remaining checks other than prefix_exact passed separately by
22:31 UTC, with prefix_exact as recorded above.

DCP 4 needs no chat waiver. The final release gate (2026-10-11) passed its
teacher with every category: pooled mean 0.0658, chat mean 0.185
against 0.20 (top-one decisive 2.9%, all 6.4%, 560 positions). The
release-candidate failure above came from one rounding-sensitive tool-call text,
which carried 98.8% of the chat increase over DCP 1. At CP > 1, rank-local
softmax maxima change exponentials and PV probabilities; the partials and
merge are F32. Only 2 of 7 chat texts cross a 64-token page, so a
regression on longer chat is not excluded; a follow-up extends the chat set
with longer texts.

The release candidate's BF16 KV gate run reported `passed` true and `qualified`
false, as `mechanics_only` then required.

### Error against the BF16 floor

An accuracy audit of the release candidate (2026-10-08) scored
the teacher gate's served captures against the reference's BF16 spread,
by [the method in numerics](numerics.md#served-error-against-the-bf16-floor):
per position, |served - F32| minus |BF16 - F32| over the 3,781 teacher
positions of 35 texts, with 95% intervals over whole texts (Student t).

| Set (texts) | BF16 run | FP8 KV | BF16 KV | FP8 KV excess | BF16 KV excess |
|---|---:|---:|---:|---|---|
| pooled (35) | 0.0627 | 0.0653 | 0.0569 | +0.0026 (-0.0012, +0.0064) | -0.0058 (-0.0114, -0.0002) |
| chat (7) | 0.158 | 0.174 | 0.154 | +0.016 (+0.008, +0.025) | -0.004 (-0.015, +0.008) |
| long_4k (1) | 0.0005 | 0.0014 | 0.0012 | +0.0009, no interval | +0.0007, no interval |

- Pooled, the FP8 KV build is 4% above the floor and the interval does not
  exclude zero; that does not show it equals the floor. The BF16 KV build
  is below it (the BF16 run rounds weights and scores the engines do not).
- Chat is above its floor with FP8 KV and not with BF16 KV; FP8 minus BF16
  KV is +0.020 in chat (+0.006, +0.034) and +0.0084 pooled (+0.0021,
  +0.0147). That excess follows the FP8 latent cache. With seven chat
  texts the interval is approximate.
- long_4k, the one teacher text where sparse selection is active, sits
  above its floor in both KV formats, so not because of the KV format. Its
  excess comes from four positions of 0.10 to 0.15 nats; one text gives no
  interval, and it is not localized. The long-context check's 64K tail
  (1,024 positions, one text) is 0.0029 against the BF16 run's 0.0028.
- In code, math, multilingual and prose the FP8 KV excess interval
  contains zero (five to nine texts each); prose-frozen (three texts) has
  no interval. The f64 run is 0.0005 from F32, so the F32 gate is
  converged.

## Deploy and serve

The deployment commands, model options, activation policy and planned
capacity are in [deploy GLM 5.3 Flash](deploy-glm5next.md). CPU tests and
the GPU encoder checks:

```sh
# Contract, compiler and bundle tests (CPU; checkpoint header checks with
# CHOPINFER_GLM5NEXT_CHECKPOINT and CHOPINFER_GLM5NEXT_EXTRA_CHECKPOINTS set)
cargo test -p model-glm5next-contract -p model-glm5next -p bundle-glm5next --locked --offline

# Encoder conformance and the whole-tower check (one GPU)
CHOPINFER_KERNEL_GPU=0 cargo test -p kernel-lab --release --test encoder_conformance_glm5next
CHOPINFER_KERNEL_GPU=0 CHOPINFER_GLM5NEXT_ENCODER_TOWER=<new dir> \
  CHOPINFER_GLM5NEXT_ENCODER_REFERENCE=<dir with vision-exact and vision-policy> \
  cargo test -p kernel-lab --release --test glm5next_encoder_tower -- --nocapture
```
