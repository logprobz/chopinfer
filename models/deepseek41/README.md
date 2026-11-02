# DeepSeek V4.1 Flash (`deepseek41`)

DeepSeek V4.1 Flash, from the official checkpoint
`deepseek-ai/DeepSeek-V4.1-Flash-1017` at revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`, served as bundle
`deepseek41.v1` at topology `deepseek41-flash-tp4`. It is not released:
`model.toml` declares `[release] enabled = false` until it qualifies.

Supports images in tool results through Chat Completions and Responses, in the official encoder's prompt order,
and in system and developer messages through Chat Completions, as deepseek-recipe 0.2.0 reads them.
Images may be PNG, JPEG, GIF (the first frame) or WebP (lossy, lossless,
with alpha, or the first frame of an animation), decoded as the reference's
Pillow `Image.open` and `convert("RGB")` read them, and may arrive as
`image_url` parts or OpenAI `file` parts whose `file_data` is an image data
URL (deepseek-recipe 0.2.0's mapping, read from OpenAI's nested file object;
the recipe's flat fields are refused). See [the request surface](../../docs/serving-request-surface.md#media-inputs).

Client text is plain text, as deepseek-recipe 0.2.0 renders a prompt. The
format names the special tokens of its prompt (`vocabulary.explicit_special_tokens`
in `frontend.json`): those the encoder's template writes, the tool
instructions and DSML markup included, each image's placeholder, and the
grounding tokens (`<｜ref｜>`, `<｜box｜>`, `<｜point｜>`, `<｜polygon｜>` and
their closes) where assistant content or reasoning spells them. The host
encodes all other text, message content, tool results, tool schemas, the
response format and call arguments included, without the added vocabulary,
so a special token spelled there is ordinary tokens: a tool result cannot
forge a turn, and an added token or the image placeholder typed in text is
accepted as text. The parser reads `<think>`, `</think>` and `｜DSML｜` only
from their token IDs (`vocabulary.id_only_markers`), in a calls block too,
so the model spelling them in ordinary tokens writes text, and a forced
call's constraint admits each marker's `｜DSML｜` only as that token. See [the request
surface](../../docs/serving-request-surface.md#request-parameters-and-defaults)
and `explicit-special-tokens` in [frontend-divergences.json](frontend-divergences.json).

| Crate | Directory | Holds |
|---|---|---|
| `model-deepseek41-contract` | `contract/` | IDs, geometry, projection declarations and kernel contracts the compiler, bundle and lab share |
| `model-deepseek41` | `compiler/` | Checkpoint admission, the tokenizer contract and the package |
| `bundle-deepseek41` | `bundle/` | The program; SM120 registration in `src/sm120/` |
| `model-deepseek41-lab` | `lab/` | Kernel GPU correctness tests on `kernel-harness`; not a default workspace member |

Build and test the roles with Cargo's `--locked --offline` flags:

```sh
cargo test --locked --offline -p model-deepseek41-contract -p model-deepseek41 -p bundle-deepseek41
```

The lab runs only when named, and its GPU tests only when
`CHOPINFER_KERNEL_GPU` or `CHOPINFER_KERNEL_GPUS` names idle GPUs:

```sh
cargo test --locked --offline -p model-deepseek41-lab --test sm120
```

CPU checks that read the checkpoint take it from
`$CHOPINFER_DEEPSEEK_CHECKPOINT`; the tokenizer corpus test also reads the
HF corpus `tokenizer_goldens.py corpus` writes, named by
`$CHOPINFER_DEEPSEEK_TOKENIZER_CORPUS`.

## Served profiles

`profiles.json` ships two experimental profiles, `tp4-ar` and `tp4-dspark`.
Both use native KV, prefix caching, at most 32 sequences, 8192-token prefill
chunks and exact upper-layer dependency cones. AR reserves 320 MiB per rank
for graphs; DSpark reserves 640 MiB. Package capacity and CPU checks do not
establish release qualification.

The scheduler admits 64-token aligned shared-prefix anchors in addition to
the chunk grid and prompt ends. For an unaligned prompt, it cuts at
`prompt_tokens / 64 * 64` and executes the remaining 1 to 63 tokens in a
separate prefill step. Each endpoint produces complete target state and,
for DSpark, the 134-row raw feature history, including six preserved older
rows. An anchor costs 53 pages (2.88 MiB) per rank for AR or 125 pages
(6.79 MiB) for DSpark. See [prefill dependencies](prefill-dependencies.md).

On 2026-10-27, TP4 shared-document and multi-turn comparisons measured a
cached-prompt-token rate of 66.46%, against 66.35% for the full schedule and
35.56% for the previous upper-skip schedule. Matched-hit TTFT fell 72.2% for
AR and 72.3% for DSpark against that previous schedule. The anchors and
grid-only arms still published response-end anchors; the full-prefill arm
did not. Those anchors are no longer published, so the measured matched-hit
TTFT gain over full prefill may be overstated. No corrected measurement is
available. Cold unaligned
prompts cost an additional 39.8 ms for AR (941 to 981 ms) and 29.3 ms for
DSpark (966 to 996 ms). The separate full-prefill and 2048-token profiles
were removed after this comparison. `prefill_upper_skip` and
`prefill_chunk_tokens` remain declared options for diagnostic configurations.

`tp4-dspark` serves `--prefill-policy adaptive` at a 750 ms decode cadence
target with adaptive timing cuts; `tp4-ar` keeps the fixed budget and whole
grid chunks. In the 2026-10-30 stateful replay A/B, adaptive resolved
gains against `off` in collapse share and gaps over 1 s in a C16
prefill-saturated cell and a cold burst of five 250K and five 30K prompts,
and held the C8 long-append TTFT p95 increase to 1.9% with a 1.6% output
cost. Its costs are lower output rates (9.9% at C16 and 24.6% in the
burst) and a later first token for the burst's last 250K prompt. Outputs
matched `off` bit for bit. See the
[adoption](../../docs/serving-profiles.md#adaptive-timing-cut-adoption).

AR decode, eligible AR prefill and DSpark target execution use prepared
per-layer Engram gathers. Their shared layer-gather lifecycle enqueues
mapped reads before the consuming layer and joins them before its decoder.
Upper-skip DMA prefill from 512 rows retains the synchronous gather, as do
media, taps and hosts without the needed service. Every host gather has an
`engram_gather` route tag; remaining fallbacks are recorded in the bundle's
route allowlist. See [Engram gathers](lab/prefill-engram.md) and
[prefill communication](prefill-communication.md).

The Engram codes and scales are about 203 GB of checkpoint tables, mapped
from the file and read through the host page cache. The weight plan keeps
`prewarm: false`. On 2026-11-01, four GPUs
at 250 W each with 251 GB of host RAM measured mean GPU gather waits of
0.04, 0.06, 0.08 and 0.03 ms per step at C1, C8, C16 and C32 with warm
tables (0.1-0.3% of a step). Evicted tables caused up to 1,150 major faults
per visit and multi-millisecond gather waits. An external sequential read
restored 99% residency in about 60-70 seconds; the in-process background
warm has not been timed at this size.

`serve --prewarm-mapped-views off|on|auto` defaults to `off`. `auto` warms
the tables in the background after readiness if `MemAvailable` can hold
the mapped bytes plus the 16 GiB margin.

A live server job can drop its external sequential read of the tables
before `serve` and pass `--prewarm-mapped-views auto`. If traffic must not
meet cold rows, wait for the first `mapped prewarm:` line other than the
`warming anyway (on)` warning. The JSON done line with `stopped: false`
signals completion; auto decline (short or unreadable RAM), warm failure
and worker spawn failure do not. When residency is visible, the required
gauge ratio can also establish warm tables; when hidden, use the done line.

The sampler runs whatever the option and publishes
`chop_mapped_view_bytes`, `chop_mapped_view_resident_bytes`,
`chop_mapped_view_residency_visible` and
`chop_mapped_view_residency_sample_seconds`.
See [mapped-view monitoring](../../docs/operations-observability.md#weight-loading-counters)
for the full visibility rule, endings and warnings, and
[example alerts](../../docs/operations-observability.md#example-alerts).

## Kernel selection and storage

Flash selection uses nine [schedule tables](contract/tuning/README.md),
with inclusive row ranges. Other geometries do not inherit Flash tuning.
Missing optional rows select the model's baseline. Required descriptors are
`dense_fold.status.decode` at row 32 and `dense_fold.status_wide.verify`,
`dense_fold.attention_q_a_status.verify`, `dense_fold.attention_o_a_status.verify`
and `dense_fold.attention_o_b_status.verify` at row 64, all with `mxfp8` weights,
`ranks=4` and bucket equal to that row count. Table loading refuses missing
descriptors and unsupported schedules or knobs. Geometry, bank, graph, storage,
workspace and precedence checks remain code gates. Route declarations use
the table admissions, and untuned-key tests pin uncovered served ranges.

Dense decode uses one template for split, ticket, prefetched and two-group
reuse folds. Local rows share the local-wide template; cluster folds share
the fused cluster body; compact and matrix prefill share a fetch knob.
Chained MoE floor/bandwidth and absorb/strip schedules share bodies with
separate scheduling knobs. Latent shared-head modes use a template parameter.
The fold families share admission, copy, packed-fetch and merge helpers where
the compiler preserves their bytes. Activation, scale, chain and reduction
policies remain separate numerical identities, not tuning knobs.

The direct MoE's absorbing path at its 32- to 64-row buckets on Flash TP4
(the backbone bank's 32-row decode and verify buckets and its fused 33- to
64-row verify buckets) serves four cache policies from the MoE table's
[direct MoE knobs](contract/tuning/README.md#direct-moe-knobs):
its queues stream their weight codes past L1 with an L2 evict_first policy
and store their K-block partials with an L2 evict_last policy, its
finishers discard each partial line from L2 once read, and its gate
finisher issues every partial load before its first add. They keep every
loaded value, product, add and output. The set replaced the
`moe_direct_stream`, `moe_direct_keep`, `moe_direct_retire` and
`moe_direct_loads` candidate options on 2026-11-01, adopted together after
served exactness passed at C8 to C32 and on the wide verify buckets. Served
step time with all four on improved 4.25% at C8 with 8 items and 32 verify
rows (spread 0.09), 5.68% at C16 with 16 items and 64 verify rows (spread
2.01) and 3.60% at C32 with 32 items and 32 verify rows (spread 0.54, after
a discarded lead visit with a clear first-visit control); C1 with one item
and 6 verify rows, which the knobs do not reach, was neutral at 0.06%. With
`wide_verify`, the wide verify buckets of 65 to 192 rows keep the plain
route and their table row sets no knob, because they have no served
step-time measurement. The knob entries read raw or width-1 tile scales; a
bank the census encodes at width 2 or 4 expands its scales first. See the
[lab qualification](lab/moe_direct_small.md#served-body-candidates).

Every MXFP8 dense projection stores E4M3 codes in
`mxfp8-fold-packed-8x128-v1`. It needs no create-time packed duplicate.
The 2026-10-23 TP4 comparison with 65,536 KV pages kept AR and DSpark outputs
bitwise equal, with greedy throughput ratios 0.9997-1.0103 over C1/C8/C32.
It freed 1922 MiB per rank before graph capture, or 2116 MiB with the drafter.
Retired builders and row-major controls needed by current qualification live
in the lab; [packed fold coverage](lab/packed_fold_tests.md) describes them.

Native Flash TP4 prefill uses block-scaled expert tiles over the trained
operand grids. The backbone gate/up codes use 64-row by 128-K tiles with
quad-lane ordering. TP1/TP2 and decode, verify and draft arithmetic keep
their own admitted paths. Native prefill changes the F32 order within a
32-K group, so its qualification is within-bound, not bitwise to the exact
BF16 expert path. See [native MoE](lab/native_moe_prefill.md).

Main-layer expert scales use lossless `scale-row-tiles-offset-v1` at TP4
and `scale-row-offset-v1` at other topologies. The census chooses an encoding
width or keeps each bank raw. Eligible tile-scale consumers read encoded
banks directly; other consumers expand them to raw scratch. Drafter banks
remain raw because their one-row floor reads scales before its PDL wait.
The TP4 tile format measured 673.39-673.46 MB per rank on 2026-10-24 and
decoded bitwise to the raw banks. The earlier row-offset served comparison
on 2026-10-23 freed 3296 MiB per rank and cost 2-4% decode throughput at
C1/C8/C32, with unchanged output bits. The host's
`CHOPINFER_WEIGHT_CENSUS=0` selects raw banks.

Native TP4 prefill rounds completed local partials to BF16, sums ranks in
ascending F32 order and rounds the result once. Decode, target verify and
draft blocks keep F32 partials. With the `phased_decode_reduce` candidate
option, their reductions that no HC boundary takes the round of (four per
rank at every width, and all 81 at the verify widths that defer no round)
run the phased F32 rounding entries, which sum in the same rank order and
round with the same instruction ([decode reductions](lab/decode-reductions.md)). The shared ordered DMA transport preserves
the selected collective's rounding. See
[partial arithmetic](communication-partials.md) and
[attention admission](prefill-attention.md).

## Checkpoint and geometry

`checkpoint-files.json` pins the official revision. Admission reads
`config.json`, checks the block-32 FP8 recipe and CSA2/Engram/DSpark
relations, then matches `geometries/flash.json`. `contract/projections.json`
declares each projection, side tensor and encoder tensor. Checkpoint binding
claims all 96,085 official tensors once before payload loading.

The 12-layer synthetic geometry, with every CSA2 role, 32 experts and one
DSpark stage, admits TP1/TP2/TP4 when `CHOPINFER_DEEPSEEK41_GEOMETRY_DIRS`
names `fixtures/deepseek41/geometries/synthetic-small`. These are test
geometries, not additional served profiles.

Native FP4 index keys use the guarded tensor-core scorer; BF16 keys use
SIMT. A scale certificate admits exact native scores, otherwise the original
SIMT order repairs the slot. Unrepackable queries use SIMT for the whole row.
The fine certificate requires scale sums in [130, 369], spread at most nine,
and individual scale bytes in [2, 252]. The
[guard argument](../../crates/kernels-sm120/cuda/pooled_indexer/position_mma_guard.md)
states its arithmetic domain.

Hierarchical merge serves native Flash decode rows 1-32. Eager cached
prefill selection requires a rounded 128K context extent, at least 32
prepared rows and top-k 512. Full C8/C32 decode and eight-item verify-32
use the tiled scorer and cached selector from 128K through 256K;
verify-32 hierarchy starts at 32K. Other layouts keep their admitted
fallbacks. Prefill tile scoring has separate low-context and layer-20
publication gates. The table ranges do not override those code checks.

## Model options

Set these package-declared options with `--model-option NAME=VALUE`. The
host records every resolved value in serving identity, metadata and
metrics. The table shows the Flash package defaults; profiles can override
them. A non-default value is unqualified.

| Option | Default | Meaning |
|---|---|---|
| `diagnostic_step_logs` | `false` | Emit DSpark phase, prefetch, Engram, commit and drafter-confidence JSON records. |
| `dspark_confidence_staged` | `false` | Candidate: drafter confidence heads load each thread's 128-weight chain into registers before the dependency wait and stage each row's operands in shared memory, keeping every chain, fold and score ([drafter cost](lab/drafter-cost.md)). Bitwise lab pairs and fixed-batch GPU replays matched. Not adopted: no served A/B was run, since its estimated gain of about 0.3% is below the 1.5% bar for a new variant. |
| `dspark_coupled_drafts` | `true` | Couple sampled drafts to the target sampler's noise. |
| `dspark_draft_chunk_items` | `12` | Drafting items per forward, from 1 to 32. |
| `dspark_graph_handles_per_rank` | `512` | Graph handles per rank, from 0 to 9223372036854775807; 0 keeps DSpark eager. The total across ranks must fit the platform's count range. |
| `dspark_hc_draft_rows` | `false` | Candidate, with `dspark_hc_fusion`: drafter forwards of one to six items (5 to 30 block rows) admit the HC finish route through the drafter's own table, so their stage boundaries fuse too ([drafter cost](lab/drafter-cost.md)). Fixed-batch GPU replays at C1 to C12 and C16 matched bitwise. Not adopted: the pair's served A/B on 2026-11-01 was invalid, and its estimated gain (about 0.8% at C1, 0.4 to 0.5% at C8 and C16) is below the 1.5% bar. |
| `dspark_hc_fusion` | `false` | Candidate: the drafter's stage boundaries fuse their all-reduce round, expansion, next mixer and input norm where the finish route admits the forward's rows (7 to 12 items), as the backbone's do ([drafter cost](lab/drafter-cost.md)). Fixed-batch GPU replays at C1 to C12 and C16 matched bitwise. Not adopted: the pair's served A/B on 2026-11-01 was invalid, and its estimated gain (about 0.8% at C1, 0.4 to 0.5% at C8 and C16) is below the 1.5% bar. |
| `dspark_markov_staged` | `false` | Candidate: drafter Markov steps of two or more items load each thread's weight chain into registers before the dependency wait and read each position's logits in place, with no row copy ([drafter cost](lab/drafter-cost.md)). Bitwise lab pairs and fixed-batch GPU replays matched. Not adopted: its served single-option A/B on 2026-11-01 gained 2.14% in C16 step time but lost 0.70% in C8 decode request throughput, a resolved regression. |
| `dspark_moe_direct` | `false` | Candidate: drafter forwards whose block rows the drafter table admits (`moe.direct_drafter.verify`) run the routed experts through the direct path instead of the expert chains ([drafter cost](lab/drafter-cost.md)). Fixed-batch GPU replays at C1 to C12 and C16 matched bitwise. Not adopted: no served A/B was run, since its estimated gain of about 0.15% is below the 1.5% bar. |
| `phased_decode_reduce` | `false` | Candidate: Flash TP4 decode and verify all-reduces whose round no HC boundary takes launch the phased F32 rounding entries, over comm state of their own, instead of the F32 reduce-scatter, all-gather and round row. They give the same bits, in the lab at 11 row counts from 1 to 192 and in fixed-batch replays ([decode reductions](lab/decode-reductions.md#fixed-batch-replays)); the deferred pairs and prefill are unchanged. Not adopted: served A/Bs on 2026-11-01 gained 0.12% and 0.68% in C1 and C8 step time and were neutral at C16 and C32, below the 1.5% bar. |
| `prefill_chunk_tokens` | `8192` | Guaranteed chunk ends: 0, 2048 or 8192 tokens. |
| `prefill_plan_cache_plans` | `16` | Kept short-prompt plans, from 1 to 64; evict the least recently used plan when full. |
| `prefill_plan_cache_verify` | `false` | Lower each cache hit and each reused plan template again and refuse a differing plan. |
| `prefill_upper_skip` | `true` | Use upper-skip prefill; requires nonzero guaranteed chunk ends. |
| `shared_overlap_max_rows` | `4294967295` | Largest graph bucket that overlaps shared-expert and eligible window-KV work. Positive values only; the default covers every admitted bucket. |
| `text_config.candidate_topk_blocks` | `2048` | Candidate block pool, checked with index top-k and the partial newest block. |
| `text_config.index_topk` | `512` | Index selection top-k, from 1 to 512. |
| `text_config.indexer_mma` | `true` | Use the guarded native Flash index scorer where eligible. |
| `text_config.indexer_prefill_tile` | `true` | Enable the qualified Flash tiled indexer routes. |
| `verify_graph_rows` | `64` | Widest graphed verify step; wider admitted steps run eagerly. |
| `wide_draft` | `false` | Candidate, requires `wide_verify`: a Flash TP4 drafter step of more than twelve drafting items runs them in the fewest forwards the widest wide verify bucket holds (25 items at 128 verify rows, 32 at 192), on the wide dense folds and the head's exact F32 row tiles, instead of chunks of at most twelve. Fixed-batch GPU replays matched the default bitwise. Kept off with `wide_verify` at both measured power limits (see its row). |
| `wide_verify` | `false` | Candidate: Flash TP4 eager verify steps of 65 to 192 rows prepare their row-local stages and HC boundaries at the wide verify buckets (128 rows and the verify rows) and launch the decode entries a 64-row step uses, fold their dense projections through wide folds at those buckets, run their routed experts through the wide direct MoE queues, and run the vocabulary head, router and compressor in their measured exact F32 row tiles. Fixed-batch GPU replays and kernel-lab MoE pairs matched the default bitwise. Kept off on this host: served depth sweeps on 2026-11-01, at a 250 W limit per GPU, and on 2026-11-02, at the 300 W default limit, chose no step above 64 rows (see the served acceptance paragraphs below). |

TP4 native-KV single-item DMA prefill overlaps both prepared Engram gathers
by default at 512-8192 rows. The model joins each layer before its decoder;
media, taps and unavailable prepared row services retain their existing
fallbacks. See [mapped-row read-ahead](../../docs/mapped-row-read-ahead.md)
for the adoption evidence and remaining synchronous callers.

Flash TP4 verify steps of at most 6 rows launch the decode steps' fused
query norm and ceil and fused window norm, RoPE, ceil and store; the
drafter's forwards of at most 6 rows (one drafting item) fuse their
queries, and its window writes of at most 6 rows quantize with the
parallel ceil32 entry instead of the serial one. The latent tuning row
`latent.row_fusions.verify` admits rows 1 to 6: verify steps and drafter
forwards by their rows, and the drafter's writes by each write's own rows.
Verify steps of 7 to 32 rows, wider drafter forwards and wider writes keep
the unfused route. The route was adopted on 2026-11-01 from the
`verify_row_fusions` candidate option, which ran the fusions at verify
steps and drafter forwards of at most 32 rows and the parallel ceil32 at
drafter writes of at most the draft rows, and which was then removed. Its
lab pairs and fixed-batch GPU replays were bitwise, and served exactness
passed at C1, C8 and C16. In the served step-time A/B of the option
against the profile, with the first-visit control clear, C1 (one item, six
verify rows a step) gained 1.85%, resolved, with paired deltas of 1.74%
and 1.95%; C8 (eight items, 32 rows a step) gained 1.45%, neutral, with
paired deltas of 0.54% and 2.35% and a spread of 1.81 points; C32, which
does not draft, was untouched. Timing acceptance needs a resolved gain at
every admitted width, so the target fusions are admitted only at the
six-row width C1 resolved. The drafter's forwards and writes are admitted
by their own rows at the same range: they are bitwise in the lab pairs and
replays, and the C8 cell already ran a superset, every drafter forward of
up to 32 rows fused and every write of up to 60 rows on the parallel
entry, without a regression in any cell ([small-kernel
latency](lab/small-kernel-latency.md#verify-row-fusions)).

The bundle no longer reads execution settings from
`CHOPINFER_DEEPSEEK41_SHARED_OVERLAP`,
`CHOPINFER_DEEPSEEK41_PREFILL_PLAN_CACHE_PLANS`,
`CHOPINFER_DEEPSEEK41_PREFILL_PLAN_CACHE` or
`CHOPINFER_DSPARK_GRAPH_HANDLES_PER_RANK`. Use the corresponding row-limit,
plan-count, plan-verification and graph-budget options above. Startup emits
one warning naming each set retired DeepSeek or DSpark switch as ignored,
including opt-outs removed during adoption. An old value such as
`CHOPINFER_DEEPSEEK41_SHARED_OVERLAP=0` cannot select serial execution.

Per-step JSON is off by default. Enable `diagnostic_step_logs=true`, and serve
with `--log-level debug`, for timing and confidence captures; this changes
diagnostic output only.
`CHOPINFER_STEP_TRACE` still controls the shared host's tracing, but no
longer selects the bundle's step or commit records. Existing diagnostic
capture directories, AR Engram tracing and the shared
`CHOPINFER_DEVICE_RESULTS` control remain available.

### Removed prompt-end grid-anchor candidate

The prompt-end grid-anchor candidate is removed. A TP4 AR C1 replay on
2026-10-29 matched all 40 turn pairs bitwise but regressed throughput by
0.66% and TTFT p95 in both pairs. An off-grid 789-row prefill took
311-313 ms, while separate 768- and 21-row steps totaled 288-294 ms.
The tail selected split-tail dense routes instead of exact-row folds.
The final DSpark C1 gate also regressed TTFT p95 in both pairs: 303.03
to 324.71 ms and 308.08 to 321.54 ms. The declared gate failed despite
the cold AR C4 throughput gain.
A future proposal needs served stateful acceptance across cold and warm
turns before adding a new grid-anchor ABI or changing the default schedule.

### Removed short-prefill graph candidate

The short-prefill CUDA graph candidate was removed after the 2026-10-28
TP4 C1 sampled continuation replay. Graphs increased median total prefill
time per continuation by 7.04% in AR and 7.08% in DSpark with five fixed
draft tokens, with exact output IDs and served logprob bits. The hot 128-row
program captured once and replayed 30 times per visit, yet those replays were
4.7-5.6 ms slower per step than the matched eager route. Each measured
visit also spent 470-584 ms capturing and instantiating 11-13 programs;
rare tail shapes often did not amortize that cost. The eager route remains.

A future attempt must measure fingerprinting and eager-gap costs separately,
avoid captures that cannot amortize, and demonstrate a net served gain on
both AR and DSpark. Parameter updates or larger graph segments must preserve
exact resolved arguments, reduction order, live-row writes and the index
selection and Engram gather dependencies. Higher cache capacity alone does
not address this result: no eviction or capacity refusal occurred.
The removal's [identity declaration](journal-changes/remove-short-prefill-graphs.json)
allows no changes to pinned journals, instance creation or device history.

## DSpark execution

`tp4-dspark` uses balanced planning, 64 speculative rows, widths
`6,8,16,32,64`, graph rows 64 and depths `1:5,8:3,16:3,32:0`. It adopts the
stored floor-one confidence calibration for both sampled and greedy
requests. The scheduler still selects the verify width; the device allocator
assigns draft prefixes within it. Only target-verified tokens are committed.

Confidence allocation scores and gathers only computed vocabulary rows.
Sampling uses the computed-to-reserved index map, preserving each item's
seed, generated index, Gumbel ordinal, stop rules and bonus row. Commit
retains the reserved feature layout. Allocation metadata uses persistent
staging and cached calibration bytes.

On 2026-10-28, TP4 native-KV ABBA comparisons at temperature 1.0, with
confidence enabled and power limits of 275/250/250/250 W, covered coding
and chat at 256 output tokens with four and eight streams, and at 1024
output tokens with four streams. Compact allocation improved coding
four-stream 256-token request throughput by a resolved 1.91% (1.57
percentage-point paired spread). The other five throughput cells were
within noise, and no cell had a resolved step-time, acceptance or
throughput regression. All 311 matched request keys and seven unmatched
replays agreed bitwise on emitted tokens and selected logprobs; the fixed
batch comparison also checked logits, proposals, acceptance and state.
Compact allocation is the default implementation. The 2026-10-28 adopted
fixed-batch GPU smoke passed at 12, 32 and 65 computed rows, with greedy
and fixed-seed logits, proposals, per-request acceptance and committed
state equal to the previous implementation. GPU exactness evidence ends
at the tested width of 65; widths 66-192 have CPU coverage only. Widths
above 64 remain unqualified for timing.

On 2026-11-02, TP4 native KV with 196608 pages per rank and the default
`tp4-dspark` profile (64 speculative rows and floor-one confidence) matched
pre-fix build `21da932dd8` against fix tip `171003328d` and serving build
`0c620f1a66` bit for bit. Each later build matched 96 requests and 12,288
emitted token IDs and selected logprobs: two repeats of greedy and sampled
requests at C8 and C16, with 128 outputs, sampled temperature 1.0 and seed
9100 (16 comparisons per kind at C8, 32 at C16). Every measured window in
all three builds executed allocated steps, including `i8-r32-v8-R48` and
`i16-r32-v15-R61`. Normal-loaded and keeper-borrowed base outputs also
matched. The serving build completed a 145-sequence mixed smoke without
errors or a restart. Its 30 history-dependent requests, six each with
presence penalty 0.5, frequency penalty 0.5, repetition penalty 1.2, strict
JSON Schema or required tool calls, verified through unallocated steps;
ordinary control cohorts in the same server still allocated at C8 and C16.

See [confidence allocation](dspark-confidence.md) for the measured coding
and chat tradeoff.

Each drafting item has five backbone rows. Allocated batches compute all
five heads. Unallocated uniform depth-three batches of 3-8 items compute
only the used heads; other batches compute all five.
Sampled drafts use the target sampler's Philox noise at the draft position's
generated index. The twelve-item forward limit splits larger batches into
near-equal chunks, for example 16 as 8+8. Target outputs do not depend on
this split. Wider admitted verify steps run eagerly above the graph budget.
The route sweep `tp4-dspark-rows192` serves the profile at the package's
192 verify rows (`--speculative-rows 192`), so its envelope adds every eager
verify width from 65 to 192 rows.

With the `wide_verify` candidate option, eager verify steps of 65 to 192
rows prepare the MoE input, dense input and latent window-write ceil
quantizers, the shared SwiGLU, the latent query rotation and the index
query at the wide verify buckets (`verify_bounds::WIDE`: 128 rows and the
package's 192, each capped by the verify rows), so they launch the decode
entries a 64-row step uses. Their HC boundaries fuse as a 64-row step's
do: 77 of each rank's 80 defer the all-reduce round into the column-block
producer and split finish (`sm120::hc_wide`). The other three keep the
one-CTA mixer, whose split form would need three times the mixer partials.
The option also folds the dense projections of those steps (query down and
up, window KV, output A and B, shared gate, up and down, and the indexer
query) through wide folds instead of the 8192-row fold: the dense
decode template with a 256-row workspace header (`dense_decode::Entry::wide`),
one per wide verify bucket, with the decode entries' fold order and
arithmetic and the schedule of its `dense_fold.wide.verify` row. Engram's projection takes its qualified matrix tile at the
step's rows, as from 33 to 64 rows. The routed experts of those steps run
the direct MoE queues compiled at the package's 192 rows
(`bundle/src/sm120/moe_direct_wide.rs`) instead of the 8192-row expert
chains; kernel-lab pairs matched the chains bitwise at every row count from
65 to 192 and were faster in every timed case
([qualification](lab/moe_direct_wide.md)). Without the option those stages
keep their prefill entries, the 8192-row folds and the expert chains, and
the boundaries stay unfused. The route sweep `tp4-dspark-rows192-wide`
serves the override with the option on. The query norm, the latent window
rotation and the remaining row norms and rounding rows launch the 64-row
entries either way, and the other stages keep their prefill entries. The
latent selected cache needs kernel, tuning or shared-limit changes to serve
more than 64 rows, and without the option the chained MoE's intermediate
quantizer is selected inside its 64-row chain. On 2026-10-31, fixed-batch GPU replays of the option's stages at
verify widths 64 to 192 (concurrency 16 and 32, four to six rows per item,
verify capacities 80 to 192) matched the previous package bitwise in
logits, proposals, acceptance and committed state. Under a CUDA trace, the
quantizer stages of a 128-row step fell from 12.6 to 1.2 ms of GPU-busy
time per device. The projections after them had overlapped those stages
and now expose most of that time, so the step fell only from 196.3 to 193.9
ms, and a 96-row step from 175.6 to 169.6 ms. With the wide dense folds,
replays of the same five configurations again matched the previous package
bitwise, and the dense folds of a 128-row step fell from 93.1 to 12.3 ms of
GPU-busy time per device (the step from 196.3 to 103.0 ms; a 96-row step
to 89.8 ms). A 192-row step (concurrency 32, six rows per item) fell from
234.9 to 132.2 ms, its dense folds from 109.6 to 17.8 ms, with the ticket
schedule. Measured on the quantizer change without
the dense folds, the HC fusion and the queries took a 128-row step from
193.8 to 182.1 ms and a 192-row step from 225.0 to 214.6 ms. Their HC
boundaries fell from 12.9 to 2.8 ms and from 13.6 to 3.9 ms, and their
index queries from 0.7 to under 0.1 ms. Replays at the same five
configurations matched bitwise. The lab pairs cover the wide entries at 65,
128, 129, 192 and 256 rows: the index and latent query pairs on their
fixtures, and the column-block pair against the ordered-chain pair on
synthesized adversarial inputs (`hc_cols::synthetic_wide`), structural
coverage that complements those real-data replays. On 2026-10-31 all three
matched bitwise on GPU, with every mutation control detected, and
compute-sanitizer found no error in the query pairs. With both changes
merged, the five
replays matched bitwise again, and a 128-row step takes 92.2 ms and a
192-row step 121.9 ms of GPU-busy time per device, against 196.3 and 234.9
ms without the option; the MoE expert chains take 41.0 and 49.9 ms of them.
With the wide fold rows from the lab sweep (weight prefetch in 14 of the 16
keys), the five replays matched bitwise again, and the wide folds of a
128-row step fell from 11.9 to 10.9 ms (the step from 92.2 to 90.8 ms) and
those of a 192-row step from 17.3 to 16.2 ms (121.9 to 120.9 ms). With the
wide direct MoE merged, the five replays matched bitwise again, eagerly and
with graphs up to each verify capacity, and a 128-row step takes 67.4 ms and a
192-row step 97.7 ms; their MoE router, experts and sum take 24.1 and 34.9
ms, against 47.8 and 57.5 ms with the expert chains. The selected cache would save about 0.6 to
0.8 ms of such a step, assuming its cost grows with the rows. At 64 rows it
takes 2.1 ms, against 3.4 ms for the direct sources of an 80-row step, 5.0
ms at 128 rows and 6.9 ms at 192. Its 192-row arena would add 84 MB per
rank. This is not served step-time acceptance.

With the option, the same steps run the vocabulary head, the MoE router and
the compressor projections in the measured whole-K exact F32 row tiles
(`sm120::f32_verify_rows`, tuning site `f32.rows.verify`) instead of the
ordered split route. A CTA owns a tile of outputs over every 128-K chain,
or over a K slice whose chain values an ordered reduction adds, and keeps
the split route's components, chains and order, so every output keeps its
bits. The index weights and index keys keep the split route, which the lab
measured as fast. On 2026-10-31, the kernel lab's verify-row timing (one
GPU at its 250 W enforced limit, in-job A/A controls within 2%) measured
the head at 0.59, 0.62, 0.81 and 0.99 ms at 96, 128, 160 and 192 rows
against 3.44, 4.69, 6.00 and 6.85 ms on the split route (a cold read of its
weights takes 0.23 ms), the router at 30 to 32 us against 53 to 93 us, and
each compressor projection at 30 to 43 us against 68 to 120 us. The lab's
comparisons matched the split route bit for bit at every tile boundary
from 1 to 192 rows and on every operand witness, for the lab's tiles and
the packaged ones. Fixed-batch GPU replays with the option at the five
configurations above matched the previous package bitwise in logits,
proposals, acceptance and committed state. This is not served step-time
acceptance.

Flash TP4 verify steps of 33 to 64 rows run the same tiles, a range the
decode route served through the cached separate spans:
the head's tile at every width and the compressor's eight K slices of 64
by 32. The router keeps its spans, which the tiles only matched there, and
which the direct MoE's fused router consumes. In the same lab run the head
took 0.40 and 0.41 ms at 48 and 64 rows against 1.08 and 1.37 ms on the
spans, and each compressor projection 28 us against 33 and 38 us.
Fixed-batch GPU replays with `wide_verify` and the `f32_row_tiles`
candidate option on, at verify widths 32 to 64
(concurrency 16 and 32, two to four rows per item, against captures of the
previous package) and at the five wide configurations, matched bitwise.

The head also runs at every decode and verify width of 32 rows or fewer
in its decode tiles (tuning site `f32.rows.decode`, a module of its
own so the verify module keeps its bytes): 96 features by 8 rows through 8
rows, 96 by 16 through 16 and 256 by 32 through 32, each a wave of CTAs that
streams its features over every chain. The DSpark drafter's head runs the
head's tiles at its own rows, the decode tiles through 32 rows and the
verify tile above to the instance's narrow verify rows, in place of its
bounded reuse, so its proposals keep their bits. The index weights at 33 to
64 verify rows keep the split route instead of the cached spans. On
2026-10-31, the kernel lab (one GPU at its 250 W enforced limit, in-job A/A
controls within 2.5%) measured the head at 240 to 244 us from 2 to 8 rows
against 262 to 277 us on the served route, 256 to 258 us from 9 to 16 rows
against 346 to 399 us and 262 to 266 us from 17 to 32 rows against 492 to
688 us. At one row it took 240 us against 246 to 250 us in three runs, a
gain near that row's A/A spread. The drafter's head took 0.40 to 0.43 ms at
35 to 60 rows against 0.89 to 1.35 ms on the bounded reuse, and the index
weights 14.6 to 14.9 us at 33 to 64 rows against 18.3 to 18.4 us on the
spans. The lab's comparisons matched the split route bit for bit at every
tile boundary and at the decode, small verify and drafter head widths from 1
to 64 rows, on every operand witness, for the lab's tiles and the packaged
ones. Fixed-batch GPU replays with the candidate option on at verify
widths 4 to 64 (concurrency 1, 16 and 32, two to six rows per item, against
captures of the previous package) matched bitwise in logits, proposals,
acceptance and committed state.

These routes served behind the `f32_row_tiles` candidate option until its
adoption on 2026-11-01, which removed the option; an instance that names it
is refused, and the results here that name it ran with it on. Served
exactness with the option alone beside the `tp4-dspark` profile gave
identical tokens at C1, C8, C16 and C32. The served step-time A/B of the
option against the profile, every visit valid, resolved a gain in every
cell: C1 (cell i1-r6) 0.33% with a spread of 0.08%, C8 (i8-r32) 3.78%
(0.96%), C16 (i16-r64) 7.00% (0.67%) and C32 (i32-r32) 1.53% (0.54%, its
lead visit discarded; the same design's A/A was clear). No cell regressed.

The lab timings above ran the General witness, whose weights all lie in the
tensor domain, which admits a nonzero operand only at 2^-30 <= |x| < 2^31.
The checkpoint holds five BF16 weights of these projections below 2^-30:
head row 49423 (tile 1068 of rank 1's slice), one in each router of layers
19, 22 and 35 and one in layer 8's compressor `wkv`, whose tiles every rank
holds; the index projections hold none, and the drafter's vocabulary head
reads the same `head.weight`. Such a 16-feature tile keeps the SIMT chains
at every row, which the row tiles first ran serially in 16 threads: on
2026-10-31 the kernel lab, run on each rank's real slices
(`CHOPINFER_F32_ROWS_SLICE`, one GPU per rank at its 250 W enforced limit,
in-job A/A controls within 2.7% and 4.5% at 65 rows),
measured rank 1's head tile at 14.3 to 14.7 ms at every width from 33 to
192 rows and its decode tiles at up to 14.1 ms, and the affected router
and compressor tiles at 0.51 to 0.63 ms. The fallback now stages the tile's
weight and activation lines through the CTA's ring and shares its chains
among every thread, so rank 1's head took 0.40 to 0.44 ms at 33 to 64 rows,
0.59 to 0.63 ms at 65 to 128 and 1.01 ms at 192, within 2% of the other
ranks, and its decode tiles 252 to 297 us at 1 to 32 rows against 240 to
266 us on the other ranks and 250 to 672 us on the served route. Router
layer 19 took 41 to 44 us against 30 to 34 us for layer 0, and compressor
layer 8 36 to 53 us against 28 to 43 us for layer 2. The lab's comparisons
matched the split route bit for bit on every rank's head slice, the four
routers, both compressors and layer 8's index projections, one of them with
a weight set below 2^-30, at every boundary row from 1 to 192, and on the
default operand matrix. Later that day the lab's `real::SLICES` matrix, read
from the checkpoint, found exactly these five tiles and matched bit for bit
on all 12 slices at all 56 boundary rows, and the default matrix gained a
domain-boundary witness: signed operands just below, at and just above
2^-30 and 2^31 at five K positions put six of every 13 rows and tiles
outside the domain, and on all five projections at every boundary row the
reference matched the outputs those operands fix, a cancelling chain whose
bits show each output's path, and every form matched the reference.

Merged with the wide direct MoE, the five wide replays matched bitwise
again, eagerly and with graphs up to each verify capacity, and so did the
four at verify widths 16 to 64 with both options on. A 128-row step takes
76.7 ms and a 192-row step 104.1 ms, against 67.4 and 97.7 ms before the
merge. On rank 1 the head's tile takes 14.3 ms at every wide width,
against 0.6 to 1.0 ms on the other ranks, and every rank waits for it in
the logits exchange: one weight of rank 1's vocabulary slice (head row
49423, about -5.9e-10) lies outside the exact-F32 tensor domain, and the
CTA that owns its 16-row group computes that group's outputs in the serial
SIMT chains. The other three ranks' slices have no such weight. With the
phased decode reduction merged, the wide replays matched bitwise again,
eagerly and with graphs, and so did the 16-64 row replays with
`f32_row_tiles` and the 16-192 row replays with `phased_decode_reduce`,
each beside `wide_verify`; the 128- and 192-row steps kept their times.
With the direct MoE cache knobs merged, the wide replays matched bitwise
against references recorded again from the previous package by the
merged host, whose frontend the earlier host cannot read, and so did the
same replays with the robust knob set on, eagerly and at 16-64 rows. The
run meant to replay them with graphs left the graph rows at their default
of 64, so it repeated the eager replays. The knob set took the MoE router, experts and sum of a
128-row step from 24.2 to 19.8 ms and of a 192-row step from 35.0 to 29.3
ms, and the steps from 76.9 to 72.2 ms and from 104.0 to 98.3 ms. This is
not served step-time acceptance.

The `wide_draft` candidate option, with `wide_verify`, runs a drafter step
of more than twelve drafting items in the fewest forwards the widest wide
verify bucket holds, five block rows an item, instead of chunks of at most
twelve: one forward of 16 items at C16 instead of 8+8, and one of 32 at
C32 and 192 verify rows instead of 11+11+10. A forward above 64 block rows
folds its dense projections through the wide folds and projects the
vocabulary in the head's exact F32 row tiles. Every drafter op keeps
one-row bits, and attention, the Markov head and the sampling noise are
per item, so the proposals do not depend on the split. On 2026-10-31,
fixed-batch GPU replays at the five wide configurations matched the
previous package bitwise. Under a CUDA trace of a live sampled load at 192
verify rows, merged with the exact F32 projection work's fix of a rank-1
head tile stall, the drafter's GPU time fell from 7.61 to 6.91 ms per C16
step and from 14.14 to 9.47 ms per C32 step, and the step period from 82.8
to 81.6 ms and from 136.7 to 132.3 ms. Without that fix the single forward
stalls on rank 1's head tile and is slower than the chunks. See
[wide drafter forwards](dspark-wide-draft.md). This is not served
step-time acceptance.

Merged into the wide verify branch with the exact F32 fallback fix, the
five wide replays matched bitwise again against the h2 references. A
128-row step takes 60.9 ms and a 192-row step 88.3 ms of GPU-busy time per
device in the target's eager verify kernels (the drafter's graph launches
are outside this measure, as in the times above), and 56.5 and 82.5 ms with
the robust knob set, against 76.9 and 104.0 ms with the rank-1 stall. The
head's row tile takes 0.6 ms on every rank at 128 rows and 1.0 ms at 192,
and the logits row exchange of a 128-row step fell from 10.9 to 0.3 ms.
With the wide drafter merged, the five wide replays matched bitwise in
logits, proposals and committed state with `wide_verify` alone and with
`wide_draft` too (whose drafter ran fewer graph launches), eagerly and,
with `wide_draft`, with graphs up to each verify capacity; so did the four
at 16-64 rows with `f32_row_tiles` and `wide_draft`, and the five graph
replays with the knob set, whose graph rows this time reached each
capacity. Allocated replays of one drafted step after the prefills, at C16
(seeded), C24 (greedy) and C32 (seeded and greedy), with five drafts an
item and 64, 96 and 128 computed of 96, 144 and 192 reserved rows, matched
captures of the previous package with `wide_verify` alone and with
`wide_draft`, whose drafter then runs each step's 16, 24 or 32 items in one
forward: the logits of every reserved row, the proposals, the device
acceptance and all 80, 120 and 160 raw F32 drafter confidence words of the
step's items (the replay tool's `draft_confidence` mode). This is not
served step-time acceptance.

Served acceptance on 2026-11-01 (four GPUs at their 250 W limit, the
fixed 65536-page pool, sampled coding prompts at temperature 1.0 with 256
output tokens) measured `wide_verify` with `wide_draft` at 192 verify and
graph rows, and neither is adopted on this host. The depth sweeps chose
8:3, 16:2 and 32:0, so no chosen depth verifies more than 64 rows. At C16
the 64-row step (54.8 ms, 47.1 tokens) gave 1.199 times plain decode's
tok/s, and the 80- and 96-row steps (66.8 and 75.2 ms) 1.116 and 1.058
times. At C32 the 128-row step (95.0 ms, 92.3 tokens) gave 1.016 times
plain decode's full-concurrency tok/s at 1.044 times its J/token and 0.973
times its request tok/s; the 160- and 192-row steps (110.1 and 124.4 ms)
fell below plain decode. Without the option the same 80- to 192-row steps
took 191.9 to 286.7 ms, 2.3 to 2.9 times the wide route. The software power
cap held the SM clocks in 76 to 97% of samples in every arm.

The GPUs now enforce their default 300 W limit. On 2026-11-02 the decode
bench and the depth sweeps ran again at 300 W on main 0c620f1a66. The 250
W bench used 5e864484fe and the 250 W sweeps cd1eab92ef, so the comparison
across limits is indicative only; each choice below compares runs of one
build at 300 W. The default profile's decode bench (medians of three
trials of 2048 output tokens) gave 281, 976, 1,140 and 1,269 tok/s at one,
eight, 16 and 32 streams, against 282, 894, 1,030 and 1,151 at 250 W: flat
at one stream and 9.1 to 10.8% higher from eight. Depth sweeps at the
profile's 64 verify rows (same workload and pool as above) chose depth 3
at C8 (1.198 times plain decode's request tok/s), depth 3 at C16 (1.172)
and depth 0 at C32 (depth 1 gave 0.978), the profile's 8:3, 16:3 and 32:0.
With `wide_verify` and `wide_draft` at 192 verify and graph rows, the 80-
and 96-row steps at C16 gave 1.033 and 1.006 times plain decode's request
tok/s and the 96- to 192-row steps at C32 0.863 to 0.927 (on the same
request basis the 250 W sweeps gave 1.023 and 0.986 at C16 and 0.897 to
0.975 at C32). The best wide depth at C32, one draft at 64 rows, gave
1.002, under the 1.5% selection margin. In the sweeps each GPU's mean SM
clock per arm was 2,120 to 2,260 MHz, with the power cap active in 0 to
87% of its samples, against 1,870 to 2,220 MHz and 76 to 96% in the 250 W
sweeps. No setting cleared the margin over the current entry, so the depth
table, rows, graph rows and widths stay and both options stay off. Every
sweep window matched its reference bitwise, with no major faults,
preemptions or prefix-cache evictions.

The drafter cost candidates merged into this branch with every existing
journal identical ([merge note](journal-changes/wide-verify-merge-drafter.md)).
At the merged tip every profile's quick route sweep passed, and the drafter
profile's at the full tier.

Prefill commit projections are batched across prompts. Undrafted pure
decode joins status once and submits private sampling before collecting it
on the native result path. Verify and undrafted commits with at least eight
retention copies per rank use the indexed byte-copy kernel; smaller commits,
prefill and mixed steps keep their ordered copy list. Every gather completes
before success, and failed device work poisons the instance.

The 2026-10-24 TP4 prefill comparison at 300 W matched full logits, target
KV, all 134 history rows, drafter state and sampled tokens. At temperature
1 with 512 outputs, median C8/C16 throughput rose from 465.93/544.39 to
472.37/553.35 tokens/s. Batched commit costs 121.91 MiB per rank, about
0.734% of the measured KV pool; gather IDs reserve another 1.5 MiB on rank 0.
The 64-row default comparison on 2026-10-25, with 1024 sampled output tokens
and a fixed 32,768-page pool, resolved a 6.357% C16 request-throughput gain
(0.273-point pair spread) and a 5.237% energy reduction; C8 was within noise.
These bounded results precede the confidence adoption and are not a final
refactor-versus-baseline verdict.

The [execution lifecycle](bundle/src/dspark/README.md) and
[lowering contract](bundle/src/lower/dspark/README.md) describe catch-up,
raw-history retention, graph reuse, cancellation and prefix restore.
For untimed transition diagnostics, `state-capture` and
`CHOPINFER_DSPARK_STATE_DIR` record selected before-draft/after-commit
boundaries. `diagnostic_step_logs` defaults to false and controls step and
confidence JSON, without changing confidence computation.

## Reference

`reference.py` is the refrun adapter (`tools/refrun`). It executes the
checkpoint's own `inference/` code (sha256-pinned) with the tilelang kernels
replaced by `reference_kernel.py`, whose ledger records each replacement and
its result against the executed vendor kernel
(`reference_vendor_control.py`). `REPLACEMENTS` lists everything else it
changes: the row-blocked prefill of `reference_memory.py`, ties to the lower
index in the indexer and router top-k, the RoPE tables the vendor computes
on CUDA, Engram rows gathered on the host, and the vision tower on its own
device. The F32 mode rounds quantizer inputs to BF16 (F32-TQ, the gate
variant); F64 widens the vendor code except its RoPE angle lines.
`reference_pure_f32.py` is the same adapter without that rounding.
`test_reference.py` checks it bitwise against the vendor's
`convert.py`, `load_model` and `Transformer.forward` on a tiny checkpoint.
Its synthetic tests read only the six small `inference/*.py` files from
`CHOPINFER_DEEPSEEK_TOKENIZER_DIR`, falling back to
`CHOPINFER_DEEPSEEK_CHECKPOINT`. The official checkpoint ships the tokenizer
and inference code together; CPU CI preserves the tokenizer directory while
leaving the official-checkpoint tests opt-in. Each source must match both
`checkpoint-files.json` and the adapter's pin before import. Synthetic tests
generate their own tokenizer, configuration and weights, and never read the
official weights. Missing vendor sources and missing checkpoint inputs have
separate skip reasons; missing Python dependencies remain CI failures.

## Recipe oracle

deepseek-recipe 0.2.0 is the frontend corpus's second authority, beside
the pinned `encoding/encoding.py` ([the recipe oracle](../../tools/refrun/README.md#frontend-corpus-recipe-oracle)).
`frontend_cases.py` pins it as `RECIPE`, and `recipe_reference.py` runs it
only when `frontend_corpus generate` runs, under
`$CHOPINFER_DEEPSEEK_RECIPE_PYTHON`.
`fixtures/deepseek41/frontend/corpus-recipe.json` records the recipe's
outcome for each committed case and round trip, and
[frontend-divergences.json](frontend-divergences.json) records each
difference exactly: a refusal by its entry's feature tags and the recipe's
refusal message, which names the failed check (`recipe_message`), and any
other difference by the fingerprint of both outcomes, reviewed for every
difference of the full generated corpus (`recipe_reviewed`).

The corpus reference marks each render as the recipe does: the recipe's
`DeepseekV41Encoding` renders a `RenderedPrompt` whose `special_ranges`
name its special tokens, and its `PromptTokenizer` encodes each range as its
added token and the text between them with a copy of the tokenizer without
the added vocabulary (`deepseek-recipe-encoding/src/lib.rs` lines 8-15,
`tokenizer.rs` lines 90-117, at the commit below). The pinned encoder
writes one string, so `encoder_render` in `frontend_cases.py` hides each
added-token spelling of client text behind a private sentinel while the
encoder writes, and returns the text with the spellings restored and the
ranges of those the encoder wrote (`Hiding`); `tools/refrun` tokenizes it
(`marked_tokenizer`). Install the recipe
into its own CPU environment, never the GPU runner's, whose package set is
part of every reference key:

```sh
python3.12 -m venv "$CHOPINFER_DATA/refrun/venv-deepseek-recipe"
"$CHOPINFER_DATA/refrun/venv-deepseek-recipe/bin/pip" install -r models/deepseek41/requirements-recipe.txt
export CHOPINFER_DEEPSEEK_RECIPE_PYTHON="$CHOPINFER_DATA/refrun/venv-deepseek-recipe/bin/python"
```

### Provenance

[requirements-recipe.txt](requirements-recipe.txt) pins every 0.2.0 wheel
and the sdist by SHA-256. The corpus pins the PyPI wheel
`deepseek_recipe-0.2.0-cp310-abi3-manylinux_2_28_x86_64.whl` (SHA-256
`259812bf382bc82668b672707c5922095cc4f5f60a6c514ff89640bb62a746f7`):
`generate` refuses an environment whose installed recipe files differ from
that wheel's RECORD (`installed_sha256`
`84ef54ed1b08cd105614efc36d0722c4f8b741f080ca4268bd90d8898b8856b0`, which
`corpus-recipe.json` records), such as another platform's wheel or a source
build. The PyPI sdist `deepseek_recipe-0.2.0.tar.gz` has SHA-256
`5dfd65dc201004f6382b13c8432a406cd2af7188831944ac5cef22f260111f68`.
Compared on 2026-10-30, and again on 2026-10-31, its 83 Rust sources and its
Python sources are identical to commit
`93b0fd229cc9ffa6b1a8c63d42b0bf741bbec170` of
[deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe); six of
its `Cargo.toml` files differ from the commit's, only in packaging metadata
(the workspace members and the readme path). Nothing shows that the
published wheels were built from those sources. The frontend's recipe
citations name lines at that commit.

### Differences the corpus cannot show

The corpus compares renders of Chat Completions requests and parses of the
turns the encoder writes, so these differences have no case and no tagged
entry. The served behavior is kept.

Output parsing (`deepseek-recipe/src/stream/output_parser.rs`):

- Text after the calls block, and a second block: the recipe drops both
  (lines 591-593); the served parser returns the text as content and the
  calls of both blocks.
- `</think>` in chat-mode answer text: the recipe drops it (lines 450-458);
  the served parser keeps it as content.
- The `<think>` token inside reasoning: the served parser drops it; the
  recipe keeps it as text. Spelled in ordinary tokens it is text in both,
  as the corpus shows (`special-tokens-in-final-turn`).
- DSML markup under `tool_choice` none or without offered tools: the recipe
  returns it as content (`chat_completion/request/convert.rs` line 150); the
  served parser reads it and drops it.
- A calls block without its `calls` wrapper, or closed in the V4 spelling
  `</｜DSML｜tool_calls>`: the recipe reads its calls (lines 394-405,
  480-494); the served parser refuses the completion.
- Client stop sequences: the recipe matches them in answer text only (lines
  411-417); the served parser also matches them in reasoning and tool
  arguments.
- `string="false"` values: the recipe writes the value text as it is; the
  served parser reads it as JSON and writes it back Python-style, reads text
  that is not JSON as a string, keeps a marker inside a JSON string as
  content, and keeps the last of a repeated parameter.
- Generation cut by `max_tokens` inside a call: both finish `length`; the
  recipe returns the partial arguments, the served parser the complete
  parameters only.

Responses API (`deepseek-recipe/src/protocol/openai/responses/request`):

- Reasoning input items: the recipe reads their `content` text and ignores
  `encrypted_content` (`convert.rs` lines 492-527); `/v1/responses` reads the
  content, else the summary, and refuses encrypted reasoning.
- Tool types: the recipe accepts a client `tool_search` tool and the custom
  `apply_patch` tool, and drops or refuses `web_search` as its options say
  (`tools.rs` lines 148-186); `/v1/responses` serves function and namespace
  tools only.

Sampling and limits: the owner decided on 2026-10-31 that the server keeps
the request's sampling and declares default output lengths. A chat or
Responses request without `max_tokens`, `max_completion_tokens` or
`max_output_tokens` generates at most 65,536 tokens when it thinks at low
or high effort or with thinking off, and 131,072 when it thinks at max,
each lowered to the context left after the prompt. An explicit limit is
honored. The sampling fields a request sets are used as sent, and omitted
ones take the declared `generation_config` (temperature 1.0, `top_p` 1.0).
The recipe (`deepseek-recipe/src/` at commit 93b0fd229c) differs as
follows, and the served behavior is kept:

- Temperature: the recipe fixes it at 1.0 in thinking mode
  (`protocol/validation.rs` lines 114-118); the server uses the request's.
- `top_p`: the recipe raises V4.1's to at least 0.95 in thinking mode and
  sets it to 1.0 outside it (`request/options.rs` lines 79-92); the server
  uses the request's in either mode.
- Default length outside thinking mode: 8,192 in the recipe
  (`request/mod.rs` lines 155-166 and 221-227), 65,536 in the server.
- Upper bound: the recipe refuses a `max_tokens` above 393,216 in either
  mode (`request/options.rs` line 96 and lines 230-240,
  `protocol/validation.rs` lines 93-106); the server takes any limit within
  the context.
- A prompt that fills the context: the recipe raises the resolved length
  to 1 (`request/mod.rs` lines 228-234); the server refuses the request
  with "prompt leaves no room for generated tokens".
- `max_completion_tokens`: the recipe's `ChatCompletionRequest` declares no
  such field and ignores unknown fields
  (`protocol/openai/chat_completion/request/schema.rs` lines 7 and 17-60),
  so a request that sets only it takes the recipe's default; the server
  gives it precedence over `max_tokens`.
- `top_k`, `min_p` and `repetition_penalty`: that struct declares none of
  them and `InferenceOptions` carries none (`request/mod.rs` lines 27-55),
  so the recipe drops them; the server applies them.
- `presence_penalty` and `frequency_penalty`: the recipe checks them
  against [-2, 2] and then drops them (`schema.rs` lines 23-25 and 34-36,
  `protocol/validation.rs` lines 53-62 and 112-124); the server applies
  them.
- `n`: the recipe refuses any value but 1 (`protocol/validation.rs` lines
  48-52); the server serves 1 through 128.

## Qualification

`model.toml` still disables release and both profiles remain experimental.
The refactor's CPU journals, kernel twins and bounded served comparisons do
not replace the release preview's full suite at the release revision.
`qual.json`, `validation.toml` and the pinned reference locks are authoritative
for the checks and their inputs.

The reference is the checkpoint's pinned vendor code in F32 with
BF16-rounded trained-quantizer inputs (F32-TQ), with BF16 and F64-TQ spread
variants. Teacher coverage is at least 3974 positions and chat at least
5270; pooled mean absolute logprob limits are 0.150 and 0.300 nats.
Category limits follow the same frozen spread policy. Paired-teacher uses
the reviewed 0.005-nat pooled and 0.01-nat category mean margins, plus the
shared top-one and tail gates. Limits and references never move to hide a
mismatch.

The suite includes tokenizer/frontend goldens, per-layer taps, long context,
GSM8K, media, prefix equality, determinism, native-KV offload and a BF16-KV
comparison. Native and BF16 KV must give the same served bits; BF16 KV is a
diagnostic comparison and is not qualified for capacity, pressure
preemption or offload. It explicitly disables speculation.

The [numerical gates](../../docs/numerics.md#serving-qualification-gates)
and [serving profiles](../../docs/serving-profiles.md) describe the shared
acceptance rules. Store captures, qualification receipts and timings outside
the repository under `CHOPINFER_DATA`.

### DSpark qualification

`tp4-dspark` requires AR and speculative arms, exact greedy equality, the
declared verify bounds and GSM8K at least 193/200 in each arm. Forced verify
capacities are judged separately against paired AR, without pooling away an
uncovered capacity.

TP4 DSpark main projections use the existing stage-2 compact MXFP8 template
for retained inner widths 129..511. This range is owned by the dense-prefill
tuning table; smaller and larger widths retain the previous routes. The
[qualification record](lab/compact-row-ranges.md) states the bitwise,
served-boundary and retirement-state evidence and the served timing gate.
