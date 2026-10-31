# DeepSeek V4.1 Flash (`deepseek41`)

DeepSeek V4.1 Flash, from the official checkpoint
`deepseek-ai/DeepSeek-V4.1-Flash-1017` at revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`, served as bundle
`deepseek41.v1` at topology `deepseek41-flash-tp4`. It is not released:
`model.toml` declares `[release] enabled = false` until it qualifies.

Supports images in tool results through Chat Completions and Responses, in the official encoder's prompt order.
Images may be PNG, JPEG, GIF (the first frame) or WebP (lossy, lossless,
with alpha, or the first frame of an animation), decoded as the reference's
Pillow `Image.open` and `convert("RGB")` read them, and may arrive as
`image_url` parts or OpenAI `file` parts whose `file_data` is an image data
URL (the deepseek-recipe's mapping in OpenAI's nested shape; its flat shape
is refused). See [the request surface](../../docs/serving-request-surface.md#media-inputs).

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

`tp4-dspark` serves `--prefill-budget tbt` at a 750 ms decode cadence
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
| `dspark_coupled_drafts` | `true` | Couple sampled drafts to the target sampler's noise. |
| `dspark_draft_chunk_items` | `12` | Drafting items per forward, from 1 to 32. |
| `dspark_graph_handles_per_rank` | `512` | Graph handles per rank, from 0 to 9223372036854775807; 0 keeps DSpark eager. The total across ranks must fit the platform's count range. |
| `phased_decode_reduce` | `false` | Candidate: Flash TP4 decode and verify all-reduces whose round no HC boundary takes launch the phased F32 rounding entries, over comm state of their own, instead of the F32 reduce-scatter, all-gather and round row. They give the same bits; the deferred pairs and prefill are unchanged. Served step-time acceptance remains required before adoption. |
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
| `wide_verify` | `false` | Candidate: Flash TP4 eager verify steps of 65 to 192 rows prepare their row-local stages and HC boundaries at the wide verify buckets (128 rows and the verify rows) and launch the decode entries a 64-row step uses. Fixed-batch GPU replays matched the default bitwise; served step-time acceptance remains required before adoption. |

TP4 native-KV single-item DMA prefill overlaps both prepared Engram gathers
by default at 512-8192 rows. The model joins each layer before its decoder;
media, taps and unavailable prepared row services retain their existing
fallbacks. See [mapped-row read-ahead](../../docs/mapped-row-read-ahead.md)
for the adoption evidence and remaining synchronous callers.

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
Without the option those stages keep their prefill entries and the
boundaries stay unfused. The route sweep `tp4-dspark-rows192-wide` serves
the override with the option on. The query norm, the latent window
rotation and the remaining row norms and rounding rows launch the 64-row
entries either way, and the other stages keep their prefill entries. The
latent selected cache, the cached F32 projections, the dense and Engram
folds and the MoE need kernel, tuning or shared-limit changes to serve more
than 64 rows, and
the chained MoE's intermediate quantizer is selected inside its 64-row
chain. On 2026-10-31, fixed-batch GPU replays of the option's stages at
verify widths 64 to 192 (concurrency 16 and 32, four to six rows per item,
verify capacities 80 to 192) matched the previous package bitwise in
logits, proposals, acceptance and committed state. Under a CUDA trace, the
quantizer stages of a 128-row step fell from 12.6 to 1.2 ms of GPU-busy
time per device. The projections after them had overlapped those stages
and now expose most of that time, so the step fell only from 196.3 to 193.9
ms, and a 96-row step from 175.6 to 169.6 ms. With the HC fusion and the
queries, a 128-row step fell from 193.8 to 182.1 ms and a 192-row step from
225.0 to 214.6 ms. Their HC boundaries fell from 12.9 to 2.8 ms and from
13.6 to 3.9 ms, and their index queries from 0.7 to under 0.1 ms. Replays
at the same five configurations matched bitwise. The lab pairs cover the
wide entries at 65, 128, 129, 192 and 256 rows: the index and latent query
pairs on their fixtures, and the column-block pair against the
ordered-chain pair on synthesized adversarial inputs
(`hc_cols::synthetic_wide`), structural coverage that complements those
real-data replays. On 2026-10-31 all three matched bitwise on GPU, with
every mutation control detected, and compute-sanitizer found no error in
the query pairs. The selected cache would
save about 0.6 to 0.8 ms of such a step, assuming its cost grows with the
rows. At 64 rows it takes 2.1 ms, against 3.4 ms for the direct sources of
an 80-row step, 5.0 ms at 128 rows and 6.9 ms at 192. Its 192-row arena
would add 84 MB per rank. This is not served step-time acceptance.

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
