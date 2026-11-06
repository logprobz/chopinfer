# DeepSeek V4.1 Flash (`deepseek41`)

DeepSeek V4.1 Flash, from the official checkpoint
`deepseek-ai/DeepSeek-V4.1-Flash-1017` at revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`, served as bundle
`deepseek41.v1` at topology `deepseek41-flash-tp4`. It is released:
`tp4-dspark` is qualified through 32 sequences by the release preview's full
suite at d3d580aa02 ([qualification](#qualification)), and `tp4-ar` is the
experimental baseline.

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

`profiles.json` ships two profiles, `tp4-ar` and `tp4-dspark`.
`tp4-dspark` is the serving profile; `tp4-ar` is its baseline, correctness
control and debugging aid. `tp4-dspark`'s `depths` end at `32:0`, so it drafts
nothing from 32 decode items. Both use native KV, prefix caching, at most 32
sequences, 8192-token prefill chunks and exact upper-layer dependency cones.
AR reserves 320 MiB per rank for graphs; DSpark reserves 640 MiB. Package
capacity and CPU checks do not establish release qualification.

The scheduler admits 64-token aligned shared-prefix anchors in addition to
the chunk grid and prompt ends. An unaligned prompt publishes an anchor at
`prompt_tokens / 64 * 64`. On these profiles the step that reaches it
continues to the prompt end and captures the anchor inside the step, for up
to eight requests per step. A chunk through the prompt end that does not fit
the step's budget stops at the anchor, and the remainder runs as a separate
prefill step. A step that also prefills an image or a scored prompt keeps no
anchors and runs the separate tail step. Each endpoint produces complete
target state and, for DSpark, the 134-row raw feature history, including six
preserved older rows. An anchor costs 53 pages (2.88 MiB) per rank for AR or
125 pages (6.79 MiB) for DSpark. `prefill_upper_skip` and
`prefill_chunk_tokens` remain declared options for diagnostic
configurations.

`tp4-dspark` serves `--prefill-policy adaptive` at a 750 ms decode cadence
target with adaptive timing cuts; `tp4-ar` keeps the fixed budget and whole
grid chunks. See the [prefill policy](../../docs/developer-preview.md#prefill-policy).

AR decode, eligible AR prefill, TP4 native-KV single-item DMA prefill at
512-8192 rows and DSpark target execution use prepared per-layer Engram
gathers. Their shared layer-gather lifecycle enqueues mapped reads before
the consuming layer and joins them before its decoder. Upper-skip DMA
prefill of other shapes retains the synchronous gather, as do media, taps
and hosts without the needed service. Every host gather has an
`engram_gather` route tag; remaining fallbacks are recorded in the bundle's
route allowlist.

The Engram codes and scales are about 203 GB of checkpoint tables, mapped
from the file and read through the host page cache. The weight plan keeps
`prewarm: false`; evicted tables cause major faults and multi-millisecond
gather waits.

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

The host's `CHOPINFER_WEIGHT_CENSUS=0` stores the main-layer expert scales
raw instead of in their lossless encoded banks.

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

## Model options

Set these package-declared options with `--model-option NAME=VALUE`. The
host records every resolved value in serving identity, metadata and
metrics. The table shows the Flash package defaults; profiles can override
them. A non-default value is unqualified.

| Option | Default | Meaning |
|---|---|---|
| `diagnostic_step_logs` | `false` | Emit DSpark phase, prefetch, Engram, commit and drafter-confidence JSON records. |
| `dspark_confidence_staged` | `false` | Candidate: drafter confidence heads load each thread's 128-weight chain into registers before the dependency wait and stage each row's operands in shared memory, keeping every chain, fold and score. Bitwise equal to the default; not adopted. |
| `dspark_coupled_drafts` | `true` | Couple sampled drafts to the target sampler's noise. |
| `dspark_draft_chunk_items` | `12` | Drafting items per forward, from 1 to 32. |
| `dspark_early_prefill_plan` | `true` | Candidate: a DSpark step plans its prefill (its upper-skip lowering or template and its Engram plan) once the drafter's launch returned, while the drafter runs, instead of after the draft readback and the verify rows' gather, and frees the plan templates that planning drops to stay within the limit once the prefill launched rather than before it. It applies where the prefill takes the prepared gathers (no encoder rows, no text taps); every launch, upload and result is unchanged. Not yet measured. |
| `dspark_graph_handles_per_rank` | `512` | Graph handles per rank, from 0 to 9223372036854775807; 0 keeps DSpark eager. The total across ranks must fit the platform's count range. |
| `dspark_hc_draft_rows` | `false` | Candidate, with `dspark_hc_fusion`: drafter forwards of one to six items (5 to 30 block rows) admit the HC finish route through the drafter's own table, so their stage boundaries fuse too. Bitwise equal to the default; not adopted. |
| `dspark_hc_fusion` | `false` | Candidate: the drafter's stage boundaries fuse their all-reduce round, expansion, next mixer and input norm where the finish route admits the forward's rows (7 to 12 items), as the backbone's do. Bitwise equal to the default; not adopted. |
| `dspark_markov_staged` | `false` | Candidate: drafter Markov steps of two or more items load each thread's weight chain into registers before the dependency wait and read each position's logits in place, with no row copy. Bitwise equal to the default; not adopted. |
| `dspark_moe_direct` | `false` | Candidate: drafter forwards whose block rows the drafter table admits (`moe.direct_drafter.verify`) run the routed experts through the direct path instead of the expert chains. Bitwise equal to the default; not adopted. |
| `phased_decode_reduce` | `false` | Candidate: Flash TP4 decode and verify all-reduces whose round no HC boundary takes launch the phased F32 rounding entries, over comm state of their own, instead of the F32 reduce-scatter, all-gather and round row. They give the same bits; the deferred pairs and prefill are unchanged. Not adopted. |
| `prefill_chunk_tokens` | `8192` | Guaranteed chunk ends: 0, 2048 or 8192 tokens. |
| `prefill_compact_dense_rows` | `false` | Candidate: Flash TP4 prefill steps whose row counts the candidate table admits (`dense_prefill.compact_range.prefill`) run those projections on the compact prefill entry at the step's exact rows instead of the 128-row tiles and their exact-row tail folds. The table admits `wkv` at 1153 to 3071 and 3073 to 3199 rows, the shared gate/up at 1153 to 2943 and 2945 to 3200, `wq_a` at 1153 to 1279, 1281 to 2368 and 2401 to 3200, and `wo_a` at 1153 to 3200. On 2026-11-06 the lab matched every output and status byte at every count from 1153 to 3200 rows for all four projections. A timing grid over that span (every 32 rows, each multiple of 128 and the count after it, with an A/A control) set the ranges: each grid count where the compact entry was slower at every visit is left out, with the counts before the next grid count. Not adopted. |
| `prefill_dma_producer_intervals` | `false` | Candidate: Flash TP4 DMA prefill reductions of any row count split their MoE producer (the routed sum with its shared expert addend) over the transport's owner intervals, `ceil(rows / 4)` rows clipped at the end, at every layer, instead of only at multiples of 32 rows where the sum writes the call's rows; every row's bytes and the owner sums are unchanged. In the 2026-11-06 live-shape mixed-step profile at 1500, 2300 and 3000 rows no reduction split, and the sum took 106 to 197 us per call, which bounds the overlap at about 3 to 5 ms per pass. Not yet measured on a GPU. |
| `prefill_norm_ceil_rows` | `false` | Candidate: Flash prefill steps of 33 to 8192 rows (`norm.input_range.prefill`) run each sublayer input norm as the fused prefill norm-ceil, as the 22 measured rows of `norm.input_admit.prefill` do, instead of the row norm, the widening of its rows and the serial input quantizer; normed, widened, codes, scales and row status are unchanged. On two GPUs on 2026-11-06 (300 W limit), the fused launch matched the chain bit for bit at every row count from 33 to 8192, and in the pair lab with an A/A control it took 51, 69 and 85 us per launch against 180, 186 and 201 us for the chain at 1500, 2300 and 3000 rows. In the 2026-11-06 live-shape mixed-step profile every serial input quantizer at those rows follows its row norm, and the unfused chain spans 148 to 179 us per site, about 100 sites per rank. Not adopted: no served A/B yet. |
| `prefill_plan_cache_plans` | `16` | Kept short-prompt plans, from 1 to 64; evict the least recently used plan when full. |
| `prefill_plan_cache_verify` | `false` | Lower each cache hit and each reused plan template again and refuse a differing plan. |
| `prefill_streamed_lowering` | `true` | Candidate: an upper-skip prefill that no kept plan or template serves lowers its body while it issues the parts already lowered, with the same launches and results, and keeps its plan once launched. A lowering failure after its first launch poisons the instance. A DSpark step-shape record's `streamed_lowering` says whether its prefill was streamed. Not yet measured. |
| `prefill_upper_skip` | `true` | Use upper-skip prefill; requires nonzero guaranteed chunk ends. |
| `shared_overlap_max_rows` | `4294967295` | Largest graph bucket that overlaps shared-expert and eligible window-KV work. Positive values only; the default covers every admitted bucket. |
| `text_config.candidate_topk_blocks` | `2048` | Candidate block pool, checked with index top-k and the partial newest block. |
| `text_config.index_topk` | `512` | Index selection top-k, from 1 to 512. |
| `text_config.indexer_mma` | `true` | Use the guarded native Flash index scorer where eligible. |
| `text_config.indexer_prefill_tile` | `true` | Enable the qualified Flash tiled indexer routes. |
| `verify_graph_rows` | `64` | Widest graphed verify step; wider admitted steps run eagerly. |
| `wide_draft` | `false` | Candidate, requires `wide_verify`: a Flash TP4 drafter step of more than twelve drafting items runs them in the fewest forwards the widest wide verify bucket holds (25 items at 128 verify rows, 32 at 192), on the wide dense folds and the head's exact F32 row tiles, instead of chunks of at most twelve. Bitwise equal to the default; kept off with `wide_verify`. |
| `wide_verify` | `false` | Candidate: Flash TP4 eager verify steps of 65 to 192 rows prepare their row-local stages and HC boundaries at the wide verify buckets (128 rows and the verify rows) and launch the decode entries a 64-row step uses, fold their dense projections through wide folds at those buckets, run their routed experts through the wide direct MoE queues, and run the vocabulary head, router and compressor in their exact F32 row tiles. Bitwise equal to the default; kept off because served depth sweeps chose no step above 64 rows. |

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

## DSpark execution

`tp4-dspark` uses balanced planning, 64 speculative rows, widths
`6,8,16,32,64`, graph rows 64 and depths `1:5,8:3,16:3,32:0`. It adopts the
stored floor-one confidence calibration for both sampled and greedy
requests. The scheduler still selects the verify width; the device allocator
assigns draft prefixes within it. Only target-verified tokens are committed.

Confidence allocation scores and gathers only computed vocabulary rows.
Sampling uses the computed-to-reserved index map, preserving each item's
seed, generated index, Gumbel ordinal, stop rules and bonus row. Commit
retains the reserved feature layout. GPU exactness evidence for allocation
ends at a width of 65 computed rows; widths 66-192 have CPU coverage only,
and widths above 64 are unqualified for timing.

Each drafting item has five backbone rows. Allocated batches compute all
five heads. Unallocated uniform depth-three batches of 3-8 items compute
only the used heads; other batches compute all five.
Sampled drafts use the target sampler's Philox noise at the draft position's
generated index. The twelve-item forward limit splits larger batches into
near-equal chunks, for example 16 as 8+8. Target outputs do not depend on
this split. Wider admitted verify steps run eagerly above the graph budget.

For untimed transition diagnostics, `state-capture` and
`CHOPINFER_DSPARK_STATE_DIR` record selected before-draft/after-commit
boundaries. `diagnostic_step_logs` defaults to false and controls step and
confidence JSON, without changing confidence computation.

A text-only mixed DSpark step whose prefill takes the prepared-gather DMA
route (one request of 512 to 8192 rows at TP4, beside a graphed verify
phase of at most 64 rows) gathers its verify rows' Engram through the
cycle's prepared layer gathers, begun once the prefill joined its last
Engram layer, instead of synchronously before the step. Every other mixed
step keeps the synchronous gather. Adopted on 2026-11-06 with the other
DSpark prefill steps ([served timing](#served-timing)).

A DSpark step whose prefill takes the prepared gathers (a drafter, no text
taps, no encoder rows) hashes its prefill rows' Engram ids and begins those
gathers before catch-up and drafting, instead of just before the prefill's
launch, so the host copies run while the drafter does. A step refused after
that begin settles its gathers before it returns. Adopted on 2026-11-06
with the other DSpark prefill steps ([served timing](#served-timing)).

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
`CHOPINFER_DEEPSEEK_CHECKPOINT`. Each source must match both
`checkpoint-files.json` and the adapter's pin before import. Synthetic tests
generate their own tokenizer, configuration and weights, and never read the
official weights.

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
Its 83 Rust sources and its Python sources are identical to commit
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

Sampling and limits: the server keeps the request's sampling and declares
default output lengths. A chat or Responses request without `max_tokens`, `max_completion_tokens` or
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

`tp4-dspark` is qualified through 32 sequences, its default, by the release
preview's full suite at d3d580aa02, which is the stored last-good result.
`tp4-ar` stays the experimental baseline that speculative output is checked
against. `qual.json`, `validation.toml` and the pinned reference locks are
authoritative for the checks and their inputs.

The reference is the checkpoint's pinned vendor code in F32 with
BF16-rounded trained-quantizer inputs (F32-TQ), with BF16 and F64-TQ spread
variants. Teacher coverage is at least 3974 positions and chat at least
5270; pooled mean absolute logprob limits are 0.150 and 0.300 nats.
Category limits follow the same frozen spread policy. Paired-teacher uses
the reviewed 0.005-nat pooled and 0.01-nat category mean margins, plus the
shared top-one and tail gates. Limits and references never move to hide a
mismatch.

The suite includes tokenizer/frontend goldens, per-layer taps, long context,
GSM8K, media, prefix equality, determinism and a BF16-KV
comparison.

Two checks keep KV storage apart from production accuracy. The `kv-bf16`
configuration's `kv_layout_equality` requires native mixed KV and BF16 KV
storage to serve the same bits, bit for bit, on the determinism request set
and on forced records (`verify_exact.py` K=0) of the exact reference's
texts and the pinned long_context text. That text is one request of a
130,048-token prompt with 1,024 forced outputs, 1,023 of them from decode
rows. In a `diagnostic-taps` build, `kv_layout_equality.py taps` also
requires every text tap and decoded KV-state tap to be equal through
prefill, decode and a prefix-hit restore. The BF16 layout stores exactly
the values the native formats decode to, and at Flash TP4 it rounds prefill
rank partials to BF16 as native does, so the comparison isolates storage.
Routes only the native layout takes are compared as served: the served
prefill attention and its decoded BF16 cache, the scheduled latent decode
and latent cache, the indexer's MMA prefill tile and its ordered decode and
verify merge, and upper-skip DMA reductions. The BF16 server runs its
admitted paths there, and the bits must still be equal.
BF16 KV is a diagnostic comparison and is not qualified for capacity,
pressure preemption or offload. It explicitly disables speculation.

Production accuracy is qualified separately, on native KV: the default
configuration's teacher (3974 positions), chat, long_context, multimodal,
`verify_exact` pairs, greedy AR versus DSpark and determinism teacher-force
or compare at matched histories against the pinned F32-TQ reference or AR.
`paired_teacher` and `gsm8k_regression` stay informational until a stored
passing baseline exists.

The bundle declares no KV offload: the server keeps no KV in host memory,
and a preempted request recomputes its KV. `qual.json` declares offload not
served, so the release preview reports it as not applicable.

The soak judges host memory as VmRSS less the Rss of the Engram tables'
read-only checkpoint mappings, against the 512 MiB limit
(`HOST_MEMORY_SCOPE` `rss_less_mapped_views` in `qual.json`, with
`MAPPED_VIEW_FILES` naming `model-00047-of-00048.safetensors` and
`model-00048-of-00048.safetensors`, the shards that hold the tables of
Engram layers 1 and 14). Only those files' shared, read-only, clean
mappings that the server holds at the first sample, and in every sample
since, are exempt; every other mapping counts, including the CUDA
driver's pinned host buffers mapped through the GPU device nodes.

The [numerical gates](../../docs/numerics.md#serving-qualification-gates)
describe the shared acceptance rules. Store captures, qualification
receipts and timings outside the repository under `CHOPINFER_DATA`.

### Diagnostic taps

A bundle built with the `diagnostic-taps` feature and served with
`CHOPINFER_TAPS_DIR` writes the taps [`taps.json`](taps.json) declares,
through the shared tap writer; `tools/qual/tap_binding.py capture` binds
them to the serving process and request. Besides each prefill's per-layer
text taps (rank 0, after each stage), two families of detail taps, kept
only in the layers `CHOPINFER_TAPS_LAYERS` names, attribute a divergence
between KV layouts:

- reduction inputs: each sublayer's completed local F32 partial on every
  rank (`L<nn>.x.attn.partial`, `L<nn>.x.mlp.partial`), read before its
  all-reduce, which may round it in place. They cover the eager prefill
  alone: a tapped instance never runs the upper-skip schedule or its
  ordered DMA reductions, decode steps write KV-state taps only and verify
  steps write none, so DMA reductions and decode and verify reductions,
  phased or not, are not tapped;
- KV state: a request's window rings, compressed latents and index keys on
  every rank (`L<nn>.x.kv.window`, `L<nn>.x.kv.compressed`,
  `L<nn>.x.kv.index`), as the logical BF16 values the attention reads,
  after each prefill chunk and decode step and before the first step of a
  request restored from a prefix hit. The native layout also writes its
  stored codes and scales (`.codes`, `.scales`).

Text and KV-state taps run every prefill stage by stage, which keeps no
prompt-end anchor, so a tapped instance offers the scheduler none and the
host ends a chunk at the prompt's last whole page (a multiple of 64 tokens)
to capture it instead. Its prefix hits therefore restore only from captures
at a committed chunk end, and the KV-state taps cover those restores. A
tapped server never restores from a prompt-end anchor, so that path stays
untapped.

`taps.json` states their files, shapes and row positions, and the coverage
they leave unresolved: reductions outside the eager prefill and, of the KV
state, verify steps, instances with the DSpark drafter loaded, anchor
restores, ring cells outside the window and the compressors' open groups.
Serving builds have no taps, and a serving bundle refuses to create while a
taps variable is set.

### DSpark qualification

`tp4-dspark` requires AR and speculative arms, exact greedy equality, the
declared verify bounds and GSM8K at least 193/200 in each arm. Forced verify
capacities are judged separately against paired AR, without pooling away an
uncovered capacity.

A package with exact verify graphs reports each verify step's live row
count as its row capacity, not a padded bucket; AR decode keeps its
power-of-two buckets. A pair at concurrency C with draft budget K therefore
forms at most C * (K + 1) verify rows; at the power-of-two concurrencies
below, its AR control forms C decode rows. Each `verify_exact` check
requires exactly those capacities in both its records and its report. The
default full cohort runs the profile's balanced pair at C1. Each
`verify-bN` configuration fixes its draft budget with `--speculation fixed`,
so adaptive depth selection cannot change it; a partly filled batch or a
short request tail can still form fewer rows, and the observed-capacity
coverage rule is what requires each shape to occur.
`tests/test_verify_capacities.py` holds the declaration to this rule.

| Configuration | Check | Draft budget | Concurrency | Verify capacity | AR capacity |
| --- | --- | ---: | ---: | ---: | ---: |
| default | `verify_exact` | 5 | 1 | 6 | 1 |
| `verify-b2` | `verify_exact_b2` | 1 | 1 | 2 | 1 |
| `verify-b4` | `verify_exact_b4` | 3 | 1 | 4 | 1 |
| `verify-b8` | `verify_exact_b8` | 3 | 2 | 8 | 2 |
| `verify-b12` | `verify_exact_b12` | 5 | 2 | 12 | 2 |
| `verify-b16` | `verify_exact_b16` | 3 | 4 | 16 | 4 |
| `verify-b24` | `verify_exact_b24` | 5 | 4 | 24 | 4 |
| `verify-b32` | `verify_exact_b32` | 3 | 8 | 32 | 8 |

The capacities name the tuned verify rows at 4, 6, 8, 12, 16 and 32, and
the shapes balanced planning serves at its depths: 6 rows at C1, 12 at C2,
24 at C4 and 32 at C8. A missing capacity or a dropped row observation
fails the check.

## Served timing

On 2026-11-06 a combined timing block on `tp4-dspark` compared four
compositions in two ABBA pairs of server visits on an idle host, every GPU
at its 300 W limit: the previous defaults (A), A with the DSpark prefill
steps adopted together (C: the prepared verify-row gather, the early Engram
begin and plan, and streamed lowering), C with the peer-copy event wait
(B), and A with the prefill kernel routes (K). Each measured cell ran 20
prefill steps at row counts no earlier step used, beside C8 verify phases
after a 93,056-token prefix: 1,536 to 1,555 rows (live 1.5K) and 2,880 to
2,899 rows (live 2.9K). Regression controls ran 1,024 to 1,043 rows beside
C8, 8,000 to 8,019 rows beside C16 after a 98,304-token prefix, and the
8192-row chunks of a cold 200,003-token prompt. Every arm produced the same
token IDs and top-1 logprobs. Each gain below is the mean over both pairs of
the reduction in the median prefill-step time.

C against A: 11.3% at live 1.5K (349.0 to 313.6 ms), 9.3% at live 2.9K
(453.9 to 415.2 ms), 11.6% at 1,024 rows, 5.8% beside C16 and 1.7% for the
chunks; decode steps were unchanged within noise.

On 2026-11-06 a fixed-batch job on `tp4-ar` compared streamed lowering on
and off: prompts of 101,305, 40,000 and 8,703 tokens, each cold and again
from the prefix cache, ran the same 114 serving steps with the same template
counts (17 misses, 3 hits, 4 refused) and returned the same token IDs and
top-5 logprobs, bit for bit.
