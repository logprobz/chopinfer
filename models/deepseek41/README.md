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

Flash TP4 prefill uses native block-scaled expert tiles. At TP4 the rank
image stores the backbone's routed gate/up bank in 64-row by 128-K row
tiles with each 64-byte block of the codes in quad-lane order
(`row-tiles-64x128-lanes`), which the native tiles read and the exact BF16
expert path does not. `CHOPINFER_DEEPSEEK41_NATIVE_MOE_PREFILL` unset or
`1` selects native; `0` or any other value selects the exact path, so a TP4
create refuses and logs why. TP1 and TP2 store rows and run the exact path.
The native tiles use the trained operand grids and change the F32
accumulation order within each 32-K group. Decode, verify and draft experts
keep their existing paths; at TP4 the backbone's decode and verify gate/up
kernels read the quad-lane tiles too, bit for bit as they read rows: each
lane of a quad loads the code bytes it converts together, so no shuffle
forms its MMA operands. Native prefill restores the row-tiled order in
shared memory, which costs it up to 3.3% of its MoE time against row tiles
(2048 to 8192 live rows, sustained at 300 W, 2026-10-23). See
[native MoE prefill qualification](lab/native_moe_prefill.md) for the
numerical contract and fixed gates.

The routed expert scale banks of the main layers are stored encoded
([encoded expert scale banks](../../docs/kernel-library.md#encoded-expert-scale-banks)).
Flash TP4 uses `scale-row-tiles-offset-v1`; other topologies use
`scale-row-offset-v1`.
The bundle exports `chop_bundle_weight_census_v1`, the host measures the
banks on the CPU before planning, and the census chooses an offset width
or keeps the bank raw. With the row-offset format, one launch at the end
of each layer's routes stage expands both banks for the experts with routes into raw banks in scratch
(106.2 MB at TP4), on the routed branch beside the shared expert. The
experts read those banks unchanged. DSpark's banks stay raw: the one-row floor kernel loads
scales before its PDL wait. On the official checkpoint at TP4, each rank's
routed scale banks shrink from 4,246,732,800 to about 697,200,000 bytes,
decoding bitwise to the raw banks (2026-10-23, CPU). Encoding is on by
default; `CHOPINFER_WEIGHT_CENSUS=0` on the host plans every bank raw, which
is also the A/B control. Served at TP4 AR on 2026-10-23 (test KV mode,
65,536 pages, a raw/encoded/encoded/raw sequence after a warm-up process),
every greedy and long-prompt token and logprob bit equaled the raw server's,
each rank had 3,296 MiB more free memory before its KV pool (60,676 more
pages at utilization 1.0), the census took 6 to 7 s at startup, and greedy
decode ran at 0.974, 0.975 and 0.960 of raw tokens per second at 1, 8 and
32 sequences. Those ratios are with the paired expansion's warp exception
search; they were 0.973, 0.974 and 0.961 with one-thread searches.

The tile-scale format keeps the same census and width-or-raw decisions.
Its declared decode and verify consumers read encoded scales in place,
keeping their raw counterpart's arithmetic and launch geometry. A
projection retains expansion if any of its selected queues lacks an
encoded variant. This includes wide gate and column queues and native
prefill. The one-row width-4 gate also retains expansion. The raw scratch
banks remain allocated for these paths; DSpark's drafter banks remain raw.
On the official checkpoint, the tile format takes 673.39 to 673.46 MB per
rank and decodes bitwise to the raw banks (2026-10-24, CPU).

Every MXFP8 dense projection stores its E4M3 codes in
`mxfp8-fold-packed-8x128-v1` ([packed FP8 fold layout](../../docs/kernel-library.md#packed-fp8-fold-layout)),
the layout every fold reader takes. Create therefore no longer builds a
packed copy of them beside row-major codes.

Served at TP4 on 2026-10-23 against the previous layout (test KV mode,
65,536 pages, a base/head/head/base sequence after a warm-up process):

- AR and DSpark outputs were bitwise equal.
- Greedy throughput was within the A/A spread or faster: AR 0.9997,
  1.0028 and 1.0103 at 1, 8 and 32 sequences, DSpark 1.0020, 1.0067 and
  1.0039.
- Each rank had 1,922 MiB more free memory at the first graph capture,
  35,382 more pages at utilization 1.0; with the drafter it was 2,116 MiB.

## Checkpoint and geometry

`checkpoint-files.json` pins every file of the official revision, for
provenance and the qualification binding. Admission reads `config.json`
alone: aliases, semantic rules, the block-32 FP8 recipe, the CSA2, Engram
and DSpark relations (`contract/src/structure.rs`), then the closed match
against `geometries/flash.json`, each refusal naming its field. The
synthetic geometry (`fixtures/deepseek41/geometries/synthetic-small/`, 12
layers carrying every CSA2 role, 32 experts, one DSpark stage) admits at
TP1, TP2 and TP4 when `CHOPINFER_DEEPSEEK41_GEOMETRY_DIRS` names its
directory.

`contract/projections.json` declares every projection, stem, side tensor
and encoder tensor once; `checkpoint::bind` binds a checkpoint's
safetensors headers to it before any payload is read. The official
revision binds all 96,085 tensors, each claimed once
(`compiler/tests/checkpoint_binding.rs`).

`text_config.indexer_mma` defaults to `true` for Flash. Native MXFP4 index
keys use the guarded FP4 tensor-core scorer in prefill and decode; setting
`--model-option text_config.indexer_mma=false` keeps the SIMT scorer for
comparison. BF16 key storage retains SIMT. A coarse scale certificate admits
safe native scores; otherwise each head's four scale sums are checked. The
fine certificate requires sums in [130, 369], a spread of at most nine, and
individual scale bytes in [2, 252]. Rejected slots use the original SIMT
operation order. Queries that cannot be repacked exactly use SIMT for the
whole row. Ordered selection, candidate provenance and failure publication
retain their contracts.

Native Flash decode buckets 1-32 use the ordered hierarchical merge. Eager
prefill uses cached local selection only when the rounded context extent is
128K, the prepared batch has at least 32 rows, and k is 512. That band keeps
the coarse scorer and original final merge. Other prefill extents retain
their previous selector; the four-row shared-key scorer remains unselected.

Full C8 and C32 decode batches and verify-32 batches of eight four-row items
use the 128-position tiled scorer and cached dense local selector from 128K
through 256K. Verify-32 also uses hierarchical merging from 32K. Contexts
are the requests' ends rounded to 64-token pages; every request must fit the
same band. These routes require index top-k 512 and 2048 candidate blocks.
Partial batches, other verify layouts, shorter contexts and
contexts above 256K retain the preceding schedule. The 32K decode schedule
stays unchanged because its measured scorer gain was small and cached
selection lost time. Prefill retains the separate policy above.

Each schedule has its own graph cache, including both DSpark target pieces.
Switching contexts retains warm graphs and the original graph-release
ledger. Source grids, scalar arguments, score strides, candidate publication,
guard predicates and SIMT repair remain unchanged. At 128K/256K, paired
eight-source measurements saved 0.225/0.334 ms per C8 step,
0.299/0.509 ms per C32 step and 0.555/0.804 ms per verify-32 step against
the preceding schedule. Verify hierarchy alone saved 0.248 ms at 32K.
These are per-rank indexer times, excluding query projection and attention,
from seeded independent-request layouts. The qualification also retained
sealed sampled prefill operands and adversarial repair cases. It establishes
bitwise indexer output equality; integrated served speed remains a separate
measurement.

`text_config.indexer_prefill_tile` defaults to true for native Flash eager
prefill with top-k 512 and 2048 candidate blocks. It selects eight score
CTAs per row for 4096-row ratio-2 and compact batches, and 512-row
layer-20 batches, when every position is below 32K. Compact scoring uses
a 128-position tile.
Each CTA reuses its packed query across more positions. Compact scoring
uses all four warps for the same ordered head-FMA sum. Guards, SIMT repair,
score layouts and selectors are retained. Smaller tails and other contexts
keep the accepted schedule. Set the option to false for the previous
schedule; disabling native MMA also disables this route.

On 2026-10-20 at 250 W, 94 paired cases and 23 memcheck cases passed
bitwise, including ten real captures, poisoned buffers, finite SIMT
packing fallbacks and chunk partitions. Across 8192-row synthetic chunks
at offsets 0, 8192, 16384 and 24576, the score-and-selection pipeline
projected across all eight indexer layers fell from 71.580/109.273/143.251/
160.122 ms to 37.154/71.434/102.156/116.954 ms per rank. The four-full-chunk
saving is 156.527 ms; upper-skip reduces the number of rows and the
available saving. The ten real 32-row cases were slower and retain their
previous route. Integrated served latency remains a separate measurement.

Under the same admission, layer 20 publishes its candidate blocks with one
fused launch per batch instead of the block-maxima, local-gather and merge
launches. It reads the selector's scores directly, keeps block keys in
registers, resolves ties toward the lower block index and leaves the public
outputs, tags, visible counts and status on their previous spans. The serving
cubin is built from the lab recipe and gated by the lab's lock records. On
2026-10-20 at 250 W, 198 cases (two real layer-20 captures, offsets 0, 8192,
16384 and 24576, runtime budgets 1, 31 and 2048, ties, holes, bad counts,
invalid scores and inactive tails) gave 1152 bitwise comparisons and 396
rejected negative controls, and 36 memcheck cases passed. At budget 2048 the
complete layer-20 pipeline on its 2496-row upper-skip suffix fell from
2.518/4.061/7.333/9.148 ms to 1.726/3.235/6.326/8.034 ms at those offsets,
about 3.7 ms per cold 32K prompt per rank. The reference arm was the
runtime-count producer; at the compiled count it runs the same gather
template as the served producer.

On 2026-10-19, captured served operands at 8K, 32K and 128K gave exact f64
top-k set agreement and recall of 1.0 for SIMT, BF16 MMA and native FP4 MMA:
475,668 selections across 960 rows. All three had score RMSE 1.4289e-7 and
maximum error 1.1509e-6. Both MMA paths had zero score-bit differences from
SIMT across 24,772,608 compared words, including tails. These measurements
qualified the original unguarded scorer on those captures. The later guard
repaired the synthetic mismatches; no repair was needed in 17.7 million
real slots. Unguarded MMA remains a lab control, not the serving default.

The unguarded cancellation witness is consistent with a different full-F32
addition order: `(2^25 + 1) - 2^25` can lose the unit that SIMT preserves by
cancelling the large terms first. It is outside the guard's scale domain.
It does not establish reduced accumulator precision. PTX specifies at least
single-precision accumulation for this E2M1 instruction, but leaves order,
rounding and subnormal-input handling unspecified. The guard's integer proof
requires exact decoded products; domain-edge and individual-code GPU probes
check this premise on SM120 before a stronger hardware-wide claim. See the
[guard argument](../../crates/kernels-sm120/cuda/pooled_indexer/position_mma_guard.md).

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

The model is not qualified yet. Reference-based numerical limits in `qual.json` are
pinned from the spread of the refrun references of the pinned vendor code
before the first candidate run, and never moved after a result. The served
checks pin checkpoint identity 2273a3d9..., tensor parallel size 4 and the
native KV layout, which `/health` reports as `mixed` (what
`--kv-cache-dtype auto` resolves to).

- teacher and chat: gated against the F32 run with BF16-rounded quantizer
  inputs (F32-TQ), `$CHOPINFER_DATA/refrun/deepseek41/m1/teacher-reference.json`
  (sha256 fcbd248b..., 35 texts, 3,974 scored positions) and
  `chat-reference.json` (03c03237..., 17 texts, 5,270 positions), with
  BF16 and F64-TQ as spread variants. The limits follow spread rule
  variant-spread.v2 with mean factor 1 (`m1/spread-limits.json`, sha256
  b275dea1...). teacher: mean |dlogprob| at most 0.15 (the 0.10 floor of a
  set of at least 1,000 positions, raised to the pooled BF16 spread 0.120
  and rounded up), top-1 decisive and near-tie at most 8%, disagreement at
  most 10%, no failed category and at least 3,974 positions scored; the
  gate computes each category's limits from its own spread (prose-frozen
  mean 0.20 and total 10.79%, prose mean 0.20, chat mean 0.40, the others
  at the floors). chat: mean |dlogprob| at most 0.30 (pooled BF16 spread
  0.292), the same top-1 limits and at least 5,270 positions. The teacher
  check reads the reference's frozen texts from
  `$CHOPINFER_DATA/deepseek41/m1/fixtures-text/teacher-texts.json`.
- paired_teacher: the candidate's teacher receipt against a stored one on
  the same reference, with the standard reviewed margins
  (paired-teacher-margins/v1: 0.005-nat pooled and 0.01-nat category mean
  margins, 1% top-one delta, Holm alpha 0.05, eight clusters, two net tail
  positions above 0.5 nats), reviewed on 2026-10-18 (its approval line is
  in `qual.json`). It activates once a stored teacher receipt exists.
- long_context (131,072 tokens): the teacher rule over the last 1,024
  positions against the F32-TQ reference alone, under the floor-only policy
  of `docs/check-kinds.md` (mean |dlogprob| 0.10, decisive and near-tie
  top-1 0.08, all top-1 flips 0.10).
- per-layer taps (`taps.json`) within the same spread;
- evals_gsm8k: the 1,319 GSM8K items with the recipe sampling (temperature
  1.0, top_p 1.0, max_tokens 262,144), workload sha256
  bb39334b089e2861ec26b6c700c74951e8293e61268b5acef3b64b93fc73fe76;
  gsm8k_regression compares it with the stored run, the same sampled
  bodies, at significance 0.05 and reports no baseline until one exists.
  evals_gsm8k gates coverage, identity and errors, gsm8k_regression gates
  against the stored baseline, and the absolute GSM8K floor is
  spec_verify's.
- frontend_goldens: the served tokenizer agrees with
  the 49 rows of `fixtures/deepseek41/frontend/tokenizer-golden.json`.
  Its live input, `$CHOPINFER_DATA/deepseek41/frontend-agentic/tokenizer-golden.json`,
  materializes the five checkpoint-file texts after checking their SHA-256
  and re-encoding them with the pinned tokenizer; every token ID stays the
  same. The 61 `chat-renders.json` rows use the fixed conversations from
  `fixtures/mimo26/capture/conversations.py`, with tool arguments serialized
  as the live capture sends them. `tools.refrun.frontend_corpus.Model`,
  `template_source` and `model_reference` render and tokenize them on CPU
  through `frontend_cases.py` and the pinned checkpoint encoder. API reads
  the same `tools-00` conversation; the frontend gate requires equality
  for both the tokenizer and chat files, except for the two text digests in
  the check's `tokenizer_refusals`: FIM and the encoder's action task are
  outside the served prompt format. Those rows require the specific HTTP
  400 prompt-special-token refusal. The chat generator uses the same
  request builder as the check, so disabled thinking reaches the vendor
  encoder through the frontend's declared thinking object.
  From the repository root, with the configured checkpoint and reference
  Python, regenerate and verify the chat fixture on CPU:

  ```sh
  CUDA_VISIBLE_DEVICES= "$CHOPINFER_REFRUN_PYTHON" -B fixtures/deepseek41/capture/chat_renders.py --output "$CHOPINFER_DATA/deepseek41/frontend-agentic/chat-renders.json"
  cmp fixtures/deepseek41/frontend/chat-renders.json "$CHOPINFER_DATA/deepseek41/frontend-agentic/chat-renders.json"
  ```

- prefix_exact: the `deepseek41` policy uses the teacher reference's schema,
  64-token pages and a 64-token prefill granule from
  `contract/src/serving.rs`. It checks page and prompt-end anchors, image
  isolation and continuations after a hit, as the shared prefix check does
  for GLM, MiMo and Qwen. The `anchor_in_window` and `anchor_past_window`
  report groups split at Flash's 128-token sliding window
  (`geometries/flash.json`); `bundle/src/instance/prefix.rs` captures and
  restores those rings and compressor tails. These settings describe the
  source contract, not a served result.
- determinism: the AR and DSpark records of one run are identical.
- kv_offload_plain and kv_offload_pressure: native-KV offload with a 16 GiB
  budget, a swap every 16 steps and 32,768 pressure tokens.
- the `kv-bf16` comparison configuration (`serve --kv-cache-dtype bf16`):
  the BF16 KV layout holds the same values as the native trained formats,
  and native (mixed) versus BF16 served bits must be equal. It runs prefix_exact
  and kv_layout_equality, which records the determinism request set on a
  native server and on the configuration's BF16 server in one run and
  requires every request's tokens and logprobs to be bit-identical, naming
  the first differing position; teacher, teacher_arms and chat join as
  they land. It is never a served default and not qualified for capacity,
  preemption under pressure or offload, since its pages are about 3.6
  times larger (3,200 bytes against 890 per token).

The needle check runs frozen token-ID prompts at total contexts of 8,192,
131,072, 262,144 and 1,048,576 tokens (7,168, 130,048, 261,120 and
1,047,552 prompt tokens, each with 1,024 tokens to answer) and requires the
three needle values in order. `fixtures/deepseek41/capture/needles.py`
builds them on CPU from MiMo's functional haystack, rendered by the
checkpoint's `encoding/encoding.py` in chat mode, with the needles near
10%, 50% and 90% of the haystack, into
`$CHOPINFER_DATA/deepseek41/needle/prompts.json`; the digests are in
`fixtures/deepseek41/needles.json` (`--check` rebuilds and compares).

`profiles.json` holds the experimental `tp4-ar` and `tp4-dspark` profiles. The
release preview's full suite at the release revision is the
qualification evidence. Both run exact upper-layer prefill intervals at
guaranteed 8192-token chunk ends ([prefill-dependencies.md](prefill-dependencies.md)),
which on 2026-10-19 matched the full schedule bitwise in served tokens,
logprobs and captured state at 8K, 32K and 128K prompts at TP4. Prefix
anchors exist only at chunk, prompt and response ends. `tp4-ar-full-prefill`
and `tp4-dspark-full-prefill` keep the full schedule and anchor anywhere;
`tp4-ar-upper-skip-2048` and `tp4-dspark-upper-skip-2048` use 2048-token
chunk ends.

DSpark prefill batches commit projections across prompts and overlaps each
Engram layer's mapped gather with target compute by default.
`CHOPINFER_DSPARK_BATCHED_COMMIT=0` restores separate commit projections;
`CHOPINFER_DEEPSEEK41_PREFILL_ENGRAM=0` restores synchronous prefill gathers.
AR prefill gathers remain opt-in. The shared host now reads ahead for both
synchronous and asynchronous mapped gathers, with independent overrides
([policy and caller audit](../../docs/mapped-row-read-ahead.md)).

On 2026-10-24, Flash TP4 at a configured 1800 MHz and 300 W passed bitwise
full-vocabulary logits, target KV, all 134 raw feature-history rows, drafter
state and sampled-token comparisons against the preceding prefill paths.
At temperature 1.0 with 512 output tokens, four processes per arm raised
median C8/C16 request throughput from 465.93/544.39 to 472.37/553.35 tok/s;
all spreads were below 0.85% with identical step histograms. Energy was
1.8250/1.5034 versus 1.8252/1.5149 J/token. Batched commit adds 121.91 MiB
per rank before KV sizing: automatic capacity fell from 320,972 to 318,615
pages per rank (0.734%). The gather ID buffers also reserve 1.5 MiB on
rank 0 before KV sizing. These bounded comparisons do not replace release
qualification.

`CHOPINFER_DSPARK_UNDRAFTED_STATUS=1` opts into one joined saved-status
readback for pure decode steps where no item drafts. On the native device
result path it submits private sampling before collecting that status and
completes every Engram gather before returning success. Prefill, mixed and
drafted steps keep their existing boundaries. This switch is experimental
until the full-state transition and served bitwise gates pass.

`CHOPINFER_DSPARK_INDEXED_RETENTION=1` opts into one byte-copy kernel per
rank when an undrafted or verify commit would otherwise submit at least
eight retention copies per rank. Prefill, mixed prefill/decode and smaller
commits retain the original copy list. Metadata reserves 768 bytes per rank
before KV sizing. All mappings and rank bindings validate before any write;
every accepted raw row keeps its original physical slot and wrap position.
This path also remains experimental until its GPU and served gates pass.
Step records expose `undrafted_ready` and `undrafted_status_packets`;
commit records expose `retention_kernel_launches`, `retention_copy_calls`
and `retention_metadata_bytes` so gates can assert which path executed.

For untimed transition diagnostics, build the bundle with `state-capture`.
This feature keeps native launches and ready logits enabled. Set
`CHOPINFER_DSPARK_STATE_DIR` to a private external directory, create its
`enabled` file, and supply `transitions.json`, for example:

```json
{"version":1,"before_draft":[4,5],"after_commit":[3,4,5],"requests":[1],"max_bytes":1073741824}
```

The step numbers count bundle executions, including prefill. Each selected
boundary captures the full physical state and target KV, plus committed
Engram lookback, lifecycle bookkeeping and resumed sampling inputs. A
`before_draft` capture follows ordinary catch-up and precedes provisional
draft writes; an `after_commit` capture does no catch-up. The plan allows
at most eight points of each kind and 32 requests, within its byte budget.
These readbacks synchronize device work. Run separate uncaptured native
exactness and timing windows to check execution ordering and performance.

### DSpark qualification

The preview selects `tp4-dspark`, with both AR and speculative quick arms
and exact greedy equality. DSpark computes all five proposals per block;
scheduler budgets select the emitted prefix. For a sampled request the
drafter proposes from the target sampler's own noise at each draft
position's generated index (model option `dspark_coupled_drafts`, default
true; `false` drafts every request greedily); emitted tokens do not depend
on it. Its profiles plan balanced
drafts at native KV and TP4: every stream drafts the deepest common depth
that fits the 32 verify rows, at most five from C1, three from C8 and none
from C16 (`depths` `1:5,8:3,16:0`), and no plan reads a step time, so
identical processes run identical step shapes. On 2026-10-23 at 300 W, four
fresh processes per arm on one install served balanced within a 2.2%
throughput spread at C8 and C16 with identical step shapes, while the
adaptive policy differed by up to 57% in C8 full-concurrency throughput.
Against no drafting, three drafts per stream at C8 raised full-concurrency
throughput by 44% greedy and 26% sampled; one draft at C16 lowered request
throughput to 0.92 greedy and 0.82 sampled, so C16 and C32 serve without
drafts. Each plan then lands on the widest verify width of 6, 8, 16 and 32
rows that its depth caps reach (`widths` `6,8,16,32`), filled by
acceptance: the target verify's MoE has its prepared plans at those widths
and runs a generic path between them, so a C8 step of 24 rows took 43.8 ms
against 37.5 ms for 32 rows. On 2026-10-23 at 300 W, against the same
profile without widths with three or four fresh processes per arm, this
raised C8 request throughput by 10.1% greedy and 9.1% sampled and lowered
energy per token by 6.4% and 5.8%, with C8 full-concurrency throughput and
C1, C16 and C32 unchanged. `--speculation adaptive` with the shared
`per-block` draft-cost model remains available. The model option
`dspark_draft_chunk_items` (default 12, at most 32) caps the drafting items
one drafter forward runs. A step splits into the fewest near-equal chunks
within that limit, with larger chunks first: 16 items use 8 + 8, 24 use
12 + 12, and 32 use 11 + 11 + 10. Target outputs do not depend on chunking.
The Flash TP4 DSpark profiles select this twelve-item limit on the bounded
head/router, HC and quantizer paths. An override above twelve items can
select the larger-row paths.
Diagnostic builds can set `CHOPINFER_DSPARK_HEAD_CAPTURE` to a private
external directory and create its `enabled` marker after startup. The
drafter then captures normalized F32 head inputs, BF16 head weights and
reference F32 outputs once at each of 40, 60, 80 and 160 rows. Readbacks
run after drafter dispatch, outside graph capture. Files are private and
never replaced; these untimed captures are for lab comparisons only.
The package declares up to 192 verify rows a step, every item at its
deepest block. The host serves 32 unless `--speculative-rows` (profile
`speculation.rows`) asks for more, and the instance prepares every op for
the rows it is asked for. The model option `verify_graph_rows`
(`--speculation-graph-rows`, profile `speculation.graph_rows`, default 32)
is the widest verify step captured in a graph; wider steps run without one.
No profile asks for more than 32 rows or graphs: on 2026-10-24 (TP4,
300 W, sampled at temperature 1.0, 128 rows, graphs up to 64 rows) the
median step took 16.4 ms at 8 rows, 29.4 ms at 16, 47.1 ms at 24,
37.8 ms at 32, 88.2 ms at 40 and 93.0 ms at 48 rows with 8 items, and
24.0 ms at 16 rows, 43.9 ms at 32, 99.6 ms at 48, 103.8 ms at 64 and
268.2 ms at 96 rows (no graph) with 16 items.
Widths without specialized kernels (24, 40, 48 and 64 rows) cost more than
32 rows or several AR steps, so full-concurrency C8 ran fastest at depth 3
(32 rows, 596 tokens/s against 488 for AR) and C16 at depth 0 (668
tokens/s; depth 1 at 32 rows 663, depths 2 and 3 at 48 and 64 rows 395 and
451), which the `tp4-dspark` depths and widths already select.
The DSpark paired-teacher policy
is keyed by `tp4-dspark` and uses the same approved margins as `tp4-ar`.

The speculative-verification copy set must have mean absolute logprob
difference at most 0.15 and top-1 disagreement at most 0.08 against AR,
as specified in [the serving qualification gates](../../docs/numerics.md#serving-qualification-gates).
Each arm records at least 16 copy requests and 16 natural requests, each
with at least one scored position, from the shared frozen teacher texts
and GSM8K questions. Both arms must bind to the official checkpoint and
the same package at TP4 with native KV.

The approved DSpark qualification plan sets the GSM8K floor at 193/200 for
both arms. This matches MiMo's floor and is one item looser than Qwen's
model-specific 194/200. These limits are declared before candidate results.

Recipe sampling (temperature 1.0, top_p 1.0) applies to serving and every
sampled check, while spec_verify, spec_greedy, greedy equality to AR and
determinism compare greedy paths by definition. Spec_verify inputs record
`{temperature: 0, top_p: 1, logprobs: 5}` as the shared tool emits them.

`verify_exact` compares target verification rows with paired AR rows against
the pinned F32-TQ teacher reference. It requires complete observed row
shapes, the same checkpoint, bound package identities and coverage of every
forced position. The relative mean margin is 0.005 and the excess top-one
bound is two standard deviations, from the serving qualification gates.
The full preview runs a separate `verify-bN` configuration for every prepared
capacity. Each records AR and speculative arms at the same concurrency,
requires the listed executed capacities in both capture and report, and
judges its own pair without pooling numerical errors across configurations:

| Configuration | Draft budget | Concurrency | Verify capacity | AR capacity |
| --- | ---: | ---: | ---: | ---: |
| `verify-b2` | 1 | 1 | 2 | 1 |
| `verify-b4` | 3 | 1 | 4 | 1 |
| `verify-b8` | 5 | 1 | 8 | 1 |
| `verify-b16` | 5 | 2 | 16 | 2 |
| `verify-b32` | 5 | 4 | 32 | 4 |

These configurations fix the draft budget so adaptive scheduling cannot
avoid a required capacity. The default full cohort also runs the C1 pair.
A missing capacity or dropped trace observation fails the check. No
speculative GPU evidence is claimed by this declaration.

The `kv-bf16` comparison explicitly disables speculation. DSpark runs on
native KV; both arms of the BF16 versus native comparison use K=0.
