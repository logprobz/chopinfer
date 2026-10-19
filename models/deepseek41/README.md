# DeepSeek V4.1 Flash (`deepseek41`)

DeepSeek V4.1 Flash, from the official checkpoint
`deepseek-ai/DeepSeek-V4.1-Flash-1017` at revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`, served as bundle
`deepseek41.v1` at topology `deepseek41-flash-tp4`. It is not released:
`model.toml` declares `[release] enabled = false` until it qualifies.

Supports images in tool results through Chat Completions and Responses, in the official encoder's prompt order.

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
keys use the qualified FP4 tensor-core scorer in prefill and decode; setting
`--model-option text_config.indexer_mma=false` keeps the SIMT scorer for
comparison. BF16 key storage retains SIMT. Selection, candidate production,
provenance checks and failure publication keep their existing kernels.
Queries that cannot be repacked exactly take the SIMT path for that row.

On 2026-10-19, captured served operands at 8K, 32K and 128K gave exact f64
top-k set agreement and recall of 1.0 for SIMT, BF16 MMA and native FP4 MMA:
475,668 selections across 960 rows. All three had score RMSE 1.4289e-7 and
maximum error 1.1509e-6. Both MMA paths had zero score-bit differences from
SIMT across 24,772,608 compared words, including tails. This qualified the
native scorer for promotion under the operator's accuracy rule. It is not
a general bitwise replacement: the synthetic cancellation witness still
loses a contribution preserved by SIMT, and randomized selections can differ.
The served A/B checks round three, the new default and selectable SIMT on
identical 8K, 32K and 128K prompts, including every published logprob bit.

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

### DSpark qualification

The preview selects `tp4-dspark`, with both AR and speculative quick arms
and exact greedy equality. DSpark computes all five proposals per block;
scheduler budgets select the emitted prefix. Its profile uses the shared
`per-block` draft-cost model and native KV at TP4. Its paired-teacher policy
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
