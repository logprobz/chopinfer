# Qwen3.8 reference inputs

The bound F32 teacher reference (`teacher-reference.json`, below) is also
the `exact` input of `models/qwen38/qual.json`: verify_exact.py,
prefix_exact.py and the greedy stability checks score against its `exact`
variant, as MiMo's do against its teacher reference. The former exact input,
`teacher-exact-reference.json` (the CPU `qwen38-oracle teacher --mtp` f64
run of 2026-09-24, SHA256
`a6df0d4455fa67f598ee37d610059f0f3f80d21d490d457cdcf3c9764f50ada2`), is
retired with the CPU reference paths (reference evaluation runs on the GPU
since 2026-10-01). Its `mtp` scores, the MTP head's, were pinned as a
variant but no gate read them, and HF qwen4_exp has no MTP module for the
runner to score.

`weight-plan-goldens.json` records, at 2 and 4 ranks, each rank's
arena length, SHA-256 and view-layout digest for the synthetic checkpoint of
`tools/load-proto/tests/synthetic_plan_goldens.rs`. The rank-image
materializer produced them on main of 2026-09-29 before it was deleted; the
same run checked that the contract's plan, loaded by the host's direct
executor, equals every image and index. That test now compares the planned
arenas with them. They cannot be regenerated from the current tree; do not
edit them to make a changed plan pass.

`image-probe-references.json` (schema
`chopinfer.qwen38.image-probe-references.v5`) holds the reference answers to
the image probes of `frontend/images.json`, from transformers 5.16.1's
`Qwen4ExpForConditionalGeneration` on the GPU reference runner
(`models/qwen38/reference.py`, [reference runs](#gpu-reference-runs)). Each
probe keeps the F32 run's greedy answer (64 tokens at most, ending at an end
token) with its text and whether it stopped, and that answer teacher-forced
as a served chat renders it (without the end token, and with a leading
space trimmed) in three variants: `exact`, the F32 run that the gate scores
against, and the BF16 and f64 runs, whose distance from it sets each
free-text probe's limits (`rule`, unchanged from the CPU references: mean
floor 0.15 in steps of 0.05, capped at 0.3; top-1 floor 0.08). A short
answer (blue-square, frame-00, four-images) is bound by its greedy entry,
whose steps it keeps under `greedy`; a free-text answer keeps the steps of
the F32 run that generated it under `generated`, which no gate reads.
`token_bytes` holds the bytes of every token an answer or a step's top 5
names. The workload, with 407 MB of the processor's F32 pixel rows, is too
large for the repository: it stays at
`$CHOPINFER_DATA/qwen38-refs/refrun/image-probe-references.workload.json`,
which `models/qwen38/qual.json` declares (`references.image_probes.workload`).
It is rebuilt from tracked inputs alone, on CPU in about 15 seconds, and
installed there (an existing file with other bytes is refused):

    CUDA_VISIBLE_DEVICES= $CHOPINFER_REFRUN_PYTHON -B models/qwen38/reference_workloads.py --probe-workload

The probe PNGs come from `capture/images.py` (including blue-square and
frame-00, which it encodes byte for byte as the MiMo functional fixtures
hold them) and must hash to `frontend/identity.json`. The pixel rows come from
the pinned processor's steps (`image-processor.json`'s recipe, with torch's
uint8 bicubic antialiased resize that torchvision runs on CPU) and must hash
to `image-processor.json`'s digests; `image-processor-identity.json` is the
captured processor identity the rows are keyed with. The sequences take
the bound prompts and answers from `image-probe-references.json`, and the
result must have the binding's workload digest. The only other input is
the checkpoint's `preprocessor_config.json` (by digest).

`image-decode.json` pins image decoding: what transformers 5.16.1's
`load_image` under Pillow 12.3 decodes from each file of runtime-media's
image decode corpus, with the checks that the image processor adds no
decode step and that every probe above needs no orientation (all are
opaque RGB). `capture/image_decode.py` writes it on CPU, and
`models/qwen38/compiler/tests/media_declaration.rs` holds the served decode
to it.

`agentic-outcomes.json` holds the two Qwen3.8 agentic model outcomes recorded
by a 2026-10-01 replay of agentic sessions that never sent a final answer: coding-000050 at C4
ended its turn inside its thinking, and coding-000035 at C8 closed its
thinking with a tool region the grammar does not define. It holds no
generated or parsed text: each turn keeps its prompt's last 64 token IDs,
every generated token ID, the finish reason, usage counts and the SHA-256 of
what Chopinfer's parser streamed, with the bytes of each token ID from the
checkpoint's `tokenizer.json`. `capture/agentic_outcomes.py` rebuilds it from
`$CHOPINFER_DATA/qwen-agentic-nofinal`; `tools/qual/test_agentic_review.py`
rebuilds each turn's parsed stream from its tokens, checks it against that
digest, and replays the sessions through the agentic driver.
The replay uses a synthetic workspace with the recorded paths, readable
line ranges, and first edit span, independent of the default coding corpus.
It preserves the recorded tokens and parsed-stream digests.

The [frontend references](frontend/README.md) use HF templates, tokenizers,
processors and F32 API answers. The [capture instructions](capture/README.md)
describe the CPU producers and queued reference runs.

## Teacher gate reference

`teacher-reference.json` (schema `chopinfer.qwen38.teacher-reference.v2`) is
the reference `models/qwen38/qual.json` gates the teacher check on. It holds
the 4,176 positions of the 35 texts in `teacher-texts.json` (the 34 short
teacher texts of `prompts.json` and `long_4k`, category `long`) in three
variants of the GPU reference runner:

- `exact`, the gate variant: the F32 run, FP8 weights times their scales
  exactly;
- `bf16`, a spread variant: the BF16 run, weights rounded to BF16 as HF loads
  them, F32 log-softmax;
- `f64`, a spread variant: the f64 run, keeping HF's own F32 steps.

Scored against `exact`, `bf16` is 0.066 mean |dlogprob| and 2.61% top-1
disagreement (109 of 4,176) and `f64` is 0.0017 and 3 (all in long_4k), so
the pooled limits are 0.10 and 0.08; each category's own spread sets its
limits (chat's BF16 spread, 0.151, gives 0.20). Each variant copies the
runner's columns: logprob, top-1, margin (the logit gap) and the top 5 IDs,
logits and logprobs. The binding names the cache entries, and
`teacher-reference.workload.json` beside it is the workload they scored.

The former reference, the retired CPU `qwen38-oracle` f64 chain with the HF
BF16 CPU run as its spread, was 0.0022 mean |dlogprob| and one top-1 flip
from the new `exact`; its HF BF16 spread was 0.067 and 2.73%.

## Long-context reference

`long-context-reference.json` (the same schema) gates `long_context`
on the last 1,024 positions of `long_32k` (32,768 tokens, from the full
prompt set `prompts.json` pins), in the same three variants. It replaces the
vLLM long-text capture as the long-context accuracy check. `long_128k` is
left out: HF's eager text attention holds a [24, 2048, 131072] score tensor
per prefill chunk and its f64 run would take about an hour, beyond a GPU job.
Scored against `exact`, `bf16` is 0.022 mean |dlogprob| and 7 top-1 flips
of 1,024 and `f64` is 0.010 and 3, so the limits are the floors for a set of
1,000 positions or more, 0.10 and 0.08. The f64 run scored a 64-token lead
sequence first, so the runner's repeat check (each mode's first computed
sequence twice) did not double the 877 s pass; emit leaves the lead out.

## GPU reference runs

The fixtures above come from `tools/refrun` runs on 2026-10-01 with
`models/qwen38/reference_workloads.py` (workloads) and
`models/qwen38/reference_fixtures.py` (skeletons), assembled by
`python3 -m tools.refrun emit --modes f32=exact,bf16,f64`. The image probes
take two steps: the F32 run generates each answer, and the forced workload
(`reference_workloads.py --answers`) renders the free-text answers as a chat
does before all three modes force them. Rerun them, rather than edit a
fixture, when the checkpoint, the adapter or the runner changes; the
bindings refuse a stale fixture (`python3 -m tools.refrun verify-binding`).

## Third-party data

The GSM8K questions in the chat and tokenizer fixtures come from
[OpenAI's grade-school-math](https://github.com/openai/grade-school-math/tree/3101c7d5072418e28b9008a6636bde82a006892c),
revision `3101c7d5072418e28b9008a6636bde82a006892c`.
Copyright (c) 2021 OpenAI. The [MIT notice](../GSM8K-LICENSE) applies to those questions.
