# qwen38

`compiler/` admits checkpoints and emits packages; `contract/` owns geometry,
precision, topology and frontend declarations; `bundle/` owns native execution.
Shared runtime code stays model-neutral.

Images may be PNG, JPEG, GIF (the first frame) or WebP (lossy, lossless,
with alpha, or the first frame of an animation), decoded as transformers'
`load_image` reads them: oriented by their EXIF or XMP orientation, alpha
discarded. [`fixtures/qwen38/image-decode.json`](../../fixtures/qwen38/image-decode.json)
pins that decode. See [the request surface](../../docs/serving-request-surface.md#media-inputs).

- [Model inventory](model.toml)
- [Serving profiles](profiles.json)
- [Qualification declaration](qual.json)
- [Checkpoint pins](checkpoint-files.json)
- [Model integration](../../BUNDLE_ARCHITECTURE.md#adding-a-model)

## Request defaults

A chat or Responses request without `max_tokens`, `max_completion_tokens`
or `max_output_tokens` generates at most 131,072 tokens when it thinks at
effort xhigh, the default, and 65,536 at medium or low effort or with
thinking off, each lowered to the context left after the prompt. An
explicit limit is honored. See
[the request surface](../../docs/serving-request-surface.md#request-parameters-and-defaults).

## Serving profiles

`tp4-mtp` is the serving profile; the AR profiles (`tp4-ar`, `tp4-ar-dcp2`,
`tp2-ar`) are its baseline and correctness control. The profile file owns
settings, status and qualified concurrency. Read the [qualification
scope](../../docs/developer-preview.md#qualification-scope) before treating
a compatible checkpoint or explicit override as qualified.

## Qualification

`qual.json` owns the exact checks, thresholds and inputs. The TP4 AR and MTP
profiles have BF16-KV text and image serving evidence; their qualified
counts are separate from the default cap. TP2 and both TP4 DCP2 profiles
remain experimental: their configurations run the declared checks at their
own topology, and a passing bounded cohort does not make a profile
release-qualified. Context-parallel sizes use separate qualification and
[stripe semantics](../../docs/numerics.md#context-parallelism).

The references are runs of the GPU reference runner (`tools/refrun`) with
the adapter `reference.py`. Rerun them, rather than edit a fixture, when the
checkpoint, the adapter or the runner changes; the bindings refuse a stale
fixture (`python3 -m tools.refrun verify-binding`). The image probe
reference's workload is too large for the repository;
`reference_workloads.py --probe-workload` rebuilds it on CPU from tracked
inputs and installs it where `qual.json` declares it.

### Teacher gate reference

`fixtures/qwen38/teacher-reference.json` (schema
`chopinfer.qwen38.teacher-reference.v2`) is the reference the teacher check
gates on, and the `exact` input that verify_exact.py, prefix_exact.py and
the greedy stability checks score against. It holds the 4,176 positions of
the 35 texts in `teacher-texts.json` in three variants:

- `exact`, the gate variant: the F32 run, FP8 weights times their scales
  exactly;
- `bf16`, a spread variant: the BF16 run, weights rounded to BF16 as HF loads
  them, F32 log-softmax;
- `f64`, a spread variant: the f64 run, keeping HF's own F32 steps.

Scored against `exact`, the spread variants set pooled limits of 0.10 mean
absolute logprob difference and 0.08 top-one disagreement; each category's
own spread sets its limits (chat's gives 0.20). Each variant holds the
runner's columns: logprob, top-1, margin (the logit gap) and the top 5 IDs,
logits and logprobs. The binding names the cache entries, and
`teacher-reference.workload.json` beside it is the workload they scored.

### Long-context reference

`fixtures/qwen38/long-context-reference.json` (the same schema) gates
`long_context` on the last 1,024 positions of `long_32k` (32,768 tokens) in
the same three variants, with the floors for a set of 1,000 positions or
more as limits: 0.10 and 0.08.

The 128K check (`long_context_128k`) scores the last 1,024 positions of a
131,072-token prompt against its GPU F32 reference. Its declaration alone
enables the F32-only floor policy; a missing BF16/f64 spread in any other
check must not enable that policy. Its mean-error floor is 0.10, decisive
and near-tie flip floors are 0.08, and the total flip floor is 0.10. DCP2
additionally requires TP4, BF16 KV and context-parallel size two.
