# mimo26

`compiler/` admits checkpoints and emits packages; `contract/` owns geometry,
precision, topology and frontend declarations; `bundle/` owns native execution.
Shared runtime code stays model-neutral.

MiMo is outside the release scope: `model.toml` disables its release, and it
keeps its CPU tests with no further GPU qualification.

Images may be PNG, JPEG, GIF (the first frame) or WebP (lossy, lossless,
with alpha, or the first frame of an animation), decoded as transformers'
`load_image` reads them: oriented by their EXIF or XMP orientation, alpha
discarded. [`fixtures/mimo26/image-decode.json`](../../fixtures/mimo26/image-decode.json)
pins that decode. See [the request surface](../../docs/serving-request-surface.md#media-inputs).

- [Model inventory](model.toml)
- [Serving profiles](profiles.json)
- [Qualification declaration](qual.json)
- [Checkpoint pins](checkpoint-files.json)
- [Model integration](../../BUNDLE_ARCHITECTURE.md#adding-a-model)

## Request defaults

A chat or Responses request without `max_tokens`, `max_completion_tokens`
or `max_output_tokens` generates at most 65,536 tokens, thinking or not,
lowered to the context left after the prompt. An explicit limit is
honored. See
[the request surface](../../docs/serving-request-surface.md#request-parameters-and-defaults).

## Serving profiles

The profile file owns settings, status and qualified concurrency. Read the
[qualification scope](../../docs/developer-preview.md#qualification-scope)
before treating a compatible checkpoint or explicit override as qualified.

## Qualification

`qual.json` owns the exact checks, thresholds and inputs. The pinned
checkpoint is MiMo V2.6 Flash MOPD; runs on other checkpoints do not
qualify its tensors. `tp4-ar` and `tp4-dflash` are qualified through 64
sequences with BF16 KV and text and image inputs. `tp4-dflash`'s pooled
speculative logprob mean, over four draws against the 0.005-nat margin, is
waived by a maintainer decision; its paired and absolute teacher gates, each
draw's top-one bound and coverage passed. The waiver covers only that
pooled mean, not the single-record predicate in `qual.json`, and must be
declared there when the pooled gate is declared; the speculative
continuation gate adds no paired margin and leaves it intact (see [the
contract](../../docs/numerics.md#speculative-continuations)). `tp4-mtp` and
`tp2-ar` remain experimental; the `tp2-ar` configuration runs its checks at
two ranks without speculation. Audio is served but not qualified: there is
no end-to-end audio reference input.

### Reference-code references

Accuracy is judged against the reference model code, not against vLLM. The
teacher and media gates score against runs of the vendor's
`modeling_mimo_v2.py` on the MOPD checkpoint on the GPU reference runner
(`tools/refrun`), with the adapter `reference.py`. Each reference carries a
`binding` block: its refset, every cache entry's key and the sub-digests of
the checkpoint lock, upstream code, adapter and runner files, mode policies
and environment. The gates verify it against the working tree, the workload
beside it (`<stem>.workload.json`) and, when present, the reference cache.

Variants, in the runner's score layout (logprob, top-1, logit margin and the
top 5 IDs, logits and logprobs):

- `exact` (the gate variant): the runner's `f32` mode, each exact
  dequantized product rounded once to F32 (`--dequant f32=rounded`, since
  MiMo's FP8 block scales are F32), F32 log-softmax;
- `bf16`: upstream BF16 loading and BF16 activations and encoders, F32
  log-softmax;
- `f64`: exact dequantized products, f64 activations and log-softmax.

`bf16` and `f64`, scored against `exact` like a server, are the spread that
sets each gate's limits (`tools/qual/limits.py`).
`fixtures/mimo26/teacher-exact-reference.json` scores the 33 texts of
`teacher-texts.json` (4,108 positions), and
`fixtures/mimo26/multimodal/exact-reference.json` each frozen media answer
forced after its prompt and processor output.
`fixtures/mimo26/capture/runner_references.py` writes their workloads and
skeletons for a reference run. After regenerating either reference, update
its pins in `qual.json` in the same change.
