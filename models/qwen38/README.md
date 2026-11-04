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
- [Model integration](../../docs/adding-a-model.md)

## Request defaults

A chat or Responses request without `max_tokens`, `max_completion_tokens`
or `max_output_tokens` generates at most 131,072 tokens when it thinks at
effort xhigh, the default, and 65,536 at medium or low effort or with
thinking off, each lowered to the context left after the prompt, as the
owner decided on 2026-10-31. An explicit limit is honored. See
[the request surface](../../docs/serving-request-surface.md#request-parameters-and-defaults).

## Serving profiles

`tp4-mtp` is the serving profile; the AR profiles (`tp4-ar`, `tp4-ar-dcp2`,
`tp2-ar`) are its baseline and correctness control. The profile file owns
settings, status and qualified concurrency. Read the [qualification scope and
exceptions](../../docs/serving-profiles.md) before treating a compatible
checkpoint or explicit override as qualified.
