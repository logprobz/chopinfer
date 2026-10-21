# mimo26

`compiler/` admits checkpoints and emits packages; `contract/` owns geometry,
precision, topology and frontend declarations; `bundle/` owns native execution.
Shared runtime code stays model-neutral.

Images may be PNG, JPEG, GIF (the first frame) or WebP (lossy, lossless,
with alpha, or the first frame of an animation), decoded as transformers'
`load_image` reads them: oriented by their EXIF or XMP orientation, alpha
discarded. [`fixtures/mimo26/image-decode.json`](../../fixtures/mimo26/image-decode.json)
pins that decode. See [the request surface](../../docs/serving-request-surface.md#media-inputs).

- [Model inventory](model.toml)
- [Serving profiles](profiles.json)
- [Qualification declaration](qual.json)
- [Checkpoint pins](checkpoint-files.json)
- [Model integration](../../docs/adding-a-model.md)

## Serving profiles

The profile file owns settings, status and qualified concurrency. Read the
[qualification scope and exceptions](../../docs/serving-profiles.md) before
treating a compatible checkpoint or explicit override as qualified.
