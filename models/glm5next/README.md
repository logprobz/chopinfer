# glm5next

`compiler/` admits checkpoints and emits packages; `contract/` owns geometry,
precision, topology and frontend declarations; `bundle/` owns native execution.
Shared runtime code stays model-neutral.

Images may be PNG, JPEG, GIF (the first frame) or WebP (lossy, lossless,
with alpha, or the first frame of an animation), decoded as transformers'
`load_image` reads them: oriented by their EXIF or XMP orientation, alpha
discarded. [`fixtures/glm5next/image-decode.json`](../../fixtures/glm5next/image-decode.json)
pins that decode. See [the request surface](../../docs/serving-request-surface.md#media-inputs).

- [Model inventory](model.toml)
- [Serving profiles](profiles.json)
- [Qualification declaration](qual.json)
- [Checkpoint pins](checkpoint-files.json)
- [Model integration](../../BUNDLE_ARCHITECTURE.md#adding-a-model)

## Request defaults

A chat or Responses request without `max_tokens`, `max_completion_tokens`
or `max_output_tokens` generates at most 131,072 tokens when it thinks at
effort max, the default, and 65,536 at low or high effort, each lowered to
the context left after the prompt. Thinking cannot be switched off. An
explicit limit is honored. See
[the request surface](../../docs/serving-request-surface.md#request-parameters-and-defaults).

## Serving profiles

`tp4-mtp`, `tp4-mtp-dcp2` and `tp4-mtp-dcp4` are the serving profiles;
`tp4-ar` is their baseline and correctness control. The profile file owns
settings, status and qualified concurrency. Read the [qualification
scope](../../docs/developer-preview.md#qualification-scope) before treating
a compatible checkpoint or explicit override as qualified.

GLM serves TP4 only. Three-draft MTP, AR, DCP2 and DCP4 are qualified
through 32 sequences; no profile has C64 qualification. Five-draft MTP
(`tp4-mtp-k5`) is experimental, text-only and never the default. FP8 latent
KV is the default; BF16 latent KV (`tp4-mtp --kv-cache-dtype bf16`) is
qualified by its own configuration, and evidence for FP8 alone does not
establish it.

### Ordered prefill DMA

The official block-FP8 layout with FP8 MTP KV at CP1 uses ordered DMA for
single-request text prefill from 512 rows. It preserves the F32 rank fold
and one BF16 output round. Other configurations keep the collective.

### Coupled sampled drafts

The MTP drafter proposes each sampled item's drafts from the target
sampler's own noise ([coupled sampled drafts](../../docs/numerics.md#coupled-sampled-drafts)).
Draft position `j` of a verify item proposes the argmax of the drafter's
logits divided by the item's temperature plus the Gumbel noise of the
generated index the target samples that position at. Each drafter run
uploads one record per draft row (seed, temperature and that index), so
one captured draft step serves every position, and the draft argmax adds
the noise to each rank's logits shard before its partial. Greedy items
draft what they drafted before. A draft is kept only when it equals the
target's sample at its row, so every emitted token is the target's sample;
GLM's verify rows are not bitwise AR rows, so other drafts (and with them
other verify shapes) can change a sample at a near tie, as any change of
drafts can.

Only an instance that runs the drafter loads the draft noise cubin (the
shared entry, byte-identical to DeepSeek's) and allocates its records.

## Checkpoint layouts

The contract's `direct.rs` admits complete layouts of text and encoder
weights. Header admission and payload checks must agree on tensor shapes,
scales and decode order. Every served projection uses BF16 operands under
the A16 policy; native activation quantization needs its own qualification.
A newly admitted checkpoint does not inherit another checkpoint's evidence.

## Reference runner

`reference.py` supplies the HF adapter. Its F32 gate uses rounded
weight dequantization, including F32 FP8 block scales and NVFP4 decoding.
The indexer sorts scores stably with lower-index ties, preserving selection
across prefill chunks; this deliberately replaces unstable upstream top-k
ordering at ties. Cache keys bind that adapter and its numerical policy.
Long-context captures need an explicit placement that fits their growing
latent/indexer state; use the reference runner's placement preflight.

### Layer reference

`text_layers` holds the served build's taps of one sequence
(`chat_effort_high`, 31 tokens) against HF's F32 run, with a floor taken
from HF's BF16 run with the served FP8 latent round trip
(`reference_latent_fp8.py`, the check's `ADAPTERS`); plain HF BF16 has no
FP8 path, and the `taps` configuration serves the default FP8 latent KV
cache. The error side stays `reference.py`'s plain F32 run, so the check
measures the served build's whole deviation, FP8 storage included, against
the model's F32 math, and allows three times what an HF BF16 implementation
of the same storage format deviates.

Only the floor takes the FP8 latent. The adapter's F32 arm quantizes the
unrounded F32 latent, so its codes are not the served writer's (the writer
quantizes the BF16 latent), and as the error side it would remove the
format's cost from the check. The BF16 arm's codes and scales are the
writer's for HF's BF16 latent. The adapter applies the round trip to the
latent rows `kv_b_proj` expands, so the attention reads the stored format
while the `x.latent` tap keeps the served tap's boundary, the normalized
latent before the FP8 write: a floor tapped after the round trip would
allow storage error at a tap that stores nothing.

The reference (`$CHOPINFER_GLM5NEXT_LAYER_REFERENCE`) is a directory of two
runner runs of the same workload, both with `--taps --tap-sites
x.hc.attn,attn.out,x.hc.mlp,mlp.out,x.latent,x.indexer.ids,router.logits,router.weights,router.ids,norm`:

- `f32/`: `--adapter models/glm5next/reference.py --modes f32 --dequant
  f32=rounded`;
- `bf16/`: `--adapter models/glm5next/reference_latent_fp8.py --modes bf16`.

Each run's cache keys must name its adapter's current digest, and the
reference lock pins every file of both.

### Routing near ties

See [the numerical contract](../../docs/numerics.md#routing-near-ties).

## Qualification checks

`qual.json` owns the exact checks, thresholds and configurations. Absolute
teacher and media limits follow the shared spread rules; the chat check's
predeclared headroom factor is 1.25. Each configuration runs against its own
matching server: default FP8 KV at DCP1, DCP2/DCP4, five-draft text-only,
BF16 KV, or diagnostic taps. Five-draft determinism records omit images.
The declared needle lengths and frozen values must all be covered in order.
There is no DCP4 chat waiver.

The soak accepts the bounded prefill-program cache under an explicit warmup
rule: after 180 seconds, host and GPU growth must each stay within 512 MiB;
whole-soak limits additionally allow 2,560 MiB host and 1,344 MiB GPU. This
is a bounded cache exception, not an unlimited soak allowance. A reviewed
row is not a waiver and changes no limit.

[Layer comparison](#layer-reference) uses F32 reference taps and the BF16
floor with the served FP8 latent round trip, with every required tap
compared and bound to the live producing process. The AR/MTP and [paired
baseline scopes](../../docs/numerics.md#paired-baseline-scope) are
distinct. Diagnostic evidence does not qualify an uncaptured server.
