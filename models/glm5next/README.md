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
- [Model integration](../../docs/adding-a-model.md)

## Request defaults

A chat or Responses request without `max_tokens`, `max_completion_tokens`
or `max_output_tokens` generates at most 131,072 tokens when it thinks at
effort max, the default, and 65,536 at low or high effort, each lowered to
the context left after the prompt, as the owner decided on 2026-10-31.
Thinking cannot be switched off. An explicit limit is honored. See
[the request surface](../../docs/serving-request-surface.md#request-parameters-and-defaults).

## Serving profiles

The profile file owns settings, status and qualified concurrency. Read the
[qualification scope and exceptions](../../docs/serving-profiles.md) before
treating a compatible checkpoint or explicit override as qualified.

### Ordered prefill DMA

The official block-FP8 layout with FP8 MTP KV at CP1 uses ordered DMA for
single-request text prefill from 512 rows. It preserves the F32 rank fold
and one BF16 output round. Other configurations keep the collective.
The [schedule audit and served comparison](journal-changes/prefill-dma.md)
record the scope, bitwise checks and 8K/32K TTFT results.

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

On 2026-10-30 coupling was adopted after an ABBA comparison with the greedy
drafter on `tp4-mtp` (adaptive speculation, K3, prefix cache, FP8 KV),
temperature 1.0, 256 tokens per request, coding and chat prompts, 15 s
closed-loop windows, matched seeds and a fresh process per visit. Request
tok/s rose at every concurrency (mean of the two pairs, spread in
parentheses): C1 +10.7% (0.1%), C2 +11.4% (6.5%), C4 +7.9% (3.3%), C8
+8.8% (3.1%) and the C16 control +11.3% (0.4%). The share of drafts
accepted rose in every cell, at C1 from 0.58 and 0.56 to 0.64 and 0.62.
Accepted drafts per verify item rose at C1 from 1.28 and 1.29 to 1.73 in
both pairs; at higher concurrency adaptive speculation drafts fewer tokens
per verify item when coupled, so that figure is not comparable there (C8:
1.32 and 1.72 greedy, 1.44 and 0.92 coupled). Every matched request
emitted the same tokens in both arms. Fixed-seed GPU checks found sampled
outputs at C1, C8 and C32 (temperature 1.0) and C8 (0.7) token-equal to
AR, and greedy outputs bitwise equal with and without coupling.

### Layer reference

`text_layers` holds the served build's taps of one sequence
(`chat_effort_high`, 31 tokens) against HF's F32 run, with an HF BF16 run's
distance from that run as the floor. The `taps` configuration serves the
default FP8 latent KV cache, and plain HF BF16 has no FP8 path. The check's
first GLM runs (2026-10-14: the TP-split candidate ae8be147c and its base
85893d6f1, bitwise equal through layer 3's router) therefore failed first
at `L03.attn.out`, 3.25 times its plain BF16 floor: every sparse layer's
attention output carries the FP8 latent's storage error. The operator
calibrated the floor to the served format instead of changing a limit
(2026-10-14). The floor is HF's BF16 run with the served FP8 latent round
trip (`reference_latent_fp8.py`, the check's `ADAPTERS`); the error side
stays `reference.py`'s plain F32 run. So the check still measures the
served build's whole deviation, FP8 storage included, against the model's
F32 math, and allows three times what an HF BF16 implementation of the same
storage format deviates. FLOOR_RATIO, RELATIVE_RMS_LIMIT, REQUIRED_TAPS and
the rules are unchanged.

Only the floor takes the FP8 latent. The adapter's F32 arm quantizes the
unrounded F32 latent, so its codes are not the served writer's (the writer
quantizes the BF16 latent), and as the error side it would remove the
format's cost from the check. The BF16 arm's codes and scales are the
writer's for HF's BF16 latent. The adapter applies the round trip to the
latent rows `kv_b_proj` expands, so the attention reads the stored format
while the `x.latent` tap keeps the served tap's boundary, the normalized
latent before the FP8 write: a floor tapped after the round trip would
allow storage error at a tap that stores nothing (at `L03.x.latent` it
would rise 5.0 times). This is the comparison ACC2 (2026-10-08)
measured on the pre-split engine: against HF F32 with this floor every
dense tap of 631 short-text rows was within three times the floor (the
median over taps of each tap's median error over floor 0.999, the worst
1.06), while the BF16 KV engine sat at the plain floor. On the first eight
rows of the check's sequence the TP-split candidate's `L03.attn.out` is at
3.72 times the plain floor and HF BF16 with the FP8 latent at 3.71.

The reference (`$CHOPINFER_GLM5NEXT_LAYER_REFERENCE`) is a directory of two
runner runs of the same workload, both with `--taps --tap-sites
x.hc.attn,attn.out,x.hc.mlp,mlp.out,x.latent,x.indexer.ids,router.logits,router.weights,router.ids,norm`:

- `f32/`: `--adapter models/glm5next/reference.py --modes f32 --dequant
  f32=rounded`;
- `bf16/`: `--adapter models/glm5next/reference_latent_fp8.py --modes bf16`.

Each run's cache keys must name its adapter's current digest, and the
reference lock pins every file of both (670 files). A one-run reference of
`reference.py` no longer binds. The pinned runs are from 2026-10-14 on four
GPUs. Each mode's time below is the runner's wall time for that mode,
weight loading included: f32 62.8 s (44.7 s of it loading the weights) and
bf16 63.7 s (46.9 s loading). The f32 run's taps and scores are bitwise
equal to those of the earlier one-run reference. The bf16 run's taps equal
those of an earlier bf16 run tapped after the round trip, except the 11
`x.latent` taps, each of which that run held round-tripped; `L03.x.latent`,
with nothing upstream of it in FP8, equals plain HF BF16's.


### Routing near ties

See [the numerical contract](../../docs/numerics.md#routing-near-ties).

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

## Qualification checks

`qual.json` owns the exact checks, thresholds and configurations. Absolute
teacher and media limits follow the shared spread rules; the chat check's
predeclared headroom factor is 1.25. Each configuration runs against its own
matching server: default FP8 KV at DCP1, DCP2/DCP4, five-draft text-only,
BF16 KV, or diagnostic taps. Five-draft determinism records omit images.
The declared needle lengths and frozen values must all be covered in order.

The [profile evidence](../../docs/serving-profiles.md#glm-evidence) preserves
soak allowances and reviewed rows. [Layer comparison](#layer-reference) uses
F32 reference taps and the BF16 floor with the served FP8 latent round trip,
with every required tap compared and bound to the live producing process.
The AR/MTP and [paired baseline scopes](../../docs/numerics.md#paired-baseline-scope)
are distinct. Diagnostic evidence does not qualify an uncaptured server.
