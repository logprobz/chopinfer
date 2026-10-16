# Chopinfer

Chopinfer is a Rust LLM serving runtime for NVIDIA RTX PRO 6000 Blackwell
(SM120) GPUs. A model-neutral host serves installable model bundles behind a
versioned C ABI, on a shared SM120 kernel base, with an OpenAI-compatible HTTP
API.

**Alpha developer preview.** Expect bugs and breaking changes to formats, flags,
packages and the bundle ABI. Please
[report problems](docs/developer-preview.md#report-a-problem).

## Why use it

- **Accuracy is checked, not assumed.** Every kernel op has a declared
  numerical contract and an f64 oracle (BF16 projection activations by
  default, F32 accumulation in a fixed order, no TF32), and each model is
  judged against its own Hugging Face reference run in F32, never another
  engine. In the 2026-10-14 to 2026-10-15 requalification, GLM 5.3 Flash's
  default profile sat 0.061 nats of mean absolute logprob error from that
  reference over 3,781 positions, and MiMo and Qwen scored 194 of 200 on
  their GSM8K screens.
- **One bundle per model** over a shared host that already does scheduling,
  sampling, the OpenAI API, prefix caching, preemption, KV offload,
  speculative decoding and context parallelism.
- **Model-defined KV pages:** opaque bytes the bundle lays out (GLM's FP8
  latent codes, scales and indexer keys), so caching and offload need no
  per-format host code.
- **Evidence an agent can run:** one command checks a model against its
  pinned HF F32 references; serving smokes and A/B runs compare outputs bit
  for bit; journal proofs bound how a change alters kernel launches.
- **Supported:** MiMo V2.6, Qwen3.8 and GLM 5.3 Flash on RTX PRO 6000
  Blackwell 96 GB GPUs ([hardware](docs/performance.md#hardware)). GLM went
  from its first commit (2026-10-06) to a release candidate passing its
  default configuration's gates (2026-10-07).
- **Limits:** chat templates and tool-call formats are per-model code or
  declarations, and tokenizers are limited to the policies the shared reader
  implements; [experimental profiles](docs/serving-profiles.md#profile-status);
  MiMo DFlash's [accuracy waiver](docs/serving-profiles.md#accuracy-waiver-2026-10-06);
  no reusable host prefix cache; outputs under concurrent load
  ([reproducibility](#reproducibility)).

## What it runs

| | |
|---|---|
| Models | MiMo V2.6 Flash (default `XiaomiMiMo/MiMo-V2.6-Flash-MOPD`), Qwen3.8 Flash (FP8, plus [mixed NVFP4](docs/quantized-checkpoints.md)) and GLM 5.3 Flash at TP4 (FP8, plus two NVFP4 exports; [status](docs/model-glm5next.md)) |
| Hardware | Linux x86_64, two or four RTX PRO 6000 Blackwell 96 GB GPUs per model (GLM 5.3 Flash needs four); other SM120 parts are untested and may be refused ([tested hardware](docs/performance.md#hardware)) |
| Topologies | TP4 and TP2; TP2 profiles are experimental ([profile status](docs/serving-profiles.md#profile-status)) |
| Inputs | Text and image by default; MiMo video and audio with `--modalities all` |

`chopinfer serve --model DIR` serves a local Hugging Face checkpoint by
loading its safetensors weights directly, and `chopinfer verify --model DIR`
checks its files against Hugging Face's recorded digests. Weights and a
prebuilt release image are not included. Read the
[alpha limits](docs/serving-profiles.md#alpha-limits) before testing.

As of `v0.1.0-alpha.2` (2026-10-03), all seven configurations passed
validation on 4x RTX PRO 6000 Blackwell Max-Q, from a Docker image built
from a fresh clone: MiMo `tp4-dflash`, `tp4-ar`, `tp2-ar`; Qwen `tp4-mtp`,
`tp4-ar`, `tp2-mtp`, `tp2-ar`. Checks covered GPU-reference teacher-forced
comparisons, API and structured outputs, soak and media. GSM8K scores were
193-195/200 for MiMo and 195-197/200 for Qwen. Fixed C1 speculative outputs
and logprobs were bit-identical to AR at the same rank count; Qwen TP4 also
matched its pinned greedy baseline. See the
[measured speed snapshot](docs/serving-profiles.md#measured-speed).
GLM 5.3 Flash joined the release after its full requalification
(2026-10-14 to 2026-10-15). Of its seven configurations, four pass
(default, k5, nvfp4 and dcp2). nvfp4-nvidia passes with two paired-teacher
rows the maintainers accepted: its pooled gap 0.0056 and chat 0.0158
exceed the 0.005 and 0.015 margins, reproducing exactly a shift accepted
on 2026-10-14, together with a passing paired GSM8K. taps and dcp4 each
have one maintainer-accepted row (a near-tie routing flip in the layer
taps and a KV offload pressure indicator) and pass every other check.
None is a waiver
([release qualification](docs/model-glm5next.md#release-qualification-stage-5)).
GLM's BF16 latent KV (`--kv-cache-dtype bf16` on `tp4-mtp`) was
requalified on 2026-10-16 ([BF16 KV](docs/model-glm5next.md#bf16-kv)).

Each serving profile declares its own status, separate from that release
validation. Today MiMo `tp4-ar` and `tp4-dflash`, Qwen `tp4-ar` and
`tp4-mtp`, and GLM `tp4-mtp`, `tp4-ar`, its DCP 2 and 4 profiles and its
two NVFP4 profiles (through 32 sequences) are qualified; GLM `tp4-mtp-k5`
and every TP2 profile are experimental. The
[profile status table](docs/serving-profiles.md#profile-status) lists every
profile with its status and sequence limits.

### Reproducibility

A request served alone is reproducible bit for bit, with or without
speculation, when it is greedy (temperature 0) or repeats the same explicit
`seed` and sampling parameters on the same configuration. A sampled request
without a seed draws a fresh one, and MiMo and Qwen sample at temperature
1.0 by default. Under concurrent load, outputs can differ from run to run
even then: the number of requests sharing a decode step selects its batch
bucket, and the bucket can change the order of F32 summation, which shifts
logprobs and can change a greedy token. Against the same requests served
one at a time, 16 concurrent requests moved likely tokens' logprobs by a
median of 0.001 to 0.004 nats (99th percentile about 0.2, at most 0.54), and
the first differing tokens followed top-2 gaps of up to 0.54 nats. Repeated
runs of Qwen3.8 and MiMo V2.6 at 4 to 32 concurrent requests differed in
some requests; GLM 5.3 Flash matched in every repeated eight-request run
measured. The differences stay within the accuracy gates for batched decode
([batch shape](docs/numerics.md#batch-shape)).

## Build and check

Users build, verify a checkpoint and serve it. Start with the
[quickstart](docs/developer-preview.md): Docker is recommended, with a native
build-from-tag path for Linux x86_64, Rust 1.89.0 and CUDA 13.2.86.

`bash ci/cpu.sh` runs every CPU check. It needs no GPU and does not qualify
GPU execution. On a 48-core
machine (2026-10-05) its host memory use peaked near 35 GB and it took about
6 minutes with a warm target and 11 with a cold one. See [contributing](docs/contributing.md)
for what a contributor runs and what the maintainers run, and
[performance](docs/performance.md) to measure, profile and submit a speed
change.

## Next steps

- [Quickstart: checkpoint to first response](docs/developer-preview.md)
- [Documentation index](docs/README.md) and [command reference](docs/cli.md)
- [Bundle architecture](BUNDLE_ARCHITECTURE.md) and [adding a model](docs/adding-a-model.md)
- [Performance](docs/performance.md) and the [engine rules](docs/numerics.md#engine-rules)
- [Contributing](.github/CONTRIBUTING.md), [security](SECURITY.md),
  [code of conduct](.github/CODE_OF_CONDUCT.md), [Apache-2.0 license](LICENSE)
