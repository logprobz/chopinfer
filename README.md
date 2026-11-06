# Chopinfer

Chopinfer is a Rust LLM server with an OpenAI-compatible HTTP API.

It targets Linux x86_64 with NVIDIA RTX PRO 6000 Blackwell 96 GB GPUs
(SM120, compute capability 12.0). The tested hardware is the Max-Q variant;
other SM120 GPUs are untested. Qwen uses two or four GPUs; GLM and DeepSeek
require four.

| Model | Qualified profiles | Experimental profiles |
|---|---|---|
| [Qwen3.8 Flash](models/qwen38/profiles.json) | `tp4-mtp` | `tp4-mtp-dcp2`, `tp2-mtp` |
| [GLM 5.3 Flash](models/glm5next/profiles.json) | `tp4-mtp`, `tp4-mtp-dcp2`, `tp4-mtp-dcp4` | `tp4-mtp-k5` |
| [DeepSeek V4.1 Flash](models/deepseek41/profiles.json) | `tp4-dspark` | None |

TP2 and TP4 use two and four GPUs. MTP and DSpark profiles decode
speculatively and emit only target-verified tokens
([numerics](docs/numerics.md#state-and-verification-contract)). DCP2 and DCP4
split attention context across two and four ranks. AR (autoregressive)
profiles such as `tp4-ar`, `tp4-ar-dcp2` and `tp2-ar` are each model's
baseline, correctness control and debugging aid, not serving profiles.
MiMo V2.6 Flash stays in the code with its CPU tests, outside the release
scope.

Qualified profiles have recorded serving evidence for their declared
configuration. That status does not qualify every checkpoint, quantization
or override. The linked profile files hold exact settings and evidence;
[build and serve](docs/developer-preview.md) explains profile selection and
qualification scope.

[Build and serve](docs/developer-preview.md) | [Operate](docs/operations-observability.md) |
[HTTP API](docs/serving-request-surface.md) | [Contribute](docs/contributing.md) |
[Architecture](BUNDLE_ARCHITECTURE.md)
