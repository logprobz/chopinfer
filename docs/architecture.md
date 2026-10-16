# Architecture

Chopinfer is a Rust and CUDA LLM serving runtime for SM120 GPUs. One process
serves one model: a model-neutral host loads one bundle through a versioned C
ABI, and the bundle runs the model on the host's CUDA services. Three
models serve today: MiMo V2.6 Flash and Qwen3.8 Flash at TP2 and TP4, and
GLM 5.3 Flash at TP4. The examples use TP4 on four RTX PRO 6000 GPUs.

**Hardware.** Chopinfer is tested and supported only on the NVIDIA RTX PRO
6000 Blackwell with 96 GB, and each model needs two or four of them (GLM 5.3
Flash needs four). The runtime requires compute capability 12.0 exactly
(`crates/runtime-cuda/src/lib.rs`), packages plan 94 GiB of device memory
per rank (`crates/model-compiler-sdk/src/device.rs`), and the kernels are
tuned for its 188 SMs. Other SM120 parts are untested and may be refused
([hardware](performance.md#hardware)).

This page is the overview. The ownership boundary, the ABI entries,
capabilities and package format are in
[BUNDLE_ARCHITECTURE.md](../BUNDLE_ARCHITECTURE.md). A model lives in its
own directory, under `models/` or in its own repository against an exported
SDK ([adding-a-model.md](adding-a-model.md),
[external-models.md](external-models.md)).

## The split

| Layer | Owns | Crates |
|---|---|---|
| Host | HTTP, request validation, SSE, stop strings, UTF-8, request limits, tokenization under the declared tokenizer policy, checks on every frontend answer, admission, continuous batching, logical KV pages, recurrent-state slot IDs, prefix-cache registry, sampling, grammar masks, CUDA contexts, memory, streams, graphs and module loading, bundle verification | `runtime-*` |
| Bundle | Model program: which kernels run in which order with which buffers; physical KV and per-request state layout; drafters and encoders; the chat format: prompt rendering, tool-call constraints and completion parsing, exported as `chop.frontend.v1` | `bundle-<model>` (a `cdylib`) |
| Model compiler | Config, tensor catalog, placement per rank, weight plan, package manifest, frontend declaration (`frontend.json`), the serving profiles and deployment policy the package carries, f64 oracle | `model-<model>` |
| Kernels | Op contracts and f64 oracles, CUDA engines, instantiation catalogs, cubin builds, GPU correctness tests | `kernel-core`, `kernels-sm120`, `kernel-build`, `kernel-harness`, `kernel-lab` |

The host never names a model and never reads a chat format. The bundle
never owns HTTP, admission, sampling or stop handling, and never links
`runtime-cuda`: it asks the host for device services through handles. Packages carry no weights; the host fills each rank's arena
from the Hugging Face checkpoint by the bundle's weight plan
([direct-loading.md](direct-loading.md)).

## Processes and devices

- **One process, one bundle.** `chopinfer serve` verifies the selected
  package (`runtime-host/src/install.rs`, `verify_selected_bundle_v1`), then
  loads its native module once for the process lifetime. There is no hot
  reload.
- **One engine thread drives every rank.** The `bundle-engine` thread
  (`runtime-host/src/generation.rs`, `worker_loop`) owns the loaded bundle,
  the engine and the scheduler, and issues all device work for all ranks. The
  Tokio runtime runs HTTP and streaming; blocking work (request parsing, chat
  rendering, tokenization, media preprocessing, grammar compilation) runs on
  Tokio's blocking pool. Video decoding has its own worker pool
  (`runtime-host/src/video.rs`).
- **One device per rank.** The rank count comes from the package topology.
  By default rank r runs on visible CUDA ordinal r; `--devices` picks other
  distinct visible ordinals. Duplicate ordinals are refused.
  See [deploy common](deploy-common.md) for TP2, TP4 and
  side-by-side servers.
- **Collectives are bundle kernels.** The bundles enable peer access
  between every pair of ranks and launch their own PCIe peer-to-peer
  all-reduce and all-gather kernels from `kernels-sm120/cuda/comm/`. The
  host's generic `execute_collective` service (`runtime-host/src/cuda.rs`)
  sums through host memory and is a fallback, not the production path.

## Request path

A streamed `POST /v1/chat/completions` follows these steps:

1. `runtime-server` validates headers, capacity, body and request fields.
2. `runtime-frontend` checks the request (media placeholders, media counts
   and sizes) and passes it to the bundle's frontend export, which renders
   the prompt as text and media segments and answers any tool-call
   constraint as a grammar document. The host checks both answers;
   `runtime-tokenizer` tokenizes the text segments under the declared
   tokenizer policy, `runtime-media` preprocesses the media, and
   `runtime-grammar` compiles the constraint under the host's budgets.
3. A bounded channel submits work to the engine thread. The scheduler
   admits requests, reserves KV pages and state slots and plans a continuous
   batch.
4. `BundleModelExecutorV1` encodes the capability-selected batch wire. The
   bundle reads it through `runtime-bundle-sdk`, packs inputs and runs its
   program eagerly or through bucketed CUDA graphs on host device services.
5. The bundle emits logits. The host samples on device or CPU, applying masks.
6. The engine commits KV and the bundle transaction (accepted rows for
   speculation), or rolls back on failure or cancel. Retirement frees state
   slots.
7. The host decoder turns tokens into bytes, holds back incomplete UTF-8
   and possible stop strings, and applies stops and limits. It pushes the
   stable bytes to the bundle's frontend parser, which answers with byte
   ranges of reasoning, content and tool calls. The host checks the ranges
   and calls, and the server streams the resulting events as SSE.

[serving-request-surface.md](serving-request-surface.md) lists the request
fields; [serving-kv-pool.md](serving-kv-pool.md) and
[serving-recurrent-state.md](serving-recurrent-state.md) describe KV pages and
state slots.

## Crates

Model-neutral host (`runtime-*`):

| Crate | Role |
|---|---|
| `runtime-server` | OpenAI-compatible HTTP transport, request validation, SSE, metrics endpoint |
| `runtime-frontend-contract` | The host-read sections of the frontend declaration (`chop-frontend-openai-v2`), the tokenizer component policy and tokenizer contract; the `format` section stays opaque |
| `runtime-frontend` | The host half of a bundle's frontend: opens the package module's `chop.frontend.v1` export, checks requests and every answer, and decodes incrementally (stop strings, UTF-8, limits). It links no chat format outside its tests |
| `runtime-tokenizer` | Byte-level BPE read under a declared component policy |
| `runtime-grammar` | Constrained decoding: byte automata and token masks |
| `runtime-engine` | Step loop, transactions, speculation policy, prompt-lookup drafter, media arena |
| `runtime-scheduler` | Admission and continuous batching |
| `runtime-kv` | Logical KV pages, prepared appends, prefix registry |
| `runtime-sampling` | Host sampling and log-probabilities |
| `runtime-model-api` | Model vocabulary contract |
| `runtime-weights` | Authenticated safetensors metadata and hashing |
| `runtime-bundle`, `runtime-bundle-abi` | Package manifest verification; the C ABI, loader and trampolines |
| `runtime-bundle-sdk` | Bundle-side helpers: batch wire reader, host errors, stream transactions, rank indexes, diagnostics |
| `runtime-cuda` | Safe CUDA driver API, NVDEC |
| `runtime-kernel-contract`, `runtime-kernels` | Kernel identities and launch contracts; host generic-op catalog and sampling cubins |
| `runtime-media` | CPU image, audio and video decoding and preprocessing |
| `runtime-host` | The `chopinfer` binary: bundle loading, host services, generation worker |
| `runtime-eval` | `chopinfer eval`: qualification declarations, pinned references and check kinds against a running server |
| `runtime-conformance` | ABI conformance tests |

Shared model-side crates:

| Crate | Role |
|---|---|
| `bundle-runtime` | SM120 bundle mechanics: launch dispatch, eager and graph pipelines, input packing, logits and page views, resources ([bundle-runtime.md](bundle-runtime.md)) |
| `model-compiler-sdk` | Package emitter, preparation helper and rank-index schema |
| `frontend-format` | The opt-in library a model's chat format is written with: the toolkit a model's Rust renderer uses, the `format.parser` section, the shared completion parsers and tool-choice grammars of the two XML tool-call families, and the generic export a bundle exports with one line ([README](../crates/frontend-format/README.md)); no host or device code |
| `encoder-spec` | Encoder declarations: typed, versioned components, validation and digests ([README](../crates/encoder-spec/README.md)); no CUDA |
| `kernel-core` | Op contracts, launch plans, codecs, weight transforms and f64 oracles; no CUDA |
| `kernels-sm120` | CUDA engines, typed builders, planners and legacy instantiation catalogs ([kernel-library.md](kernel-library.md)) |
| `kernel-build` | Build-script library: model-owned selections, `nvcc`, cubin audit, locks and source ownership |
| `kernel-harness` | Model-free GPU test harness: sessions, launches, poisoned replays, conformance runs, the conformance kit and encoder conformance; a consumer passes its cubins, catalogs and cases |
| `kernel-lab` | GPU correctness tests, probes and benchmarks of the in-tree models and the lab's own cubins, on `kernel-harness` |

Per model: `model-<m>`, `model-<m>-contract` and `bundle-<m>`, in the
model's own repository or, for the in-tree models, under `models/<m>/`
([serving-mimo26.md](serving-mimo26.md),
[model-qwen38.md](model-qwen38.md), [bundle-qwen38.md](bundle-qwen38.md),
[model-glm5next.md](model-glm5next.md)).
`bundle-tp4-fixture` and `bundle-recurrent-fixture` are synthetic bundles that
test the host boundary without model arithmetic. `xtask` holds workspace
tasks such as `check-portable`.

## Enforced boundaries

`ci/cpu.sh` runs, among its steps:

- `tests/bundle-boundary.sh`: the exact workspace member list and each
  model's `model.toml`; no `runtime-*` crate links a `model-*` or `bundle-*`
  crate (except `runtime-conformance`, which may use the fixtures); bundles
  and `bundle-runtime` do not link `runtime-cuda`, `runtime-engine`,
  `runtime-host` or `runtime-server`; `bundle-runtime` links no model
  or bundle crate, and `kernels-sm120` only for its tests (no production
  source names it); `runtime-frontend` and `runtime-host` link
  `frontend-format` only in tests, since a host serves the package's own
  frontend export; the exact sets of crates that reach `runtime-cuda` or
  compile cubins.
- `tests/model-neutral-host.sh`: no `runtime-*` source mentions a model name
  or other model vocabulary, with no exceptions. Tokenizer policies are
  declared component fields rather than named dialects, so the scan no
  longer strips dialect names (the former `DIALECTS` exemption is gone).
  The other shared crates, `frontend-format` among them, and `tools`, `ci`
  and `deploy` are scanned against the exact counts in
  `tests/model-neutral-allowlist.json`.
- the workspace tests in `cargo run -p xtask -- check-portable`
  (`ci/rust_tests.py`: cargo-nextest and the doctests, or `cargo test
  --workspace` without nextest), which include each bundle's fake-host tests
  and launch journal.

## Contributor invariants

Share a component when a second implementation uses the same model-neutral
contract. Model arithmetic, physical layouts, rounding choices and layer order
stay model-owned; shared code takes explicit geometry and capabilities, never
infers a model from a name or shape. A chat format is model-owned code:
a new format lives in the model's bundle (or reuses `frontend-format`), and
the host reads only the declaration's model-neutral sections. Shared changes
must preserve every bundle's launch journal and frontend goldens unless a
qualified behavior change is intended.

The engine validates all output rows and stages sampling before committing a
batch and publishing events. Failure or cancellation rolls back pending KV
and bundle state before releasing resources; stale handles cannot commit or
release a later allocation. Full shared KV blocks are immutable, and shared
partial tails require copy-on-write, as does a shared full last page when
the bundle declares that its steps rewrite rows there
(`batch_limits.rewrites_last_page`).

Tokenizer assets are authenticated data, never executable model repository
code. Incremental decoding retains incomplete UTF-8 and stop prefixes,
emits only complete UTF-8, and never publishes matched stop bytes. The
bundle's frontend parser sees only those released bytes, and the host fails
a completion as malformed when the parser's ranges, tool calls or finish
break the contract. Unsupported profiles and operation contracts fail closed.

Local inventory does not qualify a topology. Separate replicas require
disjoint device groups, and routing may retry only a clean admission refusal
before a backend accepts ownership or emits an event.
