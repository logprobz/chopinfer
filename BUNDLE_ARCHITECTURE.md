# Chopinfer bundle architecture

This document states the ownership boundary and bundle ABI rules.
[Adding a model](#adding-a-model) covers in-tree and independent SDK
workflows.

Chopinfer separates the model-neutral Rust serving runtime from installable
model bundles. The host loads one explicitly selected local bundle at process
start through a versioned C ABI. The initial kernel family is SM120. A
model's sources (`bundle`, `compiler`, `contract`) live in its own
repository, built against an exported SDK, or for the in-tree models in
`models/<m>/` (today `mimo26`, `qwen38`, `glm5next` and `deepseek41`).

## Ownership boundary

| Shared runtime owns | Model bundle owns |
|---|---|
| HTTP transport, streaming, cancellation, health, and metrics | Model declaration, tokenizer policy, and prompt assets |
| Tokenization, UTF-8, request limits, grammar masks, and checks on frontend answers | Chat format: prompt rendering, tool-call constraints, and completion parsing |
| Admission, batching, request lifecycle, and backpressure | Prefill and decode execution plans |
| Logical KV accounting, transaction ordering, rollback, and retirement | Physical KV and persistent-state requirements |
| Sampling, token history, stop conditions, and output events | Model arithmetic, fusion, and launch order |
| CUDA contexts, memory, streams, events, and module loading | Weight plans and native modules |
| Generic copies, peer access, and collective transport | Packing, placement, quantization, and topology |
| ABI and package validation, resource isolation, and recovery | Numerical contracts and qualified profiles |

The engine supplies scheduled items and prepared KV plans. A bundle returns
ordered logits rows. The engine validates those rows, samples tokens, commits
the batch, and publishes events. The bundle does not own HTTP or admission.
The shared runtime does not own model layers, tensor names, model arithmetic,
or chat formats.

Shared `runtime-*` crates link no model or bundle crate. Bundles use the
versioned ABI and model-neutral contracts, and never link `runtime-cuda`.
The host alone owns CUDA driver resources. `tests/bundle-boundary.sh` and
`tests/model-neutral-host.sh` reject model dependencies and model vocabulary
in shared crates.

## Bundle ABI

Ledger validation refuses a commit without changing bundle state. A successful
commit leaves the ledger, drafter bookkeeping and device history consistent.
A host failure during history/checkpoint copies, retained-row uploads/copies,
ring-tail restoration or diagnostic finalization can leave partial state.
`BundleInstanceV1` treats every failed commit or cancel callback as terminal:
it poisons the instance and refuses further lifecycle and prefix-state calls,
including destroy. Raw-callback validation tests do not establish retry support
for that client or recovery after arbitrary host failures.

The C ABI (`include/chop_bundle_v1.h`, `runtime-bundle-abi`) carries
fixed-width records, bounded byte views, and host-issued opaque handles. Rust
traits, Rust-owned values, compiler-layout-dependent types, and persistent
device addresses do not cross it. The host resolves addresses, loads modules,
and launches declared operations. A bundle may request bounded memory,
stream, event, copy, collective, and state services.

The entry table has `create`, `execute`, `commit`, `cancel` and `destroy`,
plus optional tails for request retirement, speculation
(`commit_accepted_v1`) and prefix-cache state (describe, capture and
`restore_prefix_state_v1`). A bundle exports it with `export_bundle_api_v1!`.
Optional behavior is declared as manifest capabilities, for example
`chop.serving-kv.v1`, `chop.prompt-rows.v2`, `chop.speculative.v1`,
`chop.prefix-cache.v1`, `chop.media-inputs.v1`, `chop.recurrent-state.v1`,
`chop.weight-plan.v1`, `chop.model-options.v1` and `chop.frontend.v1`
(required to serve chat, see [frontend export](#frontend-export)); the host
enables a feature
only when the manifest
declares it, and unknown versions and unsupported operation contracts fail
closed.

A package is a directory with a canonical `manifest.json`
(`chop-model-bundle-v1`), a closed file set, the native module, the frontend
declaration (`frontend.json`, schema `chop-frontend-openai-v2`), tokenizer
files and bundle metadata. The frontend declaration holds model-neutral
sections the host reads (models, context, limits, vocabulary, generation
defaults, the tokenizer's files and component policy, reasoning controls and
end markers, the tool-name byte set and argument shape, multimodal inputs and
the response's reasoning field) and an opaque `format` object of at most
192 KiB, which the host passes to the bundle's frontend unread. `model-compiler-sdk` emits it and `runtime-bundle` verifies it; the
host copies the verified library to a private image before `dlopen`. The
package also carries the model's serving profiles (the `serve-profiles`
component, from which `serve --bundle DIR --profile NAME` resolves `NAME`)
and its deployment policy (`chopinfer-deployment-policy-v1`, which the
image entrypoint validates its environment against); neither the host nor
the image keeps a per-model copy.

An installed native bundle is trusted in-process code. Verification checks
manifest structure, the closed file set, identities, and ABI version; it is
not a sandbox ([security boundary](SECURITY.md)). A process
selects one bundle until exit. Hot reload, automatic network fetching, and
runtime JIT are outside the initial contract.

The optional `chop_bundle_get_mapped_device_results_api_v1` export adds a
separate sink for mapped sampler input. It leaves existing C tables unchanged.
A request supplies a bounded computed-logits view and device U32 indices for
logical sampling rows. Each logical row keeps its host sampling metadata,
seed, generated index, stop policy and output position; repeated indices
repeat distributions, not random counters. The sampler validates source and
map ownership, retains both buffers through completion, and rejects invalid
map entries through failed sample records without reading beyond the source.
The bundle must check mapped-sampling support before requesting this path.
Hosts that support it advertise the `chop.mapped-device-results.v1`
capability, and a package whose topology depends on mapped sampling requires
it, so a host without it refuses the package at admission.
Ordinary packages and ordinary sampling continue through their existing APIs.

## Frontend export

A bundle owns its chat format. Its native module exports
`chop_bundle_frontend_api_v1` under capability `chop.frontend.v1`, and every
topology of a served package requires it. The host checks the capability
before any native code runs and then takes the export from the module it
loaded; it refuses to serve a package without the capability or a module
without the export, with no fallback. The export is a table of entries over
byte documents:

- `open` and `close`, over the authenticated frontend component bytes. The
  open answer says what the format reads of a request beyond the common
  fields (whole tool function objects, and when assistant history turns
  fall back to `reasoning_content`), so the transport carries it. It also
  lists any extra message roles the format renders besides system,
  developer, user, assistant and tool: at most 16 distinct names of ASCII
  letters, digits, `_` and `-`, each at most 64 bytes. The host refuses an
  open answer outside those bounds, and the transport then accepts a text
  message of each declared role and passes the role by name. It also says
  whether the format renders `tool_reference` content parts in tool
  messages (`{"type": "tool_reference", "name": ...}`, or `tool_name` for
  `name`); only then does the transport accept them there, and it passes
  each by its name. Otherwise the transport refuses them.
- `render`: a chat request in, the prompt out as text segments and
  references to the request's media parts, with the prompt mode, the
  reasoning effort, whether the prompt leaves reasoning open, and parser
  state. A format whose declaration sets
  `vocabulary.explicit_special_tokens` also answers `Special` segments,
  each one special token of the prompt given by its text: the special
  tokens its template writes, while client text stays in text segments.
  The request carries the client's whole `chat_template_kwargs`
  object, so the format reads any keyword its template reads.
- `constraint`: the grammar a request's tool choice imposes, as a
  `runtime-grammar` wire document; empty when there is none. A grammar may
  name a vocabulary token by its ID (a token element, which only that token
  fills and which makes the document layout 2), as a tool grammar writes
  the declaration's ID-only markers; every other grammar stays layout 1.
- `parser_open`, `parser_push` and `parser_close`: one parser per request.
  The parser holds the completion's bytes: a push keeps the first `keep`
  bytes of earlier pushes (fewer only when a client stop cuts the
  completion short before the last push) and appends a suffix. When the
  declaration names `vocabulary.id_only_markers`, every push carries marks
  (a marked push document under its own magic): the ascending, disjoint
  byte ranges of the tokens generated as those markers, which the parser
  reads only there, so their spelling in ordinary tokens is text. It answers
  with events and a cursor. Reasoning, content and tool calls carry ranges,
  which are raw-byte offsets into the whole accumulated completion, not
  into the suffix; tool-call argument deltas and the finish carry values.

Entries run on the CPU and use no host service. `render`, `constraint` and
`parser_open` may run concurrently on one frontend from any thread. A parser
is used by one thread at a time and takes no push after its final push or a
failure. A frontend closes only after every parser opened on it has closed,
and each frontend and parser handle keeps the native module mapped until it
closes. The export catches panics and reports them as the call's failure.

The host checks every answer and treats a bundle frontend as fallible:

- Before render it refuses client text that contains a media placeholder,
  unless the declaration sets explicit special tokens, under which such
  text is plain text, and it checks media counts and sizes.
- Each answer is bounded before it is copied (a refusal and parser state at
  most 1 MiB each).
- Render: media segments name request media parts at most once and in
  order, and the host writes the declared placeholder for each; the prompt
  fits `maximum_prompt_bytes`; the effort is a declared effort name in
  every mode; only a thinking
  prompt may leave reasoning open. The host tokenizes the text with the
  declared tokenizer and prompt special tokens. Under explicit special
  tokens it instead encodes each run of text segments as plain text,
  without the added vocabulary, so a special token spelled in client text
  is that text's ordinary tokens, and each `Special` segment, which must
  spell a declared boundary token, as that token's ID. When the prompt leaves
  reasoning open, the decoder starts in reasoning and every constraint
  waits for one of the declared `reasoning.end_markers`.
- Constraint: the wire document decodes under structural checks and a
  compile-depth bound, and compiles under the host's grammar budgets.
- Parser: it receives only stop-filtered bytes that end on a complete UTF-8
  character, without the bytes of special tokens when the request skips
  them (the chat default), so markers the parser reads must be non-special
  tokens. Ranges must increase, stay disjoint and lie within those bytes;
  the cursor never moves back, the host scans client stops from it, and
  neither it nor a range end may fall inside a UTF-8 character or ill-formed
  byte run; reasoning is allowed only in thinking mode; tool calls only when
  the request extracts them, with declared names, JSON-object arguments, at
  most `maximum_tool_calls`, and deltas that add up to the delivered
  arguments; the finish comes once, at the end of the bytes with the cursor
  there, and no later than the final push. Any violation ends the request
  with a malformed-completion error. The transport still checks each tool
  call against the declared argument shape and limits.

The header (`include/chop_bundle_v1.h`, the chat frontend section) fixes the
table, the sink, flags, answer limits and the thread and lifetime rules;
`runtime_bundle_abi::frontend_document` fixes each document's fields. A
bundle implements `runtime_bundle_abi::frontend::FrontendV1` and exports it
with `export_frontend_v1!`, or, as every in-tree bundle does, writes its
model's renderer in Rust as a `frontend_format::ChatFormatV1` in its contract
crate and exports it with `frontend_format::export_chat_format_v1!(Format)`.
That export renders through the model's code and parses completions and
builds tool constraints from the declaration's `format.parser` section with
the shared parsers of the opt-in `frontend-format` library. No template
runs: the format object is data the model's code and the shared parsers
read.

## Model options

`chop.model-options.v1` declares server-wide checkpoint options. It extends
the V1 manifest with an optional `model_options` member:

```json
{
  "schema": "chop.model-options.v1",
  "options": [{
    "name": "scratch_multiplier",
    "definition": {"type": "integer", "default": 1, "minimum": 1, "maximum": 4},
    "default_is_qualified": false
  }]
}
```

The compiler reads each default from the checkpoint's `config.json`. Names
are unique and sorted, and every topology requires the capability. Types
are integer (signed 64-bit inclusive bounds), number (finite inclusive
bounds), boolean, and string (a sorted list of allowed values).
`model-compiler-sdk::BundleDraftV1::with_model_options` attaches the
capability; apply it to each topology draft before joining them. The host
checks names, types and ranges and treats their meaning as bundle-owned.
Declarations without the capability fail validation. Unsupported declaration
schemas fail closed.

Packages without this member retain their canonical manifest bytes and
receive the same create record. A requiring package receives, after the
optional served-items and served-context fields and before the optional tap
producer, the ASCII tag `chop.model-options.v1`, a little-endian u32 length,
and a JSON object of every resolved typed value in name order. The payload
is bounded to 64 KiB and 64 options. The C ABI tables and symbols remain V1.
An older host refuses the required capability rather than dropping values.
`runtime-bundle-sdk::model_options::read` reads this tail; the bundle must
check the complete name set, types and compiled limits before allocation
and graph capture, and size its resources from those values.

Startup identity, model metadata and metrics expose the effective values.
Qualification accepts only defaults marked qualified; any non-default value
is unqualified. Existing identities and receipts with no option declaration
retain their admission behavior. Passing an option range is compatibility
validation, not accuracy or performance evidence. Non-default execution is
left to the operator; CPU tests cover host validation and refusal paths.

## Direct weight loading

Packages carry no weights. A bundle exports `chop_bundle_weight_plan_v1`
(capability `chop.weight-plan.v1`). Given the checkpoint's safetensors headers
and small documents, the package's bundle metadata, one image role, the rank
count and one rank, it returns that image's plan for the rank: the arena size
and alignment, the named views, the rank's authenticated components (such as
a rank index), and one record per placed piece. A record names the checkpoint
tensors it reads, a selection along one axis, the stored shape, a transform
and the arena offset. A bundle may decline an image; the load then refuses
the package, since a planned component has no other source.

Transforms come from `kernel-core`'s closed set in `weight_transform`, and
the host executes records with no model code. A transform the set cannot
express may use a bundle-owned identifier, which the host runs through the
bundle's `chop_bundle_weight_transform_v1` export. A plan may also map a
view read-only from the checkpoint instead of uploading it
(`chop.host-mapped-gather.v1`); the bundle then has the host gather rows
of it into a device buffer through an optional services tail, and never
sees the mapping. The host validates and streams those plans through
`runtime-weights`.

A plan may also store a section in a smaller lossless encoding whose size
depends on the data. Headers cannot size it, so the bundle exports
`chop_bundle_weight_census_v1`, an optional export under the same
capability. The census names the raw sections it may encode. The host
measures them on the CPU and returns the counts, with the census digest, in
the rank's plan request, and the plan then sizes the encoded sections from
those counts. The host stores the encoding of each validated raw section and
refuses counts that no longer match the data. It reuses counts it measured
for the same census over the same, unchanged checkpoint files
([weight-loading counters](docs/operations-observability.md#weight-loading-counters)).
A request without counts plans every section raw, and so does a host started
with `CHOPINFER_WEIGHT_CENSUS=0`.
The only encoding today is `scale-row-offset-v1` for one-byte block scales
(`kernel_core::precision::scale_offset`); the bundle expands an encoded bank
into its raw layout on the device before the bank's consumers read it.

## Kernel ownership and reuse

Operation resolution matches the complete semantic contract. Provider order
is bundle implementation, host-optimized implementation, then correct generic
implementation. There is no fuzzy matching, implicit adaptation, or
unqualified fallback. A missing exact implementation is unsupported.

Only generic operations (copies, simple conversions, resource operations,
collective transport) belong in the host catalog. Bundles consume `kernels-sm120` instantiations built by `kernel-build`.
Kernel sources are pinned build-time inputs: each bundle records compiler
identity, build recipe, source-closure digest and cubin identity in its own
`kernels.lock`, and a build report records source membership and model-owned
declarations. Updating one bundle must not silently replace another bundle's
implementation. `kernel-core` supplies typed contracts and CPU oracles;
`kernel-harness` is the model-free GPU test harness a model's own kernel
tests run on, and `kernel-lab` holds the in-tree models' GPU probes,
correctness tests, and benchmarks. See
[kernel-library.md](docs/kernel-library.md).

One kernel base exists per NVIDIA ISA family. A bundle selects only shapes,
formats, and capabilities it qualifies. A profile is immutable and names its
architecture, topology, artifact identities, exact kernels, and serving
limits. Compatible quantizations of one family reuse the same implementation
through [quantized checkpoint support](docs/kernel-library.md#activation-policy). A new
model needs a new bundle and profile, not a shared model-specific switch.

## Adding a model

The model owns its compiler, contract, native bundle, frontend and
qualification; the host owns transport, scheduling and device services. An
in-tree model lives under `models/<model>/{compiler,contract,bundle}`; an
independent model uses the same roles against a pinned SDK checkout. The
executable example in `examples/external-model/` holds the export, lock,
build, package and exercise commands.

`model.toml` declares Cargo role targets, optional release selection,
contributor fixtures and check inputs; `tools/model_sdk/inventory.py`
validates it. A Cargo target ships only when an enabled release declaration
selects it. `publisher` restricts the checkpoint names a model commits to
its author; other checkpoints use
[external qualification variants](docs/check-kinds.md#external-variants). Keep
geometry, projections, encoders, tuning, frontend assets, serving profiles,
deployment policy, kernel locks and qualification declarations with the
model. Small test inputs belong in fixtures; weights and captured evidence
stay outside the tree.

### External SDK workflow

Pin the SDK's full revision and content identity in `sdk-pin.json`,
reconcile and commit Cargo.lock, and keep a real nested `sdk/` checkout.
The `tools.model_sdk` workflow checks role dependency closures, locks,
exports, compatibility matrices and test execution. A model's kernel tests
use `kernel-harness` in its own test crate; new models do not register in
`kernel-lab`. Compatibility has four parts: Rust SDK source, native ABI,
package and frontend schemas, and host capabilities. Pin and test all four;
matching C symbols alone does not promise compatibility across releases.
Test independently built bundles against the intended prebuilt hosts.

### Model implementation

The compiler implements `model_compiler_sdk::preparation::helper::Provider`:
`probe` routes metadata and `emit_direct` validates the checkpoint and emits
a canonical package with planned weight components. The installed
`chopinfer-model-<model>` helper is discovered beside the host executable.
Package emission takes an explicitly built native module; it does not build
one. Each model has one bundle ID and one package containing every
supported rank topology and its resources.

Declare tensor names, shapes, split rules and precision in the model's
projection specification (`projection-spec`). Derive rank-local views from
the checkpoint, preserving scale-group boundaries and source scale origins.
Layout admission, payload validation and quality qualification are separate
checks; a format an engine supports does not qualify every activation policy
or checkpoint that uses it.

Generic model code requests math through `&dyn Backend` and model-owned
`&dyn OwnOps`. Target modules register compiled entries and prepare them at
instance creation; selection belongs to the backend, and unsupported
operations refuse at prepare. Follow the
[kernel contribution rules](docs/kernel-library.md#adding-an-op): required
conformance cases must prepare and run, and no refused or unexercised case
counts as a pass.

### Chat frontend rules

Implement the renderer in model-owned Rust with `frontend-format`, or
implement the native frontend contract ([frontend export](#frontend-export)).
Checkpoint chat templates are reference inputs, not executed server code.
Declare `output_tokens` to bound the chat and Responses requests that set no
output limit, and where a geometry lowers the context, lower those lengths
with `runtime_frontend_contract::lower_output_tokens_v1`.

Port the upstream template or encoder literally, citing source lines. Keep
HF render, tokenization and streaming vectors as goldens and run
`chopinfer bundle check-frontend` on the built package, covering reasoning,
tools, role handling, media and every byte split of streaming output. Use
the HF differential corpus (`python3 -m tools.refrun.frontend_corpus`) to
tell real divergence from intentionally unsupported or lossy template
behavior; a `no-round-trip` declaration must identify the actual union,
delimiter or tokenizer loss. A missing shared tokenizer or media capability
needs a model-neutral contract and tests, and a completion syntax absent
from `frontend-format` needs a reusable parser family. Never regenerate a
golden to hide a mismatch.

A model declares bounded contributor fixtures and exact test selectors in
`model.toml` ([contributor checks](docs/contributing.md#first-success-without-weights-or-cuda)),
and its release gate in `qual.json` with a reference lock and model-owned
reference adapter. A model with a drafter sets its per-concurrency draft
depths with the depth tuner (`tools/tune`, `models/<model>/depth-tune.json`)
and puts the selected schedule in the drafting profile's `depths`.

## Qualification

Every deployable profile needs authenticated inputs, exact operation
contracts, numerical checks, bounded resources, transaction and cancellation
checks, real-device correctness, and serving evidence. CPU fixtures do not
qualify a model or GPU kernel ([docs/contributing.md#checks](docs/contributing.md#checks)).
Unsupported profiles fail closed.

## ABI stability

The bundle ABI is alpha: any Chopinfer release may change the header, the
package format and the bundle entry points incompatibly, so a bundle or
package built against one release may not load in another. Rebuild bundles
and packages for each release. Nothing here is frozen yet.

The V1 header remains a draft. Freezing it requires both
`fixtures/bundle-abi-v1/linux-unchanged-host.json`, an authentic Linux
unchanged-host result, and `fixtures/bundle-abi-v1/sm120-smoke.json`, an
SM120 smoke receipt. Host-memory fixture tests alone do not qualify CUDA.
`crates/runtime-conformance/tests/abi_v1_stability.rs` checks that the header
and this document keep the draft notice until both prerequisites exist, and
drop it once they do.
