# Chopinfer bundle architecture

This document owns the ownership boundary and the bundle ABI rules. The
overview of processes, the request path and the crates is
[docs/architecture.md](docs/architecture.md). Independent publishers use the
[external model SDK workflow](docs/external-models.md).

Chopinfer separates the model-neutral Rust serving runtime from installable
model bundles. The host loads one explicitly selected local bundle at process
start through a versioned C ABI. The initial kernel family is SM120. A
model's sources (`bundle`, `compiler`, `contract`) live in its own
repository, built against an exported SDK, or for the in-tree models in
`models/<m>/` (today `mimo26`, `qwen38` and `glm5next`).

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
not a sandbox ([security boundary](docs/security-boundary.md)). A process
selects one bundle until exit. Hot reload, automatic network fetching, and
runtime JIT are outside the initial contract.

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
  state. The request carries the client's whole `chat_template_kwargs`
  object, so the format reads any keyword its template reads.
- `constraint`: the grammar a request's tool choice imposes, as a
  `runtime-grammar` wire document; empty when there is none.
- `parser_open`, `parser_push` and `parser_close`: one parser per request.
  The parser holds the completion's bytes: a push keeps the first `keep`
  bytes of earlier pushes (fewer only when a client stop cuts the
  completion short before the last push) and appends a suffix. It answers
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
  and it checks media counts and sizes.
- Each answer is bounded before it is copied (a refusal and parser state at
  most 1 MiB each).
- Render: media segments name request media parts at most once and in
  order, and the host writes the declared placeholder for each; the prompt
  fits `maximum_prompt_bytes`; the effort is a declared effort name in
  every mode; only a thinking
  prompt may leave reasoning open. The host tokenizes the text with the
  declared tokenizer and prompt special tokens. When the prompt leaves
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
the shared parsers of the opt-in `frontend-format` library
([its README](crates/frontend-format/README.md)). No template runs: the
format object is data the model's code and the shared parsers read.

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

The synthetic helper opts in when `config.json` contains
`scratch_multiplier` (1 through 4). Its bundle sizes its token input buffer
at create and refuses unknown options or values outside those compiled
limits. Defaults for this CPU fixture are not qualified. Legacy fixture
packages and default launch journals keep their existing bytes.

Qwen3.8 does not yet declare `indexer_budget`: the shipped contract pins
`TOP_BLOCKS = 512`, and QSA selection and attention instantiate that keep
count in their compiled geometry. Its config default of 2048 tokens means
512 blocks of four. Changing buffer sizes alone would not change selection.
Supporting another budget needs corresponding compiled kernel variants,
resource and graph planning, and separate numerical qualification. The
mechanism therefore ships with the fixture only; Qwen's current kernels,
locks, journals and snapshots are preserved.

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
sees the mapping. Identities, host-placed and host-mapped views and
streaming are in
[direct weight loading](docs/direct-loading.md).

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
through [quantized checkpoint support](docs/quantized-checkpoints.md). A new
model needs a new bundle and profile, not a shared model-specific switch.

## Qualification

Every deployable profile needs authenticated inputs, exact operation
contracts, numerical checks, bounded resources, transaction and cancellation
checks, real-device correctness, and serving evidence. CPU fixtures do not
qualify a model or GPU kernel ([docs/testing.md](docs/testing.md)).
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
