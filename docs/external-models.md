# Models maintained outside core

A model publisher can maintain its compiler, bundle, contract, frontend and
release schedule in an independent workspace. An existing host verifies and
loads its canonical package without a model registry entry or runtime patch.
The hands-on walkthrough, with every command, is the maintained
[external example](../examples/external-model/README.md); this page states the
contracts and policy. The supported workflow uses pinned SDK source checkouts
on Linux x86_64.

## The workflow

A model in its own repository edits no file of this one:

1. Export the SDK at a pinned revision with the template, the
   [external example](../examples/external-model/README.md), as the starting
   point ([delivery stages](#delivery-stages)). Exporting an in-tree model
   ([exporting an in-tree model](#exporting-an-in-tree-model)) is a migration
   helper for moving that model, not a starting point for a new one. Run
   `lock` once and `lock --update` after changing the model's own
   dependencies; commit `Cargo.lock` and `sdk-pin.json`.
2. Write the contract, compiler and bundle crates against the
   [supported source surface](#supported-source-surface).
3. Keep the model's own kernels, selections and kernel locks in its tree, and
   create or update the locks with [`relock`](#kernel-locks).
4. Run its kernel GPU tests from its own test crate on `kernel-harness`
   ([supported source surface](#supported-source-surface)).
5. Supply its chat format as native frontend entries or the opt-in
   `frontend-format` library ([chat frontend](#chat-frontend)), and check
   its render, tokenizer and streaming vectors on the built package with
   `chopinfer bundle check-frontend` ([frontend vectors](#frontend-vectors)).
6. Emit its serving profiles and deployment policy into the package
   ([serving profiles and deployment policy](#serving-profiles-and-deployment-policy)).
7. Keep its qualification declaration and modules at its model root
   ([packages, updates and qualification](#packages-updates-and-qualification)).
8. `build`, `package`, `exercise` against prebuilt hosts and record the
   [compatibility matrix](#compatibility-matrix).

What still needs a shared change is listed in
[adding a model](adding-a-model.md#what-still-touches-shared-code).

## Compatibility has four parts

| Contract | Rule |
|---|---|
| Rust SDK source | Pin one full SDK Git revision, its content identity and a reconciled external `Cargo.lock`. Changing the revision may require source changes. |
| Native ABI | Test independently built bundles and hosts. The ABI is alpha and changes incompatibly between Chopinfer releases; V1 is a draft ([ABI stability](../BUNDLE_ARCHITECTURE.md#abi-stability)), and its version number and C symbols alone are not a frozen compatibility promise. |
| Package and frontend schemas | Use the canonical emitter and existing parsers. No new manifest fields or package files. The frontend declaration is `chop-frontend-openai-v2`; its `format` section belongs to the model's own format code ([chat frontend](#chat-frontend)). |
| Host capabilities | Declare required semantics. The selected host is authoritative; unsupported capabilities and missing required callbacks fail admission. |

A package accepted by an untested host is not thereby a supported
combination. Native bundles and separately installed preparation tools are
trusted code; file integrity does not establish publisher trust or model
quality.

## Supported source surface

| Role | SDK entry points |
|---|---|
| Bundle | `runtime-bundle-abi`, `runtime-bundle-sdk`, optionally `bundle-runtime`, `kernel-core`, `kernel-build`, `kernels-sm120`, `encoder-spec`, `frontend-format`, `runtime-grammar` |
| Compiler | `model-compiler-sdk`, `runtime-bundle`, `runtime-weights`, `runtime-frontend-contract`, `runtime-tokenizer`, `kernels-sm120`, plus ABI, SDK, `kernel-core` and `encoder-spec` |
| Optional contract | ABI, SDK, `kernel-core`, `encoder-spec`, `projection-spec`, `frontend-format`, and the model-side `model-compiler-sdk`, `runtime-bundle`, `runtime-weights`, `bundle-runtime`, `kernel-build` and `kernels-sm120` |
| Test crates (members no role declares) | Every role's entry points, plus `kernel-harness`, `runtime-cuda` and `runtime-host` |

Each crossing into the SDK is checked against the surface of the role that
owns the crossing crate, so the contract, which both the bundle and the
compiler link, keeps the contract surface whichever role reaches it.

`python3 -B -m tools.model_sdk.graph --manifest M --inventory I --sdk DIR`
checks an external workspace against this surface using Cargo package IDs and
all-feature normal and build edges (and a test crate's dev edges); core uses
the same checker. Core model crates, fixture crates, `kernel-lab`,
`runtime-conformance` and second SDK checkouts are unsupported entry points,
every cdylib must be declared, and bundles cannot link `runtime-cuda`,
`runtime-engine`, `runtime-host` or `runtime-server`. A model runs its kernel
GPU correctness tests from its own test crate on the model-free kernel
harness (`kernel-harness`), never by registering in kernel-lab; keep that
crate out of the workspace's default members, since it links the CUDA
driver (the example's `lab`).

The compiler uses `emit_bundle_directory_v1` with an explicitly supplied
native module; it does not build the bundle. Small metadata a bundle reads
through the host's authenticated-component service uses the
`chop.bundle-metadata.` role prefix.
`runtime_bundle_sdk::rank_index` owns the rank-index parser.
`kernels_sm120::cuda_source_root()` exposes the shipped CUDA sources; the
external workspace includes them through a real nested `sdk/` checkout
(symlinks to sibling trees are unsupported). Encoder declarations
(`contract/encoders/*.json`) are parsed by `encoder-spec`
([its README](../crates/encoder-spec/README.md)).

## Chat frontend

The model owns its chat format. The bundle's native module exports
`chop_bundle_frontend_api_v1` under capability `chop.frontend.v1`, which
every topology of the package requires; a host refuses to serve a package
without it. The export opens over the package's authenticated
`frontend.json`, renders chat requests into prompt segments, answers the
grammar a tool choice imposes as a `runtime-grammar` wire document, and
parses output incrementally into byte ranges. The host keeps HTTP, SSE,
stops, UTF-8, limits, tokenization and grammar masks, and checks every
answer. The entries, the thread and lifetime rules and the host checks are
in [frontend export](../BUNDLE_ARCHITECTURE.md#frontend-export).

There are two ways to supply the format:

- **A Rust renderer on the shared parsers.** Port the chat template to a
  `frontend_format::ChatFormatV1` in the contract crate, declare what the
  shared parsers read in `format.parser`, and export it with
  `frontend_format::export_chat_format_v1!(Format);` at the bundle's crate
  root, as the in-tree models and the maintained example do
  (`examples/external-model/contract/src/frontend.rs`). The `format` object
  is then exactly that parser section. The
  [frontend-format README](../crates/frontend-format/README.md) describes
  the trait, the toolkit, the parser families and the `format.parser`
  section.
- **Model-owned code.** A completion syntax neither shared parser family
  reads gets a new named family in the SDK
  ([what still touches shared code](adding-a-model.md#what-still-touches-shared-code)).
  Until the SDK revision a model pins has one for its completions, the model
  implements `runtime_bundle_abi::frontend::FrontendV1` in the bundle and
  exports it with `runtime_bundle_abi::export_frontend_v1!`. The
  `format` object then holds whatever that code reads; the host checks only
  that it is a JSON object of at most 192 KiB that names no member twice.

The parser reads only the bytes the decoder keeps. Chat requests skip special
tokens by default (`skip_special_tokens`), and the decoder removes a special
token's bytes before the parser sees them, so every marker the parser must
find in a completion (a reasoning close, tool-call markers) must be a
non-special added token in `tokenizer.json`. Prompt-only markers may stay
special. A stop token's bytes are removed too unless the request sets
`include_stop_str_in_output`.

The render's `effort` must be one of the declared `reasoning.efforts` names
in every mode, chat included. A format that reads a numeric effort (for
example from `chat_template_kwargs`) maps it onto declared names.

A model-owned parser answers each push with events and a cursor, the offset
where the bytes it has not settled begin. Ranges and the cursor are raw byte
offsets into the whole completion, not into the push's suffix. The cursor
never moves back, and the host scans client stops from it; the cursor and
both ends of every range must fall on a decoding-unit boundary, never inside
a UTF-8 character or ill-formed byte run. The finish comes once, after the
parser has read every released byte with none held, and no later than the
final push. A parser that keeps JSON numbers as written (`2.50`, a 20-digit
integer) enables serde_json's `arbitrary_precision` feature on its own
dependency; the SDK's workspace dependency does not.

The render request carries the client's whole `chat_template_kwargs` object,
so a model reads its own template keywords without a host change. A model
whose template renders more message roles than system, developer, user,
assistant and tool lists them in its open answer (`OpenedV1::extra_roles`);
the host then accepts text messages with those roles and passes each role by
name. A model whose template renders
`tool_reference` parts in tool results sets `OpenedV1::tool_references`; the
host then accepts them in tool messages and passes each to the render
request as `PartV1::ToolReference` with its name. A format that leaves it
unset has the host refuse such parts, and refuses any that reaches its
render with `frontend_format::TOOL_REFERENCES_UNSUPPORTED_V1`. The limits are in
[frontend export](../BUNDLE_ARCHITECTURE.md#frontend-export).

The rest of `frontend.json` is model-neutral data the host reads: models,
context, limits, vocabulary and generation defaults; `tokenizer`, the file
identities and the component policy the host's reader applies (a component
the reader lacks is a shared capability, not a package change);
`reasoning`, including the `end_markers` that end a reasoning section;
`tools`, the tool-name byte set and argument shape; `multimodal`; and
`response.reasoning_field`.

Token-evidence qualification (the agentic sweep's raw re-parse) re-parses
outputs independently of the served parser. A format that declares a
library tool grammar uses the shared re-parser. Any other format supplies
`agentic_markup.py` in its model root (`--model-root` or
`CHOPINFER_QUAL_MODEL_ROOT`) with the interface in
[the qualification tools](../tools/qual/README.md#agentic-coding-mixed-prefilldecode-sweep);
a run without one is refused.

### Frontend vectors

`chopinfer bundle check-frontend` checks a built package's frontend against
the model's render, tokenizer and streaming vectors:

```text
chopinfer bundle check-frontend --bundle DIR [--topology ID]
  [--render FILE]... [--tokenizer FILE]... [--streaming FILE]...
```

It runs what `serve` runs: it verifies the package, opens its native module
and frontend export over the authenticated component, and builds the declared
tokenizer from the package. It needs no GPU, no checkpoint and no Rust
registration. Each case prints `pass` or `FAIL` with the first difference, and
any failure exits nonzero, as does a run whose files hold no case. Each file
is a JSON array of cases:

- **Render** (`--render`), the `chat-goldens.json` shape: `id`, `messages`,
  `tools`, `chat_template_kwargs`, `rendered_text` and `token_ids`, with
  optional `tool_choice`, `reasoning_effort`, `response_format`,
  `add_generation_prompt` (default true) and `continue_final_message`
  (default false). Other fields are ignored. The case is rendered as
  `/tokenize` renders a chat request: the transport parses it, the frontend
  renders it, and the prompt is tokenized with the prompt special tokens.
  Messages are read as the model golden tests read captures. Tool-call
  arguments given as JSON values are sent as their compact JSON text. A tool
  result without `tool_call_id`, or a call without `id` or `type`, gets an
  empty id and type `function`. An assistant turn's `reasoning_content` is
  sent under `response.reasoning_field` when the frontend reads
  `reasoning_content` from history at all. A `file://` media URL is sent
  inline, because a render places media without reading them. The text must
  equal `rendered_text`, or its tokenizer round trip must (a normalizing
  tokenizer captures normalized text). The ids must equal `token_ids`; a
  prompt with media is compared before its placeholders expand.

  A case may hold `refusal`, an object with optional string `message` and
  `parameter`, in place of `rendered_text` and `token_ids`. It passes only
  when the transport or the frontend refuses the request with the stated
  message and parameter, and fails when the request renders; `{}` accepts
  any refusal. A case with `raw: true` is sent exactly as written, with none
  of the capture changes above, so a missing id or a reasoning field tests
  what a client sent. Its media URLs are not rewritten either: write an
  inline `data:` URL, since a `file://` one is refused as a client's would
  be. A raw case may hold no field besides `id`, `raw`, its expectation,
  `messages` and the request fields.
- **Tokenizer** (`--tokenizer`), the `tokenizer-golden.json` shape: `text`
  and `token_ids`, with an optional `decoded`. The text is encoded as
  `/tokenize` encodes a prompt, and the ids decoded as `/detokenize` does.
- **Streaming** (`--streaming`): `id`, `request`, `token_ids` and `expect`,
  with optional `steps` and `separator`.
  `request` holds `messages` and any of the optional render request fields.
  `token_ids` is a generated completion: a stop token ends it, and without
  one it ends at the token limit. `expect` holds `reasoning` and `content`
  (default empty), `tool_calls` (a list of `{name, arguments}`, default
  empty, with `arguments` as the format writes them) and `finish_reason`
  (`stop`, `length` or `tool_calls`, as a chat response reports it).
  Optional `steps` pin what the response publishes as tokens arrive: each
  step holds `token_ids` (the steps' ids are `token_ids` in order),
  `reasoning`, `content` and `tool_call_deltas` (`{index, name, arguments}`,
  merged per call, with `name` only in the step that opens the call), all
  defaulting to empty. The last step also holds what the completion's end
  delivers. The ids decode through the decoder admission builds, with the
  frontend's parser and the host's checks, and every event is checked as a
  served response checks it, once with tool calls delivered whole and once
  streamed. Both deliveries must match `expect` and each step's reasoning and
  content; the streamed one must also match each step's deltas. A vector
  without steps checks only the totals, so a format that withheld everything
  until the end would still pass it. A vector runs with the request
  defaults, so special tokens are skipped: one that crosses a reasoning
  close or a tool call checks that those markers reach the parser.
  `separator` is a string: a round-trip vector's history text after its stop
  token, which the vector's generator checks against the template. The check
  reads it and never decodes it, since a served completion ends at its stop
  token.

A streaming case has this shape; its ids come from the model's tokenizer:

```json
[{"id": "reasoning-then-call",
  "request": {"messages": [{"role": "user", "content": "Hi"}],
              "tools": [{"type": "function", "function": {"name": "ping",
                "parameters": {"type": "object", "properties": {}}}}]},
  "token_ids": [1, 2, 3],
  "expect": {"reasoning": "plan", "tool_calls": [{"name": "ping", "arguments": "{}"}],
             "finish_reason": "tool_calls"},
  "steps": [{"token_ids": [1], "reasoning": "plan"},
            {"token_ids": [2], "tool_call_deltas": [{"index": 0, "name": "ping"}]},
            {"token_ids": [3], "tool_call_deltas": [{"index": 0, "arguments": "{}"}]}]}]
```

The maintained example's vectors are in `examples/external-model/fixtures/`,
and the SDK smoke runs them against its package. The in-tree models keep
theirs beside their goldens and run them through the same checks in a model
test, with the checkpoint's tokenizer. Each in-tree model's committed HF
corpus is vectors in these shapes too: `fixtures/<m>/frontend/corpus.json`
holds raw render vectors, refusals among them, and `corpus-rounds.json`
streaming round-trip vectors with their separators, which the model's
corpus test runs and also parses at every byte split
([chat frontend](adding-a-model.md#chat-frontend)). The HF corpus tools run
on in-tree models only, so an external model's frontend evidence is its
vectors.

## Delivery stages

1. Obtain a clean full SDK checkout at an immutable revision and preseed its
   Cargo registry cache before going offline.
2. `export` the template to an unrelated location (it writes `sdk-pin.json`)
   and run `lock` once to create the external `Cargo.lock`; `lock` never
   replaces an existing one. After changing the model's own dependencies,
   `lock --update` resolves the lock again offline from the SDK's pins and
   reports the packages added, removed or changed; it keeps the previous
   lock when the result moves an SDK pin or fails `check`. Commit both in
   the model repository.
3. Implement model-owned code, numerical tests and declarations. Use `check`
   and locked offline `build` with explicit output roots. SDK sources are
   read-only and verified again after builds. A development export
   (`--development`) captures local edits and cannot pass release-pin
   verification. Create and update the model's kernel locks with `relock`
   ([kernel locks](#kernel-locks)).
4. `package` the checkpoint's metadata with the supplied module; a model with
   weights plans them from the checkpoint
   ([direct weight loading](direct-loading.md#planned-components)). Run
   `exercise` against prebuilt hosts, then model-specific qualification.
5. Ship releases independently. Repeat the compatibility matrix when changing
   the SDK, host support claims, toolchain, model contract or kernel
   selections. Failed cells stay failures.

Use Rust 1.89.0, CUDA 13.2.86 and locked offline Cargo; the SDK build refuses
`KERNELS_LOCK_UPDATE`, `CHOPINFER_KERNEL_LINEINFO`, `CHOPINFER_MODEL_RELOCK_DIR`
and lock initialization overrides in its environment, and forces each empty
(with `force` and `relative = false`) for build scripts through Cargo CLI
configuration, which a model's `.cargo/config.toml` `[env]` cannot override.
Every kernel build report of a build script the build used, including one
Cargo reused from an earlier build, must show `Verified` for a lock inside
this workspace whose recorded digest matches the lock's current bytes. Cargo
shares build script outputs between exports of one revision, so `build` and
`relock` dedicate each target directory to one workspace, and each holds the
workspace and then the target exclusively for its whole run (through a
`.chopinfer-sdk-lock` file in each, which the model's `.gitignore` lists),
refusing a concurrent invocation with the holder's process ID. Both select every
role declared in `model.toml` explicitly and `build` returns only artifacts
its own Cargo run reported; a lock whose build script does not run because
its feature is off is unused by that build and is not verified. A lock mismatch is an environment
or input difference to investigate, never an instruction to rewrite SDK locks. Keep
generated outputs, logs, receipts and packages under `$CHOPINFER_DATA` or
another external directory.

### Kernel locks

A model's own kernel locks (lock files or `kernel_build::LockDir`
directories under its tree) are created and updated only by `relock`, which
takes the same workspace, target, logs and `--feature` arguments as `build`.
A build script opts a lock in with `kernel_build::ModelLock::from_env`, then
builds against `ModelLock::path` with `ModelLock::options`. Outside a relock
those are the options `BuildOptions::from_env` reads, so an in-tree build
still honors `KERNELS_LOCK_UPDATE=1`; SDK builds force it empty and verify
the lock in place. Candidate mode needs the marker file relock creates in
its candidate directory. `relock` builds once with every model lock
redirected to a candidate, refuses any lock outside the model's tree (no SDK
lock is ever rewritten), replaces only changed files and rebuilds to verify
them. Before replacing a lock it writes the report with status `pending` and
stages the old and new bytes in `relock-state/` in the logs directory. A
failed replacement or verification restores every old lock, reports
`restored` and keeps the candidates in `relock-failed-*/`; a failed
restoration reports `restore-failed` and keeps `relock-state/`. Reports are
replaced atomically, and the final `applied` report is published before the
recovery state is removed; the next relock finishes that cleanup if it was
interrupted. A rollback publishes `restoring` before it changes any lock and
changes none if it cannot. Any other interrupted run leaves its state, and the next relock
refuses to start until it is resolved; its `manifest.json` maps each lock to its old bytes
(null when the lock did not exist) and its candidate. Its `relock-report.json` lists, for each lock, the records added,
removed and changed, and classifies each change as `cubin` (the cubin bytes
changed), `recipe` (another gating field changed) or `provenance` (only the
source closure digest changed: source paths or the SDK revision moved while
the cubins stayed identical). Review every change; releases still require
each lock `Verified` by an ordinary build. In-tree crates keep
`KERNELS_LOCK_UPDATE=1` ([kernel library](kernel-library.md)).

### Exporting an in-tree model

`export --model models/<m>` is a migration helper for the in-tree models,
not a general analysis of how Rust code builds paths. It lays out an in-tree
model directory as the workspace instead of the template: the model directory
at the pinned revision becomes the root, `fixtures/<m>` its `fixtures/`, and
`Cargo.toml` is generated from the SDK's. It rewrites only the path forms
listed below, records each rewritten file in `model-export.json` and checks
that every reference it rewrites or matches names the same content, read from
the base it assigns to that reference. In text files it rewrites checkout-relative paths, and `../` paths in Markdown.

In Rust it reads each string literal with its escapes decoded and each path
normalized (`./`, `x/../`), reads raw identifiers such as `r#include_str` as
their plain names, and rewrites a path that must move as one plain literal.
It rewrites exactly these forms:

- An include macro argument (any delimiter and trivia) or a `#[path]` value
  (also in `cfg_attr`) is read from its file.
- A `concat!` of string literals as an include macro argument is one path;
  when it moves, it becomes one literal.
- The literals `concat!` joins to `env!("CARGO_MANIFEST_DIR")` are read from
  the crate directory.
- The crate directory is `Path::new` or `PathBuf::from` of
  `env!("CARGO_MANIFEST_DIR")`, or of `env::var_os` or `env::var` of it
  unwrapped.
- A known directory is the crate directory, a `.join` of a literal to a
  known directory, a `&`, clone or conversion of one, an identifier `let`
  bound to one earlier in the same function, or a `&Path` or `PathBuf`
  parameter that every call of its function in the file passes the same
  known directory.
- A climbing literal in `.join(...)` on a known directory is read from it.
- A climbing literal in `.join(...)` on any other receiver, in a crate's build
  script or `tests/` file, is kept unchanged when it stays inside the model
  directory read from the crate directory.
- A checkout-rooted literal (`crates/...`, `models/...`) in `.join(...)` is
  read from the root only when its receiver is a known directory that is the
  checkout root; in `.starts_with(...)` also when the same function joins
  that receiver identifier to the root.
- Any other checkout-rooted literal is read from the root in a file that
  joins its crate directory up to the root in one of these forms.

It stops with an error naming the file, line and literal for the
constructions it recognizes but cannot place: a climbing `.join` literal, or
a checkout-rooted `.join` or `.starts_with` literal, whose base the text does
not establish ("the base of this call could not be established"), a
checkout-rooted literal joined to a known directory other than the root that
names nothing from that directory, a literal outside these forms that climbs
to a checkout file outside the model directory or names one from the
checkout root, and a `concat!` that mixes such a literal with another macro.
A `#[path]` attribute inside an inline module also stops the export, with
its file and line.

It does not follow other ways of building a path, and code that uses them may
export without an error and then read a different file. Reviews of the
exporter found these classes, among others:

- a path completed at run time, such as a build-script variable joined with
  a `../` literal;
- a base reached through an intermediate join of a rooted directory, as in
  `let shared = root.join("crates"); shared.join("../top.txt")`;
- a binding shadowed inside a nested block;
- a path collected from parts, as in
  `["..", "..", "top.txt"].iter().collect::<PathBuf>()`;
- a climbing `.starts_with` literal, which is rewritten as if read from the
  crate directory whatever its receiver;
- a `concat!` outside an include macro, for example bound to a constant or
  variable and joined later;
- a rooted join back into the exported model spelled with `./` or escapes,
  as in `root.join("./models/<m>/file")`;
- a file name that needs escaping in a Rust literal, such as one with a
  backslash.

A reference left unchanged that names nothing in the checkout, such as a
path a program completes at run time, is kept, listed under `unresolved` in
`model-export.json` and printed as a warning with its file and line. A
checkout-rooted literal naming in-tree tooling outside `crates/`,
`fixtures/`, `include/` and `models/` is kept and reported the same way:
MiMo's oracle diagnostic tests (`compiler/src/oracle/diagnostic.rs`) run
`tools/dev/diagnostics.py` from the checkout root, so they fail in an export.
GLM's Python reference scripts (for example `reference_fixtures.py`)
compute the checkout root from their own location and import the
qualification tools under it, so they run only from the checkout. Neither
limit reaches the bundles, helpers or packages.

The in-tree bundles lock their kernels through `ModelLock`, so `relock` works
in an export; the new layout changes only source closure digests, which it
reports as `provenance` changes.

The verification of an export is the exported workspace's own
`check --release`, build and tests, which the SDK runs, and for a model that
serves, a serving comparison of the exported build against the in-tree build.
For GLM, MiMo and Qwen the exported `check --release` passed, and GLM's
exported build and tests passed. After `relock`, which changed only
provenance records, Qwen's exported build and tests passed, and MiMo's
exported build passed with its bundle tests; its test run stopped at the two
oracle diagnostic tests named above. On 2026-10-15 GLM 5.3 Flash, exported and
built this way, served token ids and logprob bits identical to the in-tree
build at TP4 (`tp4-ar` and `tp4-mtp`).

## Packages, updates and qualification

A publisher release includes its source revision, SDK pin, external
`Cargo.lock`, model kernel locks, compiler executable, native module, frontend
assets, profile declarations and tested host matrix. Keep publisher metadata
and compatibility reports beside the package; the deployment directory holds
only the canonical manifest and declared files.

Keep the same model-owned bundle ID for compatible releases. A release is its
package directory: emit each into a new directory, then start a new process
with `serve --bundle PACKAGE` (and `--checkpoint DIR` when the package plans
weights). Rollback restores the old directory and supervisor settings. There
is no hot reload or automatic fetching.

For several quantized checkpoints of one architecture, declare the
projections in `contract/projections.json` and use the shared
[per-operation precision machinery](quantized-checkpoints.md#adding-a-family-or-layout),
as the maintained example's precision exercise does; another
checkpoint inside an implemented operation domain needs a normal package
emission, not another model runtime. An optional preparation provider for
`serve --model` implements the shared
[preparation API and protocol](compatible-checkpoints.md); never discover
executables in checkpoint directories.

Pass an external `qualification.json` and `--model-root` to the release
gate (`tools/qual/release_gate.py`), which runs against an already-running
server; relative paths stay model-relative. Model-provided qualification
modules (`diagnostic.py`, and `agentic_markup.py` for a format without a
library tool grammar) sit directly in the model root, which `--model-root`
or `CHOPINFER_QUAL_MODEL_ROOT` names; nothing registers in `tools/qual`. The
[correctness runner](../tools/qual/README.md#correctness-runner) serves only
in-tree models' `qual.json` configurations. External
declarations must state a boolean `qualification.mechanics_only`. A true value
or the `--mechanics-only` flag forces an unqualified result; setting it false
is not evidence. An eligible production declaration needs model-owned oracle
evidence and the relevant service checks.

### Serving profiles and deployment policy

The package carries both. The compiler emits the model's `profiles.json`
(`chopinfer-serve-profiles-v2`, [serving profiles](serving-profiles.md)) with
`model_compiler_sdk::package_draft::serve_profiles`, so
`serve --bundle PACKAGE --profile NAME` resolves `NAME` from the package's
verified `serve-profiles` component
([selecting a profile](serving-profiles.md#selecting-a-profile)). It emits
`deploy/policy.json` (`chopinfer-deployment-policy-v1`, module
`model_compiler_sdk::deployment_policy`) with `package_draft::deployment_policy`:
the bundle ID, drafters and the model's own container variables under its
knob prefixes, none covering a shared variable. A preparation helper also
reports it (`Provider::deployment_policy`), and the image entrypoint
validates the environment against it
([package selection](deploy-common.md#checkpoint-model-cache-and-package-selection)).
Neither the host nor the image keeps a per-model file.

## Compatibility matrix

`python3 -B -m tools.model_sdk.matrix --help` runs two SDK revisions against
two pairs of prebuilt hosts (four cells, no host builds) and records
revisions, features, toolchain identity and digests outside the source tree.
Each cell checks commit, cancel/replay, cleanup, retirement over HTTP,
refusal of incompatible or damaged packages, and same-ID update and rollback.
Record each tested host's full build revision, features and artifact digest
in the publisher's compatibility report; never substitute a crate version
string for the tested source identity.

## Offline probe metadata

`runtime_bundle_sdk::diagnostic` and `tools.model_sdk.diagnostic` help model
probes write diagnostic captures (`chopinfer.diagnostic.v1`) without any ABI
or serving change; see [debugging](debugging.md).
