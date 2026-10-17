# Serving qualification checks

Qualification measures a running Chopinfer server against pinned references
and records the result as JSON receipts. The pieces:

- **Declarations.** Each model's `models/<model>/qual.json` is the single
  source of its release gate: cohorts, checks, limits, pinned references and
  inputs ([model declarations](#model-declarations)). New models start from
  [templates/qual.json](templates/qual.json).
- **Check kinds.** Each check runs through a kind that captures, verifies,
  scores and judges its receipt. The six shared kinds are specified in
  [check kinds](../../docs/check-kinds.md); `chopinfer eval` is their one
  scorer.
- **Release gates.** `release_gate.py` runs a declared cohort (`quick` or
  `full`) against an already-running server ([release gates](#release-gates)).
- **Correctness runner.** `release_preview.sh` builds a release install,
  serves a model's configurations from their checkpoints and runs its quick or
  full suite against the references and the stored baselines
  ([correctness runner](#correctness-runner)).
- **References.** Numerical references come from the
  [GPU reference runner](../../docs/reference-runner.md). The numerical
  contracts and gate limits are in [numerics](../../docs/numerics.md#serving-qualification-gates).
- **Paired accuracy.** [Paired release accuracy](../../docs/paired-accuracy.md)
  compares complete candidate and accepted-release captures under predeclared
  margins, in addition to the absolute and C1 gates. Its proposed policy
  requires operator approval before candidate GPU runs.

Run everything from the workspace root. The tools need only Python 3's
standard library; numerical tap comparison (`tapdiff.py`, which the encoder
gate runs) also needs numpy, so it runs under the reference runner's Python,
`CHOPINFER_REFRUN_PYTHON`. A nonzero exit means a
check failed or could not run.

## One qualification entry point

`python3 -m tools.qual --help` lists every command. `gate` runs or inspects a
release gate; `check`, `rescore` and `list` use the check-kind registry;
`preview` runs the [correctness runner](#correctness-runner) through
`release_preview.sh`; `refs` adopts and verifies locked inputs. Every other
command runs one tool (check names use hyphens, such as `teacher-arms`,
`multimodal` and `seeded-equality`); its help keeps the script's program
name, and each tool also runs directly as `python3 -B tools/qual/<tool>.py`.

```sh
python3 -m tools.qual gate --model mimo26 --profile quick --plan
python3 -m tools.qual list --model qwen38 --json
python3 -m tools.qual check teacher --help
python3 -m tools.qual preview --help
```

Measurement and inspection helpers outside this list live in
[dev/](dev/README.md).

### Environment

| Variable | Use |
|---|---|
| `CHOPINFER_QUAL_CONFIG` | `models/<model>/qual.json` (or an external gate config) for standalone tools that read model facts: fixtures, references, reasoning controls, verify buckets, prefix rules. No shared tool picks a model when it is unset. The release runner sets it for each worker. |
| `CHOPINFER_DATA` | Base for receipts (`$CHOPINFER_DATA/qual/<model>`), caches and external inputs. Tools that need no model facts, such as `bench.py`, write to `$CHOPINFER_DATA/qual`. |
| `CHOPINFER_QUAL_RECEIPTS` | Overrides the receipt directory. |
| `CHOPINFER_MIMO_FIXTURES` | Root of the frozen GSM8K and functional (media and needle) inputs. |
| `CHOPINFER_QUAL_MODEL_ROOT` | Root that declared paths and the reference lock resolve from (default: the checkout). |
| `OPENAI_API_KEY` | Bearer token, when the endpoint requires one. |
| `CHOPINFER_EVAL` | The `chopinfer` binary the shared kinds call, for scoring and for the checks and records `chopinfer eval` captures; default the checkout's build (`cargo build -p runtime-host --bin chopinfer`). |

See [toolchain setup](../../docs/toolchain.md) for `local.env`.
`release_gate.py` and `release_preview.sh` read it themselves. Tools validate
the config and their receipt path before sending requests.
`release_gate.py --model M --list-inputs` prints the resolved inputs, checks
and policy without HTTP, GPU access or receipt writes; it authenticates
existing files and leaves missing environment paths null.

### Portable `chopinfer eval`

`chopinfer eval` runs the shared kinds without Python, from a model directory
or an installed package:

```sh
target/debug/chopinfer eval list --model models/qwen38
target/debug/chopinfer eval run --model models/qwen38 --suite quick --url http://127.0.0.1:8441
target/debug/chopinfer eval verify-references --model models/qwen38
```

`eval run --launch [SERVE FLAGS...]` starts and stops its own server. Exit
statuses: 0 pass, 1 a gate failed, 2 usage or declaration error, 3
insufficient evidence, 4 environment error. A passing portable subset does
not qualify a release; the full gate also runs developer checks. See the
[CLI reference](../../docs/cli.md) for the options and `eval rescore`.

## Release gates

Run one command against an already-running server:

```sh
python3 -B tools/qual/release_gate.py --model mimo26 --url http://127.0.0.1:8311 --profile quick
python3 -B tools/qual/release_gate.py --model qwen38 --url http://127.0.0.1:8441 --profile full --inputs release-inputs.json
```

(`python3 -m tools.qual gate ...` is the same command.) `--model` selects
`models/<model>/qual.json`; `--served-model` selects an API alias (default:
the first `/v1/models` entry). The gate never starts or stops servers,
probes GPUs or downloads datasets: prepare fixtures and the `evals.py fetch`
cache first. Receipts go to `$CHOPINFER_DATA/qual/<model>` unless
`--receipts` or `CHOPINFER_QUAL_RECEIPTS` says otherwise.

| Profile | Checks | Meaning |
|---|---|---|
| quick | Complete teacher set, API, structured outputs, 60-second soak at C4 | A screen with a 900-second budget that never qualifies a release. Every check runs and reports its verdict. |
| full | Every declared accuracy and serving check: the quick checks, verify rows and speculation, prefix-cache equality, preemption, multimodal and encoder parity, needles, frontend goldens, paired downstream evals, generated-logprob parity and a 1,380-second soak at C32 with mixed traffic. MiMo adds grounding and the prefix hit/cold/floor comparison; Qwen adds its long-context teacher set and layer comparison. | All must pass. The run stops at the first failure and lists the rest as not run. `--plan` lists the checks. |

Each check has its declared `timeout_seconds`; a full run has a 48-hour
limit. Missing metrics, empty selections, nonfinite scores, nonzero child
exits and timeouts fail closed. Structured outputs must be 100% valid with
nothing refused or truncated; paired evals must cover all 2,819 items in
both arms with no suite significantly worse (p < 0.05).

**Full-profile inputs.** `--inputs FILE` is a JSON object of evidence the
gate cannot produce from the candidate alone; it never resumes an older run.

| Key | Content | Produced by |
|---|---|---|
| `reference_url` | An already-running reference arm serving the same API alias | operator |
| `server_log` | The candidate's startup log ([checkpoint binding](#checkpoint-binding)) | `chopinfer serve` |
| `verify_ar`, `verify_candidate` | Exact verify-row records | `verify_exact.py record` |
| `spec_ar`, `spec_candidate` | Verify-set records | `spec_verify.py record` |
| `prefix_record` | Prefix hit/cold/exact receipt | `prefix_exact.py record` |
| `preemption_reserved`, `preemption_candidate` | Preemption records at C16 | `preempt_compare.py record` |
| `preemption_reserved_c1`, `preemption_candidate_c1` | Preemption records at C1 | `preempt_compare.py record` |
| `encoder_parity_run` | Encoder parity run directory | [encoder parity](#encoder-parity) |
| `capture` | Frontend, needle, API and render captures | frontend capture tools |
| MiMo: `prefix_baseline`, `prefix_floor`, `prefix_candidate` | Prefix comparison records | `prefix_cache.py record` |
| Qwen: `layer_dump`, `layer_reference` | A [tap capture](#tap-captures) and the reference runner run of the same prompt | `qwen38_layers.py capture`, `tools.refrun run --taps` |
| `layer_dump`, `layer_reference` | The same, for the model-neutral [text layer comparison](#text-layers) | `text_layers.py capture`, `tools.refrun run --taps` |

Forced-token records need a server started with `--debug-reference-tokens`,
not the release server. External evidence must belong to the candidate
release: the summary records input paths and hashes, but cannot prove which
binary wrote a directory, so a pass depends on that provenance.

**Capture.** A model declaration's teacher, api and structured checks (the
kinds `chopinfer eval` captures) run as `chopinfer eval run --check ID
--cohort PROFILE` against the same server and the declaration (or
configuration) the policy came from, bound to `server_log` when the run has
one; the record in its report (`checks/eval-ID/`) is the check's receipt,
and a declared `--capture-out` path receives its compact capture. The
binary is `CHOPINFER_EVAL` or the checkout's build. Every other check runs
its Python module as a worker, as do the api and structured mechanics
checks of a release-gates/v1 config (the mechanics fixture, external
declarations), which `chopinfer eval` does not read.

**Output.** Each run creates a new directory under the receipts directory
with `summary.json`, `summary.md`, the frozen `policy.json`, child logs and
raw receipts, recording the Git revision, server identity, policy SHA-256,
thresholds and input and result hashes. Exit zero means the profile passed;
only a full non-mechanics run sets `qualified: true`.

**Planning.** `--plan` describes a cohort's ordered checks, arguments,
prerequisites, budget, predicates and policy digest without HTTP or
execution. `bash ci/quick.sh --report --base main --format json` uses it to
suggest the checks a change selects; see
[testing](../../docs/testing.md) for how stored
evidence applies to a candidate.

**External models** pass `--config PATH --model-root DIR` instead of
`--model`; their declarations must state `qualification.mechanics_only`.

**Mechanics self-test**, against an already-running TP4 fixture server (it
never qualifies a model; `bash ci/cpu.sh` runs it):

```sh
python3 -B tools/qual/release_gate.py --url http://127.0.0.1:PORT --mechanics-only
python3 -B -m unittest discover -s tools/qual -p test_release_gate.py -v
```

### Model declarations

`models/<model>/qual.json` (`chopinfer-qual-model/v1`) declares:

- `cohorts`: per cohort (`quick`, `full`) the ordered checks.
- `change_classes`: the checks a change of each class (`frontend`,
  `quality`, `service`, `speculation`, `prefix`) selects in
  `ci/validation.py`. Every class must be present; an empty list states that
  the gate has no check for it.
- `checks`, keyed by ID: module, arguments, `cohort_args`, receipt, limits
  (`rules`, `input_rules`, `constants`), prerequisite `files`, an
  optional `informational_reason` and optional `waivers`.
- `references`: each pinned reference with its SHA-256, schema, variants
  and optional workload ([reference cache bindings](#reference-cache-bindings)).
- `inputs`, `qualification` (the facts tools read), `receipts`,
  `limitations` and `capabilities`.
- `renamed_checks` (optional): former check ID to current ID, so stored
  receipts under the former ID are read under the current one.
- `preview` (optional): the [correctness runner](#correctness-runner)'s
  settings. Absent, the runner does not serve the model. `profile` names the
  `profiles.json` entry, and `speculation` (`drafter`, `tokens`,
  `draft_cost`) must equal that entry's, so a profile change needs a
  matching reviewed change here. `spec_greedy` says whether C1 speculative
  greedy output must equal AR (`exact`) or is reported (`informational`,
  with a `reason`); `soak_prefix_mix` (default true) adds the prefix mix to
  the speculative soak; `gsm8k_floor` is the GSM8K screen's least correct
  items per arm, out of 200; `numerical_reference` states in the report
  which reference the teacher check is gated against; `quick_arms` (default `["ar", "spec"]`) names the
  arms whose servers run the quick cohort: a declaration whose live checks
  admit only a drafting server names `["spec"]`; `package_max_sequences`
  (optional) is the most sequences the model's package serves. The previous
  preview screen's `sweep`, `greedy_pin`, `budgets` and `checkpoint_verify`
  are retired and refused as unknown fields.
- `configurations` (optional): each other serving configuration or
  checkpoint of the model, by name ([below](#configurations)).

#### Configurations

A serving configuration a gate run against the default configuration's
server cannot reach (another context-parallel size, KV element, draft depth,
tensor-parallel size or build), or a second checkpoint, is a named entry of
`qual.json`'s `configurations`. The runner reads how it is served
(`checkpoint`, `tensor_parallel_size`, `serve`, `environment`,
`install_features`, `producer`, `screens`, `experimental`); `chopinfer eval
declaration --model DIR --variant NAME` expands the rest onto the base
declaration: `checks` and `quick` select the cohorts, `serving` sets the
served profile facts on every profile rule, `references`, `inputs` and
`qualification` replace pins and facts, `rules`, `cohort_args` and `append`
change or add rule values, `waivers` (check ID to its list) gives the
configuration's checks their own waivers (a base check's are the default
configuration's alone), and `from_default` names the inputs the default
configuration's run records (such as its FP8 teacher capture). The result
names the configuration as its `variant`. Tools read it as `--model
<model>.<name>` (`release_gate.py`, `checks.py`, `refs.py`) and the runner
as `--config <name>`; `ci/validation.py` selects a configuration's checks by
its readers and change classes. A configuration reads the model's lock,
`qual-references.lock.json`, with its own entries under
`configurations.<name>.entries` where they differ, so shared inputs keep
their pins and directory scopes.

A declaration file beside `qual.json`, `models/<model>/qual.<variant>.json`
with its own lock `qual-references.<variant>.lock.json`, is still read when
`qual.json` declares no configuration of that name: an exported model root
ships its own, and `models/qwen38/reference_exports.py adopt` writes one for
a new export until `qual.json` declares it. A model or variant declaration
selected by location must name the model its directory gives it and be the
one under this workspace or under `CHOPINFER_QUAL_MODEL_ROOT`
(`--model-root`), since its lock is found there by name; one from another
tree is refused. A model or variant declared in either tree needs its lock,
and a config without a lock does not vouch for a file a locked declaration
declares.

`tools/qual/declaration.py` refuses a malformed declaration with a message
naming the field and compiles it to the `chopinfer-release-gates/v1` policy
the runner executes (`policy.json`, `policy_sha256`). Every object has a
fixed field set: missing, unknown or mistyped fields are refused. A check's
module is a module of this directory other than a test; constant overrides
are typed by the module's own default.

The rules are implemented once, in `chopinfer eval declaration`
(crates/runtime-eval/src/declaration.rs); `declaration.py` sends each
declaration to it. `modules.json` lists each module a check may name and its
constants' types; after adding a module or a top-level name, regenerate it
with `python3 -B tools/qual/module_metadata.py` (`test_module_metadata.py`
fails until it is current). The crate's `tests/data/declaration-cases.json`
holds declarations the validator must accept or refuse with their field.

### New-model template

Start with [templates/qual.json](templates/qual.json), a comment-free v1
text-and-image declaration, and copy it into your model directory as
`qual.json`. It validates structurally but cannot qualify a model: its zero
hashes, checkpoint revision, paths, limits and budgets are placeholders.
Paths containing `sample-vlm` name files you must create. Only small
reference inputs belong in `fixtures/`; captures, workloads with media and
receipts belong under `$CHOPINFER_DATA`. The
[adding-a-model workflow](../../docs/adding-a-model.md)
connects the template to reference production and the release gate.

The table covers required and optional v1 fields. Omit optional sections
that do not apply; do not add empty or invented fields.

| Field | Meaning |
|---|---|
| `schema`, `model` | Exact schema tag and model directory name (letters, digits, underscores or hyphens). No central registration is needed. |
| `cohorts.<name>.checks` | Ordered, unique check IDs. `quick` and `full` are release-gate cohorts, separate from serving profiles. A cohort has no overall deadline: each check is bounded by its `timeout_seconds`. |
| `change_classes` | All five classes map to declared IDs; an empty list means no check for that class. Include media checks in affected frontend and quality changes. |
| `checks.<id>.module`, `.args` | Existing worker module and argument strings. `kinds.py` maps modules to kinds; there is no v1 `kind` field. Placeholders such as `{url}`, `{model}` and input names resolve at execution. |
| `.timeout_seconds` | Positive deadline for this check. |
| `.receipt`, `.stdout_json`, `.json_file` | The worker's output contract: receipt glob, JSON stdout (`true`), or JSON output filename. At least one is required. |
| `.reference_file` | Optional, with `json_file`: a file whose JSON the kind scores `json_file` against, for a kind whose reference is another record (determinism's other arm). The gate records its path and SHA-256 on the row, and a rescore refuses the row if that file changed. |
| `.reference_equal` | Optional, with `reference_file`: paths (lists of keys) at which the receipt and the reference record must hold the same, present value, such as the run ID and the checkpoint and build identities of two arms that differ only in serving options. The gate and rescore also admit the reference record as they admit a receipt (no mechanics-only, skipped or failed record, no unqualified model options). |
| `.rules` | Nonempty predicates: `path` (keys or indexes, `*` selects children), `op` (`eq`, `le`, `ge`, `count_ge`, `true_count_ge`), `value`, and `source` citing its reviewed basis. |
| `.files`, `.input_rules` | Required input names and predicates on input JSON, checked before live requests and again by rescore. |
| `.constants` | Module to constant name to `{value, source}`. Overrides must match the module default's type; derive bounds from reference spread, never copy another model's limits. |
| `.cohort_args` | Extra argument lists keyed by cohort. |
| `.informational_reason` | Nonempty reason to skip execution and judgment. Not passing evidence. |
| `.waivers` | Optional recorded decisions the judge applies: each covers a failing predicate (`predicate`: the rule's `path` and `op`, `ceiling` looser than its bound) or a teacher category (`category`, `metric`, `ceiling`), with a nonempty `reason` and `decision`. The check is then `waived`, never `pass`, and the report lists each waiver applied ([waivers](../../docs/check-kinds.md#waivers)). The default configuration's alone; a configuration declares its own. |
| `references.<name>.path`, `.sha256`, `.schema` | Reference location (a repository path or the same environment-backed object as an input), digest of its exact bytes and expected schema. `teacher` is required by the teacher worker; the media worker uses `qualification.media.reference`. |
| `.variants` | Unique variant names, as the runner emits them. Required for teacher and case-format media references. |
| `.workload`, `.workload_sha256`, `.capture_sha256` | Optional workload location and workload/capture pins. Without `workload`, a bound reference uses `<reference stem>.workload.json` beside it. |
| `.checkpoint_keys` | Provenance fields to compare: `repository`, `revision`, `served_model_name`, `source_lock_sha256`, `checkpoint_identity`. |
| `inputs`, `receipts` | Inputs map names to locations; receipts names the output directory. A location is a root-relative string, `null` for unused inputs, or `{env, default?, path?}`. Keep output bases outside the repository. |
| `qualification.teacher_reference_kind`, `.teacher_receipt`, `.teacher_capture_schema` | `exact` (the schema also accepts `prompt-logprobs`, which no check captures), receipt ID and expected capture schema. New numerical references use bound GPU runner outputs. |
| `.verify_buckets`, `.chars_per_token` | Supported speculative verify widths and a positive prompt-length estimate (receipts still record actual token counts). |
| `.mechanics_only` | Optional boolean, default false; true marks fixture mechanics that never qualify a model. External gate configs must state it. |
| `.checkpoint` | Required `lock`, `repository`, `revision`, `served_model_name`, `source_lock_sha256`, `checkpoint_identity`. Optional `pin` and `registry`, together, select a registered checkpoint variant. |
| `.media` | `format`, `reference` and `receipt` select the capture format, reference key and receipt ID. `cases` also needs `capture_schema`, `expectations` and positive `audio_near_tie` (required even for image-only cases). |
| `.media.expectations` | Optional `files` and `kinds` maps, each entry naming a text `field` and boolean `single`. |
| `.media` with `image-probes` | Instead of case fields: `stop_token`, `items` (`short` or `free-text`), `expected` (nonempty string lists) and `fixture_images` (paths); optional `thinking` (default false), the thinking switch the probe requests send, for a model whose frontend refuses disabled thinking. |
| `.prefix` | Optional prefix policy: `family`, booleans `prompt_end`, `media`, `acceptance`, positive `block_tokens`, `prefill_granule`, `anchor_boundary`, nonempty `anchor_labels`, and `reference_schema`. |
| `.needle` | Optional `format` (`context-files` or `prompt-set`), positive `lengths`, `match` (`whitespace` or `alphanumeric`), and `frozen` prompt file (required for `prompt-set`). |
| `checkpoint` | Optional descriptive `repository`, `revision`, `status`; not the binding under `qualification.checkpoint`. |
| `limitations` | Strings stating the qualification scope and missing evidence. |
| `renamed_checks` | Former ID to current ID. |
| `preview` | Optional correctness runner settings (above). |

### Capabilities and required evidence

`capabilities` describes the served profile. It is optional for older v1
files; when present every field below is required. Both loaders refuse
missing coverage, including checks omitted from `cohorts.full.checks` and
checks marked informational. Module names identify the required checks, so
each model chooses its own IDs.

| Capability field | Value and required hard check modules |
|---|---|
| `text` | Boolean; `teacher`. |
| `frontend` | Boolean; `frontend_goldens`. |
| `api` | Boolean; `api`. |
| `structured` | Boolean; `structured`. |
| `media` | Unique list of `image`, `audio`, `video`; each modality needs its own nonempty reference inventory. Case-format media declares `qualification.media.coverage`, mapping each modality to reference item IDs; image probes cover images only. |
| `speculation` | `none`, `lookup`, `mtp` or `dflash`; enabled speculation requires `spec_verify` and `verify_exact` with distinct control and candidate inputs, mandatory predicates, positive coverage and observed drafting. |
| `prefix_cache` | Boolean; `prefix_exact` scores bitwise equality. |
| `preemption` | Boolean; `preempt_compare compare --exact`, without `--baseline`. |
| `long_context` | Boolean; `needle` with the declared context lengths. |
| `stability` | Boolean; a local `soak` with timed host-RSS and device-memory samples, bounded growth and traffic in every enabled mix. A `--remote-only` soak cannot satisfy it. |
| `kv_offload` | Boolean. True requires model-level `kv_offload` facts and hard plain/pressure checks in the full suite. Serving offload remains off by default. |

The soak must run the declared duration and concurrency, with samples
spanning the interval (no gap over 60 seconds) and at most 512 MiB growth in
both host RSS and device memory. A server that fills a bounded cache while it
warms up (such as pinned prefill graphs) declares the soak check's constants
`WARMUP_SECONDS` and `CACHE_ALLOWANCE_MIB` (host and device, the cache's
bound): the growth from the first sample at or after the warm-up must stay
within the limit, and the growth over the whole soak within the limit plus
the allowance (`soak.memory_rule`; the warm-up is capped at a quarter of the
duration). Undeclared, the rule is the plain one. A shorter saved soak is diagnostic only. A
capability the server does not serve is `not_applicable`, never a silently
skipped check. Capability coverage is a declaration check; the gate run must
still prove every predicate and input.

**Producing references.** Use the
[GPU reference runner](../../docs/reference-runner.md#prepare-and-run):
implement the adapter and checkpoint lock, prepare workloads and media
tensors, run the required modes on idle GPUs, then bind scores, workload
and refset with `tools.refrun emit`. Put the printed SHA-256 values, schemas
and variant names in the declaration, then adopt the inputs
([adopted qualification inputs](#adopted-qualification-inputs)) and commit
the declaration and lock together. Never invent pins.

**Schema evolution.** Within `chopinfer-qual-model/v1` fields are only added,
as optional fields whose absence keeps the previous behavior, so an older
declaration compiles to the same policy. Any change of meaning (a field
becoming required, removed, renamed, retyped, or changing how a gate judges
or selects checks) takes a new schema version. Readers refuse unknown
fields, and the Python loader and `chopinfer eval` accept the same field set
per version.

### Check kinds

Every declared check has the kind `tools/qual/kinds.py` registers for its
module; a module no kind names is a `receipt` check, judged by its declared
predicates alone. The six shared kinds (`teacher_forced`, `media_forced`,
`greedy_pin`, `determinism`, `api`, `structured`) are specified in
[check kinds](../../docs/check-kinds.md), with conformance vectors in
`tests/qual-vectors/` (written by `tools/qual/vectors.py`). Their `verify`,
`score` and `judge` steps call `chopinfer eval rescore`. `chopinfer eval`
also captures the teacher_forced, api and structured checks of a model
declaration (`eval run`) and the runner's determinism and greedy records
(`eval capture`, the only greedy record writer). Python keeps the media
capture (`multimodal.py`), the offline determinism and greedy comparisons
(`greedy_equality.py compare`), the determinism record of
`tools/ab/serving` and the serving smoke that runs through it
(`determinism.py record`: those serve checkpoints the base declaration need
not pin, which `eval capture` binds the server to) and the api and
structured mechanics captures.

Each kind has the same interface (`capture`, `reference`, `verify`, `score`,
`judge`). A live gate scores and judges the receipt its capture just wrote;
a rescore admits, verifies, scores and judges a stored one.

```sh
python3 -m tools.qual list --model qwen38
python3 -m tools.qual check teacher --model qwen38 --url http://127.0.0.1:8441 --inputs inputs.json
python3 -m tools.qual rescore --model qwen38 "$CHOPINFER_DATA/qual/qwen38/<run>-release-qwen38-quick"
```

`check ID` runs one declared check as its cohort (`--cohort`, default quick)
runs it, resolving prerequisites first; `--with ID=RECEIPT` supplies an
earlier check's receipt and `--with-raw ID=DIR` that check's raw directory.
A check reads a file an earlier check wrote as `{check_raw[ID]}/<file>`,
never `{raw}/<file>`: a gate run's checks share one `{raw}`, but each
`check` run (and so each release_preview row) has its own. `{check_raw[ID]}`
is the directory check ID wrote to once it passed earlier in the run; a
check whose producer has not passed is refused before it runs. It writes `<run id>-check-<model>-<ID>/` with the
policy, the receipt and `check.json` (`chopinfer-check-run/v1`).

`rescore PATH...` needs no server and never writes into the run. A PATH is a
gate run (its directory or `summary.json`) or a single receipt. A gate run
is judged against the current declaration (`--limits current`, the default)
or against its recorded policy and the references at its recorded revision
(`--limits recorded`). Every check the cohort selects gets a row: one the
run lacks is `insufficient`, and a worker that exited nonzero stays `fail`,
its receipt (or the record its eval report holds) shown as `receipt_status`.
`chopinfer eval run` also exits 1 for a completed verdict that failed a
declared predicate: the gate links that record and copies its capture as it
does a passing one, the row fails as `threshold predicate failed`, and a
rescore judges the record again. Only a complete report counts as that
verdict: the integer exit 1, no run failure, one check judged `fail` with
the record's digest, its metrics, eval's own threshold error and every
declared predicate whole, at least one failed, each measuring over those
metrics what the gate's scoring of the record measures. Any other exit 1,
and a row whose gate judge passed what eval failed, stays a failed run.
A check's `input_rules` are judged as the live preflight judges them: a
record they refuse fails the row, and an input the run did not record (or
whose file changed since its recorded SHA-256) leaves it `insufficient`.
The receipt and reference file stand for the inputs they are read from.
`--json FILE` writes the report (`chopinfer-rescore/v1`). Exit statuses are
in [check kinds](../../docs/check-kinds.md#exit-statuses).

## Gate inputs

Declared paths resolve from the checkout root; null inputs are unused.
`receipts` is an output, not a download.

| Declared input | Availability |
|---|---|
| Both models: `fixture_root`, `frontend`, `chat_renders`, `exact`, `teacher_reference`, `spec_teacher`, `media_cases`, `media_capture`, `frontend_capture`, `tokenizer_goldens` | Shipped in the checkout. Provenance and SHA-256 of the Qwen references are in its [fixture README](../../fixtures/qwen38/README.md#teacher-gate-reference). |
| Qwen: `needle` | `$CHOPINFER_DATA/qwen38-refs/prompts/prompts.json`. Generate with `fixtures/qwen38/capture/prompts.py` ([capture README](../../fixtures/qwen38/capture/README.md)); it also needs MiMo's `context-524288.json`, both tokenizers and the pinned wikitext cache. |
| Qwen: `greedy_baseline` | The approved C1 greedy baseline under `$CHOPINFER_DATA/qualification/qwen38/greedy/`, release preview only. |
| Both models: `gsm8k` | `$CHOPINFER_MIMO_FIXTURES/gsm8k/{examples,manifest}.json`: the first 200 test examples of `openai/grade-school-math` revision `3101c7d5072418e28b9008a6636bde82a006892c`. |
| Both models: `media_fixtures`; MiMo: `needle` | `$CHOPINFER_MIMO_FIXTURES/functional/`: `manifest.json`, hashed PNG frames, WAV/MP4 media and `context-{8192,131072,524288,1048576}.json`. |

Limit: this checkout has no generator or public distribution for the frozen
GSM8K manifest and the functional inputs. An empty directory or a new
manifest does not reproduce them.

Frontend goldens use HF templates and tokenizers; answer-dependent API
shapes use F32 greedy reference answers. Long context uses its declared HF
reference. Missing declared references stop a check before its first request.

## Reference pins and bindings

### Checkpoint binding

`qualification.checkpoint` pins the checkpoint (lock, repository, revision,
served model name, source-lock digest and the direct-load
`checkpoint_identity`, the SHA-256 of the safetensors names, lengths and
headers that `serve` logs). `references` pins each exact reference (file
SHA-256, schema, workload or capture digest, variants); its provenance must
name the pinned checkpoint. A reference with other bytes, even reformatted,
is refused.

`chopinfer eval` (the teacher, api and structured checks and the
determinism and greedy records it captures), `multimodal.py`,
`tap_binding.py capture`, `verify_exact.py record`, `prefix_exact.py
record`, `spec_verify.py record` and a scored `gsm8k.py` screen bind the
server they score through its
`/health` identity ([what `/health` reports](../../docs/deploy-common.md)):
it must name the pinned `checkpoint_identity` and a startup nonce, and serve
the pinned model name. `--server-log`, when given, must be that live server's
startup log. Every response carries `x-chopinfer-checkpoint-identity` and
`x-chopinfer-startup-nonce`; each tool refuses the run if `/health` changed
or any response named another process.
`spec_verify.py` and `gsm8k.py` also record the topology and speculation
`/health` reports (`serving`), and `spec_verify.py compare` refuses bound
arms of different checkpoints, names, topologies or KV dtypes, a
speculative arm that does not draft or a control that does, two arms with
one startup nonce, and a pair of which only one arm is bound. A record that
claims a binding must name a checkpoint identity, startup nonce, served
name, tensor parallel size, KV dtype and speculation mode and tokens, or the
comparison refuses it. The
identity covers file names, lengths and headers, not tensor bytes;
`chopinfer verify --model` checks those.

### Reference cache bindings

`common.pinned_reference` reads every pinned reference a gate loads
(teacher, multimodal, exact and prefix verification, equality checks) by its
declared SHA-256, schema, checkpoint provenance, variants and workload
digest. Its runner binding is part of those pinned bytes: the gate checks
that it names the gate's checkpoint lock and records its refset
(`reference_binding`), without verifying it against the reference cache,
the working tree's adapter and numerics files or the workload again on each
read. A reference pinned without a spread variant (`variants` `["exact"]`,
a floor-only reference) is the exception: its binding is its producer
authentication, so each read verifies it whole, workload included.
Adoption verifies a candidate's whole binding before pinning it
(`common.verified_binding`), and `tools.refrun verify-binding` re-verifies
one on demand. A mismatch reads `stale_reference: FIELD`; repair the input or
regenerate and repin, never remove a binding. The
[runner guide](../../docs/reference-runner.md#cache-and-fixture-bindings)
describes fixture assembly, the binding fields and the standalone verifier.

### Adopted qualification inputs

`models/<model>/qual-references.lock.json` (with each configuration's own
entries under `configurations`; a variant declaration file's:
`qual-references.<variant>.lock.json`; an export's beside its declaration)
pins the bytes gates read:

```sh
python3 -B -m tools.qual refs status --model mimo26
python3 -B -m tools.qual refs adopt --model mimo26 [--replace-sha256 OLD_LOCK_DIGEST] [--dry-run]
python3 -B -m tools.qual refs verify --model mimo26
```

`verify` authenticates every declared input's bytes, `status` also prints
the lock digest, and `adopt` records new or changed inputs (idempotent; a
refresh requires the old lock digest and lists the changed entries;
`--dry-run` lists them and writes nothing). `adopt` replaces only the
declared entries and the objects only they pinned. It keeps the
`configurations` section, `operator:` records, an entry's own fields
beside its pin (role, producer policy, reason) and every other object as
they are, in the lock's own key order, so adopting an unchanged
declaration reproduces the lock byte for byte. It refuses a lock it cannot
keep whole (an unknown section, a configuration holding more than entries
or pinning a missing object, an entry with its own fields whose path
changed). A configuration's entries are edited in the model lock;
`adopt --model MODEL.CONFIG` is refused.

A lock entry pins a file's SHA-256, or each member of a directory. A
directory entry's `scope` selects only the files gates read
(documentation, capture scripts and crate goldens are excluded); `adopt`
keeps the scope an entry already has and pins every file of a new
directory, and a context-files needle's scope comes from its declared
lengths. Adoption verifies a bound reference's runner binding, its workload
and preprocessing records before pinning their bytes, and an object records
what proved it (`production`). Every numerical reference
(`qualification.references`) needs a runner binding: `adopt` and gate
admission (`verify`, and the release gate before it runs any check) refuse
a reference object whose record is not `runner-binding`
(`refs.gate_reference_producers`), without verifying the binding again.
Other records, like an older lock's adoption or frontend and operator
evidence records, are information that no reader admits or refuses bytes
by. A layer reference run is a declared input the lock pins like any other. Gate
readers authenticate the bytes they consume and name the lock entry on a
mismatch; gate runs require only the inputs their selected checks read.

## Teacher-forced accuracy

The teacher check sends every text of the declared teacher reference as a
token-ID prompt and scores the server's prompt logprobs against the
reference's `exact` variant (the F32 run of the vendor model code on the GPU
reference runner; its BF16 and f64 runs are the spread). `chopinfer eval run`
captures and scores it (the `teacher_forced` kind):

```sh
target/debug/chopinfer eval run --model models/mimo26 --url http://127.0.0.1:8101 \
  --check teacher --server-log SERVER.log
```

The reference must cover exactly the frozen texts, in order, with every
pinned variant complete. Mean |dlogprob| is held to 0.15 (0.10 for at least
1,000 positions), decisive top-1 flips to 8%, near-tie flips to 8% and all
flips to 10%, pooled and per text category, each raised only by the
reference's own spread. The policy and per-model limits are in
[numerics](../../docs/numerics.md#serving-qualification-gates); the exact
computation is the [spread rule](../../docs/check-kinds.md#the-spread-rule-variant-spreadv2).

- `--reference-name NAME --receipt ID` in a check's arguments gates another
  pinned text set the same way (Qwen's `long_context`: the last 1,024
  positions of a 32,768-token prompt).
- `--capture-out PATH` in a check's (cohort) arguments also writes the
  compact capture paired comparisons read; `release_gate.py` copies it from
  the eval report to PATH.

`teacher.py` keeps the scoring helpers and the constants a check may set
(`checks.<id>.constants.teacher`) that `paired_teacher.py`,
`teacher_arms.py`, `verify_exact.py` and `prefix_exact.py` use; it has no
capture of its own.

`teacher_arms.py --arm NAME=RECEIPT=CAPTURE --arm NAME=RECEIPT=CAPTURE`
compares two serving configurations of one checkpoint (such as FP8 and BF16
KV) on a host that serves one at a time: each arm is a passing live teacher
receipt and the capture its run wrote with `--capture-out`, typically one
recorded by an earlier gate run and one by this run (the declaration names
this run's teacher capture as `{check_raw[teacher]}/teacher-capture.json`,
the teacher check's own raw directory). An arm is refused unless its
receipt passed live against the pinned checkpoint, its capture holds the
served rows the receipt recorded, is the whole file the receipt's run wrote
(the receipt records its `capture_sha256`) under the same server identity,
and both arms name the pinned teacher reference, run one build (package
identity, version and source commit) and score its workload (text names,
prompt token digests, score starts, every scored position). The
`teacher-arms` receipt reports each arm against the reference, the second
arm's mean |dlogprob| minus the first's (pooled and per category) and the
arms against each other; it passes when the arms are bound and comparable,
and gates no margin between them.

### Top-1 near ties

A flip is a scored position where the server's top-1 differs from the
reference's. It is a near tie when the reference's own gap from its top-1 to
the served token, read from its stored top-k, is at most 0.3 nats
(inclusive, `limits.near_tie`); otherwise it is decisive. The teacher and
media checks share one implementation, and the check kinds recompute the
flips from the served rows each receipt keeps
([contract](../../docs/check-kinds.md#the-near-tie-rule-top1-near-tiev1),
[policy](../../docs/numerics.md#top-one-near-ties)).

## Speculative decoding comparison

`spec_bench.py` runs 16 frozen GSM8K and 16 chat questions at C1, C4 and C16
(`--concurrencies`), greedy, 64 output tokens by default, and fails if a
request stops early. Run each arm on a freshly started server with the same
model alias, `--run-id` and budget:

```sh
python3 tools/qual/spec_bench.py --arm chopinfer-ar --url http://127.0.0.1:8311 --model MiMo-V2.6-Flash-MOPD --run-id trial-001
python3 tools/qual/spec_bench.py --arm chopinfer-spec --url http://127.0.0.1:8311 --model MiMo-V2.6-Flash-MOPD --run-id trial-001
python3 tools/qual/dev/spec_summary.py --run-id trial-001
```

The receipt stores `/metrics` scrapes around each cohort. The shared
comparison is mean acceptance length, `1 + accepted / verify steps`, with
per-position acceptance and throughput per prompt group.

### Verify rows against autoregressive rows

`spec_verify.py record` sends 32 greedy completions one at a time and stores
each token's logprob and top five alternatives. Record the same `--run-id`
with speculation on and off, then compare:

```sh
python3 tools/qual/spec_verify.py record --arm chopinfer-spec --url http://127.0.0.1:8331 --run-id verify-001
python3 tools/qual/spec_verify.py record --arm chopinfer-ar --url http://127.0.0.1:8331 --run-id verify-001
python3 tools/qual/spec_verify.py compare --spec SPEC.json --ar AR.json
```

The gated **copy set** prompts a frozen text followed by its own first half,
so prompt lookup drafts the second half and nearly every row is a verify
row: over each request's common prefix, mean |dlogprob| at most 0.15 and
top-1 disagreement at most 8%. The **natural set** (GSM8K answered step by
step) is reported. Only the common prefix and the first divergent position
are scored: later positions condition on different histories. Each set's
`coverage` reports the emitted, scored and unscored positions per request
and in aggregate, and is `complete` only when no request of either arm
leaves a position unscored. Coverage is reported, not gated: a set can pass
its bounds while an early divergence leaves most of a request unscored.

`verify_exact.py` scores verify, decode and prefill rows at the same
positions against the exact reference. The server runs with
`serve --debug-reference-tokens FILE` (from `verify_exact.py references
--out FILE`), which forces the reference continuation; `record` scores each
text at every verify depth and `report` pairs an AR record with verify
records. Mean error against exact may worsen by at most 0.005, and excess
top-1 disagreements are bounded by two standard deviations of the discordant
count. Forcing replaces only host drafts, so record a server with a model
drafter (DFlash, MTP) with `--pooled-decode`.

`report --control AR.json` (repeatable) adds further AR draws, each held to
what `--ar` must meet: K 0, no pooled decode, no drafting, the same
reference, and bound live when the reference is pinned. A position then
pairs only where every draw scored it, and its AR error is the mean over
`--ar` and the controls, so the 0.005 margin reads against that mean; the
top-1 counts keep `--ar`'s rows. Coverage requires each control to score
every expected position. Each record lists the draws and their mean errors
(`controls`) and keeps the comparison against `--ar` alone
(`single_control`) as a diagnostic. Without `--control` the report is
unchanged.

`report --pool-verify` treats the `--verify` records (all of one row kind)
as draws of one candidate, for example pooled speculative rows at several
item buckets. At the positions every draw pairs, the mean of their row
errors is held to the 0.005 margin against the AR error; `pooled_candidate`
lists each draw's arm, concurrency and mean error and the pooled gap with
its text-clustered CI95, which is reported, not gated. Each record must
still pass its own top-1 bound, coverage and bucket checks, and its own
mean gap stays in `records` as a diagnostic.

### Speculative continuations against the reference

`spec_continuation.py` scores each arm's own continuation against the GPU
reference runner on the history that produced it, so positions after a
speculative/AR divergence are scored rather than dropped. Declared as
`spec_continuation` for a profile without exact AR/speculative equality
(its preview `spec_greedy` rule is informational), it gates that profile
under spec-continuation.v1 ([numerics](../../docs/numerics.md#speculative-continuations)).
The correctness runner records both arms on the servers it starts, runs
`reference` for each arm once those servers have stopped, then judges the
check without a server (`checks.py check --offline`).

```sh
python3 tools/qual/spec_continuation.py record --arm chopinfer-spec --url URL --run-id cont-001 --server-log LOG
python3 tools/qual/spec_continuation.py record --arm chopinfer-ar --url AR_URL --run-id cont-001 --server-log AR_LOG
python3 tools/qual/spec_continuation.py workload --record SPEC.json --reference-model MODEL --out SPEC.workload.json
"$CHOPINFER_REFRUN_PYTHON" -B -m tools.refrun run ... --workload SPEC.workload.json --modes f32,bf16 --output SPEC_REF
python3 tools/qual/spec_continuation.py compare --spec SPEC.json --spec-reference SPEC_REF \
  --ar AR.json --ar-reference AR_REF
```

`reference --record SPEC.json --checkpoint DIR --out NEW_DIR` writes the
workload, runs the reference runner on the visible GPUs in the modes and
under the lock, adapter and dequantization of the gate's pinned teacher
reference (input `exact`, its `binding`), and writes `execution.json`: the
command, visible devices, GPUs, load, timings, exit status and, per mode,
the entries computed by this run and those taken from the reference cache
with the provenance they were stored with. `compare` (with the gate's
declaration in `CHOPINFER_QUAL_CONFIG`) reads a mode file only when its
policy is that mode's run of the pinned reference (name, dequantization,
digest), refuses an entry whose key does not share the pinned binding's
model, checkpoint lock, reference code, adapter, environment, outputs and
mode policy, and refuses arms that are not two bound processes of the
gate's pinned checkpoint and served name, of one install (the binary's
source commit and the direct package and metadata each /health reported)
and one profile apart from speculation, the candidate drafting and the
control not. A request that
emitted no valid token IDs, or was refused for any reason, leaves its set
incomplete. The receipt keeps `execution.gpu` (each reference run) apart
from `execution.cpu_scoring`.

`workload --sets copy,natural,open` writes only those sets, so one arm's
reference can be split over bounded runs; `--spec-reference` and
`--ar-reference` repeat to merge them.

`record` sends four frozen sets: spec_verify's copy and natural prompts and
the benchmark's chat prompts greedily (64 tokens), and the chat prompts with
16 open-ended prompts at temperature 1, top_p 1 under two seeds (128
tokens). A set's requests run together (`--concurrency` caps them), so verify
rows come in a loaded server's row buckets. It stores the prompt and emitted
token IDs, each emitted token's raw logprob and the top 20 by label. `workload` forces each request's
emitted IDs after its prompt. `score` and `compare` require, per request,
reference rows that read its own prompt and emitted IDs (anything else is
refused and leaves coverage incomplete), and report coverage per request
and in aggregate, mean |dlogprob| at the emitted token, top-1 flips on the
greedy sets by the near-tie rule, the spread rule's limits from the BF16
run over the same positions, positions by whether the reference's top 5 or
top 20 holds its 0.95 nucleus, and on the sampled set the mean of
logprob_arm - logprob_reference, an unbiased estimate of the mean KL
divergence from the reference over the whole vocabulary, with its Pinsker
bound on total variation.

Unit checks: `python3 -B -m unittest discover -s tools/qual -p 'test_spec_*.py' -v`.

## Multimodal serving

Against a server whose checkpoint has its encoders (`chopinfer serve
--model` plans them automatically):

```sh
CHOPINFER_QUAL_CONFIG=models/mimo26/qual.json python3 tools/qual/multimodal.py --url http://127.0.0.1:8361 --timeout 120
CHOPINFER_QUAL_CONFIG=models/qwen38/qual.json python3 tools/qual/multimodal.py --url http://127.0.0.1:8441 --timeout 300
python3 tools/qual/grounding.py --url http://127.0.0.1:8361 --timeout 120 --photo PHOTO.jpg
```

The server must be bound to the declaration's checkpoint pin
([checkpoint binding](#checkpoint-binding)), prompt token IDs and
`usage.prompt_tokens` must match the reference, and answers must contain the
fixture content where the prompt determines it (for example `blue` for the
blue square). The declared format (see
[check kinds](../../docs/check-kinds.md#declared-media-capture-formats))
selects how answers are judged:

- **Cases (MiMo).** The prompts of `fixtures/mimo26/multimodal/reference.json`
  are replayed and scored against the reference-code scores in
  `exact-reference.json` beside it. The reference answer, teacher-forced, is
  held to the teacher limits per case. The greedy answer is walked along the
  reference answer: every served token up to and including the first
  departure must be the exact top-1 or within 0.3 nats of it (in the top 5);
  audio permits 2.3 nats.
- **Image probes (Qwen3.8).** The seven probes of
  `fixtures/qwen38/frontend/images.json` are sent as captured; probe PNGs are
  regenerated by `fixtures/qwen38/capture/images.py` (`--images DIR` keeps
  them) and must match the captured SHA-256. The reference,
  `fixtures/qwen38/image-probe-references.json`, holds HF transformers' F32
  answers from the GPU reference runner, teacher-forced in F32, BF16 and f64;
  its workload under `$CHOPINFER_DATA` is needed to verify the binding. Short
  answers follow the greedy rule above. Free-text descriptions are scored
  teacher-forced with per-probe spread limits (mean |dlogprob| at least 0.15
  and at most 0.30, plus the near-tie flip limits); their greedy agreement is
  reported only, since their first divergence follows BF16 rounding.
  The greedy request asks for 64 output tokens, or the reference answer's
  length when longer, so a model that reasons first reaches its answer; the
  row keeps the served reasoning under `reasoning`. The probe
  refuses to start without the `chopinfer eval` scorer (`CHOPINFER_EVAL` or
  the checkout's target directory). `--probes NAME ...` runs a subset.

`grounding.py` needs no reference: red, green and
blue squares must each rank their own color first, and the optional photo's
description must be at least 1 nat more likely with the image. The
record-only `color_probe.py` compares observed runs (`--compare A B`) and
writes probe images with `--write DIR`. The declared `media_preprocessing`
check qualifies preprocessing against its HF reference.
Offline logic checks:

```sh
cd tools/qual && python3 -B -m unittest test_multimodal test_grounding test_encoder_parity test_color_probe
```

### Tap captures

A tap gate (`encoder_parity`, `qwen38_layers`, `text_layers`) counts engine taps only from
a capture bound to the process and request that wrote them. Serve a
diagnostic-taps bundle with `CHOPINFER_TAPS_DIR` set,
`CHOPINFER_REQUEST_TRACE=1`, prefix caching off and nothing else in flight,
then:

```sh
python3 tools/qual/tap_binding.py capture --url URL --server-log LOG \
  --taps-dir DIR --out OUT --request NAME=PATH=BODY.json ...
```

It sends each request once under a fresh `X-Client-Request-Id` and cache
salt and copies only the new steps that name the live startup nonce and that
request (`OUT/engine`, `OUT/tap-capture.json`). The gate then admits the
capture only if the candidate at `--url` has the capture's `/health` serving
identity, startup nonce included, and every step holds its recorded digests,
so run the tap checks while that server still serves.

### Text layers

The `text_layers` check (`text_layers.py`) is the model-neutral form of the
layer comparison: one reference sequence, captured once from a
diagnostic-taps server through `text_layers.py capture`, against the
reference runner's F32 taps with its BF16 taps as the floor, through
`tapdiff.py`. The declaration names the model; its adapter is
`models/<model>/reference.py`, its checkpoint lock the gate's pin, and the
check's `MODES` constant the dequantization each reference mode must use.
The check's `ADAPTERS` constant gives a mode another adapter (a floor that
models a served KV format the plain modeling lacks, say); one runner run
then cannot hold both modes, so the reference holds one run per mode,
`f32/` and `bf16/`, each run with its mode's adapter.
The reference run is the declared `layer_reference` input, whose files the
reference lock pins; each mode's cache keys must name that mode's adapter
and the runner numerics files with their current digests.
`first` names the first tap outside the floor. A declaration's
`REQUIRED_TAPS` (tap names, `{nn}` over the layers `taps.json` gives) must
all be compared, else the status is `insufficient`; without it the report's
`scope` is the intersection of the two sides' taps, a diagnostic. The
check's constants are read from the declaration, so the standalone
`text_layers.py capture` and comparison apply the gate's policy.

A declaration's `ROUTER` constant (`score` `sigmoid` or `softmax`,
`top_k`, `n_group`, `topk_group`, and `bias`, the declared input holding the
correction biases) has tapdiff judge routing flips by route-near-tie.v2
([numerics](../../docs/numerics.md#routing-near-ties)): a tap that fails
only its whole-tap aggregate passes when the fewest rows behind a near-tie
routing flip, largest error first, explain it, each within FLOOR_RATIO
times the BF16 run's worst row at that tap and plausible (finite, its norm
within a factor of FLOOR_RATIO of the F32 row's, a positive cosine with
it). The receipt keeps the flips,
margins and removed rows (`routes`, each row's `route_explanation`). The
bias input is a `router_bias.py extract` sidecar (`chopinfer.router-bias.v1`):
each layer's bias tensor, read once from the checkpoint after its index and
shards are verified against the checkpoint lock. The check reads it as the
reference lock authenticates it and refuses one of another model or
checkpoint pin. Both layer comparator modules take the same `ROUTER`
constant.

### Encoder parity

The `encoder_parity` check compares every encoder output row the engine
computes, and MiMo's RVQ codes, with the generic encoder reference
([runner guide](../../docs/reference-runner.md#encoder-reference)) over the
same media, through `tapdiff.py`. The encoder declarations in
`models/<model>/contract/encoders/*.json` define the taps. Its input
`encoder_parity_run` is a directory holding `parity.json`
(`chopinfer.encoder-parity-run.v1`): `model`, `engine` (a
[tap capture](#tap-captures) with `CHOPINFER_TAPS_SITES=enc.*`, each workload
prompt sent once under its sequence name), `references` (one reference run
per declaration in the `exact`, `policy` and `reference-bf16` modes),
`workload` and `source`. Run:

```sh
python3 -B tools/qual/encoder_parity.py taps --run DIR --url URL
```

Tapdiff (under `CHOPINFER_REFRUN_PYTHON`) takes `exact` as the reference and
`reference-bf16`, unfused BF16 PyTorch, as the spread:

- `enc.vision.out`, per item. Direction: with d = |engine/|engine| -
  exact/|exact|| per row and its floor the same for reference-bf16, every
  row needs d <= 3 x max(floor, 2^-8) and the median d / (3 x floor) at most
  1. Scale (largest row-norm error per item): an F32-policy encoder
  (`f32-pairs.v1`) is held to its policy run's own error, at least 0.1% and
  rounded up to 0.1%; a BF16-policy encoder (`bf16-legacy.v1`) to the larger
  of reference-bf16's and the policy run's, rounded up to 0.5%
  ([why](../../docs/numerics.md#mimo-vision-row-scale)).
- MiMo's `enc.audio.out`: relative L2 per row by the same rules, over rows
  whose RVQ codes all equal the exact run's.
- The audio encoder's `enc.audio.x.codes`: agreement with the exact codes, overall and in
  the first stage, at least reference-bf16's less 3 percentage points.

An encoder whose every input kind the declaration lists in
`capabilities.unqualified_media` (served, not qualified in-tree) has no
required rows. Its encoder parity inputs must pin no reference for it, and
the receipt's `unqualified` names it with the declared reason. An encoder
with one qualified input kind keeps every rule above. MiMo serves audio but
does not qualify it in-tree, so its gate requires no `enc.audio.*` rows.

A model can supply `models/<model>/encoder-parity-inputs.json` to produce
its declared encoder check in the correctness runner. The file pins existing
reference runs, their historical producer commits, the workload and tensor
bytes, and one-token request bodies. `encoder_capture.py preflight` admits
those inputs on CPU; `record` uses `tap_binding` to capture their actual
requests on a diagnostic arm with all provided modalities and prefix caching
off. The gate runs before that same server exits. Capture source, process and
request identities remain distinct from the tool source. Missing modes,
changed pins or incomplete declared encoder coverage refuse; this producer
never generates replacement references or changes parity thresholds.

Every tap must hold compared rows and tapdiff must report a match. Images
served from the media cache run no encoder and are covered by fake-host tests
instead. `dev/encoder_parallel.py` compares encoder bytes across ranks and
parallel modes; it does not replace this gate.

`encoder_parity.py lab --model M --engine TAPS --reference REF ...
--workload W --json RECEIPT` applies the same tapdiff rules to a lab capture
of the encoder alone (each step's request named as its workload sequence),
against reference outputs that may split the modes between runs. A lab
capture is bound to no serving process: it is evidence for the encoder
kernels and weights, not the `encoder_parity` check.

## Downstream evals

`evals.py` runs paired downstream evals against any OpenAI-compatible URL,
sending every arm byte-identical greedy chat requests:

- **GSM8K:** the 1,319-question test split, reasoning on, 8,192 output tokens.
- **MMLU-Pro:** 1,200 test questions across the 14 categories, fixed seed,
  zero-shot, reasoning on, 8,192 output tokens.
- **Long context:** a synthetic RULER-style set from WikiText-103 (multi-key
  and multi-value needles, variable tracking) at 32K and 128K, 50 items per
  task and length, reasoning off, 128 output tokens.

A response that stops for `length` is wrong.

```sh
python3 tools/qual/evals.py fetch
python3 tools/qual/evals.py run --arm baseline --run-id evals-001 --url http://127.0.0.1:8102 --concurrency 32
python3 tools/qual/evals.py run --arm candidate --run-id evals-001 --url http://127.0.0.1:8381 --concurrency 32
python3 tools/qual/evals.py compare --run-id evals-001 --arms candidate baseline
```

`fetch` downloads the datasets once into `$CHOPINFER_DATA/evals` (or
`CHOPINFER_EVALS_DIR`) with their revisions and SHA-256s. `run` resumes from
`runs/<run-id>/<arm>.jsonl`, reusing an answer only when its row names the
same producer (the server's checkpoint, package, metadata, version and
source identities, its serving profile (tensor and context parallelism,
speculation, KV cache dtype, prefix cache, sequence capacity) and resolved
model options, the served model and its modalities, the checkpoint pin a
`--server-log` run was bound to (none for an unbound run), and the request
settings) and request body, and asking every other item again; the receipt
records the producer and how many rows it reused (`resumed`). A run reuses
only rows a capture confirmed under its own producer sealed: once the live
server is confirmed, `run` appends a segment to `<arm>.seal.json` with the
results file's length and SHA-256, that producer and the bound process's
evidence, so an unbound capture's rows never serve a bound run. Rows a refused or
unfinished capture appended after the seal are moved to
`<arm>.unsealed.jsonl` and asked again, and a results file whose sealed
bytes changed refuses the run. `run --server-log LOG` binds the
live server to the gate's checkpoint pin and that startup log and confirms
at the end that the same process answered every request; `--limit`,
`--suites`, `--long-lengths`, `--long-concurrency` and `--top-logprobs`
narrow or annotate it. `compare` refuses items whose request bodies differ,
applies an exact McNemar test per suite and group, and fails if the first arm
is significantly worse (p < 0.05) on any suite. Speculation need not
reproduce AR text, so compare it on and off by scores. Each arm's receipt
records its results file's SHA-256, and `compare` refuses an arm whose
results changed since. `compare --receipt ARM=RECEIPT` takes an arm by its
receipt instead of finding it by `--run-id`: the candidate arm an earlier
check run recorded under its own run ID (a declared `evals_compare` names its
`evals_candidate` check's receipt), or an arm recorded in another gate run (a
pre-recorded reference arm on one GPU host). The receipt must be that arm's,
recording its run ID, its results file and that file's digest, and
`compare` reads the results file it names, held to the digest; the compare
receipt records each arm's run ID (`arm_runs`) and the number of items both
arms answered (`items`, the coverage a gate admits). `compare --stored
ARM=RECEIPT` takes an arm of an earlier run (the runner's stored last good)
the same way, under any arm name. `--run-id` is needed only for an arm
neither named nor stored. `compare --bound`, for qualification, refuses a
stored arm, requires the results digest of every arm and that both arms'
servers report one version and source commit (their packages may differ, as
two checkpoints' do), and records `bound: true`.

### GSM8K, needles and soak

```sh
export CHOPINFER_QUAL_CONFIG=models/mimo26/qual.json
python3 tools/qual/gsm8k.py --url http://127.0.0.1:8101 --concurrency 8 --response-dir RESPONSES
python3 tools/qual/needle.py --url http://127.0.0.1:8101 --timeout 1800
python3 tools/qual/soak.py --url http://127.0.0.1:8101 --duration 1800 --concurrency 32
python3 tools/qual/bench.py --url http://127.0.0.1:8101 --output-tokens 512 --prefill-tokens 32768
```

`gsm8k.py` scores the frozen 200-item screen. It reads the last
`#### NUMBER` line or, failing that, the last `\boxed{...}` holding only a
plain number; anything else (percent signs, units, fractions) is unparsed
and wrong, as are length-limited answers. It passes at the selected
declaration's `preview.gsm8k_floor` (a configuration's own); one that
declares none, such as a model screened at bring-up
before a floor is adopted, needs `--floor N`, which the receipt records with
`floor_source: explicit` (a declared floor cannot be replaced). `gsm8k.py rescore --output DIR
RECEIPT[=RESPONSES]...` rescores stored receipts from their retained
responses without modifying them. `needle.py` takes its prompts and match
rule from the declaration (MiMo's context files, Qwen's frozen prompt set);
`--lengths` runs a subset.

`soak.py` samples GPU memory of the serving process tree (found from the
port, or `--server-pid`) and fails on more than 512 MiB growth
(`--warmup-seconds` and `--cache-allowance-mib HOST,DEVICE` apply a declared
warm-up and cache allowance, as above).
`--prefix-mix` adds multi-turn and shared-prefix traffic with a greedy
canary that must restore its anchor's output (equal, or differing only at a
0.5-nat near tie) and zero used KV pages once idle. `--media-mix` and
`--structured-mix` each send an eighth of requests with media
(`--media-cases` selects them) or a strict JSON schema. The receipt keeps
the first 16 structured outputs that end with finish_reason `length` or fail
to parse or satisfy the schema (`structured_kept`: worker, turn, request
body, response text, finish reason, completion tokens and the server's
response id, whose suffix is the `req_` id that `CHOPINFER_REQUEST_TRACE`
records) and their total; truncation still counts in
`mixed_requests.structured_truncated` and a schema failure is still an error.
`--remote-only` drops memory sampling and cannot satisfy the stability
capability.

Every receipt `common.receipt` or a `chopinfer eval` capture writes records
`host_load`: the samples that bound the check's timed work (wall time,
uptime, load averages, cumulative CPU ticks and cumulative pressure stall
microseconds, the "some" totals of `/proc/pressure/cpu` and
`/proc/pressure/io`) and their verdict under `common.HOST_LOAD_RULE`. A
`chopinfer eval` capture samples at its start and when it writes the
receipt. A `common.receipt` caller passes `load_samples` taken around its
timed work, the first at its start and the last at its end: `soak.py` (its
start, each memory sample, and after the last request in flight),
`bench.py` (the decode passes and the timed prefill), `prefix_cache.py
record` and `spec_verify.py record` (their requests). A receipt without
them has `timing` `informational` ("no host load samples bound the timed
work"), since a window from the check's start would average warm-up into
the timed work. The host is idle when, over every interval between
samples, the CPU idle fraction (idle plus iowait; niced work is busy) is at
least 0.9 and the shares of elapsed time (uptime, not wall time) in which
some task stalled waiting for a CPU and for I/O are each at most 0.1. The
1-minute load average is recorded but not judged: it carries minutes of
work from before the window, such as a server's startup reads. Otherwise,
or with fewer than two samples, an incomplete sample, an interval whose
counters or uptime do not advance, or without uptime or pressure stall
information, `host_load.timing` is `informational`: the receipt's
throughput and latency figures are not performance evidence. The label
never changes whether a check passes. `common.py` gives the rule's
reasoning and the 2026-10-17 measurement it rests on.
`step_overlap_ab.py compare`, `agentic_compare.py`, `prefix_cache.py
compare` (its follow-up TTFT) and `spec_verify.py compare` (its
`ms_per_token`) label a comparison the same way, as evidence only when
every arm ran on an idle host; an arm record without `host_load` counts as
informational. The
scope is the host that runs the check (`check_host`): with `--remote-only`
the server runs elsewhere.

`kv_capacity.py` records the serving KV pool a running server measured:
`chop_kv_capacity_tokens` and `chop_kv_block_capacity` from `/metrics`
(tokens across a context-parallel group) and the serving profile and served
modalities from `/health`. Every supplied fact must agree: the tokens split into whole
blocks and each block into pages on the health's context-parallel size
(`--page-tokens S` pins the page size). `--server-log LOG` also checks the
startup line `serving KV capacity: P pages, T tokens` against the gauges and
the context-parallel line, present exactly above size 1, against the size,
pages and tokens; `--planned TOKENS` records the ratio to a planner's
estimate without gating on it.

```sh
python3 tools/qual/kv_capacity.py --url http://127.0.0.1:8101 --server-log SERVER.LOG --planned TOKENS
```

## Structured outputs

The structured check sends a fixed set of constrained requests and validates
each output locally: `response_format` `json_object` and `json_schema`
(strict and not), the `structured_outputs` fields `json`, `json_object`,
`regex` and `choice` on chat and completions, `tool_choice` `required` and
named, and a JSON schema with reasoning on. It reports the validity rate
(valid over valid plus invalid), latency and tokens per second. HTTP 400
counts as refused and `finish_reason: length` as truncated; neither is a
grammar violation. `chopinfer eval run --check structured` captures and
scores it (the `structured` kind); the declared arguments select the cases,
repeats, streaming, sampling controls and seed.

`structured.py` keeps the case set and its validator (soak.py's structured
mix sends them), a self-test of the validator, and the mechanics capture a
release-gates/v1 config runs:

```sh
python3 tools/qual/structured.py --url http://127.0.0.1:PORT --mechanics-only
python3 tools/qual/structured.py --self-test
```

## Prefix caching

`prefix_cache.py record` sends greedy streamed chats one at a time: a
follow-up turn after a user turn of each `--lengths` size (1K, 32K, 128K by
default), three questions under one long system prompt and three prompts
sharing about 100 tokens. Restart the server between records and use one
`--run-id`:

```sh
# --prefix-cache off, --prefill-chunk-tokens 2048
python3 tools/qual/prefix_cache.py record --arm baseline --url http://127.0.0.1:8311 --run-id prefix-001
# --prefix-cache off, --prefill-chunk-tokens 4096: the chunked-prefill noise floor
python3 tools/qual/prefix_cache.py record --arm floor --url http://127.0.0.1:8311 --run-id prefix-001
# --prefix-cache on, --prefill-chunk-tokens 2048
python3 tools/qual/prefix_cache.py record --arm candidate --url http://127.0.0.1:8311 --run-id prefix-001
python3 tools/qual/prefix_cache.py compare --baseline BASELINE.json --candidate CANDIDATE.json --floor FLOOR.json
```

`compare` passes when at least as many candidate requests emit the
baseline's tokens as floor requests do, the repeated prompts report cached
tokens, and the candidate's mean and maximum |dlogprob| stay within the
floor's plus 0.01 and 0.05.

### Hit accuracy against exact

`prefix_exact.py` checks the prefix-cache contract, a hit is bitwise equal
to a one-shot cold prefill ([numerics](../../docs/numerics.md#prefix-cache-accuracy)),
at every scored position of the gate's exact reference. Per position, under
its own `cache_salt`, it records a cold prefill, a cold prefill with a chunk
ending at the anchor (the last 64-token page boundary before the position)
and a real hit there. It requires the cached tokens the
[anchor rules](../../docs/serving-kv-pool.md#where-prefix-anchors-are-captured)
give, hit and chunked rows bitwise equal to cold rows, and hit rows no
farther from exact than cold rows. It also checks prompt-end anchors, that
a different image or geometry shares no anchor past the image while a
repeated image hits, and, with a drafter, that continuations and acceptance
after hits match cold ones.

```sh
python3 tools/qual/prefix_exact.py references --out "$CHOPINFER_DATA/prefix-references.json"
chopinfer serve --model "$CHECKPOINT" --debug-reference-tokens "$CHOPINFER_DATA/prefix-references.json" ...
python3 tools/qual/prefix_exact.py record --url http://127.0.0.1:$PORT --server-log SERVER.log --run-id prefix-exact-001 --concurrency 4
python3 tools/qual/prefix_exact.py report --record RECEIPT.json
```

Use a new `--run-id` for each record against a running server. `record
--texts` scores a subset and `report --record A --record B` combines
disjoint records from one build. `record --split C` adds a receipt for
prompts longer than the reference (hits at every multiple of C must continue
exactly as cold), passed to `report --split-record`. It first compares one
prefill-produced row at `--concurrency`, then compares full continuations
serially with separate cache salts. Run this check on an otherwise idle
server: decode reduction order depends on batch shape, so concurrent decode
rows cannot establish exactness ([batch shape](../../docs/numerics.md#batch-shape)).
Both passes require exact token IDs, generated logprobs and top-5 maps and
expected cache counts. `max_abs_dlogprob` covers all rows in both passes.

## Decode graph bucket coverage

Teacher prompt logprobs and paired teacher captures do not exercise concurrent
decode graph buckets. For a wider AR bucket, hold filler streams with
`fill_batch.py` through `concurrency-control.py measure`, then run
`verify_exact.py record --k 0 --phases 4 --decode-bucket 64 --server-log SERVER.log`
on a server with `CHOPINFER_STEP_TRACE=1` and the pinned reference tokens.
Keep the filler count above half the bucket and the filler plus probe count
at or below the bucket. Every pure decoding step in the recorded byte window must
report that bucket in `row_capacity`; row counts do not establish the executed
capacity. Prefill/mixed steps run multiple programs and have no single row
capacity; they are counted separately and cannot establish bucket coverage.
Empty pure-decode traces, missing capacities and mixed verify steps fail.

Compare the resulting capture with an AR C32 capture using
`verify_exact.py report --ar C32.json --verify B64.json --decode-bucket 64`.
The control must itself be recorded with `--decode-bucket 32`; the report
requires its C32 evidence (`--ar-decode-bucket` overrides the control bucket).
This requires both arms' bucket evidence as well as the existing position coverage and
0.005-nat relative gate against the pinned GPU reference. Each `--control`
reports the bucket its own evidence names, and with `--decode-bucket` that
evidence must pass too. Record 8- and 16-row AR draws for that use with
`--decode-bucket 8` or `16` beside 5 or 9 filler streams. Repeat for bucket
128. Keep the paired teacher, speculative copy-set, prefix and preemption
checks: this AR trace check does not qualify their separate paths.

## Generated-token logprobs

`logprobs_parity.py capture` sends fixed chat and completion requests
(non-streaming, streaming, and streaming with `return_token_ids`; thinking
on and off, hidden reasoning, `top_logprobs=0`, sampling, a stop string and a
tool call). `compare` lists where two receipts' logprob rows differ and
exits nonzero on any difference. The gate checks structural coverage
(every case, finite scores, complete arrays, matching top-k labels), not
numerical parity.

Qualified captures bind the live checkpoint to the selected gate pin, check
every response's checkpoint and startup nonce headers, and confirm the same
process through final health. Receipts record the observed responders. An
optional `--server-log` must identify that live process. External diagnostic
servers require `--mechanics-only`; those receipts cannot qualify a model.
Comparisons retain that classification if either input is mechanics-only,
so diagnostic comparisons remain readable and the quality gate rejects them.

```sh
python3 tools/qual/logprobs_parity.py capture --mechanics-only --arm external --url http://127.0.0.1:8102 --model MiMo-V2.6-Flash-MOPD
python3 tools/qual/logprobs_parity.py capture --arm chopinfer --url http://127.0.0.1:8311 --model MiMo-V2.6-Flash-MOPD
python3 tools/qual/logprobs_parity.py compare EXTERNAL.json CHOPINFER.json
```

## Preemption by recompute

Record the same prompts against a server started with `--admission
reserved` and one with `--admission optimistic --debug-preempt-every 16`:

```sh
python3 tools/qual/preempt_compare.py record --arm reserved --url http://127.0.0.1:8461 --concurrency 16
python3 tools/qual/preempt_compare.py record --arm optimistic --url http://127.0.0.1:8461 --concurrency 16
python3 tools/qual/preempt_compare.py compare --reserved RESERVED.json --preempted OPTIMISTIC.json --every 16
python3 tools/qual/preempt_compare.py burst --arm optimistic --url http://127.0.0.1:8461 --concurrency 32 --requests 64 --output-tokens 1024 --require-preemptions
```

`compare` requires identical tokens before position `--every`, near ties
(top-2 gap under `--tie-nats`, default 0.5) at every later first
divergence, and preemptions only on the optimistic server. `--exact`
requires bit-identical tokens, logprobs and alternatives, which holds when
every resume restores its own anchor
([numerics](../../docs/numerics.md#preemption-resume)).
`--ties-before-preemption` judges a first divergence before `--every` by
the same near-tie rule, for a batched pair: at C16 the two servers' batch
companions differ from the first step, so a near tie can flip before any
preemption. It refuses `--exact` and `--baseline`. A model that declares
both pairs gates C1 with `--exact` (its `preemption_exact_c1` check) and C16
with `--ties-before-preemption` (`preemption_exact`); a model that declares
only C16 keeps it `--exact`. `--baseline` compares two reserved records. In `burst`, `--min-tokens N` keeps requests
from stopping early and `--require-preemptions` is the gate.

## Agentic coding mixed prefill/decode sweep

`agentic_mixed.py` ports Runyard's pinned coding workload
([source and license](agentic_corpus/NOTICE.md)): multi-turn sessions that
list, search, read and edit private in-memory code copies (no generated code
runs). It compares prefill-budget modes under mixed prefill and decode load.

```sh
python3 -B tools/qual/agentic_mixed.py --url http://127.0.0.1:8101 \
  --model MiMo-V2.6-Flash-MOPD --levels 1,4,8,16 --seconds 60 \
  --arm tbt --label sched-1 --seed runyard-coding-v1 --temperature 0
# Run again against the fixed-budget server with --arm fixed and the same label.
python3 -B tools/qual/agentic_compare.py FIXED-SWEEP.json TBT-SWEEP.json
```

Pass the root URL (without `/v1`). Use a new shared label per pair of
sweeps and identical flags except URL and arm. A level
passes when its arm's prefill-budget mode is verified (from `/metrics` or the
local process's `--prefill-budget`), it has no request errors or timeouts,
and every invalid session is a proven model outcome (ending inside the
thinking block, or an undefined tool call, shown by re-parsing the turn's
raw tokens; at most 3 per 100 sessions). Server step histograms are the
primary evidence; client burst gaps and stalls are secondary. A pass does not
mean a performance target was met. `--dry-run` runs against an owned
loopback fake server on CPU.

The pinned corpus has three small files and nine tasks, so with shared
prefixes the prefix cache absorbs almost all prefill: earlier Qwen receipts
(2026-09-30 to 10-02, C4 and C8) show 95-99% reuse and under 200 prefill
tokens per prefill step, too few for the budget modes to differ. Two changes
add pressure:

- `--cache-salt session` (the default) sends each session its own
  `cache_salt`, derived from the run ID, seed, level and case. A session
  still reuses its own earlier turns. The request identity the arms compare
  leaves the salt out. `--cache-salt none` shares prefixes across sessions.
- `--corpus DIR` reads a larger corpus: Chopinfer's own Rust source at a
  pinned commit, staged outside the tree with
  `python3 -B tools/qual/agentic_workload.py stage --revision COMMIT`
  (default destination `$CHOPINFER_DATA/agentic/corpus-<commit>`). It picks
  64 files of 4 to 64 KiB under `crates/` by a hash of commit and path, with
  three generic tasks per file and Runyard's tools and system prompt. The
  receipt records the corpus digest and commit, not its path. The separate
  decode probe always reads the pinned Runyard source.

Each level's `prefill` record states how much prefill the window caused:
the server's prefill tokens per prefill-carrying step and prefix-cache reuse,
the client's prompt and uncached prompt tokens per turn, and the
`chop_mixed_step_actual_over_predicted` histogram (count, mean and bucket
bounds).

The tool-result limit belongs to the corpus: each task's limit is a quarter,
half or all of the manifest's `max_tool_result_bytes`. Runyard's 64 KiB
never binds on its small files. A staged corpus sets 16 KiB, so its 4 and
8 KiB tiers cut many 200-line reads. Each session records `tool_results`
and `truncated_tool_results`, and each level totals them per limit
(`tool_results`).

Every percentile comes with its sample count. The JSON distributions carry
`samples`; the summary prints n beside each figure and stars a percentile
with fewer samples than it needs (20 for p95, 100 for p99,
`agentic_metrics.TAIL_SAMPLES`); `agentic_compare.py` gives each figure's
samples in both arms, and `supported` for a percentile. `--min-samples`
(default 20) is the number of completed measured sessions each replica of a
level needs; with fewer it is insufficient, since a p95 over fewer than 20
sessions is the slowest session or next to it. Lengthen `--seconds` rather than lower
it.

Each level runs `--replicas` passes (default 3) back to back against the
same server. Replica R takes its tasks, request seeds and cache salts from
`SEED/replica-R` (replica 0 from the seed itself), so both arms run the same
replicas. Replicas are repeated measurements of one workload, not samples to
pool: each is its own level entry with its own figures, and the receipt's
`replica_spread` gives each figure's per-replica values and samples with
their min, median and max. `agentic_compare.py` pairs replica R of one arm
with replica R of the other and reports, per figure, both arms' spreads, the
per-replica differences and whether the arms' ranges overlap.

`agentic_compare.py BASELINE CANDIDATE` compares any two arms whose labels
name a prefill-budget mode, `fixed`, `tbt` or `decode-aware`, alone or with
a `-` suffix for the arm's other server settings: `--arm fixed-mbt4096` for a
fixed budget at `--max-batch-tokens 4096`. Each level's mode is verified
against the label from the listener's `--prefill-budget` or from series only
that policy renders (`chop_mixed_budget_*` and `chop_mixed_step_*` for
`tbt`, `chop_prefill_budget_decode_estimate_*` for `decode-aware`); the
selected-budget histogram comes from both and establishes neither. Pass
`--prefill-budget` to the server explicitly for a `fixed` arm, which renders
no policy series.
The output names the arms (`arms`) and reports `candidate_minus_baseline`.
Of two arms with different modes, the `fixed` one is the baseline in
either order (`fixed-mbt8192` before `tbt-150ms`).

The raw re-parse reads the served model's frontend declaration: `--frontend`,
else the `frontend` input of `CHOPINFER_QUAL_CONFIG`, else the inventoried
model whose frontend declares the served model ID. It is independent of the
served parser: it decodes token IDs through `/detokenize` and never reads
frontend-format or the server's parse. A format that declares a library tool
grammar in `format.parser.tool_grammar` (`xml-function-parameters-v1` or
`xml-key-value-arguments-v1`) uses the shared re-parser, `Markup`. Any
other format supplies its own as `agentic_markup.py` in its model root:
`--model-root`, else `CHOPINFER_QUAL_MODEL_ROOT` (set by `release_gate.py
--model-root`), else `models/<m>` of that inventoried model. The file must be
a regular file directly in the root; a symlink is refused
(`model_modules.py`). A format with neither is refused: the run prints the
refusal and its remedies on stderr, and its partial receipt records only
`runner error: ReParserUnavailable`. The module defines a
class `Markup` with this interface:

- `Markup(frontend)`: built from the parsed frontend declaration.
- `grammar`: a name for the tool grammar it reads, recorded in the receipt
  as `frontend.tool_grammar`.
- `turn(opening, text)`: the decoded prompt tail and generation of one turn,
  as `(reasoning, closed, after)`, or a class string when it is not one
  well-formed turn.
- `last_class(reasoning, closed, after)`: the class of a session's last turn.
  `RAW_REASONING_ONLY` and `RAW_MALFORMED_CALL` (`agentic_metrics.py`) are
  model outcomes; any other string is not.
- `calls(after, tools)`: the content and tool calls after the reasoning, as
  `(content, [(name, [(parameter, raw value)])])`, or None when they are not
  well-formed. `tools` is the request's tool list.
- `same_value(raw, value)`: whether a raw parameter value is the parsed
  argument value.

The module may import the modules of `tools/qual`, which are on the path.
The receipt's `frontend` record holds the declaration's SHA-256, the
re-parser's `grammar` and `reparser_sha256`: the SHA-256 of the
`agentic_markup.py` bytes that ran, or null for the shared re-parser.
`agentic_compare.py` refuses a pair whose `frontend` records differ.

## Correctness runner

`tools/qual/release_preview.sh` runs one model's correctness suite against the
HF F32 references and the stored baselines. A run serves each configuration
from one release install the way a user does (`chopinfer serve --model DIR`),
one server per arm, for as long as the suite needs.

```sh
bash tools/qual/release_preview.sh prepare --model M          # CPU: install, direct package, checkpoint verify
bash tools/qual/release_preview.sh run --model M --suite quick  # GPUs 0-3
bash tools/qual/release_preview.sh run --model M --suite full
bash tools/qual/release_preview.sh plan --model M --suite full  # the arms and steps, no GPU
```

- `prepare` builds `$CHOPINFER_DATA/validation/installs/<commit>/` if missing,
  prepares each checkpoint's direct package and runs `chopinfer verify` once
  per checkpoint. Its marker under `validation/verified/` holds the files'
  device, inode, size, mtime and ctime; `run` refuses a changed checkpoint
  instead of hashing inside a GPU run.
- `prepare` builds in a cargo target owned by this checkout: `--target-dir`,
  else the caller's `CARGO_TARGET_DIR`, else `wt-<name>-<hash of the
  checkout path>` under `$CHOPINFER_PREVIEW_TARGETS` (default
  `$CHOPINFER_DATA/validation/targets`). A `chopinfer-checkout` file in the target names its owner; a target
  another checkout owns is refused, and an unowned one is claimed only when
  empty or a `<target>.worktree` file beside it names this checkout. The
  claim links a complete owner file into place, so when two checkouts claim
  one target at once, one wins and the other is refused. Cargo judges
  freshness by the sources of whichever checkout built last, so a shared
  target can install another checkout's binaries.
- An install records its build in install.json's `build`: the commit, the
  dirty-tree digest, the checkout, the target and every file's sha256.
  `prepare` reuses an install, or adds another model's helper to it, only
  when that record names this checkout and target and every file still
  matches; otherwise it refuses, and an
  install built elsewhere (or before the record existed) must be moved
  aside, with its `<commit>-build` logs, to be rebuilt here.
- Arms: `spec` (the declared profile), `ar` (`--speculative-tokens 0`),
  `spec-repeat` (`--prefill-chunk-tokens 1024`), forced-token servers for
  verify_exact and prefix_exact records (`--debug-reference-tokens`), and
  `reserved`/`optimistic` admission servers for preemption records. Each fresh
  server records determinism first (`chopinfer eval capture --kind
  determinism`, with the install's binary). The speculative server starts
  last and answers the declared checks, each run as `checks.py check` with its
  records, its server log and earlier passing receipts as inputs (the
  teacher, api and structured checks through `chopinfer eval run`).
- Screens: GSM8K 200 on both arms (floor `preview.gsm8k_floor`) with the paired
  rule and branch probe, C1 greedy AR against speculative under
  `preview.spec_greedy` (records by `chopinfer eval capture --kind
  greedy_pin`), determinism spec against spec-repeat (and against AR
  when the declared determinism check does not run), spec_verify copy bounds,
  the queue check (2 x cap + 1 clients), quick's one needle at the largest
  declared length and one prefix_exact slice (`report --slice`), full's AR soak
  (1,200 s, judged by the soak kind with the stage's own arguments).
- A declared check counts only when `checks.py check` exits 0 with a passing
  check.json that holds the check's row and a readable receipt (exit 1 with a
  failing one); a timeout, any other exit or missing evidence fails it.
- Checkpoint verification uses every declared manifest, plain `FILE` or keyed
  `FILE#ENTRY`. Cached proof binds the checkpoint file state and the manifest's
  path, selected entry, file identity and SHA-256. A missing, linked or changed
  manifest invalidates the proof. Prepare checks it across byte verification;
  run checks it before every server arm. Old markers require fresh verification.
  Native verification still refuses unrecorded weights; only ancillary files
  may use its existing `--allow-unrecorded` behavior.
- Configurations: the screens run on `default`; another configuration runs
  its declared cohort and records on the speculative server. Configurations
  at four GPUs run in turn; those at two run two at a time, one on each GPU
  pair of the job's four.
- quick runs the declared `quick` cohort on the `preview.quick_arms` servers;
  full records every declared input and runs the `full` cohort. An input no
  record produces comes from `--input KEY=PATH`; without it the check does not
  run, which fails the run. An input that is a value rather than a file, such
  as the run ID of a check another run recorded, comes from `--input-value
  KEY=VALUE`, is given to the checks as written and is recorded in run.json's
  `input_values`; the check's own input rules still judge it.
- A check module may bring its own arms and records as module-level
  `RUNNER_ARMS = {arm: [serve words]}` and
  `RUNNER_RECORDS = {input: (arm, script, arguments)}`; arguments may use
  `{url}`, `{name}`, `{run_id}`, `{server_log}`, `{pid}`, `{depth}` and
  `{reference_tokens}`. A callable option takes the model name. A callable
  `RUNNER_ARMS` may return a function of the model facts and configuration,
  which the runner calls when the words depend on them (`kv_offload.py`).
- A module's `RUNNER_OFFLINE_RECORDS = {input: (script, arguments)}` are
  records that need no server; with `RUNNER_CHECK_ARM = "offline"` its check
  needs none either. They run on the configuration's `offline` arm, after
  every server of the configuration has stopped, with its GPUs and every
  record of the run (`{<input>}`, `{checkpoint}`, `{run_dir}`); each offline
  record's output directory is its input. The check runs as `checks.py check
  --offline`, which only such a module may use, and its run names no live
  server. The arm has no server, health or shutdown evidence
  (`spec_continuation.py`).
- `kv_offload.py` records plain and pressure scenarios, each on offload-off
  and forced-swap arms. Pressure pages come from `pressure_tokens` divided
  by the package's `kv_layout.page_tokens` and context-parallel size.
  `runtime-eval`'s `kv_offload` limit rule owns the gates; rescore recomputes
  them from the saved paired records. The pressure record's `fill` phase,
  after the warm cohort, guarantees the concurrent cohort's prefix eviction:
  its plan comes from the pool gauges and C1's longest prompt, and
  `fill.precondition` says whether it was reached. The facts are `budget_gib`,
  `swap_every`, `max_preemptions`, `min_gated` and `pressure_tokens`.
  The check has no host-prefix arm: the CLI refuses it. See the
  [scope and gates](../../docs/serving-kv-pool.md#formats-caching-and-offload).
- For development, `run` and `plan` accept repeated `--check ID` selectors;
  `prepare` does not. Each selected configuration must declare every selected
  ID in the suite's cohort, and the selection replaces that cohort. These runs
  omit the model screens, plan and check server shutdown evidence for the
  selected checks only, record the selected IDs, and can neither update last
  good nor become the seed. `rescore` replays the recorded selection. For
  example, use `--suite full --config default --check kv_offload_plain
  --check kv_offload_pressure` for the offload smoke.
- Server stop: each server runs under GNU timeout with `--foreground` and
  duration 0, so the wrapper has no expiry (no TERM source) of its own, and the
  runner stops it with one TERM to that wrapper, which forwards it to serve
  once. The runner has the backstop (below) send that TERM and waits for its
  confirmation. Without `--foreground`, timeout forwards it again to its process
  group, and native serve exits 0 at once on a second TERM, skipping its
  drain. The stop waits the drain the serve words select
  (`--drain-timeout-ms`, 30 s when absent) plus 10 s of cleanup, 40 s by
  default, then KILLs the server's process group and waits 10 s more. The
  wrapper's own KILL of serve comes 5 s after the stop's group KILL. A stop is
  clean only when serve exits 0 within the TERM wait, no KILL was sent and no
  process of its group remains; anything else fails the arm's server row. The
  runner reads the drain at serve's option positions with its unsigned 64-bit
  syntax and refuses a repeated flag, a missing or malformed value and the
  `--drain-timeout-ms=N` form before it starts a server.
- Cleanup backstop: each server starts with a watchdog in its session but in
  its own process group. The watchdog sends the server's one TERM when the
  runner's stop asks for it, or on its own when the runner dies before asking.
  It KILLs the server's process group 15 s after the stop's TERM wait would
  end, counted from that TERM (55 s by default), or the same delay after the
  wrapper exits (for example when serve crashes). A runner paused or hung before its
  stop therefore cannot shorten the declared drain, and descendants such as
  the preparation helper cannot outlive a runner that died or hung after its
  stop began, or a wrapper that exited. A runner that hangs before its stop
  keeps its server until the job's limit kills the runner. The watchdog
  signals only the wrapper and that group, KILLs before it logs (logging is
  best effort, so a full log cannot stop it), and exits once the group is
  empty or 10 s after its KILL.
- Stop records: each arm's `launch.json` holds the serve command, the wrapper
  argv, the drain (`source`, `milliseconds`, `term_wait_seconds`,
  `wrapper_kill_after_seconds`) and the backstop's `kill_after_seconds`.
  `shutdown.json` holds the exit status, the stop's stages, the drain and the
  armed backstop with its watchdog pid. A run, `rescore`, a last-good update
  and a reseed admit every planned arm's records. The drain must equal the one
  the recorded serve words select, the stop must have sent one TERM and waited
  that drain through the backstop, the wrapper must be the foreground,
  duration 0 wrapper of exactly that command, and the backstop must have started (its pid is not
  the wrapper's) and been armed (`armed` is JSON true). A record with
  none of the drain, wrapper or backstop fields predates the drain-aware stop
  and keeps its earlier verdict.
- Output under `$CHOPINFER_DATA/validation/runs/<model>/<UTC>-<commit>-<suite>/`:
  `run.json` (what was tested: commit, uncommitted changes, install digests,
  packages, checkpoints, inputs, and every used feature install's source and
  file digests), per arm its launch, health, server log and
  receipts, then `results.json` and `report.md`. Each printed row gives its
  values and limits and the stored last good and seed values. Exit 0 when every
  row passes, 1 otherwise, 2 for a refusal before any server starts.
- Baselines: `validation/last-good/<model>/<config>/` moves only when a full
  run selects and passes every declared configuration, with the validation
  checkout and every used install (plain and featured, each source recorded)
  at clean commits that main contains
  (unless `--no-baseline-update`); the result stands either way; `validation/seed/` moves only through
  `baseline reseed --model M --from RUN --reason TEXT`. `history.jsonl`
  records each update. Focused `--suite full --config C` runs remain usable and
  exit 0 when their checks pass, but cannot update last-good. Store admission
  checks the selected configurations and the result rows independently.
  `baseline show --model M` prints both.
- `rescore RUN` replays a runner run with the current code: the same plan,
  each child's kept exit status and receipt judged again, declared checks
  through `checks.py rescore`, comparisons recomputed from the arms. A row the
  plan expects without its evidence fails, and a recorded server,
  configuration or record failure is kept. Output goes to `RUN/rescore/<UTC>/`.
  `rescore RUN --legacy-preview MODEL` re-judges a previous preview run's.

## Per-layer taps against the reference (`tapdiff.py`)

`tapdiff.py` compares engine tap captures with GPU reference runner taps.
The [runner guide](../../docs/reference-runner.md#taps-and-tapdiff) is the
canonical workflow and verdict description. Numerical comparison needs numpy
(`CHOPINFER_REFRUN_PYTHON`); `--help` does not.
