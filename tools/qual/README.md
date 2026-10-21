# Qualification tools

Use `python3 -m tools.qual --help` for the command registry and each command's
help for options. Numerical policy lives in [numerics](../../docs/numerics.md)
and the versioned [check kinds](../../docs/check-kinds.md). Model declarations
own actual checks, thresholds, inputs, configurations and timeouts.

## Release gates

The gate consumes an already-running server and prepared inputs. It does
not start servers or download datasets. A quick cohort is a screen; full
qualification requires every declared check and configuration. Missing
metrics, empty selections, nonfinite values, timeouts and child failures
fail closed. Structured output must be valid without refusal or truncation.

```sh
python3 tools/qual/release_gate.py --model mimo26 --profile quick --url "$URL"
python3 tools/qual/release_gate.py --model mimo26 --profile full --url "$URL" --inputs release-inputs.json
```

`--plan` shows checks; `--list-inputs` inspects authenticated prerequisites
without serving. `CHOPINFER_QUAL_CONFIG` selects a declaration for standalone
checks, `CHOPINFER_QUAL_MODEL_ROOT` its root, `CHOPINFER_EVAL` the built host,
and `CHOPINFER_DATA`/`CHOPINFER_QUAL_RECEIPTS` external output locations.
Inputs must belong to the candidate; an input path or successful rescore
does not prove which binary originally produced it.

`chopinfer eval` captures and scores portable kinds. Its portable subset
does not replace the full runner. `python3 -m tools.qual check ID` runs one
declared check; `rescore` uses current or recorded policy without mutating
the original run. A missing input is insufficient and a failed worker stays
failed even if its partial receipt can be scored. Exit meanings are in
[the qualification contract](../../docs/check-kinds.md#exit-statuses).

### Model declarations

`qual.json` declares cohorts, named checks, required inputs, reference pins,
capabilities and optional configurations. `chopinfer eval declaration` owns
validation; the Python loader calls it. Unknown fields and mistyped constants
are refused. `modules.json` records allowed modules and constant types.
`preview` binds the runner to the serving profile and its speculation
settings. No default model is inferred by model-neutral tools.

### Configurations

A different topology, KV format, draft depth, checkpoint or build needs its
own configuration and server. The configuration selects checks and overrides
declared serving facts, pins and rules. Its waivers are its own, not inherited
permission to waive the default's failures. Locks preserve member scopes;
an unbound declaration cannot vouch for another declaration's inputs.

### External variants

An external checkpoint uses `DIR/MODEL/VARIANT.json` and its own checkpoint,
reference and overlay locks outside the tree. Select it with `--variant-dir`
or `CHOPINFER_QUAL_VARIANTS`. `python3 -m tools.qual variant` supplies lock,
pin, skeleton and adopt commands. Its overlay may narrow media/capabilities
and select base checks, but cannot change limits, add waivers, change how
the model serves or reuse an in-tree configuration name. Inherited references
must still bind its own checkpoint. Nothing it owns is read unauthenticated.
A full pass qualifies that checkpoint only and never advances the runner's
in-tree last-good baseline. Preserve the overlay/lock identities in receipts.

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
[GPU reference runner](../refrun/README.md#prepare-and-run):
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
`/health` identity ([what `/health` reports](../../docs/developer-preview.md)):
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
[runner guide](../refrun/README.md#cache-and-fixture-bindings)
describes fixture assembly, the binding fields and the standalone verifier.

### Adopted qualification inputs

`models/<model>/qual-references.lock.json` (with each configuration's own
entries under `configurations`; a variant declaration file's:
`qual-references.<variant>.lock.json`; an external variant's
`<variant>.lock.json` beside its overlay) pins the bytes gates read:

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

A fresh proof of bytes the lock already holds updates their object's
record only when entries of the lock pin them and every one of those is
replaced in the same adoption. Then the proof's production fields replace
the old ones, while the schema, the producer and the record's other fields
stay: after a pin bump changes the requirements digest a preprocessing
reference binds, its unchanged tensors name the new record. An object a
configuration or a kept entry also pins, or that no entry pins any more,
keeps its record, and a proof that contradicts the record is refused. A
tensor whose bytes differ from its record is refused before anything is
pinned.

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

Compare against the pinned model reference, not another serving engine.
Both absolute gates and declared last-good/seed paired comparisons apply;
see [paired accuracy](../../docs/numerics.md#paired-release-accuracy).

### Verify rows against autoregressive rows

Prepared candidate rows use the same scoreable positions as their AR control.
The [numerical contract](../../docs/numerics.md#serving-qualification-gates)
defines the mean margin, top-one bounds, coverage and single/pooled-draw
distinction. Exact C1 equality is separate from approximate bucket accuracy.
Declared continuation checks score after texts diverge; informational
statistics do not waive a failed gate.

## Multimodal serving

Match checkpoint binding, prompt token IDs and usage to the reference.
Declared media formats determine coverage and scoring; see
[check kinds](../../docs/check-kinds.md#declared-media-capture-formats).
Case answers are teacher-forced under per-case spread limits. A greedy walk
through its first departure must remain within 0.3 nats of the reference
top-one (2.3 for audio). Image-probe descriptions use per-probe spread bounds
with mean error at least 0.15 and at most 0.30; greedy agreement is reported.
Preprocessing and encoder checks are separate obligations. Served but
unqualified modalities must remain explicitly declared as such.

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
([runner guide](../refrun/README.md#encoder-reference)) over the
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

## Prefix caching

Hit and chunked rows must equal cold rows bitwise under the model's granule
and the context-parallel stripe rules. [Prefix accuracy](../../docs/numerics.md#prefix-cache-accuracy)
defines logprob, top-one and acceptance gates; [KV state](../../docs/numerics.md#kv-offload-and-prefix-state)
defines anchors, offload comparisons and lifetime requirements.

## KV layout equality

A comparison layout storing exactly the values the native KV formats decode
to must serve the same bits. Compare determinism records from the native and
comparison layouts in one full run and build, with matching checkpoint,
metadata, bundle, version, model and effective profile except KV dtype.
Every request's text, finish reason, token IDs, tokens and selected/top
logprobs must be bit-identical. A package identity may vary with layout and
must be reported. This full-suite comparison is not a quick-cohort check.

## Correctness runner

The runner builds a release install, verifies checkpoints and starts each
configuration's arms for the needed captures and checks:

```sh
bash tools/qual/release_preview.sh prepare --model MODEL
bash tools/qual/release_preview.sh plan --model MODEL --suite full
bash tools/qual/release_preview.sh run --model MODEL --suite full
```

Use a target owned by this checkout. Preparation records file hashes and
build identity; reuse requires matching ownership and unchanged artifacts.
Checkpoint verification markers bind both files and the selected manifest;
changed files, symlinks or manifests require fresh verification. Runs retain
actual startup identities for each arm. Supply missing files with `--input`
and literal input values with `--input-value`; every declared input must exist.

The full suite records the declared AR/speculative, determinism, prefix,
preemption, offload and media evidence. A check counts only with a passing
row and readable receipt. Baselines advance only after a full passing run of
all required configurations; reseeding requires a recorded reason. External
variants never advance in-tree baselines. Missing captures are not passes.
No rule or baseline may be changed to hide a mismatch.
