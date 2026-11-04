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
an unbound declaration cannot vouch for another declaration's inputs. A
configuration declared `"opt_in": true` runs only when selected with
`--config NAME`: a full run without `--config` leaves it out, and last good
does not require it.

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
files; when present every field below except `not_served` is required.
Both loaders refuse missing coverage, including checks omitted from
`cohorts.full.checks` and checks marked informational. Module names
identify the required checks, so each model chooses its own IDs.

| Capability field | Value and required hard check modules |
|---|---|
| `text` | Boolean; `teacher`. |
| `frontend` | Boolean; `frontend_goldens`. |
| `api` | Boolean; `api`. |
| `structured` | Boolean; `structured`. |
| `media` | Unique list of `image`, `audio`, `video`; each modality needs its own nonempty reference inventory. Case-format media declares `qualification.media.coverage`, mapping each modality to reference item IDs; image probes cover images only. |
| `speculation` | `none`, `lookup`, `mtp`, `dflash` or `dspark`; enabled speculation requires `spec_verify` and `verify_exact` with distinct control and candidate inputs, mandatory predicates, positive coverage and observed drafting. |
| `prefix_cache` | Boolean; `prefix_exact` scores bitwise equality. |
| `preemption` | Boolean; `preempt_compare compare --exact`, without `--baseline`. |
| `long_context` | Boolean; `needle` with the declared context lengths. |
| `stability` | Boolean; a local `soak` with timed host-RSS and device-memory samples, bounded growth and traffic in every enabled mix. A `--remote-only` soak cannot satisfy it. The media mix reads the `soak_media_fixtures` input when the declaration states one, else `media_fixtures`. |
| `kv_offload` | Boolean. True requires model-level `kv_offload` facts and hard plain/pressure checks in the full suite. Serving offload remains off by default. |
| `not_served` | Optional. Maps a boolean capability declared false, one the model does not serve, to its reason. The correctness runner reports each false boolean capability as a not-applicable row with its reason, never a pass, and fails one without a reason. |

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
comparison refuses it. The mode may be any the host serves (`off`, `fixed`,
`balanced`, `adaptive` or `table`), and it is `off` exactly when its tokens
are 0. The
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
python3 -B -m tools.qual refs adopt --model MODEL --entry KEY [--entry KEY ...] [--replace-sha256 OLD_LOCK_DIGEST] [--dry-run]
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

`adopt --entry KEY` refreshes only the named entries and keeps every other
entry, with the objects it pins, as the lock holds it, so it reads no other
input; when only the named inputs changed it writes what a full `adopt`
writes. It needs the current lock and refuses a name the declaration does
not declare. It also refuses a selection that leaves out an entry pinning a file a
named entry pins (the same path, or a file inside a named directory, or a
directory holding a named file), and an input a runner proof binds (a
reference, a workload, a media reference's captures, and the
preprocessing and mixed-generation proof inputs), named or overlapping;
only a full `adopt` repins those. The lock
digest and `--dry-run` rules are the same. An external variant's lock is
adopted whole.

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
request (`OUT/engine`, `OUT/tap-capture.json`). The trace, the steps'
`items.json` and the manifest must name it by its integer engine request ID;
`true` is not request 1. `--prefix-hit
NAME=PRIMER=PATH=BODY.json` sends the continuation of an earlier
`--request PRIMER` under the primer's salt (serve with prefix caching on
for it): the primer must report 0 cached tokens and the continuation
`0 < cached_tokens <=` the primer's prompt tokens, as determinism's
prefix-hit rule requires, and both bind their steps to their own requests.
The manifest also keeps the server's complete `model_options` and
`model_options_qualified`, which must be the qualified defaults. A
diagnostic option override is captured only with `--experimental` and a
sealed pin, as `determinism.py` takes one; such a capture records the pin
and `qualified: false`, and no gate admits it. The gate then admits the
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

`prefix_exact.py record --ram-hit` is the RAM-hit arm. Its server runs
`--kv-offload-gib N --kv-offload-prefix on --debug-prefix-ram`, so every hit
restores an anchor copied back from the host tier. Its record meets the
GPU-hit arm's gate, drafter acceptance included, and each request that
reports cached tokens must raise `chop_prefix_cache_ram_hits_total` once,
as it raises the all-hit counter. Report the two arms' records separately.
The runner records the RAM-hit arm for a configuration whose serve words
force the tier. A model whose bundle runs the tier may declare the opt-in
`ram-forced` configuration, which serves every arm that way
(`release_preview.sh run --model MODEL --config ram-forced --suite full`).
It runs prefix_exact, determinism, soak, spec_verify and verify_exact.
Where the default runs determinism only as a runner screen (the screens run
on `default` alone), the configuration runs the declared determinism check,
which judges the same AR and speculative captures. A configuration's checks
keep their declared input rules, so where a check's rules pin another
configuration's serving profile, the configuration states the default's
facts in `serving` (such as `context_parallel_size` 1). Its quick cohort
holds a check that reads only what a quick run records. Its prefix_exact input
predicates require RAM-hit records (`hit_tier` "ram", `checks.ram_hits`
true), so a GPU-hit record given with `--input` is refused.

`kv_offload.py`'s `kv-offload-prefix-pressure` arm runs the pressure
workload with the tier on (`--kv-offload-prefix on`) and no forced swaps,
so C16 finds the anchors the fill evicted only in RAM. `report --off
PRESSURE_OFF --tier TIER` judges it against the pressure off arm: both
records complete captures without a failed request, the off arm passing its
own end and control gates, every C1 request bitwise equal, the fill's
precondition met, promotions and RAM hits above zero, missed demotions
recorded, and no pending transfer (counted by
`chop_kv_offload_pending_transfers` or aged), swapped request or stall at
the end.
Models whose bundles declare `chop.kv-offload.v1` run it in their full
cohorts as `kv_offload_prefix_pressure`.

## KV layout equality

A comparison KV layout that stores exactly the values the native formats
decode to must serve the same bits. Bitwise equality of native against BF16
storage is required in a model's BF16 comparison configuration (`kv-bf16`)
on the determinism records, on forced records and on the decoded KV-state
taps of a `diagnostic-taps` build. The BF16 layout matches the native
layout's stored values (each decoded value, held exactly in BF16) and its
production prefill partials, so the comparison isolates KV storage. Every
route only the native layout takes is compared as served: the kernels that
read its packed codes and block scales and each fast route a package admits
at its native layout alone run on the native server, the BF16 server runs
its admitted paths there, and the bits must still be equal. A BF16 configuration that serves autoregressive
decoding only runs both layouts without speculation. The check does not
qualify production accuracy: the default configuration's teacher, chat,
long_context, multimodal, `verify_exact` pairs, greedy AR versus DSpark and
determinism gates do.

`kv_layout_equality.py compare NATIVE COMPARISON --native-forced NATIVE_FORCED
--comparison-forced COMPARISON_FORCED` reads, from one full run and build,
both layouts' determinism records and a forced record of each
(`verify_exact.py record --k 0 --require-row-shapes --reference
long_context`, on the runner's `kv-native-forced` arm and the
configuration's forced `prefix` arm). Every request's text, finish reason,
token IDs, tokens and selected and top logprobs must be bit-identical. The
forced records must hold the same positions, those the gate's pinned exact
and long-context references define, with bit-identical logprobs and top-5
at every forced position and bit-identical prefill rows; the first
differing text and position is named. The long-context text (for a
`long_128k` reference, 131,072 tokens) is forced through its prefill
(130,048 tokens) and then decoded for its 1,024 scored positions, which
reaches multi-chunk prefill and decode over a long history; the runner
adds it to the configuration's forcing file (`RUNNER_FORCED_REFERENCES`). A teacher capture cannot stand
in for it, since the teacher scores prefill rows only. A server that cannot
hold the long history refuses the record, naming the KV capacity it needs,
and the check fails.

Each pair must name one checkpoint, metadata, source commit, bundle,
version, served model, effective profile and complete `model_options` map,
compared with their JSON types, and each forced record must be live-bound
to the gate's checkpoint pin and match its layout's determinism record in
those fields and its package. The package identity may vary with the
layout and is reported. The layout fields are validated, not compared:
each `kv_cache_dtype` is `bf16`, `fp8` or `mixed`, `kv_cache_planes` is
present exactly when it is `mixed`
([serving identity](../../docs/operations-observability.md#serving-identity)),
the native planes equal the declared `NATIVE_PLANES` and the comparison
arm reports none. This full-suite comparison is not a quick-cohort check.

`kv_layout_equality.py taps NATIVE COMPARISON --native-planes PLANES.json`
compares two [tap captures](#tap-captures) of the same requests, one from a
diagnostic-taps server of each layout, bit for bit in logical
representation: KV-state taps by readback (request, phase, state end),
layer, plane, position and rank, and text taps by name, position and rank.
A row one capture lacks is a difference. The native capture's raw stored
codes and scales are not compared across layouts, but must be present for
every plane, rank and readback and decode exactly to its logical values
(window E4M3 with a UE8M0 scale per 32, compressed E2M1 with a UE4M3 scale
per 16, index E2M1 with a UE8M0 scale per 32, as `kernel_core` decodes
them); a product that is not exactly a finite BF16 is refused, never
rounded, and the comparison capture holds no raw taps. It refuses captures
that differ in prompts, steps, serving identity, profile or `model_options`
(compared with their JSON types), apart from the startup nonce, the package
identity and the KV layout fields: each `kv_cache_dtype` must be `bf16`,
`fp8` or `mixed`, `kv_cache_planes` must be present exactly when it is
`mixed`, the native planes must equal PLANES (as `NATIVE_PLANES`
declares them), and the comparison arm must report none.
It also refuses a row no position or item names, an index entry whose
dtype, columns or rows differ from the step's `capture.json` declaration,
and a comparison that compared no text rows or no KV-state rows of a plane or phase (a restore
when the captures hold a prefix-hit pair). Experimental captures compare
only with `--experimental` and never qualify.

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
preemption, offload and media evidence. Forced arms serve the
configuration's `--debug-reference-tokens` file: the exact reference's
sequences (`verify_exact.py references`) and those of each pinned reference
a selected check's module names for its forced records
(`RUNNER_FORCED_REFERENCES`, `verify_exact.py --reference`). A check counts only with a passing
row and readable receipt. Baselines advance only after a full passing run of
all required configurations; reseeding requires a recorded reason. External
variants never advance in-tree baselines. Missing captures are not passes.
No rule or baseline may be changed to hide a mismatch.
## Declarative agentic traces

`agentic_mixed.py --workload trace --trace FILE` reuses the streaming client,
metrics, qualification receipts and `agentic_compare.py` for finite synthetic
session traces. Generate a content-free declaration outside the repository:

```sh
python3 -B tools/qual/agentic_trace.py --out "$CHOPINFER_DATA/trace.json" \
  --sessions 8 --turns 4 --contexts 100000,250000 --idle-pages 0
python3 -B tools/qual/agentic_mixed.py --workload trace \
  --trace "$CHOPINFER_DATA/trace.json" --levels 8 --replicas 1 \
  --temperature 1 --seconds 600 --session-timeout-seconds 600 \
  --min-samples 8 --cache-salt group --arm fixed --label baseline
```

Both workloads write each receipt as compact, sorted JSON in one gzip
member, `<stamp>-agentic-mixed.json.gz`, beside its `<stamp>-agentic-mixed.md`
summary. The JSON and its schemas are unchanged. `agentic_compare.py` and
`agentic_trace_analysis.py` read this form and the plain `.json` receipts
written before it, as `common.read_receipt` does for other readers;
`gzip -dc RECEIPT` prints the JSON.

This four-turn example declares 72 seconds of think time per session,
leaving the rest of the 600-second measured window for requests. It disables
page-pressure parking; longer traces can enable it when sufficient other
traffic is planned. Think times and output budgets rotate within each
session, including when the session count is divisible by four.

The generator uses append shares 31/40/12/7/10 percent for 1-63,
64-511, 512-2047, 2048-8191 and 8192+ tokens over each 100 requests.
Balanced output budgets 128/256/550/1266 average 550 tokens. Think times
0/2/10/60 seconds are declared assumptions. Source files, patches and test
results are synthesized from seeds, then fitted with the served `/tokenize`.
Newest-anchor survival is unavailable without served capture frontier and
granularity evidence. Each turn instead records `warm`: whether its cached
tokens reach `warm_threshold_tokens`, its session's previous prompt length
(the primer's for the first turn) less one prefill chunk and at least one
token. The chunk is the trace's `prefill_chunk_tokens`, 2,048 when not
declared. With canonical history the threshold depends on the workload
alone, so both arms judge a return alike, and a cold return, an evicted
primer's included, is false; generated history adds each arm's own answers
to the previous prompt. Rendering loss before the previous answer can also
make a return cold.

`--returns cyclic` declares cyclic returns: a free slot goes to the least
advanced ready session, so every session returns only after each other
ready session had its turn, and no session parks on idle pages. The
sessions' contexts and histories, with per-session salts, then set the
footprint the server must hold for returns to stay warm; size it beyond a
GPU pool to measure eviction and a host tier.

No trace field accepts prompt text. The page size is explicit in the trace
and must match the served model's logical KV page size.

Each turn depends on the prior turn's completion. Thinking and parked
sessions release their HTTP slots. A finite trace must finish within
`--seconds`; incomplete, errored or timed-out traces fail. This is a maximum
elapsed window, so throughput uses actual elapsed time. The sample floor
still labels unsupported tails; eight sessions do not establish session p95.
Long contexts are primed before timing. `group` salts share each group's
prime; `session` salts isolate each session and require separate primes.
Use per-session salts for acceptance and report shared groups separately.

Every SSE event has a monotonic timestamp and token multiplicity, including
usage and completion events. Every served prompt must equal `/tokenize`
token for token, and usage/output-ID counts must agree. `canonical` history
keeps later requests matched when outputs diverge. `--trace-history generated`
instead appends the actual answer and measures LCP against the previous
prompt plus emitted IDs. Priming includes an answerable user turn; later
turns reuse its shared system prefix. A render with
`chat_template_kwargs.preserve_thinking=true` provides the diagnostic
counterfactual, including loss before the previous answer. Canonical history
replacement can also lose LCP, so do not attribute its whole loss to thinking.

Idle page demand is `ceil((prompt_tokens - cached_tokens + output_tokens) /
page_tokens)` summed over other completed requests. It excludes snapshot,
copy-on-write and speculative allocations. Cache behavior can change this
clock across arms; retain actual return schedules and the existing eviction
and cached-token metrics. It is not a measurement of physical allocation.

`agentic_trace_analysis.py RECEIPT --steps LOG --power SAMPLES` reports
50/100/250/1000 ms event-gap counts, total and excess durations, and unioned
prefill overlap, plus TTFT by append class, throughput, draft acceptance,
computed/reused prefill and anchor survival. Per cell it also reports
sessions kept warm (the share of returns whose `warm` is true), return TTFT
p50 and p95, output tokens per second and joules per output token; each
level of the replay receipt already carries the first two as
`sessions_kept_warm` and `return_ttft_ms`. Power samples must bracket the
window and retain each rank's limits and SM clocks; missing samples yield
unavailable energy. SSE gaps include transport and parser effects. Profiled
or busy-host results remain diagnostic. Run fresh-process warmup and ABBA
visits as described in [stateful replay acceptance](../../docs/contributing.md#stateful-replay-acceptance).

`agentic_verdict.py A1 B1 B2 A2 --gain METRIC --cell C --power VISIT=FILE
--major-faults VISIT=N` judges one ABBA's four replay receipts by that rule.
It refuses receipts that are not one ABBA: failed or partial replays,
different workloads or settings other than the seed, unmatched pair seeds
or arms, other cells or replicas, two visits in one server process, or
visits that did not run one after the other in A1, B1, B2, A2 order. Per
cell and metric it reports each visit's value, d1, d2, mean, spread and a
verdict; the cache-hit rate and sessions kept warm (percentage points) and
both stall rates use differences, and a metric without supported samples
in a visit is unavailable. Gains resolve only in the declared metrics and
cells, regressions in any. Joules per output token compare under identical
power limits; an energy veto needs energy in every cell. Preemptions,
prefix evictions and the RAM tier's counters are reported per cell as
evidence. `accepted` needs a resolved gain, no resolved regression or energy
veto, the protocol (temperature 1.0, per-session salts; group salts stay a
diagnostic) and complete evidence: idle-host, non-mechanics visits, power
samples with each GPU's power, limit and SM clock bracketing every measured
window, each GPU's limit constant across all four visits, and at most 1,000
major faults per visit.

Trace analysis calculates decode stalls from generated output events using
the recorded burst window. Prompt IDs, usage, finish and DONE events remain
available as separately labelled raw SSE gaps. Older receipts reconstruct
output timing from TTFT, output inter-delta times and channel kinds; missing
evidence stays unavailable. Stall count and total stall time are normalized
per 1,000 eligible burst gaps, with raw totals retained. Append-class
p50/p95/p99 values require 2/20/100 observations; unsupported tails are
unavailable, while sample counts and observed maxima remain visible.

Current agentic qualification records explicit `fixed`, `tbt` or `adaptive`
prefill policy identity from verified listener arguments or
`chop_prefill_policy_info`. Shared TBT metrics cannot establish that identity.
Use `--fake-prefill-policy fixed|tbt|adaptive` for CPU mechanics runs.
Decode-aware arm labels, including suffixed labels, fail current verification.
Historical receipt comparisons retain their original budget evidence; an
old TBT label does not distinguish today's TBT and adaptive policies and
cannot be compared as canonical policy evidence.
