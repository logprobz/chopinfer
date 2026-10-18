# Operator commands

`chopinfer --help` prints the synopsis of every command and `chopinfer eval
--help` the eval commands. Every command except `verify` also accepts
`--architecture ID` and repeatable `--capability ID`.

## Local checkpoint serving

```text
chopinfer serve --model DIR [--tensor-parallel-size N] [--model-cache-dir DIR]
  [--activation-policy auto|checkpoint|bf16]
  [--modalities auto|all|text[,image][,video][,audio]] [--load-threads N]
  [SERVING OPTIONS]
```

`serve --model DIR` serves a local Qwen3.8 Flash, MiMo V2.6 Flash or GLM 5.3
Flash checkpoint in a layout its family serves
([compatible checkpoints](compatible-checkpoints.md#serve-your-own-checkpoint))
by loading its weights directly. The family
helper installed beside the host emits a small package without weights into the
model cache, and names the serving profile it uses (Qwen: MTP where the rank
count and precision support it; MiMo: DFlash when the checkpoint's DFlash
drafter checks, else MTP, else none; GLM: the MTP profile for every layout;
the encoders the checkpoint has). The
profile's settings are defaults under any explicit serving option.
`--model` is exclusive with `--bundle`, `--topology`, `--profile` and
`--checkpoint`. It does not run qualification. See
[compatible checkpoints](compatible-checkpoints.md) for discovery, the model
cache and what serving checks.

| Option | Default | Meaning |
|---|---|---|
| `--tensor-parallel-size N` | 4 | Rank count; the helper must support it (TP2, TP4; GLM TP4 only) |
| `--model-cache-dir DIR` | `$CHOPINFER_DATA/model-cache`, then `$XDG_CACHE_HOME/chopinfer/models`, then `~/.cache/chopinfer/models` | Where packages are emitted and reused |
| `--activation-policy` | `auto` | Qwen per-operation activation recipes; MiMo refuses it. See [quantized checkpoints](quantized-checkpoints.md#activation-policy) |
| `--modalities` | `auto` | `auto` serves text and image, those of them the checkpoint provides; `all` serves every modality it provides (MiMo adds video and audio). A list must include `text`, without duplicates, and only modalities the checkpoint has. Left-out encoders free their memory for KV ([capacity](serving-profiles.md#served-modalities-and-memory)) |
| `--load-threads N` | `CHOPINFER_DIRECT_LOAD_THREADS`, else twice the CPUs divided between ranks, at most 16 | Direct-load workers per rank |

A request with a modality the package leaves out returns HTTP 400 naming
`--modalities all` and a list naming the modality. `--video-input` is gone:
`--modalities` alone decides whether video is served.

`serve --model DIR --list-model-options` prepares and verifies the small
package, prints its option declaration as JSON and exits without loading
weights or opening a GPU. `serve --bundle DIR --list-model-options` lists an
existing package. A package without declarations prints `null`.

Options apply to the whole process. Request `model_options` fields are
refused. Boolean and numeric flag values use JSON syntax; string values are
unquoted text. Repeat the flag for different names, with each name once.
Explicit flags override `CHOPINFER_MODEL_OPTIONS=name=value,...`, then the
profile's `model_options` map, then package defaults. The environment form
cannot represent a comma inside a value; use an explicit flag for that case.
Docker and systemd accept the same environment variable.

Startup logs, `/health`, `/v1/models` and `chop_model_option_info` metrics
report resolved options. Any non-default value is unqualified, even if it
falls within the allowed range. Eval and qualification receipt admission
refuse unqualified options. A default also needs the package's
`default_is_qualified` declaration before it can qualify.

The mechanism is currently exercised by the synthetic fixture. Qwen3.8's
`indexer_budget` is not declared overridable: its shipped QSA kernels compile
the keep count as 512 blocks of four tokens (2048 tokens). See the
[architecture contract](../BUNDLE_ARCHITECTURE.md#model-options).

## Checkpoint verification

```text
chopinfer verify --model DIR [--manifest FILE[#ENTRY]] [--allow-unrecorded] [--jobs N]
```

Checks every file of a checkpoint, offline, against the digest Hugging Face
recorded for it: from hub snapshot blob names, `hf download --local-dir`
metadata, or a manifest. It exits nonzero on any mismatch, missing shard or
unrecorded file. `--allow-unrecorded` accepts unrecorded files other than
safetensors files and their indexes. `--jobs N` (1 to 256) hashes N files at
once; the default is the process's CPUs, at most 8. Serving checks only
headers and lengths, so this is the content check. Details:
[what verify adds](compatible-checkpoints.md#what-serving-checks-and-what-verify-adds).

## Serving options

These apply to `serve --model` and `serve --bundle`. Defaults below are the
host's; a serving profile (applied by `serve --model` or `--profile`)
replaces them, for example with 32 sequences.

| Option | Default | Meaning |
|---|---|---|
| `--listen ADDR` | `127.0.0.1:8310` | HTTP listen address. Loopback by default; put an authenticated proxy in front before exposing it |
| `--max-sequences N` | 8 (profiles: 32) | Live sequences; bounded by the package maximum |
| `--max-context-tokens N` | package limit | Per-request context cap |
| `--max-batch-tokens N` | 8192 | Token budget of one scheduled batch |
| `--prefill-chunk-tokens N` | 2048 (at most the batch budget) | Prefill chunk size |
| `--kv-memory-fraction F` | 0.9 | Fraction in (0, 1] of the tightest rank's free memory, after other allocations and a 1 GiB reserve, given to the KV pool |
| `--kv-pages N` | none | Caps the KV pool in pages |
| `--kv-cache-dtype auto\|bf16\|fp8` | `auto` | `auto` uses the package's preferred KV layout |
| `--prefix-cache on\|off` | `on` | Prefix reuse; `off` is a diagnostic escape hatch |
| `--speculative-tokens N` | 0 (profiles set it) | Draft depth; 0 serves autoregressively and drops the profile's speculation |
| `--speculation fixed\|adaptive` | `adaptive` | Adaptive drafts only when it pays; fixed always drafts N |
| `--draft-cost per-token\|per-block` | `per-token` | Drafter cost model; DFlash profiles use `per-block` |
| `--speculative-rows N` | package limit | Caps verify rows per step |
| `--queue-depth N` | unbounded | Requests waiting beyond the live sequences; a full bounded queue returns HTTP 429 |
| `--queue-wait-ms N` | 0 (no limit) | A request not admitted within N ms returns HTTP 429 |
| `--drain-timeout-ms N` | 30000 | Shutdown drain time for admitted responses |
| `--devices 0,1,...` | profile, else the first N | Distinct ordinals within `CUDA_VISIBLE_DEVICES`, one per rank; duplicates are refused |
| `--model-option NAME=VALUE` | package default | Repeatable server-wide override of a declared checkpoint option; unknown names, wrong types and out-of-range values fail before weight loading |
| `--served-model-name NAME ...` | package or checkpoint name | Display aliases; they do not change identity or qualification |
| `--api-key KEY` | `CHOPINFER_API_KEY`, else `VLLM_API_KEY`, else none | Bearer token required on requests |

Requests beyond `--max-sequences` wait in FIFO order; a disconnected client
cancels its queued request. Other options (prefill budgets, KV offload,
admission, request size and image count limits, CORS, media fetching) are
listed by `chopinfer --help` and described in the
[request reference](serving-request-surface.md),
[mixed scheduling](mixed-scheduling.md) and [deployment](deploy-common.md).

## Packages and profiles

```text
chopinfer bundle verify --bundle DIR
chopinfer bundle check-frontend --bundle DIR [--topology ID] [--render FILE]...
  [--tokenizer FILE]... [--streaming FILE]...
chopinfer serve --bundle DIR [--checkpoint DIR [--load-threads N]]
  [--profile NAME|FILE#NAME] [--topology ID] [SERVING OPTIONS] [--fake-host-services]
chopinfer conformance-run --bundle DIR [--checkpoint DIR] --fixture-request LABEL
  [--topology ID] [--fixture-tokens COUNT] [--scenario commit|cancel|drop]
  [--kv-memory-fraction F] [--fake-host-services]
chopinfer profiles list FILE
chopinfer profiles show FILE NAME
chopinfer kernel build|inspect --manifest FILE [--cache-dir DIR]
```

`--bundle` names a package directory, such as one emitted by `qwen38-package`,
`mimo26-package` or a helper's `prepare`. `--checkpoint` names the checkpoint
whose weights the package plans; a package with planned components is refused
without one. `--profile` applies a [serving profile](serving-profiles.md) as
`serve --model` applies the helper's. Verification reads every component in
full and does not load native code; serving verifies the package before
loading it. There is no discovery, hot reload or network fetch.

`bundle check-frontend` checks a package's chat frontend against the model's
[frontend vectors](external-models.md#frontend-vectors) through the code
`serve` runs, without a GPU or checkpoint; `bundle check-frontend --help`
describes the vector files.

`conformance-run` drives one bounded batch and exits: `commit` (default),
`cancel` (cancel and replay) or `drop` (cleanup with a pending batch). A native
bundle runs as trusted code in the host process; serving one is an
authorization decision ([security boundary](security-boundary.md)).

## Qualification: `chopinfer eval`

`eval` checks a running server against a model's qualification declaration
(`models/<m>/qual.json`) with the shared [check kinds](check-kinds.md). It is
the one implementation of those kinds: the Python tooling in `tools/qual`
scores through `eval rescore`, runs a model declaration's teacher, api and
structured checks through `eval run --check ID` and takes the correctness
runner's determinism and greedy records with `eval capture`.

```text
chopinfer eval list (--model DIR | --package DIR)
  [--variant NAME [--variant-dir DIR]] [--json]
chopinfer eval run (--model DIR | --package DIR)
  [--variant NAME [--variant-dir DIR]] [--served-model NAME]
  [--suite quick|accuracy | --check ID... [--cohort quick|full]]
  [--references DIR] [--report DIR] [--json] [--timeout SECONDS]
  [--server-log LOG] (--url URL | --launch [SERVE FLAGS...])
chopinfer eval capture --kind determinism|greedy_pin
  (--model DIR | --package DIR) [--variant NAME [--variant-dir DIR]]
  [--served-model NAME] [--references DIR] [--timeout SECONDS]
  [--server-log LOG]
  --url URL --report DIR RECORD ARGUMENTS...
chopinfer eval verify-references (--model DIR | --package DIR)
  [--variant NAME [--variant-dir DIR]] [--references DIR] [--json]
chopinfer eval files (--model DIR | --package DIR) [--variant NAME]
chopinfer eval rescore < REQUEST.json
chopinfer eval declaration --model DIR [--variant NAME [--variant-dir DIR]]
chopinfer eval declaration --root ROOT < DECLARATION.json
```

- `--model DIR` is `ROOT/models/<model>` (holding `qual.json`; declared paths
  resolve under `ROOT`) or a checkpoint directory, resolved as `serve --model`
  resolves it. `--package DIR` is a package directory, whose carried serving
  profiles name its model, or an install root or checkout with one model.
  `--variant NAME` reads the configuration `NAME` that `qual.json` declares,
  expanded onto the base declaration, with the model's lock; without such a
  configuration it reads a variant declaration file `qual.NAME.json` beside
  `qual.json`, with its own pin, references and lock. `--variant-dir DIR`
  (default `$CHOPINFER_QUAL_VARIANTS`) reads a variant the tree does not
  declare from `DIR/<model>/NAME.json`, an
  [external variant](../tools/qual/README.md#external-variants) beside its
  lock.
- `list` prints the declared checks, cohorts and pinned references;
  `verify-references` verifies the checkpoint lock and every pinned reference
  offline; `files` prints every file under `ROOT` that `eval` reads (the
  release archive ships them at the same paths).
- `run` runs the `quick` cohort (default), the `full` cohort (`--suite
  accuracy`) or the checks named with `--check`. Each check takes its
  declared arguments, then those its cohort adds (`cohort_args`), as
  `tools/qual/checks.py check` runs it: the suite's cohort, or with `--check`
  the `--cohort` named (default `quick`). It binds the server: `/health`
  must report the pinned checkpoint identity and a startup nonce, and every
  response must carry the `x-chopinfer-checkpoint-identity` and
  `x-chopinfer-startup-nonce` headers of that process. The bearer token is
  `OPENAI_API_KEY` when set; only `http://` URLs are supported.
- `run --server-log LOG` also binds the server to its startup log, as
  `tools/qual` checks do with `--server-log`: the log's last `listening`
  record must name the URL's address, its startup records one checkpoint
  identity and one startup nonce, both those `/health` reports. Every record
  then names the log and its SHA-256 (`server.checkpoint.server_log`,
  `server_log_sha256`). A log of another process fails the run (exit 1); an
  unreadable log or one without a single identity leaves it unidentified
  (exit 3).
- `run --launch FLAGS...` (last on the line) starts `chopinfer serve FLAGS...`
  for the run, logs to `REPORT/server.log`, waits up to `--timeout` for
  readiness on the flags' `--listen` address, and stops the server's process
  group afterwards.

`capture` writes one record a runner compares, not a declared check: an
arm's determinism record (`--arm ARM --run-id RUN [--image] [--image-reason
TEXT]`, as `tools/qual/determinism.py record` writes it) or greedy record
(`--arm ARM [--k N] [--cold-cache] [--concurrency 1] [--prompt-tokens N]
[--max-tokens N]`, over the texts of the reference pinned where the
declaration's `exact` input points). It binds the server as `run` does,
`--server-log` included, confirms after the last request that the bound
process answered every one, and writes the record to `--report DIR` (created
when missing) as `STAMP-CHECK.json`, the name and bytes `tools/qual`'s
`common.receipt` gives a receipt; it
prints the path and SHA-256. It exits 0 when the record is written, 1 when
the server serves another checkpoint or answers invalidly, 2, 3 and 4 as
`run` does; a refused record writes nothing.

`run` writes `--report DIR`, or a new directory under the declared receipts
root, holding `report.json` (schema `chopinfer.qual-report.v1`), `summary.md`
and one raw record per check under `records/`. A teacher check whose
arguments hold `--capture-out` also writes its compact capture, the file
paired comparisons read (`tools/qual/teacher_arms.py`), to `captures/ID.json`
(`tools/qual/release_gate.py` copies it to the path the declaration names),
and its record names the capture's `capture_sha256`.
The capture is written only after the run confirms that the bound process
answered every request; a refused run writes no capture. A check's status is `pass`,
`fail`, `informational`, `insufficient` or `not_run`; `media_forced` checks
are captured by `tools/qual` and scored by `rescore`.

| Exit | Meaning |
|---|---|
| 0 | every selected check passes or is informational |
| 1 | a gate failed, or the server serves another checkpoint or model, or changed during the run |
| 2 | usage or declaration error |
| 3 | insufficient evidence: a missing, unverified or stale reference, a server without an identity, an empty selection, or a check not run |
| 4 | environment error: the server is unreachable, timed out, or a launched server failed to become ready |

When several apply, the first of 1, 4 and 3 in that order is the exit.

`rescore` reads one JSON request on stdin, `{"schema":
"chopinfer-eval-rescore/v1", "calls": [...]}`, and writes one response on
stdout with a result per call. Each call names its `op`, `kind` (a shared kind
or the offline `frontend_goldens`), `receipt`, `reference` (or null), declared
`check` and `mechanics`:

| `op` | Result |
|---|---|
| `rescore` | `status`, `metrics`, `predicates`, `recorded_fields` and `error` |
| `verify` | `recorded_fields`, or an `insufficient` or `invalid` error |
| `score` | `metrics`, or an `insufficient` or `invalid` error |
| `score_capture` | teacher or media forced-answer columns, or greedy answers: `metrics`, or an error |
| `judge` | from the call's `metrics`: `status` and `predicates`, or a `missing_metric` error |
| `limit` | no `kind`: the limit `rule` (`rounded_up`, `spread_limits`, `within`, `near_tie`, `flip`, `flips`, `top1_summary`, `paired`, `bit_equal`, `first_difference` or `constants`) over its `args`: `value`, or an `invalid` error |

An error result is `{"error": {"type", "message"}}`. A malformed request
exits 2; otherwise `rescore` exits 0. The limit rules are implemented here
only; `tools/qual/limits.py` sends `limit` calls.

`declaration --model DIR [--variant NAME]` prints a model's validated
declaration; `declaration --root ROOT` validates the declaration on stdin
with ROOT's `tools/qual/modules.json` (how `tools/qual/declaration.py`
validates). A refusal names the field and exits 2.

## Live view: chopinfer-top

```text
chopinfer-top [--target HOST:PORT]... [--targets-file FILE]
              [--no-discover] [--no-docker] [--interval SECONDS]
              [--rescan SECONDS] [--api-key KEY] [--list [--json]]
```

`chopinfer-top`, shipped beside `chopinfer`, is a live terminal view of
running `chopinfer serve` instances. It reads only `/health` and `/metrics`.
Without flags it finds this user's serve processes and Docker containers
running serve. `--list` prints the instances once and exits; `--list --json`
prints a `chopinfer.top.list.v1` document.

| Option | Default | Meaning |
|---|---|---|
| `--target HOST:PORT` | none | Watch this server; repeatable. Also `http://HOST:PORT`, or `:PORT` for 127.0.0.1. `CHOPINFER_TOP_TARGETS` adds a comma-separated list |
| `--targets-file FILE` | none | One target per line; `#` starts a comment |
| `--no-discover` | off | Do not look for serve processes or containers |
| `--no-docker` | off | Do not ask Docker for containers running serve |
| `--interval SECONDS` | 1 | Seconds between polls of each instance, 0.5 to 60 |
| `--rescan SECONDS` | 15 | Seconds between discovery scans, 2 to 300 |
| `--api-key KEY` | `CHOPINFER_API_KEY` | Bearer token for explicit targets behind an authenticating proxy |
| `--list`, `--json` | off | Print the instances once and exit; `--json` needs `--list` |

A usage error, or the interactive view without a terminal, exits 2. Sources,
states, the series behind each number and the keys are in
[watch running instances with chopinfer-top](operations-observability.md#watch-running-instances-with-chopinfer-top).
