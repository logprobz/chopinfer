# Serving profiles

Each model owns `profiles.json`; use `chopinfer profiles list FILE` or
`chopinfer profiles show FILE NAME` for settings and recorded status. Both
validate profiles without a GPU. The [README](../README.md) summarizes
supported models and hardware. Package manifests remain authoritative for
topology, limits and capabilities.

## Selecting a profile

Production serves the speculative profiles: DeepSeek `tp4-dspark`, GLM
`tp4-mtp` and its DCP variants, and Qwen `tp4-mtp`. Each model's AR profile
is the baseline and correctness control that speculative output is checked
against, and a debugging aid. `tp4-dspark` drafts nothing from 32 decode
items, so falling back to AR needs no profile switch. MiMo is outside the
release scope; [its evidence](#mimo-evidence) is history.

`serve --model DIR` selects a profile through the installed model helper and
refuses `--profile`. Explicit packages accept a packaged name or `FILE#NAME`:

```sh
chopinfer serve --bundle "$PACKAGE" --checkpoint "$CHECKPOINT" --profile tp4-mtp
chopinfer profiles list models/qwen38/profiles.json
```

Four commands describe these choices as JSON, for a launcher such as
`choptop` to build `serve --model` commands from:

```sh
chopinfer models --json
chopinfer models pick --model "$CHOPINFER_QWEN38_CHECKPOINT" --json
chopinfer models command --model "$CHOPINFER_QWEN38_CHECKPOINT" \
  --tensor-parallel-size 4 --profile tp4-mtp -- --queue-depth 64
chopinfer serve --model "$CHOPINFER_QWEN38_CHECKPOINT" --tensor-parallel-size 4 --check
```

- `models --json` lists each installed model's profiles with their status,
  rank count, drafter, sequence limits, qualification text and the
  effective settings `serve --model` resolves from each, keyed by flag name
  without `--`.
- `models pick` names, for each rank count the model's profiles serve, the
  profile `serve --model` picks for that checkpoint or the helper's refusal.
  It finds the model's helper as `serve --model` does at the default four
  ranks, or else at another rank count an installed model's profiles serve;
  a count at which every helper refused the checkpoint keeps that refusal.
  Each other profile at that count is either explicit options over the pick,
  which resolve to exactly that profile's settings, or unavailable with the
  reason: it drafts with another drafter than the package carries, or it
  leaves a setting the pick sets at a default no flag value names. Model
  options only the pick sets are listed as inherited; a launcher resets them
  to the package defaults.
- `models command` prints the `serve --model` arguments that serve a profile
  with the given options. An option replaces the profile's flag of the same
  name, `--model-option` the same option name, and `--speculative-tokens 0`
  or `--prefill-policy fixed` clears the profile's speculation or TBT
  target as it does for `serve --model`. A duplicate or invalid option is
  refused here. `--listen` and `--devices` stay with each instance.
- `serve ... --check` runs every startup check that needs neither a device
  nor the package's native module, in serving's order: package preparation
  and verification, the drafter, model options, speculation bounds, KV
  offload host memory, the declared operations against this host, the
  frontend declaration and tokenizer, the topology with its context-parallel
  groups (selected as serving selects it, except that the device driver's
  devices are only planned), the media options, the engine's batch, chunk
  and sequence limits, the declared media recipes, what prefix caching
  needs of the package (copy-on-write batch records, and request state for a
  state-slot restore), the instance configuration (KV and state contracts,
  served items and context, and whether the package needs `--checkpoint`),
  what the backend and build fix
  (host-memory device capacity, the device sampler
  `--debug-reference-tokens` needs), the `--debug-reference-tokens` file,
  and the `CHOPINFER_NVTX_RANGES` and `CHOPINFER_DEV_SKIP_WARMUP`
  variables. It prints the profile, the same effective settings and every
  model option with its value and default. A failure exits nonzero with the
  message serving would give. A start can still fail on what only it
  decides: the native module's frontend export, weight plan and callbacks,
  the driver's devices, their memory and peer access, the KV capacity they
  hold, and the listen address.
  `--check` refuses a `--max-context-tokens` past what one sequence of the
  package holds, which no KV capacity can serve.

`serve --describe-options` prints every serve option with its value shape,
group and help, and the environment variables serve reads. A value shape is
`N` (a whole number), `F` (a number), `TEXT`, `TEXT...` (one or more words
before the next flag), `DIR`, `FILE`, `JSON`, `JSON|off` (JSON or the word
`off`) or `none` (a flag without a value); a choice is `TEXT` with its
`choices`.

An explicit option overrides a profile default. `--speculative-tokens 0`
disables speculation but retains that profile's sequence cap. A profile
cannot grant qualification to an override. `max_sequences` is the serving
cap; `qualified_max_sequences` is the largest concurrency its evidence
covers, or null for experimental profiles. Startup warns above that count.
The shared default is 32; Qwen `tp4-ar` has evidence through 64. Package
capacity alone is not serving qualification.

Production deployments should set `--queue-depth N` or `--queue-wait-ms N`
([queueing](serving-request-surface.md#queueing-and-admission)). Profiles
set neither, so by default waiting requests queue without bound and
overload never returns 429.

DeepSeek ships `tp4-ar` and `tp4-dspark`. Both use exact upper-skip prefill
with 8192-token grid ends and 64-token aligned shared-prefix anchors. An
unaligned prompt's last 1-63 tokens run in the same step as its aligned
anchor, which the bundle captures inside the step, when the chunk through
the prompt end fits the step's budget; otherwise they take a separate step.
Since 2026-11-02 this fused schedule, with exact-row dense tail folds, is the
default, and with upper skip no option restores the separate tail step. On
`tp4-ar` C8 it trades lower TTFT for longer decode stalls
([prefill dependencies](../models/deepseek41/prefill-dependencies.md)).
The full-prefill and 2048-token profiles were retired after shared-prefix
hit-rate qualification. Their model options remain available for explicit
diagnostic configurations.
`tp4-dspark` uses the stored floor-one confidence calibration, with 64
verify rows and widths 6, 8, 16, 32 and 64. See the
[model documentation](../models/deepseek41/README.md#served-profiles).

## Reply history measurement

`--reply-history-observer on|off` controls the optional bounded digest
observer. The optional profile field `reply_history_observer` is a boolean;
an explicit flag wins, and omission resolves to off. Existing profiles
retain their defaults. Enabling it collects repeated generated-page and
exact reply-end eligibility counters without enabling KV reuse. Collection
has measured host overhead and keeps at most 64 MiB of observer capacity.
See [metric semantics and passive sampling](operations-observability.md#generated-reply-observations).

## Speculative token capacity and graph budget

Two serve options size a speculative verify step. Each is also a profile
field under `speculation`, so a profile sets the default and a flag
overrides it:

- `--speculative-rows N` (`speculation.rows`), the token capacity: the most
  verify rows one step may carry, one row per decoding item plus every
  draft. A package that declares `chop.speculative-requested-rows.v1` serves
  its default rows unless asked for more, up to its manifest maximum, and
  prepares its buffers for the rows it is asked for. DeepSeek's package
  fallback is 32, while `tp4-dspark` requests 64; the maximum is 192
  (32 items at its five-draft block). Startup refuses
  rows at or below `--speculative-tokens` or above the maximum.
- `--speculation-graph-rows N` (`speculation.graph_rows`), the graph budget:
  the widest verify step captured in a CUDA graph. It reaches a package that
  declares the `verify_graph_rows` model option (DeepSeek, default 64);
  wider steps run without a graph and pay their launches. Startup warms
  every width up to the package's default rows and, above them, only the
  balanced plan's listed widths (`--speculation-widths`) up to the budget.
  Startup refuses a budget above the package's verify maximum and a package
  that does not declare the option.

`--list-model-options` checks both against the package without opening a
device. A wider step pays only where it costs less than the tokens its
drafts add, so measure the step cost of each width before raising either.
Each graph captured above the default rows also takes memory from the
graph memory reserve below. On 2026-10-24 DeepSeek TP4 DSpark at 128 rows
and a 64-row graph budget, with widths of 40, 48 and 64 rows above the
default, held 302 to 304 MiB of graphs per GPU after warmup against 280 MiB
at the defaults, and its buffers for 128 rows held 32 to 64 MiB more before
the KV pool.

For persistent weight images and cross-build content proofs, see
[the weight keeper](weight-keeper.md).

## Drafter confidence

`--speculation-confidence JSON|off` (`speculation.confidence`) lets a
balanced plan leave the choice of which drafts to verify to the package. The
plan still picks the step's verify width from its depth schedule and widths,
but reserves every item's drafts up to its limits; a package that declares
`chop.verify-allocation.v1` then verifies the drafts its drafter is most
confident in within that width. Every emitted token is still verified:
allocation changes which drafts reach the target, never what a verified
draft emits. The calibration is an object:

```json
{"contract": "example.drafter-confidence.v1", "floor": 1,
 "greedy": [[1.2, -0.4], [1.1, -0.6]], "sampled": [[1.0, -0.5], [0.9, -0.8]]}
```

`contract` names the package's `confidence_contract`, the meaning of its
drafter's raw confidence; `floor` is the drafts every item verifies before
any is ranked; `greedy` and `sampled` hold one `[scale, offset]` pair per
draft position, from the first, and a raw confidence `s` at a position means
`sigmoid(scale * s + offset)`, the chance its draft is accepted when every
earlier one was. Both kinds cover the same positions, at least
`--speculative-tokens` of them and at most 16. Fit calibrations on sampled
serving data, which is the serving workload. The option needs
`--speculation balanced` and speculative tokens; an explicit other mode or
zero tokens drop a profile's calibration, and `off` replaces it. Startup logs
`startup speculation_confidence:` with the result, and serves without
confidence when the package does not declare the capability or names
another contract. A run with `--debug-reference-tokens` also serves without
it, because forced rows are scored on the host and allocated verify steps
need device results. For the same reason a step with any request whose
sampling depends on its history (a presence, frequency or repetition
penalty, or a grammar constraint such as a structured output or a required
or named tool choice) is verified on the host without allocation, with the
drafts the plan gives it without confidence.

## GPU memory

`--gpu-memory-utilization F` (profile field `gpu_memory_utilization`, image
variable `CHOPINFER_GPU_MEMORY_UTILIZATION`) is the share of each GPU's total
memory this server's own allocations may take: weights, workspaces, encoders,
drafter and request state, the sampler, the KV pool and its graph memory
reserve. Every profile uses 1.0, which gives the pool all free memory but the
reserve. Lower it to leave memory for other programs on the same GPUs. The
server counts its own allocations, so memory another program allocates or
releases does not move its share. The CUDA context and loaded kernel code are the driver's and
come on top. When F leaves no room for one KV page, startup refuses and logs
the smallest utilization that fits. `--kv-memory-fraction` is gone: it took a
share of the memory left after the model, so its 0.9 default left a tenth of
that memory unused. Evidence recorded before 2026-10-22 ran at that 0.9; the
larger pool changes KV capacity, not arithmetic.

Bundles allocate everything they use before the host sizes the pool, and the
conformance suite refuses device memory a bundle acquires after it. Only CUDA
graphs come later: they embed the pool's addresses, so they are captured once
it exists, at startup warmup and for batch shapes warmup did not reach. The
host keeps a reserve per rank out of the pool for them:
`--graph-memory-reserve-mib N` (profile field `graph_memory_reserve_mib`),
1 GiB when neither names one. Graph memory is model specific, so the reserve
is per profile. On 2026-10-23 DeepSeek TP4 DSpark at 32 sequences captured
384 graphs per GPU holding 902 MiB, which left 137 MiB of a 1 GiB reserve,
so the DeepSeek profiles reserved 1280 MiB. The 2026-10-24 build holds far less after warmup at 32
sequences: 30 to 32 MiB per GPU for TP4 AR (18 graphs) and 280 MiB for TP4
DSpark with verify graphs up to 32 rows (152 graphs), and serving captured
no further memory. A reserve is the graphs, the 256 MiB warning headroom
and about 10 % for growth: `tp4-ar` reserves 320 MiB and `tp4-dspark`
640 MiB, and the rest goes to the KV pool. These are the two shipped
DeepSeek profiles.

The GLM, MiMo and Qwen profiles were measured the same way on 2026-10-30,
on that day's build at 32 sequences and utilization 1.0. Each server warmed
up and then ran a mixed load:

- a 7,000-token prompt;
- waves of 8 and 32 requests with prompts up to 4,000 tokens and staggered
  output lengths, three of them with an image on packages that serve
  images;
- single requests;
- a soak of cold prompts at every last-chunk length, each length twice,
  then each prompt again for a prefix hit.

The soak reaches all 64 of GLM's pinned last-chunk prefill programs and the
repeated shapes it keeps, which hold most of GLM's graph memory. Under
context parallelism its lengths follow the stripe; a soak that did not
pinned only 34 of the 64 programs at DCP4 and understated that profile's
peak by 172 MiB. The peak is the most memory any rank had taken since
sizing, by the `chop_device_memory_free_low_bytes` gauge or by nvidia-smi.
Each reserve is the peak, the 256 MiB warning headroom and about 10 %,
rounded up to 64 MiB. A second server at that reserve, under the same load,
logged no post-warmup headroom warning and kept every rank at the lowest free memory below or
more (MiB per GPU):

| Profile | After warmup | Peak | Reserve | Lowest free at the reserve |
|---|---|---|---|---|
| GLM `tp4-mtp` | 116 | 454 | 832 | 387 |
| GLM `tp4-mtp-k5` (text package) | 116 | 462 | 832 | 391 |
| GLM `tp4-ar` | 24 | 376 | 704 | 305 |
| GLM `tp4-mtp-dcp2` | 124 | 460 | 832 | 349 |
| GLM `tp4-mtp-dcp4` | 134 | 480 | 832 | 331 |
| MiMo `tp4-dflash` | 54 | 54 | 384 | 321 |
| MiMo `tp4-mtp` | 38 | 463 | 832 | 368 |
| MiMo `tp4-ar` | 6 | 6 | 320 | 337 |
| MiMo `tp2-ar` | 6 | 6 | 320 | 333 |
| Qwen `tp4-mtp` | 52 | 58 | 384 | 327 |
| Qwen `tp4-mtp-dcp2` | 56 | 62 | 384 | 321 |
| Qwen `tp4-ar` | 14 | 14 | 320 | 307 |
| Qwen `tp4-ar-dcp2` | 16 | 16 | 320 | 303 |
| Qwen `tp2-mtp` | 52 | 58 | 384 | 323 |
| Qwen `tp2-ar` | 14 | 14 | 320 | 303 |

MiMo and Qwen graphs grew by at most 6 MiB after warmup. MiMo `tp4-mtp`,
served from a `mimo26-package --drafter mtp` package, reserves 832 MiB for a
transient outside any graph whose cause is not yet known: during the first
warmup wave rank 0 holds 463 MiB that no graph keeps and then releases it,
and without it the profile would need 384 MiB. The reserves move 192 to
704 MiB per GPU into the KV pool. A configuration that raises its graph
budget, or a build that changes GLM's prefill programs, must measure its
reserve again. A capture that finds no memory fails the generation worker,
so startup logs each rank's free memory after warmup and warns below
256 MiB ([observability](operations-observability.md#chopinfer-metrics));
raise the reserve or lower `--max-sequences` when a rank warns.

### GLM encoder memory

A GLM package that serves images holds its vision encoder on every rank
before the pool. On 2026-10-30 a `tp4-mtp` server held 3,387 MiB per rank
more on the image package than on the text package: the encoder's working
memory (1,800 MiB), its media arena (512 MiB) and its weights (about
1,076 MiB). The working memory is idle while the server serves only text.
Lending it to the KV pool and taking it back when an image arrives was
considered and rejected on 2026-10-30, because no design costs nothing at
32 sequences with images in the load. Lent pages holding live sequences'
KV need preemption or copies to empty them for the encoder. Lent pages
holding only prefix-cache snapshots need neither, but every image request
then evicts the anchors they hold, leaving fewer anchors than a pool
without lending, and none when long shared prefixes fill the cache. A GLM
deployment that serves no images gets this memory back for its KV pool with
the text package (`--modalities text`): all 3,387 MiB per rank when it runs
alone, but 2,304 MiB (the encoder buffers) while a weight keeper that loaded
the image package still holds the encoder weights; the measured pool grew
from 5,961 to 8,265 MiB in that case.

### Lossless weight encodings

A bundle that exports a weight census may store some weight sections in a
smaller lossless encoding, which the host measures on the CPU at startup
([direct weight loading](../BUNDLE_ARCHITECTURE.md#direct-weight-loading)). DeepSeek V4.1 stores
its main layers' routed expert scale banks encoded and expands them on the
device per layer; outputs are bitwise equal to raw banks
([numerics](numerics.md#lossless-weight-encodings)). This is on by default. At TP4 it frees about
3.3 GiB per GPU for the KV pool, adds the census to startup (about 7 s),
and on 2026-10-23 cost 2 to 4 % of decode tokens per second at 1, 8 and 32
sequences. Set `CHOPINFER_WEIGHT_CENSUS=0` in the server's environment to
plan every section raw.

## Host allocator memory

On Linux with glibc, both serving CLI programs set `M_ARENA_MAX` to 8 at
process startup, before parsing the serving options, loading a model or
starting threads. This limits allocator arena proliferation under threaded
loads. It does not impose a byte limit on an arena or on the process.

Set `CHOPINFER_MALLOC_ARENA_MAX` to a positive 32-bit integer to choose
another limit. Set it to `0` to leave glibc's startup policy unchanged,
including any glibc environment settings. Invalid values refuse startup.
The startup log reports the selected limit. This policy applies only to the
`serve` CLI command; other commands, library embedders and non-glibc hosts
retain their allocator policy. Changing the limit requires a new process:
a lower cap does not remove arenas that already exist.

On Linux with glibc, the generation worker returns unused allocator pages
with `malloc_trim(0)` after requests, deferred events and KV transfers have
drained and no command is waiting, when the allocator reports at least
64 MiB free. It runs at most once per minute after activity. If the last
burst ends during that cooldown, an idle timer performs
the reclamation when the minute ends, without needing another request.
The log reports whether pages were returned and the time spent trimming.
The threshold limits low-benefit calls, not their maximum duration. A request
arriving during a trim waits for that one call; the cooldown does not hold it.

Live allocations and their addresses remain intact. This does not evict
prefixes, release GPU pages or cap live caches. A continuously busy worker
has no drained boundary, so its allocator retention still needs separate
soak evidence. Report anonymous memory, file-backed RSS and live heap bytes
separately when judging stability.

## Qualification scope

Qualification is tied to actual checkpoint tensors, package, host, topology,
settings and workload. A compatible checkpoint needs its own evidence.
Eight-rank, multi-node, MPS and other GPU SKUs are unqualified. TP2 profiles
remain experimental, even when a release validation or bounded gate passed.
Qwen TP2 needs 51.2 GB of pinned host memory; see [setup](developer-preview.md#pinned-host-memory).

Text and image are default inputs. MiMo can also serve video/audio with all
modalities, but audio is not qualified in-tree: there is no end-to-end audio
reference input. Preprocessing tests alone do not qualify audio serving.
Offload is opt-in; reusable host prefix caching remains refused. Different
batch buckets can change output bits under concurrency; see
[numerics](numerics.md#batch-shape). Single-request and fixed-batch guarantees
do not extend to arbitrary concurrent batches.

## MiMo evidence

MiMo is outside the release scope and gets no further GPU qualification;
this section is history. The pinned checkpoint is MiMo V2.6 Flash MOPD.
Historical RL checkpoint runs do not qualify its tensors. `tp4-mtp` has
historical serving evidence only, workload-specific throughput regressions
and no accepted paired arm; it remains experimental. `tp2-ar` also remains
experimental.

### TP4 concurrency qualification, 2026-10-04

MiMo and Qwen TP4 AR passed C64 paired and absolute teacher checks, decode
bucket comparisons, prefix equality, preemption and stability checks with
BF16 KV and text/image. This is serving evidence through C64 for the
recorded implementations, not blanket production qualification or evidence
for later build identities. Qwen MTP remains qualified through 32 by its
throughput policy despite wider capacity.

### Row-local decode requalification, 2026-10-05

MiMo's row-local decode passed TP4 AR C64 accuracy, preemption and stability,
and TP2 AR accuracy. DFlash failed the speculative pooled logprob gate;
row-local verify did not remove the excess. Repeating fixed rows established
repeatability, not a pass. The eventual qualification uses the waiver below.

### Accuracy waiver, 2026-10-06

The maintainers waived MiMo DFlash's pooled speculative logprob mean after
four failures against the 0.005-nat margin. The requalification measured
+0.0057 nats (CI95 [0.0003, 0.0162]); its paired and absolute teacher gates,
each draw's top-one bound and coverage passed. The gate was waived, not
passed. Sampling evidence at temperature 1.0 and top-p 0.95 found comparable
top-5 nucleus variation to AR's own batch changes, covering only positions
whose nucleus fit in the recorded top five.

The subsequent C64 run passed the remaining teacher, preemption, stability,
prefix and coverage checks, and `tp4-dflash` is qualified through 64 with
that exception. This waiver covers only the pooled four-draw mean, not the
single-record predicate in `qual.json`. It must be declared there when the
pooled gate is declared. The later speculative-continuation gate adds no
paired margin and leaves the waiver intact; candidate-minus-AR and KL
figures are informational. See [the contract](numerics.md#speculative-continuations).

## Qwen evidence

The TP4 AR and MTP profiles have BF16-KV text/image serving evidence; their
qualified counts are separate from the default cap. TP2 and both TP4 DCP2
profiles remain experimental. Qwen's DCP2 bounded gate passed on 2026-10-07;
that did not make it release-qualified. Context-parallel sizes use separate
qualification and [stripe semantics](numerics.md#context-parallelism).

### Long-context qualification

The pinned Qwen 128K check scores the last 1,024 positions of a 131,072-token
prompt against its GPU F32 reference. Its declaration alone enables the
F32-only floor policy; a missing BF16/f64 spread in any other check must not
silently enable that policy. Reference bytes and workloads remain locked. Its mean-error floor is 0.10,
decisive and near-tie flip floors are 0.08, and the total flip floor is 0.10.
DCP2 additionally requires TP4, BF16 KV and context-parallel size two.

## GLM evidence

GLM serves TP4 only. Three-draft MTP, AR, DCP2 and DCP4 are qualified through
32. Five-draft MTP (`tp4-mtp-k5`) remains experimental, text-only and never
the default, even though its later declared checks passed. No profile has
C64 qualification. There is no DCP4 chat waiver.

### Final gate, 2026-10-11

The 2026-10-11 gate qualified the default serving configuration except its
plain soak host-growth limit. The bounded prefill-program cache was accepted
under an explicit warmup rule: after 180 seconds, host and GPU growth must
each stay within 512 MiB; whole-soak limits additionally allow 2,560 MiB
host and 1,344 MiB GPU. This is a bounded cache exception, not an unlimited
soak allowance. Profile metadata preserves that historical decision.

### Release qualification

The full 2026-10-14 to 2026-10-15 requalification passed default, k5 and DCP2.
The taps and DCP4 configurations each had one accepted review row and passed
the rest. These accepted rows were not waivers and did not change limits:
the taps row was a near-tie routing flip; DCP4's KV pressure row had zero
prefix evictions against at least one, while its other 24 checks passed.
The missing eviction was a timing-sensitive pressure indicator, not a
correctness failure. The routing comparison now has its explicit
[near-tie rule](numerics.md#routing-near-ties).

### BF16 KV

BF16 latent KV on `tp4-mtp --kv-cache-dtype bf16` passed requalification on
2026-10-16; FP8 remains the default. Its earlier held run was not retroactively
qualified by rescoring: the declared checks were rerun after the small-category
rule was adopted. Profile evidence for FP8 alone does not establish BF16 KV.

## Prefill policy

`--prefill-policy fixed|tbt|adaptive` selects how prefill shares a step
with decoding requests. The profile field is `prefill_policy`; omission
means `fixed`. Explicit flags override the profile.

| Policy | Mixed-step sizing | Timing cuts |
| --- | --- | --- |
| `fixed` | Configured token and chunk limits | Off |
| `tbt` | Existing time-between-token step model | Off |
| `adaptive` | TBT cap sized from prefill demand | On with a grid and negotiated cuts; off without a grid |

`--decode-tbt-target-ms` applies to `tbt` and `adaptive`, defaults to 100 ms,
and accepts 1 through 60,000 ms. Serving fixes the progress floor at 512
prefill tokens and extra decode-only steps at zero. Switching between
`tbt` and `adaptive` retains an inherited target. An explicit `fixed`
clears an inherited target; specifying a target explicitly with `fixed`
is an error, as is a profile declaring both.

MiMo's `tp4-mtp` and `tp4-dflash` use `tbt` at 150 ms. DeepSeek's
`tp4-dspark` uses `adaptive` at 750 ms. Every other shipped profile uses
`fixed`. `profiles show`, the launch catalog and `serve --check` report the
canonical policy. Startup prints `prefill policy: fixed|tbt|adaptive` after
bundle capability negotiation.

A grid bundle without negotiated prefill cuts refuses `adaptive`. Without
a grid, adaptive sizes the ordinary TBT cap and needs no cut capability.
For eligible grid chunks beside decoders, a cut ends at the last prefill
granule multiple the cap reaches, or the first multiple at or above the
512-token floor if that fits. Timing cuts capture no prefix anchor;
scored and media requests keep their chunks. Neither policy selection nor
spare capacity establishes accuracy or performance qualification.

`adaptive` takes `--decode-tbt-target-ms`
as a decode cadence target: a decoding request's gap is the step it
shares, so each step beside decoding
requests takes the largest legal cap (the progress floor, each 128-token
multiple up to what the waiting requests' next chunks can take, or no cap)
whose predicted time stays within a bound that depends on the prefill
backlog. A backlog that uncut chunks are predicted to clear in more than
4 s is sustained, and its steps are sized to keep the
predicted step within the target. A shorter, transient backlog lasts a few
steps, so its steps may run past the target: the predicted step with a
margin within four thirds of the target (1 s at 750 ms). The margin is the
90th percentile of the step model's last 128 relative prediction errors, at
least 5%, so a prediction close to the ceiling counts as crossing it. No
cap is taken only when the uncut step keeps the bound and its chunks share
the step with the decoding requests; a grid chunk too long to share the step
is always cut rather than run alone. Both bounds are estimates: a larger
prediction error, a progress floor or granule boundary that forces a
longer chunk, too little decode room for any cut to share the step, and
scored or media prefills, which keep their own chunks, can all run past
them. When no cap keeps the bound, as at a long context where even the
floor's step is predicted past it, the step takes the progress floor, so
the target is not a latency bound. No smaller cap is taken, since each extra
step adds its fixed cost to the prefill. Uncapped, an item may also run past
an anchor to its prompt end, and that longer step is what is priced. A
sustained backlog also prefills the prompt with the fewest uncut chunks
left first (chunks end at absolute grid boundaries, or are chunk-limit
sized), the earliest arrival among equals, so a prompt that finishes in one
chunk goes ahead of a longer one while long prompts keep their order; a
prompt queued for 30 s of step time goes first on every second step that
takes prefill (owed decode-only steps do not count). A transient backlog
keeps the served order. Without decoders the step is not capped,
and until the step model trains the cap is the progress floor. Its useful
targets in the measured adoption were about 500 ms to 1 s. It applies to
every model: a bundle with a prefill grid needs negotiated cuts and ends
grid chunks at the chosen cap
(startup prints `prefill timing cuts on (adaptive): grid chunks ...`); a
bundle without a grid takes the cap as its ordinary tbt prefill budget and
takes no timing cut (`prefill timing cuts on (adaptive): the TBT cap ...`).
A grid bundle without cuts refuses `adaptive`.

### Migrating retired controls

Remove retired flags and profile fields before the next deployment. They
fail at startup with guidance naming `--prefill-policy` or `prefill_policy`;
there are no aliases, including `strict` or `target`.

| Old budget / timing cuts | New policy |
| --- | --- |
| `fixed` / `off` | `fixed` |
| `tbt` / `off` | `tbt` |
| `tbt` / `adaptive` | `adaptive` |
| `decode-aware` / `off` | No exact replacement |
| `tbt` / `fixed` | No exact replacement |

Replace `--prefill-budget` and `--prefill-timing-cuts` (profile fields
`prefill_budget` and `prefill_timing_cuts`) together. Remove
`--prefill-itl-target-ms`, `--prefill-decode-steps`,
`--mixed-min-prefill-tokens` and `--mixed-decode-steps`. A deployment that
used a nondefault floor or extra decode steps has no identical public
configuration. Lower-level scheduler library controls remain for tests.

Decode-aware and fixed timing cuts were retired to reduce setup choices.
This is not a performance claim: fixed timing cuts had lower sixteen-session
collapse share, while adaptive improved short-append TTFT in the history
below. The historical `fixed` timing-cut arm is distinct from today's
`fixed` token-budget policy.

### Adaptive timing-cut adoption

The following measurements use the historical timing-cut arm names
`off`, `fixed` (now retired) and `adaptive`. DeepSeek's `tp4-dspark` serves
`adaptive` by default after a stateful
replay A/B on 2026-10-30 against `off` and `fixed`. Every arm ran TP4
DSpark on the deepseek-ai checkpoint with native KV, prefix caching,
`--prefill-budget tbt` at a 750 ms target, the 512-token progress floor,
32 sequences and temperature 1. Each cell ran two seed pairs in ABBA order;
every measured visit followed its own arm's exact-seed warm-up and a cache
check in fresh processes, within 1,000 major faults. A comparison
resolves a gain when both pairs improve and their mean exceeds their
spread. Before any timing, off and adaptive produced bit-identical output
tokens and logprobs on a cold burst and at one stream, and the
fixed-batch replay, prefix-cache exactness suite and determinism screens
passed. In every timed cell, each matched turn of the two arms had equal
prompt and output token IDs, answers and logprobs.

- **Sixteen prefill-saturated sessions, three turns each** (24 short and
  24 long appends). Against `off`, collapse share fell from 14.4% and
  12.7% to 1.4% and 2.0%, gaps over 1 s from 28.7 and 24.3 to 0.1 and 0.0
  per 1,000 gaps, and short-append TTFT p50 from 3.09 and 2.86 s to 2.11
  and 2.39 s: three resolved gains. Long-append TTFT p95 rose from 14.8
  and 14.6 s to 18.4 and 18.1 s, output tokens per second fell 9.9% (just
  under the 10% bound beyond which the outcome would have gone to the
  operator), and energy per output token rose from 3.78 and 3.79 J to 4.13
  and 4.04 J. Against `fixed`, short-append TTFT p50 fell from 3.27 and
  3.02 s to 2.11 and 2.39 s, while collapse share rose from 0.7% and 1.4%
  to 1.4% and 2.0%; mean output rate was 0.6% higher, unresolved.
- **Eight steady sessions, 120 turns** (every third append 8,192-12,288
  tokens, 40 long appends per visit, think times up to 30 s). Against
  `off`, gaps over 1 s fell from 1.85 and 1.40 to 0 per 1,000 gaps (a
  resolved gain); collapse share and short-append TTFT were unresolved.
  Long-append TTFT p95 moved from 3.68 and 4.80 s to 4.17 and 4.36 s, a
  mean increase of 1.9% against an accepted tolerance of 23.7%, and output
  tokens per second fell 1.6% against a 3% bound. Stalls over 250 ms rose
  from 10.8 and 12.6 to 17.6 and 17.9 per 1,000 gaps, the tenth-percentile
  stream rate fell, and energy per output token rose from 3.38 and 3.36 J
  to 3.50 and 3.43 J.
- **A burst after a cache loss** (five 250K and five 30K prompts arriving
  together, no priming). Against `off`, collapse share fell from 22.9% and
  20.9% to 1.9% and 2.8%, gaps over 1 s from 70.3 and 70.7 to 0 per 1,000
  gaps, short-prompt TTFT p50 from 25.5 s to 12.9 s in both pairs, and the
  last 30K prompt's first token from 26.4 and 26.3 s to 18.6 and 18.5 s.
  The last 250K prompt's first token moved from 156.8 s to 217.9 and
  217.6 s, output tokens per second fell 24.6% and energy per output token
  rose from 19.6 J to 24.7 J; the burst rule reports these but does not
  gate them.
  Against `fixed`, short-prompt TTFT p50 halved and the last 30K prompt's
  first token came 34% sooner; the last 250K prompt waited about 3% longer
  and output rate was 2.3% lower.

The sixteen-session cell ran in one window on 2026-10-30, and the
eight-session and burst cells in a second window the same day. Before the
second window, an amendment declared that a measured window is judged by
outside idle: host idle plus the job's own process-tree CPU at least 95%,
with CPU and I/O stall shares at most 0.1. In the burst's `off` visits the
server's own prefill CPU, about 5 cores, left host idle near 0.89 by
itself. Host clock steps also no longer refuse a visit, since no judged
metric reads server wall-clock time. The A/B measured the adaptive change
on an earlier integration base. Later integration changes, including
prefix-cache segment reclamation, drafter planning and sparse-attention
updates, were not part of the measured build. `off` and `fixed` remain
historical comparison arms: `fixed` had the lowest collapse share in the
sixteen-session cell, and `off` had the highest output rate in all three
cells.

These cells ran with the four GPUs limited to 275, 250, 250 and 250 W. On
2026-11-02, at the 300 W default limit on main 0c620f1a66, a replay of the
same cells was set up to compare the 750 ms target with 1000 ms, both
adaptive. Before any timing, the two targets produced bit-identical output
tokens and logprobs on a cold burst and at one stream, and the
prefix-cache exactness suite and determinism screens passed at each
target. No job in that gate took a timing cut, so it did not move a chunk
boundary. The replay was stopped before any cell produced a valid
comparison, so it has no verdict, and `adaptive` against `off` was not
rerun at 300 W. `tp4-dspark` keeps `adaptive` cuts at 750 ms on the
evidence above; neither setting has been re-measured at 300 W.

### Fixed timing-cut tradeoff

On 2026-10-28, DeepSeek TP4 DSpark stateful replay at C8 (eight sessions,
three turns each, 32K/64K contexts, session-specific cache salts,
temperature 1, 150 ms TBT target and 512-token progress floor) resolved a
tradeoff for `fixed` against `off`. Stall time above 250 ms per 1,000 gaps
fell 75.7% in the first pair and 84.0% in the second, and no gaps exceeded
1 s with `fixed`. TTFT p95 rose from 1.764 to 3.061 s (+73.6%) and from
1.911 to 3.046 s (+59.4%). Energy per output token rose from 3.353 to
3.764 J (+12.3%) and from 3.440 to 3.746 J (+8.9%). Output throughput
fell 4.0% and 3.0%, averaging 3.5% across the two pairs.

These percentages compare the measured quantities within each pair. The
250 ms stall threshold was declared before measurement. Stalls between
100 and 250 ms became more frequent. The now-retired fixed timing-cut
policy traded smoother concurrent streaming for first-token latency,
throughput and energy.
This evidence covers that cell and configuration; it does not support a
default change in any profile. The `tp4-dspark` default comes from the
[adaptive adoption](#adaptive-timing-cut-adoption) A/B.
