# Serving profiles

Each model owns `models/<model>/profiles.json` (schema
`chopinfer-serve-profiles-v2`; unknown fields are rejected). A profile records
the bundle and topology, rank and device mapping,
speculation (mode, tokens, cost, drafter), sequence limits, KV dtype, prefix
caching, batch and prefill budgets, queue depth, KV memory fraction, and its
status and evidence. An optional `model_options` map carries package
option defaults, and an optional `context_parallel_size` (absent means 1)
the `--context-parallel-size` default: it must divide the tensor-parallel
size, and `--profile` refuses a package that does not declare it for the
profile's topology. Explicit serve options override every field.

A v1 file is refused with a migration message: drop `fold_ranks` and set the
schema to `chopinfer-serve-profiles-v2`. The v2 schema has no `fold_ranks` field.

Qwen3.8 and MiMo serve TP2 and TP4 only. With more GPUs, run
[separate instances on disjoint GPUs](deploy-common.md#package-topology-and-separate-instances).
Shared 8-rank communication support remains for a model validated on eight
GPUs; it is unvalidated for these models.

```sh
chopinfer profiles list models/mimo26/profiles.json
chopinfer profiles show models/mimo26/profiles.json tp4-dflash
```

`list` prints the choices and their limits; `show` prints one profile as JSON
with its effective prefill budget. Both validate every profile through the
serve argument parser without a GPU.

Every shipped profile serves 32 sequences by default (maintainer decision,
2026-10-14). MiMo and Qwen `tp4-ar` and MiMo `tp4-dflash` are qualified
through 64; `--max-sequences 64` opts into that. All use KV `auto`, prefix
caching on and an unbounded waiting queue. MiMo `tp4-mtp` and `tp4-dflash` use the TBT prefill budget with a
150 ms target; other shipped profiles use the fixed budget. `devices` are
distinct ordinals within `CUDA_VISIBLE_DEVICES`, one per rank. Duplicate
device ordinals are refused, and `devices_required == tensor_parallel`.

`max_sequences` is the serving cap. `qualified_max_sequences` is the largest
concurrency the cited evidence covers, or null when only smoke evidence
exists; startup warns when an explicit cap exceeds it. A `qualified` status
means the cited serving checks ran, not that a production release passed.

## Profile status

Every profile in `models/*/profiles.json`. A `qualified` profile's cited
evidence covers concurrency up to its `qualified_max_sequences`; an
`experimental` profile has no qualified count (`none` here, null in the
file). `chopinfer profiles list` prints the current values.

| Model | Profile | Status | `max_sequences` | `qualified_max_sequences` |
|---|---|---|---:|---:|
| GLM 5.3 Flash | `tp4-mtp` | qualified | 32 | 32 |
| GLM 5.3 Flash | `tp4-mtp-k5` | experimental | 32 | none |
| GLM 5.3 Flash | `tp4-ar` | qualified | 32 | 32 |
| GLM 5.3 Flash | `tp4-mtp-dcp2` | qualified | 32 | 32 |
| GLM 5.3 Flash | `tp4-mtp-dcp4` | qualified | 32 | 32 |
| MiMo V2.6 Flash | `tp4-ar` | qualified | 32 | 64 |
| MiMo V2.6 Flash | `tp4-mtp` | experimental | 32 | none |
| MiMo V2.6 Flash | `tp4-dflash` | qualified | 32 | 64 |
| MiMo V2.6 Flash | `tp2-ar` | experimental | 32 | none |
| Qwen3.8 Flash | `tp4-ar` | qualified | 32 | 64 |
| Qwen3.8 Flash | `tp4-mtp` | qualified | 32 | 32 |
| Qwen3.8 Flash | `tp4-ar-dcp2` | experimental | 32 | none |
| Qwen3.8 Flash | `tp4-mtp-dcp2` | experimental | 32 | none |
| Qwen3.8 Flash | `tp2-ar` | experimental | 32 | none |
| Qwen3.8 Flash | `tp2-mtp` | experimental | 32 | none |

## Selecting a profile

`chopinfer serve --model DIR` applies the profile the family helper selects
(see [compatible checkpoints](compatible-checkpoints.md)) and refuses
`--profile`. An explicit package takes one with `--profile NAME` or
`--profile FILE#NAME`:

```sh
# PACKAGE is the --output of mimo26-package for the same checkpoint.
chopinfer serve --bundle "$PACKAGE" --checkpoint "$CHOPINFER_MIMO_CHECKPOINT" \
  --profile models/mimo26/profiles.json#tp4-ar
```

A bare `NAME` is one of the profiles the package carries: the compiler emits
the model's `profiles.json` into every package as its verified
`serve-profiles` component, read only when its bytes match the manifest.
`FILE#NAME` names another file instead; `FILE` is relative to the current
directory. The profile must serve the package's bundle ID and a topology the
package declares at the profile's rank count.

Both routes apply a profile the same way: its device map,
speculation and scheduling settings are defaults under explicit options, in
any order. The served name is never taken from a profile. An explicit
`--speculative-tokens 0` drops the profile's speculation policy. With
speculation on, startup refuses a package whose `speculation.drafter_kind`
does not match the profile's drafter; emit the package again rather than
editing its manifest. Package manifests
stay authoritative for limits, topology and capabilities, and a profile never
grants qualification to overrides.

## Model options

A profile may include `"model_options": {"scratch_multiplier": 1}` when its
package declares that option. Keys name checkpoint configuration fields;
values are JSON scalars, converted to flag text and parsed as the declared
type. The host validates them against the selected manifest before loading
weights. Profile loading checks argument
syntax; it cannot establish package-specific ranges by itself.

`--profile` also checks the profile's own map against the package's
declaration before any explicit option replaces an entry, and refuses a
profile whose values the package does not accept (or a package that
declares no options). An option the package declares with one allowed value
therefore binds every profile that sets it to packages with that value.

Repeat `--model-option NAME=VALUE` to replace individual profile entries.
An override of one name retains defaults for other names. Precedence is
explicit flag, `CHOPINFER_MODEL_OPTIONS`, profile map, package default.
Docker and systemd deployments may set, for example,
`CHOPINFER_MODEL_OPTIONS=scratch_multiplier=1` for the fixture package.
Names must be unique in each flag list or environment list. String flag
values are unquoted; the profile map uses JSON strings.

A profile cannot qualify an option override. Qualification covers declared
qualified defaults only; a non-default effective value is reported as
unqualified and refused by eval and receipt admission. This addition keeps
v2 profiles without `model_options` loadable.

## Quickstart helper

```sh
tools/quickstart.sh mimo26 tp4-ar "$CHOPINFER_DATA"
# Or: tools/quickstart.sh qwen38 tp4-mtp "$CHOPINFER_DATA" --verify
```

The helper builds the host, the model's helper and native module with locked,
offline Cargo, installs them into a fresh private directory under
`DATA/releases`, asks the helper which profile `serve --model` would use, with
`--verify` checks the checkpoint files, and runs `chopinfer serve --model` in
the foreground with the model cache `DATA/model-cache`. It refuses a profile
the helper would not select and names the one the checkpoint serves (a
mixed-precision Qwen checkpoint whose MTP layer stores a format without a
composition serves `tp2-ar` or `tp4-ar`); an autoregressive choice serves the
selected profile with speculation off.
Serving requires `CUDA_VISIBLE_DEVICES` unless `--allow-all-gpus` is given.
Pass serve overrides after `--`, for example `-- --max-sequences 16`.
`tools/quickstart.sh --help` lists the other options.

## Measured speed

Snapshot: `v0.1.0-alpha.2`, 2026-10-03, from the final Docker validation.
The host has 4x RTX PRO 6000 Blackwell Max-Q, 96 GB each; TP2 uses two GPUs.
Graphics clocks are locked at 1,600 MHz and power at 250 W per GPU (Max-Q),
so these are not peak numbers. Actual clocks can fall under the power cap.

All rates are tok/s; C is concurrency. Decode uses chat prompts and 64-token
outputs from the `spec_bench` workload: median over three measured passes
after warmup, with no cached prefix. Its cohort rate is concurrency times
output tokens divided by the longest request, including time to first token.
Prefill uses unique 8,192-token prompts and one output token, measured from
first submission to last completion.

| Configuration | Decode C1 | Decode C16 | Decode C32 | Prefill 8K C1 | Prefill 8K C8 |
|---|---:|---:|---:|---:|---:|
| MiMo tp4-dflash | 343.1 | 1202.1 | 1721.6 | 10645.1 | 13645.7 |
| MiMo tp4-ar | 216.2 | 1090.3 | 1497.8 | 11023.3 | 13807.3 |
| MiMo tp2-ar | 139.4 | 659.4 | 1003.0 | 6555.6 | 8913.3 |
| Qwen tp4-mtp | 239.7 | 1151.8 | 1418.9 | 9672.0 | 11116.9 |
| Qwen tp4-ar | 131.4 | 1056.9 | 1483.7 | 9976.8 | 11445.8 |
| Qwen tp2-mtp | 184.9 | 872.3 | 1135.2 | 9196.7 | 10973.3 |
| Qwen tp2-ar | 105.6 | 780.5 | 1155.2 | 9190.0 | 11279.5 |

Adaptive speculation may serve AR at high concurrency; a profile name does
not mean every step drafts. These rates describe this workload and tag.

## TP4 concurrency qualification, 2026-10-04

Since 2026-10-14 every profile defaults to 32 sequences. The 64-sequence
defaults described below are now qualified counts that `--max-sequences 64`
opts into.

The 2026-10-04 TP4 concurrency qualification passed the C64 accuracy, prefix, preemption and stability checks
for MiMo AR/DFlash and Qwen AR/MTP on the pre-checkpoint-change build.
MiMo AR defaulted to 64 and is qualified through 64. MiMo DFlash did too
until its requalification on the row-local decode build failed
([below](#row-local-decode-requalification-2026-10-05)). It is now
qualified through 64 again, with its speculative accuracy gate waived by a
maintainer decision ([below](#accuracy-waiver-2026-10-06)). Qwen AR
also defaulted to and is qualified through 64: its dedicated rerun passed
accuracy, prefix, stability, preemption and performance on the frozen
recurrent checkpoint-by-index implementation. Qwen MTP's default and
qualified count remain 32: its old-build C64 gates do not qualify the later
implementation, and its throughput policy favors 32. MiMo MTP stays at 32
because its drafter and package limit are 32;
its paired policy has no accepted arm. TP2 defaults remain 32. These are
bounded serving qualifications, not full production certification.
The C64 evidence predates the geometry refactor, which landed with
byte-identical cubins/SASS.

The C64 gates used the independently bound GPU teacher reference, complete
paired teacher captures against the pinned alpha.2 accepted arms, three
B64-versus-B32 decode repeats, pooled native speculative rows, full prefix
position coverage at concurrency 8, serial continuation/acceptance checks,
C1 preemption bit identity, a queued burst, all 200 frozen GSM8K examples and
a 600-second prefix/structured soak. The MiMo AR B64 error increases were
-0.001177, +0.003893 and +0.002426 nats; Qwen AR was +0.002838 in all repeats.
Pooled DFlash/MTP increases were +0.002347/+0.004014 nats, within the existing
0.005-nat rule. No margin or reference changed. Accepted alpha.2 identities
and captured settings remain pinned; promotion does not relabel their C32
captures or qualify a different build's implementation identity.

Measurements below compare C32 and C64 within the same C64 server start,
with identical bodies across profile arms, BF16 KV, text,image, prefix caching
and an unbounded waiting queue. Each entry is the median of three measured
passes after warmup. Requests use unique prefixes and exact output lengths.
Cohort tok/s includes TTFT. TPOT is (latency - TTFT)/(output tokens - 1);
the listed p50/p95 values are the medians of each pass's request percentiles.
Energy is the integral of summed four-board NVML power over each cohort's
submission-to-completion window, sampled every 200 ms. It includes prefill;
it is not an isolated decode-kernel efficiency measurement. Hardware is the
same four 96 GB Blackwell Max-Q GPUs described above. Evidence is the
2026-10-04 C64 qualification; the alpha.2 table remains a separate historical snapshot.

B0 denotes the frozen native build of the 2026-10-04 qualification, before
the Qwen checkpoint-index change. B1 denotes the Qwen checkpoint-index C64
rerun of 2026-10-04. Every measurement row identifies its build.
Qwen AR rows now use B1; MiMo and Qwen MTP rows retain B0 evidence.
B1's paired teacher capture is bit-identical to its alpha.2 accepted arm;
its three B64 error gaps are +0.002838 nats, unchanged from B0. The full
200-example GSM8K screen passed with 195 correct; the 600-second soak
completed 10,479 requests, full prefix equality passed, and all 36 C1 preemption
continuations were bit-identical. Build-scoped evidence does not qualify
an arbitrary later implementation identity.

With 256-token outputs, MiMo AR gained 45-49% tok/s and 43-50% tok/J;
DFlash gained 27-45% tok/s and 25-46% tok/J; Qwen AR gained 15-17% tok/s and
19-28% tok/J on the checkpoint-index rerun. Latency rises with concurrency.
Qwen MTP's 256-token GSM8K
rate fell from 1821.8 to 1692.4 tok/s (-7.1%), and its 64-token rate was
flat. At C64, its TPOT nearly matches AR, consistent with adaptive fallback.
That tradeoff keeps MTP at 32 despite passing the C64 gates. On main
(2026-10-05, cap 64 servers, GSM8K 256-token outputs) MTP rose from 1435.4
tok/s at C32 to 1647.7 at C64, 2.4% below AR's 1687.9; 92-100% of its
steps at C32 and C64 drafted nothing. The MTP drafter's catch-up runs on
those steps took 1.6% of device time (nsys), worth about 27 tok/s. A
profile with at most 32 sequences caps verify rows at 32, so with 32
decoding requests MTP cannot draft at all and serves AR. Qwen C128 is
not pursued: C64 already raises AR TPOT by roughly 72-74% for a 15-17% rate
gain. MiMo stayed at 64: the C128 evaluation failed its accuracy gate.

On the C128 server, B64 repeat 3 had mean absolute error 0.0629 nats
against its C32 control's 0.0552: +0.0077 nats (CI95 [0.0009, 0.0212]),
above the unchanged 0.005-nat margin. Other B64 gaps were +0.0026/+0.0033;
the completed B128 gaps were +0.0035/+0.0025. B128 repeat 3 did not run,
and every later C128 job stopped on dependencies. There is no qualified
C128 stability or performance evidence and no promotion or rerun to obtain
a pass. Both MiMo AR and DFlash kept their C64 defaults and evidence then.

B64 mean error was 0.059-0.063 on the C128 server versus 0.056-0.0575
on the C64 server, despite matching executed `row_capacity: 64` traces;
C32 controls remained similar. Server-cap effects, batch composition and
repeat variation are not isolated by these captures. The C64 qualification
covers the tested C64 configuration, not a B64 bucket on every server cap.
Evidence is the 2026-10-04 MiMo C128 run; diagnostic work must preserve this failure.

On B1 with 64-token outputs, Qwen AR chat gains 18% tok/s and 31% tok/J.
GSM8K throughput is nearly flat (+0.7%) and tok/J falls 8%; the C64 default
therefore does not improve every workload. The exact-length tables retain
these latency and energy tradeoffs.

At C64 with 64-token outputs, DFlash remains slower than AR: 2164 versus
2347 tok/s on chat and 1547 versus 1780 on GSM8K. Speculation has
workload-specific throughput regressions.

The release preview retains the helper-selected speculative profile for
its AR arm and disables speculation. For Qwen that is MTP at 32, so the
preview does not exercise the AR profile at 64. Its report states this gap;
the dedicated checkpoint-index C64 rerun supplies that separate evidence
for its frozen build. The benchmark sweeps still include queued clients
above each serving cap.

### Row-local decode requalification, 2026-10-05

MiMo decode attention and experts became row-local
([numerics](numerics.md#batch-shape)), and a requalification on 2026-10-05
checked that build with unchanged margins. TP4 AR passed C64 accuracy (B64 against C32 +0.0031
nats in all three repeats), preemption and stability; TP2 AR passed
accuracy. TP4 DFlash failed the C64 pooled speculative bucket gate: +0.0085
nats against the 0.005 margin (CI95 [-0.0009, 0.0275]); its paired teacher
passed, and the previous build had measured +0.0023 once. `tp4-dflash`
returned to 32 sequences.

Verify attention and the DFlash drafter's block attention then became
row-local ([numerics](numerics.md#batch-shape)), and a second run on
2026-10-05 repeated the DFlash C64 accuracy stage on that build: the paired and absolute
teacher passed again, and the pooled gate failed again at +0.0082 nats
(CI95 [0.0001, 0.0233]). A second server on the same build captured the
same 16,205 forced pooled positions bit for bit, so the pooled rows no
longer depend on batch composition and the excess over AR is systematic;
the earlier +0.0023 was most likely a favorable composition while the gate
still varied with it. The gate never ran at 32, and nothing suggests the
excess is absent there, so `tp4-dflash` keeps 32 sequences as an
experimental profile with no qualified count: DFlash serving is not
qualified at any concurrency. No rerun was made to obtain a pass. Disabling speculation keeps the selected profile's limit, so AR
through `serve --model` on the DFlash profile runs 32 sequences; the
`tp4-ar` profile keeps 64. The DFlash C64 rows below remain B0 measurements.

A later run on 2026-10-05, on main plus the several-draw AR control
([numerics](numerics.md#serving-qualification-gates)) reran TP4 accuracy,
preemption and stability. AR rows were bitwise equal across repeats and
between the C64 and C128 servers. Against the mean of the C32, B8, B16 and
B64 AR draws, AR B64 measured +0.0034 nats and AR B128 +0.0025, both
passing; DFlash pooled at C64 measured +0.0076 and failed again, so
`tp4-dflash` stays experimental at 32. AR C128 also passed preemption, GSM8K
and spec_verify, but failed the 600-second soak at 128 streams on host
memory growth: 761 MiB against the 512 MiB bound. The serving process
grew about 750 MiB in the first 30 seconds of load and about 10 MiB over
the remaining 570 (C64: 411 MiB, then about 8 MiB), so the excess is the
larger first-load working set at 128 streams rather than a leak. The
bound is measured from a sample taken before load, and changing it is a
gate decision, so `tp4-ar` stays at 64 and no rerun was made.

On 2026-10-06 the DFlash C64 gate became the mean of four pooled draws (C8,
C16 and C32 without filler, and B64) against the mean of the four AR draws,
margin unchanged ([numerics](numerics.md#serving-qualification-gates)). That
rule was adopted after the three single-draw failures above, and the
recorded receipts projected +0.0044 under it. A run on 2026-10-06 then
ran AR and DFlash C64 accuracy on the commit that added it. AR C64 accuracy passed, with B64 at +0.0034 in all three repeats and AR
rows bitwise equal across repeats. DFlash passed the paired and absolute
teacher, each draw's top-one bound and coverage, but failed the pooled gate:
+0.0057 nats (CI95 [+0.0003, +0.0162]) against 0.005. Per draw: C8 +0.0051,
C16 +0.0026, C32 +0.0074, B64 +0.0076. B64 was bitwise equal to the earlier
build's draw. The unfilled draws were not: C8 and C16 sat 0.003 and 0.002
above their earlier values, the timing-dependent swing the pooled gate was
meant to average out. As the rule was fixed before the run, no rerun was
made. The preemption, stability, prefix and performance jobs were not run.
`tp4-dflash` stayed experimental at 32 sequences.

#### Accuracy waiver, 2026-10-06

The maintainers then waived the DFlash speculative accuracy gate (the pooled
logprob mean) and qualified `tp4-dflash` through 32 sequences. The gate was
not passed: it failed four times, and the waiver says so wherever the
profile is described. The reason is that production requests sample at
temperature 1.0 and top-p 0.95, where the excess does not show.
CPU analyses of that 2026-10-06 run's receipts (kept with the
requalification evidence, outside the repository) found:

- **Sampled distributions.** Each position's 0.95 nucleus was built from the
  recorded top-5 and renormalized. Mean total variation from AR C32 is
  0.0064-0.0072 for the four DFlash draws, and 0.0065-0.0071 for AR's own
  B8, B16 and B64 draws. This covers the 2,915-2,924 of 4,053 positions
  whose nucleus fits in the top-5; high-entropy positions are not measured.
- **Where the logprob excess sits.** By the exact probability of the
  reference token: +0.0007 nats above 0.90, +0.0021 at 0.37-0.90, +0.0041 at
  0.007-0.37, and +0.059 below 0.007. That last band is 277 positions and
  71% of the excess, mostly in `chat_tool_call`, where AR also misses
  exact by 2 to 9 nats.
- **Top-1 flips.** DFlash 33-40 per draw, AR 36-50.

The paired and absolute teacher gates, each draw's top-1 bound and coverage
passed.

A further run on 2026-10-06 then ran the seven C64 jobs on main with the
waiver declared in the harness: the pooled mean is reported, while every
draw's top-1 bound and coverage, the teacher gates and every other stage
still gate. All seven passed, and `tp4-dflash` moved to 64 sequences,
qualified through 64:

- AR accuracy: B64 at +0.0034 against the four AR draws in all three
  repeats, AR rows bitwise equal across repeats.
- DFlash accuracy: paired and absolute teacher passed; pooled gap +0.0057
  (CI95 [+0.0004, +0.0162]; C8 +0.0051, C16 +0.0026, C32 +0.0074, B64
  +0.0076), reported under the waiver, with every draw's top-1 bound and
  coverage passing.
- DFlash preemption: 36 of 36 requests bitwise identical with preemption
  every 32 tokens; a 128-request burst at cap 64 completed with 256
  preemptions.
- Stability: spec_verify against the AR record passed; greedy GSM8K at
  concurrency 64 scored 194/200 (floor 193); the 600-second soak at 64
  streams completed 14,160 requests without errors, with 349 MiB host memory
  growth against the 512 MiB bound.
- Prefix: prefix_exact passed.

Performance on this build (`spec_bench`, median of three measured passes;
TTFT and TPOT are per-pass medians):

| Workload | Output tokens | C | tok/s | TTFT (ms) | TPOT (ms) | tok/J |
|---|---:|---:|---:|---:|---:|---:|
| chat | 64 | 32 | 1596.4 | 277.6 | 14.34 | 2.21 |
| chat | 64 | 64 | 2035.3 | 437.9 | 22.02 | 2.81 |
| gsm8k | 64 | 32 | 1276.9 | 531.9 | 16.22 | 2.03 |
| gsm8k | 64 | 64 | 1496.6 | 901.8 | 27.06 | 2.16 |
| chat | 256 | 32 | 1597.4 | 284.7 | 16.83 | 2.05 |
| chat | 256 | 64 | 2366.8 | 438.0 | 23.07 | 3.08 |
| gsm8k | 256 | 32 | 2143.8 | 496.1 | 11.51 | 2.85 |
| gsm8k | 256 | 64 | 2628.9 | 929.2 | 19.08 | 3.48 |

AR performance was not rerun in this batch. Against the B0 AR rows below
(an older build), DFlash at C64 is still slower with 64-token outputs (chat
2035 against 2347 tok/s, GSM8K 1497 against 1780) and faster with 256-token
outputs (chat 2367 against 2216, GSM8K 2629 against 2150).

### Exact 256-token outputs

| Profile | Build | Workload | C | tok/s | TTFT p50 / p95 (ms) | TPOT p50 / p95 (ms) | tok/J |
|---|---|---|---:|---:|---:|---:|---:|
| MiMo AR | B0 | chat | 32 | 1488.7 | 209.5 / 211.8 | 20.71 / 20.75 | 1.62 |
| MiMo AR | B0 | chat | 64 | 2215.5 | 337.4 / 341.7 | 27.59 / 27.67 | 2.43 |
| MiMo AR | B0 | gsm8k | 32 | 1480.9 | 372.4 / 375.4 | 20.20 / 20.22 | 1.73 |
| MiMo AR | B0 | gsm8k | 64 | 2150.4 | 678.2 / 683.0 | 27.11 / 27.18 | 2.48 |
| MiMo DFlash | B0 | chat | 32 | 1690.9 | 263.7 / 378.7 | 16.32 / 18.24 | 2.05 |
| MiMo DFlash | B0 | chat | 64 | 2447.7 | 426.1 / 733.1 | 21.92 / 24.58 | 2.99 |
| MiMo DFlash | B0 | gsm8k | 32 | 2083.0 | 530.5 / 850.0 | 11.34 / 14.39 | 2.68 |
| MiMo DFlash | B0 | gsm8k | 64 | 2646.1 | 903.0 / 1665.2 | 18.72 / 22.02 | 3.34 |
| Qwen AR | B1 | chat | 32 | 1629.9 | 182.6 / 185.6 | 18.96 / 19.39 | 2.36 |
| Qwen AR | B1 | chat | 64 | 1903.6 | 280.4 / 286.6 | 32.55 / 32.64 | 3.01 |
| Qwen AR | B1 | gsm8k | 32 | 1464.2 | 582.4 / 584.9 | 19.64 / 20.08 | 2.24 |
| Qwen AR | B1 | gsm8k | 64 | 1677.9 | 1033.8 / 1038.1 | 34.18 / 34.86 | 2.67 |
| Qwen MTP | B0 | chat | 32 | 1655.5 | 183.8 / 187.4 | 16.98 / 18.45 | 2.57 |
| Qwen MTP | B0 | chat | 64 | 1852.4 | 284.0 / 288.6 | 33.35 / 33.56 | 2.97 |
| Qwen MTP | B0 | gsm8k | 32 | 1821.8 | 399.6 / 402.4 | 14.77 / 15.89 | 2.84 |
| Qwen MTP | B0 | gsm8k | 64 | 1692.4 | 706.7 / 710.6 | 34.91 / 35.53 | 2.66 |

### Exact 64-token outputs

| Profile | Build | Workload | C | tok/s | TTFT p50 / p95 (ms) | TPOT p50 / p95 (ms) | tok/J |
|---|---|---|---:|---:|---:|---:|---:|
| MiMo AR | B0 | chat | 32 | 1644.1 | 207.0 / 209.4 | 16.46 / 16.50 | 1.86 |
| MiMo AR | B0 | chat | 64 | 2346.7 | 331.3 / 335.7 | 22.27 / 22.34 | 2.79 |
| MiMo AR | B0 | gsm8k | 32 | 1290.1 | 370.3 / 373.1 | 19.20 / 19.22 | 1.93 |
| MiMo AR | B0 | gsm8k | 64 | 1779.9 | 674.6 / 680.4 | 25.68 / 25.74 | 2.42 |
| MiMo DFlash | B0 | chat | 32 | 1604.6 | 261.2 / 385.5 | 13.97 / 16.40 | 2.07 |
| MiMo DFlash | B0 | chat | 64 | 2164.4 | 422.7 / 725.3 | 20.53 / 25.16 | 2.78 |
| MiMo DFlash | B0 | gsm8k | 32 | 1323.8 | 526.6 / 837.3 | 14.82 / 20.10 | 2.12 |
| MiMo DFlash | B0 | gsm8k | 64 | 1546.7 | 897.2 / 1661.1 | 26.18 / 36.20 | 2.26 |
| Qwen AR | B1 | chat | 32 | 1662.6 | 177.0 / 180.1 | 16.66 / 16.71 | 2.52 |
| Qwen AR | B1 | chat | 64 | 1967.8 | 273.4 / 277.5 | 28.63 / 28.73 | 3.29 |
| Qwen AR | B1 | gsm8k | 32 | 1336.1 | 393.0 / 395.9 | 17.94 / 19.64 | 2.56 |
| Qwen AR | B1 | gsm8k | 64 | 1344.8 | 1027.6 / 1032.6 | 31.89 / 31.99 | 2.35 |
| Qwen MTP | B0 | chat | 32 | 1875.5 | 181.3 / 184.6 | 13.12 / 14.45 | 3.00 |
| Qwen MTP | B0 | chat | 64 | 1873.1 | 281.2 / 286.2 | 29.27 / 30.22 | 3.18 |
| Qwen MTP | B0 | gsm8k | 32 | 1309.8 | 590.0 / 592.8 | 14.20 / 16.33 | 2.50 |
| Qwen MTP | B0 | gsm8k | 64 | 1311.2 | 1058.1 / 1062.9 | 32.50 / 32.73 | 2.27 |


## Alpha limits

The alpha.2 snapshot above validates the listed TP4 and TP2 configurations
on 96 GB RTX PRO 6000 Blackwell Max-Q GPUs.
Eight physical GPUs, other GPU SKUs, multi-node serving and MPS are not
qualified. Qwen TP2 additionally needs about 51.2 GB of pinned host memory for
n-grams. Package memory declarations are admission floors, not recommended
device sizes; read the startup KV capacity line, since capacity depends on
dtype, sequence count and speculation. See [MiMo topology](topology-mimo.md)
and [Qwen topology](topology-qwen.md).

Use the declared profile default of 32 sequences. MiMo and Qwen TP4 AR and
MiMo TP4 DFlash are qualified through 64 with `--max-sequences 64`. Larger package limits do not qualify every workload. Full production qualification, end-to-end FP8 KV
qualification remain outstanding. KV preemption swap passed a bounded TP4
smoke on 2026-10-12: MiMo compared 36 of 36 C1 requests and Qwen 28 of 36,
all bit-identical in plain and prefix-pressure scenarios with a 16 GiB host
arena. Qwen's other requests ended before the first forced swap. Full
configuration qualification remains part of the full correctness suite;
see [offload scope](serving-kv-pool.md#formats-caching-and-offload).
Alpha.2 media checks passed for Qwen text/image, MiMo TP4 text/image/video
and MiMo TP2 text/image. MiMo serves audio, but audio is not qualified
in-tree: the repository holds no audio reference input, so test audio with
your own inputs. The tested MiMo TP2 package correctly refused omitted
audio and video inputs; see [modality selection](#served-modalities-and-memory).
Speculation gains depend on the workload.
A request served alone is reproducible bit for bit, with or without
speculation and including across preemption and restore, when it is greedy
or repeats the same explicit `seed` and sampling parameters on the same
configuration; a sampled request without a seed draws a fresh one
([request reference](serving-request-surface.md#request-parameters-and-defaults)).
Under concurrent load, outputs can differ from run to run, with or without
speculation: the number of requests sharing a decode step selects its batch
bucket, the bucket can change the order of F32 summation, and under
speculation the verify bucket also follows the other requests' draft
lengths. That shifts logprobs and can change a greedy token. Repeated runs of Qwen3.8 and MiMo V2.6 at 4 to 32
concurrent requests differed in some requests, while single requests matched
bit for bit; GLM 5.3 Flash matched in every repeated eight-request smoke run
and in its 16-request `preemption_exact` pairs, which is as far as GLM was
measured. This is documented behavior within the accuracy gates for batched
decode, which bound error against exact, not equality between buckets ([batch shape](numerics.md#batch-shape),
[request reference](serving-request-surface.md)).
The 2026-10-14 requalification's `preemption_exact` fell short of bit
identity, as the 2026-10-13 full validation run had: Qwen DCP 2
speculative was 35 of 36 requests token-identical (25 of 36 bit-identical)
against 35 of 36 (28 of 36) then, and MiMo's default profile 22 of 36
token-identical (0 of 36 bit-identical) against 21 of 36 (0 of 36). The
check runs 16 requests at a time. The same pairs at one request at a time
(2026-10-15, on the installs that recorded the rows) were bit-identical for
all 36 requests over 56 to 70 preemptions each, for Qwen's default and DCP 2
profiles and MiMo's default profile, so preemption and restore are exact and
the rows come from batch composition. Against the same requests served one
at a time, the 16-request runs kept 13 to 22 of 36 requests token-identical
over 96 tokens. On the shared history, the logprobs of likely tokens (above
-2 nats) moved by a median of 0.001 to 0.004 nats, a 99th percentile of 0.16
to 0.22 and at most 0.54. The first differing token followed a top-2 gap
below 0.3 nats in 52 of 54 cases and at most 0.54. The maintainers classified both
rows as this behavior, not as code regressions.

A profile name or compatible checkpoint family is not a qualification
identity: a new checkpoint needs accuracy evidence for its actual tensors,
package, host, topology, settings and workload. `bash ci/quick.sh --report`
lists the selected checks and remaining evidence gaps without serving.
[Report problems](developer-preview.md#report-a-problem) with the details
listed there.

## Served modalities and memory

`serve --model DIR --modalities auto|all|LIST` selects the package's
modalities. `auto`, the default, serves text and image (those of them the
checkpoint provides); `all` serves every modality the checkpoint provides; a
list must contain text. Omitted encoders use no device image or scratch
memory, and KV takes the freed space. The same bundle ID serves every set and
rank count, with distinct package cache identities. `--max-sequences` is
still bounded by the package maximum.

MiMo shares its vision tower between image and video. Keeping video reserves
96 MiB for the vision payload region, against 24 MiB for image alone. At TP2
the default text,image package holds 495,989 KV tokens against 388,715 with
every modality (+28%): audio costs about 21%, video about 1%.

The tables below are contract estimates (package limits, not GPU
measurements) for 94 GiB available per device, BF16 global KV and the default
0.9 KV fraction, at TP2 and TP4. Each cell
is **maximum sequences / KV tokens at that maximum**. DFlash and MTP include
their images and working memory. Changing the sequence bucket changes ring
memory, so compare live KV at the same served count.

| MiMo package | Served modalities | TP2 | TP4 |
|---|---|---:|---:|
| AR | text | 64 / 612,649 | 256 / 7,635,592 |
| AR | text,image (auto) | 32 / 495,989 | 256 / 7,271,855 |
| AR | text,video | 32 / 490,091 | 256 / 7,260,058 |
| AR | text,image,video | 32 / 490,091 | 256 / 7,260,058 |
| AR | text,audio | 32 / 512,748 | 256 / 7,305,374 |
| AR | text,image,audio | 32 / 394,122 | 256 / 7,068,121 |
| AR | text,video,audio | 32 / 388,715 | 256 / 7,057,308 |
| AR | text,image,video,audio (all) | 32 / 388,715 | 256 / 7,057,308 |
| MTP | text | 32 / 626,137 | 32 / 8,044,262 |
| MTP | text,image (auto) | 32 / 444,269 | 32 / 7,680,525 |
| MTP | text,video | 32 / 438,370 | 32 / 7,668,728 |
| MTP | text,image,video | 32 / 438,370 | 32 / 7,668,728 |
| MTP | text,audio | 32 / 461,028 | 32 / 7,714,044 |
| MTP | text,image,audio | 32 / 342,402 | 32 / 7,476,791 |
| MTP | text,video,audio | 32 / 336,995 | 32 / 7,465,978 |
| MTP | text,image,video,audio (all) | 32 / 336,995 | 32 / 7,465,978 |
| DFlash | text | 32 / 432,727 | 256 / 6,647,283 |
| DFlash | text,image (auto) | 16 / 335,891 | 256 / 6,283,546 |
| DFlash | text,video | 16 / 329,993 | 256 / 6,271,749 |
| DFlash | text,image,video | 16 / 329,993 | 256 / 6,271,749 |
| DFlash | text,audio | 32 / 267,617 | 256 / 6,317,064 |
| DFlash | text,image,audio | 16 / 234,024 | 256 / 6,079,812 |
| DFlash | text,video,audio | 16 / 228,617 | 256 / 6,068,999 |
| DFlash | text,image,video,audio (all) | 16 / 228,617 | 256 / 6,068,999 |

Qwen3.8 provides text and image. Its package maxima are:

| Served modalities | TP2 AR / MTP | TP4 AR / MTP |
|---|---:|---:|
| text | 64 / 64 | 256 / 128 |
| text,image (auto, all) | 32 / 32 | 256 / 128 |

### TP4 default capacity

The 2026-10-04 qualification measured the following live BF16 KV capacities with text,image.
The AR arms reused the corresponding speculative-capable package with
speculation disabled. These are measured tokens at the stated serving cap,
not new package maxima or reservations for every request. The C64 pools
allow roughly 122K/116K tokens per resident MiMo AR/DFlash sequence and
42K tokens per Qwen AR sequence on average. Queueing and preemption still
handle requests that exceed those averages.

| Profile | Default sequences | Qualified through | Measured KV tokens at default | Package maximum |
|---|---:|---:|---:|---:|
| MiMo TP4 AR | 64 | 64 | 7,805,888 | 256 |
| MiMo TP4 DFlash | 64 | 64 | 7,423,168 | 256 |
| MiMo TP4 MTP | 32 | pending accepted arm | not measured on 2026-10-04 | 32 |
| Qwen TP4 AR | 64 | 64 | 2,679,104 | 256 |
| Qwen TP4 MTP | 32 | 32 | 2,516,992 | 128 |

Qwen AR capacity was remeasured on B1 and remains 2,679,104 tokens.
Qwen MTP capacity remains B0 evidence, before the checkpoint-index change;
current-build startup lines are authoritative.
Qwen MTP held 2,437,824 KV tokens when tested at C64. Per-device persistent
working sets and graph capture grow with the serving bucket, so the C32
capacity cannot be used as C64 headroom. Read the startup capacity line for
the actual package, modalities, free memory and speculation settings.

Live measurements at TP2 AR with 32 served sequences, which use the host's
actual free memory and so differ from the estimates:

- Qwen, 2026-10-02: text only held 1,586,176 KV tokens against 1,396,160 with
  the image encoder (+13.6%), with bit-identical C1 greedy output.
- MiMo MOPD, 2026-10-02: text only held 746,752 KV tokens and text,image
  565,824, against 458,368 with every modality (+62.9% and +23.4%), with
  bit-identical C1 greedy output. Requests for an omitted modality receive
  HTTP 400 naming `--modalities`.
