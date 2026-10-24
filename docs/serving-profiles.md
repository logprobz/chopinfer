# Serving profiles

Each model owns `profiles.json`; use `chopinfer profiles list FILE` or
`chopinfer profiles show FILE NAME` for settings and recorded status. Both
validate profiles without a GPU. The [README](../README.md) summarizes
supported models and hardware. Package manifests remain authoritative for
topology, limits and capabilities.

## Selecting a profile

`serve --model DIR` selects a profile through the installed model helper and
refuses `--profile`. Explicit packages accept a packaged name or `FILE#NAME`:

```sh
chopinfer serve --bundle "$PACKAGE" --checkpoint "$CHECKPOINT" --profile tp4-ar
chopinfer profiles list models/qwen38/profiles.json
```

An explicit option overrides a profile default. `--speculative-tokens 0`
disables speculation but retains that profile's sequence cap. A profile
cannot grant qualification to an override. `max_sequences` is the serving
cap; `qualified_max_sequences` is the largest concurrency its evidence
covers, or null for experimental profiles. Startup warns above that count.
The shared default is 32; MiMo/Qwen TP4 AR and MiMo DFlash have evidence
through 64. Package capacity alone is not serving qualification.

## Speculative token capacity and graph budget

Two serve options size a speculative verify step. Each is also a profile
field under `speculation`, so a profile sets the default and a flag
overrides it:

- `--speculative-rows N` (`speculation.rows`), the token capacity: the most
  verify rows one step may carry, one row per decoding item plus every
  draft. A package that declares `chop.speculative-requested-rows.v1` serves
  its default rows unless asked for more, up to its manifest maximum, and
  prepares its buffers for the rows it is asked for. DeepSeek serves 32 by
  default and up to 192 (32 items at its five-draft block). Startup refuses
  rows at or below `--speculative-tokens` or above the maximum.
- `--speculation-graph-rows N` (`speculation.graph_rows`), the graph budget:
  the widest verify step captured in a CUDA graph. It reaches a package that
  declares the `verify_graph_rows` model option (DeepSeek, default 32);
  wider steps run without a graph and pay their launches. Startup warms
  every width up to the package's default rows and, above them, only the
  balanced plan's listed widths (`--speculation-widths`) up to the budget.
  Startup refuses a budget above the package's verify maximum and a package
  that does not declare the option.

`--list-model-options` checks both against the package without opening a
device. A wider step pays only where it costs less than the tokens its
drafts add, so measure the step cost of each width before raising either.
Each graph captured above the default rows also takes memory from the
graph memory reserve below.

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
is per profile. Server logs from 2026-10-05 to 2026-10-22 measured about
8 MiB per GPU for MiMo TP4 AR, 56 MiB for Qwen TP4 MTP, 68 MiB for MiMo
DFlash, and at most 140 MiB live over a GLM soak; those profiles keep the
default until the post-warmup line below measures them. On 2026-10-23
DeepSeek TP4 DSpark at 32 sequences captured 384 graphs per GPU holding
902 MiB, which left 137 MiB of a 1 GiB reserve, so the DeepSeek profiles
reserve 1280 MiB (the graphs, the 256 MiB warning headroom and about 10 %
for growth). A capture that finds no memory fails the generation worker, so
startup logs each rank's free memory after warmup and warns below 256 MiB
([observability](operations-observability.md#chopinfer-metrics)); raise the
reserve or lower `--max-sequences` when a rank warns.

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

The pinned checkpoint is MiMo V2.6 Flash MOPD. Historical RL checkpoint runs
do not qualify its tensors. `tp4-mtp` has historical serving evidence only,
workload-specific throughput regressions and no accepted paired arm; it
remains experimental. `tp2-ar` also remains experimental.

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

Profiles declare fixed or time-between-token prefill budgets; explicit
options override them. Mixed prefill/decode scheduling is opt-in and requires
the package's declared support. Compare it with the same model, workload,
concurrency and precision. Neither a scheduling flag nor spare capacity
establishes accuracy or performance qualification.
