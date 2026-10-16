# glm5next

GLM 5.3 Flash (`zai-org/GLM-5.3-Flash`, Hugging Face `model_type`
`glm5_next`), pinned at revision `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`.
`checkpoint-files.json` holds that revision's file sizes and SHA-256 digests.

`contract/` (crate `model-glm5next-contract`) owns the compiled geometries,
config admission, per-rank kernel profiles, the tensor catalog, the TP4
weight plan and the `glm5next.v1` serving contract (see below).
`compiler/` (crate `model-glm5next`) holds the preparation provider, which
routes a checkpoint by its `config.json` and emits its package (helper
`chopinfer-model-glm5next`) for the FP8 checkpoint and both NVFP4 exports,
with the encoder image when the checkpoint stores the vision tower.
`bundle/` (crate `bundle-glm5next`) holds the forward program, its launch
records and the instance, whose whole program lowers for every admitted
layout and serves on four GPUs. [docs/model-glm5next.md](../../docs/model-glm5next.md)
summarizes what is served and qualified. `model.toml` declares `[release] enabled = true`, so
the released-model facts the inventory, profile and qualification tests
check apply to it. The operator adopted the final gate's results
(GATES4, 2026-10-11), whose two soak failures the declared soak warm-up now
passes (they were waived then), so `qual.json` and its
configurations are no longer `mechanics_only`, except the `taps`
configuration (see "Qualification checks"). See
[adding a model](../../docs/adding-a-model.md).

Release status: released. The GATES4 results are adopted; its two soak
failures (the default configuration's and the local-inference-lab NVFP4
export's) pass the soak's declared warm-up and cache allowance. The full
requalification through the release preview (stage 5,
2026-10-14 to 2026-10-15 at 3d5494e7b) qualified it, with accepted
rows in three configurations
([release qualification](../../docs/model-glm5next.md#release-qualification-stage-5)),
and `model.toml` enables the release. It declares the release
targets (the helper and the bundle library; the release plan ships the
`kernels.lock.d/` family files), the `[contributor]` table and the
profile target, and the deploy artifacts exist (systemd unit, container
default, the deploy doc's readiness line), and `qual.json` has its
`preview` section. `qual.json`, the DCP declarations and the NVFP4
declarations declare the `paired_teacher` check with the operator's GLM
margins (2026-10-11), pinned to accepted arms captured from the GATES4
install at 8c75e90bd; each passed its accepted-versus-accepted dry run
(2026-10-12, [paired accuracy](../../docs/paired-accuracy.md#glm-53-flash)).
`tp4-ar` is declared without an accepted arm (no GLM gate run serves it);
the AR-versus-MTP determinism, spec_verify and verify_exact pairs tie it to
`tp4-mtp`.
The preview runs the quick cohort on its speculative arm alone
(`preview.quick_arms`), since every live check here requires a drafting
server.

## Checkpoint geometry

`geometries/flash.json` is the execution part of the pinned `config.json`
(top-level token IDs, `text_config` and `vision_config`, without training
metadata). `supported.json` admits `glm5_next`, `glm5_next_text` and
`glm5_next_vision`, constrains execution semantics, and admits the `fp8`
recipe class with 128x128 block scales and the `modelopt` class (see NVFP4
checkpoint below). `contract/build.rs` generates the
geometry and its TP4 kernel profiles from these files; extra directories may
be listed in `CHOPINFER_GLM5NEXT_GEOMETRY_DIRS`. The bundle and its cubins are
built for flash at TP4 (`Shape::TP4`), so the contract admits and plans an
added geometry but the bundle cannot serve it yet
([geometries](../../docs/model-glm5next.md#geometries)).

Every geometry field is compiled. A config that differs in one is refused
before any weight is read, naming the field. `text_config.index_topk` is the
one runtime field: it defaults to the checkpoint's value and may be set
through `--model-option` (`chop.model-options.v1`) to a multiple of the pool
size `index_kpool` (4) from 4 up to the compiled maximum of 2048. The same
structural rules (per-layer lists, whole pools) refuse a malformed geometry
at build time and a malformed config at admission.

The catalog generates 76,108 tensors (328.3 GB) by layer type and count: 34
KDA layers, 11 latent attention layers, 3 dense and 42 MoE MLPs, mHC mixing
per sublayer, MTP layer 45 and the vision tower. At TP4 each GPU holds
77.65 GiB: routed experts 70.89, other main-layer tensors 3.30, MTP 1.81,
vision 1.05, embedding and head 0.59.

Tests, from the workspace root:

```sh
cargo test -p model-glm5next-contract -p model-glm5next -p bundle-glm5next --locked --offline
```

With `CHOPINFER_GLM5NEXT_CHECKPOINT` set to the pinned checkpoint, the
contract's `checkpoint_headers` test also checks every tensor name, dtype and
shape against the safetensors headers; it reads headers only. Without it,
`contract/fixtures/tensor-families.json` (read from those headers) checks the
catalog's families.

## NVFP4 checkpoint

`local-inference-lab/GLM-5.3-Flash-NVFP4` at revision
`175ae8ce3b5af842b0d0140dbeb43e9cfc557c49` is a second checkpoint of the
flash geometry. `checkpoint-files-nvfp4.json` holds its 44 indexed shards,
config, `hf_quant_config.json`, index, chat template and tokenizer files.
Tests read it from `CHOPINFER_GLM5NEXT_NVFP4_CHECKPOINT`. Its tokenizer, chat
template, processor and generation config are byte-identical to the FP8
checkpoint's.

Admission reads its ModelOpt `quantization_config` through the shared
importer, projection by projection, and selects the `nvfp4-routed` layout
(`contract/src/quantization.rs`); a recipe no layout stores is refused,
naming the module:

- routed experts (layers 3-44): NVFP4, U8 E2M1 pairs with an E4M3
  `weight_scale` per 16 K values and an F32 `weight_scale_2` per expert
  projection; the static `input_scale` is present and never read;
- MTP routed experts (layer 45): MXFP8, E4M3 codes with a U8 E8M0
  `weight_scale` per 32 K values;
- everything else BF16, including latent attention and the dense and
  shared MLPs, which are block FP8 in the base checkpoint.

Every projection runs BF16 activations (A16). `--activation-policy auto`
and `bf16` run the NVFP4 and MXFP8 weights under BF16 activations (W4A16
for NVFP4); `checkpoint` is refused, since the export declares static
NVFP4 input activations. The base checkpoint's `checkpoint` is refused
likewise (it declares dynamic FP8 activations). The compiler checks the
policy when a package or profile is requested.

The catalog for this layout has 148,498 tensors (198.0 GB). At TP4 each GPU
holds 47.38 GiB instead of 77.65: routed experts 39.87, other main-layer
tensors 3.98, MTP 1.90, vision 1.05, embedding and head 0.59; the text rank
image is 46.33 GiB. Under the capacity assumptions below, 32 slots with FP8
KV and three draft tokens plan 4.78M KV tokens at DCP 1 and 18.72M at DCP 4
for a text package, 4.33M and 16.92M with images (block FP8: 0.67M and
2.25M, 0.22M and 0.45M with images), and 64 slots with MTP fit (4.03M at
DCP 1).
`contract/tests/nvfp4_checkpoint.rs` checks the catalog and the projection
binding against the export's headers when the variable is set, and
`contract/fixtures/nvfp4-tensor-families.json` without it.

The HF reference of the export (models/glm5next/reference.py, same adapter
and teacher workload as the FP8 reference, 2026-10-06) differs from the FP8
teacher reference's F32 gate variant by the quantization itself. Scored
like a server over the 3,781 teacher positions:

| Set | NVFP4 F32: mean, max, top-one | NVFP4 BF16: mean, max, top-one | FP8 BF16 (the gate's spread) |
|---|---|---|---|
| pooled | 0.148, 9.07, 5.9% | 0.149, 9.02, 6.2% | 0.063, 7.96, 2.9% |
| chat (560) | 0.347, 5.28, 11.1% | 0.338, 5.12, 11.2% | 0.158, 5.27, 7.0% |
| code (695) | 0.115, 7.43, 3.0% | 0.114, 7.26, 3.5% | 0.044, 3.06, 1.4% |
| math (701) | 0.084, 6.54, 3.9% | 0.082, 6.80, 3.6% | 0.031, 1.16, 1.7% |
| multilingual (796) | 0.150, 7.95, 6.9% | 0.162, 7.88, 7.2% | 0.062, 1.67, 2.9% |
| prose (376) | 0.187, 9.07, 8.8% | 0.197, 9.02, 9.8% | 0.083, 7.96, 2.9% |
| prose-frozen (141) | 0.262, 3.02, 17.0% | 0.231, 3.01, 17.7% | 0.115, 1.80, 9.9% |
| long-context (512) | 0.0014, 0.15, 0.2% | 0.0024, 0.46, 0.4% | 0.0005, 0.06, 0.2% |

Mean and max are |dlogprob| in nats; top-one is the share of positions whose
top-one token differs. The export's own BF16 run sits 0.054 (pooled, top-one
2.8%) from its F32 run, close to the FP8 checkpoint's 0.063. Its F32 run
would fail the FP8 teacher gate (pooled mean 0.148 against 0.10; the chat,
multilingual, prose and prose-frozen categories fail), so NVFP4 serving
needs its own pinned reference rather than the FP8 one.

The `nvfp4` configuration of `qual.json` qualifies the export (profile
`tp4-mtp-nvfp4`) against that reference, read as `--model glm5next.nvfp4`,
with its own entries in `qual-references.lock.json`. It states the NVFP4
checkpoint pin (identity
`83b4297362ba87927e462afaff42c406e8a2d8a869fb36e14fdc65badb5201e6`, the one
the engine's direct preparation reported for the export's directory), the
teacher reference
`fixtures/glm5next/nvfp4-teacher-reference.json` (the F32 gate variant and
the BF16 spread variant above, over the FP8 teacher workload), and the checks
that hold for any checkpoint (api, structured, soak without a media mix,
spec_verify, verify_exact, prefix_exact, preemption_exact, determinism and
media_preprocessing), plus a paired GSM8K run against the FP8 arm that
`qual.json`'s gate run records (`evals_gsm8k`, arm `fp8`, the whole
1,319-item test split): `evals_candidate` answers the same items on the
export, and `evals_compare` compares the two arms by their receipts and
fails if the export is significantly worse (McNemar, p < 0.05). The FP8
arm's receipt (input `fp8_gsm8k`) must be of that run (input
`fp8_gsm8k_run`, its run ID), pin the GSM8K workload digest and report
`tp4-mtp` at TP4, FP8 KV and DCP 1 and the FP8 checkpoint pin. Every other check that reads a serving profile requires
`tp4-mtp-nvfp4` at TP4, FP8 KV and DCP 1. The reference's own spread sets
the teacher limits: pooled mean 0.10, the chat category 0.15 (its BF16
spread 0.130), every other category 0.15, top-one 8% decisive and near tie,
10% in all. Long context, image probes and encoder parity need NVFP4
references not yet run and are not declared, so the declaration claims no
media. An f64 run is not needed for the limits (the FP8 reference's f64
spread is 0.0005 against BF16's 0.063); it would show how far the F32
dequantization rounding moves the gate variant, in two GPU jobs of about
12 and 10 minutes (the FP8 f64 teacher's 525 and 414 s, scaled by the
export's slower f64 probe, 175 s against 127 s).

## NVIDIA NVFP4 checkpoint

`nvidia/GLM-5.3-Flash-NVFP4` at revision
`da920bb0b9f4a06727223a349e55468e38352348` is a third checkpoint of the
flash geometry (`checkpoint-files-nvidia-nvfp4.json`: 33 shards, config,
`hf_quant_config.json`, index, chat template and tokenizer files; tests
read it from `CHOPINFER_GLM5NEXT_NVIDIA_NVFP4_CHECKPOINT`). Its config
declares NVFP4 for every `Linear` with exclusions, which admits the
`nvfp4-dense-routed` layout:

- dense MLPs (layers 0-2) and routed experts (layers 3-44): NVFP4, as
  above; the down projection's rank slice keeps whole groups of 16 K
  values, and the global and activation scales are scalars every rank
  holds;
- MTP routed experts: BF16, a quarter of each per rank like the others;
- latent attention, shared experts, the router, embedding and head: BF16;
- side tensors in other dtypes than the base: F32 KDA convolutions, stored
  as read, and BF16 mHC base and scale vectors, widened to F32 in the rank
  image.

Its KV cache hint (`kv_cache_scheme`, FP8) is ignored; the latent KV format
is the serving `--kv-cache-dtype`. `--activation-policy checkpoint` is
refused (static NVFP4 inputs); `auto` and `bf16` run W4A16. The catalog has
147,661 tensors (204.4 GB). At TP4 each GPU holds 48.87 GiB (the text rank
image 47.82 GiB); 32 slots with FP8 KV and three draft tokens plan 4.65M KV
tokens at DCP 1 and 18.18M at DCP 4 for a text package, 4.20M and 16.38M
with images. Its tokenizer, processor config and
tokenizer config match the FP8 checkpoint's; its chat template and
generation config do not (an older template revision), so its references
reuse the FP8 checkpoint's teacher workload.

Admitting it changed code beyond declarations: the layout gained a dense
class and the two side tensor dtypes, the contract normalizes HF's
single-algorithm ModelOpt form, which the shared importer refuses or
misreads (`contract/src/quantization.rs`: the `Linear` class group must
agree with the top-level algorithm, its group size moves to the top level,
and the static FP8 KV hint is dropped for this layout only), the
weight plan gained the NVFP4 K granule, and the rank image widens BF16
side vectors. The projection declaration only admits the new side dtypes.

Its HF reference (models/glm5next/reference.py, F32 with dequant rounded
and BF16, over the FP8 teacher workload, 2026-10-07) scored like a server
over the 3,781 teacher positions, against the FP8 teacher reference's F32
gate variant and against the local-inference-lab export's runs:

| Set | NVIDIA F32 vs FP8 F32: mean, max, top-one | NVIDIA BF16 vs FP8 F32 | NVIDIA F32 vs local-inference-lab F32 |
|---|---|---|---|
| pooled | 0.147, 9.29, 5.4% | 0.149, 9.06, 5.8% | 0.130, 8.44, 5.1% |
| chat (560) | 0.375, 6.86, 11.1% | 0.405, 6.55, 12.0% | 0.363, 8.04, 10.7% |
| code (695) | 0.101, 7.25, 3.3% | 0.085, 3.41, 3.2% | 0.083, 3.71, 3.2% |
| math (701) | 0.084, 5.20, 3.9% | 0.080, 2.36, 3.9% | 0.069, 3.54, 3.0% |
| multilingual (796) | 0.151, 8.40, 6.0% | 0.149, 6.58, 6.2% | 0.134, 8.44, 6.2% |
| prose (376) | 0.169, 9.29, 8.0% | 0.170, 9.06, 9.0% | 0.127, 1.96, 5.3% |
| prose-frozen (141) | 0.232, 2.78, 9.9% | 0.256, 2.74, 12.1% | 0.195, 1.49, 14.2% |
| long-context (512) | 0.0016, 0.16, 0.2% | 0.0028, 0.39, 0.4% | 0.0012, 0.21, 0.0% |

The FP8 teacher gate refuses its F32 run as it refuses the
local-inference-lab export's: pooled mean 0.147 against 0.10 (decisive
top-one 2.7%, near tie 2.8%), with the chat, multilingual, prose and
prose-frozen categories failing; its BF16 run fails chat, prose and
prose-frozen. The two NVFP4 exports differ from each other (pooled 0.130,
top-one 5.1%) nearly as much as each differs from FP8 (0.147 and 0.148),
so neither export's reference stands in for the other's. Its own BF16 run
sits 0.065 (pooled, top-one 2.3%) from its F32 run. The short f64 probe
(the 512-token probe, 256 scored positions, dequant exact) completed with
finite scores; against the FP8 and local-inference-lab f64 probes it
differs by a mean 0.121 and 0.101 (top-one 6.6% and 5.9%), the same
quantization gap.

The `nvfp4-nvidia` configuration qualifies it (profile
`tp4-mtp-nvfp4-nvidia`) against that reference,
`fixtures/glm5next/nvfp4-nvidia-teacher-reference.json` (the F32 gate variant
and the BF16 spread variant), as `nvfp4` does the local-inference-lab
export, with its own lock entries. Its own spread sets the teacher
limits: pooled mean 0.10, the chat category 0.20 (its BF16 spread 0.160),
every other category 0.15, top-one 8% decisive and near tie, 10% in all.

## Serving contract and program

The package `glm5next.v1` serves topology `glm5next-tp4` at
context-parallel sizes 1, 2 and 4 (`contract/src/contract.rs`). Per rank,
KDA and latent attention run 16 heads, the latent, query-down, indexer and
router projections are replicated, each rank holds a quarter of every
routed expert's intermediate, the dense and shared MLPs split by column,
and the embedding and head by
vocabulary row (`views.rs` binds every text tensor to a projection ID or
the engine that reads it). A 64-token page holds, per sparse layer, the
latent rows (FP8 E4M3 with per-128 F32 scales by default, or BF16) and the
pooled indexer keys; the MTP layer's pages follow the main ones in the same
format. Each request slot keeps the KDA convolution history and state and
the indexer's incomplete-pool tail, with one checkpoint per verify row
(`state.rs`), and sixteen anchor items' spares, which keep a prompt-ending
item's state at the last multiple of 64 inside its rows for a prefix
capture there. The prefill granule is 64, the KDA chunk. `fit.rs` refuses
TP2 (the weights do not fit) and TP8 (out of scope) by name.

`bundle/src/program.rs` lists the 45 layers' stages in config order and
`launch.rs` turns each into a launch record carrying its kernel-core query;
`lower.rs` and `lower/` lower every stage to the shared engines' launches.
The fake-host journals pin the program per latent KV format and phase, and
the drafter's per pass (`journal_tests.rs`).

The latent attention partitions a row by its kind, never by the step's
bucket: a target prompt row runs one split, a generated row 64 fixed
splits, and every drafter row one split (`lower.rs`, `attention_phase`).
That makes the latent attention of a target row independent of the other
rows of its step; it is not a whole-row guarantee (the drafter's catch-up
pass still changes GEMM layout with its rows; see
[batch invariance](../../docs/model-glm5next.md#batch-invariance)). A step's
graphs are lowered only for ranks missing them, and an eager prefill whose
exact shape ran before replays the programs its second run captured
(`bundle/src/instance/prefill_graphs.rs`; a request's last prompt chunk of
at most 64 rows captures at its first run and stays captured; the internal
diagnostic switch `CHOPINFER_GLM5NEXT_LAST_CHUNK_GRAPHS=1` captures all 64
lengths at create instead, except on a context-parallel instance), except a
prefill that injects
encoder rows and any prefill under `CHOPINFER_GLM5NEXT_SYNC_STAGES` or the
text taps, which stay eager (`instance/execute.rs`, `prefill_key`).

Decode, verify and the drafter's graphs capture each MoE sublayer's shared
expert, its gate/up and its down projection, on a second stream of the
rank, forked after the sublayer's norm and joined before the routed sum, so
it runs beside the router, the routing and the routed experts
(`bundle/src/shared_overlap.rs`); eager runs keep one stream. In these
graphs the shared down projection adds zeros (`ZeroRows`) into the MLP
output, over a projection scratch of its own, and the routed sum adds that
output into the partial last (`chained::RoutedSum::add_entry`): the same
add of the same two values as a prefill's residual epilogue, which adds
the routed sum to the projection, so the bits are the prefill's order's
(kernel-lab `shared_down_beside_the_routed_sum_keeps_the_partial_poisoned`
compares both at every graph bucket, rank and live count). The overlap
itself keeps every launch and binding of the one-stream program; the
fake-host tests compare both programs launch for launch, replays included,
and check the order by vector clocks (`fake_host_tests/shared_overlap.rs`).
PERF-B12 job 1 served the gate/up overlap bitwise equal to one stream: AR
ms/token C1 16.21 to 15.47, C8 25.95 to 24.95, C16 33.6 to 32.8; MTP C4
11.0 to 10.6, C8 15.1 to 14.4 (C1 flat).
`CHOPINFER_GLM5NEXT_SHARED_OVERLAP=0` at serve start is the off-ramp; a
positive number overlaps only graph buckets of at most that many rows
(decode items, verify tokens).

An eager run (a prefill, or a phase too wide for its graph bucket) queues
every rank's launches up to its next collective, rank after rank, then the
next such segment: a rank's stream holds a bounded number of pending
launches, so queueing one rank's whole program first leaves the host
blocked on that rank's stream while it spins at its first collective. For
bring-up, `CHOPINFER_GLM5NEXT_SYNC_STAGES=1` at serve start makes every
rank synchronize after each eager segment and graph replay and log the
stages and rank before each wait, so a hang's last log line names its
stage; it serializes the ranks and is for diagnosis only.

All-reduces of `kernels_sm120::backend::BF16_TURNS_ROWS` (1,024) rows and
more take the phased two-shot whose writers take turns at each destination
(`kernels_sm120::comm::phased_turns_bf16`): the two-shot's transfers and
rank-order sums, so the same bits, but a rank starts a phase into a peer
only after that peer's previous writer finished. Prefill ranks reach their
all-reduces at different times, and the two-shot then let two or three
ranks write into one GPU at once. Kernel-lab (PERF-B22 job 4, four ranks,
8,192 rows) times the two-shot at 3.3 to 3.9 ms a call beyond an injected
skew of 0.2 to 1.1 ms and the turns entry at 2.4 ms with or without it;
the served AR 32K prefill went from 4779 to 4401 ms (PERF-B22 job 7,
bitwise).

The phased all-reduce's entries share one state per rank whose flags carry
the call's epoch and stay set, so a flag tag would repeat 2^31 calls later.
Every `engines::PHASED_RESET_RUNS` runs (2^20, at most 1024 phased calls
each, which lowering enforces; prefill runs count) the instance waits for every
rank's stream, zeroes every rank's phased flags and epoch, and waits again,
before the run's launches (`instance/execute.rs`, `quiesce_phased`).
`CHOPINFER_GLM5NEXT_PHASED_RESET_RUNS=N` at serve start lowers the bound to
N runs, a diagnostic that exercises the reset in a short run.

Prefills run whole. A row-parts prefill with its all-reduces on a comm
stream of each rank (the `CHOPINFER_GLM5NEXT_PREFILL_OVERLAP`,
`PREFILL_PARTS` and `OVERLAP_COMM_CTAS` switches) was removed on 2026-10-12.
It was bitwise equal to whole prefills but off by default and slower at every
measured length: with the tiled prefill mixer (PERF-B15 job 6, FP8 TP4) 32K
took 6643-6717 ms whole against 7066-7068 ms overlapped, and with BF16
all-reduces (PERF-B22 jobs 8 and 9) the overlap cost 7 to 9% at 32K and 14 to
30% at 3K. Its code is in the history before the removal commit.

## Projection declaration

`contract/projections.json` declares every checkpoint tensor on the shared
projection API (`crates/projection-spec`), validated at TP4 only. Projections
run A16 (BF16 activations): the FP8 128x128 latent attention (q_a, q_b, kv_a,
o), dense MLP and shared expert projections apply each block scale to its
128-K MMA partial; KDA, indexer, router (F32 output), MTP `eh_proj`, head and
embedding are BF16. Each rank holds a quarter of every routed expert's
intermediate, KDA and latent attention split 16 heads per rank, the
embedding and head split by vocabulary row, and the rest is replicated.
Layer 45 is its own MTP stems.

Side tensors bind to the engine that reads them: F32 `A_log`, `dt_bias`,
mHC base and scale and the router bias; BF16 convs, norms, `hc_*_fn`, the
indexer pooling gate and position bias, and `kv_b_proj` (the latent
attention engine's absorbed key and value halves). Vision tensors are owned
by the encoder declaration. The gate-up projections carry the SwiGLU clamp
(`swiglu_limit`) as a registered `glm5next/clamped-swiglu-*` implementation,
because `Epilogues` has no clamp. The NVFP4 routed and MXFP8 MTP expert
formats of the local-inference-lab export and the BF16 projections it keeps
are declared beside FP8; `contract/tests/projections.rs` binds the catalog,
a header-shaped stand-in for that export and, with
`CHOPINFER_GLM5NEXT_CHECKPOINT` set, the checkpoint's own headers.

## Vision encoder

`contract/encoders/glm5next-vision.json` declares the tower
(`contract/tests/encoder_declarations.rs` pins it): the 1176-wide patches
zero-padded to 1216 columns, 24 pre-norm blocks with per-head q/k RMSNorm,
axial rotary and clamped SwiGLU MLPs, the 2 x 2 merge into a linear, and the
LayerNorm-GELU-clamped-SwiGLU projector, behind the
`image.zero-padded-token-budget.v1` recipe (8000 tokens). Its new row
entries (pad, q/k norm, clamped SwiGLU, GELU) are checked in kernel-lab's
`encoder_vision` family. Its GEMMs and attention run the shared encoder
pair GEMM and varlen pair attention engines. Pair attention restarts its
PV products from zero every 64 keys and reduces QK over the 64-wide head,
so none of its chains spans 128 K. The pair GEMMs chain the high and low
halves over all of K (152 to 1280 MMAs, the longest in the projector's
K 10240 down projection) and are exempt from the 128-K rule under the gemm
engine's decision of 2026-10-04 to keep persistent BF16 chains
(`crates/kernels-sm120/cuda/gemm/README.md`, "Decision, 2026-10-04").
Measured on 2026-10-11 over all nine served pair GEMM entries (eight N x K
shapes) and six probe images, they err by at most 5.9e-6 of the summed
absolute products, with 0.09 to 0.93% of outputs one BF16 step off,
inside that decision's measured range. The tower's accuracy is judged by
`encoder_parity` against the reference. `encoder_parity_run.py requests` writes one chat request per
image probe, as the multimodal check sends it with one output token, and
prints the `tools/qual/tap_binding.py capture` command for a diagnostic-taps
server; `encoder_parity_run.py assemble` then completes the captured run
directory with the exact and policy reference runs, their workload and
`parity.json`, the `encoder_parity_run` input of the `taps` configuration's check.
The two reference runs (exact; policy and reference-bf16) are linked into
one reference directory in the layout the scorer reads.

## Chat frontend

`contract/src/frontend.rs` (`Glm53Format`) renders the prompt: a Rust port
of the pinned `chat_template.jinja` (revision eb9eb208), one function per
template block, each citing the lines it ports. It writes the `[gMASK]<sop>`
prefix, the role tags, the reasoning effort (`low`, `high` or `max`, default
`max`, rendered as `Reasoning Effort: Low|High|Max`), the
`<|assistant|><think>` generation prompt, tool definitions, `tool_reference`
results, and history tool calls in the key-value grammar
(`<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`).
The bundle exports it with `frontend_format::export_chat_format_v1!`.
`compiler/src/frontend.json` declares the model-neutral sections, the
tokenizer's policy (pair-array merges with whole-piece bypass) and
`format.parser`: the key-value tool-call family, the `<think>` markers, and
the four role tokens that open a turn. `compiler/tests/frontend_goldens.rs`
checks the format byte for byte against prompts that transformers rendered
from the pinned template, and against the pinned tokenizer's ids when
`CHOPINFER_GLM5NEXT_CHECKPOINT` is set. `goldens.py` beside the fixtures
regenerates or checks them on CPU:

```sh
F=models/glm5next/compiler/fixtures/checkpoints/glm-5.3-flash
$CHOPINFER_REFRUN_PYTHON -B $F/goldens.py $CHOPINFER_GLM5NEXT_CHECKPOINT $F --gate fixtures/glm5next/frontend --check
```

With `--gate` it also writes the frontend_goldens gate's chat reference,
`fixtures/glm5next/frontend/chat-renders.json`: the text cases and the chat
teacher conversations (`reference_workloads.py` `CHATS` and
`CHATS_EXTENDED`), rendered the same way. The gate's tokenizer reference is
`tokenizer-golden.json` beside `goldens.py`. qual.json's `frontend_goldens`
sends every row to a served `/tokenize` and requires HF's token IDs; the
image cases stay in the Rust test, since a served `/tokenize` expands the
image pad and media_preprocessing gates that.

The HF differential corpus (`frontend_cases.py`, `frontend-divergences.json`
and the committed subset `fixtures/glm5next/frontend/corpus*.json`) runs
through `compiler/tests/frontend_corpus.rs` as raw requests: renders and
token ids, refusals, and round trips parsed at every byte split, with the
checkpoint's tokenizer. A history turn ends where the next role token
begins, so its round trips end at `<|observation|>` after tool calls and at
`<|user|>` or `<|endoftext|>` after an answer.
`python3 -m tools.refrun.frontend_corpus check --model glm5next` is the
coverage and divergence gate, and a template update follows
[updating the chat template](../../docs/adding-a-model.md#updating-the-chat-template).

Offered tools are listed as the template lists them: each function object as
the client sent it, keys in order, without `strict` and `defer_loading`, and
leaving out a deferred function. The declaration refuses what the template
ignores or remaps: other reasoning efforts (the template renders Max),
thinking switched off, developer messages, and a non-boolean
`clear_thinking`.

Images use `image.zero-padded-token-budget.v1` (patch 14, temporal patch 2,
merge 2, 16 to 8,000 tokens, CLIP mean and std, alpha discarded), which the
chat format writes as
`<|begin_of_image|><|image|><|end_of_image|>` with the pad expanded to the
image's tokens. That recipe has no video form, so the frontend declares no
video modality and a video part is refused.

## Tuning and capacity

`contract/tuning/<id>/schedules.json` holds a geometry's kernel schedule rows
(see [its README](contract/tuning/README.md)); flash's grouped MoE row tiles
are hand-picked until an offline sweep replaces them.
`model_glm5next_contract::capacity` plans each rank by formula: the text
weights, with images the encoder's image and buffers (1.05 and 2.26 GiB),
named engine workspaces plus a 2 GiB allowance, the KDA and indexer tail state
(36.9 MB per copy at TP4: one copy per slot without drafts, `rows + 1` with
them, and sixteen for the anchor spares, all allocated before the KV pool),
and the KV pool the host then sizes. KV tokens scale with the context-parallel size; the state does
not. Print the table of each checkpoint layout for slots 1 to 64, MTP K 0, 3
and 5, FP8 and BF16 KV and DCP 1, 2 and 4, text packages and packages
serving images, with

```sh
cargo run -p model-glm5next-contract --example capacity-table --locked --offline
```

The plan assumes 94 GiB usable per GPU, a conservative default inherited
from the qwen38 package plan: the board reports 97,887 MiB (95.59 GiB), and
the plan does not measure GLM's free memory after the driver and CUDA
context. On that assumption, 32 slots with FP8 KV and three draft tokens plan 0.67M KV
tokens at DCP 1, 1.22M at DCP 2 and 2.25M at DCP 4 for a text package
(0.22M, 0.32M and 0.45M with images), and 64 slots with MTP do not fit; these are thresholds of the assumption, not device limits. The
context-parallel figures count each rank's group buffers (0.42 GiB at DCP 2,
0.76 GiB at DCP 4: the gathered queries, the send and received attention
partials of 2,048-row pieces and the localized key lists), which the bundle
allocates before the KV pool. All figures are
estimates: serving's own pool differs (the measured pool sizes are on the
[model page](../../docs/model-glm5next.md#context-parallelism)).

Neither the plan nor serving names a term for CUDA graphs. Create captures
its decode, verify and drafter graphs after the host sizes the KV pool
([serving KV pool](../../docs/serving-kv-pool.md)), and the prefill
programs capture while serving, so every graph comes out of the memory the
pool leaves free: the host's 1 GiB reserve and the 10% of the rest that
`--kv-memory-fraction` 0.9 leaves. The soak's declared warm-up (see
"Qualification checks") measures what that leaves for graphs.

## Serving profiles

`profiles.json` declares TP4 only: `tp4-mtp` (the default, MTP with three
draft tokens), `tp4-ar`, `tp4-mtp-dcp2` and `tp4-mtp-dcp4` (context-parallel
sizes 2 and 4) and `tp4-mtp-k5` (five draft tokens, a benchmark arm that is
never selected by default) for the block-FP8 checkpoint, `tp4-mtp-nvfp4` for
local-inference-lab's NVFP4 export and `tp4-mtp-nvfp4-nvidia` for NVIDIA's.
Each asserts its layout as the model option
`checkpoint.quantization_layout`, which the package declares with its own
layout as the one allowed value, so the host refuses a profile on a package
of another layout. `compiler/src/preparation.rs` selects `tp4-mtp`,
`tp4-mtp-nvfp4` or `tp4-mtp-nvfp4-nvidia` by the checkpoint's quantization
layout. The package declares the MTP drafter at the contract's five drafts;
the served `--speculative-tokens` (the profile's count) sets the drafts, the
state copies and the verify graphs. All seven serve. Six are qualified
through 32 sequences on the adopted final gate (GATES4, 2026-10-11):
`tp4-mtp` by `qual.json`, `tp4-ar` by each configuration's autoregressive
arms, `tp4-mtp-dcp2` and `tp4-mtp-dcp4` by the `dcp2` and `dcp4`
configurations, and the NVFP4 profiles by `nvfp4` and `nvfp4-nvidia`.
GATES4 judged the soaks of `tp4-mtp` and `tp4-mtp-nvfp4` by the operator's
waiver; the declared warm-up rule now passes them. `tp4-mtp-k5` passed its
gate (then `qual.k5.json`, now the `k5` configuration) but stays
experimental: it is
a benchmark arm, never selected by default. The
[model page](../../docs/model-glm5next.md#serving-profiles) lists their
capacity, the deploy defaults and the adopted GSM8K floor.

## Reference runner

`reference.py` runs HF `glm5_next` (transformers 5.16.1) under
`tools/refrun` ([reference runner](../../docs/reference-runner.md)); its
docstring lists every replacement. The weights stay as stored and are
dequantized per call, and the sparse layers cache the normalized latent
instead of HF's expanded keys and values. Neither changes HF's math beyond
the order of sums. One change is deliberate: the indexer chooses its pools
by a stable descending sort, ties to the lower pool index, instead of
`torch.topk`. relu makes exact ties common, and `torch.topk` orders equal
scores by kernel, so HF's own selection changes with the prefill chunking;
lower-index ties are the engine's rule. The f32 gate run needs
`--dequant f32=rounded`, since the FP8 block scales use the full F32
mantissa. `reference_workloads.py` and `reference_fixtures.py` build the
step-30 workloads and skeletons; `test_reference.py` checks the adapter on
tiny synthetic checkpoints on CPU, and `test_nvfp4_reference.py` against the
NVFP4 export's headers. The adapter reads the NVFP4 export as stored and
needs `--dequant f32=rounded` for the f32 run as well.

The long-context reference holds `long_4k` and the last 1,024 positions of
`long_128k`, 131,072 tokens (2026-10-11). The default placement does not fit
it: GPU 3 (78.2 GiB of weights and three sparse layers) peaked at 94,794 MiB
on the 65,536-token pass, and the indexer's F32 scores and the latent cache
grow with the context. The 131,072-token runs place the weights by a
weighted `--devices` list, cuda:0 x12, cuda:1 x12, cuda:2 x13, cuda:3 x11
(73.4, 77.0, 77.2 and 71.2 GiB of weights; placement is not part of the
cache key). The f32 pass took 1,390 s (`long_128k` 1,314 s) with peaks of
90,088, 92,134, 93,034 and 89,186 MiB (nvidia-smi; boards 97,887), the bf16
pass 1,287 s. Per prefill chunk of 512 rows the time grows with the context
(about 4.2 s plus 26 us per context token), so 262,144 tokens would take
about 4,600 s, beyond one GPU job, and the runner has no way to split one
sequence over jobs without changing reference-bound code. Runs set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`;
without it, loading ran out of memory on GPU 1 from fragmentation (18 GiB
reserved but unused). The f64 teacher run is split so that `long_4k` is not
the first sequence the runner computes, since the runner scores that one
twice.

## Qualification checks

`qual.json` declares the plan step 39 gates. The operator adopted their
results from the final gate on 2026-10-11: GATES4, part A (the
autoregressive arms, no drafter) at cab96b599 and part B at 8c75e90bd.
Every check passed except two soaks under the plain memory rule, which the
operator waived then; the declared warm-up rule below passes them.
So `qualification.mechanics_only` is false in `qual.json` and in every
configuration it declares, except `taps`. That configuration's
`text_layers` check still needs its pinned layer reference
([below](#layer-reference)); its `encoder_parity` passed. The pinned
checkpoint identity, `f75ee9b0e79e11157206b410df9fb4e07db58cc8b0256ea2e7d763b26e9c8fcd`,
is the one the engine reported for the zai-org FP8 checkpoint at its pinned
revision (the startup `serving_identity` of a TP4 tp4-mtp serve,
2026-10-07); it is the SHA-256 of every safetensors file's name, length and
header, so it changes only with those files. The serving-configuration
variants pin the same checkpoint. The NVFP4 variants pin their exports'
identities as the engine's direct preparation reported them at TP4
(`chopinfer-model-glm5next prepare --direct`, 2026-10-07; serve refuses a
checkpoint whose identity differs from its package's):
`83b4297362ba87927e462afaff42c406e8a2d8a869fb36e14fdc65badb5201e6` for
local-inference-lab's (44 shards) and
`aadfa0fd2df0268c5ee9cba10c33388ac33d118126fd6cfacd2c00568b19c8e2` for
NVIDIA's (33 shards).

The limits are those of [the serving qualification
gates](../../docs/numerics.md#serving-qualification-gates), computed at gate
time from each adopted reference's BF16 and f64 runs scored against its F32
`exact` run (step 30, 2026-10-06):

| Set | Positions | BF16 mean, top-one | f64 mean, top-one | Limits: mean, decisive, near tie, total |
|---|---:|---|---|---|
| teacher, pooled | 3,781 | 0.063, 2.9% | 0.0005, 0.08% | 0.10, 8%, 8%, 10% |
| teacher, chat | 560 | 0.158, 7.0% | 0.002, 0.4% | 0.20, 8%, 8%, 10% |
| teacher, other categories | | at most 0.115 | at most 0.0006 | 0.15, 8%, 8%, 10% |
| long context (long_4k, last 1,024 of long_128k) | 1,536 | 0.011, 0.39% | none | 0.10, 8%, 8%, 10% |
| image probes, pooled (each probe judged alone) | 1,548 | 0.036, 3.0% | 0.0088, 0.9% | 0.15, 8%, 8%, 10% |
| chat check (17 chat texts, 2026-10-11) | 4,340 | 0.180, 6.4% | 0.014, 0.4% | 0.25, 8%, 8%, 10% |

No limit is widened beyond that rule: only the chat category's mean limit
rises, to its BF16 spread rounded up, as Qwen3.8's chat limit does. The
chat check (`chat`, `fixtures/glm5next/chat-reference.json`) scores the seven
teacher chat texts and ten longer conversations of 251 to 603 tokens
(`reference_workloads.py --chat`); eleven of its seventeen texts cross four
64-token pages, so they reach every rank at DCP 4. Its mean limit uses the
spread rule's headroom factor 1.25 (`SPREAD_MEAN_FACTOR`, adopted
2026-10-11, fixed before the references ran): 1.25 x 0.180 rounded up to
0.25. The tool-call conversations carry most of the BF16 error (0.26 to 0.34
nats per position, against 0.06 to 0.13 for the others). The NVFP4 exports'
chat references give limits 0.20 (local-inference-lab, BF16 0.143) and 0.25
(NVIDIA, BF16 0.181). The teacher check keeps its seven chat texts and its
limits.

The needle check (qual.json, DCP 2 and 4, both NVFP4 exports) runs at total
contexts of 8,192, 131,072 and 262,144 tokens: frozen token-ID prompts of
7,168, 130,048 and 261,120 tokens, each with 1,024 tokens to answer. It
requires the three needle values in order.
`fixtures/glm5next/capture/needles.py` builds the prompts on CPU from MiMo's
functional haystack in GLM's template (reasoning effort Low, the
`<|assistant|><think>` generation prompt) into
`$CHOPINFER_DATA/glm5next/needle/prompts.json`, with the digests in
`fixtures/glm5next/needles.json` (`--check` rebuilds and compares). 262,144
is the largest context because `tp4-mtp`'s KV pool with images held 442,112
tokens at 32 slots when the needles were chosen (install e881ef050; the
current measured pool is on the
[model page](../../docs/model-glm5next.md#context-parallelism)): the
declared 1,048,576-token context does not fit one sequence on the default
package.

On 2026-10-11 the chat check passed against `tp4-mtp` (DCP 1, 2 and 4, five
drafts, BF16 KV) and both NVFP4 exports; frontend_goldens against `tp4-mtp`
at DCP 1 and both NVFP4 exports; the needle against `tp4-mtp` at DCP 1, 2
and 4 and both NVFP4 exports; and `long_context` against the 131,072-token
reference at DCP 1, 2 and 4 only. So `capabilities.frontend` and
`capabilities.long_context` are true. The results per server are on the
[model page](../../docs/model-glm5next.md#coverage-checks-2026-10-11).
`logprobs_parity` stays undeclared: it compares a reference OpenAI server's
logprobs with the candidate's, and GLM's reference is HF through
`tools/refrun`, not a server.

The
declared pooled caps (mean 0.10, decisive 8%) are the floors; spec_verify
keeps 0.15 and 8%, and verify_exact's host-drafted minimum is the teacher
reference's 3,781 positions.

`qual.json` qualifies the default configuration alone, `tp4-mtp` at TP4,
FP8 latent KV and DCP 1, in one gate run against one server: a TP4 server
takes all four GPUs, so no gate run can reach a second configuration's
server. Every check that reads a serving profile requires
`context_parallel_size` 1 (the TP4 topology declares context-parallel
groups, so its serving identity reports the size at 1 as well), TP4 and FP8
KV. The other serving configurations are declared in `qual.json`'s
`configurations` (they replaced the `qual.<name>.json` files on
2026-10-13), and each runs against its own server:

- `dcp2` and `dcp4` (`--model glm5next.dcp2`, `.dcp4`):
  step 39's DCP checks (teacher, long_context, determinism, prefix_exact,
  preemption_exact, spec_verify) and verify_exact, which the MTP capability
  requires, each requiring `context_parallel_size` 2 or 4, against a
  `tp4-mtp` server started with `--context-parallel-size` 2 or 4.
- `k5` (`--model glm5next.k5`): `tp4-mtp-k5`'s checks (teacher,
  spec_verify, verify_exact, preemption_exact, determinism) against a
  `tp4-mtp` server started with `--speculative-tokens 5 --modalities text`
  (a package serving images does not fit five drafts at 32 slots). The
  teacher and the determinism MTP record must report exactly five draft
  tokens; its determinism records carry no image request.
- `kv-bf16` (`--model glm5next.kv-bf16`): BF16 KV's teacher, chat,
  prefix_exact and `teacher_arms`, which compares its teacher capture with
  the default configuration's recorded FP8 arm. Held on 2026-10-14
  (prose-frozen 16/141 against the 0.10 limit, the BF16 reference's own
  spread 14/141), declared again on 2026-10-15 under variant-spread.v2's
  small-category rule, and requalified on 2026-10-16
  ([model page](../../docs/model-glm5next.md#bf16-kv)).
- `taps` (`--model glm5next.taps`): `encoder_parity` and
  `text_layers` on bundle-glm5next's diagnostic-taps build. The checks admit
  a tap capture only against the live process that wrote it, and only that
  build writes taps, so it has its own gate and `qual.json`'s runs on the
  release build (bundle-glm5next's `diagnostic-taps` feature,
  `bundle/src/taps.rs`). `text_layers` compares one reference sequence's
  text taps with two `tools.refrun run --taps` runs of it, HF's F32 run
  (`--dequant f32=rounded`) and, as the floor, HF's BF16 run with the
  served FP8 latent ([layer reference](#layer-reference)); the runs are the
  declared `layer_reference` input the reference lock pins. It passes only
  with every tap REQUIRED_TAPS names compared (each layer's sublayer inputs
  and outputs, the latents,
  key lists and routes, `final.norm`), so the reference runs with taps
  `x.hc.attn,attn.out,x.hc.mlp,mlp.out,x.latent,x.indexer.ids,router.*,norm`
  and the server with the same sites.

The default gate run also records the FP8 GSM8K arm the NVFP4 configurations
compare against (`evals_gsm8k`).

prefix_exact reads its record in two shards (`prefix_exact.py record --shard
I/2`, inputs `prefix_record_1` and `prefix_record_2`), each from its own
server start, in every configuration that records it (GATES4 ran DCP 4 in
three). In GATES4 (8c75e90bd) one whole record took 1,726 s at DCP 1,
1,971 s at DCP 2 and 2,392 s at DCP 4. Its report needs every shard of the
count once, and on the DCP 1 record split offline into two or three shards
it equals the whole record's report.

The soak judges GLM's memory with a declared warm-up and cache allowance
(operator, 2026-10-11; the soak check's constants in `qual.json` and both
NVFP4 declarations, `soak.memory_rule` in tools/qual). A request's last
prompt chunk captures its prefill program at its first run and keeps it
(see "Serving contract and program"), at most 64 lengths. Each length holds
about 40 MiB of host memory and about 21 MiB of GPU memory across the four
ranks (5.6 MiB per rank by free memory). Over its life a server therefore
grows by at most about 2.5 GiB of host memory and 1.3 GiB of GPU memory, and
then stays flat. So the soak's growth after its first 180 s must stay within
512 MiB, host and GPU, and its growth over the whole soak within 512 MiB plus
the 64 programs' bound: 2,560 MiB host and 1,344 MiB GPU. The full soak runs
1,200 s, so its job fits the release preview's 1,740-second bound. The final
gate (GATES4, 8c75e90bd, 2026-10-11) ran the plain rule:

| Soak | Host growth (MiB) | GPU growth (MiB) | After 180 s, host / GPU (MiB) | Plain rule | Declared rule |
|---|---:|---:|---:|---|---|
| DCP 1, `tp4-mtp` with images (22 lengths) | 1,077.5 | 440 | 43.6 / 24 | fail (waived then) | pass |
| local-inference-lab NVFP4 gate | 529.2 | 280 | 48.1 / 32 | fail (waived then) | pass |
| NVIDIA NVFP4 gate | 480.0 | 240 | 43.3 / 24 | pass | pass |
| DCP 1, all 64 lengths captured at create (`CHOPINFER_GLM5NEXT_LAST_CHUNK_GRAPHS=1`, 300 s) | 331.9 | 64 | | pass | |

The same growth appears before the GLM backend port (9867df47a), so it is
the pinned programs, not a leak; a leak still fails, after the warm-up at
512 MiB. The allowance assumes GPU memory for all 64 programs stays free
after the KV pool is sized. It does, because the graphs are captured after
the host sizes the pool, out of the 1 GiB the host keeps free and the 10% of
the rest the pool leaves (see "Tuning and capacity"). In GATES4 every agreed
configuration had at least 989 MiB free per rank before its first graph
capture. DCP 4 with MTP is the tightest at 989.6 MiB; the others had 1,021.6
(DCP 2), 1,053.6 (DCP 1 with images, and BF16 KV), 1,181.6 (K=5), about 4,000
(both NVFP4 exports) and at least 2,461.6 (`tp4-ar`). All 64 programs with
every other graph took 460 MiB per rank at DCP 1. Projected from each
configuration's measured peak at 5.6 MiB per missing length, they take at
most about 500 MiB per rank, which leaves at least 490 MiB.

The frontend refuses disabled thinking (`allow_disabled: false`). Requests
that want an answer rather than reasoning (structured, the soak's media and
structured mix, prefix_exact's image chats, determinism's image request)
therefore send no thinking switch, and the api check requires the refusal
of a disabled-thinking request. Only structured's answer-only cases
(`reasoning_max_tokens`, 4,096) and the soak's media and structured mix
(4,096) get a larger budget. prefix_exact's image chats keep 8 tokens and
determinism's image request 48, so against GLM they may stop inside the
reasoning. They still prove what they check: prefix_exact's media verdict
reads only cached-token counts and encoder-cache hits, and determinism
compares the generated tokens, logprobs, reasoning and finish reasons bit
for bit, whichever part of the answer they fall in.

Determinism is gated at each size on two determinism records (`chopinfer
eval capture --kind determinism`, as `tools/qual/determinism.py record`
writes them): `determinism_ar` from an autoregressive server and
`determinism_spec` from an MTP server of that size. The check reads the MTP
record and scores it against the AR one (`reference_file`), so speculative
output must equal AR bit for bit, and each record's cold repeat and prefix
hits must equal its cold sends. Input rules bind each record to its size,
TP4, FP8 KV and its speculation (none for AR, at least one draft for MTP).
media_preprocessing scores the served `image.zero-padded-token-budget.v1`
recipe against `tools/refrun/preprocessing.py images` over HF's
Glm5NextImageProcessor, adopted as `media_preprocessing_reference`
(`$CHOPINFER_DATA/glm5next/refrun/preprocessing`). The checkpoint has no
preprocessor_config.json, which that producer binds, so it runs on the view
`preprocessing_view.py` writes: the checkpoint's files plus that name
holding processor_config.json's `image_processor` member (SHA-256
63bfc165...), which loads the same processor. On its six images (upscaled,
resampled and the 4500x4000 one capped at 8,000 tokens) the served rows
equal HF's F32 rows exactly, integer and normalized (2026-10-06). Not
declared: needle and
frontend_goldens (no GLM needle prompts or vLLM frontend
capture), and video, which the bundle refuses. The API sample and tool
conversation, the speculation prompts, GSM8K and the soak's blue-square
image come from MiMo's fixtures, as Qwen3.8's do.
`fixtures/glm5next/gate/` holds the frozen teacher texts
(`reference_fixtures.py teacher-texts`) and a copy of the image probes'
request capture.

Step 39's other default-config gates are release prerequisites outside
this declaration, recorded separately: kernel-lab reports for each GLM
engine, and bring-up (`chopinfer verify --model`, a smoke test and a GSM8K
screen at the floor the operator adopted on 2026-10-11, 193 of 200:
`preview.gsm8k_floor`, which `tools/qual/gsm8k.py` and the release preview
read; the
[model page](../../docs/model-glm5next.md#gsm8k-floor) gives the
evidence). The first screen scored 197 of 200 (2026-10-07, before the fixed
latent splits). GATES4's whole split scored 1,283 of 1,319 (FP8), 1,285
(local-inference-lab NVFP4) and 1,288 (NVIDIA NVFP4), and the screen's 200
items within it 196, 197 and 197. None is a gate check; spec_verify's
GSM8K questions are prompts, not that screen. `chopinfer verify --model` passed for all three
checkpoints on 2026-10-07 (every file against its hub blob digest).

### Layer reference

`text_layers` holds the served build's taps of one sequence
(`chat_effort_high`, 31 tokens) against HF's F32 run, with an HF BF16 run's
distance from that run as the floor. The `taps` configuration serves the
default FP8 latent KV cache, and plain HF BF16 has no FP8 path. The check's
first GLM runs (2026-10-14: the TP-split candidate ae8be147c and its base
85893d6f1, bitwise equal through layer 3's router) therefore failed first
at `L03.attn.out`, 3.25 times its plain BF16 floor: every sparse layer's
attention output carries the FP8 latent's storage error. The operator
calibrated the floor to the served format instead of changing a limit
(2026-10-14). The floor is HF's BF16 run with the served FP8 latent round
trip (`reference_latent_fp8.py`, the check's `ADAPTERS`); the error side
stays `reference.py`'s plain F32 run. So the check still measures the
served build's whole deviation, FP8 storage included, against the model's
F32 math, and allows three times what an HF BF16 implementation of the same
storage format deviates. FLOOR_RATIO, RELATIVE_RMS_LIMIT, REQUIRED_TAPS and
the rules are unchanged.

Only the floor takes the FP8 latent. The adapter's F32 arm quantizes the
unrounded F32 latent, so its codes are not the served writer's (the writer
quantizes the BF16 latent), and as the error side it would remove the
format's cost from the check. The BF16 arm's codes and scales are the
writer's for HF's BF16 latent. The adapter applies the round trip to the
latent rows `kv_b_proj` expands, so the attention reads the stored format
while the `x.latent` tap keeps the served tap's boundary, the normalized
latent before the FP8 write: a floor tapped after the round trip would
allow storage error at a tap that stores nothing (at `L03.x.latent` it
would rise 5.0 times). This is the comparison ACC2 (2026-10-08)
measured on the pre-split engine: against HF F32 with this floor every
dense tap of 631 short-text rows was within three times the floor (the
median over taps of each tap's median error over floor 0.999, the worst
1.06), while the BF16 KV engine sat at the plain floor. On the first eight
rows of the check's sequence the TP-split candidate's `L03.attn.out` is at
3.72 times the plain floor and HF BF16 with the FP8 latent at 3.71.

The reference (`$CHOPINFER_GLM5NEXT_LAYER_REFERENCE`) is a directory of two
runner runs of the same workload, both with `--taps --tap-sites
x.hc.attn,attn.out,x.hc.mlp,mlp.out,x.latent,x.indexer.ids,router.logits,router.weights,router.ids,norm`:

- `f32/`: `--adapter models/glm5next/reference.py --modes f32 --dequant
  f32=rounded`;
- `bf16/`: `--adapter models/glm5next/reference_latent_fp8.py --modes bf16`.

Each run's cache keys must name its adapter's current digest, and the
reference lock pins every file of both (670 files). A one-run reference of
`reference.py` no longer binds. The pinned runs are from 2026-10-14 on four
GPUs. Each mode's time below is the runner's wall time for that mode,
weight loading included: f32 62.8 s (44.7 s of it loading the weights) and
bf16 63.7 s (46.9 s loading). The f32 run's taps and scores are bitwise
equal to those of the earlier one-run reference. The bf16 run's taps equal
those of an earlier bf16 run tapped after the round trip, except the 11
`x.latent` taps, each of which that run held round-tripped; `L03.x.latent`,
with nothing upstream of it in FP8, equals plain HF BF16's.

### Routing near ties

A served row whose router picks a different expert set from the F32 run's
carries a whole expert's difference into every later tap, so a few such
rows can fail a tap that agrees everywhere else. `text_layers` judges these
by route-near-tie.v2 ([numerics](../../docs/numerics.md#routing-near-ties))
with the check's `ROUTER` constant: GLM's selection score is the sigmoid of
the router logit plus the layer's correction bias, top 8 of 288 experts in
one group. The bias is not tapped, so it is the declared input
`layer_router_bias`, `fixtures/glm5next/router-bias.json`: the 43 MoE
layers' `mlp.gate.e_score_correction_bias` (F32), extracted by
`tools/qual/router_bias.py extract` from the pinned checkpoint, which
verifies every shard against `checkpoint-files.json`. The reference lock
pins its bytes, and the check refuses a sidecar of another model or
checkpoint.

On the stage 5 taps capture of 2026-10-14 the plain rules fail five taps
(`L16.mlp.out`, `L17.x.hc.mlp`, `L18.x.hc.attn`, `L19.x.hc.mlp` and
`L21.x.hc.attn`), each by its aggregate alone. Of the sequence's 31
compared positions, 30 reroute first at a near tie (row 1's at L03, with a
flip gap of 6.6e-5 against the 7.2e-3 margin there) and row 8 does not
reroute; the BF16 run has no first flip beyond its margin. Removing row 1
alone brings each aggregate to between 0.23 and 0.34 of its bound, with row
1 at 0.03 to 0.10 of its row bound, a norm 0.954 to 1.004 times the F32
row's and a cosine of at least 0.998 with it, so the check matches (offline
replay, 2026-10-16). At `L16.mlp.out` the BF16 run's worst row has a
relative error of 0.364, so the row bound is 1.09: row 1 zeroed there
(relative error 1.0) is inside it, and the plausibility guard is what fails
the tap.
