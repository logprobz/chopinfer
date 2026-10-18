# Numerics

This page states the numerical contracts kernels and serving must meet, and
the accuracy gates that qualify a served model. Use the
[first-divergence workflow](debugging.md) to choose between fixed-schedule
exactness, structural journals and matched-input oracle comparisons.

Each kernel op has an f64 CPU oracle over the exact decoded inputs the
device sees. Format decoding, reference quantization, weight repacking,
router selection and KV layout use bit-exact checks. The window ring's K
and V planes, stored by the ring write and by a storing ring RoPE KV write,
are the planes `oracle::ring_kv_write` leaves, superseded rows unstored:
copies and E4M3 encodes compare bit for bit, and a fused write's rotated K
within the rotation's bound (the conformance kit's RoPE and KV write
family, `sm120_passes_the_rope_kv_family_on_a_gpu` in
`crates/kernel-lab/tests/kit.rs`).

Other outputs must satisfy the per-element bound
`|y - y_ref| <= r(y_ref) + 2^-22 * K * S`, where `r` is half a BF16 unit in
the last place for BF16 output and zero for F32, `K` is the reduction length
and `S` is the oracle's sum of absolute products. Layer outputs require
per-token cosine similarity of at least 0.9999.

FP8 E4M3 KV caches ([serving kv pool](serving-kv-pool.md)) are checked
twice, and the two checks bound different errors. The device arithmetic
error is checked exactly as above: attention over the stored codes must
meet the bound against the oracle over the same codes. The storage error,
the oracle over the stored codes against attention over the unrounded BF16
keys and values, has a statistical allowance of six standard deviations
(`kernel_core::kv_fp8::attention_sigma`, `kernel_lab::check_fp8_kv_output`):
each stored element's error is modeled as independent and uniform within
its half E4M3 step, propagated to first order through the softmax and the
value sum. The kernel-lab tests apply it to random BF16 operands, where
the model holds. It is not a bound: values repeated across keys round the
same way, so their errors add. A zero query over 1024 equal keys whose
values are all 1.0625 (an E4M3 tie that rounds to 1) has a storage error of
0.0625, 55 standard deviations, with exact arithmetic. The check's receipt
reports the device error and the storage error separately, and a refusal
names the one that exceeded.

On real operands (f64 attention over the stored codes, CPU, measured
2026-10-12) ordinary text stays inside the allowance. MiMo's nine global
layers, with Q, K and V from the f64 layer oracle over the GPU audit's
layer inputs for three 46- to 49-token texts at the shipped scales, peaked
at 4.8 standard deviations over 10.6 million output elements. Qwen3.8 was
measured on an HF BF16 surrogate, not the served arithmetic: the reference
runner's BF16 norm taps of full-attention layers 3 and 7 over five prompts
of up to 4,096 tokens, rotated with F64 angles instead of the pinned F32
inverse frequencies and BF16 cos/sin table, and K rounded to BF16 before
the FP8 encode (serving encodes the F32 rotated K), at the served scale
1.0. It peaked at 5.2 over 54 million. Both tails are lighter than a normal
distribution. Repetitive text does not stay inside: on MiMo layer 0, one
token repeated 512 times reached 39 standard deviations and a 49-token text
repeated ten times reached 12, because repeated tokens store the same value
rows, whose errors do not average out. Against the first-order worst case
stated at `attention_sigma`, the largest measured ratio was 1 (to f64
rounding), at early causal rows where one key carries nearly all the weight
and a tie's error is its full half step. That worst case is first order only, not a bound for a
finite perturbation through the softmax. The allowance therefore checks
scale handling on synthetic random operands only. FP8 KV accuracy on real
text is a model qualification result, not a consequence of this check.

Keep weight-only and activation-quantized paths under separate numerical
contract IDs. Unsupported queries fail closed. A tolerance cannot be added
after an exact check fails.

Prepared precision records state where scales enter the arithmetic. A BF16
weight operand decoded before MMA differs from an exact FP8-to-BF16 code
conversion whose block scale is applied after accumulation. Native MX and
NVFP4 block-scaled MMA, including NVFP4's final tensor-global scale product,
have separate scale-order descriptions; the BF16 weight decoder refuses
those contracts. BF16 weight reuse selects one supported scale order across
every phase and capacity, records it with the source recipe, and keeps the
required output and accumulator precision.

NVFP4 A16 uses `ExactLocalThenGlobal`: every finite E2M1 code times its
unsigned E4M3 local scale is exactly representable in BF16, and the F32
global is applied once after the full F32 reduction, before residuals,
SwiGLU and route weighting (gate and up keep their distinct expert globals).
NVFP4 lookups round the exact code/local/global product directly to their
final BF16 output. MXFP8 and MXFP4 A16 use `ExactScaledOperand`; non-finite
products and products not exactly representable in BF16 are rejected during
payload validation, as is UE8M0 byte 255. All E2M1 codes are exact at scale
bytes 0 through 252; larger finite scales need actual-code checks. All finite
E4M3 codes are exact at UE8M0 bytes 3 through 246; outside that interval
admission checks the actual codes. These oracles consume exact decoded
weights (the NVFP4 global in f64), so the bound is the pure accumulation
bound with no weight-rounding term.

The `bf16` activation policy requires BF16 operands, including inside
kernels; quantizing a BF16 pointer's contents to FP8 or FP4 does not meet
it. Activation-quantized paths declare conversions and calibration
separately. Serving a W4A4 export's weights under BF16 changes its
semantics and needs its own quality evaluation; see
[quantized checkpoint policies](quantized-checkpoints.md).

## Engine rules

New and changed SM120 kernels, shared or model-owned, follow these rules.
The links lead to each rule's details.

1. **Oracle bound.** Each op meets the per-element bound above against its
   f64 oracle over the exact decoded inputs, or a bit-exact check where one
   is stated. A tolerance cannot be added after an exact check fails.
2. **Fixed-order F32 accumulation, 128-K chains in dense GEMMs.** Every
   reduction that produces an output has a fixed order and accumulates in F32
   or wider (the filtered sampler's guarded histogram, rule 4, is the one
   unordered sum, and only steers a decision guarded by a margin and an exact
   fallback). In dense GEMMs no MMA chain from zero spans more than 128 K:
   BF16 chains restart every eight k16 steps and FP8 promotes to F32 after
   each 128-K scale block, and each chain's F32 temporary is added into the
   accumulator in K order ([GEMM
   formats](../crates/kernels-sm120/cuda/gemm/README.md#formats-and-mma-paths),
   [BF16
   accumulation](../crates/kernels-sm120/cuda/gemm/README.md#bf16-accumulation)).
   Unquantized F32 projections preserve their input values and write F32.
   With BF16 weights, the tensor path splits each activation exactly into
   three BF16 components. Each component has its own 128-K F32 chain; the
   hi, mid and lo partials merge in that order, then enter the outer sum in
   increasing K order. Components and weights must be normal or zero, and
   each component product must be finite, normal or zero, and exact in F32.
   Finite values outside that domain use SIMT F32 FMA. Raw F32 weights also
   use SIMT. Their F32 arithmetic retains subnormals and refuses non-finite
   accumulation; FTZ instructions and fast-math flushing do not satisfy this
   contract. The fixed nvcc flags in `kernel-build::NVCC_FLAGS` select neither
   `--ftz=true` nor `--use_fast_math`.

   The dedicated F32 oracle uses the original inputs and logical K. Write
   `t=min(K,128)`, `B=ceil(K/128)`, `u=2^-24` and `eta=2^-150`. Round-to-nearest
   with gradual underflow obeys `fl(z)=z*(1+delta)+epsilon`, with
   `|delta|<=u` and `|epsilon|<=eta`, in the absence of overflow. Each tensor
   product passes through at most `t-1` component additions, two component
   merges and `B-1` outer additions. The first product and first outer add
   are exact. Because the three components share the activation's sign,
   the sum of absolute component products remains S. With
   `G=(1+u)^(t+B)`, relative error is at most `(G-1)*S`. This is the direct
   product of rounding factors; the usual `1+gamma_n=1/(1-n*u)` can replace
   a factor `(1+u)^n` when `n*u<1`, but G remains valid beyond that limit.
   There are at most `3*(K-B)+2*B+(B-1)=3*K-1` tensor rounding sites. Each
   can contribute eta, amplified by at most G. SIMT uses at most
   `t+B-1` roundings per product and `K+B-1` sites, both below the tensor
   counts. The common gate is therefore `(G-1)*S + (3*K-1)*eta*G`, computed
   with upward rounding in F64. It includes the outer reduction and absolute
   underflow error, with no BF16 output-rounding allowance. Use
   `precision::f32_gemm::check_numeric`, whose oracle type is distinct from
   the generic relative-only GEMM gate. Casting the
   activation to BF16 or TF32 is not an implementation of this contract.
   Attention PV, the encoder's pair GEMMs and PV, and the MXFP4 expert
   mainloops keep one MMA chain into their F32 accumulator over the whole
   reduction, by a measured decision of 2026-10-04 ([persistent BF16
   chains](../crates/kernels-sm120/cuda/gemm/README.md#decision-2026-10-04-keep-persistent-bf16-chains)).
3. **Declared splits, summed in order.** A split-K entry declares its split
   count, the planner refuses any other, and the last CTA of a tile sums the
   F32 partials from zero in split order
   ([GEMM README](../crates/kernels-sm120/cuda/gemm/README.md)).
4. **No atomic arrival order reaches an output, no TF32.** Atomics count,
   hand out tickets or slots, or feed a guarded estimate; combines and
   permutations follow fixed orders and ranks, never arrival order
   ([row](../crates/kernels-sm120/cuda/row/README.md),
   [MoE](../crates/kernels-sm120/cuda/moe/README.md)). In served kernels the
   one float atomic sum is the filtered sampler's top-p histogram: each
   tile's mass is an exact integer multiple of 2^-52, but the F64 atomic
   additions that merge tiles can round by arrival order. The sampler accepts
   a cutoff from that histogram only when it clears the threshold by a margin
   of at least three times a bound that includes those unordered sums, and
   otherwise runs the oracle's exact sequential passes, so the kept set does
   not depend on arrival order ([filtered device
   sampling](#filtered-device-sampling)). F32 math uses FMA in a fixed order
   and never TF32; the cubin build tests of the latent, chained MoE and
   pooled-row entries reject TF32 instructions. Ties go to the lower index.
5. **Position-invariant prefill.** A prompt row's arithmetic depends only on
   its absolute position and its data, never on chunking, the step's other
   items or the grid. Each invariant entry declares it
   (`kernels_sm120::invariance`), and generated kernel-lab tests check it
   bitwise ([prefill chunk invariance](#prefill-chunk-invariance)).
6. **Decode row-local within a bucket.** A decode or verify row's reduction
   order may follow the step's row bucket, never the other rows' lengths or
   contents. Sampling chooses each row's token from that row alone; the
   sampling entry, and in the simple family the logprob normalizer's thread
   count, follow the whole dispatch. A row's bits can therefore change with
   the number of requests sharing its step, which is documented behavior
   ([batch shape](#batch-shape)).
7. **Separate contracts.** Weight-only and activation-quantized paths keep
   separate numerical contract IDs, and the `bf16` activation policy means
   BF16 operands inside kernels too.
8. **Accuracy before speed.** A precision option is adopted only with a
   resolved accuracy benefit against the reference, and a change that can
   alter served bits needs the evidence below before any performance claim.

### A speedup that changes bits

A change that keeps every bit needs only bitwise evidence: the affected
kernel-lab tests and a serving A/B under its bitwise gate. One that changes
bits, even within the bound, follows these steps in order
([performance](performance.md), [contribution tiers](contributing.md#contribution-tiers)):

1. **f64 oracle evidence first.** The changed entries pass their kernel-lab
   oracle tests under `CHOPINFER_KERNEL_REQUIRE=1` on an idle GPU, with the
   executed cases listed, and every entry that declares invariance passes
   its generated invariance tests. The bound itself does not change.
2. **A report-only serving A/B.** [Serving A/B](../tools/ab/serving/README.md)
   with `"base_new": "report-only"` for the declared change: base against
   new is reported, while each install's repeat and the AR against
   speculative comparisons stay bitwise gates.
3. **The model gates against pinned references.** Every affected model and
   profile passes its correctness suite against its pinned HF F32
   references (`tools/qual/release_preview.sh run --model M --suite full`;
   [serving qualification gates](#serving-qualification-gates)).
4. **No limit moves after a result.** Limits, margins, floors and references
   stay as declared. A new rule is stated and justified before anyone looks
   at whether runs pass it, and a lock or golden file is never regenerated
   to hide a mismatch.

Only then do the timings count.

## Text RoPE arithmetic and qualification

MiMo and Qwen text RoPE use the model's HF angle and cos/sin arithmetic, not
the real-valued f64 angle formula. The 32 inverse-frequency words for rotary
width 64 are Torch's F32 `1 / theta ** (arange(0, 64, 2).float() / 64)`.
MiMo uses theta 1e4 for SWA and DFlash and 1e7 for GA; Qwen uses 1e7.
Position converts to F32 and the angle product rounds once to F32. Accurate
trig evaluates that argument, rounds to F32, then to BF16 as the reference
casts its cos/sin; tables store those BF16 values as F32.

Rotation keeps F32 arithmetic: `p = RN32(y*s)`, then
`z = RN32(fma(x,c,+/-p))`, with one BF16 rounding at storage. BF16 operands
and BF16 cos/sin make the products exact in F32, so only the F32 sum and
the final storage round. Qwen keeps its normalized operands in F32 until
rotation and storage. This is not literal pure-HF BF16 rotation.

The kernel-core rotation (`AngleRule::HfF32AngleBf16CosSin` tables with
`RotaryArithmetic::F32Fma`) and the Qwen gated/QSA oracles model this
definition; the usual bound applies against it. MiMo table row and DFlash
checks also require exact BF16 output bits. The distance from pure-HF BF16
rotation on the same cos/sin is bounded by `hf_rope::pure_hf_distance_bound`;
that comparison does not relax the served kernel gate.

Entries that compute their angles on the device (`RotarySource::Computed`)
have their cosines and sines judged on their own: a rotation of the pair
(1, 0) computes its cosine and sine exactly, and the BF16 output holds
their roundings. `rotary::check_computed` admits each BF16-rounded device
cosine and sine within `rotary::computed_allowance` of the exact f64
angle's, an allowance that includes the BF16 rounding.
The same kit test runs an angle op (`... angles`) per computed `rope_kv`
and `dflash_rope` entry, the RoPE alone at positions up to 2^20 - 1. The
bundles serve table entries (`RotarySource::CosSinTable`), so this checks
the catalog's computed entries, not a served path.

Accurate trig is intentional (CPU Torch's approximate F32 trig differs after
BF16 rounding at a few dozen table entries). MiMo's context-sized tables cost
512 bytes per configured context token per rank, allocated before KV sizing.

## Batch shape

The numerical contract bounds each output against its oracle; it does not
require batch independence, but prefill provides it. For MiMo V2.6 and
Qwen3.8 (AR and speculative), a prompt row's arithmetic depends only on its
absolute position and its data: never on the step's live row count, the
row's place in its chunk, the other items in the step or the launch grid. A
cold prefill therefore gives the same bits under any chunking, and a
prefix-cache hit gives the bits of a cold prefill
([prefix-cache accuracy](#prefix-cache-accuracy)). Qwen needs its chunk ends
on multiples of 64, which its scheduler granule provides.

Decode and verify steps choose their F32 reduction order from the step's
shape:

- GEMM, expert and attention split counts and tile geometry from the row
  bucket;
- all-reduce protocols, and their norm orders, from row thresholds of the
  bucket;
- paged decode and verify attention cut each row's (verify: item's) keys
  into pieces from its own length and the bucket, so its bits do not depend
  on the other rows or items
  ([attention README](../crates/kernels-sm120/cuda/attention/README.md));
- MiMo's grouped experts run the ring decode entries (exact entries for
  exact verify rows) in buckets of up to 16 rows and the prefill route and
  prefill expert entries in wider buckets; either way a row's expert sums
  follow its own routes only.

So a decoding request's logits can change bitwise with the requests that
share its steps only through their number, at a bucket boundary, and under
speculation through the verify bucket, which follows every item's draft
length. A fixed-state replay at MiMo B64
(2026-10-04, `tools/dev/fixed-state-replay.py`) measured the former
coupling at 0.02-0.035 nats mean absolute target logprob per position over
40 decode steps when only the other rows' lengths were reordered: step-wide
paged decode attention and stream-K expert entries, which cut each
expert's K range by the step's routed items, each carried part of it. With
both removed the replay's probe rows were bitwise equal.

Qwen3.8 changes its reduction order at B8 (the warp blocks of its dense
decode projections and the K splits of its FP8 gate/up experts) and at B16
and B32 (sparse attention splits, which decode context parallelism does
not use); buckets above 32 run the prefill-shaped entries. A Qwen MTP
verify step holds at most 32 rows by default when the instance serves at
most 32 sequences; the diagnostic `CHOPINFER_SPECULATIVE_WIDE_ROWS=1`
allows wider verify steps ([debugging](debugging.md#reproduce-one-device-case)). A MiMo DFlash verify step holds up to eight rows per item, up
to 256, so at 32 sequences it can cross from the decode entries to the
prefill-shaped entries.

In Qwen3.8 the first differing value between B1 and B8 is the first layer's
linear-attention input projection (`gdn.qkv`), where the dense warp blocks
change from one to two. Repeated runs of Qwen3.8 and MiMo V2.6 at 4 to 32
concurrent greedy requests differed in some requests from run to run (Qwen
TP4 AR at eight requests: 41 of 224 request pairs over eight starts), while
single requests and replays of a fixed batch matched bit for bit. GLM 5.3
Flash matched in every repeated eight-request smoke run (2026-10-12) and in
its 16-request `preemption_exact` pairs; it was not measured at other
buckets.

This batch variance is documented behavior. Decode and verify outputs are held
to accuracy gates, not to batch independence. The gates that cover batched
decode are speculative verification and the concurrency qualification's
B64-against-B32 decode repeats (MiMo AR -0.0012 to +0.0039 nats, Qwen AR
+0.0028; [serving
profiles](serving-profiles.md#tp4-concurrency-qualification-2026-10-04)).
Speculative verification ([below](#serving-qualification-gates)) has two
forms. A single verify draw, as the release declarations run, may sit at
most 0.005 nats of mean error above its AR control, with a bound on excess
top-one disagreements and a coverage check. Pooled draws at several item
buckets, as MiMo DFlash records, each keep their own top-one bound and
coverage, while only the draws' mean is held to the 0.005 margin, so one
bucket can exceed it (MiMo DFlash measured +0.0074 at C32 and +0.0076 at
B64); MiMo DFlash's pooled mean is itself waived
([accuracy waiver](serving-profiles.md#accuracy-waiver-2026-10-06)). No gate
bounds every bucket uniformly at 0.005 nats, and none requires equality
between buckets or between runs. A request served alone, and a fixed batch replayed with the same
rows, are reproducible bit for bit when decoding is greedy or uses the same
explicit seed and sampling parameters (an unseeded sampled request draws a
fresh seed). Under concurrent load the bucket's reduction order shifts
logprobs and can change a greedy token. Measured against the same
requests served one at a time, 16 concurrent requests moved likely-token
logprobs by a median of 0.001 to 0.004 nats (99th percentile about 0.2, at
most 0.54), and the first differing token followed a top-2 gap of up to 0.54
nats ([serving profiles](serving-profiles.md#alpha-limits)). The spread
between buckets is small next to the gates: on the row-local MiMo build the AR
decode buckets' mean errors against exact ranged from 0.0538 to 0.0600 nats
from C1 to B64 (sd 0.0026, [served
error](#served-error-against-the-bf16-floor)). Making every bucket give a lone
row's bits would cost decode throughput at 8 to 32 sequences and fix each
bucket's kernel choices to the one-row arithmetic, so it is not done.

MiMo decode and verify steps, at TP2 and TP4, are row-local: a row's bits
depend only on its own history and inputs and on the bucket. Verify
attention and the DFlash drafter's block attention (a bidirectional verify
variant) became row-local on 2026-10-05, after the first requalification
below. Still coupled:

1. Bucket boundaries: every bucketed kernel (dense GEMM tiles and split-K,
   the expert entries at 1, 2-16 and 32+ rows, attention splits, all-reduce
   protocols) may change its reduction order when the live row or item
   count crosses a bucket.
2. The verify bucket follows the step's total draft rows, which adaptive
   speculation varies per item, so a companion's draft length can move an
   item across a verify bucket. This is inherent in batching verify steps.
3. Speculative and AR rows differ by design; exact verify rows equal one-row
   decode only up to eight rows.

Prefill, the MTP drafter and the logits gathers stay row-local. Sampling
chooses each row's token from that row alone, with two couplings to the
dispatch. The dispatch's rows together choose the sampling entry (the
simple or the filtered family). In the simple family, a reported logprob's
normalizer is summed with a thread count chosen for the whole dispatch: 256
threads when a row asks for top logprobs or the dispatch has at most eight
rows, otherwise 64. The filtered family always uses 256 threads for its
statistics and histogram. Logprob bits can therefore change with the
requests that share a step.
Served cost against the previous build (tok/s ratio, the 2026-10-05 B64
numerics run's served A/Bs, chat and GSM8K): TP4 AR C8 0.99-1.00, C16 1.00-1.01, C32 0.94,
C64 0.96-0.97; TP2 AR C16 1.01, C32 0.99; long-context decode at 16K 1.00.

The 2026-10-05 requalification on this build passed TP4 AR C64 accuracy (paired
and absolute teacher; B64 against C32 +0.0031 nats in all three repeats,
bit-identical), preemption and stability (GSM8K, spec_verify, soak), and TP2
AR accuracy. TP4 DFlash C64 failed the pooled speculative bucket gate:
+0.0085 nats against the 0.005 margin (CI95 [-0.0009, 0.0275]; top-1 flips
2.42% against AR's 2.44%; the paired teacher passed), where the previous
build measured +0.0023 once, so `tp4-dflash` serves 32 sequences
([serving profiles](serving-profiles.md#row-local-decode-requalification-2026-10-05)).
On the row-local verify build DFlash failed the same gate at +0.0082 nats
(CI95 [0.0001, 0.0233]) with the teacher gates passing; its pooled rows were
bitwise equal across two servers (16,205 forced positions; which bucket
executed each position is not recorded), so the excess is systematic, not
composition. The gate never ran at 32, so `tp4-dflash` became experimental
at 32 (on 2026-10-06 it was qualified through 64 with this gate waived; see
below). Row-local verify cost DFlash 0.3-1.1% of served tok/s (chat C32
0.997, C64 0.991; GSM8K C32 0.993, C64 0.989). No rerun was made to obtain a
pass. The 2026-10-05 DFlash accuracy draws then recorded more draws on the same
build and rescored them against the mean of the C32, B8, B16 and B64 AR
draws ([serving qualification gates](#serving-qualification-gates)): pooled
DFlash rows pass at 8 and 16 items (+0.0021 and +0.0005 nats) and still
fail at 32 and 64 (+0.0074 and +0.0076; against C32 alone +0.0080 and
+0.0082). AR B64 sits +0.0034 and AR C1 +0.0027 from the same mean.
The 2026-10-06 verify attention study found no precision difference behind
the gap:
- **Verify attention against decode.** The kernel-lab comparison
  `attention_verify_precision_against_decode` ran MiMo verify steps of 5 to
  64 items: one to eight rows each, contexts up to 32,761 tokens, sinks, and
  grid and model-like inputs. Each verify row was compared with the decode
  row at its position on identical inputs, both against the f64 oracle.
  Verify was never less precise than decode at B64 by more than 0.00001
  BF16 steps of mean error or 0.034 percentage points of inexact outputs.
  Past 32,768 keys, decode was the less precise path, since its rows take
  one probability term there.
- **K/V write.** The verify phase's row RoPE and K/V write give AR's bits
  at 8 to 64 items (`attention_verify_pipeline_against_ar`).
- **State handling.** Forced DFlash C1 with exact rows equalled AR C1 at all
  4,053 scored positions. Four items of two rows equalled it at every
  position an exact-row step computed; the other six positions came from
  zero-draft steps run as decode at buckets 2 to 4.

With exact rows off, the verify kernels sat above the C32 control, outside
`chat_tool_call`, by +0.0005 nats per position at one item and +0.0010 at
four. AR draws sit at -0.0001 to +0.0003, and pooled draws at 8 to 64 items
at +0.0017 to +0.0024. The excess is neither a precision loss of verify
attention on these inputs nor a K/V or state difference.
The 2026-10-06 verify excess study held the verify row bucket fixed with
`CHOPINFER_MIMO_VERIFY_BUCKET`. The excess follows the bucket, not the items
in a step:

- +0.0016 at 64 rows for one to eight items, and +0.0027 at 256 rows;
- rows bitwise equal across item counts wherever their step histories
  match, and between graph and eager verify.

Running wide verify attention as per-row exact rows
(`CHOPINFER_MIMO_VERIFY_ATTENTION=rows`) makes the 64- and 256-row buckets
bitwise identical but leaves +0.0021. Verify attention arithmetic is
therefore not the whole cause, and the cause is open. At 16 items the
excess is +0.0020 outside `chat_tool_call`, so a 16-sequence cap passes the
gate on this reference without removing it.
A request served alone is reproducible bit for bit, with or without
speculation, when it is greedy or repeats the same explicit seed and
sampling parameters: its prefill does not depend on chunking, MiMo verify steps of
up to eight rows run exact rows whose bits are the one-row decode step's,
and Qwen's C1 MTP K3 rows use AR's reduction order
([adaptive speculation](adaptive-speculation.md)). Batch shape alone moved
top-1 gaps by 0.03 to 0.17 nats at measured GSM8K branch points
(2026-10-01), which is why gates treat near ties separately ([top-one near ties](#top-one-near-ties)).

### Prefill chunk invariance

Each prefill kernel fixes a row's reduction order by the row's absolute
position:

- Fragment prefill GEMMs (every `_prefill` entry of both models, including
  MiMo's router and Qwen's `head_f32_prefill`): each tile's K is cut into the
  same S segments of whole 128-K blocks at every live M (S = 1, 2, 4 or 8,
  declared per entry as `DenseKernel::FragmentPrefill { segments }`), and an
  output element is the sum from zero, in segment order, of its segments'
  sums. The stream-K schedule follows live M and the grid; the bits do not
  ([GEMM README](../crates/kernels-sm120/cuda/gemm/README.md)).
- MiMo global-attention paged prefill: query tiles start at absolute
  multiples of 8 tokens. A row's keys split at multiples of
  L = 256 * ceil(keys / 1024), where `keys` is the last key count of the
  row's aligned tile, into at most 4 segments merged in segment order; each
  row's probability term count follows its own visible keys
  ([attention README](../crates/kernels-sm120/cuda/attention/README.md)).
- MiMo sliding-window ring prefill: never split; key tiles start at absolute
  multiples of 32, and term counts are per row.
- MiMo hidden all-reduce: one rounding and the low-latency norm order at
  every prefill row count ([MiMo prefill all-reduce](#mimo-prefill-all-reduce)).
- Qwen skinny GEMMs (`lin_ab`, `hc_inject`): eager prefill splits K into a
  fixed count per entry at every live M (`Entry::prefill_plan`); graph
  phases take one 256-K chunk per split at every M.
- Qwen QSA prefill attention: always one split with direct output
  ([SPARSE.md](../crates/kernels-sm120/cuda/attention/SPARSE.md)).
- Qwen gated delta rule: its 64-row sub-chunks start at each chunk's first
  row, so a Qwen chunk that stops short of the prompt end ends on a multiple
  of 64 in absolute position, and prefix anchors sit on multiples of 64. The
  bundle declares the granule as the manifest's
  `batch_limits.prefill_granule` ([request surface](serving-request-surface.md)).

Row stages, MoE prefill and combine, the convolution, PLE, n-grams, the QSA
indexer and top-k and the hyper-connection all-reduces are invariant as
well. Each entry declares it (`kernels_sm120::invariance`), and the kernel
lab's generated invariance tests check every declaring entry bitwise
against splits at its granule ([kernel lab](../crates/kernel-lab/README.md)).
Cost at TP4 (2026-09-30): fragment prefill entries +1 to +13% at 2,048 rows,
cold AR prefill throughput 1.6% to 7.3% lower, decode unchanged.

## Sampling contract

Sampling accepts a row with a live vocabulary width of at most 1,048,576;
storage after that width is padding and is never read for a decision. Input
values must be finite. The CPU reference for raw selected-token and top-N
logprobs (N at most 20) uses max-subtracted f64 logsumexp over the live raw
logits. Top-N ranks raw logits by descending value, then ascending token ID.
Grammar masks bound the raw normalization domain; if fewer than N tokens are
admitted, masked tokens fill the tail at negative infinity in ascending ID
order (the host fills it from the row mask).

For the sampling decision: mask end tokens until the request's minimum new
token count, apply logit bias, then repetition, frequency and presence
penalties. Repetition checks presence in prompt and output; it divides
positive seen logits and multiplies nonpositive ones by the penalty.
Frequency uses the output count; presence a nonzero output count. These
steps use f64 values converted exactly from F32 or BF16. Temperature zero
selects the largest processed logit, ties to the lowest token ID. A positive
temperature below about 1.5e-267 (`kernel_core::sampling::MIN_TEMPERATURE`),
where dividing by it can overflow, is temperature zero: the sampling
parameters hold 0, so the CPU and device samplers both take the greedy path.
The HTTP API raises positive temperatures to 0.01 first, so only direct
engine callers can reach this. For positive temperature: divide by
temperature, apply min-p relative to the largest probability, keep top-k by
descending value and ascending ID, then top-p over the remaining candidates
in that order, evaluated in f64 and including the candidate that first
reaches the threshold. Zero top-k and unit top-p disable those filters. This
order follows the vLLM V1 sampler's default path.

Random sampling uses Philox4x32-10. The 64-bit request seed supplies the two
key words, low word first. The counter words are token ID, generated token
index low word, generated token index high word, and zero. Ten rounds use
multipliers `0xd2511f53` and `0xcd9e8d57`, key increments `0x9e3779b9` and
`0xbb67ae85` and the standard word permutation. The first 53 output bits
form `u = (bits + 0.5) / 2^53`. Every kept token receives the Gumbel score
`(processed_logit - row_max) - ln(-ln(u))`; the highest wins, ties to the
lowest token ID. Random numbers and selection are therefore independent of
thread order, tile size and the filtered set. The CPU advances its
generated-token index once per successful call; the device receives it
explicitly.

Integer Philox output and greedy ties are exact across implementations. F64
`exp` and `ln` may differ by an ulp between CPU and CUDA, which can move a
top-p cutoff or a Gumbel winner when scores are very close; the CPU oracle
does not claim bit-identical GPU draws there.

The device's parallel unfiltered path scores each 1024-token tile with an
F32 polynomial logarithm, then recomputes candidates within
`0.1 + 1e-5 * abs(tile_max)` of the tile's approximate maximum with the f64
Gumbel expression. A residual risk remains if the approximation errs by more
than that guard; the GPU parity suite checks 10,000 random rows per
supported width and logits type but is not a formal bound. Tiles subtract
their own maximum and the merge shifts each winner by
`(tile_max - row_max) / temperature`.

### Top-one device fast path

Unfiltered greedy rows with zero or one top logprob use a tiled argmax:
each tile reads logits once into F32 (exact for BF16 and F32 inputs, so
greedy choices keep exact ordering, including signed-zero ties), sums
max-subtracted F32 exponentials in four accumulators, and combines warp and
tile sums and the final logarithm in f64. Selected and top-one raw logprobs
must be within `1e-6` absolute of the f64 CPU reference. No Philox or Gumbel
work is done; requests for more than one top logprob keep the sort.

### Filtered device sampling

Rows with penalties, logit bias, EOS masking, constraint masks, min-p, top-k
or top-p use tiled kernels (`crates/kernels-sm120/cuda/sampling/filtered.cuh`). A token outside a
constraint mask is excluded like a masked EOS token, also from the raw
maximum, normalizer and top list. A row left without a candidate (no
admitted token, or only stop tokens masked before the minimum length) fails
only its request, with the CPU sampler's `AllTokensMasked` error; the step's
other rows are served. No tolerance is added: the kept set and token equal
the oracle's whenever CUDA and the CPU round `exp` and `ln` alike.

Processed values use the oracle's f64 operations in its order, without
fused multiply-add contraction. Per-token work uses the monotone F32 key
`RN32(RN64(value * T))`, so ordering by key never contradicts ordering by
value; exact f64 values are computed for key ties, candidates, near Gumbel
winners and tokens near the min-p cutoff. The kept set is a prefix of the
oracle's ranking, found from a histogram of survivors with exact counts and
fixed-point masses. Top-k cutoffs are exact ranks. A top-p cutoff is
accepted only when the kept prefix clears the threshold, and the previous
prefix misses it, by more than a margin of at least three times a
first-order bound on the summation and `exp` error; otherwise the row sorts
all survivors and runs the oracle's sequential passes
(the comments in `filtered.cuh` give the bounds). Greedy rows ignore
top-p, top-k and min-p. GPU tests cover synthetic and captured logits, exact
and adjacent top-p ratios, equal-value top-k cutoffs, constraint masks and
the fallback.

### Fused MiMo routing

The fused decode router uses the same BF16 MMA operands, reduction orders,
sigmoid, correction bias and normalized weights as the separate router and
route kernels. Ascending source-route ranks determine the permutation,
independent of atomic arrival order. Expert IDs, order, weights, logits and
metadata must match the separate path bitwise at 1 to 32 rows. Serving uses
fusion through eight rows.

### Sampling precision relative to vLLM

Bias and penalty processing stay in f64: the contract defines accuracy
against an f64 oracle over exactly decoded inputs. vLLM V1 converts logits
and bias to F32 first, so discrete decisions can differ at rounding
boundaries (for example logits `[1, 1]` with a `1e-8` bias on token 1
select token 1 here and token 0 in vLLM).
`processing_keeps_exact_bias_and_penalty_decisions_at_f32_boundaries`
records these cases. This is an explicit compatibility exception, not a
tolerance; there is no F32 compatibility mode.

## MiMo prefill all-reduce

Every MiMo hidden all-reduce adds the residual to the F32 rank-ordered sum
before its one rounding to BF16, as decode does. Above 32 rows prefill runs
the phased reduce-scatter (`chop_comm_reduce_scatter_residual_bf16_v1`) and
fused gather (`chop_comm_allgather_presummed_norm_bf16_v1`), whose one-warp
norm sums a row's squares in the low-latency protocol's order. Prefill steps
of 3 to 32 rows take the verify entries (`LlOrder`), and one or two rows the
low-latency protocol. A prompt row's pre-norm and normalized bits are
therefore those of the same row reduced alone, at any row count
([comm README](../crates/kernels-sm120/cuda/comm/README.md)). Changes to
these paths need fixed-batch accuracy evidence before performance claims.

## Qwen QSA block selection

Qwen's QSA indexer keeps each row's 512 highest block scores
([SPARSE.md](../crates/kernels-sm120/cuda/attention/SPARSE.md)). The engine
stores the index projection, the normalized and rotated queries and the
block keys in BF16, scores them with BF16 MMA into F32, and selects exactly
on those F32 scores, ties to the lower block. Selection must equal the
top-k of these F32 scores exactly; equality to f64 scores of the same BF16
operands is not required. Keep the score error bound, tie rule,
deterministic replay and existing quality gates.

GPU diagnostics on 2026-10-03 compared Torch IEEE F32 and f64 scores over
the same BF16 indexer operands. The 4K prose chain differed on 2 of 24,540
sparse rows, and the 8K chain on 5 of 73,692. The conservative combined
score bound included 17,069 of the 4K rows (69.56%); it is an error
guarantee, not an estimate of selection flips. The median absolute boundary
gap was 0.003738, while the two differing rows had gaps of 1.256e-6 and
5.321e-7. These are diagnostic matmuls, not the serving MMA's bits.

Changing only score precision changed mean absolute logprobs by 0.00000845
nats on 4K prose and 0.000271 on 8K prose, versus their baseline reference
errors of 0.02157 and 0.05756. Neither changed a top-one choice. On the six
8K needle-answer positions, the difference was 0.01705 nats and f64
selection reduced reference error from 0.03984 to 0.02322, also without a
top-one change. This localized sensitivity warrants broader needle checks;
these measurements do not establish a need for exact f64 selection.
Swapping every row within the conservative bound perturbs thousands of
correctly ranked rows and cannot estimate the effect of actual F32 flips.

The indexer keeps norm and rotation arithmetic in F32 until one BF16
storage rounding, and stays that way. HF's BF16 sequence (the BF16
teacher's) would also round the norm output, each RoPE product and their
sum to BF16. Upstream HF f64 rounds none of these values.

A matched-input check on 2026-10-05 held the BF16 projection, pooled-key
storage, BF16 cos/sin and f64 scoring fixed on real projection rows (12
layers at 4K and 32K, 4 at 128K). The shipped sequence leaves every query
and key element within 0.5 BF16 ulp of the f64 value, the minimum for BF16
storage. HF rounding averages 0.57 to 0.70 ulp, puts 17% to 19% of
elements beyond 0.5 ulp and raises mean score error by 12% at 4K and 26%
at 128K. It mis-selects 9%, 17% and 23% more blocks than the shipped
sequence against the f64 selection at 4K, 32K and 128K, and changes the
selection of 30% (4K) to 79% (128K) of rows.

A GPU teacher-forced pilot on 2026-10-03 had found HF rounding closer to
upstream HF f64 over 1,024 prose positions (0.03467 against 0.03956 nats,
12 against 14 top-one disagreements). Paired by position, the gain is 8K
prose alone (0.00997 nats, 94% of it from ten positions); 4K worsened by
0.000179. The top-one change was six disagreements fixed and four new, and
five of the six 4K needle positions worsened. With no operand-level
mechanism behind it, this is chain variation from a large selection
perturbation, not an accuracy gain, so no indexer rounding change is
planned.

## Prefix-cache accuracy

A prefix-cache hit must give the bits of a cold prefill of the same prompt
(tokens, logprobs and top-k), for MiMo V2.6 and Qwen3.8 in their default
profiles (MiMo `tp4-dflash`, Qwen `tp4-mtp`) and AR. It follows from
[prefill chunk invariance](#prefill-chunk-invariance): a hit restores exactly
the state the cold prefill held at the anchor and prefills the rest as
another chunk. The correctness runner's determinism screen
([correctness runner](../tools/qual/README.md#correctness-runner))
enforces it on every run: prefix hits at 1,024 and 2,000 tokens must equal
the cold send bit for bit, and 1,024-token chunks must return the bits of
2,048-token chunks.

MiMo anchors sit at any position. Qwen anchors sit on multiples of 64 (its
prefill granule): a prompt-end capture moves to the last multiple of 64 at
or below the prompt end, kept inside the prompt's last chunk from
accumulators the kernels already hold ([Qwen3.8 bundle](bundle-qwen38.md)),
so the anchor changes no bit and adds no step. Beyond two such prompts in one
step the anchor ends a chunk instead, one extra step of the last 1 to 63
rows.

Bitwise equality is the gate. `tools/qual/prefix_exact.py` checks it at
every page boundary of a model's exact reference: per scored position a cold
one-shot prefill, a cold prefill with a chunk ending at the anchor and a
real hit must give the same raw logprob and top logprobs, and each request
must report the cached tokens the [anchor rules](serving-kv-pool.md) give
with the model's prefill granule. It also scores the rows against the exact
reference: hit mean absolute logprob error may exceed cold error by at most
0.005, and hit-only top-one disagreements may exceed cold-only ones by at
most two standard deviations of the discordant count.

Different media or media geometry must not share anchors past that input.
Greedy continuations after a hit must equal cold ones, and drafter
acceptance after a hit must be at least cold acceptance minus the larger of
0.02 and two binomial standard errors. Prefix soak canaries must keep their
expected greedy output (`tools/qual/soak.py`).

## Preemption resume

With prefix caching on (the default), a preemption tries to capture the
request's committed KV and model state as a private anchor that only this
request restores. When the capture succeeds and the anchor survives until
readmission (anchors are evictable under KV pressure), the request continues
its decode where it stopped and its C1 tokens, logprobs and top alternatives
match an unpreempted run bit for bit; `preempt_compare.py compare --exact`
gates that. A concurrent pair also differs in its batch companions from the
first step, so a check that pairs C16 records may instead judge every first
divergence, before the first preemption too, by the near-tie rule below
(`--ties-before-preemption`), with a C1 pair carrying the exact gate.

Otherwise the request prefills its prompt and emitted tokens again: when the
bundle refuses the capture (Qwen with MTP refuses anchors whose drafter rows
would reach a shared page before the anchor's last one), no page is free for
the snapshot, the anchor was evicted, the restore fails or prefix caching is
off. A recompute samples the resumed position from a prefill row instead of
a decode row, which can flip a greedy token at a near tie; the recompute
rule therefore accepts first divergences at top-2 gaps under 0.5 nats.
Neither row kind is the more accurate one: against an exact reference,
decode and cold prefill rows err alike (MiMo, 2026-09-30, about 0.02 nats
mean on the chosen token).

## Qwen3.8 vision tower

The Qwen3.8 vision tower keeps F32 activations, as HF's F32 model computes
it: F32 pixel rows, F32 rows between ops and an F32 residual stream that the
`proj` and `fc2` GEMMs add into. Tensor cores multiply BF16 operands, so
every activation operand (each GEMM's input, q, k and v) is a BF16 pair
`hi = bf16(x)`, `lo = bf16(x - hi)`, and both halves multiply the same BF16
weight or key with F32 accumulation: each product carries 16 bits of the
activation (at most 2^-16 relative error) instead of 8. Weights stay BF16 as
stored. Attention probabilities keep at least two BF16 terms, and the logits
drop only q lo k lo. The merged rows the language model reads stay BF16,
like its token embeddings. The kernels are described in
`crates/kernels-sm120/src/encoder/README.md`.

The served merged rows are within 0.0017 relative L2 of HF's F32 rows, the
rows' own BF16 rounding, against 0.026 to 0.055 for a BF16 tower (TP4
encoder captures, 2026-10-01). Accuracy comes first, so the cost is accepted:
a 1920 x 1080 image encodes in 239 ms instead of 138 ms, a 16 MP image in
9.47 s instead of 3.69 s, and the encoder buffers grow by 544 MiB per rank.

## MiMo vision row scale

MiMo's vision tower writes a massive activation in block 27: one channel of
every patch row jumps to 12,000 to 61,000 while the other 1,279 channels
stay in the tens to hundreds. Every BF16 implementation misses that value by
1 to 4 BF16 ulps (64 to 256 at those magnitudes), in a direction that
depends on the input, not the implementation. The merger's RMSNorm divides
each 1280-wide row by a norm this channel dominates, so the error becomes a
scale error of the whole merged row while the row's direction stays within
BF16 noise.

The `encoder_parity` gate therefore judges vision rows by direction and,
per item, by the largest row scale error
([encoder parity](../tools/qual/README.md#encoder-parity)). MiMo's encoders
run the F32 policy (`f32-pairs.v1`, as Qwen3.8's tower), which keeps the
scale error under the gate's 0.1% floor.

## Serving qualification gates

Accuracy is judged against the reference model code: the vendor model code
of the served checkpoint run on the [GPU reference runner](reference-runner.md),
its F32 run (`exact`) for closeness and its other runs for the realistic
spread. vLLM can itself be inaccurate, so no declaration sets a
`legacy_reference` vLLM capture and no gate scores against one; frontend and
media goldens come from the HF reference. The gates score only pinned
references and a server that logged the pinned checkpoint identity.

The teacher and multimodal gates allow, against the `exact` variant, pooled,
in each teacher text category and in each media case:

| Measure | Floor | Raised to the worst other variant's |
|---|---|---|
| Mean absolute logprob error | 0.15, or 0.10 for at least 1,000 positions | mean, rounded up to 0.05 |
| Decisive top-one flips | 8% | decisive rate |
| Near-tie top-one flips | 8% | near-tie rate, rounded up to 1% |
| All top-one flips | 10% | total disagreement, unrounded |

(Gate policy, 2026-10-01.) A reference variant is scored like a
server over the same positions. `tools/qual/limits.py` implements this
spread rule and the near-tie rule for every gate, beside the paired rule (a
margin against another arm) and the bit-exact rule; the precise computation
is in [check kinds](check-kinds.md#the-spread-rule-variant-spreadv2).

A check may declare a headroom factor on the worst variant's mean
(`teacher.SPREAD_MEAN_FACTOR`, default 1): the mean limit is then that mean
times the factor, rounded up to 0.05. GLM's chat check declares 1.25
(adopted 2026-10-11, fixed before its references existed). The
served FP8 latent KV is error the BF16 reference does not model: HF's BF16
run with the served FP8 latent sat 1.21 times above plain BF16 on GLM's
seven teacher chat texts (107.08 against 88.25 nats). Without a factor, a
spread just below a multiple of 0.05 would leave the limit at the spread
itself.

Each model's spread variants are its runner BF16 and f64 runs. The BF16
spread sets the limits that differ from the floors:

| Model | Set | Positions | Limits: mean, decisive, near tie, total |
|---|---|---:|---|
| MiMo (MOPD) | teacher, pooled | 4,108 | 0.15, 8%, 8%, 10% |
| MiMo | chat | 486 | 0.45, 9.3%, 8%, 13.4% |
| MiMo | other categories, each media case | | 0.15, 8%, 8%, 10% |
| Qwen3.8 | teacher, pooled | 4,176 | 0.10, 8%, 8%, 10% |
| Qwen3.8 | chat | 899 | 0.20, 8%, 8%, 10% |
| Qwen3.8 | other categories | | 0.15, 8%, 8%, 10% |
| Qwen3.8 | long context (last 1,024 of 32,768 tokens) | 1,024 | 0.10, 8%, 8%, 10% |

MiMo's chat texts (a JSON tool schema, call and result among them) are
ill-conditioned for every BF16 implementation: the reference's own BF16 run
sits 0.42 nats from F32 there, so chat's limits follow that spread. The
fixtures' provenance is in their READMEs
([MiMo](../fixtures/mimo26/README.md#reference-code-references),
[Qwen](../fixtures/qwen38/README.md#teacher-gate-reference)).

### Small categories

A limit raised to the reference's own spread can sit at that spread: an
engine exactly as accurate as the reference's BF16 run then fails the
limit by chance about as often as it passes, at any size. Measured on the
pinned references (2026-10-15), the 141-position, three-text prose-frozen
category's total top-1 limit (0.10) sits 0.03 binomial standard errors above
GLM's BF16 rate (14 of 141, 95% interval 5.5% to 16.1%), and its near-tie
limit 0.09 above (11 of 141); MiMo's prose-frozen near-tie limit sits 0.09
above, and MiMo's 486-position chat total and decisive limits equal its BF16
rates. Such an engine passes those limits with probability 0.53 to 0.58;
more positions cannot help while the limit equals the rate.

The spread rule therefore checks each top-1 limit of a category or media
case for power (variant-spread.v2, adopted on 2026-10-15): a limit that sits less than two binomial standard errors above
the BF16 run's own rate cannot judge the set, and there the metric also
passes when the candidate's flips exceed the BF16 run's on the same
positions by at most two standard deviations of the discordant count, the
paired rule's top-1 clause
([check kinds](check-kinds.md#the-spread-rule-variant-spreadv2)). Limits
with power, every mean limit and every pooled set are judged as before.
Positions within a text are not independent, so the true false-fail rate is
above the binomial 2.3% and the fallback's power below it: at 141 positions
it fails a candidate with twice the BF16 run's flips, not one with half as
many again. On the 17 stage 5 teacher receipts no verdict changes; GLM's
held BF16 KV run of 2026-10-14 (16 of 141 against the BF16 run's 14, 9 of
them shared) would pass by the paired clause (7 against 5), and its
requalification on 2026-10-16 passed that way
([BF16 KV](model-glm5next.md#bf16-kv)).

A gate result the rules do not cover can be waived only by declaration: a
check's `waivers` name the predicate or teacher category, a ceiling no
worse result passes under, the reason and the decision, and the
report lists each waiver it applied
([waivers](check-kinds.md#waivers)).

### Served error against the BF16 floor

The BF16 spread is also the yardstick for an accuracy audit of an A16
build. Per scored position, the excess is |served - exact| minus |BF16 -
exact|, both against the same token and F32 run; it is averaged per
category and per position bucket. Positions of one text are not
independent, so intervals treat whole texts as the units: a
cluster-robust interval with Student t at texts - 1 degrees of freedom,
beside the text-clustered bootstrap `tools/qual/teacher.py` uses. With a
handful of texts either interval is a guide, not a test; a set of fewer
than five texts (a single long-context text, say) is reported without
one. An interval that contains zero does not show the build equals its
floor, only that the set cannot tell them apart.

The BF16 run does not round where the engines do: it rounds the
dequantized FP8 weights and the attention scores to BF16, which the
engines do not (block scales apply to F32 MMA partials, scores stay F32),
so a build can sit below it. A model's audit results belong on its model
page.

Every multimodal greedy token up to its first departure from the reference
answer must lie within 0.3 nats of the exact top-one (2.3 for audio).
Qwen3.8's image probes are held to the HF reference model's F32 answers:
short answers by that rule, free-text answers teacher-forced within limits
from the reference's BF16 and f64 spread (at most 0.30 mean |dlogprob|, per
probe). MiMo's frozen GSM8K gate requires at least 193/200; its needle gate
covers 8,192, 131,072, 524,288 and 1,048,576 tokens.

Speculative verification compares each prepared row bucket against its
paired AR baseline at the same concurrency, over exactly the positions the
reference forcing can score (`verify_exact.py`). Mean error against exact
may worsen by at most 0.005, and excess top-one disagreements are bounded by
two standard deviations of the discordant count. On the row-local MiMo build
each AR decode bucket is one fixed draw whose mean error against exact
ranges from 0.0538 to 0.0600 nats over C1, B8, B16, C32 and B64 (sd 0.0026),
so a single C32 control moves verdicts. `verify_exact.py report --control`
therefore compares each bucket against the mean of several AR draws: C32 as
`--ar`, with B8, B16 and B64 (B64 left out when it is the candidate) as
controls. The top-one counts still pair with C32. Against
a four-draw mean a candidate's gap has an sd of about 0.0029 nats, against
0.0037 for one draw, and the margin stays 0.005. Rescored on recorded MiMo
receipts, AR B64 is +0.0034 (+0.0031 against C32 alone) and AR C1 +0.0027;
pooled DFlash is +0.0021 at C8 and +0.0005 at C16 (pass), and +0.0074 at
C32 and +0.0076 at B64 (fail). A speculative candidate is therefore scored
as the mean of several draws at different item buckets (`report
--pool-verify`): MiMo DFlash records pooled rows at C8, C16 and C32 without
filler and at B64, and the mean of the four draws' row errors at each
position may sit at most 0.005 above the four-draw AR mean. Each draw keeps
its own top-one bound and coverage; its single-draw gap is reported, not
gated. Pooling averages across buckets, so a bucket whose own gap exceeds
the margin passes when the mean is within it. This rule was adopted on
2026-10-06, after DFlash C64 had failed the single-draw gate three times
(+0.0085, +0.0082 and +0.0076) and the verify excess study had shown that the
excess follows the verify row bucket; it changes the gate after those
failures rather than removing the excess. On the recorded receipts the
four-draw gap is +0.0044 (CI95 [+0.0007, +0.0110]). The first requalification
under the rule failed at +0.0057 (CI95 [+0.0003, +0.0162]): C8 and C16 drew
+0.0051 and +0.0026 against the +0.0021 and +0.0005 projected, while C32 and
B64 repeated +0.0074 and +0.0076 ([serving
profiles](serving-profiles.md#row-local-decode-requalification-2026-10-05)).
The maintainers then waived the pooled mean for MiMo DFlash on sampling-level
evidence ([accuracy waiver](serving-profiles.md#accuracy-waiver-2026-10-06));
the 0.005 margin itself is unchanged and still gates every other candidate.
The waiver is stated here and in the profile text, not in
`models/mimo26/qual.json`: the declared verify gate judges one record, not
the pooled four-draw mean the waiver covers, and a declared waiver must name
a declared predicate. It moves into the declaration when the pooled gate is
declared (decided 2026-10-15).
Fixed-K3 MTP state runs
and the release and preview declarations keep one AR control and one draw.
The preview keeps exact C1
greedy equality for Qwen MTP; MiMo DFlash uses the `spec_verify.py`
copy-set bounds (mean |dlogprob| <= 0.15, top-1 disagreement <= 0.08) plus
paired GSM8K. See the [speculation contract](adaptive-speculation.md).
Stochastic checks require distributional correctness, not matching text.

Qwen's layer comparison (`qwen38_layers`) judges every tap of a bundle dump
against the runner's F32 taps of the same prompt through `tapdiff.py`: the
median row error at most three times the runner's BF16 distance from F32
(0.05 where that floor is zero), and the tap's relative L2 over all rows at
most three times the BF16 run's (at least 2^-8). It counts nothing until the
dump is bound to the live candidate process and request that wrote it
([tap captures](../tools/qual/README.md#tap-captures)) and the runner run is
bound to the pinned lock. These counts and thresholds are encoded in
`models/qwen38/qual.json`; changing fixtures must not silently weaken them.
GLM's `text_layers` applies the same rules to one sequence. GLM serves its
latent KV in FP8, which the plain BF16 run does not model, so its floor is
HF's BF16 run with the served FP8 latent round trip while the error is
still taken against the plain F32 run
([layer reference](../models/glm5next/README.md#layer-reference)): the
format's cost stays in the measured error, as it does for the chat check,
whose headroom factor covers the same format on the spread side. Both
layer checks may judge routing flips separately
([routing near ties](#routing-near-ties)).

These gates are separate from exact kernel checks and do not qualify
unsupported serving profiles.

### Speculative continuations

A profile without exact AR/speculative equality is also judged on its own
output after the speculative and AR texts diverge, where spec_verify stops
scoring. `spec_continuation` records each arm's continuations (the copy,
GSM8K and chat prompts greedily at 64 tokens; the chat and 16 open-ended
prompts at temperature 1 and top_p 1 under two seeds at 128 tokens),
teacher-forces each arm's emitted token IDs through the reference runner in
the pinned teacher reference's F32 and BF16 modes, lock, adapter and
environment, and scores every emitted token against the reference row of
the history that produced it. The rule (spec-continuation.v1, fixed on
2026-10-15 before any continuation was captured, and adopted the same day
for MiMo `tp4-dflash`):

1. Coverage: every frozen request is recorded in both arms and every
   emitted token is scored; a reference row of another history, or one not
   computed as the pinned reference was, leaves the request unscored and the
   check insufficient. The sampled set holds at least 1,000 positions
   (`LARGE_SET_POSITIONS`) whose F32 reference top-5 mass is below 0.95 in
   each arm.
2. Limits: the spread rule above with its floors unchanged, per arm, per set
   and pooled over the greedy sets, on mean |dlogprob| at the emitted token
   and, on greedy sets (where the emitted token is the arm's top-1), the
   near-tie, decisive and total top-one rates; BF16 is the spread variant.
   The sampled set is judged on the mean.
3. The AR control meets the same limits; a set where it does not is
   insufficient, never a pass.
4. The check passes when 1 to 3 hold and the candidate meets every limit,
   and fails when the candidate misses a limit the control met.

The candidate-minus-control mean differences (request-clustered CI95) and
the sampled set's mean of logprob_arm - logprob_reference, an unbiased
estimate of the mean KL divergence from the reference over the whole
vocabulary, with its Pinsker bound on total variation, are reported, not
gated. The check adds no paired margin, so it leaves the DFlash pooled
verify-row waiver ([serving profiles](serving-profiles.md#accuracy-waiver-2026-10-06))
as it was. The receipt records each reference run's GPU execution (devices,
timings, which entries were computed and which came from the reference cache
with their stored provenance) apart from the CPU scoring.

### Top-one near ties

At a scored position where the engine's top-one differs from the reference's
(a flip), the gate reads the reference's own gap between its top-one and the
engine's token from the reference's stored top-k. At most 0.3 nats
(inclusive) the flip is a near tie; beyond that, or when the engine's token
is outside the stored top-k, it is decisive. One function
(`limits.near_tie`) holds the boundary for every gate, including the
image-probe and media greedy walks. A reference that stores no top-k makes
every flip decisive.

Near ties have their own limit so that systematic near ties still fail: an
engine that swaps the two leading tokens at every position while keeping
the teacher token's logprob scores zero mean error and zero decisive flips,
and fails on its near-tie rate. The cap on all flips together, never
rounded, keeps the two allowances from adding up past the reference's own
disagreement. Receipts keep the served prompt-logprob rows the flips come
from, bound by digest, and the check kinds recompute the flips from them
([check kinds](check-kinds.md#the-near-tie-rule-top1-near-tiev1)).

The 0.3-nat constant matches how far batch shape and BF16 rounding move
leading-token gaps ([batch shape](#batch-shape)): flips that counting every
flip used to fail were mostly near ties that the reference's own BF16 run
also makes.

Paired GSM8K applies the same limit between arms: for each item one arm
answered correctly, the release preview teacher-forces the arms' shared
output prefix on a live server at C1 and reads both branch tokens'
logprobs and ranks. A loss is a near-tie branch when both tokens rank within
the top 20 and their gap is at most 0.3 nats. Speculation may lose at most
one item decisively, a lost item the probe cannot measure makes the check
insufficient, and both arms must still score at least the model's declared
floor (MiMo 193/200, Qwen 194/200).

### Routing near ties

A layer comparison sees routing flips too: the engine's top-k expert set
differs from the F32 reference's at a row, so that row's MoE output and
everything after it in the row's own residual stream differ by a whole
expert's contribution, while every kernel is correct. A correct BF16
implementation does this as well: GLM's BF16 floor run routes every one of
its 31 reference rows differently at some layer between 3 and 41. A layer
check that declares its router (`ROUTER`,
[tools/qual](../tools/qual/README.md#text-layers)) judges flips by
route-near-tie.v2 (`tools/qual/tapdiff.py`):

- The gap of a flip is the reference's own: the largest pair of experts the
  engine reversed under F32's selection scores (the router's score function
  plus its correction bias, rebuilt from the F32 logits in binary64), not the
  k-th/(k+1)-th gap.
- The margin of a layer is the floor ratio (3) times the largest movement
  of F32's k-th/(k+1)-th score gap in the BF16 run, over rows the BF16 run
  has not routed differently at an earlier layer (rows that it has measure
  a different expert's output, not rounding), cumulative over the layers so
  far since error grows with depth, raised to two F32 ulps of the scores.
  This plays the part of the 0.3-nat top-one constant: the largest move
  rounding makes, with the headroom the layer check already grants an engine
  over the BF16 run. BF16 resolution at the score's own magnitude is not
  the yardstick: GLM's biased scores sit near 9.6, where one BF16 ulp
  (2^-4) spans many ranks, and neither side rounds scores to BF16.
- A row's first flip decides it: a near tie when its gap is at most the
  margin, decisive otherwise. Later flips in that row are counted, not
  judged again.
- Only a tap that fails nothing but its whole-tap aggregate is re-judged.
  It passes when removing the fewest rows behind an earlier near-tie first
  flip, largest squared error first, makes the aggregate over the remaining
  rows pass against the BF16 run's over the same rows, and each removed row
  is within the ratio times the BF16 run's worst row at that tap and is
  plausible: every value finite, its norm within a factor of the ratio of
  the F32 row's, and a positive cosine with the F32 row. A row with a
  non-finite value, a norm outside 1/3 to 3 times the F32 row's (a zeroed
  row among them) or a cosine at or below 0 is not excused. Every
  other rule (each tap's median, the `router.ids` disagreement count, router
  weights on unflipped rows) is unchanged, so nothing that passes without
  the rule fails with it, and a flip at a gap beyond the margin is judged by
  the plain rules.

Adopted with minimal exclusion on 2026-10-15, with the
plausibility guard added on 2026-10-16. On GLM's reference
the margins run from 7.2e-3 at layer 3 to 3.69e-2 from layer 18 on. GLM's
2026-10-14 layer dumps (TP-split, stage 5 taps) failed five taps on the
aggregate alone, all from row 1, whose first flip is a near tie at layer 3
(gap 6.6e-5); its layer 16 flip swapped F32's 7th expert for its 19th (gap
8.4e-4, margin 1.42e-2). Under the rule, row 1 alone explains each of the
five taps, leaving aggregates at 0.23 to 0.34 of their bounds, and row 1
sits at 0.03 to 0.10 of its per-row bound, with a norm 0.954 to 1.004 times
the F32 row's and a cosine of at least 0.998 with it. The per-row bound is
loose on `mlp.out` taps, whose worst BF16 row is itself a rerouted row: at
layer 16 its relative error is 0.364, so the bound is 1.09, and a zeroed
row (relative error 1.0) is inside it. The plausibility guard rejects such
a row (its norm ratio is 0); a moderately wrong row that keeps the F32
row's norm and direction is still caught only by the per-row bound.

## Scope of offline oracle evidence

MiMo layer boundaries keep per-token cosine 0.9999, and MiMo isolated QKV
exports the f64 result and sum of absolute products for K=4096 under the
`gemm.w-fp8-block128-f32.a-bf16.acc-f32.v1` bound. Cosine evidence never
becomes an elementwise operation claim.

`kernel-lab`'s `mimo_layers::real_weight_layers` exports real-weight layer
evidence on four idle GPUs into `CHOPINFER_MIMO_DIAGNOSTIC_DIR` (a fresh
external directory); `CHOPINFER_MIMO_CHECKPOINT_ID` selects the registered
identity (`mopd` by default), and the layer images read from
`CHOPINFER_MIMO_CHECKPOINT` must authenticate against it. Seal each
exported directory with the offline `seal` command before comparison. This
is a seeded hidden-input layer chain, not a full-model token run; missing bounds, state, ranks or operands are
insufficient. See [debugging](debugging.md) for first-divergence reporting;
Qwen eager synchronized dumps cannot clear a graph-only failure.

A precision option is adopted only with a resolved accuracy benefit against
the reference.
