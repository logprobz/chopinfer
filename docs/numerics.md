# Numerics

This page states the numerical contracts kernels and serving must meet, and
the accuracy gates that qualify a served model. Use the
[first-divergence workflow](contributing.md#diagnostics) to choose between fixed-schedule
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

FP8 E4M3 KV caches ([serving kv pool](numerics.md#kv-offload-and-prefix-state)) are checked
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
[quantized checkpoint policies](kernel-library.md#activation-policy).

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
([performance](contributing.md#measuring-a-change), [contribution tiers](contributing.md#contribution-tiers)):

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

### Lossless weight encodings

A weight stored in a smaller encoding must decode to the exact bytes of its
raw section, so its consumers' arithmetic does not change. The host packs
the raw section, validates it as before, then encodes it. An encoding admits
every byte value (UE8M0 253 and 254 stay admitted). Its CPU oracle decodes
the bytes back, and a device decoder must match that oracle bit for bit.
Evidence for an encoded bank is therefore bitwise: the decoder's kernel-lab
tests against the CPU oracle, and a serving A/B between the encoded and raw
plans (`CHOPINFER_WEIGHT_CENSUS=0`) under its bitwise gate.

## Row status

Kernels that can meet invalid data report it per row instead of trapping:
the trained block quantizer (`BlockQuantizeQuery`), quantized clamped SwiGLU
(`SiluMulQuantizeBuffers`), the fused paged index-key and compressed-KV
writers, latent query rotation (`LatentQueryRopeBuffers`), the latent window
writer (`LatentWindowWriteBuffers`) and the plain F32 RMSNorm
(`RmsNormBuffers` with `numerics = Some(RmsNormNumerics::TREE_256)`; legacy
BF16 entries pass `None` and keep their ABI).

Each launch takes a caller-owned U32 `status` plane and a U64
`status_row_offset` in words. The plane covers
`(status_row_offset + planned_rows) * 4` bytes, checked on the host, and
aliases no other role or cache plane. Kernels OR flags into live rows' words
and never clear a bit; other words, inactive rows included, are unchanged.
Callers clear the selected words once before an independent step, order
launches that share rows, and check every selected word before accepting the
step's outputs.

| Bit | Name | Meaning |
|---|---|---|
| 1 | `NONFINITE_INPUT` | An original or BF16-rounded input, weight or rotation table value is NaN or infinity |
| 2 | `SCALE_OVERFLOW` | Generic caller-reported scale failure; finite E2M1/E4M3 saturation does not set it |
| 4 | `UNREPRESENTABLE_OUTPUT` | A result is not finite and exact in BF16, or F32 arithmetic became nonfinite |

A row with a prior or new flag is written as positive zero; packed outputs
get zero codes and scale 1 (UE8M0 byte 127 or E4M3 `0x38`), never the NaN
scale `0xff`. Every group or head of a row is validated before any of it is
written. Query rotation keeps the F32 query row and zeroes every BF16 head.
The latent window store skips a flagged row and leaves both planes' previous
bytes, so do not advance retained bounds or accept a staged attention result
for a failed step. In the paged key writers, out-of-range positions, items,
rotary rows and pages are holes: they read and write nothing and leave the
status word unchanged. SwiGLU skips the product of a nonfinite operand, so an
infinite gate cannot produce NaN; its registration still refuses clamps with
`max(gate_max, 1) * up_limit` above the BF16 maximum.

E2M1 with E4M3 scales follows the pinned TileLang 0.1.12 vendor code:
`amax / 6` converts to E4M3 with SATFINITE, so a finite amax above `6 * 448`
is valid, takes scale byte `0x7e` and clips values above 2688, with no flag.
The position-key writer's `IndexerKeyFormat` selects storage: `Fp4` keeps
the E2M1 and UE8M0 planes, and `Bf16` stores each exact native product in
channel order with no scale plane (an empty scales view, scale stride zero),
after the same checks, so it admits no row `Fp4` refuses.

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

Prefill arithmetic depends on a row's absolute position and data, not its
chunk, companion requests or launch grid. MiMo and Qwen cold prefill and
prefix hits therefore keep the same bits; Qwen chunk ends must align to 64.

Decode and verify may select different F32 reduction orders at row-bucket
boundaries: GEMM/expert splits, attention splits and all-reduce protocols.
Within a bucket each row follows its own history, lengths and routes.
The verify bucket also depends on the total draft rows. Exact verify rows
match one-row decode only within their declared range.

A lone request or fixed-batch replay is bit-reproducible with greedy decoding
or the same explicit seed and sampling parameters. Unseeded sampling draws
a fresh seed. Concurrent requests may produce different logprobs and tokens
as batch composition changes; the accuracy gates do not require equality
between buckets or repeated concurrent runs.

Sampling chooses each row's token independently, but the dispatch selects
its kernel family. The simple family's raw-logprob normalizer uses 256
threads when top logprobs are requested or the dispatch has at most eight
rows, otherwise 64. The filtered family always uses 256 threads. Thus
reported logprob bits may change across dispatch shapes too.

Speculative verification allows 0.005 nats of mean error above its AR
control, plus top-one and coverage gates. Single-draw and pooled-draw rules
are distinct: a pooled mean does not bound each bucket at 0.005. MiMo
DFlash's pooled mean has an explicit
[accuracy waiver](serving-profiles.md#accuracy-waiver-2026-10-06);
the top-one, coverage and teacher gates remain required.

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

### Coupled sampled drafts

A drafter may propose for a sampled request from the target sampler's own
noise (`kernel_core::draft_noise`, capability `chop.draft-sampling.v1`).
Verify row `j` of an item samples at generated index `n + j`, where `n` is
the index of the item's first sampled row. Draft position `j` proposes the
argmax of `x / T + g` over the drafter's logits `x`, with `g` the noise of
that index, so a drafter that ranks the leading Gumbel scores as the target
does proposes the target's own sample. Acceptance does not change: a draft
is kept only when it equals the target's sample at its row, and the
emitted tokens are those of AR sampling with the same seed. The drafter
applies only the temperature; penalties, logit bias, masks, min-p, top-k
and top-p are not applied, and a draft they exclude is rejected. The
device adds the sampler's F32 pruning noise, within
`draft_noise::NOISE_ERROR` of the f64 noise, so a near tie between two
Gumbel scores can change a proposal but never an emitted token. Greedy
items' logits keep their bits.

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

The indexer keeps norm and rotation in F32 until one BF16 storage rounding.
Extra HF-style intermediate BF16 roundings increase operand error; a closer
full-chain result on a small prompt set does not establish a better arithmetic
contract. Changes to selection precision or rounding require matched-input
oracle evidence and the model's unchanged quality gates.

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
accumulators the kernels already hold ([Qwen3.8 bundle](numerics.md#kv-offload-and-prefix-state)),
so the anchor changes no bit and adds no step. Beyond two such prompts in one
step the anchor ends a chunk instead, one extra step of the last 1 to 63
rows.

Bitwise equality is the gate. `tools/qual/prefix_exact.py` checks it at
every page boundary of a model's exact reference: per scored position a cold
one-shot prefill, a cold prefill with a chunk ending at the anchor and a
real hit must give the same raw logprob and top logprobs, and each request
must report the cached tokens the [anchor rules](numerics.md#kv-offload-and-prefix-state) give
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

The F32 policy avoids the larger error of an all-BF16 vision tower. Its
extra compute and buffer cost is accepted for accuracy; changing it requires
encoder and serving qualification.

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
of the served checkpoint run on the [GPU reference runner](../tools/refrun/README.md),
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

Each model's declaration names its independently pinned spread variants.
The fixture READMEs retain reference provenance. Per-model results and
effective limits belong to the declarations and their evidence, not a
second copy of the profile table.

### Small categories

A top-one limit at the reference BF16 run's own rate can fail an equally
accurate engine by chance. More positions do not fix a limit placed exactly
at the comparison rate.

The spread rule therefore checks each top-1 limit of a category or media
case for power (variant-spread.v2, adopted on 2026-10-15): a limit that sits less than two binomial standard errors above
the BF16 run's own rate cannot judge the set, and there the metric also
passes when the candidate's flips exceed the BF16 run's on the same
positions by at most two standard deviations of the discordant count, the
paired rule's top-1 clause
([check kinds](check-kinds.md#the-spread-rule-variant-spreadv2)). Limits
with power, every mean limit and every pooled set are judged as before.
Positions within a text are not independent, so the binomial approximation
overstates power. This fallback does not change pooled or mean limits and
does not excuse errors outside its stated rule.

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
two standard deviations of the discordant count.
`verify_exact.py report --control` can compare against several AR draws;
top-one counts still pair with the main AR control. With `--pool-verify`,
the mean of candidate draw errors at each position may exceed the control
mean by at most 0.005. Every draw keeps its own top-one and coverage gate;
single-draw mean gaps are reported. This permits a bucket above the margin
when the pooled mean passes.

MiMo DFlash's pooled mean failed and was waived, not passed
([scope and evidence](serving-profiles.md#accuracy-waiver-2026-10-06)).
The margin remains unchanged for other candidates. That waiver applies to
the pooled mean, not the one-record predicate declared in `qual.json`; it
belongs in the declaration when the pooled gate itself is declared.

Fixed-K3 MTP state runs
and the release and preview declarations keep one AR control and one draw.
The preview keeps exact C1
greedy equality for Qwen MTP; MiMo DFlash uses the `spec_verify.py`
copy-set bounds (mean |dlogprob| <= 0.15, top-1 disagreement <= 0.08) plus
paired GSM8K. See the [speculation contract](numerics.md#state-and-verification-contract).
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

The rule was adopted on 2026-10-15 with the plausibility guard added on
2026-10-16. The per-row error bound can be loose when the BF16 floor row
itself reroutes; the norm/direction guards reject zeroed or implausible rows
that such a bound alone would admit.

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
insufficient. See [debugging](contributing.md#diagnostics) for first-divergence reporting;
Qwen eager synchronized dumps cannot clear a graph-only failure.

A precision option is adopted only with a resolved accuracy benefit against
the reference.

## State and verification contract

Verification prepares K + 1 rows transactionally. Commit keeps only the
accepted prefix and the correction or bonus token; rejected KV, recurrent
state and drafter state are restored before the next step. Streaming, stop
handling and logprobs describe committed target tokens only. Greedy acceptance
compares target token IDs; stochastic acceptance preserves the target
distribution rather than forcing AR text equality. Graph row capacities are
compile-time bounds; unused rows never become samples or committed state.

The MiMo release preview records both speculation and AR with
`tools/qual/spec_verify.py` (16 copy and 16 natural prompts, 64 greedy output
tokens, top-5 logprobs). Over common prefixes, the copy set must keep mean
|dlogprob| <= 0.15 and top-1 disagreement <= 0.08. Both GSM8K arms must score
at least 193/200, lose at most one AR-correct item and drop at most one total
correct answer. C1 greedy differences are informational. See
[the release gates](../tools/qual/README.md).

### Exact verify rows

MiMo verify steps of up to eight rows run one-row AR arithmetic for every
stage; wider verify steps follow their bucket's reduction order. Qwen MTP
K3 at C1 also uses AR's order. These are C1 guarantees, not concurrent batch
independence. The release declarations decide whether a particular profile's
AR/speculative equality is a hard gate or informational.

## KV offload and prefix state

Serving KV pages are opaque bundle-owned layouts. The host allocates them
after weights, workspaces and persistent state, keeping a per-profile reserve
(1 GiB by default) on each rank for the CUDA graphs captured once the pool
exists ([GPU memory](serving-profiles.md#gpu-memory)), zeroes new pages,
performs whole-page copy-on-write on every rank and commits or rolls back the
logical batch. A bundle declaring `rewrites_last_page` also copies shared full
tail pages. Bundle callbacks commit or restore recurrent state. Layout identity
is fixed for the engine lifetime; FP8 quantizes only declared KV regions.

Anchors include logical pages and exact bundle state. A request resumes from
its longest matching token/media/cache-salt prefix ending before its last
prompt token. Capture points are the prompt end, the start of its final
configured prefill chunk, and previously produced shared full pages beyond
its hit when no anchor exists there. Granules round these positions down.
A request resumed within its final chunk needs no extra shared-page anchor.
Prompt-end anchors serve extensions; same-prompt repeats must still prefill
their last token. Prompt-logprob requests cannot skip rows they must report.
Unused anchors are evictable under page pressure.

Offload is disabled by default. Transfers retain page references until all
ranks complete; swap-in restores newly reserved pages before admission.
Resume tries GPU state, host state, then recomputation. A refused capture or
restore may fall back only after confirmed device completion. Host copies
cannot be deduplicated by device allocation identity: an anchor's shared
tail can change without that identity changing. Reusable host prefix caching
(`--kv-offload-prefix on`) remains refused.

The full `kv_offload` check compares offload-off and forced-swap servers in
plain AR and prefix-pressure scenarios. C1 compares 36 prompts' token IDs,
strings, logprob bits, alternatives, finish and usage after exact resumes;
at least 18 must reach a forced swap and qualify. Plain runs permit no
fallback; pressure runs list and exclude capacity fallbacks. Concurrent runs
must complete, transfer both ways, leave the host tier idle and avoid stalls.
Pressure must exercise prefix eviction and, with speculation, verified items.
Concurrent equality and off/off controls are informational. The reference
host arena is 16 GiB; the check does not enable offload by default.

### Context parallelism

Groups contain N consecutive ranks, with N dividing the topology and declared
by the bundle. Physical page j belongs to group rank j % N at local table
index j / N. One logical stripe allocates a local page on every rank and
covers N * page_tokens positions. Replica groups must stay within a KV-head
replica set. The tightest rank bounds capacity; partial stripes still
reserve every rank's page. Group buffers are allocated before the pool.

Prefix discovery uses whole stripes. Prompt/chunk anchors retain the prefill
granule and can end inside a stripe; copy-on-write and offload cover every
rank's share. Hit outputs must equal cold prefill at each group size.
The create tail declares the group size only above one, preserving size-one
wire bytes. Each writer stores only owner-local positions and partial
attention results merge in fixed rank order. A different group size changes
reduction order and needs its own serving qualification.

### Recurrent state

Stable state-slot IDs are host-owned; physical state and speculative history
are bundle-owned. Commit, accepted-prefix commit and cancel must retain or
restore exactly the corresponding rows, including drafter state. Prefix and
preemption snapshots include that state as well as KV. A failed callback
poisons the instance as specified by the bundle ABI, not a retry promise.

## Paired release accuracy

`paired_teacher` compares this run's teacher capture with an accepted capture
of the same configuration and arm, both scored against the same independently
pinned reference, by declared margins. The accepted capture is the one the
[correctness runner](../tools/qual/README.md#correctness-runner) stores: the
check runs once against the last-good store and once against the seed store,
and both rows gate. It supplements the absolute teacher gate, including
reference-spread limits, and the existing C1 exactness checks. It does not
replace either.

The check uses `teacher.score`, `limits.paired` and the existing deterministic
95% text-clustered bootstrap (2,000 resamples). It reports both arms' mean and
maximum absolute logprob error and decisive, near-tie and total top-one counts.
It reports paired error changes, discordant counts and affected text names,
categories and absolute token positions. Pooled results and every category
must pass separately. Near ties use the pinned reference's stored top-k gaps,
including the existing 0.3-nat boundary. Classification never uses the other
serving arm as ground truth.

The [declaration fragment](../tools/qual/templates/paired-teacher.json) keeps
all acceptance numbers in one `paired-teacher-margins/v1` object: the margins,
the reference pin, the approval and its rationale. These are **approved
margins**:

| Rule | Proposed value | Meaning |
|---|---|---|
| Pooled mean error | 0.005 nats | Upper clustered CI95 of candidate-minus-accepted absolute error |
| Category mean error | 0.01 nats | Point increase, without category CI gating |
| Pooled top-one | 0.01 | Upper clustered CI95 of the per-position net added disagreement rate |
| Category top-one harm | alpha 0.05 | Whole-text one-sided sign test, with Holm over all categories |
| Minimum CI clusters | 8 texts | No bootstrap CI is computed or gated below this count |
| Sparse pooled point rules | false | Pooled evidence below the minimum text count is insufficient unless explicitly allowed |
| Tail growth threshold | 0.5 nats | Strictly greater absolute-error increase/decrease at a position |
| Pooled tail allowance | 2 net positions | Increases beyond the threshold minus decreases beyond it |

Category top-one harm sums ordinal severity per text (agreement < near
tie < decisive flip), then uses a one-sided sign test over increasing and
decreasing texts. Ties are excluded. Holm correction rejects only while
p < alpha / remaining categories. Positions within a text are not independent
trials. Sparse pooled evidence is insufficient, even for bit-identical arms,
unless a predeclared `allow_sparse_pooled` policy permits point rules with a
coverage rationale. Categories retain point means and whole-text sign tests.
The declaration owns each profile's adopted margins; GLM uses a 0.015-nat category mean margin and four net tail positions;
its margins must not be replaced by the template's.
A margin change is a reviewed policy change, not a baseline update.

## Activation and calibration

The check selects its margins by the live rank count and effective
speculation mode (`tp{N}-{mode}`); a profile running context parallelism
C > 1 selects `tp{N}-{mode}-cp{C}` when the declaration states it, and its
DCP 1 counterpart's margins otherwise. An unknown profile, or one without
margins, is refused. A single-policy external declaration can still use
`POLICY`.

The runner runs `paired_teacher` twice per arm, as `paired_teacher/last-good`
and `paired_teacher/seed`, each with that store's teacher receipt of the same
configuration and arm as `--accepted` (`{stored_teacher}`) and this run's
teacher receipt, which holds every scored text's served rows, as
`--candidate` (`{teacher}`). A store without a capture of the arm reports "no baseline
yet" as information. The comparison reads the live server's profile and
checkpoint binding, recomputes absolute errors from both arms' served rows,
reports a candidate absolute numerical failure as `fail`, and makes a
failing accepted arm `insufficient`. The paired check's constants cannot
loosen the shipped teacher policy. Last good moves only on a passing
full run of every declared configuration; the seed only through
`baseline reseed` with a recorded reason.

```sh
CHOPINFER_QUAL_CONFIG=models/MODEL/qual.json \
  python3 -B tools/qual/paired_teacher.py \
  --accepted "$STORED_CAPTURE" --candidate "$CANDIDATE_CAPTURE" \
  --url "$CANDIDATE_URL"
```

Complete capture files may be up to 64 MiB and use the shared no-follow JSON
reader; duplicate keys and nonfinite numbers are refused. Receipts carry
token-ID digests, score starts, categories and complete served rows.
Rescoring recomputes both arms and refuses missing, duplicate, reordered or
differently scored texts. To score another pinned set, add `--reference-name
NAME` and a separate `--check-id ID` to the check arguments, with the same
independently adopted reference for both arms, never a newly captured
accepted-server output as the reference.

`tools/qual/paired_calibrate.py --first A --second B --out FILE` validates two
accepted repeat captures, their absolute gates, independent startup nonces
and pinned reference. It scores both repeat directions and records the
maximum nonnegative upper CI95 for pooled mean and top-one, the largest
category point increase and net tail count in `observed_spread`, beside the
reviewed proposal in `margins`; deterministic zero spread does not adopt zero
margins. Refuse repeats that fail the reviewed proposal, including category
harm, rather than loosening it. Repeats verify observed repeatability; the
accepted-only neutral topology comparison supports margins for legitimate
numerics changes. Neither is a population noise bound. If a paired
comparison fails, preserve the evidence and resolve the cause; do not widen
the margin or move a baseline to make it pass.

## Media reference comparisons

### Vision merger normalization

MiMo's vision merger uses RMSNorm over `visual.merger.ln_q.weight` with
epsilon 1e-6. The vendor declares LayerNorm with biases absent from the
checkpoint. Paired F32 reference probes found no overall LayerNorm benefit
and worse OCR results, so RMSNorm remains the served/reference choice.

### Solid-color images

MiMo does not name the color of a uniform image: it answers "white" for solid
colors, and reads a solid red frame as blue, under either merger norm. This is
model behavior, not preprocessing. Normalization is a declared deviation: the
served recipe keeps ImageNet mean and std, which a paired F32 study of 349
probes favored over the checkpoint's CLIP statistics.

### Image resize reference

Images use the checkpoint processor's bicubic resize with a minimum of 8,192
pixels. The served recipe keeps ImageNet mean and std on 0..255 values, where
the checkpoint names CLIP constants, and caps an image at 1,048,576 pixels;
the qualification declaration records the normalization, the minimum and the
cap as explicit deviations from the checkpoint processor. A paired F32 GPU
comparison (2026-10-02) found no significant difference between bicubic and
bilinear, and the 3,136-pixel checkpoint minimum lost accuracy on small
images, so the 8,192 minimum stays. Video keeps its 8,192-pixel minimum and
262,144-pixel per-frame cap, with bicubic resize and ImageNet normalization
as a serving choice (the checkpoint has no video processor configuration).
The image reference is transformers' AutoImageProcessor on the GPU
[reference runner](../tools/refrun/README.md).

### Paired baseline scope

MiMo MTP has no accepted paired arm. GLM AR has no standalone paired teacher
baseline: its accuracy link is the same-run AR/MTP determinism and row
comparisons. Each GLM DCP size uses its own accepted captures. BF16 KV and
five-draft configurations declare no paired check. The template has an empty
approval and null reference; it is a proposal until independently adopted.

### Qwen committed snapshots

Qwen snapshots preserve committed recurrent and PLE state, live raw-key rings
and, with MTP, the drafter frontier and retained target rows. Captures follow
commit and never include speculative checkpoints. Restore targets the slot
assigned at admission. A frontier extending into an earlier shared page may
restore only its own request. In-place recurrent cancellation invalidates
touched slots and retires those requests; a sampling failure instead commits
the step and retires only the failed request. MTP tail rewrites require
copy-on-write even for a full page. Snapshot buffers stay owned until device
completion on every rank.
