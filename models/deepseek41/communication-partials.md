# BF16 partial communication experiment

`CHOPINFER_DEEPSEEK41_BF16_PARTIALS=1` selects the existing shared phased
BF16 transports for Flash TP4. Unset, `0`, and invalid values retain the
F32-partial path. This changes bits and remains unqualified and off by
default. Explicit prefill DMA must be off: its F32 adapter bypasses the
partial-rounding operation, so selecting both is refused at create.

The candidate declares `CommQuery::AllReduce` with dtype BF16. Each rank
rounds its completed F32 partial once to BF16. The transport exchanges BF16,
adds ranks in ascending order from positive F32 zero, then rounds the final
sum once to BF16. It does not change `AllReduceRounded`'s F32-input contract,
local accumulation, trained activation quantizers, or any F32 consumer.
No shared CUDA implementation changes.

| Site | Consumer format | Candidate behavior |
| --- | --- | --- |
| Attention WO-B | BF16 hidden rows for HC expansion | Round the completed local F32 projection partial once. WO-A is local. |
| Routed MoE and shared expert | BF16 hidden rows for HC expansion | Keep both components F32 through their complete local combine, then round once. There is one collective, not a separate shared-expert collective. |
| Main vocabulary embedding | BF16 embedded rows | Use the same candidate. Finite input partials are already exact BF16 values or zero. |
| Ordinary DSpark draft blocks | BF16 attention and MoE outputs | Reuse the same candidate operations and main protocol state. |
| DSpark Markov embedding | BF16, width 256 | Eligible arithmetically, but keep its existing separate F32 transport. The shared phased BF16 entries require whole 1024-element rows; adding a kernel for this small embedding is outside this experiment. |
| HC mixing, stream sums and final collapse | F32 local intermediates | These are rank-local reductions. Keep F32. |
| DSpark catch-up projection and confidence hidden rows | F32 | Keep F32, even where the projection reuses the reduction partial buffer. |
| Vocabulary gather and global argmax | F32 scores and integer IDs | These are gather or maximum operations. Keep their formats. |
| Indexer, compressor, Engram and vision local sums | Existing local formats | No inter-rank sum is added or changed. |

The same boundaries apply to prefill, AR decode, target verification and
draft blocks. Decode's deferred F32 carrier explicitly rounds to BF16 before
HC expansion. The BF16-partial backend returns an ordinary BF16 output and
does not select that deferred-round fusion. This can affect decode speed
and must be measured with the served candidate.

Create admits every shared phased BF16 entry's complete resident grid on
every participating device. Failure refuses the candidate. Unselected
instances neither load its two modules nor allocate its protocol state.
One-shot, two-shot and turns entries share only their own BF16 protocol
state. A quiescent reset drains rank streams before zeroing that state and
again before replay; it applies to decode and verification as well as
prefill. Ordinary draft blocks alias main collective state. The separate
Markov embedding state is excluded. No stream fork is added, and existing
prefill fork/join workspace ownership remains unchanged.

On 2026-10-20, a CPU simulation on 249 authenticated decode/verify MoE
captures covered 18,662,400 reduced elements. The current F32-partial fold
had zero violations of its original-input f64 bound. BF16 partials changed
7,203,358 output words and exceeded that original bound on 7,195,858
elements. The original bound is not relaxed or redefined.

| Absolute error | Current against original f64 | BF16 partials against original f64 | BF16 partials against current |
| --- | ---: | ---: | ---: |
| Mean | 0.000736562 | 0.001068301 | 0.000746008 |
| RMS | 0.004082401 | 0.005137420 | 0.006232598 |
| Median | 0.000167981 | 0.000292830 | 0 |
| p99 | 0.011926177 | 0.016012669 | 0.012695313 |
| p99.9 | 0.047329187 | 0.056774139 | 0.0625 |
| Maximum | 3.982856750 | 4.017143250 | 8 |

These are earlier diagnostic MoE inputs, not current served outputs,
attention/prefill capture coverage, or an accuracy qualification. The
changed-input contract is tested separately against its exact BF16-partial
oracle. The staged experiment also reports original-input f64 errors,
requires row and chunk invariance, counts served A/B token and logprob
differences, and keeps the absolute teacher/chat and paired-teacher gates
unchanged. A result outside the original F32-input bound must not be
described as passing that bound.

The model-local lab test `bf16_partials` consumes a frozen directory named
by `CHOPINFER_BF16_PARTIAL_INPUTS`. Its CPU discovery check verifies all 249
input hashes and the exact embedded symbols. The GPU fixture checks eager
calls and graph replays on four ranks, plus 13 full, odd-length, split and
masked row cases up to 8192 rows. GPU evidence and served accuracy results
are pending; this experiment is not a default change.
