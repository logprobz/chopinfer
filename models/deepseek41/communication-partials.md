# BF16 prefill partial communication

Native Flash TP4 defaults to BF16 partials for prefill only. Decode,
target verification and draft blocks retain their F32 transport and deferred
HC round. The default applies to all prefill row counts, including short
upper-cone intervals and odd final chunks. Other geometries, rank counts
and BF16 KV storage retain F32 partials by default.

The prefill route is adopted. The precision switch and its all-phase
experiment are removed. Prefill DMA carries the same BF16 partials with
the same rounding and rank-ordered sum; see
[prefill-communication.md](prefill-communication.md).

The prefill route declares `CommQuery::AllReduce` with dtype BF16. Each rank
rounds its completed F32 partial once to BF16. The transport exchanges BF16,
adds ranks in ascending order from positive F32 zero, then rounds the final
sum once to BF16. It does not change `AllReduceRounded`'s F32-input contract,
local accumulation, trained activation quantizers, or any F32 consumer.
No shared CUDA implementation changes.

| Site | Consumer format | Served behavior |
| --- | --- | --- |
| Attention WO-B | BF16 hidden rows for HC expansion | Round the completed local F32 projection partial once. WO-A is local. |
| Routed MoE and shared expert | BF16 hidden rows for HC expansion | Keep both components F32 through their complete local combine, then round once. There is one collective, not a separate shared-expert collective. |
| Main vocabulary embedding | BF16 embedded rows | Use the same prefill reduction. Finite input partials are already exact BF16 values or zero. |
| Ordinary DSpark draft blocks | BF16 attention and MoE outputs | Keep the existing F32 transport and main protocol state. |
| DSpark Markov embedding | BF16, width 256 | Eligible arithmetically, but keep its existing separate F32 transport. The shared phased BF16 entries require whole 1024-element rows; adding a kernel for this small embedding is outside this experiment. |
| HC mixing, stream sums and final collapse | F32 local intermediates | These are rank-local reductions. Keep F32. |
| DSpark catch-up projection and confidence hidden rows | F32 | Keep F32, even where the projection reuses the reduction partial buffer. |
| Vocabulary gather and global argmax | F32 scores and integer IDs | These are gather or maximum operations. Keep their formats. |
| Indexer, compressor, Engram and vision local sums | Existing local formats | No inter-rank sum is added or changed. |

AR decode, target verification and draft blocks keep their F32 partials
and deferred-round fusion. The prior all-phase experiment is not selectable.

Create admits every shared phased BF16 entry's complete resident grid on
every participating device. Failure refuses the instance. Other
instances neither load its two modules nor allocate its protocol state.
One-shot, two-shot and turns entries share only their own BF16 protocol
state. A quiescent reset drains rank streams before zeroing that state and
again before the next prefill. Decode and verification leave a pending
prefill reset untouched. Ordinary draft blocks alias the main F32
collective state. The separate
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
masked row cases up to 8192 rows. The GPU fixture and served accuracy gates passed on 2026-10-20.

The prefill route reuses the BF16 operations and arithmetic qualified by
the prior all-phase experiment. It adds no kernel or stream fork.
Its small-row prefill selection does not use the 512-row threshold of the
F32-partial rounded-output gather: all prefill rows use the selected partial
format. Fake-host tests cover one-row, short and odd prefills, F32 decode
fusion, draft/verify reductions and protocol reset ranges. The selected default has passed quiet fixed-KV ABBA qualification.

At TP4, native KV and 250 W per GPU, the fixed-KV quiet comparison measured
32K TTFT at 6.9131 seconds with F32 partials and 6.1952 seconds with BF16
prefill partials, a 10.38 percent reduction. AR C8 step time was
20.2446/20.3785 ms and C32 was 45.6899/46.9402 ms. All measured windows
qualified as idle, with no major faults or Xids. Both arms used 178564 KV
pages, identical prompts and seeds, and exactly 128 generated tokens.

The C32 graph union increased 1.1923 ms per step. Routed gate and down
kernels account for 1.1806 ms of that increase; GPU 0 communication rose
only 0.0105 ms. Those kernels iterate over data-dependent expert queues.
Generated tokens differ after the changed prefill state. This attribution,
with unchanged decode commands and F32 collective counts, supports workload
drift rather than a BF16 decode transport cost. It does not claim equal
free-generation latency for every prompt. Same-mode token/logprob repeats
were bitwise equal. These comparisons preceded removal of the precision switch.

The prior all-phase paired-teacher evaluation passed for AR and DSpark:
mean error change -0.000914 nats, upper 95-percent bound +0.004565 against
the fixed +0.005 margin. Absolute teacher/chat errors were 0.114723/0.291288
against 0.150/0.300. Teacher scoring uses the same BF16 prefill arithmetic
in both modes, so this evidence covers the prefill-only default. It does
not imply that fewer rounding sites monotonically reduce generated-token
differences. The original-input f64-bound limitation above still applies.
