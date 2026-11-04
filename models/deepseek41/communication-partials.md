# BF16 prefill partial communication

Flash TP4 defaults to BF16 partials for prefill only, at both KV
storages. Decode, target verification and draft blocks retain their F32
transport and deferred HC round. The default applies to all prefill row
counts, including short upper-cone intervals and odd final chunks. Other
geometries and rank counts retain F32 partials by default.

BF16 KV storage takes the native prefill partials so that comparing the
two layouts compares their stored KV alone. With F32 partials at BF16 KV,
a native-versus-BF16 comparison mixed the storage with unlike prefill
arithmetic and could not tell a storage defect from the intended rounding.
On 2026-11-04 a GPU diagnostic of 89ea9418ef with this change found BF16
KV bitwise equal to native over the 12-request determinism set (a
2710-token prompt, prefix hits and an image among them), with every text
tap equal on short and 16k prompts; without it, the first differing tap
was `L00.attn.out`, after the all-reduce. BF16 KV create therefore needs
the same resident BF16-partial grids as native. The rounded F32 prefill
gather BF16 KV used before is retired.

Prefill DMA carries the same BF16 partials with
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
| DSpark Markov embedding | BF16, width 256 | Eligible arithmetically, but keep its existing separate F32 transport. The shared phased BF16 entries require whole 1024-element rows. |
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

The BF16-partial contract is distinct from the original F32-input bound.
On 2026-10-20 a CPU diagnostic over 249 decode/verify MoE captures
(18,662,400 elements) changed 7,203,358 words and exceeded that original
bound on 7,195,858 elements when rounding partials to BF16. That experiment
does not qualify BF16 decode or verify, which remain F32. No tolerance was
relaxed. The served prefill path is judged against its BF16-partial oracle
and the unchanged absolute and paired teacher gates.

The model-local lab test `bf16_partials` consumes a frozen directory named
by `CHOPINFER_BF16_PARTIAL_INPUTS`. Its CPU discovery check verifies all 249
input hashes and the exact embedded symbols. The GPU fixture checks eager
calls and graph replays on four ranks, plus 13 full, odd-length, split and
masked row cases up to 8192 rows. The GPU fixture and served accuracy gates passed on 2026-10-20.

The prefill route reuses the BF16 operations and arithmetic qualified by
the BF16-partial qualification. It adds no kernel or stream fork.
All prefill rows use the selected partial format; the retired F32-partial
rounded-output gather began at 512 rows. Fake-host tests cover one-row,
short and odd prefills, F32 decode fusion, draft/verify reductions and
protocol reset ranges. The selected default has passed quiet fixed-KV ABBA qualification.

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
were bitwise equal. The comparison measures the adopted prefill arithmetic.

The prior all-phase paired-teacher evaluation passed for AR and DSpark:
mean error change -0.000914 nats, upper 95-percent bound +0.004565 against
the fixed +0.005 margin. Absolute teacher/chat errors were 0.114723/0.291288
against 0.150/0.300. Teacher scoring uses the same BF16 prefill arithmetic
in both modes, so this evidence covers the prefill-only default. It does
not imply that fewer rounding sites monotonically reduce generated-token
differences. The original-input f64-bound limitation above still applies.
