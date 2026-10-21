# Ordered prefill DMA

Ordered DMA is the default for the official block-FP8 checkpoint layout
with FP8 MTP KV at CP1, without the comm epilogue. It serves single-request
text target prefill from 512 rows. Other configurations retain their existing
transport. `CHOPINFER_GLM5NEXT_PREFILL_DMA=0` disables DMA; `1` opts in and
still refuses CP greater than one and the comm epilogue. Draft passes,
injected media, partial execution, synchronous stage diagnostics, short
tails and non-phased admissions retain their existing paths.

The transport replaces exactly the 42 target MoE F32 collective launches.
Each owner receives peer partials in source-rank slots, adds them in the
existing F32 rank order from positive zero, rounds once to BF16, and copies
its output interval to peers. The owner sum calls the existing shared
`phased_f32::sum_block<uint16_t>` implementation. It adds one separately
pinned cubin family; existing cubins and lock records do not change.

Producer, owner-sum and reuse joins order the three scatter permutations,
local sums and three gather permutations. Partial and reduced buffers have
private allocation-owner streams, separate from active compute and side
streams. Each rank also owns a 128 MiB inbox. Active streams drain before
release, and partial-copy or drain failures poison the instance. There is
no repartitioning of KDA, convolution, indexer or KV state. Eligible repeated
prefills use the eager transport instead of the existing prefill graph cache.

The CPU audit checks odd and full intervals, exact copy coverage and bytes,
unchanged noncollective launch order, private ownership, repeated chunks,
MTP target selection, short/decode fallback and terminal failure handling.
Vector clocks check cross-stream accesses using recorded event dependencies;
removing waits must fail that audit. The 513-row launch journal pins this
ordered schedule after the audit. A separate MTP test compares the automatic
default's complete program with the qualified explicit opt-in and checks
its cross-stream ordering. Existing journal fixtures explicitly select the
legacy transport; their pins do not change for this policy change.
The DMA journal was separately repinned after omitting the unrelated
active-down module and symbol: 20 later buffer handles move by eight.
Comparing all 12817 lines showed only those bijective identity changes,
on 2788 launch lines. Kernel names, geometry, upload hashes, scalar and
TMA arguments, buffer offsets and lengths, and stream/event order match.

On 2026-10-21, TP4 at 250 W per GPU passed token-ID and top-five-logprob bit
equality for initial and repeated 8K/32K prefills. A quiet-host ABBA repeated
the equality checks across both orders and both same-build repeats, with
48 requests, 2048-token chunks, 2048 KV pages and zero prefix reuse. Each
size supplied ten measured samples per build after reporting each arm's
first request separately. Median TTFT changed from 1176.280 to 1138.730 ms
at 8K (3.19% faster) and 4554.776 to 4443.943 ms at 32K (2.43% faster).
Both orders improved; A/A drift was 0.65% and 2.17%. Every measured DMA
sample was faster than every baseline sample of the same size. There were
no major page faults or Xid reports. This evidence covers the default MTP
text-prefill path, not AR, BF16 KV, other checkpoint layouts or decode speed.
