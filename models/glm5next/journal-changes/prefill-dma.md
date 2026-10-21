# Ordered prefill DMA

`CHOPINFER_GLM5NEXT_PREFILL_DMA=1` selects an experimental transport for
single-request text target prefill at CP1, from 512 rows. The default keeps
the qualified collective. The switch refuses CP greater than one and the
comm epilogue. Draft passes, injected media, partial execution, synchronous
stage diagnostics, short tails and non-phased admissions retain their
existing paths.

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
ordered schedule after the audit. These checks do not establish GPU equality
or speed. Full TP4 token and logprob equality is required before performance
qualification with a served A/A control.
