# Served prefill attention

Native Flash TP4 has two window layers, eighteen mixed layers with
32-position compressed pages and twenty mixed layers with 64-position
pages. Attention uses BF16 tensor-core QK/PV with fixed source order,
F32 softmax and output accumulation, BF16 probability rounding and the
original one-split sink/merge and RoPE arithmetic.

## Selection and ownership

The paired-row body serves every prefill row count and prompt packing for
window, mixed32 and mixed64. Each row reads its own slot, block table and
key lists. Two rows share a staged tile only when their validity and key
addresses agree or differ by one; otherwise the CTA stages each row alone.
The model validates the original operation's arguments before substituting
its symbol and grid. Other topologies and BF16 KV keep their admitted
reference paths.

Mixed32 and mixed64 use the fused shared-head finish when there is no
prefill-cache range: packed prompts, nonconsecutive or short chunks and
chunks beyond the cache's offsets. Sixteen local heads share one ordered
64-key tile per row in 83,200 dynamic shared bytes. The finish reuses that
tile for the one-split epilogue and writes the final output directly.
Window attention remains paired. F32 partial storage stays reserved for
fallbacks even when the fused entry does not write it.

The decoded BF16 cache requires one eager, consecutive native Flash TP4
chunk of 8192 rows at offsets 0, 8192, 16384 or 24576, without a retained-row
origin. Window layers use it at all four offsets; mixed layers use it after
offset zero. A mixed call whose active physical-page span exceeds 1024 pages
keeps native attention. Page-origin rebasing preserves every original address,
including holes and repeated pages.

Each rank reserves 75,632,640 data bytes plus a 65,536-byte page table before
KV sizing, 72.19 MiB total. The window, compressed and mark regions are
disjoint from KV, activations, collective buffers and side-stream workspace.
Cache producers and consumers remain on the main stream. The next layer
reuses the arena only after its previous consumer; no side stream or peer
copy receives cache buffers.

For cache-range contexts, the fused finish keeps its measured admissions:
native full chunks at offset zero, decoded full chunks at the later three
offsets, 4096-row parts at offsets 0/24576, and mixed64 retained tails from
320 through 2496 in steps of 128 at those two offsets. Other retained sizes
and origins use their qualified paired route. Upper-skip keeps the cache
only in the full-width prefix before the final compressed source; its
retained upper-layer rows preserve their native key and query origins.
See [prefill dependencies](prefill-dependencies.md).

Selection is recorded in the latent schedule table and checked together
with runtime cache, storage, topology and origin gates. The shared-head
normal, finish, profile and empty bodies now use one mode template;
diagnostic modes do not become served choices.

## Measured outcomes

On 2026-10-24, two-device TP4 labs compared twelve packed shapes, odd
boundaries and live-count graph replays against reference attention. Paired
rows matched every F32 partial and BF16 output. Interleaved items at 4-129
rows also passed the f64 oracle. Paired mixed attention took 0.16-0.22 of
the reference time; a fourteen-prompt, 554-token case took 7.45 ms instead
of 35.9 ms per mixed layer.

The same day's sampled DSpark depth-three serving comparison used coding
prompts of 49-938 tokens, temperature 1 and ABBA visits. Paired attention
reduced C8 prefill steps from 1542 to 1306 ms and the two C16 steps from
2548/911 to 2052/747 ms. Request throughput rose 7.4% and 11.5%, with
bitwise-equal outputs. Packed-versus-alone checks matched logits, state,
KV and first-token logprobs, excluding bytes no prefill writes.

Fused shared heads then passed the packed cases and 136 continuation,
absolute-position RoPE and distinct-sink cases bitwise. The fourteen-prompt
case took 4.23 ms per mixed layer, against 7.45 ms paired. Served prefill
steps fell from 1301 to 1264 ms at C8 and from 2043/763 to 2003/751 ms at
C16, with a 1.2% request-throughput gain at each concurrency and identical
sampled outputs and packed-versus-alone state.

Earlier full-chunk cache qualification on 2026-10-19 at 250 W passed 24
comparisons, including missing/aliased pages and poisoned replay. On
2026-10-20, shared-head staging passed 196 bitwise cases and its fused
finish passed 132 cases plus replay, partition and empty-row controls.
At offset 24576 the full mixed32 cache/attention/output chain measured
2.475-2.536 ms per layer and mixed64 2.821-2.882 ms on the two devices.
These are kernel-chain timings, not whole-request latency estimates.
