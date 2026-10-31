# DSpark wide drafter forwards

The DSpark drafter forward is lowered as a verify step of five block rows
per drafting item. Without the `wide_draft` option a forward takes at most
twelve items (`dspark_draft_chunk_items`, 60 rows), the widest the
drafter's decode buckets hold. A step of more items splits into near-equal
chunks (16 as 8+8, 32 as 11+11+10), and each chunk runs every drafter
stage: three layers, the 128-expert top-3 MoE, the Markov head and the
exact F32 vocabulary head. With `wide_verify`, C16 and C32 are the
concurrencies where verify gets cheaper per row, so the chunks' fixed cost
becomes a larger share of the step.

## Drafter cost per step

All measurements were taken on 2026-10-31 on a TP4 server with four GPUs
at their enforced 250 W limit, under a CUDA trace with graph nodes. A
closed-loop sampled load (temperature 1.0, 256 output tokens, ignore EOS,
a seed and cache salt per request) ran 25 s each at C8, C16 and C32. Graph
launches that run the Markov head are drafter forwards. The figures below
are rank 0's GPU-busy time per drafting step, as medians over steps of the
named drafter shape; the other ranks were within about 1%.

Before this option, the `tp4-dspark` profile (64 speculative rows, depths
`1:5,8:3,16:3,32:0`) drafted one forward of 8 items at C8 in 3.77 ms of a
35.0 ms step, and two forwards of 8 at C16 in 7.55 ms of a 59.7 ms
step. At C32 it drafts nothing (457 of 481 steps were plain decode). With
`wide_verify` at 128 speculative rows (depths `1:5,8:5,16:5,32:3`) the
drafter took 3.91, 7.43 and 14.07 ms at C8, C16 and C32 (one forward of 8,
8+8 and 11+11+10) of steps of 45.0, 94.2 and 116.7 ms. At 192 speculative
rows (depths `1:5,8:5,16:5,32:5`) it took 3.91, 7.46 and 14.11 ms of steps
of 44.9, 94.3 and 151.6 ms.

At 192 rows the three C32 chunks took 4.86, 4.72 and 4.51 ms. Across the
measured forwards of 6 to 11 items, a forward costs about 1.2 ms plus 0.33
ms per item. The table charges each kernel of those chunks the time it
advances its forward's latest end, so the stages sum to the forwards'
spans.

| Stage | 11+11+10 (ms) |
|---|---|
| Vocabulary head (exact F32 reuse route) | 4.40 |
| Markov head, confidence, argmax, noise | 2.37 |
| MoE experts | 2.08 |
| Communication | 1.38 |
| Dense projections | 1.26 |
| Attention | 1.17 |
| HC | 0.64 |
| Quantize, norm, elementwise | 0.41 |
| MoE router | 0.28 |
| MoE sum | 0.04 |
| Total | 14.02 |

## Bitwise independence of the split

Every drafter op asks for one-row bits (`exact::verify`), so a row's output
does not depend on which rows share its forward. Attention reads each
item's staged window. The Markov head, confidence and argmax run per item.
Sampling noise is indexed by the request's seed and generated index, not
by the chunk, and allocation runs once on the restored full arrays after
every forward. A single forward therefore produces the proposals,
confidence and committed state of the chunks.

The existing `dspark_draft_chunk_items=32` already runs one forward of up
to 160 rows, but the drafter's ops are prepared for the wide verify
buckets only at the target's expert set, so above 64 rows it lowers to the
prefill forms: 8192-row dense folds, the head's split route and the
8192-row expert chains.

## The `wide_draft` option

`wide_draft` (requires `wide_verify`) bounds a drafter forward by the
widest wide verify bucket instead of twelve items
(`Shape::wide_draft_items`): 25 items at 128 verify rows and 32 at 192. A
step of more items still splits into near-equal forwards (32 items at 128
verify rows run 16+16). A forward above 64 block rows folds its dense
projections through the wide folds at the wide verify buckets and projects
the vocabulary in the head's exact F32 row tiles. Its MoE runs the
8192-row expert chains and its HC boundaries the one-CTA mixer. Forwards
of up to twelve items keep their programs, and each item count keeps its
own drafter graph.

Fixed-batch GPU replays with both options, at the five wide verify
configurations (concurrency 16 and 32, four to six rows per item, verify
capacities 80 to 192), matched the previous package bitwise in logits,
proposals, acceptance and committed state. The replays were run on the
option itself and on the option merged with the head tile fix below. The
allocation journal proof is in the
[change declaration](journal-changes/wide-draft.md).

## Rank 1's head tile

The first live run with the option made the drafter slower: 19.8 ms at C16
and 22.6 ms at C32, against 7.5 and 14.1 ms in chunks. Without its stalls
the single forward took about 6.2 and 9.0 ms. The head's row tile took
14.3 ms on rank 1 and 0.6 to 1.0 ms on the other ranks, which wait for it
in the next all-gather. Rank 1's vocabulary shard has one row (token
49423) outside the tile's tensor domain (zero, or an exponent in [97,
158)). That flags one 16-row weight block, and the tile computes a flagged
block with one thread per output. The same stall costs every wide verify
step about 13.9 ms on the target head. A fix of that fallback is part of
the exact F32 projection work, which this package does not yet include.
Merged with that work, the `wide_verify` step period at 192 rows fell from 94.3
to 82.8 ms at C16 and from 151.6 to 136.7 ms at C32.

## Saving per step

Merged with that work, at 192 speculative rows, in the same load
(`wide_verify` alone, then with `wide_draft`):

| Concurrency | Drafter forwards | Drafter (ms) | Step period (ms) | Host step execution (ms) |
|---|---|---|---|---|
| C8 | 8, then 8 | 3.82, then 3.81 | 44.5, then 44.6 | 44.4, then 44.5 |
| C16 | 8+8, then 16 | 7.61, then 6.91 | 82.8, then 81.6 | 82.6, then 81.5 |
| C32 | 11+11+10, then 32 | 14.14, then 9.47 | 136.7, then 132.3 | 136.4, then 132.1 |

The host step execution is the step trace's median for steps of 8 items
at 48 rows, 16 at 96 and 32 at 192. The option saves 0.7 ms of drafter
time and 1.2 ms of step period at C16, and 4.7 ms and 4.3 ms at C32. The
target's GPU-busy time per step changed by at most 0.7 ms.

| Stage | C16, 8+8 | C16, 16 | C32, 11+11+10 | C32, 32 |
|---|---|---|---|---|
| Vocabulary head | 2.27 | 0.67 | 4.50 | 0.98 |
| Markov head, confidence, argmax, noise | 1.13 | 0.75 | 2.40 | 1.24 |
| MoE experts | 1.15 | 2.16 | 2.09 | 2.61 |
| Communication | 0.84 | 1.09 | 1.29 | 1.52 |
| Dense projections | 0.73 | 0.61 | 1.28 | 1.02 |
| Attention | 0.61 | 0.44 | 1.18 | 0.76 |
| HC | 0.40 | 0.90 | 0.65 | 0.96 |
| Quantize, norm, elementwise | 0.25 | 0.15 | 0.41 | 0.18 |
| MoE router | 0.19 | 0.11 | 0.29 | 0.15 |
| MoE sum | 0.02 | 0.01 | 0.04 | 0.02 |
| Total | 7.59 | 6.89 | 14.12 | 9.45 |

The wide forward loses time in two stages. Its MoE expert chains at 8192
rows take 2.16 and 2.61 ms against 1.15 and 2.09 ms for the 64-row stream
chains of the chunks, and its unsplit HC mixer takes 0.90 and 0.96 ms
against 0.40 and 0.65 ms. If those two stages took no longer than in
chunks, the option would save another 1.5 ms at C16 and 0.8 ms at C32. The
target's wide direct MoE queues and wide HC fusion cover only the target's
expert set and boundaries. The drafter head of forwards of up to twelve
items still uses its reuse route (1.14 ms of a C8 forward). This is not
served step-time acceptance.
