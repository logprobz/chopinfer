# Drafter cost

Every speculative step on the `tp4-dspark` profile runs the DSpark drafter
before the target verifies its proposals. The drafter forward is a
three-stage block over five rows per drafting item (`Phase::Verify` with
`5 x items` rows), then the vocabulary head and five sequential Markov
positions, each with its noise, confidence head and sharded argmax. The
twelve-item forward limit runs C1 as one 5-row forward, C8 as one 40-row
forward and C16 as two 40-row forwards (8+8). Nothing here changes
proposals, confidence or committed state: each change keeps its kernels'
arithmetic, or only moves work between entries with equal arithmetic, and
sits behind a candidate option that is off by default.

## Where the time goes

On 2026-10-31 a live `tp4-dspark` server of the base package ran
closed-loop sampled load (temperature 1.0, 256 output tokens, 25 s per
load) under a CUDA trace with graph nodes, on four GPUs held by the job at
their 250 W limit. Each drafter kernel is charged the advance it causes in
its forward's running end, so the charges of a forward sum to its span.
GPU 0, median per step:

| Load | Forwards | Drafter busy | Launches | Largest charges |
| --- | --- | ---: | ---: | --- |
| C1 | 1 x 5 rows | 1.544 ms | 189 | head 0.259, attention 0.232, comm 0.220, Markov chain 0.218, HC 0.161 |
| C8 | 1 x 40 rows | 3.796 ms | 212 | head 1.136, MoE experts 0.579, Markov chain 0.563, comm 0.412, dense 0.366 |
| C16 | 2 x 40 rows | 7.590 ms | 424 | head 2.272, MoE experts 1.145, Markov chain 1.123, comm 0.815, dense 0.732 |

The Markov chain is the Markov step (with its row copy), noise,
confidence head and argmax of the five positions. A forward costs about 1.2 ms plus 0.32 ms per item
(1.54 ms at one item, 5.05 ms at twelve). Under the default tables the
drafter's routed experts (128 experts, top 3) run the direct path only at
5 rows: `moe.direct_admit.verify` admits 1, 2, 4, 5, 6, 8, 16, 32, 48 and
64 rows, so the 40-row forwards of C8 and C16 run the expert chains. Its
HC boundaries never fuse: `successor` stops at the backbone's last layer.

## Candidates

Each option is a package model option (see the
[model options](../README.md#model-options)).

- `dspark_markov_staged` (`bundle/cuda/deepseek41_markov_staged.cuh`,
  `sm120::markov_staged`): a CTA owns 64 logits with one thread per
  (logit, 128-product chain). Each thread loads its chain's 128 BF16
  weights, one contiguous 256-byte run, into registers before the
  dependency wait, then runs the tiled step's chain for up to eight rows
  per pass: products in K order from +0, chains folded left to right from
  +0, then base. It reads each item's base row in place from the
  item-major block logits, five rows apart, so the row copy before the
  tiled step goes. Forwards of two or more items take it; one item keeps
  the row step.
- `dspark_confidence_staged` (`bundle/cuda/deepseek41_confidence_staged.cuh`,
  `sm120::confidence_staged`): one CTA per live row and one thread per
  128-product chain of the concatenated hidden and predecessor-embedding
  operands. Each thread loads its 128 weights into registers before the
  dependency wait; after it the CTA stages the row's operands in shared
  memory with coalesced loads. Each chain runs the parallel entry's
  products in K order from +0, and thread zero folds the chains left to
  right from +0.
- `dspark_hc_fusion` (`bundle/src/lower/hc_expand_mix.rs`,
  `Lowering::hc_successor`):
  a drafter stage's attention boundary takes its MoE as the next sublayer,
  and its MoE boundary the next stage's attention, so where the finish
  route admits the forward's rows they fuse their round, expansion, next
  mixer and input norm as the backbone's do. The shared finish table admits
  35 to 60 drafter rows (7 to 12 items). The boundary into the head keeps
  its chain.
- `dspark_hc_draft_rows` (with `dspark_hc_fusion`): drafter forwards of 5
  to 30 rows admit the same route through the drafter's own ops,
  `hc.draft_admit.verify`, `hc.draft_chain.verify` and
  `hc.draft_cols.verify`, registered only under the option, so the
  backbone's buckets and pins are unchanged.
- `dspark_moe_direct` (`sm120::moe_direct`, `moe.direct_drafter.verify`):
  drafter forwards of 10 to 60 rows, every multiple of 5, run the routed
  experts through the direct path (row-bias route, queue metadata, gate/up
  queues and finisher, down queues and finisher) instead of the route
  select, count and scatter, the expert chains and the routed sum. The
  direct cubins already serve 5 rows; the option adds prepared row counts
  and no cubin.

Together the options take a forward from 189 to 171 launches at one item,
215 to 177 at two to five items and 212 to 174 at six to twelve.

## Bitwise evidence

The kernel lab's pairs ran on each of four GPUs on 2026-10-31:

- `dspark_markov_staged::dspark_markov_staged_gpu_matches_the_tiled_step_and_oracle`:
  the staged step against the tiled step and a CPU emulation of their chain
  at TP1, TP2 and TP4 vocabulary shards, 2, 3, 5, 8, 9, 12 and 32 rows, on
  random, edge, zero and large-magnitude operands, including the aliased
  in-place case the packed heads use. All 2016 cases matched on every GPU.
  At TP4 the staged step took 18 to 27 us at 2 to 8 rows, 27 to 31 us at
  9 and 12 and 61 us at 32, against 76 to 82 us, 121 to 125 us and 215 to
  219 us for the tiled step.
- `dspark_heads::gpu::staged::confidence_staged_gpu_matches_the_served_entry_and_oracle`:
  the staged head against the parallel entry and a CPU emulation, with
  poisoned outputs, dead rows, relaunches, graphs and a PDL producer that
  holds the operands until after the consumer starts, at aligned and
  unaligned strides. All 250 cases matched on every GPU. The staged head
  took 10.2 to 12.3 us against 18.4 to 20.5 us.

The HC and MoE options add no kernel. The fused HC boundary is the
backbone's, whose lab pairs matched the unfused chain bitwise; the direct
MoE entries already serve the drafter's 5-row forwards. The fixed-batch
replays below are their GPU evidence at the drafter's rows.

Fixed-batch GPU replays (runtime-host `fixed-batch replay`) of the head
package with all five options on matched captures of the base package
bitwise in logits, proposals, acceptance and committed state: generated
traces at C1 (five rows per item) and C16 (three and five rows per item),
and version-6 transition traces at C1, C8 and C16, with mixed depths,
accepted drafts and alternating greedy and seeded steps. With
`diagnostic_step_logs` at the debug log level, every drafted block's
proposal IDs and confidence bits (`drafter-confidence.v1`) were equal: 771
records over the six configurations. These traces run drafter forwards of
5 rows (C1) and 40 rows (C8 and C16). Replays of the same transition
schedule at C2 to C7 and C9 to C12, which run the other admitted rows (10
to 35 and 45 to 60), are pending; until they pass, those rows rest on the
structural argument above, and the options must not be adopted.

The journal proof of the commits against the commit before the candidates
was PROVEN with the declaration
[drafter-candidates.json](../journal-changes/drafter-candidates.json):
every existing journal identical and six added lifetimes with the options
on. The route sweep `tp4-dspark-drafter` (the profile with all five options
on) passed at the CPU and full tiers.

## Measured cost

The same live trace with options on (GPU 0, median per step, drafter busy
time):

| Load | Base | Markov staged and HC fusion | All five |
| --- | ---: | ---: | ---: |
| C1 | 1.544 ms | 1.520 ms | 1.400 ms |
| C8 | 3.796 ms | 3.403 ms | 3.307 ms |
| C16 | 7.590 ms | 6.779 ms | 6.614 ms |

At C8 the Markov chain fell from 0.563 to 0.243 ms (0.267 ms from the
staged step, 0.053 ms from the staged confidence head), the HC boundaries
with their norms, quantizers and collectives from 0.738 to 0.612 ms, and
the routed experts with their router and sum from 0.681 to 0.632 ms. At C1
the staged confidence head saved 0.047 ms and the drafter HC rows 0.117
ms; the Markov step keeps its row path and the experts were already
direct.

To cover every row count the tables admit, two more pairs of live runs
with the head package, options off and then on, ran C2, C4, C6 and C12,
then C3, C5, C9, C10 and C11, back to back. Each row below is the load
where that forward size is the common one (7 items: C8 of the first
comparison), single-forward steps, GPU 0, median per step. The last three
columns are the change in the routed experts with their router and sum,
in the HC boundaries with their norms, quantizers and collectives, and in
the Markov chain.

| Items | Rows | Steps (off/on) | Off | On | Change | Experts | HC | Markov | Launches |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 5 | 1485/1478 | 1.544 | 1.400 | -0.144 | +0.001 | -0.117 | -0.047 | 189 to 171 |
| 2 | 10 | 1041/1070 | 2.132 | 1.643 | -0.489 | -0.045 | -0.167 | -0.300 | 215 to 177 |
| 3 | 15 | 723/740 | 2.408 | 1.858 | -0.550 | -0.036 | -0.215 | -0.301 | 215 to 177 |
| 4 | 20 | 626/651 | 2.738 | 2.229 | -0.509 | -0.030 | -0.186 | -0.301 | 215 to 177 |
| 5 | 25 | 606/616 | 3.010 | 2.529 | -0.481 | -0.023 | -0.176 | -0.297 | 215 to 177 |
| 6 | 30 | 418/418 | 3.145 | 2.681 | -0.464 | -0.031 | -0.150 | -0.299 | 212 to 174 |
| 7 | 35 | 100/85 | 3.575 | 3.068 | -0.507 | -0.037 | -0.149 | -0.315 | 212 to 174 |
| 8 | 40 | 346/325 | 3.796 | 3.307 | -0.489 | -0.050 | -0.119 | -0.320 | 212 to 174 |
| 9 | 45 | 325/333 | 4.289 | 3.572 | -0.717 | -0.048 | -0.157 | -0.524 | 212 to 174 |
| 10 | 50 | 321/307 | 4.666 | 3.923 | -0.743 | -0.044 | -0.201 | -0.515 | 212 to 174 |
| 11 | 55 | 274/300 | 4.806 | 4.104 | -0.702 | -0.045 | -0.182 | -0.509 | 212 to 174 |
| 12 | 60 | 323/317 | 5.054 | 4.323 | -0.731 | -0.043 | -0.200 | -0.509 | 212 to 174 |

No admitted row count measured slower, so every table row stays. The
routed experts gain little at any row count (0.02 to 0.05 ms): the direct
path drops the route and chain launches, but its gate/up queue costs
about what the chains' did. Both Markov steps take eight rows a pass; from
nine items the tiled step stages its weights again for the second pass,
while the staged step keeps them in registers. In these
back-to-back pairs the step period fell by about the drafter saving: by
0.58 to 0.66 ms at C2 to C6, 0.69 to 0.78 ms at C9 to C11 and 1.04 ms at
C12.

## Per-step estimates

The drafter runs between the target's verify forwards, so a step
shortens by about the drafter's saving, as the back-to-back pairs show. On
the default profile that is about 0.14 ms of a 13.9 ms C1 step (1.0%),
0.49 ms of a 35.1 ms C8 step (1.4%) and 0.98 ms of a 60.1 ms C16 step
(1.6%). The comparison with the base package's server was not back to
back: its C8 period rose from 35.08 to 35.66 ms while the drafter fell by
0.49 ms and the target held, with the host gaps between GPU work grown
from 1.0 to 2.2 ms; other lanes' builds and tests were running on the host
then.

These are GPU-busy times under a CUDA trace on a shared host, not served
step-time acceptance, which each option still needs before adoption.

## What remains

- The drafter's vocabulary head is now the largest piece: 1.12 ms of a C8
  forward and 2.29 ms of a C16 step, in the cached spans at 40 rows. The
  exact F32 row tiles (`f32_row_tiles`) took the target's head from 1.08 to
  0.40 ms at 48 rows in the lab; the drafter's head belongs to that work.
- The Markov chain's five positions are serial: each runs the peer
  embedding, the Markov step, a noise pass (sampled mode), the confidence
  head, an argmax partial of one CTA per row, an all-gather and the final
  argmax: 35 launches of a 171-launch C1 forward. Folding the noise and the
  argmax partial into the Markov step would remove 10 of them.
- At 40 rows the direct gate/up queue takes 0.27 ms of the routed experts'
  0.56 ms.
