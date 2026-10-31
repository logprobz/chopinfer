# Direct MoE at 6 to 64 rows

Flash TP4 serves decode and verify steps of up to 64 rows through the
direct MoE queues (`bundle/src/sm120/moe_direct.rs`). This note measures
their time against the weight-read floor, locates the rest, and records a
bitwise-preserving lab candidate. Nothing here changes serving: any change
to these steps still needs served step-time acceptance.

## What runs

With the served width-1 tile-scale banks, the routed experts of a step run
these launches after the router and route:

| Rows | Gate/up queue | Gate finisher | Down queue | Down finisher |
| ---: | --- | --- | --- | --- |
| 6 | resident8 chain | all loads first | bandwidth direct small | floor |
| 8 | bandwidth lanes small | bandwidth | bandwidth direct small | floor |
| 16 | resident8 chain and wide queue | all loads first | two-strip | floor |
| 32 to 64 | absorbing lanes | bandwidth | two-strip | floor |

A gate item is one expert's 64 output columns over one 128-K block, gate
and up halves together: 8 KiB of codes, read once and reused for each eight
routed rows. An expert has 360 gate items and 200 two-strip down items; the
last of the five down K blocks is half full. Every item stores one F32
chain per routed row and column, and the finishers fold the chains by
ascending K block: the gate finisher then applies the epilogue and the ceil
quantizer, the down finisher adds the routes and the shared partial. Per
rank, an expert's gate/up codes and raw scales are 3.13 MB and its down
codes and scales 1.57 MB. The partials are 184,320 bytes per route for
gate/up and 102,400 for down: 110 MB at 64 rows, 15% of the weight bytes
with verify-like routing and about a quarter at the served step's roughly
90 active experts, each written once and read once.

## Method

`moe_wide::timing_gpu` loads the served graphs exported at 6, 8, 16, 32,
48 and 64 rows into both arms, so every pair carries an in-job A/A
control. Six routing patterns: 0, every row on the same six experts; 1,
rows rotating over 24 experts; 2, routes dispersed over all experts; 3, a
seeded mix with one expert on every row; 4, each row's six experts drawn
independently, as in a decode step of distinct sequences; 5, four-row
sequences sharing four experts and drawing two of their own, as the draft
rows of a verify step. `CHOPINFER_MOE_WIDE_LAUNCHES=1` also times each
routed launch alone. The floor counts each active expert's codes and raw
scales once at 1.61 TB/s, the rate at which the lab's read flush reads a
256 MiB buffer; a device-to-device copy moves 1.49 TB/s (read plus
write). All runs used four GPUs at their 250 W limit on a loaded host,
one logical rank per GPU, on 2026-10-31.

The harness flushed L2 before each sample by writing 256 MiB. That leaves
up to the whole L2 dirty, and every timed sample first writes those lines
back: the served routed experts at 64 rows, verify-like routing, took
751 us after the write flush and 694 us after a read of the same buffer
(`CHOPINFER_MOE_WIDE_FLUSH=read`, which needs the lab cubin). The numbers
below use the read flush unless stated; a served step starts somewhere in
between.

## Where the time goes

Served routed experts, width-1 scales, median of four ranks, times in us.
Active is the count of experts with routes; each launch is timed alone,
and the queue columns add the share of the queue's own weight floor it
reaches. Serial is the graph of the four launches.

| Rows | Pattern | Active | Floor | Gate queue | Down queue | Gate finisher | Down finisher | Serial |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 4 | 35 | 102 | 72 (95%) | 45 (75%) | 10 | 8 | 121 |
| 6 | 5 | 20 | 58 | 45 (86%) | 30 (66%) | 10 | 8 | 78 |
| 8 | 4 | 46.5 | 136 | 110 (82%) | 58 (78%) | 12 | 8 | 174 |
| 8 | 5 | 24 | 70 | 59 (79%) | 34 (69%) | 12 | 8 | 98 |
| 16 | 4 | 89.5 | 261 | 193 (90%) | 114 (76%) | 16 | 11 | 308 |
| 16 | 5 | 47 | 137 | 106 (86%) | 58 (78%) | 16 | 11 | 172 |
| 32 | 4 | 150.5 | 439 | 352 (83%) | 205 (71%) | 28 | 16 | 619 |
| 32 | 5 | 87.5 | 255 | 191 (89%) | 113 (75%) | 28 | 17 | 321 |
| 48 | 4 | 203.5 | 593 | 491 (80%) | 302 (65%) | 39 | 24 | 885 |
| 48 | 5 | 123.5 | 360 | 290 (83%) | 167 (72%) | 39 | 24 | 516 |
| 64 | 4 | 242 | 705 | 616 (76%) | 375 (63%) | 50 | 31 | 1081 |
| 64 | 5 | 156 | 455 | 380 (80%) | 221 (68%) | 50 | 31 | 694 |

The serial time is 1.18 to 1.34 times the floor at 6 to 16 rows and 1.26
to 1.53 times at 32 to 64 rows. By interpolation between the 24-expert and
verify-like patterns, the served 64-row verify step measured with nsys
(10.2, 7.0 and 3.3 ms per step for the gate queue, down queue and
finishers) corresponds to roughly 90 active experts.

- K-block partials. The finishers mostly read the queues' F32 partials
  back from DRAM and fold them: 81 us at 64 rows and 44 us at 32,
  7 to 14% of the routed time. The queues store the same bytes, but the
  gate queue's time does not change when they stay in L2 (below), so the
  writes cost little; the reads do.
- Weight streaming. The gate queue reaches 76 to 95% of its floor and the
  down queue 62 to 78%, both lower at more rows. Gate items read two
  contiguous 4 KiB tiles; down items read 64 bytes from each of 128 rows
  288 bytes apart, and the half-full last K block costs a whole item.
  Issuing the code loads before the tile-scale decode (Early, below)
  changed nothing, so the per-item serial prologue is not the limit.
- Heavy experts. An expert with more than eight rows takes one pass of
  staging, MMAs and stores per eight rows, each a few microseconds of
  latency. With six experts the routed time is 2.2 to 6.9 times the floor
  and with 24 experts 1.3 to 2.0 times.
- Scale decode: raw and width-1 banks time the same (within 2%, width 1
  slightly faster), so decoding tile scales in place costs nothing visible.
- BF16 MMA issue: from one row per expert (192 active experts) to 2.5
  (156), the gate queue's time per active expert grows about 4%, which
  bounds the MMA, staging and store work of a pass.
- Queue imbalance: at 32 to 64 rows with served-like routing each queue
  runs 11 to 58 items per CTA, so the partial last wave idles at most
  about 3% of the grid; with few experts the item count, not balance,
  limits.
- Tickets: one atomic per gate-finisher CTA and none in the queues; not
  measurable here.

## L2-policy lab candidate

`lab/cuda/moe_small_l2.cuh` builds twins of the 32- to 64-row launches
(absorbing gate queue, bandwidth gate finisher, two-strip down queue,
floor down finisher), each with its served entry's ABI, launch bounds and
operands (`lab/sm120/moe_small_l2.rs`, lab lock `moe_small_l2`). Knobs:

- Stream: weight code loads skip L1 and carry an L2 evict_first policy.
  Each weight byte is read once per step.
- Keep: partial stores carry an L2 evict_last policy.
- Retire: a finisher discards each 128-byte partial line once its warp has
  read it (`discard.global.L2`), so the line is never written back and
  stops occupying L2. Every line a finisher reads was written by this
  step's queue and is read by exactly one warp.
- Loads: the gate finisher issues all 80 partial loads before its first
  add, as the served all-loads-first finisher does.
- Fold: one down item walks its five K blocks in order and folds their
  chains with the finisher's adds, storing one value per route and column.
- Early: code loads issue before the tile-scale decode.

Every knob keeps each value and each add of the served path, so the
outputs are the served bits. On 2026-10-31 all eight knob sets of the
locked cubin passed the complete pair against the served graphs
(`moe_wide::full_gpu` with `CHOPINFER_MOE_WIDE_SWAP`) at 6, 8, 16, 32, 48
and 64 rows, every scale width and patterns 0, 3, 4 and 5: 864 cases and
25,920 replays, all bitwise equal across every semantic buffer, with the
dropped-join control detected and no Xid. The swap replaces the down
finisher only beside its down queue; the harness caught a folded finisher
paired with a served queue as a mismatch.

Routed-experts time, candidate over served, median of four GPUs:

| Knob set | 32 rows, 4 / 5 / 1 | 48 rows, 4 / 5 / 1 | 64 rows, 4 / 5 / 1 |
| --- | ---: | ---: | ---: |
| Stream, Keep | 0.969 / 0.971 / 1.009 | 0.957 / 0.944 / 1.000 | 0.945 / 0.935 / 0.976 |
| Stream, Keep, Retire | 0.975 / 0.984 / 1.043 | 0.963 / 0.963 / 1.031 | 0.950 / 0.944 / 1.000 |
| and Loads | 0.963 / 0.973 / 1.020 | 0.954 / 0.955 / 1.039 | 0.947 / 0.937 / 1.000 |
| and Loads, Fold | 0.965 / 0.977 / 1.028 | 0.953 / 0.952 / 1.162 | 0.931 / 0.929 / 1.123 |
| Stream, Keep, Retire, write flush | 0.948 / 0.938 / 0.861 | 0.941 / 0.933 / 0.892 | 0.942 / 0.924 / 0.899 |

Per-case A/A ratios stayed within 0.974 to 1.017 for the routed experts
and 0.982 to 1.010 for the complete layer. With Stream, Keep, Retire and
Loads the complete layer took 0.959 of the served time at 64 rows and
0.972 at 32 rows with verify-like routing. At 6 to 16 rows the twins that
run are neutral (16 rows within 0.99 to 1.02). Fold costs 6 to 11% at 16
rows and 12 to 21% with 24 experts at 48 and 64 rows, because its items
are five times fewer. Early matched Retire alone with served-like routing
and was slower with 24 experts. Keep without Retire leaves up to 70 MB of
evict_last partials in L2 between steps, which the lab does not penalize
and a served step might. The gains come from the finishers
reading partials that stayed in L2; the queues' own times do not change.

## Proposals

1. Qualify Stream, Keep, Retire and Loads as knobs on the served bodies
   and take the 32- to 64-row buckets to served step-time acceptance. Done
   as options (below); acceptance remains.
2. Store the down bank in 64-row by 128-K tiles, as the gate/up bank is,
   so each down item reads two contiguous tiles instead of 128 separate
   64-byte runs. This changes the prepared weight format (below), so it
   is planned, not built.
3. Finish each (expert, strip) in the gate queue once its 40 K-block
   items have arrived, using a per-strip ticket. Built, paired and timed
   (below); it was slower than the same knobs without it in every case
   measured, so the option and its kernels were removed.

## Served-body candidates

Four model options, all off by default, select the candidate kernels for
the backbone bank's 32- to 64-row buckets and the wide verify buckets of
65 to 192 rows on Flash TP4 (`bundle/src/sm120/moe_direct.rs`), from two
knob modules (`routed_moe_direct_knobs`): `moe_direct_stream`,
`moe_direct_keep`, `moe_direct_retire` and `moe_direct_loads` are the
knobs above on the served bodies (`cuda/moe/chained_knobs.cuh`). Off, the
package prepares and launches the served entries, whose cubins did not
change. The four go to served step-time acceptance. Only the
raw and width-1 tile-scale entries exist; with a bank of width 2 or 4 the
candidates expand its scales first and launch the raw entries.

Proposal 3 was measured as a fifth option, `moe_direct_gate_tickets`,
since removed: the gate queue took the gate finisher's outputs and a
per-rank array of strip tickets (one word per expert and
64-column strip, 13,824 bytes, zeroed at create). After each item's last
pass its CTA arrives on the strip's ticket with a release; the arrival
that completed the strip's 40 K blocks acquired, reset the ticket and
finished the strip: one route per warp, each lane folding its four
columns of the gate or up chain by ascending K block from +0 with the
finisher's adds, then the finisher's epilogue, rounding, quantizer,
status flags and per-token tickets. A count past the block count trapped.

### Bitwise twins

`moe_wide::full_gpu` paired each candidate export with the served export
at 16, 32, 48 and 64 rows and with the 8192-row chains at 65, 96, 128,
160 and 192 rows: every routing pattern, every scale width (0, 1, 2 and
4), two poisons, three modes and the live-row changes, each GPU checking
one logical rank, on 2026-10-31.

| Candidate | Cases | Replays | Controls | Result |
| --- | ---: | ---: | --- | --- |
| Stream, Keep, Retire, Loads | 864 | 25,920 | 864 dropped joins | bitwise equal |
| Tickets | 864 | 25,920 | 864 dropped joins, 768 stale tickets | bitwise equal |
| Tickets with Stream, Keep, Retire | 864 | 25,920 | 864 dropped joins, 768 stale tickets | bitwise equal |

Every case compared every semantic buffer and, for the ticket
candidates, the strip tickets, which had to end each launch at zero (16
rows do not take the absorbing path, so they had no ticket controls).
The ticket controls planted a stale arrival on a strip of an expert the
live rows do not route: the queue had to leave it, keep the output
bitwise and end every routed strip at zero, and the zero check had to
report it. A stale arrival on a routed strip lets a later arrival count
past the block count, which traps by design and ends the context. No run
raised an Xid. The ticket candidates, their strip tickets and these
controls were removed with the option.

### Timing

Routed experts, candidate over served, median of four GPUs, read flush
(clean L2) / write flush (dirty L2). The A/A controls of these runs
stayed within 0.986 to 1.015 for the routed experts and 0.983 to 1.004
for the complete layer (case medians). The wide rows compare with the
wide direct path, not the chains.

| Candidate | Rows | Pattern 4 | Pattern 5 | Pattern 1 | Pattern 0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stream, Keep, Retire, Loads | 32 | 0.963 / 0.931 | 0.973 / 0.928 | 1.019 / 0.852 | 1.042 / 0.918 |
| | 48 | 0.951 / 0.930 | 0.955 / 0.931 | 1.035 / 0.889 | 1.058 / 0.943 |
| | 64 | 0.941 / 0.934 | 0.929 / 0.913 | 1.000 / 0.878 | 1.034 / 0.932 |
| | 128 | 0.851 / 0.869 | 0.851 / 0.881 | 0.822 / 0.838 | 0.974 / 0.916 |
| | 192 | 0.922 / 0.916 | 0.877 / 0.910 | 0.845 / 0.842 | 0.890 / 0.873 |
| Tickets | 32 | 1.288 / 1.236 | 1.300 / 1.320 | 1.531 / 1.534 | 2.089 / 1.883 |
| | 64 | 1.230 / 1.214 | 1.299 / 1.269 | 1.868 / 1.696 | 2.211 / 2.083 |
| | 128 | 1.151 / 1.164 | 1.242 / 1.255 | 1.829 / 1.764 | 2.222 / 2.112 |
| | 192 | 1.165 / 1.173 | 1.285 / 1.282 | 1.809 / 1.751 | 2.189 / 2.059 |
| Tickets, Stream, Keep, Retire | 32 | 1.188 / 1.128 | 1.262 / 1.203 | 1.626 / 1.365 | 2.135 / 1.798 |
| | 64 | 1.103 / 1.084 | 1.183 / 1.107 | 1.900 / 1.563 | 2.286 / 2.103 |
| | 128 | 0.987 / 0.982 | 1.047 / 1.051 | 1.631 / 1.532 | 2.127 / 1.984 |
| | 192 | 1.073 / 1.003 | 1.094 / 1.062 | 1.598 / 1.503 | 1.905 / 1.856 |

Per-step estimate: the slowest rank's routed-experts change times 40
layers, in ms, for pattern 4 then pattern 5; the robust set's ranges
cover two runs.

| Candidate | Flush | 32 rows | 48 rows | 64 rows | 128 rows | 192 rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Stream, Keep, Retire, Loads | read | -0.9, -0.4 | -1.8 to -1.9, -0.8 to -0.9 | -2.6 to -2.8, -2.0 | -9.5 to -10.7, -7.5 to -8.6 | -9.0 to -9.9, -9.1 to -9.2 |
| | write | -1.9 to -2.0, -1.0 | -2.1 to -2.6, -1.7 to -1.9 | -3.1 to -3.4, -2.4 to -2.5 | -8.4, -6.7 to -8.6 | -7.0 to -9.9, -8.7 to -9.4 |
| Tickets, Stream, Keep, Retire | read | +4.8, +3.3 | +4.6, +4.8 | +4.8, +4.6 | -0.8, +2.8 | +2.6, +5.2 |
| | write | +3.1, +2.7 | +4.3, +3.6 | +3.5, +2.9 | -1.5, +1.6 | +0.6, +4.3 |

The robust set gains 3 to 7% of the routed experts at 32 to 64 rows with
patterns 4 and 5 (7 to 9% after a dirty L2) and 8 to 15% at 128 and 192
rows, where the partials no longer fit beside the weights in L2. With
six or 24 experts and a clean L2 it costs up to 6% (+0.04 to +0.25 ms per
step). The complete layer took 0.923 to 0.961 of the served time at 64
rows with patterns 4 and 5.

The strip-finishing gate is slower than the robust set in every case. It
is slower than the served path too, except with Stream, Keep and Retire
at 128 rows with dispersed routing (0.98 to 0.99).
Two costs outweigh the gate finisher it removes (49 to 57 us at 64 rows
alone, less in the graph once Keep holds the partials in L2). Every item
now releases its partials and waits for the ticket's atomic before the
next item (up to 92 items per CTA at 64 rows), where the served queue
issues its stores and moves on. And each strip finishes in one CTA: with
64 routes on an expert, 16 per warp, the queue takes 214 us against
105 us for the served queue and finisher. The first version used an
acquire-release arrival, which also invalidated the SM's L1 on every
item, and split each route's K blocks across the four warps with a
barrier per warp: it took 420 us there and 1.15 times the served routed
time with dispersed routing. Finishing in the queue can save at most the
in-graph finisher time, a few percent of the routed experts, so the
option was removed rather than taken to served acceptance.

### Down bank tiles (Proposal 2, deferred)

The down bank's K per rank is 576, four and a half 128-K blocks, and the
row-tiled layouts require K to be a multiple of 128 (`kernel-core`
`precision/layout.rs`), so tiling the down codes needs a new layout with
a 64-row by 64-K tail tile (2 KiB). That is a new prepared weight format:
a new layout id in each rank's index and plan identity, a new prepared
tensor for every MoE layer of every rank, and new tiled builds and locks
of every kernel that reads the down bank (the bandwidth, two-strip,
streamed and tile-scale direct queues, the 8192-row chains that serve 65
to 192 rows by default, the prefill and lean native MoE, and the wide and
knob modules). The byte count stays the same with the narrow tail tile;
padding K to 640 instead would add about 60 MiB per layer and rank, about
2.3 GiB per rank. The down scales are already tile-ordered in the
width-1 encoding, so only the codes move. Plan: add the tail-tile layout
in `kernel-core` with transform tests; prepare it beside the current
layout behind a package option so the default package is unchanged;
build the tiled down queues as twins and pair them in the lab; then
re-prepare the package and relock in one adoption series. The prepared
plan cache re-keys on the layout id; the drafter bank is separate and
unchanged. Deferred as a package-format change.

## Rerunning

Export the served graphs with `CHOPINFER_MOE_GRAPH_ARM` unset (see
[moe_direct_wide.md](moe_direct_wide.md)) at the row counts above. Build
the lab cubin with `build_family lab/moe_small_l2 $OUT`. Then
`moe_wide::full_gpu` and `moe_wide::timing_gpu` take
`CHOPINFER_MOE_WIDE_LAB_CUBIN=$OUT/moe_small_l2_deepseek41_sm120.cubin`,
`CHOPINFER_MOE_WIDE_SWAP` set to a knob set's tag (`l2k`, `l2sk`, `l2skr`,
`l2skrl`, `l2skrf`, `l2skre`, `l2skrlf`, `l2skrlfe`) and, for timing,
`CHOPINFER_MOE_WIDE_FLUSH=read` and `CHOPINFER_MOE_WIDE_LAUNCHES=1`, with
the same served export as both reference and candidate. The option
candidates export with `CHOPINFER_MOE_GRAPH_KNOBS`, a comma list of
`stream`, `keep`, `retire` and `loads`, and pair against the served
export.
