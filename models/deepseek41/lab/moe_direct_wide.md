# Wide direct MoE verify

Flash TP4 serves verify steps of 65 to 192 rows at the package's 192 verify
rows (32 sequences of up to six rows). Their routed experts ran the generic
route (the exact F32 router projection, then select, count and scatter
into 64-row padded expert ranges), the tile-scale expansion when the banks
are encoded, and the 8192-row gate/up and down expert chains before the
routed sum. The direct queues served only steps of up to 64 rows. With
the `wide_verify` candidate option, they now serve these steps too; the
default package keeps the chains until served step-time acceptance.

The wide direct path keeps the router projection and route launches
unchanged. It then runs five launches: a pack that turns the route
metadata into the queue table and clears the gate tickets, the absorbing
gate/up queue, the direct gate finisher, the two-strip down queue and the
down finisher. These are the backbone bank's 64-row direct entries
compiled at a capacity of 192 rows (`bundle/src/sm120/moe_direct_wide.rs`,
`kernels_sm120::moe::chained::wide`). An expert takes at most one route per
token, so it holds at most the capacity's rows. The absorbing queues take
any count, eight rows a pass, and trap only above the capacity. The 64-row
wide queue still traps above 64 rows, and capacities above 64 never select
it. The kernel crate tests capacities 65, 192, 256 and 384. Every entry
keeps its twin's body, parameters, launch bounds and register count; only
the capacity and symbol change. The 1- to 64-row entries, their locks and
their selection are unchanged.

The arithmetic is that of the direct path. Each 128-K chain starts at +0;
the finishers fold chains by ascending K block with `__fadd_rn`, then apply
the existing clamp, SiLU, route weighting and BF16 epilogue and the fused
E4M3 ceil quantizer (floor 1e-4) after gate/up. Down stays F32, and the down
finisher adds the routes in order, then the shared partial. The queues read
tile-encoded scale banks in place, so no expansion precedes them.

The gate partials take 1,105,920 bytes per row: about 212 MB at 192 rows,
plus 768 bytes of tickets. The instance workspace that the 8192-row chains
already reserve at 192 verify rows is larger, so the workspace does not
grow.

## Bitwise evidence

On 2026-10-31, four GPUs at their 250 W limit ran the complete lowered MoE
programs of both paths: the reference as served before this change, and
the candidate lowered at the same revision through a lab switch that the
`wide_verify` option later replaced. The comparison covered every row
count from 65 to 192, scale widths 0, 1, 2 and 4, and four routing patterns:
every row on the same six experts, rows rotating over 24 experts, routes
dispersed over all 384 experts, and a seeded mix with one expert holding
every row, two holding alternate rows and the rest spread. Rows 65, 128 and
192 ran on all four logical ranks; each other row count ran on one.

Each of the 2,208 cases poisoned all mutable buffers with two patterns. It
replayed live rows (all, half, one, zero, all) serially and through serial
and overlapped graphs. It compared all 23 semantic buffers, from the router
outputs through the route metadata, intermediate codes and scales, shared
expert and final output, including both row-status buffers. A CPU check
recomputed the route metadata. The encoded outputs matched the raw ones,
and a dropped shared join altered the output as the negative control
requires. All 66,240 replays were bitwise equal and saw no Xid. Under
compute-sanitizer memcheck, rows 65, 128, 160 and 192 at every scale width,
with the same-expert and seeded patterns, reported no errors.

## Timing

`moe_wide::timing_gpu` times each graph in rotating reference, candidate,
reference rounds after a 256 MiB write that evicts L2. It reports the
candidate's median ratio to the first reference and the two references'
A/A ratio. Every row count from 65 to 192 ran with every scale width and
pattern, 30 rounds each, one logical rank per GPU, at 250 W on a loaded
host. The candidate was faster in all 2,048 complete-layer cases: the
median ratio was 0.655, the largest 0.944 (191 rows, 24 experts) and the
A/A ratios stayed within 0.9957 to 1.0054. The routed expert stages alone
improved in every case, with a largest ratio of 0.845.

Complete-layer microseconds, reference / wide (ratio), raw scale banks,
median of four ranks:

| Rows | Same six experts | 24 experts | All 384 experts | Seeded mix |
| ---: | ---: | ---: | ---: | ---: |
| 65 | 1148 / 861 (0.754) | 1218 / 934 (0.768) | 4268 / 2192 (0.513) | 2412 / 1244 (0.517) |
| 96 | 1334 / 982 (0.738) | 1339 / 1041 (0.780) | 4504 / 2250 (0.499) | 3053 / 1712 (0.561) |
| 128 | 1404 / 1095 (0.782) | 1381 / 1156 (0.838) | 4575 / 2348 (0.513) | 3438 / 1964 (0.569) |
| 160 | 1444 / 1248 (0.865) | 1472 / 1296 (0.882) | 4652 / 2472 (0.531) | 3789 / 2196 (0.579) |
| 192 | 1525 / 1383 (0.908) | 1538 / 1445 (0.941) | 4729 / 2593 (0.547) | 4100 / 2425 (0.592) |

With encoded banks the reference adds the expansion launch and the wide
queues decode scales in place; the median ratios of the four scale widths
lie between 0.650 and 0.657. The down finisher takes 41 to 107 us against
the routed sum's 12 to 28 us, because it folds partials that the chains
reduce internally. The two-strip down queue alone is up to 3.6% slower
than the chain's down projection when six experts hold every row. The
gate/up gains outweigh both. Over the 40 MoE layers of a step, the median
per-layer saving is 12 ms per step with every row on six experts, 10 ms on
24 experts, 92 ms dispersed over all experts and 64 ms for the seeded mix.
The smallest saving measured was 3.4 ms per step. These are kernel-lab
figures; served step-time acceptance is separate.

After the change moved behind the `wide_verify` option, both arms were
exported under it (`CHOPINFER_MOE_GRAPH_ARM=wide` against `=chains`) and
rerun on 2026-10-31 at 65, 96, 128, 160 and 192 rows, every rank and scale
width, and six routing patterns: the four above, rows drawing six experts
each independently, and four-row sequences sharing four experts. All 480
cases and 14,400 replays were bitwise equal, with no Xid. The option also
makes the router and shared expert cheaper in both arms, so the
complete-layer ratios fell: over 240 timed cases at raw and width-1
scales the median was 0.459 and the largest 0.895 (192 rows, 24
experts), with A/A ratios within 0.9968 to 1.0064.

## Rerunning

Export each arm with the ignored bundle test
`lower::moe::scales_tests::graph_export::export_wide_moe_graphs`, setting
`CHOPINFER_MOE_GRAPH_EXPORT` to a directory outside the repository and
`CHOPINFER_MOE_GRAPH_ROWS` to the ascending row counts.
`CHOPINFER_MOE_GRAPH_ARM=wide` exports the `wide_verify` candidate and
`CHOPINFER_MOE_GRAPH_ARM=chains` the same option with the 8192-row expert
chains in place of every direct bucket, the reference that differs from it
only in the routed experts. `prepare_moe_wide` writes the
packed weights. The lab tests `moe_wide::full_gpu` and
`moe_wide::timing_gpu` read `CHOPINFER_MOE_WIDE_REFERENCE`,
`CHOPINFER_MOE_WIDE_CANDIDATE`, `CHOPINFER_MOE_WIDE_WEIGHTS`,
`CHOPINFER_MOE_WIDE_ROWS`, `CHOPINFER_MOE_WIDE_RANKS` and
`CHOPINFER_MOE_WIDE_PATTERNS`. `full_gpu` also reads
`CHOPINFER_MOE_WIDE_WORKER` and `CHOPINFER_MOE_WIDE_MEMCHECK`;
`timing_gpu` reads `CHOPINFER_MOE_WIDE_SCALE_WIDTHS` and
`CHOPINFER_MOE_WIDE_ROUNDS`. Above 64 rows both refuse a pair unless the
reference launches the 8192-row expert chains and the candidate the wide
direct queues (`recorded::check_wide_pair`); every graph pair behind the
results above passes that check. `moe_job.py` and its `moe_active` tests
predate this path and do not run the wide queues.
