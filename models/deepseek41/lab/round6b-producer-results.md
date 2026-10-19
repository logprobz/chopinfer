# Rounded producer and cached norm selection

A two-device 250 W run on 2026-10-19 passed all exact comparisons. Per device,
34 rounded HC cases and 192 norm cases covered finite and special values,
changing live rows, poisoned eager/graph replays and inactive guards. The HC
comparison checks every projection chain and RMS lane as well as rounded,
expanded, collapsed, pre, post and combination outputs. The norm comparison
checks normed and optional widened values, codes, scales and full-row status.
All 74 byte controls, 14 arithmetic controls and 20 concurrent scratch cases
passed their expected checks. No Xids were recorded.

The host was busy. Selection uses gated CUDA-event measurements of one-call
and 32-call captured graphs. The table divides each 32-call event interval
by 32; it includes dependent device launch costs. It is kernel-chain timing,
not serving latency. Exclusive profiles are diagnostic only.

| Rounded HC pair | C1 us/call | C8 us/call |
| --- | ---: | ---: |
| Served 20-CTA producer and finisher | 18.752-19.200 | 19.072-19.520 |
| 160 CTAs, four warps, same finisher | 13.440-13.824 | 15.360-15.552 |
| 160 CTAs, eight warps, same finisher | 12.736-13.056 | 16.448-16.768 |

Select eight warps at C1 and four at C8. Both devices agree, and the one-call
event controls also improve. The producer still writes the original
mix-major partials. The finisher, increasing-K fold, expansion rounding,
RMS lane ownership and Sinkhorn arithmetic are unchanged. The earlier
standalone 160-CTA mixer, with its serial global-memory finisher, remains
excluded.

| Input norm | Round 6 tiles, us/call | Cached32, us/call |
| --- | ---: | ---: |
| C1, widening off | 8.832-8.960 | 5.632-5.824 |
| C1, widening on | 8.832-9.023 | 7.232-7.424 |
| C8, widening off | 9.216-9.407 | 5.760-5.824 |
| C8, widening on | 9.344-9.471 | 8.960-9.087 |

Select 32 quantizer warps at C1/C8 in both widening modes. The original
256-lane RMS tree remains unchanged. Codes and scales are computed once,
retained in registers until all groups report flags, then published or
replaced by the original whole-row failure output. The one-call C1 widened
comparison is effectively tied, within 0.032 us; its 32-call result improves
on both devices. C8 with widening is the smallest selected improvement,
about 4.1%. Cached8 regresses relative to tiles. Cached16 loses to Cached32
and regresses at C8 with widening, so neither is routed.

There are 77 rounded pairs and 80 input norms per rank and step, including
48 widened and 32 plain norms. Multiplying the 32-call event differences
projects an additional 0.642-0.650 ms at C1 and 0.415-0.439 ms at C8 relative
to the previous producer selection. Those sums are informational. They do
not include a served measurement or imply that hot repeated inputs match
all layers' cache state. Full serving equality and timing still need an A/B.

Only the measured Flash decode row buckets change. Other buckets, query
norms and prefills retain their previous routes. Kernel launch counts,
workspace sizes, argument lists, allocations and stream ordering remain
unchanged. New module registration retains the exact measured images,
including dormant variants, while the selector permits only the winners.
No CUDA source or existing kernel lock changes in this promotion.

The finisher remains a useful next target after reducing projection latency.
A subsequent exact experiment could produce the lagged collapse in the
rounded producer, then combine cooperative partial staging, warp Sinkhorn
and cached input normalization in the finisher. Preserve both original RMS
trees, BF16 rounding, increasing-K folding and whole-row quantizer status.
That would target the 77 following norm launches as well as the remaining
finisher cost. It is a proposal, not an implemented or measured route.
