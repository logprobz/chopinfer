# Wide HC mixer routing decision

The 2026-10-19 pair comparison passed bitwise outputs, live-row and graph
replays, negative controls, next-sublayer prefetch and concurrent scratch
checks on both GPUs. Its timing measurements reject the 160-CTA mixer
at every tested bucket. The correctness pass does not qualify a faster
serving implementation.

Keep the existing three standalone split mixers and 77 rounded two-stage
boundaries per Flash decode step. Do not register either wide entry, its
last-arrival finalizer, or its prefetch path in serving. The decode journal
asserts that no wide entry is launched. TP4 remains at 2,007 launches per
GPU at C1 and C8, in both KV layouts. This job saves zero additional
launches and zero additional milliseconds per serving step.

## Exclusive paired measurements

Each cell is the range across the two GPUs, in microseconds. Each arm
has 128 calls per bucket, discards the first eight, and attributes
exclusive time after sorting kernel starts and advancing the device's
completion frontier.

| Rows | Old standalone | Wide standalone | Qualified rounded pair | Wide rounded |
| --- | --- | --- | --- | --- |
| 1 | 19.731-20.047 | 27.305-27.694 | 18.024-18.539 | 30.118-30.477 |
| 2 | 19.805-20.071 | 27.637-28.008 | 17.827-18.523 | 30.522-30.888 |
| 4 | 19.837-20.135 | 28.126-28.522 | 18.246-18.642 | 35.830-36.265 |
| 8 | 20.033-20.336 | 30.838-31.280 | 18.515-18.825 | 44.233-44.834 |
| 16 | 21.075-21.336 | 34.920-35.394 | 19.583-19.865 | 61.775-62.549 |
| 32 | 24.966-25.249 | 43.967-44.719 | 23.945-24.296 | 96.134-97.754 |

Both means and medians regress in all 24 standalone/rounded comparisons.
Warm 32-call graph timings agree with the exclusive profiles. Cold
single-call comparisons also show no win; one rounded C1 comparison ties
its three-kernel baseline but loses to the qualified pair.

At C1, replacing all 80 standalone mixers would add 0.606-0.612 ms in
the paired component measurements. C8 would add 0.864-0.876 ms. Replacing
the current three standalone and 77 qualified rounded boundaries would
add 0.942-0.954 ms at C1 and 2.013-2.035 ms at C8. These are component
extrapolations, not measured whole-step regressions.

The round 3 standalone mixer budget was 2.076 ms at C1 and 2.111 ms at
C8. It cannot be directly subtracted from an isolated hot-kernel total:
the same-job old standalone measurements are lower than that served
budget. Scaling the round 3 budgets by the paired slowdown ratios gives
about 2.868-2.873 ms at C1 and 3.247-3.250 ms at C8. These are planning
estimates, not served timings. No positive saving against either round 3
budget is established. The previously accepted producer fusion remains
selected; this decision adds no saving to it.

## Prefetch and diagnostics

Prefetch improves the slower wide candidate. With 85 MiB of intervening
reads, standalone prefetch saves 0.730-0.845 us at C1 and 0.371-0.372 us
at C8. Rounded prefetch saves 0.909-0.922 us at C1 and 0.678-0.742 us at
C8. Those gains are smaller than the wide candidate's regressions. The
prefetch experiment compares wide-with-prefetch against wide-without-prefetch;
it does not qualify adding prefetch to the existing faster mixer or
rounded pair. That composition needs a separate bitwise and timing job.

The phase probes show that the finalizer is substantial: at C1/C8 the
staging, RMS and ordered-fold interval takes about 7,500-9,000 cycles,
and the subsequent pre/post and Sinkhorn interval takes about
6,500-7,000 cycles. These are instrumented same-CTA intervals. They
exclude ticket-election time and do not partition total kernel time.
Do not convert them with the receipt's initial idle SM clock or subtract
clock values from different CTAs.

The wider geometry also increases tickets from 20 to 160 per row and
uses 99 registers in the rounded form. The rounded partial stage repeats
expansion in its projection, RMS and lagged-collapse paths. These are
remaining optimization candidates; the measurements do not isolate their
individual costs. Reducing static barrier instructions from 85 to six
was insufficient to make this implementation faster.
