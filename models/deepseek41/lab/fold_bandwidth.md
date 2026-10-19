# Dense fold bandwidth experiment

This family is a lab candidate. Serving keeps its qualified stream, ticket,
prefetch and split-fold entries. The baseline is the round-five serving
selector, including its exact row buckets and every Flash topology.

The 2026-10-19 quiet-host TP4 profile at 250 W attributes these exclusive
times to each call. Bytes include per-row UE8M0 scales and both groups or
banks where applicable. Bandwidth divides compulsory weight bytes by that
attributed time; it is not a DRAM counter measurement.

| Projection | Weight MB | C1 us | C1 TB/s | C8 us | C8 TB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8192 x 1280 | 10.813 | 15.244 | 0.709 | 17.013 | 0.636 |
| 5120 x 2048 | 10.813 | 14.715 | 0.735 | 17.209 | 0.628 |
| 1024 x 4096, two groups | 8.651 | 12.311 | 0.703 | 14.667 | 0.590 |
| 1280 x 5120 | 6.758 | 11.626 | 0.581 | 14.238 | 0.475 |
| 25600 x 6144 | 162.202 | 126.390 | 1.283 | 149.000 | 1.089 |
| 5120 x 576 | 3.041 | 6.525 | 0.466 | 2.677 | 1.136 |
| 512 x 5120 | 2.703 | 7.785 | 0.347 | 9.539 | 0.283 |
| Paired 576 x 5120 | 6.083 | 10.150 | 0.599 | 12.126 | 0.502 |

Initializers are included. PDL overlap can make an operation's attributed
time smaller than its elapsed time, particularly C8 5120 x 576. These
figures describe the profile's allocation of time to operations.

The measured read reference is 0.881 TB/s cold and 1.758 TB/s prefetched
for 10.8 MB, and 1.483 and 1.806 TB/s for 85 MB. An intervening 85 MB of
pressure lowers the latter to 1.661 TB/s. The largest projection exceeds
the 128 MiB L2 capacity. A 1.8 TB/s floor is useful for comparison, but is
optimistic for a small cold transfer.

Small packed folds already launch 240 to 640 CTAs at C1. Their two or four
folds per CTA pay for padded activation staging, global partial stores,
fences, ticket arrival and ordered reloads. Retained prefetch routes also
pay an initializer. These are source and profile hypotheses, not measured
phase percentages. The candidate tests them with three schedules:

- N8 and N32 tiles retain each completed 128-K fold in registers and add
  folds from zero in ascending K order.
- N8 with four K warps stores each fold separately in CTA-local shared
  memory. The final ordered sum never uses a partition subtotal.
- All schedules keep four independent folds in flight and load immutable
  packed 16-byte weight fragments before the PDL wait. Activation operands
  are loaded directly into registers after the wait.

Each row tile admits activations once. Outputs remain tentative until all
tiles complete. Completion counters are separated by cache line, with one
root arrival per group of 32 CTAs. The last root arrival zeros rejected
rows and resets state without waiting for an unscheduled CTA. State and
diagnostic storage belong to one op. Immutable packed weights may be
shared; mutable state, output and status may not alias another op's storage.
Only entry wrappers use grid dependency instructions; task functions take
virtual CTA coordinates and a caller-owned shared arena.

The pair job covers 1, 2, 4, 6, 8, 16, 32 and 64 rows for all routed
shapes and topologies, plus unchanged 128-row prefill entries. It checks
poisoned outputs, sticky status, changing live counts, repeated graph
replays, delayed admission, a poisoned PDL producer and concurrent
independent ops at 1, 8 and 64 rows. An omitted-fold mutant must fail.
The CPU dry run authenticates every required cubin, including the timing
probe, and exercises the same one-ordinal worker selection as the GPU job.

Cold and weight-prefetched timing arms each alternate old/new order over
five measured rounds. Trace attribution charges overlap once. Routing
candidates require a greater than 3% median win in both cache modes on
both devices for both event and exclusive timings. Ties retain serving.
The table includes compulsory-byte floor ratios and separate stage clocks;
summed CTA clocks are diagnostic work counts, not exclusive wall time.
The prefetched arm does not qualify cross-sublayer look-ahead prefetch.

All 190 compiled exports have zero local and stack bytes. GPU bitwise and
performance qualification remain pending until the staged job runs.
