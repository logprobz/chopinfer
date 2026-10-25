# Packed prefill F32 routes

The native Flash TP4 router admits all prefill counts from 320 through 8192
through its existing certified projection. Its weight initialization and
verify-row refusal are unchanged. The router still runs inside the input
norm only at 8192 rows. Default journal pins are unchanged.

On 2026-10-24, a two-device lab checked 1086 counts per worker: every
count from 320 through 640, every 32-row boundary and its neighbors through
8192, seven served packed counts and 64 seeded draws. The clean and outlier
fixtures are shared with the router-in-norm lab, including zero rows, split
rows, NaNs and tiny-weight columns that require SIMT fallback.

All 2172 cases matched the unchanged baseline SIMT kernel byte for byte.
The 17376 full-output comparisons covered eager and captured execution at
full, partial and zero live counts, then a restored full count. Inactive
rows and guards remained poisoned. Both workers detected both device
mutations.

After correctness passed, both devices timed paired complete graphs at
twelve counts. The candidate-to-baseline median ratios ranged from 0.0934
to 0.1820, all below the fixed 0.97 gate. The power cap was 300 W; active
clock samples reached 1792 MHz. Peak worker memory was 1480 MiB, and free
memory returned to its initial value after both phases. There were no
kernel Xids. The combined serving comparison is reported below.

The exact prefill tiles also cover row bands for four Flash projection
shapes. Existing exact wide, blocked and tiled entries keep precedence;
decode, verify and every other shape retain their previous routes.

| Projection (N x K) | Inclusive rows | Feature tile |
|---|---:|---:|
| Compressor, 512 x 5120 | 257..1023 | 32 |
| Compressor, 512 x 5120 | 1024..8192 | 64 |
| Index, 32 x 5120 | 257..8192 | 64 |
| Indexer wk, 128 x 512 | 257..4095 | 32 |
| Indexer wk, 128 x 512 | 4096..8192 | 64 |
| Router, 384 x 5120 | 257..319 | 32 |

On 2026-10-24, four two-device labs compared both feature tiles with the
unchanged SIMT kernel. The router covered every row count in its band.
The other shapes covered every count from 257 through 640, every 32-row
boundary and its neighbors, the served packed counts, seeded draws and
the prospective band endpoints. General F32 and outlier inputs exercised
tensor execution and SIMT fallback. All 14112 cases and 112896 whole-output
comparisons passed, including poisoned eager and graph launches at full,
partial, zero and restored live counts.

Each shape timed twelve row counts. For each prospective band, a tile had
to beat 0.97 at every timed point; the eligible tile with the lower summed
candidate time won. Adjacent bands choosing the same tile were joined.
The selected tiles' candidate-to-baseline ratios ranged from 0.1204 to
0.3434. Measurements used a 300 W cap and an 1800 MHz clock lock. These
kernel timings do not establish request throughput.

CPU dispatch checks cover every row count through 8192, including exact
entry precedence, the unchanged drafter shape and other geometries. The
24 shared route tests and 17 model route/journal tests pass. All eight
default and media journals and their handle maps are byte-identical to
the before-band captures. Each packed AR and DSpark journal replaces 692
baseline launches with the intended prepare/tile pair. The audit checks
the launch geometry, original operands, scratch and initialized weight
domain, with every other line unchanged. No journal pin or kernel lock
changes.

On 2026-10-25, the combined router and tile-band change passed the sampled
serving gate against the preceding batch. Both builds used separate sealed
installs on four RTX PRO 6000 GPUs, TP4, a 300 W cap and an 1800 MHz clock
lock. Sampling used seed 9100, temperature 1, top-p 1, 128 output tokens and
DSpark depth 3 with confidence disabled. A separate untimed baseline pass
preceded eight measured windows in ABBA order, two per arm and concurrency.
All sampled outputs matched exactly, with no preemptions or cache evictions.

| Concurrency | Baseline request tok/s | Candidate request tok/s | Gain |
|---|---:|---:|---:|
| C8 | 347.30 | 371.34 | 6.92% |
| C16 | 379.22 | 415.98 | 9.69% |

These are medians of two windows per arm. The candidate's slower window
exceeded the baseline's faster window at both concurrencies. Energy per
output token fell by 4.74% at C8 and 8.19% at C16 in the same run.

An explicit normal-load prelude passed both builds' C8/C16 controls before
the borrowed run. Every borrowed server matched its own normal-load native
control. All eight rank images stayed read-only under the same keeper
process, with zero uploads, refusals or outstanding leases between servers.
The keeper remained resident afterward.

Packed-versus-alone checks passed separately with normal and borrowed
weights. Each run executed 18 requests, 40 captures and 24 comparisons at
C8/C16, checking output tokens and logprobs, full logits, full state and KV
bytes with only the established unwritten-tail and ring masks. The normal
prelude took 403 seconds and the borrowed gate took 339 seconds; both exited
successfully with no Xids.
