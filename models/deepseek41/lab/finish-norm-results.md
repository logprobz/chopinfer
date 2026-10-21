# Qualified HC finish and input norm fusion

Both devices in a 250 W pair run on 2026-10-19 passed the fused tail matrix.
Each device checked 88 row/widen/pattern cases, all twelve output roles,
poisoned inactive rows, sticky status and eager/graph replays. The 48 byte
controls, eight arithmetic controls and eight concurrent scratch cases
also passed. Receipts report exclusive devices, no competing compute
processes at start/end and no timing flags. No kernel Xids were recorded.

Select the 1024-thread variant for Flash Decode-1 and Decode-8. It wins
all one-call and 32-call event comparisons on both devices, in both
widening modes. The 512-thread variant is slower and remains unselected.
The following values are gated event medians for 32 repeated calls divided
by 32, with ranges across the two devices. Each comparison includes the
same qualified rounded producer, followed by either the served finisher
and cached32 norm or the fused tail.

| Bucket | Served three-launch chain, us | Selected two-launch chain, us |
| --- | ---: | ---: |
| C1, widening off | 18.176-18.624 | 15.168-15.488 |
| C1, widening on | 19.712-20.288 | 16.768-17.279 |
| C8, widening off | 20.992-21.247 | 17.792-17.984 |
| C8, widening on | 24.128-24.385 | 21.120-21.248 |

Cold and hot exclusive profiles independently agree in every bucket. At
C1 the fused tail alone takes 9.776-9.872 us without widening and
11.470-11.568 us with widening in the hot profile. Its cold counterparts
are 11.151-11.407 and 12.813-13.116 us. Profiles attribute overlapping
intervals once, then sum all chain components before removing warmup.

There are 77 eligible pairs per rank and step: 47 widened and 30 plain.
Their event differences project another 0.229-0.236 ms at C1 and
0.237-0.245 ms at C8 relative to the served rounded producer and cached
norm. These are kernel-chain projections; an integrated serving A/B is
still required for end-to-end latency and full-model output evidence.

Serving retains the exact qualified cubin, including its unselected
512-thread entry. The shared prepared op composes the existing HC and
norm validators, checks their common collapsed/live views and rejects
additional norm roles that alias HC data or workspace. Only qualified
buckets replace the two-entry tail. The model applies this composition
only during decode and only when the preceding reduction admitted the
rounded producer. Initial and Engram boundaries keep standalone norms;
other decode buckets, prefill and verification retain their prior routes.

The producer and workspace do not change. Fixed K folds, both original
RMS trees, Sinkhorn arithmetic, BF16 collapse and norm rounding, optional
widening and whole-row failure outputs retain their tested contracts.
The new CTA task uses 38 registers per thread and 16,688 static shared
bytes, with no reported stack or local memory. Existing kernel locks and
CUDA sources remain unchanged during this serving promotion.

The binding removes 77 launches per rank and step. TP4 native/BF16 KV
counts are 1520/1512 at C1 and 1728/1720 at C8, down from 1597/1589 and
1805/1797. The journal audit contracts only adjacent finish/norm pairs,
preserves all eighteen fused arguments and recomputes graph pieces from
the unchanged atomic-overlap-unit partition rule. Allocation sizes and
stream/event dependencies are
unchanged. Scratch audits cover every router/shared-expert fork, including
rejection controls for overlapping physical ranges.

## Two-CTA tail at Verify-6

The fused tail runs the fold and Sinkhorn on one warp while the other warps
collapse, then the whole CTA runs the cached norm. The norm reads only the
collapse, which uses the previous `pre`, so it need not wait for Sinkhorn.
`sinkhorn_hc::finish_twin` keeps the tail's ABI and every expression and
runs the two halves on two CTAs per row: CTA 0 the mixer half
(`completion::mixer`), CTA 1 the collapse and the unchanged cached norm. The
chain producer is unchanged.

A 250 W pair run on 2026-10-21 compared it, and the one-launch boundaries,
with the served producer plus fused tail on all 154 captured DSpark
Verify-6 boundaries (C1, five drafts and the root). Both devices passed
memcheck, 2464 bitwise reports (eager and two graph replays, served and
flipped widening), 5288 invariance and control checks and 8 mutation
controls, with identical outputs across devices. Paired event timings,
saved us per 77-boundary verify cycle, ranges over the two devices:

| Arm | Cold | Hot |
| --- | ---: | ---: |
| Served against itself | 0 | 85-89 |
| Two-CTA tail, 32 warps | 344-360 | 404-411 |
| Two-CTA tail, 16 warps | 254 | 251-253 |
| One-launch split boundary, 8 warps | 154-157 | 183-193 |

Serve the 32-warp two-CTA tail at Verify-6. The hot comparisons carry about
1.1 us per boundary of arm bias, so the saving is 4.5-4.7 us per boundary,
0.34-0.36 ms per cycle.
