# Native MoE tiles with a TMA producer (design, not built)

Lab design for the overlap lever that native_tune r1 found (native_tune.md):
gate/up's MMAs alone run at about the MMA floor, but they barely overlap
the loop's other work, and global loads with their issue and barriers are
about 27% of the served time.

## What changes and what does not

Unchanged, so the outputs stay bitwise: every warp's fragments, operands
and scale bytes, the MMA order (per 128-K stage, one chain per fragment
from +0 over the stage's groups, then one `__fadd_rn` into the
accumulator, stages in K order), the epilogue and its rounding, the queue,
the items and their order of K stages. Only how the stage's bytes reach
shared memory, and how warps wait for them, changes.

Today every thread issues the stage's `cp.async` copies (about five 16-byte
copies plus their address arithmetic per thread per stage), then every warp
meets at `cp.async.wait_group` and a CTA-wide `__syncthreads` per stage, so
one slow warp stalls all twelve and the copy issue sits on the MMA warps'
instruction stream.

New feed, per stage slot (3 slots, as now):

- Weights: one 2D TMA tile load per weight half (two for gate/up, one for
  down) from a host tensor map of the bank (U8, rows of K/2 bytes, box 64
  bytes by the tile's columns, 64-byte swizzle, zero fill out of bounds).
  The 64-byte swizzle puts 16-byte chunk q of row w at `q ^ ((w >> 1) & 3)`,
  which should be the served `weight_chunk` (step 1's probe confirms it
  before anything else is built): the weight bytes land where the served
  loop reads them, and down's K tail (288-byte rows) is zero-filled
  as `cp.async` zero-fills it now.
- Activations: one 1D bulk copy per live row (128 bytes; 64 for down's
  last stage), source and destination 16-byte aligned. Gathered rows rule
  out a tensor map. Rows are 144 bytes apart (16 bytes of padding) with
  rows 0 to 3 and 4 to 7 of each 8-row group interleaved
  (`slot = (r & ~7) | ((r & 3) << 1) | ((r >> 2) & 1)`), which keeps a
  warp's 8-byte fragment loads free of bank conflicts without the chunk
  swizzle a contiguous copy cannot do. Padding rows (source < 0) are zeroed
  in every slot when an item starts, as the served copies zero-fill them,
  so NaN checks never see stale bytes.
- Scales: the producer lanes copy each row's scale words with 4-byte
  `cp.async` (the served `copy_scale` logic) and signal with
  `cp.async.mbarrier.arrive.noinc`.
- Synchronization: `full[slot]` (expected arrivals: one `expect_tx` with
  the stage's bytes plus 32 scale-copy lanes) and `empty[slot]` (one
  arrival per warp after its `block`), the library's `TmaPipeline` pattern.
  No per-stage `__syncthreads`.
- Producer: warp 0's lanes, inline. A dedicated producer warpgroup with
  `setmaxnreg` would need the twelve math warps at 160 registers or fewer
  (they use 166 to 168 today); try it only if the inline producer leaves
  waits exposed.

Shared memory per CTA: gate/up r2w6s3 96 KB of stages plus routes and
barriers, about 97 KB (limit 99); down r2w4s3 86 KB; the two-CTA down
r1w4s2 38 KB per CTA.

Second step, same design: the producer takes the next item's ticket early
and issues its first two stages during the current item's epilogue, which
targets down's non-overlapped F32 epilogue (what the two-CTA tile recovers
in part).

## Expected effect

Of the 27% (gate/up) that loads, their issue and barriers cost, the feed
should recover 15 to 20 points: gate/up about 15 to 20% faster, 10 to 13 ms
per chunk on top of lean. Down about 5 to 10% (2 to 4 ms), more with the
cross-item prefetch. These are estimates from the r1 ablations, not
measurements.

## Build and qualification order

1. Probe in the tuning cubin: the 2D swizzled weight load and the per-row
   bulk copies of one stage, byte-compared in shared memory with the
   served `cp.async` layout on captured inputs.
2. A feed policy beside `Served` (the served `cp.async` feed or the TMA
   feed), with the activation row address as the only change to `block`;
   entries for gate/up r2w6s3 and down r2w4s3 and r1w4s2.
3. The native qualification with them as tile variants (bitwise against
   served on every case, prefix live counts, captured layers and ranks),
   memcheck, plus racecheck and synccheck on the memcheck lives.
4. Paired timing against the lean binding.
