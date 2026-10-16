# Latent attention

Every query head attends over one shared latent row per token, which is
both key and value, and the per-head up projection is absorbed into the
query and the output. The contracts and f64 oracles are kernel-core's
`latent_attention` and `latent_attention_oracle`; the planner and entry
list are `kernels_sm120::latent` (`entries`, `cubin_spec`). Symbols are
`chop_latent_{tag}_{name}_v1`, with `_merge_v1` for a split merge.

A step runs, each kernel launched with PDL (it releases its dependents at
entry and waits before reading the previous kernel's output):

1. **Key absorption** (`head_gemm.cuh`, `absorb_key`): `q_abs_h = W_UK_h^T
   q_nope_h` for every row and head, BF16 out.
2. **Latent KV write** (`kv_write.cuh`): one warp per row stores the row's
   BF16 latent at its position through the sequence's page table, one
   128-element block at a time (four elements per lane, so every width in
   128s partitions exactly).
3. **Attention** (`attention.cuh`; decode and verify `fixed_splits.cuh`)
   over each row's key list, with its split merge for decode and verify.
4. **Value projection** (`head_gemm.cuh`, `absorb_value`): `out_h = W_UV_h
   o_h`, BF16 out.

## Cache

Pages hold `page_size` consecutive tokens at a launch-supplied u64 byte
stride, so pages may share an allocation with other per-token data. BF16
pages are `[page_size][latent]`. FP8 pages hold every token's E4M3 codes
`[page_size][latent]`, then their F32 scales `[page_size][latent / 128]`:
528 bytes per token at latent 512. A block's scale is its maximum
magnitude divided by 448 in F32 (1 for an all-zero block), and each code is
`E4M3(x / scale)` with the F32 division and `cvt.rn.satfinite`, which the
oracle reproduces bit for bit.

## Attention

Grid [rows, heads / 16, splits], `latent / 128` warps. A row reads
entries `[0, budget)` of its `[rows, max_keys]` i32 key list; -1 entries are
holes, and the list alone decides visibility. The list is cut into units
of 32 keys (16 when 32 does not fit shared memory), and split s owns units
`[units s / S, units (s + 1) / S)`. Each unit's latent rows are staged
through two cp.async stages with one page lookup per key; FP8 codes widen
exactly to BF16 and their scales stay beside them.

- **Scores.** Warp w owns latent block w (128 dims). It forms its block's
  chain of eight m16n8k16 MMAs from zero for every key and the CTA's 16
  heads (FP8: times the key's block scale) and stores it in shared memory.
  Every warp then adds the blocks' chains in block order and multiplies by
  the softmax scale, so all warps hold the same F32 scores.
- **Softmax.** Each warp runs the same online softmax: per head the running
  maximum `m`, `p = exp2((s - m) log2 e)`, a per-lane share of `l`, and the
  rescale `exp2((m_old - m_new) log2 e)`. Holes and entries past the budget
  score -inf.
- **PV.** Warp w accumulates `O^T = C^T P^T` for its own 128 columns. Each
  probability is carried as three BF16 terms (FP8: of `p` times the key's
  block scale for that column block), one chain from zero per unit of 32
  keys, added into the rescaled output with an FMA.
- **Output.** With one split the CTA writes BF16 `o / l` (F32 division,
  zeros for an empty list) or the F32 partial `m`, `l`, `o`. Otherwise it
  writes its split partial, and the merge folds the splits in split order
  with the context-parallel merge's arithmetic (`M = max m_s`, `a_s =
  exp(m_s - M)`, FMA sums from zero) into either output.

Prefill takes one split, so a prompt row's sums do not depend on its step.
Every partial has a fixed owner and every fold a fixed order.

## Fixed splits

Decode and verify rows always take 64 splits (`MAX_SPLITS`), so a row's
partition, its partials and their fold do not depend on its bucket or on
the rows beside it. `fixed_splits.cuh` (`attention_fixed_splits`) only
groups the splits into CTAs: grid [rows, heads / 16, groups], where the
planner picks `groups` to fill the SMs twice (`ceil(2 * 188 / (rows * heads
/ 16))`, a power of two up to 64), and CTA z takes splits `[64 z / groups,
64 (z + 1) / groups)` in order. Per split it restarts the online softmax,
runs `attention`'s unit loop and writes that split's partial; empty splits
write the empty partial. The cp.async stages run across split boundaries.
Each split is therefore computed exactly as a one-split-per-CTA launch of
`attention` computes it, which every bucket of up to 8 rows ran before the
partition was fixed, and the merge folds all 64 partials as before. The
workspace holds `rows * 64 * heads * (latent + 2)` F32: 67 MB for 32 rows of
16 heads at latent 512.

## Column merge

`merge_columns.cuh` (`kernels_sm120::latent::merge_columns`, symbols
`..._merge_columns_v1`, cubin `latent_merge_columns_{tag}_sm120`) writes a
decode or verify entry's split merge outputs bit for bit from the same
workspace and ABI. Each output takes the merge's operations on the same
values in the same order: the maximum over the splits in split order (one
thread, as the merge), each split's weight, the denominator and every
numerator column as FMA sums from zero in split order, then `O / L` or the
F32 partial. Only the schedule differs:

- Grid [rows, heads, latent / 128], 128 threads, each thread one column.
  `merge` takes four columns a thread over [rows, heads] CTAs and loads
  each split's value inside its fold, one dependent load per step.
- A thread requests every split's value of its column, and thread s split
  s's maximum and denominator, before any is used; the weights are formed
  one split a thread and the folds read them from shared memory.
- CTA 0 of a row and head writes the partial's maximum and denominator.

The lab family `latent_merge_columns` builds the lab shape's column merges
and three mutants: the last split left out, the numerators folded in
descending split order, and the dependency wait after the reads. Its test
(`crates/kernel-lab/tests/latent_merge_columns.rs`) compares every output
with the split merge's.

## Slice-widened attention

`attention_slices.cuh` (`kernels_sm120::latent::slices`, cubin
`latent_attention_slices_{tag}_sm120`, symbols `..._slices_v1`) writes an
FP8 one-split (prefill) attention entry's outputs bit for bit from the same
ABI and grid. `attention` stages each FP8 unit twice: the codes, then the
whole unit widened to a shared BF16 tile, beside a shared query tile, 91,392
bytes at latent 512, so one 4-warp CTA occupies an SM. Here:

- Each stage holds the codes with their 16-byte sectors swizzled inside
  each 128-byte block (XOR of the key's low three bits), so a lane reading
  its key's sector hits its own banks.
- A warp widens one [32 keys][16 columns] slice of its own block at a time
  into a private 1 KiB buffer and reads it with the same `ldmatrix` forms:
  the eight QK steps, then the eight PV column groups widen again. Every
  MMA sees the operands, chain order and accumulator it sees in
  `attention`.
- The queries load into their B fragments from global memory once.
- Each thread follows one key: it loads the key list entry three units
  ahead and the page table entry two units ahead, so the cp.async of the
  next unit never waits on a dependent load.
- A unit whose keys are all holes or past the budget skips its widening
  and MMAs but keeps `attention`'s state update: with no live key the
  maximum stays and every probability and PV chain is +0, so the
  denominator and each running output take `fmaf(x, correction, +0)` with
  `attention`'s correction (exactly 1, or 0 before any live key). That
  update is not the identity: a running output that underflowed to -0 (a
  tiny negative output times a small correction) becomes +0, as it does in
  `attention`.

Shared memory is 46,336 bytes and the launch bounds ask for two CTAs of 128
threads per SM (255 registers each). The lab family
`latent_attention_slices` compares every output byte with the entry it
replaces.

## Head-batched GEMM

Grid [ceil(rows / 64), N / 64, heads], 4 warps, a 64 x 64 tile of one head.
The head's weight starts at a launch-supplied element stride, so both
halves of an interleaved up projection bind in place: `KByN` reads
`W_UK_h` [nope, latent] transposed through `ldmatrix.trans`, `NByK` reads
`W_UV_h` [v_dim, latent] directly. K streams in 128-wide cp.async chunks,
each one chain of eight MMAs from zero added into the accumulator in K
order; the first weight chunk is requested before the PDL wait.

## Limits

Heads in multiples of 16, latent widths whose tiles fit 101,376 bytes of
shared memory (up to 768), N in 64s and K in 128s for the GEMM. The
contract's declared extensions (an attention sink, tail RoPE with its
inverse, a second FP4 key source) refuse by name in the planner. The
attention kernels use all 255 registers without spilling; the compile test
fails on any spill.

## Testing

`tests/latent_build.rs` builds every entry at the served shape through
kernel-build (sm_120f, exact symbols, no spills, shared memory) and checks
the SASS: BF16 HMMA, LDGSTS, both ldmatrix forms, the FP8 widening and
satfinite packing, and PREEXIT and ACQBULK for PDL. GPU tests live in
kernel-lab (`tests/latent_attention.rs`); its batch-invariance case runs
probe rows alone and among other rows at buckets 1 to 256 and compares
their bytes with the one-split-per-CTA schedule at bucket 8.
