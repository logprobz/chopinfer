# Router-in-norm qualification

This candidate keeps every bit. In DeepSeek V4.1 Flash TP4 prefill the
router projection reads the MoE-input norm's widened F32 rows
(`normed_wide`) through the zero-component path (`sm120::router_prefill`,
`kernels_sm120::f32_projection::zero_prefill`): a prepare launch copies each
row's BF16 bits into component plane 0 and flags rows whose elements are
all exact-F32 tensor operands (`single`), then the 32-row projection runs
one MMA chain per 128-K on flagged rows and the SIMT path on the rest. The
widened rows are the norm's BF16 output widened, so every row is exact and
the prepare's flag is a property of the norm's output alone.

The candidate moves that flag into the norm and removes the prepare:

- the served norm (`row/rmsnorm_prefill.cuh`) with `Certify` also writes
  `certified[row]`, whether every BF16 output element, widened, is a tensor
  operand (`exact_f32::tensor_operand`); its other outputs are unchanged;
- the certified projection (`gemm/exact_f32_certified.cuh`,
  `f32_projection::certified`) stages the norm's BF16 rows directly at
  stride K and takes the same single chain, merges and fold as the zero
  path on certified rows, and the same SIMT path from the widened rows on
  the others. Its grid, block and shared memory are the zero path's.

## Criteria, fixed before measurement

Every GPU check runs with `CHOPINFER_KERNEL_REQUIRE=1` on idle GPUs, prints
its executed cases, and fails on zero executed tests or cases.

1. **Bitwise.** Over the same input rows, norm weight and router weights,
   with every output poisoned first (0xa5 bytes) and the same prior status,
   the candidate arm (certifying norm, certified projection) and the served
   arm (served norm, zero prepare, zero projection) write the same normed
   BF16 rows, widened F32 rows, E4M3 codes, scales, row status and router
   logits, byte for byte over the whole 8192-row capacity (so rows past the
   live count stay poisoned in both).
2. **Flags.** The norm's certificate of each live row equals the served
   prepare's `single` flag and a host model (every widened element a tensor
   operand).
3. **Repeatability.** A poisoned repeat and a graph replay of the candidate
   give the same bytes, and so does one more run after the negative control.
4. **Negative control.** The `certify_all` norm mutation (every row
   certified) must change the logits exactly when a live row is
   uncertified.
5. **Memory safety.** compute-sanitizer memcheck reports zero errors on
   both devices over both patterns at 1, 65 and 2048 live rows, clear
   status.
6. **Coverage.** Synthetic rows of the 8192-row capacity, two patterns:
   `clean` (pseudo-random rows, every 61st from the 11th all zero) and
   `outliers` (also every 37th row from the 3rd with one 1e15 element, so
   the rest normalize below the tensor domain, and every 53rd from the 7th
   with a NaN, which also flags the row's status); two prior status
   patterns (clear; every 97th row from the 5th flagged); 1, 65, 513, 2048,
   8191 and 8192 live rows: 24 cases per worker. Router experts 80 to 95
   are below the tensor domain, so their columns take the SIMT path in both
   arms.
7. **Timing, only after 1 to 5 pass.** Paired graphs (5 rounds of 3
   samples) of the served arm (norm, prepare, projection) against the
   candidate (certifying norm, certified projection) at 8192, 2048 and 512
   rows, clean pattern.

No case list, mutation or reference changes after a result.

## Expected time

Round 8's router lab measured the zero path at 0.603 ms per 8192-row call,
the prepare alone 0.198 ms of it; the round 7 profile charges the MoE-input
norm about 0.45 ms per full layer. The prepare reads the 168 MB widened rows
and writes 84 MB of plane 0; the certificate adds one predicate per element
to the norm, on values already in registers.

Expected, before measurement: the candidate at 0.80 to 0.88 of the served
arm at 8192 rows (about 1.05 ms), a saving of 0.13 to 0.2 ms per full layer,
so about 2.6 to 4 ms per 8192-row chunk over the 20 full layers plus about
0.5 ms in the upper-skip layers. These are estimates; the timing phase
measures them, and a served profile, not this lab timing, would qualify a
per-chunk saving.

Before a job is queued, every phase also runs once on CPU through the job
runner's `--emulate` mode with the frozen test binary: the lab's host
emulation (`tests/sm120/router_norm/emulate.rs`) runs each launch's reads,
writes, flags and paths (not its arithmetic), and the runner's idle check,
phase order and analyzers run as in the job. It is a check of the job, not
evidence for any criterion.
