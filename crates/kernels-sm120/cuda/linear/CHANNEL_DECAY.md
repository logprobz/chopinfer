# Per-channel-decay delta rule

Kernels for linear-attention layers whose forget gate decays every key
channel of every head separately (contract
`kernel_core::per_channel_decay_delta_rule`, f64 oracle
`kernel_core::per_channel_decay_delta_rule_oracle`). The engine
(`src/linear/channel_decay.rs`) owns the ABI, launch plans and workspace
sizing; its own catalog holds the served shape at four ranks as the
synthetic profile `channel-decay-16x128` (tag `h16x128`): 16 heads of 128,
a 6144-channel convolution over separate q, k and v weights with kernel 4,
the bounded gate at -5 and a gated RMSNorm with eps 1e-5. The device code is
`channel_decay_common.cuh`, `channel_decay_conv.cuh`,
`channel_decay_delta.cuh` and `channel_decay_prefill.cuh`; the norm entry
instantiates the gated delta rule's `norm.cuh` template.

Head dim 64 is declared by the contract and refused by name here. Both gate
forms compile (`Gate` template parameter): the bounded
`lower_bound * sigmoid(exp(A_log) * (f + dt_bias))` and the unbounded
`-exp(A_log) * softplus(f + dt_bias)`; the catalog lists the bounded form,
and the compile test also builds the softplus delta entries.

Capacities: prefill up to 8192 rows and 256 sequences per launch, decode up
to 256 sequences, verify up to 8 rows for each of 256 sequences. Symbols are
`chop_channel_decay_{tag}_<entry>_v1`; the cubin is
`channel_decay_{tag}_sm120`.

| Entry | Contract | Grid (x, y, z) | Block |
|---|---|---|---|
| `conv_prefill` | `recurrent.causal-conv1d-depthwise-silu-f32-weight.prefill.v1` | 24, chunks | 128 |
| `conv_decode` | `...decode.v1` | 24, sequences | 128 |
| `conv_verify` | `...verify.v1`, 8 rows | 24, sequences | 128 |
| `delta_prefill_prepare` | `recurrent.per-channel-decay-delta-rule-bounded-sigmoid.prefill.v1` | 16, aligned chunks | 256 |
| `delta_prefill_scan` | same | 8, 16, sequences | 256 |
| `delta_decode` | `...decode.v1` | 8, 16, sequences | 128 |
| `delta_verify` | `...verify.v1`, 8 rows | 8, 16, sequences | 128 |
| `gated_norm` | `row.rmsnorm-gated-sigmoid.v1` | rows | 128 |

`chunks` is `Bucket::chunks` (64-row chunks from each sequence's start, for
the row-local convolution); `aligned chunks` is `rows / 64` rounded up plus
two per sequence (`channel_decay::aligned_chunks`), which bounds the chunks
that start at absolute multiples of 64. Grids depend only on the prepared
bucket; live row and sequence counts are device scalars, so one plan serves
every graph replay. All entries are PDL launches that release dependents and
then wait before touching any input that a previous kernel may write
(decode and verify read their weights first).

## State pools

States live in pools indexed by physical entry id (i32), never in batch
order:

- convolution: BF16 `[slots, 3, 6144]`, the last three inputs, oldest first;
- delta rule: F32 `[slots, 16, 128 (value), 128 (key)]`, the transpose of
  the reference's `[key][value]` layout.

Prefill reads `slot_ids[s]` when `initial[s]` is nonzero (zero state
otherwise) and writes the final state back to the same slot.
`start_positions[s]` is the absolute position of the sequence's first row:
the delta rule's chunks start at absolute multiples of 64, so the first
chunk of a call that starts off a multiple is short. When `anchor_ids[s]` is
nonnegative and the call ends off a multiple of 64 with one inside it, both
the convolution and the delta rule also write the state as it stands at that
multiple to the anchor entry (the state a call ending there would write).
Decode advances `slot_ids[s]` in place. Verify reads the committed state
from `slot_ids[s]` and writes the state after row r to
`checkpoint_ids[s * 8 + r]`; the source slot is not modified, so a commit of
`kept` rows keeps checkpoint `kept - 1`. Pool rules are the gated delta
rule's ([README](README.md)).

## Operands

- Convolution input `[rows, input_stride]` BF16 (q, k, v columns first),
  F32 weights `weight_q`, `weight_k`, `weight_v` `[2048, 4]` (the BF16
  checkpoint weights widened exactly; a fused weight `[6144, 4]` passes as
  `weight_q`), output `[rows, 6144]` BF16.
- Delta rule `qkv` `[rows, qkv_stride]` BF16 (q, k, v of 16 x 128 each;
  usually the convolution output), gate projection `f` `[rows, f_stride]`
  BF16 (16 x 128), beta logits `b` `[rows, b_stride]` BF16 (16), `A_log`
  `[16]` and `dt_bias` `[2048]` F32, output `[rows, 16, 128]` BF16.
- Gated norm input `[rows, 2048]` BF16, gate `[rows, gate_stride]` BF16,
  weight `[128]` BF16, output `[rows, 2048]` BF16.
- `cu_seqlens` and `cu_rows` hold absolute row offsets. Strides are
  multiples of 4 elements.
- Prefill needs a workspace of `workspace_bytes`: 147,968 bytes per (aligned
  chunk, head), 2.37 MB per chunk at 16 heads, shared by `prepare` and
  `scan` and not preserved between launches.

## Numerics

All arithmetic is F32 with IEEE FMA in a fixed order; no tensor-core
instruction appears (the compile test checks the SASS). Rounding points
follow the contract: the convolution sums its taps from zero oldest first
and rounds once to BF16 after SiLU; q and k divide by
`sqrt(sum of squares + 1e-6)` and q is then scaled by `head_dim^-0.5`
(0.08838835 in F32); beta is `bf16(sigmoid(b))`; only the output rounds to
BF16.

Decode and verify run the recurrence row by row (`S *= exp(g)`,
`u = S k`, `delta = (v - u) beta`, `S += k delta`, `o = S q`), each key
reduction a lane-local FMA chain followed by the butterfly sum, so a verify
row's output and checkpoint equal decoding that row.

The prefill rewrites the recurrence per 64-row chunk with decay taken only
as differences within the chunk: with `G_i` the F32 running sum of g from
the chunk's first row, every factor is `exp(G_i - G_j)` (`j <= i`),
`exp(G_i)` or `exp(G_L - G_j)`, all at most one. At a bound of -5 the
running sum reaches -320 per chunk and with softplus it is unbounded, so
any `exp(-G)` factor would overflow F32.

- `prepare` (one CTA per aligned chunk and head, 99,328 dynamic shared
  bytes) forms the strictly lower `A`, the causal score matrix `P`, solves
  `T = (I - A)^-1` by forward substitution, and stores `W = T (beta k
  exp(G))`, `U = T (beta v)`, `QD = q exp(G)`, `KD = k exp(G_L - G)`, `P`
  and `exp(G_L)`; rows past a short chunk are zero.
- `scan` (16 value rows of a head per CTA, 62,528 dynamic shared bytes)
  walks the chunks: `V' = U - W S^T`, `O = QD S^T + P V'`,
  `S = S diag(exp(G_L)) + V'^T KD`.

Every reduction is one FMA chain from zero over at most 128 terms in
ascending index order, so the 128-K rule holds without splitting and a
row's bits depend only on its chunk and the state the chunk starts from.

Invariance: the convolution prefill keeps a row's bits under any split; the
delta-rule prefill keeps them over calls that end on multiples of 64 in
absolute position; verify entries give each row the output and state that
one-row decode steps give. The gated norm is row-local.

Phase 1 schedule: scalar F32 with operands staged through shared memory;
tensor-core or TMA schedules come later with bitwise evidence against it.

## Tiled prefill schedule

`channel_decay_prefill_tiled.cuh` schedules the same prepare and scan with
the same arithmetic: every stored value is the same F32 chain in the same
order, so the workspace, outputs and states equal the staged stages' bit for
bit, and either stage pairs with either schedule of the other. The entries
(`tiled_entries`, `PrefillSchedule::Tiled`, symbols
`chop_channel_decay_{tag}_delta_prefill_{prepare,scan}_tiled_v1`) take the
staged ABI, grids and workspace; they live in their own cubin
`channel_decay_tiled_{tag}_sm120`, which includes only the tiled header.

| Stage | Dynamic shared bytes | Schedule |
|---|---|---|
| `prepare_tiled` | 99,584 | q, k, beta rows and the running sums of g on separate warps; decayed products in 2 x 2 (i, j) blocks over 16-byte loads (the last 64 pairs single); forward substitution by columns on two warps, four rows at a time, no barrier; W and U in 4 x 4 tiles |
| `scan_tiled` | 98,816 | per chunk three phases with one barrier each; W, QD and P, KD and EGL stream by cp.async into the other of two tiles; V' and O in 2 x 2 (row, value) tiles, the state in 2 x 4 (key, value) tiles |

Shared tiles XOR-permute the 16-byte chunks of each row. kernel-lab
(`tests/channel_decay_tiled.rs`, family `channel_decay_tiled`) compares both
stages with the staged ones bit for bit and builds five test-only mutants of
the header (`TILED_MUTANTS`), which it must catch.

## Streamed prefill schedule

`channel_decay_prefill_streamed.cuh` is a third schedule of the same prepare
and scan, again with the same arithmetic and workspace, so either stage pairs
with any schedule of the other bit for bit. The entries
(`streamed_entries`, `PrefillSchedule::Streamed`, symbols
`chop_channel_decay_{tag}_delta_prefill_{prepare,scan}_streamed_v1`) take
the staged ABI, grids and workspace; the scan runs 128 threads per CTA. They
live in their own cubin `channel_decay_streamed_{tag}_sm120`.

| Stage | Dynamic shared bytes | Schedule |
|---|---|---|
| `prepare_streamed` | 100,608 | the tiled prepare with every input load first and g, the norms and the running sums spread over all warps; the forward substitution on four threads per column, finished elements passed by warp shuffles, steps in quads; W and U over j ranges split at the row blocks |
| `scan_streamed` | 98,816 | four warps; per chunk two key halves (V' and QD S^T on one state load) and two row halves (state update and P V' on one load of V'), each loading its next operands ahead into registers; W and QD stream as 64 x 64 key halves and KD as 32 x 128 row halves through four 16 KB slots |

kernel-lab (`tests/channel_decay_streamed.rs`, family
`channel_decay_streamed`) compares every pairing of the streamed stages with
the staged and tiled ones against the staged pair bit for bit, runs poisoned
replays and six test-only mutants (`STREAMED_MUTANTS`), and times the three
schedules.
