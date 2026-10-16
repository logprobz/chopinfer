# Linear-attention engine

Kernels for the gated-delta-net layers of hybrid models. The engine
(`src/linear/mod.rs`) owns the ABI, launch plans and workspace sizing; the
Qwen3.8 contract catalogs its entries
(`contract/src/sm120/linear.rs`) for the TP2 and TP4
profiles. Shapes below are the TP4 per-rank ones: 4 key heads and 12 value
heads of 128, a 2560-channel convolution over q, k and v. Other rank counts
divide the 16 key and 48 value heads by the rank count. Each registered
geometry supplies its own head counts.

The supported key and value dimensions are both 128. Value heads must be
an integral multiple of key heads, with ratio 1 through 3 for prefill.
Head dim 256 remains refused: with the current 64-row chunks,
`ScanLayout<128>` uses 100,136 shared bytes and `ScanLayout<256>` would
use 157,480, above SM120's 101,376-byte limit. Prepare uses
`70,176 + 1,280 * group` bytes at dim 128 and
`102,944 + 1,280 * group` at dim 256, so even group 1 at dim 256 needs
104,224 bytes. Supporting dim 256 requires a prefill layout redesign;
widening a decode guard alone cannot produce a served profile.

Capacities: prefill up to 8192 rows and 256 sequences per launch, decode up
to 256 sequences, verify up to 8 rows for each of 256 sequences. Symbols are
`chop_linear_{tag}_<entry>_v1`; the cubin is `linear_qwen38_flash_sm120` at TP4
and `linear_{tag}_sm120` at other rank counts.

| Entry | Contract | Grid (x, y, z) |
|---|---|---|
| `conv_prefill` | `recurrent.causal-conv1d-depthwise-silu.prefill.v1` | 10, chunks |
| `conv_decode` | `...decode.v1` | 10, sequences |
| `conv_verify` | `...verify.v1`, 8 rows | 10, sequences |
| `delta_prefill_prepare`, `delta_prefill_scan` | `recurrent.gated-delta-rule.prefill.v1` | 4, chunks; then 8, 12, sequences |
| `delta_decode` | `...decode.v1` | 8, 12, sequences |
| `delta_verify` | `...verify.v1`, 8 rows | 8, 12, sequences |
| `gated_norm` | `row.rmsnorm-gated-sigmoid.v1` | rows |

`chunks` is `rows / 64` rounded up plus one partial chunk per sequence
(`Bucket::chunks`). Grids depend only on the prepared bucket; live row and
sequence counts are device scalars, so one plan serves every graph replay.
All entries are PDL launches that release dependents and then wait before
touching any input. Refusals: a bucket above the catalog capacity, live counts
above the bucket, decode with other than one row per sequence, and verify rows
above `max_rows` per sequence.

## State pools

States live in pools indexed by physical entry id (i32), never in batch order.
The kernel operand `slot_ids` names those entries, which a bundle may map from
its logical request slots:

- convolution: BF16 `[slots, 3, 2560]`, the last three inputs, oldest first;
- delta rule: F32 `[slots, 12, 128 (value), 128 (key)]`.

Prefill reads `slot_ids[s]` when `initial[s]` is nonzero (zero state
otherwise) and writes the final state back to the same slot. When
`anchor_ids[s]` is nonnegative and the sequence ends in a partial chunk, the
state after its last whole chunk is also written to that pool entry (the state
a sequence ending there would write). Decode advances `slot_ids[s]` in place.
Verify reads the committed state from `slot_ids[s]` and writes the state after
row r to `checkpoint_ids[s * 8 + r]`; the source slot is not modified, so a
commit either keeps checkpoint `kept - 1` as the sequence's slot or copies it
back. With index commits, exchanging the selected entry with the former source
in the writable workspace preserves pool capacity. Every next launch must use
the current source and workspace IDs, including decode, prefill and snapshot
copies. Cancellation leaves ownership unchanged; retirement and reuse must
initialize or restore the mapped entry. Workspace entries must be disjoint from
all committed entries across sequences. A slot or checkpoint may alias the
source slot of its own sequence (each
thread reads its state before writing it), but two sequences must never share
a slot.

## Operands

- Convolution input `[rows, input_stride]` BF16 (q, k, v columns first),
  weight `[2560, 4]` BF16, output `[rows, 2560]` BF16.
- Delta rule `qkv` `[rows, qkv_stride]` BF16 (q 4 x 128, k 4 x 128, v 12 x 128;
  usually the convolution output), `a` and `b` `[rows, gate_stride]` BF16,
  `A_log` and `dt_bias` `[12]` F32, output `[rows, 12, 128]` BF16.
- Gated norm input `[rows, 1536]` BF16, gate `[rows, gate_stride]` BF16,
  weight `[128]` BF16, output `[rows, 1536]` BF16, eps 1e-6.
- `cu_seqlens` and `cu_rows` hold absolute row offsets, so a launch can cover
  any row range of its buffers. Strides are multiples of 8 elements.
- Prefill needs a workspace of `workspace_bytes` (557 KB per chunk at TP4),
  shared by `prepare` and `scan` and not preserved between launches.
- Plans are built for a `Bucket` (rows and sequences, at most the catalog
  capacity); a graph captured for a bucket replays for any live counts
  within it.

## Numerics

All arithmetic is F32 on the decoded BF16 inputs (F32 FMA, or tensor-core
products of exact BF16 operands with F32 accumulation); only the stored
outputs round to BF16. q and k are L2-normalized in F32 with no BF16 round.
The chunked prefill (64-row chunks) rewrites the recurrence exactly:

- `prepare` (one CTA per chunk and key head) forms the Gram matrices on the
  tensor cores (exact BF16 products), then per value head T' = (I + A)^-1
  diag(beta) by forward substitution, the causal decay-weighted score matrix
  P and the per-row scales. T' and P are stored as packed lower triangles.
  The cumulative decay is summed in F64 and kept as an F32 hi + lo pair, so
  `exp(G_i - G_j)` stays accurate for heads that decay by up to e^-500 per
  row (exp(A_log) reaches 158 in the checkpoint). It also stages the chunk's
  K and Q rows (in the scan's padded shared layout) and value columns in the
  workspace.
- `scan` walks the chunks of each sequence with 16 value rows of one head
  per CTA: `U = T'(V - c1 Kr S^T)`, `O = c2 Qr S^T + P U`,
  `S = exp(G_L) S + U^T c3 Kr`. `[Kr; Qr] S^T` and the state update run on
  the tensor cores with S and U split into BF16 hi + lo (error at most 2^-16
  of each element, under 0.09 of the oracle bound); the state lives in mma
  accumulators. Operands arrive by TMA bulk copies that refill each buffer as
  soon as its last reader is done (K and scales double buffered).

Invariance: the convolution prefill keeps a row's bits under any split; the
delta-rule prefill keeps them over splits at whole 64-row chunks. Verify
entries give each row the output and state that one-row decode steps give.
The gated norm is row-local.

Correctness is gated in kernel-lab (`tests/ops/linear.rs`,
`tests/ops/linear_invariance.rs`) against the f64 oracles in
`kernel_core::recurrent_oracle` with the numerical contract bound.
