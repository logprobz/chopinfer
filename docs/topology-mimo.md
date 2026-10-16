# MiMo V2.6 Flash at TP2

Dimensions come from the checkpoint's `config.json`, matched to a compiled
entry in `models/mimo26/geometries/supported.json`. Numeric shapes and memory
figures below describe the shipped geometry. Other admitted geometries derive
their own per-rank shapes, catalogs and resources. Topology IDs follow
`<model>-<geometry>-tp<n>`; see the
[same-architecture recipe](adding-a-model.md#same-architecture-checkpoints).

TP4 is the preview deployment. Alpha.2 validates TP4 and TP2; see
[alpha limits](serving-profiles.md#alpha-limits). This page gives the
per-rank shapes, package memory and IDs, and device selection.

## Per-rank shapes

MiMo: hidden 4096, 48 layers (9 GA at `GLOBAL_LAYERS`, 39 SWA), 64 Q heads
(qk 192, v 128), 4 GA and 8 SWA KV heads, 256 experts with intermediate 2048
(MXFP4), dense layer 0 intermediate 16384 (FP8), vocabulary 152,576. The
checkpoint stores `qkv_proj` as four chunks of Q (16 heads), K, V: 3,392 rows
(GA, one KV head) or 3,712 rows (SWA and MTP, two KV heads), with FP8 scales
restarting at each chunk. All splits come from
`kernel_core::topology::TensorParallel` (`split` for heads, rows, columns and
vocabulary; `kv_heads` for KV heads with copies).

| Per rank | TP2 | TP4 |
|---|---:|---:|
| Q heads | 32 | 16 |
| GA KV heads (copies) | 2 | 1 |
| SWA KV heads | 4 | 2 |
| qkv rows GA / SWA | 6,784 / 7,424 | 3,392 / 3,712 |
| qkv scales | `[rows/64, 32]` row groups | `[ceil(rows/128), 32]` blocks |
| qkv scale rows GA / SWA | 106 / 116 | 27 / 29 |
| `o_proj` `[4096, K]` | 4,096 | 2,048 |
| sink `[q]` | 32 | 16 |
| dense layer 0 gate_up rows, scale rows | 16,384, 128 | 8,192, 64 |
| dense layer 0 down K, scale cols | 8,192, 64 | 4,096, 32 |
| expert gate_up `[256, N, 2048 B]` | N 2,048 | 1,024 |
| expert down `[256, 4096, K B]`, scale cols | 512, 32 | 256, 16 |
| `lm_head` rows (vocabulary shard) | 76,288 | 38,144 |
| MTP `eh_proj` K (8192 split) | 4,096 | 2,048 |
| MTP qkv (SWA-like), o, dense MLP | as target SWA and layer 0 | same |
| DFlash `fc` K (20480 split) | 10,240 | 5,120 |
| DFlash Q / KV heads | 32 / 4 | 16 / 2 |
| DFlash qkv rows, `context_kv_proj` rows | 5,120, 5,120 | 2,560, 2,560 |
| DFlash gate_up rows, down K | 16,384, 8,192 | 8,192, 4,096 |
| SWA ring K / V entry bytes | 1,536 / 1,024 | 768 / 512 |
| GA page bytes (BF16) | 737,280 | 368,640 |
| Encoders | replicated | replicated |

A rank's `qkv_proj` rows are its Q heads, then its K heads, then its V heads,
so attention sees the same Q|K|V layout at every TP. TP4 keeps each source
chunk; TP2 rank `r` holds chunks `2r` and `2r+1`.
Because the regroup splits 128-row FP8 scale blocks, TP2 stores scales
per 64-row group (`fp8_f32_row_scale_64x128_v1`), each holding its source
rows' block scale, so results equal the block-scale path. The projection
declaration (`models/mimo26/contract/projections.json`) records this regroup
as the registered source transform `mimo26/qkv-scale-regroup.v1`. The MTP
`eh_proj` `[embedding, hidden]` concatenation is K-split: rank `r` reads the
normalized embedding when `r < R/2` and the hidden state otherwise.

## Device memory

A package's `batch_limits.maximum_sequences` is the largest graph bucket, up
to 256 (32 with MTP), whose KV pool holds an 8192-token average context per
sequence, from a rank's device memory (94 GiB) at the 0.9 KV fraction:

| TP | None | MTP | DFlash |
|---|---|---|---|
| 2 | 64, media 32 | 32 | 32, media 16 |
| 4 | 256 | 32 | 256 |

Its `minimum_device_memory_bytes` is the largest rank's text, drafter and
encoder images, the working set at the package maximum, the media buffers when
media is served, and 4 GiB for modules, sampling, reserve and a small KV pool.
Every topology fits a 96 GB device at its maximum. With media (about 3.3 GiB
more per rank) the TP2 minimum is 92 to 94 GiB. A rank with less
free memory than the minimum still serves with less reserve and a
`startup memory warning`.

## IDs

One package, `mimo.v26.flash.v2`, and one native module serve every rank
count it supports. The manifest declares topologies `mimo-v26-flash-tp2`
and `mimo-v26-flash-tp4`, each with its own KV pages,
ring slot, sequence maximum, graph buckets and minimum device memory
([package topologies](deploy-common.md#package-topology-and-separate-instances)).

## Device selection

`CHOPINFER_TENSOR_PARALLEL_SIZE` (or `--tensor-parallel-size`) selects the
topology, and `--devices` selects and orders visible CUDA ordinals
([device visibility](deploy-common.md#device-visibility-and-rank-selection)).
Startup rejects too few devices before loading weights; check the device
identities each rank logs. Video decoding uses rank 0's device.
Each rank requires one distinct device; duplicate ordinals are refused.
