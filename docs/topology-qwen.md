# Qwen3.8 Flash at TP2

Dimensions come from the checkpoint's `config.json`, matched to a compiled
entry in `models/qwen38/geometries/supported.json`. Numeric shapes and memory
figures below describe the shipped geometry. Other admitted geometries derive
their own per-rank shapes, catalogs and resources. Topology IDs follow
`<model>-<geometry>-tp<n>`; see the
[same-architecture recipe](adding-a-model.md#same-architecture-checkpoints).

TP4 is the preview deployment. Alpha.2 validates TP4 and TP2; see
[alpha limits](serving-profiles.md#alpha-limits). This page gives the
per-rank shapes, package memory and IDs, and device selection.

## Per-rank shapes

Qwen3.8 Flash: hidden 2560, 48 layers (12 full attention at `i % 4 == 3`, 36
linear attention), 24 Q heads with an output gate (head 256), 2 KV heads, a
replicated QSA indexer (4 heads of 128), linear attention with 16 key and 48
value heads of 128, 512 routed experts (top 10, intermediate 640) plus a
shared expert of 640, four replicated hyper-connection streams, vocabulary
248,320, an n-gram table of 16 hash heads, one MTP layer placed like a main
full-attention layer, and a replicated vision tower.

All splits come from `kernel_core::topology::TensorParallel`: `split` for
heads, rows, experts and vocabulary, `kv_heads` for the two KV heads with
copies, and `padded_split(640, 128)` for the shared expert.

| Per rank | TP2 | TP4 |
|---|---:|---:|
| Q heads (GQA group) | 12 | 6 |
| `q_proj` rows (query and gate) | 6,144 | 3,072 |
| KV head of rank `r` (copies) | `r` (1) | `r/2` (2) |
| Packed `q|k|v` row (`ATTN_ROW`) | 6,656 | 3,584 |
| `o_proj` K (`ATTN_VALUE`) | 3,072 | 1,536 |
| Linear key / value heads | 8 / 24 | 4 / 12 |
| `in_proj_qkv` rows, conv channels (`LINEAR_QKV`) | 5,120 | 2,560 |
| `in_proj_z` rows, `out_proj` K (`LINEAR_VALUE`) | 3,072 | 1,536 |
| `in_proj_a`, `in_proj_b`, `A_log`, `dt_bias` | 24 | 12 |
| `lin_ab` row stride (a then b, padded to 8) | 48 | 24 |
| Routed experts (expert parallel) | 256 | 128 |
| Shared expert columns live / stored | 320 / 384 | 160 / 256 |
| `lm_head` rows (vocabulary shard) | 124,160 | 62,080 |
| N-gram hash heads | 8 | 4 |
| Conv state per linear layer and slot | 30,720 B | 15,360 B |
| Delta state per linear layer and slot | 1,572,864 B | 786,432 B |
| KV page (BF16, per layer and token) | 1,088 B | 1,088 B |

The last rank keeps the n-gram table's 128-row alignment tail at every TP,
as rank 3 does at TP4. The KV page and the QSA ring do not change with TP
(one KV head per rank at TP2 and TP4).

New kernel entries for TP2 go into cubins whose names carry the TP
(`..._tp2`), so TP4 cubins keep their entries and bytes. Every
build defines each shape constant a header takes, TP4's included, from the
derived rank shape (`model_qwen38_contract::sm120::config::RankShape`).
`tools/dev/qwen38-topology-lab.sh --ranks 2 --gpu N` runs every per-rank
kernel entry on one GPU with sampled oracle columns.

## Device memory

A package's maximum sequences (`package::maximum_sequences`) is the largest
graph bucket, up to 256 (128 with MTP), whose per-rank floor fits a 96 GB
device's 94 GiB. The floor
(`package::minimum_device_memory`, declared per rank) is the largest rank's
device weights (plus the encoder's with media), the bundle's working set at
that count with eight verify rows, and 4 GiB for host arenas, modules,
sampling workspace and reserve, rounded up to GiB.

| Package | TP2 | TP4 |
|---|---|---|
| Text | 64 at 94 GiB | 256 at 77 GiB (MTP: 128 at 69) |
| With media | 32 at 93 GiB | 256 at 79 GiB |

The bundle refuses a create record serving more sequences than its rank fits.
At TP2 the n-gram table lives in pinned host memory
(`package::ngram_in_host_memory`), because with it on the device the floor at
32 sequences would not fit; see
[pinned host memory](deploy-common.md#pinned-host-memory).

## IDs

One package, `qwen38.flash.v2`, and one native module serve every topology
the checkpoint's precision supports: `qwen38-flash-tp2` and `qwen38-flash-tp4`,
each with its own n-gram placement, recurrent state,
sequence maximum, graph buckets, minimum device memory and precision
component. Both mixed NVFP4 layouts serve TP2 and TP4
([package topologies](deploy-common.md#package-topology-and-separate-instances)).

## Device selection

`CHOPINFER_TENSOR_PARALLEL_SIZE` (or `--tensor-parallel-size`) selects the
topology, and `--devices` selects and orders visible CUDA ordinals
([device visibility](deploy-common.md#device-visibility-and-rank-selection)).
Each rank requires one distinct device; duplicate ordinals are refused.
