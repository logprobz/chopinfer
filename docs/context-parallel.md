# Context-parallel attention

Context parallelism (DCP) splits each sequence's KV pages across a group of
N tensor-parallel ranks. Attention heads stay split across ranks as without
it; each rank attends every head of its group over the keys it holds, and the
partial results merge in a fixed rank order. KV capacity then grows with N
where the KV would otherwise be replicated on every rank: a latent cache that
all heads share, or KV heads copied because they are fewer than the ranks.

The host, the KV pool and the kernels described here are model-neutral. A
bundle opts in per topology (see [How a bundle opts in](#how-a-bundle-opts-in)).
Two bundles serve sizes above 1: `glm5next.v1` at 2 and 4, and
`qwen38.flash.v2` at 2 at TP4, where pairs of ranks replicate a KV head. Qwen3.8's
DCP 2 gate (then `qual.dcp2.json`, now the `dcp2` configuration) passed on
2026-10-07; at 32 sequences a group
holds 2.05 times the tokens of size 1.

| Bundle | Topology | Declared sizes | State |
|---|---|---|---|
| `glm5next.v1` | `glm5next-tp4` | 1, 2, 4 | Served; DCP 2 and 4 qualified through 32 sequences ([GLM 5.3 Flash](model-glm5next.md#context-parallelism)) |
| `qwen38.flash.v2` | `qwen38-flash-tp4` | 1, 2 | Served; DCP 2 gate passed, not release-qualified ([Qwen3.8 Flash](model-qwen38.md#context-parallelism)) |
| `qwen38.flash.v2` | `qwen38-flash-tp2` | 1 | KV heads already distinct per rank |
| `mimo.v26.flash.v2` | all | 1 | No context parallelism declared |

## The host flag

`serve --context-parallel-size N` selects the group size; it defaults to 1
and 0 is refused. N must divide the served topology's rank count, and the
topology must declare it: the host refuses any other size before loading,
naming the sizes the bundle declares. A [serving profile](serving-profiles.md)
may set `context_parallel_size`, which becomes the flag's default under the
same checks. The served identity reports `context_parallel_size` when the
topology declares groups ([deploy-common](deploy-common.md#what-health-reports)).
A developer serving envelope serves only size 1.

## Groups

A group is N consecutive topology ranks. Rank `t` belongs to group `t / N`
as group rank `t % N`, and the members of group g are ranks `g * N` to
`g * N + N - 1` in group-rank order, which is also the slot order of every
peer pointer array a group collective passes. The collectives see only the
group rank and size, so every group of a topology runs the same launches.

KV that every rank holds can split over any N that divides the rank count.
KV with fewer heads than ranks is replicated across consecutive ranks, so a
group must lie inside one such replica set; the bundle declares only sizes
that divide the ranks per KV head. The host checks divisibility and the
declaration, not the replica sets, so this rule is the bundle's to keep.

## Pages and block tables

The [KV pool](serving-kv-pool.md#context-parallelism) owns pages
round-robin by page: physical page `j` of a sequence, positions
`j * page_tokens` onward, lives on group rank `j % N` at the rank's local
page `table[j / N]`. The pool allocates stripes, one page on every group
rank at the same local index, so one logical page covers `N * page_tokens`
positions and the logical block table is every rank's own table of local
pages. `runtime_kv::ContextParallelPagesV1` computes each rank's pages, token
counts and positions; group rank `r` holds row `i` of its local pages at
position `(i / page_tokens * N + r) * page_tokens + i % page_tokens`.

The batch wire does not change: every append carries the logical stripe
table, and each rank derives its own pages from it. The bundle learns N once,
from the create record (a `chop.context-parallel.v1` tag and a `u32` size,
sent only above 1), so a size-1 record, every batch wire and every journal
stay as they were.

## Capacity

Every rank allocates the same number of local pages, sized by the tightest
rank, and the logical pool has that many stripes: token capacity is pages per
rank times `N * page_tokens`, counted across the group. Admission reserves
stripes, so no rank runs out of pages before another. Startup logs the pages
per rank and the group's token capacity.

The host adds no context-parallel term of its own. A bundle allocates its
group buffers (gathered queries, gathered indexer pairs, partial blocks to
send and receive, and the collectives' flags) when it is created, before it
acquires the KV pool, so they come out of the memory the pool would use. The
partial blocks dominate: a rank sends N and, with the compact exchange,
receives N, where the row exchange's receive buffer needed `N * N`. A bundle
that plans its capacity counts these buffers at each size it declares.

## Prefix cache, snapshots, preemption and offload

These work on stripes without change ([KV pool](serving-kv-pool.md#context-parallelism)).
Prefix digests use stripe-sized blocks, but anchors stay where the scheduler
captures them, so a hit can restore inside a stripe; the next append copies
the shared partial stripe, one page on every group rank. A snapshot's state
pages are logical pages, so each rank writes its own state into its local
page. Zeroing, the copy of a shared last page and offload transfers act on
each rank's local page, which for a stripe is that rank's share.

A group of N ranks over pages of P tokens therefore reuses prefixes in
stripes of `N * P` tokens: the scheduler captures a shared prefix in whole
stripes, and a prompt end (on the prefill granule) only once it spans a
stripe. At size 2 with 64-token pages, a 64-token prefix is never resumed,
and a prompt that ends on a page boundary after an odd number of pages
resumes one page less than at size 1. Hits are bitwise equal to cold
prefills at every size. `/health` reports `kv_page_tokens` beside
`profile.context_parallel_size`, and `tools/qual/prefix_exact.py` expects
its hits from both and records the stripe as `reuse_tokens`.

Tests: `dcp_sharded_pool_contract.rs` in `runtime-kv` (every page owned
once, capacity counted across ranks, prefix anchors and copies keeping each
position on its owner, whole-stripe offload and reload);
`context_parallel_contract.rs` in `runtime-scheduler` (reservations count
stripes); `context_parallel_bytes.rs` in `runtime-host` (each rank's bytes
survive stripe copies, offload and reload at sizes 1, 2 and 4). Preemption
uses the same snapshot and offload paths; there is no preemption test
specific to context parallelism yet.

## Kernels and collectives

The shared entries are in `crates/kernels-sm120` (`cuda/context_parallel`,
`cuda/owned_writes`, `cuda/paged_partials`, `cuda/pooled_indexer`), with
contracts in `kernel_core::context_parallel` and host references in
`context_parallel_oracle`. They are instantiated for groups of 2 and 4. One
attention layer of a step runs:

1. **Owner-routed KV writes.** Each appended row's position is localized to
   its owner's pages (`LocalizeKeys` at width 1; -1 on every other rank), and
   only the owner stores the row's K and V or latent. Pooled indexer keys
   are also stored only on their page's owner, but what every rank computes
   and keeps depends on the indexer:
   - The gated pooled indexer (`cuda/pooled_indexer/key_write.cuh`) runs
     the whole pooling on every rank, the gate softmax and the weighted sum
     of the group's normalized keys, and only the store is owner-only. Its
     persistent state, written on every rank, is each sequence's
     incomplete group: the normalized keys and gate logits of its tokens,
     with one checkpoint of that tail per row in a verify step.
   - The block-key path (`cuda/owned_writes`, `owned_index_key_write`)
     pools a block only on its owner and keeps the raw-key ring and the
     anchor ring on every rank, so any rank can pool a block that straddles
     launches.

   Either way the stored bits are the served kernels' bits.
2. **Per-rank indexer scoring and the exact top-k merge.** Where attention
   reads a selected key list, each rank scores and selects over its own
   pages, the group all-gathers the (score, index) pairs, and every rank
   merges them (`TopKMerge`, `TopKMergeBlocks`). Selection ranks candidates
   by score descending, -0 equal to +0, NaN below -infinity, ties to the
   lower index, padding never selected. The emitted list holds the kept
   entries in ascending index order, not score order (a pooled indexer's key
   list then appends the always-selected incomplete group), followed by -1.
   The merged list equals the single-rank list.
3. **Key localization.** Each rank turns the global list into its own
   (`localize_keys`): an owned position p becomes the row
   `p / (N * page) * page + p % page` of the rank's own page table, every
   other entry -1. The result is exact against the ownership reference, and
   mapping each rank's rows back to positions recovers the global list, but
   it is not the size-1 list: with 64-token pages, position 64 is row 64 at
   size 1, while at size 2 it is -1 on group rank 0 and row 0 on group
   rank 1.
4. **Query all-gather.** The group all-gathers the queries rank-major,
   `[N][rows][heads per rank][width]`, through the shared F32 all-gather
   or the phased group gather (BF16 moves as 32-bit words): a copy with no
   arithmetic. The shared all-gather writes each tile to every peer at once,
   which a GPU with more than one peer to feed does at about 2 GB/s on
   PCIe; the phased group gather (`context_parallel::phased`) sends to one
   peer at a time, as the tensor-parallel collectives do.
5. **Per-rank partial attention.** Each rank runs the partial entry once per
   source block, over its own keys, and writes the running maximum `m`, the
   denominator `l` and the unnormalized output `o` in F32 per row and head
   (latent attention's partial output; `PagedAttentionPartials` and
   `SelectedAttentionPartials` for grouped-query and selected-block
   attention).
6. **Owner-targeted gathers.** Block o of every rank goes to group rank o,
   which holds those heads, into a compact receive buffer of N blocks:
   either the shared all-gather run once per owner with `target = o`,
   members running owners 0 to N-1 in the same order on one state, or one
   exchange of the phased group gather, in which every GPU sends to and
   receives from one peer per phase.
7. **Fixed-order log-sum-exp merge.** The owner folds `M = max m_r`, the
   weights `a_r = exp2((m_r - M) log2 e)` (0 for an empty rank),
   `L = sum a_r l_r` and `O = sum a_r o_r` with FMAs from zero in group-rank
   order, and writes BF16 `O / L` (zeros when every rank is empty).
8. **Group barrier.** A one-word all-gather that completes only once every
   member has entered it orders the reuse of the receive buffer across
   collective states.

The layer then continues as without context parallelism: the value or output
projection and the tensor-parallel all-reduce are unchanged.

## Numerics and determinism

The entries follow the engine rules in [numerics](numerics.md): no MMA chain
from zero spans more than 128 K (eight k16 steps), each chain's F32 temporary
is added into the accumulator in K order, F32 math uses FMA in a fixed order
and never TF32, there are no float atomics, every reduction has a fixed order
and ties go to the lower index. The partial entries are gated against
Neumaier-compensated f64 within `2^-22 K S` (`partials_hold_the_accumulation_rule`,
`latent_attention_accumulates_in_128_k_chains`), and every entry is checked
eager, relaunched, from a graph and behind a PDL producer on poisoned outputs
(`kernel_lab::poison`).

What agrees exactly with a group of one, once placement is accounted for:
the stored K/V, latent and pooled-key bytes at their owner's pages, the
replicated indexer state (rings or tail), the query rows, and the merged
key and block lists. Key localization is exact against its ownership
reference instead (step 3). What is not guaranteed to agree: the merged
attention output. With equal maxima (or
one rank) every merge weight is exact and the merge equals its F32 reference
bit for bit; otherwise it holds the f64 merge within half a BF16 ulp plus
`2^-22 K S` and a stated `exp2f` allowance. The kernel-lab chains gate both
the size-1 and the group output against the same f64 reference; their paired
difference is reported, not gated. Context-parallel output is therefore not
guaranteed bitwise equal to size 1 (some inputs do agree), and a bundle
qualifies each declared size on its own.

## How a bundle opts in

1. Declare the sizes above 1 per topology in the manifest
   (`BundleTopologyV1.context_parallel_sizes`: ascending, each dividing the
   rank count). Size 1 is always served.
2. Read the create record's size with
   `runtime_bundle_sdk::context_parallel::{is_next, read}`, which refuses a
   size below 2 or one that does not divide the ranks (`InvalidArgument`)
   and an undeclared one (`Unsupported`).
3. Allocate the group buffers before acquiring the KV pool, and lower each
   attention layer as above from the shared entries.
4. Refuse every declared size above 1 at create until the lowering passes
   its GPU check, with a message that names the size to serve instead. The
   [table](#context-parallel-attention) links each bundle's own switch.
5. Qualify each declared size with its own gates and profile before calling
   it served.
