# Context-parallel entries

Kernel-core contracts: `context_parallel::ContextParallelQuery::PartialMerge`
and `LocalizeKeys`; host references in `context_parallel_oracle` and
`latent_attention_oracle::merge_partials` (f64). The top-k merge of per-rank
indexer selections lives with the indexer (`pooled_indexer/`).

A context-parallel group is `cp_size` consecutive ranks; physical KV page j
of a sequence lives on group rank `j % cp_size`, in the rank's local page
`table[j / cp_size]` (`runtime_kv::ContextParallelPagesV1`,
docs/numerics.md#kv-offload-and-prefix-state "Context parallelism"). Attention heads stay split
across ranks as without context parallelism, so a rank attends the whole
group's heads over its own keys and merges its own heads' partials.

## Key list localization

`localize_keys<Localize<Ranks, Page, Width>>`, grid `[rows, ceil(Width /
256)]`, 256 threads. A key list row `[Width]` of sequence positions (-1
holes), as the indexer's merged selection writes it, becomes group rank
`cp_rank`'s list: an owned position p becomes `p / (Ranks * Page) * Page + p
% Page`, its row in the rank's own page table; every other entry is -1. The
output is a separate buffer, so a relaunch or graph replay reads the same
input. The latent attention entries then attend the rank's own keys
unchanged: they resolve position p as `table[p / Page]`, slot `p % Page`,
and skip -1.

## Query and partial layouts

The group all-gather of the absorbed queries is rank-major: `[N][rows][H]
[width]`, H the heads per rank. Source block s, at element `s * rows * H *
width` (`query_block_offset`), is exactly the row-major `[rows, H, width]`
query the heads-H latent partial attention entry reads, so a rank runs that
entry once per source block, over its own keys and localized lists. The
launch for block s writes `m`, `l` and `o` into the rank's segment at
`partial_block_offsets`: block-major `m [N][rows][H]`, then `l` (same), then
`o [N][rows][H][width]`. Block s holds group rank s's heads.

Group rank o merges block o of every segment, read through
`MergeLayout`:

- all-gathered segments (`MergeLayout::gathered`): `rank_rows = N * rows`,
  `input_heads = H`, `head_offset = o * rows * H`, so pair `row * H + o *
  rows * H + h` is block o's row;
- exchanged blocks (`MergeLayout::exchanged`): `rank_rows = rows`,
  `input_heads = H`, `head_offset = 0`, over the receive view below.

The row exchange (`chop_comm_exchange_rows_f32_v1` with `rows = N` and
`shard = B = rows * H * (width + 2)` floats) needs its own layouts on both
sides; the block-major banks above are the all-gather's input only:

- **Send packing.** The exchange sends input row d (`input + d * B`) to
  group rank d, so its input must be destination-major `[N][B]`, each block
  holding `m [rows][H]`, `l [rows][H]`, then `o [rows][H][width]`. The
  partial attention launch for block s writes its outputs there directly
  (`out_max = input + s * B`, `out_den = + rows * H`, `out_num = + 2 * rows *
  H`); passing the separate m, l and o banks sends the wrong slices.
- **Receive view.** Source q's block for owner o lands at `outputs[o] + (o *
  N + q) * B`, not at `outputs[o] + q * B`: owner 1's blocks start at `N *
  B`. The entry takes allocation bases only, so each receive buffer spans
  `N * N * B` floats, and the merge reads the owner's `N * B` floats from the
  view at `o * N * B` (`group::exchange_buffers`,
  `group::exchange_receive_offset`); the rest is never written.

**Owner gathers.** The same compact layout, `N * B` floats per receive
buffer, comes from the shared all-gather run once per owner o with `target
= o` (`group::owner_gather_arguments`): each member passes its send block o
(`input + o * B`) and the entry stores it at `outputs[o] + source * B`, so
the merge reads the owner's whole buffer with `MergeLayout::exchanged`.
Only the owner waits for every source; a sender waits only for the owner's
acknowledgment of the state's previous call. The members therefore run the
owner sequence 0..N on one state in the same order, and since a state's
first call has no acknowledgment to wait for, a caller that reuses the
receive buffer across states orders the reuse itself (a group barrier after
the merge).

A row-major `[rows][N * H]` producer (a heads-NH partial entry) reads with
`input_heads = N * H` and `head_offset = o * H`.

## Partial merge

`partial_merge<Merge<Ranks, Heads, Width>>`, grid `[rows, Heads]`, 128
threads. Input: `Ranks` segments gathered rank-major (the shared F32
all-gather, `src/context_parallel/group.rs`), each `rank_rows *
input_heads * (Width + 2)` floats: maximum `m` `[rank_rows, input_heads]`,
denominator `l` (same shape), numerator `o` `[rank_rows, input_heads,
Width]`. The CTA of (row, head) merges pair `row * input_heads + head_offset
+ head` (see "Query and partial layouts"):

- thread 0 folds `M = max m_r`, the weights `a_r = exp2((m_r - M) log2 e)`
  (0 for `m_r = -inf`, an empty rank) and `L = sum a_r l_r` with FMAs from
  zero in rank order, exactly as the latent attention split merge does;
- each thread folds `O = sum a_r o_r` for its columns in rank order and
  writes BF16 `O / L` (F32 division; zeros when `L` is 0, every rank empty)
  to `out` `[rows, Heads, Width]`.

With equal maxima (or one rank) every weight is exact and the output equals
`context_parallel_oracle::partial_merge_f32` bit for bit; otherwise the
weights carry `exp2f`'s error and the merge is gated against the f64 merge.
No atomics; rows past the live count return after the PDL wait.

## Phased group gather

`phased_gather.cuh`, `chop_comm_group_phased_f32_v1` (host side
`src/context_parallel/phased.rs`): one cluster of 8 CTAs, 256 threads. Member
`rank` sends `input + d * input_stride` (`words` 32-bit words) to group rank
d, where it lands at `outputs[d] + rank * words`, in phases: phase p sends to
`(rank + p) % ranks` (phase 0 is the local copy), and the cluster
synchronizes before the next destination, so each GPU writes to one peer at
a time (42-47 GB/s on PCIe, against about 2 GB/s for the shared all-gather's
writes to every peer at once, `cuda/comm/README.md`). With `input_stride`
0 it is the all-gather of one shard, with `input_stride = words` the
exchange of a destination-major buffer into compact `[N][words]` receive
buffers (what the owner gathers above produce). Copies use 16-byte vectors
with four loads in flight when the words and both ends allow; the output
bits equal the input bits.

Flags: `2 * ranks` words per member, zeroed at allocation (data flags
`[source]`, acknowledgements `[ranks + receiver]`), and one epoch word, so
any shared all-gather state of the same group size holds them. On entering
call e, CTA 0 acknowledges `tagged(e - 1)` to every peer; it sends to a peer
only after reading that peer's acknowledgement (also on a state's first
call), flags `tagged(e)` after the cluster barrier that follows the copy,
and returns once every source has flagged. Graph replays need no host
update.
