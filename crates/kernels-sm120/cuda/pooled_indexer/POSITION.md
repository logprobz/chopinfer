# Position indexer

`position.cuh` implements `PositionIndexerQuery` and `CandidateBlocksQuery`.
Its Rust builders are `pooled_indexer::position::{Entry, CandidateEntry}`.
Register them with `Sm120::with_position_indexer` and
`Sm120::with_indexer_candidate_blocks`, then prepare through
`Backend::position_indexer` and `Backend::indexer_candidate_blocks`.
The existing pooled BF16 cubins do not include this header.

The score input is a prepared BF16 query containing MXFP4 fake-quantized
values, F32 projected head weights, and paged MXFP4 keys. Both dot operands
decode exactly to F32. Key codes and
UE8M0 scales have independent 64-bit page strides. RMSNorm, tail rotation,
and trained quantization precede this engine. Projections, normalization
and rotation stay F32 until the quantizer's one BF16 input rounding.
The score/select entries and their contract IDs are v3. Each compressed key is one
candidate, and a token at t sees only `(t + 1) / compression_ratio` complete
positions. A negative token position, a negative page-table entry, or a
zero candidate-mask byte excludes that position.

A warp scores one position. Lanes accumulate channels in 128-channel
groups, reduce each group in a fixed warp tree, and fold groups in channel
order. The dot is rectified before weighting. The combined inverse-square-
root scale multiplies the head weight with an explicit F32 rounding; the
head sum uses FMA in head order. Every rank runs the whole indexer head set.
There is no TF32 or floating atomic sum.

Scores are exposed in a row-major F32 buffer. Excluded positions are -inf.
Selection reads that same buffer, ranks score/index pairs as 64-bit integer
keys, and finds the exact kth key with eight byte-radix passes. Integer
histogram counts are independent of arrival order. A final prefix scan
writes positions in ascending order, then pads the list with -1. Both
signed zeros tie, and finite score ties prefer the lower index. A hole
never becomes an ordinary -inf-scored candidate.

Each live row also writes an I32 status: zero on success, one on non-finite
admitted operands or arithmetic. The score path checks decoded inputs,
chain and tree sums, the dot before ReLU, scaled weights and head sums.
The score stage marks failures with canonical NaN (0x7fc00000). Selection
detects any non-finite admitted score, replaces it with -inf, sets status to
one and writes an all -1 list. Excluded positions are never read for this
check. The caller must reject a failed step before accepting its results.
A following block filter can run safely in the same graph because its
inputs contain no failed NaN or +inf. This reports an error without a
device trap. The f64 oracle remains the finite-score accuracy gate; the F32
oracle refuses invalid native arithmetic.
The pooled BF16 engine's total-order key encoding is unchanged.

The candidate filter first writes the maximum visible score per block.
The block containing `visible - 1` gets +inf, including when the block is
full. The zero-visible case pins no block. A second radix selection keeps
the requested number of blocks, counting the pinned block within that
budget, and expands membership to a byte mask. The newest block's mask can
include future positions; second-level selection still applies visibility.
Input NaNs and +inf are outside the filter's domain and trap; -inf is the
padding value. The partial block is padded with -inf before its maximum.

The `position_indexer` lab lock family includes generic three-head,
160-channel scores crossing a 128-channel boundary, 32-head/128-channel
scores, compression ratios one and two, masked and unrestricted selection,
and eight-position block filters. Separate control symbols ignore the mask,
pin `visible / block_size`, reverse ties, drop candidate zero, shift head
weights or swap key nibbles. None is a served entry. Eight equal-score
positions must select [0, 1]; only head zero has a nonzero weight and the
key's first two channels are 1 and 2. These are explicit score and selection
witnesses for the four latter controls. Overflow, NaN-weight and NaN-dot
cases require status one and an empty selection.

The conformance kit checks continuous scores against f64, then exact
selection against the actual F32 scores from that run. Separate one-term
score cases supply exact bits for page-remapping and missing-page witnesses.
The filter cases cross 16,384 positions. Outputs and scratch are poisoned
for relaunch, graph and PDL checks, and row, chunk and batch comparisons
exercise the declared invariance. A CPU SASS audit rejects TF32, spills,
stack use and missing PDL release/wait instructions. CPU declarations and
build audits do not establish GPU correctness.

The opt-in `position_live` family preserves the Position v3 score arithmetic
and its score/index total order. Dense and compact scores use 192 CTAs per
row with warp grid-stride loops over the device-visible positions. Threads
clear the unused score tail with coalesced stores, preserving the full
buffer's -inf padding without launching a CTA per four capacity positions.
The score grid is independent of the host's context estimate.

Selection partitions the visible range into 32 contiguous shards. Each
shard retains its exact local top-k in position order. The global top-k is
contained in their union: a key excluded from a shard already has at least
k larger keys in that shard alone. A final exact cutoff over those lists
and an ordered merge preserve lower-index ties and ascending output IDs.
Warp ballots supply each scan's lane prefixes; shared warp counts supply
the inter-warp prefix. Failed rows retain the previous status, score
sanitization and output padding. The compact reader validates the complete
ID record, including unused slots, before selecting its visible slots.

The same path builds candidate block IDs with a fixed compiled stride,
including smaller runtime `candidate_topk_blocks` counts. It preserves
newest-block pinning, tags, visible counts and failure propagation. Its
workspace holds block maxima followed by the local key lists; dense and
compact selection need only the local key lists. All workspace is written
before use on each replay. Previous entries and lock families remain
available as the bitwise comparison controls.

DeepSeek registers these variants for its four Full sources and four
Reindex sources. Its index metadata already comes from one `index_upload`
per step and one upload per rank. Reuse layers perform no new index query
or selection. This scheduling change adds no rounding at the dot, weighted
product or head sum and does not introduce FP4 MMA.
