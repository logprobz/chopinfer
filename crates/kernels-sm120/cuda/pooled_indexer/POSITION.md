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

The `position_valid` variant keeps that dense path and specializes only
compact scoring/selection and candidate maxima. A binary search finds the
end of a valid retained ID prefix, so smaller runtime candidate budgets do
not score or select the padded slots. Selection still validates all IDs,
including the tail: malformed records sanitize the entire score row and
fail. Score padding, missing pages, lower-index ties and ascending outputs
retain their previous meaning.

Candidate maxima use 32 fixed CTAs per row and visit only the device-visible
block count. Their unused internal scratch is not written or read. The
live gather retains its exact cutoff, row tags, visible counts, newest-block
pin, runtime budget and full output padding. Its explicit count validation
handles malformed empty rows even when the maxima stage writes no block.

These planners also serve prefill. A caller can split an 8192-row chunk into
64-row dense/candidate blocks and 4096-row reindex blocks. Each row still
reads its own causal position from device metadata. The `position_prefill`
lab test compares all rows at offsets 0, 8192, 24576 and 122880, in native
and BF16 storage, using both compression ratios and candidate budgets 2048
and 1024. It checks every score byte, retained record and selection, poisons
all outputs and workspace, and rejects wrong causal counts and a gather
that consumes poisoned scratch beyond visibility. Per-kernel event times
sum full row blocks; a pair job joins the two 4096-row shards into complete
8192-row measurements. CPU tests establish coverage and dispatch only.


## Small-row query candidate and MMA experiment

`position_query_parallel.cuh` is a separately locked candidate for 32 heads
of 128 channels with 64 rotary channels, through 32 launch rows. Each warp
quantizes one head. Rotation keeps the original separate F32 multiplies and
adds, and the shared quantizer rounds to BF16 before its discrete decisions.
All groups contribute their flags before one sticky row-status update;
register-held pairs are published only after that admission. Invalid positions
avoid input and table reads, failed rows write positive zero, and inactive rows
write nothing. The opt-in planner uses the runtime row bucket even when the
prepared capacity also covers prefill. It introduces no cross-stream scratch.
The owning bundle enables the small-row path after paired GPU conformance;
its serving binding keeps larger row buckets on the original path.

`position_mma_probe.cuh` is a lab-only experiment. It decodes keys exactly to
BF16 and uses two m16n8k16 MMAs from zero for each 32-channel block, adds the
four block results in order, then keeps the original ReLU, scaled F32 weights
and ordered head FMAs. This is not generally the served reduction: SIMT first
accumulates four strided channels per lane, then sums lanes at offsets
16, 8, 4, 2 and 1.

A quantizer-valid witness has one active head, unit keys, q[0] = 2^25,
q[33] = 1 and q[64] = -2^25, with other query channels zero. SIMT and the
compensated f64 reference return 1/64 after head weighting; block-first
arithmetic returns zero. Exact products within each scale block therefore
do not establish bitwise scores or a deviation no larger than the current
build. No efficient general exact MMA replacement has been established.
The candidate has no serving route. Its lab reports complete score-bit and
selection differences, sampled errors against the compensated f64 oracle,
contract-bound fractions and exclusive, gated, alternating kernel timings.
The witness must reject the bitwise claim even when aggregate error on
randomized inputs improves. Approximate promotion requires the owning
model's real-activation selection and score qualification.

## Register-tiled SIMT score candidate

`position_simt_tile.cuh` assigns two complete scores to each thread in a
four-row, 64-position tile. Each score reconstructs the original 32 virtual
lane chains in registers: four strided FMAs per lane, then additions with
pairing offsets 16, 8, 4, 2 and 1. ReLU, weight scaling and the head FMA chain
retain their original order. A wrong-tree control instead combines adjacent
lanes first and must differ on the seeded fixtures.

Keys are staged once per tile in shared memory and reused by rows with the
same sequence. Mixed batches stage each distinct sequence separately. Four
heads of queries are staged at a time; shared loads broadcast a query value
within each warp. Missing pages and per-row causal visibility are checked
before score publication. Tails remain negative infinity and inactive rows
remain untouched. A nonfinite chain or contributing tree node remains
nonfinite at that head's root; the root is checked before ReLU can hide it.
Invalid scale bytes remain a separate flag. Head-sum nonfinites also persist
to the final sum.

The opt-in backend changes only dense All scoring for 32 heads of 128.
Compact scoring, candidate production and selection reuse the live pipeline.
The operation kind for candidate maxima is explicit in the planner; workspace
size only describes scratch storage. No cross-stream workspace is added.
The new family has its own lock and no serving registration until its full
bitwise matrices pass. CPU tests establish the intended tree and launch ABI;
they do not establish GPU conformance or speed.

## Native FP4 qualification candidate

`position_mma_native.cuh` repacks prepared BF16 queries once per row and
requires the decoded pairs to match their original bits. Its packed code
and UE8M0 scale planes feed four native m16n8k64 instructions per warp for
32 heads and 128 channels. Query fragments stay in registers across key
tiles. Key fragments load once for both head tiles; a padded shared dot
plane avoids the column bank conflicts of the initial BF16 probe. Weight
scaling and the ascending head FMA chain retain the served order. Native
MMA accumulation changes bits and requires separate accuracy evidence.

`position_f64_probe.cuh` evaluates every admitted score with the same
Neumaier-compensated dot, magnitude and head sums as the CPU score oracle.
Its explicit double operations prevent contraction. Replay checks sampled
GPU oracle scores and magnitudes bitwise against the CPU oracle before
using the complete f64 score matrix for selection agreement. Both kernels
are isolated in the qualification family and have no serving registration.
