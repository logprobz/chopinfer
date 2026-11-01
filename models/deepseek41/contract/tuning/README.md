# DeepSeek schedule tables

The `flash/` directory holds one `chopinfer-tuning-v1` table per family.
`model_deepseek41_contract::tuning::table` returns these tables only for
FLASH. Other geometries do not inherit them. Missing optional admission
rows select the model's baseline schedule. Required descriptor rows must
remain present, even when no launch selects them (see below). The bundle build parses and validates every
family table before compiling kernels, so malformed tables fail the build.
The contract build generates `Family`, its names and the embedded texts
from one family-to-file list.

| File | Choices |
| --- | --- |
| `dense_fold.json` | Dense fold schedules for decode and verify |
| `reuse.json` | Projection reuse, including the F32 head |
| `dense_prefill.json` | Dense prefill schedules |
| `f32.json` | Exact F32 projection schedules |
| `moe.json` | Routed expert schedules |
| `hc.json` | Hyper-connection schedules |
| `norm.json` | Normalization and quantizer schedules |
| `latent.json` | Latent attention schedules |
| `indexer.json` | Indexer schedules |

Operation names are `<family>.<site>.<phase>`. Replace `-` in a projection
or stage ID with `_`, for example `dense_fold.attention_q_b.verify`.
Shape keys come from `n`, `k`, `groups`, `paired`, `ranks`, `experts`,
`top_k` and `width`. The loader checks this set and the exact three-part
operation name, including its family prefix. Each site also declares its
supported schedules and exact knob names, types and catalog choices. Unknown
sites, ignored knobs and unimplemented schedules are refused at table load.
Kernel validators still check shape and resource constraints. Weight format, shape and
bucket match exactly. Shape queries accept any order and reject duplicate
names. Once the tables are initialized, lookup uses stack storage and
allocates nothing.

The bucket is the instance's row capacity. An optional `rows: [lo, hi]`
selects an inclusive launch-row range, with `1 <= lo <= hi <= bucket`.
For a given operation, weights, shape and bucket, either one row omits
`rows` and covers the whole bucket, or all rows specify ascending,
disjoint ranges. A range `[1, bucket]` is refused: omit `rows` for
whole-bucket coverage. A query outside every range or outside `1..=bucket`
returns no row and uses the baseline for an optional admission. Exact-key `get` and `require` retain exact-key lookup;
use `select` to match a launch row count.

Rows sort by operation, weights, shape, bucket, then range. Shape entries
and knobs sort by name. `Table::parse` requires the canonical bytes from
`Table::render`; absent ranges are omitted. Empty `knobs: {}` are valid
when the schedule name alone expresses the choice. Existing tables keep
their bytes and schema ID.

Geometry ID, drafter or target bank, eager or graph execution, item count,
chunk offsets, KV storage, runtime flags and precedence chains remain code
gates. They are never table keys. Choices that change numerics also stay
out of tables, including native MoE prefill tiles versus base tiles and
BF16 all-reduce partials. A row may choose only schedules that preserve
the operation's arithmetic.

Each family's `bundle/src/tuning_tests/<family>.rs` pins the served keys
with no row. Missing optional keys select the existing baseline. The pins enumerate
the operation, shape, capacity and uncovered launch ranges, so a newly
untuned bucket produces a test diff.

## Required descriptors

These rows construct cubin descriptors unconditionally. Each uses weight
format `mxfp8` and shape `ranks=4`; its range must include the named launch row.
Deleting it or excluding that row is a table-load error naming the missing key.

| Operation | Bucket | Required launch row |
| --- | --- | --- |
| `dense_fold.status.decode` | 32 | 32 |
| `dense_fold.status_wide.verify` | 64 | 64 |
| `dense_fold.attention_q_a_status.verify` | 64 | 64 |
| `dense_fold.attention_o_a_status.verify` | 64 | 64 |
| `dense_fold.attention_o_b_status.verify` | 64 | 64 |

Status and tail admissions using tile 16 or 32 also require the matching
`dense_fold.status.decode` descriptor in bucket 32 and the
`dense_fold.attention_{q_a,kv,o_a,o_b}_cluster.decode` descriptors at that
tile row. Tail admissions do not require the KV descriptor. Earlier status
ranges must carry the same knobs as their tile descriptor.

The three cluster status verify sites use their row-64 descriptor for every
admitted launch, so their ranges must carry the same schedule and knobs.
`dense_fold.status_wide_tails.decode` must match the required wide verify
descriptor. Latent staging rows must be singletons; preparation includes
every admitted singleton that fits its capacity. Paired prefill substitution
uses only the launch ranges declared by `latent.paired.prefill`.

## Dense decode knobs

`dense_decode` rows in `dense_fold.json` and `reuse.json` encode the
schedule through five required knobs:

| Knob | Encoding |
| --- | --- |
| `tile_n` | Output-column tile, an integer count. |
| `row_groups` | One group or two groups of sixteen rows. |
| `stages` | Ring depth, a count fixed by the supported finish. |
| `weight_prefetch` | Boolean; requires persistent weights/scales ready before the PDL predecessor. |
| `finish` | Count `0` for ticket finish, `1` for a separate merge launch. |

Ticket finish admits `tile_n` 32 or 64, `row_groups` 1 or 2 and `stages` 3;
weight prefetch requires one row group. Merge finish admits exactly
`tile_n=16`, `row_groups=1`, `stages=2`, `weight_prefetch=false`.
Two-group reuse rows additionally require `folds` 2, 4 or 8. This is the
runtime partition, not a template knob; other rows must omit it and choose
their partition at launch. `Knobs::from_row` refuses missing, mistyped,
unknown or unsupported values.

`dense_fold.wide.verify` rows choose the schedule of a wide verify fold,
the decode template with a 256-row workspace header, for each
`verify_bounds::WIDE` bucket (128 and 192 rows), keyed by the bucket at
its widest row. They admit only ticket schedules of one row group; a
missing row keeps the ticket baseline.

The 2026-10-31 wide fold evaluation (the `fold_core` lab test
`dense_wide::gpu_wide_schedules`, synthetic inputs, one GPU per shard)
compared all four admitted schedules of every key with the served ticket
at every row each bucket serves, 65 to 128 and 129 to 192: 1024 cases,
all bitwise equal through eager and graph replays with live and rejected
rows, and every changed-activation control detected. Timed at the middle
and widest row of each bucket (15 samples per arm, the ticket symbol
loaded twice as an A/A arm), weight prefetch at the ticket's column tile
was faster cold and warm in every key. A row takes it where its cold
gain exceeded 2% and twice the A/A difference at both timed rows: 13 of
the 16 keys, 3.9% to 20.7% faster cold. N4096/K1280 keeps the ticket in
both buckets: at three of its four timed rows the gain (4.9% to 6.8%)
was within twice the A/A difference (3.9% to 4.4%). N1280/K5120 in the
128-row bucket missed the rule only at 96 rows (13.6% against twice a
7.2% A/A difference; 14.5% at 128 rows against 3.8%) and takes prefetch
too, so each shape keeps one wide schedule and the module change stays
an exact rename of its entries. The other
column tile is not adopted; against prefetch at the ticket's tile it was
mixed and mostly within twice the A/A difference. Served step-time
acceptance remains required.

Every schedule retains E4M3 ceil-UE8M0 activations, UE8M0 scales per K32,
four K32 MMAs per K128 chain and ascending F32 reduction. Those policies
are numerical identity. They cannot be changed by a tuning row.

## Whole-bucket evaluation

Decode graph buckets are 1, 2, 4, 8, 16 and 32. The weighted verify widths
are 6, 8, 16, 32 and 64, but verification also serves exact intermediate
widths. The fold registry serves short prefill through 64 rows. A gap
between decode buckets is therefore not evidence that a row is unserved.
Keep the measured small-row routes, including rows 2 through 7.

An interval candidate uses a schedule already selected inside that
interval. Every changed shape and row must pass byte twins, and its journal
change must contain only declared symbol and grid substitutions. Served
bitwise comparison and the matched-seed simplification rule decide adoption.
Lab timing alone does not qualify a table change.

The 2026-10-26 fold pair evaluation retained the finer N5120/K15360 rows
in 17..32. The candidate's small timing differences were unresolved against
the in-job A/A control, which reached 6.88%; row 32 regressed on both
devices. Its existing prefetch islands remain. Split-to-prefetch intervals
also remain separate because they change launch count and arguments.
Reuse fold-count changes alter a scalar argument, so they cannot pass the
symbol/grid-only proof used by this evaluation.

The served evaluation restored N4096/K1280 rows 9..16 to their original
prefetch islands at 10..11 and 13, with ticket elsewhere. All eleven target
cells passed, but an additional weighted C8/R16 cell regressed: matched-pair
changes were -2.243% and -1.527%, mean -1.885%, spread 0.716%. This fails
the simplification rule. The restored candidate changes 99 cells and
removes 18 table rows. Its 2026-10-27 requalification passed all five
positively weighted job/cell observations across four served
configurations, with 437 matched request keys bitwise equal. All-family
fallback maxima and means stayed within the base caps, and power limits
remained constant. No schedule at rows 1..8 changes.

## Latent, indexer, F32 and dense prefill

These operation sites retain the existing schedule choices:

| Family | Sites and phases |
| --- | --- |
| `latent` | `sources`, `cache`, `staging` in decode and verify; `query` in decode and verify; `row_fusions` in verify; `paired` in prefill |
| `indexer` | `hierarchy` in decode; `query` in verify; `cached`, `candidate_scores`, `compressed_scores` in prefill |
| `f32` | `exact`, `band`, `blocked`, `wide`, `router_zero`, `router_norm` in prefill; `head_last` in decode; `bounded_reuse`, `cached` in verify |
| `dense_prefill` | `compact`, `matrix`, `tma_scales` in prefill |

The sites identify the existing registration or preparation stage. Small-row
F32 and latent query registrations also serve callers of other phases through
the existing prepared-op lookup. Bank eligibility stays at those callers.
The F32 head's projection-reuse choice belongs to `reuse.json`.

`latent.row_fusions.verify` (weights `none`, shape `ranks`, bucket 32)
admits verify steps of its rows to the decode row fusions: the fused query
norm and ceil and the fused window norm, RoPE, ceil and store that decode
steps launch, prepared for 32 rows. A shape with a row also quantizes the
drafter's window writes of at most its draft rows with the parallel decode
ceil entry instead of the serial one. The fused entries keep every value
and every add of the chains they replace, in the same order, so the row is
not numerical identity. A missing row keeps the chains.

Exact F32 rows take precedence over bands. Blocked and wide overrides retain
their existing order, followed by head reuse, head-last and optional bounded
reuse at registration. Their separate sites retain the original entry
metadata as well as the final launch. The shared split-F32 construction and
its 33-256-row baseline stay in code for every geometry.

Latent source qualification still checks the attention dimensions, kernel
tag, base geometry and topology. Draft-block exclusion, native-KV checks,
source compatibility, available caches and scratch remain code gates. The
selected cache precedes decoded prefill, optional shared heads and the base
attention. Cached rows are clipped by both the prepared capacity and the
instance's verification budget. The query table selects the parallel registration. Its 48-row capacity stays
shared with the generic prepared-op adapter, whose capacity lookup also
serves baseline geometries. The paired prefill registration needs
the full 8192-row capacity and serves every valid launch count within it.
Native prefill finish, staged cache, query-window fusion, sequence count,
chunk offsets, origins and storage/layout checks remain in code.

Indexer hierarchy retains native-KV, exact geometry and runtime-MMA gates.
Its scratch limits remain resource checks. Diagnostic verify routing,
source-layer eligibility, positions, score-byte limits, eager execution,
FP4 storage, candidate availability and 512/4096-row batching remain in
code. Candidate and compressed scoring sites are selected only after their
source/layout gates. Cached selection retains compression-ratio and width
compatibility; its table enumerates each actual bounded preparation capacity
and uses ranges for the admitted launch rows. Wide query and F32 preparation
still require the target bank and available operations; resource bounds are
checked independently of schedule selection.

The certified router retains weight-domain, workspace, aliasing and
verification-overlap checks. Router-in-norm retains its producer and
certificate ABI checks. Its route metadata is derived from the corresponding table admissions.

Dense-prefill registration keeps compact's `ROWS` domain, which is also
consumed by dense-fold registration. The lab retains qualified controls; the package loads served entries and
required dependencies. `mxfp8_fold::prefill_tail_policy` remains a code
gate: its shape thresholds and row-remainder evaluator decide the split
tail. `lower/dense.rs` retains fused/local precedence, layer and bank gates.
`ops/dense.rs` retains prefill, short-prefill, wide Engram, stream and workspace
precedence, including the 128-row prefill threshold and the qualified
1-64-row short-prefill path. These rules are outside the table lookup.

The shared GEMM templates keep the existing operation keys and row ranges.
Local rows name `local_wide` with stage count and tile sizes; cluster rows
name `cluster_fused` with CTA-local storage (`distributed: false`) and the
existing split, stage and column-tile values. Direct rows name `direct`
with `n_warps` and `k_warps`. Compact and matrix prefill rows name
`prefill_tiles` with `matrix_fetch`, `stages` and `tile_rows`; scalar-fetch
tiles retain the existing row-count selection. These choices preserve the
qualified symbols, launch plans and numerical policies. Staged-scale
prefill remains a separate schedule.

The `cluster_fused` schedule refuses `distributed: true`; its qualified
fused entries require CTA-local storage.
