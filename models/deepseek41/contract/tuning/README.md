# DeepSeek schedule tables

The `flash/` directory holds one `chopinfer-tuning-v1` table per family.
`model_deepseek41_contract::tuning::table` returns these tables only for
FLASH. Other geometries do not inherit them. A missing row means the
model's baseline schedule. The bundle build parses and validates every
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
operation name, including its family prefix. Weight format, shape and
bucket match exactly. Shape queries accept any order and reject duplicate
names. Once the tables are initialized, lookup uses stack storage and
allocates nothing.

The bucket is the instance's row capacity. An optional `rows: [lo, hi]`
selects an inclusive launch-row range, with `1 <= lo <= hi <= bucket`.
For a given operation, weights, shape and bucket, either one row omits
`rows` and covers the whole bucket, or all rows specify ascending,
disjoint ranges. A range `[1, bucket]` is refused: omit `rows` for
whole-bucket coverage. A query outside every range or outside `1..=bucket`
returns no row and uses the baseline. Exact-key `get` and `require` retain exact-key lookup;
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
with no row. Missing keys select the existing baseline. The pins enumerate
the operation, shape, capacity and uncovered launch ranges, so a newly
untuned bucket produces a test diff.

## Latent, indexer, F32 and dense prefill

These operation sites retain the existing schedule choices:

| Family | Sites and phases |
| --- | --- |
| `latent` | `sources`, `cache`, `staging` in decode and verify; `query` in decode and verify; `paired` in prefill |
| `indexer` | `hierarchy` in decode; `query` in verify; `cached`, `candidate_scores`, `compressed_scores` in prefill |
| `f32` | `exact`, `band`, `blocked`, `wide`, `router_zero`, `router_norm` in prefill; `head_last` in decode; `bounded_reuse`, `cached` in verify |
| `dense_prefill` | `compact`, `matrix`, `tma_scales` in prefill |

The sites identify the existing registration or preparation stage. Small-row
F32 and latent query registrations also serve callers of other phases through
the existing prepared-op lookup. Bank eligibility stays at those callers.
The F32 head's projection-reuse choice belongs to `reuse.json`.

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
certificate ABI checks. Its route metadata and the zero router's historical
measurement domain remain unchanged pending the separate route-tag refactor.

Dense-prefill registration keeps compact's `ROWS` domain, which is also
consumed by dense-fold registration. Kernel catalogs retain every qualified
control and unused variant. `mxfp8_fold::prefill_tail_policy` remains a code
gate: its shape thresholds and row-remainder evaluator decide the split
tail. `lower/dense.rs` retains fused/local precedence, layer and bank gates.
`ops/dense.rs` retains prefill, short-prefill, wide Engram, stream and workspace
precedence, including the 128-row prefill threshold and the qualified
1-64-row short-prefill path. These rules are outside the table lookup.
