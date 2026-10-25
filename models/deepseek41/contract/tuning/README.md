# DeepSeek schedule tables

The `flash/` directory holds one `chopinfer-tuning-v1` table per family.
`model_deepseek41_contract::tuning::table` returns these tables only for
FLASH. Other geometries do not inherit them. A missing row means the
model's baseline schedule.

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
`top_k` and `width`. Weight format, shape and bucket match exactly.

The bucket is the instance's row capacity. An optional `rows: [lo, hi]`
selects an inclusive launch-row range, with `1 <= lo <= hi <= bucket`.
For a given operation, weights, shape and bucket, either one row omits
`rows` and covers the whole bucket, or all rows specify ascending,
disjoint ranges. Gaps use the baseline. Queries outside `1..=bucket`
return no row. Exact-key `get` and `require` retain exact-key lookup;
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

The initial tables are empty and do not change serving selection. Each
family's owner fills its table and replaces its stub in
`bundle/src/tuning_tests/<family>.rs` with served-key coverage. That test
pins the served keys with no row, so a newly untuned bucket produces a
test diff. Keep each family's table and tests in its own files.
