The attention pair lab compares additive source and query schedules with the
current TP4 Flash kernels. The serving registration keeps the reference
entries. GPU bitwise comparisons and paired timings must pass before a
candidate is adopted. CPU compilation does not establish either result.

The source candidate keeps the reference 64-key tiles, eight decode/verify
splits, QK chains, online softmax updates, and ordered merge. At up to four
rows it assigns one head to each CTA. Larger batches retain eight heads per
CTA. This increases repeated KV reads at small batches, so its benefit must
be measured. All-hole tiles avoid widening and matrix products but still
apply the reference FMA updates, including the transition from -0 to +0.
Widening groups four independent pairs into a 16-byte shared-memory store
and reuses their common scale. Unaligned source addresses retain byte
loads; the accepted strides and the individual BF16 conversions stay fixed.

The query candidate uses four groups of 256 threads within one CTA. Every
group processes a different subset of heads. The original row-wide status
barrier remains between validation and writes: any failed head preserves
the entire F32 row and zeros every BF16 query head. The original arithmetic
and sticky status protocol are unchanged.

The model-owned families `latent_sources_scheduled` and
`latent_query_parallel` have separate locks. The reference locks and C entry
ABIs remain unchanged. The indexer scoring and selection implementation is
not part of these candidates.

The `attention_pair` test binary covers 72 combinations: visible contexts
1,024, 8,192, 32,768 and 131,072; 1, 8 and 32 sequences; decode and six-row
DSpark verification; and window-only, ratio-1 and ratio-2 sources. Context
is the visible length of the first query. Verification adds five following
positions. Sparse lists retain a randomly selected cached prefix. The
source check compares every output byte and every F32 split-workspace byte.
Additional cases cover window and page boundaries, holes, poisoned unused
cells, unaligned strides, inactive rows and signed-zero state. A wrong sink
multiplicity must change the result. Query comparisons also cover NaNs, overflow, sticky
status, invalid RoPE tables and zero live rows.

Timings are isolated per-stage measurements with cold-cache preparation.
Source, output merge and their combined path use alternating A/B and B/A
rounds. Query and window preparation are timed separately; query inputs are
restored outside the timed interval. These measurements are not a serving
step or a complete projection-to-projection layer benchmark.

The KV floor counts unique selected payload cells, not the allocated cache:
528 bytes per native window cell and 288 bytes per compressed cell at width
512. Receipts also count payload staged repeatedly across heads and rows.
They report the floor normalized to 1 TB/s, `unique_bytes / 1,000,000` in
microseconds. Divide this by the measured bandwidth in TB/s for a device
estimate. This is a lower bound, not a hardware DRAM traffic counter; it
excludes query, list, workspace and output traffic and cache-line overhead.

Build outputs and snapshots belong under `CHOPINFER_DATA`. CPU checks are:

```sh
export CARGO_TARGET_DIR="$CHOPINFER_DATA/targets/attention-pair"
cargo test -p kernels-sm120 --lib --locked --offline sources
cargo test -p kernels-sm120 --lib --locked --offline query_parallel
cargo test -p kernels-sm120 --test latent_sources_scheduled_compile --locked --offline
cargo test -p kernels-sm120 --test latent_query_parallel_compile --locked --offline
cargo test -p model-deepseek41-lab --test attention_pair --locked --offline cpu_
```

`build_attention_pair OUT --initialize` initializes missing candidate
families and verifies every existing family. Without `--initialize` it only
verifies. The snapshot additionally needs `probe_sm120.cubin` from a
kernel-lab build, with its digest and complete symbol set checked against
the shared lab lock and added to `attention-modules.json`.

Set `CHOPINFER_ATTENTION_CUBINS` to that absolute snapshot directory and
run `frozen_modules --ignored --exact` on the frozen test binary. This
checks every module digest and exported symbol without opening CUDA. GPU
tests are explicitly ignored until an allocated job selects idle devices,
sets `CHOPINFER_KERNEL_REQUIRE=1` and selects shard 0 or 1 with
`CHOPINFER_ATTENTION_SHARD`. The pair job must require both shards, count
markers anywhere in each log line, enforce the 250 W cap, and check the
kernel journal for Xids.
