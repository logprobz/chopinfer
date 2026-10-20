# Captured verify head groups

This lab compares the served attention route with selected-key caching and
fixed groups of two, four and eight heads. Group zero retains the original
cache schedule (one head at four rows, eight heads at thirty-two rows).
Every candidate keeps eight numerical splits, the original 64-key units,
the per-head QK and PV arithmetic, and the original ordered merge.

The prerequisite attention capture must pass its exact F32 partial and BF16
output replay. The pair uses all 24 records: four-row and thirty-two-row
forwards at cold 8K and 32K contexts, each with window, mixed32 and mixed64
layers. Native pages, query, selections and RoPE are captured operands.

Each GPU owns one row-count shard. Memcheck precedes correctness, which
precedes timing. Each correctness record runs eight operand controls, two
scratch poisons, four head schedules, four row-partition comparisons and a
dropped-key negative control. The full check also replays captured graphs. Instrumented consumers must
reproduce the same bits and write the expected clock64 records for every
CTA. A separate unprivileged driver probe records function attributes and
occupancy; instrumented kernels never enter the performance comparison.
Timing measures cold source calls, cold attention including merge, and
32 repetitions of the complete attention graph. Each timed arm is checked
again after measurement. No job builds code or queues another job.

Run `test_run.py` on CPU before staging. Build the attention lab's example
with `--initialize` only for the new additive lab lock, then build its release
test binary. `stage.py` accepts the successful capture run, frozen cubins,
test binary and sanitizer. Run the generated job with `--dry-run` through
the CPU admission wrapper, then provide its path and SHA256 to the queue
owner. The job allows 360 seconds for memcheck, 180 for correctness and 360
for paired timing, with an overall work deadline of 1080 seconds and an
outer job deadline of 1140 seconds. These bounds must be checked against
recorded comparable runs before submission.

This is boundary-level kernel evidence. Served token/logprob equivalence
and TTFT require a separate integration qualification.
