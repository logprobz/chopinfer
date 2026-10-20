# Ordered last-producer head reduction

This experiment fuses the LM head's separate F32 reducer into the last
producing CTA for each output tile. Engram already performs its merge in
the last CTA in both the native stream and ticket paths, so it is outside
this pair. Other exact-F32 reducers remain unchanged pending this result.

The original 128-K partial planes remain at their original workspace
offsets. Each plane retains the same four-warp producer, three BF16
component chains, eight k16 steps, component merge and scalar fallback.
C1/C8 use the existing eight-row producer arithmetic; C32 uses the
qualified thirty-two-row weight-reuse producer.

The row-classification launch also clears one integer arrival counter per
output tile. Counters follow the aligned end of the original workspace.
Every producing thread fences its partial stores before the CTA publishes
an acquire/release integer arrival. Only the last CTA reduces the tile.
Every output starts at positive zero and adds the original global planes
in ascending chain order with `__fadd_rn`. A CTA barrier precedes counter
reset. No floating-point atomic reaches the output.

The pair crosses column-fast and split-fast producer order with scalar
and sixteen-plane prefetched reducer loads. Prefetch changes load
scheduling, never addition order. Entry wrappers alone translate physical
CTA indexes and issue grid dependency instructions; device tasks receive
virtual indexes and a shared arena. Serving selects only the split-fast,
scalar-load entries at measured capacities 1, 8 and 32 for Flash TP4.
Other capacities retain their previous entry. AR pads live sequences to
graph capacity; verification retains its exact row width.

The captured corpus supplies AR eight-row and thirty-two-row heads and
two thirty-two-row verification heads. C1 uses the leading real row from
each source; C8 uses the leading eight. No row or activation is synthesized
for timing. This is qualification on captured row prefixes, not a new C1
server capture. The reference at C32 includes the qualified M32 head
entry; C1/C8 retain their current entry and separate reducer.

Each GPU executes eleven source/row combinations times four variants.
The pair asserts 88 configurations, 704 output comparisons, 704 full
partial-plane and classification comparisons, and 704 counter-reset
checks. Cases include full, tail, empty and scalar-fallback inputs in
eager and graph mode, with three candidate graph replays. Tail inputs
are poisoned. C1 tail and empty cases intentionally overlap at zero live
rows. Every configuration also executes a GPU bit flip and reversed-fold
controls in eager and graph modes: 264 detected controls in total.
Cold and warm timing each contain nine samples per arm, for 176 paired
timings and 3,168 event samples.

The staging preflight replays the previous projection parser against
recorded receipts and tests the new partial/counter/order requirements on
synthetic output. The four-CPU dry run checks frozen exports, selectors,
captured operand hashes, row-prefix bindings and the missing-capture
refusal. CUDA compilation audits zero stack and local memory. All plans
use static shared memory below 48 KiB; no dynamic opt-in is required.

The job has a 1,200-second cap and a 780-second timeout for each GPU.
The preceding captured projection pair took 930 seconds, including 160
head configurations. Assigning all that time to the head gives a
conservative 511.5 seconds for this pair's 88 configurations. The device
deadline exceeds that estimate by more than 50 percent. Frozen binaries
run directly inside the job; CPU builds and preflight use the admission
wrapper. The job checks 250 W before and after and fails on kernel Xids
or unavailable kernel-journal access.

Each comparison arm owns its partials, counters and output. No workspace
is shared across forked streams. The captured pair passed all declared
output, partial-plane, counter and negative-control checks on two devices
at 250 W on 2026-10-20. The selected candidate/current ratios were
0.939-0.949 at one row, 0.890-0.912 at eight rows and 0.923-0.950 at
thirty-two rows. Sixteen-plane loads added no consistent benefit; the
column-fast variants tied at one row and regressed at eight and thirty-two.
All timing samples reported SM clock drift, so these are paired event-time
results, not a served-throughput claim. The bundle retains the qualified
cubin bytes and bounds the appended counter region explicitly. Journal
pins change only after auditing every retained operation and buffer span.
