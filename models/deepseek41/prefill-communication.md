# Prefill communication

The optional attention schedule splits the final output projection into two
row ranges for chunks of at least 1024 rows. The first range is aligned to
16 rows. The full-chunk quantizer runs first. Each range keeps the prepared
prefill entry, including its TMA admission launch. A communication stream
reduces A while the compute stream projects B. Both reductions finish
before the original full-row hyper-connection expansion.

The reduction retains the composed transport: F32 scatter, F32 gather,
then one BF16 rounding of the rank-ordered F32 sum. Its complete launch
sequence runs on one stream, including the final local rounding. The next
reduction starts only after the previous reduction on that rank finishes.
No transport registration or kernel changes accompany the schedule.

Admission requires independent producer rows, no workspace, no persistent
grid, no collective producer and no multi-CTA cluster. Resolved pointer
arguments, including tensor-map bases and peer pointer arrays, must have
disjoint byte ranges between A's reduction and B's producer. The original
scratch-lifetime checker resolves these arguments. An independent CPU
journal audit checks the submitted spans too.

Two live-count cells are uploaded once per prefill. The producer records
each completion before its reduction waits on it. Main joins both reduction
completions before scratch reuse, status readback, commit or prefix capture.
Event and partial-rank submission failures poison the instance.

Compressor, KV, ring and index writes keep their full-chunk launch order
and arguments. Decode and verify keep their existing schedule. Short chunks
and diagnostic text taps use the sequential prefill path.

`CHOPINFER_DEEPSEEK41_PREFILL_OVERLAP=1` enables this candidate;
`CHOPINFER_DEEPSEEK41_PREFILL_OVERLAP=0` selects the sequential control.
The default remains sequential pending GPU bitwise checks and paired
per-chunk timings. CPU ordering and scratch checks are not GPU correctness
or performance evidence. This schedule covers attention reductions only;
embedding, MoE and whole-layer micro-batches require separate evaluation.
