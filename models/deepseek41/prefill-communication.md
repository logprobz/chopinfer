# Prefill communication

Native Flash TP4 now defaults to BF16 input partials for prefill only; see
[communication-partials.md](communication-partials.md). Set
`CHOPINFER_DEEPSEEK41_BF16_PARTIALS=0` to retain the F32-partial arithmetic
described below. The DMA transport carries the partial format of the
collective it replaces. Overlap remains off by default and must be bitwise
against its selected sequential path.

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
The default remains sequential because the measured attention-only overlap
gain is below 10 ms per chunk. Mode 1 covers attention reductions only.

## Two parts across sublayers

Experimental mode `CHOPINFER_DEEPSEEK41_PREFILL_OVERLAP=2` also overlaps
embedding and MoE reductions. It admits one TP4 AR text request, starting
on the 64-token granule, with 1024 to 8192 rows. A ends at the midpoint
rounded up to that granule; B holds the remainder. Both parts fit within
4096 rows, including an odd final chunk. Other cases use mode 1's schedule.

Embedding lookup, RoPE tables and window-key lists are prepared for the
original chunk. Main computes each sublayer for A then B. The communication
stream reduces A then B, in the same sublayer order. Main waits for A's
reduction before A's next sublayer and for B's before B's next sublayer.
Thus A's reduction can overlap B's compute, and B's can overlap A's next
compute. The last hyper-connection expansion and the logits epilogue remain
full-row operations after both parts rejoin main.

`Lowering::prefill_row_origin` distinguishes `None`, the original full
chunk, from `Some(first)`, including A's `Some(0)`. Residual streams, mixer
coefficients, partials, reduced outputs, status, selections, candidates and
precomputed per-row metadata keep global row coordinates. Other compute
scratch keeps local row zero and its prepared capacity. It is reused only
on main. Token IDs, positions and item IDs are input subspans; each part has
its own immutable live count. Decode and verify ignore the origin.

Each layer gathers its old window prefix once, before A. The staged window
retains A's writes while B reads it. Compressor, compressed KV, index and
ring operations execute A then B on main, so B observes A's completed writes.
The original chunk's window bounds remain in use. The aligned split never
opens a ratio-2 compression group between A and B. Selections and candidate
metadata retain both global row ranges for later reuse layers; reindex
reads the part's count, while dense index blocks retain their original
64-row boundaries.

The resolved-span audit checks A's reduction against B's compute in every
block, and B's against A's compute in the next block. No shared byte range
is allowed across either boundary, including backend workspace. CPU tests
also inspect the submitted journal, retained selection and window views,
odd endpoints after offset 24576, prefix snapshot lifecycle and poisoning.
Prefix capture remains a committed-chunk operation; no internal part
endpoint becomes a new capture point. All communication joins precede the
existing status, commit and capture paths.

The original mode passed full-chunk GPU bitwise and prefix-hit comparisons.
It remains experimental because its original wall time was slower. CPU
ordering checks alone do not establish numerical equality or a speedup.

## BF16 gather after the rank sum

Flash defaults to the existing phased F32 rounded transport for target
prefills at TP4, with whole 1024-element hidden-row granules and at least
512 rows per call. `CHOPINFER_DEEPSEEK41_PREFILL_BF16_GATHER=0` retains the
composed F32 scatter, F32 gather and BF16 round. Value `1` opts other
compatible geometries into the transport. The overlap switch selects the
compute schedule independently and still defaults to sequential execution.

Every owner adds F32 partials from zero in ascending rank order, converts
its final sum once with the same BF16 conversion, then gathers BF16. The
embedding broadcast and attention/MoE HC expansion already consume BF16.
HC streams are BF16 even though the expansion computes in F32; its
coefficients, the rank partials, final head/norm and logits remain F32.
The separate F32 logits gather is unchanged.

At 8192 rows of width 5120 and TP4, each rank's gather payload falls from
125829120 to 62914560 bytes per reduction. Across 81 hidden reductions,
that saves 5096079360 bytes per rank per chunk. The existing phased entry also changes the sending
schedule and uses 16 senders plus 32 workers, so its timing is not a pure
ablation of payload width.

Create prepares a separate operation and protocol-state range for the
optional target prefill path. Every rank must admit both phased grids
against reported residency; missing, restricted or MPS reports retain
the composed fallback. Decode, verify and the drafter retain their
original operations and state ranges. Protocol tags reset only after a
quiescent join and well before they repeat, without resetting the composed
transport. A failed drain or partial rank submission poisons the instance.

CPU checks cover all 81 reductions, odd rows, phase isolation, admission
fallback, exact reset spans and the 644 micro-schedule scratch regions.
On 2026-10-19, Flash TP4 at 250 W passed exact served token/logprob and
full-vocabulary F32 comparisons against the sequential control and its
repeat. Prefix snapshots, committed KV bytes and restored continuations
matched at 8192, 24576, 32767 and 32768 tokens. The paired F32/BF16/BF16/F32
GPU envelopes were 3571.3/3452.6 ms at offset 0 and 5731.5/5645.3 ms at
offset 24576. Wall medians improved from 3762.2 to 3627.4 ms and from
5829.1 to 5759.1 ms. No Xids were recorded. The automatic route is limited
to the measured Flash geometry and retains the residency fallback.

The attention-only overlap changed the GPU envelope by less than 10 ms
per chunk. Its second half projection overlaps the first reduction, but
only about 15 ms of independent projection work is available across all
layers. The initial two-part schedule increased the envelope about 150 ms.
Its 4095/4096-row parts missed the exact F32 tiled routes, and HC mixing
slowed during concurrent communication. The exact row routes have since
passed bitwise GPU checks and now select the measured existing tiled entries.
Both overlap modes remain explicit experiments.

## Ordered DMA transport

An unset `CHOPINFER_DEEPSEEK41_PREFILL_DMA` selects DMA for the negotiated
upper-skip executor only (see [sparse DMA admission](#sparse-dma-admission)):
the create record's prefill profile must negotiate upper-skip chunks and the
BF16 gather must be prepared. Package defaults are 8192-row chunks with
upper-layer skip, so they take it. Legacy create records without prefill
options, the explicit full-prefill profile and chunked profiles without skip
allocate no DMA inbox, owner stream or events. DSpark instances take the
upper-skip default for their target prefill chunks. Full-schedule and
two-part executions keep the SM transport unless `1` is set, and DSpark
instances keep it there even then. `0`, or any
value other than `1`, turns DMA off everywhere. Negotiated skip uses its own
executor. `CHOPINFER_DEEPSEEK41_PREFILL_BF16_GATHER=0` selects the original
F32 transport when DMA is off, and also turns off the upper-skip default.
Overlap remains off by default, and `CHOPINFER_DEEPSEEK41_PREFILL_OVERLAP=0`
retains sequential work.

`CHOPINFER_DEEPSEEK41_PREFILL_DMA=1` selects a separate transport for single
text-request Flash TP4 autoregressive prefills. It applies to sequential
execution and the two-part schedule, with at least 512 rows per reduction.
Shorter parts retain their previous transport. Attention-only overlap,
media, packed requests, decode, verify and DSpark keep their existing path.
This explicit setting also works with BF16 gather disabled and retains
the existing experimental row coverage. The
original and integrated transports passed served and raw-state comparisons.

Each rank owns a contiguous interval of complete rows. Three ordered rank
permutations copy F32 partials into local slots indexed by source rank.
An ordinary local kernel calls the existing phased transport's sum helper:
positive F32 zero, sources in ascending rank order, then one BF16 rounding.
Three ordered permutations gather those BF16 owner intervals. Changing the
owner interval does not change any element's arithmetic. Cross-device
events join three points: every partial before any scatter, every owner sum
before any gather, and all final peer reads before reuse. Each rank issues
its scatter copies, its sum and its gather copies on its own stream, so the
permutations keep their order there without a barrier between them. On
2026-10-20 at 250 W, a four-GPU probe of this schedule against one with a
barrier after every permutation matched every output element and guard at
8192, 8191, 4096, 2496 and 640 rows, and took 3.921 against 3.963 ms per
8192-row reduction, about 3.3 ms less per upper-skip chunk.

Where the step's collective reduces BF16 partials (the two-shot and turns
entries of the BF16-partial contract), the transport carries BF16 partials
instead. Each rank first rounds the rows it sends, every row outside its
own interval, once to BF16 with the phased BF16 entries' rounding helper.
The scatter copies those BF16 rows, half the F32 bytes, into BF16 slots.
The owner sum calls the phased BF16 sum helper: its own F32 rows rounded as
sent, every source's BF16 value widened, positive F32 zero, ascending rank
order, one BF16 rounding. The gather is unchanged. The output is the
replaced collective's bits. The rounded rows use the second half of the
same inbox allocation, so the BF16 payload allocates nothing more. The
payload is chosen from the replaced launch's symbol on every rank; a
collective the transport does not recognize keeps the step on the
collectives.

Each rank keeps a private 160 MiB inbox and eight events. A and B reuse the
inbox only after the previous collective's global join. The overlap audit
includes complete DMA input/output intervals and every inbox, including
remote source spans that the local sum kernel does not name. A CPU journal
audit independently reconstructs happens-before from stream order and
event records/waits; removing waits must expose a compute/DMA alias.

The host's explicit peer-copy service validates context ownership, directed
peer access and pool grants before using the driver peer-copy operation.
Private completion events retain both allocations and join both allocation
owners. Resource release and access revocation drain outstanding copies.
The bounded queue retires completed copies and fences its oldest transfer
under backpressure, retaining leases until completion.
Partial submission failures poison the model instance.

On 2026-10-19 at 250 W, the isolated four-rank transport passed every output
element and guard at 4095, 4096, 8191 and 8192 rows, including cancellation,
signed zero, subnormal and rounding-tie inputs. Its 8192-row median was
3.947 ms per reduction, or an estimated 319.7 ms for 81 reductions. Nsight
confirmed peer DMA traffic and owner-local sum kernels. This estimate
excludes serving interactions; the served comparison remains required.

The 2026-10-19 served comparison at 250 W passed exact tokens/logprobs,
full-vocabulary F32 output, prefix snapshots, committed KV and restored
continuations for all four BF16/DMA and sequential/two-part variants.
At offset 0, BF16 sequential and DMA sequential GPU envelopes were
3442.2 and 3251.4 ms; wall times were 3533.4 and 3371.1 ms. DMA two-part
execution reduced the envelope to 3086.4 ms but took 3557.0 ms wall time.
At offset 24576, sequential DMA reduced the envelope from 5637.5 to
5427.5 ms and wall time from 5692.4 to 5500.8 ms. No Xids were recorded.

The two-part host cost came mainly before the first GPU launch. Its strict
scratch audits resolved and classified the same compute pointer arguments
inside nested span comparisons. The audits now resolve each launch once
and retain every intersection check, including peer arrays and tensor-map
bases. They do not need const-input classification because even read-only
aliasing is forbidden across these prefill branches. The decode shared-expert
audit retains its const-input checks. A release CPU fixture reduced part
construction and DMA auditing from 405.9 to 37.6 ms. This changes no launch,
event, copy, arithmetic or mutation order.

### Private allocation ownership

With DMA enabled, each rank allocates `partial`, `reduced`, `embedded` and
its inbox on a private lifetime stream. This stream performs no compute or
transport work. The qualified host still queues completion waits on both
allocation owners after a peer copy; those waits now land on lifetime
streams instead of delaying unrelated rows on compute. Using the busy
communication stream as the allocation owner would also introduce avoidable
copy dependencies, so the lifetime stream is separate from both lanes.

The existing buffers total 480 MiB per rank at 8192 rows and width 5120.
Their ownership changes without adding tensor storage or staging copies.
There is one additional CUDA stream per rank. Allocation completion is
established before create returns. Every explicit producer/consumer event,
range-disjointness audit and final peer-read join remains in place. Before
normal destruction releases graphs or buffers, the bundle drains compute,
communication and shared-expert streams, including after a nonpoisoned step
error. A failed drain releases no buffer; the SDK makes failed destruction terminal
and retains resources under the existing lifecycle rule.

Six CPU ordering tests cover the qualified implicit joins, independent row
parts, subsequent reuse, allocation readiness, final compute reads and the
memory accounting. The model host journal records actual allocation owners
and inserts the qualified host's implicit copy waits; it checks all peer
regions, creation and destruction fences, failed drain retention and the
existing scratch audit. The ownership regression fails with compute-owned
buffers.

### Integrated qualification and routing

On 2026-10-19, the integrated runtime at 250 W passed every full-vocabulary
F32 row, snapshot, KV page and restored continuation at 8192, 24576, 32767
and 32768 tokens. Baseline repeat, sequential DMA and DMA with two parts
matched the sealed BF16 baseline. All resources were released, and no Xids
were recorded. Fresh-server ABBA profiles also matched served token and
logprob bits. GPU exclusive attribution includes peer copies.

For sequential DMA, GPU envelopes at offsets 0 and 24576 fell from
2014.0/2142.1 ms to 1877.1/1980.4 ms, gains of 6.8%/7.6%. Exclusive GPU time
fell from 2014.0/2142.1 ms to 1858.2/1960.7 ms. Communication unions fell
from 525.7/553.8 ms to 343.0/343.1 ms. The measured full-chunk route is the
automatic choice; its inbox adds 160 MiB per GPU relative to BF16 transport.

Two-part DMA hid about 404 ms of communication but increased its paired GPU
envelopes from 2024.2/2131.6 ms to 2302.7/2423.0 ms. The 4096-row halves
lose the exact-8192 paired attention route. In the first candidate repeat,
attention exclusive time rose from 232.2 to 1101.2 ms at offset zero. This
loss exceeds the hidden communication, so two-part overlap remains opt-in.
Fast attention for 4096 rows needs separate qualification.

The audit overhead fix is visible in the new trace: the first micro plan
interval is 73.5-77.0 ms rather than the old 392.9 ms. Micro wall minus GPU
is 156.3 ms versus its current baseline's 153.8 ms at offset zero, and
118.6 versus 116.3 ms at the late offset. Its current loss comes from GPU
work rather than the old audit delay. These are different integration heads,
so the plan-interval difference is not a paired wall-speed claim.

With concurrent builds, sequential DMA upload time varied from 249.9 to
832.1 ms across the first-chunk repeats. Wall medians were 2485.7/2334.0 ms
for DMA versus 2125.0/2207.8 ms for BF16. CUDA traces place the long gaps
inside host upload work between driver calls, with short transfer calls.
The GPU-time route decision does not establish a quiet-host TTFT gain.

### Sparse DMA admission

The exact upper-skip schedule takes DMA for native Flash TP4, single-request
text prefill with overlap disabled, both by default and with
`CHOPINFER_DEEPSEEK41_PREFILL_DMA=1`; `0` retains the SM transport. This
includes a DSpark instance's target prefill chunks: their Engram rows,
feature captures and feature packing run as ordinary launches on each
rank's main stream, and every DMA call joins all main streams on its final
gather before the next launch. The CPU journal of a DSpark target chunk keeps
every compute and feature launch with its operands and replaces only the 73
phased reductions. The inbox costs 160 MiB per rank of KV pages on DSpark
profiles too. Under package defaults the unset and explicit settings produce
the same create, execute and release journal, for AR and DSpark instances. The adapter recognizes only
the F32-partial, BF16-result phased contract and the BF16-partial two-shot
and turns entries, and carries their payload. It validates every rank's
original input and output row interval before substituting a DMA call.
It does not select a partial format or change any compute dispatch, KV mutation,
ring update, index write or head-row copy. Intervals below 512 rows retain
the composed transport.

Every call uses the shared owner-sum kernel and the runtime's qualified
peer-copy service. The owner sum calls the same rank-ordered F32 block sum
as the phased kernel, with one final BF16 conversion. All peer reads join
before compute or the next call can reuse partials, outputs or inboxes.
The journal audit checks every compute and DMA span against stream/event
ordering. Removing the waits must expose an alias. Tests cover full and odd
chunks, smaller cones, cached attention composition and resource release.

As with full-chunk DMA, sparse DMA uses synchronous Engram uploads. Selecting
both experimental switches must finish every mapped row upload before the
first compute launch; the async Engram plan cannot bypass those uploads.
The execution entry rejects simultaneous DMA and async-gather plans.

The earlier isolated sparse qualification on 2026-10-19 passed all 40 raw
state artifacts and served token/logprob bits. At offset 0, communication
exclusive time fell from 316.15 to 200.90 ms and the GPU envelope from
1361.82 to 1278.20 ms. These are historical measurements on a different
integration. The trace analyzer checks owner sums against incoming scatter
copies and outgoing gather copies on all ranks; independent outgoing
scatters can legally overlap an owner's sum.

On 2026-10-20 at 250 W, this port on round 7 passed the raw-state comparison
of DMA=1 against DMA=0: all 40 artifacts (8 full-vocabulary F32 logit rows,
16 rank snapshots and 16 committed KV files at 8192, 24576, 32767 and 32768
tokens, including restored continuations) were byte-identical, with no live
resources left and no Xids. That job stopped at its first served arm on a
pinned KV page count, so the TTFT ABBA and Nsight profiles of this port
remain to be measured, now against the unset default.

That evidence, and every earlier DMA measurement here, used F32 partials.
Native Flash TP4 prefill now defaults to BF16 partials, which the transport
carries as BF16. Its bitwise and TTFT qualification against the
within-bound head is pending; the BF16 scatter moves half the F32 bytes, so
the earlier timings do not carry over.
