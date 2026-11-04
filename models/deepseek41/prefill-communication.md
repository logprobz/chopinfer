# Served prefill communication

Flash TP4 prefill uses BF16 partials at both KV storages; decode, verify
and draft blocks keep F32 partials. The model selects the arithmetic and
eligible row intervals. `bundle_runtime::dma` owns byte transport, events,
the push-capability probe and draining. See
[partial arithmetic](communication-partials.md).

## Collective admission

Negotiated upper-skip, single-request text prefill uses ordered DMA where
its shape and capability checks pass, from 512 rows through 8192. This
includes DSpark target chunks. Smaller cone intervals keep the selected SM
collective. Media, unsupported schedules and configurations without upper
skip retain SM transport and allocate no unused DMA state.

The adapter recognizes the F32-partial/BF16-result phased collective and
the BF16-partial two-shot/turns entries, checking all four ranks' original
input/output intervals. It cannot choose a different partial precision.
The current native default reaches the BF16 payload path. The shared
transport also supports F32 payloads for callers with that contract.

Flash TP4 BF16-KV prefill takes the same BF16-partial collectives as
native, on the SM transport: upper-skip DMA stays native-only. Residency
admission is required on every rank at both storages, and a failure
refuses the instance. The phased F32 gather with BF16 output that BF16-KV
prefill used before is retired. Logits gather is separate and stays F32.
Protocol state resets only after a quiescent join, without resetting
unrelated collectives. Partial submission or failed draining poisons the
instance.

## Ordered transport

Four ranks own contiguous intervals of complete rows. Three permutations
scatter each peer's partial into a source-indexed inbox slot. The owner's
kernel adds ranks from positive F32 zero in ascending order and rounds once
to BF16. Three more permutations gather the finished owner intervals.
For BF16 payloads, sent F32 rows round once before copying; the owner rounds
its local F32 contribution the same way before summing. Changing owners
does not change the arithmetic.

Each rank reserves a 160 MiB inbox, a private allocation-owner stream, a
copy lane, eight phase events and three copy-completion events. Rounded
BF16 rows use the second half of that inbox. Input, reduced and embedding
buffers retain private lifetime owners so the host's automatic peer-copy
joins do not delay unrelated compute. All producers precede scatter, every
owner sum precedes gather, and the final join completes peer reads before
buffers or event records are reused.

At creation, `pushes` probes one 16-byte copy per ordered rank pair before
the inboxes contain live data. Only `InvalidArgument` on the first copy
selects pull mode; any later failure fails creation. Pass that result to
`execute_ordered` for the same rank set and host. Push mode is a caller-owned
boolean, not a second capability check. Pushes issue on the source's main
stream or copy lane; pulls issue on the destination's. Destinations wait
for each sender's copy before reading its interval.

Reuse an ordered rank set only with the same main streams, in the same
rank order, on every call. Its reusable events assume those stream histories;
a caller must not substitute streams or submit overlapping calls on that
state. The caller also owns poisoned-state refusal and error poisoning.
Before freeing graphs or private allocations, `drain` synchronizes main
streams, then copy lanes, then caller side streams. It stops at the first
failure, retaining resources under the instance's terminal-failure rule.

## Interval compute

Eligible gathered intervals feed the row-local HC expand, eight-row prefill
mix and cached norm-ceil chain as soon as each copy completes. The local
owner's consumers follow its sum; remote consumers wait for their own copy.
The row count must split into whole eight-row tiles, and unrelated chain
arguments cannot alias transport buffers. Embeddings, odd chunks and other
chains remain serial.

Eligible MoE route sums produce and round the sent intervals in scatter
order, then produce the local interval last. This requires a BF16 payload,
32-row owner intervals and exact output bindings. Each copy waits for its
source's round, and pushed copies also wait for the destination to reach
the call. The attention projection's TMA producer stays whole.

At an ordinary 8192-row default chunk, 40 reductions per rank split their
consumer chains and 36 split their MoE producers. Smaller cone calls use
the same admission checks. The final peer-read join is unchanged. CPU copy
coverage and vector-clock audits check all compute/DMA spans and reject
missing waits and omitted intervals.

DMA prefill still completes synchronous Engram uploads before the first
compute launch. Prepared per-layer gathers are used on supported non-DMA
schedules; [Engram qualification](lab/prefill-engram.md) does not claim their
integration with DMA. The removed full-schedule DMA and overlap experiments
have no execution switch or serving profile.

## Measured outcomes

On 2026-10-19 at 250 W, phased BF16 gather preserved served logits, tokens,
KV, prefix state and restored continuations at 8192, 24576, 32767 and 32768
tokens. At 8192 rows and width 5120 it halves gather traffic to 62,914,560
bytes per rank per reduction, saving 5,096,079,360 bytes over 81 reductions.
Its sending schedule also differs, so timing is not a payload-only ablation.

On 2026-10-20 at 250 W, sparse DMA matched all 40 raw-state artifacts
against the SM transport. On 2026-10-21 the BF16-payload build matched those
artifacts and every token/logprob in a quiet TP4 AR ABBA comparison. TTFT
improved by 3.6 ms at 8K and 8.9 ms at 32K. Per-interval MoE production then
kept all bytes and lowered TTFT from 0.718 to 0.707 s at 8K and from 2.921
to 2.881 s at 32K.

The 2026-10-21 four-rank push/pull probe at 250 W matched every copied row
at 8192, 8191, 2496, 1600 and 576 rows. Three full-row copy phases took
1.21 ms pushed against 1.59-1.95 ms pulled; this is transport timing, not
request throughput. Shared extraction on 2026-10-27 preserved all 18
DeepSeek and 67 GLM journals, including ordering and failure-path audits.
