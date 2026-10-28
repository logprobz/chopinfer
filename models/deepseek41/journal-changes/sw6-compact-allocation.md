# Compact confidence allocation

The [declaration](sw6-compact-allocation.json) describes the adoption of
computed-row target heads and logits gathers whenever confidence allocation
runs. Sampling retains the reserved row order through a device index map.
The candidate switch and the unused reserved-head path are removed.

## Declared changes

Each rank replaces separate item, rule, live-item and reserved-count
allocations with persistent metadata storage and removes the unused F32
reserved-head buffer. Calibration bytes are uploaded when they change.
Target heads and logits gathers use computed rows; feature gathers retain
the reserved layout needed by commit. Allocated depth-three batches keep
all five drafter heads, including successor reuse.

The five allocated lifecycle fixtures cover widths 12, 16, 32, 65 and 192,
two cycles, accepted commits, destruction and balanced releases. The
shared-prefix and six verify-HC histories also declare their changed
create/warm-up metadata and handle numbering. Unrelated operations retain
their ordering and typed resource mapping. No module payload or kernel
lock replacement is declared.

## Candidate-on comparison and commit boundary

The adoption commit combines the extension above 64 computed rows with the
default flip and path removal. This follow-up uses the companion-note
alternative authorized at review. The source-bound declaration and
passed GPU smoke already identify that combined implementation. Splitting
it now would replace that qualified source identity and require separate
intermediate resource declarations for a policy that was not GPU tested.
This series preserves that provenance and records the two effects below.

A separate fake-host comparison on 2026-10-28 enabled the candidate in the
pre-adoption source after wide mapped ABI validation, then compared it
with adoption. At widths 12, 16 and 32, normalized journals differ only by
four unused 3,932,160-byte reserved-head allocations and their four releases.
All remaining recorded operations agree, including head routes, metadata,
sampling transfer and five-head drafting. Thus the default flip adopts the
existing candidate-on program for these measured widths, with unused
storage removed.

At widths 65 and 192 the old candidate still selected its reserved-head
fallback. Their differences are the intended route extension: remove the
reserved-head gather, score computed rows, gather logits to the sampler,
and use mapped result transfer. Width 65 reduces the head's row grid from
12 tiles to 9; width 192 retains 24 tiles. These wide comparisons establish
the intended differences, not candidate-on identity.

The [candidate-on declaration](sw6-compact-candidate-on.json) reproduces
these observations using a sealed test-only control and the sanctioned
full-history proof. The control enables the existing option only in the
five allocated lifecycle fixtures; all other fixtures keep their original
default. For the three narrow fixtures, the declaration removes four unused
allocations and four releases, with no other operation edits. The wide
fixtures declare the head, gather and result-transfer changes explicitly.

Both declarations compare authenticated histories with exact edits and
typed handle mapping. They do not establish GPU numerics or unrecorded host
state; upload-hash and bare-release limits remain those of the journal
proof. The candidate-on control is separate from production history.

## Exactness boundary and host admission

The adopted fixed-batch GPU smoke passed on 2026-10-28 at 12, 32 and 65
computed rows. Greedy and fixed-seed runs compared every matched request's
full-vocabulary logits, proposals, token/logprob samples, per-request
acceptance and committed state, with repeat controls. The widest tested
GPU width is 65. Widths 66-192 have CPU coverage only, and widths above 64
remain unqualified for timing.

TP4 packages declare `chop.mapped-device-results.v1`. A host without that
capability is refused during package admission, before native creation or
weight loading. Existing C ABI tables and symbols are unchanged. Execute
retains its defensive sink checks for callers that bypass package admission.
