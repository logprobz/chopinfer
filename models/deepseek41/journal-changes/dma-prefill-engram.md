# Prepared DMA prefill gather adoption

`dma-prefill-engram.json` declares the exact positional history edits for
making prepared DMA Engram gathers the default. It adds no allocation,
handle, stream, event, module, resolve or arithmetic policy. The declaration
must be read at the adopted head; no wildcard operation edits are permitted.

The DSpark shared-prefix fixture prefills 4032 rows, captures and restores
state, then continues for 4160 rows and a 65-row tail. Its first two steps
now use prepared DMA gathers. The tail already used prepared gathers. The
pinned lifetime changes from 71670 lines and FNV 0x6419520edc320231 to 71698
lines and FNV 0x1fd8bd0d42749e1c.

The extra 28 records are four private-ID uploads, four prepared-gather ID
records, sixteen stream joins and four completions. All pre-existing
operations remain. Gather result uploads move from pre-compute staging to
their layer joins; independent decoder launches on different ranks change
their global interleaving. Every stream's complete launch sequence and
arguments remain identical. The sequence of all other lifetime, transport
and status operations is unchanged. The exact edit list retains each moved
operation's text and position.

All other pinned fixtures must compare with empty rules. The pinned AR
shared-prefix fixture has no mapped Engram tables, so its history remains
identical. Mapped AR DMA defaults are covered by the per-stream consumer
ordering, immediate/delayed completion, cancellation/reuse, refusal and
route-count tests. The default-route test fails on the previous default
with sixteen synchronous calls versus the required zero, then passes after
adoption. Served AR and DSpark comparisons establish full-logit/state and
output-bit equality, including the odd final DMA chunk.

The test-default audit removes all candidate overrides, the candidate-only
route sweeps and their two synchronous-DMA allowlist entries. The ordinary
AR and DSpark sweeps now exercise the adopted route. The reuse test counts
three prepared steps instead of two because its initial 8192-row chunk is
now prepared. Short-prefill graph tests retain their separate graph option
and require wide DMA steps to stay eager with prepared gathers. Option
listing tests remove one declaration and shift the later sorted indices;
other defaults and qualifications remain the same. The older mapped DMA
fixtures now require two enqueues before compute, eight joins, two
completions and all sixteen mapped planes at the joins.

Generate the layout spec against sealed integration-base and adopted-head
captures with `specs/layout_spec.py`, then run `proof.py`. The proof must
compare all twenty-three full histories and accept only this declared DSpark
change. Prove each untouched model with an empty identity spec. Numerical
and timing evidence is summarized in
[the mapped-row audit](../../../docs/mapped-row-read-ahead.md#adoption-evidence).
