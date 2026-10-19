The proposed 2560-row upper-layer prefill cutoff is only partly valid.
Flash has a finite dependency cone above its final compressed-KV source,
but that cone alone does not preserve the bytes of the current prefix
snapshot when DSpark is enabled. This analysis stops before changing the
ABI, scheduler, bundle, kernels or snapshot format. No TTFT improvement is
established.

The analysis was checked on 2026-10-19. Repository paths below are relative
to the repository root. Vendor paths refer to
`$CHOPINFER_DEEPSEEK_CHECKPOINT/inference/`. The authenticated sources are
listed in `models/deepseek41/reference.py:72-84`; the adapter identifies
their vendor revision there. The CPU source-authentication test passed.
The pinned `model.py:1244-1275` computes every prompt row through every
backbone layer. It selects the final row in the head (`model.py:1010-1014`).
It does not implement an upper-layer last-128-row prefill shortcut. Its
last-128 copies in `model.py:709-716` and `model.py:1046-1054` retain KV rings
after projecting the input rows. They are not such a shortcut. An alleged
shortcut in a different vendor revision is not evidence about this pin.

The layer and state dependencies are as follows. Positions and layer IDs
are zero-based. A boundary token count N follows row N-1.

| Consumer | Required producer and temporal extent | Source anchors |
| --- | --- | --- |
| Backbone window attention at layer l, row p | Layer l's normalized attention inputs at rows max(0,p-127)..p, including the current row | Vendor `model.py:410-424,701-721,766-791`; `models/deepseek41/contract/src/latent_glue.rs:45-49` |
| Compressed KV and index keys | Sources 2, 8 and 14 pool pairs; source 20 stores one entry per input row. Every completed entry can be selected by a later query, so retain all source inputs and incomplete pair tails | `models/deepseek41/geometries/flash.json:40-108`; vendor `model.py:440-485,536-558,740-764`; `models/deepseek41/contract/src/state.rs:137-150,203-241` |
| Upper-layer attention's compressed branch | Layers 20-39 read source 20's KV and index keys. Layers 24, 28, 32 and 36 reindex; they do not write new historical keys | `models/deepseek41/contract/src/indexer.rs:22-38`; `models/deepseek41/bundle/src/program.rs:237-256`; vendor `model.py:496-518,723-764` |
| Candidate lists and reused selections | Source 20's candidates, and the latest index source's selected positions, for the same query row. Query rows do not pool with one another | Vendor `model.py:551-611`; `models/deepseek41/contract/src/indexer.rs:22-38` |
| Hyper-connections | All four streams and the lagged pre-mix at the same row. Mix, collapse, expansion, normalization and MoE add no temporal edge | Vendor `model.py:950-996`; `models/deepseek41/contract/src/hc.rs:1-8`; `models/deepseek41/bundle/src/lower.rs:76-100` |
| Engram | Token-ID n-grams through length four and the current row's streams, at layers 1 and 14 only. No upper-layer hidden-state history | `models/deepseek41/geometries/flash.json:112-123`; vendor `engram.py:160-174`, `model.py:350-365,1252-1266`; restore reconstructs token history in `models/deepseek41/bundle/src/instance/prefix.rs:177-189` |
| DSpark | Concatenated means of the target's pre-attention streams at layers 37, 38 and 39. Main projection, norm and each drafter stage's KV projection are per row; the drafter reads the last 128 committed main rows and its full five-position draft block | Vendor `model.py:1267-1269,1034-1069,1125-1138`; `models/deepseek41/bundle/src/lower/dspark/ops.rs:231-261,365-416`; `crates/kernel-core/src/latent_visibility.rs:10-19,52-82` |
| Prefix capture | All physical target-ring entries, compressor-tail entries and DSpark raw-feature entries, plus the header. Compressed KV and index keys already occupy shared serving pages. Drafter rings are rebuilt after restore | `models/deepseek41/contract/src/state.rs:154-186,228-268`; `crates/bundle-runtime/src/state.rs:56-67,555-560,596-625`; `models/deepseek41/bundle/src/instance/prefix.rs:115-145` |

Here is the dependency closure for a proposed schedule. Let X_l(p) include
the streams entering layer l and the lagged pre-mix needed to collapse
them. Let R_l be the rows of X_l needed by some consumer. Let C_l be the
rows whose window KV at layer l must be retained or copied. Let F_l be
required raw-feature rows at a DSpark target layer, empty elsewhere.
For a set S, define W(S) as the union of [max(0,p-127),p] for every p in S.
Ignoring already available source-20 pages, the upper-layer closure is:

```text
R_40 = every requested logits row
Q_l  = R_(l+1)                         complete layer-output rows
K_l  = W(Q_l) union C_l                window-KV producer rows
R_l  = K_l union F_l                   layer-input rows, l = 39 down to 20
```

Also retain a query row's candidate and selected-position lists at their
source layers whenever a later reuse layer needs them. These are same-row
edges; they do not extend the backward temporal radius. Each complete
layer's attention and MoE run on Q_l. Its window-KV branch runs on K_l.
Computing only a uniform suffix of complete layers without the earlier
KV producers would read missing rows at its leading edge.

Source 20's compressor and index-key branches require X_20 for every
prompt row regardless of this closure. That forces all of layers 0-19,
including their Engram and stream-mixing work, to run on every prompt row.
Layer 20's collapse, normalization and compressed-key production also run
on every row. Its attention output and MoE can, in principle, use Q_20.
This follows the stage ordering in
`models/deepseek41/bundle/src/program.rs:221-262,291-303`.

With only a logits root b=N-1, the closure is
R_l=[max(0,b-(40-l)*127),b]. It contains 2541 input rows at layer 20 and
2414 complete layer-20 output rows. Adding a boundary's last 128 window
keys fits the same cone. Preserving all 136 physical target
ring cells increases that to 2549 input rows, still covered by rounding to
2560. Target-ring headroom is eight cells
(`models/deepseek41/contract/src/state.rs:27-30,132-135`). Source 20's input
is already available for every row; these numbers describe the backward
cone, not a license to omit its compressed-key producers.

The live DSpark feature window alone fits inside this cone. At a boundary
it roots X_37, X_38 and X_39 at [max(0,N-128),N-1]. The three draft stages
all project those raw main features; they do not run three additional
causal passes over the prompt. The target features are attention inputs,
not final target hidden states. The current commit implementation does
nevertheless read every accepted row of a large prefill chunk to seed the
drafter rings (`models/deepseek41/bundle/src/dspark/transfers.rs:110-148`,
`models/deepseek41/bundle/src/instance/dspark/launch.rs:162-175`). A future
skip implementation must remove obsolete projection reads as well as
target computation. Dropping target writes alone is unsafe.

The physical DSpark snapshot invalidates the proposed bound. Its raw
feature history has 128+6=134 cells, addressed by absolute position modulo
134 (`models/deepseek41/bundle/src/dspark/state.rs:45-52`,
`models/deepseek41/contract/src/serving.rs:17-20`). For a chunk larger than
134 rows, commit projects every row but retains only the last 128 raw
features (`models/deepseek41/bundle/src/dspark/state.rs:468-477`,
`models/deepseek41/bundle/src/dspark/transfers.rs:154-170`). Six physical
cells therefore retain earlier values.

`Capture::LastRows { rows: 128 }` describes what a restored request reads;
it does not restrict the byte copy. The shared snapshot copies the entire
134-cell entry (`crates/bundle-runtime/src/state.rs:56-67,596-625`). The
bundle calls that path directly after drafter catch-up
(`models/deepseek41/bundle/src/instance/prefix.rs:123-145`). The compact
mapping helper `dspark::transfers::snapshot_copies` is not used by this
prefix callback. Catch-up writes derived KV, not the six stale raw cells.

A CPU probe compiled the existing `dspark/state.rs` and `dspark/transfers.rs`
and used their actual `retain_copies` and `staged_projection` functions.
It labeled each raw row with its absolute producer position and replayed
2048-row commits from zero. It observed:

| Capture token count | Six older raw-feature rows still in the copied entry | Earliest required layer-20 output for target-39 feature | Proposed suffix start |
| --- | --- | --- | --- |
| 6144 | 4000..4005 | 1714 | 3584 |
| 8192 | 6048..6053 | 3762 | 5632 |
| 30720 | 28576..28581 | 26290 | 28160 |
| 32768 | 30624..30629 | 28338 | 30208 |

For example, X_39(28576) depends on layer-20 outputs X_21 as early as
28576-18*127=26290. The scheduler's normal cold 32K capture set includes
30720 and 32768 with a configured 2048-row chunk size: it captures the
start of the final chunk and the prompt end
(`crates/runtime-scheduler/src/lib.rs:3469-3485,3529-3549`). The union of
2560-row suffixes starts at 28160. Thus the first capture has dependencies
outside the union, even when both capture boundaries are supplied.

These six rows are outside the raw window used to rebuild the drafter
after restore. They matter here because the requested invariant includes
bitwise-identical captures. The probe demonstrates a missing dependency
in the proposed schedule, not a measured logit mismatch on checkpoint
weights. Preserving only subsequently consumed semantic state would be a
different invariant.

For exact current snapshot bytes, F_l must include the last writer of
every copied physical history cell at every capture, including these old
rows. The general derivation is therefore a last-writer pass over the
actual chunk commits, followed by the closure above. At a large commit,
update writer[p % 134] for its final 128 positions; at a short commit,
update it for every committed position. A capture roots every initialized
writer in that array at each of the three target layers. Preserve initial
cell bytes too. Unlike the live-state cone, this row set depends on the
chunk history as well as the boundary list. Zeroing or compacting unused
cells would change today's snapshot bytes and is not an exact substitute.

Chunking does not reset a dependency cone. Intersect the global closure
with each chunk and keep the preceding keys needed by a retained query in
a later chunk. The current staging assumes all keys in its interval were
populated (`models/deepseek41/contract/src/latent_glue.rs:100-111`), so
merely omitting upper-layer launches for an early chunk is insufficient.
The host also needs to include every requested prompt-logprob row, not
only the final prompt row (`crates/runtime-scheduler/src/lib.rs:3569-3575`).
With all prompt rows scored, the corresponding output roots cover the
whole prompt. Optimistic admission can introduce a further private capture
when preempting a live request (`crates/runtime-scheduler/src/lib.rs:2933-2946,
3001-3034`); a fixed-boundary scheme must account for that possibility.

The recommended basis for subsequent work is the physical last-writer
closure, since it preserves the stated snapshot-byte invariant. Relaxing
that invariant to live restored state would allow a simpler bounded
closure, but requires an explicit acceptance change. Neither approach
has been implemented or GPU-qualified here. The earlier 34%, 46% and 51%
figures are not validated TTFT predictions: capture locations, old physical
cells, prompt scoring, and the cost of the retained branches all affect
the savings. This documentation change saves no serving time. There is no
before/after GPU job because there is no candidate implementation.
