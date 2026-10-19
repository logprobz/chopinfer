The accepted serving profile guarantees a fixed chunk grid (2048 or 8192
rows), every prompt end, and complete state after each chunk. It restricts
public prefill captures to those boundaries and publishes complete terminal
decode state for multi-turn reuse. The exact dependency cone is implemented
behind the `prefill_upper_skip` model option, which the default `tp4-ar` and
`tp4-dspark` profiles set. Their `-full-prefill` profiles keep full execution.

For the implemented 64-row outward alignment, the 32K equal-cost layer-row
estimate is 20.07% at chunk size 2048 and 42.20% at 8192. These are CPU
work-count estimates. On 2026-10-19 at TP4 with 8192-row chunks and 250 W, the
served skip matched the full schedule bitwise in tokens, logprobs and captured
state, and AR cold TTFT fell from 3.58, 18.97 and 117.4 s to 2.24, 10.84 and
70.5 s at 8K, 32K and 128K prompts.
The earlier fixed-schedule and all-choice analyses below remain derivations
of their respective contracts; their 44.7% estimate does not describe the
stronger policy that makes every actual chunk end complete.

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

The physical last-writer closure is implemented as an independent CPU
oracle in `models/deepseek41/prefill_cone.py`. It emits every layer's input,
output, window-KV and feature interval sets, together with each snapshot's
physical writer map. It starts from the authenticated Flash geometry and
does not read weights or run GPU operations. For example:

```sh
python3 -B models/deepseek41/prefill_cone.py --chunk 2048
python3 -B models/deepseek41/prefill_cone.py --chunk 8192
python3 -B -m unittest -v models.deepseek41.test_prefill_cone
```

The following counts use cold text prompts, one final logits row, prefix
caching enabled, and captures at N-2048 and N. AR means DSpark is unloaded.
The denominator for each upper-row fraction is 20*N. Input preparation and
compressed/index-key production at source 20 still cover every row.
"Output skip" counts attention-output and MLP rows; "KV skip" separately
counts window-KV rows, whose dependency intervals are larger.

| Prompt | Mode | Actual chunk cap | Upper output rows retained | Output skip | KV skip | Estimated prefill saving |
| --- | --- | --- | --- | --- | --- | --- |
| 8192 | AR | 2048 or 8192 | 47862 | 70.79% | 67.98% | 35.39% |
| 8192 | DSpark | 2048 | 69222 | 57.75% | 53.72% | 28.88% |
| 8192 | DSpark | 8192 | 47862 | 70.79% | 67.98% | 35.39% |
| 32768 | AR | 2048 or 8192 | 47862 | 92.70% | 92.00% | 46.35% |
| 32768 | DSpark | 2048 | 69222 | 89.44% | 88.43% | 44.72% |
| 32768 | DSpark | 8192 | 69693 | 89.37% | 88.30% | 44.68% |
| 131072 | AR | 2048 or 8192 | 47862 | 98.17% | 98.00% | 49.09% |
| 131072 | DSpark | 2048 | 69222 | 97.36% | 97.11% | 48.68% |
| 131072 | DSpark | 8192 | 69693 | 97.34% | 97.07% | 48.67% |

The estimates divide saved complete layer-output rows by 40*N. They assume
equal time per such row and no fixed overhead, and omit differences
between projection, attention, compressed-key production and copy costs.
They are a decision model, not a timing prediction with measured error
bounds. If 25% of TTFT were unaffected by these rows, the 32K DSpark
estimate would instead be about 33.5%. Under this model the optimization
still clears a 20% threshold unless unaffected time exceeds about 55%.
Applying 44.7% to the reported 594-token/s 32K baseline gives about 30.5 s
instead of 55.2 s, before overhead and any change in kernel efficiency.
That arithmetic does not authenticate the baseline or establish a speedup.

Both chunk caps matter. The profiles configure 2048-row chunks with an
8192-row step budget (`models/deepseek41/profiles.json:27-29,58-60`). The
host enables lone-prefill expansion when bundle limits exist
(`crates/runtime-host/src/command.rs:3781`), and the scheduler then uses
the step budget as its chunk cap
(`crates/runtime-scheduler/src/lib.rs:1960-1974`). At 32K a lone prompt
therefore has chunk ends 8192, 16384, 24576, 30720 and 32768. Its layer-20
output intervals are [22270,24562) and [28298,32768). The first interval
preserves six old snapshot cells from rows 24556..24561; these cells would
be missed by either final suffix. Configured 2048-row chunks instead need
layer-20 output interval [26290,32768). Higher layers narrow these sets;
layer 39's complete output is only row 32767 in both cases. Every layer's
full interval list is available from the oracle command.

The six CPU tests compare sparse and full integer fake-kernel schedules
with missing-row reads raising an error. They check final output and every
captured target-KV and DSpark feature value, old snapshot cells, disjoint
intervals, dependencies crossing chunk boundaries, short chunks, initial
history bytes, and the all-prompt-logits case. They do not establish GPU
bitwise equality.

An online implementation also has to specify what it knows about future
chunks. The current scheduler can switch between expanded, configured and
decode-aware chunks as traffic changes, and optimistic admission can
preempt a request before its planned captures. Boundary counts alone do
not promise a future chunk history. A safe alternative keeps the complete
persistent state exact at every actual chunk end. Treating every chunk end
as an additional boundary yields the following equal-row estimates for
both AR and DSpark (the feature roots then fit the target-ring cones):

| Prompt | Actual chunk cap | Upper output skip | Estimated prefill saving |
| --- | --- | --- | --- |
| 8192 | 2048 | 42.48% | 21.24% |
| 32768 | 2048 | 42.48% | 21.24% |
| 131072 | 2048 | 42.48% | 21.24% |
| 8192 | 8192 | 70.79% | 35.39% |
| 32768 | 8192 | 81.57% | 40.79% |
| 131072 | 8192 | 84.27% | 42.13% |

This additional-boundary policy preserves dynamic scheduling and permits
an unplanned snapshot. Its numerical estimates above condition on a
particular sequence of actual chunks. They are not a minimum saving over
all chunk and capture choices. An online executor could specialize to the
item already scheduled, preserving complete persistent state after it,
without promising the next item's size. Such specialization can recover
the larger savings when actual chunks are long, but gives no 20% saving
guarantee under arbitrary traffic and captures.

For the conservative envelope over all scheduler choices, a boundary may
occur at any positive multiple of the 64-token prefill granule. Shared
prefix capture uses the observed shared block count and rounds it to that
granule (`crates/runtime-scheduler/src/lib.rs:3479-3484,3514-3527`). The
capture ends a chunk through `chunk_remaining` and `chunk_tokens`
(`crates/runtime-scheduler/src/lib.rs:3455-3466,3592-3622`). These are real
state-consumer boundaries, not new logits roots. The envelope is a union
over separately possible boundary choices; it does not assume the default
server captures at every granule in a single request.

For any prompt row p, choose boundary
e=64*ceil((p+1)/64), clipped to the prompt end. Then p is in
[max(0,e-136),e), the copied target-ring interval at e. Thus the union of
possible C_l sets is [0,N) for every layer. Already at layer 39, that roots
every X_39 input, forcing every complete layer-38 output. Induction forces
every output at layers 20-38. The layer-20 compressed/index-key producers
and all lower layers were required in full already. DSpark's possible
raw-feature writers are also within [0,N), so they add no rows beyond
this envelope. Layer 39 still needs only its final complete output at
N-1; its window-KV and input-preparation branches run on every row.

Consequently, for DSpark plus prefix caching, the exact conservative sets
are:

```text
layers 0-19: every row, all ordinary work
layer 20: every row for compressed and index-key production
layers 20-39: input and window-KV rows [0,N)
layers 20-38: complete output rows [0,N)
layer 39: complete output row [N-1,N)
```

The skippable complete-output fraction is (N-1)/(20*N), and the equal-row
time estimate is (N-1)/(40*N). No upper-layer window-KV rows can be skipped.
This time estimate is optimistic for the omitted final-layer row work:
its input preparation and KV branch still run, while the model credits
each skipped complete output as a full layer-row. Prompt scoring would
root more final-layer outputs and reduce the saving further.

| Prompt | Conservative upper output skip | Conservative KV skip | Conservative estimated TTFT saving | Guaranteed 2048-chunk output skip / TTFT saving | Guaranteed lone-chunk output skip / TTFT saving |
| --- | --- | --- | --- | --- | --- |
| 8192 | 4.999390% | 0% | at most about 2.50% | 57.75% / 28.88% | 70.79% / 35.39% |
| 32768 | 4.999847% | 0% | at most about 2.50% | 89.44% / 44.72% | 89.37% / 44.68% |
| 131072 | 4.999962% | 0% | at most about 2.50% | 97.36% / 48.68% | 97.34% / 48.67% |

Here "lone-chunk" means the explicit 8192-cap chunk schedule with the
configured repeat and prompt-end captures used above, guaranteed in
advance. An idle queue observed once is not by itself such a guarantee.
On the reported 594-token/s 32K baseline, the conservative row estimate
would save about 1.38 s from 55.2 s, before overhead, rather than the roughly
24.7 s fixed-schedule estimate. These remain illustrative estimates.

Run the conservative calculation with:

```sh
python3 -B models/deepseek41/prefill_cone.py --contract all-choices
```

Two additional CPU tests verify the closed-form counts at all three prompt
lengths and check that the envelope equals the union of independently
planned single-extra-capture cones. This avoids treating a synthetic dense
capture schedule as evidence that every capture occurs in one real run.
The eight oracle tests and the existing scheduler shared-prefix and
granule tests pass on CPU. The conservative contract does not meet the
implementation threshold; the smaller fixed-schedule contract remains a
separate scheduling decision.

The implemented complete-chunk contract (2026-10-19)

Let an item's current chunk be [s,e). Earlier chunks have already committed
exact state. For each upper layer l, C_l is [max(s,e-136),e), the new rows
needed by every physical target-ring cell. Old cells outside this interval
remain untouched. Q_39 starts with requested logits rows, empty in an
intermediate chunk. Walking backward, expand Q_l outward to 64-row packed
boundaries, set K_l = align64(W(Q_l) union C_l), and require
X_l on K_l union F_l. W adds 127 preceding rows within the same item and
current chunk; older keys come from its exact retained prefix. That input
requirement becomes Q_(l-1). Union and alignment happen before proceeding
to the preceding layer. Packed-item windows never cross item boundaries.
This is `contract/src/prefill_cone.rs:127` and
`bundle/src/instance/execute/skip.rs:183` under `models/deepseek41`.

At layers 37-39, F_l contains the raw feature rows actually retained or
projected by the drafter. Chunks of at most 134 rows retain all their rows.
Larger chunks retain their final 128 raw rows, leaving the other six
physical history cells unchanged from the preceding exact commit. A large
chunk's direct projection additionally keeps every original projection
dispatch overlapping the final 128-row window. At a non-128-aligned prompt
end this can require up to 127 extra feature rows. Keeping the original
dispatch size avoids changing the GEMM selected at the 64-row threshold
(`bundle/src/dspark/transfers.rs:125`,
`contract/src/prefill_cone.rs:118`). Those extra rows are projected, then
overwritten in the ring, but are never copied into raw history. At aligned
2048/8192 grid ends, only the last 128 features are needed.

Layers 0-19 run completely. Layer 20's HC/input preparation, compressor,
index-key production and compressed-KV write also run completely
(`bundle/src/instance/execute/skip.rs:171`). Upper window producers run on
K_l. Complete attention and MLP outputs run on Q_l, with their small HC,
norm and quantizer prefix recomputed into local transient scratch. Carried
HC streams, selections, candidates and window metadata keep original packed
row addresses. Final streams and lagged pre coefficients are gathered in
ascending logits order before the head. No skipped hidden row reaches the
head. Engram remains entirely in the complete lower prefix.

The scheduler's grid and capture restrictions are in
`crates/runtime-scheduler/src/lib.rs:3596,3727`. The optional
`chop_bundle_prefill_api_v1` export declares the grid and accepts a bounded
envelope around unchanged batch bytes. Legacy tables and batch schemas
remain unchanged; a legacy execute call always runs the full schedule.
Scored/media requests and diagnostic per-layer taps use full execution.
The device-result executor path also keeps full execution until it has a
negotiated prefill entry; it still respects the declared capture grid.

For terminal text decode, the engine supplies the accepted token history
excluding its final sampled token, which has no KV yet. The scheduler
publishes that complete boundary as a public anchor
(`crates/runtime-scheduler/src/lib.rs:3123`). This covers autoregressive and
accepted speculative histories. Existing private preemption snapshots keep
their semantics. No intermediate historical decode boundary is synthesized.

| Prompt rows | Chunk rows | Upper output rows skipped | Upper window rows skipped | Equal-row TTFT estimate |
| --- | --- | --- | --- | --- |
| 8192 | 2048 | 40.1172% | 35.1563% | 20.0586% |
| 32768 | 2048 | 40.1465% | 35.1563% | 20.0732% |
| 131072 | 2048 | 40.1538% | 35.1563% | 20.0769% |
| 8192 | 8192 | 84.3750% | 82.8125% | 42.1875% |
| 32768 | 8192 | 84.4043% | 82.8125% | 42.2021% |
| 131072 | 8192 | 84.4116% | 82.8125% | 42.2058% |

These counts are identical for AR-only and DSpark at these aligned lengths:
the feature roots fit inside the target-ring roots. Without outward alignment,
the corresponding 32K equal-row bounds are 21.2402% and 42.5838%. The
implemented estimate is optimistic: layer 20's full producers, repeated input
preparation, sparse dispatch overhead and snapshot costs are not free. At
594 tokens/s, 32K takes about 55.2 seconds; proportional savings would be
11.1 seconds or 23.3 seconds. These are estimates, not performance evidence.

For a reusable prefix uniformly distributed across the 64-token residues of
a C-token grid, and with no retained prompt/response anchor between grid
points, additional recompute is 0,64,...,C-64. Its mean is (C-64)/2: 992
rows for C=2048, or 4064 for C=8192. Its maximum is 1984 or 8128 rows.
An exact retained prompt or response boundary loses no additional reuse.
The last sampled response token still needs one row of computation because
it has not been a model input; this is separate from the capture restriction.
The actual workload mean depends on where turns diverge and cache eviction.

Reproduce the implemented counts with:

```sh
python3 -B models/deepseek41/prefill_cone.py --contract guaranteed --chunk 2048
python3 -B models/deepseek41/prefill_cone.py --contract guaranteed --chunk 8192
```

The diagnostic build's opt-in capture hook records full physical snapshot
buffers, retained compressed/index pages and raw F32 logits. It records
state both at each chunk commit and after prefix capture catches up the
drafter. The GPU jobs must compare these bytes and served token/logprob bits,
then measure cold TTFT with capture instrumentation disabled. CPU dependency
and fake-host tests establish scheduling and address invariants only.
