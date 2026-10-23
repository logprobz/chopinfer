# DSpark synthetic lowering

`Prepared` binds the target-feature mean, feature packing, main projection,
F32 normalization, trained quantizers and native KV operations. `block`
reuses the backbone's ratio-zero attention and MoE lowering with each MTP
stage's checkpoint views. The synthetic geometry has the same 32 experts
and top-k for backbone and drafter. A different drafter expert set refuses
until it has its own prepared MoE operations.

`metadata` always describes five rows per drafting item, regardless of K.
Every query sees its committed 128-cell window and all five provisional
positions. `Metadata::slots` maps requests to committed state;
`Metadata::staged_slots` is the identity mapping into the per-item draft
windows. Upload both in separate buffers. The initialized bonus/noise
tokens, positions, text-router bias and position-row indices are independent
of K. Each position selects its draft by the shared sharded argmax, with
lower-token-ID ties. In `DraftMode::Sampled`, the serving default, a
sampled item's Markov logits first take the target sampler's noise
(`kernel_core::draft_noise`): `x / T + g` with the request's seed and the
generated index of the draft position, from the host's V6 draft sampling
records (`chop.draft-sampling.v1`). Greedy items' logits keep their bits,
so their drafts equal `DraftMode::Greedy`'s. `CHOPINFER_DSPARK_COUPLED=0`
drafts every item greedily.

The caller uses these operations in order:

1. Bind each committed projection chunk from `dspark::features` and
   `dspark::transfers`. `catch_up` applies `main_proj`, `main_norm` and the
   stage's KV projection, then writes the native committed ring.
   `catch_up_stages` projects rows gathered from several slots through
   every stage with one main projection. Check all row status words before
   acknowledging a projection cursor.
2. Copy the committed physical window into a separate per-item window with
   `rings::stage`. Reserve five additional cells. Upload `metadata`, run
   `initialize` once, then run every stage's `block` in order. Use
   `stage_views` to alias MTP weights into the backbone lowering namespace.
3. Bind `HeadViews::streams` to the final draft streams and `pre` to the
   last draft mixer's F32 collapse weights. `Heads::vocabulary` first calls
   the shared `SigmoidWeightedF32` weighted head into `hidden_f32`, then
   normalizes a separate F32 copy and projects vocabulary logits. Its raw
   hidden output must have separate storage from streams, pre weights,
   normalized rows and logits; overlapping bindings are refused.
4. `Heads::position` processes positions 0 through 4 sequentially
   on every rank, starting from the bonus IDs. In sampled mode bind
   `PositionViews::noise` to one noise record per drafting item. Allocate one immutable
   `position_rows` buffer per position. Initialize the candidate-pair planes
   and both collective states, and use normal collective segmentation.
   Set `PositionViews::hidden_f32` to the same `HeadViews::hidden_f32` output.
   Confidence reads this original F32 collapse before RMSNorm.
5. `dspark::staging::Adapter` reads five IDs and score words per drafting
   item, hashes the bonus plus K proposals, gathers mapped Engram rows and
   uploads each rank's verify inputs. Its `Program` callbacks dispatch draft
   and target programs and commit feature copies. Each callback checks
   device status before returning success. Use one Engram history owner;
   do not also commit the target-only Engram staging path.

Target `DsparkFeature` taps call `Prepared::feature` before attention. It
writes target-major BF16 mean rows. `feature_indices` and `pack_features`
transpose these into row-major concatenated feature rows. Packing splits
at the shared gather capacity while preserving full prefill capacity.
Upload one device count per chunk from `feature_pack_counts(capacity, live)`.
Only committed target rows enter retained history. Prefix restore and slot
retirement must update both the feature lifecycle and Engram history.

The fake-host tests dispatch actual prepared launches for catch-up and
complete blocks at synthetic TP1, TP2 and TP4. Head journals cover every
compiled topology; mapped Engram journals force acceptance depths 0 through
5, cancellation, K=0 intervals, K=1 through 4, restore and slot reuse. They
verify bindings and ordering, not GPU arithmetic or model output parity.
