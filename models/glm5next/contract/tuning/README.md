# Geometry tuning

A geometry's schedule table is `tuning/<id>/schedules.json` in the shared
`chopinfer-tuning-v1` form (`crates/kernel-build/src/tuning.rs`): one row per
operation, weight format, per-rank shape and token bucket, in canonical
order. An extra geometry directory (`CHOPINFER_GLM5NEXT_GEOMETRY_DIRS`) keeps
its tables in `tuning/<id>/` beside its `supported.json`.

Flash's rows are hand-picked defaults: the grouped MoE engine's row tile, 16
for the 32-token decode bucket and 64 for the 8192-token prefill bucket. An
offline sweep replaces a row only under `kernel_build::tuning::accepts`, with
bitwise-equal output against the row it replaces. Only the grouped MoE engine
has a schedule choice today; the latent attention, indexer, mHC and KDA
engines derive theirs from the bucket.

A geometry without a table tries the tuned geometries in increasing distance
in hidden width, expert count, expert intermediate width and top-k, and takes
the first row for the same operation and bucket whose knobs are legal at its
own shape; otherwise it uses the model default (the same tiles).
`model_glm5next_contract::sm120::tuning::entry_status` labels every such choice
with its source. Untuned entries need their own numerical and performance
evidence before qualification.
