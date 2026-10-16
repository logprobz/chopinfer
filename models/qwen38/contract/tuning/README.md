# Geometry tuning

A `tuning: true` entry in `geometries/supported.json` loads `dense.json`,
`moe.json` and `schedules.json` from `contract/tuning/<id>/`. Extra geometry
directories load their tables from `tuning/<id>/` beside `supported.json`.
`pinned_names: true` preserves the legacy module names and encoder ownership.
It is independent of measured tuning. Flash currently has both flags.

Geometry selection does not imply tuning or qualification. An unmeasured
geometry tries measured geometries in increasing distance in hidden width,
expert count, intermediate width and top-k. Dense K segments and skinny
splits require their K granules to divide the new K; otherwise the model
uses one K chain. Dense tiles use the shared GEMM defaults. The F32 head
always uses one K segment to preserve exact logits across row buckets.

Cluster warp blocks preserve a measured setting only when every bucket in
its group accepts it: [1, 2, 4] or [8, 16, 32]. Otherwise each group searches
1 through 8 and takes the smallest legal value. A failed search refuses the
build with geometry, role, K and bucket group before catalog generation.

MoE decode inheritance checks all exact-verify buckets together, including
an equal K split. If any bucket refuses, all use the next measured geometry
or the model-chosen default (four warps, one K split). The model's prefill
default uses two weight stages and two activation stages. Every choice must
pass engine validation. Template schedule distance uses the row's shape
fields for the same operation, weight format and bucket. Inherited rows
must pass the shared template's complete legality, resource and reduction
order checks; otherwise the engine's formula schedule is validated and used.

Build reports label only unmeasured entries with tunable knobs. Labels name
the inherited geometry and row or the model/engine default actually used.
Rows, attention, communication and bundle entries without choices have no
label. Untuned entries need independent numerical and performance evidence
before qualification.

`GeometryModel::hc_precompute()` reports whether all four HC producers pass
`hc_precompute_spec`. The build also emits `GEOMETRY_HC_PRECOMPUTE` beside
`GEOMETRY_CUBINS`. A bundle selecting a geometry must force HC-off when false,
including when padded or precomputed mixing was requested. The existing
qualification requires one strip and two warp blocks. The small fixture has
four strips, so it has no qualified precompute module. Extending this path
requires separate GPU numerical evidence.

Extra geometries are fixture inputs enabled through
`CHOPINFER_QWEN38_GEOMETRY_DIRS`. Kernel-lab refuses them before CUDA compilation.
Release bundle builds also refuse them, except explicit recipe tests setting
`CHOPINFER_QWEN38_RECIPE_TEST=1`. Default-run tests generate a second geometry's
catalog entries only in memory; it is never registered or built to cubins.
