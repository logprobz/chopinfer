# MiMo geometry tuning

`moe.json` and `schedules.json` map geometry IDs to measured tables. The
schedule rows retain their operation, weight, shape and bucket keys. IDs
must name listed geometries. The catalog contains every listed geometry at
TP4 and TP2; only v26-flash owns the historical names and projection aliases.

An untuned geometry tries measured MoE tables in shape-distance order.
Every decode, exact-row, verify and prefill variant at both ranks must pass
before a table is accepted. Otherwise the shared fragment defaults go
through the same checks. Exact-row variants retain bucket 1's reduction.
Equal shape keys share one engine schedule and must agree across measured
geometries; conflicting tables are refused. Template schedule inheritance tries rows with the same operation, weight
format and bucket, nearest shape first. Each row must pass `validate_row`,
including reduction order, engine legality and launch resources. Engine
schedule defaults pass that check too.

Dense roles inherit the v26-flash settings when the complete typed request
passes for every emitted bucket. Decode chooses one kernel for all buckets
1, 2, 4, 8, 16 and 32. If the measured kernel refuses any bucket, all use the
validated whole-K default. Prefill segments divide K into whole 128-column
blocks; the fallback is one validated segment. A refusal names the geometry,
rank count, role, N and K. `geometry::validate_catalogs` exposes these checks
for compiler admission. Build reports label tunable entries with the source
row or default selected. These labels do not change kernel identities.

The current fragment and route domain still fixes hidden width at 4096,
256 experts, top-k 8 and expert intermediate width 2048. Additional fixture
geometries can change other dimensions inside the existing engine domain.
They are refused by the model's kernel-lab hooks and release bundle builds.
A release recipe test alone may set `CHOPINFER_MIMO26_RECIPE_TEST=1`.

A generated `geometry-cubins.rs` records each cubin's geometry and rank count.
Rank zero serves both topologies of that geometry. The common transport and shape-independent medium-row GEMMs
use geometry `*`. Nondefault encoder declarations are not built yet; their
absence in membership must be respected by bundle consumers. Catalog and
CPU build coverage do not qualify a new geometry numerically or measure its
performance.
