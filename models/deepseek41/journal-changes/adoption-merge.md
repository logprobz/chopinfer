# Adoption merge journals

The integration branch takes three adoptions as merge commits, in this
order: the verify row fusions
([verify row fusions adoption journals](verify-row-fusions-adoption.md)),
`f32_row_tiles` ([exact F32 row tiles by default](f32-row-tiles-adoption.md))
and the MoE knob set
([direct MoE knob set as the served route](moe-direct-knobs.md)). Each lane
proved its change against its own base. This note records how the merges
resolve what two adoptions both change, and the proof that the merged
tip's journals are the integration base's with exactly the three declared
changes, composed where they meet.

## Conflicts

"Merge the verify row fusions adoption" has no conflicts. Since the lane
branched, the integration base changed only the tuner and the model
README, so the merge has the adoption's bundle sources.

"Merge the f32_row_tiles adoption":

- The model README keeps the integration rows for the options not adopted
  and drops the `f32_row_tiles` candidate row, as the lane does.
- The module comment of `conformance/wide.rs` takes the f32 side (its
  tiles profile sentence removed) and keeps the row fusions sentence.
- Pins both adoptions move: the five C4 lifetimes (default, depth 3,
  phased, HC draft rows, drafter options), verify-hc-4 and -6, and the
  DSpark long-context and failed-template digests. Each merged line count
  is the base count plus both adoptions' deltas (C4: 69334, +0 and +116,
  is 69450; verify-hc-4: 7880, -480 and -4, is 7396), with the merged
  journal's digest. The prefix capture pin, which only the row fusions
  side holds, merged at its value and moves with `f32_row_tiles` by 32
  lines (40136).

"Merge the MoE knob set adoption":

- The README drops the four `moe_direct_*` candidate rows.
- Both adoptions retire their options, so the contract's declarations
  (`execution_options.rs`, 13 left), the geometry checks (`geometry.rs`:
  22 options, the later options' indices, no `moe_direct_` option left)
  and the compiler's package helper remove both. The bundle's option
  fields (`config.rs`, `instance.rs`, `shape.rs`) merge cleanly the same
  way.
- 35 pins move with both: every TP4 allocation lifetime, verify-hc-4 to
  -32, the TP4 native and BF16 Flash journals and their phased variants,
  media TP4, both shared anchors, the four template digests and the
  prefix capture. Each count is the `f32_row_tiles` merge's plus the
  knob set's delta; the prefix capture, which the knob lane does not
  have, moves by 124 lines like every completed DSpark lifetime (40260).

No kernel lock or golden file changes in the merges.

## Proof

Each step compares two sealed captures of the bundle suite with the
journal proof. The chain runs from the integration base to the merged tip
through the row fusions lane's own captures and two scratch commits: the
row fusions merge with the `f32_row_tiles` narrow configure commit on top,
and a scratch merge with the same tree as the `f32_row_tiles` merge. The
last step's head is a capture of the merged tip.

| Step | Base | Head | Spec | Verdict |
|---|---|---|---|---|
| S1a | integration base | "Pin the default prefix capture lifetime beside the row fusions one" | base identity, the lane's five journals added | PROVEN: 37 identical, 5 declared added |
| S1b | that commit | "Serve the decode row fusions at verify steps of at most six rows" | [verify-row-fusions-flip.json](verify-row-fusions-flip.json) | PROVEN: 34 identical, 8 proven (2,544 removed and 1,092 added operations, 24 graph handle records each way), 42 of 42 histories, full history proof |
| S1c | the flip | "Retire the verify_row_fusions option, its overrides and profiles" | [verify-row-fusions-retire.json](verify-row-fusions-retire.json) | PROVEN: 38 identical, the 4 option-on lifetimes declared removed |
| A1 | the option commit | row fusions merge with the narrow configure commit | identity with the module replacement of [f32-row-tiles-narrow-configure.json](f32-row-tiles-narrow-configure.json) | PROVEN: 34 identical, 4 proven (16 configure changes), 38 of 38 histories, full history proof |
| A2 | that commit | the `f32_row_tiles` merge | [adoption-merge-f32-row-tiles.json](adoption-merge-f32-row-tiles.json) | PROVEN: 5 identical, 33 proven (1,788 added and 864 removed operations, 1,240 added and 128 removed handle records), 38 of 38 histories, full history proof |
| B | the `f32_row_tiles` merge | the merged tip | [adoption-merge-moe-direct-knobs.json](adoption-merge-moe-direct-knobs.json) | PROVEN: 5 identical, 33 proven (3,840 launches renamed by the four kernel rules, 1,728 added operations, 1,620 added handle records), 38 of 38 histories, full history proof |

Together the steps show that the merged tip's journals are the
integration base's 37, each with the declared changes of every adoption
that touches it, in merge order, and the prefix capture, which the row
fusions lane adds, with the `f32_row_tiles` and knob set changes on it.
Only the five TP1 and TP2 journals are identical through every step.

The two merge declarations are the `f32_row_tiles` and knob set lanes'
declarations regenerated for the merge's positions with the lanes' own
tools, from captures of each merge's parents. Each has its lane's module
activations, handle counts and operation edits in the 32 fixtures the
lane declared, and the same change in the prefix capture fixture: the
decode tile module and its eight vocabulary head decode launches, and the
knob module loaded on each rank with its 15 entries resolved per rank and
released at destroy. The proofs read them at the commit that adds them
(`--declaration-revision`).

## Template digests

The long-context and failed-template digests in
`prefill_plan_templates.rs` are outside the sealed journal set. A
temporary, uncommitted patch wrote those journals out of tree copies of
the integration base, each adoption's tip and the merged tip; every dump
reproduced its revision's pins. With function, stream and event handle
numbers named (functions by kernel and resolve occurrence, streams and
events by first appearance), each adoption's edits of the base journals
were applied to them. Where two adoptions rewrite the same line, they
only renumber handles (buffers, mapped gathers and graphs), and the line
takes the sum of their shifts; where one replaces a launch the others
renumber, it takes their buffer renumbering. No other edits overlap, and
the merged tip's six journals (the AR and DSpark long-context lifecycles
in both verification modes and both failed templates) equal the
composition exactly. The failed templates show each adoption's own
change: the knob set adds 60 resolves (15 per rank; the destroy fails
before any release), `f32_row_tiles` 16 resolves and 4 configures, and
the row fusions move 12 two-row drafter window writes to the parallel
ceil32.

## Checks

On 2026-11-01, at the merged tip: the bundle suite in its sealed capture
(725 tests, the 4 listed ignored); the contract (172), kernels-sm120 (728),
kernel lab families (142), compiler (75), runtime-host (859) and its fake
driver (672) tests; clippy and fmt; the leak, bundle boundary and
model-neutral guards; the quick route sweeps of all nine profiles and the
full sweeps of `tp4-dspark` and `tp4-dspark-rows192-wide` (32 pieces
each). The data volume had about 90 to 120 GiB free, so every capture ran
with `--start-free-gib 80 --minimum-free-gib 60`.
