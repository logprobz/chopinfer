# Drafter cost candidates over the merged wide verify branch

The merge of the [drafter cost candidates](drafter-candidates.md) into the
wide verify branch adds the branch's six allocation lifetimes with drafter
options on (the C8 lifetime with each of `dspark_markov_staged`,
`dspark_hc_fusion`, `dspark_confidence_staged` and `dspark_moe_direct`, the
C4 lifetime with `dspark_hc_fusion` and `dspark_hc_draft_rows`, and the C4
lifetime with all five) at the branch's pins. In the merged allocation
test they are its twelfth to seventeenth instances, after the wide drafter
fixtures. The options are off by default, so every other fixture keeps its
selections.

The merged tip's capture first failed on the drafter profile's route sweep:
the head's six cached-stage allowances, which the profile matched through
`tp4-dspark*` on the drafter branch, name the DSpark profiles one by one
since the row tile work. The fix (5f192c6e03) adds `tp4-dspark-drafter` to
those six entries and changes no journal.

On 2026-11-01 the journal proof of that fix against a2a66aff0f, the last
earlier commit that changes the bundle, was PROVEN: 31 journals identical
and the 6 declared additions, with 31 of 37 fixture histories compared (the
added fixtures have no base history). Its specification was the identity
specification of a2a66aff0f's capture with the six allocation fixtures and
their journals declared added, as in
[drafter-candidates.json](drafter-candidates.json), at devices 12 to 17.
