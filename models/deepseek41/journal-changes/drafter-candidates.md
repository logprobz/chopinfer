# Drafter candidate journals

The five drafter candidate options (`dspark_markov_staged`,
`dspark_hc_fusion`, `dspark_confidence_staged`, `dspark_moe_direct` and
`dspark_hc_draft_rows`) are off by default, so every existing journal,
create sequence and device history must remain identical. The change adds
six complete fake-device lifetimes of `allocated_lifetimes_journal`. Each
runs one drafter forward per rank over the lifetime's items: five Markov
positions, three stages' routed experts and five stage boundaries that can
fuse (every boundary but the one into the head). Counts below are
over the four ranks.

- `deepseek41-allocation-c8-markov-staged.journal`: the C8 lifetime (40
  block rows) with `dspark_markov_staged`. Its 20 Markov steps launch the
  staged step, which reads each item's row in place, instead of the row
  copy and the tiled step: 20 fewer launches. Each rank resolves the staged
  module.
- `deepseek41-allocation-c8-hc-fusion.journal`: the C8 lifetime with
  `dspark_hc_fusion`. Its 20 stage boundaries each drop the lagged mixer,
  input norm, quantizer, round and expansion (and, into a stage's MoE, the
  widening) for one column-block producer and one split finisher, and keep
  their collectives: 72 fewer launches.
- `deepseek41-allocation-c8-confidence-staged.journal`: the C8 lifetime with
  `dspark_confidence_staged`. Its 20 confidence heads launch the staged
  entry instead of the parallel one, from a module each rank resolves.
- `deepseek41-allocation-c8-moe-direct.journal`: the C8 lifetime with
  `dspark_moe_direct`. Its 12 routed-expert runs take the direct path at 40
  rows: the row-bias route, the queue metadata, the gate/up queues and
  finisher and the down queues and finisher (8 launches) replace the route
  select, count and scatter, the intermediate quantizer, the eight expert
  chain launches and the routed sum (13 launches): 60 fewer launches.
- `deepseek41-allocation-c4-hc-draft-rows.journal`: the C4 lifetime (20
  block rows) with `dspark_hc_fusion` and `dspark_hc_draft_rows`. The
  drafter table admits 20 rows, so its 20 boundaries fuse as the C8
  lifetime's do: 72 fewer launches.
- `deepseek41-allocation-c4-drafter.journal`: the C4 lifetime with all five
  options. It combines the changes above at 20 rows: 152 fewer launches and
  two more modules resolved on every rank.

The declaration [drafter-candidates.json](drafter-candidates.json) names
only these added journals. Existing fixtures have empty rules, so no option
can excuse any change to their launches, copies, resources or ordering.
Journal identity establishes structural preservation only; the lab bitwise
pairs, the fixed-batch GPU replays and the route sweep supply the numerical
and route evidence ([drafter cost](../lab/drafter-cost.md)).

On 2026-10-31 the journal proof of the fixture commit against the commit
before the candidates, with this declaration, was PROVEN: 29 journals
identical and the 6 declared additions, with 29 of 35 fixture histories
compared (the added fixtures have no base history).
