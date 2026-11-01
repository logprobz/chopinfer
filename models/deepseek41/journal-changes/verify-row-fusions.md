# Verify row fusions journals

The `verify_row_fusions` candidate option is off by default, so every
existing journal, create sequence and device history must remain
identical. The change adds two complete fake-device lifetimes with the
option on:

- `deepseek41-allocation-c4-rowfuse.journal` and
  `deepseek41-allocation-c8-rowfuse.journal`: the C4 and C8 allocated
  lifetimes (16- and 32-row verify steps). Each target layer's query
  norm and ceil become the fused query entry and its window RoPE, ceil32
  and store the fused window entry, the drafter's queries fuse, and the
  drafter's window writes launch the parallel ceil32 entry instead of the
  serial one (36 launches): 492 and 480 fewer launches. The option adds
  no allocation, module or resolve.

The option changes only verify steps, so the AR journals, whose steps are
decode steps, would equal the default ones and are not added. The
`attention_merge_ceil` and `router_prepare_fusion` candidates that were
pinned beside it were retired with their journals.

The declaration [verify-row-fusions.json](verify-row-fusions.json) names
only these added journals. Existing fixtures have empty rules, so the
option cannot excuse any change to their launches, copies, resources or
ordering. Journal identity establishes structural preservation only; the
lab pairs, the fixed-batch GPU replays and the route sweep supply the
numerical and route evidence.

On 2026-11-01 the journal proof of the fixture commit after the two
retirements against the commit before the candidates, with this
declaration, was PROVEN: 29 journals identical and the 2 declared
additions, with 29 of 31 fixture histories compared (the added fixtures
have no base history).
