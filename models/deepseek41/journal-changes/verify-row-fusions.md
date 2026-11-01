# Verify row fusions journals

The `verify_row_fusions` candidate option is off by default, so every
existing journal, create sequence and device history must remain
identical. The option changes verify steps and the drafter's window
writes, which its catch-up selects by row count wherever it runs: in
DSpark steps, prefill steps included, and, outside any target step, in a
prefix capture, which catches the drafter up before it copies the
snapshot (`instance/prefix.rs`). The change adds four complete
fake-device lifetimes with the option on:

- `deepseek41-allocation-c4-rowfuse.journal` and
  `deepseek41-allocation-c8-rowfuse.journal`: the C4 and C8 allocated
  lifetimes (16- and 32-row verify steps). Each target layer's query
  norm and ceil become the fused query entry and its window RoPE, ceil32
  and store the fused window entry, the drafter's queries fuse, and the
  drafter's window writes launch the parallel ceil32 entry instead of the
  serial one (36 launches): 492 and 480 fewer launches.
- `flash-tp4-dspark-shared-anchor-rowfuse.journal`: the DSpark shared
  prefix lifetime (a 4,032-token prefill to a prefix endpoint, capture,
  restore into another slot, two more chunks). Its capture has nothing to
  catch up; the first chunk after the restore catches the drafter up and
  quantizes the 52- and 12-row window pieces it writes (three of each on
  every rank) with the parallel ceil32 entry, 24 launches, and its 64-row
  pieces, above the draft rows, with the serial one as before. Every
  other line is unchanged.
- `flash-tp4-dspark-prefix-capture-rowfuse.journal`: an eager DSpark
  lifetime whose capture catches up a three-token prompt, then a restore
  into another slot and a drafted 6-row verify step. Up to that step it
  equals the lifetime without the option but for 12 window writes inside
  the capture, which take the parallel ceil32 entry
  (`dspark::catch_up::prefix_capture_catch_up_quantizes_small_writes_with_the_row_fusions`
  checks this against the option-off run); the verify step fuses as in
  the allocated lifetimes.

The option adds no allocation, module or resolve. AR instances have no
drafter and run no verify step, so the AR journals would equal the
default ones and are not added. The `attention_merge_ceil` and
`router_prepare_fusion` candidates that were pinned beside it were
retired with their journals.

The declaration [verify-row-fusions.json](verify-row-fusions.json) names
only these added journals. Existing fixtures have empty rules, so the
option cannot excuse any change to their launches, copies, resources or
ordering. Journal identity establishes structural preservation only; the
lab pairs, the fixed-batch GPU replays and the route sweeps supply the
numerical and route evidence.
