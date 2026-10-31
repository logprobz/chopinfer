# Small-kernel candidate journals

The `verify_row_fusions`, `attention_merge_ceil` and
`router_prepare_fusion` candidate options are off by default, so every
existing journal, create sequence and device history must remain
identical. The change adds ten complete fake-device lifetimes with one
option on:

- `deepseek41-allocation-c4-rowfuse.journal` and
  `deepseek41-allocation-c8-rowfuse.journal`: the C4 and C8 allocated
  lifetimes (16- and 32-row verify steps) with `verify_row_fusions`. Each
  target layer's query norm and ceil become the fused query entry and its
  window RoPE, ceil32 and store the fused window entry, the drafter's
  queries fuse, and the drafter's window writes launch the parallel ceil32
  entry instead of the serial one (36 launches): 492 and 480 fewer
  launches.
- `deepseek41-allocation-c4-mergeceil.journal`,
  `deepseek41-allocation-c8-mergeceil.journal`,
  `deepseek41-flash-tp4.native-mergeceil.journal` and
  `deepseek41-flash-tp4.bf16-mergeceil.journal`: the same allocated
  lifetimes and the AR journal lifecycle of Flash TP4 at both KV layouts
  with `attention_merge_ceil`. Each attention heads merge whose WO_a folds
  row status, the target's and the drafter's, launches its quantizing twin
  and the sliced quantizer after it no longer launches (172 fewer launches
  in the allocated lifetimes, 160 in the AR journals).
- `deepseek41-allocation-c4-routerprep.journal`,
  `deepseek41-allocation-c8-routerprep.journal`,
  `deepseek41-flash-tp4.native-routerprep.journal` and
  `deepseek41-flash-tp4.bf16-routerprep.journal`: the same four lifetimes
  with `router_prepare_fusion`. Each target layer's router finish prepare
  and router F32 producer become one launch of the classifying producer
  (160 fewer launches).

The two kernel options load their family's cubin at create and resolve
its entries on every rank; no option adds an allocation. The
`verify_row_fusions` option changes only verify steps, so the AR journals,
whose steps are decode steps, would equal the default ones and are not
added.

The declaration [small-kernel-candidates.json](small-kernel-candidates.json)
names only these added journals. Existing fixtures have empty rules, so
the options cannot excuse any change to their launches, copies, resources
or ordering. Journal identity establishes structural preservation only;
the lab pairs, the fixed-batch GPU replays and the route sweeps supply the
numerical and route evidence.

On 2026-10-31 the journal proof of the fixture commit against the commit
before the candidates, with this declaration, was PROVEN: 29 journals
identical and the 10 declared additions, with 29 of 39 fixture histories
compared (the added fixtures have no base history).
