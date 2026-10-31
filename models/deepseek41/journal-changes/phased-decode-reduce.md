# Phased decode reduction journals

The `phased_decode_reduce` candidate option is off by default, so every
existing journal, create sequence and device history must remain
identical. The change adds four complete fake-device lifetimes with the
option on:

- `deepseek41-flash-tp4.native-phased.journal` and
  `deepseek41-flash-tp4.bf16-phased.journal`: the AR journal lifecycle of
  Flash TP4 at both KV layouts, on residency that holds the phased grids.
  Their two-row decode graph launches the one-shot round entry for the four
  complete all-reduces per rank (16 launches) and keeps the composed pair
  for the 77 whose round the next HC boundary takes.
- `deepseek41-allocation-c4-phased.journal` and
  `deepseek41-allocation-c16-odd-phased.journal`: the C4 and 65-row C16
  allocated lifetimes. The 16-row verify steps defer the round at 77 of each
  rank's 81 all-reduces and launch the round two-shot for the other four;
  the 65-row steps launch it, with workers, for all 81 (648 launches over
  two cycles), and launch no reduce-scatter or round row.

Each instance also allocates and zeroes the phased reduction's comm state
on every rank at create (inbox, flags and epoch: 12 allocations and 12
zeros), and with native KV, whose prefill does not otherwise load the
phased F32 module, resolves its two round entries on every rank. Each
complete reduction launches one entry instead of three, so the AR
journals launch 32 fewer kernels. Prefill keeps its reductions.

The declaration [phased-decode-reduce.json](phased-decode-reduce.json)
names only these added journals. Existing fixtures have empty rules, so
the option cannot excuse any change to their launches, copies, resources or
ordering. Journal identity establishes structural preservation only; the
lab parity test, the fixed-batch GPU replays and the route sweep supply the
numerical and route evidence.

On 2026-10-31 the journal proof of the fixture commit against the commit
before the candidate, with this declaration, was PROVEN: 25 journals
identical and the 4 declared additions, with 25 of 29 fixture histories
compared (the added fixtures have no base history).
