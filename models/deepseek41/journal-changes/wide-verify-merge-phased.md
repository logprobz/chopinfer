# Phased decode reduction over the merged wide verify branch

The merge of the [phased decode reduction](phased-decode-reduce.md) into
the wide verify branch adds the same four journals with the
`phased_decode_reduce` option as on its own branch, at the same pins: the
Flash TP4 AR lifecycles at both KV layouts, the C4 allocated lifetime
(69358 lines) and the 65-row C16 allocated lifetime (189194 lines). The
option is off by default and no `-wide` fixture sets it, so the two
`-wide` journals keep the pins of the first parent.

On 2026-10-31 the journal proof of the merge commit against the first
parent, the wide verify branch, with the branch's own spec
[phased-decode-reduce.json](phased-decode-reduce.json), was PROVEN: 25
journals identical and the 4 declared additions, with 25 of 29 fixture
histories compared (the added fixtures have no base history).
