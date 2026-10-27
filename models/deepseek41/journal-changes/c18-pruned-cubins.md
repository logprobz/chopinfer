# Prune unregistered entries from mixed cubins

The package keeps the registered split quantizers, ordered head, matrix
prefill projections and latent cache entries. It removes 118 unlaunched
entries from five mixed cubins:

| Family | Retained entries | Removed entries |
| --- | ---: | ---: |
| ceil_split | 2 | 4 |
| f32_decode_last | 6 | 18 |
| mxfp8_prefill_matrix | 30 | 78 |
| latent_decode_cache | 18 | 6 |
| latent_cache_groups | 12 | 12 |

The exact module names, before and after lock digests, and removed symbols
are in `c18-pruned-cubin-symbols.json`. The matrix catalog retains the wide
Engram projection registered directly by verification. The original full
ceil and matrix cubins remain authenticated model-lab references; their
lock and provenance pairs are unchanged.

Removing entry wrappers changes these five whole-cubin digests. Every
retained entry keeps its text, constant bytes, shared-memory extent, ABI,
resource record and launch attributes. The ELF audit permits only the
constant-bank symbol index to move within the per-entry information
section, and verifies that it still references the same constant section.
No other retained ELF field or section changes.

`c18-pruned-cubins.json` declares only the exact teardown release positions
of the removed symbols. An external probe observes each typed Symbol in
the parent fake host release callback, records its symbol and next note
ordinal, and emits the original note unchanged. All four parent lifecycle
tests pass their pins. Each observation uniquely matches its symbol ID and
release note in the authenticated parent capture. The frozen
`operation_edits.compact` function generates the explicit position lists.

The proof binds both revisions' module locks and the exact removed resolves,
then reads the layout declaration at the head. Every retained launch, grid,
operand, copy, event and remaining history operation must match under typed
handle mapping. All 16 pin constants are checked against the observed journals. The 11
changed constants, including both shared-prefix anchors, are regenerated;
the two TP4 aliases inherit the same pins. Positions and pins are specific to this parent and must be
regenerated after any rebase.
