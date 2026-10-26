# Prepared AR prefill gather declaration

`ar-prefill-engram.json` permits no edits to the existing pinned journals or
their device lifetimes. Ordinary pinned AR fixtures use a host without mapped
Engram tables, and pinned DSpark fixtures already prepare prefill gathers.
Removing the selector therefore requires an identity proof for every pinned
create, journal and full device history. No pin or kernel lock changes.

The adoption adds two rank-0 ID allocations and two prepared layer handles
to mapped AR instances. There is no pinned mapped-AR lifecycle journal, so
this declaration does not prove those additions. The mapped-host prefill
tests separately check pre-KV reservation, exact hash bytes, consumer joins,
delayed completion, cancellation, reuse and release. Served paired tests
also compare full logits and committed state bitwise; the measured domain
and results are in [the prefill lab note](../lab/prefill-engram.md).

The frozen journal proof tools read sealed base and head captures. Run
`specs/operation_edits.py` in compact form against this declaration and
require zero proposed edits. Generate a layout spec with `specs/layout_spec.py`
and prove it with the declaration read at the head. Also generate an empty
identity spec with `specs/identity_spec.py` and run `proof.py --require identity`
to require every identity checklist item. Repeat after rebasing; do not add
an allowance to hide a changed history.
