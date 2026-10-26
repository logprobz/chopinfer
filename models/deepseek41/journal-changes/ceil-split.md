# Ceil-split journal declaration

The test default now loads the served one-row ceil-split module for Flash
TP4. The declaration names four module loads and their six resolves per
fixture, with exact creation positions. Fixture-scoped activation requires
the module lock entry to remain byte-identical. The affected full lifecycle
histories add only the corresponding release records; retained launches
and operands remain exact under the ordered identity map.

Creator declarations were derived from sealed adjacent-commit captures,
constrained to the resources above. `tools/journal_proof/specs/operation_edits.py`
generated the compact exact operation edits. The edits were reviewed
against source-derived history audits. `layout_spec.py` and `proof.py`
read this declaration from the flip commit and check all pinned journals,
creation records and device history through each last pinned checkpoint.
