# Prefill BF16 gather journal declaration

The test default now prepares the served rounded-gather route for the
supported Flash TP4 geometry. The declaration activates the unchanged
locked rounded-gather module and adds three aligned protocol buffers per
rank: a 184549376-byte inbox, 33800-byte flags and a four-byte epoch.
Creator positions, initialization and release records are exact. The
source audit checks all four ranks, allocation identities and zeroing
before use; retained operations remain ordered and unchanged.

Creator declarations were derived from sealed adjacent-commit captures,
constrained to the resources above. `tools/journal_proof/specs/operation_edits.py`
generated the compact exact operation edits. The edits were reviewed
against source-derived history audits. `layout_spec.py` and `proof.py`
read this declaration from the flip commit and check all pinned journals,
creation records and device history through each last pinned checkpoint.
