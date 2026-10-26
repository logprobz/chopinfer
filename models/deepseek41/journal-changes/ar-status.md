# AR status packet journal declaration

The test default now uses the served AR status packet. The declaration adds
one rank-zero packet allocation and one event per peer, at their exact
creation positions. The packet has 1280, 1536 or 2048 bytes at one, two or
four ranks. Each read joins the peer streams, copies the status fields and
performs one download. Exact operation ordinals replace the previous
individual synchronizations and downloads, including release records.

The source audit checks packet offsets, copy lengths, peer joins and
readback order. Ordinary lifecycle fixtures exercise five reads; media
fixtures exercise three. All other operations retain their order and
ordered resource identities.

Creator declarations were derived from sealed adjacent-commit captures,
constrained to the resources above. `tools/journal_proof/specs/operation_edits.py`
generated the compact exact operation edits. The edits were reviewed
against source-derived history audits. `layout_spec.py` and `proof.py`
read this declaration from the flip commit and check all pinned journals,
creation records and device history through each last pinned checkpoint.
