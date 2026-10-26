# Prefill BF16 partials journal declaration

The test default now uses the served BF16-partial contract for native
Flash TP4 prefill. The declaration replaces each rank's 184549376-byte
inbox with 92274688 bytes and its 33800-byte flags with 33864 bytes. The
four-byte epochs retain their identity. Positional replacement handles
scope these changes to the native fixtures. The rounded module is
inactive there and the two unchanged locked BF16 modules become active.

Exact operation ordinals cover initialization and prefill reduction
substitutions. The source audit derives replacement operands and launch
configurations from the previous F32 reductions. Decode, verification,
BF16-KV routes and unrelated operations retain their existing behavior.

Creator declarations were derived from sealed adjacent-commit captures,
constrained to the resources above. `tools/journal_proof/specs/operation_edits.py`
generated the compact exact operation edits. The edits were reviewed
against source-derived history audits. `layout_spec.py` and `proof.py`
read this declaration from the flip commit and check all pinned journals,
creation records and device history through each last pinned checkpoint.
