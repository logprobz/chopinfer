# Coupled drafter fixtures declaration

`coupled-drafter-fixtures.json` declares the journal change of pinning the
adopted coupled drafter route ([coupled-drafts-default.md](coupled-drafts-default.md))
in the lowering journals. The shared test helpers (`Ops::for_tests`,
`journal_tests::rank_buffers_within`, `journal_tests::draft_regions` and
the own ops tests' rank buffers) prepared the greedy draft argmax and
allocated the scratch without the noise records, so the drafter's pinned
pass journals kept the route the adoption replaced. They now prepare and
allocate as a drafting instance does. The invariance plans couple too;
their asks journal keeps its body, since the noise's guarantees keep the
argmax's granted claims.

Two journals change, the drafter's Extend pass at a verify bucket of 32
rows over 8 items and its Step at 8 rows
(`journal_tests::drafter_journals_are_pinned_per_pass`, pins (576, 552)
lines to (584, 560)). On each of the four ranks:

- one 1024-byte allocation on the rank's stream, the noise records of the
  32 draft rows, last in its scratch, so every handle allocated after it
  renumbers;
- one `chop_row_draft_noise_f32_v1` launch before the rank's argmax
  partial, binding the logits shard, the noise records and the live width
  and draft counts.

Every other journal keeps its body. Creator positions were written by an
alignment of the base and head issuance sequences, which refused any
addition other than noise records allocations, the draft noise module and
its resolve, and re-captured graphs; it found only the four 1024-byte
records allocations per fixture.
`tools/journal_proof/specs/operation_edits.py` generated the 8 exact
operation edits (the noise launches), which were reviewed against the
design above. `layout_spec.py` and `proof.py` read this declaration at the
commit that adds it, and check every pinned journal, creation record and
device history through each fixture's last pinned checkpoint. Positions
belong to one capture pair: regenerate them and the edits after a rebase.
