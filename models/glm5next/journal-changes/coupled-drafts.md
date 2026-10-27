# Coupled drafts declaration

`coupled-drafts.json` declares the one journal change of
`runtime.mtp_coupled_drafts`: the block-FP8 drafted fixture
(`fake_host_tests::nvfp4::the_block_fp8_and_nvfp4_drafter_runs_its_fp8_experts`)
now sets the option and drafts a sampled item (seed, generated index 1,
temperature 1.0). Every other journal keeps its body: the option is off by
default, and the commits before this one were proven identical to their
base, all 67 journals byte for byte.

On each of the fixture's four ranks the head adds, at create:

- one load of the unchanged locked `draft_noise_glm5next_sm120` module (its
  payload is the lock's, byte-identical to DeepSeek's `dspark_noise` cubin)
  and one resolve of its `chop_row_draft_noise_f32_v1` entry;
- one 1024-byte allocation, the noise records of the 32 draft rows.

The prompt chunk's drafter run uploads greedy records (its batch carries
none) and launches the draft noise before its argmax partial. The drafted
step's three drafter runs upload the item's record at generated indices 1,
2 and 3, one payload per run and rank. The extend graph (42 nodes) and the
draft step graph (36 nodes) each capture one more node, the draft noise
launch before the argmax partial, so their creators are declared removed
and added and their replays move to the added graphs. The draft step graph
is captured once and replayed for both of its positions. The fixture's
prefill and drafted pins moved from (6192, 4720) to (6200, 4740) lines.

Creator positions were written by an alignment of the base and head
issuance sequences, which refused any addition beyond the 12 expected
issuances and any graph that changed other than by one node.
`tools/journal_proof/specs/operation_edits.py` generated the 52 exact
operation edits (28 additions of uploads and launches, 12 replay removals
and 12 replay additions), which were reviewed line by line against the
design above. `layout_spec.py` and `proof.py` read this declaration at the
commit that adds it, and check every pinned journal, creation record and
device history through each fixture's last pinned checkpoint. Positions
belong to one capture pair: regenerate them and the edits after a rebase.
