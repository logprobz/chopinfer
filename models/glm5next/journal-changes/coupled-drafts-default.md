# Coupled drafts default declaration

`coupled-drafts-default.json` declares the journal change of making
`runtime.mtp_coupled_drafts` the default: a record that omits the option
now couples the MTP drafter's sampled drafts to the target sampler's
noise. The block-FP8 drafted fixture already set the option
([coupled-drafts.md](coupled-drafts.md)) and keeps its bodies. The four
drafting fixtures that did not set it change. Every other journal keeps
its body:

- the context-parallel served drafter journals at sizes 2 and 4
  (`context_parallel_served_tests::served_groups_run_the_drafters_passes`,
  pins (11712, 12000) lines to (11724, 12012));
- the NVFP4 routed and NVFP4 dense-routed drafter fixtures'
  prefill and drafted steps
  (`fake_host_tests::nvfp4::the_nvfp4_routed_drafter_runs_its_mxfp8_experts`
  and `..._dense_routed_drafter_runs_its_bf16_experts`, pins (6192, 4720)
  lines to (6200, 4740)).

Their batches carry no draft sampling records, so their drafter runs
upload greedy records and draft the tokens they drafted before. On each
fixture's four ranks the head adds, at create:

- one load of the unchanged locked `draft_noise_glm5next_sm120` module and
  one resolve of its `chop_row_draft_noise_f32_v1` entry;
- one 1024-byte allocation, the noise records of the 32 draft rows.

Each drafter run uploads its records and launches the draft noise before
its argmax partial (16 uploads and 12 launches per fixture). The drafter's
graphs each capture the added launch, so the eight captured graph creators
of each fixture are declared removed and added, and their replays (and, in
the served fixtures, their releases) move to the added graphs. In the
size 4 served fixture the added launch moves the drafter's extend
program across a multiple of six launches, so its second graph piece ends
one launch later (`bundle_runtime::graphs::pieces`): each rank's
`chop_proj_bf16_dense_n4096_k8192` projection leaves the start of the
third piece (51 nodes before and after) for the end of the second (7 to 8
nodes). Those four launches are declared removed at their base position
and added at their head position, byte-identical under the handle map.

Creator positions were written by an alignment of the base and head
issuance sequences, which refused any addition other than the 12 records
allocations, modules and resolves per fixture and re-captured graphs, any
removal other than a graph, and unequal graph counts.
`tools/journal_proof/specs/operation_edits.py` generated the 248 exact
operation edits: per fixture 16 record uploads, 12 draft noise launches
and 12 replays moved to the added graphs, in the served fixtures 8 graph
releases moved, and in the size 4 served fixture the 4 moved projections.
They were reviewed against the design above. `layout_spec.py` and
`proof.py` read this declaration at the commit that adds it, and check
every pinned journal, creation record and device history through each
fixture's last pinned checkpoint. Positions belong to one capture pair:
regenerate them and the edits after a rebase.
