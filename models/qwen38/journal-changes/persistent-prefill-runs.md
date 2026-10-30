# Persistent prefill runs declaration

`persistent-prefill-runs.json` declares the one journal change of running
Qwen drafter prefill-phase runs of prefilling rows through the persistent
stages by default: the context-parallel MTP prefill journals
(`fake_host_tests::context_parallel::dcp2_journals_are_pinned_per_package_and_phase`,
`qwen38-dcp2-journal-mtp-bf16-prefill.txt` and
`qwen38-dcp2-journal-mtp-fp8-prefill.txt`). Every other journal keeps its
body: the change touches only drafter runs of prefilling rows that are in
the prefill phase (more than 32 rows), and the launch journals serve no
drafter. The decode and verify segments hold no prefilling item, and runs
of at most 32 rows keep their graph buckets.

The prefill step prefills cold prompts of 5 and 40 tokens; its
post-target drafter run of 43 rows ran in the prefill phase already. On
each of the fixture's four ranks the head:
- drops that run's 35 launches of the 18 stages after the QSA key write
  (the index query, selection, sparse attention, gate and output, the
  attention `Reduce`, the MLP sublayer's hyper-connection stages, the MoE
  and shared experts and the MLP `Reduce`) and of the inject projection of
  the attention sublayer's `HcIn`, whose only reader is the dropped
  `Reduce`;
- keeps the launches of the 13 persistent stages with their plans and
  arguments, issued rank by rank (the whole program on one rank, then the
  next) instead of stage by stage across the ranks.

No allocation, module, resolve or graph changes. `tools/journal_proof/
specs/operation_edits.py` generated the exact operation edits from the
captures of the candidate's commit (off by default) and of the flip: per
fixture 140 removals and 42 additions. They were reviewed against the
list above: every removal is a launch of a dropped stage or an inject (35
per rank), or one of the 42 persistent-stage launches the reorder moves;
every addition is one of those 42 at its new position, byte for byte the
removed text. The two prefill pins moved from (7234, 0x19D66D0D930AF5BC)
and (7234, 0x44F77C4B70AE5F46) to (7094, 0xCE90782AF36DEA78) and (7094,
0xF3E2F6B39C15CA16). Positions belong to one capture pair: regenerate them
and the edits after a rebase.
