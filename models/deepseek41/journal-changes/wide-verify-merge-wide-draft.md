# Wide drafter forwards over the merged wide verify branch

The merge of the [wide drafter forwards](wide-draft.md) into the wide
verify branch adds the branch's two journals with `wide_verify` and
`wide_draft` on, the C16 odd-width and C32 maximum allocation lifetimes, at
the branch's pins (191702 and 343532 lines). In the merged allocation test
they are its tenth and eleventh instances, after the phased fixtures.
Where the branch's drafter head prepared its own verify row tiles, the
merge keeps the exact F32 side's `prepare_drafter` and adds the verify
site's tiles from the 65th row to the widest wide verify bucket when
`wide_draft` is on; the option is off by default, so every other fixture
keeps its selections.

On 2026-10-31 the journal proof of the merge commit against the first
parent was PROVEN: 29 journals identical and the 2 declared additions,
with 29 of 31 fixture histories compared (the added fixtures have no base
history). Its specification was the identity specification of the first
parent's capture with the two allocation fixtures and their journals
declared added, as in [wide-draft.json](wide-draft.json).
