# Wide drafter candidate option

The `wide_draft` model option (`execution_options::WIDE_DRAFT`, default
off, requires `wide_verify`) runs a Flash TP4 drafter step of more than
twelve drafting items in the fewest forwards the widest wide verify bucket
holds, five block rows an item (`Shape::wide_draft_items`), instead of
chunks of at most twelve items. A drafter forward above 64 block rows
folds its dense projections at the wide verify buckets and projects the
vocabulary in the target head's exact F32 row tiles. Without the option
the drafter prepares, allocates and launches what it did before, so every
existing fixture keeps its selections. The allocation test pins two more
fixtures with `wide_verify` and `wide_draft` on: the C16 odd-width fixture
(verify steps of 65 rows) and the C32 maximum fixture (192 rows).

The [change specification](wide-draft.json) is the identity specification
of every fixture, with those two journals declared added. On 2026-10-31
its journal proof was PROVEN: 25 journals identical, 2 declared added, 25
of 27 fixture histories compared (the added fixtures are compared to
nothing). The added journals allocate as many buffers as their
`wide_verify` twins. They differ in the drafter steps of more than twelve
items, which run in fewer forwards with the head's row tiles and the wide
dense folds instead of chunks of at most twelve items with the head's
reuse route and the decode folds.
