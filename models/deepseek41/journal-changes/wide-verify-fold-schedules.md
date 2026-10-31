# Wide verify fold schedules from the lab sweep

The `dense_fold.wide.verify` rows give 14 of the 16 wide fold keys weight
prefetch at the ticket's column tile; N4096/K1280 keeps the ticket in both
buckets (the [tuning README](../contract/tuning/README.md) records the lab
sweep). Each shape keeps one schedule in both wide buckets, so the wide fold
module `dense_decode_wide_deepseek41_sm120_v1` keeps 16 entries and renames
14 of them, the admission and compute entries of seven shapes, from
`..._r1s3w0f0_v1_{admit,packed}_v1` to `..._r1s3w1f0_v1_{admit,packed}_v1`.
The renamed entries keep their parameters, launch bounds and resource
reports. Only the two allocation fixtures with the `wide_verify` option
(the C16 odd-width fixture at 65 rows and the C32 maximum fixture at 192
rows) change.

The [change specification](wide-verify-fold-schedules.json) is the base
capture's identity specification with one module replacement and one
launch rule per renamed entry in each of those two journals. The
replacement authenticates both revisions' locked payloads and the exact
renames, with the resolves in lock order. The launch rules move every
launch of a renamed entry to its new name: 320 launches per entry, 640 for
N576/K5120, whose entry serves the shared gate and up, so 5120 per journal.
Grids, arguments, allocations, resolve IDs, releases, line counts and
order are unchanged. Every other journal, create range and history is
identical.

On 2026-10-31 the journal proof of the tuning commit against its parent
was PROVEN with full history coverage: 23 journals identical, 2 proven
(5120 launch changes and 56 resolve renames each), 25 of 25 fixture
histories compared.
