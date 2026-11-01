# Verify row fusions adoption journals

The series adopts the `verify_row_fusions` candidate
([verify row fusions journals](verify-row-fusions.md)) as the Flash TP4
default at six rows or fewer, through the latent tuning row
`latent.row_fusions.verify` (rows 1 to 6), and then retires the option.
The served A/B resolved a gain only at C1's six-row verify step (1.85%,
paired deltas 1.74% and 1.95%); C8's 32-row step was neutral (1.45%,
paired deltas 0.54% and 2.35%, spread 1.81 points). So the target fusions
are admitted at verify steps of at most six rows, and verify steps of 7 to
32 rows keep the chains. The drafter is admitted by the same row range on
its own operations: a drafter forward of at most six rows (one drafting
item, five rows) fuses its queries, and a drafter window write of at most
six rows quantizes with the parallel ceil32 entry, wherever they occur.
Both are bitwise in the lab pairs and fixed-batch replays, and the C8 cell
of the A/B already ran a superset (every drafter forward of up to 32 rows
fused, every write of up to 60 rows parallel) without a regression in any
cell.

## The flip

The flip commit ("Serve the decode row fusions at verify steps of at most
six rows") adds the row and turns the paired option tests into checks of
the default. Its layout declaration,
[verify-row-fusions-flip.json](verify-row-fusions-flip.json), was
generated from sealed captures of the commit before it (the default
prefix-capture pin) and of the flip, and committed after the flip. The
proof reads it there (`--declaration-revision`). It lists every changed
launch, 3,636 operation edits in eight device fixtures. Every changed
launch is at a verify step of at most six rows, in a drafter forward of
five rows, or a drafter window write of at most six rows:

| Fixture | Before count and FNV-1a | After count and FNV-1a | Changed launches |
|---|---|---|---|
| C4 | 69334, `94b43e6c45e6f09b` | 69334, `97009e6ff4b56ef3` | 12 four-row drafter window writes, serial to parallel ceil32 |
| C4 depth 3 | 69502, `dedf204e99d57b9b` | 69502, `f3563cf8a337e753` | the same 12 writes |
| C4 phased | 69358, `fdbd662fbe69bdf6` | 69358, `227f917f14837e3e` | the same 12 writes |
| C4 HC draft rows | 69262, `4db4a4e157875315` | 69262, `1485da7370037915` | the same 12 writes |
| C4 drafter options | 69206, `f7b1328155f83920` | 69206, `9649bad3b349d138` | the same 12 writes |
| verify-hc-4 | 7880, `790492fb319c0e52` | 7400, `3be5137a5a40b0c0` | 160 query pairs to the fused query (general entry, 4 rows), 160 window chains to the fused window; 12 graphs |
| verify-hc-6 | 8040, `3185f0f2834b3de2` | 7560, `8a0263a1930b0bf8` | 160 query pairs to the fused query (cached16, 6 rows), 160 window chains to the fused window; 12 graphs |
| DSpark prefix capture | 40596, `207ddbc5c00ec41b` | 40104, `00ca4ee0728ce1b9` | see below |

The C4 lifetimes' 16- and 12-row verify steps keep their chains; only the
drafter's four-row window writes change, each with the same arguments and
grid. In the verify-hc fixtures each target layer's query norm and ceil
pair becomes one fused query launch and its window RoPE, ceil32 and store
become one fused window write on every rank, and each of the 12 captured
graphs that holds them is declared removed and added at its positions,
with its node count lower by one per query pair and two per window chain.

The prefix-capture device runs a three-token DSpark prefill, a prefix
capture whose catch-up writes the three prompt rows, a restore into
another slot and a drafted six-row verify step. Its 36 drafter window
writes of three and five rows (24 and 12; 12 of the three-row writes are
the capture's catch-up, which the test checks) take the parallel ceil32;
the verify step's 160 query pairs and 160 window chains fuse (cached16 and
the fused window at six rows), and the drafter's five-row forward fuses
its 12 query pairs on the general entry. The default lifetime now equals
the option's, which the flip commit's test asserts.

Unchanged: the C8, C16 odd and C32 maximum lifetimes and the other
candidate lifetimes (none of them runs a verify step, drafter forward or
drafter window write of at most six rows), the DSpark and AR shared-anchor
lifetimes (the DSpark one's 52- and 12-row catch-up writes stay serial), verify-hc-8,
-12, -16 and -32, every AR and TP1/TP2 journal, and the four `-rowfuse`
lifetimes, whose option already ran the fused route.

The long-context and failed-template digests in
`prefill_plan_templates.rs` are direct digests outside the sealed journal
set, so a temporary, uncommitted patch dumped their journals at the
adopted code and with the row removed. The row-removed dumps reproduce the
previous pins exactly (DSpark long context 193246, `9c0a26cdfbc32100`, in
both verification modes; DSpark failed template 33213,
`9cbd9f933b384cbb`) and the adopted dumps the new ones (192754,
`95c7cbfd89de3432`; 33213, `565c2a3e287f283f`). AR stays byte-identical.
Every hunk of the DSpark long-context journal is a fused query (172: 160
target pairs of the mixed step's six-row verify item and 12 of the
drafter's five-row forward), a fused window (160), a serial to parallel
ceil32 write of 2, 4 or 5 rows (48) or a captured graph whose node count
falls (16); the failed-template journal changes only in 12 two-row
drafter writes, serial to parallel. Both modes of the long-context
lifecycle give identical journals.

## The retirement

The option commit ("Retire the verify_row_fusions option, its overrides
and profiles") removes the option and its four lifetimes. Its spec,
[verify-row-fusions-retire.json](verify-row-fusions-retire.json), is the
identity of the flipped package with those journals declared removed.

## Proofs and evidence

On 2026-11-01 the flip's proof (base the commit before it, head the flip,
the declaration read at the commit that adds it) was PROVEN: 34 journals
identical and the 8 above proven, 2,544 removed and 1,092 added operations
and 24 graph handle records each way, with all 42 fixture histories
compared (179,854 history operations; a full history proof). The removal
proof (base the flip, head the option commit) was PROVEN: 38 journals
identical and the 4 declared removals, with 38 of 42 fixture histories
compared. The removed lifetimes differ from the defaults but for the
prefix capture, whose default now equals it: the option fused the C4 and
C8 lifetimes' 16- and 32-row verify steps and the shared anchor's 52- and
12-row catch-up writes, which the adopted route keeps unfused.

The data volume had about 100 to 140 GiB free, short of the capture
helper's 180 GiB start threshold, so every capture ran with
`--start-free-gib 80 --minimum-free-gib 60`. Two earlier capture attempts,
started with thresholds of 70 and 40 GiB, were stopped and rerun at 80
and 60.
Captures of earlier versions of the flip, which failed on the quick route
sweeps before the fused query allowlist entry covered the drafter's
five-row forwards, or were stopped, were deleted.

Full route sweeps of `tp4-dspark` and `tp4-dspark-rows192-wide` at the
option commit passed (32 pieces each). Their symbol sets equal the
pre-flip integration sweeps' (755 and 817 symbols): the serial ceil32
still serves prefill and wider writes, the unfused query norm, ceil and
window chain serve wider verify steps and drafter forwards, and the fused
entries already served decode. No kernel entry stops serving, so nothing
is pruned and no symbol inventory, ELF audit or relock applies. The
general fused query entry serves verify steps of 2 to 5 rows and the
drafter's five-row forward at every verify step with one drafting item,
which the sweep records at that step's rows; the allowlist entry covers
verify rows 2 to 37 (one drafting item beside up to 31 decode items).

No kernel lock or golden file changes.
