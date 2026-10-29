# Compact DSpark main-projection adoption

This series makes the TP4 Flash main projection use the existing stage-2
compact template at every retained inner width 129..511. The tuning table
owns that range. It removes the candidate model option, its test overrides
and the duplicate candidate sweep. The ordinary served envelope now checks
the adopted route. The short-row route and generic routes outside the range
remain needed and retain their previous launches and tags.

The paired lab qualified all 385 widths 128..512, including both neighbors,
with 43,744 bitwise output/status comparisons and 782 timings. Served
exactness on 2026-10-29 then passed ten inner widths, 20 greedy/sampled A/B
cases, two A/A controls, 38 retirement-state comparisons and 1,007 complete
logit views, with zero masking. Each case checked its real inner projection
width and route. The frozen GPU install precedes a reporting-only route
correction and an added guarantees check; neither changes the measured
launches or arithmetic. That exactness result therefore applies to this
admission. This route provides no AR speedup and admits no other model.

The served DSpark C1 ABBA passed on 2026-10-29. The 192-row target gained
4.629% in paired inverse step time against 0.596 percentage points of
spread, saving 8.223 ms per step. Controls had no resolved regression and
the provisional energy check had no veto. The
[qualification document](../lab/compact-row-ranges.md) records the visit
medians, support and frozen-install scope. The source remains unlanded
until the journal audit, proof, CPU gates and review finish.

The [change specification](dspark-main-adoption.json) permits exactly four
additional `configure` calls in each of the five allocated-lifetime
fixtures. Each sets 50,944 bytes of dynamic shared memory on the already
resolved compact symbol, once per rank. These fixtures retain every
allocation, handle, module, launch, argument, transfer and lifetime record.
Their pins gain four records each:

| Fixture | Before count and FNV-1a | After count and FNV-1a |
|---|---|---|
| C4 | 69330, `185fae6559e6719a` | 69334, `94b43e6c45e6f09b` |
| C8 | 106922, `322dc6e49e1050f2` | 106926, `8d92d278eff57bef` |
| C4 depth 3 | 69498, `6be615ffa3c974c4` | 69502, `dedf204e99d57b9b` |
| C16 odd | 190430, `bc3be4efd4d537f0` | 190434, `a53d7eeecf401235` |
| C32 maximum | 343168, `a34eacdd446a8626` | 343172, `423f402926f10c51` |

The shared-prefix DSpark fixture adds the same four configurations and
replaces twelve 192-row main projections with the qualified launch pairs.
Its pin changes from 71698 records, `1fd8bd0d42749e1c`, to 71714,
`87f6864c155ff0eb`. The specification uses the prover's existing
`added_configures` policy with exact IDs, counts, positions and byte values,
and `expansions` with exact source positions, grids, argument maps and
output bindings. The compute operands and all unrelated records remain
identical. No verifier rule changes.

The six verify-HC fixtures keep their pinned bodies identical and add the
same four configurations in their create ranges. Those ranges have their
own exact policies. The full-history proof compares all 23 fixtures through
their last pinned checkpoints: six changed bodies, 17 identical bodies,
48 added configurations across 12 create ranges and 12 launch expansions
in the shared-prefix body. It includes 179,854 other history operations.

The long-context template test uses direct digests, outside the sealed
`check_journal` fixture set. Private before/after captures of its unchanged
lifecycle matched the old and new pins in both verification modes. AR
stayed byte-identical. DSpark changed from 193182 records,
`f5e2b3a9adc105b5`, to 193246, `9c0a26cdfbc32100`: the same four
configurations plus 60 main projections at 192 retained rows. Each generic
projection becomes an admission launch and a compact compute launch. The
compute operands remain byte-identical; admission receives exactly the
activation, scale, live-row and status operands and the same row count.
Grids and shared memory match the qualified template. Every other raw
record stays identical, and fresh and reused template histories match.

The failed-template digest has the same four configurations and twelve
projection replacements before the injected failure. Its DSpark pin
changes from 33197 records, `271a60d0c37d72d6`, to 33213,
`9cbd9f933b384cbb`; AR remains byte-identical. A separate raw audit checks
the exact rewrite and preserves the failed operation and terminal poison
behavior. Both direct-digest audits keep their original assertions.

No kernel lock or golden changes. Any unrelated allocation, module,
lifetime or arithmetic difference fails the audit.
