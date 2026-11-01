# Direct MoE knob set as the served route

The MoE tuning table's rows for the direct MoE's absorbing path at its 32-
to 64-row buckets on Flash TP4 (`moe.absorb.decode`, `moe.absorb.verify`
and `moe.wide.verify`) set the four knobs `stream`, `keep`, `retire` and
`loads`, so the test default now serves the knob set that the
`moe_direct_*` candidate options selected there. This one commit flips all
four, as they were accepted together. The wide verify row
(`moe.direct_wide.verify`, 65 to 192 rows) sets none: those buckets keep
the plain route because they have no served step-time measurement. The
knob entries live in the unchanged locked module
`bandwidth_moe_deepseek41_knobs_sm120` (family `routed_moe_direct_knobs`);
no lock, golden or kernel changes.

Every pinned journal change is the knob set and nothing else:

- **Module activation.** Each fixture that prepares the backbone bank's
  direct MoE at TP4 loads the knob module once per rank and resolves its
  15 entries. A lifetime that completes its destroy releases the added
  symbols and module: 16 bare releases per rank. The added resolves shift
  later issued IDs, which the typed handle mapping absorbs. That is all
  that changes in the 26 journals whose steps never reach a 32-row bucket,
  among them the four wide verify lifetimes and five of the six verify HC
  journals, which keep their line counts.
- **Launch substitution.** Where a step reaches a 32-row bucket, the
  absorbing gate queue becomes its `_policy_sk` entry (stream and keep),
  the two-strip down queue its `_policy_sk` entry, the gate finisher the
  all-loads-first finisher's `_policy_r` entry (loads and retire) and the
  down finisher its `_policy_r` entry (retire). Grids, blocks, shared
  memory, flags and every argument stay as they were: 960 substitutions of
  each entry, in the five C8 allocated lifetimes and `verify-hc-32`. The
  gate finisher at other rows, the two-strip down queue at 16 rows, the
  down finisher at other rows and every launch of the wide verify buckets
  keep their entries.

TP1 and TP2 journals are identical. The declaration
[moe-direct-knobs.json](moe-direct-knobs.json) lists the module
activations with their positional handles, the four launch rules
(selected by the base launch's grid, or for the 32-row down queue by the
32-row gate finisher before it on its stream), each rewriting only the
symbol, and the 1,664 added releases.
`tools/journal_proof/specs/operation_edits.py` generated the releases from
sealed captures of the parent commit and this change. Positions belong to
that capture pair; regenerate them after a rebase.

| Journal | Before (lines, FNV-1a) | After (lines, FNV-1a) |
|---|---|---|
| `deepseek41-allocation-c4.journal` | 69334, `94b43e6c45e6f09b` | 69458, `31e8feae163fcdae` |
| `deepseek41-allocation-c4-d3.journal` | 69502, `dedf204e99d57b9b` | 69626, `18ef6fee16402a52` |
| `deepseek41-allocation-c4-drafter.journal` | 69206, `f7b1328155f83920` | 69330, `d5210fed492f9516` |
| `deepseek41-allocation-c4-hc-draft-rows.journal` | 69262, `4db4a4e157875315` | 69386, `ef446c1cc9c37708` |
| `deepseek41-allocation-c4-phased.journal` | 69358, `fdbd662fbe69bdf6` | 69482, `c36f79e0d5d3d8ae` |
| `deepseek41-allocation-c8.journal` | 106926, `8d92d278eff57bef` | 107050, `2c1031ee4933ca42` |
| `deepseek41-allocation-c8-confidence-staged.journal` | 106938, `0156530484d50f83` | 107062, `e888efa9eb5e76e7` |
| `deepseek41-allocation-c8-hc-fusion.journal` | 106854, `85c7e7bd95972561` | 106978, `1ce1fc7778fc43a4` |
| `deepseek41-allocation-c8-markov-staged.journal` | 106918, `ec2cda8f7c64a88f` | 107042, `e90fed7432635aca` |
| `deepseek41-allocation-c8-moe-direct.journal` | 106866, `39dbc028f5685113` | 106990, `63361922e9fdd91a` |
| `deepseek41-allocation-c16-odd.journal` | 190434, `a53d7eeecf401235` | 190558, `759dd31d2c863d27` |
| `deepseek41-allocation-c16-odd-phased.journal` | 189194, `b6fd29e113df566c` | 189318, `2162072561a7ef58` |
| `deepseek41-allocation-c16-odd-wide.journal` | 191802, `64bd42c3caef55c0` | 191926, `49511b69ebd37509` |
| `deepseek41-allocation-c16-odd-wide-draft.journal` | 191702, `7ec0e494350dc4a8` | 191826, `b23641665a5e27f7` |
| `deepseek41-allocation-c32-maximum.journal` | 343172, `423f402926f10c51` | 343296, `1a96fa62a770e259` |
| `deepseek41-allocation-c32-maximum-wide.journal` | 344540, `07785bc19d29a6ff` | 344664, `8df4cd20aa5fc564` |
| `deepseek41-allocation-c32-maximum-wide-draft.journal` | 343532, `f9ee11d6af367dda` | 343656, `9f646bd1031a80ff` |
| `deepseek41-flash-tp4.native.journal` | 72873, `b81a960e61384165` | 72997, `e6423caa989339f2` |
| `deepseek41-flash-tp4.bf16.journal` | 74481, `45a93dba79cec4f5` | 74605, `c95ae28e64906dd7` |
| `deepseek41-flash-tp4.native-phased.journal` | 72897, `d48990d5176a5efa` | 73021, `85eefb57719d3ced` |
| `deepseek41-flash-tp4.bf16-phased.journal` | 74485, `102e1b6ae4b16d92` | 74609, `530585999f6a0603` |
| `deepseek41-flash-tp4.media.journal` | 47281, `9002432756ab2a61` | 47405, `083c4e787f317981` |
| `flash-tp4-ar-shared-anchor.journal` | 68792, `982e03f4c49988d2` | 68916, `eccfc0aa64869145` |
| `flash-tp4-dspark-shared-anchor.journal` | 71714, `87f6864c155ff0eb` | 71838, `9e437b99380b577e` |
| `verify-hc-4.journal` | 7880, `790492fb319c0e52` | 7880, `9075f6d78174632c` |
| `verify-hc-6.journal` | 8040, `3185f0f2834b3de2` | 8040, `fb28f6f67af9e99c` |
| `verify-hc-8.journal` | 7720, `19bda5468a9871c1` | 7720, `4d16bed4d20d88d1` |
| `verify-hc-12.journal` | 8216, `f938777c820e1a25` | 8216, `6890db76fe0795ef` |
| `verify-hc-16.journal` | 8052, `82a85847466cc04a` | 8052, `6d02b84bd45e8e5e` |
| `verify-hc-32.journal` | 7744, `abf9ab2b237f17fc` | 7744, `49a90fc9570d6dee` |

The two native-MoE aliases of the TP4 journals move with them. Two
template tests pin their journals by digest outside `check_journal`, so
the proof does not read them; they move by the same activation: the
long-context template lifetimes by 124 lines like the other complete
lifetimes (15 resolves and 16 releases per rank), and the failed-template
journals by 60 (15 resolves per rank; their destroy fails before any
release).
