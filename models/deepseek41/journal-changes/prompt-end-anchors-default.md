# Prompt-end anchor default

The `prefill_anchor_items` contract default becomes 8. Both TP4 profiles
already select upper-skip and guaranteed cuts, so they inherit this value
without another profile override. The allowance remains restricted to
Flash TP4 upper-skip with cuts and capped at served sequences. Other
topologies and configurations still declare zero. Using the contract default
also makes default fake-host instances exercise the adopted allocation.

The [layout declaration](prompt-end-anchors-default.json) permits separate
anchor buffers in eight fixtures: the AR and DSpark shared-prefix fixtures,
both long-context template verification modes, and both failed-template
fixtures. Each rank adds a window-ring spare, compressor-tail spare and
row-sequence zero buffer; DSpark also adds raw feature history. Each buffer
is zeroed once. Successful teardown releases it, while poisoned instances
keep their existing refusal behavior. Existing allocations, launches,
operands, transfers and snapshots retain their contents under the ordered
handle map. No kernel lock changes.

The template tests previously checked four digest pins directly. They now
use `check_journal`, with six named bodies because the two long-context
verification modes share their pins. The comparison base changes only those
checks, keeping every original expected digest. Its
[fixture specification](prompt-end-anchors-fixtures.json) proves 38 original
journals identical and six declared additions. It does not claim a history
comparison for the added fixtures. The default-change proof then compares
all 44 fixture histories against that expanded base.

The CPU proof is PROVEN: 36 journals identical, eight changed only as
declared, and all 44 histories compared. It authorizes 112 added allocation
records, 112 zeros and 84 releases. The failed-template fixtures account
for the 28 buffers without teardown releases. The six distinct pin changes
are below; each long-context pin covers both verification modes.

| Fixture | Before count and FNV-1a | After count and FNV-1a |
|---|---|---|
| AR shared prefix | 68956, `90a6e235cbb80c43` | 68992, `479e3f096080caae` |
| DSpark shared prefix | 71878, `c987852dd2e788cc` | 71926, `a1ebf5e3963e9586` |
| AR long-context templates | 164705, `fb12b2a6af5ce6e0` | 164741, `27b8e2307434c1fd` |
| DSpark long-context templates | 192910, `9d8aa84421c28b98` | 192958, `79d77d6ee383ebab` |
| AR failed template | 31864, `e005e62b7bf8f901` | 31888, `332486104ca433a5` |
| DSpark failed template | 33293, `66f0da55bc326a9f` | 33325, `9ad7a4f758f942e6` |

Captures use `tools/journal_proof/capture.py` with the per-test nextest
runner and four workers. The four existing export-only ignored tests are
listed explicitly through `--allow-ignored`. The declaration and generated
witness are checked by `tools/journal_proof/proof.py`; the landing proof
reads the declaration at the compared head. Captures and reports remain
outside the repository. These are structural CPU checks, not numerical or
performance evidence.
