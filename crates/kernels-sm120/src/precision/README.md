# Generated projection offers

`generated::offer(Request, adoptions)` takes a kernel-core `OperationSpec`,
format plug-in ID, use and explicit implementation schedule. The
caller supplies already sliced, padded rank geometry. Dense, grouped and lookup
construction still goes through the typed builders. Generation checks the
plug-in recipe and binds the complete resulting portable capability before
returning a retained `KernelOffer`. `generated::layout_copy` covers exact copies.

Symbols have the form
`chop_proj_<format>_<kind>_n<rows>_k<k>_<digest>_v1`. The digest includes
canonical JSON of neutral-symbol code, ABI, headers, defines, arithmetic and
launch declarations. Typed and retained entries use the same identity rule. Runtime
phase labels and source tensor names do not create another symbol when the
executable use is identical. Reduction schedules remain part of the identity.

Row stages (activation quantizers, gathers, layout copies and lookups) never
hash their capacity, which only sizes the grid: one entry serves every
capacity of its body. Each builder also takes a `row::precision::ShapeBinding`.
`Compiled` keeps widths, heads and vocabulary in the code. `Launch` appends
them (and copy strides and offsets) as `u32`/`u64` arguments. The identity
then holds the body, ABI, grid rule, threads, PDL and numerical contract but
no geometry, so the symbol is shape-free
(`chop_proj_<format>_<kind>_<digest>_v1`) and one entry serves every layout
and rank geometry. `precision::selected_cubins_with` with `LaunchRows::Shared`
leaves launch-shape stages out of an owner's `{owner}_rows_v1`, and
`precision::launch_row_cubin` places them in one cubin that every geometry of
a bundle shares (membership `*`). Layout copies use `Launch`. Lookups,
quantizers and gathers default to `Compiled`: their launch-shape forms
measured 1-15% slower in kernel-lab (`launch_shape_row_stage_timing`), while
copies measured equal or faster.

`generated::adopt` compares an independently constructed retained offer's
capability, entries and launch declarations before proposing aliases. An
adoption records `symbol`, `generated`, `entry_sha256` and cubin placement
(name and entry position). A caller selects the
aliases for one retained implementation before generating its adopted offer.
One generated identity can have several historical aliases; the full bundle
ledger does not choose an implementation. Ambiguous aliases are refused.

Bundle `projection-adoptions.json` files are canonical JSON arrays and append
only. `tests/projection-adoptions.py` checks their approved-base prefix and
refuses deleted ledgers, rewrites, unchanged duplicate records and symlinks.
A later record supersedes an earlier record for the same symbol. A record with
`replaced_by` retires its symbol: it has no placement and names the entry that
now serves the symbol's uses, and `generated::active` drops the symbol from
selection and alias lookup. Retiring appends; no record is deleted. Exporters
keep the existing prefix and append discoveries. Selection fills recorded cubin
positions with generated entries and preserves the remaining catalog entries
in their original order.
Model comparisons check the current records against the ledger. Existing cubin
locks remain authoritative for complete source closures and compiled bytes.

`generated::comparison` is a transitional comparison adapter for existing
finite inventories. Its synthetic spans describe a supplied capability's
rank-local domain. They are not checkpoint import records. Declaration-derived
operations can replace these inputs through the same regeneration API. Explicit
catalog adapters retain legacy implementations that ordinary builders cannot
represent; their stage definitions and operation domains are checked.

The lab enumerates cases from generated model inventories and creates a
separate audited selection with an external lock. Oracle cases, zero-live
refusals, replay checks, declared chunk properties and exact copies retain the
same supplied handles. GPU execution uses prepared binaries and cubins. CPU
preparation and passing CPU tests are not numerical device evidence.

`generated::schedule` supplies shared shape/use defaults and append-only tuning
records. An optional `projection` key is the declaration's projection ID and
only disambiguates colliding schedules. The selector is checked against the
operation ID and does not enter the executable identity. Model exporters capture
full schedule data from their independent references; consumers resolve the
policy plus the committed overrides.
