# Shared-prefix anchor journals

The change adds two complete fake-device lifetime journals, one for AR and
one for DSpark. Every existing journal, create sequence and device history
must remain identical. No kernel, locked module, operation substitution,
resource rewrite or journal removal is declared.

Both new fixtures prefill to a scheduler-selected endpoint at token 4032
inside an 8192-token grid, capture complete state, retire the first request,
restore its prefix into a new request, continue to the grid at 8192 and then
to the prompt end at 8257, and release the instance. The fixture also checks
that the original prefill entry refuses the extra endpoint. Snapshot page
counts remain 53 for AR and 125 for DSpark.

The declarations in [shared-prefix-anchors.json](shared-prefix-anchors.json)
name only these added journals. The proof reads an exact copy extracted from
the candidate commit. Its reported spec SHA256 must equal that committed
file, every existing journal must be IDENTICAL, and exactly these two
journals must be DECLARED_ADD. Their fixture bindings are checked against
sealed captures by the journal proof tool. Existing fixtures have empty
rewrite policies, so adding endpoint coverage cannot excuse any change to
their launches, copies, resources or ordering.

Journal identity establishes structural preservation only. CPU cone checks,
complete-state capture and restore tests, and the required GPU comparisons
supply separate evidence for the added endpoints. The pin values follow the
base's test transport defaults; a dependency rebase must audit any resulting
pin change before the final proof.

The layout-declaration schema requires unchanged journal membership. These
additions therefore use the supported added-journal spec, bound to the
candidate commit as described above. A declared addition authenticates a new
fixture against its source pin; it does not claim equivalence to a fixture
that did not exist at the base.

The DSpark fixture includes the adopted verify-allocation startup check,
indexed-retention storage and early-Engram resources. These add 168 lifecycle
operations to this new fixture: 49 allocations, eight symbol resolves, four
startup launches, 20 uploads, six events, one stream, four synchronizations,
four mapped-gather releases and 72 resource releases. Existing launch order,
grids, scalar arguments, offsets and lengths are unchanged after removing
opaque handle labels for the diagnostic comparison. The AR fixture is
unchanged. This pin audit does not replace the sealed journal proof.
