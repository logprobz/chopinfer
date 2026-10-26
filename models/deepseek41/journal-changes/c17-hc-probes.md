# HC probe catalog teardown

`c17-hc-probes.json` declares the teardown of five unlaunched cluster
controls removed from the mixed HC mixer module. The module replacement
spec authenticates the old and new payloads against their bundle locks and
removes exactly those five resolves per rank. Every retained symbol, launch,
grid, operand and allocation stays unchanged under the typed handle map.

The fake host records resource releases as bare `release` lines. Removing
five symbol handles also removes five releases per rank. Ten lifecycle
fixtures destroy their instances before their pinned checkpoints and contain
140 such releases in total. The six verify-only fixtures end their compared
histories before teardown and need no release edits. The declaration lists
every removal by its exact one-based note ordinal in the complete fixture
history; its compact lists contain explicit positions, not ranges.

The positions were regenerated after the integration rebase from a fresh
sealed parent capture and an external typed-release probe. The probe observed
the typed Symbol passed to the fake host's release callback and the next
note ordinal, then emitted the original note unchanged. All three selected
lifecycle test functions passed their parent pins. Each of the 140
observations uniquely matched its symbol ID and release position in the
authenticated parent trace. The operation_edits.compact function grouped
these exact observations into the declaration's explicit position lists.
Module loading holds symbols in catalog order; instance teardown releases
them newest first. For each rank the removed cluster controls release in
reverse, ordered, broadcast, gather, barrier order. This supplies the symbol
attribution that a bare journal line cannot provide on its own.

The combined proof compares all 16 pinned journals and complete fixture
histories. It allows only the locked module replacement, its removed
resolves, the ordered identity mapping and these exact teardown edits.
A missing or extra release, a changed retained resolve, or a changed launch
remains an unexplained difference. The declaration's positions must be
revalidated against fresh captures when another change alters a fixture.

The six retained entries have identical text, constant and shared sections
under CUDA 13.2.86. Rounding the finalizer arena extent to 16 bytes preserves
the original 16704-byte shared allocation after pruning. Every retained
resource field is unchanged. The info sections differ only in the
EIATTR_PARAM_CBANK symbol-table index, which resolves to the same constant
section; after that index mapping, their bytes match. On
2026-10-26, the served warp finalizer passed 1120 bitwise twin cases on an
exclusive SM120 device: rows 1 through 32, every live count from zero to
capacity, two input patterns, and poisoned eager, graph and PDL replays.
This is kernel evidence. Served bitwise comparison and the contract A
simplification timing verdict remain separate gates.
