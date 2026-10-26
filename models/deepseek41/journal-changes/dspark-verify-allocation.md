# DSpark verify allocation default

The tp4-dspark profile uses the stored floor-one sampled confidence
calibration, with the same five pairs for greedy requests. It preserves
64 speculative rows, widths 6,8,16,32,64, graph rows 64 and depths
`1:5,8:3,16:3,32:0`. Allocation preparation is enabled by default before
removing the model option. The confidence head and shared V7 wire remain.

Each of the six pinned verify fixtures adds ten buffers per rank and loads
the two unchanged locked allocation modules on all four ranks. The exact
creator declarations cover 40 allocations, eight module loads and eight
resolves per fixture. The existing startup check adds 20 uploads, four
allocator launches and four stream synchronizations. It submits an invalid
reservation that writes only status. No retained target or verify operation
changes under the ordered typed handle map.

Creator positions are authored from instrumented archive traces and checked
against sealed predecessor and head captures.
`tools/journal_proof/specs/operation_edits.py` generated the compact 168
operation additions. An independent complete-history and pinned-body audit
matches every generated edit. The proof authenticates the unchanged locks,
exact positions and operands, and reads this declaration at the flip commit.
Regenerate positions and operation edits after a rebase.
