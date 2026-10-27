# Dense decode template

[The declaration](dense-decode-template.json) replaces four decode modules
with one template module and renames 88 entry symbols one to one. The table
knobs preserve the ticket, prefetch, reuse and split schedules, including the
runtime reuse partition. All captured fixtures, including the AR and DSpark
shared-prefix anchors, retain the same launches, arguments and resources.

Each replacement has identical instruction text, constant data, shared-memory
extent, register count and parameter ABI. CPU checks cover every selected
launch configuration. Poisoned GPU replay covers 1,514 served row cases and
897 synthetic cases, including paired reuse, half folds, padded group strides
and partial live rows. Both missing-write controls are rejected.

The symbol mapping comes from the served twin catalog. The frozen journal
proof tools compare the authenticated base and head captures, generate any
exact operation edits for review, and read this declaration at the compared
head. Pin changes are limited to the resulting journal fingerprints and
module/symbol bookkeeping counts.

The reviewed operation edits remove exactly 3,176 teardown `release` lines
across twelve fixtures. They account for the function and module handles
removed by consolidation: two fixtures remove 75 releases each, three remove
150 each, and seven remove 368 each. Every edit names an exact line position;
no launch, synchronization, allocation or peer-result operation is edited.
The two shared-prefix anchors participate in the same full-history proof.
