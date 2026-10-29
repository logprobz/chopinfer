# Compact DSpark main-projection candidate

The [declaration](dspark-main-compact.json) permits no change to the
existing pinned journals or complete device histories. The declared model
option `dspark_main_compact` defaults to false. The candidate selects one
contiguous tuning-table range, TP4 Flash N5120/K15360/G1 at 129..511 rows,
using stage 2 of the existing shared compact template. It changes no CUDA
source, packed weight layout, allocation, workspace, ABI or kernel lock.
The generic projection remains outside that range; the current small-row
projection keeps precedence through 64 rows.

Default journal identity does not prove candidate-on execution. A served
fake-host regression executes cold single and packed prompts with the option
off and on. It asserts three projection passes and their actual retained
row totals at the admission boundaries and template boundaries. A 256-row
prompt retains only 128 main rows; a packed cohort is needed for 511 rows.
The candidate replaces each generic launch with the template's admission
and compute launches. A separate CPU sweep checks every count through 8192
at TP1, TP2 and TP4, requiring unchanged launches and route tags outside
the one admitted range and unchanged operand layouts and guarantees.
Disabled calls return the original prepared operator and its Fast generic
route tags. The candidate adds only Fast compact tags and needs no new
fallback allowance. Its served route sweep uses the same allowlist and
checks the quick tier while the option remains default-off.

Generate a layout spec from sealed integration-base and candidate-head
captures using this declaration. Also generate an empty identity spec and
run `proof.py --require identity`. Every existing pin and device history
must match. New unit and route tests add no journal pins. Do not authorize
an operation edit to conceal a failure of this identity requirement.

On 2026-10-29, both devices passed all 385 widths from 128 through 512,
including the candidate band's neighbors. Four bounded exclusive labs
recorded 43,744 output/status comparisons and 782 per-width timings
including duplicate shard neighbors. Every timing gain exceeded its
bracketing A/A spread. These establish pair-lab evidence only. The option
remains default-off until full served logits/state exactness and DSpark
step-time acceptance pass.

Adoption is a separate commit that flips only this route, audits its actual
journal edits against a new source-bound declaration, and removes the
candidate option and its test overrides. Keep
other projection admissions separate. No AR benefit is claimed: AR does
not execute this DSpark projection.
