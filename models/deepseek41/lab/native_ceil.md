# Served native MoE ceil fusion

Native Flash TP4 prefill fuses the intermediate E4M3/32 ceil-UE8M0
quantizer into gate/up tiles at at most 2048 rows. The tile writes group
codes/scales and ORs row flags; a following fixup zeros every flagged
live token's codes and writes scale 127. Inactive rows remain untouched.
The result is bitwise equal to native tiles followed by the separate
quantizer. Larger rows retain that separate pass.

This is a bitwise optimization of the adopted
[native MoE arithmetic](native_moe_prefill.md), whose difference from the
exact BF16 engine has its own within-bound qualification. Tests retain
poisoned repeats, graph replays, prior status, captured-bank authentication
and a missing-fixup negative control.

## Result (2026-10-21, Flash TP4, two RTX PRO 6000 at 250 W)

Criteria 1 to 6 passed on both workers: 56 bitwise records (8 synthetic,
48 captured), memcheck clean, captured cases authenticated. The clear
pattern flagged no token; the flagged pattern flagged 85, 22, 3 and 1
tokens before the fixup, and every unfixed control differed.

Timing (fused / separate quantizer, median of paired rounds): captured
8192-row layers 2.953 / 2.908 ms (1.015); synthetic 8192 rows 1.010, 2048
rows 0.947, 512 rows 0.855. Per 8192-row chunk the full layers lose 0.88 ms
and the upper-skip layers gain 1.23 ms. The 8192-row ratio is above the
expected 0.92 to 0.99: the epilogue costs the full tiles more than the
separate pass. The bundle therefore fuses only steps of at most 2048 rows,
where it measured faster; the bits do not depend on that choice.
