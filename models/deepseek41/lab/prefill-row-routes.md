# Packed prefill router row sweep

The native Flash TP4 router admits all prefill counts from 320 through 8192
through its existing certified projection. Its weight initialization and
verify-row refusal are unchanged. The router still runs inside the input
norm only at 8192 rows. Default journal pins are unchanged.

On 2026-10-24, a two-device lab checked 1086 counts per worker: every
count from 320 through 640, every 32-row boundary and its neighbors through
8192, seven served packed counts and 64 seeded draws. The clean and outlier
fixtures are shared with the router-in-norm lab, including zero rows, split
rows, NaNs and tiny-weight columns that require SIMT fallback.

All 2172 cases matched the unchanged baseline SIMT kernel byte for byte.
The 17376 full-output comparisons covered eager and captured execution at
full, partial and zero live counts, then a restored full count. Inactive
rows and guards remained poisoned. Both workers detected both device
mutations.

After correctness passed, both devices timed paired complete graphs at
twelve counts. The candidate-to-baseline median ratios ranged from 0.0934
to 0.1820, all below the fixed 0.97 gate. The power cap was 300 W; active
clock samples reached 1792 MHz. Peak worker memory was 1480 MiB, and free
memory returned to its initial value after both phases. There were no
kernel Xids. This is kernel evidence; the combined serving comparison and
packed-prompt invariance gate remain pending.
