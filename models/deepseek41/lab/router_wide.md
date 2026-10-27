# Served 64-expert router tiles

Native Flash TP4's 8192-row router-in-norm projection uses 64 experts per
tile, 32 rows and 128 threads. It stages one BF16 component plane, uses
24,576 shared bytes and a 6 by 256 grid. Each output retains the original
eight-instruction chain per 128 K, the same component merges and ordered
fold. Columns outside the tensor domain use the original SIMT arithmetic.

The 2026-10-21 pair evaluation at 250 W matched logits bitwise on captured
layers 14/39 and synthetic clean/outlier cases, with poisoned inactive
rows and SIMT fallback coverage. The selected projection measured 0.77 of
the 32-expert projection on the captured cases. Wider unselected tile
experiments do not change serving admission. See
[router-in-norm](router_norm.md) for the producer contract.
