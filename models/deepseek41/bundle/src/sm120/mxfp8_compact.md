# Compact dense prefill routing

The 2026-10-20 paired 250 W matrix passed bitwise comparison for all 20 Flash
shapes across TP1, TP2 and TP4, including tails, graph replay, admission,
PDL producer ordering and chunk partitions. Arithmetic remains four k32 MMAs
per 128-K chain, then F32 folds in ascending K order.

A bucket qualifies only when both event and exclusive median ratios are below
0.97 on both devices. Each cell is selected stages / worst-device speedup;
`old` retains the previous route. Counts between these exact measured rows,
other geometries and decode/verify keep their previous selections. Selection
uses the best worst-device speedup among qualified stages. No workspace is added;
output and sticky row status retain their existing per-op bindings.

| N | K | Groups | 128 | 512 | 2048 | 4096 | 8192 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 5120 | 1 | 2 / 2.370x | 2 / 1.802x | 2 / 1.832x | 2 / 1.097x | 2 / 1.389x |
| 576 | 5120 | 1 | 2 / 2.390x | 2 / 1.823x | 2 / 1.884x | 2 / 1.144x | 2 / 1.138x |
| 1024 | 4096 | 2 | 3 / 2.348x | 2 / 1.858x | 2 / 1.444x | 2 / 1.148x | 2 / 1.224x |
| 1024 | 4096 | 4 | 3 / 2.008x | 2 / 1.095x | 2 / 1.145x | 2 / 1.218x | 2 / 1.140x |
| 1024 | 4096 | 8 | 2 / 1.395x | 2 / 1.342x | 2 / 1.207x | 2 / 1.141x | 2 / 1.136x |
| 1152 | 5120 | 1 | 2 / 2.370x | 2 / 1.866x | 2 / 1.130x | 2 / 1.112x | 2 / 1.239x |
| 1280 | 5120 | 1 | 2 / 2.383x | 2 / 1.849x | 2 / 1.158x | 2 / 1.143x | 2 / 1.269x |
| 2304 | 5120 | 1 | 3 / 2.375x | 2 / 1.878x | 2 / 1.120x | 2 / 1.245x | 2 / 1.215x |
| 4096 | 1280 | 1 | 3 / 1.904x | old | old | old | old |
| 5120 | 576 | 1 | 3 / 1.673x | old | old | old | old |
| 5120 | 1152 | 1 | 3 / 1.894x | old | old | old | old |
| 5120 | 2048 | 1 | 3 / 1.951x | old | old | old | old |
| 5120 | 2304 | 1 | 3 / 1.925x | old | old | old | old |
| 5120 | 4096 | 1 | 3 / 2.481x | 2 / 1.205x | 2 / 1.356x | 2 / 1.237x | 2 / 1.234x |
| 5120 | 8192 | 1 | 3 / 2.469x | 2 / 1.230x | 2 / 1.360x | 2 / 1.243x | 2 / 1.234x |
| 5120 | 15360 | 1 | 3 / 2.306x | 2 / 1.223x | 2 / 1.296x | 2 / 1.203x | 2 / 1.197x |
| 8192 | 1280 | 1 | 2 / 1.487x | 2 / 1.089x | old | old | old |
| 16384 | 1280 | 1 | old | old | old | old | old |
| 25600 | 6144 | 1 | 2 / 1.276x | 2 / 1.125x | 2 / 1.046x | old | old |
| 32768 | 1280 | 1 | 2 / 1.133x | old | old | old | old |
