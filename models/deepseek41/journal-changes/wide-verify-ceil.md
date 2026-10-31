# Wide verify ceil quantizers

At Flash TP4 with more than 64 verify rows, the MoE input quantizer, the
dense projection input quantizers and the latent window write prepare one
op per wide verify bucket the instance reaches (`verify_bounds::WIDE`, 128
and 192 rows, each capped by the verify rows). Each decode ceil entry
registered at 64 rows gains a copy at 192 rows with the same symbol, so a
verify step of 65 to 192 rows launches the entry a 64-row step launches
instead of the prefill entry. Steps of 64 rows or fewer, prefill and
drafter forwards keep their entries.

The [change specification](wide-verify-ceil.json) is the identity
specification of every fixture except two journals. In the C16 odd-width
allocation fixture (verify steps of 65 rows) and the C32 maximum fixture
(192 rows), launch rules move every launch of the six serial or prefill
ceil quantizers at the step's rows to the decode symbol and its block
width, 2000 launches in each:

| Base symbol | Head symbol | Block | Launches |
|---|---|---|---|
| `moe_ceil_w5120_v1` | `moe_ceil_w5120_cached_v1` | 1024 | 640 |
| `dense_ceil_f32_w1280_prefill_w8_v1` | `dense_ceil_f32_w1280_parallel_v1` | 1024 | 384 |
| `dense_ceil_f32_w2048_prefill_w8_v1` | `dense_ceil_f32_w2048_parallel_v1` | 1024 | 320 |
| `dense_ceil_w8192_prefill_w16_v1` | `dense_ceil_w8192_cached_v1` | 1024 | 320 |
| `dense_ceil_w6144_prefill_w8_v1` | `dense_ceil_w6144_cached_v1` | 1024 | 16 |
| `latent_ceil32_v1` | `latent_ceil32_parallel_v1` | 512 | 320 |

The rules are per journal because prefill launches of the same symbols at
65 and 192 rows (the shared-prefix anchor fixtures) keep their entries and
match the verify launches in every argument. Every grid, argument,
allocation, module, resolve, launch count and operation order is
preserved; every other journal, create range and device history is
identical. The serial and prefill quantizers and the parallel and cached
ones compute each group with the same arithmetic (the serial header's group
body differs only in lane indexing), so codes, scales and row status are
unchanged.

On 2026-10-31 the journal proof of this commit against its parent was
PROVEN with full history coverage: 21 journals identical, 2 proven, 23 of
23 fixture histories compared.
