# SM120 grouped expert GEMM

The catalog builds MiMo V2.6 Flash and a standard MoE profile at top-8 for
decode capacities 1, 2, 4, 8, 16 and 32 tokens, plus an 8192-token prefill
capacity. MiMo uses 256 experts, hidden 4096 and per-rank intermediate 512.
The standard profile uses 128 experts, hidden 4096 and per-rank intermediate
384. Gate and up support concatenated halves and interleaved columns. Down
writes weighted BF16 routed rows. Qwen3.8 Flash (top-10 of 512 experts,
expert-sharded, FP8 block weights) has its own routes and combine in
`qwen.cuh` in the Qwen bundle's `cuda/moe/` ([Qwen3.8 Flash expert-sharded FP8 MoE](#qwen38-flash-expert-sharded-fp8-moe)).

The served MiMo and Qwen3.8 expert GEMMs are instances of one
[grouped MoE template](#grouped-moe-template). The per-model engines it
replaced are kept only as kernel-lab's bitwise references, in
`crates/kernel-lab/cuda/baselines`.

## Row-engine interface

The router writes `topk_ids` int32 `[tokens, top_k]` and `topk_weights` f32
`[tokens, top_k]`, already normalized and scaled. Permute and align writes
`expert_offsets` int32 `[experts + 1]`, `sorted_rows` int32
`[padded_rows_capacity]`, and `num_padded_rows` int32 `[1]`. Each expert range
is padded to a multiple of the compile-time row block. `sorted_rows[slot]`
is the source routed-row index `r = token * top_k + k`, or `-1` for padding.
Rows within an expert are in ascending `r` order. The row engine also writes
the device scalars `live_tokens` and `active_experts`.

The MiMo correction bias only ranks experts. The row router, the standalone
decode route and the fused router GEMM route all weight a selected expert by
its unbiased F32 sigmoid, as the oracle does.

The A16 gate/up path gathers a BF16 token row from `r / top_k`; A16 down
gathers the gate/up BF16 row at `r`. The A8 path reads pre-gathered packed
activation rows indexed by the padded slot. Down writes one BF16 row at
`output[r, hidden]` after multiplying by `topk_weights[r]`. The row engine
combines the top-k routed rows in fixed k order. No float atomic is used for
combine.

## Formats and splits

| Weight | BF16 activation | Quantized activation |
|---|---|---|
| MXFP4, UE8M0 per K32 | BF16 MMA after register dequantization | MXFP8 E4M3, UE8M0 per K32, native 1X block-scaled MMA |
| MXFP8, UE8M0 per K32 | BF16 MMA after register dequantization | MXFP8 E4M3, UE8M0 per K32, native 1X block-scaled MMA |
| FP8 E4M3, F32 per 128x128 | BF16 MMA after register dequantization | FP8 E4M3 per K128, native FP8 MMA with F32 block-scale promotion |
| NVFP4, UE4M3 per K16 and F32 global | BF16 MMA after register dequantization | NVFP4, `ldmatrix.b8x16.b4x16_p64` expansion and repacking into native UE4M3 4X block-scaled MMA fragments, with F32 global promotion |

MiMo MXFP4 weights use `Mxfp4MoeFragmentLayout`: 16 rows by 64 K per 512-byte
tile, with one parallel 32-byte scale tile; expert rows are concatenated,
gate rows precede up rows, and the low nibble is even K. MiMo A8 prefill and
decode bulk-copy one packed K64 tile into shared memory and form native MXFP8
MMA operands from 16-byte lane fragments. The standard profile keeps a 2D
tensor-map weight path with `ScaleTileLayout` scale tiles.

Gate/up writes `SiLU(gate) * up` to BF16. MiMo also catalogs a fused MXFP8
output epilogue for its A8 path (each quantizing warp owns 32 output
columns and writes E4M3 codes plus a UE8M0 scale per routed row); the row
engine permutes this output before the A8 down GEMM. Other quantized outputs
use the row quantizer.

Split-K: the non-template gate/up paths use four splits at capacities 1 and
2, two at 4 and 8, and one at 16 and 32; FP8 block, MXFP8 and fused quantized
gate/up use at least two. Down and prefill use one split. With several
splits, compute writes raw F32 gate/up partials to a caller-provided
workspace indexed by split, routed row, output column, then gate or up (split
stride from the device `live_tokens * 8`, allocation from token capacity),
and a second entry (`reduce.cuh`) sums splits in ascending order and applies
the epilogue. Compute writes the partials of the routed rows in the expert
ranges it visits, and the reduction writes exactly those rows, so a split
entry leaves every other row's output as the unsplit entry does. Compute
writes every such partial before reduction reads it, so graph replay needs
no semaphore reset. The planner returns the workspace size and both
launch plans. Launches above 48 KB of dynamic shared memory need the
per-function opt-in.

## PDL, determinism and refusals

The entry calls `griddepcontrol.launch_dependents` at the start. The active
expert set is unknown until routing completes, so weight addresses that
depend on it are selected after `griddepcontrol.wait`. All CTAs read
`live_tokens`, `num_padded_rows`, `active_experts`, offsets and sorted rows
from device memory after the wait. Grid dimensions depend only on prepared
capacity. A CTA writes each routed output element once, without atomics, and
split partials are read in ascending split order, so replay of identical
inputs and route metadata uses the same instruction order.

The planner refuses unlisted weight and activation pairs, non-BF16 base
output, residual fusion, unsupported output quantizers, per-expert capacity,
and live split factors that differ from the compiled factor. A fused
deterministic token combine is cataloged for MiMo decode and preferred at one
token.

## Grouped MoE template

`grouped.cuh`, `grouped_ring.cuh` and `grouped_prefill.cuh` hold one
grouped-MoE template for every model. `prefill_staging.cuh` keeps the
persistent schedule's staging pieces, and `fragments.cuh` the fragment
helpers. The
fixed-shape MiMo down combine is in
`models/mimo26/bundle/cuda/moe/mimo26_fragments.cuh`; model-specific routing
lives in the owning bundle's `cuda/moe/`. Every instance is compiled per model at its own
shape. The template takes the weight format, the rank shape (local experts,
top-k, hidden, per-rank intermediate), the projection and bucket, and a
schedule struct with the tiling, split-K, ring depths and staging choices.
Routing kernels (`route.cuh`, `route_parallel.cuh`, Qwen's router,
permutation and combine) stay model owned; the template consumes the common
route metadata (`expert_offsets`, `sorted_rows`, `num_padded_rows`,
`live_tokens`, `active_experts`, `topk_weights`) and keeps the served entry
ABI.

| Served path | Template schedule | Parameters that differ by model |
|---|---|---|
| Qwen FP8 decode, every bucket 1-32 | ring | 10 routes; warps and splits from Qwen's `tuning/moe.json` (gate/up 10 x 10 up to 4 tokens, 5 x 5 above; down 8 or 4 warps, one split) |
| MiMo MXFP4 decode, gate/up and down 1-16 | ring | 8 routes; 16 warps at one-token gate/up, else 8, splits equal to warps for gate/up |
| MiMo MXFP4 decode and verify above 16 rows | persistent prefill | the prefill entry and route |
| Qwen FP8 prefill 8192 | persistent prefill | 3 weight and 2 activation stages |
| MiMo MXFP4 prefill 8192 | persistent prefill | weight and activation stages per row: TP4 gate/up 6 and 2, TP4 down 3 and 2, TP2 gate/up 3 and 3, TP2 down 2 and 4; gather trimming on |

- Ring (`grouped_ring.cuh`): one CTA per (active expert, output tile) item
  over the full K, with `warps / split_k` strips of 16 columns. Each warp
  streams its fragments and scales through its own `cp.async` ring as soon
  as the expert ID is known; above one token each K64 step's routed rows
  travel in the weight ring slot, at one token the CTA stages its row once.
  K splits cover whole 128-K blocks for FP8, and the first split's warp adds
  the later splits' F32 sums from shared memory in split order.
- Persistent prefill (`grouped_prefill.cuh`): one CTA per SM with two math
  warpgroups and a producer warpgroup; a work item is 256 weight rows (gate/up:
  128 gate rows and the matching up rows) by up to 128 routed rows, reduced
  over K in K64 stages within one CTA. Each CTA builds its schedule from
  `expert_offsets` and `sorted_rows`, skips tiles of pure padding, and takes
  items in snake order across rounds (full items first, then partial ones,
  largest first), without atomics.

Epilogues and rounding points: gate/up writes
`BF16(g / (1 + exp(-g)) * u)` of the F32 gate and up sums; down writes
`BF16(y * topk_weights[r])`. MXFP4 folds each K32 block's power of two into
the BF16 operand (the scaled E2M1 conversion is exact) and accumulates in
F32. FP8 converts codes exactly to BF16; ring decode accumulates each 128-K
block into a temporary and promotes it once with `fmaf(block, scale, sum)`,
while prefill keeps its 32 x 128 tile in output units and promotes each K32
half of each 16-row group the same way, skipping blocks whose scale is zero
(no sum is ever rescaled by a ratio of scales). The remaining differences
between MiMo and Qwen (when the one-token row load issues, whether routed rows
are reloaded from `sorted_rows`, bulk-copied weight strips, gather trimming)
are schedule flags that change no arithmetic operation or its order.

### Format parameterization

`Weights` names every format. The fragment schedules serve `mxfp4` and
`fp8_block128` and refuse the others at compile time; NVFP4 and MXFP8 decode
run the packed ring below. Every format follows one rule: MMA operands are
exact BF16 values, and any F32 scale multiplies F32 sums.

- `nvfp4`: operand `BF16(e2m1 * ue4m3)`, exact for every finite code and
  local scale. The per-expert F32 global (gate and up distinct) is applied
  once with `__fmul_rn` to each full F32 sum, after split or portion
  reduction and before SwiGLU or the route weight.
- `mxfp8`: operand from the scaled E4M3 conversion with the K32 UE8M0 scale
  (admission rejects payloads whose products are not exact).
- `mxfp4`: the same scaled E2M1 conversion in dense and grouped paths.
  Admission rejects non-finite or inexact BF16 products and scale byte 255.
  All fragment schedules and native fused-down entries rely on admission
  for exact operands. They have no operand guards or traps and retain the
  original nonvolatile scaled conversion. Paired read-sweep timing preserved
  every output bit but rejected the guards under the 1% per-path limit:
  prefill gate/up at one live row rose from 0.0676 to 0.561 ms (2026-10-03),
  and ring decode reached a 52.6% regression at capacity four with four live
  rows and skewed routing (2026-10-04). Stream and fused-down guards also
  failed. Restoring the original fragment converter avoids their cost.
  The packed grouped converter also relies on admission. The non-served
  dense MXFP4 converter keeps its trap for scale byte 255 and for actual
  codes that overflow at bytes 253 or 254. Exhaustive operand probes compare
  the dense and grouped conversions against the exact f64 product.
- `bf16`: operand as stored.

`grouped.cuh` holds the shared stage producer (`produce_stage`), consumer
(`consume_stage`) and staged epilogue (`store_tile`) of the persistent
schedule.

### Packed NVFP4 and MXFP8 ring decode

`grouped_packed.cuh` (`packed_ring_decode`, `src/moe/packed.rs`) is the ring
decode schedule over the packed layout that NVFP4 and MXFP8 expert banks
keep: row-major codes (K/2 bytes per row with even K in the low nibble, or K
bytes) and local scales in 16-row by 8-block 128-byte tiles
(`packed_scale_offset`), plus NVFP4's F32 globals per expert and projection.
Prepared images need no repacking, and the entry ABI and tensor map are the
packed engine's (`moe.cuh`, `chop::moe::run`), which still serves prefill and
the other packed entries. A grouped request selects the ring with knobs whose
`tile_k` is 64; `packed_ring_knobs` derives them from the entry's
`packed_ring` row of Qwen3.8's `contract/tuning/schedules.json`
(key `grouped_moe.<projection>.decode`, weights `nvfp4` or `mxfp8`, the rank
shape and bucket; knobs `strips`, `split_parts`, `depth` and optional
`prefetch`). `template::validate_packed_row` checks rows. The ring never
splits K, so no row can change the arithmetic. A packed decode entry without
a row keeps the packed engine.

A CTA owns one (active expert, output tile) item over the full K, with one
warp per 16-column strip or, for gate/up with `split_parts`, a gate warp and
an up warp per strip. Each warp streams its strip's K64 steps through its own
`cp.async` ring. Rounding points are the engine's, and so are the bits:

- K is never split; every accumulator starts at zero and adds its K16 MMAs in
  ascending K.
- NVFP4 operand: `BF16(E2M1) x BF16(UE4M3)` with `mul.rn.bf16x2`, exact
  (at most six significant bits between 2^-10 and 2688). UE4M3 codes at or
  above 0x80 are rejected at admission.
- MXFP8 operand: the engine's `scaled_mxfp8_bf16x2`, which traps on an
  inexact product.
- NVFP4 globals: `__fmul_rn` of each full F32 sum by the expert's gate, up
  or down global before SwiGLU or the route weight, then one BF16 rounding.

The grid is `min(experts, top_k x tokens)` items per output tile; CTAs past
the live items exit. Kernel-lab builds the Qwen3.8 TP4 instances of both
formats at every decode bucket (`template::packed_lab_instances`) and checks
them bit for bit against the packed engine (`chop_moe_eng_` references);
`grouped_moe_packed_ring_sweep` sweeps packed rows. The served Qwen NVFP4
decode entries are these instances in their own `_grouped_ring_v1` cubin. A
`packed_staged` one-token variant (`packed_staged_decode`) is a sweep
candidate but about 8% slower, so no row uses it.

### Verification

Every served A16 MiMo and Qwen3.8 expert entry (TP2, TP4) is a template
instance (`src/moe/template.rs`). The replaced engines are frozen in
`crates/kernel-lab/cuda/baselines` (`qwen_moe_engine_83c99d4c8.cuh`,
`mimo_moe_engine_83c99d4c8.cuh`) and built as the `chop_moe_eng_*`
reference cubins (`crates/kernel-lab/engine_references.rs`), so every input,
including ones added later, is checked bit for bit.
The frozen MXFP4 reference and served fragment paths retain the original
converter; admission enforces exactness before either is used.
`tests/moe_template.rs` (`grouped_moe_template_matches_served_entries`)
runs each pair on identical poisoned buffers: random codes over every finite
value with wide-range scales, activations over 31 binades, skewed and hot
routes, every live count of `poison::live_counts`, and real experts when the
checkpoints are set. Outputs must match bit for bit. Timing alternates engine and template A B A B.

### Schedule table and offline sweep

The template's schedules are rows of `tuning/schedules.json` tables
(`kernel_build::tuning`, schema `chopinfer-tuning-v1`): the kernel base's in
`crates/kernels-sm120/tuning/` and each model's in its contract crate
(`models/<name>/contract/tuning/`). `Catalogs::new` merges the tables of the
models a consumer chooses and refuses a key two tables hold or a row outside
its model's schedule keys (`ModelCatalog::schedule_keys`); each grouped entry
carries its resolved schedule (`Variant::schedule`) and takes its body,
threads and launch plan from it (`Variant::plan`; `Variant::engine_plan`
keeps the engine's). A row is keyed by operation
(`grouped_moe.gate_up.decode`, `grouped_moe.down.prefill`, ...), weight
format, rank shape (`experts`, `hidden`, `intermediate`, `top_k`) and bucket,
and holds the schedule (`ring`, `prefill`) and its knobs: ring
`split_k`, `warps`, `depth`, `early_row`, `reload_rows`; prefill `weight_stages`,
`activation_stages`, `trim_gather`, `bulk_weights`. A tuned row changes its
served cubin and so its bundle's lock. A new shape or topology needs rows,
not kernel code; `served_table` writes the rows a served model implies.

A row may change only knobs that keep the arithmetic: `split_k` and
`portions` decide the F32 reduction order and must stay at the engine's value
(`ScheduleKnobs::arithmetic`; `template::knobs` refuses a row that changes
them). Rows must also pass `template::legality`: the persistent prefill's
activation ring must be shallower than an item's K64 stages, or the row
producer can overwrite the route slot of an item still being stored. A row
must have a planner-accepted launch plan within the candidates'
shared-memory limit (`template::fits`). `template::validate_row` applies
every row check for the build, the candidates and the merge.

The offline sweep is `grouped_moe_schedule_sweep` in
`crates/kernel-lab/tests/moe_template.rs`:

| Variable | Meaning |
|---|---|
| `CHOPINFER_TUNE_MODEL`, `CHOPINFER_TUNE_RANKS` | a model directory name under `models/` and a served rank count (2 or 4) |
| `CHOPINFER_TUNE_OPS`, `CHOPINFER_TUNE_BUCKETS` | optional comma lists that narrow the rows |
| `CHOPINFER_TUNE_PHASE` | `plan` (candidate counts), `compile` (no GPU) or `run` (default) |
| `CHOPINFER_TUNE_OUT` | output directory outside the tree |
| `CHOPINFER_TUNE_MARGIN`, `CHOPINFER_TUNE_REPETITIONS` | replacement margin (0.01) and A B A B repetitions (7) |
| `CHOPINFER_TUNE_LIVE` | optional wide counts (of 32, 64, 128, 256) at which to time prefill rows instead of at capacity |
| `CHOPINFER_TUNE_ROUTES` | optional route sets per live count (`chopinfer.tune-routes.v1`), for example recorded router choices |

For each selected row, `template::candidates` lists every knob set of the
same arithmetic that the static checks and dynamic shared memory admit. The
sweep compiles them (refusing any with local or stack bytes), checks each
against the engine on the poisoned cases bit for bit (a difference is a
`template::legality` defect and is reported as illegal), times survivors
A B A B against the current entry, and replaces the row only if a candidate
beats it by more than its noise band plus the margin twice in a row
(`tuning::best`, `tuning::accepts`). A row whose null control (the current
entry against itself) would pass is kept and listed as unstable. Outputs are
`plan.json`, `compile.json`, `refused.txt`, `progress.jsonl`, `sweep.json`,
`changed.txt` and `rows.json` (a canonical partial table).

Decode and verify steps wider than a model's decode buckets run its prefill
entries at their live rows, so prefill rows are also checked at 32, 64, 128
and 256 live rows (and on the first given route set of each), and each
candidate still equal is captured once in a graph at its capacity plan and
replayed while the route metadata shrinks from 256 rows to zero; every
replay must write exactly its rows and equal the engine. With
`CHOPINFER_TUNE_LIVE` the sweep times prefill rows at those counts instead
of at capacity: per count a null control, every equal candidate screened on
the first route set, and the best confirmed on the other sets and fresh
skewed routes, A B A B in 32-launch batches that rotate through four weight
banks so weights come from DRAM. Winners are then timed at capacity and
half capacity as non-regression controls. Rows do not change in this mode;
a wide winner would need its own compiled selection for wide steps.
`grouped_moe_schedule_merge` folds several runs' `rows.json` into the
committed tables, writing each to `CHOPINFER_TUNE_TABLE_OUT/<model>.json`.
Swept rows were up to 34.5% faster per launch, but end to end at TP4 the
gain was within restart noise.

## MiMo schedules

- Decode up to 16 tokens (ring, decode route): gate/up CTAs have eight
  warps (sixteen at one token), one K split each, on one 16-column strip;
  down CTAs have eight warps on eight strips (four at TP2's 16 rows) and one
  K split, bucket 1's order. No split workspace or reduction launch. At 32
  rows each warp's slot holds the bucket's rows, the ring is one deep, and
  the prefill entries are faster (2026-10-05 sweep).
- Decode and verify buckets above 16 tokens route as prefill does and
  run the persistent prefill entries (the contract's
  `workspace::moe::prefill_route`). Each routed row's sums are reduced over
  K within one CTA, so they do not depend on the other rows' routes. The
  prefill rows are tuned for these steps too: the 2026-10-05 sweep at 32-256
  live rows (`CHOPINFER_TUNE_LIVE`, real routes) found gather trimming with
  shallower weight rings 6-12% faster per launch on TP4 down and 7-9% on TP2
  at 32 rows. Per launch at 8192 and 4096 rows, TP4 down measured 0.995-0.996
  and TP2 down 0.990-0.994 of the old rows; TP2 gate/up's capacity timing was
  inconclusive (4.5-8% noise), so a served TP2 prefill A/B at 8K and 32K
  prompts decides that row (it reverts below 0.99). Served AR tok/s against
  the old rows: TP4 C32 1.013-1.015 and C64 1.019, TP2 C32 1.026-1.033, TP2
  C16 (ring entries) unchanged. TP4 gate/up keeps its row: shallower rings
  gained 3-5% at 32-64 rows but lost 2-3% at 256.

A decode row's expert output therefore depends only on its own routes and
the bucket. Kernel-lab checks this at buckets 2-256
(`fragment_experts_ignore_their_companions_at_decode_buckets`). The served
cost, the paths that stay coupled (bucket boundaries and the verify bucket)
and the requalification are in
[numerics](../../../../docs/numerics.md#batch-shape).
- Prefill 8192 (persistent): output is bitwise identical in eager, graph
  and serialized runs.

### Exact entries for verify rows (2 to 8 tokens)

A token's output rows depend on its bucket's reduction order (gate/up sums 16
K splits at one token and 8 at 2, 4 and 8; down sums one at every bucket).
The exact entries (`Variant::exact`, symbols
`chop_moe_{profile}_{gateup,down}_exact_mxfp4_a16_t{N}_v1`, MiMo's
`kernels::moe::EXACT_BUCKETS`) give a bucket the order of bucket 1. They
exist where the decode entry differs (gate/up at 2, 4 and 8), at
every MiMo rank count, as template instances keyed
`grouped_moe.{gate_up,down}.exact`. MiMo's exact verify rows run them, so
every token's rows equal the token's rows at bucket 1. Kernel-lab
`grouped_moe_mimo_exact_rows_equal_one_token` checks this bit for bit;
`grouped_moe_mimo_exact_rows_bench` times both.

## Qwen3.8 Flash expert-sharded FP8 MoE

The TP4 layout places 128 of the 512 routed experts on each rank and
replicates the activations. `qwen.cuh` (in the Qwen bundle) and `src/moe/expert_parallel.rs`
(entries in `model_qwen38_contract::sm120::moe`) implement one rank's
MoE: hidden 2560, expert intermediate 640, FP8 E4M3 weights with one BF16
`weight_scale_inv` per 128 x 128 block (uploaded as F32, which is exact),
and the contracts `row.router-softmax-topk.v1`,
`row.moe-local-expert-routes.v1` and
`row.moe-shared-expert-sigmoid-combine.v1`.

The router writes global expert IDs and weights `[tokens, 10]`: the ten
largest logits (ties to the lower expert) and their softmax weights
renormalized over the ten. A rank keeps the routes whose expert lies in
`[rank * 128, rank * 128 + 128)`; the rank is a launch scalar, so every rank
runs the same graph. Route metadata has the MiMo layout over the 128 local
experts, with `sorted_rows` padded to 16 rows per expert. A routed row keeps
its global route index `r = token * 10 + k`. Gate/up writes
`SiLU(gate) * up` at `intermediate[r, 640]` and down writes `weight * y` at
`routed[r, 2560]`, only for this rank's routes.

The combine writes this rank's partial of the layer output per token: its
routed rows in route order (preweighted BF16, summed in F32) plus
`sigmoid(x . w_gate) * shared`, where `shared` is this rank's shard of the
shared expert output and `w_gate` the replicated 2560-wide gate. Both terms
are linear in the per-rank shards, so the comm engine's all-reduce of the
partials gives the layer output. The gate dot product is reduced per CTA in
a fixed order; below a gate logit of -80 the gated product is formed in F64
(`common/sigmoid.cuh`).

The shared expert runs on the dense engine
(`chop_gemm_qwen38_shared_gate_up_*`, `chop_gemm_qwen38_shared_down_*`). Its
160 per-rank intermediate columns are zero-padded to 256 by the weight plan
(gate/up 512 x 2560, down 2560 x 256), which is exact. Expert weights use
`FragmentWeightLayout` with `Fp8E4M3` over all local experts' rows (gate rows
then up rows per expert): 16 rows by K32 per 512-byte tile. Scales stay
row-major, `[128 * N / 128, K / 128]`.

| Entry | Launch | Work |
|---|---|---|
| `chop_moe_qwen38_decode_route_t{T}` | 1 CTA | softmax top-10 per token (a warp each), local filter, 16-row padded stable permutation |
| `chop_moe_qwen38_decode_route_parallel_t32` | rows across CTAs | the 32-row decode route (see [Parallel routes](#parallel-routes)) |
| `chop_moe_qwen38_router` | a warp per token | prefill softmax top-10 |
| `chop_moe_qwen38_permute_{count,prefix,scan,scatter}` | 256-route tiles | the row engine's tile permutation over the rank's experts |
| `chop_moe_qwen38_{gateup,down}_decode_fp8block_a16_t{T}` | (active expert, strip tile) CTAs | ring decode |
| `chop_moe_qwen38_{gateup,down}_prefill_fp8block_a16_t8192` | 188 persistent CTAs | persistent prefill |
| `chop_moe_qwen38_combine` | a CTA per token | routed sum plus gated shared output |

Decode gate/up warps split K ten ways up to four tokens and five above; down
warps take one 16-row strip each (K = 640, one split). Every K split covers
whole 128-K blocks. The contract's `tuning/moe.json` sets warps and
splits per bucket; the planner sizes each warp's ring for two CTAs per SM, or
one where two would leave it shallower than two steps. Prefill uses three
weight and two activation stages of 16 KB.

## Parallel routes

`route_parallel.cuh` distributes route rows across CTAs instead of placing
all rows on one SM. One warp selects each row (`select_route_warp`, with the
one-CTA route's comparisons, expert-ID tie breaking and serial weight sum); a
release/acquire completion ticket publishes the selected IDs, and the final
CTA builds the stable expert ordering, 16-row padding, counts and active
list. The ticket reuses the zero-initialized GEMM semaphore buffer and resets
on each invocation, including empty replay.

- Qwen3.8 runs `chop_moe_qwen38_decode_route_parallel_t32_v1` for its 32-row
  decode bucket (`use_parallel_route`). MiMo's decode buckets (up to 16 rows)
  run the one-CTA decode route or the fused router, wider buckets the tiled
  prefill route.

Live counts above the capacity trap. Kernel-lab checks
(`tests/qwen38_route_parallel.rs`) cover every live count, empty and
decreasing-count replay, ties, negative scores, NaN rows and maximum padding;
IDs, weight bits and metadata must equal the one-CTA route.

## Chained expert GEMM

`chained_experts.cuh`, `route_local.cuh` and `swiglu_clamp.cuh`
(`src/moe/chained.rs`) serve SwiGLU MLPs whose contract is a
`kernel_core::gated_mlp::GatedMlpProjectionQuery`: grouped experts at any
local expert count (288 and 96 included) and dense or shared-expert MLPs, in
FP8 E4M3 with F32 128 x 128 block scales, NVFP4, MXFP8 or BF16 weights with
BF16 activations, with an optional clamped SwiGLU. No existing entry
includes these headers. `chained::catalog` lists the served TP4 dense MLP
and shared-expert instances of a 288-expert top-8 model (`TP4_288_TOP8`:
hidden 4096, dense intermediates 3072 and 512 per rank, clamp 10), at
decode (16-row tiles, 32 tokens) and prefill (64-row tiles, 8192 tokens).
Its routed experts are split over the ranks on their intermediate
(`chained::split`): all 288 experts are local on every rank, each with
expert intermediate 512 per rank, and `split::catalog` lists them at the
same capacities.

Weights keep their checkpoint layouts, row-major per expert: FP8 codes
`[rows, K]` with F32 scales `[rows / 128, K / 128]`; NVFP4 E2M1 pairs
`[rows, K / 2]` (even K in the low nibble), UE4M3 `[rows, K / 16]` and F32
globals (gate and up per expert for gate/up, one per expert for down);
MXFP8 codes `[rows, K]` with UE8M0 `[rows, K / 32]`; BF16 `[rows, K]`.
Gate/up holds the N gate rows, then the N up rows.

Arithmetic, fixed per instance:

- An F32 scale (FP8 block scale, NVFP4 global) below 1 splits into a power
  of two `2^e`, with `e` its exponent clamped at -100, and an exact residual
  `s * 2^-e` in [1, 2) (at least 2^-49 below 2^-100) (`split_scale`). `2^e`
  goes into the operand, so a large activation against a small scale cannot
  overflow F32 before the scale applies. A fold only shrinks products and
  partial sums by an exact power of two, so a chain overflows only where
  the unsplit chain would; scales of 1 and above stay whole in the residual
  (the unsplit arithmetic, which keeps cancelling chains finite). A zero
  scale zeroes the operand, so its block adds exactly zero. The folded
  result equals the unsplit one bit for bit while no product or partial sum
  leaves F32's normal range; under underflow the subnormal bits may differ,
  and the gate (`2^-22 K S`) is not claimed for sums whose products are all
  below F32's normal range.
- Scale precondition, checked at admission (`chained::admit_scales`): every
  F32 block scale and NVFP4 global is finite and not negative (zero allowed).
- MMA operands are exact BF16: an E4M3 code times `2^e`; E2M1 times its
  UE4M3 scale times `2^e` (`mul.rn.bf16x2`); E4M3 scaled by its UE8M0
  (`cvt.rn.scaled`, which traps on an inexact product, as the packed ring's
  converter does); or BF16.
- Each MMA chain starts from zero and spans one 128-K block (eight K16
  steps in ascending K). Its F32 temporary is then added into the sum:
  `fmaf(chain, residual, sum)` with the FP8 block scale's residual,
  `sum + chain` otherwise. K is never split and no float atomic is used.
- NVFP4 global residuals multiply each full sum once (`__fmul_rn`).
- Gate/up writes `BF16(silu(min(g, gate_max)) * clamp(u, -up_limit,
  up_limit))`; the clamps compare and select, so NaN stays NaN. The gate
  has no lower bound: below `common/sigmoid.cuh`'s tail (-80), where
  `exp(-g)` overflows, the unit is `g * u * exp(g)` in F64, rounded once.
  Grouped down
  writes the F32 `y * topk_weights[r]` at routed row `r`, and the route sum
  adds a token's local routes in top-k order in F32 and rounds to BF16 once
  (the rank's partial; zero for a token without local routes). Its F32 form
  runs the same sum and writes it unrounded, for a later F32 add such as a
  shared expert's residual epilogue. Dense down writes `BF16(y)`.

A CTA of four warps owns 64 output columns of one row tile; each warp's 16
columns hold both the gate and the up rows, so the clamped epilogue needs no
exchange. Grouped tiles cover one expert's padded range: the route padding
equals the row tile. Grids are capacity-sized; every CTA reads
`num_padded_rows` or `live_tokens` after `griddepcontrol.wait` (traps above
the grid's capacity) and CTAs past them exit, so a launch writes exactly the
live routed rows or tokens.

`chained_block_warps.cuh` (`chained::Schedule::BlockWarps`,
`chained::block_warp_catalog`) runs the 16-row decode projections over FP8
128 x 128 or BF16 weights with four times the CTAs and the same bits. A CTA
of four warps owns 16 output columns; warp `w` computes the chains of the
128-K blocks `w, w + 4, ...` with the same operands, fragment layout and
eight ascending K16 steps, hands them over shared memory after every four
blocks, and each thread adds its elements' chains into its sums in K order
with the same `fmaf` or add, then runs the same epilogue. Each warp's next
32 K of weights and activation rows go through a `cp.async` ring of four
stages (three for BF16; static shared memory stays under 48 KiB), and block
scales load a round ahead. Its grouped grid covers the padded rows the live
routes can need, `tokens * top_k + min(tokens * top_k, experts) * 15`,
rather than every expert's padding. Kernel-lab compares it with the column
schedule bit for bit at every live count and route pool
(`block_warps_match_the_column_schedule_bit_for_bit`), and qualifies them
behind the route kernels and a PDL producer that holds the input at NaN, in
back-to-back sequences replayed from graphs, with checks that must reject
lab-only mutants (`block_warps::Mutation`: no or a late dependency wait, a
reversed chain fold; served entries take `none`).

`chained_block_warps_fp4.cuh` runs the same schedule over NVFP4 weights
(the split's routed decode gate/up and down, `split::forms`): each operand
pair is the column schedule's expression, the E2M1 pair times its UE4M3
scale times the power of two of the expert's global (`split_scale`), each a
`mul.rn.bf16x2`; one K16 step spans one 16-value scale group, and each
thread's eight UE4M3 codes per weight row and block load a round ahead
(byte loads). Chains are added with `__fadd_rn` in K order and the full
sums multiplied once by the global's residual, as in `chained::run`. The
ring copies 16 bytes of E2M1 codes per weight row and warp. Kernel-lab runs
the split's NVFP4 routed layers through the same bitwise, oracle and
producer checks, and its own mutants at the split's shapes
(`chop_moe_chain_lab_split_bw4_fp4_mutants`) must each give an output
difference (`nvfp4_block_warp_checks_reject_every_mutant`).

`chained_decoded.cuh` (`chained::Schedule::DecodedTiles`,
`Projection::decoded_tiles`) runs the dense FP8 128 x 128 gate/up at the
64-row prefill tile with each weight code decoded once per CTA. A CTA of
384 threads owns 128 tokens by 64 output columns (the 64 gate rows and the
matching 64 up rows): warp 0 copies 64-K activation stages (three, 128-byte
rows swizzled) with `cp.async` and arrives on each stage's mbarrier when
its copies land; warps 1-3 copy their own 16-byte code chunks into two
private raw stages (`cp.async` groups) and decode them into two swizzled
BF16 stages (`mul.rn.bf16x2` of the exact code pair by the power of two of
its block scale, as the column schedule's operand); eight math warps each
run 32 gate and the matching 32 up rows by 32 tokens with `ldmatrix`
fragments. The block scales of the CTA's gate and up rows load into shared
memory once. Each element keeps the column schedule's operands, 128-K
chains from zero, `fmaf(chain, residual, sum)` in K order and clamped
SwiGLU, so outputs are bitwise the column schedule's; 8-token strips past
the live count skip their MMAs and tokens past it are neither loaded nor
written. The grid covers the token capacity in 128-token tiles; 99,328
dynamic shared bytes. Kernel-lab compares it with the column schedule at
every residue of the 128-token tile and the part sizes serving uses
(`decoded_tiles_match_the_column_schedule_bit_for_bit`), qualifies the
MLPs behind a PDL producer, relaunched, back to back and from graphs, and
rejects its mutants (`decoded_tiles::Mutation`: no or a late dependency
wait, 256-K chains).

`route_local.cuh` selects with `select_route_warp` (the scoring policies,
comparisons, lower-expert tie breaking and serial weight sum of
[parallel routes](#parallel-routes)) and filters to the rank's experts
`[rank * L, (rank + 1) * L)` for any L. Decode (up to 32 tokens) is one CTA;
prefill runs `route_select_rows` (a warp per row), `route_count_local` and
`route_scatter_local` (a CTA per local expert, routes in ascending route
order by ballots). The metadata is the common layout above, padded to the
projection's row tile. Grouped down's F32 routed rows take
`routes x hidden x 4` bytes (1 GiB at 8192 tokens, top 8, hidden 4096); a
later schedule can sum routes within the down kernel.

`chained::streamed_resident` compiles the small gate/up stage of
`chained_streamed.cuh` (`chained::streamed`, `Stage::Small`) again under an
eight-CTA launch bound (`_small_resident8_v1`). Source, operands, chains,
fold order and partial addresses are the small stage's; only register
allocation and scheduling may change, and the build rejects spills. It
accepts the 64-row ABI with at most 32 live rows.

`tests/moe_chained_build.rs` builds every served instance, dense and split
(but the split's tile pairs, which mark their live fragments with a
shared-memory `atomicOr`), and checks the audits and SASS (BF16 HMMA with
F32 accumulation, no float atomics). The
f64 oracle is `kernel_core::gated_mlp_oracle`.

### Quad-lane gate queues

The direct gate small and wide queues, the streamed module's gate small
and wide queues, the resident8 gate and the absorbing gate small queue
have builds reading their bank in quad-lane row tiles
(`row-tiles-64x128-lanes`, `chained_lanes.cuh`, `chained::lanes`). The
bank is `row-tiles-64x128-groups` with each row's 64-byte K block
reordered (`kernel_core::transform::QuadLaneLayout`): byte
`16 * g + 4 * q + t` moves to `16 * t + 4 * g + q`.

- Lane t of a quad then loads exactly the bytes it converts. Word g of its
  16-byte load holds group g's bytes, and one 4-byte load holds the row's
  four group scales.
- The row-tiled queues give each lane one 32-K group, shuffle every word to
  the quad and shift out the lane's byte. That operand movement is about
  40% of their executed instructions.
- Here each byte converts in place: `F2FP.BF16.E2M1.UNPACK_B` reads it with
  a byte selector.
- The A operands, MMAs, chains, K order and partial addresses are the
  row-tiled queues', so the partials are theirs bit for bit. Each build
  keeps its row-tiled twin's parameters, launch and residency.

The staging warp also checks four E4M3 codes per instruction and each
group's activation scale once, and the weight check tests four scales per
instruction (`chained_lanes_checks.h`). They trap on exactly the codes and
scales the row-tiled queues trap on: `tests/chained_lanes_checks.cpp`
compares them with the per-code rules and the lane bytes with the shuffled
ones on the host. `tests/chained_lanes.rs` builds the Flash backbone's four
queues beside their row-tiled twins: no spills, no more registers, as many
BF16 MMAs and no shuffles.

The column kernel's quad-lane build (`chained::Config`'s `Lanes`,
`quantized::Projection::lanes_entry`) changes only its tiled load. Each
thread loads its eight bytes of the step's 64-K half of a tile row, which
are that half's row-tiled bytes `4 * b + part`, and stores each converted
BF16 pair where the row-tiled load stores it, with the same scale. The
shared tile, and so everything after the load, is the row-tiled kernel's.

The native prefill gate/up tiles' quad-lane builds (`native_grouped.cuh`'s
`Lanes`, `native_grouped::lanes_entry`) cannot convert in place: the
block-scaled MMA takes whole 4-byte words of a row's K. Each thread copies
whole tile rows into the row-tiled shared slots instead and, after its own
stage wait, restores their row-tiled order in place (`untangle`: word g of
the four lane chunks, transposed as bytes, is chunk g) before the stage
barrier. Every warp then reads the row-tiled build's shared bytes, so the
outputs are its bit for bit. `tests/native_grouped_layout.cpp` checks the
untangle on the host, and `tests/native_grouped_build.rs` builds the fused
ceil and scanned tiles beside their row-tiled twins.

## Native block-scaled prefill tiles

`native_grouped.cuh` (`kernels_sm120::moe::native_grouped`) runs MXFP4
weights by E4M3/32 UE8M0 activations through
`mma.sync kind::mxf8f6f4.block_scale` without decoding to BF16, for any
expert count, top-k, K (a multiple of 32) and N (a multiple of 16). It
changes bits against the BF16-decoded engines and qualifies on the
within-bound track: products are exact, each 128-K chain starts from +0
and is added into the F32 accumulator in ascending K order, and the
instruction reduces each 32-K group. Lane t of a quad owns K elements
8t..8t+7 of a group in both operands, a fixed permutation of the
instruction's K slots shared by every entry; `native_grouped_layout.h`
holds the operand layout and `tests/native_grouped_layout.cpp` emulates it
on the host against the PTX fragment layout.

One warp packs the active experts into row tiles of 64 or 128 rows; a
persistent grid takes (tile, column strip) items from an integer ticket,
with three cp.async stages of 128 K. Gate/up warps hold one gate and one up
fragment of 16 columns, down warps two fragments of 32. Admission traps on
the codes and scales the exact engines trap on, and the gate/up epilogue is
`chained::before_down`. Bundles select these entries explicitly after
qualifying their served outputs.

`native_ceil.cuh` (`native_grouped::Ceil`) moves the intermediate E4M3/32
ceil quantizer (`row/block_quantize.cuh`) into the gate/up tiles: each
tile stages its BF16 outputs in the freed operand stages, quantizes every
32-column group with the quantizer's arithmetic, writes codes and scales as
if the token's row were clean and ORs the group's flags into the token's
status; a fixup launch then rewrites every live token whose status is set
to zero codes and scale 127, as the quantizer does for a flagged row. Codes,
scales and status are the separate quantizer's bit for bit.

## Diagnostics

Kernel-lab MoE switches (`tests/ops/moe.rs`, `tests/ops/moe_prefill.rs`):

- `CHOPINFER_MOE_TILED_ROUTE=1` selects the tiled route;
  `CHOPINFER_MOE_PREFILL_TOKENS` and `CHOPINFER_MOE_PREFILL_ROUTING=hot`
  (every token to experts 0-3) shape `grouped_moe_mimo_prefill_skewed`.
- `CHOPINFER_MOE_REAL_ROUTE_BENCH=1` and `CHOPINFER_MOE_47_TOKENS` run
  `grouped_moe_mimo_real_route_47_layer_graph`; `CHOPINFER_MOE_47_FUSED`
  selects fused routing there and `CHOPINFER_MOE_FUSED_ROUTER` the fused
  router elsewhere.
- `CHOPINFER_MIMO_AUDIT_CAPACITY` (default 64),
  `CHOPINFER_MIMO_AUDIT_ROUTES_ONLY=1`, `CHOPINFER_MIMO_AUDIT_GSM8K=N` and
  `CHOPINFER_MIMO_AUDIT_PROMPTS=FILE` configure `tests/mimo_model_audit.rs`.

`tools/profile/moe_smalle.py` and `tools/profile/moe_smalle_tp4.py` run
few-expert kernel and TP4 server comparisons under a deadline; see their
`--help`. GPU access rules and the receipts directory are in
[kernel-lab](../../../kernel-lab/README.md).
