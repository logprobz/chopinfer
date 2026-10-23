# Packed fold test survey

The packed switch replaces the weight readers in the v2, ticket, prefetch,
split decode, reuse, prefill, split prefill, compact, matrix, staged-scale,
fused and wide bundle catalogs. Admission, merge, quantization and other
non-weight readers keep their symbols. A replaced compute symbol gains
`_packed_v1`; prefill TMA readers also need the packed weight tensor map.

The table identifies tests by their Rust module path below `tests/sm120`.
A catalog name denotes every shape selected by that test, not one fixed N/K.

| Tests | Comparison and bundle symbols | Effect of the switch |
|---|---|---|
| `mxfp8_fold::{model_fold_entries_prepare, model_fold_gpu}` | Every geometry's `mxfp8_fold::catalog`: `chop_deepseek41_mxfp8_fold_m*_n*_k*_g*_f32_v2`. F64 conformance, poison, PDL, row and batch invariance. | Replace v2 launches and pack the conformance weight images. Preserve the oracle and all property checks. |
| `mxfp8_fold_decode::{decode_fold_dry_run, decode_fold_gpu}` | Flash TP1/2/4 split decode against the prefill-capacity v2 fold; replay, sticky status, partials, concurrency and timing. Split admission/partials/merge symbols come from `decode_catalog`. | Pack both bundle arms and rename v2 and split partials only. |
| `mxfp8_fold_ticket::{ticket_fold_dry_run, ticket_fold_gpu}` | Ticket versus split decode or v2, every row 1..64, paired banks and preserved prefill. Calls share `mxfp8_fold_decode::compare_impl`. | Pack both bundle arms; ticket compute and split partials/v2 move. |
| `mxfp8_fold_prefetch::{prefetch_fold_dry_run, prefetch_fold_gpu}` | Prefetch versus ticket/split decode, every row 1..64, paired banks and preserved prefill. | Pack both bundle arms; prefetch and ticket compute move. |
| `mxfp8_fold_decode::modules::dry_run_refuses_missing_or_wrong_timing_probe` | Checks embedded/staged module digests and exported symbols for v2, split, ticket and prefetch. | Expect packed compute exports; preserve admission/merge/probe validation. |
| `mxfp8_fold_decode::decode_fold_topologies_are_covered` | Catalog coverage and preparation for all registered topologies. | Keep coverage assertions; preparation uses packed declarations. |
| `dspark_catch_up_kv::dspark_catch_up_kv_gpu_matches_the_prefill_shaped_fold` | N512/K5120 v2 versus split, then split versus prefetch for rows 1..8,16,32. | Inherits the packed comparison fixture. Its CPU routing check remains applicable. |
| `mxfp8_fold_stream::{stream_fold_dry_run, stream_fold_gpu}` | Staged, locked stream/deferred candidates against the bundle's selected v2/split/ticket/prefetch route. | Bundle baseline moves. Stream compute already reads packed codes; setup can bind CPU-packed banks and zero state without prepare. |
| `mxfp8_fold_stream::ahead::stream_ahead_gpu` | Two stream folds with and without weight look-ahead; checks output and final state. | Stream compute and prefetch symbols are unchanged. Bind packed banks directly for serving-style setup. |
| `mxfp8_fold_stream::sublayer::stream_sublayer_gpu` | Paired shared gate/up, activation and down stream folds, with and without look-ahead. | Stream compute and activation symbols are unchanged; use packed-bank setup. |
| `mxfp8_fold_stream::packed_layout::{packed_layout_prepare_cases_are_embedded_and_sized, packed_layout_transform_equals_prepare_on_a_gpu}` | Load-time transform against the locked stream prepare permutation, including paired banks, groups, half folds and state clearing. | Keep the deliberate row-major prepare input. This tests the historical permutation, not a serving prepare operation; stream cubins are not replaced. |
| `mxfp8_fold_bandwidth::{direct_fold_cpu_matrix, direct_fold_dry_run, direct_fold_gpu}` | Lab-owned direct variants against the selected bundle fold/stream route; output/status, graph replay, concurrency and timing. | Only the bundle baseline changes. Direct kernels already read packed banks. Preserve lab-owned cubin identity and its controls. |
| `fold_core::{cpu_catalog, prefill_reference_keeps_small_rows_on_the_served_fold, gpu_pair}` | Lab tiled/direct, row-local/deep and matrix candidates against the selected TP4 fold route. Below 128 prefill rows the baseline is v2; larger rows use compact or split/TMA prefill. | Pack and rename the bundle baseline only. Lab candidate row-major inputs remain row-major; existing packed candidates keep packed inputs. |
| `fold_core::{gpu_served_profile, profile::served_profile_catalog}` | Selected decode route against lab deep rings on synthetic/captured inputs, plus timing/profiling. The special Q-b C32 path explicitly launches a lab row-local entry. | Selected bundle routes move; the explicit row-local and deep-ring arms remain row-major. |
| `fold_core::cluster::{cluster_catalog, gpu_cluster}` | Lab cluster schedules and omission/delay controls against bundle row-local entries or the generic selected route. | Row-local comparisons stay unchanged. Generic bundle baselines move to packed codes. |
| `fold_core::plain::{plain_catalog, gpu_plain}` | Historical main cubins, lab `CHOP_FOLD_PLAIN_ROWS` rebuilds and current served cubins; graph/eager correctness, status and timing. Includes ticket, prefetch, split, reuse and row-local modules, plus deep controls. | Preserve row-major historical/plain arms and their existing locks. Only replaced current bundle readers use packed symbols/banks. Compare corresponding logical calls across layouts. |
| `fold_core::fused::{fused_catalog, gpu_fused}` | Explicit old bundle cluster/row-local schedules versus lab-owned fused/status/wide candidates and quantizer/ignore-status controls. | Unaffected. These tests do not launch the bundle's new `fold_fused` catalog. Keep their historical baselines and lab controls. |
| `compact_dense::{cpu_matrix, bitwise_gpu, timing_gpu}` | Lab compact tiles versus bundle-selected v2/compact/split/TMA prefill. Includes all rows, live-count poison, PDL and partition/chunk invariance through `matrix.rs`. | Only the bundle arm moves. Keep lab compact symbols, row-major weights and omission controls unchanged. |
| `moe_active::{active_moe_cpu_dry_run, active_moe_backbone_gpu, active_moe_dspark_gpu, profile_moe_gpu}` | Routed-expert variants with the same shared-expert side chain: v2 gate/up N576/K5120 and down N5120/K576, then the routed sum. | Pack shared-expert weights and rename these three v2 launches. Routed-expert formats and assertions stay unchanged. |
| `moe_active::prefill::{prefill_cpu_dry_run, prefill_backbone_gpu, prefill_dspark_gpu}` | Prefill routed-expert comparisons with the same shared-expert chain. | Inherits the shared-side-chain port. |
| `exact_rows::served::served_plans_match_the_executed_bitwise_recipes` and the exact-row GPU recipe callers | HC round/expand/mix and cached norm/quantize symbols resolved through `fold_cubin_for_symbol`. | Unaffected: the lookup also serves non-fold kernels. No weight-reading fold launches occur here. |

`mxfp8_fold_decode/concurrent.rs` and `compact_dense/matrix.rs` provide
assertions and execution helpers, not independent test entry points.
The stream worker tests only check ordinal selection. Fold data tests only
check fixture construction. F32 head/decode tests use the generic cubin
lookup but launch F32 projections, not MXFP8 folds. Other lab users of
`CUBINS` or `fold_cubin_for_symbol` launch HC, attention, MoE, normalization,
embedding, routing or producer probes and are unaffected.

The `fold_plain` lab recipe currently calls the bundle cubin constructors.
After those constructors switch, keep its existing row-major recipe by
building from the same catalog entries with the kernel-family constructors;
do not regenerate its lock around packed symbols.

## Production-shape parity

`packed_serving` owns one cubin with row-major and packed twins from the
bundle catalogs. It covers 367 distinct packed compute symbols in 506
comparisons across Flash TP1/TP2/TP4, deduplicating symbols shared by
multiple topologies. Decode entries exercise one row, a partial tile and
capacity. Large v2/prefill entries use 129 rows, which exercises full and
tail paths in split prefill. Exact-row entries use their smallest registered
bucket; retained unused tile/stage variants use a valid smallest bucket
for that symbol. Wide folds run both 16 and 32 rows.

Deliberately incorrect omit-first controls have no packed twin and are not
parity candidates. They remain in their existing lab families.

The CPU test checks every twin's full entry declaration and verifies that
the prepared launches cover every packed compute symbol. GPU parity compares
all data views, including status, workspace and fused quantized outputs.
Each of the twelve families also has a moved-strip negative control.
The parity run has a 600-second elapsed budget and fails on an incomplete
run instead of skipping remaining symbols. Its nominal upload estimate is
about 36 GB before paired-bank and negative-control overhead; allocation,
host packing and readback can dominate. No runtime estimate is measured
without an assigned GPU.
