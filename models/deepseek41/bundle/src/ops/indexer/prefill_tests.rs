use crate::ops::indexer::{KeyPages, SelectViews};
use crate::{ops::indexer::prefill::Factory, shape::Shape};
use bundle_runtime::geometry::Ranks;
use kernel_core::attention::BlockTables;
use kernel_core::pooled_indexer::IndexerKeyFormat;
use model_deepseek41_contract::geometry::FLASH;
use runtime_bundle_abi::{BundleDeviceBufferHandleV1, DeviceBufferSpanV1};

fn span(id: u32, bytes: u64) -> DeviceBufferSpanV1 {
    DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_parts(id, 1), 0, bytes)
}

#[test]
fn eager_launches_replace_only_the_measured_local_selector() {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let factory = Factory::new(&shape, IndexerKeyFormat::Fp4).unwrap();
    for layer in [2, 20] {
        for context in [32768, 131072] {
            for rows in [31, 32, 448, 512, 1024] {
                let ratio = FLASH.compress_ratio(layer);
                let page = 64 / ratio;
                let width = context / ratio;
                let n = u64::from(rows);
                let views = SelectViews {
                    workspace: span(1, 1 << 30),
                    q: span(2, n * 8192),
                    head_weights: span(3, n * 128),
                    keys: KeyPages {
                        codes: span(4, u64::from(width) * 64),
                        scales: span(5, u64::from(width) * 4),
                        pages: width / page,
                        code_page_stride: u64::from(page) * 64,
                        scale_page_stride: u64::from(page) * 4,
                        page_tables: BlockTables {
                            tables: span(6, u64::from(width / page) * 4),
                            sequences: 1,
                            max_pages: width / page,
                        },
                    },
                    positions: span(7, n * 4),
                    row_batch: span(8, n * 4),
                    live_rows: span(9, 4),
                    scores: span(10, n * u64::from(width) * 4),
                    status: span(11, n * 4),
                    output: span(12, n * 2048),
                };
                let batch = factory.prepare(layer, rows, width, 1 << 30).unwrap();
                let calls = batch.select(rows, width, &views).unwrap();
                assert_eq!(calls.len(), 4);
                assert!(calls[0].launch.symbol.ends_with("_pack_fp4_guard_v1"));
                assert!(calls[1].launch.symbol.ends_with("_mma_fp4_guard_fast_v1"));
                assert!(calls[3].launch.symbol.ends_with("_select_v3_live_merge_v1"));
                let cached = context == 131072 && rows >= 32;
                assert_eq!(
                    calls[2].launch.symbol.ends_with("_select_v3_cached4_v1"),
                    cached
                );
                assert_eq!(calls[2].launch.plan.grid, [32, rows, 1]);
                assert_eq!(calls[2].launch.plan.block, [256, 1, 1]);
            }
        }
    }
}

#[test]
fn eager_factory_keeps_unmeasured_bands_and_nondefault_counts() {
    for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
        for top_k in [1, 31, 512] {
            let mut shape = Shape::new(&FLASH, ranks).unwrap();
            shape.runtime.index_topk = top_k;
            let factory = Factory::new(&shape, IndexerKeyFormat::Fp4).unwrap();
            for layer in [2, 8, 14, 20] {
                for context in [32768, 65536, 131008, 131072, 131136, 262144] {
                    for rows in [1, 31, 32, 65, 448, 512, 1024, 4096] {
                        let width = context / FLASH.compress_ratio(layer);
                        let prepared = factory.prepare(layer, rows, width, 1 << 30);
                        if layer == 20 && rows == 4096 {
                            // Candidate publication exceeds this fixture's
                            // workspace; serving plans this source in 512 rows.
                            assert_eq!(
                                prepared.err().as_deref(),
                                Some("bounded prefill exceeds the existing indexer workspace")
                            );
                            continue;
                        }
                        let batch = prepared.unwrap();
                        assert_eq!(
                            batch.uses_cached_selection(),
                            context == 131072 && rows >= 32 && top_k == 512
                        );
                    }
                }
            }
            assert!(Factory::new(&shape, IndexerKeyFormat::Bf16).is_none());
            shape.runtime.indexer_mma = false;
            assert!(Factory::new(&shape, IndexerKeyFormat::Fp4).is_none());
        }
    }
}
