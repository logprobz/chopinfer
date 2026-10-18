#[path = "../../../bundle/src/sm120/compressor.rs"]
#[allow(dead_code, reason = "the bundle build consumes the cubin declarations")]
mod compressor;
#[path = "../../../bundle/src/sm120/rows.rs"]
#[allow(
    dead_code,
    reason = "the bundle build consumes row cubin specifications"
)]
mod rows;

use compressor::{CompressorKernels, WriteQuery};
use kernel_core::backend::{Backend, Need, OpInfo};
use kernels_sm120::backend::Sm120;
use model_deepseek41_contract::admission::{SUPPORTED_RANKS, check_ranks};
use model_deepseek41_contract::geometry::GEOMETRIES;
use model_deepseek41_contract::serving;
use std::collections::BTreeSet;

#[test]
fn every_geometry_prepares_pooling_and_ordered_key_writes_at_each_rank_count() {
    let mut prepared = 0;
    for geometry in GEOMETRIES {
        let entries = CompressorKernels::serving(geometry).unwrap();
        let specs = entries.cubins().unwrap();
        assert_eq!(entries.selected().unwrap().len(), specs.len());
        let mut symbols: BTreeSet<_> = specs
            .iter()
            .flat_map(|spec| spec.entries.iter().map(|entry| entry.symbol.clone()))
            .collect();
        symbols.insert(rows::norm_symbol(
            geometry,
            geometry.shape.attention.head_dim,
        ));
        let check = |info: &OpInfo| {
            assert!(!info.symbols.is_empty());
            for (symbol, _) in &info.symbols {
                assert!(symbols.contains(symbol), "unbuilt symbol {symbol}");
            }
        };
        for &ranks in SUPPORTED_RANKS {
            check_ranks(geometry, ranks).unwrap();
            let backend = entries.register(rows::register(Sm120::new(), geometry, ranks));
            let need = Need::rows(serving::PREFILL_TOKENS);
            for entry in &entries.pooling {
                let op = backend.gated_pooling(&entry.query(), need).unwrap();
                check(op.info());
                assert_eq!(op.info().symbols.len(), 2);
                prepared += 1;
            }
            let mut published_index = BTreeSet::new();
            for (_, query) in &entries.writes {
                match query {
                    WriteQuery::Index(q) => {
                        assert_eq!(
                            u64::from(q.page_size * q.compression_ratio),
                            serving::PAGE_TOKENS
                        );
                        let op = backend.position_indexer_key_write(q, need).unwrap();
                        check(op.info());
                        published_index.insert(q.compression_ratio);
                    }
                    WriteQuery::Compressed(q) => {
                        assert_eq!(
                            u64::from(q.page_size * q.compression_ratio),
                            serving::PAGE_TOKENS
                        );
                        assert!(published_index.contains(&q.compression_ratio));
                        let op = backend.compressed_kv_write(q, need).unwrap();
                        check(op.info());
                    }
                }
                prepared += 1;
            }
            let ratio_one = entries
                .ratio_one
                .expect("the admitted models use ratio one");
            assert_eq!(ratio_one.width, geometry.shape.attention.head_dim);
            assert_eq!(ratio_one.input, kernel_core::contract::DType::F32);
            assert_eq!(ratio_one.output, kernel_core::contract::DType::F32);
            let op = backend.rmsnorm(&ratio_one, need).unwrap();
            check(op.info());
            assert_eq!(op.info().workspace, 0);
            prepared += 1;
        }
    }
    assert_eq!(prepared, GEOMETRIES.len() * SUPPORTED_RANKS.len() * 10);
    assert!(prepared > 0);
}
