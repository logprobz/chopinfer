//! Geometry-derived MXFP4 position entries: the dense Position v3
//! selection of every Full source, the compact selection of every Reindex
//! source over the candidate source's block IDs, and the block-ID producer.
//! Every rank registers the same complete index heads and reduction order.

use kernel_build::CubinSpec;
use kernels_sm120::backend::Sm120;
use kernels_sm120::pooled_indexer::{
    position, position_bf16, position_blocks, position_candidates_runtime, position_live,
    position_valid,
};
use model_deepseek41_contract::{geometry::Geometry, indexer, serving::PREFILL_TOKENS};

pub const FAMILY: &str = "position_indexer";

/// The dense selection of each Full source's query.
pub fn entries(g: &Geometry) -> Vec<position::Entry> {
    let mut entries: Vec<position::Entry> = Vec::new();
    for layer in 0..g.layers() {
        if indexer::block_selection(g, layer, PREFILL_TOKENS).is_some() {
            continue;
        }
        let Some(query) = indexer::selection(g, layer, PREFILL_TOKENS) else {
            continue;
        };
        if entries.iter().any(|e| e.query == query) {
            continue;
        }
        entries.push(position::Entry {
            name: format!("{}_r{}_all", g.kernel_tag, query.compression_ratio),
            query,
        });
    }
    entries
}

/// The compact selection of each Reindex source's query.
pub fn block_entries(g: &Geometry) -> Vec<position_blocks::Entry> {
    let mut entries: Vec<position_blocks::Entry> = Vec::new();
    for layer in 0..g.layers() {
        let Some(query) = indexer::block_selection(g, layer, PREFILL_TOKENS) else {
            continue;
        };
        if entries.iter().any(|e| e.query == query) {
            continue;
        }
        entries.push(position_blocks::Entry {
            name: format!("{}_r{}", g.kernel_tag, query.indexer.compression_ratio),
            query,
        });
    }
    entries
}

/// The candidate source's block-ID producer.
pub fn candidates(g: &Geometry) -> position_blocks::CandidateEntry {
    position_blocks::CandidateEntry {
        name: g.kernel_tag.to_owned(),
        query: indexer::candidates(g, PREFILL_TOKENS),
    }
}

/// `g`'s dense selection cubin.
pub fn cubin(g: &Geometry) -> Result<CubinSpec, String> {
    let mut specs = Vec::new();
    for entry in entries(g) {
        specs.extend(entry.specs().map_err(|e| e.0)?);
    }
    Ok(position::cubin_spec(
        format!("position_indexer_{}_sm120", g.kernel_tag),
        specs,
    ))
}

/// The name of `g`'s compact selection and block-ID producer cubin.
pub fn blocks_cubin_name(g: &Geometry) -> String {
    format!("position_blocks_{}_sm120", g.kernel_tag)
}

/// `g`'s compact selection and block-ID producer cubin.
pub fn blocks_cubin(g: &Geometry) -> Result<CubinSpec, String> {
    let mut specs = Vec::new();
    for entry in block_entries(g) {
        specs.extend(entry.specs().map_err(|e| e.0)?);
    }
    specs.extend(candidates(g).specs().map_err(|e| e.0)?);
    Ok(position_blocks::cubin_spec(blocks_cubin_name(g), specs))
}

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    let mut specs = Vec::new();
    for g in geometries {
        specs.push((FAMILY, cubin(g)?));
        specs.push((FAMILY, blocks_cubin(g)?));
    }
    Ok(specs)
}

/// Separate comparison readers leave the native cubins and locks unchanged.
pub const BF16_FAMILY: &str = "position_indexer_bf16";
pub fn bf16_cubin_names(g: &Geometry) -> [String; 2] {
    [
        format!("position_indexer_{}_bf16_sm120", g.kernel_tag),
        format!("position_blocks_{}_bf16_sm120", g.kernel_tag),
    ]
}
pub fn bf16_cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    let mut result = Vec::new();
    for g in geometries {
        let [dense_name, compact_name] = bf16_cubin_names(g);
        let mut dense = Vec::new();
        for e in entries(g) {
            dense.extend(position_bf16::dense_specs(&e).map_err(|e| e.0)?);
        }
        let mut compact = Vec::new();
        for e in block_entries(g) {
            compact.extend(position_bf16::compact_specs(&e).map_err(|e| e.0)?);
        }
        result.push((BF16_FAMILY, position_bf16::cubin_spec(dense_name, dense)));
        result.push((
            BF16_FAMILY,
            position_bf16::cubin_spec(compact_name, compact),
        ));
    }
    Ok(result)
}

/// Separately locked runtime producers; defaults never load these cubins.
pub const RUNTIME_CANDIDATES_FAMILY: &str = "position_candidates_runtime";
pub fn runtime_candidates_cubin_name(g: &Geometry) -> String {
    format!("position_candidates_runtime_{}_sm120", g.kernel_tag)
}
pub fn runtime_candidates_cubins(
    geometries: &[Geometry],
) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    geometries
        .iter()
        .map(|g| {
            Ok((
                RUNTIME_CANDIDATES_FAMILY,
                position_candidates_runtime::cubin_spec(
                    runtime_candidates_cubin_name(g),
                    position_candidates_runtime::specs(&candidates(g)).map_err(|e| e.0)?,
                ),
            ))
        })
        .collect()
}

/// A new family preserves all previous cubins and their lock records.
pub const LIVE_FAMILY: &str = "position_live";
pub fn live_cubin_name(g: &Geometry) -> String {
    format!("position_live_{}_sm120", g.kernel_tag)
}
pub fn live_cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    let mut result = Vec::new();
    for g in geometries {
        let mut specs = Vec::new();
        for bf16 in [false, true] {
            for entry in entries(g) {
                specs.extend(position_live::dense_specs(&entry, bf16).map_err(|e| e.0)?);
            }
            for entry in block_entries(g) {
                specs.extend(position_live::compact_specs(&entry, bf16).map_err(|e| e.0)?);
            }
        }
        for runtime in [false, true] {
            specs.extend(position_live::candidate_specs(&candidates(g), runtime).map_err(|e| e.0)?);
        }
        result.push((
            LIVE_FAMILY,
            position_live::cubin_spec(live_cubin_name(g), specs),
        ));
    }
    Ok(result)
}

/// Retained-prefix compact work and live-only candidate maxima.
pub const VALID_FAMILY: &str = "position_valid";
pub fn valid_cubin_name(g: &Geometry) -> String {
    format!("position_valid_{}_sm120", g.kernel_tag)
}
pub fn valid_cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    let mut result = Vec::new();
    for g in geometries {
        let mut specs = Vec::new();
        for bf16 in [false, true] {
            for entry in block_entries(g) {
                specs.extend(position_valid::compact_specs(&entry, bf16).map_err(|e| e.0)?);
            }
        }
        for runtime in [false, true] {
            specs
                .extend(position_valid::candidate_specs(&candidates(g), runtime).map_err(|e| e.0)?);
        }
        result.push((
            VALID_FAMILY,
            position_valid::cubin_spec(valid_cubin_name(g), specs),
        ));
    }
    Ok(result)
}

pub fn register(sm120: Sm120, g: &Geometry) -> Sm120 {
    sm120
        .with_valid_position_candidates()
        .with_position_indexer(entries(g))
        .with_position_indexer_bf16(entries(g))
        .with_position_block_indexer(block_entries(g))
        .with_position_block_indexer_bf16(block_entries(g))
        .with_indexer_candidate_block_ids(vec![candidates(g)])
        .with_indexer_candidate_block_ids_runtime(vec![candidates(g)])
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::backend::{Backend, Need};
    use kernel_core::properties::Ties;
    use model_deepseek41_contract::{admission, geometry::GEOMETRIES};

    /// Every Full source prepares its dense selection, every Reindex source
    /// its compact one, and the candidate source its block-ID producer, each
    /// from the registered entries.
    #[test]
    fn every_index_source_prepares_from_the_registered_entries_at_every_rank_count() {
        for g in GEOMETRIES {
            let dense = live_cubins(&[*g]).unwrap().remove(0).1;
            let blocks = valid_cubins(&[*g]).unwrap().remove(0).1;
            let registered = |symbol: &str| {
                dense
                    .entries
                    .iter()
                    .chain(&blocks.entries)
                    .any(|entry| entry.symbol == symbol)
            };
            for &ranks in admission::SUPPORTED_RANKS {
                if admission::check_ranks(g, ranks).is_err() {
                    continue;
                }
                let backend = register(Sm120::new(), g);
                for layer in 0..g.layers() {
                    let Some(query) = indexer::selection(g, layer, 8) else {
                        continue;
                    };
                    let width = indexer::position_capacity(g, layer).unwrap();
                    let info = match indexer::block_selection(g, layer, 8) {
                        None => backend
                            .position_indexer(&query, query.max_top_k, width, Need::rows(8))
                            .unwrap()
                            .info()
                            .clone(),
                        Some(compact) => backend
                            .position_block_indexer(&compact, query.max_top_k, width, Need::rows(8))
                            .unwrap()
                            .info()
                            .clone(),
                    };
                    assert_eq!(info.guarantees.ties, Ties::LowerIndex);
                    assert_eq!(info.symbols.len(), 3);
                    for (symbol, _) in &info.symbols {
                        assert!(registered(symbol), "{symbol}");
                    }
                }
                let query = indexer::candidates(g, 8);
                let width = indexer::position_capacity(g, g.shape.candidates.source).unwrap();
                let producer = backend
                    .indexer_candidate_block_ids(&query, width, Need::rows(8))
                    .unwrap();
                assert_eq!(
                    producer.info().workspace,
                    8 * u64::from(width.div_ceil(query.block_size)) * 4
                        + position_live::workspace_bytes(8, query.top_k_blocks)
                );
                for (symbol, _) in &producer.info().symbols {
                    assert!(registered(symbol), "{symbol}");
                }
            }
        }
    }
    #[test]
    fn bf16_index_sources_prepare_from_separate_cubins_at_every_rank_count() {
        use kernel_core::pooled_indexer::IndexerKeyFormat;
        for g in GEOMETRIES {
            let mut cubins = live_cubins(&[*g]).unwrap();
            cubins.extend(valid_cubins(&[*g]).unwrap());
            assert_eq!(cubins.len(), 2);
            for &ranks in admission::SUPPORTED_RANKS {
                if admission::check_ranks(g, ranks).is_err() {
                    continue;
                }
                let backend = register(Sm120::new(), g);
                for layer in 0..g.layers() {
                    let Some(q) = indexer::selection(g, layer, 8) else {
                        continue;
                    };
                    let width = indexer::position_capacity(g, layer).unwrap();
                    let info = match indexer::block_selection(g, layer, 8) {
                        None => backend
                            .position_indexer_with_storage(
                                &q,
                                IndexerKeyFormat::Bf16,
                                q.max_top_k,
                                width,
                                Need::rows(8),
                            )
                            .unwrap()
                            .info()
                            .clone(),
                        Some(c) => backend
                            .position_block_indexer_with_storage(
                                &c,
                                IndexerKeyFormat::Bf16,
                                q.max_top_k,
                                width,
                                Need::rows(8),
                            )
                            .unwrap()
                            .info()
                            .clone(),
                    };
                    assert_eq!(info.guarantees.ties, Ties::LowerIndex);
                    for (symbol, _) in info.symbols {
                        assert!(
                            cubins
                                .iter()
                                .any(|(_, c)| c.entries.iter().any(|e| e.symbol == symbol)),
                            "{symbol}"
                        );
                    }
                }
            }
        }
    }
}
