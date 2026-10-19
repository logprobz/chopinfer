//! Index scoring and candidate filtering derived from the CSA2 geometry.
//! Every rank scores all heads in the same F32 order. Reindex reads the
//! latest KV owner's keys inside the candidate source's published blocks,
//! which it reads as block IDs (`kernel_core::position_blocks`).

use kernel_core::pooled_indexer::{CandidateBlocksQuery, IndexerCandidates, PositionIndexerQuery};
use kernel_core::position_blocks::PositionBlockIndexerQuery;

use crate::geometry::{Geometry, LayerRole};
use crate::serving::PAGE_TOKENS;

pub mod prefill;

/// The source layers whose storage a layer binds on every step. Binding
/// does not depend on whether this step completed a compression group:
/// the owner republishes its existing keys on an incomplete group too.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Sources {
    pub keys: u32,
    pub selection: u32,
    pub candidates: Option<u32>,
}

pub fn sources(g: &Geometry, layer: u32) -> Option<Sources> {
    if layer >= g.layers() || g.layer_role(layer) == LayerRole::Window {
        return None;
    }
    let latest = |layers: &[u32]| layers.iter().rev().copied().find(|&l| l <= layer);
    let selection = latest(g.shape.csa.index_sources)?;
    Some(Sources {
        keys: latest(g.shape.csa.kv_sources)?,
        selection,
        candidates: (g.layer_role(selection) == LayerRole::Reindex)
            .then_some(g.shape.candidates.source),
    })
}

/// An index source's Position v3 query. Full sources score every visible
/// key, including the candidate source itself; a Reindex source scores
/// inside its block IDs (`block_selection`), so no source binds a dense
/// mask. Reuse has no new query.
pub fn selection(g: &Geometry, layer: u32, capacity: u32) -> Option<PositionIndexerQuery> {
    if layer >= g.layers() {
        return None;
    }
    match g.layer_role(layer) {
        LayerRole::Full | LayerRole::Reindex => {}
        LayerRole::Reuse | LayerRole::Window => return None,
    }
    let compression_ratio = g.compress_ratio(layer);
    Some(PositionIndexerQuery {
        heads: g.shape.indexer.heads,
        head_dim: g.shape.indexer.dim,
        compression_ratio,
        tail_rope_dims: g.shape.attention.rope_dim,
        key_norm_eps_bits: (g.shape.eps as f32).to_bits(),
        max_top_k: g.shape.indexer.top_k,
        page_size: PAGE_TOKENS as u32 / compression_ratio,
        candidates: IndexerCandidates::All,
        capacity,
    })
}

/// A Reindex source's compact query: its Position v3 geometry inside the
/// candidate source's top blocks. Other layers have none.
pub fn block_selection(
    g: &Geometry,
    layer: u32,
    capacity: u32,
) -> Option<PositionBlockIndexerQuery> {
    if layer >= g.layers() || g.layer_role(layer) != LayerRole::Reindex {
        return None;
    }
    Some(PositionBlockIndexerQuery {
        indexer: selection(g, layer, capacity)?,
        block_size: g.shape.candidates.block,
        top_k_blocks: g.shape.candidates.top_blocks,
    })
}

pub const fn candidates(g: &Geometry, capacity: u32) -> CandidateBlocksQuery {
    CandidateBlocksQuery {
        block_size: g.shape.candidates.block,
        top_k_blocks: g.shape.candidates.top_blocks,
        capacity,
    }
}

/// The completed compressed positions the owner's page table can hold.
pub fn position_capacity(g: &Geometry, layer: u32) -> Option<u32> {
    selection(g, layer, 1).map(|q| g.shape.context / q.compression_ratio)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::FLASH;
    use kernel_core::contract::Validate;

    #[test]
    fn full_and_reindex_queries_follow_the_geometry() {
        let mut owners = Vec::new();
        let mut reindex = Vec::new();
        for layer in 0..FLASH.layers() {
            let Some(q) = selection(&FLASH, layer, 8192) else {
                continue;
            };
            q.validate().unwrap();
            assert_eq!(q.heads, 32);
            assert_eq!(q.head_dim, 128);
            assert_eq!(q.tail_rope_dims, 64);
            assert_eq!(q.max_top_k, 512);
            assert_eq!(q.page_size * q.compression_ratio, PAGE_TOKENS as u32);
            assert_eq!(q.candidates, IndexerCandidates::All);
            match block_selection(&FLASH, layer, 8192) {
                None => owners.push(layer),
                Some(compact) => {
                    compact.validate().unwrap();
                    assert_eq!(compact.indexer, q);
                    assert_eq!(compact.score_slots(), Ok(16384));
                    reindex.push(layer);
                }
            }
        }
        assert_eq!(owners, [2, 8, 14, 20]);
        assert_eq!(reindex, [24, 28, 32, 36]);
        let filter = candidates(&FLASH, 8192);
        filter.validate().unwrap();
        assert_eq!((filter.block_size, filter.top_k_blocks), (8, 2048));
        assert_eq!(position_capacity(&FLASH, 2), Some(1 << 19));
        assert_eq!(position_capacity(&FLASH, 20), Some(1 << 20));
    }

    #[test]
    fn csa2_bindings_republish_each_owner_even_without_a_completed_group() {
        // At token 2 ratio-2 layers produce no new key. A prior traversal
        // ended at owner 20; binding layer 2 must still return owner 2.
        let mut previous = sources(&FLASH, 39).unwrap();
        assert_eq!(previous.keys, 20);
        for token in [1, 2, 3, 4] {
            for layer in 2..40 {
                let current = sources(&FLASH, layer).unwrap();
                let expected_key = match layer {
                    2..=7 => 2,
                    8..=13 => 8,
                    14..=19 => 14,
                    _ => 20,
                };
                let expected_selection = match layer {
                    24..=27 => 24,
                    28..=31 => 28,
                    32..=35 => 32,
                    36.. => 36,
                    _ => expected_key,
                };
                assert_eq!(current.keys, expected_key, "token {token}, layer {layer}");
                assert_eq!(current.selection, expected_selection);
                assert_eq!(current.candidates, (layer >= 24).then_some(20));
                if layer == 2 {
                    assert_ne!(current.keys, previous.keys);
                    let q = selection(&FLASH, layer, 1).unwrap();
                    assert_eq!(q.visible_positions(token), Some(token.div_ceil(2)));
                }
                previous = current;
            }
        }
        assert_eq!(sources(&FLASH, 0), None);
        assert_eq!(sources(&FLASH, 40), None);
    }
}
