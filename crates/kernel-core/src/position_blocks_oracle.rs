//! Exact compact-list oracle over the kernel's own F32 scores. Score bounds
//! still use position_indexer_oracle::scores on the decoded operands.

use crate::contract::{Refusal, Validate};
use crate::pooled_indexer::{CandidateBlocksQuery, IndexerCandidates, PositionIndexerQuery};
use crate::position_blocks::PositionBlockIndexerQuery;
use crate::position_indexer_oracle;

/// The dense filter's exact membership, represented as sorted block IDs.
pub fn candidate_ids(
    query: CandidateBlocksQuery,
    scores: &[f32],
    visible: usize,
) -> Result<Vec<i32>, Refusal> {
    let mask = position_indexer_oracle::select_candidate_blocks(query, scores, visible)?;
    let mut ids = mask
        .chunks(query.block_size as usize)
        .enumerate()
        .filter_map(|(i, block)| block[0].then_some(i as i32))
        .collect::<Vec<_>>();
    ids.resize(query.top_k_blocks as usize, -1);
    Ok(ids)
}

/// Refusal corresponds to device status 1 and an empty selection. The
/// producer's visible count is compared after clipping to physical width,
/// just as Position v3 clips its device visibility.
#[allow(clippy::too_many_arguments)]
pub fn validate_ids(
    query: PositionBlockIndexerQuery,
    width: u32,
    visible: u32,
    ids: &[i32],
    current_tag: u64,
    stored_tag: u64,
    stored_visible: u32,
    source_status: i32,
) -> Result<(), Refusal> {
    query.validate()?;
    if width == 0
        || width > i32::MAX as u32
        || visible > width
        || ids.len() != query.top_k_blocks as usize
        || current_tag != stored_tag
        || visible != stored_visible
        || source_status != 0
    {
        return Err(Refusal("compact candidate row provenance mismatch"));
    }
    let mut previous = -1;
    let mut padded = false;
    let mut pinned = visible == 0;
    for &id in ids {
        if id == -1 {
            padded = true;
            continue;
        }
        if id < 0
            || padded
            || id <= previous
            || id as u64 * query.block_size as u64 >= visible as u64
        {
            return Err(Refusal(
                "compact candidate IDs are not sorted unique visible blocks",
            ));
        }
        previous = id;
        pinned |= visible > 0 && id as u32 == (visible - 1) / query.block_size;
    }
    if !pinned {
        return Err(Refusal(
            "compact candidate list dropped the pinned newest block",
        ));
    }
    Ok(())
}

/// Validate provenance separately with validate_ids. `admitted` intersects
/// compact slots with page holes. Select exactly on original position keys,
/// including signed-zero ties, with no recomputation of the F32 scores.
pub fn select(
    query: PositionBlockIndexerQuery,
    top_k: u32,
    width: u32,
    visible: u32,
    ids: &[i32],
    scores: &[f32],
    admitted: &[bool],
) -> Result<Vec<i32>, Refusal> {
    validate_ids(query, width, visible, ids, 0, 0, visible, 0)?;
    if scores.len() != query.score_slots()? as usize || admitted.len() != scores.len() {
        return Err(Refusal("compact position score shape mismatch"));
    }
    let mut values = vec![f32::NEG_INFINITY; width as usize];
    let mut mask = vec![false; width as usize];
    for (slot, &score) in scores.iter().enumerate() {
        let id = ids[slot / query.block_size as usize];
        if id < 0 {
            continue;
        }
        let index = id as usize * query.block_size as usize + slot % query.block_size as usize;
        if index < visible as usize && admitted[slot] {
            values[index] = score;
            mask[index] = true;
        }
    }
    // Convert the already computed compressed count to a ratio-one query;
    // an empty row has no admitted positions, even at token position zero.
    let dense = PositionIndexerQuery {
        candidates: IndexerCandidates::Mask,
        compression_ratio: 1,
        ..query.indexer
    };
    position_indexer_oracle::select(
        dense,
        top_k,
        u64::from(visible.saturating_sub(1)),
        &values,
        Some(&mask),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::NumericalContract;

    fn query() -> PositionBlockIndexerQuery {
        PositionBlockIndexerQuery {
            indexer: PositionIndexerQuery {
                heads: 3,
                head_dim: 160,
                compression_ratio: 2,
                tail_rope_dims: 64,
                key_norm_eps_bits: 1e-20_f32.to_bits(),
                max_top_k: 3,
                page_size: 64,
                candidates: IndexerCandidates::All,
                capacity: 8192,
            },
            block_size: 8,
            top_k_blocks: 2,
        }
    }

    #[test]
    fn compact_membership_matches_dense_with_pin_holes_and_ties() {
        let q = query();
        q.numerical_contract_id().unwrap();
        for width in [8, 16392, 1 << 20] {
            let mut scores = (0..width).map(|i| (i % 7) as f32 - 3.0).collect::<Vec<_>>();
            scores[..8].fill(f32::NEG_INFINITY);
            for visible in [0, 1, 8, width - 1, width] {
                let ids = candidate_ids(q.filter(), &scores, visible).unwrap();
                validate_ids(
                    q,
                    width as u32,
                    visible as u32,
                    &ids,
                    9,
                    9,
                    visible as u32,
                    0,
                )
                .unwrap();
                let dense =
                    position_indexer_oracle::select_candidate_blocks(q.filter(), &scores, visible)
                        .unwrap();
                for (i, &kept) in dense.iter().enumerate() {
                    assert_eq!(kept, ids.contains(&((i / 8) as i32)));
                }
            }
        }
    }

    #[test]
    fn compact_candidate_filter_refuses_visible_nonfinite_scores() {
        let q = query().filter();
        for value in [f32::NAN, f32::INFINITY] {
            for index in [0, 16] {
                let mut scores = vec![1.0; 40];
                scores[index] = value;
                assert!(candidate_ids(q, &scores, 17).is_err());
            }
            let mut scores = vec![1.0; 40];
            scores[17] = value;
            assert_eq!(candidate_ids(q, &scores, 17).unwrap(), [0, 2]);
            assert_eq!(candidate_ids(q, &[value; 40], 0).unwrap(), [-1, -1]);
        }
        assert_eq!(
            candidate_ids(q, &[f32::NEG_INFINITY; 40], 17).unwrap(),
            [2, -1]
        );
        assert!(candidate_ids(q, &[1.0; 40], 41).is_err());
    }

    #[test]
    fn compact_validation_rejects_malformed_and_stale_rows() {
        let q = query();
        let valid = [0, 2];
        assert!(validate_ids(q, 24, 17, &valid, 3, 3, 17, 0).is_ok());
        for ids in [[2, 0], [2, 2], [0, -1], [-1, 2], [0, 3], [-2, 2]] {
            assert!(
                validate_ids(q, 24, 17, &ids, 3, 3, 17, 0).is_err(),
                "{ids:?}"
            );
        }
        for (stored_tag, stored_visible, status) in [(2, 17, 0), (3, 16, 0), (3, 17, 1)] {
            assert!(
                validate_ids(q, 24, 17, &valid, 3, stored_tag, stored_visible, status).is_err()
            );
        }
        assert!(validate_ids(q, 24, 0, &[-1, -1], 3, 3, 0, 0).is_ok());
        assert!(validate_ids(q, 24, 0, &[0, -1], 3, 3, 0, 0).is_err());
    }

    #[test]
    fn compact_selection_keeps_original_ties_and_refuses_nonfinite() {
        let q = query();
        let ids = [0, 2];
        let mut scores = [0.0; 16];
        scores[0] = -0.0;
        assert_eq!(
            select(q, 3, 24, 17, &ids, &scores, &[true; 16]).unwrap(),
            [0, 1, 2]
        );
        let mut admitted = [true; 16];
        admitted[0] = false;
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            scores[0] = value;
            assert!(select(q, 3, 24, 17, &ids, &scores, &[true; 16]).is_err());
            assert_eq!(
                select(q, 3, 24, 17, &ids, &scores, &admitted).unwrap(),
                [1, 2, 3]
            );
        }
    }
}
