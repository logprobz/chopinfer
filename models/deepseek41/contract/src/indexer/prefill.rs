//! Eager prefill batches over the exact host metadata uploaded for these rows.
//!
//! The guarded scorer and both selectors already accept a runtime score stride.
//! Bounding that stride lets more rows share the existing score allocation.
//! Each device row still uses its own causal count. This plan must be rebuilt
//! when the uploaded positions change; it is not a decode graph plan.

use kernel_core::contract::Refusal;
use std::ops::Range;

/// The existing full-index score allocation, also used by compact selection.
pub const SCORE_BYTES: u64 = 256 << 20;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Batch {
    pub rows: Range<u32>,
    /// Physical score stride, rounded to a key page and candidate block.
    pub width: u32,
}

/// Plan only the rows whose metadata is supplied, including disjoint packed
/// items and upper-layer suffixes. No monotonicity or last-row shortcut is used.
pub fn batches(
    positions: &[i32],
    ratio: u32,
    capacity: u32,
    score_bytes: u64,
    max_rows: u32,
) -> Result<Vec<Batch>, Refusal> {
    if !matches!(ratio, 1 | 2) || capacity == 0 || !capacity.is_multiple_of(64 / ratio) {
        return Err(Refusal(
            "prefill indexer requires whole ratio-1 or ratio-2 pages",
        ));
    }
    let rows = u32::try_from(positions.len()).map_err(|_| Refusal("prefill rows overflow"))?;
    if rows == 0 || rows > crate::serving::PREFILL_TOKENS || max_rows == 0 {
        return Err(Refusal("prefill indexer rows exceed one chunk"));
    }
    // Match position_indexer::visible, including empty rows and the capacity
    // clamp. i32::MAX + 1 must be evaluated after widening.
    let width = positions
        .iter()
        .map(|&p| {
            if p < 0 {
                0
            } else {
                ((p as u64 + 1) / u64::from(ratio)).min(u64::from(capacity)) as u32
            }
        })
        .max()
        .unwrap()
        .max(1)
        .next_multiple_of(64 / ratio);
    let fit = (score_bytes / (u64::from(width) * 4)).min(u64::from(rows.min(max_rows))) as u32;
    if fit == 0 {
        return Err(Refusal("prefill score allocation cannot hold one row"));
    }
    // Keep the old block boundary where possible. The largest capacity
    // returns the old 64-row schedule under the existing budget.
    let fit = if fit >= 64 { fit / 64 * 64 } else { fit };
    Ok((0..rows)
        .step_by(fit as usize)
        .map(|first| Batch {
            rows: first..(first + fit).min(rows),
            width,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_chunks_fit_the_existing_score_budget() {
        for ratio in [1, 2] {
            for (context, expected) in [(8192, 1), (32768, 4), (131072, 16)] {
                let positions: Vec<_> = (context - 8192..context).collect();
                let plan =
                    batches(&positions, ratio, (1 << 20) / ratio, SCORE_BYTES, 8192).unwrap();
                assert_eq!(plan.len(), (expected / ratio).max(1) as usize);
                assert_eq!(plan.first().unwrap().rows.start, 0);
                assert_eq!(plan.last().unwrap().rows.end, 8192);
                for batch in plan {
                    assert!(
                        u64::from(batch.rows.len() as u32) * u64::from(batch.width) * 4
                            <= SCORE_BYTES
                    );
                    assert_eq!(batch.width, context as u32 / ratio);
                }
            }
        }
        assert_eq!(
            batches(&vec![(1 << 20) - 1; 8192], 1, 1 << 20, SCORE_BYTES, 8192)
                .unwrap()
                .len(),
            128
        );
        assert_eq!(
            batches(&vec![8191; 8192], 2, 1 << 19, SCORE_BYTES, 4096)
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            batches(&vec![8191; 8192], 1, 1 << 20, SCORE_BYTES, 512)
                .unwrap()
                .len(),
            16
        );
    }

    #[test]
    fn partial_skip_and_unsorted_rows_use_every_position() {
        for rows in [1, 63, 64, 65, 127, 511, 512, 513, 4033, 4096, 8191] {
            let mut positions: Vec<_> = (131072 - rows..131072).collect();
            positions.reverse();
            for ratio in [1, 2] {
                let plan =
                    batches(&positions, ratio, (1 << 20) / ratio, SCORE_BYTES, 8192).unwrap();
                let covered: Vec<_> = plan.iter().flat_map(|b| b.rows.clone()).collect();
                assert_eq!(covered, (0..rows as u32).collect::<Vec<_>>());
                assert!(plan.iter().all(|b| b.width == 131072 / ratio));
            }
        }
        assert_eq!(
            batches(&[-1, 0], 2, 128, SCORE_BYTES, 8192).unwrap()[0].width,
            32
        );
        assert_eq!(
            batches(&[i32::MAX, -1], 1, 128, SCORE_BYTES, 8192).unwrap()[0].width,
            128
        );
        assert_eq!(
            batches(&[8191, 0], 1, 1 << 20, SCORE_BYTES, 8192).unwrap()[0].width,
            8192
        );
    }

    #[test]
    fn invalid_geometry_and_insufficient_storage_are_refused() {
        for (positions, ratio, capacity, budget) in [
            (vec![], 1, 64, SCORE_BYTES),
            (vec![0; 8193], 1, 64, SCORE_BYTES),
            (vec![0], 0, 64, SCORE_BYTES),
            (vec![0], 3, 64, SCORE_BYTES),
            (vec![0], 1, 0, SCORE_BYTES),
            (vec![0], 1, 65, SCORE_BYTES),
            (vec![63], 1, 64, 255),
        ] {
            assert!(batches(&positions, ratio, capacity, budget, 8192).is_err());
        }
    }
}
