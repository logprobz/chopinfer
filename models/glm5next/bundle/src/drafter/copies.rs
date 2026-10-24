//! Ordered copies of the target's hidden rows into MTP staging or history.
//! Only consecutive, equally spaced rows batch. Each destination row is
//! distinct within a run; phase changes and ring wraps end it.

use bundle_runtime::drafter::Retain;
use model_glm5next_contract::contract::{DRAFT_ITEMS, head_rows};

/// The immutable live-count table covers either phase's staging capacity.
pub fn capacity() -> u32 {
    DRAFT_ITEMS.max(head_rows())
}

pub fn counts() -> Vec<u8> {
    (0..=capacity()).flat_map(u32::to_le_bytes).collect()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Run {
    pub first: Retain,
    pub rows: u32,
    /// Strides in whole hidden rows.
    pub source_stride: u32,
    pub destination_stride: u32,
}

impl Run {
    pub fn contiguous(self) -> bool {
        self.source_stride == 1 && self.destination_stride == 1
    }
}

/// Keep original order, including repeated destinations between runs.
/// Contiguous pairs use one DMA. A strided run needs four rows to replace
/// its individual copies with a launch; shorter runs keep the old copies.
pub fn runs(copies: &[Retain]) -> Vec<Run> {
    let mut result = Vec::new();
    let mut start = 0;
    while start < copies.len() {
        let first = copies[start];
        let step = |a: Retain, b: Retain| {
            if a.phase != b.phase {
                return None;
            }
            let source = b.staged.checked_sub(a.staged)?;
            let destination = b.retained.checked_sub(a.retained)?;
            (source > 0 && destination > 0).then_some((source, destination))
        };
        let strides = copies.get(start + 1).and_then(|&next| step(first, next));
        let mut rows = 1;
        if let Some(strides) = strides {
            while start + rows < copies.len()
                && rows < capacity() as usize
                && step(copies[start + rows - 1], copies[start + rows]) == Some(strides)
            {
                rows += 1;
            }
            if rows >= 4 || strides == (1, 1) {
                result.push(Run {
                    first,
                    rows: rows as u32,
                    source_stride: strides.0,
                    destination_stride: strides.1,
                });
                start += rows;
                continue;
            }
        }
        result.push(Run {
            first,
            rows: 1,
            source_stride: 1,
            destination_stride: 1,
        });
        start += 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expand(runs: &[Run]) -> Vec<Retain> {
        runs.iter()
            .flat_map(|run| {
                (0..run.rows).map(|row| Retain {
                    phase: run.first.phase,
                    staged: run.first.staged + row * run.source_stride,
                    retained: run.first.retained + row * run.destination_stride,
                })
            })
            .collect()
    }

    #[test]
    fn decode_slots_batch_and_ring_wraps_keep_original_order() {
        for count in [0, 1, 2, 3, 4, 8, 16, 32] {
            for position in 0..super::super::RETAINED_ROWS {
                let copies = (0..count)
                    .map(|slot| Retain {
                        phase: 1,
                        staged: slot,
                        retained: slot * super::super::RETAINED_ROWS + position,
                    })
                    .collect::<Vec<_>>();
                let plan = runs(&copies);
                assert_eq!(expand(&plan), copies);
                assert_eq!(plan.len(), if count >= 4 { 1 } else { count as usize });
            }
        }
        let copies = [5, 0, 1, 2, 3, 4, 5, 0]
            .into_iter()
            .enumerate()
            .map(|(i, retained)| Retain {
                phase: 1,
                staged: i as u32,
                retained,
            })
            .collect::<Vec<_>>();
        assert_eq!(expand(&runs(&copies)), copies);
        assert_eq!(
            runs(&copies).iter().map(|r| r.rows).collect::<Vec<_>>(),
            [1, 6, 1]
        );
    }

    #[test]
    fn mixed_phases_rejections_and_repeated_destinations_keep_copy_semantics() {
        let mut seed = 81_u64;
        for _ in 0..1000 {
            let copies = (0..64)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    Retain {
                        phase: (seed >> 32) as usize % 2,
                        staged: (seed >> 40) as u32 % 32,
                        retained: (seed >> 48) as u32 % 192,
                    }
                })
                .collect::<Vec<_>>();
            assert_eq!(expand(&runs(&copies)), copies);
        }
        let copies = (0..32)
            .map(|row| Retain {
                phase: 0,
                staged: row * 512 + 511,
                retained: row,
            })
            .collect::<Vec<_>>();
        let plan = runs(&copies);
        assert_eq!(plan.len(), 1);
        assert_eq!(expand(&plan), copies);
        assert_eq!(counts().len(), (capacity() as usize + 1) * 4);
    }
}
