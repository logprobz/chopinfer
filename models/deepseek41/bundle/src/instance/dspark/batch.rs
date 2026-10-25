//! Keep full five-position drafts below the small-row dispatch boundary.
use super::Instance;
use crate::dspark::device::slice;
use crate::dspark::execution::DraftInput;
use crate::dspark::state::BLOCK_ROWS;
use model_deepseek41_contract::serving::DECODE_BATCH;

fn compact_heads(inputs: &[DraftInput]) -> bool {
    (3..=8).contains(&inputs.len()) && inputs.iter().all(|input| input.block.emit_rows == 3)
}

fn chunk_ranges(items: usize, maximum: usize) -> impl Iterator<Item = std::ops::Range<usize>> {
    let chunks = items.div_ceil(maximum);
    let small = items / chunks.max(1);
    let extra = items % chunks.max(1);
    (0..chunks).map(move |chunk| {
        let start = chunk * small + chunk.min(extra);
        start..start + small + usize::from(chunk < extra)
    })
}

impl Instance {
    pub(super) fn dspark_draft(&mut self, inputs: &[DraftInput]) -> Result<(), String> {
        // A reused successor already holds every item's draft; no chunk runs.
        if self
            .dspark
            .as_mut()
            .is_some_and(|device| std::mem::take(&mut device.prefetch.reused))
        {
            return Ok(());
        }
        // Validate the complete batch before any chunk can mutate staging.
        let device = self.dspark.as_ref().ok_or("DSpark is absent")?;
        crate::lower::dspark::metadata(device.shape.geometry, inputs)?;
        let chunk_items = device.draft_chunk_items;
        if inputs.len() <= chunk_items {
            let positions = if crate::sm120::dspark_projection::enabled(
                &device.shape.geometry,
                device.shape.ranks(),
            ) && compact_heads(inputs)
            {
                3
            } else {
                BLOCK_ROWS
            };
            #[cfg(feature = "diagnostic-taps")]
            let positions = if device
                .head_capture
                .as_ref()
                .is_some_and(|c| c.pending(inputs.len() as u32 * BLOCK_ROWS))
            {
                BLOCK_ROWS
            } else {
                positions
            };
            #[cfg(feature = "diagnostic-taps")]
            let reference = if positions != BLOCK_ROWS {
                self.draft_reference(inputs)?
            } else {
                None
            };
            self.dspark_draft_chunk_at_depth(inputs, positions)?;
            #[cfg(feature = "diagnostic-taps")]
            if let Some(reference) = reference {
                self.check_draft_prefix(inputs.len(), positions, &reference)?;
            }
            return Ok(());
        }
        #[cfg(feature = "diagnostic-taps")]
        let reference = self.draft_reference(inputs)?;
        let plane = u64::from(DECODE_BATCH * BLOCK_ROWS) * 4;
        let item_bytes = u64::from(BLOCK_ROWS) * 4;
        for range in chunk_ranges(inputs.len(), chunk_items) {
            let offset = range.start as u64 * item_bytes;
            let batch = &inputs[range];
            self.dspark_draft_chunk(batch)?;
            let device = self.dspark.as_ref().ok_or("DSpark is absent")?;
            let bytes = batch.len() as u64 * item_bytes;
            for (rank, stored) in device.ranks.iter().enumerate() {
                for (base, source) in [(0, stored.draft_ids), (plane, stored.confidence)] {
                    self.services
                        .copy_device_to_device(
                            self.ranks[rank].2,
                            slice(stored.chunk_output, base + offset, bytes)?,
                            slice(source, 0, bytes)?,
                        )
                        .map_err(|e| format!("DSpark chunk retention: {e:?}"))?;
                }
            }
        }
        // Every rank uses its existing stream. Retention precedes scratch
        // reuse, and the restored arrays precede target packing and diagnostics.
        let device = self.dspark.as_ref().ok_or("DSpark is absent")?;
        let bytes = inputs.len() as u64 * item_bytes;
        for (rank, stored) in device.ranks.iter().enumerate() {
            for (base, destination) in [(0, stored.draft_ids), (plane, stored.confidence)] {
                self.services
                    .copy_device_to_device(
                        self.ranks[rank].2,
                        slice(destination, 0, bytes)?,
                        slice(stored.chunk_output, base, bytes)?,
                    )
                    .map_err(|e| format!("DSpark chunk restore: {e:?}"))?;
            }
        }
        #[cfg(feature = "diagnostic-taps")]
        if let Some(reference) = reference {
            self.check_draft_reference(inputs.len(), &reference)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_head_boundary_requires_three_to_eight_uniform_items() {
        use crate::dspark::{execution::Noise, state::DraftBlock};
        for items in 0..=32 {
            let mut inputs = (0..items)
                .map(|i| DraftInput {
                    block: DraftBlock {
                        request: i as u64,
                        slot: i,
                        main_rows: 3,
                        compute_rows: 5,
                        emit_rows: 3,
                        verify_rows: 4,
                    },
                    bonus: 7,
                    noise: Noise::GREEDY,
                })
                .collect::<Vec<_>>();
            assert_eq!(compact_heads(&inputs), (3..=8).contains(&items));
            for index in 0..inputs.len() {
                for depth in [1, 2, 4, 5] {
                    inputs[index].block.emit_rows = depth;
                    assert!(
                        !compact_heads(&inputs),
                        "items={items} mixed index={index} depth={depth}"
                    );
                }
                inputs[index].block.emit_rows = 3;
            }
        }
    }

    #[test]
    fn chunks_balance_work_and_cover_every_retained_item() {
        for (items, expected) in [
            (0, vec![]),
            (8, vec![8]),
            (12, vec![12]),
            (16, vec![8, 8]),
            (17, vec![9, 8]),
            (24, vec![12, 12]),
            (25, vec![9, 8, 8]),
            (32, vec![11, 11, 10]),
        ] {
            assert_eq!(
                chunk_ranges(items, 12)
                    .map(|range| range.len())
                    .collect::<Vec<_>>(),
                expected,
                "items={items}"
            );
        }
        for maximum in 1..=DECODE_BATCH as usize {
            for items in 0..=DECODE_BATCH as usize {
                let ranges: Vec<_> = chunk_ranges(items, maximum).collect();
                assert_eq!(ranges.len(), items.div_ceil(maximum));
                assert_eq!(
                    ranges.iter().cloned().flatten().collect::<Vec<_>>(),
                    (0..items).collect::<Vec<_>>()
                );
                assert!(
                    ranges
                        .iter()
                        .all(|range| (1..=maximum).contains(&range.len()))
                );
                assert!(ranges.windows(2).all(|pair| {
                    pair[0].len() >= pair[1].len() && pair[0].len() - pair[1].len() <= 1
                }));
            }
        }
    }
}
