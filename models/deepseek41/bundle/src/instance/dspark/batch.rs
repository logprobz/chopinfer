//! Keep full five-position drafts below the small-row dispatch boundary.
use super::Instance;
use crate::dspark::device::slice;
use crate::dspark::execution::DraftInput;
use crate::dspark::state::BLOCK_ROWS;
use model_deepseek41_contract::serving::DECODE_BATCH;

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
            return self.dspark_draft_chunk(inputs);
        }
        #[cfg(feature = "diagnostic-taps")]
        let reference = self.draft_reference(inputs)?;
        let plane = u64::from(DECODE_BATCH * BLOCK_ROWS) * 4;
        let item_bytes = u64::from(BLOCK_ROWS) * 4;
        for (chunk, batch) in inputs.chunks(chunk_items).enumerate() {
            self.dspark_draft_chunk(batch)?;
            let device = self.dspark.as_ref().ok_or("DSpark is absent")?;
            let offset = (chunk * chunk_items) as u64 * item_bytes;
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
