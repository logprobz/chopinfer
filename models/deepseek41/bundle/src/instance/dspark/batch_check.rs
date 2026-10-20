//! Diagnostic-only comparison of chunked and whole drafter outputs.
use super::Instance;
use crate::dspark::device::slice;
use crate::dspark::execution::DraftInput;
use crate::dspark::state::BLOCK_ROWS;

impl Instance {
    fn draft_words(&self, items: usize) -> Result<Vec<Vec<u8>>, String> {
        let device = self.dspark.as_ref().ok_or("DSpark is absent")?;
        let bytes = items * BLOCK_ROWS as usize * 4;
        device
            .ranks
            .iter()
            .enumerate()
            .map(|(rank, stored)| {
                let mut result = vec![0; bytes * 2];
                for (output, span) in result
                    .chunks_mut(bytes)
                    .zip([stored.draft_ids, stored.confidence])
                {
                    self.services
                        .copy_device_to_host(
                            self.ranks[rank].2,
                            output,
                            slice(span, 0, bytes as u64)?,
                        )
                        .map_err(|e| format!("DSpark draft comparison read: {e:?}"))?;
                }
                Ok(result)
            })
            .collect()
    }

    pub(super) fn draft_reference(
        &mut self,
        inputs: &[DraftInput],
    ) -> Result<Option<Vec<Vec<u8>>>, String> {
        let device = self.dspark.as_mut().ok_or("DSpark is absent")?;
        let Some(path) = &device.draft_check else {
            return Ok(None);
        };
        if !path.is_file() {
            // The diagnostic harness removes the marker after startup.
            // Later requests do not perform filesystem checks.
            device.draft_check = None;
            return Ok(None);
        }
        self.dspark_draft_chunk(inputs)?;
        self.check_dspark_status()?;
        self.draft_words(inputs.len()).map(Some)
    }

    pub(super) fn check_draft_reference(
        &self,
        items: usize,
        reference: &[Vec<u8>],
    ) -> Result<(), String> {
        if self.draft_words(items)? != reference {
            return Err("DSpark chunked draft IDs or confidence bits differ".into());
        }
        self.log(&format!(
            "{{\"schema\":\"chopinfer.dspark-draft-parity.v1\",\"items\":{items},\"positions\":{},\"ranks\":{},\"bitwise\":true}}",
            BLOCK_ROWS,
            reference.len()
        ));
        Ok(())
    }
}
