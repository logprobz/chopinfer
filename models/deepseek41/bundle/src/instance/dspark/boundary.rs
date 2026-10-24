//! Join C1 validation and diagnostics without changing status or result bytes.
use super::Instance;
use crate::dspark::device::slice;
use crate::dspark::state::BLOCK_ROWS;
use runtime_bundle_abi::DeviceBufferSpanV1 as Span;

impl Instance {
    /// The caller admits only one undrafted decode phase. The saved status
    /// buffers and packet have the same lifetime as the drafted boundary.
    pub(in crate::instance) fn check_undrafted_status(&self) -> Result<bool, String> {
        let device = self.dspark.as_ref().ok_or("DSpark is absent")?;
        if !device.cycle.undrafted_status || !device.cycle.boundary_enabled {
            return Ok(false);
        }
        if device.cycle.pending.is_some() || device.results.metadata.is_some() {
            return Err("DSpark undrafted boundary contains draft results".into());
        }
        let packet = device
            .cycle
            .diagnostics
            .ok_or("DSpark boundary packet is absent")?;
        let sources = device
            .ranks
            .iter()
            .map(|rank| rank.status.pending())
            .collect::<Result<Vec<_>, _>>()?;
        let bytes = sources.iter().try_fold(0_u64, |sum, span| {
            sum.checked_add(span.length)
                .ok_or("DSpark status extent overflow")
        })?;
        if bytes > packet.length {
            return Ok(false);
        }
        self.join_saved_status(packet, &sources)?;
        let mut output = vec![0; bytes as usize];
        self.services
            .copy_device_to_host(self.ranks[0].2, &mut output, slice(packet, 0, bytes)?)
            .map_err(|error| error.to_string())?;
        let mut at = 0;
        for (rank, source) in sources.iter().enumerate() {
            let end = at + source.length as usize;
            device.ranks[rank].status.validate(rank, &output[at..end])?;
            at = end;
        }
        Ok(true)
    }

    pub(super) fn check_dspark_boundary(&self) -> Result<bool, String> {
        let Some(device) = &self.dspark else {
            return Ok(false);
        };
        let Some(pending) = &device.cycle.pending else {
            return Ok(false);
        };
        if !device.cycle.boundary_enabled
            || (!device.prefetch.enabled
                && (pending.inputs.len() != 1 || pending.drafts.len() != 1))
        {
            return Ok(false);
        }
        if pending.packed_diagnostics.borrow().is_some() {
            return Err("DSpark boundary was collected twice".into());
        }
        let packet = device
            .cycle
            .diagnostics
            .ok_or("DSpark boundary packet is absent")?;
        let sources = device
            .ranks
            .iter()
            .map(|rank| rank.status.pending())
            .collect::<Result<Vec<_>, _>>()?;
        let diagnostic_bytes = pending.drafts.len() as u64 * u64::from(BLOCK_ROWS) * 8;
        let feedback = device
            .results
            .metadata
            .as_ref()
            .filter(|_| device.prefetch.enabled)
            .map(|metadata| {
                let rows: u64 = metadata
                    .items
                    .iter()
                    .map(|item| u64::from(item.acceptance.draft_tokens) + 1)
                    .sum();
                (metadata, crate::dspark::results::SAMPLE_OFFSET + rows * 256)
            });
        let status_bytes = sources.iter().try_fold(0_u64, |sum, span| {
            sum.checked_add(span.length)
                .ok_or("DSpark status extent overflow")
        })?;
        let length = status_bytes
            .checked_add(diagnostic_bytes)
            .and_then(|length| length.checked_add(feedback.map_or(0, |(_, bytes)| bytes)))
            .ok_or("DSpark boundary extent overflow")?;
        if length > packet.length {
            // Long catch-up validation retains the original bounded reads.
            return Ok(false);
        }
        self.join_saved_status(packet, &sources)?;
        let stream = self.ranks[0].2;
        let words = diagnostic_bytes / 2;
        for (offset, source) in [
            (status_bytes, device.ranks[0].draft_ids),
            (status_bytes + words, device.ranks[0].confidence),
        ] {
            self.services
                .copy_device_to_device(
                    stream,
                    slice(packet, offset, words)?,
                    slice(source, 0, words)?,
                )
                .map_err(|e| e.to_string())?;
        }
        if let Some((_, bytes)) = feedback {
            let memory = device
                .results
                .memory
                .ok_or("DSpark successor results are absent")?;
            self.services
                .copy_device_to_device(
                    stream,
                    slice(packet, status_bytes + diagnostic_bytes, bytes)?,
                    slice(memory.packet, 0, bytes)?,
                )
                .map_err(|e| e.to_string())?;
        }
        let mut output = vec![0; length as usize];
        self.services
            .copy_device_to_host(stream, &mut output, slice(packet, 0, length)?)
            .map_err(|e| e.to_string())?;
        let mut at = 0;
        for (rank, source) in sources.iter().enumerate() {
            let end = at + source.length as usize;
            device.ranks[rank].status.validate(rank, &output[at..end])?;
            at = end;
        }
        if let Some((metadata, _)) = feedback {
            let packet = output.split_off(at + diagnostic_bytes as usize);
            *pending.feedback.borrow_mut() = Some(crate::dspark::prefetch::Feedback {
                metadata: metadata.clone(),
                inputs: pending.inputs.clone(),
                packet,
            });
        }
        *pending.packed_diagnostics.borrow_mut() = Some(output.split_off(at));
        Ok(true)
    }

    fn join_saved_status(&self, packet: Span, sources: &[Span]) -> Result<(), String> {
        let device = self.dspark.as_ref().ok_or("DSpark is absent")?;
        if sources.len() != self.ranks.len()
            || device.cycle.boundary_ready.len() != self.ranks.len()
        {
            return Err("DSpark boundary events differ from ranks".into());
        }
        // Every producer finishes its saved status before rank zero reads it.
        // This packet never aliases target, sampler or collective workspace.
        for (&ready, &(_, _, stream)) in device.cycle.boundary_ready.iter().zip(&self.ranks) {
            self.services
                .record_event(ready, stream)
                .map_err(|error| error.to_string())?;
        }
        let stream = self.ranks[0].2;
        for &ready in device.cycle.boundary_ready.iter().skip(1) {
            self.services
                .wait_event(stream, ready)
                .map_err(|error| error.to_string())?;
        }
        let mut at = 0;
        for (rank, &source) in sources.iter().enumerate() {
            if source.length != 0 {
                let destination = slice(packet, at, source.length)?;
                if rank == 0 {
                    self.services
                        .copy_device_to_device(stream, destination, source)
                } else {
                    self.services.copy_peer_to_peer(
                        stream,
                        self.ranks[0].0,
                        destination,
                        self.ranks[rank].0,
                        source,
                    )
                }
                .map_err(|error| error.to_string())?;
            }
            at += source.length;
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) struct TestView {
    pub enabled: bool,
    pub packet: runtime_bundle_abi::DeviceBufferSpanV1,
    pub status: Vec<runtime_bundle_abi::DeviceBufferSpanV1>,
    pub diagnostics: [runtime_bundle_abi::DeviceBufferSpanV1; 2],
    pub ready: Vec<(u32, u32)>,
}

#[cfg(test)]
pub(crate) fn for_test(
    handle: runtime_bundle_abi::BundleHandleV1,
    enabled: Option<bool>,
    capacity: Option<u64>,
) -> TestView {
    crate::instance::INSTANCES.with(|table| {
        let mut table = table.borrow_mut();
        let instance = table.get_mut(handle).expect("live instance");
        let device = instance.dspark.as_mut().expect("drafter");
        if let Some(enabled) = enabled {
            device.cycle.boundary_enabled = enabled;
        }
        if let Some(capacity) = capacity {
            let packet = device.cycle.diagnostics.expect("packet");
            device.cycle.diagnostics = Some(slice(packet, 0, capacity).unwrap());
        }
        TestView {
            enabled: device.cycle.boundary_enabled,
            packet: device.cycle.diagnostics.expect("packet"),
            status: device
                .ranks
                .iter()
                .map(|r| r.status.pending().unwrap())
                .collect(),
            diagnostics: [device.ranks[0].draft_ids, device.ranks[0].confidence],
            ready: instance
                .ranks
                .iter()
                .zip(&device.cycle.boundary_ready)
                .map(|(rank, event)| (rank.2.slot(), event.slot()))
                .collect(),
        }
    })
}
