//! Event-ordered sampler submission and one host-owned completion packet.
use super::Instance;
use crate::dspark::{
    device::slice,
    execution::Input,
    results::{SAMPLE_OFFSET, validate_metadata},
};
use bundle_runtime::inputs::PackedStep;
use runtime_bundle_abi::device_results::*;
use runtime_bundle_abi::{
    BundleAbiError, BundleDeviceBufferViewV1, BundleLogitsViewV1, BundleStatusV1,
    DeviceBufferSpanV1 as Span, ExecuteResultSinkV1,
};

fn view(span: Span) -> BundleDeviceBufferViewV1 {
    BundleDeviceBufferViewV1 {
        buffer: span.buffer,
        offset: span.offset,
        length: span.length,
        ..BundleDeviceBufferViewV1::empty()
    }
}
impl Instance {
    pub(crate) fn begin_dspark_results(
        &mut self,
        sink: &mut ExecuteResultSinkV1,
        inputs: &[Input],
        step: &PackedStep,
    ) -> Result<(), String> {
        let device = self.dspark.as_mut().ok_or("DSpark is absent")?;
        device.results.metadata = None;
        if !sink.device_results_enabled()
            || !step.prefill.request_ids.is_empty()
            || !inputs.iter().any(|input| input.rows.kind.drafts() > 0)
        {
            return Ok(());
        }
        let metadata = match sink.device_cycle_metadata() {
            Ok(metadata) => metadata,
            Err(BundleAbiError::Status(BundleStatusV1::Unsupported)) => return Ok(()),
            Err(error) => return Err(error.to_string()),
        };
        validate_metadata(
            &metadata,
            inputs,
            device.shape.geometry.shape.vocab,
            device.results.last_epoch,
        )?;
        let state = device
            .runtime
            .as_ref()
            .ok_or("DSpark lifecycle is borrowed")?
            .state();
        for input in inputs {
            let context = state
                .context(input.rows.request)
                .ok_or("DSpark result request has no committed state")?;
            if context.slot != input.rows.slot {
                return Err("DSpark result state slot differs".into());
            }
        }
        sink.declare_input_dependency(DEVICE_RESULTS_HOST_VALUES_V1)
            .map_err(|e| e.to_string())?;
        let memory = device
            .results
            .memory
            .ok_or("DSpark result storage absent")?;
        for (span, bytes) in [
            (
                memory.items,
                metadata
                    .items
                    .iter()
                    .flat_map(|item| item.acceptance.to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
            (
                memory.stops,
                metadata
                    .termination_stop_ids
                    .iter()
                    .flat_map(|id| id.to_le_bytes())
                    .collect(),
            ),
        ] {
            if !bytes.is_empty() {
                self.services
                    .copy_host_to_device(
                        self.ranks[0].2,
                        slice(span, 0, bytes.len() as u64)?,
                        &bytes,
                    )
                    .map_err(|e| e.to_string())?;
            }
        }
        device.results.last_epoch = metadata.cycle_epoch;
        device.results.metadata = Some(metadata);
        Ok(())
    }

    pub(crate) fn emit_dspark_device_drafts(
        &self,
        sink: &mut ExecuteResultSinkV1,
    ) -> Result<(), String> {
        let device = self.dspark.as_ref().ok_or("DSpark is absent")?;
        let Some(metadata) = &device.results.metadata else {
            return Ok(());
        };
        let ready = device
            .cycle
            .draft_joined
            .ok_or("DSpark draft join absent")?;
        for item in &metadata.items {
            let k = item.acceptance.draft_tokens;
            if k == 0 {
                continue;
            }
            let offset = u64::from(item.acceptance.draft_item_index) * 20;
            sink.emit_device_drafts(BundleDeviceDraftsV1 {
                struct_size: size_of::<BundleDeviceDraftsV1>() as u32,
                flags: 0,
                cycle_epoch: metadata.cycle_epoch,
                request_id: item.request_id,
                slot_generation: item.slot_generation,
                requested_tokens: k,
                computed_tokens: 5,
                token_stride_bytes: 4,
                proposals: BundleReadyDeviceViewV1 {
                    context: self.ranks[0].0,
                    rank: 0,
                    reserved: 0,
                    buffer: view(slice(device.ranks[0].draft_ids, offset, 20)?),
                    ready,
                },
                confidence: view(slice(device.ranks[0].confidence, offset, 20)?),
            })
            .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    pub(crate) fn submit_dspark_results(
        &self,
        sink: &mut ExecuteResultSinkV1,
        views: &[BundleLogitsViewV1],
    ) -> Result<(), String> {
        let device = self.dspark.as_ref().ok_or("DSpark is absent")?;
        let result = &device.results;
        let metadata = result
            .metadata
            .as_ref()
            .ok_or("DSpark result metadata absent")?;
        let memory = result.memory.ok_or("DSpark result storage absent")?;
        let owners = views
            .iter()
            .map(|view| view.rank as usize)
            .collect::<std::collections::BTreeSet<_>>();
        for &rank in &owners {
            self.services
                .record_event(result.target_ready[rank], self.ranks[rank].2)
                .map_err(|e| e.to_string())?;
        }
        for &logits in views {
            let rank = logits.rank as usize;
            let output = slice(
                result.samples[rank],
                u64::from(logits.first_item_index) * 256,
                u64::from(logits.row_count) * 256,
            )?;
            sink.sample_device_logits(BundleDeviceSamplingRequestV1 {
                struct_size: size_of::<BundleDeviceSamplingRequestV1>() as u32,
                reserved: 0,
                logits: BundleReadyLogitsViewV1 {
                    struct_size: size_of::<BundleReadyLogitsViewV1>() as u32,
                    flags: 0,
                    cycle_epoch: metadata.cycle_epoch,
                    logits,
                    ready: result.target_ready[rank],
                },
                output: view(output),
                done: result.sampled[rank],
            })
            .map_err(|e| e.to_string())?;
        }
        let stream = self.ranks[0].2;
        for rank in owners {
            self.services
                .wait_event(stream, result.sampled[rank])
                .map_err(|e| e.to_string())?;
        }
        let mut rows = 0;
        for logits in views {
            if logits.first_item_index != rows {
                return Err("DSpark sampler rows are not contiguous".into());
            }
            rows += logits.row_count;
            let offset = u64::from(logits.first_item_index) * 256;
            let bytes = u64::from(logits.row_count) * 256;
            let destination = slice(memory.packet, SAMPLE_OFFSET + offset, bytes)?;
            let source = slice(result.samples[logits.rank as usize], offset, bytes)?;
            if logits.rank == 0 {
                self.services
                    .copy_device_to_device(stream, destination, source)
            } else {
                self.services.copy_peer_to_peer(
                    stream,
                    self.ranks[0].0,
                    destination,
                    self.ranks[logits.rank as usize].0,
                    source,
                )
            }
            .map_err(|e| e.to_string())?;
        }
        let expected: u32 = metadata
            .items
            .iter()
            .map(|item| item.acceptance.draft_tokens + 1)
            .sum();
        if rows != expected {
            return Err("DSpark sampler rows differ from acceptance metadata".into());
        }
        for lowered in result.launch(device.ranks[0].draft_ids, metadata.items.len() as u32)? {
            let launch = lowered.launch;
            bundle_runtime::dispatch::launch(
                &self.services,
                stream,
                &self.symbols[0],
                &launch.symbol,
                launch.plan,
                &launch.arguments,
                &launch.bindings,
            )
            .map_err(|e| format!("DSpark acceptance launch: {e:?}"))?;
        }
        let ready = result.ready.ok_or("DSpark result event absent")?;
        self.services
            .record_event(ready, stream)
            .map_err(|e| e.to_string())?;
        sink.emit_device_completion(BundleDeviceCompletionV1 {
            struct_size: size_of::<BundleDeviceCompletionV1>() as u32,
            flags: 0,
            cycle_epoch: metadata.cycle_epoch,
            item_count: metadata.items.len() as u32,
            sample_rows: rows,
            acceptance_offset: 0,
            samples_offset: SAMPLE_OFFSET,
            packet: BundleReadyDeviceViewV1 {
                context: self.ranks[0].0,
                rank: 0,
                reserved: 0,
                buffer: view(slice(
                    memory.packet,
                    0,
                    SAMPLE_OFFSET + u64::from(rows) * 256,
                )?),
                ready,
            },
        })
        .map_err(|e| e.to_string())
    }
}
