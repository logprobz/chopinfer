//! Device target tokens and overlapped mapped rows, behind the lifecycle adapter.
mod graphs;
use super::Instance;
use crate::dspark::cycle::Pending;
use crate::dspark::device::{DECODE_BASE, slice};
use crate::dspark::execution::{DraftInput, Input};
use crate::dspark::staging::DeviceTarget;
use crate::dspark::state::BLOCK_ROWS;
use bundle_runtime::graphs::GraphCache;
use bundle_runtime::inputs::{PackedStep, PhaseInputs};
use bundle_runtime::resources::Held;
use bundle_runtime::upload::{DeviceInputs, InputLayout};
use runtime_bundle_abi::{
    BundleAbiError, BundleStatusV1, ExecuteResultSinkV1, MappedGatherPlaneV1,
};

pub(crate) struct TargetContext<'a> {
    pub inputs: &'a [Input],
    pub drafts: &'a [DraftInput],
    pub history: &'a [i64],
    pub text: &'a [u8],
}

impl Instance {
    pub(crate) fn prepare_dspark_cycle(&mut self) -> Result<(), BundleAbiError> {
        let Some(device) = &mut self.dspark else {
            return Ok(());
        };
        let memory = device
            .cycle
            .memory
            .ok_or(BundleAbiError::InvalidRange("DSpark hash storage"))?;
        let image = self
            .image
            .ok_or(BundleAbiError::InvalidRange("DSpark mapped image"))?;
        let tables = self
            .engram
            .as_ref()
            .and_then(|e| e.tables())
            .ok_or(BundleAbiError::InvalidRange("DSpark mapped tables"))?;
        let shape = device.cycle.hash.shape;
        let columns = shape.columns();
        let mut planes = Vec::new();
        for (rank, buffers) in self.buffers.iter().enumerate() {
            let destinations = device
                .engram_destinations(&buffers.scratch)
                .map_err(|_| BundleAbiError::InvalidRange("DSpark mapped destinations"))?;
            for (layer, (table, dest)) in tables.iter().zip(destinations).enumerate() {
                for (plane, span) in [(&table.codes, dest.codes), (&table.scales, dest.scales)] {
                    let destination = slice(
                        span,
                        u64::from(DECODE_BASE) * columns * plane.row_bytes,
                        u64::from(shape.stride) * columns * plane.row_bytes,
                    )
                    .map_err(|_| {
                        BundleAbiError::InvalidRange("DSpark mapped destination extent")
                    })?;
                    planes.push(MappedGatherPlaneV1 {
                        image,
                        rank: rank as u32,
                        view_id: plane.view.clone(),
                        row_bytes: plane.row_bytes,
                        index_offset: layer as u64 * u64::from(shape.stride) * columns * 4,
                        stream: self.ranks[rank].2,
                        destination,
                    });
                }
            }
        }
        // All drafter ranks must finish before any target rank reuses shared
        // scratch or collective operands. These events never wait on the host.
        for &(context, _, _) in &self.ranks {
            let event = self.services.create_event(context)?;
            self.held.push(Held::Event(event));
            device.cycle.draft_done.push(event);
        }
        let joined = self.services.create_event(self.ranks[0].0)?;
        self.held.push(Held::Event(joined));
        device.cycle.draft_joined = Some(joined);
        device.cycle.tail = Some(GraphCache::sharing(&self.graphs));
        device.cycle.gather = Some(self.services.create_mapped_gather(
            self.ranks[0].2,
            memory.ids,
            &planes,
        )?);
        Ok(())
    }

    pub(crate) fn device_dspark_target(
        &mut self,
        request: &[u8],
        sink: &mut ExecuteResultSinkV1,
        step: PackedStep,
        target: TargetContext<'_>,
    ) -> Result<DeviceTarget, String> {
        let TargetContext {
            inputs,
            drafts,
            history,
            text,
        } = target;
        if !step.prefill.request_ids.is_empty() || inputs.len() != step.decode.request_ids.len() {
            return Err("DSpark device target requires one decode phase".into());
        }
        for (item, input) in inputs.iter().enumerate() {
            if input.rows.request != step.decode.request_ids[item]
                || input.rows.kind.target_rows()? != step.decode.q_len[item] as u32
            {
                return Err("DSpark device target metadata differs from packed inputs".into());
            }
        }
        let device = self.dspark.as_mut().ok_or("DSpark is absent")?;
        if device.cycle.pending.is_some() || device.cycle.gathering {
            return Err("DSpark device cycle is already pending".into());
        }
        let joined = device
            .cycle
            .draft_joined
            .ok_or("DSpark draft join is absent")?;
        for (&event, &(_, _, stream)) in device.cycle.draft_done.iter().zip(&self.ranks) {
            self.services
                .record_event(event, stream)
                .map_err(|e| e.to_string())?;
        }
        for &event in device.cycle.draft_done.iter().skip(1) {
            self.services
                .wait_event(self.ranks[0].2, event)
                .map_err(|e| e.to_string())?;
        }
        self.services
            .record_event(joined, self.ranks[0].2)
            .map_err(|e| e.to_string())?;
        for &(_, _, stream) in self.ranks.iter().skip(1) {
            self.services
                .wait_event(stream, joined)
                .map_err(|e| e.to_string())?;
        }
        device.cycle.pending = Some(Pending {
            inputs: inputs.to_vec(),
            drafts: drafts.to_vec(),
            history: history.to_vec(),
            text: text.to_vec(),
            result: None,
        });
        let status = self.execute_packed(request, sink, step);
        let pending = self
            .dspark
            .as_mut()
            .expect("drafter retained")
            .cycle
            .pending
            .take()
            .ok_or("DSpark device cycle lost its result")?;
        if status != BundleStatusV1::Success {
            return Err(format!("DSpark device target execute: {status:?}"));
        }
        pending
            .result
            .ok_or_else(|| "DSpark target emitted no device diagnostics".into())
    }

    pub(crate) fn upload_dspark_cycle(
        &mut self,
        arena: usize,
        layout: InputLayout,
        inputs: &PhaseInputs,
    ) -> Result<(), String> {
        let Some(device) = &self.dspark else {
            return Ok(());
        };
        let Some(pending) = &device.cycle.pending else {
            return Ok(());
        };
        if !inputs.decode || pending.inputs.len() != inputs.request_ids.len() {
            return Err("DSpark device target phase differs".into());
        }
        let rows = inputs.live_tokens();
        if pending.text.len() != rows as usize {
            return Err("DSpark device text rows differ".into());
        }
        for (rank, stored) in device.ranks.iter().enumerate() {
            let target = DeviceInputs::within(self.arenas[arena][rank], &layout);
            for (item, &request) in inputs.request_ids.iter().enumerate() {
                let Some((index, draft)) = pending
                    .drafts
                    .iter()
                    .enumerate()
                    .find(|(_, draft)| draft.block.request == request)
                else {
                    continue;
                };
                if draft.block.emit_rows + 1 != inputs.q_len[item] as u32 {
                    return Err("DSpark emitted budget differs from target".into());
                }
                let bytes = u64::from(draft.block.emit_rows) * 4;
                let source = slice(
                    stored.draft_ids,
                    index as u64 * u64::from(BLOCK_ROWS) * 4,
                    bytes,
                )?;
                let destination = slice(
                    target.token_ids,
                    (inputs.q_start[item] as u64 + 1) * 4,
                    bytes,
                )?;
                self.services
                    .copy_device_to_device(self.ranks[rank].2, destination, source)
                    .map_err(|e| e.to_string())?;
            }
        }
        let memory = device.cycle.memory.ok_or("DSpark hash memory is absent")?;
        let stream = self.ranks[0].2;
        let history: Vec<_> = pending
            .history
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        self.services
            .copy_host_to_device(
                stream,
                slice(memory.history, 0, history.len() as u64)?,
                &history,
            )
            .map_err(|e| e.to_string())?;
        self.services
            .copy_host_to_device(
                stream,
                slice(memory.text, 0, pending.text.len() as u64)?,
                &pending.text,
            )
            .map_err(|e| e.to_string())?;
        let target = DeviceInputs::within(self.arenas[arena][0], &layout);
        let status = crate::buffers::rows(self.buffers[0].scratch.row_status, rows);
        for lowered in device.cycle.hash.launch(memory, target, status, rows)? {
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
            .map_err(|e| format!("Engram device hash launch: {e:?}"))?;
        }
        let gather = device
            .cycle
            .gather
            .ok_or("DSpark mapped gather is absent")?;
        self.services
            .enqueue_mapped_gather(gather, u64::from(rows) * device.cycle.hash.shape.columns())
            .map_err(|e| e.to_string())?;
        self.dspark
            .as_mut()
            .expect("drafter retained")
            .cycle
            .gathering = true;
        Ok(())
    }

    pub(crate) fn wait_dspark_rows(&self) -> Result<(), BundleAbiError> {
        let Some(device) = &self.dspark else {
            return Ok(());
        };
        if device.cycle.gathering {
            let gather = device
                .cycle
                .gather
                .ok_or(BundleAbiError::InvalidRange("DSpark mapped gather"))?;
            for &(_, _, stream) in &self.ranks {
                self.services.wait_mapped_gather(gather, stream)?;
            }
        }
        Ok(())
    }

    pub(crate) fn finish_dspark_cycle(
        &mut self,
        sink: &mut ExecuteResultSinkV1,
    ) -> Result<(), String> {
        let Some(device) = &self.dspark else {
            return Ok(());
        };
        let Some(pending) = &device.cycle.pending else {
            return Ok(());
        };
        let gather = device
            .cycle
            .gather
            .ok_or("DSpark mapped gather is absent")?;
        let stats = self
            .services
            .complete_mapped_gather(gather)
            .map_err(|e| e.to_string())?
            .ok_or("DSpark mapped upload is incomplete at target boundary")?;
        let packet = device
            .cycle
            .diagnostics
            .ok_or("DSpark diagnostic packet is absent")?;
        let bytes = pending.drafts.len() as u64 * u64::from(BLOCK_ROWS) * 4;
        let stream = self.ranks[0].2;
        for (offset, source) in [
            (0, device.ranks[0].draft_ids),
            (bytes, device.ranks[0].confidence),
        ] {
            self.services
                .copy_device_to_device(
                    stream,
                    slice(packet, offset, bytes)?,
                    slice(source, 0, bytes)?,
                )
                .map_err(|e| e.to_string())?;
        }
        let mut output = vec![0; bytes as usize * 2];
        self.services
            .copy_device_to_host(stream, &mut output, slice(packet, 0, bytes * 2)?)
            .map_err(|e| e.to_string())?;
        let words = |data: &[u8]| {
            data.chunks_exact(BLOCK_ROWS as usize * 4)
                .map(|row| {
                    std::array::from_fn(|column| {
                        u32::from_le_bytes(
                            row[column * 4..column * 4 + 4]
                                .try_into()
                                .expect("U32 diagnostic word"),
                        )
                    })
                })
                .collect::<Vec<[u32; BLOCK_ROWS as usize]>>()
        };
        let ids = words(&output[..bytes as usize]);
        let confidence_bits = words(&output[bytes as usize..]);
        if ids
            .iter()
            .flatten()
            .any(|&token| token >= device.shape.geometry.shape.vocab)
        {
            return Err("DSpark device draft ID is outside vocabulary".into());
        }
        for (draft, tokens) in pending.drafts.iter().zip(&ids) {
            sink.emit_draft_tokens(
                draft.block.request,
                &tokens[..draft.block.emit_rows as usize],
            )
            .map_err(|e| e.to_string())?;
        }
        // The existing routine schema uses bounded best-effort logging. It
        // reports exposed time without making logging a coordinator wait.
        let message = format!(
            concat!(
                "{{\"schema\":\"chopinfer.deepseek41.step-shape.v1\",",
                "\"phase\":\"dspark-engram\",\"items\":{},\"rows\":{},",
                "\"host_ns\":{},\"exposed_ns\":{}}}"
            ),
            pending.inputs.len(),
            pending.text.len(),
            stats.host_nanoseconds,
            stats.exposed_nanoseconds
        );
        let _ = self.services.log(message.as_bytes());
        let cycle = &mut self.dspark.as_mut().expect("drafter retained").cycle;
        cycle.gathering = false;
        cycle.pending.as_mut().expect("pending cycle").result = Some(DeviceTarget {
            ids,
            confidence_bits,
        });
        Ok(())
    }
}
