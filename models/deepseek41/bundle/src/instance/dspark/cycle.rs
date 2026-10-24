//! Device target tokens and overlapped mapped rows, behind the lifecycle adapter.
mod graphs;
use super::Instance;
use crate::dspark::cycle::{Deferred, Early, LayerGather, Pending};
use crate::dspark::device::{DECODE_BASE, slice};
use crate::dspark::execution::{DraftInput, Input};
use crate::dspark::staging::DeviceTarget;
use crate::dspark::state::BLOCK_ROWS;
use crate::engram_rows::Hashed;
use crate::instance::ar_engram;
use bundle_runtime::graphs::GraphCache;
use bundle_runtime::inputs::{PackedStep, PhaseInputs};
use bundle_runtime::resources::{Held, upload_bytes};
use bundle_runtime::upload::{DeviceInputs, InputLayout};
use model_deepseek41_contract::serving::DECODE_BATCH;
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
        device.cycle.undrafted = ar_engram::enabled().map_err(BundleAbiError::Status)?;
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
        let mut layer_planes = vec![Vec::new(); tables.len()];
        let mut early_planes = vec![Vec::new(); tables.len()];
        for (rank, buffers) in self.buffers.iter().enumerate() {
            // Each layer's codes, then its scales, one row per decode item.
            if let Some(early) = device
                .ranks
                .get(rank)
                .and_then(|stored| stored.engram_early)
            {
                let mut at = 0;
                for (layer, table) in tables.iter().enumerate() {
                    for plane in [&table.codes, &table.scales] {
                        let bytes = u64::from(DECODE_BATCH) * columns * plane.row_bytes;
                        let destination = slice(early, at, bytes).map_err(|_| {
                            BundleAbiError::InvalidRange("DSpark early Engram destination")
                        })?;
                        at += bytes;
                        early_planes[layer].push(MappedGatherPlaneV1 {
                            image,
                            rank: rank as u32,
                            view_id: plane.view.clone(),
                            row_bytes: plane.row_bytes,
                            index_offset: 0,
                            stream: self.ranks[rank].2,
                            destination,
                        });
                    }
                }
            }
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
                    layer_planes[layer].push(MappedGatherPlaneV1 {
                        image,
                        rank: rank as u32,
                        view_id: plane.view.clone(),
                        row_bytes: plane.row_bytes,
                        index_offset: 0,
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
            let event = self.services.create_event(context)?;
            self.held.push(Held::Event(event));
            device.cycle.boundary_ready.push(event);
        }
        let joined = self.services.create_event(self.ranks[0].0)?;
        self.held.push(Held::Event(joined));
        device.cycle.draft_joined = Some(joined);
        if device.prefetch.enabled {
            for &(context, _, _) in &self.ranks {
                let event = self.services.create_event(context)?;
                self.held.push(Held::Event(event));
                device.prefetch.done.push(event);
            }
            let event = self.services.create_event(self.ranks[0].0)?;
            self.held.push(Held::Event(event));
            device.prefetch.joined = Some(event);
        }
        for &(context, _, _) in &self.ranks {
            for events in [
                &mut device.results.target_ready,
                &mut device.results.sampled,
            ] {
                let event = self.services.create_event(context)?;
                self.held.push(Held::Event(event));
                events.push(event);
            }
        }
        let ready = self.services.create_event(self.ranks[0].0)?;
        self.held.push(Held::Event(ready));
        device.results.ready = Some(ready);
        let bytes = u64::from(shape.stride) * columns * 4;
        for (index, planes) in layer_planes.into_iter().enumerate() {
            let indices = slice(memory.ids, index as u64 * bytes, bytes)
                .map_err(|_| BundleAbiError::InvalidRange("DSpark layer hash extent"))?;
            let gather = self
                .services
                .create_mapped_gather(self.ranks[0].2, indices, &planes)?;
            let mut graphs = GraphCache::sharing(&self.graphs);
            graphs.label("target", None, index as u32 + 1);
            device.cycle.layers.push(LayerGather {
                layer: device.shape.geometry.shape.engram.layers[index],
                gather,
                indices,
                destinations: planes.iter().map(|plane| plane.destination).collect(),
                graphs,
            });
        }
        if let (Some(stream), Some(ids)) = (device.cycle.early_stream, device.cycle.early_ids) {
            if early_planes
                .iter()
                .any(|planes| planes.len() != 2 * self.ranks.len())
            {
                return Err(BundleAbiError::InvalidRange("DSpark early Engram ranks"));
            }
            let mut gathers = Vec::with_capacity(early_planes.len());
            for (index, planes) in early_planes.into_iter().enumerate() {
                let indices = slice(ids, index as u64 * bytes, bytes)
                    .map_err(|_| BundleAbiError::InvalidRange("DSpark early hash extent"))?;
                gathers.push(
                    self.services
                        .create_mapped_gather(stream, indices, &planes)?,
                );
            }
            device.cycle.early = Some(Early {
                ids,
                gathers,
                rows: None,
            });
        }
        let early = device.cycle.early.is_some();
        if device.cycle.undrafted {
            let layers = device.cycle.layers.len();
            self.log(&format!(
                "CHOPINFER_DEEPSEEK41_AR_ENGRAM: DSpark decode steps without drafts join {layers} Engram layers at their consumers"
            ));
        }
        if early {
            self.log(
                "CHOPINFER_DSPARK_ENGRAM_EARLY: drafted steps gather each item's first Engram rows before the drafter",
            );
        }
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
        let batch = runtime_bundle_sdk::BatchCodecV1::decode(request)
            .map_err(|e| format!("DSpark batch: {e:?}"))?;
        let allocation = crate::dspark::allocation::Step::of(&batch, &step.decode)?;
        let device = self.dspark.as_mut().ok_or("DSpark is absent")?;
        if device.cycle.pending.is_some() || device.cycle.gathering {
            return Err("DSpark device cycle is already pending".into());
        }
        if let Some(step) = &allocation {
            let op = device
                .allocation
                .as_ref()
                .ok_or("DSpark verify allocation is not prepared")?
                .op;
            if step.reserved_rows > op.reserved_rows || step.width > op.width {
                return Err("DSpark allocated step exceeds the prepared rows".into());
            }
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
            allocation,
            packed_diagnostics: Default::default(),
            feedback: Default::default(),
        });
        self.emit_dspark_device_drafts(sink)?;
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
            return self.upload_deferred_engram(inputs);
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
        // An allocated step hashes its computed rows, after the allocation
        // compacts them.
        let rows = pending.allocation.as_ref().map_or(rows, |step| step.width);
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
        if let Some(step) = &pending.allocation {
            let targets: Vec<_> = (0..self.ranks.len())
                .map(|rank| DeviceInputs::within(self.arenas[arena][rank], &layout))
                .collect();
            self.allocate_dspark_step(step, &targets, memory.text)?;
        }
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
        for layer in &device.cycle.layers {
            self.services
                .enqueue_mapped_gather(
                    layer.gather,
                    u64::from(rows) * device.cycle.hash.shape.columns(),
                )
                .map_err(|e| e.to_string())?;
        }
        self.dspark
            .as_mut()
            .expect("drafter retained")
            .cycle
            .gathering = true;
        Ok(())
    }

    /// Starts a drafted step's early gathers (`dspark::cycle::Early`) of
    /// `hashed`, each item's first target row, at the start of its execute.
    /// The ids go up on rank 0's side stream, so the reads start at once,
    /// while this step's drafter or the successor drafter queued by the
    /// previous step runs. Only the copies into scratch follow each rank's
    /// queued work.
    pub(crate) fn prefetch_dspark_engram(&mut self, hashed: &Hashed) -> Result<(), String> {
        let services = &self.services;
        let Some(cycle) = self.dspark.as_mut().map(|device| &mut device.cycle) else {
            return Ok(());
        };
        let Some(stream) = cycle.early_stream else {
            return Ok(());
        };
        let (stride, columns) = (
            cycle.hash.shape.stride as usize,
            cycle.hash.shape.columns() as usize,
        );
        let busy = cycle.gathering || cycle.pending.is_some();
        let Some(early) = &mut cycle.early else {
            return Ok(());
        };
        if busy || early.rows.is_some() {
            return Err("DSpark early Engram rows are already pending".into());
        }
        if hashed.rows() == 0 || hashed.rows() > DECODE_BATCH as usize {
            return Ok(());
        }
        let ids = ar_engram::layer_major(hashed, early.gathers.len(), columns, stride)
            .map_err(|error| error.to_string())?;
        upload_bytes(services, stream, early.ids, &ids).map_err(|e| e.to_string())?;
        let rows = (hashed.rows() * columns) as u64;
        for &gather in &early.gathers {
            services
                .enqueue_mapped_gather(gather, rows)
                .map_err(|e| e.to_string())?;
        }
        early.rows = Some(hashed.rows() as u32);
        Ok(())
    }

    /// Takes an undrafted decode step's host-hashed ids for its target
    /// launch, which uploads them where the device hash writes a drafted
    /// step's and gathers each layer's rows through the same prepared
    /// gathers (`upload_deferred_engram`). The ids, the rows' destinations
    /// and the target's launches are the synchronous gather's.
    pub(crate) fn defer_dspark_engram(&mut self, hashed: &Hashed) -> Result<bool, String> {
        let Some(device) = self.dspark.as_mut() else {
            return Ok(false);
        };
        let cycle = &mut device.cycle;
        let stride = cycle.hash.shape.stride as usize;
        if !cycle.undrafted || cycle.layers.is_empty() || hashed.rows() > stride {
            return Ok(false);
        }
        if cycle.pending.is_some() || cycle.gathering || cycle.deferred.is_some() {
            return Err("DSpark device cycle is already pending".into());
        }
        let columns = cycle.hash.shape.columns() as usize;
        let ids = ar_engram::layer_major(hashed, cycle.layers.len(), columns, stride)
            .map_err(|error| error.to_string())?;
        cycle.deferred = Some(Deferred {
            ids,
            rows: hashed.rows() as u32,
        });
        Ok(true)
    }

    /// Uploads a deferred step's ids on rank 0's stream, the gathers'
    /// producer, and enqueues every layer's gather of its rows.
    fn upload_deferred_engram(&mut self, inputs: &PhaseInputs) -> Result<(), String> {
        let services = &self.services;
        let stream = self.ranks[0].2;
        let Some(cycle) = self.dspark.as_mut().map(|device| &mut device.cycle) else {
            return Ok(());
        };
        let Some(deferred) = &cycle.deferred else {
            return Ok(());
        };
        if cycle.gathering || !inputs.decode || inputs.live_tokens() != deferred.rows {
            return Err("DSpark deferred Engram rows differ from the target phase".into());
        }
        let memory = cycle.memory.ok_or("DSpark hash memory is absent")?;
        upload_bytes(services, stream, memory.ids, &deferred.ids).map_err(|e| e.to_string())?;
        let rows = u64::from(deferred.rows) * cycle.hash.shape.columns();
        for layer in &cycle.layers {
            services
                .enqueue_mapped_gather(layer.gather, rows)
                .map_err(|e| e.to_string())?;
        }
        cycle.gathering = true;
        Ok(())
    }

    /// Completes a deferred step's gathers once the phase's streams
    /// finished: a copy still running refuses the step before any result
    /// is published.
    fn finish_deferred_engram(&mut self) -> Result<(), String> {
        let Some(device) = self.dspark.as_mut() else {
            return Ok(());
        };
        let sequence = device.diagnostic.step_id;
        let cycle = &mut device.cycle;
        let Some(deferred) = cycle.deferred.take() else {
            return Ok(());
        };
        if !cycle.gathering {
            return Err("DSpark deferred Engram rows were never gathered".into());
        }
        let mut layers = Vec::new();
        for layer in &cycle.layers {
            let stats = self
                .services
                .complete_mapped_gather(layer.gather)
                .map_err(|e| e.to_string())?
                .ok_or("DSpark mapped layer upload is incomplete at target boundary")?;
            layers.push(ar_engram::layer_trace(layer.layer, &stats));
        }
        cycle.gathering = false;
        if ar_engram::traced() {
            let message = ar_engram::step_trace(sequence, deferred.rows, &layers);
            let _ = self.services.log(message.as_bytes());
        }
        Ok(())
    }

    pub(crate) fn wait_dspark_rows(&self, layer: usize) -> Result<(), BundleAbiError> {
        let Some(device) = &self.dspark else {
            return Ok(());
        };
        if device.cycle.gathering {
            let gather = device
                .cycle
                .layers
                .get(layer)
                .ok_or(BundleAbiError::InvalidRange("DSpark mapped layer"))?
                .gather;
            // The early copy finished while the drafter ran; joining it
            // here keeps the boundary's completion check for both.
            let early = device
                .cycle
                .early
                .as_ref()
                .filter(|early| early.rows.is_some())
                .map(|early| early.gathers.get(layer).copied())
                .map(|gather| gather.ok_or(BundleAbiError::InvalidRange("DSpark early layer")))
                .transpose()?;
            for &(_, _, stream) in &self.ranks {
                if let Some(early) = early {
                    self.services.wait_mapped_gather(early, stream)?;
                }
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
            return self.finish_deferred_engram();
        };
        // Add the host callback scopes and each consumer's exposed wait.
        // Per-layer values retain attribution when the callbacks overlap.
        let mut host_ns = 0_u64;
        let mut exposed_ns = 0_u64;
        let mut layers = Vec::new();
        for layer in &device.cycle.layers {
            let stats = self
                .services
                .complete_mapped_gather(layer.gather)
                .map_err(|e| e.to_string())?
                .ok_or("DSpark mapped layer upload is incomplete at target boundary")?;
            host_ns += stats.host_nanoseconds;
            exposed_ns += stats.exposed_nanoseconds;
            layers.push(format!(
                "{{\"layer\":{},\"host_ns\":{},\"exposed_ns\":{}}}",
                layer.layer, stats.host_nanoseconds, stats.exposed_nanoseconds,
            ));
        }
        let mut early_layers = Vec::new();
        if let Some(early) = &device.cycle.early
            && let Some(rows) = early.rows
        {
            for (layer, &gather) in device.cycle.layers.iter().zip(&early.gathers) {
                let stats = self
                    .services
                    .complete_mapped_gather(gather)
                    .map_err(|e| e.to_string())?
                    .ok_or("DSpark early Engram upload is incomplete at target boundary")?;
                early_layers.push(format!(
                    "{{\"layer\":{},\"rows\":{rows},\"host_ns\":{},\"exposed_ns\":{}}}",
                    layer.layer, stats.host_nanoseconds, stats.exposed_nanoseconds,
                ));
            }
        }
        let packet = device
            .cycle
            .diagnostics
            .ok_or("DSpark diagnostic packet is absent")?;
        let bytes = pending.drafts.len() as u64 * u64::from(BLOCK_ROWS) * 4;
        let stream = self.ranks[0].2;
        let (output, boundary_reads) =
            if let Some(output) = pending.packed_diagnostics.borrow_mut().take() {
                (output, 1)
            } else {
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
                (output, self.ranks.len() + 1)
            };
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
        let feedback = pending.feedback.borrow_mut().take();
        if ids
            .iter()
            .flatten()
            .any(|&token| token >= device.shape.geometry.shape.vocab)
        {
            return Err("DSpark device draft ID is outside vocabulary".into());
        }
        for (draft, tokens) in pending
            .drafts
            .iter()
            .zip(&ids)
            .filter(|_| device.results.metadata.is_none())
        {
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
                "\"host_ns\":{},\"exposed_ns\":{},\"device_results\":{},\"bundle_readbacks\":{},\"layers\":[{}],{}\"explicit_bundle_syncs\":0}}"
            ),
            pending.inputs.len(),
            pending.text.len(),
            host_ns,
            exposed_ns,
            device.results.metadata.is_some(),
            boundary_reads + device.cycle.layers.len(),
            layers.join(","),
            if early_layers.is_empty() {
                String::new()
            } else {
                format!("\"early\":[{}],", early_layers.join(","))
            },
        );
        let _ = self.services.log(message.as_bytes());
        let device = self.dspark.as_mut().expect("drafter retained");
        device.prefetch.feedback = feedback;
        let cycle = &mut device.cycle;
        cycle.gathering = false;
        if let Some(early) = &mut cycle.early {
            early.rows = None;
        }
        cycle.pending.as_mut().expect("pending cycle").result = Some(DeviceTarget {
            ids,
            confidence_bits,
        });
        Ok(())
    }
}
