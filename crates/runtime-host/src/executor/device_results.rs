//! Opt-in device output collection. Device counts are validated against the
//! engine's metadata, and all sample suffix rows still reach engine validation.
mod activation;
use super::*;
use activation::ActivationReceipt;
use runtime_bundle_abi::device_results::*;
use runtime_engine::{AcceptedPrefix, AsyncInputDependency};

pub(super) struct DeviceResults {
    metadata: DeviceCycleMetadataV1,
    dependency: Option<AsyncInputDependency>,
    // Retain owners before submission, including partial or failed enqueue.
    contexts: BTreeMap<u64, (runtime_bundle_abi::BundleContextHandleV1, u32)>,
    drafts: BTreeMap<RequestId, BundleDeviceDraftsV1>,
    leases: Vec<Box<dyn std::any::Any>>,
    expanded: Vec<ModelBatchItem>,
    groups: Vec<(std::ops::Range<usize>, SamplingDispatch)>,
    external_rows: BTreeSet<usize>,
    ready: Vec<(PendingReadback, Vec<ModelBatchItem>)>,
    completion: Option<(BundleDeviceCompletionV1, PendingReadback)>,
}

struct PendingReadback {
    sampler: Rc<dyn DeviceLogitsSamplerV1>,
    view: BundleReadyDeviceViewV1,
    ticket: Option<u64>,
}

impl PendingReadback {
    fn collect(&mut self) -> HostServiceResultV1<Vec<u8>> {
        let bytes = self
            .sampler
            .collect_device_readback(self.view, self.ticket.ok_or(BundleStatusV1::StaleHandle)?)?;
        self.ticket = None;
        Ok(bytes)
    }
}

impl Drop for PendingReadback {
    fn drop(&mut self) {
        if let Some(ticket) = self.ticket {
            // A failed cancellation leaves ownership in the backend until its
            // teardown fence. No output from that execution may be published.
            let _ = self.sampler.cancel_device_readback(self.view, ticket);
        }
    }
}

pub(super) fn cycle_metadata(
    items: &[ModelBatchItem],
    pool: Option<&crate::recurrent::StateSlotPoolV1>,
) -> Option<DeviceCycleMetadataV1> {
    if items.iter().any(|item| {
        !item.prompt_targets().is_empty()
            || item
                .sampling()
                .is_some_and(|work| !work.history_independent())
            || item
                .draft()
                .is_some_and(|draft| draft.source() != DraftSource::Bundle)
    }) {
        return None;
    }
    let mut metadata = DeviceCycleMetadataV1 {
        cycle_epoch: 0,
        items: Vec::new(),
        termination_stop_ids: Vec::new(),
    };
    let mut first_sample_row = 0_u32;
    let mut draft_index = 0_u32;
    for item in items.iter().filter(|item| item.expects_logits()) {
        let cycle = item.output_cycle()?;
        if metadata.cycle_epoch != 0 && metadata.cycle_epoch != cycle.epoch {
            return None;
        }
        metadata.cycle_epoch = cycle.epoch;
        let slot_generation = match pool {
            Some(pool) => pool.lease(item.request_id())?.1,
            None => cycle.epoch,
        };
        let draft_tokens = u32::try_from(item.draft().map_or(0, |draft| draft.tokens())).ok()?;
        let draft_item_index = if draft_tokens == 0 {
            BUNDLE_ACCEPTANCE_NO_DRAFT_V1
        } else {
            let index = draft_index;
            draft_index += 1;
            index
        };
        let input = BundleAcceptanceInputV1 {
            cycle_epoch: cycle.epoch,
            remaining_tokens: u64::try_from(cycle.remaining_tokens).ok()?,
            first_sample_row,
            draft_item_index,
            draft_tokens,
            stop_offset: u32::try_from(metadata.termination_stop_ids.len()).ok()?,
            stop_count: u32::try_from(cycle.termination_stop_ids.len()).ok()?,
            reserved: 0,
        };
        first_sample_row = first_sample_row.checked_add(draft_tokens.checked_add(1)?)?;
        metadata
            .termination_stop_ids
            .extend_from_slice(&cycle.termination_stop_ids);
        metadata.items.push(DeviceCycleItemV1 {
            request_id: item.request_id().get(),
            slot_generation,
            acceptance: input,
        });
    }
    (!metadata.items.is_empty()).then_some(metadata)
}

impl DeviceResults {
    pub(super) fn new(
        metadata: DeviceCycleMetadataV1,
        items: &[ModelBatchItem],
        sampler: &dyn DeviceLogitsSamplerV1,
        limits: ResultSinkLimitsV1,
    ) -> HostServiceResultV1<Self> {
        metadata.validate(limits)?;
        let mut expanded = Vec::new();
        for item in items.iter().filter(|item| item.expects_logits()) {
            for row in 0..item.draft().map_or(1, |draft| draft.tokens() + 1) {
                let work = item
                    .sampling()
                    .ok_or(BundleStatusV1::InvalidArgument)?
                    .for_device_verify_row(row)
                    .map_err(|_| BundleStatusV1::Unsupported)?;
                expanded.push(item.clone().with_sampling(work));
            }
        }
        let groups = gathered_ranges(items)
            .into_iter()
            .map(|range| {
                let rows = expanded
                    .get(range.clone())
                    .ok_or(BundleStatusV1::InvalidArgument)?;
                let dispatch = sampler.sampling_dispatch(&rows.iter().collect::<Vec<_>>())?;
                Ok((range, dispatch))
            })
            .collect::<HostServiceResultV1<Vec<_>>>()?;
        Ok(Self {
            metadata,
            dependency: None,
            contexts: BTreeMap::new(),
            drafts: BTreeMap::new(),
            leases: Vec::new(),
            expanded,
            groups,
            external_rows: BTreeSet::new(),
            ready: Vec::new(),
            completion: None,
        })
    }

    pub(super) fn dependency(&self) -> AsyncInputDependency {
        self.dependency.unwrap_or_default()
    }

    pub(super) fn epoch(&self) -> u64 {
        self.metadata.cycle_epoch
    }

    fn retain_context(&mut self, context: runtime_bundle_abi::BundleContextHandleV1, rank: u32) {
        self.contexts.entry(context.raw).or_insert((context, rank));
    }

    fn require_epoch(&self, epoch: u64) -> HostServiceResultV1<()> {
        if epoch != self.metadata.cycle_epoch {
            return Err(BundleStatusV1::StaleHandle);
        }
        if self.dependency.is_none() || self.completion.is_some() {
            return Err(BundleStatusV1::InvalidState);
        }
        Ok(())
    }

    fn partitions(
        &self,
        raw: runtime_bundle_abi::BundleLogitsViewV1,
    ) -> HostServiceResultV1<Vec<(runtime_bundle_abi::BundleLogitsViewV1, SamplingDispatch)>> {
        let start = raw.first_item_index as usize;
        let end = start
            .checked_add(raw.row_count as usize)
            .ok_or(BundleStatusV1::InvalidArgument)?;
        let rows = self
            .expanded
            .get(start..end)
            .ok_or(BundleStatusV1::InvalidArgument)?;
        if rows.is_empty()
            || rows
                .iter()
                .any(|item| item.draft().is_some() && !self.drafts.contains_key(&item.request_id()))
        {
            return Err(BundleStatusV1::InvalidState);
        }
        let mut parts = Vec::new();
        for (range, dispatch) in &self.groups {
            let first = start.max(range.start);
            let last = end.min(range.end);
            for at in (first..last).step_by(SAMPLER_VIEW_ROWS) {
                let count = (last - at).min(SAMPLER_VIEW_ROWS);
                parts.push((
                    sampling_subview(OwnedLogitsViewV1::new(raw), at - start, count)?.raw(),
                    *dispatch,
                ));
            }
        }
        if parts
            .iter()
            .map(|(part, _)| part.row_count as usize)
            .sum::<usize>()
            != rows.len()
        {
            return Err(BundleStatusV1::InvalidArgument);
        }
        Ok(parts)
    }
}

impl CollectingSinkV1<'_> {
    pub(super) fn device_metadata(&self) -> HostServiceResultV1<DeviceCycleMetadataV1> {
        Ok(self
            .device
            .as_ref()
            .ok_or(BundleStatusV1::Unsupported)?
            .metadata
            .clone())
    }

    pub(super) fn device_dependency(&mut self, dependency: u32) -> HostServiceResultV1<()> {
        let device = self.device.as_mut().ok_or(BundleStatusV1::Unsupported)?;
        if device.dependency.is_some() {
            return Err(BundleStatusV1::InvalidState);
        }
        device.dependency = Some(match dependency {
            DEVICE_RESULTS_HOST_VALUES_V1 => AsyncInputDependency::HostValues,
            DEVICE_RESULTS_DEVICE_FEEDBACK_V1 => AsyncInputDependency::DeviceFeedback,
            _ => return Err(BundleStatusV1::Unsupported),
        });
        Ok(())
    }

    pub(super) fn device_drafts(
        &mut self,
        drafts: BundleDeviceDraftsV1,
    ) -> HostServiceResultV1<()> {
        drafts.validate()?;
        let device = self.device.as_mut().ok_or(BundleStatusV1::Unsupported)?;
        device.require_epoch(drafts.cycle_epoch)?;
        let item = device
            .metadata
            .items
            .iter()
            .find(|item| item.request_id == drafts.request_id)
            .ok_or(BundleStatusV1::InvalidArgument)?;
        if item.slot_generation != drafts.slot_generation {
            return Err(BundleStatusV1::StaleHandle);
        }
        if item.acceptance.draft_tokens != drafts.requested_tokens
            || device
                .drafts
                .contains_key(&RequestId::new(drafts.request_id))
            || self.drafts.contains_key(&RequestId::new(drafts.request_id))
        {
            return Err(BundleStatusV1::InvalidArgument);
        }
        // The bundle owns proposal consumers. Its borrowed spans must remain
        // unchanged through final collection, as the ABI contract requires.
        device.leases.push(
            self.sampler
                .as_ref()
                .ok_or(BundleStatusV1::Unsupported)?
                .retain_device_drafts(drafts)?,
        );
        device
            .drafts
            .insert(RequestId::new(drafts.request_id), drafts);
        Ok(())
    }

    pub(super) fn device_sample(
        &mut self,
        request: BundleDeviceSamplingRequestV1,
    ) -> HostServiceResultV1<()> {
        request
            .validate()
            .map_err(|_| BundleStatusV1::InvalidArgument)?;
        let sampler = self.sampler.as_ref().ok_or(BundleStatusV1::Unsupported)?;
        let device = self.device.as_mut().ok_or(BundleStatusV1::Unsupported)?;
        device.require_epoch(request.logits.cycle_epoch)?;
        let raw = request.logits.logits;
        let start = raw.first_item_index as usize;
        let end = start + raw.row_count as usize;
        if u64::from(raw.live_width) != self.limits.maximum_logits_per_row
            || (start..end).any(|index| self.sampled_indices.contains(&index))
            || !self.rows.is_empty()
        {
            return Err(BundleStatusV1::InvalidArgument);
        }
        for (part, dispatch) in device.partitions(raw)? {
            let at = part.first_item_index as usize;
            let count = part.row_count as usize;
            let mut submitted = request;
            submitted.logits.logits = part;
            submitted.output.offset += (at - start) as u64 * DEVICE_SAMPLE_RECORD_BYTES_V1;
            submitted.output.length = count as u64 * DEVICE_SAMPLE_RECORD_BYTES_V1;
            device.retain_context(part.context, part.rank);
            sampler.sample_device_logits(
                submitted,
                &device.expanded[at..at + count].iter().collect::<Vec<_>>(),
                dispatch,
            )?;
        }
        device.external_rows.extend(start..end);
        self.sampled_indices.extend(start..end);
        Ok(())
    }

    pub(super) fn device_ready(
        &mut self,
        view: BundleReadyLogitsViewV1,
    ) -> HostServiceResultV1<()> {
        view.validate()
            .map_err(|_| BundleStatusV1::InvalidArgument)?;
        let sampler = self.sampler.as_ref().ok_or(BundleStatusV1::Unsupported)?;
        let device = self.device.as_mut().ok_or(BundleStatusV1::Unsupported)?;
        device.require_epoch(view.cycle_epoch)?;
        let raw = view.logits;
        let start = raw.first_item_index as usize;
        let end = start + raw.row_count as usize;
        if u64::from(raw.live_width) != self.limits.maximum_logits_per_row
            || (start..end).any(|index| self.sampled_indices.contains(&index))
            || !self.rows.is_empty()
            || !device.external_rows.is_empty()
        {
            return Err(BundleStatusV1::InvalidArgument);
        }
        for (part, dispatch) in device.partitions(raw)? {
            let at = part.first_item_index as usize;
            let items = device.expanded[at..at + part.row_count as usize].to_vec();
            if items.iter().any(|item| item.draft().is_some()) {
                return Err(BundleStatusV1::Unsupported);
            }
            device.retain_context(raw.context, raw.rank);
            let ticket = sampler.issue_ready_logits(
                BundleReadyLogitsViewV1 {
                    logits: part,
                    ..view
                },
                &items.iter().collect::<Vec<_>>(),
                dispatch,
            )?;
            device.ready.push((
                PendingReadback {
                    sampler: sampler.clone(),
                    view: BundleReadyDeviceViewV1 {
                        context: raw.context,
                        rank: raw.rank,
                        reserved: 0,
                        buffer: raw.buffer,
                        ready: view.ready,
                    },
                    ticket: Some(ticket),
                },
                items,
            ));
        }
        self.sampled_indices.extend(start..end);
        Ok(())
    }

    pub(super) fn device_completion(
        &mut self,
        completion: BundleDeviceCompletionV1,
    ) -> HostServiceResultV1<()> {
        completion.validate()?;
        let sampler = self.sampler.as_ref().ok_or(BundleStatusV1::Unsupported)?;
        let device = self.device.as_mut().ok_or(BundleStatusV1::Unsupported)?;
        device.require_epoch(completion.cycle_epoch)?;
        if completion.item_count as usize != device.metadata.items.len()
            || completion.sample_rows as usize != device.expanded.len()
            || device.external_rows.len() != device.expanded.len()
            || !device.ready.is_empty()
            || !self.pending.is_empty()
            || !self.rows.is_empty()
        {
            return Err(BundleStatusV1::InvalidArgument);
        }
        device.retain_context(completion.packet.context, completion.packet.rank);
        let ticket = sampler.enqueue_device_readback(completion.packet)?;
        device.completion = Some((
            completion,
            PendingReadback {
                sampler: sampler.clone(),
                view: completion.packet,
                ticket: Some(ticket),
            },
        ));
        Ok(())
    }

    pub(super) fn discard_device(&mut self) -> HostServiceResultV1<()> {
        let Some(device) = &mut self.device else {
            return Ok(());
        };
        let sampler = self.sampler.as_ref().ok_or(BundleStatusV1::Unsupported)?;
        let mut failure = None;
        for &(context, rank) in device.contexts.values() {
            if let Err(error) = sampler.drain_device_results(context, rank) {
                failure.get_or_insert(error);
            }
        }
        if let Some(error) = failure {
            return Err(error);
        }
        // The backend drained all tickets, including failed enqueues that
        // never returned one. Do not cancel these stale tickets in Drop.
        for (pending, _) in &mut device.ready {
            pending.ticket = None;
        }
        if let Some((_, pending)) = &mut device.completion {
            pending.ticket = None;
        }
        device.contexts.clear();
        device.ready.clear();
        device.completion = None;
        device.leases.clear();
        Ok(())
    }

    pub(super) fn collect_device(&mut self) -> HostServiceResultV1<Vec<AcceptedPrefix>> {
        let (accepted, receipt) = self.collect_device_observed()?;
        if runtime_engine::profile::is_tracing()
            && let Some(receipt) = receipt
        {
            runtime_engine::diagnostic!("{}", receipt.record());
        }
        Ok(accepted)
    }

    fn collect_device_observed(
        &mut self,
    ) -> HostServiceResultV1<(Vec<AcceptedPrefix>, Option<ActivationReceipt>)> {
        let Some(device) = &mut self.device else {
            return Ok((Vec::new(), None));
        };
        let receipt = ActivationReceipt::new(
            device,
            self.expected
                .iter()
                .filter(|item| matches!(item.phase(), BatchPhase::Prefill { .. }))
                .count(),
        );
        let mut accepted = Vec::new();
        for (mut pending, items) in std::mem::take(&mut device.ready) {
            let bytes = pending.collect()?;
            self.sampled.extend(crate::sampling_records::decode_output(
                &bytes,
                &items.iter().collect::<Vec<_>>(),
                self.limits.maximum_logits_per_row as usize,
            )?);
        }
        if let Some((completion, mut pending)) = device.completion.take() {
            let bytes = pending.collect()?;
            let start = completion.samples_offset as usize;
            let end =
                start + completion.sample_rows as usize * DEVICE_SAMPLE_RECORD_BYTES_V1 as usize;
            let samples = bytes
                .get(start..end)
                .ok_or(BundleStatusV1::InvalidArgument)?;
            self.sampled.extend(crate::sampling_records::decode_output(
                samples,
                &device.expanded.iter().collect::<Vec<_>>(),
                self.limits.maximum_logits_per_row as usize,
            )?);
            for (index, item) in device.metadata.items.iter().enumerate() {
                let offset = completion.acceptance_offset as usize + index * 32;
                let record = BundleAcceptanceResultV1::from_le_bytes(
                    bytes
                        .get(offset..offset + 32)
                        .ok_or(BundleStatusV1::InvalidArgument)?,
                )?;
                let (drafted, count, emitted, terminal, rejected, empty_row) =
                    match record.validate(item.acceptance)? {
                        DeviceAcceptanceV1::Prefix {
                            accepted_drafts,
                            emitted_rows,
                            verified_drafts,
                            terminal,
                            rejected,
                        } => (
                            verified_drafts as usize,
                            accepted_drafts as usize,
                            emitted_rows as usize,
                            terminal,
                            rejected,
                            None,
                        ),
                        DeviceAcceptanceV1::EmptyDistribution {
                            row,
                            verified_drafts,
                        } => (
                            verified_drafts as usize,
                            0,
                            0,
                            false,
                            false,
                            Some(row as usize),
                        ),
                    };
                accepted.push(AcceptedPrefix {
                    request_id: RequestId::new(item.request_id),
                    cycle_epoch: record.cycle_epoch,
                    drafted,
                    accepted: count,
                    emitted,
                    terminal,
                    rejected,
                    empty_row,
                });
            }
        } else if !device.external_rows.is_empty() {
            return Err(BundleStatusV1::InvalidState);
        }
        Ok((accepted, receipt))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runtime_bundle_abi::{
        BUNDLE_LOGITS_ELEMENT_V1_F32, BundleContextHandleV1, BundleDeviceBufferHandleV1,
        BundleDeviceBufferViewV1, BundleEventHandleV1, BundleLogitsViewV1,
    };
    use runtime_engine::{DraftWork, OutputCycle, SamplingWork};
    use runtime_sampling::{SamplingParameters, SamplingState};
    use std::cell::Cell;

    #[derive(Default)]
    struct Sampler {
        packet: RefCell<Vec<u8>>,
        issued: RefCell<Vec<(u32, Vec<u64>, SamplingDispatch)>>,
        collects: Cell<usize>,
        cancels: Cell<usize>,
        fail_collection: Cell<bool>,
        fail_drain: Cell<bool>,
        fail_issue: Cell<bool>,
        drains: RefCell<Vec<u64>>,
        trace: Rc<RefCell<Vec<&'static str>>>,
    }
    impl DeviceLogitsSamplerV1 for Sampler {
        fn sample_view(
            &self,
            _: OwnedLogitsViewV1,
            _: &[&ModelBatchItem],
        ) -> HostServiceResultV1<Vec<SampledRow>> {
            panic!("legacy sampling");
        }
        fn supports_device_results(&self) -> bool {
            true
        }
        fn drain_device_results(
            &self,
            context: BundleContextHandleV1,
            _: u32,
        ) -> HostServiceResultV1<()> {
            self.drains.borrow_mut().push(context.raw);
            self.trace.borrow_mut().push("drain");
            if self.fail_drain.get() {
                Err(BundleStatusV1::HostError)
            } else {
                Ok(())
            }
        }
        fn retain_device_drafts(
            &self,
            _: BundleDeviceDraftsV1,
        ) -> HostServiceResultV1<Box<dyn std::any::Any>> {
            Ok(Box::new(()))
        }
        fn sample_device_logits(
            &self,
            request: BundleDeviceSamplingRequestV1,
            items: &[&ModelBatchItem],
            dispatch: SamplingDispatch,
        ) -> HostServiceResultV1<()> {
            self.issued.borrow_mut().push((
                request.logits.logits.first_item_index,
                items
                    .iter()
                    .map(|item| item.sampling().unwrap().state.generated())
                    .collect(),
                dispatch,
            ));
            if self.fail_issue.get() {
                Err(BundleStatusV1::HostError)
            } else {
                Ok(())
            }
        }
        fn issue_ready_logits(
            &self,
            view: BundleReadyLogitsViewV1,
            items: &[&ModelBatchItem],
            dispatch: SamplingDispatch,
        ) -> HostServiceResultV1<u64> {
            self.issued.borrow_mut().push((
                view.logits.first_item_index,
                items
                    .iter()
                    .map(|item| item.sampling().unwrap().state.generated())
                    .collect(),
                dispatch,
            ));
            Ok(self.issued.borrow().len() as u64)
        }
        fn enqueue_device_readback(&self, _: BundleReadyDeviceViewV1) -> HostServiceResultV1<u64> {
            Ok(1)
        }
        fn collect_device_readback(
            &self,
            _: BundleReadyDeviceViewV1,
            _: u64,
        ) -> HostServiceResultV1<Vec<u8>> {
            self.collects.set(self.collects.get() + 1);
            if self.fail_collection.get() {
                Err(BundleStatusV1::HostError)
            } else {
                Ok(self.packet.borrow().clone())
            }
        }
        fn cancel_device_readback(
            &self,
            _: BundleReadyDeviceViewV1,
            _: u64,
        ) -> HostServiceResultV1<()> {
            self.cancels.set(self.cancels.get() + 1);
            Ok(())
        }
    }
    fn item(id: u64, k: usize) -> ModelBatchItem {
        let item = ModelBatchItem::new(
            RequestId::new(id),
            if k == 0 {
                BatchPhase::Decode
            } else {
                BatchPhase::Verify { draft_tokens: k }
            },
            vec![1],
            true,
        )
        .with_sampling(SamplingWork {
            parameters: SamplingParameters::new(1.0).unwrap(),
            state: SamplingState::new(7),
            history: vec![1].into(),
            emitted_tokens: 0,
            stop_ids: vec![3],
            constraint: None,
        })
        .with_output_cycle(OutputCycle {
            epoch: 9,
            remaining_tokens: 8,
            termination_stop_ids: vec![2],
        });
        if k == 0 {
            item
        } else {
            item.with_draft(DraftWork::new(k, DraftSource::Bundle))
        }
    }
    fn buffer(slot: u32, length: u64) -> BundleDeviceBufferViewV1 {
        BundleDeviceBufferViewV1 {
            buffer: BundleDeviceBufferHandleV1::from_parts(1, slot),
            length,
            ..BundleDeviceBufferViewV1::empty()
        }
    }
    fn ready(slot: u32, length: u64) -> BundleReadyDeviceViewV1 {
        BundleReadyDeviceViewV1 {
            context: BundleContextHandleV1::from_parts(1, 1),
            rank: 0,
            reserved: 0,
            buffer: buffer(slot, length),
            ready: BundleEventHandleV1::from_parts(1, 2),
        }
    }
    fn logits(first: u32, rows: u32) -> BundleReadyLogitsViewV1 {
        BundleReadyLogitsViewV1 {
            struct_size: std::mem::size_of::<BundleReadyLogitsViewV1>() as u32,
            flags: 0,
            cycle_epoch: 9,
            ready: BundleEventHandleV1::from_parts(1, 2),
            logits: BundleLogitsViewV1 {
                struct_size: std::mem::size_of::<BundleLogitsViewV1>() as u32,
                reserved: 0,
                reserved_tail: 0,
                context: BundleContextHandleV1::from_parts(1, 1),
                rank: 0,
                element_type: BUNDLE_LOGITS_ELEMENT_V1_F32,
                first_item_index: first,
                row_count: rows,
                live_width: 8,
                row_stride_elements: 8,
                buffer: buffer(3, u64::from(rows) * 32),
            },
        }
    }
    fn sink<'a>(items: &'a [ModelBatchItem], sampler: Rc<Sampler>) -> CollectingSinkV1<'a> {
        let limits = ResultSinkLimitsV1::new(192, 8, 1536);
        let metadata = cycle_metadata(items, None).unwrap();
        let device = Some(DeviceResults::new(metadata, items, sampler.as_ref(), limits).unwrap());
        CollectingSinkV1 {
            limits,
            rows: Vec::new(),
            sampled: Vec::new(),
            prompt_view_rows: Vec::new(),
            expected: items,
            sampled_indices: BTreeSet::new(),
            drafts: BTreeMap::new(),
            sampler: Some(sampler),
            defer: false,
            pending: Vec::new(),
            sampling_groups: BTreeMap::new(),
            device,
        }
    }
    fn draft(id: u64, k: u32) -> BundleDeviceDraftsV1 {
        BundleDeviceDraftsV1 {
            struct_size: std::mem::size_of::<BundleDeviceDraftsV1>() as u32,
            flags: 0,
            cycle_epoch: 9,
            request_id: id,
            slot_generation: 9,
            requested_tokens: k,
            computed_tokens: 5,
            token_stride_bytes: 4,
            proposals: ready(4, 20),
            confidence: BundleDeviceBufferViewV1::empty(),
        }
    }
    fn sample_record(token: u32, status: u32) -> Vec<u8> {
        let mut bytes = vec![0xff; 256];
        bytes[..4].copy_from_slice(&token.to_le_bytes());
        bytes[4..8].copy_from_slice(&status.to_le_bytes());
        bytes[8..16].copy_from_slice(&(-0.3125_f64).to_le_bytes());
        bytes
    }
    fn completion(items: u32, rows: u32) -> BundleDeviceCompletionV1 {
        BundleDeviceCompletionV1 {
            struct_size: std::mem::size_of::<BundleDeviceCompletionV1>() as u32,
            flags: 0,
            cycle_epoch: 9,
            item_count: items,
            sample_rows: rows,
            acceptance_offset: 0,
            samples_offset: u64::from(items) * 32,
            packet: ready(7, u64::from(items) * 32 + u64::from(rows) * 256),
        }
    }
    fn sample_request(first: u32, rows: u32) -> BundleDeviceSamplingRequestV1 {
        BundleDeviceSamplingRequestV1 {
            struct_size: std::mem::size_of::<BundleDeviceSamplingRequestV1>() as u32,
            reserved: 0,
            logits: logits(first, rows),
            output: buffer(8, u64::from(rows) * 256),
            done: BundleEventHandleV1::from_parts(1, 9),
        }
    }

    #[test]
    fn device_verify_preserves_partition_dispatch_rng_and_full_suffix_validation() {
        for concurrency in [1, 8, 32] {
            let items = (1..=concurrency).map(|id| item(id, 2)).collect::<Vec<_>>();
            let sampler = Rc::new(Sampler::default());
            let mut sink = sink(&items, sampler.clone());
            assert_eq!(
                sink.device_metadata().unwrap().termination_stop_ids,
                vec![2; concurrency as usize]
            );
            sink.device_dependency(1).unwrap();
            for id in 1..=concurrency {
                sink.device_drafts(draft(id, 2)).unwrap();
            }
            let total = concurrency as u32 * 3;
            for first in (0..total).step_by(4) {
                sink.device_sample(sample_request(first, (total - first).min(4)))
                    .unwrap();
            }
            let expected_groups = gathered_ranges(&items);
            for (first, generated, dispatch) in sampler.issued.borrow().iter() {
                let group = expected_groups
                    .iter()
                    .find(|group| group.contains(&(*first as usize)))
                    .unwrap();
                assert_eq!(dispatch.rows, group.len() as u32);
                for (offset, value) in generated.iter().enumerate() {
                    assert_eq!(*value, (*first as usize + offset) as u64 % 3);
                }
            }
            let mut packet = Vec::new();
            for _ in &items {
                packet.extend(
                    BundleAcceptanceResultV1 {
                        cycle_epoch: 9,
                        accepted_drafts: 0,
                        emitted_rows: 1,
                        flags: BUNDLE_ACCEPTANCE_REJECTED_V1,
                        status: 0,
                        failure_row: BUNDLE_ACCEPTANCE_NO_FAILURE_V1,
                        verified_drafts: 0,
                    }
                    .to_le_bytes(),
                );
            }
            for _ in &items {
                packet.extend(sample_record(4, 0));
                packet.extend(sample_record(u32::MAX, 2));
                packet.extend(sample_record(5, 0));
            }
            *sampler.packet.borrow_mut() = packet;
            sink.device_completion(completion(concurrency as u32, total))
                .unwrap();
            assert_eq!(sampler.collects.get(), 0);
            let (prefixes, activation) = sink.collect_device_observed().unwrap();
            let activation = activation.unwrap().record();
            assert_eq!(activation["kind"], "acceptance_packet");
            assert_eq!(activation["items"], concurrency);
            assert_eq!(activation["sample_rows"], total);
            assert_eq!(activation["verify_items"], concurrency);
            assert_eq!(activation["prefill_items"], 0);
            assert_eq!(activation["readback_packets"], 1);
            assert_eq!(prefixes.len(), concurrency as usize);
            assert!(
                prefixes
                    .iter()
                    .all(|prefix| prefix.emitted == 1 && prefix.rejected)
            );
            assert_eq!(sink.sampled.len(), total as usize);
            assert_eq!(sink.sampled[0].logprob().to_bits(), (-0.3125_f64).to_bits());
            assert!(sink.sampled[1].failure().is_some());
        }
    }

    #[test]
    fn stale_drafts_duplicate_rows_missing_completion_and_failed_collection_are_refused() {
        let items = vec![item(1, 2)];
        let sampler = Rc::new(Sampler::default());
        let mut sink = sink(&items, sampler.clone());
        sink.device_dependency(1).unwrap();
        assert_eq!(
            sink.device_drafts(BundleDeviceDraftsV1 {
                slot_generation: 8,
                ..draft(1, 2)
            }),
            Err(BundleStatusV1::StaleHandle)
        );
        sink.device_drafts(draft(1, 2)).unwrap();
        assert!(sink.device_drafts(draft(1, 2)).is_err());
        sink.device_sample(sample_request(0, 3)).unwrap();
        assert!(sink.device_sample(sample_request(0, 1)).is_err());
        assert!(sink.collect_device().is_err());
        sink.device_completion(completion(1, 3)).unwrap();
        sampler.fail_collection.set(true);
        assert_eq!(sink.collect_device(), Err(BundleStatusV1::HostError));
        assert_eq!(sampler.cancels.get(), 1);
    }

    #[test]
    fn invalid_unused_sample_suffix_is_fatal_and_cancellation_discards_pending_packet() {
        for status in [1, 3, 77] {
            let items = vec![item(1, 2)];
            let sampler = Rc::new(Sampler::default());
            let mut sink = sink(&items, sampler.clone());
            sink.device_dependency(1).unwrap();
            sink.device_drafts(draft(1, 2)).unwrap();
            sink.device_sample(sample_request(0, 3)).unwrap();
            let mut packet = BundleAcceptanceResultV1 {
                cycle_epoch: 9,
                accepted_drafts: 0,
                emitted_rows: 1,
                flags: BUNDLE_ACCEPTANCE_REJECTED_V1,
                status: 0,
                failure_row: BUNDLE_ACCEPTANCE_NO_FAILURE_V1,
                verified_drafts: 0,
            }
            .to_le_bytes()
            .to_vec();
            packet.extend(sample_record(4, 0));
            packet.extend(sample_record(4, 0));
            packet.extend(sample_record(u32::MAX, status));
            *sampler.packet.borrow_mut() = packet;
            sink.device_completion(completion(1, 3)).unwrap();
            assert_eq!(sink.collect_device(), Err(BundleStatusV1::HostError));
        }
        let items = vec![item(1, 2)];
        let sampler = Rc::new(Sampler::default());
        let mut sink = sink(&items, sampler.clone());
        sink.device_dependency(1).unwrap();
        sink.device_drafts(draft(1, 2)).unwrap();
        sink.device_sample(sample_request(0, 3)).unwrap();
        sink.device_completion(completion(1, 3)).unwrap();
        drop(sink);
        assert_eq!(sampler.collects.get(), 0);
        assert_eq!(sampler.cancels.get(), 1);
    }
    struct Owner {
        trace: Rc<RefCell<Vec<&'static str>>>,
        poisoned: bool,
        fail_cancel: bool,
    }
    impl Owner {
        fn new(sampler: &Sampler) -> Self {
            Self {
                trace: sampler.trace.clone(),
                poisoned: false,
                fail_cancel: false,
            }
        }
        fn poison(&mut self) {
            self.poisoned = true;
            self.trace.borrow_mut().push("poison");
        }
    }
    impl BundleExecutionV1 for Owner {
        fn execute_batch(
            &mut self,
            _: &[u8],
            _: ResultSinkLimitsV1,
        ) -> Result<Vec<OwnedLogitsRowV1>, BundleAbiError> {
            self.trace.borrow_mut().push("execute");
            Ok(Vec::new())
        }
        fn commit(&mut self) -> Result<(), BundleAbiError> {
            self.trace.borrow_mut().push("commit");
            Ok(())
        }
        fn cancel(&mut self) -> Result<(), BundleAbiError> {
            self.trace.borrow_mut().push("cancel");
            if self.fail_cancel {
                self.poison();
                Err(BundleAbiError::Poisoned)
            } else {
                Ok(())
            }
        }
        fn destroy(&mut self) -> Result<(), BundleAbiError> {
            self.trace.borrow_mut().push("destroy");
            Ok(())
        }
        fn is_poisoned(&self) -> bool {
            self.poisoned
        }
    }

    #[test]
    fn submission_returns_before_collection_for_ar_and_verify_batches() {
        for concurrency in [1, 8, 32] {
            for k in [0, 2] {
                let items = (1..=concurrency).map(|id| item(id, k)).collect::<Vec<_>>();
                let sampler = Rc::new(Sampler::default());
                let mut sink = sink(&items, sampler.clone());
                sink.device_dependency(1).unwrap();
                if k == 0 {
                    sampler
                        .packet
                        .replace((0..concurrency).flat_map(|_| sample_record(4, 0)).collect());
                    sink.device_ready(logits(0, concurrency as u32)).unwrap();
                } else {
                    for id in 1..=concurrency {
                        sink.device_drafts(draft(id, k as u32)).unwrap();
                    }
                    let rows = concurrency as u32 * 3;
                    for first in (0..rows).step_by(64) {
                        let mut request = sample_request(first, (rows - first).min(64));
                        request.output.offset = u64::from(first) * 256;
                        sink.device_sample(request).unwrap();
                    }
                    let mut packet = Vec::new();
                    for _ in &items {
                        packet.extend(
                            BundleAcceptanceResultV1 {
                                cycle_epoch: 9,
                                accepted_drafts: 0,
                                emitted_rows: 1,
                                flags: BUNDLE_ACCEPTANCE_REJECTED_V1,
                                status: 0,
                                failure_row: BUNDLE_ACCEPTANCE_NO_FAILURE_V1,
                                verified_drafts: 0,
                            }
                            .to_le_bytes(),
                        );
                    }
                    packet.extend((0..rows).flat_map(|_| sample_record(4, 0)));
                    sampler.packet.replace(packet);
                    sink.device_completion(completion(concurrency as u32, rows))
                        .unwrap();
                }
                let mut owner = Owner::new(&sampler);
                let pending = PendingBundleExecutionV1::submitted(&mut owner, sink, Owner::poison);
                assert_eq!(
                    pending.input_dependency(),
                    AsyncInputDependency::DeviceFeedback
                );
                assert_eq!(pending.cycle_epoch(), Some(9));
                assert_eq!(
                    sampler.collects.get(),
                    0,
                    "host preparation precedes the result fence"
                );
                assert!(!sampler.issued.borrow().is_empty());
                let rows = pending.collect().unwrap();
                assert_eq!(rows.sampled.len(), concurrency as usize * (k + 1));
                assert!(
                    rows.sampled
                        .iter()
                        .all(|row| row.logprob().to_bits() == (-0.3125_f64).to_bits())
                );
                assert_eq!(sampler.collects.get(), 1);
                assert!(
                    owner.trace.borrow().is_empty(),
                    "collection does not finalize the bundle"
                );
                owner.commit().unwrap();
                assert_eq!(*owner.trace.borrow(), ["commit"]);
            }
        }
    }

    #[test]
    fn submission_drop_and_cancel_fence_before_bundle_rewrites() {
        for explicit in [false, true] {
            let items = vec![item(1, 0)];
            let sampler = Rc::new(Sampler::default());
            let mut sink = sink(&items, sampler.clone());
            sink.device_dependency(0).unwrap();
            sink.device_ready(logits(0, 1)).unwrap();
            let mut owner = Owner::new(&sampler);
            let pending = PendingBundleExecutionV1::submitted(&mut owner, sink, Owner::poison);
            if explicit {
                pending.cancel().unwrap();
            } else {
                drop(pending);
            }
            assert_eq!(*owner.trace.borrow(), ["drain", "cancel"]);
            assert_eq!(
                sampler.collects.get(),
                0,
                "cancelled output is never decoded"
            );
            assert_eq!(
                sampler.cancels.get(),
                0,
                "drained tickets are not cancelled twice"
            );
        }
    }

    #[test]
    fn partial_submission_without_a_completion_drains_every_owner() {
        let items = vec![item(1, 0), item(2, 0)];
        let sampler = Rc::new(Sampler::default());
        let mut sink = sink(&items, sampler.clone());
        sink.device_dependency(1).unwrap();
        sink.device_sample(sample_request(0, 1)).unwrap();
        let mut second = sample_request(1, 1);
        second.logits.logits.context = BundleContextHandleV1::from_parts(1, 2);
        second.logits.logits.rank = 1;
        sampler.fail_issue.set(true);
        assert_eq!(sink.device_sample(second), Err(BundleStatusV1::HostError));
        let mut owner = Owner::new(&sampler);
        let pending = PendingBundleExecutionV1::submitted(&mut owner, sink, Owner::poison);
        assert!(
            pending.collect().is_err(),
            "missing packet is never publishable"
        );
        assert_eq!(sampler.drains.borrow().len(), 2);
        assert_eq!(*owner.trace.borrow(), ["drain", "drain", "cancel"]);
    }

    #[test]
    fn failed_submission_fence_poison_prevents_cancel_and_double_finalization() {
        let items = vec![item(1, 0)];
        let sampler = Rc::new(Sampler::default());
        let mut sink = sink(&items, sampler.clone());
        sink.device_dependency(1).unwrap();
        sink.device_ready(logits(0, 1)).unwrap();
        sampler.fail_drain.set(true);
        let mut owner = Owner::new(&sampler);
        let pending = PendingBundleExecutionV1::submitted(&mut owner, sink, Owner::poison);
        assert!(pending.cancel().is_err());
        assert!(owner.poisoned);
        assert_eq!(*owner.trace.borrow(), ["drain", "poison"]);
        assert_eq!(sampler.collects.get(), 0);
    }

    #[test]
    fn collection_and_cancel_failures_keep_the_instance_terminal() {
        for cancel_fails in [false, true] {
            let items = vec![item(1, 0)];
            let sampler = Rc::new(Sampler::default());
            let mut sink = sink(&items, sampler.clone());
            sink.device_dependency(0).unwrap();
            sink.device_ready(logits(0, 1)).unwrap();
            sampler.fail_collection.set(true);
            let mut owner = Owner::new(&sampler);
            owner.fail_cancel = cancel_fails;
            let pending = PendingBundleExecutionV1::submitted(&mut owner, sink, Owner::poison);
            assert!(pending.collect().is_err());
            assert_eq!(owner.poisoned, cancel_fails);
            assert_eq!(
                owner
                    .trace
                    .borrow()
                    .iter()
                    .filter(|&&call| call == "cancel")
                    .count(),
                1
            );
            assert_eq!(&owner.trace.borrow()[..2], ["drain", "cancel"]);
        }
    }

    #[test]
    fn legacy_submission_still_collects_synchronously_and_defaults_to_host_values() {
        let sampler = Sampler::default();
        let mut owner = Owner::new(&sampler);
        let pending = owner
            .submit_with_device_sampling(&[], ResultSinkLimitsV1::new(1, 8, 8), &[], None, None)
            .unwrap();
        assert_eq!(pending.input_dependency(), AsyncInputDependency::HostValues);
        assert_eq!(pending.cycle_epoch(), None);
        assert_eq!(*sampler.trace.borrow(), ["execute"]);
        pending.cancel().unwrap();
        assert_eq!(*sampler.trace.borrow(), ["execute", "cancel"]);
    }
    #[test]
    fn activation_requires_collected_device_work_and_never_marks_fallback_or_failure() {
        let items = vec![item(1, 0)];
        for fail in [false, true] {
            let sampler = Rc::new(Sampler::default());
            let mut sink = sink(&items, sampler.clone());
            assert!(
                sink.collect_device_observed().unwrap().1.is_none(),
                "capability alone is not activation"
            );
            sink.device_dependency(0).unwrap();
            assert!(
                sink.collect_device_observed().unwrap().1.is_none(),
                "a dependency declaration is not activation"
            );
            sink.device_ready(logits(0, 1)).unwrap();
            sampler.packet.replace(sample_record(4, 0));
            sampler.fail_collection.set(fail);
            let result = sink.collect_device_observed();
            if fail {
                assert!(
                    result.is_err(),
                    "failed collection has no activation receipt"
                );
            } else {
                let (rows, activation) = result.unwrap();
                assert!(rows.is_empty());
                let record = activation.unwrap().record();
                assert_eq!(record["schema"], "chopinfer.device-results.v1");
                assert_eq!(record["kind"], "ready_logits");
                assert_eq!(record["cycle_epoch"], 9);
                assert_eq!(record["items"], 1);
                assert_eq!(record["sample_rows"], 1);
                assert_eq!(record["verify_items"], 0);
                assert_eq!(record["readback_packets"], 1);
                assert!(
                    sink.collect_device_observed().unwrap().1.is_none(),
                    "a receipt cannot be repeated"
                );
            }
        }
    }
}
