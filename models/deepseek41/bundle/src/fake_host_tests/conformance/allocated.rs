//! Device-result execution for the profile's allocated startup and route waves.
use super::*;
use conformance::Startup;
use runtime_bundle_abi::device_results::*;
use runtime_bundle_sdk::VerifyAllocationV1;

struct Sink<'a> {
    host: &'a Host,
    metadata: DeviceCycleMetadataV1,
    drafts: Vec<u64>,
    rows: u32,
    completion: Option<BundleDeviceCompletionV1>,
}

impl LogitsRowSinkV1 for Sink<'_> {
    fn device_results_enabled(&self) -> bool {
        true
    }

    fn limits(&self) -> ResultSinkLimitsV1 {
        Limits::limits()
    }

    fn emit_row(&mut self, _row: OwnedLogitsRowV1) -> HostServiceResultV1<()> {
        Err(BundleStatusV1::Unsupported)
    }

    fn device_cycle_metadata(&self) -> HostServiceResultV1<DeviceCycleMetadataV1> {
        Ok(self.metadata.clone())
    }

    fn declare_input_dependency(&mut self, dependency: u32) -> HostServiceResultV1<()> {
        assert_eq!(dependency, DEVICE_RESULTS_HOST_VALUES_V1);
        Ok(())
    }

    fn emit_device_drafts(&mut self, drafts: BundleDeviceDraftsV1) -> HostServiceResultV1<()> {
        drafts.validate()?;
        self.drafts.push(drafts.request_id);
        Ok(())
    }

    fn sample_device_logits(
        &mut self,
        sample: BundleDeviceSamplingRequestV1,
    ) -> HostServiceResultV1<()> {
        sample.validate().unwrap();
        self.rows += sample.logits.logits.row_count;
        let stream = BundleStreamHandleV1::from_parts(
            1,
            self.host.rank_streams.borrow()[&sample.logits.logits.rank],
        );
        self.host.record_event(sample.done, stream)
    }

    fn emit_device_completion(
        &mut self,
        completion: BundleDeviceCompletionV1,
    ) -> HostServiceResultV1<()> {
        completion.validate()?;
        assert!(self.completion.replace(completion).is_none());
        Ok(())
    }
}

pub(super) fn execute(
    adapter: &Served,
    instance: &mut BundleInstanceV1,
    items: &[conformance::Item],
    allocation: VerifyAllocationV1,
) -> Result<(), BundleAbiError> {
    let width = allocation.width();
    let reserved: usize = items.iter().map(|item| item.tokens.len()).sum();
    assert!((allocation.width() as usize) < reserved);
    let batch = BatchRequestV1::decode(&adapter.step(items))
        .unwrap()
        .with_verify_allocation(allocation);
    let epoch = adapter.allocated_epoch.get().checked_add(1).unwrap();
    adapter.allocated_epoch.set(epoch);
    let mut row = 0;
    let mut drafted = 0;
    let metadata = DeviceCycleMetadataV1 {
        cycle_epoch: epoch,
        termination_stop_ids: vec![3],
        items: items
            .iter()
            .map(|item| {
                let first = row;
                row += item.tokens.len() as u32;
                let draft_item_index = if item.phase == Phase::Verify {
                    let index = drafted;
                    drafted += 1;
                    index
                } else {
                    u32::MAX
                };
                DeviceCycleItemV1 {
                    request_id: item.request,
                    slot_generation: 1,
                    acceptance: BundleAcceptanceInputV1 {
                        cycle_epoch: epoch,
                        remaining_tokens: 16,
                        first_sample_row: first,
                        draft_item_index,
                        draft_tokens: item.tokens.len() as u32 - 1,
                        stop_offset: 0,
                        stop_count: 1,
                        reserved: 0,
                    },
                }
            })
            .collect(),
    };
    let mut sink = Sink {
        host: &adapter.host,
        metadata,
        drafts: Vec::new(),
        rows: 0,
        completion: None,
    };
    let from = adapter.host.journal.borrow().len();
    instance.execute(&batch.encode().unwrap(), &mut sink)?;
    assert_eq!(
        sink.drafts,
        items
            .iter()
            .filter(|item| item.phase == Phase::Verify)
            .map(|item| item.request)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        sink.rows as usize, reserved,
        "sampling uses the reserved row layout"
    );
    assert_eq!(sink.completion.unwrap().sample_rows as usize, reserved);
    let journal = adapter.host.journal.borrow();
    let step = &journal[from..];
    let hash = step
        .iter()
        .position(|line| {
            line.starts_with("launch ") && line.contains(crate::sm120::engram_hash_rows::SYMBOL)
        })
        .expect("compacted target rows are hashed");
    let accepted = step
        .iter()
        .position(|line| {
            line.starts_with("launch ")
                && line.contains(crate::sm120::allocated_acceptance_entries::SYMBOL)
        })
        .expect("allocated acceptance runs");
    let target = &step[hash..accepted];
    let wide_head = crate::sm120::f32_verify::entries()
        .unwrap()
        .into_iter()
        .find(|entry| entry.entry().query().n == adapter.host.shape.vocab_rows())
        .unwrap()
        .symbols();
    for stream in adapter.host.rank_streams.borrow().values() {
        let prefix = format!("launch {stream} ");
        assert!(
            step[..hash].iter().any(|line| line.starts_with(&prefix)
                && line.contains(crate::sm120::allocate_entries::SYMBOL)),
            "each rank compacts rows before target hashing"
        );
        let mut symbols: Vec<_> = target
            .iter()
            .filter_map(|line| line.strip_prefix(&prefix))
            .map(|line| line.split_whitespace().next().unwrap())
            .collect();
        // Ordinary journals record a replay without repeating its launches.
        // Read the original capture; route journals already expand replays.
        if !adapter.host.device.symbols_only.get() {
            let replay = format!("replay {stream} ");
            for graph in target.iter().filter_map(|line| line.strip_prefix(&replay)) {
                let captured = format!("captured {stream} {graph} ");
                let end = journal
                    .iter()
                    .position(|line| line.starts_with(&captured))
                    .expect("replayed graph was captured");
                let capture = format!("capture {stream}");
                let start = journal[..end]
                    .iter()
                    .rposition(|line| line == &capture)
                    .expect("capture began");
                symbols.extend(
                    journal[start..end]
                        .iter()
                        .filter_map(|line| line.strip_prefix(&prefix))
                        .map(|line| line.split_whitespace().next().unwrap()),
                );
            }
        }
        assert!(
            symbols
                .iter()
                .filter(|symbol| symbol.starts_with("chop_row_gather_by_index"))
                .count()
                >= 3,
            "width {width}, reserved {reserved}, stream {stream}: packed feature and reserved head gathers"
        );
        assert_eq!(
            symbols
                .iter()
                .any(|symbol| symbol.starts_with("chop_deepseek41_head_f32")),
            reserved <= 32,
            "width {width}, reserved {reserved}, stream {stream}: target head route follows reserved rows"
        );
        for symbol in &wide_head {
            assert_eq!(
                symbols.contains(&symbol.as_str()),
                reserved > 32,
                "width {width}, reserved {reserved}, stream {stream}: wide target head {symbol}"
            );
        }
    }
    Ok(())
}
