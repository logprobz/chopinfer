use super::*;
use runtime_bundle_abi::device_results::*;

struct DeviceSink {
    host: Rc<Host>,
    metadata: Option<DeviceCycleMetadataV1>,
    legacy: Sink,
    drafts: Vec<BundleDeviceDraftsV1>,
    samples: Vec<BundleDeviceSamplingRequestV1>,
    completion: Option<BundleDeviceCompletionV1>,
}
impl LogitsRowSinkV1 for DeviceSink {
    fn device_results_enabled(&self) -> bool {
        true
    }
    fn limits(&self) -> ResultSinkLimitsV1 {
        self.legacy.limits()
    }
    fn emit_row(&mut self, row: OwnedLogitsRowV1) -> HostServiceResultV1<()> {
        self.legacy.emit_row(row)
    }
    fn emit_view(&mut self, view: OwnedLogitsViewV1) -> HostServiceResultV1<()> {
        self.legacy.emit_view(view)
    }
    fn emit_draft_tokens(&mut self, drafts: OwnedDraftTokensV1) -> HostServiceResultV1<()> {
        self.legacy.emit_draft_tokens(drafts)
    }
    fn device_cycle_metadata(&self) -> HostServiceResultV1<DeviceCycleMetadataV1> {
        self.metadata.clone().ok_or(BundleStatusV1::Unsupported)
    }
    fn declare_input_dependency(&mut self, dependency: u32) -> HostServiceResultV1<()> {
        assert_eq!(dependency, DEVICE_RESULTS_HOST_VALUES_V1);
        self.host.note("device dependency".into());
        Ok(())
    }
    fn emit_device_drafts(&mut self, drafts: BundleDeviceDraftsV1) -> HostServiceResultV1<()> {
        assert!(self.samples.is_empty());
        assert!(self.host.journal.borrow().iter().any(|line| {
            line.ends_with(&format!(" {}", drafts.proposals.ready.slot()))
                && line.starts_with("record ")
        }));
        self.host.check_span(DeviceBufferSpanV1::new(
            drafts.proposals.buffer.buffer,
            drafts.proposals.buffer.offset,
            drafts.proposals.buffer.length,
        ));
        self.drafts.push(drafts);
        self.host.note("device drafts".into());
        Ok(())
    }
    fn sample_device_logits(
        &mut self,
        sample: BundleDeviceSamplingRequestV1,
    ) -> HostServiceResultV1<()> {
        assert!(!self.drafts.is_empty());
        assert!(self.host.journal.borrow().iter().any(|line| {
            line.ends_with(&format!(" {}", sample.logits.ready.slot()))
                && line.starts_with("record ")
        }));
        sample.validate().unwrap();
        self.host.check_span(DeviceBufferSpanV1::new(
            sample.output.buffer,
            sample.output.offset,
            sample.output.length,
        ));
        let stream = BundleStreamHandleV1::from_parts(
            1,
            self.host.rank_streams.borrow()[&sample.logits.logits.rank],
        );
        self.host.record_event(sample.done, stream)?;
        self.host.note("device sample".into());
        self.samples.push(sample);
        Ok(())
    }
    fn emit_device_completion(
        &mut self,
        packet: BundleDeviceCompletionV1,
    ) -> HostServiceResultV1<()> {
        packet.validate()?;
        assert!(self.completion.is_none());
        let journal = self.host.journal.borrow();
        for sample in &self.samples {
            assert!(journal.iter().any(|line| line.starts_with("wait ")
                && line.ends_with(&format!(" {}", sample.done.slot()))));
        }
        assert!(journal.iter().any(|line| {
            line.contains(crate::sm120::host_acceptance_entries::SYMBOL)
                && line.starts_with("launch ")
        }));
        drop(journal);
        self.completion = Some(packet);
        self.host.note("device completion readback".into());
        Ok(())
    }
}

fn batch(items: u32, start: u64, k: u32) -> Vec<u8> {
    BatchRequestV1::new(
        BatchWireVersionV1::V3,
        (0..items)
            .map(|slot| BatchItemV1::verify(u64::from(slot + 1), k + 1, true, vec![7]))
            .collect(),
        (0..items)
            .map(|slot| {
                BatchAppendV1::new(
                    start,
                    u64::from(k + 1),
                    start + u64::from(k + 1),
                    vec![u64::from(slot)],
                    None,
                )
            })
            .collect(),
    )
    .with_state_slots((0..items).map(u64::from).collect())
    .encode()
    .unwrap()
}
fn metadata(items: u32, k: u32, epoch: u64) -> DeviceCycleMetadataV1 {
    DeviceCycleMetadataV1 {
        cycle_epoch: epoch,
        termination_stop_ids: vec![3],
        items: (0..items)
            .map(|slot| DeviceCycleItemV1 {
                request_id: u64::from(slot + 1),
                slot_generation: 11,
                acceptance: BundleAcceptanceInputV1 {
                    cycle_epoch: epoch,
                    remaining_tokens: 9,
                    first_sample_row: slot * (k + 1),
                    draft_item_index: slot,
                    draft_tokens: k,
                    stop_offset: 0,
                    stop_count: 1,
                    reserved: 0,
                },
            })
            .collect(),
    }
}

#[test]
fn device_results_preserve_draft_order_wait_for_sampling_and_collect_once() {
    for shape in crate::shape::topologies() {
        for (items, k) in [(1_u32, 5_u32), (8, 3)] {
            let (host, mut instance) = create_with_limits(shape, 8, 32);
            for slot in 0..items {
                execute(
                    &mut instance,
                    &host,
                    &super::super::prompt(u64::from(slot + 1), u64::from(slot), 0, 3, true),
                );
                instance.commit().unwrap();
            }
            for epoch in 1..=3 {
                let mut sink = DeviceSink {
                    host: Rc::clone(&host),
                    metadata: Some(metadata(items, k, epoch)),
                    legacy: Sink::default(),
                    drafts: Vec::new(),
                    samples: Vec::new(),
                    completion: None,
                };
                let start = host.journal.borrow().len();
                instance
                    .execute(
                        &batch(items, 3 + (epoch - 1) * u64::from(k + 1), k),
                        &mut sink,
                    )
                    .unwrap_or_else(|e| panic!("{e:?} {:?}", host.logs.borrow()));
                assert_eq!(sink.drafts.len(), items as usize);
                assert!(sink.legacy.draft_tokens().is_empty());
                assert_eq!(
                    sink.samples
                        .iter()
                        .map(|s| s.logits.logits.row_count)
                        .sum::<u32>(),
                    items * (k + 1)
                );
                let packet = sink.completion.unwrap();
                assert_eq!(
                    (packet.cycle_epoch, packet.item_count, packet.sample_rows),
                    (epoch, items, items * (k + 1))
                );
                let journal = host.journal.borrow();
                let step = &journal[start..];
                assert_eq!(
                    step.iter()
                        .filter(|line| line.starts_with("synchronize "))
                        .count(),
                    0
                );
                assert_eq!(
                    step.iter()
                        .filter(|line| line.starts_with("download "))
                        .count(),
                    shape.ranks() as usize + 1
                );
                let completed = step
                    .iter()
                    .position(|line| line == "device completion readback")
                    .unwrap();
                let first_download = step
                    .iter()
                    .position(|line| line.starts_with("download "))
                    .unwrap();
                assert!(
                    completed < first_download,
                    "sampling and acceptance must submit before host status checks"
                );
                println!(
                    "{}",
                    serde_json::json!({"ranks":shape.ranks(),"items":items,"k":k,"epoch":epoch,"explicit_syncs":0,"physical_readbacks":shape.ranks()+3,"completion_readbacks":1})
                );
                drop(journal);
                if epoch < 3 {
                    instance
                        .commit_accepted(
                            &(0..items)
                                .map(|slot| kept(u64::from(slot + 1), k + 1))
                                .collect::<Vec<_>>(),
                        )
                        .unwrap();
                } else {
                    instance.cancel().unwrap();
                    for slot in 0..items {
                        assert!(
                            crate::instance::dspark_context(instance.handle(), u64::from(slot + 1))
                                .is_none()
                        );
                    }
                }
            }
            instance.destroy().unwrap();
            assert_eq!(host.acquired.get(), host.released.get());
        }
    }
}

#[test]
fn unsupported_metadata_keeps_legacy_draft_results() {
    let (host, mut instance) = create(Shape::new(&FLASH, Ranks::TP1).unwrap());
    execute(
        &mut instance,
        &host,
        &super::super::prompt(1, 0, 0, 3, true),
    );
    instance.commit().unwrap();
    let mut sink = DeviceSink {
        host: Rc::clone(&host),
        metadata: None,
        legacy: Sink::default(),
        drafts: Vec::new(),
        samples: Vec::new(),
        completion: None,
    };
    instance.execute(&batch(1, 3, 5), &mut sink).unwrap();
    assert_eq!(sink.legacy.draft_tokens().len(), 1);
    assert!(sink.drafts.is_empty() && sink.samples.is_empty() && sink.completion.is_none());
    instance.cancel().unwrap();
    instance.destroy().unwrap();
}
