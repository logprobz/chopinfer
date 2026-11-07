use super::*;
use runtime_bundle_abi::device_results::*;

pub(super) struct DeviceSink {
    pub(super) host: Rc<Host>,
    pub(super) metadata: Option<DeviceCycleMetadataV1>,
    pub(super) legacy: Sink,
    pub(super) drafts: Vec<BundleDeviceDraftsV1>,
    pub(super) samples: Vec<BundleDeviceSamplingRequestV1>,
    pub(super) completion: Option<BundleDeviceCompletionV1>,
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
        // An allocated step's acceptance walks its verified drafts.
        assert!(journal.iter().any(|line| {
            (line.contains(crate::sm120::host_acceptance_entries::SYMBOL)
                || line.contains(crate::sm120::allocated_acceptance_entries::SYMBOL))
                && line.starts_with("launch ")
        }));
        drop(journal);
        self.completion = Some(packet);
        self.host.note("device completion readback".into());
        Ok(())
    }
}

/// One verify step of `items` items in slots from 0, each `k` drafts
/// after `start` committed tokens.
pub(super) fn batch(items: u32, start: u64, k: u32) -> Vec<u8> {
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
pub(super) fn metadata(items: u32, k: u32, epoch: u64) -> DeviceCycleMetadataV1 {
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
fn c8_device_feedback_queues_a_successor_after_current_completion() {
    let items = 8;
    let (host, mut instance) =
        create_with_limits(Shape::new(&FLASH, Ranks::TP4).unwrap(), items, 32);
    let prompt = BatchRequestV1::new(
        BatchWireVersionV1::V3,
        (1..=items)
            .map(|request| BatchItemV1::new(u64::from(request), false, 3, true, vec![7; 3]))
            .collect(),
        (0..items)
            .map(|slot| BatchAppendV1::new(0, 3, 3, vec![u64::from(slot)], None))
            .collect(),
    )
    .with_state_slots((0..items).map(u64::from).collect())
    .encode()
    .unwrap();
    execute(&mut instance, &host, &prompt);
    instance.commit().unwrap();
    for cycle in 0..2 {
        let epoch = cycle + 1;
        let projected = if cycle == 0 { items * 3 } else { items * 4 };
        let status: usize = (0..4)
            .map(|rank| {
                let indexes = if rank == 0 {
                    FLASH.shape.csa.index_sources.len()
                } else {
                    0
                };
                4 * (2 * (projected as usize + items as usize * 5)
                    + items as usize * 4 * (indexes + 2))
            })
            .sum();
        let mut bytes = vec![0; status + items as usize * 40];
        let offset = bytes.len();
        bytes.resize(
            offset + crate::dspark::results::SAMPLE_OFFSET as usize + items as usize * 4 * 256,
            0,
        );
        for index in 0..items as usize {
            let accepted = BundleAcceptanceResultV1 {
                cycle_epoch: epoch,
                accepted_drafts: 3,
                emitted_rows: 4,
                flags: 0,
                status: 0,
                failure_row: BUNDLE_ACCEPTANCE_NO_FAILURE_V1,
                verified_drafts: 0,
            };
            bytes[offset + index * 32..offset + (index + 1) * 32]
                .copy_from_slice(&accepted.to_le_bytes());
            let sample =
                offset + crate::dspark::results::SAMPLE_OFFSET as usize + (index * 4 + 3) * 256;
            bytes[sample..sample + 4].copy_from_slice(&7_u32.to_le_bytes());
        }
        host.injected.replace(Some(bytes));
        let begin = host.journal.borrow().len();
        let mut sink = DeviceSink {
            host: Rc::clone(&host),
            metadata: Some(metadata(items, 3, epoch)),
            legacy: Sink::default(),
            drafts: Vec::new(),
            samples: Vec::new(),
            completion: None,
        };
        instance
            .execute(&batch(items, 3 + cycle * 4, 3), &mut sink)
            .unwrap();
        assert!(
            host.injected.borrow().is_none(),
            "feedback packet length must match the real boundary"
        );
        assert!(sink.completion.is_some());
        let journal = host.journal.borrow();
        let step = &journal[begin..];
        let complete = step
            .iter()
            .position(|s| s == "device completion readback")
            .unwrap();
        let download = step
            .iter()
            .position(|s| s.starts_with("download "))
            .unwrap();
        assert!(complete < download);
        assert_eq!(
            step.iter().filter(|s| s.starts_with("download ")).count(),
            1
        );
        assert_eq!(
            step[download + 1..]
                .iter()
                .filter(|s| s.starts_with("replay "))
                .count(),
            // Per rank: one projection, then the default eight-item draft
            // chunks.
            4 * (1 + (items as usize).div_ceil(8)),
            "private projection and draft queue after current result collection"
        );
        if cycle == 1 {
            let hash = step
                .iter()
                .position(|s| s.contains(crate::sm120::engram_hash_rows::SYMBOL))
                .unwrap();
            assert!(!step[..hash].iter().any(|s| s.starts_with("replay ")));
        }
        drop(journal);
        instance
            .commit_accepted(
                &(1..=items)
                    .map(|id| kept(u64::from(id), 4))
                    .collect::<Vec<_>>(),
            )
            .unwrap();
    }
    for request in 1..=items {
        instance.retire_request(u64::from(request)).unwrap();
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

#[test]
fn default_device_results_activate_and_zero_restores_legacy_dspark_journals() {
    fn run(value: Option<&'static str>, legacy_host: bool) -> (Vec<String>, bool) {
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let (host, mut instance) = create_with_limits(shape, 1, 32);
        execute(
            &mut instance,
            &host,
            &super::super::prompt(1, 0, 0, 3, true),
        );
        instance.commit().unwrap();
        let enabled = !legacy_host
            && crate::device_results::with_override(value, || {
                crate::device_results::enabled(shape, KvStorage::Native)
            });
        let mut packet = false;
        if enabled {
            let mut sink = DeviceSink {
                host: Rc::clone(&host),
                metadata: Some(metadata(1, 5, 1)),
                legacy: Sink::default(),
                drafts: Vec::new(),
                samples: Vec::new(),
                completion: None,
            };
            instance.execute(&batch(1, 3, 5), &mut sink).unwrap();
            packet = sink.completion.is_some();
            assert!(sink.legacy.draft_tokens().is_empty());
        } else {
            let sink = execute(&mut instance, &host, &batch(1, 3, 5));
            assert_eq!(sink.draft_tokens().len(), 1);
        }
        instance.commit_accepted(&[kept(1, 1)]).unwrap();
        instance.retire_request(1).unwrap();
        instance.destroy().unwrap();
        let journal = host.journal.borrow().clone();
        (journal, packet)
    }
    let (default, active) = run(None, false);
    assert!(active);
    let (disabled, active) = run(Some("0"), false);
    assert!(!active);
    let (legacy, active) = run(None, true);
    assert!(!active);
    assert_eq!(disabled, legacy);
    assert_ne!(default, legacy);
}

#[test]
fn device_results_preserve_draft_order_wait_for_sampling_and_collect_once() {
    for shape in crate::shape::topologies() {
        for (items, k) in [(1_u32, 5_u32), (8, 3)] {
            let (host, mut instance) = create_with_logs(shape, 8, 32, true);
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
                let log = host.logs.borrow();
                let timing = log
                    .iter()
                    .rev()
                    .find(|line| line.contains("\"phase\":\"dspark-engram\""))
                    .unwrap();
                let timing: serde_json::Value = serde_json::from_str(timing).unwrap();
                assert_eq!(timing["device_results"], true);
                // The default successor joins every width's status in one packet.
                let downloads = 1;
                assert_eq!(timing["bundle_readbacks"], downloads + 2);
                assert_eq!(timing["explicit_bundle_syncs"], 0);
                drop(log);
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
                    downloads as usize
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
                    serde_json::json!({"ranks":shape.ranks(),"items":items,"k":k,"epoch":epoch,"explicit_syncs":0,"physical_readbacks":downloads+2,"completion_readbacks":1})
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

/// Diagnostic taps read the logits of a device-results verify step, which
/// reach the host as no view, as they read host-view logits: the capture
/// holds a dump of the verify step.
#[cfg(feature = "diagnostic-taps")]
#[test]
fn device_result_logits_reach_the_diagnostic_capture() {
    let root = std::path::PathBuf::from(std::env::var_os("CHOPINFER_DATA").unwrap())
        .join("deepseek41-tests")
        .join(format!("dspark-device-logits-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let (host, mut instance) = create_with_options(
        shape,
        1,
        32,
        &[(
            model_deepseek41_contract::prefill_profile::CHUNK,
            8192.into(),
        )],
    );
    crate::instance::set_dspark_state_capture_root_for_test(instance.handle(), Some(root.clone()));
    std::fs::write(root.join("enabled"), []).unwrap();
    execute(
        &mut instance,
        &host,
        &super::super::prompt(1, 0, 0, 3, true),
    );
    instance.commit().unwrap();
    let dumps = || {
        std::fs::read_dir(root.join("logits"))
            .map(|entries| {
                entries
                    .map(|entry| entry.unwrap().path())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    };
    let before = dumps();
    let mut sink = DeviceSink {
        host: Rc::clone(&host),
        metadata: Some(metadata(1, 5, 1)),
        legacy: Sink::default(),
        drafts: Vec::new(),
        samples: Vec::new(),
        completion: None,
    };
    instance.execute(&batch(1, 3, 5), &mut sink).unwrap();
    assert!(sink.completion.is_some());
    let fresh: Vec<_> = dumps()
        .into_iter()
        .filter(|path| !before.contains(path))
        .collect();
    assert!(!fresh.is_empty(), "no verify logits dump");
    for path in fresh {
        let bytes = std::fs::metadata(&path).unwrap().len();
        assert!(bytes > 0 && bytes % 4 == 0, "{path:?}: {bytes} bytes");
    }
    instance.commit_accepted(&[kept(1, 1)]).unwrap();
    instance.retire_request(1).unwrap();
    instance.destroy().unwrap();
    std::fs::remove_dir_all(root).unwrap();
}

/// Diagnostic taps record each drafting cycle's ordered proposal IDs with
/// the tokens requested, the draft tokens the host reserved for the step it
/// verifies, and the rows the host keeps at commit.
#[cfg(feature = "diagnostic-taps")]
#[test]
fn device_result_proposals_and_kept_rows_reach_the_diagnostic_capture() {
    let root = std::path::PathBuf::from(std::env::var_os("CHOPINFER_DATA").unwrap())
        .join("deepseek41-tests")
        .join(format!("dspark-device-drafts-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let (host, mut instance) = create_with_options(
        shape,
        1,
        32,
        &[(
            model_deepseek41_contract::prefill_profile::CHUNK,
            8192.into(),
        )],
    );
    crate::instance::set_dspark_state_capture_root_for_test(instance.handle(), Some(root.clone()));
    std::fs::write(root.join("enabled"), []).unwrap();
    execute(
        &mut instance,
        &host,
        &super::super::prompt(1, 0, 0, 3, true),
    );
    instance.commit().unwrap();
    let mut sink = DeviceSink {
        host: Rc::clone(&host),
        metadata: Some(metadata(1, 5, 1)),
        legacy: Sink::default(),
        drafts: Vec::new(),
        samples: Vec::new(),
        completion: None,
    };
    instance.execute(&batch(1, 3, 5), &mut sink).unwrap();
    instance.commit_accepted(&[kept(1, 1)]).unwrap();
    let read = |prefix: &str| -> Vec<serde_json::Value> {
        let mut paths: Vec<_> = std::fs::read_dir(root.join("drafts"))
            .map(|entries| entries.map(|entry| entry.unwrap().path()).collect())
            .unwrap_or_default();
        paths.retain(|path: &std::path::PathBuf| {
            path.file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .starts_with(prefix)
        });
        paths
            .iter()
            .map(|path| serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap())
            .collect()
    };
    let cycles = read("step-");
    let proposals: Vec<_> = cycles
        .iter()
        .flat_map(|cycle| cycle["proposals"].as_array().unwrap().clone())
        .filter(|item| item["request"] == 1)
        .collect();
    assert!(!proposals.is_empty(), "no proposal record: {cycles:?}");
    for item in &proposals {
        let requested = item["requested"].as_u64().unwrap();
        assert!(requested > 0);
        assert_eq!(item["tokens"].as_array().unwrap().len() as u64, requested);
    }
    assert!(cycles.iter().any(|cycle| {
        cycle["reserved"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item["request"] == 1 && item["tokens"] == 5)
    }));
    let kept = read("kept-");
    assert_eq!(kept.len(), 1, "{kept:?}");
    assert_eq!(kept[0]["kept"][0]["request"], 1);
    assert_eq!(kept[0]["kept"][0]["kept"], 1);
    instance.retire_request(1).unwrap();
    instance.destroy().unwrap();
    std::fs::remove_dir_all(root).unwrap();
}
