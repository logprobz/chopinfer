use super::*;
use runtime_bundle_abi::device_results::*;

struct ReadySink {
    host: Rc<Host>,
    epoch: u64,
    ids: Vec<u64>,
    ready: Vec<BundleReadyLogitsViewV1>,
    legacy: Vec<OwnedLogitsViewV1>,
}
impl LogitsRowSinkV1 for ReadySink {
    fn limits(&self) -> ResultSinkLimitsV1 {
        ResultSinkLimitsV1::new(256, 1 << 20, 256 << 20)
    }
    fn emit_row(&mut self, _: OwnedLogitsRowV1) -> HostServiceResultV1<()> {
        Err(BundleStatusV1::Unsupported)
    }
    fn emit_view(&mut self, view: OwnedLogitsViewV1) -> HostServiceResultV1<()> {
        self.legacy.push(view);
        Ok(())
    }
    fn device_results_enabled(&self) -> bool {
        true
    }
    fn device_cycle_metadata(&self) -> HostServiceResultV1<DeviceCycleMetadataV1> {
        Ok(DeviceCycleMetadataV1 {
            cycle_epoch: self.epoch,
            termination_stop_ids: Vec::new(),
            items: self
                .ids
                .iter()
                .enumerate()
                .map(|(row, &id)| DeviceCycleItemV1 {
                    request_id: id,
                    slot_generation: 1,
                    acceptance: BundleAcceptanceInputV1 {
                        cycle_epoch: self.epoch,
                        remaining_tokens: 8,
                        first_sample_row: row as u32,
                        draft_item_index: BUNDLE_ACCEPTANCE_NO_DRAFT_V1,
                        draft_tokens: 0,
                        stop_offset: 0,
                        stop_count: 0,
                        reserved: 0,
                    },
                })
                .collect(),
        })
    }
    fn declare_input_dependency(&mut self, dependency: u32) -> HostServiceResultV1<()> {
        assert_eq!(dependency, DEVICE_RESULTS_HOST_VALUES_V1);
        Ok(())
    }
    fn emit_ready_logits_view(&mut self, view: BundleReadyLogitsViewV1) -> HostServiceResultV1<()> {
        assert_eq!(view.cycle_epoch, self.epoch);
        self.host.device.note("ready-sample".to_owned());
        self.ready.push(view);
        Ok(())
    }
}
fn create_ready(host: &Rc<Host>, items: u32) -> BundleInstanceV1 {
    crate::device_results::with_override(None, || {
        let record = crate::config::tests::record(&host.shape, host.kv, items, false);
        crate::instance::PARTIAL.with(|partial| partial.set(true));
        let result = api()
            .with_device_results_api(crate::api::BUNDLE_DEEPSEEK41_DEVICE_RESULTS_API_V1)
            .unwrap()
            .create(HostServicesTableV1::new(host.clone()), &record);
        crate::instance::PARTIAL.with(|partial| partial.set(false));
        result.unwrap()
    })
}
#[test]
fn default_ar_readiness_precedes_status_waits_and_releases_every_event() {
    for count in [1, 8, 32] {
        let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
        let mut instance = create_ready(&host, count);
        for slot in 0..count {
            step(
                &mut instance,
                &prompt(u64::from(slot) + 1, u64::from(slot), 0, 3, true),
                &mut Views::default(),
            );
        }
        let ids = (1..=u64::from(count)).collect::<Vec<_>>();
        let request = decode(
            &ids.iter()
                .enumerate()
                .map(|(slot, &id)| (id, slot as u64, 3, 1))
                .collect::<Vec<_>>(),
        );
        let mut sink = ReadySink {
            host: host.clone(),
            epoch: 9,
            ids,
            ready: Vec::new(),
            legacy: Vec::new(),
        };
        let from = host.device.journal.borrow().len();
        instance.execute(&request, &mut sink).unwrap();
        assert!(sink.legacy.is_empty());
        assert_eq!(
            sink.ready
                .iter()
                .map(|view| view.logits.row_count)
                .sum::<u32>(),
            count
        );
        let journal = host.device.journal.borrow();
        let lines = &journal[from..];
        let sample = lines
            .iter()
            .position(|line| line == "ready-sample")
            .unwrap();
        let wait = lines
            .iter()
            .position(|line| line.starts_with("synchronize "))
            .unwrap();
        assert!(
            sample < wait,
            "sampling must be submitted before waiting on status"
        );
        drop(journal);
        instance.commit().unwrap();
        for id in &sink.ids {
            instance.retire_request(*id).unwrap();
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}
#[test]
fn invalid_status_after_ready_submission_still_refuses_the_step() {
    let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
    let mut instance = create_ready(&host, 4);
    step(
        &mut instance,
        &prompt(7, 0, 0, 3, true),
        &mut Views::default(),
    );
    host.injected.replace(Some(1_u32.to_le_bytes().to_vec()));
    let mut sink = ReadySink {
        host: host.clone(),
        epoch: 9,
        ids: vec![7],
        ready: Vec::new(),
        legacy: Vec::new(),
    };
    assert!(
        instance
            .execute(&decode(&[(7, 0, 3, 1)]), &mut sink)
            .is_err()
    );
    assert!(!sink.ready.is_empty());
    assert!(instance.commit().is_err());
    assert!(host.injected.borrow().is_none());
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

#[test]
fn zero_device_results_restores_the_legacy_journals() {
    crate::device_results::with_override(Some("0"), || {
        super::every_geometry_topology_runs_the_fake_host_journal_and_lifecycle();
    });
}

#[test]
fn default_device_results_only_select_qualified_topologies_and_kv() {
    for shape in crate::shape::topologies() {
        for kv in [KvStorage::Native, KvStorage::Bf16] {
            let expected =
                shape.geometry.id == FLASH.id && shape.ranks() == 4 && kv == KvStorage::Native;
            assert_eq!(
                crate::device_results::with_override(None, || crate::device_results::enabled(
                    shape, kv
                )),
                expected
            );
            assert!(!crate::device_results::with_override(Some("0"), || {
                crate::device_results::enabled(shape, kv)
            }));
            assert!(crate::device_results::with_override(Some("1"), || {
                crate::device_results::enabled(shape, kv)
            }));
        }
    }
}

/// With device results, AR decode's per-layer Engram uploads complete after
/// the ready sample is submitted and before the step can commit; every
/// stream launches what the synchronous route launches, and the next step
/// uploads again.
#[test]
fn ready_ar_steps_complete_their_per_layer_engram_uploads() {
    use super::ar_engram::{launches, order};
    let count = 8_u32;
    let run = |on: bool| {
        let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
        host.drafter.set(true);
        let mut instance = crate::device_results::with_override(None, || {
            crate::instance::ar_engram::with_enabled(on, || {
                let record = crate::config::tests::record(&host.shape, host.kv, count, false);
                api()
                    .with_device_results_api(crate::api::BUNDLE_DEEPSEEK41_DEVICE_RESULTS_API_V1)
                    .unwrap()
                    .create(
                        HostServicesTableV1::new(Rc::new(super::dspark::host::MappedHost::new(
                            Rc::clone(&host),
                        ))),
                        &record,
                    )
                    .unwrap()
            })
        });
        for slot in 0..count {
            step(
                &mut instance,
                &prompt(u64::from(slot) + 1, u64::from(slot), 0, 3, true),
                &mut Views::default(),
            );
        }
        let ids = (1..=u64::from(count)).collect::<Vec<_>>();
        let from = host.device.journal.borrow().len();
        let mut steps = Vec::new();
        for (epoch, position) in [(9, 3), (10, 4)] {
            let request = decode(
                &ids.iter()
                    .enumerate()
                    .map(|(slot, &id)| (id, slot as u64, position, 1 + slot as u32))
                    .collect::<Vec<_>>(),
            );
            let mut sink = ReadySink {
                host: host.clone(),
                epoch,
                ids: ids.clone(),
                ready: Vec::new(),
                legacy: Vec::new(),
            };
            let begin = host.device.journal.borrow().len();
            instance.execute(&request, &mut sink).unwrap();
            assert!(sink.legacy.is_empty());
            assert_eq!(
                sink.ready
                    .iter()
                    .map(|view| view.logits.row_count)
                    .sum::<u32>(),
                count
            );
            steps.push(host.device.journal.borrow()[begin..].to_vec());
            instance.commit().unwrap();
        }
        let launched = launches(&host, from);
        for id in &ids {
            instance.retire_request(*id).unwrap();
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
        (launched, steps)
    };
    let (off, off_steps) = run(false);
    let (on, on_steps) = run(true);
    assert_eq!(off, on);
    for (off, on) in off_steps.iter().zip(&on_steps) {
        assert!(order(off).contains(&"sync") && !order(off).contains(&"ids"));
        let kinds = order(on);
        assert!(!kinds.contains(&"sync"));
        assert_eq!(kinds.iter().filter(|&&k| k == "ids").count(), 2);
        assert_eq!(kinds.iter().filter(|&&k| k == "wait").count(), 2 * 4);
        assert_eq!(kinds.iter().filter(|&&k| k == "complete").count(), 2);
        let sample = on.iter().position(|line| line == "ready-sample").unwrap();
        let complete = on
            .iter()
            .position(|line| line.starts_with("async mapped complete"))
            .unwrap();
        assert!(sample < complete, "uploads complete after the ready sample");
    }
}
