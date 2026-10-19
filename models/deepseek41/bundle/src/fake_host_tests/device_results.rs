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
    crate::device_results::with_enabled(|| {
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
fn ar_readiness_precedes_status_waits_and_releases_every_event() {
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
