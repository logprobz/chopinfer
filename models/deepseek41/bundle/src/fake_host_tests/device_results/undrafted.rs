use super::*;
use crate::fake_host_tests::dspark::host::MappedHost;

mod capacity;
#[cfg(feature = "state-capture")]
mod capture;

fn create(count: u32) -> (Rc<Host>, BundleInstanceV1) {
    create_with_sampling(count, false)
}

fn create_with_sampling(count: u32, sampled: bool) -> (Rc<Host>, BundleInstanceV1) {
    let (host, instance, _) = create_with_mapping(count, sampled);
    (host, instance)
}

fn create_with_mapping(count: u32, sampled: bool) -> (Rc<Host>, BundleInstanceV1, Rc<MappedHost>) {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    host.drafter.set(true);
    host.speculation.set(Some((5, 32)));
    let mapped = Rc::new(MappedHost::with_delayed_completion(Rc::clone(&host)));
    let record = crate::config::tests::with_coupled_drafts(
        crate::config::tests::record(&shape, host.kv, count, false),
        sampled,
    );
    let record = crate::config::tests::with_option(
        record,
        model_deepseek41_contract::execution_options::DIAGNOSTIC_STEP_LOGS,
        true.into(),
    );
    let instance = crate::device_results::with_override(None, || {
        api()
            .with_device_results_api(crate::api::BUNDLE_DEEPSEEK41_DEVICE_RESULTS_API_V1)
            .unwrap()
            .create(HostServicesTableV1::new(Rc::clone(&mapped)), &record)
            .unwrap()
    });
    (host, instance, mapped)
}

#[test]
fn failed_undrafted_gather_completion_refuses_private_ready_results() {
    let (host, mut instance, mapped) = create_with_mapping(1, false);
    prefill(&host, &mut instance, 1);
    mapped.fail_gather_after("complete", 0);
    let mut result = sink(&host, 1, 9);
    assert!(
        instance
            .execute(&decode(&[(1, 0, 3, 17)]), &mut result)
            .is_err()
    );
    assert!(!result.ready.is_empty());
    assert!(result.legacy.is_empty());
    assert!(
        host.journal
            .borrow()
            .iter()
            .any(|line| line == "async mapped failure complete")
    );
    assert_eq!(
        crate::instance::dspark_context(instance.handle(), 1)
            .unwrap()
            .end,
        3
    );
    assert!(instance.commit().is_err());
    let _ = instance.destroy();
}

fn sink(host: &Rc<Host>, count: u32, epoch: u64) -> ReadySink {
    ReadySink {
        host: Rc::clone(host),
        epoch,
        ids: (1..=u64::from(count)).collect(),
        ready: Vec::new(),
        legacy: Vec::new(),
    }
}

fn prefill(host: &Host, instance: &mut BundleInstanceV1, count: u32) {
    for slot in 0..count {
        let mut sink = Views::default();
        let from = host.journal.borrow().len();
        instance
            .execute(
                &prompt(u64::from(slot) + 1, u64::from(slot), 0, 3, true),
                &mut sink,
            )
            .unwrap_or_else(|error| panic!("prefill: {error:?}, {:?}", host.logs.borrow()));
        assert_eq!(
            host.journal.borrow()[from..]
                .iter()
                .filter(|line| line.starts_with("download "))
                .count(),
            4
        );
        instance.commit().unwrap();
    }
}

#[test]
fn undrafted_ready_steps_submit_sampling_before_joined_status() {
    for count in [1, 16, 32] {
        let (host, mut instance) = create(count);
        prefill(&host, &mut instance, count);
        for position in 3..7 {
            let request = decode(
                &(0..count)
                    .map(|slot| (u64::from(slot) + 1, u64::from(slot), position, slot + 17))
                    .collect::<Vec<_>>(),
            );
            let mut sink = sink(&host, count, position + 10);
            let from = host.journal.borrow().len();
            instance
                .execute(&request, &mut sink)
                .unwrap_or_else(|error| panic!("decode: {error:?}, {:?}", host.logs.borrow()));
            let lines = host.journal.borrow()[from..].to_vec();
            let record = host
                .logs
                .borrow()
                .iter()
                .rev()
                .find_map(|line| {
                    serde_json::from_str::<serde_json::Value>(line)
                        .ok()
                        .filter(|value| value["schema"] == "chopinfer.deepseek41.step-shape.v1")
                })
                .unwrap();
            assert_eq!(record["undrafted_ready"], 1);
            assert_eq!(record["undrafted_status_packets"], 1);
            let downloads = lines
                .iter()
                .filter(|line| line.starts_with("download "))
                .count();
            assert_eq!(downloads, 1);

            assert!(sink.legacy.is_empty());
            assert_eq!(
                sink.ready
                    .iter()
                    .map(|view| view.logits.row_count)
                    .sum::<u32>(),
                count
            );
            let first_read = lines
                .iter()
                .position(|line| line.starts_with("download "))
                .unwrap();
            let last_sample = lines
                .iter()
                .rposition(|line| line == "ready-sample")
                .unwrap();
            assert!(last_sample < first_read);

            assert_eq!(
                lines
                    .iter()
                    .filter(|line| line.starts_with("replay "))
                    .count(),
                12
            );
            instance.commit().unwrap();
            for id in &sink.ids {
                let context = crate::instance::dspark_context(instance.handle(), *id).unwrap();
                assert_eq!(context.end, position + 1);
            }
        }
        for id in 1..=u64::from(count) {
            instance.retire_request(id).unwrap();
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

#[test]
fn undrafted_joined_status_validates_before_legacy_emission() {
    for invalid_status in [false, true] {
        let (host, mut instance) = create(1);
        prefill(&host, &mut instance, 1);
        if invalid_status {
            let mut packet = vec![0; FLASH.shape.csa.index_sources.len() * 4 + 4 * 8];
            packet[0] = 1;
            host.injected.replace(Some(packet));
        }
        let mut views = Views::default();
        let from = host.journal.borrow().len();
        let result = instance.execute(&decode(&[(1, 0, 3, 17)]), &mut views);
        assert_eq!(result.is_err(), invalid_status);
        assert_eq!(views.rows(), u32::from(!invalid_status));
        assert_eq!(
            host.journal.borrow()[from..]
                .iter()
                .filter(|line| line.starts_with("download "))
                .count(),
            1
        );
        assert!(
            !host.journal.borrow()[from..]
                .iter()
                .any(|line| line == "ready-sample")
        );
        if invalid_status {
            assert!(host.injected.borrow().is_none());
            assert!(instance.commit().is_err());
            assert_eq!(
                crate::instance::dspark_context(instance.handle(), 1)
                    .unwrap()
                    .end,
                3
            );
        } else {
            instance.commit().unwrap();
            assert_eq!(
                crate::instance::dspark_context(instance.handle(), 1)
                    .unwrap()
                    .end,
                4
            );
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

#[test]
fn undrafted_status_errors_refuse_commit_after_private_ready_submission() {
    let index_bytes = FLASH.shape.csa.index_sources.len() * 4;
    let bytes = index_bytes + 4 * 8;
    for offset in std::iter::once(0).chain((0..4).map(|rank| index_bytes + rank * 8)) {
        let (host, mut instance) = create(1);
        prefill(&host, &mut instance, 1);
        let mut invalid = vec![0; bytes];
        invalid[offset] = 1;
        host.injected.replace(Some(invalid));
        let mut sink = sink(&host, 1, 7);
        assert!(
            instance
                .execute(&decode(&[(1, 0, 3, 17)]), &mut sink)
                .is_err()
        );
        assert!(host.injected.borrow().is_none());
        assert!(!sink.ready.is_empty());
        assert!(sink.legacy.is_empty());
        assert_eq!(
            crate::instance::dspark_context(instance.handle(), 1)
                .unwrap()
                .end,
            3
        );
        assert!(instance.commit().is_err());
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

#[test]
fn undrafted_small_status_packet_keeps_rank_validation() {
    let (host, mut instance) = create(1);
    prefill(&host, &mut instance, 1);
    crate::instance::dspark_boundary_for_test(instance.handle(), Some(4));
    let mut sink = sink(&host, 1, 7);
    let from = host.journal.borrow().len();
    instance
        .execute(&decode(&[(1, 0, 3, 17)]), &mut sink)
        .unwrap();
    assert_eq!(
        host.journal.borrow()[from..]
            .iter()
            .filter(|line| line.starts_with("download "))
            .count(),
        4
    );
    assert!(!sink.ready.is_empty());
    instance.commit().unwrap();
    instance.retire_request(1).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}
