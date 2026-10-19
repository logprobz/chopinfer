use super::*;
use crate::shared_overlap::{Settings, with_settings_in_tests};

fn captured(settings: Settings, ranks: Ranks) -> Vec<String> {
    let shape = Shape::new(&FLASH, ranks).unwrap();
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    let mut instance = with_settings_in_tests(settings, || create(&host, 4));
    step(
        &mut instance,
        &prompt(7, 0, 0, 3, true),
        &mut Views::default(),
    );
    let from = host.device.journal.borrow().len();
    step(
        &mut instance,
        &decode(&[(7, 0, 3, 9)]),
        &mut Views::default(),
    );
    let lines = host.device.journal.borrow()[from..].to_vec();
    let count = lines
        .iter()
        .filter(|line| line.starts_with("launch "))
        .count();
    assert!(count > 100);
    let from = host.device.journal.borrow().len();
    step(
        &mut instance,
        &decode(&[(7, 0, 4, 9)]),
        &mut Views::default(),
    );
    assert!(
        !host.device.journal.borrow()[from..]
            .iter()
            .any(|s| s.starts_with("capture "))
    );
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    lines
}

#[test]
fn shared_overlap_captures_identical_launches_and_reuses_the_graph() {
    for ranks in [Ranks::TP2, Ranks::TP4] {
        let serial = captured(Settings::OFF, ranks);
        let parallel = captured(Settings::ALL, ranks);
        let launches = |lines: &[String]| {
            let mut values = lines
                .iter()
                .filter_map(|s| s.strip_prefix("launch "))
                .map(|s| s.split_once(' ').unwrap().1.to_owned())
                .collect::<Vec<_>>();
            values.sort();
            values
        };
        assert_eq!(launches(&serial), launches(&parallel));
        let waits = |lines: &[String]| lines.iter().filter(|s| s.starts_with("wait ")).count();
        assert!(waits(&parallel) > waits(&serial));
        let streams = |lines: &[String]| {
            lines
                .iter()
                .filter_map(|s| s.strip_prefix("launch "))
                .map(|s| s.split_once(' ').unwrap().0.to_owned())
                .collect::<std::collections::BTreeSet<_>>()
        };
        let count = if ranks == Ranks::TP2 { 2 } else { 4 };
        assert_eq!(streams(&serial).len(), count);
        assert_eq!(streams(&parallel).len(), count * 2);
        assert_eq!(
            serial
                .iter()
                .any(|s| s.starts_with("launch ") && s.contains("_stream_merge_v1")),
            ranks == Ranks::TP4
        );
    }
}

#[test]
fn shared_overlap_closes_captures_after_fork_or_join_failure() {
    for fault in 0..4 {
        let shape = Shape::new(&FLASH, Ranks::TP2).unwrap();
        let host = FakeHost::new(shape, KV_LAYOUTS[0]);
        let mut instance = create(&host, 4);
        step(
            &mut instance,
            &prompt(7, 0, 0, 3, true),
            &mut Views::default(),
        );
        if fault < 2 {
            host.device.fail_record.set(Some(fault));
        } else {
            host.device.fail_wait.set(Some(fault - 2));
        }
        let mut views = Views::default();
        assert!(
            instance
                .execute(&decode(&[(7, 0, 3, 9)]), &mut views)
                .is_err()
        );
        assert_eq!(views.rows(), 0);
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

#[test]
fn shared_overlap_replay_failure_poisoning_prevents_a_second_submission() {
    let shape = Shape::new(&FLASH, Ranks::TP2).unwrap();
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    let mut instance = create(&host, 4);
    step(
        &mut instance,
        &prompt(7, 0, 0, 3, true),
        &mut Views::default(),
    );
    step(
        &mut instance,
        &decode(&[(7, 0, 3, 9)]),
        &mut Views::default(),
    );
    host.device.fail_graph_launch.set(Some(1));
    let batch = decode(&[(7, 0, 4, 9)]);
    assert!(instance.execute(&batch, &mut Views::default()).is_err());
    let before = host.device.journal.borrow().len();
    assert!(instance.execute(&batch, &mut Views::default()).is_err());
    assert_eq!(host.device.journal.borrow().len(), before);
    let released = host.released.get();
    assert!(instance.destroy().is_err());
    assert!(instance.is_poisoned());
    // A rank may still be blocked in a collective. Its resources stay held
    // rather than being released underneath the queued work.
    assert_eq!(host.released.get(), released);
    assert_eq!(host.device.journal.borrow().len(), before);
}

#[test]
fn shared_overlap_refuses_invalid_quantized_decode_rows() {
    for settings in [Settings::OFF, Settings::ALL] {
        let shape = Shape::new(&FLASH, Ranks::TP2).unwrap();
        let host = FakeHost::new(shape, KV_LAYOUTS[0]);
        let mut instance = with_settings_in_tests(settings, || create(&host, 4));
        step(
            &mut instance,
            &prompt(7, 0, 0, 3, true),
            &mut Views::default(),
        );
        host.injected.replace(Some(vec![1, 0, 0, 0]));
        host.injected_after
            .set(FLASH.shape.csa.index_sources.len() + 1);
        let mut views = Views::default();
        assert_eq!(
            instance
                .execute(&decode(&[(7, 0, 3, 9)]), &mut views)
                .unwrap_err(),
            BundleAbiError::Status(BundleStatusV1::BundleError)
        );
        assert_eq!(views.rows(), 0);
        assert!(host.injected.borrow().is_none());
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}
