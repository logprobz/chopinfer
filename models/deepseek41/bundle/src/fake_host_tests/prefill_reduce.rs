use super::*;
use crate::instance::execute::overlap::with_mode;
use crate::ops::with_bf16_gather;
const ROUND_SYMBOL: &str = "chop_comm_allreduce_phased_round_f32_v1";

fn report() -> BundleRawClusterResidencyV1 {
    BundleRawClusterResidencyV1 {
        multiprocessor_count: 188,
        usable_multiprocessors: 188,
        max_active_clusters: 188,
        flags: BUNDLE_CLUSTER_RESIDENCY_V1_CLUSTERS_REPORTED
            | BUNDLE_CLUSTER_RESIDENCY_V1_COOPERATIVE_LAUNCH,
        ..BundleRawClusterResidencyV1::unreported()
    }
}
fn host() -> Rc<Host> {
    let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
    host.device.residency.set(Some(report()));
    host
}
fn count(journal: &[String], symbol: &str) -> usize {
    journal
        .iter()
        .filter(|s| s.starts_with("launch ") && s.contains(symbol))
        .count()
}

#[test]
fn prefill_bf16_gather_routes_every_hidden_reduction_after_admission() {
    for (rows, expected) in [(511, 0), (512, 81), (1057, 81), (8191, 81), (8192, 81)] {
        let host = host();
        let mut instance = with_bf16_gather(true, || create(&host, 4));
        assert_eq!(host.device.residency_queries.borrow().len(), 8);
        host.device.journal.borrow_mut().clear();
        step(
            &mut instance,
            &prompt(7, 0, 0, rows, true),
            &mut Views::default(),
        );
        let journal = host.device.journal.borrow();
        assert_eq!(count(&journal, ROUND_SYMBOL), 4 * expected);
        assert_eq!(
            count(&journal, "chop_comm_allgather_scattered_f32_v1"),
            4 * (81 - expected)
        );
        drop(journal);
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

#[test]
fn prefill_bf16_gather_keeps_decode_on_its_original_transport() {
    let mut traces = Vec::new();
    for enabled in [false, true] {
        let host = host();
        let mut instance = with_bf16_gather(enabled, || create(&host, 4));
        step(
            &mut instance,
            &prompt(7, 0, 0, 1024, true),
            &mut Views::default(),
        );
        host.device.journal.borrow_mut().clear();
        step(
            &mut instance,
            &decode(&[(7, 0, 1024, 9)]),
            &mut Views::default(),
        );
        let journal = host.device.journal.borrow();
        assert_eq!(count(&journal, ROUND_SYMBOL), 0);
        assert_eq!(
            count(&journal, "chop_comm_allgather_scattered_f32_v1"),
            4 * 81
        );
        traces.push(
            journal
                .iter()
                .filter(|s| s.starts_with("launch "))
                .map(|s| s.split_whitespace().nth(2).unwrap().to_owned())
                .collect::<Vec<_>>(),
        );
        drop(journal);
        instance.destroy().unwrap();
    }
    assert_eq!(traces[0], traces[1]);
}

#[test]
fn prefill_bf16_gather_keeps_verify_and_drafter_transports() {
    let mut traces = Vec::new();
    for enabled in [false, true] {
        let host = host();
        host.speculation.set(Some((3, 16)));
        let mut instance = with_bf16_gather(enabled, || create(&host, 4));
        step(
            &mut instance,
            &prompt(7, 0, 0, 1024, true),
            &mut Views::default(),
        );
        host.device.journal.borrow_mut().clear();
        instance
            .execute(
                &verify(&[(7, 0, 1024, &[7, 8, 9, 10])]),
                &mut Views::default(),
            )
            .unwrap();
        instance.commit_accepted(&[kept(7, 4)]).unwrap();
        let journal = host.device.journal.borrow();
        assert_eq!(count(&journal, ROUND_SYMBOL), 0);
        assert!(count(&journal, "chop_comm_allgather_scattered_f32_v1") >= 4 * 81);
        traces.push(
            journal
                .iter()
                .filter(|s| s.starts_with("launch "))
                .map(|s| s.split_whitespace().nth(2).unwrap().to_owned())
                .collect::<Vec<_>>(),
        );
        drop(journal);
        instance.destroy().unwrap();
    }
    assert_eq!(traces[0], traces[1]);
}

#[test]
fn prefill_bf16_gather_falls_back_on_unproven_residency() {
    for kind in 0..4 {
        let host = host();
        let mut value = report();
        match kind {
            0 => value = BundleRawClusterResidencyV1::unreported(),
            1 => value.max_active_clusters = 0,
            2 => value.flags |= BUNDLE_CLUSTER_RESIDENCY_V1_MPS_CLIENT,
            _ => value.usable_multiprocessors = 1,
        }
        host.device.residency.set(Some(value));
        let mut instance = with_bf16_gather(true, || create(&host, 4));
        host.device.journal.borrow_mut().clear();
        step(
            &mut instance,
            &prompt(7, 0, 0, 1024, true),
            &mut Views::default(),
        );
        let journal = host.device.journal.borrow();
        assert_eq!(count(&journal, ROUND_SYMBOL), 0);
        assert_eq!(
            count(&journal, "chop_comm_allgather_scattered_f32_v1"),
            4 * 81
        );
        drop(journal);
        instance.destroy().unwrap();
    }
}

#[test]
fn prefill_bf16_gather_micro_journal_keeps_all_cross_stream_spans_disjoint() {
    let host = host();
    let mut instance = with_bf16_gather(true, || with_mode(2, || create(&host, 4)));
    host.device.journal.borrow_mut().clear();
    step(
        &mut instance,
        &prompt(7, 0, 0, 8191, true),
        &mut Views::default(),
    );
    let journal = host.device.journal.borrow().clone();
    assert_eq!(count(&journal, ROUND_SYMBOL), 4 * 162);
    super::prefill_micro::audit_journal(&host, &journal, ROUND_SYMBOL);
    instance.destroy().unwrap();
}

#[test]
fn prefill_bf16_gather_resets_only_its_protocol_after_join() {
    use bundle_runtime::entry::Bundle;
    let host = host();
    let mut instance = with_bf16_gather(true, || with_mode(2, || create(&host, 4)));
    step(
        &mut instance,
        &prompt(7, 0, 0, 1024, false),
        &mut Views::default(),
    );
    let spans = <crate::instance::Instance as Bundle>::instances().with(|table| {
        table
            .borrow_mut()
            .get_mut(instance.handle())
            .unwrap()
            .reset_prefill_reduce_next_run()
    });
    assert_eq!(spans.len(), 12);
    host.device.journal.borrow_mut().clear();
    step(
        &mut instance,
        &prompt(7, 0, 1024, 1024, true),
        &mut Views::default(),
    );
    let journal = host.device.journal.borrow();
    let first_launch = journal
        .iter()
        .position(|s| s.starts_with("launch "))
        .unwrap();
    let reset = &journal[..first_launch];
    let first_zero = reset.iter().position(|s| s.starts_with("zero ")).unwrap();
    assert_eq!(
        reset[..first_zero]
            .iter()
            .filter(|s| s.starts_with("synchronize "))
            .count(),
        4
    );
    let actual = reset[first_zero..first_zero + spans.len()]
        .iter()
        .flat_map(|s| super::shared_overlap::journal_spans(s))
        .collect::<Vec<_>>();
    assert_eq!(
        actual,
        spans
            .iter()
            .map(|span| (span.buffer.raw, span.offset, span.length))
            .collect::<Vec<_>>()
    );
    assert!(
        reset[first_zero + spans.len()..first_zero + spans.len() + 4]
            .iter()
            .all(|s| s.starts_with("synchronize "))
    );
    drop(journal);
    instance.destroy().unwrap();
}

#[test]
fn prefill_bf16_gather_partial_rank_failure_poisoning_is_terminal() {
    let reference = host();
    let mut instance = with_bf16_gather(true, || with_mode(2, || create(&reference, 4)));
    reference.device.journal.borrow_mut().clear();
    step(
        &mut instance,
        &prompt(7, 0, 0, 1024, true),
        &mut Views::default(),
    );
    let lines = reference.device.journal.borrow().clone();
    instance.destroy().unwrap();
    let at = lines
        .iter()
        .position(|s| s.starts_with("launch ") && s.contains(ROUND_SYMBOL))
        .unwrap();
    let before = lines[..at]
        .iter()
        .filter(|s| s.starts_with("launch "))
        .count();
    let host = host();
    let mut instance = with_bf16_gather(true, || with_mode(2, || create(&host, 4)));
    host.device.fail_launch.set(Some(before as u32 + 1));
    assert_eq!(
        instance.execute(&prompt(7, 0, 0, 1024, true), &mut Views::default()),
        Err(BundleAbiError::Status(BundleStatusV1::HostError))
    );
    let submitted = host.device.journal.borrow().clone();
    assert_eq!(
        instance.capture_prefix_state(7, 1024, &[200]),
        Err(BundleAbiError::Status(BundleStatusV1::InvalidState))
    );
    assert_eq!(
        instance.destroy(),
        Err(BundleAbiError::Status(BundleStatusV1::InvalidState))
    );
    assert_eq!(*host.device.journal.borrow(), submitted);
}
