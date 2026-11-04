use super::*;

fn host() -> Rc<Host> {
    // BF16 KV takes the native prefill partials and their protocol state,
    // on the residency the fake device gives Flash TP4 by default.
    FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[1])
}

/// A pending prefill reset at BF16 KV drains every rank, zeroes exactly
/// the prefill partials' protocol state and drains again before the next
/// prefill's first launch.
#[test]
fn prefill_partials_reset_only_their_protocol_after_join_at_bf16_kv() {
    use bundle_runtime::entry::Bundle;
    let host = host();
    let mut instance = create(&host, 4);
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

/// A prefill whose BF16-partial collective one rank has submitted and the
/// next rank's launches fail to reach poisons the instance at both KV
/// layouts: capture and destroy refuse it and submit nothing more.
#[test]
fn prefill_partials_partial_rank_failure_poisoning_is_terminal() {
    const PARTIALS: &str = "chop_comm_allreduce_phased_";
    let partial =
        |s: &String| s.starts_with("launch ") && s.contains(PARTIALS) && s.contains("bf16_v1");
    for kv in KV_LAYOUTS {
        let reference = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), kv);
        let mut instance = create(&reference, 4);
        reference.device.journal.borrow_mut().clear();
        step(
            &mut instance,
            &prompt(7, 0, 0, 1024, true),
            &mut Views::default(),
        );
        let lines = reference.device.journal.borrow().clone();
        instance.destroy().unwrap();
        let at = lines.iter().position(partial).unwrap();
        let stream = |s: &str| s.split_whitespace().nth(1).map(str::to_owned);
        // Its peers' parts of the same collective follow on other streams.
        let next = lines[at + 1..].iter().position(partial).unwrap() + at + 1;
        assert_ne!(stream(&lines[at]), stream(&lines[next]), "{:?}", kv.storage);
        let before = lines[..at]
            .iter()
            .filter(|s| s.starts_with("launch "))
            .count();
        let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), kv);
        let mut instance = create(&host, 4);
        host.device.fail_launch.set(Some(before as u32 + 1));
        assert_eq!(
            instance.execute(&prompt(7, 0, 0, 1024, true), &mut Views::default()),
            Err(BundleAbiError::Status(BundleStatusV1::HostError)),
            "{:?}",
            kv.storage
        );
        let submitted = host.device.journal.borrow().clone();
        assert!(submitted.iter().any(partial), "{:?}", kv.storage);
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
}
