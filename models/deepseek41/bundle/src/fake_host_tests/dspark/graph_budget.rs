//! Graph pressure preserves the same lifecycle and mapped gather handles.
use super::*;

#[test]
fn exact_width_graph_pressure_retires_before_recapture_on_every_topology() {
    for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
        let (host, mut instance) = create(Shape::new(&FLASH, ranks).unwrap());
        let capacity = 8 * ranks.count();
        crate::instance::dspark_graphs_for_test(instance.handle(), Some(capacity));
        execute(
            &mut instance,
            &host,
            &super::super::prompt(7, 0, 0, 3, true),
        );
        instance.commit().unwrap();
        for rows in [3, 4, 5, 6, 3, 4, 5, 6] {
            let start = state(&instance, 7).end;
            let begin = host.journal.borrow().len();
            execute(&mut instance, &host, &verify(7, 0, start, rows - 1));
            instance.commit_accepted(&[kept(7, 1)]).unwrap();
            assert_eq!(state(&instance, 7).end, start + 1);
            let stats = crate::instance::dspark_graphs_for_test(instance.handle(), None);
            assert!(stats.live <= capacity && stats.peak <= capacity);
            assert_eq!(stats.reserved, 0);
            let journal = host.journal.borrow();
            let step = &journal[begin..];
            assert!(!step.iter().any(|s| s.starts_with("async mapped release")));
            if let Some(released) = step.iter().position(|s| s.starts_with("release graph")) {
                assert!(
                    step[..released]
                        .iter()
                        .filter(|s| s.starts_with("synchronize "))
                        .count()
                        >= ranks.count()
                );
                let capture = step.iter().position(|s| s.starts_with("capture ")).unwrap();
                assert!(released < capture);
            }
        }
        let stats = crate::instance::dspark_graphs_for_test(instance.handle(), None);
        assert!(stats.denied > 0 && stats.released > 0 && stats.epoch > 0);
        let logs = host.logs.borrow();
        assert!(
            logs.iter()
                .any(|s| s.contains("\"event\":\"eviction-begin\""))
        );
        if let Some(root) = std::env::var_os("CHOPINFER_DSPARK_GRAPH_LOGS") {
            let root = std::path::PathBuf::from(root);
            std::fs::create_dir_all(&root).unwrap();
            std::fs::write(
                root.join(format!("tp{}.jsonl", ranks.get())),
                logs.join("\n"),
            )
            .unwrap();
        }
        drop(logs);
        instance.retire_request(7).unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

#[test]
fn zero_graph_capacity_keeps_the_full_draft_and_target_eager() {
    let (host, mut instance) = create(Shape::new(&FLASH, Ranks::TP4).unwrap());
    crate::instance::dspark_graphs_for_test(instance.handle(), Some(0));
    execute(
        &mut instance,
        &host,
        &super::super::prompt(7, 0, 0, 3, true),
    );
    instance.commit().unwrap();
    for accepted in 0..=5 {
        let start = state(&instance, 7).end;
        let sink = execute(&mut instance, &host, &verify(7, 0, start, 5));
        assert_eq!(sink.draft_tokens(), [(7, vec![0; 5])]);
        instance.commit_accepted(&[kept(7, accepted + 1)]).unwrap();
        assert_eq!(state(&instance, 7).end, start + u64::from(accepted + 1));
        assert_eq!(
            crate::instance::dspark_graphs_for_test(instance.handle(), None).live,
            0
        );
    }
    assert!(
        !host
            .journal
            .borrow()
            .iter()
            .any(|s| s.starts_with("capture "))
    );
    instance.retire_request(7).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

#[test]
fn verify32_after_full_prefill_reserves_the_selected_context_bank() {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    host.kv_pages.set(8 * 1024);
    host.drafter.set(true);
    host.speculation.set(Some((3, 32)));
    let record = crate::config::tests::record(&shape, host.kv, 8, false);
    let mut instance = api()
        .create(
            HostServicesTableV1::new(Rc::new(MappedHost::new(Rc::clone(&host)))),
            &record,
        )
        .unwrap();
    let batch = |start: u64| {
        BatchRequestV1::new(
            BatchWireVersionV1::V3,
            (0..8)
                .map(|slot| BatchItemV1::verify(slot + 1, 4, true, vec![7]))
                .collect(),
            (0..8)
                .map(|slot| {
                    BatchAppendV1::new(
                        start,
                        4,
                        start + 4,
                        (0..(start + 4).div_ceil(64))
                            .map(|page| slot * 1024 + page)
                            .collect(),
                        None,
                    )
                })
                .collect(),
        )
        .with_state_slots((0..8).collect())
        .encode()
        .unwrap()
    };
    // Warmup captures the short-context Verify32 bank before real requests.
    for slot in 0..8 {
        execute(
            &mut instance,
            &host,
            &super::super::prompt(slot + 1, slot, 0, 3, true),
        );
        instance.commit().unwrap();
    }
    execute(&mut instance, &host, &batch(3));
    instance
        .commit_accepted(&(1..=8).map(|id| kept(id, 1)).collect::<Vec<_>>())
        .unwrap();
    for id in 1..=8 {
        instance.retire_request(id).unwrap();
    }
    // Four full 8192-row chunks per request select the long-context bank.
    // Use disjoint pages even though fake kernels do not compute their values.
    for slot in 0..8 {
        for chunk in 0..4 {
            let start = chunk * 8192;
            let end = start + 8192;
            let request = BatchRequestV1::new(
                BatchWireVersionV1::V3,
                vec![BatchItemV1::new(
                    slot + 1,
                    false,
                    8192,
                    chunk == 3,
                    vec![7; 8192],
                )],
                vec![BatchAppendV1::new(
                    start,
                    8192,
                    end,
                    (0..end.div_ceil(64))
                        .map(|page| slot * 1024 + page)
                        .collect(),
                    None,
                )],
            )
            .with_state_slots(vec![slot])
            .encode()
            .unwrap();
            execute(&mut instance, &host, &request);
            instance.commit().unwrap();
        }
    }
    for start in [32768, 32769] {
        let before = crate::instance::dspark_graphs_for_test(instance.handle(), None);
        let begin = host.journal.borrow().len();
        let sink = execute(&mut instance, &host, &batch(start));
        assert_eq!(
            sink.draft_tokens(),
            (1..=8).map(|id| (id, vec![0; 3])).collect::<Vec<_>>()
        );
        assert_eq!(
            sink.views
                .iter()
                .map(|view| view.raw().row_count)
                .sum::<u32>(),
            32
        );
        instance
            .commit_accepted(&(1..=8).map(|id| kept(id, 1)).collect::<Vec<_>>())
            .unwrap();
        let after = crate::instance::dspark_graphs_for_test(instance.handle(), None);
        assert_eq!(after.reserved, 0);
        assert_eq!(after.denied, before.denied);
        // Catch-up can capture its own new row count on the second cycle.
        // Hashing separates those graphs from the target bank under test.
        let journal = host.journal.borrow();
        let step = &journal[begin..];
        let hashed = step
            .iter()
            .position(|line| line.contains(crate::sm120::engram_hash_entries::SYMBOL))
            .unwrap();
        let target = &step[hashed + 1..];
        assert_eq!(
            target
                .iter()
                .filter(|line| line.starts_with("captured "))
                .count(),
            if start == 32768 { 12 } else { 0 }
        );
        assert_eq!(
            target
                .iter()
                .filter(|line| line.starts_with("replay "))
                .count(),
            12
        );
    }
    for id in 1..=8 {
        assert_eq!(state(&instance, id).end, 32770);
        instance.retire_request(id).unwrap();
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}
