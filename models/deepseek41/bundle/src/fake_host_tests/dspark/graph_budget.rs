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
