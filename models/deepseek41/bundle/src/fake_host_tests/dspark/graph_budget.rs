//! Graph pressure preserves the same lifecycle and mapped gather handles.
use super::*;

#[test]
fn selected_adaptive_warmup_geometries_fit_the_graph_budget_and_replay() {
    let shapes = runtime_bundle_sdk::warmup::verify_shapes(
        32,
        &runtime_bundle_sdk::warmup::VerifyWarmup {
            item_rows: (2..=6).collect(),
            maximum_draft_tokens: 5,
            step_rows: 32,
            drafting_items: None,
            exact_graphs: true,
        },
    );
    assert_eq!(shapes.len(), 97);
    let (host, mut instance) = create_with_limits(Shape::new(&FLASH, Ranks::TP4).unwrap(), 32, 32);
    for slot in 0..32 {
        execute(
            &mut instance,
            &host,
            &super::super::prompt(slot + 1, slot, 0, 3, true),
        );
        instance.commit().unwrap();
    }
    for count in [1, 2, 4, 8, 16, 32] {
        let items: Vec<_> = (0..count)
            .map(|slot| (slot + 1, slot, state(&instance, slot + 1).end, 7))
            .collect();
        execute(&mut instance, &host, &super::super::decode(&items));
        instance.commit().unwrap();
    }
    for repeat in 0..2 {
        for shape in &shapes {
            let mut items = Vec::new();
            let mut appends = Vec::new();
            let mut kept_rows = Vec::new();
            for (slot, &drafts) in shape.drafts.iter().enumerate() {
                let request = slot as u64 + 1;
                let start = state(&instance, request).end;
                let rows = drafts as u32 + 1;
                items.push(if drafts == 0 {
                    BatchItemV1::new(request, true, 1, true, vec![7])
                } else {
                    kept_rows.push(kept(request, 1));
                    BatchItemV1::verify(request, rows, true, vec![7])
                });
                appends.push(BatchAppendV1::new(
                    start,
                    u64::from(rows),
                    start + u64::from(rows),
                    (0..(start + u64::from(rows)).div_ceil(64))
                        .map(|page| slot as u64 * 8 + page)
                        .collect(),
                    None,
                ));
            }
            let batch = BatchRequestV1::new(BatchWireVersionV1::V3, items, appends)
                .with_state_slots((0..shape.item_bucket as u64).collect())
                .encode()
                .unwrap();
            let from = host.logs.borrow().len();
            execute(&mut instance, &host, &batch);
            instance.commit_accepted(&kept_rows).unwrap();
            if repeat != 0 {
                assert!(!host.logs.borrow()[from..].iter().any(|line| {
                    line.contains("\"family\":\"target\"") && line.contains("\"event\":\"capture\"")
                }));
            }
        }
        let stats = crate::instance::dspark_graphs_for_test(instance.handle(), None);
        assert_eq!(stats.epoch, 0, "warmup evicted its earlier shapes");
        assert_eq!(stats.denied, 0);
        assert!(stats.peak <= 512 * 4);
        eprintln!("selected warmup pass {repeat}: {stats:?}");
    }
    for request in 1..=32 {
        instance.retire_request(request).unwrap();
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

/// Target keys (rows, items) a TP4 DSpark startup on round 9 captured or
/// lowered eagerly, in order: the decode waves (zero items), the selected
/// verify shapes and the completion tails of their waves. 131 keys of three
/// pieces on four ranks, plus catch-up and draft graphs, need 1692 handles.
const RECORDED_STARTUP_KEYS: [(u32, u32); 131] = [
    (1, 0),
    (2, 0),
    (4, 0),
    (8, 0),
    (16, 0),
    (32, 0),
    (2, 1),
    (3, 1),
    (4, 1),
    (5, 1),
    (6, 1),
    (3, 2),
    (4, 2),
    (5, 2),
    (6, 2),
    (7, 2),
    (8, 2),
    (9, 2),
    (10, 2),
    (11, 2),
    (12, 2),
    (6, 3),
    (18, 3),
    (12, 3),
    (9, 3),
    (5, 4),
    (6, 4),
    (7, 4),
    (8, 4),
    (9, 4),
    (10, 4),
    (11, 4),
    (12, 4),
    (13, 4),
    (14, 4),
    (15, 4),
    (16, 4),
    (17, 4),
    (18, 4),
    (19, 4),
    (20, 4),
    (21, 4),
    (22, 4),
    (23, 4),
    (24, 4),
    (10, 5),
    (30, 5),
    (20, 5),
    (15, 5),
    (12, 6),
    (32, 6),
    (22, 6),
    (17, 6),
    (14, 7),
    (32, 7),
    (22, 7),
    (17, 7),
    (12, 7),
    (9, 8),
    (10, 8),
    (11, 8),
    (12, 8),
    (13, 8),
    (14, 8),
    (15, 8),
    (16, 8),
    (17, 8),
    (18, 8),
    (19, 8),
    (20, 8),
    (21, 8),
    (22, 8),
    (23, 8),
    (24, 8),
    (25, 8),
    (26, 8),
    (27, 8),
    (28, 8),
    (29, 8),
    (30, 8),
    (31, 8),
    (32, 8),
    (18, 9),
    (32, 9),
    (24, 9),
    (19, 9),
    (14, 9),
    (20, 10),
    (32, 10),
    (24, 10),
    (15, 10),
    (22, 11),
    (32, 11),
    (24, 11),
    (20, 11),
    (16, 11),
    (24, 12),
    (32, 12),
    (20, 12),
    (16, 12),
    (26, 13),
    (32, 13),
    (25, 13),
    (21, 13),
    (17, 13),
    (28, 14),
    (32, 14),
    (26, 14),
    (22, 14),
    (18, 14),
    (30, 15),
    (32, 15),
    (26, 15),
    (23, 15),
    (19, 15),
    (17, 16),
    (18, 16),
    (19, 16),
    (20, 16),
    (21, 16),
    (22, 16),
    (23, 16),
    (24, 16),
    (25, 16),
    (26, 16),
    (27, 16),
    (28, 16),
    (29, 16),
    (30, 16),
    (31, 16),
    (32, 16),
];

/// The recorded startup keys must all be captured: the default budget held
/// 1536 handles, filled at (20, 16) and lowered the 13 widest 16-item
/// verify keys eagerly, which C16 serving reaches.
#[test]
fn recorded_startup_warmup_keys_fit_the_graph_budget() {
    let (host, mut instance) = create_with_limits(Shape::new(&FLASH, Ranks::TP4).unwrap(), 32, 32);
    for slot in 0..32 {
        execute(
            &mut instance,
            &host,
            &super::super::prompt(slot + 1, slot, 0, 3, true),
        );
        instance.commit().unwrap();
    }
    for &(rows, items) in &RECORDED_STARTUP_KEYS {
        if items == 0 {
            let decode: Vec<_> = (0..u64::from(rows))
                .map(|slot| (slot + 1, slot, state(&instance, slot + 1).end, 7))
                .collect();
            execute(&mut instance, &host, &super::super::decode(&decode));
            instance.commit().unwrap();
            continue;
        }
        // Five drafts on the first items, the rest on the next one.
        let mut left = rows - items;
        let drafts: Vec<u32> = (0..items)
            .map(|_| {
                let take = left.min(5);
                left -= take;
                take
            })
            .collect();
        assert_eq!(left, 0, "({rows}, {items}) is unreachable");
        let mut batch_items = Vec::new();
        let mut appends = Vec::new();
        let mut kept_rows = Vec::new();
        for (slot, &depth) in drafts.iter().enumerate() {
            let request = slot as u64 + 1;
            let start = state(&instance, request).end;
            let width = depth + 1;
            batch_items.push(if depth == 0 {
                BatchItemV1::new(request, true, 1, true, vec![7])
            } else {
                kept_rows.push(kept(request, 1));
                BatchItemV1::verify(request, width, true, vec![7])
            });
            appends.push(BatchAppendV1::new(
                start,
                u64::from(width),
                start + u64::from(width),
                (0..(start + u64::from(width)).div_ceil(64))
                    .map(|page| slot as u64 * 8 + page)
                    .collect(),
                None,
            ));
        }
        let batch = BatchRequestV1::new(BatchWireVersionV1::V3, batch_items, appends)
            .with_state_slots((0..u64::from(items)).collect())
            .encode()
            .unwrap();
        execute(&mut instance, &host, &batch);
        instance.commit_accepted(&kept_rows).unwrap();
    }
    let stats = crate::instance::dspark_graphs_for_test(instance.handle(), None);
    eprintln!("recorded startup keys: {stats:?}");
    assert_eq!(stats.denied, 0, "{stats:?}");
    assert_eq!(stats.epoch, 0, "{stats:?}");
    let eager = host
        .logs
        .borrow()
        .iter()
        .filter(|line| line.contains("\"event\":\"target-eager\""))
        .count();
    assert_eq!(eager, 0);
    for request in 1..=32 {
        instance.retire_request(request).unwrap();
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

#[test]
fn bounded_draft_chunks_preserve_full_proposals_and_replay_small_graphs() {
    let (host, mut instance) = create_with_limits(Shape::new(&FLASH, Ranks::TP4).unwrap(), 32, 192);
    let mut captured = false;
    for slot in 0..16 {
        execute(
            &mut instance,
            &host,
            &super::super::prompt(slot + 1, slot, 0, 3, true),
        );
        instance.commit().unwrap();
    }
    for count in [8, 16, 8, 16] {
        let items: Vec<_> = (0..count)
            .map(|slot| (slot + 1, slot, state(&instance, slot + 1).end))
            .collect();
        let from = host.logs.borrow().len();
        let sink = execute(&mut instance, &host, &verify_all(&items, 1));
        assert_eq!(
            sink.draft_tokens(),
            (1..=count).map(|id| (id, vec![0])).collect::<Vec<_>>()
        );
        instance
            .commit_accepted(&(1..=count).map(|id| kept(id, 1)).collect::<Vec<_>>())
            .unwrap();
        let logs = host.logs.borrow();
        let draft_widths: Vec<_> = logs[from..]
            .iter()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .filter(|r| r["schema"] == "chopinfer.graph-cache.v1" && r["family"] == "draft")
            .filter_map(|r| r["width"].as_u64())
            .collect();
        captured |= !draft_widths.is_empty();
        assert!(draft_widths.iter().all(|&n| n == 8), "{draft_widths:?}");
        drop(logs);
        for record in records(&host).iter().rev().take(count as usize) {
            assert_eq!(record["computed_positions"], 5);
            assert_eq!(record["draft_ids"].as_array().unwrap().len(), 5);
            assert_eq!(record["confidence_f32_bits"].as_array().unwrap().len(), 5);
        }
    }
    assert!(captured, "draft graphs were never captured");
    for request in 1..=16 {
        instance.retire_request(request).unwrap();
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

#[test]
fn warmed_full_batch_survives_mixed_tail_admission_and_matches_replay() {
    let run = |retain: bool| {
        let (host, mut instance) =
            create_with_limits(Shape::new(&FLASH, Ranks::TP4).unwrap(), 32, 32);
        for slot in 0..16 {
            execute(
                &mut instance,
                &host,
                &super::super::prompt(slot + 1, slot, 0, 3, true),
            );
            instance.commit().unwrap();
        }
        let mut outputs = Vec::new();
        // The recorded failure retained 32/16, filled on a mixed 31/14 tail,
        // then captured 32/16 again. Compare the same sequence with room to
        // capture the tail. Fake kernels validate metadata and state, while
        // the served job must establish device arithmetic equality.
        let tail = [5, 5, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        for (index, depths) in [vec![1; 16], tail.to_vec(), tail.to_vec(), vec![1; 16]]
            .iter()
            .enumerate()
        {
            if retain && index == 1 {
                let stats = crate::instance::dspark_graphs_for_test(instance.handle(), None);
                crate::instance::dspark_graphs_for_test(instance.handle(), Some(stats.live));
            }
            let mut items = Vec::new();
            let mut appends = Vec::new();
            let mut accepted = Vec::new();
            for (slot, &depth) in depths.iter().enumerate() {
                let id = slot as u64 + 1;
                let start = state(&instance, id).end;
                let rows = depth + 1;
                items.push(if depth == 0 {
                    BatchItemV1::new(id, true, 1, true, vec![7])
                } else {
                    accepted.push(kept(id, 1));
                    BatchItemV1::verify(id, rows, true, vec![7])
                });
                appends.push(BatchAppendV1::new(
                    start,
                    u64::from(rows),
                    start + u64::from(rows),
                    vec![slot as u64 * 8],
                    None,
                ));
            }
            let batch = BatchRequestV1::new(BatchWireVersionV1::V3, items, appends)
                .with_state_slots((0..depths.len() as u64).collect())
                .encode()
                .unwrap();
            let begin = host.logs.borrow().len();
            let sink = execute(&mut instance, &host, &batch);
            instance.commit_accepted(&accepted).unwrap();
            outputs.push((
                sink.draft_tokens(),
                (1..=16)
                    .map(|id| state(&instance, id).end)
                    .collect::<Vec<_>>(),
            ));
            let stats = crate::instance::dspark_graphs_for_test(instance.handle(), None);
            assert_eq!(stats.epoch, 0, "a tail evicted the warmed full batch");
            assert_eq!(stats.released, 0);
            assert_eq!(stats.reserved, 0);
            let logs = host.logs.borrow();
            let eager = logs[begin..]
                .iter()
                .any(|s| s.contains("\"event\":\"target-eager\""));
            let captured = logs[begin..].iter().any(|s| {
                s.contains("\"event\":\"capture\"") && s.contains("\"family\":\"target\"")
            });
            assert_eq!(eager, retain && (index == 1 || index == 2));
            if index == 3 || index == 2 {
                assert!(!captured, "a warmed target was captured again");
            }
        }
        for id in 1..=16 {
            instance.retire_request(id).unwrap();
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
        outputs
    };
    assert_eq!(run(true), run(false));
}

#[test]
fn exact_width_graph_pressure_retains_warm_keys_on_every_topology() {
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
            assert!(!step.iter().any(|s| s.starts_with("release graph")));
        }
        let stats = crate::instance::dspark_graphs_for_test(instance.handle(), None);
        assert!(stats.denied > 0);
        assert_eq!(stats.released, 0);
        assert_eq!(stats.epoch, 0);
        assert!(!stats.pressure);
        let logs = host.logs.borrow();
        assert!(
            !logs
                .iter()
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
            .position(|line| line.contains(crate::sm120::engram_hash_rows::SYMBOL))
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
