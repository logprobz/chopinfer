//! Count host boundaries in warm DSpark steps without executing GPU kernels.
use super::*;

fn counts(lines: &[String]) -> serde_json::Value {
    let count = |prefix: &str| lines.iter().filter(|line| line.starts_with(prefix)).count();
    serde_json::json!({
        "stream_syncs": count("synchronize "),
        "device_to_host": count("download "),
        "async_device_to_host": count("async mapped ids "),
        "physical_readbacks": count("download ") + count("async mapped ids "),
        "event_records": count("record "),
        "event_waits": count("wait "),
        "mapped_waits": count("async mapped wait "),
        "host_to_device": count("upload "),
        "mapped_gathers": count("mapped gather "),
        "graph_replays": count("replay "),
        "graph_captures": count("captured "),
    })
}

#[test]
fn warm_cycle_host_boundary_inventory() {
    let output = std::env::var_os("CHOPINFER_DSPARK_CYCLE_JOURNAL").map(std::path::PathBuf::from);
    if let Some(output) = &output {
        std::fs::create_dir_all(output).unwrap();
    }
    // The successor joins status into one packet at every width; the serial
    // path keeps per-rank status reads above C1.
    for (successor, items, verify_rows) in [(true, 1_u32, 6_u32), (true, 8, 4), (false, 8, 4)] {
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let (host, mut instance) =
            crate::dspark::prefetch::with_enabled(successor, || create_with_limits(shape, 8, 32));
        let joined = successor || items == 1;
        let batch = |start: u64, rows: u32, draft: bool| {
            BatchRequestV1::new(
                BatchWireVersionV1::V3,
                (0..items)
                    .map(|slot| {
                        if draft {
                            BatchItemV1::verify(u64::from(slot + 1), rows, true, vec![7])
                        } else {
                            BatchItemV1::new(
                                u64::from(slot + 1),
                                false,
                                rows,
                                true,
                                vec![7; rows as usize],
                            )
                        }
                    })
                    .collect(),
                (0..items)
                    .map(|slot| {
                        let end = start + u64::from(rows);
                        BatchAppendV1::new(
                            start,
                            u64::from(rows),
                            end,
                            (0..end.div_ceil(64))
                                .map(|page| u64::from(slot) * 8 + page)
                                .collect(),
                            None,
                        )
                    })
                    .collect(),
            )
            .with_state_slots((0..items).map(u64::from).collect())
            .encode()
            .unwrap()
        };
        execute(&mut instance, &host, &batch(0, 3, false));
        instance.commit().unwrap();
        for cycle in 0..3 {
            let start = host.journal.borrow().len();
            let position = 3 + u64::from(cycle * verify_rows);
            let result = execute(&mut instance, &host, &batch(position, verify_rows, true));
            assert_eq!(result.draft_tokens().len(), items as usize);
            let logs = host.logs.borrow();
            let timing = logs
                .iter()
                .rev()
                .find(|line| line.contains("\"phase\":\"dspark-engram\""))
                .expect("device gather timing record");
            let timing: serde_json::Value = serde_json::from_str(timing).unwrap();
            assert_eq!(timing["schema"], "chopinfer.deepseek41.step-shape.v1");
            assert_eq!(timing["items"], items);
            assert_eq!(timing["rows"], items * verify_rows);
            assert!(timing["host_ns"].is_u64() && timing["exposed_ns"].is_u64());
            drop(logs);
            let commit = host.journal.borrow().len();
            instance
                .commit_accepted(
                    &(0..items)
                        .map(|slot| kept(u64::from(slot + 1), verify_rows))
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            if cycle == 2 {
                let journal = host.journal.borrow();
                let execute = &journal[start..commit];
                let hash = execute
                    .iter()
                    .position(|line| line.contains(crate::sm120::engram_hash_entries::SYMBOL))
                    .unwrap();
                let before_hash: Vec<_> = execute[..hash]
                    .iter()
                    .enumerate()
                    .filter_map(|(index, line)| line.starts_with("replay ").then_some(index))
                    .collect();
                assert_eq!(
                    before_hash.len(),
                    4 * (1 + items.div_ceil(6)) as usize,
                    "one catch-up graph and bounded draft chunks per rank"
                );
                let draft = before_hash[4];
                let enqueue = execute
                    .iter()
                    .position(|line| line.starts_with("async mapped ids "))
                    .unwrap();
                let wait = execute
                    .iter()
                    .position(|line| line.starts_with("async mapped wait "))
                    .unwrap();
                let gathered = execute
                    .iter()
                    .rposition(|line| line.starts_with("mapped gather "))
                    .unwrap()
                    + 2;
                let last_replay = execute
                    .iter()
                    .rposition(|line| line.starts_with("replay "))
                    .unwrap();
                assert!(
                    draft < hash
                        && hash < enqueue
                        && enqueue < wait
                        && wait < gathered
                        && gathered < last_replay
                );
                assert!(execute[gathered - 1].starts_with("upload "));
                assert_eq!(
                    counts(&execute[..last_replay])["device_to_host"],
                    0,
                    "no intermediate readback before target replay"
                );
                assert_eq!(
                    counts(&execute[enqueue + 1..wait])["graph_replays"],
                    4,
                    "each target prefix must run before waiting for Engram"
                );
                assert_eq!(
                    counts(&execute[draft..hash])["event_records"],
                    5,
                    "record all drafter ranks and their join"
                );
                assert_eq!(
                    counts(&execute[draft..hash])["event_waits"],
                    6,
                    "join every drafter rank before target scratch reuse"
                );
                let totals = counts(&journal[start..]);
                assert_eq!(
                    totals["stream_syncs"], 0,
                    "warm DSpark needs no explicit stream drain"
                );
                assert_eq!(
                    totals["device_to_host"],
                    if joined { 1 } else { 5 },
                    "status and diagnostics are joined in one packet"
                );
                assert_eq!(
                    totals["async_device_to_host"], 2,
                    "disjoint ID slices feed the two layer gathers"
                );
                assert_eq!(totals["physical_readbacks"], if joined { 3 } else { 7 });
                assert_eq!(totals["mapped_waits"], 8);
                assert_eq!(
                    totals["graph_replays"],
                    4 * (4 + items.div_ceil(6)),
                    "one catch-up, bounded draft chunks and three target pieces per rank"
                );
                assert_eq!(
                    totals["graph_captures"], 0,
                    "inventory must use warm graphs"
                );
                let record = serde_json::json!({
                    "items":items,"successor":successor,"capacity":8,"draft_budget":verify_rows-1,"forced_kept_rows":verify_rows,
                    "catch_up_and_draft_metadata":counts(&execute[..draft]),
                    "draft_and_device_join":counts(&execute[draft..hash]),
                    "hash_and_async_id_copy":counts(&execute[hash..enqueue+1]),
                    "overlapped_target_prefix":counts(&execute[enqueue+1..wait]),
                    "engram_consumer_waits":counts(&execute[wait..gathered]),
                    "target_suffix_and_results":counts(&execute[gathered..]),
                    "commit":counts(&journal[commit..]),"total":totals,
                    "scope":"bundle execute and commit; includes the prepared gather ID copy; excludes executor sampling and scheduler acceptance"
                });
                println!("{record}");
                if let Some(output) = output.as_ref().filter(|_| successor) {
                    std::fs::write(
                        output.join(format!("c{items}.json")),
                        serde_json::to_vec_pretty(&record).unwrap(),
                    )
                    .unwrap();
                    std::fs::write(
                        output.join(format!("c{items}.journal")),
                        journal[start..].join("\n"),
                    )
                    .unwrap();
                }
            }
        }
        for request in 1..=u64::from(items) {
            instance.retire_request(request).unwrap();
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

#[test]
fn saved_catch_up_and_index_failures_refuse_target_logits() {
    for index_failure in [false, true] {
        let shape = Shape::new(&FLASH, Ranks::TP1).unwrap();
        let (host, mut instance) = create(shape);
        execute(
            &mut instance,
            &host,
            &super::super::prompt(7, 0, 0, 3, true),
        );
        instance.commit().unwrap();
        // Batched catch-up saves both status planes once across all stages;
        // the full five-row draft adds its two planes afterward.
        let prior_rows = 2 * (3 + 5);
        let total_rows = prior_rows + 6 * (FLASH.shape.csa.index_sources.len() + 2);
        let mut status = vec![0; total_rows * 4 + 40];
        status[if index_failure { prior_rows * 4 } else { 0 }] = 1;
        host.injected.replace(Some(status));
        let mut sink = Sink::default();
        assert!(instance.execute(&verify(7, 0, 3, 5), &mut sink).is_err());
        assert!(
            sink.views.is_empty(),
            "invalid saved status published target logits"
        );
        assert!(host.injected.borrow().is_none());
        assert!(
            host.logs
                .borrow()
                .iter()
                .any(|line| line.contains(if index_failure {
                    "DSpark index score"
                } else {
                    "DSpark quantized input"
                }))
        );
        assert_eq!(state(&instance, 7).end, 3);
        assert!(instance.commit_accepted(&[kept(7, 6)]).is_err());
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

#[test]
fn failed_snapshot_catch_up_poisoned_cursors_cannot_resume() {
    let shape = Shape::new(&FLASH, Ranks::TP1).unwrap();
    let (host, mut instance) = create(shape);
    execute(
        &mut instance,
        &host,
        &super::super::prompt(7, 0, 0, 3, true),
    );
    instance.commit().unwrap();
    let pages = (0..instance.describe_prefix_state().unwrap())
        .map(|page| 40 + page)
        .collect::<Vec<_>>();
    let mut status = vec![0; 3 * 8];
    status[0] = 1;
    host.injected.replace(Some(status));
    assert!(instance.capture_prefix_state(7, 3, &pages).is_err());
    assert!(host.injected.borrow().is_none());
    assert!(
        host.logs
            .borrow()
            .iter()
            .any(|line| line.contains("DSpark quantized input"))
    );
    assert!(
        instance
            .execute(&verify(7, 0, 3, 5), &mut Sink::default())
            .is_err()
    );
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}
