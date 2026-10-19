//! Count host boundaries in warm DSpark steps without executing GPU kernels.
use super::*;

fn counts(lines: &[String]) -> serde_json::Value {
    let count = |prefix: &str| lines.iter().filter(|line| line.starts_with(prefix)).count();
    serde_json::json!({
        "stream_syncs": count("synchronize "),
        "device_to_host": count("download "),
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
    for (items, verify_rows) in [(1_u32, 6_u32), (8, 4)] {
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let (host, mut instance) = create_with_limits(shape, 8, 32);
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
                let draft = execute
                    .iter()
                    .position(|line| line.starts_with("replay "))
                    .unwrap();
                let gather = execute
                    .iter()
                    .position(|line| line.starts_with("mapped gather "))
                    .unwrap();
                let gathered = execute
                    .iter()
                    .rposition(|line| line.starts_with("mapped gather "))
                    .unwrap()
                    + 2;
                assert!(draft < gather);
                assert!(execute[gathered - 1].starts_with("upload "));
                assert_eq!(
                    counts(&execute[..gather])["stream_syncs"],
                    0,
                    "completed status readbacks must not be bracketed by stream drains"
                );
                let totals = counts(&journal[start..]);
                assert_eq!(
                    totals["stream_syncs"], 0,
                    "warm DSpark needs no explicit stream drain"
                );
                assert_eq!(
                    totals["device_to_host"], 6,
                    "four final rank checks, IDs and confidence"
                );
                assert_eq!(counts(&execute[..draft])["device_to_host"], 0);
                assert_eq!(
                    counts(&execute[draft..gather])["device_to_host"],
                    1,
                    "only the pending Engram ID boundary remains"
                );
                assert_eq!(
                    totals["graph_captures"], 0,
                    "inventory must use warm graphs"
                );
                let record = serde_json::json!({
                    "items": items, "capacity": 8, "draft_budget": verify_rows - 1, "forced_kept_rows": verify_rows,
                    "catch_up_and_draft_metadata": counts(&execute[..draft]),
                    "draft_and_readback": counts(&execute[draft..gather]),
                    "engram_gather": counts(&execute[gather..gathered]),
                    "target": counts(&execute[gathered..]),
                    "commit": counts(&journal[commit..]), "total": totals,
                    "scope": "bundle execute and commit; excludes executor sampling and scheduler acceptance",
                });
                println!("{record}");
                if let Some(output) = &output {
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
        let prior_rows = 3 * FLASH.dspark_stages() as usize + 5;
        let total_rows = prior_rows + 6 * (FLASH.shape.csa.index_sources.len() + 1);
        let mut status = vec![0; total_rows * 4];
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
    let mut status = vec![0; 3 * FLASH.dspark_stages() as usize * 4];
    status[0] = 1;
    host.injected.replace(Some(status));
    assert!(instance.capture_prefix_state(7, 3, &pages).is_err());
    assert!(host.injected.borrow().is_none());
    assert!(
        instance
            .execute(&verify(7, 0, 3, 5), &mut Sink::default())
            .is_err()
    );
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}
