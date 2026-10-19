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
                let totals = counts(&journal[start..]);
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
