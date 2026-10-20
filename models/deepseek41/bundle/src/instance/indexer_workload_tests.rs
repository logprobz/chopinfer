//! Exercise real lowering and graph ownership on the CPU fake host.
use super::*;
use crate::fake_host_tests::indexer_workloads::host;
use bundle_runtime::geometry::Ranks;

fn inputs(rows: u32, per: u32, context: u32) -> PhaseInputs {
    let items = rows / per;
    let prefix = (0..items)
        .map(|i| (context - per - i % 3 * per) as i32)
        .collect::<Vec<_>>();
    let ends = prefix.iter().map(|&p| p + per as i32).collect::<Vec<_>>();
    PhaseInputs {
        decode: true,
        verify: per > 1,
        verify_rows: vec![per > 1; items as usize],
        item_order: (0..items as usize).collect(),
        request_ids: (1..=u64::from(items)).collect(),
        emits_row: vec![true; items as usize],
        prompt_rows: vec![false; items as usize],
        anchors: vec![false; items as usize],
        token_ids: vec![7; rows as usize],
        positions: prefix
            .iter()
            .flat_map(|&p| (0..per).map(move |r| p + r as i32))
            .collect(),
        token_batch: (0..items)
            .flat_map(|i| vec![i as i32; per as usize])
            .collect(),
        block_tables: ends
            .iter()
            .enumerate()
            .map(|(item, &end)| {
                (0..(end as u32).div_ceil(64))
                    .map(|p| (p * items + (item as u32 * 3 + 1) % items) as i32)
                    .collect()
            })
            .collect(),
        slot_ids: (0..items as i32).collect(),
        q_start: (0..items).map(|i| (i * per) as i32).collect(),
        q_len: vec![per as i32; items as usize],
        prefix_len: prefix,
        last_positions: ends.iter().map(|&n| i64::from(n - 1)).collect(),
        seq_lens: ends,
        ..PhaseInputs::default()
    }
}

#[test]
fn indexer_context_routes_capture_once_and_replay_the_matching_schedule() {
    for (rows, per) in [(8, 1), (32, 1), (32, 4)] {
        let (host, mut instance) = host(Ranks::TP4, false);
        super::super::INSTANCES.with(|table| {
            let mut table = table.borrow_mut();
            let live = table.get_mut(instance.handle()).unwrap();
            let mut seen = std::collections::BTreeMap::new();
            for context in [8192, 32768, 131072, 262144, 32768, 8192, 131072, 262145] {
                let start = host.device.journal.borrow().len();
                let mut run = live.plan_run(inputs(rows, per, context), None).unwrap();
                let route = run.indexer_route;
                let expected = if (131072..=262144).contains(&context) {
                    IndexerRoute::Combined
                } else if per == 4 && context == 32768 {
                    IndexerRoute::VerifyHierarchy
                } else {
                    IndexerRoute::Accepted
                };
                assert_eq!(route, expected);
                let fresh = !seen.contains_key(&route);
                assert!(run.programs.iter().all(|calls| calls.is_empty() != fresh));
                if fresh {
                    for calls in &run.programs {
                        let count =
                            |part: &str| calls.iter().filter(|c| c.symbol.contains(part)).count();
                        assert_eq!(
                            count("_mma_fp4_tile4_mode0_trace0_v1"),
                            if route == IndexerRoute::Combined {
                                4
                            } else {
                                0
                            }
                        );
                        assert_eq!(
                            count("_select_v3_cached4_v1"),
                            if route == IndexerRoute::Combined {
                                4
                            } else {
                                0
                            }
                        );
                        assert_eq!(
                            count("_hier_group_v1"),
                            if per == 1 || route != IndexerRoute::Accepted {
                                8
                            } else {
                                0
                            }
                        );
                        assert_eq!(
                            count("_hier_final_v1"),
                            if per == 1 || route != IndexerRoute::Accepted {
                                8
                            } else {
                                0
                            }
                        );
                        assert_eq!(count("_pack_fp4_guard_v1"), 8);
                        assert_eq!(count("_counts_v1"), 0);
                    }
                }
                live.launch(&mut run).unwrap();
                let journal = host.device.journal.borrow();
                let lines = &journal[start..];
                let captured = lines
                    .iter()
                    .filter(|s| s.starts_with("captured "))
                    .map(|s| s.split_whitespace().nth(2).unwrap().to_owned())
                    .collect::<Vec<_>>();
                let replayed = lines
                    .iter()
                    .filter(|s| s.starts_with("replay "))
                    .map(|s| s.split_whitespace().nth(2).unwrap().to_owned())
                    .collect::<std::collections::BTreeSet<_>>();
                assert!(!replayed.is_empty());
                if fresh {
                    assert_eq!(
                        captured
                            .iter()
                            .cloned()
                            .collect::<std::collections::BTreeSet<_>>(),
                        replayed
                    );
                    seen.insert(route, replayed);
                } else {
                    assert!(captured.is_empty());
                    assert_eq!(seen[&route], replayed);
                }
            }
            if let Some(dir) = std::env::var_os("CHOPINFER_INDEXER_WORKLOAD_JOURNALS") {
                let dir = std::path::PathBuf::from(dir);
                std::fs::create_dir_all(&dir).unwrap();
                std::fs::write(
                    dir.join(format!("rows{rows}-per{per}.journal")),
                    host.device.journal.borrow().join("\n"),
                )
                .unwrap();
            }
        });
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

#[test]
fn indexer_context_banks_keep_dspark_prefix_and_suffix_together() {
    let (host, mut instance) = host(Ranks::TP1, true);
    super::super::INSTANCES.with(|table| {
        let mut table = table.borrow_mut();
        let live = table.get_mut(instance.handle()).unwrap();
        let key = GraphKey {
            rank: 0,
            bucket: 32,
            items: 8,
        };
        let mut expected = std::collections::BTreeMap::new();
        for route in [
            IndexerRoute::Accepted,
            IndexerRoute::VerifyHierarchy,
            IndexerRoute::Combined,
            IndexerRoute::Accepted,
            IndexerRoute::Combined,
            IndexerRoute::VerifyHierarchy,
        ] {
            let fresh = !expected.contains_key(&route);
            assert_eq!(live.indexer_graph_contains(route, key), !fresh);
            live.activate_indexer_graphs(route);
            let stream = live.ranks[0].2;
            live.graphs
                .capture::<BundleAbiError>(&live.services, stream, key, 1, |_| Ok(()))
                .unwrap();
            let layers = &mut live.dspark.as_mut().unwrap().cycle.layers;
            assert!(!layers.is_empty());
            let mut pair = vec![live.graphs.cached(&[key]).unwrap()[0][0]];
            for layer in layers {
                layer.graphs.capture::<BundleAbiError>(&live.services, stream, key, 1, |_| Ok(())).unwrap();
                let graph = layer.graphs.cached(&[key]).unwrap()[0][0];
                assert!(!pair.contains(&graph));
                pair.push(graph);
            }
            if fresh {
                expected.insert(route, pair);
            } else {
                assert_eq!(expected[&route], pair);
            }
            assert!(live.indexer_graph_contains(route, key));
        }
        assert_eq!(live.graphs.ledger_len(), 3 * (1 + live.dspark.as_ref().unwrap().cycle.layers.len()));
        live.graphs.release_all(&live.services).unwrap();
        for route in [
            IndexerRoute::Accepted,
            IndexerRoute::VerifyHierarchy,
            IndexerRoute::Combined,
        ] {
            assert!(!live.indexer_graph_contains(route, key));
        }
    });
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}
