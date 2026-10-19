//! The model option reaches both dense and compact serving, including graphs.
use super::*;

#[test]
fn native_indexer_default_and_simt_option_reach_prefill_and_decode() {
    for enabled in [false, true] {
        for kv in KV_LAYOUTS {
            let mut shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
            shape.runtime.indexer_mma = enabled;
            let host = FakeHost::new(shape, kv);
            let mut instance = create(&host, 1);
            instance
                .execute(&prompt(7, 0, 0, 64, true), &mut Views::default())
                .unwrap_or_else(|error| panic!("{error:?}: {:?}", host.logs.borrow()));
            instance.commit().unwrap();
            instance
                .execute(&decode(&[(7, 0, 64, 11)]), &mut Views::default())
                .unwrap_or_else(|error| panic!("{error:?}: {:?}", host.logs.borrow()));
            instance.commit().unwrap();
            let journal = host.device.journal.borrow();
            let scores: Vec<_> = journal
                .iter()
                .filter(|line| {
                    line.starts_with("launch ")
                        && (line.contains("chop_position_indexer_")
                            || line.contains("chop_position_blocks_"))
                        && line.contains("_scores_v3")
                        && !line.contains("_pack_fp4_")
                })
                .collect();
            assert!(!scores.is_empty());
            let native = enabled && kv.storage == KvStorage::Native;
            assert!(
                scores
                    .iter()
                    .all(|line| line.contains("_mma_fp4_guard_fast_v1") == native)
            );
            assert!(scores.iter().any(|line| line.contains("grid_y: 64,")));
            assert!(scores.iter().any(|line| line.contains("grid_y: 1,")));
            let packed = journal
                .iter()
                .filter(|line| line.starts_with("launch ") && line.contains("_pack_fp4_guard_v1"))
                .count();
            assert_eq!(packed, if native { scores.len() } else { 0 });
            let hierarchy: Vec<_> = journal
                .iter()
                .filter(|line| line.starts_with("launch ") && line.contains("_hier_"))
                .collect();
            assert_eq!(hierarchy.len(), if native { 4 * 8 * 2 } else { 0 });
            assert!(
                hierarchy
                    .iter()
                    .all(|line| line.contains("grid_y: 1,") && !line.contains("negative"))
            );
            assert_eq!(
                host.device
                    .module_cubins
                    .borrow()
                    .values()
                    .any(|c| c.starts_with("position_mma_")),
                enabled
            );
            assert_eq!(
                host.device
                    .module_cubins
                    .borrow()
                    .values()
                    .any(|name| name.starts_with("position_mma_guard_fast_")),
                native
            );
            assert_eq!(
                host.device
                    .module_cubins
                    .borrow()
                    .values()
                    .any(|name| name.starts_with("position_hierarchical_")),
                native
            );
            assert!(
                !journal
                    .iter()
                    .any(|line| line.starts_with("launch ") && line.contains("_counts_v1"))
            );
            drop(journal);
            instance.retire_request(7).unwrap();
            instance.destroy().unwrap();
            assert_eq!(host.acquired.get(), host.released.get());
        }
    }
}

#[test]
fn hierarchical_decode_binds_ordered_groups_to_disjoint_score_tail() {
    let spans = super::shared_overlap::journal_spans;
    for rows in [1, 2, 4, 6, 8, 16, 32] {
        // C6 replays the C8 graph; its device live count still admits six rows.
        let bucket = u32::next_power_of_two(rows);
        let shape = Shape::new(&FLASH, Ranks::TP1).unwrap();
        let host = FakeHost::new(shape, KV_LAYOUTS[0]);
        let mut instance = create(&host, rows);
        for i in 0..rows {
            instance
                .execute(
                    &prompt(u64::from(i + 7), u64::from(i), 0, 1, true),
                    &mut Views::default(),
                )
                .unwrap();
            instance.commit().unwrap();
        }
        let start = host.device.journal.borrow().len();
        let items: Vec<_> = (0..rows)
            .map(|i| (u64::from(i + 7), u64::from(i), 1, 11))
            .collect();
        instance
            .execute(&decode(&items), &mut Views::default())
            .unwrap();
        instance.commit().unwrap();
        let journal = host.device.journal.borrow();
        let mut checked = 0;
        for (at, group) in journal
            .iter()
            .enumerate()
            .skip(start)
            .filter(|(_, line)| line.starts_with("launch ") && line.contains("_hier_group_v1 "))
        {
            let partial = &journal[at - 1];
            let final_call = &journal[at + 1];
            let symbol = partial.split_whitespace().nth(2).unwrap();
            let old_merge = symbol
                .replace("_live_v1", "_live_merge_v1")
                .replace("_valid_v1", "_valid_merge_v1");
            assert!(group.contains(&format!("{old_merge}_hier_group_v1")));
            assert!(final_call.contains(&format!("{old_merge}_hier_final_v1")));
            assert!(group.contains(&format!("grid_x: 4, grid_y: {bucket}, grid_z: 1")));
            assert!(final_call.contains(&format!("grid_x: 1, grid_y: {bucket}, grid_z: 1")));
            assert_eq!(
                group.split_once(" } [").unwrap().1,
                final_call.split_once(" } [").unwrap().1
            );
            let p = spans(partial);
            let g = spans(group);
            assert_eq!(p, g[..p.len()]);
            assert_eq!(g, spans(final_call));
            assert_eq!(g.len(), p.len() + 1);
            let lists = *g.last().unwrap();
            let score = p[0];
            assert_eq!(lists.0, score.0);
            assert_eq!(lists.2, u64::from(bucket) * 4 * 513 * 8);
            assert_eq!(lists.1 + lists.2, score.1 + score.2);
            assert!(lists.1 >= score.1 + u64::from(bucket) * (1 << 20) * 4);
            assert_eq!(
                group.split_whitespace().nth(1),
                final_call.split_whitespace().nth(1)
            );
            checked += 1;
        }
        assert_eq!(checked, 8);
        drop(journal);
        for i in 0..rows {
            instance.retire_request(u64::from(i + 7)).unwrap();
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}
