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
                    .all(|line| line.contains("_mma_fp4_guard_v1") == native)
            );
            assert!(scores.iter().any(|line| line.contains("grid_y: 64,")));
            assert!(scores.iter().any(|line| line.contains("grid_y: 1,")));
            let packed = journal
                .iter()
                .filter(|line| line.starts_with("launch ") && line.contains("_pack_fp4_guard_v1"))
                .count();
            assert_eq!(packed, if native { scores.len() } else { 0 });
            assert_eq!(
                host.device
                    .module_cubins
                    .borrow()
                    .values()
                    .any(|c| c.starts_with("position_mma_")),
                enabled
            );
            drop(journal);
            instance.retire_request(7).unwrap();
            instance.destroy().unwrap();
            assert_eq!(host.acquired.get(), host.released.get());
        }
    }
}
