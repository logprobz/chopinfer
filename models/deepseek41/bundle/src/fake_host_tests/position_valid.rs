//! Full prefill uses the same live/valid planners as decode and verify.
use super::*;

#[test]
fn prefill_indexer_blocks_use_live_and_valid_grids() {
    for kv in KV_LAYOUTS {
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let host = FakeHost::new(shape, kv);
        let record = crate::config::tests::record(&shape, kv, 1, false);
        let mut instance = api()
            .create(HostServicesTableV1::new(Rc::clone(&host)), &record)
            .unwrap();
        instance
            .execute(&prompt(7, 0, 0, 8192, true), &mut Views::default())
            .unwrap();
        let journal = host.device.journal.borrow();
        let scores = |family: &str| {
            journal
                .iter()
                .filter(|line| {
                    line.starts_with("launch ")
                        && line.contains(family)
                        && line.contains("_scores_v3")
                })
                .collect::<Vec<_>>()
        };
        let dense = scores("chop_position_indexer_");
        assert_eq!(dense.len(), 4 * 4 * 128);
        for line in dense {
            assert!(line.contains("_scores_v3_live_v1"), "{line}");
            assert!(
                line.contains("grid_x: 192, grid_y: 64, grid_z: 1"),
                "{line}"
            );
        }
        let compact = scores("chop_position_blocks_");
        assert_eq!(compact.len(), 4 * 4 * 2);
        for line in compact {
            assert!(line.contains("_scores_v3_valid_v1"), "{line}");
            assert!(
                line.contains("grid_x: 192, grid_y: 4096, grid_z: 1"),
                "{line}"
            );
        }
        let candidates = scores("chop_candidate_block_ids_");
        assert_eq!(candidates.len(), 4 * 128);
        for line in candidates {
            assert!(line.contains("_scores_v3_valid_v1"), "{line}");
            assert!(line.contains("grid_x: 32, grid_y: 64, grid_z: 1"), "{line}");
        }
        drop(journal);
        instance.commit().unwrap();
        instance.retire_request(7).unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}
