//! Real-input exact HC row tiles and fused tails at eight and thirty-two rows.
mod artifacts;
#[path = "../hc_replay_buffers.rs"]
mod buffers;
#[path = "../hc_replay_corpus.rs"]
#[allow(dead_code, reason = "shared authenticated capture loader")]
mod corpus;
mod device;
mod registration;
mod replay;
mod served;
mod timing;

#[test]
fn plans_reject_impossible_cases_and_keep_original_partial_slots() {
    assert_eq!(registration::FAMILY, "sinkhorn_hc_round_rows");
    let entry = registration::entry();
    for rows in [1, 3, 6, 8, 31, 32] {
        for tile in [2, 4, 8] {
            for live in [0, 1, rows - 1, rows] {
                let plan = entry.plan(rows, live, tile).unwrap();
                assert_eq!(plan.grid, [160, rows.div_ceil(tile), 1]);
                assert_eq!(plan.block, [128 * tile, 1, 1]);
                assert_eq!(plan.workspace_bytes, u64::from(rows) * 4096 * 4);
                let mut projection = std::collections::BTreeSet::new();
                let mut squares = std::collections::BTreeSet::new();
                for group in 0..rows.div_ceil(tile) {
                    for chain in 0..160 {
                        for warp in 0..tile * 4 {
                            let row = group * tile + warp / 4;
                            if row >= live {
                                continue;
                            }
                            for mix in (warp % 4..24).step_by(4) {
                                assert!(projection.insert(row * 4096 + mix * 160 + chain));
                            }
                            if warp % 4 == 0 && chain < 8 {
                                for lane in 0..32 {
                                    assert!(squares.insert(row * 4096 + 3840 + chain * 32 + lane));
                                }
                            }
                        }
                    }
                }
                assert_eq!(projection.len(), live as usize * 3840);
                assert_eq!(squares.len(), live as usize * 256);
                assert!(projection.is_disjoint(&squares));
            }
        }
        assert!(entry.plan(rows, rows + 1, 4).is_err());
    }
    for (rows, live, tile) in [(0, 0, 4), (1, 8, 4), (8, 8, 0), (32, 32, 3)] {
        assert!(entry.plan(rows, live, tile).is_err());
    }
    let source =
        include_str!("../../../../../../crates/kernels-sm120/cuda/sinkhorn_hc/round_rows.cuh");
    assert!(!source.contains("griddep_") && !source.contains("blockIdx"));
    assert_eq!(
        entry.cubin_spec(registration::CUBIN).unwrap().entries.len(),
        3
    );
}
