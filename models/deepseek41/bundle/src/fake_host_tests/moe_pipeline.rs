//! Native MoE prefill's gate/up and down tiles at Flash TP4.
use super::*;

/// Native MoE prefill runs gate/up through the fused ceil's r2w4s3 tiles in
/// steps of at most `FUSED_CEIL_ROWS` rows and through the wide r2w6s3
/// tiles' lean loop above, never the other in either; down runs the served
/// r2w4s3 tiles up to `FUSED_CEIL_ROWS` rows and the two-CTA r1w4s2 lean
/// tiles above.
#[test]
fn native_prefill_runs_wide_gate_up_tiles_above_the_fused_rows() {
    crate::sm120::moe_native::with_native(true, check_native_gate_up_tiles);
}

fn check_native_gate_up_tiles() {
    for kv in KV_LAYOUTS {
        for rows in [2048, 2049, 8192] {
            let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
            let host = FakeHost::new(shape, kv);
            let record = crate::config::tests::record(&shape, kv, 1, false);
            let mut instance = api()
                .create(HostServicesTableV1::new(Rc::clone(&host)), &record)
                .unwrap();
            instance
                .execute(&prompt(7, 0, 0, rows, true), &mut Views::default())
                .unwrap();
            let journal = host.device.journal.borrow();
            assert_launches_fit_configured_shared(&journal, &shape.topology());
            let launches = |pattern: &str| {
                journal
                    .iter()
                    .filter(|l| l.starts_with("launch ") && l.contains(pattern))
                    .count()
            };
            let gate_up = "chop_moe_native_gateup_e384k6_n576_k5120_t8192_clamp4120000041200000_";
            let fused = rows <= u64::from(crate::sm120::moe_native::FUSED_CEIL_ROWS);
            let (tile, other) = if fused {
                ("r2w4s3", "r2w6s3")
            } else {
                ("r2w6s3", "r2w4s3")
            };
            assert_eq!(launches(&format!("{gate_up}{tile}_pack_v1 ")), 4 * 40);
            // The backbone gate/up bank is in row tiles at TP4, which the
            // compute tiles' row-tiled builds read (`sm120::moe_row_tiles`).
            let tiles = if fused {
                "_tiles_ceil"
            } else {
                "_tiles_scanned_row_tiles_v1 "
            };
            assert_eq!(launches(&format!("{gate_up}{tile}{tiles}")), 4 * 40);
            assert_eq!(launches(&format!("{gate_up}{tile}_tiles_")), 4 * 40);
            assert_eq!(launches(&format!("{gate_up}{other}_")), 0);
            let down = "chop_moe_native_down_e384k6_n5120_k576_t8192_";
            let (tile, tiles, other) = if fused {
                ("r2w4s3", "_tiles_v1 ", "r1w4s2")
            } else {
                ("r1w4s2", "_tiles_scanned_v1 ", "r2w4s3")
            };
            assert_eq!(launches(&format!("{down}{tile}_pack_v1 ")), 4 * 40);
            assert_eq!(launches(&format!("{down}{tile}{tiles}")), 4 * 40);
            assert_eq!(launches(&format!("{down}{other}_")), 0);
            drop(journal);
            instance.commit().unwrap();
            instance.retire_request(7).unwrap();
            instance.destroy().unwrap();
            assert_eq!(host.acquired.get(), host.released.get());
        }
    }
}
