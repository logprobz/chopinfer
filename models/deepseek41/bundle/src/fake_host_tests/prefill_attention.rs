//! The measured paired schedule retains the reference merge and fallbacks.
use super::*;

#[test]
fn retained_prefill_uses_measured_pairs_and_the_original_merge() {
    let trace = super::prefill_cache::composed_trace(0, false, true, false);
    let attention = trace
        .iter()
        .filter(|line| {
            line.starts_with("launch ")
                && line.contains("h16_d512_mixed_normalized_s1_r64_prefill_reuse2_v1")
        })
        .collect::<Vec<_>>();
    assert_eq!(attention.len(), 2 * 4 * 20);
    for rows in (64..=2496).step_by(128) {
        let grid = format!("grid_x: {}, grid_y: 2,", rows / 2);
        let launches = attention
            .iter()
            .filter(|s| s.contains(&grid))
            .collect::<Vec<_>>();
        assert_eq!(launches.len(), 8, "retained rows={rows}");
        assert!(
            launches
                .iter()
                .all(|s| s.contains("block_x: 256,") && s.contains("shared_memory_bytes: 85504,"))
        );
        let merge_grid = format!("grid_x: {rows}, grid_y: 16,");
        assert_eq!(
            trace
                .iter()
                .filter(|s| s.starts_with("launch ")
                    && s.contains("h16_d512_mixed_normalized_s1_r64_prefill_merge_v1")
                    && s.contains(&merge_grid))
                .count(),
            8
        );
    }
}

#[test]
fn paired_prefill_configures_shared_memory_and_keeps_unqualified_routes() {
    for (ranks, layout, rows, route) in [
        (Ranks::TP4, KV_LAYOUTS[0], 8192, 1),
        (Ranks::TP4, KV_LAYOUTS[0], 8191, 0),
        (Ranks::TP4, KV_LAYOUTS[1], 8192, 0),
        (Ranks::TP2, KV_LAYOUTS[0], 8192, 0),
        (Ranks::TP4, KV_LAYOUTS[0], 64, 2),
        (Ranks::TP4, KV_LAYOUTS[0], 192, 2),
        (Ranks::TP4, KV_LAYOUTS[0], 2496, 2),
        (Ranks::TP4, KV_LAYOUTS[0], 65, 0),
        (Ranks::TP4, KV_LAYOUTS[0], 128, 2),
        (Ranks::TP4, KV_LAYOUTS[0], 1281, 0),
        (Ranks::TP4, KV_LAYOUTS[0], 1408, 0),
        (Ranks::TP4, KV_LAYOUTS[0], 2560, 0),
        (Ranks::TP4, KV_LAYOUTS[0], 4095, 0),
        (Ranks::TP4, KV_LAYOUTS[0], 4096, 1),
        (Ranks::TP4, KV_LAYOUTS[0], 4097, 0),
        (Ranks::TP4, KV_LAYOUTS[1], 64, 0),
        (Ranks::TP2, KV_LAYOUTS[0], 64, 0),
        (Ranks::TP4, KV_LAYOUTS[1], 4096, 0),
        (Ranks::TP2, KV_LAYOUTS[0], 4096, 0),
    ]
    .into_iter()
    .chain(
        (256..=1280)
            .step_by(128)
            .map(|rows| (Ranks::TP4, KV_LAYOUTS[0], rows, 2)),
    ) {
        let shape = Shape::new(&FLASH, ranks).unwrap();
        let host = FakeHost::new(shape, layout);
        let record = crate::config::tests::record(&shape, layout, 1, false);
        let mut instance = api()
            .create(HostServicesTableV1::new(Rc::clone(&host)), &record)
            .unwrap();
        instance
            .execute(&prompt(7, 0, 0, rows, true), &mut Views::default())
            .unwrap();
        let lines = host.device.journal.borrow();
        assert_launches_fit_configured_shared(&lines, &shape.topology());
        let attention: Vec<_> = lines
            .iter()
            .filter(|line| {
                line.starts_with("launch ")
                    && line.contains("chop_latent_sources_")
                    && line.contains("_prefill")
                    && !line.contains("_merge_")
            })
            .collect();
        assert_eq!(attention.len(), shape.ranks() as usize * 40);
        for line in &attention {
            let shared = route == 1 && rows == 8192 && line.contains("_mixed_");
            let paired = !shared
                && (route == 1
                    || (route == 2 && line.contains("_mixed_") && !line.contains("_p32_")));
            assert_eq!(line.contains("_head_shared_v1"), shared, "{line}");
            assert_eq!(line.contains("_reuse2_v1"), paired, "{line}");
            if shared {
                assert!(
                    line.contains("grid_x: 8192, grid_y: 1, grid_z: 1,"),
                    "{line}"
                );
                assert!(line.contains("block_x: 256,"), "{line}");
                assert!(line.contains("shared_memory_bytes: 83200,"), "{line}");
            }
            if paired {
                assert!(
                    line.contains(&format!("grid_x: {}, grid_y: 2, grid_z: 1,", rows / 2)),
                    "{line}"
                );
                assert!(line.contains("block_x: 256,"), "{line}");
                assert!(line.contains("shared_memory_bytes: 85504,"), "{line}");
            }
        }
        let merges: Vec<_> = lines
            .iter()
            .filter(|line| {
                line.starts_with("launch ")
                    && line.contains("chop_latent_sources_")
                    && line.contains("_prefill")
                    && line.contains("_merge_")
            })
            .collect();
        assert_eq!(merges.len(), attention.len());
        assert!(merges.iter().all(|line| !line.contains("reuse")));
        drop(lines);
        instance.commit().unwrap();
        let from = host.device.journal.borrow().len();
        instance
            .execute(&decode(&[(7, 0, rows, 1)]), &mut Views::default())
            .unwrap();
        assert!(
            host.device.journal.borrow()[from..]
                .iter()
                .all(|line| !line.contains("reuse2"))
        );
        instance.commit().unwrap();
        instance.retire_request(7).unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}
