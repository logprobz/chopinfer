//! The measured paired schedule retains the reference merge and fallbacks.
use super::*;

#[test]
fn paired_prefill_configures_shared_memory_and_keeps_unqualified_routes() {
    for (ranks, layout, rows, paired) in [
        (Ranks::TP4, KV_LAYOUTS[0], 8192, true),
        (Ranks::TP4, KV_LAYOUTS[0], 8191, false),
        (Ranks::TP4, KV_LAYOUTS[1], 8192, false),
        (Ranks::TP2, KV_LAYOUTS[0], 8192, false),
    ] {
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
            assert_eq!(line.contains("_reuse2_v1"), paired, "{line}");
            if paired {
                assert!(
                    line.contains("grid_x: 4096, grid_y: 2, grid_z: 1,"),
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
