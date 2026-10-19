//! Prefill uses staged tiles at serving and smaller chunk sizes.
use super::*;
#[test]
fn prefill_dense_launches_use_staged_tiles() {
    for rows in [1024, 2048, 8192] {
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let host = FakeHost::new(shape, KV_LAYOUTS[0]);
        let record = crate::config::tests::record(&shape, KV_LAYOUTS[0], 1, false);
        let mut instance = api()
            .create(HostServicesTableV1::new(Rc::clone(&host)), &record)
            .unwrap();
        instance
            .execute(&prompt(7, 0, 0, rows, true), &mut Views::default())
            .unwrap();
        let lines = host.device.journal.borrow();
        let dense: Vec<_> = lines
            .iter()
            .filter(|l| {
                l.starts_with("launch ")
                    && l.contains("chop_deepseek41_mxfp8_")
                    && !l.contains("_prepare_v1")
            })
            .collect();
        assert_eq!(
            dense.len(),
            4 * 330 * 2,
            "{rows}: each dense op admits then computes"
        );
        assert_eq!(
            dense.iter().filter(|l| l.contains("_admit_v1")).count(),
            4 * 330
        );
        for line in dense.iter().filter(|l| !l.contains("_admit_v1")) {
            assert!(line.contains("_prefill_tma_v1"), "{line}");
            assert!(line.contains(&format!("grid_y: {},", rows / 128)), "{line}");
            assert!(line.contains("block_x: 384,"), "{line}");
            assert!(line.contains("shared_memory_bytes: 66560,"), "{line}");
        }
        if let Some(dir) = std::env::var_os("CHOPINFER_DEEPSEEK41_PREFILL_JOURNAL_DIR") {
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(
                std::path::Path::new(&dir).join(format!("tp4-native-{rows}.journal")),
                lines.join("\n"),
            )
            .unwrap();
        }
        drop(lines);
        instance.commit().unwrap();
        let from = host.device.journal.borrow().len();
        instance
            .execute(&decode(&[(7, 0, rows, 1)]), &mut Views::default())
            .unwrap();
        let lines = host.device.journal.borrow();
        let dense: Vec<_> = lines[from..]
            .iter()
            .filter(|l| {
                l.starts_with("launch ")
                    && l.contains("chop_deepseek41_mxfp8_")
                    && !l.contains("_prepare_v1")
            })
            .collect();
        // Decode selects only the qualified packed, ticket and split-fold buckets.
        // Each paired shared gate/up replaces two individual projections.
        let completed = dense
            .iter()
            .filter(|line| {
                let symbol = line.split_whitespace().nth(2).unwrap();
                symbol.contains("mxfp8_stream_")
                    || ((symbol.contains("mxfp8_ticket_") || symbol.contains("mxfp8_prefetch_"))
                        && !symbol.contains("_initialize_"))
                    || (symbol.contains("mxfp8_decode_")
                        && !symbol.ends_with("_admit_v1")
                        && !symbol.ends_with("_partials_v1"))
            })
            .count();
        assert_eq!(completed, 4 * (330 - 40));
        assert_eq!(dense.len(), 4 * 370);
        assert!(dense.iter().all(|l| !l.contains("_prefill_tma_v1")));
        drop(lines);
        instance.commit().unwrap();
        instance.retire_request(7).unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}
