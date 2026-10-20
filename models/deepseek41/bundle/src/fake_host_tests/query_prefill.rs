use super::*;

#[test]
fn default_cone_fuses_twenty_full_row_query_boundaries() {
    use bundle_runtime::fake_host::RecordingSink;
    use runtime_bundle_sdk::prefill::{Boundary, encode};
    let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
    host.kv_pages.set(1024);
    let mut instance = super::prefill_skip::create_profile(&host, 8192, true);
    for offset in [0, 8192, 16384, 24576] {
        let request = encode(
            &prompt(1, 0, offset, 8192, offset == 24576),
            &[Boundary {
                request_id: 1,
                prompt_tokens: 32768,
                chunk_end: offset + 8192,
                chunk_tokens: 8192,
            }],
        )
        .unwrap();
        let start = host.device.journal.borrow().len();
        instance
            .execute_prefill(
                &request,
                &mut RecordingSink::<super::conformance::Limits>::default(),
            )
            .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
        let lines = host.device.journal.borrow()[start..].to_vec();
        let launches = lines
            .iter()
            .filter(|l| l.starts_with("launch "))
            .collect::<Vec<_>>();
        assert_eq!(
            launches
                .iter()
                .filter(|l| l.contains("chop_prefill_query_norm_ceil_cached_v1"))
                .count(),
            20 * 4
        );
        assert!(!launches.iter().any(|l| l.contains("chop_deepseek41_q_norm_f32_v1") && l.contains("grid_x: 8192,")));
        assert!(
            launches.iter().any(
                |l| l.contains("chop_deepseek41_q_norm_f32_v1") && !l.contains("grid_x: 8192,")
            )
        );
        eprintln!(
            "QUERY_PREFILL_DEFAULT offset={offset} saved_per_rank=20 launches_per_rank={}",
            launches.len() / 4
        );
        instance.commit().unwrap();
    }
    instance.retire_request(1).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

// This fixture also runs against the previous serving tree. Keep its
// workload independent of the candidate registration and selection.
#[test]
#[ignore = "requires a staged directory named by CHOPINFER_QUERY_PREFILL_JOURNALS"]
fn export_query_prefill_lifetimes() {
    let dir =
        std::path::PathBuf::from(std::env::var_os("CHOPINFER_QUERY_PREFILL_JOURNALS").unwrap());
    std::fs::create_dir_all(&dir).unwrap();
    for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
        for rows in [64, 8192] {
            let shape = Shape::new(&FLASH, ranks).unwrap();
            let host = FakeHost::new(shape, KV_LAYOUTS[0]);
            let mut instance = create(&host, 4);
            step(
                &mut instance,
                &prompt(7, 0, 0, rows, true),
                &mut Views::default(),
            );
            step(
                &mut instance,
                &decode(&[(7, 0, rows, 9)]),
                &mut Views::default(),
            );
            instance.retire_request(7).unwrap();
            instance.destroy().unwrap();
            assert_eq!(host.acquired.get(), host.released.get());
            let name = format!("query-tp{}-rows{rows}.journal", shape.ranks());
            std::fs::write(dir.join(&name), host.device.journal.borrow().join("\n")).unwrap();
            decode_attention::export_handles(&host.device, &dir, &name);
        }
    }
}

#[test]
fn full_chunk_query_fusion_preserves_indexer_status_revalidation() {
    for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
        for rows in [64, 8192] {
            let shape = Shape::new(&FLASH, ranks).unwrap();
            let host = FakeHost::new(shape, KV_LAYOUTS[0]);
            let mut instance = create(&host, 4);
            let start = host.device.journal.borrow().len();
            step(
                &mut instance,
                &prompt(7, 0, 0, rows, true),
                &mut Views::default(),
            );
            let lines = host.device.journal.borrow()[start..].to_vec();
            let launches = lines
                .iter()
                .filter(|l| l.starts_with("launch "))
                .collect::<Vec<_>>();
            let count = |symbol: &str| launches.iter().filter(|l| l.contains(symbol)).count();
            let phase = crate::program::Phase::Prefill { rows: rows as u32 };
            let stages = crate::program::stages(&FLASH, phase);
            let queries = stages
                .iter()
                .filter(|s| matches!(s, crate::program::Stage::QueryNorm(_)))
                .count()
                * shape.ranks() as usize;
            let indexers = stages
                .iter()
                .filter(|s| matches!(s, crate::program::Stage::Project(_, "indexer.q")))
                .count()
                * shape.ranks() as usize;
            let fused = rows == 8192;
            assert_eq!(
                count("chop_prefill_query_norm_ceil_cached_v1"),
                if fused { queries } else { 0 }
            );
            assert_eq!(count("chop_prefill_query_norm_ceil_reread_v1"), 0);
            assert_eq!(
                count("chop_deepseek41_q_norm_f32_v1"),
                if fused { 0 } else { queries }
            );
            assert_eq!(
                count("chop_deepseek41_dense_ceil_f32_w1280_prefill_w8_v1"),
                indexers + if fused { 0 } else { queries }
            );
            for line in launches
                .iter()
                .filter(|l| l.contains("chop_prefill_query_norm_ceil_cached_v1"))
            {
                assert!(line.contains("grid_x: 8192,"), "{line}");
                assert!(line.contains("block_x: 256,"), "{line}");
                assert!(line.contains("shared_memory_bytes: 0,"), "{line}");
            }
            if let Some(dir) = std::env::var_os("CHOPINFER_QUERY_PREFILL_JOURNALS") {
                let dir = std::path::PathBuf::from(dir);
                std::fs::create_dir_all(&dir).unwrap();
                std::fs::write(
                    dir.join(format!("tp{}-rows{rows}.journal", shape.ranks())),
                    lines.join("\n"),
                )
                .unwrap();
            }
            let start = host.device.journal.borrow().len();
            step(
                &mut instance,
                &decode(&[(7, 0, rows, 9)]),
                &mut Views::default(),
            );
            assert!(
                !host.device.journal.borrow()[start..].iter().any(
                    |l| l.starts_with("launch ") && l.contains("chop_prefill_query_norm_ceil_")
                )
            );
            instance.retire_request(7).unwrap();
            instance.destroy().unwrap();
            assert_eq!(host.acquired.get(), host.released.get());
        }
    }
}
