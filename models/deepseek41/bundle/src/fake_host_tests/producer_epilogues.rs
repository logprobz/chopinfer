use super::*;

#[test]
fn decode_fuses_77_hc_boundaries() {
    for kv in KV_LAYOUTS {
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let host = FakeHost::new(shape, kv);
        let mut instance = create(&host, 4);
        step(
            &mut instance,
            &prompt(7, 0, 0, 3, true),
            &mut Views::default(),
        );
        let start = host.device.journal.borrow().len();
        step(
            &mut instance,
            &decode(&[(7, 0, 3, 9)]),
            &mut Views::default(),
        );
        let lines = host.device.journal.borrow()[start..].to_vec();
        let launches: Vec<_> = lines.iter().filter(|l| l.starts_with("launch ")).collect();
        let count = |part: &str| launches.iter().filter(|l| l.contains(part)).count();
        assert_eq!(count("hc_round_expand_mix_partials_v1"), 77 * 4);
        assert_eq!(count("hc_round_expand_mix_finish_v1"), 77 * 4);
        assert_eq!(count("lagged_mix_collapse_split_v1"), 3 * 4);
        assert_eq!(count("chop_sinkhorn_hc_deepseek41_expand_f32_v1"), 3 * 4);
        assert_eq!(count("chop_row_rounding_round_deepseek41_w5120_v1"), 3 * 4);
        assert_eq!(count("chop_deepseek41_embedding_replicated_tp4_v1"), 4);
        assert_eq!(count("chop_row_embedding_deepseek41_tp4_v1"), 0);
        assert_eq!(count("chop_comm_reduce_scatter_f32_v1"), 80 * 4);
        assert_eq!(count("chop_comm_allgather_scattered_f32_v1"), 80 * 4);
        eprintln!(
            "PRODUCER_LAUNCHES {:?} rows=1 after={} norm_window_removed=248 hc_pairs=77 round_removed=77 embedding_removed=5",
            kv.storage,
            launches.len() / 4
        );
        if let Some(dir) = std::env::var_os("CHOPINFER_PRODUCER_JOURNALS") {
            let dir = std::path::PathBuf::from(dir);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(
                dir.join(format!("decode.{:?}.journal", kv.storage)),
                lines.join("\n"),
            )
            .unwrap();
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}
