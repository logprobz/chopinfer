use super::*;

#[test]
fn decode_fuses_77_hc_boundaries() {
    for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
        for rows in [1u32, 8] {
            for kv in KV_LAYOUTS {
                let shape = Shape::new(&FLASH, ranks).unwrap();
                let host = FakeHost::new(shape, kv);
                let mut instance = create(&host, rows.max(4));
                let nr = shape.ranks() as usize;
                let peer = nr == 2;
                for slot in 0..u64::from(rows) {
                    step(
                        &mut instance,
                        &prompt(7 + slot, slot, 0, 3, true),
                        &mut Views::default(),
                    );
                }
                let start = host.device.journal.borrow().len();
                let items = (0..u64::from(rows))
                    .map(|slot| (7 + slot, slot, 3, 9))
                    .collect::<Vec<_>>();
                step(&mut instance, &decode(&items), &mut Views::default());
                let lines = host.device.journal.borrow()[start..].to_vec();
                let launches: Vec<_> = lines.iter().filter(|l| l.starts_with("launch ")).collect();
                let count = |part: &str| launches.iter().filter(|l| l.contains(part)).count();
                assert_eq!(count("hc_round_expand_mix_partials_v1"), 77 * nr);
                assert_eq!(count("hc_round_expand_mix_finish_v1"), 77 * nr);
                assert_eq!(count("lagged_mix_collapse_split_v1"), 0);
                assert_eq!(count("hc_mixer_probe_warp_v1"), 3 * nr);
                assert_eq!(count("hc_chain_mix"), 0);
                assert_eq!(count("input_tiles_v1"), 0);
                assert_eq!(count("input_local_v1"), 0);
                for line in launches.iter().filter(|l| l.contains("hc_mixer_probe")) {
                    assert!(
                        line.contains("hc_mixer_probe_warp_v1"),
                        "diagnostic route entered serving"
                    );
                }
                assert_eq!(count("chop_sinkhorn_hc_deepseek41_expand_f32_v1"), 3 * nr);
                let embedding_round = usize::from(!peer);
                assert_eq!(
                    count("chop_row_rounding_round_deepseek41_w5120_v1"),
                    (3 + embedding_round) * nr
                );
                assert_eq!(count("embedding_replicated_tp"), usize::from(peer) * nr);
                assert_eq!(
                    count("chop_row_embedding_deepseek41_tp"),
                    usize::from(!peer) * nr
                );
                if nr > 1 {
                    assert_eq!(
                        count("chop_comm_reduce_scatter_f32_v1"),
                        (80 + embedding_round) * nr
                    );
                    assert_eq!(
                        count("chop_comm_allgather_scattered_f32_v1"),
                        (80 + embedding_round) * nr
                    );
                }
                assert_eq!(
                    count("hc_wide"),
                    0,
                    "measured wide mixer regression entered serving"
                );
                eprintln!(
                    "PRODUCER_LAUNCHES {:?} ranks={nr} rows={rows} after={} norm_window_removed=248 hc_pairs=77 round_removed=77 embedding_removed={}",
                    kv.storage,
                    launches.len() / nr,
                    if peer { 5 } else { 0 }
                );
                if let Some(dir) = std::env::var_os("CHOPINFER_PRODUCER_JOURNALS") {
                    let dir = std::path::PathBuf::from(dir);
                    std::fs::create_dir_all(&dir).unwrap();
                    std::fs::write(
                        dir.join(format!("decode.tp{nr}.c{rows}.{:?}.journal", kv.storage)),
                        lines.join("\n"),
                    )
                    .unwrap();
                }
                instance.destroy().unwrap();
                assert_eq!(host.acquired.get(), host.released.get());
            }
        }
    }
}
