use super::*;

#[test]
fn decode_fuses_77_hc_boundaries() {
    for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
        for rows in [1u32, 8, 32] {
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
                let created = host.device.journal.borrow()[..start].to_vec();
                let items = (0..u64::from(rows))
                    .map(|slot| (7 + slot, slot, 3, 9))
                    .collect::<Vec<_>>();
                step(&mut instance, &decode(&items), &mut Views::default());
                let lines = host.device.journal.borrow()[start..].to_vec();
                let launches: Vec<_> = lines.iter().filter(|l| l.starts_with("launch ")).collect();
                let count = |part: &str| launches.iter().filter(|l| l.contains(part)).count();
                // Measured C1 and C8 buckets join the chain producer and the
                // finish/norm tail in one launch with a zeroed per-row ticket.
                let fused = crate::sm120::hc_finish_norm::COMPLETION_BUCKETS
                    .iter()
                    .find(|(bucket, _)| *bucket == rows)
                    .map(|&(_, warps)| warps);
                assert_eq!(fused.is_some(), rows != 32);
                assert_eq!(count("hc_round_expand_mix_partials_v1"), 0);
                assert_eq!(count("hc_round_chain_w8_v1"), 0);
                assert_eq!(count("hc_round_chain_w4_v1"), 0);
                assert_eq!(count("_hc_completion_"), usize::from(rows != 32) * 77 * nr);
                assert_eq!(
                    count("hc_round_rows_r2_v1"),
                    usize::from(rows == 32) * 77 * nr
                );
                assert_eq!(count("hc_round_rows_r4_v1"), 0);
                assert_eq!(count("hc_round_rows_r8_v1"), 0);
                assert_eq!(count("hc_round_expand_mix_finish_v1"), 0);
                assert_eq!(
                    count("hc_finish_norm_w32_v1"),
                    usize::from(rows == 32) * 77 * nr
                );
                assert_eq!(count("hc_finish_norm_w16_v1"), 0);
                assert_eq!(
                    count("lagged_mix_collapse_split_v1"),
                    usize::from(rows == 32) * 3 * nr
                );
                assert_eq!(
                    count("hc_mixer_probe_warp_v1"),
                    usize::from(rows != 32) * 3 * nr
                );
                assert_eq!(count("hc_chain_mix"), 0);
                assert_eq!(count("input_tiles_v1"), 0);
                assert_eq!(
                    count("input_cached32_v1"),
                    match rows {
                        1 => 3 * nr,
                        8 => 3 * nr,
                        _ => 0,
                    }
                );
                assert_eq!(count("input_v1"), usize::from(rows == 32) * 3 * nr);
                assert_eq!(count("input_cached8_v1"), 0);
                assert_eq!(count("input_cached16_v1"), 0);

                assert_eq!(count("input_local_v1"), 0);
                if let Some(warps) = fused {
                    let entry = crate::sm120::hc_completion::served();
                    let symbol = format!("{} ", entry.symbol(warps));
                    let mut tickets = std::collections::BTreeSet::new();
                    for line in launches.iter().filter(|l| l.contains("_hc_completion_")) {
                        assert!(line.contains(&symbol), "{line}");
                        assert!(line.contains(&format!("grid_x: 161, grid_y: {rows}, grid_z: 1")));
                        assert!(
                            line.contains(&format!(
                                "block_x: {}, block_y: 1, block_z: 1",
                                warps * 32
                            ))
                        );
                        let spans = shared_overlap::journal_spans(line);
                        // Twenty-three spans, or 22 where the norm does not widen.
                        assert!(matches!(spans.len(), 22 | 23));
                        tickets.insert(*spans.last().unwrap());
                    }
                    // One ticket array per rank, a word per row, zeroed once
                    // when the instance was created and never bound elsewhere.
                    assert_eq!(tickets.len(), nr);
                    for &(buffer, offset, length) in &tickets {
                        assert!(length >= u64::from(rows) * 4);
                        assert!(created.iter().any(|l| {
                            l.starts_with("zero ")
                                && shared_overlap::journal_spans(l).iter().any(|z| {
                                    z.0 == buffer && z.1 <= offset && offset + length <= z.1 + z.2
                                })
                        }));
                        let other = launches
                            .iter()
                            .filter(|l| {
                                !l.contains("_hc_completion_")
                                    && !l.contains("lagged_mix")
                                    && !l.contains("hc_mixer_probe")
                            })
                            .flat_map(|l| shared_overlap::journal_spans(l))
                            .any(|s| s.0 == buffer && s.1 < offset + length && offset < s.1 + s.2);
                        assert!(!other, "tickets bound outside the HC mixers and boundaries");
                    }
                }
                for line in launches.iter().filter(|l| l.contains("hc_round_rows_")) {
                    assert!(line.contains("grid_x: 160, grid_y: 16, grid_z: 1"));
                    assert!(line.contains("block_x: 256, block_y: 1, block_z: 1"));
                }
                for line in launches.iter().filter(|l| {
                    l.contains("input_cached32_v1") || l.contains("hc_finish_norm_w32_v1")
                }) {
                    assert!(line.contains(&format!("grid_x: 1, grid_y: {rows}, grid_z: 1")));
                    assert!(line.contains("block_x: 1024, block_y: 1, block_z: 1"));
                }
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
                    decode_attention::export_handles(
                        &host.device,
                        &dir,
                        &format!("decode.tp{nr}.c{rows}.{:?}.journal", kv.storage),
                    );
                }
                let (regions, collisions) = shared_overlap::audit_scratch(&lines);
                assert_eq!((regions, collisions), (40 * nr, 0));
                instance.destroy().unwrap();
                assert_eq!(host.acquired.get(), host.released.get());
            }
        }
    }
}

#[test]
fn measured_producer_modules_retain_their_qualified_bytes() {
    for (symbol, expected) in [
        (
            "chop_deepseek41_hc_round_rows_r2_v1",
            "e8be72f62793188f4c5552d2cdd66dff1def03efa1e7513992c4ca9bde9843a1",
        ),
        (
            "chop_deepseek41_hc_finish_norm_w32_v1",
            "ee3c894ebf533a2acad15e277d4539840ce4cb862836689a940c4a1d8b20829f",
        ),
        (
            "chop_norm_quantize_deepseek41_input_cached32_v1",
            "870dfd3bac6205cf9373943b2ee7a3146c8f101ad27ca9b7af82f277ceb736dc",
        ),
        (
            "chop_deepseek41_hc_round_chain_w4_v1",
            "91593a6603ebf2854a4e3256a9d4859101dc74d378964e573d0a37e060752e5a",
        ),
        (
            "chop_deepseek41_hc_round_chain_w8_v1",
            "91593a6603ebf2854a4e3256a9d4859101dc74d378964e573d0a37e060752e5a",
        ),
        (
            "chop_deepseek41_hc_round_expand_mix_finish_v1",
            "2f35e8c35d3b0acf5c11488f84afd4e0f8ebf73d452e183c5c64b24729a536a9",
        ),
    ] {
        let (_, bytes) =
            crate::sm120::fold_cubin_for_symbol(symbol).expect("qualified serving symbol");
        assert_eq!(kernel_build::digest(bytes), expected, "{symbol}");
    }
}
