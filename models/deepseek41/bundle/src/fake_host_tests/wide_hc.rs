use super::*;

/// Each of the 77 HC boundaries per rank of a 1-, 8-, 16- or 32-row decode
/// runs the column-block producer and split finish at their measured
/// buckets (`hc_cols::BUCKETS`), and the served producer and fused tail
/// elsewhere: one pair per boundary, the finish next on the producer's
/// stream.
#[test]
fn decode_serves_measured_wide_hc_pairs() {
    let producer = crate::sm120::hc_cols::producer();
    let finish = crate::sm120::hc_cols::finish();
    for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
        for rows in [1u32, 8, 16, 32] {
            for kv in KV_LAYOUTS {
                let shape = Shape::new(&FLASH, ranks).unwrap();
                let host = FakeHost::new(shape, kv);
                let mut instance = create(&host, rows);
                let nr = shape.ranks() as usize;
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
                let served = match rows {
                    1 => "_hc_round_chain_w8_v1 ",
                    32 => "_hc_round_rows_r2_v1 ",
                    _ => "_hc_round_chain_w4_v1 ",
                };
                let bucket = crate::sm120::hc_cols::BUCKETS
                    .iter()
                    .find(|b| b.rows == rows);
                let (first, second) = match bucket {
                    Some(b) => {
                        let first = format!("{} ", producer.symbol(b.tile, b.transposed));
                        let second = format!("{} ", finish.symbol(b.warps));
                        assert_eq!(count(served), 0);
                        assert_eq!(count("_hc_finish_norm_w32_v1 "), 0);
                        for line in launches.iter().filter(|l| l.contains(&first)) {
                            assert!(line.contains(&format!(
                                "grid_x: 40, grid_y: {}, grid_z: 1",
                                rows.div_ceil(b.tile)
                            )));
                            assert!(line.contains("block_x: 256, block_y: 1, block_z: 1"));
                        }
                        for line in launches.iter().filter(|l| l.contains(&second)) {
                            assert!(
                                line.contains(&format!("grid_x: 2, grid_y: {rows}, grid_z: 1"))
                            );
                            assert!(line.contains(&format!(
                                "block_x: {}, block_y: 1, block_z: 1",
                                b.warps * 32
                            )));
                        }
                        (first, second)
                    }
                    None => {
                        assert_eq!(count("_hc_round_cols_"), 0);
                        assert_eq!(count("_hc_finish_split_"), 0);
                        (served.to_owned(), "_hc_finish_norm_w32_v1 ".to_owned())
                    }
                };
                assert_eq!(count(&first), 77 * nr);
                assert_eq!(count(&second), 77 * nr);
                assert_eq!(count("_hc_round_cols_") + count(served), 77 * nr);
                assert_eq!(
                    count("_hc_finish_split_") + count("_hc_finish_norm_w32_v1 "),
                    77 * nr
                );
                assert_eq!(count("_hc_completion_") + count("_hc_finish_twin_"), 0);
                // The finish is the next launch on its producer's stream.
                let stream = |l: &str| l.split_whitespace().nth(1).unwrap().to_owned();
                for (index, line) in launches.iter().enumerate() {
                    if line.contains(&first) {
                        let next = launches[index + 1..]
                            .iter()
                            .find(|l| stream(l) == stream(line))
                            .unwrap();
                        assert!(next.contains(&second), "{next}");
                    }
                }
                if let Some(dir) = std::env::var_os("CHOPINFER_PRODUCER_JOURNALS") {
                    let dir = std::path::PathBuf::from(dir);
                    std::fs::create_dir_all(&dir).unwrap();
                    let name = format!("wide.tp{nr}.c{rows}.{:?}.journal", kv.storage);
                    std::fs::write(dir.join(&name), lines.join("\n")).unwrap();
                    decode_attention::export_handles(&host.device, &dir, &name);
                }
                let (regions, collisions) = shared_overlap::audit_scratch(&lines);
                assert_eq!((regions, collisions), (40 * nr, 0));
                instance.destroy().unwrap();
                assert_eq!(host.acquired.get(), host.released.get());
            }
        }
    }
}
