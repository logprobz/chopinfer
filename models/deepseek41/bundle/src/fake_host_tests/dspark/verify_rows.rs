//! Export the target schedule at exact verify widths, including mixed depths.
use super::*;

#[test]
fn target_programs_lower_at_wide_verify_rows() {
    let (host, mut instance) = create_with_limits(Shape::new(&FLASH, Ranks::TP4).unwrap(), 16, 128);
    // Eager execution exposes every launch, independent of the graph budget.
    crate::instance::dspark_graphs_for_test(instance.handle(), Some(0));
    for slot in 0..16 {
        execute(
            &mut instance,
            &host,
            &super::super::prompt(slot + 1, slot, 0, 3, true),
        );
        instance.commit().unwrap();
    }
    let directory = std::env::var_os("CHOPINFER_VERIFY_ROW_JOURNALS").map(std::path::PathBuf::from);
    let mut cases = 0;
    for count in [8_u32, 16] {
        for rows in [24_u32, 32].into_iter().chain(33..=64) {
            // A five-position drafter cannot produce more than six rows per item.
            if rows > count * 6 {
                continue;
            }
            let widths: Vec<_> = (0..count)
                .map(|slot| rows / count + u32::from(slot < rows % count))
                .collect();
            assert_eq!(widths.iter().sum::<u32>(), rows);
            let mut requests = Vec::new();
            let mut appends = Vec::new();
            let mut accepted = Vec::new();
            for (slot, &width) in widths.iter().enumerate() {
                let request = slot as u64 + 1;
                let start = state(&instance, request).end;
                requests.push(if width == 1 {
                    BatchItemV1::new(request, true, 1, true, vec![7])
                } else {
                    accepted.push(kept(request, 1));
                    BatchItemV1::verify(request, width, true, vec![7])
                });
                appends.push(BatchAppendV1::new(
                    start,
                    u64::from(width),
                    start + u64::from(width),
                    vec![slot as u64 * 8],
                    None,
                ));
            }
            let batch = BatchRequestV1::new(BatchWireVersionV1::V3, requests, appends)
                .with_state_slots((0..u64::from(count)).collect())
                .encode()
                .unwrap();
            let from = host.journal.borrow().len();
            let sink = execute(&mut instance, &host, &batch);
            assert_eq!(
                sink.draft_tokens(),
                widths
                    .iter()
                    .enumerate()
                    .filter(|(_, width)| **width > 1)
                    .map(|(slot, width)| (slot as u64 + 1, vec![0; (width - 1) as usize]))
                    .collect::<Vec<_>>()
            );
            let journal = host.journal.borrow();
            let step = &journal[from..];
            let hash = crate::sm120::engram_hash_rows::SYMBOL;
            assert_eq!(step.iter().filter(|line| line.contains(hash)).count(), 1);
            let target_start = step.iter().rposition(|line| line.contains(hash)).unwrap() + 1;
            let target: Vec<_> = step[target_start..]
                .iter()
                .filter(|line| line.starts_with("launch "))
                .cloned()
                .collect();
            assert!(target.len() > 1000);
            assert!(!target.iter().any(|line| line.contains("dspark_markov")));
            if rows > 32 {
                let count_symbol = |symbol: &str| {
                    target
                        .iter()
                        .filter(|line| line.split_whitespace().nth(2) == Some(symbol))
                        .count()
                };
                assert_eq!(
                    count_symbol(
                        "chop_deepseek41_verify_f32_n32320_k5120_v1_cached_decode_v1_spans_m64_compute_v1"
                    ),
                    4
                );
                assert_eq!(
                    count_symbol(
                        "chop_deepseek41_verify_f32_n384_k5120_v1_cached_decode_v1_spans_m64_compute_v1"
                    ),
                    40 * 4
                );
                assert!(
                    !target
                        .iter()
                        .any(|line| line.contains("verify_f32_n128_k512"))
                );
                for symbol in [
                    "chop_deepseek41_verify_mxfp8_n1280_k5120_g1_f32_cluster64_n64_c8s5_status_v1_packed_v1",
                    "chop_deepseek41_verify_mxfp8_n1024_k4096_g2_f32_cluster64_n64_c8s3_statusq_v1_packed_v1",
                    "chop_deepseek41_verify_mxfp8_n5120_k2048_g1_f32_cluster64_n32_c2s4_statusp_v1_packed_v1",
                    "chop_deepseek41_dense_ceil_w8192_sliced_s16w8_v1",
                ] {
                    assert_eq!(count_symbol(symbol), 40 * 4, "{symbol}");
                }
                assert!(
                    !target
                        .iter()
                        .any(|line| line.contains("verify_mxfp8_n512_k5120"))
                );
                assert_eq!(
                    count_symbol("chop_deepseek41_hc_round_cols_r4_tx_v1"),
                    77 * 4
                );
                assert_eq!(
                    count_symbol("chop_deepseek41_hc_finish_split_w32_v1"),
                    77 * 4
                );
                assert_eq!(
                    count_symbol("chop_sinkhorn_hc_deepseek41_lagged_mix_collapse_v1"),
                    0
                );
                assert_eq!(count_symbol("chop_deepseek41_moe_ceil_w5120_v1"), 0);
                assert_eq!(
                    count_symbol("chop_deepseek41_shared_swiglu_w576_parallel_v1"),
                    40 * 4
                );
                assert_eq!(
                    target
                        .iter()
                        .filter(
                            |line| line.contains("_verify") && line.contains("_cached_profile0_v1")
                        )
                        .count(),
                    40 * 4
                );
            }
            if let Some(directory) = &directory {
                std::fs::create_dir_all(directory).unwrap();
                let stem = format!("items-{count}-rows-{rows}");
                std::fs::write(directory.join(format!("{stem}.target")), target.join("\n"))
                    .unwrap();
                std::fs::write(directory.join(format!("{stem}.journal")), step.join("\n")).unwrap();
                std::fs::write(
                    directory.join(format!("{stem}.json")),
                    serde_json::to_vec_pretty(&serde_json::json!({
                        "items": count, "rows": rows, "item_rows": widths,
                        "ranks": 4, "target_launches": target.len(), "eager": true,
                    }))
                    .unwrap(),
                )
                .unwrap();
            }
            drop(journal);
            instance.commit_accepted(&accepted).unwrap();
            cases += 1;
        }
    }
    assert_eq!(cases, 52);
    for request in 1..=16 {
        instance.retire_request(request).unwrap();
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}
