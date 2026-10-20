use super::*;

// Six and 32 rows replace 77 five/six-launch chains with two launches and
// retain all original operands. Eight and 16 rows retain the unfused HC path.
// The measured MoE grid changes only 160 small-gate grid_x fields at 32
// verify rows (four ranks, 40 layers); all other journal lines stay identical.
// Resources the prefill query fusion, grouped down and router bindings load
// before verify renumber handles in these fragments. The certified router
// rows also widen the router projection's workspace view (251691008 to
// 251756544 bytes) on its verify launches; grids and other operands match.
// Native prefill resources change handle numbers before these fragments;
// every normalized verify line is identical to the exact expert control.
// The resident small gate replaces those 160 launches' symbol and grid_x
// (1128 to 1504) with identical operands. Its module renumbers buffer, stream,
// event and graph handles at every width; no other line changes.
// The prefill matrix module changes resource IDs only in these verify
// bodies. Full creation histories and typed handles are audited separately.
// The prefill matrix modules change resource IDs only in these verify
const JOURNALS: [(u32, (usize, u64)); 4] = [
    (6, (8040, 0xdc91ad4ed589171e)),
    (8, (8992, 0x87e217aaf14ffd6d)),
    (16, (9964, 0x5320812fc0eb315d)),
    (32, (8384, 0x3067459b622060d9)),
];

#[test]
fn verify_hc_fuses_only_measured_rows_and_keeps_scratch_disjoint() {
    let mut missing = Vec::new();
    let mut journal_mismatches = Vec::new();
    for rows in [6u32, 8, 16, 32] {
        // DSpark supports the trained native KV layout only.
        for kv in KV_LAYOUTS
            .into_iter()
            .filter(|kv| kv.storage == KvStorage::Native)
        {
            let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
            let host = FakeHost::new(shape, kv);
            host.speculation.set(Some((5, 32)));
            let (items, width) = if rows == 6 { (1, 6) } else { (rows / 4, 4) };
            let mut instance = create(&host, items);
            for slot in 0..u64::from(items) {
                step(
                    &mut instance,
                    &prompt(7 + slot, slot, 0, 3, true),
                    &mut Views::default(),
                );
            }
            let tokens = vec![9; width as usize];
            let batch = (0..u64::from(items))
                .map(|slot| (7 + slot, slot, 3, tokens.as_slice()))
                .collect::<Vec<_>>();
            let start = host.device.journal.borrow().len();
            let mut views = Views::default();
            instance.execute(&verify(&batch), &mut views).unwrap();
            assert_eq!(views.rows(), rows);
            let lines = host.device.journal.borrow()[start..].to_vec();
            // Save complete before/after journals even when the admission
            // assertion fails, so the launch and binding delta is auditable.
            if let Some(dir) = std::env::var_os("CHOPINFER_VERIFY_HC_JOURNALS") {
                let dir = std::path::PathBuf::from(dir);
                std::fs::create_dir_all(&dir).unwrap();
                std::fs::write(
                    dir.join(format!("verify.tp4.r{rows}.{:?}.journal", kv.storage)),
                    lines.join("\n"),
                )
                .unwrap();
                decode_attention::export_handles(
                    &host.device,
                    &dir,
                    &format!("verify.tp4.r{rows}.{:?}.journal", kv.storage),
                );
                std::fs::write(
                    dir.join(format!("verify.tp4.r{rows}.{:?}.journal.full", kv.storage)),
                    host.device.journal.borrow().join("\n"),
                )
                .unwrap();
            }
            let expected = JOURNALS.iter().find(|(r, _)| *r == rows).unwrap().1;
            if let Err(error) = bundle_runtime::fake_host::check_journal(
                &lines,
                expected,
                &format!("verify-hc-{rows}.journal"),
            ) {
                journal_mismatches.push((rows, error));
            }
            let launches: Vec<_> = lines.iter().filter(|l| l.starts_with("launch ")).collect();
            let count = |symbol: &str| launches.iter().filter(|l| l.contains(symbol)).count();
            let qualified = matches!(rows, 6 | 32);
            let pairs = if qualified { 77 * 4 } else { 0 };
            if count("hc_finish_norm_w32_v1") != pairs {
                missing.push((rows, kv.storage, count("hc_finish_norm_w32_v1"), pairs));
            }
            eprintln!(
                "VERIFY_HC_LAUNCHES rows={rows} kv={:?} per_rank={} tails={} producers={}",
                kv.storage,
                launches.len() / 4,
                count("hc_finish_norm_w32_v1") / 4,
                count("hc_round_chain_w4_v1") / 4
            );
            if count("hc_finish_norm_w32_v1") == pairs {
                assert_eq!(
                    count("hc_round_chain_w4_v1"),
                    if rows == 6 { pairs } else { 0 }
                );
                assert_eq!(
                    count("hc_round_rows_r2_v1"),
                    if rows == 32 { pairs } else { 0 }
                );
                assert_eq!(count("hc_round_chain_w8_v1"), 0);
                assert_eq!(count("hc_round_expand_mix_partials_v1"), 0);
                assert_eq!(count("hc_round_expand_mix_finish_v1"), 0);
                assert_eq!(
                    count("lagged_mix_collapse_split_v1"),
                    (if qualified { 0 } else { 80 }) * 4
                );
                assert_eq!(
                    count("chop_row_rmsnorm_deepseek41_w5120_v1"),
                    (if qualified { 3 } else { 80 }) * 4
                );
                assert_eq!(
                    count("hc_mixer_probe_warp_v1"),
                    (if qualified { 3 } else { 0 }) * 4
                );
                for line in launches
                    .iter()
                    .filter(|l| l.contains("hc_round_chain_w4_v1"))
                {
                    assert!(line.contains(&format!("grid_x: 160, grid_y: {rows}, grid_z: 1")));
                    assert!(line.contains("block_x: 128, block_y: 1, block_z: 1"));
                }
                for line in launches
                    .iter()
                    .filter(|l| l.contains("hc_round_rows_r2_v1"))
                {
                    assert_eq!(rows, 32);
                    assert!(line.contains("grid_x: 160, grid_y: 16, grid_z: 1"));
                    assert!(line.contains("block_x: 256, block_y: 1, block_z: 1"));
                }
                for line in launches
                    .iter()
                    .filter(|l| l.contains("hc_finish_norm_w32_v1"))
                {
                    assert!(line.contains(&format!("grid_x: 1, grid_y: {rows}, grid_z: 1")));
                    assert!(line.contains("block_x: 1024, block_y: 1, block_z: 1"));
                }
                let (regions, collisions) = shared_overlap::audit_scratch(&lines);
                // Every exact-width target bucket uses the streamed MoE schedule.
                assert_eq!(regions, 40 * 4);
                assert_eq!(collisions, 0, "verify shared and routed scratch overlaps");
            }
            let kept_rows: Vec<_> = batch
                .iter()
                .map(|&(request, _, _, _)| kept(request, width))
                .collect();
            instance.commit_accepted(&kept_rows).unwrap();
            instance.destroy().unwrap();
            assert_eq!(host.acquired.get(), host.released.get());
        }
    }
    assert!(journal_mismatches.is_empty(), "{journal_mismatches:#?}");
    assert!(
        missing.is_empty(),
        "qualified verify tails absent: {missing:?}"
    );
}
