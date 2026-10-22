use super::*;

// Four, six, 12, 16 and 32 rows replace 77 five/six-launch chains with two
// launches and retain all original operands (the eight-warp chain at four
// rows, the four-warp chain at six, 12 and 16, the two-row producer at 32).
// Eight rows take the qualified one-launch C8 boundary with a zeroed ticket.
// Six rows run each tail on two CTAs per row (fold and Sinkhorn beside the
// collapse and norm) with the tail's own operands.
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
// The prefill matrix modules, the fused candidate publisher's create-time
// module and resolve, the prefill arena and the verify-4 grouped catalog
// only renumber resources in these verify bodies; every normalized line is
// otherwise equal. Full lifetime audits retain every operation at these
// verify widths.
// The one-row Markov step's early-staged weight tiles and the row-parallel
// Engram hash module are created before verify and renumber its buffer,
// stream and event handles; numbered by first use, all four journals equal
// the previous ones.
// The early-stop selectors' module and six resolves per rank do the same;
// verify launches keep the served selectors. The DSpark successor's
// create-time resources renumber handles as well. The r2 router finish
// module renumbers handles again. At 32 rows its 160 finish launches change
// symbol to r2v7 with identical config and operands, and each sublayer's join
// (record and wait) moves from before the routed sum to directly after that
// finish; no other line changes.
// Staged-scale resources renumber handles. Verify bodies remain identical;
// full creation histories differ only by audited short prefill substitutions.
// The wide gate/up and gate finisher loads modules renumber handles only.
// Loading the one-launch boundary module shifts their resource ids only.
// Loading the one-launch boundary and cached query norm modules shifts their
// resource ids only.
// Combined C1 query/KV adds private storage before capture. All six verify
// bodies remain identical after typed handle mapping; no verify uses it.
// At 4 and 32 rows the routed gate after each r2 join launches without
// programmatic serialization: 160 gate launches change flags 1 to 0.
// The absorbing small queues at 32 rows rename the 160 gate and down small
// launches and drop the 320 wide launches; the absorb module renumbers
// handles only at the other rows.
// The local C32 query module shifts resource identities only. These verify
// launches and their operands retain the reference path at every row count.
// Verify-6 serves the same C1 stack: per stream the same launches with the
// same arguments, 160 router finishes renamed to r2v7 and 160 gate
// finishers to the all-loads entry (same configuration), 160 gate small
// launches to resident8 (same grid, flags 1 to 0); the gate launches move
// after each join. Verify-16 also fuses the router: each layer's five
// unfused router launches become the fused prepare, the same producer launch
// and the r2v7 finish (160 groups, 320 fewer captured nodes); the routed
// launches bind the same workspace spans 983552 bytes later (the fused
// router's partials precede the route metadata); plus the Verify-6 renames.
// Loading the two-CTA HC finish module shifts their resource ids only.
// At 6 rows its 308 tails (77 per rank) change symbol to the 32-warp twin and
// grid_x 1 to 2 with identical operands; no other line changes.
const JOURNALS: [(u32, (usize, u64)); 6] = [
    (4, (7880, 0x74213e2192c807be)),
    (6, (8040, 0xab479619617d88d2)),
    (8, (7572, 0xeb105d9fa1790b93)),
    (12, (9496, 0xf4fbfc520a20219f)),
    (16, (8532, 0x2b622bcd786b272c)),
    (32, (8064, 0xeac8275f65d0eacd)),
];

#[test]
fn verify_hc_fuses_only_measured_rows_and_keeps_scratch_disjoint() {
    let mut missing = Vec::new();
    let mut journal_mismatches = Vec::new();
    for rows in [4u32, 6, 8, 12, 16, 32] {
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
            let qualified = matches!(rows, 4 | 6 | 8 | 12 | 16 | 32);
            let pairs = if qualified { 77 * 4 } else { 0 };
            // Eight rows take the one-launch boundary of the C8 decode bucket;
            // six rows run the tail on two CTAs per row.
            let twin = crate::sm120::hc_finish_norm::TWIN_BUCKETS
                .iter()
                .find(|(bucket, _)| *bucket == rows)
                .map(|&(_, warps)| warps);
            let twin_warps = twin.unwrap_or(32);
            let twin_symbol = format!(
                "{} ",
                crate::sm120::hc_finish_twin::served().symbol(twin_warps)
            );
            let tails = count("hc_finish_norm_w32_v1") + count("_hc_finish_twin_");
            let boundaries = tails + count("hc_completion_split_w8_v1 ");
            if boundaries != pairs {
                missing.push((rows, kv.storage, boundaries, pairs));
            }
            eprintln!(
                "VERIFY_HC_LAUNCHES rows={rows} kv={:?} per_rank={} tails={} producers={}",
                kv.storage,
                launches.len() / 4,
                count("hc_finish_norm_w32_v1") / 4,
                count("hc_round_chain_w4_v1") / 4
            );
            if boundaries == pairs {
                assert_eq!(
                    count("hc_completion_split_w8_v1 "),
                    if rows == 8 { pairs } else { 0 }
                );
                assert_eq!(
                    count("hc_finish_norm_w32_v1"),
                    if rows == 8 || twin.is_some() {
                        0
                    } else {
                        pairs
                    }
                );
                assert_eq!(twin.is_some(), rows == 6);
                assert_eq!(count(&twin_symbol), if rows == 6 { pairs } else { 0 });
                assert_eq!(count("_hc_finish_twin_"), count(&twin_symbol));
                for line in launches.iter().filter(|l| l.contains(&twin_symbol)) {
                    assert!(line.contains(&format!("grid_x: 2, grid_y: {rows}, grid_z: 1")));
                    assert!(line.contains(&format!(
                        "block_x: {}, block_y: 1, block_z: 1",
                        twin_warps * 32
                    )));
                }
                for line in launches
                    .iter()
                    .filter(|l| l.contains("hc_completion_split_w8_v1 "))
                {
                    assert!(line.contains("grid_x: 161, grid_y: 8, grid_z: 1"));
                    assert!(line.contains("block_x: 256, block_y: 1, block_z: 1"));
                }
                assert_eq!(
                    count("hc_round_chain_w4_v1"),
                    if matches!(rows, 6 | 12 | 16) {
                        pairs
                    } else {
                        0
                    }
                );
                assert_eq!(
                    count("hc_round_rows_r2_v1"),
                    if rows == 32 { pairs } else { 0 }
                );
                assert_eq!(
                    count("hc_round_chain_w8_v1"),
                    if rows == 4 { pairs } else { 0 }
                );
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
                for (symbol, threads) in
                    [("hc_round_chain_w4_v1", 128), ("hc_round_chain_w8_v1", 256)]
                {
                    for line in launches.iter().filter(|l| l.contains(symbol)) {
                        assert!(line.contains(&format!("grid_x: 160, grid_y: {rows}, grid_z: 1")));
                        assert!(
                            line.contains(&format!("block_x: {threads}, block_y: 1, block_z: 1"))
                        );
                    }
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
