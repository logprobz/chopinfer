use super::*;

// Four, six, 12, 16 and 32 rows replace 77 five/six-launch chains with two
// launches and retain all original operands (the eight-warp chain at four
// rows, the four-warp chain at six, 12 and 16, the two-row producer at 32).
// Eight rows took the one-launch C8 boundary with a zeroed ticket, and now
// share the C8 decode bucket's column-block pair.
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
// Loading the two-strip routed down module shifts their resource ids only.
// Each layer's window KV projection binds the shared expert's workspace and
// row status and runs on the side stream beside its query chain: its
// launches change stream and those spans, and its fork and join add four
// event lines per layer and rank (640 lines).
// The row-local rings add eight resolves per rank, which precede the side
// streams and events and shift their ids by 32. At 16 rows each WQ_b fold's admit,
// partials and final launch become the ring's admit and ring (160 fewer
// lines) and WO_b's pair becomes the ring's; at 32 rows WQ_b's ticket pair
// becomes the pair loop's and WO_b's reuse pair the ring's (audited by
// `fold_local_rows::local_rows_change_only_measured_decode_and_verify_projections`).
// Loading the column-block HC pair modules shifts their resource ids only.
// At 16 and 32 rows each of the 308 producer and tail launches (77 per rank)
// becomes the column-block producer (grid 40 x rows/4, 256 threads) and the
// split finish (grid_x 1 to 2) with identical operands; no other line
// changes.
// At 8 rows each of the 308 one-launch boundaries (77 per rank) becomes the
// column-block producer (grid 40 x 2, 256 threads) and the split finish
// (grid 2 x 8, 1024 threads): the producer over the boundary's first seven
// operands and its workspace and live count, the finish over the tail's own;
// the ticket is no longer bound. Graph pieces keep their count and gain those
// nodes; no other line changes. No instance then loads the one-launch
// module, which shifts the excerpts' resource ids only.
// The cluster folds' sixteen resolves shift ids only. At 16 and 32 rows the
// WQ_a and WKV initialize/fold pairs, WO_a's pair (reuse at 32 rows) and
// WO_b's ring pair become their cluster admit/fold pairs (1280 launches,
// audited by `fold_cluster::cluster_folds_change_only_measured_decode_and_verify_projections`).
// Verify-32 and Verify-16 serve the two-strip down queue: at 32 rows each
// absorbing down small launch becomes the two-strip launch with the same
// operands (160 launches); at 16 rows each direct down small and wide pair
// becomes one two-strip launch with the small launch's operands (160 fewer
// launches and captured nodes). No other line changes.
// The status folds' eleven resolves shift ids only. At 16 and 32 rows each
// layer's twelve launches (the WQ_a, WKV, WQ_b and WO_b admit/fold pairs,
// WO_a's input quantizer and pair and WO_b's input quantizer) become six:
// the status folds of WQ_a, WKV and WQ_b, the sliced quantizer, WO_a and
// WO_b (1920 launches to 960, audited by
// `fold_fused::status_folds_change_only_native_decode_and_verify_projections`).
// Scratch regions (`buffers::REGIONS`) give each rank's region members one
// handle and renumber the other buffers and the streams; audited against
// the same verifies bound without regions, every line is otherwise equal.
// Shared fold state and workspace give each fold operation's layers one
// handle and renumber the rest; every line is otherwise equal.
// The 8-row bucket launches the 16-row status folds with 8 rows: at 8 rows
// each layer's eleven launches (the WKV and WQ_a prefetch initialize/fold
// pairs, WQ_b's stream fold, WO_a's input quantizer and prefetch pair and
// WO_b's input quantizer and prefetch pair) become six with the same
// operands, the stream fold's packed weights and state replaced by the
// layer's weights (1760 launches to 960, same audit). No other line
// changes.
// The 4-row bucket launches them with 4 rows: each layer's ten launches
// (the WKV and WQ_a prefetch initialize/fold pairs, WQ_b's stream fold,
// WO_a's input quantizer and prefetch pair, WO_b's input quantizer and
// stream fold) become six the same way (1600 launches to 960, same audit).
// The drafter no longer holds stream fold storage for its five-row blocks,
// which no stream bucket serves: create drops 15 packed copies and their
// prepare launches per rank, and the step journals only renumber handles.
const JOURNALS: [(u32, (usize, u64)); 6] = [
    (4, (7880, 0x4db8b7a58e93c470)),
    (6, (8680, 0xbeeba458db0230d5)),
    (8, (7720, 0xf2019e0f883abc13)),
    (12, (10136, 0x21a0f577b1fd0575)),
    (16, (7892, 0x831286b40e02777d)),
    (32, (7744, 0x9dda015153147e96)),
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
            // Six rows run the tail on two CTAs per row.
            let twin = crate::sm120::hc_finish_norm::TWIN_BUCKETS
                .iter()
                .find(|(bucket, _)| *bucket == rows)
                .map(|&(_, warps)| warps);
            let twin_warps = twin.unwrap_or(32);
            let twin_symbol = format!(
                "{} ",
                crate::sm120::hc_finish_twin::served().symbol(twin_warps)
            );
            // Eight, 16 and 32 rows run the column-block producer and split
            // finish (eight rows share the C8 decode bucket).
            let wide = crate::sm120::hc_cols::BUCKETS
                .iter()
                .find(|b| b.rows == rows);
            let tails = count("hc_finish_norm_w32_v1")
                + count("_hc_finish_twin_")
                + count("_hc_finish_split_");
            let boundaries = tails + count("_hc_completion_");
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
                assert_eq!(count("_hc_completion_"), 0);
                assert_eq!(
                    count("hc_finish_norm_w32_v1"),
                    if twin.is_some() || wide.is_some() {
                        0
                    } else {
                        pairs
                    }
                );
                let producer = crate::sm120::hc_cols::producer();
                let finish = crate::sm120::hc_cols::finish();
                assert_eq!(wide.is_some(), matches!(rows, 8 | 16 | 32));
                if let Some(b) = wide {
                    let first = format!("{} ", producer.symbol(b.tile, b.transposed));
                    let second = format!("{} ", finish.symbol(b.warps));
                    assert_eq!(count(&first), pairs);
                    assert_eq!(count(&second), pairs);
                    for line in launches.iter().filter(|l| l.contains(&first)) {
                        assert!(line.contains(&format!(
                            "grid_x: 40, grid_y: {}, grid_z: 1",
                            rows.div_ceil(b.tile)
                        )));
                    }
                    for line in launches.iter().filter(|l| l.contains(&second)) {
                        assert!(line.contains(&format!("grid_x: 2, grid_y: {rows}, grid_z: 1")));
                        assert!(line.contains(&format!(
                            "block_x: {}, block_y: 1, block_z: 1",
                            b.warps * 32
                        )));
                    }
                }
                assert_eq!(
                    count("_hc_round_cols_"),
                    if wide.is_some() { pairs } else { 0 }
                );
                assert_eq!(count("_hc_finish_split_"), count("_hc_round_cols_"));
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
                assert_eq!(
                    count("hc_round_chain_w4_v1"),
                    if matches!(rows, 6 | 12) { pairs } else { 0 }
                );
                assert_eq!(count("hc_round_rows_r2_v1"), 0);
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
