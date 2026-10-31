//! Export the target schedule at exact verify widths, including mixed depths.
use super::*;
use model_deepseek41_contract::execution_options::DSPARK_GRAPH_HANDLES_PER_RANK;

#[test]
fn wide_query_keeps_short_prefill_on_its_original_entry() {
    let (host, mut instance) = create_with_options(
        Shape::new(&FLASH, Ranks::TP4).unwrap(),
        16,
        128,
        &[(DSPARK_GRAPH_HANDLES_PER_RANK, 0.into())],
    );
    let from = host.journal.borrow().len();
    execute(
        &mut instance,
        &host,
        &super::super::prompt(1, 0, 0, 64, true),
    );
    let journal = host.journal.borrow();
    let symbols: Vec<_> = journal[from..]
        .iter()
        .filter(|line| line.starts_with("launch "))
        .filter_map(|line| line.split_whitespace().nth(2))
        .collect();
    assert!(symbols.contains(&"chop_deepseek41_latent_query_rope_v1"));
    assert!(!symbols.contains(&"chop_deepseek41_latent_query_rope_heads_v1"));
    drop(journal);
    instance.commit().unwrap();
    instance.retire_request(1).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

#[test]
fn target_programs_lower_at_wide_verify_rows() {
    let (host, mut instance) = create_with_options(
        Shape::new(&FLASH, Ranks::TP4).unwrap(),
        16,
        128,
        &[(DSPARK_GRAPH_HANDLES_PER_RANK, 0.into())],
    );
    // Eager execution exposes every launch, independent of the graph budget.
    for slot in 0..16 {
        execute(
            &mut instance,
            &host,
            &super::super::prompt(slot + 1, slot, 0, 3, true),
        );
        instance.commit().unwrap();
    }
    let directory = journal_directory();
    let mut cases = 0;
    for count in [8_u32, 16] {
        for rows in [24_u32, 32].into_iter().chain(33..=64) {
            // A five-position drafter cannot produce more than six rows per item.
            if rows > count * 6 {
                continue;
            }
            let (target, step, accepted) = verify_step(&mut instance, &host, count, rows);
            if rows > 32 {
                let count_symbol = |symbol: &str| {
                    target
                        .iter()
                        .filter(|line| line.split_whitespace().nth(2) == Some(symbol))
                        .count()
                };
                assert_eq!(
                    count_symbol("chop_deepseek41_latent_query_rope_heads_v1"),
                    40 * 4
                );
                assert_eq!(
                    count_symbol("chop_position_query_deepseek41_parallel_v1"),
                    8 * 4
                );
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
                assert_eq!(
                    count_symbol(
                        "chop_deepseek41_verify_f32_n512_k5120_v1_cached_decode_v1_spans_m64_compute_v1"
                    ),
                    7 * 4
                );
                assert_eq!(
                    count_symbol(
                        "chop_deepseek41_verify_f32_n32_k5120_v1_cached_decode_v1_spans_m64_compute_v1"
                    ),
                    8 * 4
                );
                assert!(
                    !target
                        .iter()
                        .any(|line| line.contains("verify_f32_n128_k512"))
                );
                for symbol in [
                    "chop_deepseek41_verify_mxfp8_n1280_k5120_g1_f32_cluster64_n64_c8s5_status_v1_packed_v1",
                    "chop_deepseek41_verify_mxfp8_n8192_k1280_g1_f32_local32_n48_s4x_status_v1_rows64_v1_packed_v1",
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
                    count_symbol(
                        "chop_deepseek41_mxfp8_fold_m8192_n25600_k6144_g1_f32_v2_matrix_m64_n128_s2_v1_packed_v1"
                    ),
                    2 * 4
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
                write_journals(directory, count, rows, &target, &step);
            }
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

/// `CHOPINFER_VERIFY_ROW_JOURNALS`, a directory outside the repository that
/// receives each exported step's journals.
fn journal_directory() -> Option<std::path::PathBuf> {
    std::env::var_os("CHOPINFER_VERIFY_ROW_JOURNALS").map(std::path::PathBuf::from)
}

/// The item widths of a `rows`-row step of `count` items: the first
/// `rows % count` items one row deeper.
fn item_widths(count: u32, rows: u32) -> Vec<u32> {
    let widths: Vec<_> = (0..count)
        .map(|slot| rows / count + u32::from(slot < rows % count))
        .collect();
    assert_eq!(widths.iter().sum::<u32>(), rows);
    widths
}

/// Executes one eager `rows`-row step over the first `count` committed
/// requests, each in its own slot, and checks its drafts. Returns the
/// target's launches (after the step's one Engram hash), the step's whole
/// journal and the commit that keeps one row of each verifying item.
fn verify_step(
    instance: &mut BundleInstanceV1,
    host: &Host,
    count: u32,
    rows: u32,
) -> (Vec<String>, Vec<String>, Vec<BundleKeptRowsV1>) {
    let (step, accepted) = verify_batch(instance, host, count, rows);
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
    (target, step, accepted)
}

/// Executes the `rows`-row step of `verify_step`, eager or with graphs,
/// and checks its drafts. Returns the step's whole journal and the commit
/// that keeps one row of each verifying item.
fn verify_batch(
    instance: &mut BundleInstanceV1,
    host: &Host,
    count: u32,
    rows: u32,
) -> (Vec<String>, Vec<BundleKeptRowsV1>) {
    let widths = item_widths(count, rows);
    let mut requests = Vec::new();
    let mut appends = Vec::new();
    let mut accepted = Vec::new();
    for (slot, &width) in widths.iter().enumerate() {
        let request = slot as u64 + 1;
        let start = state(instance, request).end;
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
    let sink = execute(instance, host, &batch);
    assert_eq!(
        sink.draft_tokens(),
        widths
            .iter()
            .enumerate()
            .filter(|(_, width)| **width > 1)
            .map(|(slot, width)| (slot as u64 + 1, vec![0; (width - 1) as usize]))
            .collect::<Vec<_>>()
    );
    let step = host.journal.borrow()[from..].to_vec();
    (step, accepted)
}

fn write_journals(
    directory: &std::path::Path,
    count: u32,
    rows: u32,
    target: &[String],
    step: &[String],
) {
    std::fs::create_dir_all(directory).unwrap();
    let stem = format!("items-{count}-rows-{rows}");
    std::fs::write(directory.join(format!("{stem}.target")), target.join("\n")).unwrap();
    std::fs::write(directory.join(format!("{stem}.journal")), step.join("\n")).unwrap();
    std::fs::write(
        directory.join(format!("{stem}.json")),
        serde_json::to_vec_pretty(&serde_json::json!({
            "items": count, "rows": rows, "item_rows": item_widths(count, rows),
            "ranks": 4, "target_launches": target.len(), "eager": true,
        }))
        .unwrap(),
    )
    .unwrap();
}

/// Each launched symbol's grids `(x, y, z)` in `target`, with their counts.
fn grids(target: &[String]) -> Grids {
    let field = |line: &str, name: &str| -> u32 {
        let (_, rest) = line
            .split_once(&format!("{name}: "))
            .unwrap_or_else(|| panic!("{name} in {line}"));
        rest[..rest.find(',').unwrap()].parse().unwrap()
    };
    let mut grids = Grids::new();
    for line in target {
        let symbol = line.split_whitespace().nth(2).unwrap().to_owned();
        let grid = (
            field(line, "grid_x"),
            field(line, "grid_y"),
            field(line, "grid_z"),
        );
        *grids.entry(symbol).or_default().entry(grid).or_default() += 1;
    }
    grids
}

type Grids = std::collections::BTreeMap<String, std::collections::BTreeMap<(u32, u32, u32), usize>>;

/// One tracked stage: the symbol that serves it, its launches in one
/// step (every layer on every rank) and its grid at the step's rows.
type Stage = (&'static str, usize, fn(u32) -> (u32, u32, u32));

/// One CTA per row.
fn per_row(rows: u32) -> (u32, u32, u32) {
    (rows, 1, 1)
}

/// One CTA per row and each of the ten HC columns.
fn per_column_row(rows: u32) -> (u32, u32, u32) {
    (10, rows, 1)
}

/// The rounding rows' power-of-two row bucket.
fn row_bucket(rows: u32) -> (u32, u32, u32) {
    (rows.next_power_of_two(), 1, 1)
}

/// The HC column-block producer: one CTA per 128 columns and four rows.
fn column_tile(rows: u32) -> (u32, u32, u32) {
    (40, rows.div_ceil(4), 1)
}

/// The HC split finish: two CTAs per row.
fn row_pair(rows: u32) -> (u32, u32, u32) {
    (2, rows, 1)
}

/// Row-local stages of a verify step above the 64-row budget. Each grids
/// the step's live rows, never a prepared capacity, so the symbol says
/// which registered entry serves the stage. With the `wide_verify` option
/// the ceil quantizers, the shared SwiGLU and the latent and index queries
/// take the decode entries of their wide verify buckets
/// (`verify_bounds::WIDE`), and the HC boundary that of a 64-row step
/// ([`WIDE_HC_STAGES`]); without it they keep the entries of
/// [`CANDIDATE_STAGES`] and the unfused boundary.
const WIDE_STAGES: &[Stage] = &[
    ("chop_deepseek41_moe_ceil_w5120_cached_v1", 320, per_row),
    (
        "chop_deepseek41_shared_swiglu_w576_parallel_v1",
        160,
        per_row,
    ),
    (
        "chop_deepseek41_dense_ceil_f32_w1280_parallel_v1",
        192,
        per_row,
    ),
    (
        "chop_deepseek41_dense_ceil_f32_w2048_parallel_v1",
        160,
        per_row,
    ),
    ("chop_deepseek41_dense_ceil_w8192_cached_v1", 160, per_row),
    ("chop_deepseek41_dense_ceil_w6144_cached_v1", 8, per_row),
    ("chop_deepseek41_latent_ceil32_parallel_v1", 160, per_row),
    ("chop_deepseek41_latent_window_rope_v1", 160, per_row),
    ("chop_deepseek41_latent_query_rope_heads_v1", 160, per_row),
    ("chop_deepseek41_q_norm_f32_v1", 160, per_row),
    ("chop_position_query_deepseek41_parallel_v1", 32, per_row),
    ("chop_row_rmsnorm_deepseek41_w5120_v1", 320, per_row),
    (
        "chop_sinkhorn_hc_deepseek41_lagged_mix_collapse_v1",
        320,
        per_row,
    ),
    (
        "chop_sinkhorn_hc_deepseek41_expand_f32_v1",
        320,
        per_column_row,
    ),
    (
        "chop_row_rounding_round_deepseek41_w5120_v1",
        324,
        row_bucket,
    ),
    (
        "chop_row_rounding_widen_deepseek41_w5120_v1",
        196,
        row_bucket,
    ),
];

/// The decode form of the HC mixer that serves 33 to 64 rows and launches
/// nowhere above them: the wide boundary keeps the one-CTA mixer.
const NARROW_STAGES: &[&str] = &["chop_sinkhorn_hc_deepseek41_lagged_mix_collapse_split_v1"];

/// With the `wide_verify` option, the HC boundary of a 64-row step: the
/// column-block pair fuses 77 of each rank's 80 boundaries, writing their
/// all-reduce round, expansion, next mixer and input norm and quantizer,
/// so those stages launch only where a 64-row step runs them unfused. The
/// mixers there keep the one-CTA entry. Each replaces the stage's pin in
/// [`WIDE_STAGES`] or adds one.
const WIDE_HC_STAGES: &[Stage] = &[
    ("chop_deepseek41_hc_round_cols_r4_tx_v1", 308, column_tile),
    ("chop_deepseek41_hc_finish_split_w32_v1", 308, row_pair),
    (
        "chop_sinkhorn_hc_deepseek41_lagged_mix_collapse_v1",
        12,
        per_row,
    ),
    (
        "chop_sinkhorn_hc_deepseek41_expand_f32_v1",
        12,
        per_column_row,
    ),
    ("chop_row_rmsnorm_deepseek41_w5120_v1", 12, per_row),
    (
        "chop_row_rounding_round_deepseek41_w5120_v1",
        16,
        row_bucket,
    ),
    ("chop_row_rounding_widen_deepseek41_w5120_v1", 8, row_bucket),
    ("chop_deepseek41_moe_ceil_w5120_cached_v1", 12, per_row),
];

/// The stages the `wide_verify` option moves to their wide verify
/// buckets' decode entries, as (decode entry, default entry), listing a
/// decode entry once for each default launch it replaces (the first is
/// pinned): without the option each keeps the prefill entry, and each form
/// launches nowhere the other serves.
const CANDIDATE_STAGES: &[(&str, &str)] = &[
    (
        "chop_deepseek41_moe_ceil_w5120_cached_v1",
        "chop_deepseek41_moe_ceil_w5120_v1",
    ),
    (
        "chop_deepseek41_shared_swiglu_w576_parallel_v1",
        "chop_deepseek41_shared_swiglu_w576_v1",
    ),
    (
        "chop_deepseek41_dense_ceil_f32_w1280_parallel_v1",
        "chop_deepseek41_dense_ceil_f32_w1280_prefill_w8_v1",
    ),
    (
        "chop_deepseek41_dense_ceil_f32_w2048_parallel_v1",
        "chop_deepseek41_dense_ceil_f32_w2048_prefill_w8_v1",
    ),
    (
        "chop_deepseek41_dense_ceil_w8192_cached_v1",
        "chop_deepseek41_dense_ceil_w8192_prefill_w16_v1",
    ),
    (
        "chop_deepseek41_dense_ceil_w6144_cached_v1",
        "chop_deepseek41_dense_ceil_w6144_prefill_w8_v1",
    ),
    (
        "chop_deepseek41_latent_ceil32_parallel_v1",
        "chop_deepseek41_latent_ceil32_v1",
    ),
    (
        "chop_deepseek41_latent_query_rope_heads_v1",
        "chop_deepseek41_latent_query_rope_v1",
    ),
    (
        "chop_position_query_deepseek41_parallel_v1",
        "chop_position_query_deepseek41_mxfp4_v2",
    ),
    (
        "chop_position_query_deepseek41_parallel_v1",
        "chop_position_query_deepseek41_rotate_v3",
    ),
];

/// The wide verify folds of a target step above the 64-row budget with the
/// `wide_verify` option, as (projection shape, launches of each of its
/// admission and compute entries in one step): every backbone projection
/// on every layer and rank (the shared gate and up share one entry), and
/// the indexer query on its layers.
const WIDE_FOLDS: &[(&str, usize)] = &[
    ("n1280_k5120_g1_single", 160),
    ("n8192_k1280_g1_single", 160),
    ("n4096_k1280_g1_single", 32),
    ("n512_k5120_g1_single", 160),
    ("n1024_k4096_g2_single", 160),
    ("n5120_k2048_g1_single", 160),
    ("n576_k5120_g1_single", 320),
    ("n5120_k576_g1_single", 160),
];

/// The Engram projection's matrix tile, which serves its verify rows above
/// 32 (two Engram layers on four ranks).
const ENGRAM_TILE: &str =
    "chop_deepseek41_mxfp8_fold_m8192_n25600_k6144_g1_f32_v2_matrix_m64_n128_s2_v1_packed_v1";

/// The dense folds of a `rows`-row target step: with the `wide_verify`
/// option each [`WIDE_FOLDS`] entry, its admission one CTA per row and
/// its compute 16-row tiles, Engram's matrix tile and no 8192-row fold;
/// without it no wide fold.
fn check_dense_folds(observed: &Grids, rows: u32, enabled: bool, what: &str) {
    let wide: std::collections::BTreeMap<_, _> = observed
        .iter()
        .filter(|(symbol, _)| symbol.contains("_rows256_"))
        .collect();
    let folds: Vec<_> = observed
        .keys()
        .filter(|symbol| symbol.contains("_mxfp8_fold_m8192_") && !symbol.contains("_matrix_"))
        .collect();
    if !enabled {
        assert!(wide.is_empty(), "{what}: {wide:?}");
        assert!(!folds.is_empty(), "{what}");
        return;
    }
    assert!(folds.is_empty(), "{what}: {folds:?}");
    assert_eq!(wide.len(), 2 * WIDE_FOLDS.len(), "{what}: {wide:?}");
    // The schedule of the narrowest wide bucket holding the step's rows.
    let bucket = model_deepseek41_contract::verify_bounds::WIDE
        .into_iter()
        .find(|&bucket| rows <= bucket)
        .expect("a wide verify bucket");
    let bases = crate::sm120::mxfp8_fold::wide_bases(&FLASH, 4).unwrap();
    for &(shape, launches) in WIDE_FOLDS {
        let stem = format!("chop_deepseek41_dense_decode_{shape}");
        let base = bases
            .iter()
            .find(|base| base.symbol.starts_with(&format!("{stem}_")))
            .unwrap_or_else(|| panic!("{what}: no wide base {stem}"));
        let entry = crate::sm120::mxfp8_fold::wide_entry(&FLASH, base, bucket, bucket)
            .unwrap()
            .symbol;
        assert!(
            entry.starts_with(&format!("{stem}_rows256_")),
            "{what}: {entry}"
        );
        let admit = &observed[&format!("{entry}_admit_v1")];
        assert_eq!(admit, &[((rows, 1, 1), launches)].into(), "{what}: {entry}");
        let compute = &observed[&format!("{entry}_packed_v1")];
        assert_eq!(compute.values().sum::<usize>(), launches, "{what}: {entry}");
        assert!(
            compute.keys().all(|grid| grid.1 == rows.div_ceil(16)),
            "{what}: {entry} {compute:?}"
        );
    }
    assert_eq!(observed[ENGRAM_TILE].values().sum::<usize>(), 8, "{what}");
}

/// The wide direct MoE queues of a `rows`-row target step with the
/// `wide_verify` option, at the package's verify rows, with their grids:
/// the pack, the absorbing gate queue, the gate finisher (one CTA per
/// routed row and each of five column groups), the two-strip down queue
/// and the down finisher (one CTA per row and each of 40 column strips).
fn wide_experts(rows: u32) -> [(&'static str, (u32, u32, u32)); 5] {
    [
        ("chop_deepseek41_moe_wide_t192_e384_k6_pack_v1", (1, 1, 1)),
        (
            "chop_deepseek41_moe_absorb_lanes_t192_e384_k6_gate_small_v1",
            (1504, 1, 1),
        ),
        (
            "chop_deepseek41_moe_bandwidth_direct_t192_e384_k6_gate_finish_v1",
            (5, 6 * rows, 1),
        ),
        (
            "chop_deepseek41_moe_strip2_t192_e384_k6_down_small_v1",
            (1504, 1, 1),
        ),
        (
            "chop_deepseek41_moe_floor_t192_e384_k6_down_finish_v1",
            (40, rows, 1),
        ),
    ]
}

/// The 8192-row expert chains those queues replace: the gate/up chain, its
/// ceil quantizer (one CTA per row), the down chain and the routed sum.
const CHAIN_EXPERTS: [&str; 4] = [
    "deepseek41_chop_moe_chain_mxfp4_a8ceil_before_gateup_e384k6_n1152_k5120_r64_t8192_clamp4120000041200000_lanes_v1",
    "chop_deepseek41_moe_ceil_w3456_v1",
    "deepseek41_chop_moe_chain_mxfp4_a8ceil_before_down_e384k6_n5120_k576_r64_t8192_v1",
    "chop_moe_deepseek41_sum_add_f32_k6_w5120_l384_t8192_v1",
];

/// The routed experts of a `rows`-row target step, once per layer on every
/// rank: with the `wide_verify` option the wide direct queues
/// ([`wide_experts`]) and no chain; without it the chains and no wide
/// queue.
fn check_routed_experts(observed: &Grids, rows: u32, enabled: bool, what: &str) {
    for (symbol, grid) in wide_experts(rows) {
        let expected = enabled.then(|| [(grid, 160)].into());
        assert_eq!(observed.get(symbol), expected.as_ref(), "{what}: {symbol}");
    }
    for symbol in CHAIN_EXPERTS {
        let launches = observed.get(symbol).map_or(0, |g| g.values().sum());
        assert_eq!(launches, if enabled { 0 } else { 160 }, "{what}: {symbol}");
    }
    if !enabled {
        assert_eq!(
            observed[CHAIN_EXPERTS[1]],
            [(per_row(rows), 160)].into(),
            "{what}"
        );
    }
}

/// Widths above the 64-row budget a step of 32 items (and 16 up to its 96
/// rows) can take: both sides of 64, 96, 128 and 160, and 192.
const WIDE_STEPS: [(u32, u32); 17] = [
    (16, 65),
    (16, 80),
    (16, 96),
    (32, 65),
    (32, 66),
    (32, 80),
    (32, 95),
    (32, 96),
    (32, 97),
    (32, 127),
    (32, 128),
    (32, 129),
    (32, 159),
    (32, 160),
    (32, 161),
    (32, 191),
    (32, 192),
];

/// Verify steps above the profile's 64-row budget lower eagerly at the
/// package's 192 verify rows (`--speculative-rows 192`), with and without
/// the `wide_verify` option. With `CHOPINFER_VERIFY_ROW_JOURNALS` each
/// step's journals are exported, under `wide/` with the option and
/// `default/` without it.
#[test]
fn target_programs_lower_above_the_default_verify_rows() {
    use model_deepseek41_contract::execution_options::WIDE_VERIFY;
    for enabled in [false, true] {
        let (host, mut instance) = create_with_options(
            Shape::new(&FLASH, Ranks::TP4).unwrap(),
            32,
            192,
            &[
                (DSPARK_GRAPH_HANDLES_PER_RANK, 0.into()),
                (WIDE_VERIFY, enabled.into()),
            ],
        );
        for slot in 0..32 {
            execute(
                &mut instance,
                &host,
                &super::super::prompt(slot + 1, slot, 0, 3, true),
            );
            instance.commit().unwrap();
        }
        let served = |symbol: &'static str| match CANDIDATE_STAGES.iter().find(|s| s.0 == symbol) {
            Some(&(_, default)) if !enabled => default,
            _ => symbol,
        };
        let hc: &[Stage] = if enabled { WIDE_HC_STAGES } else { &[] };
        let stages: Vec<Stage> = WIDE_STAGES
            .iter()
            .map(|&(symbol, launches, grid)| {
                hc.iter()
                    .find(|stage| stage.0 == symbol)
                    .copied()
                    .unwrap_or((served(symbol), launches, grid))
            })
            .chain(
                hc.iter()
                    .filter(|stage| !WIDE_STAGES.iter().any(|s| s.0 == stage.0))
                    .copied(),
            )
            .collect();
        let absent: Vec<_> = NARROW_STAGES
            .iter()
            .copied()
            .chain(
                CANDIDATE_STAGES
                    .iter()
                    .map(|&(wide, default)| if enabled { default } else { wide }),
            )
            .chain(
                WIDE_HC_STAGES
                    .iter()
                    .map(|stage| stage.0)
                    .filter(|symbol| !enabled && !WIDE_STAGES.iter().any(|s| s.0 == *symbol)),
            )
            .collect();
        let directory =
            journal_directory().map(|d| d.join(if enabled { "wide" } else { "default" }));
        for (count, rows) in WIDE_STEPS {
            let (target, step, accepted) = verify_step(&mut instance, &host, count, rows);
            if let Some(directory) = &directory {
                write_journals(directory, count, rows, &target, &step);
            }
            let observed = grids(&target);
            for &(symbol, launches, grid) in &stages {
                assert_eq!(
                    observed.get(symbol),
                    Some(&[(grid(rows), launches)].into()),
                    "{symbol} at {count} items and {rows} rows ({enabled})"
                );
            }
            for &symbol in &absent {
                assert!(
                    !observed.contains_key(symbol),
                    "{symbol} at {count} items and {rows} rows ({enabled})"
                );
            }
            let what = format!("{count} items and {rows} rows ({enabled})");
            check_dense_folds(&observed, rows, enabled, &what);
            check_routed_experts(&observed, rows, enabled, &what);
            instance.commit_accepted(&accepted).unwrap();
        }
        for request in 1..=32 {
            instance.retire_request(request).unwrap();
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

/// Wide verify steps below and above each wide bucket: 65 rows, the
/// 128-row bucket and one row past it, and the widest 192-row step.
const WIDE_GRAPH_STEPS: [u32; 4] = [65, 128, 129, 192];

/// With graphs up to the package's 192 verify rows, with and without the
/// `wide_verify` option, the first step at each width above 64 rows
/// captures its graphs and every repeat replays them without capturing
/// again. On each rank, a step's launches with each replay expanded to the
/// launches its graph captured are the launches of the eager step at the
/// same rows and option, with the same grids and arguments: each stream's
/// in eager order, together all of them (a graph runs the shared expert on
/// its rank's overlap stream). With `CHOPINFER_VERIFY_ROW_JOURNALS` each
/// step's journal is exported, under `wide/` or `default/` and then
/// `graphs/` or `eager/`.
#[test]
fn verify_graphs_above_64_rows_replay_the_eager_programs() {
    use model_deepseek41_contract::execution_options::WIDE_VERIFY;
    use model_deepseek41_contract::verify_graphs::GRAPH_ROWS;
    const REPEATS: usize = 3;
    let run = |wide: bool, graphed: bool| {
        let budget = if graphed {
            (GRAPH_ROWS, 192.into())
        } else {
            (DSPARK_GRAPH_HANDLES_PER_RANK, 0.into())
        };
        let (host, mut instance) = create_with_options(
            Shape::new(&FLASH, Ranks::TP4).unwrap(),
            32,
            192,
            &[(WIDE_VERIFY, wide.into()), budget],
        );
        for slot in 0..32 {
            execute(
                &mut instance,
                &host,
                &super::super::prompt(slot + 1, slot, 0, 3, true),
            );
            instance.commit().unwrap();
        }
        let mut graphs = Graphs::new();
        let mut programs = Vec::new();
        for repeat in 0..REPEATS {
            for rows in WIDE_GRAPH_STEPS {
                let (step, accepted) = verify_batch(&mut instance, &host, 32, rows);
                if let Some(directory) = journal_directory() {
                    let directory = directory
                        .join(if wide { "wide" } else { "default" })
                        .join(if graphed { "graphs" } else { "eager" });
                    std::fs::create_dir_all(&directory).unwrap();
                    let name = format!("rows-{rows}-repeat-{repeat}.journal");
                    std::fs::write(directory.join(name), step.join("\n")).unwrap();
                }
                let ranks = host.device.stream_ranks.borrow().clone();
                let (program, captures, replays) = replayed_program(&step, &mut graphs, &ranks);
                let what = format!("{rows} rows, repeat {repeat}, wide {wide}, graphs {graphed}");
                if graphed {
                    assert_eq!(captures > 0, repeat == 0, "{what}: {captures} captures");
                    assert!(replays > 0, "{what}");
                } else {
                    assert_eq!((captures, replays), (0, 0), "{what}");
                }
                assert_eq!(program.len(), 4, "{what}");
                assert!(
                    program.values().all(|launches| launches.len() > 1000),
                    "{what}"
                );
                programs.push((what, program));
                instance.commit_accepted(&accepted).unwrap();
            }
        }
        let stats = crate::instance::dspark_graphs_for_test(instance.handle(), None);
        if graphed {
            assert_eq!(
                (stats.denied, stats.released, stats.epoch),
                (0, 0, 0),
                "{stats:?}"
            );
        }
        for request in 1..=32 {
            instance.retire_request(request).unwrap();
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
        programs
    };
    for wide in [false, true] {
        let eager = run(wide, false);
        let graphed = run(wide, true);
        assert_eq!(eager.len(), graphed.len());
        for ((_, eager), (what, graphed)) in eager.iter().zip(&graphed) {
            for (rank, launches) in eager {
                let mut streams = std::collections::BTreeMap::<u32, Vec<&String>>::new();
                for (stream, launch) in &graphed[rank] {
                    streams.entry(*stream).or_default().push(launch);
                }
                if let Err(error) = interleaves(launches, &streams) {
                    panic!("{what}: rank {rank}: {error}");
                }
            }
        }
    }
}

/// Whether `streams` (one rank's graphed launches on each stream) together
/// hold exactly the `eager` launches, each stream's in eager order.
fn interleaves(
    eager: &[(u32, String)],
    streams: &std::collections::BTreeMap<u32, Vec<&String>>,
) -> Result<(), String> {
    let mut next: std::collections::BTreeMap<u32, usize> =
        streams.keys().map(|&stream| (stream, 0)).collect();
    for (index, (_, launch)) in eager.iter().enumerate() {
        let stream = streams
            .iter()
            .find(|(stream, launches)| launches.get(next[*stream]) == Some(&launch))
            .map(|(stream, _)| *stream)
            .ok_or_else(|| format!("eager launch {index} is no stream's next: {launch}"))?;
        *next.get_mut(&stream).unwrap() += 1;
    }
    for (stream, launches) in streams {
        if next[stream] != launches.len() {
            return Err(format!(
                "stream {stream} runs {} launches the eager step does not",
                launches.len() - next[stream]
            ));
        }
    }
    Ok(())
}

/// Each live graph's captured launch lines, by graph slot.
type Graphs = std::collections::BTreeMap<String, Vec<String>>;

/// Each rank's launches in issue order, each with its stream.
type Program = std::collections::BTreeMap<u32, Vec<(u32, String)>>;

/// The launches `step` runs on each rank (by `ranks`, each stream's rank),
/// in issue order with their stream, with each replay expanded to the
/// launches of the graph it replays. A capture records the launches of
/// its stream and of the streams that joined it through an event recorded
/// inside it (the fake device's capture rule) into `graphs`; a captured
/// launch runs only when its graph replays. Returns the program and the
/// step's captures and replays.
fn replayed_program(
    step: &[String],
    graphs: &mut Graphs,
    ranks: &std::collections::BTreeMap<u32, u32>,
) -> (Program, usize, usize) {
    use std::collections::{BTreeMap, BTreeSet};
    let mut program = Program::new();
    let mut run = |launch: &str| {
        let (stream, rest) = launch["launch ".len()..].split_once(' ').unwrap();
        let stream = stream.parse::<u32>().unwrap();
        program
            .entry(ranks[&stream])
            .or_default()
            .push((stream, rest.to_owned()));
    };
    // Each open capture's joined streams and launches, by capturing stream.
    let mut open = BTreeMap::<String, (BTreeSet<String>, Vec<String>)>::new();
    let mut events = BTreeMap::<String, String>::new();
    let (mut captures, mut replays) = (0, 0);
    let capture_of = |open: &BTreeMap<String, (BTreeSet<String>, Vec<String>)>, stream: &str| {
        open.iter()
            .find(|(capture, (members, _))| *capture == stream || members.contains(stream))
            .map(|(capture, _)| capture.clone())
    };
    for line in step {
        let fields: Vec<_> = line.split_whitespace().collect();
        match fields[..] {
            ["capture", stream] => {
                open.insert(stream.into(), Default::default());
            }
            ["record", stream, event] => match capture_of(&open, stream) {
                Some(capture) => {
                    events.insert(event.into(), capture);
                }
                None => {
                    events.remove(event);
                }
            },
            ["wait", stream, event] => {
                if let Some(capture) = events.get(event)
                    && capture != stream
                {
                    open.get_mut(capture).unwrap().0.insert(stream.into());
                }
            }
            ["captured", stream, id, nodes] => {
                let (_, launches) = open.remove(stream).expect("an open capture");
                assert_eq!(launches.len().to_string(), nodes, "{line}");
                events.retain(|_, capture| capture != stream);
                graphs.insert(id.into(), launches);
                captures += 1;
            }
            ["replay", _, id] => {
                graphs[id].iter().for_each(|launch| run(launch));
                replays += 1;
            }
            ["release", "graph", id] => {
                graphs.remove(id);
            }
            _ if fields[0] == "launch" => match capture_of(&open, fields[1]) {
                Some(capture) => open.get_mut(&capture).unwrap().1.push(line.clone()),
                None => run(line),
            },
            _ => {}
        }
    }
    assert!(open.is_empty(), "a capture stays open after the step");
    (program, captures, replays)
}
