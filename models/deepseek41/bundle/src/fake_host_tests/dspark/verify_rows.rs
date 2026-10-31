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
    let journal = host.journal.borrow();
    let step = journal[from..].to_vec();
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

/// Row-local stages of a verify step above the 64-row budget. Each grids
/// the step's live rows, never a prepared capacity, so the symbol says
/// which registered entry serves the stage. With the `wide_verify` option
/// the ceil quantizers and the shared SwiGLU take the decode entries of
/// their wide verify buckets (`verify_bounds::WIDE`); without it they keep
/// the entries of [`CANDIDATE_STAGES`].
const WIDE_STAGES: &[Stage] = &[
    ("chop_deepseek41_moe_ceil_w5120_cached_v1", 320, per_row),
    ("chop_deepseek41_moe_ceil_w3456_v1", 160, per_row),
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
    ("chop_deepseek41_latent_query_rope_v1", 160, per_row),
    ("chop_deepseek41_q_norm_f32_v1", 160, per_row),
    ("chop_position_query_deepseek41_mxfp4_v2", 32, per_row),
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

/// Decode forms of those stages that serve 33 to 64 rows and launch
/// nowhere above them.
const NARROW_STAGES: &[&str] = &[
    "chop_deepseek41_latent_query_rope_heads_v1",
    "chop_position_query_deepseek41_parallel_v1",
    "chop_deepseek41_hc_round_cols_r4_tx_v1",
    "chop_deepseek41_hc_finish_split_w32_v1",
    "chop_sinkhorn_hc_deepseek41_lagged_mix_collapse_split_v1",
];

/// The stages the `wide_verify` option moves to their wide verify
/// buckets' decode entries, as (decode entry, default entry): without the
/// option each keeps the prefill entry, and each form launches nowhere
/// the other serves.
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
];

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
        let absent: Vec<_> = NARROW_STAGES
            .iter()
            .copied()
            .chain(
                CANDIDATE_STAGES
                    .iter()
                    .map(|&(wide, default)| if enabled { default } else { wide }),
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
            for &(symbol, launches, grid) in WIDE_STAGES {
                let symbol = served(symbol);
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
            instance.commit_accepted(&accepted).unwrap();
        }
        for request in 1..=32 {
            instance.retire_request(request).unwrap();
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}
