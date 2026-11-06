//! Every stage of an upper-skip lowering has a route throughout the prefill
//! envelope: each full-width stage at every row count up to 8,192, each
//! cone layer's key and output stages as a part of every row count up to
//! 4,096, and the head at every scored row count (`plan_timing::sweep`).
//! A lowering made of these calls can then fail on a row count only where
//! this sweep would. The quick and cpu tiers of `CHOPINFER_ROUTE_SWEEP`
//! sample the row counts at the route boundaries; full and exhaustive lower
//! at every one.
use super::*;
use model_deepseek41_contract::serving::{PREFILL_TOKENS, SCORED_ROWS};

/// The row counts of the configured tier: every one for full and
/// exhaustive; otherwise 1 to 64, every 128th (16th for cpu), each fused
/// norm island 64 + 128i, each side of the class thresholds and powers of
/// two, and the live row counts.
fn rows() -> Vec<u32> {
    let tier = std::env::var("CHOPINFER_ROUTE_SWEEP").unwrap_or_else(|_| "quick".into());
    let n = PREFILL_TOKENS;
    if matches!(tier.as_str(), "full" | "exhaustive") {
        return (1..=n).collect();
    }
    let step = if tier == "cpu" { 16 } else { 128 };
    let mut rows: std::collections::BTreeSet<u32> = (1..=64).collect();
    rows.extend((1..=n).filter(|r| r % step == 0));
    rows.extend((0..).map(|i| 64 + 128 * i).take_while(|&r| r <= n));
    let thresholds = [32, 48, 112, 320, 960, 1472, 2240, 2496, 2624];
    for boundary in thresholds.into_iter().chain((0..14).map(|i| 1 << i)) {
        rows.extend([boundary - 1, boundary, boundary + 1]);
    }
    rows.extend([1500, 2300, 3000]);
    rows.retain(|&r| (1..=n).contains(&r));
    rows.into_iter().collect()
}

fn sweep(dspark: bool, mapped: bool) {
    let host = host(dspark);
    host.drafter.set(dspark || mapped);
    let mut instance = create(&host, false);
    let ids: Vec<_> = (0..100_032).map(|p| token(0, p)).collect();
    plan_timing::restore_without_snapshot(instance.handle(), 1, 0, &ids).unwrap();
    let request = batch(
        &[chunk(1, 0, 100_032, u64::from(PREFILL_TOKENS))],
        &[],
        false,
    )
    .0;
    let heads: Vec<_> = (1..=SCORED_ROWS).collect();
    let rows = rows();
    let (checked, failures) =
        plan_timing::sweep(instance.handle(), &request, &rows, &heads).unwrap();
    instance.destroy().unwrap();
    // Every row count lowers full-width stages, each up to 4,096 also the
    // twenty cone layers' parts, and every head row count the five head
    // stages.
    let parts = rows.iter().filter(|&&m| m <= 4096).count();
    assert!(checked[0] >= 100 * rows.len(), "{checked:?}");
    assert!(checked[1] >= 20 * 6 * parts, "{checked:?}");
    assert!(checked[2] >= 20 * 10 * parts, "{checked:?}");
    assert_eq!(checked[3], 5 * heads.len(), "{checked:?}");
    assert!(
        failures.is_empty(),
        "dspark={dspark} mapped={mapped}: {} failures:\n{}",
        failures.len(),
        failures[..failures.len().min(40)].join("\n")
    );
}

#[test]
fn every_ar_stage_has_a_route_at_every_row_count() {
    sweep(false, false);
}

#[test]
fn every_mapped_ar_stage_has_a_route_at_every_row_count() {
    sweep(false, true);
}

#[test]
fn every_dspark_stage_has_a_route_at_every_row_count() {
    sweep(true, true);
}
