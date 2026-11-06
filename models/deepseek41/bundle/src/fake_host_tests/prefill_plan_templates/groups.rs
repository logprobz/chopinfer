//! An upper-skip body lowered one group at a time: each group adds whole
//! layers' launches on every rank, the last group ends the body, and, in a
//! release build with `CHOPINFER_PLAN_GROUP_BENCH=1`, each group's host
//! time at the live row counts: `cargo test --release -p bundle-deepseek41
//! --lib prefill_plan_templates::groups -- --nocapture`.
use super::*;

#[test]
fn each_group_lowers_whole_layers_on_every_rank() {
    let timed = std::env::var("CHOPINFER_PLAN_GROUP_BENCH").is_ok_and(|v| v == "1");
    let repeats = if timed { 21 } else { 1 };
    for dspark in [false, true] {
        let host = host(dspark);
        host.drafter.set(true);
        let mut instance = create(&host, false);
        let ids: Vec<_> = (0..100_032).map(|p| token(0, p)).collect();
        plan_timing::restore_without_snapshot(instance.handle(), 1, 0, &ids).unwrap();
        for rows in [512, 1500, 2300, 3000] {
            let request = batch(&[chunk(1, 0, 100_032, rows)], &[], false).0;
            let plan_timing::InGroups {
                groups,
                stages,
                launches,
            } = plan_timing::lower_in_groups(instance.handle(), &request, repeats)
                .unwrap_or_else(|e| panic!("dspark={dspark} rows={rows}: {e}"));
            let what = format!("dspark={dspark} rows={rows}");
            let mut start = vec![0; launches.len()];
            for group in &groups {
                for (rank, (&from, &to)) in start.iter().zip(&group.launches).enumerate() {
                    assert!(from <= to, "{what}: rank {rank} lost launches");
                }
                // Rank 0's launches of the group all belong to its layer.
                let layers = stages[start[0]..group.launches[0]]
                    .iter()
                    .map(|stage| stage.layer())
                    .collect::<std::collections::BTreeSet<_>>();
                assert!(
                    layers.iter().all(|&layer| layer == group.layer),
                    "{what}: group of layer {:?} lowered layers {layers:?}",
                    group.layer
                );
                start.clone_from(&group.launches);
            }
            assert_eq!(start, launches, "{what}: the last group ends the body");
            if timed {
                let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
                let total: std::time::Duration = groups.iter().map(|g| g.elapsed).sum();
                let row = groups
                    .iter()
                    .map(|g| format!("{:?}={:.3}", g.layer, ms(g.elapsed)))
                    .collect::<Vec<_>>()
                    .join(" ");
                println!(
                    "{what}: {:.3} ms in {} groups: {row}",
                    ms(total),
                    groups.len()
                );
            }
        }
        instance.destroy().unwrap();
    }
}
