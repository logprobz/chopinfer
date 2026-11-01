//! The `router_prepare_fusion` candidate: at decode and verify widths of at
//! most 32 rows, each backbone layer's router finish prepare and the router
//! producer after it become one launch of the producer that classifies its
//! own rows: the producer's arguments followed by the prepare's ticket, in
//! eight CTAs of five warps, one cluster per token block and feature tile.
//! Every other launch keeps its symbol, grid and arguments; the drafter's
//! router and steps above 32 rows keep both launches.
use super::merge_ceil::same;
use super::*;
use model_deepseek41_contract::execution_options::{
    DSPARK_GRAPH_HANDLES_PER_RANK, ROUTER_PREPARE_FUSION,
};

const PRODUCER: &str = "chop_deepseek41_dense_f32_m8192_n384_k5120_v1_cached_decode_v1";
const CLASSIFIED: &str =
    "chop_deepseek41_dense_f32_m8192_n384_k5120_v1_cached_decode_classified_v1";

fn prepare(symbol: &str) -> bool {
    symbol.contains("_router_finish_") && symbol.ends_with("_prepare_v1")
}

/// `served`'s launch lines with each prepare and the backbone router
/// producer after it folded into one classifying launch. Returns the lines
/// and the number of folds.
fn folded(served: &[String]) -> (Vec<String>, usize) {
    use super::super::query_window::{arguments, symbol};
    let mut lines: Vec<String> = Vec::with_capacity(served.len());
    let mut folds = 0;
    let mut at = 0;
    while at < served.len() {
        let line = &served[at];
        if prepare(symbol(line))
            && let Some(next) = served.get(at + 1)
            && symbol(next) == PRODUCER
        {
            // Input, live count, rows, workspace, ticket.
            let ticket = arguments(line)[4].clone();
            let mut items = arguments(next);
            let head = &next[..next.len() - items.join(", ").len() - 2];
            items.push(ticket);
            let config = head
                .replacen(PRODUCER, CLASSIFIED, 1)
                .replacen("grid_z: 10,", "grid_z: 8,", 1)
                .replacen("block_x: 128,", "block_x: 160,", 1)
                .replacen(
                    "cluster: LaunchClusterV1 { x: 1, y: 1, z: 1 }",
                    "cluster: LaunchClusterV1 { x: 1, y: 1, z: 8 }",
                    1,
                );
            lines.push(format!("{config}[{}]", items.join(", ")));
            folds += 1;
            at += 2;
            continue;
        }
        lines.push(line.clone());
        at += 1;
    }
    (lines, folds)
}

fn count(lines: &[String], name: &str) -> usize {
    use super::super::query_window::symbol;
    lines.iter().filter(|l| symbol(l) == name).count()
}

#[test]
fn classifying_router_replaces_the_prepare_and_producer_at_verify_widths() {
    let create = |enabled: bool| {
        create_with_options(
            Shape::new(&FLASH, Ranks::TP4).unwrap(),
            16,
            128,
            &[
                (DSPARK_GRAPH_HANDLES_PER_RANK, 0.into()),
                (ROUTER_PREPARE_FUSION, enabled.into()),
            ],
        )
    };
    let (served_host, mut served) = create(false);
    let (fused_host, mut fused) = create(true);
    for (host, instance) in [(&served_host, &mut served), (&fused_host, &mut fused)] {
        for slot in 0..16 {
            execute(
                instance,
                host,
                &super::super::prompt(slot + 1, slot, 0, 3, true),
            );
            instance.commit().unwrap();
        }
    }
    let launches = |step: &[String]| -> Vec<String> {
        step.iter()
            .filter(|line| line.starts_with("launch "))
            .cloned()
            .collect()
    };
    for (items, rows) in [(1_u32, 6_u32), (2, 12), (8, 32), (16, 33), (16, 64)] {
        let (a, step_a, accepted_a) =
            super::verify_rows::verify_step(&mut served, &served_host, items, rows);
        let (b, step_b, accepted_b) =
            super::verify_rows::verify_step(&mut fused, &fused_host, items, rows);
        let (expected, folds) = folded(&launches(&step_a));
        same(&expected, &launches(&step_b), &format!("{rows} rows"));
        // Only the target's 40 layers on each of the 4 ranks fold.
        let (target, target_folds) = folded(&a);
        same(&target, &b, &format!("{rows} target rows"));
        let routers = if rows <= 32 { 40 * 4 } else { 0 };
        assert_eq!((folds, target_folds), (routers, routers), "{rows} rows");
        assert_eq!(count(&b, CLASSIFIED), routers, "{rows} rows");
        if rows <= 32 {
            assert_eq!(count(&b, PRODUCER), 0, "{rows} rows");
        }
        served.commit_accepted(&accepted_a).unwrap();
        fused.commit_accepted(&accepted_b).unwrap();
    }
    for (host, instance) in [(&served_host, &mut served), (&fused_host, &mut fused)] {
        for request in 1..=16 {
            instance.retire_request(request).unwrap();
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

/// Autoregressive decode steps of 1, 2 and 8 rows, and the cubin loads:
/// only an instance with the option loads the classifying producer.
#[test]
fn classifying_router_replaces_the_prepare_and_producer_at_decode() {
    let run = |enabled: bool| {
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let host = FakeHost::new(shape, KV_LAYOUTS[0]);
        let mut instance =
            super::super::create_with_options(&host, 8, &[(ROUTER_PREPARE_FUSION, enabled.into())]);
        let loaded = host
            .device
            .module_cubins
            .borrow()
            .values()
            .any(|name| *name == crate::sm120::router_classified::CUBIN);
        assert_eq!(loaded, enabled);
        for slot in 0..8 {
            execute(
                &mut instance,
                &host,
                &super::super::prompt(slot + 1, slot, 0, 1, true),
            );
            instance.commit().unwrap();
        }
        let mut ends = [1_u64; 8];
        let mut steps = Vec::new();
        for rows in [1_u32, 2, 8] {
            let from = host.device.journal.borrow().len();
            let items = (0..rows)
                .map(|slot| (u64::from(slot) + 1, u64::from(slot), ends[slot as usize], 7))
                .collect::<Vec<_>>();
            execute(&mut instance, &host, &super::super::decode(&items));
            instance.commit().unwrap();
            steps.push(
                host.device.journal.borrow()[from..]
                    .iter()
                    .filter(|l| l.starts_with("launch "))
                    .cloned()
                    .collect::<Vec<_>>(),
            );
            for end in &mut ends[..rows as usize] {
                *end += 1;
            }
        }
        for id in 1..=8 {
            instance.retire_request(id).unwrap();
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
        steps
    };
    let (served, fused) = (run(false), run(true));
    for (rows, (a, b)) in [1, 2, 8].into_iter().zip(served.iter().zip(&fused)) {
        let (expected, folds) = folded(a);
        same(&expected, b, &format!("{rows} decode rows"));
        assert_eq!(folds, 40 * 4, "{rows} rows");
        assert_eq!(count(b, CLASSIFIED), folds);
    }
}
