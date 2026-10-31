//! The `phased_decode_reduce` candidate on decode steps: the complete
//! all-reduces take the phased F32 rounding entries, the ones whose round
//! the next HC boundary defers keep the composed pair.
use super::*;
use model_deepseek41_contract::execution_options::PHASED_DECODE_REDUCE;

const ONE_SHOT: &str = "chop_comm_allreduce_phased_oneshot_round_f32_v1 ";
const TWO_SHOT: &str = "chop_comm_allreduce_phased_round_f32_v1 ";
const SCATTER: &str = "chop_comm_reduce_scatter_f32_v1 ";
const GATHER: &str = "chop_comm_allgather_scattered_f32_v1 ";
const ROUND: &str = "chop_row_rounding_round_deepseek41_w5120_v1 ";
/// Each rank's hidden all-reduces per decode step, and those whose round
/// the next HC boundary takes (`wide_hc::decode_serves_measured_wide_hc_pairs`).
const REDUCTIONS: usize = 81;
const DEFERRED: usize = 77;

fn instance(host: &Rc<Host>, items: u32, enabled: bool) -> BundleInstanceV1 {
    create_with_options(host, items, &[(PHASED_DECODE_REDUCE, enabled.into())])
}

/// One decode step of `rows` requests after their prompts; its journal.
fn decode_step(host: &Rc<Host>, instance: &mut BundleInstanceV1, rows: u32) -> Vec<String> {
    for slot in 0..u64::from(rows) {
        step(
            instance,
            &prompt(7 + slot, slot, 0, 3, true),
            &mut Views::default(),
        );
    }
    let start = host.device.journal.borrow().len();
    let items = (0..u64::from(rows))
        .map(|slot| (7 + slot, slot, 3, 9))
        .collect::<Vec<_>>();
    step(instance, &decode(&items), &mut Views::default());
    host.device.journal.borrow()[start..].to_vec()
}

fn count(journal: &[String], symbol: &str) -> usize {
    journal
        .iter()
        .filter(|line| line.starts_with("launch ") && line.contains(symbol))
        .count()
}

/// With the option, the four complete reductions per rank of every decode
/// bucket launch the phased entry of their rows (the one-shot round at one
/// to four rows); the 77 deferred ones launch the composed pair, as
/// without it, and no standalone round remains.
#[test]
fn complete_decode_reductions_take_the_phased_entries() {
    for rows in [1_u32, 2, 4, 8, 16, 32] {
        for enabled in [false, true] {
            let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
            let mut instance = instance(&host, rows, enabled);
            let journal = decode_step(&host, &mut instance, rows);
            let complete = 4 * (REDUCTIONS - DEFERRED);
            let phased = if !enabled { 0 } else { complete };
            let (one_shot, two_shot) = if rows <= 4 { (phased, 0) } else { (0, phased) };
            assert_eq!(count(&journal, ONE_SHOT), one_shot, "{rows} {enabled}");
            assert_eq!(count(&journal, TWO_SHOT), two_shot, "{rows} {enabled}");
            let composed = 4 * DEFERRED + complete - phased;
            assert_eq!(count(&journal, SCATTER), composed, "{rows} {enabled}");
            assert_eq!(count(&journal, GATHER), composed, "{rows} {enabled}");
            assert_eq!(
                count(&journal, ROUND),
                complete - phased,
                "{rows} {enabled}"
            );
            instance.destroy().unwrap();
            assert_eq!(host.acquired.get(), host.released.get());
        }
    }
}

/// The option changes no prefill: native prefills keep their BF16
/// partials, and no prefill launches a phased F32 entry.
#[test]
fn prefill_keeps_its_reductions_with_the_option() {
    let journals = [false, true].map(|enabled| {
        let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
        let mut instance = instance(&host, 4, enabled);
        let start = host.device.journal.borrow().len();
        step(
            &mut instance,
            &prompt(7, 0, 0, 600, true),
            &mut Views::default(),
        );
        let journal = host.device.journal.borrow()[start..].to_vec();
        instance.destroy().unwrap();
        journal
    });
    for journal in &journals {
        assert_eq!(count(journal, ONE_SHOT) + count(journal, TWO_SHOT), 0);
    }
    let launches = |journal: &[String]| {
        journal
            .iter()
            .filter(|line| line.starts_with("launch "))
            .filter_map(|line| line.split_whitespace().nth(2).map(str::to_owned))
            .collect::<Vec<_>>()
    };
    assert_eq!(launches(&journals[0]), launches(&journals[1]));
}

/// Without proven residency for the phased grids, decode keeps the
/// composed form with the option on.
#[test]
fn decode_falls_back_on_unproven_residency() {
    for kind in 0..4 {
        let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[1]);
        let mut value = BundleRawClusterResidencyV1 {
            multiprocessor_count: 188,
            usable_multiprocessors: 188,
            max_active_clusters: 188,
            flags: BUNDLE_CLUSTER_RESIDENCY_V1_CLUSTERS_REPORTED
                | BUNDLE_CLUSTER_RESIDENCY_V1_COOPERATIVE_LAUNCH,
            ..BundleRawClusterResidencyV1::unreported()
        };
        match kind {
            0 => value = BundleRawClusterResidencyV1::unreported(),
            1 => value.max_active_clusters = 0,
            2 => value.flags |= BUNDLE_CLUSTER_RESIDENCY_V1_MPS_CLIENT,
            _ => value.usable_multiprocessors = 1,
        }
        host.device.residency.set(Some(value));
        let mut instance = instance(&host, 1, true);
        let journal = decode_step(&host, &mut instance, 1);
        assert_eq!(count(&journal, ONE_SHOT) + count(&journal, TWO_SHOT), 0);
        assert_eq!(count(&journal, SCATTER), 4 * REDUCTIONS);
        assert_eq!(count(&journal, ROUND), 4 * (REDUCTIONS - DEFERRED));
        instance.destroy().unwrap();
    }
}

/// Every `RESET_RUNS` executes, the next one drains every rank, zeroes
/// exactly the phased decode reduction's state regions and drains again
/// before its first launch.
#[test]
fn phased_decode_resets_only_its_protocol_before_an_execute() {
    use bundle_runtime::entry::Bundle;
    let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
    let mut instance = instance(&host, 1, true);
    step(
        &mut instance,
        &prompt(7, 0, 0, 3, true),
        &mut Views::default(),
    );
    let spans = <crate::instance::Instance as Bundle>::instances().with(|table| {
        table
            .borrow_mut()
            .get_mut(instance.handle())
            .unwrap()
            .reset_phased_decode_next_run()
    });
    // Inbox, flags and epoch on each of four ranks.
    assert_eq!(spans.len(), 12);
    host.device.journal.borrow_mut().clear();
    step(
        &mut instance,
        &decode(&[(7, 0, 3, 9)]),
        &mut Views::default(),
    );
    let journal = host.device.journal.borrow();
    let first_launch = journal
        .iter()
        .position(|s| s.starts_with("launch "))
        .unwrap();
    let reset = &journal[..first_launch];
    let first_zero = reset.iter().position(|s| s.starts_with("zero ")).unwrap();
    assert_eq!(
        reset[..first_zero]
            .iter()
            .filter(|s| s.starts_with("synchronize "))
            .count(),
        4
    );
    let actual = reset[first_zero..first_zero + spans.len()]
        .iter()
        .flat_map(|s| super::shared_overlap::journal_spans(s))
        .collect::<Vec<_>>();
    assert_eq!(
        actual,
        spans
            .iter()
            .map(|span| (span.buffer.raw, span.offset, span.length))
            .collect::<Vec<_>>()
    );
    assert!(
        reset[first_zero + spans.len()..first_zero + spans.len() + 4]
            .iter()
            .all(|s| s.starts_with("synchronize "))
    );
    drop(journal);
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}
