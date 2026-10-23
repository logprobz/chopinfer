//! Block step shapes priced by their drafting sequences, and trials that
//! run their own steps before an earlier verdict stops them. Replays of
//! DSpark's served C16 and C8 steps at 300 W (lane OSP speed run,
//! three processes per policy) and of a C1 cross-model harness cohort.
use super::*;

/// A DSpark-like policy at `items` items: K5 blocks, a 32-row verify cap,
/// exact rows, AR steps of `ar` microseconds at these items and 9 ms at one.
fn dspark(items: usize, ar: u64) -> SpeculationPolicy {
    let mut policy = SpeculationPolicy::new(
        SpeculationConfig::new(5, 32, SpeculationMode::Adaptive, DraftSource::Bundle)
            .unwrap()
            .with_draft_cost_model(DraftCostModel::PerBlock),
    )
    .with_step_rows(|rows| rows)
    .with_diagnostics(true);
    for _ in 0..3 {
        policy.observe_step(&[0], 0, 9_000);
        policy.observe_step(&vec![0; items], 0, ar);
    }
    policy
}

/// `drafting` of `items` items draft `each` tokens.
fn drafts(items: usize, drafting: usize, each: usize) -> Vec<usize> {
    let mut drafts = vec![0; items];
    drafts[..drafting].fill(each);
    drafts
}

fn verified(positions: f64) -> AcceptanceEstimate {
    AcceptanceEstimate {
        unverified: false,
        positions: [positions; MAX_DRAFT_TOKENS],
    }
}

fn candidates(acceptance: &AcceptanceEstimate, items: usize) -> Vec<PlanCandidate<'_>> {
    vec![
        PlanCandidate {
            maximum_draft_tokens: 5,
            acceptance,
            fresh: false,
            remaining_tokens: usize::MAX,
        };
        items
    ]
}

/// At C16 every process drafted the timed cohort in the 32-row bucket. Base
/// processes settled on four drafters of four drafts, 37.0 ms a step; the
/// trial's processes on eight drafters of two, 40.1 ms, with as many
/// accepted drafts: 698 against 722 to 750 tok/s. Priced by rows alone, the
/// bucket costs the same either way, and eight shallow drafters expect more
/// tokens than four deep ones.
#[test]
fn fewer_deeper_drafters_win_where_drafting_sequences_cost_more() {
    let acceptance = verified(0.85);
    for steady in [false, true] {
        let mut policy = dspark(16, 24_800);
        // Earlier steady steps of four drafters, then the most recent of
        // eight. Without steady counts (probes), only the bucket's average
        // is known.
        for (sequences, micros) in [(4, 37_000), (8, 40_100)] {
            let shape = drafts(16, sequences, 16 / sequences);
            for _ in 0..4 {
                if !steady {
                    policy.probing = Some(16);
                }
                policy.observe_step(&shape, 0, micros);
            }
        }
        policy.zero_draft[16].disabled = false;
        let plan = policy.plan(&candidates(&acceptance, 16), DraftBudget::new(16, 16));
        let cost = |sequences| {
            policy.step_cost(
                16,
                DraftShape::from_counts(&drafts(16, sequences, 16 / sequences)),
                0.0,
            )
        };
        let (four, eight) = (cost(4), cost(8));
        println!("C16_SHAPE steady={steady} {plan:?} four={four} eight={eight}");
        if steady {
            // Fitted to both counts and the prior: about 0.7 ms a sequence.
            assert!((37_000.0..37_500.0).contains(&four), "{four}");
            assert!((39_500.0..40_500.0).contains(&eight), "{eight}");
            assert_eq!(plan, drafts(16, 4, 4));
        } else {
            assert_eq!(four, eight);
            assert_eq!(plan, drafts(16, 8, 2));
        }
    }
}

/// C8 under replacement load: a 139-step drafting average had settled just
/// above AR (485.5 against 480 tok/s) when drafting resumed. Its first step,
/// with low acceptance, tipped the average below AR, and drafting stayed off
/// at eight items for the rest of the window although the trial's next steps
/// beat AR.
#[test]
fn a_trial_runs_its_own_steps_before_an_earlier_verdict_stops_it() {
    let acceptance = verified(0.55);
    let mut policy = dspark(8, 16_600);
    let shape = drafts(8, 6, 4);
    for _ in 0..3 {
        policy.observe_step(&shape, 0, 34_500);
    }
    // Every bucket ran before, so the first step counts.
    for rows in 9..=32 {
        policy.warm_shapes.insert((8, rows, DraftShape::default()));
    }
    policy.realized.insert(
        8,
        Realized {
            ar: Some(480.0e-6),
            drafting: Some(485.5e-6),
            drafting_weight: 1.0,
            drafting_steps: 139,
        },
    );
    policy.zero_draft[8].disabled = true;
    // The trial's first step accepted 5 of 24 drafts in 39.3 ms, the next
    // ones 11 in 34.5 ms.
    let mut reasons = Vec::new();
    for step in 0..16 {
        let plan = policy.plan(&candidates(&acceptance, 8), DraftBudget::new(24, 24));
        reasons.push(
            policy.decision().unwrap().record()["reason"]
                .as_str()
                .unwrap()
                .to_owned(),
        );
        let drafted = plan.iter().any(|&count| count > 0);
        let (accepted, micros) = match (drafted, step) {
            (false, _) => (0, 16_600),
            (true, 0) => (5, 39_300),
            (true, _) => (11, 34_500),
        };
        policy.step_accepted = accepted;
        policy.observe_step(&plan, 0, micros);
    }
    println!("C8_STALE {reasons:?} {:?}", policy.realized[&8]);
    // Every plan drafts: the first under the earlier verdict, the next
    // seven as the rest of the trial, though the first step tipped that
    // verdict below AR.
    assert!(
        reasons.iter().all(|r| r == "drafting" || r == "trial"),
        "{reasons:?}"
    );
    // The trial's steps lift the same average back above AR.
    let realized = policy.realized[&8];
    assert!(
        realized.holding() && realized.drafting_steps == 139 + 16,
        "{realized:?}"
    );
}

/// A block C1 cohort where drafting wins by about a third (cross-model
/// harness, diverse target): three probes had lifted a 278-step verdict back
/// above AR when drafting resumed. The first step paid its catch-up, and
/// the request's estimate then priced drafting below the trial band. The
/// verdict, still holding, carries the trial; a trial that dropped it ended
/// there, and C1 stayed at AR for the rest of the cohort.
#[test]
fn a_trial_under_a_verdict_that_holds_drafts_through_a_dip_in_its_estimates() {
    // AR steps of about 8.7 ms, 115 tok/s.
    let mut policy = dspark(1, 8_560);
    for _ in 0..3 {
        policy.observe_step(&[5], 0, 26_000);
    }
    for rows in 2..=6 {
        policy.warm_shapes.insert((1, rows, DraftShape::default()));
    }
    policy.realized.insert(
        1,
        Realized {
            ar: Some(115.0e-6),
            drafting: Some(147.0e-6),
            drafting_weight: 1.0,
            drafting_steps: 278,
        },
    );
    policy.zero_draft[1].disabled = true;
    // Rejected drafts price drafting below the band, so only the verdict
    // keeps the plan.
    let dip = verified(0.01);
    let mut reasons = Vec::new();
    for step in 0..8 {
        let plan = policy.plan(&candidates(&dip, 1), DraftBudget::new(5, 31));
        reasons.push(
            policy.decision().unwrap().record()["reason"]
                .as_str()
                .unwrap()
                .to_owned(),
        );
        let drafted = plan[0] > 0;
        let (accepted, micros) = match (drafted, step) {
            (false, _) => (0, 8_560),
            (true, 0) => (1, 30_000),
            (true, _) => (3, 26_000),
        };
        policy.step_accepted = accepted;
        policy.observe_step(&plan, 0, micros);
    }
    println!("C1_HOLDING {reasons:?} {:?}", policy.realized[&1]);
    assert!(reasons.iter().all(|r| r == "drafting"), "{reasons:?}");
    assert!(policy.realized[&1].holding(), "{:?}", policy.realized[&1]);
}
