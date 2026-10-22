//! Block step shapes priced by their drafting sequences. Replays of
//! DeepSeek DSpark's served C16 steps at 300 W (lane OSP speed run, three
//! processes per policy).
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
