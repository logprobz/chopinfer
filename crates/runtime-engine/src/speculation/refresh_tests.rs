//! Cost-feedback regressions; recorded samples are supplied outside the tree.
use super::*;

fn block_policy() -> SpeculationPolicy {
    SpeculationPolicy::new(
        SpeculationConfig::new(5, 32, SpeculationMode::Adaptive, DraftSource::Bundle)
            .unwrap()
            .with_draft_cost_model(DraftCostModel::PerBlock),
    )
    .with_diagnostics(true)
}

fn seed(policy: &mut SpeculationPolicy, ar: f64, block: f64, sequences: usize) {
    policy.costs.micros.insert(16, ar);
    policy.block_steps.record((16, 32), block, sequences, 1.0);
    policy.fresh_costs.insert(16, 0.0);
    policy.warm_shapes.insert((16, 32, DraftShape::default()));
    policy.warm_shapes.insert((16, 0, DraftShape::default()));
}

fn candidates(acceptance: &AcceptanceEstimate, fresh: bool) -> Vec<PlanCandidate<'_>> {
    vec![
        PlanCandidate {
            maximum_draft_tokens: 1,
            acceptance,
            fresh,
            remaining_tokens: 192,
        };
        16
    ]
}

#[test]
fn a_known_block_probe_gets_two_caught_up_rechecks_before_backoff() {
    let mut policy = block_policy();
    seed(&mut policy, 90_000.0, 185_000.0, 8);
    let estimate = AcceptanceEstimate::initial();
    let fresh = candidates(&estimate, true);
    let caught_up = candidates(&estimate, false);
    policy.zero_draft[16].idle = PROBE_INTERVAL;
    assert_eq!(policy.plan(&fresh, DraftBudget::new(16, 16)), [1; 16]);
    policy.step_accepted = 10;
    policy.observe_step(&[1; 16], 16, 209_000);
    for _ in 0..BLOCK_RECHECKS {
        assert_eq!(policy.plan(&caught_up, DraftBudget::new(16, 16)), [1; 16]);
        assert_eq!(policy.decision().unwrap().record()["reason"], "probe");
        policy.step_accepted = 10;
        policy.observe_step(&[1; 16], 0, 64_000);
    }
    assert_eq!(policy.zero_draft[16].rechecks, 0);
    assert_eq!(policy.block_steps.steps[&(16, 32)].micros, 64_000.0);
    for _ in 0..ZERO_DRAFT_RECHECK {
        let drafts = policy.plan(&caught_up, DraftBudget::new(16, 16));
        assert_eq!(drafts, [0; 16]);
        policy.observe_step(&drafts, 0, 90_000);
    }
    assert_eq!(policy.plan(&caught_up, DraftBudget::new(16, 16)), [1; 16]);
    assert_eq!(policy.decision().unwrap().record()["reason"], "drafting");
}

#[test]
fn losing_rechecks_are_bounded_and_their_total_time_is_charged() {
    let mut policy = block_policy();
    seed(&mut policy, 30_000.0, 185_000.0, 16);
    let estimate = AcceptanceEstimate::initial();
    let fresh = candidates(&estimate, true);
    let caught_up = candidates(&estimate, false);
    policy.zero_draft[16].idle = PROBE_INTERVAL;
    for index in 0..=BLOCK_RECHECKS {
        let proposals = policy.plan(
            if index == 0 { &fresh } else { &caught_up },
            DraftBudget::new(16, 16),
        );
        assert_eq!(proposals, [1; 16]);
        policy.step_accepted = 0;
        policy.observe_step(&proposals, if index == 0 { 16 } else { 0 }, 200_000);
    }
    assert_eq!(policy.zero_draft[16].rechecks, 0);
    assert!(policy.zero_draft[16].probe_interval >= (PROBE_SPACING * 600_000.0 / 30_000.0) as u16);
    for _ in 0..64 {
        let proposals = policy.plan(&caught_up, DraftBudget::new(16, 16));
        assert_eq!(proposals, [0; 16]);
        policy.observe_step(&proposals, 0, 30_000);
    }
}

#[test]
fn cheaper_fresh_steps_bound_the_known_steady_cost() {
    let mut policy = block_policy();
    seed(&mut policy, 95_000.0, 209_000.0, 8);
    let estimate = AcceptanceEstimate::initial();
    let candidates = candidates(&estimate, true);
    assert_eq!(policy.plan(&candidates, DraftBudget::new(16, 16)), [0; 16]);
    policy.step_accepted = 5;
    policy.observe_step(&[1; 16], 16, 74_000);
    assert_eq!(policy.block_steps.steps[&(16, 32)].micros, 74_000.0);
    assert!(
        policy
            .plan(&candidates, DraftBudget::new(16, 16))
            .iter()
            .any(|&k| k > 0)
    );
}

#[test]
#[ignore = "requires a staged directory named by CHOPINFER_SPECULATION_REPLAY"]
fn recorded_cost_feedback_leaves_the_stuck_estimated_fallback() {
    let root = std::env::var_os("CHOPINFER_SPECULATION_REPLAY").expect("staged replay directory");
    let bytes = std::fs::read(std::path::Path::new(&root).join("replay.json")).unwrap();
    let replay: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(replay["schema"], "chopinfer.speculation-cost-replay.v1");
    let initial = &replay["initial"];
    let mut policy = block_policy();
    seed(
        &mut policy,
        initial["ar_us"].as_f64().unwrap(),
        initial["verify_us"].as_f64().unwrap(),
        initial["drafting_items"].as_u64().unwrap() as usize,
    );
    let estimate = AcceptanceEstimate::initial();
    let candidates = candidates(&estimate, true);
    let before = policy.plan(&candidates, DraftBudget::new(16, 16));
    assert_eq!(before, [0; 16]);
    assert_eq!(
        policy.decision().unwrap().record()["reason"],
        "estimated_slower"
    );
    let observations = replay["observations"].as_array().unwrap();
    assert!(observations.len() >= 16);
    for (index, observation) in observations.iter().enumerate() {
        let drafts = observation["drafts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_u64().unwrap() as usize)
            .collect::<Vec<_>>();
        assert_eq!(drafts, [1; 16]);
        let fresh = observation["fresh"].as_u64().unwrap() as usize;
        let micros = observation["micros"].as_u64().unwrap();
        policy.step_accepted = observation["accepted"].as_u64().unwrap() as usize;
        policy.observe_step(&drafts, fresh, micros);
        if index == 0 {
            assert!(
                fresh > 0,
                "first recorded step must exercise the fresh bound"
            );
            assert!(policy.block_steps.steps[&(16, 32)].micros <= micros as f64);
            assert!(
                policy
                    .plan(&candidates, DraftBudget::new(16, 16))
                    .iter()
                    .any(|&k| k > 0)
            );
        }
    }
    let price = policy.block_steps.steps[&(16, 32)].micros;
    assert!(price < initial["verify_us"].as_f64().unwrap() * 0.5);
    println!(
        "RECORDED_COST_REFRESH_OK samples={} initial_us={} learned_us={price}",
        observations.len(),
        initial["verify_us"]
    );
}
