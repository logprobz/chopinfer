//! Recorded profitable low-concurrency feedback must keep adaptive drafting.
use super::*;

#[test]
#[ignore = "requires a staged directory named by CHOPINFER_SPECULATION_REPLAY"]
fn recorded_profitable_single_request_keeps_drafting() {
    let root = std::env::var_os("CHOPINFER_SPECULATION_REPLAY").expect("staged replay directory");
    let bytes = std::fs::read(std::path::Path::new(&root).join("low-concurrency.json")).unwrap();
    let replay: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(replay["schema"], "chopinfer.low-concurrency-replay.v1");
    assert_eq!(replay["concurrency"], 1);
    assert_eq!(replay["audit"]["per_item_exact"], true);
    let mut policy = SpeculationPolicy::new(
        SpeculationConfig::new(5, 32, SpeculationMode::Adaptive, DraftSource::Bundle)
            .unwrap()
            .with_draft_cost_model(DraftCostModel::PerBlock),
    )
    .with_step_rows(|rows| rows)
    .with_diagnostics(true);
    let ar = replay["ar_micros"].as_array().unwrap();
    assert!(ar.len() >= 100);
    for micros in ar {
        policy.observe_step(&[0], 0, micros.as_u64().unwrap());
    }
    let observations = replay["observations"].as_array().unwrap();
    assert!(observations.len() >= 64);
    let mut estimate = AcceptanceEstimate::initial();
    let mut previous = None;
    let mut reasons = BTreeMap::new();
    for row in observations {
        let request = row["request"].as_str().unwrap();
        let fresh = previous != Some(request);
        if fresh {
            estimate = AcceptanceEstimate::initial();
        }
        let verified = row["drafts"].as_u64().unwrap() as usize;
        let accepted = row["accepted"].as_u64().unwrap() as usize;
        assert!(verified > 0 && accepted <= verified);
        assert_eq!(row["live_rows"].as_u64().unwrap() as usize, 1 + verified);
        let proposal = policy.plan(
            &[PlanCandidate {
                maximum_draft_tokens: verified,
                acceptance: &estimate,
                fresh,
                remaining_tokens: verified + 1,
            }],
            DraftBudget::new(1, 31),
        );
        let reason = policy.decision().unwrap().record()["reason"]
            .as_str()
            .unwrap()
            .to_owned();
        assert!(
            proposal[0] > 0,
            "batch {} fell back: {reason}",
            row["batch"]
        );
        *reasons.entry(reason).or_insert(0_usize) += 1;
        // The trace supplies the executed width and its exact outcomes.
        // Never assign its elapsed time to a different proposed width.
        policy.observe(&mut estimate, verified, accepted);
        policy.observe_step(
            &[verified],
            usize::from(fresh),
            row["micros"].as_u64().unwrap(),
        );
        previous = Some(request);
    }
    println!(
        "LOW_CONCURRENCY_REPLAY_OK ar_samples={} draft_samples={} fallback=0 reasons={reasons:?}",
        ar.len(),
        observations.len()
    );
}
