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
    for items in 1..=4 {
        // C1 is measured. Larger batches replicate its request outcomes and
        // scale both costs linearly to check policy behavior, not throughput.
        replay_profitable(&replay, items);
    }
}

fn replay_profitable(replay: &serde_json::Value, items: usize) {
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
        policy.observe_step(&vec![0; items], 0, micros.as_u64().unwrap() * items as u64);
    }
    let observations = replay["observations"].as_array().unwrap();
    assert!(observations.len() >= 64);
    let mut estimates = vec![AcceptanceEstimate::initial(); items];
    let mut previous = None;
    let mut reasons = BTreeMap::new();
    for row in observations {
        let request = row["request"].as_str().unwrap();
        let fresh = previous != Some(request);
        if fresh {
            estimates.fill(AcceptanceEstimate::initial());
        }
        let verified = row["drafts"].as_u64().unwrap() as usize;
        let accepted = row["accepted"].as_u64().unwrap() as usize;
        assert!(verified > 0 && accepted <= verified);
        assert_eq!(row["live_rows"].as_u64().unwrap() as usize, 1 + verified);
        let candidates = estimates
            .iter()
            .map(|estimate| PlanCandidate {
                maximum_draft_tokens: verified,
                acceptance: estimate,
                fresh,
                remaining_tokens: verified + 1,
            })
            .collect::<Vec<_>>();
        let proposal = policy.plan(&candidates, DraftBudget::new(items, 32 - items));
        let reason = policy.decision().unwrap().record()["reason"]
            .as_str()
            .unwrap()
            .to_owned();
        assert!(
            proposal.iter().all(|&count| count > 0),
            "C{items} batch {} fell back: {reason}",
            row["batch"]
        );
        *reasons.entry(reason).or_insert(0_usize) += 1;
        // The trace supplies the executed width and its exact outcomes.
        // Never assign its elapsed time to a different proposed width.
        for estimate in &mut estimates {
            policy.observe(estimate, verified, accepted);
        }
        policy.observe_step(
            &vec![verified; items],
            usize::from(fresh) * items,
            row["micros"].as_u64().unwrap() * items as u64,
        );
        previous = Some(request);
    }
    println!(
        "LOW_CONCURRENCY_REPLAY_OK items={items} linear_replication={} ar_samples={} draft_samples={} fallback=0 reasons={reasons:?}",
        items > 1,
        ar.len(),
        observations.len()
    );
}
