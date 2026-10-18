use super::*;
use serde_json::{Value, json};

const CONTEXT: Context = Context {
    instance_id: u64::MAX,
    step_id: u64::MAX - 1,
};

fn diagnostic() -> Diagnostic {
    Diagnostic {
        request: u64::MAX - 2,
        main_rows: 128,
        proposed: [11, 12, 13, 14, 15],
        confidence_bits: [
            0x3f80_8000,
            0x8000_0000,
            0x7fc0_0123,
            0x7f80_0000,
            0xff80_0000,
        ],
        outcomes: [
            Outcome::Accepted,
            Outcome::Unknown,
            Outcome::Unknown,
            Outcome::NotEmitted,
            Outcome::NotEmitted,
        ],
    }
}

#[test]
fn partial_budget_preserves_every_score_bit_and_large_identity() {
    let input = diagnostic();
    let encoded = encode(CONTEXT, &input).unwrap();
    assert!(!encoded.contains('\n'));
    let value: Value = serde_json::from_str(&encoded).unwrap();
    assert_eq!(value["schema"], SCHEMA);
    assert!(SCHEMA.starts_with("chopinfer.") && SCHEMA.ends_with(".drafter-confidence.v1"));
    assert_eq!(value["confidence_contract"], CONFIDENCE_CONTRACT);
    assert_eq!(value["instance_id"], u64::MAX.to_string());
    assert_eq!(value["step_id"], (u64::MAX - 1).to_string());
    assert_eq!(value["request_id"], input.request.to_string());
    assert_eq!(value["computed_positions"], 5);
    assert_eq!(value["emitted_positions"], 3);
    assert_eq!(value["verify_rows"], 4);
    assert_eq!(value["draft_ids"], json!(input.proposed));
    assert_eq!(value["confidence_f32_bits"], json!(input.confidence_bits));
    assert_eq!(value["completion"], "committed");
    assert_eq!(value["acceptance_evidence"], "retained_rows_lower_bound");
    assert_eq!(
        value["outcomes"],
        json!([
            "accepted",
            "unknown",
            "unknown",
            "not_emitted",
            "not_emitted"
        ])
    );
}

#[test]
fn cancellation_is_distinct_from_unknown_and_all_accepted() {
    let mut input = diagnostic();
    input.outcomes[..3].fill(Outcome::Cancelled);
    let cancelled: Value = serde_json::from_str(&encode(CONTEXT, &input).unwrap()).unwrap();
    assert_eq!(cancelled["completion"], "cancelled");
    assert_eq!(
        cancelled["outcomes"],
        json!([
            "cancelled",
            "cancelled",
            "cancelled",
            "not_emitted",
            "not_emitted"
        ])
    );
    input.outcomes.fill(Outcome::Accepted);
    let accepted: Value = serde_json::from_str(&encode(CONTEXT, &input).unwrap()).unwrap();
    assert_eq!(accepted["completion"], "committed");
    assert_eq!(accepted["emitted_positions"], 5);
    assert_eq!(accepted["verify_rows"], 6);
    assert_eq!(accepted["outcomes"], json!(vec!["accepted"; 5]));
}

#[test]
fn reject_unproven_rejection_labels_and_malformed_prefixes() {
    for outcomes in [
        [Outcome::NotEmitted; 5],
        [Outcome::Rejected; 5],
        [
            Outcome::Unknown,
            Outcome::Accepted,
            Outcome::Unknown,
            Outcome::Unknown,
            Outcome::Unknown,
        ],
        [
            Outcome::Unknown,
            Outcome::NotEmitted,
            Outcome::Unknown,
            Outcome::NotEmitted,
            Outcome::NotEmitted,
        ],
        [
            Outcome::Cancelled,
            Outcome::Unknown,
            Outcome::Unknown,
            Outcome::NotEmitted,
            Outcome::NotEmitted,
        ],
    ] {
        let mut input = diagnostic();
        input.outcomes = outcomes;
        assert!(encode(CONTEXT, &input).is_err());
    }
}

#[test]
fn rejection_has_a_named_refusal() {
    let mut input = diagnostic();
    for outcomes in [
        [Outcome::Rejected; 5],
        [
            Outcome::Accepted,
            Outcome::Rejected,
            Outcome::Unknown,
            Outcome::NotEmitted,
            Outcome::NotEmitted,
        ],
    ] {
        input.outcomes = outcomes;
        assert_eq!(
            encode(CONTEXT, &input),
            Err("confidence record does not support rejection labels without verifier evidence")
        );
    }
}

#[test]
fn log_failure_does_not_stop_later_records_and_empty_steps_log_nothing() {
    let input = diagnostic();
    let mut seen = Vec::new();
    record(CONTEXT, &[input.clone(), input], |message| {
        seen.push(serde_json::from_slice::<Value>(message).unwrap());
        Err("injected logging failure")
    });
    assert_eq!(seen.len(), 2);
    record(CONTEXT, &[], |_| -> Result<(), ()> {
        panic!("K=0 has no block to log")
    });
}
