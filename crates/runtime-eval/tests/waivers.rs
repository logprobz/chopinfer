//! Declared waivers (`checks.<id>.waivers`) as the judge applies them and
//! the report lists them: a failure is `waived` only where a waiver covers
//! it within its ceiling; anything else still fails.

use runtime_eval::declaration::{Check, Predicate};
use runtime_eval::kind::{self, Judged, Status};
use runtime_eval::report::{CheckReport, Report, Tool};
use serde_json::{Value, json};

fn rule(path: Value, op: &str, value: Value) -> Predicate {
    Predicate {
        path: path.as_array().unwrap().clone(),
        op: op.into(),
        value,
        source: "test".into(),
    }
}

fn check(rules: Vec<Predicate>, waivers: Value) -> Check {
    Check {
        id: "teacher".into(),
        module: "teacher".into(),
        timeout_seconds: 60.0,
        rules,
        informational_reason: None,
        declared: json!({"waivers": waivers}),
    }
}

fn judge(metrics: &Value, check: &Check) -> (Status, Vec<Judged>) {
    let judged = check
        .rules
        .iter()
        .map(|rule| Judged {
            passed: kind::predicate(metrics, rule).unwrap(),
            rule: rule.clone(),
            waiver: None,
        })
        .collect();
    kind::waive(metrics, check, metrics["passed"] != false, judged).unwrap()
}

fn spread(mean: f64, decisive: f64, near: f64, total: f64) -> Value {
    json!({"mean_abs_dlogprob": mean, "top1_decisive": decisive, "top1_near_tie": near,
           "top1_disagreement": total})
}

/// A teacher verdict: pooled within its limits, `prose-frozen` failing
/// its total top-1 limit at `total` (and its near-tie limit at `near`).
fn teacher(total: f64, near: f64, pooled_total: f64) -> Value {
    let mut frozen = spread(0.1, 0.03, near, total);
    frozen["limits"] = spread(0.15, 0.08, 0.1, 0.1);
    frozen["passed_by"] = json!({"top1_decisive": "limit",
                                 "top1_near_tie": if near <= 0.1 { json!("limit") } else { Value::Null },
                                 "top1_disagreement": if total <= 0.1 { json!("limit") } else { Value::Null }});
    let mut metrics = spread(0.06, 0.008, 0.05, pooled_total);
    metrics["limits"] = spread(0.1, 0.08, 0.08, 0.1);
    metrics["by_category"] = json!({"prose-frozen": frozen});
    metrics["failed_categories"] = json!(["prose-frozen"]);
    metrics["passed"] = json!(false);
    metrics["scored"] = json!(3781);
    metrics
}

fn teacher_rules() -> Vec<Predicate> {
    vec![
        rule(json!(["passed"]), "eq", json!(true)),
        rule(json!(["failed_categories"]), "eq", json!([])),
        rule(json!(["scored"]), "ge", json!(3781)),
    ]
}

fn category_waiver(ceiling: f64) -> Value {
    json!([{"category": "prose-frozen", "metric": "top1_disagreement", "ceiling": ceiling,
            "reason": "near-tie churn at the reference's own rate", "decision": "operator, 2026-10-15"}])
}

#[test]
fn a_predicate_waiver_covers_its_predicate_up_to_its_ceiling() {
    let rules = vec![rule(json!(["mean_gap"]), "le", json!(0.005))];
    let waived = check(
        rules.clone(),
        json!([{"predicate": {"path": ["mean_gap"], "op": "le"}, "ceiling": 0.0057,
                "reason": "sampling-level evidence", "decision": "maintainers, 2026-10-06"}]),
    );
    let (status, judged) = judge(&json!({"mean_gap": 0.0057}), &waived);
    assert_eq!(status, Status::Waived);
    let record = judged[0].to_json();
    assert_eq!(record["passed"], false);
    assert_eq!(record["waiver"]["measured"], json!([0.0057]));
    assert_eq!(record["waiver"]["ceiling"], 0.0057);
    assert_eq!(record["waiver"]["decision"], "maintainers, 2026-10-06");
    // A worse value than the waiver accepted fails.
    assert_eq!(judge(&json!({"mean_gap": 0.0058}), &waived).0, Status::Fail);
    // A passing check is a pass, with no waiver applied.
    let (status, judged) = judge(&json!({"mean_gap": 0.004}), &waived);
    assert_eq!(status, Status::Pass);
    assert!(judged[0].waiver.is_none());
    // Without a waiver the failure stands.
    assert_eq!(
        judge(&json!({"mean_gap": 0.0057}), &check(rules, json!([]))).0,
        Status::Fail
    );
}

#[test]
fn a_category_waiver_covers_only_its_metric_within_its_ceiling() {
    let waived = check(teacher_rules(), category_waiver(0.1135));
    let (status, judged) = judge(&teacher(0.1135, 0.07, 0.0619), &waived);
    assert_eq!(status, Status::Waived);
    for item in &judged[..2] {
        let applied = &item.waiver.as_ref().unwrap()["categories"][0];
        assert_eq!(applied["category"], "prose-frozen");
        assert_eq!(applied["measured"], 0.1135);
    }
    assert!(judged[2].passed && judged[2].waiver.is_none());
    // Above its ceiling, failing another metric too, a pooled failure, or
    // a failed category without a waiver: the check fails.
    assert_eq!(judge(&teacher(0.12, 0.07, 0.0619), &waived).0, Status::Fail);
    assert_eq!(
        judge(&teacher(0.1135, 0.11, 0.0619), &waived).0,
        Status::Fail
    );
    assert_eq!(judge(&teacher(0.1135, 0.07, 0.11), &waived).0, Status::Fail);
    let mut other = teacher(0.1135, 0.07, 0.0619);
    other["failed_categories"] = json!(["prose-frozen", "chat"]);
    other["by_category"]["chat"] = other["by_category"]["prose-frozen"].clone();
    assert_eq!(judge(&other, &waived).0, Status::Fail);
    // A metric the paired fallback passed is not a failing one.
    let mut fallback = teacher(0.1135, 0.11, 0.0619);
    fallback["by_category"]["prose-frozen"]["passed_by"]["top1_near_tie"] =
        json!("paired_reference");
    assert_eq!(judge(&fallback, &waived).0, Status::Waived);
}

#[test]
fn the_report_lists_applied_waivers_and_a_waived_run_qualifies() {
    let waived = check(teacher_rules(), category_waiver(0.1135));
    let (status, judged) = judge(&teacher(0.1135, 0.07, 0.0619), &waived);
    let mut entry =
        CheckReport::without_verdict("teacher", "teacher_forced", "teacher_forced.v1", status, "");
    entry.error = None;
    entry.predicates = judged.iter().map(Judged::to_json).collect();
    let report = Report {
        tool: Tool {
            name: "chopinfer".into(),
            version: "test".into(),
            revision: None,
        },
        model: "m".into(),
        declaration_sha256: "0".repeat(64),
        suite: "accuracy".into(),
        at: "20261015T000000000000Z".into(),
        server: json!({}),
        checks: vec![entry],
        seconds: 1.0,
        failure: None,
    };
    assert!(report.passed() && report.qualified());
    let json = report.to_json();
    assert_eq!(json["checks"][0]["status"], "waived");
    let waivers = json["waivers"].as_array().unwrap();
    assert_eq!(waivers.len(), 2);
    assert!(waivers.iter().all(|waiver| waiver["check"] == "teacher"
        && waiver["waiver"]["categories"][0]["decision"] == "operator, 2026-10-15"));
    assert!(report.summary().contains("1 waived"));
}

/// A predicate waiver every waiver path must refuse: no reason or decision
/// (empty or blank), or a ceiling no looser than its rule.
fn invalid_waivers() -> Vec<(&'static str, Value)> {
    let waiver = |reason: &str, decision: &str, ceiling: f64| {
        json!([{"predicate": {"path": ["mean_gap"], "op": "le"}, "ceiling": ceiling,
                "reason": reason, "decision": decision}])
    };
    vec![
        ("reason", waiver("", "maintainers, 2026-10-06", 0.0057)),
        ("reason", waiver("  ", "maintainers, 2026-10-06", 0.0057)),
        ("decision", waiver("sampling-level evidence", "", 0.0057)),
        ("decision", waiver("sampling-level evidence", " \t", 0.0057)),
        (
            "ceiling",
            waiver("sampling-level evidence", "maintainers, 2026-10-06", 0.005),
        ),
    ]
}

#[test]
fn an_invalid_waiver_covers_nothing() {
    // Default deny: the judge applies a check's waivers only when the
    // declaration validator accepts them, whatever path built the check.
    let rules = vec![rule(json!(["mean_gap"]), "le", json!(0.005))];
    for (field, waivers) in invalid_waivers() {
        let (status, judged) = judge(&json!({"mean_gap": 0.0057}), &check(rules.clone(), waivers));
        assert_eq!(status, Status::Fail, "{field}");
        assert!(judged[0].waiver.is_none(), "{field}");
    }
}

#[test]
fn rescore_refuses_a_check_with_invalid_waivers_and_validates_them_alone() {
    use runtime_eval::rescore::{REQUEST, respond};
    let rules = json!([{"path": ["mean_gap"], "op": "le", "value": 0.005, "source": "test"}]);
    let valid = json!([{"predicate": {"path": ["mean_gap"], "op": "le"}, "ceiling": 0.0057,
                        "reason": "sampling-level evidence", "decision": "maintainers, 2026-10-06"}]);
    let alone = |waivers: &Value| {
        respond(&json!({"schema": REQUEST, "calls": [{"op": "waivers",
            "check": {"module": "tapdiff", "rules": rules, "waivers": waivers}}]}))
        .unwrap()["results"][0]
            .clone()
    };
    assert_eq!(alone(&valid), json!({"valid": true}));
    for (field, waivers) in invalid_waivers() {
        let refused = alone(&waivers);
        assert_eq!(refused["error"]["type"], "invalid", "{field}");
        assert!(
            refused["error"]["message"]
                .as_str()
                .unwrap()
                .contains(&format!(".{field}")),
            "{field}: {refused}"
        );
        // A shared kind's judge never runs with them.
        let judged = respond(
            &json!({"schema": REQUEST, "calls": [{"op": "judge", "kind": "teacher_forced",
            "check": {"module": "teacher", "rules": rules, "waivers": waivers},
            "metrics": {"mean_gap": 0.0057}}]}),
        );
        assert!(
            judged
                .as_ref()
                .is_err_and(|message| message.contains("waivers")),
            "{field}: {judged:?}"
        );
    }
}
