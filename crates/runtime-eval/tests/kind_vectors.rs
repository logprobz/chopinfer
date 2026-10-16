//! Every shared kind against its conformance vectors (tests/qual-vectors,
//! docs/check-kinds.md#conformance-vectors): the vector's receipt, reference
//! and check must give its expected status, error and metrics. This build
//! holds itself to exact agreement: equal JSON types, and floats equal in
//! their IEEE 754 bits, where the contract allows a relative 1e-12.

use std::path::Path;

use runtime_eval::declaration::{Check, Predicate};
use runtime_eval::kind::{self, Kind, Status};
use runtime_eval::kinds;
use serde_json::{Map, Value};

fn vector_files() -> Vec<Value> {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/qual-vectors");
    let mut paths: Vec<_> = std::fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "json")
        })
        .collect();
    paths.sort();
    paths
        .into_iter()
        .map(|path| runtime_eval::json::parse(&std::fs::read(path).unwrap()).unwrap())
        .filter(|file| file["kind"] != "references")
        .collect()
}

/// A declared check holding the vector's predicates or rule.
fn check(kind: &str, declared: &Value) -> Check {
    let rules = declared["rules"]
        .as_array()
        .map(|rules| {
            rules
                .iter()
                .map(|rule| Predicate {
                    path: rule["path"].as_array().unwrap().clone(),
                    op: rule["op"].as_str().unwrap().into(),
                    value: rule["value"].clone(),
                    source: rule["source"].as_str().unwrap_or_default().into(),
                })
                .collect()
        })
        .unwrap_or_default();
    Check {
        id: kind.into(),
        module: declared["module"].as_str().unwrap_or(kind).into(),
        timeout_seconds: 60.0,
        rules,
        informational_reason: None,
        declared: declared.clone(),
    }
}

/// Exact agreement: the same JSON type and value, floats bit for bit.
fn agree(expected: &Value, actual: &Value, at: &str) -> Result<(), String> {
    match (expected, actual) {
        (Value::Number(a), Value::Number(b)) => {
            if a.is_f64() != b.is_f64() {
                return Err(format!(
                    "{at}: {actual} has another number type than {expected}"
                ));
            }
            let same = if a.is_f64() {
                a.as_f64().map(f64::to_bits) == b.as_f64().map(f64::to_bits)
            } else {
                a == b
            };
            if same {
                Ok(())
            } else {
                Err(format!("{at}: {actual} != {expected}"))
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            if a.len() != b.len() {
                return Err(format!("{at}: {} items, not {}", b.len(), a.len()));
            }
            for (index, (a, b)) in a.iter().zip(b).enumerate() {
                agree(a, b, &format!("{at}[{index}]"))?;
            }
            Ok(())
        }
        (Value::Object(a), Value::Object(b)) => {
            let mut keys: Vec<&String> = a.keys().collect();
            let mut others: Vec<&String> = b.keys().collect();
            keys.sort();
            others.sort();
            if keys != others {
                return Err(format!("{at}: keys {others:?}, not {keys:?}"));
            }
            for (key, value) in a {
                agree(value, &b[key], &format!("{at}.{key}"))?;
            }
            Ok(())
        }
        _ if expected == actual => Ok(()),
        _ => Err(format!("{at}: {actual} != {expected}")),
    }
}

#[test]
fn every_vector_reproduces_exactly() {
    let implemented = kinds::all();
    let mut failures = Vec::new();
    let mut counted: Map<String, Value> = Map::new();
    for file in vector_files() {
        let name = file["kind"].as_str().unwrap();
        let Some(kind) = implemented.iter().find(|kind| kind.info().name == name) else {
            continue;
        };
        let kind: &dyn Kind = kind.as_ref();
        let vectors = file["vectors"].as_array().unwrap();
        counted.insert(name.into(), vectors.len().into());
        for vector in vectors {
            let label = format!("{name} {}", vector["name"].as_str().unwrap());
            let reference = vector.get("reference").filter(|value| !value.is_null());
            let result = kind::rescore(
                kind,
                &vector["receipt"],
                reference,
                &check(name, &vector["check"]),
                vector["mechanics"].as_bool().unwrap_or(false),
            );
            let expected = &vector["expected"];
            if result.status == Status::NotApplicable {
                assert_eq!(
                    result.reason.as_deref(),
                    expected["reason"].as_str(),
                    "{label}"
                );
            }
            if expected["status"].as_str() != Some(result.status.as_str()) {
                failures.push(format!(
                    "{label}: status {}, not {} ({:?})",
                    result.status.as_str(),
                    expected["status"],
                    result.error
                ));
                continue;
            }
            if let Some(error) = expected["error"].as_str()
                && !result.error.as_deref().unwrap_or_default().contains(error)
            {
                failures.push(format!("{label}: error {:?} lacks {error:?}", result.error));
            }
            if let Some(metrics) = expected.get("metrics") {
                let actual = Value::Object(result.metrics.clone().unwrap_or_default());
                if let Err(difference) = agree(metrics, &actual, "metrics") {
                    failures.push(format!("{label}: {difference}"));
                }
            }
            if result.status == Status::Pass {
                assert!(result.metrics.is_some(), "{label}");
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} disagreements:\n{}",
        failures.len(),
        failures.join("\n")
    );
    for kind in [
        "teacher_forced",
        "media_forced",
        "greedy_pin",
        "determinism",
        "api",
        "structured",
    ] {
        assert!(
            counted
                .get(kind)
                .is_some_and(|count| count.as_u64() > Some(0)),
            "{kind}"
        );
    }
}

fn vector(kind: &str, name: &str) -> Value {
    let file = vector_files()
        .into_iter()
        .find(|file| file["kind"] == kind)
        .unwrap();
    file["vectors"]
        .as_array()
        .unwrap()
        .iter()
        .find(|vector| vector["name"] == name)
        .unwrap()
        .clone()
}

fn kind_named(name: &str) -> Box<dyn Kind> {
    kinds::all()
        .into_iter()
        .find(|kind| kind.info().name == name)
        .unwrap()
}

#[test]
fn impossible_media_scores_are_insufficient() {
    let base = vector("media_forced", "pass");
    let kind = kind_named("media_forced");
    let changes = [
        serde_json::json!({"mean_abs_dlogprob": -1, "max_abs_dlogprob": -1}),
        serde_json::json!({"top1_disagreement": -1}),
        serde_json::json!({"top1_disagreement": 1.5}),
        serde_json::json!({"top1_disagreement": 0.4}),
        serde_json::json!({"mean_abs_dlogprob": 0.05, "max_abs_dlogprob": 0}),
        serde_json::json!({"mean_abs_dlogprob": null}),
    ];
    for change in changes {
        let mut receipt = base["receipt"].clone();
        let scores = receipt["rows"][0]["logprobs"].as_object_mut().unwrap();
        for (key, value) in change.as_object().unwrap() {
            scores.insert(key.clone(), value.clone());
        }
        let result = kind::rescore(
            kind.as_ref(),
            &receipt,
            Some(&base["reference"]),
            &check("media_forced", &base["check"]),
            false,
        );
        assert_eq!(
            result.status,
            Status::Insufficient,
            "{change}: {:?}",
            result.error
        );
    }
}

#[test]
fn impossible_reference_scores_are_insufficient() {
    for name in ["teacher_forced", "media_forced"] {
        let base = vector(name, "pass");
        let kind = kind_named(name);
        for logprob in [serde_json::json!(0.5), serde_json::json!(null)] {
            let mut reference = base["reference"].clone();
            let gate = reference["gate"].as_str().unwrap().to_owned();
            let scores = if name == "teacher_forced" {
                &mut reference["texts"][0][gate.as_str()]
            } else {
                &mut reference["cases"][0]["exact"][gate.as_str()]
            };
            scores["logprob"][0] = logprob.clone();
            let result = kind::rescore(
                kind.as_ref(),
                &base["receipt"],
                Some(&reference),
                &check(name, &base["check"]),
                false,
            );
            assert_eq!(
                result.status,
                Status::Insufficient,
                "{name} {logprob}: {:?}",
                result.error
            );
        }
    }
}

#[test]
fn rescore_requests_give_the_vector_verdicts() {
    let mut calls = Vec::new();
    let mut expected = Vec::new();
    for file in vector_files() {
        for vector in file["vectors"].as_array().unwrap() {
            calls.push(serde_json::json!({
                "op": "rescore", "kind": file["kind"], "receipt": vector["receipt"],
                "reference": vector.get("reference").cloned().unwrap_or(Value::Null),
                "check": vector["check"], "mechanics": vector["mechanics"].as_bool().unwrap_or(false)}));
            expected.push(vector["expected"].clone());
        }
    }
    let request = serde_json::json!({"schema": runtime_eval::rescore::REQUEST, "calls": calls});
    let response = runtime_eval::rescore::respond(&request).unwrap();
    let results = response["results"].as_array().unwrap();
    assert_eq!(results.len(), expected.len());
    for (result, expected) in results.iter().zip(&expected) {
        assert_eq!(result["status"], expected["status"], "{result}");
        if let Some(metrics) = expected.get("metrics") {
            agree(metrics, &result["metrics"], "metrics").unwrap();
        }
    }
    // The single steps answer as the verdict procedure takes them.
    let file = vector_files()
        .into_iter()
        .find(|file| file["kind"] == "api")
        .unwrap();
    let vector = &file["vectors"][0];
    let step = |op: &str, extra: Value| {
        let mut call = serde_json::json!({"op": op, "kind": "api", "receipt": vector["receipt"],
                                          "check": vector["check"]});
        call.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        let request =
            serde_json::json!({"schema": runtime_eval::rescore::REQUEST, "calls": [call]});
        runtime_eval::rescore::respond(&request).unwrap()["results"][0].clone()
    };
    assert_eq!(
        step("verify", serde_json::json!({}))["recorded_fields"],
        serde_json::json!(["rows"])
    );
    let metrics = step("score", serde_json::json!({}))["metrics"].clone();
    assert_eq!(metrics["passed"], true);
    let judged = step("judge", serde_json::json!({"metrics": metrics}));
    assert_eq!(judged["status"], "pass");
    let missing = step("judge", serde_json::json!({"metrics": {}}));
    assert_eq!(missing["error"]["type"], "missing_metric");
    let short = step("verify", serde_json::json!({"receipt": {"rows": []}}));
    assert_eq!(short["error"]["type"], "insufficient");
    assert!(runtime_eval::rescore::respond(&serde_json::json!({"calls": []})).is_err());
}

#[test]
fn malformed_protocol_rules_are_refused() {
    use runtime_eval::rescore::{REQUEST, respond};
    use serde_json::json;
    let valid = json!({"path": ["passed"], "op": "eq", "value": true, "source": "test"});
    let mut checks = vec![json!({}), json!({"rules": []})];
    for rules in [Value::Null, json!({}), json!("bad"), json!(true), json!(1)] {
        checks.push(json!({"rules": rules}));
    }
    for (field, value) in [
        ("path", json!({})),
        ("path", json!([null])),
        ("path", json!([1.5])),
        ("op", json!("bad")),
        ("op", json!(false)),
        ("source", json!(12)),
    ] {
        let mut rule = valid.clone();
        rule[field] = value;
        checks.push(json!({"rules": [rule]}));
    }
    for (op, value) in [
        ("le", json!("1")),
        ("ge", json!(null)),
        ("count_ge", json!(-1)),
        ("true_count_ge", json!(0.5)),
    ] {
        let mut rule = valid.clone();
        rule["op"] = json!(op);
        rule["value"] = value;
        checks.push(json!({"rules": [rule]}));
    }
    for field in ["path", "op", "value", "source"] {
        let mut rule = valid.clone();
        rule.as_object_mut().unwrap().remove(field);
        checks.push(json!({"rules": [rule]}));
    }
    for check in checks {
        for op in ["judge", "rescore"] {
            let request = json!({"schema": REQUEST, "calls": [{"op": op, "kind": "api",
                "check": check, "receipt": {"passed": false}, "metrics": {"passed": false}}]});
            assert!(respond(&request).is_err(), "{request}");
        }
    }
    for kind in ["api", "greedy_pin", "determinism"] {
        for op in ["verify", "score", "judge", "rescore"] {
            let request = json!({"schema": REQUEST, "calls": [{"op": op, "kind": kind,
                "check": {"rules": null}, "metrics": {"passed": true, "equal": true}}]});
            assert!(respond(&request).is_err(), "{request}");
        }
    }
}
