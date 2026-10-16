//! The limit rules on their boundary cases (tests/data/limit-cases.json),
//! directly and through `eval rescore`'s `limit` op, which the Python
//! tooling calls (tools/qual/limits.py).

use std::path::Path;

use runtime_eval::json::python_number;
use runtime_eval::limits::{
    self, ANSWER_FIELDS, NEAR_TIE_NATS, POSITION_FIELDS, Spread, SpreadRule,
};
use serde_json::{Value, json};

fn vectors() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/limit-cases.json");
    let value = runtime_eval::json::parse(&std::fs::read(path).unwrap()).unwrap();
    assert_eq!(value["schema"], "chopinfer-qual-limit-vectors/v1");
    value
}

fn cases<'a>(vectors: &'a Value, rule: &str) -> &'a [Value] {
    let cases = vectors[rule].as_array().unwrap();
    assert!(!cases.is_empty(), "{rule} has no vectors");
    cases
}

/// Equal JSON, numbers equal bit for bit (an int equals the float of its value).
fn same(expected: &Value, actual: &Value) -> bool {
    match (expected, actual) {
        (Value::Number(a), Value::Number(b)) => {
            let (x, y) = (a.as_f64().unwrap(), b.as_f64().unwrap());
            x.to_bits() == y.to_bits() || ((!a.is_f64() || !b.is_f64()) && x == y)
        }
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| same(a, b))
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len() && a.iter().all(|(k, v)| b.get(k).is_some_and(|w| same(v, w)))
        }
        _ => expected == actual,
    }
}

fn check(rule: &str, case: &Value, actual: Value) {
    assert!(
        same(&case["expected"], &actual),
        "{rule} {case}: got {actual}"
    );
}

fn number(value: &Value) -> f64 {
    python_number(value).unwrap()
}

fn spread(value: &Value) -> Spread {
    Spread {
        mean_abs_dlogprob: number(&value["mean_abs_dlogprob"]),
        top1_decisive: number(&value["top1_decisive"]),
        top1_near_tie: number(&value["top1_near_tie"]),
        top1_disagreement: number(&value["top1_disagreement"]),
    }
}

fn flip_of(value: &Value) -> Option<limits::Flip> {
    (!value.is_null()).then(|| limits::Flip {
        gap: value["gap"].as_f64(),
        decisive: value["decisive"].as_bool().unwrap(),
    })
}

fn arm(value: &Value) -> Vec<(f64, bool)> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|pair| (number(&pair[0]), pair[1].as_bool().unwrap()))
        .collect()
}

#[test]
fn rounding_up_to_the_step() {
    let vectors = vectors();
    for case in cases(&vectors, "rounded_up") {
        let actual = match limits::rounded_up(number(&case["value"]), number(&case["step"])) {
            Ok(rounded) => json!(rounded),
            Err(_) => json!({"error": true}),
        };
        check("rounded_up", case, actual);
    }
}

#[test]
fn spread_limits_and_within() {
    let vectors = vectors();
    for case in cases(&vectors, "spread_limits") {
        let mut rule = SpreadRule::default();
        for (key, value) in case["rule"].as_object().unwrap() {
            match key.as_str() {
                "mean_floor" => rule.mean_floor = number(value),
                "large_set_mean_floor" => rule.large_set_mean_floor = number(value),
                "large_set_positions" => rule.large_set_positions = value.as_u64().unwrap(),
                "top1_floor" => rule.top1_floor = number(value),
                "mean_step" => rule.mean_step = number(value),
                "mean_cap" => rule.mean_cap = Some(number(value)),
                "mean_factor" => rule.mean_factor = number(value),
                "near_tie_floor" => rule.near_tie_floor = number(value),
                "near_tie_step" => rule.near_tie_step = number(value),
                "total_floor" => rule.total_floor = number(value),
                other => panic!("unknown rule parameter {other}"),
            }
        }
        let variants: Vec<Spread> = case["variants"]
            .as_array()
            .unwrap()
            .iter()
            .map(spread)
            .collect();
        let positions = case["positions"].as_u64().unwrap();
        let actual = limits::spread_limits(&variants, positions, &rule).unwrap();
        check("spread_limits", case, serde_json::to_value(actual).unwrap());
    }
    for case in cases(&vectors, "within") {
        let actual = limits::within(&spread(&case["measured"]), &spread(&case["limits"]));
        check("within", case, json!(actual));
    }
}

#[test]
fn near_ties_flips_and_top1_counts() {
    let vectors = vectors();
    for case in cases(&vectors, "near_tie") {
        let actual = limits::near_tie(case["gap"].as_f64(), number(&case["limit"]));
        check("near_tie", case, json!(actual));
    }
    for case in cases(&vectors, "flip") {
        let top: Option<Vec<(Value, f64)>> = case["top"].as_array().map(|pairs| {
            pairs
                .iter()
                .map(|pair| (pair[0].clone(), number(&pair[1])))
                .collect()
        });
        let limit = case.get("limit").map_or(NEAR_TIE_NATS, number);
        let actual = limits::flip(&case["top1"], &case["served"], top.as_deref(), limit);
        check("flip", case, serde_json::to_value(actual).unwrap());
    }
    for case in cases(&vectors, "top1_summary") {
        let flips: Vec<_> = case["flips"]
            .as_array()
            .unwrap()
            .iter()
            .map(flip_of)
            .collect();
        let actual = limits::top1_summary(&flips, case["scored"].as_u64().unwrap());
        check("top1_summary", case, serde_json::to_value(actual).unwrap());
    }
}

#[test]
fn the_paired_rule() {
    let vectors = vectors();
    for case in cases(&vectors, "paired") {
        let actual = limits::paired(
            &arm(&case["test"]),
            &arm(&case["baseline"]),
            number(&case["mean_margin"]),
            number(&case["sigmas"]),
        );
        let actual = match actual {
            Ok(verdict) => serde_json::to_value(verdict).unwrap(),
            Err(_) => json!({"error": true}),
        };
        check("paired", case, actual);
    }
}

#[test]
fn the_exact_rule() {
    let vectors = vectors();
    for case in cases(&vectors, "bit_equal") {
        let actual = limits::bit_equal(&case["left"], &case["right"]);
        check("bit_equal", case, json!(actual));
    }
    for case in cases(&vectors, "first_difference") {
        let actual = limits::first_difference(
            case["left"].as_object().unwrap(),
            case["right"].as_object().unwrap(),
            &POSITION_FIELDS,
            &ANSWER_FIELDS,
        )
        .unwrap();
        check(
            "first_difference",
            case,
            actual.map_or(Value::Null, |difference| difference.to_json()),
        );
    }
}

/// Every case again through the `limit` op the Python tooling calls: the
/// same values, an `invalid` refusal where the rule cannot judge.
#[test]
fn every_case_through_the_limit_op() {
    let vectors = vectors();
    let arguments = |rule: &str, case: &Value| -> Value {
        let mut args = case.as_object().unwrap().clone();
        args.remove("expected");
        if rule == "spread_limits" && args["rule"].as_object().is_some_and(|rule| rule.is_empty()) {
            args.insert("rule".into(), Value::Null);
        }
        Value::Object(args)
    };
    for rule in [
        "rounded_up",
        "spread_limits",
        "within",
        "near_tie",
        "flip",
        "top1_summary",
        "paired",
        "bit_equal",
        "first_difference",
    ] {
        for case in cases(&vectors, rule) {
            let actual = match limits::call(rule, &arguments(rule, case)) {
                Ok(value) => value,
                Err(_) => json!({"error": true}),
            };
            check(rule, case, actual);
        }
    }
    assert_eq!(
        limits::call("constants", &Value::Null).unwrap()["NEAR_TIE_NATS"],
        json!(NEAR_TIE_NATS)
    );
    assert!(limits::call("unknown", &Value::Null).is_err());
}

/// Calls `eval rescore` must refuse rather than judge: a missing operand is
/// not a null one, an empty field selection compares nothing, a boolean is
/// no decimal, and arithmetic past the finite numbers has no JSON result.
#[test]
fn incomplete_mistyped_or_overflowing_calls_are_refused() {
    let refused = [
        json!({"op": "limit", "rule": "bit_equal"}),
        json!({"op": "limit", "rule": "bit_equal", "args": []}),
        json!({"op": "limit", "rule": "bit_equal", "args": {}}),
        json!({"op": "limit", "rule": "bit_equal", "args": {"left": null}}),
        json!({"op": "limit", "rule": "bit_equal", "args": {"left": 1, "right": 1, "other": 1}}),
        json!({"op": "limit", "rule": "flip", "args": {}}),
        json!({"op": "limit", "rule": "flip", "args": {"top1": 7}}),
        json!({"op": "limit", "rule": "flips", "args": {"positions": [{}]}}),
        json!({"op": "limit", "rule": "flips", "args": {}}),
        json!({"op": "limit", "rule": "first_difference",
               "args": {"left": {"text": "a"}, "right": {"text": "b"}, "positions": [], "answer": []}}),
        json!({"op": "limit", "rule": "first_difference", "args": {"left": {"text": "a"}}}),
        json!({"op": "limit", "rule": "rounded_up", "args": {"value": true, "step": 0.05}}),
        json!({"op": "limit", "rule": "rounded_up", "args": {"value": 0.1, "step": false}}),
        json!({"op": "limit", "rule": "rounded_up", "args": {"value": 0.1, "step": 0.0}}),
        json!({"op": "limit", "rule": "rounded_up", "args": {"value": 0.1}}),
        json!({"op": "limit", "rule": "rounded_up",
               "args": {"value": 1.0000000000000002e308, "step": 1e308}}),
        json!({"op": "limit", "rule": "paired", "args": {"test": [[0.0, true], [0.0, true], [0.0, true], [0.0, true]],
               "baseline": [[0.0, false], [0.0, false], [0.0, false], [0.0, false]],
               "mean_margin": 0.0, "sigmas": 1e308}}),
        json!({"op": "limit", "rule": "paired", "args": {"test": [[1e308, false], [1e308, false]],
               "baseline": [[-1e308, false], [-1e308, false]], "mean_margin": 0.0, "sigmas": 2.0}}),
        json!({"op": "limit", "rule": "flip",
               "args": {"top1": 1, "served": 2, "top": [[1, 1e308], [2, -1e308]]}}),
        json!({"op": "limit", "rule": "spread_limits", "args": {"positions": 10}}),
        json!({"op": "limit", "rule": "spread_limits", "args": {"variants": [{"mean_abs_dlogprob": 0.1}],
               "positions": 10}}),
        json!({"op": "limit", "rule": "top1_summary", "args": {"flips": [{"gap": 0.1}], "scored": 1}}),
        json!({"op": "limit", "rule": "within", "args": {"measured": {}}}),
        json!({"op": "limit", "rule": "near_tie", "args": {}}),
    ];
    let response = runtime_eval::rescore::respond(
        &json!({"schema": runtime_eval::rescore::REQUEST, "calls": refused}),
    )
    .unwrap();
    for (call, result) in refused.iter().zip(response["results"].as_array().unwrap()) {
        assert_eq!(result["error"]["type"], "invalid", "{call} -> {result}");
    }
    // An explicit null operand and an omitted optional argument stay values.
    let accepted = [
        (
            json!({"op": "limit", "rule": "bit_equal", "args": {"left": null, "right": null}}),
            json!(true),
        ),
        (
            json!({"op": "limit", "rule": "flip", "args": {"top1": 7, "served": 8}}),
            json!({"gap": null, "decisive": true}),
        ),
        (
            json!({"op": "limit", "rule": "near_tie", "args": {"gap": null}}),
            json!(false),
        ),
        (
            json!({"op": "limit", "rule": "first_difference",
                "args": {"left": {"text": "a"}, "right": {"text": "a"}, "answer": []}}),
            Value::Null,
        ),
    ];
    let calls: Vec<Value> = accepted.iter().map(|(call, _)| call.clone()).collect();
    let response = runtime_eval::rescore::respond(
        &json!({"schema": runtime_eval::rescore::REQUEST, "calls": calls}),
    )
    .unwrap();
    for ((call, expected), result) in accepted.iter().zip(response["results"].as_array().unwrap()) {
        assert_eq!(&result["value"], expected, "{call} -> {result}");
    }
}

/// The `limit` op over tests/data/limit-op-cases.json: each call's result
/// is the previous Python rules' (a value, or invalid where Python
/// raised), except the narrowed cases, which the op refuses although
/// Python judged them: a boolean as a number, an int beyond what the
/// rule's doubles hold exactly. Booleans are refused at every level, list
/// and object tokens wherever Python hashed them, and ints stay exact
/// where Python's arithmetic kept them so.
#[test]
fn the_limit_op_matches_the_previous_rules_or_refuses() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/limit-op-cases.json");
    let file = runtime_eval::json::parse(&std::fs::read(path).unwrap()).unwrap();
    assert_eq!(file["schema"], "chopinfer-limit-op-cases/v1");
    let cases = file["cases"].as_array().unwrap();
    let calls: Vec<Value> = cases
        .iter()
        .map(|case| json!({"op": "limit", "rule": case["rule"], "args": case["args"]}))
        .collect();
    let response = runtime_eval::rescore::respond(
        &json!({"schema": runtime_eval::rescore::REQUEST, "calls": calls}),
    )
    .unwrap();
    let results = response["results"].as_array().unwrap();
    assert_eq!(results.len(), cases.len());
    let mut narrowed = 0;
    for (case, result) in cases.iter().zip(results) {
        let expected = &case["expected"];
        if case.get("narrowed").is_some() {
            narrowed += 1;
            assert!(case["python"].get("value").is_some(), "{case}");
        }
        let matched = match expected.get("value") {
            Some(value) => result
                .get("value")
                .is_some_and(|actual| same(value, actual)),
            None => result["error"]["type"] == expected["error"],
        };
        assert!(
            matched,
            "{}: expected {expected}, got {result}",
            case["name"]
        );
    }
    assert!(
        cases.len() >= 200 && narrowed >= 20,
        "{} cases, {narrowed} narrowed",
        cases.len()
    );
}
