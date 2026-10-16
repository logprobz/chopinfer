//! `chopinfer eval rescore`: the offline steps of the shared kinds (verify,
//! score, judge and the whole verdict procedure) over JSON, so the Python
//! tooling scores with this implementation instead of its own.
//!
//! The request (stdin) is `{"schema": REQUEST, "calls": [CALL...]}`; each
//! call names an `op` (`rescore`, `verify`, `score` or `judge`), the
//! `kind`, and what the op reads: the `receipt`, the `reference` as the
//! kind reads it (null for none), the declared `check` (its `rules` or
//! `rule`, `constants` and `args`), `mechanics`, and for `judge` the
//! `metrics`. A `limit` call names a `rule` of [`crate::limits`] and its
//! `args`, and returns the rule's `value` (or an `invalid` error): the
//! Python tooling's limit rules run here. A `waivers` call validates a
//! `check`'s waivers (its `module`, `rules` and `waivers`) and returns
//! `{"valid": true}` or an `invalid` error; every call's check is held to
//! the same validator ([`declaration::validate_waivers`]), so a judge
//! never runs with waivers it refuses. The response (stdout) holds one
//! result per call, in order.

use serde_json::{Map, Value, json};

use crate::declaration::{self, Check};
use crate::kind::{self, Judged, Kind, KindError, MissingMetric};
use crate::kinds;

pub const REQUEST: &str = "chopinfer-eval-rescore/v1";
pub const RESPONSE: &str = "chopinfer-eval-rescore-result/v1";

/// The predicates as the Python tooling records them: each rule with its
/// result.
fn predicates(judged: &[Judged]) -> Value {
    judged.iter().map(Judged::to_json).collect()
}

/// A declared check from its JSON form.
fn check(value: &Value, kind: &dyn Kind, op: &str) -> Result<Check, String> {
    let declared = value.as_object().ok_or("check: expected an object")?;
    let judges = matches!(op, "rescore" | "judge");
    let fixed = matches!(kind.info().name, "greedy_pin" | "determinism");
    let rules = match declared.get("rules") {
        None if !judges || fixed => Vec::new(),
        Some(Value::Array(items)) if items.is_empty() && (!judges || fixed) => Vec::new(),
        value => declaration::rules(value.unwrap_or(&Value::Null), "check.rules")
            .map_err(|error| error.to_string())?,
    };
    if let Some(rule) = declared.get("rule")
        && !matches!(rule.as_str(), Some("exact" | "informational"))
    {
        return Err("check.rule: expected exact or informational".into());
    }
    let module = declared
        .get("module")
        .and_then(Value::as_str)
        .unwrap_or(kind.info().modules[0]);
    declaration::validate_waivers(value, module, &rules).map_err(|error| error.to_string())?;
    Ok(Check {
        id: declared
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .into(),
        module: module.into(),
        timeout_seconds: declared
            .get("timeout_seconds")
            .and_then(Value::as_f64)
            .unwrap_or(0.0),
        rules,
        informational_reason: declared
            .get("informational_reason")
            .and_then(Value::as_str)
            .map(String::from),
        declared: value.clone(),
    })
}

fn error(kind: &str, message: impl Into<String>) -> Value {
    json!({"error": {"type": kind, "message": message.into()}})
}

fn kind_error(failure: KindError) -> Value {
    match failure {
        KindError::Insufficient(message) => error("insufficient", message),
        KindError::Environment(message) => error("environment", message),
        KindError::Invalid(message) | KindError::Unsupported(message) => error("invalid", message),
    }
}

/// One call's result.
fn call(value: &Value, implemented: &[Box<dyn Kind>]) -> Result<Value, String> {
    if value["op"] == "limit" {
        let rule = value["rule"]
            .as_str()
            .ok_or("call: a limit call needs its rule")?;
        return Ok(match crate::limits::call(rule, &value["args"]) {
            Ok(result) => json!({"value": result}),
            Err(failure) => error("invalid", failure.0),
        });
    }
    if value["op"] == "waivers" {
        let check = &value["check"];
        let module = check["module"]
            .as_str()
            .ok_or("call: a waivers call needs the check's module")?;
        let rules = declaration::rules(&check["rules"], "check.rules")
            .map_err(|error| error.to_string())?;
        return Ok(match declaration::validate_waivers(check, module, &rules) {
            Ok(()) => json!({"valid": true}),
            Err(failure) => error("invalid", failure.to_string()),
        });
    }
    let name = value["kind"].as_str().ok_or("call: no kind")?;
    let kind = implemented
        .iter()
        .find(|kind| kind.info().name == name)
        .ok_or_else(|| format!("call: no shared kind {name}"))?
        .as_ref();
    let op = value["op"].as_str().ok_or("call: no op")?;
    let check = check(&value["check"], kind, op)?;
    if value
        .get("mechanics")
        .is_some_and(|value| !value.is_boolean())
    {
        return Err("call: mechanics must be a boolean".into());
    }
    let receipt = &value["receipt"];
    let reference = value
        .get("reference")
        .filter(|reference| !reference.is_null());
    let mechanics = value["mechanics"].as_bool().unwrap_or(false);
    Ok(match value["op"].as_str() {
        Some("rescore") => {
            let result = kind::rescore(kind, receipt, reference, &check, mechanics);
            let mut map = Map::new();
            map.insert("status".into(), json!(result.status.as_str()));
            map.insert(
                "metrics".into(),
                result.metrics.map_or(Value::Null, Value::Object),
            );
            map.insert(
                "predicates".into(),
                result.predicates.as_deref().map_or(Value::Null, predicates),
            );
            map.insert("recorded_fields".into(), json!(result.recorded_fields));
            if result.status == kind::Status::NotApplicable {
                map.insert("reason".into(), json!(result.reason));
            }
            if let Some(message) = result.error {
                map.insert("error".into(), json!(message));
            }
            Value::Object(map)
        }
        Some("verify") => match kind.verify(receipt, reference, &check) {
            Ok(fields) => json!({"recorded_fields": fields}),
            Err(failure) => kind_error(failure),
        },
        Some("score") => match kind.score(receipt, reference, &check, mechanics) {
            Ok(metrics) => json!({"metrics": metrics}),
            Err(failure) => kind_error(failure),
        },
        Some("score_capture") => {
            let reference = reference.ok_or("call: score_capture needs reference")?;
            let result = match name {
                "teacher_forced" => kinds::teacher::score_capture(receipt, reference, &check),
                "media_forced" => kinds::media::score_capture(receipt, reference, &check),
                _ => return Err("call: this kind has no capture scorer".into()),
            };
            match result {
                Ok(metrics) => json!({"metrics": metrics}),
                Err(failure) => kind_error(failure),
            }
        }
        Some("judge") => {
            let metrics = value["metrics"]
                .as_object()
                .ok_or("call: judge needs metrics")?;
            match kind.judge(metrics, &check) {
                Ok((status, judged)) => {
                    json!({"status": status.as_str(), "predicates": predicates(&judged)})
                }
                Err(MissingMetric(key)) => error("missing_metric", key),
            }
        }
        other => return Err(format!("call: unknown op {other:?}")),
    })
}

/// The response to a request, or why the request is malformed.
pub fn respond(request: &Value) -> Result<Value, String> {
    if request["schema"] != REQUEST {
        return Err(format!("the request schema is not {REQUEST}"));
    }
    let calls = request["calls"]
        .as_array()
        .ok_or("the request has no calls list")?;
    let implemented = kinds::all();
    let mut results = Vec::with_capacity(calls.len());
    for (index, value) in calls.iter().enumerate() {
        results.push(
            call(value, &implemented).map_err(|message| format!("calls[{index}]: {message}"))?,
        );
    }
    Ok(json!({"schema": RESPONSE, "results": results}))
}
