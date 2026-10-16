//! `greedy_pin.v1`: C1 greedy tokens of every request bit-equal to a
//! baseline record, the pinned approved capture or the same run's AR arm
//! (docs/check-kinds.md#greedy_pin-greedy_pinv1).

use serde_json::{Map, Value, json};

use crate::declaration::Check;
use crate::json::{python_eq, truthy};
use crate::kind::{
    Capture, Judged, Kind, KindError, KindInfo, MissingMetric, SHARED_KINDS, Status, natural,
};
use crate::kinds::{applied_constants, get, ids, invalid, require};
use crate::repr;

pub struct GreedyPin;

/// The workload fields both records must hold.
const WORKLOAD: [&str; 3] = ["prompt_tokens", "max_tokens", "concurrency"];

/// Insufficient unless `record` holds every request's token IDs, its
/// finish reasons and the workload fields (`kinds.check_greedy`).
fn check_greedy(record: &Value, where_: &str) -> Result<(), KindError> {
    let requests = get(record, "requests").and_then(Value::as_object);
    require(
        requests.is_some_and(|requests| {
            !requests.is_empty()
                && requests
                    .values()
                    .all(|ids| ids.as_array().is_some_and(|ids| ids.iter().all(natural)))
        }),
        || format!("{where_}: no requests"),
    )?;
    let requests = requests.expect("checked");
    let finishes = get(record, "finish_reasons").and_then(Value::as_object);
    require(
        finishes.is_some_and(|finishes| {
            finishes.len() == requests.len()
                && finishes.keys().all(|name| requests.contains_key(name))
        }),
        || format!("{where_}: finish reasons do not cover the requests"),
    )?;
    require(
        WORKLOAD
            .iter()
            .all(|key| get(record, key).is_some_and(natural))
            && get(record, "reference_sha256").is_some_and(Value::is_string),
        || format!("{where_}: missing workload fields"),
    )
}

/// The first position where two token lists differ, a missing token as
/// null (`greedy_equality.first_difference` of one request).
fn first_difference(left: &[Value], right: &[Value]) -> Option<(usize, Value, Value)> {
    (0..left.len().max(right.len())).find_map(|position| {
        let a = left.get(position).cloned().unwrap_or(Value::Null);
        let b = right.get(position).cloned().unwrap_or(Value::Null);
        (!python_eq(&a, &b)).then_some((position, a, b))
    })
}

impl Kind for GreedyPin {
    fn info(&self) -> &'static KindInfo {
        &SHARED_KINDS[2]
    }

    /// The pinned exact reference whose texts' prompt prefixes run.
    fn reference_name(&self, _check: &Check) -> Option<String> {
        Some("exact".into())
    }

    fn verify(
        &self,
        value: &Value,
        reference: Option<&Value>,
        check: &Check,
    ) -> Result<Vec<String>, KindError> {
        applied_constants(check, &[])?;
        let reference =
            reference.ok_or_else(|| KindError::Insufficient("no baseline record".into()))?;
        check_greedy(value, "record")?;
        check_greedy(reference, "baseline")?;
        Ok(Vec::new())
    }

    /// The comparison with the baseline record `reference`; a different
    /// workload fails (`kinds.GreedyPin.score`).
    fn score(
        &self,
        value: &Value,
        reference: Option<&Value>,
        _check: &Check,
        _mechanics: bool,
    ) -> Result<Map<String, Value>, KindError> {
        let reference =
            reference.ok_or_else(|| invalid("'NoneType' object is not subscriptable"))?;
        let read = |record: &Value, key: &str| {
            get(record, key)
                .cloned()
                .ok_or_else(|| invalid(format!("'{key}'")))
        };
        let same = |condition: bool, message: &str| {
            if condition {
                Ok(())
            } else {
                Err(invalid(message))
            }
        };
        same(
            python_eq(
                &read(reference, "reference_sha256")?,
                &read(value, "reference_sha256")?,
            ),
            "greedy exact reference changed",
        )?;
        for key in WORKLOAD {
            same(
                python_eq(&read(reference, key)?, &read(value, key)?),
                &format!("greedy workload mismatch: {key}"),
            )?;
        }
        let object = |record: &Value, key: &str| -> Result<Map<String, Value>, KindError> {
            read(record, key)?
                .as_object()
                .cloned()
                .ok_or_else(|| invalid(format!("{key} is not an object")))
        };
        let (baseline, requests) = (object(reference, "requests")?, object(value, "requests")?);
        same(
            baseline.len() == requests.len()
                && baseline.keys().all(|name| requests.contains_key(name)),
            "greedy request coverage changed",
        )?;
        same(
            !baseline.is_empty() && baseline.values().all(truthy) && requests.values().all(truthy),
            "empty greedy tokens",
        )?;
        let (baseline_finishes, finishes) = (
            object(reference, "finish_reasons")?,
            object(value, "finish_reasons")?,
        );
        for (record, reasons) in [(&baseline, &baseline_finishes), (&requests, &finishes)] {
            same(
                reasons.len() == record.len()
                    && reasons.keys().all(|name| record.contains_key(name))
                    && reasons
                        .values()
                        .all(|reason| reason == "stop" || reason == "length"),
                "invalid greedy finishes",
            )?;
        }
        let mut differences = Vec::new();
        let mut finish_differences = Vec::new();
        for (name, tokens) in &baseline {
            let other = &requests[name];
            let (left, right) = (ids(tokens)?, ids(other)?);
            if let Some((position, a, b)) = first_difference(left, right) {
                differences.push(
                    json!({"request": name, "position": position, "ar_token": a, "other_token": b}),
                );
            }
            if !python_eq(&baseline_finishes[name], &finishes[name]) {
                finish_differences.push(
                    json!({"request": name, "ar": baseline_finishes[name], "other": finishes[name]}),
                );
            }
        }
        let mut metrics = Map::new();
        metrics.insert(
            "equal".into(),
            json!(differences.is_empty() && finish_differences.is_empty()),
        );
        metrics.insert("first_differences".into(), Value::Array(differences));
        metrics.insert(
            "finish_differences".into(),
            Value::Array(finish_differences),
        );
        Ok(metrics)
    }

    /// Rule `exact` fails on any difference; rule `informational` reports it.
    fn judge(
        &self,
        metrics: &Map<String, Value>,
        check: &Check,
    ) -> Result<(Status, Vec<Judged>), MissingMetric> {
        let equal = metrics
            .get("equal")
            .ok_or_else(|| MissingMetric("'equal'".into()))?;
        if truthy(equal) {
            return Ok((Status::Pass, Vec::new()));
        }
        let rule = check
            .declared
            .get("rule")
            .ok_or_else(|| MissingMetric("'rule'".into()))?;
        let informational = rule == "informational";
        Ok((
            if informational {
                Status::Informational
            } else {
                Status::Fail
            },
            Vec::new(),
        ))
    }

    fn capture(&self, capture: &mut Capture<'_>) -> Result<Value, KindError> {
        capture_greedy(capture)
    }
}

/// A non-negative integer argument of the check, else its default.
fn integer(check: &Check, option: &str, default: u64) -> Result<u64, KindError> {
    match crate::capture::option(check, option) {
        None => Ok(default),
        Some(text) => text.parse().map_err(|_| {
            invalid(format!(
                "argument {option}: invalid int value: {}",
                repr::repr(&json!(text))
            ))
        }),
    }
}

/// The greedy record: every text of the pinned exact reference run greedy
/// and unforced from its prompt prefix, each with a fresh cache salt and zero
/// cached tokens, one at a time. Concurrent greedy records are not supported.
fn capture_greedy(capture: &mut Capture<'_>) -> Result<Value, KindError> {
    let check = capture.check;
    let arm = crate::capture::option(check, "--arm")
        .ok_or_else(|| invalid("record needs --arm and concurrency >= 1"))?;
    let k = integer(check, "--k", 0)?;
    let concurrency = integer(check, "--concurrency", 1)?;
    let prompt_tokens = integer(check, "--prompt-tokens", 256)?;
    let max_tokens = integer(check, "--max-tokens", 256)?;
    if concurrency < 1 {
        return Err(invalid("record needs --arm and concurrency >= 1"));
    }
    if concurrency > 1 {
        return Err(KindError::Unsupported(
            "chopinfer eval records greedy requests one at a time; concurrent greedy records are not supported".into(),
        ));
    }
    let verified = capture
        .reference
        .ok_or_else(|| KindError::Insufficient("the gate pins no exact reference".into()))?;
    let texts = ids(&verified.value["texts"])?.to_vec();
    let model = capture.server.model.clone();
    let mut requests = Map::new();
    let mut finishes = Map::new();
    for text in &texts {
        crate::capture::remaining(capture)?;
        let all = ids(&text["prompt_token_ids"])?;
        let length = (prompt_tokens as usize).min(all.len().saturating_sub(1));
        let body = json!({"model": model, "prompt": all[..length], "max_tokens": max_tokens,
                          "temperature": 0, "top_p": 1, "return_token_ids": true,
                          "skip_special_tokens": false,
                          "cache_salt": format!("greedy-cold-{}", crate::capture::salt()?)});
        let (response, _) = capture
            .client
            .json("/v1/completions", Some(&body))
            .map_err(crate::capture::http_error)?;
        if response["usage"]["prompt_tokens_details"]["cached_tokens"] != json!(0) {
            return Err(invalid(
                "cold greedy request requires reported cached_tokens=0",
            ));
        }
        let choice = &response["choices"][0];
        let tokens = Some(&choice["token_ids"])
            .filter(|tokens| truthy(tokens))
            .cloned()
            .unwrap_or(json!([]));
        let reason = choice
            .get("finish_reason")
            .cloned()
            .ok_or_else(|| invalid("'finish_reason'"))?;
        let name = repr::str(&text["name"]);
        requests.insert(name.clone(), tokens);
        finishes.insert(name, reason);
    }
    let mut result = Map::new();
    result.insert("arm".into(), json!(arm));
    result.insert("k".into(), json!(k));
    result.insert("concurrency".into(), json!(concurrency));
    result.insert("prompt_tokens".into(), json!(prompt_tokens));
    result.insert("max_tokens".into(), json!(max_tokens));
    result.insert("cold_cache".into(), json!(true));
    result.insert("reference_sha256".into(), json!(verified.sha256));
    result.insert("requests".into(), Value::Object(requests));
    result.insert("finish_reasons".into(), Value::Object(finishes));
    result.insert("passed".into(), json!(true));
    Ok(crate::capture::receipt(
        &format!("greedy-equality-{arm}"),
        capture,
        result,
    ))
}
