//! The image-probes format of `media_forced.v1`: per declared probe in the
//! captured order, a short answer's greedy margin within the reference's
//! near-tie rule, or a free-text answer teacher-forced within its own spread
//! limits, with its served top-1 identities judged by the near-tie rule
//! (docs/check-kinds.md#declared-media-capture-formats, `kinds.MediaForced`
//! `verify_probes`, `probe_flips`, `score_probes`).

use serde_json::{Map, Value, json};

use crate::json::{python_eq, truthy};
use crate::kind::{KindError, finite, natural};
use crate::kinds::{
    agrees, field, get, ids, invalid, judged, number, require, spread_of, top1_fields,
};
use crate::limits;
use crate::repr;

/// The receipt field holding a probe's scores: free-text probes are
/// teacher-forced, short ones walked greedily.
fn key(item: &Value) -> &'static str {
    if item["kind"] == json!("free-text") {
        "teacher_forced"
    } else {
        "greedy"
    }
}

/// The probe `name` of the reference (`reference["probes"][name]`).
fn probe<'a>(reference: &'a Value, name: &Value) -> Result<&'a Value, KindError> {
    let probes = field(reference, "probes")?;
    name.as_str()
        .and_then(|name| probes.get(name))
        .ok_or_else(|| invalid(repr::repr(name)))
}

/// The served top-1 identities a free-text probe's near-tie report records,
/// checked against the reference's forced rows and judged by the near-tie
/// rule against its stored top-k (`MediaForced.probe_flips`).
fn probe_flips(scores: &Value, item: &Value, where_: &str) -> Result<Vec<Value>, KindError> {
    let report = get(scores, "near_tie_report");
    let flips = report
        .filter(|report| report.is_object() && report.get("available") == Some(&Value::Bool(true)))
        .and_then(|report| report.get("flips"))
        .and_then(Value::as_array);
    require(flips.is_some(), || {
        format!("{where_}: no recorded probe top-1 identities")
    })?;
    let top1: Vec<Value> = ids(field(item, "forced")?)?
        .iter()
        .map(|row| {
            row.get(2)
                .cloned()
                .ok_or_else(|| invalid("list index out of range"))
        })
        .collect::<Result<_, _>>()?;
    let mut identities: Vec<(usize, Value)> = Vec::new();
    for row in flips.expect("checked") {
        let position = get(row, "position");
        let served = get(row, "served_top1");
        let valid = row.is_object()
            && position.is_some_and(natural)
            && position
                .and_then(Value::as_u64)
                .is_some_and(|at| (at as usize) < top1.len())
            && served.is_some_and(natural)
            && {
                let at = position.and_then(Value::as_u64).expect("checked") as usize;
                !python_eq(served.expect("checked"), &top1[at])
                    && get(row, "reference_top1").is_some_and(|stated| python_eq(stated, &top1[at]))
            };
        require(valid, || format!("{where_}: invalid probe top-1 identity"))?;
        identities.push((
            position.and_then(Value::as_u64).expect("checked") as usize,
            served.expect("checked").clone(),
        ));
    }
    let distinct = identities
        .iter()
        .enumerate()
        .all(|(index, (offset, _))| identities[..index].iter().all(|(other, _)| other != offset));
    require(
        distinct
            && !top1.is_empty()
            && agrees(
                get(scores, "top1_disagreement"),
                identities.len() as f64 / top1.len() as f64,
            ),
        || format!("{where_}: probe top-1 identities do not add up to the disagreement rate"),
    )?;
    let reference = json!({"top1": top1, "top5": field(item, "top")?});
    judged(
        identities.iter().map(|(offset, served)| (*offset, served)),
        &reference,
        &Map::new(),
    )
}

/// `MediaForced.verify_probes`: one row per declared probe in the captured
/// order, each with its expectation and token-count results and the scores
/// its kind needs.
pub fn verify(value: &Value, reference: &Value) -> Result<Vec<String>, KindError> {
    let stated = match get(value, "reference") {
        None => None,
        Some(Value::Object(map)) => map.get("sha256"),
        Some(other) => {
            return Err(invalid(format!(
                "'{}' object has no attribute 'get'",
                repr::type_name(other)
            )));
        }
    };
    require(stated == Some(field(reference, "sha256")?), || {
        "the receipt scored another media reference".into()
    })?;
    let probes = field(reference, "probes")?
        .as_object()
        .ok_or_else(|| invalid("the reference probes are not an object"))?;
    let order = get(reference, "order").and_then(Value::as_array);
    let covered = order.is_some_and(|order| {
        order.iter().all(Value::is_string)
            && order.len() == probes.len()
            && probes
                .keys()
                .all(|name| order.iter().any(|stated| stated == name))
            && order
                .iter()
                .all(|stated| probes.contains_key(stated.as_str().expect("checked")))
    });
    require(covered, || {
        "the reference order does not cover its probes".into()
    })?;
    let order = order.expect("checked");
    let rows = get(value, "rows").and_then(Value::as_array);
    let in_order = rows.is_some_and(|rows| {
        rows.len() == order.len()
            && rows
                .iter()
                .zip(order)
                .all(|(row, name)| python_eq(get(row, "id").unwrap_or(&Value::Null), name))
    });
    require(in_order, || {
        "the rows do not cover the declared probes in order".into()
    })?;
    for row in rows.expect("checked") {
        if crate::applicability::unserved_row(value, row, "image")? {
            continue;
        }
        let item = probe(reference, &row["id"])?;
        require(
            ["fixture_expectation_match", "token_count_match"]
                .iter()
                .all(|key| get(row, key).is_some_and(Value::is_boolean)),
            || "missing expectation or token count result".into(),
        )?;
        let key = key(item);
        let scores = get(row, key);
        require(
            scores.is_some_and(|scores| get(scores, "passed").is_some_and(Value::is_boolean)),
            || "missing probe scores".into(),
        )?;
        let scores = scores.expect("checked");
        if get(scores, "error").is_some() {
            require(get(row, "passed") == Some(&Value::Bool(false)), || {
                "probe passes without scores".into()
            })?;
            continue;
        }
        if key == "teacher_forced" {
            let count = get(scores, "scored");
            let mean = get(scores, "mean_abs_dlogprob");
            let rate = get(scores, "top1_disagreement");
            let consistent = count.is_some_and(|count| python_eq(count, &item["scored"]))
                && mean.is_some_and(finite)
                && mean
                    .and_then(crate::json::python_number)
                    .is_some_and(|mean| mean >= 0.0)
                && rate.is_some_and(finite)
                && {
                    let rate = crate::json::python_number(rate.expect("checked")).expect("finite");
                    let count = count
                        .and_then(crate::json::python_number)
                        .unwrap_or(f64::NAN);
                    (0.0..=1.0).contains(&rate)
                        && agrees(Some(&json!((rate * count).round_ties_even())), rate * count)
                };
            require(consistent, || "inconsistent probe scores".into())?;
            probe_flips(scores, item, &format!("probe {}", repr::str(&row["id"])))?;
        } else {
            let steps = ids(field(item, "steps")?)?;
            let divergence = get(scores, "divergence").filter(|value| !value.is_null());
            require(
                divergence.is_none_or(|divergence| {
                    natural(divergence)
                        && divergence
                            .as_u64()
                            .is_some_and(|at| (at as usize) < steps.len())
                }),
                || "probe divergence outside the answer".into(),
            )?;
            if !get(scores, "exact").is_some_and(truthy)
                && get(scores, "reference_margin").is_some()
            {
                let margin = &scores["reference_margin"];
                require(finite(margin), || "invalid greedy margin".into())?;
                if let Some(divergence) = divergence {
                    let step = &steps[divergence.as_u64().expect("checked") as usize];
                    let (token, chosen, top) =
                        (&step[0], number(&step[1], "logprob")?, ids(&step[2])?);
                    let listed = top.iter().any(|alternative| {
                        !python_eq(&alternative[0], token)
                            && crate::json::python_number(&alternative[1])
                                .is_some_and(|logprob| agrees(Some(margin), chosen - logprob))
                    });
                    require(listed, || "probe margin not in reference top-k".into())?;
                }
            }
        }
    }
    Ok(vec!["rows".into()])
}

/// `MediaForced.score_probes`: each probe's verdict recomputed: a
/// free-text probe's flips, rates and limits, a short probe's greedy margin
/// against the near-tie rule; every probe's verdict and the receipt's.
pub fn score(
    value: &Value,
    reference: &Value,
    metrics: &mut Map<String, Value>,
) -> Result<(), KindError> {
    let near_tie = number(field(field(reference, "rule")?, "near_tie")?, "near_tie")?;
    let mut rows = Vec::new();
    for row in ids(field(value, "rows")?)? {
        if crate::applicability::unserved_row(value, row, "image")? {
            rows.push(row.clone());
            continue;
        }
        let item = probe(reference, field(row, "id")?)?;
        let key = key(item);
        let mut scores = field(row, key)?.clone();
        if get(&scores, "error").is_none() {
            if key == "teacher_forced" {
                let flips =
                    probe_flips(&scores, item, &format!("probe {}", repr::str(&row["id"])))?;
                let scored = crate::kinds::count(field(&scores, "scored")?, "scored")?;
                let item_limits = field(item, "limits")?.clone();
                let map = scores
                    .as_object_mut()
                    .ok_or_else(|| invalid("probe scores are not an object"))?;
                map.extend(top1_fields(&flips, scored));
                map.insert("limits".into(), item_limits.clone());
                let passed = limits::within(&spread_of(&scores)?, &spread_of(&item_limits)?);
                scores["passed"] = json!(passed);
            } else {
                // Legacy probe receipts retain the divergent token's reference
                // margin, not its bytes: an exact answer passes, a margin within
                // the near-tie rule passes, anything else fails.
                let margin = get(&scores, "reference_margin");
                let passed = get(&scores, "exact") == Some(&Value::Bool(true))
                    || (margin.is_some_and(finite)
                        && get(&scores, "divergence")
                            .is_some_and(|divergence| !divergence.is_null())
                        && limits::near_tie(margin.and_then(crate::json::python_number), near_tie));
                scores["passed"] = json!(passed);
            }
        }
        // Python's `a and b and c` gives the first false operand, else the last.
        let operands = [
            field(&scores, "passed")?.clone(),
            field(row, "fixture_expectation_match")?.clone(),
            field(row, "token_count_match")?.clone(),
        ];
        let passed = operands
            .iter()
            .find(|operand| !truthy(operand))
            .unwrap_or(&operands[2])
            .clone();
        let mut entry = row
            .as_object()
            .cloned()
            .ok_or_else(|| invalid("a row is not an object"))?;
        entry.insert(key.into(), scores);
        entry.insert("passed".into(), passed);
        rows.push(Value::Object(entry));
    }
    let passed = crate::applicability::passed(&rows);
    let free_text: Map<String, Value> = field(reference, "probes")?
        .as_object()
        .into_iter()
        .flatten()
        .filter(|(_, item)| item["kind"] == json!("free-text"))
        .map(|(name, item)| (name.clone(), item["limits"].clone()))
        .collect();
    metrics.insert("rows".into(), Value::Array(rows));
    metrics.insert("passed".into(), json!(passed));
    metrics.insert(
        "limits".into(),
        json!({"near_tie": reference["rule"]["near_tie"], "teacher_forced": free_text}),
    );
    Ok(())
}

/// Judge captured answer bytes against the image probe's reference steps.
pub(crate) fn score_greedy(
    captured: &Value,
    reference: &Value,
) -> Result<Map<String, Value>, KindError> {
    let expected = ids(field(field(reference, "answer")?, "tokens")?)?;
    let actual = ids(field(captured, "tokens")?)?;
    if python_eq(&json!(actual), &json!(expected)) {
        return Ok(json!({"passed": true, "exact": true})
            .as_object()
            .cloned()
            .expect("object"));
    }
    for (position, (want, got)) in expected.iter().zip(actual).enumerate() {
        if python_eq(want, got) {
            continue;
        }
        let step = ids(field(reference, "steps")?)?
            .get(position)
            .ok_or_else(|| invalid("missing reference step"))?;
        let alternatives = ids(&step[2])?;
        let mut result = json!({"exact": false, "divergence": position})
            .as_object()
            .cloned()
            .expect("object");
        let Some(alternative) = alternatives
            .iter()
            .rev()
            .find(|row| python_eq(&row[2], got))
        else {
            result.extend(
                json!({"passed": false, "reason": "divergent token outside reference top-k"})
                    .as_object()
                    .cloned()
                    .expect("object"),
            );
            return Ok(result);
        };
        let margin = number(&step[1], "reference logprob")?
            - number(&alternative[1], "alternative logprob")?;
        let limit = number(field(captured, "near_tie")?, "near_tie")?;
        result.insert("reference_margin".into(), json!(margin));
        result.insert(
            "passed".into(),
            json!(limits::near_tie(Some(margin), limit)),
        );
        return Ok(result);
    }
    Ok(json!({"passed": false, "exact": false, "reason": "answer length differs without divergence"})
        .as_object().cloned().expect("object"))
}

/// Judge captured forced-answer columns with the image probe's spread limits.
pub(crate) fn score_forced(
    captured: &Value,
    reference: &Value,
) -> Result<Map<String, Value>, KindError> {
    let forced = ids(field(reference, "forced")?)?;
    let logprobs = ids(field(captured, "logprob")?)?;
    let served = ids(field(captured, "top1")?)?;
    let tops = crate::kinds::top(&json!({"top5": field(reference, "top")?}))?;
    if forced.is_empty()
        || forced.len() != logprobs.len()
        || forced.len() != served.len()
        || forced.len() != tops.len()
        || reference["scored"].as_u64() != Some(forced.len() as u64)
    {
        return Err(invalid("forced probe columns do not cover the reference"));
    }
    let differences = forced
        .iter()
        .zip(logprobs)
        .map(|(row, logprob)| {
            Ok((number(logprob, "served logprob")? - number(&row[1], "reference logprob")?).abs())
        })
        .collect::<Result<Vec<_>, KindError>>()?;
    let flips = forced
        .iter()
        .zip(served)
        .zip(&tops)
        .map(|((row, token), top)| limits::flip(&row[2], token, Some(top), limits::NEAR_TIE_NATS))
        .collect::<Vec<_>>();
    let mut result = serde_json::to_value(limits::top1_summary(&flips, forced.len() as u64))
        .map_err(|error| invalid(error.to_string()))?
        .as_object()
        .cloned()
        .expect("object");
    result.insert("scored".into(), json!(forced.len()));
    result.insert(
        "mean_abs_dlogprob".into(),
        json!(crate::exact::mean(&differences)),
    );
    let limits = field(reference, "limits")?;
    result.insert("limits".into(), limits.clone());
    let passed = limits::within(
        &spread_of(&Value::Object(result.clone()))?,
        &spread_of(limits)?,
    );
    result.insert("passed".into(), json!(passed));
    Ok(result)
}
