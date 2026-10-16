//! `media_forced.v1`: media probes, each reference item's answer
//! teacher-forced against the pinned media reference's gate variant within
//! the item's own spread limits, its greedy answer walked along the
//! reference answer, its content and prompt token count checked
//! (docs/check-kinds.md#media_forced-media_forcedv1).

use serde_json::{Map, Value, json};

use crate::declaration::Check;
use crate::json::python_eq;
use crate::kind::{
    Kind, KindError, KindInfo, SHARED_KINDS, finite, match_flips, natural, served_flips,
};
use crate::kinds::{
    AGGREGATE_TOLERANCE, POWER_VARIANT, agrees, applied_constants, check_reference_scores,
    constant, count, field, flips_by_offset, get, ids, invalid, judged, number, require,
    set_verdict, spread_json, spread_of, top, top1_fields, walk,
};
use crate::limits::{self, SpreadRule};
use crate::repr;

pub struct MediaForced;

/// The item fields every row holds.
const ITEM: [&str; 4] = [
    "logprobs",
    "greedy",
    "fixture_expectation_match",
    "token_count_match",
];

/// The constants of `multimodal.py` this kind applies.
const CONSTANTS: [(&str, &str); 7] = [
    ("multimodal", "GATE_VARIANT"),
    ("multimodal", "MEAN_ABS_LOGPROB_LIMIT"),
    ("multimodal", "LARGE_SET_MEAN_LIMIT"),
    ("multimodal", "LARGE_SET_POSITIONS"),
    ("multimodal", "TOP1_DISAGREEMENT_LIMIT"),
    ("multimodal", "MEAN_STEP"),
    ("multimodal", "NEAR_TIE_LIMIT"),
];

fn multimodal(check: &Check, name: &str, default: Value) -> Value {
    constant(check, "multimodal", name, default)
}

/// `multimodal.case_limits`: the floors, raised to the item's reference
/// variants scored against the gate variant, and the flips of the run the
/// power test reads (`POWER_VARIANT`) at each position, if pinned.
fn case_limits(case: &Value, gate: &str, check: &Check) -> Result<CaseLimits, KindError> {
    let exact = field(case, "exact")?
        .as_object()
        .ok_or_else(|| invalid("the item's variants are not an object"))?;
    let reference = field(case, "exact")?;
    let gate_scores = field(reference, gate)?;
    let gate_logprobs = ids(field(gate_scores, "logprob")?)?;
    let gate_top1 = ids(field(gate_scores, "top1")?)?;
    let tops = limits::gap_top(gate_scores)
        .ok_or_else(|| invalid("the gate variant has no top-k rows"))?;
    let mut spread = Vec::new();
    let mut power = None;
    for (name, rows) in exact {
        if name == gate {
            continue;
        }
        let logprobs = ids(field(rows, "logprob")?)?;
        let served = ids(field(rows, "top1")?)?;
        if logprobs.len() != gate_logprobs.len()
            || served.len() != gate_top1.len()
            || tops.len() != served.len()
        {
            return Err(invalid("zip() arguments differ in length"));
        }
        // Python's sum starts from 0 and adds in order.
        let mut total = 0.0;
        for (a, b) in logprobs.iter().zip(gate_logprobs) {
            total += (number(a, "logprob")? - number(b, "logprob")?).abs();
        }
        let positions = logprobs.len();
        let flips: Vec<Option<limits::Flip>> = served
            .iter()
            .zip(gate_top1)
            .zip(&tops)
            .map(|((token, top1), pairs)| {
                limits::flip(top1, token, Some(pairs), limits::NEAR_TIE_NATS)
            })
            .collect();
        let summary = limits::top1_summary(&flips, positions as u64);
        if name == POWER_VARIANT {
            power = Some(flips.clone());
        }
        let rate = |value: Option<f64>| value.ok_or_else(|| invalid("division by zero"));
        spread.push(limits::Spread {
            mean_abs_dlogprob: total / positions as f64,
            top1_decisive: rate(summary.top1_decisive)?,
            top1_near_tie: rate(summary.top1_near_tie)?,
            top1_disagreement: rate(summary.top1_disagreement)?,
        });
    }
    let read = |name: &str, default: f64| number(&multimodal(check, name, json!(default)), name);
    let positions = multimodal(
        check,
        "LARGE_SET_POSITIONS",
        json!(limits::LARGE_SET_POSITIONS),
    );
    let rule = SpreadRule {
        mean_floor: read("MEAN_ABS_LOGPROB_LIMIT", limits::MEAN_FLOOR)?,
        large_set_mean_floor: read("LARGE_SET_MEAN_LIMIT", limits::LARGE_SET_MEAN_FLOOR)?,
        large_set_positions: number(&positions, "LARGE_SET_POSITIONS")?.ceil() as u64,
        top1_floor: read("TOP1_DISAGREEMENT_LIMIT", limits::TOP1_FLOOR)?,
        mean_step: read("MEAN_STEP", limits::MEAN_STEP)?,
        ..SpreadRule::default()
    };
    let limits = limits::spread_limits(&spread, gate_logprobs.len() as u64, &rule)
        .map_err(|error| invalid(error.0))?;
    Ok((limits, power))
}

/// A media case's limits and its power-test run's flips.
type CaseLimits = (limits::Spread, Option<Vec<Option<limits::Flip>>>);

/// `multimodal.receipt_limits`: the floors and near-tie limits a media
/// receipt records.
fn receipt_limits(check: &Check, audio_near_tie: &Value) -> Value {
    json!({
        "mean_abs_dlogprob_floor": multimodal(check, "MEAN_ABS_LOGPROB_LIMIT", json!(limits::MEAN_FLOOR)),
        "top1_decisive_floor": multimodal(check, "TOP1_DISAGREEMENT_LIMIT", json!(limits::TOP1_FLOOR)),
        "top1_near_tie_floor": limits::NEAR_TIE_FLOOR,
        "top1_disagreement_floor": limits::TOP1_TOTAL_FLOOR,
        "top1_near_tie": limits::NEAR_TIE_NATS,
        "near_tie": multimodal(check, "NEAR_TIE_LIMIT", json!(limits::NEAR_TIE_NATS)),
        "audio_near_tie": audio_near_tie,
    })
}

/// Whether the reference is of the image-probes capture format.
fn image_probes(reference: &Value) -> bool {
    get(reference, "format") == Some(&json!("image-probes"))
}

fn answer_tokens(case: &Value) -> Option<&[Value]> {
    get(case, "answer_token_ids")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
}

/// Mixed cases gate only the forced answer, including exact prompt placement.
/// Recompute from retained answer columns instead of trusting saved verdicts.
fn mixed_metrics(
    value: &Value,
    reference: &Value,
    check: &Check,
) -> Result<Option<Value>, KindError> {
    let Some(reference) = reference.get("mixed") else {
        return Ok(None);
    };
    let actual = field(value, "mixed")?;
    require(
        actual.get("reference_sha256") == reference.get("sha256"),
        || "mixed receipt scored another reference".into(),
    )?;
    let cases = ids(field(reference, "cases")?)?;
    let rows = ids(field(actual, "rows")?)?;
    require(
        !cases.is_empty()
            && rows.len() == cases.len()
            && actual["case_count"] == json!(cases.len()),
        || "mixed rows do not cover the reference cases".into(),
    )?;
    let mut result = Vec::new();
    for (row, case) in rows.iter().zip(cases) {
        require(ids(field(case, "answer_token_ids")?)?.len() >= 32, || {
            "mixed reference needs at least 32 scored positions per case".into()
        })?;
        require(row.get("id") == case.get("id"), || {
            "mixed case order differs".into()
        })?;
        if crate::applicability::unserved_row(
            value,
            row,
            case["kind"].as_str().unwrap_or_default(),
        )? {
            result.push(row.clone());
            continue;
        }
        if row.get("error").is_some() {
            require(row["passed"] == json!(false), || {
                "mixed error row cannot pass".into()
            })?;
            result.push(row.clone());
            continue;
        }
        let mut prompt = ids(field(case, "prompt_token_ids")?)?.to_vec();
        prompt.extend_from_slice(ids(field(case, "answer_token_ids")?)?);
        require(row.get("prompt_token_ids") == Some(&json!(prompt)), || {
            "mixed prompt token positions differ from the reference".into()
        })?;
        let captured = field(row, "capture")?;
        require(ids(field(captured, "top1")?)?.iter().all(natural), || {
            "mixed top-1 token IDs are invalid".into()
        })?;
        let scores = score_capture(captured, case, check)?;
        let mut observed = row
            .as_object()
            .cloned()
            .ok_or_else(|| invalid("mixed row is not an object"))?;
        observed.insert("passed".into(), scores["passed"].clone());
        observed.insert("prompt_token_ids_match".into(), json!(true));
        observed.insert("logprobs".into(), Value::Object(scores));
        result.push(Value::Object(observed));
    }
    let mut metrics = actual
        .as_object()
        .cloned()
        .ok_or_else(|| invalid("mixed receipt is not an object"))?;
    metrics.insert(
        "passed".into(),
        json!(crate::applicability::passed(&result)),
    );
    metrics.insert("rows".into(), json!(result));
    Ok(Some(Value::Object(metrics)))
}

impl Kind for MediaForced {
    fn info(&self) -> &'static KindInfo {
        &SHARED_KINDS[1]
    }

    fn verify(
        &self,
        value: &Value,
        reference: Option<&Value>,
        check: &Check,
    ) -> Result<Vec<String>, KindError> {
        applied_constants(check, &CONSTANTS)?;
        let reference = reference.ok_or_else(|| {
            KindError::Insufficient("no media reference to verify the items against".into())
        })?;
        if image_probes(reference) {
            return crate::kinds::probes::verify(value, reference);
        }
        require(
            get(value, "exact_reference_sha256") == Some(field(reference, "sha256")?),
            || "the receipt scored another media reference".into(),
        )?;
        let cases = ids(field(reference, "cases")?)?;
        let gate = repr::str(field(reference, "gate")?);
        check_reference_scores(
            cases,
            ids(field(reference, "variants")?)?,
            &gate,
            answer_tokens,
            |case| get(case, "exact").filter(|exact| crate::json::truthy(exact)),
        )?;
        let rows = get(value, "rows").and_then(Value::as_array);
        let in_order = rows.is_some_and(|rows| {
            rows.len() == cases.len()
                && rows.iter().zip(cases).all(|(row, case)| {
                    python_eq(get(row, "id").unwrap_or(&Value::Null), &case["id"])
                })
        });
        require(in_order, || {
            format!(
                "the rows cover {} items, not the reference's {} in order",
                rows.map_or(0, Vec::len),
                cases.len()
            )
        })?;
        for (row, case) in rows.expect("checked").iter().zip(cases) {
            if crate::applicability::unserved_row(
                value,
                row,
                case["kind"].as_str().unwrap_or_default(),
            )? {
                continue;
            }
            let where_ = format!("item {}", repr::str(&case["id"]));
            require(
                get(row, "kind").is_some_and(|kind| python_eq(kind, &case["kind"]))
                    && ITEM.iter().all(|key| get(row, key).is_some()),
                || format!("{where_}: missing kind or {}", ITEM.join(", ")),
            )?;
            let greedy = &row["greedy"];
            let answer = answer_tokens(case).map_or(0, <[Value]>::len);
            require(
                greedy.is_object()
                    && get(greedy, "passed").is_some_and(Value::is_boolean)
                    && row["fixture_expectation_match"].is_boolean()
                    && row["token_count_match"].is_boolean(),
                || format!("{where_}: invalid greedy, expectation or token count result"),
            )?;
            let divergence = get(greedy, "divergence").filter(|value| !value.is_null());
            require(
                divergence.is_none_or(|divergence| {
                    natural(divergence)
                        && divergence.as_u64().is_some_and(|at| (at as usize) < answer)
                }),
                || {
                    format!(
                        "{where_}: greedy divergence {} outside the {answer}-token answer",
                        repr::str_of(get(greedy, "divergence"))
                    )
                },
            )?;
            let served = get(greedy, "served_token_id").filter(|value| !value.is_null());
            require(
                served.is_none_or(|served| natural(served) && divergence.is_some()),
                || format!("{where_}: a served greedy token without its divergence"),
            )?;
            let scores = &row["logprobs"];
            require(scores.is_object(), || format!("{where_}: no logprobs"))?;
            if get(scores, "error").is_some() {
                require(get(row, "passed") == Some(&Value::Bool(false)), || {
                    format!("{where_}: passes without scores")
                })?;
                continue;
            }
            require(
                get(scores, "scored").is_some_and(|scored| python_eq(scored, &json!(answer))),
                || {
                    format!(
                        "{where_}: scored {} of {answer} answer positions",
                        repr::str_of(get(scores, "scored"))
                    )
                },
            )?;
            let (mean, largest, rate) = (
                get(scores, "mean_abs_dlogprob"),
                get(scores, "max_abs_dlogprob"),
                get(scores, "top1_disagreement"),
            );
            let ordered = match (mean, largest) {
                (Some(mean), Some(largest)) if finite(mean) && finite(largest) => {
                    let (mean, largest) = (number(mean, "mean")?, number(largest, "max")?);
                    0.0 <= mean && mean <= largest + AGGREGATE_TOLERANCE
                }
                _ => false,
            };
            require(ordered, || {
                format!(
                    "{where_}: mean {} and max {} |dlogprob| are not 0 <= mean <= max",
                    repr::str_of(mean),
                    repr::str_of(largest)
                )
            })?;
            let whole = rate
                .filter(|rate| finite(rate))
                .map(|rate| number(rate, "rate").expect("finite"));
            require(
                whole.is_some_and(|rate| {
                    (0.0..=1.0).contains(&rate)
                        && agrees(
                            Some(&json!((rate * answer as f64).round_ties_even())),
                            rate * answer as f64,
                        )
                }),
                || {
                    format!(
                        "{where_}: top-1 disagreement {} is not a count of the {answer} positions",
                        repr::str_of(rate)
                    )
                },
            )?;
            let gate_scores = &case["exact"][gate.as_str()];
            let derived = served_flips(
                Some(scores),
                answer_tokens(case).unwrap_or(&[]),
                ids(field(gate_scores, "top1")?)?,
                &where_,
            )?;
            match_flips(get(scores, "top1_flips"), &derived, &where_)?;
            let expected = (whole.expect("checked") * answer as f64).round_ties_even();
            require(
                scores["top1_flips"].as_array().map_or(0, Vec::len) as f64 == expected,
                || {
                    format!(
                        "{where_}: the top-1 flips do not add up to its disagreement {}",
                        repr::str_of(rate)
                    )
                },
            )?;
        }
        if mixed_metrics(value, reference, check)?.is_some() {
            return Ok(vec!["rows".into(), "mixed".into()]);
        }
        Ok(vec!["rows".into()])
    }

    fn score(
        &self,
        value: &Value,
        reference: Option<&Value>,
        check: &Check,
        mechanics: bool,
    ) -> Result<Map<String, Value>, KindError> {
        let mut metrics = crate::kind::default_score(self, value, check, mechanics)?;
        let Some(reference) = reference else {
            return Ok(metrics);
        };
        if image_probes(reference) {
            crate::kinds::probes::score(value, reference, &mut metrics)?;
            return Ok(metrics);
        }
        let gate = repr::str(field(reference, "gate")?);
        let near_tie = number(
            &multimodal(check, "NEAR_TIE_LIMIT", json!(limits::NEAR_TIE_NATS)),
            "NEAR_TIE_LIMIT",
        )?;
        // The declared policy's audio limit, which the reference carries.
        let audio_limit = get(reference, "audio_near_tie")
            .filter(|limit| !limit.is_null())
            .cloned()
            .ok_or_else(|| invalid("the media reference declares no audio near-tie limit"))?;
        let audio = number(&audio_limit, "audio_near_tie")?;
        let mut rows = Vec::new();
        for (row, case) in ids(field(value, "rows")?)?
            .iter()
            .zip(ids(field(reference, "cases")?)?)
        {
            if crate::applicability::unserved_row(
                value,
                row,
                case["kind"].as_str().unwrap_or_default(),
            )? {
                rows.push(row.clone());
                continue;
            }
            let mut scores = field(row, "logprobs")?.clone();
            let gate_scores = field(field(case, "exact")?, &gate)?;
            let tokens =
                answer_tokens(case).ok_or_else(|| invalid("the item has no answer token IDs"))?;
            if get(&scores, "error").is_none() {
                let where_ = format!("item {}", repr::str(&case["id"]));
                let derived = served_flips(
                    Some(&scores),
                    tokens,
                    ids(field(gate_scores, "top1")?)?,
                    &where_,
                )?;
                let flips = judged(
                    derived.iter().map(|(offset, served)| (*offset, served)),
                    gate_scores,
                    &Map::new(),
                )?;
                let scored = count(field(&scores, "scored")?, "scored")?;
                let map = scores.as_object_mut().expect("verified an object");
                map.extend(top1_fields(&flips, scored));
                map.insert("top1_flips".into(), Value::Array(flips));
                let (limits, power) = case_limits(case, &gate, check)?;
                let map = scores.as_object_mut().expect("an object");
                for key in ["spread_rule", "passed_by", "power", "paired_reference"] {
                    map.remove(key);
                }
                let candidate = flips_by_offset(
                    map["top1_flips"].as_array().map_or(&[][..], Vec::as_slice),
                    scored as usize,
                )?;
                let (passed, verdict) =
                    set_verdict(&spread_of(&scores)?, &limits, &candidate, power.as_deref())?;
                let map = scores.as_object_mut().expect("an object");
                map.insert("limits".into(), spread_json(&limits));
                map.extend(verdict);
                map.insert("passed".into(), json!(passed));
            }
            let limit = if case["kind"] == json!("audio") {
                audio
            } else {
                near_tie
            };
            let recorded = field(row, "greedy")?;
            let mut greedy = recorded
                .as_object()
                .cloned()
                .ok_or_else(|| invalid("greedy is not an object"))?;
            greedy.insert(
                "near_tie_limit".into(),
                if case["kind"] == json!("audio") {
                    audio_limit.clone()
                } else {
                    multimodal(check, "NEAR_TIE_LIMIT", json!(limit))
                },
            );
            greedy.extend(walk(tokens, &top(gate_scores)?, limit, recorded)?);
            // Python's `a and b and c and d` gives the first false operand,
            // else the last.
            let operands = [
                greedy["passed"].clone(),
                field(row, "fixture_expectation_match")?.clone(),
                field(row, "token_count_match")?.clone(),
                field(&scores, "passed")?.clone(),
            ];
            let passed = operands
                .iter()
                .find(|operand| !crate::json::truthy(operand))
                .unwrap_or(&operands[3])
                .clone();
            let mut item = row
                .as_object()
                .cloned()
                .ok_or_else(|| invalid("a row is not an object"))?;
            item.insert("logprobs".into(), scores);
            item.insert("greedy".into(), Value::Object(greedy));
            item.insert("passed".into(), passed);
            rows.push(Value::Object(item));
        }
        let mut passed = crate::applicability::passed(&rows);
        if let Some(mixed) = mixed_metrics(value, reference, check)? {
            passed &= mixed["passed"] == json!(true);
            metrics.insert("mixed".into(), mixed);
        }
        metrics.insert("rows".into(), Value::Array(rows));
        metrics.insert("passed".into(), json!(passed));
        metrics.insert("limits".into(), receipt_limits(check, &audio_limit));
        Ok(metrics)
    }
}

/// Score validated forced-answer columns from the Python capture adapter.
pub(crate) fn score_capture(
    value: &Value,
    case: &Value,
    check: &Check,
) -> Result<Map<String, Value>, KindError> {
    if let Some(probe) = value.get("probe_greedy") {
        return crate::kinds::probes::score_greedy(probe, case);
    }
    if let Some(probe) = value.get("probe_forced") {
        return crate::kinds::probes::score_forced(probe, case);
    }
    if let Some(greedy) = value.get("greedy") {
        return score_greedy(case, greedy, check);
    }
    let gate = repr::str(&multimodal(check, "GATE_VARIANT", json!("exact")));
    let reference = field(field(case, "exact")?, &gate)?;
    let logprobs = ids(field(value, "logprob")?)?;
    let expected = ids(field(reference, "logprob")?)?;
    let served = ids(field(value, "top1")?)?;
    let top1 = ids(field(reference, "top1")?)?;
    let tops = top(reference)?;
    if logprobs.is_empty()
        || logprobs.len() != expected.len()
        || served.len() != expected.len()
        || top1.len() != expected.len()
        || tops.len() != expected.len()
    {
        return Err(invalid("forced answer columns differ in length"));
    }
    let differences = logprobs
        .iter()
        .zip(expected)
        .map(|(a, b)| Ok((number(a, "logprob")? - number(b, "logprob")?).abs()))
        .collect::<Result<Vec<_>, KindError>>()?;
    let flips = served
        .iter()
        .zip(top1)
        .zip(&tops)
        .map(|((token, reference), top)| {
            limits::flip(reference, token, Some(top), limits::NEAR_TIE_NATS)
        })
        .collect::<Vec<_>>();
    let mut result = serde_json::to_value(limits::top1_summary(&flips, logprobs.len() as u64))
        .map_err(|error| invalid(error.to_string()))?
        .as_object()
        .cloned()
        .expect("summary object");
    result.insert("scored".into(), json!(logprobs.len()));
    result.insert(
        "mean_abs_dlogprob".into(),
        json!(crate::exact::mean(&differences)),
    );
    result.insert(
        "max_abs_dlogprob".into(),
        json!(differences.iter().copied().reduce(f64::max)),
    );
    let recorded: Vec<Value> = flips.iter().enumerate().filter_map(|(offset, flip)| flip.map(|flip| {
        json!({"offset": offset, "served": served[offset], "gap": flip.gap, "decisive": flip.decisive})
    })).collect();
    result.insert("top1_flips".into(), json!(recorded));
    let rows = field(value, "served_rows")?;
    result.insert("served_rows".into(), rows.clone());
    result.insert(
        "served_rows_sha256".into(),
        json!(crate::json::canonical_sha256(rows)),
    );
    let (limits, power) = case_limits(case, &gate, check)?;
    let (passed, verdict) = set_verdict(
        &spread_of(&Value::Object(result.clone()))?,
        &limits,
        &flips,
        power.as_deref(),
    )?;
    result.insert("limits".into(), spread_json(&limits));
    result.extend(verdict);
    result.insert("passed".into(), json!(passed));
    Ok(result)
}

/// Judge a raw greedy answer before the Python adapter records its first divergence.
fn score_greedy(
    case: &Value,
    captured: &Value,
    check: &Check,
) -> Result<Map<String, Value>, KindError> {
    let gate = repr::str(&multimodal(check, "GATE_VARIANT", json!("exact")));
    let prompt = ids(field(case, "forced_prompt_token_ids")?)?;
    let start = count(field(case, "answer_token_start")?, "answer_token_start")? as usize;
    let tokens = prompt
        .get(start..)
        .ok_or_else(|| invalid("answer starts beyond the prompt"))?;
    let exact = field(captured, "exact")?
        .as_bool()
        .ok_or_else(|| invalid("exact must be a boolean"))?;
    let generated = if exact {
        Some(tokens)
    } else {
        captured["generated_ids"]
            .as_array()
            .filter(|tokens| {
                !tokens.is_empty() && tokens.iter().all(|token| token.is_i64() || token.is_u64())
            })
            .map(Vec::as_slice)
    };
    let Some(generated) = generated else {
        return Ok(
            json!({"passed": false, "exact": false, "reason": "missing generated token IDs"})
                .as_object()
                .cloned()
                .expect("object"),
        );
    };
    let stated_limit = field(captured, "near_tie_limit")?;
    let limit = number(stated_limit, "near_tie_limit")?;
    let tops = top(field(field(case, "exact")?, &gate)?)?;
    if tokens.len() != tops.len() {
        return Err(invalid("zip() arguments differ in length"));
    }
    let mut result =
        json!({"exact": exact, "near_tie_limit": stated_limit, "reference_margin": 0.0})
            .as_object()
            .cloned()
            .expect("object");
    let mut margin = 0.0_f64;
    for (offset, (reference, top)) in tokens.iter().zip(&tops).enumerate() {
        let Some(served) = generated.get(offset) else {
            result.extend(
                json!({"passed": false, "divergence": offset,
                "reason": "served answer ended inside the reference answer"})
                .as_object()
                .cloned()
                .expect("object"),
            );
            return Ok(result);
        };
        let ranked = top.iter().rev().find(|(token, _)| python_eq(token, served));
        let failure = if let Some((_, logprob)) = ranked {
            let gap = top
                .first()
                .ok_or_else(|| invalid("empty reference top rows"))?
                .1
                - logprob;
            margin = margin.max(gap);
            result.insert("reference_margin".into(), json!(margin));
            (!limits::near_tie(Some(gap), limit))
                .then_some("served token beyond the near-tie limit")
        } else {
            Some("served token outside the exact reference's top 5")
        };
        if failure.is_some() || !python_eq(served, reference) {
            result.extend(
                json!({"passed": failure.is_none(), "divergence": offset,
                "served_token_id": served, "reference_token_id": reference})
                .as_object()
                .cloned()
                .expect("object"),
            );
            if let Some(reason) = failure {
                result.insert("reason".into(), json!(reason));
            }
            return Ok(result);
        }
    }
    result.insert("passed".into(), json!(true));
    Ok(result)
}
