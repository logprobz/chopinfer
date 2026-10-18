//! `teacher_forced.v1`: teacher-forced prompt logprobs against a pinned
//! reference's gate variant, with limits from the reference's own variant
//! spread for the pooled positions and each text category
//! (docs/check-kinds.md#teacher_forced-teacher_forcedv1).

use std::collections::BTreeMap;

use serde_json::{Map, Value, json};

use crate::declaration::Check;
use crate::exact;
use crate::json::python_eq;
use crate::kind::{
    Capture, Kind, KindError, KindInfo, SHARED_KINDS, match_flips, natural, served_flips,
    served_top1,
};
use crate::kinds::{
    POWER_VARIANT, applied_constants, check_parts, check_reference_scores, check_summary, constant,
    count, field, flips_by_offset, get, ids, invalid, judged, number, require, set_verdict,
    spread_json, spread_of, top1_fields,
};
use crate::limits::{self, NEAR_TIE_NATS, Spread, SpreadRule};
use crate::references::Verified;
use crate::repr;

pub struct TeacherForced;

/// The pooled summary fields a receipt keeps as recorded.
const SCORES: [&str; 5] = [
    "scored",
    "mean_abs_dlogprob",
    "max_abs_dlogprob",
    "top1_disagreements",
    "top1_disagreement",
];

/// Recognize the declared teacher constants. Legacy comparison constants
/// remain accepted for declaration compatibility and do not set HF limits.
const CONSTANTS: [(&str, &str); 10] = [
    ("teacher", "ALLOW_FLOOR_ONLY"),
    ("teacher", "GATE_VARIANT"),
    ("teacher", "MAX_MEAN"),
    ("teacher", "LARGE_SET_MEAN"),
    ("teacher", "LARGE_SET_POSITIONS"),
    ("teacher", "MAX_TOP1"),
    ("teacher", "MEAN_STEP"),
    ("teacher", "SPREAD_MEAN_FACTOR"),
    ("teacher", "MEAN_MARGIN"),
    ("teacher", "TOP1_SIGMAS"),
];

/// The gate variant the check scores against (`teacher.GATE_VARIANT`).
fn gate_variant(check: &Check) -> String {
    repr::str(&constant(check, "teacher", "GATE_VARIANT", json!("exact")))
}

fn allow_floor_only(check: &Check) -> Result<bool, KindError> {
    constant(check, "teacher", "ALLOW_FLOOR_ONLY", json!(false))
        .as_bool()
        .ok_or_else(|| invalid("ALLOW_FLOOR_ONLY must be boolean"))
}

/// The opt-in's evidence after normal pin and producer authentication.
fn floor_only_reference(
    texts: &[Value],
    variants: &[Value],
    binding: &Value,
    check: &Check,
) -> Result<bool, KindError> {
    if !allow_floor_only(check)? {
        return Ok(false);
    }
    let mode = &binding["modes"]["f32"];
    let policy = crate::references::mode_policy_sha256("f32", &json!("exact"))
        .map_err(|error| invalid(error.to_string()))?;
    let digest = binding["workload_sha256"].as_str().is_some_and(|value| {
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    });
    require(
        gate_variant(check) == "exact"
            && variants == [json!("exact")]
            && binding["schema"] == crate::references::BINDING
            && binding["modes"]
                .as_object()
                .is_some_and(|modes| modes.len() == 1)
            && mode["mode"] == "f32"
            && mode["variant"] == "exact"
            && mode["dequant"] == "exact"
            && mode["policy_sha256"] == policy
            && digest,
        || "floor-only reference needs a bound F32 exact variant".into(),
    )?;
    let names: Vec<Value> = texts.iter().map(|text| text["name"].clone()).collect();
    check_reference(texts, &names, variants).map_err(KindError::Insufficient)?;
    let positions: usize = texts
        .iter()
        .map(|text| text["exact"]["logprob"].as_array().map_or(0, Vec::len))
        .sum();
    require(positions >= limits::LARGE_SET_POSITIONS as usize, || {
        "floor-only reference needs at least 1000 scored positions".into()
    })?;
    Ok(true)
}

/// The spread rule with the check's `teacher.py` floors.
fn rule(check: &Check) -> Result<SpreadRule, KindError> {
    let read =
        |name: &str, default: f64| number(&constant(check, "teacher", name, json!(default)), name);
    let positions = constant(
        check,
        "teacher",
        "LARGE_SET_POSITIONS",
        json!(limits::LARGE_SET_POSITIONS),
    );
    let defaults = SpreadRule::default();
    Ok(SpreadRule {
        mean_floor: read("MAX_MEAN", limits::MEAN_FLOOR)?,
        large_set_mean_floor: read("LARGE_SET_MEAN", limits::LARGE_SET_MEAN_FLOOR)?,
        // Python compares the position count with the constant as numbers.
        large_set_positions: number(&positions, "LARGE_SET_POSITIONS")?.ceil() as u64,
        top1_floor: read("MAX_TOP1", limits::TOP1_FLOOR)?,
        mean_step: read("MEAN_STEP", limits::MEAN_STEP)?,
        mean_factor: read("SPREAD_MEAN_FACTOR", limits::MEAN_FACTOR)?,
        ..defaults
    })
}

/// A scored position (`teacher.score`): its text, category and offset among
/// the text's scored positions, the scored top-1, |dlogprob| against the
/// variant scored against, and the flip.
struct Position {
    text: Value,
    category: String,
    offset: usize,
    served: Value,
    difference: f64,
    flip: Option<limits::Flip>,
}

/// One text's scores to judge: its logprobs and top-1 tokens per scored
/// position.
type Scores<'a> = (&'a [Value], &'a [Value]);

/// `teacher.score`: per-position |dlogprob| and top-1 flip of `captured`
/// (one entry per text) against the reference variant `variant`.
fn score(
    texts: &[Value],
    captured: &[Scores<'_>],
    variant: &str,
) -> Result<Vec<Position>, KindError> {
    if texts.len() != captured.len() {
        return Err(invalid("zip() arguments differ in length"));
    }
    let mut positions = Vec::new();
    for (text, (logprobs, top1)) in texts.iter().zip(captured) {
        let category = repr::str(field(text, "category")?);
        let row = field(text, variant)?;
        let tops =
            limits::gap_top(row).ok_or_else(|| invalid("the reference has no top-k rows"))?;
        let reference = ids(field(row, "logprob")?)?;
        let reference_top1 = ids(field(row, "top1")?)?;
        if logprobs.len() != reference.len() {
            return Err(invalid(format!(
                "{}: {} scores for {} reference positions",
                repr::str(field(text, "name")?),
                logprobs.len(),
                reference.len()
            )));
        }
        if top1.len() != logprobs.len() {
            return Err(invalid("zip() arguments differ in length"));
        }
        for (index, (logprob, served)) in logprobs.iter().zip(top1.iter()).enumerate() {
            let flip = limits::flip(
                reference_top1
                    .get(index)
                    .ok_or_else(|| invalid("list index out of range"))?,
                served,
                tops.get(index).map(Vec::as_slice),
                NEAR_TIE_NATS,
            );
            positions.push(Position {
                text: text["name"].clone(),
                category: category.clone(),
                offset: index,
                served: served.clone(),
                difference: (number(logprob, "logprob")? - number(&reference[index], "logprob")?)
                    .abs(),
                flip,
            });
        }
    }
    Ok(positions)
}

/// `teacher.variant_scores`: one reference variant's rows as a capture.
fn variant_scores<'a>(texts: &'a [Value], variant: &str) -> Result<Vec<Scores<'a>>, KindError> {
    texts
        .iter()
        .map(|text| {
            let scores = field(text, variant)?;
            Ok((
                ids(field(scores, "logprob")?)?,
                ids(field(scores, "top1")?)?,
            ))
        })
        .collect()
}

/// `teacher.summary` of scored positions.
fn summary(positions: &[&Position]) -> Map<String, Value> {
    let differences: Vec<f64> = positions
        .iter()
        .map(|position| position.difference)
        .collect();
    let largest = differences
        .iter()
        .copied()
        .reduce(|seen, value| if value > seen { value } else { seen });
    let flips: Vec<Option<limits::Flip>> = positions.iter().map(|position| position.flip).collect();
    let mut map = Map::new();
    map.insert("scored".into(), json!(positions.len()));
    map.insert("mean_abs_dlogprob".into(), json!(exact::mean(&differences)));
    map.insert("max_abs_dlogprob".into(), json!(largest));
    if let Ok(Value::Object(top1)) =
        serde_json::to_value(limits::top1_summary(&flips, positions.len() as u64))
    {
        map.extend(top1);
    }
    map
}

/// `teacher.by_category`: the summary of each category, sorted by name.
fn by_category(positions: &[Position]) -> Map<String, Value> {
    let mut groups: BTreeMap<&str, Vec<&Position>> = BTreeMap::new();
    for position in positions {
        groups.entry(&position.category).or_default().push(position);
    }
    groups
        .into_iter()
        .map(|(name, rows)| (name.to_owned(), Value::Object(summary(&rows))))
        .collect()
}

/// Each spread variant's positions scored against the gate variant.
type VariantPositions = Vec<(String, Vec<Position>)>;

/// `teacher.reference_limits`: each other variant scored against the gate
/// variant, and the limits that spread implies for the texts' positions.
fn reference_limits(
    texts: &[Value],
    gate: &str,
    variants: &[Value],
    rule: &SpreadRule,
) -> Result<(Spread, VariantPositions), KindError> {
    let mut spread = Vec::new();
    for variant in variants {
        let variant = repr::str(variant);
        if variant != gate {
            let positions = score(texts, &variant_scores(texts, &variant)?, gate)?;
            spread.push((variant, positions));
        }
    }
    let mut positions = 0u64;
    for text in texts {
        positions += ids(field(field(text, gate)?, "logprob")?)?.len() as u64;
    }
    let summaries: Vec<Spread> = spread
        .iter()
        .map(|(_, rows)| spread_of(&Value::Object(summary(&rows.iter().collect::<Vec<_>>()))))
        .collect::<Result<_, _>>()?;
    let limits =
        limits::spread_limits(&summaries, positions, rule).map_err(|error| invalid(error.0))?;
    Ok((limits, spread))
}

/// The flips of the reference run the power test reads (`POWER_VARIANT`)
/// at each position of a spread, if the reference pins that run.
fn power_reference(spread: &VariantPositions) -> Option<Vec<Option<limits::Flip>>> {
    spread
        .iter()
        .find(|(variant, _)| variant == POWER_VARIANT)
        .map(|(_, positions)| positions.iter().map(|position| position.flip).collect())
}

/// The scored token IDs of a reference text.
fn scored_tokens(text: &Value) -> Result<&[Value], KindError> {
    let all = ids(field(text, "prompt_token_ids")?)?;
    let from = count(field(text, "score_from")?, "score_from")? as usize;
    Ok(&all[from.min(all.len())..])
}

impl Kind for TeacherForced {
    fn info(&self) -> &'static KindInfo {
        &SHARED_KINDS[0]
    }

    fn reference_name(&self, check: &Check) -> Option<String> {
        let args = check.declared["args"].as_array()?;
        let named = args
            .iter()
            .position(|arg| arg == "--reference-name")
            .and_then(|index| args.get(index + 1))
            .and_then(Value::as_str);
        Some(named.unwrap_or("teacher").to_owned())
    }

    fn reference(
        &self,
        check: &Check,
        verified: Option<&Verified>,
    ) -> Result<Option<Value>, KindError> {
        let Some(verified) = verified else {
            require(!allow_floor_only(check)?, || {
                "floor-only check needs a pinned reference".into()
            })?;
            return Ok(None);
        };
        let variants = verified.variants.clone().unwrap_or_default();
        let mut reference = json!({"sha256": verified.sha256, "gate": gate_variant(check),
                                   "variants": variants, "texts": verified.value["texts"]});
        if allow_floor_only(check)? {
            require(
                matches!(verified.binding, crate::references::Binding::Bound { .. }),
                || {
                    "floor-only reference needs an authenticated producer and workload binding"
                        .into()
                },
            )?;
            floor_only_reference(
                ids(field(&reference, "texts")?)?,
                ids(field(&reference, "variants")?)?,
                &verified.value["binding"],
                check,
            )?;
            reference["binding"] = verified.value["binding"].clone();
        }
        Ok(Some(reference))
    }

    fn capture(&self, capture: &mut Capture<'_>) -> Result<Value, KindError> {
        capture_teacher(capture)
    }

    fn fields(&self, receipt: &Value, check: &Check) -> Vec<String> {
        let mut fields = crate::kind::default_fields(self.info(), receipt, check);
        for key in receipt.as_object().into_iter().flat_map(Map::keys) {
            if key.starts_with("vs_") && !fields.contains(key) {
                fields.push(key.clone());
            }
        }
        fields
    }

    fn verify(
        &self,
        value: &Value,
        reference: Option<&Value>,
        check: &Check,
    ) -> Result<Vec<String>, KindError> {
        applied_constants(check, &CONSTANTS)?;
        require(!allow_floor_only(check)? || reference.is_some(), || {
            "floor-only check needs a pinned reference".into()
        })?;
        let rows = get(value, "rows")
            .and_then(Value::as_array)
            .filter(|rows| !rows.is_empty() && rows.iter().all(Value::is_object));
        require(rows.is_some(), || "no per-text rows".into())?;
        let rows = rows.expect("checked");
        let variants_of_receipt: Vec<String> = value
            .as_object()
            .into_iter()
            .flat_map(Map::keys)
            .filter(|key| key.starts_with("vs_"))
            .cloned()
            .collect();
        let mut recorded: Vec<String> = SCORES.iter().map(|key| (*key).to_owned()).collect();
        recorded.push("by_category".into());
        recorded.extend(variants_of_receipt);
        let reference = reference.ok_or_else(|| {
            KindError::Insufficient(
                "no exact reference to verify the teacher receipt against".into(),
            )
        })?;
        require(
            get(value, "reference_sha256") == Some(field(reference, "sha256")?),
            || "the receipt scored another teacher reference".into(),
        )?;
        let texts = ids(field(reference, "texts")?)?;
        let gate = repr::str(field(reference, "gate")?);
        let variants = ids(field(reference, "variants")?)?;
        for text in texts {
            let valid = get(text, "prompt_token_ids")
                .and_then(Value::as_array)
                .zip(get(text, "score_from"))
                .is_some_and(|(all, start)| {
                    all.iter().all(natural)
                        && natural(start)
                        && start
                            .as_u64()
                            .is_some_and(|start| 1 <= start && (start as usize) < all.len())
                })
                && get(text, "category").is_some_and(Value::is_string);
            require(valid, || {
                format!(
                    "reference {}: invalid token IDs, score_from or category",
                    repr::str_of(get(text, "name"))
                )
            })?;
        }
        if !floor_only_reference(texts, variants, &reference["binding"], check)? {
            check_reference_scores(
                texts,
                variants,
                &gate,
                |text| scored_tokens(text).ok(),
                Some,
            )?;
        }
        let names: Vec<&Value> = rows
            .iter()
            .map(|row| get(row, "name").unwrap_or(&Value::Null))
            .collect();
        let in_order = names.len() == texts.len()
            && names
                .iter()
                .zip(texts)
                .all(|(name, text)| python_eq(name, &text["name"]));
        require(in_order, || {
            format!(
                "the rows cover {} texts, not the reference's {} in order",
                rows.len(),
                texts.len()
            )
        })?;
        let mut positions: Vec<(String, u64)> = Vec::new();
        for (row, text) in rows.iter().zip(texts) {
            let scored = ids(field(&text[gate.as_str()], "logprob")?)?.len() as u64;
            let prompt = text["prompt_token_ids"].as_array().map_or(0, Vec::len) as u64;
            require(
                get(row, "prompt_tokens").is_some_and(|value| python_eq(value, &json!(prompt)))
                    && get(row, "scored").is_some_and(|value| python_eq(value, &json!(scored))),
                || {
                    format!(
                        "text {}: the row scored {} of {scored} positions",
                        repr::str(&text["name"]),
                        repr::str_of(get(row, "scored"))
                    )
                },
            )?;
            let category = repr::str(&text["category"]);
            match positions.iter_mut().find(|(name, _)| *name == category) {
                Some((_, total)) => *total += scored,
                None => positions.push((category, scored)),
            }
        }
        for key in ["limits", "by_category", "failed_categories"] {
            require(get(value, key).is_some(), || format!("no {key}"))?;
        }
        let pooled: u64 = positions.iter().map(|(_, count)| count).sum();
        check_summary(Some(value), "pooled", Some(&json!(pooled)))?;
        let categories = value["by_category"].as_object();
        let mut expected: Vec<String> = positions.iter().map(|(name, _)| name.clone()).collect();
        expected.sort();
        let stated = categories.map(|categories| {
            let mut names: Vec<String> = categories.keys().cloned().collect();
            names.sort();
            names
        });
        require(stated.as_ref() == Some(&expected), || {
            format!(
                "categories {} are not the reference's {}",
                stated
                    .as_ref()
                    .map_or_else(|| "None".to_owned(), |names| repr::repr(&json!(names))),
                repr::repr(&json!(expected))
            )
        })?;
        let categories = categories.expect("checked");
        for (name, count) in &positions {
            check_summary(
                categories.get(name),
                &format!("category {name}"),
                Some(&json!(count)),
            )?;
        }
        let parts: Vec<&Value> = categories.values().collect();
        check_parts(value, &parts, "categories")?;
        for variant in variants {
            let variant = repr::str(variant);
            if variant != gate {
                check_summary(
                    get(value, &format!("vs_{variant}")),
                    &format!("vs_{variant}"),
                    Some(&value["scored"]),
                )?;
            }
        }
        let flips = get(value, "top1_flips")
            .and_then(Value::as_array)
            .filter(|flips| {
                flips.iter().all(|flip| {
                    flip.is_object()
                        && texts.iter().any(|text| {
                            get(flip, "text").is_some_and(|name| python_eq(name, &text["name"]))
                        })
                        && {
                            // by_name keeps the last text of a name, as a dict does.
                            let text = texts
                                .iter()
                                .rev()
                                .find(|text| python_eq(&flip["text"], &text["name"]))
                                .expect("found above");
                            get(flip, "category")
                                .is_some_and(|category| python_eq(category, &text["category"]))
                        }
                })
            });
        require(flips.is_some(), || {
            "no top-1 flips of the reference's texts (top1_flips; receipts before the near-tie rule have none)"
                .into()
        })?;
        let flips = flips.expect("checked");
        for (row, text) in rows.iter().zip(texts) {
            let where_ = format!("text {}", repr::str(&text["name"]));
            let top1 = ids(field(&text[gate.as_str()], "top1")?)?;
            let derived = served_flips(Some(row), scored_tokens(text)?, top1, &where_)?;
            let of_text: Vec<Value> = flips
                .iter()
                .filter(|flip| python_eq(&flip["text"], &text["name"]))
                .cloned()
                .collect();
            match_flips(Some(&Value::Array(of_text)), &derived, &where_)?;
        }
        let adds_up = python_eq(&json!(flips.len()), &value["top1_disagreements"])
            && categories.iter().all(|(name, row)| {
                let of_category = flips
                    .iter()
                    .filter(|flip| flip["category"] == json!(name))
                    .count();
                get(row, "top1_disagreements")
                    .is_some_and(|stated| python_eq(&json!(of_category), stated))
            });
        require(adds_up, || {
            "the top-1 flips do not add up to the recorded disagreements".into()
        })?;
        Ok(recorded)
    }

    fn score(
        &self,
        value: &Value,
        reference: Option<&Value>,
        check: &Check,
        mechanics: bool,
    ) -> Result<Map<String, Value>, KindError> {
        require(!allow_floor_only(check)? || reference.is_some(), || {
            "floor-only check needs a pinned reference".into()
        })?;
        let mut metrics = crate::kind::default_score(self, value, check, mechanics)?;
        let Some(reference) = reference.filter(|_| get(value, "limits").is_some()) else {
            return Ok(metrics);
        };
        require(
            get(value, "reference_sha256") == Some(field(reference, "sha256")?),
            || "the receipt scored another teacher reference".into(),
        )?;
        let rule = rule(check)?;
        let texts = ids(field(reference, "texts")?)?;
        let gate = repr::str(field(reference, "gate")?);
        let variants = ids(field(reference, "variants")?)?;
        floor_only_reference(texts, variants, &reference["binding"], check)?;
        // The flips the served rows show, judged from the reference; the
        // near-tie, decisive and total counts and rates follow from them.
        let rows: Vec<&Value> = get(value, "rows")
            .and_then(Value::as_array)
            .map(|rows| rows.iter().filter(|row| row.is_object()).collect())
            .unwrap_or_default();
        let mut flips = Vec::new();
        for text in texts {
            // A dict of rows by name keeps the last row of a name.
            let row = rows
                .iter()
                .rev()
                .find(|row| {
                    get(row, "name").is_some_and(|name| {
                        python_eq(name, field(text, "name").unwrap_or(&Value::Null))
                    })
                })
                .copied();
            let where_ = format!("text {}", repr::str(field(text, "name")?));
            let top1 = ids(field(field(text, &gate)?, "top1")?)?;
            let derived = served_flips(row, scored_tokens(text)?, top1, &where_)?;
            let mut fields = Map::new();
            fields.insert("text".into(), text["name"].clone());
            fields.insert("category".into(), text["category"].clone());
            flips.extend(judged(
                derived.iter().map(|(offset, served)| (*offset, served)),
                &text[gate.as_str()],
                &fields,
            )?);
        }
        let scored = count(field(value, "scored")?, "scored")?;
        let pooled = top1_fields(&flips, scored);
        let (limits, spread) = reference_limits(texts, &gate, variants, &rule)?;
        let mut categories = Map::new();
        let recorded_categories = field(value, "by_category")?
            .as_object()
            .ok_or_else(|| invalid("by_category is not an object"))?;
        for (name, row) in recorded_categories {
            let row_map = row
                .as_object()
                .ok_or_else(|| invalid("a category is not an object"))?;
            // The verdict's own fields are recomputed, never kept.
            let mut scores: Map<String, Value> = row_map
                .iter()
                .filter(|(key, _)| {
                    ![
                        "limits",
                        "passed",
                        "spread_rule",
                        "passed_by",
                        "power",
                        "paired_reference",
                    ]
                    .contains(&key.as_str())
                })
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect();
            let of_category: Vec<Value> = flips
                .iter()
                .filter(|flip| flip["category"] == json!(name))
                .cloned()
                .collect();
            scores.extend(top1_fields(
                &of_category,
                count(field(row, "scored")?, "scored")?,
            ));
            let category_texts: Vec<Value> = texts
                .iter()
                .filter(|text| text["category"] == json!(name))
                .cloned()
                .collect();
            let (category_limits, category_spread) =
                reference_limits(&category_texts, &gate, variants, &rule)?;
            let mut candidate = Vec::new();
            for text in &category_texts {
                let scored = ids(field(field(text, &gate)?, "logprob")?)?.len();
                let of_text: Vec<Value> = of_category
                    .iter()
                    .filter(|flip| python_eq(&flip["text"], &text["name"]))
                    .cloned()
                    .collect();
                candidate.extend(flips_by_offset(&of_text, scored)?);
            }
            let (passed, verdict) = set_verdict(
                &spread_of(&Value::Object(scores.clone()))?,
                &category_limits,
                &candidate,
                power_reference(&category_spread).as_deref(),
            )?;
            scores.insert("limits".into(), spread_json(&category_limits));
            scores.extend(verdict);
            scores.insert("passed".into(), json!(passed));
            categories.insert(name.clone(), Value::Object(scores));
        }
        let failed: Vec<Value> = categories
            .iter()
            .filter(|(_, row)| row["passed"] != json!(true))
            .map(|(name, _)| json!(name))
            .collect();
        let mut measured = value.as_object().cloned().unwrap_or_default();
        measured.extend(pooled.clone());
        let passed =
            limits::within(&spread_of(&Value::Object(measured))?, &limits) && failed.is_empty();
        metrics.extend(pooled);
        metrics.insert("top1_flips".into(), Value::Array(flips));
        metrics.insert("near_tie_nats".into(), json!(NEAR_TIE_NATS));
        metrics.insert("limits".into(), spread_json(&limits));
        metrics.insert("by_category".into(), Value::Object(categories));
        metrics.insert("failed_categories".into(), Value::Array(failed));
        let reference_spread: Map<String, Value> = spread
            .iter()
            .map(|(variant, positions)| {
                let all: Vec<&Position> = positions.iter().collect();
                (
                    variant.clone(),
                    json!({"all": summary(&all), "by_category": by_category(positions)}),
                )
            })
            .collect();
        metrics.insert("reference_spread".into(), Value::Object(reference_spread));
        // An informational or rescored capture keeps its verdict as a diagnostic.
        let key = if get(value, "diagnostic_passed").is_some() {
            "diagnostic_passed"
        } else {
            "passed"
        };
        metrics.insert(key.into(), json!(passed));
        Ok(metrics)
    }
}

/// The arguments of `teacher.py` a live eval capture does not take: partial
/// text sets, offline rescores of a capture and the prompt-logprobs (vLLM
/// context) mode stay in tools/qual.
const DEVELOPER_OPTIONS: [&str; 6] = [
    "--texts",
    "--from-capture",
    "--reference",
    "--reference-kind",
    "--exact",
    "--long",
];

/// The token IDs a mechanics run sends instead of the first text's.
const MECHANICS_PROMPT: [u64; 4] = [785, 2310, 6733, 13];

/// A served prompt-logprob row ({token: {logprob, rank}}) as receipts keep
/// it (`common.served_entries`): [token, logprob, rank] entries, rank null
/// when the server gave none, in top-1 order.
fn served_entries(row: &Map<String, Value>) -> Result<Vec<Value>, String> {
    let mut entries = Vec::new();
    for (token, value) in row {
        let token: u64 = token.trim().parse().map_err(|_| {
            format!(
                "invalid literal for int() with base 10: {}",
                repr::repr(&json!(token))
            )
        })?;
        let logprob = value
            .get("logprob")
            .cloned()
            .ok_or_else(|| "'logprob'".to_owned())?;
        let rank = value.get("rank").cloned().unwrap_or(Value::Null);
        entries.push(json!([token, logprob, rank]));
    }
    let rank = |entry: &Value| {
        if entry[2].is_null() {
            json!(1_000_000_000)
        } else {
            entry[2].clone()
        }
    };
    let order = |a: &Value, b: &Value| {
        crate::json::python_cmp(&rank(a), &rank(b))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                crate::json::python_cmp(&b[1], &a[1]).unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| {
                crate::json::python_cmp(&a[0], &b[0]).unwrap_or(std::cmp::Ordering::Equal)
            })
    };
    entries.sort_by(order);
    Ok(entries)
}

/// One text's capture (`teacher.capture_text`): per scored position the
/// served logprob of its token, the served top-1, the margin of the served
/// top two and the served row.
struct Captured {
    logprob: Vec<Value>,
    top1: Vec<Value>,
    margin: Vec<Value>,
    served_rows: Vec<Value>,
}

/// `teacher.top_margin`: the served top-1 logprob less the runner-up's, in
/// served order; null for a row of one entry.
fn top_margin(entries: &[Value]) -> Value {
    match entries {
        [first, second, ..] => match (&first[1], &second[1]) {
            (Value::Number(a), Value::Number(b)) if a.is_i64() && b.is_i64() => {
                let (a, b) = (a.as_i64().unwrap_or(0), b.as_i64().unwrap_or(0));
                a.checked_sub(b).map_or(Value::Null, Value::from)
            }
            (a, b) => match (crate::json::python_number(a), crate::json::python_number(b)) {
                (Some(a), Some(b)) => json!(a - b),
                _ => Value::Null,
            },
        },
        _ => Value::Null,
    }
}

fn capture_text(text: &Value, ids: &[Value], rows: &[Value]) -> Result<Captured, String> {
    let from = text["score_from"].as_u64().unwrap_or(0) as usize;
    let mut captured = Captured {
        logprob: Vec::new(),
        top1: Vec::new(),
        margin: Vec::new(),
        served_rows: Vec::new(),
    };
    for position in from..ids.len() {
        let key = repr::str(&ids[position]);
        let row = rows[position]
            .as_object()
            .filter(|row| row.contains_key(&key))
            .ok_or_else(|| format!("missing actual token logprob at position {position}"))?;
        captured.logprob.push(
            row[&key]
                .get("logprob")
                .cloned()
                .ok_or_else(|| "'logprob'".to_owned())?,
        );
        let entries = served_entries(row)?;
        captured.top1.push(served_top1(&entries).clone());
        captured.margin.push(top_margin(&entries));
        captured.served_rows.push(Value::Array(entries));
    }
    Ok(captured)
}

/// The prompt token IDs and prompt-logprob rows of a response
/// (`teacher.response_prompt_rows`).
fn response_prompt_rows(observed: &Value) -> Result<(Vec<Value>, Vec<Value>), String> {
    let choice = &observed["choices"][0];
    let pick = |key: &str| {
        Some(&choice[key])
            .filter(|value| crate::json::truthy(value))
            .or_else(|| observed.get(key))
            .and_then(Value::as_array)
            .cloned()
    };
    match (pick("prompt_token_ids"), pick("prompt_logprobs")) {
        (Some(ids), Some(rows)) if ids.len() == rows.len() => Ok((ids, rows)),
        _ => Err("missing prompt token IDs or logprob rows".into()),
    }
}

/// `teacher.check_reference`: the reference scores exactly the frozen texts
/// `names`, in order, each with every pinned variant complete over its
/// scored positions.
fn check_reference(texts: &[Value], names: &[Value], variants: &[Value]) -> Result<(), String> {
    let stated: Vec<&Value> = texts
        .iter()
        .map(|text| text.get("name").unwrap_or(&Value::Null))
        .collect();
    let distinct = names
        .iter()
        .enumerate()
        .all(|(index, name)| names[..index].iter().all(|other| !python_eq(other, name)));
    if stated.len() != names.len()
        || !stated.iter().zip(names).all(|(a, b)| python_eq(a, b))
        || !distinct
    {
        return Err("the reference does not score the frozen teacher texts in order".into());
    }
    for text in texts {
        let name = repr::str(&text["name"]);
        let all = text.get("prompt_token_ids").and_then(Value::as_array);
        let start = text
            .get("score_from")
            .filter(|start| start.is_i64() || start.is_u64());
        let (Some(all), Some(start)) = (all, start.and_then(Value::as_i64)) else {
            return Err(format!("{name}: invalid token IDs or score_from"));
        };
        if !(1 <= start && (start as usize) < all.len()) {
            return Err(format!("{name}: invalid token IDs or score_from"));
        }
        if !text.get("category").is_some_and(Value::is_string) {
            return Err(format!("{name}: no category"));
        }
        for variant in variants {
            let variant = repr::str(variant);
            crate::kinds::check_scores(
                text.get(&variant),
                Some(&all[start as usize..]),
                &format!("{name} {variant}"),
            )?;
        }
    }
    Ok(())
}

/// `teacher.evaluate`: the capture scored against the reference into
/// `result`; whether it passes. The pooled scores must meet the limits of
/// the whole reference's spread, and each category those of its own.
fn evaluate(
    captured: &[Captured],
    texts: &[Value],
    variants: &[Value],
    gate: &str,
    rule: &SpreadRule,
    result: &mut Map<String, Value>,
) -> Result<bool, KindError> {
    let rows: Vec<Scores<'_>> = captured
        .iter()
        .map(|text| (text.logprob.as_slice(), text.top1.as_slice()))
        .collect();
    let mut gate_positions = Vec::new();
    for variant in variants {
        let variant = repr::str(variant);
        let positions = score(texts, &rows, &variant)?;
        if variant == gate {
            gate_positions = positions;
        } else {
            let all: Vec<&Position> = positions.iter().collect();
            result.insert(format!("vs_{variant}"), Value::Object(summary(&all)));
        }
    }
    let all: Vec<&Position> = gate_positions.iter().collect();
    let pooled = summary(&all);
    result.extend(pooled.clone());
    let (limits, spread) = reference_limits(texts, gate, variants, rule)?;
    result.insert("limits".into(), spread_json(&limits));
    result.insert("near_tie_nats".into(), json!(NEAR_TIE_NATS));
    let flips: Vec<Value> = gate_positions
        .iter()
        .filter_map(|position| {
            position.flip.map(|flip| {
                json!({"text": position.text, "category": position.category, "offset": position.offset,
                       "served": position.served, "gap": flip.gap, "decisive": flip.decisive})
            })
        })
        .collect();
    result.insert("top1_flips".into(), Value::Array(flips));
    let reference_spread: Map<String, Value> = spread
        .iter()
        .map(|(variant, positions)| {
            let all: Vec<&Position> = positions.iter().collect();
            (
                variant.clone(),
                json!({"all": summary(&all), "by_category": by_category(positions)}),
            )
        })
        .collect();
    result.insert("reference_spread".into(), Value::Object(reference_spread));
    let mut categories = Map::new();
    for (category, row) in by_category(&gate_positions) {
        let of_category: Vec<Value> = texts
            .iter()
            .filter(|text| text["category"] == json!(category))
            .cloned()
            .collect();
        let (category_limits, category_spread) =
            reference_limits(&of_category, gate, variants, rule)?;
        let candidate: Vec<Option<limits::Flip>> = gate_positions
            .iter()
            .filter(|position| position.category == category)
            .map(|position| position.flip)
            .collect();
        let (passed, verdict) = set_verdict(
            &spread_of(&row)?,
            &category_limits,
            &candidate,
            power_reference(&category_spread).as_deref(),
        )?;
        let mut row = row.as_object().cloned().unwrap_or_default();
        row.insert("limits".into(), spread_json(&category_limits));
        row.extend(verdict);
        row.insert("passed".into(), json!(passed));
        categories.insert(category, Value::Object(row));
    }
    let failed: Vec<Value> = categories
        .iter()
        .filter(|(_, row)| row["passed"] != json!(true))
        .map(|(name, _)| json!(name))
        .collect();
    let passed = limits::within(&spread_of(&Value::Object(pooled))?, &limits) && failed.is_empty();
    result.insert("by_category".into(), Value::Object(categories));
    result.insert("failed_categories".into(), Value::Array(failed));
    Ok(passed)
}

/// Score Python's captured columns with the same evaluator as live Rust capture.
pub(crate) fn score_capture(
    value: &Value,
    reference: &Value,
    check: &Check,
) -> Result<Map<String, Value>, KindError> {
    let captured = ids(value)?
        .iter()
        .map(|row| {
            Ok(Captured {
                logprob: ids(field(row, "logprob")?)?.to_vec(),
                top1: ids(field(row, "top1")?)?.to_vec(),
                margin: Vec::new(),
                served_rows: Vec::new(),
            })
        })
        .collect::<Result<Vec<_>, KindError>>()?;
    let texts = ids(field(reference, "texts")?)?;
    let variants = ids(field(reference, "variants")?)?;
    if !floor_only_reference(texts, variants, &reference["binding"], check)? {
        require(
            variants
                .iter()
                .any(|variant| variant == &gate_variant(check))
                && variants.len() >= 2,
            || "the pinned teacher variants need the gate and a spread variant".into(),
        )?;
    }
    let mut result = Map::new();
    let passed = evaluate(
        &captured,
        texts,
        variants,
        &gate_variant(check),
        &rule(check)?,
        &mut result,
    )?;
    result.insert("passed".into(), json!(passed));
    Ok(result)
}

/// `teacher.exact_main` against a live server: every text of the pinned
/// reference teacher-forced with prompt logprobs, scored against its gate
/// variant within the spread limits. This receipt carries no informational
/// comparison against a serving-system baseline.
fn capture_teacher(capture: &mut Capture<'_>) -> Result<Value, KindError> {
    let check = capture.check;
    let qualification = &capture.declaration.declaration.qualification;
    for option in DEVELOPER_OPTIONS.iter().chain(&["--no-long"]) {
        if crate::capture::flag(check, option) {
            return Err(KindError::Unsupported(format!(
                "teacher {option} runs with tools/qual, not chopinfer eval"
            )));
        }
    }
    if qualification["teacher_reference_kind"] != json!("exact") {
        return Err(KindError::Unsupported(
            "teacher capture requires an exact HF reference".into(),
        ));
    }
    let name =
        crate::capture::option(check, "--reference-name").unwrap_or_else(|| "teacher".into());
    let receipt_id = crate::capture::option(check, "--receipt")
        .or_else(|| qualification["teacher_receipt"].as_str().map(String::from))
        .ok_or_else(|| invalid("the declaration names no teacher receipt"))?;
    let mechanics = capture.mechanics;
    let verified = capture.reference.ok_or_else(|| {
        KindError::Insufficient(format!("refused: the gate pins no {name} reference"))
    })?;
    let texts = ids(field(&verified.value, "texts")?)?.to_vec();
    let variants: Vec<Value> = verified
        .variants
        .clone()
        .unwrap_or_default()
        .into_iter()
        .map(Value::from)
        .collect();
    let gate = gate_variant(check);
    let rule = rule(check)?;
    if !mechanics {
        let floor = floor_only_reference(&texts, &variants, &verified.value["binding"], check)?;
        if floor {
            require(
                matches!(verified.binding, crate::references::Binding::Bound { .. }),
                || {
                    "floor-only reference needs an authenticated producer and workload binding"
                        .into()
                },
            )?;
        }
        if !floor
            && (gate != "exact"
                || !variants.iter().any(|variant| variant == "exact")
                || variants.len() < 2)
        {
            return Err(invalid(format!(
                "the pinned teacher variants {} need {gate} and a spread variant",
                repr::repr(&Value::Array(variants.clone()))
            )));
        }
        let frozen: Vec<Value> = if name == "teacher" {
            let fixture_root =
                crate::references::input_path(capture.declaration, "fixture_root", &|name| {
                    capture.declaration.environment(name)
                })
                .ok_or_else(|| {
                    KindError::Insufficient("the declaration has no fixture_root".into())
                })?;
            let path = fixture_root.join("teacher-texts.json");
            let bytes = crate::lock::read(capture.declaration, "fixture_root", &path)
                .map_err(KindError::Insufficient)?;
            let value = crate::json::parse(&bytes)
                .map_err(|error| KindError::Insufficient(format!("{}: {error}", path.display())))?;
            ids(field(&value, "texts")?)?
                .iter()
                .map(|text| text.get("name").cloned().unwrap_or(Value::Null))
                .collect()
        } else {
            let pinned = capture
                .declaration
                .declaration
                .reference(&name)
                .is_some_and(|pin| pin.workload_sha256.is_some());
            if !pinned {
                return Err(KindError::Insufficient(format!(
                    "refused teacher reference {}: the {name} reference pins no workload",
                    verified.path.display()
                )));
            }
            texts.iter().map(|text| text["name"].clone()).collect()
        };
        check_reference(&texts, &frozen, &variants).map_err(|error| {
            KindError::Insufficient(format!(
                "refused teacher reference {}: {error}",
                verified.path.display()
            ))
        })?;
    }
    let model = capture.server.model.clone();
    let sent: &[Value] = if mechanics {
        &texts[..texts.len().min(1)]
    } else {
        &texts
    };
    let (mut captured, mut rows, mut errors) = (Vec::new(), Vec::new(), Vec::new());
    for text in sent {
        crate::capture::remaining(capture)?;
        let prompt: Vec<Value> = if mechanics {
            MECHANICS_PROMPT.iter().map(|id| json!(id)).collect()
        } else {
            ids(field(text, "prompt_token_ids")?)?.to_vec()
        };
        let body = json!({"model": model, "prompt": prompt, "echo": true, "logprobs": 5,
                          "prompt_logprobs": 5, "return_token_ids": true, "max_tokens": 1,
                          "temperature": 0});
        let outcome = match capture.client.json("/v1/completions", Some(&body)) {
            Err(crate::http::HttpError::Transport(message)) => {
                return Err(KindError::Environment(message));
            }
            Err(error) => Err(error.to_string()),
            Ok((observed, latency)) => (|| {
                let (actual_ids, actual_rows) = response_prompt_rows(&observed)?;
                if !observed["choices"][0]["logprobs"]["token_logprobs"][0].is_null() {
                    return Err("echo first token logprob must be null".to_owned());
                }
                if !python_eq(
                    &Value::Array(actual_ids.clone()),
                    &Value::Array(prompt.clone()),
                ) {
                    return Err("returned prompt token IDs differ from the request".into());
                }
                let mut row = Map::new();
                row.insert("name".into(), text["name"].clone());
                row.insert("prompt_tokens".into(), json!(prompt.len()));
                if mechanics {
                    row.insert("scored".into(), json!(0));
                    row.insert("latency_seconds".into(), json!(latency));
                    return Ok((row, None));
                }
                let text_capture = capture_text(text, &actual_ids, &actual_rows)?;
                row.insert("scored".into(), json!(text_capture.logprob.len()));
                row.insert("latency_seconds".into(), json!(latency));
                let served = Value::Array(text_capture.served_rows.clone());
                row.insert(
                    "served_rows_sha256".into(),
                    json!(crate::kind::rows_digest(&served)),
                );
                row.insert("served_rows".into(), served);
                // The inputs a paired comparison checks (`teacher.py`).
                row.insert("prompt_token_ids_sha256".into(), json!(ids_digest(&prompt)));
                row.insert("score_from".into(), text["score_from"].clone());
                row.insert("category".into(), text["category"].clone());
                Ok((row, Some(text_capture)))
            })(),
        };
        match outcome {
            Ok((row, text_capture)) => {
                captured.extend(text_capture);
                rows.push(Value::Object(row));
            }
            Err(error) => errors.push(json!({"name": text["name"], "error": error})),
        }
    }
    let mut result = Map::new();
    if capture.capture_out.is_some() && errors.is_empty() && !mechanics {
        let bytes = compact_capture(capture, verified.sha256.as_str(), sent, &captured)?;
        // The capture this receipt scored, whole: a later comparison of
        // arms reads its per-position scores only under this digest.
        result.insert(
            "capture_sha256".into(),
            json!(crate::json::sha256_hex(&bytes)),
        );
        if let Some(out) = &mut capture.capture_out {
            out.bytes = Some(bytes);
        }
    }
    result.insert("errors".into(), Value::Array(errors.clone()));
    result.insert("rows".into(), Value::Array(rows));
    let mut skipped = Vec::new();
    result.insert("reference_sha256".into(), json!(verified.sha256));
    result.insert(
        "reference_binding".into(),
        if mechanics {
            Value::Null
        } else {
            verified.binding.to_json()
        },
    );
    let mut passed = errors.is_empty();
    if mechanics {
        skipped.push(json!("numerical quality"));
    } else if errors.is_empty() {
        passed = evaluate(&captured, &texts, &variants, &gate, &rule, &mut result)?;
    }
    result.insert("skipped".into(), Value::Array(skipped));
    // The first teacher check's informational reason, as teacher.py reads it.
    let policy = capture
        .declaration
        .declaration
        .checks
        .iter()
        .find(|declared| declared.module == "teacher");
    if let Some(reason) = policy.and_then(|policy| policy.informational_reason.clone()) {
        result.insert("informational_reason".into(), json!(reason));
        result.entry("diagnostic_passed").or_insert(json!(passed));
        passed = false;
    }
    let mut value = Map::new();
    value.insert("passed".into(), json!(passed));
    value.extend(result);
    Ok(crate::capture::receipt(&receipt_id, capture, value))
}

/// The compact capture of `texts` (`teacher.py --capture-out`), as
/// teacher.py writes it: its schema, the bound server, the reference digest
/// and per text the scored columns.
fn compact_capture(
    capture: &Capture<'_>,
    reference_sha256: &str,
    texts: &[Value],
    captured: &[Captured],
) -> Result<Vec<u8>, KindError> {
    let schema = capture.declaration.declaration.qualification["teacher_capture_schema"].clone();
    require(schema.is_string(), || {
        "the declaration names no qualification.teacher_capture_schema".into()
    })?;
    let mut server = capture.server.to_json();
    server["checkpoint"] = capture.bound.to_json();
    let rows: Vec<Value> = texts
        .iter()
        .zip(captured)
        .map(|(text, columns)| {
            let prompt = ids(field(text, "prompt_token_ids")?)?;
            let mut row = Map::new();
            row.insert("name".into(), text["name"].clone());
            row.insert("category".into(), text["category"].clone());
            row.insert("prompt_token_ids_sha256".into(), json!(ids_digest(prompt)));
            row.insert("score_from".into(), text["score_from"].clone());
            row.insert("logprob".into(), Value::Array(columns.logprob.clone()));
            row.insert("top1".into(), Value::Array(columns.top1.clone()));
            row.insert("margin".into(), Value::Array(columns.margin.clone()));
            row.insert(
                "served_rows".into(),
                Value::Array(columns.served_rows.clone()),
            );
            Ok(Value::Object(row))
        })
        .collect::<Result<_, KindError>>()?;
    let mut value = Map::new();
    value.insert("schema".into(), schema);
    value.insert("server".into(), server);
    value.insert("reference_sha256".into(), json!(reference_sha256));
    value.insert("texts".into(), Value::Array(rows));
    let mut bytes = crate::json::dumps(&Value::Object(value));
    bytes.push(b'\n');
    Ok(bytes)
}

/// `teacher.ids_digest`: the SHA-256 of the token IDs as Python's
/// `json.dumps` writes a list of ints.
fn ids_digest(ids: &[Value]) -> String {
    let text: Vec<String> = ids.iter().map(Value::to_string).collect();
    crate::json::sha256_hex(format!("[{}]", text.join(", ")).as_bytes())
}
