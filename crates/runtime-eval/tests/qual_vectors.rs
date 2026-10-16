//! The parts of the check-kind conformance vectors (tests/qual-vectors,
//! docs/check-kinds.md) that this crate implements before the kinds
//! themselves: reading a pinned reference, admission, the declared
//! predicates, the spread rule's limits and verdicts, and the near-tie rule's
//! flips recomputed from the reference. The kinds' own verify and score are
//! checked against the same vectors once they are registered.

use std::path::{Path, PathBuf};

use runtime_eval::declaration::{CheckpointPin, Input, Predicate, ReferencePin};
use runtime_eval::kind::{self, KindError};
use runtime_eval::limits::{self, Flip, NEAR_TIE_NATS, Spread, SpreadRule};
use runtime_eval::references::{self, ReferenceError};
use serde_json::{Value, json};

fn vectors() -> Vec<(String, Value)> {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/qual-vectors");
    let mut files: Vec<PathBuf> = std::fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "json")
        })
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no conformance vectors");
    files
        .into_iter()
        .map(|path| {
            let value = runtime_eval::json::parse(&std::fs::read(&path).unwrap()).unwrap();
            assert_eq!(
                value["schema"],
                "chopinfer-qual-vectors/v1",
                "{}",
                path.display()
            );
            (value["kind"].as_str().unwrap().to_owned(), value)
        })
        .collect()
}

/// The vectors of every kind file (not the reference-reading ones).
fn kind_vectors() -> Vec<(String, Value)> {
    vectors()
        .into_iter()
        .filter(|(kind, _)| kind != "references")
        .flat_map(|(kind, file)| {
            file["vectors"]
                .as_array()
                .unwrap()
                .iter()
                .map(|vector| (kind.clone(), vector.clone()))
                .collect::<Vec<_>>()
        })
        .collect()
}

fn text(value: &Value) -> String {
    value.as_str().unwrap_or_default().to_owned()
}

#[test]
fn reading_pinned_references_reproduces() {
    let (_, file) = vectors()
        .into_iter()
        .find(|(kind, _)| kind == "references")
        .expect("reference-reading vectors");
    let scratch =
        std::env::temp_dir().join(format!("runtime-eval-qual-vectors-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    let path = scratch.join("reference.json");
    for vector in file["vectors"].as_array().unwrap() {
        let name = &vector["name"];
        let stated = &vector["checkpoint"];
        let checkpoint = CheckpointPin {
            lock: text(&stated["lock"]),
            repository: text(&stated["repository"]),
            revision: text(&stated["revision"]),
            served_model_name: text(&stated["served_model_name"]),
            source_lock_sha256: text(&stated["source_lock_sha256"]),
            checkpoint_identity: text(&stated["checkpoint_identity"]),
            registry: stated["pin"]
                .as_str()
                .map(|pin| (pin.to_owned(), text(&stated["registry"]))),
        };
        let spec = &vector["pin"];
        let strings = |value: &Value| {
            value
                .as_array()
                .map(|items| items.iter().map(text).collect::<Vec<_>>())
        };
        let pin = ReferencePin {
            name: "teacher".into(),
            path: Input::Path("reference.json".into()),
            sha256: text(&spec["sha256"]),
            schema: text(&spec["schema"]),
            variants: strings(&spec["variants"]),
            workload_sha256: spec["workload_sha256"].as_str().map(String::from),
            capture_sha256: spec["capture_sha256"].as_str().map(String::from),
            checkpoint_keys: strings(&spec["checkpoint_keys"]),
            workload: spec["workload"]
                .as_str()
                .map(|path| Input::Path(path.into())),
        };
        let bytes = vector["fixture"].as_str().unwrap().as_bytes().to_vec();
        let result =
            references::verify_pinned(&scratch, &checkpoint, &pin, &path, bytes, &|| Ok(()), None);
        let expected = &vector["expected"];
        match result {
            Ok(verified) => {
                assert_eq!(expected["status"], "accepted", "{name}");
                let mut binding = verified.binding.to_json();
                binding.as_object_mut().unwrap().remove("unchecked");
                assert_eq!(binding, expected["binding"], "{name}");
            }
            Err(error) => {
                assert_eq!(expected["status"], "refused", "{name}: {error}");
                let message = error
                    .to_string()
                    .replace(&path.display().to_string(), "<fixture>");
                assert!(
                    message.contains(expected["error"].as_str().unwrap()),
                    "{name}: {message}"
                );
                assert!(!matches!(error, ReferenceError::Missing(_)), "{name}");
            }
        }
    }
    std::fs::remove_dir_all(scratch).unwrap();
}

#[test]
fn admission_matches_the_vectors() {
    let admission = [
        "expected a JSON object receipt",
        "quality check reported mechanics-only or skipped coverage",
        "check reported errors",
    ];
    let mut admitted = 0;
    for (kind, vector) in kind_vectors() {
        let name = &vector["name"];
        let mechanics = vector["mechanics"].as_bool().unwrap_or(false);
        let expected = &vector["expected"];
        let error = expected["error"].as_str().unwrap_or_default();
        match kind::admit(&vector["receipt"], mechanics) {
            Err(refused) => {
                assert_eq!(expected["status"], "fail", "{kind} {name}");
                assert!(
                    error.contains(&refused.to_string()),
                    "{kind} {name}: {refused}"
                );
            }
            Ok(()) => {
                admitted += 1;
                assert!(!admission.contains(&error), "{kind} {name}: admitted");
            }
        }
    }
    assert!(admitted > 0);
}

fn predicates(check: &Value) -> Vec<Predicate> {
    check["rules"]
        .as_array()
        .unwrap()
        .iter()
        .map(|rule| Predicate {
            path: rule["path"].as_array().unwrap().clone(),
            op: text(&rule["op"]),
            value: rule["value"].clone(),
            source: text(&rule["source"]),
        })
        .collect()
}

#[test]
fn declared_predicates_judge_the_recorded_metrics() {
    let mut judged = 0;
    for (kind, vector) in kind_vectors() {
        let expected = &vector["expected"];
        let (Some(metrics), Some(_)) = (expected.get("metrics"), vector["check"].get("rules"))
        else {
            continue;
        };
        let mut verdict = if metrics["passed"] == false {
            "fail"
        } else {
            "pass"
        };
        for rule in predicates(&vector["check"]) {
            match kind::predicate(metrics, &rule) {
                Ok(true) => {}
                Ok(false) | Err(_) => verdict = "fail",
            }
        }
        if expected["status"] == "not_applicable" {
            assert!(
                expected["reason"]
                    .as_str()
                    .is_some_and(|reason| !reason.is_empty())
            );
            let all_unserved = metrics["rows"]
                .as_array()
                .unwrap()
                .iter()
                .all(|row| row["status"] == "not_applicable");
            if all_unserved {
                assert_eq!(metrics["passed"], false);
            } else {
                assert_eq!(verdict, "pass", "served rows must still pass");
            }
        } else {
            assert_eq!(expected["status"], verdict, "{kind} {}", vector["name"]);
        }
        judged += 1;
    }
    assert!(judged > 0);
}

fn spread(summary: &Value) -> Spread {
    let number = |key: &str| {
        summary[key]
            .as_f64()
            .unwrap_or_else(|| panic!("{key} in {summary}"))
    };
    Spread {
        mean_abs_dlogprob: number("mean_abs_dlogprob"),
        top1_decisive: number("top1_decisive"),
        top1_near_tie: number("top1_near_tie"),
        top1_disagreement: number("top1_disagreement"),
    }
}

fn limits_json(limits: &Spread) -> Value {
    serde_json::to_value(limits).unwrap()
}

/// variant-spread.v2's verdict of a category or media case, derived
/// independently: the mean within its limit, and each top-1 metric within
/// its limit or, where the power test says the limit cannot judge the set
/// (against `reference`, the BF16 run's rates there, when given), within
/// the paired margin its recorded discordant counts give.
fn v2_within(row: &Value, limits: &Spread, reference: Option<&Value>, name: &Value) -> bool {
    let measured = spread(row);
    let mut within = measured.mean_abs_dlogprob <= limits.mean_abs_dlogprob;
    let positions = row["scored"].as_u64().unwrap();
    for (metric, value, limit) in [
        (
            "top1_decisive",
            measured.top1_decisive,
            limits.top1_decisive,
        ),
        (
            "top1_near_tie",
            measured.top1_near_tie,
            limits.top1_near_tie,
        ),
        (
            "top1_disagreement",
            measured.top1_disagreement,
            limits.top1_disagreement,
        ),
    ] {
        if value <= limit {
            assert_eq!(row["passed_by"][metric], json!("limit"), "{name}: {metric}");
            continue;
        }
        let power = &row["power"][metric];
        if power.is_null() {
            within = false;
            continue;
        }
        let rate = power["rate"].as_f64().unwrap();
        if let Some(reference) = reference {
            assert_eq!(
                json!(rate),
                reference[metric],
                "{name}: {metric} power rate"
            );
        }
        let has_power = limits::power(rate, limit, positions, limits::CATEGORY_SIGMAS).has_power;
        assert_eq!(
            power["has_power"],
            json!(has_power),
            "{name}: {metric} power"
        );
        let paired = &row["paired_reference"][metric];
        let fallback = !has_power && {
            let (test, baseline) = (
                paired["test_only"].as_u64().unwrap(),
                paired["baseline_only"].as_u64().unwrap(),
            );
            let margin = limits::CATEGORY_SIGMAS * ((test + baseline) as f64).sqrt();
            assert_eq!(paired["margin"], json!(margin), "{name}: {metric} margin");
            test as f64 - baseline as f64 <= margin
        };
        let by = if fallback {
            json!("paired_reference")
        } else {
            Value::Null
        };
        assert_eq!(row["passed_by"][metric], by, "{name}: {metric}");
        within &= fallback;
    }
    within
}

#[test]
fn recorded_limits_and_verdicts_follow_the_spread_rule() {
    let rule = SpreadRule::default();
    let mut checked = 0;
    for (kind, vector) in kind_vectors() {
        let name = &vector["name"];
        let Some(metrics) = vector["expected"].get("metrics") else {
            continue;
        };
        if kind == "teacher_forced" && metrics.get("reference_spread").is_some() {
            let variants = metrics["reference_spread"].as_object().unwrap();
            let pooled: Vec<Spread> = variants
                .values()
                .map(|spread_of| spread(&spread_of["all"]))
                .collect();
            let scored = metrics["scored"].as_u64().unwrap();
            let limits = limits::spread_limits(&pooled, scored, &rule).unwrap();
            assert_eq!(
                limits_json(&limits),
                metrics["limits"],
                "{name}: pooled limits"
            );
            let mut failed = Vec::new();
            for (category, row) in metrics["by_category"].as_object().unwrap() {
                let spreads: Vec<Spread> = variants
                    .values()
                    .map(|spread_of| spread(&spread_of["by_category"][category]))
                    .collect();
                let category_limits =
                    limits::spread_limits(&spreads, row["scored"].as_u64().unwrap(), &rule)
                        .unwrap();
                assert_eq!(
                    limits_json(&category_limits),
                    row["limits"],
                    "{name}: {category} limits"
                );
                let bf16 = variants
                    .get("bf16")
                    .map(|spread_of| &spread_of["by_category"][category]);
                let within = v2_within(row, &category_limits, bf16, name);
                assert_eq!(row["passed"], json!(within), "{name}: {category} verdict");
                if !within {
                    failed.push(json!(category));
                }
            }
            assert_eq!(
                metrics["failed_categories"],
                Value::Array(failed.clone()),
                "{name}"
            );
            let verdict = limits::within(&spread(metrics), &limits) && failed.is_empty();
            let key = if metrics.get("diagnostic_passed").is_some() {
                "diagnostic_passed"
            } else {
                "passed"
            };
            assert_eq!(metrics[key], json!(verdict), "{name}: verdict");
            checked += 1;
        }
        if kind == "media_forced" {
            for row in metrics["rows"].as_array().unwrap() {
                let scores = &row["logprobs"];
                if scores.get("error").is_some() || scores.get("limits").is_none() {
                    continue;
                }
                let within = v2_within(scores, &spread(&scores["limits"]), None, name);
                assert_eq!(
                    scores["passed"],
                    json!(within),
                    "{name}: item {}",
                    row["id"]
                );
                checked += 1;
            }
        }
    }
    assert!(checked > 0);
}

/// Whether an expected error is one of the served-row and flip checks.
fn served_row_error(error: &str) -> bool {
    [
        "no top-1 flips",
        "served rows",
        "served row ",
        "recorded top-1 flips",
        "do not add up to the recorded disagreements",
        "do not add up to its disagreement",
    ]
    .iter()
    .any(|message| error.contains(message))
}

/// Flips derived from served rows, judged by the near-tie rule from the
/// reference variant `scores` (`kinds.judged`), in offset order, each
/// after `fields`.
fn judged(
    derived: &std::collections::BTreeMap<usize, Value>,
    scores: &Value,
    fields: &Value,
) -> Vec<Value> {
    let tops = limits::gap_top(scores).expect("top-k rows");
    derived
        .iter()
        .map(|(offset, served)| {
            let flip = limits::flip(
                &scores["top1"][*offset],
                served,
                Some(&tops[*offset]),
                NEAR_TIE_NATS,
            )
            .expect("a derived flip differs from the reference");
            let mut entry = fields.clone();
            entry["offset"] = json!(offset);
            entry["served"] = served.clone();
            entry["gap"] = json!(flip.gap);
            entry["decisive"] = json!(flip.decisive);
            entry
        })
        .collect()
}

/// The counts and rates of `flips` over `scored` must be those of `summary`.
fn check_counts(flips: &[Value], scored: u64, summary: &Value, where_: &str) {
    let flips: Vec<Option<Flip>> = flips
        .iter()
        .map(|flip| {
            Some(Flip {
                gap: flip["gap"].as_f64(),
                decisive: flip["decisive"].as_bool().unwrap(),
            })
        })
        .collect();
    let counts = serde_json::to_value(limits::top1_summary(&flips, scored)).unwrap();
    for (key, value) in counts.as_object().unwrap() {
        assert_eq!(&summary[key], value, "{where_}: {key}");
    }
}

fn ids(value: &Value) -> Vec<Value> {
    value.as_array().unwrap().clone()
}

/// The teacher kind's served-row checks in its verify order, then its
/// flips from the served rows (`TeacherForced.verify`, `.score`).
fn teacher_flips(receipt: &Value, reference: &Value) -> Result<Vec<Value>, KindError> {
    let gate = reference["gate"].as_str().unwrap();
    let texts = reference["texts"].as_array().unwrap();
    let flips = receipt["top1_flips"]
        .as_array()
        .filter(|flips| {
            flips.iter().all(|flip| {
                texts.iter().any(|text| text["name"] == flip["text"] && text["category"] == flip["category"])
            })
        })
        .ok_or_else(|| {
            KindError::Insufficient(
                "no top-1 flips of the reference's texts (top1_flips; receipts before the near-tie rule have none)"
                    .into(),
            )
        })?;
    let rows = receipt["rows"].as_array().unwrap();
    let mut judged_flips = Vec::new();
    for text in texts {
        let where_ = format!("text {}", text["name"].as_str().unwrap());
        let from = text["score_from"].as_u64().unwrap() as usize;
        let tokens = &ids(&text["prompt_token_ids"])[from..];
        let row = rows.iter().find(|row| row["name"] == text["name"]);
        let derived = kind::served_flips(row, tokens, &ids(&text[gate]["top1"]), &where_)?;
        let of_text = Value::Array(
            flips
                .iter()
                .filter(|flip| flip["text"] == text["name"])
                .cloned()
                .collect(),
        );
        kind::match_flips(Some(&of_text), &derived, &where_)?;
        let fields = json!({"text": text["name"], "category": text["category"]});
        judged_flips.extend(judged(&derived, &text[gate], &fields));
    }
    let categories = receipt["by_category"].as_object().unwrap();
    let adds_up = flips.len() as u64 == receipt["top1_disagreements"].as_u64().unwrap_or(u64::MAX)
        && categories.iter().all(|(name, row)| {
            flips
                .iter()
                .filter(|flip| flip["category"] == json!(name))
                .count() as u64
                == row["top1_disagreements"].as_u64().unwrap_or(u64::MAX)
        });
    if !adds_up {
        return Err(KindError::Insufficient(
            "the top-1 flips do not add up to the recorded disagreements".into(),
        ));
    }
    Ok(judged_flips)
}

/// The media kind's served-row checks for one item, then its flips.
fn media_flips(scores: &Value, case: &Value, gate: &str) -> Result<Vec<Value>, KindError> {
    let where_ = format!("item {}", case["id"].as_str().unwrap());
    let variant = &case["exact"][gate];
    let derived = kind::served_flips(
        Some(scores),
        &ids(&case["answer_token_ids"]),
        &ids(&variant["top1"]),
        &where_,
    )?;
    kind::match_flips(scores.get("top1_flips"), &derived, &where_)?;
    let rate = scores["top1_disagreement"].as_f64().unwrap();
    let answer = case["answer_token_ids"].as_array().unwrap().len() as f64;
    if scores["top1_flips"].as_array().unwrap().len() as f64 != (rate * answer).round_ties_even() {
        return Err(KindError::Insufficient(format!(
            "{where_}: the top-1 flips do not add up to its disagreement {}",
            runtime_eval::json::python_float_repr(rate)
        )));
    }
    Ok(judged(&derived, variant, &json!({})))
}

#[test]
fn flips_follow_the_served_rows_and_the_near_tie_rule() {
    let mut checked = 0;
    for (kind_name, vector) in kind_vectors() {
        let name = vector["name"].as_str().unwrap();
        let expected = &vector["expected"];
        let error = expected["error"].as_str().unwrap_or_default();
        let metrics = expected.get("metrics");
        let reference = &vector["reference"];
        let receipt = &vector["receipt"];
        if kind_name == "teacher_forced"
            && reference.get("texts").is_some()
            && receipt.get("limits").is_some()
        {
            if served_row_error(error) {
                let refused = teacher_flips(receipt, reference).expect_err(name);
                assert!(error.contains(&refused.to_string()), "{name}: {refused}");
                checked += 1;
            } else if let Some(metrics) = metrics {
                let flips = teacher_flips(receipt, reference)
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                assert_eq!(
                    Value::Array(flips.clone()),
                    metrics["top1_flips"],
                    "{name}: flips"
                );
                check_counts(&flips, metrics["scored"].as_u64().unwrap(), metrics, name);
                for (category, row) in metrics["by_category"].as_object().unwrap() {
                    let of_category: Vec<Value> = flips
                        .iter()
                        .filter(|flip| flip["category"] == json!(category))
                        .cloned()
                        .collect();
                    check_counts(
                        &of_category,
                        row["scored"].as_u64().unwrap(),
                        row,
                        &format!("{name}: {category}"),
                    );
                }
                checked += 1;
            }
        }
        if kind_name == "media_forced" && reference.get("cases").is_some() {
            let gate = reference["gate"].as_str().unwrap();
            let cases = reference["cases"].as_array().unwrap();
            let rows = receipt["rows"].as_array().cloned().unwrap_or_default();
            if served_row_error(error) {
                let refused = rows
                    .iter()
                    .zip(cases)
                    .filter(|(row, _)| row["logprobs"].get("error").is_none())
                    .find_map(|(row, case)| media_flips(&row["logprobs"], case, gate).err())
                    .unwrap_or_else(|| panic!("{name}: accepted"));
                assert!(error.contains(&refused.to_string()), "{name}: {refused}");
                checked += 1;
            } else if let Some(metrics) = metrics {
                for ((row, case), recomputed) in rows
                    .iter()
                    .zip(cases)
                    .zip(metrics["rows"].as_array().unwrap())
                {
                    if row["status"] == "not_applicable" {
                        assert_eq!(row, recomputed, "unserved evidence must be preserved");
                        assert!(row["passed"].is_null());
                        continue;
                    }
                    let scores = &recomputed["logprobs"];
                    if row["logprobs"].get("error").is_some() {
                        continue;
                    }
                    let flips = media_flips(&row["logprobs"], case, gate)
                        .unwrap_or_else(|error| panic!("{name}: {error}"));
                    assert_eq!(
                        Value::Array(flips.clone()),
                        scores["top1_flips"],
                        "{name}: {}",
                        row["id"]
                    );
                    check_counts(
                        &flips,
                        scores["scored"].as_u64().unwrap(),
                        scores,
                        &format!("{name}: {}", row["id"]),
                    );
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 0, "no vector records top-1 flips");
}
