//! The check kinds: the contract of `tools/qual/kinds.py`. A kind captures a receipt from a live server, then
//! verifies, scores and judges a receipt offline. The verdict procedure
//! ([`rescore`]) is the same for a receipt just captured and a stored one.

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::time::Instant;

use serde_json::{Map, Value, json};

use crate::declaration::{Check, Loaded, Predicate};
use crate::http::{BoundServer, Client, ServerIdentity};
use crate::json::{self, python_cmp, python_eq, sha256_hex, truthy};
use crate::limits;
use crate::references::Verified;

/// A check's verdict.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Status {
    Pass,
    Fail,
    /// Declared informational: neither run nor judged.
    Informational,
    /// What is stored or served cannot be judged.
    Insufficient,
    /// Not run: the run stopped before it, or this build cannot run it.
    NotRun,
    /// A validated check of modalities this server does not serve.
    NotApplicable,
    /// Failed only where the check's declared waivers cover it, each within
    /// its ceiling.
    Waived,
}

impl Status {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Fail => "fail",
            Self::Informational => "informational",
            Self::Insufficient => "insufficient",
            Self::NotRun => "not_run",
            Self::NotApplicable => "not_applicable",
            Self::Waived => "waived",
        }
    }
}

/// What a kind is: the facts `kinds.describe` lists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KindInfo {
    pub name: &'static str,
    pub version: &'static str,
    /// The qualification modules whose checks are this kind.
    pub modules: &'static [&'static str],
    /// The pinned reference the kind reads, if any.
    pub reference: Option<&'static str>,
    /// The kind's metric fields.
    pub metrics: &'static [&'static str],
}

/// The shared kinds `chopinfer eval` runs (docs/check-kinds.md).
pub const SHARED_KINDS: [KindInfo; 6] = [
    KindInfo {
        name: "teacher_forced",
        version: "teacher_forced.v1",
        modules: &["teacher"],
        reference: Some("teacher"),
        metrics: &[
            "passed",
            "scored",
            "mean_abs_dlogprob",
            "max_abs_dlogprob",
            "top1_disagreements",
            "top1_disagreement",
            "top1_near_ties",
            "top1_near_tie",
            "top1_decisive_flips",
            "top1_decisive",
            "top1_flips",
            "near_tie_nats",
            "limits",
            "reference_spread",
            "by_category",
            "failed_categories",
            "reference_sha256",
            "reference_binding",
            "server",
            "diagnostic_passed",
            "informational_reason",
        ],
    },
    KindInfo {
        name: "media_forced",
        version: "media_forced.v1",
        modules: &["multimodal"],
        reference: Some("media"),
        metrics: &[
            "passed",
            "rows",
            "mixed",
            "limits",
            "reference",
            "reference_sha256",
            "exact_reference_sha256",
            "capture_sha256",
            "reference_binding",
            "server",
        ],
    },
    KindInfo {
        name: "greedy_pin",
        version: "greedy_pin.v1",
        modules: &["greedy_equality"],
        reference: None,
        metrics: &["equal", "first_differences", "finish_differences"],
    },
    KindInfo {
        name: "determinism",
        version: "determinism.v1",
        modules: &["determinism"],
        reference: None,
        metrics: &[
            "passed",
            "left_arm",
            "right_arm",
            "requests",
            "tokens",
            "equal",
            "first_difference",
            "max_abs_difference",
            "cold_repeat",
            "prefix_hit",
        ],
    },
    KindInfo {
        name: "api",
        version: "api.v1",
        modules: &["api"],
        reference: None,
        metrics: &["passed", "rows"],
    },
    KindInfo {
        name: "structured",
        version: "structured.v1",
        modules: &["structured"],
        reference: None,
        metrics: &["passed", "summary", "settings"],
    },
];

/// The shared kind of a check run by `module`; `None` for a check the
/// Python suite judges by its declared predicates alone (`receipt`).
pub fn shared_kind(module: &str) -> Option<&'static KindInfo> {
    SHARED_KINDS
        .iter()
        .find(|kind| kind.modules.contains(&module))
}

/// Why a kind cannot capture, verify or score.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KindError {
    /// A required field or item is missing, counts disagree, or the receipt
    /// scored another reference (`kinds.Insufficient`).
    Insufficient(String),
    /// The receipt is malformed or failed before any predicate.
    Invalid(String),
    /// The server could not be reached or timed out.
    Environment(String),
    /// This build cannot run the check live.
    Unsupported(String),
}

impl std::fmt::Display for KindError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Insufficient(message)
            | Self::Invalid(message)
            | Self::Environment(message)
            | Self::Unsupported(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for KindError {}

/// A predicate path the metrics lack (Python's `KeyError`), named as
/// Python's `KeyError` prints the key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MissingMetric(pub String);

/// The values at a predicate path; `*` takes every child.
pub fn select<'a>(value: &'a Value, path: &[Value]) -> Result<Vec<&'a Value>, MissingMetric> {
    let Some((key, tail)) = path.split_first() else {
        return Ok(vec![value]);
    };
    let missing = || MissingMetric(crate::repr::repr(key));
    if key.as_str() == Some("*") {
        let children: Vec<&Value> = match value {
            Value::Object(map) => map.values().collect(),
            Value::Array(items) => items.iter().collect(),
            _ => return Err(missing()),
        };
        let mut selected = Vec::new();
        for child in children {
            selected.extend(select(child, tail)?);
        }
        return Ok(selected);
    }
    let child = match (value, key) {
        (Value::Object(map), Value::String(name)) => map.get(name),
        // Python indexes a list with an int, and a bool is an int.
        (Value::Array(items), Value::Number(_) | Value::Bool(_)) => key
            .as_i64()
            .or_else(|| key.as_bool().map(i64::from))
            .and_then(|index| {
                let length = items.len() as i64;
                let index = if index < 0 { index + length } else { index };
                usize::try_from(index).ok()
            })
            .and_then(|index| items.get(index)),
        _ => None,
    };
    select(child.ok_or_else(missing)?, tail)
}

/// Python's `type(a) is type(b)` for JSON values: int, float and bool are
/// distinct types.
fn same_type(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(a), Value::Number(b)) => a.is_f64() == b.is_f64(),
        _ => std::mem::discriminant(a) == std::mem::discriminant(b),
    }
}

fn length(value: &Value) -> Option<usize> {
    match value {
        Value::String(text) => Some(text.chars().count()),
        Value::Array(items) => Some(items.len()),
        Value::Object(map) => Some(map.len()),
        _ => None,
    }
}

/// Whether a declared predicate holds over `value` (`kinds.predicate`).
pub fn predicate(value: &Value, rule: &Predicate) -> Result<bool, MissingMetric> {
    let values = select(value, &rule.path)?;
    if values.is_empty() {
        return Ok(false);
    }
    let expected = &rule.value;
    let at_least = |count: usize| {
        count > 0
            && python_cmp(&Value::from(count as u64), expected)
                .is_some_and(|order| order != Ordering::Less)
    };
    Ok(match rule.op.as_str() {
        "true_count_ge" => at_least(
            values
                .iter()
                .filter(|item| ***item == Value::Bool(true))
                .count(),
        ),
        "le" | "ge" => values.iter().all(|item| {
            // A finite number, not a bool, compared exactly with the bound.
            let number =
                matches!(item, Value::Number(_)) && item.as_f64().is_some_and(f64::is_finite);
            let wanted = if rule.op == "le" {
                Ordering::Greater
            } else {
                Ordering::Less
            };
            number && python_cmp(item, expected).is_some_and(|order| order != wanted)
        }),
        "eq" => values
            .iter()
            .all(|item| same_type(item, expected) && python_eq(item, expected)),
        "count_ge" => values.iter().all(|item| length(item).is_some_and(at_least)),
        _ => false,
    })
}

/// Refuses a receipt the gate fails before any predicate (`kinds.admit`):
/// not an object, mechanics-only or skipped coverage outside a mechanics
/// run, or errors.
pub fn admit(receipt: &Value, mechanics: bool) -> Result<(), KindError> {
    let Value::Object(map) = receipt else {
        return Err(KindError::Invalid("expected a JSON object receipt".into()));
    };
    if let Some(server) = map.get("server") {
        let serving = server
            .pointer("/health/serving")
            .or_else(|| server.get("serving"))
            .unwrap_or(server);
        crate::http::require_qualified_model_options(serving).map_err(KindError::Invalid)?;
    }
    let flag = |key: &str| map.get(key).is_some_and(truthy);
    if !mechanics && (flag("mechanics_only") || flag("skipped")) {
        return Err(KindError::Invalid(
            "quality check reported mechanics-only or skipped coverage".into(),
        ));
    }
    if flag("errors") {
        return Err(KindError::Invalid("check reported errors".into()));
    }
    Ok(())
}

/// A JSON int at least zero (Python's `type(v) is int and v >= 0`).
pub(crate) fn natural(value: &Value) -> bool {
    matches!(value, Value::Number(number) if !number.is_f64() && number.as_u64().is_some())
}

/// A finite JSON number, int or float (not a bool).
pub(crate) fn finite(value: &Value) -> bool {
    matches!(value, Value::Number(number) if number.as_f64().is_some_and(f64::is_finite))
}

/// The SHA-256 of served rows as compact JSON (`common.rows_digest`).
pub fn rows_digest(rows: &Value) -> String {
    sha256_hex(&json::compact(rows))
}

/// The served top-1 of a well-formed row of [token, logprob, rank]
/// entries (`common.entries_top1`): the lowest rank (a missing rank after
/// every given one), then the highest logprob, then the lowest token ID;
/// the first such entry.
pub fn served_top1(entries: &[Value]) -> &Value {
    let order = |a: &Value, b: &Value| -> Ordering {
        let rank = |entry: &Value| entry[2].as_u64().map_or(1_000_000_000, i128::from);
        rank(a)
            .cmp(&rank(b))
            .then_with(|| python_cmp(&b[1], &a[1]).unwrap_or(Ordering::Equal))
            .then_with(|| python_cmp(&a[0], &b[0]).unwrap_or(Ordering::Equal))
    };
    let mut best = &entries[0];
    for entry in &entries[1..] {
        if order(entry, best) == Ordering::Less {
            best = entry;
        }
    }
    &best[0]
}

/// The top-1 flips a holder's served rows show (`kinds.served_flips`):
/// offset to served top-1 wherever it differs from the reference's `top1`.
/// Insufficient unless the holder keeps one served row (`served_rows`, lists
/// of [token, logprob, rank] entries) per scored token of `tokens`, each
/// well formed and holding that token, matching their digest
/// (`served_rows_sha256`).
pub fn served_flips(
    holder: Option<&Value>,
    tokens: &[Value],
    top1: &[Value],
    where_: &str,
) -> Result<BTreeMap<usize, Value>, KindError> {
    let insufficient = |message: String| KindError::Insufficient(format!("{where_}: {message}"));
    let rows = holder
        .filter(|holder| holder.is_object())
        .and_then(|holder| holder.get("served_rows"))
        .and_then(Value::as_array)
        .filter(|rows| rows.len() == tokens.len())
        .ok_or_else(|| {
            insufficient(format!(
                "no served rows for its {} positions (served_rows; receipts before they were recorded have none)",
                tokens.len()
            ))
        })?;
    let holder = holder.expect("rows came from it");
    if holder.get("served_rows_sha256").and_then(Value::as_str)
        != Some(rows_digest(&holder["served_rows"]).as_str())
    {
        return Err(insufficient(
            "the served rows differ from their digest".into(),
        ));
    }
    let mut flips = BTreeMap::new();
    for (offset, (row, token)) in rows.iter().zip(tokens).enumerate() {
        let well_formed = row
            .as_array()
            .filter(|entries| !entries.is_empty())
            .filter(|entries| {
                entries.iter().all(|entry| {
                    entry.as_array().is_some_and(|parts| {
                        parts.len() == 3
                            && natural(&parts[0])
                            && finite(&parts[1])
                            && (parts[2].is_null()
                                || (natural(&parts[2]) && parts[2].as_u64() >= Some(1)))
                    })
                }) && entries.iter().enumerate().all(|(index, entry)| {
                    entries[..index]
                        .iter()
                        .all(|other| !python_eq(&other[0], &entry[0]))
                }) && entries.iter().any(|entry| python_eq(&entry[0], token))
            });
        let Some(entries) = well_formed else {
            return Err(insufficient(format!(
                "served row {offset} is malformed or lacks its scored token"
            )));
        };
        let served = served_top1(entries);
        if !top1.get(offset).is_some_and(|top| python_eq(served, top)) {
            flips.insert(offset, served.clone());
        }
    }
    Ok(flips)
}

/// Insufficient unless the recorded top-1 `flips` are exactly the flips
/// `derived` from the served rows: the same offsets and tokens
/// (`kinds.match_flips`).
pub fn match_flips(
    flips: Option<&Value>,
    derived: &BTreeMap<usize, Value>,
    where_: &str,
) -> Result<(), KindError> {
    let flips = flips
        .and_then(Value::as_array)
        .filter(|flips| flips.iter().all(Value::is_object))
        .ok_or_else(|| {
            KindError::Insufficient(format!(
                "{where_}: no top-1 flips (top1_flips; receipts before the near-tie rule have none)"
            ))
        })?;
    let null = Value::Null;
    let recorded: Vec<(&Value, &Value)> = flips
        .iter()
        .map(|flip| {
            (
                flip.get("offset").unwrap_or(&null),
                flip.get("served").unwrap_or(&null),
            )
        })
        .collect();
    let distinct = recorded.iter().enumerate().all(|(index, (offset, _))| {
        recorded[..index]
            .iter()
            .all(|(other, _)| !python_eq(other, offset))
    });
    let same = recorded.len() == derived.len()
        && recorded.iter().all(|(offset, served)| {
            derived.iter().any(|(at, derived_served)| {
                python_eq(offset, &Value::from(*at as u64)) && python_eq(served, derived_served)
            })
        });
    if distinct && same {
        Ok(())
    } else {
        Err(KindError::Insufficient(format!(
            "{where_}: the recorded top-1 flips are not those of its served rows"
        )))
    }
}

/// One judged predicate.
#[derive(Clone, Debug, PartialEq)]
pub struct Judged {
    pub rule: Predicate,
    pub passed: bool,
    /// The declared waiver that covers this failing predicate, as applied.
    pub waiver: Option<Value>,
}

impl Judged {
    /// The predicate as reports and rescore results state it.
    pub fn to_json(&self) -> Value {
        let mut value = json!({"path": self.rule.path, "op": self.rule.op, "value": self.rule.value,
                               "source": self.rule.source, "passed": self.passed});
        if let Some(waiver) = &self.waiver {
            value["waiver"] = waiver.clone();
        }
        value
    }
}

/// The verdict of judged predicates and the kind's own `passed`, with the
/// check's declared waivers (`checks.<id>.waivers`) applied: `Pass` when
/// everything passes, `Waived` when every failure is covered, else `Fail`.
/// A predicate waiver covers its failing predicate when the predicate holds
/// with the waiver's ceiling in place of the rule's value. Category waivers
/// cover a failed teacher verdict when the pooled scores are within their
/// limits and each failed category fails only the waived metric, at most
/// its ceiling; they then cover the `passed` and `failed_categories`
/// predicates. Each applied waiver is recorded on what it covers. Default
/// deny: waivers the declaration validator refuses
/// ([`crate::declaration::validate_waivers`]) cover nothing, whatever path
/// built the check.
pub fn waive(
    value: &Value,
    check: &Check,
    passed: bool,
    mut judged: Vec<Judged>,
) -> Result<(Status, Vec<Judged>), MissingMetric> {
    if passed && judged.iter().all(|rule| rule.passed) {
        return Ok((Status::Pass, judged));
    }
    let waivers =
        if crate::declaration::validate_waivers(&check.declared, &check.module, &check.rules)
            .is_ok()
        {
            check.declared["waivers"]
                .as_array()
                .cloned()
                .unwrap_or_default()
        } else {
            Vec::new()
        };
    let categories = if passed {
        None
    } else {
        category_waivers(value, &waivers)
    };
    for item in judged.iter_mut().filter(|item| !item.passed) {
        let path = Value::Array(item.rule.path.clone());
        if let Some(waiver) = waivers.iter().find(|waiver| {
            waiver["predicate"]["path"] == path
                && waiver["predicate"]["op"] == item.rule.op.as_str()
        }) {
            let loosened = Predicate {
                value: waiver["ceiling"].clone(),
                ..item.rule.clone()
            };
            if predicate(value, &loosened)? {
                item.waiver = Some(
                    json!({"predicate": waiver["predicate"], "measured": select(value, &item.rule.path)?,
                                          "ceiling": waiver["ceiling"], "reason": waiver["reason"],
                                          "decision": waiver["decision"]}),
                );
                continue;
            }
        }
        if let Some(applied) = &categories {
            let verdict = item.rule.op == "eq"
                && ((path == json!(["passed"]) && item.rule.value == Value::Bool(true))
                    || (path == json!(["failed_categories"]) && item.rule.value == json!([])));
            if verdict {
                item.waiver = Some(json!({"categories": applied}));
            }
        }
    }
    let covered = passed || categories.is_some();
    let status = if covered
        && judged
            .iter()
            .all(|item| item.passed || item.waiver.is_some())
    {
        Status::Waived
    } else {
        Status::Fail
    };
    Ok((status, judged))
}

/// The category waivers that cover a failed teacher verdict, as applied, or
/// None when the pooled scores fail their limits or a failed category is
/// not covered.
fn category_waivers(value: &Value, waivers: &[Value]) -> Option<Vec<Value>> {
    use crate::kinds::spread_of;
    let failed = value["failed_categories"]
        .as_array()
        .filter(|failed| !failed.is_empty())?;
    if !limits::within(&spread_of(value).ok()?, &spread_of(&value["limits"]).ok()?) {
        return None;
    }
    let mut applied = Vec::new();
    for name in failed {
        let row = &value["by_category"][name.as_str()?];
        let failing: Vec<&str> = crate::declaration::WAIVER_METRICS
            .iter()
            .copied()
            .filter(|metric| {
                let over = row[*metric]
                    .as_f64()
                    .zip(row["limits"][*metric].as_f64())
                    .is_none_or(|(measured, limit)| measured > limit);
                over && row["passed_by"][*metric].is_null()
            })
            .collect();
        let waiver = waivers.iter().find(|waiver| {
            waiver["category"] == *name
                && failing.len() == 1
                && waiver["metric"] == failing[0]
                && row[failing[0]]
                    .as_f64()
                    .zip(waiver["ceiling"].as_f64())
                    .is_some_and(|(measured, ceiling)| measured <= ceiling)
        })?;
        applied.push(
            json!({"category": name, "metric": waiver["metric"], "measured": row[failing[0]],
                            "ceiling": waiver["ceiling"], "reason": waiver["reason"],
                            "decision": waiver["decision"]}),
        );
    }
    Some(applied)
}

/// What a live capture runs against.
pub struct Capture<'a> {
    pub client: &'a mut Client,
    pub declaration: &'a Loaded,
    pub check: &'a Check,
    pub server: &'a ServerIdentity,
    pub bound: &'a BoundServer,
    /// The pinned reference of the kind, verified.
    pub reference: Option<&'a Verified>,
    pub deadline: Instant,
    /// The host load when the capture started (`capture::host_load::sample`).
    pub host_load_start: Value,
    /// A mechanics run: its receipts never qualify.
    pub mechanics: bool,
    /// The compact capture the check composes when its arguments ask for
    /// one (teacher `--capture-out`).
    pub capture_out: Option<CompactCapture>,
}

/// A check's compact capture: where it goes and, once the check composed
/// it, its bytes. The kind only composes it; the run publishes it after
/// confirming that the bound process wrote every response, so a refused
/// run leaves no capture.
pub struct CompactCapture {
    pub path: std::path::PathBuf,
    pub bytes: Option<Vec<u8>>,
}

/// A check kind. Defaults are the `receipt` kind's: the metrics are the
/// receipt's fields and the declared predicates are the verdict.
pub trait Kind {
    fn info(&self) -> &'static KindInfo;

    /// Live: runs the check against the server and returns its receipt (a
    /// `chopinfer-qualification/v1` object).
    fn capture(&self, _capture: &mut Capture<'_>) -> Result<Value, KindError> {
        Err(KindError::Unsupported(format!(
            "this build does not capture the {} kind",
            self.info().name
        )))
    }

    /// The metric fields of a receipt: the kind's, then those a declared
    /// predicate reads. A predicate over the whole receipt or each of its
    /// fields reads every field.
    fn fields(&self, receipt: &Value, check: &Check) -> Vec<String> {
        default_fields(self.info(), receipt, check)
    }

    /// The pinned reference the check reads, by its declared name.
    fn reference_name(&self, _check: &Check) -> Option<String> {
        self.info().reference.map(String::from)
    }

    /// The pinned reference as the kind reads it (`kinds.<Kind>.reference`).
    fn reference(
        &self,
        _check: &Check,
        _verified: Option<&Verified>,
    ) -> Result<Option<Value>, KindError> {
        Ok(None)
    }

    /// Offline: refuses with [`KindError::Insufficient`] unless the receipt
    /// holds every field and item the kind requires, with counts that agree
    /// with each other and with the reference. Returns the metric fields
    /// read as recorded.
    fn verify(
        &self,
        receipt: &Value,
        _reference: Option<&Value>,
        check: &Check,
    ) -> Result<Vec<String>, KindError> {
        Ok(self.fields(receipt, check))
    }

    /// Offline: the receipt's metrics, recomputed where the kind derives
    /// them.
    fn score(
        &self,
        receipt: &Value,
        _reference: Option<&Value>,
        check: &Check,
        mechanics: bool,
    ) -> Result<Map<String, Value>, KindError> {
        default_score(self, receipt, check, mechanics)
    }

    /// `pass` when every declared predicate holds, else `fail`.
    fn judge(
        &self,
        metrics: &Map<String, Value>,
        check: &Check,
    ) -> Result<(Status, Vec<Judged>), MissingMetric> {
        let value = Value::Object(metrics.clone());
        let mut judged = Vec::new();
        for rule in &check.rules {
            judged.push(Judged {
                passed: predicate(&value, rule)?,
                rule: rule.clone(),
                waiver: None,
            });
        }
        let passed = value["passed"] != false;
        waive(&value, check, passed, judged)
    }
}

/// The default metric fields (`kinds.Kind.fields`): the kind's metrics,
/// then the first key of every declared predicate path, as the receipt
/// holds them; every field for a predicate over the whole receipt.
pub fn default_fields(info: &KindInfo, receipt: &Value, check: &Check) -> Vec<String> {
    let map = receipt.as_object().cloned().unwrap_or_default();
    let paths: Vec<&Vec<Value>> = check.rules.iter().map(|rule| &rule.path).collect();
    if paths
        .iter()
        .any(|path| path.first().is_none_or(|key| key.as_str() == Some("*")))
    {
        return map.keys().cloned().collect();
    }
    let mut fields: Vec<String> = Vec::new();
    let names = info.metrics.iter().map(|name| (*name).to_owned()).chain(
        paths
            .iter()
            .filter_map(|path| path[0].as_str().map(String::from)),
    );
    for name in names {
        if map.contains_key(&name) && !fields.contains(&name) {
            fields.push(name);
        }
    }
    fields
}

/// The default score (`kinds.Kind.score`): the receipt admitted, its metric
/// fields as recorded.
pub fn default_score<K: Kind + ?Sized>(
    kind: &K,
    receipt: &Value,
    check: &Check,
    mechanics: bool,
) -> Result<Map<String, Value>, KindError> {
    admit(receipt, mechanics)?;
    let map = receipt.as_object().cloned().unwrap_or_default();
    if !mechanics {
        let mut evidence: Vec<_> = [
            "rows",
            "records",
            "cases",
            "requests",
            "items",
            "texts",
            "positions",
            "scored",
            "completed",
            "count",
        ]
        .iter()
        .filter_map(|key| map.get(*key))
        .collect();
        if evidence.is_empty() && kind.info().name == "receipt" {
            for rule in &check.rules {
                if rule.op == "count_ge"
                    || rule.path.last().and_then(Value::as_str).is_some_and(|key| {
                        [
                            "requests",
                            "items",
                            "positions",
                            "scored",
                            "completed",
                            "count",
                        ]
                        .contains(&key)
                    })
                {
                    evidence.extend(select(receipt, &rule.path).unwrap_or_default());
                }
            }
        }
        if evidence.iter().any(|value| match value {
            Value::Array(v) => v.is_empty(),
            Value::Object(v) => v.is_empty(),
            Value::Number(v) => v.as_f64().is_none_or(|v| v <= 0.0),
            _ => true,
        }) || (kind.info().name == "receipt" && evidence.is_empty())
        {
            return Err(KindError::Insufficient(
                "no item coverage in receipt".into(),
            ));
        }
    }
    Ok(kind
        .fields(receipt, check)
        .into_iter()
        .filter_map(|name| map.get(&name).map(|value| (name, value.clone())))
        .collect())
}

/// One receipt's verdict.
#[derive(Clone, Debug, PartialEq)]
pub struct Rescored {
    pub status: Status,
    pub metrics: Option<Map<String, Value>>,
    pub predicates: Option<Vec<Judged>>,
    pub recorded_fields: Vec<String>,
    pub error: Option<String>,
    pub reason: Option<String>,
}

/// The verdict of one receipt (`kinds.rescore`): admit it as the gate does,
/// verify its fields and coverage (not for a mechanics-only receipt, which
/// never qualifies), score and judge.
pub fn rescore(
    kind: &dyn Kind,
    receipt: &Value,
    reference: Option<&Value>,
    check: &Check,
    mechanics: bool,
) -> Rescored {
    let mut result = Rescored {
        status: Status::Fail,
        metrics: None,
        predicates: None,
        recorded_fields: Vec::new(),
        error: None,
        reason: None,
    };
    let scored = admit(receipt, mechanics).and_then(|()| {
        result.reason = crate::applicability::inapplicable_reason(receipt, check)?;
        if !receipt.get("mechanics_only").is_some_and(truthy) {
            result.recorded_fields = kind.verify(receipt, reference, check)?;
        }
        kind.score(receipt, reference, check, mechanics)
    });
    let metrics = match scored {
        Ok(metrics) => metrics,
        Err(KindError::Insufficient(message)) => {
            result.status = Status::Insufficient;
            result.error = Some(message);
            return result;
        }
        Err(error) => {
            result.error = Some(error.to_string());
            return result;
        }
    };
    let all_unserved = result.reason.is_some() && crate::applicability::all_unserved(&metrics);
    result.metrics = Some(metrics);
    if all_unserved {
        result.status = Status::NotApplicable;
        return result;
    }
    match kind.judge(result.metrics.as_ref().expect("set above"), check) {
        Ok((status, predicates)) => {
            result.status = if status == Status::Pass && result.reason.is_some() {
                Status::NotApplicable
            } else {
                status
            };
            result.predicates = Some(predicates);
            if status == Status::Fail {
                result.error = Some("threshold predicate failed".into());
            }
        }
        Err(MissingMetric(key)) => result.error = Some(format!("missing metric {key}")),
    }
    result
}

/// The kinds this build can run.
pub fn implemented() -> Vec<Box<dyn Kind>> {
    crate::kinds::all()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn rule(path: Value, op: &str, value: Value) -> Predicate {
        Predicate {
            path: path.as_array().unwrap().clone(),
            op: op.into(),
            value,
            source: "docs/check-kinds.md".into(),
        }
    }

    #[test]
    fn predicates_hold_as_the_python_contract_states() {
        let receipt = json!({"passed": true, "scored": 10, "mean": 0.1, "rate": 1,
                             "rows": [{"ok": true}, {"ok": false}, {"ok": true}],
                             "by": {"a": {"n": 2}, "b": {"n": 3}}, "none": [], "flag": false});
        let cases = [
            (rule(json!(["passed"]), "eq", json!(true)), true),
            (rule(json!(["scored"]), "eq", json!(10.0)), false),
            (rule(json!(["rate"]), "eq", json!(true)), false),
            (rule(json!(["scored"]), "ge", json!(10)), true),
            (rule(json!(["mean"]), "le", json!(0.1)), true),
            (rule(json!(["passed"]), "le", json!(1)), false),
            (
                rule(json!(["rows", "*", "ok"]), "true_count_ge", json!(2)),
                true,
            ),
            (
                rule(json!(["rows", "*", "ok"]), "true_count_ge", json!(3)),
                false,
            ),
            (rule(json!(["rows"]), "count_ge", json!(3)), true),
            (rule(json!(["by", "*", "n"]), "ge", json!(2)), true),
            (rule(json!(["by", "*", "n"]), "ge", json!(3)), false),
            (rule(json!(["rows", 1, "ok"]), "eq", json!(false)), true),
            (rule(json!(["rows", -1, "ok"]), "eq", json!(true)), true),
            (rule(json!(["none", "*"]), "eq", json!(1)), false),
            (rule(json!(["flag"]), "eq", json!(false)), true),
        ];
        for (rule, expected) in cases {
            assert_eq!(predicate(&receipt, &rule), Ok(expected), "{rule:?}");
        }
        // Exact integer bounds, and bool keys that index lists as Python's do.
        let exact = json!({"n": 9007199254740993u64, "flags": [false, true],
                           "big": u64::MAX});
        for (rule, expected) in [
            (rule(json!(["n"]), "le", json!(9007199254740992u64)), false),
            (rule(json!(["n"]), "ge", json!(9007199254740993u64)), true),
            (rule(json!(["n"]), "le", json!(9007199254740992.0)), false),
            (rule(json!(["n"]), "ge", json!(9007199254740992.0)), true),
            (
                rule(json!(["big"]), "le", json!(18446744073709551615u64)),
                true,
            ),
            (
                rule(json!(["big"]), "ge", json!(1.8446744073709552e19)),
                false,
            ),
            (rule(json!(["flags", true]), "eq", json!(true)), true),
            (rule(json!(["flags", false]), "eq", json!(false)), true),
            (
                rule(json!(["flags"]), "count_ge", json!(18446744073709551615u64)),
                false,
            ),
        ] {
            assert_eq!(predicate(&exact, &rule), Ok(expected), "{rule:?}");
        }
        assert!(predicate(&receipt, &rule(json!(["absent"]), "eq", json!(1))).is_err());
        assert!(predicate(&receipt, &rule(json!(["rows", 5]), "eq", json!(1))).is_err());
    }

    #[test]
    fn the_served_top1_is_the_lowest_rank_then_highest_logprob_then_lowest_token() {
        for (entries, top1) in [
            (json!([[5, -1.0, null], [6, -2.0, 2]]), json!(6)),
            (json!([[5, -1.0, 1], [6, -0.5, 1]]), json!(6)),
            (json!([[7, -1.0, 1], [6, -1.0, 1]]), json!(6)),
            (json!([[7, -1.0, null], [6, -0.5, null]]), json!(6)),
            (
                json!([[9007199254740993u64, -1.0, 1], [9007199254740992u64, -1, 1]]),
                json!(9007199254740992u64),
            ),
            (json!([[4, -1.0, 3], [3, -1.0, 3], [2, -9.0, 2]]), json!(2)),
        ] {
            assert_eq!(served_top1(entries.as_array().unwrap()), &top1, "{entries}");
        }
    }

    #[test]
    fn served_rows_are_bound_by_digest_and_checked() {
        let rows = json!([[[2, -0.49, 1]], [[3, -0.99, 1], [4, -1.5, 2]]]);
        let holder = json!({"served_rows": rows, "served_rows_sha256": rows_digest(&rows)});
        let tokens = [json!(2), json!(3)];
        let flips = served_flips(Some(&holder), &tokens, &[json!(2), json!(4)], "text t").unwrap();
        assert_eq!(flips, BTreeMap::from([(1, json!(3))]));
        match_flips(Some(&json!([{"offset": 1, "served": 3}])), &flips, "text t").unwrap();
        for wrong in [
            json!([]),
            json!([{"offset": 1, "served": 4}]),
            json!([{"offset": 1, "served": 3}, {"offset": 1, "served": 3}]),
        ] {
            assert!(
                match_flips(Some(&wrong), &flips, "text t").is_err(),
                "{wrong}"
            );
        }
        assert!(match_flips(None, &flips, "text t").is_err());
        let refused = |holder: &Value| {
            served_flips(Some(holder), &tokens, &[json!(2), json!(4)], "text t")
                .unwrap_err()
                .to_string()
        };
        assert!(
            refused(&json!({"served_rows": rows, "served_rows_sha256": "0"}))
                .ends_with("differ from their digest")
        );
        for bad in [
            json!([[[2, -0.49, 0]], [[3, -0.99, 1]]]),
            json!([[[2, -0.49, 1]], [[5, -0.99, 1]]]),
            json!([[[2, -0.49, 1]], [[3, -0.99, 1], [3, -1.0, 2]]]),
            json!([[[2, true, 1]], [[3, -0.99, 1]]]),
            json!([[], [[3, -0.99, 1]]]),
        ] {
            let holder = json!({"served_rows": bad, "served_rows_sha256": rows_digest(&bad)});
            assert!(
                refused(&holder).contains("is malformed or lacks its scored token"),
                "{bad}"
            );
        }
        assert!(refused(&json!({})).contains("no served rows for its 2 positions"));
        // The digest is Python's compact JSON in document key order.
        assert_eq!(
            rows_digest(&json!([[[2, -0.49, 1]], [[3, -0.99, 1]]])),
            "1fedf304ec64637651ec91930b4cc70e2ed10460c425b561571e5eaaca1890e8"
        );
    }

    #[test]
    fn admission_refuses_errors_and_mechanics_runs() {
        assert!(admit(&json!([]), false).is_err());
        assert!(admit(&json!({"errors": [{"name": "x"}]}), false).is_err());
        assert!(admit(&json!({"mechanics_only": true}), false).is_err());
        assert!(admit(&json!({"mechanics_only": true}), true).is_ok());
        assert!(admit(&json!({"skipped": [], "errors": []}), false).is_ok());
    }

    #[test]
    fn every_module_maps_to_one_shared_kind() {
        let mut modules: Vec<&str> = SHARED_KINDS
            .iter()
            .flat_map(|kind| kind.modules.iter().copied())
            .collect();
        let count = modules.len();
        modules.sort_unstable();
        modules.dedup();
        assert_eq!(modules.len(), count);
        assert_eq!(shared_kind("teacher").unwrap().name, "teacher_forced");
        assert!(shared_kind("soak").is_none());
    }
}
