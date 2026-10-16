//! The shared check kinds (docs/check-kinds.md), one module each, and the
//! rules they share: the receipt summaries, the reference scores, the
//! near-tie flips and the greedy walk (`tools/qual/kinds.py`).

pub mod api;
pub mod determinism;
pub mod frontend;
pub mod greedy;
pub mod media;
pub mod probes;
pub mod structured;
pub mod teacher;

use serde_json::{Map, Value, json};

use crate::declaration::Check;
use crate::json::{python_cmp, python_eq, python_number};
use crate::kind::{Kind, KindError, finite, natural};
use crate::limits::{self, NEAR_TIE_NATS};
use crate::repr;

/// Every shared kind this build implements.
pub fn all() -> Vec<Box<dyn Kind>> {
    vec![
        Box::new(teacher::TeacherForced),
        Box::new(media::MediaForced),
        Box::new(greedy::GreedyPin),
        Box::new(determinism::Determinism),
        Box::new(api::Api),
        Box::new(structured::Structured),
        Box::new(frontend::FrontendGoldens),
    ]
}

/// Relative agreement of a stored aggregate with the same aggregate summed
/// from its parts: they differ only in floating-point summation order.
pub const AGGREGATE_TOLERANCE: f64 = 1e-9;

/// Insufficient unless `condition` holds (`kinds.require`).
pub fn require(condition: bool, message: impl FnOnce() -> String) -> Result<(), KindError> {
    if condition {
        Ok(())
    } else {
        Err(KindError::Insufficient(message()))
    }
}

/// An error before any predicate, as Python's `ValueError`, `KeyError` or
/// `TypeError` fail a rescore.
pub fn invalid(message: impl Into<String>) -> KindError {
    KindError::Invalid(message.into())
}

/// `math.isclose(a, b, rel_tol, abs_tol)`.
pub fn is_close(a: f64, b: f64, relative: f64, absolute: f64) -> bool {
    if a == b {
        return true;
    }
    if a.is_infinite() || b.is_infinite() {
        return false;
    }
    let difference = (a - b).abs();
    difference <= (relative * b.abs()).max(relative * a.abs()).max(absolute)
}

/// A stored aggregate agrees with the computed one (`kinds.agrees`).
pub fn agrees(stored: Option<&Value>, computed: f64) -> bool {
    stored.is_some_and(finite)
        && is_close(
            python_number(stored.expect("checked")).expect("finite"),
            computed,
            AGGREGATE_TOLERANCE,
            1e-12,
        )
}

/// A field of an object, or `None` (Python's `value.get(key)`).
pub fn get<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
    value.as_object().and_then(|map| map.get(key))
}

/// A field read as Python's `value[key]` reads it: a missing key is a
/// `KeyError`.
pub fn field<'a>(value: &'a Value, key: &str) -> Result<&'a Value, KindError> {
    get(value, key).ok_or_else(|| invalid(format!("'{key}'")))
}

/// A JSON number as an f64 where Python computes with it.
pub fn number(value: &Value, what: &str) -> Result<f64, KindError> {
    python_number(value).ok_or_else(|| invalid(format!("{what} is not a number: {value}")))
}

/// A natural JSON int, as Python indexes or counts with it.
pub fn count(value: &Value, what: &str) -> Result<u64, KindError> {
    value
        .as_u64()
        .filter(|_| natural(value))
        .ok_or_else(|| invalid(format!("{what} is not a count: {}", repr::repr(value))))
}

/// Raises Insufficient unless `summary` holds consistent logprob scores:
/// positions, mean and max |dlogprob|, top-1 disagreements and their rate
/// (`kinds.check_summary`). `scored`, when given, is the count it must hold.
pub fn check_summary(
    summary: Option<&Value>,
    where_: &str,
    scored: Option<&Value>,
) -> Result<(), KindError> {
    let summary = summary.filter(|summary| summary.is_object());
    require(summary.is_some(), || format!("{where_}: no scores"))?;
    let summary = summary.expect("checked");
    let count = get(summary, "scored");
    let flips = get(summary, "top1_disagreements");
    require(
        count.is_some_and(|count| {
            natural(count)
                && count.as_u64() > Some(0)
                && scored.is_none_or(|scored| python_eq(count, scored))
        }),
        || {
            format!(
                "{where_}: scored {} positions, not {}",
                repr::str_of(count),
                repr::str_of(scored)
            )
        },
    )?;
    let count = count.and_then(Value::as_u64).expect("checked");
    let mean = get(summary, "mean_abs_dlogprob");
    let largest = get(summary, "max_abs_dlogprob");
    let ordered = match (mean, largest) {
        (Some(mean), Some(largest)) if finite(mean) && finite(largest) => {
            let (mean, largest) = (
                python_number(mean).expect("finite"),
                python_number(largest).expect("finite"),
            );
            0.0 <= mean && mean <= largest + AGGREGATE_TOLERANCE
        }
        _ => false,
    };
    require(ordered, || {
        format!("{where_}: invalid mean or max |dlogprob|")
    })?;
    let consistent = flips.is_some_and(|flips| {
        natural(flips)
            && flips.as_u64().is_some_and(|flips| {
                flips <= count
                    && agrees(
                        get(summary, "top1_disagreement"),
                        flips as f64 / count as f64,
                    )
            })
    });
    require(consistent, || {
        format!(
            "{where_}: top-1 disagreements {} and rate {} disagree",
            repr::str_of(flips),
            repr::str_of(get(summary, "top1_disagreement"))
        )
    })
}

/// Raises Insufficient unless the summaries `parts` add up to `whole`
/// (`kinds.check_parts`): positions and flips exactly, the max exactly and
/// the mean weighted by positions within the aggregate tolerance.
pub fn check_parts(whole: &Value, parts: &[&Value], where_: &str) -> Result<(), KindError> {
    let read =
        |value: &Value, key: &str| -> Result<f64, KindError> { number(field(value, key)?, key) };
    let whole_scored = field(whole, "scored")?;
    let mut scored: u64 = 0;
    let mut flips: u64 = 0;
    let mut weighted = 0.0;
    let mut largest: Option<&Value> = None;
    for part in parts {
        scored += count(field(part, "scored")?, "scored")?;
        flips += count(field(part, "top1_disagreements")?, "top1_disagreements")?;
        let maximum = field(part, "max_abs_dlogprob")?;
        // Python's max keeps the first of equal values.
        if largest.is_none_or(|seen| python_cmp(maximum, seen) == Some(std::cmp::Ordering::Greater))
        {
            largest = Some(maximum);
        }
        weighted += read(part, "mean_abs_dlogprob")? * read(part, "scored")?;
    }
    let largest = largest.ok_or_else(|| invalid("max() arg is an empty sequence"))?;
    let adds_up = python_eq(&json!(scored), whole_scored)
        && python_eq(&json!(flips), field(whole, "top1_disagreements")?)
        && python_eq(largest, field(whole, "max_abs_dlogprob")?)
        && agrees(
            get(whole, "mean_abs_dlogprob"),
            weighted / read(whole, "scored")?,
        );
    require(adds_up, || {
        format!("{where_}: the parts' scores do not add up to the stored totals")
    })
}

/// Flips (offset and served top-1, in their order) judged again by the
/// near-tie rule against the reference variant `scores`: each keeps
/// `fields` and gains its `offset`, `served`, `gap` and `decisive`
/// (`kinds.judged`).
pub fn judged<'a>(
    flips: impl IntoIterator<Item = (usize, &'a Value)>,
    scores: &Value,
    fields: &Map<String, Value>,
) -> Result<Vec<Value>, KindError> {
    let tops = limits::gap_top(scores)
        .ok_or_else(|| invalid("the reference scores hold no top-k rows"))?;
    let top1 = field(scores, "top1")?
        .as_array()
        .ok_or_else(|| invalid("the reference top1 is not a list"))?;
    let mut judged = Vec::new();
    for (offset, served) in flips {
        let (Some(reference), Some(top)) = (top1.get(offset), tops.get(offset)) else {
            return Err(invalid("list index out of range"));
        };
        let flip = limits::flip(reference, served, Some(top), NEAR_TIE_NATS)
            .ok_or_else(|| invalid("a flip that agrees with the reference top-1"))?;
        let mut entry = fields.clone();
        entry.insert("offset".into(), json!(offset));
        entry.insert("served".into(), served.clone());
        entry.insert("gap".into(), json!(flip.gap));
        entry.insert("decisive".into(), json!(flip.decisive));
        judged.push(Value::Object(entry));
    }
    Ok(judged)
}

/// The near-tie counts and rates of judged `flips` over `scored` positions
/// (`limits.top1_summary`) as receipt fields.
pub fn top1_fields(flips: &[Value], scored: u64) -> Map<String, Value> {
    let flips: Vec<Option<limits::Flip>> = flips
        .iter()
        .map(|flip| {
            Some(limits::Flip {
                gap: flip["gap"].as_f64(),
                decisive: flip["decisive"] == Value::Bool(true),
            })
        })
        .collect();
    match serde_json::to_value(limits::top1_summary(&flips, scored)) {
        Ok(Value::Object(map)) => map,
        _ => unreachable!("a summary serializes as an object"),
    }
}

/// The spread rule a category's or media case's verdict follows.
pub const SPREAD_RULE: &str = "variant-spread.v2";

/// The reference variant whose own flips the power test and the paired
/// fallback read.
pub const POWER_VARIANT: &str = "bf16";

/// variant-spread.v2's verdict of one category or media case: the
/// candidate's `measured` scores within `set_limits`, except that a top-1
/// metric whose limit cannot judge the set (`limits::power` against the
/// reference BF16 run's own rate there) also passes by the paired rule's
/// top-1 clause against that run. `candidate` and `reference` hold each
/// scored position's flip, in the same order; without a reference run each
/// metric is held to its limit alone. Returns the verdict and the fields
/// the set's row gains: `spread_rule`, `passed_by` (per top-1 metric
/// "limit", "paired_reference" or null), and with a reference run `power`
/// and, where the fallback was tried, `paired_reference`.
pub fn set_verdict(
    measured: &limits::Spread,
    set_limits: &limits::Spread,
    candidate: &[Option<limits::Flip>],
    reference: Option<&[Option<limits::Flip>]>,
) -> Result<(bool, Map<String, Value>), KindError> {
    type Pick = fn(&limits::Flip) -> bool;
    let metrics: [(&str, f64, f64, Pick); 3] = [
        (
            "top1_decisive",
            measured.top1_decisive,
            set_limits.top1_decisive,
            |flip| flip.decisive,
        ),
        (
            "top1_near_tie",
            measured.top1_near_tie,
            set_limits.top1_near_tie,
            |flip| !flip.decisive,
        ),
        (
            "top1_disagreement",
            measured.top1_disagreement,
            set_limits.top1_disagreement,
            |_| true,
        ),
    ];
    if reference.is_some_and(|reference| reference.len() != candidate.len()) {
        return Err(invalid(
            "the reference run scores other positions than the candidate",
        ));
    }
    let positions = candidate.len() as u64;
    let mut passed = measured.mean_abs_dlogprob <= set_limits.mean_abs_dlogprob;
    let (mut power, mut paired, mut passed_by) = (Map::new(), Map::new(), Map::new());
    for (name, value, limit, pick) in metrics {
        let flags = |flips: &[Option<limits::Flip>]| -> Vec<bool> {
            flips
                .iter()
                .map(|flip| flip.as_ref().is_some_and(pick))
                .collect()
        };
        let mut by = (value <= limit).then_some("limit");
        if let Some(reference) = reference.filter(|_| positions > 0) {
            let baseline = flags(reference);
            let rate = baseline.iter().filter(|flag| **flag).count() as f64 / positions as f64;
            let test = limits::power(rate, limit, positions, limits::CATEGORY_SIGMAS);
            power.insert(name.into(), json!(test));
            if by.is_none() && !test.has_power {
                let clause =
                    limits::paired_top1(&flags(candidate), &baseline, limits::CATEGORY_SIGMAS)
                        .map_err(|error| invalid(error.0))?;
                paired.insert(name.into(), json!(clause));
                if clause.passed {
                    by = Some("paired_reference");
                }
            }
        }
        passed &= by.is_some();
        passed_by.insert(name.into(), json!(by));
    }
    let mut fields = Map::new();
    fields.insert("spread_rule".into(), json!(SPREAD_RULE));
    fields.insert("passed_by".into(), Value::Object(passed_by));
    if reference.is_some() {
        fields.insert("power".into(), Value::Object(power));
        if !paired.is_empty() {
            paired.insert("variant".into(), json!(POWER_VARIANT));
            fields.insert("paired_reference".into(), Value::Object(paired));
        }
    }
    Ok((passed, fields))
}

/// The flip at each of `scored` positions from judged flips (`offset`,
/// `gap`, `decisive` each).
pub fn flips_by_offset(
    flips: &[Value],
    scored: usize,
) -> Result<Vec<Option<limits::Flip>>, KindError> {
    let mut out = vec![None; scored];
    for flip in flips {
        let offset = count(field(flip, "offset")?, "offset")? as usize;
        let slot = out
            .get_mut(offset)
            .ok_or_else(|| invalid("a flip beyond the scored positions"))?;
        *slot = Some(limits::Flip {
            gap: flip["gap"].as_f64(),
            decisive: flip["decisive"] == Value::Bool(true),
        });
    }
    Ok(out)
}

/// Measured scores or limits read as a spread (`limits.within` reads these
/// four fields).
pub fn spread_of(value: &Value) -> Result<limits::Spread, KindError> {
    let read = |key: &str| number(field(value, key)?, key);
    Ok(limits::Spread {
        mean_abs_dlogprob: read("mean_abs_dlogprob")?,
        top1_decisive: read("top1_decisive")?,
        top1_near_tie: read("top1_near_tie")?,
        top1_disagreement: read("top1_disagreement")?,
    })
}

/// A spread as the receipt fields Python writes.
pub fn spread_json(spread: &limits::Spread) -> Value {
    serde_json::to_value(spread).expect("a spread serializes")
}

/// A module constant as the check sets it (`checks.<id>.constants`), else
/// `default`.
pub fn constant(check: &Check, module: &str, name: &str, default: Value) -> Value {
    match &check.declared["constants"][module][name]["value"] {
        Value::Null => default,
        value => value.clone(),
    }
}

/// Refuses a check that sets a module constant this build does not apply
/// to a different value: the Python rules would read it, so the verdicts
/// would diverge.
pub fn applied_constants(check: &Check, applied: &[(&str, &str)]) -> Result<(), KindError> {
    let Some(modules) = check.declared["constants"].as_object() else {
        return Ok(());
    };
    for (module, entries) in modules {
        for name in entries.as_object().into_iter().flat_map(Map::keys) {
            if !applied.contains(&(module.as_str(), name.as_str())) {
                return Err(invalid(format!(
                    "checks.{}.constants.{module}.{name} is not applied by chopinfer eval",
                    check.id
                )));
            }
        }
    }
    Ok(())
}

/// The token IDs a reference scores as JSON values.
pub fn ids(value: &Value) -> Result<&[Value], KindError> {
    value
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| invalid("token IDs are not a list"))
}

// The exact references' score layouts (`tools/qual/exact_reference.py`).

const COLUMNS: [&str; 4] = ["logprob", "top1", "margin", "top5"];
const RUNNER_COLUMNS: [&str; 6] = [
    "logprob",
    "top1",
    "margin",
    "topk_ids",
    "topk_logits",
    "topk_logprobs",
];
const TOP: usize = 5;
/// Values recomputed from the same f64 row agree to rounding of the JSON
/// float text, far below any gated difference.
const TOLERANCE: f64 = 1e-9;

fn token_id(value: &Value) -> bool {
    natural(value)
}

fn close(a: f64, b: f64) -> bool {
    is_close(a, b, 1e-9, TOLERANCE)
}

/// Whether a row of finite values is non-increasing.
fn ordered(values: &[f64]) -> bool {
    values.windows(2).all(|pair| pair[0] >= pair[1])
}

/// Whether `ids` repeats a token (Python's `len(set(ids)) != len(ids)`).
fn repeats(ids: &[&Value]) -> bool {
    ids.iter()
        .enumerate()
        .any(|(index, id)| ids[..index].iter().any(|other| python_eq(other, id)))
}

/// The column `name` of `scores`, as long as `tokens`.
fn column<'a>(
    scores: &'a Value,
    name: &str,
    tokens: usize,
    where_: &str,
) -> Result<&'a [Value], String> {
    match scores.get(name).and_then(Value::as_array) {
        Some(values) if values.len() == tokens => Ok(values),
        values => Err(format!(
            "{where_}: {name} has {} entries for {tokens} positions",
            values.map_or_else(|| "no".to_owned(), |values| values.len().to_string())
        )),
    }
}

/// `exact_reference.check_scores`: a complete, consistent row of `scores`
/// for each of `tokens`, in either layout. The error is Python's message.
pub fn check_scores(
    scores: Option<&Value>,
    tokens: Option<&[Value]>,
    where_: &str,
) -> Result<(), String> {
    let (Some(scores), Some(tokens)) = (
        scores.filter(|scores| scores.is_object()),
        tokens.filter(|tokens| !tokens.is_empty()),
    ) else {
        return Err(format!("{where_}: no scores"));
    };
    let runner = scores.get("topk_ids").is_some();
    let names: &[&str] = if runner { &RUNNER_COLUMNS } else { &COLUMNS };
    let columns: Vec<&[Value]> = names
        .iter()
        .map(|name| column(scores, name, tokens.len(), where_))
        .collect::<Result<_, _>>()?;
    let finite_number =
        |value: &Value| finite(value).then(|| python_number(value).expect("finite"));
    for (index, token) in tokens.iter().enumerate() {
        let at = format!("{where_} position {index}");
        let logprob = finite_number(&columns[0][index]);
        let Some(logprob) = logprob.filter(|logprob| token_id(token) && *logprob <= TOLERANCE)
        else {
            return Err(format!("{at}: token or logprob is invalid"));
        };
        let (top1, margin) = (&columns[1][index], &columns[2][index]);
        let (ids, gaps, logprobs): (Vec<&Value>, Vec<f64>, Vec<f64>) = if runner {
            let rows: Vec<Option<&Vec<Value>>> = (3..6)
                .map(|column| {
                    columns[column][index]
                        .as_array()
                        .filter(|row| row.len() == TOP)
                })
                .collect();
            let (Some(ids), Some(logits), Some(logprobs)) = (rows[0], rows[1], rows[2]) else {
                return Err(format!(
                    "{at}: the top-k rows are not {TOP} token IDs with finite logits and logprobs"
                ));
            };
            let logits: Option<Vec<f64>> = logits.iter().map(finite_number).collect();
            let logprobs: Option<Vec<f64>> = logprobs.iter().map(finite_number).collect();
            let (Some(logits), Some(logprobs)) = (logits, logprobs) else {
                return Err(format!(
                    "{at}: the top-k rows are not {TOP} token IDs with finite logits and logprobs"
                ));
            };
            if !ids.iter().all(token_id) {
                return Err(format!(
                    "{at}: the top-k rows are not {TOP} token IDs with finite logits and logprobs"
                ));
            }
            let ids: Vec<&Value> = ids.iter().collect();
            if repeats(&ids) || !ordered(&logits) || !ordered(&logprobs) {
                return Err(format!(
                    "{at}: the top-k rows repeat a token or are out of order"
                ));
            }
            (ids, logits, logprobs)
        } else {
            let pairs = columns[3][index]
                .as_array()
                .filter(|row| row.len() == TOP)
                .and_then(|row| {
                    row.iter()
                        .map(|pair| {
                            let pair = pair.as_array().filter(|pair| pair.len() == 2)?;
                            (token_id(&pair[0]))
                                .then(|| finite_number(&pair[1]).map(|value| (&pair[0], value)))
                                .flatten()
                        })
                        .collect::<Option<Vec<_>>>()
                });
            let Some(pairs) = pairs else {
                return Err(format!("{at}: top5 is not {TOP} [token ID, logprob] pairs"));
            };
            let ids: Vec<&Value> = pairs.iter().map(|pair| pair.0).collect();
            let values: Vec<f64> = pairs.iter().map(|pair| pair.1).collect();
            if repeats(&ids) || !ordered(&values) {
                return Err(format!("{at}: top5 repeats a token or is out of order"));
            }
            (ids, values.clone(), values)
        };
        let (first, gap_name, lists) = if runner {
            ("top-k", "top-k logit gap", "the top-k logprobs")
        } else {
            ("top5", "top5 gap", "top5")
        };
        let Some(margin) = finite_number(margin).filter(|_| python_eq(top1, ids[0])) else {
            return Err(format!("{at}: top1 is not the first {first} token"));
        };
        if !close(margin, gaps[0] - gaps[1]) {
            return Err(format!("{at}: margin differs from the {gap_name}"));
        }
        let listed = ids.iter().position(|id| python_eq(id, token));
        if logprob > logprobs[0] + TOLERANCE
            || listed.is_some_and(|listed| !close(logprob, logprobs[listed]))
        {
            return Err(format!("{at}: logprob disagrees with {lists}"));
        }
    }
    Ok(())
}

/// Per position, the top [token ID, logprob] pairs of either layout, best
/// first (`exact_reference.top`).
pub fn top(scores: &Value) -> Result<Vec<Vec<(Value, f64)>>, KindError> {
    let pairs = |ids: &Value, values: &Value| -> Option<Vec<(Value, f64)>> {
        ids.as_array()?
            .iter()
            .zip(values.as_array()?)
            .map(|(id, value)| Some((id.clone(), python_number(value)?)))
            .collect()
    };
    let rows: Option<Vec<Vec<(Value, f64)>>> = if scores.get("topk_ids").is_some() {
        let ids = scores["topk_ids"].as_array();
        let logprobs = scores["topk_logprobs"].as_array();
        match (ids, logprobs) {
            (Some(ids), Some(logprobs)) if ids.len() == logprobs.len() => ids
                .iter()
                .zip(logprobs)
                .map(|(ids, logprobs)| pairs(ids, logprobs))
                .collect(),
            _ => None,
        }
    } else {
        scores["top5"].as_array().and_then(|rows| {
            rows.iter()
                .map(|row| {
                    row.as_array()?
                        .iter()
                        .map(|pair| Some((pair.get(0)?.clone(), python_number(pair.get(1)?)?)))
                        .collect()
                })
                .collect()
        })
    };
    rows.ok_or_else(|| invalid("the reference scores hold no top-k rows"))
}

/// Raises Insufficient unless the reference holds the gate variant, at
/// least one spread variant, and for every item every variant's complete
/// and consistent scores of its tokens (`kinds.check_reference_scores`).
pub fn check_reference_scores<'a>(
    items: &'a [Value],
    variants: &[Value],
    gate: &str,
    tokens: impl Fn(&'a Value) -> Option<&'a [Value]>,
    scores: impl Fn(&'a Value) -> Option<&'a Value>,
) -> Result<(), KindError> {
    require(
        variants.iter().any(|variant| variant == gate) && variants.len() >= 2,
        || {
            format!(
                "the reference variants {} need {gate} and a spread variant",
                repr::repr(&Value::Array(variants.to_vec()))
            )
        },
    )?;
    for item in items {
        let label = match (get(item, "name"), get(item, "id")) {
            (Some(name), _) if crate::json::truthy(name) => repr::str(name),
            (_, id) => repr::str_of(id),
        };
        for variant in variants {
            let variant = repr::str(variant);
            let holder = scores(item);
            let stated = holder.and_then(|holder| get(holder, &variant));
            check_scores(
                stated,
                tokens(item),
                &format!("reference {label} {variant}"),
            )
            .map_err(KindError::Insufficient)?;
        }
    }
    Ok(())
}

/// The served greedy answer walked along the reference answer `ids` with
/// the gate variant's top-k rows `top`: every walked token must be in the
/// top 5 with its logprob at most `limit` below the top-1's (`kinds.walk`).
/// Returns `passed` and `reference_margin`.
pub fn walk(
    ids: &[Value],
    top: &[Vec<(Value, f64)>],
    limit: f64,
    greedy: &Value,
) -> Result<Map<String, Value>, KindError> {
    let result = |passed: bool, margin: Value| {
        let mut map = Map::new();
        map.insert("passed".into(), Value::Bool(passed));
        map.insert("reference_margin".into(), margin);
        map
    };
    if get(greedy, "error").is_some()
        || get(greedy, "reason") == Some(&json!("missing generated token IDs"))
    {
        return Ok(result(
            false,
            get(greedy, "reference_margin")
                .cloned()
                .unwrap_or(Value::Null),
        ));
    }
    let divergence = get(greedy, "divergence").filter(|value| !value.is_null());
    let served = get(greedy, "served_token_id").filter(|value| !value.is_null());
    let tokens: Vec<&Value> = match divergence {
        None => ids.iter().collect(),
        Some(divergence) => {
            let at = count(divergence, "divergence")? as usize;
            ids[..at.min(ids.len())].iter().chain(served).collect()
        }
    };
    let mut margin = 0.0f64;
    for (offset, token) in tokens.into_iter().enumerate() {
        let row = top
            .get(offset)
            .ok_or_else(|| invalid("list index out of range"))?;
        // A dict of the first five pairs: a repeated token keeps its last.
        let ranked = row[..row.len().min(5)]
            .iter()
            .rev()
            .find(|(id, _)| python_eq(id, token));
        let Some((_, logprob)) = ranked else {
            return Ok(result(false, json!(margin)));
        };
        let first = row
            .first()
            .ok_or_else(|| invalid("list index out of range"))?;
        let gap = first.1 - logprob;
        if gap > margin {
            margin = gap;
        }
        if !limits::near_tie(Some(gap), limit) {
            return Ok(result(false, json!(margin)));
        }
    }
    Ok(result(
        divergence.is_none() || served.is_some(),
        json!(margin),
    ))
}
